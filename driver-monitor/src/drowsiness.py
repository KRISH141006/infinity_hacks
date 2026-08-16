import math
import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

class DrowsinessDetector:
    def __init__(
        self,
        ear_closure_ratio=0.70,
        min_ear_threshold=0.16,
        default_baseline_ear=0.28,
        use_yolo=True
    ):
        self.ear_closure_ratio = ear_closure_ratio
        self.min_ear_threshold = min_ear_threshold
        self.use_yolo = use_yolo

        # MediaPipe Eye indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Mouth indices for MAR
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291

        # 1. Adaptive EAR Calibration
        self.ear_baseline = default_baseline_ear
        self.ear_closed_threshold = max(self.min_ear_threshold, self.ear_closure_ratio * self.ear_baseline)
        self.stable_open_ears = []
        self.calibration_limit = 45  # frames of stable open eyes
        self.is_calibrated = False

        # 2. Short and Long Temporal Tracking Windows
        self.eye_closed_frames = 0
        self.eye_closed_duration = 0.0
        self.last_valid_ear = default_baseline_ear
        self.last_valid_mar = 0.18

        # Long window for PERCLOS (e.g. 45 seconds buffer at 30 fps = ~1350 frames)
        self.perclos_window_size = 900  # ~30-36s window
        self.closure_history = deque(maxlen=self.perclos_window_size)

        # Blink tracking
        self.current_blink_frames = 0
        self.eye_was_closed = False
        self.total_blinks = 0
        self.slow_blinks = 0

        # Yawn tracking
        self.yawn_frames = 0
        self.yawn_duration = 0.0

        # Head pitch tracking for nodding-off
        self.last_pitch = None
        self.last_pitch_time = None
        self.pitch_velocity = 0.0

        # Landmark loss persistence tracking (<300 ms grace window)
        self.lost_frames = 0
        self.max_lost_grace_frames = 8  # ~260-320 ms at 25-30 fps

        # State Machine: ALERT, FATIGUED, DROWSY_CANDIDATE, DROWSY, DROWSY_NODDING, RECOVERING
        self.state = "ALERT"
        self.drowsiness_score = 0
        self.recovery_frames = 0

        # YOLO Drowsiness Classification Model
        self.yolo_model = None
        self.last_yolo_drowsy_prob = 0.0

        if self.use_yolo:
            try:
                print("Checking/Downloading YOLO Drowsiness Classification model...")
                model_dir = "models/drowsiness_model"
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "best.pt")
                if not os.path.exists(model_path):
                    model_path = hf_hub_download(
                        repo_id="mosesb/drowsiness-detection-yolo-cls",
                        filename="best.pt",
                        local_dir=model_dir
                    )
                self.yolo_model = YOLO(model_path)
                print("YOLO Drowsiness model loaded successfully!")
            except Exception as e:
                print(f"Failed to load YOLO drowsiness model: {e}. Falling back to rule-based engine.")
                self.use_yolo = False

    def distance_3d(self, p1, p2):
        # 3D landmark Euclidean distance
        z1 = getattr(p1, 'z', 0.0)
        z2 = getattr(p2, 'z', 0.0)
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (z1 - z2)**2)

    def compute_ear(self, landmarks, eye_indices):
        p1 = landmarks[eye_indices[0]] # outer
        p2 = landmarks[eye_indices[1]] # top 1
        p3 = landmarks[eye_indices[2]] # top 2
        p4 = landmarks[eye_indices[3]] # inner
        p5 = landmarks[eye_indices[4]] # bot 2
        p6 = landmarks[eye_indices[5]] # bot 1

        vertical_1 = self.distance_3d(p2, p6)
        vertical_2 = self.distance_3d(p3, p5)
        horizontal = self.distance_3d(p1, p4)

        if horizontal <= 1e-6:
            return self.last_valid_ear
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def update_adaptive_ear(self, current_ear, is_head_forward, mar):
        """
        Calibrates EAR baseline dynamically using stable open-eye frames.
        Accepts any reasonable non-zero open eye sample when head is forward.
        """
        if is_head_forward and mar < 0.40 and current_ear > 0.14:
            self.stable_open_ears.append(current_ear)
            if len(self.stable_open_ears) > 150:
                self.stable_open_ears.pop(0)

            if len(self.stable_open_ears) >= 15:
                # Use 75th percentile of open eyes to avoid dragging baseline down by squinting
                self.ear_baseline = float(np.percentile(self.stable_open_ears, 60))
                self.ear_closed_threshold = max(
                    self.min_ear_threshold,
                    self.ear_closure_ratio * self.ear_baseline
                )
                self.is_calibrated = True

    def process_frame(
        self,
        landmarks,
        frame,
        fps,
        frame_number=0,
        relative_pitch=0.0,
        head_forward=True
    ):
        dt = 1.0 / max(1.0, float(fps))
        current_time = frame_number * dt

        # -------------------------------------------------------------
        # 1. LANDMARK LOSS HANDLING
        # -------------------------------------------------------------
        if not landmarks:
            self.lost_frames += 1
            is_tracking_lost = self.lost_frames > self.max_lost_grace_frames

            # If lost while drowsy/nodding, retain and escalate drowsiness rather than resetting
            if self.state in ["DROWSY_CANDIDATE", "DROWSY", "DROWSY_NODDING"] or self.drowsiness_score >= 60:
                self.eye_closed_frames += 1
                self.eye_closed_duration = self.eye_closed_frames * dt
                self.closure_history.append(1.0)
                if self.eye_closed_duration >= 1.5:
                    self.state = "DROWSY_NODDING" if (self.state == "DROWSY_NODDING" or relative_pitch < -10.0) else "DROWSY"
                    self.drowsiness_score = max(self.drowsiness_score, 90)
            else:
                self.closure_history.append(1.0 if self.eye_closed_frames > 0 else 0.0)
                if is_tracking_lost:
                    self.state = "FACE_TRACKING_LOST"

            perclos = sum(self.closure_history) / max(1, len(self.closure_history))
            return {
                "ear": self.last_valid_ear,
                "ear_baseline": round(self.ear_baseline, 3),
                "ear_threshold": round(self.ear_closed_threshold, 3),
                "mar": self.last_valid_mar,
                "perclos": round(perclos, 3),
                "drowsiness_score": self.drowsiness_score,
                "drowsiness_state": self.state,
                "eye_closed_duration": round(self.eye_closed_duration, 2),
                "is_eye_closed": self.eye_closed_duration > 0.25 or self.state in ["DROWSY", "DROWSY_NODDING"],
                "is_nodding": self.state == "DROWSY_NODDING",
                "head_pitch_velocity": round(self.pitch_velocity, 1),
                "blink_count": self.total_blinks,
                "slow_blink_count": self.slow_blinks,
                "yawn_detected": self.yawn_duration >= 1.0,
                "yolo_drowsy_prob": self.last_yolo_drowsy_prob,
                "tracking_lost": is_tracking_lost
            }


        # Landmarks are present: reset lost frame counter
        self.lost_frames = 0

        # -------------------------------------------------------------
        # 2. EXTRACT EAR & MAR
        # -------------------------------------------------------------
        left_ear = self.compute_ear(landmarks, self.LEFT_EYE)
        right_ear = self.compute_ear(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        self.last_valid_ear = ear

        # MAR calculation
        mouth_vert = self.distance_3d(landmarks[self.MOUTH_TOP], landmarks[self.MOUTH_BOTTOM])
        mouth_horiz = self.distance_3d(landmarks[self.MOUTH_LEFT], landmarks[self.MOUTH_RIGHT])
        mar = mouth_vert / mouth_horiz if mouth_horiz > 1e-6 else 0.15
        self.last_valid_mar = mar

        # -------------------------------------------------------------
        # 3. ADAPTIVE CALIBRATION
        # -------------------------------------------------------------
        self.update_adaptive_ear(ear, head_forward, mar)

        # -------------------------------------------------------------
        # 4. EYE CLOSURE & BLINK TEMPORAL TRACKING
        # -------------------------------------------------------------
        is_closed = ear < self.ear_closed_threshold

        if is_closed:
            self.eye_closed_frames += 1
            self.current_blink_frames += 1
            self.closure_history.append(1.0)
            self.eye_was_closed = True
        else:
            self.closure_history.append(0.0)
            if self.eye_was_closed:
                # Blink ended
                blink_dur_sec = self.current_blink_frames * dt
                self.total_blinks += 1
                if blink_dur_sec >= 0.35:
                    self.slow_blinks += 1
                self.current_blink_frames = 0
                self.eye_was_closed = False

            # Smooth decay on opening
            self.eye_closed_frames = max(0, self.eye_closed_frames - 3)

        self.eye_closed_duration = self.eye_closed_frames * dt
        perclos = sum(self.closure_history) / max(1, len(self.closure_history))

        # -------------------------------------------------------------
        # 5. YAWNING & MOUTH DURATION
        # -------------------------------------------------------------
        if mar > 0.45:
            self.yawn_frames += 1
        else:
            self.yawn_frames = max(0, self.yawn_frames - 2)
        self.yawn_duration = self.yawn_frames * dt
        is_yawning = self.yawn_duration >= 1.0

        # -------------------------------------------------------------
        # 6. HEAD PITCH VELOCITY & NODDING DETECTION
        # -------------------------------------------------------------
        if self.last_pitch is not None and self.last_pitch_time is not None:
            time_diff = current_time - self.last_pitch_time
            if time_diff > 0:
                raw_velocity = (relative_pitch - self.last_pitch) / time_diff
                # Smooth velocity
                self.pitch_velocity = float(0.7 * self.pitch_velocity + 0.3 * raw_velocity)
        self.last_pitch = relative_pitch
        self.last_pitch_time = current_time

        # Nodding occurs when eyes are closed or low EAR AND head drops down
        is_nodding_down = (
            (is_closed or ear < self.ear_closed_threshold * 1.15) and
            (relative_pitch < -12.0 or self.pitch_velocity < -18.0)
        )

        # -------------------------------------------------------------
        # 7. YOLO DROWSINESS INFERENCE (PERIODIC)
        # -------------------------------------------------------------
        if self.use_yolo and self.yolo_model is not None and frame is not None and frame_number % 5 == 0:
            try:
                results = self.yolo_model.predict(source=frame, verbose=False)
                if results and results[0].probs:
                    probs = results[0].probs
                    drowsy_idx = None
                    for idx, name in self.yolo_model.names.items():
                        if 'non' not in name.lower() and 'drowsy' in name.lower():
                            drowsy_idx = idx
                            break
                    if drowsy_idx is not None:
                        self.last_yolo_drowsy_prob = float(probs.data[drowsy_idx])
                    else:
                        self.last_yolo_drowsy_prob = float(probs.top1conf) if probs.top1 == 0 else 1.0 - float(probs.top1conf)
            except Exception:
                pass

        # -------------------------------------------------------------
        # 8. TEMPORAL STATE MACHINE & SCORING
        # -------------------------------------------------------------
        # Short window priority:
        # - normal blink (<0.35s): score remains low
        # - prolonged closure >= 1.5s: DROWSY / CRITICAL
        # - prolonged closure + head drop: DROWSY_NODDING (critical 95-100)
        # - PERCLOS long window: FATIGUED baseline

        calculated_score = 0

        # Base PERCLOS fatigue component (up to 30 points)
        fatigue_points = min(30.0, perclos * 100.0 * 1.5)
        calculated_score += fatigue_points

        # Yawn component (up to 20 points)
        if is_yawning:
            calculated_score += 20.0

        # YOLO confidence agreement (up to 15 points)
        if self.use_yolo and self.last_yolo_drowsy_prob > 0.60:
            calculated_score += (self.last_yolo_drowsy_prob - 0.60) * 37.5

        # Eye closure duration rules (Short Window Dominance)
        if self.eye_closed_duration >= 1.5:
            if is_nodding_down or relative_pitch < -12.0:
                self.state = "DROWSY_NODDING"
                calculated_score = max(calculated_score, 95.0 + min(5.0, (self.eye_closed_duration - 1.5) * 5))
            else:
                self.state = "DROWSY"
                calculated_score = max(calculated_score, 85.0 + min(15.0, (self.eye_closed_duration - 1.5) * 10))
        elif self.eye_closed_duration >= 0.8:
            self.state = "DROWSY_CANDIDATE"
            closure_ramp = (self.eye_closed_duration - 0.8) / 0.7  # 0.0 to 1.0
            calculated_score = max(calculated_score, 55.0 + closure_ramp * 25.0)
        elif self.eye_closed_duration >= 0.35:
            # Long blink sluggishness
            calculated_score = max(calculated_score, 30.0 + (self.eye_closed_duration - 0.35) * 40.0)
            if self.state not in ["DROWSY", "DROWSY_NODDING"]:
                self.state = "FATIGUED" if perclos > 0.15 else "ALERT"
        else:
            # Normal state or recovering
            if self.state in ["DROWSY", "DROWSY_NODDING", "DROWSY_CANDIDATE"]:
                self.state = "RECOVERING"
                self.recovery_frames = int(fps * 1.5)  # 1.5s recovery decay
            elif self.state == "RECOVERING":
                self.recovery_frames -= 1
                if self.recovery_frames <= 0:
                    self.state = "FATIGUED" if perclos > 0.15 else "ALERT"
            else:
                self.state = "FATIGUED" if perclos > 0.15 else "ALERT"

        # Smooth score recovery (prevent instantaneous drop)
        target_score = int(min(100, max(0, calculated_score)))
        if self.state == "RECOVERING":
            self.drowsiness_score = max(target_score, int(self.drowsiness_score * 0.92))
        else:
            self.drowsiness_score = target_score

        return {
            "ear": round(ear, 3),
            "ear_baseline": round(self.ear_baseline, 3),
            "ear_threshold": round(self.ear_closed_threshold, 3),
            "mar": round(mar, 3),
            "perclos": round(perclos, 3),
            "drowsiness_score": self.drowsiness_score,
            "drowsiness_state": self.state,
            "eye_closed_duration": round(self.eye_closed_duration, 2),
            "is_eye_closed": is_closed,
            "is_nodding": is_nodding_down,
            "head_pitch_velocity": round(self.pitch_velocity, 1),
            "blink_count": self.total_blinks,
            "slow_blink_count": self.slow_blinks,
            "yawn_detected": is_yawning,
            "yolo_drowsy_prob": round(self.last_yolo_drowsy_prob, 2),
            "tracking_lost": False
        }
