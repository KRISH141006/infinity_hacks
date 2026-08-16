# eye_engine.py - Independent Eye Analysis, Adaptive EAR, Velocity & Blink State Machine
import math
import numpy as np
from collections import deque

class EyeEngine:
    def __init__(
        self,
        default_baseline=0.28,
        closure_ratio=0.72,
        min_ear_threshold=0.15,
        max_ear_threshold=0.35
    ):
        self.closure_ratio = closure_ratio
        self.min_ear_threshold = min_ear_threshold
        self.max_ear_threshold = max_ear_threshold
        
        # 6-point MediaPipe eye landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Adaptive baseline tracking
        self.ear_baseline = default_baseline
        self.ear_threshold = max(min_ear_threshold, min(max_ear_threshold, default_baseline * closure_ratio))
        self.stable_open_buffer = deque(maxlen=200)
        self.is_calibrated = False

        # Independent eye metrics
        self.last_valid_left_ear = default_baseline
        self.last_valid_right_ear = default_baseline
        self.last_valid_ear = default_baseline

        # Temporal velocities
        self.last_ear = default_baseline
        self.ear_velocity = 0.0  # dEAR / dt (negative = closing, positive = opening)

        # Eye State Machine: OPEN, CLOSING, CLOSED, REOPENING
        self.eye_state = "OPEN"
        self.eye_closed_frames = 0
        self.eye_closed_duration = 0.0

        # Blink event detector
        # OPEN -> CLOSING -> CLOSED -> REOPENING -> OPEN
        self.blink_phase = "OPEN"
        self.current_blink_duration = 0.0
        self.last_blink_time = 0.0
        self.total_blinks = 0
        self.normal_blinks = 0
        self.long_blinks = 0
        self.last_completed_blink_duration = 0.0
        self.last_inter_blink_interval = 0.0

        # Rolling history for blink rate (last 60 seconds)
        self.blink_timestamps = deque(maxlen=100)

    def dist_3d(self, p1, p2):
        z1 = getattr(p1, 'z', 0.0)
        z2 = getattr(p2, 'z', 0.0)
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (z1 - z2)**2)

    def compute_single_ear(self, landmarks, idxs):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idxs]
        v1 = self.dist_3d(p2, p6)
        v2 = self.dist_3d(p3, p5)
        h = self.dist_3d(p1, p4)
        if h <= 1e-6:
            return 0.25, 0.0, 0.0
        ear = (v1 + v2) / (2.0 * h)
        eyelid_dist = (v1 + v2) / 2.0
        return ear, h, eyelid_dist

    def update_adaptive_baseline(self, ear, is_stable_upright, mar):
        """
        Updates personalized EAR baseline during verified upright, non-yawning, non-blinking frames.
        """
        if is_stable_upright and mar < 0.40 and ear > self.min_ear_threshold:
            self.stable_open_buffer.append(ear)
            if len(self.stable_open_buffer) >= 20:
                # Use 60th percentile of open eyes to prevent downward bias from sluggish eyelids
                self.ear_baseline = float(np.percentile(self.stable_open_buffer, 60))
                self.ear_threshold = max(
                    self.min_ear_threshold,
                    min(self.max_ear_threshold, self.ear_baseline * self.closure_ratio)
                )
                self.is_calibrated = True

    def process(
        self,
        landmarks,
        dt,
        current_time,
        is_stable_upright=True,
        mar=0.15,
        tracking_reliable=True
    ):
        if not landmarks or not tracking_reliable:
            # Maintain previous closure state without fake reset
            if self.eye_state in ["CLOSED", "CLOSING"]:
                self.eye_closed_frames += 1
                self.eye_closed_duration += dt
            else:
                self.eye_closed_duration = max(0.0, self.eye_closed_duration - dt)

            norm_ear = self.last_valid_ear / max(0.05, self.ear_baseline)
            return {
                "ear_left": round(self.last_valid_left_ear, 3),
                "ear_right": round(self.last_valid_right_ear, 3),
                "ear": round(self.last_valid_ear, 3),
                "normalized_ear": round(norm_ear, 3),
                "ear_baseline": round(self.ear_baseline, 3),
                "ear_threshold": round(self.ear_threshold, 3),
                "eye_state": self.eye_state,
                "is_eye_closed": self.eye_state == "CLOSED",
                "eye_closed_duration": round(self.eye_closed_duration, 2),
                "ear_velocity": 0.0,
                "blink_count": self.total_blinks,
                "last_blink_duration": round(self.last_completed_blink_duration, 3),
                "blink_rate_per_min": len(self.blink_timestamps),
                "inter_blink_interval": round(self.last_inter_blink_interval, 2),
                "eye_agreement": 1.0
            }

        # Compute independent eye EARs
        left_ear, left_w, left_h = self.compute_single_ear(landmarks, self.LEFT_EYE)
        right_ear, right_w, right_h = self.compute_single_ear(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        
        self.last_valid_left_ear = left_ear
        self.last_valid_right_ear = right_ear
        self.last_valid_ear = ear

        # Calculate EAR velocity (dEAR/dt)
        if dt > 0:
            raw_vel = (ear - self.last_ear) / dt
            self.ear_velocity = float(0.7 * self.ear_velocity + 0.3 * raw_vel)
        self.last_ear = ear

        # Update dynamic baseline
        self.update_adaptive_baseline(ear, is_stable_upright, mar)
        normalized_ear = ear / max(0.05, self.ear_baseline)

        # Eye agreement (0.0 to 1.0)
        ear_diff = abs(left_ear - right_ear)
        eye_agreement = max(0.0, min(1.0, 1.0 - (ear_diff / max(0.1, ear))))

        # -------------------------------------------------------------
        # BLINK EVENT & EYE CLOSURE STATE MACHINE
        # -------------------------------------------------------------
        is_closed = (ear < self.ear_threshold) or (normalized_ear < self.closure_ratio)

        if is_closed:
            self.eye_closed_frames += 1
            self.eye_closed_duration += dt
            self.current_blink_duration += dt

            if self.blink_phase == "OPEN":
                self.blink_phase = "CLOSING"
            elif self.blink_phase == "CLOSING" and self.eye_closed_duration >= 0.08:
                self.blink_phase = "CLOSED"

            self.eye_state = "CLOSED"
        else:
            # Eyes are open
            if self.blink_phase in ["CLOSING", "CLOSED"]:
                # Blink completion event
                blink_dur = self.current_blink_duration
                self.last_completed_blink_duration = blink_dur
                self.total_blinks += 1

                if self.last_blink_time > 0:
                    self.last_inter_blink_interval = current_time - self.last_blink_time
                self.last_blink_time = current_time
                self.blink_timestamps.append(current_time)

                if 0.08 <= blink_dur < 0.40:
                    self.normal_blinks += 1
                elif 0.40 <= blink_dur < 0.85:
                    self.long_blinks += 1

                self.current_blink_duration = 0.0
                self.blink_phase = "OPEN"

            self.eye_closed_frames = max(0, self.eye_closed_frames - 2)
            self.eye_closed_duration = max(0.0, self.eye_closed_duration - 1.5 * dt)
            self.eye_state = "OPEN"

        # Prune old blink timestamps (>60s)
        while self.blink_timestamps and (current_time - self.blink_timestamps[0]) > 60.0:
            self.blink_timestamps.popleft()

        return {
            "ear_left": round(left_ear, 3),
            "ear_right": round(right_ear, 3),
            "ear": round(ear, 3),
            "normalized_ear": round(normalized_ear, 3),
            "ear_baseline": round(self.ear_baseline, 3),
            "ear_threshold": round(self.ear_threshold, 3),
            "eye_state": self.eye_state,
            "is_eye_closed": is_closed,
            "eye_closed_duration": round(self.eye_closed_duration, 2),
            "ear_velocity": round(self.ear_velocity, 2),
            "blink_count": self.total_blinks,
            "normal_blink_count": self.normal_blinks,
            "long_blink_count": self.long_blinks,
            "last_blink_duration": round(self.last_completed_blink_duration, 3),
            "blink_rate_per_min": len(self.blink_timestamps),
            "inter_blink_interval": round(self.last_inter_blink_interval, 2),
            "eye_agreement": round(eye_agreement, 2)
        }
