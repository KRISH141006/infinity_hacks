import math
import os
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.20, use_yolo=True):
        self.ear_threshold = ear_threshold
        self.use_yolo = use_yolo

        # Indices for EAR calculation
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Mouth indices
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291

        # Temporal trackers
        self.eye_closed_frames = 0
        self.ear_history = []
        self.window_size = 150  # 6 seconds window at 25 FPS for PERCLOS

        # Blink behavior tracking
        self.eye_was_closed = False
        self.blink_count = 0
        
        # Yawning tracking
        self.yawn_frames = 0

        # YOLO Drowsiness Classification Model cache
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
                print(f"Failed to load YOLO drowsiness model: {e}. Falling back to rule-based.")
                self.use_yolo = False

    def distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def compute_ear(self, landmarks, eye_indices):
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)
        horizontal = self.distance(p1, p4)

        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def process_frame(self, landmarks, frame, fps, frame_number=0):
        if not landmarks:
            # Maintain previous states or return default values if face is missing
            return {
                "ear": 0.25,
                "mar": 0.15,
                "perclos": sum(self.ear_history) / len(self.ear_history) if self.ear_history else 0.0,
                "drowsiness_score": 0,
                "eye_closed_duration": 0.0,
                "blink_count": self.blink_count,
                "yawn_detected": False,
                "yolo_drowsy_prob": self.last_yolo_drowsy_prob
            }

        # 1. Calculate EAR
        left_ear = self.compute_ear(landmarks, self.LEFT_EYE)
        right_ear = self.compute_ear(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        # Update eye closed duration
        is_closed = ear < self.ear_threshold
        if is_closed:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = max(0, self.eye_closed_frames - 2)

        eye_closed_duration = self.eye_closed_frames / fps

        # Track Blink Events (Transition closed -> open is 1 blink)
        if is_closed and not self.eye_was_closed:
            self.eye_was_closed = True
        elif not is_closed and self.eye_was_closed:
            self.blink_count += 1
            self.eye_was_closed = False

        # Keep history for PERCLOS
        self.ear_history.append(1.0 if is_closed else 0.0)
        if len(self.ear_history) > self.window_size:
            self.ear_history.pop(0)

        perclos = sum(self.ear_history) / len(self.ear_history) if self.ear_history else 0.0

        # 2. Calculate Mouth Aspect Ratio (MAR) and Yawn detection
        mouth_vertical = self.distance(landmarks[self.MOUTH_TOP], landmarks[self.MOUTH_BOTTOM])
        mouth_horizontal = self.distance(landmarks[self.MOUTH_LEFT], landmarks[self.MOUTH_RIGHT])
        mar = mouth_vertical / mouth_horizontal if mouth_horizontal != 0 else 0.0

        # Persistent yawn tracking (MAR > 0.5 for >= 1.0 second)
        is_yawning = False
        if mar > 0.45:
            self.yawn_frames += 1
        else:
            self.yawn_frames = max(0, self.yawn_frames - 2)
            
        if self.yawn_frames >= int(fps * 1.0):
            is_yawning = True

        # 3. YOLO Drowsiness Prediction (every 5th frame)
        if self.use_yolo and self.yolo_model is not None and frame is not None and frame_number % 5 == 0:
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

        # 4. Compute Drowsiness Score (0-100)
        # Eye closure duration (max out at 2.5 seconds)
        duration_factor = min(1.0, eye_closed_duration / 2.5)
        # Yawning MAR factor
        mar_factor = 1.0 if is_yawning else min(1.0, max(0.0, (mar - 0.25) / 0.25))

        # Base rule scoring
        rule_score = (duration_factor * 50.0) + (perclos * 35.0) + (mar_factor * 15.0)
        
        if self.use_yolo:
            drowsiness_score = int(rule_score * 0.6 + self.last_yolo_drowsy_prob * 100.0 * 0.4)
        else:
            drowsiness_score = int(rule_score)

        drowsiness_score = min(100, max(0, drowsiness_score))

        return {
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "drowsiness_score": drowsiness_score,
            "eye_closed_duration": eye_closed_duration,
            "blink_count": self.blink_count,
            "yawn_detected": is_yawning,
            "yolo_drowsy_prob": self.last_yolo_drowsy_prob
        }
