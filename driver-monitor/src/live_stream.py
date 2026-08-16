# live_stream.py - Real-Time Live Webcam DMS Engine with Async Scheduling & Continuous Decision Inference
import cv2
import time
import os
import numpy as np
from src.face_landmarks import FaceLandmarkerHelper
from src.dms import (
    FaceTracker,
    EyeEngine,
    GazeEngine,
    HeadPoseEngine,
    YawnEngine,
    FatigueEngine,
    DistractionEngine,
    DrowsinessEngine,
    PhoneEngine,
    FusionEngine
)
from src.phone_detection import PhoneDetector

class LiveDMSStream:
    def __init__(self, model_dir="models", camera_index=0):
        self.camera_index = camera_index
        face_task_path = os.path.join(model_dir, "face_landmarker.task")
        
        self.landmarker = FaceLandmarkerHelper(face_task_path)
        self.face_tracker = FaceTracker(max_lost_grace_sec=0.35)
        self.eye_engine = EyeEngine()
        self.gaze_engine = GazeEngine()
        self.head_pose_engine = HeadPoseEngine()
        self.yawn_engine = YawnEngine()
        self.fatigue_engine = FatigueEngine(fps=30.0)
        self.distraction_engine = DistractionEngine()
        self.drowsiness_engine = DrowsinessEngine(use_yolo=False) # Fast CPU live mode
        self.phone_engine = PhoneEngine()
        self.fusion_engine = FusionEngine()
        self.phone_detector = PhoneDetector(check_interval=15)

        self.last_time = time.time()
        self.frame_count = 0
        self.last_phone_detected = False
        self.last_phone_conf = 0.0

    def process_frame(self, frame):
        now = time.time()
        dt = max(1e-3, min(0.1, now - self.last_time))
        self.last_time = now
        self.frame_count += 1
        h, w = frame.shape[:2]

        # Use monotonic millisecond timestamp
        timestamp_ms = int(now * 1000)

        # 1. MediaPipe Landmarks (Fast CPU)
        landmarks = self.landmarker.process_frame(frame, timestamp_ms)

        # 2. Face Tracking
        face_data = self.face_tracker.update(landmarks, (h, w), dt)
        tracking_reliable = face_data["is_confident"]

        # 3. Heavy Models (Phone detector every 15 frames)
        if self.frame_count % 15 == 0:
            det, conf = self.phone_detector.process_frame(frame, self.frame_count)
            self.last_phone_detected = det
            self.last_phone_conf = conf

        # 4. Yawn Engine
        yawn_data = self.yawn_engine.process(landmarks, dt, now)

        # 5. Eye Engine
        eye_data = self.eye_engine.process(
            landmarks=landmarks,
            dt=dt,
            current_time=now,
            is_stable_upright=True,
            mar=yawn_data["mar"],
            tracking_reliable=tracking_reliable
        )

        # 6. Head Pose & Nodding Engine
        head_pose_data = self.head_pose_engine.process(
            landmarks=landmarks,
            frame_shape=(h, w),
            dt=dt,
            is_eye_open=not eye_data["is_eye_closed"],
            is_eye_closed=eye_data["is_eye_closed"]
        )

        # 7. Gaze Engine
        gaze_data = self.gaze_engine.process(
            landmarks=landmarks,
            is_eye_open=not eye_data["is_eye_closed"],
            is_stable_upright=head_pose_data["is_head_forward"],
            tracking_reliable=tracking_reliable
        )

        # 8. Fatigue Engine
        fatigue_data = self.fatigue_engine.process(
            is_eye_closed=eye_data["is_eye_closed"],
            yawn_data=yawn_data,
            eye_data=eye_data,
            dt=dt
        )

        # 9. Drowsiness Engine
        drowsiness_data = self.drowsiness_engine.process(
            eye_data=eye_data,
            head_pose_data=head_pose_data,
            fatigue_data=fatigue_data,
            yawn_data=yawn_data,
            yolo_drowsy_prob=0.0,
            dt=dt
        )

        # 10. Phone Engine
        phone_data = self.phone_engine.process(
            raw_detected=self.last_phone_detected,
            raw_conf=self.last_phone_conf,
            dt=dt
        )

        # 11. Distraction Engine
        distraction_data = self.distraction_engine.process(
            gaze_data=gaze_data,
            head_pose_data=head_pose_data,
            eye_data=eye_data,
            drowsiness_data=drowsiness_data,
            face_track_data=face_data,
            dt=dt
        )

        # 12. State & Attention Fusion
        frame_record, _ = self.fusion_engine.process(
            timestamp_sec=round(self.frame_count * dt, 2),
            face_data=face_data,
            eye_data=eye_data,
            gaze_data=gaze_data,
            head_pose_data=head_pose_data,
            yawn_data=yawn_data,
            fatigue_data=fatigue_data,
            distraction_data=distraction_data,
            drowsiness_data=drowsiness_data,
            phone_data=phone_data
        )

        return frame_record, landmarks
