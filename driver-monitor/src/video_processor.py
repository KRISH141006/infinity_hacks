import cv2
import json
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
from src.blackbox import BlackBoxRecorder
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

class VideoProcessor:
    def __init__(
        self,
        model_dir="models",
        input_video="input/driver.mp4",
        output_video="output/processed_driver.mp4",
        telemetry_path="output/telemetry.json",
        use_yolo_drowsy=True,
        use_yolo_phone=True
    ):
        self.input_video = input_video
        self.output_video = output_video
        self.telemetry_path = telemetry_path
        self.model_dir = model_dir

        # Create output directories if needed
        os.makedirs(os.path.dirname(self.output_video) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.telemetry_path) or ".", exist_ok=True)

        face_task_path = os.path.join(model_dir, "face_landmarker.task")

        # 1. Initialize MediaPipe Face Landmarker
        self.landmarker = FaceLandmarkerHelper(face_task_path)

        # 2. Initialize Lightweight DMS Engines (Per-Frame)
        self.face_tracker = FaceTracker(max_lost_grace_sec=0.35)
        self.eye_engine = EyeEngine()
        self.gaze_engine = GazeEngine()
        self.head_pose_engine = HeadPoseEngine()
        self.yawn_engine = YawnEngine()
        self.fatigue_engine = FatigueEngine(fps=30.0)
        self.distraction_engine = DistractionEngine()
        self.drowsiness_engine = DrowsinessEngine(use_yolo=use_yolo_drowsy)
        self.phone_engine = PhoneEngine()
        self.fusion_engine = FusionEngine()
        self.blackbox = BlackBoxRecorder(model_dir=model_dir, output_dir=os.path.dirname(telemetry_path))

        # 3. Initialize Heavy Models (Scheduled Inference)
        self.use_yolo_drowsy = use_yolo_drowsy
        self.yolo_drowsy_model = None
        self.last_yolo_drowsy_prob = 0.0

        if self.use_yolo_drowsy:
            try:
                print("Checking/Downloading YOLO Drowsiness Classification model...")
                drowsy_dir = os.path.join(model_dir, "drowsiness_model")
                os.makedirs(drowsy_dir, exist_ok=True)
                drowsy_path = os.path.join(drowsy_dir, "best.pt")
                if not os.path.exists(drowsy_path):
                    drowsy_path = hf_hub_download(
                        repo_id="mosesb/drowsiness-detection-yolo-cls",
                        filename="best.pt",
                        local_dir=drowsy_dir
                    )
                self.yolo_drowsy_model = YOLO(drowsy_path)
                print("YOLO Drowsiness model loaded successfully!")
            except Exception as e:
                print(f"Failed to load YOLO drowsiness model: {e}. Running rule-based fusion.")
                self.use_yolo_drowsy = False

        self.use_yolo_phone = use_yolo_phone
        self.phone_detector = PhoneDetector(check_interval=10) # 3 FPS schedule
        self.last_raw_phone_detected = False
        self.last_raw_phone_conf = 0.0

    def process(self):
        print(f"Opening input video: {self.input_video}")
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            print(f"Error: Could not open input video {self.input_video}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_output = os.path.join(os.path.dirname(self.output_video), "temp_uncompressed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        print(f"Processing {total_frames} frames ({width}x{height} @ {fps:.2f} FPS)...")

        telemetry_records = []
        events_log = []
        frame_number = 0
        start_time = time.time()
        dt = 1.0 / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = round(frame_number * dt, 2)
            timestamp_ms = int(frame_number * dt * 1000)

            # ---------------------------------------------------------
            # 1. MediaPipe Landmarks (Fast ~60 FPS)
            # ---------------------------------------------------------
            landmarks = self.landmarker.process_frame(frame, timestamp_ms)

            # ---------------------------------------------------------
            # 2. Face Tracking & Quality Monitor
            # ---------------------------------------------------------
            face_data = self.face_tracker.update(landmarks, (height, width), dt)
            tracking_reliable = face_data["is_confident"]

            # ---------------------------------------------------------
            # 3. Scheduled Heavy Models (Periodic)
            # ---------------------------------------------------------
            # Drowsiness model at ~6 FPS (every 5th frame)
            if self.use_yolo_drowsy and self.yolo_drowsy_model is not None and frame_number % 5 == 0:
                try:
                    res = self.yolo_drowsy_model.predict(source=frame, verbose=False)
                    if res and res[0].probs:
                        probs = res[0].probs
                        drowsy_idx = None
                        for idx, name in self.yolo_drowsy_model.names.items():
                            if 'non' not in name.lower() and 'drowsy' in name.lower():
                                drowsy_idx = idx
                                break
                        if drowsy_idx is not None:
                            self.last_yolo_drowsy_prob = float(probs.data[drowsy_idx])
                        else:
                            self.last_yolo_drowsy_prob = float(probs.top1conf) if probs.top1 == 0 else 1.0 - float(probs.top1conf)
                except Exception:
                    pass

            # Phone detector at ~3 FPS (every 10th frame)
            if self.use_yolo_phone and frame_number % 10 == 0:
                phone_det, phone_conf = self.phone_detector.process_frame(frame, frame_number)
                self.last_raw_phone_detected = phone_det
                self.last_raw_phone_conf = phone_conf

            # ---------------------------------------------------------
            # 4. Yawn Engine
            # ---------------------------------------------------------
            yawn_data = self.yawn_engine.process(landmarks, dt, current_time)

            # ---------------------------------------------------------
            # 5. Eye Engine
            # ---------------------------------------------------------
            eye_data = self.eye_engine.process(
                landmarks=landmarks,
                dt=dt,
                current_time=current_time,
                is_stable_upright=True,
                mar=yawn_data["mar"],
                tracking_reliable=tracking_reliable
            )

            # ---------------------------------------------------------
            # 6. Head Pose & Nodding Engine
            # ---------------------------------------------------------
            head_pose_data = self.head_pose_engine.process(
                landmarks=landmarks,
                frame_shape=(height, width),
                dt=dt,
                is_eye_open=not eye_data["is_eye_closed"],
                is_eye_closed=eye_data["is_eye_closed"]
            )

            # ---------------------------------------------------------
            # 7. Gaze Engine (Gated by Eye Open State)
            # ---------------------------------------------------------
            gaze_data = self.gaze_engine.process(
                landmarks=landmarks,
                is_eye_open=not eye_data["is_eye_closed"],
                is_stable_upright=head_pose_data["is_head_forward"],
                tracking_reliable=tracking_reliable
            )

            # ---------------------------------------------------------
            # 8. Fatigue Engine (Multi-Window PERCLOS)
            # ---------------------------------------------------------
            fatigue_data = self.fatigue_engine.process(
                is_eye_closed=eye_data["is_eye_closed"],
                yawn_data=yawn_data,
                eye_data=eye_data,
                dt=dt
            )

            # ---------------------------------------------------------
            # 9. Drowsiness Engine (Microsleep & Nodding Fusion)
            # ---------------------------------------------------------
            drowsiness_data = self.drowsiness_engine.process(
                eye_data=eye_data,
                head_pose_data=head_pose_data,
                fatigue_data=fatigue_data,
                yawn_data=yawn_data,
                yolo_drowsy_prob=self.last_yolo_drowsy_prob,
                dt=dt
            )

            # ---------------------------------------------------------
            # 10. Phone Engine (Temporal Confirmation)
            # ---------------------------------------------------------
            phone_data = self.phone_engine.process(
                raw_detected=self.last_raw_phone_detected,
                raw_conf=self.last_raw_phone_conf,
                dt=dt
            )

            # ---------------------------------------------------------
            # 11. Distraction Engine (Gaze > Head, Drowsiness Gating)
            # ---------------------------------------------------------
            distraction_data = self.distraction_engine.process(
                gaze_data=gaze_data,
                head_pose_data=head_pose_data,
                eye_data=eye_data,
                drowsiness_data=drowsiness_data,
                face_track_data=face_data,
                dt=dt
            )

            # ---------------------------------------------------------
            # 12. Master State & Attention Fusion Engine
            # ---------------------------------------------------------
            frame_record, new_events = self.fusion_engine.process(
                timestamp_sec=current_time,
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

            # ---------------------------------------------------------
            # 13. RoadGuardian Black Box EDR Inference
            # ---------------------------------------------------------
            t_sec = int(frame_number * dt)
            is_event_trigger = (t_sec == 20)
            if is_event_trigger:
                speed_val = 75.0
                accel_val = -8.0
                brake_val = 0.98
                steering_val = 0.7
                distance_val = 2.0
                hazard_val = 1
            else:
                state_seed = frame_number // int(fps)
                rng = np.random.default_rng(state_seed)
                speed_val = float(rng.uniform(35, 60))
                accel_val = float(rng.uniform(-1, 1))
                brake_val = float(rng.uniform(0, 0.2))
                steering_val = float(rng.uniform(0, 0.2))
                distance_val = float(rng.uniform(20, 50))
                hazard_val = 0

            features_dict = {
                "speed": speed_val,
                "acceleration": accel_val,
                "braking": brake_val,
                "steering_deviation": steering_val,
                "nearest_vehicle_distance": distance_val,
                "pedestrian_detected": 0,
                "road_hazard_detected": hazard_val,
                "driver_distraction": float(frame_record["distraction_score"] / 100.0),
                "driver_drowsiness": float(frame_record["drowsiness_score"] / 100.0),
                "visibility": 0.95
            }
            bb_record = self.blackbox.process_frame(t_sec, features_dict)
            frame_record.update({
                "frame_number": frame_number,
                "event_status": bb_record["event_status"],
                "event_type": bb_record["event_type"],
                "status_confidence": bb_record["status_confidence"],
                "type_confidence": bb_record["type_confidence"],
                "speed": bb_record["speed"],
                "braking": bb_record["braking"],
                "distance": bb_record["nearest_vehicle_distance"],
                "hazard": bb_record["road_hazard_detected"],
                "visibility": bb_record["visibility"]
            })

            telemetry_records.append(frame_record)
            events_log.extend(new_events)

            # ---------------------------------------------------------
            # 14. Video Annotation & HUD Rendering
            # ---------------------------------------------------------
            annotated_frame = frame.copy()

            if landmarks:
                # Draw Face Bounding Box
                bbox = face_data.get("face_bbox")
                if bbox:
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (50, 80, 50), 1)

                # Draw iris highlights when open
                if not eye_data["is_eye_closed"] and len(landmarks) > 473:
                    for idx in [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]:
                        lm = landmarks[idx]
                        cv2.circle(annotated_frame, (int(lm.x * width), int(lm.y * height)), 2, (0, 255, 255), -1)

            # Render Translucent Cockpit HUD (Left Panel)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (15, 15), (380, 330), (10, 15, 25), -1)
            cv2.addWeighted(overlay, 0.72, annotated_frame, 0.28, 0, annotated_frame)
            cv2.rectangle(annotated_frame, (15, 15), (380, 330), (55, 65, 81), 1)

            y_offset = 36
            def draw_hud(label, val, col=(255, 255, 255), font_scale=0.45, thick=1):
                nonlocal y_offset
                cv2.putText(annotated_frame, f"{label}: {val}", (26, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thick, cv2.LINE_AA)
                y_offset += 21

            # Master State Header
            state_col = (0, 255, 0)
            if "DROWSY" in frame_record["master_driver_state"]:
                state_col = (0, 0, 255)
            elif frame_record["master_driver_state"] in ["DISTRACTED", "YAWNING", "FATIGUED", "PHONE_USAGE"]:
                state_col = (0, 165, 255)
            
            cv2.putText(annotated_frame, f"STATE: {frame_record['master_driver_state']}", (26, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_col, 2, cv2.LINE_AA)
            y_offset += 23

            # Core Scores
            att_col = (0, 255, 0) if frame_record["attention_score"] > 65 else ((0, 165, 255) if frame_record["attention_score"] > 40 else (0, 0, 255))
            draw_hud("Driver Attention Index", f"{frame_record['attention_score']} / 100", att_col, 0.48, 2)
            
            d_col = (0, 255, 0) if frame_record["drowsiness_score"] < 40 else ((0, 165, 255) if frame_record["drowsiness_score"] < 70 else (0, 0, 255))
            draw_hud("Drowsiness Score", f"{frame_record['drowsiness_score']} [{frame_record['drowsiness_state']}]", d_col)
            
            f_col = (0, 255, 0) if frame_record["fatigue_score"] < 45 else (0, 165, 255)
            draw_hud("Fatigue Score", f"{frame_record['fatigue_score']} [{frame_record['fatigue_state']}]", f_col)

            dist_col = (0, 255, 0) if frame_record["distraction_score"] < 40 else (0, 0, 255)
            draw_hud("Distraction Score", f"{frame_record['distraction_score']} [{frame_record['distraction_state']}]", dist_col)

            # Granular Metrics
            ear_stat = f"{frame_record['ear']:.2f} / Base:{frame_record['ear_baseline']:.2f} (Norm:{frame_record['normalized_ear']:.2f})"
            draw_hud("EAR", ear_stat, (0, 255, 0) if not frame_record["eye_closed_duration"] > 0.25 else (0, 0, 255))
            draw_hud("Eyes / Closure", f"{frame_record['eye_state']} ({frame_record['eye_closed_duration']:.2f}s | Blinks:{frame_record['blink_count']})")
            draw_hud("PERCLOS (S/M/L)", f"{frame_record['perclos_short']:.2f} / {frame_record['perclos_medium']:.2f} / {frame_record['perclos_long']:.2f}")
            draw_hud("Gaze Direction", f"{frame_record['gaze_state']} (Conf:{frame_record['gaze_confidence']:.2f})")
            
            pose_txt = f"{frame_record['head_pose_state']} (Y:{frame_record['yaw']:.0f}° P:{frame_record['pitch']:.0f}° V:{frame_record['head_pitch_velocity']:.0f}°/s)"
            draw_hud("Head Pose", pose_txt)
            draw_hud("Mouth / Yawn", f"MAR:{frame_record['mar']:.2f} [{frame_record['yawn_state']}] (P:{frame_record['yawn_probability']:.2f})")
            draw_hud("Phone Usage", "DETECTED 📱" if frame_record["phone_usage"] else "NONE", (0, 0, 255) if frame_record["phone_usage"] else (200, 200, 200))

            # Render Active Warning Banners on Center Bottom
            alert_y = height - 35
            for alert in frame_record["alerts"]:
                (tw, th), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
                cx = int(width / 2)
                cv2.rectangle(annotated_frame, (cx - int(tw/2) - 12, alert_y - th - 8), (cx + int(tw/2) + 12, alert_y + 6), (0, 0, 0), -1)
                cv2.rectangle(annotated_frame, (cx - int(tw/2) - 12, alert_y - th - 8), (cx + int(tw/2) + 12, alert_y + 6), (0, 0, 255), 2)
                cv2.putText(annotated_frame, alert, (cx - int(tw/2), alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
                alert_y -= 38

            out.write(annotated_frame)
            frame_number += 1
            if frame_number % 50 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")

        cap.release()
        out.release()
        self.landmarker.close()

        # Write Telemetry Data JSON file
        print(f"Writing telemetry logs to {self.telemetry_path}...")
        with open(self.telemetry_path, "w") as f:
            json.dump({
                "fps": fps,
                "total_frames": frame_number,
                "duration_sec": round(frame_number / fps, 2),
                "records": telemetry_records
            }, f, indent=2)

        # Write Events JSON file
        events_path = os.path.join(os.path.dirname(self.telemetry_path), "events.json")
        print(f"Writing event logs to {events_path}...")
        with open(events_path, "w") as f:
            json.dump(events_log, f, indent=2)

        # Transcode video to standard browser-compatible H.264
        print("Transcoding video to browser-compatible H.264 format using FFmpeg...")
        if os.path.exists(self.output_video):
            os.remove(self.output_video)
        
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_output, "-vcodec", "libx264", "-acodec", "aac", self.output_video
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(temp_output):
            os.remove(temp_output)

        processing_time = time.time() - start_time
        print("\n--- PERFORMANCE SUMMARY ---")
        print(f"Frames processed: {frame_number}")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Processing speed: {frame_number / max(0.1, processing_time):.2f} FPS")
        print(f"Final output video: {self.output_video}")
        return True
