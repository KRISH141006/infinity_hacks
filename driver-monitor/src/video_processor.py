import cv2
import json
import time
import os
import numpy as np
from src.face_landmarks import FaceLandmarkerHelper
from src.drowsiness import DrowsinessDetector
from src.head_pose import HeadPoseEstimator
from src.gaze import GazeTracker
from src.phone_detection import PhoneDetector
from src.attention import AttentionManager
from src.blackbox import BlackBoxRecorder

class VideoProcessor:
    def __init__(
        self,
        model_dir="models",
        input_video="input/driver.mp4",
        output_video="output/processed_driver.mp4",
        telemetry_path="output/telemetry.json"
    ):
        self.input_video = input_video
        self.output_video = output_video
        self.telemetry_path = telemetry_path

        # Create output directories if needed
        os.makedirs(os.path.dirname(self.output_video) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.telemetry_path) or ".", exist_ok=True)

        face_task_path = os.path.join(model_dir, "face_landmarker.task")

        # Initialize core modular engines
        self.landmarker = FaceLandmarkerHelper(face_task_path)
        self.drowsiness = DrowsinessDetector(use_yolo=True)
        self.head_pose = HeadPoseEstimator()
        self.gaze = GazeTracker()
        self.phone = PhoneDetector(check_interval=5)
        self.attention = AttentionManager()
        self.blackbox = BlackBoxRecorder(model_dir=model_dir, output_dir=os.path.dirname(telemetry_path))

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

        telemetry_data = []
        frame_number = 0
        start_time = time.time()

        events_log = []
        active_states = {
            "CRITICAL_DROWSY_NODDING": False,
            "DROWSINESS_DETECTED": False,
            "DRIVER_DISTRACTED": False,
            "DRIVER_LOOKING_AWAY": False,
            "PHONE_USAGE_DETECTED": False,
            "LOW_DRIVER_ATTENTION": False
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_number / fps * 1000)
            
            # 1. MediaPipe Landmarks Extraction
            landmarks = self.landmarker.process_frame(frame, timestamp_ms)

            # 2. Extract Head Pose first to guide adaptive calibration
            raw_yaw, raw_pitch, raw_roll, coarse_pose = self.head_pose.estimate_pose(landmarks, width, height)
            head_pose_data = (raw_yaw, raw_pitch, raw_roll, coarse_pose)
            
            # Approximate head forward check for calibration
            is_head_forward = (coarse_pose in ["FORWARD", "UNKNOWN", "CALIBRATING"])

            # 3. Process Drowsiness with Adaptive EAR and Nodding Analysis
            drowsy_data = self.drowsiness.process_frame(
                landmarks=landmarks,
                frame=frame,
                fps=fps,
                frame_number=frame_number,
                relative_pitch=raw_pitch - self.attention.neutral_pitch,
                head_forward=is_head_forward
            )

            # 4. Gaze Tracking with Eye-Open Gating
            is_eye_open = not drowsy_data.get("is_eye_closed", False)
            gaze_state, gaze_conf = self.gaze.estimate_gaze(
                landmarks=landmarks,
                is_eye_open=is_eye_open,
                ear_val=drowsy_data["ear"]
            )

            # 5. Phone Usage Detection
            phone_detected, phone_conf = self.phone.process_frame(frame, frame_number)
            phone_data = (phone_detected, phone_conf)

            # 6. Unified Temporal Decision & Attention Fusion
            alert_data = self.attention.compute_scores(
                drowsy_data=drowsy_data,
                head_pose_data=head_pose_data,
                gaze_data=(gaze_state, gaze_conf),
                phone_data=phone_data,
                fps=fps
            )

            # 7. RoadGuardian Black Box simulation / prediction
            t_sec = int(frame_number / fps)
            is_event_trigger = (t_sec == 20)
            
            if is_event_trigger:
                speed_val = 75.0
                accel_val = -8.0
                brake_val = 0.98
                steering_val = 0.7
                distance_val = 2.0
                pedestrian_val = 0
                hazard_val = 1
                visibility_val = 0.9
            else:
                state_seed = frame_number // int(fps)
                rng = np.random.default_rng(state_seed)
                speed_val = float(rng.uniform(35, 60))
                accel_val = float(rng.uniform(-1, 1))
                brake_val = float(rng.uniform(0, 0.2))
                steering_val = float(rng.uniform(0, 0.2))
                distance_val = float(rng.uniform(20, 50))
                pedestrian_val = 0
                hazard_val = 0
                visibility_val = float(rng.uniform(0.8, 1.0))
                
            features_dict = {
                "speed": speed_val,
                "acceleration": accel_val,
                "braking": brake_val,
                "steering_deviation": steering_val,
                "nearest_vehicle_distance": distance_val,
                "pedestrian_detected": pedestrian_val,
                "road_hazard_detected": hazard_val,
                "driver_distraction": float(alert_data["distraction_score"] / 100.0),
                "driver_drowsiness": float(drowsy_data["drowsiness_score"] / 100.0),
                "visibility": visibility_val
            }
            
            bb_record = self.blackbox.process_frame(t_sec, features_dict)

            # 8. State-Transition Event Logging (Deduplicated Transitions)
            current_time_sec = round(frame_number / fps, 2)
            alert_map = {
                "🔴 CRITICAL: DROWSY NODDING OFF": ("CRITICAL_DROWSY_NODDING", drowsy_data["drowsiness_score"]),
                "🔴 DROWSINESS DETECTED": ("DROWSINESS_DETECTED", drowsy_data["drowsiness_score"]),
                "🟠 DRIVER DISTRACTED": ("DRIVER_DISTRACTED", alert_data["distraction_score"]),
                "⚠️ DRIVER LOOKING AWAY": ("DRIVER_LOOKING_AWAY", alert_data["distraction_score"]),
                "📱 PHONE USAGE DETECTED": ("PHONE_USAGE_DETECTED", 100),
                "⚠️ LOW DRIVER ATTENTION": ("LOW_DRIVER_ATTENTION", alert_data["attention_score"])
            }
            
            frame_alerts = [a.strip() for a in alert_data["alerts"]]
            for alert_text, (event_tag, val) in alert_map.items():
                is_active = alert_text in frame_alerts
                if is_active and not active_states[event_tag]:
                    active_states[event_tag] = True
                    events_log.append({
                        "timestamp": current_time_sec,
                        "event": event_tag,
                        "value": val
                    })
                elif not is_active and active_states[event_tag]:
                    active_states[event_tag] = False
                    events_log.append({
                        "timestamp": current_time_sec,
                        "event": f"{event_tag}_RESOLVED",
                        "value": 0
                    })

            # 9. Comprehensive Telemetry Record (All Section 12 Debug Fields Included)
            telemetry_data.append({
                "frame_number": frame_number,
                "timestamp_sec": current_time_sec,
                
                # EAR & Eye metrics
                "ear": drowsy_data["ear"],
                "ear_baseline": drowsy_data["ear_baseline"],
                "ear_threshold": drowsy_data["ear_threshold"],
                "eye_closed_duration": drowsy_data["eye_closed_duration"],
                "is_eye_closed": drowsy_data["is_eye_closed"],
                "perclos": drowsy_data["perclos"],
                "mar": drowsy_data["mar"],
                "blink_count": drowsy_data["blink_count"],
                "slow_blink_count": drowsy_data["slow_blink_count"],
                "yawn_detected": drowsy_data["yawn_detected"],
                
                # Head pose & Pitch velocity
                "raw_yaw": alert_data["raw_yaw"],
                "raw_pitch": alert_data["raw_pitch"],
                "raw_roll": alert_data["raw_roll"],
                "neutral_yaw": alert_data["neutral_yaw"],
                "neutral_pitch": alert_data["neutral_pitch"],
                "neutral_roll": alert_data["neutral_roll"],
                "yaw": alert_data["relative_yaw"],
                "pitch": alert_data["relative_pitch"],
                "roll": alert_data["relative_roll"],
                "head_pitch_velocity": drowsy_data["head_pitch_velocity"],
                "head_pose_state": alert_data["head_pose_state"],
                "is_nodding": drowsy_data["is_nodding"],
                
                # Gaze tracking
                "gaze_state": alert_data["gaze_state"],
                "gaze_confidence": alert_data["gaze_confidence"],
                
                # Scores & States
                "drowsiness_score": drowsy_data["drowsiness_score"],
                "drowsiness_state": drowsy_data["drowsiness_state"],
                "distraction_score": alert_data["distraction_score"],
                "distraction_state": alert_data["distraction_state"],
                "distraction_duration": alert_data["distraction_duration"],
                "attention_score": alert_data["attention_score"],
                
                # Phone & Alerts
                "phone_usage": phone_detected,
                "phone_conf": round(phone_conf, 2),
                "alerts": alert_data["alerts"],
                
                # RoadGuardian blackbox features
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

            # 10. Visual Overlays & HUD
            annotated_frame = frame.copy()

            if landmarks:
                # Draw select facial mesh points
                for lm in landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)

                # Highlight Irises if eyes open
                if is_eye_open and len(landmarks) > 473:
                    for idx in [468, 469, 470, 471, 472]:
                        lm = landmarks[idx]
                        cv2.circle(annotated_frame, (int(lm.x * width), int(lm.y * height)), 2, (0, 255, 255), -1)
                    for idx in [473, 474, 475, 476, 477]:
                        lm = landmarks[idx]
                        cv2.circle(annotated_frame, (int(lm.x * width), int(lm.y * height)), 2, (0, 255, 255), -1)

            # Dashboard HUD Overlay (Translucent Box)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (15, 15), (370, 310), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, annotated_frame, 0.35, 0, annotated_frame)
            cv2.rectangle(annotated_frame, (15, 15), (370, 310), (55, 65, 81), 1)

            # Render HUD Metrics
            y_offset = 38
            def draw_stat(label, val, color=(255, 255, 255)):
                nonlocal y_offset
                cv2.putText(annotated_frame, f"{label}: {val}", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                y_offset += 22

            cv2.putText(annotated_frame, "ROADGUARDIAN DMS INTELLIGENCE", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 24

            # EAR status with baseline
            ear_col = (0, 255, 0) if not drowsy_data["is_eye_closed"] else (0, 0, 255)
            draw_stat("EAR / Base (Thresh)", f"{drowsy_data['ear']:.2f} / {drowsy_data['ear_baseline']:.2f} ({drowsy_data['ear_threshold']:.2f})", ear_col)
            draw_stat("Eye Closure Duration", f"{drowsy_data['eye_closed_duration']:.2f} s", (0, 0, 255) if drowsy_data['eye_closed_duration'] >= 1.5 else (209, 213, 219))
            draw_stat("PERCLOS / MAR", f"{drowsy_data['perclos']:.2f} / {drowsy_data['mar']:.2f}", (209, 213, 219))
            
            # Drowsiness state & score
            d_color = (0, 255, 0)
            if drowsy_data['drowsiness_score'] >= 70:
                d_color = (0, 0, 255)
            elif drowsy_data['drowsiness_score'] >= 40:
                d_color = (0, 165, 255)
            draw_stat("Drowsiness", f"{drowsy_data['drowsiness_score']} [{drowsy_data['drowsiness_state']}]", d_color)

            # Distraction state & score
            dist_color = (0, 255, 0)
            if alert_data['distraction_score'] >= 60:
                dist_color = (0, 0, 255)
            elif alert_data['distraction_score'] >= 35:
                dist_color = (0, 165, 255)
            draw_stat("Distraction", f"{alert_data['distraction_score']} [{alert_data['distraction_state']}]", dist_color)

            # Attention index
            att_color = (0, 255, 0)
            if alert_data['attention_score'] <= 40:
                att_color = (0, 0, 255)
            elif alert_data['attention_score'] <= 70:
                att_color = (0, 165, 255)
            draw_stat("Driver Attention Index", f"{alert_data['attention_score']} / 100", att_color)
            
            draw_stat("Head Pose", f"{alert_data['head_pose_state']} (Y:{alert_data['relative_yaw']:.0f}° P:{alert_data['relative_pitch']:.0f}° V:{drowsy_data['head_pitch_velocity']:.0f}°/s)")
            draw_stat("Gaze Direction", f"{alert_data['gaze_state']} (Conf:{alert_data['gaze_confidence']:.2f})")
            draw_stat("Phone Usage", "DETECTED 📱" if phone_detected else "NONE", (0, 0, 255) if phone_detected else (209, 213, 219))

            # Display active alerts banner on bottom center
            alert_y = height - 40
            for alert in alert_data["alerts"]:
                (tw, th), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                cx = int(width / 2)
                cv2.rectangle(annotated_frame, (cx - int(tw/2) - 10, alert_y - th - 8), (cx + int(tw/2) + 10, alert_y + 6), (0, 0, 0), -1)
                cv2.putText(annotated_frame, alert, (cx - int(tw/2), alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA)
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
                "records": telemetry_data
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
