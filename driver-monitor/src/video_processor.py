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
    def __init__(self, model_dir="models", input_video="input/driver.mp4", output_video="output/processed_driver.mp4", telemetry_path="output/telemetry.json"):
        self.input_video = input_video
        self.output_video = output_video
        self.telemetry_path = telemetry_path

        # Create output directories if needed
        os.makedirs(os.path.dirname(self.output_video), exist_ok=True)

        face_task_path = os.path.join(model_dir, "face_landmarker.task")

        # Initialize core pipelines
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
        if fps <= 0:
            fps = 25.0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_output = "output/temp_uncompressed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        print(f"Processing {total_frames} frames ({width}x{height} @ {fps:.2f} FPS)...")

        telemetry_data = []
        frame_number = 0
        start_time = time.time()

        events_log = []
        active_states = {
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
            
            # 1. MediaPipe Landmarks
            landmarks = self.landmarker.process_frame(frame, timestamp_ms)

            # 2. Extract Features
            # Drowsiness
            drowsy_data = self.drowsiness.process_frame(landmarks, frame, fps, frame_number)
            
            # Head pose
            yaw, pitch, roll, pose_state = self.head_pose.estimate_pose(landmarks, width, height)
            head_pose_data = (yaw, pitch, roll, pose_state)

            # Gaze
            gaze_state = self.gaze.estimate_gaze(landmarks)

            # Phone usage (every 5th frame)
            phone_detected, phone_conf = self.phone.process_frame(frame, frame_number)
            phone_data = (phone_detected, phone_conf)

            # 3. Decision Logic & Alert System
            alert_data = self.attention.compute_scores(
                drowsy_data,
                head_pose_data,
                gaze_state,
                phone_data,
                fps
            )

            # RoadGuardian Black Box simulation / prediction
            t_sec = int(frame_number / fps)
            
            # Sudden critical accident scenario occurs at t = 20 seconds
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
                # normal random fluctuations
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

            # Track transitions and log events
            current_time_sec = round(frame_number / fps, 1)
            alert_map = {
                "🔴 DROWSINESS DETECTED": ("DROWSINESS_DETECTED", drowsy_data["drowsiness_score"]),
                "🟠 DRIVER DISTRACTED": ("DRIVER_DISTRACTED", alert_data["distraction_score"]),
                "⚠️ DRIVER LOOKING AWAY": ("DRIVER_LOOKING_AWAY", alert_data["distraction_score"]),
                "🔴 PHONE USAGE DETECTED": ("PHONE_USAGE_DETECTED", 100),
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

            # 4. Save Telemetry
            telemetry_data.append({
                "frame_number": frame_number,
                "timestamp_sec": round(frame_number / fps, 2),
                "ear": round(drowsy_data["ear"], 3),
                "mar": round(drowsy_data["mar"], 3),
                "drowsiness_score": drowsy_data["drowsiness_score"],
                "distraction_score": alert_data["distraction_score"],
                "attention_score": alert_data["attention_score"],
                "gaze_state": gaze_state,
                "head_pose_state": alert_data["head_pose_state"],
                "yaw": round(alert_data["relative_yaw"], 1),
                "pitch": round(alert_data["relative_pitch"], 1),
                "roll": round(alert_data["relative_roll"], 1),
                "phone_usage": phone_detected,
                "phone_conf": round(phone_conf, 2),
                "alerts": alert_data["alerts"],
                
                # RoadGuardian values
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

            # 5. Visual Overlays
            annotated_frame = frame.copy()

            if landmarks:
                # Draw Face Mesh (select points)
                for lm in landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)

                # Draw Iris highlights
                # Left Iris indices
                for idx in [468, 469, 470, 471, 472]:
                    lm = landmarks[idx]
                    cv2.circle(annotated_frame, (int(lm.x * width), int(lm.y * height)), 2, (0, 255, 255), -1)
                # Right Iris indices
                for idx in [473, 474, 475, 476, 477]:
                    lm = landmarks[idx]
                    cv2.circle(annotated_frame, (int(lm.x * width), int(lm.y * height)), 2, (0, 255, 255), -1)

            # Dashboard HUD Overlay (Translucent Box)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (15, 15), (320, 240), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            # Render Hud Metrics
            y_offset = 40
            def draw_stat(label, val, color=(255, 255, 255)):
                nonlocal y_offset
                cv2.putText(annotated_frame, f"{label}: {val}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y_offset += 25

            draw_stat("EAR", f"{drowsy_data['ear']:.2f}")
            draw_stat("MAR", f"{drowsy_data['mar']:.2f}")
            
            d_color = (0, 255, 0)
            if drowsy_data['drowsiness_score'] > 60:
                d_color = (0, 0, 255)
            elif drowsy_data['drowsiness_score'] > 30:
                d_color = (0, 165, 255)
            draw_stat("Drowsiness Score", f"{drowsy_data['drowsiness_score']}", d_color)

            dist_color = (0, 255, 0)
            if alert_data['distraction_score'] > 50:
                dist_color = (0, 0, 255)
            draw_stat("Distraction Score", f"{alert_data['distraction_score']}", dist_color)

            att_color = (0, 255, 0)
            if alert_data['attention_score'] < 40:
                att_color = (0, 0, 255)
            elif alert_data['attention_score'] < 70:
                att_color = (0, 165, 255)
            draw_stat("Attention Score", f"{alert_data['attention_score']}", att_color)
            
            draw_stat("Head Pose", f"{alert_data['head_pose_state']} (Y:{alert_data['relative_yaw']:.0f} P:{alert_data['relative_pitch']:.0f})")
            draw_stat("Gaze", gaze_state)
            draw_stat("Phone Usage", "DETECTED" if phone_detected else "NONE", (0, 0, 255) if phone_detected else (255, 255, 255))

            # Display active alerts on bottom-center of the screen
            alert_y = height - 40
            for alert in alert_data["alerts"]:
                # Draw black background label
                cv2.putText(annotated_frame, alert, (int(width/2) - 150, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                alert_y -= 30

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
                "total_frames": total_frames,
                "duration_sec": total_frames / fps,
                "records": telemetry_data
            }, f, indent=2)

        # Write Events JSON file
        events_path = os.path.join(os.path.dirname(self.telemetry_path), "events.json")
        print(f"Writing event logs to {events_path}...")
        with open(events_path, "w") as f:
            json.dump(events_log, f, indent=2)

        # Transcode video to standard H.264
        print("Transcoding video to browser-compatible H.264 format using FFmpeg...")
        if os.path.exists(self.output_video):
            os.remove(self.output_video)
        
        # Call FFmpeg
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
        print(f"Processing speed: {frame_number / processing_time:.2f} FPS")
        print(f"Final output video: {self.output_video}")
        return True
