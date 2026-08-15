# road_processor.py - Full Pipeline Video Processor for Road / External Camera
import cv2
import json
import time
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from .config import (
    PROCESS_WIDTH,
    PROCESS_HEIGHT,
    TARGET_CLASSES,
    DEFAULT_OUTPUT_DIR
)
from .lane_detection import AdvancedLaneDetector
from .distance_estimator import ADASDistanceEstimator
from .collision_analysis import CollisionSafetyAnalyzer
from .video_recorder import SafetyVideoRecorder
from .witness_finder import GoogleMapsWitnessFinder

class RoadVideoProcessor:
    def __init__(
        self,
        model_path="models/yolov8m.pt",
        input_video="input/external_road.mp4",
        output_video="output/processed_road.mp4",
        telemetry_path="output/road_telemetry.json",
        google_api_key=None
    ):
        self.input_video = input_video
        self.output_video = output_video
        self.telemetry_path = telemetry_path
        self.output_dir = os.path.dirname(self.output_video) or DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.telemetry_path) or self.output_dir, exist_ok=True)

        # Ensure model exists; fallback if needed
        if not os.path.exists(model_path):
            alt_models = ["models/yolov8m.pt", "models/yolo11n.pt", "yolo11n.pt", "yolov8n.pt"]
            for alt in alt_models:
                if os.path.exists(alt):
                    model_path = alt
                    break
        
        print(f"Loading YOLO Model for Road Perception: {model_path}...")
        self.model = YOLO(model_path)

        # Initialize sub-modules
        self.lane_detector = AdvancedLaneDetector(width=PROCESS_WIDTH, height=PROCESS_HEIGHT)
        self.distance_estimator = ADASDistanceEstimator()
        self.collision_analyzer = CollisionSafetyAnalyzer()
        self.recorder = SafetyVideoRecorder(output_dir=self.output_dir)
        self.witness_finder = GoogleMapsWitnessFinder(api_key=google_api_key)
        self.target_classes = TARGET_CLASSES

        # Default GPS
        self.mock_lat = 37.774929
        self.mock_lon = -122.419416

    def process(self):
        print(f"Opening input road video: {self.input_video}")
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            print(f"Error: Could not open road video source {self.input_video}")
            return False

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.recorder.initialize_buffer(fps)

        temp_output = os.path.join(self.output_dir, "temp_road_uncompressed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (PROCESS_WIDTH, PROCESS_HEIGHT))

        print(f"Processing road perception: {total_frames} frames ({orig_width}x{orig_height} -> {PROCESS_WIDTH}x{PROCESS_HEIGHT} @ {fps:.2f} FPS)...")

        telemetry_data = []
        events_log = []
        active_states = {
            "LANE_DRIFT_WARNING": False,
            "ROAD_HAZARD_WARNING": False,
            "HIGH_COLLISION_RISK": False,
            "ACCIDENT_EVENT": False
        }

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time_sec = round(frame_idx / fps, 2)

            # Resize to standard ADAS resolution
            frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            display_frame = frame_resized.copy()

            # 1. Advanced Lane Detection
            display_frame, lane_msg, lane_color, lane_deviation = self.lane_detector.process(display_frame)

            # 2. YOLO Object Detection & Tracking
            results = self.model.track(
                source=frame_resized,
                persist=True,
                classes=list(self.target_classes.keys()),
                verbose=False,
                conf=0.25
            )

            vehicle_count = 0
            pedestrian_count = 0
            cars_count = 0
            trucks_count = 0
            buses_count = 0
            bikes_count = 0

            road_hazard_detected = False
            highest_collision_risk = "Low"
            nearest_dist = 999.0
            current_detections = []
            accident_should_trigger = False

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    track_id = int(box.id[0].item()) if box.id is not None else -1
                    conf = float(box.conf[0].item())
                    cls_idx = int(box.cls[0].item())
                    class_name = self.target_classes.get(cls_idx, "vehicle")

                    if class_name == "pedestrian":
                        pedestrian_count += 1
                    else:
                        vehicle_count += 1
                        if class_name == "car":
                            cars_count += 1
                        elif class_name in ["truck"]:
                            trucks_count += 1
                        elif class_name in ["bus"]:
                            buses_count += 1
                        elif class_name in ["motorcycle", "bicycle"]:
                            bikes_count += 1

                    # Estimate distance
                    dist_m = self.distance_estimator.estimate_distance(
                        xyxy, class_name, PROCESS_HEIGHT, PROCESS_WIDTH
                    )
                    if dist_m < nearest_dist:
                        nearest_dist = dist_m

                    # Collision risk & TTC
                    risk = "Low"
                    ttc = 999.0
                    rel_speed = 0.0
                    if track_id != -1:
                        risk, ttc, rel_speed = self.collision_analyzer.analyze_collision_risk(
                            track_id, dist_m, current_time_sec
                        )
                        if risk == "High":
                            highest_collision_risk = "High"
                        elif risk == "Medium" and highest_collision_risk != "High":
                            highest_collision_risk = "Medium"

                    # Check road hazard
                    hazard = self.collision_analyzer.check_road_hazard(xyxy, dist_m, PROCESS_WIDTH)
                    if hazard:
                        road_hazard_detected = True

                    # Automatic accident condition
                    if self.collision_analyzer.check_accident_impact(risk, dist_m, rel_speed):
                        accident_should_trigger = True

                    # Record detection object
                    current_detections.append({
                        "track_id": track_id,
                        "class_name": class_name,
                        "confidence": round(conf, 2),
                        "bbox": [int(x) for x in xyxy],
                        "distance_m": dist_m,
                        "collision_risk": risk,
                        "ttc_sec": round(ttc, 2) if ttc < 999.0 else None,
                        "relative_speed_mps": round(rel_speed, 2)
                    })

                    # Bounding Box Color
                    box_color = (0, 255, 0)
                    if risk == "Medium":
                        box_color = (0, 165, 255)
                    elif risk == "High":
                        box_color = (0, 0, 255)

                    # Draw Bounding Box & Label
                    cv2.rectangle(display_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
                    
                    label_text = f"ID:{track_id} {class_name.upper()} {dist_m:.1f}m"
                    if risk != "Low":
                        label_text += f" [{risk} RISK]"
                    
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(display_frame, (xyxy[0], max(0, xyxy[1] - 18)), (xyxy[0] + tw + 6, max(18, xyxy[1])), box_color, -1)
                    cv2.putText(
                        display_frame,
                        label_text,
                        (xyxy[0] + 3, max(14, xyxy[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 0, 0) if risk == "Low" else (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

            if nearest_dist == 999.0:
                nearest_dist = 0.0

            # 3. Trigger / Buffer Accident Event
            if accident_should_trigger:
                if not self.recorder.is_recording:
                    witness_info = self.witness_finder.discover_nearby_witness_vehicles(
                        self.mock_lat, self.mock_lon, datetime.now().isoformat()
                    )
                    self.recorder.trigger_accident(
                        (PROCESS_WIDTH, PROCESS_HEIGHT), current_detections, witness_info
                    )

            if self.recorder.is_recording:
                # Add flashing red banner
                rec_frame = display_frame.copy()
                if int(time.time() * 2) % 2 == 0:
                    cv2.rectangle(rec_frame, (0, 0), (PROCESS_WIDTH, 48), (0, 0, 255), -1)
                    cv2.putText(
                        rec_frame,
                        "CRITICAL ACCIDENT DETECTED - RECORDING WITNESS BLACKBOX BUFFER",
                        (PROCESS_WIDTH // 2 - 340, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                self.recorder.write_post_frame(rec_frame)
                display_frame = rec_frame
            else:
                self.recorder.buffer_frame(display_frame)

            # 4. ADAS Dashboard Telemetry HUD Overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (15, 15), (380, 215), (10, 15, 25), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            cv2.rectangle(display_frame, (15, 15), (380, 215), (55, 65, 81), 1)

            cv2.putText(display_frame, "ROADGUARDIAN ADAS INTELLIGENCE", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Vehicles Ahead: {vehicle_count} (Cars:{cars_count}, Trucks:{trucks_count})", (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1)
            cv2.putText(display_frame, f"Pedestrians: {pedestrian_count}", (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1)
            
            # Lane departure
            cv2.putText(display_frame, f"Lane Status: {lane_msg}", (28, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, lane_color, 2 if "Drift" in lane_msg else 1)
            
            # Road hazard
            hazard_txt = "⚠️ HAZARD DETECTED!" if road_hazard_detected else "CLEAR 🟢"
            hazard_col = (0, 0, 255) if road_hazard_detected else (0, 255, 0)
            cv2.putText(display_frame, f"Road Corridor: {hazard_txt}", (28, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.48, hazard_col, 2 if road_hazard_detected else 1)
            
            # Risk
            risk_col = (0, 255, 0) if highest_collision_risk == "Low" else ((0, 165, 255) if highest_collision_risk == "Medium" else (0, 0, 255))
            cv2.putText(display_frame, f"Collision Risk: {highest_collision_risk.upper()}", (28, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.48, risk_col, 2 if highest_collision_risk != "Low" else 1)

            # Nearest vehicle
            cv2.putText(display_frame, f"Nearest Target: {nearest_dist:.1f} m", (28, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1)

            # Write annotated frame
            out.write(display_frame)

            # 5. Events Logging
            if "Drift" in lane_msg and not active_states["LANE_DRIFT_WARNING"]:
                active_states["LANE_DRIFT_WARNING"] = True
                events_log.append({"timestamp": current_time_sec, "event": "LANE_DRIFT_WARNING", "value": round(lane_deviation, 1)})
            elif "Drift" not in lane_msg and active_states["LANE_DRIFT_WARNING"]:
                active_states["LANE_DRIFT_WARNING"] = False

            if road_hazard_detected and not active_states["ROAD_HAZARD_WARNING"]:
                active_states["ROAD_HAZARD_WARNING"] = True
                events_log.append({"timestamp": current_time_sec, "event": "ROAD_HAZARD_WARNING", "value": round(nearest_dist, 1)})
            elif not road_hazard_detected and active_states["ROAD_HAZARD_WARNING"]:
                active_states["ROAD_HAZARD_WARNING"] = False

            if highest_collision_risk == "High" and not active_states["HIGH_COLLISION_RISK"]:
                active_states["HIGH_COLLISION_RISK"] = True
                events_log.append({"timestamp": current_time_sec, "event": "HIGH_COLLISION_RISK", "value": 100})
            elif highest_collision_risk != "High" and active_states["HIGH_COLLISION_RISK"]:
                active_states["HIGH_COLLISION_RISK"] = False

            # 6. Save Telemetry Record
            telemetry_data.append({
                "frame_number": frame_idx,
                "timestamp_sec": current_time_sec,
                "vehicle_count": vehicle_count,
                "pedestrian_count": pedestrian_count,
                "cars_count": cars_count,
                "trucks_count": trucks_count,
                "buses_count": buses_count,
                "bikes_count": bikes_count,
                "lane_status": lane_msg,
                "lane_deviation_px": round(lane_deviation, 1),
                "road_hazard": 1 if road_hazard_detected else 0,
                "collision_risk": highest_collision_risk,
                "nearest_vehicle_distance": round(nearest_dist, 1),
                "detections": current_detections
            })

            if frame_idx % 50 == 0:
                print(f"Road processor: {frame_idx}/{total_frames} frames processed...")

        cap.release()
        out.release()
        self.recorder.close()

        # Save Road Telemetry JSON
        print(f"Writing road telemetry logs to {self.telemetry_path}...")
        with open(self.telemetry_path, "w") as f:
            json.dump({
                "fps": fps,
                "total_frames": frame_idx,
                "duration_sec": round(frame_idx / fps, 2),
                "records": telemetry_data
            }, f, indent=2)

        # Save Road Events JSON
        events_path = os.path.join(os.path.dirname(self.telemetry_path), "road_events.json")
        print(f"Writing road events logs to {events_path}...")
        with open(events_path, "w") as f:
            json.dump(events_log, f, indent=2)

        # Transcode video to standard H.264
        print(f"Transcoding labeled road video to H.264: {self.output_video}...")
        if os.path.exists(self.output_video):
            os.remove(self.output_video)

        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_output, "-vcodec", "libx264", "-acodec", "aac", self.output_video
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(temp_output):
            os.remove(temp_output)

        processing_time = time.time() - start_time
        print("\n--- ROAD ADAS PERFORMANCE SUMMARY ---")
        print(f"Frames processed: {frame_idx}")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Processing speed: {frame_idx / max(0.1, processing_time):.2f} FPS")
        print(f"Final labeled road output video: {self.output_video}")
        return True
