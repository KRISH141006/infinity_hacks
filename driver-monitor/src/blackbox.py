
import joblib

import pandas as pd
import numpy as np
import os
import json
from collections import deque

class BlackBoxRecorder:
    def __init__(self, model_dir="models", output_dir="output"):
        self.model_dir = model_dir
        self.output_dir = output_dir

        self.status_model_path = os.path.join(model_dir, "roadguardian", "roadguardian_event_status_rf_v2.pkl")
        self.type_model_path = os.path.join(model_dir, "roadguardian", "roadguardian_event_type_rf_v1.pkl")

        # Feature column ordering required by the models
        self.feature_cols = [
            "speed",
            "acceleration",
            "braking",
            "steering_deviation",
            "nearest_vehicle_distance",
            "pedestrian_detected",
            "road_hazard_detected",
            "driver_distraction",
            "driver_drowsiness",
            "visibility"
        ]

        # Load models
        print("Loading RoadGuardian Event Classification models...")
        self.status_model = joblib.load(self.status_model_path)
        self.type_model = joblib.load(self.type_model_path)
        print("RoadGuardian models loaded successfully!")

        # Buffers for Black Box
        self.pre_event_buffer = deque(maxlen=15)
        self.post_event_data = []
        self.event_record = None

        # Tracking state
        self.event_detected = False
        self.event_time = None
        self.post_event_limit = 10  # Gather 10 post-event frames
        self.finished = False

    def process_frame(self, t, features_dict):
        # Format values into DataFrame with exact features
        sample = pd.DataFrame([features_dict])[self.feature_cols]

        # Predict Event Status
        status_pred = self.status_model.predict(sample)[0]
        status_prob = self.status_model.predict_proba(sample).max()

        # Predict Event Type
        type_pred = self.type_model.predict(sample)[0]
        type_prob = self.type_model.predict_proba(sample).max()

        # Consistency Rule: If type is COLLISION, make status CRITICAL
        final_status = status_pred
        if type_pred == "COLLISION":
            final_status = "CRITICAL"

        # Construct final frame log record
        current_record = {
            "time": t,
            **features_dict,
            "event_status": final_status,
            "event_type": type_pred,
            "status_confidence": float(status_prob),
            "type_confidence": float(type_prob)
        }

        # Event trigger logic (detect the first transition to CRITICAL)
        if not self.event_detected and final_status == "CRITICAL":
            self.event_detected = True
            self.event_time = t
            self.event_record = current_record.copy()
            # Lock the 15 pre-event frames
            self.pre_event_data = list(self.pre_event_buffer)
            print(f"🚨 ROADGUARDIAN: CRITICAL EVENT DETECTED at t={t}s! Status: {final_status}, Type: {type_pred}")

        elif self.event_detected and not self.finished:
            # Collect post-event records
            self.post_event_data.append(current_record)
            # If we reached 10 post-event records, dump the blackbox event data
            if len(self.post_event_data) >= self.post_event_limit:
                self.save_black_box()
                self.finished = True

        # Keep feeding rolling buffer
        self.pre_event_buffer.append(current_record)

        return current_record

    def save_black_box(self):
        # Create consolidated 26-frame list
        black_box_data = (
            self.pre_event_data +
            [self.event_record] +
            self.post_event_data
        )

        # Save to CSV
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "roadguardian_blackbox_event.csv")
        df = pd.DataFrame(black_box_data)
        df.to_csv(csv_path, index=False)
        print(f"📁 RoadGuardian Black Box CSV saved to {csv_path}")

        # Save JSON Incident Summary
        summary_path = os.path.join(self.output_dir, "roadguardian_incident_summary.json")
        summary = {
            "event_id": f"RG_{int(self.event_time)}",
            "event_time_sec": float(self.event_time),
            "event_status": self.event_record["event_status"],
            "event_type": self.event_record["event_type"],
            "status_confidence": float(self.event_record["status_confidence"]),
            "type_confidence": float(self.event_record["type_confidence"]),
            "trigger_speed_kmh": float(self.event_record["speed"]),
            "trigger_braking_level": float(self.event_record["braking"]),
            "trigger_distraction": float(self.event_record["driver_distraction"]),
            "trigger_drowsiness": float(self.event_record["driver_drowsiness"]),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📁 RoadGuardian Incident Summary JSON saved to {summary_path}")
