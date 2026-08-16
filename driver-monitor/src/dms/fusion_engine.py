# fusion_engine.py - Multi-Signal Driver State Fusion, Temporal Buffer, Attention & Alert Engine
import numpy as np
from collections import deque

class FusionEngine:
    def __init__(
        self,
        buffer_len=300,
        weight_drowsiness=0.45,
        weight_distraction=0.35,
        weight_phone=0.20
    ):
        self.buffer_len = buffer_len
        self.temporal_buffer = deque(maxlen=buffer_len)

        self.w_drowsy = weight_drowsiness
        self.w_distract = weight_distraction
        self.w_phone = weight_phone

        # Global Master State
        # ALERT, FATIGUED, YAWNING, DISTRACTED, DROWSY, DROWSY_NODDING, FACE_UNCERTAIN
        self.master_driver_state = "ALERT"
        self.driver_attention_score = 100

        # State Transition Event Trackers
        self.active_alert_states = {
            "CRITICAL_DROWSY_NODDING": False,
            "DROWSINESS_DETECTED": False,
            "FATIGUE_DETECTED": False,
            "YAWNING_DETECTED": False,
            "DRIVER_DISTRACTED": False,
            "DRIVER_LOOKING_AWAY": False,
            "PHONE_USAGE_DETECTED": False,
            "LOW_DRIVER_ATTENTION": False
        }

    def process(
        self,
        timestamp_sec,
        face_data,
        eye_data,
        gaze_data,
        head_pose_data,
        yawn_data,
        fatigue_data,
        distraction_data,
        drowsiness_data,
        phone_data
    ):
        drowsy_score = drowsiness_data.get("drowsiness_score", 0)
        drowsy_state = drowsiness_data.get("drowsiness_state", "ALERT")
        is_nodding = drowsiness_data.get("is_nodding_off", False)

        distract_score = distraction_data.get("distraction_score", 0)
        distract_state = distraction_data.get("distraction_state", "ATTENTIVE")
        is_distracted = distraction_data.get("is_distracted", False)

        fatigue_score = fatigue_data.get("fatigue_score", 0)
        fatigue_state = fatigue_data.get("fatigue_state", "ALERT")

        is_yawning = yawn_data.get("is_yawning", False)
        phone_usage = phone_data.get("phone_usage", False)
        tracking_reliable = face_data.get("is_confident", True)

        # -------------------------------------------------------------
        # 1. DRIVER ATTENTION INDEX (0 to 100)
        # -------------------------------------------------------------
        phone_penalty = 100.0 if phone_usage else 0.0

        risk_score = (
            (drowsy_score * self.w_drowsy) +
            (distract_score * self.w_distract) +
            (phone_penalty * self.w_phone)
        )
        if not tracking_reliable:
            risk_score = max(risk_score, 30.0)

        attention_score = int(max(0, min(100, 100.0 - risk_score)))
        self.driver_attention_score = attention_score

        # -------------------------------------------------------------
        # 2. STATE CONFLICT RESOLUTION & MASTER DRIVER STATE
        # Priority: DROWSY_NODDING > DROWSY > PHONE > DISTRACTED > YAWNING > FATIGUE > ALERT
        # -------------------------------------------------------------
        if not tracking_reliable and face_data.get("tracking_state") == "FACE_TRACKING_LOST":
            master_state = "FACE_TRACKING_LOST"
        elif drowsy_state == "DROWSY_NODDING" or (drowsy_score >= 90 and is_nodding):
            master_state = "DROWSY_NODDING"
        elif drowsy_state in ["DROWSY", "CRITICAL_DROWSY"] or drowsy_score >= 70:
            master_state = "DROWSY"
        elif phone_usage:
            master_state = "PHONE_USAGE"
        elif (is_distracted and distract_score >= 60) or (distract_state == "DISTRACTED" and distract_score >= 60):
            master_state = "DISTRACTED"

        elif is_yawning:
            master_state = "YAWNING"
        elif fatigue_state in ["FATIGUED", "HIGH_FATIGUE"] or fatigue_score >= 50:
            master_state = "FATIGUED"
        elif drowsy_state == "DROWSY_CANDIDATE" or drowsy_score >= 50:
            master_state = "DROWSY_CANDIDATE"
        else:
            master_state = "ALERT"

        self.master_driver_state = master_state

        # -------------------------------------------------------------
        # 3. STATE-TRANSITION ALERT GENERATION & DEDUPLICATION
        # -------------------------------------------------------------
        alerts = []
        new_events = []

        alert_conditions = {
            "CRITICAL_DROWSY_NODDING": (
                master_state == "DROWSY_NODDING" or (drowsy_score >= 90 and is_nodding),
                "🔴 DROWSY NODDING DETECTED",
                drowsy_score
            ),
            "DROWSINESS_DETECTED": (
                drowsy_score >= 70 and master_state != "DROWSY_NODDING",
                "🔴 DROWSINESS DETECTED",
                drowsy_score
            ),
            "PHONE_USAGE_DETECTED": (
                phone_usage,
                "🔴 PHONE USAGE DETECTED",
                100
            ),
            "DRIVER_DISTRACTED": (
                is_distracted and master_state not in ["DROWSY_NODDING", "DROWSY"],
                "🟠 DRIVER DISTRACTED",
                distract_score
            ),
            "YAWNING_DETECTED": (
                is_yawning,
                "🟠 YAWNING DETECTED",
                int(yawn_data.get("yawn_probability", 1.0) * 100)
            ),
            "FATIGUE_DETECTED": (
                fatigue_score >= 55 and not is_yawning and drowsy_score < 70,
                "🟡 FATIGUE DETECTED",
                fatigue_score
            ),
            "LOW_DRIVER_ATTENTION": (
                attention_score <= 40,
                "⚠️ LOW DRIVER ATTENTION",
                attention_score
            )
        }

        for event_key, (is_triggered, alert_label, event_val) in alert_conditions.items():
            if is_triggered:
                alerts.append(alert_label)
                if not self.active_alert_states[event_key]:
                    self.active_alert_states[event_key] = True
                    new_events.append({
                        "timestamp": timestamp_sec,
                        "event": event_key,
                        "value": event_val
                    })
            else:
                if self.active_alert_states[event_key]:
                    self.active_alert_states[event_key] = False
                    new_events.append({
                        "timestamp": timestamp_sec,
                        "event": f"{event_key}_RESOLVED",
                        "value": 0
                    })

        # -------------------------------------------------------------
        # 4. STORE COMPREHENSIVE RECORD INTO TEMPORAL BUFFER
        # -------------------------------------------------------------
        frame_record = {
            "timestamp_sec": timestamp_sec,
            "master_driver_state": master_state,
            "attention_score": attention_score,
            
            # Sub-scores
            "drowsiness_score": drowsy_score,
            "drowsiness_state": drowsy_state,
            "distraction_score": distract_score,
            "distraction_state": distract_state,
            "fatigue_score": fatigue_score,
            "fatigue_state": fatigue_state,
            
            # Eye & EAR
            "ear": eye_data.get("ear", 0.28),
            "ear_left": eye_data.get("ear_left", 0.28),
            "ear_right": eye_data.get("ear_right", 0.28),
            "normalized_ear": eye_data.get("normalized_ear", 1.0),
            "ear_baseline": eye_data.get("ear_baseline", 0.28),
            "ear_threshold": eye_data.get("ear_threshold", 0.20),
            "eye_state": eye_data.get("eye_state", "OPEN"),
            "eye_closed_duration": eye_data.get("eye_closed_duration", 0.0),
            "ear_velocity": eye_data.get("ear_velocity", 0.0),
            "blink_count": eye_data.get("blink_count", 0),
            "blink_rate_per_min": eye_data.get("blink_rate_per_min", 0),
            
            # PERCLOS
            "perclos_short": fatigue_data.get("perclos_short", 0.0),
            "perclos_medium": fatigue_data.get("perclos_medium", 0.0),
            "perclos_long": fatigue_data.get("perclos_long", 0.0),
            
            # Yawn & MAR
            "mar": yawn_data.get("mar", 0.15),
            "mar_velocity": yawn_data.get("mar_velocity", 0.0),
            "yawn_state": yawn_data.get("yawn_state", "NO_YAWN"),
            "is_yawning": is_yawning,
            "yawn_probability": yawn_data.get("yawn_probability", 0.0),
            "yawn_count": yawn_data.get("yawn_count", 0),
            
            # Head Pose & Nodding
            "raw_yaw": head_pose_data.get("raw_yaw", 0.0),
            "raw_pitch": head_pose_data.get("raw_pitch", 0.0),
            "raw_roll": head_pose_data.get("raw_roll", 0.0),
            "neutral_yaw": head_pose_data.get("neutral_yaw", 0.0),
            "neutral_pitch": head_pose_data.get("neutral_pitch", 0.0),
            "neutral_roll": head_pose_data.get("neutral_roll", 0.0),
            "yaw": head_pose_data.get("relative_yaw", 0.0),
            "pitch": head_pose_data.get("relative_pitch", 0.0),
            "roll": head_pose_data.get("relative_roll", 0.0),
            "head_pitch_velocity": head_pose_data.get("pitch_velocity", 0.0),
            "head_pose_state": head_pose_data.get("head_pose_state", "FORWARD"),
            "is_nodding": is_nodding,
            
            # Gaze
            "gaze_state": gaze_data.get("gaze_state", "UNKNOWN"),
            "gaze_confidence": gaze_data.get("gaze_confidence", 0.0),
            "gaze_dx": gaze_data.get("gaze_dx", 0.0),
            "gaze_dy": gaze_data.get("gaze_dy", 0.0),
            
            # Phone & Tracking
            "phone_usage": phone_usage,
            "phone_confidence": phone_data.get("phone_confidence", 0.0),
            "tracking_quality": face_data.get("tracking_quality", 1.0),
            "tracking_state": face_data.get("tracking_state", "FACE_TRACKED"),
            
            # Alerts
            "alerts": alerts,
            "evidence_breakdown": drowsiness_data.get("evidence_breakdown", {})
        }

        self.temporal_buffer.append(frame_record)

        return frame_record, new_events
