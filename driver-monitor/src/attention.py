import numpy as np

def angle_difference(a, b):
    """Returns angular difference wrapped to [-180, 180] range."""
    return ((a - b + 180.0) % 360.0) - 180.0

class AttentionManager:
    def __init__(self, config=None):
        self.config = config or {
            "weight_drowsiness": 0.45,
            "weight_distraction": 0.35,
            "weight_phone": 0.20,
            "yaw_threshold": 22.0,
            "pitch_threshold": 18.0,
            "extreme_yaw_threshold": 38.0,
            "extreme_pitch_threshold": 32.0,
            "distraction_time_threshold": 1.8, # ~1.8 seconds sustained looking away
            "drowsiness_alert_threshold": 70,
            "distraction_alert_threshold": 65,
            "attention_critical_threshold": 40,
        }

        # Calibration state
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.neutral_roll = 0.0
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_limit = 35

        # Distraction temporal state
        self.distraction_duration = 0.0
        self.distraction_state = "ATTENTIVE"  # ATTENTIVE, GLANCE, DISTRACTED_CANDIDATE, DISTRACTED
        self.distraction_score = 0

        # Gaze history smoothing
        self.gaze_history = []
        self.gaze_history_limit = 10

        # Alert State Transition Trackers (Deduplication)
        self.active_alerts = {
            "drowsiness": False,
            "drowsy_nodding": False,
            "distraction": False,
            "looking_away": False,
            "phone": False,
            "low_attention": False
        }

    def calibrate(self, yaw, pitch, roll, is_eye_open):
        if self.is_calibrated:
            return
        # Only calibrate on valid upright frames
        if is_eye_open and abs(pitch) < 45.0 and abs(yaw) < 45.0:
            self.calibration_samples.append((yaw, pitch, roll))
            if len(self.calibration_samples) >= self.calibration_limit:
                self.neutral_yaw = float(np.median([s[0] for s in self.calibration_samples]))
                self.neutral_pitch = float(np.median([s[1] for s in self.calibration_samples]))
                self.neutral_roll = float(np.median([s[2] for s in self.calibration_samples]))
                self.is_calibrated = True
                print(f"\n--- DMS BASELINE CALIBRATED ---")
                print(f"Neutral Yaw: {self.neutral_yaw:.2f}° | Neutral Pitch: {self.neutral_pitch:.2f}° | Neutral Roll: {self.neutral_roll:.2f}°\n")

    def compute_scores(
        self,
        drowsy_data,
        head_pose_data,
        gaze_data,
        phone_data,
        fps
    ):
        raw_yaw, raw_pitch, raw_roll, _ = head_pose_data
        raw_gaze_state, gaze_confidence = gaze_data if isinstance(gaze_data, (tuple, list)) else (gaze_data, 1.0)
        phone_detected, phone_conf = phone_data
        dt = 1.0 / max(1.0, float(fps))

        # -------------------------------------------------------------
        # 1. HEAD POSE CALIBRATION & RELATIVE ANGLES
        # -------------------------------------------------------------
        is_eye_open = not drowsy_data.get("is_eye_closed", False)
        if not self.is_calibrated:
            self.calibrate(raw_yaw, raw_pitch, raw_roll, is_eye_open)
            relative_yaw = 0.0
            relative_pitch = 0.0
            relative_roll = 0.0
            pose_state = "CALIBRATING"
        else:
            relative_yaw = angle_difference(raw_yaw, self.neutral_yaw)
            relative_pitch = angle_difference(raw_pitch, self.neutral_pitch)
            relative_roll = angle_difference(raw_roll, self.neutral_roll)

            # Classify Head Orientation
            if abs(relative_yaw) <= self.config["yaw_threshold"] and abs(relative_pitch) <= self.config["pitch_threshold"]:
                pose_state = "FORWARD"
            elif relative_yaw < -self.config["yaw_threshold"]:
                pose_state = "LEFT"
            elif relative_yaw > self.config["yaw_threshold"]:
                pose_state = "RIGHT"
            elif relative_pitch < -self.config["pitch_threshold"]:
                pose_state = "DOWN"
            elif relative_pitch > self.config["pitch_threshold"]:
                pose_state = "UP"
            else:
                pose_state = "FORWARD"

        # -------------------------------------------------------------
        # 2. GAZE SMOOTHING & RELIABILITY GATING
        # -------------------------------------------------------------
        if not is_eye_open or drowsy_data.get("tracking_lost", False):
            smoothed_gaze = "UNKNOWN"
            gaze_confidence = 0.0
        else:
            self.gaze_history.append(raw_gaze_state)
            if len(self.gaze_history) > self.gaze_history_limit:
                self.gaze_history.pop(0)
            from collections import Counter
            valid_gazes = [g for g in self.gaze_history if g != "UNKNOWN"]
            if valid_gazes:
                smoothed_gaze = Counter(valid_gazes).most_common(1)[0][0]
            else:
                smoothed_gaze = raw_gaze_state

        # -------------------------------------------------------------
        # 3. DISTRACTION ENGINE & ELIGIBILITY (STATE-BASED)
        # -------------------------------------------------------------
        drowsiness_state = drowsy_data.get("drowsiness_state", "ALERT")
        drowsiness_score = drowsy_data.get("drowsiness_score", 0)
        is_nodding = drowsy_data.get("is_nodding", False)

        # Distraction is INELIGIBLE when drowsiness/nodding is active or eyes are closed
        is_drowsy_active = (
            drowsiness_state in ["DROWSY", "DROWSY_NODDING", "DROWSY_CANDIDATE"] or
            drowsiness_score >= 60 or
            not is_eye_open
        )

        if is_drowsy_active:
            # Drowsiness suppresses distraction; head drop / closed eyes is NOT distraction!
            self.distraction_duration = max(0.0, self.distraction_duration - 2.5 * dt)
            self.distraction_state = "ATTENTIVE"
            distraction_score = int(min(15, self.distraction_score * 0.85))
        else:
            # Distraction evaluation (Eyes are confirmed open)
            gaze_away = smoothed_gaze not in ["CENTER", "UNKNOWN"]
            head_away = pose_state not in ["FORWARD", "CALIBRATING"]
            extreme_head_away = (
                abs(relative_yaw) > self.config["extreme_yaw_threshold"] or
                abs(relative_pitch) > self.config["extreme_pitch_threshold"]
            )

            # Gaze agreement with head
            gaze_head_agree = (
                (smoothed_gaze == "LEFT" and relative_yaw < -self.config["yaw_threshold"]) or
                (smoothed_gaze == "RIGHT" and relative_yaw > self.config["yaw_threshold"]) or
                (smoothed_gaze == "UP" and relative_pitch > self.config["pitch_threshold"]) or
                (smoothed_gaze == "DOWN" and relative_pitch < -self.config["pitch_threshold"])
            )

            # Looking away trigger condition
            # Gaze is primary; head pose is supporting
            if smoothed_gaze == "UNKNOWN" or drowsy_data.get("tracking_lost", False):
                is_looking_away = False
            elif smoothed_gaze == "CENTER" and not extreme_head_away:
                # Gaze Center overrides moderate head deviation (e.g. driver looking ahead while head turned)
                is_looking_away = False
            elif (gaze_away and gaze_confidence >= 0.25) or extreme_head_away:
                is_looking_away = True
            elif head_away and is_eye_open:
                is_looking_away = True
            else:
                is_looking_away = False

            if is_looking_away:
                self.distraction_duration += dt
            else:
                self.distraction_duration = max(0.0, self.distraction_duration - 2.0 * dt)


            # Calculate Gradual Distraction Score (0-100)
            if self.distraction_duration <= 0.35:
                # Brief glance - normal scanning behavior
                self.distraction_state = "ATTENTIVE"
                base_score = 10 if (gaze_away or head_away) else 0
            elif self.distraction_duration < 1.0:
                self.distraction_state = "GLANCE"
                base_score = 30 + (20 if gaze_head_agree else 0)
            elif self.distraction_duration < self.config["distraction_time_threshold"]:
                self.distraction_state = "DISTRACTED_CANDIDATE"
                base_score = 55 + (20 if gaze_head_agree else 0)
            else:
                # Sustained distraction >= 1.8 seconds
                self.distraction_state = "DISTRACTED"
                duration_factor = min(1.0, (self.distraction_duration - 1.8) / 1.5)
                base_score = 80 + (15 if gaze_head_agree else 5) + int(duration_factor * 5)

            distraction_score = int(min(100, max(0, base_score)))

        self.distraction_score = distraction_score

        # -------------------------------------------------------------
        # 4. DRIVER ATTENTION SCORE (0-100)
        # -------------------------------------------------------------
        phone_penalty = 100.0 if phone_detected else 0.0

        risk = (
            (drowsiness_score * self.config["weight_drowsiness"]) +
            (distraction_score * self.config["weight_distraction"]) +
            (phone_penalty * self.config["weight_phone"])
        )
        attention_score = int(max(0, min(100, 100.0 - risk)))

        # -------------------------------------------------------------
        # 5. STATE-PRIORITIZED ALERT GENERATION (DEDUPLICATED)
        # -------------------------------------------------------------
        alerts = []

        # Priority 1: Drowsy Nodding Off / Critical Drowsiness
        if drowsiness_state == "DROWSY_NODDING" or (drowsiness_score >= 85 and is_nodding):
            self.active_alerts["drowsy_nodding"] = True
            alerts.append("🔴 CRITICAL: DROWSY NODDING OFF")
        elif drowsiness_state in ["DROWSY", "DROWSY_CANDIDATE"] or drowsiness_score >= self.config["drowsiness_alert_threshold"]:
            self.active_alerts["drowsiness"] = True
            alerts.append("🔴 DROWSINESS DETECTED")

        # Priority 2: Phone Usage (independent)
        if phone_detected:
            self.active_alerts["phone"] = True
            alerts.append("📱 PHONE USAGE DETECTED")

        # Priority 3: Distraction (Only if Drowsiness is NOT dominating)
        if not is_drowsy_active:
            if distraction_score >= self.config["distraction_alert_threshold"] or self.distraction_state == "DISTRACTED":
                self.active_alerts["distraction"] = True
                alerts.append("🟠 DRIVER DISTRACTED")
            elif self.distraction_duration >= self.config["distraction_time_threshold"]:
                self.active_alerts["looking_away"] = True
                alerts.append("⚠️ DRIVER LOOKING AWAY")

        # Priority 4: Low Attention
        if attention_score <= self.config["attention_critical_threshold"]:
            self.active_alerts["low_attention"] = True
            alerts.append("⚠️ LOW DRIVER ATTENTION")

        return {
            "relative_yaw": round(relative_yaw, 1),
            "relative_pitch": round(relative_pitch, 1),
            "relative_roll": round(relative_roll, 1),
            "head_pose_state": pose_state,
            "gaze_state": smoothed_gaze,
            "gaze_confidence": round(gaze_confidence, 2),
            "distraction_score": distraction_score,
            "distraction_state": self.distraction_state,
            "distraction_duration": round(self.distraction_duration, 2),
            "attention_score": attention_score,
            "alerts": alerts,
            
            # Debug telemetry fields
            "raw_yaw": round(raw_yaw, 1),
            "raw_pitch": round(raw_pitch, 1),
            "raw_roll": round(raw_roll, 1),
            "neutral_yaw": round(self.neutral_yaw, 1),
            "neutral_pitch": round(self.neutral_pitch, 1),
            "neutral_roll": round(self.neutral_roll, 1)
        }
