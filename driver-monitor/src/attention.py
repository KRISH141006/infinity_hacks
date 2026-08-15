import numpy as np

def angle_difference(a, b):
    # Returns difference wrapped in [-180, 180] range to prevent Euler wrapping issues
    return (a - b + 180) % 360 - 180

class AttentionManager:
    def __init__(self, config=None):
        # Configuration parameters for hackathon customization
        self.config = config or {
            "weight_drowsiness": 0.40,
            "weight_distraction": 0.35,
            "weight_phone": 0.25,
            "yaw_threshold": 25.0,
            "pitch_threshold": 20.0,
            "roll_threshold": 25.0,
            "extreme_yaw_threshold": 40.0,
            "extreme_pitch_threshold": 35.0,
            "drowsiness_alert_threshold": 60,
            "distraction_alert_threshold": 50,
            "attention_critical_threshold": 40,
            "distraction_time_threshold": 2.0,  # 2.0 seconds sustained to trigger major alert
            "persistence_frames": 25  # ~1.0 second at 25 FPS
        }

        # Temporal counters for persistence alerts
        self.drowsy_streak = 0
        self.distracted_streak = 0
        self.phone_streak = 0
        self.low_attention_streak = 0
        
        # State trackers
        self.distraction_duration = 0.0
        
        # Calibration state
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.neutral_roll = 0.0
        self.is_calibrated = False
        self.calibration_frames = []
        self.calibration_limit = 30  # Number of frames to calibrate

        # Gaze smoothing queue
        self.gaze_history = []
        self.gaze_history_limit = 15  # Smooth gaze over last 15 frames

        # Alert State Transition Trackers (for deduplication)
        self.active_alerts = {
            "drowsiness": False,
            "distraction": False,
            "looking_away": False,
            "phone": False,
            "low_attention": False
        }

    def calibrate(self, yaw, pitch, roll):
        if self.is_calibrated:
            return
        
        self.calibration_frames.append((yaw, pitch, roll))
        if len(self.calibration_frames) >= self.calibration_limit:
            self.neutral_yaw = float(np.median([f[0] for f in self.calibration_frames]))
            self.neutral_pitch = float(np.median([f[1] for f in self.calibration_frames]))
            self.neutral_roll = float(np.median([f[2] for f in self.calibration_frames]))
            self.is_calibrated = True
            print(f"\n--- CALIBRATED BASELINE ---")
            print(f"Neutral Yaw: {self.neutral_yaw:.2f}")
            print(f"Neutral Pitch: {self.neutral_pitch:.2f}")
            print(f"Neutral Roll: {self.neutral_roll:.2f}\n")

    def is_looking_away(self, gaze_state, relative_yaw, relative_pitch, relative_roll):
        # Determine if head is near neutral or has extreme deviation
        head_near_neutral = (
            abs(relative_yaw) < self.config["yaw_threshold"] and
            abs(relative_pitch) < self.config["pitch_threshold"]
        )
        extreme_head_deviation = (
            abs(relative_yaw) > self.config["extreme_yaw_threshold"] or
            abs(relative_pitch) > self.config["extreme_pitch_threshold"]
        )
        
        # Rule 1: GAZE CENTER overrides moderate head deviations
        if gaze_state == "CENTER" and not extreme_head_deviation:
            return False, "ATTENTIVE"

        # Rule 2: Gaze and Head agree (looks away left/right/up/down)
        if gaze_state == "LEFT" and relative_yaw < -self.config["yaw_threshold"]:
            return True, "STRONG_DISTRACTION"
        if gaze_state == "RIGHT" and relative_yaw > self.config["yaw_threshold"]:
            return True, "STRONG_DISTRACTION"
        if gaze_state == "UP" and relative_pitch > self.config["pitch_threshold"]:
            return True, "STRONG_DISTRACTION"
        if gaze_state == "DOWN" and relative_pitch < -self.config["pitch_threshold"]:
            return True, "STRONG_DISTRACTION"

        # Rule 3: Gaze away but head normal (eyes looking aside)
        if gaze_state != "CENTER" and head_near_neutral:
            return True, "POSSIBLE_DISTRACTION"

        # Rule 4: Head looking away beyond thresholds
        if not head_near_neutral:
            if extreme_head_deviation:
                return True, "STRONG_DISTRACTION"
            return True, "HEAD_LOOKING_AWAY"

        return False, "ATTENTIVE"

    def compute_scores(self, drowsiness_data, head_pose_data, raw_gaze_state, phone_usage_data, fps):
        yaw, pitch, roll, _ = head_pose_data
        phone_detected, _ = phone_usage_data

        # 1. Smooth Gaze State
        self.gaze_history.append(raw_gaze_state)
        if len(self.gaze_history) > self.gaze_history_limit:
            self.gaze_history.pop(0)
        
        # Majority voting for gaze
        from collections import Counter
        gaze_state = Counter(self.gaze_history).most_common(1)[0][0]

        # 2. Handle Calibration & Relative Angle Calculations
        if not self.is_calibrated:
            self.calibrate(yaw, pitch, roll)
            relative_yaw = 0.0
            relative_pitch = 0.0
            relative_roll = 0.0
            pose_state = "CALIBRATING"
        else:
            relative_yaw = angle_difference(yaw, self.neutral_yaw)
            relative_pitch = angle_difference(pitch, self.neutral_pitch)
            relative_roll = angle_difference(roll, self.neutral_roll)
            
            # Classify Pose State relative to Calibrated Position
            if abs(relative_yaw) < self.config["yaw_threshold"] and abs(relative_pitch) < self.config["pitch_threshold"]:
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

        # 3. Determine Distraction State using Gaze/Head Pose Fusion
        looking_away, level = self.is_looking_away(gaze_state, relative_yaw, relative_pitch, relative_roll)
        
        if looking_away:
            self.distraction_duration += 1.0 / fps
        else:
            # Decay recovery
            self.distraction_duration = max(0.0, self.distraction_duration - 1.5 / fps)

        # 4. Gradual Distraction Score (0-100)
        score = 0
        gaze_score = 0
        head_score = 0
        duration_score = 0

        if gaze_state != "CENTER":
            gaze_score = 45
            score += gaze_score

        # Check for significant head deviation
        head_near_neutral = (
            abs(relative_yaw) < self.config["yaw_threshold"] and
            abs(relative_pitch) < self.config["pitch_threshold"]
        )
        if not head_near_neutral:
            head_score = 25
            score += head_score

        # Check if gaze and head pose agree
        gaze_head_agree = (
            (gaze_state == "LEFT" and relative_yaw < -self.config["yaw_threshold"]) or
            (gaze_state == "RIGHT" and relative_yaw > self.config["yaw_threshold"]) or
            (gaze_state == "UP" and relative_pitch > self.config["pitch_threshold"]) or
            (gaze_state == "DOWN" and relative_pitch < -self.config["pitch_threshold"])
        )
        if gaze_head_agree:
            score += 15

        # Persistence score boost
        if self.distraction_duration >= self.config["distraction_time_threshold"]:
            duration_score = 15
            score += duration_score

        # IMPORTANT OVERRIDE RULE: Gaze CENTER overrides moderate head deviations
        extreme_head_deviation = (
            abs(relative_yaw) > self.config["extreme_yaw_threshold"] or
            abs(relative_pitch) > self.config["extreme_pitch_threshold"]
        )
        if gaze_state == "CENTER" and not extreme_head_deviation:
            score = min(score, 15)

        # Phone usage penalty
        if phone_detected:
            score = min(100, score + 35)

        distraction_score = min(100, max(0, int(score)))

        # 5. Driver Attention Score (0-100)
        drowsiness_score = drowsiness_data["drowsiness_score"]
        phone_penalty = 100 if phone_detected else 0

        risk_score = (
            (drowsiness_score * self.config["weight_drowsiness"]) +
            (distraction_score * self.config["weight_distraction"]) +
            (phone_penalty * self.config["weight_phone"])
        )

        attention_score = int(100.0 - risk_score)
        attention_score = max(0, min(100, attention_score))

        # 6. Alert Deduplication Log System
        alerts = []

        # Drowsiness alert trigger
        if drowsiness_score >= self.config["drowsiness_alert_threshold"]:
            self.drowsy_streak += 1
        else:
            self.drowsy_streak = max(0, self.drowsy_streak - 2)

        if self.drowsy_streak >= self.config["persistence_frames"]:
            self.active_alerts["drowsiness"] = True
            alerts.append("🔴 DROWSINESS DETECTED")
        else:
            self.active_alerts["drowsiness"] = False

        # Distraction alert trigger
        if distraction_score >= self.config["distraction_alert_threshold"]:
            self.distracted_streak += 1
        else:
            self.distracted_streak = max(0, self.distracted_streak - 2)

        if self.distracted_streak >= self.config["persistence_frames"]:
            self.active_alerts["distraction"] = True
            alerts.append("🟠 DRIVER DISTRACTED")
        else:
            self.active_alerts["distraction"] = False

        # Looking away alert trigger
        if self.distraction_duration >= self.config["distraction_time_threshold"]:
            self.active_alerts["looking_away"] = True
            alerts.append("⚠️ DRIVER LOOKING AWAY")
        else:
            self.active_alerts["looking_away"] = False

        # Phone usage alert trigger
        if phone_detected:
            self.phone_streak += 1
        else:
            self.phone_streak = max(0, self.phone_streak - 2)

        if self.phone_streak >= self.config["persistence_frames"]:
            self.active_alerts["phone"] = True
            alerts.append("🔴 PHONE USAGE DETECTED")
        else:
            self.active_alerts["phone"] = False

        # Low attention alert trigger
        if attention_score <= self.config["attention_critical_threshold"]:
            self.low_attention_streak += 1
        else:
            self.low_attention_streak = max(0, self.low_attention_streak - 2)

        if self.low_attention_streak >= self.config["persistence_frames"]:
            self.active_alerts["low_attention"] = True
            alerts.append("⚠️ LOW DRIVER ATTENTION")
        else:
            self.active_alerts["low_attention"] = False

        # Return full telemetry including debug outputs
        return {
            "relative_yaw": relative_yaw,
            "relative_pitch": relative_pitch,
            "relative_roll": relative_roll,
            "head_pose_state": pose_state,
            "distraction_score": distraction_score,
            "attention_score": attention_score,
            "distraction_duration": self.distraction_duration,
            "alerts": alerts,
            
            # Temporary/Internal Debug Fields
            "raw_yaw": yaw,
            "raw_pitch": pitch,
            "raw_roll": roll,
            "neutral_yaw": self.neutral_yaw,
            "neutral_pitch": self.neutral_pitch,
            "neutral_roll": self.neutral_roll,
            "gaze_state": gaze_state,
            "gaze_away_score": gaze_score,
            "head_away_score": head_score,
            "duration_score": duration_score
        }
