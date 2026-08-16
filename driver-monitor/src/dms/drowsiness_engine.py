# drowsiness_engine.py - Multi-Signal Drowsiness & Microsleep Evidence Fusion Engine
import numpy as np

class DrowsinessEngine:
    def __init__(self, use_yolo=True):
        self.use_yolo = use_yolo

        # State Machine: ALERT, FATIGUED, DROWSY_CANDIDATE, DROWSY, CRITICAL_DROWSY, DROWSY_NODDING, RECOVERING
        self.state = "ALERT"
        self.drowsiness_score = 0
        self.microsleep_count = 0
        self.recovery_countdown = 0.0

    def process(
        self,
        eye_data,
        head_pose_data,
        fatigue_data,
        yawn_data,
        yolo_drowsy_prob,
        dt
    ):
        eye_closed_dur = eye_data.get("eye_closed_duration", 0.0)
        is_eye_closed = eye_data.get("is_eye_closed", False)
        norm_ear = eye_data.get("normalized_ear", 1.0)
        long_blinks = eye_data.get("long_blink_count", 0)

        is_nodding_off = head_pose_data.get("is_nodding_off", False)
        rel_pitch = head_pose_data.get("relative_pitch", 0.0)
        p_vel = head_pose_data.get("pitch_velocity", 0.0)

        perclos_short = fatigue_data.get("perclos_short", 0.0)
        perclos_med = fatigue_data.get("perclos_medium", 0.0)
        perclos_long = fatigue_data.get("perclos_long", 0.0)
        fatigue_score = fatigue_data.get("fatigue_score", 0)

        is_yawning = yawn_data.get("is_yawning", False)
        yawn_prob = yawn_data.get("yawn_probability", 0.0)

        # -------------------------------------------------------------
        # 1. MICROSLEEP & IMMEDIATE EYE CLOSURE EVIDENCE
        # -------------------------------------------------------------
        # Direct rapid response: does NOT wait for PERCLOS_long!
        microsleep_points = 0.0
        is_microsleep = False

        if eye_closed_dur >= 2.0:
            is_microsleep = True
            microsleep_points = 90.0 + min(10.0, (eye_closed_dur - 2.0) * 10.0)
        elif eye_closed_dur >= 1.5:
            is_microsleep = True
            microsleep_points = 80.0 + (eye_closed_dur - 1.5) * 20.0
        elif eye_closed_dur >= 0.85:
            # Drowsy candidate closure ramp
            microsleep_points = 55.0 + ((eye_closed_dur - 0.85) / 0.65) * 25.0
        elif eye_closed_dur >= 0.40:
            # Long sluggish blink
            microsleep_points = 25.0 + (eye_closed_dur - 0.40) * 40.0

        # -------------------------------------------------------------
        # 2. NODDING-OFF SYNERGY EVIDENCE
        # -------------------------------------------------------------
        nodding_points = 0.0
        if is_nodding_off or (rel_pitch < -12.0 and is_eye_closed):
            nodding_points = 35.0
            if is_microsleep or eye_closed_dur >= 1.2:
                nodding_points = 50.0  # Critical amplifier

        # -------------------------------------------------------------
        # 3. MEDIUM/LONG TERM FATIGUE & PERCLOS EVIDENCE
        # -------------------------------------------------------------
        perclos_points = min(35.0, perclos_med * 100.0 * 0.8 + perclos_long * 100.0 * 0.7)
        yawn_points = min(20.0, yawn_prob * 18.0) if is_yawning else 0.0
        sluggish_blink_points = min(15.0, long_blinks * 2.5)

        # -------------------------------------------------------------
        # 4. PRETRAINED YOLO MODEL EVIDENCE
        # -------------------------------------------------------------
        model_points = 0.0
        if self.use_yolo and yolo_drowsy_prob > 0.50:
            model_points = min(20.0, (yolo_drowsy_prob - 0.50) * 40.0)

        # -------------------------------------------------------------
        # 5. MULTI-SIGNAL EVIDENCE SYNTHESIS (DYNAMIC STATE WEIGHTING)
        # -------------------------------------------------------------
        # Short window dominates during immediate microsleep / nodding
        if is_microsleep and (is_nodding_off or rel_pitch < -10.0):
            total_evidence = max(95.0, microsleep_points + nodding_points * 0.3)
            self.state = "DROWSY_NODDING"
        elif is_microsleep:
            total_evidence = max(85.0, microsleep_points + perclos_points * 0.2 + model_points * 0.3)
            self.state = "DROWSY"
        elif eye_closed_dur >= 0.85:
            total_evidence = max(60.0, microsleep_points + perclos_points * 0.4 + model_points * 0.4)
            self.state = "DROWSY_CANDIDATE"
        else:
            # Baseline wakefulness / cumulative fatigue
            cumulative_evidence = (
                (perclos_points * 0.40) +
                (fatigue_score * 0.30) +
                (yawn_points * 0.15) +
                (sluggish_blink_points * 0.10) +
                (model_points * 0.25)
            )
            total_evidence = cumulative_evidence

            if self.state in ["DROWSY", "DROWSY_NODDING", "DROWSY_CANDIDATE"]:
                self.state = "RECOVERING"
                self.recovery_countdown = 2.0  # 2s smooth recovery
            elif self.state == "RECOVERING":
                self.recovery_countdown = max(0.0, self.recovery_countdown - dt)
                if self.recovery_countdown <= 0:
                    self.state = "FATIGUED" if fatigue_score >= 45 else "ALERT"
            else:
                self.state = "FATIGUED" if fatigue_score >= 45 else "ALERT"

        target_score = int(min(100, max(0, total_evidence)))

        # Score smoothing
        if self.state == "RECOVERING":
            self.drowsiness_score = max(target_score, int(self.drowsiness_score * 0.90))
        else:
            self.drowsiness_score = target_score

        return {
            "drowsiness_score": self.drowsiness_score,
            "drowsiness_state": self.state,
            "is_microsleep": is_microsleep,
            "is_nodding_off": is_nodding_off or self.state == "DROWSY_NODDING",
            "evidence_breakdown": {
                "microsleep_points": round(microsleep_points, 1),
                "nodding_points": round(nodding_points, 1),
                "perclos_points": round(perclos_points, 1),
                "yawn_points": round(yawn_points, 1),
                "model_points": round(model_points, 1)
            }
        }
