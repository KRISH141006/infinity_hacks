# distraction_engine.py - State-Based Temporal Distraction Engine with Gaze Priority & Hysteresis
import numpy as np

class DistractionEngine:
    def __init__(
        self,
        trigger_threshold=70,
        release_threshold=42,
        min_distract_duration=1.8
    ):
        self.trigger_threshold = trigger_threshold
        self.release_threshold = release_threshold
        self.min_distract_duration = min_distract_duration

        # State Machine: ATTENTIVE, POSSIBLE_DISTRACTION, DISTRACTED, RECOVERING
        self.state = "ATTENTIVE"
        self.distraction_score = 0
        self.distraction_duration = 0.0
        self.is_distraction_active = False  # Hysteresis flag

    def process(
        self,
        gaze_data,
        head_pose_data,
        eye_data,
        drowsiness_data,
        face_track_data,
        dt
    ):
        gaze_state = gaze_data.get("gaze_state", "UNKNOWN")
        gaze_conf = gaze_data.get("gaze_confidence", 0.0)
        is_gaze_away = gaze_data.get("is_gaze_away", False)

        rel_yaw = head_pose_data.get("relative_yaw", 0.0)
        rel_pitch = head_pose_data.get("relative_pitch", 0.0)
        head_pose_state = head_pose_data.get("head_pose_state", "FORWARD")
        is_extreme_pose = head_pose_data.get("is_extreme_pose", False)

        is_eye_closed = eye_data.get("is_eye_closed", False)
        eye_closed_dur = eye_data.get("eye_closed_duration", 0.0)

        drowsy_state = drowsiness_data.get("drowsiness_state", "ALERT")
        drowsy_score = drowsiness_data.get("drowsiness_score", 0)
        is_nodding = drowsiness_data.get("is_nodding_off", False)

        is_face_tracked = face_track_data.get("is_confident", True)

        # -------------------------------------------------------------
        # 1. DISTRACTION ELIGIBILITY GATING
        # -------------------------------------------------------------
        # Distraction is completely INELIGIBLE if driver is drowsy, sleeping, nodding, eyes closed, or tracking lost
        is_drowsy_dominating = (
            drowsy_state in ["DROWSY", "CRITICAL_DROWSY", "DROWSY_NODDING", "DROWSY_CANDIDATE"] or
            drowsy_score >= 60 or
            is_nodding or
            is_eye_closed or
            eye_closed_dur > 0.25
        )

        is_gaze_unreliable = (gaze_state == "UNKNOWN" or gaze_conf < 0.20 or not is_face_tracked)

        if is_drowsy_dominating or is_gaze_unreliable:
            # Drowsiness / Sleep / Low confidence suppresses distraction!
            self.distraction_duration = max(0.0, self.distraction_duration - 2.5 * dt)
            self.is_distraction_active = False
            self.state = "ATTENTIVE"
            self.distraction_score = int(max(0, min(15, self.distraction_score * 0.80)))

            return {
                "distraction_score": self.distraction_score,
                "distraction_state": self.state,
                "distraction_duration": round(self.distraction_duration, 2),
                "is_distracted": False,
                "distraction_eligible": False
            }


        # -------------------------------------------------------------
        # 2. GAZE & HEAD POSE AGREEMENT EVALUATION
        # Priority: GAZE > HEAD POSE
        # -------------------------------------------------------------
        gaze_head_agree = (
            (gaze_state == "LEFT" and rel_yaw < -18.0) or
            (gaze_state == "RIGHT" and rel_yaw > 18.0) or
            (gaze_state == "UP" and rel_pitch > 15.0) or
            (gaze_state == "DOWN" and rel_pitch < -15.0)
        )

        if gaze_state == "CENTER" and not is_extreme_pose:
            # Gaze CENTER on road overrides moderate head rotation!
            is_looking_away = False
        elif is_gaze_away and gaze_conf >= 0.30:
            is_looking_away = True
        elif is_extreme_pose:
            is_looking_away = True
        elif head_pose_state != "FORWARD" and gaze_state != "CENTER":
            is_looking_away = True
        else:
            is_looking_away = False

        # Accumulate or decay looking-away duration
        if is_looking_away:
            self.distraction_duration += dt
        else:
            self.distraction_duration = max(0.0, self.distraction_duration - 1.8 * dt)

        # -------------------------------------------------------------
        # 3. GRADUAL DISTRACTION SCORING (0 to 100)
        # -------------------------------------------------------------
        if self.distraction_duration <= 0.35:
            # Brief natural scan / glance
            base_score = 10 if (is_gaze_away or head_pose_state != "FORWARD") else 0
            curr_state = "ATTENTIVE"
        elif self.distraction_duration < 1.0:
            base_score = 30 + (25 if gaze_head_agree else 10)
            curr_state = "POSSIBLE_DISTRACTION"
        elif self.distraction_duration < self.min_distract_duration:
            base_score = 55 + (25 if gaze_head_agree else 15)
            curr_state = "POSSIBLE_DISTRACTION"
        else:
            # Sustained distraction >= 1.8 seconds
            dur_ramp = min(1.0, (self.distraction_duration - self.min_distract_duration) / 1.5)
            base_score = 80 + (15 if gaze_head_agree else 5) + int(dur_ramp * 5)
            curr_state = "DISTRACTED"

        target_score = int(min(100, max(0, base_score)))
        self.distraction_score = target_score

        # -------------------------------------------------------------
        # 4. HYSTERESIS STATE TRANSITION
        # -------------------------------------------------------------
        if not self.is_distraction_active:
            if self.distraction_score >= self.trigger_threshold or self.distraction_duration >= self.min_distract_duration:
                self.is_distraction_active = True
                self.state = "DISTRACTED"
            else:
                self.state = curr_state
        else:
            if self.distraction_score <= self.release_threshold and self.distraction_duration < 0.5:
                self.is_distraction_active = False
                self.state = "RECOVERING" if curr_state == "ATTENTIVE" else curr_state
            else:
                self.state = "DISTRACTED"

        return {
            "distraction_score": self.distraction_score,
            "distraction_state": self.state,
            "distraction_duration": round(self.distraction_duration, 2),
            "is_distracted": self.is_distraction_active,
            "distraction_eligible": True
        }
