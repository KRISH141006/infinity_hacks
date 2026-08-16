import math

class GazeTracker:
    def __init__(self):
        # MediaPipe iris and eye landmarks
        # Left eye: corners at 33 and 133, iris center at 468
        # Right eye: corners at 362 and 263, iris center at 473
        self.last_state = "UNKNOWN"
        self.last_confidence = 0.0

    def estimate_gaze(self, landmarks, is_eye_open=True, ear_val=None):
        """
        Estimates gaze direction from iris position relative to eye corners.
        If eyes are closed (is_eye_open=False) or landmarks are unreliable,
        returns ('UNKNOWN', 0.0) so no false distraction is triggered while sleeping.
        """
        if not landmarks or len(landmarks) <= 473 or not is_eye_open:
            return "UNKNOWN", 0.0

        try:
            # Left Eye
            left_corner_l = landmarks[33]
            right_corner_l = landmarks[133]
            iris_l = landmarks[468]

            # Right Eye
            left_corner_r = landmarks[362]
            right_corner_r = landmarks[263]
            iris_r = landmarks[473]

            def get_horizontal_ratio(left, right, iris):
                width = right.x - left.x
                if abs(width) < 1e-4:
                    return 0.5
                return (iris.x - left.x) / width

            ratio_l = get_horizontal_ratio(left_corner_l, right_corner_l, iris_l)
            ratio_r = get_horizontal_ratio(left_corner_r, right_corner_r, iris_r)
            avg_ratio = (ratio_l + ratio_r) / 2.0

            # Vertical ratio
            eye_y_avg = (left_corner_l.y + right_corner_l.y + left_corner_r.y + right_corner_r.y) / 4.0
            iris_y_avg = (iris_l.y + iris_r.y) / 2.0
            v_diff = iris_y_avg - eye_y_avg

            # Confidence based on symmetry and ear
            ratio_diff = abs(ratio_l - ratio_r)
            confidence = max(0.2, min(1.0, 1.0 - (ratio_diff * 1.5)))

            state = "CENTER"
            if avg_ratio < 0.43:
                state = "LEFT"
            elif avg_ratio > 0.57:
                state = "RIGHT"
            elif v_diff < -0.018:
                state = "UP"
            elif v_diff > 0.018:
                state = "DOWN"

            self.last_state = state
            self.last_confidence = confidence
            return state, confidence

        except Exception:
            return "UNKNOWN", 0.0
