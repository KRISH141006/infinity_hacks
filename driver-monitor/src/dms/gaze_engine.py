# gaze_engine.py - Iris Gaze Tracking, Dynamic Center Calibration & Confidence Gating
import numpy as np
from collections import deque, Counter

class GazeEngine:
    def __init__(self, history_len=12):
        # Iris & Corner Indices
        # Left eye: corners 33, 133, iris 468
        # Right eye: corners 362, 263, iris 473
        self.gaze_center_x = 0.50
        self.gaze_center_y = 0.0
        self.calib_samples_x = []
        self.calib_samples_y = []
        self.is_calibrated = False

        self.history = deque(maxlen=history_len)
        self.last_state = "CENTER"
        self.last_confidence = 1.0

    def calibrate_center(self, raw_rx, raw_ry, is_stable_upright, is_eye_open):
        """Calibrates driver's natural forward gaze center."""
        if self.is_calibrated:
            return
        if is_stable_upright and is_eye_open and 0.35 < raw_rx < 0.65:
            self.calib_samples_x.append(raw_rx)
            self.calib_samples_y.append(raw_ry)
            if len(self.calib_samples_x) >= 30:
                self.gaze_center_x = float(np.median(self.calib_samples_x))
                self.gaze_center_y = float(np.median(self.calib_samples_y))
                self.is_calibrated = True

    def process(self, landmarks, is_eye_open=True, is_stable_upright=True, tracking_reliable=True):
        """
        Estimates gaze direction and confidence.
        If eyes are closed or tracking is unreliable, returns UNKNOWN with 0 confidence.
        """
        if not landmarks or len(landmarks) <= 473 or not is_eye_open or not tracking_reliable:
            self.history.append("UNKNOWN")
            return {
                "gaze_state": "UNKNOWN",
                "gaze_confidence": 0.0,
                "gaze_dx": 0.0,
                "gaze_dy": 0.0,
                "gaze_center_x": round(self.gaze_center_x, 3),
                "is_gaze_away": False
            }

        try:
            # Left Eye horizontal ratio
            lc_l = landmarks[33]
            rc_l = landmarks[133]
            iris_l = landmarks[468]
            w_l = rc_l.x - lc_l.x
            rx_l = (iris_l.x - lc_l.x) / w_l if abs(w_l) > 1e-4 else 0.5

            # Right Eye horizontal ratio
            lc_r = landmarks[362]
            rc_r = landmarks[263]
            iris_r = landmarks[473]
            w_r = rc_r.x - lc_r.x
            rx_r = (iris_r.x - lc_r.x) / w_r if abs(w_r) > 1e-4 else 0.5

            avg_rx = (rx_l + rx_r) / 2.0

            # Vertical delta
            eye_y = (lc_l.y + rc_l.y + lc_r.y + rc_r.y) / 4.0
            iris_y = (iris_l.y + iris_r.y) / 2.0
            ry = iris_y - eye_y

            # Calibrate center
            self.calibrate_center(avg_rx, ry, is_stable_upright, is_eye_open)

            # Deviations from personalized center
            dx = avg_rx - self.gaze_center_x
            dy = ry - self.gaze_center_y

            # Confidence based on inter-eye agreement
            diff = abs(rx_l - rx_r)
            confidence = max(0.2, min(1.0, 1.0 - (diff * 1.8)))

            # Directional classification
            if dx < -0.075:
                raw_state = "LEFT"
            elif dx > 0.075:
                raw_state = "RIGHT"
            elif dy < -0.018:
                raw_state = "UP"
            elif dy > 0.018:
                raw_state = "DOWN"
            else:
                raw_state = "CENTER"

            self.history.append(raw_state)
            
            # Temporal majority filter
            smoothed_state = Counter(self.history).most_common(1)[0][0]
            self.last_state = smoothed_state
            self.last_confidence = confidence

            return {
                "gaze_state": smoothed_state,
                "gaze_confidence": round(confidence, 2),
                "gaze_dx": round(dx, 3),
                "gaze_dy": round(dy, 3),
                "gaze_center_x": round(self.gaze_center_x, 3),
                "is_gaze_away": smoothed_state not in ["CENTER", "UNKNOWN"]
            }

        except Exception:
            return {
                "gaze_state": "UNKNOWN",
                "gaze_confidence": 0.0,
                "gaze_dx": 0.0,
                "gaze_dy": 0.0,
                "gaze_center_x": round(self.gaze_center_x, 3),
                "is_gaze_away": False
            }
