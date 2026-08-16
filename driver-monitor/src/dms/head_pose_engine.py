# head_pose_engine.py - 3D PnP Head Pose, Dynamic Calibration, Derivatives & Nodding Detector
import cv2
import numpy as np

def angle_diff(a, b):
    """Computes angular difference wrapped in [-180, 180] degrees."""
    return ((a - b + 180.0) % 360.0) - 180.0

class HeadPoseEngine:
    def __init__(
        self,
        yaw_thresh=22.0,
        pitch_thresh=18.0,
        extreme_yaw=38.0,
        extreme_pitch=32.0
    ):
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.extreme_yaw = extreme_yaw
        self.extreme_pitch = extreme_pitch

        # Standard 3D facial feature model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (1)
            (0.0, -330.0, -65.0),        # Chin (152)
            (-225.0, 170.0, -135.0),     # Left eye left corner (33)
            (225.0, 170.0, -135.0),      # Right eye right corner (263)
            (-150.0, -150.0, -125.0),    # Left mouth corner (61)
            (150.0, -150.0, -125.0)      # Right mouth corner (291)
        ], dtype=np.float32)

        self.LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

        # Neutral calibration
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.neutral_roll = 0.0
        self.calib_samples = []
        self.is_calibrated = False

        # State tracking
        self.last_pitch = None
        self.last_pitch_vel = 0.0
        self.pitch_velocity = 0.0       # deg/sec
        self.pitch_acceleration = 0.0   # deg/sec^2
        self.yaw_velocity = 0.0
        self.last_yaw = None

        # Nodding-off temporal detector
        self.nodding_frames = 0
        self.is_nodding_off = False

    def calibrate(self, yaw, pitch, roll, is_eye_open):
        """Calibrates neutral forward baseline head pose."""
        if self.is_calibrated:
            return
        if is_eye_open and abs(pitch) < 40.0 and abs(yaw) < 40.0:
            self.calib_samples.append((yaw, pitch, roll))
            if len(self.calib_samples) >= 30:
                self.neutral_yaw = float(np.median([s[0] for s in self.calib_samples]))
                self.neutral_pitch = float(np.median([s[1] for s in self.calib_samples]))
                self.neutral_roll = float(np.median([s[2] for s in self.calib_samples]))
                self.is_calibrated = True

    def process(self, landmarks, frame_shape, dt, is_eye_open=True, is_eye_closed=False):
        h, w = frame_shape[:2]

        if not landmarks or len(landmarks) < 292:
            return {
                "raw_yaw": 0.0,
                "raw_pitch": 0.0,
                "raw_roll": 0.0,
                "neutral_yaw": round(self.neutral_yaw, 1),
                "neutral_pitch": round(self.neutral_pitch, 1),
                "neutral_roll": round(self.neutral_roll, 1),
                "relative_yaw": 0.0,
                "relative_pitch": 0.0,
                "relative_roll": 0.0,
                "head_pose_state": "UNKNOWN",
                "pitch_velocity": 0.0,
                "pitch_acceleration": 0.0,
                "is_nodding_off": self.is_nodding_off,
                "is_head_forward": False,
                "is_extreme_pose": False
            }

        try:
            # 2D Landmark projection
            image_points = np.array([
                (landmarks[idx].x * w, landmarks[idx].y * h) for idx in self.LANDMARK_INDICES
            ], dtype=np.float32)

            # Camera matrix approximation
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                raise ValueError("PnP solve failed")

            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            raw_pitch, raw_yaw, raw_roll = float(angles[0]), float(angles[1]), float(angles[2])

            # Update calibration
            self.calibrate(raw_yaw, raw_pitch, raw_roll, is_eye_open)

            # Relative angles
            if not self.is_calibrated:
                rel_yaw = 0.0
                rel_pitch = 0.0
                rel_roll = 0.0
                pose_state = "CALIBRATING"
            else:
                rel_yaw = angle_diff(raw_yaw, self.neutral_yaw)
                rel_pitch = angle_diff(raw_pitch, self.neutral_pitch)
                rel_roll = angle_diff(raw_roll, self.neutral_roll)

                # Head Pose State Classification
                if abs(rel_yaw) <= self.yaw_thresh and abs(rel_pitch) <= self.pitch_thresh:
                    pose_state = "FORWARD"
                elif rel_yaw < -self.yaw_thresh:
                    pose_state = "LEFT"
                elif rel_yaw > self.yaw_thresh:
                    pose_state = "RIGHT"
                elif rel_pitch < -self.pitch_thresh:
                    pose_state = "DOWN"
                elif rel_pitch > self.pitch_thresh:
                    pose_state = "UP"
                else:
                    pose_state = "FORWARD"


            # Extreme deviation
            is_extreme = (abs(rel_yaw) > self.extreme_yaw) or (abs(rel_pitch) > self.extreme_pitch)
            is_forward = (pose_state in ["FORWARD", "CALIBRATING"])


            # -------------------------------------------------------------
            # NODDING-OFF PATTERN DETECTOR
            # Pitch moves down + downward velocity + eyes closed/closing
            # -------------------------------------------------------------
            is_downward_motion = (rel_pitch < -10.0) or (self.pitch_velocity < -15.0)
            if is_downward_motion and (is_eye_closed or not is_eye_open):
                self.nodding_frames += 1
            else:
                self.nodding_frames = max(0, self.nodding_frames - 2)

            self.is_nodding_off = self.nodding_frames >= 4  # ~130ms confirmation

            return {
                "raw_yaw": round(raw_yaw, 1),
                "raw_pitch": round(raw_pitch, 1),
                "raw_roll": round(raw_roll, 1),
                "neutral_yaw": round(self.neutral_yaw, 1),
                "neutral_pitch": round(self.neutral_pitch, 1),
                "neutral_roll": round(self.neutral_roll, 1),
                "relative_yaw": round(rel_yaw, 1),
                "relative_pitch": round(rel_pitch, 1),
                "relative_roll": round(rel_roll, 1),
                "head_pose_state": pose_state,
                "pitch_velocity": round(self.pitch_velocity, 1),
                "pitch_acceleration": round(self.pitch_acceleration, 1),
                "yaw_velocity": round(self.yaw_velocity, 1),
                "is_nodding_off": self.is_nodding_off,
                "is_head_forward": is_forward,
                "is_extreme_pose": is_extreme
            }

        except Exception:
            return {
                "raw_yaw": 0.0,
                "raw_pitch": 0.0,
                "raw_roll": 0.0,
                "neutral_yaw": round(self.neutral_yaw, 1),
                "neutral_pitch": round(self.neutral_pitch, 1),
                "neutral_roll": round(self.neutral_roll, 1),
                "relative_yaw": 0.0,
                "relative_pitch": 0.0,
                "relative_roll": 0.0,
                "head_pose_state": "UNKNOWN",
                "pitch_velocity": 0.0,
                "pitch_acceleration": 0.0,
                "is_nodding_off": False,
                "is_head_forward": False,
                "is_extreme_pose": False
            }
