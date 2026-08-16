import cv2
import numpy as np
import math

class HeadPoseEstimator:
    def __init__(self):
        # 3D model points of standard human face features
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)

        # Corresponding MediaPipe landmark indices
        self.LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def estimate_pose(self, landmarks, width, height):
        if not landmarks:
            return 0.0, 0.0, 0.0, "UNKNOWN"

        try:
            # Extract 2D coordinates of the landmark points
            image_points = []
            for idx in self.LANDMARK_INDICES:
                lm = landmarks[idx]
                image_points.append((lm.x * width, lm.y * height))
            image_points = np.array(image_points, dtype=np.float32)

            # Camera intrinsic matrix (approximation)
            focal_length = width
            center = (width / 2.0, height / 2.0)
            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            # Solve for PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return 0.0, 0.0, 0.0, "UNKNOWN"

            # Get rotation matrix
            rmat, _ = cv2.Rodrigues(rotation_vector)

            # Decompose rotation matrix into Euler angles (Pitch=X, Yaw=Y, Roll=Z) in degrees
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])

            # Classify coarse head pose state
            yaw_threshold = 18.0
            pitch_threshold = 15.0

            if yaw < -yaw_threshold:
                state = "LEFT"
            elif yaw > yaw_threshold:
                state = "RIGHT"
            elif pitch < -pitch_threshold:
                state = "DOWN"
            elif pitch > pitch_threshold:
                state = "UP"
            else:
                state = "FORWARD"

            return yaw, pitch, roll, state

        except Exception:
            return 0.0, 0.0, 0.0, "UNKNOWN"
