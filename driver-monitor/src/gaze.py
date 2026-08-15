import math

class GazeTracker:
    def __init__(self):
        # MediaPipe iris and eye landmarks
        # Left eye: corners at 33 and 133, iris center at 468
        # Right eye: corners at 362 and 263, iris center at 473
        pass

    def estimate_gaze(self, landmarks):
        if not landmarks:
            return "UNKNOWN"

        # Check if iris coordinates exist in landmarks (indices 468 to 477)
        if len(landmarks) <= 473:
            return "UNKNOWN"

        # Left Eye
        left_corner_l = landmarks[33]
        right_corner_l = landmarks[133]
        iris_l = landmarks[468]

        # Right Eye
        left_corner_r = landmarks[362]
        right_corner_r = landmarks[263]
        iris_r = landmarks[473]

        # Function to calculate horizontal relative ratio (0 to 1, where 0.5 is center)
        def get_horizontal_ratio(left, right, iris):
            width = right.x - left.x
            if width == 0:
                return 0.5
            return (iris.x - left.x) / width

        ratio_l = get_horizontal_ratio(left_corner_l, right_corner_l, iris_l)
        ratio_r = get_horizontal_ratio(left_corner_r, right_corner_r, iris_r)
        
        # Average ratio
        avg_ratio = (ratio_l + ratio_r) / 2.0

        # Estimate vertical ratio (using vertical coordinate of iris relative to corner height)
        # Eye corners vertical coordinate
        eye_y_avg = (left_corner_l.y + right_corner_l.y + left_corner_r.y + right_corner_r.y) / 4.0
        iris_y_avg = (iris_l.y + iris_r.y) / 2.0
        
        # A simple delta threshold
        v_diff = iris_y_avg - eye_y_avg

        # Thresholds
        # Looking right: iris shifts closer to the outer/right corner of the left eye
        # and inner/right corner of the right eye. The ratio increases.
        # Looking left: the ratio decreases.
        
        # Left eye is index 33 (left corner) to 133 (right corner)
        # Right eye is index 362 (left corner) to 263 (right corner)
        # (Note: MediaPipe x coordinates are mirrored or normal depending on flip)
        # Normally, smaller x is left of screen, larger x is right of screen.
        
        state = "CENTER"
        
        if avg_ratio < 0.44:
            state = "LEFT"
        elif avg_ratio > 0.56:
            state = "RIGHT"
        elif v_diff < -0.015:
            state = "UP"
        elif v_diff > 0.015:
            state = "DOWN"

        return state
