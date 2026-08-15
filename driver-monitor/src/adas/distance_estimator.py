# distance_estimator.py - ADAS Flat-Ground & Homography Distance Estimator
import numpy as np
try:
    from .config import CAMERA_HEIGHT, CAMERA_PITCH, FOCAL_LENGTH_PX
except ImportError:
    from config import CAMERA_HEIGHT, CAMERA_PITCH, FOCAL_LENGTH_PX

class ADASDistanceEstimator:
    def __init__(self, camera_height=CAMERA_HEIGHT, camera_pitch=CAMERA_PITCH, focal_length=FOCAL_LENGTH_PX):
        self.camera_height = camera_height
        self.pitch_rad = np.radians(camera_pitch)
        self.focal_length = focal_length

    def estimate_distance(self, bbox, class_name, frame_height, frame_width):
        """Estimates vehicle distance using flat-ground homography projection.
        
        Projects the contact point (bottom-center of bounding box) onto ground plane.
        """
        x1, y1, x2, y2 = bbox
        bottom_y = y2
        
        # Horizon approximation (58% down the frame is typical for forward dashcams)
        cy = frame_height * 0.58
        y_diff = bottom_y - cy
        
        if y_diff > 10:
            # Safe below horizon
            pixel_angle = np.arctan(y_diff / self.focal_length)
            distance = self.camera_height / np.tan(self.pitch_rad + pixel_angle)
        else:
            # Scale-based fallback if tires are near or above horizon
            bbox_h = max(1, y2 - y1)
            real_h = 1.5
            if class_name in ["truck", "bus"]:
                real_h = 3.0
            elif class_name == "pedestrian":
                real_h = 1.7
            elif class_name in ["motorcycle", "bicycle"]:
                real_h = 1.4
            distance = (self.focal_length * real_h) / bbox_h
            
        return round(float(distance), 1)
