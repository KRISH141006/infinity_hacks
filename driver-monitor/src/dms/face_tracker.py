# face_tracker.py - Robust Temporal Face Tracking & Quality Monitor
import numpy as np

class FaceTracker:
    def __init__(self, max_lost_grace_sec=0.35):
        self.max_lost_grace_sec = max_lost_grace_sec
        self.lost_frames = 0
        self.lost_duration = 0.0
        
        self.last_bbox = None  # [x1, y1, x2, y2]
        self.last_quality = 0.0
        self.tracking_state = "NO_FACE"  # FACE_TRACKED, FACE_UNCERTAIN, FACE_TRACKING_LOST
        self.is_confident = False

    def update(self, landmarks, frame_shape, dt):
        """
        Updates face track state based on raw MediaPipe landmarks.
        landmarks: list of NormalizedLandmarks or None.
        frame_shape: (height, width)
        dt: frame delta time in seconds.
        """
        h, w = frame_shape[:2]

        if landmarks and len(landmarks) >= 468:
            self.lost_frames = 0
            self.lost_duration = 0.0

            # Calculate tight bounding box from landmarks
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1, x2 = max(0, int(min(xs))), min(w, int(max(xs)))
            y1, y2 = max(0, int(min(ys))), min(h, int(max(ys)))
            
            bbox_area = max(1, (x2 - x1) * (y2 - y1))
            frame_area = max(1, w * h)
            area_ratio = bbox_area / frame_area

            # Quality metric based on face size, presence of key features, and coordinate validity
            quality = min(1.0, area_ratio * 4.0)  # sensible quality score 0.0 to 1.0
            if getattr(landmarks[1], 'z', 0.0) != 0.0:
                quality = min(1.0, quality + 0.2)
            quality = max(0.4, min(1.0, quality))

            self.last_bbox = [x1, y1, x2, y2]
            self.last_quality = quality
            self.tracking_state = "FACE_TRACKED"
            self.is_confident = True

            return {
                "tracking_state": self.tracking_state,
                "is_confident": True,
                "face_bbox": self.last_bbox,
                "tracking_quality": round(self.last_quality, 2),
                "lost_duration": 0.0,
                "lost_frames": 0
            }

        # Landmarks missing
        self.lost_frames += 1
        self.lost_duration += dt

        if self.lost_duration <= self.max_lost_grace_sec:
            self.tracking_state = "FACE_UNCERTAIN"
            self.is_confident = False
            quality = max(0.1, self.last_quality * 0.7)
        else:
            self.tracking_state = "FACE_TRACKING_LOST"
            self.is_confident = False
            quality = 0.0

        return {
            "tracking_state": self.tracking_state,
            "is_confident": False,
            "face_bbox": self.last_bbox,
            "tracking_quality": round(quality, 2),
            "lost_duration": round(self.lost_duration, 2),
            "lost_frames": self.lost_frames
        }
