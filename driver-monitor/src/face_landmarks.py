import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceLandmarkerHelper:
    def __init__(self, model_path="models/face_landmarker.task"):
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame, timestamp_ms):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def close(self):
        self.landmarker.close()
