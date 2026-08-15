import os
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, confidence_threshold=0.25, check_interval=5):
        self.confidence_threshold = confidence_threshold
        self.check_interval = check_interval  # Process every Nth frame

        # Load YOLOv8n/YOLOv11n pretrained model (will detect cell phone, class 67 in COCO)
        print("Loading YOLO model for phone detection...")
        self.model = YOLO("yolo11n.pt")  # Light model
        
        # Temporal smoothing variables
        self.history = []
        self.max_history_len = 5
        self.last_detection = False
        self.last_confidence = 0.0

    def process_frame(self, frame, frame_number):
        # Only run object detection every Nth frame to save CPU processing power
        if frame_number % self.check_interval == 0:
            results = self.model.predict(source=frame, verbose=False)
            detected = False
            max_conf = 0.0

            if results:
                # Iterate through detections
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # COCO class 67 is 'cell phone'
                    if class_id == 67 and conf >= self.confidence_threshold:
                        detected = True
                        if conf > max_conf:
                            max_conf = conf

            # Append to history
            self.history.append((detected, max_conf))
            if len(self.history) > self.max_history_len:
                self.history.pop(0)

            # Determine smoothed state
            detections_count = sum(1 for det, _ in self.history if det)
            self.last_detection = detections_count >= 3  # persistent detection required
            self.last_confidence = max((c for _, c in self.history if c > 0.0), default=0.0)

        return self.last_detection, self.last_confidence
