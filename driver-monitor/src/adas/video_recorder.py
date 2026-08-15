# video_recorder.py - Rolling Pre/Post Accident Event Recorder
import cv2
import os
import json
from collections import deque
from datetime import datetime
try:
    from .config import PRE_ACCIDENT_SECONDS, POST_ACCIDENT_SECONDS, DEFAULT_OUTPUT_DIR
except ImportError:
    from config import PRE_ACCIDENT_SECONDS, POST_ACCIDENT_SECONDS, DEFAULT_OUTPUT_DIR

class SafetyVideoRecorder:
    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR, pre_sec=PRE_ACCIDENT_SECONDS, post_sec=POST_ACCIDENT_SECONDS):
        self.pre_buffer_seconds = pre_sec
        self.post_buffer_seconds = post_sec
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.fps = 30
        self.pre_buffer = None
        self.post_frames_left = 0
        self.writer = None
        self.is_recording = False
        self.current_timestamp = None

    def initialize_buffer(self, fps):
        self.fps = max(1, int(fps))
        max_pre_frames = self.pre_buffer_seconds * self.fps
        self.pre_buffer = deque(maxlen=max_pre_frames)

    def buffer_frame(self, frame):
        """Encodes frame as JPEG in memory to save RAM in rolling circular buffer."""
        if self.is_recording:
            return
        if self.pre_buffer is None:
            self.initialize_buffer(self.fps)
            
        success, encoded = cv2.imencode('.jpg', frame)
        if success:
            self.pre_buffer.append(encoded)

    def trigger_accident(self, frame_size, current_detections, witness_info=None):
        """Triggers recording, flushes pre-buffer, and compiles incident report."""
        if self.is_recording:
            return None
            
        self.is_recording = True
        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.post_frames_left = self.post_buffer_seconds * self.fps
        
        report_path = os.path.join(self.output_dir, f"accident_report_{self.current_timestamp}.json")
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "trigger_event": "ADAS Automatic / Manual Collision Trigger",
            "nearby_vehicles": current_detections,
            "witness_discovery": witness_info if witness_info else {}
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
            
        print(f"\n[ADAS ALERT] >>> ACCIDENT DETECTED! <<<")
        print(f"[ADAS ALERT] Incident report compiled at: {report_path}")
        
        video_path = os.path.join(self.output_dir, f"accident_clip_{self.current_timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(video_path, fourcc, self.fps, frame_size)
        
        pre_written = 0
        while self.pre_buffer:
            enc_frame = self.pre_buffer.popleft()
            frame = cv2.imdecode(enc_frame, cv2.IMREAD_COLOR)
            if frame is not None:
                self.writer.write(frame)
                pre_written += 1
                
        print(f"[ADAS ALERT] Wrote {pre_written} pre-accident frames to accident file.")
        return report_path

    def write_post_frame(self, frame):
        """Writes live frames during post-accident time window."""
        if not self.is_recording or self.writer is None:
            return False
            
        self.writer.write(frame)
        self.post_frames_left -= 1
        
        if self.post_frames_left <= 0:
            self.writer.release()
            self.writer = None
            self.is_recording = False
            print(f"[ADAS ALERT] Accident incident clip fully recorded.")
            return True
            
        return False

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
