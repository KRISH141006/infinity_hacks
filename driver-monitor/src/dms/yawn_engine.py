# yawn_engine.py - Temporal Yawn Curve Event Detector (Opening -> Holding -> Closing)
import math
import numpy as np
from collections import deque

class YawnEngine:
    def __init__(self, open_mar_thresh=0.42, peak_mar_thresh=0.50, min_duration_sec=1.2):
        self.open_mar_thresh = open_mar_thresh
        self.peak_mar_thresh = peak_mar_thresh
        self.min_duration_sec = min_duration_sec

        # MediaPipe Mouth Indices
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291

        # Temporal states: NO_YAWN, MOUTH_OPENING, YAWN_CANDIDATE, YAWN_CONFIRMED, RECOVERED
        self.state = "NO_YAWN"
        self.last_mar = 0.15
        self.mar_velocity = 0.0
        self.mar_acceleration = 0.0
        self.last_mar_vel = 0.0

        self.current_event_duration = 0.0
        self.opening_duration = 0.0
        self.holding_duration = 0.0
        self.closing_duration = 0.0
        self.peak_mar_reached = 0.0

        # Oscillation tracker to distinguish speech from yawning
        self.mar_history = deque(maxlen=60)  # ~2 seconds
        self.yawn_count = 0
        self.yawn_probability = 0.0
        self.last_yawn_time = 0.0
        self.yawn_history_timestamps = deque(maxlen=30)

    def dist_3d(self, p1, p2):
        z1 = getattr(p1, 'z', 0.0)
        z2 = getattr(p2, 'z', 0.0)
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (z1 - z2)**2)

    def compute_mar(self, landmarks):
        if not landmarks or len(landmarks) < 292:
            return self.last_mar
        top = landmarks[self.MOUTH_TOP]
        bot = landmarks[self.MOUTH_BOTTOM]
        left = landmarks[self.MOUTH_LEFT]
        right = landmarks[self.MOUTH_RIGHT]

        v = self.dist_3d(top, bot)
        h = self.dist_3d(left, right)
        if h <= 1e-6:
            return self.last_mar
        return v / h

    def process(self, landmarks, dt, current_time):
        mar = self.compute_mar(landmarks)
        self.mar_history.append(mar)

        # Derivatives
        if dt > 0:
            raw_vel = (mar - self.last_mar) / dt
            self.mar_velocity = float(0.7 * self.mar_velocity + 0.3 * raw_vel)
            raw_acc = (self.mar_velocity - self.last_mar_vel) / dt
            self.mar_acceleration = float(0.7 * self.mar_acceleration + 0.3 * raw_acc)
            self.last_mar_vel = self.mar_velocity
        self.last_mar = mar

        # Speech vs Yawn metric: Speech has high variance / zero-crossings in velocity; Yawn is smooth & sustained
        mar_variance = float(np.var(self.mar_history)) if len(self.mar_history) > 10 else 0.0

        # -------------------------------------------------------------
        # YAWN CURVE STATE MACHINE
        # -------------------------------------------------------------
        is_mouth_open = mar >= self.open_mar_thresh

        if is_mouth_open:
            self.current_event_duration += dt
            self.peak_mar_reached = max(self.peak_mar_reached, mar)

            if self.state == "NO_YAWN":
                self.state = "MOUTH_OPENING"
                self.opening_duration += dt
            elif self.state == "MOUTH_OPENING":
                self.opening_duration += dt
                if self.current_event_duration >= 0.6 and self.peak_mar_reached >= self.peak_mar_thresh:
                    self.state = "YAWN_CANDIDATE"
            elif self.state in ["YAWN_CANDIDATE", "YAWN_CONFIRMED"]:
                self.holding_duration += dt
                if self.current_event_duration >= self.min_duration_sec and self.peak_mar_reached >= self.peak_mar_thresh:
                    if self.state != "YAWN_CONFIRMED":
                        self.state = "YAWN_CONFIRMED"
                        self.yawn_count += 1
                        self.last_yawn_time = current_time
                        self.yawn_history_timestamps.append(current_time)

            # Calculate continuous probability (0.0 to 1.0)
            dur_factor = min(1.0, self.current_event_duration / 2.0)
            peak_factor = min(1.0, max(0.0, (self.peak_mar_reached - 0.38) / 0.20))
            self.yawn_probability = float(min(1.0, 0.4 * dur_factor + 0.6 * peak_factor))

        else:
            # Mouth closed / closing
            if self.state in ["MOUTH_OPENING", "YAWN_CANDIDATE", "YAWN_CONFIRMED"]:
                self.closing_duration += dt
                if self.closing_duration >= 0.4:
                    # Reset event
                    self.state = "RECOVERED"
            elif self.state == "RECOVERED":
                self.state = "NO_YAWN"
                self.current_event_duration = 0.0
                self.opening_duration = 0.0
                self.holding_duration = 0.0
                self.closing_duration = 0.0
                self.peak_mar_reached = 0.0
                self.yawn_probability = 0.0
            else:
                self.current_event_duration = max(0.0, self.current_event_duration - 1.5 * dt)
                self.yawn_probability = max(0.0, self.yawn_probability - 1.5 * dt)

        # Prune old yawn timestamps (>10 mins)
        while self.yawn_history_timestamps and (current_time - self.yawn_history_timestamps[0]) > 600.0:
            self.yawn_history_timestamps.popleft()

        return {
            "mar": round(mar, 3),
            "mar_velocity": round(self.mar_velocity, 2),
            "mar_acceleration": round(self.mar_acceleration, 2),
            "yawn_state": self.state,
            "is_yawning": self.state == "YAWN_CONFIRMED" or self.yawn_probability >= 0.70,
            "yawn_probability": round(self.yawn_probability, 2),
            "yawn_duration": round(self.current_event_duration, 2),
            "yawn_count": self.yawn_count,
            "recent_yawns_10m": len(self.yawn_history_timestamps),
            "speech_variance": round(mar_variance, 4)
        }
