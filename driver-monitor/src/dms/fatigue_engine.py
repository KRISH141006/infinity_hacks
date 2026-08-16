# fatigue_engine.py - Multi-Window PERCLOS (Short/Med/Long) & Cumulative Driver Fatigue
from collections import deque
import numpy as np

class FatigueEngine:
    def __init__(self, fps=30.0):
        self.fps = max(1.0, float(fps))
        
        # Multi-window buffer sizes
        self.short_len = int(self.fps * 4.0)    # 4 seconds
        self.med_len = int(self.fps * 15.0)     # 15 seconds
        self.long_len = int(self.fps * 45.0)    # 45 seconds

        self.closure_history = deque(maxlen=self.long_len)
        self.fatigue_score = 0
        self.fatigue_state = "ALERT"  # ALERT, MILD_FATIGUE, FATIGUED, HIGH_FATIGUE

    def process(self, is_eye_closed, yawn_data, eye_data, dt):
        val = 1.0 if is_eye_closed else 0.0
        self.closure_history.append(val)
        n = len(self.closure_history)

        # Calculate PERCLOS across the 3 windows
        # 1. Short (2-5s)
        n_short = min(n, self.short_len)
        perclos_short = sum(list(self.closure_history)[-n_short:]) / max(1, n_short)

        # 2. Medium (10-20s)
        n_med = min(n, self.med_len)
        perclos_med = sum(list(self.closure_history)[-n_med:]) / max(1, n_med)

        # 3. Long (30-60s)
        perclos_long = sum(self.closure_history) / max(1, n)

        # -------------------------------------------------------------
        # CUMULATIVE FATIGUE SCORING (0 to 100)
        # Slow accumulation based on PERCLOS_long, repeated yawns, slow blinks
        # -------------------------------------------------------------
        score = 0.0

        # PERCLOS long-term contribution (up to 55 points)
        score += min(55.0, perclos_long * 100.0 * 2.2)

        # PERCLOS medium-term contribution (up to 20 points)
        score += min(20.0, perclos_med * 100.0 * 0.8)

        # Yawn history contribution (up to 20 points)
        recent_yawns = yawn_data.get("recent_yawns_10m", 0)
        score += min(20.0, recent_yawns * 8.0)
        if yawn_data.get("is_yawning", False):
            score += 10.0

        # Sluggish long blink frequency contribution (up to 15 points)
        long_blinks = eye_data.get("long_blink_count", 0)
        score += min(15.0, long_blinks * 3.0)

        target_score = int(min(100, max(0, score)))
        
        # Smooth fatigue score progression
        if target_score > self.fatigue_score:
            self.fatigue_score = int(self.fatigue_score + max(1, (target_score - self.fatigue_score) * 0.15))
        else:
            self.fatigue_score = int(self.fatigue_score - max(1, (self.fatigue_score - target_score) * 0.05))

        self.fatigue_score = max(0, min(100, self.fatigue_score))

        # Classify state
        if self.fatigue_score >= 70:
            self.fatigue_state = "HIGH_FATIGUE"
        elif self.fatigue_score >= 45:
            self.fatigue_state = "FATIGUED"
        elif self.fatigue_score >= 25:
            self.fatigue_state = "MILD_FATIGUE"
        else:
            self.fatigue_state = "ALERT"

        return {
            "perclos_short": round(perclos_short, 3),
            "perclos_medium": round(perclos_med, 3),
            "perclos_long": round(perclos_long, 3),
            "fatigue_score": self.fatigue_score,
            "fatigue_state": self.fatigue_state,
            "is_fatigued": self.fatigue_score >= 45
        }
