# phone_engine.py - Temporal Confirmation & Phone Usage Engine
import os

class PhoneEngine:
    def __init__(self, confirmation_sec=0.35, cooldown_sec=1.5):
        self.confirmation_sec = confirmation_sec
        self.cooldown_sec = cooldown_sec

        self.detection_streak_duration = 0.0
        self.cooldown_duration = 0.0
        self.is_confirmed = False
        self.phone_duration = 0.0
        self.phone_confidence = 0.0

    def process(self, raw_detected, raw_conf, dt):
        if raw_detected:
            self.detection_streak_duration += dt
            self.cooldown_duration = self.cooldown_sec
            self.phone_confidence = max(self.phone_confidence, float(raw_conf))

            if self.detection_streak_duration >= self.confirmation_sec:
                self.is_confirmed = True
                self.phone_duration += dt
        else:
            self.detection_streak_duration = max(0.0, self.detection_streak_duration - dt)
            if self.cooldown_duration > 0:
                self.cooldown_duration -= dt
                if self.is_confirmed:
                    self.phone_duration += dt
            else:
                self.is_confirmed = False
                self.phone_duration = max(0.0, self.phone_duration - 1.5 * dt)
                self.phone_confidence = max(0.0, self.phone_confidence - 0.1)

        return {
            "phone_usage": self.is_confirmed,
            "phone_confidence": round(self.phone_confidence, 2),
            "phone_duration": round(self.phone_duration, 2)
        }
