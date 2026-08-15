# collision_analysis.py - ADAS Time-To-Collision (TTC) & Road Hazard Analyzer
from collections import deque
try:
    from .config import (
        COLLISION_TTC_WARN,
        COLLISION_TTC_CRITICAL,
        ACCIDENT_MIN_DISTANCE,
        ACCIDENT_MIN_RELATIVE_SPEED,
    )
except ImportError:
    from config import (
        COLLISION_TTC_WARN,
        COLLISION_TTC_CRITICAL,
        ACCIDENT_MIN_DISTANCE,
        ACCIDENT_MIN_RELATIVE_SPEED,
    )

class CollisionSafetyAnalyzer:
    def __init__(self):
        # Dictionary of track_id -> deque of (timestamp, distance)
        self.track_history = {}

    def analyze_collision_risk(self, track_id, current_dist, current_time):
        """Calculates Time-To-Collision (TTC) and relative speed to classify risk."""
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=20)
            
        history = self.track_history[track_id]
        history.append((current_time, current_dist))
        
        # Need at least 5 measurements for stable rate calculation
        if len(history) < 5:
            return "Low", 999.0, 0.0
            
        prev_time, prev_dist = history[0]
        curr_time, curr_dist = history[-1]
        dt = curr_time - prev_time
        
        if dt <= 0:
            return "Low", 999.0, 0.0
            
        # Relative speed in m/s (positive = approaching/closing)
        rel_speed = (prev_dist - curr_dist) / dt
        
        # Time-to-collision
        if rel_speed > 0.3:
            ttc = curr_dist / rel_speed
        else:
            ttc = 999.0
            
        # Risk classification
        if curr_dist < ACCIDENT_MIN_DISTANCE or (ttc < COLLISION_TTC_CRITICAL and curr_dist < 15.0):
            risk = "High"
        elif curr_dist < 12.0 or (ttc < COLLISION_TTC_WARN and curr_dist < 25.0):
            risk = "Medium"
        else:
            risk = "Low"
            
        return risk, ttc, rel_speed

    def check_road_hazard(self, bbox, distance, frame_width):
        """Identifies objects directly in the host vehicle's path."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        
        lane_left_bound = frame_width * 0.36
        lane_right_bound = frame_width * 0.64
        
        is_in_lane = lane_left_bound < center_x < lane_right_bound
        
        if is_in_lane and distance < 15.0:
            return True
            
        return False
        
    def check_accident_impact(self, risk, distance, rel_speed):
        """Evaluates telemetry conditions to automatically trigger an accident report."""
        if risk == "High" and distance < ACCIDENT_MIN_DISTANCE and rel_speed >= ACCIDENT_MIN_RELATIVE_SPEED:
            return True
        return False
