# config.py - ADAS and Dashcam Perception Settings
import os

# --- CAMERA PARAMETERS & CALIBRATION ---
CAMERA_HEIGHT = 1.5        # Height of camera above ground (meters)
CAMERA_PITCH = 0.0         # Pitch angle of the camera (degrees)
FOCAL_LENGTH_PX = 950.0    # Estimated camera focal length in pixels (for 720p resolution)

# --- PROCESSING RESOLUTION ---
PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

# --- LANE DETECTION CONFIGURATION ---
# Region of Interest (ROI) coordinates normalized (0.0 to 1.0)
LANE_ROI = [
    (0.12, 0.95),  # Bottom-Left
    (0.43, 0.62),  # Top-Left
    (0.57, 0.62),  # Top-Right
    (0.88, 0.95)   # Bottom-Right
]
LANE_DEVIATION_THRESHOLD = 50  # Drift tolerance (pixels deviation from lane center)

# --- COLLISION & RISK ASSESSMENT ---
COLLISION_TTC_WARN = 3.0       # Time-to-Collision warning threshold (seconds)
COLLISION_TTC_CRITICAL = 1.5   # Time-to-Collision critical threshold (seconds)
ACCIDENT_MIN_DISTANCE = 2.0    # Distance in meters indicating collision/impact
ACCIDENT_MIN_RELATIVE_SPEED = 4.0 # Minimum approach speed indicating high impact deceleration (m/s)

# --- RECORDING CONFIGURATION ---
PRE_ACCIDENT_SECONDS = 30
POST_ACCIDENT_SECONDS = 30
DEFAULT_OUTPUT_DIR = "output"

# COCO indices for ADAS targets
TARGET_CLASSES = {
    0: "pedestrian",  # person
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
