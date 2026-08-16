import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.dms import (
    FaceTracker,
    EyeEngine,
    GazeEngine,
    HeadPoseEngine,
    YawnEngine,
    FatigueEngine,
    DistractionEngine,
    DrowsinessEngine,
    PhoneEngine,
    FusionEngine
)

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def make_landmarks(ear=0.30, mar=0.15, gaze_offset=0.0, yaw=0.0, pitch=0.0):
    """Synthesizes realistic 478 MediaPipe face landmarks."""
    lms = [MockLandmark(0.5, 0.5) for _ in range(478)]

    # Left eye: corners 33, 133
    lms[33] = MockLandmark(0.40, 0.40)
    lms[133] = MockLandmark(0.50, 0.40)
    h = ear * 0.10
    lms[160] = MockLandmark(0.43, 0.40 - h/2)
    lms[158] = MockLandmark(0.47, 0.40 - h/2)
    lms[144] = MockLandmark(0.43, 0.40 + h/2)
    lms[153] = MockLandmark(0.47, 0.40 + h/2)

    # Right eye: corners 362, 263
    lms[362] = MockLandmark(0.55, 0.40)
    lms[263] = MockLandmark(0.65, 0.40)
    lms[385] = MockLandmark(0.58, 0.40 - h/2)
    lms[387] = MockLandmark(0.62, 0.40 - h/2)
    lms[380] = MockLandmark(0.58, 0.40 + h/2)
    lms[373] = MockLandmark(0.62, 0.40 + h/2)

    # Irises with gaze offset (left/right/center)
    lms[468] = MockLandmark(0.45 + gaze_offset, 0.40)
    lms[473] = MockLandmark(0.60 + gaze_offset, 0.40)

    # Mouth: top 13, bottom 14, left 61, right 291
    lms[61] = MockLandmark(0.44, 0.70)
    lms[291] = MockLandmark(0.56, 0.70)
    mh = mar * 0.12
    lms[13] = MockLandmark(0.50, 0.70 - mh/2)
    lms[14] = MockLandmark(0.50, 0.70 + mh/2)

    # PnP Feature landmarks: Nose (1), Chin (152)
    lms[1] = MockLandmark(0.50, 0.50)
    lms[152] = MockLandmark(0.50, 0.85)

    return lms

class FullDMSTestRig:
    def __init__(self):
        self.face = FaceTracker()
        self.eye = EyeEngine()
        self.gaze = GazeEngine()
        self.head = HeadPoseEngine()
        self.yawn = YawnEngine()
        self.fatigue = FatigueEngine()
        self.distract = DistractionEngine()
        self.drowsy = DrowsinessEngine(use_yolo=False)
        self.phone = PhoneEngine()
        self.fusion = FusionEngine()
        self.t = 0.0

    def step(self, lms, dt=0.033, phone_det=False, phone_conf=0.0, custom_head=None, custom_gaze=None):
        self.t += dt
        face_d = self.face.update(lms, (720, 1280), dt)
        yawn_d = self.yawn.process(lms, dt, self.t)
        eye_d = self.eye.process(lms, dt, self.t, is_stable_upright=True, mar=yawn_d["mar"], tracking_reliable=face_d["is_confident"])
        
        if custom_head:
            head_d = custom_head
        else:
            head_d = self.head.process(lms, (720, 1280), dt, is_eye_open=not eye_d["is_eye_closed"], is_eye_closed=eye_d["is_eye_closed"])

        if custom_gaze:
            gaze_d = custom_gaze
        else:
            gaze_d = self.gaze.process(lms, is_eye_open=not eye_d["is_eye_closed"], is_stable_upright=head_d["is_head_forward"], tracking_reliable=face_d["is_confident"])

        fatigue_d = self.fatigue.process(eye_d["is_eye_closed"], yawn_d, eye_d, dt)
        drowsy_d = self.drowsy.process(eye_d, head_d, fatigue_d, yawn_d, yolo_drowsy_prob=0.0, dt=dt)
        phone_d = self.phone.process(phone_det, phone_conf, dt)
        distract_d = self.distract.process(gaze_d, head_d, eye_d, drowsy_d, face_d, dt)

        record, events = self.fusion.process(
            self.t, face_d, eye_d, gaze_d, head_d, yawn_d, fatigue_d, distract_d, drowsy_d, phone_d
        )
        return record

def test_A_normal_forward_driver():
    rig = FullDMSTestRig()
    lms = make_landmarks(ear=0.30, mar=0.15)
    for _ in range(60): # 2 seconds
        rec = rig.step(lms)
    assert rec["drowsiness_score"] < 25, f"Expected low drowsiness, got {rec['drowsiness_score']}"
    assert rec["distraction_score"] < 20, f"Expected low distraction, got {rec['distraction_score']}"
    assert rec["attention_score"] >= 80, f"Expected high attention, got {rec['attention_score']}"
    assert rec["master_driver_state"] == "ALERT"

def test_B_normal_blinking():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30)
    closed_lms = make_landmarks(ear=0.08)
    for _ in range(30): rig.step(open_lms) # calibrate
    for _ in range(6): rig.step(closed_lms) # 0.2s blink
    for _ in range(30): rec = rig.step(open_lms) # reopen
    assert rec["drowsiness_score"] < 35, f"Normal blink should not trigger drowsiness: {rec['drowsiness_score']}"
    assert not any("DROWSINESS" in a for a in rec["alerts"])

def test_C_long_blink():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30)
    closed_lms = make_landmarks(ear=0.08)
    for _ in range(30): rig.step(open_lms)
    for _ in range(18): rec = rig.step(closed_lms) # 0.6s long blink
    assert 30 <= rec["drowsiness_score"] <= 65, f"Expected moderate long-blink fatigue, got {rec['drowsiness_score']}"

def test_D_microsleep():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30)
    closed_lms = make_landmarks(ear=0.08)
    for _ in range(30): rig.step(open_lms)
    for _ in range(55): rec = rig.step(closed_lms) # 1.82s closure
    assert rec["drowsiness_score"] >= 85, f"Expected high drowsiness >= 85 for 1.8s sleep, got {rec['drowsiness_score']}"
    assert rec["master_driver_state"] in ["DROWSY", "DROWSY_NODDING"]
    assert rec["distraction_score"] <= 15, "Distraction MUST be low during sleep"

def test_E_sleeping_and_nodding():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30)
    closed_lms = make_landmarks(ear=0.08)
    for _ in range(30): rig.step(open_lms)
    # Pitch drops downward while eyes close
    custom_nod_head = {
        "raw_yaw": 0.0, "raw_pitch": -22.0, "raw_roll": 0.0,
        "neutral_yaw": 0.0, "neutral_pitch": 0.0, "neutral_roll": 0.0,
        "relative_yaw": 0.0, "relative_pitch": -22.0, "relative_roll": 0.0,
        "head_pose_state": "DOWN", "pitch_velocity": -25.0, "pitch_acceleration": 0.0,
        "yaw_velocity": 0.0, "is_nodding_off": True, "is_head_forward": False, "is_extreme_pose": False
    }
    for _ in range(55): rec = rig.step(closed_lms, custom_head=custom_nod_head)
    assert rec["drowsiness_score"] >= 90, f"Expected critical drowsiness >= 90, got {rec['drowsiness_score']}"
    assert rec["master_driver_state"] == "DROWSY_NODDING"
    assert rec["distraction_score"] <= 15, "MUST NOT classify head drop as distraction!"
    assert any("DROWSY NODDING" in a for a in rec["alerts"])

def test_F_yawning_event():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30, mar=0.15)
    yawn_lms = make_landmarks(ear=0.30, mar=0.55)
    for _ in range(30): rig.step(open_lms)
    for _ in range(45): rec = rig.step(yawn_lms) # 1.5s yawn
    assert rec["is_yawning"] or rec["yawn_probability"] >= 0.60
    assert rec["fatigue_score"] > 0, "Yawning should increase fatigue evidence"

def test_G_looking_left_3s():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30, gaze_offset=-0.08) # gaze left
    custom_left_head = {
        "raw_yaw": -30.0, "raw_pitch": 0.0, "raw_roll": 0.0,
        "neutral_yaw": 0.0, "neutral_pitch": 0.0, "neutral_roll": 0.0,
        "relative_yaw": -30.0, "relative_pitch": 0.0, "relative_roll": 0.0,
        "head_pose_state": "LEFT", "pitch_velocity": 0.0, "pitch_acceleration": 0.0,
        "yaw_velocity": 0.0, "is_nodding_off": False, "is_head_forward": False, "is_extreme_pose": False
    }
    for _ in range(30): rig.step(open_lms)
    for _ in range(90): rec = rig.step(open_lms, custom_head=custom_left_head) # 3s
    assert rec["distraction_score"] >= 80, f"Expected high distraction >= 80, got {rec['distraction_score']}"
    assert rec["master_driver_state"] == "DISTRACTED"
    assert rec["drowsiness_score"] < 25, "Drowsiness must remain low when looking left awake"

def test_H_looking_right_3s():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30, gaze_offset=+0.08) # gaze right
    custom_right_head = {
        "raw_yaw": 30.0, "raw_pitch": 0.0, "raw_roll": 0.0,
        "neutral_yaw": 0.0, "neutral_pitch": 0.0, "neutral_roll": 0.0,
        "relative_yaw": 30.0, "relative_pitch": 0.0, "relative_roll": 0.0,
        "head_pose_state": "RIGHT", "pitch_velocity": 0.0, "pitch_acceleration": 0.0,
        "yaw_velocity": 0.0, "is_nodding_off": False, "is_head_forward": False, "is_extreme_pose": False
    }
    for _ in range(30): rig.step(open_lms)
    for _ in range(90): rec = rig.step(open_lms, custom_head=custom_right_head) # 3s
    assert rec["distraction_score"] >= 80, f"Expected high distraction >= 80, got {rec['distraction_score']}"
    assert rec["master_driver_state"] == "DISTRACTED"

def test_I_looking_down_awake():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30, mar=0.15)
    custom_down_head = {
        "raw_yaw": 0.0, "raw_pitch": -25.0, "raw_roll": 0.0,
        "neutral_yaw": 0.0, "neutral_pitch": 0.0, "neutral_roll": 0.0,
        "relative_yaw": 0.0, "relative_pitch": -25.0, "relative_roll": 0.0,
        "head_pose_state": "DOWN", "pitch_velocity": 0.0, "pitch_acceleration": 0.0,
        "yaw_velocity": 0.0, "is_nodding_off": False, "is_head_forward": False, "is_extreme_pose": False
    }
    for _ in range(30): rig.step(open_lms)
    for _ in range(70): rec = rig.step(open_lms, custom_head=custom_down_head)
    assert rec["drowsiness_score"] < 40, f"Awake looking down must not equal microsleep, got {rec['drowsiness_score']}"

def test_J_phone_usage():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30)
    for _ in range(30): rig.step(open_lms)
    for _ in range(25): rec = rig.step(open_lms, phone_det=True, phone_conf=0.90)
    assert rec["phone_usage"] is True
    assert any("PHONE" in a for a in rec["alerts"])

def test_K_temporary_landmark_loss():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30)
    closed_lms = make_landmarks(ear=0.08)
    for _ in range(30): rig.step(open_lms)
    for _ in range(40): rec = rig.step(closed_lms)
    score_before = rec["drowsiness_score"]
    # Drop landmarks for 5 frames
    for _ in range(5): rec = rig.step(None)
    assert rec["drowsiness_score"] >= score_before - 5, "Drowsiness MUST NOT reset to zero on landmark loss!"

def test_L_camera_offset_with_gaze_center():
    rig = FullDMSTestRig()
    open_lms = make_landmarks(ear=0.30, gaze_offset=0.0) # gaze firmly forward
    custom_cam_offset_head = {
        "raw_yaw": 24.0, "raw_pitch": 8.0, "raw_roll": 0.0,
        "neutral_yaw": 0.0, "neutral_pitch": 0.0, "neutral_roll": 0.0,
        "relative_yaw": 24.0, "relative_pitch": 8.0, "relative_roll": 0.0,
        "head_pose_state": "RIGHT", "pitch_velocity": 0.0, "pitch_acceleration": 0.0,
        "yaw_velocity": 0.0, "is_nodding_off": False, "is_head_forward": False, "is_extreme_pose": False
    }
    for _ in range(30): rig.step(open_lms)
    for _ in range(60): rec = rig.step(open_lms, custom_head=custom_cam_offset_head)
    assert rec["distraction_score"] <= 20, f"Gaze center with camera angle MUST NOT produce high distraction: {rec['distraction_score']}"
    assert rec["master_driver_state"] == "ALERT"

if __name__ == "__main__":
    tests = [
        test_A_normal_forward_driver,
        test_B_normal_blinking,
        test_C_long_blink,
        test_D_microsleep,
        test_E_sleeping_and_nodding,
        test_F_yawning_event,
        test_G_looking_left_3s,
        test_H_looking_right_3s,
        test_I_looking_down_awake,
        test_J_phone_usage,
        test_K_temporary_landmark_loss,
        test_L_camera_offset_with_gaze_center,
    ]
    print("=" * 60)
    print("   🧪 DMS MULTI-SIGNAL ENGINE COMPREHENSIVE VALIDATION")
    print("=" * 60)
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ {t.__name__} FAILED: {e}")
    print(f"\nResults: {passed}/{len(tests)} tests passed.")
    if passed != len(tests):
        exit(1)
