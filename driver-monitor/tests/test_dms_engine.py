import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.drowsiness import DrowsinessDetector
from src.attention import AttentionManager

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def create_mock_landmarks(ear=0.30, mar=0.15):
    """Creates a mock 478 MediaPipe landmark structure with specified EAR and MAR."""
    landmarks = [MockLandmark(0.5, 0.5) for _ in range(478)]

    # Left eye: 33 (outer), 160 (top1), 158 (top2), 133 (inner), 153 (bot2), 144 (bot1)
    # Horizontal width = 0.10
    landmarks[33] = MockLandmark(0.40, 0.40)
    landmarks[133] = MockLandmark(0.50, 0.40)
    # Height = ear * width
    h = ear * 0.10
    landmarks[160] = MockLandmark(0.43, 0.40 - h/2)
    landmarks[158] = MockLandmark(0.47, 0.40 - h/2)
    landmarks[144] = MockLandmark(0.43, 0.40 + h/2)
    landmarks[153] = MockLandmark(0.47, 0.40 + h/2)

    # Right eye: 362 (inner), 385 (top1), 387 (top2), 263 (outer), 373 (bot2), 380 (bot1)
    landmarks[362] = MockLandmark(0.55, 0.40)
    landmarks[263] = MockLandmark(0.65, 0.40)
    landmarks[385] = MockLandmark(0.58, 0.40 - h/2)
    landmarks[387] = MockLandmark(0.62, 0.40 - h/2)
    landmarks[380] = MockLandmark(0.58, 0.40 + h/2)
    landmarks[373] = MockLandmark(0.62, 0.40 + h/2)

    # Irises
    landmarks[468] = MockLandmark(0.45, 0.40) # Left iris center
    landmarks[473] = MockLandmark(0.60, 0.40) # Right iris center

    # Mouth: top 13, bottom 14, left 61, right 291
    # width = 0.12
    landmarks[61] = MockLandmark(0.44, 0.70)
    landmarks[291] = MockLandmark(0.56, 0.70)
    mh = mar * 0.12
    landmarks[13] = MockLandmark(0.50, 0.70 - mh/2)
    landmarks[14] = MockLandmark(0.50, 0.70 + mh/2)

    return landmarks

def test_1_normal_forward_driving():
    """Test 1: Normal forward driving -> Drowsiness LOW, Distraction LOW."""
    drowsy = DrowsinessDetector(use_yolo=False)
    attention = AttentionManager()
    fps = 30

    landmarks = create_mock_landmarks(ear=0.30, mar=0.15)
    head_pose = (0.0, 0.0, 0.0, "FORWARD")
    gaze = ("CENTER", 0.95)
    phone = (False, 0.0)

    for f in range(60): # 2 seconds
        d_res = drowsy.process_frame(landmarks, None, fps, f, relative_pitch=0.0, head_forward=True)
        a_res = attention.compute_scores(d_res, head_pose, gaze, phone, fps)

    assert d_res["drowsiness_score"] < 25, f"Expected low drowsiness, got {d_res['drowsiness_score']}"
    assert a_res["distraction_score"] < 20, f"Expected low distraction, got {a_res['distraction_score']}"
    assert a_res["attention_score"] > 80, f"Expected high attention, got {a_res['attention_score']}"

def test_2_normal_blinking():
    """Test 2: Normal blinking (<0.35s) -> No drowsiness alert."""
    drowsy = DrowsinessDetector(use_yolo=False)
    attention = AttentionManager()
    fps = 30

    open_lm = create_mock_landmarks(ear=0.30)
    closed_lm = create_mock_landmarks(ear=0.10)

    # 1. Calibrate open eyes for 1 sec
    for f in range(30):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("CENTER", 0.95), (False, 0.0), fps)

    # 2. Blink for 6 frames (0.2 seconds)
    for f in range(30, 36):
        d_res = drowsy.process_frame(closed_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        a_res = attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("UNKNOWN", 0.0), (False, 0.0), fps)

    # 3. Eyes reopen
    for f in range(36, 60):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        a_res = attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("CENTER", 0.95), (False, 0.0), fps)

    assert d_res["drowsiness_score"] < 35, f"Normal blink should not produce high drowsiness score: {d_res['drowsiness_score']}"
    assert not any("DROWSINESS" in alert for alert in a_res["alerts"]), "Normal blink should not trigger drowsiness alert"

def test_3_prolonged_eye_closure():
    """Test 3: Prolonged eye closure for 1.5–2.0 sec -> Drowsiness HIGH."""
    drowsy = DrowsinessDetector(use_yolo=False)
    attention = AttentionManager()
    fps = 30

    open_lm = create_mock_landmarks(ear=0.30)
    closed_lm = create_mock_landmarks(ear=0.08)

    # Calibrate open eyes
    for f in range(30):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("CENTER", 0.95), (False, 0.0), fps)

    # Close eyes for 50 frames (1.67 seconds)
    for f in range(30, 80):
        d_res = drowsy.process_frame(closed_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        a_res = attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("UNKNOWN", 0.0), (False, 0.0), fps)

    assert d_res["drowsiness_score"] >= 80, f"Expected high drowsiness score >= 80, got {d_res['drowsiness_score']}"
    assert d_res["drowsiness_state"] in ["DROWSY", "DROWSY_NODDING"], f"Expected DROWSY state, got {d_res['drowsiness_state']}"
    assert any("DROWSINESS" in alert for alert in a_res["alerts"]), "Expected Drowsiness alert to be active"

def test_4_eye_closure_plus_head_drop():
    """Test 4: Eye closure + downward head drop -> DROWSY_NODDING / DROWSINESS HIGH, NOT DRIVER DISTRACTED."""
    drowsy = DrowsinessDetector(use_yolo=False)
    attention = AttentionManager()
    fps = 30

    open_lm = create_mock_landmarks(ear=0.30)
    closed_lm = create_mock_landmarks(ear=0.08)

    # Calibrate open eyes
    for f in range(30):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("CENTER", 0.95), (False, 0.0), fps)

    # Eyes closed + head pitch dropping to -25 degrees for 1.8s
    for f in range(30, 85):
        d_res = drowsy.process_frame(closed_lm, None, fps, f, relative_pitch=-25.0, head_forward=False)
        a_res = attention.compute_scores(d_res, (0.0, -25.0, 0.0, "DOWN"), ("UNKNOWN", 0.0), (False, 0.0), fps)

    assert d_res["drowsiness_score"] >= 90, f"Expected critical drowsiness >= 90, got {d_res['drowsiness_score']}"
    assert d_res["drowsiness_state"] == "DROWSY_NODDING", f"Expected DROWSY_NODDING, got {d_res['drowsiness_state']}"
    assert a_res["distraction_score"] <= 20, f"Head-drop during sleep MUST NOT be scored as distraction! Got {a_res['distraction_score']}"
    assert not any("DISTRACTED" in alert for alert in a_res["alerts"]), "Must NOT trigger driver distraction alert during drowsy nodding"

def test_5_looking_right_eyes_open_3s():
    """Test 5: Looking right with eyes open for 3 seconds -> Distraction HIGH."""
    drowsy = DrowsinessDetector(use_yolo=False)
    attention = AttentionManager()
    fps = 30

    open_lm = create_mock_landmarks(ear=0.30)

    # Calibrate
    for f in range(30):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("CENTER", 0.95), (False, 0.0), fps)

    # Look right for 90 frames (3 seconds) with eyes open
    for f in range(30, 120):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=False)
        a_res = attention.compute_scores(d_res, (35.0, 0.0, 0.0, "RIGHT"), ("RIGHT", 0.90), (False, 0.0), fps)

    assert a_res["distraction_score"] >= 80, f"Expected high distraction score >= 80, got {a_res['distraction_score']}"
    assert a_res["distraction_state"] == "DISTRACTED", f"Expected DISTRACTED state, got {a_res['distraction_state']}"
    assert any("DISTRACTED" in alert for alert in a_res["alerts"]), "Expected Distraction alert"
    assert d_res["drowsiness_score"] < 25, "Drowsiness should remain low"

def test_6_brief_glance_right_0_3s():
    """Test 6: Brief glance right for only 0.3s -> No major distraction alert."""
    drowsy = DrowsinessDetector(use_yolo=False)
    attention = AttentionManager()
    fps = 30

    open_lm = create_mock_landmarks(ear=0.30)

    # Calibrate
    for f in range(30):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("CENTER", 0.95), (False, 0.0), fps)

    # Glance right for 9 frames (0.3s)
    for f in range(30, 39):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=False)
        a_res = attention.compute_scores(d_res, (25.0, 0.0, 0.0, "RIGHT"), ("RIGHT", 0.85), (False, 0.0), fps)

    assert a_res["distraction_score"] < 50, f"Brief glance should not produce high distraction: {a_res['distraction_score']}"
    assert not any("DISTRACTED" in alert for alert in a_res["alerts"]), "Brief glance should not trigger distraction alert"

def test_7_gaze_center_with_moderate_head_offset():
    """Test 7: Gaze center with moderate head offset -> Distraction LOW."""
    drowsy = DrowsinessDetector(use_yolo=False)
    attention = AttentionManager()
    fps = 30

    open_lm = create_mock_landmarks(ear=0.30)

    # Calibrate
    for f in range(30):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("CENTER", 0.95), (False, 0.0), fps)

    # Head yaw at 24 deg (moderate turn) but gaze firmly CENTER at road
    for f in range(30, 90): # 2 seconds
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=False)
        a_res = attention.compute_scores(d_res, (24.0, 0.0, 0.0, "RIGHT"), ("CENTER", 0.95), (False, 0.0), fps)

    assert a_res["distraction_score"] <= 20, f"Gaze center on road must keep distraction low: {a_res['distraction_score']}"
    assert not any("DISTRACTED" in alert for alert in a_res["alerts"]), "Should not alert distraction when gaze is center"

def test_8_temporary_landmark_loss_during_drowsiness():
    """Test 8: Temporary landmark loss (<300ms) during drowsiness -> Do not reset drowsiness to zero."""
    drowsy = DrowsinessDetector(use_yolo=False)
    attention = AttentionManager()
    fps = 30

    open_lm = create_mock_landmarks(ear=0.30)
    closed_lm = create_mock_landmarks(ear=0.08)

    # Calibrate
    for f in range(30):
        d_res = drowsy.process_frame(open_lm, None, fps, f, relative_pitch=0.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, 0.0, 0.0, "FORWARD"), ("CENTER", 0.95), (False, 0.0), fps)

    # Eyes closed for 1.2s -> DROWSY_CANDIDATE
    for f in range(30, 66):
        d_res = drowsy.process_frame(closed_lm, None, fps, f, relative_pitch=-10.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, -10.0, 0.0, "FORWARD"), ("UNKNOWN", 0.0), (False, 0.0), fps)

    score_before_loss = d_res["drowsiness_score"]
    assert score_before_loss > 50, f"Score before loss should be elevated, got {score_before_loss}"

    # Missing landmarks for 5 frames (~160ms < 300ms)
    for f in range(66, 71):
        d_res = drowsy.process_frame(None, None, fps, f, relative_pitch=-10.0, head_forward=True)
        attention.compute_scores(d_res, (0.0, -10.0, 0.0, "FORWARD"), ("UNKNOWN", 0.0), (False, 0.0), fps)

    assert d_res["drowsiness_score"] > 50, f"Drowsiness score MUST NOT reset to 0 on brief landmark loss! Got {d_res['drowsiness_score']}"
    assert d_res["drowsiness_state"] in ["DROWSY_CANDIDATE", "DROWSY", "DROWSY_NODDING"], f"State should not reset, got {d_res['drowsiness_state']}"

if __name__ == "__main__":
    tests = [
        test_1_normal_forward_driving,
        test_2_normal_blinking,
        test_3_prolonged_eye_closure,
        test_4_eye_closure_plus_head_drop,
        test_5_looking_right_eyes_open_3s,
        test_6_brief_glance_right_0_3s,
        test_7_gaze_center_with_moderate_head_offset,
        test_8_temporary_landmark_loss_during_drowsiness,
    ]
    print("Running DMS Engine Test Suite...")
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ {t.__name__} FAILED: {e}")
    print(f"\nResult: {passed}/{len(tests)} tests passed.")
    if passed != len(tests):
        exit(1)

