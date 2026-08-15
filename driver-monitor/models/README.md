# MediaPipe & YOLO Model Files

These model weights are excluded from git due to their size.

## Download models

Run the setup helper:
```bash
cd driver-monitor
python run.py --setup
```

Or download manually:

| File | Source |
|---|---|
| `face_landmarker.task` | [MediaPipe Models](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| `yolo11n.pt` | Auto-downloaded by `ultralytics` on first run |
| `drowsiness_model/best.pt` | Auto-downloaded from HuggingFace on first run |
| `roadguardian/*.pkl` | Run `python training/train_roadguardian.py` to retrain |
