# RoadGuardian Model Files

These `.pkl` model files are too large for GitHub and are excluded via `.gitignore`.

## Download / Retrain

To get the models, run the training script from the project root:

```bash
cd driver-monitor
python training/train_roadguardian.py
```

This will train both models and save them here automatically:
- `roadguardian_event_status_rf_v2.pkl`
- `roadguardian_event_type_rf_v1.pkl`

You will need the training data CSV in `training/roadguardian_event_training_data_v1.csv`.
