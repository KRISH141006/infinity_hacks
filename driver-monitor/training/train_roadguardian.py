"""
RoadGuardian — Event Classification Model Training
====================================================
Trains two Random Forest classifiers:
  1. Event Status  : NORMAL | WARNING | CRITICAL
  2. Event Type    : NORMAL | HARD_BRAKING | NEAR_MISS | LOSS_OF_CONTROL | COLLISION

Converted from blackbox.ipynb (Colab notebook).

Usage:
    python training/train_roadguardian.py

Outputs saved to: models/roadguardian/
"""

import os
import json
import uuid
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "training", "roadguardian_event_training_data_v1.csv")
OUTPUT_MODEL_DIR   = os.path.join(BASE_DIR, "models", "roadguardian")

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature columns (must match inference in src/blackbox.py)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "speed",
    "acceleration",
    "braking",
    "steering_deviation",
    "nearest_vehicle_distance",
    "pedestrian_detected",
    "road_hazard_detected",
    "driver_distraction",
    "driver_drowsiness",
    "visibility",
]


# ===========================================================================
# STEP 1 — Load & Inspect Raw Data
# ===========================================================================

print("=" * 60)
print("STEP 1 — Loading raw training data")
print("=" * 60)

df = pd.read_csv(TRAINING_DATA_PATH)
print(f"Shape: {df.shape}")
print(df["event_status"].value_counts())
print()


# ===========================================================================
# STEP 2 — Augment CRITICAL samples to reach class balance
# ===========================================================================

print("=" * 60)
print("STEP 2 — Augmenting CRITICAL class samples")
print("=" * 60)

rng = np.random.default_rng(42)

critical = df[df["event_status"] == "CRITICAL"].copy()
normal   = df[df["event_status"] == "NORMAL"].copy()
warning  = df[df["event_status"] == "WARNING"].copy()

TARGET_SAMPLES = 1500
needed = TARGET_SAMPLES - len(critical)

augmented = critical.sample(n=needed, replace=True, random_state=42).copy()

# Add small realistic Gaussian noise to continuous features
continuous_cols = [
    "speed", "acceleration", "braking", "steering_deviation",
    "nearest_vehicle_distance", "driver_distraction", "driver_drowsiness", "visibility",
]
noise_scale = {
    "speed": 0.05, "acceleration": 0.08, "braking": 0.04,
    "steering_deviation": 0.04, "nearest_vehicle_distance": 0.08,
    "driver_distraction": 0.04, "driver_drowsiness": 0.04, "visibility": 0.03,
}

for col in continuous_cols:
    std = df[col].std()
    augmented[col] += rng.normal(0, std * noise_scale[col], size=len(augmented))

# Clip to physically valid ranges
augmented["speed"]                   = augmented["speed"].clip(0, 130)
augmented["braking"]                 = augmented["braking"].clip(0, 1)
augmented["steering_deviation"]      = augmented["steering_deviation"].clip(0, 1)
augmented["nearest_vehicle_distance"]= augmented["nearest_vehicle_distance"].clip(0, 100)
augmented["driver_distraction"]      = augmented["driver_distraction"].clip(0, 1)
augmented["driver_drowsiness"]       = augmented["driver_drowsiness"].clip(0, 1)
augmented["visibility"]              = augmented["visibility"].clip(0, 1)
augmented["event_status"]            = "CRITICAL"

# Combine all classes with equal sampling
critical_final = pd.concat([critical, augmented], ignore_index=True).sample(TARGET_SAMPLES, random_state=42)
normal_final   = normal.sample(min(TARGET_SAMPLES, len(normal)), random_state=42)
warning_final  = warning.sample(min(TARGET_SAMPLES, len(warning)), random_state=42)

balanced_df = (
    pd.concat([normal_final, warning_final, critical_final], ignore_index=True)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

balanced_csv_path = os.path.join(BASE_DIR, "training", "roadguardian_event_training_data_v2.csv")
balanced_df.to_csv(balanced_csv_path, index=False)

print("Balanced class distribution:")
print(balanced_df["event_status"].value_counts())
print(f"Saved balanced dataset → {balanced_csv_path}")
print()


# ===========================================================================
# STEP 3 — Train Event STATUS model (NORMAL / WARNING / CRITICAL)
# ===========================================================================

print("=" * 60)
print("STEP 3 — Training Event Status classifier (RF v2)")
print("=" * 60)

X = balanced_df[FEATURE_COLS]
y_status = balanced_df["event_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_status, test_size=0.20, random_state=42, stratify=y_status
)

status_model = RandomForestClassifier(
    n_estimators=300, random_state=42,
    class_weight="balanced", min_samples_leaf=2, n_jobs=-1,
)
status_model.fit(X_train, y_train)

y_pred = status_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["NORMAL", "WARNING", "CRITICAL"]))
print()

status_model_path = os.path.join(OUTPUT_MODEL_DIR, "roadguardian_event_status_rf_v2.pkl")
joblib.dump(status_model, status_model_path)
print(f"✅ Status model saved → {status_model_path}")
print()


# ===========================================================================
# STEP 4 — Train Event TYPE model (NORMAL / HARD_BRAKING / NEAR_MISS / ...)
# ===========================================================================

print("=" * 60)
print("STEP 4 — Training Event Type classifier (RF v1)")
print("=" * 60)

y_type = balanced_df["event_type"]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X, y_type, test_size=0.20, random_state=42, stratify=y_type
)

type_model = RandomForestClassifier(
    n_estimators=300, random_state=42,
    class_weight="balanced", min_samples_leaf=2, n_jobs=-1,
)
type_model.fit(X_train_t, y_train_t)

y_pred_t = type_model.predict(X_test_t)
print(f"Accuracy: {accuracy_score(y_test_t, y_pred_t):.4f}")
print(classification_report(y_test_t, y_pred_t, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(
    y_test_t, y_pred_t,
    labels=["NORMAL", "HARD_BRAKING", "NEAR_MISS", "LOSS_OF_CONTROL", "COLLISION"],
))
print()

type_model_path = os.path.join(OUTPUT_MODEL_DIR, "roadguardian_event_type_rf_v1.pkl")
joblib.dump(type_model, type_model_path)
print(f"✅ Type model saved → {type_model_path}")
print()


# ===========================================================================
# STEP 5 — Quick Inference Smoke-Test
# ===========================================================================

print("=" * 60)
print("STEP 5 — Smoke testing trained models")
print("=" * 60)

def test_scenario(label, inputs):
    sample = pd.DataFrame([inputs])[FEATURE_COLS]

    s_pred = status_model.predict(sample)[0]
    s_prob = status_model.predict_proba(sample).max()

    t_pred = type_model.predict(sample)[0]
    t_prob = type_model.predict_proba(sample).max()

    final_status = "CRITICAL" if t_pred == "COLLISION" else s_pred

    print(f"\n── {label}")
    print(f"   Event Status : {final_status}  ({s_prob:.1%})")
    print(f"   Event Type   : {t_pred}  ({t_prob:.1%})")

test_scenario("Normal Driving", {
    "speed": 40, "acceleration": 0.2, "braking": 0.05, "steering_deviation": 0.08,
    "nearest_vehicle_distance": 40, "pedestrian_detected": 0, "road_hazard_detected": 0,
    "driver_distraction": 0.05, "driver_drowsiness": 0.05, "visibility": 0.95,
})
test_scenario("Warning Situation", {
    "speed": 65, "acceleration": -2.0, "braking": 0.45, "steering_deviation": 0.35,
    "nearest_vehicle_distance": 8, "pedestrian_detected": 0, "road_hazard_detected": 1,
    "driver_distraction": 0.45, "driver_drowsiness": 0.35, "visibility": 0.75,
})
test_scenario("Collision", {
    "speed": 75, "acceleration": -8.0, "braking": 0.98, "steering_deviation": 0.7,
    "nearest_vehicle_distance": 2, "pedestrian_detected": 0, "road_hazard_detected": 1,
    "driver_distraction": 0.7, "driver_drowsiness": 0.2, "visibility": 0.7,
})
test_scenario("Hard Braking", {
    "speed": 65, "acceleration": -5.5, "braking": 0.9, "steering_deviation": 0.15,
    "nearest_vehicle_distance": 15, "pedestrian_detected": 0, "road_hazard_detected": 0,
    "driver_distraction": 0.2, "driver_drowsiness": 0.1, "visibility": 0.9,
})
test_scenario("Near Miss — Pedestrian", {
    "speed": 60, "acceleration": -4.0, "braking": 0.8, "steering_deviation": 0.75,
    "nearest_vehicle_distance": 4, "pedestrian_detected": 1, "road_hazard_detected": 0,
    "driver_distraction": 0.4, "driver_drowsiness": 0.1, "visibility": 0.8,
})

print("\n" + "=" * 60)
print("Training complete. Models are in models/roadguardian/")
print("=" * 60)
