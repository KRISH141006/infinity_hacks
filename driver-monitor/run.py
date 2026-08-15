#!/usr/bin/env python3
import subprocess
import os
import sys
import urllib.request

def download_file_if_missing(url, target_path, description):
    if not os.path.exists(target_path):
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        print(f"Downloading {description} to {target_path}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"✅ Successfully downloaded {description}.")
        except Exception as e:
            print(f"⚠️ Warning: Failed to download {description} from {url}: {e}")

def main():
    print("=" * 60)
    print("   🚗 RoadGuardian DMS & ADAS Safety Cockpit Launcher")
    print("=" * 60)

    # 1. Ensure models directory exists and download face_landmarker.task if missing
    face_task = "models/face_landmarker.task"
    if not os.path.exists(face_task):
        parent_task = "../models/face_landmarker.task"
        if os.path.exists(parent_task):
            print(f"Copying face_landmarker.task from parent directory...")
            os.makedirs("models", exist_ok=True)
            import shutil
            shutil.copy(parent_task, face_task)
        else:
            task_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            download_file_if_missing(task_url, face_task, "MediaPipe Face Landmarker Model")

    # 2. Check input and output directories
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # 3. Launch Streamlit Dashboard
    print("\n🚀 Launching Unified RoadGuardian Cockpit Dashboard...")
    print("👉 Access the UI at: http://localhost:8501")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\nStopping RoadGuardian Cockpit.")

if __name__ == "__main__":
    main()
