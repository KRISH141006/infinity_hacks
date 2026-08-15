import subprocess
import os
import sys

def main():
    print("==================================================")
    # 1. Check if input video exists
    input_video = "input/driver.mp4"
    if not os.path.exists(input_video):
        # Let's copy it from parent folder if it exists there
        parent_video = "../input/driver.mp4"
        if os.path.exists(parent_video):
            print(f"Copying driver.mp4 from {parent_video} to {input_video}...")
            os.makedirs("input", exist_ok=True)
            import shutil
            shutil.copy(parent_video, input_video)
        else:
            # Let's copy a clip instead if it exists
            parent_clip = "../input/clip_1.mp4"
            if os.path.exists(parent_clip):
                print(f"Copying clip_1.mp4 to {input_video}...")
                os.makedirs("input", exist_ok=True)
                import shutil
                shutil.copy(parent_clip, input_video)
            else:
                print(f"Error: Input video not found at '{input_video}'. Please place a video there.")
                sys.exit(1)

    # 2. Check if face landmarker model exists
    face_task = "models/face_landmarker.task"
    if not os.path.exists(face_task):
        parent_task = "../models/face_landmarker.task"
        if os.path.exists(parent_task):
            print(f"Copying face_landmarker.task from {parent_task} to {face_task}...")
            os.makedirs("models", exist_ok=True)
            import shutil
            shutil.copy(parent_task, face_task)
        else:
            print(f"Error: face_landmarker.task not found at '{face_task}'.")
            sys.exit(1)

    # 3. Process video if output doesn't exist yet
    processed_video = "output/processed_driver.mp4"
    if not os.path.exists(processed_video):
        print("Output video not found. Starting processing first...")
        result = subprocess.run([sys.executable, "process_video.py"])
        if result.returncode != 0:
            print("Error occurred during video processing.")
            sys.exit(1)
    else:
        print("Processed video already found. Skipping processing step (Use sidebar button in Streamlit to re-process).")

    # 4. Start Streamlit Dashboard
    print("\nLaunching Streamlit Dashboard...")
    try:
        subprocess.run(["streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\nStopping Driver Monitoring System.")

if __name__ == "__main__":
    main()
