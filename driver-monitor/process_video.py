import argparse
import sys
import os
from src.video_processor import VideoProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path_str):
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(BASE_DIR, path_str))

def main():
    parser = argparse.ArgumentParser(description="Driver Monitoring System (DMS) - Offline Video Processor")
    parser.add_argument("--input", default="input/driver.mp4", help="Path to input video file")
    parser.add_argument("--output", default="output/processed_driver.mp4", help="Path to output processed video file")
    parser.add_argument("--models", default="models", help="Path to directory containing models")
    parser.add_argument("--telemetry", default="output/telemetry.json", help="Path to output telemetry JSON file")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    models_path = resolve_path(args.models)
    telemetry_path = resolve_path(args.telemetry)

    # Verify input exists
    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' not found.")
        sys.exit(1)

    # Verify Face Landmarker task file exists
    face_task = os.path.join(models_path, "face_landmarker.task")
    if not os.path.exists(face_task):
        parent_task = os.path.join(BASE_DIR, "../models/face_landmarker.task")
        if os.path.exists(parent_task):
            print(f"Copying face_landmarker.task from {parent_task} to {face_task}...")
            os.makedirs(models_path, exist_ok=True)
            import shutil
            shutil.copy(parent_task, face_task)
        else:
            print(f"Error: Face landmarker task file not found at '{face_task}'.")
            sys.exit(1)

    print("Starting Driver Monitoring System processing pipeline...")
    processor = VideoProcessor(
        model_dir=models_path,
        input_video=input_path,
        output_video=output_path,
        telemetry_path=telemetry_path
    )

    success = processor.process()
    if success:
        print("Processing finished successfully!")
    else:
        print("Processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
