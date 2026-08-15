import argparse
import sys
import os
from src.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Driver Monitoring System (DMS) - Offline Video Processor")
    parser.add_argument("--input", default="input/driver.mp4", help="Path to input video file")
    parser.add_argument("--output", default="output/processed_driver.mp4", help="Path to output processed video file")
    parser.add_argument("--models", default="models", help="Path to directory containing models")
    parser.add_argument("--telemetry", default="output/telemetry.json", help="Path to output telemetry JSON file")
    args = parser.parse_args()

    # Verify input exists
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found.")
        sys.exit(1)

    # Verify Face Landmarker task file exists
    face_task = os.path.join(args.models, "face_landmarker.task")
    if not os.path.exists(face_task):
        # Let's copy it from parent folder if it exists there
        parent_task = "../models/face_landmarker.task"
        if os.path.exists(parent_task):
            print(f"Copying face_landmarker.task from {parent_task} to {face_task}...")
            os.makedirs(args.models, exist_ok=True)
            import shutil
            shutil.copy(parent_task, face_task)
        else:
            print(f"Error: Face landmarker task file not found at '{face_task}'.")
            sys.exit(1)

    print("Starting Driver Monitoring System processing pipeline...")
    processor = VideoProcessor(
        model_dir=args.models,
        input_video=args.input,
        output_video=args.output,
        telemetry_path=args.telemetry
    )

    success = processor.process()
    if success:
        print("Processing finished successfully!")
    else:
        print("Processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
