import argparse
import sys
import os
from src.adas.road_processor import RoadVideoProcessor

def main():
    parser = argparse.ArgumentParser(description="ADAS & Dashcam Road Perception - Offline Video Processor")
    parser.add_argument("--input", default="input/external_road.mp4", help="Path to input external road video file")
    parser.add_argument("--output", default="output/processed_road.mp4", help="Path to output labeled road video file")
    parser.add_argument("--model", default="models/yolov8m.pt", help="Path to YOLO model weights (e.g., yolov8m.pt or yolo11n.pt)")
    parser.add_argument("--telemetry", default="output/road_telemetry.json", help="Path to output road telemetry JSON file")
    args = parser.parse_args()

    # Verify input exists
    if not os.path.exists(args.input):
        print(f"Error: Input road video '{args.input}' not found.")
        sys.exit(1)

    print("Starting ADAS Road Perception & Labeling pipeline...")
    processor = RoadVideoProcessor(
        model_path=args.model,
        input_video=args.input,
        output_video=args.output,
        telemetry_path=args.telemetry
    )

    success = processor.process()
    if success:
        print("Road video processing finished successfully!")
    else:
        print("Road video processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
