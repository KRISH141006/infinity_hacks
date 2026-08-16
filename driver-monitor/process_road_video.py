import argparse
import sys
import os
from src.adas.road_processor import RoadVideoProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path_str):
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(BASE_DIR, path_str))

def main():
    parser = argparse.ArgumentParser(description="ADAS & Dashcam Road Perception - Offline Video Processor")
    parser.add_argument("--input", default="input/external_road.mp4", help="Path to input external road video file")
    parser.add_argument("--output", default="output/processed_road.mp4", help="Path to output labeled road video file")
    parser.add_argument("--model", default="models/yolov8m.pt", help="Path to YOLO model weights (e.g., yolov8m.pt or yolo11n.pt)")
    parser.add_argument("--telemetry", default="output/road_telemetry.json", help="Path to output road telemetry JSON file")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    model_path = resolve_path(args.model)
    telemetry_path = resolve_path(args.telemetry)

    # Verify input exists
    if not os.path.exists(input_path):
        print(f"Error: Input road video '{input_path}' not found.")
        sys.exit(1)

    print("Starting ADAS Road Perception & Labeling pipeline...")
    processor = RoadVideoProcessor(
        model_path=model_path,
        input_video=input_path,
        output_video=output_path,
        telemetry_path=telemetry_path
    )

    success = processor.process()
    if success:
        print("Road video processing finished successfully!")
    else:
        print("Road video processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
