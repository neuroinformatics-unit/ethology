import argparse
import json
import sys
from pathlib import Path

from detectors.inference.yolo_inference import YOLODetector

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run object detector on video and output JSON detections."
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video.")
    parser.add_argument("--detector_type", type=str, required=True, choices=["yolo", "frcnn"], help="Type of detector.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--visualize", action="store_true", help="Show video with bounding boxes drawn.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if args.detector_type == "yolo":
            detector = YOLODetector(args.model_path)
        else:
            raise ValueError(f"Unsupported detector type: {args.detector_type}")

        print("Running inference...")
        all_detections = detector.run(args.video_path, visualize=args.visualize)

        print(f"Inference complete. Total detections: {len(all_detections)}")

        with open(args.output_path, "w") as f:
            json.dump(all_detections, f, indent=2)

        print(f"Detections written to: {args.output_path}")

    except Exception as e:
        print(f"\n Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()