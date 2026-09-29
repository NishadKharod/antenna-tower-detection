import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_TEST_IMAGES = str(
    PROJECT_ROOT / "cell tower antenna detection.v2i.yolov11" / "test" / "images"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with trained YOLO11 Antenna Tower detector (1 class: 'antenna')"
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument(
        "--source",
        type=str,
        default=BUNDLED_TEST_IMAGES,
        help=f"Image path, folder, video, or 0 for webcam (default: bundled test images folder)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--project", type=str, default="runs/tower_detection")
    parser.add_argument("--name", type=str, default="predict")
    return parser.parse_args()


def resolve_device(requested):
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def main():
    args = parse_args()
    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}")
        sys.exit(1)

    device = resolve_device(args.device)
    model = YOLO(args.weights)

    print(f"\nRunning detection on: {args.source}")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=args.save,
        project=args.project,
        name=args.name,
    )

    out_dir = os.path.join(args.project, args.name)
    print(f"\nSaved predictions to: {out_dir}")
    print(f"Processed {len(results)} image(s) / frame(s).")


if __name__ == "__main__":
    main()
