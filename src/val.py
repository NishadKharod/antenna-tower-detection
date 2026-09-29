import argparse
import os
import sys

import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a trained YOLO11 Antenna Tower detector"
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--device", type=str, default=None)
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


def load_class_names(data_yaml):
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)


def main():
    args = parse_args()
    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}")
        sys.exit(1)

    device = resolve_device(args.device)
    model = YOLO(args.weights)
    class_names = load_class_names(args.data)

    print(f"\nValidating on {args.split} split ...")
    metrics = model.val(
        data=args.data, split=args.split,
        imgsz=args.imgsz, batch=args.batch, device=device,
    )

    print("\n========================================")
    print(f" {args.split.upper()} RESULTS")
    print("========================================")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP75    : {metrics.box.map75:.4f}")

    print("\nPer-class mAP50-95:")
    for i, name in enumerate(class_names):
        score = metrics.box.maps[i]
        print(f"  {i:>2} {name:20s}: {score:.4f}")


if __name__ == "__main__":
    main()
