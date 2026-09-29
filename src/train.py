import argparse
import glob
import os
import sys
import zipfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 for Antenna / Cell Tower Detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to your dataset ZIP file (e.g. dataset.zip)",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data/dataset",
        help="Directory where dataset is / will be extracted",
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=None,
        help="Direct path to data.yaml (skips ZIP extraction)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO11 model file (yolo11n/s/m/l/x.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 0 for GPU, cpu for CPU, leave empty for auto",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project", type=str, default="runs/tower_detection"
    )
    parser.add_argument("--name", type=str, default="yolo11_tower_detector")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Inference confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU threshold"
    )
    return parser.parse_args()


def extract_zip(zip_path: str, extract_dir: str):
    if not zip_path.lower().endswith(".zip"):
        raise ValueError(f"Not a ZIP file: {zip_path}")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {zip_path} -> {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Done.")


def find_data_yaml(root_dir: str):
    matches = glob.glob(os.path.join(root_dir, "**", "data.yaml"), recursive=True)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"data.yaml not found under {root_dir}. "
            "Make sure your dataset ZIP contains a data.yaml file."
        )
    return matches[0]


def check_structure(dataset_root: str):
    required = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
    ]
    optional = ["test/images", "test/labels"]
    print("\n=== Dataset Structure Check ===")
    ok = True
    for folder in required:
        p = os.path.join(dataset_root, folder)
        if os.path.exists(p):
            n = len(os.listdir(p)) if os.path.isdir(p) else 0
            print(f"  [ OK ] {folder:18s} ({n} items)")
        else:
            print(f"  [FAIL] {folder:18s} NOT FOUND")
            ok = False
    for folder in optional:
        p = os.path.join(dataset_root, folder)
        if os.path.exists(p):
            n = len(os.listdir(p)) if os.path.isdir(p) else 0
            print(f"  [OPT ] {folder:18s} ({n} items)")
        else:
            print(f"  [ ---] {folder:18s} not present (optional)")
    if not ok:
        raise RuntimeError("Required dataset folders missing. Aborting.")
    return ok


def fix_data_yaml(data_yaml_in: str, dataset_root: str, out_yaml: str):
    with open(data_yaml_in, "r") as f:
        data = yaml.safe_load(f) or {}
    data["path"] = str(Path(dataset_root).resolve())
    data["train"] = str(Path(dataset_root, "train", "images").resolve())
    data["val"] = str(Path(dataset_root, "valid", "images").resolve())
    test_images = Path(dataset_root, "test", "images")
    if test_images.exists():
        data["test"] = str(test_images.resolve())
    os.makedirs(os.path.dirname(out_yaml) or ".", exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"\nCorrected data.yaml -> {out_yaml}")
    return data, out_yaml


def print_classes(data):
    names = data.get("names", [])
    print("\n=== Class Names ===")
    if isinstance(names, dict):
        sorted_keys = sorted(names.keys(), key=lambda x: int(x))
        class_names = [names[k] for k in sorted_keys]
        for k in sorted_keys:
            print(f"  {int(k):>2}: {names[k]}")
    else:
        class_names = list(names)
        for idx, name in enumerate(class_names):
            print(f"  {idx:>2}: {name}")
    print(f"  Total classes: {len(class_names)}")
    return class_names


def count_images(dataset_root: str):
    def count(folder):
        p = Path(dataset_root, folder, "images")
        return len(list(p.glob("*"))) if p.exists() else 0
    print("\n=== Dataset Size ===")
    for split in ("train", "valid", "test"):
        print(f"  {split:6s}: {count(split)} images")


def resolve_device(requested: str):
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            print("\nCUDA available. Using GPU 0.")
            return 0
    except Exception:
        pass
    print("\nCUDA not available. Using CPU (this will be slow!).")
    return "cpu"


def main():
    args = parse_args()

    # 1. Extract ZIP if provided, otherwise locate data.yaml
    if args.data_yaml:
        data_yaml_original = args.data_yaml
        dataset_root = str(Path(data_yaml_original).resolve().parent)
    elif args.dataset:
        extract_zip(args.dataset, args.extract_dir)
        data_yaml_original = find_data_yaml(args.extract_dir)
        dataset_root = str(Path(data_yaml_original).resolve().parent)
    else:
        # Try to auto-find
        try:
            data_yaml_original = find_data_yaml(args.extract_dir)
            dataset_root = str(Path(data_yaml_original).resolve().parent)
        except FileNotFoundError:
            print(
                "ERROR: Provide --dataset <your.zip> or --data-yaml <path/data.yaml>"
            )
            sys.exit(1)

    print(f"\nOriginal data.yaml: {data_yaml_original}")
    print(f"Dataset root       : {dataset_root}")

    # 2. Validate structure
    check_structure(dataset_root)

    # 3. Fix and save data.yaml with absolute paths
    data_yaml_final = "data.yaml"
    data, data_yaml_final = fix_data_yaml(
        data_yaml_original, dataset_root, data_yaml_final
    )

    # 4. Print classes and sizes
    class_names = print_classes(data)
    count_images(dataset_root)

    # 5. Resolve device
    device = resolve_device(args.device)

    # 6. Load model
    print(f"\nLoading YOLO11 model: {args.model}")
    model = YOLO(args.model)

    # 7. Train
    print("\n========================================")
    print(" STARTING YOLO11 TRAINING")
    print("========================================\n")

    results = model.train(
        data=data_yaml_final,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        save=True,
        save_period=5,
        seed=args.seed,
        project=args.project,
        name=args.name,
    )

    training_folder = os.path.join(args.project, args.name)
    best_model_path = os.path.join(training_folder, "weights", "best.pt")
    last_model_path = os.path.join(training_folder, "weights", "last.pt")

    print("\n========================================")
    print(" TRAINING COMPLETED")
    print("========================================")
    print(f"  Training folder : {training_folder}")
    print(f"  Best model      : {best_model_path}")
    print(f"  Last checkpoint : {last_model_path}")

    if not os.path.exists(best_model_path):
        print("WARNING: best.pt not found. Aborting evaluation.")
        sys.exit(0)

    # 8. Load best model and validate
    best_model = YOLO(best_model_path)

    print("\n========================================")
    print(" VALIDATION (VAL SPLIT)")
    print("========================================\n")
    val_metrics = best_model.val(
        data=data_yaml_final, split="val", imgsz=args.imgsz,
        batch=args.batch, device=device,
    )
    print(f"\n  mAP50-95 : {val_metrics.box.map:.4f}")
    print(f"  mAP50    : {val_metrics.box.map50:.4f}")
    print(f"  mAP75    : {val_metrics.box.map75:.4f}")

    print("\nPer-class mAP50-95:")
    for i, name in enumerate(class_names):
        score = val_metrics.box.maps[i]
        print(f"  {i:>2} {name:20s}: {score:.4f}")

    # 9. Test evaluation (if test exists)
    test_images = Path(dataset_root, "test", "images")
    if test_images.exists():
        print("\n========================================")
        print(" TEST DATASET EVALUATION")
        print("========================================\n")
        try:
            test_metrics = best_model.val(
                data=data_yaml_final, split="test",
                imgsz=args.imgsz, batch=args.batch, device=device,
            )
            print(f"  Test mAP50-95 : {test_metrics.box.map:.4f}")
            print(f"  Test mAP50    : {test_metrics.box.map50:.4f}")
            print(f"  Test mAP75    : {test_metrics.box.map75:.4f}")
        except Exception as e:
            print(f"  Test evaluation skipped: {e}")

        # 10. Predict on test images
        print("\n========================================")
        print(" RUNNING DETECTION ON TEST IMAGES")
        print("========================================\n")
        best_model.predict(
            source=str(test_images),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            save=True,
            project=args.project,
            name="test_predictions",
        )
    else:
        print("\nTest folder not found. Skipping test evaluation & prediction.")

    print("\nAll done! 🎉")


if __name__ == "__main__":
    main()
