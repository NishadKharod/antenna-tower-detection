# Antenna Tower Detection using YOLO11

An object detection project for detecting antenna / cell towers in images using **Ultralytics YOLO11**.

## 📌 Overview

This project trains a YOLO11 Nano (`yolo11n`) model on a custom dataset for antenna / cell tower detection. The pipeline includes:

- Dataset preparation (ZIP upload, extraction, YAML validation)
- YOLO11 training with early stopping
- Validation & per-class mAP reporting
- Test set evaluation
- Inference / prediction with bounding box visualization

## 🧰 Requirements

- Python 3.9+
- PyTorch (with or without CUDA)
- Ultralytics YOLO11

```bash
pip install -r requirements.txt
```

## 📁 Dataset Structure

Your dataset ZIP must follow the YOLO format:

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/   (optional)
    ├── images/
    └── labels/
```

## 🚀 Usage

### 1. Train the model

```bash
python src/train.py --dataset your_dataset.zip --epochs 100 --imgsz 640 --batch 16
```

### 2. Only validate an existing model

```bash
python src/val.py --weights runs/tower_detection/yolo11_tower_detector/weights/best.pt --data data/data.yaml
```

### 3. Run inference on images

```bash
python src/predict.py --weights runs/tower_detection/yolo11_tower_detector/weights/best.pt --source path/to/images
```

### 4. Use the Colab notebook

Upload `notebooks/yolo11_antenna_tower_detection.ipynb` to Google Colab and run it step-by-step.

## 📊 Expected Outputs

Training outputs are saved to `runs/tower_detection/yolo11_tower_detector/`:

```
weights/
  ├── best.pt    ← Best model (lowest validation loss)
  └── last.pt    ← Last checkpoint
results.csv        ← Training metrics per epoch
confusion_matrix.png
PR_curve.png
F1_curve.png
```

## 🔧 Training Hyperparameters (defaults)

| Param       | Value | Description                       |
|-------------|-------|-----------------------------------|
| model       | yolo11n | YOLO11 Nano (fast & lightweight) |
| epochs      | 100   | Max training epochs               |
| imgsz       | 640   | Input image size (pixels)         |
| batch       | 16    | Batch size (adjust based on VRAM) |
| patience    | 20    | Early stopping patience           |
| seed        | 42    | Random seed for reproducibility   |
| device      | auto  | 0 for GPU, cpu for CPU            |

## 📝 License

MIT
