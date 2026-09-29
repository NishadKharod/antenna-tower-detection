# 📡 Antenna Tower Detection using YOLO11

An end-to-end object detection project for identifying **antenna / cell towers** in images, built with **Ultralytics YOLO11**. Includes training, validation, inference, **and a production-ready Flask HTTP API for website integration**.

## ✨ Features

- 🧠 **YOLO11 Nano (`yolo11n`)** — fast, lightweight, accurate
- 📦 **Bundled dataset** — *Cell Tower Antenna Detection v2i* (Roboflow) is already committed to the repo → **zero setup to train**
- 🧪 **Full pipeline** — train → validate → test-set eval → prediction + rendered boxes
- 🎯 **Correct classes** — `nc: 1`, single class: **`antenna`** (matches `data.yaml` exactly)
- 🌐 **Flask HTTP API** — endpoints your website can call (with CORS enabled)
- 🖱️ **Built-in drag-and-drop web UI** on `/` for quick testing
- ⚙️ **Configurable** — GPU auto-detection, early stopping, deterministic seed

---

## 📦 Dataset (Bundled)

The dataset you uploaded via the GitHub UI — **"Cell Tower Antenna Detection v2i" (Roboflow)** — is already in the repo.

| Item | Value |
|---|---|
| Local folder | `cell tower antenna detection.v2i.yolov11/` |
| Format | YOLO (images + .txt labels) |
| Classes | **nc = 1**, `names: ['antenna']` |
| Train | **30** images |
| Valid | **6** images |
| Test | **6** images |
| Total | **42** images |
| License | CC BY 4.0 |
| Source | https://universe.roboflow.com/cradpole/cell-tower-antenna-detection/dataset/2 |

Canonical dataset config lives in the project root at **[data.yaml](data.yaml)** with correct relative paths.

---

## 🧰 Requirements

- Python **3.9+**
- PyTorch (with or without CUDA)
- CUDA GPU is recommended for training (CPU works but is ~10–50× slower)

```bash
cd antenna-tower-detection
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ Train the model

No arguments needed — the bundled dataset is used **automatically**:

```bash
python src/train.py
```

That's it! It will:
1. Load the bundled dataset + `./data.yaml` (1 class: `antenna`)
2. Validate dataset structure
3. Print class names + image counts
4. Train YOLO11n for 100 epochs (early-stop patience = 20)
5. Validate on `val` + `test` splits
6. Save predictions on all test images

Output goes to:
```
runs/tower_detection/yolo11_tower_detector/
├── weights/
│   ├── best.pt     ← 👈 use this for the API / inference
│   └── last.pt
├── results.csv
├── confusion_matrix.png
├── PR_curve.png, F1_curve.png, R_curve.png, P_curve.png
└── ...
```

Optional flags:
```bash
python src/train.py --epochs 150 --imgsz 640 --batch 16 --device 0
# Train with a different dataset ZIP:
# python src/train.py --dataset new_dataset.zip --extract-dir data/new_dataset
# Or with a manual data.yaml:
# python src/train.py --data-yaml /path/to/data.yaml
```

### 2️⃣ Validate a trained checkpoint

```bash
python src/val.py --weights runs/tower_detection/yolo11_tower_detector/weights/best.pt
# --data defaults to ./data.yaml (bundled dataset)
# --split val  (or test)
```

### 3️⃣ Run inference

```bash
# Default: runs on the bundled test split images
python src/predict.py --weights runs/tower_detection/yolo11_tower_detector/weights/best.pt

# Or your own image(s):
python src/predict.py --weights .../best.pt --source path/to/my_photo.jpg
python src/predict.py --weights .../best.pt --source path/to/folder  --conf 0.3
```

---

## 🌐 Website Integration — Flask HTTP API

### Start the server

```bash
# Windows:
run_server.bat runs\tower_detection\yolo11_tower_detector\weights\best.pt

# Linux / macOS:
./run_server.sh  runs/tower_detection/yolo11_tower_detector/weights/best.pt

# Or directly:
python src/app.py --weights runs/tower_detection/yolo11_tower_detector/weights/best.pt --port 5000
```

### HTTP Endpoints

| Method | URL | Description |
|---|---|---|
| `GET`  | `/` | **Web UI** — drag & drop an image to test detection in the browser |
| `GET`  | `/health` | Health check + dataset metadata (classes, split sizes, loaded status) |
| `GET`  | `/api/info` | Project / dataset / API description (JSON) |
| `POST` | `/api/detect` | Upload image → get detections as JSON (bboxes, class, confidence, counts) |
| `POST` | `/api/detect/render` | Upload image → get **annotated PNG** back (boxes + labels drawn) |

### `POST /api/detect` — request / response

**Request** (`multipart/form-data`):
| Field | Type | Default | Notes |
|---|---|---|---|
| `image`  | file  | required | JPG/PNG/WebP etc. |
| `conf`   | float | 0.25     | Minimum confidence (0.01–0.99) |
| `iou`    | float | 0.45     | NMS IoU threshold |
| `imgsz`  | int   | 640      | Inference size (256–2048) |
| `image_id` | str | random   | Optional identifier echoed back |

**Example response (JSON):**
```json
{
  "image_id": "tower_001",
  "image_width": 1024,
  "image_height": 682,
  "inference_ms": 42.7,
  "counts": { "antenna": 3 },
  "detections": [
    {
      "class_id": 0,
      "class_name": "antenna",
      "confidence": 0.9234,
      "bbox": [ 120, 80, 540, 910 ]
    },
    ...
  ]
}
```

### Call the API from your website frontend

```html
<input type="file" id="file" accept="image/*"/>
<script>
document.getElementById('file').addEventListener('change', async (e) => {
  const fd = new FormData();
  fd.append('image', e.target.files[0]);
  fd.append('conf', 0.3);

  const res = await fetch('http://localhost:5000/api/detect', { method: 'POST', body: fd });
  const json = await res.json();
  console.log('Detections:', json.detections);
  console.log('Count per class:', json.counts); // { antenna: 3 }

  // Also get rendered image:
  const res2 = await fetch('http://localhost:5000/api/detect/render', { method: 'POST', body: fd });
  const blob = await res2.blob();
  document.getElementById('preview').src = URL.createObjectURL(blob);
});
</script>
```

> 💡 **CORS** is already enabled in the Flask app, so you can call this from any website origin (localhost, deployed domain, etc.).

### Call from Python backend

```python
from src.detector import AntennaTowerDetector

detector = AntennaTowerDetector(
    "runs/tower_detection/yolo11_tower_detector/weights/best.pt",
    class_names=["antenna"],   # matches nc=1 in data.yaml
)
detector.load()

result = detector.predict("photo.jpg", conf=0.3, imgsz=640)
print(result.to_dict())
# {'image_id': 'photo.jpg', 'counts': {'antenna': 2}, 'detections': [...], ...}

# Or get rendered PNG bytes back:
meta, png_bytes = detector.predict_and_render("photo.jpg")
with open("out.png", "wb") as f:
    f.write(png_bytes)
```

---

## 📁 Project Structure

```
antenna-tower-detection/
├── data.yaml                                        ← canonical config (nc=1, class 0: 'antenna')
├── requirements.txt
├── .gitignore
├── README.md
├── run_server.sh / run_server.bat                   ← one-click API server launchers
├── cell tower antenna detection.v2i.yolov11/         ← 📦 BUNDLED DATASET
│   ├── data.yaml
│   ├── train/  (30 images + labels)
│   ├── valid/  ( 6 images + labels)
│   └── test/   ( 6 images + labels)
├── data/
│   ├── data.example.yaml                             ← example/reference data.yaml
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── train.py                                      ← train + val + test + predict
    ├── val.py                                        ← validate a checkpoint
    ├── predict.py                                    ← CLI inference
    ├── detector.py                                   ← AntennaTowerDetector class (import into any Python web app)
    └── app.py                                        ← Flask API + Web UI
```

---

## 🔧 Default Hyperparameters

| Param | Value |
|---|---|
| Model | `yolo11n.pt` (YOLO11 Nano) |
| Epochs | 100 (early stops at patience=20) |
| Image size | 640 |
| Batch | 16 |
| Optimizer | SGD (Ultralytics default) |
| Seed | 42 (reproducible) |
| Conf threshold | 0.25 |
| NMS IoU | 0.45 |
| Device | Auto → GPU 0 if CUDA available, else CPU |

---

## 📝 License

Code: **MIT**  
Dataset (bundled): **CC BY 4.0** — Roboflow / `cradpole` project [Cell Tower Antenna Detection v2](https://universe.roboflow.com/cradpole/cell-tower-antenna-detection/dataset/2)
