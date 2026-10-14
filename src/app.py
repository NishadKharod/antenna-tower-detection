"""Flask web server exposing Antenna Tower Detection via HTTP.

Dataset bundled with this repo:
    "Cell Tower Antenna Detection v2i" (Roboflow) — 1 class: antenna

Endpoints
---------
GET  /health               → health check + dataset / classes info
GET  /                     → simple browser UI for drag-and-drop uploads
GET  /api/info             → dataset classes, paths, defaults
POST /api/detect           → JSON: upload image, get detections + bboxes
POST /api/detect/render    → PNG: upload image, get annotated image back

Usage
-----
    # 1. First train to produce a best.pt
    python src/train.py

    # 2. Then launch the API server
    python src/app.py --weights runs/tower_detection/yolo11_tower_detector/weights/best.pt --host 0.0.0.0 --port 5000

Then integrate from your website by POSTing multipart/form-data to
/api/detect with a file field called "image".
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from pathlib import Path

import yaml
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
)
from flask_cors import CORS

try:
    from .detector import AntennaTowerDetector
except ImportError:  # running as `python src/app.py` or gunicorn --chdir src
    from detector import AntennaTowerDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_DATASET_DIR = "cell tower antenna detection.v2i.yolov11"
BUNDLED_DATA_YAML = PROJECT_ROOT / "data.yaml"


def _load_dataset_meta():
    """Load class names and dataset structure from the bundled data.yaml."""
    meta = {
        "dataset_dir": BUNDLED_DATASET_DIR,
        "nc": 1,
        "class_names": ["antenna"],
        "splits": {"train": 0, "valid": 0, "test": 0},
    }
    if BUNDLED_DATA_YAML.exists():
        with open(BUNDLED_DATA_YAML, "r") as f:
            data = yaml.safe_load(f) or {}
        names = data.get("names", meta["class_names"])
        if isinstance(names, dict):
            meta["class_names"] = [names[k] for k in sorted(names.keys(), key=int)]
        else:
            meta["class_names"] = list(names)
        meta["nc"] = len(meta["class_names"])

        def count_imgs(split):
            p = PROJECT_ROOT / BUNDLED_DATASET_DIR / split / "images"
            return len([x for x in p.iterdir()]) if p.is_dir() else 0
        for s in ("train", "valid", "test"):
            meta["splits"][s] = count_imgs(s)
    return meta


DATASET_META = _load_dataset_meta()


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Antenna Tower Detector</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         max-width:900px;margin:40px auto;padding:0 20px;color:#1f2937}
    h1{margin-bottom:4px}
    .sub{color:#6b7280;margin-bottom:30px}
    .drop{border:2px dashed #9ca3af;border-radius:12px;padding:40px;text-align:center;
         background:#f9fafb;cursor:pointer;transition:.15s}
    .drop.hover{background:#eff6ff;border-color:#3b82f6}
    input[type=file]{display:none}
    .btn{display:inline-block;margin-top:16px;padding:10px 18px;border:0;
         background:#2563eb;color:#fff;border-radius:8px;cursor:pointer;font-weight:600}
    .btn:disabled{opacity:.6;cursor:not-allowed}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:24px}
    @media(max-width:720px){.row{grid-template-columns:1fr}}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;
         padding:18px;box-shadow:0 1px 2px rgba(0,0,0,.04);overflow:hidden}
    .card h3{margin-top:0}
    img{max-width:100%;height:auto;border-radius:8px}
    pre{background:#0f172a;color:#e2e8f0;padding:14px;border-radius:8px;
        overflow:auto;font-size:13px}
    label{display:block;margin-top:10px;font-size:13px;color:#374151}
    input[type=number], select{padding:6px 8px;border:1px solid #d1d5db;border-radius:6px;width:180px}
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;
          background:#dbeafe;color:#1e40af;font-size:12px;margin-right:6px}
  </style>
</head>
<body>
  <h1>📡 Antenna Tower Detector</h1>
  <p class="sub">YOLO11-powered object detection API. Model trained on Roboflow <em>Cell Tower Antenna Detection</em> dataset (1 class: <code>antenna</code>).</p>

  <div id="drop" class="drop">
    <strong>Drop an image here</strong> or click to browse
    <input id="file" type="file" accept="image/*"/>
  </div>

  <div style="margin-top:20px;display:flex;gap:20px;flex-wrap:wrap">
    <div><label>Confidence</label>
      <input id="conf" type="number" min="0.05" max="1" step="0.05" value="0.25"/></div>
    <div><label>IoU</label>
      <input id="iou"  type="number" min="0.05" max="1" step="0.05" value="0.45"/></div>
    <div><label>Image size</label>
      <select id="imgsz">
        <option>416</option><option selected>640</option><option>832</option><option>1024</option>
      </select>
    </div>
  </div>

  <div style="margin-top:16px">
    <button id="run" class="btn" disabled>Run detection</button>
    <span id="info" class="sub" style="margin-left:14px"></span>
  </div>

  <div class="row">
    <div class="card"><h3>Rendered</h3><div id="rendered"><em>no image</em></div></div>
    <div class="card"><h3>JSON response</h3><pre id="out">{ }</pre></div>
  </div>

<script>
  const drop = document.getElementById('drop');
  const file = document.getElementById('file');
  const run  = document.getElementById('run');
  const out  = document.getElementById('out');
  const info = document.getElementById('info');
  const rendered = document.getElementById('rendered');
  let selected = null;

  drop.addEventListener('click', () => file.click());
  drop.addEventListener('dragover', e => {e.preventDefault(); drop.classList.add('hover')});
  drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('hover');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });
  file.addEventListener('change', () => file.files[0] && setFile(file.files[0]));

  function setFile(f){ selected = f; run.disabled = false; info.textContent = f.name; }

  run.addEventListener('click', async () => {
    if(!selected) return;
    run.disabled = true;
    const fd = new FormData();
    fd.append('image', selected);
    fd.append('conf',  document.getElementById('conf').value);
    fd.append('iou',   document.getElementById('iou').value);
    fd.append('imgsz', document.getElementById('imgsz').value);

    try{
      const [jsonRes, renderRes] = await Promise.all([
        fetch('/api/detect', {method:'POST', body: fd}),
        fetch('/api/detect/render', {method:'POST', body: new FormData(fd)})
      ]);
      const j = await jsonRes.json();
      out.textContent = JSON.stringify(j, null, 2);
      if(renderRes.ok){
        const blob = await renderRes.blob();
        rendered.innerHTML = '';
        const img = document.createElement('img');
        img.src = URL.createObjectURL(blob);
        rendered.appendChild(img);
      }
    }catch(err){
      out.textContent = String(err);
    }finally{
      run.disabled = false;
    }
  });
</script>
</body>
</html>
"""


def create_app(
    weights_path: str,
    class_names: list[str] | None = None,
    device: str | int | None = None,
) -> Flask:
    if not os.path.exists(weights_path):
        print(f"ERROR: weights not found at {weights_path}")
        print("Train first: python src/train.py")
        sys.exit(1)

    app = Flask(__name__)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

    effective_class_names = class_names if class_names else DATASET_META["class_names"]
    detector = AntennaTowerDetector(
        weights_path, class_names=effective_class_names, device=device
    )
    app.detector = detector  # type: ignore[attr-defined]

    # Warm up lazily on first request
    @app.before_request
    def _warmup():
        if not getattr(app, "_detector_loaded", False):
            t0 = time.perf_counter()
            detector.load()
            app._detector_loaded = True  # type: ignore[attr-defined]
            print(f"Model loaded in { (time.perf_counter()-t0)*1000:.0f}ms")

    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "model": os.path.basename(weights_path),
            "loaded": getattr(app, "_detector_loaded", False),
            "dataset": {
                "name": DATASET_META["dataset_dir"],
                "nc": DATASET_META["nc"],
                "class_names": DATASET_META["class_names"],
                "splits": DATASET_META["splits"],
            },
        })

    @app.route("/api/info")
    def api_info():
        return jsonify({
            "project": "antenna-tower-detection",
            "model_arch": "YOLO11 (ultralytics)",
            "dataset": {
                "source": "Roboflow - Cell Tower Antenna Detection v2i",
                "dir": DATASET_META["dataset_dir"],
                "license": "CC BY 4.0",
                "nc": DATASET_META["nc"],
                "class_names": DATASET_META["class_names"],
                "splits": DATASET_META["splits"],
            },
            "api": {
                "POST /api/detect": "multipart/form-data {image: file, conf, iou, imgsz} -> JSON",
                "POST /api/detect/render": "same body -> image/png with boxes drawn",
                "GET  /health": "status + dataset info",
            },
        })

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    def _parse_params():
        try:
            conf  = float(request.form.get("conf", 0.25))
            iou   = float(request.form.get("iou", 0.45))
            imgsz = int(request.form.get("imgsz", 640))
        except ValueError:
            abort(400, "Invalid conf / iou / imgsz parameters")
        conf = max(0.01, min(0.99, conf))
        iou  = max(0.01, min(0.99, iou))
        imgsz = max(256, min(2048, imgsz))
        return conf, iou, imgsz

    def _read_image_file():
        if "image" not in request.files:
            abort(400, "Missing file field 'image' in multipart/form-data request")
        f = request.files["image"]
        if not f or f.filename == "":
            abort(400, "No file selected")
        return f

    @app.route("/api/detect", methods=["POST"])
    def api_detect():
        file = _read_image_file()
        conf, iou, imgsz = _parse_params()
        image_id = request.form.get("image_id") or secrets.token_hex(6)
        try:
            result = detector.predict(
                file.read(),
                image_id=image_id,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
            )
        except Exception as e:
            return jsonify({"error": f"inference failed: {e}"}), 500
        return jsonify(result.to_dict())

    @app.route("/api/detect/render", methods=["POST"])
    def api_detect_render():
        file = _read_image_file()
        conf, iou, imgsz = _parse_params()
        try:
            _result, png_bytes = detector.predict_and_render(
                file.read(), imgsz=imgsz, conf=conf, iou=iou,
            )
        except Exception as e:
            return jsonify({"error": f"inference failed: {e}"}), 500
        return Response(png_bytes, mimetype="image/png")

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Flask server for Antenna Tower Detection")
    p.add_argument("--weights", type=str,
                   default=os.environ.get("ATD_WEIGHTS", "runs/tower_detection/yolo11_tower_detector/weights/best.pt"))
    p.add_argument("--host", type=str, default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args.weights, device=args.device)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
