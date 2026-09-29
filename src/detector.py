"""Thin service wrapper around the YOLO11 Antenna Tower detector.

This module exposes pure functions / a Detector class that can be imported
from your web framework (Flask, FastAPI, Django, etc.) so that the inference
logic is decoupled from the HTTP layer.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    image_id: str
    detections: List[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    inference_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_ms": round(self.inference_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
            "counts": self._counts(),
        }

    def _counts(self) -> dict:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class AntennaTowerDetector:
    """Loads a YOLO11 checkpoint and runs antenna tower detection."""

    def __init__(
        self,
        weights_path: str,
        class_names: Optional[List[str]] = None,
        device: Optional[Union[str, int]] = None,
    ) -> None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found: {weights_path}. "
                "Train a model first or point to a valid best.pt file."
            )
        self.weights_path = weights_path
        self.class_names = class_names or []
        self.device = self._resolve_device(device)
        self._model: Optional[YOLO] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(requested):
        if requested is not None:
            return requested
        try:
            import torch
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return "cpu"

    def load(self) -> "AntennaTowerDetector":
        """Lazily load the model (call once before predict())."""
        if self._model is None:
            self._model = YOLO(self.weights_path)
            if not self.class_names:
                self.class_names = list(self._model.names.values()) if hasattr(self._model, "names") else []
        return self

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self.load()
        return self._model

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(source: Union[str, bytes, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if isinstance(source, str):
            if os.path.exists(source):
                return Image.open(source).convert("RGB")
            raise FileNotFoundError(f"Image path does not exist: {source}")
        raise TypeError(f"Unsupported image source type: {type(source)}")

    def predict(
        self,
        image_source: Union[str, bytes, Image.Image, np.ndarray],
        image_id: str = "image",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> DetectionResult:
        """Run detection on a single image."""
        pil_image = self._load_image(image_source)
        width, height = pil_image.size
        t0 = 0.0

        import time
        t0 = time.perf_counter()
        results = self.model.predict(
            source=pil_image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        if not results:
            return DetectionResult(image_id, [], width, height, inference_ms)

        result = results[0]
        names = self.class_names or (
            list(result.names.values()) if hasattr(result, "names") else []
        )

        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf_val = float(boxes.conf[i])
                xyxy = list(map(int, boxes.xyxy[i].tolist()))
                cls_name = (
                    names[cls_id] if 0 <= cls_id < len(names) else f"class_{cls_id}"
                )
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=round(conf_val, 4),
                        bbox=xyxy,
                    )
                )

        return DetectionResult(
            image_id=image_id,
            detections=detections,
            image_width=width,
            image_height=height,
            inference_ms=inference_ms,
        )

    # ------------------------------------------------------------------
    # Visualization — returns PNG bytes with boxes drawn on top
    # ------------------------------------------------------------------
    def predict_and_render(
        self,
        image_source: Union[str, bytes, Image.Image, np.ndarray],
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> tuple[DetectionResult, bytes]:
        """Run detection and return (result, rendered_png_bytes)."""
        pil_image = self._load_image(image_source)

        result_obj = self.predict(
            pil_image, image_id="rendered", imgsz=imgsz, conf=conf, iou=iou
        )

        import cv2
        import random

        # Build color palette per class
        palette = {}
        for d in result_obj.detections:
            if d.class_name not in palette:
                rng = random.Random(hash(d.class_name) & 0xFFFFFFFF)
                palette[d.class_name] = (
                    rng.randint(50, 255),
                    rng.randint(50, 255),
                    rng.randint(50, 255),
                )

        arr = np.array(pil_image)[:, :, ::-1].copy()  # RGB -> BGR
        for d in result_obj.detections:
            x1, y1, x2, y2 = d.bbox
            color = palette[d.class_name]
            cv2.rectangle(arr, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(arr, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                arr, label, (x1 + 2, max(y1 - 4, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError("Failed to encode rendered image as PNG")
        return result_obj, buf.tobytes()
