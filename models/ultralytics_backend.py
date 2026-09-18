from __future__ import annotations

import time
from typing import Any

import numpy as np

from .adapters import get_adapter
from .base import BaseYoloModel, PredictionResult, TaskType, TimingStats


class UltralyticsYoloModel(BaseYoloModel):
    """PyTorch/Ultralytics backend for .pt, .onnx, .engine and exported models."""

    def __init__(self, model_path: str, task: TaskType, device: str | None = None, **kwargs: Any) -> None:
        super().__init__(model_path, task, **kwargs)
        self.device = device
        self.model: Any = None
        self.adapter = get_adapter(self.task)

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics") from exc

        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        # Ultralytics accepts .pt/.onnx/.engine through the same YOLO entry point.
        self.model = YOLO(self.model_path, task=self.task.value if self.task != TaskType.DEPTH else "detect")

    def predict(self, image: np.ndarray) -> PredictionResult:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        image = np.ascontiguousarray(image)

        t0 = time.perf_counter()
        # Ultralytics can accept a numpy BGR image directly.
        predict_kwargs: dict[str, Any] = {
            "source": image,
            "verbose": False,
            "conf": self.kwargs.get("conf", 0.25),
            "iou": self.kwargs.get("iou", 0.45),
            "imgsz": self.kwargs.get("imgsz", 640),
        }
        if self.device:
            predict_kwargs["device"] = self.device
        raw = self.model.predict(**predict_kwargs)[0]
        pre_ms = raw.speed.get("preprocess", 0.0) or 0.0
        infer_ms = raw.speed.get("inference", 0.0) or 0.0
        post_ms = raw.speed.get("postprocess", 0.0) or 0.0

        # Ultralytics speed is already ms; fall back to wall-clock split if absent.
        if pre_ms == 0.0 and infer_ms == 0.0 and post_ms == 0.0:
            infer_ms = (time.perf_counter() - t0) * 1000.0

        result = self.adapter.parse(raw, image)
        result.timing = TimingStats(
            preprocess_ms=float(pre_ms),
            inference_ms=float(infer_ms),
            postprocess_ms=float(post_ms),
        )
        return result

    def model_info(self) -> dict[str, Any]:
        if self.model is None:
            return {}
        try:
            info = self.model.info(detailed=True, verbose=False)
            return dict(info) if isinstance(info, dict) else {}
        except Exception:
            return {}

    def close(self) -> None:
        self.model = None
