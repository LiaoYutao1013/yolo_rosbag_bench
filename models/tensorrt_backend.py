from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from .base import BaseYoloModel, PredictionResult, TaskType
from .adapters import get_adapter


def read_ultralytics_engine_metadata(model_path: str) -> dict[str, Any]:
    """Read the JSON metadata prefix written by Ultralytics TensorRT export."""
    try:
        with open(model_path, "rb") as f:
            prefix = f.read(4)
            if len(prefix) < 4:
                return {}
            metadata_len = struct.unpack("<I", prefix)[0]
            if metadata_len <= 0 or metadata_len > 16 * 1024 * 1024:
                return {}
            metadata = json.loads(f.read(metadata_len).decode("utf-8"))
            return metadata if isinstance(metadata, dict) else {}
    except Exception:
        return {}


def task_from_engine_metadata(model_path: str) -> TaskType | None:
    metadata = read_ultralytics_engine_metadata(model_path)
    task_name = str(metadata.get("task", "")).lower()
    if not task_name:
        return None
    try:
        return TaskType(task_name)
    except ValueError:
        return None


class TensorRtYoloModel(BaseYoloModel):
    """Optional TensorRT backend.

    For YOLO models exported by `ultralytics`, the simplest supported route is
    to let Ultralytics load the `.engine` file via `YOLO("model.engine")`.  This
    class is intentionally kept explicit so advanced users can replace the
    inference implementation with raw TensorRT bindings while preserving the
    unified `predict(image)` contract.
    """

    def __init__(self, model_path: str, task: TaskType, device: str = "cuda", **kwargs: Any) -> None:
        super().__init__(model_path, task, **kwargs)
        self.device = device
        self._backend: BaseYoloModel | None = None
        self.adapter = get_adapter(self.task)

    def load(self) -> None:
        try:
            from .ultralytics_backend import UltralyticsYoloModel
        except ImportError as exc:
            raise RuntimeError("ultralytics backend is unavailable") from exc
        self._backend = UltralyticsYoloModel(self.model_path, self.task, device=self.device, **self.kwargs)
        self._backend.load()

    def predict(self, image: np.ndarray) -> PredictionResult:
        if self._backend is None:
            raise RuntimeError("TensorRT backend is not loaded")
        return self._backend.predict(image)

    def model_info(self) -> dict[str, Any]:
        return self._backend.model_info() if self._backend is not None else {}

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
        self._backend = None
