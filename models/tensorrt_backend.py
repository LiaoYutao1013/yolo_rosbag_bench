from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseYoloModel, PredictionResult, TaskType


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
