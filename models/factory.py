from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseYoloModel, TaskType
from .tensorrt_backend import TensorRtYoloModel
from .ultralytics_backend import UltralyticsYoloModel


def create_model(
    model_path: str,
    task: TaskType,
    device: str | None = None,
    prefer_tensorrt: bool = False,
    **kwargs: Any,
) -> BaseYoloModel:
    task = TaskType(task)
    suffix = Path(model_path).suffix.lower()
    if prefer_tensorrt and suffix == ".engine":
        return TensorRtYoloModel(model_path, task, device=device or "cuda", **kwargs)
    return UltralyticsYoloModel(model_path, task, device=device, **kwargs)
