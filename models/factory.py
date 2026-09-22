from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseYoloModel, TaskType
from .tensorrt_backend import TensorRtYoloModel, task_from_engine_metadata
from .ultralytics_backend import UltralyticsYoloModel
from .yolov5_backend import Yolov5Model, is_inside_yolov5_repo, is_yolov5_checkpoint


def create_model(
    model_path: str,
    task: TaskType,
    device: str | None = None,
    prefer_tensorrt: bool = False,
    **kwargs: Any,
) -> BaseYoloModel:
    task = TaskType(task)
    suffix = Path(model_path).suffix.lower()
    if suffix == ".engine":
        detected_task = task_from_engine_metadata(model_path)
        if detected_task is not None:
            task = detected_task
        return TensorRtYoloModel(model_path, task, device=device or "cuda", **kwargs)
    if suffix == ".pt" and task == TaskType.DETECT and (
        is_inside_yolov5_repo(model_path) or is_yolov5_checkpoint(model_path)
    ):
        return Yolov5Model(model_path, task, device=device, **kwargs)
    return UltralyticsYoloModel(model_path, task, device=device, **kwargs)
