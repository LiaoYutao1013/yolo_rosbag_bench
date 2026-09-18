from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class TaskType(str, Enum):
    DETECT = "detect"
    POSE = "pose"
    SEGMENT = "segment"
    DEPTH = "depth"


@dataclass
class TimingStats:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def as_dict(self) -> dict[str, float]:
        return {
            "preprocess_ms": round(self.preprocess_ms, 4),
            "inference_ms": round(self.inference_ms, 4),
            "postprocess_ms": round(self.postprocess_ms, 4),
            "total_ms": round(self.total_ms, 4),
        }


@dataclass
class PredictionResult:
    task: TaskType
    image: np.ndarray
    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    class_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    names: dict[int, str] = field(default_factory=dict)
    keypoints: Optional[np.ndarray] = None  # [N, K, 3]
    skeleton: Optional[list[tuple[int, int]]] = None
    masks: Optional[np.ndarray] = None  # [N, H, W] bool
    depth_map: Optional[np.ndarray] = None  # [H, W] float32
    timing: TimingStats = field(default_factory=TimingStats)
    extra: dict[str, Any] = field(default_factory=dict)


class BaseYoloModel(ABC):
    """Unified YOLO model interface.

    Implementations are free to use PyTorch/Ultralytics, TensorRT, ONNX
    Runtime, or any other engine as long as they return a `PredictionResult`.
    """

    def __init__(self, model_path: str, task: TaskType, **kwargs: Any) -> None:
        self.model_path = model_path
        self.task = TaskType(task)
        self.kwargs = kwargs
        self.warmup_done = False

    @abstractmethod
    def load(self) -> None:
        """Load weights/engine and prepare the runtime."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> PredictionResult:
        """Run inference on a BGR or RGB uint8 image."""

    @abstractmethod
    def close(self) -> None:
        """Release all native resources."""

    def warmup(self, dummy_image: np.ndarray, iterations: int = 3) -> None:
        for _ in range(max(1, iterations)):
            self.predict(dummy_image)
        self.warmup_done = True

    def model_info(self) -> dict[str, Any]:
        """Return best-effort static model metadata (parameters, GFLOPs, etc.)."""
        return {}

    def __enter__(self) -> "BaseYoloModel":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
