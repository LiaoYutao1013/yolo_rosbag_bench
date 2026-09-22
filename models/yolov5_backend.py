from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import get_adapter
from .base import BaseYoloModel, PredictionResult, TaskType, TimingStats


class Yolov5Model(BaseYoloModel):
    """YOLOv5-compatible backend.

    This backend intentionally does not use `ultralytics.YOLO`, which rejects
    YOLOv5 checkpoints.  It loads the checkpoint through YOLOv5's own
    `hubconf.py`, preferring a local YOLOv5 repository when the selected
    `.pt` file lives inside one (for example `TCone-YOLO`).
    """

    def __init__(self, model_path: str, task: TaskType, device: str | None = None, **kwargs: Any) -> None:
        super().__init__(model_path, task, **kwargs)
        self.device = device
        self.model: Any = None
        self.adapter = get_adapter(self.task)

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for YOLOv5 models") from exc

        warnings.filterwarnings(
            "ignore",
            message=".*torch\\.cuda\\.amp\\.autocast.*deprecated.*",
            category=FutureWarning,
        )

        repo = find_yolov5_repo(self.model_path)
        load_kwargs: dict[str, Any] = {
            "path": self.model_path,
            "autoshape": True,
            "device": self.device or None,
            "_verbose": False,
        }
        if repo:
            AutoShape, DetectMultiBackend, select_device, restore = self._import_local_yolov5(repo)
            try:
                selected_device = select_device(self.device or "")
                backend = DetectMultiBackend(self.model_path, device=selected_device, fuse=True)
                self.model = AutoShape(backend)
            finally:
                restore()
            return
        try:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", trust_repo=True, **load_kwargs)
        except TypeError:
            # Older PyTorch releases do not expose trust_repo.
            self.model = torch.hub.load("ultralytics/yolov5", "custom", **load_kwargs)

    def predict(self, image: np.ndarray) -> PredictionResult:
        if self.model is None:
            raise RuntimeError("YOLOv5 model is not loaded")
        image = np.ascontiguousarray(image)
        size = int(self.kwargs.get("imgsz", 640))

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*torch\\.cuda\\.amp\\.autocast.*deprecated.*",
                category=FutureWarning,
            )
            raw = self.model(image, size=size)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = raw.xyxy[0] if raw.xyxy is not None else None
        if isinstance(detections, list):
            detections = detections[0] if detections else None
        if detections is not None and hasattr(detections, "detach"):
            detections = detections.detach().cpu().numpy()
        elif detections is not None:
            detections = np.asarray(detections)

        if detections is not None and len(detections):
            boxes = np.asarray(detections[:, :4], dtype=np.float32)
            scores = np.asarray(detections[:, 4], dtype=np.float32)
            class_ids = np.asarray(detections[:, 5], dtype=np.int64)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int64)

        names = self.model.names if hasattr(self.model, "names") else {}
        if isinstance(names, (list, tuple)):
            names = {idx: str(name) for idx, name in enumerate(names)}
        else:
            names = {int(k): str(v) for k, v in dict(names).items()}

        result = PredictionResult(
            task=self.task,
            image=image,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            names=names,
            timing=TimingStats(preprocess_ms=0.0, inference_ms=float(inference_ms), postprocess_ms=0.0),
        )
        return result

    def model_info(self) -> dict[str, Any]:
        try:
            import torch

            if self.model is None:
                return {}
            parameters = int(sum(p.numel() for p in self.model.parameters()))
            return {"parameters": parameters, "GFLOPs": None}
        except Exception:
            return {}

    def close(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _find_yolov5_repo(self) -> str | None:
        return find_yolov5_repo(self.model_path)

    @staticmethod
    def _import_local_yolov5(repo: str) -> tuple[Any, Any, Any, Any]:
        """Import local YOLOv5 modules without permanently polluting sys.path.

        The YOLOv5 repository also has top-level packages named ``models`` and
        ``utils``.  Our application uses the same names, so the imports are done
        while those names are temporarily removed from ``sys.modules``.
        """
        saved_models = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "models" or name.startswith("models.")
        }
        saved_utils = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "utils" or name.startswith("utils.")
        }
        repo_added = repo not in sys.path
        if repo_added:
            sys.path.insert(0, repo)

        def _drop_conflicting_modules() -> None:
            for name in list(sys.modules):
                if (
                    name == "models"
                    or name.startswith("models.")
                    or name == "utils"
                    or name.startswith("utils.")
                ):
                    del sys.modules[name]

        def restore() -> None:
            if repo_added:
                try:
                    sys.path.remove(repo)
                except ValueError:
                    pass
            _drop_conflicting_modules()
            sys.modules.update(saved_models)
            sys.modules.update(saved_utils)

        _drop_conflicting_modules()

        try:
            from models.common import AutoShape, DetectMultiBackend
            from utils.torch_utils import select_device
        except Exception:
            restore()
            raise
        return AutoShape, DetectMultiBackend, select_device, restore


def find_yolov5_repo(model_path: str) -> str | None:
    path = Path(model_path).resolve()
    for parent in path.parents:
        hubconf = parent / "hubconf.py"
        if hubconf.exists() and "yolov5" in hubconf.read_text(encoding="utf-8", errors="ignore").lower():
            return str(parent)
    return None


def is_inside_yolov5_repo(model_path: str) -> bool:
    return find_yolov5_repo(model_path) is not None


def is_yolov5_checkpoint(model_path: str) -> bool:
    """Detect a YOLOv5 checkpoint without instantiating the model."""
    if is_inside_yolov5_repo(model_path):
        return True
    try:
        import torch
    except ImportError:
        return False
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if model is None:
        return False
    module = type(model).__module__
    return "ultralytics" not in module and ("models." in module or "yolov5" in module.lower())
