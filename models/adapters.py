from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

from .base import PredictionResult, TaskType


_COLORS = [
    (56, 56, 255),
    (0, 165, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
]


def _color(index: int) -> tuple[int, int, int]:
    return _COLORS[index % len(_COLORS)]


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


class BaseTaskAdapter(ABC):
    task_type: TaskType

    @abstractmethod
    def parse(self, raw: Any, image: np.ndarray) -> PredictionResult:
        """Convert engine-specific output into a `PredictionResult`."""

    def render(self, result: PredictionResult) -> np.ndarray:
        image = np.ascontiguousarray(result.image.copy())
        if result.task == TaskType.POSE:
            image = self._draw_boxes(image, result, label_kind="pose")
        else:
            image = self._draw_boxes(image, result, label_kind="detect")
        return image

    def _draw_boxes(self, image: np.ndarray, result: PredictionResult, label_kind: str) -> np.ndarray:
        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (int(v) for v in box)
            color = _color(int(result.class_ids[idx]))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            class_name = result.names.get(int(result.class_ids[idx]), str(int(result.class_ids[idx])))
            score = float(result.scores[idx]) if idx < len(result.scores) else 0.0
            label = f"{class_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return image


class DetectAdapter(BaseTaskAdapter):
    task_type = TaskType.DETECT

    def parse(self, raw: Any, image: np.ndarray) -> PredictionResult:
        boxes = _as_numpy(raw.boxes.xyxy) if raw.boxes is not None else np.empty((0, 4), np.float32)
        scores = _as_numpy(raw.boxes.conf) if raw.boxes is not None else np.empty((0,), np.float32)
        classes = _as_numpy(raw.boxes.cls).astype(np.int64) if raw.boxes is not None else np.empty((0,), np.int64)
        return PredictionResult(
            task=self.task_type,
            image=image,
            boxes=boxes,
            scores=scores,
            class_ids=classes,
            names=dict(raw.names or {}),
        )


class PoseAdapter(BaseTaskAdapter):
    task_type = TaskType.POSE

    def parse(self, raw: Any, image: np.ndarray) -> PredictionResult:
        boxes = _as_numpy(raw.boxes.xyxy) if raw.boxes is not None else np.empty((0, 4), np.float32)
        scores = _as_numpy(raw.boxes.conf) if raw.boxes is not None else np.empty((0,), np.float32)
        classes = _as_numpy(raw.boxes.cls).astype(np.int64) if raw.boxes is not None else np.empty((0,), np.int64)
        keypoints = None
        if raw.keypoints is not None:
            keypoints_data = getattr(raw.keypoints, "data", raw.keypoints)
            if keypoints_data is not None:
                keypoints = _as_numpy(keypoints_data)
        skeleton = getattr(raw, "skeleton", None)
        if skeleton is None and raw.names:
            # Use COCO skeleton if the model does not expose one.
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9),
                (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
            ]
        return PredictionResult(
            task=self.task_type,
            image=image,
            boxes=boxes,
            scores=scores,
            class_ids=classes,
            names=dict(raw.names or {}),
            keypoints=keypoints,
            skeleton=skeleton,
        )

    def render(self, result: PredictionResult) -> np.ndarray:
        image = super().render(result)
        if result.keypoints is None:
            return image
        skeleton = result.skeleton or []
        for person in result.keypoints:
            points = [(float(p[0]), float(p[1]), float(p[2])) for p in person]
            for a, b in skeleton:
                if a >= len(points) or b >= len(points):
                    continue
                if points[a][2] < 0.5 or points[b][2] < 0.5:
                    continue
                cv2.line(
                    image,
                    (int(points[a][0]), int(points[a][1])),
                    (int(points[b][0]), int(points[b][1])),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            for x, y, conf in points:
                if conf < 0.5:
                    continue
                cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
        return image


class SegmentAdapter(BaseTaskAdapter):
    task_type = TaskType.SEGMENT

    def parse(self, raw: Any, image: np.ndarray) -> PredictionResult:
        boxes = _as_numpy(raw.boxes.xyxy) if raw.boxes is not None else np.empty((0, 4), np.float32)
        scores = _as_numpy(raw.boxes.conf) if raw.boxes is not None else np.empty((0,), np.float32)
        classes = _as_numpy(raw.boxes.cls).astype(np.int64) if raw.boxes is not None else np.empty((0,), np.int64)
        masks = None
        if getattr(raw, "masks", None) is not None:
            masks_data = getattr(raw.masks, "data", raw.masks)
            if masks_data is not None:
                masks = _as_numpy(masks_data).astype(bool)
        return PredictionResult(
            task=self.task_type,
            image=image,
            boxes=boxes,
            scores=scores,
            class_ids=classes,
            names=dict(raw.names or {}),
            masks=masks,
        )

    def render(self, result: PredictionResult) -> np.ndarray:
        image = np.ascontiguousarray(result.image.copy())
        if result.masks is not None:
            overlay = image.copy()
            for idx, mask in enumerate(result.masks):
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0])) > 0
                color = _color(int(result.class_ids[idx]))
                overlay[mask] = color
            image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
        result.image = image
        return self._draw_boxes(image, result, label_kind="segment")


class DepthAdapter(BaseTaskAdapter):
    task_type = TaskType.DEPTH

    def parse(self, raw: Any, image: np.ndarray) -> PredictionResult:
        depth = None
        if hasattr(raw, "depth") and raw.depth is not None:
            depth = raw.depth.cpu().numpy() if hasattr(raw.depth, "cpu") else np.asarray(raw.depth, np.float32)
        elif hasattr(raw, "masks") and raw.masks is not None:
            depth = raw.masks.data.cpu().numpy().astype(np.float32)
        elif isinstance(raw, np.ndarray):
            depth = np.asarray(raw, np.float32)
        if depth is not None and depth.ndim == 3:
            depth = depth.squeeze(0)
        return PredictionResult(task=self.task_type, image=image, depth_map=depth)

    def render(self, result: PredictionResult) -> np.ndarray:
        if result.depth_map is None:
            return result.image.copy()
        depth = result.depth_map.astype(np.float32)
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            return result.image.copy()
        dmin, dmax = float(finite.min()), float(finite.max())
        if dmax - dmin < 1e-8:
            norm = np.zeros_like(depth, dtype=np.uint8)
        else:
            norm = np.clip((depth - dmin) / (dmax - dmin) * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        if color.shape[:2] != result.image.shape[:2]:
            color = cv2.resize(color, (result.image.shape[1], result.image.shape[0]))
        return cv2.addWeighted(result.image, 0.35, color, 0.65, 0)


ADAPTERS: dict[TaskType, type[BaseTaskAdapter]] = {
    TaskType.DETECT: DetectAdapter,
    TaskType.POSE: PoseAdapter,
    TaskType.SEGMENT: SegmentAdapter,
    TaskType.DEPTH: DepthAdapter,
}


def get_adapter(task: TaskType) -> BaseTaskAdapter:
    return ADAPTERS[task]()
