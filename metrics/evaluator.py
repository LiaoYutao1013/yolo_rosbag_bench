from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from models.base import BaseYoloModel, PredictionResult

from .coco_metrics import DetectionEvaluator, GroundTruthRecord, load_yolo_ground_truth


def evaluate_detection_on_images(
    model: BaseYoloModel,
    image_paths: list[str],
    ground_truth_loader: Callable[[str], tuple[list[GroundTruthRecord], dict[int, str]]],
) -> dict[str, Any]:
    evaluator = DetectionEvaluator()
    all_names: dict[int, str] = {}
    total_inference_ms = 0.0
    frame_count = 0
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        gt, names = ground_truth_loader(image_path)
        all_names.update(names)
        image_id = Path(image_path).stem
        evaluator.add_ground_truth(image_id, np.array([g.box for g in gt], np.float64), np.array([g.class_id for g in gt], np.int64))
        result: PredictionResult = model.predict(image)
        evaluator.add_prediction(image_id, result.boxes, result.scores, result.class_ids)
        total_inference_ms += result.timing.total_ms
        frame_count += 1

    evaluator.class_names = all_names
    metric = evaluator.compute(iou_threshold=0.5)
    metric["mAP50"] = metric.get("mAP", 0.0)
    metric["mAP50-95"] = evaluator.compute_map50_95()
    metric["avg_latency_ms"] = total_inference_ms / frame_count if frame_count else 0.0
    metric["frames"] = frame_count
    return metric


def make_yolo_ground_truth_loader(
    labels_dir: str,
    image_dir: str,
    class_names: dict[int, str] | None = None,
) -> Callable[[str], tuple[list[GroundTruthRecord], dict[int, str]]]:
    labels_dir = Path(labels_dir)
    image_dir = Path(image_dir)

    def loader(image_path: str) -> tuple[list[GroundTruthRecord], dict[int, str]]:
        stem = Path(image_path).stem
        label_candidates = [
            labels_dir / f"{stem}.txt",
            labels_dir / (Path(image_path).with_suffix(".txt").name),
        ]
        image = cv2.imread(image_path)
        h, w = (image.shape[:2] if image is not None else (0, 0))
        for candidate in label_candidates:
            if candidate.exists():
                return load_yolo_ground_truth(str(candidate), w, h, class_names)
        return [], class_names or {}

    return loader
