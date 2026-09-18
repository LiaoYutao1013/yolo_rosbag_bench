from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class DetectionRecord:
    image_id: Any
    box: np.ndarray  # xyxy
    score: float
    class_id: int


@dataclass
class GroundTruthRecord:
    image_id: Any
    box: np.ndarray  # xyxy
    class_id: int


@dataclass
class KeypointRecord:
    image_id: Any
    keypoints: np.ndarray  # [K, 3], x, y, visibility
    class_id: int

    @property
    def area(self) -> float:
        visible = self.keypoints[self.keypoints[:, 2] > 0]
        if visible.shape[0] < 2:
            return 0.0
        xmin, ymin = visible[:, 0].min(), visible[:, 1].min()
        xmax, ymax = visible[:, 0].max(), visible[:, 1].max()
        return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def _ap_from_tp_fp(tp: list[int], fp: list[int], num_gt: int) -> float:
    if num_gt == 0:
        return 0.0 if not tp else 1.0
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, num_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        p = float(precisions[recalls >= t].max()) if np.any(recalls >= t) else 0.0
        ap += p
    return ap / 101.0


class DetectionEvaluator:
    """A lightweight COCO-style detector evaluator.

    It is not a byte-for-byte COCO API implementation, but follows the same
    per-class, per-IoU matching and 101-point AP interpolation, which is enough
    for comparing model versions in this tool.
    """

    def __init__(self, class_names: dict[int, str] | None = None) -> None:
        self.class_names = class_names or {}
        self.predictions: list[DetectionRecord] = []
        self.ground_truth: list[GroundTruthRecord] = []

    def add_prediction(self, image_id: Any, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> None:
        for box, score, class_id in zip(boxes, scores, class_ids):
            self.predictions.append(
                DetectionRecord(image_id=image_id, box=np.asarray(box, np.float64), score=float(score), class_id=int(class_id))
            )

    def add_ground_truth(self, image_id: Any, boxes: np.ndarray, class_ids: np.ndarray) -> None:
        for box, class_id in zip(boxes, class_ids):
            self.ground_truth.append(
                GroundTruthRecord(image_id=image_id, box=np.asarray(box, np.float64), class_id=int(class_id))
            )

    def compute(self, iou_threshold: float = 0.5) -> dict[str, Any]:
        preds_by_class: dict[int, list[DetectionRecord]] = defaultdict(list)
        for p in self.predictions:
            preds_by_class[p.class_id].append(p)
        gt_by_class: dict[int, list[GroundTruthRecord]] = defaultdict(list)
        for g in self.ground_truth:
            gt_by_class[g.class_id].append(g)

        class_aps: dict[int, float] = {}
        all_tp: list[int] = []
        all_fp: list[int] = []
        total_gt = len(self.ground_truth)

        for class_id, gt_items in gt_by_class.items():
            class_preds = sorted(preds_by_class.get(class_id, []), key=lambda x: x.score, reverse=True)
            image_gts: dict[Any, list[GroundTruthRecord]] = defaultdict(list)
            for g in gt_items:
                image_gts[g.image_id].append(g)
            matched: dict[Any, list[bool]] = {img: [False] * len(gts) for img, gts in image_gts.items()}
            tp: list[int] = []
            fp: list[int] = []
            for p in class_preds:
                candidates = image_gts.get(p.image_id, [])
                best_iou = 0.0
                best_idx = -1
                for idx, g in enumerate(candidates):
                    if matched[p.image_id][idx]:
                        continue
                    value = iou(p.box, g.box)
                    if value > best_iou:
                        best_iou = value
                        best_idx = idx
                if best_idx >= 0 and best_iou >= iou_threshold:
                    matched[p.image_id][best_idx] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            all_tp.extend(tp)
            all_fp.extend(fp)
            class_aps[class_id] = _ap_from_tp_fp(tp, fp, len(gt_items))

        # Micro-averaged precision/recall at the conventional score threshold 0.5.
        tp_at_half = sum(
            1 for p in self.predictions if p.score >= 0.5 and any(
                g.image_id == p.image_id and g.class_id == p.class_id and iou(p.box, g.box) >= iou_threshold
                for g in self.ground_truth
            )
        )
        pred_at_half = sum(1 for p in self.predictions if p.score >= 0.5)
        precision = tp_at_half / pred_at_half if pred_at_half else 0.0
        recall = tp_at_half / total_gt if total_gt else 0.0
        mAP = float(np.mean(list(class_aps.values()))) if class_aps else 0.0
        return {
            "mAP": mAP,
            "precision": precision,
            "recall": recall,
            "per_class_ap": {str(k): v for k, v in class_aps.items()},
            "class_names": self.class_names,
            "num_gt": total_gt,
            "num_pred": len(self.predictions),
        }

    def compute_map50_95(self) -> float:
        values = [self.compute(iou_threshold=t)["mAP"] for t in np.arange(0.5, 1.0, 0.05)]
        return float(np.mean(values)) if values else 0.0


def load_coco_ground_truth(path: str) -> tuple[dict[Any, list[GroundTruthRecord]], dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names: dict[int, str] = {}
    for cat in data.get("categories", []):
        names[int(cat["id"])] = cat.get("name", str(cat["id"]))
    image_sizes: dict[Any, tuple[int, int]] = {}
    for image in data.get("images", []):
        image_sizes[image["id"]] = (int(image["width"]), int(image["height"]))
    gt: dict[Any, list[GroundTruthRecord]] = defaultdict(list)
    for ann in data.get("annotations", []):
        x, y, w, h = (float(ann["bbox"][i]) for i in range(4))
        gt[ann["image_id"]].append(
            GroundTruthRecord(image_id=ann["image_id"], box=np.array([x, y, x + w, y + h], np.float64), class_id=int(ann["category_id"]))
        )
    return dict(gt), names


def load_coco_image_name_map(path: str) -> dict[str, Any]:
    """Return a mapping from image file name to COCO image id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result: dict[str, Any] = {}
    for image in data.get("images", []):
        name = image.get("file_name") or f"{image.get('id')}.jpg"
        result[Path(name).name] = image["id"]
    return result


def load_coco_pose_ground_truth(path: str) -> dict[Any, list[KeypointRecord]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt: dict[Any, list[KeypointRecord]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if "keypoints" not in ann:
            continue
        kp = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        if kp.size:
            gt[ann["image_id"]].append(
                KeypointRecord(image_id=ann["image_id"], keypoints=kp, class_id=int(ann.get("category_id", 0)))
            )
    return dict(gt)


def oks(pred: np.ndarray, gt: np.ndarray, sigmas: Iterable[float] | None = None) -> float:
    """Object Keypoint Similarity for a pair of [K, 3] keypoint arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    k = max(pred.shape[0], gt.shape[0])
    if pred.shape[0] != gt.shape[0]:
        return 0.0
    if sigmas is None:
        sigmas = [0.26] * k
    sigmas = list(sigmas)[:k]
    vis = gt[:, 2] > 0
    if not np.any(vis):
        return 0.0
    delta = pred[vis, :2] - gt[vis, :2]
    area = float(np.prod(np.ptp(gt[vis, :2], axis=0))) or 1.0
    s2 = np.array(sigmas[:k], dtype=np.float64)[vis] ** 2
    e = np.sum((delta**2).sum(axis=1) / (2.0 * s2 * area))
    return float(np.exp(-e))


class PoseEvaluator:
    """Approximate pose evaluator using OKS instead of box IoU matching."""

    def __init__(self, sigmas: Iterable[float] | None = None) -> None:
        self.sigmas = list(sigmas) if sigmas is not None else None
        self.ground_truth: list[KeypointRecord] = []
        self.predictions: list[KeypointRecord] = []

    def add_ground_truth(self, image_id: Any, keypoints: np.ndarray, class_id: int = 0) -> None:
        if keypoints.size:
            self.ground_truth.append(KeypointRecord(image_id, keypoints, class_id))

    def add_prediction(self, image_id: Any, keypoints: np.ndarray, class_id: int = 0) -> None:
        if keypoints.size:
            self.predictions.append(KeypointRecord(image_id, keypoints, class_id))

    def compute(self, oks_threshold: float = 0.5) -> dict[str, Any]:
        if not self.ground_truth:
            return {"mOKS": 0.0, "matched": 0, "gt": 0, "oks": []}
        # Greedy global matching by descending OKS.
        pairs: list[tuple[float, int, int]] = []
        for gi, gt in enumerate(self.ground_truth):
            for pi, pred in enumerate(self.predictions):
                if pred.image_id != gt.image_id or pred.class_id != gt.class_id:
                    continue
                value = oks(pred.keypoints, gt.keypoints, self.sigmas)
                if value >= oks_threshold:
                    pairs.append((value, gi, pi))
        pairs.sort(reverse=True)
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        oks_values: list[float] = []
        for value, gi, pi in pairs:
            if gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            oks_values.append(value)
        mean_oks = sum(oks_values) / len(self.ground_truth)
        return {
            "mOKS": mean_oks,
            "matched": len(oks_values),
            "gt": len(self.ground_truth),
            "oks": oks_values,
        }


def load_yolo_ground_truth(
    label_path: str,
    image_width: int,
    image_height: int,
    class_names: dict[int, str] | None = None,
) -> tuple[list[GroundTruthRecord], dict[int, str]]:
    records: list[GroundTruthRecord] = []
    names = class_names or {}
    path = Path(label_path)
    if not path.exists():
        return records, names
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        xc, yc, w, h = (float(x) for x in parts[1:5])
        x1 = (xc - w / 2.0) * image_width
        y1 = (yc - h / 2.0) * image_height
        x2 = (xc + w / 2.0) * image_width
        y2 = (yc + h / 2.0) * image_height
        records.append(GroundTruthRecord(image_id=label_path, box=np.array([x1, y1, x2, y2], np.float64), class_id=class_id))
    return records, names
