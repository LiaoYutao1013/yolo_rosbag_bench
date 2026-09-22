from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import yaml


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {str(k): v for k, v in data.items()}
    except Exception:
        return {}


def _find_training_dir(model_path: str) -> Path | None:
    path = Path(model_path).resolve()
    candidates: list[Path] = []
    if path.is_file():
        candidates.append(path.parent)
        candidates.append(path.parent.parent)
        candidates.append(path.parent.parent.parent)
    candidates.extend(path.parents)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "results.csv").exists() or (candidate / "results.txt").exists():
            return candidate
    return None


def _parse_yolov5_results_csv(path: Path) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                clean_key = key.strip()
                numeric = _to_float(value)
                if numeric is not None:
                    row[clean_key] = numeric
            if row:
                rows.append(row)
    if not rows:
        return {}
    def best_value(column: str) -> float:
        return max((row.get(column, -1.0) for row in rows), default=0.0)

    final = rows[-1]
    best_map_row = max(rows, key=lambda row: row.get("metrics/mAP_0.5", -1.0))
    result: dict[str, Any] = {
        "precision": round(best_value("metrics/precision"), 4),
        "recall": round(best_value("metrics/recall"), 4),
        "mAP50": round(best_value("metrics/mAP_0.5"), 4),
        "mAP50-95": round(best_value("metrics/mAP_0.5:0.95"), 4),
        "best_epoch": int(best_map_row.get("epoch", 0)) + 1,
        "train/box_loss": round(final.get("train/box_loss", 0.0), 4),
        "train/obj_loss": round(final.get("train/obj_loss", 0.0), 4),
        "train/cls_loss": round(final.get("train/cls_loss", 0.0), 4),
        "val/box_loss": round(final.get("val/box_loss", 0.0), 4),
        "val/obj_loss": round(final.get("val/obj_loss", 0.0), 4),
        "val/cls_loss": round(final.get("val/cls_loss", 0.0), 4),
    }
    return result


def _parse_results_txt(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    result: dict[str, Any] = {}
    summary = re.search(r"(\d+)\s+parameters", text)
    if summary:
        result["参数量(M)"] = round(int(summary.group(1)) / 1_000_000, 2)
    flops = re.search(r"([0-9.]+)\s+GFLOPs", text)
    if flops:
        result["GFLOPs"] = round(float(flops.group(1)), 2)
    all_match = re.search(r"\ball\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text)
    if all_match:
        result["val_images"] = int(all_match.group(1))
        result["val_instances"] = int(all_match.group(2))
        result["all_precision"] = round(float(all_match.group(3)), 4)
        result["all_recall"] = round(float(all_match.group(4)), 4)
        result["all_mAP50"] = round(float(all_match.group(5)), 4)
        result["all_mAP50-95"] = round(float(all_match.group(6)), 4)
    return result


def load_dynamic_model_metrics(model_path: str) -> dict[str, Any]:
    """Load metrics from the training run beside the selected weights.

    No values are hardcoded here.  YOLOv5 runs expose results.csv, opt.yaml and
    results.txt in the same run directory as weights/best.pt.
    """
    training_dir = _find_training_dir(model_path)
    if training_dir is None:
        return {}

    metrics: dict[str, Any] = {"run_dir": str(training_dir)}

    results_csv = training_dir / "results.csv"
    if results_csv.exists():
        csv_metrics = _parse_yolov5_results_csv(results_csv)
        metrics.update(csv_metrics)

    results_txt = training_dir / "results.txt"
    if results_txt.exists():
        metrics.update(_parse_results_txt(results_txt))

    opt_yaml = training_dir / "opt.yaml"
    if not opt_yaml.exists():
        opt_yaml = training_dir / "args.yaml"
    opt = _read_yaml(opt_yaml) if opt_yaml.exists() else {}
    if opt:
        metrics["Batchsize"] = opt.get("batch_size", opt.get("batch", "N/A"))
        metrics["workers"] = opt.get("workers", "N/A")
        metrics["Epochs"] = opt.get("epochs", "N/A")
        cfg = str(opt.get("cfg", opt.get("model", "")))
        if cfg:
            metrics["配置"] = cfg
        hyp = opt.get("hyp") if isinstance(opt.get("hyp"), dict) else {}
        if hyp:
            metrics["iou_loss"] = hyp.get("iou_loss", "")
            metrics["hsv_h"] = hyp.get("hsv_h", "")
            metrics["hsv_s"] = hyp.get("hsv_s", "")
            metrics["hsv_v"] = hyp.get("hsv_v", "")

    if not metrics.get("mAP50"):
        metrics["mAP50"] = metrics.get("all_mAP50")
    if not metrics.get("mAP50-95"):
        metrics["mAP50-95"] = metrics.get("all_mAP50-95")
    if not metrics.get("precision"):
        metrics["precision"] = metrics.get("all_precision")
    if not metrics.get("recall"):
        metrics["recall"] = metrics.get("all_recall")
    return metrics
