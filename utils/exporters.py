from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _rows_to_csv(path: str, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_results(path: str, results: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        return str(path)
    if path.suffix.lower() == ".csv":
        rows = results.get("frames", [])
        if not rows and "metrics" in results:
            rows = [results["metrics"]]
        _rows_to_csv(str(path), rows)
        return str(path)
    raise ValueError("Only .json and .csv export formats are supported")


def export_records(path: str, records: list[dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"created": datetime.now().isoformat(), "records": records}, f, ensure_ascii=False, indent=2, default=str)
        return str(path)
    _rows_to_csv(str(path), records)
    return str(path)
