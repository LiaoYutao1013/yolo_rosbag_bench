from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from models.base import TaskType


@dataclass
class AppConfig:
    model_path: str = ""
    task: TaskType = TaskType.DETECT
    device: str = ""
    source_mode: str = "bag"  # bag | live
    bag_path: str = ""
    bag_topics: list[str] = field(default_factory=list)
    live_topic: str = "/camera/color/image_raw"
    qos_depth: int = 1
    best_effort: bool = True
    realtime: bool = True
    play_rate: float = 1.0
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    frame_queue_size: int = 8
    labels_dir: str = ""
    image_dir: str = ""
    export_dir: str = "results"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task"] = self.task.value if hasattr(self.task, "value") else str(self.task)
        return data
