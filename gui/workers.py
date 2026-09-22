from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from PySide6.QtCore import QObject, QThread, Signal
except ImportError:  # pragma: no cover - allows PyQt6 fallback
    from PyQt6.QtCore import QObject, QThread, Signal

from bag_reader.ros_bridge_client import RosBridgeClient
from metrics.performance_monitor import FrameRateMeter, LatencyTracker, SystemMonitor
from models.base import BaseYoloModel, PredictionResult, TaskType
from models.factory import create_model
from utils.logger import setup_logger


logger = setup_logger("yolo_bench.workers")


@dataclass
class InferenceFrame:
    timestamp_ns: int
    topic: str
    raw: np.ndarray
    rendered: np.ndarray
    result: PredictionResult
    fps: float = 0.0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    system: dict[str, float] = field(default_factory=dict)
    frame_index: int = 0

    def as_record(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "topic": self.topic,
            "frame_index": self.frame_index,
            "fps": round(self.fps, 2),
            "latency_ms": round(self.latency_ms, 3),
            "jitter_ms": round(self.jitter_ms, 3),
            **{k: round(v, 3) for k, v in self.system.items()},
            **self.result.timing.as_dict(),
            "num_objects": int(len(self.result.boxes)),
        }


class BagPlayerWorker(QThread):
    frame_ready = Signal(object)  # tuple(timestamp_ns, topic, np.ndarray)
    progress_changed = Signal(float)  # -1 when indeterminate
    finished = Signal()
    error = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        bag_path: str,
        frame_queue: queue.Queue,
        topics: Optional[list[str]] = None,
        realtime: bool = True,
        rate: float = 1.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.bag_path = bag_path
        self.frame_queue = frame_queue
        self.topics = topics or []
        self.realtime = realtime
        self.rate = rate
        self._client: Optional[RosBridgeClient] = None
        self._finished_event = threading.Event()
        self._stop_requested = False
        self._pause_requested = False
        self._seek_requested: Optional[float] = None
        self._rate = max(0.05, rate)

    def stop(self) -> None:
        self._stop_requested = True
        if self._client is not None:
            self._client.stop()

    def pause(self) -> None:
        self._pause_requested = True
        if self._client is not None:
            self._client.pause()

    def resume(self) -> None:
        self._pause_requested = False
        if self._client is not None:
            self._client.resume()

    def set_rate(self, rate: float) -> None:
        self._rate = max(0.05, rate)
        if self._client is not None:
            self._client.set_rate(self._rate)

    def seek(self, ratio: float) -> None:
        self._seek_requested = min(1.0, max(0.0, ratio))
        if self._client is not None:
            self._client.seek(self._seek_requested)

    def run(self) -> None:
        self._finished_event.clear()
        client = RosBridgeClient(
            on_frame=self._handle_bridge_frame,
            on_progress=lambda ratio: self.progress_changed.emit(ratio),
            on_status=lambda message: self.status.emit(message),
            on_error=self._handle_bridge_error,
            on_finished=self._finished_event.set,
        )
        self._client = client
        try:
            self.status.emit("Starting ROS bridge process...")
            client.start()
            client.open_bag(self.bag_path, self.topics, self.realtime, self._rate)
            if self._pause_requested:
                client.pause()
            if self._seek_requested is not None:
                client.seek(self._seek_requested)
                self._seek_requested = None
            while not self._stop_requested and not self._finished_event.wait(0.1):
                if not client.is_running:
                    detail = client.stderr_text.strip()
                    message = "ROS bridge process exited unexpectedly"
                    if detail:
                        message += f"\n{detail}"
                    self.error.emit(message)
                    break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Bag player failed")
            self.error.emit(str(exc))
        finally:
            client.close()
            self._client = None
            self.finished.emit()

    def _handle_bridge_frame(self, timestamp_ns: int, topic: str, frame: np.ndarray) -> None:
        if self._stop_requested:
            return
        try:
            self.frame_queue.put(
                (timestamp_ns, topic, frame),
                timeout=0.5,
            )
        except queue.Full:
            logger.warning("Frame queue is full; dropping bag frame")
        self.frame_ready.emit((timestamp_ns, topic, frame))

    def _handle_bridge_error(self, message: str) -> None:
        if not self._stop_requested:
            self.error.emit(message)
        self._finished_event.set()


class InferenceWorker(QThread):
    frame_processed = Signal(object)  # InferenceFrame
    model_loaded = Signal(str)
    error = Signal(str)
    status = Signal(str)
    finished = Signal()

    def __init__(
        self,
        frame_queue: queue.Queue,
        model_path: str,
        task: TaskType,
        device: str = "",
        conf: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.frame_queue = frame_queue
        self.model_path = model_path
        self.task = TaskType(task)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.image_size = image_size
        self._stop_requested = False
        self.model: Optional[BaseYoloModel] = None
        self._fps_meter = FrameRateMeter()
        self._latency_tracker = LatencyTracker()
        self._system_monitor: Optional[SystemMonitor] = None
        self._frame_index = 0
        self._paused = threading.Event()

    def stop(self) -> None:
        self._stop_requested = True
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def run(self) -> None:
        try:
            self.status.emit("Loading model...")
            model = create_model(
                self.model_path,
                self.task,
                device=self.device or None,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.image_size,
            )
            model.load()
            dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            model.warmup(dummy, iterations=2)
            self.model = model
            self._system_monitor = SystemMonitor()
            self.model_loaded.emit(self.model_path)
            self.status.emit("Inference running")

            while not self._stop_requested:
                try:
                    item = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if self._paused.is_set():
                    continue
                timestamp_ns, topic, frame = item
                self._frame_index += 1
                start = time.monotonic()
                result = model.predict(frame)
                rendered = model.adapter.render(result)
                wall_ms = (time.monotonic() - start) * 1000.0
                latency = result.timing.total_ms if result.timing.total_ms > 0 else wall_ms
                fps = self._fps_meter.update()
                self._latency_tracker.update(latency)
                system = self._system_monitor.sample() if self._system_monitor else {}
                payload = InferenceFrame(
                    timestamp_ns=timestamp_ns,
                    topic=topic,
                    raw=frame,
                    rendered=rendered,
                    result=result,
                    fps=fps,
                    latency_ms=latency,
                    jitter_ms=self._fps_meter.jitter_ms,
                    system=system,
                    frame_index=self._frame_index,
                )
                self.frame_processed.emit(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference worker failed")
            self.error.emit(str(exc))
        finally:
            try:
                if self.model is not None:
                    self.model.close()
            except Exception:
                logger.exception("Error while closing model")
            if self._system_monitor is not None:
                self._system_monitor.close()
            self.finished.emit()


class RosSubscriberWorker(QThread):
    frame_ready = Signal(object)
    error = Signal(str)
    status = Signal(str)
    finished = Signal()

    def __init__(
        self,
        topic: str,
        frame_queue: queue.Queue,
        qos_depth: int = 1,
        best_effort: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.topic = topic
        self.frame_queue = frame_queue
        self.qos_depth = qos_depth
        self.best_effort = best_effort
        self._client: Optional[RosBridgeClient] = None
        self._finished_event = threading.Event()
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True
        if self._client is not None:
            self._client.stop()

    def run(self) -> None:
        self._finished_event.clear()
        client = RosBridgeClient(
            on_frame=self._handle_bridge_frame,
            on_status=lambda message: self.status.emit(message),
            on_error=self._handle_bridge_error,
            on_finished=self._finished_event.set,
        )
        self._client = client
        try:
            self.status.emit(f"Subscribing to {self.topic}...")
            client.start()
            client.subscribe(self.topic, self.qos_depth, self.best_effort)
            while not self._stop_requested and not self._finished_event.wait(0.1):
                if not client.is_running:
                    detail = client.stderr_text.strip()
                    message = "ROS bridge process exited unexpectedly"
                    if detail:
                        message += f"\n{detail}"
                    self.error.emit(message)
                    break
        except Exception as exc:  # noqa: BLE001
            logger.exception("ROS subscriber worker failed")
            self.error.emit(str(exc))
        finally:
            client.close()
            self._client = None
            self.finished.emit()

    def _handle_bridge_frame(self, timestamp_ns: int, topic: str, frame: np.ndarray) -> None:
        if self._stop_requested:
            return
        try:
            self.frame_queue.put_nowait((timestamp_ns, topic, frame))
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait((timestamp_ns, topic, frame))
            except queue.Full:
                pass
        self.frame_ready.emit((timestamp_ns, topic, frame))

    def _handle_bridge_error(self, message: str) -> None:
        if not self._stop_requested:
            self.error.emit(message)
        self._finished_event.set()


class SystemMonitorWorker(QThread):
    sample_ready = Signal(dict)

    def __init__(self, interval_s: float = 1.0, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.interval_s = interval_s
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        monitor = SystemMonitor()
        try:
            while not self._stop_requested:
                self.sample_ready.emit(monitor.sample())
                time.sleep(self.interval_s)
        finally:
            monitor.close()


class ValidationWorker(QThread):
    progress = Signal(int, int)
    metrics_ready = Signal(dict)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        model_path: str,
        task: TaskType,
        image_dir: str,
        labels_dir: str,
        device: str = "",
        conf: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.model_path = model_path
        self.task = TaskType(task)
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.device = device
        self.conf = conf
        self.iou = iou
        self.image_size = image_size
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            from metrics.evaluator import make_yolo_ground_truth_loader
            import cv2

            image_dir = Path(self.image_dir)
            image_paths = sorted(
                str(p)
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                for p in image_dir.glob(ext)
            )
            if not image_paths:
                self.error.emit("No validation images found")
                return
            label_path = Path(self.labels_dir)
            coco_names: dict[int, str] = {}
            coco_gt: dict[Any, list[Any]] = {}
            coco_image_map: dict[str, Any] = {}
            if label_path.is_file() and label_path.suffix.lower() == ".json":
                from metrics.coco_metrics import load_coco_ground_truth, load_coco_image_name_map

                coco_gt, coco_names = load_coco_ground_truth(str(label_path))
                coco_image_map = load_coco_image_name_map(str(label_path))
            if self.task == TaskType.POSE and not coco_image_map:
                self.error.emit("Pose validation requires a COCO JSON with keypoints.")
                return
            class_names = coco_names if coco_names else self._read_class_names(label_path / "classes.txt")
            model = create_model(
                self.model_path,
                self.task,
                device=self.device or None,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.image_size,
            )
            model.load()
            model_info: dict[str, Any] = {}
            try:
                model_info = model.model_info()
            except Exception:
                logger.exception("Unable to read model static metadata")
            training_args: dict[str, Any] = {}
            try:
                from utils.training_metrics import parse_ultralytics_args_yaml

                training_args = parse_ultralytics_args_yaml(self.model_path)
            except Exception:
                logger.exception("Unable to parse training args")
            yolo_loader = make_yolo_ground_truth_loader(self.labels_dir, self.image_dir, class_names)

            def loader(image_path: str) -> tuple[list[Any], dict[int, str]]:
                if coco_image_map:
                    image_id = coco_image_map.get(Path(image_path).name)
                    return coco_gt.get(image_id, []), coco_names
                return yolo_loader(image_path)
            total = len(image_paths)

            if self.task == TaskType.POSE and coco_image_map:
                from metrics.coco_metrics import PoseEvaluator, load_coco_pose_ground_truth

                pose_gt = load_coco_pose_ground_truth(str(label_path))
                pose_evaluator = PoseEvaluator()
                processed = 0
                total_ms = 0.0
                for idx, image_path in enumerate(image_paths, 1):
                    if self._stop_requested:
                        break
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    image_id = coco_image_map.get(Path(image_path).name)
                    for gt in pose_gt.get(image_id, []):
                        pose_evaluator.add_ground_truth(image_id, gt.keypoints, gt.class_id)
                    result = model.predict(image)
                    if result.keypoints is not None:
                        for keypoints in result.keypoints:
                            pose_evaluator.add_prediction(image_id, keypoints, 0)
                    total_ms += result.timing.total_ms
                    processed += 1
                    self.progress.emit(idx, total)
                pose_metrics = pose_evaluator.compute(oks_threshold=0.5)
                pose_metrics["avg_latency_ms"] = total_ms / max(1, processed)
                pose_metrics["frames"] = processed
                pose_metrics["model_path"] = self.model_path
                pose_metrics["task"] = self.task.value
                pose_metrics["model_info"] = model_info
                pose_metrics["training_args"] = training_args
                pose_metrics["GFLOPs"] = model_info.get("GFLOPs")
                pose_metrics["parameters"] = model_info.get("parameters")
                pose_metrics["epochs"] = training_args.get("epochs")
                pose_metrics["batch"] = training_args.get("batch")
                self.metrics_ready.emit(pose_metrics)
                return

            # evaluate_detection_on_images processes all images; we wrap it to emit progress.
            from metrics.coco_metrics import DetectionEvaluator

            evaluator = DetectionEvaluator()
            all_names: dict[int, str] = {}
            total_ms = 0.0
            processed = 0
            for idx, image_path in enumerate(image_paths, 1):
                if self._stop_requested:
                    break
                image = cv2.imread(image_path)
                if image is None:
                    continue
                gt, names = loader(image_path)
                all_names.update(names)
                image_id = Path(image_path).stem
                evaluator.add_ground_truth(
                    image_id,
                    np.array([g.box for g in gt], np.float64),
                    np.array([g.class_id for g in gt], np.int64),
                )
                result = model.predict(image)
                evaluator.add_prediction(image_id, result.boxes, result.scores, result.class_ids)
                total_ms += result.timing.total_ms
                processed += 1
                self.progress.emit(idx, total)
            evaluator.class_names = all_names
            metrics = evaluator.compute(iou_threshold=0.5)
            metrics["mAP50"] = metrics.get("mAP", 0.0)
            metrics["mAP50-95"] = evaluator.compute_map50_95()
            metrics["avg_latency_ms"] = total_ms / max(1, processed)
            metrics["frames"] = processed
            metrics["model_path"] = self.model_path
            metrics["task"] = self.task.value
            metrics["model_info"] = model_info
            metrics["training_args"] = training_args
            metrics["GFLOPs"] = model_info.get("GFLOPs")
            metrics["parameters"] = model_info.get("parameters")
            metrics["epochs"] = training_args.get("epochs")
            metrics["batch"] = training_args.get("batch")
            self.metrics_ready.emit(metrics)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Validation worker failed")
            self.error.emit(str(exc))
        finally:
            try:
                if "model" in locals() and hasattr(model, "close"):
                    model.close()
            except Exception:
                logger.exception("Error closing validation model")
            self.finished.emit()

    @staticmethod
    def _read_class_names(path: Path) -> dict[int, str]:
        if not path.exists():
            return {}
        names: dict[int, str] = {}
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            name = line.strip()
            if name:
                names[idx] = name
        return names
