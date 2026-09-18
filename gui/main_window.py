from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any, Optional

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

from metrics.performance_monitor import SystemSample
from models.base import TaskType
from utils.config import AppConfig
from utils.exporters import export_records, export_results
from utils.logger import setup_logger
from utils.ros_env import ensure_ros2
from utils.training_metrics import parse_ultralytics_results_csv

from .plot_panel import PlotPanel
from .video_panel import VideoPanel
from .workers import (
    BagPlayerWorker,
    InferenceFrame,
    InferenceWorker,
    RosSubscriberWorker,
    SystemMonitorWorker,
    ValidationWorker,
)


logger = setup_logger("yolo_bench.gui")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO ROS 2 Bag Benchmark")
        self.resize(1560, 960)
        self.config = AppConfig()
        self.records: list[dict[str, Any]] = []
        self.validation_metrics: dict[str, Any] = {}
        self.frame_queue: Optional[queue.Queue] = None
        self.player_worker: Optional[BagPlayerWorker] = None
        self.inference_worker: Optional[InferenceWorker] = None
        self.live_worker: Optional[RosSubscriberWorker] = None
        self.monitor_worker: Optional[SystemMonitorWorker] = None
        self.validation_worker: Optional[ValidationWorker] = None
        self._paused = False
        self._started_at = time.monotonic()
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_panel())
        self.video_panel = VideoPanel()
        splitter.addWidget(self.video_panel)
        self.plot_panel = PlotPanel()
        splitter.addWidget(self.plot_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([390, 640, 520])
        root.addWidget(splitter, 1)
        root.addWidget(self._build_metrics_panel(), 0)

        self.statusBar().showMessage("Ready")

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)

        model_group = QGroupBox("Model")
        form = QFormLayout(model_group)
        self.model_edit = QLineEdit()
        model_browse = QPushButton("...")
        model_browse.setFixedWidth(34)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(model_browse)
        form.addRow("Weights", model_row)
        self.task_combo = QComboBox()
        for task in TaskType:
            self.task_combo.addItem(task.value, task)
        form.addRow("Task", self.task_combo)
        self.device_edit = QComboBox()
        self.device_edit.setEditable(True)
        self.device_edit.addItems(["", "cpu", "0", "cuda:0"])
        form.addRow("Device", self.device_edit)
        layout.addWidget(model_group)

        source_group = QGroupBox("Source")
        source_form = QFormLayout(source_group)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Rosbag (.db3/.mcap)", "bag")
        self.source_combo.addItem("Live ROS 2 topic", "live")
        source_form.addRow("Mode", self.source_combo)
        self.bag_edit = QLineEdit()
        bag_browse = QPushButton("...")
        bag_browse.setFixedWidth(34)
        bag_row = QHBoxLayout()
        bag_row.addWidget(self.bag_edit, 1)
        bag_row.addWidget(bag_browse)
        source_form.addRow("Bag path", bag_row)
        self.bag_topics_edit = QLineEdit()
        self.bag_topics_edit.setPlaceholderText("optional, comma separated")
        source_form.addRow("Image topics", self.bag_topics_edit)
        self.live_topic_edit = QLineEdit("/camera/color/image_raw")
        source_form.addRow("Live topic", self.live_topic_edit)
        self.best_effort_check = QCheckBox("Best-effort QoS")
        self.best_effort_check.setChecked(True)
        source_form.addRow("", self.best_effort_check)
        layout.addWidget(source_group)

        run_group = QGroupBox("Inference / playback")
        run_form = QFormLayout(run_group)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        run_form.addRow("Confidence", self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        run_form.addRow("IoU", self.iou_spin)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        run_form.addRow("Image size", self.imgsz_spin)
        self.realtime_check = QCheckBox("Realtime pacing")
        self.realtime_check.setChecked(True)
        run_form.addRow("", self.realtime_check)
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.05, 20.0)
        self.rate_spin.setSingleStep(0.25)
        self.rate_spin.setValue(1.0)
        run_form.addRow("Play rate", self.rate_spin)
        layout.addWidget(run_group)

        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        button_row = QHBoxLayout()
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.pause_btn)
        button_row.addWidget(self.stop_btn)
        layout.addLayout(button_row)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setEnabled(False)
        layout.addWidget(self.progress_slider)
        self.progress_label = QLabel("Progress: -")
        layout.addWidget(self.progress_label)

        validation_group = QGroupBox("Validation")
        val_form = QFormLayout(validation_group)
        self.image_dir_edit = QLineEdit()
        image_browse = QPushButton("...")
        image_browse.setFixedWidth(34)
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_dir_edit, 1)
        image_row.addWidget(image_browse)
        val_form.addRow("Image dir", image_row)
        self.labels_dir_edit = QLineEdit()
        self.labels_dir_edit.setPlaceholderText("YOLO labels dir or COCO JSON")
        label_browse = QPushButton("...")
        label_browse.setToolTip("Choose YOLO labels directory")
        label_browse.setFixedWidth(34)
        coco_browse = QPushButton("JSON")
        coco_browse.setToolTip("Choose COCO annotation JSON")
        label_row = QHBoxLayout()
        label_row.addWidget(self.labels_dir_edit, 1)
        label_row.addWidget(label_browse)
        label_row.addWidget(coco_browse)
        val_form.addRow("Labels dir", label_row)
        self.validate_btn = QPushButton("Run validation")
        val_form.addRow("", self.validate_btn)
        layout.addWidget(validation_group)

        export_row = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_json_btn = QPushButton("Export JSON")
        self.load_train_btn = QPushButton("Load train CSV")
        export_row.addWidget(self.export_csv_btn)
        export_row.addWidget(self.export_json_btn)
        layout.addLayout(export_row)
        layout.addWidget(self.load_train_btn)
        layout.addStretch(1)

        model_browse.clicked.connect(self._browse_model)
        bag_browse.clicked.connect(self._browse_bag)
        image_browse.clicked.connect(self._browse_image_dir)
        label_browse.clicked.connect(self._browse_labels_dir)
        coco_browse.clicked.connect(self._browse_coco_json)
        return panel

    def _build_metrics_panel(self) -> QWidget:
        panel = QGroupBox("Metrics / validation")
        layout = QVBoxLayout(panel)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setMinimumHeight(170)
        layout.addWidget(self.metrics_table)
        return panel

    def _connect_signals(self) -> None:
        self.source_combo.currentIndexChanged.connect(self._update_source_state)
        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop)
        self.validate_btn.clicked.connect(self._run_validation)
        self.export_csv_btn.clicked.connect(lambda: self._export("csv"))
        self.export_json_btn.clicked.connect(lambda: self._export("json"))
        self.load_train_btn.clicked.connect(self._load_training_csv)
        self.progress_slider.valueChanged.connect(self._on_slider_moved)
        self._update_source_state()

    # -------------------------------------------------------------- actions
    def _update_source_state(self) -> None:
        is_bag = self.source_combo.currentData() == "bag"
        self.bag_edit.setEnabled(is_bag)
        self.bag_topics_edit.setEnabled(is_bag)
        self.realtime_check.setEnabled(is_bag)
        self.rate_spin.setEnabled(is_bag)
        self.live_topic_edit.setEnabled(not is_bag)
        self.best_effort_check.setEnabled(not is_bag)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select model", "", "Model (*.pt *.onnx *.engine);;All files (*)")
        if path:
            self.model_edit.setText(path)

    def _browse_bag(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select rosbag", "", "ROS 2 bag (*.db3 *.mcap);;All files (*)")
        if path:
            self.bag_edit.setText(path)

    def _browse_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select validation image directory")
        if path:
            self.image_dir_edit.setText(path)

    def _browse_labels_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select YOLO label directory")
        if path:
            self.labels_dir_edit.setText(path)

    def _browse_coco_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select COCO JSON", "", "COCO JSON (*.json);;All files (*)")
        if path:
            self.labels_dir_edit.setText(path)

    def _collect_config(self, require_source: bool = True) -> Optional[AppConfig]:
        model_path = self.model_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Invalid model", "Select an existing model file (.pt/.onnx/.engine).")
            return None
        source_mode = self.source_combo.currentData()
        if require_source and source_mode == "bag":
            bag_path = self.bag_edit.text().strip()
            if not bag_path or not Path(bag_path).exists():
                QMessageBox.warning(self, "Invalid bag", "Select an existing .db3 or .mcap bag.")
                return None
        config = AppConfig(
            model_path=model_path,
            task=TaskType(self.task_combo.currentData()),
            device=self.device_edit.currentText().strip(),
            source_mode=source_mode,
            bag_path=self.bag_edit.text().strip() if source_mode == "bag" else "",
            bag_topics=[x.strip() for x in self.bag_topics_edit.text().split(",") if x.strip()],
            live_topic=self.live_topic_edit.text().strip() or "/camera/color/image_raw",
            best_effort=self.best_effort_check.isChecked(),
            realtime=self.realtime_check.isChecked(),
            play_rate=float(self.rate_spin.value()),
            confidence=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            image_size=int(self.imgsz_spin.value()),
            frame_queue_size=8,
            labels_dir=self.labels_dir_edit.text().strip(),
            image_dir=self.image_dir_edit.text().strip(),
        )
        self.config = config
        return config

    def _start(self) -> None:
        config = self._collect_config()
        if config is None:
            return
        try:
            ensure_ros2()
        except RuntimeError as exc:
            QMessageBox.warning(self, "ROS 2 unavailable", str(exc))
            return
        self._stop_workers()
        if any(
            worker is not None and worker.isRunning()
            for worker in (self.player_worker, self.live_worker, self.inference_worker, self.monitor_worker, self.validation_worker)
        ):
            QMessageBox.warning(self, "Busy", "A worker is still stopping. Please wait and try again.")
            return
        self.records = []
        self.validation_metrics = {}
        self.plot_panel.reset()
        self._started_at = time.monotonic()
        self.frame_queue = queue.Queue(maxsize=config.frame_queue_size)

        self.inference_worker = InferenceWorker(
            frame_queue=self.frame_queue,
            model_path=config.model_path,
            task=config.task,
            device=config.device,
            conf=config.confidence,
            iou=config.iou,
            image_size=config.image_size,
        )
        self.inference_worker.frame_processed.connect(self._on_frame_processed)
        self.inference_worker.model_loaded.connect(self._on_model_loaded)
        self.inference_worker.error.connect(self._on_worker_error)
        self.inference_worker.finished.connect(self._on_inference_finished)
        self.inference_worker.status.connect(self.statusBar().showMessage)

        if config.source_mode == "bag":
            self.player_worker = BagPlayerWorker(
                bag_path=config.bag_path,
                frame_queue=self.frame_queue,
                topics=config.bag_topics,
                realtime=config.realtime,
                rate=config.play_rate,
            )
            self.player_worker.progress_changed.connect(self._on_progress)
            self.player_worker.error.connect(self._on_worker_error)
            self.player_worker.finished.connect(self._on_player_finished)
            self.player_worker.status.connect(self.statusBar().showMessage)
        else:
            self.live_worker = RosSubscriberWorker(
                topic=config.live_topic,
                frame_queue=self.frame_queue,
                qos_depth=config.qos_depth,
                best_effort=config.best_effort,
            )
            self.live_worker.error.connect(self._on_worker_error)
            self.live_worker.status.connect(self.statusBar().showMessage)

        self.monitor_worker = SystemMonitorWorker(interval_s=1.0)
        self.monitor_worker.sample_ready.connect(self._on_system_sample)

        self.inference_worker.start()
        if self.player_worker is not None:
            self.player_worker.start()
        if self.live_worker is not None:
            self.live_worker.start()
        self.monitor_worker.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self._paused = False
        self.progress_slider.setEnabled(config.source_mode == "bag")
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(0)
        self.progress_slider.blockSignals(False)

    def _toggle_pause(self) -> None:
        if self._paused:
            if self.player_worker is not None:
                self.player_worker.resume()
            if self.inference_worker is not None:
                self.inference_worker.resume()
            self._paused = False
            self.pause_btn.setText("Pause")
        else:
            if self.player_worker is not None:
                self.player_worker.pause()
            if self.inference_worker is not None:
                self.inference_worker.pause()
            self._paused = True
            self.pause_btn.setText("Resume")

    def _stop(self) -> None:
        self._stop_workers()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_slider.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    def _stop_workers(self) -> None:
        for worker in (self.player_worker, self.live_worker, self.inference_worker, self.monitor_worker, self.validation_worker):
            if worker is not None and worker.isRunning():
                worker.stop() if hasattr(worker, "stop") else worker.requestInterruption()
        workers = (
            self.player_worker,
            self.live_worker,
            self.inference_worker,
            self.monitor_worker,
            self.validation_worker,
        )
        for worker in workers:
            if worker is not None and worker.isRunning():
                worker.wait(4000)
                if worker.isRunning():
                    logger.warning("Worker %s did not stop in time", type(worker).__name__)
        if self.player_worker is not None and not self.player_worker.isRunning():
            self.player_worker = None
        if self.live_worker is not None and not self.live_worker.isRunning():
            self.live_worker = None
        if self.inference_worker is not None and not self.inference_worker.isRunning():
            self.inference_worker = None
        if self.monitor_worker is not None and not self.monitor_worker.isRunning():
            self.monitor_worker = None
        if self.validation_worker is not None and not self.validation_worker.isRunning():
            self.validation_worker = None
        self.frame_queue = None

    def _on_slider_moved(self, value: int) -> None:
        if self.player_worker is not None:
            self.player_worker.seek(value / 1000.0)
            self.progress_label.setText(f"Progress: seeking {value / 10.0:.1f}%")

    def _run_validation(self) -> None:
        config = self._collect_config(require_source=False)
        if config is None:
            return
        if any(
            worker is not None and worker.isRunning()
            for worker in (self.player_worker, self.live_worker, self.inference_worker, self.monitor_worker)
        ):
            QMessageBox.information(self, "Validation", "Stop live playback before running validation.")
            return
        if not config.image_dir or not Path(config.image_dir).is_dir():
            QMessageBox.warning(self, "Validation", "Select a validation image directory.")
            return
        if self.validation_worker is not None and self.validation_worker.isRunning():
            QMessageBox.information(self, "Validation", "Validation is already running.")
            return
        self.validation_worker = ValidationWorker(
            model_path=config.model_path,
            task=config.task,
            image_dir=config.image_dir,
            labels_dir=config.labels_dir,
            device=config.device,
            conf=config.confidence,
            iou=config.iou,
            image_size=config.image_size,
        )
        self.validation_worker.progress.connect(lambda done, total: self.statusBar().showMessage(f"Validation {done}/{total}"))
        self.validation_worker.metrics_ready.connect(self._on_validation_metrics)
        self.validation_worker.error.connect(self._on_worker_error)
        self.validation_worker.start()

    def _load_training_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Ultralytics results.csv", "", "CSV (*.csv)")
        if not path:
            return
        metrics = parse_ultralytics_results_csv(path)
        if not metrics:
            QMessageBox.warning(self, "Training metrics", "No metrics found in the selected CSV.")
            return
        self.validation_metrics.update(metrics)
        self._populate_metrics(metrics)

    # ---------------------------------------------------------------- slots
    def _on_model_loaded(self, path: str) -> None:
        self.statusBar().showMessage(f"Model loaded: {path}")

    def _on_frame_processed(self, payload: InferenceFrame) -> None:
        self.video_panel.show_raw(payload.raw)
        count = int(len(payload.result.boxes))
        info = (
            f"{payload.topic}\n"
            f"objects={count}  fps={payload.fps:.1f}  latency={payload.latency_ms:.1f}ms\n"
            f"jitter={payload.jitter_ms:.1f}ms  CPU={payload.system.get('cpu_percent', 0):.0f}% "
            f"GPU={payload.system.get('gpu_util', 0):.0f}%"
        )
        self.video_panel.show_result(payload.rendered, info)
        elapsed = time.monotonic() - self._started_at
        self.plot_panel.update(elapsed, payload.fps, payload.latency_ms, payload.system)
        self.records.append(payload.as_record())
        if len(self.records) > 20000:
            del self.records[:10000]

    def _on_progress(self, ratio: float) -> None:
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(int(ratio * 1000))
        self.progress_slider.blockSignals(False)
        self.progress_label.setText(f"Progress: {ratio * 100.0:.1f}%")

    def _on_system_sample(self, sample: dict[str, float]) -> None:
        # The chart is primarily updated by frame_processed, which carries a
        # fresher inference-time system sample. This slot exists so CPU/GPU can
        # continue updating while a bag is paused or between frames.
        if not self.records:
            elapsed = time.monotonic() - self._started_at
            self.plot_panel.update(elapsed, 0.0, 0.0, sample)

    def _on_player_finished(self) -> None:
        self.statusBar().showMessage("Bag playback finished")

    def _on_inference_finished(self) -> None:
        if not self.records:
            self.statusBar().showMessage("Inference stopped without output")

    def _on_worker_error(self, message: str) -> None:
        logger.error(message)
        self.statusBar().showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Worker error", message)

    def _on_validation_metrics(self, metrics: dict[str, Any]) -> None:
        self.validation_metrics = metrics
        display_metrics = dict(metrics)
        for nested_key in ("model_info", "training_args"):
            nested = display_metrics.pop(nested_key, None)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    display_metrics[f"{nested_key}.{key}"] = value
        self._populate_metrics(display_metrics)
        self.statusBar().showMessage("Validation complete")

    def _populate_metrics(self, metrics: dict[str, Any]) -> None:
        preferred_order = [
            "mAP50",
            "mAP50-95",
            "precision",
            "recall",
            "train/box_loss",
            "train/cls_loss",
            "train/obj_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/obj_loss",
            "avg_latency_ms",
            "frames",
            "model_path",
        ]
        ordered = [k for k in preferred_order if k in metrics]
        ordered += [k for k in metrics if k not in ordered]
        self.metrics_table.setRowCount(len(ordered))
        for row, key in enumerate(ordered):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(key)))
            value = metrics[key]
            if isinstance(value, (dict, list)):
                import json

                value = json.dumps(value, ensure_ascii=False, default=str)
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(value)))

    # ---------------------------------------------------------------- export
    def _export(self, extension: str) -> None:
        default_dir = Path(self.config.export_dir or "results")
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export report",
            str(default_dir / f"yolo_bench_report.{extension}"),
            f"{extension.upper()} (*.{extension})",
        )
        if not path:
            return
        payload: dict[str, Any] = {
            "config": self.config.as_dict(),
            "metrics": self.validation_metrics,
            "frames": self.records,
        }
        try:
            if extension == "json":
                export_results(path, payload)
            else:
                export_records(path, self.records)
            self.statusBar().showMessage(f"Exported to {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))

    def closeEvent(self, event: Any) -> None:
        self._stop_workers()
        super().closeEvent(event)
