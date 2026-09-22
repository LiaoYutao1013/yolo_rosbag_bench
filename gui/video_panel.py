from __future__ import annotations

import numpy as np

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

from utils.image_utils import safe_qimage


class VideoPanel(QWidget):
    def __init__(self, max_width: int = 960, max_height: int = 540, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.max_width = max_width
        self.max_height = max_height
        self.raw_label = self._make_label("Raw rosbag / topic")
        self.result_label = self._make_label("YOLO result")
        self.raw_info = QLabel("Waiting for frames...")
        self.result_info = QLabel("Waiting for inference...")
        self.keypoint_table = self._make_keypoint_table()
        self.raw_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        raw_col = QVBoxLayout()
        raw_col.addWidget(self.raw_label, 1)
        raw_col.addWidget(self.raw_info)
        result_col = QVBoxLayout()
        result_col.addWidget(self.result_label, 1)
        result_col.addWidget(self.result_info)
        result_col.addWidget(self.keypoint_table, 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(raw_col, 1)
        layout.addLayout(result_col, 1)

    def _make_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setFrameShape(QFrame.Shape.Box)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(320, 240)
        label.setScaledContents(False)
        return label

    def _make_keypoint_table(self) -> QTableWidget:
        table = QTableWidget(0, 1)
        table.setHorizontalHeaderLabels(["Keypoint confidence"])
        table.setMaximumHeight(120)
        table.setMinimumHeight(50)
        table.setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def show_raw(self, frame: np.ndarray) -> None:
        self._set_image(self.raw_label, frame)

    def show_result(self, frame: np.ndarray, info: str | None = None) -> None:
        self._set_image(self.result_label, frame)
        if info:
            self.result_info.setText(info)

    def show_pair(
        self,
        raw: np.ndarray,
        result: np.ndarray,
        info: str | None = None,
        keypoints: np.ndarray | None = None,
    ) -> None:
        """Atomically refresh both panels in one GUI slot to avoid partial flicker."""
        self._set_image(self.raw_label, raw)
        self._set_image(self.result_label, result)
        if info:
            self.result_info.setText(info)
        self._update_keypoints(keypoints)

    def _update_keypoints(self, keypoints: np.ndarray | None) -> None:
        if keypoints is None or len(keypoints) == 0:
            self.keypoint_table.setVisible(False)
            self.keypoint_table.clearContents()
            self.keypoint_table.setRowCount(0)
            return
        if keypoints.ndim != 3:
            self.keypoint_table.setVisible(False)
            return
        num_people = keypoints.shape[0]
        num_keypoints = keypoints.shape[1]
        if num_people == 0 or num_keypoints == 0:
            self.keypoint_table.setVisible(False)
            return

        headers = ["Person"] + [f"KP{i}" for i in range(num_keypoints)]
        self.keypoint_table.setColumnCount(len(headers))
        self.keypoint_table.setHorizontalHeaderLabels(headers)
        self.keypoint_table.setRowCount(num_people)
        for person_idx, person in enumerate(keypoints):
            person_item = QTableWidgetItem(f"P{person_idx + 1}")
            self.keypoint_table.setItem(person_idx, 0, person_item)
            for kp_idx, kp in enumerate(person):
                conf = float(kp[2]) if len(kp) > 2 else 0.0
                item = QTableWidgetItem(f"{conf:.2f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.keypoint_table.setItem(person_idx, kp_idx + 1, item)
        self.keypoint_table.resizeColumnsToContents()
        self.keypoint_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, num_keypoints + 1):
            self.keypoint_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.keypoint_table.setVisible(True)

    def _set_image(self, label: QLabel, frame: np.ndarray) -> None:
        if frame is None:
            return
        qimg = safe_qimage(frame)
        if qimg is None:
            return
        pixmap = QPixmap.fromImage(qimg)
        if pixmap.width() > self.max_width or pixmap.height() > self.max_height:
            pixmap = pixmap.scaled(
                self.max_width,
                self.max_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        label.setPixmap(pixmap)

    def clear(self) -> None:
        self.raw_label.clear()
        self.raw_label.setText("Raw rosbag / topic")
        self.result_label.clear()
        self.result_label.setText("YOLO result")
        self.raw_info.setText("Waiting for frames...")
        self.result_info.setText("Waiting for inference...")
        self.keypoint_table.setVisible(False)
        self.keypoint_table.clearContents()
        self.keypoint_table.setRowCount(0)
