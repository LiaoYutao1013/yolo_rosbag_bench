from __future__ import annotations

import numpy as np

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget
except ImportError:  # pragma: no cover
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

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
        self.raw_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        raw_col = QVBoxLayout()
        raw_col.addWidget(self.raw_label, 1)
        raw_col.addWidget(self.raw_info)
        result_col = QVBoxLayout()
        result_col.addWidget(self.result_label, 1)
        result_col.addWidget(self.result_info)

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

    def show_raw(self, frame: np.ndarray) -> None:
        self._set_image(self.raw_label, frame)

    def show_result(self, frame: np.ndarray, info: str | None = None) -> None:
        self._set_image(self.result_label, frame)
        if info:
            self.result_info.setText(info)

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
