from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_width / max(1, w), max_height / max(1, h))
    if scale >= 1.0:
        return image
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def safe_qimage(image: np.ndarray) -> Optional[object]:
    """Convert a BGR frame to a Qt image. Returns None if Qt is unavailable."""
    try:
        from PySide6.QtGui import QImage
    except ImportError:
        try:
            from PyQt6.QtGui import QImage
        except ImportError:
            return None
    rgb = to_rgb(np.ascontiguousarray(image))
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def draw_error(image: np.ndarray, message: str) -> np.ndarray:
    canvas = np.ascontiguousarray(image.copy())
    h, _ = canvas.shape[:2]
    y = max(24, int(h * 0.15))
    for i, line in enumerate(message.splitlines()):
        cv2.putText(canvas, line, (16, y + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas
