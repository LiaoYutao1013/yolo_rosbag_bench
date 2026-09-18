from __future__ import annotations

import os
import sys


def main() -> int:
    # High-DPI attributes must be set before QApplication is created.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PySide6 is required. Install it with: pip install PySide6") from exc

    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO ROS2 Bag Benchmark")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
