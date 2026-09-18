from __future__ import annotations

from collections import deque

try:
    from PySide6.QtWidgets import QVBoxLayout, QWidget
except ImportError:  # pragma: no cover
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

import pyqtgraph as pg


class PlotPanel(QWidget):
    def __init__(self, max_points: int = 300, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background="k", foreground="w")
        self.max_points = max_points
        self.fps = deque(maxlen=max_points)
        self.latency = deque(maxlen=max_points)
        self.cpu = deque(maxlen=max_points)
        self.gpu = deque(maxlen=max_points)
        self.memory = deque(maxlen=max_points)

        self.fps_plot = self._make_plot("FPS", "frames/s")
        self.latency_plot = self._make_plot("Latency", "ms")
        self.cpu_plot = self._make_plot("CPU / memory", "%")
        self.gpu_plot = self._make_plot("GPU / GPU memory", "%")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.fps_plot)
        layout.addWidget(self.latency_plot)
        layout.addWidget(self.cpu_plot)
        layout.addWidget(self.gpu_plot)

    def _make_plot(self, title: str, unit: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setTitle(title)
        plot.setLabel("left", unit)
        plot.setLabel("bottom", "time (s)")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.addLegend(offset=(10, 10))
        return plot

    def update(self, timestamp_s: float, fps: float, latency_ms: float, system: dict[str, float]) -> None:
        self.fps.append((timestamp_s, float(fps)))
        self.latency.append((timestamp_s, float(latency_ms)))
        self.cpu.append((timestamp_s, float(system.get("cpu_percent", 0.0))))
        self.memory.append((timestamp_s, float(system.get("memory_percent", 0.0))))
        self.gpu.append((timestamp_s, float(system.get("gpu_util", 0.0))))

        def plot_series(plot: pg.PlotWidget, series: deque, name: str, color: str) -> None:
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            plot.clear()
            plot.plot(xs, ys, pen=pg.mkPen(color, width=2), name=name)

        plot_series(self.fps_plot, self.fps, "FPS", "#4caf50")
        plot_series(self.latency_plot, self.latency, "latency", "#ff9800")
        plot_series(self.cpu_plot, self.cpu, "CPU", "#42a5f5")
        plot_series(self.cpu_plot, self.memory, "memory", "#ab47bc")
        plot_series(self.gpu_plot, self.gpu, "GPU util", "#e53935")

    def reset(self) -> None:
        for series in (self.fps, self.latency, self.cpu, self.gpu, self.memory):
            series.clear()
        for plot in (self.fps_plot, self.latency_plot, self.cpu_plot, self.gpu_plot):
            plot.clear()
