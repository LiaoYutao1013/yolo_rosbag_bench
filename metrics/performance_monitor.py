from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class SystemSample:
    timestamp_s: float
    fps: float
    latency_ms: float
    jitter_ms: float
    cpu_percent: float
    memory_percent: float
    gpu_util: float
    gpu_memory_percent: float
    gpu_memory_mb: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class FrameRateMeter:
    def __init__(self, window: int = 30) -> None:
        self.window = window
        self._timestamps: deque[float] = deque(maxlen=window)
        self._intervals: deque[float] = deque(maxlen=window)

    def update(self, timestamp_s: Optional[float] = None) -> float:
        now = timestamp_s if timestamp_s is not None else time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            self._intervals.append(max(0.0, self._timestamps[-1] - self._timestamps[-2]))
        if len(self._intervals) < 1:
            return 0.0
        mean_interval = sum(self._intervals) / len(self._intervals)
        return 1.0 / mean_interval if mean_interval > 0 else 0.0

    @property
    def jitter_ms(self) -> float:
        if len(self._intervals) < 2:
            return 0.0
        mean = sum(self._intervals) / len(self._intervals)
        var = sum((x - mean) ** 2 for x in self._intervals) / (len(self._intervals) - 1)
        return (var**0.5) * 1000.0


class LatencyTracker:
    def __init__(self, window: int = 30) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def update(self, latency_ms: float) -> None:
        if latency_ms >= 0:
            self._values.append(float(latency_ms))

    @property
    def average_ms(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class SystemMonitor:
    """Cross-version CPU/GPU monitor using psutil and optional pynvml."""

    def __init__(self) -> None:
        self._psutil: Any = None
        self._pynvml: Any = None
        self._handle: Any = None
        self._lock = threading.Lock()
        self._gpu_available = False
        self._init()

    def _init(self) -> None:
        try:
            import psutil

            self._psutil = psutil
        except ImportError:
            self._psutil = None
        try:
            import pynvml

            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_available = True
        except Exception:
            self._gpu_available = False

    def close(self) -> None:
        with self._lock:
            if self._pynvml is not None and self._gpu_available:
                try:
                    self._pynvml.nvmlShutdown()
                except Exception:
                    pass
            self._gpu_available = False

    def sample(self) -> dict[str, float]:
        with self._lock:
            cpu = 0.0
            memory = 0.0
            if self._psutil is not None:
                cpu = self._psutil.cpu_percent(interval=None)
                memory = self._psutil.virtual_memory().percent
            gpu_util = 0.0
            gpu_mem_pct = 0.0
            gpu_mem_mb = 0.0
            if self._gpu_available and self._pynvml is not None:
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    gpu_util = float(util.gpu)
                    gpu_mem_mb = float(mem.used) / (1024.0 * 1024.0)
                    gpu_mem_pct = (float(mem.used) / float(mem.total)) * 100.0 if mem.total else 0.0
                except Exception:
                    pass
            return {
                "cpu_percent": cpu,
                "memory_percent": memory,
                "gpu_util": gpu_util,
                "gpu_memory_percent": gpu_mem_pct,
                "gpu_memory_mb": gpu_mem_mb,
            }
