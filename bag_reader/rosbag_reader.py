from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np

from utils.ros_env import ensure_ros2


@dataclass
class BagFrame:
    timestamp_ns: int
    frame: np.ndarray
    topic: str
    ros_time_s: float


class RosbagImageReader:
    """Read image messages from ROS 2 `.db3` / `.mcap` bags using rosbag2_py.

    The reader performs deserialization of `sensor_msgs/msg/Image` messages and
    converts common encodings to BGR.  Non-Image topics are ignored, which
    keeps image data flow decoupled from the GUI.
    """

    def __init__(self, bag_path: str, topic_filter: Optional[list[str]] = None) -> None:
        self.bag_path = Path(bag_path)
        self.topic_filter = topic_filter or []
        self._reader: Any = None
        self._storage_filter: Any = None
        self._closed = True
        self._lock = threading.Lock()
        self._frame_count = 0
        self._duration_s = 0.0
        self._start_time_s: Optional[float] = None
        self._total_frames = 0
        self._total_duration_s = 0.0
        self._start_frame = 0
        self._frame_index = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_s(self) -> float:
        return self._duration_s

    def open(self) -> None:
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag does not exist: {self.bag_path}")
        ensure_ros2()
        try:
            from rclpy.serialization import deserialize_message
            from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError("ROS 2 Python packages could not be imported") from exc

        storage_id = self._detect_storage_id(self.bag_path)
        storage_options = StorageOptions(uri=str(self.bag_path), storage_id=storage_id)
        converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = {
            meta.name: meta.type
            for meta in reader.get_all_topics_and_types()
        }
        image_topics = [t for t, msg_type in topic_types.items() if msg_type.endswith("/msg/Image")]
        if self.topic_filter:
            requested = [t for t in image_topics if any(t == f or t.endswith(f) for f in self.topic_filter)]
            if requested:
                image_topics = requested
        if not image_topics:
            raise ValueError("No sensor_msgs/msg/Image topic found in bag")

        reader.set_filter(StorageFilter(topics=image_topics))
        self._reader = reader
        self._storage_filter = StorageFilter(topics=image_topics)
        self._closed = False
        self._frame_count = 0
        self._duration_s = 0.0
        self._start_time_s = None
        self._frame_index = 0
        self._deserialize_message = deserialize_message
        self._Image = Image

    def scan(self, should_stop: Optional[Callable[[], bool]] = None) -> None:
        """Count frames and total duration once, then reopen the bag.

        This gives the GUI a determinate progress bar and enables approximate
        seek-by-skip for `SequentialReader`, which has no built-in random access.
        """
        if self._reader is None:
            self.open()
        assert self._reader is not None
        total_frames = 0
        start_s: Optional[float] = None
        end_s = 0.0
        should_stop = should_stop or (lambda: False)
        while True:
            try:
                if should_stop():
                    break
                if not self._reader.has_next():
                    break
                _, data, _ = self._reader.read_next()
                msg = self._deserialize_message(data, self._Image)
                ts_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                if start_s is None:
                    start_s = ts_s
                end_s = max(end_s, ts_s)
                total_frames += 1
            except Exception:
                break
        self._total_frames = total_frames
        self._total_duration_s = max(0.0, end_s - (start_s or 0.0))
        self.close()
        self.open()

    def seek_ratio(self, ratio: float) -> None:
        ratio = min(1.0, max(0.0, ratio))
        self._start_frame = int(self._total_frames * ratio)
        self._frame_index = 0

    def _detect_storage_id(self, path: Path) -> str:
        if path.suffix.lower() == ".mcap":
            return "mcap"
        if path.suffix.lower() == ".db3":
            return "sqlite3"
        # Empty string lets rosbag2 auto-detect the installed storage plugin.
        return ""

    def iter_frames(self) -> Iterator[BagFrame]:
        if self._reader is None:
            self.open()
        while self.has_next():
            try:
                topic, data, t = self._reader.read_next()
                if self._frame_index < self._start_frame:
                    self._frame_index += 1
                    continue
                self._frame_index += 1
                msg = self._deserialize_message(data, self._Image)
                frame = self.image_to_bgr(msg)
                if frame is None:
                    continue
                ts_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                if self._start_time_s is None:
                    self._start_time_s = ts_s
                self._duration_s = max(self._duration_s, ts_s - self._start_time_s)
                self._frame_count += 1
                yield BagFrame(
                    timestamp_ns=int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
                    frame=frame,
                    topic=topic,
                    ros_time_s=ts_s,
                )
            except Exception:
                # A malformed single message should not crash the whole bag scan.
                if not self.has_next():
                    break

    def has_next(self) -> bool:
        if self._reader is None:
            return False
        try:
            return self._reader.has_next()
        except Exception:
            return False

    def seek(self, timestamp_ns: int) -> None:
        raise NotImplementedError(
            "SequentialReader does not expose random access on all ROS 2 versions. "
            "Use the GUI progress bar to restart playback from the beginning, or "
            "use `ros2 bag play` for true seek behaviour."
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._reader = None

    @staticmethod
    def image_to_bgr(msg: Any) -> Optional[np.ndarray]:
        """Convert a ROS Image message to BGR uint8 without depending on cv_bridge.

        cv_bridge is still the recommended conversion for external subscribers;
        this fallback keeps the bag reader usable when only message definitions
        are available.
        """
        encoding = msg.encoding
        if encoding in ("rgb8", "bgr8", "mono8", "8uc1"):
            channels = 1 if encoding in ("mono8", "8uc1") else 3
            arr = RosbagImageReader._decode_row_stride(msg, np.uint8, channels)
        elif encoding == "16uc1":
            arr = RosbagImageReader._decode_row_stride(msg, np.uint16, 1)
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif encoding == "32fc1":
            arr = RosbagImageReader._decode_row_stride(msg, np.float32, 1)
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            try:
                from cv_bridge import CvBridge

                bridge = CvBridge()
                arr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                return np.ascontiguousarray(arr)
            except Exception:
                return None

        if arr.ndim == 3 and arr.shape[2] == 3 and encoding == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _decode_row_stride(msg: Any, dtype: Any, channels: int) -> np.ndarray:
        h, w = int(msg.height), int(msg.width)
        itemsize = int(np.dtype(dtype).itemsize)
        expected_step = w * channels * itemsize
        step = int(msg.step) if getattr(msg, "step", 0) else expected_step
        buf = np.frombuffer(msg.data, dtype=dtype)
        if step == expected_step and buf.size >= h * w * channels:
            return buf[: h * w * channels].reshape(h, w, channels)
        arr = np.zeros((h, w, channels), dtype=dtype)
        elems_per_row = step // itemsize
        for y in range(h):
            start = y * elems_per_row
            end = start + w * channels
            if end > buf.size:
                break
            arr[y] = buf[start:end].reshape(w, channels)
        return arr


class BagFrameGenerator:
    """Small stateful wrapper used by the Qt worker.

    It supports pause, resume, play-rate and a simple wall-clock pacing mode.
    `realtime=False` decodes frames as fast as possible; `realtime=True`
    inserts sleeps based on inter-frame ROS timestamps.
    """

    def __init__(
        self,
        reader: RosbagImageReader,
        realtime: bool = True,
        rate: float = 1.0,
        on_frame: Optional[Callable[[BagFrame], None]] = None,
    ) -> None:
        self.reader = reader
        self.realtime = realtime
        self.rate = max(0.05, rate)
        self.on_frame = on_frame
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self.resume()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        last_ts: Optional[float] = None
        for bag_frame in self.reader.iter_frames():
            if self.stopped:
                break
            self._pause_event.wait()
            if self.stopped:
                break
            if self.realtime and last_ts is not None:
                delta = max(0.0, (bag_frame.ros_time_s - last_ts) / self.rate)
                # Interruptible sleep so pause/stop remain responsive.
                self._interruptible_sleep(delta)
            last_ts = bag_frame.ros_time_s
            if self.on_frame and not self.stopped:
                self.on_frame(bag_frame)
        if not self.stopped and self.on_frame:
            self.on_frame(None)  # sentinel for end-of-bag

    @staticmethod
    def _interruptible_sleep(duration: float) -> None:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            time.sleep(min(0.01, max(0.0, end - time.monotonic())))
