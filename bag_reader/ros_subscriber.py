from __future__ import annotations

import queue
import threading
from typing import Any, Optional

import cv2
import numpy as np

from utils.ros_env import ensure_ros2


class RosImageSubscriber:
    """Thread-safe ROS 2 Image subscription.

    A dedicated executor lives in its own Python thread, so no ROS callback is
    ever invoked on the Qt GUI thread.
    """

    def __init__(
        self,
        topic: str,
        frame_queue: queue.Queue,
        qos_depth: int = 1,
        node_name: str = "yolo_bench_subscriber",
        use_best_effort: bool = True,
    ) -> None:
        self.topic = topic
        self.frame_queue = frame_queue
        self.qos_depth = qos_depth
        self.node_name = node_name
        self.use_best_effort = use_best_effort
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._bridge: Any = None
        self._executor: Any = None
        self._node: Any = None
        self._last_error: str = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self.node_name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    @property
    def last_error(self) -> str:
        return self._last_error

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        try:
            ensure_ros2()
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
            from sensor_msgs.msg import Image
        except RuntimeError as exc:
            self._last_error = str(exc)
            return
        except ImportError as exc:
            self._last_error = f"Missing ROS/cv_bridge dependency: {exc}"
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = rclpy.create_node(self.node_name)
            self._bridge = CvBridge()
            reliability = (
                QoSReliabilityPolicy.BEST_EFFORT if self.use_best_effort else QoSReliabilityPolicy.RELIABLE
            )
            qos = QoSProfile(
                depth=self.qos_depth,
                reliability=reliability,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self._node.create_subscription(Image, self.topic, self._image_callback, qos)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            while not self._stop_event.is_set():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as exc:  # noqa: BLE001 - surfaced to GUI
            self._last_error = str(exc)
        finally:
            self._shutdown()

    def _image_callback(self, msg: Any) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            stamp = msg.header.stamp
            ts_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            item = (ts_ns, self.topic, np.ascontiguousarray(frame))
            try:
                self.frame_queue.put_nowait(item)
            except queue.Full:
                # Drop oldest frame to keep latency bounded for live inference.
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(item)
                except queue.Full:
                    pass
        except Exception as exc:  # noqa: BLE001 - conversion errors should not kill ROS node
            self._last_error = str(exc)

    def _shutdown(self) -> None:
        try:
            if self._executor is not None:
                self._executor.shutdown(timeout_sec=0.2)
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        self._executor = None
        self._node = None
