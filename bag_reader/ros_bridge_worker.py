#!/usr/bin/env python3
from __future__ import annotations

import json
import queue as py_queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

# Make local project modules importable when this script is executed by the
# system ROS Python interpreter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from bag_reader.ros_bridge_protocol import encode_message
from bag_reader.rosbag_reader import BagFrameGenerator, RosbagImageReader
from bag_reader.ros_subscriber import RosImageSubscriber


class RosBridgeWorker:
    def __init__(self) -> None:
        self._stdout_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._paused_event.set()
        self._command_queue: py_queue.Queue[dict[str, Any]] = py_queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._current_generator: Optional[BagFrameGenerator] = None
        self._current_subscriber: Optional[RosImageSubscriber] = None
        self._state_lock = threading.Lock()
        self._seek_ratio: Optional[float] = None
        self._seek_pending = False
        self._rate = 1.0
        self._mode: Optional[str] = None

    # ------------------------------------------------------------- protocol
    def send(self, message_type: str, payload: bytes = b"", **fields: Any) -> None:
        data = encode_message(message_type, payload, **fields)
        with self._stdout_lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    def send_status(self, message: str) -> None:
        self.send("status", message=message)

    def send_error(self, message: str) -> None:
        self.send("error", message=message)

    def send_frame(self, timestamp_ns: int, topic: str, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            return
        self.send("frame", encoded.tobytes(), timestamp_ns=timestamp_ns, topic=topic)

    # ------------------------------------------------------------ command io
    def run(self) -> None:
        reader_thread = threading.Thread(target=self._read_commands, name="bridge-stdin", daemon=True)
        reader_thread.start()
        self.send_status("ROS bridge ready")
        while True:
            command = self._command_queue.get()
            if command is None:
                break
            try:
                self._handle_command(command)
            except Exception as exc:  # noqa: BLE001
                self.send_error(str(exc))
            if command.get("cmd") == "quit":
                break
        self._stop_event.set()
        self._stop_playback()
        self.send("finished")

    def _read_commands(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                self._command_queue.put(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                self.send_error(f"Invalid bridge command: {exc}")
        self._command_queue.put({"cmd": "quit"})

    def _handle_command(self, command: dict[str, Any]) -> None:
        cmd = command.get("cmd")
        if cmd == "ping":
            self.send("pong")
        elif cmd == "open_bag":
            self._stop_playback()
            self._mode = "bag"
            self._stop_event.clear()
            self._paused_event.set()
            self._rate = float(command.get("rate", 1.0))
            self._start_playback(self._bag_loop, command)
        elif cmd == "subscribe":
            self._stop_playback()
            self._mode = "live"
            self._stop_event.clear()
            self._paused_event.set()
            self._start_playback(self._live_loop, command)
        elif cmd == "pause":
            self._paused_event.clear()
            self._pause_generator()
        elif cmd == "resume":
            self._paused_event.set()
            self._resume_generator()
        elif cmd == "seek":
            with self._state_lock:
                self._seek_ratio = min(1.0, max(0.0, float(command.get("ratio", 0.0))))
                self._seek_pending = True
            self._stop_generator()
        elif cmd == "rate":
            self._rate = max(0.05, float(command.get("value", 1.0)))
            with self._state_lock:
                generator = self._current_generator
            if generator is not None:
                generator.rate = self._rate
        elif cmd == "stop":
            self._stop_playback()
        elif cmd == "quit":
            pass

    def _start_playback(self, target: Callable[[dict[str, Any]], None], config: dict[str, Any]) -> None:
        self._playback_thread = threading.Thread(
            target=target,
            args=(config,),
            name="bridge-playback",
            daemon=True,
        )
        self._playback_thread.start()

    def _stop_playback(self) -> None:
        self._stop_event.set()
        self._stop_generator()
        if self._current_subscriber is not None:
            self._current_subscriber.stop()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)
        self._playback_thread = None

    def _pause_generator(self) -> None:
        with self._state_lock:
            generator = self._current_generator
        if generator is not None:
            generator.pause()

    def _resume_generator(self) -> None:
        with self._state_lock:
            generator = self._current_generator
        if generator is not None:
            generator.resume()

    def _stop_generator(self) -> None:
        with self._state_lock:
            generator = self._current_generator
        if generator is not None:
            generator.stop()

    # ------------------------------------------------------------ playback
    def _bag_loop(self, config: dict[str, Any]) -> None:
        bag_path = str(config["bag_path"])
        topics = list(config.get("topics") or [])
        realtime = bool(config.get("realtime", True))
        try:
            while not self._stop_event.is_set():
                self.send_status("Opening bag...")
                reader = RosbagImageReader(bag_path, topics)
                reader.scan(should_stop=self._stop_event.is_set)
                if self._stop_event.is_set():
                    reader.close()
                    break
                self.send_status(f"Playing {reader._total_frames} frames")
                if getattr(reader, "image_topics", None):
                    self.send_status("Image topics: " + ", ".join(reader.image_topics[:6]))

                with self._state_lock:
                    ratio = self._seek_ratio
                    self._seek_ratio = None
                    self._seek_pending = False
                if ratio is not None:
                    reader.seek_ratio(ratio)

                generator = BagFrameGenerator(
                    reader,
                    realtime=realtime,
                    rate=self._rate,
                    on_frame=lambda bag_frame, reader=reader: self._on_bag_frame(bag_frame, reader),
                )
                with self._state_lock:
                    self._current_generator = generator
                if not self._paused_event.is_set():
                    generator.pause()
                generator.run()
                with self._state_lock:
                    self._current_generator = None

                if self._stop_event.is_set():
                    break
                with self._state_lock:
                    if self._seek_pending:
                        continue
                break
        except Exception as exc:  # noqa: BLE001
            self.send_error(str(exc))
        finally:
            if not self._stop_event.is_set():
                self.send("finished")

    def _on_bag_frame(self, bag_frame: Any, reader: RosbagImageReader) -> None:
        if bag_frame is None or self._stop_event.is_set():
            return
        self.send_frame(bag_frame.timestamp_ns, bag_frame.topic, bag_frame.frame)
        if reader._total_frames:
            ratio = min(1.0, max(0.0, reader._frame_index / reader._total_frames))
            self.send("progress", ratio=ratio)

    def _live_loop(self, config: dict[str, Any]) -> None:
        topic = str(config["topic"])
        qos_depth = int(config.get("qos_depth", 1))
        best_effort = bool(config.get("best_effort", True))
        local_queue: py_queue.Queue[tuple[int, str, np.ndarray]] = py_queue.Queue(maxsize=8)
        subscriber = RosImageSubscriber(
            topic=topic,
            frame_queue=local_queue,
            qos_depth=qos_depth,
            node_name="yolo_bench_bridge_subscriber",
            use_best_effort=best_effort,
        )
        with self._state_lock:
            self._current_subscriber = subscriber
        try:
            subscriber.start()
            self.send_status(f"Subscribing to {topic}...")
            while not self._stop_event.is_set():
                if not subscriber.is_alive():
                    if subscriber.last_error:
                        self.send_error(subscriber.last_error)
                    break
                try:
                    timestamp_ns, frame_topic, frame = local_queue.get(timeout=0.1)
                except py_queue.Empty:
                    continue
                if self._paused_event.is_set():
                    self.send_frame(timestamp_ns, frame_topic, frame)
        except Exception as exc:  # noqa: BLE001
            self.send_error(str(exc))
        finally:
            subscriber.stop()
            with self._state_lock:
                self._current_subscriber = None
            if not self._stop_event.is_set():
                self.send("finished")


def main() -> int:
    worker = RosBridgeWorker()
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
