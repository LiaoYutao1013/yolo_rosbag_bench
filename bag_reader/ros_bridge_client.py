from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from utils.logger import setup_logger

from .ros_bridge_protocol import decode_message


logger = setup_logger("yolo_bench.ros_bridge")


class RosBridgeClient:
    """Spawn the ROS bridge in a system ROS environment outside conda."""

    def __init__(
        self,
        on_frame: Optional[Callable[[int, str, np.ndarray], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
    ) -> None:
        self.on_frame = on_frame
        self.on_progress = on_progress
        self.on_status = on_status
        self.on_error = on_error
        self.on_finished = on_finished
        self._process: Optional[subprocess.Popen] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()
        self._started = False
        self._finished = False
        self._stderr_tail: list[str] = []

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._process is not None:
            return
        worker = Path(__file__).with_name("ros_bridge_worker.py")
        python = os.environ.get("ROS_BRIDGE_PYTHON", "/usr/bin/python3")
        setup_script = self._find_ros_setup_script()
        if setup_script:
            command = f"source {shlex.quote(setup_script)} && exec {shlex.quote(python)} {shlex.quote(str(worker))}"
            shell = True
        else:
            command = [python, str(worker)]
            shell = False
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            shell=shell,
            executable="/bin/bash" if shell else None,
        )
        self._started = True
        self._finished = False
        self._stdout_thread = threading.Thread(target=self._read_stdout, name="ros-bridge-stdout", daemon=True)
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, name="ros-bridge-stderr", daemon=True)
        self._stderr_thread.start()

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self.send_command({"cmd": "quit"})
        except Exception:
            pass
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(self._process, stream_name)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def stderr_text(self) -> str:
        return "\n".join(self._stderr_tail[-8:])

    # ------------------------------------------------------------ commands
    def send_command(self, command: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("ROS bridge is not running")
        line = json.dumps(command, ensure_ascii=True) + "\n"
        with self._send_lock:
            self._process.stdin.write(line.encode("utf-8"))
            self._process.stdin.flush()

    def open_bag(self, bag_path: str, topics: Optional[list[str]] = None, realtime: bool = True, rate: float = 1.0) -> None:
        self.send_command(
            {
                "cmd": "open_bag",
                "bag_path": bag_path,
                "topics": topics or [],
                "realtime": realtime,
                "rate": rate,
            }
        )

    def subscribe(self, topic: str, qos_depth: int = 1, best_effort: bool = True) -> None:
        self.send_command(
            {
                "cmd": "subscribe",
                "topic": topic,
                "qos_depth": qos_depth,
                "best_effort": best_effort,
            }
        )

    def pause(self) -> None:
        self.send_command({"cmd": "pause"})

    def resume(self) -> None:
        self.send_command({"cmd": "resume"})

    def seek(self, ratio: float) -> None:
        self.send_command({"cmd": "seek", "ratio": ratio})

    def set_rate(self, rate: float) -> None:
        self.send_command({"cmd": "rate", "value": rate})

    def stop(self) -> None:
        try:
            self.send_command({"cmd": "stop"})
        except Exception:
            pass

    # ------------------------------------------------------------ protocol
    def _read_stdout(self) -> None:
        try:
            assert self._process is not None and self._process.stdout is not None
            while True:
                header, payload = decode_message(self._process.stdout)
                self._dispatch(header, payload)
        except EOFError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("ROS bridge stdout reader failed")
            if self.on_error:
                self.on_error(str(exc))

    def _read_stderr(self) -> None:
        try:
            assert self._process is not None and self._process.stderr is not None
            for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    logger.debug("ROS bridge stderr: %s", text)
                    self._stderr_tail.append(text)
                    del self._stderr_tail[:-40]
        except Exception:
            pass

    def _dispatch(self, header: dict[str, Any], payload: bytes) -> None:
        message_type = header.get("type")
        if message_type == "frame":
            encoded = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is not None and self.on_frame:
                self.on_frame(int(header.get("timestamp_ns", 0)), str(header.get("topic", "")), frame)
        elif message_type == "progress" and self.on_progress:
            self.on_progress(float(header.get("ratio", -1.0)))
        elif message_type == "status" and self.on_status:
            self.on_status(str(header.get("message", "")))
        elif message_type == "error" and self.on_error:
            self.on_error(str(header.get("message", "")))
        elif message_type == "finished":
            self._finished = True
            if self.on_finished:
                self.on_finished()

    @staticmethod
    def _find_ros_setup_script() -> Optional[str]:
        override = os.environ.get("ROS_SETUP_SCRIPT")
        if override:
            return override if Path(override).exists() else None
        ros_distro = os.environ.get("ROS_DISTRO")
        if ros_distro:
            candidate = Path("/opt/ros") / ros_distro / "setup.bash"
            if candidate.exists():
                return str(candidate)
        opt_ros = Path("/opt/ros")
        if not opt_ros.exists():
            return None
        preferred = ("rolling", "jazzy", "iron", "humble", "galactic", "foxy", "eloquent", "dashing")
        distros = [p.name for p in opt_ros.iterdir() if p.is_dir()]
        ordered = [d for d in preferred if d in distros]
        ordered += sorted(d for d in distros if d not in preferred)
        for distro in ordered:
            distro_path = opt_ros / distro
            setup_path = distro_path / "setup.bash"
            if setup_path.exists():
                return str(setup_path)
        return None
