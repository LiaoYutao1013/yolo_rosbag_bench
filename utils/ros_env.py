from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec


ROS2_PYTHON_MODULES = (
    "rclpy",
    "rosbag2_py",
    "sensor_msgs",
    "cv_bridge",
)


@lru_cache(maxsize=1)
def ros2_available() -> bool:
    """Return True when the current Python environment can import ROS 2 modules.

    ROS 2 Python packages are usually provided by a system ROS installation or
    a conda environment created with `ros-humble-*` packages, not by pip.
    """
    return all(find_spec(name) is not None for name in ROS2_PYTHON_MODULES)


def missing_ros2_modules() -> list[str]:
    return [name for name in ROS2_PYTHON_MODULES if find_spec(name) is None]


def ensure_ros2() -> None:
    """Raise a clear, actionable error when ROS 2 is unavailable."""
    if ros2_available():
        return
    missing = ", ".join(missing_ros2_modules())
    raise RuntimeError(
        "当前 Python 环境缺少 ROS 2 Python 包，无法读取 rosbag 或订阅 ROS Topic。"
        f"缺失模块: {missing}。请先 source ROS 2 环境（例如 source /opt/ros/$ROS_DISTRO/setup.bash）"
        "，或使用包含 rosbag2、cv_bridge 等 ROS Python 包的 conda 环境。"
        "如果只做本地图片 Validation，可直接使用 Run validation，无需启动 rosbag/live 播放。"
    )
