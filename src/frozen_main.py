"""PyInstaller entry point; bypasses the source launcher entirely (DECISION-018)."""
from __future__ import annotations

import sys

# 必须在 app_entry（以及任何 torch 导入）之前挂载 GPU 运行时：
# 内置的是 CPU 版 torch，用户在「设置 → 运行设备」里装过 CUDA 版时走这里。
import device_runtime

device_runtime.activate()

from app_entry import run_app  # noqa: E402


if __name__ == "__main__":
    sys.exit(run_app())

