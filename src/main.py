#!/usr/bin/env python
"""Source/development entry point.

Importing ``startup`` performs the source runtime environment bootstrap
(``PYTHONPATH`` isolation and geospatial configuration); the frozen launcher
is ``frozen_main.py``.  See DECISION-018.
"""
from __future__ import annotations

import os
import sys

ALIGN_DIR = os.path.dirname(os.path.abspath(__file__))
if ALIGN_DIR not in sys.path:
    sys.path.insert(0, ALIGN_DIR)


def main() -> int:
    import startup  # noqa: F401  导入即初始化运行环境（路径隔离 + 地理环境）

    # 源码模式下同样支持「设置 → 运行设备」装好的 CUDA 运行时
    import device_runtime

    device_runtime.activate()

    from app_entry import run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
