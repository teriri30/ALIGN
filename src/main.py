#!/usr/bin/env python
"""Source/development entry point.

The source launcher may still offer dependency installation.  The frozen
launcher is ``frozen_main.py`` and never imports that installer.
"""
from __future__ import annotations

import os
import sys

REUI_DIR = os.path.dirname(os.path.abspath(__file__))
if REUI_DIR not in sys.path:
    sys.path.insert(0, REUI_DIR)


def main() -> int:
    try:
        from startup import ensure_environment
        if not ensure_environment():
            return 0
    except Exception as exc:
        print(f"环境检查失败，程序已停止: {exc}", file=sys.stderr)
        return 1

    from app_entry import run_app
    return run_app()


if __name__ == "__main__":
    sys.exit(main())
