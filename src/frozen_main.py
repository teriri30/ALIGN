"""PyInstaller entry point; deliberately bypasses source dependency setup."""
from __future__ import annotations

import sys

from app_entry import run_app


if __name__ == "__main__":
    sys.exit(run_app())

