"""PyInstaller entry point; bypasses the source launcher entirely (DECISION-018)."""
from __future__ import annotations

import sys

from app_entry import run_app


if __name__ == "__main__":
    sys.exit(run_app())

