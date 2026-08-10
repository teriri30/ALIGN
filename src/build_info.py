"""Build metadata loading and generation for source and frozen runs."""
from __future__ import annotations

import datetime as _datetime
import json
import os
from pathlib import Path
from typing import Any

from runtime_paths import bundled_resource
from version import APP_VERSION


SCIENTIFIC_CODE_PATHS = [
    "src/geo.py",
    "src/inference.py",
    "src/raster_preprocessing.py",
    "src/mask_processor.py",
    "src/row_geometry.py",
    "src/legacy_route_coverage.py",
    "src/footprint_planner.py",
    "src/path_planner.py",
    "src/provenance.py",
    "src/config.py",
    "src/cache.py",
    "src/pyside6_app/workers.py",
]


def load_build_info() -> dict[str, Any]:
    """Load generated metadata, falling back safely during source runs."""
    path = bundled_resource("build_info.json")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and payload.get("version"):
            return payload
    except (OSError, ValueError, TypeError):
        pass
    return {
        "schema": 1,
        "version": APP_VERSION,
        "commit_sha": "",
        "channel": "development",
        "build_time_utc": "",
        "scientific_code": {"files": {}, "fingerprint": ""},
        "generated": False,
    }


def make_build_info(root: str | os.PathLike, *, version: str | None = None,
                    channel: str = "stable") -> dict[str, Any]:
    """Create reproducible build metadata before a PyInstaller build."""
    from provenance import code_identity, git_is_dirty, git_revision

    root_path = Path(root).resolve()
    return {
        "schema": 1,
        "version": str(version or APP_VERSION),
        "commit_sha": git_revision(root_path),
        "channel": str(channel or "stable"),
        "build_time_utc": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat(timespec="seconds"),
        "git_dirty": git_is_dirty(root_path),
        "scientific_code": code_identity(root_path, SCIENTIFIC_CODE_PATHS),
        "generated": True,
    }
