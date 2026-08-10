"""Runtime paths shared by source and PyInstaller builds.

DECISION-018: files shipped with the application are read-only bundle
resources; configuration, caches, logs and user models live below the user's
local application-data directory so replacing a release cannot erase the
researcher's state.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


APP_DATA_NAME = "reUI"


def is_frozen() -> bool:
    """Return whether the current interpreter is a PyInstaller executable."""
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Return the repository/package root for source and frozen runs."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    # Source modules live below ``src``; data, docs and configuration remain
    # beside that directory at the project root.
    return Path(__file__).resolve().parents[1]


def executable_dir() -> Path:
    """Directory containing the launched executable in a frozen build."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return project_root()


def bundle_root() -> Path:
    """Return the source root or the directory containing ``reUI.exe``."""
    return executable_dir()


def _bundle_candidates() -> list[Path]:
    """Return possible roots for data in source and onedir layouts.

    PyInstaller's onedir layout may put support files in ``_internal`` while
    the executable remains in the top-level directory.  Searching both roots
    keeps resource resolution independent of that layout detail.  A source
    run deliberately does not search the repository's generated ``_internal``
    directory: tests and development runs must use the checked-out source
    resources, not a stale packaged copy.
    """
    candidates: list[Path] = []
    if is_frozen():
        roots = (
            bundle_root(),
            executable_dir() / "_internal",
            Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "_MEIPASS", "") else None,
        )
    else:
        roots = (project_root(), Path(__file__).resolve().parent)
    for candidate in roots:
        if not candidate:
            continue
        candidate = Path(candidate).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def bundled_resource(relative_path: str | os.PathLike) -> Path:
    """Resolve a read-only resource shipped with the application.

    A missing path is still returned relative to the preferred bundle root so
    callers can produce a useful error message.  The function never creates
    or writes bundle files.
    """
    raw = Path(relative_path)
    if raw.is_absolute():
        return raw.resolve()
    for root in _bundle_candidates():
        candidate = root / raw
        if candidate.exists():
            return candidate
    return bundle_root() / raw


def user_data_root() -> Path:
    """Return the writable per-user data directory.

    ``REUI_USER_DATA_DIR`` is intentionally supported for tests and portable
    lab deployments.  Normal Windows installs use ``%LOCALAPPDATA%``.
    """
    override = str(os.environ.get("REUI_USER_DATA_DIR", "") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    base = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    if not base:
        base = str(Path.home() / "AppData" / "Local")
    return (Path(base) / APP_DATA_NAME).resolve()


def config_dir() -> Path:
    return user_data_root() / "config"


def cache_dir() -> Path:
    return user_data_root() / "cache"


def logs_dir() -> Path:
    return user_data_root() / "logs"


def user_models_dir() -> Path:
    return user_data_root() / "models"


def updates_dir() -> Path:
    return user_data_root() / "updates"


def user_config_path() -> Path:
    return config_dir() / "config.json"


def legacy_config_path() -> Path:
    """Return the old source-tree config path used for one-time migration."""
    return project_root() / "config.json"


def builtin_model_dir() -> Path:
    return bundled_resource(Path("data") / "models")


def resolve_model_path(model_path: str | os.PathLike) -> Path:
    """Resolve configured model paths in both source and frozen builds."""
    raw = Path(str(model_path or "")).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        user_models_dir() / raw,
        bundled_resource(raw),
        builtin_model_dir() / raw.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[1].resolve()


def ensure_user_dirs() -> dict[str, Path]:
    """Create only writable user directories and return their paths."""
    paths = {
        "root": user_data_root(),
        "config": config_dir(),
        "cache": cache_dir(),
        "logs": logs_dir(),
        "models": user_models_dir(),
        "updates": updates_dir(),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
