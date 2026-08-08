"""Project-session lifecycle helpers extracted from the main window."""
from copy import deepcopy
import math


_PATH_DERIVED_FIELDS = (
    "auto_path",
    "auto_path_segments",
    "auto_path_geo",
    "path_points",
    "path_status",
    "path_artifact",
)


def _is_canonical_path_segment(segment) -> bool:
    """Return whether a serialized segment retains the route semantics needed to edit."""
    if not isinstance(segment, dict):
        return False
    if not str(segment.get("segment_type") or "").strip():
        return False
    try:
        length_m = float(segment["length_m"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(length_m) or length_m < 0.0:
        return False
    points = segment.get("points")
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        return False
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(x) or not math.isfinite(y):
            return False
    return True


def cached_path_restore_plan(saved, *, path_cache_valid: bool) -> dict:
    """Plan a fail-closed cache restore without inferring a route from derived data.

    ``auto_path`` is the only serialized source that retains segment boundaries,
    segment semantics, and explicit lengths.  Flattened path points and geo
    records are display derivatives and cannot be promoted back into a route.
    """
    saved = saved if isinstance(saved, dict) else {}
    canonical_segments = saved.get("auto_path")
    has_canonical_path = (
        bool(path_cache_valid)
        and isinstance(canonical_segments, (list, tuple))
        and bool(canonical_segments)
        and all(_is_canonical_path_segment(segment) for segment in canonical_segments)
    )
    historical_path_present = any(
        bool(saved.get(field))
        for field in (*_PATH_DERIVED_FIELDS, "auto_path_planned")
    )
    common_updates = {
        "auto_path_valid": False,
        "simulation_done": False,
        "export_done": False,
    }
    if has_canonical_path:
        return {
            "restore_editable_path": True,
            "historical_path_present": historical_path_present,
            "state_updates": {
                **common_updates,
                "auto_path": deepcopy(list(canonical_segments)),
                "auto_path_segments": deepcopy(list(canonical_segments)),
                "auto_path_geo": [],
                "path_points": [],
                "path_status": [],
                "auto_path_planned": True,
                "path_artifact": {},
            },
        }

    return {
        "restore_editable_path": False,
        "historical_path_present": historical_path_present,
        "state_updates": {
            **common_updates,
            **{field: [] for field in _PATH_DERIVED_FIELDS},
            "auto_path_planned": False,
            "auto_path_desc": "",
            "path_validation": {},
            "path_provenance": {},
            "path_runtime": {},
            "path_artifact": {},
        },
    }


def reset_project_state(state, *, preserve_model=True, preserve_machine=True) -> int:
    """Reset image-owned state and return the new monotonically increasing generation."""
    generation = int(getattr(state, "project_generation", 0) or 0) + 1
    preserved = {}
    if preserve_machine:
        preserved.update(
            harvester_params=deepcopy(getattr(state, "harvester_params", {}) or {}),
            harvester_param_sources=deepcopy(
                getattr(state, "harvester_param_sources", {}) or {}
            ),
            turn_strategy=str(getattr(state, "turn_strategy", "auto") or "auto"),
        )
    if preserve_model:
        preserved.update(
            current_model_name=str(getattr(state, "current_model_name", "") or ""),
            current_model_path=str(getattr(state, "current_model_path", "") or ""),
            model_sha256=str(getattr(state, "model_sha256", "") or ""),
            cache_model_matches_active_engine=bool(
                getattr(state, "cache_model_matches_active_engine", True)
            ),
            model_list=deepcopy(getattr(state, "model_list", []) or []),
        )
    state.reset()
    state.project_generation = generation
    for name, value in preserved.items():
        setattr(state, name, value)
    return generation


def generation_matches(state, generation) -> bool:
    """Return whether an asynchronous result belongs to the current project."""
    return int(generation) == int(getattr(state, "project_generation", 0) or 0)
