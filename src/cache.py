"""项目缓存：保存影像对应的完整工作状态，支持跨会话继续处理。"""
import os, json, hashlib, datetime
import math
import threading
import time
import uuid
from contextlib import nullcontext
from typing import Optional, Tuple, Dict, Any
import numpy as np
from runtime_paths import cache_dir

# DECISION-002/018: cached project evidence is user data, not an install file.
CACHE_ROOT = str(cache_dir())
PROJECT_STATE_SCHEMA = 6
_PROJECT_LOCKS = {}
_PROJECT_LOCKS_GUARD = threading.Lock()
_PENDING_STATE_SAVES = {}
_ACTIVE_STATE_SAVERS = set()
_STATE_SAVE_QUEUE_LOCK = threading.Lock()


def source_identity(tif_path: str) -> Dict[str, Any]:
    """Cheap source identity used to reject stale spatial project state."""
    try:
        stat = os.stat(tif_path)
        sample_size = 65536
        digest = hashlib.sha256()
        with open(tif_path, "rb") as handle:
            digest.update(handle.read(sample_size))
            if stat.st_size > sample_size:
                handle.seek(max(0, stat.st_size - sample_size))
                digest.update(handle.read(sample_size))
        return {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "edge_sha256": digest.hexdigest(),
        }
    except (OSError, ValueError):
        return {}


def _project_lock(tif_path: str) -> threading.RLock:
    key = os.path.abspath(tif_path or "")
    with _PROJECT_LOCKS_GUARD:
        lock = _PROJECT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROJECT_LOCKS[key] = lock
        return lock


def _atomic_write_json(target: str, payload: dict, *, compact: bool = False) -> None:
    """Write JSON through a unique temp file and tolerate brief Windows locks."""
    temporary = (
        f"{target}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            if compact:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            else:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())

        delays = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temporary, target)
                return
            except OSError as exc:
                retryable = getattr(exc, "winerror", None) in (5, 32, 33)
                if not retryable or attempt == len(delays) - 1:
                    raise
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass


def _cache_dir(tif_path: str) -> str:
    h = hashlib.md5(tif_path.encode("utf-8")).hexdigest()[:12]
    return os.path.join(CACHE_ROOT, h)


def _meta_path(tif_path: str) -> str:
    return os.path.join(_cache_dir(tif_path), "meta.json")


def _mask_path(tif_path: str, suffix: str = "") -> str:
    return os.path.join(_cache_dir(tif_path), f"mask{suffix}.npz")


def _project_state_path(tif_path: str) -> str:
    return os.path.join(_cache_dir(tif_path), "project_state.json")


def _image_cache_path(tif_path: str) -> str:
    return os.path.join(_cache_dir(tif_path), "image_bgr.npy")


def _image_cache_meta_path(tif_path: str) -> str:
    return os.path.join(_cache_dir(tif_path), "image_bgr.json")


def _meta_path_from_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, "meta.json")


def _load_meta_from_dir(cache_dir: str) -> dict:
    p = _meta_path_from_dir(cache_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def has_cache(tif_path: str, suffix: str = "") -> bool:
    """检查是否有缓存的推理结果"""
    return os.path.exists(_mask_path(tif_path, suffix))


def save_mask(tif_path: str, mask: np.ndarray, offset_x: int = 0, offset_y: int = 0,
              suffix: str = "", meta: dict = None, commit_callback=None,
              extra_arrays: Optional[Dict[str, np.ndarray]] = None):
    """保存掩码到缓存"""
    d = _cache_dir(tif_path)
    os.makedirs(d, exist_ok=True)
    target = _mask_path(tif_path, suffix)
    temporary = (
        f"{target}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        payload = {
            "mask": np.asarray(mask),
            "offset_x": offset_x,
            "offset_y": offset_y,
        }
        for key, value in dict(extra_arrays or {}).items():
            array = np.asarray(value)
            if array.shape[:2] != np.asarray(mask).shape[:2]:
                raise ValueError(f"cached mask layer {key!r} has a different shape")
            payload[str(key)] = array
        np.savez_compressed(temporary, **payload)
        if commit_callback is not None and not bool(commit_callback()):
            return False
        os.replace(temporary, target)
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
    if meta and not save_meta(tif_path, meta, commit_callback=commit_callback):
        return False
    return True


def save_meta(tif_path: str, meta: dict, commit_callback=None) -> bool:
    """保存项目元数据，不要求掩码一定存在。"""
    if not tif_path:
        return False
    if commit_callback is not None and not bool(commit_callback()):
        return False
    with _project_lock(tif_path):
        if commit_callback is not None and not bool(commit_callback()):
            return False
        d = _cache_dir(tif_path)
        os.makedirs(d, exist_ok=True)
        merged = _load_meta_from_dir(d)
        merged.update(meta or {})
        _atomic_write_json(_meta_path(tif_path), merged)
    return True


def _jsonable(value):
    """Convert state values to JSON without persisting heavyweight image arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _mask_result_summary(mask_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not mask_result:
        return None
    # Dense semantic layers are stored together in mask_processed.npz.
    dense_mask_keys = {
        "processed_mask", "raw_mask", "debug_images", "regularized_body_mask",
        "body_residual_mask", "headland_mask", "uncertain_residual_mask",
        "gap_repair_added_mask", "support_clip_removed_mask",
        "headland_corridor_mask",
        "neutral_support_mask", "planning_support_mask",
    }
    keep = {
        key: value
        for key, value in mask_result.items()
        if key not in dense_mask_keys
    }
    return _jsonable(keep)


def _build_project_state_payload(tif_path: str, state, stage: str = "") -> Dict[str, Any]:
    """Freeze a consistent, JSON-safe project snapshot at request time."""
    if not tif_path or state is None:
        return {}

    auto_path = []
    for segment in getattr(state, "auto_path", []) or []:
        auto_path.append({
            "status": int(getattr(segment, "status", 1)),
            "segment_type": str(getattr(segment, "segment_type", "work")),
            "corridor_id": int(getattr(segment, "corridor_id", -1)),
            "length_m": float(getattr(segment, "length_m", 0.0)),
            "points": [
                [float(point.pixel_x), float(point.pixel_y)]
                for point in (getattr(segment, "points", []) or [])
            ],
        })

    source_sha256 = str(getattr(state, "source_sha256", "") or "")
    if not source_sha256 and os.path.isfile(tif_path):
        from provenance import file_sha256
        source_sha256 = file_sha256(tif_path)

    payload = {
        "schema": PROJECT_STATE_SCHEMA,
        "project_generation": int(getattr(state, "project_generation", 0) or 0),
        "source_identity": source_identity(tif_path),
        "source_sha256": source_sha256,
        "source_metadata": _jsonable(getattr(state, "source_metadata", {}) or {}),
        "raster_preprocessing": _jsonable(getattr(state, "raster_preprocessing", {}) or {}),
        "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": str(stage or ""),
        "tif_path": os.path.abspath(tif_path),
        "field_boundary": _jsonable(getattr(state, "field_boundary", [])),
        "field_area_m2": float(getattr(state, "field_area_m2", 0.0) or 0.0),
        "forbidden_regions": _jsonable(getattr(state, "forbidden_regions", [])),
        "forbidden_regions_confirmed": bool(
            getattr(state, "forbidden_regions_confirmed", False)
        ),
        "current_model_name": str(getattr(state, "current_model_name", "") or ""),
        "current_model_path": str(getattr(state, "current_model_path", "") or ""),
        "model_sha256": str(getattr(state, "model_sha256", "") or ""),
        "inference_provenance": _jsonable(getattr(state, "inference_provenance", {}) or {}),
        "inference_runtime": _jsonable(getattr(state, "inference_runtime", {}) or {}),
        "inference_done": bool(getattr(state, "inference_done", False)),
        "mask_processed": bool(getattr(state, "mask_processed", False)),
        "mask_offset_x": int(getattr(state, "mask_offset_x", 0)),
        "mask_offset_y": int(getattr(state, "mask_offset_y", 0)),
        "mask_result": _mask_result_summary(getattr(state, "mask_result", None)),
        "mask_provenance": _jsonable(getattr(state, "mask_provenance", {}) or {}),
        "harvester_params": _jsonable(getattr(state, "harvester_params", {})),
        "harvester_param_sources": _jsonable(
            getattr(state, "harvester_param_sources", {})
        ),
        "harvester_params_confirmed": bool(
            getattr(state, "harvester_params_confirmed", False)
        ),
        "turn_strategy": str(getattr(state, "turn_strategy", "bow") or "bow"),
        "entry_point": _jsonable(getattr(state, "entry_point", None)),
        "exit_point": _jsonable(getattr(state, "exit_point", None)),
        "unload_points": _jsonable(getattr(state, "unload_points", [])),
        "unload_point": _jsonable(
            (getattr(state, "unload_points", []) or [getattr(state, "unload_point", None)])[0]
        ),
        "entry_point_locked": bool(getattr(state, "entry_point_locked", False)),
        "exit_point_locked": bool(getattr(state, "exit_point_locked", False)),
        "unload_point_locked": bool(getattr(state, "unload_point_locked", False)),
        "workflow_step": int(getattr(state, "workflow_step", 0)),
        "simulation_done": bool(getattr(state, "simulation_done", False)),
        "export_done": bool(getattr(state, "export_done", False)),
        "auto_path_planned": bool(getattr(state, "auto_path_planned", False)),
        "auto_path_valid": bool(getattr(state, "auto_path_valid", False)),
        "path_validation_snapshot": {
            "valid_at_save_time": bool(getattr(state, "auto_path_valid", False)),
            "source": "current_run" if bool(getattr(state, "auto_path_valid", False)) else "unverified",
        },
        "auto_path_desc": str(getattr(state, "auto_path_desc", "") or ""),
        "path_provenance": _jsonable(getattr(state, "path_provenance", {}) or {}),
        "path_runtime": _jsonable(getattr(state, "path_runtime", {}) or {}),
        "path_validation": _jsonable(getattr(state, "path_validation", {}) or {}),
        "path_artifact": _jsonable(getattr(state, "path_artifact", {}) or {}),
        "auto_path": auto_path,
        "auto_path_geo": _jsonable(getattr(state, "auto_path_geo", [])),
        "path_points": _jsonable(getattr(state, "path_points", [])),
        "path_status": _jsonable(getattr(state, "path_status", [])),
        "last_total_path_m": float(getattr(state, "last_total_path_m", 0.0) or 0.0),
        "last_work_path_m": float(getattr(state, "last_work_path_m", 0.0) or 0.0),
        "last_turn_path_m": float(getattr(state, "last_turn_path_m", 0.0) or 0.0),
        "last_entry_exit_path_m": float(getattr(state, "last_entry_exit_path_m", 0.0) or 0.0),
        "last_crop_mask_geometric_coverage_pct": float(
            getattr(state, "last_crop_mask_geometric_coverage_pct", 0.0) or 0.0
        ),
        "last_planned_target_geometric_coverage_pct": float(
            getattr(state, "last_planned_target_geometric_coverage_pct", 0.0) or 0.0
        ),
        "last_detected_mask_geometric_coverage_pct": float(
            getattr(state, "last_detected_mask_geometric_coverage_pct", 0.0) or 0.0
        ),
        "last_rolling_rate": float(getattr(state, "last_rolling_rate", 0.0) or 0.0),
        "last_field_efficiency": float(
            getattr(state, "last_field_efficiency", 0.0) or 0.0
        ),
        "view_scale": float(getattr(state, "view_scale", 1.0) or 1.0),
        "view_x": float(getattr(state, "view_x", 0.0) or 0.0),
        "view_y": float(getattr(state, "view_y", 0.0) or 0.0),
        "view_rotation_deg": float(getattr(state, "view_rotation_deg", 0.0) or 0.0),
    }

    return payload


def _write_project_state_payload(tif_path: str, payload: Dict[str, Any]) -> bool:
    if not tif_path or not payload:
        return False
    os.makedirs(_cache_dir(tif_path), exist_ok=True)
    target = _project_state_path(tif_path)
    incoming_generation = int(payload.get("project_generation", 0) or 0)
    try:
        with open(target, "r", encoding="utf-8") as handle:
            current_generation = int((json.load(handle) or {}).get("project_generation", 0) or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        current_generation = -1
    if incoming_generation < current_generation:
        return False
    _atomic_write_json(target, payload, compact=True)
    return True


def save_project_state(tif_path: str, state, stage: str = "", commit_callback=None) -> bool:
    """Serialize one project's state in call order without temp-file collisions."""
    if not tif_path or state is None:
        return False
    state_lock = getattr(state, "state_lock", None)
    with state_lock if state_lock is not None else nullcontext():
        payload = _build_project_state_payload(tif_path, state, stage)
    if commit_callback is not None and not bool(commit_callback()):
        return False
    with _project_lock(tif_path):
        if commit_callback is not None and not bool(commit_callback()):
            return False
        return _write_project_state_payload(tif_path, payload)


def request_project_state_save(
    tif_path: str,
    state,
    stage: str = "",
    on_error=None,
) -> None:
    """Coalesce UI save bursts and perform serialization outside the UI thread."""
    if not tif_path or state is None:
        return
    key = os.path.abspath(tif_path)
    state_lock = getattr(state, "state_lock", None)
    with state_lock if state_lock is not None else nullcontext():
        payload = _build_project_state_payload(tif_path, state, stage)
    with _STATE_SAVE_QUEUE_LOCK:
        _PENDING_STATE_SAVES[key] = (tif_path, payload, stage, on_error)
        if key in _ACTIVE_STATE_SAVERS:
            return
        _ACTIVE_STATE_SAVERS.add(key)

    def worker():
        while True:
            with _STATE_SAVE_QUEUE_LOCK:
                item = _PENDING_STATE_SAVES.pop(key, None)
                if item is None:
                    _ACTIVE_STATE_SAVERS.discard(key)
                    return
            path, current_payload, current_stage, error_callback = item
            try:
                with _project_lock(path):
                    _write_project_state_payload(path, current_payload)
            except Exception as exc:
                if error_callback is not None:
                    try:
                        error_callback(current_stage, exc)
                    except Exception:
                        pass

    threading.Thread(
        target=worker,
        name=f"project-save-{hashlib.md5(key.encode('utf-8')).hexdigest()[:8]}",
        daemon=True,
    ).start()


def wait_for_project_state_saves(tif_path: str = "", timeout: float = 10.0) -> bool:
    """Wait for queued saves; primarily used on shutdown and by regression tests."""
    key = os.path.abspath(tif_path) if tif_path else ""
    deadline = time.time() + max(0.0, timeout)
    while time.time() <= deadline:
        with _STATE_SAVE_QUEUE_LOCK:
            if key:
                busy = key in _ACTIVE_STATE_SAVERS or key in _PENDING_STATE_SAVES
            else:
                busy = bool(_ACTIVE_STATE_SAVERS or _PENDING_STATE_SAVES)
        if not busy:
            return True
        time.sleep(0.01)
    return False


def load_project_state(tif_path: str, expected_source_sha256: str = "") -> Dict[str, Any]:
    """DECISION-002: load a snapshot only when its exact source still matches."""
    path = _project_state_path(tif_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        schema = int(payload.get("schema", 0))
        if schema != PROJECT_STATE_SCHEMA:
            return {}
        saved_identity = payload.get("source_identity") or {}
        current_identity = source_identity(tif_path)
        if not saved_identity or saved_identity != current_identity:
            return {}
        saved_source_sha256 = str(payload.get("source_sha256", "") or "")
        if not saved_source_sha256:
            return {}
        if expected_source_sha256:
            current_source_sha256 = str(expected_source_sha256)
        else:
            from provenance import file_sha256
            current_source_sha256 = file_sha256(tif_path)
        if saved_source_sha256 != current_source_sha256:
            return {}
        return payload
    except Exception:
        return {}


def load_image_cache(tif_path: str) -> Optional[np.ndarray]:
    """Memory-map an exact full-resolution BGR cache when the source is unchanged."""
    image_path = _image_cache_path(tif_path)
    meta_path = _image_cache_meta_path(tif_path)
    if not os.path.exists(image_path) or not os.path.exists(meta_path):
        return None
    try:
        source = os.stat(tif_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("source_size", -1)) != int(source.st_size):
            return None
        if abs(float(meta.get("source_mtime", -1.0)) - float(source.st_mtime)) > 1e-3:
            return None
        image = np.load(image_path, mmap_mode="r")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return None
        return image
    except Exception:
        return None


def save_image_cache(tif_path: str, image_bgr: np.ndarray) -> None:
    """Write an exact cache atomically; intended for a background thread."""
    if not tif_path or image_bgr is None:
        return
    d = _cache_dir(tif_path)
    os.makedirs(d, exist_ok=True)
    target = _image_cache_path(tif_path)
    temporary = target + ".tmp.npy"
    np.save(temporary, np.asarray(image_bgr, dtype=np.uint8), allow_pickle=False)
    os.replace(temporary, target)
    source = os.stat(tif_path)
    meta_target = _image_cache_meta_path(tif_path)
    meta_temporary = meta_target + ".tmp"
    with open(meta_temporary, "w", encoding="utf-8") as f:
        json.dump({
            "source_size": int(source.st_size),
            "source_mtime": float(source.st_mtime),
            "shape": list(image_bgr.shape),
        }, f, ensure_ascii=False, indent=2)
    os.replace(meta_temporary, meta_target)


def load_mask_bundle(
    tif_path: str,
    suffix: str = "",
) -> Tuple[Optional[np.ndarray], int, int, Dict[str, np.ndarray]]:
    """Load a mask and its semantic layers from one atomic artifact."""
    p = _mask_path(tif_path, suffix)
    if not os.path.exists(p):
        return None, 0, 0, {}
    with np.load(p, allow_pickle=False) as data:
        mask = np.asarray(data["mask"]).copy()
        extras = {
            key: np.asarray(data[key]).copy()
            for key in data.files
            if key not in {"mask", "offset_x", "offset_y"}
        }
        return mask, int(data["offset_x"]), int(data["offset_y"]), extras


def load_mask(tif_path: str, suffix: str = "") -> Tuple[Optional[np.ndarray], int, int]:
    """加载缓存的掩码"""
    p = _mask_path(tif_path, suffix)
    if not os.path.exists(p):
        return None, 0, 0
    with np.load(p, allow_pickle=False) as data:
        return (
            np.asarray(data["mask"]).copy(),
            int(data["offset_x"]),
            int(data["offset_y"]),
        )


def _load_payload(mask_path: str, cache_dir: str) -> Optional[Dict[str, Any]]:
    try:
        with np.load(mask_path, allow_pickle=False) as data:
            return {
                "mask": np.asarray(data["mask"]).copy(),
                "offset_x": int(data["offset_x"]),
                "offset_y": int(data["offset_y"]),
                "meta": _load_meta_from_dir(cache_dir),
                "cache_dir": cache_dir,
            }
    except Exception:
        return None


def load_preferred_cache(tif_path: str,
                         image_shape: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    """Load only caches bound to the exact source path; never match by shape."""
    if has_cache(tif_path, suffix="_best"):
        return _load_payload(_mask_path(tif_path, "_best"), _cache_dir(tif_path))
    if has_cache(tif_path):
        return _load_payload(_mask_path(tif_path), _cache_dir(tif_path))
    return None


def load_preferred_mask(tif_path: str,
                        image_shape: Optional[Tuple[int, int]] = None) -> Tuple[Optional[np.ndarray], int, int]:
    payload = load_preferred_cache(tif_path, image_shape=image_shape)
    if not payload:
        return None, 0, 0
    return payload["mask"], payload["offset_x"], payload["offset_y"]


def load_meta(tif_path: str) -> dict:
    """加载缓存元数据"""
    return _load_meta_from_dir(_cache_dir(tif_path))


def clear_cached_masks(tif_path: str, suffixes=None):
    """清除当前影像关联的掩码缓存，保留 meta.json。"""
    if not tif_path:
        return
    clear_cached_masks_in_dir(_cache_dir(tif_path), suffixes=suffixes)


def clear_cached_masks_in_dir(cache_dir: str, suffixes=None):
    if not cache_dir:
        return
    suffixes = tuple(suffixes or ("", "_best", "_processed", "_smoothed"))
    for suffix in suffixes:
        p = os.path.join(cache_dir, f"mask{suffix}.npz")
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def save_bands_json(tif_path: str, bands_data: list):
    """保存种植带数据（轻量 JSON）"""
    d = _cache_dir(tif_path)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "bands.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bands_data, f, ensure_ascii=False, indent=2)
