"""
workers.py — PySide6 后台工作线程

处理推理、路径规划等耗时操作，不阻塞 UI。
"""
from typing import Optional, Dict, List, Tuple, Callable
from types import SimpleNamespace
import traceback
import os
import time

import numpy as np

from PySide6.QtCore import QThread, Signal, QObject


QT_RENDER_MAX_DIMENSION = 4096


class PaperExperimentWorker(QObject):
    """Run a frozen paper-experiment matrix outside the UI thread."""

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, runner, plan, token=None, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.plan = dict(plan or {})
        self.token = token
        self._cancelled = False
        self.runner.progress_callback = self._forward_progress
        self.runner.cancel_callback = lambda: self._cancelled or bool(
            getattr(self.token, "invalidated", False)
        )

    def cancel(self):
        self._cancelled = True

    def _forward_progress(self, value, message):
        self.progress.emit(int(max(0.0, min(1.0, float(value))) * 100), str(message))

    def run(self):
        try:
            if self.runner.cancel_callback():
                self.finished.emit({
                    "task_token": self.token,
                    "status": "cancelled",
                    "result": None,
                    "error": "",
                })
                return
            output_dir = self.runner.run(plan=self.plan)
            if self.runner.cancel_callback():
                self.finished.emit({
                    "task_token": self.token,
                    "status": "cancelled",
                    "result": None,
                    "error": "",
                })
                return
            self.progress.emit(100, "论文实验资料包已生成")
            self.finished.emit({
                "task_token": self.token,
                "status": "completed",
                "result": {"output_dir": str(output_dir)},
                "error": "",
            })
        except Exception as exc:
            self.finished.emit({
                "task_token": self.token,
                "status": "failed",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            })


class MaskDiagnosticExportWorker(QObject):
    """Build a mask diagnostic ZIP without blocking the GUI thread."""

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, output_path, raw_mask, mask_result, options=None, parent=None):
        super().__init__(parent)
        self.output_path = str(output_path)
        self.raw_mask = raw_mask
        self.mask_result = dict(mask_result or {})
        self.options = dict(options or {})
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                return
            self.progress.emit(10, "正在分析掩膜连通区域...")
            from mask_diagnostics import export_mask_diagnostic_bundle
            result = export_mask_diagnostic_bundle(
                self.output_path,
                self.raw_mask,
                self.mask_result,
                **self.options,
            )
            if self._cancelled:
                return
            self.progress.emit(100, "掩膜诊断包已生成")
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


def display_affine(source_affine, full_width, full_height, display_width, display_height):
    """Map display pixel centres to the exact source raster footprint."""
    from affine import Affine

    if min(full_width, full_height, display_width, display_height) <= 0:
        raise ValueError("raster dimensions must be positive")
    return source_affine * Affine.scale(
        float(full_width) / float(display_width),
        float(full_height) / float(display_height),
    )


def preview_shape(full_height, full_width, max_dimension):
    """Return an aspect-preserving, bounded preview shape for UI rendering."""
    full_height, full_width, max_dimension = (
        int(full_height), int(full_width), int(max_dimension)
    )
    if min(full_height, full_width, max_dimension) <= 0:
        raise ValueError("raster dimensions and maximum preview dimension must be positive")
    longest = max(full_height, full_width)
    if longest <= max_dimension:
        return full_height, full_width
    scale = float(max_dimension) / float(longest)
    return (
        max(1, int(round(full_height * scale))),
        max(1, int(round(full_width * scale))),
    )


def bounded_render_preview(image, max_dimension=QT_RENDER_MAX_DIMENSION):
    """DECISION-004: bound Qt texture size without mutating analysis pixels."""
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.size == 0:
        raise ValueError("preview image must be a non-empty 2D or 3D array")
    render_h, render_w = preview_shape(array.shape[0], array.shape[1], max_dimension)
    if (render_h, render_w) == array.shape[:2]:
        return np.ascontiguousarray(array)
    import cv2
    return np.ascontiguousarray(
        cv2.resize(array, (render_w, render_h), interpolation=cv2.INTER_AREA)
    )


def require_metric_scale(geo, state, shape) -> float:
    """DECISION-003: return verified metres-per-pixel or stop processing.

    Crop geometry and machine dimensions are physical quantities. Missing,
    guessed, anisotropic or spatially unstable scale must never be replaced by
    a convenient pixel fallback. See docs/DECISIONS.md.
    """
    if geo is None or not geo.is_ready():
        raise RuntimeError("影像没有有效地理配准，不能执行米制掩膜处理或路径规划")
    if state is None or shape is None or len(shape) < 2:
        raise RuntimeError("缺少影像状态或掩膜尺寸，不能验证地面分辨率")
    height, width = int(shape[0]), int(shape[1])
    offset_x = float(getattr(state, "mask_offset_x", 0) or 0)
    offset_y = float(getattr(state, "mask_offset_y", 0) or 0)
    value = float(
        geo.meters_per_pixel(offset_x + width / 2.0, offset_y + height / 2.0)
    )
    if not np.isfinite(value) or not 1e-5 <= value <= 5.0:
        raise RuntimeError(f"影像地面分辨率无效: {value!r} m/px")
    if hasattr(geo, "pixel_distance_m"):
        samples = [
            (offset_x + width / 2.0, offset_y + height / 2.0),
            (offset_x, offset_y),
            (offset_x + max(0, width - 2), offset_y),
            (offset_x, offset_y + max(0, height - 2)),
            (offset_x + max(0, width - 2), offset_y + max(0, height - 2)),
        ]
        means = []
        anisotropy = []
        for px, py in samples:
            dx = float(geo.pixel_distance_m((px, py), (px + 1.0, py)))
            dy = float(geo.pixel_distance_m((px, py), (px, py + 1.0)))
            if not np.isfinite(dx) or not np.isfinite(dy) or min(dx, dy) <= 0.0:
                raise RuntimeError("影像横向或纵向地面分辨率无效")
            means.append((dx + dy) / 2.0)
            anisotropy.append(max(dx, dy) / min(dx, dy))
        from config import Config
        geo_config = Config().section("geo")
        max_anisotropy = float(geo_config.get("max_pixel_anisotropy_ratio", 1.15))
        max_variation = float(geo_config.get("max_gsd_variation_ratio", 0.10))
        if max(anisotropy) > max_anisotropy:
            raise RuntimeError(
                f"影像像素横纵尺度差异过大: ratio={max(anisotropy):.4f}, "
                f"limit={max_anisotropy:.4f}"
            )
        variation = max(means) / min(means) - 1.0
        if variation > max_variation:
            raise RuntimeError(
                f"影像范围内地面分辨率变化过大: variation={variation:.2%}, "
                f"limit={max_variation:.2%}"
            )
        value = float(np.median(means))
    return value


def path_planning_config_snapshot(runtime_config=None):
    """DECISION-013: freeze one default-plus-runtime planner configuration."""
    from config import Config

    config = Config().section("path_planning")
    config.update(dict(runtime_config or {}))
    config.setdefault("headland_damage_weight", 1.0)
    return config


def planning_config_with_mask_evidence(config_snapshot, mask_result, state, shape):
    """DECISION-009/013: inject the same semantic evidence into every planner path."""
    config = dict(config_snapshot or {})
    mask_result = dict(mask_result or {})
    config["headland_mask"] = mask_result.get("headland_mask")
    config["planning_support_mask"] = mask_result.get("planning_support_mask")
    config["neutral_support_mask"] = mask_result.get("neutral_support_mask")
    config["uncertain_mask"] = mask_result.get("uncertain_residual_mask")

    from footprint_planner import rasterize_forbidden_regions

    forbidden_confirmed = (
        getattr(state, "forbidden_regions_confirmed", False) is True
    )
    forbidden_mask = rasterize_forbidden_regions(shape, state)
    if forbidden_confirmed and forbidden_mask is None:
        forbidden_mask = np.zeros(shape[:2], dtype=np.uint8)
    config["forbidden_mask"] = forbidden_mask
    config["forbidden_mask_confirmed"] = forbidden_confirmed
    return config


def semantic_mask_fingerprints(executed_config):
    """DECISION-001/002/009: bind every planner semantic mask by content."""
    from provenance import artifact_sha256

    evidence = dict(executed_config or {})
    return {
        key: artifact_sha256(evidence.get(key))
        for key in (
            "headland_mask",
            "planning_support_mask",
            "neutral_support_mask",
            "uncertain_mask",
            "forbidden_mask",
        )
    }


def attach_path_provenance(path_result, mask_result, state, executed_config):
    """DECISION-001/013: bind a completed path to its exact scalar inputs."""
    from path_planner import PATH_PLANNING_VERSION
    from provenance import make_stage_record

    excluded = {
        "headland_mask", "planning_support_mask", "uncertain_residual_mask",
        "uncertain_mask", "forbidden_mask", "neutral_support_mask",
    }
    path_config = {
        key: value for key, value in dict(executed_config or {}).items()
        if not str(key).startswith("_") and key not in excluded
    }
    path_inputs = {
        "mask_fingerprint": str(
            (dict(mask_result or {}).get("provenance", {}) or {}).get(
                "fingerprint", ""
            )
        ),
        "path_config": path_config,
        "harvester": dict(getattr(state, "harvester_params", {}) or {}),
        "turn_strategy": str(getattr(state, "turn_strategy", "auto")),
        "entry_point": getattr(state, "entry_point", None),
        "exit_point": getattr(state, "exit_point", None),
        "unload_points": list(getattr(state, "unload_points", []) or []),
        "forbidden_regions": list(getattr(state, "forbidden_regions", []) or []),
        "forbidden_regions_confirmed": bool(
            getattr(state, "forbidden_regions_confirmed", False)
        ),
        "semantic_mask_fingerprints": semantic_mask_fingerprints(executed_config),
    }
    path_artifact = {
        "full_path": path_result.get("full_path", []),
        "segment_types": path_result.get("segment_types", []),
        "layout": path_result.get("layout", {}),
    }
    path_result["provenance"] = make_stage_record(
        "path", PATH_PLANNING_VERSION, path_inputs, path_artifact
    )
    return path_result


class TifLoadWorker(QObject):
    """Load/downsample GeoTIFF data outside the GUI thread."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit(5, "正在打开影像...")
            import rasterio
            from rasterio.enums import Resampling
            from pyproj import Transformer
            from config import Config
            from provenance import file_sha256
            from raster_preprocessing import read_rgb_raster
            with rasterio.open(self.path) as src:
                crs = src.crs
                transform = src.transform
                transformer = None
                geo_error = ""
                determinant = float(transform.a * transform.e - transform.b * transform.d)
                has_valid_affine = not bool(transform.is_identity) and abs(determinant) > 1e-18
                if crs and has_valid_affine:
                    try:
                        # Every geographic display/export is WGS84 longitude,
                        # latitude. Never substitute a guessed projected CRS.
                        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    except Exception as exc:
                        geo_error = f"CRS transformation unavailable: {exc}"
                elif crs:
                    geo_error = "GeoTIFF has a CRS but no valid affine geotransform"
                else:
                    geo_error = "GeoTIFF has no CRS"
                full_h, full_w = src.height, src.width
                # Display should not load the full 15k×21k raster into a QPixmap.
                # Keep the longest display side bounded; processing uses masks/caches,
                # not this RGB preview.
                max_display_dim = 6000
                downsample = max(1, int(np.ceil(max(full_h, full_w) / max_display_dim)))
                out_h = max(1, full_h // downsample)
                out_w = max(1, full_w // downsample)
                self.progress.emit(25, f"正在读取影像预览 {out_w}×{out_h}...")
                rgb, preview_preprocessing = read_rgb_raster(
                    src,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.average,
                    config=Config().section("raster_preprocessing"),
                )
                source_metadata = {
                    "width": int(src.width),
                    "height": int(src.height),
                    "band_count": int(src.count),
                    "dtypes": list(src.dtypes),
                    "nodata": [
                        None if value is None or not np.isfinite(float(value)) else float(value)
                        for value in src.nodatavals
                    ],
                    "color_interpretation": [
                        getattr(value, "name", str(value)) for value in src.colorinterp
                    ],
                    "crs": str(crs) if crs else "",
                    "affine": [float(value) for value in tuple(transform)[:6]],
                    "overview_factors": list(src.overviews(1)) if src.count else [],
                    "rgb_band_indexes": list(preview_preprocessing.get("source_band_indexes", [])),
                    "rgb_selection_assumed": not all(
                        name in [str(value).lower() for value in preview_preprocessing.get("source_color_interpretation", [])]
                        for name in ("red", "green", "blue")
                    ),
                }
            if self._cancelled:
                return
            render_rgb = bounded_render_preview(rgb)
            preview_preprocessing.update({
                "analysis_preview_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
                "render_preview_shape": [
                    int(render_rgb.shape[0]), int(render_rgb.shape[1])
                ],
                "render_preview_max_dimension": QT_RENDER_MAX_DIMENSION,
            })
            self.progress.emit(75, "正在计算源影像完整性哈希...")
            source_sha256 = file_sha256(self.path)
            self.finished.emit({
                "path": self.path,
                "rgb": rgb,
                "render_rgb": render_rgb,
                "full_h": full_h,
                "full_w": full_w,
                "display_h": out_h,
                "display_w": out_w,
                "downsample": downsample,
                "crs": str(crs) if crs else "",
                "transform": display_affine(
                    transform, full_w, full_h, out_w, out_h
                ),
                "source_transform": transform,
                "transformer": transformer,
                "geo_error": geo_error,
                "source_sha256": source_sha256,
                "source_metadata": source_metadata,
                "preview_preprocessing": preview_preprocessing,
            })
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class CacheRestoreWorker(QObject):
    """DECISION-002: restore only artifacts with a complete valid evidence chain.

    Source, model, configuration, upstream fingerprint and artifact hash are
    checked stage by stage. A mismatch invalidates that stage and all of its
    dependants instead of guessing legacy compatibility.
    """
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, path: str, expected_source_sha256: str = "", token=None, parent=None):
        super().__init__(parent)
        self.path = path
        self.expected_source_sha256 = str(expected_source_sha256 or "")
        self.token = token
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _is_cancelled(self):
        return self._cancelled or bool(getattr(self.token, "invalidated", False))

    def _finish(self, status, result=None, error=""):
        self.finished.emit({
            "task_token": self.token,
            "status": status,
            "result": result,
            "error": str(error or ""),
        })

    def run(self):
        try:
            if self._is_cancelled():
                self._finish("cancelled")
                return
            self.progress.emit(10, "正在读取项目缓存...")
            from cache import load_project_state, load_mask, load_mask_bundle
            from row_geometry import MASK_REGULARIZATION_VERSION
            saved = load_project_state(self.path, self.expected_source_sha256)
            if self._is_cancelled():
                self._finish("cancelled")
                return
            if not saved:
                self._finish("completed", {"path": self.path, "saved": {}})
                return
            from config import Config
            from inference import INFERENCE_PIPELINE_VERSION
            from path_planner import PATH_PLANNING_VERSION
            from provenance import ProvenanceError, file_sha256, verify_stage_record
            cfg = Config()
            self.progress.emit(35, "正在读取 AI 掩膜缓存...")
            raw_mask, raw_ox, raw_oy = load_mask(self.path, "") if saved.get('inference_done') else (None, 0, 0)
            reasons = []
            inference_valid = False
            inference_record = saved.get("inference_provenance") or {}
            try:
                if raw_mask is None:
                    raise ProvenanceError("inference artifact is missing")
                current_inputs = dict(inference_record.get("inputs") or {})
                current_inputs["source_sha256"] = self.expected_source_sha256 or saved.get("source_sha256", "")
                model_path = str(saved.get("current_model_path", "") or "")
                if not model_path or not os.path.isfile(model_path):
                    raise ProvenanceError("inference model file is missing")
                current_inputs["model_sha256"] = file_sha256(model_path)
                current_inputs["inference_config"] = {
                    "capture_size": int(cfg.TILE_CAPTURE_SIZE),
                    "overlap": float(cfg.TILE_OVERLAP),
                    "conf": float(cfg.MODEL_CONF),
                    "iou": float(cfg.MODEL_IOU),
                    "batch_size": 4,
                }
                current_inputs["preprocessing_config"] = cfg.section("raster_preprocessing")
                if str(inference_record.get("algorithm_version", "")) != INFERENCE_PIPELINE_VERSION:
                    raise ProvenanceError("inference algorithm version changed")
                verify_stage_record(inference_record, "inference", current_inputs, raw_mask)
                inference_valid = True
            except Exception as exc:
                reasons.append(f"AI 掩膜缓存失效: {exc}")
                saved["inference_done"] = False
                saved["mask_processed"] = False
                saved["auto_path_planned"] = False
                raw_mask = None

            self.progress.emit(65, "正在读取掩膜处理缓存...")
            processed_mask, proc_ox, proc_oy, processed_layers = (
                load_mask_bundle(self.path, "_processed")
                if inference_valid and saved.get("mask_processed")
                else (None, 0, 0, {})
            )
            processed_cache_valid = False
            mask_record = saved.get("mask_provenance") or {}
            try:
                if not inference_valid or processed_mask is None:
                    raise ProvenanceError("mask artifact is missing or its inference input is invalid")
                mask_inputs = dict(mask_record.get("inputs") or {})
                mask_inputs["inference_fingerprint"] = str(inference_record.get("fingerprint", ""))
                mask_inputs["raw_mask_sha256"] = str(inference_record.get("artifact_sha256", ""))
                mask_inputs["mask_config"] = cfg.section("mask_processing")
                if str(mask_record.get("algorithm_version", "")) != str(MASK_REGULARIZATION_VERSION):
                    raise ProvenanceError("mask algorithm version changed")
                verify_stage_record(mask_record, "mask", mask_inputs, processed_mask)
                processed_cache_valid = True
            except Exception as exc:
                if saved.get("mask_processed"):
                    reasons.append(f"掩膜处理缓存失效: {exc}")
                saved["mask_processed"] = False
                saved["auto_path_planned"] = False
                processed_mask = None
                processed_layers = {}

            path_cache_valid = False
            path_record = saved.get("path_provenance") or {}
            try:
                if not processed_cache_valid or not saved.get("auto_path_planned"):
                    raise ProvenanceError("path input mask is invalid or path is absent")
                path_inputs = dict(path_record.get("inputs") or {})
                path_inputs["mask_fingerprint"] = str(mask_record.get("fingerprint", ""))
                current_path_config = cfg.section("path_planning")
                harvester = dict(saved.get("harvester_params") or {})
                current_path_config.update({
                    "min_turn_radius_m": float(harvester.get("turn_radius_m", cfg.MIN_TURN_RADIUS_M)),
                    "turn_strategy": str(saved.get("turn_strategy", "auto")),
                    "forbidden_mask_confirmed": bool(
                        saved.get("forbidden_regions_confirmed", False)
                    ),
                })
                restored_mask_result = dict(processed_layers)
                restored_mask_result["processed_mask"] = processed_mask
                restored_state = SimpleNamespace(
                    forbidden_regions=list(saved.get("forbidden_regions") or []),
                    forbidden_regions_confirmed=bool(
                        saved.get("forbidden_regions_confirmed", False)
                    ),
                )
                restored_config = planning_config_with_mask_evidence(
                    current_path_config,
                    restored_mask_result,
                    restored_state,
                    processed_mask.shape,
                )
                path_inputs.update({
                    "path_config": current_path_config,
                    "harvester": harvester,
                    "turn_strategy": str(saved.get("turn_strategy", "auto")),
                    "entry_point": saved.get("entry_point"),
                    "exit_point": saved.get("exit_point"),
                    "unload_points": list(saved.get("unload_points") or []),
                    "forbidden_regions": list(saved.get("forbidden_regions") or []),
                    "forbidden_regions_confirmed": bool(
                        saved.get("forbidden_regions_confirmed", False)
                    ),
                    "semantic_mask_fingerprints": semantic_mask_fingerprints(
                        restored_config
                    ),
                })
                path_artifact = {
                    "full_path": [item.get("points", []) for item in saved.get("auto_path", [])],
                    "segment_types": [item.get("segment_type", "work") for item in saved.get("auto_path", [])],
                    "layout": dict(saved.get("path_artifact") or {}).get("layout", {}),
                }
                if str(path_record.get("algorithm_version", "")) != PATH_PLANNING_VERSION:
                    raise ProvenanceError("path algorithm version changed")
                verify_stage_record(path_record, "path", path_inputs, path_artifact)
                path_cache_valid = True
            except Exception as exc:
                if saved.get("auto_path_planned"):
                    reasons.append(f"路径缓存失效: {exc}")
                saved["auto_path_planned"] = False
                saved["auto_path_valid"] = False
                saved["simulation_done"] = False
                saved["export_done"] = False
            if self._is_cancelled():
                self._finish("cancelled")
                return
            self._finish("completed", {
                "path": self.path,
                "saved": saved,
                "processed_cache_valid": processed_cache_valid,
                "path_cache_valid": path_cache_valid,
                "cache_validation_messages": reasons,
                "raw_mask": raw_mask,
                "raw_ox": raw_ox,
                "raw_oy": raw_oy,
                "processed_mask": processed_mask,
                "processed_layers": processed_layers,
                "proc_ox": proc_ox,
                "proc_oy": proc_oy,
            })
        except Exception as e:
            self._finish("failed", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class PipelineWorker(QObject):
    """完整管线的后台工作线程。

    工作流程:
      1. 模型推理 (TiledInference)
      2. 掩膜处理 (process_mask)
      3. 行带分析 (classify_bands + extract_centerlines)
      4. 路径规划 (plan_path)
    """
    progress = Signal(int, str)   # (百分比, 消息)
    step_result = Signal(str, object)  # (步骤名, 数据)
    finished = Signal(dict)       # 全部完成
    error = Signal(str)           # 异常

    def __init__(self, state, geo, model_engine, image: np.ndarray,
                 params: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self.geo = geo
        self.model_engine = model_engine
        self.image = image
        self.params = dict(params or {})
        self.path_config = path_planning_config_snapshot(
            self.params.get("path_config") or {}
        )
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _emit_plan_progress(self, progress, message):
        try:
            value = float(progress)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 1.0:
            value *= 100.0
        value = max(0.0, min(100.0, value))
        self.progress.emit(min(99, 80 + int(value * 0.15)), message)

    def run(self):
        try:
            result = {}

            # ── Step 1: 推理 ──
            self.progress.emit(5, "正在执行 Tiled 推理...")
            from inference import TiledInference
            inference = TiledInference(self.model_engine)
            raw_mask = inference.run(
                self.image,
                capture_sz=self.params.get("slice_size", 1280),
                overlap=self.params.get("overlap", 0.2),
            )
            if self._cancelled:
                return
            if raw_mask is None:
                raise RuntimeError("AI 推理没有返回有效掩膜")
            self.progress.emit(30, "推理完成，处理掩膜...")
            result["mask"] = raw_mask
            # 通知 UI：AI 识别完成
            self.step_result.emit("segment_done", raw_mask)

            # ── Step 2: 掩膜处理 ──
            self.progress.emit(40, "掩膜处理中...")
            from mask_processor import process_mask, extract_centerlines
            from config import Config
            mask_config = Config().section("mask_processing")
            require_metric_scale(self.geo, self.state, raw_mask.shape)
            processed = process_mask(
                raw_mask,
                geo=self.geo,
                state=self.state,
                config=mask_config,
            )
            if processed is None:
                raise RuntimeError("掩膜处理失败：未能提取有效行带")

            # process_mask 返回 dict:
            #   processed_mask, main_angle, wide_bands, narrow_bands, all_bands
            processed_mask = processed.get("processed_mask", raw_mask)
            main_angle = processed.get("main_angle", 0.0)
            wide_bands = processed.get("wide_bands", [])
            result["processed"] = processed
            result["main_angle"] = main_angle

            # Notify UI with the complete processing result; it needs bands and metadata too.
            self.step_result.emit("process_done", processed)
            self.progress.emit(60, "提取中心线...")

            # ── Step 3: 提取中心线 ──
            centerlines = extract_centerlines(
                wide_bands, main_angle,
                full_mask=processed_mask,
                geo=self.geo,
                state=self.state,
            )
            result["centerlines"] = centerlines
            result["bands"] = wide_bands
            self.progress.emit(70, "中心线提取完成")

            # ── Step 4: 路径规划 ──
            self.progress.emit(80, "路径规划...")
            from path_planner import build_band_mask, plan_path
            planning_bands = wide_bands  # 只对4行带规划

            planning_mask = build_band_mask(processed_mask.shape, planning_bands)
            if not np.any(planning_mask):
                planning_mask = processed_mask

            path_config = dict(self.path_config)
            path_config.setdefault(
                "min_turn_radius_m", self.params.get("turn_radius", 2.5)
            )
            path_config.setdefault(
                "turn_strategy", self.params.get("turn_strategy", "auto")
            )
            path_config = planning_config_with_mask_evidence(
                path_config, processed, self.state, processed_mask.shape
            )
            path_result = plan_path(
                planning_bands, processed_mask, main_angle,
                geo=self.geo,
                state=self.state,
                config=path_config,
                progress_callback=self._emit_plan_progress,
                detected_mask=raw_mask,
                planned_mask=planning_mask,
                all_bands=processed.get("all_bands", []),
            )
            attach_path_provenance(
                path_result, processed, self.state, path_config
            )
            result["path"] = path_result

            self.progress.emit(100, "处理完成")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _extract_mask(self, results, img_shape) -> np.ndarray:
        """从 ultralytics 结果提取二值掩膜。"""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if results is None:
            return mask
        import cv2
        for r in results if hasattr(results, '__iter__') else [results]:
            if hasattr(r, 'masks') and r.masks is not None:
                for i in range(len(r.masks)):
                    m = r.masks[i].data.cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    m = np.where(m > 0.5, 1, 0).astype(np.uint8)
                    m = cv2.resize(m, (w, h), interpolation=0)
                    mask = np.maximum(mask, m)
        return mask

class MaskProcessWorker(QObject):
    """Run mask processing outside the UI thread."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, raw_mask: np.ndarray, geo, state, config: dict, token=None, parent=None):
        super().__init__(parent)
        self.raw_mask = raw_mask
        self.geo = geo
        self.state = state
        self.config = dict(config or {})
        self.token = token
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _is_cancelled(self):
        return self._cancelled or bool(getattr(self.token, "invalidated", False))

    def _finish(self, status, result=None, error=""):
        self.finished.emit({
            "task_token": self.token,
            "status": status,
            "result": result,
            "error": str(error or ""),
        })

    def _emit_mask_progress(self, progress, message):
        """Forward mask-processing progress and yield to the GUI event loop.

        For 15000×21000 GeoTIFF masks, OpenCV kernels plus Python row loops can
        keep CPU cores saturated long enough that Windows labels the main window
        as not responding.  Regular queued progress signals and sleep(0) keep
        the UI message pump getting time slices while the worker continues.
        """
        try:
            value = float(progress)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 1.0:
            value *= 100.0
        value = max(0.0, min(100.0, value))
        pct = min(94, 35 + int(value * 0.58))
        self.progress.emit(pct, str(message or "掩膜处理中..."))
        time.sleep(0)

    def run(self):
        previous_cv_threads = None
        started_at = time.perf_counter()
        try:
            if self._is_cancelled():
                self._finish("cancelled")
                return
            try:
                import cv2
                previous_cv_threads = cv2.getNumThreads()
                cv2.setNumThreads(max(1, min(2, (os.cpu_count() or 2) - 1)))
            except Exception:
                previous_cv_threads = None
            self.progress.emit(15, "正在读取 AI 掩膜...")
            from mask_processor import process_mask
            self.progress.emit(35, "正在提取水稻行带...")
            config = dict(self.config)
            config["_cancel_callback"] = self._is_cancelled
            require_metric_scale(self.geo, self.state, self.raw_mask.shape)
            processed = process_mask(
                self.raw_mask,
                geo=self.geo,
                state=self.state,
                config=config,
                progress_callback=self._emit_mask_progress,
            )
            if self._is_cancelled():
                self._finish("cancelled")
                return
            if processed is None:
                raise RuntimeError("掩膜处理没有生成有效结果")
            processed_mask = processed.get("processed_mask")
            if not isinstance(processed_mask, np.ndarray):
                raise RuntimeError("掩膜处理结果缺少有效掩膜")
            from provenance import make_stage_record
            from row_geometry import MASK_REGULARIZATION_VERSION
            provenance_inputs = {
                "inference_fingerprint": str(
                    (getattr(self.state, "inference_provenance", {}) or {}).get("fingerprint", "")
                ),
                "raw_mask_sha256": str(
                    (getattr(self.state, "inference_provenance", {}) or {}).get("artifact_sha256", "")
                ),
                "mask_config": {
                    key: value for key, value in self.config.items()
                    if not str(key).startswith("_")
                },
            }
            processed["provenance"] = make_stage_record(
                "mask",
                MASK_REGULARIZATION_VERSION,
                provenance_inputs,
                processed_mask,
            )
            processed["runtime"] = {
                "stage": "mask_processing",
                "seconds": float(time.perf_counter() - started_at),
                "input_shape": [int(value) for value in self.raw_mask.shape[:2]],
                "input_nonzero_pixels": int(np.count_nonzero(self.raw_mask)),
            }
            self.progress.emit(95, "掩膜处理完成，正在刷新界面...")
            if self._is_cancelled():
                self._finish("cancelled")
                return
            self._finish("completed", processed)
        except Exception as e:
            self._finish("failed", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            if previous_cv_threads is not None:
                try:
                    import cv2
                    cv2.setNumThreads(previous_cv_threads)
                except Exception:
                    pass


class PlanWorker(QObject):
    """Run path planning outside the UI thread."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, mask_result: dict, geo, state, config: dict, detected_mask=None, token=None, parent=None):
        super().__init__(parent)
        self.mask_result = dict(mask_result or {})
        self.geo = geo
        self.state = state
        # DECISION-013: the executed planner configuration must match the configuration
        # recorded in path provenance. Per-run values override persisted ones.
        self.config = path_planning_config_snapshot(config)
        self.detected_mask = detected_mask
        self.token = token
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _is_cancelled(self):
        return self._cancelled or bool(getattr(self.token, "invalidated", False))

    def _finish(self, status, result=None, error=""):
        self.finished.emit({
            "task_token": self.token,
            "status": status,
            "result": result,
            "error": str(error or ""),
        })

    def _emit_plan_progress(self, progress, message):
        try:
            value = float(progress)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 1.0:
            value *= 100.0
        value = max(0.0, min(100.0, value))
        self.progress.emit(min(99, 20 + int(value * 0.75)), str(message or "路径规划中..."))

    def run(self):
        started_at = time.perf_counter()
        try:
            if self._is_cancelled():
                self._finish("cancelled")
                return
            processed_mask = self.mask_result.get("processed_mask")
            if processed_mask is None:
                raise RuntimeError("缺少处理后的掩膜")
            require_metric_scale(self.geo, self.state, processed_mask.shape)
            main_angle = self.mask_result.get("main_angle", 0.0)
            wide_bands = self.mask_result.get("wide_bands", [])
            self.progress.emit(20, "正在构建可作业区域...")
            from path_planner import build_band_mask, plan_path
            planning_mask = build_band_mask(processed_mask.shape, wide_bands)
            if not np.any(planning_mask):
                planning_mask = processed_mask
            if self._is_cancelled():
                self._finish("cancelled")
                return
            self.progress.emit(35, "正在计算作业线与转弯段...")
            config = planning_config_with_mask_evidence(
                self.config, self.mask_result, self.state, processed_mask.shape
            )
            config["_cancel_callback"] = self._is_cancelled
            path_result = plan_path(
                wide_bands, processed_mask, main_angle,
                geo=self.geo,
                state=self.state,
                config=config,
                progress_callback=self._emit_plan_progress,
                detected_mask=self.detected_mask,
                planned_mask=planning_mask,
                all_bands=self.mask_result.get("all_bands", []),
            )
            if self._is_cancelled():
                self._finish("cancelled")
                return
            attach_path_provenance(
                path_result, self.mask_result, self.state, config
            )
            path_result["runtime"] = {
                "stage": "path_planning",
                "seconds": float(time.perf_counter() - started_at),
                "mask_shape": [int(value) for value in processed_mask.shape[:2]],
                "band_count": len(wide_bands),
            }
            self.progress.emit(98, "路径规划完成，正在刷新界面...")
            if self._is_cancelled():
                self._finish("cancelled")
                return
            self._finish("completed", {
                "path": path_result,
                "processed": self.mask_result,
                "bands": wide_bands,
            })
        except Exception as e:
            self._finish("failed", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
