"""Configuration management and logging for the PySide6 application."""
import os, json, logging, datetime, tempfile, glob as _glob, threading, math, shutil
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional
from runtime_paths import bundled_resource, legacy_config_path, user_config_path, logs_dir


class ConfigValidationError(ValueError):
    """Configuration is syntactically valid JSON but scientifically unsafe."""


DEFAULT_HARVESTER_PARAMS = {
    # 幅宽 2.0 m 为实机实测值。原默认 1.80 m 与实机不符：1.80 m 是按
    # 「轨距 + 履带宽 + 两侧余量」反推的名义匹配值（履带 0.45 时得 1.90 m），
    # 属推算值，不能作为实机参数。
    "cutter_width_m": 2.00,
    "cutter_forward_offset_m": 1.35,
    # 履带宽度实测值 0.35 m。原默认 0.45 m 与实机不符：按通道宽 0.45 m、
    # 轨距 1.30 m 核算，0.45 m 时履带外缘超出通道外缘 3.7 cm（碾压宽行带），
    # 0.35 m 时外侧余量 +1.3 cm 方可通过「履带完整落入通道」判据。
    "track_width_m": 0.35,
    "track_gauge_m": 1.30,
    "wheelbase_m": 1.80,
    "track_length_m": 1.70,
    "turn_radius_m": 0.90,
    "work_speed_mps": 1.20,
    "turn_speed_mps": 0.60,
    "entry_speed_mps": 0.80,
}

# Parameter provenance is deliberately orthogonal to the numerical defaults.
# A value copied from config or a heuristic recommendation remains unverified
# until an operator explicitly records a measured source.
HARVESTER_PARAM_KEYS = tuple(DEFAULT_HARVESTER_PARAMS)
HARVESTER_PARAM_SOURCE_LABELS = {
    "default": "系统默认值",
    "recommended": "智能推荐值",
    "user_entered": "用户填写（未提供实测证据）",
    "user_measured": "用户实测记录",
    "legacy_unverified": "历史缓存（来源未记录）",
}
HARVESTER_FORMAL_MEASURED_SOURCES = frozenset({"user_measured"})

# The mask pipeline has one audited operating profile. Older configuration
# files may still contain the removed profile names; validation canonicalizes
# them to this value so stale settings cannot silently change the result.
MASK_PROCESSING_STRENGTH = "standard"
LEGACY_MASK_PROCESSING_STRENGTHS = frozenset(
    {"light", "standard", "strong", "very_strong"}
)

# Keep the active log filename and cross-run cleanup pattern coupled.
LOG_FILE_PREFIX = "app_log"


def _validated_number(section: dict, key: str, *, minimum=None, maximum=None, strict_min=False):
    if key not in section:
        return
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{key} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigValidationError(f"{key} must be finite")
    if minimum is not None and (value < minimum or (strict_min and value == minimum)):
        op = ">" if strict_min else ">="
        raise ConfigValidationError(f"{key} must be {op} {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigValidationError(f"{key} must be <= {maximum}")


def validate_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ConfigValidationError("configuration root must be an object")
    schema_version = str(raw.get("_schema_version", "1.1"))
    if schema_version not in {"1.1", "1.2", "1.3"}:
        raise ConfigValidationError(f"unsupported _schema_version: {schema_version}")
    for name in ("model", "harvester", "path_planning", "mask_processing", "raster_preprocessing", "geo"):
        if name in raw and not isinstance(raw[name], dict):
            raise ConfigValidationError(f"{name} must be an object")

    model = raw.get("model", {})
    _validated_number(model, "conf_threshold", minimum=0.0, maximum=1.0)
    _validated_number(model, "iou_threshold", minimum=0.0, maximum=1.0)
    _validated_number(model, "tile_overlap", minimum=0.0, maximum=0.95)
    _validated_number(model, "tile_capture_size", minimum=64, maximum=8192)
    _validated_number(model, "inference_batch_size", minimum=1, maximum=32)
    _validated_number(model, "eval_batch_size", minimum=1, maximum=32)
    _validated_number(model, "max_source_crop_pixels", minimum=1)
    _validated_number(
        model, "orthomosaic_recovery_conf_threshold", minimum=0.0, maximum=1.0
    )
    _validated_number(
        model, "orthomosaic_primary_mask_threshold", minimum=0.0, maximum=1.0
    )
    _validated_number(
        model, "orthomosaic_tile_edge_weight_floor", minimum=0.0, maximum=1.0
    )
    _validated_number(
        model, "orthomosaic_context_capture_multiplier", minimum=1, maximum=4
    )
    _validated_number(
        model, "orthomosaic_context_overlap", minimum=0.0, maximum=0.90
    )
    _validated_number(
        model, "orthomosaic_context_support_ratio", minimum=0.0, maximum=1.0
    )
    _validated_number(
        model, "orthomosaic_context_recovery_distance_m", minimum=0.0, maximum=5.0
    )
    _validated_number(
        model, "orthomosaic_context_separator_ratio", minimum=0.0, maximum=1.0
    )
    _validated_number(
        model, "orthomosaic_context_separator_min_votes", minimum=1, maximum=16
    )
    _validated_number(
        model, "orthomosaic_context_separator_min_length_m", minimum=0.0, maximum=20.0
    )
    if "inference_device" in model:
        if str(model["inference_device"]).lower() not in {
            "auto", "cpu", "cuda", "cuda:0", "0"
        }:
            raise ConfigValidationError(
                "inference_device must be auto, cpu, or cuda"
            )
    if "inference_precision" in model:
        if str(model["inference_precision"]).lower() not in {
            "auto", "fp32", "fp16", "half"
        }:
            raise ConfigValidationError(
                "inference_precision must be auto, fp32, or fp16"
            )

    harvester = raw.get("harvester", {})
    for key in (
        "cutter_width_m", "track_width_m", "track_gauge_m", "wheelbase_m",
        "track_length_m", "turn_radius_m", "work_speed_mps",
        "turn_speed_mps", "entry_speed_mps",
    ):
        _validated_number(harvester, key, minimum=0.0, strict_min=True)
    if harvester.get("cutter_forward_offset_m") is not None:
        _validated_number(
            harvester, "cutter_forward_offset_m", minimum=0.0, strict_min=True
        )
        cutter_forward_offset = float(harvester["cutter_forward_offset_m"])
        track_length = float(
            harvester.get(
                "track_length_m", DEFAULT_HARVESTER_PARAMS["track_length_m"]
            )
        )
        wheelbase = float(
            harvester.get("wheelbase_m", DEFAULT_HARVESTER_PARAMS["wheelbase_m"])
        )
        max_forward_offset = max(3.0, track_length + wheelbase)
        if cutter_forward_offset > max_forward_offset:
            raise ConfigValidationError(
                "cutter_forward_offset_m is implausibly far from "
                f"track_contact_center; must be <= {max_forward_offset:.2f} "
                "for the configured track_length_m and wheelbase_m"
            )
    if "track_width_m" in harvester and "track_gauge_m" in harvester:
        if float(harvester["track_width_m"]) >= float(harvester["track_gauge_m"]):
            raise ConfigValidationError("track_width_m must be smaller than track_gauge_m")

    planning = raw.get("path_planning", {})
    if "work_line_mode" in planning and str(planning["work_line_mode"]) not in {
        "band_centerline", "footprint_optimized",
        "band_centerline_with_footprint_validation",
    }:
        raise ConfigValidationError(
            "work_line_mode must be band_centerline, footprint_optimized, or "
            "band_centerline_with_footprint_validation"
        )
    if "validation_profile" in planning and str(planning["validation_profile"]) not in {
        "research", "field_trial", "machine_candidate",
    }:
        raise ConfigValidationError(
            "validation_profile must be research, field_trial, or machine_candidate"
        )
    if "track_core_overlap_policy" in planning and str(
        planning["track_core_overlap_policy"]
    ) not in {"report_only", "hard_gate"}:
        raise ConfigValidationError(
            "track_core_overlap_policy must be report_only or hard_gate"
        )
    if "footprint_metric_policy" in planning and str(
        planning["footprint_metric_policy"]
    ) not in {"report_only", "hard_gate"}:
        raise ConfigValidationError(
            "footprint_metric_policy must be report_only or hard_gate"
        )
    if "service_point_mode" in planning and str(
        planning["service_point_mode"]
    ) not in {"reference_only", "connect"}:
        raise ConfigValidationError(
            "service_point_mode must be reference_only or connect"
        )
    _validated_number(planning, "min_turn_radius_m", minimum=0.0, strict_min=True)
    _validated_number(planning, "planning_max_dim", minimum=256)
    _validated_number(planning, "validation_max_dim", minimum=256)
    _validated_number(planning, "endpoint_short_line_ratio", minimum=0.5, maximum=1.0)
    _validated_number(planning, "endpoint_shortfall_min_m", minimum=0.0)
    _validated_number(planning, "endpoint_extension_max_m", minimum=0.0)
    _validated_number(planning, "work_line_endpoint_extension_m", minimum=0.0)
    _validated_number(planning, "centerline_simplify_max_deviation_m", minimum=0.0)
    _validated_number(
        planning, "centerline_simplify_max_heading_change_deg",
        minimum=0.0, maximum=180.0,
    )
    _validated_number(
        planning, "centerline_simplify_max_points",
        minimum=2.0, maximum=64.0,
    )
    _validated_number(
        planning, "centerline_simplify_hard_max_points",
        minimum=2.0, maximum=256.0,
    )
    _validated_number(planning, "centerline_endpoint_repair_completion_ratio",
                      minimum=0.5, maximum=1.0)
    _validated_number(planning, "centerline_endpoint_repair_min_cohort",
                      minimum=2.0)
    _validated_number(planning, "centerline_endpoint_repair_max_m", minimum=0.0)
    _validated_number(
        planning, "uniform_zone_min_line_length_ratio",
        minimum=0.5, maximum=1.0,
    )
    _validated_number(planning, "uniform_zone_min_width_m", minimum=0.1)
    _validated_number(
        planning, "planting_region_ambiguity_margin_m", minimum=0.0,
    )
    _validated_number(
        planning, "uniform_internal_gap_bridge_m", minimum=0.0,
    )
    if planning.get("uniform_zone_line_spacing_m") is not None:
        _validated_number(planning, "uniform_zone_line_spacing_m", minimum=0.0, strict_min=True)
    if "uniform_zone_spacing_basis" in planning and str(
        planning["uniform_zone_spacing_basis"]
    ) not in {"track_gauge_overlap", "explicit"}:
        raise ConfigValidationError(
            "uniform_zone_spacing_basis must be track_gauge_overlap or explicit"
        )
    _validated_number(planning, "endpoint_outlier_mad_scale", minimum=0.0)
    _validated_number(planning, "footprint_pose_step_m", minimum=0.01, maximum=0.20)
    _validated_number(planning, "pose_heading_window_m", minimum=0.01, maximum=2.0)
    _validated_number(planning, "lateral_search_m", minimum=0.0)
    _validated_number(planning, "lateral_step_m", minimum=0.0, strict_min=True)
    _validated_number(planning, "lateral_smooth_penalty", minimum=0.0)
    _validated_number(planning, "lateral_uncertain_penalty", minimum=0.0)
    _validated_number(planning, "station_interval_m", minimum=0.0, strict_min=True)
    _validated_number(planning, "station_window_m", minimum=0.0, strict_min=True)
    _validated_number(planning, "path_smooth_m", minimum=0.0)
    for key in (
        "max_track_core_overlap_pct", "min_geometric_coverage_pct",
        "max_track_outside_field_pct", "max_track_outside_support_pct",
        "max_track_uncertain_overlap_pct", "max_track_forbidden_overlap_pct",
    ):
        _validated_number(planning, key, minimum=0.0, maximum=100.0)
    _validated_number(
        planning, "headland_damage_weight",
        minimum=0.0, maximum=1.0, strict_min=True,
    )
    _validated_number(
        planning, "missing_work_band_gap_multiplier",
        minimum=1.0,
    )
    _validated_number(
        planning, "missing_work_band_integer_tolerance",
        minimum=0.0, maximum=1.0, strict_min=True,
    )
    _validated_number(
        planning, "missing_work_band_min_support_ratio",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        planning, "missing_work_band_min_axial_ratio",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        planning, "missing_work_band_max_missing_count",
        minimum=1.0,
    )
    _validated_number(
        planning, "missing_work_band_phase_neighbor_tolerance",
        minimum=0.0,
    )
    _validated_number(
        planning, "missing_work_band_cross_section_half_width_ratio",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        planning, "missing_work_band_min_cross_section_occupancy",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        planning, "service_headland_axial_tolerance_m",
        minimum=0.0,
    )
    _validated_number(
        planning, "inter_region_tangent_alignment_min_cos",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        planning, "turn_radius_geometry_tolerance_m",
        minimum=0.0,
    )
    _validated_number(
        planning, "service_point_terminal_depth_m",
        minimum=0.1, maximum=20.0,
    )
    if "recover_missing_work_bands" in planning and not isinstance(
        planning["recover_missing_work_bands"], bool
    ):
        raise ConfigValidationError(
            "recover_missing_work_bands must be boolean"
        )

    mask = raw.get("mask_processing", {})
    if "strength" in mask:
        strength = str(mask["strength"]).strip().lower()
        if strength not in LEGACY_MASK_PROCESSING_STRENGTHS:
            raise ConfigValidationError(
                "strength must be the supported standard mask profile"
            )
        # Keep old config files readable while making the single active
        # profile explicit in the canonical in-memory configuration.
        mask["strength"] = MASK_PROCESSING_STRENGTH
    if "headland_corridor_enabled" in mask and not isinstance(
        mask["headland_corridor_enabled"], bool
    ):
        raise ConfigValidationError("headland_corridor_enabled must be boolean")
    if "headland_corridor_fill_envelope" in mask and not isinstance(
        mask["headland_corridor_fill_envelope"], bool
    ):
        raise ConfigValidationError("headland_corridor_fill_envelope must be boolean")
    if "headland_corridor_follow_raw_terminal" in mask and not isinstance(
        mask["headland_corridor_follow_raw_terminal"], bool
    ):
        raise ConfigValidationError(
            "headland_corridor_follow_raw_terminal must be boolean"
        )
    if "headland_terminal_separator_enabled" in mask and not isinstance(
        mask["headland_terminal_separator_enabled"], bool
    ):
        raise ConfigValidationError(
            "headland_terminal_separator_enabled must be boolean"
        )
    if "headland_terminal_dense_seed_enabled" in mask and not isinstance(
        mask["headland_terminal_dense_seed_enabled"], bool
    ):
        raise ConfigValidationError(
            "headland_terminal_dense_seed_enabled must be boolean"
        )
    _validated_number(mask, "max_work_dim", minimum=256)
    _validated_number(mask, "band_width_threshold_m", minimum=0.0, strict_min=True)
    for key in (
        "band_internal_gap_close_m", "band_end_gap_close_m",
        "band_endpoint_gap_close_m", "band_end_guard_m", "band_end_trim_m",
        "band_internal_gap_bridge_m", "band_terminal_fragment_min_m",
        "residual_body_attach_m", "residual_body_max_angle_deg",
        "residual_body_min_elongation", "headland_corridor_close_m",
        "headland_corridor_depth_smooth_m",
        "headland_corridor_terminal_depth_m", "headland_corridor_max_body_cut_m",
        "headland_corridor_endpoint_margin_m",
        "headland_terminal_follow_max_depth_m",
        "headland_terminal_wall_smooth_m",
        "headland_terminal_wall_outlier_tolerance_m",
        "headland_terminal_field_edge_max_abs_slope",
        "headland_terminal_wall_quantized_mad_px",
        "headland_terminal_dense_seed_depth_m",
        "headland_terminal_dense_seed_gap_m",
        "headland_terminal_dense_seed_min_depth_m",
        "headland_terminal_separator_smooth_m",
        "headland_terminal_separator_min_gap_m",
        "headland_terminal_separator_support_m",
        "headland_terminal_separator_endpoint_tolerance_m",
    ):
        _validated_number(mask, key, minimum=0.0)
    _validated_number(
        mask, "headland_terminal_field_edge_min_span_ratio",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(mask, "residual_body_min_attach_ratio", minimum=0.0, maximum=1.0)
    _validated_number(mask, "headland_corridor_min_span_ratio", minimum=0.0, maximum=1.0)
    _validated_number(
        mask, "headland_corridor_min_component_span_ratio",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(mask, "headland_corridor_min_seed_ratio", minimum=0.0, maximum=1.0)
    _validated_number(
        mask, "headland_terminal_dense_seed_min_occupancy",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        mask, "headland_terminal_dense_seed_fallback_occupancy",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        mask, "headland_terminal_wall_min_axial_span_ratio",
        minimum=0.25, maximum=1.0,
    )
    _validated_number(
        mask, "headland_terminal_region_min_support_ratio",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        mask, "headland_terminal_separator_occupancy_ratio",
        minimum=0.0, maximum=1.0,
    )
    _validated_number(
        mask, "headland_terminal_separator_max_occupancy",
        minimum=0.0, maximum=1.0,
    )

    raster = raw.get("raster_preprocessing", {})
    if "mode" in raster and str(raster["mode"]) not in {
        "percentile_per_band", "uint8_identity",
    }:
        raise ConfigValidationError("raster_preprocessing.mode is unsupported")
    _validated_number(raster, "lower_percentile", minimum=0.0, maximum=100.0)
    _validated_number(raster, "upper_percentile", minimum=0.0, maximum=100.0)
    lower = float(raster.get("lower_percentile", 2.0))
    upper = float(raster.get("upper_percentile", 98.0))
    if lower >= upper:
        raise ConfigValidationError("lower_percentile must be smaller than upper_percentile")

    geo = raw.get("geo", {})
    _validated_number(geo, "max_pixel_anisotropy_ratio", minimum=1.0)
    _validated_number(geo, "max_gsd_variation_ratio", minimum=0.0, maximum=1.0)

    export = raw.get("export", {})
    if not isinstance(export, dict):
        raise ConfigValidationError("export must be an object")
    if "gate_mode" in export and str(export["gate_mode"]) not in {
        "fail_closed", "confirm",
    }:
        raise ConfigValidationError(
            "export.gate_mode must be fail_closed or confirm"
        )
    return raw

def save_json_atomic(path: str, data: dict):
    """原子写入 JSON 文件：先写临时文件再替换，防止写入中断导致文件损坏。
    写入前进行 JSON 序列化校验，确保数据可序列化。"""
    content = json.dumps(data, indent=2, ensure_ascii=False)
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        os.replace(tmp_path, path)
        try:
            from project_log import ProjectJournal
            ProjectJournal().change(
                "JSON configuration saved",
                module="config",
                details={"path": os.path.abspath(path)},
            )
        except Exception:
            pass
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


class Config:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
                cls._instance._raw = {}
                cls._instance._path = ""
                cls._instance._lock = threading.RLock()
        return cls._instance

    def load(self, path: str = "") -> bool:
        explicit_path = bool(str(path or "").strip())
        if explicit_path:
            target_path = os.path.abspath(str(path))
            example_path = os.path.join(os.path.dirname(target_path), "config.example.json")
        else:
            # DECISION-018: user settings are outside the install directory.
            target_path = os.path.abspath(
                str(os.environ.get("ALIGN_CONFIG_PATH", "") or user_config_path())
            )
            example_path = str(bundled_resource("config.example.json"))
            if not os.path.exists(target_path):
                legacy_path = legacy_config_path()
                source_path = legacy_path if legacy_path.is_file() else Path(example_path)
                if source_path.is_file() and os.path.abspath(str(source_path)) != target_path:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(str(source_path), target_path)
        load_path = target_path
        if not os.path.exists(load_path) and os.path.exists(example_path):
            load_path = example_path
        if os.path.exists(load_path):
            try:
                with open(load_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                validated = validate_config(raw)
                with self._lock:
                    self._raw = validated
                    self._loaded = True
                    self._path = os.path.abspath(target_path if not explicit_path else load_path)
                return True
            except ConfigValidationError:
                with self._lock:
                    self._raw = {}
                    self._loaded = False
                raise
            except Exception as e:
                print(f"[Config] 加载配置文件失败: {e}")
        with self._lock:
            self._raw = {}
            self._loaded = False
        return False

    def config_path(self) -> str:
        """Return the writable config file used by the current session."""
        with self._lock:
            return str(self._path or user_config_path())

    def _get(self, *keys: str, default: Any = None) -> Any:
        with self._lock:
            val = self._raw
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k)
                    if val is None:
                        return default
                else:
                    return default
            return val if val is not None else default

    def section(self, name: str) -> dict:
        with self._lock:
            value = self._raw.get(str(name), {})
            return deepcopy(value) if isinstance(value, dict) else {}

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy(self._raw)

    def replace(self, raw: dict) -> dict:
        candidate = validate_config(deepcopy(raw))
        with self._lock:
            self._raw = candidate
            self._loaded = True
            return deepcopy(self._raw)

    def update_section(self, name: str, updates: dict) -> dict:
        with self._lock:
            candidate = deepcopy(self._raw)
            section = candidate.setdefault(str(name), {})
            if not isinstance(section, dict):
                raise ConfigValidationError(f"{name} must be an object")
            section.update(deepcopy(updates or {}))
            validate_config(candidate)
            self._raw = candidate
            return deepcopy(self._raw)

    @contextmanager
    def temporary_section(self, name: str, updates: dict):
        with self._lock:
            original = deepcopy(self._raw)
            candidate = deepcopy(self._raw)
            candidate.setdefault(str(name), {}).update(deepcopy(updates or {}))
            validate_config(candidate)
            self._raw = candidate
            try:
                yield
            finally:
                self._raw = original

    @property
    def WIN_W(self) -> int: return self._get("window", "width", default=1500)
    @property
    def WIN_H(self) -> int: return self._get("window", "height", default=900)
    @property
    def UI_HEIGHT(self) -> int: return 96
    @property
    def PANEL_W(self) -> int: return 320
    @property
    def CANVAS_TOOLBAR_H(self) -> int: return 76
    @property
    def CANVAS_W(self) -> int: return self.WIN_W - self.PANEL_W
    @property
    def CANVAS_H(self) -> int:
        return self.WIN_H - self.UI_HEIGHT - self.CANVAS_TOOLBAR_H
    @property
    def INFO_BAR_H(self) -> int: return 120   # 左侧面板底部信息枢纽高度
    @property
    def TITLE(self) -> str: return self._get("window", "title", default="ALIGN 农机作业规划系统")

    @property
    def CV2_WIN_ID(self) -> str:
        """cv2 内部窗口标识（必须为 ASCII，用于 namedWindow/setMouseCallback/imshow 匹配）。
        中文标题会导致 Windows 上 OpenCV 回调注册失败，因此内部用英文 ID，
        创建窗口后通过 Win32 SetWindowTextW 修改标题栏显示为中文。"""
        return "HarvesterPlanner"

    @property
    def MODEL_CONF(self) -> float: return self._get("model", "conf_threshold", default=0.25)
    @property
    def MODEL_IOU(self) -> float: return self._get("model", "iou_threshold", default=0.45)
    @property
    def TILE_SIZE(self) -> int: return self._get("model", "tile_size", default=640)
    @property
    def TILE_OVERLAP(self) -> float: return self._get("model", "tile_overlap", default=0.4)
    @property
    def TILE_CAPTURE_SIZE(self) -> int:
        return self._get("model", "tile_capture_size", default=640)

    @property
    def INFERENCE_BATCH_SIZE(self) -> int:
        value = self._get(
            "model",
            "inference_batch_size",
            default=self._get("model", "eval_batch_size", default=8),
        )
        try:
            return max(1, min(32, int(value)))
        except (TypeError, ValueError):
            return 8

    @property
    def ERODE_KERNEL_SIZE(self) -> int:
        return self._get("model", "erode_kernel_size", default=7)

    @property
    def ERODE_ITERATIONS(self) -> int:
        return self._get("model", "erode_iterations", default=1)

    @property
    def MIN_BAND_AREA_PX(self) -> int: return self._get("path_planning", "min_band_area_px", default=100)
    @property
    def BAND_ASPECT_RATIO_MIN(self) -> float: return self._get("path_planning", "band_aspect_ratio_min", default=3.0)
    @property
    def MORPH_CLOSE_KERNEL(self) -> tuple:
        w = self._get("path_planning", "morph_close_kernel_w", default=25)
        h = self._get("path_planning", "morph_close_kernel_h", default=5)
        return (w, h)
    @property
    def PATH_POINT_INTERVAL_M(self) -> float: return self._get("path_planning", "path_point_interval_m", default=0.5)
    @property
    def HEADLAND_BUFFER_M(self) -> float: return self._get("path_planning", "headland_buffer_m", default=3.0)
    @property
    def MIN_CORRIDOR_WIDTH_M(self) -> float: return self._get("path_planning", "min_corridor_width_m", default=0.8)
    @property
    def MIN_TURN_RADIUS_M(self) -> float: return self._get("path_planning", "min_turn_radius_m", default=1.0)

    @property
    def HEADLAND_ANGLE_THRESH_DEG(self) -> float: return self._get("mask_processing", "headland_angle_thresh_deg", default=60.0)
    @property
    def HEADLAND_MIN_AREA_RATIO(self) -> float: return self._get("mask_processing", "headland_min_area_ratio", default=0.005)
    @property
    def AGGREGATE_CLOSE_KERNEL(self) -> int: return self._get("mask_processing", "aggregate_close_kernel", default=15)
    @property
    def ADHESION_THIN_KERNEL(self) -> int: return self._get("mask_processing", "adhesion_thin_kernel", default=12)
    @property
    def ADHESION_USE_DIST_TRANSFORM(self) -> bool: return self._get("mask_processing", "adhesion_use_distance_transform", default=True)
    @property
    def ADHESION_DIST_SPLIT_RATIO(self) -> float: return self._get("mask_processing", "adhesion_distance_split_ratio", default=0.3)
    @property
    def BAND_WIDTH_THRESHOLD_M(self) -> float: return self._get("mask_processing", "band_width_threshold_m", default=0.55)
    @property
    def BAND_AUTO_CALIBRATE(self) -> bool: return self._get("mask_processing", "band_auto_calibrate", default=True)
    @property
    def MASK_PROCESSING_STRENGTH(self) -> str: return MASK_PROCESSING_STRENGTH

    @property
    def CUTTER_WIDTH_M(self) -> float: return self._get("harvester", "cutter_width_m", default=DEFAULT_HARVESTER_PARAMS["cutter_width_m"])
    @property
    def CUTTER_FORWARD_OFFSET_M(self) -> Optional[float]:
        value = self._get(
            "harvester",
            "cutter_forward_offset_m",
            default=DEFAULT_HARVESTER_PARAMS["cutter_forward_offset_m"],
        )
        return (
            float(value)
            if value not in (None, "")
            else float(DEFAULT_HARVESTER_PARAMS["cutter_forward_offset_m"])
        )
    @property
    def TRACK_WIDTH_M(self) -> float: return self._get("harvester", "track_width_m", default=DEFAULT_HARVESTER_PARAMS["track_width_m"])
    @property
    def TRACK_GAUGE_M(self) -> float: return self._get("harvester", "track_gauge_m", default=DEFAULT_HARVESTER_PARAMS["track_gauge_m"])
    @property
    def WHEELBASE_M(self) -> float: return self._get("harvester", "wheelbase_m", default=DEFAULT_HARVESTER_PARAMS["wheelbase_m"])
    @property
    def TRACK_LENGTH_M(self) -> float: return self._get("harvester", "track_length_m", default=DEFAULT_HARVESTER_PARAMS["track_length_m"])
    @property
    def TURN_RADIUS_M(self) -> float: return self._get("harvester", "turn_radius_m", default=DEFAULT_HARVESTER_PARAMS["turn_radius_m"])

    @property
    def WORK_SPEED_MPS(self) -> float: return self._get("harvester", "work_speed_mps", default=DEFAULT_HARVESTER_PARAMS["work_speed_mps"])
    @property
    def TURN_SPEED_MPS(self) -> float: return self._get("harvester", "turn_speed_mps", default=DEFAULT_HARVESTER_PARAMS["turn_speed_mps"])
    @property
    def ENTRY_SPEED_MPS(self) -> float: return self._get("harvester", "entry_speed_mps", default=DEFAULT_HARVESTER_PARAMS["entry_speed_mps"])

    @property
    def DEFAULT_EPSG(self) -> str: return self._get("geo", "default_epsg", default="32650")
    @property
    def MASK_OPACITY(self) -> float: return self._get("ui", "mask_opacity", default=0.3)

    @property
    def ROUTE_ID(self) -> int: return self._get("export", "route_id", default=0)
    @property
    def OFFSET_X(self) -> str: return self._get("export", "offset_x", default="0.00")
    @property
    def OFFSET_Y(self) -> str: return self._get("export", "offset_y", default="0.00")
    @property
    def INFM_LINE(self) -> str: return self._get("export", "infoline", default="$INFM,0,0.7,0.7,0.7,0.7,0,5,0*")


class AppLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._status_callback = None
        return cls._instance

    def init(self, log_dir: str = ""):
        if self._initialized:
            return
        if not log_dir:
            log_dir = str(logs_dir())
        os.makedirs(log_dir, exist_ok=True)
        self._cleanup_old_logs(log_dir, max_keep=5)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{LOG_FILE_PREFIX}_{ts}.log")

        self._logger = logging.getLogger("HarvesterPro")
        self._logger.setLevel(logging.DEBUG)

        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024,
                                 backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self._logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._logger.addHandler(ch)

        self._status_callback = None
        self._initialized = True
        self.info(f"日志文件: {log_path}")

    @staticmethod
    def _cleanup_old_logs(log_dir: str, max_keep: int = 5):
        """清理旧日志文件，仅保留最近 max_keep 个。"""
        pattern = os.path.join(log_dir, f"{LOG_FILE_PREFIX}_*.log*")
        log_files = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for old_file in log_files[max_keep:]:
            try:
                os.remove(old_file)
            except OSError:
                pass

    def set_status_callback(self, cb):
        self._status_callback = cb

    def _log(self, level: int, msg: str, event_type: str = "operation"):
        try:
            if self._initialized:
                self._logger.log(level, msg)
            else:
                print(msg)
        except Exception:
            pass  # 日志写入失败不应影响主流程
        try:
            from project_log import ProjectJournal
            if level >= logging.ERROR:
                ProjectJournal().error(msg, module="application")
            else:
                ProjectJournal().write(
                    logging.getLevelName(level),
                    event_type,
                    msg,
                    module="application",
                )
        except Exception:
            pass
        if self._status_callback and level >= logging.WARNING:
            self._status_callback(msg)

    def debug(self, msg: str): self._log(logging.DEBUG, msg)
    def info(self, msg: str): self._log(logging.INFO, msg)
    def change(self, msg: str): self._log(logging.INFO, msg, event_type="change")
    def warning(self, msg: str): self._log(logging.WARNING, msg)
    def error(self, msg: str): self._log(logging.ERROR, msg)
