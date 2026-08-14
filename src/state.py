"""数据类 + 应用状态"""
import threading
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np


@dataclass
class PathPoint:
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    locked: bool = False


@dataclass
class BandInstance:
    id: int = 0
    mask: Any = None
    contour: Any = None
    centroid: Tuple[float, float] = (0, 0)
    angle: float = 0.0
    length: float = 0.0
    width: float = 0.0
    bbox: Tuple = (0, 0, 0, 0)


@dataclass
class Corridor:
    left_id: int = 0
    right_id: int = 0
    centerline: List[Tuple[float, float]] = field(default_factory=list)
    min_width_m: float = 0.0
    avg_width_m: float = 0.0
    passable: bool = True
    risk: bool = False
    risk_segments: List[int] = field(default_factory=list)


@dataclass
class AutoPathSegment:
    points: List[PathPoint] = field(default_factory=list)
    status: int = 1  # 0=转弯, 1=作业, 2=进场, 3=离场, 4=倒车转弯
    corridor_id: int = -1
    length_m: float = 0.0
    segment_type: str = "work"  # "work", "turn", "entry", "exit"


class AppState:
    def __init__(self):
        self.state_lock = threading.Lock()
        self.reset()

    def reset(self):
        self.tif_path: str = ""
        self.dataset = None
        self.transformer = None
        self.img_raw: Optional[np.ndarray] = None
        self.img_h: int = 0
        self.img_w: int = 0
        self.source_img_h: int = 0  # GeoTIFF 原始高度
        self.source_img_w: int = 0  # GeoTIFF 原始宽度
        self.display_h: int = 0       # 降采样后显示高度
        self.display_w: int = 0       # 降采样后显示宽度
        self.downsample_factor: int = 1  # 降采样倍率
        self.display_scale_x: float = 1.0  # 原图宽度 / 显示宽度
        self.display_scale_y: float = 1.0  # 原图高度 / 显示高度
        self.source_sha256: str = ""
        self.project_generation: int = 0
        self.source_metadata: Dict[str, Any] = {}
        self.raster_preprocessing: Dict[str, Any] = {}

        self.view_scale: float = 1.0
        self.view_x: float = 0.0
        self.view_y: float = 0.0

        self.path_points: List[Tuple[float, float]] = []
        self.path_status: List[int] = []
        self.undo_stack: List[Dict] = []
        self.redo_stack: List[Dict] = []

        self.current_mode: str = "FIELD"  # 默认进入圈选田块模式
        self.dragging_idx: int = -1
        self.drag_start_pt: Optional[Tuple] = None
        self.is_panning: bool = False
        self.pan_start_x: float = 0.0
        self.pan_start_y: float = 0.0
        self._was_drag: bool = False

        self.selected_type: Optional[str] = None
        self.selected_idx: int = -1

        self.list_scroll_y: int = 0
        self.panel_content_h: int = 0
        self.ui_list_items: List[Dict] = []
        self.panel_hitboxes: List[Dict] = []
        self.route_list_scroll_y: int = 0
        self.route_list_content_h: int = 0
        self.route_list_view_h: int = 0

        self.mask_raw: Optional[np.ndarray] = None
        self.inference_original_mask: Optional[np.ndarray] = None
        self.mask_overlay: Optional[np.ndarray] = None
        self.inference_done: bool = False
        self.inference_running: bool = False
        self.inference_progress: float = 0.0
        self.inference_provenance: Dict[str, Any] = {}
        self.inference_runtime: Dict[str, Any] = {}

        # 掩膜处理状态
        self.mask_processed: bool = False         # 掩膜处理是否完成
        self.mask_processing_running: bool = False
        self.mask_processing_progress: float = 0.0  # 0.0~1.0
        self.mask_result: Optional[Dict] = None   # process_mask() 返回的完整结果
        self.mask_provenance: Dict[str, Any] = {}

        # 路径规划进度
        self.plan_running: bool = False
        self.plan_progress: float = 0.0             # 0.0~1.0
        self.path_provenance: Dict[str, Any] = {}
        self.path_runtime: Dict[str, Any] = {}
        self.path_validation: Dict[str, Any] = {}
        self.path_artifact: Dict[str, Any] = {}

        self.smooth_running: bool = False
        self.smooth_progress: float = 0.0
        self.smooth_stage: str = ""
        self.smooth_log: List[str] = []

        self.band_list: List[BandInstance] = []
        self.corridors: List[Corridor] = []
        self.auto_path: List[AutoPathSegment] = []
        self.auto_path_planned: bool = False

        self.need_redraw: bool = True
        self.status_message: str = ""
        self._status_msg_time: float = 0.0  # 状态消息显示起始时间
        self.btn_hitboxes: List[Dict] = []
        self.canvas_tool_hitboxes: List[Dict] = []
        self.view_rotation_deg: float = 0.0
        self.rotation_dragging: bool = False
        self.rotation_drag_start_x: int = 0
        self.rotation_drag_start_deg: float = 0.0
        self.hover_btn_val: Optional[str] = None
        self.hover_btn_type: Optional[str] = None
        self._click_flash: float = 0.0

        self.field_boundary: List[Tuple[float, float]] = []
        self.field_area_m2: float = 0.0
        self.field_drawing_active: bool = False
        self.field_temp_point: Optional[Tuple[float, float]] = None
        self.field_drag_idx: int = -1
        self.field_edit_existing: bool = False
        self.field_original_boundary: List[Tuple[float, float]] = []
        self.forbidden_regions: List[List[Tuple[float, float]]] = []
        self.forbidden_regions_confirmed: bool = False

        self.harvester_params: Dict[str, float] = {}
        self.harvester_param_sources: Dict[str, str] = {}
        self.harvester_params_confirmed: bool = False

        # 进场/离场点
        self.entry_point: Optional[Tuple[float, float]] = None   # 像素坐标 (x, y)
        self.exit_point: Optional[Tuple[float, float]] = None
        self.unload_points: List[Tuple[float, float]] = []
        # 单个卸粮点字段保留读取能力，用于兼容历史项目快照。
        self.unload_point: Optional[Tuple[float, float]] = None
        self.entry_point_locked: bool = False   # 用户是否锁定自定义进场点
        self.exit_point_locked: bool = False
        self.unload_point_locked: bool = False
        self.entry_exit_mode: bool = False       # 是否处于进出点编辑模式
        self._drag_entry_exit = None  # entry / exit / ("unload", index)

        self.anim_active: bool = False
        self.anim_paused: bool = False
        self.anim_speed: float = 1.0
        self.anim_frame: int = 0
        self.anim_total_frames: int = 0
        self.anim_all_pts: List[Tuple[float, float]] = []
        self.anim_all_status: List[int] = []
        self.anim_all_types: List[str] = []
        self.anim_report: str = ""
        self.anim_speed_mps: float = 1.2   # 作业速度 (m/s)
        self.anim_frac: float = 0.0        # 亚帧插值余量 (米, 当前段内已走过的距离)
        self.anim_heading_rad: float = 0.0
        self.anim_curvature_1pm: float = 0.0
        self.anim_track_left_mps: float = 0.0
        self.anim_track_right_mps: float = 0.0
        self.anim_turn_radius_m: float = float("inf")

        # 悬浮控制面板状态
        self.ctrl_panel_x: int = -1         # 面板左上角 x（像素，-1=自动定位）
        self.ctrl_panel_y: int = -1         # 面板左上角 y
        self.ctrl_panel_dragging: bool = False
        self.ctrl_panel_drag_off: Tuple[int, int] = (0, 0)
        self.ctrl_panel_visible: bool = False  # 仅动画激活时可见

        self.workflow_step: int = 0
        self.simulation_done: bool = False
        self.export_done: bool = False
        self.workflow_steps: List[str] = [
            "圈选田块", "AI识别", "掩膜处理", "路径规划", "预演示", "导出"
        ]

        self.model_list: List[Dict] = []
        self.model_eval_scores: Dict = {}
        self.current_model_name: str = ""      # 当前加载的模型名
        self.current_model_path: str = ""      # 当前加载的模型完整路径
        self.model_sha256: str = ""
        # Cached identity is historical until it is verified against the
        # currently loaded ModelEngine. It is never an executable model.
        self.historical_model_name: str = ""
        self.historical_model_path: str = ""
        self.historical_model_sha256: str = ""
        self.cache_model_matches_active_engine: bool = True
        self.model_dropdown_open: bool = False  # 模型下拉菜单是否展开
        self.model_dropdown_rect = None         # 下拉菜单按钮的 hitbox（x1,y1,x2,y2）
        self.model_dropdown_items: List[Dict] = []  # 下拉菜单项 [{name, path, is_import}]
        self.model_dropdown_hover_idx: int = -1  # 当前悬停的下拉菜单项索引
        self.mask_offset_x: int = 0            # mask_raw 相对于全图的 x 偏移
        self.mask_offset_y: int = 0            # mask_raw 相对于全图的 y 偏移

        # 转弯策略下拉选择
        self.turn_strategy: str = "contour"     # 当前选择的转弯策略（默认套圈收割）
        # Legacy caches could contain a wheel-mutated strategy without any
        # explicit operator action. Only a popup/button selection sets this.
        self.turn_strategy_user_selected: bool = False
        self.turn_dropdown_open: bool = False
        self.turn_dropdown_rect = None
        self.turn_dropdown_items: List[Dict] = [
            {"key": "bow", "label": "弓形转弯 (Bow)"},
            {"key": "semicircle", "label": "半圆转弯 (Semicircle)"},
            {"key": "pear", "label": "梨形转弯 (Pear)"},
            {"key": "fishtail", "label": "鱼尾折返 (Fishtail, 含倒车)"},
        ]
        self.turn_dropdown_hover_idx: int = -1
        self.active_cache_dir: str = ""
        self.smooth_original_mask: Optional[np.ndarray] = None  # 平滑前备份
        self.smoothed_result: Optional[np.ndarray] = None       # 平滑后的结果
        self._show_original: bool = False
        self._coord_cache: Dict = {}
        self._mask_overlay_dirty: bool = True

        # 路径规划结果字段
        self.auto_path_segments: List = []        # 路径段列表 [[(x,y),...], ...]
        self.auto_path_geo: List[Dict] = []       # 地理坐标路径 [{"lon","lat",...}, ...]
        self.auto_path_valid: bool = False        # 路径是否通过合理性验证
        self.auto_path_desc: str = ""             # 路径验证描述

        self.pending_action: Optional[str] = None
        self.pending_turn_fallback: Optional[Dict] = None
        self._temp_field_data: Optional[dict] = None

        self.last_rolling_rate: float = 0.0
        self.last_crop_mask_geometric_coverage_pct: float = 0.0
        self.last_planned_target_geometric_coverage_pct: float = 0.0
        self.last_total_path_m: float = 0.0
        self.last_work_path_m: float = 0.0
        self.last_turn_path_m: float = 0.0
        self.last_entry_exit_path_m: float = 0.0
        self.last_detected_mask_geometric_coverage_pct: float = 0.0
        self.last_field_efficiency: float = 0.0

    def _copy_array(self, arr):
        return None if arr is None else arr.copy()

    @staticmethod
    def _snapshot_clone(value):
        """Clone mutable state while sharing immutable heavyweight ndarrays."""
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, dict):
            return {
                key: AppState._snapshot_clone(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [AppState._snapshot_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(AppState._snapshot_clone(item) for item in value)
        return deepcopy(value)

    def capture_system_snapshot(self, label: str = "系统操作") -> Dict[str, Any]:
        """Capture one logical workflow transaction for global undo/redo."""
        fields = (
            "field_boundary", "field_area_m2", "field_original_boundary",
            "forbidden_regions", "forbidden_regions_confirmed",
            "current_model_name", "current_model_path", "model_sha256",
            "historical_model_name", "historical_model_path", "historical_model_sha256",
            "cache_model_matches_active_engine",
            "mask_raw", "inference_original_mask", "mask_offset_x", "mask_offset_y",
            "inference_done", "mask_processed", "mask_result",
            "smooth_original_mask", "smoothed_result", "_show_original",
            "band_list", "corridors", "harvester_params", "harvester_param_sources",
            "turn_strategy", "turn_strategy_user_selected",
            "harvester_params_confirmed",
            "entry_point", "exit_point", "unload_points", "unload_point",
            "entry_point_locked", "exit_point_locked", "unload_point_locked",
            "path_points", "path_status", "auto_path", "auto_path_planned",
            "auto_path_segments", "auto_path_geo", "auto_path_valid", "auto_path_desc",
            "workflow_step", "simulation_done", "export_done",
            "last_rolling_rate", "last_crop_mask_geometric_coverage_pct",
            "last_planned_target_geometric_coverage_pct",
            "last_detected_mask_geometric_coverage_pct", "last_field_efficiency",
            "last_total_path_m", "last_work_path_m", "last_turn_path_m",
            "last_entry_exit_path_m", "path_validation", "path_provenance",
            "path_runtime", "path_artifact",
        )
        return {
            "kind": "system",
            "label": str(label or "系统操作"),
            "values": {
                name: self._snapshot_clone(getattr(self, name))
                for name in fields
            },
        }

    def _capture_route_snapshot(self) -> Dict[str, Any]:
        return self.capture_route_snapshot("路线编辑")

    def capture_route_snapshot(self, label: str = "路线编辑") -> Dict[str, Any]:
        """Capture only route-owned state so interactive edits stay responsive."""
        fields = (
            "path_points", "path_status", "auto_path", "auto_path_planned",
            "auto_path_segments", "auto_path_geo", "auto_path_valid",
            "auto_path_desc", "simulation_done", "export_done",
            "entry_point", "exit_point", "entry_point_locked",
            "exit_point_locked",
            "last_rolling_rate", "last_crop_mask_geometric_coverage_pct",
            "last_planned_target_geometric_coverage_pct",
            "last_detected_mask_geometric_coverage_pct",
            "last_field_efficiency", "last_total_path_m",
            "last_work_path_m", "last_turn_path_m",
            "last_entry_exit_path_m", "path_validation", "path_provenance",
            "path_runtime", "path_artifact",
        )
        return {
            "kind": "route_edit",
            "label": str(label or "路线编辑"),
            "values": {
                name: self._snapshot_clone(getattr(self, name))
                for name in fields
            },
        }

    def capture_mask_snapshot(self) -> Dict[str, Any]:
        return self.capture_system_snapshot("掩膜操作")

    def push_history(self, snapshot: Dict[str, Any], clear_redo: bool = True):
        if len(self.undo_stack) >= 50:
            self.undo_stack.pop(0)
        self.undo_stack.append(snapshot)
        if clear_redo:
            self.redo_stack.clear()

    def save_undo(self, label: str = "路线编辑"):
        self.push_history(self.capture_system_snapshot(label))

    def save_route_undo(self, label: str = "路线编辑"):
        self.push_history(self.capture_route_snapshot(label))

    def save_mask_undo(self, label: str = "掩膜操作"):
        self.push_history(self.capture_system_snapshot(label))

    def _snapshot_like(self, kind: str) -> Dict[str, Any]:
        return self.capture_system_snapshot("恢复前状态")

    def _restore_snapshot(self, snapshot: Dict[str, Any]):
        kind = snapshot.get("kind", "route")
        if kind in ("system", "route_edit"):
            for name, value in (snapshot.get("values") or {}).items():
                setattr(self, name, self._snapshot_clone(value))
        elif kind == "mask":
            self.mask_raw = self._copy_array(snapshot.get("mask_raw"))
            self.mask_offset_x = int(snapshot.get("mask_offset_x", 0))
            self.mask_offset_y = int(snapshot.get("mask_offset_y", 0))
            self.inference_done = bool(snapshot.get("inference_done", False))
            self.band_list = deepcopy(snapshot.get("band_list", []))
            self.corridors = deepcopy(snapshot.get("corridors", []))
            self.auto_path = deepcopy(snapshot.get("auto_path", []))
            self.auto_path_planned = bool(snapshot.get("auto_path_planned", False))
            self.path_points = deepcopy(snapshot.get("path_points", []))
            self.path_status = deepcopy(snapshot.get("path_status", []))
            self.entry_point = deepcopy(snapshot.get("entry_point"))
            self.exit_point = deepcopy(snapshot.get("exit_point"))
            self.unload_points = deepcopy(snapshot.get("unload_points", []))
            self.unload_point = deepcopy(snapshot.get("unload_point"))
            if not self.unload_points and self.unload_point is not None:
                self.unload_points = [self.unload_point]
            self.entry_point_locked = bool(snapshot.get("entry_point_locked", False))
            self.exit_point_locked = bool(snapshot.get("exit_point_locked", False))
            self.unload_point_locked = bool(snapshot.get("unload_point_locked", False))
            self.entry_exit_mode = bool(snapshot.get("entry_exit_mode", False))
            self.smooth_original_mask = self._copy_array(snapshot.get("smooth_original_mask"))
            self.smoothed_result = self._copy_array(snapshot.get("smoothed_result"))
            self._show_original = bool(snapshot.get("_show_original", False))
            self.workflow_step = int(snapshot.get("workflow_step", self.workflow_step))
            self._mask_overlay_dirty = True
            # 恢复路径规划结果字段
            self.auto_path_segments = deepcopy(snapshot.get("auto_path_segments", []))
            self.auto_path_geo = deepcopy(snapshot.get("auto_path_geo", []))
            self.auto_path_valid = bool(snapshot.get("auto_path_valid", False))
            self.auto_path_desc = str(snapshot.get("auto_path_desc", ""))
            self.path_validation = deepcopy(snapshot.get("path_validation", {}))
            self.path_provenance = deepcopy(snapshot.get("path_provenance", {}))
            self.path_runtime = deepcopy(snapshot.get("path_runtime", {}))
            self.path_artifact = deepcopy(snapshot.get("path_artifact", {}))
            # 恢复掩膜处理状态
            self.mask_processed = bool(snapshot.get("mask_processed", False))
            self.mask_result = snapshot.get("mask_result")
            self.turn_strategy = str(snapshot.get("turn_strategy", "contour"))
        else:
            self.path_points = deepcopy(snapshot.get("points", []))
            self.path_status = deepcopy(snapshot.get("status", []))
        self.turn_strategy = {
            "uturn": "bow",
            "bulb": "pear",
            "alpha": "fishtail",
        }.get(self.turn_strategy, self.turn_strategy)
        self.selected_type = None
        self.selected_idx = -1
        self.entry_exit_mode = False
        self._drag_entry_exit = None
        self.field_drawing_active = False
        self.field_drag_idx = -1
        self.anim_active = False
        self.anim_paused = False
        self.ctrl_panel_visible = False
        self._coord_cache = {}
        self._mask_overlay_dirty = True
        self.need_redraw = True

    def undo(self):
        if not self.undo_stack:
            self.status_message = "没有可撤回的系统操作"
            self.need_redraw = True
            return False
        last = self.undo_stack.pop()
        redo_label = f"恢复：{last.get('label', '系统操作')}"
        redo_snapshot = (
            self.capture_route_snapshot(redo_label)
            if last.get("kind") == "route_edit"
            else self.capture_system_snapshot(redo_label)
        )
        self.redo_stack.append(redo_snapshot)
        self._restore_snapshot(last)
        self.status_message = f"已撤回：{last.get('label', '系统操作')}"
        self._persist_history_restore("undo")
        return True

    def redo(self):
        if not self.redo_stack:
            self.status_message = "没有可恢复的系统操作"
            self.need_redraw = True
            return False
        nxt = self.redo_stack.pop()
        undo_label = nxt.get("label", "系统操作").replace("恢复：", "")
        self.undo_stack.append(
            self.capture_route_snapshot(undo_label)
            if nxt.get("kind") == "route_edit"
            else self.capture_system_snapshot(undo_label)
        )
        self._restore_snapshot(nxt)
        self.status_message = f"已恢复：{nxt.get('label', '系统操作').replace('恢复：', '')}"
        self._persist_history_restore("redo")
        return True

    def _persist_history_restore(self, stage: str):
        if not self.tif_path:
            return
        try:
            from cache import request_project_state_save
            request_project_state_save(self.tif_path, self, stage=stage)
        except Exception:
            pass

    def safe_update(self, **kwargs):
        """线程安全的状态更新：后台线程应通过此方法批量写入结果，避免竞态条件。
        用法: state.safe_update(mask_raw=mask, inference_done=True, status_message="完成")"""
        with self.state_lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.need_redraw = True
        tracked_keys = {
            "field_boundary",
            "field_area_m2",
            "forbidden_regions",
            "forbidden_regions_confirmed",
            "mask_raw",
            "mask_processed",
            "mask_result",
            "current_model_path",
            "harvester_params",
            "entry_point",
            "exit_point",
            "unload_points",
            "auto_path",
            "auto_path_planned",
            "auto_path_valid",
        }
        changed = {
            key: self._journal_value_summary(value)
            for key, value in kwargs.items()
            if key in tracked_keys
        }
        if changed:
            try:
                from project_log import ProjectJournal
                ProjectJournal().change(
                    "Application state updated",
                    module="state",
                    details=changed,
                )
            except Exception:
                pass

    @staticmethod
    def _journal_value_summary(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "shape"):
            return {
                "type": type(value).__name__,
                "shape": list(value.shape),
                "dtype": str(getattr(value, "dtype", "")),
            }
        if isinstance(value, (list, tuple, set)):
            return {"type": type(value).__name__, "count": len(value)}
        if isinstance(value, dict):
            return {
                "type": "dict",
                "keys": sorted(str(key) for key in value.keys())[:30],
            }
        return {"type": type(value).__name__}

    def safe_read_mask(self):
        """线程安全地读取 mask_raw 及其偏移量，返回副本避免后台线程覆盖。"""
        with self.state_lock:
            mask = self.mask_raw.copy() if self.mask_raw is not None else None
            return mask, self.mask_offset_x, self.mask_offset_y
