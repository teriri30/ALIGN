
"""
main_window.py - PySide6 主窗口。
"""
import os, sys, time, json, math, threading, hashlib, datetime, shutil, tempfile
from copy import deepcopy
from typing import Optional, List, Tuple
import numpy as np
import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Slot, QPointF
from PySide6.QtGui import QAction, QKeySequence, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFileDialog, QMessageBox, QStatusBar, QTabWidget, QLabel, QProgressBar,
    QDialog, QInputDialog, QApplication, QPushButton,
)
from config import (
    Config,
    AppLogger,
    save_json_atomic,
    DEFAULT_HARVESTER_PARAMS,
    HARVESTER_PARAM_KEYS,
    HARVESTER_PARAM_SOURCE_LABELS,
    MASK_PROCESSING_STRENGTH,
)
from state import AppState, AutoPathSegment, PathPoint
from geo import GeoUtils
from inference import ModelEngine, scan_model_dir
from pyside6_app.image_view import DISPLAY_LAYER_KEYS, ImageViewerGroup
from pyside6_app.harvester_visual import HarvesterItem
from pyside6_app.panels import (
    TASK_PANEL_COMPACT_MIN_WIDTH,
    TaskPanel,
    ParamPanel,
    LogPanel,
    RouteInfoPanel,
)
from pyside6_app.ui_motion import BreathingDot
from pyside6_app.top_bar import TopToolbar
from pyside6_app.workers import (
    PipelineWorker, MaskProcessWorker, CenterlineCatalogWorker, PlanWorker, TifLoadWorker,
    CacheRestoreWorker, MaskDiagnosticExportWorker, InferenceDiagnosticExportWorker,
    ModelLoadWorker,
)
from pyside6_app.event_handler import EventBridge
from pyside6_app.dialogs import (
    HarvesterParamsDialog, ModelManagerDialog, SettingsDialog,
    ProjectLogDialog, ActionLogDialog, ReportDialog, DiagnosisDialog, LoadingProgress,
    MachineRecommendationDialog,
)
from pyside6_app.launcher import LauncherDialog
from pyside6_app.styles import COLORS, ThemeManager, ThemeMode, THEMES, build_stylesheet
from pyside6_app.task_state import TASK_ORDER, TASK_LABELS, derive_task_statuses, rollback_summary
from workflow import WorkflowUpdater, AnimationEngine
from project_log import ProjectJournal
from task_lifecycle import admit_task_result, make_task_token
from pyside6_app.project_session import cached_path_restore_plan, reset_project_state
from runtime_paths import builtin_model_dir
import interaction_log
import operator_paths


# ``ImageViewerGroup`` accepts BGR tuples and reverses them before creating a
# Qt colour.  Keep the field-selection boundary blue at every redraw site.
FIELD_BOUNDARY_COLOR_BGR = (255, 140, 50)
# DECISION-028: 人工参考线（中心线评价的基准证据，研究工具）。同 BGR 口径：
# 参考线橙色（RGB 橙 = BGR (0,160,255)），桩点红橙色。
REFERENCE_LINE_COLOR_BGR = (0, 160, 255)
REFERENCE_STATION_COLOR_BGR = (70, 70, 235)
#: 拟合基准线（评价用的直线口径）。必须比桩点连线暗、细，否则用户分不清
#: "我画的线"和"评价用的线"，也就会把拟合偏差看成界面错位。
REFERENCE_FIT_COLOR_BGR = (120, 120, 120)
REFERENCE_DOUBLE_CLICK_SECONDS = 0.45
REFERENCE_DOUBLE_CLICK_DISTANCE_PX = 12.0
# DECISION-060: 作业线标定（人工纠偏）。人拖出来的线就是本次规划采用的线，
# 颜色与算法中心线（绿）区分开，避免"改了看不出改没改"。
OPERATOR_LINE_COLOR_BGR = (235, 200, 60)
OPERATOR_EDIT_COLOR_BGR = (60, 255, 255)
CALIBRATION_PICK_TOLERANCE_M = 1.0
CALIBRATION_PREVIEW_POINT_LIMIT = 400


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALIGN 农机作业规划系统")
        self.resize(1600, 1000); self.setMinimumSize(1024, 680)
        self._theme_mgr = ThemeManager.instance()
        self._theme_mgr.theme_changed.connect(self._on_theme_changed)
        self.apply_stylesheet()
        self.cfg = Config(); self.cfg.load()
        self.state = AppState(); self.geo = GeoUtils()
        self._mask_processing_strength = MASK_PROCESSING_STRENGTH
        # 从 config.json 加载农机参数；缺项使用 Config 默认值补齐，避免默认参数被判定“不完整”。
        self.state.harvester_params = self._effective_harvester_params({})
        self.state.harvester_param_sources = self._default_harvester_param_sources(
            self.state.harvester_params
        )
        self.model_engine = ModelEngine()
        self._tif_rgb: Optional[np.ndarray] = None
        self._tif_path: Optional[str] = None
        self._tif_source_identity: dict = {}
        self._file_hash_cache: dict = {}
        # Keep the last directory used for each diagnostic stage so a second
        # export continues numbering beside the first one, even when the
        # operator changed the save dialog from the TIFF directory to Desktop.
        self._diagnostic_export_dirs: dict = {}
        self._snapshot_directory: str = ""
        self._pipeline_result: Optional[dict] = None
        # Transient visual-only route retained while a changed service point is
        # waiting for formal replanning.  It is deliberately not part of
        # AppState snapshots/provenance/export and can never certify a route.
        self._stale_path_preview = None
        # Exact, transient copy of the most recently written machine
        # compatibility files.  This is a display artifact only: canonical
        # planning geometry remains the source for simulation and validation.
        self._machine_route_preview = None
        self._route_display_mode = "continuous"
        self._log = AppLogger()
        self._journal = ProjectJournal()
        # 交互日志：记的是"用户做了什么"，与项目日志（错误/配置变更）分工不同。
        # 版本与影像上下文随每条记录落盘，日志可以脱离本次会话单独阅读。
        try:
            from version import APP_VERSION

            interaction_log.configure(app="ALIGN", version=APP_VERSION)
            interaction_log.record("app.start", APP_VERSION)
        except Exception:
            pass
        self.event_bridge = EventBridge(self.state, self.geo)
        # 动画引擎
        self._anim_engine: Optional[AnimationEngine] = None
        self._anim_timer: Optional[QTimer] = None
        self._anim_overlay_items = []
        # 真实机器视觉层（与足迹校验模型分离）
        self._harvester_item: Optional[HarvesterItem] = None
        self._cut_swath_item = None
        self._anim_phase = 0.0
        # 田块边界绘制状态
        self._field_drawing = False
        self._field_pts: List[Tuple[int, int]] = []
        self._forbidden_drawing = False
        self._forbidden_pts: List[Tuple[int, int]] = []
        # DECISION-028: 参考线标注（中心线评价的人工基准，研究工具）
        self._reference_drawing = False
        self._reference_pass = 1
        self._reference_pts: List[Tuple[float, float]] = []
        self._reference_last_click = None
        self._reference_payload = {}
        self._reference_payload_tif = ""
        # DECISION-060: 作业线标定（人工纠偏）。_calibration_edits 按中心线 id
        # 存拖动记录（米制弧长 + 横向偏移），人线 = 算法线 + 这些偏移；每条
        # 拖动立即写 sidecar，不依赖"保存项目"。
        self._calibration_drawing = False
        self._calibration_item_id = ""
        self._calibration_press = None
        self._calibration_preview = None
        self._calibration_payload = {}
        self._calibration_payload_tif = ""
        self._calibration_edits: dict = {}
        self._calibration_history: List[Tuple[str, list]] = []
        self._calibration_originals: dict = {}
        self._measure_tool = ""
        self._measure_points: List[Tuple[int, int]] = []
        self._measure_overlay_items = []
        self._measure_complete = False
        self._route_edit_active = False
        # DECISION-019: manual route editing operates on work-line objects,
        # while ``path_points/path_status/path_geometry_modes`` remains the
        # single control-route contract shared with compatibility export.
        self._route_edit_mode = "select"
        self._route_new_line_start = None
        self._route_drag_idx = -1
        self._route_drag_work_index = -1
        self._route_drag_work_origin = None
        self._route_drag_work_points = None
        self._route_edit_dynamic_items = []
        self._show_mask_before = False
        self._service_points_visible = False
        self._service_point_overlay_items = []
        self._service_point_drag_origin = None
        self._service_point_drag_undo_saved = False
        self._service_point_drag_had_path = False
        self._service_point_drag_moved = False
        self._service_point_drag_preview_items = []
        self._service_point_drag_last_correction = {}
        self._service_point_replan_timer = QTimer(self)
        self._service_point_replan_timer.setSingleShot(True)
        self._service_point_replan_timer.timeout.connect(
            self._run_service_point_drag_replan
        )
        # 起终点与卸粮点放置状态：0=起点, 1=终点, 2+=卸粮点
        self._entry_exit_click_idx = 0
        # 模型列表
        self._model_list: list = []
        self._model_actions = []
        self._build_menu(); self._build_central(); self._build_statusbar()
        self._ensure_field_trial_planning_mode()
        self._on_theme_changed(self._theme_mgr.current_theme())
        self.task_panel.set_mask_strength(self._mask_processing_strength)
        self.task_panel.set_turn_strategy(self.state.turn_strategy)
        self._connect_signals()
        # DECISION-025: save_undo() is called immediately before a logical UI
        # mutation.  A short single-shot delay checkpoints the completed state
        # and its new immutable history entry without serialising masks on the
        # GUI thread for every mouse-move event.
        self._history_checkpoint_timer = QTimer(self)
        self._history_checkpoint_timer.setSingleShot(True)
        self._history_checkpoint_timer.setInterval(650)
        self._history_checkpoint_timer.timeout.connect(
            self._persist_completed_history_checkpoint
        )
        self.state.set_history_persist_callback(
            self._schedule_history_checkpoint
        )
        self._state_timer = QTimer(self)
        self._state_timer.timeout.connect(self._refresh_state); self._state_timer.start(200)
        self._pending_segment_display = False
        self._pending_plan_display = False
        self._pending_timer = QTimer(self)
        self._pending_timer.timeout.connect(self._process_pending); self._pending_timer.start(50)
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[PipelineWorker] = None
        self._task_cancel_requested = False
        self._active_inference_token = None
        self._active_inference_runner = None
        # Model construction can initialise CUDA and deserialize a large
        # PyTorch checkpoint.  Keep the request state separate from the
        # active engine so a failed/cancelled load never half-replaces it.
        self._model_load_in_progress = False
        self._active_model_load_token = None
        self._pending_model_load = None
        self._pending_model_action = None
        self._active_mask_token = None
        self._active_centerline_token = None
        self._active_plan_token = None
        self._active_cache_restore_token = None
        self._active_plan_generation = None
        self._close_after_worker_cancel = False
        self._close_cancel_started_at = 0.0
        self._close_cancel_warning_shown = False
        self._processed_mask_save_generation = 0
        self._processed_mask_save_threads = []
        self._processed_mask_save_lock = threading.Lock()
        self._log.info("desktop startup complete")
        if not os.environ.get("ALIGN_DISABLE_AUTO_START"):
            QTimer.singleShot(300, self._auto_start)


    # UI 构建

    def _effective_harvester_params(self, override=None):
        """Return complete harvester geometry, filling missing config/project keys with defaults."""
        params = {
            "cutter_width_m": float(self.cfg.CUTTER_WIDTH_M),
            "cutter_forward_offset_m": float(self.cfg.CUTTER_FORWARD_OFFSET_M),
            "track_width_m": float(self.cfg.TRACK_WIDTH_M),
            "track_gauge_m": float(self.cfg.TRACK_GAUGE_M),
            "wheelbase_m": float(self.cfg.WHEELBASE_M),
            "track_length_m": float(self.cfg.TRACK_LENGTH_M),
            "turn_radius_m": float(self.cfg.TURN_RADIUS_M),
            "work_speed_mps": float(self.cfg.WORK_SPEED_MPS),
            "turn_speed_mps": float(self.cfg.TURN_SPEED_MPS),
            "entry_speed_mps": float(self.cfg.ENTRY_SPEED_MPS),
        }
        for source in (self.cfg.section("harvester"), override or {}):
            if not isinstance(source, dict):
                continue
            for key in params:
                value = source.get(key)
                if value not in (None, ""):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        params[key] = value
        return params

    def _desktop_path_planning_runtime_config(self, harvester=None):
        """Return the runtime overrides shared by planning and export checks.

        DECISION-001/007: the path provenance must describe the exact desktop
        execution contract.  In particular, the desktop workflow keeps
        service points as references and writes the final short exit
        extension into the shared compatibility control route.  A contour
        request owns its field-wide ordering; a legacy ``route_order`` value
        must not bring back adjacent-row execution.
        """
        params = self._effective_harvester_params(
            harvester if isinstance(harvester, dict) else {}
        )
        turn_strategy = str(
            getattr(self.state, "turn_strategy", "auto") or "auto"
        )
        runtime = {
            "min_turn_radius_m": float(params["turn_radius_m"]),
            "turn_strategy": turn_strategy,
            "service_point_mode": "reference_only",
            "include_route_finish_extension": True,
            "forbidden_mask_confirmed": bool(
                getattr(self.state, "forbidden_regions_confirmed", False)
            ),
        }
        selected_ids = list(
            getattr(self.state, "selected_centerline_ids", None)
            or getattr(self.state, "selected_work_band_ids", [])
            or []
        )
        selection_report = (
            getattr(self.state, "centerline_selection_report", None)
            or getattr(self.state, "work_band_selection_report", {})
            or {}
        )
        if selected_ids and selection_report.get("selection_origin") == "operator":
            runtime["selected_centerline_ids"] = selected_ids
        # 人工标定线优先：只要本项目存在与当前掩膜版本匹配的标定线，规划就用它
        # 替换算法中心线（导出、转弯、栅格化随同一条线走）。
        try:
            operator_overrides = self._operator_overrides()
        except Exception:
            operator_overrides = {}
        if operator_overrides:
            runtime["operator_centerlines"] = operator_overrides
        try:
            from path_planner import normalize_turn_strategy

            if normalize_turn_strategy(turn_strategy) == "contour":
                runtime["route_order"] = "contour"
        except Exception:
            if turn_strategy.strip().lower() in {"contour", "spiral"}:
                runtime["route_order"] = "contour"
        return runtime

    @staticmethod
    def _normalise_harvester_param_source(source) -> str:
        value = str(source or "").strip().lower()
        aliases = {
            "默认": "default",
            "系统默认值": "default",
            "智能推荐": "recommended",
            "推荐": "recommended",
            "用户自定义": "user_entered",
            "custom": "user_entered",
            "用户填写": "user_entered",
            "用户实测": "user_measured",
            "实测": "user_measured",
        }
        value = aliases.get(value, value)
        return value if value in HARVESTER_PARAM_SOURCE_LABELS else "legacy_unverified"

    def _default_harvester_param_sources(self, params=None, fallback="default"):
        values = params if isinstance(params, dict) else self._effective_harvester_params({})
        source = self._normalise_harvester_param_source(fallback)
        return {
            key: source
            for key in HARVESTER_PARAM_KEYS
            if key in values
        }

    def _normalise_harvester_param_sources(self, params=None, sources=None, *, fallback="legacy_unverified"):
        values = params if isinstance(params, dict) else self._effective_harvester_params({})
        raw_sources = sources if isinstance(sources, dict) else {}
        default_source = self._normalise_harvester_param_source(fallback)
        return {
            key: self._normalise_harvester_param_source(
                raw_sources.get(key, default_source)
            )
            for key in HARVESTER_PARAM_KEYS
            if key in values
        }

    def _harvester_source_summary(self, params=None, sources=None):
        values = params if isinstance(params, dict) else self._effective_harvester_params({})
        normalised = self._normalise_harvester_param_sources(
            values,
            sources if sources is not None else getattr(self.state, "harvester_param_sources", {}),
        )
        return {
            key: {
                "source": normalised.get(key, "legacy_unverified"),
                "label": HARVESTER_PARAM_SOURCE_LABELS.get(
                    normalised.get(key, "legacy_unverified"),
                    normalised.get(key, "legacy_unverified"),
                ),
            }
            for key in HARVESTER_PARAM_KEYS
            if key in values
        }

    def apply_stylesheet(self):
        self.setStyleSheet(build_stylesheet(self._theme_mgr.colors()))

    def _on_theme_changed(self, theme):
        # 更新全局 COLORS dict，供其他模块引用。
        import pyside6_app.styles as _styles
        new_colors = dict(self._theme_mgr.colors())
        _styles.COLORS.clear()
        _styles.COLORS.update(new_colors)
        self.apply_stylesheet()
        # 刷新所有面板内联样式。
        self.task_panel.refresh_theme()
        self.top_toolbar.refresh_theme()
        self.top_toolbar.update_theme_icon(theme)
        self.route_info.refresh_theme()
        self.log_panel.refresh_theme()
        strip = getattr(self, "status_strip", None)
        if strip is not None:
            strip.refresh_theme()
        for v in (self.image_view,):
            v.refresh_theme()
        # 强制重绘。
        self.update()
        self._log.info(f"主题切换: {theme}")

    def _build_menu(self):
        """菜单栏：每个功能只出现一次，按"文件 / 视图 / 工具 / 设置 / 帮助"归位。

        旧版把"系统设置/农机参数/模型管理/项目日志/诊断"同时挂在「系统设置」和
        「工具」下，同一个动作注册两遍、快捷键也重名（Qt 会报 ambiguous shortcut），
        而且文件菜单平铺了 8 个导出项。这里统一收进子菜单，删掉重复项。
        """
        mb = self.menuBar()
        # Windows 上 QMenuBar 默认是原生菜单条：它吃不到 QSS，永远是系统浅色，
        # 还会把窗口顶部挤成"双层"。关掉原生，菜单才会跟深色主题一致。
        mb.setNativeMenuBar(False)

        # ---------------------------------------------------------------- 文件
        fm = mb.addMenu("文件")
        self._add_action(fm, "打开影像…", QKeySequence.Open, self._on_open_tif)
        self._add_action(fm, "打开内置测试影像", "Ctrl+Shift+O", self._on_open_builtin_test)
        self._add_action(fm, "保存项目", QKeySequence.Save, self._on_save_project)
        self._add_action(fm, "加载模型…", "Ctrl+M", self._on_load_model)
        fm.addSeparator()

        export_menu = fm.addMenu("导出")
        self._add_action(
            export_menu, "TXT 作业路径（$PATH）…", None, self._on_export_path_format
        )
        self._add_action(
            export_menu, "TXT 编号路径（001.txt、002.txt）…", None,
            self._on_export_machine_compatibility_path,
        )
        export_menu.addSeparator()
        self._add_action(export_menu, "GeoJSON…", "Ctrl+E", self._on_export_geojson)
        self._add_action(export_menu, "CSV（经纬度）…", "Ctrl+Shift+E", self._on_export_csv_geo)
        self._add_action(export_menu, "KML…", None, self._on_export_kml)
        self._add_action(export_menu, "JSON…", None, self._on_export_json)
        export_menu.addSeparator()
        self._add_action(export_menu, "快照 PNG…", None, self._on_export_img)
        self._add_action(export_menu, "试验资料包…", None, self._on_export_trial_package)
        export_menu.addSeparator()
        self._add_action(
            export_menu, "识别诊断…", None, self._on_export_inference_diagnostics
        )
        self._add_action(export_menu, "掩膜优化诊断…", None, self._on_export_mask_diagnostics)

        fm.addSeparator()
        self._add_action(fm, "退出", "Ctrl+Q", self.close)

        # ---------------------------------------------------------------- 视图
        vm = mb.addMenu("视图")
        self._add_action(vm, "适应窗口", "Ctrl+0", lambda: self.image_view.viewer.fit_to_view())
        self._add_action(vm, "放大", "Ctrl++", lambda: self.image_view.viewer.zoom_in())
        self._add_action(vm, "缩小", "Ctrl+-", lambda: self.image_view.viewer.zoom_out())
        self._add_action(vm, "重置视图", None, lambda: self.image_view.viewer.reset_view())
        # 不给单键快捷键：焦点在文本框时按 R 也会转图（旧版就是 "R"）
        self._add_action(
            vm, "旋转 90°", None,
            lambda: self.image_view.viewer.set_rotation(self.image_view.viewer._rotation_deg + 90),
        )
        vm.addSeparator()
        theme_menu = vm.addMenu("主题")
        self._add_action(theme_menu, "深色", None, lambda: self._set_theme_mode(ThemeMode.DARK))
        self._add_action(theme_menu, "浅色", None, lambda: self._set_theme_mode(ThemeMode.LIGHT))
        self._add_action(theme_menu, "跟随系统", None, lambda: self._set_theme_mode(ThemeMode.SYSTEM))

        # ---------------------------------------------------------------- 工具
        tm = mb.addMenu("工具")
        self._add_action(tm, "运行识别", "Ctrl+R", self._on_run)
        self._add_action(tm, "掩膜优化", None, self._on_mask_process)
        tm.addSeparator()
        self._add_action(tm, "田块圈选", "Ctrl+B", self._on_toggle_field_drawing)
        self._add_action(tm, "绘制禁行区", "Ctrl+Shift+B", self._on_toggle_forbidden_drawing)
        self._add_action(tm, "清除全部禁行区", None, self._on_clear_forbidden_regions)
        self._add_action(tm, "布设起终点与卸粮点", None, self._on_toggle_entry_exit)
        tm.addSeparator()
        ref_menu = tm.addMenu("参考线标注（研究工具）")
        self._add_action(ref_menu, "第 1 遍", None, lambda: self._set_reference_mode(1))
        self._add_action(ref_menu, "第 2 遍（重复性）", None, lambda: self._set_reference_mode(2))
        self._add_action(ref_menu, "完成当前参考线（保存）", None, self._on_finish_reference_line)
        self._add_action(ref_menu, "显示 / 隐藏参考线图层", None, self._on_toggle_reference_layer)
        self._add_action(ref_menu, "导出参考线 JSON…", None, self._on_export_reference_lines)
        ref_menu.addSeparator()
        self._add_action(
            ref_menu, "删除最后一条参考线", None, self._on_delete_last_reference_line
        )
        self._add_action(
            ref_menu, "清除本项目参考线…", None, self._on_clear_reference_lines
        )
        tm.addSeparator()
        cal_menu = tm.addMenu("作业线标定（人工纠偏）")
        self._add_action(cal_menu, "开始 / 结束标定", None, self._on_toggle_calibration)
        self._add_action(cal_menu, "清除当前线标定", None, self._on_clear_calibration)
        self._add_action(cal_menu, "导入参考线为标定线", None, self._on_import_reference_as_operator)
        self._add_action(cal_menu, "导出标定案例…", None, self._on_export_calibration_case)

        # ---------------------------------------------------------------- 设置
        sm = mb.addMenu("设置")
        self._add_action(sm, "系统设置…", "Ctrl+,", self._on_settings_dialog)
        self._add_action(sm, "农机参数…", "Ctrl+P", self._on_params_dialog)
        self._model_actions.append(
            self._add_action(sm, "模型管理…", "Ctrl+Shift+M", self._on_model_manager)
        )
        sm.addSeparator()
        self._add_action(sm, "项目日志…", None, self._on_project_log)
        self._add_action(sm, "操作记录…", None, self._on_action_log_dialog)
        self._add_action(sm, "系统诊断…", None, self._on_diagnose)
        self._add_action(sm, "打开数据目录", None, self._on_open_data_dir)

        # ---------------------------------------------------------------- 帮助
        hm = mb.addMenu("帮助")
        self._add_action(hm, "使用说明", "F1", self._on_open_manual)
        self._add_action(hm, "关于 ALIGN", None, self._on_about)

    # ------------------------------------------------------------ 菜单配套动作
    def _on_open_builtin_test(self):
        """直接载入随包发布的内置测试影像，省掉找文件的步骤（调试常用）。"""
        from builtin_test_assets import builtin_test_image

        path = builtin_test_image()
        if not path:
            self.log_panel.warn("内置测试影像缺失 · 安装包不完整")
            return
        self._load_tif_path(path)

    def _on_open_data_dir(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from runtime_paths import user_data_root

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_data_root())))

    def _on_open_manual(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from runtime_paths import bundled_resource

        for name in ("docs/USER_MANUAL.md", "README.md"):
            path = bundled_resource(name)
            if os.path.isfile(path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
                return
        self.log_panel.warn("没找到使用说明 · 安装包不完整")

    def _on_about(self):
        from PySide6.QtWidgets import QMessageBox
        from version import APP_VERSION
        from app_entry import describe_device

        QMessageBox.about(
            self,
            "关于 ALIGN",
            f"<b>ALIGN</b> {APP_VERSION}<br>"
            "Autonomous Line-guided Intelligent Guidance &amp; Navigation<br><br>"
            f"运行设备：{describe_device()}<br>"
            "安徽农业大学 · 农业工程",
        )

    def _add_action(self, menu, text, shortcut, slot):
        act = QAction(text, self)
        if shortcut: act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        # 交互日志：菜单/快捷键触发的动作全部留痕（含没保存项目的那部分）。
        act.triggered.connect(
            lambda _checked=False, _text=text, _menu=menu.title(): self._log_action(
                "menu.action", "%s / %s" % (_menu, _text), menu=_menu, label=_text
            )
        )
        menu.addAction(act); return act

    # ------------------------------------------------------------ 交互日志
    def _log_action(self, action, target="", **payload):
        """统一入口：任何用户动作都走这里，写盘失败也不影响界面。"""
        try:
            interaction_log.record(action, target, tif=str(getattr(self.state, "tif_path", "") or ""), **payload)
        except Exception:
            pass

    def _build_central(self):
        c = QWidget(); c.setObjectName("central"); self.setCentralWidget(c)
        lo = QVBoxLayout(c); lo.setContentsMargins(0,0,0,0); lo.setSpacing(0)
        self.top_toolbar = TopToolbar(); lo.addWidget(self.top_toolbar)
        sp = QSplitter(Qt.Horizontal); sp.setHandleWidth(2)
        # 左侧栏：任务清单 + 路径结果（垂直分割）
        left_split = QSplitter(Qt.Vertical); left_split.setHandleWidth(2)
        self.task_panel = TaskPanel()
        left_split.addWidget(self.task_panel)
        self.route_info = RouteInfoPanel()
        left_split.addWidget(self.route_info)
        left_split.setStretchFactor(0, 5); left_split.setStretchFactor(1, 2)
        left_split.setSizes([700, 240])
        sp.addWidget(left_split)
        # 右侧：影像视图（底图、掩膜、路径）+ 日志
        rp = QWidget()
        rl = QVBoxLayout(rp); rl.setContentsMargins(2,2,2,2); rl.setSpacing(2)
        self.image_view = ImageViewerGroup()
        rl.addWidget(self.image_view, 1)
        self.log_panel = LogPanel()
        rl.addWidget(self.log_panel)
        sp.addWidget(rp); sp.setStretchFactor(0,0); sp.setStretchFactor(1,1); lo.addWidget(sp, 1)
        # 左侧任务清单在启动时保持一个不挤占的宽度（约 1/6 屏），避免被
        # 右侧影像区挤压到最小宽度；同时设最小宽度防止拖得太窄文字被裁切。
        left_split.setMinimumWidth(TASK_PANEL_COMPACT_MIN_WIDTH)
        _left_w = max(TASK_PANEL_COMPACT_MIN_WIDTH, self.width() // 6)
        sp.setSizes([_left_w, self.width() - _left_w])
        self._left_split = left_split
        self._outer_splitter = sp

    def _connect_signals(self):
        tp = self.task_panel
        tp.load_tif_requested.connect(self._on_open_tif)
        tp.load_model_requested.connect(self._on_load_model)
        tp.run_requested.connect(self._on_run)
        tp.segment_requested.connect(self._on_segment)
        tp.export_inference_diagnostics_requested.connect(
            self._on_export_inference_diagnostics
        )
        tp.process_requested.connect(self._on_mask_process)
        tp.export_mask_diagnostics_requested.connect(self._on_export_mask_diagnostics)
        tp.plan_requested.connect(self._on_plan)
        tp.simulate_requested.connect(self._on_simulate_task_action)
        tp.params_requested.connect(self._on_params_dialog)
        tp.entry_exit_toggle.connect(self._on_toggle_entry_exit)
        tp.export_machine_compatibility_requested.connect(
            self._on_export_machine_compatibility_path
        )
        tp.export_csv_requested.connect(self._on_export_csv_geo)
        tp.export_img_requested.connect(self._on_export_img)
        tp.field_drawing_requested.connect(self._on_toggle_field_drawing)
        tp.model_changed.connect(self._on_model_combo_changed)
        tp.mask_strength_changed.connect(self._on_mask_strength_changed)
        tp.turn_strategy_changed.connect(self._on_turn_strategy_changed)
        tp.confirm_no_forbidden_requested.connect(self._on_confirm_no_forbidden)
        tp.draw_forbidden_requested.connect(self._on_toggle_forbidden_drawing)
        tp.recommend_service_points_requested.connect(
            self._on_recommend_service_points
        )
        tp.route_edit_requested.connect(self._on_route_edit_requested)
        tp.route_edit_tool_requested.connect(self._on_route_edit_tool_requested)
        tp.step_deleted.connect(self._on_step_deleted)
        t = self.top_toolbar
        t.action_clicked.connect(self._on_toolbar_action)
        t.tool_changed.connect(self._on_tool_changed)
        t.turn_strategy_changed.connect(self._on_turn_strategy_changed)
        t.model_changed.connect(self._on_model_combo_changed)
        t.model_dropdown_opened.connect(self._on_model_dropdown_open)
        t.theme_toggle.connect(self._toggle_theme)
        self.route_info.segment_selected.connect(self._on_segment_selected)
        self.route_info.service_point_selected.connect(self._on_service_point_selected)
        self.route_info.edit_segment_selected.connect(
            self._on_route_list_segment_selected
        )
        self.route_info.edit_action_requested.connect(
            self._on_route_list_edit_action
        )
        self.route_info.work_reorder_requested.connect(
            self._on_route_list_work_reordered
        )
        self.task_panel.centerline_selection_changed.connect(
            self._on_centerline_selection_changed
        )
        self.task_panel.centerline_selection_confirm_requested.connect(
            self._confirm_centerline_selection
        )
        self.task_panel.centerline_generate_requested.connect(
            self._start_centerline_catalog_generation
        )
        # 鼠标事件：田块圈选、起终点与卸粮点布设
        self.event_bridge.double_clicked.connect(self._on_field_double_click)
        vw = self.image_view.viewer
        vw.mouse_pressed.connect(self._on_viewer_mouse_pressed)
        vw.mouse_released.connect(self._on_viewer_mouse_released)
        vw.mouse_moved.connect(self._on_viewer_mouse_moved)
        vw.mouse_dragged.connect(self.event_bridge.on_mouse_dragged)
        vw.set_direct_drag_hit_test(self._service_point_direct_drag_available)
        vw.view_scale_changed.connect(self._on_view_scale_changed)
        self.image_view.snapshot_requested.connect(self._on_save_snapshot)
        self.image_view.mask_compare_requested.connect(self._on_toggle_mask_compare)
        self.image_view.display_layer_changed.connect(
            self._on_display_layer_changed
        )
        self.image_view.measure_tool_changed.connect(self._on_measure_tool_changed)
        self.image_view.route_display_mode_changed.connect(
            self._on_route_display_mode_changed
        )
        self.image_view.animation_pause_requested.connect(self._on_animation_pause)
        self.image_view.animation_seek_requested.connect(self._on_animation_seek)
        self.image_view.animation_speed_requested.connect(self._on_animation_speed)

    def set_status_text(self, text, tone="idle"):
        """Single entry point for the status bar text + its tone dot."""
        strip = getattr(self, "status_strip", None)
        if strip is not None:
            strip.set_phase(text, tone)
        else:
            self.lbl_status.setText(str(text))
        dot = getattr(self, "status_dot", None)
        if dot is not None:
            tone_colors = {
                "busy": COLORS.get("accent", "#34D399"),
                "ready": COLORS.get("green", "#34D399"),
                "error": COLORS.get("red", "#E85D63"),
                "idle": COLORS.get("text_dim", "#94A3B8"),
            }
            dot.set_tone(tone, tone_colors.get(tone, COLORS.get("text_dim", "#94A3B8")))

    def _build_statusbar(self):
        self.status_bar = QStatusBar()
        # Runtime dot in front of the status text. Reference behaviour: the
        # dot breathes (1.2 s) while a job runs, is solid green when ready and
        # red on failure — the text alone reads as static.
        # 状态条按「步骤 / 阶段 / 提示 / 指标」四段排（DECISION-049）。
        # lbl_status 保留为兼容别名：154 处旧调用 setText() 会走 set_phase()。
        from pyside6_app.panels import StatusStrip

        self.status_strip = StatusStrip()
        self.status_dot = self.status_strip.dot
        self.status_dot.set_tone("idle", COLORS.get("text_dim", "#94A3B8"))
        self.lbl_status = self.status_strip
        self.lbl_status.setText("就绪")
        self.lbl_speed = QLabel("")
        self.progress_bar = QProgressBar(); self.progress_bar.setMaximumWidth(180)
        self.progress_bar.setMaximumHeight(6); self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        self.btn_cancel_task = QPushButton("取消")
        self.btn_cancel_task.setToolTip("取消当前耗时任务")
        self.btn_cancel_task.setAccessibleName("取消当前耗时任务")
        self.btn_cancel_task.setFixedHeight(22)
        self.btn_cancel_task.hide()
        self.btn_cancel_task.clicked.connect(self._cancel_background_worker)
        self.status_bar.addWidget(self.status_strip, 1)
        self.status_bar.addPermanentWidget(self.lbl_speed)
        self.status_bar.addPermanentWidget(self.btn_cancel_task)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)

    # 启动与影像加载


    def _inference_task_is_running(self):
        """Return the real inference activity, including its native thread."""
        if bool(getattr(self.state, "inference_running", False)):
            return True
        runner = getattr(self, "_active_inference_runner", None)
        is_running = getattr(runner, "is_running", None)
        try:
            return bool(is_running and is_running())
        except Exception:
            # A broken probe must fail closed while protecting the model
            # engine from being replaced concurrently with inference.
            return True

    def _has_background_worker(self):
        thread_running = bool(
            getattr(self, "_worker_thread", None)
            and self._worker_thread.isRunning()
        )
        return thread_running or self._inference_task_is_running()

    def _set_model_loading_ui(self, loading):
        """Keep model controls and inference actions disabled during loading."""
        loading = bool(loading)
        self._model_load_in_progress = loading
        for action in list(getattr(self, "_model_actions", []) or []):
            try:
                action.setEnabled(not loading)
            except RuntimeError:
                pass
        task_panel = getattr(self, "task_panel", None)
        if task_panel is not None and hasattr(task_panel, "set_model_loading"):
            task_panel.set_model_loading(loading)
        toolbar = getattr(self, "top_toolbar", None)
        if toolbar is not None and hasattr(toolbar, "set_model_loading"):
            toolbar.set_model_loading(loading)

    def _restore_model_selector(self):
        """Restore visible selectors to the authoritative loaded model."""
        current = str(getattr(getattr(self, "state", None), "current_model_path", "") or "")
        task_panel = getattr(self, "task_panel", None)
        if task_panel is not None and hasattr(task_panel, "set_model_options"):
            task_panel.set_model_options(getattr(self, "_model_list", []), current)
        toolbar = getattr(self, "top_toolbar", None)
        if toolbar is not None:
            names = [m.get("name", "") for m in getattr(self, "_model_list", [])]
            if hasattr(toolbar, "populate_models"):
                toolbar.populate_models(names)
            if current and hasattr(toolbar, "set_current_model"):
                toolbar.set_current_model(os.path.basename(current))

    def _start_background_worker(self, worker, run_message, *, warn_if_busy=True, busy_message=None):
        """Start one cancellable worker without blocking the Qt UI thread."""
        if self._has_background_worker():
            if warn_if_busy:
                self.log_panel.warn(
                    f"任务正在运行：{busy_message or '当前已有耗时任务正在执行，请等待完成后再操作。'}"
                )
            return False
        self._worker_thread = QThread(self)
        self._worker = worker
        self._task_cancel_requested = False
        worker.moveToThread(self._worker_thread)
        worker.progress.connect(self._on_progress)
        worker.error.connect(self._on_worker_error)
        self._worker_thread.started.connect(worker.run)
        # The worker owns its event loop. Schedule QObject deletion while that
        # loop is still alive; deleting from thread.finished is too late and
        # can leave the Shiboken wrapper/native QObject racing at teardown.
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        worker.finished.connect(self._worker_thread.quit)
        worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._clear_background_worker_refs)
        self._worker_thread.start()
        self.progress_bar.show()
        if hasattr(self, 'btn_cancel_task'):
            self.btn_cancel_task.show()
        self.top_toolbar.set_worker_cancel_visible(True)
        self.log_panel.info(run_message)
        return True

    def _cancel_background_worker(self):
        """Request cancellation without treating it as a processing failure."""
        inference_runner = getattr(self, '_active_inference_runner', None)
        inference_running = bool(getattr(self.state, 'inference_running', False) and inference_runner)
        if not self._has_background_worker() and not inference_running:
            return
        self._task_cancel_requested = True
        inference_token = getattr(self, '_active_inference_token', None)
        if inference_token is not None:
            inference_token.invalidate()
        if inference_runner is not None:
            inference_runner.cancel()
        for token_name in (
            "_active_mask_token",
            "_active_centerline_token",
            "_active_plan_token",
            "_active_cache_restore_token",
            "_active_model_load_token",
        ):
            token = getattr(self, token_name, None)
            if token is not None:
                token.invalidate()
        if getattr(self.state, 'mask_processing_running', False):
            self.state.mask_processing_running = False
            self.state.mask_processing_progress = 0.0
            self.task_panel.set_task_status("process", "available", text="已取消")
        if getattr(self.state, 'plan_running', False):
            self.state.plan_running = False
            self.state.plan_progress = 0.0
            self.task_panel.set_task_status("plan", "available", text="已取消")
        if self._worker and hasattr(self._worker, 'cancel'):
            self._worker.cancel()
        self.set_status_text("正在取消当前任务...", "busy")
        self.log_panel.info("已请求取消当前任务")
        if hasattr(self, 'btn_cancel_task'):
            self.btn_cancel_task.setEnabled(False)

    def _invalidate_active_inference_runner(self):
        """Prevent a prior project or model inference from committing later."""
        token = getattr(self, "_active_inference_token", None)
        if token is not None:
            token.invalidate()
        runner = getattr(self, "_active_inference_runner", None)
        if runner is not None:
            runner.cancel()
        self._active_inference_token = None
        self._active_inference_runner = None
        self._pending_segment_display = False

    def _release_finished_inference_runner(self):
        """Release a naturally completed runner after its terminal state is visible."""
        runner = getattr(self, "_active_inference_runner", None)
        if runner is None or bool(runner.is_running()):
            return False
        self._active_inference_runner = None
        self._active_inference_token = None
        return True

    def _finish_cancelled_background_worker(self):
        if getattr(self.state, 'mask_processing_running', False):
            self.state.mask_processing_running = False
            self.state.mask_processing_progress = 0.0
            self.task_panel.set_task_status("process", "available", text="已取消")
        if getattr(self.state, 'plan_running', False):
            self.state.plan_running = False
            self.state.plan_progress = 0.0
            self.task_panel.set_task_status("plan", "available", text="已取消")
        self.progress_bar.hide()
        if hasattr(self, 'btn_cancel_task'):
            self.btn_cancel_task.hide()
            self.btn_cancel_task.setEnabled(True)
        self.top_toolbar.set_progress(0)
        self.top_toolbar.set_worker_cancel_visible(False)
        self.set_status_text("已取消，可重试", "idle")
        self.log_panel.info("当前任务已取消，可重新执行")
        self._sync_task_statuses()
        self.top_toolbar.refresh_actions(self.state)

    @Slot()
    def _clear_background_worker_refs(self):
        was_cancelled = bool(getattr(self, '_task_cancel_requested', False))
        finished_thread = self._worker_thread
        self._worker_thread = None
        self._worker = None
        self._task_cancel_requested = False
        if hasattr(self, 'btn_cancel_task'):
            self.btn_cancel_task.hide()
            self.btn_cancel_task.setEnabled(True)
        if was_cancelled:
            self._finish_cancelled_background_worker()
        if getattr(self, '_close_after_worker_cancel', False):
            self._close_after_worker_cancel = False
            self._close_cancel_started_at = 0.0
            self._close_cancel_warning_shown = False
            QTimer.singleShot(0, self.close)
        # QThread(self) 的 parent 是主窗口：只把 Python 引用置空，Qt 对象仍留在
        # 主窗口的 children 列表里，每跑一次后台任务就泄漏一个线程对象（连同
        # 线程栈）。worker 本身已经在 finished/error 时 deleteLater，这里补上
        # 线程自己的。此槽由 thread.finished 触发，线程已结束，排队删除安全。
        if finished_thread is not None:
            finished_thread.deleteLater()

    def _warn_close_cancel_still_running(self):
        if (
            not getattr(self, '_close_after_worker_cancel', False)
            or not self._worker_thread
            or not self._worker_thread.isRunning()
            or getattr(self, '_close_cancel_warning_shown', False)
        ):
            return
        self._close_cancel_warning_shown = True
        message = "后台任务仍在取消中，请等待当前计算阶段安全退出后自动关闭。"
        self.set_status_text(message, "busy")
        self.log_panel.info(message)
        self._log.info(message)

    def _classify_task_error(self, msg):
        """Convert technical exceptions into user-actionable failure categories."""
        text = str(msg or "")
        lower = text.lower()
        rules = [
            ("掩膜为空", ["mask is none", "empty mask", "mask_raw", "掩膜为空", "空掩膜", "没有掩膜", "缺少处理后的掩膜"],
             "当前没有可用于处理的 AI 掩膜。",
             "请重新执行 作物识别；如果识别结果仍为空，请检查模型是否匹配当前影像。"),
            ("种植带重构失败", ["种植带重构失败", "metric mask regularization failed", "无法建立田块支持区域", "仅检测到"],
             "当前 AI 掩膜没有形成足够连续、可分离的种植带几何，系统已停止在路径规划之前。",
             "请检查田块边界和模型训练来源，重新选择或训练与当前生育期、航高和地块匹配的模型后重新执行 作物识别；不要用放宽阈值或旧掩膜强行生成路径。"),
            ("没有有效行带", ["未能提取有效行带", "没有有效行带", "未生成有效作业线", "wide_bands", "work_lines", "no valid band", "no bands", "路径点不足"],
             "系统没有从掩膜中提取到足够的水稻行带。",
             "请重新圈选田块，或重新执行 作物识别与掩膜优化；必要时降低最小行带长度/面积阈值。"),
            ("起点缺失", ["entry_point", "起点", "缺少起点"],
             "路径规划缺少已确认的起点。",
             "请确认系统推荐起点，或在影像上手动布设起点。终点会沿最终作业段自动延伸，卸粮点按需添加。"),
            ("农机参数不完整", ["harvester", "cutter_width", "turn_radius_m", "农机参数", "割台", "履带", "参数不完整"],
             "农机几何参数不完整，无法判断作物掩膜几何覆盖和履带足迹。",
             "请打开“农机参数”，检查有效割幅、割刀前置距离、履带尺寸和最小转弯半径。"),
            ("转弯半径无法满足", ["2r", "turn radius", "min_turn_radius", "转弯半径", "无法相切", "半径", "行距ω小于2r"],
             "当前行距或田头空间无法满足所选调头方式的最小转弯半径。",
             "请尝试切换为“自动推荐/鱼尾折返/紧凑 Alpha”，或调整农机最小转弯半径后重算。"),
            ("路径越界严重", ["outside", "越界", "track_outside", "超出田块", "field boundary", "边界外", "履带越界率"],
             "生成路径存在明显越界或履带超出田块边界。",
             "请检查田块边界是否闭合准确，并适当增加田头缓冲或重新规划。"),
        ]
        for title, keywords, what, action in rules:
            if any(key.lower() in lower for key in keywords):
                return {"title": title, "what": what, "action": action, "raw": text}
        return {
            "title": "任务失败",
            "what": "处理过程中出现未分类错误。",
            "action": "请查看日志中的详细异常；若反复出现，请保存当前项目状态后重新执行上一步。",
            "raw": text,
        }

    def _task_error_message(self, msg):
        info = self._classify_task_error(msg)
        return info["title"], (
            f"{info['what']}\n\n建议：{info['action']}\n\n"
            "详细异常已写入项目日志。"
        )

    @Slot(str)
    def _on_worker_error(self, msg):
        if getattr(self, '_task_cancel_requested', False):
            if self._worker_thread:
                self._worker_thread.quit()
            return
        if getattr(self, "_model_load_in_progress", False):
            # ModelLoadWorker normally reports failures through its terminal
            # envelope.  Keep this fallback so an unexpected Qt/engine error
            # cannot leave the model selector permanently disabled.
            self._on_model_load_done({
                "task_token": getattr(self, "_active_model_load_token", None),
                "status": "failed",
                "result": None,
                "error": str(msg or "未知错误"),
            })
            if self._worker_thread:
                self._worker_thread.quit()
            return
        if getattr(self.state, 'mask_processing_running', False):
            self.state.mask_processing_running = False
            self.state.mask_processing_progress = 0.0
            self.task_panel.set_task_status("process", "failed", text="失败")
        if getattr(self.state, 'plan_running', False):
            self.state.plan_running = False
            self.state.plan_progress = 0.0
            self.task_panel.set_task_status("plan", "failed", text="失败")
        self.progress_bar.hide()
        if hasattr(self, 'btn_cancel_task'):
            self.btn_cancel_task.hide()
        self.top_toolbar.set_progress(0)
        self.top_toolbar.set_worker_cancel_visible(False)
        title, message = self._task_error_message(msg)
        self.set_status_text(title, "error")
        self.log_panel.error(f"{title}: {msg[:300]}")
        if self._worker_thread:
            self._worker_thread.quit()
        QMessageBox.warning(self, title, message)

    @Slot(dict)
    def _on_mask_worker_done(self, payload):
        processed = self._unwrap_task_result("mask", payload)
        if processed is None:
            self._finish_failed_task_completion(
                "mask", payload, "掩膜优化", "process",
            )
            self._clear_task_token_if_current("mask", payload)
            return
        self.state.mask_processing_running = False
        self.state.mask_processing_progress = 1.0
        self.progress_bar.hide()
        if hasattr(self, 'btn_cancel_task'):
            self.btn_cancel_task.hide()
        self.top_toolbar.set_progress(0)
        self.top_toolbar.set_worker_cancel_visible(False)
        self._apply_mask_process_result(processed)
        self._active_mask_token = None

    @Slot(dict)
    def _on_centerline_worker_done(self, payload):
        result = self._unwrap_task_result("centerline", payload)
        if result is None:
            if isinstance(payload, dict) and payload.get("status") == "failed":
                self.task_panel.set_task_status(
                    "centerline_generate", "failed", text="失败"
                )
                self.log_panel.error(
                    f"作业线生成失败: {str(payload.get('error') or '未知错误')}"
                )
            self._clear_task_token_if_current("centerline", payload)
            return
        catalog = dict(
            result.get("catalog")
            or (result.get("layout") or {}).get("centerline_catalog")
            or {}
        )
        if not catalog.get("items"):
            self.task_panel.clear_centerline_selection()
            self.task_panel.set_task_status(
                "centerline_generate", "failed", text="无有效中心线"
            )
            layout = dict(result.get("layout") or {})
            geometry = dict(layout.get("field_boundary_geometry") or {})
            if geometry.get("repaired"):
                self.log_panel.warn(
                    "田块边界闭合点重复，已自动修复后继续生成作业线"
                )
            for reason in layout.get("centerline_hard_reasons") or []:
                self.log_panel.warn(str(reason))
            self.log_panel.warn("当前没有可供选择的有效中心线")
            self._clear_task_token_if_current("centerline", payload)
            return
        selected_ids = list(
            getattr(self.state, "selected_centerline_ids", None)
            or getattr(self.state, "selected_work_band_ids", [])
            or []
        )
        valid_ids = {str(item.get("id")) for item in catalog.get("items") or []}
        selected_ids = [value for value in selected_ids if str(value) in valid_ids]
        if not selected_ids:
            selected_ids = [str(item.get("id")) for item in catalog["items"]]
            origin = "default_all"
        else:
            origin = "operator"
        report = {
            "algorithm_version": "operator-centerline-selection-v1",
            "selection_origin": origin,
            "selection_confirmed": False,
            "selected_ids": list(selected_ids),
            "selected_count": len(selected_ids),
            "selection_mode": "single" if len(selected_ids) == 1 else "multi",
            "candidate_count": len(catalog.get("items") or []),
        }
        self._save_system_undo("生成作业线")
        self.state.safe_update(
            centerline_catalog=catalog,
            selected_centerline_ids=list(selected_ids),
            centerline_selection_report=dict(report),
            work_band_catalog=catalog,
            selected_work_band_ids=list(selected_ids),
            work_band_selection_report=dict(report),
        )
        self.task_panel.update_centerline_selection(catalog, selected_ids)
        self._restore_visual_from_state()
        self._sync_task_statuses()
        from ui_messages import centerlines_done

        self.log_panel.success(centerlines_done(len(catalog.get("items"))))
        strip = getattr(self, "status_strip", None)
        if strip is not None:
            strip.set_step("第 5 步 · 生成作业线")
            strip.set_phase("已生成", "ready")
            strip.set_hint("下一步：勾选需要的作业线")
            strip.set_metrics({"作业线": f"{len(catalog.get('items'))} 条"})
        geometry = dict((result.get("layout") or {}).get("field_boundary_geometry") or {})
        if geometry.get("repaired"):
            self.log_panel.info(
                "田块边界闭合点重复，已按圈选轮廓自动修复"
                f"（丢弃面积 {float(geometry.get('dropped_area_ratio', 0.0)) * 100:.4f}%）"
            )
        self._persist_centerline_selection()
        self._clear_task_token_if_current("centerline", payload)

    def _persist_centerline_selection(self):
        tif_path = self._tif_path or getattr(self.state, "tif_path", "")
        if not tif_path:
            return
        try:
            from cache import request_project_state_save
            request_project_state_save(tif_path, self.state, stage="centerline_selection")
        except Exception as exc:
            self._log.info(f"centerline selection cache save failed: {exc}")

    def _schedule_centerline_selection_persist(self, delay_ms=400):
        """合并连续的勾选/取消，只落一次盘。

        每点一次复选框就同步写一次 project_state.json 会把 UI 线程拖住；
        勾选是高频操作，用单次定时器合并成一次保存。
        """
        timer = getattr(self, "_centerline_persist_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._persist_centerline_selection)
            self._centerline_persist_timer = timer
        timer.start(int(delay_ms))

    def _confirm_centerline_selection(self):
        """采纳当前勾选：把「选择作业线」标成已完成并推进到下一步。

        生成作业线后系统默认全选，但那是建议不是确认；用户要能停在卡片上
        逐条看，确认后才进入起点设置。
        """
        catalog = getattr(self.state, "centerline_catalog", None) or getattr(
            self.state, "work_band_catalog", {}
        ) or {}
        items = list(catalog.get("items") or [])
        valid_ids = [str(item.get("id")) for item in items]
        selected = [
            str(value)
            for value in (
                getattr(self.state, "selected_centerline_ids", None)
                or getattr(self.state, "selected_work_band_ids", [])
                or []
            )
            if str(value) in valid_ids
        ]
        if not selected:
            self.log_panel.warn("当前未勾选任何作业线；请至少保留一条再确认")
            return
        report = {
            "algorithm_version": "operator-centerline-selection-v1",
            "selection_origin": "operator",
            "selection_confirmed": True,
            "selected_ids": list(selected),
            "selected_count": len(selected),
            "candidate_count": len(valid_ids),
            "selection_mode": "single" if len(selected) == 1 else "multi",
        }
        # 快照要在改动之前抓，撤销才能回到「未确认」。
        self.state.save_centerline_selection_undo("确认作业线选择")
        self.state.safe_update(
            centerline_selection_report=dict(report),
            work_band_selection_report=dict(report),
        )
        self.top_toolbar.refresh_actions(self.state)
        self._persist_centerline_selection()
        self._sync_task_statuses()
        from ui_messages import selection_confirmed

        self.log_panel.success(selection_confirmed(len(selected), len(valid_ids)))
        strip = getattr(self, "status_strip", None)
        if strip is not None:
            strip.set_step("第 6 步 · 选择作业线")
            strip.set_phase("已确认", "ready")
            strip.set_hint("下一步：设置起点")
            strip.set_metrics({"已选": f"{len(selected)} 条"})

    def _start_centerline_catalog_generation(self):
        """Generate all candidate centerlines once the processed mask is committed."""
        mask_result = getattr(self.state, "mask_result", None)
        if not isinstance(mask_result, dict) or not isinstance(
            mask_result.get("processed_mask"), np.ndarray
        ):
            return False
        catalog = getattr(self.state, "centerline_catalog", None) or getattr(
            self.state, "work_band_catalog", {}
        ) or {}
        if catalog.get("items"):
            self.task_panel.update_centerline_selection(
                catalog,
                getattr(self.state, "selected_centerline_ids", None)
                or getattr(self.state, "selected_work_band_ids", []),
            )
            self._sync_task_statuses()
            return True
        if self._has_background_worker():
            QTimer.singleShot(100, self._start_centerline_catalog_generation)
            return False
        self._active_centerline_token = make_task_token(self.state, "centerline")
        runtime = self._desktop_path_planning_runtime_config(
            self._effective_harvester_params(
                getattr(self.state, "harvester_params", {}) or {}
            )
        )
        runtime.pop("selected_centerline_ids", None)
        runtime.pop("selected_work_band_ids", None)
        worker = CenterlineCatalogWorker(
            mask_result,
            self.geo,
            self.state,
            runtime,
            token=self._active_centerline_token,
        )
        worker.finished.connect(self._on_centerline_worker_done)
        if not self._start_background_worker(
            worker,
            "正在生成全部候选中心线，完成后可在左侧逐线勾选",
            busy_message="当前已有耗时任务正在执行，请等待完成后再生成作业线。",
        ):
            self._active_centerline_token = None
            return False
        return True

    def _on_centerline_selection_changed(self, centerline_id, checked):
        catalog = getattr(self.state, "centerline_catalog", None) or getattr(
            self.state, "work_band_catalog", {}
        ) or {}
        items = list(catalog.get("items") or [])
        valid_ids = [str(item.get("id")) for item in items]
        selected = [
            str(value)
            for value in (
                getattr(self.state, "selected_centerline_ids", None)
                or getattr(self.state, "selected_work_band_ids", [])
                or []
            )
            if str(value) in valid_ids
        ]
        centerline_id = str(centerline_id)
        if checked and centerline_id not in selected:
            selected.append(centerline_id)
        elif not checked:
            selected = [value for value in selected if value != centerline_id]
        order = {value: index for index, value in enumerate(valid_ids)}
        selected.sort(key=lambda value: order.get(value, 0))
        previous_report = getattr(self.state, "centerline_selection_report", {}) or {}
        previous_origin = str(previous_report.get("selection_origin") or "")
        # 勾选/取消只是「正在调整」：已经确认过的选择不要因为再动一下就退回待确认。
        still_confirmed = bool(
            previous_report.get(
                "selection_confirmed", previous_origin != "default_all"
            )
        )
        report = {
            "algorithm_version": "operator-centerline-selection-v1",
            "selection_origin": "operator",
            "selection_confirmed": still_confirmed,
            "selected_ids": list(selected),
            "selected_count": len(selected),
            "candidate_count": len(valid_ids),
            "selection_mode": (
                "empty" if not selected
                else "single" if len(selected) == 1
                else "multi"
            ),
        }
        # 勾选框的撤销只回退「作业线选择」，不抓系统快照：系统快照包含
        # mask_raw / mask_result 等大数组，每点一次复选框就深拷贝一次，
        # 14 条线来回勾选会明显卡顿。
        self.state.save_centerline_selection_undo("调整作业线选择")
        self.top_toolbar.refresh_actions(self.state)
        self.state.safe_update(
            centerline_catalog=catalog,
            selected_centerline_ids=list(selected),
            centerline_selection_report=dict(report),
            work_band_catalog=catalog,
            selected_work_band_ids=list(selected),
            work_band_selection_report=dict(report),
        )
        self._invalidate_analysis_from(
            "path", "作业线选择已变化，未勾选作业线隐藏且不进入掉头连接"
        )
        # A stale route preview may contain the line just unchecked.  The
        # checkbox contract is an immediate visual exclusion, so do not keep
        # that obsolete path as a dashed overlay after the selection changes.
        self._stale_path_preview = None
        self.task_panel.sync_centerline_selection(selected)
        # DECISION-024: the checkbox is a display-level edit.  The formal
        # route remains invalidated above, but only the centerline and route
        # layers need to change on the canvas; the base image and mask
        # evidence stay in place for immediate feedback.
        self._refresh_centerline_selection_overlay()
        self._schedule_centerline_selection_persist()
        if not selected:
            self.log_panel.warn("当前未勾选任何作业线；请至少保留一条后再生成路线")
        # 勾选是高频操作：每点一次都往日志里写一行会把日志刷成流水账，
        # 当前勾选数量已经由面板上的「已选择 N/M 条」实时显示。

    def _admit_task_completion(self, task_type, token):
        """Reject cancelled or stale worker output before it mutates project state."""
        active_token = getattr(self, f"_active_{task_type}_token", None)
        if active_token is not None and token is not active_token:
            return False
        if token is not None:
            return admit_task_result(token, self.state)
        generation = getattr(self, f"_active_{task_type}_generation", None)
        if generation is not None:
            return int(generation) == int(getattr(self.state, "project_generation", 0) or 0)
        return not bool(getattr(self, "_task_cancel_requested", False))

    def _unwrap_task_result(self, task_type, payload):
        """Return an admitted completed worker result, otherwise fail closed."""
        if not isinstance(payload, dict):
            return None
        active_token = getattr(self, f"_active_{task_type}_token", None)
        is_envelope = "status" in payload or "task_token" in payload
        if not is_envelope:
            if active_token is not None or not self._admit_task_completion(task_type, None):
                return None
            return payload
        token = payload.get("task_token")
        if token is not active_token or payload.get("status") != "completed":
            return None
        if not self._admit_task_completion(task_type, token):
            return None
        result = payload.get("result")
        return result if isinstance(result, dict) else None

    def _clear_task_token_if_current(self, task_type, payload):
        """Release a terminal token only when the completing worker owns it."""
        if not isinstance(payload, dict):
            return
        token = payload.get("task_token")
        name = f"_active_{task_type}_token"
        if token is not None and token is getattr(self, name, None):
            setattr(self, name, None)

    def _finish_failed_task_completion(self, task_type, payload, label, task_id):
        """DECISION-002/007: clear only the active task after an owned failure."""
        if not isinstance(payload, dict) or payload.get("status") != "failed":
            return False
        token = payload.get("task_token")
        if token is not getattr(self, f"_active_{task_type}_token", None):
            return False
        if not self._admit_task_completion(task_type, token):
            return False
        if task_type == "mask":
            self.state.mask_processing_running = False
            self.state.mask_processing_progress = 0.0
        elif task_type == "plan":
            self.state.plan_running = False
            self.state.plan_progress = 0.0
        self.progress_bar.hide()
        if hasattr(self, "btn_cancel_task"):
            self.btn_cancel_task.hide()
        self.top_toolbar.set_progress(0)
        self.top_toolbar.set_worker_cancel_visible(False)
        self.task_panel.set_task_status(task_id, "failed", text="失败")
        self.lbl_status.setText(f"{label}失败")
        self.log_panel.error(f"{label}失败: {str(payload.get('error') or '未知错误')}")
        self._clear_task_token_if_current(task_type, payload)
        return True

    def _apply_centerline_layout(self, layout):
        """Commit the planner's centerline catalog and checkbox provenance."""
        layout = dict(layout or {})
        catalog = layout.get("centerline_catalog")
        if not isinstance(catalog, dict) or not catalog.get("items"):
            catalog = layout.get("work_band_catalog")
        selection = layout.get("centerline_selection")
        if not isinstance(selection, dict):
            selection = layout.get("work_band_selection")
        if not isinstance(catalog, dict):
            return False
        selection = dict(selection or {})
        selected = list(selection.get("selected_ids") or [])
        report = dict(selection)
        self.state.safe_update(
            centerline_catalog=dict(catalog),
            selected_centerline_ids=list(selected),
            centerline_selection_report=report,
            work_band_catalog=dict(catalog),
            selected_work_band_ids=list(selected),
            work_band_selection_report=dict(report),
        )
        self.task_panel.update_centerline_selection(catalog, selected)
        return True

    @Slot(dict)
    def _on_plan_worker_done(self, payload):
        result = self._unwrap_task_result("plan", payload)
        if result is None:
            self._finish_failed_task_completion(
                "plan", payload, "路径规划", "plan",
            )
            self._clear_task_token_if_current("plan", payload)
            return
        self.state.plan_running = False
        self.state.plan_progress = 1.0
        path_payload = result.get("path") or {}
        layout_payload = path_payload.get("layout") or {}
        self._log_action(
            "plan.done", "path",
            segments=len(path_payload.get("full_path") or []),
            valid=bool(path_payload.get("is_valid")),
            operator_applied=len(
                (layout_payload.get("operator_centerline_application") or {}).get("applied") or []
            ),
            strategy=str(path_payload.get("requested_strategy") or ""),
        )
        self.progress_bar.hide()
        if hasattr(self, 'btn_cancel_task'):
            self.btn_cancel_task.hide()
        self.top_toolbar.set_progress(0)
        self.top_toolbar.set_worker_cancel_visible(False)
        path_result = self._store_path_result(result.get("path", {}), result)
        path_layout = path_result.get("layout") or {}
        self._apply_centerline_layout(path_layout)
        self._service_points_visible = True
        # The requested service points remain editable provenance inputs.  The
        # route planner uses them as start/end references; it does not replace
        # the visible markers with canonical route vertices.
        self._refresh_service_points_panel()
        if (
            not path_result.get("full_path")
            and isinstance(getattr(self, "_stale_path_preview", None), dict)
        ):
            # A failed formal replan must not turn the canvas into an empty
            # image; keep the explicitly labelled, non-exportable old route
            # visible until the user changes inputs or a later replan succeeds.
            self._show_path(self._stale_path_preview, stale_preview=True)
        else:
            self._show_path(path_result)
        self.route_info.update_route(path_result, path_result.get("validation", {}))
        WorkflowUpdater.advance(self.state, "PATH_PLANNED")
        self._sync_task_statuses()
        bands_count = len(result.get("bands", []))
        tracks = len(path_result.get("tracks", []))
        turns = len(path_result.get("formal_turns", path_result.get("turns", [])))
        self.log_panel.success(self._path_quality_summary(path_result, bands_count, tracks, turns))
        self.log_panel.detail(self._path_quality_details(path_result))
        strip = getattr(self, "status_strip", None)
        if strip is not None:
            strip.set_step("第 7 步 · 路径规划")
            strip.set_phase("已生成", "ready")
            strip.set_hint("")
            strip.set_metrics({"作业段": f"{tracks} 段", "转弯": f"{turns} 处"})
        turn_assessment = path_result.get("turn_assessment") or {}
        service_points = turn_assessment.get("service_points") or {}
        if service_points.get("entry_reference_used"):
            distance_m = service_points.get("entry_actual_distance_m")
            distance_text = (
                f"，最终首端点距起点标记 {float(distance_m):.1f}m"
                if isinstance(distance_m, (int, float))
                and math.isfinite(float(distance_m))
                else ""
            )
            if service_points.get("entry_nearest_feasible_verified") is not True:
                self.log_panel.warn(
                    "起点参与了规划，但最终路线未通过‘全田最近可行首端点’复核；"
                    "当前路线仅供检查，不能把起点约束视为已满足。"
                )
            elif service_points.get(
                "entry_side_constrained_by_terminal_short_line"
            ):
                self.log_panel.info(
                    "起点已在‘截短作业线安全收尾’约束下选择全田最近可行首端点"
                    f"{distance_text}；收尾线被田界阻断，因此进入侧受限。"
                    "当前为参考点模式，不会把标记点强行拉接进作业线。"
                )
            else:
                self.log_panel.info(
                    "起点已选择全田最终作业线中的最近首端点和进入方向"
                    f"{distance_text}；可选终点仅在已确认时参与末段方向选择。"
                    "当前为参考点模式，不会把标记点强行拉接进作业线。"
                )
        self._log_turn_strategy_result(path_result)
        self._journal.info(f"路径规划完成: {bands_count}条带 {tracks}条作业线 {turns}条转弯线")
        validation = path_result.get("validation", {}) or {}
        if validation:
            self._show_validation_report(path_result, validation)
            QTimer.singleShot(0, lambda path=path_result: self._warn_path_validation_issues(path))
        self._active_plan_token = None
        self._active_plan_generation = None

    def _set_theme_mode(self, mode):
        self._theme_mgr.set_mode(mode)
        labels = {
            ThemeMode.DARK: "深色",
            ThemeMode.LIGHT: "浅色",
            ThemeMode.SYSTEM: "跟随系统",
        }
        self.log_panel.info(f"主题: {labels.get(mode, mode)}")

    def _toggle_theme(self):
        # The toolbar button is a direct dark/light toggle. SYSTEM remains an
        # explicit menu choice; cycling through it can look like a no-op when
        # the system theme resolves to the same colour scheme.
        target = (
            ThemeMode.LIGHT
            if self._theme_mgr.current_theme() == "dark"
            else ThemeMode.DARK
        )
        self._set_theme_mode(target)
    def _builtin_model_dir(self):
        return str(builtin_model_dir())

    def _model_registry(self, model_dir=None):
        registry_path = os.path.join(
            model_dir or self._builtin_model_dir(), "model_registry.json"
        )
        try:
            with open(registry_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return {
                str(item.get("name", "")): dict(item)
                for item in payload.get("models", [])
                if isinstance(item, dict) and item.get("name")
            }
        except (OSError, ValueError, TypeError):
            return {}

    def _model_registry_entry(self, model_path):
        if not model_path:
            return {}
        return self._model_registry().get(os.path.basename(str(model_path)), {})

    def _builtin_model_list(self, model_dir=None):
        from builtin_test_assets import builtin_test_models

        model_dir = model_dir or self._builtin_model_dir()
        registry = self._model_registry(model_dir)
        models = []
        for model in builtin_test_models(existing_only=True):
            item = dict(model)
            registry_item = registry.get(os.path.basename(item["path"]), {})
            if registry_item.get("sha256"):
                item["sha256"] = str(registry_item["sha256"])
            models.append(item)
        return models

    @staticmethod
    def _is_trusted_builtin_model_path(model_path):
        from builtin_test_assets import builtin_test_model_entry

        return bool(builtin_test_model_entry(model_path))

    def _refresh_builtin_models(self):
        merged = []
        seen = set()
        for model in self._builtin_model_list() + self._custom_model_list():
            path = model.get("path", "")
            if not path or path in seen:
                continue
            seen.add(path)
            merged.append({"name": model.get("name") or self._model_display_name(model), "path": path})
        self._model_list = merged
        names = [m["name"] for m in self._model_list]
        self.top_toolbar.populate_models(names)
        self.task_panel.set_model_options(self._model_list, self.state.current_model_path)
        return self._model_list

    def _ensure_model_loaded(self):
        if self.model_engine.is_loaded():
            return True
        if getattr(self, "_model_load_in_progress", False):
            return False
        models = self._model_list or self._refresh_builtin_models()
        if not models:
            self.log_panel.error("未找到内置模型，请检查 data/models")
            return False
        # Keep the old convenience entry point (作物识别 auto-selects the
        # first built-in model), but defer the actual action until the model
        # worker commits on the GUI thread.
        self._pending_model_action = "segment"
        started = self._on_load_model_at(models[0]["path"])
        if not started:
            self._pending_model_action = None
        return False

    def _auto_start(self):
        data_models = self._builtin_model_dir()

        self._model_list = self._builtin_model_list(data_models)
        if self._model_list:
            names = [m.get("name", os.path.basename(m.get("path", ""))) for m in self._model_list]
            self.top_toolbar.populate_models(names)
            self.task_panel.populate_models(names)
            self._log.info(f"loaded {len(self._model_list)} built-in models")

        if self._tif_path:
            return

        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            self._load_tif_path(sys.argv[1])
            return

        dlg = LauncherDialog(self)
        if dlg.exec() == QDialog.Accepted:
            path = getattr(dlg, "selected_path", "")
            if path:
                QTimer.singleShot(100, lambda: self._load_tif_path(path))
    def _load_tif_path(self, path):
        """Load a GeoTIFF image and initialise display/georeference state."""
        self._log_action("project.open_tif", os.path.basename(os.fspath(path)))
        if self._has_background_worker():
            self.log_panel.warn("任务正在运行：当前已有耗时任务正在执行，请等待完成后再打开影像。")
            return
        path = os.fspath(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        self._invalidate_active_inference_runner()
        # Clear the previous image's transform immediately. A CRS-less image
        # must never inherit geographic coordinates from the prior project.
        self.geo.clear()
        self._tif_source_identity = {}
        worker = TifLoadWorker(path)
        worker.finished.connect(self._on_tif_load_done)
        self.lbl_status.setText("正在加载影像...")
        self.top_toolbar.set_progress(5, "影像加载中...")
        self.progress_bar.show()
        self.progress_bar.setValue(5)
        if not self._start_background_worker(worker, "影像加载已开始，请稍候..."):
            self.progress_bar.hide()
            self.top_toolbar.set_progress(0)

    @Slot(dict)
    def _on_tif_load_done(self, result):
        # A cancellation is still a terminal worker result: it exists to let
        # the worker QThread stop, not to mutate the current project.
        if (
            result.get("status") == "cancelled"
            or getattr(self, "_task_cancel_requested", False)
        ):
            return
        try:
            # Keep the full source identity in visible logs and cache state;
            # the basename is reserved for compact window labels only.
            path = os.fspath(result["path"])
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            log_path = os.path.abspath(path)
            rgb = result["rgb"]
            render_rgb = result.get("render_rgb", rgb)
            full_h = int(result["full_h"])
            full_w = int(result["full_w"])
            out_h = int(result["display_h"])
            out_w = int(result["display_w"])
            downsample = int(result["downsample"])
            transformer = result.get("transformer")
            self.geo.clear()
            if transformer is not None and result.get("transform") is not None:
                try:
                    self.geo.set_affine(
                        result["transform"],
                        transformer,
                        source_crs=result.get("crs"),
                    )
                    extent = self.geo.validate_raster_extent(out_w, out_h)
                    self._log.info(
                        f"georeference validated (source CRS: {result.get('crs')}, "
                        "output CRS: EPSG:4326, "
                        f"extent: {extent['min_lon']:.8f},{extent['min_lat']:.8f} - "
                        f"{extent['max_lon']:.8f},{extent['max_lat']:.8f})"
                    )
                except Exception as exc:
                    self.geo.clear()
                    self._log.info(
                        "georeference validation failed; geographic export is disabled: "
                        f"{exc}"
                    )
            else:
                self._log.info(
                    "georeference unavailable; geographic export is disabled: "
                    f"{result.get('geo_error') or 'missing CRS'}"
                )
            self._log.info(f"read image: 3 bands, {out_w}x{out_h}")
            reset_project_state(self.state)
            self._pipeline_result = None
            self._stale_path_preview = None
            self._clear_machine_route_preview()
            self._service_points_visible = False
            self._tif_rgb = rgb
            self._tif_path = path
            from cache import source_identity
            self._tif_source_identity = source_identity(path)
            self.state.safe_update(
                tif_path=path,
                source_img_h=full_h,
                source_img_w=full_w,
                img_h=out_h,
                img_w=out_w,
                display_h=out_h,
                display_w=out_w,
                downsample_factor=downsample,
                display_scale_x=float(full_w) / float(out_w),
                display_scale_y=float(full_h) / float(out_h),
                source_sha256=str(result.get("source_sha256", "") or ""),
                source_metadata=dict(result.get("source_metadata") or {}),
                raster_preprocessing=dict(result.get("preview_preprocessing") or {}),
            )
            self.top_toolbar.set_filename(path)
            self.image_view.set_image(render_rgb, logical_shape=rgb.shape[:2])
            try:
                if self.geo.is_ready():
                    cx, cy = full_w // 2, full_h // 2
                    ppm = self.geo.pixels_per_meter(cx // max(1, downsample), cy // max(1, downsample))
                    self._log.info(f"scale ppm={ppm:.4f} at center ({cx},{cy})")
                    self.image_view.set_ppm(ppm)
            except Exception as e:
                self._log.info(f"scale calculation failed: {e}")
            basename = os.path.basename(path)
            self.log_panel.success(f"影像加载: {log_path} ({full_w}x{full_h})")
            metadata = dict(result.get("source_metadata") or {})
            if self.geo.is_ready():
                try:
                    center_lon, center_lat = self.geo.pixel_to_lonlat(out_w / 2.0, out_h / 2.0)
                    gsd_x = self.geo.pixel_distance_m((out_w / 2.0, out_h / 2.0), (out_w / 2.0 + 1.0, out_h / 2.0))
                    gsd_y = self.geo.pixel_distance_m((out_w / 2.0, out_h / 2.0), (out_w / 2.0, out_h / 2.0 + 1.0))
                    self.log_panel.info(
                        f"坐标摘要: {result.get('crs') or '无 CRS'} → EPSG:4326 | "
                        f"中心 {center_lon:.7f}, {center_lat:.7f} | "
                        f"预览GSD {gsd_x:.3f}/{gsd_y:.3f} m/px"
                    )
                except Exception as exc:
                    self.log_panel.warn(f"坐标摘要计算失败: {exc}")
            if int(metadata.get("band_count", 0) or 0) > 3 and metadata.get("rgb_selection_assumed"):
                indexes = metadata.get("rgb_band_indexes") or [1, 2, 3]
                self.log_panel.warn(
                    "请确认 RGB 波段：该多波段影像没有明确声明 RGB 色彩解释，"
                    f"当前使用波段 {indexes} 作为 RGB，请确认与模型训练输入一致"
                )
            self.lbl_status.setText(f"已加载 {basename}")
            self._journal.info(f"影像加载: {log_path}")
            self.task_panel.set_task_status("import_img", "done")
            self.task_panel.set_task_action("import_img", "重选")
            self._sync_task_statuses()
            self._queue_cache_restore_after_current_worker(path)
        finally:
            self.progress_bar.hide()
            self.top_toolbar.set_progress(0)
            if self._worker_thread:
                self._worker_thread.quit()

    def _queue_cache_restore_after_current_worker(self, path):
        """Start cache restore after the current load worker has fully released the worker slot."""
        thread = self._worker_thread
        if thread and thread.isRunning():
            thread.finished.connect(
                lambda p=path: QTimer.singleShot(0, lambda: self._start_cache_restore(p))
            )
            return
        QTimer.singleShot(0, lambda p=path: self._start_cache_restore(p))

    def _start_cache_restore(self, path):
        token = make_task_token(self.state, "cache_restore")
        self._active_cache_restore_token = token
        worker = CacheRestoreWorker(
            path,
            expected_source_sha256=str(getattr(self.state, "source_sha256", "") or ""),
            token=token,
        )
        worker.finished.connect(self._on_cache_restore_done)
        if not self._start_background_worker(
            worker,
            "正在后台恢复项目缓存...",
            warn_if_busy=False,
        ):
            self._log.info("cache restore skipped: background worker busy")
            token.invalidate()
            self._clear_task_token_if_current(
                "cache_restore", {"task_token": token},
            )
            return

    @staticmethod
    def _cache_model_identity_updates(saved):
        """Keep cache model metadata historical until that weight is loaded again."""
        return {
            "historical_model_name": str(saved.get("current_model_name", "") or ""),
            "historical_model_path": str(saved.get("current_model_path", "") or ""),
            "historical_model_sha256": str(saved.get("model_sha256", "") or ""),
        }

    @staticmethod
    def _cache_model_matches_active_engine(state, engine):
        """DECISION-002/006: cache provenance only matches the loaded weight bytes."""
        historical_path = str(getattr(state, "historical_model_path", "") or "")
        historical_sha256 = str(getattr(state, "historical_model_sha256", "") or "")
        if not historical_path and not historical_sha256:
            return True
        if not historical_path or not historical_sha256:
            return False
        if engine is None or not bool(getattr(engine, "is_loaded", lambda: False)()):
            return False
        engine_path = str(getattr(engine, "_model_path", "") or "")
        active_path = str(getattr(state, "current_model_path", "") or "")
        active_sha256 = str(getattr(state, "model_sha256", "") or "")
        try:
            from provenance import file_sha256

            engine_sha256 = file_sha256(engine_path)
        except (OSError, ValueError):
            return False
        try:
            same_path = (
                os.path.normcase(os.path.abspath(historical_path))
                == os.path.normcase(os.path.abspath(engine_path))
                == os.path.normcase(os.path.abspath(active_path))
            )
        except (OSError, ValueError):
            return False
        return bool(
            same_path
            and historical_sha256 == engine_sha256
            and active_sha256 == engine_sha256
        )

    def _sync_cache_model_identity(self):
        """Record whether restored historical cache evidence matches the active engine."""
        matches = self._cache_model_matches_active_engine(self.state, self.model_engine)
        self.state.cache_model_matches_active_engine = matches
        return matches

    def _schedule_history_checkpoint(self):
        """Restart the disk checkpoint delay for the current logical action."""
        timer = getattr(self, "_history_checkpoint_timer", None)
        if timer is not None:
            timer.start()

    def _persist_completed_history_checkpoint(self):
        """Persist latest state plus ordered operation history off the UI thread."""
        tif_path = self._tif_path or getattr(self.state, "tif_path", "")
        if not tif_path:
            return
        try:
            from cache import request_project_state_save
            request_project_state_save(
                tif_path,
                self.state,
                stage="operation_history",
            )
        except Exception as exc:
            self._log.info(f"operation history cache save failed: {exc}")

    @Slot(dict)
    def _on_cache_restore_done(self, payload):
        result = self._unwrap_task_result("cache_restore", payload)
        if result is not None:
            self._restore_cache_payload(result)
        elif (
            isinstance(payload, dict)
            and payload.get("status") == "failed"
            and self._admit_task_completion(
                "cache_restore", payload.get("task_token")
            )
        ):
            self.lbl_status.setText("项目缓存恢复失败")
            self.log_panel.error(
                f"项目缓存恢复失败: {str(payload.get('error') or '未知错误')}"
            )
        self._clear_task_token_if_current("cache_restore", payload)

    def _restore_cache_payload(self, payload):
        """Apply project cache already loaded by CacheRestoreWorker."""
        try:
            self._stale_path_preview = None
            self._clear_machine_route_preview()
            path = payload.get("path")
            saved = payload.get("saved") or {}
            if not saved:
                return
            processed_cache_valid = bool(payload.get("processed_cache_valid"))
            raw_mask = payload.get("raw_mask")
            raw_ox = int(payload.get("raw_ox", 0) or 0)
            raw_oy = int(payload.get("raw_oy", 0) or 0)
            processed_mask = payload.get("processed_mask")
            proc_ox = int(payload.get("proc_ox", 0) or 0)
            proc_oy = int(payload.get("proc_oy", 0) or 0)
            processed_layers = payload.get("processed_layers") or {}
            saved_turn_strategy = str(
                saved.get("turn_strategy", "contour") or "contour"
            )
            turn_strategy_user_selected = (
                saved.get("turn_strategy_user_selected") is True
            )
            legacy_unconfirmed_turn = bool(
                not turn_strategy_user_selected
                and saved_turn_strategy != "contour"
            )
            path_restore = cached_path_restore_plan(
                saved,
                path_cache_valid=(
                    bool(payload.get("path_cache_valid"))
                    and not legacy_unconfirmed_turn
                ),
            )
            restored_state = {
                k: v for k, v in saved.items()
                if k in (
                    'field_boundary', 'field_area_m2', 'entry_point', 'exit_point',
                    'forbidden_regions', 'forbidden_regions_confirmed',
                    'unload_points', 'unload_point', 'entry_point_locked',
                    'exit_point_locked', 'unload_point_locked', 'workflow_step',
                    'harvester_param_sources',
                    'harvester_params_confirmed',
                    'inference_done', 'mask_processed', 'simulation_done', 'export_done',
                    'turn_strategy', 'turn_strategy_user_selected',
                    'centerline_catalog', 'selected_centerline_ids',
                    'centerline_selection_report',
                    'work_band_catalog', 'selected_work_band_ids',
                    'work_band_selection_report',
                    'source_sha256', 'source_metadata', 'raster_preprocessing',
                    'inference_provenance', 'mask_provenance',
                    'inference_runtime',
                    'last_total_path_m', 'last_work_path_m', 'last_turn_path_m',
                    'last_entry_exit_path_m', 'last_crop_mask_geometric_coverage_pct',
                    'last_planned_target_geometric_coverage_pct', 'last_detected_mask_geometric_coverage_pct',
                    'last_rolling_rate', 'last_field_efficiency',
                )
            }
            restored_state.update(self._cache_model_identity_updates(saved))
            restored_state.update(path_restore["state_updates"])
            # A cache written before the closing-loop repair can still hold a
            # self-intersecting outline.  Repair it on restore as well, so an
            # existing project is usable without redrawing the field; an
            # unrepairable outline is kept verbatim and fails closed downstream.
            cached_boundary = list(restored_state.get('field_boundary') or [])
            if len(cached_boundary) >= 3:
                boundary_points, boundary_report = self._normalise_field_boundary(
                    cached_boundary
                )
                if boundary_points is not None:
                    restored_state['field_boundary'] = boundary_points
                    if boundary_report.get("repaired"):
                        self.log_panel.info(
                            "缓存中的田块边界闭合点重复，已自动修复"
                        )
            # Cache schema before this marker could not distinguish a deliberate
            # popup selection from an accidental wheel mutation. Migrate such a
            # non-contour value back to the documented default and invalidate
            # its path so visible state and geometry cannot disagree.
            restored_state['turn_strategy_user_selected'] = (
                turn_strategy_user_selected
            )
            if not turn_strategy_user_selected:
                restored_state['turn_strategy'] = "contour"
            restored_state['harvester_params'] = self._effective_harvester_params(
                saved.get('harvester_params') or {}
            )
            restored_state['harvester_param_sources'] = (
                self._normalise_harvester_param_sources(
                    restored_state['harvester_params'],
                    saved.get('harvester_param_sources'),
                    fallback=(
                        "legacy_unverified"
                        if saved.get('harvester_params')
                        else "default"
                    ),
                )
            )
            self.state.safe_update(**restored_state)
            # Migrate the old canvas-selected work-band snapshot into the
            # canonical centerline checkbox contract before rendering it.
            from work_band_selection import migrate_cached_centerline_catalog
            cached_catalog = getattr(self.state, "centerline_catalog", None)
            if not isinstance(cached_catalog, dict) or not cached_catalog.get("items"):
                cached_catalog = getattr(self.state, "work_band_catalog", {})
            centerline_catalog, id_map = migrate_cached_centerline_catalog(
                cached_catalog if isinstance(cached_catalog, dict) else {}
            )
            cached_selected = list(
                getattr(self.state, "selected_centerline_ids", []) or []
            )
            if not cached_selected:
                cached_selected = list(
                    getattr(self.state, "selected_work_band_ids", []) or []
                )
            selected_centerline_ids = [
                id_map.get(str(value), str(value)) for value in cached_selected
            ]
            centerline_report = getattr(
                self.state, "centerline_selection_report", {}
            )
            if not isinstance(centerline_report, dict) or not centerline_report:
                centerline_report = getattr(
                    self.state, "work_band_selection_report", {}
                ) or {}
            if id_map and isinstance(centerline_report, dict):
                centerline_report = dict(centerline_report)
                centerline_report["selected_ids"] = [
                    id_map.get(str(value), str(value))
                    for value in (centerline_report.get("selected_ids") or [])
                ]
            self.state.safe_update(
                centerline_catalog=centerline_catalog,
                selected_centerline_ids=selected_centerline_ids,
                centerline_selection_report=dict(centerline_report),
                work_band_catalog=centerline_catalog,
                selected_work_band_ids=list(selected_centerline_ids),
                work_band_selection_report=dict(centerline_report),
            )
            # DECISION-007: cache restore changes the authoritative
            # planning input after the widgets were initialised.  Keep both
            # selectors on that restored value; otherwise the task panel can
            # display the default ``contour`` while PlanWorker actually reads
            # a cached ``bow`` value from AppState.
            self.task_panel.set_turn_strategy(self.state.turn_strategy)
            self.top_toolbar.set_current_strategy(self.state.turn_strategy)
            if not self._sync_cache_model_identity():
                self.log_panel.info(
                    "缓存模型仅作为历史记录；可加载任一已注册或已确认信任的模型重新识别，"
                    "旧掩膜不会复用"
                )
            self.log_panel.info("已恢复最近项目；任务状态将按实际缓存内容同步")
            for message in payload.get("cache_validation_messages", []) or []:
                self.log_panel.warn(str(message))

            self._service_points_visible = False

            if raw_mask is not None:
                self.state.mask_raw = raw_mask
                self.state.inference_original_mask = raw_mask
                self.state.mask_offset_x = raw_ox
                self.state.mask_offset_y = raw_oy
                self.state.inference_done = True

            if saved.get('field_boundary'):
                self._draw_field_boundary()
                self.task_panel.set_task_status("import_img", "done")
                self.task_panel.set_task_status("draw_field", "done")

            if saved.get('inference_done') and raw_mask is not None:
                self.task_panel.set_task_status("segment", "done")
                self.task_panel.set_task_action("segment", "\u91cd\u8dd1")
                if not processed_cache_valid:
                    self.state.mask_processed = False
                    self.state.mask_result = None
                    self.state.centerline_catalog = {}
                    self.state.selected_centerline_ids = []
                    self.state.centerline_selection_report = {}
                    self.state.work_band_catalog = {}
                    self.state.selected_work_band_ids = []
                    self.state.work_band_selection_report = {}
                    self.task_panel.clear_centerline_selection()
                    self._sync_task_statuses()
                    # 左右必须一致：状态已经回退到"未生成作业线"，画布上就不能再留着
                    # 上一条作业线 / 路径（用户反馈"明明画出中线，清单还停在识别"）。
                    self._stale_path_preview = None
                    self._clear_display_overlay_layers()
                    self._show_mask_overlay(raw_mask)
                    if saved.get('mask_processed'):
                        self.log_panel.info("掩膜优化缓存版本已过期，请重新执行掩膜优化")
            elif saved.get('inference_done'):
                self.state.inference_done = False
                self._stale_path_preview = None
                self._clear_display_overlay_layers()
                self._sync_task_statuses()
                self.log_panel.error("未找到缓存的 AI 掩膜，请重新运行 作物识别")

            if processed_cache_valid:
                if processed_mask is not None:
                    mr = saved.get("mask_result") or {}
                    if isinstance(mr, dict):
                        mr = dict(mr)
                        mr["processed_mask"] = processed_mask
                        for key, value in processed_layers.items():
                            if isinstance(value, np.ndarray) and value.shape[:2] == processed_mask.shape[:2]:
                                mr[key] = value
                        uncertain = mr.get("uncertain_residual_mask")
                        if isinstance(uncertain, np.ndarray):
                            mr["neutral_support_mask"] = uncertain
                        support = processed_mask.copy()
                        headland = mr.get("headland_mask")
                        if isinstance(headland, np.ndarray):
                            support = cv2.bitwise_or(support, headland)
                        if isinstance(uncertain, np.ndarray):
                            support = cv2.bitwise_or(support, uncertain)
                        mr["planning_support_mask"] = support
                        self.state.mask_result = mr
                    self.state.mask_offset_x = proc_ox
                    self.state.mask_offset_y = proc_oy
                    self.state.mask_processed = True
                    self.task_panel.set_task_status("process", "done")
                    self.task_panel.set_task_action("process", "\u91cd\u505a")
                else:
                    self.state.mask_processed = False
                    self.state.mask_result = None
                    self._sync_task_statuses()
                    self.log_panel.error("未找到缓存的掩膜优化结果，请重新执行掩膜优化")


            if path_restore["restore_editable_path"]:
                restored_path = self._restore_cached_path_geometry(saved)
                if restored_path:
                    self.route_info.update_route(restored_path, restored_path.get("validation", {}))
            elif path_restore["historical_path_present"]:
                self.log_panel.warn(
                    "缓存路线缺少可编辑的 canonical segment 语义，已仅作为历史记录处理"
                )
            if path_restore["restore_editable_path"] and self.state.mask_processed:
                self.task_panel.set_task_status("plan", "done")
            if self._pipeline_result is None and path_restore["historical_path_present"]:
                self._log.info("cached path state restored without full path geometry")
            # 一致性兜底：掩膜没恢复成功时，缓存里残留的作业线/路径必须一起丢掉。
            # 否则会出现"画布上还画着作业线，任务清单却退回作物识别"的左右错位
            # （用户 2026-10-23 反馈；derive_task_statuses 在 process_done=False 时
            # 本来就把中心线当不存在，只是画布还在画）。
            #
            # 顺序铁律：先清状态、再按状态重画画布。反过来（先画再清）会把刚按
            # 状态画好的识别掩膜与田块边界一起擦掉，用户看到的就是"左边已经到第 4 步、
            # 右边一片空白"（2026-10-26 反馈；缓存 inference_done=True 且
            # mask_processed=False 时必现）。清理也只能清派生层，不能清底图证据层。
            if not getattr(self.state, "mask_processed", False):
                cleared_centerlines = bool(
                    ((getattr(self.state, "centerline_catalog", {}) or {}).get("items"))
                    or getattr(self.state, "selected_centerline_ids", None)
                )
                self.state.centerline_catalog = {}
                self.state.work_band_catalog = {}
                self.state.selected_centerline_ids = []
                self.state.selected_work_band_ids = []
                self.state.centerline_selection_report = {}
                self.state.work_band_selection_report = {}
                if cleared_centerlines:
                    self.log_panel.warn(
                        "掩膜缓存未通过校验，已一并清除缓存中的作业线与路径，请重新执行作物识别"
                    )
                self.task_panel.clear_centerline_selection()
                self._stale_path_preview = None
                self._clear_derived_path_layers()
            # 画布永远是状态的函数：所有会改变状态的分支收敛之后，再统一重绘一次。
            self._restore_visual_from_state()
            if (
                getattr(self.state, "mask_processed", False)
                and isinstance(getattr(self.state, "mask_result", None), dict)
                and not self._centerline_state_view()[0].get("items")
            ):
                self.log_panel.info("缓存已恢复；请在任务清单中点击’生成作业线’。")
            self._repair_canvas_if_state_disagrees(where="cache_restore")
            # DECISION-002/013: the task panel is derived from the completely
            # restored domain state.  The individual green checks above are
            # only progressive feedback while large cache arrays are applied;
            # without this final synchronization later actionable rows remain
            # disabled even though ``mask_result`` is already available.
            self._sync_task_statuses()
            history = payload.get("history") or {}
            self.state.undo_stack = list(history.get("undo_stack") or [])
            self.state.redo_stack = list(history.get("redo_stack") or [])
            for message in history.get("messages", []) or []:
                self.log_panel.warn(str(message))
            if self.state.undo_stack or self.state.redo_stack:
                self.log_panel.info(
                    "已恢复逐步操作历史："
                    f"可撤回 {len(self.state.undo_stack)} 步，"
                    f"可重做 {len(self.state.redo_stack)} 步"
                )
            # 恢复结束必须给结论并收起进度条：否则状态条一直停在
            # 「正在读取逐步操作历史…」、进度条停在 90%，看起来像程序卡死
            # （用户在 2026-10-26 截图里就是这个状态）。
            self.progress_bar.hide()
            self.top_toolbar.set_progress(0)
            from ui_messages import project_restored

            inference_ready = bool(
                getattr(self.state, "inference_done", False)
                and isinstance(getattr(self.state, "mask_raw", None), np.ndarray)
            )
            if getattr(self.state, "mask_processed", False):
                next_step = "掩膜优化结果可用，可生成作业线"
            elif inference_ready:
                next_step = "识别结果可用，可执行掩膜优化"
            elif len(getattr(self.state, "field_boundary", []) or []) >= 3:
                next_step = "田块已圈选，可运行作物识别"
            else:
                next_step = ""
            self.set_status_text(project_restored(next_step), "ready")
        except Exception as e:
            self._log.info(f"cache restore failed: {e}")

    def _on_save_project(self):
        if not self._tif_path:
            self.log_panel.info("请先打开影像或项目。")
            return
        try:
            from cache import save_project_state, wait_for_project_state_saves
            save_project_state(self._tif_path, self.state, stage="manual_save")
            wait_for_project_state_saves(self._tif_path, timeout=2.0)
            self.log_panel.success("项目已保存")
            self.lbl_status.setText("项目已保存")
        except Exception as e:
            self.log_panel.error(f"项目保存失败: {e}")
            QMessageBox.warning(self, "项目保存失败", str(e))

    def _on_open_tif(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开影像", "", "GeoTIFF (*.tif *.tiff);;TIFF (*.tif);;All files (*)")
        if path:
            self._load_tif_path(path)

    def _on_load_model(self):
        if getattr(self, "_model_load_in_progress", False) or self._has_background_worker():
            self.log_panel.warn(f"有任务在运行 · 等它结束再切换模型")
            return
        path, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "PyTorch (*.pt);;All files (*)")
        if path:
            self._on_load_model_at(path)

    @staticmethod
    def _model_display_name(model):
        path = model.get("path", "") if isinstance(model, dict) else str(model or "")
        raw_name = model.get("name", "") if isinstance(model, dict) else ""
        base = os.path.basename(path or raw_name)
        if os.path.basename(os.path.dirname(path)).lower() == "weights":
            run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
            parent_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            if run_name and run_name.lower() in ("w", "weights") and parent_name:
                run_name = parent_name
            if run_name:
                return f"{run_name} · {base}"
        return raw_name or base or path

    def _custom_model_list(self):
        items = self.cfg.section("models").get("custom", [])
        clean = []
        seen = set()
        for item in items if isinstance(items, list) else []:
            path = item.get("path", "") if isinstance(item, dict) else str(item or "")
            if not path or path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            clean.append({"name": item.get("name") or self._model_display_name({"path": path}), "path": path})
        return clean

    def _persist_custom_models(self):
        builtin_paths = {m.get("path") for m in self._builtin_model_list()}
        custom = []
        seen = set()
        for model in self._model_list:
            path = model.get("path", "")
            if not path or path in builtin_paths or path in seen:
                continue
            seen.add(path)
            custom.append({"name": model.get("name") or self._model_display_name(model), "path": path})
        config_snapshot = self.cfg.update_section("models", {"custom": custom})
        save_json_atomic(self.cfg.config_path(), config_snapshot)

    def _on_model_combo_changed(self, model_path_or_name):
        if model_path_or_name == "__IMPORT_MODEL_FOLDER__" or model_path_or_name == "导入模型文件夹...":
            self._on_import_model_folder()
            return
        matches = []
        for m in self._model_list:
            path = m.get("path", "")
            name = m.get("name", os.path.basename(path))
            base = os.path.basename(path)
            if model_path_or_name in (name, path, base):
                matches.append(m)
        if len(matches) == 1:
            if not self._on_load_model_at(matches[0].get("path", "")):
                self._restore_model_selector()
        elif len(matches) > 1:
            self.log_panel.error(f"模型名称存在多个匹配: {model_path_or_name}")
            self._restore_model_selector()
        elif os.path.isfile(model_path_or_name):
            if not self._on_load_model_at(model_path_or_name):
                self._restore_model_selector()

    def _merge_model_list(self, models):
        existing = {m.get("path") for m in self._model_list}
        added = []
        for model in models or []:
            path = model.get("path", "") if isinstance(model, dict) else str(model)
            if not path or path in existing:
                continue
            name = self._model_display_name(model if isinstance(model, dict) else {"path": path})
            item = {"name": name, "path": path}
            self._model_list.append(item)
            existing.add(path)
            added.append(item)
        if added:
            self._persist_custom_models()
            self.top_toolbar.populate_models([m.get("name") for m in self._model_list])
            self.task_panel.set_model_options(self._model_list, self.state.current_model_path)
        return added

    def _on_import_model_folder(self):
        if getattr(self, "_model_load_in_progress", False) or self._has_background_worker():
            self.log_panel.warn(f"有任务在运行 · 等它结束再导入或切换模型")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择模型文件夹", "")
        if not folder:
            self.task_panel.set_model_options(self._model_list, self.state.current_model_path)
            return
        found = scan_model_dir(folder)
        if not found:
            self.log_panel.info("未找到模型：所选文件夹及其子文件夹中没有找到 .pt 模型文件。")
            self.task_panel.set_model_options(self._model_list, self.state.current_model_path)
            return
        from pyside6_app.dialogs import ModelImportDialog
        dlg = ModelImportDialog(found, self, best_only=True)
        if dlg.exec() == QDialog.Accepted:
            added = self._merge_model_list(dlg.selected_models)
            self.log_panel.success(f"已加入 {len(added)} 个模型")
            if added:
                self._on_load_model_at(added[0].get("path", ""))
        self.task_panel.set_model_options(self._model_list, self.state.current_model_path)

    def _on_model_dropdown_open(self):
        if getattr(self, "_model_load_in_progress", False) or self._has_background_worker():
            return
        self._refresh_builtin_models()

    def _on_model_manager(self):
        if getattr(self, "_model_load_in_progress", False) or self._has_background_worker():
            self.log_panel.warn(f"有任务在运行 · 等它结束再切换模型")
            return
        models = self._model_list or self._refresh_builtin_models()
        cur = getattr(self.state, 'current_model_path', None)
        dlg = ModelManagerDialog(models, cur or "", self)
        if dlg.exec() == QDialog.Accepted and dlg.selected_path:
            self._merge_model_list([{"name": self._model_display_name({"path": dlg.selected_path}), "path": dlg.selected_path}])
            self._on_load_model_at(dlg.selected_path)

    def _on_load_model_at(self, path):
        """DECISION-006: trust-check, then load weights outside the GUI thread.

        The expensive Ultralytics constructor is delegated to
        ``ModelLoadWorker``.  A separate engine is returned and committed only
        after the result is admitted on the GUI thread, so failed/cancelled
        loads cannot leave the visible model identity pointing at a half-loaded
        engine.
        """
        if not path:
            self.log_panel.error(f"模型文件不存在: {path}")
            return False
        path = os.path.abspath(os.fspath(path))
        if not os.path.isfile(path):
            self.log_panel.error(f"模型文件不存在: {path}")
            return False
        if getattr(self, "_model_load_in_progress", False):
            self.log_panel.warn("模型正在加载，请等待当前加载完成")
            return False
        if self._inference_task_is_running():
            # DECISION-006/007: never race model construction with a live
            # predict call.  Cancellation is cooperative and is not an
            # immediate CUDA-safety guarantee.
            self.log_panel.warn(f"有任务在运行 · 等它结束再切换模型")
            return False
        if self._has_background_worker():
            self.log_panel.warn(f"有任务在运行 · 等它结束再切换模型")
            return False
        try:
            builtin_root = os.path.abspath(self._builtin_model_dir())
            in_builtin_dir = os.path.commonpath([builtin_root, path]) == builtin_root
        except ValueError:
            in_builtin_dir = False
        registry_entry = self._model_registry_entry(path)
        is_registered = bool(
            registry_entry and str(registry_entry.get("sha256", "") or "").strip()
        )
        # DECISION-006: only explicitly registered weights are trusted without
        # confirmation. A .pt file inside data/models that is absent from the
        # registry is treated like an external weight.
        needs_confirmation = (not in_builtin_dir) or (not is_registered)
        if needs_confirmation and QMessageBox.question(
            self,
            "模型安全确认",
            "PyTorch .pt 文件可能包含可执行的序列化内容。仅应加载由你训练或来源可信的模型。\n\n"
            f"文件：{path}\n\n确认信任并继续加载吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            self.log_panel.info("已取消加载未经确认的模型")
            return False

        previous_path = str(getattr(self.state, "current_model_path", "") or "")
        previous_sha256 = str(getattr(self.state, "model_sha256", "") or "")
        token = make_task_token(self.state, "model_load")
        self._active_model_load_token = token
        self._pending_model_load = {
            "path": path,
            "previous_path": previous_path,
            "previous_sha256": previous_sha256,
        }
        worker = ModelLoadWorker(
            path,
            expected_sha256=str(registry_entry.get("sha256", "") or ""),
            token=token,
            engine_factory=ModelEngine,
        )
        worker.finished.connect(self._on_model_load_done)
        self.lbl_status.setText("正在后台加载模型，窗口仍可操作...")
        self.top_toolbar.set_progress(5, "模型加载中...")
        self.progress_bar.show()
        self.progress_bar.setValue(5)
        if not self._start_background_worker(
            worker,
            "模型加载已转入后台，完成前不会阻塞界面。",
            warn_if_busy=False,
        ):
            token.invalidate()
            self._pending_model_load = None
            self._active_model_load_token = None
            self._restore_model_selector()
            self._finish_model_load_ui()
            return False
        self._set_model_loading_ui(True)
        return True

    def _finish_model_load_ui(self):
        """Clear transient model-loading UI state after a terminal result."""
        self._set_model_loading_ui(False)
        if hasattr(self, "progress_bar"):
            self.progress_bar.hide()
        if hasattr(self, "btn_cancel_task"):
            self.btn_cancel_task.hide()
            self.btn_cancel_task.setEnabled(True)
        if hasattr(self, "top_toolbar"):
            self.top_toolbar.set_progress(0)
            self.top_toolbar.set_worker_cancel_visible(False)

    def _run_pending_model_action(self, action):
        """Run an action queued by ``_ensure_model_loaded`` after thread exit."""
        if self._has_background_worker():
            QTimer.singleShot(
                50,
                lambda queued_action=action: self._run_pending_model_action(
                    queued_action
                ),
            )
            return
        if action == "segment":
            self._on_segment()

    @Slot(dict)
    def _on_model_load_done(self, payload):
        """Admit and commit a completed background model load on the GUI thread."""
        payload = payload if isinstance(payload, dict) else {}
        active_token = getattr(self, "_active_model_load_token", None)
        token = payload.get("task_token")
        status = str(payload.get("status", "") or "")
        if token is not active_token:
            return

        if status == "cancelled" or getattr(self, "_task_cancel_requested", False):
            result = payload.get("result") or {}
            engine = result.get("engine") if isinstance(result, dict) else None
            if engine is not None:
                try:
                    engine.unload()
                except Exception:
                    pass
            self._pending_model_load = None
            self._pending_model_action = None
            self._clear_task_token_if_current("model_load", payload)
            self._restore_model_selector()
            self._finish_model_load_ui()
            return

        result = self._unwrap_task_result("model_load", payload)
        if result is None:
            # A failed or stale completion must not leave controls locked.
            self._pending_model_load = None
            self._pending_model_action = None
            self._clear_task_token_if_current("model_load", payload)
            self._restore_model_selector()
            self._finish_model_load_ui()
            if status == "failed":
                self.lbl_status.setText("模型加载失败")
                self.log_panel.error(
                    f"模型加载失败: {str(payload.get('error') or '未知错误')}"
                )
            return

        new_engine = result.get("engine")
        model_path = (
            os.path.abspath(os.fspath(result.get("model_path") or ""))
            if result.get("model_path")
            else ""
        )
        model_sha256 = str(result.get("model_sha256") or "").lower()
        if (
            new_engine is None
            or not model_path
            or not model_sha256
            or not bool(getattr(new_engine, "is_loaded", lambda: False)())
        ):
            self._pending_model_load = None
            self._pending_model_action = None
            self._clear_task_token_if_current("model_load", payload)
            self._restore_model_selector()
            self._finish_model_load_ui()
            self.lbl_status.setText("模型加载失败")
            self.log_panel.error("模型加载失败: 后台结果缺少有效模型身份")
            return

        pending = dict(getattr(self, "_pending_model_load", {}) or {})
        previous_path = str(pending.get("previous_path", "") or "")
        previous_sha256 = str(pending.get("previous_sha256", "") or "")
        old_engine = getattr(self, "model_engine", None)
        self.model_engine = new_engine
        if old_engine is not None and old_engine is not new_engine:
            try:
                old_engine.unload()
            except Exception as exc:
                self._log.info(f"旧模型资源释放提示: {exc}")

        name = os.path.basename(model_path)
        identity_changed = bool(
            (
                previous_path
                and os.path.normcase(os.path.abspath(previous_path))
                != os.path.normcase(model_path)
            )
            or (previous_sha256 and previous_sha256 != model_sha256)
        )
        derived_evidence_present = bool(
            getattr(self.state, "inference_done", False)
            or getattr(self.state, "mask_processed", False)
            or getattr(self.state, "auto_path_planned", False)
        )
        # If an old snapshot has derived results but no content identity, do
        # not let the newly loaded weight inherit those results silently.
        model_identity_changed = bool(
            identity_changed
            or (derived_evidence_present and (not previous_path or not previous_sha256))
        )
        if model_identity_changed or not previous_path:
            self._save_system_undo("加载模型")
        self.top_toolbar.set_model_name(name)
        self.top_toolbar.set_current_model(name)
        self.task_panel.set_model_path(name)
        self.state.current_model_name = name
        self.state.current_model_path = model_path
        self.state.model_sha256 = model_sha256
        cache_model_matches = self._sync_cache_model_identity()
        historical_model_differs = bool(
            not cache_model_matches
            and (
                getattr(self.state, "historical_model_path", "")
                or getattr(self.state, "historical_model_sha256", "")
            )
        )
        if historical_model_differs or model_identity_changed:
            # DECISION-007: only an admitted successful model change
            # invalidates derived inference/mask/path evidence.
            self._invalidate_active_inference_runner()
            self._invalidate_analysis_from(
                "inference",
                "识别模型已变化，历史识别掩膜和路径已清除",
            )
        if historical_model_differs:
            self.log_panel.info(
                "当前模型已载入本次项目；它与历史缓存模型不同，"
                "可直接重新识别，系统不会复用旧掩膜"
            )
        self.log_panel.success(f"模型加载: {name}")
        pending_action = self._pending_model_action
        self._pending_model_action = None
        self._pending_model_load = None
        self._clear_task_token_if_current("model_load", payload)
        self._finish_model_load_ui()
        self._sync_task_statuses()
        if pending_action:
            QTimer.singleShot(
                0,
                lambda action=pending_action: self._run_pending_model_action(action),
            )

    def _on_toolbar_action(self, aid):
        H = {
            "OPEN": self._on_open_tif,
            "MODEL": self._on_load_model,
            "AI_INFER": self._on_segment,
            "MASK_PROCESS": self._on_mask_process,
            "PARAMS": self._on_params_dialog,
            "ENTRY_EXIT": self._on_toggle_entry_exit,
            "TURN_STRATEGY": self._on_turn_strategy_btn,
            "PLAN": self._on_plan,
            "PLAY": self._on_toggle_animation,
            # The primary workflow export is the terminal-facing numbered
            # TXT projection (001.txt, 002.txt, ...).  Research single-file
            # TXT and GeoJSON remain available through their explicit File
            # menu actions.
            "EXPORT": self._on_export_machine_compatibility_path,
            "SETTINGS": self._on_settings_dialog,
            "UNDO": self._on_undo,
            "REDO": self._on_redo,
            "CANCEL_TASK": self._cancel_background_worker,
        }
        h = H.get(aid)
        if h:
            h()

    def _on_tool_changed(self, t):
        self.log_panel.info(f"当前工具: {t}")

    def _save_system_undo(self, label):
        """Record a user-visible system operation for global undo/redo."""
        self.state.save_undo(label)
        self.top_toolbar.refresh_actions(self.state)

    def _on_undo(self):
        if self.state.undo():
            self.log_panel.info("已撤销")
            self._restore_visual_from_state()
            self._sync_task_statuses()
            self.top_toolbar.refresh_actions(self.state)
        else:
            self.log_panel.info("没有可撤销操作")
            self.top_toolbar.refresh_actions(self.state)

    def _on_redo(self):
        if self.state.redo():
            self.log_panel.info("已重做")
            self._restore_visual_from_state()
            self._sync_task_statuses()
            self.top_toolbar.refresh_actions(self.state)
        else:
            self.log_panel.info("没有可重做操作")
            self.top_toolbar.refresh_actions(self.state)

    def _path_result_from_state(self):
        """Build the current route projection from AppState-owned route fields.

        Undo/redo restores AppState, not the transient pipeline cache.  Rebuilding
        here prevents a later pipeline result from being rendered or exported as
        if it were the restored route.
        """
        segments = list(getattr(self.state, "auto_path_segments", []) or [])
        if not segments:
            segments = list(getattr(self.state, "auto_path", []) or [])
        full_path, segment_types, segment_lengths_m = [], [], []
        for index, segment in enumerate(segments):
            points = self._coerce_points(getattr(segment, "points", None))
            if len(points) < 2:
                continue
            full_path.append(points)
            segment_types.append(
                str(getattr(segment, "segment_type", "") or "work")
            )
            segment_lengths_m.append(
                float(getattr(segment, "length_m", 0.0) or self._segment_length(points))
            )
        if not full_path:
            return {}
        validation = dict(getattr(self.state, "path_validation", {}) or {})
        artifact = dict(getattr(self.state, "path_artifact", {}) or {})
        metadata = {
            key: artifact[key]
            for key in (
                "turn_assessment",
                "planning_factors",
                "requested_strategy",
                "coordinate_space",
                "machine_readiness",
                "machine_executable",
                "path_candidates",
                "layout",
                "manual_route",
            )
            if key in artifact
        }
        manual_route = metadata.get("manual_route")
        if isinstance(manual_route, dict):
            metadata["manual_route_edited"] = True
            metadata["manual_route_validated"] = bool(
                validation.get("manual_route_validated") is True
            )
        return self._normalise_path_result(
            {
                **metadata,
                "full_path": full_path,
                "segment_types": segment_types,
                "segment_lengths_m": segment_lengths_m,
                "is_valid": bool(
                    getattr(self.state, "auto_path_valid", False)
                    and validation.get("valid", False)
                ),
                "validation": validation,
                "description": str(getattr(self.state, "auto_path_desc", "") or ""),
                "geo_points": list(getattr(self.state, "auto_path_geo", []) or []),
                "provenance": dict(getattr(self.state, "path_provenance", {}) or {}),
                "runtime": dict(getattr(self.state, "path_runtime", {}) or {}),
            }
        )

    def _stash_stale_path_preview(self, reason=""):
        """Keep the last canonical route visible while its replacement runs.

        The canonical AppState route is still cleared by invalidation.  This
        copy is display-only and is never used by export, simulation, or
        validation, so preserving it cannot make stale geometry executable.
        """
        if not getattr(self.state, "auto_path_planned", False):
            return
        path_result = self._path_result_from_state()
        if not path_result or not path_result.get("full_path"):
            return
        stale = deepcopy(path_result)
        stale["is_valid"] = False
        stale["validation"] = {}
        stale["stale_preview"] = True
        stale["description"] = (
            "待重新规划预览（不可导出）"
            + (f"：{reason}" if reason else "")
        )
        self._stale_path_preview = stale

    @staticmethod
    def _build_machine_route_preview(manifest_points, protocol_metadata=None):
        """Build an exact pixel-space view of the files just written.

        File boundaries matter because every <=50-point volume is an
        independent route and its final record has no outgoing edge.  The
        preview therefore never invents a connector between two files.
        """
        metadata = dict(protocol_metadata or {})
        records = []
        for point in list(manifest_points or []):
            if not isinstance(point, dict):
                continue
            try:
                pixel_x = float(point["pixel_x"])
                pixel_y = float(point["pixel_y"])
                status = int(point.get("status", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (pixel_x, pixel_y)):
                continue
            records.append({
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "status": 1 if status == 1 else 0,
            })
        if len(records) < 2:
            return None

        points_per_file = []
        for value in list(metadata.get("points_per_file") or []):
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                points_per_file.append(count)
        if not points_per_file or sum(points_per_file) != len(records):
            points_per_file = [len(records)]

        output_files = [
            os.path.abspath(os.fspath(path))
            for path in list(metadata.get("output_files") or [])
            if path
        ]
        while len(output_files) < len(points_per_file):
            output_files.append(f"第{len(output_files) + 1}份")

        full_path = []
        segment_types = []
        file_chunks = []
        work_edge_count = 0
        turn_edge_count = 0
        offset = 0
        for file_index, count in enumerate(points_per_file):
            chunk = [dict(item) for item in records[offset:offset + count]]
            offset += count
            if not chunk:
                continue
            # The file now writes the fixed status 2 for every record, so the
            # preview cannot read work/turn from the file.  It keeps the
            # internal outgoing-edge statuses of the manifest points so the
            # 作业边/转弯边 breakdown still reflects the control route.
            chunk[-1]["status"] = 0
            for item in chunk:
                item["file_index"] = file_index
            for edge_index in range(max(0, len(chunk) - 1)):
                source = chunk[edge_index]
                target = chunk[edge_index + 1]
                segment_type = "work" if source["status"] == 1 else "turn"
                full_path.append([
                    (source["pixel_x"], source["pixel_y"]),
                    (target["pixel_x"], target["pixel_y"]),
                ])
                segment_types.append(segment_type)
                if segment_type == "work":
                    work_edge_count += 1
                else:
                    turn_edge_count += 1
            file_chunks.append({
                "path": output_files[file_index],
                "records": chunk,
                "point_count": len(chunk),
            })
        if not full_path:
            return None

        file_names = [os.path.basename(item["path"]) for item in file_chunks]
        if len(file_names) == 1:
            file_label = file_names[0]
        else:
            file_label = f"{file_names[0]}…{file_names[-1]}"
        info = (
            f"{file_label}｜{len(records)}点｜{len(file_chunks)}份｜"
            f"作业边{work_edge_count}/转弯边{turn_edge_count}"
        )
        tooltip_lines = [
            "农机文件视图：严格按已写出的 $PATH 控制点、文件边界和出边状态绘制。",
            "每条记录直接以 $PATH 开始；文件状态码固定为 2，预览仍按 0=转弯、1=作业 区分内部段；"
            "转弯是协议直线弦，不代表车辆实际转向曲线。",
        ]
        tooltip_lines.extend(
            f"{item['path']}（{item['point_count']}点）" for item in file_chunks
        )
        return {
            "path_result": {
                "full_path": full_path,
                "segment_types": segment_types,
                "validation": {},
                "is_valid": False,
            },
            "file_chunks": file_chunks,
            "point_count": len(records),
            "file_count": len(file_chunks),
            "work_edge_count": work_edge_count,
            "turn_edge_count": turn_edge_count,
            "info": info,
            "tooltip": "\n".join(tooltip_lines),
        }

    def _set_machine_route_preview(self, manifest_points, protocol_metadata):
        preview = self._build_machine_route_preview(
            manifest_points, protocol_metadata
        )
        self._machine_route_preview = preview
        if preview is None:
            self.image_view.set_machine_route_available(False)
            return None
        self.image_view.set_machine_route_available(
            True, preview["info"], preview["tooltip"]
        )
        self.image_view.set_route_display_visible(True)
        return preview

    def _clear_machine_route_preview(self):
        self._machine_route_preview = None
        self._route_display_mode = "continuous"
        if hasattr(self, "image_view"):
            viewer = self.image_view.viewer
            if hasattr(viewer, "clear_overlay_layer"):
                viewer.clear_overlay_layer("route")
            self.image_view.set_machine_route_available(False)
            self.image_view.set_route_display_mode("continuous")

    @staticmethod
    def _canonical_route_display_info(path_result):
        path_result = path_result or {}
        work_count = len(path_result.get("tracks") or [])
        turn_items = list(path_result.get("turns") or [])
        direct_count = sum(
            1
            for item in turn_items
            if str(item.get("type") or "") in {"turn", "turn_reverse", "turn_aux"}
            and len(item.get("points") or []) <= 2
        )
        detail = f"连续规划｜{work_count}条作业线｜{len(turn_items)}条非作业段"
        if direct_count:
            detail += f"｜含{direct_count}条人工可调直连"
        return detail, direct_count

    @staticmethod
    def _split_polyline_by_field_boundary(points, boundary):
        """Split a displayed polyline into field-inside/outside pieces.

        This is a display-only operation.  It preserves every original route
        vertex and inserts only the exact intersections with the selected
        boundary, so an outside curve can be highlighted without changing the
        canonical path or its validation result.
        """
        clean = []
        for point in points or []:
            try:
                value = (float(point[0]), float(point[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if not clean or value != clean[-1]:
                clean.append(value)
        if len(clean) < 2:
            return []
        try:
            polygon = np.asarray(boundary, dtype=np.float32)
        except (TypeError, ValueError):
            return [(clean, True)]
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            return [(clean, True)]

        def cross(first, second):
            return float(first[0] * second[1] - first[1] * second[0])

        def subtract(first, second):
            return (first[0] - second[0], first[1] - second[1])

        def add_scaled(first, direction, scale):
            return (
                float(first[0] + direction[0] * scale),
                float(first[1] + direction[1] * scale),
            )

        def distance(first, second):
            return math.hypot(first[0] - second[0], first[1] - second[1])

        def intersection_parameters(start, end, edge_start, edge_end):
            direction = subtract(end, start)
            edge = subtract(edge_end, edge_start)
            offset = subtract(edge_start, start)
            denominator = cross(direction, edge)
            epsilon = 1e-9
            if abs(denominator) > epsilon:
                line_t = cross(offset, edge) / denominator
                edge_u = cross(offset, direction) / denominator
                if -epsilon <= line_t <= 1.0 + epsilon and -epsilon <= edge_u <= 1.0 + epsilon:
                    return [max(0.0, min(1.0, float(line_t)))]
                return []
            if abs(cross(offset, direction)) > epsilon:
                return []
            length_sq = direction[0] ** 2 + direction[1] ** 2
            if length_sq <= epsilon:
                return []
            return [
                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            ((point[0] - start[0]) * direction[0]
                             + (point[1] - start[1]) * direction[1])
                            / length_sq
                        ),
                    ),
                )
                for point in (edge_start, edge_end)
            ]

        pieces = []
        for start, end in zip(clean[:-1], clean[1:]):
            direction = subtract(end, start)
            parameters = [0.0, 1.0]
            for index in range(len(polygon)):
                edge_start = (
                    float(polygon[index][0]),
                    float(polygon[index][1]),
                )
                edge_end = (
                    float(polygon[(index + 1) % len(polygon)][0]),
                    float(polygon[(index + 1) % len(polygon)][1]),
                )
                parameters.extend(
                    intersection_parameters(start, end, edge_start, edge_end)
                )
            parameters = sorted(parameters)
            unique_parameters = []
            for value in parameters:
                if not unique_parameters or abs(value - unique_parameters[-1]) > 1e-7:
                    unique_parameters.append(value)
            for left, right in zip(unique_parameters[:-1], unique_parameters[1:]):
                if right - left <= 1e-7:
                    continue
                left_point = add_scaled(start, direction, left)
                right_point = add_scaled(start, direction, right)
                middle = add_scaled(start, direction, (left + right) * 0.5)
                inside = cv2.pointPolygonTest(
                    polygon,
                    (float(middle[0]), float(middle[1])),
                    False,
                ) >= -1e-6
                if (
                    pieces
                    and pieces[-1][1] is inside
                    and distance(pieces[-1][0][-1], left_point) <= 1e-6
                ):
                    pieces[-1][0].append(right_point)
                else:
                    pieces.append(([left_point, right_point], inside))
        return pieces or [(clean, True)]

    def _centerline_state_view(self):
        """Return the canonical centerline catalog and checkbox state."""
        canonical = getattr(self.state, "centerline_catalog", None)
        legacy = getattr(self.state, "work_band_catalog", None)
        catalog = (
            canonical if isinstance(canonical, dict) and canonical.get("items")
            else legacy if isinstance(legacy, dict) else {}
        ) or {}
        selected = list(getattr(self.state, "selected_centerline_ids", []) or [])
        if not selected:
            selected = list(getattr(self.state, "selected_work_band_ids", []) or [])
        report = getattr(self.state, "centerline_selection_report", None)
        if not isinstance(report, dict) or not report:
            report = getattr(self.state, "work_band_selection_report", {}) or {}
        return catalog, selected, dict(report or {})

    def _refresh_centerline_selector_panel(self):
        if not hasattr(self, "route_info"):
            return
        catalog, selected, _report = self._centerline_state_view()
        if catalog.get("items"):
            self.task_panel.update_centerline_selection(catalog, selected)
        else:
            self.task_panel.clear_centerline_selection()

    def _sanitize_centerline_state_to_field(self):
        """Remove cached centerlines that are not covered by the selected field.

        DECISION-024: a cache may have been written before the final field
        boundary gate was added.  Sanitize that legacy catalog before any
        selector or route overlay is restored, so reopening a project cannot
        bring back a detached group of selectable centerlines.
        """
        canonical = getattr(self.state, "centerline_catalog", None)
        legacy = getattr(self.state, "work_band_catalog", None)
        catalog = (
            canonical
            if isinstance(canonical, dict) and canonical.get("items")
            else legacy
            if isinstance(legacy, dict)
            else {}
        ) or {}
        items = list(catalog.get("items") or [])
        if not items:
            return False

        mask = getattr(self.state, "mask_raw", None)
        if not isinstance(mask, np.ndarray):
            result = getattr(self.state, "mask_result", None)
            mask = result.get("processed_mask") if isinstance(result, dict) else None
        if not isinstance(mask, np.ndarray):
            return False

        from path_planner import segment_within_field_boundary
        from row_geometry import field_polygon_local

        polygon = field_polygon_local(self.state, mask.shape[:2])
        if polygon is None:
            return False

        kept = []
        excluded = []
        for item_index, item in enumerate(items):
            points = item.get("centerline") or item.get("work_line") or []
            if segment_within_field_boundary(points, polygon):
                kept.append(item)
                continue
            excluded.append(
                {
                    "item_index": int(item_index),
                    "id": str(item.get("id") or ""),
                    "reason": "outside_field_boundary",
                }
            )
        if not excluded:
            return False

        updated_catalog = dict(catalog)
        updated_catalog["items"] = kept
        updated_catalog["count"] = len(kept)
        valid_ids = {str(item.get("id")) for item in kept}
        selected = list(getattr(self.state, "selected_centerline_ids", []) or [])
        if not selected:
            selected = list(getattr(self.state, "selected_work_band_ids", []) or [])
        selected = [str(value) for value in selected if str(value) in valid_ids]

        report = getattr(self.state, "centerline_selection_report", None)
        if not isinstance(report, dict) or not report:
            report = getattr(self.state, "work_band_selection_report", {}) or {}
        report = dict(report)
        report["selected_ids"] = list(selected)
        report["selected_count"] = len(selected)
        report["candidate_count"] = len(kept)
        report["selection_mode"] = (
            "empty"
            if not selected
            else "single"
            if len(selected) == 1
            else "multi"
        )
        report["centerline_field_boundary_filter"] = {
            "algorithm_version": "centerline-field-boundary-filter-v1",
            "input_line_count": len(items),
            "kept_line_count": len(kept),
            "excluded": excluded,
            "excluded_count": len(excluded),
            "reason": "excluded_cached_lines_outside_field_boundary",
        }
        self.state.safe_update(
            centerline_catalog=updated_catalog,
            selected_centerline_ids=list(selected),
            centerline_selection_report=dict(report),
            work_band_catalog=updated_catalog,
            selected_work_band_ids=list(selected),
            work_band_selection_report=dict(report),
        )
        return True

    def _sync_display_layer_availability(self):
        """Expose current drawable state to the display-only layer panel."""
        if not hasattr(self, "image_view"):
            # A few state-only reliability tests intentionally construct the
            # window without QWidget initialisation; display synchronisation
            # is irrelevant for those probes.
            return
        if getattr(self, "_show_mask_before", False):
            # Source comparison intentionally shows only the immutable TIF;
            # do not leave rows from the previous result selectable.
            for key in DISPLAY_LAYER_KEYS:
                self.image_view.set_display_layer_available(key, False)
            return
        result = getattr(self.state, "mask_result", None)
        result = result if isinstance(result, dict) else {}
        raw = getattr(self.state, "mask_raw", None)
        processed = result.get("processed_mask")
        has_raw = isinstance(raw, np.ndarray)
        has_processed = isinstance(processed, np.ndarray)
        has_headland = isinstance(result.get("headland_mask"), np.ndarray)
        has_uncertain = isinstance(
            result.get("uncertain_residual_mask"), np.ndarray
        )
        catalog, selected_centerline_ids, _report = self._centerline_state_view()
        selected_centerline_ids = {
            str(value) for value in selected_centerline_ids if value is not None
        }
        has_centerlines = False
        for item in list(catalog.get("items") or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("id") or "") not in selected_centerline_ids:
                continue
            points = item.get("centerline")
            if points is None:
                points = item.get("work_line")
            try:
                points = np.asarray(points, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if (
                points.ndim == 2
                and points.shape[0] >= 2
                and points.shape[1] == 2
                and np.isfinite(points).all()
            ):
                has_centerlines = True
                break
        path_result = self._path_result_from_state()
        has_route = bool(
            isinstance(path_result, dict) and path_result
        ) or bool(
            isinstance(getattr(self, "_stale_path_preview", None), dict)
            and getattr(self, "_stale_path_preview", {}).get("full_path")
        ) or isinstance(getattr(self, "_machine_route_preview", None), dict)
        boundary = getattr(self.state, "field_boundary", []) or []
        has_boundary = len(boundary) >= 3
        has_forbidden = bool(getattr(self.state, "forbidden_regions", []) or [])
        has_service = bool(
            getattr(self.state, "entry_point", None)
            or getattr(self.state, "exit_point", None)
            or getattr(self.state, "unload_points", [])
            or (has_route and getattr(self.state, "auto_path_planned", False))
        )
        has_reference = self._reference_lines_present()
        availability = {
            "field_boundary": (has_boundary, has_boundary),
            # Keep the current processed view uncluttered by default, but keep
            # the raw row available because the raw evidence still exists and
            # can be composed with the processed mask on demand.
            "raw_mask": (has_raw, not has_processed),
            "processed_mask": (has_processed, has_processed),
            "headland_mask": (has_headland, has_headland),
            "uncertain_residual": (has_uncertain, has_uncertain),
            "centerlines": (has_centerlines, has_centerlines),
            # DECISION-063：参考线是研究工具的标注证据，不该常驻画布。
            # 有数据就把行亮起来（可勾选），但默认不画；用户勾选图层或进入
            # 参考线标注模式时才显示。
            "reference_lines": (has_reference, False),
            "route": (has_route, has_route),
            "forbidden_regions": (has_forbidden, has_forbidden),
            "service_points": (
                has_service,
                has_service and self._service_points_should_show(),
            ),
        }
        for key in DISPLAY_LAYER_KEYS:
            available, default_visible = availability[key]
            self.image_view.set_display_layer_available(
                key,
                available,
                default_visible=default_visible,
            )

    def _on_display_layer_changed(self, key, visible):
        """Apply a display toggle without mutating AppState or provenance."""
        key = str(key or "").strip()
        visible = bool(visible)
        viewer = self.image_view.viewer
        viewer.set_overlay_layer_visible(key, visible)
        if visible and not self._show_mask_before and not viewer.has_overlay_layer(key):
            # A row can be enabled just before a stage's normal redraw.  Make
            # the first click useful while keeping the redraw idempotent.
            self._restore_visual_from_state()

    def _restore_visual_from_state(self):
        self.top_toolbar.refresh_actions(self.state)
        if self._sanitize_centerline_state_to_field():
            self._invalidate_analysis_from(
                "path",
                "缓存中存在田外中心线，已移除并要求重新规划路径",
            )
            # The invalidation helper retains a stale preview for ordinary
            # edits.  A cached line removed by this hard boundary gate must
            # not remain visible in that preview.
            self._stale_path_preview = None
        self._refresh_centerline_selector_panel()
        self._sync_display_layer_availability()
        if (
            getattr(self, "_route_display_mode", "continuous") == "machine_file"
            and isinstance(getattr(self, "_machine_route_preview", None), dict)
        ):
            self._show_machine_route_preview(self._machine_route_preview)
            return
        path_result = self._path_result_from_state()
        if getattr(self.state, 'auto_path_planned', False) and path_result:
            self._pipeline_result = {"path": path_result}
            self._show_path(path_result)
            return
        stale_preview = getattr(self, "_stale_path_preview", None)
        if isinstance(stale_preview, dict) and stale_preview.get("full_path"):
            self._pipeline_result = None
            self._show_path(stale_preview, stale_preview=True)
            return
        result = getattr(self.state, 'mask_result', None)
        if isinstance(result, dict) and isinstance(result.get("processed_mask"), np.ndarray):
            self._show_mask_overlay(result["processed_mask"])
            return
        raw = getattr(self.state, 'mask_raw', None)
        if isinstance(raw, np.ndarray):
            self._show_mask_overlay(raw)
            return
        if self._tif_rgb is not None:
            # DECISION-004: even the empty-state redraw used by undo/redo is
            # a layer refresh, not a navigation command.
            viewer = self.image_view.viewer
            view_state = (
                viewer.capture_view_state()
                if getattr(viewer, "_pixmap", None) is not None
                else None
            )
            self.image_view.set_image(self._tif_rgb)
            if view_state is not None:
                viewer.restore_view_state(view_state)
        else:
            self.image_view.viewer.clear_overlays()
        self._redraw_field_if_any()
        self._draw_reference_lines_overlay()

    @Slot(str)
    def _on_step_deleted(self, tid):
        """Rollback a workflow step after an explicit user confirmation."""
        if tid not in TASK_ORDER:
            return
        label = TASK_LABELS.get(tid, tid)

        if QMessageBox.question(
            self,
            f"确认回退“{label}”",
            rollback_summary(tid),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self._rollback_from_task(tid)

    def _rollback_from_task(self, tid):
        try:
            idx = TASK_ORDER.index(tid)
        except ValueError:
            return
        self.state.save_undo(f"rollback {tid}")
        affected = set(TASK_ORDER[idx:])

        if "import_img" in affected:
            self._tif_rgb = None
            self._tif_path = None
            self._tif_source_identity = {}
            self._pipeline_result = None
            self.state.tif_path = ""
            self.state.img_h = 0
            self.state.img_w = 0
            self.state.display_h = 0
            self.state.display_w = 0
            self.state.downsample_factor = 1
            self.state.display_scale_x = 1.0
            self.state.display_scale_y = 1.0
            self.state.field_boundary = []
            self.state.forbidden_regions = []
            self.state.forbidden_regions_confirmed = False
            self.state.mask_raw = None
            self.state.inference_original_mask = None
            self.state.mask_result = None
            self.state.inference_done = False
            self.state.mask_processed = False
            self._field_pts.clear()
            self.geo.clear()
            self.top_toolbar.set_filename("")
            self.image_view.clear_image()
        elif "draw_field" in affected:
            self.state.field_boundary = []
            self.state.forbidden_regions = []
            self.state.forbidden_regions_confirmed = False
            self._field_pts.clear()
            self._field_drawing = False
            self._sync_drawing_mode()
            self.state.inference_done = False
            self.state.mask_raw = None
            self.state.inference_original_mask = None
            self.state.mask_processed = False
            self.state.mask_result = None
            self._pipeline_result = None
            self._stale_path_preview = None
        elif "segment" in affected:
            self.state.inference_done = False
            self.state.mask_raw = None
            self.state.inference_original_mask = None
            self.state.mask_processed = False
            self.state.mask_result = None
            self._pipeline_result = None
            self._stale_path_preview = None
        elif "process" in affected:
            self.state.mask_processed = False
            self.state.mask_result = None
            self._stale_path_preview = None

        if any(step in affected for step in ("import_img", "draw_field", "segment", "process")):
            self.state.safe_update(
                centerline_catalog={},
                selected_centerline_ids=[],
                centerline_selection_report={},
                work_band_catalog={},
                selected_work_band_ids=[],
                work_band_selection_report={},
            )
            self.task_panel.clear_centerline_selection()

        if "centerline_generate" in affected:
            self.state.safe_update(
                centerline_catalog={},
                selected_centerline_ids=[],
                centerline_selection_report={},
                work_band_catalog={},
                selected_work_band_ids=[],
                work_band_selection_report={},
            )
            self.task_panel.clear_centerline_selection()
        elif "centerline_select" in affected:
            catalog = getattr(self.state, "centerline_catalog", None) or {}
            self.state.safe_update(
                selected_centerline_ids=[],
                selected_work_band_ids=[],
                centerline_selection_report={
                    "algorithm_version": "operator-centerline-selection-v1",
                    "selection_origin": "rollback",
                    "selected_ids": [],
                    "selected_count": 0,
                    "candidate_count": len(catalog.get("items") or []),
                    "selection_mode": "empty",
                },
                work_band_selection_report={
                    "algorithm_version": "operator-centerline-selection-v1",
                    "selection_origin": "rollback",
                    "selected_ids": [],
                    "selected_count": 0,
                    "candidate_count": len(catalog.get("items") or []),
                    "selection_mode": "empty",
                },
            )
            self.task_panel.update_centerline_selection(catalog, [])

        if "params" in affected:
            self.state.harvester_params = {}
            self.state.harvester_param_sources = {}
        if "entry" in affected:
            self.state.entry_point = None
            self.state.entry_point_locked = False
        if "exit" in affected:
            self.state.exit_point = None
            self.state.exit_point_locked = False
        if "unload" in affected:
            self.state.unload_points = []
            self.state.unload_point = None
            self.state.unload_point_locked = False
        if "plan" in affected:
            self.state.auto_path_planned = False
            self.state.auto_path = []
            self.state.auto_path_segments = []
            self.state.auto_path_geo = []
            self.state.auto_path_valid = False
            self.state.auto_path_desc = ""
            self.state.path_points = []
            self.state.path_status = []
            self.state.path_geometry_modes = []
            self._pipeline_result = None
            self._stale_path_preview = None
            self._clear_machine_route_preview()
            self.route_info.clear_info()
        if "simulate" in affected:
            if self._anim_timer:
                self._anim_timer.stop()
            self._anim_timer = None
            self._anim_engine = None
            self.state.anim_active = False
            self.state.anim_paused = False
            self.state.simulation_done = False
        if "export" in affected:
            self.state.export_done = False

        if tid != "import_img":
            self._service_points_visible = bool(
                getattr(self.state, 'entry_point', None)
                or getattr(self.state, 'exit_point', None)
                or getattr(self.state, 'unload_points', None)
            )
            self._restore_visual_from_state()
            self._refresh_service_points_panel()
        self._sync_task_statuses()
        self.top_toolbar.refresh_actions(self.state)
        self.log_panel.info(f"已回退到“{TASK_LABELS.get(tid, tid)}”之前")


    def _sync_task_statuses(self):
        """Render task panel from the actual domain state (single source of truth)."""
        statuses = derive_task_statuses(self.state, has_image=self._tif_rgb is not None)
        self.task_panel.apply_task_statuses(statuses)
        # ``derive_task_statuses`` intentionally knows only domain stages;
        # model deserialisation is a transient runtime task.  Re-apply this
        # UI lock after every status refresh so a timer or another callback
        # cannot re-enable model selection while the worker is still active.
        if getattr(self, "_model_load_in_progress", False):
            self._set_model_loading_ui(True)
        if getattr(self.state, "anim_active", False):
            self.task_panel.set_task_action(
                "simulate",
                "继续" if getattr(self.state, "anim_paused", False) else "暂停",
            )

    def _invalidate_analysis_from(self, stage, reason=""):
        """DECISION-007: invalidate every result derived from changed input.

        The dependency order is inference -> mask -> path -> simulation/export.
        Keeping a later artifact after an earlier input changes would make the
        displayed route disagree with the data used to calculate it.
        """
        stage = str(stage or "path")
        if stage not in {"inference", "mask", "path"}:
            raise ValueError(f"unknown invalidation stage: {stage}")

        self._stash_stale_path_preview(reason)
        self._clear_machine_route_preview()

        updates = {}
        if stage == "inference":
            updates.update(
                mask_raw=None,
                inference_original_mask=None,
                inference_done=False,
                inference_provenance={},
                inference_runtime={},
                mask_provenance={},
                path_provenance={},
                path_runtime={},
                workflow_step=0,
            )
        if stage in {"inference", "mask"}:
            self.task_panel.clear_centerline_selection()
            updates.update(
                mask_processed=False,
                mask_result=None,
                mask_provenance={},
                centerline_catalog={},
                selected_centerline_ids=[],
                centerline_selection_report={},
                work_band_catalog={},
                selected_work_band_ids=[],
                work_band_selection_report={},
                path_provenance={},
                path_runtime={},
                entry_point=None,
                exit_point=None,
                unload_points=[],
                unload_point=None,
                entry_point_locked=False,
                exit_point_locked=False,
                unload_point_locked=False,
                workflow_step=0 if stage == "inference" else 1,
            )
            self._service_points_visible = False

        updates.update(
            auto_path=[],
            auto_path_segments=[],
            auto_path_geo=[],
            auto_path_planned=False,
            auto_path_valid=False,
            auto_path_desc="",
            path_provenance={},
            path_runtime={},
            path_validation={},
            path_artifact={},
            path_points=[],
            path_status=[],
            path_geometry_modes=[],
            simulation_done=False,
            export_done=False,
            last_total_path_m=0.0,
            last_work_path_m=0.0,
            last_turn_path_m=0.0,
            last_entry_exit_path_m=0.0,
        )
        if stage == "path":
            updates["workflow_step"] = min(int(getattr(self.state, "workflow_step", 2)), 2)

        if self._anim_timer:
            self._anim_timer.stop()
        self._anim_timer = None
        self._anim_engine = None
        self.image_view.set_animation_controls_visible(False)
        self.lbl_speed.clear()
        self.state.safe_update(**updates)
        self._pipeline_result = None
        self.route_info.clear_info()
        self._sync_task_statuses()
        # 状态变了就必须同步画布。这里不指望 25 个调用点都记得重绘：漏一个就是
        # "左边清单说没有了、右边画布还画着"（用户反复反馈的左右不同步）。
        # 交互式绘制/拖拽期间不动画布，避免和正在进行的绘制抢图层。
        if not (
            getattr(self, "_field_drawing", False)
            or getattr(self, "_forbidden_drawing", False)
            or int(getattr(self, "_route_drag_idx", -1) or -1) >= 0
        ):
            self._repair_canvas_if_state_disagrees(where=f"invalidate:{stage}")
        if reason:
            self.log_panel.info(f"已使下游结果失效：{reason}")

    def _on_mask_strength_changed(self, strength_key):
        # Compatibility hook for older panels/cached UI state.  The current
        # application has one fixed mask-processing profile.
        strength_key = MASK_PROCESSING_STRENGTH
        if strength_key == self._mask_processing_strength:
            return
        self._mask_processing_strength = strength_key
        config_snapshot = self.cfg.update_section("mask_processing", {"strength": strength_key})
        save_json_atomic(self.cfg.config_path(), config_snapshot)
        self.task_panel.set_mask_strength(strength_key)
        if getattr(self.state, 'mask_processed', False):
            self._invalidate_analysis_from(
                "mask",
                "掩膜优化参数已变化，必须重新处理",
            )
        else:
            self.log_panel.info("掩膜优化强度固定为：标准")

    def _ensure_field_trial_planning_mode(self):
        current = self.cfg.section("path_planning")
        desired = {
            # The default desktop route follows the extracted four-row
            # centerlines and reports the final footprint metric.  Lateral
            # footprint optimization remains an explicit low-level mode, not
            # an implicit step in the default field-trial route.
            "work_line_mode": "band_centerline_with_footprint_validation",
            "validation_profile": "field_trial",
            "footprint_metric_policy": "report_only",
            # The operator drives the machine to the field edge manually.
            # Service points select route order/direction and are not exported
            # as artificial connector lines.
            "service_point_mode": "reference_only",
        }
        if all(current.get(key, value) == value for key, value in desired.items()):
            return
        snapshot = self.cfg.update_section("path_planning", desired)
        save_json_atomic(self.cfg.config_path(), snapshot)

    def _on_turn_strategy_changed(self, strategy_key):
        """DECISION-007: a new turn strategy invalidates the planned route."""
        self.state.turn_strategy_user_selected = True
        if strategy_key == getattr(self.state, 'turn_strategy', None):
            self.state._persist_history_restore("turn_strategy")
            return
        self._save_system_undo("切换调头方式")
        had_path = bool(getattr(self.state, 'auto_path_planned', False))
        self.state.turn_strategy = strategy_key
        labels = {
            "auto": "自动推荐",
            "bow": "弓形调头",
            "semicircle": "半圆调头",
            "pear": "梨形调头",
            "fishtail": "鱼尾折返",
            "alpha": "紧凑 Alpha",
            "contour": "套圈收割",
        }
        self.task_panel.set_turn_strategy(strategy_key)
        self.top_toolbar.set_current_strategy(strategy_key)
        if had_path:
            self._invalidate_analysis_from(
                "path",
                f"调头方式已切换为 {labels.get(strategy_key, strategy_key)}，必须重新规划",
            )
        else:
            self.log_panel.info(f"调头方式: {labels.get(strategy_key, strategy_key)}")
        self.state._persist_history_restore("turn_strategy")

    def _on_turn_strategy_btn(self):
        strategies = ["自动", "弓形", "半圆", "梨形", "鱼尾", "Alpha", "套圈收割"]
        keys = ["auto", "bow", "semicircle", "pear", "fishtail", "alpha", "contour"]
        cur_key = getattr(self.state, 'turn_strategy', 'auto')
        cur_idx = keys.index(cur_key) if cur_key in keys else 0
        item, ok = QInputDialog.getItem(self, "转弯策略", "请选择转弯策略:", strategies, cur_idx, False)
        if ok and item:
            idx = strategies.index(item)
            self._on_turn_strategy_changed(keys[idx])

    def _on_toggle_field_drawing(self):
        if not self._field_drawing:
            self._forbidden_drawing = False
            self._forbidden_pts = []
        self._field_drawing = not self._field_drawing
        self._sync_drawing_mode()
        if self._field_drawing:
            self._field_pts = []
            self.log_panel.info("田块圈选模式：点击添加顶点，点击首点闭合")
        else:
            self._field_pts = []
            self._clear_field_overlay()
            self.log_panel.info("已关闭田块圈选")
    def _on_viewer_mouse_pressed(self, x, y, button):
        if self._measure_tool and self._handle_measure_click(x, y, button):
            return
        if self._route_edit_active and self._handle_route_edit_press(x, y, button):
            return
        if self._field_drawing:
            if button == 1:
                if len(self._field_pts) >= 3:
                    first = self._field_pts[0]
                    close_r = max(14, self._field_vertex_radius() + 4)
                    if math.hypot(x - first[0], y - first[1]) <= close_r:
                        self._finish_field_polygon()
                        return
                self._field_pts.append((x, y))
                self._redraw_field_overlay()
            elif button == 2:
                self._field_pts = []
                self._clear_field_overlay()
                self.log_panel.info("已取消田块圈选")
            return
        if self._forbidden_drawing:
            if button == 1:
                if len(self._forbidden_pts) >= 3:
                    first = self._forbidden_pts[0]
                    if math.hypot(x - first[0], y - first[1]) <= 14:
                        self._finish_forbidden_polygon()
                        return
                self._forbidden_pts.append((x, y))
                self._redraw_forbidden_overlay(mouse_pos=(x, y))
            elif button == 2:
                self._forbidden_pts = []
                self._forbidden_drawing = False
                self._sync_drawing_mode()
                self._restore_visual_from_state()
                self.log_panel.info("已取消本次禁行区绘制")
            return
        if self._reference_drawing:
            self._handle_reference_click(x, y, button)
            return
        if self._calibration_drawing:
            self._handle_calibration_press(x, y, button)
            return
        if button == 1 and self._begin_service_point_drag(x, y):
            return
        if getattr(self.state, 'entry_exit_mode', False):
            if button == 1:
                self._place_entry_exit_point(x, y)
            return
        self.event_bridge.on_mouse_pressed(x, y, button)

    def _on_viewer_mouse_released(self, x, y, button):
        if self._calibration_drawing and self._calibration_press is not None:
            self._handle_calibration_release(x, y)
            return
        if self._route_edit_active and self._route_drag_work_index >= 0:
            self._route_drag_work_index = -1
            self._route_drag_work_origin = None
            self._route_drag_work_points = None
            self._sync_auto_path_from_editable_route()
            self._draw_route_edit_overlay()
            self._update_route_edit_cursor(x, y)
            self.log_panel.info(
                "整条作业线已平移；前后控制端点和相邻转弯已同步更新"
            )
            return
        if self._route_edit_active and self._route_drag_idx >= 0:
            self._route_drag_idx = -1
            self._sync_auto_path_from_editable_route()
            self._draw_route_edit_overlay()
            self._update_route_edit_cursor(x, y)
            self.log_panel.info(
                "路线端点已更新；相邻作业线与橙色调头弧线已重新衔接"
            )
            return
        if button == 1 and getattr(self.state, "_drag_entry_exit", None) is not None:
            self._finish_service_point_drag(x, y)
            return
        self.event_bridge.on_mouse_released(x, y, button)

    def _on_field_double_click(self, x, y):
        return

    def _on_viewer_mouse_moved(self, x, y, buttons):
        if (
            self._route_edit_active
            and self._route_drag_work_index >= 0
            and (buttons & 1)
            and self._route_drag_work_origin is not None
            and self._route_drag_work_points is not None
        ):
            records = self._route_edit_work_records()
            if self._route_drag_work_index < len(records):
                record = records[self._route_drag_work_index]
                dx = float(x) - float(self._route_drag_work_origin[0])
                dy = float(y) - float(self._route_drag_work_origin[1])
                points = list(self._route_drag_work_points)
                for node_index in range(
                    record["node_start"], record["node_end"] + 1
                ):
                    source = self._route_drag_work_points[node_index]
                    points[node_index] = (
                        float(source[0]) + dx,
                        float(source[1]) + dy,
                    )
                self.state.path_points = points
                self._draw_route_edit_dynamic_overlay(lightweight=True)
            return
        if self._route_edit_active and self._route_drag_idx >= 0 and (buttons & 1):
            self._route_edit_move_point(self._route_drag_idx, (x, y), save_undo=False)
            self._draw_route_edit_dynamic_overlay(lightweight=True)
            return
        if self._route_edit_active and not buttons:
            self._update_route_edit_cursor(x, y)
            return
        if getattr(self.state, "_drag_entry_exit", None) is not None and (buttons & 1):
            self._move_service_point_drag(x, y)
            return
        if self._calibration_drawing and self._calibration_press is not None:
            if self._handle_calibration_move(x, y, buttons):
                return
        if self._field_drawing and self._field_pts:
            self._redraw_field_overlay(mouse_pos=(x, y))
        elif self._forbidden_drawing and self._forbidden_pts:
            self._redraw_forbidden_overlay(mouse_pos=(x, y))
        elif self._reference_drawing and self._reference_pts:
            self._redraw_reference_interaction(mouse_pos=(x, y))
        self.event_bridge.on_mouse_moved(x, y, buttons)

    def _field_line_width(self):
        return 4

    def _field_vertex_radius(self):
        return 7

    def _redraw_field_overlay(self, mouse_pos=None):
        viewer = self.image_view.viewer
        # The in-progress polygon is an interaction layer.  Keep the base TIF
        # and completed analysis layers intact while the user is clicking.
        viewer.clear_overlay_layer("interaction_field")
        viewer.clear_overlay_layer("field_boundary")
        viewer.begin_overlay_layer("interaction_field")
        pts = self._field_pts
        lw = self._field_line_width()
        vr = self._field_vertex_radius()
        try:
            for i in range(len(pts) - 1):
                viewer.draw_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                                 color=FIELD_BOUNDARY_COLOR_BGR, width=lw)
            if mouse_pos and pts:
                viewer.draw_line(pts[-1][0], pts[-1][1], mouse_pos[0], mouse_pos[1],
                                 color=FIELD_BOUNDARY_COLOR_BGR, width=max(3, lw // 2), style=Qt.DashLine)
            for i, (px, py) in enumerate(pts):
                c = (255, 255, 0) if i == 0 else FIELD_BOUNDARY_COLOR_BGR
                viewer.draw_circle(px, py, radius=vr, color=c)
        finally:
            viewer.end_overlay_layer()

    def _clear_field_overlay(self):
        viewer = self.image_view.viewer
        viewer.clear_overlay_layer("interaction_field")
        boundary = getattr(self.state, "field_boundary", []) or []
        if len(boundary) >= 3:
            self._draw_field_boundary()

    def _finish_field_polygon(self):
        if len(self._field_pts) < 3:
            self.log_panel.error("至少需要 3 个点")
            return
        points, geometry_report = self._normalise_field_boundary(self._field_pts)
        if points is None:
            self.log_panel.error(
                "田块边界自相交，且截掉的面积超过允许范围，无法在不改变圈选范围的前提下修复；"
                "请重新圈选田块边界"
            )
            return
        area_m2 = self._field_area_m2(points)
        had_analysis = bool(
            getattr(self.state, "inference_done", False)
            or getattr(self.state, "mask_processed", False)
            or getattr(self.state, "auto_path_planned", False)
        )
        self._save_system_undo("田块圈选")
        self.state.safe_update(
            field_boundary=list(points),
            field_area_m2=area_m2,
            forbidden_regions=[],
            forbidden_regions_confirmed=False,
        )
        if had_analysis:
            self._invalidate_analysis_from(
                "inference",
                "田块边界已变化，必须重新识别、处理和规划",
            )
        self._field_drawing = False
        self._sync_drawing_mode()
        self._draw_field_boundary()
        area_info = self._estimate_field_area(points)
        if geometry_report.get("repaired"):
            self.log_panel.info(
                "田块边界闭合处存在重复点击形成的自相交，已按圈选轮廓自动修复"
                f"（丢弃面积 {float(geometry_report.get('dropped_area_ratio', 0.0)) * 100:.4f}%）"
            )
        self.log_panel.success(f"田块已设置: {len(points)} 个顶点 | 面积 = {area_info}")
        self._journal.info(f"田块已设置: {len(points)} 个顶点, 面积={area_info}")
        self._log_action(
            "field.finish", "",
            vertices=len(points),
            area_m2=round(float(area_m2), 3),
            dropped_area_ratio=round(
                float(geometry_report.get("dropped_area_ratio", 0.0) or 0.0), 6
            ),
        )
        self.task_panel.set_task_status("draw_field", "done")
        self._sync_task_statuses()
        WorkflowUpdater.advance(self.state, "BOUNDARY_SET")

    @staticmethod
    def _normalise_field_boundary(points):
        """Return a simple field ring plus the repair report, or ``(None, report)``.

        A double click on the closing vertex appends duplicate vertices a few
        pixels from the first one and turns the ring into a bow-tie.  Every
        shapely-based boundary gate then rejects everything, which surfaced much
        later as "无有效中心线".  Repair that closing loop here so the stored
        boundary is the ring the operator actually meant.

        The result stays a plain list of ``(x, y)`` tuples: ``field_boundary`` is
        persisted, undone and compared as ordinary JSON-like data.
        """
        from row_geometry import sanitize_field_polygon_report

        polygon, report = sanitize_field_polygon_report(points)
        if polygon is None:
            return None, report
        return [(float(point[0]), float(point[1])) for point in polygon], report

    @staticmethod
    def _polygon_area_px(pts):
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def _estimate_field_area(self, pts):
        area_px = self._polygon_area_px(pts)
        if self.geo.is_ready():
            area_m2 = self._field_area_m2(pts)
            if area_m2 > 0:
                if area_m2 >= 10000:
                    return f"{area_m2 / 10000:.2f} ha ({area_m2:.0f} m2)"
                return f"{area_m2:.1f} m2"
        return f"{area_px:.0f} px2"

    def _field_area_m2(self, pts):
        if not pts or len(pts) < 3:
            return 0.0
        if self.geo.is_ready():
            try:
                area_m2 = float(self.geo.pixel_polygon_area_m2(pts))
                if math.isfinite(area_m2) and area_m2 > 0:
                    return area_m2
            except Exception:
                pass
            try:
                arr = np.asarray(pts, dtype=np.float64)
                cx, cy = np.mean(arr[:, 0]), np.mean(arr[:, 1])
                ppm = self.geo.pixels_per_meter(cx, cy)
                if ppm > 0:
                    return self._polygon_area_px(pts) / (ppm * ppm)
            except Exception:
                pass
        return 0.0

    def _draw_field_boundary(self):
        boundary = getattr(self.state, 'field_boundary', [])
        viewer = self.image_view.viewer
        viewer.clear_overlay_layer("field_boundary")
        if not boundary or len(boundary) < 3:
            return
        lw = self._field_line_width()
        vr = self._field_vertex_radius()
        viewer.begin_overlay_layer("field_boundary")
        try:
            for i in range(len(boundary)):
                p1 = boundary[i]
                p2 = boundary[(i + 1) % len(boundary)]
                viewer.draw_line(p1[0], p1[1], p2[0], p2[1], color=FIELD_BOUNDARY_COLOR_BGR, width=lw)
            for px, py in boundary:
                viewer.draw_circle(px, py, radius=vr, color=(255, 255, 0))
        finally:
            viewer.end_overlay_layer()
        self._draw_forbidden_regions(clear_first=False)

    def _on_toggle_forbidden_drawing(self):
        if not getattr(self.state, "field_boundary", None):
            self.log_panel.warn("请先完成田块圈选，再绘制禁行区。")
            return
        self._field_drawing = False
        self._field_pts = []
        self._forbidden_drawing = not self._forbidden_drawing
        self._forbidden_pts = []
        self._sync_drawing_mode()
        self._restore_visual_from_state()
        if self._forbidden_drawing:
            self.log_panel.info("禁行区绘制：点击添加顶点，点击首点闭合；右键取消")

    def _on_confirm_no_forbidden(self):
        regions = list(getattr(self.state, "forbidden_regions", []) or [])
        if regions:
            self.log_panel.info(f"已有禁行区：当前已经绘制 {len(regions)} 处禁行区。")
            return
        if getattr(self.state, "forbidden_regions_confirmed", False):
            self.log_panel.info("已确认田块内无已知禁行区")
            return
        self._save_system_undo("确认无禁行区")
        had_path = bool(getattr(self.state, "auto_path_planned", False))
        self.state.safe_update(forbidden_regions_confirmed=True)
        if had_path:
            self._invalidate_analysis_from("path", "禁行区确认状态已变化，必须重新规划")
        self.task_panel.set_forbidden_state(0, True)
        self.log_panel.success("已确认田块内无已知禁行区")

    def _redraw_forbidden_overlay(self, mouse_pos=None):
        self._restore_visual_from_state()
        viewer = self.image_view.viewer
        viewer.clear_overlay_layer("interaction_forbidden")
        viewer.begin_overlay_layer("interaction_forbidden")
        pts = self._forbidden_pts
        try:
            for index in range(len(pts) - 1):
                viewer.draw_line(*pts[index], *pts[index + 1], color=(40, 40, 230), width=3)
            if mouse_pos and pts:
                viewer.draw_line(*pts[-1], *mouse_pos, color=(40, 40, 230), width=2, style=Qt.DashLine)
            for index, point in enumerate(pts):
                color = (255, 255, 0) if index == 0 else (40, 40, 230)
                viewer.draw_circle(*point, radius=7, color=color)
        finally:
            viewer.end_overlay_layer()

    def _finish_forbidden_polygon(self):
        if len(self._forbidden_pts) < 3:
            return
        field = np.asarray(getattr(self.state, "field_boundary", []), dtype=np.float32)
        if any(
            cv2.pointPolygonTest(field, (float(x), float(y)), False) < 0
            for x, y in self._forbidden_pts
        ):
            QMessageBox.warning(self, "禁行区越界", "禁行区必须完全位于已确认的田块边界内。")
            return
        self._save_system_undo("新增禁行区")
        regions = list(getattr(self.state, "forbidden_regions", []) or [])
        regions.append(list(self._forbidden_pts))
        had_path = bool(getattr(self.state, "auto_path_planned", False))
        self.state.safe_update(
            forbidden_regions=regions,
            forbidden_regions_confirmed=True,
        )
        self._forbidden_pts = []
        self._forbidden_drawing = False
        self._sync_drawing_mode()
        if had_path:
            self._invalidate_analysis_from("path", "禁行区已变化，必须重新规划和验证")
        self._restore_visual_from_state()
        self.log_panel.success(f"禁行区已确认：共 {len(regions)} 个区域")

    def _on_clear_forbidden_regions(self):
        regions = list(getattr(self.state, "forbidden_regions", []) or [])
        if not regions:
            self.log_panel.info("当前没有禁行区")
            return
        if QMessageBox.question(
            self, "清除禁行区", "确定清除全部已确认禁行区吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self._save_system_undo("清除禁行区")
        had_path = bool(getattr(self.state, "auto_path_planned", False))
        self.state.safe_update(forbidden_regions=[], forbidden_regions_confirmed=False)
        if had_path:
            self._invalidate_analysis_from("path", "禁行区已清除，必须重新规划和验证")
        self._restore_visual_from_state()

    def _draw_forbidden_regions(self, clear_first=False):
        viewer = self.image_view.viewer
        # ``clear_first`` is kept for callers of the old API; clearing is now
        # scoped to this persistent layer so masks and routes survive.
        viewer.clear_overlay_layer("forbidden_regions")
        viewer.begin_overlay_layer("forbidden_regions")
        try:
            for index, region in enumerate(getattr(self.state, "forbidden_regions", []) or []):
                if len(region) < 3:
                    continue
                viewer.draw_polygon(
                    region, color=(35, 35, 220), outline=(20, 20, 255),
                    width=2, alpha=75,
                )
                center = np.mean(np.asarray(region, dtype=np.float64), axis=0)
                viewer.draw_text(
                    float(center[0]), float(center[1]), f"禁行 {index + 1}",
                    color=(245, 245, 245), size=10,
                )
        finally:
            viewer.end_overlay_layer()

    def _redraw_field_if_any(self):
        boundary = getattr(self.state, 'field_boundary', [])
        if boundary and len(boundary) >= 3:
            self._draw_field_boundary()
        else:
            self.image_view.viewer.clear_overlay_layer("field_boundary")

    # ── 参考线标注（DECISION-028：中心线评价的人工基准，研究证据） ──

    def _set_reference_mode(self, pass_index):
        """进入/退出参考线标注模式；与田块/禁行区绘制模式互斥。"""
        pass_index = int(pass_index)
        if self._reference_drawing and self._reference_pass == pass_index:
            self._exit_reference_mode()
            return
        if not str(getattr(self.state, "tif_path", "") or ""):
            self.log_panel.warn("请先导入影像，再标注参考线。")
            return
        if self._field_drawing:
            self._field_drawing = False
            self._field_pts = []
            self._clear_field_overlay()
        if self._forbidden_drawing:
            self._forbidden_drawing = False
            self._forbidden_pts = []
            self.image_view.viewer.clear_overlay_layer("interaction_forbidden")
        if self._reference_drawing:
            self.image_view.viewer.clear_overlay_layer("interaction_reference")
        if self._reference_drawing and self._reference_pass != pass_index:
            # 切到另一遍之前先把手上的线存好，不能因为换遍数丢掉。
            if not self._finalize_pending_reference("切换标注遍数"):
                return
        self._reference_drawing = True
        self._reference_pass = pass_index
        self._reference_pts = []
        self._reference_last_click = None
        # 进入模式时重读 sidecar，保证与磁盘一致（离线工具可能已更新）
        self._reference_payload = {}
        self._reference_payload_tif = ""
        draft = self._load_reference_draft(pass_index)
        draft_points = [
            (float(point[0]), float(point[1]))
            for point in (draft.get("points") or [])
            if len(point) >= 2
        ]
        if draft_points:
            self._reference_pts = draft_points
            self.log_panel.info(
                "已恢复上次未完成的参考线：第 %d 遍，%d 个桩点。继续加点，双击或点"
                "“完成当前参考线”保存。"
                % (pass_index, len(draft_points))
            )
            self._log_action(
                "reference.restore_draft", "", pass_index=pass_index, points=len(draft_points)
            )
        self._sync_drawing_mode()
        self._draw_reference_lines_overlay()
        self._redraw_reference_interaction()
        self._sync_display_layer_availability()
        # DECISION-063：进模式才把参考线图层打开（默认关闭，避免常驻画布）。
        # 必须放在 _sync_display_layer_availability 之后，否则会被默认值覆盖。
        self._set_reference_layer_visible(True)
        existing = len(self._reference_lines_payload().get("lines") or [])
        self.log_panel.info(
            "参考线标注（第 %d 遍）：左键沿宽行带中心逐点添加桩点（建议每 5 m 一点，"
            "一条带 10-16 个点），最后一个桩点上双击完成并自动保存；也可以点菜单"
            "“完成当前参考线”保存；右键撤销上一个桩点（Ctrl+右键丢弃整条）；再次点击同一条菜单退出"
            "（退出前会自动保存已画好的线）。保存后橙色实线按桩点连、暗色细虚线是"
            "拟合基准线（评价用），两者分开说明桩点不在一条直线上。当前已有 %d 条参考线。"
            % (self._reference_pass, existing)
        )
        self._log_action(
            "reference.mode", "enter", pass_index=pass_index, existing=existing
        )

    def _load_reference_draft(self, pass_index):
        tif = str(getattr(self.state, "tif_path", "") or "")
        if not tif:
            return {}
        import cache

        draft = cache.load_reference_draft(tif, int(pass_index))
        if not isinstance(draft, dict) or not draft.get("points"):
            return {}
        if int(draft.get("pass", pass_index) or pass_index) != int(pass_index):
            return {}
        return draft

    def _persist_reference_draft(self):
        """每加一个桩点就落盘：退出、切菜单、崩溃都不会丢正在画的线。"""
        tif = str(getattr(self.state, "tif_path", "") or "")
        if not tif:
            return
        import cache

        cache.save_reference_draft(
            tif,
            {
                "schema": "reference-draft-v1",
                "tif_path": tif,
                "pass": int(self._reference_pass),
                "points": [[float(x), float(y)] for x, y in self._reference_pts],
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        )

    def _clear_reference_draft(self, pass_index=None):
        tif = str(getattr(self.state, "tif_path", "") or "")
        if not tif:
            return
        import cache

        cache.clear_reference_draft(
            tif, int(self._reference_pass if pass_index is None else pass_index)
        )

    def _exit_reference_mode(self):
        """退出标注：有待完成桩点先自动保存，绝不静默丢弃。"""
        pending = len(self._reference_pts)
        if not self._finalize_pending_reference("退出"):
            return
        self._reference_drawing = False
        self._reference_pts = []
        self._reference_last_click = None
        self.image_view.viewer.clear_overlay_layer("interaction_reference")
        self._sync_drawing_mode()
        # DECISION-063：退出标注就把参考线收起来，画布回到干净状态；
        # 需要看/复核时从“参考线标注 → 显示 / 隐藏参考线图层”或图层面板打开。
        self._set_reference_layer_visible(False)
        self._log_action(
            "reference.exit", "", pass_index=self._reference_pass, pending=pending
        )

    def _finalize_pending_reference(self, reason=""):
        """离开参考线模式前的收口：2 点以上自动保存，1 点丢弃并留痕。"""
        pending = len(self._reference_pts)
        if pending >= 2:
            if not self._finish_reference_line():
                self.log_panel.error(
                    "参考线保存失败，已保留当前 %d 个桩点；请重试或检查缓存目录权限。"
                    % pending
                )
                return False
            self.log_panel.info(
                "%s前已自动保存当前参考线（%d 个桩点）" % (reason or "离开", pending)
            )
            return True
        if pending == 1:
            self.log_panel.warn("只有 1 个桩点，至少需要 2 个，未保存。")
            self._clear_reference_draft()
            self._log_action(
                "reference.discard", "single_point",
                pass_index=self._reference_pass, points=1,
            )
        return True

    @staticmethod
    def _reference_cancel_modifier_active():
        """Ctrl+右键＝丢弃整条未保存的参考线（普通右键只撤一个桩点）。"""
        try:
            return bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        except Exception:
            return False

    def _handle_reference_click(self, x, y, button):
        """参考线标注模式下的一次点击：左键加桩点，双击收线，右键撤销一个桩点。"""
        if button == 2:
            # 右键 ＝ **撤销最后 1 个桩点**。旧行为是一键丢弃整条未保存的线，
            # 用户 2026-10-26 实测误触一次就丢了 34 个桩点。整条丢弃改为
            # Ctrl+右键（明确意图），空线状态下右键＝退出标注。
            if self._reference_cancel_modifier_active():
                discarded = len(self._reference_pts)
                self._reference_pts = []
                self._reference_last_click = None
                self._clear_reference_draft()
                self.image_view.viewer.clear_overlay_layer("interaction_reference")
                self._sync_drawing_mode()
                self.log_panel.info(
                    "已丢弃当前参考线（未保存，丢弃 %d 个桩点）" % discarded
                )
                self._log_action(
                    "reference.cancel", "", pass_index=self._reference_pass, points=discarded
                )
                return
            if not self._reference_pts:
                self._exit_reference_mode()
                return
            removed = self._reference_pts.pop()
            self._reference_last_click = None
            self._persist_reference_draft()
            self._redraw_reference_interaction()
            self._log_action(
                "reference.undo_point", "",
                pass_index=self._reference_pass,
                remaining=len(self._reference_pts),
                x=round(float(removed[0]), 1),
                y=round(float(removed[1]), 1),
            )
            if self._reference_pts:
                self.log_panel.info(
                    "已撤销最后 1 个桩点，剩 %d 个" % len(self._reference_pts)
                )
            else:
                self.log_panel.info("已撤销最后 1 个桩点，当前没有桩点")
            self.lbl_status.setText(
                "参考线标注（第 %d 遍）：已加 %d 点，双击或菜单“完成当前参考线”保存，"
                "右键撤销上一个点"
                % (self._reference_pass, len(self._reference_pts))
            )
            return
        if button != 1:
            return
        point = (float(x), float(y))
        now = time.monotonic()
        last = self._reference_last_click
        if (
            len(self._reference_pts) >= 2
            and last is not None
            and (now - last[0]) <= REFERENCE_DOUBLE_CLICK_SECONDS
            and math.hypot(point[0] - last[1][0], point[1] - last[1][1])
            <= REFERENCE_DOUBLE_CLICK_DISTANCE_PX
        ):
            # 双击的第二下：视为"完成当前参考线"，不新增桩点。
            # 绘制模式下点击事件不会经过 event_bridge，双击在此自行检测。
            self._reference_last_click = None
            self._finish_reference_line()
            return
        self._reference_last_click = (now, point)
        self._reference_pts.append(point)
        # 每加一点立即落盘：切菜单、退出、崩溃都不会丢正在画的线。
        self._persist_reference_draft()
        self._log_action(
            "reference.point", "",
            pass_index=self._reference_pass,
            index=len(self._reference_pts),
            x=round(point[0], 1),
            y=round(point[1], 1),
        )
        self._redraw_reference_interaction(mouse_pos=point)
        self.lbl_status.setText(
            "参考线标注（第 %d 遍）：已加 %d 点，双击或菜单“完成当前参考线”保存"
            % (self._reference_pass, len(self._reference_pts))
        )

    def _redraw_reference_interaction(self, mouse_pos=None):
        """绘制进行中的参考线（interaction 层，不触碰已保存图层）。"""
        viewer = self.image_view.viewer
        viewer.clear_overlay_layer("interaction_reference")
        if not self._reference_pts:
            return
        viewer.begin_overlay_layer("interaction_reference")
        pts = self._reference_pts
        try:
            for index in range(len(pts) - 1):
                viewer.draw_line(
                    pts[index][0], pts[index][1], pts[index + 1][0], pts[index + 1][1],
                    color=REFERENCE_LINE_COLOR_BGR, width=3,
                )
            if mouse_pos:
                viewer.draw_line(
                    pts[-1][0], pts[-1][1], mouse_pos[0], mouse_pos[1],
                    color=REFERENCE_LINE_COLOR_BGR, width=2, style=Qt.DashLine,
                )
            for index, point in enumerate(pts):
                color = (0, 255, 255) if index == 0 else REFERENCE_STATION_COLOR_BGR
                viewer.draw_circle(point[0], point[1], radius=6, color=color)
        finally:
            viewer.end_overlay_layer()

    def _finish_reference_line(self):
        """拟合并保存当前参考线（缓存目录 sidecar，不进项目状态/撤销历史）。"""
        import reference_lines

        pts = [[float(x), float(y)] for x, y in self._reference_pts]
        if len(pts) < 2:
            self.log_panel.warn("参考线至少需要 2 个桩点（当前 %d 个）" % len(pts))
            return False
        tif = str(getattr(self.state, "tif_path", "") or "")
        if not tif:
            self.log_panel.warn("当前没有打开的项目，无法保存参考线。")
            return False
        payload = deepcopy(self._reference_lines_payload() or {})
        if not payload or payload.get("schema") != reference_lines.SCHEMA:
            payload = reference_lines.empty_payload(tif)
        # 一条参考线对应一条行带。用户常把两条带连着画（中间直角过渡），
        # 那样一条记录里就混了两条直线的点，拟合直线必然与点整体错位
        # （实测 58 点里后 4 点偏 114 px ≈ 6.4 m）。这里按转折自动拆开。
        mpp = self._reference_px_to_m(1.0, pts[0][0], pts[0][1])
        # 1.5 m 是"参考线该有多直"的工程口径：行带本身约 1 m 宽，桩点偏离基准线
        # 超过一个半行距就说明这段横跨了两条行带；拿不到米制比例时退回 30 px。
        max_deviation_px = (1.5 / mpp) if mpp else 30.0
        segments = reference_lines.split_reference_lines(
            pts, min_segment_points=2, max_deviation_px=max_deviation_px,
        )
        dropped = len(pts) - sum(len(segment) for segment in segments)
        if len(segments) > 1:
            detail = "，丢弃 %d 个转折过渡点" % dropped if dropped > 0 else ""
            self.log_panel.warn(
                "这条桩点跨了 %d 条行带，已自动拆成 %d 条参考线（一条行带一条线%s）。"
                "下次换行带前先双击收线。"
                % (len(segments), len(segments), detail)
            )
        saved = []
        for segment in segments:
            band_id, band_label, band_note = self._nearest_reference_band(segment)
            record = reference_lines.new_line_record(
                segment,
                band_id=band_id,
                band_label=band_label,
                pass_index=self._reference_pass,
            )
            replaced = False
            if band_id in (None, ""):
                replaced = self._reference_replace_nearest(payload, record)
            if not replaced:
                reference_lines.upsert_line(payload, record)
            saved.append((record, replaced, band_id, band_label, band_note, segment))
        payload["tif_path"] = str(payload.get("tif_path") or tif)
        payload["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            import cache

            cache.save_reference_lines(tif, payload)
        except Exception as exc:  # OSError 等：保存失败必须显式报告
            self.log_panel.error("参考线保存失败：%s" % exc)
            return False
        self._reference_payload = payload
        self._reference_payload_tif = tif
        self._clear_reference_draft()
        self._reference_pts = []
        self._reference_last_click = None
        self.image_view.viewer.clear_overlay_layer("interaction_reference")
        self._draw_reference_lines_overlay()
        self._sync_display_layer_availability()
        for record, replaced, band_id, band_label, band_note, segment in saved:
            band_part = ""
            if band_id not in (None, ""):
                band_part = "，匹配 %s" % (band_label or band_id)
            elif band_note:
                band_part = "，%s" % band_note
            self.log_panel.success(
                "参考线已保存（%s%s）：第 %d 遍，%d 个桩点%s"
                % (
                    "替换 " if replaced else "",
                    record.get("id", ""),
                    self._reference_pass,
                    len(segment),
                    band_part,
                )
            )
            coverage = self._reference_coverage_hint(segment)
            if coverage is not None and coverage < 0.6:
                self.log_panel.warn(
                    "参考线落膜覆盖率约 %.0f%%，疑似偏离行带主体，建议放大复核。"
                    % (coverage * 100.0)
                )
            # 桩点质量提示（不拦截保存）：只看 RMS 会把"个别点跨行"平均掉，
            # 必须同时报最大垂距与它是第几个点。
            fit_info = record.get("fit") or {}
            rms_m = self._reference_px_to_m(
                float(fit_info.get("residual_rms_px") or 0.0),
                segment[0][0], segment[0][1],
            )
            dev_px = float(fit_info.get("max_deviation_px") or 0.0)
            worst = int(fit_info.get("max_deviation_station") or 0)
            anchor = segment[min(max(worst - 1, 0), len(segment) - 1)]
            dev_m = self._reference_px_to_m(dev_px, anchor[0], anchor[1])
            if dev_m is not None and dev_m > 0.5:
                self.log_panel.warn(
                    "参考线拟合偏差偏大：第 %d 个桩点偏离基准线 %.2f m（均方根 %.2f m）。"
                    "橙色实线是按桩点连的线，暗色细虚线是拟合基准——两者分开说明"
                    "桩点不在一条直线上，若有点跨到别的行，右键取消后重画。"
                    % (worst, dev_m, rms_m if rms_m is not None else 0.0)
                )
            self._log_action(
                "reference.finish", str(record.get("id") or ""),
                pass_index=self._reference_pass,
                points=len(segment),
                band_id=band_id,
                band_label=band_label,
                replaced=bool(replaced),
                coverage=None if coverage is None else round(float(coverage), 3),
                rms_m=None if rms_m is None else round(float(rms_m), 3),
                max_dev_m=None if dev_m is None else round(float(dev_m), 3),
                max_dev_station=worst,
                segments=len(segments),
                dropped_stations=dropped,
            )
        self._sync_drawing_mode()
        return True

    def _reference_lines_payload(self):
        """读取参考线 sidecar（按 tif 路径记忆，避免频繁读盘）。"""
        tif = str(getattr(self.state, "tif_path", "") or "")
        if not tif:
            return {}
        if getattr(self, "_reference_payload_tif", "") != tif:
            import cache

            loaded = cache.load_reference_lines(tif)
            if loaded:
                try:
                    import reference_lines

                    loaded = reference_lines.normalize_payload(loaded)
                except ValueError:
                    loaded = {}
            self._reference_payload = loaded if isinstance(loaded, dict) else {}
            self._reference_payload_tif = tif
        return self._reference_payload or {}

    def _reference_lines_present(self):
        return bool(self._reference_lines_payload().get("lines"))

    def _reference_px_to_m(self, px, x, y):
        """显示像素 → 米（借用 GeoUtils）；不可用时返回 None。"""
        try:
            if self.geo.is_ready():
                value = float(self.geo.meters_per_pixel(x, y))
                if value > 0:
                    return float(px) * value
        except Exception:
            pass
        return None

    def _nearest_reference_band(self, pts):
        """把参考线匹配到目录中最接近的中心线（信息列，非硬门槛）。"""
        import reference_lines

        catalog = (
            getattr(self.state, "centerline_catalog", None)
            or getattr(self.state, "work_band_catalog", {})
            or {}
        )
        items = [
            item for item in (catalog.get("items") or []) if isinstance(item, dict)
        ]
        if not items:
            return None, "", "目录中没有可匹配的中心线"
        fit = reference_lines.fit_line(pts)
        samples = reference_lines.sample_along_line(fit["p1"], fit["p2"], 40.0)
        best = None
        for item in items:
            disp = self._centerline_display_points(item)
            if len(disp) < 2:
                continue
            dists = reference_lines.polyline_distances(samples, disp)
            mean_px = float(np.mean(dists))
            if best is None or mean_px < best[0]:
                best = (mean_px, item)
        if best is None:
            return None, "", "目录中没有可匹配的中心线"
        mean_px, item = best
        mean_m = self._reference_px_to_m(mean_px, pts[0][0], pts[0][1])
        if mean_m is not None and mean_m > 1.5:
            return None, "", "最近中心线距离 %.2f m（未匹配）" % mean_m
        label = str(item.get("label") or item.get("id") or "")
        note = "" if mean_m is None else "最近中心线距离 %.2f m" % mean_m
        return str(item.get("id") or ""), label, note

    def _reference_replace_nearest(self, payload, record):
        """无带身份时，与同遍旧参考线近似重合则原位替换，避免重复追加。"""
        import reference_lines

        fit_new = record.get("fit") or {}
        if not fit_new:
            return False
        anchor = record["points"][0]
        mpp = self._reference_px_to_m(1.0, anchor[0], anchor[1])
        limit_px = (0.75 / mpp) if mpp else 40.0
        samples = reference_lines.sample_along_line(
            fit_new["p1"], fit_new["p2"], max(10.0, limit_px / 2.0)
        )
        lines = payload.get("lines") or []
        for index, item in enumerate(lines):
            if int(item.get("pass", 1) or 1) != int(record.get("pass", 1) or 1):
                continue
            if str(item.get("band_id") or "").strip() not in ("", "None", "null"):
                continue
            try:
                old_fit = reference_lines.reference_fit(item)
                dists = reference_lines.polyline_distances(
                    samples, [old_fit["p1"], old_fit["p2"]]
                )
            except Exception:
                continue
            if float(np.mean(dists)) <= limit_px:
                record["id"] = str(item.get("id") or "") or str(record.get("id") or "")
                lines[index] = record
                return True
        return False

    def _reference_coverage_hint(self, pts):
        """参考线落在原始行带掩膜内的采样比例（自动质检提示，不拦截保存）。"""
        import reference_lines

        raw = getattr(self.state, "mask_raw", None)
        if not isinstance(raw, np.ndarray):
            return None
        try:
            fit = reference_lines.fit_line(pts)
            ox = float(getattr(self.state, "mask_offset_x", 0) or 0)
            oy = float(getattr(self.state, "mask_offset_y", 0) or 0)
            local = {
                "p1": [fit["p1"][0] - ox, fit["p1"][1] - oy],
                "p2": [fit["p2"][0] - ox, fit["p2"][1] - oy],
            }
            return reference_lines.line_inside_mask_fraction(local, raw, 20.0)
        except Exception:
            return None

    def _draw_reference_lines_overlay(self, viewer=None):
        """绘制已保存的参考线到 reference_lines 图层（可在图层面板显隐）。

        三层画法，缺一层用户就看不懂：
        1. 暗色细虚线 = 桩点拟合出的**基准直线**（评价口径用它算 Eθ / 沿线采样）；
        2. 橙色实线 = **按桩点连起来的线**（用户画的那条，点和线必须重合）；
        3. 红橙圆点 = 桩点本身。
        之前只画了 1 + 3，58 个桩点里只要有 4 个跨到别的行（实测偏 114 px），
        拟合直线就与点整体错位约 6 m，用户看到的就是"点和线错位"。
        """
        import reference_lines

        viewer = viewer if viewer is not None else self.image_view.viewer
        if not hasattr(viewer, "clear_overlay_layer"):
            return
        viewer.clear_overlay_layer("reference_lines")
        lines = list(self._reference_lines_payload().get("lines") or [])
        if not lines:
            return
        viewer.begin_overlay_layer("reference_lines")
        try:
            for line in lines:
                points = [
                    (float(point[0]), float(point[1]))
                    for point in (line.get("points") or [])
                    if len(point) >= 2
                ]
                if len(points) < 2:
                    continue
                try:
                    fit = reference_lines.reference_fit(line)
                    p1, p2 = fit["p1"], fit["p2"]
                except ValueError:
                    p1 = p2 = None
                if p1 is not None:
                    viewer.draw_line(
                        p1[0], p1[1], p2[0], p2[1],
                        color=REFERENCE_FIT_COLOR_BGR, width=1,
                        style=Qt.DashLine,
                    )
                viewer.draw_polyline(
                    points, color=REFERENCE_LINE_COLOR_BGR, width=2,
                )
                for point in points:
                    viewer.draw_circle(
                        point[0], point[1], radius=5,
                        color=REFERENCE_STATION_COLOR_BGR,
                    )
        finally:
            viewer.end_overlay_layer()

    def _on_finish_reference_line(self):
        """菜单里的“完成当前参考线”：与双击等价，避免用户找不到收线手势。"""
        if not self._reference_drawing:
            self.log_panel.warn("请先进入参考线标注（第 1 遍或第 2 遍）。")
            return
        if len(self._reference_pts) < 2:
            self.log_panel.warn("至少需要 2 个桩点，当前 %d 个。" % len(self._reference_pts))
            return
        self._finish_reference_line()

    def _on_export_reference_lines(self):
        """导出当前项目的参考线 JSON 副本（用于归档/离线评价）。"""
        tif = str(getattr(self.state, "tif_path", "") or "")
        payload = self._reference_lines_payload() if tif else {}
        if not payload or not payload.get("lines"):
            self.log_panel.warn("当前项目还没有参考线，先标注再导出。")
            return
        stem = os.path.splitext(os.path.basename(tif))[0] or "reference_lines"
        default_path = os.path.join(
            os.path.expanduser("~"), "reference_lines_%s.json" % stem
        )
        path, _selected = QFileDialog.getSaveFileName(
            self, "导出参考线 JSON", default_path, "JSON 文件 (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            self.log_panel.error("导出失败：%s" % exc)
            return
        self.log_panel.success("参考线已导出：%s" % path)

    def _set_reference_layer_visible(self, visible):
        """打开/关闭参考线图层的显示（DECISION-063：默认关闭）。"""
        self.image_view.set_display_layer_visibility(
            "reference_lines", bool(visible), user=True
        )

    def _on_toggle_reference_layer(self):
        """菜单入口：手动显示/隐藏已保存的参考线，不动数据。"""
        if not self._reference_lines_present():
            self.log_panel.warn("当前项目还没有参考线，先标注再显示。")
            return
        current = self.image_view.display_layer_visible("reference_lines", False)
        self._set_reference_layer_visible(not current)
        self._draw_reference_lines_overlay()
        if current:
            self.log_panel.info("参考线图层已隐藏。")
        else:
            count = len(self._reference_lines_payload().get("lines") or [])
            self.log_panel.info("参考线图层已显示（%d 条）。" % count)

    def _after_reference_edit(self, message, action, **fields):
        """删除/清除后的统一收口：图层、可用性、内存缓存、日志一起对齐。"""
        # 清掉按 tif 记忆的 sidecar 缓存，否则界面还会按旧内容重画。
        self._reference_payload = {}
        self._reference_payload_tif = ""
        self.image_view.viewer.clear_overlay_layer("reference_lines")
        self._sync_display_layer_availability()
        if self._reference_lines_present():
            self._draw_reference_lines_overlay()
        else:
            self._reference_last_click = None
            if self._reference_drawing:
                self._redraw_reference_interaction()
        self.log_panel.info(message)
        self._log_action(action, "", **fields)

    def _on_delete_last_reference_line(self):
        """删除最后一条已保存的参考线（画错的单条，不必清空全部）。"""
        import cache
        import reference_lines

        tif = str(getattr(self.state, "tif_path", "") or "")
        payload = self._reference_lines_payload()
        lines = list(payload.get("lines") or [])
        if not tif or not lines:
            self.log_panel.warn("当前项目还没有已保存的参考线。")
            return
        removed = lines[-1]
        remaining = lines[:-1]
        new_payload = dict(payload)
        new_payload["lines"] = remaining
        new_payload["schema"] = new_payload.get("schema") or reference_lines.SCHEMA
        try:
            if remaining:
                cache.save_reference_lines(tif, new_payload)
            else:
                cache.delete_reference_lines(tif)
        except OSError as exc:
            self.log_panel.error("删除参考线失败：%s" % exc)
            return
        self._after_reference_edit(
            "已删除最后 1 条参考线（第 %s 遍，%d 个桩点），剩余 %d 条"
            % (
                removed.get("pass", 1),
                len(removed.get("points") or []),
                len(remaining),
            ),
            "reference.delete",
            pass_index=int(removed.get("pass", 1) or 1),
            remaining=len(remaining),
        )

    def _on_clear_reference_lines(self):
        """清除当前项目全部参考线与未完成草稿（需确认，删的是落盘数据）。"""
        import cache

        tif = str(getattr(self.state, "tif_path", "") or "")
        if not tif:
            self.log_panel.warn("当前没有打开的项目。")
            return
        lines = list(self._reference_lines_payload().get("lines") or [])
        draft_pending = len(self._reference_pts)
        if not lines and not draft_pending:
            self.log_panel.warn("当前项目没有参考线可清除。")
            return
        answer = QMessageBox.question(
            self,
            "清除参考线",
            "将删除当前项目已保存的 %d 条参考线%s。\n"
            "参考线是中心线评价的人工基准，删除后无法撤销，只能重新标注。\n"
            "确定清除吗？" % (len(lines), "和正在画的 %d 个桩点" % draft_pending if draft_pending else ""),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            cache.delete_reference_lines(tif)
            for pass_index in (1, 2):
                cache.clear_reference_draft(tif, pass_index)
        except OSError as exc:
            self.log_panel.error("清除参考线失败：%s" % exc)
            return
        self._reference_pts = []
        self._reference_last_click = None
        self.image_view.viewer.clear_overlay_layer("interaction_reference")
        self._after_reference_edit(
            "已清除本项目参考线（%d 条%s）" % (
                len(lines),
                "，草稿 %d 个桩点" % draft_pending if draft_pending else "",
            ),
            "reference.clear",
            lines=len(lines),
            draft_points=draft_pending,
        )

    # ── 作业线标定（人工纠偏，DECISION-060） ──────────────────────────
    #
    # 算法线的"中心"是否等于人认可的中心，只能由人来回答。这一节把人的回答
    # 变成可以复用的数据：拖一下就改一条线，改完立即落盘，规划时人线优先。
    # 没有单独的新图层：改完的线直接替换候选目录里那条线的几何，界面看到的
    # 就是本次规划会走的线。

    def _operator_artifact(self):
        """标定线绑定的产物指纹；掩膜版本变了就不再套用旧标定。"""
        from row_geometry import MASK_REGULARIZATION_VERSION

        artifact = {"mask_version": int(MASK_REGULARIZATION_VERSION)}
        tif = str(getattr(self.state, "tif_path", "") or "")
        try:
            if tif and os.path.isfile(tif):
                stat = os.stat(tif)
                artifact["tif_mtime"] = round(float(stat.st_mtime), 3)
                artifact["tif_size"] = int(stat.st_size)
        except OSError:
            pass
        return artifact

    def _operator_paths_payload(self):
        """读取 operator_paths.json（按影像路径记忆，避免频繁读盘）。"""
        tif = str(getattr(self.state, "tif_path", "") or "")
        if not tif:
            return {}
        if getattr(self, "_calibration_payload_tif", "") != tif:
            import cache

            loaded = cache.load_operator_paths(tif)
            if loaded:
                try:
                    loaded = operator_paths.normalize_payload(loaded)
                except ValueError:
                    loaded = {}
            self._calibration_payload = loaded if isinstance(loaded, dict) else {}
            self._calibration_payload_tif = tif
        return self._calibration_payload or {}

    def _operator_paths_save(self, payload):
        import cache

        tif = str(getattr(self.state, "tif_path", "") or "")
        if not tif:
            raise OSError("当前没有打开的项目")
        payload["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        cache.save_operator_paths(tif, payload)
        self._calibration_payload = payload
        self._calibration_payload_tif = tif

    def _operator_overrides(self):
        """规划用的替换表；版本/影像对不上的旧标定一律不用。"""
        payload = self._operator_paths_payload()
        if not payload.get("lines"):
            return {}
        from row_geometry import MASK_REGULARIZATION_VERSION

        overrides, _report = operator_paths.planner_overrides(
            payload,
            mask_version=int(MASK_REGULARIZATION_VERSION),
            tif_path=str(getattr(self.state, "tif_path", "") or ""),
        )
        return overrides

    def _calibration_items(self):
        catalog = getattr(self.state, "centerline_catalog", None) or {}
        return [item for item in (catalog.get("items") or []) if isinstance(item, dict)]

    def _calibration_item(self, item_id):
        for item in self._calibration_items():
            if str(item.get("id")) == str(item_id):
                return item
        return None

    def _band_by_id(self, band_id):
        result = getattr(self.state, "mask_result", None) or {}
        for key in ("all_bands", "wide_bands", "narrow_bands"):
            for band in result.get(key) or []:
                if str(band.get("id")) == str(band_id):
                    return band
        return None

    def _calibration_width_m(self, item):
        for value in item.get("source_band_ids") or []:
            band = self._band_by_id(value)
            if band:
                width = float(band.get("width_m") or 0.0)
                if width > 0:
                    return width
        try:
            return max(float(item.get("width_m") or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _calibration_scale(self, x=None, y=None):
        try:
            if self.geo.is_ready():
                return float(self.geo.meters_per_pixel(x or 0.0, y or 0.0))
        except Exception:
            pass
        return None

    def _calibration_local_to_display(self, points):
        ox = float(getattr(self.state, "mask_offset_x", 0) or 0)
        oy = float(getattr(self.state, "mask_offset_y", 0) or 0)
        return [(float(x) + ox, float(y) + oy) for x, y in points]

    def _calibration_state_for(self, item):
        item_id = str(item.get("id") or "")
        state = self._calibration_edits.get(item_id)
        if state:
            return state
        base = operator_paths.as_points(
            item.get("centerline") or item.get("work_line") or []
        )
        if base.size == 0:
            return None
        state = {
            "base": [(float(x), float(y)) for x, y in base],
            "edits": [],
            "record_id": "",
        }
        record = operator_paths.find_record(
            self._operator_paths_payload(),
            centerline_id=item_id,
            legacy_id=str(item.get("legacy_id") or ""),
            source_band_ids=item.get("source_band_ids") or [],
        )
        if record:
            state["record_id"] = str(record.get("id") or "")
            saved_base = operator_paths.as_points(record.get("base_points") or [])
            state["base"] = (
                [(float(x), float(y)) for x, y in saved_base]
                if saved_base.size
                else state["base"]
            )
            state["edits"] = [dict(edit) for edit in record.get("edits") or []]
            if not state["edits"]:
                saved = operator_paths.as_points(record.get("points") or [])
                if saved.size:
                    # 描点导入的线没有拖动记录：把它当作新基准，后续拖动在其上叠加
                    state["base"] = [(float(x), float(y)) for x, y in saved]
        self._calibration_edits[item_id] = state
        return state

    def _on_toggle_calibration(self):
        self._set_calibration_mode(not self._calibration_drawing)

    def _set_calibration_mode(self, active):
        if not active:
            self._calibration_drawing = False
            self._calibration_press = None
            self._calibration_preview = None
            self._calibration_item_id = ""
            self.image_view.viewer.clear_overlay_layer("interaction_calibration")
            self._sync_drawing_mode()
            self.log_panel.info("已退出作业线标定")
            return
        if not str(getattr(self.state, "tif_path", "") or ""):
            self.log_panel.warn("请先导入影像，再进行作业线标定。")
            return
        if not self._calibration_items():
            self.log_panel.warn("请先生成作业线，再标定。")
            return
        if self._route_edit_active or self._measure_tool:
            self.log_panel.warn("请先退出路线编辑或测量工具，再进入作业线标定。")
            return
        if self._reference_drawing:
            # 进标定前先保存手上未完成的参考线，不能一换工具就丢。
            if not self._finalize_pending_reference("进入标定"):
                return
            self._reference_drawing = False
        if self._field_drawing or self._forbidden_drawing or self._reference_drawing:
            self._field_drawing = False
            self._forbidden_drawing = False
            self._reference_pts = []
            self._field_pts = []
            self._forbidden_pts = []
            for layer in ("interaction_field", "interaction_forbidden", "interaction_reference"):
                self.image_view.viewer.clear_overlay_layer(layer)
        self._calibration_drawing = True
        self._calibration_press = None
        self._calibration_preview = None
        self._calibration_item_id = ""
        self._sync_drawing_mode()
        self._draw_operator_paths_overlay()
        self.log_panel.info(
            "作业线标定：左键点住要微调的作业线拖动（不按 Shift = 整条平移，"
            "按住 Shift = 只纠局部弯曲），松手即保存；右键撤销上一步；"
            "再次点击同一菜单退出。"
        )
        self._log_action("calibration.mode", "enter", items=len(self._calibration_items()))

    def _calibration_pick(self, x, y):
        """找离点击位置最近、且在容差内的已勾选作业线（只标会进规划的那几条）。"""
        scale = self._calibration_scale(x, y) or 0.0
        tolerance_px = max(12.0, CALIBRATION_PICK_TOLERANCE_M / scale) if scale > 0 else 40.0
        selected = {
            str(value)
            for value in (
                getattr(self.state, "selected_centerline_ids", None)
                or getattr(self.state, "selected_work_band_ids", [])
                or []
            )
        }
        best = None
        for item in self._calibration_items():
            if selected and str(item.get("id")) not in selected:
                continue
            points = self._calibration_local_to_display(
                operator_paths.as_points(item.get("centerline") or item.get("work_line") or [])
            )
            if len(points) < 2:
                continue
            arc, distance = operator_paths.nearest_arc(points, (x, y))
            if best is None or distance < best[0]:
                best = (distance, item, arc)
        if best is None or best[0] > tolerance_px:
            return None, 0.0
        return best[1], float(best[2])

    def _handle_calibration_press(self, x, y, button):
        if button == 2:
            self._undo_calibration_edit()
            return True
        if button != 1:
            return True
        item, _arc = self._calibration_pick(x, y)
        if item is None:
            self.log_panel.warn("请点在要标定的作业线上（容差约 1 m）。")
            return True
        self._calibration_item_id = str(item.get("id") or "")
        self._calibration_press = (float(x), float(y))
        self._calibration_preview = None
        self._draw_operator_paths_overlay()
        label = str(item.get("label") or item.get("id") or "")
        self.log_panel.info("已选中 %s：拖动调整，松手保存" % label)
        return True

    def _calibration_offset_for(self, x, y):
        """把鼠标位移投影到作业线法向，返回 (偏移米, 站点帧, 是否局部纠弯)。"""
        if not self._calibration_press or not self._calibration_item_id:
            return None, None, None
        item = self._calibration_item(self._calibration_item_id)
        if item is None:
            return None, None, None
        state = self._calibration_state_for(item)
        if state is None:
            return None, None, None
        scale = self._calibration_scale(x, y)
        if not scale:
            return None, None, None
        anchor = state["base"]
        arc, _distance = operator_paths.nearest_arc(anchor, self._calibration_press)
        position, tangent = operator_paths.point_at_arc(anchor, arc)
        normal = np.array([-tangent[1], tangent[0]])
        delta = np.array([x - self._calibration_press[0], y - self._calibration_press[1]])
        offset_m = float(np.dot(delta, normal) * scale)
        local = bool(QGuiApplication.keyboardModifiers() & Qt.ShiftModifier)
        return offset_m, (position, normal, arc * scale), local

    def _calibration_preview_points(self, x, y):
        offset_m, frame, local = self._calibration_offset_for(x, y)
        if offset_m is None:
            return None
        item = self._calibration_item(self._calibration_item_id)
        state = self._calibration_state_for(item)
        scale = self._calibration_scale(x, y)
        preview_edit = {
            "kind": "local" if local else "uniform",
            "offset_m": offset_m,
            "arc_m": (frame[2] if frame else 0.0),
            "sigma_m": operator_paths.DEFAULT_SIGMA_M,
        }
        points = operator_paths.path_from_edits(
            state["base"],
            list(state["edits"]) + [preview_edit],
            scale,
            width_m=self._calibration_width_m(item) or 1.0,
        )
        return points, preview_edit, offset_m

    def _handle_calibration_move(self, x, y, buttons):
        if not (buttons & 1) or not self._calibration_press:
            return False
        preview = self._calibration_preview_points(x, y)
        if preview is None:
            return True
        points, edit, offset_m = preview
        self._calibration_preview = (points, edit)
        self._draw_operator_paths_overlay(cursor=(x, y), offset_m=offset_m)
        return True

    def _handle_calibration_release(self, x, y):
        if not self._calibration_press:
            return False
        preview = self._calibration_preview_points(x, y)
        self._calibration_press = None
        self._calibration_preview = None
        if preview is None:
            self._draw_operator_paths_overlay()
            return True
        points, edit, offset_m = preview
        if abs(offset_m) < 0.005:
            self._draw_operator_paths_overlay()
            self.log_panel.info("位移不足 5 mm，未记录标定")
            return True
        self._commit_calibration_edit(self._calibration_item_id, edit)
        return True

    def _write_calibration_record(self, item, state, points, width_m):
        """把当前人线写进 sidecar（新增/替换），返回 (action, record, profile)。"""
        scale = self._calibration_scale() or 0.0
        profile = operator_paths.build_profile(state["base"], points, scale)
        record = operator_paths.new_record(
            points,
            centerline_id=str(item.get("id") or ""),
            legacy_id=str(item.get("legacy_id") or ""),
            source_band_ids=item.get("source_band_ids") or [],
            band_label=str(item.get("label") or ""),
            band_type=str(item.get("kind") or ""),
            width_m=width_m,
            mode="drag",
            edits=state["edits"],
            profile_m=profile,
            artifact=self._operator_artifact(),
            record_id=state.get("record_id") or "",
        )
        # 基准线是人线可复现的前提：没有它，后续拖动无法叠加到同一条线上。
        record["base_points"] = [list(point) for point in state["base"]]
        payload = deepcopy(self._operator_paths_payload() or {})
        if not payload or payload.get("schema") != operator_paths.SCHEMA:
            payload = operator_paths.empty_payload(
                str(getattr(self.state, "tif_path", "") or ""), self._operator_artifact()
            )
        payload.setdefault("artifact", self._operator_artifact())
        action = operator_paths.upsert_record(payload, record)
        self._operator_paths_save(payload)
        state["record_id"] = record.get("id") or state.get("record_id", "")
        return action, record, profile

    def _delete_calibration_record(self, item_id):
        """删除该作业线的标定记录；没有记录时返回 False。"""
        item = self._calibration_item(item_id)
        payload = deepcopy(self._operator_paths_payload() or {})
        record = self._operator_record_for_item(item, payload) if item else None
        if not record:
            return False
        payload["lines"] = [
            entry for entry in (payload.get("lines") or [])
            if not (isinstance(entry, dict) and str(entry.get("id") or "") == str(record.get("id") or ""))
        ]
        self._operator_paths_save(payload)
        return True

    def _commit_calibration_edit(self, item_id, edit):
        item = self._calibration_item(item_id)
        if item is None:
            return False
        state = self._calibration_state_for(item)
        if state is None:
            return False
        scale = self._calibration_scale()
        if not scale:
            self.log_panel.warn("缺少米制比例，无法保存标定。")
            return False
        self._calibration_history.append((item_id, list(state["edits"])))
        state["edits"] = list(state["edits"]) + [dict(edit)]
        width_m = self._calibration_width_m(item) or 1.0
        points = operator_paths.path_from_edits(
            state["base"], state["edits"], scale, width_m=width_m
        )
        if len(points) < 2:
            self.log_panel.warn("标定后的线无效，已忽略本次拖动。")
            return False
        try:
            action, record, profile = self._write_calibration_record(
                item, state, points, width_m
            )
        except OSError as exc:
            self.log_panel.error("标定保存失败：%s" % exc)
            return False
        self._calibration_originals.setdefault(
            str(item.get("id")),
            [tuple(point) for point in operator_paths.as_points(
                item.get("centerline") or item.get("work_line") or []
            )],
        )
        item["centerline"] = [tuple(point) for point in points]
        item["work_line"] = [tuple(point) for point in points]
        item["line_source"] = "operator"
        self._invalidate_analysis_from(
            "path", "作业线已人工标定，旧路径作废，重新生成路径后生效"
        )
        self._draw_centerline_overlay(self.image_view.viewer)
        self._draw_operator_paths_overlay()
        classification = operator_paths.classify_offset_profile(
            [value for _arc, value in profile], width_m
        )
        try:
            self.log_panel.success(
                "已更新 %s（%s）：本次偏移 %+.2f m，%s；规划将采用这条线"
                % (
                    item.get("label") or item.get("id"),
                    action,
                    float(edit.get("offset_m", 0.0)),
                    classification.get("reason", ""),
                )
            )
        except Exception:
            pass
        self._log_action(
            "calibration.edit",
            str(item.get("id") or ""),
            record_id=record.get("id"),
            kind=edit.get("kind"),
            offset_m=round(float(edit.get("offset_m", 0.0)), 4),
            arc_m=round(float(edit.get("arc_m", 0.0)), 3),
            edit_count=len(state["edits"]),
            width_m=round(width_m, 3),
            classification=classification.get("kind"),
        )
        return True

    def _undo_calibration_edit(self):
        if not self._calibration_history:
            self.log_panel.info("没有可撤销的标定操作")
            return
        item_id, previous = self._calibration_history.pop()
        item = self._calibration_item(item_id)
        state = self._calibration_edits.get(str(item_id))
        if item is None or state is None:
            return
        state["edits"] = list(previous)
        if not state["edits"]:
            try:
                self._delete_calibration_record(item_id)
            except OSError as exc:
                self.log_panel.error("撤销标定失败：%s" % exc)
                return
            self._restore_calibration_item(item_id)
            self._draw_centerline_overlay(self.image_view.viewer)
            self._draw_operator_paths_overlay()
            self._invalidate_analysis_from("path", "已撤销作业线标定")
            self.log_panel.info("已撤销 %s 的全部标定" % (item.get("label") or item_id))
            self._log_action("calibration.undo", str(item_id), remaining_edits=0)
            return
        scale = self._calibration_scale() or 0.0
        width_m = self._calibration_width_m(item) or 1.0
        points = operator_paths.path_from_edits(
            state["base"], state["edits"], scale, width_m=width_m
        )
        try:
            self._write_calibration_record(item, state, points, width_m)
        except OSError as exc:
            self.log_panel.error("撤销标定失败：%s" % exc)
            return
        item["centerline"] = [tuple(point) for point in points]
        item["work_line"] = [tuple(point) for point in points]
        self._draw_centerline_overlay(self.image_view.viewer)
        self._draw_operator_paths_overlay()
        self._invalidate_analysis_from("path", "已撤销上一次作业线标定")
        self.log_panel.info("已撤销上一次标定：%s" % (item.get("label") or item_id))

    def _restore_calibration_item(self, item_id):
        original = self._calibration_originals.pop(str(item_id), None)
        item = self._calibration_item(item_id)
        if item is not None and original:
            item["centerline"] = list(original)
            item["work_line"] = list(original)
            item.pop("line_source", None)
        self._calibration_edits.pop(str(item_id), None)

    def _on_clear_calibration(self):
        payload = self._operator_paths_payload()
        if not payload.get("lines"):
            self.log_panel.info("当前项目没有作业线标定")
            return
        item_id = self._calibration_item_id
        if not item_id:
            self.log_panel.warn("请先在图上点选一条要清除标定的作业线。")
            return
        try:
            removed = self._delete_calibration_record(item_id)
        except OSError as exc:
            self.log_panel.error("清除标定失败：%s" % exc)
            return
        if not removed:
            self.log_panel.info("该作业线没有标定记录")
            return
        self._restore_calibration_item(item_id)
        self._draw_centerline_overlay(self.image_view.viewer)
        self._draw_operator_paths_overlay()
        self._invalidate_analysis_from("path", "已清除作业线标定")
        self.log_panel.success("已清除该线的标定，恢复算法中心线")
        self._log_action("calibration.clear", str(item_id))

    def _on_import_reference_as_operator(self):
        """把已有参考线（人工描点）转成标定线：同一批桩点既当基准也当作业线。"""
        reference_payload = self._reference_lines_payload()
        if not reference_payload.get("lines"):
            self.log_panel.warn("当前项目还没有参考线，先标注再导入。")
            return
        lookup = {}
        band_counts: dict = {}
        for item in self._calibration_items():
            for band_id in item.get("source_band_ids") or []:
                band_counts[str(band_id)] = band_counts.get(str(band_id), 0) + 1
        for item in self._calibration_items():
            for band_id in item.get("source_band_ids") or []:
                # 一条带拆成多股作业线时无法自动判断桩点属于哪一股，
                # 宁可不绑定，也不要把一条参考线套到整片作业线上。
                if band_counts.get(str(band_id), 0) > 1:
                    continue
                lookup[str(band_id)] = {
                    "centerline_id": str(item.get("id") or ""),
                    "legacy_id": str(item.get("legacy_id") or ""),
                    "source_band_ids": item.get("source_band_ids") or [],
                    "label": str(item.get("label") or ""),
                    "band_type": str(item.get("kind") or ""),
                    "width_m": self._calibration_width_m(item),
                }
        records, report = operator_paths.from_reference_lines(
            reference_payload,
            offset_x=float(getattr(self.state, "mask_offset_x", 0) or 0),
            offset_y=float(getattr(self.state, "mask_offset_y", 0) or 0),
            artifact=self._operator_artifact(),
            band_lookup=lookup,
        )
        if not records:
            self.log_panel.warn("参考线里没有可转换的桩点（%d 条被跳过）" % report.get("skipped", 0))
            return
        payload = deepcopy(self._operator_paths_payload() or {})
        if not payload or payload.get("schema") != operator_paths.SCHEMA:
            payload = operator_paths.empty_payload(
                str(getattr(self.state, "tif_path", "") or ""), self._operator_artifact()
            )
        applied = 0
        for record in records:
            if not record.get("centerline_id") and not record.get("source_band_ids"):
                continue
            operator_paths.upsert_record(payload, record)
            applied += 1
        if not applied:
            self.log_panel.warn("参考线没有匹配到当前作业线，未导入。")
            return
        try:
            self._operator_paths_save(payload)
        except OSError as exc:
            self.log_panel.error("导入标定线失败：%s" % exc)
            return
        self._calibration_edits.clear()
        for record in payload.get("lines") or []:
            item_id = str(record.get("centerline_id") or "")
            item = self._calibration_item(item_id) if item_id else None
            if item is None:
                continue
            points = operator_paths.as_points(record.get("points") or [])
            if points.size == 0:
                continue
            self._calibration_originals.setdefault(
                item_id,
                [tuple(point) for point in operator_paths.as_points(
                    item.get("centerline") or item.get("work_line") or []
                )],
            )
            item["centerline"] = [tuple(point) for point in points]
            item["work_line"] = [tuple(point) for point in points]
            item["line_source"] = "operator"
        self._draw_centerline_overlay(self.image_view.viewer)
        self._draw_operator_paths_overlay()
        self._invalidate_analysis_from("path", "已导入人工标定线")
        self.log_panel.success("已把 %d 条人工参考线转为标定线，规划将采用这些线" % applied)
        self._log_action("calibration.import_reference", "reference", applied=applied)

    def _on_export_calibration_case(self):
        """导出离线评价用的案例包（人工线 + 带几何 + 掩膜路径）。"""
        import centerline_calibration

        payload = self._operator_paths_payload()
        if not payload.get("lines"):
            self.log_panel.warn("当前项目还没有标定线，先标定再导出。")
            return
        scale = self._calibration_scale()
        if not scale:
            self.log_panel.warn("缺少米制比例，无法导出标定案例。")
            return
        import cache

        tif = str(getattr(self.state, "tif_path", "") or "")
        try:
            mask_path = os.path.join(
                os.path.dirname(cache.operator_paths_path(tif)), "mask.npz"
            )
        except Exception:
            mask_path = ""
        if not mask_path or not os.path.isfile(mask_path):
            self.log_panel.warn("项目缓存缺少 mask.npz，无法导出标定案例。")
            return
        bands = []
        mapping = {}
        result = getattr(self.state, "mask_result", None) or {}
        for key in ("all_bands", "wide_bands", "narrow_bands"):
            for band in result.get(key) or []:
                band_id = band.get("id")
                if band_id is None:
                    continue
                points = operator_paths.as_points(band.get("centerline") or [])
                if points.size == 0:
                    continue
                mapping[str(band_id)] = band
                if any(item.get("band_id") is not None and str(item.get("band_id")) == str(band_id) for item in bands):
                    continue
                record = {
                    "band_id": band_id,
                    "band_type": band.get("band_type"),
                    "width_m": band.get("width_m"),
                    "profile_width_m": band.get("profile_width_m"),
                    "centerline": [[float(x), float(y)] for x, y in points],
                }
                pre = operator_paths.as_points(band.get("centerline_pre_recenter") or [])
                if pre.size:
                    record["centerline_pre_recenter"] = [[float(x), float(y)] for x, y in pre]
                contour = np.asarray(band.get("contour"), dtype=np.float64).reshape(-1, 2)
                if contour.shape[0] >= 3:
                    step = max(1, contour.shape[0] // 400)
                    record["contour"] = [[float(x), float(y)] for x, y in contour[::step]]
                bands.append(record)
        operator_lines = []
        for record in payload.get("lines") or []:
            points = operator_paths.as_points(record.get("points") or [])
            if points.size == 0:
                continue
            band_id = None
            for value in record.get("source_band_ids") or []:
                if str(value) in mapping:
                    band_id = value
                    break
            if band_id is None:
                continue
            operator_lines.append({
                "band_id": band_id,
                "points": [[float(x), float(y)] for x, y in points],
                "centerline_id": record.get("centerline_id"),
                "edits": record.get("edits") or [],
            })
        if not operator_lines:
            self.log_panel.warn("标定线没有匹配到带几何，无法导出案例。")
            return
        case = {
            "schema": centerline_calibration.SCHEMA,
            "case_id": "%s_%s" % (
                os.path.splitext(os.path.basename(tif))[0] or "case",
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            ),
            "tif_path": tif,
            "mpp": float(scale),
            "mask_path": mask_path,
            "artifact": payload.get("artifact") or self._operator_artifact(),
            "bands": bands,
            "operator_lines": operator_lines,
        }
        out_dir = os.path.join(os.path.dirname(mask_path), "calibration")
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, case["case_id"] + ".json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(case, handle, ensure_ascii=False, indent=2, default=str)
        except OSError as exc:
            self.log_panel.error("导出标定案例失败：%s" % exc)
            return
        self.log_panel.success(
            "标定案例已导出：%s（%d 条带 / %d 条人线）"
            % (path, len(bands), len(operator_lines))
        )
        self._log_action("calibration.export_case", path, bands=len(bands), lines=len(operator_lines))

    def _on_action_log_dialog(self):
        """查看本次会话的操作记录（按时间列出动作与参数），可导出副本。"""
        log = interaction_log.action_log()
        text = interaction_log.format_records(log.recent(500))
        if not text:
            text = "本次会话还没有操作记录。"
        ActionLogDialog(
            text,
            str(log.log_path()),
            export_callback=self._on_export_action_log,
            parent=self,
        ).exec()

    def _on_export_action_log(self):
        """把本次会话的操作记录复制到用户目录，返回落盘目录（供对话框提示）。"""
        try:
            paths = interaction_log.action_log().export_to(
                os.path.join(os.path.expanduser("~"), "ALIGN_操作记录")
            )
        except Exception as exc:
            self.log_panel.error("导出操作记录失败：%s" % exc)
            raise
        if not paths:
            self.log_panel.info("本次会话还没有可导出的操作记录。")
            return ""
        destination = os.path.dirname(paths[0])
        self.log_panel.success("操作记录已导出：%s" % destination)
        return destination

    def _draw_operator_paths_overlay(self, viewer=None, cursor=None, offset_m=None):
        """标定模式下的叠加层：已保存人线 + 正在拖动的预览。"""
        viewer = viewer if viewer is not None else self.image_view.viewer
        if not hasattr(viewer, "clear_overlay_layer"):
            return
        viewer.clear_overlay_layer("interaction_calibration")
        if not self._calibration_drawing:
            return
        payload = self._operator_paths_payload()
        viewer.begin_overlay_layer("interaction_calibration")
        try:
            for record in payload.get("lines") or []:
                points = self._calibration_local_to_display(
                    operator_paths.as_points(record.get("points") or [])
                )
                if len(points) < 2:
                    continue
                selected = str(record.get("centerline_id") or "") == self._calibration_item_id
                viewer.draw_polyline(
                    points,
                    color=OPERATOR_EDIT_COLOR_BGR if selected else OPERATOR_LINE_COLOR_BGR,
                    width=6 if selected else 4,
                )
                scale = self._calibration_scale() or 0.0
                for edit in record.get("edits") or []:
                    try:
                        arc_m = float(edit.get("arc_m", 0.0))
                    except (TypeError, ValueError):
                        continue
                    if scale <= 0:
                        continue
                    position, _tangent = operator_paths.point_at_arc(points, arc_m / scale)
                    viewer.draw_circle(position[0], position[1], radius=4, color=OPERATOR_EDIT_COLOR_BGR)
            if self._calibration_preview:
                points = self._calibration_local_to_display(self._calibration_preview[0])
                if len(points) > 1:
                    step = max(1, len(points) // CALIBRATION_PREVIEW_POINT_LIMIT)
                    viewer.draw_polyline(
                        points[::step] + [points[-1]], color=OPERATOR_EDIT_COLOR_BGR, width=5
                    )
                if cursor and self._calibration_press:
                    viewer.draw_line(
                        self._calibration_press[0], self._calibration_press[1],
                        cursor[0], cursor[1],
                        color=OPERATOR_EDIT_COLOR_BGR, width=2, style=Qt.DashLine,
                    )
            if self._calibration_item_id:
                item = self._calibration_item(self._calibration_item_id)
                if item is not None:
                    points = self._calibration_local_to_display(
                        operator_paths.as_points(
                            item.get("centerline") or item.get("work_line") or []
                        )
                    )
                    if len(points) > 1:
                        viewer.draw_polyline(points, color=OPERATOR_EDIT_COLOR_BGR, width=7)
        finally:
            viewer.end_overlay_layer()


    def _route_service_anchor(self, kind):
        """Return the visible route endpoint for a service-point marker.

        ``entry_point``/``exit_point`` are user inputs and remain unchanged in
        ``AppState`` for provenance.  This helper is retained for animation
        and diagnostics, but the editable marker must never be replaced by a
        canonical route vertex: reference-only planning intentionally keeps a
        user-selected service point independent from the nearest work-line
        endpoint.
        """
        if kind not in {"entry", "exit"}:
            return None
        pipeline = getattr(self, "_pipeline_result", None)
        if not isinstance(pipeline, dict):
            return None
        path_result = pipeline.get("path")
        if not isinstance(path_result, dict):
            return None
        full_path = path_result.get("full_path") or []
        segments = [
            self._coerce_points(segment)
            for segment in full_path
        ]
        segments = [segment for segment in segments if len(segment) >= 2]
        if not segments:
            return None
        point = segments[0][0] if kind == "entry" else segments[-1][-1]
        return (float(point[0]), float(point[1]))

    def _display_service_point_value(self, target):
        # The marker is the editable state value, both before and after
        # planning.  The old route-anchor substitution made every successful
        # replan visually snap the marker back to the canonical first/last
        # route vertex, so dragging appeared to have no effect.
        return self._service_point_state_value(target)

    def _service_point_state_value(self, target):
        kind, index = target
        if kind == "entry":
            # DECISION-019: while the route editor is active, the start marker
            # is the first shared control node rather than a second service
            # point drawn at an unrelated coordinate.  This also makes the
            # marker follow the node during a live endpoint drag.
            if self._route_edit_active:
                endpoint = self._route_control_endpoint("entry")
                if endpoint is not None:
                    return endpoint
            return getattr(self.state, "entry_point", None)
        if kind == "exit":
            if self._route_edit_active:
                endpoint = self._route_control_endpoint("exit")
                if endpoint is not None:
                    return endpoint
            explicit = getattr(self.state, "exit_point", None)
            if explicit is not None:
                return explicit
            # A generated route always has a concrete final extension.  Treat
            # that endpoint as the draggable end marker until the operator
            # moves it, at which point it becomes an explicit exit input.
            if getattr(self.state, "auto_path_planned", False):
                return self._route_finish_point_from_result()
            return None
        points = list(getattr(self.state, "unload_points", []) or [])
        return points[index] if 0 <= index < len(points) else None

    def _route_control_endpoint(self, kind, points=None):
        """Return the endpoint node shared by the route editor and markers."""
        values = list(
            points
            if points is not None
            else (getattr(self.state, "path_points", []) or [])
        )
        if not values:
            return None
        if kind == "entry":
            point = values[0]
        elif kind == "exit":
            point = values[-1]
        else:
            return None
        return (float(point[0]), float(point[1]))

    def _service_point_control_style(self, target, point=None):
        """Use the same status colour and node size as route control points."""
        kind, _index = target
        marker = point or self._service_point_state_value(target)
        points = list(getattr(self.state, "path_points", []) or [])
        statuses = list(getattr(self.state, "path_status", []) or [])
        node_index = None
        if marker is not None and points:
            endpoint = self._route_control_endpoint(kind, points)
            if endpoint is not None and math.hypot(
                float(marker[0]) - endpoint[0],
                float(marker[1]) - endpoint[1],
            ) <= 1e-5:
                node_index = 0 if kind == "entry" else len(points) - 1
        if node_index is None:
            # Before a route exists, entry and finish connectors are both
            # status-0 references.  Once a control route exists, the endpoint
            # branch above uses its actual outgoing/inherited status.
            color = self._route_edit_control_point_color([0], 0)
        else:
            color = self._route_edit_control_point_color(statuses, node_index)
        return color, node_index

    def _sync_service_points_to_route_control_endpoints(self, points=None):
        """Keep endpoint service references attached to shared control nodes.

        Manual route editing owns ``path_points`` and its parallel edge modes.
        If its first or final node
        is removed/replaced, the visible start/end reference follows the new
        endpoint instead of remaining as a detached marker.  An unset end is
        intentionally kept unset so the planner can continue to generate its
        automatic final extension; the display still resolves it from the
        final control node.
        """
        values = list(
            points
            if points is not None
            else (getattr(self.state, "path_points", []) or [])
        )
        if len(values) < 2:
            return False
        first = self._route_control_endpoint("entry", values)
        last = self._route_control_endpoint("exit", values)
        changed = False
        if (
            getattr(self.state, "entry_point", None) is not None
            or self._route_edit_active
        ):
            if getattr(self.state, "entry_point", None) != first:
                self.state.entry_point = first
                changed = True
            if not getattr(self.state, "entry_point_locked", False):
                self.state.entry_point_locked = True
                changed = True
        if getattr(self.state, "exit_point", None) is not None:
            if getattr(self.state, "exit_point", None) != last:
                self.state.exit_point = last
                changed = True
            if not getattr(self.state, "exit_point_locked", False):
                self.state.exit_point_locked = True
                changed = True
        return changed

    def _draw_service_points_overlay(self, clear_first=False):
        viewer = self.image_view.viewer
        # Service markers are a persistent, independently switchable layer;
        # the drag preview remains a separate interaction-only collection.
        viewer.clear_overlay_layer("service_points")
        self._service_point_overlay_items = []
        if not (
            self._service_points_should_show()
            or self.image_view.display_layer_visible("service_points")
        ):
            return
        points = []
        ep = self._display_service_point_value(("entry", 0))
        xp = self._display_service_point_value(("exit", 0))
        if ep:
            points.append((ep, "起点", (0, 255, 80)))
        if xp:
            points.append((xp, "终点", (255, 80, 80)))
        for index, point in enumerate(list(getattr(self.state, 'unload_points', []) or []), start=1):
            points.append((point, f"卸粮点{index}", (255, 210, 40)))
        viewer.begin_overlay_layer("service_points")
        try:
            for point, label, color in points:
                x, y = float(point[0]), float(point[1])
                kind = "entry" if label == "起点" else (
                    "exit" if label == "终点" else "unload"
                )
                if kind in {"entry", "exit"}:
                    color, _node_index = self._service_point_control_style(
                        (kind, 0), point
                    )
                    radius = 6
                else:
                    radius = 8
                self._service_point_overlay_items.append(
                    viewer.draw_circle(x, y, radius=radius, color=color)
                )
                self._service_point_overlay_items.append(
                    viewer.draw_text(x + 8, y - 10, label, color=color, size=10)
                )
        finally:
            viewer.end_overlay_layer()

    def _route_finish_point_from_result(self, path_result=None):
        """Return the last canonical route vertex for display-only marking."""
        result = path_result if isinstance(path_result, dict) else None
        if result is None:
            pipeline = getattr(self, "_pipeline_result", None)
            result = pipeline.get("path") if isinstance(pipeline, dict) else None
        result = result if isinstance(result, dict) else {}
        for key in ("full_path", "turns", "tracks"):
            for item in reversed(list(result.get(key) or [])):
                raw_points = item.get("points", []) if isinstance(item, dict) else item
                points = self._coerce_points(raw_points)
                if points:
                    return points[-1]
        segments = list(getattr(self.state, "auto_path_segments", []) or [])
        if not segments:
            segments = list(getattr(self.state, "auto_path", []) or [])
        for segment in reversed(segments):
            points = self._coerce_points(getattr(segment, "points", []))
            if points:
                return points[-1]
        return None

    def _draw_route_finish_overlay(self, path_result=None, *, stale_preview=False):
        """Fallback finish marker when service-point labels are hidden."""
        if (
            stale_preview
            or getattr(self.state, "exit_point", None) is not None
            or self._service_points_should_show()
        ):
            return
        finish = self._route_finish_point_from_result(path_result)
        if finish is None:
            return
        viewer = self.image_view.viewer
        x, y = float(finish[0]), float(finish[1])
        viewer.draw_circle(x, y, radius=9, color=(255, 80, 80))
        viewer.draw_text(x + 12, y - 16, "终点", color=(255, 80, 80), size=12)

    def _service_point_hit_radius(self):
        """Return a screen-stable hit radius expressed in image coordinates."""
        transform = self.image_view.viewer.transform()
        scale = math.hypot(float(transform.m11()), float(transform.m12()))
        return 14.0 / max(scale, 1e-6)

    def _service_point_hit_test(self, x, y):
        """Find the nearest visible service point without relying on image side."""
        if not self._service_points_should_show():
            return None
        candidates = []
        entry = self._display_service_point_value(("entry", 0))
        exit_point = self._display_service_point_value(("exit", 0))
        if entry:
            candidates.append(("entry", 0, entry))
        if exit_point:
            candidates.append(("exit", 0, exit_point))
        for index, point in enumerate(
            list(getattr(self.state, "unload_points", []) or [])
        ):
            candidates.append(("unload", index, point))
        if not candidates:
            return None
        target = min(
            candidates,
            key=lambda item: math.hypot(
                float(item[2][0]) - float(x),
                float(item[2][1]) - float(y),
            ),
        )
        distance = math.hypot(
            float(target[2][0]) - float(x),
            float(target[2][1]) - float(y),
        )
        return (target[0], target[1]) if distance <= self._service_point_hit_radius() else None

    def _service_point_direct_drag_available(self, x, y):
        # Route editing uses the same direct-capture path as service markers.
        # drawing_mode otherwise treats a left drag as canvas panning.
        if self._route_edit_active:
            if self._route_edit_nearest_node((x, y)) >= 0:
                return True
            # Creation tools commit only on an un-dragged release. Capturing a
            # line on press would reinterpret a drag as an insertion.
            if self._route_edit_mode in {"add_point", "add_work"}:
                return False
            edge_index = self._route_edit_nearest_edge((x, y))
            return self._route_edit_work_index_for_edge(edge_index) >= 0
        if (
            self._show_mask_before
            or self._measure_tool
            or self._field_drawing
            or self._forbidden_drawing
        ):
            return False
        return self._service_point_hit_test(x, y) is not None

    def _service_point_value(self, target):
        return self._display_service_point_value(target)

    def _set_service_point_value(self, target, point, journal=False):
        kind, index = target
        point = (float(point[0]), float(point[1]))
        if kind == "entry":
            if journal:
                self.state.safe_update(entry_point=point, entry_point_locked=True)
            else:
                self.state.entry_point = point
                self.state.entry_point_locked = True
            return
        if kind == "exit":
            if journal:
                self.state.safe_update(exit_point=point, exit_point_locked=True)
            else:
                self.state.exit_point = point
                self.state.exit_point_locked = True
            return
        points = list(getattr(self.state, "unload_points", []) or [])
        if not (0 <= index < len(points)):
            return
        points[index] = point
        if journal:
            self.state.safe_update(
                unload_points=points,
                unload_point=points[0] if points else None,
                unload_point_locked=True,
            )
        else:
            self.state.unload_points = points
            self.state.unload_point = points[0] if points else None
            self.state.unload_point_locked = True

    def _begin_service_point_drag(self, x, y):
        """DECISION-007: begin direct manipulation of an existing service point."""
        target = self._service_point_hit_test(x, y)
        if target is None:
            return False
        origin = self._service_point_value(target)
        if origin is None:
            return False
        self.state._drag_entry_exit = target
        self._service_point_drag_origin = (
            float(origin[0]), float(origin[1])
        )
        self._service_point_drag_undo_saved = False
        self._service_point_drag_had_path = bool(
            getattr(self.state, "auto_path_planned", False)
        )
        self._service_point_drag_moved = False
        self._service_point_drag_last_correction = {}
        # Service points follow the cursor in both image axes.  Keep only the
        # confirmed field boundary as a hard safety constraint; do not apply a
        # main-angle/terminal-depth snap during the drag.
        self._service_point_replan_timer.stop()
        self._clear_service_point_drag_preview()
        self.image_view.viewer.setCursor(Qt.ClosedHandCursor)
        return True

    def _move_service_point_drag(self, x, y):
        target = getattr(self.state, "_drag_entry_exit", None)
        if target is None:
            return False
        point, correction = self._constrain_service_point(
            target, (x, y)
        )
        origin = self._service_point_drag_origin
        if origin is None or math.hypot(
            float(point[0]) - origin[0], float(point[1]) - origin[1]
        ) <= 1e-7:
            self._service_point_drag_last_correction = correction
            self._draw_service_points_overlay(clear_first=False)
            self._draw_service_point_drag_preview(
                target, (x, y), point, correction
            )
            return True
        if not self._service_point_drag_undo_saved:
            labels = {"entry": "移动起点", "exit": "移动终点", "unload": "移动卸粮点"}
            self._save_system_undo(labels.get(target[0], "移动服务点"))
            self._service_point_drag_undo_saved = True
        self._set_service_point_value(target, point, journal=False)
        self._service_point_drag_moved = True
        self._service_point_drag_last_correction = correction
        self._draw_service_points_overlay(clear_first=False)
        self._draw_service_point_drag_preview(target, (x, y), point, correction)
        self._refresh_service_points_panel()
        return True

    def _finish_service_point_drag(self, x, y):
        target = getattr(self.state, "_drag_entry_exit", None)
        if target is None:
            return False
        had_path = bool(self._service_point_drag_had_path)
        moved = bool(self._service_point_drag_moved)
        if moved:
            point, correction = self._constrain_service_point(
                target, (x, y)
            )
            self._set_service_point_value(target, point, journal=True)
            self._service_point_drag_last_correction = correction
            if had_path:
                self._invalidate_analysis_from(
                    "path",
                    "起点、可选终点或卸粮点已拖动，必须重新规划和验证",
                )
            labels = {"entry": "起点", "exit": "终点", "unload": "卸粮点"}
            self.log_panel.success(
                f"{labels.get(target[0], '服务点')}已移动到: "
                f"({float(point[0]):.1f}, {float(point[1]):.1f})"
            )
        self._clear_service_point_drag_preview()
        self.state._drag_entry_exit = None
        self._service_point_drag_origin = None
        self._service_point_drag_undo_saved = False
        self._service_point_drag_had_path = False
        self._service_point_drag_moved = False
        self._sync_drawing_mode()
        self._restore_visual_from_state()
        self._refresh_service_points_panel()
        self._sync_task_statuses()
        if moved:
            self._schedule_history_checkpoint()
        if had_path and self._service_drag_can_auto_replan():
            self._service_point_replan_timer.start(400)
        return True

    def _clear_service_point_drag_preview(self):
        viewer = self.image_view.viewer
        viewer.remove_overlays(self._service_point_drag_preview_items)
        self._service_point_drag_preview_items = []

    def _work_endpoint_candidates(self):
        endpoints = []
        segments = list(getattr(self.state, "auto_path_segments", []) or [])
        if not segments:
            segments = list(getattr(self.state, "auto_path", []) or [])
        for segment in segments:
            if isinstance(segment, dict):
                segment_type = str(segment.get("segment_type", ""))
                raw_points = segment.get("points")
            else:
                segment_type = str(getattr(segment, "segment_type", ""))
                raw_points = getattr(segment, "points", None)
            if segment_type != "work":
                continue
            points = self._coerce_points(raw_points)
            if len(points) >= 2:
                endpoints.extend((points[0], points[-1]))
        return endpoints

    def _draw_service_point_drag_preview(
        self, target, requested_point, corrected_point, correction
    ):
        """DECISION-007: show a cheap live preview; never certify it as a path."""
        self._clear_service_point_drag_preview()
        viewer = self.image_view.viewer
        requested = np.asarray(requested_point, dtype=np.float64)
        corrected = np.asarray(corrected_point, dtype=np.float64)
        if float(np.linalg.norm(requested - corrected)) > 1e-6:
            self._service_point_drag_preview_items.append(viewer.draw_line(
                float(requested[0]), float(requested[1]),
                float(corrected[0]), float(corrected[1]),
                color=(0, 210, 255), width=2, style=Qt.DashLine,
            ))
            self._service_point_drag_preview_items.append(viewer.draw_text(
                float(corrected[0]) + 10.0,
                float(corrected[1]) + 10.0,
                "动态纠正",
                color=(0, 210, 255),
                size=10,
            ))
        if target[0] not in ("entry", "exit"):
            return
        endpoints = self._work_endpoint_candidates()
        if not endpoints:
            return
        endpoint = np.asarray(min(
            endpoints,
            key=lambda item: math.hypot(
                float(item[0]) - float(corrected[0]),
                float(item[1]) - float(corrected[1]),
            ),
        ), dtype=np.float64)
        angle = float(
            (getattr(self.state, "mask_result", None) or {}).get(
                "main_angle", 0.0
            )
        )
        axis = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
        delta = endpoint - corrected
        axial_delta = float(np.dot(delta, axis))
        corner = endpoint - axial_delta * axis
        for first, second in ((corrected, corner), (corner, endpoint)):
            if float(np.linalg.norm(second - first)) <= 1e-6:
                continue
            self._service_point_drag_preview_items.append(viewer.draw_line(
                float(first[0]), float(first[1]),
                float(second[0]), float(second[1]),
                color=(0, 210, 255), width=2, style=Qt.DashLine,
            ))

    def _service_drag_can_auto_replan(self):
        return bool(
            getattr(self.state, "mask_processed", False)
            and getattr(self.state, "entry_point", None)
            and getattr(self.state, "entry_point_locked", False)
            and not self._has_background_worker()
        )

    def _run_service_point_drag_replan(self):
        if self._service_drag_can_auto_replan():
            self.log_panel.info("服务点拖动已稳定，正在按纠正后位置重新规划路径")
            self._on_plan()

    def _service_points_should_show(self):
        return bool(
            getattr(self, "_service_points_visible", False)
            or getattr(self.state, 'entry_exit_mode', False)
        )

    def _refresh_service_points_panel(self):
        if not self._service_points_should_show():
            self.route_info.update_service_points(None, None, [])
            return
        entry = self._display_service_point_value(("entry", 0))
        exit_point = self._display_service_point_value(("exit", 0))
        unload_points = list(getattr(self.state, 'unload_points', []) or [])
        if not (entry or exit_point or unload_points):
            self.route_info.clear_info()
            return
        self.route_info.update_service_points(entry, exit_point, unload_points)

    def _on_service_point_selected(self, kind, index):
        point = self._display_service_point_value((kind, index))
        if not point:
            return
        self._draw_service_points_overlay(clear_first=False)
        viewer = self.image_view.viewer
        viewer.clear_overlay_layer("interaction_service_point")
        viewer.begin_overlay_layer("interaction_service_point")
        try:
            viewer.draw_circle(
                float(point[0]), float(point[1]),
                radius=13, color=(0, 220, 255),
            )
        finally:
            viewer.end_overlay_layer()

    def _constrain_point_to_field(self, point):
        boundary = getattr(self.state, 'field_boundary', None)
        if not boundary or len(boundary) < 3:
            return (float(point[0]), float(point[1]))
        poly = np.asarray(boundary, dtype=np.float32)
        pt = (float(point[0]), float(point[1]))
        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            return pt
        vertices = np.asarray(boundary, dtype=np.float64)
        requested = np.asarray(pt, dtype=np.float64)
        nearest = None
        nearest_distance = float("inf")
        for index, start in enumerate(vertices):
            end = vertices[(index + 1) % len(vertices)]
            segment = end - start
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 1e-12:
                candidate = start
            else:
                fraction = float(np.clip(
                    np.dot(requested - start, segment) / length_sq,
                    0.0,
                    1.0,
                ))
                candidate = start + fraction * segment
            distance = float(np.linalg.norm(candidate - requested))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = candidate
        if nearest is None:
            nearest = vertices[0]
        return (float(nearest[0]), float(nearest[1]))

    def _constrain_service_point(self, target, point):
        """Keep a dragged service point inside the confirmed field only.

        Entry and exit points are user-selected execution references, not
        crop-row anchors.  Projecting them onto the main-angle terminal wall
        can move an exit to the opposite end of the last work line and create
        an artificial ``turn_reverse`` segment.  The live connector preview
        remains visual-only; the point itself follows the cursor in both axes.
        """
        requested = (float(point[0]), float(point[1]))
        constrained = self._constrain_point_to_field(requested)
        # QGraphicsView scene-to-image transforms can produce values such as
        # 30.000000000000004 for an exact pixel coordinate.  Keep sub-pixel
        # precision for real edits while canonicalising harmless binary noise
        # before the value enters AppState/provenance and undo snapshots.
        constrained = (
            round(float(constrained[0]), 12),
            round(float(constrained[1]), 12),
        )
        report = {
            "requested_point": requested,
            "corrected_point": constrained,
            "field_corrected": math.hypot(
                constrained[0] - requested[0], constrained[1] - requested[1]
            ) > 1e-7,
            "terminal_corrected": False,
            "terminal_side": None,
            "terminal_source": None,
            "preview_only": True,
        }
        return constrained, report

    def _suggest_service_points(self):
        mask_result = getattr(self.state, 'mask_result', None) or {}
        wide_bands = mask_result.get("wide_bands", [])
        # Historical builds auto-filled an unlocked end point.  It must not
        # silently remain an active planning input now that end is optional.
        if not getattr(self.state, 'exit_point_locked', False):
            self.state.exit_point = None
            self.state.exit_point_locked = False
        if not wide_bands:
            return
        entry_pt = None
        try:
            from path_planner import prepare_band_centerlines
            lines, _ = prepare_band_centerlines(
                wide_bands,
                float(mask_result.get("main_angle", 0.0) or 0.0),
                geo=self.geo,
                state=self.state,
                config=self.cfg.section("path_planning"),
            )
            if lines:
                first = lines[0]
                entry_local = np.asarray(first[0], dtype=np.float64)
                entry_next = np.asarray(first[1], dtype=np.float64)
                ref_global = (
                    float(entry_local[0]) + float(getattr(self.state, 'mask_offset_x', 0) or 0),
                    float(entry_local[1]) + float(getattr(self.state, 'mask_offset_y', 0) or 0),
                )
                ppm = self.geo.pixels_per_meter(*ref_global) if self.geo.is_ready() else 1.0
                approach_px = max(1.0, 1.0 * ppm)

                def extend(endpoint, neighbor):
                    direction = endpoint - neighbor
                    norm = float(np.linalg.norm(direction))
                    if norm <= 1e-9:
                        return endpoint
                    return endpoint + direction / norm * approach_px

                entry_local = extend(entry_local, entry_next)
                ox = float(getattr(self.state, 'mask_offset_x', 0) or 0)
                oy = float(getattr(self.state, 'mask_offset_y', 0) or 0)
                entry_pt = (float(entry_local[0]) + ox, float(entry_local[1]) + oy)
        except Exception as exc:
            self._log.info(f"service point suggestion failed: {exc}")
        if entry_pt and (not getattr(self.state, 'entry_point', None) or not getattr(self.state, 'entry_point_locked', False)):
            self.state.entry_point = self._constrain_point_to_field(entry_pt)

    def _on_recommend_service_points(self):
        """Recommend the required start while leaving the end optional."""
        if not getattr(self.state, "mask_processed", False):
            self.log_panel.warn("请先完成掩膜优化，再推荐起点。")
            return False
        before = getattr(self.state, "entry_point", None)
        self._suggest_service_points()
        entry = getattr(self.state, "entry_point", None)
        if not entry:
            self.log_panel.error("当前行带不足以推荐起点，请使用“布设”手动设置")
            return False
        self._service_points_visible = True
        self._sync_display_layer_availability()
        self._draw_service_points_overlay(clear_first=False)
        self._refresh_service_points_panel()
        self._sync_task_statuses()
        if entry != before:
            self.log_panel.success(
                "起点已推荐 · 可拖动调整；终点随最后一段自动延伸，"
                "也可在画布上拖动调整"
            )
        else:
            self.log_panel.info("已保留当前起点；锁定的人工设置不会被覆盖")
        return True

    def _on_toggle_entry_exit(self, target=None):
        if target in ("entry", "exit"):
            point_name = f"{target}_point"
            lock_name = f"{target}_point_locked"
            if target == "entry" and not getattr(self.state, point_name, None):
                self._on_recommend_service_points()
            if getattr(self.state, point_name, None):
                self.state.safe_update(**{lock_name: True, "entry_exit_mode": False})
                self._service_points_visible = True
                self._refresh_service_points_panel()
                self._sync_task_statuses()
            else:
                self.state.entry_exit_mode = True
                self._entry_exit_click_idx = 0 if target == "entry" else 1
                self._service_points_visible = True
                self._sync_drawing_mode()
                self.log_panel.info(
                    "请在影像上点击布设起点" if target == "entry"
                    else "请在影像上点击布设终点"
                )
            return
        if target == "unload":
            self.state.entry_exit_mode = True
            self._entry_exit_click_idx = 2
            self._service_points_visible = True
            self._sync_drawing_mode()
            self.log_panel.info("请在影像上点击新增卸粮点；可连续添加多个")
            return
        self._toggle_entry_exit_mode()

    def _toggle_entry_exit_mode(self):
        self.state.entry_exit_mode = not getattr(self.state, 'entry_exit_mode', False)
        self._sync_drawing_mode()
        self._entry_exit_click_idx = 0
        self.log_panel.info(f"服务点布设: {'开启' if self.state.entry_exit_mode else '关闭'}")
        if self.state.entry_exit_mode:
            self._service_points_visible = True
            self._suggest_service_points()
            result = getattr(self.state, 'mask_result', None) or {}
            mask = result.get("processed_mask", getattr(self.state, 'mask_raw', None))
            if mask is not None:
                self._show_mask_overlay(mask)
            else:
                self._restore_visual_from_state()
            self._refresh_service_points_panel()
            self._sync_task_statuses()
            self.log_panel.info(
                "起点已推荐 · 可拖动调整；终点默认沿最后一段延伸，点地图可重设，"
                "卸粮点按需添加"
            )
        else:
            self._restore_visual_from_state()
        self.top_toolbar.set_action_active("ENTRY_EXIT", self.state.entry_exit_mode)

    def _place_entry_exit_point(self, x, y):
        """DECISION-007: changed service points invalidate route geometry."""
        self._service_points_visible = True
        had_path = bool(getattr(self.state, "auto_path_planned", False))
        idx = self._entry_exit_click_idx
        label = "设置起点" if idx == 0 else ("设置终点" if idx == 1 else "新增卸粮点")
        self._save_system_undo(label)
        if idx == 0:
            self.state.safe_update(entry_point=(x, y), entry_point_locked=True)
            self.log_panel.success(f"起点: ({x}, {y})")
        elif idx == 1:
            self.state.safe_update(exit_point=(x, y), exit_point_locked=True)
            self.log_panel.success(f"终点: ({x}, {y})")
        else:
            pts = list(getattr(self.state, 'unload_points', []) or [])
            pts.append((x, y))
            self.state.safe_update(
                unload_points=pts,
                unload_point=pts[0] if pts else None,
                unload_point_locked=True,
            )
            self.log_panel.success(f"卸粮点 #{len(pts)}: ({x}, {y})")
        if had_path:
            self._invalidate_analysis_from(
                "path",
                "起点、可选终点或卸粮点已变化，必须重新规划和验证",
            )
        self._restore_visual_from_state()
        self._refresh_service_points_panel()
        self._sync_task_statuses()
        if idx in (0, 1):
            # Start/end placement is explicit and one-shot.  Do not turn the
            # next accidental click into an optional end point.
            self.state.entry_exit_mode = False
            self._entry_exit_click_idx = 0
            self._sync_drawing_mode()
            self.top_toolbar.set_action_active("ENTRY_EXIT", False)
            if idx == 0:
                self.log_panel.info(
                    "起点已确认；终点会沿最终作业段自动延伸，可直接生成路径或拖动调整"
                )
        else:
            self._entry_exit_click_idx += 1

    def _on_segment(self):
        if self._has_background_worker():
            self.log_panel.warn("已有后台任务正在运行，请等待完成后再开始识别")
            return
        if self._tif_rgb is not None and not self.model_engine.is_loaded() and not self._ensure_model_loaded():
            self.log_panel.warn(
                "模型正在后台加载，请稍候" if getattr(self, "_model_load_in_progress", False)
                else "未找到内置模型，请检查 data/models"
            )
            return
        if self._tif_rgb is None:
            self.log_panel.warn("请先加载影像")
            return
        if not self.model_engine.is_loaded():
            self.log_panel.warn("请先加载模型")
            return
        model_path = str(getattr(self.state, "current_model_path", "") or "")
        if model_path:
            engine_model_path = str(getattr(self.model_engine, "_model_path", "") or "")
            try:
                same_loaded_path = bool(engine_model_path) and (
                    os.path.normcase(os.path.abspath(engine_model_path))
                    == os.path.normcase(os.path.abspath(model_path))
                )
            except (OSError, ValueError):
                same_loaded_path = False
            if not same_loaded_path:
                self.log_panel.warn(
                    "当前模型状态与实际加载权重不一致，已停止识别；请重新加载所选模型"
                )
                return
            try:
                from provenance import file_sha256
                current_model_sha256 = file_sha256(model_path)
            except OSError as exc:
                QMessageBox.warning(self, "模型文件不可用", f"无法读取当前模型文件：{exc}")
                return
            loaded_model_sha256 = str(getattr(self.state, "model_sha256", "") or "")
            if loaded_model_sha256 and loaded_model_sha256 != current_model_sha256:
                self.model_engine.unload()
                self._invalidate_analysis_from(
                    "inference",
                    "模型文件内容已在加载后发生变化，必须重新加载模型并重新识别",
                )
                self.log_panel.warn(
                    "模型内容已变化：当前 .pt 文件与已加载模型的哈希不一致，"
                    "已停止识别，请重新加载模型"
                )
                return
            self.state.model_sha256 = current_model_sha256
        if not self._sync_cache_model_identity():
            self.log_panel.info(
                "当前模型与历史缓存模型不同：本次将使用当前模型重新识别，"
                "不会读取历史掩膜"
            )
        if getattr(self.state, 'inference_running', False):
            self.log_panel.warn("作物识别正在运行")
            return
        self._save_system_undo("作物识别")
        self._invalidate_analysis_from(
            "inference",
            "已开始新的 作物识别，本次结果提交前不保留旧掩膜",
        )
        self.top_toolbar.set_progress(1, "作物识别中...")
        self.progress_bar.show()
        self.progress_bar.setValue(5)
        self.task_panel.set_task_status("segment", "in_progress", 5)
        self.lbl_status.setText("作物识别中...")

        from inference import InferenceRunner
        self._active_inference_token = make_task_token(self.state, 'inference')
        runner = InferenceRunner(
            self.state,
            self.model_engine,
            self.geo,
            token=self._active_inference_token,
        )
        self._active_inference_runner = runner
        runner.start_from_tif(
            self._tif_path,
            self._tif_rgb.shape[:2],
            self.state.field_boundary,
            max(1, int(getattr(self.state, 'downsample_factor', 1) or 1)),
        )
        self._pending_segment_display = True

    def _planning_preflight(self, mask_result):
        if not getattr(self.state, "forbidden_regions_confirmed", False):
            self.state.safe_update(forbidden_regions_confirmed=True)
        missing = []
        if len(getattr(self.state, "field_boundary", []) or []) < 3:
            missing.append("田块边界")
        if not isinstance(mask_result.get("planning_support_mask"), np.ndarray):
            missing.append("场地支撑层")
        if not self.geo.is_ready():
            missing.append("GeoTIFF 地理坐标")
        if missing:
            self.log_panel.warn("规划前检查未通过：缺少 " + "、".join(missing) + "，请补齐后再生成田间试验路径。")
            return False
        return True

    def _on_plan(self):
        self._log_action(
            "plan.request",
            "path",
            selected=len(
                getattr(self.state, "selected_centerline_ids", None)
                or getattr(self.state, "selected_work_band_ids", [])
                or []
            ),
            operator_lines=len(self._operator_paths_payload().get("lines") or []),
        )
        payload = self._operator_paths_payload()
        if payload.get("lines"):
            from row_geometry import MASK_REGULARIZATION_VERSION

            _overrides, report = operator_paths.planner_overrides(
                payload,
                mask_version=int(MASK_REGULARIZATION_VERSION),
                tif_path=str(getattr(self.state, "tif_path", "") or ""),
            )
            reason = str(report.get("reason") or "")
            if reason in {"mask_version_mismatch", "tif_mismatch"}:
                self.log_panel.warn(
                    "人工标定线已失效（%s），本次规划使用算法中心线；请在标定工具里重新标定。"
                    % ("掩膜算法版本已变化" if reason == "mask_version_mismatch" else "影像不一致")
                )
        mask_result = getattr(self.state, 'mask_result', None)
        if mask_result is None:
            self.log_panel.warn("请先完成掩膜优化")
            return
        catalog = getattr(self.state, "centerline_catalog", None) or getattr(
            self.state, "work_band_catalog", {}
        ) or {}
        if not catalog.get("items"):
            self.log_panel.warn("请先完成’生成作业线’，再在’选择作业线’中勾选要采用的作业线。")
            return
        selected_ids = list(
            getattr(self.state, "selected_centerline_ids", None)
            or getattr(self.state, "selected_work_band_ids", [])
            or []
        )
        if not selected_ids:
            self.log_panel.warn("请先在’选择作业线’中至少勾选一条作业线。")
            return
        hp = self._effective_harvester_params(getattr(self.state, 'harvester_params', {}) or {})
        self.state.harvester_params = hp
        ep = getattr(self.state, 'entry_point', None)
        xp = getattr(self.state, 'exit_point', None)
        if not ep or not getattr(self.state, 'entry_point_locked', False):
            self.log_panel.warn(
                "请先设置并确认起点；终点会沿最终作业段自动延伸，卸粮点按需添加。"
            )
            return
        if xp and not getattr(self.state, 'exit_point_locked', False):
            # Ignore stale auto-recommended end points restored from older
            # project caches.  Only an explicitly confirmed optional end may
            # influence the route.
            self.state.safe_update(exit_point=None, exit_point_locked=False)
            xp = None
        if not hp.get("cutter_width_m"):
            self.log_panel.warn("请先配置收割机参数")
            return
        if not hp.get("cutter_forward_offset_m"):
            self.log_panel.warn(
                "缺少割刀前置距离：请在农机参数中实测填写履带接地区域几何中心到"
                "割刀切割线的水平前向距离（影响田头端部作物掩膜几何覆盖）"
            )
            return
        if self._has_background_worker():
            self.log_panel.warn("任务正在运行：当前已有耗时任务正在执行，请等待完成后再生成路径。")
            return

        try:
            from pyside6_app.workers import require_metric_scale
            require_metric_scale(
                self.geo,
                self.state,
                mask_result.get("processed_mask").shape,
            )
        except Exception as exc:
            QMessageBox.warning(self, "地理尺度不可用", str(exc))
            return

        if not self._planning_preflight(mask_result):
            return

        self._save_system_undo("路径规划")
        self._invalidate_analysis_from(
            "path",
            "已开始新的路径规划，本次结果提交前不保留旧路径",
        )
        path_config = self._desktop_path_planning_runtime_config(hp)
        self._active_plan_token = make_task_token(self.state, "plan")
        worker = PlanWorker(
            mask_result,
            self.geo,
            self.state,
            path_config,
            detected_mask=getattr(self.state, 'inference_original_mask', None),
            token=self._active_plan_token,
        )
        self._active_plan_generation = self._active_plan_token.project_generation
        worker.finished.connect(self._on_plan_worker_done)
        self.state.plan_running = True
        self.state.plan_progress = 0.05
        self.top_toolbar.set_progress(5, "路径规划中...")
        self.progress_bar.show()
        self.progress_bar.setValue(5)
        self.task_panel.set_task_status("plan", "in_progress", 5)
        self.lbl_status.setText("路径规划中...")
        if not self._start_background_worker(worker, "路径规划已开始：按所选转弯策略生成轨迹，田头碾压不参与策略选择。"):
            self.state.plan_running = False
            self.state.plan_progress = 0.0
            self.progress_bar.hide()
            self.top_toolbar.set_progress(0)
            self._sync_task_statuses()

    def _on_run(self):
        self._log_action("inference.request", "segment")
        if self._tif_rgb is None:
            self.log_panel.warn("请先加载影像")
            return
        if not self.model_engine.is_loaded() and not self._ensure_model_loaded():
            self.log_panel.warn(
                "模型正在后台加载，请稍候" if getattr(self, "_model_load_in_progress", False)
                else "请先加载模型"
            )
            return
        self._on_segment()
    @Slot(int, str)
    def _on_progress(self, pct, msg):
        """进度只由**真实在跑的任务**驱动，绝不按百分比猜步骤。

        旧实现末尾有 `pct < 30 -> segment / < 70 -> process / else plan`
        三段兜底：任何一次进度事件（包括启动时加载模型）都会把"作物识别"
        标成"处理中 25%"，即使还没圈选田块、也没点过处理。任务行状态必须
        来自 state 里的运行标志，进度条只负责显示数值。
        """
        self.progress_bar.setValue(pct)
        self.top_toolbar.set_progress(pct, msg)
        self.set_status_text(msg, "busy")
        if getattr(self.state, 'inference_running', False):
            self.state.inference_progress = max(0.0, min(1.0, pct / 100.0))
            self.task_panel.set_task_status("segment", "in_progress", pct)
        elif getattr(self.state, 'mask_processing_running', False):
            self.state.mask_processing_progress = max(0.0, min(1.0, pct / 100.0))
            self.task_panel.set_task_status("process", "in_progress", pct)
        elif getattr(self.state, 'plan_running', False):
            self.state.plan_progress = max(0.0, min(1.0, pct / 100.0))
            self.task_panel.set_task_status("plan", "in_progress", pct)
        elif getattr(self, "_active_centerline_token", None) is not None:
            self.task_panel.set_task_status(
                "centerline_generate", "in_progress", pct, text="生成作业线"
            )
        # 模型加载等非流水线阶段：只更新状态栏和进度条，不碰任务行。

    @staticmethod
    def _coerce_points(points):
        clean = []
        for point in points or []:
            if hasattr(point, "pixel_x") and hasattr(point, "pixel_y"):
                clean.append((float(point.pixel_x), float(point.pixel_y)))
            elif len(point) >= 2:
                clean.append((float(point[0]), float(point[1])))
        return clean

    @staticmethod
    def _status_for_segment_type(segment_type):
        if segment_type == "work":
            return 1
        if segment_type == "entry":
            return 2
        if segment_type == "exit":
            return 3
        if segment_type in ("turn_reverse", "turn_aux"):
            return 4
        return 0

    @staticmethod
    def _segment_length(points):
        pts = MainWindow._coerce_points(points)
        return sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )

    def _normalise_path_result(self, path_result):
        if not isinstance(path_result, dict):
            return {}
        normalised = dict(path_result)
        full_path = normalised.get("full_path") or []
        segment_types = list(normalised.get("segment_types") or [])
        segment_geometry_modes = list(
            normalised.get("segment_geometry_modes")
            or normalised.get("geometry_modes")
            or []
        )

        if full_path:
            tracks, turns, formal_turns, approaches = [], [], [], []
            reverse_turns, service_segments, ordered = [], [], []
            for index, raw_points in enumerate(full_path):
                points = self._coerce_points(raw_points)
                if len(points) < 2:
                    continue
                segment_type = (
                    segment_types[index]
                    if index < len(segment_types)
                    else ("work" if index % 2 == 0 else "turn")
                )
                geometry_mode = self._normalise_route_geometry_mode(
                    segment_geometry_modes[index]
                    if index < len(segment_geometry_modes)
                    else None,
                    fallback=(
                        "turn"
                        if str(segment_type).startswith("turn")
                        else "straight"
                    ),
                )
                lengths_m = normalised.get("segment_lengths_m") or []
                display_length = (
                    float(lengths_m[index])
                    if index < len(lengths_m)
                    else self._segment_length(points)
                )
                item = {
                    "points": points,
                    "type": segment_type,
                    "segment_index": index,
                    "length": display_length,
                    "geometry_mode": geometry_mode,
                }
                if segment_type == "work":
                    layout = normalised.get("layout") or {}
                    region_sources = {
                        str(region.get("region_id")): region
                        for region in (layout.get("work_regions") or [])
                        if isinstance(region, dict) and region.get("region_id") is not None
                    }
                    region_routes = (
                        (normalised.get("turn_assessment") or {}).get(
                            "region_routes"
                        )
                        or []
                    )
                    for route in region_routes:
                        if not isinstance(route, dict):
                            continue
                        start = route.get("segment_start_index")
                        end = route.get("segment_end_index")
                        if start is None or end is None:
                            continue
                        if not (int(start) <= index <= int(end)):
                            continue
                        region_id = route.get("region_id")
                        region = region_sources.get(str(region_id), {})
                        item["region_id"] = region_id
                        item["region_kind"] = route.get("kind") or region.get("kind")
                        item["line_source"] = (
                            region.get("line_source")
                            or (
                                "cutter_width_uniform_swaths"
                                if item["region_kind"] == "uniform_block"
                                else "four_row_band_centerlines"
                            )
                        )
                        break
                    if not item.get("line_source") and len(region_routes) == 1:
                        route = region_routes[0]
                        region_id = route.get("region_id") if isinstance(route, dict) else None
                        region = region_sources.get(str(region_id), {})
                        item["region_id"] = region_id
                        item["region_kind"] = (
                            (route.get("kind") if isinstance(route, dict) else None)
                            or region.get("kind")
                        )
                        item["line_source"] = (
                            region.get("line_source")
                            or (
                                "cutter_width_uniform_swaths"
                                if item["region_kind"] == "uniform_block"
                                else "four_row_band_centerlines"
                            )
                        )
                ordered.append(item)
                if segment_type == "work":
                    tracks.append(item)
                else:
                    turns.append(item)
                    if segment_type == "turn":
                        formal_turns.append(item)
                    elif segment_type == "turn_approach":
                        approaches.append(item)
                    elif segment_type in ("turn_reverse", "turn_aux"):
                        reverse_turns.append(item)
                    elif segment_type in ("entry", "exit", "unload"):
                        service_segments.append(item)
            normalised["full_path"] = [item["points"] for item in ordered]
            normalised["segment_types"] = [item["type"] for item in ordered]
            normalised["segment_geometry_modes"] = [
                item["geometry_mode"] for item in ordered
            ]
            normalised["tracks"] = tracks
            normalised["turns"] = turns
            normalised["formal_turns"] = formal_turns
            normalised["turn_approaches"] = approaches
            normalised["reverse_turns"] = reverse_turns
            normalised["service_segments"] = service_segments
            normalised["ordered_segments"] = ordered
        else:
            tracks = normalised.get("tracks", [])
            turns = normalised.get("turns", [])
            if not normalised.get("segment_geometry_modes"):
                normalised["segment_geometry_modes"] = [
                    self._normalise_route_geometry_mode(
                        item.get("geometry_mode") if isinstance(item, dict) else None,
                        fallback=(
                            "turn"
                            if str(item.get("type", "")).startswith("turn")
                            else "straight"
                        ),
                    )
                    for item in list(tracks) + list(turns)
                ]
            normalised.setdefault("formal_turns", list(turns))
            normalised.setdefault("turn_approaches", [])
            normalised.setdefault("reverse_turns", [])
            normalised.setdefault("service_segments", [])
            normalised["ordered_segments"] = list(tracks) + list(turns)
        return normalised

    @staticmethod
    def _path_result_point_count(path_result):
        return sum(
            len(segment.get("points", []) or [])
            for segment in (path_result.get("ordered_segments", []) or [])
            if isinstance(segment, dict)
        )

    def _verify_scientific_provenance(self, path_result):
        """Verify every cached artifact against current inputs before export."""
        from inference import INFERENCE_PIPELINE_VERSION, build_inference_config
        from path_planner import PATH_PLANNING_VERSION
        from provenance import ProvenanceError, file_sha256, verify_stage_record
        from pyside6_app.workers import (
            path_planning_config_snapshot,
            planning_config_with_mask_evidence,
            semantic_mask_fingerprints,
        )
        from row_geometry import MASK_REGULARIZATION_VERSION

        raw_mask = getattr(self.state, "mask_raw", None)
        processed_result = getattr(self.state, "mask_result", None) or {}
        processed_mask = processed_result.get("processed_mask")
        if not isinstance(raw_mask, np.ndarray) or not isinstance(processed_mask, np.ndarray):
            raise ProvenanceError("缺少可验证的原始或处理后掩膜")

        inference_record = dict(getattr(self.state, "inference_provenance", {}) or {})
        inference_inputs = dict(inference_record.get("inputs") or {})
        model_path = str(getattr(self.state, "current_model_path", "") or "")
        if not model_path or not os.path.isfile(model_path):
            # 模型文件缺失不改变已生成掩膜/路径的坐标本身，但会破坏"由哪个
            # 模型生成"的溯源完整性；confirm 模式下展示风险并由用户决定。
            self._confirm_export_risk(
                "导出确认",
                "生成识别结果的模型文件不存在；本次导出将沿用识别时记录的模型"
                "指纹，请确认该溯源记录可信后再导出",
            )
            model_sha256 = str(inference_inputs.get("model_sha256", "") or "")
        else:
            model_sha256 = file_sha256(model_path)
        source_sha256 = str(getattr(self.state, "source_sha256", "") or "")
        if not source_sha256:
            raise ProvenanceError("源影像完整 SHA-256 缺失")
        inference_inputs.update({
            "source_sha256": source_sha256,
            "model_sha256": model_sha256,
            "inference_config": build_inference_config(
                self.cfg,
                getattr(self, "model_engine", None),
            ),
            "preprocessing_config": self.cfg.section("raster_preprocessing"),
        })
        if str(inference_record.get("algorithm_version", "")) != INFERENCE_PIPELINE_VERSION:
            raise ProvenanceError("AI 推理算法版本不一致")
        verify_stage_record(inference_record, "inference", inference_inputs, raw_mask)

        mask_record = dict(
            getattr(self.state, "mask_provenance", {})
            or processed_result.get("provenance")
            or {}
        )
        mask_inputs = dict(mask_record.get("inputs") or {})
        mask_inputs.update({
            "inference_fingerprint": str(inference_record.get("fingerprint", "")),
            "raw_mask_sha256": str(inference_record.get("artifact_sha256", "")),
            "mask_config": self.cfg.section("mask_processing"),
        })
        if str(mask_record.get("algorithm_version", "")) != str(MASK_REGULARIZATION_VERSION):
            raise ProvenanceError("掩膜优化算法版本不一致")
        verify_stage_record(mask_record, "mask", mask_inputs, processed_mask)

        path_record = dict(
            path_result.get("provenance")
            or getattr(self.state, "path_provenance", {})
            or {}
        )
        current_harvester = self._effective_harvester_params(
            getattr(self.state, "harvester_params", {}) or {}
        )
        path_config = path_planning_config_snapshot(
            self._desktop_path_planning_runtime_config(current_harvester)
        )
        semantic_config = planning_config_with_mask_evidence(
            path_config,
            processed_result,
            self.state,
            processed_mask.shape,
        )
        path_inputs = dict(path_record.get("inputs") or {})
        path_inputs.update({
            "mask_fingerprint": str(mask_record.get("fingerprint", "")),
            "path_config": path_config,
            "harvester": current_harvester,
            "harvester_param_sources": self._normalise_harvester_param_sources(
                current_harvester,
                getattr(self.state, "harvester_param_sources", {}) or {},
            ),
            "turn_strategy": str(getattr(self.state, "turn_strategy", "auto")),
            "entry_point": getattr(self.state, "entry_point", None),
            "exit_point": getattr(self.state, "exit_point", None),
            "unload_points": list(getattr(self.state, "unload_points", []) or []),
            "forbidden_regions": list(getattr(self.state, "forbidden_regions", []) or []),
            "forbidden_regions_confirmed": bool(
                getattr(self.state, "forbidden_regions_confirmed", False)
            ),
            "semantic_mask_fingerprints": semantic_mask_fingerprints(
                semantic_config
            ),
        })
        path_artifact = {
            "full_path": path_result.get("full_path", []),
            "segment_types": path_result.get("segment_types", []),
            "layout": path_result.get("layout", {}),
        }
        if str(path_record.get("algorithm_version", "")) != PATH_PLANNING_VERSION:
            raise ProvenanceError("路径规划算法版本不一致")
        verify_stage_record(path_record, "path", path_inputs, path_artifact)
        return {
            "inference": inference_record,
            "mask": mask_record,
            "path": path_record,
        }

    def _export_gate_mode(self) -> str:
        """Export gate behaviour: 'confirm' (default) or 'fail_closed'.

        Research output keeps the computed route metrics and provenance in the
        manifest, but does not turn theoretical footprint thresholds into a
        file-format/export blocker.  ``fail_closed`` remains available for
        strict CI or controlled-trial runs.  The ``ALIGN_EXPORT_GATE``
        environment variable overrides the config so automated test suites can
        select a deterministic gate.
        """
        env_mode = str(os.environ.get("ALIGN_EXPORT_GATE", "")).strip().lower()
        if env_mode:
            return env_mode if env_mode in {"fail_closed", "confirm"} else "fail_closed"
        try:
            mode = (self.cfg.section("export") or {}).get("gate_mode", "confirm")
        except Exception:
            mode = "confirm"
        return str(mode or "confirm").strip().lower() or "confirm"

    def _confirm_export_risk(self, title: str, detail: str) -> None:
        """Apply the export gate: abort in ``fail_closed``, notify in ``confirm``.

        ``confirm`` (default) lets the export proceed but must leave a
        user-visible notice in the log panel — the operator asked for
        "放行 + 有提醒", so silently returning would hide a real risk.
        ``fail_closed`` raises ``ValueError(detail)`` and aborts the export.
        Callers must word their message as "fail_closed 模式将中止导出"
        rather than "禁止导出", otherwise the prompt contradicts the default.
        """
        if self._export_gate_mode() != "confirm":
            raise ValueError(detail)
        notice = f"{title}：{detail}"
        if not hasattr(self, "_export_risk_notices"):
            self._export_risk_notices = []
        if notice not in self._export_risk_notices:
            self._export_risk_notices.append(notice)
        panel = getattr(self, "log_panel", None)
        if panel is not None:
            panel.warn(f"{notice}（已放行）")

    def _record_compatibility_preview_risk(self, detail: str) -> None:
        """Surface compatibility diagnostics in the log without blocking."""
        panel = getattr(self, "log_panel", None)
        if panel is not None:
            panel.info(f"预览提示：{detail}")

    @staticmethod
    def _layout_has_planning_recovery(layout) -> bool:
        """Detect explicit or nested cached automatic work-band recovery."""
        def walk(value):
            if isinstance(value, dict):
                if value.get("planning_recovered") is True:
                    return True
                return any(walk(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(walk(item) for item in value)
            return False

        return walk(layout)

    def _enforce_formal_parameter_and_recovery_gate(self, path_result):
        """Report provenance/recovery risks without hiding them in exports.

        The route data still carries the original source and recovery flags.
        In strict ``fail_closed`` mode these conditions remain blocking; in
        the normal research export mode they are recorded as warnings so a
        reproducible research artifact can still be written.
        """
        path_result = path_result or {}
        validation = path_result.get("validation", {}) or {}
        profile = str(
            validation.get("validation_profile")
            or self.cfg.section("path_planning").get("validation_profile", "field_trial")
        )
        if profile not in {"field_trial", "machine_candidate"}:
            return
        planning_factors = path_result.get("planning_factors", {}) or {}
        source_status = dict(
            planning_factors.get("harvester_parameter_source_status")
            or (path_result.get("machine_readiness", {}) or {}).get(
                "harvester_parameter_source_status"
            )
            or {}
        )
        source_map = dict(
            planning_factors.get("harvester_param_sources")
            or source_status.get("sources")
            or {}
        )
        required_params = list(
            (planning_factors.get("harvester") or {}).keys()
            or source_status.get("required")
            or []
        )
        unverified = [
            key for key in required_params
            if source_map.get(key) != "user_measured"
        ]
        if unverified:
            self._confirm_export_risk(
                "导出确认",
                "农机参数来源未完成实测记录，fail_closed 模式将中止导出："
                + ", ".join(unverified)
            )
        if self._layout_has_planning_recovery(path_result.get("layout", {})):
            self._confirm_export_risk(
                "导出确认",
                "当前缓存路径包含 planning_recovered=True 的自动恢复作业带，"
                "fail_closed 模式将中止导出；请重新规划或逐条人工确认后再导出"
            )

    def _geo_export_payload(self, *, formal=True):
        """DECISION-001: validate export data before serializing coordinates.

        GeoJSON, CSV, KML, JSON and $PATH must all pass this function. The
        checks are intentionally redundant with stage validation because a
        route may enter a field trial. Research mode keeps computed metrics in
        the manifest without turning them into a hard export blocker; source,
        coordinate and route-data consistency checks remain mandatory.
        See docs/DECISIONS.md and the DECISION-001 invariant tests.
        """
        if not self.geo.is_ready():
            raise ValueError("当前影像没有可用地理坐标，不能导出经纬度文件")
        if self._tif_path and self._tif_source_identity:
            from cache import source_identity
            if source_identity(self._tif_path) != self._tif_source_identity:
                raise ValueError("GeoTIFF 文件在加载后已发生变化，请重新打开影像并重新规划")
        path_result = self._path_result_from_state()
        if not path_result.get("ordered_segments"):
            raise ValueError("当前没有可导出的路径，请先生成路径")
        validation = path_result.get("validation", {}) or {}
        manual_compatibility_route = bool(
            not formal
            and str(validation.get("route_geometry_source") or "").startswith(
                "manual_control_edit"
            )
        )
        if validation.get("validation_source") == "cache_snapshot" or not bool(
            validation.get("current_validation_complete", True)
        ):
            detail = "路径来自缓存快照，尚未使用当前输入重新验证"
            if formal:
                self._confirm_export_risk(
                    "导出确认",
                    detail + "，fail_closed 模式将中止导出",
                )
            else:
                self._record_compatibility_preview_risk(detail)
        path_valid = bool(
            path_result.get("is_valid", validation.get("valid", False))
            and validation.get("valid", path_result.get("is_valid", False))
        )
        hard_reasons = list(
            (path_result.get("turn_assessment", {}) or {}).get("hard_reasons") or []
        )
        if bool((path_result.get("turn_assessment", {}) or {}).get("kinematically_infeasible")):
            hard_reasons.append("路径包含仅供显示的不可执行直线连接")
        if not path_valid or hard_reasons:
            issues = list(validation.get("issues") or []) + hard_reasons
            detail = "；".join(str(item) for item in issues[:5]) or "路径未经验证"
            if manual_compatibility_route:
                self._record_compatibility_preview_risk(detail)
            else:
                self._confirm_export_risk(
                    "导出确认",
                    f"路径未通过安全验证，fail_closed 模式将中止导出：{detail}",
                )

        planned_harvester = (
            (path_result.get("planning_factors", {}) or {}).get("harvester", {}) or {}
        )
        if planned_harvester:
            current_harvester = self._effective_harvester_params(
                getattr(self.state, "harvester_params", {}) or {}
            )
            changed = []
            for key, value in planned_harvester.items():
                if key not in current_harvester:
                    continue
                current_value = current_harvester[key]
                if value is None or current_value is None:
                    if value is not current_value:
                        changed.append(key)
                elif abs(float(value) - float(current_value)) > 1e-9:
                    changed.append(key)
            if changed:
                detail = (
                    "当前农机参数与路径规划时不一致："
                    + ", ".join(changed)
                )
                if formal:
                    self._confirm_export_risk(
                        "导出确认",
                        detail + "，必须重新规划",
                    )
                else:
                    self._record_compatibility_preview_risk(detail)
        if formal:
            self._enforce_formal_parameter_and_recovery_gate(path_result)
        elif self._layout_has_planning_recovery(path_result.get("layout", {})):
            self._record_compatibility_preview_risk(
                "路径包含 planning_recovered=True 的自动恢复作业带"
            )
        requested_strategy = path_result.get("requested_strategy")
        if (
            requested_strategy
            and str(requested_strategy) != str(getattr(self.state, "turn_strategy", "auto"))
        ):
            detail = "当前调头策略与路径规划时不一致"
            if formal:
                self._confirm_export_risk(
                    "导出确认",
                    detail + "，必须重新规划",
                )
            else:
                self._record_compatibility_preview_risk(detail)
        full_path = path_result.get("full_path", [])
        segment_types = path_result.get("segment_types", [])
        display_h, display_w = (
            self._tif_rgb.shape[:2]
            if isinstance(self._tif_rgb, np.ndarray)
            else (
                int(getattr(self.state, "display_h", 0) or 0),
                int(getattr(self.state, "display_w", 0) or 0),
            )
        )
        if display_w > 0 and display_h > 0:
            for segment_index, segment in enumerate(full_path):
                for point_index, (pixel_x, pixel_y) in enumerate(segment):
                    if not (
                        0.0 <= float(pixel_x) <= float(display_w - 1)
                        and 0.0 <= float(pixel_y) <= float(display_h - 1)
                    ):
                        raise ValueError(
                            "路径点超出影像范围，禁止使用仿射外推坐标: "
                            f"segment={segment_index}, point={point_index}, "
                            f"pixel=({float(pixel_x):.3f},{float(pixel_y):.3f})"
                        )
        field_boundary = getattr(self.state, "field_boundary", None) or []
        if len(field_boundary) >= 3:
            from path_planner import segment_within_field_boundary
            from row_geometry import sanitize_field_polygon

            # Same geometry the planner gates on: a self-intersecting outline
            # would otherwise reject every segment (fail closed for the wrong
            # reason) instead of the repaired ring the operator drew.
            polygon = sanitize_field_polygon(field_boundary)
            if polygon is not None:
                for segment_index, segment in enumerate(full_path):
                    segment_type = (
                        segment_types[segment_index]
                        if segment_index < len(segment_types)
                        else ("work" if segment_index % 2 == 0 else "turn")
                    )
                    if not segment_within_field_boundary(segment, polygon):
                        self._confirm_export_risk(
                            "导出确认",
                            "route segment leaves field boundary; fail_closed mode will abort export: "
                            f"segment={segment_index}, type={segment_type}",
                        )
        if manual_compatibility_route:
            # A dragged control route is intentionally no longer the untouched
            # output of the automatic planner, so its old path-stage
            # fingerprint cannot remain valid. Compatibility export still
            # recomputes every coordinate from the current on-screen pixels
            # below, after the GeoTIFF identity, image bounds and field checks
            # above. Formal scientific export continues to require the full
            # provenance chain.
            path_result["manual_route_edited"] = True
        else:
            self._verify_scientific_provenance(path_result)
        expected_count = self._path_result_point_count(path_result)
        from path_planner import global_path_to_geo, validate_geo_points

        # Pixel geometry shown in the viewer is authoritative. Recompute at
        # export time so stale cache data can never diverge from the route on
        # screen, then validate every point's segment and pixel identity.
        geo_points = global_path_to_geo(full_path, self.geo, segment_types)
        geo_points = validate_geo_points(
            geo_points,
            expected_count=expected_count,
            path_segments=full_path,
            segment_types=segment_types,
        )
        path_result["geo_points"] = geo_points
        self.state.auto_path_geo = geo_points
        return path_result, geo_points

    def _show_geo_export_error(self, label, error):
        message = (
            f"{label} 导出已停止。\n\n"
            f"{error}\n\n"
            "请检查当前路径和 GeoTIFF 坐标信息后重试。"
        )
        self.log_panel.error(message.replace("\n", " "))
        QMessageBox.warning(self, f"{label} 导出失败", message)

    @staticmethod
    def _manifest_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return MainWindow._manifest_jsonable(value.item())
        if isinstance(value, dict):
            return {
                str(key): MainWindow._manifest_jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [MainWindow._manifest_jsonable(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _sha256_file(self, path, use_cache=True):
        path = os.path.abspath(str(path or ""))
        if not path or not os.path.isfile(path):
            return ""
        stat = os.stat(path)
        key = (path, int(stat.st_size), int(stat.st_mtime_ns))
        if use_cache and key in self._file_hash_cache:
            return self._file_hash_cache[key]
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        value = digest.hexdigest()
        if use_cache:
            self._file_hash_cache[key] = value
        return value

    @staticmethod
    def _snapshot_machine_export_paths(output_path):
        """Remember existing files so a failed new export never removes them."""
        requested = os.path.abspath(os.fspath(output_path))
        suffix = os.path.splitext(requested)[1]
        if os.path.isdir(requested) or not suffix:
            if not os.path.isdir(requested):
                return set()
            try:
                return {
                    os.path.abspath(os.path.join(requested, name))
                    for name in os.listdir(requested)
                    if os.path.isfile(os.path.join(requested, name))
                }
            except OSError:
                return set()
        preserved = {requested} if os.path.exists(requested) else set()
        manifest_path = requested + ".manifest.json"
        if os.path.exists(manifest_path):
            preserved.add(manifest_path)
        return preserved

    @staticmethod
    def _cleanup_machine_export_artifacts(
        protocol_metadata,
        manifest_path,
        preserved_paths=None,
    ):
        """Remove only files created by the failed compatibility export."""
        preserved = {
            os.path.abspath(os.fspath(path))
            for path in (preserved_paths or set())
        }
        paths = []
        if isinstance(protocol_metadata, dict):
            paths.extend(protocol_metadata.get("output_files") or [])
        if manifest_path:
            paths.append(manifest_path)
        for path in paths:
            absolute = os.path.abspath(os.fspath(path))
            if absolute in preserved:
                continue
            try:
                os.remove(absolute)
            except OSError:
                pass

    @staticmethod
    def _machine_route_preview_integrity_errors(preview):
        """Recheck the files before displaying a cached machine-file preview."""
        if not isinstance(preview, dict):
            return []
        manifest_path = str(preview.get("manifest_path") or "")
        if not manifest_path:
            # Older in-memory test/integration previews have no manifest path.
            return []
        manifest_path = os.path.abspath(manifest_path)
        if not os.path.isfile(manifest_path):
            return [f"导出清单不存在: {manifest_path}"]
        from integrity_check import verify_manifest

        try:
            return list(verify_manifest(manifest_path))
        except Exception as exc:
            return [f"导出清单无法读取: {manifest_path} ({exc})"]

    @classmethod
    def _verify_machine_export_manifest(cls, manifest_path):
        """Fail closed when the just-written TXT and manifest disagree."""
        manifest_path = os.path.abspath(os.fspath(manifest_path or ""))
        if not manifest_path or not os.path.isfile(manifest_path):
            # The production manifest writer always returns an existing file;
            # test integrations may intentionally replace it with a marker.
            return []
        from integrity_check import verify_manifest

        try:
            errors = list(verify_manifest(manifest_path))
        except Exception as exc:
            errors = [f"导出清单无法读取: {manifest_path} ({exc})"]
        if errors:
            detail = "；".join(str(error) for error in errors[:6])
            if len(errors) > 6:
                detail += f"；另有 {len(errors) - 6} 项"
            raise ValueError(f"农机兼容 TXT 完整性校验失败：{detail}")
        return []

    def _write_export_manifest(
        self,
        output_path,
        output_format,
        path_result,
        geo_points,
        protocol_metadata=None,
    ):
        """DECISION-001: make a route inseparable from its audit evidence.

        A manifest failure is an export failure; callers must not report a
        route as complete without its source, model, code and stage hashes.
        """
        from importlib import metadata
        from cache import source_identity
        from build_info import SCIENTIFIC_CODE_PATHS, load_build_info
        from provenance import APP_VERSION, code_identity, git_is_dirty, git_revision
        from runtime_paths import is_frozen, project_root as runtime_project_root

        package_names = (
            "numpy", "opencv-python", "PySide6", "pyproj",
            "rasterio", "shapely", "ultralytics",
        )
        versions = {}
        for package in package_names:
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "not-installed"

        config_data = self._manifest_jsonable(self.cfg.snapshot())
        config_bytes = json.dumps(
            config_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        route_data = self._manifest_jsonable(geo_points)
        route_bytes = json.dumps(
            route_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        model_path = str(getattr(self.state, "current_model_path", "") or "")
        mask_result = getattr(self.state, "mask_result", None) or {}
        diagnostics = mask_result.get("diagnostics", {}) if isinstance(mask_result, dict) else {}
        extent = {}
        source_metadata = dict(getattr(self.state, "source_metadata", {}) or {})
        if self.geo.is_ready() and isinstance(self._tif_rgb, np.ndarray):
            height, width = self._tif_rgb.shape[:2]
            extent = self.geo.validate_raster_extent(width, height)
            center = (width / 2.0, height / 2.0)
            gsd_x = float(self.geo.pixel_distance_m(center, (center[0] + 1.0, center[1])))
            gsd_y = float(self.geo.pixel_distance_m(center, (center[0], center[1] + 1.0)))
            source_metadata["display_gsd_m"] = {
                "x": gsd_x,
                "y": gsd_y,
                "anisotropy_ratio": max(gsd_x, gsd_y) / min(gsd_x, gsd_y),
            }

        source_sha256 = str(getattr(self.state, "source_sha256", "") or "")
        if not source_sha256:
            raise ValueError("源影像完整 SHA-256 缺失，不能生成科研溯源清单")

        build_info = load_build_info()
        # ``main_window.py`` is under ``src/pyside6_app`` after source
        # organization; resolve the project root through the shared runtime
        # helper instead of assuming the old flat layout.
        project_root = str(runtime_project_root())
        if is_frozen():
            scientific_code = dict(build_info.get("scientific_code") or {})
            if not scientific_code.get("fingerprint"):
                raise ValueError("冻结包缺少科学代码构建指纹，不能生成溯源清单")
            build_revision = str(build_info.get("commit_sha") or "")
            build_dirty = build_info.get("git_dirty")
        else:
            scientific_code = code_identity(project_root, SCIENTIFIC_CODE_PATHS)
            build_revision = git_revision(project_root)
            build_dirty = git_is_dirty(project_root)
        readiness = dict(path_result.get("machine_readiness") or {})
        # DECISION-001/011: manifest classification must verify the complete
        # readiness assessment; a caller-supplied boolean must never upgrade a
        # route to controlled field-trial status.
        controlled_trial = bool(
            readiness.get("schema") == "align.machine-readiness.v1"
            and readiness.get("controlled_trial_ready") is True
            and readiness.get("eligible_for_machine_export") is True
            and readiness.get("controlled_trial_blockers") == []
            and readiness.get("blockers") == []
            and readiness.get("manual_review_required") is False
            and readiness.get("current_route_classification")
            == "machine_export_candidate"
            and readiness.get("terminal_protocol_validated") is True
        )
        output_metadata = {
            "path": os.path.abspath(output_path),
            "format": str(output_format),
            "sha256": self._sha256_file(output_path, use_cache=False),
            "point_count": len(geo_points),
            "coordinate_crs": "EPSG:4326",
            "internal_path_reference_point": str(
                (path_result.get("validation", {}) or {}).get(
                    "footprint_reference_point", "track_contact_center"
                )
            ),
            "footprint_model": str(
                (path_result.get("validation", {}) or {}).get(
                    "footprint_model", "planar_rigid_footprint_v2"
                )
            ),
            "route_classification": (
                "controlled_field_trial" if controlled_trial
                else "research_manual_review"
            ),
            "machine_executable": False,
            "usage_restriction": (
                "CONTROLLED_FIELD_TRIAL_LOW_SPEED_MANUAL_SUPERVISION"
                if controlled_trial else "RESEARCH_MANUAL_REVIEW_ONLY"
            ),
        }
        output_format_upper = str(output_format).upper()
        if output_format_upper in {"$PATH", "$PATH_MACHINE_COMPAT"}:
            compatibility_protocol = output_format_upper == "$PATH_MACHINE_COMPAT"
            compatibility_preview = bool(
                compatibility_protocol
                and isinstance(protocol_metadata, dict)
                and protocol_metadata.get("preview_only") is True
            )
            output_metadata["candidate_machine_path"] = True
            output_metadata["terminal_protocol_validated"] = bool(
                controlled_trial
                and readiness.get("terminal_protocol_validated") is True
            )
            output_metadata["machine_executable"] = output_metadata[
                "terminal_protocol_validated"
            ]
            path_protocol = {
                "record_prefix": "$PATH",
                "coordinate_order": ["latitude", "longitude"],
                "point_index_base": 0,
                "record_terminator": "*",
                "protocol_variant": (
                    "machine_harvester_straight_turn_v1"
                    if compatibility_protocol
                    else "research_compatibility"
                ),
                "route_id": 0 if compatibility_protocol else self.cfg.ROUTE_ID,
                "status_codes": (
                    {
                        "file_record_status": 2,
                        "internal_control_route": {
                            "turn_or_other": 0,
                            "work": 1,
                        },
                    }
                    if compatibility_protocol
                    else {
                        "turn_or_other": 0,
                        "work": 1,
                        "entry": 2,
                        "exit": 3,
                        "turn_reverse_or_aux": 4,
                    }
                ),
                "final_record": (
                    "$INFM,0,0.7,0.7,0.7,0.7,0,5,0*"
                    if compatibility_protocol
                    else self.cfg.INFM_LINE
                ),
            }
            if compatibility_protocol:
                path_protocol.update(self._manifest_jsonable(protocol_metadata or {}))
                output_files = list(path_protocol.get("output_files") or [])
                if not output_files:
                    output_files = [output_path]
                points_per_file = list(path_protocol.get("points_per_file") or [])
                output_file_integrity = []
                for index, file_path in enumerate(output_files):
                    absolute = os.path.abspath(os.fspath(file_path))
                    if not os.path.isfile(absolute):
                        raise ValueError(f"农机兼容 TXT 缺失，不能生成清单: {absolute}")
                    item = {
                        "path": absolute,
                        "sha256": self._sha256_file(absolute, use_cache=False),
                        "size_bytes": int(os.path.getsize(absolute)),
                    }
                    if index < len(points_per_file):
                        item["point_count"] = int(points_per_file[index])
                    output_file_integrity.append(item)
                path_protocol["output_file_integrity"] = output_file_integrity
            if compatibility_preview:
                output_metadata["preview_only"] = True
                output_metadata["route_classification"] = "compatibility_preview"
                output_metadata["usage_restriction"] = (
                    "COMPATIBILITY_TEST_ONLY_MANUAL_REVIEW"
                )
                output_metadata["terminal_protocol_validated"] = False
                output_metadata["machine_executable"] = False
            output_metadata["path_protocol"] = path_protocol
        manifest = {
            "schema": 2,
            "application_version": str(build_info.get("version") or APP_VERSION),
            "git_revision": build_revision,
            "git_dirty": build_dirty,
            "build_info": self._manifest_jsonable(build_info),
            "scientific_code": scientific_code,
            "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "output": output_metadata,
            "source_image": {
                "path": os.path.abspath(self._tif_path) if self._tif_path else "",
                "identity": source_identity(self._tif_path) if self._tif_path else {},
                "sha256": source_sha256,
                "metadata": self._manifest_jsonable(
                    source_metadata
                ),
                "geographic_extent": extent,
            },
            "model": {
                "name": str(getattr(self.state, "current_model_name", "") or ""),
                "path": os.path.abspath(model_path) if model_path else "",
                "sha256": self._sha256_file(model_path),
                "registry": self._manifest_jsonable(
                    self._model_registry_entry(model_path)
                ),
            },
            "analysis": {
                "mask_algorithm_version": diagnostics.get("algorithm_version"),
                "mask_strength": diagnostics.get("strength", self._mask_processing_strength),
                "fallback_pipeline": bool(diagnostics.get("fallback_pipeline", False)),
                "harvester_params": self._manifest_jsonable(
                    getattr(self.state, "harvester_params", {}) or {}
                ),
                "harvester_param_sources": self._manifest_jsonable(
                    getattr(self.state, "harvester_param_sources", {}) or {}
                ),
                "turn_strategy": str(getattr(self.state, "turn_strategy", "")),
                "validation": self._manifest_jsonable(
                    path_result.get("validation", {}) or {}
                ),
                "layout": self._manifest_jsonable(
                    path_result.get("layout", {}) or {}
                ),
                "machine_readiness": self._manifest_jsonable(
                    path_result.get("machine_readiness", {}) or {}
                ),
                "forbidden_regions": self._manifest_jsonable(
                    getattr(self.state, "forbidden_regions", []) or []
                ),
                "forbidden_regions_confirmed": bool(
                    getattr(self.state, "forbidden_regions_confirmed", False)
                ),
                "planning_factors": self._manifest_jsonable(
                    path_result.get("planning_factors", {}) or {}
                ),
                "stage_provenance": {
                    "inference": self._manifest_jsonable(
                        getattr(self.state, "inference_provenance", {}) or {}
                    ),
                    "mask": self._manifest_jsonable(
                        getattr(self.state, "mask_provenance", {}) or {}
                    ),
                    "path": self._manifest_jsonable(
                        path_result.get("provenance")
                        or getattr(self.state, "path_provenance", {})
                        or {}
                    ),
                },
            },
            "integrity": {
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "route_data_sha256": hashlib.sha256(route_bytes).hexdigest(),
            },
            "runtime": {
                "python": sys.version,
                "packages": versions,
                "gdal_data": os.environ.get("GDAL_DATA", ""),
                "proj_data": os.environ.get("PROJ_DATA", ""),
                "stages": {
                    "inference": self._manifest_jsonable(
                        getattr(self.state, "inference_runtime", {}) or {}
                    ),
                    "mask": self._manifest_jsonable(
                        (getattr(self.state, "mask_result", None) or {}).get("runtime", {})
                    ),
                    "path": self._manifest_jsonable(
                        path_result.get("runtime")
                        or getattr(self.state, "path_runtime", {})
                        or {}
                    ),
                },
            },
        }
        manifest_path = str(output_path) + ".manifest.json"
        try:
            save_json_atomic(manifest_path, manifest)
        except Exception:
            cleanup_paths = [output_path]
            if isinstance(protocol_metadata, dict):
                cleanup_paths.extend(protocol_metadata.get("output_files") or [])
            for cleanup_path in set(cleanup_paths):
                try:
                    os.remove(cleanup_path)
                except OSError:
                    pass
            raise
        return manifest_path

    def _auto_segments_from_path(self, path_result):
        path_result = self._normalise_path_result(path_result)
        segments = []
        geometry_modes = list(
            path_result.get("segment_geometry_modes")
            or path_result.get("geometry_modes")
            or []
        )
        for index, points in enumerate(path_result.get("full_path", [])):
            segment_type = path_result.get("segment_types", ["work"])[index]
            pts = [PathPoint(float(x), float(y)) for x, y in self._coerce_points(points)]
            if len(pts) < 2:
                continue
            lengths_m = path_result.get("segment_lengths_m") or []
            length_m = (
                float(lengths_m[index])
                if index < len(lengths_m)
                else self._segment_length(points)
            )
            segments.append(AutoPathSegment(
                points=pts,
                status=self._status_for_segment_type(segment_type),
                corridor_id=index,
                length_m=length_m,
                segment_type=segment_type,
                geometry_mode=(
                    self._normalise_route_geometry_mode(
                        geometry_modes[index]
                        if index < len(geometry_modes)
                        else None,
                        fallback=(
                            "turn"
                            if str(segment_type).startswith("turn")
                            else "straight"
                        ),
                    )
                ),
            ))
        return segments

    def _restore_cached_path_geometry(self, saved):
        """Rebuild in-memory route geometry from project_state.json cache."""
        cached_segments = []
        for index, item in enumerate(saved.get("auto_path") or []):
            if not isinstance(item, dict):
                continue
            points = self._coerce_points(item.get("points") or [])
            if len(points) < 2:
                continue
            segment_type = str(item.get("segment_type") or ("work" if index % 2 == 0 else "turn"))
            cached_length = item.get("length_m")
            if cached_length is None:
                return {}
            cached_segments.append({
                "points": points,
                "type": segment_type,
                "segment_index": index,
                "length": float(cached_length),
                "geometry_mode": self._normalise_route_geometry_mode(
                    item.get("geometry_mode"),
                    fallback=(
                        "turn"
                        if segment_type.startswith("turn")
                        else "straight"
                    ),
                ),
            })
        if not cached_segments:
            return {}
        segment_lengths_m = [float(seg["length"]) for seg in cached_segments]
        work_length_m = sum(
            seg["length"] for seg in cached_segments if seg["type"] == "work"
        )
        turn_length_m = sum(
            seg["length"] for seg in cached_segments if seg["type"].startswith("turn")
        )
        entry_exit_length_m = sum(
            seg["length"]
            for seg in cached_segments
            if seg["type"] in ("entry", "exit", "unload")
        )
        total_length_m = sum(segment_lengths_m)
        valid_at_save_time = bool(saved.get("auto_path_valid", False))
        validation = {
            "valid": False,
            "valid_at_save_time": valid_at_save_time,
            "current_validation_complete": False,
            "validation_source": "cache_snapshot",
            "total_length_m": float(saved.get("last_total_path_m") or total_length_m),
            "work_length_m": float(saved.get("last_work_path_m") or work_length_m),
            "turn_length_m": float(saved.get("last_turn_path_m") or turn_length_m),
            "entry_exit_length_m": float(
                saved.get("last_entry_exit_path_m") or entry_exit_length_m
            ),
        }
        optional_metrics = {
            "rolling_canopy_pct": "last_rolling_rate",
            "crop_mask_geometric_coverage_pct": "last_crop_mask_geometric_coverage_pct",
            "planned_target_geometric_coverage_pct": "last_planned_target_geometric_coverage_pct",
            "detected_mask_geometric_coverage_pct": "last_detected_mask_geometric_coverage_pct",
            "field_efficiency_pct": "last_field_efficiency",
        }
        for metric_name, saved_name in optional_metrics.items():
            if saved_name in saved and saved[saved_name] is not None:
                validation[metric_name] = float(saved[saved_name])
        path_result = {
            "full_path": [seg["points"] for seg in cached_segments],
            "segment_types": [seg["type"] for seg in cached_segments],
            "segment_geometry_modes": [
                seg.get("geometry_mode") for seg in cached_segments
            ],
            "segment_lengths_m": segment_lengths_m,
            "is_valid": False,
            "validation": validation,
            "description": saved.get("auto_path_desc", ""),
            "geo_points": saved.get("auto_path_geo", []),
            "provenance": saved.get("path_provenance", {}),
            "runtime": saved.get("path_runtime", {}),
        }
        restored = self._store_path_result(path_result, {"path": path_result})
        saved_artifact = saved.get("path_artifact") or {}
        saved_validation = saved.get("path_validation") or {}
        manual_saved = bool(
            isinstance(saved_artifact, dict) and saved_artifact.get("manual_route")
        ) or str(saved_validation.get("route_geometry_source", "")).startswith(
            "manual_control_edit"
        )
        saved_points = saved.get("path_points") or []
        saved_status = saved.get("path_status") or []
        if manual_saved and len(saved_points) >= 2:
            points = self._coerce_points(saved_points)
            edge_count = len(points) - 1
            if edge_count > 0:
                statuses = list(saved_status)[:edge_count]
                modes = self._route_edge_geometry_modes(
                    points,
                    statuses,
                    saved.get("path_geometry_modes") or [],
                )
                self.state.safe_update(
                    path_points=points,
                    path_status=statuses,
                    path_geometry_modes=modes,
                )
                self._sync_auto_path_from_editable_route()
        return restored

    def _store_path_result(self, path_result, pipeline_result=None):
        normalised = self._normalise_path_result(path_result)
        if self.geo.is_ready() and normalised.get("full_path"):
            from path_planner import global_path_to_geo
            normalised["geo_points"] = global_path_to_geo(
                normalised["full_path"],
                self.geo,
                normalised.get("segment_types", []),
            )
        elif not self.geo.is_ready():
            normalised["geo_points"] = []
        auto_segments = self._auto_segments_from_path(normalised)
        if auto_segments:
            self._clear_machine_route_preview()
        # Keep one compact control polyline for the route editor and the
        # machine-compatibility projection.  The dense canonical segments stay
        # in ``auto_path`` for scientific validation and footprint metrics.
        control_points, control_status = self._build_route_control_polyline(
            auto_segments
        )
        control_geometry_modes = [
            "straight" if int(status) == 1 else "turn"
            for status in control_status
        ]

        geo_points = normalised.get("geo_points", [])
        validation = normalised.get("validation", {}) or {}
        path_artifact = {
            key: self._manifest_jsonable(normalised[key])
            for key in (
                "turn_assessment",
                "planning_factors",
                "requested_strategy",
                "coordinate_space",
                "machine_readiness",
                "machine_executable",
                "path_candidates",
                "layout",
            )
            if key in normalised
        }
        state_updates = dict(
            auto_path=auto_segments,
            auto_path_segments=auto_segments,
            auto_path_geo=geo_points,
            auto_path_valid=bool(normalised.get("is_valid", validation.get("valid", False))),
            auto_path_desc=normalised.get("description", ""),
            auto_path_planned=bool(auto_segments),
            path_points=control_points,
            path_status=control_status,
            path_geometry_modes=control_geometry_modes,
            path_validation=dict(validation),
            path_provenance=dict(normalised.get("provenance") or {}),
            path_runtime=dict(normalised.get("runtime") or {}),
            path_artifact=path_artifact,
        )
        metric_state_fields = {
            "last_rolling_rate": "rolling_canopy_pct",
            "last_crop_mask_geometric_coverage_pct": "crop_mask_geometric_coverage_pct",
            "last_planned_target_geometric_coverage_pct": "planned_target_geometric_coverage_pct",
            "last_detected_mask_geometric_coverage_pct": "detected_mask_geometric_coverage_pct",
            "last_field_efficiency": "field_efficiency_pct",
            "last_total_path_m": "total_length_m",
            "last_work_path_m": "work_length_m",
            "last_turn_path_m": "turn_length_m",
            "last_entry_exit_path_m": "entry_exit_length_m",
        }
        for state_name, metric_name in metric_state_fields.items():
            if metric_name in validation and validation[metric_name] is not None:
                state_updates[state_name] = float(validation[metric_name])
        self.state.safe_update(**state_updates)

        # A newly committed canonical result supersedes any display-only
        # stale route retained during drag/replan.  An empty/failed result
        # leaves the stale preview in place so the user can still see what was
        # invalidated and why a retry is needed.
        if auto_segments:
            self._stale_path_preview = None

        result = dict(pipeline_result or self._pipeline_result or {})
        result["path"] = normalised
        self._pipeline_result = result
        return normalised


    TURN_STRATEGY_LABELS = {
        "auto": "自动推荐",
        "bow": "弓形调头",
        "semicircle": "半圆调头",
        "pear": "梨形调头",
        "fishtail": "鱼尾折返",
        "alpha": "紧凑 Alpha",
        "contour": "套圈收割",
    }

    def _turn_strategy_label(self, path_result=None):
        path_result = path_result or {}
        assessment = path_result.get("turn_assessment", {}) if isinstance(path_result, dict) else {}
        key = assessment.get("strategy") or path_result.get("requested_strategy") or getattr(self.state, 'turn_strategy', 'auto')
        return assessment.get("label") or self.TURN_STRATEGY_LABELS.get(str(key), str(key))

    def _log_turn_strategy_result(self, path_result):
        validation = (path_result or {}).get("validation", {}) or {}
        label = self._turn_strategy_label(path_result)
        total = validation.get("total_length_m")
        turn = validation.get("turn_length_m")
        coverage = validation.get(
            "crop_mask_geometric_coverage_pct",
            validation.get("planned_target_geometric_coverage_pct"),
        )
        rolling = validation.get("rolling_canopy_pct")
        parts = [f"[{label}]"]
        if total is not None: parts.append(f"总长 {self._fmt_metric(total)}")
        if turn is not None: parts.append(f"转弯 {self._fmt_metric(turn)}")
        if coverage is not None: parts.append(f"覆盖 {self._fmt_metric(coverage, '%')}")
        if rolling is not None: parts.append(f"碾压 {self._fmt_metric(rolling, '%')}")
        self.log_panel.info("调头方式: " + " / ".join(parts))

    @staticmethod
    def _fmt_metric(value, suffix="m"):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "--"
        return f"{value:.1f}{suffix}"

    def _path_quality_summary(self, path_result, bands_count=None, tracks_count=None, turns_count=None):
        """一句话结果：作业段 / 转弯 / 总长。其余指标进 _path_quality_details。"""
        path_result = self._normalise_path_result(path_result)
        tracks_count = len(path_result.get("tracks", [])) if tracks_count is None else tracks_count
        turns_count = len(path_result.get("formal_turns", path_result.get("turns", []))) if turns_count is None else turns_count
        validation = path_result.get("validation", {}) or {}
        total = validation.get("total_length_m", path_result.get("total_length_m"))
        if total is None:
            return f"路径已生成 · 作业段 {tracks_count} 段，转弯 {turns_count} 处。下一步：导出或模拟演示"
        return (
            f"路径已生成 · 作业段 {tracks_count} 段，转弯 {turns_count} 处，"
            f"总长 {self._fmt_metric(total)}。下一步：导出或模拟演示"
        )

    def _path_quality_details(self, path_result) -> str:
        """明细：调头方式、分段长度、覆盖率、效率、风险数（只进"详情"）。"""
        path_result = self._normalise_path_result(path_result)
        validation = path_result.get("validation", {}) or {}
        parts = [f"调头方式 {self._turn_strategy_label(path_result)}"]
        work = validation.get("work_length_m", path_result.get("work_length_m"))
        turn = validation.get("turn_length_m", path_result.get("turn_length_m"))
        coverage = validation.get(
            "crop_mask_geometric_coverage_pct",
            validation.get("planned_target_geometric_coverage_pct"),
        )
        efficiency = validation.get("field_efficiency_pct", validation.get("field_efficiency"))
        if work is not None:
            parts.append(f"作业长度 {self._fmt_metric(work)}")
        if turn is not None:
            parts.append(f"转弯长度 {self._fmt_metric(turn)}")
        if coverage is not None:
            parts.append(f"掩膜覆盖 {self._fmt_metric(coverage, '%')}")
        if efficiency is not None:
            parts.append(f"田间效率 {self._fmt_metric(efficiency, '%')}")
        issues = validation.get("issues") or []
        if issues:
            parts.append(f"风险 {len(issues)} 处")
        return " · ".join(parts)

    @Slot(str, object)
    def _on_step_result(self, step, data):
        if step == "segment_done":
            self.task_panel.set_task_status("segment", "done")
            self.task_panel.set_task_action("segment", "\u91cd\u8dd1")
            if isinstance(data, np.ndarray):
                self.state.safe_update(
                    mask_raw=data,
                    inference_original_mask=data,
                    inference_done=True,
                )
                self._show_mask_overlay(data)
            self._journal.info("作物识别完成")
            WorkflowUpdater.advance(self.state, "INFERENCE_DONE")
        elif step == "process_done":
            self.task_panel.set_task_status("process", "done")
            self.task_panel.set_task_action("process", "\u91cd\u505a")
            if isinstance(data, dict):
                processed_mask = data.get("processed_mask")
                self.state.safe_update(
                    mask_processed=True,
                    mask_result=data,
                    mask_provenance=dict(data.get("provenance") or {}),
                    centerline_catalog={},
                    selected_centerline_ids=[],
                    centerline_selection_report={},
                    work_band_catalog={},
                    selected_work_band_ids=[],
                    work_band_selection_report={},
                )
                self.task_panel.clear_centerline_selection()
                if isinstance(processed_mask, np.ndarray):
                    self._show_mask_overlay(processed_mask)
                self.log_panel.info("掩膜已更新；请在任务清单中点击’生成作业线’。")
            elif isinstance(data, np.ndarray):
                self._show_mask_overlay(data)
            self._journal.info("掩膜优化完成")

            self._sync_task_statuses()
            WorkflowUpdater.advance(self.state, "MASK_PROCESSED")
        elif step == "mask" and isinstance(data, np.ndarray):
            self._show_mask_overlay(data)

    @Slot(dict)
    def _on_pipeline_done(self, result):
        result = dict(result or {})
        has_new_mask = isinstance(result.get("mask"), np.ndarray)
        has_new_processed = isinstance(result.get("processed"), dict)
        if has_new_mask:
            self.state.safe_update(
                mask_raw=result["mask"],
                inference_original_mask=result["mask"],
                inference_done=True,
            )
        if has_new_processed:
            self.state.safe_update(
                mask_processed=True,
                mask_result=result["processed"],
                mask_provenance=dict(result["processed"].get("provenance") or {}),
            )
        if has_new_mask or has_new_processed:
            self.state.safe_update(
                centerline_catalog={},
                selected_centerline_ids=[],
                centerline_selection_report={},
                work_band_catalog={},
                selected_work_band_ids=[],
                work_band_selection_report={},
            )
            self.task_panel.clear_centerline_selection()
            # A combined pipeline result establishes a new mask basis; an old
            # drag/replan preview must not be rendered over that new analysis.
            self._stale_path_preview = None
        pr = self._store_path_result(result.get("path", {}), result)
        self._apply_centerline_layout(pr.get("layout") or {})
        if has_new_processed and not self._centerline_state_view()[0].get("items"):
            self.log_panel.info("识别结果已更新；请在任务清单中点击’生成作业线’。")
        self.top_toolbar.set_progress(0)
        self.progress_bar.hide()
        self.lbl_status.setText("就绪")
        self._sync_task_statuses()
        WorkflowUpdater.advance(self.state, "PATH_PLANNED")
        if self._tif_rgb is not None:
            self._service_points_visible = True
            if pr.get("full_path"):
                self._show_path(pr)
            elif isinstance(getattr(self, "_stale_path_preview", None), dict):
                self._show_path(self._stale_path_preview, stale_preview=True)
        tracks = len(pr.get("tracks", []))
        turns = len(pr.get("formal_turns", pr.get("turns", [])))
        bands = len(result.get("bands", []))
        self.log_panel.success(self._path_quality_summary(pr, bands, tracks, turns))
        self._log_turn_strategy_result(pr)
        self.route_info.update_route(pr, pr.get("validation", {}))
        self._journal.info(f"路径: {bands}条带 {tracks}作业线 {turns}转弯线")
        validation = pr.get("validation", {})
        if validation:
            self._show_validation_report(pr, validation)
            self._warn_path_validation_issues(pr)

    @Slot(str)
    def _on_pipeline_error(self, msg):
        self.top_toolbar.set_progress(0)
        self.progress_bar.hide()
        title, message = self._task_error_message(msg)
        self.lbl_status.setText(title)
        self.log_panel.error(f"{title}: {msg[:200]}")
        for tid in ("segment", "process", "plan"):
            w = self.task_panel._task_widgets.get(tid)
            if w and w._in_progress:
                self.task_panel.set_task_status(tid, "failed", text="失败")
        QMessageBox.critical(self, title, message)

    def _apply_mask_process_result(self, processed):
        mask = getattr(self.state, 'mask_raw', None)
        if processed is None:
            self.log_panel.error("掩膜优化失败")
            self.task_panel.set_task_status("process", "failed", text="失败")
            QMessageBox.warning(self, "掩膜优化失败", "掩膜优化没有生成有效结果。\n\n请检查 作物识别结果或重新圈选田块后重试。")
            return
        processed_mask = processed.get("processed_mask", mask)
        wide_bands = processed.get("wide_bands", [])
        processed = self._compact_mask_result_for_ui(processed)
        processed_mask = processed.get("processed_mask", processed_mask)
        self._save_processed_mask_async(processed_mask, processed)
        self.state.safe_update(
            mask_processed=True,
            mask_result=processed,
            mask_provenance=dict(processed.get("provenance") or {}),
            centerline_catalog={},
            selected_centerline_ids=[],
            centerline_selection_report={},
            work_band_catalog={},
            selected_work_band_ids=[],
            work_band_selection_report={},
        )
        self.task_panel.clear_centerline_selection()
        # A newly committed mask is a new visual/input basis.  Do not let a
        # transient route preview from an earlier drag/replan survive on top
        # of it; only a path-stage replan may retain that preview.
        self._stale_path_preview = None
        centers = [b for b in wide_bands if b.get("centerline")]

        # 统计识别结果并提供详细说明
        all_bands = processed.get("all_bands") or []
        narrow_count = len([b for b in all_bands if b.get("band_type") == "narrow"])

        # 详细友好提示
        from ui_messages import bands_summary, mask_done

        # 一句话讲结果，换算说明进"详情"（原来四行碎句全堆在交互栏）
        self.log_panel.success(mask_done(len(wide_bands), narrow_count))
        self.log_panel.detail(
            "幅宽 1.8 m = 左 1 行 + 宽行 + 4 行 + 宽行 + 右 1 行；"
            f"本块田 {bands_summary(len(wide_bands), narrow_count)}"
        )
        self.task_panel.set_task_status("process", "done")
        self.task_panel.set_task_action("process", "\u91cd\u505a")

        self._sync_task_statuses()
        WorkflowUpdater.advance(self.state, "MASK_PROCESSED")
        strip = getattr(self, "status_strip", None)
        if strip is not None:
            strip.set_step("第 4 步 · 掩膜优化")
            strip.set_phase("已完成", "ready")
            strip.set_hint("下一步：生成作业线")
            strip.set_metrics({"主体区": f"{len(wide_bands)} 条", "边缘区": f"{narrow_count} 条"})
        QTimer.singleShot(50, lambda: self._show_mask_overlay(processed_mask))

    def _compact_mask_result_for_ui(self, processed):
        """Remove dense per-band arrays before UI state/cache metadata saves."""
        if not isinstance(processed, dict):
            return processed
        compact = dict(processed)
        for key in ("wide_bands", "narrow_bands", "all_bands"):
            items = []
            for band in compact.get(key, []) or []:
                if isinstance(band, dict):
                    cleaned = {k: v for k, v in band.items() if k not in {"contour", "debug", "mask"}}
                    contour_value = band.get("contour")
                    if contour_value is not None:
                        try:
                            contour = np.asarray(
                                contour_value, dtype=np.float32
                            ).reshape(-1, 2)
                            if len(contour) >= 3:
                                closed = contour.reshape(-1, 1, 2)
                                perimeter = float(cv2.arcLength(closed, True))
                                epsilon = max(1.0, perimeter * 0.0015)
                                simplified = cv2.approxPolyDP(
                                    closed, epsilon, True
                                ).reshape(-1, 2)
                                if len(simplified) > 256:
                                    indexes = np.linspace(
                                        0, len(simplified) - 1, 256
                                    ).round().astype(np.int32)
                                    simplified = simplified[indexes]
                                if len(simplified) >= 3:
                                    cleaned["planning_contour"] = [
                                        [float(point[0]), float(point[1])]
                                        for point in simplified
                                    ]
                        except (TypeError, ValueError):
                            pass
                    items.append(cleaned)
                else:
                    items.append(band)
            compact[key] = items
        return compact

    def _save_processed_mask_async(self, processed_mask, mask_result=None):
        """Persist the large processed mask off the GUI thread.

        np.savez_compressed on a 15078×21021 uint8 mask can take several seconds
        and runs after the worker reports 92–95%, which previously froze the
        main window exactly when the algorithm had already completed.
        """
        tif_path = self._tif_path or getattr(self.state, 'tif_path', '')
        if not tif_path or not isinstance(processed_mask, np.ndarray):
            return
        ox = int(getattr(self.state, 'mask_offset_x', 0) or 0)
        oy = int(getattr(self.state, 'mask_offset_y', 0) or 0)
        semantic_layers = {
            key: value
            for key in (
                "regularized_body_mask", "body_residual_mask",
                "gap_repair_added_mask", "support_clip_removed_mask",
                "headland_corridor_mask", "headland_mask",
                "uncertain_residual_mask",
            )
            if isinstance((value := (mask_result or {}).get(key)), np.ndarray)
            and value.shape[:2] == processed_mask.shape[:2]
        }
        with self._processed_mask_save_lock:
            self._processed_mask_save_generation += 1
            generation = self._processed_mask_save_generation

        def save_job():
            try:
                from cache import save_mask
                def is_current_generation():
                    with self._processed_mask_save_lock:
                        return generation == self._processed_mask_save_generation
                saved = save_mask(
                    tif_path,
                    processed_mask,
                    ox,
                    oy,
                    suffix="_processed",
                    commit_callback=is_current_generation,
                    extra_arrays=semantic_layers,
                )
                if not saved:
                    self._log.info("processed mask cache save skipped: superseded by newer result")
            except Exception as e:
                self._log.info(f"processed mask cache save failed: {e}")
            finally:
                current = threading.current_thread()
                with self._processed_mask_save_lock:
                    self._processed_mask_save_threads = [
                        t for t in self._processed_mask_save_threads
                        if t is not current and t.is_alive()
                    ]

        thread = threading.Thread(target=save_job, name="align-save-processed-mask", daemon=True)
        with self._processed_mask_save_lock:
            self._processed_mask_save_threads.append(thread)
        thread.start()

    def _wait_processed_mask_saves(self, timeout_s=5.0):
        deadline = time.time() + max(0.0, float(timeout_s))
        while True:
            with self._processed_mask_save_lock:
                threads = [t for t in self._processed_mask_save_threads if t.is_alive()]
                self._processed_mask_save_threads = threads
            if not threads:
                return True
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            threads[0].join(min(0.2, remaining))

    def _on_mask_process(self):
        self._log_action("mask.request", "process")
        mask = getattr(self.state, 'mask_raw', None)
        # DECISION-020 diagnostics: log the exact array that reaches mask
        # processing so a mismatch with the inference artifact can be seen
        # in the journal without re-instrumenting the exe.
        try:
            if mask is not None:
                import numpy as _np
                self._journal.info(
                    'mask process input: '
                    f'nonzero={int(_np.count_nonzero(mask))} '
                    f'shape={list(mask.shape)} dtype={str(mask.dtype)} '
                    f'offset=({self.state.mask_offset_x},{self.state.mask_offset_y}) '
                    f'field_pts={len(self.state.field_boundary or [])}'
                )
            else:
                self._journal.info('mask process input: None')
        except Exception:
            pass
        if mask is None:
            self.log_panel.warn("请先完成 作物识别")
            return
        if (
            getattr(self.state, "inference_done", False) is True
            and getattr(self.state, "cache_model_matches_active_engine", True)
            is False
        ):
            self.log_panel.warn(
                "当前掩膜来自历史模型缓存，但该模型尚未加载；"
                "请先加载同一模型，或重新执行 作物识别后再处理掩膜。"
            )
            return
        if self._has_background_worker():
            self.log_panel.warn("任务正在运行：当前已有耗时任务正在执行，请等待完成后再处理掩膜。")
            return
        try:
            from pyside6_app.workers import require_metric_scale
            require_metric_scale(self.geo, self.state, mask.shape)
        except Exception as exc:
            QMessageBox.warning(self, "地理尺度不可用", str(exc))
            return
        self._save_system_undo("掩膜优化")
        self._invalidate_analysis_from(
            "mask",
            "已开始新的掩膜优化，本次结果提交前不保留旧处理结果和路径",
        )
        mask_config = self.cfg.section("mask_processing")
        mask_config["strength"] = self._mask_processing_strength
        self._active_mask_token = make_task_token(self.state, "mask")
        worker = MaskProcessWorker(
            mask,
            self.geo,
            self.state,
            mask_config,
            token=self._active_mask_token,
        )
        worker.finished.connect(self._on_mask_worker_done)
        self.state.mask_processing_running = True
        self.state.mask_processing_progress = 0.05
        self.top_toolbar.set_progress(5, "掩膜优化中...")
        self.progress_bar.show()
        self.progress_bar.setValue(5)
        self.task_panel.set_task_status("process", "in_progress", 5)
        self.lbl_status.setText("掩膜优化中...")
        if not self._start_background_worker(worker, "掩膜优化已开始，完成后请进入“路径规划”确认起点；终点将沿最终作业段自动延伸。"):
            self.state.mask_processing_running = False
            self.state.mask_processing_progress = 0.0
            self.progress_bar.hide()
            self.top_toolbar.set_progress(0)
            self._sync_task_statuses()

    @staticmethod
    def _next_snapshot_path(directory: str) -> str:
        """Return the next numbered PNG path within one selected folder."""
        highest = 0
        try:
            names = os.listdir(directory)
        except OSError:
            names = []
        for name in names:
            stem, extension = os.path.splitext(str(name))
            if extension.casefold() not in {".png", ".jpg", ".jpeg"}:
                continue
            if not stem.startswith("快照"):
                continue
            suffix = stem[len("快照"):]
            if suffix.isdecimal():
                highest = max(highest, int(suffix))
        return os.path.join(directory, f"快照{highest + 1}.png")

    def _on_save_snapshot(self):
        start_directory = self._snapshot_directory
        if not start_directory or not os.path.isdir(start_directory):
            start_directory = (
                os.path.dirname(os.path.abspath(self._tif_path))
                if self._tif_path
                else os.getcwd()
            )
        directory = QFileDialog.getExistingDirectory(
            self, "选择快照保存文件夹", start_directory
        )
        if not directory:
            return
        directory = os.path.abspath(directory)
        self._snapshot_directory = directory
        path = self._next_snapshot_path(directory)
        if self.image_view.viewer.grab().save(path):
            self.log_panel.success(f"已保存: {path}")
        else:
            self.log_panel.error(f"保存失败: {path}")

    def _on_toggle_mask_compare(self):
        """DECISION-004: compare the immutable source image with current state."""
        if self._tif_rgb is None:
            self.log_panel.error("请先加载原始影像")
            return
        # Comparing is a layer swap, not a navigation command.  Preserve the
        # exact image-space viewport so a magnified crop stays in place.
        view_state = self.image_view.viewer.capture_view_state()
        self._show_mask_before = not self._show_mask_before
        if self._show_mask_before:
            self._sync_display_layer_availability()
            self.image_view.set_image(self._tif_rgb)
            self.log_panel.info("对比：显示原始影像")
        else:
            self._restore_visual_from_state()
            self.log_panel.info("对比：显示当前操作结果")
        self.image_view.viewer.restore_view_state(view_state)

    def _clear_measurement_overlay(self):
        viewer = self.image_view.viewer
        viewer.remove_overlays(self._measure_overlay_items)
        self._measure_overlay_items = []

    def _measure_draw(self, draw_call, *args, **kwargs):
        item = draw_call(*args, **kwargs)
        if item is not None:
            self._measure_overlay_items.append(item)
        return item

    def _reset_measurement_geometry(self):
        self._clear_measurement_overlay()
        self._measure_points = []
        self._measure_complete = False

    def _on_measure_tool_changed(self, tool):
        self._reset_measurement_geometry()
        self._measure_tool = tool
        if tool:
            self._field_drawing = False
            self._sync_drawing_mode()
            tips = {
                "point": "点选模式：点击地图查看经纬度",
                "line": "测距模式：点击两个点，右键确认长度；左键开始下一条",
                "area": "测面积模式：点击多边形顶点，右键确认面积；左键开始下一个",
            }
            self.log_panel.info(tips.get(tool, "测量工具"))
        else:
            self._sync_drawing_mode()
            self.log_panel.info("已关闭测量工具")

    def _format_distance(self, meters):
        if meters >= 1000:
            return f"{meters / 1000:.3f} km"
        if meters >= 1:
            return f"{meters:.2f} m"
        return f"{meters * 100:.1f} cm"

    def _handle_measure_click(self, x, y, button):
        viewer = self.image_view.viewer
        tool = self._measure_tool
        if tool == "point" and button == 1:
            self._reset_measurement_geometry()
            self._measure_points = [(x, y)]
            self._measure_draw(
                viewer.draw_circle, x, y, radius=7, color=(255, 200, 0)
            )
            if self.geo.is_ready():
                lon, lat = self.geo.pixel_to_lonlat(x, y)
                text = f"lon:{lon:.10f}\nlat:{lat:.10f}"
            else:
                text = f"x:{float(x):.6f}\ny:{float(y):.6f}"
            self._measure_draw(
                viewer.draw_text,
                x + 10,
                y + 10,
                text,
                color=(255, 255, 255),
                size=11,
            )
            self.log_panel.info(text.replace("\n", "  "))
            self._measure_complete = True
            return True
        if tool == "line" and button == 1:
            if self._measure_complete or len(self._measure_points) >= 2:
                self._reset_measurement_geometry()
            self._measure_points.append((x, y))
            self._measure_draw(
                viewer.draw_circle, x, y, radius=6, color=(0, 220, 255)
            )
            if len(self._measure_points) == 2:
                p1, p2 = self._measure_points
                self._measure_draw(
                    viewer.draw_line,
                    p1[0], p1[1], p2[0], p2[1],
                    color=(0, 220, 255), width=3,
                )
            return True
        if tool == "line" and button == 2:
            if len(self._measure_points) != 2:
                self.log_panel.warn("测距需要先点击两个点")
                return True
            if self._measure_complete:
                return True
            p1, p2 = self._measure_points
            dist = (
                self.geo.pixel_distance_m(p1, p2)
                if self.geo.is_ready()
                else math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            )
            unit = (
                self._format_distance(dist)
                if self.geo.is_ready()
                else f"{dist:.1f} px"
            )
            self._measure_draw(
                viewer.draw_text,
                (p1[0] + p2[0]) / 2,
                (p1[1] + p2[1]) / 2,
                unit,
                color=(255, 255, 255),
                size=12,
            )
            self.log_panel.info(f"测距: {unit}")
            self._measure_complete = True
            return True
        if tool == "area":
            if button == 2 and len(self._measure_points) >= 3:
                self._finish_measure_area()
                return True
            if button == 2:
                self.log_panel.warn("测面积需要先点击至少三个顶点")
                return True
            if button != 1:
                return True
            if self._measure_complete:
                self._reset_measurement_geometry()
            if len(self._measure_points) >= 3:
                first = self._measure_points[0]
                if math.hypot(x - first[0], y - first[1]) <= 12:
                    self._finish_measure_area()
                    return True
            self._measure_points.append((x, y))
            self._measure_draw(
                viewer.draw_circle, x, y, radius=6, color=(255, 220, 0)
            )
            if len(self._measure_points) >= 2:
                p1, p2 = self._measure_points[-2], self._measure_points[-1]
                self._measure_draw(
                    viewer.draw_line,
                    p1[0], p1[1], p2[0], p2[1],
                    color=(255, 220, 0), width=3,
                )
            return True
        return False

    def _finish_measure_area(self):
        pts = list(self._measure_points)
        if len(pts) < 3 or self._measure_complete:
            return
        viewer = self.image_view.viewer
        self._measure_draw(
            viewer.draw_line,
            pts[-1][0], pts[-1][1], pts[0][0], pts[0][1],
            color=(255, 220, 0), width=3,
        )
        area_info = self._estimate_field_area(pts)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        self._measure_draw(
            viewer.draw_text,
            cx,
            cy,
            f"面积: {area_info}",
            color=(255, 255, 255),
            size=12,
        )
        self.log_panel.info(f"面积测量: {area_info}")
        self._measure_complete = True
    def _mask_for_overlay(self, mask):
        if self._tif_rgb is None or mask is None:
            return None
        full_h, full_w = self._tif_rgb.shape[:2]
        if mask.shape[:2] == (full_h, full_w):
            return mask
        x0 = int(getattr(self.state, 'mask_offset_x', 0) or 0)
        y0 = int(getattr(self.state, 'mask_offset_y', 0) or 0)
        out = np.zeros((full_h, full_w), dtype=np.uint8)
        h, w = mask.shape[:2]
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(full_w, x0 + w)
        dst_y1 = min(full_h, y0 + h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return out
        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y0 + (dst_y1 - dst_y0),
                                                 src_x0:src_x0 + (dst_x1 - dst_x0)]
        return out

    def _normalise_mask_for_overlay(self, mask):
        """Return a mask and offset in the viewer's display coordinate space."""
        if mask is None:
            return None, 0, 0
        mask = np.asarray(mask)
        if mask.ndim < 2 or mask.size == 0:
            return mask, 0, 0

        display_h, display_w = self._tif_rgb.shape[:2]
        source_h = int(getattr(self.state, "source_img_h", 0) or 0)
        source_w = int(getattr(self.state, "source_img_w", 0) or 0)
        mask_h, mask_w = mask.shape[:2]
        offset_x = int(getattr(self.state, "mask_offset_x", 0) or 0)
        offset_y = int(getattr(self.state, "mask_offset_y", 0) or 0)
        scale_x = float(source_w) / float(display_w) if source_w > display_w else 1.0
        scale_y = float(source_h) / float(display_h) if source_h > display_h else 1.0
        source_space = bool(
            scale_x > 1.0 and scale_y > 1.0 and
            (mask_h > display_h or mask_w > display_w or
             offset_x + mask_w > display_w or offset_y + mask_h > display_h)
        )
        if source_space:
            import cv2
            mask = cv2.resize(
                mask,
                (max(1, int(round(mask_w / scale_x))), max(1, int(round(mask_h / scale_y)))),
                interpolation=cv2.INTER_NEAREST,
            )
            offset_x = int(round(offset_x / scale_x))
            offset_y = int(round(offset_y / scale_y))
        return mask, offset_x, offset_y

    def _overlay_mask_in_place(self, ov, mask):
        """Overlay full-size or cropped masks without allocating full-size temp images."""
        if ov is None or mask is None:
            return ov
        full_h, full_w = ov.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        if (mask_h, mask_w) == (full_h, full_w):
            dst_x0 = dst_y0 = src_x0 = src_y0 = 0
            dst_x1, dst_y1 = full_w, full_h
        else:
            x0 = int(getattr(self.state, 'mask_offset_x', 0) or 0)
            y0 = int(getattr(self.state, 'mask_offset_y', 0) or 0)
            dst_x0 = max(0, x0)
            dst_y0 = max(0, y0)
            dst_x1 = min(full_w, x0 + mask_w)
            dst_y1 = min(full_h, y0 + mask_h)
            if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
                return ov
            src_x0 = dst_x0 - x0
            src_y0 = dst_y0 - y0
        clipped = mask[src_y0:src_y0 + (dst_y1 - dst_y0),
                       src_x0:src_x0 + (dst_x1 - dst_x0)]
        if clipped.size == 0:
            return ov
        if cv2.countNonZero(clipped.astype(np.uint8, copy=False)) == 0:
            return ov
        roi = ov[dst_y0:dst_y1, dst_x0:dst_x1]
        chunk_rows = 256
        for y in range(0, clipped.shape[0], chunk_rows):
            m = clipped[y:y + chunk_rows]
            nz = m > 0
            if not np.any(nz):
                continue
            r = roi[y:y + chunk_rows]
            b = r[:, :, 0]
            g = r[:, :, 1]
            rr = r[:, :, 2]
            b[nz] = (b[nz].astype(np.float32) * 0.6).astype(np.uint8)
            g[nz] = (g[nz].astype(np.float32) * 0.6 + m[nz].astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8)
            rr[nz] = (rr[nz].astype(np.float32) * 0.6).astype(np.uint8)
        return ov

    #: 只有这些图层是"路径/作业线的派生物"：掩膜失效时必须一起消失，
    #: 而田块边界、识别掩膜、参考线属于证据层，只要各自的状态还在就必须留着。
    DERIVED_PATH_LAYERS = ("centerlines", "route", "route_edit")

    def _clear_derived_path_layers(self):
        """Drop only path-derived overlays and keep the evidence layers."""
        viewer = self.image_view.viewer
        for name in self.DERIVED_PATH_LAYERS:
            viewer.clear_overlay_layer(name)

    def canvas_state_mismatches(self):
        """Return human-readable disagreements between domain state and canvas.

        左清单与右画布必须由同一份状态驱动（用户要求"左右一起同步"）。这个
        检查把"状态里有什么、画布上就该有什么"（Rule A）与"状态里没有的，
        画布上也不许有"（Rule B）写成可断言的清单，既是回归测试的判据，
        也是状态改动后的自检依据。
        """
        problems = []
        if self._tif_rgb is None:
            return problems
        viewer = self.image_view.viewer
        state = self.state
        if getattr(viewer, "_pixmap", None) is None:
            problems.append("影像已导入但画布没有底图")

        raw = getattr(state, "mask_raw", None)
        has_raw = isinstance(raw, np.ndarray) and raw.size > 0
        inference_done = bool(getattr(state, "inference_done", False)) and has_raw
        mask_processed = bool(getattr(state, "mask_processed", False))
        mask_result = getattr(state, "mask_result", None)
        has_processed = mask_processed and isinstance(mask_result, dict)

        # Rule A：状态里有，画布上就必须有。
        if len(getattr(state, "field_boundary", []) or []) >= 3:
            if not viewer.has_overlay_layer("field_boundary"):
                problems.append("田块边界已存在但画布没有边界图层")
        if has_raw and np.count_nonzero(np.asarray(raw) > 2):
            # draw_mask_overlay 把像素值当强度（alpha = v*115/255），0/1 二值掩膜
            # 渲染出来是全透明，那是另一条问题，不算"应该有图层"。
            if not (
                viewer.has_overlay_layer("raw_mask")
                or viewer.has_overlay_layer("processed_mask")
            ):
                problems.append("作物识别结果已存在但画布没有掩膜图层")

        # Rule B：状态里没有，画布上就不许留。
        if not inference_done:
            for name in (
                "raw_mask",
                "processed_mask",
                "headland_mask",
                "uncertain_residual",
                "centerlines",
            ):
                if viewer.has_overlay_layer(name):
                    problems.append(f"作物识别结果已失效，画布仍有 {name} 图层")
        elif not has_processed:
            for name in ("processed_mask", "headland_mask", "uncertain_residual"):
                if viewer.has_overlay_layer(name):
                    problems.append(f"掩膜尚未优化，画布却有 {name} 图层")
        if not mask_processed and viewer.has_overlay_layer("centerlines"):
            problems.append("掩膜尚未优化，画布却有 centerlines 图层")
        if not getattr(state, "auto_path_planned", False):
            stale_preview = getattr(self, "_stale_path_preview", None)
            stale_ok = bool(
                isinstance(stale_preview, dict) and stale_preview.get("full_path")
            )
            if not stale_ok and viewer.has_overlay_layer("route"):
                problems.append("路径已失效但画布仍画着路线")
        return problems

    def _repair_canvas_if_state_disagrees(self, *, where=""):
        """画布与状态打架时，以状态为准重绘一次（视图缩放/平移不动）。

        「左边清单和右边画布必须一起同步」是用户的硬要求。任何改动状态的
        流程只要留下"状态说没有、画布还在画"或"状态说有、画布空着"，都会被
        这里发现并按状态重绘；重绘不改变用户当前看的位置。
        """
        problems = self.canvas_state_mismatches()
        if not problems:
            return False
        viewer = self.image_view.viewer
        view_state = (
            viewer.capture_view_state()
            if getattr(viewer, "_pixmap", None) is not None
            else None
        )
        self._restore_visual_from_state()
        if view_state is not None and getattr(viewer, "_pixmap", None) is not None:
            try:
                viewer.restore_view_state(view_state)
            except Exception:
                pass
        remaining = self.canvas_state_mismatches()
        self._log.info(
            "画布按状态重绘（%s）：修复 %s；剩余 %s"
            % (where or "state change", "、".join(problems), "、".join(remaining) or "无")
        )
        return True

    def _clear_display_overlay_layers(self, *, include_route=True):
        viewer = self.image_view.viewer
        names = [
            "raw_mask",
            "processed_mask",
            "headland_mask",
            "uncertain_residual",
            "centerlines",
            "reference_lines",
            "field_boundary",
            "forbidden_regions",
            "service_points",
        ]
        if include_route:
            names.append("route")
        for name in names:
            viewer.clear_overlay_layer(name)

    def _draw_mask_overlay_layer(self, viewer, mask, x, y, *, color, layer):
        if not isinstance(mask, np.ndarray):
            return False
        viewer.begin_overlay_layer(layer)
        try:
            viewer.draw_mask_overlay(mask, x, y, color=color)
        finally:
            viewer.end_overlay_layer()
        return True

    def _draw_raw_mask_layer(self, viewer):
        raw = getattr(self.state, "mask_raw", None)
        if not isinstance(raw, np.ndarray):
            return False
        raw, raw_x, raw_y = self._normalise_mask_for_overlay(raw)
        raw = self._display_mask_clipped_to_field(raw, raw_x, raw_y)
        return self._draw_mask_overlay_layer(
            viewer,
            raw,
            raw_x,
            raw_y,
            color=(100, 210, 255),
            layer="raw_mask",
        )

    def _draw_processed_mask_layers(self, viewer, mask=None):
        """Draw processed evidence into independently switchable layers."""
        mask_result = getattr(self.state, "mask_result", None)
        mask_result = mask_result if isinstance(mask_result, dict) else {}
        if mask is None:
            mask = mask_result.get("processed_mask")
        if mask is None:
            # Backward-compatible direct callers may ask for the current
            # evidence before a processed result exists.  The normal static
            # renderer draws ``mask_raw`` in its own raw layer first.
            mask = getattr(self.state, "mask_raw", None)
        if not isinstance(mask, np.ndarray):
            return False
        mask, mask_x, mask_y = self._normalise_mask_for_overlay(mask)
        mask = self._display_mask_clipped_to_field(mask, mask_x, mask_y)

        drawn = False
        headland = mask_result.get("headland_mask")
        if isinstance(headland, np.ndarray):
            headland, headland_x, headland_y = self._normalise_mask_for_overlay(
                headland
            )
            if headland.shape[:2] == mask.shape[:2]:
                headland = self._display_mask_clipped_to_field(
                    headland, headland_x, headland_y
                )
                drawn = self._draw_mask_overlay_layer(
                    viewer,
                    headland,
                    headland_x,
                    headland_y,
                    color=(210, 230, 120),
                    layer="headland_mask",
                ) or drawn

        uncertain = mask_result.get("uncertain_residual_mask")
        if isinstance(uncertain, np.ndarray):
            uncertain, uncertain_x, uncertain_y = self._normalise_mask_for_overlay(
                uncertain
            )
            if uncertain.shape[:2] == mask.shape[:2]:
                uncertain = self._display_mask_clipped_to_field(
                    uncertain, uncertain_x, uncertain_y
                )
                drawn = self._draw_mask_overlay_layer(
                    viewer,
                    uncertain,
                    uncertain_x,
                    uncertain_y,
                    color=(170, 170, 190),
                    layer="uncertain_residual",
                ) or drawn

        drawn = self._draw_mask_overlay_layer(
            viewer,
            mask,
            mask_x,
            mask_y,
            color=(120, 255, 140),
            layer="processed_mask",
        ) or drawn
        return drawn

    def _draw_static_display_layers(self, viewer, mask=None):
        """Rebuild persistent display layers while preserving interactions."""
        self._clear_display_overlay_layers(include_route=True)
        raw = getattr(self.state, "mask_raw", None)
        if isinstance(raw, np.ndarray):
            self._draw_raw_mask_layer(viewer)

        mask_result = getattr(self.state, "mask_result", None)
        mask_result = mask_result if isinstance(mask_result, dict) else {}
        processed = mask_result.get("processed_mask")
        if not isinstance(processed, np.ndarray):
            # ``mask`` is a processed argument for the normal caller.  Do not
            # duplicate a raw-only state into both visible mask rows.
            if isinstance(mask, np.ndarray) and (raw is None or mask is not raw):
                processed = mask
        if isinstance(processed, np.ndarray):
            self._draw_processed_mask_layers(viewer, processed)

        self._draw_centerline_overlay(viewer)
        self._draw_reference_lines_overlay(viewer)
        self._draw_field_boundary()
        if len(getattr(self.state, "field_boundary", []) or []) < 3:
            self._draw_forbidden_regions(clear_first=False)
        self._draw_service_points_overlay(clear_first=False)

    def _show_mask_overlay(self, mask):
        if self._tif_rgb is None or mask is None:
            return
        self.image_view.set_route_display_visible(False)
        self._show_mask_before = False
        self._sync_display_layer_availability()
        viewer = self.image_view.viewer
        if getattr(viewer, "_pixmap", None) is None:
            self.image_view.set_image(self._tif_rgb)
        else:
            viewer.clear_overlays()
        self._draw_static_display_layers(viewer, mask=mask)

    def _refresh_centerline_selection_overlay(self):
        """Refresh only selected centerlines after a checkbox edit."""
        viewer = self.image_view.viewer
        self._draw_centerline_overlay(viewer)

    def _on_view_scale_changed(self, _scale=None):
        """缩放变化后重排作业线标注。

        标注是按当前缩放的像素间距做碰撞过滤的：放大后间距变大，原本被
        过滤掉的编号应该重新出现，所以缩放（含滚轮、按钮、适应窗口）后
        防抖重画一次。
        """
        timer = getattr(self, "_centerline_label_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._redraw_centerline_labels)
            self._centerline_label_timer = timer
        timer.start(220)

    def _redraw_centerline_labels(self):
        catalog = getattr(self.state, "centerline_catalog", None) or getattr(
            self.state, "work_band_catalog", {}
        ) or {}
        if not catalog.get("items"):
            return
        viewer = self.image_view.viewer
        if getattr(viewer, "_pixmap", None) is None:
            return
        if hasattr(viewer, "overlay_layer_visible") and not viewer.overlay_layer_visible(
            "centerlines"
        ):
            return
        self._refresh_centerline_selection_overlay()

    def _display_mask_clipped_to_field(self, mask, offset_x, offset_y):
        """Return a display-only field clip while preserving analysis arrays."""
        if not isinstance(mask, np.ndarray):
            return mask
        boundary = getattr(self.state, "field_boundary", []) or []
        if len(boundary) < 3:
            return mask
        from row_geometry import clip_mask_to_polygon_stripes

        polygon = np.asarray(boundary, dtype=np.float64) - np.asarray(
            (offset_x, offset_y), dtype=np.float64
        )
        return clip_mask_to_polygon_stripes(mask.copy(), polygon)

    def _draw_mask_evidence_layers(self, viewer, mask=None):
        """Compatibility wrapper for the split processed evidence layers."""
        return self._draw_processed_mask_layers(viewer, mask)

    def _centerline_display_points(self, item):
        points = np.asarray(
            item.get("centerline") or item.get("work_line") or [],
            dtype=np.float64,
        )
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            return []
        # DECISION-004/024: the catalog is built from ``processed_mask``.
        # ``mask_raw`` and every processed layer are already normalised to
        # the downsampled display grid by inference, while the stored offset
        # is also a display-grid origin.  Do not divide the local points and
        # offset by the source/display scale here: doing so detaches the
        # centerlines from the orthomosaic and leaves them on the black scene
        # background (the old "mini-map" symptom).
        offset_x = float(getattr(self.state, "mask_offset_x", 0) or 0)
        offset_y = float(getattr(self.state, "mask_offset_y", 0) or 0)
        points[:, 0] += offset_x
        points[:, 1] += offset_y
        return [(float(x), float(y)) for x, y in points]

    @staticmethod
    def _point_along_polyline(points, fraction):
        """Return the point at *fraction* of the polyline's arc length.

        按点序号取整会导致"标签落点"跟实际几何无关（简化后的作业线只有 2–3
        个点，0.5 会落在中点、0.15 却仍在端点），标签就挤在同一处。
        """
        fraction = max(0.0, min(1.0, float(fraction)))
        if len(points) < 2:
            return points[0]
        lengths = []
        total = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            seg = math.hypot(dx, dy)
            lengths.append(seg)
            total += seg
        if total <= 0:
            return points[0]
        target = fraction * total
        acc = 0.0
        for i, seg in enumerate(lengths):
            if seg <= 0:
                continue
            if acc + seg >= target:
                ratio = (target - acc) / seg
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                return (x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio)
            acc += seg
        return points[-1]

    @staticmethod
    def _overlay_text_polygon(viewer, text_item):
        """Text item 在视口里的四个角（含旋转）。

        文字带 ItemIgnoresTransformations（按 1:1 绘制），所以尺寸要取
        boundingRect 并按自身旋转角旋转，不能再乘视图缩放。
        """
        if text_item is None:
            return None
        try:
            pos = viewer.mapFromScene(text_item.pos())
            rect = text_item.boundingRect()
            angle = math.radians(float(text_item.rotation() or 0.0))
        except Exception:
            return None
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = []
        for cx, cy in ((rect.left(), rect.top()), (rect.right(), rect.top()),
                       (rect.right(), rect.bottom()), (rect.left(), rect.bottom())):
            corners.append((
                pos.x() + cx * cos_a - cy * sin_a,
                pos.y() + cx * sin_a + cy * cos_a,
            ))
        return corners

    @staticmethod
    def _inflate_polygon(corners, margin):
        """旋转矩形的四角沿自身两轴外扩 margin 像素。

        只判"不重叠"不够：两个编号贴在一起看着就是一团。留出间隙再判，
        密集时自然会少画几个，放大后自动补齐。
        """
        if not corners or margin <= 0:
            return corners
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
        ux, uy = x1 - x0, y1 - y0
        vx, vy = x3 - x0, y3 - y0
        u_len = math.hypot(ux, uy) or 1.0
        v_len = math.hypot(vx, vy) or 1.0
        ux, uy = ux / u_len * margin, uy / u_len * margin
        vx, vy = vx / v_len * margin, vy / v_len * margin
        return [
            (x0 - ux - vx, y0 - uy - vy),
            (x1 + ux - vx, y1 + uy - vy),
            (x2 + ux + vx, y2 + uy + vy),
            (x3 - ux + vx, y3 - uy + vy),
        ]

    def _aligned_label_anchors(self, items, selected, fraction=0.5):
        """每条已选作业线一个标签落点，全部落在同一条横截线上。

        取统一的主方向投影位置（默认 50%，即各线中点附近），按各自弧长取会
        因线长不同而高低错落（"一会上一会下"）。
        """
        lines = []
        for item in items:
            cid = str(item.get("id"))
            if cid not in selected:
                continue
            points = self._centerline_display_points(item)
            if len(points) >= 2:
                lines.append((cid, points))
        if not lines:
            return {}
        longest = max(
            lines,
            key=lambda row: math.hypot(
                row[1][-1][0] - row[1][0][0], row[1][-1][1] - row[1][0][1]
            ),
        )[1]
        dx = longest[-1][0] - longest[0][0]
        dy = longest[-1][1] - longest[0][1]
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        projections = [
            [point[0] * ux + point[1] * uy for point in points]
            for _cid, points in lines
        ]
        flat = [value for row in projections for value in row]
        start = min(flat)
        span = max(flat) - start
        # 取所有线都覆盖到的投影区间（交集），保证每条线都在同一位置落点；
        # 用全局 min/max 会让短线的 5% 落在自身端点附近，标点就散开了。
        low = max(min(row) for row in projections)
        high = min(max(row) for row in projections)
        if high <= low:
            low, high = start, start + span
        target = low + (high - low) * max(0.0, min(1.0, float(fraction)))
        anchors = {}
        for (cid, points), projs in zip(lines, projections):
            if projs[0] > projs[-1]:
                points = list(reversed(points))
                projs = list(reversed(projs))
            anchor = points[0]
            for i in range(len(projs) - 1):
                a, b = projs[i], projs[i + 1]
                if b > a and a <= target <= b:
                    ratio = (target - a) / (b - a)
                    anchor = (
                        points[i][0] + (points[i + 1][0] - points[i][0]) * ratio,
                        points[i][1] + (points[i + 1][1] - points[i][1]) * ratio,
                    )
                    break
                if target > b:
                    anchor = points[i + 1]
                elif target < a:
                    anchor = points[i]
                    break
            anchors[cid] = anchor
        return anchors

    @staticmethod
    def _polygons_overlap(poly_a, poly_b):
        """分离轴判定（SAT）：两个凸四边形是否真的相交。

        旋转后的标注用外接矩形判重叠会误杀：两条平行标注沿法向错开 16 px、
        各自只有 14 px 厚，视觉上并不相交，但轴对齐外接框会大面积重叠。
        """
        for poly in (poly_a, poly_b):
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                axis_x, axis_y = -(y2 - y1), (x2 - x1)
                norm = math.hypot(axis_x, axis_y) or 1.0
                ux, uy = axis_x / norm, axis_y / norm
                proj_a = [px * ux + py * uy for px, py in poly_a]
                proj_b = [px * ux + py * uy for px, py in poly_b]
                if max(proj_a) <= min(proj_b) or max(proj_b) <= min(proj_a):
                    return False
        return True

    @staticmethod
    def _centerline_label_size(viewer, anchors):
        """按当前缩放下相邻编号的间距选字号。

        间距只有十几像素时用 9 pt 会互相压字（旋转后字高就是法向厚度），
        这里按间距退到 8/7 pt，保证整排编号都放得下。
        """
        points = [(float(p[0]), float(p[1])) for p in anchors.values()]
        if len(points) < 2:
            return 9
        nearest = []
        for i, (x0, y0) in enumerate(points):
            best = None
            for j, (x1, y1) in enumerate(points):
                if i == j:
                    continue
                distance = math.hypot(x1 - x0, y1 - y0)
                if best is None or distance < best:
                    best = distance
            if best:
                nearest.append(best)
        nearest.sort()
        median = nearest[len(nearest) // 2]
        try:
            scale = float(viewer.transform().m11()) or 1.0
        except Exception:
            scale = 1.0
        pitch_px = median * scale
        if pitch_px >= 19.0:
            return 9
        if pitch_px >= 14.5:
            return 8
        return 7

    def _operator_record_for_item(self, item, payload=None):
        """按精确身份找该作业线的人工标定记录（只认 centerline/legacy id）。

        不允许按 source_band_ids 兜底：等距区一条带可能拆成多股，按带匹配会
        把同一段标定画到邻居线上。
        """
        data = payload if isinstance(payload, dict) else self._operator_paths_payload()
        item_id = str(item.get("id") or "")
        legacy_id = str(item.get("legacy_id") or "")
        for record in data.get("lines") or []:
            if not isinstance(record, dict):
                continue
            if item_id and str(record.get("centerline_id") or "") == item_id:
                return record
        if legacy_id:
            for record in data.get("lines") or []:
                if isinstance(record, dict) and str(record.get("legacy_id") or "") == legacy_id:
                    return record
        return None

    def _draw_centerline_overlay(self, viewer):
        if hasattr(viewer, "clear_overlay_layer"):
            viewer.clear_overlay_layer("centerlines")
        catalog = getattr(self.state, "centerline_catalog", None) or getattr(
            self.state, "work_band_catalog", {}
        ) or {}
        items = list(catalog.get("items") or [])
        if not items:
            return
        selected = set(
            getattr(self.state, "selected_centerline_ids", None)
            or getattr(self.state, "selected_work_band_ids", [])
            or []
        )
        if hasattr(viewer, "begin_overlay_layer"):
            viewer.begin_overlay_layer("centerlines")
        label_rects = []
        anchors = self._aligned_label_anchors(items, selected)
        label_size = self._centerline_label_size(viewer, anchors)
        operator_payload = self._operator_paths_payload()
        try:
            for item in items:
                points = self._centerline_display_points(item)
                # 标定过的线在画布上直接画人线（与规划同源），否则"重启后画布
                # 还是算法线、导出的却是人线"，状态与画布对不上。
                operator_points = None
                record = self._operator_record_for_item(item, operator_payload)
                if record is not None:
                    local = operator_paths.as_points(record.get("points") or [])
                    if local.size >= 2:
                        operator_points = self._calibration_local_to_display(local)
                        points = operator_points
                if len(points) < 2:
                    continue
                is_selected = str(item.get("id")) in selected
                if not is_selected:
                    continue
                # 线在编号处要断开一小段，所以先建编号、量长度，再画线（见下）。
                # 线段要有两个端点：只画线时"从哪到哪"看不出来，两条相邻线
                # 的端头也容易连成一片。端点用固定屏幕半径的圆点，缩放不变形。
                for end_x, end_y in (points[0], points[-1]):
                    viewer.draw_circle(end_x, end_y, radius=4, color=(120, 255, 150))
                # 编号写全称"中心线 N"（用户要求：只写数字信息量不够）；文字带
                # 描边、线全程连续，被字压住的只有十来个字宽，不影响判断整条线。
                label = str(item.get("label") or item.get("id") or "中心线")
                # 14 条线在同一端头排成一行，横排标签必然互相压字：
                #   1) 所有编号取同一个相对位置（离端头约一成），排成整齐的一列；
                #   2) 标注旋转到与作业线同向，密集时按列并排而不是横向叠字；
                #   3) 最后做像素级碰撞检查，仍重叠的标签不画。
                angle = math.degrees(
                    math.atan2(
                        points[-1][1] - points[0][1],
                        points[-1][0] - points[0][0],
                    )
                )
                while angle > 90.0:
                    angle -= 180.0
                while angle < -90.0:
                    angle += 180.0
                # 编号放在各条作业线的中点附近：一条线一个号，落在整排中部比挤在
                # 端头更好认，也不跟端头的地头/起点标记抢位置。
                anchor = anchors.get(str(item.get("id"))) or self._point_along_polyline(points, 0.5)
                # 编号压在线上：文字带描边、无底色；线全程连续（用户要求：
                # 不许断线，断线会影响"这条线到底从哪到哪"的判断）。
                draw_label = getattr(viewer, "draw_label", None)
                if draw_label is not None:
                    text_item = draw_label(
                        anchor[0],
                        anchor[1],
                        label,
                        size=label_size,
                        rotation=angle,
                        center=True,
                    )
                else:
                    text_item = viewer.draw_text(
                        anchor[0],
                        anchor[1],
                        label,
                        color=(245, 247, 250),
                        size=label_size,
                        rotation=angle,
                        center=True,
                    )
                polygon = self._overlay_text_polygon(viewer, text_item)
                if polygon is not None:
                    polygon = self._inflate_polygon(polygon, 2.0)
                if polygon is not None and any(
                    self._polygons_overlap(polygon, other) for other in label_rects
                ):
                    if hasattr(viewer, "remove_overlays"):
                        viewer.remove_overlays([text_item])
                    text_item, polygon = None, None
                viewer.draw_polyline(
                    points,
                    color=OPERATOR_LINE_COLOR_BGR if operator_points else (40, 220, 80),
                    width=5,
                )
                if polygon is not None:
                    label_rects.append(polygon)
        finally:
            if hasattr(viewer, "end_overlay_layer"):
                viewer.end_overlay_layer()

    def _show_path(self, pr, stale_preview=False, route_projection=False):
        if self._tif_rgb is None:
            return
        self._show_mask_before = False
        pr = self._normalise_path_result(pr)
        self._route_display_mode = (
            "machine_file" if route_projection else "continuous"
        )
        self.image_view.set_route_display_mode(self._route_display_mode)
        self.image_view.set_route_display_visible(True)
        if route_projection and isinstance(self._machine_route_preview, dict):
            self.image_view.set_route_display_info(
                self._machine_route_preview["info"],
                tooltip=self._machine_route_preview["tooltip"],
            )
        else:
            route_info, _degenerate_count = self._canonical_route_display_info(pr)
            self.image_view.set_route_display_info(
                route_info,
                tooltip=(
                    "连续规划轨迹用于模拟和理论评价，不是实测 GNSS 轨迹。"
                    "未形成曲线的转弯先按端点直连显示，可进入路线编辑人工调整；"
                    "红色段表示路径超出已确认田块边界；重新验证前不得作为机器执行路径。"
                ),
            )
        viewer = self.image_view.viewer
        if getattr(viewer, "_pixmap", None) is None:
            self.image_view.set_image(self._tif_rgb)

        self._sync_display_layer_availability()
        self._draw_static_display_layers(viewer)
        boundary = getattr(self.state, 'field_boundary', [])
        if hasattr(viewer, "clear_overlay_layer"):
            viewer.clear_overlay_layer("route")
        route_style = Qt.DashLine if stale_preview else Qt.SolidLine
        track_color = (150, 160, 170) if stale_preview else (0, 255, 70)
        turn_color = (190, 150, 90) if stale_preview else (70, 120, 255)
        outside_field_color = (40, 40, 255)
        display_boundary = boundary if len(boundary or []) >= 3 else None
        if hasattr(viewer, "begin_overlay_layer"):
            viewer.begin_overlay_layer("route")
        try:
            for t in pr.get("tracks", []):
                pts = self._coerce_points(t.get("points", []))
                if len(pts) > 1:
                    viewer.draw_polyline(
                        pts, color=track_color, width=4, style=route_style
                    )
            for t in pr.get("turns", []):
                pts = self._coerce_points(t.get("points", []))
                if len(pts) > 1:
                    # DECISION-010/012: the planner keeps the two-point fallback
                    # as an explicitly unverified direct connector.  Render that
                    # geometry as a solid, editable connection instead of a red
                    # dashed pseudo-error; its audit flag and export gate remain
                    # unchanged until the user adjusts and revalidates the route.
                    for piece, inside_field in self._split_polyline_by_field_boundary(
                        pts, display_boundary
                    ):
                        viewer.draw_polyline(
                            piece,
                            color=turn_color if inside_field else outside_field_color,
                            width=3,
                            style=route_style,
                        )
            self._draw_route_finish_overlay(pr, stale_preview=stale_preview)
            if stale_preview:
                label_point = None
                for item in pr.get("tracks", []) + pr.get("turns", []):
                    points = self._coerce_points(item.get("points", []))
                    if points:
                        label_point = points[0]
                        break
                if label_point is None:
                    full_path = pr.get("full_path", []) or []
                    if full_path:
                        points = self._coerce_points(full_path[0])
                        if points:
                            label_point = points[0]
                if label_point is not None:
                    viewer.draw_text(
                        float(label_point[0]) + 12.0,
                        float(label_point[1]) + 12.0,
                        "待重新规划预览（不可导出）",
                        color=(255, 190, 70),
                        size=11,
                    )
        finally:
            if hasattr(viewer, "end_overlay_layer"):
                viewer.end_overlay_layer()

    def _show_machine_route_preview(self, preview):
        if not isinstance(preview, dict):
            return
        path_result = self._normalise_path_result(
            preview.get("path_result") or {}
        )
        self._show_path(path_result, route_projection=True)
        # Keep the route-result list bound to the geometry currently shown.
        # Otherwise a two-point TXT preview can sit beside stale work/turn rows
        # from the pre-edit continuous planner, which is not WYSIWYG.
        self.route_info.update_route(path_result, {})
        viewer = self.image_view.viewer
        if hasattr(viewer, "begin_overlay_layer"):
            viewer.begin_overlay_layer("route")
        try:
            for chunk in preview.get("file_chunks") or []:
                records = list(chunk.get("records") or [])
                for record in records:
                    color = (0, 255, 70) if int(record.get("status", 0)) == 1 else (70, 120, 255)
                    viewer.draw_circle(
                        float(record["pixel_x"]),
                        float(record["pixel_y"]),
                        radius=3,
                        color=color,
                    )
                if records:
                    first = records[0]
                    viewer.draw_circle(
                        float(first["pixel_x"]),
                        float(first["pixel_y"]),
                        radius=5,
                        color=(255, 255, 0),
                    )
                    viewer.draw_text(
                        float(first["pixel_x"]) + 8.0,
                        float(first["pixel_y"]) + 8.0,
                        os.path.basename(str(chunk.get("path") or "农机文件")),
                        color=(255, 255, 0),
                        size=10,
                    )
        finally:
            if hasattr(viewer, "end_overlay_layer"):
                viewer.end_overlay_layer()

    @Slot(str)
    def _on_route_display_mode_changed(self, mode):
        mode = str(mode or "continuous")
        if mode == "machine_file":
            if not isinstance(self._machine_route_preview, dict):
                self._route_display_mode = "continuous"
                self.image_view.set_route_display_mode("continuous")
                self.lbl_status.setText("尚未导出农机文件")
                return
            integrity_errors = self._machine_route_preview_integrity_errors(
                self._machine_route_preview
            )
            if integrity_errors:
                detail = "；".join(str(error) for error in integrity_errors[:4])
                if len(integrity_errors) > 4:
                    detail += f"；另有 {len(integrity_errors) - 4} 项"
                self._clear_machine_route_preview()
                self.lbl_status.setText("TXT 路径已被修改，请重新导出")
                self.log_panel.error(
                    "农机 TXT 完整性校验失败，已停止显示旧文件：" + detail
                )
                return
            self._show_machine_route_preview(self._machine_route_preview)
            self.lbl_status.setText(self._machine_route_preview["info"])
            self.log_panel.info(
                "路径显示：农机文件控制折线（0=转弯，1=作业，文件间不连线）"
            )
            return
        self._route_display_mode = "continuous"
        path_result = self._path_result_from_state()
        if getattr(self.state, "auto_path_planned", False) and path_result:
            self._show_path(path_result)
            self.route_info.update_route(
                path_result, path_result.get("validation", {})
            )
            self.lbl_status.setText("路径显示：连续规划轨迹")
            self.log_panel.info(
                "路径显示：连续规划轨迹；用于模拟和理论评价，不是实测轨迹"
            )
        else:
            self._restore_visual_from_state()
    def _on_segment_selected(self, idx):
        pr = self._pipeline_result or {}
        path = self._normalise_path_result(pr.get("path", {}))
        all_segs = list(path.get("ordered_segments", []) or [])
        seg = next(
            (item for item in all_segs if int(item.get("segment_index", -1)) == idx),
            None,
        )
        if seg is not None:
            self.log_panel.info(f"选中路线段 {idx + 1}: {seg['type']}")
            if self._tif_rgb is not None:
                ov = self._tif_rgb.copy()
                for t in all_segs:
                    pts = np.array(t.get("points", []))
                    if len(pts) > 1:
                        is_work = t.get("type") == "work"
                        color = (0, 150, 40) if is_work else (150, 50, 50)
                        cv2.polylines(ov, [pts.astype(np.int32)], False, color, 2 if is_work else 1)
                if seg.get("points"):
                    pts = np.array(seg["points"])
                    if len(pts) > 1:
                        hl_color = (0, 255, 80) if seg["type"] == "work" else (80, 80, 255)
                        cv2.polylines(ov, [pts.astype(np.int32)], False, hl_color, 5)
                self.image_view.set_image(ov)

    def _on_simulate_task_action(self):
        if getattr(self.state, "anim_active", False):
            self._on_animation_pause()
            return
        self._on_toggle_animation()

    def _handle_animation_error(self, exc, path=None):
        """Stop a failed preview without rolling back or discarding the route."""
        if self._anim_timer:
            self._anim_timer.stop()
        self._anim_timer = None
        self._anim_engine = None
        self.state.anim_active = False
        self.state.anim_paused = False
        self.image_view.set_animation_controls_visible(False)
        self.lbl_speed.clear()
        self._clear_anim_overlay()
        self._dispose_harvester_item()
        try:
            if path:
                self._show_path(path)
        except Exception as restore_exc:
            self.log_panel.error(
                f"模拟演示失败后恢复路线显示也失败: "
                f"{type(restore_exc).__name__}: {restore_exc}"
            )
        self._sync_task_statuses()
        self.top_toolbar.refresh_actions(self.state)
        self.log_panel.error(
            f"模拟演示启动/刷新失败，路线未回退: "
            f"{type(exc).__name__}: {exc}"
        )

    def _on_toggle_animation(self):
        if getattr(self.state, 'anim_active', False):
            self.state.anim_active = False
            self.state.anim_paused = False
            if self._anim_timer:
                self._anim_timer.stop()
                self._anim_timer = None
            self.image_view.set_animation_controls_visible(False)
            self.lbl_speed.clear()
            self.log_panel.info("已结束演示")
            self._dispose_harvester_item()
            self._show_path(
                self._path_result_from_state()
                or (self._pipeline_result or {}).get("path", {})
            )
            self._sync_task_statuses()
            self.top_toolbar.refresh_actions(self.state)
            return
        pr = self._pipeline_result or {}
        path = self._normalise_path_result(pr.get("path", {}))
        auto_segs = self.state.auto_path_segments
        if not auto_segs and not path:
            self.log_panel.warn("请先生成路径后再开始模拟演示。")
            return
        segments = self._build_anim_segments(path, auto_segs)
        if not segments:
            self.log_panel.warn("路径点不足：当前路径点不足，无法进行模拟演示，请重新执行路径规划。")
            return
        # DECISION-012: simulation always takes over from the continuous
        # canonical route.  The machine-file control chords remain available
        # through the selector after playback, but are never animation input.
        self._route_display_mode = "continuous"
        self.image_view.set_route_display_mode("continuous")
        continuous_path = self._path_result_from_state() or path
        if continuous_path:
            self._show_path(continuous_path)
        try:
            self._dispose_harvester_item()
            self._anim_phase = 0.0
            self._anim_engine = AnimationEngine(self.state, self.geo)
            self._anim_engine.setup(segments)
            # 首帧立即绘制，既让用户看到演示已经接管画布，也把几何/参数
            # 错误拦截在启动阶段，避免 Qt 定时器回调静默中断。
            self._render_anim_frame()
        except Exception as exc:
            self._handle_animation_error(exc, path)
            return
        self.state.anim_active = True
        self.state.anim_paused = False
        self.state.anim_speed = 1.0
        self.state.anim_frame = 0
        self.image_view.set_animation_controls_visible(True)
        self.image_view.set_animation_paused(False)
        self.image_view.set_animation_speed(1.0)
        self.image_view.set_animation_progress(0.0)
        self._sync_task_statuses()
        self.top_toolbar.refresh_actions(self.state)
        self.log_panel.info("开始演示")
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_frame_tick)
        self._anim_timer.start(33)

    def _build_anim_segments(self, path, auto_segs):
        if auto_segs:
            return auto_segs
        path = self._normalise_path_result(path)
        if path.get("full_path"):
            return self._auto_segments_from_path(path)
        segments = []
        tracks = path.get("tracks", [])
        turns = path.get("turns", [])
        for i, t in enumerate(tracks):
            pts = t.get("points", [])
            if len(pts) >= 2:
                seg = AutoPathSegment(
                    points=[PathPoint(float(x), float(y)) for x, y in self._coerce_points(pts)],
                    status=1, corridor_id=i,
                    length_m=self._segment_length(pts), segment_type="work")
                segments.append(seg)
            if i < len(turns):
                tpts = turns[i].get("points", [])
                if len(tpts) >= 2:
                    seg = AutoPathSegment(
                        points=[PathPoint(float(x), float(y)) for x, y in self._coerce_points(tpts)],
                        status=0, corridor_id=i,
                        length_m=self._segment_length(tpts), segment_type="turn")
                    segments.append(seg)
        return segments

    def _anim_frame_tick(self):
        if not self._anim_engine or not self.state.anim_active:
            if self._anim_timer:
                self._anim_timer.stop()
                self._anim_timer = None
            return
        path = (self._pipeline_result or {}).get("path", {})
        try:
            finished = not self._anim_engine.advance()
            s = self.state
            if finished or s.anim_frame >= s.anim_total_frames - 1:
                self.state.anim_active = False
                if self._anim_timer:
                    self._anim_timer.stop()
                    self._anim_timer = None
                self.image_view.set_animation_progress(1.0)
                self.image_view.set_animation_controls_visible(False)
                self.lbl_speed.clear()
                self.log_panel.info("演示完成")
                self._dispose_harvester_item()
                report = self._anim_engine._gen_report()
                if report:
                    self.log_panel.info(report)
                self._show_path(path)
                WorkflowUpdater.advance(self.state, "ANIMATION_DONE")
                self._sync_task_statuses()
                self.top_toolbar.refresh_actions(self.state)
                return
            self._render_anim_frame()
            ratio = self._anim_engine.progress_ratio()
            pct = int(round(ratio * 100))
            self.image_view.set_animation_progress(ratio)
            remaining = self._anim_engine.get_remaining_seconds()
            rem_str = f"{remaining:.0f}s" if remaining > 0 else "..."
            speed_str = f"{s.anim_speed:g}x"
            self.lbl_speed.setText(f"  进度 {pct}%  播放 {speed_str}  剩余 {rem_str}  ")
        except Exception as exc:
            self._handle_animation_error(exc, path)

    def _on_animation_pause(self):
        if not self._anim_engine or not self.state.anim_active:
            return
        self._anim_engine.toggle_pause()
        self.image_view.set_animation_paused(self.state.anim_paused)
        self._sync_task_statuses()
        self.lbl_status.setText("演示已暂停" if self.state.anim_paused else "演示中")

    def _on_animation_seek(self, ratio):
        if not self._anim_engine or not self.state.anim_active:
            return
        self._anim_engine.seek_ratio(float(ratio))
        self._render_anim_frame()
        actual_ratio = self._anim_engine.progress_ratio()
        self.image_view.set_animation_progress(actual_ratio)
        self.lbl_speed.setText(
            f"  进度 {actual_ratio * 100:.0f}%  播放 {self.state.anim_speed:g}x  "
        )

    def _on_animation_speed(self, ratio):
        if not self._anim_engine:
            return
        self._anim_engine.set_speed_ratio(float(ratio))
        self.image_view.set_animation_speed(self.state.anim_speed)
        self.lbl_status.setText(f"演示播放倍速 {self.state.anim_speed:g}x")

    def _clear_anim_overlay(self):
        viewer = self.image_view.viewer
        for item in getattr(self, "_anim_overlay_items", []):
            try:
                viewer._scene.removeItem(item)
            except RuntimeError:
                pass
            if item in viewer._overlay_group:
                viewer._overlay_group.remove(item)
        self._anim_overlay_items = []

    def _render_anim_frame(self):
        if self._tif_rgb is None:
            return
        pos = self._anim_engine.get_interpolated_position()
        if pos is None:
            return
        if isinstance(pos, dict):
            px, py = pos.get("x", 0), pos.get("y", 0)
            heading = pos.get("heading_rad", 0)
        else:
            point, heading = pos
            px, py = point
        if not getattr(self.state, "anim_paused", False):
            self._anim_phase += 0.033
        self._update_harvester_item(px, py, heading)

    def _draw_harvester_pose(self, x, y, heading):
        """Draw the same canonical planar geometry used by footprint validation."""
        from harvester_geometry import planar_geometry_px, transform_polygon

        viewer = self.image_view.viewer
        params = self._effective_harvester_params(getattr(self.state, 'harvester_params', {}) or {})
        try:
            ppm = self.geo.pixels_per_meter(x, y) if self.geo.is_ready() else 20.0
        except Exception:
            ppm = 20.0
        ppm = max(1.0, float(ppm))
        geometry = planar_geometry_px(params, (float(x), float(y)), heading, ppm)

        def polygon(name, fill, outline):
            points = [(float(px), float(py)) for px, py in geometry[name]]
            return viewer.draw_polygon(points, color=fill, outline=outline, width=1)

        polygon("body", (30, 150, 205), (190, 225, 240))
        polygon("cutter_visual", (178, 132, 48), (244, 210, 110))
        cutter_line = geometry["cutter_visual_edge"]
        viewer.draw_line(
            float(cutter_line[0][0]),
            float(cutter_line[0][1]),
            float(cutter_line[1][0]),
            float(cutter_line[1][1]),
            color=(255, 225, 120),
            width=max(2, int(round(ppm * 0.045))),
        )
        polygon("left_track", (32, 36, 34), (115, 125, 120))
        polygon("right_track", (32, 36, 34), (115, 125, 120))
        dimensions = geometry["dimensions_m"]
        tank_local = np.asarray([
            (-dimensions["body_length_m"] * 0.325, -dimensions["body_width_m"] * 0.275),
            (-dimensions["body_length_m"] * 0.025, -dimensions["body_width_m"] * 0.275),
            (-dimensions["body_length_m"] * 0.025, dimensions["body_width_m"] * 0.275),
            (-dimensions["body_length_m"] * 0.325, dimensions["body_width_m"] * 0.275),
        ])
        tank = transform_polygon(tank_local, (x, y), heading, ppm)
        viewer.draw_polygon(
            [(float(px), float(py)) for px, py in tank],
            color=(54, 122, 170), outline=(180, 213, 228), width=1,
        )
        cab_local = np.asarray([[
            dimensions["body_length_m"] * 0.18,
            -dimensions["body_width_m"] * 0.25,
        ]])
        cab = transform_polygon(cab_local, (x, y), heading, ppm)[0]
        cab_radius = max(
            3,
            int(min(dimensions["body_length_m"], dimensions["body_width_m"]) * ppm * 0.19),
        )
        viewer.draw_circle(
            float(cab[0]), float(cab[1]),
            radius=cab_radius,
            color=(150, 205, 220),
            scene_scaled=True,
        )
        center_radius = max(2, int(round(ppm * 0.08)))
        viewer.draw_circle(
            x, y,
            radius=center_radius,
            color=(255, 255, 0),
            scene_scaled=True,
        )

    def _update_harvester_item(self, x, y, heading):
        """Place the detailed visual machine at the animation pose."""
        viewer = self.image_view.viewer
        scene = getattr(viewer, "_scene", None)
        if scene is None:
            return
        if self._harvester_item is None:
            stale = True
        else:
            try:
                stale = self._harvester_item.scene() is not scene
            except RuntimeError:
                stale = True
        if stale:
            self._dispose_harvester_item()
            params = self._effective_harvester_params(
                getattr(self.state, "harvester_params", {}) or {}
            )
            self._harvester_item = HarvesterItem(params)
            scene.addItem(self._harvester_item)
        try:
            ppm = self.geo.pixels_per_meter(x, y) if self.geo.is_ready() else 20.0
        except Exception:
            ppm = 20.0
        ppm = max(2.0, float(ppm))
        self._harvester_item.set_pose(x, y, heading, ppm, self._anim_phase)
        self._update_cut_swath(viewer, x, y, heading, ppm)

    def _update_cut_swath(self, viewer, x, y, heading, ppm):
        """Draw the already-cut swath as a wide stroke along the worked path."""
        from PySide6.QtGui import QColor, QPainterPath, QPen
        from PySide6.QtWidgets import QGraphicsPathItem

        scene = getattr(viewer, "_scene", None)
        if scene is None:
            return
        if self._cut_swath_item is None:
            stale = True
        else:
            try:
                stale = self._cut_swath_item.scene() is not scene
            except RuntimeError:
                stale = True
        if stale:
            self._cut_swath_item = QGraphicsPathItem()
            self._cut_swath_item.setZValue(18)
            scene.addItem(self._cut_swath_item)

        pts = list(getattr(self.state, "anim_all_pts", []) or [])
        types = list(getattr(self.state, "anim_all_types", []) or [])
        frame = int(getattr(self.state, "anim_frame", 0) or 0)
        if not pts:
            return
        end = min(len(pts), frame + 1)
        step = max(1, end // 700)
        indices = list(range(0, end, step))
        if indices and indices[-1] != end - 1:
            indices.append(end - 1)
        path = QPainterPath()
        started = False
        for index in indices:
            if types and index < len(types) and types[index] != "work":
                started = False
                continue
            px, py = pts[index]
            if not started:
                path.moveTo(float(px), float(py))
                started = True
            else:
                path.lineTo(float(px), float(py))
        if types and 0 <= frame < len(types) and types[frame] == "work":
            import math
            offset = float(self._harvester_item.cutter_x)
            path.lineTo(
                float(x) + math.cos(heading) * offset,
                float(y) + math.sin(heading) * offset,
            )

        pen = QPen(QColor(214, 205, 130, 150))
        pen.setWidthF(max(2.0, float(self._harvester_item.cw) * ppm))
        pen.setCapStyle(Qt.FlatCap)
        pen.setJoinStyle(Qt.RoundJoin)
        self._cut_swath_item.setPen(pen)
        self._cut_swath_item.setPath(path)

    def _dispose_harvester_item(self):
        """Remove the visual machine and cut swath from the scene."""
        for attr in ("_harvester_item", "_cut_swath_item"):
            item = getattr(self, attr, None)
            if item is None:
                continue
            try:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            except RuntimeError:
                pass
            setattr(self, attr, None)

    def _on_export_geojson(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 GeoJSON", "path_result.geojson", "GeoJSON (*.geojson)")
        if not path:
            return
        features = []
        try:
            path_result, geo_points = self._geo_export_payload()
        except Exception as e:
            self._show_geo_export_error("GeoJSON", e)
            return
        points_by_segment = {}
        for point in geo_points:
            segment_index = int(point.get("segment_index", 0))
            points_by_segment.setdefault(segment_index, []).append(point)
        for i, segment in enumerate(path_result.get("ordered_segments", [])):
            segment_points = points_by_segment.get(i, [])
            if len(segment_points) >= 2:
                coords = [
                    [float(point["lon"]), float(point["lat"])]
                    for point in segment_points
                ]
                segment_type = str(segment.get("type", "work"))
                features.append({
                    "type": "Feature",
                    "properties": {
                        "id": i + 1,
                        "segment_index": int(segment.get("segment_index", i)),
                        "type": segment_type,
                    },
                    "geometry": {"type": "LineString", "coordinates": coords},
                })
        try:
            save_json_atomic(
                path,
                {"type": "FeatureCollection", "features": features},
            )
            manifest = self._write_export_manifest(
                path, "GeoJSON", path_result, geo_points
            )
            WorkflowUpdater.advance(self.state, "EXPORTED")
            self.log_panel.success(f"已导出 GeoJSON：{os.path.basename(path)}；校验记录已保存")
        except Exception as e:
            self._show_geo_export_error("GeoJSON", e)

    def _on_export_csv_geo(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "path_result.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            _path_result, geo_points = self._geo_export_payload()
            from path_planner import export_csv
            export_csv(geo_points, path)
            manifest = self._write_export_manifest(path, "CSV", _path_result, geo_points)
            WorkflowUpdater.advance(self.state, "EXPORTED")
            self.log_panel.success(f"已导出 CSV：{os.path.basename(path)}；校验记录已保存")
        except Exception as e:
            self._show_geo_export_error("CSV", e)

    def _on_export_kml(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 KML", "path_result.kml", "KML (*.kml)")
        if not path:
            return
        try:
            _path_result, geo_points = self._geo_export_payload()
            from path_planner import export_kml
            export_kml(geo_points, path)
            manifest = self._write_export_manifest(path, "KML", _path_result, geo_points)
            WorkflowUpdater.advance(self.state, "EXPORTED")
            self.log_panel.success(f"已导出 KML：{os.path.basename(path)}；校验记录已保存")
        except Exception as e:
            self._show_geo_export_error("KML", e)

    def _on_export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", "path_result.json", "JSON (*.json)")
        if not path:
            return
        try:
            path_result, geo_points = self._geo_export_payload()
            from path_planner import export_json
            export_json(geo_points, path_result.get("validation", {}), path)
            manifest = self._write_export_manifest(path, "JSON", path_result, geo_points)
            WorkflowUpdater.advance(self.state, "EXPORTED")
            self.log_panel.success(f"已导出 JSON：{os.path.basename(path)}；校验记录已保存")
        except Exception as e:
            self._show_geo_export_error("JSON", e)

    def _on_export_path_format(self):
        """Directly export the research $PATH route as text.

        The trial-package export remains available from the File menu
        ("导出完整试验资料包...") for operators who need the full bundle.
        """
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "导出科研路径", "research_path.txt",
                "Text (*.txt)",
            )
            if not path:
                return
            path = self._ensure_txt_output_path(path)
            self._export_candidate_path(path)
            message = (
                f"已导出科研路径：{os.path.basename(path)}；"
                "路径参数与校验记录已保存"
            )
            WorkflowUpdater.advance(self.state, "EXPORTED")
            self.log_panel.success(message)
            self.lbl_status.setText("路径导出完成")
        except Exception as e:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            self._show_geo_export_error("$PATH", e)

    @staticmethod
    def _ensure_txt_output_path(path):
        """Keep user-facing PATH exports as UTF-8 ``.txt`` files."""
        path = os.path.abspath(os.fspath(path))
        stem, suffix = os.path.splitext(path)
        return path if suffix.lower() == ".txt" else f"{stem}.txt" if suffix else f"{path}.txt"

    def _on_export_machine_compatibility_path(self):
        """Export the terminal-specific 0/1 PATH projection as numbered files."""
        try:
            path = QFileDialog.getExistingDirectory(
                self,
                "选择农机兼容测试文件保存目录",
            )
            if not path:
                return
            route_path, _manifest_path = self._export_candidate_path(
                path,
                machine_compatibility=True,
                preview_only=True,
            )
            if isinstance(self._machine_route_preview, dict):
                self.image_view.set_route_display_mode(
                    "machine_file", emit=True
                )
            message = (
                f"已导出编号 TXT 路径：{os.path.dirname(route_path)}，首份为 {os.path.basename(route_path)}；"
                "转弯已压缩为端点直线，记录格式严格匹配终端样例，状态码固定为 2；"
                "已回读校验全部 TXT 与 manifest，画布已切换到“TXT 路径”视图"
            )
            self.log_panel.success(message)
            self.lbl_status.setText("农机兼容测试文件导出完成")
        except Exception as e:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            self._show_geo_export_error("农机兼容测试 $PATH", e)

    def _on_export_trial_package(self):
        """Export the complete trial package into a chosen directory."""
        parent = QFileDialog.getExistingDirectory(
            self, "选择完整试验资料包的保存位置"
        )
        if not parent:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            package_dir, _path, _manifest = self._export_complete_trial_package(parent)
        except Exception as exc:
            self._show_geo_export_error("完整试验资料包", exc)
            return
        finally:
            QApplication.restoreOverrideCursor()
        WorkflowUpdater.advance(self.state, "EXPORTED")
        self.log_panel.success(f"完整试验资料包已导出：{os.path.basename(package_dir)}")
        self.lbl_status.setText("资料包导出完成")

    def _export_candidate_path(
        self,
        path,
        path_result=None,
        geo_points=None,
        *,
        machine_compatibility=False,
        preview_only=False,
    ):
        """DECISION-001: every candidate path remains inseparable from its manifest."""
        if preview_only and not machine_compatibility:
            raise ValueError("只有农机兼容路径允许生成非正式测试文件")
        preserved_machine_files = (
            self._snapshot_machine_export_paths(path)
            if machine_compatibility
            else set()
        )
        if path_result is None or geo_points is None:
            path_result, geo_points = self._geo_export_payload(
                formal=not preview_only
            )
        else:
            if not preview_only:
                self._enforce_formal_parameter_and_recovery_gate(path_result)
        canonical_point_count = len(geo_points or [])
        compatibility_route_source = "canonical_dense_path_fallback"
        if machine_compatibility:
            control_geo_points = self._control_route_geo_points()
            if control_geo_points:
                # The terminal preview must serialize exactly the control
                # polyline shown by route editing.  The canonical dense route
                # remains the source for formal research validation above.
                geo_points = control_geo_points
                compatibility_route_source = "shared_route_control_polyline"
        protocol_metadata = None
        manifest_format = "$PATH"
        if machine_compatibility:
            from path_planner import export_machine_compatibility_path

            manifest_points, protocol_metadata = export_machine_compatibility_path(
                geo_points,
                path,
                field_polygon=(
                    np.asarray(self.state.field_boundary, dtype=np.float64)
                    if len(getattr(self.state, "field_boundary", []) or []) >= 3
                    else None
                ),
                forbidden_regions=(
                    getattr(self.state, "forbidden_regions", []) or []
                ),
            )
            manifest_points = list(manifest_points)
            manifest_format = "$PATH_MACHINE_COMPAT"
            protocol_metadata.update({
                "route_source": compatibility_route_source,
                "control_route_point_count": int(
                    len(getattr(self.state, "path_points", []) or [])
                    if compatibility_route_source == "shared_route_control_polyline"
                    else len(geo_points)
                ),
                "control_route_geo_record_count": int(len(geo_points)),
                "canonical_route_point_count": int(canonical_point_count),
                "control_route_status_semantics": "outgoing_edge_0_turn_1_work",
            })
            if preview_only:
                protocol_metadata.update({
                    "preview_only": True,
                    "machine_executable": False,
                    "formal_gate_bypassed": [
                        "harvester_parameter_source",
                        "machine_readiness",
                        "terminal_protocol_validation",
                    ],
                })
        else:
            from path_planner import export_path_format

            export_path_format(geo_points, path)
            manifest_points = geo_points
        manifest_output_path = path
        if isinstance(protocol_metadata, dict):
            output_files = list(protocol_metadata.get("output_files") or [])
            if output_files:
                manifest_output_path = output_files[0]
        try:
            if protocol_metadata is None:
                # Keep the original call shape for existing package/export
                # integrations that replace the manifest writer in tests.
                manifest = self._write_export_manifest(
                    manifest_output_path,
                    manifest_format,
                    path_result,
                    manifest_points,
                )
            else:
                manifest = self._write_export_manifest(
                    manifest_output_path,
                    manifest_format,
                    path_result,
                    manifest_points,
                    protocol_metadata=protocol_metadata,
                )
        except Exception:
            if machine_compatibility:
                self._cleanup_machine_export_artifacts(
                    protocol_metadata,
                    None,
                    preserved_paths=preserved_machine_files,
                )
            raise
        if machine_compatibility:
            try:
                self._verify_machine_export_manifest(manifest)
            except Exception:
                self._cleanup_machine_export_artifacts(
                    protocol_metadata,
                    manifest,
                    preserved_paths=preserved_machine_files,
                )
                raise
            # Only expose the preview after both the PATH text and manifest
            # have been written successfully.  It is built from the exact
            # post-compression records, not from the dense canonical route.
            preview = self._set_machine_route_preview(
                manifest_points, protocol_metadata or {}
            )
            if isinstance(preview, dict):
                preview["manifest_path"] = os.path.abspath(os.fspath(manifest))
        return manifest_output_path, manifest

    @staticmethod
    def _unique_package_path(parent):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(parent, f"align_field_trial_{stamp}")
        candidate = base
        suffix = 2
        while os.path.exists(candidate):
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _export_complete_trial_package(self, parent):
        """Export review visuals plus the same fail-closed candidate route."""
        from trial_package import export_visual_artifacts, write_package_index

        parent = os.path.abspath(parent)
        if not os.path.isdir(parent):
            raise ValueError("资料包保存位置不存在")
        path_result, geo_points = self._geo_export_payload()
        raw_mask = getattr(self.state, "mask_raw", None)
        mask_result = getattr(self.state, "mask_result", None) or {}
        processed = mask_result.get("processed_mask") if isinstance(mask_result, dict) else None
        if not isinstance(self._tif_rgb, np.ndarray):
            raise ValueError("缺少源影像显示数据，不能生成完整资料包")
        if not isinstance(raw_mask, np.ndarray) or not isinstance(processed, np.ndarray):
            raise ValueError("缺少原始或处理后掩膜，不能生成完整资料包")

        final_dir = self._unique_package_path(parent)
        staging_dir = tempfile.mkdtemp(prefix=".align_package_build_", dir=parent)
        moved = False
        try:
            export_visual_artifacts(
                staging_dir,
                source_image=self._tif_rgb,
                raw_mask=raw_mask,
                mask_result=mask_result,
                path_result=path_result,
                offset=(
                    int(getattr(self.state, "mask_offset_x", 0) or 0),
                    int(getattr(self.state, "mask_offset_y", 0) or 0),
                ),
                field_boundary=getattr(self.state, "field_boundary", []) or [],
                forbidden_regions=getattr(self.state, "forbidden_regions", []) or [],
                service_points={
                    "entry": getattr(self.state, "entry_point", None),
                    "exit": getattr(self.state, "exit_point", None),
                    "unload": getattr(self.state, "unload_points", []) or [],
                },
                geo=self.geo,
                mask_config=self.cfg.section("mask_processing"),
                inference_metadata=self._diagnostic_metadata("识别001", "inference"),
            )
            os.replace(staging_dir, final_dir)
            moved = True
            route_path = os.path.join(final_dir, "mission_field_trial.txt")
            route_path, manifest_path = self._export_candidate_path(
                route_path, path_result, geo_points
            )
            write_package_index(final_dir)
            return final_dir, route_path, manifest_path
        except Exception:
            cleanup = final_dir if moved else staging_dir
            if os.path.isdir(cleanup):
                common = os.path.commonpath((parent, os.path.abspath(cleanup)))
                if common == parent and os.path.abspath(cleanup) != parent:
                    shutil.rmtree(cleanup, ignore_errors=True)
            raise

    def _on_export_img(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "result.png", "PNG (*.png);;JPEG (*.jpg)")
        if not path or self._tif_rgb is None:
            return
        overlay = self._tif_rgb.copy()
        pr = self._pipeline_result or {}
        path_result = self._normalise_path_result(pr.get("path", {}))
        for track in path_result.get("tracks", []):
            pts = np.array(track.get("points", []))
            if len(pts) > 1:
                cv2.polylines(overlay, [pts.astype(np.int32)], False, (0, 200, 50), 3)
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        self.log_panel.success(f"已保存图像: {path}")

    def _next_diagnostic_default_path(self, prefix: str) -> str:
        """Return a non-overwriting default name for a stage diagnostic ZIP."""
        directory = str(self._diagnostic_export_dirs.get(str(prefix), "") or "")
        if not directory:
            directory = os.path.dirname(os.path.abspath(self._tif_path)) if self._tif_path else ""
        if not directory or not os.path.isdir(directory):
            directory = os.getcwd()
        index = 1
        while os.path.exists(os.path.join(directory, f"{prefix}{index:03d}.zip")):
            index += 1
        return os.path.join(directory, f"{prefix}{index:03d}.zip")

    def _reserve_diagnostic_path(self, path: str, prefix: str) -> str:
        """Make a user-selected diagnostic path collision-safe and remember it."""
        target = os.path.abspath(str(path or "").strip())
        if not target.lower().endswith(".zip"):
            target += ".zip"
        directory = os.path.dirname(target) or os.getcwd()
        os.makedirs(directory, exist_ok=True)
        stem = os.path.splitext(os.path.basename(target))[0]
        prefix = str(prefix or "诊断")
        if stem.startswith(prefix) and stem[len(prefix):].isdigit():
            index = max(1, int(stem[len(prefix):]))
            make_name = lambda value: f"{prefix}{value:03d}.zip"
        else:
            index = 1
            base = stem or prefix
            make_name = lambda value: f"{base}_{value:03d}.zip"
        candidate = target
        while os.path.exists(candidate):
            index += 1
            candidate = os.path.join(directory, make_name(index))
        self._diagnostic_export_dirs[prefix] = directory
        return candidate

    def _diagnostic_metadata(self, run_label: str, stage: str) -> dict:
        """Build the shared source/model/coordinate evidence for review exports."""
        source_path = str(self._tif_path or getattr(self.state, "tif_path", "") or "")
        source_width = int(getattr(self.state, "source_img_w", 0) or 0)
        source_height = int(getattr(self.state, "source_img_h", 0) or 0)
        display_width = int(getattr(self.state, "display_w", 0) or 0)
        display_height = int(getattr(self.state, "display_h", 0) or 0)
        if isinstance(self._tif_rgb, np.ndarray) and self._tif_rgb.ndim >= 2:
            display_height, display_width = [int(value) for value in self._tif_rgb.shape[:2]]
        scale_x = float(getattr(self.state, "display_scale_x", 0.0) or 0.0)
        scale_y = float(getattr(self.state, "display_scale_y", 0.0) or 0.0)
        if scale_x <= 0 and source_width and display_width:
            scale_x = float(source_width) / float(display_width)
        if scale_y <= 0 and source_height and display_height:
            scale_y = float(source_height) / float(display_height)
        scale_x = scale_x if scale_x > 0 else 1.0
        scale_y = scale_y if scale_y > 0 else 1.0

        boundary_display = [
            [float(point[0]), float(point[1])]
            for point in (getattr(self.state, "field_boundary", []) or [])
            if len(point) >= 2
        ]
        boundary_source = [
            [float(point[0]) * scale_x, float(point[1]) * scale_y]
            for point in boundary_display
        ]
        boundary_lonlat = []
        if self.geo.is_ready():
            for point in boundary_source:
                try:
                    lon, lat = self.geo.pixel_to_lonlat(point[0], point[1])
                    boundary_lonlat.append([float(lon), float(lat)])
                except Exception:
                    boundary_lonlat = []
                    break

        model_path = str(getattr(self.state, "current_model_path", "") or "")
        inference_record = dict(getattr(self.state, "inference_provenance", {}) or {})
        inference_inputs = dict(inference_record.get("inputs") or {})
        source_identity_value = {}
        try:
            from cache import source_identity
            source_identity_value = source_identity(source_path) if source_path else {}
        except Exception:
            source_identity_value = {}
        source_sha256 = str(getattr(self.state, "source_sha256", "") or "")
        model_sha256 = str(getattr(self.state, "model_sha256", "") or "")
        if not model_sha256 and model_path:
            model_sha256 = self._sha256_file(model_path)
        crop_window = inference_inputs.get("source_crop_window") or []
        metadata = {
            "run_label": str(run_label or ""),
            "stage": str(stage),
            "source_image": {
                "path": os.path.abspath(source_path) if source_path else "",
                "sha256": source_sha256,
                "identity": source_identity_value,
                "source_raster_shape": [source_height, source_width],
                "display_shape": [display_height, display_width],
                "display_to_source_scale": {"x": scale_x, "y": scale_y},
                "metadata": self._manifest_jsonable(
                    getattr(self.state, "source_metadata", {}) or {}
                ),
            },
            "model": {
                "name": str(getattr(self.state, "current_model_name", "") or ""),
                "path": os.path.abspath(model_path) if model_path else "",
                "sha256": model_sha256,
                "registry": self._manifest_jsonable(self._model_registry_entry(model_path)),
            },
            "image_position": {
                "source_crop_window_xywh": self._manifest_jsonable(crop_window),
                "mask_offset_xy_display": [
                    int(getattr(self.state, "mask_offset_x", 0) or 0),
                    int(getattr(self.state, "mask_offset_y", 0) or 0),
                ],
                "field_boundary_display": boundary_display,
                "field_boundary_source": boundary_source,
                "field_boundary_lonlat": boundary_lonlat,
            },
            "inference": {
                "algorithm_version": str(inference_record.get("algorithm_version", "") or ""),
                "config": self._manifest_jsonable(
                    inference_inputs.get("inference_config")
                    or self.cfg.section("model")
                ),
                "fingerprint": str(inference_record.get("fingerprint", "") or ""),
                "artifact_sha256": str(inference_record.get("artifact_sha256", "") or ""),
            },
            "preprocessing": self._manifest_jsonable(
                getattr(self.state, "raster_preprocessing", {}) or {}
            ),
            "runtime": self._manifest_jsonable(
                getattr(self.state, "inference_runtime", {}) or {}
            ),
            "stage_provenance": {
                "inference": self._manifest_jsonable(inference_record),
                "mask": self._manifest_jsonable(
                    getattr(self.state, "mask_provenance", {}) or {}
                ),
            },
        }
        if self.geo.is_ready() and source_width > 0 and source_height > 0:
            try:
                metadata["source_image"]["geographic_extent"] = self._manifest_jsonable(
                    self.geo.validate_raster_extent(source_width, source_height)
                )
            except Exception:
                pass
        return metadata

    def _on_export_inference_diagnostics(self):
        raw_mask = getattr(self.state, "mask_raw", None)
        if (
            not getattr(self.state, "inference_done", False)
            or not isinstance(raw_mask, np.ndarray)
            or raw_mask.ndim != 2
            or not isinstance(self._tif_rgb, np.ndarray)
        ):
            self.log_panel.warn("请先完成 作物识别，再导出识别诊断。")
            return
        default_path = self._next_diagnostic_default_path("识别")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出识别诊断数据",
            default_path,
            "ZIP archive (*.zip)",
        )
        if not path:
            return
        path = self._reserve_diagnostic_path(path, "识别")
        run_label = os.path.splitext(os.path.basename(path))[0]
        preprocessing = dict(getattr(self.state, "raster_preprocessing", {}) or {})
        tiled = dict(preprocessing.get("tiled_inference") or {})
        tile_trace = preprocessing.get("tile_trace") or tiled.get("tile_trace") or []
        context_trace = preprocessing.get("context_tile_trace") or []
        worker = InferenceDiagnosticExportWorker(
            path,
            self._tif_rgb,
            raw_mask,
            options={
                "metadata": self._diagnostic_metadata(run_label, "inference"),
                "mask_offset": (
                    int(getattr(self.state, "mask_offset_x", 0) or 0),
                    int(getattr(self.state, "mask_offset_y", 0) or 0),
                ),
                "field_boundary": list(getattr(self.state, "field_boundary", []) or []),
                "tile_trace": tile_trace,
                "context_tile_trace": context_trace,
            },
        )
        worker.finished.connect(self._on_inference_diagnostic_exported)
        self._start_background_worker(worker, "正在生成识别诊断数据...")

    @Slot(dict)
    def _on_inference_diagnostic_exported(self, result):
        self.progress_bar.hide()
        self.top_toolbar.set_progress(0)
        self.top_toolbar.set_worker_cancel_visible(False)
        if hasattr(self, "btn_cancel_task"):
            self.btn_cancel_task.hide()
        filename = os.path.basename(str(result.get("path", "识别001.zip")))
        summary = dict(result.get("tile_summary", {}).get("primary", {}) or {})
        self.log_panel.success(
            f"识别诊断数据已导出：{filename}；主切片 {summary.get('tile_count', 0)} 块，"
            f"有实例切片 {summary.get('tiles_with_accepted_instances', 0)} 块"
        )
        self.lbl_status.setText("识别诊断数据已导出")
        if self._worker_thread:
            self._worker_thread.quit()

    def _on_export_mask_diagnostics(self):
        raw_mask = getattr(self.state, "mask_raw", None)
        mask_result = getattr(self.state, "mask_result", None) or {}
        processed = mask_result.get("processed_mask") if isinstance(mask_result, dict) else None
        if not isinstance(raw_mask, np.ndarray) or not isinstance(processed, np.ndarray):
            self.log_panel.warn("请先完成 作物识别和掩膜优化。")
            return
        default_name = self._next_diagnostic_default_path("掩膜优化")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出掩膜诊断数据",
            default_name,
            "ZIP archive (*.zip)",
        )
        if not path:
            return
        path = self._reserve_diagnostic_path(path, "掩膜优化")
        worker = MaskDiagnosticExportWorker(
            path,
            raw_mask,
            mask_result,
            options={
                "offset": (
                    int(getattr(self.state, "mask_offset_x", 0) or 0),
                    int(getattr(self.state, "mask_offset_y", 0) or 0),
                ),
                "geo": self.geo,
                "config": self.cfg.section("mask_processing"),
                "source_image": self._tif_rgb,
                "metadata": self._diagnostic_metadata(
                    os.path.splitext(os.path.basename(path))[0],
                    "mask_optimization",
                ),
            },
        )
        worker.finished.connect(self._on_mask_diagnostic_exported)
        self._start_background_worker(worker, "正在生成掩膜诊断数据...")

    @Slot(dict)
    def _on_mask_diagnostic_exported(self, result):
        self.progress_bar.hide()
        self.top_toolbar.set_progress(0)
        self.top_toolbar.set_worker_cancel_visible(False)
        if hasattr(self, "btn_cancel_task"):
            self.btn_cancel_task.hide()
        filename = os.path.basename(str(result.get("path", "mask_diagnostic.zip")))
        count = int(result.get("review_component_count", 0) or 0)
        self.log_panel.success(
            f"掩膜诊断数据已导出：{filename}；需复核区域 {count} 个"
        )
        self.lbl_status.setText("掩膜诊断数据已导出")
        if self._worker_thread:
            self._worker_thread.quit()
    def _on_params_dialog(self):
        """DECISION-007: changed machine footprint requires replanning."""
        mode = self._prompt_machine_parameter_mode()
        if mode == "custom":
            current = self._effective_harvester_params(
                getattr(self.state, "harvester_params", {}) or {}
            )
            # 参数来源继续保存在内部 provenance 中，但不再要求用户在
            # 参数窗口里重复选择；窗口只负责编辑物理数值。
            dlg = HarvesterParamsDialog(current, self)
            if dlg.exec() == QDialog.Accepted:
                values = dlg.get_params()
                self._apply_harvester_params(values, "user_entered")
        elif mode == "recommend":
            self._show_machine_recommendation(None)

    def _prompt_machine_parameter_mode(self):
        box = QMessageBox(self)
        box.setWindowTitle("农机尺寸配置")
        box.setIcon(QMessageBox.Question)
        box.setText(
            "请选择参数设置方式。智能推荐会根据处理后的宽窄行通道净宽和田头空间"
            "给出候选尺寸；具体机型仍需人工核对。"
        )
        custom_button = box.addButton("自定义输入", QMessageBox.AcceptRole)
        recommend_button = box.addButton("智能推荐尺寸", QMessageBox.ActionRole)
        cancel_button = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(custom_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is custom_button:
            return "custom"
        if clicked is recommend_button:
            return "recommend"
        return ""

    def _apply_harvester_params(self, values, source, *, require_cutter_offset=True):
        values = dict(values or {})
        explicit_source = values.pop("_parameter_source", None)
        source_key = self._normalise_harvester_param_source(
            explicit_source or source
        )
        source_label = HARVESTER_PARAM_SOURCE_LABELS.get(source_key, source_key)
        params = self._effective_harvester_params(values)
        if require_cutter_offset and not params.get("cutter_forward_offset_m"):
            self.log_panel.warn("农机参数不完整：割刀前置距离必须根据实际收获机测量，不能由系统推荐或推算。")
            return False
        previous = self._effective_harvester_params(
            getattr(self.state, "harvester_params", {}) or {}
        )
        current_raw = getattr(self.state, "harvester_params", {}) or {}
        previous_sources = self._normalise_harvester_param_sources(
            previous,
            getattr(self.state, "harvester_param_sources", {}) or {},
        )
        next_sources = self._default_harvester_param_sources(
            params,
            fallback=source_key,
        )
        def differs(key):
            before = previous.get(key)
            after = params.get(key)
            if before is None or after is None:
                return before is not after
            return abs(float(before) - float(after)) > 1e-9

        def state_needs_sync():
            if not isinstance(current_raw, dict):
                return True
            for key, after in params.items():
                before = current_raw.get(key)
                if before is None or after is None:
                    if before is not after:
                        return True
                else:
                    try:
                        if abs(float(before) - float(after)) > 1e-9:
                            return True
                    except (TypeError, ValueError):
                        if before != after:
                            return True
            return False

        changed = any(differs(key) for key in params)
        sources_changed = previous_sources != next_sources
        if not changed:
            if state_needs_sync() or sources_changed:
                self.state.harvester_params = params
                self.state.harvester_param_sources = next_sources
                self.state.harvester_params_confirmed = True
                self._sync_task_statuses()
                self.log_panel.info(
                    f"{source_label}农机参数与当前配置一致，已应用到当前任务；"
                    f"来源已记录为{source_label}"
                )
                return True
            self.state.harvester_params_confirmed = True
            self._sync_task_statuses()
            self.log_panel.info(f"{source_label}农机参数与当前配置一致，已确认")
            return True
        self._save_system_undo("修改农机参数")
        self.state.harvester_params = params
        self.state.harvester_param_sources = next_sources
        self.state.harvester_params_confirmed = True
        config_snapshot = self.cfg.update_section("harvester", params)
        save_json_atomic(self.cfg.config_path(), config_snapshot)
        if getattr(self.state, "auto_path_planned", False):
            self._invalidate_analysis_from(
                "path",
                "农机物理参数已变化，旧路径足迹验证已经失效",
            )
        self._sync_task_statuses()
        cutter_offset = params.get("cutter_forward_offset_m")
        cutter_offset_text = (
            f"{float(cutter_offset):.2f}m" if cutter_offset else "未设置"
        )
        self.log_panel.success(
            f"{source_label}农机参数已保存: "
            f"割台={params['cutter_width_m']:.2f}m, "
            f"割刀前置={cutter_offset_text}, "
            f"履带={params['track_width_m']:.2f}m, "
            f"中心距={params['track_gauge_m']:.2f}m, "
            f"转弯半径={params['turn_radius_m']:.2f}m"
        )
        if source_key != "user_measured":
            self.log_panel.warn(
                f"参数来源仍为{source_label}；当前参数建议用于几何规划，请按实际机具补充测量"
            )
        if not cutter_offset:
            self.log_panel.warn("割刀前置距离未设置；路径规划前仍需按实际收获机测量填写")
        return True

    def _on_settings_dialog(self):
        previous = self.cfg.snapshot()
        if SettingsDialog(self.cfg.snapshot(), self).exec() == QDialog.Accepted:
            self.cfg.load()
            current = self.cfg.snapshot()
            self._mask_processing_strength = MASK_PROCESSING_STRENGTH
            self.task_panel.set_mask_strength(self._mask_processing_strength)
            if previous.get("model", {}) != current.get("model", {}):
                self._invalidate_analysis_from(
                    "inference",
                    "AI 推理参数已变化，旧识别掩膜和路径不能继续使用",
                )
            elif previous.get("mask_processing", {}) != current.get("mask_processing", {}):
                self._invalidate_analysis_from(
                    "mask",
                    "掩膜优化参数已变化，必须重新处理和规划",
                )
            elif previous.get("path_planning", {}) != current.get("path_planning", {}):
                self._invalidate_analysis_from(
                    "path",
                    "路径规划参数已变化，必须重新规划和验证",
                )

    def _on_project_log(self):
        entries = self._journal.recent(100)
        ProjectLogDialog(entries, self).exec()

    def _on_diagnose(self):
        DiagnosisDialog({
            "Python": sys.version,
            "影像": os.path.basename(self._tif_path or "?"),
            "模型": "已加载" if self.model_engine.is_loaded() else "未加载",
            "田块边界": f"{len(getattr(self.state, 'field_boundary', []))} 点",
            "路径点": f"{len(getattr(self.state, 'path_points', []))} 点",
            "地理坐标": "可用" if self.geo.is_ready() else "不可用",
            "转弯策略": getattr(self.state, 'turn_strategy', 'auto'),
        }, self).exec()

    def _build_machine_recommendation(self, path_result):
        from machine_recommendation import recommend_machine_dimensions

        mask_result = getattr(self.state, "mask_result", None) or {}
        hp = self._effective_harvester_params(getattr(self.state, "harvester_params", {}) or {})
        return recommend_machine_dimensions(mask_result, hp, path_result or {})

    def _show_machine_recommendation(self, path_result):
        recommendation = self._build_machine_recommendation(path_result)
        dlg = MachineRecommendationDialog(recommendation, self)
        if dlg.exec() == QDialog.Accepted:
            self._apply_harvester_params(
                dlg.get_params(),
                "recommended",
                require_cutter_offset=False,
            )

    def _show_validation_report(self, path_result, validation):
        path_result = self._normalise_path_result(path_result)
        data = {
            "track_count": len(path_result.get("tracks", [])),
            "turn_count": len(path_result.get("formal_turns", path_result.get("turns", []))),
            "total_length_m": validation.get("total_length_m", 0),
            "area_m2": validation.get("harvest_area_m2", 0),
            "turn_strategy": getattr(self.state, 'turn_strategy', 'auto'),
            "cutter_width": getattr(self.state, 'harvester_params', {}).get(
                'cutter_width_m', DEFAULT_HARVESTER_PARAMS["cutter_width_m"]
            ),
        }
        extra = []
        if "crop_mask_geometric_coverage_pct" in validation:
            extra.append(
                f"作物掩膜几何覆盖率: "
                f"{validation['crop_mask_geometric_coverage_pct']:.1f}%"
            )
        if "rolling_canopy_pct" in validation:
            extra.append(f"预计碾压率: {validation['rolling_canopy_pct']:.1f}%")
        if "field_efficiency" in validation:
            extra.append(f"效率: {validation['field_efficiency']:.1f}%")
        if "crossing_count" in validation:
            extra.append(f"交叉点: {validation['crossing_count']}")
        if extra:
            self.log_panel.info("路径指标: " + " | ".join(extra))

    def _warn_path_validation_issues(self, path_result):
        path_result = path_result or {}
        validation = path_result.get("validation", {}) or {}
        assessment = path_result.get("turn_assessment", {}) or {}
        issues = []
        for item in list(validation.get("issues") or []) + list(
            assessment.get("hard_reasons") or []
        ):
            text = str(item)
            if text and text not in issues:
                issues.append(text)
        warnings = list(assessment.get("warnings") or [])
        if issues or warnings:
            # Metrics and the original issue list remain visible in the route
            # result/manifest. This callback must not duplicate them as a
            # warning or change a research export into a blocked state.
            self.lbl_status.setText("路径已生成，指标已更新")

    def _on_route_edit_requested(self):
        if self._route_edit_active:
            self._route_edit_active = False
            self._route_edit_mode = "select"
            self._route_new_line_start = None
            self._route_drag_idx = -1
            self._route_drag_work_index = -1
            self._sync_drawing_mode()
            # DECISION-007/019: leaving the editor must keep the edited
            # control route as the current route.  The old pipeline snapshot
            # is only provenance for the automatically planned route and
            # must not overwrite the user's edits on exit.
            self._sync_auto_path_from_editable_route()
            self._restore_visual_from_state()
            self.task_panel.set_route_editing_active(False)
            self.log_panel.info("已退出路线编辑")
            return
        has_route = bool(
            getattr(self.state, 'auto_path', None)
            or len(getattr(self.state, 'path_points', [])) >= 2
        )
        if not has_route:
            self.log_panel.info("尚无路线：请在画布依次点击第一条作业线的起点和终点。")
        # ``path_points/path_status`` is the editor's canonical control route.
        # Rebuilding it from ``auto_path`` here would collapse user-inserted
        # work-line waypoints back to the original endpoints every time the
        # editor is reopened.  Only bootstrap it when no control route exists.
        if (
            getattr(self.state, 'auto_path', None)
            and len(getattr(self.state, "path_points", []) or []) < 2
        ):
            self._prepare_route_editing()
        self._route_edit_active = True
        self._route_edit_mode = "select" if has_route else "add_work"
        self._route_new_line_start = None
        self._route_drag_idx = -1
        self._route_drag_work_index = -1
        self.state.selected_type = None
        self.state.selected_idx = -1
        self.state.entry_exit_mode = False
        self._field_drawing = False
        self._sync_drawing_mode()
        self.task_panel.set_route_editing_active(True)
        self._refresh_route_edit_info_panel()
        self._draw_route_edit_overlay()
        self.log_panel.info(
            "人工路线编辑已开启：可拖动点线、在线段上插入或在末端追加控制点；"
            "下方路径结果可单击联动、切换直线/转弯、右击编辑/验证并拖动作业线排序"
        )

    def _on_route_edit_tool_requested(self, action):
        """DECISION-019: dispatch visible work-line editing operations."""
        action = str(action or "select")
        if action in {"select", "add_work", "add_point"}:
            self._route_edit_mode = action
            self._route_new_line_start = None
            self.task_panel.set_route_edit_mode(action)
            labels = {
                "select": "选择/拖动：点击端点自由移动，点击线段选择",
                "add_work": "新增作业线：依次点击作业线起点和终点",
                "add_point": "添加控制点：在线段上插入，空白处从路线末端追加",
            }
            self.lbl_status.setText(labels[action])
            self._draw_route_edit_overlay()
            return

        work_index = self._route_edit_selected_work_index()
        edge_index = (
            int(getattr(self.state, "selected_idx", -1))
            if getattr(self.state, "selected_type", None) == "edge"
            else -1
        )
        changed = False
        if action == "delete_work":
            changed = self._route_edit_delete_work_line(work_index)
        elif action == "delete_point":
            selected_node = (
                int(getattr(self.state, "selected_idx", -1))
                if getattr(self.state, "selected_type", None) == "node"
                else -1
            )
            changed = self._route_edit_delete_point(selected_node)
            if changed:
                self._sync_auto_path_from_editable_route()
        elif action == "move_earlier":
            changed = self._route_edit_reorder_work_line(work_index, -1) != work_index
        elif action == "move_later":
            changed = self._route_edit_reorder_work_line(work_index, 1) != work_index
        elif action == "reverse":
            changed = self._route_edit_reverse_work_line(work_index)
        elif action == "set_work":
            changed = self._route_edit_set_edge_status(edge_index, 1)
        elif action == "set_turn":
            changed = self._route_edit_set_edge_status(edge_index, 0)
        elif action == "toggle_status":
            statuses = list(getattr(self.state, "path_status", []) or [])
            if 0 <= edge_index < len(statuses):
                selected_record = next(
                    (
                        record for record in self._route_edit_segment_records()
                        if int(record["edge_start"]) <= edge_index
                        <= int(record["edge_end"])
                    ),
                    None,
                )
                if selected_record is not None:
                    changed = self._route_edit_set_segment_status(
                        selected_record["edge_start"],
                        selected_record["edge_end"],
                        0 if int(statuses[edge_index]) == 1 else 1,
                    )
        elif action == "toggle_geometry":
            modes = self._route_edge_geometry_modes(
                getattr(self.state, "path_points", []) or [],
                getattr(self.state, "path_status", []) or [],
                getattr(self.state, "path_geometry_modes", []) or [],
            )
            if 0 <= edge_index < len(modes):
                # The canvas selection identifies one point-to-point edge.
                # The result-card context menu may still pass a whole run,
                # but the compact toolbar must not change neighbouring edges
                # just because they share the same displayed card.
                changed = self._route_edit_set_segment_geometry_mode(
                    edge_index,
                    edge_index,
                    "straight" if modes[edge_index] == "turn" else "turn",
                )
        elif action == "undo":
            changed = bool(self.state.undo())
            if changed:
                self._sync_auto_path_from_editable_route()
        elif action == "redo":
            changed = bool(self.state.redo())
            if changed:
                self._sync_auto_path_from_editable_route()
        elif action == "validate":
            try:
                validation = self._validate_manual_route()
            except (TypeError, ValueError, RuntimeError) as exc:
                self.log_panel.error(f"人工路线验证失败：{exc}")
                self.lbl_status.setText("人工路线验证失败")
            else:
                issues = list(validation.get("issues") or [])
                if validation.get("valid") is True:
                    self.log_panel.success(
                        "人工路线几何验证通过；正式科研导出仍保留人工来源标记"
                    )
                    self.lbl_status.setText("人工路线几何验证通过")
                else:
                    self.log_panel.error(
                        "人工路线未通过验证：" + ("；".join(issues[:3]) or "未知原因")
                    )
                    self.lbl_status.setText("人工路线未通过验证")
            self._draw_route_edit_overlay()
            return

        if changed:
            self.log_panel.info("人工路线结构已更新，相邻转弯已重新衔接")
        else:
            self.log_panel.info("请先点击一条作业线或控制边")
        self._draw_route_edit_overlay()

    def _on_route_list_segment_selected(self, edge_start, edge_end):
        """DECISION-019: list and canvas share the same selected route edge."""
        if not self._route_edit_active:
            return
        points = list(getattr(self.state, "path_points", []) or [])
        edge_start = int(edge_start)
        if not 0 <= edge_start < len(points) - 1:
            return
        self.state.selected_type = "edge"
        self.state.selected_idx = edge_start
        self._route_edit_mode = "select"
        self.task_panel.set_route_edit_mode("select")
        self._draw_route_edit_overlay()
        self.lbl_status.setText(
            f"已从路径结果选中线段 {edge_start + 1}–{int(edge_end) + 1}"
        )

    def _route_edit_set_segment_status(self, edge_start, edge_end, status):
        points = list(getattr(self.state, "path_points", []) or [])
        edge_count = max(0, len(points) - 1)
        first = max(0, int(edge_start))
        last = min(edge_count - 1, int(edge_end))
        if first > last:
            return False
        statuses = list(getattr(self.state, "path_status", []) or [])[:edge_count]
        if len(statuses) < edge_count:
            statuses.extend([1] * (edge_count - len(statuses)))
        geometry_modes = self._route_edge_geometry_modes(
            points,
            statuses,
            getattr(self.state, "path_geometry_modes", []) or [],
        )
        target = 1 if int(status) == 1 else 0
        if all(int(statuses[index]) == target for index in range(first, last + 1)):
            return False
        self.state.save_route_undo(
            "整段设为作业" if target == 1 else "整段设为转弯"
        )
        for index in range(first, last + 1):
            statuses[index] = target
            geometry_modes[index] = "straight" if target == 1 else "turn"
        self.state.path_status = statuses
        self.state.path_geometry_modes = geometry_modes
        self.state.selected_type = "edge"
        self.state.selected_idx = first
        self._sync_auto_path_from_editable_route()
        return True

    def _route_edit_set_segment_geometry_mode(
        self, edge_start, edge_end, geometry_mode
    ):
        """Set the physical geometry of one or more adjacent control edges."""
        points = list(getattr(self.state, "path_points", []) or [])
        edge_count = max(0, len(points) - 1)
        first = max(0, int(edge_start))
        last = min(edge_count - 1, int(edge_end))
        if first > last:
            return False
        target = self._normalise_route_geometry_mode(geometry_mode)
        modes = self._route_edge_geometry_modes(
            points,
            getattr(self.state, "path_status", []) or [],
            getattr(self.state, "path_geometry_modes", []) or [],
        )
        if all(modes[index] == target for index in range(first, last + 1)):
            return False
        self.state.save_route_undo(
            "整段设为直线" if target == "straight" else "整段设为转弯"
        )
        for index in range(first, last + 1):
            modes[index] = target
        self.state.path_geometry_modes = modes
        self.state.selected_type = "edge"
        self.state.selected_idx = first
        self._sync_auto_path_from_editable_route()
        return True

    def _on_route_list_edit_action(self, action, first, last):
        """Apply a result-card edit to the authoritative control polyline."""
        if not self._route_edit_active:
            self.lbl_status.setText("请先点击“编辑路线”再修改路径结果")
            return
        action = str(action or "")
        first, last = int(first), int(last)
        changed = False
        message = ""
        if action == "validate_route":
            self._on_route_edit_tool_requested("validate")
            return
        if action == "toggle_segment":
            statuses = list(getattr(self.state, "path_status", []) or [])
            if 0 <= first < len(statuses):
                target = 0 if int(statuses[first]) == 1 else 1
                changed = self._route_edit_set_segment_status(first, last, target)
                message = "线段已切换为作业" if target == 1 else "线段已切换为转弯"
        elif action == "toggle_geometry":
            modes = self._route_edge_geometry_modes(
                getattr(self.state, "path_points", []) or [],
                getattr(self.state, "path_status", []) or [],
                getattr(self.state, "path_geometry_modes", []) or [],
            )
            if 0 <= first < len(modes):
                target = "straight" if modes[first] == "turn" else "turn"
                changed = self._route_edit_set_segment_geometry_mode(
                    first, last, target
                )
                message = "线段已切换为直线" if target == "straight" else "线段已切换为转弯"
        elif action == "add_waypoint":
            points = list(getattr(self.state, "path_points", []) or [])
            candidates = [
                index for index in range(max(0, first), min(last, len(points) - 2) + 1)
            ]
            if candidates:
                edge_index = max(
                    candidates,
                    key=lambda index: math.hypot(
                        float(points[index + 1][0]) - float(points[index][0]),
                        float(points[index + 1][1]) - float(points[index][1]),
                    ),
                )
                midpoint = (
                    0.5 * (float(points[edge_index][0]) + float(points[edge_index + 1][0])),
                    0.5 * (float(points[edge_index][1]) + float(points[edge_index + 1][1])),
                )
                node_index = self._route_edit_insert_point(
                    midpoint, preserve_position=True, edge_index=edge_index
                )
                self.state.selected_type = "node"
                self.state.selected_idx = node_index
                self._sync_auto_path_from_editable_route()
                changed = True
                message = f"已在线段中点添加航点 {node_index + 1}"
        elif action == "delete_waypoint":
            changed = self._route_edit_delete_point(first)
            if changed:
                self._sync_auto_path_from_editable_route()
                message = f"已删除航点 {first + 1}"
        elif action == "add_work_after":
            work_index = self._route_edit_work_index_for_edge(first)
            if work_index >= 0:
                self.state.selected_type = "edge"
                self.state.selected_idx = first
                self._on_route_edit_tool_requested("add_work")
                self.lbl_status.setText(
                    f"将在第 {work_index + 1} 条作业线后新增：请在画布依次点击起点和终点"
                )
            return
        elif action == "delete_work":
            work_index = self._route_edit_work_index_for_edge(first)
            changed = self._route_edit_delete_work_line(work_index)
            if changed:
                message = f"已删除第 {work_index + 1} 条作业线"
        elif action == "reverse_work":
            work_index = self._route_edit_work_index_for_edge(first)
            changed = self._route_edit_reverse_work_line(work_index)
            if changed:
                message = f"已反向第 {work_index + 1} 条作业线"
        if changed:
            self._draw_route_edit_overlay()
            self.lbl_status.setText(message)
            self.log_panel.info(message + "；路径验证与导出证据已失效")
        else:
            self.lbl_status.setText("当前卡片操作无法执行")

    def _on_route_list_work_reordered(self, source_work_index, target_work_index):
        if not self._route_edit_active:
            return
        source = int(source_work_index)
        target = int(target_work_index)
        new_index = self._route_edit_reorder_work_line(source, target - source)
        if new_index == target and source != target:
            self._draw_route_edit_overlay()
            self.lbl_status.setText(
                f"作业线已从第 {source + 1} 位移动到第 {target + 1} 位"
            )
            self.log_panel.info("已按卡片拖拽顺序重建相邻转弯连接")

    def _build_route_control_polyline(self, segments):
        """Build the single compact polyline shared by editing and compat export.

        Every work line is represented by its two endpoints.  Every contiguous
        non-work run is represented by one straight chord, because the target
        terminal accepts only the outgoing status values 1 (work) and 0
        (turn/other).  The canonical ``auto_path`` remains untouched and
        retains the dense research geometry used for scientific validation.
        """
        points = []
        statuses = []
        route_segments = list(segments or [])
        def simplify_work(source_points):
            values = np.asarray(source_points, dtype=np.float32)
            if len(values) <= 2:
                return [tuple(map(float, point)) for point in values]
            return [
                tuple(map(float, values[0])),
                tuple(map(float, values[-1])),
            ]

        def append_control_point(point, status):
            point = (float(point[0]), float(point[1]))
            if points and math.hypot(
                point[0] - points[-1][0], point[1] - points[-1][1]
            ) <= 1e-6:
                return
            if points:
                statuses.append(int(status))
            points.append(point)

        def same_point(first, second):
            return math.hypot(
                float(first[0]) - float(second[0]),
                float(first[1]) - float(second[1]),
            ) <= 1e-5

        # The terminal-facing sample starts with the user-selected entry
        # point and then enters the first work line.  Reference-only planning
        # keeps that approach out of the scientific ``auto_path``; add it to
        # the shared control projection so the numbered TXT file contains the
        # actual route start without changing the continuous validation path.
        entry_point = getattr(self.state, "entry_point", None)
        first_work_start = None
        first_segment_type = ""
        for candidate in route_segments:
            candidate_type = str(
                getattr(candidate, "segment_type", "") or "work"
            )
            candidate_points = self._coerce_points(
                getattr(candidate, "points", [])
            )
            if candidate_type == "work" and len(candidate_points) >= 2:
                first_work_start = candidate_points[0]
                first_segment_type = candidate_type
                break
            if candidate_points:
                first_segment_type = candidate_type
        has_explicit_entry_segment = bool(route_segments) and str(
            getattr(route_segments[0], "segment_type", "") or ""
        ) == "entry"
        if (
            entry_point
            and first_work_start is not None
            and not has_explicit_entry_segment
            and not same_point(entry_point, first_work_start)
        ):
            append_control_point(entry_point, 0)
            # The status is attached to the outgoing edge.  The edge from the
            # entry marker to the first work endpoint is therefore 0; the
            # following edge becomes 1 when the work segment is appended.
            append_control_point(first_work_start, 0)

        index = 0
        while index < len(route_segments):
            segment = route_segments[index]
            segment_type = str(
                getattr(segment, "segment_type", "") or "work"
            )
            if segment_type == "work":
                source_points = self._coerce_points(
                    getattr(segment, "points", [])
                )
                for point in simplify_work(source_points):
                    append_control_point(point, 1)
                index += 1
                continue

            # DECISION-016/017: preserve a topology-critical full retrace of
            # the immediately preceding harvested pass.  Collapsing this
            # turn_reverse together with the following turn_aux would recreate
            # an opposite-headland diagonal in the editor and machine file.
            previous_work_points = []
            if index > 0:
                previous_segment = route_segments[index - 1]
                previous_type = str(
                    getattr(previous_segment, "segment_type", "") or "work"
                )
                if previous_type == "work":
                    previous_work_points = simplify_work(
                        self._coerce_points(
                            getattr(previous_segment, "points", [])
                        )
                    )
            current_points = self._coerce_points(
                getattr(segment, "points", [])
            )
            preserves_harvested_retrace = bool(
                segment_type == "turn_reverse"
                and len(previous_work_points) >= 2
                and len(current_points) >= 2
                and same_point(current_points[0], previous_work_points[-1])
                and same_point(current_points[-1], previous_work_points[0])
            )
            if preserves_harvested_retrace:
                append_control_point(current_points[0], 0)
                append_control_point(current_points[-1], 0)
                index += 1
                run_end = None
                while index < len(route_segments):
                    candidate = route_segments[index]
                    candidate_type = str(
                        getattr(candidate, "segment_type", "") or "work"
                    )
                    if candidate_type == "work":
                        break
                    candidate_points = self._coerce_points(
                        getattr(candidate, "points", [])
                    )
                    if candidate_points:
                        run_end = candidate_points[-1]
                    index += 1
                if run_end is not None:
                    append_control_point(run_end, 0)
                continue

            # Collapse turn/entry/exit/reverse runs as one compatibility chord.
            run_start = None
            run_end = None
            next_index = index
            while next_index < len(route_segments):
                candidate = route_segments[next_index]
                candidate_type = str(
                    getattr(candidate, "segment_type", "") or "work"
                )
                if candidate_type == "work":
                    break
                candidate_points = self._coerce_points(
                    getattr(candidate, "points", [])
                )
                if candidate_points:
                    if run_start is None:
                        run_start = candidate_points[0]
                    run_end = candidate_points[-1]
                next_index += 1
            if run_start is not None:
                append_control_point(run_start, 0)
                if run_end is not None:
                    append_control_point(run_end, 0)
            index = next_index

        return points, statuses[: max(0, len(points) - 1)]

    def _control_route_segments(self, points, statuses, geometry_modes=None):
        """Group the control polyline by status and physical edge geometry."""
        points = [tuple(map(float, point)) for point in points or []]
        if len(points) < 2:
            return [], []
        expected = len(points) - 1
        normalised_statuses = []
        for value in list(statuses or [])[:expected]:
            try:
                normalised_statuses.append(1 if int(value) == 1 else 0)
            except (TypeError, ValueError):
                normalised_statuses.append(0)
        if len(normalised_statuses) < expected:
            normalised_statuses.extend([1] * (expected - len(normalised_statuses)))
        normalised_geometry_modes = self._route_edge_geometry_modes(
            points, normalised_statuses, geometry_modes
        )

        status_to_type = {0: "turn", 1: "work"}
        segments = []
        current_status = normalised_statuses[0]
        current_geometry_mode = normalised_geometry_modes[0]
        current_points = [points[0], points[1]]
        for edge_index in range(1, expected):
            edge_status = normalised_statuses[edge_index]
            edge_geometry_mode = normalised_geometry_modes[edge_index]
            if (
                edge_status == current_status
                and edge_geometry_mode == current_geometry_mode
            ):
                current_points.append(points[edge_index + 1])
                continue
            segments.append(
                self._route_segment_from_points(
                    current_points,
                    current_status,
                    status_to_type,
                    current_geometry_mode,
                )
            )
            current_status = edge_status
            current_geometry_mode = edge_geometry_mode
            current_points = [points[edge_index], points[edge_index + 1]]
        segments.append(
            self._route_segment_from_points(
                current_points,
                current_status,
                status_to_type,
                current_geometry_mode,
            )
        )
        return segments, normalised_statuses

    def _validate_pixel_path_within_image(self, full_path):
        """Reject control points outside the currently loaded display raster."""
        display_h, display_w = (
            self._tif_rgb.shape[:2]
            if isinstance(self._tif_rgb, np.ndarray)
            else (
                int(getattr(self.state, "display_h", 0) or 0),
                int(getattr(self.state, "display_w", 0) or 0),
            )
        )
        if display_w <= 0 or display_h <= 0:
            return
        for segment_index, segment in enumerate(full_path):
            for point_index, (pixel_x, pixel_y) in enumerate(segment):
                if not (
                    0.0 <= float(pixel_x) <= float(display_w - 1)
                    and 0.0 <= float(pixel_y) <= float(display_h - 1)
                ):
                    raise ValueError(
                        "璺緞鎺у埗鐐硅秴鍑哄奖鍍忚寖鍥达紝绂佹浣跨敤浠垮皠澶栨帹鍧愭爣: "
                        f"segment={segment_index}, point={point_index}, "
                        f"pixel=({float(pixel_x):.3f},{float(pixel_y):.3f})"
                    )

    def _control_route_geo_points(self):
        """Convert the visible/editor control route to validated geo points."""
        points = list(getattr(self.state, "path_points", []) or [])
        statuses = list(getattr(self.state, "path_status", []) or [])
        if len(points) < 2 and getattr(self.state, "auto_path", None):
            points, statuses = self._build_route_control_polyline(
                getattr(self.state, "auto_path", [])
            )
        if len(points) < 2:
            return []
        segments, _ = self._control_route_segments(
            points,
            statuses,
            getattr(self.state, "path_geometry_modes", []),
        )
        if not segments:
            return []
        full_path = [
            [(point.pixel_x, point.pixel_y) for point in segment.points]
            for segment in segments
        ]
        segment_types = [segment.segment_type for segment in segments]
        self._validate_pixel_path_within_image(full_path)
        from path_planner import global_path_to_geo, validate_geo_points

        geo_points = global_path_to_geo(full_path, self.geo, segment_types)
        return validate_geo_points(
            geo_points,
            expected_count=sum(len(segment) for segment in full_path),
            path_segments=full_path,
            segment_types=segment_types,
        )

    def _prepare_route_editing(self):
        """Build the same compact control polyline used by compat export."""
        st = self.state
        if not getattr(st, "auto_path", None):
            st.path_geometry_modes = self._route_edge_geometry_modes(
                getattr(st, "path_points", []) or [],
                getattr(st, "path_status", []) or [],
                getattr(st, "path_geometry_modes", []) or [],
            )
            return
        points, statuses = self._build_route_control_polyline(st.auto_path)
        if len(points) >= 2:
            if len(getattr(st, "path_points", []) or []) < 2:
                st.path_points = points
                st.path_status = statuses
                st.path_geometry_modes = self._route_edge_geometry_modes(
                    points, statuses, []
                )
            else:
                st.path_geometry_modes = self._route_edge_geometry_modes(
                    st.path_points,
                    st.path_status,
                    getattr(st, "path_geometry_modes", []) or [],
                )
            st.selected_type = None
            st.selected_idx = -1

    def _route_edit_segment_records(self):
        """Describe status runs without creating a second route model."""
        points = [
            tuple(map(float, point))
            for point in (getattr(self.state, "path_points", []) or [])
        ]
        control_segments, statuses = self._control_route_segments(
            points,
            getattr(self.state, "path_status", []),
            getattr(self.state, "path_geometry_modes", []),
        )
        if not statuses:
            return []
        records = []
        turn_index = 0
        edge_start = 0
        current_status = int(statuses[0])
        geometry_modes = self._route_edge_geometry_modes(
            points,
            statuses,
            getattr(self.state, "path_geometry_modes", []),
        )
        current_geometry_mode = geometry_modes[0]
        for edge_index in range(1, len(statuses) + 1):
            run_finished = (
                edge_index == len(statuses)
                or int(statuses[edge_index]) != current_status
                or geometry_modes[edge_index] != current_geometry_mode
            )
            if not run_finished:
                continue
            segment_index = len(records)
            edge_end = edge_index - 1
            if current_status == 1:
                display_number = 0
            else:
                turn_index += 1
                display_number = turn_index
            length_m = (
                float(control_segments[segment_index].length_m)
                if segment_index < len(control_segments)
                else 0.0
            )
            record_work_index = (
                self._route_edit_work_index_for_edge(edge_start)
                if current_status == 1
                else -1
            )
            records.append({
                "segment_index": segment_index,
                "type": "work" if current_status == 1 else "turn",
                "status": current_status,
                "geometry_mode": current_geometry_mode,
                "edge_start": edge_start,
                "edge_end": edge_end,
                "node_start": edge_start,
                "node_end": edge_index,
                "work_index": record_work_index,
                "display_number": (
                    record_work_index + 1
                    if current_status == 1
                    else display_number
                ),
                "length_m": length_m,
                "point_count": edge_index - edge_start + 1,
            })
            if edge_index < len(statuses):
                edge_start = edge_index
                current_status = int(statuses[edge_index])
                current_geometry_mode = geometry_modes[edge_index]
        return records

    def _refresh_route_edit_info_panel(self):
        self.route_info.update_editable_route(self._route_edit_segment_records())
        self.route_info.set_edit_selection(
            getattr(self.state, "selected_type", None),
            getattr(self.state, "selected_idx", -1),
        )

    def _simulation_segments_from_control_route(self, control_segments):
        """Expand compact turn chords for the virtual-machine preview only.

        The machine protocol deliberately receives one straight turn chord
        between two work-line endpoints.  The animation must not mistake that
        protocol projection for the physical path, so rebuild a bow-shaped
        preview from the same endpoints while leaving ``path_points`` and the
        export projection untouched.
        """
        segments = list(control_segments or [])
        if not segments:
            return []
        try:
            from path_planner import _turn_bow
            from row_geometry import meters_per_pixel
        except Exception:
            _turn_bow = None
            meters_per_pixel = None

        params = self._effective_harvester_params(
            getattr(self.state, "harvester_params", {}) or {}
        )
        try:
            turn_radius_m = float(params.get("turn_radius_m", self.cfg.TURN_RADIUS_M))
        except (TypeError, ValueError):
            turn_radius_m = float(self.cfg.TURN_RADIUS_M)
        if not math.isfinite(turn_radius_m) or turn_radius_m <= 0.0:
            turn_radius_m = float(self.cfg.TURN_RADIUS_M)

        mpp = 1.0
        mask = getattr(self.state, "mask_raw", None)
        if meters_per_pixel is not None and isinstance(mask, np.ndarray):
            try:
                value = float(meters_per_pixel(mask, self.geo, self.state))
                if math.isfinite(value) and value > 0.0:
                    mpp = value
            except (TypeError, ValueError, RuntimeError):
                pass

        def point_value(point):
            return (float(point.pixel_x), float(point.pixel_y))

        def work_direction(turn_index):
            for candidate_index in range(turn_index - 1, -1, -1):
                candidate = segments[candidate_index]
                if str(getattr(candidate, "segment_type", "")) != "work":
                    continue
                points = list(getattr(candidate, "points", []) or [])
                if len(points) < 2:
                    continue
                a = point_value(points[-2])
                b = point_value(points[-1])
                dx, dy = b[0] - a[0], b[1] - a[1]
                norm = math.hypot(dx, dy)
                if norm > 1e-9:
                    return dx / norm, dy / norm
            for candidate_index in range(turn_index + 1, len(segments)):
                candidate = segments[candidate_index]
                if str(getattr(candidate, "segment_type", "")) != "work":
                    continue
                points = list(getattr(candidate, "points", []) or [])
                if len(points) < 2:
                    continue
                a = point_value(points[0])
                b = point_value(points[1])
                dx, dy = b[0] - a[0], b[1] - a[1]
                norm = math.hypot(dx, dy)
                if norm > 1e-9:
                    return dx / norm, dy / norm
            return (1.0, 0.0)

        result = []
        for index, segment in enumerate(segments):
            segment_type = str(getattr(segment, "segment_type", "") or "work")
            points = list(getattr(segment, "points", []) or [])
            if len(points) < 2:
                result.append(segment)
                continue
            if self._normalise_route_geometry_mode(
                getattr(segment, "geometry_mode", "turn"), fallback="turn"
            ) != "turn":
                # A status-0 edge can deliberately be a straight connector;
                # keep its exact control chord for the physical preview.
                result.append(segment)
                continue
            if len(points) > 2:
                # DECISION-019: a turn with explicit intermediate control
                # points is operator-authored canonical geometry.  Only a
                # two-endpoint status-0 chord requests automatic turn-curve
                # reconstruction; never discard the operator's waypoints.
                result.append(segment)
                continue

            start = point_value(points[0])
            end = point_value(points[-1])
            direction = work_direction(index)
            lateral_x = end[0] - start[0]
            lateral_y = end[1] - start[1]
            cross = direction[0] * lateral_y - direction[1] * lateral_x
            turn_sign = 1.0 if cross >= 0.0 else -1.0
            gap_m = abs(cross) * mpp
            if gap_m <= 1e-9:
                gap_m = math.hypot(lateral_x, lateral_y) * mpp
            geometry_radius_m = min(
                turn_radius_m,
                max(1e-6, gap_m * 0.5),
            )
            curve = None
            if _turn_bow is not None:
                try:
                    curve = _turn_bow(
                        start,
                        end,
                        direction,
                        turn_sign,
                        mpp,
                        geometry_radius_m,
                        0.0,
                    )
                except (TypeError, ValueError, RuntimeError):
                    curve = None
            if not curve or len(curve) < 3:
                # Very short chords cannot support two sampled arcs.  Keep a
                # small quadratic bend so the virtual preview is still visibly
                # curved; the protocol/export chord remains unchanged.
                perp = (-direction[1] * turn_sign, direction[0] * turn_sign)
                chord_length = math.hypot(lateral_x, lateral_y)
                offset = min(30.0, max(1.0, chord_length * 0.35))
                curve = [
                    start,
                    (
                        (start[0] + end[0]) * 0.5 + perp[0] * offset,
                        (start[1] + end[1]) * 0.5 + perp[1] * offset,
                    ),
                    end,
                ]
            curve[0], curve[-1] = start, end
            preview_segment = AutoPathSegment(
                points=[PathPoint(float(x), float(y)) for x, y in curve],
                status=int(getattr(segment, "status", 0) or 0),
                corridor_id=int(getattr(segment, "corridor_id", -1) or -1),
                segment_type=segment_type,
                geometry_mode="turn",
            )
            preview_segment.length_m = self._segment_length(curve)
            result.append(preview_segment)
        return result

    def _sync_auto_path_from_editable_route(self):
        """Rebuild automatic path segments after a LabelMe-style manual route edit."""
        self._clear_machine_route_preview()
        st = self.state
        points = [tuple(map(float, point)) for point in getattr(st, 'path_points', [])]
        # DECISION-007/019: service markers are references to the shared route
        # endpoints after a manual edit.  Do this before rebuilding the
        # derived segments so redraw, panel state and later export agree.
        self._sync_service_points_to_route_control_endpoints(points)
        if len(points) < 2:
            st.auto_path = []
            st.auto_path_segments = []
            st.auto_path_geo = []
            st.path_geometry_modes = []
            st.auto_path_planned = False
            if self._route_edit_active:
                self._refresh_route_edit_info_panel()
            else:
                self.route_info.update_from_auto_path([])
            self._schedule_history_checkpoint()
            return
        geometry_modes = self._route_edge_geometry_modes(
            points,
            getattr(st, "path_status", []),
            getattr(st, "path_geometry_modes", []),
        )
        segments, statuses = self._control_route_segments(
            points,
            getattr(st, "path_status", []),
            geometry_modes,
        )
        st.path_status = statuses
        st.path_geometry_modes = geometry_modes
        simulation_segments = self._simulation_segments_from_control_route(segments)
        st.auto_path = simulation_segments
        st.auto_path_segments = simulation_segments
        control_full_path = [
            [(point.pixel_x, point.pixel_y) for point in segment.points]
            for segment in segments
        ]
        control_segment_types = [segment.segment_type for segment in segments]
        control_geo_points = []
        if self.geo.is_ready():
            from path_planner import global_path_to_geo
            control_geo_points = global_path_to_geo(
                control_full_path, self.geo, control_segment_types
            )
        simulation_full_path = [
            [(point.pixel_x, point.pixel_y) for point in segment.points]
            for segment in simulation_segments
        ]
        segment_types = [segment.segment_type for segment in simulation_segments]
        geo_points = []
        if self.geo.is_ready():
            from path_planner import global_path_to_geo
            geo_points = global_path_to_geo(
                simulation_full_path, self.geo, segment_types
            )
        st.auto_path_geo = geo_points
        st.auto_path_planned = True
        st.auto_path_valid = False
        st.auto_path_desc = "路线已手动编辑，作物掩膜几何覆盖率与碾压率需重新规划后评估"
        st.path_validation = {
            "valid": False,
            "route_geometry_source": "manual_control_edit",
            "issues": [
                "route was manually edited; rerun planning before scientific evaluation or formal export"
            ],
        }
        st.path_provenance = {}
        st.path_runtime = {}
        st.path_artifact = {}
        st.last_work_path_m = sum(item.length_m for item in simulation_segments if item.segment_type == "work")
        st.last_turn_path_m = sum(item.length_m for item in simulation_segments if item.segment_type.startswith("turn"))
        st.last_entry_exit_path_m = sum(item.length_m for item in simulation_segments if item.segment_type in ("entry", "exit"))
        st.last_total_path_m = st.last_work_path_m + st.last_turn_path_m + st.last_entry_exit_path_m
        st.path_validation.update(
            total_length_m=float(st.last_total_path_m),
            work_length_m=float(st.last_work_path_m),
            turn_length_m=float(st.last_turn_path_m),
            entry_exit_length_m=float(st.last_entry_exit_path_m),
        )
        st.simulation_done = False
        st.export_done = False
        path_result = {
            # The canvas, simulation and route information all use the curved
            # physical preview after an edit.  The compact 0/1 protocol chords
            # remain explicit side data for editing and machine export; they
            # must never replace the continuous route shown to the user.
            "full_path": simulation_full_path,
            "segment_types": segment_types,
            "segment_geometry_modes": [
                str(getattr(segment, "geometry_mode", "straight") or "straight")
                for segment in simulation_segments
            ],
            "segment_lengths_m": [
                float(segment.length_m) for segment in simulation_segments
            ],
            "geo_points": geo_points,
            "validation": dict(st.path_validation),
            "is_valid": False,
            "description": st.auto_path_desc,
            "simulation_full_path": simulation_full_path,
            "control_full_path": control_full_path,
            "control_segment_types": control_segment_types,
            "control_geo_points": control_geo_points,
        }
        pipeline_result = dict(self._pipeline_result or {})
        pipeline_result["path"] = self._normalise_path_result(path_result)
        self._pipeline_result = pipeline_result
        if self._route_edit_active:
            self._refresh_route_edit_info_panel()
        else:
            self.route_info.update_from_auto_path(st.auto_path)
        self._schedule_history_checkpoint()

    def _validate_manual_route(self):
        """DECISION-019: formally recompute metrics for current manual geometry.

        Manual validation uses the same footprint/path validator and semantic
        masks as automatic planning.  It certifies the current geometry only;
        automatic planning provenance remains cleared and the manual source is
        explicit in validation and manifest data.

        When validating manual routes, issues are separated into hard errors
        (田界越界、禁行区冲突、坐标异常) and soft warnings (碾压率略高、覆盖率略低).
        Only hard errors set auto_path_valid to False and block export.
        Soft warnings are displayed to the user but do not block operations.
        """
        self._sync_auto_path_from_editable_route()
        st = self.state
        mask_result = getattr(st, "mask_result", None)
        if not isinstance(mask_result, dict):
            raise ValueError("缺少当前掩膜优化结果，不能验证人工路线")
        processed_mask = mask_result.get("processed_mask")
        if not isinstance(processed_mask, np.ndarray):
            raise ValueError("缺少处理后作物掩膜，不能验证人工路线")
        if not self.geo.is_ready():
            raise ValueError("当前影像没有可靠地理配准，不能进行米制路线验证")
        simulation_segments = list(getattr(st, "auto_path", []) or [])
        if not simulation_segments:
            raise ValueError("当前没有人工路线")

        offset_x = float(getattr(st, "mask_offset_x", 0) or 0)
        offset_y = float(getattr(st, "mask_offset_y", 0) or 0)
        local_path = [
            [
                (
                    float(point.pixel_x) - offset_x,
                    float(point.pixel_y) - offset_y,
                )
                for point in list(getattr(segment, "points", []) or [])
            ]
            for segment in simulation_segments
        ]
        segment_types = [
            str(getattr(segment, "segment_type", "") or "work")
            for segment in simulation_segments
        ]
        if any(len(segment) < 2 for segment in local_path):
            raise ValueError("人工路线包含少于两个点的退化路径段")

        from path_planner import build_band_mask, validate_path
        from pyside6_app.workers import (
            path_planning_config_snapshot,
            planning_config_with_mask_evidence,
        )

        config = path_planning_config_snapshot()
        config = planning_config_with_mask_evidence(
            config, mask_result, st, processed_mask.shape
        )
        # Enable manual_edit_mode for severity classification
        config["manual_edit_mode"] = True
        wide_bands = list(mask_result.get("wide_bands") or [])
        planned_mask = build_band_mask(processed_mask.shape, wide_bands)
        detected_mask = (
            getattr(st, "inference_original_mask", None)
            if isinstance(getattr(st, "inference_original_mask", None), np.ndarray)
            else getattr(st, "mask_raw", None)
        )
        min_turn_radius_m = float(
            self._effective_harvester_params(
                getattr(st, "harvester_params", {}) or {}
            ).get("turn_radius_m", self.cfg.TURN_RADIUS_M)
        )
        validation = validate_path(
            local_path,
            processed_mask,
            float(mask_result.get("main_angle", 0.0) or 0.0),
            geo=self.geo,
            state=st,
            min_turn_radius_m=min_turn_radius_m,
            config=config,
            segment_types=segment_types,
            detected_mask=detected_mask,
            planned_mask=planned_mask,
        )
        minimum_observed_radius_m = self._manual_route_minimum_turn_radius_m(
            local_path, segment_types, processed_mask
        )
        validation["manual_minimum_observed_turn_radius_m"] = (
            minimum_observed_radius_m
        )
        validation["manual_required_turn_radius_m"] = min_turn_radius_m
        if (
            minimum_observed_radius_m is not None
            and minimum_observed_radius_m + 0.05 < min_turn_radius_m
        ):
            # Curvature violation is a hard error
            issues = list(validation.get("issues") or [])
            issues.append({
                "description": (
                    f"人工转弯局部曲率半径 {minimum_observed_radius_m:.2f}m "
                    f"< {min_turn_radius_m:.2f}m"
                ),
                "severity": "error"
            })
            validation["issues"] = issues
            validation["valid"] = False

        # Separate errors and warnings
        all_issues = validation.get("issues") or []
        errors = []
        warnings = []
        for issue in all_issues:
            if isinstance(issue, dict):
                if issue.get("severity") == "warning":
                    warnings.append(issue["description"])
                else:
                    errors.append(issue["description"])
            else:
                # Backward compatibility: treat string issues as errors
                errors.append(str(issue))

        # Only hard errors affect validity
        validation["valid"] = len(errors) == 0
        validation["errors"] = errors
        validation["warnings"] = warnings

        validation.update({
            "route_geometry_source": "manual_control_edit_validated",
            "manual_route_validated": True,
            "automatic_planning_provenance_valid": False,
            "current_validation_complete": True,
        })
        st.auto_path_valid = bool(validation.get("valid") is True)
        st.path_validation = dict(validation)
        st.auto_path_desc = (
            "人工路线已按当前掩膜、农机、田界和禁行区完成几何验证"
            if st.auto_path_valid
            else "人工路线存在需要处理的硬错误（田界越界/禁行区冲突/曲率不足）"
        )
        if warnings:
            if st.auto_path_valid:
                st.auto_path_desc += f"（{len(warnings)}个软提示：碾压率/覆盖率可优化）"
        st.path_provenance = {}
        st.path_runtime = {
            "route_geometry_source": "manual_control_edit_validated"
        }
        st.path_artifact = {
            "manual_route": {
                "schema": "align.manual-route.v1",
                "work_line_count": len(self._route_edit_work_records()),
                "control_point_count": len(getattr(st, "path_points", []) or []),
                "control_status": list(getattr(st, "path_status", []) or []),
                "control_geometry_modes": list(
                    getattr(st, "path_geometry_modes", []) or []
                ),
                "automatic_planning_provenance_valid": False,
            }
        }
        metric_state_fields = {
            "last_rolling_rate": "rolling_canopy_pct",
            "last_crop_mask_geometric_coverage_pct": "crop_mask_geometric_coverage_pct",
            "last_planned_target_geometric_coverage_pct": "planned_target_geometric_coverage_pct",
            "last_detected_mask_geometric_coverage_pct": "detected_mask_geometric_coverage_pct",
            "last_field_efficiency": "field_efficiency_pct",
            "last_total_path_m": "total_length_m",
            "last_work_path_m": "work_length_m",
            "last_turn_path_m": "turn_length_m",
            "last_entry_exit_path_m": "entry_exit_length_m",
        }
        for state_name, metric_name in metric_state_fields.items():
            value = validation.get(metric_name)
            if value is not None and math.isfinite(float(value)):
                setattr(st, state_name, float(value))

        path_result = dict((self._pipeline_result or {}).get("path") or {})
        path_result.update({
            "validation": dict(validation),
            "is_valid": bool(validation.get("valid") is True),
            "description": st.auto_path_desc,
            "manual_route_edited": True,
            "manual_route_validated": True,
            "manual_route": dict(st.path_artifact["manual_route"]),
        })
        pipeline_result = dict(self._pipeline_result or {})
        pipeline_result["path"] = self._normalise_path_result(path_result)
        self._pipeline_result = pipeline_result
        self.route_info.update_from_auto_path(st.auto_path)
        return validation

    def _manual_route_minimum_turn_radius_m(self, full_path, segment_types,
                                            processed_mask):
        """Estimate the smallest circumradius across manual turn geometry."""
        try:
            from row_geometry import meters_per_pixel
            mpp = float(meters_per_pixel(processed_mask, self.geo, self.state))
        except (TypeError, ValueError, RuntimeError):
            return None
        if not math.isfinite(mpp) or mpp <= 0.0:
            return None

        radii = []
        for index, (segment, segment_type) in enumerate(
            zip(full_path or [], segment_types or [])
        ):
            if not str(segment_type).startswith("turn") or len(segment) < 2:
                continue
            points = [tuple(map(float, point)) for point in segment]
            if index > 0 and len(full_path[index - 1]) >= 2:
                points.insert(0, tuple(map(float, full_path[index - 1][-2])))
            if index + 1 < len(full_path) and len(full_path[index + 1]) >= 2:
                points.append(tuple(map(float, full_path[index + 1][1])))
            compact = []
            for point in points:
                if not compact or math.hypot(
                    point[0] - compact[-1][0], point[1] - compact[-1][1]
                ) > 1e-7:
                    compact.append(point)
            for first, middle, last in zip(
                compact, compact[1:], compact[2:]
            ):
                a = math.hypot(
                    middle[0] - first[0], middle[1] - first[1]
                )
                b = math.hypot(last[0] - middle[0], last[1] - middle[1])
                c = math.hypot(last[0] - first[0], last[1] - first[1])
                double_area = abs(
                    (middle[0] - first[0]) * (last[1] - first[1])
                    - (middle[1] - first[1]) * (last[0] - first[0])
                )
                if min(a, b, c) <= 1e-7 or double_area <= 1e-9:
                    continue
                radius_px = (a * b * c) / (2.0 * double_area)
                if math.isfinite(radius_px) and radius_px > 0.0:
                    radii.append(radius_px * mpp)
        return float(min(radii)) if radii else None

    @staticmethod
    def _normalise_route_geometry_mode(value, fallback="straight"):
        value = str(value or "").strip().lower()
        if value in {"turn", "curve", "curved", "arc", "转弯", "曲线"}:
            return "turn"
        if value in {"straight", "line", "直线"}:
            return "straight"
        return "turn" if str(fallback) == "turn" else "straight"

    def _route_edge_geometry_modes(self, points, statuses=None, modes=None):
        """Return one physical geometry mode for every control edge."""
        edge_count = max(0, len(points or []) - 1)
        status_values = []
        for value in list(statuses or [])[:edge_count]:
            try:
                status_values.append(1 if int(value) == 1 else 0)
            except (TypeError, ValueError):
                status_values.append(0)
        if len(status_values) < edge_count:
            status_values.extend([1] * (edge_count - len(status_values)))
        raw_modes = list(modes or [])
        return [
            self._normalise_route_geometry_mode(
                raw_modes[index] if index < len(raw_modes) else None,
                fallback="straight" if status_values[index] == 1 else "turn",
            )
            for index in range(edge_count)
        ]

    @staticmethod
    def _route_status_values(points, statuses=None):
        """Return a normalized 0/1 value for every edge in ``points``."""
        edge_count = max(0, len(points or []) - 1)
        values = []
        for raw in list(statuses or [])[:edge_count]:
            try:
                values.append(1 if int(raw) == 1 else 0)
            except (TypeError, ValueError):
                values.append(0)
        if len(values) < edge_count:
            values.extend([1] * (edge_count - len(values)))
        return values

    def _route_segment_from_points(
        self, points, status, status_to_type, geometry_mode=None
    ):
        segment_type = status_to_type.get(int(status), "work")
        segment = AutoPathSegment(
            status=int(status),
            segment_type=segment_type,
            geometry_mode=self._normalise_route_geometry_mode(
                geometry_mode,
                fallback="straight" if int(status) == 1 else "turn",
            ),
        )
        segment.points = [PathPoint(float(x), float(y)) for x, y in points]
        segment.length_m = sum(self.geo.pixel_distance_m(points[index - 1], points[index]) if self.geo.is_ready() else math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]) for index in range(1, len(points)))
        return segment

    def _route_edit_work_records(self):
        """Return work-line objects derived from the shared 0/1 control route.

        DECISION-019 keeps one authoritative serialized control route, but the
        editor presents contiguous status-1 runs as first-class work lines.
        Edge and node indexes are retained so endpoint and whole-line edits can
        update the flat contract without maintaining a second hidden route.
        """
        points = [
            tuple(map(float, point))
            for point in (getattr(self.state, "path_points", []) or [])
        ]
        edge_count = max(0, len(points) - 1)
        statuses = []
        for value in list(getattr(self.state, "path_status", []) or [])[:edge_count]:
            try:
                statuses.append(1 if int(value) == 1 else 0)
            except (TypeError, ValueError):
                statuses.append(0)
        if len(statuses) < edge_count:
            statuses.extend([1] * (edge_count - len(statuses)))

        records = []
        edge_index = 0
        while edge_index < edge_count:
            if statuses[edge_index] != 1:
                edge_index += 1
                continue
            start_edge = edge_index
            work_points = [points[edge_index]]
            while edge_index < edge_count and statuses[edge_index] == 1:
                work_points.append(points[edge_index + 1])
                edge_index += 1
            records.append({
                "work_index": len(records),
                "edge_start": start_edge,
                "edge_end": edge_index - 1,
                "node_start": start_edge,
                "node_end": edge_index,
                "points": work_points,
            })
        return records

    def _route_edit_work_index_for_edge(self, edge_index):
        for record in self._route_edit_work_records():
            if record["edge_start"] <= edge_index <= record["edge_end"]:
                return int(record["work_index"])
        return -1

    def _route_edit_selected_work_index(self):
        selected_type = getattr(self.state, "selected_type", None)
        selected_raw = getattr(self.state, "selected_idx", -1)
        try:
            selected_index = int(selected_raw)
        except (TypeError, ValueError):
            selected_index = -1
        if selected_type == "edge":
            return self._route_edit_work_index_for_edge(selected_index)
        if selected_type == "node":
            records = self._route_edit_work_records()
            for record in records:
                if record["node_start"] <= selected_index <= record["node_end"]:
                    return int(record["work_index"])
        return -1

    @staticmethod
    def _route_edit_control_point_color(statuses, index):
        """Return a control-point color from the outgoing edge status.

        A junction belongs visually to the segment that follows it, so a
        work/turn boundary uses the color of the next edge.  The final node
        has no outgoing edge and therefore inherits the preceding edge color.
        Colors use the same BGR tuples as the route polylines.
        """
        work_color = (0, 255, 70)
        turn_color = (70, 120, 255)
        values = list(statuses or [])
        if not values:
            return work_color
        edge_index = int(index)
        if edge_index >= len(values):
            edge_index = len(values) - 1
        try:
            status = int(values[max(0, edge_index)])
        except (TypeError, ValueError):
            status = 1
        return work_color if status == 1 else turn_color

    def _route_edit_replace_work_lines(self, work_lines, selected_work_index=-1):
        """Replace route topology from ordered work lines and local turn chords."""
        points = []
        statuses = []
        geometry_modes = []
        for line in work_lines or []:
            line_points = [tuple(map(float, point)) for point in line or []]
            if len(line_points) < 2:
                continue
            if not points:
                points.extend(line_points)
                statuses.extend([1] * (len(line_points) - 1))
                geometry_modes.extend(["straight"] * (len(line_points) - 1))
                continue
            # A structural order change deliberately regenerates only the
            # local connector between consecutive work lines.  The two-point
            # status-0 chord is expanded to a physical curve for the canvas.
            points.append(line_points[0])
            statuses.append(0)
            geometry_modes.append("turn")
            for point in line_points[1:]:
                points.append(point)
                statuses.append(1)
                geometry_modes.append("straight")
        self.state.path_points = points
        self.state.path_status = statuses[: max(0, len(points) - 1)]
        self.state.path_geometry_modes = geometry_modes[: max(0, len(points) - 1)]
        self.state.selected_type = None
        self.state.selected_idx = -1
        records = self._route_edit_work_records()
        if 0 <= selected_work_index < len(records):
            record = records[selected_work_index]
            self.state.selected_type = "edge"
            self.state.selected_idx = int(record["edge_start"])
        self._sync_auto_path_from_editable_route()

    def _route_edit_add_work_line(self, start, end, after_work_index=None,
                                  save_undo=True):
        start = (float(start[0]), float(start[1]))
        end = (float(end[0]), float(end[1]))
        if math.hypot(end[0] - start[0], end[1] - start[1]) <= 1e-6:
            return -1
        records = self._route_edit_work_records()
        lines = [list(record["points"]) for record in records]
        if after_work_index is None:
            selected = self._route_edit_selected_work_index()
            after_work_index = selected if selected >= 0 else len(lines) - 1
        insert_index = max(0, min(len(lines), int(after_work_index) + 1))
        if save_undo:
            self.state.save_route_undo("新增作业线")
        lines.insert(insert_index, [start, end])
        self._route_edit_replace_work_lines(lines, insert_index)
        return insert_index

    def _route_edit_delete_work_line(self, work_index, save_undo=True):
        records = self._route_edit_work_records()
        if not (0 <= int(work_index) < len(records)):
            return False
        if save_undo:
            self.state.save_route_undo("删除作业线")
        lines = [list(record["points"]) for record in records]
        lines.pop(int(work_index))
        selected = min(int(work_index), len(lines) - 1)
        self._route_edit_replace_work_lines(lines, selected)
        return True

    def _route_edit_translate_work_line(self, work_index, delta,
                                        save_undo=True):
        records = self._route_edit_work_records()
        if not (0 <= int(work_index) < len(records)):
            return False
        if save_undo:
            self.state.save_route_undo("整条作业线平移")
        dx, dy = float(delta[0]), float(delta[1])
        points = list(getattr(self.state, "path_points", []) or [])
        record = records[int(work_index)]
        for node_index in range(record["node_start"], record["node_end"] + 1):
            point = points[node_index]
            points[node_index] = (float(point[0]) + dx, float(point[1]) + dy)
        self.state.path_points = points
        self.state.selected_type = "edge"
        self.state.selected_idx = int(record["edge_start"])
        if save_undo:
            self._sync_auto_path_from_editable_route()
        return True

    def _route_edit_reverse_work_line(self, work_index, save_undo=True):
        records = self._route_edit_work_records()
        if not (0 <= int(work_index) < len(records)):
            return False
        if save_undo:
            self.state.save_route_undo("作业线反向")
        lines = [list(record["points"]) for record in records]
        lines[int(work_index)] = list(reversed(lines[int(work_index)]))
        self._route_edit_replace_work_lines(lines, int(work_index))
        return True

    def _route_edit_reorder_work_line(self, work_index, offset, save_undo=True):
        records = self._route_edit_work_records()
        source = int(work_index)
        target = source + int(offset)
        if not (0 <= source < len(records) and 0 <= target < len(records)):
            return source if 0 <= source < len(records) else -1
        if save_undo:
            self.state.save_route_undo("调整作业顺序")
        lines = [list(record["points"]) for record in records]
        line = lines.pop(source)
        lines.insert(target, line)
        self._route_edit_replace_work_lines(lines, target)
        return target

    def _route_edit_set_edge_status(self, edge_index, status, save_undo=True):
        points = list(getattr(self.state, "path_points", []) or [])
        if not (0 <= int(edge_index) < len(points) - 1):
            return False
        if save_undo:
            self.state.save_route_undo(
                "设为作业边" if int(status) == 1 else "设为转弯边"
            )
        statuses = list(getattr(self.state, "path_status", []) or [])
        if len(statuses) < len(points) - 1:
            statuses.extend([1] * (len(points) - 1 - len(statuses)))
        geometry_modes = self._route_edge_geometry_modes(
            points,
            statuses,
            getattr(self.state, "path_geometry_modes", []) or [],
        )
        statuses[int(edge_index)] = 1 if int(status) == 1 else 0
        geometry_modes[int(edge_index)] = (
            "straight" if int(status) == 1 else "turn"
        )
        self.state.path_status = statuses[: len(points) - 1]
        self.state.path_geometry_modes = geometry_modes[: len(points) - 1]
        self.state.selected_type = "edge"
        self.state.selected_idx = int(edge_index)
        self._sync_auto_path_from_editable_route()
        return True

    def _route_edit_nearest_node(self, point, threshold=12.0):
        # Mouse coordinates arrive in scene/image units.  Convert the desired
        # screen-pixel hit radius back to scene units so endpoint dragging is
        # still practical when a 20k x 29k orthomosaic is fitted to the window.
        try:
            transform = self.image_view.viewer.transform()
            view_scale = math.hypot(
                float(transform.m11()), float(transform.m12())
            )
            if math.isfinite(view_scale) and view_scale > 1e-6:
                threshold = float(threshold) / view_scale
        except (AttributeError, TypeError, ValueError):
            threshold = float(threshold)
        best_idx = -1
        best_dist = float("inf")
        for idx, route_point in enumerate(getattr(self.state, 'path_points', []) or []):
            dist = math.hypot(float(route_point[0]) - float(point[0]), float(route_point[1]) - float(point[1]))
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx if best_dist <= threshold else -1

    def _route_edit_nearest_edge(self, point, threshold=10.0):
        try:
            transform = self.image_view.viewer.transform()
            view_scale = math.hypot(
                float(transform.m11()), float(transform.m12())
            )
            if math.isfinite(view_scale) and view_scale > 1e-6:
                threshold = float(threshold) / view_scale
        except (AttributeError, TypeError, ValueError):
            threshold = float(threshold)
        points = list(getattr(self.state, "path_points", []) or [])
        best_index = -1
        best_distance = float("inf")
        for index in range(len(points) - 1):
            projected = self._project_point_to_segment(
                point, points[index], points[index + 1]
            )
            distance = math.hypot(
                float(point[0]) - projected[0],
                float(point[1]) - projected[1],
            )
            if distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index if best_distance <= threshold else -1

    def _route_edit_insert_point(self, point, save_undo=True,
                                 preserve_position=False, edge_index=None):
        points = list(getattr(self.state, 'path_points', []) or [])
        if save_undo:
            self.state.save_route_undo("路线插点")
        point = (float(point[0]), float(point[1]))
        if len(points) < 2:
            points.append(point)
            self.state.path_points = points
            if len(points) >= 2 and not self.state.path_status:
                self.state.path_status = [1]
                self.state.path_geometry_modes = ["straight"]
            return len(points) - 1
        old_points = list(points)
        old_statuses = self._route_status_values(
            old_points, getattr(self.state, "path_status", []) or []
        )
        old_geometry_modes = self._route_edge_geometry_modes(
            old_points,
            old_statuses,
            getattr(self.state, "path_geometry_modes", []) or [],
        )
        best_index = 0
        best_distance = float("inf")
        best_point = point
        candidate_indices = range(len(points) - 1)
        if edge_index is not None and 0 <= int(edge_index) < len(points) - 1:
            candidate_indices = [int(edge_index)]
        for index in candidate_indices:
            projected = self._project_point_to_segment(point, points[index], points[index + 1])
            distance = math.hypot(point[0] - projected[0], point[1] - projected[1])
            if distance < best_distance:
                best_index = index
                best_distance = distance
                best_point = projected
        insert_at = best_index + 1
        points.insert(insert_at, point if preserve_position else best_point)
        edge_status = old_statuses[best_index]
        statuses = list(old_statuses)
        statuses.insert(insert_at, edge_status)
        geometry_modes = list(old_geometry_modes)
        edge_geometry_mode = old_geometry_modes[best_index]
        geometry_modes.insert(insert_at, edge_geometry_mode)
        self.state.path_points = points
        self.state.path_status = statuses[:len(points) - 1]
        self.state.path_geometry_modes = geometry_modes[:len(points) - 1]
        return insert_at

    def _route_edit_append_point(self, point, save_undo=True):
        """Append a free canvas point as the next vertex of the ordered route."""
        point = (float(point[0]), float(point[1]))
        points = list(getattr(self.state, "path_points", []) or [])
        if save_undo:
            self.state.save_route_undo("路线末端新增点")
        if not points:
            points.append(point)
            self.state.path_points = points
            self.state.path_status = []
            self.state.path_geometry_modes = []
            return 0
        statuses = list(getattr(self.state, "path_status", []) or [])
        modes = self._route_edge_geometry_modes(
            points,
            statuses,
            getattr(self.state, "path_geometry_modes", []) or [],
        )
        points.append(point)
        statuses.append(1)
        modes.append("straight")
        self.state.path_points = points
        self.state.path_status = statuses[: len(points) - 1]
        self.state.path_geometry_modes = modes[: len(points) - 1]
        return len(points) - 1

    def _route_edit_delete_point(self, index, save_undo=True):
        points = list(getattr(self.state, 'path_points', []) or [])
        if not (0 <= index < len(points)) or len(points) <= 2:
            return False
        if save_undo:
            self.state.save_route_undo("删除路线点")
        geometry_modes = self._route_edge_geometry_modes(
            points,
            getattr(self.state, "path_status", []) or [],
            getattr(self.state, "path_geometry_modes", []) or [],
        )
        points.pop(index)
        statuses = list(getattr(self.state, 'path_status', []) or [])
        if index < len(statuses):
            statuses.pop(index)
        elif statuses:
            statuses.pop()
        if index < len(geometry_modes):
            geometry_modes.pop(index)
        elif geometry_modes:
            geometry_modes.pop()
        self.state.path_points = points
        self.state.path_status = statuses[:len(points) - 1]
        self.state.path_geometry_modes = geometry_modes[:len(points) - 1]
        # Deleting the first/final shared node moves the corresponding service
        # reference to the new first/final node; it must not remain detached
        # from the edited route.
        self._sync_service_points_to_route_control_endpoints(points)
        self.state.selected_type = None
        self.state.selected_idx = -1
        return True

    def _route_edit_move_point(self, index, point, save_undo=True):
        points = list(getattr(self.state, 'path_points', []) or [])
        if not (0 <= index < len(points)):
            return False
        if save_undo:
            self.state.save_route_undo("移动路线点")
        points[index] = (float(point[0]), float(point[1]))
        self.state.path_points = points
        self.state.selected_type = 'node'
        self.state.selected_idx = index
        return True

    @staticmethod
    def _project_point_to_segment(point, start, end):
        px, py = float(point[0]), float(point[1])
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return ax, ay
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        return ax + t * dx, ay + t * dy

    def _handle_route_edit_press(self, x, y, button):
        if button == 2:
            self._route_new_line_start = None
            self._route_drag_work_index = -1
            self._draw_route_edit_overlay()
            self.lbl_status.setText("已取消当前人工绘制操作")
            return True
        if button != 1:
            return False
        point = (float(x), float(y))
        # Existing control points always take precedence over creation tools.
        # The viewer has already captured this press as a direct drag, so an
        # add-point/add-work branch must not insert a duplicate before moving it.
        existing_node = self._route_edit_nearest_node(point)
        if existing_node >= 0:
            self.state.save_route_undo("移动路线点")
            self._route_drag_idx = existing_node
            self.state.selected_type = "node"
            self.state.selected_idx = existing_node
            return True
        modifiers = QGuiApplication.keyboardModifiers()
        if self._route_edit_mode == "add_work":
            if self._route_new_line_start is None:
                self._route_new_line_start = point
                self.lbl_status.setText("已确定作业线起点，请点击终点；右键取消")
            else:
                inserted = self._route_edit_add_work_line(
                    self._route_new_line_start, point
                )
                self._route_new_line_start = None
                if inserted >= 0:
                    self.lbl_status.setText(
                        f"已新增第 {inserted + 1} 条作业线，可继续绘制下一条"
                    )
                else:
                    self.lbl_status.setText("作业线过短，请重新点击起点和终点")
            self._draw_route_edit_overlay()
            return True
        if self._route_edit_mode == "add_point":
            statuses = list(getattr(self.state, "path_status", []) or [])
            selected_edge = (
                int(getattr(self.state, "selected_idx", -1))
                if getattr(self.state, "selected_type", None) == "edge"
                else -1
            )
            nearest_edge = self._route_edit_nearest_edge(point)
            target_edge = (
                selected_edge
                if 0 <= selected_edge < len(statuses)
                else nearest_edge
            )
            if not 0 <= target_edge < len(statuses):
                points = list(getattr(self.state, "path_points", []) or [])
                selected_node = (
                    int(getattr(self.state, "selected_idx", -1))
                    if getattr(self.state, "selected_type", None) == "node"
                    else -1
                )
                if 0 <= selected_node < len(points) - 1:
                    index = self._route_edit_insert_point(
                        point,
                        preserve_position=True,
                        edge_index=selected_node,
                    )
                    message = "已在选中点之后添加控制点，连接模式保持不变"
                else:
                    index = self._route_edit_append_point(point)
                    message = "已在路线末端添加控制点，新增连接默认为直线"
            else:
                index = self._route_edit_insert_point(
                    point, preserve_position=True, edge_index=target_edge
                )
                message = "已按点击位置添加控制点，线段模式保持不变"
            self.state.selected_type = "node"
            self.state.selected_idx = index
            self._sync_auto_path_from_editable_route()
            self._draw_route_edit_overlay()
            self.lbl_status.setText(message)
            return True
        if modifiers & Qt.AltModifier:
            nearest_edge = self._route_edit_nearest_edge(point)
            if nearest_edge < 0:
                # DECISION-019: an empty canvas click is a selection command,
                # not an implicit route insertion at the first edge.
                self.state.selected_type = None
                self.state.selected_idx = -1
                self._route_drag_idx = -1
                self._route_drag_work_index = -1
                self._draw_route_edit_overlay()
                self.lbl_status.setText("已取消当前选择")
                return True
            statuses = list(getattr(self.state, "path_status", []) or [])
            selected_edge = (
                int(getattr(self.state, "selected_idx", -1))
                if getattr(self.state, "selected_type", None) == "edge"
                else -1
            )
            target_edge = (
                selected_edge
                if 0 <= selected_edge < len(statuses)
                and int(statuses[selected_edge]) == 0
                else nearest_edge
            )
            preserve_position = bool(
                0 <= target_edge < len(statuses)
                and int(statuses[target_edge]) == 0
            )
            index = self._route_edit_insert_point(
                point,
                preserve_position=preserve_position,
                edge_index=target_edge if preserve_position else None,
            )
            self.state.selected_type = 'node'
            self.state.selected_idx = index
            self._sync_auto_path_from_editable_route()
            self._draw_route_edit_overlay()
            return True
        edge_index = self._route_edit_nearest_edge(point)
        idx = self._route_edit_nearest_node(point)
        if idx >= 0:
            self.state.save_route_undo("移动路线点")
            self._route_drag_idx = idx
            self.state.selected_type = 'node'
            self.state.selected_idx = idx
            return True
        work_index = self._route_edit_work_index_for_edge(edge_index)
        if work_index >= 0:
            self.state.save_route_undo("整条作业线平移")
            self._route_drag_work_index = work_index
            self._route_drag_work_origin = point
            self._route_drag_work_points = list(
                getattr(self.state, "path_points", []) or []
            )
            record = self._route_edit_work_records()[work_index]
            self.state.selected_type = "edge"
            self.state.selected_idx = int(record["edge_start"])
            return True
        if edge_index >= 0:
            self.state.selected_type = "edge"
            self.state.selected_idx = int(edge_index)
            self._draw_route_edit_overlay()
            return True
        # A left click on empty canvas clears the current node/edge selection;
        # it must not leave the previous line highlighted indefinitely.
        self.state.selected_type = None
        self.state.selected_idx = -1
        self._route_drag_idx = -1
        self._route_drag_work_index = -1
        self._draw_route_edit_overlay()
        self.lbl_status.setText("已取消当前选择")
        return True

    def _draw_route_edit_overlay(self):
        viewer = self.image_view.viewer
        # DECISION-019: route editing is additive.  Clear only the persistent
        # display layers that are about to be rebuilt and keep the base TIF
        # plus display preferences intact.
        self._clear_display_overlay_layers(include_route=True)
        viewer.clear_overlay_layer("route_edit")
        self._route_edit_dynamic_items = []
        if self._tif_rgb is not None and getattr(viewer, "_pixmap", None) is None:
            self.image_view.set_image(self._tif_rgb)
        self._sync_display_layer_availability()
        self._draw_static_display_layers(viewer)
        self._draw_route_edit_dynamic_overlay(lightweight=False)

    def _draw_route_edit_dynamic_overlay(self, *, lightweight=False):
        """Replace only the route graphics while a control point is dragged."""
        viewer = self.image_view.viewer
        viewer.remove_overlays(self._route_edit_dynamic_items)
        self._route_edit_dynamic_items = []
        viewer.begin_overlay_layer("route_edit")
        overlay_start = len(viewer._overlay_group)
        points = list(getattr(self.state, 'path_points', []) or [])
        statuses = list(getattr(self.state, 'path_status', []) or [])
        control_segments, _normalised_statuses = self._control_route_segments(
            points,
            statuses,
            getattr(self.state, "path_geometry_modes", []),
        )
        preview_segments = self._simulation_segments_from_control_route(
            control_segments
        )
        for segment in preview_segments:
            segment_points = [
                (float(point.pixel_x), float(point.pixel_y))
                for point in list(getattr(segment, "points", []) or [])
            ]
            if len(segment_points) < 2:
                continue
            is_work = str(
                getattr(segment, "segment_type", "") or "work"
            ) == "work"
            viewer.draw_polyline(
                segment_points,
                color=(0, 255, 70) if is_work else (70, 120, 255),
                width=4 if is_work else 3,
            )
        selected_type = getattr(self.state, "selected_type", None)
        selected_work_index = self._route_edit_selected_work_index()
        work_records = self._route_edit_work_records()
        if selected_type == "edge" and 0 <= selected_work_index < len(work_records):
            selected_points = work_records[selected_work_index]["points"]
            viewer.draw_polyline(
                selected_points,
                color=(255, 255, 255),
                width=7,
            )
        elif selected_type == "edge":
            edge_index = int(getattr(self.state, "selected_idx", -1))
            selected_record = next(
                (
                    record for record in self._route_edit_segment_records()
                    if int(record["edge_start"]) <= edge_index
                    <= int(record["edge_end"])
                ),
                None,
            )
            if selected_record is not None:
                node_start = int(selected_record["node_start"])
                node_end = int(selected_record["node_end"])
                viewer.draw_polyline(
                    points[node_start:node_end + 1],
                    color=(255, 255, 255),
                    width=6,
                )
        # Work-line labels such as “作业1/作业2” obscure the crop evidence on
        # dense fields.  The segment list already carries the work-line index;
        # keep the canvas focused on geometry and control points.
        for index, point in enumerate(points):
            selected = selected_type == "node" and getattr(
                self.state, "selected_idx", -1
            ) == index
            point_color = self._route_edit_control_point_color(statuses, index)
            if selected:
                # Keep the segment color visible while using a white halo to
                # distinguish the active node.  Selecting/inserting a node no
                # longer turns its entire parent work line white.
                viewer.draw_circle(
                    point[0], point[1], radius=9, color=(255, 255, 255)
                )
                viewer.draw_circle(
                    point[0], point[1], radius=6, color=point_color
                )
            else:
                viewer.draw_circle(
                    point[0], point[1], radius=6, color=point_color
                )
            if not lightweight and len(points) <= 30:
                viewer.draw_text(point[0] + 8, point[1] - 8, str(index + 1), color=(245, 245, 245), size=10)
        if not lightweight:
            self.route_info.set_edit_selection(
                getattr(self.state, "selected_type", None),
                getattr(self.state, "selected_idx", -1),
            )
        if self._route_new_line_start is not None:
            viewer.draw_circle(
                self._route_new_line_start[0],
                self._route_new_line_start[1],
                radius=9,
                color=(0, 255, 255),
            )
            viewer.draw_text(
                self._route_new_line_start[0] + 10,
                self._route_new_line_start[1] - 10,
                "新作业线起点",
                color=(0, 255, 255),
                size=10,
            )
        self._route_edit_dynamic_items = list(
            viewer._overlay_group[overlay_start:]
        )
        viewer.end_overlay_layer()

    def _update_route_edit_cursor(self, x, y):
        if not self._route_edit_active:
            return
        viewer = self.image_view.viewer
        if self._route_drag_idx >= 0 or self._route_drag_work_index >= 0:
            viewer.setCursor(Qt.ClosedHandCursor)
        elif self._route_edit_mode in {"add_work", "add_point"}:
            viewer.setCursor(Qt.CrossCursor)
        elif self._route_edit_nearest_node((x, y)) >= 0:
            viewer.setCursor(Qt.SizeAllCursor)
        elif self._route_edit_nearest_edge((x, y)) >= 0:
            viewer.setCursor(Qt.PointingHandCursor)
        else:
            viewer.setCursor(Qt.ArrowCursor)

    def _sync_drawing_mode(self):
        active = (
            self._field_drawing
            or self._forbidden_drawing
            or self._reference_drawing
            or self._calibration_drawing
            or bool(self._measure_tool)
            or getattr(self.state, 'entry_exit_mode', False)
            or self._route_edit_active
        )
        self.image_view.viewer.drawing_mode = active
        if self._route_edit_active and not (
            self._field_drawing or self._forbidden_drawing or self._measure_tool
            or getattr(self.state, 'entry_exit_mode', False)
        ):
            cursor = (
                Qt.CrossCursor
                if self._route_edit_mode in {"add_work", "add_point"}
                else Qt.ArrowCursor
            )
        else:
            cursor = Qt.CrossCursor if active else Qt.ArrowCursor
        self.image_view.viewer.setCursor(cursor)
        if self._field_drawing:
            self.lbl_status.setText("请先完成影像导入和田块圈选")
        elif self._forbidden_drawing:
            self.lbl_status.setText("禁行区绘制：点击首点闭合，右键取消")
        elif self._reference_drawing:
            self.lbl_status.setText(
                "参考线标注（第 %d 遍）：已加 %d 点，双击或菜单“完成当前参考线”保存，右键撤销上一个点"
                % (self._reference_pass, len(self._reference_pts))
            )
        elif self._calibration_drawing:
            self.lbl_status.setText(
                "作业线标定：拖动整条平移，Shift+拖动只纠局部弯曲，右键撤销上一步"
            )
        elif self._measure_tool == "point":
            self.lbl_status.setText("点选测量：左键选择点，下一次左键替换")
        elif self._measure_tool == "line":
            self.lbl_status.setText("测距：左键选两个点，右键显示长度；下一次左键新建")
        elif self._measure_tool == "area":
            self.lbl_status.setText("测面积：左键添加顶点，右键显示面积；下一次左键新建")
        elif getattr(self.state, 'entry_exit_mode', False):
            self.lbl_status.setText("服务点编辑：起点必选，终点自动延伸且可拖动，卸粮点按需添加")
        elif self._route_edit_active:
            labels = {
                "select": "路线编辑：选择线段或拖动端点；Shift+拖动作业线可整线平移",
                "add_work": "新增作业线：依次点击起点和终点；右键取消当前起点",
                "add_point": "添加控制点：先选择任意线段，再点击新的控制点位置",
            }
            self.lbl_status.setText(
                labels.get(self._route_edit_mode, labels["select"])
            )
        else:
            self.lbl_status.setText("就绪")
    def _refresh_state(self):
        self._release_finished_inference_runner()
        self.top_toolbar.refresh_actions(self.state)
        mask_result = getattr(self.state, "mask_result", None)
        raw_mask = getattr(self.state, "mask_raw", None)
        self.task_panel.set_inference_diagnostic_available(
            bool(
                getattr(self.state, "inference_done", False)
                and isinstance(raw_mask, np.ndarray)
                and raw_mask.ndim == 2
            )
        )
        self.task_panel.set_mask_diagnostic_available(
            bool(
                getattr(self.state, "mask_processed", False)
                and isinstance(mask_result, dict)
                and isinstance(mask_result.get("processed_mask"), np.ndarray)
            )
        )
        self.task_panel.set_forbidden_state(
            len(getattr(self.state, "forbidden_regions", []) or []),
            bool(getattr(self.state, "forbidden_regions_confirmed", False)),
        )
        if (self._pending_segment_display
                and getattr(self.state, 'inference_done', False)
                and not getattr(self.state, 'inference_running', True)):
            self._pending_segment_display = False
            raw_mask = getattr(self.state, 'mask_raw', None)
            if raw_mask is not None:
                self._show_mask_overlay(raw_mask)
                WorkflowUpdater.advance(self.state, "INFERENCE_DONE")
            self._sync_task_statuses()
            self.top_toolbar.refresh_actions(self.state)
            self.progress_bar.hide()
            self.top_toolbar.set_progress(0)
            self.lbl_status.setText("作物识别完成，请继续掩膜优化")
            self.log_panel.success("作物识别完成，请继续掩膜优化")
        elif (self._pending_segment_display
              and not getattr(self.state, 'inference_running', True)):
            self._pending_segment_display = False
            msg = getattr(self.state, 'status_message', '') or "作物识别中"
            self._sync_task_statuses()
            self.progress_bar.hide()
            self.top_toolbar.set_progress(0)
            self.lbl_status.setText(msg)
            self.log_panel.error(msg)

        if getattr(self.state, 'inference_running', False):
            pct = getattr(self.state, 'inference_progress', 0.0)
            pct_int = int(pct * 100)
            self.progress_bar.setValue(pct_int)
            self.task_panel.set_task_status("segment", "in_progress", pct_int)
            self.lbl_status.setText(getattr(self.state, 'status_message', '作物识别中...'))

    def _process_pending(self):
        if self.event_bridge.check_pending():
            self.event_bridge.handle_pending_action()

    def keyPressEvent(self, event):
        if self._route_edit_active:
            modifiers = event.modifiers()
            key = event.key()
            if modifiers & Qt.ControlModifier and key in (Qt.Key_Z, Qt.Key_Y):
                self._on_route_edit_tool_requested("undo" if key == Qt.Key_Z else "redo")
                event.accept()
                return
            if modifiers & Qt.ControlModifier and key in (Qt.Key_Up, Qt.Key_Down):
                self._on_route_edit_tool_requested(
                    "move_earlier" if key == Qt.Key_Up else "move_later"
                )
                event.accept()
                return
            if not (modifiers & Qt.ControlModifier) and key == Qt.Key_R:
                self._on_route_edit_tool_requested("reverse")
                event.accept()
                return
        if self._route_edit_active and event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            idx = getattr(self.state, 'selected_idx', -1)
            selected_type = getattr(self.state, 'selected_type', None)
            deleted = False
            if selected_type == "edge":
                deleted = self._route_edit_delete_work_line(
                    self._route_edit_selected_work_index()
                )
            elif selected_type == 'node':
                deleted = self._route_edit_delete_point(idx)
            if deleted:
                self._sync_auto_path_from_editable_route()
                self._draw_route_edit_overlay()
                event.accept()
                return
        if self._route_edit_active and event.key() == Qt.Key_Escape:
            self._route_new_line_start = None
            self._route_edit_mode = "select"
            self.task_panel.set_route_edit_mode("select")
            self._draw_route_edit_overlay()
            event.accept()
            return
        if self._route_edit_active and event.key() in (Qt.Key_0, Qt.Key_1):
            if getattr(self.state, "selected_type", None) == "edge":
                status = 1 if event.key() == Qt.Key_1 else 0
                if self._route_edit_set_edge_status(
                    getattr(self.state, "selected_idx", -1), status
                ):
                    self._draw_route_edit_overlay()
                    event.accept()
                    return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        history_timer = getattr(self, "_history_checkpoint_timer", None)
        if history_timer is not None:
            history_timer.stop()
        if self._anim_timer:
            self._anim_timer.stop()
        if self._worker_thread and self._worker_thread.isRunning():
            event.ignore()
            first_close_request = not getattr(self, '_close_after_worker_cancel', False)
            self._close_after_worker_cancel = True
            if first_close_request:
                self._close_cancel_started_at = time.time()
                self._close_cancel_warning_shown = False
                QTimer.singleShot(10000, self._warn_close_cancel_still_running)
                # Keep the worker alive long enough to release GDAL/PyTorch
                # resources safely. Keep the window visible so cancellation
                # progress remains observable to the user.
            if not getattr(self, '_task_cancel_requested', False):
                self._cancel_background_worker()
            elif (
                not getattr(self, '_close_cancel_warning_shown', False)
                and time.time() - getattr(self, '_close_cancel_started_at', time.time()) >= 10.0
            ):
                self._warn_close_cancel_still_running()
            if not getattr(self, '_close_cancel_warning_shown', False):
                self.lbl_status.setText("正在取消后台任务，完成后自动关闭...")
            return
        if not self._wait_processed_mask_saves(timeout_s=8.0):
            self._log.info("processed mask cache save still running during shutdown")
        if self._tif_path:
            try:
                from cache import save_project_state, wait_for_project_state_saves
                self.state.safe_update(
                    inference_done=bool(getattr(self.state, 'inference_done', False)),
                    mask_processed=bool(getattr(self.state, 'mask_processed', False)),
                    auto_path_planned=bool(getattr(self.state, 'auto_path_planned', False)),
                )
                save_project_state(self._tif_path, self.state)
                wait_for_project_state_saves()
            except Exception:
                pass
        self._state_timer.stop()
        self._pending_timer.stop()
        self._log.info("program exit")
        super().closeEvent(event)
