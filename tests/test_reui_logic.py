import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from affine import Affine
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QPushButton

from pyside6_app.main_window import MainWindow
from pyside6_app.image_view import MetricScaleBar


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.state.harvester_params["cutter_forward_offset_m"] = 1.2
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    return window


def test_snapshot_save_uses_next_number_in_selected_folder(monkeypatch, tmp_path):
    window = _window()
    (tmp_path / "snapshot.png").write_bytes(b"old-default")
    (tmp_path / "快照1.png").write_bytes(b"one")
    (tmp_path / "快照2.jpg").write_bytes(b"two")
    saved = []

    class FakeGrab:
        def save(self, path):
            target = Path(path)
            target.write_bytes(b"snapshot")
            saved.append(target)
            return True

    def reject_filename_dialog(*_args, **_kwargs):
        raise AssertionError("snapshot saving must select a folder, not overwrite a file")

    monkeypatch.setattr(
        "pyside6_app.main_window.QFileDialog.getSaveFileName",
        reject_filename_dialog,
    )
    monkeypatch.setattr(
        "pyside6_app.main_window.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(window.image_view.viewer, "grab", lambda: FakeGrab())

    window._on_save_snapshot()
    window._on_save_snapshot()

    assert saved == [tmp_path / "快照3.png", tmp_path / "快照4.png"]
    assert window._snapshot_directory == str(tmp_path)
    window.close()


class _FixedMetricGeo:
    @staticmethod
    def is_ready():
        return True

    @staticmethod
    def meters_per_pixel(_x, _y):
        return 0.01


class _FixedMetricState:
    mask_offset_x = 0
    mask_offset_y = 0
    field_boundary = []


def test_task_export_action_routes_to_numbered_machine_txt(monkeypatch):
    calls = []
    monkeypatch.setattr(
        MainWindow,
        "_on_export_machine_compatibility_path",
        lambda self: calls.append("compatibility"),
    )
    window = _window()

    window.task_panel.export_machine_compatibility_requested.emit()

    assert calls == ["compatibility"]


def test_inference_runner_participates_in_background_task_exclusion():
    window = _window()
    window.state.inference_running = True

    assert window._has_background_worker() is True

    window.state.inference_running = False
    assert window._has_background_worker() is False


def test_segment_does_not_start_while_mask_or_plan_thread_is_running():
    window = _window()
    messages = []

    class RunningThread:
        def isRunning(self):
            return True

    window._worker_thread = RunningThread()
    window.log_panel.warn = messages.append

    window._on_segment()

    assert messages == ["已有后台任务正在运行，请等待完成后再开始识别"]


def test_task_error_dialog_keeps_raw_traceback_in_log_only():
    window = _window()

    title, message = window._task_error_message("Traceback: SECRET_INTERNAL_DETAIL")

    assert title == "任务失败"
    assert "SECRET_INTERNAL_DETAIL" not in message
    assert "项目日志" in message


def test_path_result_is_normalised_and_stored_for_current_ui():
    window = _window()
    path_result = {
        "full_path": [
            [(0, 0), (10, 0)],
            [(10, 0), (10, 5)],
            [(10, 5), (0, 5)],
        ],
        "segment_types": ["work", "turn", "work"],
        "geo_points": [{"segment_index": 0, "point_index": 0, "x": 0, "y": 0}],
        "validation": {
            "valid": True,
            "total_length_m": 15,
            "work_length_m": 10,
            "turn_length_m": 5,
            "entry_exit_length_m": 0,
            "rolling_canopy_pct": 2.5,
            "crop_mask_geometric_coverage_pct": 94.0,
            "planned_target_geometric_coverage_pct": 96.0,
            "detected_mask_geometric_coverage_pct": 91.0,
            "field_efficiency_pct": 66.7,
        },
        "is_valid": True,
        "description": "ok",
    }

    normalised = window._store_path_result(path_result)

    assert len(normalised["tracks"]) == 2
    assert len(normalised["turns"]) == 1
    assert window.state.auto_path_planned is True
    assert len(window.state.auto_path) == 3
    assert len(window.state.path_points) == 4
    assert window._pipeline_result["path"]["tracks"]
    assert all(hasattr(p, "pixel_x") for p in window.state.auto_path[0].points)
    assert window._build_anim_segments(normalised, window.state.auto_path_segments) is window.state.auto_path_segments
    assert window.state.last_rolling_rate == 2.5
    assert window.state.last_planned_target_geometric_coverage_pct == 96.0
    assert window.state.last_total_path_m == 15.0


def test_path_overlay_keeps_the_current_base_image(monkeypatch):
    """The live route overlay must not replace/reload the orthomosaic."""
    window = _window()
    set_image_calls = []
    monkeypatch.setattr(
        window.image_view,
        "set_image",
        lambda *args, **kwargs: set_image_calls.append((args, kwargs)),
    )

    window._show_path(
        {
            "full_path": [[(1.0, 2.0), (18.0, 2.0)]],
            "segment_types": ["work"],
        }
    )

    assert set_image_calls == []
    window.close()


def test_route_layer_can_be_removed_without_removing_mask_evidence():
    """Live route edits must leave the base and evidence items in the scene."""
    window = _window()
    viewer = window.image_view.viewer
    viewer.begin_overlay_layer("mask_evidence")
    mask_item = viewer.draw_line(2, 2, 20, 2, color=(120, 255, 140))
    viewer.end_overlay_layer()
    viewer.begin_overlay_layer("route")
    route_item = viewer.draw_line(2, 8, 20, 8, color=(0, 255, 70))
    viewer.end_overlay_layer()

    viewer.clear_overlay_layer("route")

    assert route_item.scene() is None
    assert mask_item.scene() is viewer._scene
    assert viewer.has_overlay_layer("mask_evidence") is True
    window.close()


def test_named_overlay_visibility_can_be_toggled_independently():
    """Display controls must hide one overlay without changing the base or route."""
    window = _window()
    viewer = window.image_view.viewer
    viewer.begin_overlay_layer("raw_mask")
    raw_item = viewer.draw_line(2, 2, 20, 2, color=(120, 255, 140))
    viewer.end_overlay_layer()
    viewer.begin_overlay_layer("route")
    route_item = viewer.draw_line(2, 8, 20, 8, color=(0, 255, 70))
    viewer.end_overlay_layer()

    viewer.set_overlay_layer_visible("raw_mask", False)

    assert viewer._pixmap_item.isVisible() is True
    assert raw_item.isVisible() is False
    assert route_item.isVisible() is True

    viewer.set_overlay_layer_visible("raw_mask", True)
    assert raw_item.isVisible() is True
    window.close()


def test_display_panel_lists_layers_and_keeps_visibility_view_only():
    """The expanded display tree controls rows without changing planning state."""
    window = _window()
    group = window.image_view
    selected_before = list(window.state.selected_centerline_ids)
    group.set_display_layer_available("raw_mask", True, default_visible=True)
    group.set_display_layer_available("route", True, default_visible=False)
    window.show()
    QApplication.processEvents()

    raw_item = group._display_layer_items["raw_mask"]
    route_item = group._display_layer_items["route"]
    assert raw_item.text(0) == "原始掩膜"
    assert raw_item.checkState(0) == Qt.Checked
    assert route_item.checkState(0) == Qt.Unchecked

    group.display_toggle_button.click()
    assert group.display_layers_panel.isVisible() is True
    raw_item.setCheckState(0, Qt.Unchecked)

    assert group.display_layer_visible("raw_mask") is False
    assert window.state.selected_centerline_ids == selected_before
    window.close()


def test_display_panel_hides_unavailable_rows_and_stale_centerlines():
    """Only currently drawable layers remain selectable in the popup."""
    window = _window()
    group = window.image_view

    window.state.centerline_catalog = {
        "items": [
            {"id": "invalid", "centerline": [(1.0, 1.0)]},
            {"id": "valid", "centerline": [(1.0, 1.0), (8.0, 8.0)]},
        ]
    }
    window.state.selected_centerline_ids = ["invalid"]
    window._sync_display_layer_availability()

    centerline_item = group._display_layer_items["centerlines"]
    assert centerline_item.isHidden() is True

    window.state.selected_centerline_ids = ["valid"]
    window._sync_display_layer_availability()
    assert centerline_item.isHidden() is False

    window.state.mask_raw = np.ones((4, 4), dtype=np.uint8)
    window.state.mask_result = {
        "processed_mask": np.ones((4, 4), dtype=np.uint8),
    }
    window._sync_display_layer_availability()
    assert group._display_layer_items["raw_mask"].isHidden() is False
    assert group._display_layer_items["raw_mask"].checkState(0) == Qt.Unchecked

    group.set_display_layer_available("raw_mask", False)
    assert group._display_layer_items["raw_mask"].isHidden() is True
    window.close()


def test_display_layers_popup_does_not_consume_main_layout_height():
    """The compact layer chooser overlays the canvas like a combo popup."""
    window = _window()
    group = window.image_view
    group.set_display_layer_available("route", True, default_visible=True)
    window.show()
    QApplication.processEvents()

    viewer_height_before = group.viewer.height()
    assert group.layout().indexOf(group.display_layers_panel) == -1

    group.display_toggle_button.click()
    QApplication.processEvents()

    assert group.display_layers_panel.windowFlags() & Qt.Popup
    assert group.display_layers_panel.isVisible() is True
    assert group.display_layers_panel.height() <= 70
    assert group.viewer.height() == viewer_height_before
    window.close()


def test_pipeline_step_results_update_state():
    window = _window()
    raw = np.ones((20, 20), dtype=np.uint8) * 255
    processed = {
        "processed_mask": raw,
        "main_angle": 0.0,
        "wide_bands": [{"centerline": [(0, 0), (1, 1)]}],
    }

    window._on_step_result("segment_done", raw)
    assert window.state.inference_done is True
    assert window.state.mask_raw is raw

    window._on_step_result("process_done", processed)
    assert window.state.mask_processed is True
    assert window.state.mask_result is processed


def test_workflow_separates_centerlines_from_path_planning_and_advanced_settings():
    from pyside6_app.panels import STEP_GROUPS

    advanced_group = next(group for group in STEP_GROUPS if group["label"] == "高级设置")
    centerline_generation_group = next(
        group for group in STEP_GROUPS if group["label"] == "生成中心线"
    )
    centerline_selection_group = next(
        group for group in STEP_GROUPS if group["label"] == "选择中心线"
    )
    planning_group = next(group for group in STEP_GROUPS if group["label"] == "路径规划")

    assert advanced_group["tasks"] == ["params"]
    assert centerline_generation_group["tasks"] == ["centerline_generate"]
    assert centerline_selection_group["tasks"] == ["centerline_select"]
    assert planning_group["tasks"] == ["entry", "exit", "unload", "plan"]


def test_task_rows_stack_actions_below_summary_without_overlap():
    window = _window()
    window.resize(1200, 800)
    window.show()
    app = QApplication.instance()
    app.processEvents()
    panel = window.task_panel
    card = next(
        card for card in panel._step_cards if card._title.text().endswith("路径规划")
    )
    card.set_expanded(True)
    for other in panel._step_cards:
        if other is not card:
            other.set_expanded(False)
    app.processEvents()

    for tid in ("entry", "exit", "unload", "plan"):
        row = panel._task_widgets[tid]
        assert row.minimumHeight() >= 40
        assert not row._action_btn.geometry().intersects(row._delete_btn.geometry())
        assert row._action_btn.geometry().width() >= 56
        assert row._delete_btn.geometry().width() >= 50

    assert panel.recommend_service_points_btn.minimumHeight() >= 30
    assert panel.confirm_no_forbidden_btn.geometry().width() > 0
    assert panel.draw_forbidden_btn.geometry().width() > 0
    window.close()


def test_planning_card_fully_expands_in_scroll_area_without_overlap():
    """回归：展开的路径规划卡片必须在滚动区内撑开内容高度。

    旧实现给 StepCard 设置了固定最小高度（仅表头高度），QScrollArea 内
    布局只按各卡片最小高度分配空间，把展开的卡片压扁：任务行互相重叠，
    后面的智能推荐/禁行区/调头方式控件盖住上方的任务行。

    注意：窗口必须先真实显示（show + 事件循环）再展开，否则布局处于
    半激活状态，几何结果不可信。
    """
    window = _window()
    window.resize(1200, 800)
    window.show()
    app = QApplication.instance()
    app.processEvents()
    panel = window.task_panel
    panel.set_route_editing_active(True)
    app.processEvents()
    card = next(c for c in panel._step_cards if c._title.text().endswith("高级设置"))
    card.set_expanded(True)
    for other in panel._step_cards:
        if other is not card:
            other.set_expanded(False)
    app.processEvents()

    rows = [panel._task_widgets["params"]]
    tops = [row.mapTo(panel, QPoint(0, 0)).y() for row in rows]
    for i in range(1, len(rows)):
        assert tops[i] >= tops[i - 1] + rows[i - 1].height() - 1, f"任务行 {rows[i].tid} 与上一行重叠"

    # 卡片底部必须低于所有任务行（下方设置区可见），且设置区最后一个
    # 控件（编辑路线）不越出卡片边界
    card_bottom = card.mapTo(panel, QPoint(0, card.height())).y()
    assert card_bottom > max(tops[i] + rows[i].height() for i in range(len(rows)))
    edit_bottom = panel.route_edit_btn.mapTo(panel, QPoint(0, panel.route_edit_btn.height())).y()
    assert edit_bottom <= card_bottom + 1
    assert not hasattr(panel, "route_edit_tools")

    # 滚动区内容高度应大于视口（出现滚动条），而不是把卡片压扁
    assert panel._scroll.widget().height() > panel._scroll.viewport().height()
    window.close()


def test_edit_mode_does_not_inject_a_duplicate_tool_grid_into_planning_card():
    """DECISION-019: entering edit mode keeps the compact planning card stable."""
    window = _window()
    window.resize(720, 800)
    window.show()
    app = QApplication.instance()
    app.processEvents()
    panel = window.task_panel
    card = next(c for c in panel._step_cards if c._title.text().endswith("高级设置"))
    card.set_expanded(True)
    for other in panel._step_cards:
        if other is not card:
            other.set_expanded(False)
    app.processEvents()
    height_before = card.height()

    panel.set_route_editing_active(True)
    app.processEvents()

    # Editing exposes the three compact route tools in the same advanced card;
    # the card must grow to contain that row rather than injecting a second
    # tool grid or clipping the existing controls.
    assert card.height() > height_before
    assert panel.route_edit_action_row.isHidden() is False
    assert not hasattr(panel, "route_edit_tools")
    window.close()


def test_centerline_selection_is_rendered_in_its_own_task_card():
    from pyside6_app.panels import CenterlineSelectionRow, RouteInfoPanel, TaskPanel

    app = QApplication.instance() or QApplication([])
    panel = TaskPanel()
    panel.show()
    app.processEvents()

    centerline_card = next(
        card for card in panel._step_cards if card._title.text().endswith("选择中心线")
    )
    advanced_card = next(
        card for card in panel._step_cards if card._title.text().endswith("高级设置")
    )
    planning_card = next(
        card for card in panel._step_cards if card._title.text().endswith("路径规划")
    )
    assert not centerline_card._expanded
    assert not advanced_card._expanded
    assert panel._task_widgets["exit"]._label.text() == "终点（可选）"
    assert panel._task_widgets["unload"]._label.text() == "卸粮点（可选）"
    assert panel._task_widgets["params"].parentWidget() is advanced_card._content
    assert panel.route_edit_btn.parentWidget() is advanced_card._content
    assert panel.route_edit_btn.parentWidget() is not planning_card._content

    catalog = {
        "items": [
            {
                "id": "centerline-001",
                "sequence": 1,
                "label": "中心线 1",
                "centerline": [(1.0, 1.0), (2.0, 2.0)],
            }
        ]
    }
    panel.update_centerline_selection(catalog, ["centerline-001"])
    app.processEvents()
    assert len(panel.centerline_selector.findChildren(CenterlineSelectionRow)) == 1
    row = panel.centerline_selector.findChildren(CenterlineSelectionRow)[0]
    row.checkbox.setChecked(False)
    app.processEvents()
    assert panel.centerline_selector._count_label.text() == "已选择 0/1 条"

    result_panel = RouteInfoPanel()
    assert not result_panel.findChildren(CenterlineSelectionRow)
    result_panel.close()
    panel.close()


def test_step_card_stylesheet_has_balanced_qss_braces():
    """A malformed card stylesheet must not emit Qt parse warnings."""
    window = _window()
    stylesheet = window.task_panel._step_cards[0].styleSheet()

    assert ";}}" not in stylesheet
    window.close()


def test_task_panel_uses_smallest_width_without_horizontal_squeeze():
    """路径规划卡片展开并纵向滚动时，紧凑左栏不得再出现横向滚动。"""
    from pyside6_app.panels import TASK_PANEL_COMPACT_MIN_WIDTH, TaskPanel

    app = QApplication.instance() or QApplication([])
    panel = TaskPanel()
    panel.resize(TASK_PANEL_COMPACT_MIN_WIDTH, 520)
    panel.show()
    planning_card = next(
        card for card in panel._step_cards if card._title.text().endswith("路径规划")
    )
    planning_card.set_expanded(True)
    for other in panel._step_cards:
        if other is not planning_card:
            other.set_expanded(False)
    app.processEvents()

    assert panel.width() == TASK_PANEL_COMPACT_MIN_WIDTH
    assert panel._scroll.verticalScrollBar().maximum() > 0
    assert panel._scroll.horizontalScrollBar().maximum() == 0
    panel.close()


def test_planning_panel_recommend_button_only_requires_start_recommendation(monkeypatch):
    window = _window()
    window.state.mask_processed = True
    window.state.mask_result = {"wide_bands": [{}, {}], "main_angle": 0.0}
    calls = []
    monkeypatch.setattr(
        window,
        "_suggest_service_points",
        lambda: (
            calls.append("suggest"),
            window.state.safe_update(
                entry_point=(2.0, 3.0),
            ),
        ),
    )

    window.task_panel.recommend_service_points_btn.click()

    assert calls == ["suggest"]
    assert window.state.entry_point == (2.0, 3.0)
    assert window.state.exit_point is None
    assert window._service_points_visible is True
    window.close()


def test_mask_processing_defers_service_points_until_path_planning(monkeypatch):
    """DECISION-007/014: service points enter the workflow with route planning."""
    window = _window()
    raw = np.ones((20, 20), dtype=np.uint8) * 255
    processed = {
        "processed_mask": raw,
        "main_angle": 0.0,
        "wide_bands": [{"centerline": [(0, 0), (1, 1)]}],
    }
    suggestions = []
    window.state.safe_update(
        mask_raw=raw,
        entry_point=(2, 2),
        exit_point=(17, 17),
        auto_path_planned=True,
    )
    monkeypatch.setattr(
        window,
        "_suggest_service_points",
        lambda: suggestions.append("suggested"),
    )

    assert window._service_points_should_show() is False
    window._apply_mask_process_result(processed)

    assert suggestions == []
    assert window._service_points_visible is False
    assert window._service_points_should_show() is False

    window._on_toggle_entry_exit()

    assert suggestions == ["suggested"]
    assert window._service_points_visible is True
    assert window._service_points_should_show() is True

def test_mask_processing_start_prompt_defers_service_points_to_path_planning(monkeypatch):
    from pyside6_app import main_window
    from pyside6_app import workers

    window = _window()
    window.state.mask_raw = np.ones((20, 20), dtype=np.uint8)
    messages = []

    class _Signal:
        def connect(self, _callback):
            return None

    class _Worker:
        def __init__(self, *_args, **_kwargs):
            self.finished = _Signal()

    monkeypatch.setattr(main_window, "MaskProcessWorker", _Worker)
    monkeypatch.setattr(workers, "require_metric_scale", lambda *_args: None)
    monkeypatch.setattr(window, "_has_background_worker", lambda: False)
    monkeypatch.setattr(window, "_save_system_undo", lambda _label: None)
    monkeypatch.setattr(window, "_invalidate_analysis_from", lambda *_args: None)
    monkeypatch.setattr(
        window,
        "_start_background_worker",
        lambda _worker, message: messages.append(message) or False,
    )

    window._on_mask_process()

    assert messages == [
        "掩膜处理已开始，完成后请进入“路径规划”确认起点；终点将沿最终作业段自动延伸。"
    ]


def test_mask_processing_rejects_historical_mask_without_active_model(monkeypatch):
    window = _window()
    messages = []
    window.state.safe_update(
        mask_raw=np.ones((20, 20), dtype=np.uint8),
        inference_done=True,
        cache_model_matches_active_engine=False,
    )
    monkeypatch.setattr(
        window.log_panel,
        "warn",
        lambda message: messages.append(message),
    )
    monkeypatch.setattr(window, "_has_background_worker", lambda: False)

    window._on_mask_process()

    assert messages == [
        "当前掩膜来自历史模型缓存，但该模型尚未加载；"
        "请先加载同一模型，或重新执行 AI识别后再处理掩膜。"
    ]
    window.close()

def test_pending_segment_completion_enables_mask_process_action():
    window = _window()
    raw = np.ones((20, 20), dtype=np.uint8) * 255
    window._pending_segment_display = True
    window.state.safe_update(
        tif_path="D:/zhlonly/project/reUI/data/tif/result.tif",
        field_boundary=[(1, 1), (10, 1), (10, 10)],
        mask_raw=raw,
        inference_original_mask=raw,
        inference_done=True,
        inference_running=False,
    )

    window._refresh_state()

    process_row = window.task_panel._task_widgets["process"]
    assert process_row.isEnabled()
    assert process_row._action_btn.isEnabled()
    assert process_row._status.text() == "可操作"
    assert process_row._action_btn.text() == "处理"


def test_field_overlay_uses_viewer_instance():
    window = _window()
    window._field_pts = [(1, 1), (10, 1)]
    window._redraw_field_overlay(mouse_pos=(10, 10))

    window.state.safe_update(field_boundary=[(1, 1), (10, 1), (10, 10)])
    window._draw_field_boundary()


def test_confirmed_forbidden_polygon_is_stored_and_invalidates_old_path():
    window = _window()
    window.state.field_boundary = [(0, 0), (19, 0), (19, 19), (0, 19)]
    window.state.auto_path_planned = True
    window.state.auto_path_valid = True
    window._forbidden_pts = [(5, 5), (10, 5), (10, 10), (5, 10)]

    window._finish_forbidden_polygon()

    assert window.state.forbidden_regions == [[(5, 5), (10, 5), (10, 10), (5, 10)]]
    assert window.state.forbidden_regions_confirmed is True
    assert window.state.auto_path_planned is False
    window._refresh_state()
    assert "已绘制 1 处" in window.task_panel.forbidden_status_label.text()


def test_replacing_field_boundary_invalidates_old_inference_and_forbidden_regions():
    window = _window()
    window.state.inference_done = True
    window.state.mask_raw = np.ones((10, 10), dtype=np.uint8)
    window.state.forbidden_regions = [[(2, 2), (4, 2), (4, 4), (2, 4)]]
    window.state.forbidden_regions_confirmed = True
    window._field_pts = [(1, 1), (18, 1), (18, 18), (1, 18)]

    window._finish_field_polygon()

    assert window.state.field_boundary == [(1, 1), (18, 1), (18, 18), (1, 18)]
    assert window.state.inference_done is False
    assert window.state.mask_raw is None
    assert window.state.forbidden_regions == []
    assert window.state.forbidden_regions_confirmed is False


def test_auto_start_does_not_load_bundled_tif_without_user_choice(monkeypatch):
    window = _window()
    loaded = []

    class RejectingLauncher:
        file_selected = ""

        def __init__(self, parent=None):
            pass

        def exec(self):
            return 0

    monkeypatch.setattr("pyside6_app.main_window.LauncherDialog", RejectingLauncher)
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr(window, "_load_tif_path", lambda path: loaded.append(path))

    window._auto_start()

    assert loaded == []


def test_auto_start_loads_launcher_selected_path(monkeypatch):
    window = _window()
    selected = r"D:\zhlonly\project\reUI\data\tif\result.tif"
    loaded = []

    class AcceptingLauncher:
        file_selected = object()

        def __init__(self, parent=None):
            self.selected_path = selected

        def exec(self):
            return 1

    monkeypatch.setattr("pyside6_app.main_window.LauncherDialog", AcceptingLauncher)
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr("pyside6_app.main_window.QTimer.singleShot", lambda _ms, fn: fn())
    monkeypatch.setattr(window, "_load_tif_path", lambda path: loaded.append(path))

    window._auto_start()

    assert loaded == [selected]


def test_scale_bar_uses_continuous_distance():
    bar = MetricScaleBar()
    bar.update_scale(ppm=10.0, view_scale=1.0)
    first = bar._scale_m
    bar.update_scale(ppm=10.0, view_scale=1.1)
    second = bar._scale_m

    assert first == 10.0
    assert 9.0 < second < 9.2
    assert MetricScaleBar._format_distance(second).endswith("m")


def test_render_animation_frame_accepts_tuple_pose(monkeypatch):
    window = _window()
    window._pipeline_result = {"path": {"tracks": [], "turns": []}}

    class TuplePoseEngine:
        def get_interpolated_position(self):
            return (5.0, 6.0), 0.25

    window._anim_engine = TuplePoseEngine()

    monkeypatch.setattr(window.image_view, "set_image", lambda image: (_ for _ in ()).throw(AssertionError("set_image should not run per animation frame")))
    monkeypatch.setattr(window.image_view.viewer, "draw_polyline", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("animation trail should not be drawn")))

    window._render_anim_frame()


def test_animation_setup_falls_back_when_geo_distance_is_nan():
    from state import AutoPathSegment, PathPoint, AppState
    from workflow import AnimationEngine

    class NanGeo:
        def pixel_distance_m(self, _p1, _p2):
            return float("nan")

    state = AppState()
    engine = AnimationEngine(state, NanGeo())
    segment = AutoPathSegment(
        points=[PathPoint(0, 0), PathPoint(3, 4)],
        status=1,
        segment_type="work",
    )

    engine.setup([segment])

    assert engine._seg_dists == [5.0]
    assert engine._cum_dists == [0.0, 5.0]
    assert state.anim_total_frames == 2


def test_geo_projected_distance_uses_dataset_xy_not_lonlat_geod():
    from geo import GeoUtils

    class ProjectedDataset:
        crs = object()
        def xy(self, row, col):
            return float(col) * 0.0035, float(row) * 0.0035

    class IdentityTransformer:
        def transform(self, x, y):
            return x, y

    geo = GeoUtils()
    geo.set_dataset(ProjectedDataset(), IdentityTransformer())

    assert abs(geo.pixel_distance_m((0, 0), (3, 4)) - 0.0175) < 1e-9


def test_geo_display_affine_scales_downsampled_preview_pixels_to_source_metres():
    from geo import GeoUtils

    class IdentityTransformer:
        def transform(self, x, y):
            return x, y

    geo = GeoUtils()
    source_transform = Affine.scale(0.0035, 0.0035)
    display_transform = source_transform * Affine.scale(4, 4)
    geo.set_affine(display_transform, IdentityTransformer())

    assert abs(geo.meters_per_pixel(10, 10) - 0.014) < 1e-12
    assert abs(geo.pixels_per_meter(10, 10) - (1.0 / 0.014)) < 1e-9
    assert abs(geo.pixel_distance_m((0, 0), (3, 4)) - 0.07) < 1e-12


def test_field_area_uses_display_geotransform_and_is_persisted():
    from geo import GeoUtils

    class IdentityTransformer:
        def transform(self, x, y):
            return x, y

    window = _window()
    window.geo = GeoUtils()
    source_transform = Affine.scale(0.0035, 0.0035)
    window.geo.set_affine(source_transform * Affine.scale(4, 4), IdentityTransformer())
    points = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    area_m2 = window._field_area_m2(points)
    area_info = window._estimate_field_area(points)

    assert abs(area_m2 - 196.0) < 1e-9
    assert area_info == "196.0 m2"


def test_segment_uses_source_geotiff_instead_of_downsampled_preview(monkeypatch):
    """DECISION-004: formal inference reads source pixels, never preview pixels."""
    window = _window()
    window._tif_path = "D:/zhlonly/project/reUI/data/tif/result.tif"
    window.state.tif_path = window._tif_path
    window._tif_rgb = np.zeros((100, 150, 3), dtype=np.uint8)
    window.state.source_img_h = 400
    window.state.source_img_w = 600
    window.state.downsample_factor = 4
    window.state.field_boundary = [(10, 20), (50, 20), (50, 60), (10, 60)]
    window.model_engine._loaded = True
    window.model_engine._model = object()
    calls = []

    class FakeRunner:
        def __init__(self, state, model_engine, geo, **_kwargs):
            pass

        def start_from_tif(self, tif_path, display_shape, field_boundary, downsample):
            calls.append((tif_path, display_shape, field_boundary, downsample))

    monkeypatch.setattr("inference.InferenceRunner", FakeRunner)

    window._on_segment()

    assert calls == [(
        window._tif_path,
        (100, 150),
        window.state.field_boundary,
        4,
    )]


def test_source_tif_inference_streams_oversized_crop_at_source_tile_size(monkeypatch):
    """DECISION-004: oversized crops stream native source tiles without global resize."""
    import sys
    import types

    import cache
    import inference
    import provenance
    from raster_preprocessing import EmptyRasterWindowError
    from state import AppState

    state = AppState()
    state.tif_path = "D:/fake/result.tif"
    state.field_boundary = [(0, 0), (3762, 0), (3762, 5139), (0, 5139)]
    state.current_model_path = "D:/fake/model.pt"
    state.current_model_name = "model.pt"
    state.source_sha256 = "source-sha"
    state.model_sha256 = "model-sha"
    state.cache_model_matches_active_engine = False
    engine = inference.ModelEngine()
    engine._loaded = True
    engine._model = object()
    runner = inference.InferenceRunner(state, engine, geo=None)
    runner.cfg.update_section("model", {"max_source_crop_pixels": 36_000_000})
    calls = {}

    class FakeDataset:
        width = 15078
        height = 21021
        count = 3
        colorinterp = ()
        nodatavals = (None, None, None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(path):
        calls["opened"] = path
        return FakeDataset()

    def fake_read_rgb_raster(dataset, *, window=None, out_shape=None, resampling=None, config=None):
        calls.setdefault("windows", []).append(window)
        calls.setdefault("out_shapes", []).append(out_shape)
        calls.setdefault("resampling", []).append(resampling)
        if len(calls["windows"]) == 1:
            calls["empty_tiles"] = calls.get("empty_tiles", 0) + 1
            raise EmptyRasterWindowError("RGB window contains no valid pixels")
        height = int(window.height)
        width = int(window.width)
        return np.ones((height, width, 3), dtype=np.uint8), {"fake": True}

    class FakeTensor:
        def __init__(self, array):
            self._array = array

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class FakeMasks:
        def __init__(self, shape):
            self.data = FakeTensor(np.ones((1, shape[0], shape[1]), dtype=np.float32))

    class FakeResult:
        def __init__(self, shape):
            self.masks = FakeMasks(shape)

    def fake_predict(images, conf=None, iou=None):
        if isinstance(images, list):
            calls.setdefault("predict_batches", []).append(len(images))
            calls["inference_shape"] = images[0].shape
            return [FakeResult(img.shape[:2]) for img in images]
        calls.setdefault("predict_singles", []).append(images.shape)
        calls["inference_shape"] = images.shape
        return [FakeResult(images.shape[:2])]

    saved = []
    fake_workflow = types.SimpleNamespace(
        WorkflowUpdater=types.SimpleNamespace(advance=lambda *args, **kwargs: None)
    )

    monkeypatch.setitem(sys.modules, "workflow", fake_workflow)
    monkeypatch.setattr("rasterio.open", fake_open)
    monkeypatch.setattr("raster_preprocessing.read_rgb_raster", fake_read_rgb_raster)
    monkeypatch.setattr(engine, "predict", fake_predict)
    monkeypatch.setattr(
        cache,
        "save_mask",
        lambda tif_path, mask, offset_x=0, offset_y=0, suffix="", meta=None, commit_callback=None: saved.append(
            (mask.shape, offset_x, offset_y, suffix, meta)
        ) or True,
    )
    monkeypatch.setattr(cache, "save_project_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(provenance, "file_sha256", lambda path: "sha")
    monkeypatch.setattr(provenance, "make_stage_record", lambda *args, **kwargs: {"stage": "inference"})

    runner._run_from_tif(
        "D:/fake/result.tif",
        (5255, 3769),
        state.field_boundary,
        downsample=4,
    )

    assert calls["opened"] == "D:/fake/result.tif"
    assert calls["windows"]
    assert calls["empty_tiles"] == 1
    assert all(shape is None for shape in calls["out_shapes"])
    assert all(resampling is None for resampling in calls["resampling"])
    first_window = calls["windows"][0]
    assert int(first_window.width) == 640
    assert int(first_window.height) == 640
    assert calls["inference_shape"][:2] == (640, 640)
    assert state.inference_done is True
    assert state.inference_running is False
    assert state.cache_model_matches_active_engine is True
    assert state.mask_raw.shape == (5255, 3769)
    assert state.mask_offset_x == 0
    assert state.mask_offset_y == 0
    assert saved and saved[0][0] == state.mask_raw.shape


def test_source_tif_inference_rejects_crop_with_only_nodata_tiles(monkeypatch):
    """DECISION-004: an entirely empty crop must not become a valid background result."""
    import pytest

    import inference
    from raster_preprocessing import EmptyRasterWindowError
    from state import AppState

    runner = inference.InferenceRunner(AppState(), inference.ModelEngine(), geo=None)
    monkeypatch.setattr(
        "raster_preprocessing.read_rgb_raster",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            EmptyRasterWindowError("RGB window contains no valid pixels")
        ),
    )

    with pytest.raises(RuntimeError, match="no valid RGB pixels"):
        runner._run_source_tif_tiles(
            object(),
            x0_src=0,
            y0_src=0,
            crop_w=640,
            crop_h=640,
            out_w=160,
            out_h=160,
        )


def test_default_harvester_params_match_the_research_machine_geometry():
    from config import DEFAULT_HARVESTER_PARAMS

    window = _window()
    window.state.harvester_params = {}
    window.cfg._raw.setdefault("harvester", {})["cutter_forward_offset_m"] = None

    params = window._effective_harvester_params()

    assert DEFAULT_HARVESTER_PARAMS["cutter_width_m"] == pytest.approx(2.15)
    assert DEFAULT_HARVESTER_PARAMS["track_width_m"] == pytest.approx(0.40)
    assert DEFAULT_HARVESTER_PARAMS["track_gauge_m"] == pytest.approx(1.60)
    assert DEFAULT_HARVESTER_PARAMS["wheelbase_m"] == pytest.approx(1.80)
    assert DEFAULT_HARVESTER_PARAMS["turn_radius_m"] == pytest.approx(0.90)
    assert params["cutter_forward_offset_m"] == pytest.approx(1.35)


def test_machine_recommendation_dialog_allows_partial_candidate_without_cutter_offset():
    from PySide6.QtWidgets import QPushButton
    from pyside6_app.dialogs import MachineRecommendationDialog

    QApplication.instance() or QApplication([])
    dlg = MachineRecommendationDialog({
        "title": "候选",
        "message": "只推荐底盘尺寸，割刀前置距离仍需实测。",
        "evidence_quality": "partial",
        "params": {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": None,
            "track_width_m": 0.40,
            "track_gauge_m": 1.0,
            "wheelbase_m": 1.8,
            "track_length_m": 1.6,
            "turn_radius_m": 0.9,
        },
    })

    apply_buttons = [
        button for button in dlg.findChildren(QPushButton)
        if button.text() == "应用候选参数"
    ]
    assert apply_buttons
    assert apply_buttons[0].isEnabled() is True
    assert "割刀前置距离仍需实测" in apply_buttons[0].toolTip()


def test_recommended_harvester_params_can_be_saved_without_inventing_cutter_offset(monkeypatch):
    import pyside6_app.main_window as main_window

    window = _window()
    window.state.harvester_params = {
        "cutter_width_m": 2.0,
        "cutter_forward_offset_m": None,
        "track_width_m": 0.45,
        "track_gauge_m": 1.2,
        "wheelbase_m": 2.0,
        "track_length_m": 1.8,
        "turn_radius_m": 1.0,
    }
    window.cfg._raw.setdefault("harvester", {})["cutter_forward_offset_m"] = None
    monkeypatch.setattr(window, "_save_system_undo", lambda _label: None)
    monkeypatch.setattr(main_window, "save_json_atomic", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window.cfg,
        "update_section",
        lambda section, values: {section: values},
    )

    ok = window._apply_harvester_params(
        {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": None,
            "track_width_m": 0.40,
            "track_gauge_m": 1.0,
            "wheelbase_m": 1.8,
            "track_length_m": 1.6,
            "turn_radius_m": 0.9,
        },
        "智能推荐",
        require_cutter_offset=False,
    )

    assert ok is True
    assert window.state.harvester_params["track_width_m"] == 0.40
    assert window.state.harvester_params["wheelbase_m"] == 1.8
    assert window.state.harvester_params["cutter_forward_offset_m"] == pytest.approx(1.35)


def test_recommended_harvester_params_sync_state_when_same_as_config(monkeypatch):
    import pyside6_app.main_window as main_window

    window = _window()
    raw = np.ones((3, 3), dtype=np.uint8)
    window.state.tif_path = "dummy.tif"
    window.state.field_boundary = [(0, 0), (1, 0), (1, 1)]
    window.state.inference_done = True
    window.state.mask_raw = raw
    window.state.mask_processed = True
    window.state.mask_result = {"processed_mask": raw}
    window.state.entry_point = (1, 1)
    window.state.exit_point = (2, 2)
    window.state.entry_point_locked = True
    window.state.exit_point_locked = True
    window.state.harvester_params_confirmed = True
    window.state.harvester_params = {}
    window.cfg._raw.setdefault("harvester", {})["cutter_forward_offset_m"] = None
    defaults = window._effective_harvester_params({})
    monkeypatch.setattr(window, "_save_system_undo", lambda _label: None)
    monkeypatch.setattr(
        main_window,
        "save_json_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged config should not be written")
        ),
    )

    ok = window._apply_harvester_params(
        defaults,
        "智能推荐",
        require_cutter_offset=False,
    )

    assert ok is True
    assert window.state.harvester_params["track_width_m"] == defaults["track_width_m"]
    assert window.state.harvester_params["track_length_m"] == defaults["track_length_m"]
    assert window.state.harvester_params["cutter_forward_offset_m"] == pytest.approx(1.35)
    assert window.task_panel._task_widgets["centerline_generate"]._action_btn.isEnabled()
    assert not window.task_panel._task_widgets["plan"]._action_btn.isEnabled()


def test_restore_visual_prefers_planned_path_over_boundary_only(monkeypatch):
    from state import AutoPathSegment

    window = _window()
    called = []
    window.state.auto_path_segments = [
        AutoPathSegment(points=[(0, 0), (5, 0)], segment_type="work")
    ]
    window.state.auto_path_planned = True
    monkeypatch.setattr(window, "_show_path", lambda path: called.append(("path", path)))

    window._restore_visual_from_state()

    assert called and called[0][0] == "path"


def test_animation_render_does_not_replace_large_base_image(monkeypatch):
    window = _window()
    window._pipeline_result = {"path": {"tracks": [], "turns": []}}

    class TuplePoseEngine:
        def get_interpolated_position(self):
            return (5.0, 6.0), 0.25

    window._anim_engine = TuplePoseEngine()
    monkeypatch.setattr(window.image_view, "set_image", lambda image: (_ for _ in ()).throw(AssertionError("set_image should not run per animation frame")))
    monkeypatch.setattr(window.image_view.viewer, "draw_polyline", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("animation trail should not be drawn")))

    window._render_anim_frame()


def test_cached_auto_path_rebuilds_pipeline_result_for_restore():
    window = _window()
    saved = {
        "auto_path_planned": True,
        "auto_path": [
            {"segment_type": "work", "status": 1, "corridor_id": 0, "length_m": 5.0, "points": [[0, 0], [500, 0]]},
            {"segment_type": "turn", "status": 0, "corridor_id": 1, "length_m": 3.0, "points": [[500, 0], [500, 300]]},
        ],
        "path_status": [1, 0],
        "auto_path_valid": True,
        "auto_path_desc": "cached",
        "last_total_path_m": 8.0,
        "last_work_path_m": 5.0,
        "last_turn_path_m": 3.0,
        "last_entry_exit_path_m": 0.0,
        "last_rolling_rate": 2.5,
        "last_crop_mask_geometric_coverage_pct": 94.0,
    }

    window._restore_cached_path_geometry(saved)

    assert len(window.state.auto_path_segments) == 2
    assert window.state.auto_path_planned is True
    assert [segment.length_m for segment in window.state.auto_path_segments] == [5.0, 3.0]
    assert window.state.last_total_path_m == 8.0
    assert window.state.last_rolling_rate == 2.5
    assert window._pipeline_result["path"]["full_path"] == [[(0.0, 0.0), (500.0, 0.0)], [(500.0, 0.0), (500.0, 300.0)]]
    assert window._pipeline_result["path"]["segment_types"] == ["work", "turn"]
    assert window._pipeline_result["path"]["segment_lengths_m"] == [5.0, 3.0]
    assert window._pipeline_result["path"]["validation"]["total_length_m"] == 8.0
    assert window._pipeline_result["path"]["is_valid"] is False
    assert window.state.auto_path_valid is False
    assert window._pipeline_result["path"]["validation"]["valid_at_save_time"] is True
    assert window._pipeline_result["path"]["validation"]["current_validation_complete"] is False
    assert window._pipeline_result["path"]["validation"]["validation_source"] == "cache_snapshot"


def test_task_panel_accordion_keeps_only_one_card_expanded():
    window = _window()
    panel = window.task_panel

    panel._step_cards[0].set_expanded(True)
    panel._on_card_toggled(0)
    panel._step_cards[1].set_expanded(True)
    panel._on_card_toggled(1)

    expanded = [i for i, card in enumerate(panel._step_cards) if card._expanded]
    assert expanded == [1]


def test_task_panel_does_not_expose_internal_usage_mode_selector():
    window = _window()

    assert not hasattr(window.task_panel, "usage_mode_combo")


def test_task_actions_reset_after_rollback_to_pending():
    from state import AppState
    from pyside6_app.task_state import derive_task_statuses

    state = AppState()
    panel = _window().task_panel

    state.tif_path = "dummy.tif"
    state.field_boundary = [(0, 0), (1, 0), (1, 1)]
    raw = np.ones((3, 3), dtype=np.uint8)
    state.mask_raw = raw
    state.inference_done = True
    state.mask_processed = True
    state.mask_result = {"processed_mask": raw}
    panel.apply_task_statuses(derive_task_statuses(state, has_image=True))
    assert panel._task_widgets["process"]._action_btn.text() == "重做"

    state.mask_processed = False
    state.mask_result = None
    panel.apply_task_statuses(derive_task_statuses(state, has_image=True))

    row = panel._task_widgets["process"]
    assert row._action_btn.text() == "处理"
    assert row._action_btn.isEnabled()


def test_path_planning_action_stays_clickable_when_only_cutter_offset_is_missing():
    from state import AppState
    from pyside6_app.task_state import derive_task_statuses

    raw = np.ones((3, 3), dtype=np.uint8)
    state = AppState()
    state.tif_path = "dummy.tif"
    state.field_boundary = [(0, 0), (1, 0), (1, 1)]
    state.inference_done = True
    state.mask_raw = raw
    state.mask_processed = True
    state.mask_result = {"processed_mask": raw}
    state.centerline_catalog = {
        "algorithm_version": "operator-centerline-catalog-v1",
        "items": [{"id": "centerline-001", "sequence": 1}],
    }
    state.selected_centerline_ids = ["centerline-001"]
    state.harvester_params = {
        "cutter_width_m": 2.15,
        "cutter_forward_offset_m": None,
        "track_width_m": 0.40,
        "track_gauge_m": 1.00,
        "wheelbase_m": 1.80,
        "track_length_m": 1.70,
        "turn_radius_m": 0.90,
    }
    state.entry_point = (1, 1)
    state.exit_point = (2, 2)
    state.entry_point_locked = True
    state.exit_point_locked = True
    state.harvester_params_confirmed = True

    statuses = derive_task_statuses(state, has_image=True)
    panel = _window().task_panel
    panel.apply_task_statuses(statuses)

    assert statuses["params"].status == "available"
    assert statuses["params"].text == "缺割刀前置"
    assert statuses["plan"].status == "available"
    assert panel._task_widgets["plan"]._action_btn.isEnabled()


def test_mask_completion_stops_at_parameter_confirmation_before_path_planning():
    from state import AppState
    from pyside6_app.task_state import derive_task_statuses

    raw = np.ones((3, 3), dtype=np.uint8)
    state = AppState()
    state.tif_path = "dummy.tif"
    state.field_boundary = [(0, 0), (1, 0), (1, 1)]
    state.inference_done = True
    state.mask_raw = raw
    state.mask_processed = True
    state.mask_result = {"processed_mask": raw}
    state.centerline_catalog = {
        "algorithm_version": "operator-centerline-catalog-v1",
        "items": [{"id": "centerline-001", "sequence": 1}],
    }
    state.selected_centerline_ids = ["centerline-001"]
    state.harvester_params = {
        "cutter_width_m": 2.15,
        "cutter_forward_offset_m": 0.65,
        "track_width_m": 0.40,
        "track_gauge_m": 1.00,
        "wheelbase_m": 1.80,
        "track_length_m": 1.70,
        "turn_radius_m": 0.90,
    }
    state.entry_point = (1, 1)
    state.exit_point = (2, 2)

    statuses = derive_task_statuses(state, has_image=True)
    assert statuses["params"].status == "available"
    assert statuses["entry"].status == "available"
    assert statuses["exit"].status == "available"
    assert statuses["plan"].status == "pending"


def test_path_planning_unlocks_after_parameters_and_start_are_confirmed():
    from state import AppState
    from pyside6_app.task_state import derive_task_statuses

    raw = np.ones((3, 3), dtype=np.uint8)
    state = AppState()
    state.tif_path = "dummy.tif"
    state.field_boundary = [(0, 0), (1, 0), (1, 1)]
    state.inference_done = True
    state.mask_raw = raw
    state.mask_processed = True
    state.mask_result = {"processed_mask": raw}
    state.centerline_catalog = {
        "algorithm_version": "operator-centerline-catalog-v1",
        "items": [{"id": "centerline-001", "sequence": 1}],
    }
    state.selected_centerline_ids = ["centerline-001"]
    state.harvester_params = {
        "cutter_width_m": 2.15,
        "cutter_forward_offset_m": 0.65,
        "track_width_m": 0.40,
        "track_gauge_m": 1.00,
        "wheelbase_m": 1.80,
        "track_length_m": 1.70,
        "turn_radius_m": 0.90,
    }
    state.harvester_params_confirmed = True
    state.entry_point = (1, 1)
    state.exit_point = None
    state.entry_point_locked = True
    state.exit_point_locked = False

    statuses = derive_task_statuses(state, has_image=True)
    assert statuses["params"].status == "done"
    assert statuses["entry"].status == "done"
    assert statuses["exit"].status == "done"
    assert statuses["exit"].text == "自动延伸"
    assert statuses["plan"].status == "available"


def test_plan_action_accepts_confirmed_start_without_optional_exit(monkeypatch):
    from pyside6_app import main_window, workers

    window = _window()
    raw = np.ones((20, 20), dtype=np.uint8) * 255
    window.state.safe_update(
        mask_processed=True,
        mask_result={
            "processed_mask": raw,
            "planning_support_mask": raw,
        },
        field_boundary=[(0, 0), (19, 0), (19, 19), (0, 19)],
        harvester_params_confirmed=True,
        entry_point=(2.0, 3.0),
        entry_point_locked=True,
        exit_point=None,
        exit_point_locked=False,
        centerline_catalog={
            "algorithm_version": "operator-centerline-catalog-v1",
            "items": [{
                "id": "centerline-001",
                "sequence": 1,
                "label": "中心线 1",
                "centerline": [(2.0, 3.0), (18.0, 3.0)],
            }],
        },
        selected_centerline_ids=["centerline-001"],
        centerline_selection_report={
            "selection_origin": "operator",
            "selected_ids": ["centerline-001"],
        },
    )
    created = []

    class _Signal:
        def connect(self, _callback):
            return None

    class _Worker:
        def __init__(self, _mask, _geo, state, config, **_kwargs):
            self.finished = _Signal()
            created.append((state.entry_point, state.exit_point, dict(config)))

    monkeypatch.setattr(main_window, "PlanWorker", _Worker)
    monkeypatch.setattr(workers, "require_metric_scale", lambda *_args: None)
    monkeypatch.setattr(window, "_planning_preflight", lambda _result: True)
    monkeypatch.setattr(window, "_has_background_worker", lambda: False)
    monkeypatch.setattr(window, "_save_system_undo", lambda _label: None)
    monkeypatch.setattr(window, "_invalidate_analysis_from", lambda *_args: None)
    monkeypatch.setattr(window, "_start_background_worker", lambda *_args: True)

    window._on_plan()

    assert created
    assert created[0][0] == (2.0, 3.0)
    assert created[0][1] is None
    assert created[0][2]["service_point_mode"] == "reference_only"
    assert created[0][2]["include_route_finish_extension"] is True
    assert created[0][2].get("route_order") != "adjacent_snake"


def test_start_recommendation_removes_only_unconfirmed_legacy_exit(monkeypatch):
    import path_planner

    window = _window()
    window.state.mask_result = {
        "main_angle": 0.0,
        "wide_bands": [{"id": 1}],
    }
    window.state.exit_point = (17.0, 18.0)
    window.state.exit_point_locked = False
    monkeypatch.setattr(
        path_planner,
        "prepare_band_centerlines",
        lambda *_args, **_kwargs: ([[(2.0, 3.0), (12.0, 3.0)]], {}),
    )

    window._suggest_service_points()

    assert window.state.entry_point is not None
    assert window.state.exit_point is None
    assert window.state.exit_point_locked is False


def test_recommended_service_point_confirmation_is_explicit_and_stepwise():
    window = _window()
    window.state.entry_point = (10, 20)
    window.state.exit_point = (30, 40)
    window.state.entry_point_locked = False
    window.state.exit_point_locked = False

    window._on_toggle_entry_exit("entry")
    assert window.state.entry_point_locked is True
    assert window.state.exit_point_locked is False
    assert window.state.entry_exit_mode is False

    window._on_toggle_entry_exit("exit")
    assert window.state.exit_point_locked is True
    assert window.state.entry_exit_mode is False


def test_canvas_compare_toggles_source_image_and_current_result(monkeypatch):
    """DECISION-004: 对比固定为原图与当前结果，不是相邻处理步骤。"""
    window = _window()
    calls = []
    monkeypatch.setattr(
        window.image_view,
        "set_image",
        lambda image: calls.append(("source", image is window._tif_rgb)),
    )
    monkeypatch.setattr(
        window,
        "_restore_visual_from_state",
        lambda: calls.append(("current", True)),
    )

    window._on_toggle_mask_compare()
    assert window._show_mask_before is True
    assert calls == [("source", True)]

    window._on_toggle_mask_compare()
    assert window._show_mask_before is False
    assert calls[-1] == ("current", True)


def test_canvas_compare_hides_all_overlay_choices_in_source_mode(monkeypatch):
    """Source TIF mode must not leave stale overlay rows selectable."""
    window = _window()
    group = window.image_view
    group.set_display_layer_available("route", True, default_visible=True)
    route_item = group._display_layer_items["route"]
    assert route_item.isHidden() is False

    monkeypatch.setattr(group, "set_image", lambda *_args, **_kwargs: None)
    window._on_toggle_mask_compare()

    assert window._show_mask_before is True
    assert route_item.isHidden() is True
    assert group.display_toggle_button.isEnabled() is False
    window.close()


def test_canvas_compare_preserves_zoom_rotation_and_image_location():
    """DECISION-004: source/current comparison must stay on the same crop."""
    window = _window()
    window._tif_rgb = np.zeros((1000, 1200, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    window.resize(1000, 720)
    window.show()
    QApplication.processEvents()

    viewer = window.image_view.viewer
    viewer.resetTransform()
    viewer.scale(20.0, 20.0)
    viewer.set_rotation(7.0)
    viewer._update_scene_rect_padding()
    viewer.centerOn(735.0, 418.0)
    QApplication.processEvents()

    scale_before = viewer.get_view_scale()
    center_before = viewer.mapToScene(viewer.viewport().rect().center())
    window._on_toggle_mask_compare()
    QApplication.processEvents()
    center_source = viewer.mapToScene(viewer.viewport().rect().center())

    assert viewer.get_view_scale() == pytest.approx(scale_before)
    assert center_source.x() == pytest.approx(center_before.x(), abs=0.2)
    assert center_source.y() == pytest.approx(center_before.y(), abs=0.2)

    window._on_toggle_mask_compare()
    QApplication.processEvents()
    center_current = viewer.mapToScene(viewer.viewport().rect().center())
    assert viewer.get_view_scale() == pytest.approx(scale_before)
    assert center_current.x() == pytest.approx(center_before.x(), abs=0.2)
    assert center_current.y() == pytest.approx(center_before.y(), abs=0.2)
    window.close()


def test_compact_mask_result_keeps_small_planning_contour():
    """DECISION-009/017: UI 可删稠密轮廓，但必须保留轻量规划边界。"""
    from path_planner import build_band_mask

    window = _window()
    contour = np.asarray(
        [(2, 2), (17, 2), (17, 17), (2, 17)], dtype=np.float32
    )
    result = {
        "wide_bands": [{
            "id": 1,
            "contour": contour,
            "centerline": [(3.0, 9.0), (16.0, 9.0)],
        }],
        "narrow_bands": [],
        "all_bands": [],
    }

    compact = window._compact_mask_result_for_ui(result)
    band = compact["wide_bands"][0]

    assert "contour" not in band
    assert 3 <= len(band["planning_contour"]) <= 256
    assert np.count_nonzero(build_band_mask((20, 20), [band])) > 0


def test_unchanged_harvester_parameters_can_still_be_confirmed(monkeypatch):
    window = _window()
    defaults = window._effective_harvester_params({})
    defaults["cutter_forward_offset_m"] = 0.65
    window.state.harvester_params = dict(defaults)
    window.state.harvester_params_confirmed = False
    monkeypatch.setattr(
        window.cfg,
        "update_section",
        lambda *args, **kwargs: window.cfg.snapshot(),
    )

    assert window._apply_harvester_params(defaults, "custom") is True
    assert window.state.harvester_params_confirmed is True


def test_harvester_parameter_confirmation_is_persisted_in_project_state(tmp_path):
    import cache
    from state import AppState

    tif = tmp_path / "field.tif"
    tif.write_bytes(b"field")
    state = AppState()
    state.harvester_params = {"cutter_width_m": 2.15}
    state.harvester_params_confirmed = True
    payload = cache._build_project_state_payload(str(tif), state, "params_confirmed")

    assert payload["harvester_params_confirmed"] is True


def test_project_state_persists_route_geometry_modes(tmp_path):
    import cache
    from state import AppState

    tif = tmp_path / "field.tif"
    tif.write_bytes(b"field")
    state = AppState()
    state.path_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]
    state.path_status = [1, 0]
    state.path_geometry_modes = ["straight", "turn"]

    payload = cache._build_project_state_payload(str(tif), state, "manual_route")

    assert payload["path_points"] == [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]]
    assert payload["path_status"] == [1, 0]
    assert payload["path_geometry_modes"] == ["straight", "turn"]


def test_cache_restore_reenables_confirmed_parameters_after_processed_mask(monkeypatch):
    """DECISION-002/013: restored domain state must drive the actionable task row."""
    window = _window()
    mask = np.ones((8, 8), dtype=np.uint8)
    params = {
        "cutter_width_m": 1.75,
        "cutter_forward_offset_m": 1.35,
        "track_width_m": 0.60,
        "track_gauge_m": 1.00,
        "wheelbase_m": 1.80,
        "track_length_m": 1.70,
        "turn_radius_m": 0.90,
    }
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)
    monkeypatch.setattr(window, "_restore_visual_from_state", lambda: None)

    window._restore_cache_payload({
        "path": "dummy.tif",
        "saved": {
            "inference_done": True,
            "mask_processed": True,
            "mask_result": {"main_angle": 0.0, "wide_bands": []},
            "harvester_params": params,
            "harvester_params_confirmed": True,
        },
        "raw_mask": mask,
        "processed_mask": mask,
        "processed_layers": {},
        "processed_cache_valid": True,
        "path_cache_valid": False,
    })

    row = window.task_panel._task_widgets["params"]
    assert window.state.harvester_params_confirmed is True
    assert row._done is True
    assert row._action_btn.isEnabled()


def test_cache_restore_keeps_visible_turn_strategy_equal_to_restored_state(monkeypatch):
    """DECISION-007: the visible selector must not lie about planning input."""
    window = _window()
    assert window.state.turn_strategy == "contour"
    assert window.task_panel.turn_combo.currentData() == "contour"
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)
    monkeypatch.setattr(window, "_restore_visual_from_state", lambda: None)

    window._restore_cache_payload({
        "path": "dummy.tif",
        "saved": {
            "turn_strategy": "bow",
            "turn_strategy_user_selected": True,
            "inference_done": False,
            "mask_processed": False,
        },
        "processed_cache_valid": False,
        "path_cache_valid": False,
    })

    assert window.state.turn_strategy == "bow"
    assert window.task_panel.turn_combo.currentData() == "bow"


def test_legacy_unconfirmed_turn_strategy_restores_contour_default(monkeypatch):
    """DECISION-007: an old wheel-mutated cache cannot replace the contour default."""
    window = _window()
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)
    monkeypatch.setattr(window, "_restore_visual_from_state", lambda: None)

    window._restore_cache_payload({
        "path": "dummy.tif",
        "saved": {
            "turn_strategy": "bow",
            "inference_done": False,
            "mask_processed": False,
        },
        "processed_cache_valid": False,
        "path_cache_valid": False,
    })

    assert window.state.turn_strategy == "contour"
    assert window.task_panel.turn_combo.currentData() == "contour"


def test_closed_turn_selector_ignores_mouse_wheel():
    """The turn strategy changes only through an explicitly opened selector."""
    window = _window()
    combo = window.task_panel.turn_combo
    assert combo.currentData() == "contour"
    event = QWheelEvent(
        QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(), QPoint(0, -120),
        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
    )

    QApplication.sendEvent(combo, event)

    assert combo.currentData() == "contour"
    assert window.state.turn_strategy == "contour"
    window.close()


def test_undo_redo_buttons_follow_history_stack_state():
    from state import AppState

    window = _window()
    state = AppState()
    window.top_toolbar.refresh_actions(state)
    assert not window.top_toolbar._history_buttons["UNDO"].isEnabled()
    assert not window.top_toolbar._history_buttons["REDO"].isEnabled()

    state.save_undo("change")
    window.top_toolbar.refresh_actions(state)
    assert window.top_toolbar._history_buttons["UNDO"].isEnabled()
    assert not window.top_toolbar._history_buttons["REDO"].isEnabled()

    state.undo()
    window.top_toolbar.refresh_actions(state)
    assert not window.top_toolbar._history_buttons["UNDO"].isEnabled()
    assert window.top_toolbar._history_buttons["REDO"].isEnabled()


def test_connect_work_lines_uses_global_service_points_with_local_work_lines():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_offset_x = 100
    state.mask_offset_y = 200
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    state.entry_point = (110, 220)
    state.exit_point = (150, 260)

    full_path, segment_types = _connect_work_lines(
        [[(10, 20), (10, 60)], [(20, 60), (20, 20)]],
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=1.0,
        turn_strategy="auto",
        config={"service_point_mode": "connect"},
    )

    assert segment_types[0] == "entry"
    assert full_path[0][0] == (10.0, 20.0)
    assert full_path[0][-1] == full_path[1][0]
    assert segment_types[-1] == "exit"
    assert full_path[-1][-1] == (50.0, 60.0)


def test_service_points_default_to_reference_only_without_route_connectors():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_offset_x = 100
    state.mask_offset_y = 200
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    state.entry_point = (130.0, 250.0)
    state.exit_point = (190.0, 280.0)

    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(10, 20), (10, 60)], [(20, 60), (20, 20)]],
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=1.0,
        turn_strategy="auto",
        return_diagnostics=True,
    )

    assert segment_types[0] == "work"
    assert segment_types[-1] == "work"
    assert diagnostics["service_points"]["mode"] == "reference_only"
    assert diagnostics["service_points"]["entry_reference_used"] is True
    assert diagnostics["service_points"]["exit_reference_used"] is True
    assert diagnostics["service_points"]["entry_connector_emitted"] is False
    assert diagnostics["service_points"]["exit_connector_emitted"] is False
    assert "entry" not in segment_types
    assert "exit" not in segment_types


def test_missing_optional_exit_finishes_on_last_work_line_without_return():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((120, 160), dtype=np.uint8)
    state.entry_point = (0.0, 30.0)
    state.exit_point = None

    full_path, segment_types, diagnostics = _connect_work_lines(
        [
            [(0.0, 10.0), (100.0, 10.0)],
            [(0.0, 20.0), (100.0, 20.0)],
            [(0.0, 30.0), (100.0, 30.0)],
            [(0.0, 40.0), (100.0, 40.0)],
        ],
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=1.0,
        turn_strategy="contour",
        global_main_angle=0.0,
        return_diagnostics=True,
    )

    first_work = next(
        segment for segment, kind in zip(full_path, segment_types)
        if kind == "work"
    )
    assert first_work[0] == (0.0, 30.0)
    assert "entry" not in segment_types
    assert "exit" not in segment_types
    assert "turn_reverse" not in segment_types
    assert diagnostics["service_points"]["exit_present"] is False


def test_contour_route_ignores_legacy_adjacent_snake_override_and_writes_finish_extension():
    """DECISION-016: contour must keep the half-split order at the desktop boundary."""
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((120, 160), dtype=np.uint8)
    state.entry_point = (0.0, 10.0)
    lines = [
        [(0.0, 10.0), (100.0, 10.0)],
        [(0.0, 20.0), (100.0, 20.0)],
        [(0.0, 30.0), (100.0, 30.0)],
        [(0.0, 40.0), (100.0, 40.0)],
    ]

    full_path, segment_types, diagnostics = _connect_work_lines(
        lines,
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=1.0,
        turn_strategy="contour",
        global_main_angle=0.0,
        config={
            "route_order": "adjacent_snake",
            "include_route_finish_extension": True,
            "route_finish_extension_m": 2.0,
        },
        return_diagnostics=True,
    )

    work_lines = [
        segment for segment, kind in zip(full_path, segment_types)
        if kind == "work"
    ]
    assert [line[0][1] for line in work_lines] == [10.0, 30.0, 20.0, 40.0]
    assert segment_types[-1] == "exit"
    assert full_path[-1][0] == work_lines[-1][-1]
    assert full_path[-1][-1] != full_path[-1][0]
    assert np.linalg.norm(
        np.asarray(full_path[-1][-1]) - np.asarray(full_path[-1][0])
    ) > 0.0
    assert diagnostics["contour_ordering"]["ordering"] == (
        "contour_half_split_interleaved"
    )


def test_service_points_can_explicitly_remain_reference_only():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_offset_x = 100
    state.mask_offset_y = 200
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    state.entry_point = (130.0, 250.0)
    state.exit_point = (190.0, 280.0)

    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(10, 20), (10, 60)], [(20, 60), (20, 20)]],
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=1.0,
        turn_strategy="auto",
        config={"service_point_mode": "reference_only"},
        return_diagnostics=True,
    )

    assert segment_types[0] == "work"
    assert "entry" not in segment_types
    assert "exit" not in segment_types
    assert diagnostics["service_points"]["mode"] == "reference_only"
    assert diagnostics["service_points"]["entry_connector_emitted"] is False
    assert diagnostics["service_points"]["exit_connector_emitted"] is False
    assert full_path[0][0] != (30.0, 50.0)


def test_contour_starts_with_the_pair_containing_nearest_entry_row():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((120, 160), dtype=np.uint8)
    state.entry_point = (0.0, 30.0)

    full_path, segment_types, diagnostics = _connect_work_lines(
        [
            [(0.0, 10.0), (100.0, 10.0)],
            [(0.0, 20.0), (100.0, 20.0)],
            [(0.0, 30.0), (100.0, 30.0)],
            [(0.0, 40.0), (100.0, 40.0)],
            [(0.0, 50.0), (100.0, 50.0)],
            [(0.0, 60.0), (100.0, 60.0)],
        ],
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=1.0,
        turn_strategy="contour",
        global_main_angle=0.0,
        return_diagnostics=True,
    )

    first_work_index = segment_types.index("work")
    assert full_path[first_work_index][0] == (0.0, 30.0)
    assert diagnostics["contour_ordering"]["entry_priority_applied"] is True
    assert diagnostics["contour_ordering"]["ordered_original_indices"][0] == 2


def test_service_point_references_keep_nearest_safe_entry_row_authoritative():
    """终点不能覆盖起点；四行时起点优先于相邻数最小值。"""
    from path_planner import _connect_work_lines
    from state import AppState

    def plan(exit_point):
        state = AppState()
        state.mask_offset_x = 0
        state.mask_offset_y = 0
        state.mask_raw = np.zeros((200, 200), dtype=np.uint8)
        state.entry_point = (10.0, 10.0)
        state.exit_point = exit_point
        return _connect_work_lines(
            [
                [(0.0, 10.0), (100.0, 10.0)],
                [(0.0, 20.0), (100.0, 20.0)],
                [(0.0, 30.0), (100.0, 30.0)],
                [(0.0, 40.0), (100.0, 40.0)],
            ],
            state=state,
            geo=_FixedMetricGeo(),
            min_turn_radius_m=1.0,
            turn_strategy="contour",
            return_diagnostics=True,
            global_main_angle=0.0,
            processed_mask=state.mask_raw,
        )

    first_path, first_types, first_diagnostics = plan((90.0, 40.0))
    second_path, second_types, second_diagnostics = plan((10.0, 10.0))
    first_work = [
        segment for segment, kind in zip(first_path, first_types)
        if kind == "work"
    ]
    second_work = [
        segment for segment, kind in zip(second_path, second_types)
        if kind == "work"
    ]

    assert first_diagnostics["reference_ordering"]["exit_reference_used"] is True
    assert second_diagnostics["reference_ordering"]["exit_reference_used"] is True
    assert first_work == second_work
    # Four lines cannot start at the outermost row, visit all four and avoid
    # every adjacent-row transition.  The confirmed entry remains the first
    # priority; the optional exit must not move the route to the second row.
    assert first_work[0][0] == (0.0, 10.0)
    contour = first_diagnostics["contour_ordering"]
    assert contour["entry_priority_overrode_minimum_adjacency"] is True
    assert contour["minimum_possible_adjacent_transition_count"] == 0
    assert contour["selected_adjacent_transition_count"] == 1
    assert contour["entry_nearest_feasible_verified"] is True
    assert first_diagnostics["reference_ordering"]["selected_exit_distance_m"] <= 10.0
    actual_exit_distance_m = float(
        np.hypot(
            first_work[-1][-1][0] - 90.0,
            first_work[-1][-1][1] - 40.0,
        ) * 0.01
    )
    assert first_diagnostics["reference_ordering"]["selected_exit_distance_m"] == pytest.approx(
        actual_exit_distance_m
    )



def test_manual_bow_falls_back_when_spacing_is_less_than_two_radius():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(0, 0), (0, 20)], [(3, 20), (3, 0)]],
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=2.0,
        turn_strategy="bow",
        return_diagnostics=True,
    )

    assert "turn" in segment_types
    assert diagnostics["used_strategies"] == ["pear"]
    assert diagnostics["requested_strategy"] == "bow"


def test_path_segments_are_globalised_for_display_when_mask_is_cropped():
    from path_planner import _path_to_global_coords
    from state import AppState

    state = AppState()
    state.mask_offset_x = 100
    state.mask_offset_y = 200

    assert _path_to_global_coords([[(1, 2), (3, 4)]], state) == [[(101.0, 202.0), (103.0, 204.0)]]


def test_bow_fallback_removes_requested_bow_hard_error():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    _path, _types, diagnostics = _connect_work_lines(
        [[(0, 0), (0, 20)], [(3, 20), (3, 0)]],
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=2.0,
        turn_strategy="bow",
        return_diagnostics=True,
    )

    assert diagnostics["used_strategies"] == ["pear"]
    assert diagnostics["fallback_reasons"]
    assert not any("弓形转弯" in reason for reason in diagnostics["hard_reasons"])


def test_turn_strategy_selector_matrix():
    from path_planner import _select_turn_strategy

    assert _select_turn_strategy("auto", 5.0, 2.0)["strategy"] == "bow"
    assert _select_turn_strategy("auto", 4.02, 2.0)["strategy"] == "semicircle"
    assert _select_turn_strategy("auto", 3.0, 2.0)["strategy"] == "pear"
    decision = _select_turn_strategy("bow", 3.0, 2.0)
    assert decision["strategy"] == "pear"
    assert decision["fallback_from"] == "bow"


def test_unload_point_is_explicit_service_segment_before_exit():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    state.entry_point = (0.0, 0.0)
    state.exit_point = (80.0, 80.0)
    state.unload_points = [(60.0, 70.0), (10.0, 90.0)]

    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(0, 0), (0, 40)], [(20, 40), (20, 0)]],
        state=state,
        geo=_FixedMetricGeo(),
        min_turn_radius_m=1.0,
        turn_strategy="auto",
        config={"service_point_mode": "connect"},
        return_diagnostics=True,
    )

    assert "unload" in segment_types
    unload_index = segment_types.index("unload")
    assert unload_index < segment_types.index("exit")
    assert full_path[unload_index][-1] == (60.0, 70.0)
    assert full_path[unload_index + 1][0] == (60.0, 70.0)
    assert diagnostics["service_points"]["unload_visit_count"] == 1


def test_planning_factor_report_records_core_decision_inputs():
    from path_planner import _build_planning_factor_report
    from state import AppState

    state = AppState()
    state.mask_offset_x = 100
    state.mask_offset_y = 200
    state.mask_raw = np.zeros((50, 60), dtype=np.uint8)
    state.entry_point = (110.0, 220.0)
    state.exit_point = (150.0, 240.0)
    state.unload_points = [(130.0, 230.0)]
    state.harvester_params = {
        "cutter_width_m": 2.2,
        "cutter_forward_offset_m": 1.2,
        "track_width_m": 0.45,
        "track_gauge_m": 1.65,
        "turn_radius_m": 2.1,
    }

    report = _build_planning_factor_report(
        state=state,
        work_lines=[[(0, 0), (0, 10)], [(10, 10), (10, 0)]],
        layout={"generated_pass_count": 2},
        turn_assessment={"strategy": "fishtail", "fallback_reasons": ["narrow headland"]},
        validation={"crop_mask_geometric_coverage_pct": 91.0, "track_core_overlap_pct": 1.2},
    )

    assert report["harvester"]["turn_radius_m"] == 2.1
    assert report["mask"]["offset_xy"] == [100, 200]
    assert report["service_points"]["entry_local"] == [10.0, 20.0]
    assert report["service_points"]["unload_count"] == 1
    assert report["route_structure"]["work_line_count"] == 2
    assert report["decision_chain"][0] == "processed_mask_and_field_support"


def test_auto_strategy_request_runs_one_deterministic_turn_implementation():
    from path_planner import _candidate_strategies_for_request

    assert _candidate_strategies_for_request("auto") == ["auto"]
    assert _candidate_strategies_for_request("bow") == ["bow"]


def test_remove_headlands_can_preserve_headland_layer():
    from mask_processor import remove_headlands

    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[35:45, 10:110] = 255  # horizontal field-body row band
    mask[10:70, 2:12] = 255    # vertical headland-like component

    body, headland = remove_headlands(mask, 0.0, angle_thresh_deg=45.0, min_area_ratio=0.01, return_headland=True)

    assert body[40, 60] == 255
    assert body[40, 5] == 0
    assert headland[40, 5] == 255
    assert np.count_nonzero(headland & body) == 0


def test_validate_footprints_reports_headland_overlap_separately():
    from footprint_planner import validate_footprints

    body = np.zeros((80, 120), dtype=np.uint8)
    body[35:45, 20:100] = 255
    headland = np.zeros_like(body)
    headland[10:70, 2:14] = 255
    path = [[(8.0, 15.0), (8.0, 65.0)]]

    metrics = validate_footprints(
        path,
        body,
        harvester={
            "cutter_width_m": 2.0, "cutter_forward_offset_m": 0.5,
            "track_width_m": 1.0,
            "track_gauge_m": 0.0, "wheelbase_m": 0.5,
            "track_length_m": 0.5, "turn_radius_m": 0.5,
        },
        config={"headland_mask": headland},
        segment_types=["turn"],
        geo=_FixedMetricGeo(),
        state=_FixedMetricState(),
    )

    assert metrics["track_headland_overlap_pct"] > 0
    assert metrics["track_core_overlap_pct"] == 0


def test_validate_footprints_uses_neutral_support_without_counting_it_as_crop():
    """DECISION-009: neutral support affects bounds, not crop-overlap evidence."""
    from footprint_planner import validate_footprints

    body = np.zeros((80, 120), dtype=np.uint8)
    body[35:45, 60:100] = 255
    support = body.copy()
    support[10:70, 2:20] = 255
    path = [[(10.0, 15.0), (10.0, 65.0)]]

    metrics = validate_footprints(
        path,
        body,
        harvester={
            "cutter_width_m": 0.04, "cutter_forward_offset_m": 0.04,
            "track_width_m": 0.04,
            "track_gauge_m": 0.0, "wheelbase_m": 0.04,
            "track_length_m": 0.04, "turn_radius_m": 0.04,
        },
        support_mask=support,
        segment_types=["turn"],
        geo=_FixedMetricGeo(),
        state=_FixedMetricState(),
    )

    assert metrics["track_outside_field_pct"] == 0
    assert metrics["track_core_overlap_pct"] == 0


def test_validate_footprints_keeps_field_support_uncertain_and_forbidden_separate():
    from footprint_planner import validate_footprints

    class State:
        field_boundary = [(0, 0), (119, 0), (119, 79), (0, 79)]
        mask_offset_x = 0
        mask_offset_y = 0

    body = np.zeros((80, 120), dtype=np.uint8)
    body[35:45, 50:90] = 255
    support = np.zeros_like(body)
    support[:, :30] = 255
    uncertain = np.zeros_like(body)
    uncertain[:, 55:65] = 255
    forbidden = np.zeros_like(body)
    forbidden[:, 58:62] = 255
    path = [[(60.0, 10.0), (60.0, 70.0)]]

    metrics = validate_footprints(
        path,
        body,
        state=State(),
        geo=_FixedMetricGeo(),
        harvester={
                "cutter_width_m": 0.04, "cutter_forward_offset_m": 0.04,
                "track_width_m": 0.04,
            "track_gauge_m": 0.0, "wheelbase_m": 0.04,
            "track_length_m": 0.04, "turn_radius_m": 0.04,
        },
        support_mask=support,
        uncertain_mask=uncertain,
        forbidden_mask=forbidden,
        segment_types=["turn"],
    )

    assert metrics["track_outside_field_pct"] == 0
    assert metrics["track_outside_support_pct"] > 0
    assert metrics["track_uncertain_overlap_pct"] > 0
    assert metrics["track_forbidden_overlap_pct"] > 0
    assert metrics["forbidden_mask_present"] is True


def test_field_trial_profile_turns_support_metric_into_a_hard_constraint():
    from path_planner import validate_path

    class Geo:
        def is_ready(self):
            return True

        def meters_per_pixel(self, _x, _y):
            return 0.1

    class State:
        field_boundary = [(0, 0), (119, 0), (119, 79), (0, 79)]
        mask_offset_x = 0
        mask_offset_y = 0
        harvester_params = {
                "cutter_width_m": 0.2, "cutter_forward_offset_m": 0.5,
                "track_width_m": 0.2,
            "track_gauge_m": 0.4, "wheelbase_m": 1.0,
            "track_length_m": 1.0, "turn_radius_m": 1.0,
        }

    body = np.zeros((80, 120), dtype=np.uint8)
    support = np.zeros_like(body)
    support[:, :30] = 255
    path = [[(60.0, 10.0), (60.0, 70.0)]]
    result = validate_path(
        path, body, 0.0, geo=Geo(), state=State(), segment_types=["turn"],
        config={
            "validation_profile": "field_trial",
            "planning_support_mask": support,
            "max_track_core_overlap_pct": 100.0,
            "min_geometric_coverage_pct": 0.0,
            "max_track_outside_field_pct": 100.0,
        },
    )

    assert result["validation_profile"] == "field_trial"
    assert result["validation_limits_pct"]["track_outside_support"] == 5.0
    assert any("离开语义支撑区" in issue for issue in result["issues"])


def test_field_trial_preflight_can_explicitly_confirm_no_forbidden_area(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = _window()
    window.cfg._raw.setdefault("path_planning", {}).update({
        "work_line_mode": "footprint_optimized",
        "validation_profile": "field_trial",
    })
    window.state.field_boundary = [(0, 0), (19, 0), (19, 19), (0, 19)]
    window.state.forbidden_regions_confirmed = False

    class Geo:
        def is_ready(self):
            return True

    window.geo = Geo()
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)

    assert window._planning_preflight({"planning_support_mask": np.ones((10, 10), dtype=np.uint8)}) is True
    assert window.state.forbidden_regions_confirmed is True


def test_confirming_no_forbidden_area_invalidates_existing_path():
    window = _window()
    window.state.safe_update(
        auto_path_planned=True,
        auto_path_valid=True,
        auto_path=[[(1.0, 1.0), (10.0, 1.0)]],
        forbidden_regions=[],
        forbidden_regions_confirmed=False,
    )

    window._on_confirm_no_forbidden()

    assert window.state.forbidden_regions_confirmed is True
    assert window.state.auto_path_planned is False
    assert window.state.auto_path_valid is False
    assert window.state.auto_path == []


def test_mask_overlay_hides_analysis_context_outside_selected_field(monkeypatch):
    """The display boundary clips copies without discarding analysis context."""
    window = _window()
    window._tif_rgb = np.zeros((30, 40, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    mask = np.full((30, 40), 255, dtype=np.uint8)
    window.state.safe_update(field_boundary=[(10, 5), (30, 5), (30, 25), (10, 25)])
    calls = []
    monkeypatch.setattr(
        window.image_view.viewer,
        "draw_mask_overlay",
        lambda drawn, *_args, **_kwargs: calls.append(drawn),
    )

    window._show_mask_overlay(mask)

    assert len(calls) == 1
    assert calls[0] is not mask
    assert calls[0][15, 20] == 255
    assert calls[0][2, 2] == 0
    assert mask[2, 2] == 255


def test_mask_overlay_draws_headland_and_uncertain_as_separate_layers(monkeypatch):
    """DECISION-009: uncertain residuals must not look like work-body pixels."""
    window = _window()
    window._tif_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    body = np.zeros((80, 120), dtype=np.uint8)
    body[35:45, 20:100] = 255
    headland = np.zeros_like(body)
    headland[10:70, 2:14] = 255
    uncertain = np.zeros_like(body)
    uncertain[15:25, 40:55] = 255
    window.state.mask_result = {
        "processed_mask": body,
        "headland_mask": headland,
        "uncertain_residual_mask": uncertain,
    }
    calls = []

    def fake_draw(mask, x=0, y=0, color=(120, 255, 140), max_dim=2200):
        calls.append((mask, color))

    monkeypatch.setattr(window.image_view.viewer, "draw_mask_overlay", fake_draw)

    window._show_mask_overlay(body)

    assert calls[0][0] is headland
    assert calls[0][1] != calls[1][1]
    assert calls[1][0] is uncertain
    assert calls[2][0] is body
    assert len({calls[0][1], calls[1][1], calls[2][1]}) == 3


def test_mask_display_layers_can_be_composed_without_replacing_the_tif():
    """Raw/processed overlays are independent and all-off returns to the TIF."""
    window = _window()
    window._tif_rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    raw = np.zeros((40, 60), dtype=np.uint8)
    raw[4:18, 4:22] = 255
    processed = np.zeros_like(raw)
    processed[22:36, 30:56] = 255
    window.state.mask_raw = raw
    window.state.mask_result = {"processed_mask": processed}

    window._show_mask_overlay(processed)
    viewer = window.image_view.viewer
    assert set(("raw_mask", "processed_mask")) <= set(viewer._overlay_layers)
    assert all(not item.isVisible() for item in viewer._overlay_layers["raw_mask"])
    assert all(item.isVisible() for item in viewer._overlay_layers["processed_mask"])

    window.image_view.set_display_layer_visibility(
        "raw_mask", True, user=True
    )
    window.image_view.set_display_layer_visibility(
        "processed_mask", False, user=True
    )
    assert all(item.isVisible() for item in viewer._overlay_layers["raw_mask"])
    assert all(not item.isVisible() for item in viewer._overlay_layers["processed_mask"])

    for key in list(window.image_view.display_layer_states()):
        window.image_view.set_display_layer_visibility(key, False, user=True)
    assert viewer._pixmap_item.isVisible() is True
    assert all(not item.isVisible() for item in viewer._overlay_group)
    window.close()


def test_centerline_overlay_keeps_display_offset_in_display_coordinates():
    """DECISION-004/024: display-space centerlines must not rescale their offset."""
    window = _window()
    window._tif_rgb = np.zeros((1000, 1600, 3), dtype=np.uint8)
    window.state.safe_update(
        source_img_h=4000,
        source_img_w=6400,
        display_h=1000,
        display_w=1600,
        mask_offset_x=1200,
        mask_offset_y=300,
    )

    points = window._centerline_display_points({
        "centerline": [(20.0, 30.0), (200.0, 430.0)],
    })

    np.testing.assert_allclose(points, [(1220.0, 330.0), (1400.0, 730.0)])
    window.close()


def test_source_resolution_mask_is_normalised_to_display_coordinates():
    """Source-resolution masks must share the viewer's display coordinate space."""
    window = _window()
    window._tif_rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    window.state.safe_update(
        source_img_h=20,
        source_img_w=40,
        display_h=10,
        display_w=20,
        mask_offset_x=8,
        mask_offset_y=4,
    )
    source_mask = np.zeros((10, 12), dtype=np.uint8)

    normalised, x, y = window._normalise_mask_for_overlay(source_mask)

    assert normalised.shape == (5, 6)
    assert (x, y) == (4, 2)


def test_source_resolution_mask_overlay_stays_inside_base_image():
    window = _window()
    window._tif_rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    window.state.safe_update(
        source_img_h=20,
        source_img_w=40,
        display_h=10,
        display_w=20,
        mask_offset_x=8,
        mask_offset_y=4,
    )
    source_mask = np.full((10, 12), 255, dtype=np.uint8)

    window._show_mask_overlay(source_mask)

    base_bounds = window.image_view.viewer._pixmap_item.sceneBoundingRect()
    overlay_bounds = window.image_view.viewer._overlay_group[0].sceneBoundingRect()
    assert base_bounds.contains(overlay_bounds)


def test_mask_diagnostic_bundle_preserves_stages_and_flags_detached_components(tmp_path):
    import cv2
    import io
    import json
    import zipfile

    from mask_diagnostics import export_mask_diagnostic_bundle

    shape = (100, 140)
    raw = np.zeros(shape, dtype=np.uint8)
    raw[45:55, 20:90] = 255
    regularized = raw.copy()
    body_residual = np.zeros(shape, dtype=np.uint8)
    body_residual[45:55, 108:118] = 255
    gap_added = np.zeros(shape, dtype=np.uint8)
    gap_added[45:55, 90:98] = 255
    processed = cv2.bitwise_or(regularized, body_residual)
    processed = cv2.bitwise_or(processed, gap_added)
    result = {
        "processed_mask": processed,
        "regularized_body_mask": regularized,
        "body_residual_mask": body_residual,
        "gap_repair_added_mask": gap_added,
        "support_clip_removed_mask": np.zeros(shape, dtype=np.uint8),
        "headland_mask": np.zeros(shape, dtype=np.uint8),
        "uncertain_residual_mask": np.zeros(shape, dtype=np.uint8),
        "planning_support_mask": processed,
        "main_angle": 0.0,
        "all_bands": [{"id": 7, "centerline": [(20, 50), (90, 50)]}],
        "diagnostics": {
            "algorithm_version": 4,
            "meters_per_pixel": 0.1,
            "strength": "standard",
        },
    }

    summary = export_mask_diagnostic_bundle(
        str(tmp_path / "diagnostic.zip"),
        raw,
        result,
        config={
            "residual_body_attach_m": 0.12,
            "residual_body_min_attach_ratio": 0.02,
            "band_end_guard_m": 1.0,
        },
        source_image=np.zeros((*shape, 3), dtype=np.uint8),
    )

    assert summary["review_component_count"] >= 2
    with zipfile.ZipFile(summary["path"], "r") as archive:
        assert {"manifest.json", "components.csv", "components.json", "layers.npz", "overview.png"} <= set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        records = json.loads(archive.read("components.json"))
        with np.load(io.BytesIO(archive.read("layers.npz"))) as layers:
            assert "after_residual_merge" in layers.files
            assert "gap_repair_added" in layers.files
            assert np.count_nonzero(layers["gap_repair_added"]) == 80
    detached = next(item for item in records if item["source_layer"] == "body_residual")
    assert "detached_body_residual" in detached["flags"]
    assert detached["nearest_band_id"] == 7


def test_inference_diagnostic_bundle_records_source_model_and_tile_positions(tmp_path):
    import io
    import json
    import zipfile

    from inference_diagnostics import export_inference_diagnostic_bundle

    source = np.full((80, 120, 3), 90, dtype=np.uint8)
    raw = np.zeros((40, 60), dtype=np.uint8)
    raw[8:28, 12:48] = 255
    summary = export_inference_diagnostic_bundle(
        str(tmp_path / "识别001.zip"),
        source,
        raw,
        metadata={
            "run_label": "识别001",
            "source_image": {"path": "E:/field/result.tif", "sha256": "source"},
            "model": {"name": "tiller12.pt", "path": "E:/models/tiller12.pt", "sha256": "model"},
            "image_position": {"source_crop_window_xywh": [100, 200, 600, 400]},
            "inference": {"algorithm_version": "test-v1", "config": {"overlap": 0.4}},
        },
        mask_offset=(20, 10),
        field_boundary=[(22, 12), (76, 12), (76, 45), (22, 45)],
        tile_trace=[{
            "tile_index": 0,
            "row": 0,
            "column": 0,
            "source_window_xywh": [100, 200, 640, 640],
            "output_window_xywh": [0, 0, 32, 32],
            "accepted_instance_count": 2,
            "max_confidence": 0.83,
            "tile_probability_pixels": 120,
            "skipped_nodata": False,
        }],
    )

    with zipfile.ZipFile(summary["path"], "r") as archive:
        assert {
            "manifest.json", "source_preview.png", "raw_overlay.png",
            "field_clipped_overlay.png", "tile_grid.png", "raw_mask.png",
            "raw_mask.npz", "tile_trace.csv", "tile_trace.json",
        } <= set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema"] == "reui.inference-diagnostic.v1"
        assert manifest["run_label"] == "识别001"
        assert manifest["source_image"]["path"].endswith("result.tif")
        assert manifest["model"]["name"] == "tiller12.pt"
        assert manifest["image_position"]["source_crop_window_xywh"] == [100, 200, 600, 400]
        assert manifest["tile_summary"]["primary"]["tile_count"] == 1
        with np.load(io.BytesIO(archive.read("raw_mask.npz"))) as arrays:
            assert arrays["raw_mask"].shape == raw.shape


def test_inference_diagnostic_field_clipped_overlay_uses_mask_offset(tmp_path):
    import cv2
    import json
    import zipfile

    from inference_diagnostics import export_inference_diagnostic_bundle

    source = np.full((12, 16, 3), 90, dtype=np.uint8)
    raw = np.full((6, 8), 255, dtype=np.uint8)
    output = export_inference_diagnostic_bundle(
        str(tmp_path / "识别002.zip"),
        source,
        raw,
        metadata={"run_label": "识别002"},
        mask_offset=(4, 3),
        field_boundary=[(4, 3), (11, 3), (11, 9), (4, 9)],
    )

    with zipfile.ZipFile(output["path"], "r") as archive:
        overlay = cv2.imdecode(
            np.frombuffer(archive.read("field_clipped_overlay.png"), np.uint8),
            cv2.IMREAD_COLOR,
        )
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["image_position"]["mask_offset_xy_display"] == [4, 3]
    # A point outside the selected field remains the original gray pixel.
    assert np.all(overlay[1, 1] == [90, 90, 90])
    # A point inside the offset mask receives the green blend.
    assert not np.all(overlay[6, 7] == [90, 90, 90])


def test_mask_diagnostic_manifest_includes_stage_provenance_and_run_label(tmp_path):
    import cv2
    import json
    import zipfile
    from mask_diagnostics import export_mask_diagnostic_bundle

    shape = (30, 40)
    raw = np.zeros(shape, dtype=np.uint8)
    raw[8:22, 4:34] = 255
    result = {
        "processed_mask": raw.copy(),
        "regularized_body_mask": raw.copy(),
        "headland_mask": np.zeros(shape, dtype=np.uint8),
        "uncertain_residual_mask": np.zeros(shape, dtype=np.uint8),
        "planning_support_mask": raw.copy(),
        "main_angle": 0.0,
        "all_bands": [],
        "diagnostics": {"meters_per_pixel": 0.1, "algorithm_version": 17},
    }
    summary = export_mask_diagnostic_bundle(
        str(tmp_path / "掩膜优化001.zip"),
        raw,
        result,
        metadata={
            "run_label": "掩膜优化001",
            "source_image": {"path": "E:/field/result.tif"},
            "model": {"name": "tiller12.pt"},
        },
        source_image=np.zeros((*shape, 3), dtype=np.uint8),
    )
    with zipfile.ZipFile(summary["path"], "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["stage"] == "mask_optimization"
        assert manifest["run_label"] == "掩膜优化001"
        assert manifest["provenance"]["model"]["name"] == "tiller12.pt"


def test_mask_diagnostic_button_requires_processed_mask():
    window = _window()
    assert window.task_panel.inference_diagnostic_btn.isEnabled() is False
    assert window.task_panel.mask_diagnostic_btn.isEnabled() is False

    window.state.inference_done = True
    window.state.mask_raw = np.ones((20, 20), dtype=np.uint8)
    window.state.mask_processed = True
    window.state.mask_result = {"processed_mask": np.ones((20, 20), dtype=np.uint8)}
    window._refresh_state()

    assert window.task_panel.inference_diagnostic_btn.isEnabled() is True
    assert window.task_panel.mask_diagnostic_btn.isEnabled() is True


def test_diagnostic_export_path_continues_numbering_in_selected_directory(tmp_path):
    window = _window()
    window._tif_path = str(tmp_path / "result.tif")
    (tmp_path / "识别001.zip").write_bytes(b"existing")
    assert os.path.basename(window._next_diagnostic_default_path("识别")) == "识别002.zip"

    selected_dir = tmp_path / "desktop"
    selected_dir.mkdir()
    (selected_dir / "识别001.zip").write_bytes(b"existing")
    selected = window._reserve_diagnostic_path(
        str(selected_dir / "识别001.zip"), "识别"
    )
    assert os.path.basename(selected) == "识别002.zip"
    (selected_dir / "识别002.zip").write_bytes(b"existing")
    assert (
        os.path.abspath(window._next_diagnostic_default_path("识别"))
        == os.path.abspath(str(selected_dir / "识别003.zip"))
    )


def _synthetic_terminal_headland():
    import cv2
    raw = np.zeros((160, 240), dtype=np.uint8)
    rebuilt = np.zeros_like(raw)
    for y in range(24, 140, 18):
        raw[y:y + 7, 20:180] = 255
        rebuilt[y:y + 7, 20:205] = 255
    raw[18:146, 180:205] = 255
    headland_seed = cv2.bitwise_and(raw, cv2.bitwise_not(rebuilt))
    return raw, rebuilt, headland_seed


def test_terminal_headland_corridor_removes_periodic_body_islands():
    import cv2
    from row_geometry import resolve_terminal_headland_overlap

    raw, rebuilt, headland_seed = _synthetic_terminal_headland()
    clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw, rebuilt, headland_seed, 0.0, 0.1,
        {
            "headland_corridor_close_m": 1.0,
            "headland_corridor_terminal_depth_m": 3.0,
            "headland_corridor_min_span_ratio": 0.5,
            "headland_corridor_max_body_cut_m": 3.0,
            "headland_corridor_min_seed_ratio": 0.15,
        },
    )

    assert clean[27, 170] == 255
    assert clean[27, 190] == 0
    assert headland[27, 190] == 255
    assert corridor[27, 190] == 255
    assert diagnostics["removed_body_px"] > 0
    assert diagnostics["corridor_count"] == 1


def test_terminal_headland_corridor_envelope_moves_supported_body_islands_to_headland():
    import cv2
    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((170, 240), dtype=np.uint8)
    body = np.zeros_like(raw)
    # Normal longitudinal body before the terminal headland.
    body[42:58, 30:180] = 255
    body[84:100, 30:180] = 255
    # A strong terminal cross-row headland strip with two body islands inside it.
    raw[20:145, 180:210] = 255
    body[42:58, 185:205] = 255
    body[84:100, 185:205] = 255
    headland_seed = cv2.bitwise_and(raw, cv2.bitwise_not(body))

    clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        headland_seed,
        0.0,
        0.1,
        {
            "headland_corridor_close_m": 0.3,
            "headland_corridor_terminal_depth_m": 3.0,
            "headland_corridor_min_span_ratio": 0.5,
            "headland_corridor_max_body_cut_m": 3.0,
            "headland_corridor_min_seed_ratio": 0.15,
        },
    )

    assert clean[50, 170] == 255
    assert clean[50, 190] == 0
    assert headland[50, 190] == 255
    assert corridor[50, 190] == 255
    assert diagnostics["envelope_added_px"] > 0


def test_terminal_headland_corridor_uses_seed_depth_not_full_body_cut_limit():
    import cv2
    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((170, 260), dtype=np.uint8)
    body = np.zeros_like(raw)
    for y0 in (36, 76, 116):
        body[y0:y0 + 10, 30:216] = 255
        body[y0:y0 + 10, 224:238] = 255
    raw[:] = body
    raw[24:150, 220:246] = 255
    headland_seed = cv2.bitwise_and(raw, cv2.bitwise_not(body))

    clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        headland_seed,
        0.0,
        0.1,
        {
            "headland_corridor_close_m": 0.8,
            "headland_corridor_terminal_depth_m": 4.0,
            "headland_corridor_max_body_cut_m": 4.0,
            "headland_corridor_endpoint_margin_m": 0.2,
            "headland_corridor_min_span_ratio": 0.5,
            "headland_corridor_min_seed_ratio": 0.10,
        },
    )

    assert clean[40, 205] == 255
    assert clean[40, 230] == 0
    assert headland[40, 230] == 255
    assert corridor[40, 230] == 255
    assert diagnostics["corridor_count"] == 1


def test_headland_corridor_does_not_cut_internal_transverse_structure():
    import cv2
    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((160, 240), dtype=np.uint8)
    rebuilt = np.zeros_like(raw)
    for y in range(24, 140, 18):
        raw[y:y + 7, 20:220] = 255
        rebuilt[y:y + 7, 20:220] = 255
    raw[18:146, 105:118] = 255
    headland_seed = cv2.bitwise_and(raw, cv2.bitwise_not(rebuilt))

    clean, _headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw, rebuilt, headland_seed, 0.0, 0.1,
        {"headland_corridor_terminal_depth_m": 2.0},
    )

    assert np.array_equal(clean, rebuilt)
    assert np.count_nonzero(corridor) == 0
    assert diagnostics["corridor_count"] == 0


def test_terminal_separator_gap_disables_raw_row_follow_and_body_cut():
    """DECISION-009: a clear wide-row gap owns the terminal wall."""
    import cv2
    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((160, 240), dtype=np.uint8)
    body = np.zeros_like(raw)
    for y in range(24, 140, 18):
        raw[y:y + 7, 20:160] = 255
        body[y:y + 7, 20:160] = 255
    # Leave a visible non-mask separator before the dense terminal strip.
    raw[18:146, 175:205] = 255
    seed = cv2.bitwise_and(raw, cv2.bitwise_not(body))

    clean, _headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        seed,
        0.0,
        0.1,
        {
            "headland_corridor_terminal_depth_m": 3.0,
            "headland_corridor_max_body_cut_m": 3.0,
            "headland_corridor_min_span_ratio": 0.5,
            "headland_corridor_min_seed_ratio": 0.1,
        },
    )

    assert diagnostics["terminal_separator"]["sides"] == ["end"]
    assert diagnostics["terminal_shape"]["follow_raw_terminal"] is False
    assert diagnostics["terminal_shape"]["effective_max_body_cut_m"] == 0.0
    assert clean[60, 158] == 255
    assert corridor[60, 165] == 0
    assert corridor[60, 175] == 255


def test_headland_corridor_rejects_sparse_terminal_evidence():
    from row_geometry import resolve_terminal_headland_overlap

    raw, rebuilt, _headland_seed = _synthetic_terminal_headland()
    sparse_seed = np.zeros_like(raw)
    sparse_seed[20:42, 182:200] = 255

    clean, _headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw, rebuilt, sparse_seed, 0.0, 0.1,
        {"headland_corridor_min_span_ratio": 0.5},
    )

    assert np.array_equal(clean, rebuilt)
    assert np.count_nonzero(corridor) == 0
    assert diagnostics["corridor_count"] == 0


def test_headland_corridor_recovers_dense_second_terminal_without_angle_seed():
    """DECISION-009: connected terminal crop still supplies dense end evidence."""
    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((170, 260), dtype=np.uint8)
    body = np.zeros_like(raw)
    for y0 in (36, 76, 116):
        body[y0:y0 + 10, 30:181] = 255
    raw[:] = body
    raw[24:150, 180:221] = 255
    no_orientation_seed = np.zeros_like(raw)

    clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        no_orientation_seed,
        0.0,
        0.1,
        {
            "headland_terminal_dense_seed_min_occupancy": 0.80,
            "headland_terminal_dense_seed_min_depth_m": 0.2,
            "headland_corridor_min_span_ratio": 0.5,
        },
    )

    assert diagnostics["terminal_dense_seed"]["sides_added"] == ["end"]
    assert diagnostics["corridor_count"] == 1
    assert headland[80, 200] == 255
    assert corridor[80, 200] == 255
    assert clean[80, 180] == 0


def test_headland_dense_seed_does_not_relabel_parallel_row_tails():
    """DECISION-009: sparse longitudinal tails are body, not a headland."""
    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((170, 260), dtype=np.uint8)
    body = np.zeros_like(raw)
    for y0 in (36, 76, 116):
        body[y0:y0 + 10, 30:181] = 255
        raw[y0:y0 + 10, 30:221] = 255

    clean, _headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        np.zeros_like(raw),
        0.0,
        0.1,
        {"headland_terminal_dense_seed_min_occupancy": 0.80},
    )

    assert diagnostics["terminal_dense_seed"]["sides_added"] == []
    assert diagnostics["corridor_count"] == 0
    assert np.array_equal(clean, body)
    assert np.count_nonzero(corridor) == 0


def test_terminal_headland_follows_curved_crop_support_not_rectangle():
    """DECISION-009: an accepted headland expands along crop endpoints/support."""
    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((180, 260), dtype=np.uint8)
    body = np.zeros_like(raw)
    reference = np.zeros_like(raw)
    seed = np.zeros_like(raw)
    for row_y in range(25, 156):
        endpoint = int(round(150.0 + 0.25 * row_y))
        raw[row_y, endpoint:endpoint + 40] = 255
        seed[row_y, endpoint:endpoint + 5] = 255
    for row_y in range(30, 151, 20):
        endpoint = int(round(150.0 + 0.25 * row_y))
        body[row_y:row_y + 6, 25:endpoint + 1] = 255
        reference[row_y:row_y + 6, 25:endpoint + 1] = 255
    raw = np.maximum(raw, body)

    _clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        seed,
        0.0,
        0.1,
        {
            "headland_corridor_close_m": 0.5,
            "headland_corridor_terminal_depth_m": 4.0,
            "headland_corridor_min_span_ratio": 0.6,
            "headland_corridor_min_seed_ratio": 0.1,
            "headland_terminal_follow_max_depth_m": 5.0,
            "headland_terminal_wall_smooth_m": 0.5,
        },
        axial_reference_mask=reference,
    )

    # The same x is outside the shallow upper terminal but inside the lower,
    # proving the corridor follows the sloped crop support instead of a box.
    assert corridor[40, 210] == 0
    assert corridor[140, 210] == 255
    assert headland[140, 210] == 255
    assert diagnostics["terminal_shape"]["accepted_sides"] == ["end"]
    assert diagnostics["terminal_shape"]["raw_support_added_px"] > 0
    assert diagnostics["terminal_shape"]["wall_source"] == (
        "regularized_longitudinal_body_endpoints"
    )


def test_terminal_headland_wall_ignores_short_internal_work_fragment():
    """DECISION-009: a short interior row cannot become a terminal wall."""
    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((180, 240), dtype=np.uint8)
    reference = np.zeros_like(raw)
    reference[20:55, 30:181] = 255
    reference[120:160, 30:181] = 255
    # Only a short fragment exists at the planting-region seam. The former
    # per-row endpoint rule treated x=90 as the field start and painted it as
    # a long yellow headland strip.
    reference[55:120, 90:151] = 255
    body = reference.copy()
    seed = np.zeros_like(raw)
    seed[20:160, 20:31] = 255
    raw = np.maximum(reference, seed)

    clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        seed,
        0.0,
        0.1,
        {
            "headland_corridor_close_m": 0.5,
            "headland_corridor_terminal_depth_m": 2.0,
            "headland_corridor_min_span_ratio": 0.6,
            "headland_terminal_follow_max_depth_m": 4.5,
            "headland_terminal_wall_min_axial_span_ratio": 0.72,
        },
        axial_reference_mask=reference,
    )

    assert corridor[85, 100] == 0
    assert headland[85, 100] == 0
    assert clean[85, 100] == 255
    assert corridor[40, 25] == 255
    assert diagnostics["terminal_shape"]["wall_rejected_short_row_count"] > 0


def test_terminal_headland_wall_rejects_long_span_inward_endpoint_wedge():
    """DECISION-009: a local inward endpoint defect cannot bend the field wall."""
    import cv2

    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((180, 260), dtype=np.uint8)
    body = np.zeros_like(raw)
    reference = np.zeros_like(raw)
    seed = np.zeros_like(raw)
    body[20:160, 30:181] = 255
    reference[20:160, 30:181] = 255
    # This 3 m inward notch remains longer than the old 72% span gate, so the
    # former endpoint interpolation treated it as a trusted wall and moved
    # supported main-work pixels into the terminal corridor.
    reference[70:112, 151:181] = 0
    seed[20:160, 181:221] = 255
    # The terminal candidate itself contains the same inward wedge.  A robust
    # reference wall must bound both the generated corridor and the accepted
    # seed; otherwise the candidate edge silently reintroduces the triangle.
    seed[70:112, 151:181] = 255
    raw = cv2.bitwise_or(body, seed)

    clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        seed,
        0.0,
        0.1,
        {
            "headland_corridor_close_m": 0.5,
            "headland_corridor_terminal_depth_m": 4.0,
            "headland_corridor_min_span_ratio": 0.6,
            "headland_corridor_min_seed_ratio": 0.1,
            "headland_terminal_follow_max_depth_m": 5.0,
            "headland_terminal_wall_outlier_tolerance_m": 0.8,
        },
        axial_reference_mask=reference,
    )

    assert corridor[90, 165] == 0
    assert headland[90, 165] == 0
    assert clean[90, 165] == 255
    assert corridor[90, 200] == 255
    assert diagnostics["terminal_shape"]["wall_outlier_replaced_row_count"] > 0


def test_terminal_headland_restores_quantized_wall_from_field_end_edge():
    """DECISION-009: a common axial clip must not flatten a sloped field end."""
    import cv2

    from row_geometry import resolve_terminal_headland_overlap

    raw = np.zeros((180, 260), dtype=np.uint8)
    reference = np.zeros_like(raw)
    reference[20:160, 80:201] = 255
    body = reference.copy()
    seed = np.zeros_like(raw)
    seed[20:160, 40:80] = 255
    raw = cv2.bitwise_or(reference, seed)
    # The selected field end spans the row lattice and preserves a 0.2 px/px
    # physical slope.  The regularized reference above is deliberately flat,
    # reproducing the quantization caused by one common axial body interval.
    field_polygon = np.asarray(
        [(30.0, 20.0), (58.0, 160.0), (240.0, 160.0), (240.0, 20.0)],
        dtype=np.float64,
    )

    clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        seed,
        0.0,
        0.1,
        {
            "band_end_trim_m": 0.10,
            "headland_corridor_close_m": 0.5,
            "headland_corridor_terminal_depth_m": 5.0,
            "headland_corridor_min_span_ratio": 0.6,
            "headland_corridor_min_seed_ratio": 0.1,
            "headland_terminal_follow_max_depth_m": 5.0,
            "headland_terminal_wall_smooth_m": 0.1,
            "headland_terminal_field_edge_min_span_ratio": 0.6,
            "headland_terminal_field_edge_max_abs_slope": 0.25,
            "headland_terminal_wall_quantized_mad_px": 0.5,
        },
        axial_reference_mask=reference,
        field_polygon=field_polygon,
    )

    # The same x lies outside the restored upper wall but inside the lower
    # wall.  A flat x=80 wall would classify both as headland.
    assert corridor[30, 75] == 0
    assert headland[30, 75] == 0
    assert corridor[150, 75] == 255
    assert headland[150, 75] == 255
    assert clean[30, 80] == 255
    shape = diagnostics["terminal_shape"]
    assert "start" in shape["field_slope_applied_sides"]
    assert shape["field_slope_hints"]["start"][
        "slope_px_per_cross_row_px"
    ] == pytest.approx(0.2)
    assert shape["restored_band_end_trim_m"] == pytest.approx(0.1)


def test_terminal_headland_classifies_regions_before_exposing_seed():
    """DECISION-009: rejected interior region seeds never become yellow headland."""
    import cv2
    from row_geometry import resolve_terminal_headland_overlap

    raw, body, terminal_seed = _synthetic_terminal_headland()
    # Add a transverse interior residual at a planting-region seam.  It is
    # intentionally large enough to survive morphology but does not touch a
    # global terminal envelope.
    raw[30:135, 96:108] = 255
    interior_seed = np.zeros_like(raw)
    interior_seed[30:135, 96:108] = 255
    seed = cv2.bitwise_or(terminal_seed, interior_seed)

    _clean, headland, corridor, diagnostics = resolve_terminal_headland_overlap(
        raw,
        body,
        seed,
        0.0,
        0.1,
        {
            "headland_corridor_close_m": 0.5,
            "headland_corridor_terminal_depth_m": 3.0,
            "headland_corridor_min_span_ratio": 0.5,
            "headland_corridor_min_seed_ratio": 0.10,
            "headland_terminal_region_min_support_ratio": 0.55,
        },
        axial_reference_mask=body,
    )

    assert diagnostics["region_partition"]["region_count"] >= 2
    assert diagnostics["region_partition"]["rejected_fragment_count"] >= 1
    assert np.count_nonzero(corridor[:, 96:108]) == 0
    assert np.count_nonzero(headland[:, 96:108]) == 0


def test_band_geometry_is_clipped_to_resolved_body_component():
    from row_geometry import clip_bands_to_body

    body = np.zeros((80, 240), dtype=np.uint8)
    body[30:42, 20:175] = 255
    body[30:42, 190:205] = 255
    band = {
        "id": 3,
        "band_type": "wide",
        "contour": np.asarray([[20, 30], [205, 30], [205, 41], [20, 41]]),
        "centerline": [(20.0, 35.0), (80.0, 35.0), (140.0, 35.0), (195.0, 35.0)],
    }

    clipped, dropped = clip_bands_to_body([band], body)

    assert dropped == []
    assert len(clipped) == 1
    assert max(point[0] for point in clipped[0]["centerline"]) < 180
    assert np.max(np.asarray(clipped[0]["contour"])[:, 0]) < 180
