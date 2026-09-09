import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow
from state import AppState


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    return window


class _ViewerMouseEvent:
    def __init__(self, position, *, button=Qt.NoButton, buttons=Qt.NoButton):
        self._position = position
        self._button = button
        self._buttons = buttons
        self.accepted = False

    def pos(self):
        return self._position

    def button(self):
        return self._button

    def buttons(self):
        return self._buttons

    def accept(self):
        self.accepted = True


def test_field_polygon_finish_is_global_undo_redo_operation():
    window = _window()
    window._field_pts = [(1, 1), (30, 1), (30, 30), (1, 30)]

    window._finish_field_polygon()

    assert window.state.field_boundary == [(1, 1), (30, 1), (30, 30), (1, 30)]
    assert window.state.undo_stack
    assert window.top_toolbar._history_buttons["UNDO"].isEnabled()

    window._on_undo()
    assert window.state.field_boundary == []
    assert window.top_toolbar._history_buttons["REDO"].isEnabled()

    window._on_redo()
    assert window.state.field_boundary == [(1, 1), (30, 1), (30, 30), (1, 30)]


def test_turn_strategy_change_is_global_undo_redo_operation():
    window = _window()
    window.state.turn_strategy = "bow"

    window._on_turn_strategy_changed("semicircle")

    assert window.state.turn_strategy == "semicircle"
    assert window.state.undo_stack

    window._on_undo()
    assert window.state.turn_strategy == "bow"

    window._on_redo()
    assert window.state.turn_strategy == "semicircle"


def test_entry_exit_point_placement_is_global_undo_redo_operation():
    window = _window()
    window.state.entry_exit_mode = True
    window._entry_exit_click_idx = 0

    window._place_entry_exit_point(10, 20)

    assert window.state.entry_point == (10, 20)
    assert window.state.entry_point_locked is True
    assert window.state.undo_stack

    window._on_undo()
    assert window.state.entry_point is None
    assert window.state.entry_point_locked is False

    window._on_redo()
    assert window.state.entry_point == (10, 20)
    assert window.state.entry_point_locked is True


def test_project_history_persists_independent_mask_stage_snapshots(tmp_path, monkeypatch):
    """DECISION-025: disk history keeps pre-operation state and dense masks."""
    import cache
    from provenance import file_sha256

    source = tmp_path / "field.tif"
    source.write_bytes(b"stable-geotiff-fixture")
    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))

    state = AppState()
    state.tif_path = str(source)
    state.source_sha256 = file_sha256(str(source))

    state.save_undo("田块圈选")
    state.field_boundary = [(1.0, 1.0), (8.0, 1.0), (8.0, 8.0)]
    state.field_area_m2 = 24.0

    raw_mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    state.mask_raw = raw_mask
    state.inference_original_mask = raw_mask
    state.inference_done = True
    state.save_undo("掩膜处理")

    processed_mask = np.array([[0, 255], [255, 255]], dtype=np.uint8)
    state.mask_processed = True
    state.mask_result = {
        "processed_mask": processed_mask,
        "planning_support_mask": processed_mask.copy(),
        "main_angle": 12.5,
    }
    state.save_undo("路径规划")
    state.path_points = [(1.0, 1.0), (8.0, 8.0)]
    state.path_status = [1, 1]
    state.auto_path_planned = True

    assert cache.save_project_state(str(source), state, stage="history-test") is True
    history = cache.load_project_history(
        str(source), expected_source_sha256=state.source_sha256
    )

    assert [item["label"] for item in history["undo_stack"]] == [
        "田块圈选",
        "掩膜处理",
        "路径规划",
    ]
    before_boundary = history["undo_stack"][0]["values"]
    before_processing = history["undo_stack"][1]["values"]
    assert before_boundary["field_boundary"] == []
    assert before_processing["field_boundary"] == state.field_boundary
    assert before_processing["mask_processed"] is False
    np.testing.assert_array_equal(before_processing["mask_raw"], raw_mask)
    before_path = history["undo_stack"][2]["values"]
    assert before_path["mask_processed"] is True
    np.testing.assert_array_equal(
        before_path["mask_result"]["processed_mask"], processed_mask
    )

    # The raw mask is shared by two checkpoints and the processed/support
    # layers have identical bytes, so only two dense array artifacts exist.
    artifacts = list(
        (tmp_path / "cache").glob("*/history/artifacts/*.npz")
    )
    assert len(artifacts) == 2

    # Later mutations of the live state must not alter the stored checkpoint.
    raw_mask[:, :] = 7
    reloaded = cache.load_project_history(
        str(source), expected_source_sha256=state.source_sha256
    )
    np.testing.assert_array_equal(
        reloaded["undo_stack"][1]["values"]["mask_raw"],
        np.array([[0, 255], [255, 0]], dtype=np.uint8),
    )


def test_persisted_history_undoes_one_stage_at_a_time_after_restart(tmp_path, monkeypatch):
    """DECISION-025: reopening a project restores the ordered undo chain."""
    import cache
    from provenance import file_sha256

    source = tmp_path / "field.tif"
    source.write_bytes(b"stable-geotiff-fixture")
    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))

    original = AppState()
    original.tif_path = str(source)
    original.source_sha256 = file_sha256(str(source))
    original.save_undo("田块圈选")
    original.field_boundary = [(1.0, 1.0), (8.0, 1.0), (8.0, 8.0)]
    original.field_area_m2 = 24.0

    original.mask_raw = np.ones((3, 3), dtype=np.uint8) * 255
    original.inference_original_mask = original.mask_raw
    original.inference_done = True
    original.save_undo("掩膜处理")
    original.mask_processed = True
    original.mask_result = {
        "processed_mask": np.eye(3, dtype=np.uint8) * 255,
        "main_angle": 8.0,
    }
    assert cache.save_project_state(str(source), original, stage="restart-test")

    history = cache.load_project_history(
        str(source), expected_source_sha256=original.source_sha256
    )
    reopened = AppState()
    reopened.tif_path = str(source)
    reopened.source_sha256 = original.source_sha256
    reopened.field_boundary = list(original.field_boundary)
    reopened.field_area_m2 = original.field_area_m2
    reopened.mask_raw = original.mask_raw.copy()
    reopened.inference_original_mask = reopened.mask_raw
    reopened.inference_done = True
    reopened.mask_processed = True
    reopened.mask_result = {
        "processed_mask": original.mask_result["processed_mask"].copy(),
        "main_angle": 8.0,
    }
    reopened.undo_stack = history["undo_stack"]
    reopened.redo_stack = history["redo_stack"]

    assert reopened.undo() is True
    assert reopened.field_boundary == original.field_boundary
    assert reopened.inference_done is True
    assert reopened.mask_processed is False
    np.testing.assert_array_equal(reopened.mask_raw, original.mask_raw)

    assert reopened.undo() is True
    assert reopened.field_boundary == []
    assert reopened.inference_done is False
    assert reopened.mask_raw is None

    assert reopened.redo() is True
    assert reopened.field_boundary == original.field_boundary
    assert reopened.inference_done is True
    assert reopened.mask_processed is False


def test_project_history_rejects_a_changed_source_image(tmp_path, monkeypatch):
    """DECISION-002/025: history is bound to the exact source GeoTIFF."""
    import cache
    from provenance import file_sha256

    source = tmp_path / "field.tif"
    source.write_bytes(b"source-version-one")
    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))

    state = AppState()
    state.tif_path = str(source)
    state.source_sha256 = file_sha256(str(source))
    state.save_undo("田块圈选")
    state.field_boundary = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)]
    assert cache.save_project_state(str(source), state, stage="source-binding-test")

    source.write_bytes(b"source-version-two")
    assert cache.load_project_history(str(source)) == {
        "undo_stack": [],
        "redo_stack": [],
        "messages": ["历史记录与当前源影像不一致，已停止恢复"],
    }


def test_cache_restore_installs_persisted_operation_stacks(monkeypatch):
    """DECISION-025: the GUI receives cross-session undo and redo stacks."""
    window = _window()
    checkpoint = AppState().capture_system_snapshot("田块圈选")
    checkpoint["history_id"] = "a" * 32
    checkpoint["created_at"] = "2026-09-01T10:00:00+08:00"
    redo = AppState().capture_system_snapshot("恢复：田块圈选")
    redo["history_id"] = "b" * 32
    redo["created_at"] = "2026-09-01T10:01:00+08:00"
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)

    window._restore_cache_payload(
        {
            "path": "field.tif",
            "saved": {
                "turn_strategy": "contour",
                "turn_strategy_user_selected": True,
                "harvester_params": {},
            },
            "processed_cache_valid": False,
            "path_cache_valid": False,
            "history": {
                "undo_stack": [checkpoint],
                "redo_stack": [redo],
                "messages": [],
            },
        }
    )

    assert [item["history_id"] for item in window.state.undo_stack] == ["a" * 32]
    assert [item["history_id"] for item in window.state.redo_stack] == ["b" * 32]


def test_older_same_project_save_cannot_overwrite_newer_state_or_history(
    tmp_path, monkeypatch
):
    """DECISION-025: a delayed checkpoint cannot roll disk history backward."""
    import cache

    class DeferredThread:
        targets = []

        def __init__(self, target, **_kwargs):
            self.__class__.targets.append(target)

        def start(self):
            pass

    source = tmp_path / "field.tif"
    source.write_bytes(b"stable-geotiff-fixture")
    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(cache.threading, "Thread", DeferredThread)
    cache._PENDING_STATE_SAVES.clear()
    cache._ACTIVE_STATE_SAVERS.clear()

    state = AppState()
    state.tif_path = str(source)
    state.field_area_m2 = 10.0
    state.save_undo("旧操作")
    cache.request_project_state_save(str(source), state, stage="old")

    state.field_area_m2 = 99.0
    state.undo_stack.clear()
    state.save_undo("新操作")
    assert cache.save_project_state(str(source), state, stage="new")
    DeferredThread.targets[-1]()

    restored = cache.load_project_state(str(source))
    history = cache.load_project_history(
        str(source),
        expected_save_session_id=restored["save_session_id"],
        expected_save_revision=restored["save_revision"],
        expected_project_generation=restored["project_generation"],
    )
    assert restored["field_area_m2"] == 99.0
    assert [item["label"] for item in history["undo_stack"]] == ["新操作"]


def test_project_history_reclaims_unreferenced_array_artifacts(tmp_path, monkeypatch):
    """DECISION-025: bounded history also bounds content-addressed arrays."""
    import cache
    from provenance import file_sha256

    source = tmp_path / "field.tif"
    source.write_bytes(b"stable-geotiff-fixture")
    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))

    state = AppState()
    state.tif_path = str(source)
    state.source_sha256 = file_sha256(str(source))
    state.mask_raw = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    state.save_undo("掩膜处理")
    assert cache.save_project_state(str(source), state, stage="history-cleanup")

    artifacts_dir = tmp_path / "cache" / next(
        path.name for path in (tmp_path / "cache").iterdir()
    ) / "history" / "artifacts"
    referenced = list(artifacts_dir.glob("*.npz"))
    assert len(referenced) == 1

    orphan = artifacts_dir / ("f" * 64 + ".npz")
    np.savez_compressed(orphan, array=np.array([1, 2, 3], dtype=np.uint8))
    assert orphan.is_file()

    assert cache.save_project_state(str(source), state, stage="history-cleanup-again")

    assert not orphan.exists()
    assert referenced[0].is_file()


def test_recommended_service_point_can_be_dragged_directly_and_invalidates_path():
    """DECISION-007: 推荐点可直接拖动，旧路径随服务点变化失效。"""
    window = _window()
    window.state.entry_point = (10.0, 20.0)
    window.state.exit_point = (90.0, 60.0)
    window.state.entry_point_locked = False
    window.state.auto_path_planned = True
    window.state.auto_path_valid = True
    window.state.path_provenance = {"fingerprint": "old-path"}
    window._service_points_visible = True

    window._on_viewer_mouse_pressed(10, 20, 1)
    assert window.state._drag_entry_exit == ("entry", 0)
    window._on_viewer_mouse_moved(24, 32, 1)
    window._on_viewer_mouse_released(24, 32, 1)

    assert window.state._drag_entry_exit is None
    assert window.state.entry_point == (24.0, 32.0)
    assert window.state.entry_point_locked is True
    assert window.state.auto_path_planned is False
    assert window.state.path_provenance == {}
    assert len(window.state.undo_stack) == 1

    window._on_undo()
    assert window.state.entry_point == (10.0, 20.0)
    assert window.state.entry_point_locked is False
    assert window.state.auto_path_planned is True
    assert window.state.path_provenance == {"fingerprint": "old-path"}


def test_service_point_drag_preserves_free_2d_position_and_schedules_replan():
    """DECISION-007: drag previews the connector without snapping the point."""
    from state import AutoPathSegment, PathPoint

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def pixels_per_meter(_x, _y):
            return 5.0

    window = _window()
    window.geo = Geo()
    window.state.field_boundary = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)
    ]
    window.state.mask_result = {
        "main_angle": 0.0,
        "processed_mask": np.ones((60, 100), dtype=np.uint8) * 255,
    }
    window.state.mask_processed = True
    window.state.harvester_params_confirmed = True
    window.state.entry_point = (10.0, 30.0)
    window.state.exit_point = (90.0, 30.0)
    window.state.entry_point_locked = True
    window.state.exit_point_locked = True
    window.state.auto_path_planned = True
    window.state.auto_path_segments = [
        AutoPathSegment(
            points=[PathPoint(20.0, 30.0), PathPoint(80.0, 30.0)],
            segment_type="work",
        )
    ]
    window._service_points_visible = True

    assert window._begin_service_point_drag(10, 30)
    window._move_service_point_drag(50, 45)

    # The point follows the cursor in both axes.  Only the confirmed field
    # boundary is a hard constraint; terminal/headland projection is preview
    # geometry, not a coordinate snap.
    assert window.state.entry_point == (50.0, 45.0)
    assert window._service_point_drag_last_correction["terminal_corrected"] is False
    assert window._service_point_drag_preview_items

    window._finish_service_point_drag(50, 45)
    assert window.state.auto_path_planned is False
    assert window._stale_path_preview is not None
    assert window._stale_path_preview.get("stale_preview") is True
    assert window._stale_path_preview.get("full_path")
    assert window._service_point_replan_timer.isActive()
    window._service_point_replan_timer.stop()


def test_service_point_drag_does_not_snap_to_headland_corridor():
    """DECISION-007/009: a detected corridor must not lock the point to one axis."""
    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def pixels_per_meter(_x, _y):
            return 5.0

    window = _window()
    window.geo = Geo()
    window.state.field_boundary = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)
    ]
    corridor = np.zeros((60, 100), dtype=np.uint8)
    corridor[:, :12] = 255
    corridor[:, 88:] = 255
    window.state.mask_result = {
        "main_angle": 0.0,
        "processed_mask": np.ones((60, 100), dtype=np.uint8) * 255,
        "headland_corridor_mask": corridor,
    }

    point, correction = window._constrain_service_point(
        ("entry", 0), (40.0, 45.0)
    )

    assert point == (40.0, 45.0)
    assert correction["terminal_corrected"] is False
    assert correction["terminal_source"] is None


def test_service_markers_keep_user_positions_after_planning_without_mutating_inputs():
    """规划后起终点标记、面板和动画共享路线首尾点。"""
    window = _window()
    window.state.entry_point = (10.0, 20.0)
    window.state.exit_point = (90.0, 60.0)
    window._pipeline_result = {
        "path": {
            "full_path": [
                [(25.0, 30.0), (75.0, 30.0)],
                [(75.0, 30.0), (75.0, 70.0)],
            ],
        }
    }
    window._service_points_visible = True

    assert window._route_service_anchor("entry") == (25.0, 30.0)
    assert window._route_service_anchor("exit") == (75.0, 70.0)
    # Reference-only service points remain editable user coordinates after a
    # route exists; only the diagnostic helper exposes route anchors.
    assert window._service_point_value(("entry", 0)) == (10.0, 20.0)
    assert window._service_point_value(("exit", 0)) == (90.0, 60.0)
    assert window.state.entry_point == (10.0, 20.0)
    assert window.state.exit_point == (90.0, 60.0)


def test_measurement_tools_replace_previous_result_and_confirm_on_right_click():
    window = _window()

    window._on_measure_tool_changed("point")
    window._handle_measure_click(10, 10, 1)
    first_count = len(window._measure_overlay_items)
    window._handle_measure_click(20, 20, 1)
    assert window._measure_points == [(20, 20)]
    assert window._measure_complete is True
    assert len(window._measure_overlay_items) == first_count

    window._on_measure_tool_changed("line")
    window._handle_measure_click(10, 10, 1)
    window._handle_measure_click(30, 10, 1)
    assert window._measure_points == [(10, 10), (30, 10)]
    assert window._measure_complete is False
    window._handle_measure_click(30, 10, 2)
    assert window._measure_complete is True
    window._handle_measure_click(40, 20, 1)
    assert window._measure_points == [(40, 20)]
    assert window._measure_complete is False

    window._on_measure_tool_changed("area")
    for point in ((10, 10), (30, 10), (30, 30)):
        window._handle_measure_click(*point, 1)
    window._handle_measure_click(30, 30, 2)
    assert window._measure_complete is True
    window._handle_measure_click(50, 50, 1)
    assert window._measure_points == [(50, 50)]
    window._on_measure_tool_changed("")
    assert window._measure_overlay_items == []


def test_point_measurement_preserves_subpixel_precision_and_export_digits(monkeypatch):
    window = _window()
    captured = []
    monkeypatch.setattr(
        window.geo,
        "is_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        window.geo,
        "pixel_to_lonlat",
        lambda x, y: (
            captured.append((x, y)) or 116.451869291234,
            30.601055451234,
        ),
    )
    messages = []
    monkeypatch.setattr(window.log_panel, "info", messages.append)

    window._on_measure_tool_changed("point")
    window._handle_measure_click(10.375, 20.625, 1)

    assert window._measure_points == [(10.375, 20.625)]
    assert captured == [(10.375, 20.625)]
    assert "116.4518692912" in messages[-1]
    assert "30.6010554512" in messages[-1]


def test_route_editor_exposes_add_point_and_toggle_status_buttons():
    window = _window()
    panel = window.task_panel

    assert panel.route_add_point_btn.text() == "新增点"
    assert panel.route_toggle_status_btn.text() == "作业/转弯"
    assert panel.route_add_point_btn.isVisible() is False
    assert panel.route_toggle_status_btn.isVisible() is False

    panel.set_route_editing_active(True)

    # The test window is intentionally not shown.  ``isVisible()`` includes
    # ancestor visibility, while this contract is about the row's own state.
    assert panel.route_edit_action_row.isHidden() is False
    assert panel.route_add_point_btn.isHidden() is False
    assert panel.route_toggle_status_btn.isHidden() is False
    window.close()


def test_viewer_drag_existing_route_node_never_inserts_point():
    """The real press/move/release chain must move one node, not add one."""
    window = _window()
    window._route_edit_active = True
    window._route_edit_mode = "add_point"
    window._sync_drawing_mode()
    window.state.path_points = [(20.0, 20.0), (80.0, 20.0)]
    window.state.path_status = [1]
    window._draw_route_edit_overlay()
    viewer = window.image_view.viewer
    viewer.setTransform(QTransform().scale(4.0, 4.0))

    press_pos = viewer.mapFromScene(QPointF(20.0, 20.0))
    move_pos = viewer.mapFromScene(QPointF(28.25, 31.75))
    before_count = len(window.state.path_points)

    press = _ViewerMouseEvent(
        press_pos, button=Qt.LeftButton, buttons=Qt.LeftButton
    )
    viewer.mousePressEvent(press)
    assert viewer._direct_drag_active is True
    assert len(window.state.path_points) == before_count

    move = _ViewerMouseEvent(move_pos, buttons=Qt.LeftButton)
    viewer.mouseMoveEvent(move)
    release = _ViewerMouseEvent(
        move_pos, button=Qt.LeftButton, buttons=Qt.NoButton
    )
    viewer.mouseReleaseEvent(release)

    assert len(window.state.path_points) == before_count
    assert window.state.path_points[0] == (28.25, 31.75)
    window.close()


def test_viewer_captures_service_point_drag_before_left_button_pan():
    """真实画布事件必须优先拖服务点，不能被左键平移吞掉。"""
    window = _window()
    window.state.entry_point = (10.0, 20.0)
    window.state.exit_point = (90.0, 60.0)
    window._service_points_visible = True
    viewer = window.image_view.viewer
    press_pos = viewer.mapFromScene(QPointF(10.0, 20.0))
    move_pos = viewer.mapFromScene(QPointF(30.0, 35.0))

    press = _ViewerMouseEvent(
        press_pos, button=Qt.LeftButton, buttons=Qt.LeftButton
    )
    viewer.mousePressEvent(press)
    assert press.accepted is True
    assert viewer._direct_drag_active is True
    assert viewer._pan_origin is None

    move = _ViewerMouseEvent(move_pos, buttons=Qt.LeftButton)
    viewer.mouseMoveEvent(move)
    release = _ViewerMouseEvent(
        move_pos, button=Qt.LeftButton, buttons=Qt.NoButton
    )
    viewer.mouseReleaseEvent(release)

    assert move.accepted is True
    assert release.accepted is True
    assert viewer._direct_drag_active is False
    assert window.state._drag_entry_exit is None
    # Hidden/offscreen Qt fixtures can quantize the device-pixel mapping by a
    # fraction of a scene unit.  The interaction contract is that the dragged
    # point remains at the requested scene location within that subpixel error,
    # rather than requiring binary-identical floating-point coordinates.
    assert window.state.entry_point == pytest.approx((30.0, 35.0), abs=0.25)
