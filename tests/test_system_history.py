import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow


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
    assert window.state.entry_point == (30.0, 35.0)
