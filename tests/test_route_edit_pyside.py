"""Regression tests for PySide route editing controls."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QPushButton

from pyside6_app.main_window import MainWindow
from state import AutoPathSegment, PathPoint


class DummyGeo:
    def is_ready(self):
        return False

    def pixel_distance_m(self, first, second):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        return (dx * dx + dy * dy) ** 0.5


class ReadyDummyGeo(DummyGeo):
    def is_ready(self):
        return True

    def pixel_to_lonlat(self, x, y):
        return 116.0 + float(x) * 1e-6, 30.0 - float(y) * 1e-6

    def meters_per_pixel(self, _x, _y):
        return 1.0


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.geo = DummyGeo()
    window._persist_project_state = lambda stage: None
    return window


def _segment(segment_type, status, points):
    segment = AutoPathSegment(segment_type=segment_type, status=status)
    segment.points = [PathPoint(float(x), float(y)) for x, y in points]
    return segment


def test_task_panel_has_route_edit_button_and_signal():
    window = _window()
    called = []
    window.task_panel.route_edit_requested.connect(lambda: called.append(True))

    assert hasattr(window.task_panel, "route_edit_btn")
    window.task_panel.route_edit_btn.click()

    assert called == [True]


def test_prepare_route_editing_compacts_auto_path_into_control_points():
    window = _window()
    dense = [(float(i), 100.0) for i in range(200)]
    window.state.auto_path = [
        _segment("work", 1, dense),
        _segment("turn_reverse", 4, [(199.0, 100.0), (210.0, 120.0), (220.0, 100.0)]),
    ]

    window._prepare_route_editing()

    assert len(window.state.path_points) < len(dense)
    assert window.state.path_points[0] == dense[0]
    assert window.state.path_points[-1] == (220.0, 100.0)
    assert window.state.path_points == [
        (0.0, 100.0),
        (199.0, 100.0),
        (220.0, 100.0),
    ]
    assert window.state.path_status == [1, 0]


def test_store_path_result_and_route_editor_share_the_same_control_polyline():
    window = _window()
    dense_work = [(float(i), 50.0) for i in range(100)]
    path_result = {
        "full_path": [
            dense_work,
            [(99.0, 50.0), (110.0, 65.0), (120.0, 50.0)],
        ],
        "segment_types": ["work", "turn_reverse"],
        "validation": {"valid": True},
        "is_valid": True,
    }

    window._store_path_result(path_result)
    stored_points = list(window.state.path_points)
    stored_status = list(window.state.path_status)

    window._prepare_route_editing()

    assert window.state.path_points == stored_points
    assert window.state.path_status == stored_status
    assert stored_points == [(0.0, 50.0), (99.0, 50.0), (120.0, 50.0)]
    assert stored_status == [1, 0]


def test_route_control_polyline_uses_only_work_line_endpoints():
    window = _window()
    curved_work = [
        (0.0, 0.0),
        (25.0, 3.0),
        (50.0, -2.0),
        (75.0, 4.0),
        (100.0, 0.0),
    ]
    points, statuses = window._build_route_control_polyline([
        _segment("work", 1, curved_work),
        _segment("turn", 0, [(100.0, 0.0), (108.0, 12.0), (116.0, 0.0)]),
        _segment("work", 1, [(116.0, 0.0), (140.0, -3.0), (160.0, 0.0)]),
    ])

    assert points == [
        (0.0, 0.0),
        (100.0, 0.0),
        (116.0, 0.0),
        (160.0, 0.0),
    ]
    assert statuses == [1, 0, 1]


def test_route_control_polyline_preserves_full_harvested_retrace_vertex():
    window = _window()
    points, statuses = window._build_route_control_polyline([
        _segment("work", 1, [(0.0, 0.0), (10.0, 0.0)]),
        _segment("turn_reverse", 4, [(10.0, 0.0), (0.0, 0.0)]),
        _segment("turn_aux", 4, [(0.0, 0.0), (0.0, 10.0)]),
        _segment("work", 1, [(0.0, 10.0), (10.0, 10.0)]),
    ])

    assert points == [
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 0.0),
        (0.0, 10.0),
        (10.0, 10.0),
    ]
    assert statuses == [1, 0, 0, 1]
    window.close()


def test_machine_file_preview_uses_outgoing_status_and_never_joins_volumes():
    window = _window()
    records = [
        {"pixel_x": 0.0, "pixel_y": 0.0, "status": 1},
        {"pixel_x": 5.0, "pixel_y": 0.0, "status": 0},
        {"pixel_x": 5.0, "pixel_y": 5.0, "status": 1},
        {"pixel_x": 20.0, "pixel_y": 20.0, "status": 0},
        {"pixel_x": 25.0, "pixel_y": 20.0, "status": 1},
        {"pixel_x": 25.0, "pixel_y": 25.0, "status": 0},
    ]

    preview = window._build_machine_route_preview(records, {
        "points_per_file": [3, 3],
        "output_files": ["C:/route/004.txt", "C:/route/005.txt"],
    })

    assert preview["point_count"] == 6
    assert preview["file_count"] == 2
    assert preview["path_result"]["segment_types"] == [
        "work", "turn", "turn", "work"
    ]
    assert [
        [(float(x), float(y)) for x, y in edge]
        for edge in preview["path_result"]["full_path"]
    ] == [
        [(0.0, 0.0), (5.0, 0.0)],
        [(5.0, 0.0), (5.0, 5.0)],
        [(20.0, 20.0), (25.0, 20.0)],
        [(25.0, 20.0), (25.0, 25.0)],
    ]
    assert "004.txt…005.txt" in preview["info"]
    window.close()


def test_continuous_display_uses_solid_direct_turn_for_manual_adjustment():
    window = _window()
    result = window._normalise_path_result({
        "full_path": [
            [(0.0, 0.0), (10.0, 0.0)],
            [(10.0, 0.0), (10.0, 8.0)],
            [(10.0, 8.0), (0.0, 8.0)],
        ],
        "segment_types": ["work", "turn_aux", "work"],
    })

    info, degraded_count = window._canonical_route_display_info(result)

    assert degraded_count == 1
    assert "1条人工可调直连" in info
    assert "不可行直连" not in info
    assert result["full_path"][1] == [(10.0, 0.0), (10.0, 8.0)]
    window.close()


def test_continuous_display_draws_direct_turn_as_solid_connection(monkeypatch):
    from PySide6.QtCore import Qt
    import numpy as np

    window = _window()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    drawn = []
    labels = []
    monkeypatch.setattr(
        window.image_view.viewer,
        "draw_polyline",
        lambda points, **kwargs: drawn.append((points, kwargs)),
    )
    monkeypatch.setattr(
        window.image_view.viewer,
        "draw_text",
        lambda *args, **kwargs: labels.append((args, kwargs)),
    )

    window._show_path(window._normalise_path_result({
        "full_path": [
            [(0.0, 0.0), (10.0, 0.0)],
            [(10.0, 0.0), (10.0, 8.0)],
            [(10.0, 8.0), (0.0, 8.0)],
        ],
        "segment_types": ["work", "turn_aux", "work"],
    }))

    assert drawn[-1][1]["color"] == (70, 120, 255)
    assert drawn[-1][1]["style"] == Qt.SolidLine
    assert not any(
        len(args) > 2 and args[2] == "不可行直连"
        for args, _kwargs in labels
    )
    window.close()


def test_continuous_display_highlights_only_outside_turn_parts(monkeypatch):
    import numpy as np

    window = _window()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.state.field_boundary = [
        (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)
    ]
    drawn = []
    monkeypatch.setattr(
        window.image_view.viewer,
        "draw_polyline",
        lambda points, **kwargs: drawn.append((points, kwargs)),
    )

    window._show_path(window._normalise_path_result({
        "full_path": [
            [(2.0, 5.0), (4.0, 5.0)],
            [(4.0, 5.0), (8.0, 5.0), (12.0, 5.0), (14.0, 5.0)],
        ],
        "segment_types": ["work", "turn"],
    }))

    turn_draws = [item for item in drawn if item[1].get("width") == 3]
    assert len(turn_draws) == 2
    assert turn_draws[0][0] == [(4.0, 5.0), (8.0, 5.0), (10.0, 5.0)]
    assert turn_draws[0][1]["color"] == (70, 120, 255)
    assert turn_draws[1][0] == [(10.0, 5.0), (12.0, 5.0), (14.0, 5.0)]
    assert turn_draws[1][1]["color"] == (40, 40, 255)
    window.close()


def test_route_edit_insert_delete_drag_and_sync_auto_path():
    window = _window()
    window.state.path_points = [(0.0, 0.0), (10.0, 0.0)]
    window.state.path_status = [1]

    inserted = window._route_edit_insert_point((5.0, 2.0))
    assert inserted == 1
    assert window.state.path_points == [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    assert window.state.path_status == [1, 1]

    window._route_edit_move_point(1, (5.0, 3.0))
    assert window.state.path_points[1] == (5.0, 3.0)

    window._sync_auto_path_from_editable_route()
    assert window.state.auto_path_planned is True
    assert window.state.auto_path_valid is False
    assert window.state.auto_path[0].segment_type == "work"
    assert len(window.state.auto_path[0].points) == 3

    window._route_edit_delete_point(1)
    assert window.state.path_points == [(0.0, 0.0), (10.0, 0.0)]
    assert window.state.path_status == [1]


def test_stale_plan_worker_result_is_ignored():
    window = _window()
    window.state.project_generation = 5
    window.state.plan_running = True
    window._pipeline_result = {"path": {"sentinel": "current"}}

    window._active_plan_generation = 4
    window._on_plan_worker_done({"path": {"sentinel": "stale"}})

    assert window._pipeline_result == {"path": {"sentinel": "current"}}
    assert window.state.plan_running is True


def test_route_drag_release_commits_geometry_and_invalidates_validation():
    window = _window()
    window._route_edit_active = True
    window._route_drag_idx = 1
    window.state.path_points = [(0.0, 0.0), (5.0, 3.0), (10.0, 0.0)]
    window.state.path_status = [1, 1]
    window.state.auto_path_valid = True
    window._pipeline_result = {
        "path": {
            "full_path": [[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]],
            "validation": {"valid": True},
            "is_valid": True,
        }
    }

    window._on_viewer_mouse_released(5.0, 3.0, 1)

    assert window._route_drag_idx == -1
    assert window.state.auto_path_valid is False
    assert window._pipeline_result["path"]["full_path"][0][1] == (5.0, 3.0)
    assert window._pipeline_result["path"]["validation"]["valid"] is False


def test_route_edit_control_point_drag_is_captured_before_canvas_pan():
    window = _window()
    window._route_edit_active = True
    window.state.path_points = [(0.0, 0.0), (20.0, 0.0)]
    window.state.path_status = [1]

    assert window._service_point_direct_drag_available(0.0, 0.0) is True
    window._on_viewer_mouse_pressed(0.0, 0.0, 1)
    assert window._route_drag_idx == 0
    window._on_viewer_mouse_moved(4.0, 6.0, 1)
    window._on_viewer_mouse_released(4.0, 6.0, 1)

    assert window.state.path_points[0] == (4.0, 6.0)
    assert window._route_drag_idx == -1


def test_route_edit_whole_line_drag_is_captured_before_canvas_pan():
    """DECISION-019: whole-line mode captures the line body, not only its nodes."""
    window = _window()
    window._route_edit_active = True
    window._route_edit_mode = "move_line"
    window.state.path_points = [(0.0, 0.0), (100.0, 0.0)]
    window.state.path_status = [1]

    assert window._service_point_direct_drag_available(50.0, 0.0) is True
    window._on_viewer_mouse_pressed(50.0, 0.0, 1)
    assert window._route_drag_work_index == 0
    window._on_viewer_mouse_moved(55.0, 8.0, 1)
    window._on_viewer_mouse_released(55.0, 8.0, 1)

    assert window.state.path_points == [(5.0, 8.0), (105.0, 8.0)]
    assert window._route_drag_work_index == -1
    window.close()


def test_edit_route_keeps_protocol_chord_but_simulation_turn_is_curved():
    window = _window()
    window.state.path_points = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 30.0), (0.0, 30.0)
    ]
    window.state.path_status = [1, 0, 1]

    window._sync_auto_path_from_editable_route()

    # The editable/export projection remains one straight 0-status chord.
    assert window.state.path_points[1:3] == [(100.0, 0.0), (100.0, 30.0)]
    turn = [segment for segment in window.state.auto_path if segment.segment_type == "turn"]
    assert len(turn) == 1
    assert len(turn[0].points) > 2
    displayed = window._pipeline_result["path"]
    displayed_turn = displayed["full_path"][1]
    assert len(displayed_turn) > 2
    assert displayed["control_full_path"][1] == [
        (100.0, 0.0), (100.0, 30.0)
    ]
    assert window._build_anim_segments({}, window.state.auto_path_segments) == window.state.auto_path_segments


def test_dragged_missing_endpoint_reconnects_work_and_turn_without_a_gap():
    """DECISION-016 v13: one shared node owns both the work end and turn start."""
    window = _window()
    window.state.path_points = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 30.0),
        (0.0, 30.0),
    ]
    window.state.path_status = [1, 0, 1]

    moved = (118.0, -12.0)
    assert window._route_edit_move_point(1, moved) is True
    window._sync_auto_path_from_editable_route()

    work_before, turn, work_after = window.state.auto_path
    assert work_before.segment_type == "work"
    assert turn.segment_type == "turn"
    assert work_after.segment_type == "work"
    assert len(turn.points) > 2
    assert (work_before.points[-1].pixel_x, work_before.points[-1].pixel_y) == moved
    assert (turn.points[0].pixel_x, turn.points[0].pixel_y) == moved
    assert (
        turn.points[-1].pixel_x,
        turn.points[-1].pixel_y,
    ) == (
        work_after.points[0].pixel_x,
        work_after.points[0].pixel_y,
    )
    displayed = window._pipeline_result["path"]
    assert displayed["full_path"][0][-1] == moved
    assert displayed["full_path"][1][0] == moved
    assert len(displayed["full_path"][1]) > 2
    assert displayed["control_full_path"][1] == [moved, (100.0, 30.0)]
    window.close()


def test_route_editor_can_create_a_complete_route_from_work_lines():
    """DECISION-019: manual editing can create work lines, not just drag points."""
    window = _window()

    assert window._route_edit_add_work_line((0.0, 0.0), (100.0, 0.0)) == 0
    assert window._route_edit_add_work_line((100.0, 30.0), (0.0, 30.0)) == 1
    assert window._route_edit_add_work_line((0.0, 60.0), (100.0, 60.0)) == 2

    assert window.state.path_points == [
        (0.0, 0.0), (100.0, 0.0),
        (100.0, 30.0), (0.0, 30.0),
        (0.0, 60.0), (100.0, 60.0),
    ]
    assert window.state.path_status == [1, 0, 1, 0, 1]
    assert [
        item["points"] for item in window._route_edit_work_records()
    ] == [
        [(0.0, 0.0), (100.0, 0.0)],
        [(100.0, 30.0), (0.0, 30.0)],
        [(0.0, 60.0), (100.0, 60.0)],
    ]
    window.close()


def test_route_editor_moves_deletes_reorders_and_reverses_whole_work_lines():
    """DECISION-019: structural edits reconnect adjacent turns atomically."""
    window = _window()
    for start, end in (
        ((0.0, 0.0), (100.0, 0.0)),
        ((100.0, 30.0), (0.0, 30.0)),
        ((0.0, 60.0), (100.0, 60.0)),
    ):
        window._route_edit_add_work_line(start, end)

    assert window._route_edit_translate_work_line(1, (8.0, -4.0)) is True
    assert window._route_edit_work_records()[1]["points"] == [
        (108.0, 26.0), (8.0, 26.0)
    ]
    assert window._route_edit_reverse_work_line(1) is True
    assert window._route_edit_work_records()[1]["points"] == [
        (8.0, 26.0), (108.0, 26.0)
    ]
    assert window._route_edit_reorder_work_line(1, -1) == 0
    assert window._route_edit_work_records()[0]["points"] == [
        (8.0, 26.0), (108.0, 26.0)
    ]
    assert window._route_edit_delete_work_line(1) is True
    records = window._route_edit_work_records()
    assert len(records) == 2
    assert window.state.path_status == [1, 0, 1]
    window.close()


def test_manual_turn_waypoints_are_preserved_in_canonical_preview():
    """DECISION-019: explicit manual turn geometry is not replaced by an auto bow."""
    window = _window()
    manual_turn = [(100.0, 0.0), (118.0, 12.0), (100.0, 30.0)]
    window.state.path_points = [
        (0.0, 0.0),
        *manual_turn,
        (0.0, 30.0),
    ]
    window.state.path_status = [1, 0, 0, 1]

    window._sync_auto_path_from_editable_route()

    turn = next(
        item for item in window.state.auto_path if item.segment_type == "turn"
    )
    assert [
        (point.pixel_x, point.pixel_y) for point in turn.points
    ] == manual_turn
    assert window._pipeline_result["path"]["full_path"][1] == manual_turn
    window.close()


def test_task_panel_uses_canvas_and_route_results_without_duplicate_tool_grid():
    """DECISION-019: the planning card must not duplicate route-result actions."""
    window = _window()
    panel = window.task_panel

    assert not hasattr(panel, "route_edit_tools")
    panel.set_route_editing_active(True)
    assert panel.route_edit_btn.text() == "完成编辑"
    panel.set_route_editing_active(False)
    assert panel.route_edit_btn.text() == "编辑路线"
    window.close()


def test_route_edit_overlay_preserves_processed_mask_layers(monkeypatch):
    """DECISION-019: route editing is additive and keeps prior mask evidence."""
    window = _window()
    window._tif_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    body = np.zeros((80, 120), dtype=np.uint8)
    body[10:70, 20:100] = 255
    headland = np.zeros_like(body)
    headland[10:20, 20:100] = 255
    uncertain = np.zeros_like(body)
    uncertain[60:70, 20:100] = 255
    window.state.mask_result = {
        "processed_mask": body,
        "headland_mask": headland,
        "uncertain_residual_mask": uncertain,
    }
    window.state.path_points = [(20.0, 20.0), (100.0, 20.0)]
    window.state.path_status = [1]
    drawn_masks = []
    monkeypatch.setattr(
        window.image_view.viewer,
        "draw_mask_overlay",
        lambda mask, x=0, y=0, color=(120, 255, 140), **_kwargs:
        drawn_masks.append((mask, x, y, color)),
    )

    window._draw_route_edit_overlay()

    assert [item[0] for item in drawn_masks] == [headland, uncertain, body]
    assert drawn_masks[-1][3] == (120, 255, 140)
    window.close()


def test_route_editor_two_click_creation_and_direct_drag_move_whole_line():
    """DECISION-019: canvas creation and ordinary work-line drag stay direct."""
    window = _window()
    window._route_edit_active = True
    window._route_edit_mode = "add_work"

    assert window._handle_route_edit_press(10.0, 20.0, 1) is True
    assert window._route_new_line_start == (10.0, 20.0)
    assert window._handle_route_edit_press(110.0, 20.0, 1) is True
    assert window._route_edit_work_records()[0]["points"] == [
        (10.0, 20.0), (110.0, 20.0)
    ]

    window._route_edit_mode = "select"
    assert window._handle_route_edit_press(60.0, 20.0, 1) is True
    window._on_viewer_mouse_moved(66.0, 29.0, 1)
    window._on_viewer_mouse_released(66.0, 29.0, 1)
    assert window._route_edit_work_records()[0]["points"] == [
        (16.0, 29.0), (116.0, 29.0)
    ]
    window.close()


def test_route_edit_hover_cursor_exposes_nodes_edges_and_empty_canvas():
    """DECISION-019: hover feedback shows which canvas geometry is editable."""
    window = _window()
    window._route_edit_active = True
    window._route_edit_mode = "select"
    window.state.path_points = [(10.0, 20.0), (110.0, 20.0)]
    window.state.path_status = [1]
    window._sync_drawing_mode()

    window._on_viewer_mouse_moved(10.0, 20.0, 0)
    assert window.image_view.viewer.cursor().shape() == Qt.SizeAllCursor
    window._on_viewer_mouse_moved(60.0, 20.0, 0)
    assert window.image_view.viewer.cursor().shape() == Qt.PointingHandCursor
    window._on_viewer_mouse_moved(60.0, 80.0, 0)
    assert window.image_view.viewer.cursor().shape() == Qt.ArrowCursor
    window.close()


def test_route_drag_uses_dynamic_overlay_without_rebuilding_mask_evidence(monkeypatch):
    """DECISION-019: a drag frame must not rebuild the full mask overlay."""
    window = _window()
    window._route_edit_active = True
    window.state.path_points = [(10.0, 20.0), (110.0, 20.0)]
    window.state.path_status = [1]
    window._route_drag_idx = 0
    dynamic_calls = []
    monkeypatch.setattr(
        window,
        "_draw_route_edit_overlay",
        lambda: pytest.fail("drag frame rebuilt the full evidence overlay"),
    )
    monkeypatch.setattr(
        window,
        "_draw_route_edit_dynamic_overlay",
        lambda **kwargs: dynamic_calls.append(kwargs),
    )

    window._on_viewer_mouse_moved(20.0, 30.0, 1)

    assert window.state.path_points[0] == (20.0, 30.0)
    assert dynamic_calls == [{"lightweight": True}]
    window.close()


def test_empty_route_enters_two_click_work_line_creation_mode():
    """DECISION-019: an empty route remains constructible without an upper tool grid."""
    window = _window()

    window._on_route_edit_requested()

    assert window._route_edit_active is True
    assert window._route_edit_mode == "add_work"
    assert window.image_view.viewer.cursor().shape() == Qt.CrossCursor
    window.close()


def test_manual_turn_point_insertion_keeps_exact_clicked_geometry():
    """DECISION-019: turn control points are free geometry, not line projection."""
    window = _window()
    window.state.path_points = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 30.0), (0.0, 30.0)
    ]
    window.state.path_status = [1, 0, 1]

    inserted = window._route_edit_insert_point(
        (118.0, 12.0), preserve_position=True, edge_index=1
    )
    window._sync_auto_path_from_editable_route()

    assert inserted == 2
    assert window.state.path_points[2] == (118.0, 12.0)
    turn = next(item for item in window.state.auto_path if item.segment_type == "turn")
    assert [(point.pixel_x, point.pixel_y) for point in turn.points] == [
        (100.0, 0.0), (118.0, 12.0), (100.0, 30.0)
    ]
    window.close()


def test_visible_add_point_tool_keeps_exact_geometry_on_any_selected_edge():
    """DECISION-019: the visible point tool bends work or turn edges freely."""
    window = _window()
    window._route_edit_active = True
    window._route_edit_mode = "add_point"
    window.state.path_points = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 30.0), (0.0, 30.0)
    ]
    window.state.path_status = [1, 0, 1]
    window.state.selected_type = "edge"
    window.state.selected_idx = 0

    assert window._handle_route_edit_press(50.0, 18.0, 1) is True

    assert window.state.path_points[1] == (50.0, 18.0)
    assert window.state.path_status == [1, 1, 0, 1]
    work = next(item for item in window.state.auto_path if item.segment_type == "work")
    assert [(point.pixel_x, point.pixel_y) for point in work.points] == [
        (0.0, 0.0), (50.0, 18.0), (100.0, 0.0)
    ]
    window.close()


def test_structural_route_edit_shortcuts_reverse_undo_and_redo():
    """DECISION-019: low-frequency structural actions use documented shortcuts."""
    window = _window()
    window._route_edit_add_work_line((0.0, 0.0), (100.0, 0.0))
    window._route_edit_add_work_line((100.0, 30.0), (0.0, 30.0))
    original = list(window.state.path_points)
    second = window._route_edit_work_records()[1]
    window.state.selected_type = "edge"
    window.state.selected_idx = second["edge_start"]

    window._route_edit_active = True
    window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_R, Qt.NoModifier))
    reversed_points = list(window.state.path_points)
    assert reversed_points != original

    window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier))
    assert window.state.path_points == original
    window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Y, Qt.ControlModifier))
    assert window.state.path_points == reversed_points
    window.close()


def test_visible_toggle_switches_the_selected_logical_segment():
    window = _window()
    window.state.path_points = [
        (0.0, 0.0), (50.0, 10.0), (100.0, 0.0), (100.0, 30.0)
    ]
    window.state.path_status = [1, 1, 0]
    window.state.selected_type = "edge"
    window.state.selected_idx = 0

    window._on_route_edit_tool_requested("toggle_status")
    assert window.state.path_status == [0, 0, 0]
    window._on_route_edit_tool_requested("toggle_status")
    assert window.state.path_status == [1, 1, 1]
    window.close()


def test_documented_shortcuts_reorder_set_status_and_delete_work_line():
    window = _window()
    for y in (0.0, 30.0, 60.0):
        window._route_edit_add_work_line((0.0, y), (100.0, y))
    window._route_edit_active = True
    middle = window._route_edit_work_records()[1]
    middle_points = list(middle["points"])
    window.state.selected_type = "edge"
    window.state.selected_idx = middle["edge_start"]

    window.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.ControlModifier)
    )
    assert window._route_edit_work_records()[0]["points"] == middle_points

    selected_edge = window._route_edit_work_records()[0]["edge_start"]
    window.state.selected_type = "edge"
    window.state.selected_idx = selected_edge
    window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_0, Qt.NoModifier))
    assert window.state.path_status[selected_edge] == 0
    window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_1, Qt.NoModifier))
    assert window.state.path_status[selected_edge] == 1

    window.state.selected_type = "edge"
    window.state.selected_idx = selected_edge
    window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
    assert len(window._route_edit_work_records()) == 2
    window.close()


def test_route_result_edit_cards_expose_linked_actions_and_work_drag_order():
    """DECISION-019: result cards are editable views of the shared route."""
    from pyside6_app.panels import RouteInfoPanel

    app = QApplication.instance() or QApplication([])
    panel = RouteInfoPanel()
    selected = []
    actions = []
    reordered = []
    panel.edit_segment_selected.connect(lambda first, last: selected.append((first, last)))
    panel.edit_action_requested.connect(
        lambda action, first, last: actions.append((action, first, last))
    )
    panel.work_reorder_requested.connect(
        lambda source, target: reordered.append((source, target))
    )
    panel.update_editable_route([
        {
            "segment_index": 0, "type": "work", "status": 1,
            "edge_start": 0, "edge_end": 0, "node_start": 0, "node_end": 1,
            "work_index": 0, "length_m": 100.0, "point_count": 2,
        },
        {
            "segment_index": 1, "type": "turn", "status": 0,
            "edge_start": 1, "edge_end": 2, "node_start": 1, "node_end": 3,
            "work_index": -1, "length_m": 20.0, "point_count": 3,
        },
        {
            "segment_index": 2, "type": "work", "status": 1,
            "edge_start": 3, "edge_end": 3, "node_start": 3, "node_end": 4,
            "work_index": 1, "length_m": 100.0, "point_count": 2,
        },
    ])
    panel.resize(330, 260)
    panel.show()
    app.processEvents()

    assert not panel._edit_hint.isHidden()
    assert len(panel._segments) == 3
    assert [row._label.text() for row in panel._segments] == [
        "作业线 01", "正式调头 01", "作业线 02"
    ]
    assert panel._segments[0].acceptDrops() is True
    assert panel._segments[1].acceptDrops() is False
    assert "add_work_after" in panel._segments[0].available_edit_actions()
    assert "reverse_work" in panel._segments[0].available_edit_actions()
    assert "delete_work" in panel._segments[0].available_edit_actions()
    assert "validate_route" in panel._segments[0].available_edit_actions()
    assert "toggle_segment" in panel._segments[1].available_edit_actions()
    assert "delete_waypoint" in panel._segments[1].available_edit_actions()
    for index, row in enumerate(panel._segments[1:], start=1):
        previous = panel._segments[index - 1]
        assert row.y() >= previous.y() + previous.height()

    panel._segments[1].edit_clicked.emit(1, 2)
    panel._segments[1].trigger_edit_action("toggle_segment")
    panel._segments[1].trigger_edit_action("delete_waypoint", 2)
    panel._segments[0].reorder_requested.emit(0, 1)
    panel.set_edit_selection("edge", 1)

    assert selected == [(1, 2)]
    assert actions == [
        ("toggle_segment", 1, 2),
        ("delete_waypoint", 2, 2),
    ]
    assert reordered == [(0, 1)]
    assert panel._segments[1]._edit_selected is True
    panel.close()


def test_route_result_card_operations_immediately_update_shared_route_and_canvas(monkeypatch):
    """DECISION-019: card mutations update state, list, and canvas atomically."""
    window = _window()
    for y in (0.0, 30.0, 60.0):
        window._route_edit_add_work_line((0.0, y), (100.0, y))
    window._route_edit_active = True
    draws = []
    monkeypatch.setattr(window, "_draw_route_edit_overlay", lambda: draws.append(True))
    window._refresh_route_edit_info_panel()

    window.route_info.edit_segment_selected.emit(0, 0)
    assert (window.state.selected_type, window.state.selected_idx) == ("edge", 0)

    window.route_info.edit_action_requested.emit("add_waypoint", 0, 0)
    assert window.state.path_points[1] == (50.0, 0.0)
    assert window.state.selected_type == "node"
    inserted_node = window.state.selected_idx

    window.route_info.edit_action_requested.emit(
        "delete_waypoint", inserted_node, inserted_node
    )
    assert window.state.path_points[:2] == [(0.0, 0.0), (100.0, 0.0)]

    window.route_info.work_reorder_requested.emit(2, 0)
    assert window._route_edit_work_records()[0]["points"] == [
        (0.0, 60.0), (100.0, 60.0)
    ]

    first = window._route_edit_work_records()[0]
    first_points = list(first["points"])
    window.route_info.edit_action_requested.emit(
        "reverse_work", first["edge_start"], first["edge_end"]
    )
    assert window._route_edit_work_records()[0]["points"] == list(
        reversed(first_points)
    )

    first = window._route_edit_work_records()[0]
    window.route_info.edit_action_requested.emit(
        "delete_work", first["edge_start"], first["edge_end"]
    )
    assert len(window._route_edit_work_records()) == 2

    first = window._route_edit_work_records()[0]
    window.route_info.edit_action_requested.emit(
        "add_work_after", first["edge_start"], first["edge_end"]
    )
    assert window._route_edit_mode == "add_work"
    window._handle_route_edit_press(0.0, 90.0, 1)
    window._handle_route_edit_press(100.0, 90.0, 1)
    assert window._route_edit_work_records()[1]["points"] == [
        (0.0, 90.0), (100.0, 90.0)
    ]
    monkeypatch.setattr(window, "_validate_manual_route", lambda: {"valid": True})
    window.route_info.edit_action_requested.emit("validate_route", 0, 0)
    assert window.lbl_status.text() == "人工路线几何验证通过"
    assert len(draws) >= 7
    window.close()


def test_route_result_card_right_click_switches_the_whole_segment_semantics():
    window = _window()
    window.state.path_points = [
        (0.0, 0.0), (100.0, 0.0), (110.0, 10.0),
        (100.0, 30.0), (0.0, 30.0),
    ]
    window.state.path_status = [1, 0, 0, 1]
    window._route_edit_active = True

    window.route_info.edit_action_requested.emit("toggle_segment", 1, 2)

    assert window.state.path_status == [1, 1, 1, 1]
    assert window.state.selected_type == "edge"
    assert window.state.selected_idx == 1
    window.close()


def test_manual_route_validation_runs_the_real_shared_validator():
    """DECISION-019: the real footprint validator accepts current manual data."""
    window = _window()
    window.geo = ReadyDummyGeo()
    mask = np.ones((160, 160), dtype=np.uint8) * 255
    window.state.path_points = [
        (10.0, 20.0), (140.0, 20.0),
        (150.0, 35.0), (140.0, 50.0), (10.0, 50.0),
    ]
    window.state.path_status = [1, 0, 0, 1]
    window.state.mask_result = {
        "processed_mask": mask,
        "main_angle": 0.0,
        "wide_bands": [],
    }
    window.state.inference_original_mask = mask
    window.state.forbidden_regions_confirmed = True

    validation = window._validate_manual_route()

    assert validation["route_geometry_source"] == "manual_control_edit_validated"
    assert validation["current_validation_complete"] is True
    assert "total_length_m" in validation
    assert "rolling_canopy_pct" in validation
    assert validation["automatic_planning_provenance_valid"] is False
    window.close()


def test_manual_route_validation_reuses_planner_validator_and_records_source(monkeypatch):
    """DECISION-019: validation certifies manual geometry without auto provenance."""
    window = _window()
    window.geo = ReadyDummyGeo()
    window.state.path_points = [
        (0.0, 0.0), (100.0, 0.0),
        (118.0, 12.0), (100.0, 30.0), (0.0, 30.0),
    ]
    window.state.path_status = [1, 0, 0, 1]
    window.state.mask_result = {
        "processed_mask": np.ones((160, 160), dtype=np.uint8) * 255,
        "main_angle": 0.37,
        "wide_bands": [],
    }
    captured = {}

    def fake_validate_path(full_path, mask, main_angle, **kwargs):
        captured["full_path"] = full_path
        captured["mask"] = mask
        captured["main_angle"] = main_angle
        captured["segment_types"] = kwargs["segment_types"]
        return {
            "valid": True,
            "issues": [],
            "total_length_m": 230.0,
            "work_length_m": 200.0,
            "turn_length_m": 30.0,
            "entry_exit_length_m": 0.0,
        }

    monkeypatch.setattr("path_planner.validate_path", fake_validate_path)
    monkeypatch.setattr(
        window, "_manual_route_minimum_turn_radius_m", lambda *_args: 1.2
    )

    validation = window._validate_manual_route()

    assert captured["main_angle"] == 0.37
    assert captured["segment_types"] == ["work", "turn", "work"]
    assert captured["full_path"][1] == [
        (100.0, 0.0), (118.0, 12.0), (100.0, 30.0)
    ]
    assert validation["valid"] is True
    assert validation["route_geometry_source"] == "manual_control_edit_validated"
    assert validation["automatic_planning_provenance_valid"] is False
    assert window.state.path_provenance == {}
    assert window.state.path_artifact["manual_route"]["schema"] == "reui.manual-route.v1"
    restored = window._path_result_from_state()
    assert restored["manual_route_edited"] is True
    assert restored["manual_route_validated"] is True
    window.close()


def test_harvester_dialog_hides_internal_wheelbase_and_source_selector():
    from PySide6.QtWidgets import QLabel
    from pyside6_app.dialogs import HarvesterParamsDialog

    dialog = HarvesterParamsDialog({
        "cutter_width_m": 2.0,
        "cutter_forward_offset_m": 1.35,
        "track_width_m": 0.35,
        "track_gauge_m": 1.30,
        "wheelbase_m": 1.80,
        "track_length_m": 1.70,
        "turn_radius_m": 0.90,
    }, source="user_measured")

    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert "轴距:" not in labels
    assert "参数来源:" not in labels
    assert not hasattr(dialog, "spin_wheelbase")
    assert not hasattr(dialog, "combo_source")
    assert "wheelbase_m" not in dialog.get_params()
