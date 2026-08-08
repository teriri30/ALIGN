"""Regression tests for PySide route editing controls."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")

from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow
from state import AutoPathSegment, PathPoint


class DummyGeo:
    def is_ready(self):
        return False

    def pixel_distance_m(self, first, second):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        return (dx * dx + dy * dy) ** 0.5


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
    assert window._build_anim_segments({}, window.state.auto_path_segments) == window.state.auto_path_segments


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
