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
    assert 1 in window.state.path_status
    assert 4 in window.state.path_status


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
