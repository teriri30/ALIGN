import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")

import pytest
import numpy as np
from PySide6.QtWidgets import QApplication

from pyside6_app.image_view import ImageViewerGroup
from state import AppState, AutoPathSegment, PathPoint
from workflow import AnimationEngine


class EuclideanGeo:
    @staticmethod
    def pixel_distance_m(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def _engine():
    state = AppState()
    engine = AnimationEngine(state, EuclideanGeo())
    engine.setup([
        AutoPathSegment(
            points=[PathPoint(0, 0), PathPoint(10, 0), PathPoint(20, 0)],
            status=1,
            segment_type="work",
        )
    ])
    return state, engine


def test_animation_seek_uses_path_distance_and_supports_both_ends():
    state, engine = _engine()

    engine.seek_ratio(0.75)
    point, _heading = engine.get_interpolated_position()

    assert state.anim_frame == 1
    assert state.anim_frac == pytest.approx(5.0)
    assert point == pytest.approx((15.0, 0.0))
    assert engine.progress_ratio() == pytest.approx(0.75)

    engine.seek_ratio(0.0)
    assert engine.get_interpolated_position()[0] == pytest.approx((0.0, 0.0))
    engine.seek_ratio(1.0)
    assert engine.get_interpolated_position()[0] == pytest.approx((20.0, 0.0))


def test_animation_controls_emit_seek_pause_and_slow_speed():
    app = QApplication.instance() or QApplication([])
    group = ImageViewerGroup()
    seek_values = []
    pauses = []
    speeds = []
    group.animation_seek_requested.connect(seek_values.append)
    group.animation_pause_requested.connect(lambda: pauses.append(True))
    group.animation_speed_requested.connect(speeds.append)

    group.set_animation_controls_visible(True)
    assert group.anim_play_btn.text() == "暂停"
    group.anim_slider.sliderMoved.emit(725)
    group.anim_play_btn.click()
    group.set_animation_paused(True)
    assert group.anim_play_btn.text() == "继续"
    group.set_animation_paused(False)
    assert group.anim_play_btn.text() == "暂停"
    group.anim_speed_combo.setCurrentIndex(group.anim_speed_combo.findData(0.5))

    assert group.animation_bar.isVisible() is False  # Parent is not shown offscreen.
    assert group.animation_bar.isHidden() is False
    assert seek_values == [pytest.approx(0.725)]
    assert pauses == [True]
    assert speeds[-1] == pytest.approx(0.5)
    group.close()


def test_task_panel_simulate_action_pauses_instead_of_exiting_active_animation():
    from pyside6_app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    class FakeEngine:
        def __init__(self, state):
            self.state = state

        def toggle_pause(self):
            self.state.anim_paused = not self.state.anim_paused

    window._anim_engine = FakeEngine(window.state)
    window.state.anim_active = True
    window.state.anim_paused = False

    window._on_simulate_task_action()
    assert window.state.anim_active is True
    assert window.state.anim_paused is True
    assert window.task_panel._task_widgets["simulate"]._action_btn.text() == "继续"

    window._on_simulate_task_action()
    assert window.state.anim_active is True
    assert window.state.anim_paused is False
    assert window.task_panel._task_widgets["simulate"]._action_btn.text() == "暂停"
    window.close()


def test_simulation_render_failure_does_not_roll_back_or_leave_animation_active():
    from pyside6_app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._pipeline_result = {
        "path": {
            "full_path": [[(0.0, 0.0), (10.0, 0.0)]],
            "segment_types": ["work"],
        }
    }

    def fail_render():
        raise RuntimeError("pose render probe")

    window._render_anim_frame = fail_render
    window._on_toggle_animation()

    assert window.state.anim_active is False
    assert window._anim_timer is None
    assert window._pipeline_result["path"]["full_path"] == [[(0.0, 0.0), (10.0, 0.0)]]
    window.close()


def test_animation_and_footprint_validation_share_corner_pose_heading():
    from footprint_planner import _sample_planar_poses

    state = AppState()
    engine = AnimationEngine(state, EuclideanGeo())
    engine.setup([
        AutoPathSegment(
            points=[PathPoint(0, 0), PathPoint(10, 0), PathPoint(10, 10)],
            status=1,
            segment_type="work",
        )
    ])
    engine.seek_ratio(0.5)
    point, heading = engine.get_interpolated_position()
    poses = _sample_planar_poses(
        np.asarray([(0, 0), (10, 0), (10, 10)], dtype=np.float64),
        10.0,
        heading_window_px=0.30,
    )

    assert point == pytest.approx(poses[1][:2])
    assert heading == pytest.approx(poses[1][2])


def test_animation_uses_configured_machine_speeds_before_playback_multiplier():
    state, engine = _engine()
    state.harvester_params.update({
        "work_speed_mps": 0.9,
        "turn_speed_mps": 0.4,
        "entry_speed_mps": 0.7,
    })

    assert engine._base_speed_mps("work") == pytest.approx(0.9)
    assert engine._base_speed_mps("turn") == pytest.approx(0.4)
    assert engine._base_speed_mps("entry") == pytest.approx(0.7)
