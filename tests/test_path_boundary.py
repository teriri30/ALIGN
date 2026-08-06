import numpy as np
import pytest


def test_field_support_uses_field_boundary_not_convex_hull():
    from footprint_planner import _field_support
    from state import AppState

    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[10:70, 10:110] = 255
    state = AppState()
    state.field_boundary = [(30, 10), (110, 10), (110, 70), (30, 70)]

    support = _field_support(mask, state)

    assert support[40, 20] == 0
    assert support[40, 40] == 255


def test_connect_work_lines_turn_points_stay_inside_field_boundary_when_outward_arc_would_leave():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.ones((120, 120), dtype=np.uint8) * 255
    state.field_boundary = [(10, 10), (110, 10), (110, 110), (10, 110)]

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(80.0, 12.0), (12.0, 12.0)], [(12.0, 24.0), (80.0, 24.0)]],
        state=state,
        geo=Geo(),
        min_turn_radius_m=1.0,
        turn_strategy="semicircle",
        config={"turn_outward_clearance_m": 0.6, "turn_safety_margin_m": 0.1},
        return_diagnostics=True,
    )

    turn = full_path[segment_types.index("turn")]
    assert min(x for x, _ in turn) >= 10.0
    assert max(y for _, y in turn) <= 110.0
    assert diagnostics["boundary_constraints"]["field_boundary_present"] is True
    assert diagnostics["boundary_constraints"]["turns_adjusted"] >= 1


def test_reverse_turns_do_not_bypass_field_boundary_constraint():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.ones((140, 140), dtype=np.uint8) * 255
    state.field_boundary = [(10, 10), (130, 10), (130, 80), (10, 80)]

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(120.0, 12.0), (12.0, 12.0)], [(12.0, 22.0), (120.0, 22.0)]],
        state=state,
        geo=Geo(),
        min_turn_radius_m=1.0,
        turn_strategy="fishtail",
        config={"turn_outward_clearance_m": 0.6, "turn_safety_margin_m": 0.1},
        return_diagnostics=True,
    )

    for segment_type, segment in zip(segment_types, full_path):
        if segment_type.startswith("turn"):
            assert min(x for x, _ in segment) >= 10.0
            assert max(x for x, _ in segment) <= 130.0
            assert min(y for _, y in segment) >= 10.0
            assert max(y for _, y in segment) <= 80.0
    assert "turn_reverse" not in segment_types
    assert diagnostics["boundary_constraints"]["turns_adjusted"] >= 1


@pytest.mark.parametrize("strategy", ["bow", "pear", "semicircle", "alpha", "fishtail", "contour", "auto"])
def test_all_turn_strategies_keep_non_work_segments_inside_field_boundary(strategy):
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.ones((140, 140), dtype=np.uint8) * 255
    state.field_boundary = [(10, 10), (130, 10), (130, 88), (10, 88)]

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(120.0, 14.0), (14.0, 14.0)], [(14.0, 26.0), (120.0, 26.0)]],
        state=state,
        geo=Geo(),
        min_turn_radius_m=1.0,
        turn_strategy=strategy,
        config={"turn_outward_clearance_m": 0.8, "turn_safety_margin_m": 0.1},
        return_diagnostics=True,
    )

    assert diagnostics["boundary_constraints"]["field_boundary_present"] is True
    for segment_type, segment in zip(segment_types, full_path):
        if segment_type == "work":
            continue
        assert min(x for x, _ in segment) >= 10.0
        assert max(x for x, _ in segment) <= 130.0
        assert min(y for _, y in segment) >= 10.0
        assert max(y for _, y in segment) <= 88.0


def test_entry_exit_and_unload_segments_report_field_boundary_violations():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.ones((120, 120), dtype=np.uint8) * 255
    state.field_boundary = [(10, 10), (110, 10), (110, 90), (10, 90)]
    state.entry_point = (-5.0, 20.0)
    state.unload_points = [(125.0, 55.0)]
    state.exit_point = (125.0, 70.0)

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    _path, _types, diagnostics = _connect_work_lines(
        [[(20.0, 20.0), (95.0, 20.0)], [(95.0, 45.0), (20.0, 45.0)]],
        state=state,
        geo=Geo(),
        min_turn_radius_m=1.0,
        turn_strategy="semicircle",
        config={"turn_outward_clearance_m": 0.1, "turn_safety_margin_m": 0.1},
        return_diagnostics=True,
    )

    violations = diagnostics["boundary_constraints"]["non_work_violations"]
    assert {item["segment_type"] for item in violations} >= {"entry", "unload", "exit"}
    hard_reasons = " ".join(diagnostics.get("hard_reasons") or [])
    assert "entry segment leaves field boundary" in hard_reasons
    assert diagnostics["feasible"] is False


def test_segment_boundary_check_rejects_line_crossing_concave_notch():
    from path_planner import _segment_inside_field_boundary

    field_polygon = np.asarray([
        (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (6.0, 10.0),
        (6.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0),
    ], dtype=np.float32)

    assert not _segment_inside_field_boundary([(2.0, 8.0), (8.0, 8.0)], field_polygon)


def test_segment_boundary_check_accepts_line_fully_covered_by_concave_field():
    from path_planner import _segment_inside_field_boundary

    field_polygon = np.asarray([
        (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (6.0, 10.0),
        (6.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0),
    ], dtype=np.float32)

    assert _segment_inside_field_boundary([(1.0, 2.0), (9.0, 2.0)], field_polygon)


@pytest.mark.parametrize("segment_type", [
    "entry", "exit", "unload", "turn", "turn_reverse", "turn_aux", "turn_approach",
])
def test_all_non_work_segment_types_reject_concave_boundary_crossing(segment_type):
    from path_planner import _non_work_boundary_violations

    field_polygon = np.asarray([
        (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (6.0, 10.0),
        (6.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0),
    ], dtype=np.float32)
    violations = _non_work_boundary_violations(
        [[(2.0, 8.0), (8.0, 8.0)]], [segment_type], field_polygon
    )

    assert violations == [{"segment_index": 0, "segment_type": segment_type}]


def test_prepare_band_centerlines_does_not_trim_harvestable_row_ends_by_default():
    from path_planner import _smooth_band_centerline

    points = [(float(x), 20.0) for x in range(10, 111, 10)]
    line = _smooth_band_centerline(
        points,
        main_angle=0.0,
        mpp=0.1,
        config={"centerline_interval_m": 1.0},
    )

    assert line[0][0] == 10.0
    assert line[-1][0] == 110.0
