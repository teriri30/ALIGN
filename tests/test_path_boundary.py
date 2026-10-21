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


def test_outward_turn_keeps_curve_visible_and_fails_boundary_validation():
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
    assert len(turn) > 2
    assert min(x for x, _ in turn) < 10.0
    assert diagnostics["boundary_constraints"]["field_boundary_present"] is True
    assert diagnostics["boundary_constraints"]["turns_adjusted"] >= 1
    assert diagnostics["boundary_constraints"]["non_work_violations"]
    assert diagnostics["machine_executable"] is False


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
def test_all_turn_strategies_either_stay_inside_or_expose_failed_curve(strategy):
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
    outside_segments = []
    for segment_index, (segment_type, segment) in enumerate(zip(segment_types, full_path)):
        if segment_type == "work":
            continue
        inside = (
            min(x for x, _ in segment) >= 10.0
            and max(x for x, _ in segment) <= 130.0
            and min(y for _, y in segment) >= 10.0
            and max(y for _, y in segment) <= 88.0
        )
        if not inside:
            outside_segments.append(segment_index)
            assert len(segment) > 2
    if outside_segments:
        assert strategy in {"bow", "pear", "semicircle", "auto"}
        reported = {
            item["segment_index"]
            for item in diagnostics["boundary_constraints"]["non_work_violations"]
        }
        assert set(outside_segments) <= reported
        assert diagnostics["machine_executable"] is False


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
        config={
            "turn_outward_clearance_m": 0.1,
            "turn_safety_margin_m": 0.1,
            "service_point_mode": "connect",
        },
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


def test_field_boundary_closing_loop_is_repaired_instead_of_rejecting_everything():
    """A double click on the closing vertex must not empty the whole plan.

    Reproduces the 2026-10-21 project: the operator closed the polygon with two
    extra clicks ~29 px from the first vertex, which made the ring a bow-tie.
    Every shapely gate then answered False and 生成作业线 reported 无有效中心线.
    """
    from row_geometry import field_polygon_local_report
    from path_planner import segment_within_field_boundary

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (100.0, 20.0), (300.0, 60.0), (150.0, 400.0), (10.0, 350.0),
            (98.0, 21.5), (101.5, 19.0),
        ]

    polygon, report = field_polygon_local_report(State(), (420, 320))

    assert polygon is not None
    assert report["repaired"] is True
    assert report["reason"] == "field_boundary_closing_loop_removed"
    assert report["dropped_area_ratio"] < 0.02
    # The repaired ring is the operator's outline: an interior line is covered
    # again instead of being rejected by an invalid geometry.
    assert segment_within_field_boundary([(150.0, 150.0), (150.0, 300.0)], polygon)


def test_genuine_field_boundary_crossing_still_fails_closed():
    """A real crossing drops a real part of the field: never guess it away."""
    from path_planner import segment_within_field_boundary
    from row_geometry import field_polygon_local_report

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        # A drawn "8": the ring crosses itself in the middle of the field.
        field_boundary = [
            (10.0, 10.0), (200.0, 200.0), (200.0, 10.0), (10.0, 200.0),
        ]

    polygon, report = field_polygon_local_report(State(), (260, 260))

    # ``present`` is False: the outline is kept only so every gate fails closed.
    assert report["present"] is False
    assert report["boundary_present"] is True
    assert report["repaired"] is False
    assert report["reason"] == "field_boundary_self_intersection_too_large"
    assert report["dropped_area_ratio"] > 0.02
    assert not segment_within_field_boundary([(50.0, 50.0), (60.0, 60.0)], polygon)


def test_invalid_field_outline_reports_its_own_reason():
    """The gate must name the unusable outline, not claim the lines are outside."""
    from path_planner import segment_field_boundary_status

    bow_tie = np.asarray([
        (0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0),
    ], dtype=np.float64)

    covered, reason = segment_field_boundary_status([(1.0, 1.0), (2.0, 2.0)], bow_tie)

    assert covered is False
    assert reason == "field_boundary_self_intersecting"


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


def test_centerline_simplification_keeps_a_straight_band_to_two_points():
    from path_planner import _smooth_band_centerline

    points = [(float(x), 20.0) for x in range(10, 111, 10)]
    line = _smooth_band_centerline(
        points,
        main_angle=0.0,
        mpp=0.1,
        config={
            "centerline_interval_m": 1.0,
            "centerline_simplify_max_deviation_m": 0.20,
            "centerline_simplify_max_heading_change_deg": 8.0,
            "centerline_simplify_max_points": 8,
        },
    )

    assert line == [(10.0, 20.0), (110.0, 20.0)]


def test_curved_centerline_keeps_extra_points_but_one_checkbox_item():
    from path_planner import _smooth_band_centerline
    from work_band_selection import build_centerline_catalog

    points = [(float(x), 0.05 * float(x) ** 2) for x in range(0, 11)]
    line = _smooth_band_centerline(
        points,
        main_angle=0.0,
        mpp=1.0,
        config={
            "centerline_interval_m": 1.0,
            "centerline_simplify_max_deviation_m": 0.20,
            "centerline_simplify_max_heading_change_deg": 8.0,
            "centerline_simplify_max_points": 8,
        },
    )
    catalog = build_centerline_catalog(
        [{"region_id": "field", "work_lines": [line]}],
        main_angle=0.0,
    )

    assert 3 <= len(line) <= 8
    assert catalog["count"] == 1
    assert catalog["items"][0]["centerline"] == line


def test_centerline_point_budget_never_overrides_deviation_tolerance():
    """DECISION-017: compact output must not erase an observed bend."""
    from path_planner import _simplify_centerline_polyline
    from shapely.geometry import LineString, Point

    points = [(float(x), float(np.sin(x / 3.0))) for x in range(61)]
    result = _simplify_centerline_polyline(
        points,
        mpp=1.0,
        config={
            "centerline_simplify_max_deviation_m": 0.05,
            "centerline_simplify_max_heading_change_deg": 180.0,
            "centerline_simplify_max_points": 2,
        },
    )
    line = LineString(result)
    error = max(Point(point).distance(line) for point in points)

    assert error <= 0.05 + 1e-9
    assert result[0] == points[0]
    assert result[-1] == points[-1]


def test_default_centerline_budget_preserves_repeated_bends_as_one_item():
    """DECISION-017/024: more vertices must not become extra work bands."""
    from path_planner import _simplify_centerline_polyline
    from work_band_selection import build_centerline_catalog
    from shapely.geometry import LineString, Point

    points = [(float(x), float(np.sin(x / 3.0))) for x in range(61)]
    line = _simplify_centerline_polyline(points, 1.0, {})
    error = max(Point(p).distance(LineString(line)) for p in points)
    catalog = build_centerline_catalog(
        [{"region_id": "field", "work_lines": [line]}], main_angle=0.0,
    )

    assert error <= 0.20 + 1e-9
    assert len(line) > 8
    assert catalog["count"] == 1
    assert catalog["items"][0]["centerline"] == line


def test_centerline_budget_preserves_heading_when_distance_is_small():
    """DECISION-017: direction is checked even below the distance limit."""
    from path_planner import _simplify_centerline_polyline

    points = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.0)]
    line = _simplify_centerline_polyline(
        points, 1.0,
        {"centerline_simplify_max_points": 2,
         "centerline_simplify_max_deviation_m": 0.2,
         "centerline_simplify_max_heading_change_deg": 1.0},
    )

    assert line == points


def test_preserving_bends_does_not_move_existing_endpoint_repair_anchor():
    """DECISION-017: output density must not relocate a repair anchor."""
    from types import SimpleNamespace
    from path_planner import prepare_band_centerlines, _smooth_band_centerline

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    curve = [(float(x), 30.0 + float(np.sin(x / 3.0))) for x in range(61)]
    bands = [
        {"id": i, "centerline": [(0.0, y), (100.0, y)],
         "longitudinal_extent_m": 100.0}
        for i, y in enumerate((5.0, 10.0, 15.0), 1)
    ] + [{"id": 4, "centerline": curve, "longitudinal_extent_m": 100.0}]
    state = SimpleNamespace(
        mask_raw=np.zeros((60, 130), dtype=np.uint8),
        field_boundary=[(0, 0), (129, 0), (129, 59), (0, 59)],
        mask_offset_x=0, mask_offset_y=0,
    )
    config = {"centerline_simplify_max_points": 2}
    # The former two-point output is the endpoint-repair reference, while
    # the new detailed output must retain the observed internal bends.
    reference = _smooth_band_centerline(
        curve, 0.0, 1.0,
        {**config, "centerline_simplify_max_deviation_m": 100.0,
         "centerline_simplify_max_heading_change_deg": 180.0},
    )
    expected_y = float(np.median(np.asarray(reference)[:, 1]))
    lines, report = prepare_band_centerlines(
        bands, 0.0, geo=Geo(), state=state, config=config,
    )
    curve_line = lines[report["ordered_band_ids"].index(4)]

    assert len(curve_line) > 8
    assert curve_line[-1] == pytest.approx((100.0, expected_y))


def test_soft_budget_spends_extra_points_before_dense_fallback():
    """DECISION-017: accuracy may exceed the soft budget before going dense."""
    from path_planner import _simplify_centerline_polyline
    from shapely.geometry import LineString, Point

    points = [(float(x), float(np.sin(x / 3.0))) for x in range(61)]
    diagnostics = {}
    result = _simplify_centerline_polyline(
        points,
        mpp=1.0,
        config={
            "centerline_simplify_max_deviation_m": 0.20,
            "centerline_simplify_max_heading_change_deg": 180.0,
            "centerline_simplify_max_points": 2,
            "centerline_simplify_hard_max_points": 64,
        },
        diagnostics=diagnostics,
    )
    error = max(Point(point).distance(LineString(result)) for point in points)

    assert 2 < len(result) < len(points)
    assert error <= 0.20 + 1e-9
    assert diagnostics["soft_point_budget"] == 2
    assert diagnostics["hard_point_budget"] == 64
    assert diagnostics["budget_exceeded"] is True
    assert diagnostics["dense_fallback"] is False
    assert diagnostics["detail_preserved"] is True
    assert len(diagnostics["compact_reference"]) == 2


def test_hard_cap_falls_back_to_smoothed_trace():
    """DECISION-017: dense output is the last resort, not the first remedy."""
    from path_planner import _simplify_centerline_polyline

    points = [(float(x), float(np.sin(x / 3.0))) for x in range(61)]
    diagnostics = {}
    result = _simplify_centerline_polyline(
        points,
        mpp=1.0,
        config={
            "centerline_simplify_max_deviation_m": 0.01,
            "centerline_simplify_max_heading_change_deg": 180.0,
            "centerline_simplify_max_points": 2,
            "centerline_simplify_hard_max_points": 4,
        },
        diagnostics=diagnostics,
    )

    assert result == points
    assert diagnostics["dense_fallback"] is True
    assert diagnostics["hard_cap_reached"] is True


def test_hard_cap_never_sits_below_the_soft_budget():
    from path_planner import _centerline_simplification_settings

    settings = _centerline_simplification_settings({
        "centerline_simplify_max_points": 16,
        "centerline_simplify_hard_max_points": 8,
    })

    assert settings["max_points"] == 16
    assert settings["hard_max_points"] == 16
