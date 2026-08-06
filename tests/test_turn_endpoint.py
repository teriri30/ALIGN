import numpy as np
import pytest


class MetricGeo:
    def is_ready(self):
        return True

    def meters_per_pixel(self, _x, _y):
        return 0.1


class FieldState:
    mask_offset_x = 0
    mask_offset_y = 0
    entry_point = None
    exit_point = None
    unload_points = []
    unload_point = None
    harvester_params = {
        "cutter_width_m": 2.0,
        "track_width_m": 0.4,
        "track_gauge_m": 1.7,
        "wheelbase_m": 1.9,
        "track_length_m": 1.8,
        "turn_radius_m": 2.0,
    }

    def __init__(self):
        self.mask_raw = np.zeros((120, 140), dtype=np.uint8)
        self.field_boundary = [(2, 2), (125, 2), (125, 110), (2, 110)]


def test_single_short_line_uses_separate_headland_approach_before_turn():
    """DECISION-010: one short crop line must not move its turn anchor inward."""
    from path_planner import _connect_work_lines

    lines = [
        [(10.0, 20.0), (100.0, 20.0)],
        [(10.0, 40.0), (70.0, 40.0)],
        [(10.0, 60.0), (100.0, 60.0)],
    ]

    path, segment_types, diagnostics = _connect_work_lines(
        lines,
        geo=MetricGeo(),
        state=FieldState(),
        min_turn_radius_m=2.0,
        turn_strategy="pear",
        config={
            "endpoint_short_line_ratio": 0.85,
            "endpoint_shortfall_min_m": 0.5,
            "endpoint_extension_max_m": 4.0,
        },
        return_diagnostics=True,
    )

    first_turn = path[segment_types.index("turn")]
    assert first_turn[0][0] == pytest.approx(100.0)
    assert first_turn[-1][0] == pytest.approx(100.0)
    # DECISION-010 v2: 锚点延长沿主方向合并进作业线本身（不再产生独立
    # turn_approach 段）——短线直接延伸到田头锚点，转弯保持短小。
    assert "turn_approach" not in segment_types
    work_indices = [i for i, t in enumerate(segment_types) if t == "work"]
    short_line = path[work_indices[1]]
    assert short_line[0] == pytest.approx((100.0, 40.0))
    assert short_line[-1] == pytest.approx((10.0, 40.0))
    assert diagnostics["endpoint_alignment"]["corrected_endpoint_count"] == 1
    assert diagnostics["endpoint_alignment"].get(
        "long_anchor_extensions_merged_into_work_lines"
    )
    decision = diagnostics["turn_decisions"][0]
    assert decision["requested_strategy"] == "pear"
    assert decision["candidate_strategy"] == "pear"
    assert decision["effective_strategy"] == "pear"
    assert len(decision["geometry_signature"]) == 64
    assert decision["pass_spacing_m"] == pytest.approx(2.0)
    assert decision["turn_radius_m"] == pytest.approx(2.0)
    assert decision["turn_start"] == pytest.approx([100.0, 20.0])
    assert decision["turn_end"] == pytest.approx([100.0, 40.0])


def test_endpoint_alignment_refuses_target_outside_field_boundary():
    """DECISION-010: endpoint correction remains fail-closed at field boundaries."""
    from path_planner import _connect_work_lines

    state = FieldState()
    state.field_boundary = [(2, 2), (80, 2), (80, 110), (2, 110)]
    lines = [
        [(10.0, 20.0), (100.0, 20.0)],
        [(10.0, 40.0), (70.0, 40.0)],
        [(10.0, 60.0), (100.0, 60.0)],
    ]

    _path, _types, diagnostics = _connect_work_lines(
        lines,
        geo=MetricGeo(),
        state=state,
        min_turn_radius_m=2.0,
        turn_strategy="pear",
        config={
            "endpoint_short_line_ratio": 0.85,
            "endpoint_shortfall_min_m": 0.5,
            "endpoint_extension_max_m": 4.0,
        },
        return_diagnostics=True,
    )

    assert diagnostics["endpoint_alignment"]["corrected_endpoint_count"] == 0
    assert diagnostics["endpoint_alignment"]["blocked_endpoint_count"] >= 1


def test_single_endpoint_shortfall_is_detected_when_total_line_is_not_short():
    """One early endpoint must not be hidden by a normal/long opposite end."""
    from path_planner import _connect_work_lines

    lines = [
        [(10.0, 20.0), (100.0, 20.0)],
        [(0.0, 40.0), (80.0, 40.0)],
        [(10.0, 60.0), (100.0, 60.0)],
    ]
    _path, _types, diagnostics = _connect_work_lines(
        lines,
        geo=MetricGeo(),
        state=FieldState(),
        min_turn_radius_m=2.0,
        turn_strategy="pear",
        config={
            "endpoint_short_line_ratio": 0.85,
            "endpoint_shortfall_min_m": 0.5,
            "endpoint_extension_max_m": 4.0,
            "endpoint_outlier_mad_scale": 3.0,
        },
        return_diagnostics=True,
        global_main_angle=0.0,
    )

    alignment = diagnostics["endpoint_alignment"]
    assert alignment["corrected_endpoint_count"] == 1
    assert alignment["corrections"][0]["extension_m"] == pytest.approx(2.0)
    assert alignment["direction_source"] == "global_main_angle"


def test_spacing_below_two_radius_never_selects_semicircle():
    """DECISION-010: omega/2 below R cannot produce an executable semicircle."""
    from path_planner import _select_turn_strategy

    assert _select_turn_strategy("auto", 3.0, 2.0)["strategy"] == "pear"
    assert _select_turn_strategy("bow", 3.0, 2.0)["strategy"] == "pear"
    assert _select_turn_strategy("semicircle", 3.0, 2.0)["strategy"] == "pear"


def test_semicircle_generator_rejects_subminimum_radius():
    """DECISION-010: geometry generation itself guards the minimum radius."""
    from path_planner import _turn_semicircle

    with pytest.raises(ValueError, match="minimum turn radius"):
        _turn_semicircle(
            (0.0, 0.0),
            (0.0, 30.0),
            (1.0, 0.0),
            1.0,
            mpp=0.1,
            min_radius_m=2.0,
        )


def test_boundary_direct_fallback_is_reported_as_kinematic_failure():
    """DECISION-010: a straight connector cannot masquerade as a valid turn."""
    from path_planner import _connect_work_lines

    state = FieldState()
    state.field_boundary = [(5, 5), (80, 5), (80, 80), (5, 80)]
    lines = [
        [(20.0, 20.0), (80.0, 20.0)],
        [(20.0, 30.0), (80.0, 30.0)],
    ]

    path, segment_types, diagnostics = _connect_work_lines(
        lines,
        geo=MetricGeo(),
        state=state,
        min_turn_radius_m=0.5,
        turn_strategy="semicircle",
        config={"turn_outward_clearance_m": 0.6},
        return_diagnostics=True,
    )

    turn = path[segment_types.index("turn")]
    assert len(turn) == 2
    assert diagnostics["feasible"] is False
    assert diagnostics["kinematically_infeasible"] is True
    assert diagnostics["display_only_connectors"]
    connector = diagnostics["display_only_connectors"][0]
    assert connector["display_only_connector"] is True
    assert connector["kinematically_infeasible"] is True
    assert any("straight segment" in reason for reason in diagnostics["hard_reasons"])


def test_fishtail_and_alpha_use_distinct_outer_and_inner_reverse_geometry():
    """DECISION-010: named reverse strategies must not be duplicate templates."""
    from path_planner import _turn_alpha, _turn_fishtail

    arguments = ((0.0, 0.0), (0.0, 30.0), (1.0, 0.0), 1.0, 0.1, 2.0)
    fishtail = np.asarray(_turn_fishtail(*arguments), dtype=np.float64)
    alpha = np.asarray(_turn_alpha(*arguments), dtype=np.float64)

    assert fishtail.shape == alpha.shape
    assert not np.allclose(fishtail, alpha)
