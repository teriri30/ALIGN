"""DECISION-012/013 cross-module mathematical contract tests."""

import math

import numpy as np
import pytest


def test_path_crossing_count_uses_segment_types_when_entry_or_aux_segments_exist():
    from path_planner import _count_path_intersections_proper

    full_path = [
        [(0.0, 0.0), (10.0, 0.0)],   # entry
        [(0.0, 2.0), (10.0, 2.0)],   # work 1
        [(5.0, -1.0), (5.0, 3.0)],  # turn
        [(7.0, 1.0), (7.0, 9.0)],   # work 2; parallel to the other work line
        [(0.0, 5.0), (10.0, 5.0)],  # exit/auxiliary geometry
    ]
    segment_types = ["entry", "work", "turn", "work", "exit"]

    assert _count_path_intersections_proper(full_path, segment_types) == 0


def _candidate(strategy, rolling, core):
    return {
        "strategy": strategy,
        "validation": {
            "valid": True,
            "validation_profile": "field_trial",
            "rolling_canopy_pct": rolling,
            "rolling_body_pct": rolling,
            "track_core_overlap_pct": core,
            "track_outside_field_pct": 0.0,
            "track_outside_support_pct": 0.0,
            "track_uncertain_overlap_pct": 0.0,
            "track_forbidden_overlap_pct": 0.0,
            "crop_mask_geometric_coverage_pct": 96.0,
            "total_length_m": 100.0,
        },
        "turn_assessment": {},
        "segment_types": ["work"],
    }


def test_valid_candidate_ranking_uses_body_rolling_as_primary_crop_metric():
    from path_planner import _rank_path_candidates

    ranked = _rank_path_candidates([
        _candidate("low_core_but_more_crushing", rolling=4.0, core=1.0),
        _candidate("less_crushing", rolling=1.0, core=7.0),
    ])

    assert ranked[0]["strategy"] == "less_crushing"
    assert ranked[0]["candidate_metrics"]["rolling_body_pct"] == 1.0


def test_strategy_ranking_ignores_headland_rolling_and_uses_body_risk():
    from path_planner import _rank_path_candidates

    low_headland = _candidate("low_headland", rolling=1.0, core=2.0)
    low_headland["validation"].update({
        "rolling_body_pct": 4.0,
        "rolling_headland_pct": 0.0,
        "agronomic_damage_cost_m2": 1.0,
        "total_length_m": 100.0,
    })
    high_headland = _candidate("high_headland", rolling=80.0, core=2.0)
    high_headland["validation"].update({
        "rolling_body_pct": 1.0,
        "rolling_headland_pct": 95.0,
        "agronomic_damage_cost_m2": 500.0,
        "total_length_m": 90.0,
    })

    ranked = _rank_path_candidates([low_headland, high_headland])

    assert ranked[0]["strategy"] == "high_headland"
    assert ranked[0]["candidate_metrics"]["rolling_body_pct"] == 1.0


def test_headland_weighted_damage_is_report_only_for_strategy_ranking():
    from path_planner import _rank_path_candidates

    lower_physical = _candidate("lower_physical", rolling=1.0, core=1.0)
    lower_physical["validation"]["agronomic_damage_cost_m2"] = 500.0
    lower_agronomic = _candidate("lower_agronomic", rolling=2.0, core=1.0)
    lower_agronomic["validation"]["agronomic_damage_cost_m2"] = 1.0

    ranked = _rank_path_candidates([lower_physical, lower_agronomic])

    assert ranked[0]["strategy"] == "lower_physical"
    assert ranked[0]["candidate_metrics"]["rolling_body_pct"] == 1.0
    assert "agronomic_damage_cost_m2" not in ranked[0]["candidate_metrics"]


def test_headland_weight_changes_planning_cost_without_changing_physical_rolling():
    """DECISION-013: a planning weight must never rewrite measured rolling."""
    from footprint_planner import validate_footprints

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [(0, 0), (119, 0), (119, 79), (0, 79)]

    body = np.zeros((80, 120), dtype=np.uint8)
    body[35:45, 50:100] = 255
    headland = np.zeros_like(body)
    headland[10:70, 2:16] = 255
    path = [[(8.0, 15.0), (8.0, 65.0)]]
    harvester = {
        "cutter_width_m": 2.0,
        "cutter_forward_offset_m": 0.5,
        "track_width_m": 0.8,
        "track_gauge_m": 0.0,
        "wheelbase_m": 0.5,
        "track_length_m": 0.8,
        "turn_radius_m": 0.5,
    }

    conservative = validate_footprints(
        path, body, geo=Geo(), state=State(), harvester=harvester,
        headland_mask=headland, segment_types=["turn"],
        config={"headland_damage_weight": 1.0},
    )
    calibrated = validate_footprints(
        path, body, geo=Geo(), state=State(), harvester=harvester,
        headland_mask=headland, segment_types=["turn"],
        config={"headland_damage_weight": 0.5},
    )

    assert conservative["rolling_body_pct"] == 0.0
    assert conservative["rolling_headland_pct"] > 0.0
    assert conservative["rolling_canopy_pct"] == pytest.approx(
        calibrated["rolling_canopy_pct"]
    )
    assert conservative["physical_crop_rolling_area_m2"] == pytest.approx(
        calibrated["physical_crop_rolling_area_m2"]
    )
    assert calibrated["agronomic_damage_cost_m2"] < conservative["agronomic_damage_cost_m2"]


def test_headland_damage_weight_rejects_zero_cost_crop():
    """DECISION-013: headland crop can never be treated as a free road."""
    from config import ConfigValidationError, validate_config

    with pytest.raises(ConfigValidationError):
        validate_config({"path_planning": {"headland_damage_weight": 0.0}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_numbers(value):
    from config import ConfigValidationError, validate_config

    with pytest.raises(ConfigValidationError, match="finite"):
        validate_config({"path_planning": {"max_track_core_overlap_pct": value}})


@pytest.mark.parametrize("key,value", [
    ("lateral_search_m", float("nan")),
    ("lateral_search_m", -0.1),
    ("lateral_step_m", 0.0),
    ("lateral_step_m", float("inf")),
    ("lateral_smooth_penalty", -0.1),
])
def test_config_rejects_invalid_lateral_optimization_numbers(key, value):
    from config import ConfigValidationError, validate_config

    with pytest.raises(ConfigValidationError):
        validate_config({"path_planning": {key: value}})


def test_work_line_mode_uses_truthful_footprint_validation_name(monkeypatch):
    import path_planner

    processed = np.ones((20, 30), dtype=np.uint8) * 255
    wide_bands = [{
        "id": 1,
        "contour": np.asarray([[2, 4], [12, 4], [12, 15], [2, 15]], dtype=np.int32),
        "centerline": [(2.0, 9.0), (12.0, 9.0)],
    }]

    class State:
        mask_raw = processed
        mask_offset_x = 0
        mask_offset_y = 0
        harvester_params = {}

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    lines, layout = path_planner._prepare_work_line_layout(
        wide_bands,
        processed,
        0.0,
        geo=Geo(),
        state=State(),
        config={"work_line_mode": "band_centerline_with_footprint_validation"},
    )

    assert lines
    assert layout["work_line_mode"] == "band_centerline_with_footprint_validation"
    assert layout["footprint_optimization_scope"] == "validation_only"


def test_headland_space_precheck_marks_outward_probe_outside_field_as_limited():
    from path_planner import _headland_space_is_limited
    import numpy as np

    field = np.asarray([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], dtype=float)

    assert _headland_space_is_limited(
        start=(9.5, 5.0),
        work_direction=(1.0, 0.0),
        field_polygon=field,
        required_depth_px=2.0,
    ) is True
    assert _headland_space_is_limited(
        start=(5.0, 5.0),
        work_direction=(1.0, 0.0),
        field_polygon=field,
        required_depth_px=2.0,
    ) is False


def test_candidate_dedup_uses_actual_strategy_sequence_and_geometry():
    from path_planner import _deduplicate_path_candidates

    base = {
        "path_segments": [[(0.0, 0.0), (1.0, 0.0)]],
        "segment_types": ["turn"],
        "validation": {"valid": True},
        "turn_assessment": {"used_strategies": ["pear"]},
    }
    candidates = [
        {**base, "strategy": "bow"},
        {**base, "strategy": "semicircle"},
        {**base, "strategy": "pear"},
    ]

    unique = _deduplicate_path_candidates(candidates)

    assert len(unique) == 1
    assert unique[0]["equivalent_candidate_labels"] == ["bow", "semicircle", "pear"]


def test_unload_points_are_declared_as_end_route_candidates():
    from path_planner import _build_planning_factor_report

    class State:
        harvester_params = {
            "cutter_width_m": 2.15,
            "cutter_forward_offset_m": 1.0,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "wheelbase_m": 1.8,
            "track_length_m": 1.7,
            "turn_radius_m": 0.9,
        }
        mask_offset_x = 0
        mask_offset_y = 0
        entry_point = (0.0, 0.0)
        exit_point = (10.0, 10.0)
        unload_points = [(8.0, 8.0), (9.0, 9.0)]
        unload_point = None

    summary = _build_planning_factor_report(State(), [], None)

    assert summary["service_points"]["unload_semantics"] == "select_one_nearest_route_end"
    assert summary["service_points"]["unload_count"] == 2


def test_candidate_ranking_rejects_non_finite_metrics_deterministically():
    from path_planner import _rank_path_candidates

    invalid = _candidate("invalid", rolling=float("nan"), core=0.0)
    invalid["validation"]["rolling_body_pct"] = float("nan")
    finite = _candidate("finite", rolling=1.0, core=0.0)

    first = _rank_path_candidates([invalid, finite])
    second = _rank_path_candidates([finite, invalid])

    assert first[0]["strategy"] == "finite"
    assert second[0]["strategy"] == "finite"
    assert first[-1]["validation"]["valid"] is False


def test_plan_worker_executes_the_same_persisted_headland_weight_it_records(monkeypatch):
    """DECISION-013: runtime planning and provenance use one config snapshot."""
    from config import Config
    from pyside6_app.workers import PlanWorker

    monkeypatch.setattr(
        Config,
        "section",
        lambda self, name: (
            {"headland_damage_weight": 0.75, "validation_profile": "field_trial"}
            if name == "path_planning"
            else {}
        ),
    )
    worker = PlanWorker({}, None, None, {"min_turn_radius_m": 1.2})

    assert worker.config["headland_damage_weight"] == 0.75
    assert worker.config["validation_profile"] == "field_trial"
    assert worker.config["min_turn_radius_m"] == 1.2


def test_metric_geometry_rejects_missing_georeferenced_scale():
    from row_geometry import meters_per_pixel

    with pytest.raises(ValueError, match="metric scale"):
        meters_per_pixel(np.zeros((20, 30), dtype=np.uint8))


def test_planar_geometry_rejects_incomplete_physical_parameters():
    from harvester_geometry import planar_geometry_m

    with pytest.raises(ValueError, match="missing harvester geometry"):
        planar_geometry_m({"cutter_width_m": 2.0})


def test_cutter_is_a_measured_offset_line_without_invented_depth():
    from harvester_geometry import planar_geometry_m

    geometry = planar_geometry_m({
        "cutter_width_m": 2.2,
        "cutter_forward_offset_m": 1.35,
        "track_width_m": 0.4,
        "track_gauge_m": 1.6,
        "wheelbase_m": 1.8,
        "track_length_m": 1.7,
        "turn_radius_m": 0.9,
    })

    cutter = geometry["cutter_line"]
    cutter_visual = geometry["cutter_visual"]
    cutter_visual_edge = geometry["cutter_visual_edge"]
    assert cutter.shape == (2, 2)
    assert cutter[:, 0] == pytest.approx([1.35, 1.35])
    assert np.ptp(cutter[:, 1]) == pytest.approx(2.2)
    assert cutter_visual.shape == (4, 2)
    assert np.min(cutter_visual[:, 0]) == pytest.approx(np.max(geometry["body"][:, 0]))
    assert np.max(cutter_visual[:, 0]) < np.max(cutter[:, 0])
    assert cutter_visual_edge[:, 0] == pytest.approx([np.max(cutter_visual[:, 0])] * 2)
    assert "cutter" not in geometry
    assert "cutter_depth_m" not in geometry["dimensions_m"]


def test_machine_readiness_requires_current_footprint_contract():
    from machine_route_validator import assess_machine_readiness

    result = {
        "is_valid": True,
        "validation": {
            "valid": True,
            "validation_profile": "field_trial",
            "field_boundary_present": True,
            "semantic_support_present": True,
            "forbidden_mask_present": True,
            "footprint_resolution_m": 0.02,
        },
        "layout": {"work_line_mode": "footprint_optimized"},
        "turn_assessment": {"hard_reasons": []},
        "planning_factors": {"harvester": {"track_width_m": 0.4}},
    }

    readiness = assess_machine_readiness(
        result, {"forbidden_mask_confirmed": True}
    )

    assert readiness["controlled_trial_ready"] is False
    assert "footprint_contract_mismatch" in readiness["controlled_trial_blockers"]


def test_legacy_rolling_crop_alias_is_not_exported_as_a_second_definition():
    from footprint_planner import validate_footprints

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [(0, 0), (79, 0), (79, 79), (0, 79)]

    result = validate_footprints(
        [[(10.0, 40.0), (70.0, 40.0)]],
        np.ones((80, 80), dtype=np.uint8) * 255,
        geo=Geo(),
        state=State(),
        harvester={
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": 1.2,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "wheelbase_m": 1.8,
            "track_length_m": 1.7,
            "turn_radius_m": 0.9,
        },
        segment_types=["work"],
    )

    assert "rolling_crop_pct" not in result
    assert "harvest_coverage_pct" not in result
    assert "crop_mask_geometric_coverage_pct" in result
    assert math.isfinite(result["rolling_canopy_pct"])


def test_cutter_line_forward_offset_controls_terminal_geometric_coverage():
    from footprint_planner import validate_footprints

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [(0, 0), (99, 0), (99, 79), (0, 79)]

    crop = np.zeros((80, 100), dtype=np.uint8)
    crop[35:46, 70:76] = 255
    common = {
        "cutter_width_m": 2.0,
        "track_width_m": 0.4,
        "track_gauge_m": 1.6,
        "wheelbase_m": 1.8,
        "track_length_m": 1.7,
        "turn_radius_m": 0.9,
    }
    path = [[(20.0, 40.0), (60.0, 40.0)]]
    near = validate_footprints(
        path, crop, geo=Geo(), state=State(),
        harvester={**common, "cutter_forward_offset_m": 0.5},
        segment_types=["work"],
    )
    far = validate_footprints(
        path, crop, geo=Geo(), state=State(),
        harvester={**common, "cutter_forward_offset_m": 1.5},
        segment_types=["work"],
    )

    assert near["crop_mask_geometric_coverage_pct"] == 0.0
    assert far["crop_mask_geometric_coverage_pct"] > 90.0


def test_work_line_reference_stays_on_crop_centerline_not_cutter_front():
    from path_planner import _connect_work_lines

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    class State:
        mask_raw = np.zeros((40, 40), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = []
        entry_point = None
        exit_point = None
        unload_points = []
        unload_point = None
        harvester_params = {"cutter_forward_offset_m": 2.0}

    path, segment_types, diagnostics = _connect_work_lines(
        [[(10.0, 20.0), (30.0, 20.0)]],
        geo=Geo(),
        state=State(),
        return_diagnostics=True,
    )

    work = path[segment_types.index("work")]
    assert work[0] == pytest.approx((10.0, 20.0))
    assert work[-1] == pytest.approx((30.0, 20.0))
    assert diagnostics["endpoint_alignment"]["path_reference_shift_applied"] is False


def test_footprint_optimized_layout_optimizes_from_four_row_band_centerlines(monkeypatch):
    """DECISION-011: real optimization starts from accepted band lines."""
    import path_planner

    processed = np.ones((20, 30), dtype=np.uint8) * 255
    wide_bands = [{
        "id": 1,
        "contour": np.asarray([[2, 4], [12, 4], [12, 15], [2, 15]], dtype=np.int32),
        "centerline": [(2.0, 9.0), (12.0, 9.0)],
    }]
    calls = []

    def fake_generate_work_lines(*args, **kwargs):
        calls.append((args, kwargs))
        baselines = kwargs["baseline_lines"]
        assert baselines[0][0] == pytest.approx((2.0, 9.0))
        assert baselines[0][-1] == pytest.approx((12.0, 9.0))
        return [[(2.0, 8.5), (12.0, 8.5)]], {
            "source": "four_row_band_centerlines_lateral_optimization",
            "footprint_optimization_scope": "centerline_anchored_lateral_search",
            "optimized_pass_count": 1,
            "line_optimization": [
                {"line_index": 0, "optimized": True},
            ],
        }

    class State:
        mask_raw = processed
        harvester_params = {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": 1.0,
            "track_width_m": 0.4,
            "track_gauge_m": 1.0,
            "wheelbase_m": 1.8,
            "track_length_m": 1.7,
            "turn_radius_m": 0.9,
        }

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    monkeypatch.setattr(path_planner, "generate_work_lines", fake_generate_work_lines)
    formal_results = iter([
        {"track_core_overlap_pct": 10.0, "planned_target_geometric_coverage_pct": 70.0,
         "track_outside_field_pct": 0.0, "track_outside_support_pct": 0.0,
         "track_uncertain_overlap_pct": 0.0, "track_forbidden_overlap_pct": 0.0},
        {"track_core_overlap_pct": 9.0, "planned_target_geometric_coverage_pct": 75.0,
         "track_outside_field_pct": 0.0, "track_outside_support_pct": 0.0,
         "track_uncertain_overlap_pct": 0.0, "track_forbidden_overlap_pct": 0.0},
    ])
    monkeypatch.setattr(
        path_planner, "validate_footprints",
        lambda *_args, **_kwargs: next(formal_results),
    )

    work_lines, layout = path_planner._prepare_work_line_layout(
        wide_bands,
        processed,
        0.0,
        geo=Geo(),
        state=State(),
        config={"work_line_mode": "footprint_optimized", "centerline_interval_m": 1.0},
    )

    assert len(calls) == 1
    assert layout["source"] == "four_row_band_centerlines_lateral_optimization"
    assert layout["work_line_mode"] == "footprint_optimized"
    assert layout["footprint_optimization_scope"] == "centerline_anchored_lateral_search"
    assert layout["ordered_band_ids"] == [1]
    assert layout["line_optimization"] == [
        {"line_index": 0, "band_id": 1, "optimized": True},
    ]
    assert work_lines[0][0] == pytest.approx((2.0, 8.5))
    assert work_lines[0][-1] == pytest.approx((12.0, 8.5))


def test_real_footprint_optimization_keeps_centerline_topology_and_search_bounds():
    from footprint_planner import generate_work_lines

    processed = np.zeros((80, 120), dtype=np.uint8)
    processed[12:28, 5:115] = 255
    processed[52:68, 5:115] = 255
    baselines = [
        [(10.0, 20.0), (110.0, 20.0)],
        [(10.0, 60.0), (110.0, 60.0)],
    ]

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.10

    class State:
        field_boundary = [(0.0, 0.0), (119.0, 0.0), (119.0, 79.0), (0.0, 79.0)]
        mask_offset_x = 0
        mask_offset_y = 0
        forbidden_regions = []

    optimized, layout = generate_work_lines(
        processed,
        0.0,
        geo=Geo(),
        state=State(),
        harvester={
            "cutter_width_m": 1.6,
            "track_width_m": 0.4,
            "track_gauge_m": 1.0,
        },
        config={
            "planning_resolution_m": 0.10,
            "planning_max_dim": 1000,
            "station_interval_m": 0.50,
            "station_window_m": 0.20,
            "lateral_search_m": 0.30,
            "lateral_step_m": 0.10,
            "lateral_smooth_penalty": 0.12,
            "path_smooth_m": 0.50,
            "headland_buffer_m": 0.0,
        },
        baseline_lines=baselines,
    )

    assert layout["optimization_complete"] is True
    assert len(optimized) == len(baselines)
    assert len(layout["line_optimization"]) == len(baselines)
    assert all(
        item["baseline_objective"] is None
        or item["optimized_objective"] <= item["baseline_objective"] + 1e-9
        for item in layout["line_optimization"]
    )
    for baseline, line in zip(baselines, optimized):
        assert line[0][0] == pytest.approx(baseline[0][0], abs=1.0)
        assert line[-1][0] == pytest.approx(baseline[-1][0], abs=1.0)
        baseline_y = baseline[0][1]
        assert max(abs(point[1] - baseline_y) for point in line) <= 3.0 + 1e-6


def test_footprint_comparison_rejects_any_degraded_safety_metric():
    from footprint_planner import compare_work_line_footprints

    baseline = {
        "track_core_overlap_pct": 10.0,
        "planned_target_geometric_coverage_pct": 70.0,
        "track_outside_field_pct": 1.0,
        "track_outside_support_pct": 5.0,
        "track_uncertain_overlap_pct": 4.0,
        "track_forbidden_overlap_pct": 0.0,
    }
    optimized = dict(baseline)
    optimized.update({"track_core_overlap_pct": 10.5, "planned_target_geometric_coverage_pct": 75.0})

    comparison = compare_work_line_footprints(baseline, optimized)

    assert comparison["accepted"] is False
    assert comparison["degraded_metrics"] == ["track_core_overlap_pct"]


def test_footprint_comparison_rejects_missing_required_metric():
    from footprint_planner import compare_work_line_footprints

    baseline = {
        "planned_target_geometric_coverage_pct": 70.0,
        "track_outside_field_pct": 0.0,
    }
    optimized = {
        "planned_target_geometric_coverage_pct": 75.0,
        "track_outside_field_pct": 0.0,
    }

    comparison = compare_work_line_footprints(baseline, optimized)

    assert comparison["accepted"] is False
    assert comparison["invalid_metrics"] == ["track_core_overlap_pct"]


def test_footprint_comparison_accepts_non_degrading_strict_improvement():
    from footprint_planner import compare_work_line_footprints

    baseline = {
        "track_core_overlap_pct": 10.0,
        "planned_target_geometric_coverage_pct": 70.0,
        "track_outside_field_pct": 1.0,
        "track_outside_support_pct": 5.0,
        "track_uncertain_overlap_pct": 4.0,
        "track_forbidden_overlap_pct": 0.0,
    }
    optimized = dict(baseline)
    optimized.update({
        "track_core_overlap_pct": 9.0,
        "planned_target_geometric_coverage_pct": 75.0,
        "track_outside_support_pct": 4.0,
    })

    comparison = compare_work_line_footprints(baseline, optimized)

    assert comparison["accepted"] is True
    assert set(comparison["improved_metrics"]) == {
        "track_core_overlap_pct",
        "planned_target_geometric_coverage_pct",
        "track_outside_support_pct",
    }


def test_interval_mean_treats_outside_profile_as_zero_without_index_error():
    from footprint_planner import _interval_mean


    cumulative = np.asarray([0.0, 1.0, 3.0, 6.0])

    assert _interval_mean(cumulative, -5, -1) == 0.0
    assert _interval_mean(cumulative, 4, 8) == 0.0
    assert _interval_mean(cumulative, -2, 2) == pytest.approx(0.75)
    assert _interval_mean(cumulative, 2, 8) == pytest.approx(0.5)


def test_centerline_optimizer_rejects_mismatched_semantic_layer_shape():
    from footprint_planner import _optimize_baseline_line

    optimized, diagnostics = _optimize_baseline_line(
        np.asarray([(1.0, 5.0), (8.0, 5.0)]),
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
        np.ones((10, 10), dtype=np.uint8),
        None,
        np.zeros((8, 10), dtype=np.uint8),
        0.1,
        1.6,
        0.4,
        0.8,
        {},
    )

    assert optimized is None
    assert diagnostics["reason"] == "uncertain_shape_mismatch"


def test_centerline_optimizer_avoids_uncertain_track_corridor():
    from footprint_planner import _optimize_baseline_line


    shape = (41, 61)
    crop = np.zeros(shape, dtype=np.uint8)
    core = np.zeros(shape, dtype=np.uint8)
    support = np.ones(shape, dtype=np.uint8) * 255
    uncertain = np.zeros(shape, dtype=np.uint8)
    uncertain[14:18, :] = 255
    uncertain[22:26, :] = 255
    baseline = np.asarray([(5.0, 20.0), (55.0, 20.0)], dtype=np.float64)

    optimized, diagnostics = _optimize_baseline_line(
        baseline,
        crop,
        core,
        support,
        None,
        uncertain,
        0.10,
        1.6,
        0.4,
        0.8,
        {
            "station_interval_m": 0.5,
            "station_window_m": 0.2,
            "lateral_search_m": 0.5,
            "lateral_step_m": 0.1,
            "lateral_smooth_penalty": 0.1,
            "path_smooth_m": 0.5,
            "lateral_uncertain_penalty": 20.0,
        },
    )

    assert optimized is not None
    assert diagnostics["max_abs_lateral_shift_m"] > 0.0
    assert np.mean(optimized[:, 1]) != pytest.approx(20.0)


def test_work_line_layout_falls_back_when_formal_metric_degrades(monkeypatch):
    import path_planner

    processed = np.ones((20, 30), dtype=np.uint8) * 255
    wide_bands = [{"id": 1, "centerline": [(2.0, 9.0), (12.0, 9.0)]}]

    class State:
        mask_raw = processed
        harvester_params = {
            "cutter_width_m": 2.0, "cutter_forward_offset_m": 1.0,
            "track_width_m": 0.4, "track_gauge_m": 1.0,
            "wheelbase_m": 1.8, "track_length_m": 1.7, "turn_radius_m": 0.9,
        }

    class Geo:
        def is_ready(self): return True
        def meters_per_pixel(self, _x, _y): return 1.0

    monkeypatch.setattr(path_planner, "generate_work_lines", lambda *_a, **_k: (
        [[(2.0, 8.5), (12.0, 8.5)]], {
            "optimization_complete": True,
            "line_optimization": [
                {"line_index": 0, "optimized": True},
            ],
        },
    ))
    validations = iter([
        {"track_core_overlap_pct": 10.0, "planned_target_geometric_coverage_pct": 70.0,
         "track_outside_field_pct": 0.0, "track_outside_support_pct": 0.0,
         "track_uncertain_overlap_pct": 0.0, "track_forbidden_overlap_pct": 0.0},
        {"track_core_overlap_pct": 11.0, "planned_target_geometric_coverage_pct": 75.0,
         "track_outside_field_pct": 0.0, "track_outside_support_pct": 0.0,
         "track_uncertain_overlap_pct": 0.0, "track_forbidden_overlap_pct": 0.0},
    ])
    monkeypatch.setattr(path_planner, "validate_footprints", lambda *_a, **_k: next(validations))

    lines, layout = path_planner._prepare_work_line_layout(
        wide_bands, processed, 0.0, geo=Geo(), state=State(),
        config={"work_line_mode": "footprint_optimized"},
    )

    assert lines[0][0] == pytest.approx((2.0, 9.0))
    assert layout["work_line_mode"] == "band_centerline_with_footprint_validation"
    assert layout["optimization_fallback_reason"] == "formal_footprint_metric_degraded"
    assert layout["formal_comparison"]["degraded_metrics"] == ["track_core_overlap_pct"]


def test_work_line_formal_comparison_uses_final_validation_masks(monkeypatch):
    """DECISION-011: selection and final validation use identical mask evidence."""
    import path_planner

    processed = np.ones((20, 30), dtype=np.uint8) * 255
    detected = np.zeros_like(processed)
    detected[2:18, 2:28] = 255
    planned = np.zeros_like(processed)
    planned[6:14, 3:27] = 255
    wide_bands = [{"id": 1, "centerline": [(2.0, 9.0), (12.0, 9.0)]}]

    class State:
        mask_raw = processed
        harvester_params = {
            "cutter_width_m": 2.0, "cutter_forward_offset_m": 1.0,
            "track_width_m": 0.4, "track_gauge_m": 1.0,
            "wheelbase_m": 1.8, "track_length_m": 1.7,
            "turn_radius_m": 0.9,
        }

    class Geo:
        def is_ready(self): return True
        def meters_per_pixel(self, _x, _y): return 1.0

    monkeypatch.setattr(path_planner, "generate_work_lines", lambda *_a, **_k: (
        [[(2.0, 8.5), (12.0, 8.5)]], {
            "optimization_complete": True,
            "line_optimization": [
                {"line_index": 0, "optimized": True},
            ],
        },
    ))
    calls = []

    def fake_validate(*_args, **kwargs):
        calls.append(kwargs)
        core = 10.0 if len(calls) == 1 else 9.0
        return {
            "track_core_overlap_pct": core,
            "planned_target_geometric_coverage_pct": 95.0,
            "track_outside_field_pct": 0.0,
            "track_outside_support_pct": 0.0,
            "track_uncertain_overlap_pct": 0.0,
            "track_forbidden_overlap_pct": 0.0,
        }

    monkeypatch.setattr(path_planner, "validate_footprints", fake_validate)

    path_planner._prepare_work_line_layout(
        wide_bands, processed, 0.0, geo=Geo(), state=State(),
        config={"work_line_mode": "footprint_optimized"},
        detected_mask=detected, planned_mask=planned,
    )

    assert len(calls) == 2
    assert all(call["detected_mask"] is detected for call in calls)
    assert all(call["planned_mask"] is planned for call in calls)


def test_incomplete_footprint_optimization_falls_back_truthfully(monkeypatch):
    import path_planner



    processed = np.ones((20, 30), dtype=np.uint8) * 255
    wide_bands = [{
        "id": 1,
        "centerline": [(2.0, 9.0), (12.0, 9.0)],
    }]

    class State:
        mask_raw = processed
        harvester_params = {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": 1.0,
            "track_width_m": 0.4,
            "track_gauge_m": 1.0,
        }

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    monkeypatch.setattr(
        path_planner,
        "generate_work_lines",
        lambda *_args, **_kwargs: ([], {
            "optimization_complete": False,
            "reason": "no_safe_lateral_candidate",
        }),
    )

    work_lines, layout = path_planner._prepare_work_line_layout(
        wide_bands,
        processed,
        0.0,
        geo=Geo(),
        state=State(),
        config={"work_line_mode": "footprint_optimized"},
    )

    assert work_lines
    assert layout["work_line_mode"] == "band_centerline_with_footprint_validation"
    assert layout["optimization_complete"] is False
    assert layout["optimization_fallback_reason"] == "no_safe_lateral_candidate"


def test_crossing_optimized_centerlines_fall_back_before_formal_comparison(monkeypatch):
    """DECISION-011: independent lateral shifts may not cross or reorder passes."""
    import path_planner

    processed = np.ones((24, 30), dtype=np.uint8) * 255
    wide_bands = [
        {"id": 1, "centerline": [(2.0, 8.0), (12.0, 8.0)]},
        {"id": 2, "centerline": [(2.0, 14.0), (12.0, 14.0)]},
    ]

    class State:
        mask_raw = processed
        harvester_params = {
            "cutter_width_m": 2.0, "cutter_forward_offset_m": 1.0,
            "track_width_m": 0.4, "track_gauge_m": 1.0,
            "wheelbase_m": 1.8, "track_length_m": 1.7,
            "turn_radius_m": 0.9,
        }

    class Geo:
        def is_ready(self): return True
        def meters_per_pixel(self, _x, _y): return 1.0

    monkeypatch.setattr(path_planner, "generate_work_lines", lambda *_a, **_k: (
        [
            [(2.0, 7.0), (12.0, 15.0)],
            [(2.0, 15.0), (12.0, 7.0)],
        ],
        {"optimization_complete": True},
    ))
    monkeypatch.setattr(
        path_planner,
        "validate_footprints",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("crossing work lines must fail before formal comparison")
        ),
    )

    lines, layout = path_planner._prepare_work_line_layout(
        wide_bands, processed, 0.0, geo=Geo(), state=State(),
        config={"work_line_mode": "footprint_optimized"},
    )

    assert len(lines) == 2
    assert layout["work_line_mode"] == "band_centerline_with_footprint_validation"
    assert layout["optimization_complete"] is False
    assert layout["optimization_fallback_reason"] == "optimized_centerline_topology_violation"
