"""DECISION-012/013 cross-module mathematical contract tests."""

import math

import numpy as np
import pytest


def _candidate(strategy, rolling, core):
    return {
        "strategy": strategy,
        "validation": {
            "valid": True,
            "validation_profile": "field_trial",
            "rolling_canopy_pct": rolling,
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


def test_valid_candidate_ranking_uses_expected_canopy_rolling_as_primary_metric():
    from path_planner import _rank_path_candidates

    ranked = _rank_path_candidates([
        _candidate("low_core_but_more_crushing", rolling=4.0, core=1.0),
        _candidate("less_crushing", rolling=1.0, core=7.0),
    ])

    assert ranked[0]["strategy"] == "less_crushing"
    assert ranked[0]["candidate_metrics"]["rolling_canopy_pct"] == 1.0


def test_agronomic_damage_cost_precedes_physical_rolling_for_calibrated_candidates():
    """DECISION-013: planning cost and physical rolling remain distinct."""
    from path_planner import _rank_path_candidates

    lower_physical = _candidate("lower_physical", rolling=1.0, core=1.0)
    lower_physical["validation"]["agronomic_damage_cost_m2"] = 3.0
    lower_agronomic = _candidate("lower_agronomic", rolling=2.0, core=1.0)
    lower_agronomic["validation"]["agronomic_damage_cost_m2"] = 1.0

    ranked = _rank_path_candidates([lower_physical, lower_agronomic])

    assert ranked[0]["strategy"] == "lower_agronomic"
    assert ranked[0]["candidate_metrics"]["rolling_canopy_pct"] == 2.0
    assert ranked[0]["candidate_metrics"]["agronomic_damage_cost_m2"] == 1.0


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


def test_footprint_optimized_layout_preserves_four_row_band_centerlines(monkeypatch):
    import path_planner

    processed = np.ones((20, 30), dtype=np.uint8) * 255
    wide_bands = [{
        "id": 1,
        "contour": np.asarray([[2, 4], [12, 4], [12, 15], [2, 15]], dtype=np.int32),
        "centerline": [(2.0, 9.0), (12.0, 9.0)],
    }]

    def fake_generate_work_lines(*_args, **_kwargs):
        raise AssertionError("footprint geometry must not rewrite work-line references")

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

    work_lines, layout = path_planner._prepare_work_line_layout(
        wide_bands,
        processed,
        0.0,
        geo=Geo(),
        state=State(),
        config={"work_line_mode": "footprint_optimized", "centerline_interval_m": 1.0},
    )

    assert layout["source"] == "four_row_band_centerlines"
    assert layout["work_line_mode"] == "footprint_optimized"
    assert work_lines[0][0] == pytest.approx((2.0, 9.0))
    assert work_lines[0][-1] == pytest.approx((12.0, 9.0))
