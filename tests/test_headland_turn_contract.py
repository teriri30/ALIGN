import numpy as np
import pytest


def test_headland_turn_config_rejects_non_boolean_switch():
    from config import ConfigValidationError, validate_config

    with pytest.raises(ConfigValidationError, match="headland_turn_optimization_enabled"):
        validate_config({"path_planning": {"headland_turn_optimization_enabled": "yes"}})


def test_connect_work_lines_reports_missing_headland_evidence_without_changing_turn():
    import path_planner

    class Geo:
        def is_ready(self):
            return True

        def meters_per_pixel(self, _x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        mask_raw = np.ones((80, 120), dtype=np.uint8) * 255
        field_boundary = [(0.0, 0.0), (119.0, 0.0), (119.0, 79.0), (0.0, 79.0)]
        harvester_params = {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": 0.8,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "wheelbase_m": 1.8,
            "track_length_m": 1.6,
            "turn_radius_m": 1.0,
        }

    mask = State.mask_raw
    before, before_types = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=Geo(), state=State(), min_turn_radius_m=1.0, turn_strategy="bow",
        config={"headland_turn_optimization_enabled": False}, processed_mask=mask,
    )
    after, after_types, diagnostics = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=Geo(), state=State(), min_turn_radius_m=1.0, turn_strategy="bow",
        config={"headland_turn_optimization_enabled": True}, processed_mask=mask,
        return_diagnostics=True,
    )

    optimization = diagnostics["turn_decisions"][0]["headland_turn_optimization"]
    assert optimization["optimization_applied"] is False
    assert optimization["fallback_reason"] == "headland_mask_missing"
    assert after == before
    assert after_types == before_types


def test_headland_turn_candidates_reject_forbidden_and_outside_field(monkeypatch):
    import path_planner

    class Geo:
        def is_ready(self):
            return True

        def meters_per_pixel(self, _x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        mask_raw = np.ones((80, 120), dtype=np.uint8) * 255
        field_boundary = [(0.0, 0.0), (119.0, 0.0), (119.0, 79.0), (0.0, 79.0)]
        harvester_params = {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": 0.8,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "wheelbase_m": 1.8,
            "track_length_m": 1.6,
            "turn_radius_m": 1.0,
        }

    mask = State.mask_raw
    headland = np.zeros_like(mask)
    headland[34:57, 35:85] = 255
    forbidden = np.zeros_like(mask)
    forbidden[0:80, 45:75] = 255
    monkeypatch.setattr(
        path_planner,
        "validate_footprints",
        lambda *_args, **_kwargs: {
            "track_outside_field_pct": 0.0,
            "track_forbidden_overlap_pct": 100.0,
            "turn_outside_field_pct": 0.0,
            "turn_forbidden_overlap_area_m2": 1.0,
            "headland_turn_track_overlap_area_m2": 0.0,
            "turn_cost_m2": 0.0,
        },
    )

    _, _, diagnostics = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=Geo(), state=State(), min_turn_radius_m=1.0, turn_strategy="bow",
        config={
            "headland_turn_optimization_enabled": True,
            "headland_mask": headland,
            "planning_support_mask": mask,
            "uncertain_mask": np.zeros_like(mask),
            "forbidden_mask": forbidden,
        },
        processed_mask=mask, return_diagnostics=True, global_main_angle=0.0,
    )

    optimization = diagnostics["turn_decisions"][0]["headland_turn_optimization"]
    assert optimization["optimization_applied"] is False
    assert optimization["fallback_reason"] == "no_safe_turn_candidate"
    assert optimization["rejected_candidate_count"] >= 1


# DECISION-015: turn fallback and hard-constraint behavior are invariant-tested.

def test_headland_turn_candidates_reject_invalid_formal_validation(monkeypatch):
    """DECISION-011/015: invalid formal footprint validation is a hard rejection."""
    import path_planner

    class Geo:
        def is_ready(self):
            return True

        def meters_per_pixel(self, _x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        mask_raw = np.ones((80, 120), dtype=np.uint8) * 255
        field_boundary = [(0.0, 0.0), (119.0, 0.0), (119.0, 79.0), (0.0, 79.0)]
        harvester_params = {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": 0.8,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "wheelbase_m": 1.8,
            "track_length_m": 1.6,
            "turn_radius_m": 1.0,
        }

    mask = State.mask_raw
    headland = np.zeros_like(mask)
    headland[34:57, 35:85] = 255

    def invalid_validation(*_args, **_kwargs):
        return {
            "valid": False,
            "issues": ["formal coverage gate failed"],
            "validation_profile": "field_trial",
            "track_core_overlap_pct": 99.0,
            "planned_target_geometric_coverage_pct": 0.0,
            "track_outside_field_pct": 0.0,
            "track_outside_support_pct": 0.0,
            "track_uncertain_overlap_pct": 0.0,
            "track_forbidden_overlap_pct": 0.0,
            "headland_turn_track_overlap_area_m2": 0.01,
            "turn_cost_m2": 0.01,
            "turn_body_overlap_area_m2": 0.0,
            "turn_uncertain_overlap_area_m2": 0.0,
            "turn_repeat_twice_or_more_area_m2": 0.0,
            "total_length_m": 1.0,
        }

    monkeypatch.setattr(path_planner, "validate_footprints", invalid_validation)
    _, _, diagnostics = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=Geo(), state=State(), min_turn_radius_m=1.0, turn_strategy="bow",
        config={
            "headland_turn_optimization_enabled": True,
            "headland_mask": headland,
            "planning_support_mask": mask,
            "uncertain_mask": np.zeros_like(mask),
            "forbidden_mask": np.zeros_like(mask),
            "validation_profile": "field_trial",
        },
        processed_mask=mask, return_diagnostics=True, global_main_angle=0.0,
    )

    optimization = diagnostics["turn_decisions"][0]["headland_turn_optimization"]
    assert optimization["optimization_applied"] is False
    assert optimization["fallback_reason"] == "no_safe_turn_candidate"
    assert optimization["rejected_candidate_count"] >= 1
    assert any(
        "validation_invalid" in reason
        for item in optimization["rejected_candidates"]
        for reason in item["hard_reasons"]
    )
    assert any(
        "formal coverage gate failed" in issue
        for item in optimization["rejected_candidates"]
        for issue in item.get("validation_issues", [])
    )
def test_headland_turn_candidate_missing_core_metric_fails_closed(monkeypatch):
    """DECISION-011/015: a required formal metric cannot default to zero."""
    import path_planner

    class Geo:
        def is_ready(self):
            return True

        def meters_per_pixel(self, _x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        mask_raw = np.ones((80, 120), dtype=np.uint8) * 255
        field_boundary = [(0.0, 0.0), (119.0, 0.0), (119.0, 79.0), (0.0, 79.0)]
        harvester_params = {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": 0.8,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "wheelbase_m": 1.8,
            "track_length_m": 1.6,
            "turn_radius_m": 1.0,
        }

    mask = State.mask_raw
    headland = np.zeros_like(mask)
    headland[34:57, 35:85] = 255

    monkeypatch.setattr(
        path_planner,
        "validate_footprints",
        lambda *_args, **_kwargs: {
            "valid": True,
            "issues": [],
            "validation_profile": "field_trial",
            "planned_target_geometric_coverage_pct": 100.0,
            "track_outside_field_pct": 0.0,
            "track_outside_support_pct": 0.0,
            "track_uncertain_overlap_pct": 0.0,
            "track_forbidden_overlap_pct": 0.0,
            "turn_outside_field_pct": 0.0,
            "turn_forbidden_overlap_area_m2": 0.0,
            "headland_turn_track_overlap_area_m2": 0.0,
            "turn_cost_m2": 0.0,
        },
    )
    _, _, diagnostics = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=Geo(), state=State(), min_turn_radius_m=1.0, turn_strategy="bow",
        config={
            "headland_turn_optimization_enabled": True,
            "headland_mask": headland,
            "planning_support_mask": mask,
            "uncertain_mask": np.zeros_like(mask),
            "forbidden_mask": np.zeros_like(mask),
            "validation_profile": "field_trial",
        },
        processed_mask=mask, return_diagnostics=True, global_main_angle=0.0,
    )

    optimization = diagnostics["turn_decisions"][0]["headland_turn_optimization"]
    assert optimization["optimization_applied"] is False
    assert any(
        "validation_metric_missing:track_core_overlap_pct" in item["hard_reasons"]
        for item in optimization["rejected_candidates"]
    )


def test_formal_validation_requires_explicit_valid_true():
    """DECISION-011: complete metrics without validity evidence fail closed."""
    from footprint_planner import assess_footprint_hard_constraints

    result = assess_footprint_hard_constraints({
        "issues": [],
        "track_core_overlap_pct": 0.0,
        "planned_target_geometric_coverage_pct": 100.0,
        "track_outside_field_pct": 0.0,
    })

    assert result["valid"] is False
    assert "validation_validity_metric_missing" in result["hard_reasons"]
