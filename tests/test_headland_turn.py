import numpy as np
import pytest


class _MetricGeo:
    def is_ready(self):
        return True

    def meters_per_pixel(self, _x, _y):
        return 0.1


class _State:
    mask_offset_x = 0
    mask_offset_y = 0
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

    def __init__(self, mask):
        self.mask_raw = mask


def test_fixed_turn_validation_preserves_the_requested_geometry():
    from headland_turn_validation import validate_fixed_turn_candidate

    candidate = {
        "candidate_id": "bow:fixed",
        "path": [[(2.0, 2.0), (3.0, 4.0), (4.0, 5.0)]],
        "hard_reasons": [],
        "metrics": {"turn_body_overlap_area_m2": 0.3},
    }
    result = validate_fixed_turn_candidate(
        candidate,
        fallback_candidate={"candidate_id": "original"},
    )

    assert result["selected"]["candidate_id"] == "bow:fixed"
    assert result["diagnostics"]["validation_passed"] is True
    assert result["diagnostics"]["validation_policy"] == (
        "fixed_geometry_hard_constraints"
    )


def test_fixed_turn_validation_rejects_hard_constraint_failure():
    from headland_turn_validation import validate_fixed_turn_candidate

    result = validate_fixed_turn_candidate(
        {
            "candidate_id": "bow:fixed",
            "hard_reasons": ["forbidden_overlap"],
            "metrics": {"turn_body_overlap_area_m2": 0.0},
        },
        fallback_candidate={"candidate_id": "original"},
    )

    assert result["selected"]["candidate_id"] == "original"
    assert result["diagnostics"]["validation_passed"] is False


def test_fixed_turn_validation_keeps_fallback_when_evidence_is_missing():
    from headland_turn_validation import validate_fixed_turn_candidate

    fallback = {
        "candidate_id": "original",
        "path": [[(2.0, 2.0), (3.0, 2.0)]],
    }
    result = validate_fixed_turn_candidate(
        {"candidate_id": "fixed", "hard_reasons": ["headland_mask_missing"]},
        fallback_candidate=fallback,
        fallback_reason="headland_mask_missing",
    )

    assert result["selected"] is fallback
    assert result["diagnostics"]["fallback_reason"] == "headland_mask_missing"
    assert result["diagnostics"]["validation_passed"] is False


def test_fixed_turn_validation_rejects_straight_display_only_candidate():
    from headland_turn_validation import validate_fixed_turn_candidate

    result = validate_fixed_turn_candidate(
        {
            "candidate_id": "straight",
            "path": [[(0.0, 0.0), (5.0, 0.0)]],
            "kinematically_feasible": False,
            "hard_reasons": ["display_only_connector"],
            "metrics": {"headland_turn_track_overlap_area_m2": 0.0},
        },
        fallback_candidate={"candidate_id": "fallback"},
    )

    assert result["selected"]["candidate_id"] == "fallback"


def test_validate_footprints_reports_segment_and_repeat_access_metrics():
    from footprint_planner import validate_footprints

    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[35:45, 5:115] = 255
    headland = np.zeros_like(mask)
    headland[20:60, 55:75] = 255
    state = _State(mask)
    result = validate_footprints(
        [
            [(10.0, 40.0), (50.0, 40.0)],
            [(50.0, 40.0), (60.0, 30.0), (70.0, 40.0)],
            [(70.0, 40.0), (110.0, 40.0)],
        ],
        mask,
        geo=_MetricGeo(),
        state=state,
        harvester=state.harvester_params,
        config={
            "headland_mask": headland,
            "planning_support_mask": np.ones_like(mask) * 255,
            "uncertain_mask": np.zeros_like(mask),
            "forbidden_mask": np.zeros_like(mask),
            "footprint_pose_step_m": 0.1,
        },
        segment_types=["work", "turn", "work"],
    )

    assert result["segment_metrics"]
    assert result["turn_segment_metrics"][0]["segment_type"] == "turn"
    assert result["track_rolling_visit_once_area_m2"] >= 0.0
    assert result["track_rolling_visit_twice_or_more_area_m2"] >= 0.0
    assert result["track_rolling_visit_once_area_m2"] + result["track_rolling_visit_twice_or_more_area_m2"] == pytest.approx(
        result["track_swept_area_m2"], rel=0.05
    )
    assert result["turn_metrics_attribution_rule"] == "exclusive_first_segment"
    assert result["headland_turn_track_overlap_area_m2"] >= 0.0
    assert result["turn_cost_m2"] >= 0.0


def test_headland_turn_config_requires_only_fixed_validation_inputs():
    from config import validate_config

    path_config = validate_config({
        "path_planning": {"headland_damage_weight": 1.0}
    })["path_planning"]
    assert "headland_turn_optimization_enabled" not in path_config
    assert path_config.get("headland_damage_weight", 1.0) == pytest.approx(1.0)

def test_turn_corridor_is_bounded_by_endpoint_local_envelope():
    """DECISION-015: full support evidence cannot make the corridor global."""
    from headland_turn_validation import build_turn_corridor_mask

    shape = (100, 100)
    support = np.ones(shape, dtype=np.uint8) * 255
    corridor, diagnostics = build_turn_corridor_mask(
        shape,
        field_boundary=[(0, 0), (99, 0), (99, 99), (0, 99)],
        planning_support_mask=support,
        headland_mask=np.zeros(shape, dtype=np.uint8),
        uncertain_residual_mask=np.zeros(shape, dtype=np.uint8),
        forbidden_mask=np.zeros(shape, dtype=np.uint8),
        endpoints=[(20.0, 40.0), (30.0, 60.0)],
        corridor_radius_px=5.0,
    )

    assert diagnostics["corridor_local_bbox_px"] == [15, 35, 35, 65]
    assert corridor[50, 50] == 0
    assert corridor[45, 25] > 0


def test_turn_corridor_dilates_only_the_endpoint_local_roi(monkeypatch):
    from headland_turn_validation import build_turn_corridor_mask
    import headland_turn_validation

    shape = (1200, 1600)
    support = np.ones(shape, dtype=np.uint8) * 255
    original_dilate = headland_turn_validation.cv2.dilate
    dilated_shapes = []

    def record_dilate(source, *args, **kwargs):
        dilated_shapes.append(tuple(source.shape[:2]))
        return original_dilate(source, *args, **kwargs)

    monkeypatch.setattr(headland_turn_validation.cv2, "dilate", record_dilate)
    corridor, _ = build_turn_corridor_mask(
        shape,
        field_boundary=[(0, 0), (1599, 0), (1599, 1199), (0, 1199)],
        planning_support_mask=support,
        headland_mask=np.zeros(shape, dtype=np.uint8),
        uncertain_residual_mask=np.zeros(shape, dtype=np.uint8),
        forbidden_mask=np.zeros(shape, dtype=np.uint8),
        endpoints=[(700.0, 500.0), (720.0, 540.0)],
        corridor_radius_px=12.0,
    )

    assert corridor.shape == shape
    assert dilated_shapes
    assert dilated_shapes[0][0] < 100
    assert dilated_shapes[0][1] < 100


def test_turn_corridor_avoids_large_morphology_kernels(monkeypatch):
    from headland_turn_validation import build_turn_corridor_mask
    import headland_turn_validation

    shape = (900, 1200)
    support = np.zeros(shape, dtype=np.uint8)
    support[430:470, 570:630] = 255

    def reject_large_dilate(*_args, **_kwargs):
        raise AssertionError("large-radius corridor must not use cv2.dilate")

    monkeypatch.setattr(headland_turn_validation.cv2, "dilate", reject_large_dilate)
    corridor, diagnostics = build_turn_corridor_mask(
        shape,
        field_boundary=[(0, 0), (1199, 0), (1199, 899), (0, 899)],
        planning_support_mask=support,
        headland_mask=np.zeros(shape, dtype=np.uint8),
        uncertain_residual_mask=np.zeros(shape, dtype=np.uint8),
        forbidden_mask=np.zeros(shape, dtype=np.uint8),
        endpoints=[(590.0, 440.0), (610.0, 460.0)],
        corridor_radius_px=150.0,
    )

    assert diagnostics["corridor_radius_px"] == 150
    assert corridor[450, 600] > 0
    assert corridor[300, 600] > 0
    assert corridor[100, 600] == 0


def test_missing_field_boundary_is_reported_by_fixed_turn_validation():
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
        field_boundary = None
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
    before, before_types = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=Geo(), state=State(), min_turn_radius_m=1.0, turn_strategy="bow",
        config={}, processed_mask=mask,
    )
    after, after_types, diagnostics = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=Geo(), state=State(), min_turn_radius_m=1.0, turn_strategy="bow",
        config={
            "headland_mask": headland,
            "planning_support_mask": mask,
            "uncertain_mask": np.zeros_like(mask),
            "forbidden_mask": np.zeros_like(mask),
        },
        processed_mask=mask, return_diagnostics=True,
    )

    turn_validation = diagnostics["turn_decisions"][0]["headland_turn_validation"]
    assert turn_validation["validation_passed"] is False
    assert turn_validation["fallback_reason"] == "field_boundary_missing"
    assert after == before
    assert after_types == before_types
def test_segment_attribution_sums_to_full_path_agronomic_damage_cost():
    """DECISION-015: exclusive segment costs reconcile with the full-path cost."""
    from footprint_planner import validate_footprints

    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[35:45, 5:115] = 255
    headland = np.zeros_like(mask)
    headland[20:60, 55:75] = 255
    state = _State(mask)
    result = validate_footprints(
        [[(10.0, 40.0), (50.0, 40.0)], [(50.0, 40.0), (60.0, 30.0), (70.0, 40.0)], [(70.0, 40.0), (110.0, 40.0)]],
        mask, geo=_MetricGeo(), state=state, harvester=state.harvester_params,
        config={"headland_mask": headland, "planning_support_mask": np.ones_like(mask) * 255, "footprint_pose_step_m": 0.1},
        segment_types=["work", "turn", "work"],
    )

    assert sum(item["agronomic_damage_cost_m2"] for item in result["segment_metrics"]) == pytest.approx(
        result["agronomic_damage_cost_m2"], rel=0.05
    )
def test_real_footprint_metrics_are_used_for_fixed_turn_candidate(monkeypatch):
    import path_planner

    mask = np.ones((80, 120), dtype=np.uint8) * 255
    state = _State(mask)
    headland = np.zeros_like(mask)
    headland[34:57, 35:85] = 255

    def valid_footprint_metrics(*_args, **_kwargs):
        return {
            "valid": True,
            "issues": [],
            "validation_profile": "field_trial",
            "track_core_overlap_pct": 0.0,
            "planned_target_geometric_coverage_pct": 100.0,
            "track_outside_field_pct": 0.0,
            "track_outside_support_pct": 0.0,
            "track_uncertain_overlap_pct": 0.0,
            "track_forbidden_overlap_pct": 0.0,
            "turn_outside_field_pct": 0.0,
            "turn_forbidden_overlap_area_m2": 0.0,
            "headland_turn_track_overlap_area_m2": 0.2,
            "turn_cost_m2": 0.2,
            "turn_body_overlap_area_m2": 0.0,
            "turn_uncertain_overlap_area_m2": 0.0,
            "turn_repeat_twice_or_more_area_m2": 0.0,
        }

    monkeypatch.setattr(path_planner, "validate_footprints", valid_footprint_metrics)
    _, _, diagnostics = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=_MetricGeo(), state=state, min_turn_radius_m=1.0, turn_strategy="bow",
        config={
            "headland_mask": headland,
            "planning_support_mask": mask,
            "uncertain_mask": np.zeros_like(mask),
            "forbidden_mask": np.zeros_like(mask),
            "headland_damage_weight": 1.0,
        },
        processed_mask=mask, return_diagnostics=True, global_main_angle=0.0,
    )

    turn_validation = diagnostics["turn_decisions"][0]["headland_turn_validation"]
    assert turn_validation["candidate_count"] == 1
    assert turn_validation["safe_candidate_count"] >= 1
    assert turn_validation["validation_passed"] is True
    assert all(
        "footprint_validation_unavailable" not in reason
        for item in turn_validation["rejected_candidates"]
        for reason in item["hard_reasons"]
    )
