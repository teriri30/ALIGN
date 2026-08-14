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


def test_connect_work_lines_validates_only_the_fixed_turn_geometry(monkeypatch):
    import path_planner

    mask = np.ones((80, 120), dtype=np.uint8) * 255
    state = _State(mask)
    headland = np.zeros_like(mask)
    headland[34:57, 35:85] = 255
    captured = []
    captured_segment_types = []

    def fake_validate(paths, *_args, **kwargs):
        captured.append(paths)
        captured_segment_types.append(list(kwargs["segment_types"]))
        points = [point for segment in paths for point in segment]
        lateral_peak = max(float(point[1]) for point in points)
        overlap = abs(lateral_peak - 49.0)
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
            "headland_turn_track_overlap_area_m2": overlap,
            "turn_cost_m2": overlap,
            "turn_body_overlap_area_m2": 0.0,
            "turn_uncertain_overlap_area_m2": 0.0,
            "turn_repeat_twice_or_more_area_m2": 0.0,
        }

    monkeypatch.setattr(path_planner, "validate_footprints", fake_validate)
    _, _, diagnostics = path_planner._connect_work_lines(
        [[(20.0, 25.0), (80.0, 25.0)], [(20.0, 55.0), (80.0, 55.0)]],
        geo=_MetricGeo(),
        state=state,
        min_turn_radius_m=1.0,
        turn_strategy="bow",
        config={
            "headland_mask": headland,
            "planning_support_mask": mask,
            "uncertain_mask": np.zeros_like(mask),
            "forbidden_mask": np.zeros_like(mask),
            "headland_damage_weight": 1.0,
        },
        processed_mask=mask,
        return_diagnostics=True,
        global_main_angle=0.0,
    )

    decision = diagnostics["turn_decisions"][0]
    turn_validation = decision["headland_turn_validation"]
    assert captured
    assert captured_segment_types == [["turn"]]
    assert len(captured) == 1
    assert len(captured[0]) == 1
    assert turn_validation["candidate_count"] == 1
    assert turn_validation["validation_passed"] is True
    assert decision["fallback_reason"] == ""
    candidate_geometries = {
        tuple(
            tuple((round(float(x), 6), round(float(y), 6)) for x, y in segment)
            for segment in candidate
        )
        for candidate in captured
    }
    assert len(candidate_geometries) == 1
    endpoints = {
        (
            tuple(float(value) for value in candidate[0][0]),
            tuple(float(value) for value in candidate[-1][-1]),
        )
        for candidate in captured
    }
    assert len(endpoints) == 1
# DECISION-015: integration preserves requested/candidate/effective turn semantics.
