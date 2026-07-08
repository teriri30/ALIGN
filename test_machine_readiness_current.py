import numpy as np


def _valid_path_result(mode="footprint_optimized"):
    return {
        "is_valid": True,
        "validation": {
            "valid": True,
            "validation_profile": "machine_candidate",
            "field_boundary_present": True,
            "semantic_support_present": True,
            "forbidden_mask_present": True,
            "footprint_model": "planar_rigid_footprint_v2",
            "footprint_reference_point": "track_contact_center",
        },
        "layout": {"work_line_mode": mode},
        "turn_assessment": {"hard_reasons": []},
    }


def test_machine_readiness_is_fail_closed_without_external_evidence():
    from machine_route_validator import assess_machine_readiness

    readiness = assess_machine_readiness(_valid_path_result())

    assert readiness["eligible_for_machine_export"] is False
    assert "external_geo_accuracy_not_verified" in readiness["blockers"]
    assert "full_vehicle_kinematics_not_validated" in readiness["blockers"]
    assert readiness["current_route_classification"] == "research_manual_review"


def test_controlled_trial_readiness_rejects_coarse_planning_resolution():
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["validation"]["footprint_resolution_m"] = 0.2
    result["planning_factors"] = {"harvester": {"track_width_m": 0.4}}
    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert "planning_resolution_too_coarse" in readiness["controlled_trial_blockers"]


def test_machine_readiness_accepts_only_complete_verified_evidence():
    from machine_route_validator import assess_machine_readiness

    readiness = assess_machine_readiness(
        _valid_path_result(),
        {
            "forbidden_mask_confirmed": True,
            "geo_accuracy_report": {
                "verified": True,
                "rmse_m": 0.03,
                "within_tolerance": True,
            },
            "vehicle_reference_point": "rear_axle_center",
            "gnss_offset_m": [0.0, 0.0, 1.8],
            "kinematic_model_validated": True,
            "terminal_adapter": "verified_test_adapter",
            "terminal_adapter_validated": True,
            "field_tracking_report": {"verified": True},
            "manual_review_signed": True,
        },
    )

    assert readiness["eligible_for_machine_export"] is True
    assert readiness["controlled_trial_ready"] is True
    assert readiness["blockers"] == []


def test_footprint_work_line_mode_requires_four_row_band_references(monkeypatch):
    import path_planner

    called = False

    def fake_generate(*_args, **_kwargs):
        nonlocal called
        called = True
        return [[(1.0, 2.0), (3.0, 4.0)]], {"generated_pass_count": 1}

    monkeypatch.setattr(path_planner, "generate_work_lines", fake_generate)
    class State:
        harvester_params = {
            "cutter_width_m": 2.0,
            "cutter_forward_offset_m": 1.2,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "wheelbase_m": 1.8,
            "track_length_m": 1.7,
            "turn_radius_m": 0.9,
        }

    lines, layout = path_planner._prepare_work_line_layout(
        [],
        np.zeros((20, 30), dtype=np.uint8),
        0.25,
        state=State(),
        config={"work_line_mode": "footprint_optimized"},
    )

    assert lines == []
    assert layout["work_line_mode"] == "footprint_optimized"
    assert layout["reason"] == "no_valid_four_row_centerlines"
    assert called is False


def test_forbidden_regions_are_rasterized_in_local_mask_coordinates():
    from footprint_planner import rasterize_forbidden_regions

    class State:
        mask_offset_x = 100
        mask_offset_y = 200
        forbidden_regions = [[(105, 205), (115, 205), (115, 215), (105, 215)]]

    mask = rasterize_forbidden_regions((30, 30), State())

    assert mask is not None
    assert mask[10, 10] == 255
    assert mask[0, 0] == 0


def test_project_cache_keeps_confirmed_forbidden_regions(tmp_path):
    from cache import _build_project_state_payload

    tif_path = tmp_path / "source.tif"
    tif_path.write_bytes(b"source")

    class State:
        source_sha256 = "test-source"
        forbidden_regions = [[(1, 1), (5, 1), (5, 5), (1, 5)]]
        forbidden_regions_confirmed = True

    payload = _build_project_state_payload(str(tif_path), State(), "test")

    assert payload["forbidden_regions"] == [[[1, 1], [5, 1], [5, 5], [1, 5]]]
    assert payload["forbidden_regions_confirmed"] is True
