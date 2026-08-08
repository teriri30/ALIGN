import numpy as np
import pytest


def _valid_path_result(mode="footprint_optimized"):
    return {
        "is_valid": True,
        "validation": {
            "valid": True,
            "validation_profile": "machine_candidate",
            "field_boundary_present": True,
            "semantic_support_present": True,
            "uncertain_mask_present": True,
            "forbidden_mask_present": True,
            "footprint_model": "planar_rigid_footprint_v2",
            "footprint_reference_point": "track_contact_center",
            "footprint_resolution_m": 0.05,
            "machine_outside_field_pct": 0.0,
            "machine_forbidden_overlap_pct": 0.0,
            "validation_limits_pct": {
                "machine_outside_field": 0.0,
                "machine_forbidden_overlap": 0.0,
            },
        },
        "layout": {
            "work_line_mode": mode,
            "optimization_complete": mode == "footprint_optimized",
            "footprint_optimization_scope": (
                "centerline_anchored_lateral_search"
                if mode == "footprint_optimized"
                else "validation_only"
            ),
            "baseline_pass_count": 2,
            "optimized_pass_count": 2 if mode == "footprint_optimized" else 0,
            "ordered_band_ids": ["band-a", "band-b"],
            "line_optimization": [
                {"line_index": 0, "band_id": "band-a", "optimized": True},
                {"line_index": 1, "band_id": "band-b", "optimized": True},
            ],
            "formal_comparison": {
                "accepted": mode == "footprint_optimized",
                "degraded_metrics": [],
                "improved_metrics": (
                    [
                        "track_core_overlap_pct",
                        "track_outside_support_pct",
                        "track_uncertain_overlap_pct",
                    ]
                    if mode == "footprint_optimized"
                    else []
                ),
                "invalid_metrics": [],
            },
            "baseline_footprint_metrics": {
                "track_core_overlap_pct": 2.0,
                "planned_target_geometric_coverage_pct": 95.0,
                "track_outside_field_pct": 0.0,
                "track_outside_support_pct": 1.0,
                "track_uncertain_overlap_pct": 1.0,
                "track_forbidden_overlap_pct": 0.0,
            },
            "optimized_footprint_metrics": {
                "track_core_overlap_pct": 1.0,
                "planned_target_geometric_coverage_pct": 95.0,
                "track_outside_field_pct": 0.0,
                "track_outside_support_pct": 0.5,
                "track_uncertain_overlap_pct": 0.5,
                "track_forbidden_overlap_pct": 0.0,
            },
        },
        "planning_factors": {
            "harvester": {"track_width_m": 0.4},
            "harvester_param_sources": {"track_width_m": "user_measured"},
        },
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


@pytest.mark.parametrize("resolution", [None, float("nan"), float("inf"), "0.05"])
def test_controlled_trial_readiness_requires_finite_numeric_resolution(resolution):
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["validation"]["footprint_resolution_m"] = resolution
    result["planning_factors"] = {"harvester": {"track_width_m": 0.4}}

    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert "planning_resolution_missing_or_invalid" in readiness["controlled_trial_blockers"]


def test_machine_readiness_rejects_optimized_label_without_complete_optimization():
    """DECISION-011: readiness verifies evidence, not the mode label alone."""
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["layout"]["optimization_complete"] = False
    result["layout"]["optimized_pass_count"] = 1

    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert "footprint_optimization_incomplete" in readiness["controlled_trial_blockers"]


def test_machine_readiness_rejects_truthy_string_optimization_complete():
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["layout"]["optimization_complete"] = "true"

    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert "footprint_optimization_incomplete" in readiness["controlled_trial_blockers"]


def test_machine_readiness_rejects_wrong_footprint_optimization_scope():
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["layout"]["footprint_optimization_scope"] = "validation_only"

    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert "footprint_optimized_work_lines_not_used" in readiness["controlled_trial_blockers"]


@pytest.mark.parametrize("mutation", ["missing", "rejected", "invalid"])
def test_machine_readiness_requires_accepted_formal_footprint_comparison(mutation):
    """DECISION-011: readiness requires finite accepted before/after evidence."""
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    if mutation == "missing":
        result["layout"].pop("formal_comparison")
    elif mutation == "rejected":
        result["layout"]["formal_comparison"]["accepted"] = False
    else:
        result["layout"]["formal_comparison"]["invalid_metrics"] = [
            "track_core_overlap_pct"
        ]

    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert (
        "formal_footprint_comparison_not_accepted"
        in readiness["controlled_trial_blockers"]
    )


def test_machine_readiness_recomputes_formal_comparison_from_metric_summaries():
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["layout"]["optimized_footprint_metrics"][
        "track_core_overlap_pct"
    ] = 3.0

    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert (
        "formal_footprint_comparison_not_accepted"
        in readiness["controlled_trial_blockers"]
    )


@pytest.mark.parametrize(
    "presence_key,metric",
    [
        ("semantic_support_present", "track_outside_support_pct"),
        ("uncertain_mask_present", "track_uncertain_overlap_pct"),
        ("forbidden_mask_present", "track_forbidden_overlap_pct"),
    ],
)
def test_machine_readiness_requires_metrics_for_present_semantic_layers(
    presence_key, metric
):
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    assert result["validation"][presence_key] is True
    result["layout"]["baseline_footprint_metrics"].pop(metric)
    result["layout"]["optimized_footprint_metrics"].pop(metric)

    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert (
        "formal_footprint_comparison_not_accepted"
        in readiness["controlled_trial_blockers"]
    )


def test_machine_readiness_rejects_mismatched_band_identity_mapping():
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["layout"]["line_optimization"][1]["band_id"] = "band-a"

    readiness = assess_machine_readiness(
        result,
        {"forbidden_mask_confirmed": True},
    )

    assert readiness["controlled_trial_ready"] is False
    assert (
        "footprint_optimization_identity_unverified"
        in readiness["controlled_trial_blockers"]
    )


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


@pytest.mark.parametrize(
    "key",
    [
        "forbidden_mask_confirmed",
        "kinematic_model_validated",
        "terminal_adapter_validated",
        "manual_review_signed",
    ],
)
def test_machine_export_rejects_truthy_string_evidence(key):
    from machine_route_validator import assess_machine_readiness

    config = {
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
    }
    config[key] = "false"

    readiness = assess_machine_readiness(_valid_path_result(), config)

    assert readiness["eligible_for_machine_export"] is False


def test_machine_candidate_rejects_unacceptable_machine_envelope_risks():
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["validation"].update({
        "machine_outside_field_pct": 0.1,
        "machine_forbidden_overlap_pct": 0.1,
        "validation_limits_pct": {
            "machine_outside_field": 0.0,
            "machine_forbidden_overlap": 0.0,
        },
    })

    readiness = assess_machine_readiness(result, {"forbidden_mask_confirmed": True})

    assert readiness["controlled_trial_ready"] is False
    assert "machine_envelope_leaves_field" in readiness["controlled_trial_blockers"]
    assert "machine_envelope_enters_forbidden_region" in readiness["controlled_trial_blockers"]


def test_machine_candidate_fails_closed_when_machine_envelope_metrics_are_missing():
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["validation"].pop("machine_outside_field_pct")
    result["validation"].pop("machine_forbidden_overlap_pct")

    readiness = assess_machine_readiness(result, {"forbidden_mask_confirmed": True})

    assert readiness["controlled_trial_ready"] is False
    assert "machine_envelope_validation_missing" in readiness["controlled_trial_blockers"]


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), "invalid"])
def test_machine_candidate_readiness_rejects_non_finite_machine_envelope_metric(metric):
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    result["validation"]["machine_outside_field_pct"] = metric

    readiness = assess_machine_readiness(result, {"forbidden_mask_confirmed": True})

    assert readiness["controlled_trial_ready"] is False
    assert "machine_envelope_validation_missing" in readiness["controlled_trial_blockers"]


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
    assert layout["work_line_mode"] == "band_centerline_with_footprint_validation"
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


def test_cache_keeps_confirmed_forbidden_regions(tmp_path):
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


@pytest.mark.parametrize(
    "mutation,expected_blocker",
    [
        ("geo_rmse_bool", "external_geo_accuracy_not_verified"),
        ("gnss_bool", "vehicle_reference_geometry_incomplete"),
        ("machine_metric_bool", "machine_envelope_validation_missing"),
        ("machine_limit_bool", "machine_envelope_validation_missing"),
    ],
)
def test_machine_export_rejects_boolean_numeric_evidence(
    mutation, expected_blocker
):
    from machine_route_validator import assess_machine_readiness

    result = _valid_path_result()
    config = {
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
    }
    if mutation == "geo_rmse_bool":
        config["geo_accuracy_report"]["rmse_m"] = True
    elif mutation == "gnss_bool":
        config["gnss_offset_m"] = [False, 0.0, 1.8]
    elif mutation == "machine_metric_bool":
        result["validation"]["machine_outside_field_pct"] = False
    else:
        result["validation"]["validation_limits_pct"][
            "machine_outside_field"
        ] = False

    readiness = assess_machine_readiness(result, config)

    assert readiness["eligible_for_machine_export"] is False
    assert expected_blocker in readiness["blockers"]
