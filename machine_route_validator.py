"""Fail-closed readiness assessment for future machine route exporters."""

import math
from typing import Dict, Optional

from config import HARVESTER_FORMAL_MEASURED_SOURCES
from harvester_geometry import FOOTPRINT_MODEL, FOOTPRINT_REFERENCE_POINT
from footprint_planner import compare_work_line_footprints


MACHINE_READINESS_SCHEMA = "reui.machine-readiness.v1"


def _verified_report(report: object) -> bool:
    return isinstance(report, dict) and report.get("verified") is True


def _finite_real(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _verified_optimization_identity(layout: Dict, pass_count: int) -> bool:
    band_ids = layout.get("ordered_band_ids")
    diagnostics = layout.get("line_optimization")
    if (
        not isinstance(band_ids, list)
        or not isinstance(diagnostics, list)
        or len(band_ids) != pass_count
        or len(diagnostics) != pass_count
        or any(
            band_id == earlier
            for index, band_id in enumerate(band_ids)
            for earlier in band_ids[:index]
        )
    ):
        return False
    for index, (band_id, item) in enumerate(zip(band_ids, diagnostics)):
        if not isinstance(item, dict):
            return False
        line_index = item.get("line_index")
        if (
            not isinstance(line_index, int)
            or isinstance(line_index, bool)
            or line_index != index
            or item.get("band_id") != band_id
            or item.get("optimized") is not True
        ):
            return False
    return True


def _accepted_footprint_comparison(layout: Dict, validation: Dict) -> bool:
    comparison = layout.get("formal_comparison")
    baseline = layout.get("baseline_footprint_metrics")
    optimized = layout.get("optimized_footprint_metrics")
    if not all(isinstance(value, dict) for value in (comparison, baseline, optimized)):
        return False
    if (
        comparison.get("accepted") is not True
        or comparison.get("degraded_metrics") != []
        or comparison.get("invalid_metrics") != []
        or not comparison.get("improved_metrics")
    ):
        return False
    recomputed = compare_work_line_footprints(baseline, optimized)
    if (
        recomputed.get("accepted") is not True
        or comparison.get("degraded_metrics")
        != recomputed.get("degraded_metrics")
        or comparison.get("improved_metrics")
        != recomputed.get("improved_metrics")
        or comparison.get("invalid_metrics")
        != recomputed.get("invalid_metrics")
    ):
        return False
    required_metrics = [
        "track_core_overlap_pct",
        "planned_target_geometric_coverage_pct",
        "track_outside_field_pct",
    ]
    for presence_key, metric in (
        ("semantic_support_present", "track_outside_support_pct"),
        ("uncertain_mask_present", "track_uncertain_overlap_pct"),
        ("forbidden_mask_present", "track_forbidden_overlap_pct"),
    ):
        if validation.get(presence_key) is True:
            required_metrics.append(metric)
    return all(
        _finite_real(summary.get(name))
        for summary in (baseline, optimized)
        for name in required_metrics
    )


def assess_machine_readiness(
    path_result: Dict,
    execution_config: Optional[Dict] = None,
) -> Dict:
    """Report whether a validated route may enter a machine-adapter pipeline.

    This function does not make current GIS exports machine executable. It
    only records missing evidence and remains fail-closed by default.
    """
    result = dict(path_result or {})
    cfg = dict(execution_config or {})
    validation = dict(result.get("validation") or {})
    layout = dict(result.get("layout") or {})
    turning = dict(result.get("turn_assessment") or {})
    planning_factors = dict(result.get("planning_factors") or {})
    controlled_trial_blockers = []

    harvester = dict(planning_factors.get("harvester") or {})
    raw_sources = dict(planning_factors.get("harvester_param_sources") or {})
    unverified_harvester_params = [
        key
        for key in harvester
        if raw_sources.get(key) not in HARVESTER_FORMAL_MEASURED_SOURCES
    ]
    parameter_source_status = {
        "required": list(harvester),
        "sources": {key: raw_sources.get(key, "missing") for key in harvester},
        "unverified": unverified_harvester_params,
        "formal_measured_sources": sorted(HARVESTER_FORMAL_MEASURED_SOURCES),
    }
    recovery_report = dict(layout.get("coverage_gap_recovery") or {})
    planning_recovered = bool(
        layout.get("planning_recovered") is True
        or recovery_report.get("planning_recovered") is True
        or recovery_report.get("recovered_band_ids")
    )

    if result.get("is_valid", validation.get("valid", False)) is not True:
        controlled_trial_blockers.append("path_validation_failed")
    if turning.get("hard_reasons"):
        controlled_trial_blockers.append("turn_hard_errors_present")
    # DECISION-011: a label alone is insufficient; readiness requires the
    # complete centerline-anchored optimization contract and matching counts.
    optimization_scope = layout.get("footprint_optimization_scope")
    if (
        layout.get("work_line_mode") != "footprint_optimized"
        or optimization_scope != "centerline_anchored_lateral_search"
    ):
        controlled_trial_blockers.append("footprint_optimized_work_lines_not_used")
    else:
        baseline_count = layout.get("baseline_pass_count")
        optimized_count = layout.get("optimized_pass_count")
        complete_counts = (
            isinstance(baseline_count, int)
            and not isinstance(baseline_count, bool)
            and baseline_count > 0
            and isinstance(optimized_count, int)
            and not isinstance(optimized_count, bool)
            and optimized_count == baseline_count
        )
        if layout.get("optimization_complete") is not True or not complete_counts:
            controlled_trial_blockers.append("footprint_optimization_incomplete")
        if complete_counts and not _verified_optimization_identity(
            layout, baseline_count
        ):
            controlled_trial_blockers.append(
                "footprint_optimization_identity_unverified"
            )
        if not _accepted_footprint_comparison(layout, validation):
            controlled_trial_blockers.append(
                "formal_footprint_comparison_not_accepted"
            )
    if validation.get("validation_profile") not in {"field_trial", "machine_candidate"}:
        controlled_trial_blockers.append("field_trial_validation_profile_not_used")
    profile = str(validation.get("validation_profile", "research"))
    if profile in {"field_trial", "machine_candidate"}:
        if unverified_harvester_params:
            controlled_trial_blockers.append("harvester_parameter_source_unverified")
        if planning_recovered:
            controlled_trial_blockers.append("planning_recovered_work_bands")
    if validation.get("field_boundary_present") is not True:
        controlled_trial_blockers.append("confirmed_field_boundary_missing")
    if validation.get("semantic_support_present") is not True:
        controlled_trial_blockers.append("semantic_support_missing")
    if (
        validation.get("footprint_model") != FOOTPRINT_MODEL
        or validation.get("footprint_reference_point") != FOOTPRINT_REFERENCE_POINT
    ):
        controlled_trial_blockers.append("footprint_contract_mismatch")
    if not (
        validation.get("forbidden_mask_present") is True
        and cfg.get("forbidden_mask_confirmed") is True
    ):
        controlled_trial_blockers.append("confirmed_forbidden_map_missing")
    resolution = validation.get("footprint_resolution_m")
    track_width = harvester.get("track_width_m")
    resolution_valid = (
        isinstance(resolution, (int, float))
        and not isinstance(resolution, bool)
        and math.isfinite(float(resolution))
        and float(resolution) > 0.0
    )
    track_width_valid = (
        isinstance(track_width, (int, float))
        and not isinstance(track_width, bool)
        and math.isfinite(float(track_width))
        and float(track_width) > 0.0
    )
    if not resolution_valid or not track_width_valid:
        controlled_trial_blockers.append("planning_resolution_missing_or_invalid")
    else:
        resolution_limit = min(0.10, float(track_width) / 4.0)
        if float(resolution) > resolution_limit:
            controlled_trial_blockers.append("planning_resolution_too_coarse")

    if validation.get("validation_profile") == "machine_candidate":
        machine_outside = validation.get("machine_outside_field_pct")
        machine_forbidden = validation.get("machine_forbidden_overlap_pct")
        limits = dict(validation.get("validation_limits_pct") or {})
        max_machine_outside = limits.get("machine_outside_field")
        max_machine_forbidden = limits.get("machine_forbidden_overlap")
        if not all(
            _finite_real(value)
            for value in (
                machine_outside,
                machine_forbidden,
                max_machine_outside,
                max_machine_forbidden,
            )
        ):
            controlled_trial_blockers.append("machine_envelope_validation_missing")
        else:
            if float(machine_outside) > float(max_machine_outside):
                controlled_trial_blockers.append("machine_envelope_leaves_field")
            if float(machine_forbidden) > float(max_machine_forbidden):
                controlled_trial_blockers.append("machine_envelope_enters_forbidden_region")

    controlled_trial_blockers = list(dict.fromkeys(controlled_trial_blockers))
    blockers = list(controlled_trial_blockers)
    if validation.get("validation_profile") != "machine_candidate":
        blockers.append("machine_validation_profile_not_used")

    geo_report = cfg.get("geo_accuracy_report")
    geo_rmse = geo_report.get("rmse_m") if isinstance(geo_report, dict) else None
    if not (
        _verified_report(geo_report)
        and _finite_real(geo_rmse)
        and geo_report.get("within_tolerance") is True
    ):
        blockers.append("external_geo_accuracy_not_verified")

    reference_point = str(cfg.get("vehicle_reference_point", "")).strip()
    gnss_offset = cfg.get("gnss_offset_m")
    if not reference_point or not (
        isinstance(gnss_offset, (list, tuple))
        and len(gnss_offset) == 3
        and all(_finite_real(value) for value in gnss_offset)
    ):
        blockers.append("vehicle_reference_geometry_incomplete")
    if cfg.get("kinematic_model_validated") is not True:
        blockers.append("full_vehicle_kinematics_not_validated")
    if not (
        str(cfg.get("terminal_adapter", "")).strip()
        and cfg.get("terminal_adapter_validated") is True
    ):
        blockers.append("terminal_adapter_not_validated")
    if not _verified_report(cfg.get("field_tracking_report")):
        blockers.append("field_tracking_validation_missing")
    if cfg.get("manual_review_signed") is not True:
        blockers.append("manual_review_not_signed")

    blockers = list(dict.fromkeys(blockers))
    terminal_protocol_validated = bool(
        cfg.get("terminal_adapter_validated") is True
        and str(cfg.get("terminal_adapter", "")).strip()
    )
    return {
        "schema": MACHINE_READINESS_SCHEMA,
        "controlled_trial_ready": not controlled_trial_blockers,
        "controlled_trial_blockers": controlled_trial_blockers,
        "eligible_for_machine_export": not blockers,
        "terminal_protocol_validated": terminal_protocol_validated,
        "current_route_classification": (
            "machine_export_candidate" if not blockers else "research_manual_review"
        ),
        "blockers": blockers,
        "manual_review_required": bool(blockers),
        "machine_executable": bool(not controlled_trial_blockers and not blockers),
        "harvester_parameter_source_status": parameter_source_status,
        "planning_recovered": planning_recovered,
    }
