"""Pure helpers for fixed-strategy headland-turn validation.

This module does not inflate obstacles and does not run a global path search.
The planner supplies one endpoint-preserving turn geometry selected by the user.
This module builds a local semantic corridor and evaluates hard constraints; it
does not search offsets, rank alternatives, or change the requested geometry.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


TURN_VALIDATOR_VERSION = "fixed-headland-turn-validation-v1"


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _hard_reasons(candidate: Dict[str, Any]) -> List[str]:
    reasons = [str(item) for item in candidate.get("hard_reasons") or [] if str(item)]
    if candidate.get("kinematically_feasible") is False:
        if "kinematically_infeasible" not in reasons:
            reasons.append("kinematically_infeasible")
    for key, reason in (
        ("outside_field", "outside_field"),
        ("forbidden_overlap", "forbidden_overlap"),
        ("corridor_violation", "corridor_violation"),
    ):
        value = candidate.get(key)
        if value is True and reason not in reasons:
            reasons.append(reason)
        number = _finite(value)
        if number is not None and number > 0 and reason not in reasons:
            reasons.append(reason)
    return reasons


def validate_fixed_turn_candidate(
    candidate: Dict[str, Any],
    *,
    fallback_candidate: Optional[Dict[str, Any]] = None,
    fallback_reason: str = "",
) -> Dict[str, Any]:
    """Validate one fixed geometry and preserve the original on failure."""
    item = dict(candidate or {})
    reasons = _hard_reasons(item)
    item["hard_reasons"] = reasons
    item["kinematically_feasible"] = not bool(reasons)
    item["metrics"] = dict(item.get("metrics") or {})
    passed = not bool(reasons)
    rejected = [] if passed else [{
        "candidate_id": str(item.get("candidate_id", "")),
        "hard_reasons": reasons,
        "validation_issues": list(item.get("validation_issues") or []),
    }]
    selected = item if passed else fallback_candidate
    selected_reason = "" if passed else str(
        fallback_reason or "fixed_turn_validation_failed"
    )
    return {
        "selected": selected,
        "candidates": [item] if passed else [],
        "rejected_candidates": rejected,
        "diagnostics": {
            "validator_version": TURN_VALIDATOR_VERSION,
            "validation_passed": passed,
            "candidate_count": 1,
            "safe_candidate_count": int(passed),
            "rejected_candidate_count": len(rejected),
            "fallback_reason": selected_reason,
            "validation_policy": "fixed_geometry_hard_constraints",
        },
    }


def field_mask(shape: Tuple[int, int], field_boundary: Optional[Sequence[Sequence[float]]]) -> np.ndarray:
    result = np.zeros(tuple(shape[:2]), dtype=np.uint8)
    if field_boundary is None:
        return result
    points = np.asarray(field_boundary, dtype=np.float64)
    if points.ndim == 2 and points.shape[0] >= 3 and np.isfinite(points).all():
        cv2.fillPoly(result, [np.rint(points).astype(np.int32)], 255)
    return result


def _mask_or_zero(value: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.uint8)
    array = np.asarray(value)
    if array.shape[:2] != tuple(shape[:2]):
        return np.zeros(shape, dtype=np.uint8)
    return (array > 0).astype(np.uint8) * 255


def build_turn_corridor_mask(
    shape: Tuple[int, int],
    *,
    field_boundary: Optional[Sequence[Sequence[float]]],
    planning_support_mask: Optional[np.ndarray],
    headland_mask: Optional[np.ndarray],
    uncertain_residual_mask: Optional[np.ndarray],
    forbidden_mask: Optional[np.ndarray],
    endpoints: Iterable[Sequence[float]],
    corridor_radius_px: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a finite local corridor from semantic layers and endpoint geometry.

    DECISION-015: semantic evidence is clipped to the endpoint-local envelope
    before dilation, so a full planning-support mask cannot become a global
    search region.
    """
    shape = tuple(int(value) for value in shape[:2])
    boundary = field_mask(shape, field_boundary)
    support = _mask_or_zero(planning_support_mask, shape)
    headland = _mask_or_zero(headland_mask, shape)
    uncertain = _mask_or_zero(uncertain_residual_mask, shape)
    forbidden = _mask_or_zero(forbidden_mask, shape)
    endpoint_line = np.zeros(shape, dtype=np.uint8)
    points = [tuple(float(value) for value in point[:2]) for point in endpoints]
    if len(points) >= 2:
        cv2.polylines(endpoint_line, [np.rint(points).astype(np.int32)], False, 255, 1)
    elif points:
        x, y = (int(round(value)) for value in points[0])
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            endpoint_line[y, x] = 255
    radius = max(1, int(round(float(corridor_radius_px))))
    local_bbox = None
    if points:
        finite_points = np.asarray(points, dtype=np.float64)
        if finite_points.ndim != 2 or finite_points.shape[1] < 2 or not np.isfinite(finite_points[:, :2]).all():
            return np.zeros(shape, dtype=np.uint8), {
                "field_boundary_present": bool(np.count_nonzero(boundary)),
                "planning_support_present": bool(np.count_nonzero(support)),
                "headland_mask_present": bool(np.count_nonzero(headland)),
                "uncertain_residual_present": bool(np.count_nonzero(uncertain)),
                "forbidden_mask_present": bool(np.count_nonzero(forbidden)),
                "corridor_radius_px": radius,
                "corridor_pixel_count": 0,
                "corridor_local_bbox_px": None,
            }
        min_x = max(0, int(math.floor(float(np.min(finite_points[:, 0])) - radius)))
        max_x = min(shape[1] - 1, int(math.ceil(float(np.max(finite_points[:, 0])) + radius)))
        min_y = max(0, int(math.floor(float(np.min(finite_points[:, 1])) - radius)))
        max_y = min(shape[0] - 1, int(math.ceil(float(np.max(finite_points[:, 1])) + radius)))
        if min_x <= max_x and min_y <= max_y:
            local_bbox = [min_x, min_y, max_x, max_y]

    corridor = np.zeros(shape, dtype=np.uint8)
    if local_bbox is None:
        return corridor, {
            "field_boundary_present": bool(np.count_nonzero(boundary)),
            "planning_support_present": bool(np.count_nonzero(support)),
            "headland_mask_present": bool(np.count_nonzero(headland)),
            "uncertain_residual_present": bool(np.count_nonzero(uncertain)),
            "forbidden_mask_present": bool(np.count_nonzero(forbidden)),
            "corridor_radius_px": radius,
            "corridor_pixel_count": 0,
            "corridor_local_bbox_px": None,
        }

    semantic = cv2.bitwise_or(cv2.bitwise_or(support, headland), uncertain)
    min_x, min_y, max_x, max_y = local_bbox
    roi_min_x = max(0, min_x - radius)
    roi_max_x = min(shape[1] - 1, max_x + radius)
    roi_min_y = max(0, min_y - radius)
    roi_max_y = min(shape[0] - 1, max_y + radius)
    local_seed = np.zeros(
        (roi_max_y - roi_min_y + 1, roi_max_x - roi_min_x + 1),
        dtype=np.uint8,
    )
    seed_y0 = min_y - roi_min_y
    seed_y1 = max_y - roi_min_y + 1
    seed_x0 = min_x - roi_min_x
    seed_x1 = max_x - roi_min_x + 1
    local_seed[seed_y0:seed_y1, seed_x0:seed_x1] = cv2.bitwise_or(
        semantic[min_y:max_y + 1, min_x:max_x + 1],
        endpoint_line[min_y:max_y + 1, min_x:max_x + 1],
    )
    if radius <= 32:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        local_corridor = cv2.dilate(local_seed, kernel, iterations=1)
    elif np.any(local_seed):
        distance_source = np.where(local_seed > 0, 0, 255).astype(np.uint8)
        distance = cv2.distanceTransform(
            distance_source,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        local_corridor = (distance <= float(radius)).astype(np.uint8) * 255
    else:
        local_corridor = np.zeros_like(local_seed)
    corridor[roi_min_y:roi_max_y + 1, roi_min_x:roi_max_x + 1] = local_corridor
    if np.count_nonzero(boundary):
        corridor = cv2.bitwise_and(corridor, boundary)
    corridor = cv2.bitwise_and(corridor, cv2.bitwise_not(forbidden))
    return corridor, {
        "field_boundary_present": bool(np.count_nonzero(boundary)),
        "planning_support_present": bool(np.count_nonzero(support)),
        "headland_mask_present": bool(np.count_nonzero(headland)),
        "uncertain_residual_present": bool(np.count_nonzero(uncertain)),
        "forbidden_mask_present": bool(np.count_nonzero(forbidden)),
        "corridor_radius_px": radius,
        "corridor_pixel_count": int(np.count_nonzero(corridor)),
        "corridor_local_bbox_px": local_bbox,
    }


def polyline_inside_mask(path: Sequence[Sequence[Sequence[float]]], mask: np.ndarray) -> bool:
    """Check every rasterized candidate edge, not only its vertices."""
    candidate = np.zeros_like(np.asarray(mask, dtype=np.uint8))
    for segment in path or []:
        points = np.asarray(segment, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 1:
            return False
        rounded = np.rint(points).astype(np.int32)
        if points.shape[0] == 1:
            cv2.circle(candidate, tuple(rounded[0]), 0, 255, -1)
        else:
            cv2.polylines(candidate, [rounded], False, 255, 1)
    return bool(np.all(np.asarray(mask)[candidate > 0] > 0)) if np.any(candidate) else False


def has_non_degenerate_curve(path: Sequence[Sequence[Sequence[float]]]) -> bool:
    """Reject display-only endpoint connectors masquerading as executable turns."""
    for segment in path or []:
        points = [point for point in segment or []]
        if len(points) < 3:
            return False
        direct = math.hypot(
            float(points[-1][0]) - float(points[0][0]),
            float(points[-1][1]) - float(points[0][1]),
        )
        length = sum(
            math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
            for a, b in zip(points, points[1:])
        )
        if length <= max(direct * 1.001, 1e-6):
            return False
    return bool(path)
