"""
路径规划器 — 中心线串联 / 多策略转弯 / 地理坐标转换 / 路径验证 / 输出

转弯策略 (文献依据):
  - bow:         Bow turning — R < ω/2 时的弓形转弯
  - semicircle:  Semicircular turning — R ≈ ω/2 时的半圆转弯
  - pear:        Pear-shaped turning — R > ω/2 时的三圆相切转弯
  - fishtail:    Fishtail turning — 田头受限时的倒车折返
  - contour:     Contour/spiral ordering — 外圈向内访问现有作业中心线
"""
import math, csv, json, os, tempfile, hashlib
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from shapely.geometry import LineString, Point, Polygon
from config import (
    Config,
    AppLogger,
    DEFAULT_HARVESTER_PARAMS,
    HARVESTER_FORMAL_MEASURED_SOURCES,
    HARVESTER_PARAM_SOURCE_LABELS,
)
from footprint_planner import (
    assess_footprint_hard_constraints,
    compare_work_line_footprints,
    generate_work_lines,
    summarize_work_line_footprint,
    validate_footprints,
)
from row_geometry import field_polygon_local, meters_per_pixel, smooth_1d
from headland_turn_validation import (
    build_turn_corridor_mask,
    has_non_degenerate_curve,
    polyline_inside_mask,
    validate_fixed_turn_candidate,
)


PATH_PLANNING_VERSION = "footprint-path-v31"


def _adjacent_snake_order_work_lines(
    work_lines: List[List[Tuple[float, float]]],
    entry_point: Optional[Tuple[float, float]] = None,
    main_angle: Optional[float] = None,
) -> Tuple[List[List[Tuple[float, float]]], Dict]:
    """Order existing passes by adjacent normal position for the UI route.

    The desktop workflow is an implementation-oriented planner: after the
    required start row is selected, it should visit neighbouring crop rows in
    sequence.  A half-split contour order intentionally jumps from one side
    of the field to the other, which is useful for a separate contour study
    but is surprising as the default route shown to an operator.  This helper
    only changes the visit order of already generated work lines; it never
    invents or laterally moves a line.
    """
    diagnostics = {
        "ordering": "adjacent_snake",
        "algorithm_version": "adjacent-snake-v1",
        "ordering_basis": "parallel_line_normal_axis",
        "normal_order_reversed": False,
        "field_boundary_required": False,
        "fallback_reason": None,
    }
    if not work_lines:
        diagnostics["fallback_reason"] = "no_work_lines"
        return [], diagnostics

    angle = None
    if main_angle is not None and math.isfinite(float(main_angle)):
        angle = float(main_angle)
    else:
        weighted_cos = 0.0
        weighted_sin = 0.0
        for line in work_lines:
            points = np.asarray(line, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 2:
                continue
            dx = float(points[-1][0] - points[0][0])
            dy = float(points[-1][1] - points[0][1])
            weight = math.hypot(dx, dy)
            if weight <= 1e-9:
                continue
            line_angle = math.atan2(dy, dx)
            weighted_cos += weight * math.cos(2.0 * line_angle)
            weighted_sin += weight * math.sin(2.0 * line_angle)
        if weighted_cos or weighted_sin:
            angle = 0.5 * math.atan2(weighted_sin, weighted_cos)
    if angle is None:
        diagnostics["fallback_reason"] = "direction_unavailable"
        return [list(line) for line in work_lines], diagnostics

    normal = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    records = []
    for index, line in enumerate(work_lines):
        points = np.asarray(line, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] < 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            diagnostics["fallback_reason"] = "invalid_work_line"
            return [list(item) for item in work_lines], diagnostics
        midpoint = points.mean(axis=0)
        records.append({
            "index": index,
            "line": list(line),
            "pos": float(midpoint @ normal),
        })
    records.sort(key=lambda item: (round(item["pos"], 6), item["index"]))

    def endpoint_distance(record):
        if entry_point is None:
            return 0.0
        return min(
            math.hypot(
                float(point[0]) - float(entry_point[0]),
                float(point[1]) - float(entry_point[1]),
            )
            for point in (record["line"][0], record["line"][-1])
        )

    if entry_point is not None and len(records) >= 2:
        low_distance = endpoint_distance(records[0])
        high_distance = endpoint_distance(records[-1])
        if high_distance < low_distance:
            records.reverse()
            diagnostics["normal_order_reversed"] = True
        diagnostics.update({
            "entry_priority_applied": True,
            "entry_selected_line_distance_px": float(
                endpoint_distance(records[0])
            ),
            "entry_selected_endpoint_local": [
                float(value) for value in min(
                    (records[0]["line"][0], records[0]["line"][-1]),
                    key=lambda point: math.hypot(
                        float(point[0]) - float(entry_point[0]),
                        float(point[1]) - float(entry_point[1]),
                    ),
                )
            ],
        })
    diagnostics.update({
        "ordered_original_indices": [item["index"] for item in records],
        "normal_positions": [float(item["pos"]) for item in records],
        "sorted_normal_positions": sorted(float(item["pos"]) for item in records),
        "split_index": None,
        "low_block_original_indices": [],
        "high_block_original_indices": [],
        "ordering_variant": "adjacent_normal_snake",
        "pair_count": (len(records) + 1) // 2,
        "adjacent_transition_count": max(0, len(records) - 1),
        "minimum_normal_index_gap": 1 if len(records) > 1 else 0,
        "main_angle_rad": float(angle),
    })
    diagnostics["corresponding_pairs"] = [
        {
            "pair_index": pair_index,
            "original_indices": [item["index"] for item in records[index:index + 2]],
            "normal_positions": [float(item["pos"]) for item in records[index:index + 2]],
        }
        for pair_index, index in enumerate(range(0, len(records), 2))
    ]
    return [list(item["line"]) for item in records], diagnostics


def _contour_order_work_lines(
    work_lines: List[List[Tuple[float, float]]],
    field_polygon: Optional[np.ndarray],
    main_angle: Optional[float] = None,
) -> Tuple[List[List[Tuple[float, float]]], Dict]:
    """DECISION-016: order parallel work lines by half-split interleaving.

    This is the *order* part of a contour/spiral baseline. It does not invent
    a closed boundary contour: every returned pass remains an existing
    crop-supported work line. The normal-axis records are split into two
    contiguous blocks and interleaved (low-half[0], high-half[0], ...). This
    keeps each turn between corresponding, well-separated rows instead of
    forcing the final transition between the two central rows.
    """
    diagnostics = {
        "ordering": "contour_half_split_interleaved",
        "algorithm_version": "contour-order-v5",
        "ordering_basis": "parallel_line_half_split_interleave",
        "normal_order_reversed": False,
        "field_boundary_required": False,
        "fallback_reason": "",
    }
    if not work_lines:
        diagnostics["fallback_reason"] = "no_work_lines"
        return [], diagnostics

    angle = None
    if main_angle is not None and math.isfinite(float(main_angle)):
        angle = float(main_angle)
    else:
        weighted_cos = 0.0
        weighted_sin = 0.0
        for line in work_lines:
            points = np.asarray(line, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 2:
                continue
            dx = float(points[-1][0] - points[0][0])
            dy = float(points[-1][1] - points[0][1])
            weight = math.hypot(dx, dy)
            if weight <= 1e-9:
                continue
            line_angle = math.atan2(dy, dx)
            weighted_cos += weight * math.cos(2.0 * line_angle)
            weighted_sin += weight * math.sin(2.0 * line_angle)
        if weighted_cos or weighted_sin:
            angle = 0.5 * math.atan2(weighted_sin, weighted_cos)
    if angle is None:
        diagnostics["fallback_reason"] = "direction_unavailable"
        return [list(line) for line in work_lines], diagnostics

    nx, ny = -math.sin(angle), math.cos(angle)
    records = []
    for index, line in enumerate(work_lines):
        points = np.asarray(line, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] < 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            diagnostics["fallback_reason"] = "invalid_work_line"
            return [list(item) for item in work_lines], diagnostics
        midpoint = points.mean(axis=0)
        pos = float(midpoint[0] * nx + midpoint[1] * ny)
        records.append({"index": index, "line": list(line), "pos": pos})
    records.sort(key=lambda item: (round(item["pos"], 6), item["index"]))

    # Split the normal-axis sequence into two blocks and interleave them. For
    # n=8 this is 0,4,1,5,2,6,3,7; for n=9 it is 0,5,1,6,2,7,3,8,4.
    # Four lines are the only useful small case where the simple split would
    # still leave a central adjacent transition; use the parity order there.
    split_index = (len(records) + 1) // 2
    low_block = records[:split_index]
    high_block = records[split_index:]
    if len(records) == 4:
        ordered_records = [records[index] for index in (1, 3, 0, 2)]
        ordering_variant = "parity_avoid_adjacent_centre"
    else:
        ordered_records = []
        for index in range(max(len(low_block), len(high_block))):
            if index < len(low_block):
                ordered_records.append(low_block[index])
            if index < len(high_block):
                ordered_records.append(high_block[index])
        ordering_variant = "half_split_interleave"

    pairs = []
    for index in range(0, len(ordered_records), 2):
        pair = ordered_records[index:index + 2]
        pairs.append(pair)
    ordered_positions = [item["pos"] for item in ordered_records]
    diagnostics["split_index"] = int(split_index)
    diagnostics["low_block_original_indices"] = [item["index"] for item in low_block]
    diagnostics["high_block_original_indices"] = [item["index"] for item in high_block]
    diagnostics["ordering_variant"] = ordering_variant
    diagnostics["ordered_original_indices"] = [item["index"] for item in ordered_records]
    diagnostics["normal_positions"] = ordered_positions
    diagnostics["sorted_normal_positions"] = [item["pos"] for item in records]
    diagnostics["pair_count"] = len(pairs)
    normal_rank = {
        item["index"]: rank for rank, item in enumerate(records)
    }
    normal_index_gaps = [
        abs(normal_rank[right["index"]] - normal_rank[left["index"]])
        for left, right in zip(ordered_records, ordered_records[1:])
    ]
    diagnostics["adjacent_transition_count"] = sum(
        gap == 1 for gap in normal_index_gaps
    )
    diagnostics["minimum_normal_index_gap"] = min(normal_index_gaps, default=0)
    diagnostics["corresponding_pairs"] = [
        {
            "pair_index": pair_index,
            "original_indices": [item["index"] for item in pair],
            "normal_positions": [float(item["pos"]) for item in pair],
        }
        for pair_index, pair in enumerate(pairs)
    ]
    diagnostics["main_angle_rad"] = float(angle)
    diagnostics["fallback_reason"] = None
    return [item["line"] for item in ordered_records], diagnostics


def _refresh_contour_order_diagnostics(
    diagnostics: Dict,
    ordered_indices: List[int],
    normal_positions: List[Optional[float]],
) -> None:
    """Recompute DECISION-016 invariants after every ordering mutation.

    The initial half-split sequence is only an intermediate result.  Entry
    rotation and cross-region handoff selection may change it, so reporting
    the pre-mutation adjacency count would make an unsafe final order look
    valid.
    """
    diagnostics["ordered_original_indices"] = list(ordered_indices)
    diagnostics["normal_positions"] = list(normal_positions)
    diagnostics["corresponding_pairs"] = [
        {
            "pair_index": pair_index,
            "original_indices": ordered_indices[index:index + 2],
            "normal_positions": normal_positions[index:index + 2],
        }
        for pair_index, index in enumerate(range(0, len(ordered_indices), 2))
    ]
    diagnostics["pair_count"] = (len(ordered_indices) + 1) // 2
    if (
        len(normal_positions) != len(ordered_indices)
        or any(value is None for value in normal_positions)
    ):
        ranks = {value: rank for rank, value in enumerate(sorted(ordered_indices))}
        gaps = [
            abs(ranks[right] - ranks[left])
            for left, right in zip(ordered_indices, ordered_indices[1:])
        ]
    else:
        ranked = sorted(
            range(len(normal_positions)),
            key=lambda index: (float(normal_positions[index]), ordered_indices[index]),
        )
        rank_by_original_index = {
            ordered_indices[position_index]: rank
            for rank, position_index in enumerate(ranked)
        }
        gaps = [
            abs(rank_by_original_index[right] - rank_by_original_index[left])
            for left, right in zip(ordered_indices, ordered_indices[1:])
        ]
    diagnostics["adjacent_transition_count"] = int(sum(gap == 1 for gap in gaps))
    diagnostics["minimum_normal_index_gap"] = int(min(gaps, default=0))
    diagnostics["final_order_validated"] = True


def _select_contour_entry_rotation(
    ordered_lines: List[List[Tuple[float, float]]],
    diagnostics: Dict,
    entry_point: Optional[Tuple[float, float]],
    policy: str = "nearest",
    terminal_original_index: Optional[int] = None,
    terminal_endpoint: Optional[Tuple[float, float]] = None,
    main_angle: Optional[float] = None,
) -> List[List[Tuple[float, float]]]:
    """Choose the contour entry while preserving an optional terminal line.

    The operator's confirmed entry is the primary ordering objective.  Normal
    adjacency is only a tie-break between equally close entry candidates.  A
    uniquely short, blocked line may be reserved as the final line; it is
    removed from the rotation cycle and appended afterwards so that reserving
    it does not silently reduce the possible first lines to one or two.
    """
    lines = [list(line) for line in ordered_lines]
    indices = list(diagnostics.get("ordered_original_indices") or [])
    positions = list(diagnostics.get("normal_positions") or [])
    if len(indices) != len(lines):
        indices = list(range(len(lines)))
    if len(positions) != len(lines):
        positions = [None] * len(lines)
    if not lines:
        _refresh_contour_order_diagnostics(diagnostics, indices, positions)
        return lines

    records = list(zip(lines, indices, positions))
    terminal_record = None
    if terminal_original_index is not None:
        terminal_record = next(
            (item for item in records if item[1] == terminal_original_index),
            None,
        )
    rotatable_records = [
        item for item in records if item is not terminal_record
    ]
    variants = []
    seen = set()
    for reversed_order, base in (
        (False, rotatable_records),
        (True, list(reversed(rotatable_records))),
    ):
        offsets = range(len(base)) if base else range(1)
        for offset in offsets:
            candidate = base[offset:] + base[:offset]
            if terminal_record is not None:
                candidate = candidate + [terminal_record]
            key = tuple(item[1] for item in candidate)
            if key in seen:
                continue
            seen.add(key)
            candidate_diagnostics = {}
            _refresh_contour_order_diagnostics(
                candidate_diagnostics,
                [item[1] for item in candidate],
                [item[2] for item in candidate],
            )
            variants.append({
                "records": candidate,
                "adjacent": int(
                    candidate_diagnostics["adjacent_transition_count"]
                ),
                "offset": int(offset),
                "reversed": bool(reversed_order),
            })

    minimum_adjacent = min(item["adjacent"] for item in variants)

    def candidate_start_point(item):
        line = item["records"][0][0]
        if entry_point is None:
            return (float(line[0][0]), float(line[0][1]))
        return min(
            (line[0], line[-1]),
            key=lambda point: math.hypot(
                float(point[0]) - float(entry_point[0]),
                float(point[1]) - float(entry_point[1]),
            ),
        )

    def endpoint_distance(item):
        if entry_point is None:
            return 0.0
        point = candidate_start_point(item)
        return math.hypot(
            float(point[0]) - float(entry_point[0]),
            float(point[1]) - float(entry_point[1]),
        )

    normalized_policy = str(policy or "nearest").strip().lower()
    if normalized_policy not in {"nearest", "farthest"}:
        normalized_policy = "nearest"
    if entry_point is None:
        selected = min(
            variants,
            key=lambda item: (
                item["adjacent"],
                item["reversed"],
                item["offset"],
            ),
        )
    elif normalized_policy == "farthest":
        selected = min(
            variants,
            key=lambda item: (
                -endpoint_distance(item),
                item["adjacent"],
                item["reversed"],
                item["offset"],
            ),
        )
    else:
        selected = min(
            variants,
            key=lambda item: (
                endpoint_distance(item),
                item["adjacent"],
                item["reversed"],
                item["offset"],
            ),
        )

    selected_records = selected["records"]
    selected_indices = [item[1] for item in selected_records]
    selected_positions = [item[2] for item in selected_records]
    _refresh_contour_order_diagnostics(
        diagnostics, selected_indices, selected_positions
    )
    diagnostics.update({
        "algorithm_version": "contour-order-v8",
        "entry_selection_policy": normalized_policy,
        "entry_rotation_offset": int(selected["offset"]),
        "entry_rotation_reversed": bool(selected["reversed"]),
        "normal_order_reversed": normalized_policy == "farthest",
        "entry_priority_applied": bool(entry_point is not None),
        "entry_eligible_candidate_count": int(len(variants)),
        "minimum_possible_adjacent_transition_count": int(minimum_adjacent),
        "selected_adjacent_transition_count": int(selected["adjacent"]),
        "entry_priority_overrode_minimum_adjacency": bool(
            entry_point is not None
            and selected["adjacent"] > minimum_adjacent
        ),
        # DECISION-016/017: the terminal short line may remain last, but its
        # blocked endpoint is a local final-transition constraint.  It must
        # never reverse the whole route away from the confirmed entry.
        "entry_orientation_constrained_by_terminal": False,
        "terminal_original_index": terminal_original_index,
        "terminal_constraint_applied": bool(terminal_record is not None),
    })
    if entry_point is not None:
        diagnostics["entry_selected_line_distance_px"] = float(
            endpoint_distance(selected)
        )
        diagnostics["entry_selected_endpoint_local"] = [
            float(value) for value in candidate_start_point(selected)
        ]
    return [list(item[0]) for item in selected_records]


def _build_contour_connector_anchors(
    oriented_lines: List[List[Tuple[float, float]]],
    main_angle: float,
    field_polygon: Optional[np.ndarray],
    mpp: float,
    config: Optional[Dict] = None,
) -> Tuple[List[Dict[str, Tuple[float, float]]], Dict]:
    """DECISION-016: build the two shared headland walls for nested loops.

    Work lines are already ordered by the contour half-split interleave and
    oriented alternately. Every endpoint on the same physical field end is
    projected to one common axial wall. The following bow turns therefore
    place their transverse tangent on the same headland line; their lateral
    spans are retained in the actual interleaved order and need not shrink
    monotonically.

    The transverse span is intentionally *not* capped at a few metres: the
    first loop crosses from the left outer pass to the right outer pass.  Only
    the along-row endpoint compensation is bounded.  This helper does not
    mutate the supplied geometry.  The caller may merge a boundary-checked,
    cohort-supported axial correction into the work line under DECISION-010.
    """
    cfg = dict(config or {})
    diagnostics = {
        "enabled": True,
        "algorithm_version": "contour-connectors-v2",
        "connector_count": 0,
        "shared_headland_lines": False,
        "center_single_finish": bool(len(oriented_lines) % 2),
        "fallback_reason": None,
        "boundary_supported": field_polygon is not None,
    }
    if len(oriented_lines) < 2:
        diagnostics["fallback_reason"] = "fewer_than_two_lines"
        return [], diagnostics
    if field_polygon is None:
        diagnostics["fallback_reason"] = "missing_field_boundary"
        return [], diagnostics
    direction = np.asarray(
        (math.cos(float(main_angle)), math.sin(float(main_angle))),
        dtype=np.float64,
    )
    try:
        max_extension_m = float(
            cfg.get(
                "contour_max_endpoint_extension_m",
                cfg.get("endpoint_extension_max_m", 4.0),
            )
        )
    except (TypeError, ValueError):
        max_extension_m = 4.0
    if not math.isfinite(max_extension_m) or max_extension_m < 0.0:
        diagnostics["fallback_reason"] = "invalid_endpoint_extension_limit"
        return [], diagnostics

    endpoint_records = []
    low_projections = []
    high_projections = []
    for line_index, line in enumerate(oriented_lines):
        if len(line) < 2:
            diagnostics["fallback_reason"] = "invalid_work_line"
            return [], diagnostics
        start = np.asarray(line[0], dtype=np.float64)
        end = np.asarray(line[-1], dtype=np.float64)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            diagnostics["fallback_reason"] = "invalid_work_line"
            return [], diagnostics
        start_projection = float(start @ direction)
        end_projection = float(end @ direction)
        if abs(start_projection - end_projection) <= 1e-9:
            diagnostics["fallback_reason"] = "work_line_axial_extent_degenerate"
            return [], diagnostics
        start_side = "low" if start_projection < end_projection else "high"
        end_side = "high" if start_side == "low" else "low"
        low_projections.append(min(start_projection, end_projection))
        high_projections.append(max(start_projection, end_projection))
        endpoint_records.append({
            "line_index": line_index,
            "start": start,
            "end": end,
            "start_projection": start_projection,
            "end_projection": end_projection,
            "start_side": start_side,
            "end_side": end_side,
        })

    # Use the outer envelope of crop-supported endpoints.  This never truncates
    # a longer crop pass; shorter endpoints may be extended only within the
    # explicit compensation limit and the selected field boundary.
    wall_projection = {
        "low": float(min(low_projections)),
        "high": float(max(high_projections)),
    }
    try:
        endpoint_extension_m = float(
            cfg.get("work_line_endpoint_extension_m", 0.0)
        )
    except (TypeError, ValueError):
        endpoint_extension_m = 0.0
    if not math.isfinite(endpoint_extension_m) or endpoint_extension_m < 0.0:
        diagnostics["fallback_reason"] = "invalid_work_line_endpoint_extension"
        return [], diagnostics
    diagnostics["configured_work_line_endpoint_extension_m"] = (
        endpoint_extension_m
    )
    forced_walls = cfg.get("_contour_forced_wall_projections")
    if isinstance(forced_walls, dict):
        forced_low = _finite_float(forced_walls.get("low"))
        forced_high = _finite_float(forced_walls.get("high"))
        if (
            forced_low is not None
            and forced_high is not None
            and forced_high > forced_low
        ):
            wall_projection = {"low": forced_low, "high": forced_high}
            diagnostics["shared_headland_scope"] = "all_classified_regions"
        else:
            diagnostics["shared_headland_scope"] = "current_region"
    else:
        diagnostics["shared_headland_scope"] = "current_region"
    # A small, explicit outward extension gives the machine a usable straight
    # headland approach.  It is applied after any field-wide forced walls so
    # every classified region receives the same extension.  Boundary checking
    # below keeps this from becoming a hidden excursion outside the field.
    endpoint_extension_px = endpoint_extension_m / max(float(mpp), 1e-9)
    wall_projection["low"] -= endpoint_extension_px
    wall_projection["high"] += endpoint_extension_px
    anchors = [
        {
            "start": (float(line[0][0]), float(line[0][1])),
            "end": (float(line[-1][0]), float(line[-1][1])),
        }
        for line in oriented_lines
    ]
    endpoint_adjustments = []
    for record in endpoint_records:
        line_index = record["line_index"]
        for endpoint_name in ("start", "end"):
            endpoint = record[endpoint_name]
            terminal_unaligned_endpoint = cfg.get(
                "_contour_terminal_unaligned_endpoint_name"
            )
            if terminal_unaligned_endpoint is None and bool(
                cfg.get("_contour_terminal_unaligned_finish", False)
            ):
                terminal_unaligned_endpoint = "end"
            if (
                line_index == len(endpoint_records) - 1
                and endpoint_name == terminal_unaligned_endpoint
            ):
                anchors[line_index][endpoint_name] = tuple(map(float, endpoint))
                endpoint_adjustments.append(0.0)
                diagnostics["terminal_unaligned_endpoint"] = endpoint_name
                continue
            projection = record[f"{endpoint_name}_projection"]
            side = record[f"{endpoint_name}_side"]
            target_projection = wall_projection[side]
            target = endpoint + direction * (target_projection - projection)
            extension_m = float(np.linalg.norm(target - endpoint) * mpp)
            if extension_m > max_extension_m + 1e-6:
                diagnostics["fallback_reason"] = "headland_wall_extension_too_long"
                diagnostics["failed_line"] = line_index
                diagnostics["failed_endpoint"] = endpoint_name
                diagnostics["failed_extension_m"] = extension_m
                return [], diagnostics
            if extension_m > 1e-7 and not _segment_inside_field_boundary(
                [tuple(endpoint), tuple(target)], field_polygon
            ):
                diagnostics["fallback_reason"] = "headland_wall_extension_leaves_boundary"
                diagnostics["failed_line"] = line_index
                diagnostics["failed_endpoint"] = endpoint_name
                return [], diagnostics
            anchors[line_index][endpoint_name] = tuple(map(float, target))
            endpoint_adjustments.append(extension_m)

    turn_distances_m = []
    transition_walls = []
    for transition_index in range(len(oriented_lines) - 1):
        current_target = np.asarray(
            anchors[transition_index]["end"], dtype=np.float64
        )
        next_target = np.asarray(
            anchors[transition_index + 1]["start"], dtype=np.float64
        )
        current_projection = float(current_target @ direction)
        next_projection = float(next_target @ direction)
        if abs(current_projection - next_projection) * mpp > 1e-6:
            diagnostics["fallback_reason"] = "transition_endpoints_not_on_shared_wall"
            diagnostics["failed_transition"] = transition_index
            return [], diagnostics
        span_m = float(np.linalg.norm(next_target - current_target) * mpp)
        if not math.isfinite(span_m) or span_m <= 1e-9:
            diagnostics["fallback_reason"] = "transition_span_degenerate"
            diagnostics["failed_transition"] = transition_index
            return [], diagnostics
        side = endpoint_records[transition_index]["end_side"]
        if side != endpoint_records[transition_index + 1]["start_side"]:
            diagnostics["fallback_reason"] = "transition_endpoints_on_opposite_walls"
            diagnostics["failed_transition"] = transition_index
            return [], diagnostics
        turn_distances_m.append(span_m)
        transition_walls.append(side)
        diagnostics["connector_count"] += 1

    turn_distances_nonincreasing = not any(
        later > earlier + 1e-6
        for earlier, later in zip(turn_distances_m, turn_distances_m[1:])
    )

    diagnostics.update({
        "fallback_reason": None,
        "shared_headland_lines": True,
        "headland_wall_projections_px": wall_projection,
        "transition_walls": transition_walls,
        "turn_distances_m": turn_distances_m,
        "turn_distances_nonincreasing": turn_distances_nonincreasing,
        "max_endpoint_extension_m": max(endpoint_adjustments, default=0.0),
        "configured_max_endpoint_extension_m": max_extension_m,
    })
    return anchors, diagnostics


def _finite_float(value):
    """Return a finite numeric value, or ``None`` for missing/invalid metrics."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_band_mask(shape, bands: List[Dict]) -> np.ndarray:
    """Build a planning-only mask without modifying the optimized crop mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    contours = []
    for band in bands:
        contour = band.get("contour")
        if contour is None:
            contour = band.get("planning_contour")
        if contour is None:
            continue
        contour = np.asarray(contour, dtype=np.int32)
        if contour.ndim == 2 and contour.shape[0] >= 3:
            contours.append(contour)
    if contours:
        cv2.fillPoly(mask, contours, 255)
    return mask


def _centerline_simplification_settings(config: Dict) -> Dict[str, float | int]:
    """Return bounded controls for the operator-facing centerline polyline."""
    cfg = config or {}
    try:
        max_deviation_m = float(
            cfg.get("centerline_simplify_max_deviation_m", 0.20)
        )
    except (TypeError, ValueError):
        max_deviation_m = 0.20
    if not math.isfinite(max_deviation_m) or max_deviation_m < 0.0:
        max_deviation_m = 0.20

    try:
        max_heading_change_deg = float(
            cfg.get("centerline_simplify_max_heading_change_deg", 8.0)
        )
    except (TypeError, ValueError):
        max_heading_change_deg = 8.0
    if not math.isfinite(max_heading_change_deg) or max_heading_change_deg < 0.0:
        max_heading_change_deg = 8.0

    try:
        max_points = int(cfg.get("centerline_simplify_max_points", 8))
    except (TypeError, ValueError):
        max_points = 8
    max_points = max(2, min(max_points, 64))
    return {
        "max_deviation_m": max_deviation_m,
        "max_heading_change_deg": max_heading_change_deg,
        "max_points": max_points,
    }


def _simplify_centerline_polyline(
    points: List[Tuple[float, float]],
    mpp: float,
    config: Dict,
) -> List[Tuple[float, float]]:
    """Keep endpoints and add vertices only where a band bends materially.

    The dense row-profile trace is useful as source evidence, but it should not
    become the operator's selectable geometry.  This bounded recursive
    simplifier keeps one polyline per band, using perpendicular chord error and
    local heading change as the two curvature signals.
    """
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] != 2
        or not np.all(np.isfinite(values))
    ):
        return []
    if values.shape[0] == 2:
        return [(float(x), float(y)) for x, y in values]

    settings = _centerline_simplification_settings(config)
    try:
        meters_per_pixel_value = float(mpp)
    except (TypeError, ValueError):
        meters_per_pixel_value = 0.0
    if not math.isfinite(meters_per_pixel_value) or meters_per_pixel_value <= 0.0:
        return [(float(x), float(y)) for x, y in values[[0, -1]]]
    deviation_px = float(settings["max_deviation_m"]) / meters_per_pixel_value
    heading_limit = float(settings["max_heading_change_deg"])
    max_points = int(settings["max_points"])

    def segment_heading(start: int, end: int) -> Optional[float]:
        dx = float(values[end, 0] - values[start, 0])
        dy = float(values[end, 1] - values[start, 1])
        if math.hypot(dx, dy) <= 1e-9:
            return None
        return math.atan2(dy, dx)

    def heading_delta(first: float, second: float) -> float:
        return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)

    def range_error(start: int, end: int):
        chord = values[end] - values[start]
        chord_length = float(np.linalg.norm(chord))
        if chord_length <= 1e-9:
            deviations = np.linalg.norm(values[start + 1:end] - values[start], axis=1)
        else:
            relative = values[start + 1:end] - values[start]
            deviations = np.abs(
                chord[0] * relative[:, 1] - chord[1] * relative[:, 0]
            ) / chord_length
        if deviations.size:
            deviation_offset = int(np.argmax(deviations))
            max_deviation = float(deviations[deviation_offset])
            deviation_index = start + 1 + deviation_offset
        else:
            max_deviation = 0.0
            deviation_index = None

        chord_heading = segment_heading(start, end)
        max_heading = 0.0
        heading_index = None
        if chord_heading is not None:
            for index in range(start, end):
                local_heading = segment_heading(index, index + 1)
                if local_heading is None:
                    continue
                delta = heading_delta(local_heading, chord_heading)
                if delta > max_heading:
                    max_heading = delta
                    heading_index = index
        return (
            max_deviation,
            deviation_index,
            math.degrees(max_heading),
            heading_index,
        )

    def split_index(start: int, end: int, error):
        max_deviation, deviation_index, max_heading, heading_index = error
        deviation_score = (
            max_deviation / max(deviation_px, 1e-9)
            if deviation_index is not None
            else 0.0
        )
        heading_score = (
            max_heading / max(heading_limit, 1e-9)
            if heading_index is not None
            else 0.0
        )
        if deviation_score <= 1.0 and heading_score <= 1.0:
            return None, 0.0
        if deviation_score >= heading_score and deviation_index is not None:
            candidate = deviation_index
        elif heading_index is not None:
            candidate = heading_index + 1
        else:
            candidate = deviation_index
        candidate = max(start + 1, min(end - 1, int(candidate)))
        return candidate, max(deviation_score, heading_score)

    kept = {0, values.shape[0] - 1}
    pending = [(0, values.shape[0] - 1)]
    while pending and len(kept) < max_points:
        best = None
        for start, end in pending:
            error = range_error(start, end)
            candidate, score = split_index(start, end, error)
            if candidate is None:
                continue
            if best is None or score > best[0]:
                best = (score, start, end, candidate)
        if best is None:
            break
        _score, start, end, candidate = best
        pending.remove((start, end))
        kept.add(candidate)
        pending.extend(((start, candidate), (candidate, end)))

    return [
        (float(values[index, 0]), float(values[index, 1]))
        for index in sorted(kept)
    ]


def _smooth_band_centerline(
    points: List[Tuple[float, float]],
    main_angle: float,
    mpp: float,
    config: Dict,
) -> List[Tuple[float, float]]:
    """Fit one band centerline in row-aligned coordinates without overshoot."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return []

    direction = np.asarray(
        [math.cos(main_angle), math.sin(main_angle)],
        dtype=np.float64,
    )
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    along = values @ direction
    lateral = values @ normal
    order = np.argsort(along)
    along = along[order]
    lateral = lateral[order]

    keep = np.concatenate(([True], np.diff(along) > 1e-6))
    along = along[keep]
    lateral = lateral[keep]
    if along.size < 2:
        return []

    interval_m = float(config.get("centerline_interval_m", 0.60))
    interval_px = max(1.0, interval_m / max(mpp, 1e-6))
    trim_m = float(config.get("work_line_end_trim_m", 0.0))
    trim_px = max(0.0, trim_m / max(mpp, 1e-6))
    start = float(along[0])
    end = float(along[-1])
    if end - start > 2.0 * trim_px + 2.0 * interval_px:
        start += trim_px
        end -= trim_px
    if end - start < interval_px:
        start, end = float(along[0]), float(along[-1])

    sample_count = max(2, int(math.ceil((end - start) / interval_px)) + 1)
    sample_along = np.linspace(start, end, sample_count)
    sample_lateral = np.interp(sample_along, along, lateral)

    outlier_window = float(config.get("centerline_outlier_window_m", 1.4))
    outlier_samples = max(3, int(round(
        outlier_window / max(interval_m, 1e-6)
    )))
    if outlier_samples % 2 == 0:
        outlier_samples += 1
    if outlier_samples > sample_lateral.size:
        outlier_samples = (
            sample_lateral.size
            if sample_lateral.size % 2 == 1
            else sample_lateral.size - 1
        )
    if outlier_samples >= 3:
        radius = outlier_samples // 2
        padded = np.pad(sample_lateral, (radius, radius), mode="edge")
        trend = np.asarray([
            np.median(padded[index:index + outlier_samples])
            for index in range(sample_lateral.size)
        ])
    else:
        trend = sample_lateral.copy()
    residual = sample_lateral - trend
    median_residual = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median_residual)))
    max_outlier_px = float(config.get("centerline_outlier_m", 0.10)) / max(
        mpp,
        1e-6,
    )
    threshold = max(max_outlier_px, 3.5 * 1.4826 * mad)
    trusted = np.abs(residual - median_residual) <= threshold
    if np.count_nonzero(trusted) >= 2:
        sample_lateral = np.interp(
            sample_along,
            sample_along[trusted],
            sample_lateral[trusted],
        )

    smooth_m = float(config.get("centerline_smooth_m", 1.0))
    sample_lateral = smooth_1d(
        sample_lateral,
        max(3, smooth_m / max(interval_m, 1e-6)),
    )
    fitted = (
        sample_along[:, None] * direction[None, :]
        + sample_lateral[:, None] * normal[None, :]
    )
    return _simplify_centerline_polyline(fitted, mpp, config)


def prepare_band_centerlines(
    wide_bands: List[Dict],
    main_angle: float,
    geo=None,
    state=None,
    config: Optional[Dict] = None,
    reference_bands: Optional[List[Dict]] = None,
) -> Tuple[List[List[Tuple[float, float]]], Dict]:
    """Filter, smooth and order four-row band centerlines across the field."""
    cfg = dict(config or {})
    mask = (
        getattr(state, "mask_raw", None)
        if state is not None and getattr(state, "mask_raw", None) is not None
        else np.zeros((2, 2), dtype=np.uint8)
    )
    mpp = meters_per_pixel(mask, geo, state)
    direction = np.asarray(
        [math.cos(main_angle), math.sin(main_angle)],
        dtype=np.float64,
    )
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)

    candidates = []
    for band in wide_bands:
        source_centerline = list(band.get("centerline", []))
        centerline = _smooth_band_centerline(
            source_centerline,
            main_angle,
            mpp,
            cfg,
        )
        if len(centerline) < 2:
            continue
        points = np.asarray(centerline, dtype=np.float64)
        along = points @ direction
        span_m = float(np.max(along) - np.min(along)) * mpp
        candidates.append({
            "id": int(band.get("id", len(candidates) + 1)),
            "line": centerline,
            "normal": float(np.median(points @ normal)),
            "span_m": span_m,
            "declared_span_m": _finite_float(
                band.get("longitudinal_extent_m")
            ),
            "observed_span_ratio": _finite_float(
                band.get("observed_span_ratio")
            ),
            "source_point_count": len(source_centerline),
        })

    if not candidates:
        return [], {"reason": "no_valid_four_row_centerlines"}

    endpoint_repair_report = {
        "enabled": bool(cfg.get("centerline_endpoint_repair_enabled", True)),
        "applied": False,
        "reference_basis": None,
        "reference_band_count": 0,
        "cohort_count": 0,
        "target_low_projection_px": None,
        "target_high_projection_px": None,
        "target_span_m": None,
        "lines": [],
        "rejected": {},
    }
    # Automatic endpoint repair remains evidence-bound: a band may reach a
    # neighbouring common wall only when its own declared longitudinal extent
    # supports that wall and the complete extension stays inside the selected
    # field. Orthomosaic gaps outside that evidence are corrected explicitly in
    # route editing, where the user moves the shared work/turn control point.
    if endpoint_repair_report["enabled"] and state is not None:
        field_polygon = field_polygon_local(state, mask.shape[:2])
        declared = [
            item["declared_span_m"] for item in candidates
            if item["declared_span_m"] is not None
            and item["declared_span_m"] > 0.0
        ]
        min_cohort = max(2, int(cfg.get(
            "centerline_endpoint_repair_min_cohort", 3
        )))
        if field_polygon is None:
            endpoint_repair_report["rejected"]["missing_field_boundary"] = 1
        elif len(declared) < 2:
            endpoint_repair_report["rejected"]["missing_declared_extent"] = 1
        else:
            declared_median = float(np.median(np.asarray(declared)))
            completion_threshold = float(cfg.get(
                "centerline_endpoint_repair_completion_ratio", 0.85
            ))
            cohort = [
                item for item in candidates
                if item["declared_span_m"] is not None
                and item["declared_span_m"] > 0.0
                and item["span_m"] / item["declared_span_m"]
                >= completion_threshold
                and item["span_m"] >= declared_median * 0.80
            ]
            endpoint_repair_report["reference_basis"] = (
                "candidate_declared_longitudinal_extent"
            )
            endpoint_repair_report["reference_band_count"] = len(
                candidates
            )
            endpoint_repair_report["cohort_count"] = len(cohort)
            if len(cohort) < min_cohort:
                endpoint_repair_report["rejected"]["insufficient_full_span_cohort"] = 1
            else:
                cohort_low = [
                    float(np.min(np.asarray(item["line"]) @ direction))
                    for item in cohort
                ]
                cohort_high = [
                    float(np.max(np.asarray(item["line"]) @ direction))
                    for item in cohort
                ]
                target_low = float(np.median(np.asarray(cohort_low)))
                target_high = float(np.median(np.asarray(cohort_high)))
                target_span_m = (target_high - target_low) * mpp
                endpoint_repair_report.update({
                    "target_low_projection_px": target_low,
                    "target_high_projection_px": target_high,
                    "target_span_m": float(target_span_m),
                })
                max_repair_m = max(0.0, float(cfg.get(
                    "centerline_endpoint_repair_max_m", 60.0
                )))
                for item in candidates:
                    line = list(item["line"])
                    source_span_m = float(item["span_m"])
                    points = np.asarray(line, dtype=np.float64)
                    projections = points @ direction
                    laterals = points @ normal
                    low = float(np.min(projections))
                    high = float(np.max(projections))
                    low_extension_m = max(0.0, (low - target_low) * mpp)
                    high_extension_m = max(0.0, (target_high - high) * mpp)
                    self_evidence_supported = (
                        item["declared_span_m"] is not None
                        and item["declared_span_m"]
                        >= target_span_m * 0.80
                    )
                    eligible = self_evidence_supported and (
                        low_extension_m > 1e-6
                        or high_extension_m > 1e-6
                    )
                    if not eligible:
                        continue
                    if low_extension_m > max_repair_m or high_extension_m > max_repair_m:
                        endpoint_repair_report["rejected"][
                            "endpoint_extension_over_limit"
                        ] = int(endpoint_repair_report["rejected"].get(
                            "endpoint_extension_over_limit", 0
                        )) + 1
                        continue

                    repaired_line = line
                    low_target = None
                    high_target = None
                    endpoint_sample_count = min(5, len(line))
                    if low_extension_m > 1e-6:
                        lateral = float(np.median(laterals[:endpoint_sample_count]))
                        low_target = target_low * direction + lateral * normal
                        if not segment_within_field_boundary(
                            [tuple(low_target), tuple(points[0])],
                            field_polygon,
                        ):
                            endpoint_repair_report["rejected"][
                                "low_extension_leaves_field"
                            ] = int(endpoint_repair_report["rejected"].get(
                                "low_extension_leaves_field", 0
                            )) + 1
                            low_target = None
                        else:
                            repaired_line = [tuple(map(float, low_target))] + repaired_line
                    if high_extension_m > 1e-6:
                        lateral = float(np.median(laterals[-endpoint_sample_count:]))
                        high_target = target_high * direction + lateral * normal
                        if not segment_within_field_boundary(
                            [tuple(points[-1]), tuple(high_target)],
                            field_polygon,
                        ):
                            endpoint_repair_report["rejected"][
                                "high_extension_leaves_field"
                            ] = int(endpoint_repair_report["rejected"].get(
                                "high_extension_leaves_field", 0
                            )) + 1
                            high_target = None
                        else:
                            repaired_line = repaired_line + [
                                tuple(map(float, high_target))
                            ]
                    if low_target is None and high_target is None:
                        continue
                    item["line"] = repaired_line
                    repaired_points = np.asarray(repaired_line, dtype=np.float64)
                    item["span_m"] = float(
                        np.ptp(repaired_points @ direction) * mpp
                    )
                    item["normal"] = float(
                        np.median(repaired_points @ normal)
                    )
                    endpoint_repair_report["applied"] = True
                    endpoint_repair_report["lines"].append({
                        "band_id": item["id"],
                        "source_span_m": source_span_m,
                        "low_extension_m": float(
                            0.0 if low_target is None else low_extension_m
                        ),
                        "high_extension_m": float(
                            0.0 if high_target is None else high_extension_m
                        ),
                        "result_span_m": float(item["span_m"]),
                        "completion_kind": "declared_extent_repair",
                        "inferred": False,
                    })

    median_span = float(np.median([item["span_m"] for item in candidates]))
    min_span_ratio = float(cfg.get("centerline_min_span_ratio", 0.65))
    accepted = [
        item
        for item in candidates
        if item["span_m"] >= median_span * min_span_ratio
    ]
    if not accepted:
        accepted = candidates
    accepted.sort(key=lambda item: item["normal"])

    simplification = _centerline_simplification_settings(cfg)
    simplification["algorithm_version"] = "centerline-adaptive-polyline-v1"
    simplification["records"] = [
        {
            "band_id": item["id"],
            "source_point_count": int(item.get("source_point_count", 0)),
            "centerline_point_count": int(len(item["line"])),
        }
        for item in accepted
    ]
    return [item["line"] for item in accepted], {
        "source": "four_row_band_centerlines",
        "candidate_four_row_count": len(wide_bands),
        "accepted_centerline_count": len(accepted),
        "median_centerline_span_m": median_span,
        "ordered_band_ids": [item["id"] for item in accepted],
        "pass_count": len(accepted),
        "centerline_simplification": simplification,
        "endpoint_repair": endpoint_repair_report,
    }


# ═══════════════════════════════════════════════════════════════
#  坐标空间辅助函数
# ═══════════════════════════════════════════════════════════════

def _ensure_local_coords(full_path: List[List[Tuple[float, float]]],
                         mask: np.ndarray, state=None
                         ) -> List[List[Tuple[float, float]]]:
    """确保路径坐标与掩膜坐标系一致。

    如果路径点全部超出 mask 范围（可能是全图坐标），
    且 state 中存在 mask_offset，则自动减去偏移量。
    """
    if not full_path or mask is None:
        return full_path

    h, w = mask.shape[:2]
    # 收集所有路径点
    all_pts = []
    for seg in full_path:
        all_pts.extend(seg)
    if not all_pts:
        return full_path

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 判断：如果路径坐标明显超出 mask 范围且有 offset 信息
    ox = int(getattr(state, 'mask_offset_x', 0)) if state else 0
    oy = int(getattr(state, 'mask_offset_y', 0)) if state else 0

    if (ox != 0 or oy != 0) and (x_min >= w or y_min >= h or x_max >= w * 2 or y_max >= h * 2):
        # 路径坐标在全图空间，需要减偏移量回到局部坐标
        return [[(px - ox, py - oy) for px, py in seg] for seg in full_path]

    return full_path

def _path_to_global_coords(full_path: List[List[Tuple[float, float]]], state=None) -> List[List[Tuple[float, float]]]:
    """Convert local mask-space path coordinates to full-image display coordinates."""
    ox = float(getattr(state, "mask_offset_x", 0) or 0) if state is not None else 0.0
    oy = float(getattr(state, "mask_offset_y", 0) or 0) if state is not None else 0.0
    if ox == 0.0 and oy == 0.0:
        return [[(float(x), float(y)) for x, y in seg] for seg in full_path]
    return [[(float(x) + ox, float(y) + oy) for x, y in seg] for seg in full_path]


def _status_for_segment_type(segment_type: str) -> int:
    segment_type = str(segment_type or "turn")
    if segment_type == "work":
        return 1
    if segment_type == "entry":
        return 2
    if segment_type == "exit":
        return 3
    if segment_type in ("turn_reverse", "turn_aux"):
        return 4
    return 0


def _field_boundary_local_polygon(state=None) -> Optional[np.ndarray]:
    """Return the selected field boundary in local mask coordinates, if available."""
    if state is None:
        return None
    mask = getattr(state, "mask_raw", None)
    if mask is None:
        return None
    return field_polygon_local(state, mask.shape[:2])


def _points_inside_polygon(points, polygon: Optional[np.ndarray], tolerance: float = 1e-6) -> bool:
    if polygon is None:
        return True
    poly = np.asarray(polygon, dtype=np.float32)
    for x, y in points:
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) < -tolerance:
            return False
    return True


def segment_within_field_boundary(segment, polygon: Optional[np.ndarray]) -> bool:
    """Return whether an entire polyline is covered by the selected field.

    DECISION-010/011: checking only sampled route vertices is insufficient for
    concave field boundaries because one edge can cross a notch between two
    in-boundary endpoints.  This helper deliberately uses ``covers`` so a
    route exactly on the selected boundary remains valid.
    """
    if polygon is None:
        return True
    try:
        boundary_points = [
            (float(point[0]), float(point[1]))
            for point in np.asarray(polygon, dtype=np.float64)
        ]
        if len(boundary_points) < 3:
            return False
        field = Polygon(boundary_points)
        if field.is_empty or not field.is_valid:
            return False

        points = []
        for point in segment or []:
            current = (float(point[0]), float(point[1]))
            if not points or current != points[-1]:
                points.append(current)
        if not points:
            return False
        geometry = Point(points[0]) if len(points) == 1 else LineString(points)
        return bool(field.covers(geometry))
    except (TypeError, ValueError, IndexError):
        return False


def _segment_inside_field_boundary(segment, polygon: Optional[np.ndarray]) -> bool:
    """Backward-compatible private name for complete field-boundary coverage."""
    return segment_within_field_boundary(segment, polygon)


def _filter_generated_centerline_regions_to_field(
    work_regions,
    field_polygon: Optional[np.ndarray],
) -> Tuple[List[Dict], Dict]:
    """Drop generated centerlines that are not covered by the selected field.

    Row reconstruction and uniform-zone generation both have their own mask
    evidence gates.  The final operator catalog still needs one common hard
    boundary gate because a detached mask component can otherwise become a
    selectable centerline outside the blue field boundary.
    """
    regions = [dict(region) for region in (work_regions or [])]
    report = {
        "algorithm_version": "centerline-field-boundary-filter-v1",
        "enabled": field_polygon is not None,
        "input_line_count": sum(
            len(region.get("work_lines") or []) for region in regions
        ),
        "kept_line_count": 0,
        "excluded": [],
        "reason": None,
    }
    if field_polygon is None:
        report["reason"] = "missing_field_boundary"
        report["kept_line_count"] = report["input_line_count"]
        return regions, report

    filtered_regions = []
    for region_order, region in enumerate(regions):
        lines = list(region.get("work_lines") or [])
        kept_indices = []
        for line_index, line in enumerate(lines):
            if _segment_inside_field_boundary(line, field_polygon):
                kept_indices.append(line_index)
                continue
            report["excluded"].append({
                "region_id": region.get("region_id"),
                "region_order": int(region_order),
                "region_line_index": int(line_index),
                "reason": "centerline_outside_field_boundary",
            })

        if not kept_indices:
            continue

        updated = dict(region)
        updated["work_lines"] = [lines[index] for index in kept_indices]
        updated["line_count"] = len(kept_indices)

        source_ids = list(region.get("work_line_source_ids") or [])
        if len(source_ids) == len(lines):
            updated["work_line_source_ids"] = [
                source_ids[index] for index in kept_indices
            ]

        band_ids = list(region.get("band_ids") or [])
        if len(band_ids) == len(lines):
            updated["band_ids"] = [
                band_ids[index] for index in kept_indices
            ]
        filtered_regions.append(updated)

    report["kept_line_count"] = sum(
        len(region.get("work_lines") or []) for region in filtered_regions
    )
    report["excluded_count"] = len(report["excluded"])
    if report["excluded"]:
        report["reason"] = "excluded_lines_outside_field_boundary"
    return filtered_regions, report


def _non_work_boundary_violations(full_path, segment_types, polygon: Optional[np.ndarray]) -> List[Dict]:
    if polygon is None:
        return []
    violations = []
    for index, segment in enumerate(full_path or []):
        segment_type = (
            segment_types[index]
            if segment_types and index < len(segment_types)
            else ("work" if index % 2 == 0 else "turn")
        )
        if segment_type == "work":
            continue
        if not _segment_inside_field_boundary(segment, polygon):
            violations.append({
                "segment_index": int(index),
                "segment_type": str(segment_type),
            })
    return violations


def _adjust_turn_to_field_boundary(
    start: Tuple[float, float],
    end: Tuple[float, float],
    work_direction: Tuple[float, float],
    turn_sign: float,
    mpp: float,
    min_turn_radius_m: float,
    selected_strategy: str,
    clearance_m: float,
    polygon: Optional[np.ndarray],
    semicircle_radius_tolerance_m: float = 0.0,
    semicircle_tangent_radius_m: Optional[float] = None,
    preserve_curve_on_boundary_failure: bool = False,
) -> Tuple[List[Tuple[float, float]], str, Optional[str]]:
    """Generate a turn, shrinking outward clearance when it would leave the selected field.

    The normal planner still prefers a boundary-contained candidate.  Review
    needs to show the actual named curved candidate even when the selected
    field polygon cannot contain it: the GUI can then colour the outside pieces
    red and the operator can adjust the shared endpoint. Keeping that candidate
    also prevents a bow/semicircle/pear turn from being silently replaced by a
    visually misleading two-point chord. Validation remains fail-closed.
    """
    turn_function = TURN_STRATEGIES.get(selected_strategy, _turn_bow)
    clearances = []
    for scale in (1.0, 0.5, 0.25, 0.0):
        value = max(0.0, float(clearance_m) * scale)
        if value not in clearances:
            clearances.append(value)
    best_turn = None
    for value in clearances:
        if selected_strategy == "semicircle":
            turn = turn_function(
                start,
                end,
                work_direction,
                turn_sign,
                mpp,
                min_turn_radius_m,
                value,
                radius_tolerance_m=semicircle_radius_tolerance_m,
                tangent_radius_m=semicircle_tangent_radius_m,
            )
        else:
            turn = turn_function(
                start,
                end,
                work_direction,
                turn_sign,
                mpp,
                min_turn_radius_m,
                value,
            )
        if best_turn is None:
            best_turn = turn
        if _segment_inside_field_boundary(turn, polygon):
            reason = None
            if abs(value - float(clearance_m)) > 1e-9:
                reason = "转弯外扩会越出框选田块，已自动收缩地头外扩距离"
            return turn, selected_strategy, reason

    if preserve_curve_on_boundary_failure and best_turn and len(best_turn) > 2:
        return (
            best_turn,
            selected_strategy,
            "转弯曲线超出框选田块，保留曲线供人工调整并以红色标示越界部分",
        )

    # Last safe fallback: connect endpoints directly. It is less smooth, but never
    # knowingly leaves the user's selected field boundary.
    direct = [start, end]
    if _segment_inside_field_boundary(direct, polygon):
        return direct, selected_strategy, "转弯曲线会越出框选田块，已退化为边界内端点连接"
    return best_turn or direct, selected_strategy, "转弯曲线仍可能越出框选田块，请调整边界/起终点或手动编辑路线"



# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════

def _turn_point_count(
    arc_length_px: float,
    mpp: float,
    minimum: int,
) -> int:
    config = Config().section("path_planning")
    interval_m = float(config.get("turn_point_interval_m", 0.25))
    interval_px = max(2.0, interval_m / max(mpp, 1e-6))
    return max(minimum, int(math.ceil(arc_length_px / interval_px)))


def _turn_frame(start, end, work_dir):
    start_v = np.asarray(start, dtype=np.float64)
    end_v = np.asarray(end, dtype=np.float64)
    chord = end_v - start_v
    distance = float(np.linalg.norm(chord))
    if distance <= 1e-9:
        return start_v, end_v, distance, None, None
    lateral = chord / distance
    outward = np.asarray(work_dir, dtype=np.float64)
    outward -= lateral * float(np.dot(outward, lateral))
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= 1e-9:
        outward = np.asarray((-lateral[1], lateral[0]), dtype=np.float64)
        if float(np.dot(outward, np.asarray(work_dir, dtype=np.float64))) < 0:
            outward *= -1.0
    else:
        outward /= outward_norm
    return start_v, end_v, distance, lateral, outward


def _append_distinct(target, points):
    for point in points:
        point = (float(point[0]), float(point[1]))
        if target and math.hypot(
            point[0] - target[-1][0],
            point[1] - target[-1][1],
        ) <= 1e-7:
            continue
        target.append(point)


def _sample_arc(center, radius, start_angle, sweep_angle, mpp, minimum=6):
    count = _turn_point_count(abs(sweep_angle) * radius, mpp, minimum)
    return [
        (
            float(center[0] + radius * math.cos(start_angle + sweep_angle * i / count)),
            float(center[1] + radius * math.sin(start_angle + sweep_angle * i / count)),
        )
        for i in range(count + 1)
    ]


def _turn_clearance_px(clearance_m, mpp):
    return float(clearance_m) / max(mpp, 1e-6)


def _turn_bow(start: Tuple[float, float], end: Tuple[float, float],
              work_dir: Tuple[float, float], perp_sign: float,
              mpp: float, min_radius_m: float,
              clearance_m: float = 0.0) -> List[Tuple[float, float]]:
    """Bow/U turn: two radius-R quarter arcs plus a lateral tangent."""
    start_v, end_v, distance, lateral, outward = _turn_frame(
        start, end, work_dir
    )
    if distance < 2.0 or lateral is None:
        return [start, end]
    radius = min_radius_m / max(mpp, 1e-6)

    clearance = _turn_clearance_px(clearance_m, mpp)
    start_out = start_v + outward * clearance
    end_out = end_v + outward * clearance
    first_center = start_out + lateral * radius
    second_center = end_out - lateral * radius
    first_top = first_center + outward * radius
    second_top = second_center + outward * radius

    base_angle = math.atan2(-lateral[1], -lateral[0])
    top_angle = math.atan2(outward[1], outward[0])
    first_sweep = -math.pi / 2.0
    if abs(
        ((base_angle + first_sweep - top_angle + math.pi) % (2 * math.pi))
        - math.pi
    ) > 1e-4:
        first_sweep = math.pi / 2.0
    second_start_angle = top_angle
    second_end_angle = math.atan2(lateral[1], lateral[0])
    second_sweep = -math.pi / 2.0
    if abs(
        ((second_start_angle + second_sweep - second_end_angle + math.pi)
         % (2 * math.pi)) - math.pi
    ) > 1e-4:
        second_sweep = math.pi / 2.0

    points = []
    _append_distinct(points, [start_v, start_out])
    _append_distinct(
        points,
        _sample_arc(
            first_center, radius, base_angle, first_sweep, mpp, minimum=8
        ),
    )
    _append_distinct(points, [first_top, second_top])
    _append_distinct(
        points,
        _sample_arc(
            second_center,
            radius,
            second_start_angle,
            second_sweep,
            mpp,
            minimum=8,
        ),
    )
    _append_distinct(points, [end_out, end_v])
    points[0], points[-1] = start, end
    return points


def _turn_semicircle(start: Tuple[float, float], end: Tuple[float, float],
                     work_dir: Tuple[float, float], perp_sign: float,
                     mpp: float, min_radius_m: float,
                     clearance_m: float = 0.0,
                     radius_tolerance_m: float = 0.0,
                     tangent_radius_m: Optional[float] = None,
                     ) -> List[Tuple[float, float]]:
    """Pure semicircular turn whose radius is half the pass spacing."""
    start_v, end_v, distance, lateral, outward = _turn_frame(
        start, end, work_dir
    )
    if distance < 2.0 or lateral is None:
        return [start, end]
    physical_radius_m = distance * max(mpp, 1e-6) * 0.5
    radius_tolerance_m = max(0.0, float(radius_tolerance_m))
    if physical_radius_m < float(min_radius_m) - radius_tolerance_m - 1e-6:
        raise ValueError(
            f"semicircle radius {physical_radius_m:.3f} m is below the "
            f"minimum turn radius {float(min_radius_m):.3f} m"
        )
    tangent_radius_m = max(
        physical_radius_m,
        float(tangent_radius_m)
        if tangent_radius_m is not None
        else physical_radius_m,
    )
    # DECISION-016: a near-threshold centre turn stays on the same
    # headland tangent as the R-radius bows by moving the smaller arc outward
    # by exactly its audited radius shortfall.
    clearance = _turn_clearance_px(
        float(clearance_m) + tangent_radius_m - physical_radius_m,
        mpp,
    )
    start_out = start_v + outward * clearance
    end_out = end_v + outward * clearance
    center = (start_out + end_out) * 0.5
    radius = distance * 0.5
    start_angle = math.atan2(
        start_out[1] - center[1],
        start_out[0] - center[0],
    )
    candidates = (math.pi, -math.pi)

    def outward_score(sweep):
        midpoint = center + radius * np.asarray((
            math.cos(start_angle + sweep * 0.5),
            math.sin(start_angle + sweep * 0.5),
        ))
        return float(np.dot(midpoint - center, outward))

    sweep = max(candidates, key=outward_score)
    points = []
    _append_distinct(points, [start_v, start_out])
    _append_distinct(
        points,
        _sample_arc(
            center,
            radius,
            start_angle,
            sweep,
            mpp,
            minimum=12,
        ),
    )
    _append_distinct(points, [end_out, end_v])
    points[0], points[-1] = start, end
    return points


def _turn_pear(start: Tuple[float, float], end: Tuple[float, float],
               work_dir: Tuple[float, float], perp_sign: float,
               mpp: float, min_radius_m: float,
               clearance_m: float = 0.0) -> List[Tuple[float, float]]:
    """Forward-only pear/bulb turn made from three tangent radius-R arcs.

    The middle arc is the short tangent arc. Adding a full revolution here
    produces the erroneous stacked double-circle path seen in the UI.
    """
    start_v, end_v, distance, lateral, outward = _turn_frame(
        start, end, work_dir
    )
    if distance < 2.0 or lateral is None:
        return [start, end]
    radius = min_radius_m / max(mpp, 1e-6)
    clearance = _turn_clearance_px(clearance_m, mpp)
    start_out = start_v + outward * clearance
    end_out = end_v + outward * clearance

    # The three-circle pear construction exists only for spacing below 2R.
    # Return an explicit failed candidate instead of silently substituting
    # another strategy; the UI will explain and offer a validated alternative.
    if distance >= 2.0 * radius - 1e-6:
        return [
            (float(start_v[0]), float(start_v[1])),
            (float(start_out[0]), float(start_out[1])),
            (float(end_out[0]), float(end_out[1])),
            (float(end_v[0]), float(end_v[1])),
        ]

    # For spacing < 2R, the first and last circle centres must sit outside
    # the two passes. Putting them between the passes makes the centres cross
    # and forces nearly complete revolutions, which appears as two circles.
    first_center = start_out - lateral * radius
    third_center = end_out + lateral * radius
    circle_gap = float(np.linalg.norm(third_center - first_center))
    if circle_gap > 4.0 * radius + 1e-6:
        return [start, end]
    center_mid = (first_center + third_center) * 0.5
    center_axis = third_center - first_center
    axis_norm = float(np.linalg.norm(center_axis))
    if axis_norm <= 1e-9:
        axis_unit = lateral
    else:
        axis_unit = center_axis / axis_norm
    normal = np.asarray((-axis_unit[1], axis_unit[0]), dtype=np.float64)
    if float(np.dot(normal, outward)) < 0:
        normal *= -1.0
    middle_height = math.sqrt(max((2.0 * radius) ** 2 - (circle_gap * 0.5) ** 2, 0.0))
    middle_center = center_mid + normal * middle_height
    tangent_12 = (first_center + middle_center) * 0.5
    tangent_23 = (middle_center + third_center) * 0.5
    handedness = float(
        lateral[0] * outward[1] - lateral[1] * outward[0]
    )
    outer_direction = 1 if handedness >= 0.0 else -1
    middle_direction = -outer_direction

    def angle(center, point):
        return math.atan2(point[1] - center[1], point[0] - center[0])

    def directed_sweep(start_angle, end_angle, direction):
        if direction > 0:
            sweep = (end_angle - start_angle) % (2.0 * math.pi)
        else:
            sweep = -((start_angle - end_angle) % (2.0 * math.pi))
        return sweep

    points = []
    _append_distinct(points, [start_v, start_out])
    first_start = angle(first_center, start_out)
    first_end = angle(first_center, tangent_12)
    _append_distinct(
        points,
        _sample_arc(
            first_center,
            radius,
            first_start,
            directed_sweep(first_start, first_end, outer_direction),
            mpp,
            minimum=8,
        ),
    )
    middle_start = angle(middle_center, tangent_12)
    middle_end = angle(middle_center, tangent_23)
    _append_distinct(
        points,
        _sample_arc(
            middle_center,
            radius,
            middle_start,
            directed_sweep(
                middle_start,
                middle_end,
                middle_direction,
            ),
            mpp,
            minimum=18,
        ),
    )
    third_start = angle(third_center, tangent_23)
    third_end = angle(third_center, end_out)
    _append_distinct(
        points,
        _sample_arc(
            third_center,
            radius,
            third_start,
            directed_sweep(third_start, third_end, outer_direction),
            mpp,
            minimum=8,
        ),
    )
    _append_distinct(points, [end_out, end_v])
    points[0], points[-1] = start, end
    return points


# Backward-compatible name used by older project snapshots and tests.
_turn_bulb = _turn_pear


def _turn_fishtail(start: Tuple[float, float], end: Tuple[float, float],
                   work_dir: Tuple[float, float], perp_sign: float,
                   mpp: float, min_radius_m: float,
                   clearance_m: float = 0.0) -> List[Tuple[float, float]]:
    segments = _turn_reverse_segments(
        start,
        end,
        work_dir,
        mpp,
        min_radius_m,
        clearance_m,
        inner=False,
    )
    points = []
    for segment, _ in segments:
        _append_distinct(points, segment)
    return points or [start, end]


def _turn_alpha(start: Tuple[float, float], end: Tuple[float, float],
                work_dir: Tuple[float, float], perp_sign: float,
                mpp: float, min_radius_m: float,
                clearance_m: float = 0.0) -> List[Tuple[float, float]]:
    segments = _turn_reverse_segments(
        start,
        end,
        work_dir,
        mpp,
        min_radius_m,
        clearance_m,
        inner=True,
    )
    points = []
    for segment, _ in segments:
        _append_distinct(points, segment)
    return points or [start, end]


def _turn_reverse_segments(
        start,
        end,
        work_dir,
        mpp,
        min_radius_m,
        clearance_m,
        inner,
):
    """Build a three-point tracked turn: forward arc, reverse, forward arc."""
    start_v, end_v, distance, lateral, outward = _turn_frame(
        start, end, work_dir
    )
    if distance < 2.0 or lateral is None:
        return [([start, end], "turn")]
    radius = min_radius_m / max(mpp, 1e-6)
    clearance = _turn_clearance_px(clearance_m, mpp)
    start_out = start_v + outward * clearance
    end_out = end_v + outward * clearance

    # Inner alpha switchback is only kinematically a reverse manoeuvre when
    # row spacing is below 2R. For wider spacing the outer fish-tail is used.
    use_inner = bool(inner and distance < 2.0 * radius - 1e-6)
    side = 1.0 if use_inner else -1.0
    first_center = start_out + lateral * side * radius
    second_center = end_out - lateral * side * radius
    first_cusp = first_center + outward * radius
    second_cusp = second_center + outward * radius

    start_angle = math.atan2(
        (start_out - first_center)[1],
        (start_out - first_center)[0],
    )
    cusp_angle = math.atan2(outward[1], outward[0])
    sweep = math.pi / 2.0 if side < 0 else -math.pi / 2.0
    candidate = start_angle + sweep
    if abs(
        ((candidate - cusp_angle + math.pi) % (2.0 * math.pi)) - math.pi
    ) > 1e-4:
        sweep *= -1.0

    first = []
    _append_distinct(first, [start_v, start_out])
    _append_distinct(
        first,
        _sample_arc(
            first_center, radius, start_angle, sweep, mpp, minimum=8
        ),
    )
    first[-1] = (float(first_cusp[0]), float(first_cusp[1]))

    reverse = [
        (float(first_cusp[0]), float(first_cusp[1])),
        (float(second_cusp[0]), float(second_cusp[1])),
    ]

    second_start_angle = cusp_angle
    end_angle = math.atan2(
        (end_out - second_center)[1],
        (end_out - second_center)[0],
    )
    second_sweep = sweep
    if abs(
        ((second_start_angle + second_sweep - end_angle + math.pi)
         % (2.0 * math.pi)) - math.pi
    ) > 1e-4:
        second_sweep *= -1.0
    second = _sample_arc(
        second_center,
        radius,
        second_start_angle,
        second_sweep,
        mpp,
        minimum=8,
    )
    _append_distinct(second, [end_out, end_v])
    second[0] = (float(second_cusp[0]), float(second_cusp[1]))
    second[-1] = end
    return [
        (first, "turn"),
        (reverse, "turn_reverse"),
        (second, "turn_aux"),
    ]


def _arc_between(start: Tuple[float, float], end: Tuple[float, float],
                 radius: float, perp_sign: float,
                 mpp: float) -> List[Tuple[float, float]]:
    """在两点之间生成一段圆弧。"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 2:
        return [start, end]

    # 弧心在两点中点的垂直方向偏移
    perp_x = -dy / dist * perp_sign
    perp_y = dx / dist * perp_sign

    # 确保半径足够大
    half_dist = dist / 2.0
    if radius < half_dist:
        radius = half_dist

    # 弧心高度
    h = math.sqrt(max(radius**2 - half_dist**2, 0.01))
    cx = (start[0] + end[0]) / 2 + perp_x * h
    cy = (start[1] + end[1]) / 2 + perp_y * h

    a_start = math.atan2(start[1] - cy, start[0] - cx)
    a_end = math.atan2(end[1] - cy, end[0] - cx)

    angle_diff = a_end - a_start
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    # 选择弧的方向
    if perp_sign > 0 and angle_diff < 0:
        angle_diff += 2 * math.pi
    elif perp_sign < 0 and angle_diff > 0:
        angle_diff -= 2 * math.pi

    n_points = _turn_point_count(abs(angle_diff) * radius, mpp, 8)
    points = []
    for i in range(n_points + 1):
        t = i / n_points
        a = a_start + angle_diff * t
        px = cx + radius * math.cos(a)
        py = cy + radius * math.sin(a)
        points.append((float(px), float(py)))

    if points:
        points[0] = start
        points[-1] = end
    return points


# 转弯策略分发映射
def _turn_outward_u(start: Tuple[float, float], end: Tuple[float, float],
                    work_dir: Tuple[float, float], perp_sign: float,
                    mpp: float, min_radius_m: float,
                    clearance_m: float = 0.0) -> List[Tuple[float, float]]:
    return _turn_bow(
        start,
        end,
        work_dir,
        perp_sign,
        mpp,
        min_radius_m,
        clearance_m,
    )


TURN_STRATEGY_ALIASES = {
    "uturn": "bow",
    "bulb": "pear",
    "spiral": "contour",
    "round_and_round": "contour",
}

TURN_STRATEGY_LABELS = {
    "bow": "弓形转弯",
    "semicircle": "半圆转弯",
    "pear": "梨形转弯",
    "fishtail": "鱼尾折返",
    "alpha": "紧凑α折返",
    "contour": "套圈收割",
    "auto": "按行距自适应组合",
}


def normalize_turn_strategy(strategy: str) -> str:
    value = str(strategy or "bow").strip().lower()
    return TURN_STRATEGY_ALIASES.get(value, value)


TURN_STRATEGIES = {
    "bow": _turn_bow,
    "semicircle": _turn_semicircle,
    "pear": _turn_pear,
    "fishtail": _turn_fishtail,
    "alpha": _turn_alpha,
    # ``contour`` changes pass order; wide local turns use a shared-line bow.
    "contour": _turn_bow,
    # Backward-compatible project/cache keys.
    "uturn": _turn_bow,
    "bulb": _turn_pear,
}


def recommend_turn_strategy(turn_radius_m: float,
                            pass_spacing_m: float,
                            headland_limited: bool = False) -> dict:
    """Recommend a typical turn from actual pass spacing omega.

    文献依据:
    - Olcay et al. (2018): Headland Turn Automation for Autonomous Farming
    - He et al. (2023): Dynamic path planning for headland turning
    - Trendafilov & Delchev (2025): Modification of Typical Headland Manoeuvres

    选择标准:
    - R < omega/2: bow turn
    - R = omega/2: semicircular turn
    - R > omega/2: pear turn
    - R > omega/2 with limited headland: fishtail/reverse turn

    Args:
        turn_radius_m: 农机最小转弯半径 (m)
        pass_spacing_m: actual adjacent work-pass spacing omega (m)

    Returns:
        {"strategy": str, "reason": str}
    """
    radius = max(1e-6, float(turn_radius_m))
    spacing = max(0.0, float(pass_spacing_m))
    tolerance = max(0.05, radius * 0.05)
    if radius < spacing * 0.5 - tolerance:
        return {
            "strategy": "bow",
            "reason": (
                f"R={radius:.2f}m < ω/2={spacing*0.5:.2f}m，"
                "弓形转弯可保持最小半径并缩短路径"
            ),
        }
    if abs(radius - spacing * 0.5) <= tolerance:
        return {
            "strategy": "semicircle",
            "reason": (
                f"R={radius:.2f}m ≈ ω/2={spacing*0.5:.2f}m，"
                "半圆转弯与相邻作业线自然相切"
            ),
        }
    if headland_limited:
        return {
            "strategy": "fishtail",
            "reason": (
                f"R={radius:.2f}m > ω/2={spacing*0.5:.2f}m，"
                "且田头空间受限，鱼尾倒车折返占用纵深较小"
            ),
        }
    return {
        "strategy": "pear",
        "reason": (
            f"R={radius:.2f}m > ω/2={spacing*0.5:.2f}m，"
            "梨形三圆相切轨迹适用于该行距"
        ),
    }


def _select_turn_strategy(
    requested_strategy: str,
    pass_spacing_m: float,
    turn_radius_m: float,
    headland_limited: bool = False,
    geometry_tolerance_m: Optional[float] = None,
) -> Dict:
    """Choose an executable turn strategy for one adjacent-pass transition.

    This is the single decision point for turn fallback. It returns both the
    requested strategy and the executable strategy so UI/logs can explain when
    the planner intentionally overrides an impossible choice.
    """
    requested = (
        "auto"
        if str(requested_strategy).strip().lower() == "auto"
        else normalize_turn_strategy(requested_strategy)
    )
    spacing = max(0.0, float(pass_spacing_m))
    radius = max(1e-6, float(turn_radius_m))
    tolerance = max(0.05, radius * 0.05)
    radius_tolerance = (
        max(0.0, float(geometry_tolerance_m))
        if geometry_tolerance_m is not None
        else max(0.05, radius * 0.05)
    )
    two_radius = 2.0 * radius
    radius_shortfall = max(0.0, radius - spacing * 0.5)
    fallback_from = None
    fallback_reason = ""

    def build_decision(strategy, reason):
        effective = str(strategy)
        signature = hashlib.sha256(
            json.dumps(
                {
                    "requested_strategy": requested,
                    "candidate_strategy": requested,
                    "effective_strategy": effective,
                    "pass_spacing_m": round(spacing, 9),
                    "turn_radius_m": round(radius, 9),
                    "headland_limited": bool(headland_limited),
                    "geometry_tolerance_m": round(radius_tolerance, 9),
                    "radius_shortfall_m": round(radius_shortfall, 9),
                    "fallback_reason": fallback_reason,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "requested_strategy": requested,
            "candidate_strategy": requested,
            "effective_strategy": effective,
            # Compatibility for existing consumers; use effective_strategy for
            # new records so requested/candidate/effective remain distinct.
            "strategy": effective,
            "reason": reason,
            "fallback_from": fallback_from,
            "fallback_reason": fallback_reason,
            "pass_spacing_m": spacing,
            "turn_radius_m": radius,
            "geometry_tolerance_m": radius_tolerance,
            "radius_shortfall_m": radius_shortfall,
            "geometry_signature": signature,
        }

    # DECISION-010: omega/2 below R is never an executable semicircle.
    # 小间距（ω<2R）优先梨形：fishtail 倒车折返在田头空间不足时容易退化为
    # 直线连接（被足迹验证判为不可行）；pear 前进-倒车-前进在窄田头更稳健。
    if spacing < two_radius - 1e-6 and requested in {
        "auto", "bow", "semicircle"
    }:
        fallback_from = None if requested == "auto" else requested
        if radius_shortfall <= radius_tolerance + 1e-9:
            strategy = "semicircle"
            fallback_reason = (
                f"omega={spacing:.2f}m gives a {radius_shortfall:.3f}m "
                "radius shortfall within the audited geometry tolerance; "
                "using a shared-tangent near-semicircle"
            )
        else:
            strategy = "pear"
            fallback_reason = (
                f"omega={spacing:.2f}m gives omega/2={spacing * 0.5:.2f}m "
                f"below R={radius:.2f}m; using {TURN_STRATEGY_LABELS[strategy]} "
                "instead of a subminimum-radius semicircle"
            )
        return build_decision(strategy, fallback_reason)

    if requested == "auto":
        if spacing < two_radius - tolerance:
            strategy = "semicircle"
            reason = (
                f"ω={spacing:.2f}m < 2R={two_radius:.2f}m，"
                "自动采用端点直连半圆/近半圆调头，避免鱼尾倒车自交"
            )
        else:
            recommendation = recommend_turn_strategy(
                radius,
                spacing,
                headland_limited=headland_limited,
            )
            strategy = recommendation["strategy"]
            reason = recommendation.get("reason", "自动推荐")
    else:
        strategy = requested
        reason = "用户指定"
        if requested == "bow" and spacing < two_radius - tolerance:
            fallback_from = requested
            strategy = "semicircle"
            fallback_reason = (
                f"用户选择弓形调头，但ω={spacing:.2f}m < 2R={two_radius:.2f}m，"
                "弓形两段R圆弧无法相切，已改用端点直连半圆/近半圆调头"
            )
            reason = fallback_reason
        elif requested == "pear" and spacing > two_radius + tolerance:
            fallback_from = requested
            strategy = "bow"
            fallback_reason = (
                f"用户选择梨形调头，但ω={spacing:.2f}m > 2R={two_radius:.2f}m，"
                "三圆相切梨形几何不存在，已改用弓形调头"
            )
            reason = fallback_reason
        elif requested == "semicircle" and spacing < two_radius - tolerance:
            reason = (
                f"用户选择半圆调头；ω/2={spacing * 0.5:.2f}m < R={radius:.2f}m，"
                "按实际相邻端点生成近半圆连接，并在风险提示中报告半径不足"
            )

    return build_decision(strategy, reason)


def assess_turn_strategy(
    strategy: str,
    pass_spacings_m: List[float],
    turn_radius_m: float,
) -> Dict:
    """Assess a selected strategy without replacing it.

    The geometry conditions follow the typical headland manoeuvre definitions
    in Lin et al. (2025), doi:10.3390/app15169157:
    bow for R < omega/2, semicircle for R ~= omega/2, pear for R > omega/2,
    and fishtail for R > omega/2 when reversing is acceptable.
    """
    selected = normalize_turn_strategy(strategy)
    ordering_strategy = selected
    contour_requested = selected == "contour"
    if contour_requested:
        # The nested contour needs a shared transverse tangent.  Bow geometry
        # supplies the finite paired transition.  Unlike an ordinary bow
        # request, contour ordering must not silently become a pear/bulb at a
        # narrow region boundary: the machine-facing topology is still the
        # same two-sided contour pass.
        selected = "bow"
    spacings = [
        float(value)
        for value in pass_spacings_m
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    radius = max(1e-6, float(turn_radius_m))
    tolerance = max(0.05, radius * 0.05)
    hard_reasons = []
    warnings = []

    if not spacings:
        hard_reasons.append("未获得相邻作业线间距，无法验证转弯几何")
    else:
        minimum = min(spacings)
        maximum = max(spacings)
        two_radius = 2.0 * radius
        below = [value for value in spacings if value < two_radius - tolerance]
        above = [value for value in spacings if value > two_radius + tolerance]
        near = [
            value
            for value in spacings
            if abs(value - two_radius) <= tolerance
        ]

        if selected == "bow":
            if below:
                message = (
                    f"{len(below)}处行距ω小于2R（最小{minimum:.2f}m，"
                    f"2R={two_radius:.2f}m）"
                )
                if contour_requested:
                    warnings.append(
                        message + "，套圈转场保留有限弓形连接，不回退为梨形灯泡"
                    )
                else:
                    hard_reasons.append(
                        message + "，弓形转弯的两段R圆弧无法相切连接"
                    )
        elif selected == "semicircle":
            if below:
                hard_reasons.append(
                    f"{len(below)}处半圆半径ω/2小于最小转弯半径R"
                    f"（最小ω/2={minimum*0.5:.2f}m，R={radius:.2f}m）"
                )
            if above:
                warnings.append(
                    f"{len(above)}处行距大于2R，半圆仍可生成，但弓形转弯通常更短"
                )
        elif selected == "pear":
            if above:
                hard_reasons.append(
                    f"{len(above)}处行距ω大于2R（最大{maximum:.2f}m，"
                    f"2R={two_radius:.2f}m），三圆相切梨形几何不存在"
                )
            if near:
                warnings.append(
                    f"{len(near)}处行距接近2R，半圆转弯结构更简单"
                )
        elif selected in ("fishtail", "alpha"):
            if above:
                warnings.append(
                    f"{len(above)}处行距大于2R，倒车折返可生成，但通常没有必要"
                )
        elif selected != "auto":
            hard_reasons.append(f"未知转弯策略: {strategy}")

    return {
        "strategy": ordering_strategy,
        "turn_geometry_strategy": selected,
        "label": TURN_STRATEGY_LABELS.get(ordering_strategy, ordering_strategy),
        "feasible": not hard_reasons,
        "needs_confirmation": bool(hard_reasons),
        "hard_reasons": hard_reasons,
        "warnings": warnings,
        "pass_spacing_min_m": min(spacings) if spacings else 0.0,
        "pass_spacing_max_m": max(spacings) if spacings else 0.0,
        "turn_radius_m": radius,
        "reference": "R与相邻作业线间距ω的典型地头转弯适用关系",
    }


# ═══════════════════════════════════════════════════════════════
#  1. 中心线串联 — 最近邻 + 可选转弯策略
# ═══════════════════════════════════════════════════════════════

def connect_centerlines(bands: List[Dict], main_angle: float,
                        geo=None, state=None,
                        min_turn_radius_m: float = 1.0,
                        turn_strategy: str = "bow") -> List[List[Tuple[float, float]]]:
    """将多条4行带的中心线串联成一条完整路径。

    策略：从一端出发，按投影排序，蛇形连接。
    转弯段使用指定的 turn_strategy 生成几何曲线路径。

    Args:
        bands: 4行带列表（已提取 centerline）
        main_angle: 主方向角度 (rad)
        geo: GeoUtils（用于米/像素比例）
        state: AppState
        min_turn_radius_m: 最小转弯半径（米）
        turn_strategy: 转弯策略名称
            ("bow"/"semicircle"/"pear"/"fishtail")

    Returns:
        路径段列表: [work_0, turn_0, work_1, turn_1, ...]
        奇数索引为转弯段，偶数索引为工作段。
    """
    if not bands:
        return []

    log = AppLogger()

    # 计算每条带的端点和方向
    band_info = []
    for b in bands:
        cl = b.get("centerline", [])
        if len(cl) < 2:
            continue
        start = cl[0]
        end = cl[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dy, dx)
        band_info.append({
            "band": b,
            "centerline": cl,
            "start": start,
            "end": end,
            "angle": angle,
            "centroid": b["centroid"],
        })

    if len(band_info) < 2:
        if band_info:
            return [band_info[0]["centerline"]]
        return []

    # 按主方向投影排序（从一端到另一端）
    dx_main = math.cos(main_angle)
    dy_main = math.sin(main_angle)
    for bi in band_info:
        bi["proj"] = bi["centroid"][0] * dx_main + bi["centroid"][1] * dy_main
    band_info.sort(key=lambda x: x["proj"])

    # 蛇形串联：奇数索引的带翻转方向，使路径自然连接
    work_segments = []
    for idx, bi in enumerate(band_info):
        cl = list(bi["centerline"])
        if idx % 2 == 1:
            cl = list(reversed(cl))
        work_segments.append(cl)

    # 工作方向向量（用于转弯几何计算）
    work_dir = (dx_main, dy_main)

    # 计算米/像素比例
    mpp = 1.0
    if geo and state and geo.is_ready():
        ox = int(getattr(state, "mask_offset_x", 0))
        oy = int(getattr(state, "mask_offset_y", 0))
        # 使用实际掩膜尺寸（优先从 mask_raw 获取，否则用 mask_result）
        h, w = 1000, 1000
        if hasattr(state, 'mask_raw') and state.mask_raw is not None:
            h, w = state.mask_raw.shape[:2]
        elif hasattr(state, 'mask_result') and state.mask_result is not None:
            pm = state.mask_result.get('processed_mask')
            if pm is not None:
                h, w = pm.shape[:2]
        mpp = float(geo.meters_per_pixel(ox + w / 2, oy + h / 2))
        if not math.isfinite(mpp) or not 1e-5 <= mpp <= 5.0:
            raise ValueError(f"invalid metric scale: {mpp!r} m/px")

    # 获取转弯策略函数
    turn_strategy = normalize_turn_strategy(turn_strategy)
    turn_fn = TURN_STRATEGIES.get(turn_strategy, _turn_bow)

    # 构建完整路径: [work_0, turn_0, work_1, turn_1, ..., work_n]
    full_path = []
    for i, work_seg in enumerate(work_segments):
        full_path.append(work_seg)

        if i < len(work_segments) - 1:
            # 生成转弯段连接当前工作段尾部和下一工作段头部
            turn_start = work_seg[-1]
            turn_end = work_segments[i + 1][0]

            # 交替转弯方向（偶数段向右，奇数段向左）
            perp_sign = 1.0 if i % 2 == 0 else -1.0

            turn_pts = turn_fn(turn_start, turn_end, work_dir, perp_sign,
                               mpp, min_turn_radius_m)
            full_path.append(turn_pts)

    log.info(f"串联完成: {len(work_segments)} 条工作段, "
             f"{len(work_segments)-1} 条转弯段, 策略={turn_strategy}")
    return full_path


def _harvester_specs(state=None) -> Dict[str, float]:
    cfg = Config()
    values = getattr(state, "harvester_params", {}) or {}
    cutter_forward_offset = values.get("cutter_forward_offset_m")
    if cutter_forward_offset in (None, ""):
        cutter_forward_offset = cfg.CUTTER_FORWARD_OFFSET_M
    if cutter_forward_offset is None:
        raise ValueError(
            "missing measured cutter_forward_offset_m; "
            "the system will not infer cutter position"
        )
    return {
        "cutter_width_m": float(values.get("cutter_width_m", cfg.CUTTER_WIDTH_M)),
        "cutter_forward_offset_m": float(cutter_forward_offset),
        "track_width_m": float(values.get("track_width_m", cfg.TRACK_WIDTH_M)),
        "track_gauge_m": float(values.get("track_gauge_m", cfg.TRACK_GAUGE_M)),
        "wheelbase_m": float(values.get("wheelbase_m", cfg.WHEELBASE_M)),
        "track_length_m": float(values.get("track_length_m", cfg.TRACK_LENGTH_M)),
        "turn_radius_m": float(values.get("turn_radius_m", cfg.TURN_RADIUS_M)),
    }


def _find_terminal_short_line(
    work_lines: List[List[Tuple[float, float]]],
    main_angle: Optional[float],
    mpp: float,
    field_polygon: Optional[np.ndarray],
    config: Optional[Dict] = None,
) -> Optional[Dict]:
    """Find one field-edge-truncated pass that can safely finish the route.

    The candidate is not extended or deleted.  It is eligible only when its
    missing axial tail is both longer than the configured contour compensation
    and blocked by the selected field boundary.  Putting that pass last leaves
    its real crop-supported endpoint as the route finish and avoids a diagonal
    connector across the unharvested field.
    """
    cfg = dict(config or {})
    if (
        len(work_lines) < 3
        or main_angle is None
        or not math.isfinite(float(main_angle))
        or field_polygon is None
    ):
        return None
    direction = np.asarray(
        (math.cos(float(main_angle)), math.sin(float(main_angle))),
        dtype=np.float64,
    )
    projections = []
    for line in work_lines:
        if len(line) < 2:
            return None
        values = [
            float(np.dot(np.asarray(point, dtype=np.float64), direction))
            for point in (line[0], line[-1])
        ]
        projections.append(values)
    target_low = float(np.median([min(values) for values in projections]))
    target_high = float(np.median([max(values) for values in projections]))
    max_extension_m = float(
        cfg.get(
            "contour_max_endpoint_extension_m",
            cfg.get("endpoint_extension_max_m", 4.0),
        )
    )
    candidates = []
    for line_index, (line, values) in enumerate(zip(work_lines, projections)):
        for endpoint_index, (point, projection) in enumerate(
            zip((line[0], line[-1]), values)
        ):
            low_side = projection <= 0.5 * (values[0] + values[1])
            target_projection = target_low if low_side else target_high
            delta = target_projection - projection
            outward = delta < -1e-9 if low_side else delta > 1e-9
            extension_m = abs(delta) * mpp
            if not outward or extension_m <= max_extension_m + 1e-6:
                continue
            target = (
                float(point[0] + direction[0] * delta),
                float(point[1] + direction[1] * delta),
            )
            if _segment_inside_field_boundary([point, target], field_polygon):
                continue
            candidates.append({
                "original_index": int(line_index),
                "endpoint_index": int(endpoint_index),
                "endpoint": (float(point[0]), float(point[1])),
                "side": "low" if low_side else "high",
                "extension_m": float(extension_m),
                "reason": "long_extension_blocked_by_field_boundary",
            })
    return max(candidates, key=lambda item: item["extension_m"], default=None)


def _align_short_turn_anchors(
    work_lines: List[List[Tuple[float, float]]],
    main_angle: float,
    mpp: float,
    field_polygon: Optional[np.ndarray],
    config: Optional[Dict] = None,
) -> Tuple[List[Dict[str, Tuple[float, float]]], Dict]:
    """Build turn anchors without changing crop-evidence work segments.

    DECISION-011 evaluates each endpoint independently against the cohort.
    The extra distance is emitted later as a non-work turn approach, and
    correction is refused without a verified in-field segment.
    """
    cfg = dict(config or {})
    # DECISION-015: finite semantic-corridor search preserves per-turn attribution.
    anchors = [
        {
            "start": (float(line[0][0]), float(line[0][1])),
            "end": (float(line[-1][0]), float(line[-1][1])),
        }
        for line in work_lines
    ]
    diagnostics = {
        "enabled": True,
        "corrected_endpoint_count": 0,
        "blocked_endpoint_count": 0,
        "corrections": [],
        "blocked": [],
    }
    if len(work_lines) < 3:
        diagnostics["reason"] = "fewer_than_three_lines"
        return anchors, diagnostics

    direction = np.asarray(
        (math.cos(float(main_angle)), math.sin(float(main_angle))),
        dtype=np.float64,
    )
    endpoint_projections = []
    for line in work_lines:
        first = float(np.dot(np.asarray(line[0], dtype=np.float64), direction))
        last = float(np.dot(np.asarray(line[-1], dtype=np.float64), direction))
        endpoint_projections.append((first, last))
    low_values = np.asarray(
        [min(values) for values in endpoint_projections], dtype=np.float64
    )
    high_values = np.asarray(
        [max(values) for values in endpoint_projections], dtype=np.float64
    )
    spans = high_values - low_values
    median_span_px = float(np.median(spans))
    target_low = float(np.median(low_values))
    target_high = float(np.median(high_values))
    low_mad_px = float(np.median(np.abs(low_values - target_low)))
    high_mad_px = float(np.median(np.abs(high_values - target_high)))
    diagnostics["median_span_m"] = median_span_px * mpp
    diagnostics["target_projection_m"] = [target_low * mpp, target_high * mpp]
    diagnostics["direction_rad"] = float(main_angle)

    min_shortfall_m = float(cfg.get("endpoint_shortfall_min_m", 0.50))
    max_extension_m = float(cfg.get("endpoint_extension_max_m", 4.0))
    mad_scale = float(cfg.get("endpoint_outlier_mad_scale", 3.0))
    side_thresholds_m = {
        "low": max(min_shortfall_m, 1.4826 * low_mad_px * mpp * mad_scale),
        "high": max(min_shortfall_m, 1.4826 * high_mad_px * mpp * mad_scale),
    }
    diagnostics["outlier_threshold_m"] = dict(side_thresholds_m)
    for line_index, (line, projections) in enumerate(
        zip(work_lines, endpoint_projections)
    ):
        midpoint = 0.5 * (projections[0] + projections[1])
        for endpoint_name, point, projection in (
            ("start", line[0], projections[0]),
            ("end", line[-1], projections[1]),
        ):
            terminal_unaligned_endpoint = cfg.get(
                "_contour_terminal_unaligned_endpoint_name"
            )
            if terminal_unaligned_endpoint is None and bool(
                cfg.get("_contour_terminal_unaligned_finish", False)
            ):
                terminal_unaligned_endpoint = "end"
            if (
                line_index == len(work_lines) - 1
                and endpoint_name == terminal_unaligned_endpoint
            ):
                continue
            low_side = projection <= midpoint
            target_projection = target_low if low_side else target_high
            delta_px = target_projection - projection
            outward = delta_px < -1e-9 if low_side else delta_px > 1e-9
            extension_m = abs(delta_px) * mpp
            side = "low" if low_side else "high"
            outlier_threshold_m = side_thresholds_m[side]
            if not outward or extension_m < outlier_threshold_m:
                continue
            # 短带通常只是田头断行：若相邻带的同侧端点（按空间侧，而非
            # start/end 名称——蛇形反转后名称可能不对应同侧）比自身更靠近
            # 目标，允许延长到相邻带端点（有作物证据，不受 max_extension_m
            # 限制，仍受田界检查约束）。
            neighbor_same_side = None
            for neighbor_index in (line_index - 1, line_index + 1):
                if not (0 <= neighbor_index < len(work_lines)):
                    continue
                neighbor_values = endpoint_projections[neighbor_index]
                candidate = min(neighbor_values) if low_side else max(neighbor_values)
                if neighbor_same_side is None or (
                    (candidate - projection) * (target_projection - projection)
                    > (neighbor_same_side - projection) * (target_projection - projection)
                ):
                    neighbor_same_side = candidate
            capped_to_neighbor = False
            if neighbor_same_side is not None:
                # 邻居不在目标反方向即可作为证据（等距区细分线端点彼此对齐，
                # 邻居与自身同位置时乘积为 0，同样允许延长到田头线）。
                toward_target = (
                    (neighbor_same_side - projection) * (target_projection - projection)
                    >= 0
                )
                if toward_target:
                    # 延长目标取邻居与中位目标中较远者：有相邻带作物证据时
                    # 不受 max_extension_m 限制（仍受田界检查约束）。等距区
                    # 细分线比相邻带短（区域真实形状），必须延长到相邻带
                    # 同侧端点，田头转弯才能保持短小。
                    if abs(neighbor_same_side - projection) > abs(delta_px):
                        delta_px = neighbor_same_side - projection
                    capped_to_neighbor = True
                    extension_m = abs(delta_px) * mpp
            target = (
                float(point[0] + direction[0] * delta_px),
                float(point[1] + direction[1] * delta_px),
            )
            blocked_reason = ""
            if extension_m > max_extension_m and not capped_to_neighbor:
                blocked_reason = "extension_limit"
            elif field_polygon is None:
                blocked_reason = "missing_field_boundary"
            elif not _segment_inside_field_boundary([point, target], field_polygon):
                blocked_reason = "outside_field_boundary"
            if blocked_reason:
                blocked_reason = (
                    blocked_reason
                    if not capped_to_neighbor
                    else f"{blocked_reason}_neighbor_evidence"
                )
                diagnostics["blocked_endpoint_count"] += 1
                diagnostics["blocked"].append({
                    "line_index": line_index,
                    "endpoint": endpoint_name,
                    "extension_m": extension_m,
                    "outlier_threshold_m": outlier_threshold_m,
                    "reason": blocked_reason,
                })
                continue
            anchors[line_index][endpoint_name] = target
            diagnostics["corrected_endpoint_count"] += 1
            diagnostics["corrections"].append({
                "line_index": line_index,
                "endpoint": endpoint_name,
                "extension_m": extension_m,
                "outlier_threshold_m": outlier_threshold_m,
                "side": side,
                "crop_endpoint": [float(point[0]), float(point[1])],
                "turn_anchor": [target[0], target[1]],
            })
    return anchors, diagnostics


def _headland_space_is_limited(
    start: Tuple[float, float],
    work_direction: Tuple[float, float],
    field_polygon,
    required_depth_px: float,
) -> bool:
    """Pre-screen whether the forward headland probe fits inside the field."""
    if field_polygon is None:
        return False
    depth = max(0.0, float(required_depth_px))
    norm = math.hypot(float(work_direction[0]), float(work_direction[1]))
    if depth <= 0.0 or norm <= 1e-9:
        return False
    target = (
        float(start[0]) + float(work_direction[0]) / norm * depth,
        float(start[1]) + float(work_direction[1]) / norm * depth,
    )
    return not _segment_inside_field_boundary([start, target], field_polygon)


def _build_exit_service_route(
    last_work_line: List[Tuple[float, float]],
    exit_point: Tuple[float, float],
    main_angle: float,
    mpp: float,
    field_polygon,
    config: Optional[Dict] = None,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """Use harvested geometry when the exit lies on the opposite headland."""
    cfg = dict(config or {})
    line = [tuple(map(float, point)) for point in (last_work_line or [])]
    target = tuple(map(float, exit_point))
    report = {
        "algorithm_version": "harvested-exit-route-v1",
        "mode": "direct_same_headland",
        "retrace_last_work": False,
        "direct_axial_distance_m": None,
        "selected_axial_distance_m": None,
        "inside_field_boundary": True,
    }
    if len(line) < 2:
        report["mode"] = "direct_no_work_geometry"
        return ([line[-1], target] if line else [target]), report

    direction = np.asarray(
        [math.cos(float(main_angle)), math.sin(float(main_angle))],
        dtype=np.float64,
    )
    current = np.asarray(line[-1], dtype=np.float64)
    opposite = np.asarray(line[0], dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    current_axial_m = abs(float((target_array - current) @ direction)) * mpp
    opposite_axial_m = abs(float((target_array - opposite) @ direction)) * mpp
    current_distance_m = float(np.linalg.norm(target_array - current) * mpp)
    opposite_distance_m = float(np.linalg.norm(target_array - opposite) * mpp)
    tolerance_m = max(
        1.0,
        float(cfg.get("service_headland_axial_tolerance_m", 3.0)),
    )
    use_opposite = bool(
        current_axial_m > tolerance_m
        and opposite_axial_m + 1e-6 < current_axial_m
        and opposite_distance_m < current_distance_m
    )
    if use_opposite:
        route = list(reversed(line))
        if math.hypot(
            route[-1][0] - target[0], route[-1][1] - target[1]
        ) > 1e-7:
            route.append(target)
        report.update({
            "mode": "retrace_last_harvested_work_then_exit_headland",
            "retrace_last_work": True,
            "selected_axial_distance_m": opposite_axial_m,
            "retrace_distance_m": float(sum(
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(route[:-1], route[1:-1])
            ) * mpp),
        })
    else:
        route = [tuple(current), target]
        report["selected_axial_distance_m"] = current_axial_m
    report.update({
        "direct_axial_distance_m": current_axial_m,
        "current_distance_m": current_distance_m,
        "opposite_endpoint_distance_m": opposite_distance_m,
        "headland_axial_tolerance_m": tolerance_m,
        "inside_field_boundary": bool(
            _segment_inside_field_boundary(route, field_polygon)
        ),
    })
    return route, report


def _build_route_finish_extension(
    last_work_line: List[Tuple[float, float]],
    mpp: float,
    field_polygon,
    *,
    extension_m: float = 0.5,
    target_point: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """Build the final point used by the desktop route and TXT projection.

    When the operator has not supplied an end marker, the finish is a short
    continuation of the final work segment.  A supplied/dragged end marker is
    honoured as the target instead.  The segment is clipped to the confirmed
    field polygon so the generated terminal point remains part of the route
    rather than becoming a display-only label.
    """
    line = [tuple(map(float, point)) for point in (last_work_line or [])]
    report = {
        "algorithm_version": "route-finish-extension-v1",
        "mode": "automatic_final_work_extension",
        "requested_extension_m": float(extension_m),
        "target_point": None,
        "clipped_to_field": False,
        "inside_field_boundary": False,
    }
    if len(line) < 2:
        report["mode"] = "no_work_geometry"
        return line[-1:] if line else [], report
    start = line[-1]
    if target_point is None:
        try:
            extension = max(0.0, float(extension_m)) / max(float(mpp), 1e-9)
        except (TypeError, ValueError):
            extension = 0.0
        direction = np.asarray(line[-1], dtype=np.float64) - np.asarray(
            line[-2], dtype=np.float64
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9 or extension <= 1e-9:
            report["mode"] = "zero_extension"
            report["inside_field_boundary"] = True
            return [start], report
        direction /= norm
        target = (
            float(start[0] + direction[0] * extension),
            float(start[1] + direction[1] * extension),
        )
    else:
        target = (float(target_point[0]), float(target_point[1]))
        report["mode"] = "explicit_dragged_finish"
    report["target_point"] = [float(target[0]), float(target[1])]

    if field_polygon is None or _segment_inside_field_boundary(
        [start, target], field_polygon
    ):
        report["inside_field_boundary"] = True
        return [start, target], report

    # The endpoint is normally inside the field and only the short outward
    # tail crosses a boundary.  Intersect the requested ray with the polygon
    # and retain its farthest forward point.  This avoids inventing an
    # extrapolated coordinate when a field edge truncates the last row.
    try:
        boundary_points = [
            (float(point[0]), float(point[1]))
            for point in np.asarray(field_polygon, dtype=np.float64)
        ]
        field = Polygon(boundary_points)
        ray = LineString([start, target])
        intersection = field.intersection(ray)
        points = []

        def collect(geometry):
            if geometry is None or geometry.is_empty:
                return
            kind = str(getattr(geometry, "geom_type", ""))
            if kind == "Point":
                points.append((float(geometry.x), float(geometry.y)))
                return
            if kind in {"LineString", "LinearRing"}:
                points.extend(
                    (float(x), float(y)) for x, y in geometry.coords
                )
                return
            for child in getattr(geometry, "geoms", ()):
                collect(child)

        collect(intersection)
        vector = np.asarray(target, dtype=np.float64) - np.asarray(
            start, dtype=np.float64
        )
        length_sq = float(vector @ vector)
        candidates = []
        if length_sq > 1e-12:
            for point in points:
                value = np.asarray(point, dtype=np.float64)
                projection = float((value - np.asarray(start)) @ vector)
                if projection >= -1e-7 and projection <= length_sq + 1e-7:
                    candidates.append((projection, point))
        if candidates:
            finish = max(candidates, key=lambda item: item[0])[1]
            finish = (float(finish[0]), float(finish[1]))
            if math.hypot(finish[0] - start[0], finish[1] - start[1]) > 1e-7:
                report["clipped_to_field"] = True
                report["inside_field_boundary"] = _segment_inside_field_boundary(
                    [start, finish], field_polygon
                )
                report["target_point"] = [float(finish[0]), float(finish[1])]
                return [start, finish], report
    except (TypeError, ValueError, IndexError):
        pass
    report["inside_field_boundary"] = False
    return [start, target], report


def _connect_work_lines(
        work_lines: List[List[Tuple[float, float]]],
        geo=None,
        state=None,
        min_turn_radius_m: float = 1.0,
        turn_strategy: str = "bow",
        config: Optional[Dict] = None,
        return_diagnostics: bool = False,
        global_main_angle: Optional[float] = None,
        processed_mask: Optional[np.ndarray] = None,
):
    """Connect ordered coverage passes with direction-aware headland turns."""
    if not work_lines:
        if return_diagnostics:
            diagnostics = assess_turn_strategy(
                turn_strategy,
                [],
                min_turn_radius_m,
            )
            return [], [], diagnostics
        return [], []

    cfg = dict(config or {})
    clearance_m = max(
        float(cfg.get("turn_outward_clearance_m", 0.6)),
        float(cfg.get("turn_safety_margin_m", 0.1)),
    )
    turn_radius_geometry_tolerance_m = max(
        0.0, float(cfg.get("turn_radius_geometry_tolerance_m", 0.05))
    )
    requested_strategy = (
        "auto"
        if str(turn_strategy).strip().lower() == "auto"
        else normalize_turn_strategy(turn_strategy)
    )

    mpp = meters_per_pixel(
        getattr(state, "mask_raw", None)
        if state is not None and getattr(state, "mask_raw", None) is not None
        else np.zeros((2, 2), dtype=np.uint8),
        geo,
        state,
    )
    ox = int(getattr(state, "mask_offset_x", 0)) if state is not None else 0
    oy = int(getattr(state, "mask_offset_y", 0)) if state is not None else 0
    entry_global = getattr(state, "entry_point", None) if state is not None else None
    exit_global = getattr(state, "exit_point", None) if state is not None else None
    unload_globals = []
    if state is not None:
        unload_globals = list(getattr(state, "unload_points", []) or [])
        snapshot_unload = getattr(state, "unload_point", None)
        if snapshot_unload and not unload_globals:
            unload_globals = [snapshot_unload]

    def to_local(point):
        return (float(point[0]) - ox, float(point[1]) - oy) if point else None

    service_point_mode = str(
        cfg.get("service_point_mode", "reference_only")
    ).strip().lower()
    if service_point_mode not in {"reference_only", "connect"}:
        service_point_mode = "reference_only"
    connect_service_points = service_point_mode == "connect"
    include_entry_service = bool(
        connect_service_points and cfg.get("_include_entry_service", True)
    )
    include_exit_service = bool(
        connect_service_points and cfg.get("_include_exit_service", True)
    )
    entry_point = to_local(entry_global)
    if "_route_start_reference_local" in cfg:
        reference = cfg.get("_route_start_reference_local")
        entry_point = (
            (float(reference[0]), float(reference[1]))
            if reference is not None
            else None
        )
    exit_point = to_local(exit_global)
    # A classified-region route receives the user's exit reference only on
    # its final region.  Intermediate regions still use the previous region's
    # endpoint as their entry reference and must not be reordered toward the
    # global exit point.
    if "_route_end_reference_local" in cfg:
        reference = cfg.get("_route_end_reference_local")
        exit_point = (
            (float(reference[0]), float(reference[1]))
            if reference is not None
            else None
        )
    unload_points = [to_local(point) for point in unload_globals if point]
    # DECISION-016: a contour request owns the field-wide half-split order.
    # Legacy/runtime adjacent-snake labels are intentionally ignored.
    terminal_short_line = None
    if (
        requested_strategy == "contour"
        and bool(cfg.get("_contour_allow_terminal_short_line", False))
        and exit_point is None
    ):
        terminal_short_line = _find_terminal_short_line(
            work_lines,
            global_main_angle,
            mpp,
            _field_boundary_local_polygon(state),
            cfg,
        )
    ordered_lines = [list(line) for line in work_lines]
    contour_ordering = {
        "ordering": "snake_row_order",
        "algorithm_version": "contour-order-v1",
        "field_boundary_required": False,
        "fallback_reason": None,
    }
    turn_request_strategy = requested_strategy
    # DECISION-010/016/017: a named curved candidate that reaches the selected
    # field edge remains visible for manual endpoint adjustment.  The boundary
    # validator still fails it closed; this only changes displayed/canonical
    # geometry from a misleading chord to the curve the operator must inspect.
    preserve_boundary_curve = bool(
        cfg.get(
            "_preserve_curve_on_boundary_failure",
            True,
        )
    )
    if requested_strategy == "contour":
        ordered_lines, contour_ordering = _contour_order_work_lines(
            ordered_lines,
            _field_boundary_local_polygon(state),
            main_angle=(
                float(global_main_angle)
                if global_main_angle is not None
                and math.isfinite(float(global_main_angle))
                else None
            ),
        )
        # Wide nested-loop transitions use the two-quarter-arc bow geometry so
        # their transverse tangents overlap on one headland line.  Keep the
        # contour request visible to the selector: a narrow contour transition
        # must not be replaced by the pear/bulb fallback used by ordinary bow
        # requests.
        turn_request_strategy = "contour"
        # The contour candidate is also the machine-facing straight-line
        # topology: retain only the crop-supported endpoints of each already
        # generated parallel pass.  This makes the route's work segments agree
        # with the two-point control/export contract; the other candidate
        # strategies continue to retain their dense canonical geometry.
        ordered_lines = [
            [
                (float(line[0][0]), float(line[0][1])),
                (float(line[-1][0]), float(line[-1][1])),
            ]
            for line in ordered_lines
            if len(line) >= 2
        ]
        contour_ordering["work_line_geometry"] = "straight_endpoint_chord"
        contour_ordering["work_line_point_counts"] = [
            len(line) for line in ordered_lines
        ]
        entry_policy = str(
            cfg.get(
                "_contour_entry_policy",
                "farthest"
                if bool(cfg.get("_contour_reverse_normal_order", False))
                else "nearest",
            )
        )
        ordered_lines = _select_contour_entry_rotation(
            ordered_lines,
            contour_ordering,
            entry_point,
            policy=entry_policy,
            terminal_original_index=(
                terminal_short_line.get("original_index")
                if terminal_short_line is not None
                else None
            ),
            terminal_endpoint=(
                terminal_short_line.get("endpoint")
                if terminal_short_line is not None
                else None
            ),
            main_angle=(
                float(global_main_angle)
                if global_main_angle is not None
                and math.isfinite(float(global_main_angle))
                else contour_ordering.get("main_angle_rad")
            ),
        )
        if terminal_short_line is not None:
            contour_ordering["terminal_short_line"] = {
                key: (list(value) if key == "endpoint" else value)
                for key, value in terminal_short_line.items()
            }
    if entry_point and len(ordered_lines) >= 2:
        first_distance = min(
            math.hypot(point[0] - entry_point[0], point[1] - entry_point[1])
            for point in (ordered_lines[0][0], ordered_lines[0][-1])
        )
        if requested_strategy != "contour":
            last_distance = min(
                math.hypot(point[0] - entry_point[0], point[1] - entry_point[1])
                for point in (ordered_lines[-1][0], ordered_lines[-1][-1])
            )
            if last_distance < first_distance:
                ordered_lines.reverse()

    alignment_angle = global_main_angle
    alignment_direction_source = "global_main_angle"
    if alignment_angle is None or not math.isfinite(float(alignment_angle)):
        contour_angle = contour_ordering.get("main_angle_rad")
        if contour_angle is not None and math.isfinite(float(contour_angle)):
            alignment_angle = float(contour_angle)
            alignment_direction_source = "contour_order_axial_mean"
        else:
            weighted_cos = 0.0
            weighted_sin = 0.0
            for line in ordered_lines:
                if len(line) < 2:
                    continue
                dx = float(line[-1][0] - line[0][0])
                dy = float(line[-1][1] - line[0][1])
                weight = math.hypot(dx, dy)
                if weight <= 1e-9:
                    continue
                angle = math.atan2(dy, dx)
                weighted_cos += weight * math.cos(2.0 * angle)
                weighted_sin += weight * math.sin(2.0 * angle)
            alignment_angle = 0.5 * math.atan2(weighted_sin, weighted_cos)
            alignment_direction_source = "all_work_lines_axial_mean"

    has_internal_handoff_reference = "_route_start_reference_local" in cfg
    allow_reference_ordering_with_handoff = bool(
        cfg.get("_allow_reference_ordering_with_handoff", False)
    )
    reference_ordering = {
        # ``_route_start_reference_local`` is an internal handoff from a
        # preceding classified region, not a second user choice.  Region
        # reversal is handled by _connect_work_regions so that its adjacent
        # transition check remains authoritative.
        "enabled": bool(
            (entry_point is not None or exit_point is not None)
            and (
                not has_internal_handoff_reference
                or (
                    allow_reference_ordering_with_handoff
                    and exit_point is not None
                )
            )
        ),
        "candidate_count": 1,
        "selected_variant": "current",
        "entry_reference_used": bool(entry_point is not None),
        "exit_reference_used": bool(exit_point is not None),
    }

    if terminal_short_line is not None:
        contour_ordering["terminal_finish_orientation_forced"] = False

    def _axially_normalized_line(line):
        points = list(line)
        if requested_strategy == "contour" and len(points) >= 2:
            direction = (
                math.cos(float(alignment_angle)),
                math.sin(float(alignment_angle)),
            )
            axial_delta = (
                (float(points[-1][0]) - float(points[0][0])) * direction[0]
                + (float(points[-1][1]) - float(points[0][1])) * direction[1]
            )
            if axial_delta < 0.0:
                points.reverse()
        return points

    def _entry_start_reversed(lines):
        if entry_point is None or not lines or len(lines[0]) < 2:
            return False
        # Upstream centreline endpoint order is geometric, not an execution
        # direction.  Compare the confirmed entry only after normalizing the
        # first line to the common axial direction; otherwise the subsequent
        # normalization silently flips the chosen entry side.
        first = _axially_normalized_line(lines[0])
        distance_start = math.hypot(
            first[0][0] - entry_point[0],
            first[0][1] - entry_point[1],
        )
        distance_end = math.hypot(
            first[-1][0] - entry_point[0],
            first[-1][1] - entry_point[1],
        )
        return distance_end < distance_start

    def _preview_oriented(lines):
        preview = []
        start_reverse = _entry_start_reversed(lines)
        for index, line in enumerate(lines):
            points = _axially_normalized_line(line)
            if bool(index % 2) ^ start_reverse:
                points.reverse()
            if len(points) >= 2:
                preview.append(points)
        return preview

    selected_preview = None
    if reference_ordering["enabled"] and len(ordered_lines) >= 2:
        candidate_specs = []
        current_indices = list(
            contour_ordering.get("ordered_original_indices") or []
        )
        current_positions = list(contour_ordering.get("normal_positions") or [])
        if len(current_indices) != len(ordered_lines):
            current_indices = list(range(len(ordered_lines)))
        if len(current_positions) != len(ordered_lines):
            current_positions = [None] * len(ordered_lines)
        candidate_specs.append((
            "current", [list(line) for line in ordered_lines],
            current_indices, current_positions,
        ))
        if requested_strategy != "contour":
            base_specs = list(candidate_specs)
            for name, lines, indices, positions in base_specs:
                candidate_specs.append((
                    f"{name}_reversed", list(reversed(lines)),
                    list(reversed(indices)), list(reversed(positions)),
                ))

        scored = []
        for candidate_index, (name, lines, indices, positions) in enumerate(
            candidate_specs
        ):
            preview = _preview_oriented(lines)
            if not preview:
                continue
            score = 0.0
            entry_distance = None
            exit_distance = None
            if entry_point is not None:
                # The route enters at the first point of the first oriented
                # line.  Taking the minimum endpoint distance here can score
                # a candidate whose *unvisited* endpoint is close while the
                # actual route start is far away.
                entry_distance = math.hypot(
                    preview[0][0][0] - entry_point[0],
                    preview[0][0][1] - entry_point[1],
                )
                score += entry_distance
            if exit_point is not None:
                # Likewise the route leaves at the final point of the final
                # oriented line.  This is important when entry and exit are
                # on the same field side.
                exit_distance = math.hypot(
                    preview[-1][-1][0] - exit_point[0],
                    preview[-1][-1][1] - exit_point[1],
                )
                score += exit_distance
            scored.append((score, candidate_index, name, lines, indices, positions,
                           entry_distance, exit_distance, preview))
        if scored:
            if entry_point is not None:
                # The user-selected start row is authoritative.  Exit-side
                # proximity may break ties, but must not move the route away
                # from the row nearest the start marker.
                selected = min(
                    scored,
                    key=lambda item: (
                        float("inf") if item[6] is None else item[6],
                        item[0],
                        item[1],
                    ),
                )
            else:
                selected = min(scored, key=lambda item: (item[0], item[1]))
            (
                score, candidate_index, name, selected_lines, selected_indices,
                selected_positions, entry_distance, exit_distance,
                selected_preview,
            ) = selected
            ordered_lines = selected_lines
            reference_ordering.update({
                "candidate_count": len(candidate_specs),
                "selected_variant": name,
                "selected_score_px": float(score),
                "selected_entry_distance_m": (
                    None if entry_distance is None else float(entry_distance * mpp)
                ),
                "selected_exit_distance_m": (
                    None if exit_distance is None else float(exit_distance * mpp)
                ),
            })
            if requested_strategy == "contour":
                contour_ordering["ordered_original_indices"] = selected_indices
                contour_ordering["normal_positions"] = selected_positions
                contour_ordering["corresponding_pairs"] = [
                    {
                        "pair_index": pair_index,
                        "original_indices": selected_indices[index:index + 2],
                        "normal_positions": selected_positions[index:index + 2],
                    }
                    for pair_index, index in enumerate(
                        range(0, len(selected_indices), 2)
                    )
                ]
                contour_ordering["reference_order_variant"] = name

    if requested_strategy == "contour":
        _refresh_contour_order_diagnostics(
            contour_ordering,
            list(contour_ordering.get("ordered_original_indices") or []),
            list(contour_ordering.get("normal_positions") or []),
        )

    if selected_preview is not None:
        # Candidate scoring already produced the exact oriented route.  Do
        # not orient it a second time using only the entry point: that used to
        # undo the exit-side choice for same-side start/end markers.
        oriented = [list(line) for line in selected_preview]
    else:
        start_reversed = _entry_start_reversed(ordered_lines)

        oriented = []
        for index, line in enumerate(ordered_lines):
            points = _axially_normalized_line(line)
            if bool(index % 2) ^ start_reversed:
                points.reverse()
            if len(points) >= 2:
                oriented.append(points)

    if (
        requested_strategy == "contour"
        and terminal_short_line is not None
        and contour_ordering.get("terminal_constraint_applied")
        and oriented
    ):
        terminal_endpoint = np.asarray(
            terminal_short_line["endpoint"], dtype=np.float64
        )
        final_start = np.asarray(oriented[-1][0], dtype=np.float64)
        final_end = np.asarray(oriented[-1][-1], dtype=np.float64)
        terminal_endpoint_role = (
            "start"
            if float(np.linalg.norm(final_start - terminal_endpoint))
            <= float(np.linalg.norm(final_end - terminal_endpoint))
            else "end"
        )
        cfg["_contour_terminal_unaligned_endpoint_name"] = (
            terminal_endpoint_role
        )
        contour_ordering.update({
            "terminal_blocked_endpoint_role": terminal_endpoint_role,
            "terminal_blocked_endpoint_is_route_finish": bool(
                terminal_endpoint_role == "end"
            ),
            "terminal_local_incoming_turn_required": bool(
                terminal_endpoint_role == "start"
            ),
            "terminal_finish_orientation_forced": False,
        })

    full_path = []
    segment_types = []
    pass_spacings_m = []
    used_strategies = []
    turn_decisions = []
    fallback_reasons = []
    boundary_adjustments = []
    boundary_hard_reasons = []
    display_only_connectors = []
    exit_service_route = {
        "algorithm_version": "harvested-exit-route-v1",
        "mode": "not_requested",
    }
    field_polygon = _field_boundary_local_polygon(state)
    turn_anchors, endpoint_alignment = _align_short_turn_anchors(
        oriented,
        main_angle=float(alignment_angle),
        mpp=mpp,
        field_polygon=field_polygon,
        config=cfg,
    )
    endpoint_alignment["direction_source"] = alignment_direction_source

    contour_connector_anchors = []
    contour_connector_diagnostics = {
        "enabled": requested_strategy == "contour",
        "fallback_reason": "not_requested",
    }
    if requested_strategy == "contour":
        contour_seed_lines = []
        for line_index, line in enumerate(oriented):
            seed = list(line)
            if line_index < len(turn_anchors):
                seed[0] = turn_anchors[line_index]["start"]
                seed[-1] = turn_anchors[line_index]["end"]
            contour_seed_lines.append(seed)
        contour_connector_anchors, contour_connector_diagnostics = (
            _build_contour_connector_anchors(
                contour_seed_lines,
                float(alignment_angle),
                field_polygon,
                mpp,
                config=cfg,
            )
        )
        if (
            not contour_connector_anchors
            and contour_connector_diagnostics.get("fallback_reason")
            == "headland_wall_extension_leaves_boundary"
            and float(cfg.get("work_line_endpoint_extension_m", 0.0) or 0.0)
            > 0.0
        ):
            # The optional outward straight must yield to the selected field
            # boundary.  Retry on the crop-supported shared wall itself; this
            # keeps every actual turn aligned and avoids degrading the route
            # to a long diagonal merely because the extra 0.5 m is unavailable
            # at one notched boundary endpoint.
            retry_cfg = dict(cfg)
            original_extension_m = float(
                retry_cfg.get("work_line_endpoint_extension_m", 0.0) or 0.0
            )
            retry_cfg["work_line_endpoint_extension_m"] = 0.0
            retry_anchors, retry_diagnostics = _build_contour_connector_anchors(
                contour_seed_lines,
                float(alignment_angle),
                field_polygon,
                mpp,
                config=retry_cfg,
            )
            if retry_anchors:
                contour_connector_anchors = retry_anchors
                contour_connector_diagnostics = retry_diagnostics
                contour_connector_diagnostics.update({
                    "outward_endpoint_extension_reduced": True,
                    "outward_endpoint_extension_requested_m": (
                        original_extension_m
                    ),
                    "outward_endpoint_extension_applied_m": 0.0,
                    "outward_endpoint_extension_reduction_reason": (
                        "selected_field_boundary"
                    ),
                })
        if contour_connector_anchors:
            # A boundary-supported contour owns its endpoint geometry.  Do
            # not subsequently replace these common pair anchors with the
            # cohort/MAD short-line anchors; doing so recreates the long chord.
            turn_anchors = contour_connector_anchors
            endpoint_alignment["contour_boundary_anchors_applied"] = True
        else:
            endpoint_alignment["contour_boundary_anchors_applied"] = False
            contour_connector_diagnostics["fallback_only"] = True

    contour_route_mode = bool(contour_connector_anchors)
    contour_fallback_reason = contour_connector_diagnostics.get("fallback_reason")
    if requested_strategy == "contour" and not contour_route_mode:
        # A failed contour-geometry attempt is explicit: do not silently
        # present the ordinary U-turn path as a successful 套圈 route.
        fallback_reasons.append(
            f"contour_geometry_unavailable:{contour_fallback_reason or 'unknown'}"
        )
        boundary_hard_reasons.append(
            f"contour geometry unavailable: {contour_fallback_reason or 'unknown'}"
        )

    # 锚点延长合并进作业线本身：锚点沿主方向延长（同法向、叉积小）时，
    # 直接把作业线端点延伸到锚点位置。否则转弯段会包含数十米的
    # turn_approach 空驶（用户视角的"长转弯"），且直线看起来没有走完；
    # 合并后作业线完整延伸到田头，转弯保持短小。侧面调整（法向偏移大）
    # 仍保留 approach 段。
    if oriented and turn_anchors:
        direction_x = math.cos(float(alignment_angle))
        direction_y = math.sin(float(alignment_angle))
        merge_side_tolerance_px = 0.3 / max(mpp, 1e-9)
        max_merge_distance_px = 40.0 / max(mpp, 1e-9)
        for line_index, line in enumerate(oriented):
            if len(line) < 2 or line_index >= len(turn_anchors):
                continue
            for endpoint_name, point_index in (("start", 0), ("end", -1)):
                anchor = turn_anchors[line_index].get(endpoint_name)
                if anchor is None:
                    continue
                point = line[point_index]
                dx = anchor[0] - point[0]
                dy = anchor[1] - point[1]
                distance = math.hypot(dx, dy)
                if distance < 1e-6 or distance > max_merge_distance_px:
                    continue
                # 锚点必须沿主方向延长（叉积小）；偏离超过 0.3m 视为侧面调整
                if abs(dx * direction_y - dy * direction_x) > merge_side_tolerance_px:
                    continue
                line[point_index] = anchor
        endpoint_alignment["long_anchor_extensions_merged_into_work_lines"] = True

    # DECISION-016/017: audit the entry against the geometry that will actually
    # be emitted.  Ordering happens before short-end repair and shared-headland
    # projection, so the pre-alignment endpoint is not sufficient evidence that
    # the final canonical route still starts at the nearest feasible endpoint.
    if requested_strategy == "contour" and entry_point is not None and oriented:
        before_endpoint = contour_ordering.get("entry_selected_endpoint_local")
        before_distance = contour_ordering.get("entry_selected_line_distance_px")
        if before_endpoint is not None:
            contour_ordering[
                "entry_selected_endpoint_before_headland_alignment_local"
            ] = list(before_endpoint)
        if before_distance is not None:
            contour_ordering[
                "entry_selected_line_distance_before_headland_alignment_px"
            ] = float(before_distance)

        actual_start = (
            float(oriented[0][0][0]),
            float(oriented[0][0][1]),
        )
        actual_distance_px = math.hypot(
            actual_start[0] - entry_point[0],
            actual_start[1] - entry_point[1],
        )
        terminal_constrained = bool(
            contour_ordering.get("entry_orientation_constrained_by_terminal")
        )
        direction = np.asarray((
            math.cos(float(alignment_angle)),
            math.sin(float(alignment_angle)),
        ), dtype=np.float64)
        actual_projection = float(np.dot(np.asarray(actual_start), direction))
        first_line_projections = [
            float(np.dot(np.asarray(point, dtype=np.float64), direction))
            for point in (oriented[0][0], oriented[0][-1])
        ]
        actual_side = (
            "low"
            if actual_projection <= min(first_line_projections) + 1e-7
            else "high"
        )
        terminal_original_index = contour_ordering.get("terminal_original_index")
        ordered_original_indices = list(
            contour_ordering.get("ordered_original_indices") or []
        )
        eligible = []
        unconstrained = []
        for line_index, line in enumerate(oriented):
            if len(line) < 2:
                continue
            original_index = (
                ordered_original_indices[line_index]
                if line_index < len(ordered_original_indices)
                else line_index
            )
            if (
                terminal_original_index is not None
                and original_index == terminal_original_index
            ):
                continue
            endpoints = [
                (float(line[0][0]), float(line[0][1])),
                (float(line[-1][0]), float(line[-1][1])),
            ]
            unconstrained.extend(endpoints)
            if terminal_constrained:
                projections = [
                    float(np.dot(np.asarray(point), direction))
                    for point in endpoints
                ]
                endpoint = endpoints[
                    int(np.argmin(projections))
                    if actual_side == "low"
                    else int(np.argmax(projections))
                ]
                eligible.append(endpoint)
            else:
                eligible.extend(endpoints)

        def _entry_distance(point):
            return math.hypot(
                point[0] - entry_point[0], point[1] - entry_point[1]
            )

        nearest_feasible = min(eligible, key=_entry_distance, default=actual_start)
        nearest_feasible_distance_px = _entry_distance(nearest_feasible)
        nearest_unconstrained = min(
            unconstrained, key=_entry_distance, default=actual_start
        )
        nearest_unconstrained_distance_px = _entry_distance(nearest_unconstrained)
        nearest_feasible_verified = bool(
            actual_distance_px <= nearest_feasible_distance_px + 1e-6
        )
        contour_ordering.update({
            "entry_selected_endpoint_local": list(actual_start),
            "entry_selected_line_distance_px": float(actual_distance_px),
            "entry_selected_line_distance_m": float(actual_distance_px * mpp),
            "entry_nearest_feasible_endpoint_local": list(nearest_feasible),
            "entry_nearest_feasible_distance_px": float(
                nearest_feasible_distance_px
            ),
            "entry_nearest_feasible_distance_m": float(
                nearest_feasible_distance_px * mpp
            ),
            "entry_nearest_feasible_verified": nearest_feasible_verified,
            "entry_nearest_unconstrained_endpoint_local": list(
                nearest_unconstrained
            ),
            "entry_nearest_unconstrained_distance_m": float(
                nearest_unconstrained_distance_px * mpp
            ),
            "entry_side_constraint": actual_side if terminal_constrained else None,
            "entry_side_constrained_by_terminal_short_line": terminal_constrained,
            "entry_unconstrained_endpoint_blocked_by_terminal_constraint": bool(
                terminal_constrained
                and nearest_unconstrained_distance_px + 1e-6
                < nearest_feasible_distance_px
            ),
        })
        if not nearest_feasible_verified:
            boundary_hard_reasons.append(
                "final route does not start at the nearest feasible entry endpoint"
            )

    endpoint_alignment["path_reference_shift_applied"] = False
    turn_validation_cfg = dict(cfg)

    contour_shared_clearance = {"low": clearance_m, "high": clearance_m}
    contour_clearance_records = []
    if requested_strategy == "contour" and len(oriented) >= 2:
        axial = np.asarray((
            math.cos(float(alignment_angle)),
            math.sin(float(alignment_angle)),
        ), dtype=np.float64)
        clearance_candidates = []
        for scale in (1.0, 0.5, 0.25, 0.0):
            value = max(0.0, float(clearance_m) * scale)
            if value not in clearance_candidates:
                clearance_candidates.append(value)
        safe_by_side = {"low": [], "high": []}
        for transition_index in range(len(oriented) - 1):
            line = oriented[transition_index]
            start = turn_anchors[transition_index]["end"]
            end = turn_anchors[transition_index + 1]["start"]
            direction = np.asarray(start, dtype=np.float64) - np.asarray(
                line[-2], dtype=np.float64
            )
            if float(np.dot(axial, direction)) < 0.0:
                direction = -axial
            else:
                direction = axial.copy()
            work_direction = (float(direction[0]), float(direction[1]))
            side = "high" if float(np.dot(direction, axial)) >= 0.0 else "low"
            lateral = np.asarray(end, dtype=np.float64) - np.asarray(
                start, dtype=np.float64
            )
            gap_m = abs(float(
                direction[0] * lateral[1] - direction[1] * lateral[0]
            )) * mpp
            if gap_m <= 1e-9:
                gap_m = float(np.linalg.norm(lateral) * mpp)
            decision = _select_turn_strategy(
                "contour",
                gap_m,
                min_turn_radius_m,
                geometry_tolerance_m=turn_radius_geometry_tolerance_m,
            )
            strategy = (
                "bow" if str(decision["strategy"]) == "contour"
                else str(decision["strategy"])
            )
            function = TURN_STRATEGIES.get(strategy, _turn_bow)
            geometry_radius_m = min(
                float(min_turn_radius_m), max(1e-6, gap_m * 0.5)
            ) if str(decision["strategy"]) == "contour" else float(min_turn_radius_m)
            selected_clearance = 0.0
            for candidate_clearance in clearance_candidates:
                if strategy == "semicircle":
                    candidate = function(
                        start, end, work_direction, 1.0, mpp,
                        geometry_radius_m, candidate_clearance,
                        radius_tolerance_m=turn_radius_geometry_tolerance_m,
                        tangent_radius_m=min_turn_radius_m,
                    )
                else:
                    candidate = function(
                        start, end, work_direction, 1.0, mpp,
                        geometry_radius_m, candidate_clearance,
                    )
                if _segment_inside_field_boundary(candidate, field_polygon):
                    selected_clearance = candidate_clearance
                    break
            safe_by_side[side].append(selected_clearance)
            contour_clearance_records.append({
                "transition_index": int(transition_index),
                "headland_side": side,
                "individually_safe_clearance_m": float(selected_clearance),
                "strategy": strategy,
                "geometry_radius_m": float(geometry_radius_m),
            })
        for side in ("low", "high"):
            if safe_by_side[side]:
                contour_shared_clearance[side] = float(min(safe_by_side[side]))
        forced_clearances = cfg.get(
            "_contour_forced_turn_clearance_m_by_side"
        )
        clearance_scope = "current_region"
        if isinstance(forced_clearances, dict):
            for side in ("low", "high"):
                forced_value = _finite_float(forced_clearances.get(side))
                if forced_value is not None and forced_value >= 0.0:
                    contour_shared_clearance[side] = min(
                        float(contour_shared_clearance[side]),
                        float(forced_value),
                    )
            clearance_scope = "all_classified_regions"
        contour_connector_diagnostics["shared_turn_clearance_m_by_side"] = dict(
            contour_shared_clearance
        )
        contour_connector_diagnostics["shared_turn_clearance_scope"] = (
            clearance_scope
        )
        contour_connector_diagnostics["turn_clearance_records"] = (
            contour_clearance_records
        )

    if include_entry_service and entry_point and oriented:
        full_path.append([entry_point, oriented[0][0]])
        segment_types.append("entry")
    for index, line in enumerate(oriented):
        full_path.append(line)
        segment_types.append("work")
        if index >= len(oriented) - 1:
            continue
        next_line = oriented[index + 1]
        start = turn_anchors[index]["end"]
        end = turn_anchors[index + 1]["start"]
        if math.hypot(start[0] - line[-1][0], start[1] - line[-1][1]) > 1e-7:
            full_path.append([line[-1], start])
            segment_types.append("turn_approach")
        direction_x = start[0] - line[-2][0]
        direction_y = start[1] - line[-2][1]
        direction_norm = math.hypot(direction_x, direction_y)
        if direction_norm <= 1e-9:
            work_direction = (math.cos(0.0), math.sin(0.0))
        else:
            work_direction = (
                direction_x / direction_norm,
                direction_y / direction_norm,
            )
        if requested_strategy == "contour":
            # DECISION-016: every nested turn uses the same audited main-line
            # axis.  A short/noisy local centreline tail must not rotate one
            # bow and destroy the shared, overlapping headland tangent.
            axial = np.asarray((
                math.cos(float(alignment_angle)),
                math.sin(float(alignment_angle)),
            ), dtype=np.float64)
            if float(np.dot(axial, (direction_x, direction_y))) < 0.0:
                axial *= -1.0
            work_direction = (float(axial[0]), float(axial[1]))
        alignment_axis = np.asarray((
            math.cos(float(alignment_angle)),
            math.sin(float(alignment_angle)),
        ), dtype=np.float64)
        headland_side = (
            "high"
            if float(np.dot(alignment_axis, work_direction)) >= 0.0
            else "low"
        )
        turn_clearance_m = (
            float(contour_shared_clearance[headland_side])
            if requested_strategy == "contour"
            else float(clearance_m)
        )
        lateral_x = end[0] - start[0]
        lateral_y = end[1] - start[1]
        cross = work_direction[0] * lateral_y - work_direction[1] * lateral_x
        turn_sign = 1.0 if cross >= 0 else -1.0
        gap_m = abs(
            work_direction[0] * lateral_y - work_direction[1] * lateral_x
        ) * mpp
        if gap_m <= 1e-9:
            gap_m = math.hypot(lateral_x, lateral_y) * mpp
        pass_spacings_m.append(gap_m)
        headland_limited = _headland_space_is_limited(
            start,
            work_direction,
            field_polygon,
            required_depth_px=(2.0 * min_turn_radius_m + turn_clearance_m)
            / max(mpp, 1e-9),
        )
        decision = _select_turn_strategy(
            turn_request_strategy,
            gap_m,
            min_turn_radius_m,
            headland_limited=headland_limited,
            geometry_tolerance_m=turn_radius_geometry_tolerance_m,
        )
        selected_strategy = (
            "bow" if str(decision["strategy"]) == "contour"
            else decision["strategy"]
        )
        geometry_radius_m = (
            min(float(min_turn_radius_m), max(1e-6, gap_m * 0.5))
            if requested_strategy == "contour"
            else float(min_turn_radius_m)
        )
        turn_decisions.append({
            "transition_index": index,
            "requested_strategy": decision.get("requested_strategy"),
            "candidate_strategy": decision.get("candidate_strategy"),
            "effective_strategy": selected_strategy,
            "strategy": selected_strategy,
            "pass_spacing_m": float(gap_m),
            "turn_radius_m": float(min_turn_radius_m),
            "geometry_radius_m": float(geometry_radius_m),
            "radius_capped_for_contour_pairing": bool(
                requested_strategy == "contour"
                and geometry_radius_m < float(min_turn_radius_m) - 1e-9
            ),
            "geometry_tolerance_m": float(
                decision.get("geometry_tolerance_m", 0.0)
            ),
            "radius_shortfall_m": float(
                decision.get("radius_shortfall_m", 0.0)
            ),
            "shared_headland_side": headland_side,
            "shared_clearance_m": turn_clearance_m,
            "reason": decision.get("reason", ""),
            "fallback_reason": decision.get("fallback_reason", ""),
            "geometry_signature": decision.get("geometry_signature", ""),
            "crop_exit": [float(line[-1][0]), float(line[-1][1])],
            "turn_start": [float(start[0]), float(start[1])],
            "turn_end": [float(end[0]), float(end[1])],
            "crop_entry": [float(next_line[0][0]), float(next_line[0][1])],
        })
        if requested_strategy == "contour":
            turn_decisions[-1]["ordering_strategy"] = "contour"
            turn_decisions[-1]["turn_request_strategy"] = "contour"
        if decision.get("fallback_reason"):
            fallback_reasons.append(decision["fallback_reason"])
        used_strategies.append(selected_strategy)

        if selected_strategy in ("fishtail", "alpha"):
            turn_parts = _turn_reverse_segments(
                start,
                end,
                work_direction,
                mpp,
                min_turn_radius_m,
                turn_clearance_m,
                inner=(selected_strategy == "alpha"),
            )
            base_turn_parts = list(turn_parts)
            turn_parts, turn_validation = _validate_fixed_headland_turn(
                base_turn_parts, start, end, work_direction, turn_sign,
                selected_strategy,
                mpp=mpp,
                min_turn_radius_m=min_turn_radius_m,
                clearance_m=turn_clearance_m,
                processed_mask=processed_mask,
                field_polygon=field_polygon,
                geo=geo,
                state=state,
                config=turn_validation_cfg,
                validation_work_lines=oriented,
                validation_turn_index=index,
            )
            turn_decisions[-1]["headland_turn_validation"] = turn_validation
            reverse_leaves_boundary = (
                field_polygon is not None
                and any(
                    not _segment_inside_field_boundary(turn_part, field_polygon)
                    for turn_part, _turn_type in turn_parts
                )
            )
            if reverse_leaves_boundary:
                direct = [start, end]
                if _segment_inside_field_boundary(direct, field_polygon):
                    connector_index = len(full_path)
                    full_path.append(direct)
                    segment_types.append("turn")
                    reason = (
                        "reverse turn geometry degraded to a straight segment because no "
                        "kinematically valid reverse manoeuvre fit inside the field boundary"
                    )
                    boundary_adjustments.append("倒车折返会越出框选田块，已退化为边界内端点连接")
                    boundary_hard_reasons.append(reason)
                    connector = {
                        "segment_index": connector_index,
                        "transition_index": index,
                        "display_only_connector": True,
                        "kinematically_infeasible": True,
                        "reason": reason,
                    }
                    display_only_connectors.append(connector)
                    turn_decisions[-1].update(connector)
                else:
                    boundary_adjustments.append(
                        "倒车折返仍可能越出框选田块，请调整边界/起终点或手动编辑路线"
                    )
                    for turn_part, turn_type in turn_parts:
                        full_path.append(turn_part)
                        segment_types.append(turn_type)
            else:
                for turn_part, turn_type in turn_parts:
                    full_path.append(turn_part)
                    segment_types.append(turn_type)
        else:
            turn, _turn_strategy, boundary_reason = _adjust_turn_to_field_boundary(
                start,
                end,
                work_direction,
                turn_sign,
                mpp,
                geometry_radius_m,
                selected_strategy,
                turn_clearance_m,
                field_polygon,
                semicircle_radius_tolerance_m=(
                    turn_radius_geometry_tolerance_m
                    if selected_strategy == "semicircle"
                    else 0.0
                ),
                semicircle_tangent_radius_m=(
                    min_turn_radius_m
                    if requested_strategy == "contour"
                    and selected_strategy == "semicircle"
                    else None
                ),
                preserve_curve_on_boundary_failure=preserve_boundary_curve,
            )
            base_turn_parts = [(turn, "turn")]
            turn_parts, turn_validation = _validate_fixed_headland_turn(
                base_turn_parts, start, end, work_direction, turn_sign,
                selected_strategy,
                mpp=mpp,
                min_turn_radius_m=min_turn_radius_m,
                clearance_m=turn_clearance_m,
                processed_mask=processed_mask,
                field_polygon=field_polygon,
                geo=geo,
                state=state,
                config=turn_validation_cfg,
                validation_work_lines=oriented,
                validation_turn_index=index,
            )
            turn_decisions[-1]["headland_turn_validation"] = turn_validation
            if boundary_reason:
                boundary_adjustments.append(boundary_reason)
                if len(turn) > 2 and preserve_boundary_curve:
                    turn_decisions[-1][
                        "boundary_curve_preserved_for_review"
                    ] = True
                if len(turn) <= 2:
                    reason = (
                        "turn geometry degraded to a straight segment because no "
                        "kinematically valid curve fit inside the field boundary"
                    )
                    boundary_hard_reasons.append(reason)
                    connector = {
                        "segment_index": len(full_path),
                        "transition_index": index,
                        "display_only_connector": True,
                        "kinematically_infeasible": True,
                        "reason": reason,
                    }
                    display_only_connectors.append(connector)
                    turn_decisions[-1].update(connector)
            full_path.append(turn)
            segment_types.append("turn")
        if math.hypot(
            end[0] - next_line[0][0],
            end[1] - next_line[0][1],
        ) > 1e-7:
            full_path.append([end, next_line[0]])
            segment_types.append("turn_approach")

    if include_exit_service and exit_point and oriented:
        route_end = oriented[-1][-1]
        remaining_unloads = list(unload_points)
        if remaining_unloads:
            nearest_index = min(
                range(len(remaining_unloads)),
                key=lambda index: math.hypot(
                    remaining_unloads[index][0] - route_end[0],
                    remaining_unloads[index][1] - route_end[1],
                ),
            )
            unload_point = remaining_unloads[nearest_index]
            if math.hypot(unload_point[0] - route_end[0], unload_point[1] - route_end[1]) > 1e-6:
                full_path.append([route_end, unload_point])
                segment_types.append("unload")
                route_end = unload_point
        if math.hypot(exit_point[0] - route_end[0], exit_point[1] - route_end[1]) > 1e-6:
            if remaining_unloads:
                exit_route = [route_end, exit_point]
                exit_service_route = {
                    "algorithm_version": "harvested-exit-route-v1",
                    "mode": "direct_from_explicit_unload_point",
                    "inside_field_boundary": bool(
                        _segment_inside_field_boundary(
                            [route_end, exit_point], field_polygon
                        )
                    ),
                }
            else:
                exit_route, exit_service_route = _build_exit_service_route(
                    oriented[-1],
                    exit_point,
                    float(alignment_angle),
                    mpp,
                    field_polygon,
                    config=cfg,
                )
            if exit_service_route.get("retrace_last_work") and len(exit_route) >= 3:
                full_path.append(exit_route[:-1])
                segment_types.append("turn_reverse")
                full_path.append([exit_route[-2], exit_route[-1]])
            else:
                full_path.append(exit_route)
            segment_types.append("exit")
    elif (
        bool(cfg.get("include_route_finish_extension", False))
        and bool(cfg.get("_include_exit_service", True))
        and oriented
    ):
        # Desktop planning keeps service points as references, but its
        # terminal TXT route still needs a concrete final point.  Generate a
        # short final-work extension by default, or use an explicitly dragged
        # end marker as the target.
        try:
            finish_extension_m = float(
                cfg.get(
                    "route_finish_extension_m",
                    cfg.get("work_line_endpoint_extension_m", 0.5),
                )
            )
        except (TypeError, ValueError):
            finish_extension_m = 0.5
        finish_route, exit_service_route = _build_route_finish_extension(
            oriented[-1],
            mpp,
            field_polygon,
            extension_m=finish_extension_m,
            target_point=exit_point,
        )
        if len(finish_route) >= 2:
            full_path.append(finish_route)
            segment_types.append("exit")
    non_work_boundary_violations = _non_work_boundary_violations(
        full_path,
        segment_types,
        field_polygon,
    )
    for violation in non_work_boundary_violations:
        boundary_hard_reasons.append(
            f"{violation['segment_type']} segment leaves field boundary"
        )
    assessment_strategy = (
        requested_strategy
        if not fallback_reasons
        else (used_strategies[0] if len(set(used_strategies)) == 1 else "auto")
    )
    assessment = assess_turn_strategy(
        assessment_strategy,
        pass_spacings_m,
        min_turn_radius_m,
    )
    if boundary_hard_reasons:
        assessment.setdefault("hard_reasons", []).extend(boundary_hard_reasons)
        assessment["feasible"] = False
        assessment["needs_confirmation"] = True
    diagnostics = {
        **assessment,
        "requested_strategy": requested_strategy,
        "ordering_strategy": requested_strategy,
        "contour_ordering": contour_ordering,
        "used_strategies": used_strategies,
        "fallback_reasons": fallback_reasons,
        "display_only_connectors": display_only_connectors,
        "kinematically_infeasible": bool(display_only_connectors),
        "machine_executable": not bool(display_only_connectors or boundary_hard_reasons),
        "boundary_constraints": {
            "field_boundary_present": field_polygon is not None,
            "turns_adjusted": len(boundary_adjustments),
            "adjustment_reasons": boundary_adjustments,
            "non_work_violations": non_work_boundary_violations,
        },
        "pass_spacings_m": pass_spacings_m,
        "turn_decisions": turn_decisions,
        "endpoint_alignment": endpoint_alignment,
        "reference_ordering": reference_ordering,
        "contour_connector": contour_connector_diagnostics,
        "contour_route_mode": contour_route_mode,
        "service_points": {
            "mode": service_point_mode,
            "entry_present": bool(entry_point is not None),
            "exit_present": bool(
                exit_point is not None
                or any(segment_type == "exit" for segment_type in segment_types)
            ),
            "entry_reference_used": bool(entry_point is not None and oriented),
            "exit_reference_used": bool(exit_point is not None and oriented),
            "entry_connector_emitted": bool(
                include_entry_service and entry_point is not None and oriented
            ),
            "exit_connector_emitted": bool(
                any(segment_type == "exit" for segment_type in segment_types)
            ),
            "finish_extension_emitted": bool(
                any(segment_type == "exit" for segment_type in segment_types)
            ),
            "unload_count": len(unload_points) if include_exit_service else 0,
            "unload_visit_count": sum(1 for item in segment_types if item == "unload"),
            "route_start_reference_present": entry_point is not None,
            "entry_actual_start_local": contour_ordering.get(
                "entry_selected_endpoint_local"
            ),
            "entry_actual_distance_m": contour_ordering.get(
                "entry_selected_line_distance_m"
            ),
            "entry_nearest_feasible_verified": contour_ordering.get(
                "entry_nearest_feasible_verified"
            ),
            "entry_side_constrained_by_terminal_short_line": (
                contour_ordering.get(
                    "entry_side_constrained_by_terminal_short_line", False
                )
            ),
            "entry_unconstrained_endpoint_blocked_by_terminal_constraint": (
                contour_ordering.get(
                    "entry_unconstrained_endpoint_blocked_by_terminal_constraint",
                    False,
                )
            ),
            "exit_route": exit_service_route,
        },
    }
    if return_diagnostics:
        return full_path, segment_types, diagnostics
    return full_path, segment_types


def _count_polyline_self_intersections(
    points: List[Tuple[float, float]],
) -> int:
    """Count proper self intersections while ignoring adjacent segments."""
    count = 0
    for first in range(len(points) - 1):
        for second in range(first + 2, len(points) - 1):
            if first == 0 and second == len(points) - 2:
                continue
            if _line_segments_cross(
                points[first],
                points[first + 1],
                points[second],
                points[second + 1],
            ):
                count += 1
    return count


# ═══════════════════════════════════════════════════════════════
#  2. 路径验证 — 基于实际段类型计算
# ═══════════════════════════════════════════════════════════════

def validate_path(full_path: List[List[Tuple[float, float]]],
                  mask: np.ndarray,
                  main_angle: float,
                  geo=None, state=None,
                  min_turn_radius_m: float = 1.0,
                  config: Optional[Dict] = None,
                  segment_types: Optional[List[str]] = None,
                  detected_mask: Optional[np.ndarray] = None,
                  planned_mask: Optional[np.ndarray] = None) -> Dict:
    """验证路径是否合理。

    检查项：
    1. 路径是否横跨掩膜（碾压检查）
    2. 总长度（按段类型分别统计）
    3. 转弯半径是否满足
    4. 路径交叉检测

    提供 ``segment_types`` 时按段语义区分 ``work``、转弯和进出场段；
    只有缺少语义标签的历史路径才回退到偶数索引兼容规则。
    注意: full_path 和 mask 必须在同一坐标系中（均为局部裁切坐标或均为全图坐标）。
    若检测到路径坐标超出 mask 范围，将自动回退到局部坐标进行验证。

    Returns:
        {
            "valid": bool,
            "total_length_m": float,
            "work_length_m": float,
            "turn_length_m": float,
            "crossing_count": int,
            "issues": [str],
        }
    """
    cfg = dict(config or {})
    specs = _harvester_specs(state)
    footprint = validate_footprints(
        full_path,
        mask,
        geo=geo,
        state=state,
        harvester=specs,
        config=cfg,
        segment_types=segment_types,
        detected_mask=detected_mask,
        planned_mask=planned_mask,
        headland_mask=cfg.get("headland_mask"),
        support_mask=cfg.get("planning_support_mask"),
        uncertain_mask=cfg.get("uncertain_mask"),
        forbidden_mask=cfg.get("forbidden_mask"),
    )

    mpp = meters_per_pixel(mask, geo, state)
    work_length_px = 0.0
    turn_length_px = 0.0
    entry_exit_length_px = 0.0
    for segment_index, segment in enumerate(full_path):
        length_px = sum(
            math.hypot(
                segment[index][0] - segment[index - 1][0],
                segment[index][1] - segment[index - 1][1],
            )
            for index in range(1, len(segment))
        )
        segment_type = (
            segment_types[segment_index]
            if segment_types and segment_index < len(segment_types)
            else ("work" if segment_index % 2 == 0 else "turn")
        )
        if segment_type == "work":
            work_length_px += length_px
        elif segment_type.startswith("turn"):
            turn_length_px += length_px
        else:
            entry_exit_length_px += length_px

    work_length_m = work_length_px * mpp
    turn_length_m = turn_length_px * mpp
    entry_exit_length_m = entry_exit_length_px * mpp
    total_length_m = work_length_m + turn_length_m + entry_exit_length_m
    crossing_count = _count_path_intersections_proper(
        full_path,
        segment_types=segment_types,
    )
    issues = []

    profile = str(cfg.get("validation_profile", "field_trial"))
    profile_limits = {
        "field_trial": {
            "outside_field": 1.0,
            "outside_support": 5.0,
            "uncertain": 10.0,
            "forbidden": 0.0,
        },
        "machine_candidate": {
            "outside_field": 0.5,
            "outside_support": 2.0,
            "uncertain": 5.0,
            "forbidden": 0.0,
            "machine_outside_field": 0.0,
            "machine_forbidden_overlap": 0.0,
        },
    }.get(profile)
    max_core_overlap = float(cfg.get("max_track_core_overlap_pct", 8.0))
    core_overlap_policy = str(
        cfg.get("track_core_overlap_policy", "report_only")
    ).strip().lower()
    if core_overlap_policy not in {"report_only", "hard_gate"}:
        core_overlap_policy = "report_only"
    footprint_metric_policy = str(
        cfg.get("footprint_metric_policy", "report_only")
    ).strip().lower()
    if footprint_metric_policy not in {"report_only", "hard_gate"}:
        footprint_metric_policy = "report_only"
    min_geometric_coverage = float(cfg.get("min_geometric_coverage_pct", 90.0))
    max_outside = float(
        profile_limits["outside_field"] if profile_limits
        else cfg.get("max_track_outside_field_pct", 2.0)
    )
    max_outside_support = float(
        profile_limits["outside_support"] if profile_limits
        else cfg.get("max_track_outside_support_pct", 100.0)
    )
    max_uncertain = float(
        profile_limits["uncertain"] if profile_limits
        else cfg.get("max_track_uncertain_overlap_pct", 100.0)
    )
    max_forbidden = float(
        profile_limits["forbidden"] if profile_limits
        else cfg.get("max_track_forbidden_overlap_pct", 0.0)
    )
    if profile == "machine_candidate":
        max_machine_outside = float(profile_limits["machine_outside_field"])
        max_machine_forbidden = float(profile_limits["machine_forbidden_overlap"])
    else:
        max_machine_outside = None
        max_machine_forbidden = None
    if (
        core_overlap_policy == "hard_gate"
        and footprint["track_core_overlap_pct"] > max_core_overlap
    ):
        issues.append(
            f"履带与稻株核心重叠 {footprint['track_core_overlap_pct']:.1f}% "
            f"> {max_core_overlap:.1f}%"
        )
    if (
        footprint_metric_policy == "hard_gate"
        and footprint["planned_target_geometric_coverage_pct"] < min_geometric_coverage
    ):
        issues.append(
            f"规划目标几何覆盖率 {footprint['planned_target_geometric_coverage_pct']:.1f}% "
            f"< {min_geometric_coverage:.1f}%"
        )
    if footprint["track_outside_field_pct"] > max_outside:
        issues.append(
            f"履带越界率 {footprint['track_outside_field_pct']:.1f}% "
            f"> {max_outside:.1f}%"
        )
    outside_support = footprint.get("track_outside_support_pct")
    if outside_support is not None and outside_support > max_outside_support:
        issues.append(
            f"履带离开语义支撑区 {outside_support:.1f}% "
            f"> {max_outside_support:.1f}%"
        )
    uncertain_overlap = footprint.get("track_uncertain_overlap_pct")
    if uncertain_overlap is not None and uncertain_overlap > max_uncertain:
        issues.append(
            f"履带经过不确定区 {uncertain_overlap:.1f}% > {max_uncertain:.1f}%"
        )
    forbidden_overlap = footprint.get("track_forbidden_overlap_pct")
    if forbidden_overlap is not None and forbidden_overlap > max_forbidden:
        issues.append(
            f"履带进入禁行区 {forbidden_overlap:.1f}% > {max_forbidden:.1f}%"
        )
    machine_outside = _finite_float(footprint.get("machine_outside_field_pct"))
    machine_outside_limit = _finite_float(max_machine_outside)
    if max_machine_outside is not None:
        if machine_outside_limit is None:
            issues.append("整机平面包络田块越界门限无效")
        elif machine_outside is None:
            issues.append("整机平面包络田块越界指标无效")
        elif machine_outside > machine_outside_limit:
            issues.append(
                f"整机平面包络越出田块 {machine_outside:.1f}% "
                f"> {machine_outside_limit:.1f}%"
            )
    machine_forbidden = _finite_float(footprint.get("machine_forbidden_overlap_pct"))
    machine_forbidden_limit = _finite_float(max_machine_forbidden)
    if max_machine_forbidden is not None:
        if machine_forbidden_limit is None:
            issues.append("整机平面包络禁行区门限无效")
        elif machine_forbidden is None:
            issues.append("整机平面包络禁行区指标无效")
        elif machine_forbidden > machine_forbidden_limit:
            issues.append(
                f"整机平面包络进入禁行区 {machine_forbidden:.1f}% "
                f"> {machine_forbidden_limit:.1f}%"
            )
    if crossing_count > 0:
        issues.append(f"非相邻作业线交叉 {crossing_count} 处")

    result = {
        **footprint,
        "valid": len(issues) == 0,
        "total_length_m": float(total_length_m),
        "work_length_m": float(work_length_m),
        "turn_length_m": float(turn_length_m),
        "entry_exit_length_m": float(entry_exit_length_m),
        "crossing_count": int(crossing_count),
        "issues": issues,
        "validation_profile": profile,
        "track_core_overlap_policy": core_overlap_policy,
        "footprint_metric_policy": footprint_metric_policy,
        "validation_limits_pct": {
            "track_outside_field": max_outside,
            "track_outside_support": max_outside_support,
            "track_uncertain_overlap": max_uncertain,
            "track_forbidden_overlap": max_forbidden,
            "machine_outside_field": max_machine_outside,
            "machine_forbidden_overlap": max_machine_forbidden,
        },
        "field_efficiency_pct": (
            work_length_m / total_length_m * 100.0 if total_length_m > 0 else 0.0
        ),
        "idle_ratio_pct": (
            (turn_length_m + entry_exit_length_m) / total_length_m * 100.0
            if total_length_m > 0
            else 0.0
        ),
        "turn_count": sum(
            1
            for index in range(len(full_path))
            if (
                segment_types[index] == "turn"
                if segment_types and index < len(segment_types)
                else index % 2 == 1
            )
        ),
    }
    return result

def path_to_geo(full_path: List[List[Tuple[float, float]]],
                geo, state,
                segment_types: Optional[List[str]] = None) -> List[Dict]:
    """将像素坐标路径转换为地理坐标。

    Returns:
        [{"lon": float, "lat": float, "pixel_x": float, "pixel_y": float}, ...]
    """
    if not geo or not geo.is_ready():
        return []

    ox = int(getattr(state, "mask_offset_x", 0))
    oy = int(getattr(state, "mask_offset_y", 0))

    global_path = [
        [(float(px) + ox, float(py) + oy) for px, py in segment]
        for segment in full_path
    ]
    return global_path_to_geo(global_path, geo, segment_types)


def global_path_to_geo(full_path: List[List[Tuple[float, float]]],
                       geo,
                       segment_types: Optional[List[str]] = None) -> List[Dict]:
    """Convert display/global pixel path points to WGS84 lon/lat points."""
    if not geo or not geo.is_ready():
        raise ValueError("当前影像没有可用地理坐标")

    result = []
    for segment_index, seg in enumerate(full_path):
        segment_type = (
            segment_types[segment_index]
            if segment_types and segment_index < len(segment_types)
            else ("work" if segment_index % 2 == 0 else "turn")
        )
        for point_index, (px, py) in enumerate(seg):
            gx = float(px)
            gy = float(py)
            try:
                lon, lat = geo.pixel_to_lonlat(gx, gy)
            except Exception as exc:
                raise ValueError(
                    "地理坐标转换失败: "
                    f"segment={segment_index}, point={point_index}, "
                    f"pixel=({gx:.3f},{gy:.3f})"
                ) from exc
            result.append({
                "lon": float(lon),
                "lat": float(lat),
                "pixel_x": float(gx),
                "pixel_y": float(gy),
                "segment_index": int(segment_index),
                "point_index": int(point_index),
                "segment_type": segment_type,
                "status": _status_for_segment_type(segment_type),
            })

    validate_geo_points(
        result,
        expected_count=sum(len(seg) for seg in full_path),
        path_segments=full_path,
        segment_types=segment_types,
    )
    return result


# ═══════════════════════════════════════════════════════════════
#  4. 路径合理性自动验证
# ═══════════════════════════════════════════════════════════════

def validate_geo_points(geo_points: List[Dict],
                        expected_count: Optional[int] = None,
                        path_segments: Optional[List[List[Tuple[float, float]]]] = None,
                        segment_types: Optional[List[str]] = None,
                        pixel_tolerance: float = 1e-6) -> List[Dict]:
    """Validate geographic route points before writing any field-use export."""
    if not geo_points:
        raise ValueError("没有有效地理坐标点，不能导出地理文件")
    if expected_count is not None and len(geo_points) != int(expected_count):
        raise ValueError(
            f"地理坐标点数量不一致: geo={len(geo_points)}, path={int(expected_count)}"
        )
    for index, point in enumerate(geo_points):
        if not isinstance(point, dict):
            raise ValueError(f"第 {index + 1} 个地理坐标点格式无效")
        missing = [key for key in ("lon", "lat", "pixel_x", "pixel_y") if key not in point]
        if missing:
            raise ValueError(f"第 {index + 1} 个地理坐标点缺少字段: {', '.join(missing)}")
        try:
            lon = float(point["lon"])
            lat = float(point["lat"])
            pixel_x = float(point["pixel_x"])
            pixel_y = float(point["pixel_y"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {index + 1} 个地理坐标点包含非数字值") from exc
        if not all(math.isfinite(value) for value in (lon, lat, pixel_x, pixel_y)):
            raise ValueError(f"第 {index + 1} 个地理坐标点包含无穷或 NaN")
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise ValueError(
                f"第 {index + 1} 个经纬度超出合法范围: lon={lon:.8f}, lat={lat:.8f}"
            )
        if "status" in point:
            try:
                status = int(point["status"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {index + 1} 个地理点的作业状态无效") from exc
            if status not in (0, 1, 2, 3, 4):
                raise ValueError(f"第 {index + 1} 个地理点的作业状态超出允许范围: {status}")
    if path_segments is not None:
        flat_expected = []
        for segment_index, segment in enumerate(path_segments):
            segment_type = (
                segment_types[segment_index]
                if segment_types and segment_index < len(segment_types)
                else ("work" if segment_index % 2 == 0 else "turn")
            )
            for point_index, (pixel_x, pixel_y) in enumerate(segment):
                flat_expected.append((
                    segment_index,
                    point_index,
                    str(segment_type),
                    float(pixel_x),
                    float(pixel_y),
                ))
        if len(flat_expected) != len(geo_points):
            raise ValueError(
                f"地理坐标与显示路径点数不一致: geo={len(geo_points)}, "
                f"display={len(flat_expected)}"
            )
        for index, (point, expected) in enumerate(zip(geo_points, flat_expected)):
            segment_index, point_index, segment_type, pixel_x, pixel_y = expected
            if int(point.get("segment_index", -1)) != segment_index:
                raise ValueError(f"第 {index + 1} 个地理点的路径段索引与显示路径不一致")
            if int(point.get("point_index", -1)) != point_index:
                raise ValueError(f"第 {index + 1} 个地理点的段内索引与显示路径不一致")
            if str(point.get("segment_type", "")) != segment_type:
                raise ValueError(f"第 {index + 1} 个地理点的路径类型与显示路径不一致")
            if (
                abs(float(point["pixel_x"]) - pixel_x) > pixel_tolerance
                or abs(float(point["pixel_y"]) - pixel_y) > pixel_tolerance
            ):
                raise ValueError(f"第 {index + 1} 个地理点与显示路径像素位置不一致")
    return list(geo_points)


def auto_validate_path(full_path: List[List[Tuple[float, float]]],
                       mask: np.ndarray,
                       wide_bands: List[Dict],
                       main_angle: float,
                       geo=None, state=None) -> Tuple[bool, str]:
    """自动验证路径是否为真实可用的路径。

    核心检查：
    1. 路径方向一致性（工作段与主方向对齐）
    2. 掩膜内占比
    3. 非相邻段交叉
    4. 工作段覆盖度（是否覆盖了所有 4 行带）

    Returns:
        (is_valid, description)
    """
    metrics = validate_footprints(
        full_path,
        mask,
        geo=geo,
        state=state,
        harvester=_harvester_specs(state),
        config={},
    )
    # Core overlap remains in the returned description as a measured metric,
    # but it is report-only by default and must not make this legacy helper
    # reject an otherwise geometrically valid route.
    is_valid = (
        metrics["crop_mask_geometric_coverage_pct"] >= 90.0
        and metrics["track_outside_field_pct"] <= 2.0
    )
    description = (
        f"履带核心重叠(仅报告)={metrics['track_core_overlap_pct']:.1f}%, "
        f"全田作物预计碾压={metrics['rolling_canopy_pct']:.1f}%, "
        f"作物核心碾压={metrics['rolling_core_pct']:.1f}%, "
        f"作物掩膜几何覆盖={metrics['crop_mask_geometric_coverage_pct']:.1f}%, "
        f"越界={metrics['track_outside_field_pct']:.1f}%"
    )
    return is_valid, description


def _count_path_intersections_proper(
        full_path: List[List[Tuple[float, float]]],
        segment_types: Optional[List[str]] = None) -> int:
    """统计非相邻作业段之间的真实交叉次数。

    路径现在允许在第一条作业线前包含 ``entry``，在最后一条作业线后
    包含 ``exit``，运动学策略也可能插入 ``turn_reverse``。因此不能再用
    段索引的奇偶性推断作业段；有语义标签时只统计 ``segment_type ==
    "work"`` 的段。没有标签的旧调用仍保留原有的偶数索引兼容行为。
    """
    if segment_types and len(segment_types) >= len(full_path):
        work_segs = [
            (index, segment)
            for index, (segment, segment_type) in enumerate(
                zip(full_path, segment_types)
            )
            if str(segment_type) == "work"
        ]
    else:
        # Historical canonical paths had work segments at even indices.
        work_segs = [
            (index, full_path[index])
            for index in range(0, len(full_path), 2)
        ]

    cross_count = 0
    for a_idx in range(len(work_segs)):
        for b_idx in range(a_idx + 2, len(work_segs)):  # 跳过相邻
            seg_a = work_segs[a_idx][1]
            seg_b = work_segs[b_idx][1]
            cross_count += _segments_intersect_count(seg_a, seg_b)

    return cross_count


def _segments_intersect_count(
        seg_a: List[Tuple[float, float]],
        seg_b: List[Tuple[float, float]]) -> int:
    """计算两条折线之间的交叉点数量。"""
    count = 0
    for i in range(len(seg_a) - 1):
        a1, a2 = seg_a[i], seg_a[i + 1]
        for j in range(len(seg_b) - 1):
            b1, b2 = seg_b[j], seg_b[j + 1]
            if _line_segments_cross(a1, a2, b1, b2):
                count += 1
    return count


def _line_segments_cross(
        a1: Tuple[float, float], a2: Tuple[float, float],
        b1: Tuple[float, float], b2: Tuple[float, float]) -> bool:
    """判断两条线段是否相交（参数化方法）。"""
    d1x = a2[0] - a1[0]
    d1y = a2[1] - a1[1]
    d2x = b2[0] - b1[0]
    d2y = b2[1] - b1[1]

    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-10:
        return False  # 平行

    t = ((b1[0] - a1[0]) * d2y - (b1[1] - a1[1]) * d2x) / denom
    u = ((b1[0] - a1[0]) * d1y - (b1[1] - a1[1]) * d1x) / denom

    # 排除端点重叠（0 < t < 1, 0 < u < 1）
    eps = 0.01
    return eps < t < (1 - eps) and eps < u < (1 - eps)


# ═══════════════════════════════════════════════════════════════
#  5. 输出
# ═══════════════════════════════════════════════════════════════

def _atomic_text_writer(output_path: str, encoding: str = "utf-8", newline=None):
    """Create a text writer that atomically replaces the final route file."""
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".route-", suffix=".tmp", dir=directory)

    class _Writer:
        def __enter__(self):
            self.handle = os.fdopen(fd, "w", encoding=encoding, newline=newline)
            return self.handle

        def __exit__(self, exc_type, exc, traceback):
            try:
                if not self.handle.closed:
                    if exc_type is None:
                        self.handle.flush()
                        os.fsync(self.handle.fileno())
                    self.handle.close()
                if exc_type is None:
                    os.replace(temporary, output_path)
            finally:
                if os.path.exists(temporary):
                    try:
                        os.remove(temporary)
                    except OSError:
                        pass
            return False

    return _Writer()


def export_csv(geo_points: List[Dict], output_path: str):
    """导出路径为 CSV 文件（含坐标及路径段类型）。"""
    geo_points = validate_geo_points(geo_points)
    fieldnames = [
        "lon",
        "lat",
        "pixel_x",
        "pixel_y",
        "segment_index",
        "point_index",
        "segment_type",
        "status",
    ]
    with _atomic_text_writer(output_path, newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(geo_points)


def export_kml(geo_points: List[Dict], output_path: str,
               name: str = "作业路径"):
    """导出路径为 KML 文件。"""
    geo_points = validate_geo_points(geo_points)
    coords_str = "\n".join(
        f"            {p['lon']},{p['lat']},0" for p in geo_points
    )
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Style id="pathStyle">
      <LineStyle>
        <color>ff00ff00</color>
        <width>3</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>{name}</name>
      <styleUrl>#pathStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
{coords_str}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""
    with _atomic_text_writer(output_path, encoding='utf-8') as f:
        f.write(kml)


def export_json(geo_points: List[Dict], validation: Dict,
                output_path: str):
    """导出路径+验证结果为 JSON。"""
    geo_points = validate_geo_points(geo_points)
    data = {
        "crs": "EPSG:4326",
        "path": geo_points,
        "validation": validation,
    }
    with _atomic_text_writer(output_path, encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_path_format(geo_points: List[Dict], output_path: str):
    """Export research-compatibility $PATH text; not a validated machine protocol."""
    geo_points = validate_geo_points(geo_points)
    cfg = Config()
    with _atomic_text_writer(output_path, encoding="utf-8") as f:
        total = len(geo_points)
        for index, point in enumerate(geo_points):
            status = int(
                point.get(
                    "status",
                    _status_for_segment_type(point.get("segment_type", "turn")),
                )
            )
            f.write(
                f"$PATH,{cfg.ROUTE_ID},{index},{total},"
                f"{float(point['lat']):.8f},{float(point['lon']):.8f},"
                f"{cfg.OFFSET_X},{cfg.OFFSET_Y},{status}*\n"
            )
        f.write(f"{cfg.INFM_LINE}\n")


MACHINE_COMPATIBILITY_PROTOCOL = "machine_harvester_straight_turn_v1"
MACHINE_COMPATIBILITY_DEDUP_DISTANCE_M = 0.02
MACHINE_COMPATIBILITY_MAX_POINTS_PER_FILE = 50


def _machine_compatibility_output_paths(
    output_path: str,
    chunk_count: int,
) -> List[str]:
    """Choose non-overwriting numbered files for a split machine export.

    A legacy file path is kept for a one-file programmatic export so existing
    integrations remain compatible.  A directory, or any export that needs
    more than one file, uses the terminal-facing ``001.txt`` convention and
    starts after the highest numeric TXT file already present in that folder.
    """
    requested = os.path.abspath(os.fspath(output_path))
    requested_suffix = os.path.splitext(requested)[1]
    numbered = os.path.isdir(requested) or not requested_suffix or chunk_count > 1
    if not numbered:
        return [requested]

    directory = requested if os.path.isdir(requested) or not requested_suffix else os.path.dirname(requested)
    os.makedirs(directory, exist_ok=True)
    highest = 0
    try:
        names = os.listdir(directory)
    except OSError:
        names = []
    for name in names:
        stem, suffix = os.path.splitext(name)
        if suffix.lower() != ".txt" or not stem.isdigit():
            continue
        try:
            highest = max(highest, int(stem))
        except ValueError:
            continue

    paths = []
    next_index = highest + 1
    for _ in range(chunk_count):
        while True:
            candidate = os.path.join(directory, f"{next_index:03d}.txt")
            next_index += 1
            if not os.path.exists(candidate) and candidate not in paths:
                paths.append(candidate)
                break
    return paths


def _plain_offset_field(value) -> str:
    """Render an export offset as a plain number (``0.00`` -> ``0``)."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


MACHINE_COMPATIBILITY_INFM_LINE = "$INFM,0,0.7,0.7,0.7,0.7,0,5,0*"


def _write_machine_compatibility_file(
    output_path: str,
    points: List[Dict],
    cfg: Config,
) -> None:
    """Write one independently indexed <=50-point terminal PATH file.

    Terminal sample layout: every line starts directly with ``$PATH``;
    offsets are plain numbers, every record carries the fixed terminal
    status code ``2`` (``...2*``), and the final ``$INFM`` line is fixed.
    """
    # The verified terminal sample uses LF, including on Windows.  Do not let
    # text-mode newline translation turn it into CRLF.
    with _atomic_text_writer(output_path, encoding="utf-8", newline="\n") as handle:
        total = len(points)
        for index, point in enumerate(points):
            handle.write(
                f"$PATH,0,{index},{total},"
                f"{float(point['lat']):.8f},{float(point['lon']):.8f},"
                f"{_plain_offset_field(cfg.OFFSET_X)},"
                f"{_plain_offset_field(cfg.OFFSET_Y)},2*\n"
            )
        handle.write(f"{MACHINE_COMPATIBILITY_INFM_LINE}\n")


def _machine_compatibility_distance_m(first: Dict, second: Dict) -> float:
    """Return an equirectangular WGS84 distance for nearby route points."""
    lat1 = math.radians(float(first["lat"]))
    lat2 = math.radians(float(second["lat"]))
    mean_lat = (lat1 + lat2) / 2.0
    dlat = math.radians(float(second["lat"]) - float(first["lat"]))
    dlon = math.radians(float(second["lon"]) - float(first["lon"]))
    earth_radius_m = 6371008.8
    return earth_radius_m * math.hypot(dlat, dlon * math.cos(mean_lat))


def _machine_compatibility_segment_type(point: Dict) -> str:
    segment_type = str(point.get("segment_type") or "").strip().lower()
    if segment_type:
        return segment_type
    try:
        status = int(point.get("status", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("农机兼容导出点缺少有效段类型和状态") from exc
    return "work" if status == 1 else "turn"


def _machine_compatibility_segment_groups(
    geo_points: List[Dict],
) -> List[Dict]:
    """Group flattened geographic points back into ordered logical segments."""
    groups = []
    for order, point in enumerate(geo_points):
        segment_type = _machine_compatibility_segment_type(point)
        raw_index = point.get("segment_index")
        if raw_index is None:
            # Legacy point lists without segment indices can still be consumed,
            # but a type change starts a new logical segment.
            group_key = ("legacy", segment_type)
        else:
            try:
                group_key = ("explicit", int(raw_index))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"第 {order + 1} 个农机兼容点的 segment_index 无效"
                ) from exc
        if groups and groups[-1]["key"] == group_key:
            groups[-1]["points"].append(point)
            continue
        groups.append({
            "key": group_key,
            "segment_type": segment_type,
            "points": [point],
        })
    return groups


def _machine_compatibility_copy_point(
    point: Dict,
    *,
    status: int,
    compatibility_segment_type: str,
) -> Dict:
    copied = dict(point)
    copied["status"] = int(status)
    copied["segment_type"] = str(compatibility_segment_type)
    copied["compatibility_segment_type"] = str(compatibility_segment_type)
    copied["source_segment_type"] = str(point.get("segment_type") or "")
    return copied


def _machine_compatibility_same_point(first: Dict, second: Dict) -> bool:
    """Compare route points in pixel space when that evidence is available."""
    try:
        first_xy = (float(first["pixel_x"]), float(first["pixel_y"]))
        second_xy = (float(second["pixel_x"]), float(second["pixel_y"]))
    except (KeyError, TypeError, ValueError):
        return _machine_compatibility_distance_m(first, second) <= 1e-4
    if not all(math.isfinite(value) for value in (*first_xy, *second_xy)):
        return _machine_compatibility_distance_m(first, second) <= 1e-4
    return math.hypot(
        second_xy[0] - first_xy[0],
        second_xy[1] - first_xy[1],
    ) <= 1e-5


def _machine_compatibility_edges(groups: List[Dict]) -> List[Tuple[Dict, Dict, int, str]]:
    """Build work edges and one straight chord for every non-work run.

    The target terminal protocol does not receive the research turn geometry
    or its internal motion labels.  The one topology-critical exception is a
    full retrace of the immediately preceding harvested work line: collapsing
    that retrace together with the following transfer would recreate the very
    cross-field diagonal it prevents.  Preserve its endpoint as a second
    status-0 edge, while every other non-work run remains one chord.
    """
    if not groups:
        raise ValueError("没有有效路径段，不能生成农机兼容文件")

    work_indices = [
        index
        for index, group in enumerate(groups)
        if str(group["segment_type"]) == "work"
    ]
    if not work_indices:
        raise ValueError("农机兼容导出至少需要一条 work 作业线")

    edges = []
    first_work_index = work_indices[0]
    if first_work_index > 0:
        entry_start = groups[0]["points"][0]
        first_work_start = groups[first_work_index]["points"][0]
        edges.append((entry_start, first_work_start, 0, "turn"))

    for position, work_index in enumerate(work_indices):
        work_points = groups[work_index]["points"]
        if len(work_points) < 2:
            raise ValueError(
                f"第 {work_index + 1} 条 work 作业线少于两个点，不能生成兼容路径"
            )
        # The target controller consumes one straight work edge per logical
        # work line.  Intermediate research poses are deliberately discarded
        # here; otherwise the visible two-point route and the exported file
        # would diverge again for curved/dense canonical segments.
        edges.append((
            work_points[0],
            work_points[-1],
            1,
            "work",
        ))

        if position + 1 < len(work_indices):
            next_work_index = work_indices[position + 1]
            next_work = groups[next_work_index]
            between_groups = groups[work_index + 1:next_work_index]
            non_work_points = []
            for group in between_groups:
                for point in group["points"]:
                    if (
                        non_work_points
                        and _machine_compatibility_same_point(
                            non_work_points[-1], point
                        )
                    ):
                        non_work_points[-1] = point
                    else:
                        non_work_points.append(point)
            retrace_endpoint = next(
                (
                    point
                    for point in non_work_points[1:]
                    if _machine_compatibility_same_point(point, work_points[0])
                ),
                None,
            )
            preserves_harvested_retrace = bool(
                len(non_work_points) >= 2
                and _machine_compatibility_same_point(
                    non_work_points[0], work_points[-1]
                )
                and retrace_endpoint is not None
            )
            if preserves_harvested_retrace:
                edges.append((
                    work_points[-1],
                    retrace_endpoint,
                    0,
                    "turn_reverse",
                ))
                if not _machine_compatibility_same_point(
                    retrace_endpoint, next_work["points"][0]
                ):
                    edges.append((
                        retrace_endpoint,
                        next_work["points"][0],
                        0,
                        "turn_aux",
                    ))
            else:
                # The controller receives only this chord for the whole
                # ordinary non-work run.  It is deliberately not used for
                # scientific validation.
                edges.append((
                    work_points[-1],
                    next_work["points"][0],
                    0,
                    "turn",
                ))
        elif work_index < len(groups) - 1:
            exit_end = groups[-1]["points"][-1]
            edges.append((work_points[-1], exit_end, 0, "turn"))

    return edges


def _deduplicate_machine_compatibility_points(
    points: List[Dict],
    threshold_m: float,
) -> Tuple[List[Dict], int]:
    if not math.isfinite(float(threshold_m)) or float(threshold_m) <= 0.0:
        raise ValueError("农机兼容重复点阈值必须是正数")
    deduplicated = []
    removed = 0
    for point in points:
        if deduplicated and _machine_compatibility_distance_m(
            deduplicated[-1], point
        ) < float(threshold_m):
            # The later record represents the outgoing state of the next
            # logical edge, so it wins when a boundary has nearly identical
            # coordinates.
            deduplicated[-1] = point
            removed += 1
        else:
            deduplicated.append(point)
    if len(deduplicated) < 2:
        raise ValueError("清理重复点后有效兼容路径少于两个点")
    for index, point in enumerate(deduplicated):
        point["compatibility_index"] = int(index)
    return deduplicated, removed


def prepare_machine_compatibility_points(
    geo_points: List[Dict],
    *,
    duplicate_distance_m: Optional[float] = None,
) -> Tuple[List[Dict], Dict]:
    """Project the research route to the harvest-machine compatibility format.

    The research route keeps its physical turn curves.  This projection
    collapses each contiguous non-work run between two work lines to one
    straight chord and assigns the status to the outgoing edge, matching the
    supplied terminal sample (1=work, 0=turn).
    """
    geo_points = validate_geo_points(geo_points)
    cfg = Config()
    compatibility_config = cfg.section("export").get("machine_compatibility", {})
    try:
        configured_route_id = int(compatibility_config.get("route_id", 0))
        configured_work_status = int(
            compatibility_config.get("work_status", 1)
        )
        configured_non_work_status = int(
            compatibility_config.get("non_work_status", 0)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("农机兼容协议配置中的固定字段无效") from exc
    configured_protocol = str(
        compatibility_config.get("protocol", MACHINE_COMPATIBILITY_PROTOCOL)
    )
    configured_turn_geometry = str(
        compatibility_config.get("turn_geometry", "straight_chord")
    )
    if (
        configured_protocol != MACHINE_COMPATIBILITY_PROTOCOL
        or configured_route_id != 0
        or configured_work_status != 1
        or configured_non_work_status != 0
        or configured_turn_geometry != "straight_chord"
    ):
        raise ValueError(
            "农机兼容协议配置必须固定为 straight_turn_v1、route_id=0、"
            "work=1、non_work=0、turn_geometry=straight_chord"
        )
    threshold = (
        compatibility_config.get(
            "deduplicate_distance_m",
            MACHINE_COMPATIBILITY_DEDUP_DISTANCE_M,
        )
        if duplicate_distance_m is None
        else duplicate_distance_m
    )
    threshold = float(threshold)
    groups = _machine_compatibility_segment_groups(geo_points)
    edges = _machine_compatibility_edges(groups)

    raw_points = []
    for edge_index, (start, end, status, segment_type) in enumerate(edges):
        next_status = (
            int(edges[edge_index + 1][2])
            if edge_index + 1 < len(edges)
            else 0
        )
        next_segment_type = (
            str(edges[edge_index + 1][3])
            if edge_index + 1 < len(edges)
            else "turn"
        )
        start_point = _machine_compatibility_copy_point(
            start,
            status=int(status),
            compatibility_segment_type=str(segment_type),
        )
        end_point = _machine_compatibility_copy_point(
            end,
            status=next_status,
            compatibility_segment_type=next_segment_type,
        )
        if raw_points and _machine_compatibility_distance_m(
            raw_points[-1], start_point
        ) <= 1e-9:
            # Segment endpoints are intentionally shared.  Do not count this
            # serialization boundary as an input duplicate.
            raw_points[-1] = start_point
        else:
            raw_points.append(start_point)
        if raw_points and _machine_compatibility_distance_m(
            raw_points[-1], end_point
        ) <= 1e-9:
            raw_points[-1] = end_point
        else:
            raw_points.append(end_point)

    compat_points, removed_count = _deduplicate_machine_compatibility_points(
        raw_points,
        threshold,
    )
    for point in compat_points:
        status = int(point.get("status", 0))
        if status not in (0, 1):
            raise ValueError(
                f"农机兼容导出产生非法状态码: {status}，只允许 0 或 1"
            )
    diagnostics = {
        "protocol": MACHINE_COMPATIBILITY_PROTOCOL,
        "turn_geometry": "straight_chord",
        "status_semantics": "internal_outgoing_edge_only",
        "file_record_status": 2,
        "source_point_count": int(len(geo_points)),
        "output_point_count": int(len(compat_points)),
        "source_segment_count": int(len(groups)),
        "turn_run_count": int(
            sum(
                1
                for index, group in enumerate(groups)
                if str(group["segment_type"]) != "work"
                and (
                    index == 0
                    or str(groups[index - 1]["segment_type"]) == "work"
                )
            )
        ),
        "preserved_harvested_retrace_count": int(sum(
            str(edge[3]) == "turn_reverse" for edge in edges
        )),
        "removed_near_duplicate_count": int(removed_count),
        "work_line_geometry": "straight_endpoint_chord",
        "work_line_point_compression": True,
        "deduplicate_distance_m": float(threshold),
        "route_id": 0,
        "status_codes": {
            "file_record_status": 2,
            "internal_control_route": {"turn_or_other": 0, "work": 1},
        },
        "infoline": MACHINE_COMPATIBILITY_INFM_LINE,
    }
    return compat_points, diagnostics


def validate_machine_compatibility_geometry(
    compat_points: List[Dict],
    *,
    field_polygon: Optional[np.ndarray] = None,
    forbidden_regions: Optional[List[List[Tuple[float, float]]]] = None,
) -> Dict:
    """Validate the actual straight segments sent to the terminal."""
    if len(compat_points) < 2:
        raise ValueError("农机兼容路径少于两个点，不能进行几何校验")
    forbidden_polygons = []
    for region_index, region in enumerate(forbidden_regions or []):
        try:
            region_points = [
                (float(point[0]), float(point[1]))
                for point in np.asarray(region, dtype=np.float64)
            ]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"禁行区 {region_index + 1} 几何无效，不能生成兼容文件"
            ) from exc
        if len(region_points) < 3:
            raise ValueError(
                f"禁行区 {region_index + 1} 几何无效，不能生成兼容文件"
            )
        polygon = Polygon(region_points)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError(
                f"禁行区 {region_index + 1} 几何无效，不能生成兼容文件"
            )
        forbidden_polygons.append(polygon)

    for index in range(1, len(compat_points)):
        previous = compat_points[index - 1]
        current = compat_points[index]
        segment = [
            (float(previous["pixel_x"]), float(previous["pixel_y"])),
            (float(current["pixel_x"]), float(current["pixel_y"])),
        ]
        if not segment_within_field_boundary(segment, field_polygon):
            raise ValueError(
                "农机兼容转弯直线越出田块边界："
                f"point={index - 1}->{index}"
            )
        geometry = LineString(segment)
        for region_index, polygon in enumerate(forbidden_polygons):
            if polygon.intersects(geometry):
                raise ValueError(
                    "农机兼容路径直线进入禁行区："
                    f"point={index - 1}->{index}, region={region_index + 1}"
                )
    return {
        "field_boundary_checked": field_polygon is not None,
        "forbidden_regions_checked": len(forbidden_polygons),
        "valid": True,
    }


def export_machine_compatibility_path(
    geo_points: List[Dict],
    output_path: str,
    *,
    duplicate_distance_m: Optional[float] = None,
    field_polygon: Optional[np.ndarray] = None,
    forbidden_regions: Optional[List[List[Tuple[float, float]]]] = None,
) -> Tuple[List[Dict], Dict]:
    """Export straight-turn terminal PATH files, max 50 points each.

    Each written record follows the supplied terminal sample: a standalone
    point-index line, then ``$PATH,0,index,total,lat,lon,offset_x,offset_y,2*``
    (offsets as plain numbers, status code fixed to ``2``).

    When one file is enough, a caller-supplied file path is preserved.  When
    the route is split, or when a directory is supplied, numbered files are
    created without overwriting existing ``NNN.txt`` files.  Point indices
    and the declared total restart in each file because the terminal imports
    each file as an independent PATH object.
    """
    compat_points, diagnostics = prepare_machine_compatibility_points(
        geo_points,
        duplicate_distance_m=duplicate_distance_m,
    )
    geometry_diagnostics = validate_machine_compatibility_geometry(
        compat_points,
        field_polygon=field_polygon,
        forbidden_regions=forbidden_regions,
    )
    diagnostics["geometry"] = geometry_diagnostics
    cfg = Config()
    max_points = MACHINE_COMPATIBILITY_MAX_POINTS_PER_FILE
    chunks = [
        compat_points[index:index + max_points]
        for index in range(0, len(compat_points), max_points)
    ]
    if not chunks:
        raise ValueError("鍐滄満鍏煎璺緞娌℃湁鍙鍑虹偣")
    output_paths = _machine_compatibility_output_paths(output_path, len(chunks))
    for path, chunk in zip(output_paths, chunks):
        _write_machine_compatibility_file(path, chunk, cfg)
    diagnostics.update({
        "max_points_per_file": int(max_points),
        "file_count": int(len(output_paths)),
        "output_files": list(output_paths),
        "points_per_file": [int(len(chunk)) for chunk in chunks],
        "point_index_scope": "per_file",
        "file_boundary_point_overlap": False,
    })
    return compat_points, diagnostics


def _candidate_strategies_for_request(requested_strategy: str) -> List[str]:
    """Return the one requested turn implementation for this planning run.

    ``auto`` is a deterministic per-transition geometry rule in
    ``_select_turn_strategy``; it is not a global search or a comparison of
    alternative routes. Keeping this helper makes the single-run contract
    explicit for diagnostics and older callers.
    """
    requested = str(requested_strategy or "auto").strip().lower()
    return ["auto"] if requested == "auto" else [normalize_turn_strategy(requested)]


def _build_planning_factor_report(
    state=None,
    work_lines: Optional[List[List[Tuple[float, float]]]] = None,
    layout: Optional[Dict] = None,
    turn_assessment: Optional[Dict] = None,
    validation: Optional[Dict] = None,
) -> Dict:
    """Summarise the physical and agronomic factors used by route planning."""
    # DECISION-013: headland damage remains a reported metric and never
    # changes the selected turn implementation.
    specs = _harvester_specs(state)
    ox = int(getattr(state, "mask_offset_x", 0) or 0) if state is not None else 0
    oy = int(getattr(state, "mask_offset_y", 0) or 0) if state is not None else 0
    mask = getattr(state, "mask_raw", None) if state is not None else None
    entry = getattr(state, "entry_point", None) if state is not None else None
    exit_point = getattr(state, "exit_point", None) if state is not None else None
    unload_points = list(getattr(state, "unload_points", []) or []) if state is not None else []
    snapshot_unload = getattr(state, "unload_point", None) if state is not None else None
    if snapshot_unload and not unload_points:
        unload_points = [snapshot_unload]

    def local(point):
        if not point:
            return None
        return [float(point[0]) - ox, float(point[1]) - oy]

    work_lines = work_lines or []
    layout = layout or {}
    turn_assessment = turn_assessment or {}
    validation = validation or {}
    raw_sources = (
        getattr(state, "harvester_param_sources", {}) or {}
        if state is not None
        else {}
    )
    harvester_param_sources = {
        key: str(raw_sources.get(key, "legacy_unverified"))
        for key in specs
    }
    unverified_harvester_params = [
        key
        for key, source in harvester_param_sources.items()
        if source not in HARVESTER_FORMAL_MEASURED_SOURCES
    ]
    return {
        "decision_chain": [
            "processed_mask_and_field_support",
            "harvester_footprint_geometry",
            "row_band_centerlines_or_generated_work_lines",
            "entry_exit_unload_service_points",
            "turn_strategy_selection_and_fallback",
            "footprint_validation_and_route_metrics",
        ],
        "harvester": {
            key: float(value)
            for key, value in specs.items()
        },
        "harvester_param_sources": harvester_param_sources,
        "harvester_param_source_labels": {
            key: HARVESTER_PARAM_SOURCE_LABELS.get(source, source)
            for key, source in harvester_param_sources.items()
        },
        "harvester_param_source_complete": not unverified_harvester_params,
        "unverified_harvester_params": unverified_harvester_params,
        "mask": {
            "shape_hw": list(mask.shape[:2]) if mask is not None else None,
            "offset_xy": [ox, oy],
        },
        "service_points": {
            "entry_global": list(map(float, entry)) if entry else None,
            "entry_local": local(entry),
            "exit_global": list(map(float, exit_point)) if exit_point else None,
            "exit_local": local(exit_point),
            "unload_count": len(unload_points),
            "unload_local": [local(point) for point in unload_points],
            "unload_semantics": "select_one_nearest_route_end",
            "unload_scheduling_enabled": False,
        },
        "route_structure": {
            "work_line_count": len(work_lines),
            "generated_pass_count": int(layout.get("generated_pass_count", len(work_lines)) or 0),
            "layout": layout,
        },
        "turning": turn_assessment,
        "validation": validation,
    }


# ═══════════════════════════════════════════════════════════════
#  6. 完整路径规划管道
# ═══════════════════════════════════════════════════════════════

def _optimized_centerline_topology_preserved(
    baseline_lines,
    optimized_lines,
    main_angle,
) -> bool:
    """Keep one-to-one pass order and reject any crossing optimized lines."""
    if len(optimized_lines) != len(baseline_lines) or not optimized_lines:
        return False
    values = []
    profiles = []
    normal = np.asarray(
        [-math.sin(main_angle), math.cos(main_angle)], dtype=np.float64
    )
    sample_positions = np.linspace(0.0, 1.0, 33)
    for line in optimized_lines:
        points = np.asarray(line, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] < 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            return False
        projected = points @ normal
        values.append(float(np.median(projected)))
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))
        if cumulative[-1] <= 1e-9:
            return False
        profiles.append(
            np.interp(
                sample_positions,
                cumulative / cumulative[-1],
                projected,
            )
        )
    if any(right <= left + 1e-9 for left, right in zip(values, values[1:])):
        return False
    for left_profile, right_profile in zip(profiles, profiles[1:]):
        if np.any(right_profile <= left_profile + 1e-9):
            return False
    for left_index, left in enumerate(optimized_lines):
        for right in optimized_lines[left_index + 1:]:
            if _segments_intersect_count(left, right) > 0:
                return False
    return True


def _validate_work_line_layout(
    work_lines,
    processed_mask,
    *,
    geo,
    state,
    config,
    detected_mask=None,
    planned_mask=None,
):
    """Run formal footprint metrics on work segments before accepting lateral shifts."""
    return validate_footprints(
        work_lines,
        processed_mask,
        geo=geo,
        state=state,
        harvester=_harvester_specs(state),
        config=config,
        segment_types=["work"] * len(work_lines),
        detected_mask=(detected_mask if detected_mask is not None else processed_mask),
        planned_mask=(planned_mask if planned_mask is not None else processed_mask),
    )


def _uniform_zone_reference_geometry(
    zone_band: Dict,
    main_angle: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Return axial support points and centroid from compact or full band data.

    The UI deliberately removes dense contours before keeping a mask result in
    application state.  Uniform-zone planning must therefore remain possible
    from the compact four-row record: its centerline defines the axial extent,
    while the raw/processed mask still owns the actual transverse support.
    """
    if not isinstance(zone_band, dict):
        return None, None, "missing_reference_geometry"

    def _finite_points(value, minimum_rows):
        """Decode compact JSON points without allowing malformed cache data to raise."""
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if array.ndim == 1:
            if array.size == 0 or array.size % 2:
                return None
            array = array.reshape(-1, 2)
        if (
            array.ndim != 2
            or array.shape[1] != 2
            or array.shape[0] < int(minimum_rows)
            or not np.all(np.isfinite(array))
        ):
            return None
        return array

    def _finite_vector(value, minimum_size):
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if array.ndim != 1 or array.size < int(minimum_size):
            return None
        if not np.all(np.isfinite(array[:minimum_size])):
            return None
        return array

    contour_value = zone_band.get("contour")
    contour_source = "contour"
    if contour_value is None:
        contour_value = zone_band.get("planning_contour")
        contour_source = "planning_contour"
    if contour_value is not None:
        contour = _finite_points(contour_value, 3)
        if contour is not None:
            return contour, contour.mean(axis=0), contour_source

    centerline_value = zone_band.get("centerline")
    if centerline_value is not None:
        centerline = _finite_points(centerline_value, 2)
        if centerline is not None:
            centroid_value = zone_band.get("centroid")
            centroid = None
            if centroid_value is not None:
                candidate = _finite_vector(centroid_value, 2)
                if candidate is not None:
                    centroid = candidate[:2]
            if centroid is None:
                centroid = centerline.mean(axis=0)
            return centerline, centroid, "compact_centerline_and_mask"

    bbox_value = zone_band.get("bbox")
    if bbox_value is not None:
        bbox = _finite_vector(bbox_value, 4)
        if bbox is not None:
            x, y, width, height = map(float, bbox[:4])
            if width > 0.0 and height > 0.0:
                points = np.asarray([
                    (x, y), (x + width, y),
                    (x + width, y + height), (x, y + height),
                ], dtype=np.float64)
                return points, points.mean(axis=0), "compact_bbox_and_mask"
    return None, None, "missing_reference_geometry"


def _generate_uniform_zone_lines(
    zone_band: Dict,
    main_angle: float,
    spacing_m: float,
    geo=None,
    state=None,
    mask: Optional[np.ndarray] = None,
    inner_limit_offset: Optional[float] = None,
    diagnostics: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> List[List[Tuple[float, float]]]:
    """在等距种植大片区域内按固定间距生成平行作业线（常规全覆盖）。

    大片等距区（剖面宽 >> 割幅）无法用一条中心线覆盖，按解析后的固定
    作业间距（默认轨距）沿法向生成与主方向平行的作业线。宽度与线段必须基于真实掩膜：
    contour 只是边界跟踪的骨架（可能远窄于实际种植区），掩膜才是
    实际可作业范围。
    """
    points, centroid, _geometry_source = _uniform_zone_reference_geometry(
        zone_band,
        main_angle,
    )
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "reference_geometry_source": _geometry_source,
            "effective_geometry_source": _geometry_source,
            "profile_width_m": zone_band.get("profile_width_m"),
        })
    if points is None or centroid is None:
        return []
    mpp = meters_per_pixel(
        getattr(state, "mask_raw", None)
        if state is not None and getattr(state, "mask_raw", None) is not None
        else np.zeros((2, 2), dtype=np.uint8),
        geo,
        state,
    )
    spacing_px = max(1e-6, float(spacing_m) / max(mpp, 1e-9))
    direction = np.asarray([math.cos(float(main_angle)), math.sin(float(main_angle))])
    normal = np.asarray([-direction[1], direction[0]])
    cx, cy = float(centroid[0]), float(centroid[1])
    # ``white_at`` interprets ``t_off`` as a displacement from the centroid.
    # Using absolute projections here adds the centroid projection twice and
    # clips roughly half of an otherwise full-length rectangular work line.
    # DECISION-016: keep the scan and reconstruction in one local axial frame.
    along = (points - centroid) @ direction
    t_lo, t_hi = float(along.min()), float(along.max())
    mask_h, mask_w = (mask.shape[:2] if mask is not None else (0, 0))
    # 等距区是完整大片：优先用原始推理掩膜（processed_mask 经黏连切割
    # 后大片可能被切成碎片），保证细分线横贯整个种植区。
    # DECISION-009/017: this mask is the crop body after terminal headlands
    # have been separated.  Raw inference is only a compatibility fallback;
    # otherwise uniform work lines would continue through headland crop.
    zone_mask = mask
    if zone_mask is None and state is not None:
        raw = getattr(state, "mask_raw", None)
        if raw is not None and getattr(raw, "ndim", 0) == 2:
            zone_mask = raw
    zone_mask_h, zone_mask_w = (
        zone_mask.shape[:2] if zone_mask is not None else (0, 0)
    )

    def white_at(n_off: float, t_off: float) -> bool:
        x = int(round(cx + normal[0] * n_off + direction[0] * t_off))
        y = int(round(cy + normal[1] * n_off + direction[1] * t_off))
        if zone_mask is None or not (0 <= x < zone_mask_w and 0 <= y < zone_mask_h):
            return False
        return bool(zone_mask[y, x] > 100)

    # 法向范围：优先使用等距区自身的几何边界，再用掩膜确认每条线的
    # 轴向支撑。只在 t=0 这一条剖面上找边界，会在该位置恰好落到
    # 非掩膜宽行/断点时把整个等距区误判成半宽，进而漏掉交界处的一趟。
    # 对完整 contour 使用几何范围；只有紧凑 centerline 记录才退回单剖面扫掩膜。
    normal_projection = (points - centroid) @ normal
    geometry_bounds = None
    if len(points) >= 3 and np.all(np.isfinite(normal_projection)):
        geometry_bounds = (
            float(normal_projection.min()),
            float(normal_projection.max()),
        )
    # A regularized equal-distance block can retain a narrow tracking contour
    # while its measured profile spans the complete block.  Using that contour
    # as the only transverse limit silently clips the block (the real 12.47 m
    # block in the field cache was reduced to 3.91 m here).  DECISION-009/017:
    # use the measured profile width as the planning envelope when it provides
    # materially stronger evidence than the contour; keep the mask as the
    # axial support check and do not let a derived neighbour guard remove the
    # measured boundary swath.
    effective_geometry_source = _geometry_source
    profile_width_m = _finite_float(zone_band.get("profile_width_m"))
    profile_width_px = None
    geometry_width_px = None
    if profile_width_m is not None and profile_width_m > 0.0:
        profile_width_px = profile_width_m / max(mpp, 1e-9)
    if geometry_bounds is not None:
        geometry_width_px = max(0.0, geometry_bounds[1] - geometry_bounds[0])
    expansion_gap_px = max(8.0, spacing_px * 0.5)
    if (
        geometry_bounds is not None
        and profile_width_px is not None
        and profile_width_px > geometry_width_px + expansion_gap_px
        and profile_width_px > geometry_width_px * 1.5
    ):
        geometry_bounds = (-0.5 * profile_width_px, 0.5 * profile_width_px)
        effective_geometry_source = (
            f"{_geometry_source}_expanded_from_profile_width"
        )
    inner_limit_applied = inner_limit_offset is not None
    # The expanded profile is evidence for the complete equal-distance block,
    # but it can extend across the neighbouring classified row region. Keep
    # the inner boundary guard in that case: the nearest regular centreline
    # owns the boundary swath, so an equal-distance line must not be generated
    # on top of it. The processed-mask support check below remains a hard
    # condition for every generated line.
    if diagnostics is not None:
        diagnostics.update({
            "effective_geometry_source": effective_geometry_source,
            "profile_width_px": profile_width_px,
            "reference_geometry_width_px": geometry_width_px,
            "effective_transverse_width_m": (
                None
                if geometry_bounds is None
                else (geometry_bounds[1] - geometry_bounds[0]) * mpp
            ),
            "inner_limit_applied": inner_limit_applied,
        })
    scan_lo = 0
    while scan_lo > -4000 and white_at(scan_lo, 0.0):
        scan_lo -= 2
    scan_hi = 0
    while scan_hi < 4000 and white_at(scan_hi, 0.0):
        scan_hi += 2
    if geometry_bounds is not None:
        n_lo, n_hi = geometry_bounds
    else:
        n_lo, n_hi = float(scan_lo), float(scan_hi)
    # 内侧限制：等距区与相邻带在掩膜上可能相连（无间隙），细分线不得
    # 越过相邻带中心线 -1m 安全带（避免作业线落到相邻 4 行带上）。
    if inner_limit_offset is not None:
        if inner_limit_offset < 0:
            n_lo = max(n_lo, inner_limit_offset)
        else:
            n_hi = min(n_hi, inner_limit_offset)
    if n_hi - n_lo < spacing_px * 0.5:
        return []

    t_start = int(t_lo) - 30
    t_end = int(t_hi) + 30
    cfg = dict(config or {})
    try:
        configured_bridge_m = max(
            0.0,
            float(cfg.get("uniform_internal_gap_bridge_m", 30.0)),
        )
    except (TypeError, ValueError):
        configured_bridge_m = 30.0
    axial_span_m = max(0.0, float(t_hi - t_lo) * mpp)
    internal_bridge_m = min(configured_bridge_m, axial_span_m * 0.60)
    internal_bridge_px = int(round(internal_bridge_m / max(mpp, 1e-9)))
    terminal_guard_px = int(round(0.5 / max(mpp, 1e-9)))
    try:
        terminal_fragment_min_m = max(
            0.0,
            float(cfg.get("uniform_terminal_fragment_min_m", 4.0)),
        )
    except (TypeError, ValueError):
        terminal_fragment_min_m = 4.0
    terminal_fragment_min_px = int(round(
        terminal_fragment_min_m / max(mpp, 1e-9)
    ))

    def longest_supported_run(n_off: float):
        """Return the longest mask-supported axial run at one normal offset."""
        white_positions = [
            t_off for t_off in range(t_start, t_end)
            if white_at(n_off, float(t_off))
        ]
        if not white_positions:
            return None
        segments = []
        seg_start = white_positions[0]
        prev = white_positions[0]
        for value in white_positions[1:]:
            if value - prev > 60:
                segments.append((seg_start, prev))
                seg_start = value
            prev = value
        segments.append((seg_start, prev))
        # A local inference seam inside the crop body must not become a
        # turn endpoint.  Merge only substantial neighbouring runs, and
        # retain the terminal-fragment guard so a short headland residual is
        # still rejected by the existing outer-run filter below.
        merged = []
        for start, end in segments:
            if not merged:
                merged.append([start, end])
                continue
            previous_start, previous_end = merged[-1]
            gap = start - previous_end - 1
            previous_length = previous_end - previous_start + 1
            current_length = end - start + 1
            previous_is_short_terminal = (
                len(merged) == 1
                and previous_start <= t_start + terminal_guard_px
                and previous_length < terminal_fragment_min_px * 2
            )
            current_is_short_terminal = (
                end >= t_end - terminal_guard_px
                and current_length < terminal_fragment_min_px * 2
            )
            if (
                0 <= gap <= internal_bridge_px
                and previous_length >= terminal_fragment_min_px
                and current_length >= terminal_fragment_min_px
                and not previous_is_short_terminal
                and not current_is_short_terminal
            ):
                merged[-1][1] = end
            else:
                merged.append([start, end])
        segments = [tuple(item) for item in merged]
        longest = max(segments, key=lambda seg: seg[1] - seg[0])
        return longest if longest[1] - longest[0] >= 4 else None

    # ``track_gauge`` is the machine-compatible pass spacing: the right track
    # of one pass and the left track of the next pass must share a centreline.
    # Do not redistribute the residual width into ``(n_hi-n_lo)/pass_count``;
    # that made every equal-distance pass use a different spacing.  Center the
    # residual edge margin while keeping every adjacent pass exactly
    # ``spacing_px`` apart.
    pass_count = max(1, int(math.ceil((n_hi - n_lo) / spacing_px - 1e-9)))
    if pass_count == 1:
        nominal_positions = [0.5 * (n_lo + n_hi)]
    else:
        residual_px = max(0.0, (n_hi - n_lo) - (pass_count - 1) * spacing_px)
        first_position = n_lo + 0.5 * residual_px
        nominal_positions = [
            first_position + spacing_px * pass_index
            for pass_index in range(pass_count)
        ]

    # A single global phase may be selected to bridge a narrow mask seam, but
    # it is applied to the whole equal-distance block.  Per-line phase search
    # silently turns a regular grid into irregular lanes.
    phase_candidates = [0.0]
    for fraction in (0.15, 0.30, 0.45):
        delta = spacing_px * fraction
        phase_candidates.extend((-delta, delta))
    phase_evaluations = []
    for phase in phase_candidates:
        positions = [position + phase for position in nominal_positions]
        if positions and (
            min(positions) < n_lo - 1e-6 or max(positions) > n_hi + 1e-6
        ):
            continue
        runs = [longest_supported_run(position) for position in positions]
        supported_count = sum(run is not None for run in runs)
        supported_length = sum(
            (run[1] - run[0]) for run in runs if run is not None
        )
        phase_evaluations.append((
            supported_count,
            supported_length,
            -abs(phase),
            float(phase),
            positions,
            runs,
        ))
    if not phase_evaluations:
        return []
    nominal_evaluation = next(
        (item for item in phase_evaluations if abs(item[3]) <= 1e-9),
        None,
    )
    # Preserve the centred phase whenever it already supports every planned
    # pass.  A mask-supported phase shift is only justified when the nominal
    # phase actually loses a pass at a local seam.
    selected_evaluation = (
        nominal_evaluation
        if nominal_evaluation is not None
        and nominal_evaluation[0] == pass_count
        else max(phase_evaluations, key=lambda item: item[:4])
    )
    (
        _supported_count,
        _supported_length,
        _negative_phase_abs,
        phase_offset_px,
        selected_positions,
        selected_runs,
    ) = selected_evaluation
    if diagnostics is not None:
        diagnostics.update({
            "pass_count": pass_count,
            "nominal_spacing_m": float(spacing_m),
            "actual_spacing_m": float(spacing_m),
            "phase_offset_m": float(phase_offset_px * mpp),
            "spacing_is_uniform": True,
            "supported_pass_count": int(_supported_count),
        })

    try:
        min_line_length_ratio = float(
            cfg.get("uniform_zone_min_line_length_ratio", 0.65)
        )
    except (TypeError, ValueError):
        min_line_length_ratio = 0.65
    min_line_length_ratio = float(np.clip(min_line_length_ratio, 0.5, 1.0))

    # A terminal mask fragment can be much shorter than the common crop span.
    # Feeding that fragment into contour ordering creates an artificial long
    # diagonal to the next full pass (the 003.txt failure was 18.9 m followed
    # by 55.9 m).  Only discard a short *outer* run: an interior gap remains
    # explicit evidence and is never silently filled by this filter.
    run_lengths_m = [
        (float(run[1] - run[0]) * mpp) if run is not None else None
        for run in selected_runs
    ]
    finite_lengths = [value for value in run_lengths_m if value is not None]
    median_length_m = float(np.median(finite_lengths)) if finite_lengths else 0.0
    short_edge_indices = []
    if median_length_m > 1e-9:
        threshold_m = median_length_m * min_line_length_ratio
        for index, length_m in enumerate(run_lengths_m):
            if (
                length_m is not None
                and length_m < threshold_m
                and index in {0, len(run_lengths_m) - 1}
            ):
                short_edge_indices.append(index)
    accepted_indices = {
        index for index, run in enumerate(selected_runs)
        if run is not None and index not in short_edge_indices
    }
    if diagnostics is not None:
        diagnostics.update({
            "line_lengths_m": run_lengths_m,
            "median_supported_line_length_m": median_length_m,
            "min_line_length_ratio": min_line_length_ratio,
            "short_edge_line_indices": [int(index) for index in short_edge_indices],
            "short_edge_line_filter_applied": bool(short_edge_indices),
            "accepted_pass_indices": sorted(int(index) for index in accepted_indices),
            "internal_gap_bridge_limit_m": float(internal_bridge_m),
            "internal_gap_bridge_enabled": bool(internal_bridge_px > 0),
        })

    lines = []
    for index, (position, longest) in enumerate(
        zip(selected_positions, selected_runs)
    ):
        if index not in accepted_indices:
            continue
        p1 = (
            cx + normal[0] * position + direction[0] * longest[0],
            cy + normal[1] * position + direction[1] * longest[0],
        )
        p2 = (
            cx + normal[0] * position + direction[0] * longest[1],
            cy + normal[1] * position + direction[1] * longest[1],
        )
        lines.append([p1, p2])
    return lines


def _repair_uniform_zone_endpoints(
    zone_lines: List[List[Tuple[float, float]]],
    zone_band: Dict,
    reference_lines: List[List[Tuple[float, float]]],
    main_angle: float,
    *,
    geo=None,
    state=None,
    mask: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
) -> Tuple[List[List[Tuple[float, float]]], Dict]:
    """Extend truncated uniform passes to the shared crop-body end walls.

    A compact equal-distance contour can stop at an inference seam even when
    the band's measured longitudinal extent and neighbouring complete rows
    both show that the crop body continues.  Use the neighbouring full-span
    row cohort as the end-wall reference, but only when every extension is
    covered by the selected field boundary.  The pass normal positions are
    unchanged, so equal track-gauge spacing remains exact.
    """
    cfg = dict(config or {})
    report = {
        "enabled": bool(cfg.get("centerline_endpoint_repair_enabled", True)),
        "applied": False,
        "source": "regular_centerline_shared_headland_walls",
        "reference_line_count": len(reference_lines or []),
        "target_low_projection_px": None,
        "target_high_projection_px": None,
        "target_span_m": None,
        "lines": [],
        "reason": None,
    }
    if not report["enabled"]:
        report["reason"] = "disabled"
        return [list(line) for line in zone_lines], report
    if not zone_lines:
        report["reason"] = "no_uniform_lines"
        return [], report
    if len(reference_lines or []) < 3:
        report["reason"] = "insufficient_reference_lines"
        return [list(line) for line in zone_lines], report
    if mask is None or state is None:
        report["reason"] = "missing_mask_or_state"
        return [list(line) for line in zone_lines], report

    field_polygon = field_polygon_local(state, mask.shape[:2])
    if field_polygon is None:
        report["reason"] = "missing_field_boundary"
        return [list(line) for line in zone_lines], report
    mpp = meters_per_pixel(mask, geo, state)
    direction = np.asarray(
        [math.cos(float(main_angle)), math.sin(float(main_angle))],
        dtype=np.float64,
    )
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)

    reference_records = []
    for line in reference_lines:
        points = np.asarray(line, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        projections = points @ direction
        reference_records.append({
            "low": float(np.min(projections)),
            "high": float(np.max(projections)),
            "span_m": float(np.ptp(projections) * mpp),
        })
    if len(reference_records) < 3:
        report["reason"] = "insufficient_valid_reference_lines"
        return [list(line) for line in zone_lines], report

    median_reference_span_m = float(np.median([
        item["span_m"] for item in reference_records
    ]))
    full_reference = [
        item for item in reference_records
        if item["span_m"] >= median_reference_span_m * 0.85
    ]
    if len(full_reference) < 3:
        report["reason"] = "insufficient_full_span_reference_cohort"
        return [list(line) for line in zone_lines], report

    target_low = float(np.median([item["low"] for item in full_reference]))
    target_high = float(np.median([item["high"] for item in full_reference]))
    target_span_m = float((target_high - target_low) * mpp)
    declared_span_m = _finite_float(zone_band.get("longitudinal_extent_m"))
    report.update({
        "target_low_projection_px": target_low,
        "target_high_projection_px": target_high,
        "target_span_m": target_span_m,
        "declared_span_m": declared_span_m,
    })
    if target_span_m <= 0.0:
        report["reason"] = "invalid_reference_span"
        return [list(line) for line in zone_lines], report
    if (
        declared_span_m is None
        or declared_span_m < target_span_m * 0.80
    ):
        report["reason"] = "zone_extent_does_not_support_shared_walls"
        return [list(line) for line in zone_lines], report

    try:
        max_repair_m = max(0.0, float(cfg.get(
            "centerline_endpoint_repair_max_m", 60.0
        )))
    except (TypeError, ValueError):
        max_repair_m = 60.0
    repaired_lines = []
    for line_index, source_line in enumerate(zone_lines):
        line = [tuple(map(float, point)) for point in source_line]
        points = np.asarray(line, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2:
            repaired_lines.append(line)
            continue
        projections = points @ direction
        if projections[0] > projections[-1]:
            line.reverse()
            points = np.asarray(line, dtype=np.float64)
            projections = points @ direction
        laterals = points @ normal
        low = float(projections[0])
        high = float(projections[-1])
        low_extension_m = max(0.0, (low - target_low) * mpp)
        high_extension_m = max(0.0, (target_high - high) * mpp)
        source_span_m = float(np.ptp(projections) * mpp)
        if low_extension_m > max_repair_m or high_extension_m > max_repair_m:
            repaired_lines.append(line)
            report["lines"].append({
                "line_index": int(line_index),
                "repaired": False,
                "reason": "endpoint_extension_over_limit",
                "source_span_m": source_span_m,
                "low_extension_m": low_extension_m,
                "high_extension_m": high_extension_m,
            })
            continue

        repaired = list(line)
        low_repaired = False
        high_repaired = False
        if low_extension_m > 1e-6:
            low_target = target_low * direction + float(laterals[0]) * normal
            if segment_within_field_boundary(
                [tuple(low_target), tuple(points[0])], field_polygon
            ):
                repaired[0] = tuple(map(float, low_target))
                low_repaired = True
        if high_extension_m > 1e-6:
            high_target = target_high * direction + float(laterals[-1]) * normal
            if segment_within_field_boundary(
                [tuple(points[-1]), tuple(high_target)], field_polygon
            ):
                repaired[-1] = tuple(map(float, high_target))
                high_repaired = True
        repaired_lines.append(repaired)
        repaired_points = np.asarray(repaired, dtype=np.float64)
        result_span_m = float(np.ptp(repaired_points @ direction) * mpp)
        repaired_now = bool(low_repaired or high_repaired)
        report["applied"] = bool(report["applied"] or repaired_now)
        report["lines"].append({
            "line_index": int(line_index),
            "repaired": repaired_now,
            "source_span_m": source_span_m,
            "low_extension_m": float(low_extension_m if low_repaired else 0.0),
            "high_extension_m": float(
                high_extension_m if high_repaired else 0.0
            ),
            "result_span_m": result_span_m,
            "completion_kind": "declared_extent_repair",
            "inferred": False,
        })
    if report["applied"]:
        report["reason"] = "cohort_and_field_boundary_supported"
    else:
        report["reason"] = "no_boundary_supported_extension"
    return repaired_lines, report


def _uniform_zone_inner_limit(
    zone_band: Dict,
    other_bands: List[Dict],
    main_angle: float,
    geo=None,
    state=None,
    clearance_m: Optional[float] = None,
) -> Optional[float]:
    """等距区的内侧法向限制（质心坐标系）。

    等距区与相邻种植带在掩膜上可能相连（无间隙），细分作业线若按掩膜
    白色范围延伸会落到相邻带上。取相邻带（法向最近的 4 行带）中心线
    位置并回退一个等距作业间隔作为内限，避免共享边界出现近重复线。
    """
    _points, centroid, _geometry_source = _uniform_zone_reference_geometry(
        zone_band,
        main_angle,
    )
    if centroid is None:
        return None
    direction = np.asarray([math.cos(float(main_angle)), math.sin(float(main_angle))])
    normal = np.asarray([-direction[1], direction[0]])
    zone_n = float(centroid @ normal)
    neighbor_n = None
    for band in other_bands:
        centerline = band.get("centerline")
        if not centerline or len(centerline) < 2:
            continue
        band_n = float(np.asarray(centerline, dtype=np.float64).mean(axis=0) @ normal)
        distance = band_n - zone_n
        if abs(distance) < 0.01:
            continue
        if neighbor_n is None or abs(distance) < abs(neighbor_n - zone_n):
            neighbor_n = band_n
    if neighbor_n is None:
        return None
    mpp = meters_per_pixel(
        getattr(state, "mask_raw", None)
        if state is not None and getattr(state, "mask_raw", None) is not None
        else np.zeros((2, 2), dtype=np.uint8),
        geo,
        state,
    )
    if clearance_m is None:
        values = getattr(state, "harvester_params", {}) or {}
        candidate = values.get("track_gauge_m") if isinstance(values, dict) else None
        if _finite_float(candidate) is None:
            try:
                candidate = Config().section("harvester").get("track_gauge_m")
            except Exception:
                candidate = None
        clearance_m = (
            float(candidate)
            if _finite_float(candidate) is not None and float(candidate) > 0.0
            else float(DEFAULT_HARVESTER_PARAMS["track_gauge_m"])
        )
    safety_px = max(1.0, float(clearance_m) / max(mpp, 1e-9))
    direction_sign = 1.0 if neighbor_n > zone_n else -1.0
    inner_n = neighbor_n - direction_sign * safety_px
    return float(inner_n - zone_n)


def _band_normal_position(
    band: Dict,
    main_angle: float,
) -> Optional[float]:
    """Return a stable normal-axis position without assuming left/right semantics."""
    centerline = band.get("centerline")
    points = np.asarray(
        [] if centerline is None else centerline,
        dtype=np.float64,
    )
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        contour_value = band.get("contour")
        if contour_value is None:
            contour_value = band.get("planning_contour")
        contour = np.asarray(
            [] if contour_value is None else contour_value,
            dtype=np.float64,
        )
        if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 2:
            return None
        points = contour
    normal = np.asarray(
        (-math.sin(float(main_angle)), math.cos(float(main_angle))),
        dtype=np.float64,
    )
    return float(np.median(points @ normal))


def _classify_planting_regions(
    bands: List[Dict],
    main_angle: float,
    config: Optional[Dict] = None,
) -> Tuple[List[Dict], Dict]:
    """DECISION-017: classify planting geometry before choosing path geometry.

    The decision uses measured band width, never an image-side label.  A broad
    connected band is a uniform-spacing block; ordinary extracted four-row
    bands remain centerline-constrained wide/narrow rows.  Measurements near
    the decision boundary are explicit ``ambiguous`` evidence and fail closed.
    """
    cfg = dict(config or {})
    try:
        # A normal four-row band in the field-trial imagery is about 2.5 m
        # wide. The old 2.5 m default put complete four-row bands inside the
        # uniform-zone ambiguity guard and could fail an otherwise usable
        # mixed-field plan closed. Keep explicit config overrides intact, but
        # require a clearly broader (4 m+) transverse profile before treating
        # a band as a uniform block by default.
        threshold_m = float(cfg.get("uniform_zone_min_width_m", 4.0))
    except (TypeError, ValueError):
        threshold_m = 4.0
    try:
        ambiguity_margin_m = float(
            cfg.get("planting_region_ambiguity_margin_m", 0.25)
        )
    except (TypeError, ValueError):
        ambiguity_margin_m = 0.25
    threshold_m = max(0.1, threshold_m)
    ambiguity_margin_m = max(0.0, min(ambiguity_margin_m, threshold_m * 0.8))
    lower_m = threshold_m - ambiguity_margin_m
    upper_m = threshold_m + ambiguity_margin_m

    records = []
    any_measured_width = False
    for index, band in enumerate(bands or []):
        widths = {}
        for key in ("profile_width_m", "width_m"):
            value = band.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number > 0.0:
                widths[key] = number
        any_measured_width = any_measured_width or bool(widths)
        # `profile_width_m` is the transverse profile/run measurement used by
        # DECISION-017.  `width_m` is only a legacy fallback; taking the max
        # of the two can turn a normal four-row band into a false borderline
        # uniform block when the local reconstructed width is inflated.
        effective_width_m = (
            widths.get("profile_width_m")
            if widths.get("profile_width_m") is not None
            else widths.get("width_m")
        )
        record = {
            "band_index": int(index),
            "band_id": band.get("id"),
            "profile_width_m": widths.get("profile_width_m"),
            "width_m": widths.get("width_m"),
            "effective_width_m": effective_width_m,
            "normal_position": _band_normal_position(band, main_angle),
            "upstream_band_type": str(band.get("band_type", "") or "").strip().lower(),
        }
        records.append(record)

    for record in records:
        width_m = record["effective_width_m"]
        borderline_resolution = None
        if width_m is None:
            if any_measured_width:
                classification = "ambiguous"
                reason = "missing_width_measurement_in_measured_cohort"
            else:
                # Backward-compatible contract: callers historically pass a
                # list already named/typed as extracted four-row wide bands.
                classification = "wide_narrow_rows"
                reason = "four_row_input_contract_without_width_measurement"
        elif width_m <= lower_m:
            classification = "wide_narrow_rows"
            reason = "effective_width_below_guarded_uniform_threshold"
        elif width_m >= upper_m:
            classification = "uniform_block"
            reason = "effective_width_above_guarded_uniform_threshold"
        else:
            # The mask-processing stage has already performed an independent
            # width classification.  A borderline profile is not genuinely
            # ambiguous when that upstream evidence says it is a wide band,
            # its reconstructed local width is clearly below the uniform-zone
            # guard, and the centerline spans almost the complete body.  This
            # resolves the real 2.449 m band at the wide/uniform boundary
            # without guessing from image position.  Synthetic/legacy inputs
            # without this complete evidence still fail closed below.
            record_index = int(record["band_index"])
            band = (bands or [])[record_index] if record_index < len(bands or []) else {}
            local_width_m = _finite_float(record.get("width_m"))
            observed_span_ratio = _finite_float(band.get("observed_span_ratio"))
            longitudinal_extent_ratio = _finite_float(
                band.get("longitudinal_extent_ratio")
            )
            complete_wide_evidence = (
                record.get("upstream_band_type") == "wide"
                and local_width_m is not None
                and local_width_m <= lower_m
                and observed_span_ratio is not None
                and observed_span_ratio >= 0.80
                and longitudinal_extent_ratio is not None
                and longitudinal_extent_ratio >= 0.80
            )
            if complete_wide_evidence:
                classification = "wide_narrow_rows"
                reason = "borderline_profile_resolved_by_complete_wide_band_evidence"
                borderline_resolution = {
                    "source": "upstream_band_type_and_continuous_centerline_support",
                    "upstream_band_type": "wide",
                    "local_width_m": float(local_width_m),
                    "observed_span_ratio": float(observed_span_ratio),
                    "longitudinal_extent_ratio": float(longitudinal_extent_ratio),
                }
            else:
                classification = "ambiguous"
                reason = "effective_width_inside_ambiguity_guard_band"
        record["classification"] = classification
        record["reason"] = reason
        if borderline_resolution is not None:
            record["borderline_resolution"] = borderline_resolution["source"]
            record["borderline_resolution_evidence"] = borderline_resolution

    groups = []
    regular_records = [
        item for item in records if item["classification"] == "wide_narrow_rows"
    ]
    uniform_records = [
        item for item in records if item["classification"] == "uniform_block"
    ]

    # DECISION-017: a planting *type* is not a physical work region.  The old
    # implementation put every four-row band in one global group, so a broad
    # uniform block lying between two four-row cohorts made contour ordering
    # pair lines across that block.  Partition the regular records wherever a
    # measured uniform-block centre lies between two consecutive normal
    # positions.  This is rotation/mirror invariant and uses only the same
    # geometry evidence already persisted by the classifier.
    regular_partitions = []
    positioned_regular = sorted(
        (
            item for item in regular_records
            if item.get("normal_position") is not None
        ),
        key=lambda item: float(item["normal_position"]),
    )
    unpositioned_regular = [
        item for item in regular_records
        if item.get("normal_position") is None
    ]
    uniform_positions = sorted(
        float(item["normal_position"])
        for item in uniform_records
        if item.get("normal_position") is not None
    )
    for record in positioned_regular:
        if regular_partitions:
            previous = regular_partitions[-1][-1]
            low = float(previous["normal_position"])
            high = float(record["normal_position"])
            if any(low < position < high for position in uniform_positions):
                regular_partitions.append([])
        if not regular_partitions:
            regular_partitions.append([])
        regular_partitions[-1].append(record)
    if unpositioned_regular:
        if not regular_partitions:
            regular_partitions.append([])
        regular_partitions[0].extend(unpositioned_regular)

    for group_index, partition in enumerate(regular_partitions):
        groups.append({
            "region_id": f"wide_narrow_rows_{group_index + 1}",
            "kind": "wide_narrow_rows",
            "band_indices": [item["band_index"] for item in partition],
            "band_ids": [item["band_id"] for item in partition],
            "normal_position": float(np.median([
                item["normal_position"]
                for item in partition
                if item["normal_position"] is not None
            ])) if any(
                item["normal_position"] is not None for item in partition
            ) else None,
        })
    for group_index, record in enumerate(uniform_records):
        groups.append({
            "region_id": f"uniform_block_{group_index + 1}",
            "kind": "uniform_block",
            "band_indices": [record["band_index"]],
            "band_ids": [record["band_id"]],
            "normal_position": record["normal_position"],
        })

    ambiguous = [
        item for item in records if item["classification"] == "ambiguous"
    ]
    diagnostics = {
        "algorithm_version": "planting-region-classifier-v2",
        "classification_basis": "measured_band_width_not_image_side",
        "partition_basis": (
            "uniform_block_normal_position_separates_regular_rows"
        ),
        "wide_narrow_region_count": len(regular_partitions),
        "uniform_threshold_m": threshold_m,
        "ambiguity_margin_m": ambiguity_margin_m,
        "guard_band_m": [lower_m, upper_m],
        "records": records,
        "counts": {
            "wide_narrow_rows": len(regular_records),
            "uniform_block": sum(
                item["classification"] == "uniform_block" for item in records
            ),
            "ambiguous": len(ambiguous),
        },
        "borderline_resolved_count": sum(
            bool(item.get("borderline_resolution")) for item in records
        ),
        "mixed_field": bool(regular_records) and any(
            item["classification"] == "uniform_block" for item in records
        ),
        "automatic_planning_allowed": not ambiguous and bool(groups),
        "hard_reasons": [
            "planting region classification is ambiguous for band "
            f"{item['band_id']}"
            for item in ambiguous
        ],
    }
    return groups, diagnostics


def _uniform_zone_spacing(
    config: Optional[Dict] = None,
    state=None,
) -> Tuple[float, str, Dict]:
    """Resolve equal-distance-zone spacing from the track-overlap contract.

    Adjacent machine paths are separated by the track gauge: the right-track
    centreline of one pass and the left-track centreline of the next pass then
    coincide.  A caller may still provide an explicit spacing for a controlled
    research comparison; the basis is reported so that it cannot be mistaken
    for the default machine-compatible layout.
    """
    cfg = dict(config or {})
    raw_explicit = cfg.get("uniform_zone_line_spacing_m")
    explicit = _finite_float(raw_explicit)
    if explicit is not None and explicit > 0.0:
        return float(explicit), "explicit_config", {
            "track_gauge_m": None,
            "adjacent_track_centerline_offset_m": None,
        }

    candidates = []
    state_values = getattr(state, "harvester_params", {}) or {}
    if isinstance(state_values, dict):
        candidates.append(state_values.get("track_gauge_m"))
    try:
        candidates.append(Config().section("harvester").get("track_gauge_m"))
    except Exception:
        pass
    candidates.append(DEFAULT_HARVESTER_PARAMS.get("track_gauge_m"))
    gauge = next(
        (
            float(value)
            for value in candidates
            if _finite_float(value) is not None and float(value) > 0.0
        ),
        float(DEFAULT_HARVESTER_PARAMS["track_gauge_m"]),
    )
    return gauge, "track_gauge_overlap", {
        "track_gauge_m": gauge,
        "adjacent_track_centerline_offset_m": 0.0,
    }


def _filter_work_bands_to_selection(
    bands: List[Dict],
    mask: np.ndarray,
    main_angle: float,
    geo=None,
    state=None,
    config: Optional[Dict] = None,
) -> Tuple[List[Dict], Dict]:
    """Keep automatic passes whose lateral cutter footprint fits the selection."""
    candidates = list(bands or [])
    report = {
        "algorithm_version": "selection-work-band-filter-v1",
        "enabled": False,
        "input_band_ids": [item.get("id") for item in candidates],
        "kept_band_ids": [item.get("id") for item in candidates],
        "excluded": [],
    }
    if not candidates or state is None or mask is None:
        report["reason"] = "missing_bands_state_or_mask"
        return candidates, report
    polygon = field_polygon_local(state, mask.shape[:2])
    if polygon is None:
        report["reason"] = "missing_field_boundary"
        return candidates, report
    try:
        mpp = meters_per_pixel(mask, geo, state)
    except (RuntimeError, ValueError):
        report["reason"] = "metric_scale_unavailable"
        return candidates, report

    cfg = dict(config or {})
    specs = _harvester_specs(state)
    cutter_half_m = 0.5 * float(specs["cutter_width_m"])
    normal = np.asarray(
        (-math.sin(float(main_angle)), math.cos(float(main_angle))),
        dtype=np.float64,
    )
    polygon_positions = np.asarray(polygon, dtype=np.float64) @ normal
    low_position = float(np.min(polygon_positions))
    high_position = float(np.max(polygon_positions))
    records = []
    for index, band in enumerate(candidates):
        position = _band_normal_position(band, main_angle)
        clearance_m = None
        if position is not None:
            clearance_m = min(
                float(position) - low_position,
                high_position - float(position),
            ) * mpp
        records.append({
            "band_index": int(index),
            "band_id": band.get("id"),
            "normal_position": position,
            "normal_clearance_m": clearance_m,
            "profile_width_m": _finite_float(band.get("profile_width_m")),
            "reasons": [],
        })

    measured_widths = [
        float(item["profile_width_m"])
        for item in records
        if item["profile_width_m"] is not None
    ]
    cohort_width_m = float(np.median(measured_widths)) if measured_widths else None
    positioned = [item for item in records if item["normal_position"] is not None]
    edge_ids = set()
    if positioned:
        edge_ids.add(min(positioned, key=lambda item: item["normal_position"])["band_id"])
        edge_ids.add(max(positioned, key=lambda item: item["normal_position"])["band_id"])
    uniform_threshold_m = max(
        0.1, float(cfg.get("uniform_zone_min_width_m", 4.0))
    )
    outlier_ratio = max(
        1.0, float(cfg.get("selection_edge_width_outlier_ratio", 1.75))
    )
    tolerance_m = max(
        0.0, float(cfg.get("selection_cutter_clearance_tolerance_m", 0.05))
    )
    for item in records:
        clearance_m = item["normal_clearance_m"]
        if (
            clearance_m is not None
            and clearance_m + tolerance_m < cutter_half_m
        ):
            item["reasons"].append("insufficient_cutter_normal_clearance")
        width_m = item["profile_width_m"]
        if (
            item["band_id"] in edge_ids
            and width_m is not None
            and cohort_width_m is not None
            and width_m < uniform_threshold_m
            and width_m > cohort_width_m * outlier_ratio
        ):
            item["reasons"].append("edge_width_outlier_below_uniform_threshold")

    kept = [
        band for band, item in zip(candidates, records)
        if not item["reasons"]
    ]
    excluded = [item for item in records if item["reasons"]]
    report.update({
        "enabled": True,
        "reason": None,
        "cutter_half_width_m": cutter_half_m,
        "clearance_tolerance_m": tolerance_m,
        "cohort_profile_width_m": cohort_width_m,
        "edge_width_outlier_ratio": outlier_ratio,
        "uniform_threshold_m": uniform_threshold_m,
        "records": records,
        "kept_band_ids": [item.get("id") for item in kept],
        "excluded": excluded,
    })
    return kept, report


def _prepare_work_line_layout(
    wide_bands: List[Dict],
    processed_mask: np.ndarray,
    main_angle: float,
    geo=None,
    state=None,
    config: Optional[Dict] = None,
    progress_callback=None,
    detected_mask: Optional[np.ndarray] = None,
    planned_mask: Optional[np.ndarray] = None,
    reference_bands: Optional[List[Dict]] = None,
    return_regions: bool = False,
):
    """Select the declared work-line generator without silent mode fallback."""
    cfg = dict(config or {})
    mode = str(
        cfg.get("work_line_mode", "band_centerline_with_footprint_validation")
    ).strip().lower()
    precomputed_selection_filter = cfg.get("_selection_work_band_filter")
    if isinstance(precomputed_selection_filter, dict):
        selection_filter = dict(precomputed_selection_filter)
        wide_bands = list(wide_bands or [])
    else:
        wide_bands, selection_filter = _filter_work_bands_to_selection(
            wide_bands,
            processed_mask,
            main_angle,
            geo=geo,
            state=state,
            config=cfg,
        )
    # 等距区分离：剖面宽度远大于正常作物带（正常约 1m）的带是整片等距
    # 种植区（如右侧 12.45m 宽的大片区域），一趟固定作业间距收不完，
    # 必须细分多条平行作业线（常规全覆盖），而不是过滤掉或只提一条中心线。
    region_groups, region_classification = _classify_planting_regions(
        wide_bands,
        main_angle,
        config=cfg,
    )
    if not region_classification["automatic_planning_allowed"]:
        empty_input = not wide_bands
        layout = {
            "reason": (
                "no_valid_four_row_centerlines"
                if empty_input
                else "ambiguous_planting_region_classification"
            ),
            "planting_region_classification": region_classification,
            "work_regions": [],
            "footprint_optimization_scope": (
                "not_run" if mode == "footprint_optimized" else "validation_only"
            ),
            "optimization_complete": False,
            "work_line_mode": (
                "band_centerline_with_footprint_validation"
                if mode == "footprint_optimized"
                else mode
            ),
            "requested_work_line_mode": mode,
            "selection_work_band_filter": selection_filter,
        }
        if return_regions:
            return [], layout, []
        return [], layout
    regular_groups = [
        item for item in region_groups if item["kind"] == "wide_narrow_rows"
    ]
    workable_bands = [
        wide_bands[index]
        for group in regular_groups
        for index in group.get("band_indices", [])
    ]
    uniform_groups = [
        item for item in region_groups if item["kind"] == "uniform_block"
    ]
    uniform_zones = [
        wide_bands[item["band_indices"][0]] for item in uniform_groups
    ]
    wide_bands = workable_bands
    # DECISION-011: footprint_optimized means completed centerline-anchored
    # lateral search; a requested mode may truthfully fall back.
    if not wide_bands and uniform_zones:
        work_lines = []
        layout = {
            "reason": "uniform_block_uses_cutter_width_swaths",
            "footprint_optimization_scope": "not_applicable_to_uniform_block",
            "optimization_complete": False,
            "work_line_mode": "uniform_block_cutter_width_swaths",
        }
    elif mode == "footprint_optimized":
        if not wide_bands:
            return [], {
                "reason": "no_valid_four_row_centerlines",
                "footprint_optimization_scope": "not_run",
                "optimization_complete": False,
                "work_line_mode": "band_centerline_with_footprint_validation",
                "requested_work_line_mode": mode,
            }
        baseline_lines, baseline_layout = prepare_band_centerlines(
            wide_bands,
            main_angle,
            geo=geo,
            state=state,
            config=cfg,
            reference_bands=reference_bands,
        )
        if not baseline_lines:
            work_lines = []
            layout = dict(baseline_layout or {})
            layout.update({
                "reason": layout.get("reason", "no_valid_four_row_centerlines"),
                "footprint_optimization_scope": "not_run",
                "optimization_complete": False,
                "work_line_mode": "band_centerline_with_footprint_validation",
            })
        else:
            optimized_lines, optimization_layout = generate_work_lines(
                processed_mask,
                main_angle,
                geo=geo,
                state=state,
                harvester=_harvester_specs(state),
                config=cfg,
                progress_callback=progress_callback,
                baseline_lines=baseline_lines,
            )
            optimization_layout = dict(optimization_layout or {})
            ordered_band_ids = list(
                baseline_layout.get("ordered_band_ids") or []
            )
            diagnostics = optimization_layout.get("line_optimization")
            if (
                isinstance(diagnostics, list)
                and len(diagnostics) == len(ordered_band_ids)
            ):
                bound_diagnostics = []
                for index, (band_id, item) in enumerate(
                    zip(ordered_band_ids, diagnostics)
                ):
                    diagnostic = dict(item) if isinstance(item, dict) else {}
                    if diagnostic.get("line_index") != index:
                        diagnostic["line_index"] = index
                        diagnostic["optimized"] = False
                    diagnostic["band_id"] = band_id
                    bound_diagnostics.append(diagnostic)
                optimization_layout["line_optimization"] = bound_diagnostics
            else:
                optimization_layout["line_optimization"] = []
            identity_complete = (
                len(optimization_layout["line_optimization"])
                == len(ordered_band_ids)
                and all(
                    isinstance(item, dict)
                    and item.get("line_index") == index
                    and item.get("band_id") == band_id
                    and item.get("optimized") is True
                    for index, (band_id, item) in enumerate(
                        zip(
                            ordered_band_ids,
                            optimization_layout["line_optimization"],
                        )
                    )
                )
            )
            geometry_complete = (
                optimization_layout.get("optimization_complete", True) is True
                and len(optimized_lines) == len(baseline_lines)
            )
            if geometry_complete and not _optimized_centerline_topology_preserved(
                baseline_lines, optimized_lines, main_angle
            ):
                optimization_layout["optimization_complete"] = False
                optimization_layout["reason"] = (
                    "optimized_centerline_topology_violation"
                )
            optimizer_claimed_complete = geometry_complete and identity_complete
            optimization_complete = bool(optimizer_claimed_complete)
            layout = dict(baseline_layout or {})
            layout.update(optimization_layout)
            if optimization_complete:
                baseline_footprint = _validate_work_line_layout(
                    baseline_lines,
                    processed_mask,
                    geo=geo,
                    state=state,
                    config=cfg,
                    detected_mask=detected_mask,
                    planned_mask=planned_mask,
                )
                optimized_footprint = _validate_work_line_layout(
                    optimized_lines,
                    processed_mask,
                    geo=geo,
                    state=state,
                    config=cfg,
                    detected_mask=detected_mask,
                    planned_mask=planned_mask,
                )
                comparison = compare_work_line_footprints(
                    baseline_footprint,
                    optimized_footprint,
                )
                layout["formal_comparison"] = comparison
                layout["baseline_footprint_metrics"] = summarize_work_line_footprint(
                    baseline_footprint
                )
                layout["optimized_footprint_metrics"] = summarize_work_line_footprint(
                    optimized_footprint
                )
                if comparison["accepted"]:
                    work_lines = optimized_lines
                    layout.update({
                        "source": "four_row_band_centerlines_lateral_optimization",
                        "footprint_optimization_scope": "centerline_anchored_lateral_search",
                        "optimization_complete": True,
                        "work_line_mode": "footprint_optimized",
                    })
                else:
                    work_lines = baseline_lines
                    layout.update({
                        "source": "four_row_band_centerlines",
                        "footprint_optimization_scope": "validation_only",
                        "optimization_complete": False,
                        "optimization_fallback_reason": (
                            "formal_footprint_metric_degraded"
                            if comparison["degraded_metrics"]
                            else "formal_footprint_no_strict_improvement"
                        ),
                        "work_line_mode": "band_centerline_with_footprint_validation",
                    })
            else:
                work_lines = baseline_lines
                layout.update({
                    "source": "four_row_band_centerlines",
                    "footprint_optimization_scope": "validation_only",
                    "optimization_complete": False,
                    "optimization_fallback_reason": optimization_layout.get(
                        "reason", "incomplete_centerline_optimization"
                    ),
                    "work_line_mode": "band_centerline_with_footprint_validation",
                })
    elif mode == "band_centerline_with_footprint_validation":
        if not wide_bands:
            work_lines, layout = [], {
                "reason": "no_valid_four_row_centerlines",
                "footprint_optimization_scope": "validation_only",
            }
        else:
            work_lines, layout = prepare_band_centerlines(
                wide_bands,
                main_angle,
                geo=geo,
                state=state,
                config=cfg,
                reference_bands=reference_bands,
            )
            layout["footprint_optimization_scope"] = "validation_only"
        if not work_lines and wide_bands:
            work_lines = [
                list(item.get("centerline", []))
                for item in wide_bands
                if len(item.get("centerline", [])) >= 2
            ]
            layout = {
                "fallback": "raw_band_centerlines",
                "footprint_optimization_scope": "validation_only",
                "ordered_band_ids": [
                    int(item.get("id", index + 1))
                    for index, item in enumerate(wide_bands)
                    if len(item.get("centerline", [])) >= 2
                ],
            }
    elif mode == "band_centerline":
        work_lines, layout = prepare_band_centerlines(
            wide_bands,
            main_angle,
            geo=geo,
            state=state,
            config=cfg,
            reference_bands=reference_bands,
        )
        if not work_lines and wide_bands:
            work_lines = [
                list(item.get("centerline", []))
                for item in wide_bands
                if len(item.get("centerline", [])) >= 2
            ]
            layout = {
                "fallback": "raw_band_centerlines",
                "ordered_band_ids": [
                    int(item.get("id", index + 1))
                    for index, item in enumerate(wide_bands)
                    if len(item.get("centerline", [])) >= 2
                ],
            }
    else:
        raise ValueError(f"unsupported work_line_mode: {mode}")
    layout = dict(layout or {})
    regular_work_lines = list(work_lines)
    endpoint_reference_lines = list(regular_work_lines)
    # 等距区（大片等距种植区）按固定作业间距细分平行作业线，与宽窄行中心线统筹
    uniform_zone_lines = []
    uniform_region_lines = []
    if uniform_zones:
        spacing_m, spacing_basis, spacing_diagnostics = _uniform_zone_spacing(
            cfg, state
        )
        uniform_zone_report = []
        for zone in uniform_zones:
            _zone_points, _zone_centroid, zone_geometry_source = (
                _uniform_zone_reference_geometry(zone, main_angle)
            )
            inner_limit_offset = _uniform_zone_inner_limit(
                zone,
                workable_bands,
                main_angle,
                geo,
                state,
                clearance_m=spacing_m,
            )
            zone_diagnostics = {}
            zone_lines = _generate_uniform_zone_lines(
                zone, main_angle, spacing_m, geo=geo, state=state,
                mask=processed_mask,
                inner_limit_offset=inner_limit_offset,
                diagnostics=zone_diagnostics,
                config=cfg,
            )
            pre_repair_lengths_m = list(
                zone_diagnostics.get("line_lengths_m") or []
            )
            zone_lines, uniform_endpoint_repair = (
                _repair_uniform_zone_endpoints(
                    zone_lines,
                    zone,
                    endpoint_reference_lines,
                    main_angle,
                    geo=geo,
                    state=state,
                    mask=processed_mask,
                    config=cfg,
                )
            )
            zone_diagnostics["endpoint_repair"] = uniform_endpoint_repair
            if uniform_endpoint_repair.get("applied"):
                repaired_lengths_m = [
                    float(sum(
                        math.hypot(
                            float(right[0]) - float(left[0]),
                            float(right[1]) - float(left[1]),
                        )
                        for left, right in zip(line, line[1:])
                    ) * meters_per_pixel(processed_mask, geo, state))
                    for line in zone_lines
                ]
                zone_diagnostics["pre_repair_line_lengths_m"] = (
                    pre_repair_lengths_m
                )
                zone_diagnostics["line_lengths_m"] = repaired_lengths_m
                zone_diagnostics["median_supported_line_length_m"] = float(
                    np.median(repaired_lengths_m)
                    if repaired_lengths_m else 0.0
                )
            uniform_zone_lines.extend(zone_lines)
            uniform_region_lines.append(list(zone_lines))
            uniform_zone_report.append({
                "id": zone.get("id"),
                "profile_width_m": zone.get("profile_width_m"),
                "line_spacing_m": spacing_m,
                "line_spacing_basis": spacing_basis,
                **spacing_diagnostics,
                "shared_boundary_clearance_m": spacing_m,
                "line_count": len(zone_lines),
                "reference_geometry_source": zone_geometry_source,
                "effective_geometry_source": zone_diagnostics.get(
                    "effective_geometry_source", zone_geometry_source
                ),
                "profile_width_px": zone_diagnostics.get("profile_width_px"),
                "reference_geometry_width_px": zone_diagnostics.get(
                    "reference_geometry_width_px"
                ),
                "effective_transverse_width_m": zone_diagnostics.get(
                    "effective_transverse_width_m"
                ),
                "pass_count": zone_diagnostics.get("pass_count"),
                "nominal_spacing_m": zone_diagnostics.get("nominal_spacing_m"),
                "actual_spacing_m": zone_diagnostics.get("actual_spacing_m"),
                "phase_offset_m": zone_diagnostics.get("phase_offset_m"),
                "spacing_is_uniform": zone_diagnostics.get(
                    "spacing_is_uniform", False
                ),
                "supported_pass_count": zone_diagnostics.get(
                    "supported_pass_count"
                ),
                # Keep the edge-fragment decision in the persisted layout
                # report.  A route reviewer must be able to distinguish a
                # deliberately omitted unsupported terminal fragment from a
                # missing equal-distance pass or a mask-processing failure.
                "line_lengths_m": zone_diagnostics.get("line_lengths_m"),
                "median_supported_line_length_m": zone_diagnostics.get(
                    "median_supported_line_length_m"
                ),
                "min_line_length_ratio": zone_diagnostics.get(
                    "min_line_length_ratio"
                ),
                "short_edge_line_indices": zone_diagnostics.get(
                    "short_edge_line_indices", []
                ),
                "short_edge_line_filter_applied": zone_diagnostics.get(
                    "short_edge_line_filter_applied", False
                ),
                "accepted_pass_indices": zone_diagnostics.get(
                    "accepted_pass_indices", []
                ),
                "endpoint_repair": zone_diagnostics.get("endpoint_repair"),
                "pre_repair_line_lengths_m": zone_diagnostics.get(
                    "pre_repair_line_lengths_m"
                ),
                "inner_limit_applied": zone_diagnostics.get(
                    "inner_limit_applied"
                ),
            })
        layout["uniform_zones"] = uniform_zone_report
        layout["uniform_zone_line_count"] = len(uniform_zone_lines)
        if uniform_zone_lines:
            work_lines = uniform_zone_lines + list(work_lines)
    excluded_bands = cfg.get("_excluded_wide_bands")
    if excluded_bands:
        layout["excluded_wide_bands"] = excluded_bands
    layout.setdefault("work_line_mode", mode)
    layout["requested_work_line_mode"] = mode
    layout["selection_work_band_filter"] = selection_filter
    work_regions = []
    ordered_regular_band_ids = list(layout.get("ordered_band_ids") or [])
    lines_by_band_id = {
        band_id: line
        for band_id, line in zip(
            ordered_regular_band_ids,
            regular_work_lines,
        )
    }
    for regular_group in regular_groups:
        region_lines = [
            lines_by_band_id[band_id]
            for band_id in regular_group.get("band_ids", [])
            if band_id in lines_by_band_id
        ]
        if not region_lines:
            continue
        work_regions.append({
            **regular_group,
            "work_lines": region_lines,
            "line_count": len(region_lines),
            "line_source": "four_row_band_centerlines",
        })
    for group, zone_lines in zip(uniform_groups, uniform_region_lines):
        if not zone_lines:
            continue
        work_regions.append({
            **group,
            "work_lines": zone_lines,
            "line_count": len(zone_lines),
            "line_source": "cutter_width_uniform_swaths",
        })
    layout["planting_region_classification"] = region_classification
    layout["work_regions"] = [
        {key: value for key, value in region.items() if key != "work_lines"}
        for region in work_regions
    ]
    generated_region_ids = {item.get("region_id") for item in work_regions}
    missing_region_ids = [
        item.get("region_id")
        for item in region_groups
        if item.get("region_id") not in generated_region_ids
    ]
    if missing_region_ids:
        layout["reason"] = "classified_region_has_no_valid_work_lines"
        layout["region_generation_hard_reasons"] = [
            f"classified region {region_id} has no valid work lines"
            for region_id in missing_region_ids
        ]
        if return_regions:
            return [], layout, []
        return [], layout
    if return_regions:
        return work_lines, layout, work_regions
    return work_lines, layout


def _recover_lattice_supported_work_bands(
    wide_bands: List[Dict],
    all_bands: Optional[List[Dict]],
    processed_mask: np.ndarray,
    main_angle: float,
    *,
    geo=None,
    state=None,
    config: Optional[Dict] = None,
) -> Tuple[List[Dict], Dict]:
    """DECISION-009/011/017: repair one evidenced four-row lattice omission.

    A narrow-band centre is evidence for a two-row strip, not for a missing
    four-row pass, so it must never be promoted.  An isolated gap close to two
    regular four-row spacings may instead contribute one interpolated lattice
    line when the complete line is supported by the processed crop-body mask.
    """
    cfg = dict(config or {})
    original = [dict(item) for item in (wide_bands or [])]
    validation_profile = str(cfg.get("validation_profile", "field_trial"))
    recovery_enabled = bool(cfg.get("recover_missing_work_bands", False))
    if validation_profile in {"field_trial", "machine_candidate"}:
        recovery_enabled = False
    report = {
        "algorithm_version": "four-row-lattice-recovery-v2",
        "enabled": recovery_enabled,
        "validation_profile": validation_profile,
        "candidate_source": "regular_four_row_lattice_plus_processed_body_support",
        "recovered_band_ids": [],
        "records": [],
        "reason": None,
    }
    if not report["enabled"]:
        report["reason"] = "disabled"
        return original, report
    if len(original) < 3:
        report["reason"] = "insufficient_band_evidence"
        return original, report

    region_groups, classification = _classify_planting_regions(
        original, main_angle, config=cfg
    )
    regular_group = next(
        (item for item in region_groups if item.get("kind") == "wide_narrow_rows"),
        None,
    )
    regular_indices = list((regular_group or {}).get("band_indices") or [])
    if len(regular_indices) < 3 or not classification.get(
        "automatic_planning_allowed", False
    ):
        report["reason"] = "regular_region_unavailable"
        return original, report

    normal = np.asarray(
        [-math.sin(float(main_angle)), math.cos(float(main_angle))],
        dtype=np.float64,
    )

    def band_position(band):
        points = np.asarray(band.get("centerline") or [], dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            return None
        if not np.all(np.isfinite(points)):
            return None
        return float(points.mean(axis=0) @ normal)

    regular = []
    for index in regular_indices:
        position = band_position(original[index])
        if position is not None:
            regular.append((position, original[index]))
    regular.sort(key=lambda item: item[0])
    gaps_px = np.diff([item[0] for item in regular])
    positive_gaps_px = gaps_px[np.isfinite(gaps_px) & (gaps_px > 1e-6)]
    if positive_gaps_px.size < 2:
        report["reason"] = "regular_spacing_unavailable"
        return original, report

    mpp = meters_per_pixel(processed_mask, geo, state)
    nominal_spacing_m = float(np.median(positive_gaps_px) * mpp)
    gap_multiplier = max(
        1.25, float(cfg.get("missing_work_band_gap_multiplier", 1.45))
    )
    gap_threshold_m = nominal_spacing_m * gap_multiplier
    integer_tolerance = max(
        0.05, float(cfg.get("missing_work_band_integer_tolerance", 0.35))
    )
    min_support_ratio = float(np.clip(
        cfg.get("missing_work_band_min_support_ratio", 0.65), 0.0, 1.0
    ))
    min_axial_ratio = float(np.clip(
        cfg.get("missing_work_band_min_axial_ratio", 0.80), 0.0, 1.0
    ))
    max_missing_count = max(
        1, int(cfg.get("missing_work_band_max_missing_count", 1))
    )
    phase_neighbor_tolerance = max(
        0.05,
        float(cfg.get("missing_work_band_phase_neighbor_tolerance", 0.35)),
    )
    cross_section_half_width_ratio = float(np.clip(
        cfg.get("missing_work_band_cross_section_half_width_ratio", 0.50),
        0.05,
        1.0,
    ))
    min_cross_section_occupancy = float(np.clip(
        cfg.get("missing_work_band_min_cross_section_occupancy", 0.12),
        0.0,
        1.0,
    ))
    interval_px = max(
        2.0,
        float(cfg.get("centerline_interval_m", 0.60)) / max(mpp, 1e-9),
    )

    direction = np.asarray(
        [math.cos(float(main_angle)), math.sin(float(main_angle))],
        dtype=np.float64,
    )
    regular_widths_m = []
    regular_lengths_m = []
    for _position, item in regular:
        width = _finite_float(
            item.get("profile_width_m", item.get("width_m"))
        )
        if width is not None and width > 0.0:
            regular_widths_m.append(width)
        points = np.asarray(item.get("centerline") or [], dtype=np.float64)
        regular_lengths_m.append(float(
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum() * mpp
        ))
    if not regular_widths_m or not regular_lengths_m:
        report["reason"] = "regular_geometry_unavailable"
        return original, report
    recovered_width_m = float(np.median(regular_widths_m))
    median_length_m = float(np.median(regular_lengths_m))

    existing_ids = {item.get("id") for item in original}
    nonwide_positions = []
    for item in all_bands or []:
        if item.get("id") in existing_ids:
            continue
        position = band_position(item)
        if position is not None:
            nonwide_positions.append((position, item))

    mask_h, mask_w = processed_mask.shape[:2]

    def interpolate_line(left_band, right_band, fraction):
        left_points = np.asarray(
            left_band.get("centerline") or [], dtype=np.float64
        )
        right_points = np.asarray(
            right_band.get("centerline") or [], dtype=np.float64
        )
        left_t = left_points @ direction
        right_t = right_points @ direction
        left_order = np.argsort(left_t)
        right_order = np.argsort(right_t)
        left_t, left_n = left_t[left_order], left_points[left_order] @ normal
        right_t, right_n = right_t[right_order], right_points[right_order] @ normal
        left_keep = np.concatenate(([True], np.diff(left_t) > 1e-6))
        right_keep = np.concatenate(([True], np.diff(right_t) > 1e-6))
        left_t, left_n = left_t[left_keep], left_n[left_keep]
        right_t, right_n = right_t[right_keep], right_n[right_keep]
        if left_t.size < 2 or right_t.size < 2:
            return [], 0.0, 0.0, 0.0, 0.0
        axial_lo = float(max(left_t[0], right_t[0]))
        axial_hi = float(min(left_t[-1], right_t[-1]))
        if axial_hi <= axial_lo:
            return [], 0.0, 0.0, 0.0, 0.0
        axial_length_m = (axial_hi - axial_lo) * mpp
        axial_ratio = axial_length_m / max(median_length_m, 1e-9)
        count = max(
            2, int(math.ceil((axial_hi - axial_lo) / interval_px)) + 1
        )
        sample_t = np.linspace(axial_lo, axial_hi, count)
        sample_n = (
            (1.0 - fraction) * np.interp(sample_t, left_t, left_n)
            + fraction * np.interp(sample_t, right_t, right_n)
        )
        points = sample_t[:, None] * direction + sample_n[:, None] * normal
        x = np.rint(points[:, 0]).astype(np.int64)
        y = np.rint(points[:, 1]).astype(np.int64)
        inside = (x >= 0) & (x < mask_w) & (y >= 0) & (y < mask_h)
        direct_support_ratio = float(
            np.mean(processed_mask[y[inside], x[inside]] > 0)
        ) if np.any(inside) else 0.0
        half_width_px = max(
            1.0,
            recovered_width_m * cross_section_half_width_ratio
            / max(mpp, 1e-9),
        )
        cross_count = max(
            5,
            int(math.ceil(
                2.0 * half_width_px
                / max(1.0, 0.08 / max(mpp, 1e-9))
            )) + 1,
        )
        if cross_count % 2 == 0:
            cross_count += 1
        offsets = np.linspace(-half_width_px, half_width_px, cross_count)
        cross_points = points[:, None, :] + offsets[None, :, None] * normal
        cross_x = np.rint(cross_points[:, :, 0]).astype(np.int64)
        cross_y = np.rint(cross_points[:, :, 1]).astype(np.int64)
        cross_inside = (
            (cross_x >= 0) & (cross_x < mask_w)
            & (cross_y >= 0) & (cross_y < mask_h)
        )
        cross_evidence = np.zeros(cross_inside.shape, dtype=bool)
        cross_evidence[cross_inside] = (
            processed_mask[cross_y[cross_inside], cross_x[cross_inside]] > 0
        )
        cross_occupancy = np.mean(cross_evidence, axis=1)
        support_ratio = float(np.mean(
            cross_occupancy >= min_cross_section_occupancy
        )) if cross_occupancy.size else 0.0
        mean_cross_occupancy = float(np.mean(cross_occupancy)) \
            if cross_occupancy.size else 0.0
        return (
            [tuple(map(float, point)) for point in points],
            support_ratio,
            axial_ratio,
            direct_support_ratio,
            mean_cross_occupancy,
        )

    def planning_contour(centerline):
        points = np.asarray(centerline, dtype=np.float64)
        half_width_px = max(
            1.0, recovered_width_m * 0.5 / max(mpp, 1e-9)
        )
        upper = points + normal * half_width_px
        lower = points - normal * half_width_px
        return [
            tuple(map(float, point))
            for point in np.vstack((upper, lower[::-1]))
        ]

    regular_positions = np.asarray([item[0] for item in regular], dtype=np.float64)
    nominal_spacing_px = nominal_spacing_m / max(mpp, 1e-9)

    def phase_supported_target(gap_index):
        left_indices = [gap_index]
        cursor = gap_index
        while cursor > 0:
            local_gap = regular_positions[cursor] - regular_positions[cursor - 1]
            if abs(local_gap / nominal_spacing_px - 1.0) > phase_neighbor_tolerance:
                break
            left_indices.insert(0, cursor - 1)
            cursor -= 1
        right_indices = [gap_index + 1]
        cursor = gap_index + 1
        while cursor < len(regular_positions) - 1:
            local_gap = regular_positions[cursor + 1] - regular_positions[cursor]
            if abs(local_gap / nominal_spacing_px - 1.0) > phase_neighbor_tolerance:
                break
            right_indices.append(cursor + 1)
            cursor += 1

        def extrapolate(indices, toward):
            values = regular_positions[indices]
            steps = np.arange(values.size, dtype=np.float64)
            slope, intercept = np.polyfit(steps, values, 1)
            return float(intercept - slope if toward == "before" else (
                intercept + slope * values.size
            ))

        if len(right_indices) >= 3 and len(right_indices) > len(left_indices):
            return (
                extrapolate(right_indices, "before"),
                "extrapolated_from_longer_right_four_row_cohort",
                len(left_indices),
                len(right_indices),
            )
        if len(left_indices) >= 3 and len(left_indices) > len(right_indices):
            return (
                extrapolate(left_indices, "after"),
                "extrapolated_from_longer_left_four_row_cohort",
                len(left_indices),
                len(right_indices),
            )
        return (
            float((regular_positions[gap_index] + regular_positions[gap_index + 1]) * 0.5),
            "midpoint_when_both_four_row_cohorts_are_equally_supported",
            len(left_indices),
            len(right_indices),
        )

    recovered = []
    for gap_index, ((left_pos, left_band), (right_pos, right_band)) in enumerate(
        zip(regular, regular[1:])
    ):
        gap_m = float((right_pos - left_pos) * mpp)
        if gap_m <= gap_threshold_m:
            continue
        spacing_multiple = gap_m / max(nominal_spacing_m, 1e-9)
        missing_count = max(0, int(round(spacing_multiple)) - 1)
        ignored_nonwide_ids = [
            item.get("id") for position, item in nonwide_positions
            if left_pos < position < right_pos
        ]
        if (
            missing_count != 1
            or missing_count > max_missing_count
            or abs(spacing_multiple - (missing_count + 1)) > integer_tolerance
        ):
            report["records"].append({
                "between_band_ids": [left_band.get("id"), right_band.get("id")],
                "gap_m": gap_m,
                "spacing_multiple": spacing_multiple,
                "recovered": False,
                "ignored_nonwide_band_ids": ignored_nonwide_ids,
                "reason": "gap_is_not_one_supported_four_row_lattice_omission",
            })
            continue
        target_position, phase_method, left_cohort_count, right_cohort_count = (
            phase_supported_target(gap_index)
        )
        min_subgap_px = nominal_spacing_px * 0.50
        if (
            target_position <= left_pos + min_subgap_px
            or target_position >= right_pos - min_subgap_px
        ):
            target_position = (left_pos + right_pos) * 0.5
            phase_method = "midpoint_after_phase_extrapolation_guard"
        fraction = (target_position - left_pos) / max(
            right_pos - left_pos, 1e-9
        )
        (
            centerline,
            support_ratio,
            axial_ratio,
            direct_support_ratio,
            mean_cross_occupancy,
        ) = interpolate_line(
            left_band, right_band, fraction
        )
        if (
            len(centerline) < 2
            or support_ratio < min_support_ratio
            or axial_ratio < min_axial_ratio
        ):
            report["records"].append({
                "between_band_ids": [left_band.get("id"), right_band.get("id")],
                "gap_m": gap_m,
                "spacing_multiple": spacing_multiple,
                "target_normal_position_px": target_position,
                "phase_method": phase_method,
                "phase_cohort_counts": [left_cohort_count, right_cohort_count],
                "processed_body_support_ratio": support_ratio,
                "processed_body_center_support_ratio": direct_support_ratio,
                "processed_body_mean_cross_section_occupancy": (
                    mean_cross_occupancy
                ),
                "min_cross_section_occupancy": min_cross_section_occupancy,
                "axial_length_ratio": axial_ratio,
                "ignored_nonwide_band_ids": ignored_nonwide_ids,
                "recovered": False,
                "reason": "interpolated_four_row_line_lacks_body_support",
            })
            continue
        recovery_label = (
            f"planning-gap-{left_band.get('id')}-{right_band.get('id')}-1"
        )
        synthetic_id = -1
        while synthetic_id in existing_ids:
            synthetic_id -= 1
        existing_ids.add(synthetic_id)
        contour = planning_contour(centerline)
        contour_array = np.asarray(contour, dtype=np.float64)
        promoted = {
            "id": synthetic_id,
            "centerline": centerline,
            "planning_contour": contour,
            "centroid": tuple(map(float, np.mean(centerline, axis=0))),
            "bbox": (
                float(contour_array[:, 0].min()),
                float(contour_array[:, 1].min()),
                float(np.ptp(contour_array[:, 0])),
                float(np.ptp(contour_array[:, 1])),
            ),
            "profile_width_m": recovered_width_m,
            "width_m": recovered_width_m,
            "band_type": "wide",
            "planning_recovered": True,
            "planning_recovery_reason": (
                "isolated_four_row_lattice_gap_with_processed_body_support"
            ),
            "planning_recovery_id": recovery_label,
            "source_band_type": None,
            "source_band_ids": [left_band.get("id"), right_band.get("id")],
        }
        recovered.append(promoted)
        report["recovered_band_ids"].append(synthetic_id)
        report["records"].append({
            "between_band_ids": [left_band.get("id"), right_band.get("id")],
            "recovered_band_id": synthetic_id,
            "planning_recovery_id": recovery_label,
            "gap_m": gap_m,
            "spacing_multiple": spacing_multiple,
            "target_normal_position_px": target_position,
            "phase_method": phase_method,
            "phase_cohort_counts": [left_cohort_count, right_cohort_count],
            "processed_body_support_ratio": support_ratio,
            "processed_body_center_support_ratio": direct_support_ratio,
            "processed_body_mean_cross_section_occupancy": mean_cross_occupancy,
            "min_cross_section_occupancy": min_cross_section_occupancy,
            "axial_length_ratio": axial_ratio,
            "ignored_nonwide_band_ids": ignored_nonwide_ids,
            "recovered": True,
            "reason": (
                "isolated_four_row_lattice_gap_with_processed_body_support"
            ),
        })

    report.update({
        "nominal_regular_spacing_m": nominal_spacing_m,
        "gap_threshold_m": gap_threshold_m,
        "reason": "recovered" if recovered else "no_qualified_spacing_gap",
    })
    return original + recovered, report


def _region_route_endpoint(
    path_segments: List[List[Tuple[float, float]]],
    segment_types: List[str],
    first: bool,
) -> Optional[Tuple[float, float]]:
    pairs = list(zip(path_segments or [], segment_types or []))
    iterator = pairs if first else reversed(pairs)
    for segment, segment_type in iterator:
        if segment_type == "work" and len(segment) >= 2:
            point = segment[0] if first else segment[-1]
            return (float(point[0]), float(point[1]))
    return None


def _region_route_last_work_retrace(
    path_segments: List[List[Tuple[float, float]]],
    segment_types: List[str],
) -> Optional[List[Tuple[float, float]]]:
    """Return the final harvested pass in reverse travel order.

    DECISION-016/017: an even-pass target region can have its entry side fixed
    by a terminal short line.  When that side differs from the previous
    region's exit, the only crop-supported way to change headlands is to
    retrace the already harvested final pass; a direct chord would cross the
    unharvested field.
    """
    last_work = next(
        (
            segment
            for segment, segment_type in reversed(
                list(zip(path_segments or [], segment_types or []))
            )
            if segment_type == "work" and len(segment) >= 2
        ),
        None,
    )
    if last_work is None:
        return None
    retrace = [
        (float(point[0]), float(point[1]))
        for point in reversed(last_work)
    ]
    if sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(retrace, retrace[1:])
    ) <= 1e-9:
        return None
    return retrace


def _contour_region_transition_gap_m(
    previous_path: List[List[Tuple[float, float]]],
    previous_types: List[str],
    next_path: List[List[Tuple[float, float]]],
    next_types: List[str],
    *,
    main_angle: Optional[float],
    geo=None,
    state=None,
) -> Optional[float]:
    """Return the normal-axis gap at a classified-region handoff.

    A region transfer is made at the headland, so the endpoint-to-endpoint
    distance alone is not enough to detect the failure mode seen in the real
    field: two neighbouring region passes can have endpoints only a metre
    apart while both still look like separate paths.  Compare the final and
    initial work-line centres in the common normal frame instead.
    """
    if main_angle is None or not math.isfinite(float(main_angle)):
        return None
    previous_work = next(
        (
            segment
            for segment, segment_type in reversed(
                list(zip(previous_path or [], previous_types or []))
            )
            if segment_type == "work" and len(segment) >= 2
        ),
        None,
    )
    next_work = next(
        (
            segment
            for segment, segment_type in zip(
                next_path or [], next_types or []
            )
            if segment_type == "work" and len(segment) >= 2
        ),
        None,
    )
    if previous_work is None or next_work is None:
        return None
    normal = np.asarray(
        (-math.sin(float(main_angle)), math.cos(float(main_angle))),
        dtype=np.float64,
    )
    previous_midpoint = (
        np.asarray(previous_work[0], dtype=np.float64)
        + np.asarray(previous_work[-1], dtype=np.float64)
    ) * 0.5
    next_midpoint = (
        np.asarray(next_work[0], dtype=np.float64)
        + np.asarray(next_work[-1], dtype=np.float64)
    ) * 0.5
    try:
        mpp = meters_per_pixel(
            getattr(state, "mask_raw", None)
            if state is not None and getattr(state, "mask_raw", None) is not None
            else np.zeros((2, 2), dtype=np.uint8),
            geo,
            state,
        )
    except (TypeError, ValueError):
        return None
    return float(abs(float((next_midpoint - previous_midpoint) @ normal)) * mpp)


def _minimum_contour_region_transition_gap_m(
    config: Optional[Dict] = None,
    state=None,
) -> float:
    """Resolve the minimum normal separation for a cross-region contour turn.

    The explicit setting is an escape hatch for a measured machine.  The
    default is derived from the already configured machine geometry: two
    track-gauge intervals (or 1.25 cutter widths) keeps the next region from
    being entered on the immediately neighbouring pass.  This only controls
    the region handoff; the regular within-region 0,4,1,5 contour pairing is
    unchanged.
    """
    cfg = dict(config or {})
    try:
        explicit = float(cfg.get("contour_min_region_transition_spacing_m"))
    except (TypeError, ValueError):
        explicit = float("nan")
    if math.isfinite(explicit) and explicit > 0.0:
        return float(explicit)
    values = getattr(state, "harvester_params", {}) or {}
    try:
        track_gauge = float(
            values.get("track_gauge_m", DEFAULT_HARVESTER_PARAMS["track_gauge_m"])
        )
    except (TypeError, ValueError):
        track_gauge = float(DEFAULT_HARVESTER_PARAMS["track_gauge_m"])
    try:
        cutter_width = float(
            values.get("cutter_width_m", DEFAULT_HARVESTER_PARAMS["cutter_width_m"])
        )
    except (TypeError, ValueError):
        cutter_width = float(DEFAULT_HARVESTER_PARAMS["cutter_width_m"])
    return max(2.0 * max(track_gauge, 0.0), 1.25 * max(cutter_width, 0.0))


def _headland_side_for_endpoint(
    endpoint: Tuple[float, float],
    path_groups: List[Tuple[List[List[Tuple[float, float]]], List[str]]],
    main_angle: Optional[float],
) -> Optional[str]:
    """Classify a region endpoint against the shared low/high headland walls."""
    if main_angle is None or not math.isfinite(float(main_angle)):
        return None
    axis = np.asarray((
        math.cos(float(main_angle)),
        math.sin(float(main_angle)),
    ), dtype=np.float64)
    projections = []
    for path, types in path_groups:
        for segment, segment_type in zip(path or [], types or []):
            if segment_type != "work" or len(segment) < 2:
                continue
            projections.extend(
                float(np.asarray(point, dtype=np.float64) @ axis)
                for point in (segment[0], segment[-1])
            )
    if len(projections) < 2:
        return None
    low = min(projections)
    high = max(projections)
    if high - low <= 1e-9:
        return None
    projection = float(np.asarray(endpoint, dtype=np.float64) @ axis)
    return "low" if abs(projection - low) <= abs(projection - high) else "high"


def _build_inter_region_transfer(
    previous_path: List[List[Tuple[float, float]]],
    previous_types: List[str],
    next_path: List[List[Tuple[float, float]]],
    next_types: List[str],
    *,
    geo=None,
    state=None,
    min_turn_radius_m: float,
    requested_strategy: str = "bow",
    config: Optional[Dict] = None,
    global_main_angle: Optional[float] = None,
    previous_work_retrace: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """DECISION-017: build one formal, boundary-checked region transfer."""
    cfg = dict(config or {})
    direct_start = _region_route_endpoint(
        previous_path, previous_types, first=False
    )
    retrace = [
        (float(point[0]), float(point[1]))
        for point in (previous_work_retrace or [])
    ]
    retrace_requested = bool(retrace)
    retrace_valid = bool(
        len(retrace) >= 2
        and direct_start is not None
        and math.hypot(
            retrace[0][0] - direct_start[0],
            retrace[0][1] - direct_start[1],
        ) <= 1e-6
    )
    start = retrace[-1] if retrace_valid else direct_start
    end = _region_route_endpoint(next_path, next_types, first=True)
    diagnostics = {
        "algorithm_version": "inter-region-transfer-v3",
        "segment_type": "turn_aux",
        "start": list(start) if start else None,
        "end": list(end) if end else None,
        "retrace_previous_work": bool(retrace_valid),
        "retrace_start": list(retrace[0]) if retrace_valid else None,
        "retrace_end": list(retrace[-1]) if retrace_valid else None,
        "valid": False,
        "machine_executable": False,
        "hard_reasons": [],
    }
    if retrace_requested and not retrace_valid:
        diagnostics["hard_reasons"].append(
            "previous work retrace does not start at the region exit"
        )
        return [], diagnostics
    if start is None or end is None:
        diagnostics["hard_reasons"].append("missing region work endpoint")
        return [], diagnostics

    last_work = next(
        (
            segment for segment, segment_type in reversed(
                list(zip(previous_path, previous_types))
            )
            if segment_type == "work" and len(segment) >= 2
        ),
        None,
    )
    heading_source = retrace if retrace_valid else last_work
    dx = float(heading_source[-1][0] - heading_source[-2][0])
    dy = float(heading_source[-1][1] - heading_source[-2][1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        diagnostics["hard_reasons"].append("previous region exit heading unavailable")
        return [], diagnostics
    work_direction = (dx / norm, dy / norm)
    lateral_x = float(end[0] - start[0])
    lateral_y = float(end[1] - start[1])
    cross = work_direction[0] * lateral_y - work_direction[1] * lateral_x
    turn_sign = 1.0 if cross >= 0.0 else -1.0
    mask = (
        getattr(state, "mask_raw", None)
        if state is not None and getattr(state, "mask_raw", None) is not None
        else np.zeros((2, 2), dtype=np.uint8)
    )
    mpp = meters_per_pixel(mask, geo, state)
    spacing_m = abs(cross) * mpp
    if spacing_m <= 1e-9:
        spacing_m = math.hypot(lateral_x, lateral_y) * mpp
    field_polygon = _field_boundary_local_polygon(state)
    clearance_m = max(
        float(cfg.get("turn_outward_clearance_m", 0.6)),
        float(cfg.get("turn_safety_margin_m", 0.1)),
    )
    headland_limited = _headland_space_is_limited(
        start,
        work_direction,
        field_polygon,
        required_depth_px=(2.0 * min_turn_radius_m + clearance_m)
        / max(mpp, 1e-9),
    )
    normalized_request = normalize_turn_strategy(requested_strategy)
    selector_request = (
        "contour" if normalized_request == "contour" else "bow"
    )
    decision = _select_turn_strategy(
        selector_request,
        spacing_m,
        min_turn_radius_m,
        headland_limited=headland_limited,
        geometry_tolerance_m=max(
            0.0, float(cfg.get("turn_radius_geometry_tolerance_m", 0.05))
        ),
    )
    selected_strategy = (
        "bow" if str(decision["strategy"]) == "contour"
        else str(decision["strategy"])
    )
    geometry_radius_m = (
        min(float(min_turn_radius_m), max(1e-6, spacing_m * 0.5))
        if normalized_request == "contour"
        else float(min_turn_radius_m)
    )
    effective_main_angle = global_main_angle
    if effective_main_angle is None:
        effective_main_angle = _finite_float(cfg.get("_global_main_angle"))
    previous_headland_side = _headland_side_for_endpoint(
        start,
        [(previous_path, previous_types), (next_path, next_types)],
        effective_main_angle,
    )
    next_headland_side = _headland_side_for_endpoint(
        end,
        [(previous_path, previous_types), (next_path, next_types)],
        effective_main_angle,
    )
    same_headland_side = (
        None
        if previous_headland_side is None or next_headland_side is None
        else previous_headland_side == next_headland_side
    )
    transfer, _, boundary_reason = _adjust_turn_to_field_boundary(
        start,
        end,
        work_direction,
        turn_sign,
        mpp,
        geometry_radius_m,
        selected_strategy,
        clearance_m,
        field_polygon,
        semicircle_radius_tolerance_m=(
            max(0.0, float(cfg.get("turn_radius_geometry_tolerance_m", 0.05)))
            if selected_strategy == "semicircle"
            else 0.0
        ),
        preserve_curve_on_boundary_failure=bool(
            cfg.get(
                "_preserve_curve_on_boundary_failure",
                normalized_request == "contour",
            )
        ),
    )
    inside_boundary = _segment_inside_field_boundary(transfer, field_polygon)
    curved = len(transfer) > 2 and has_non_degenerate_curve([transfer])
    next_work = next(
        (
            segment for segment, segment_type in zip(next_path, next_types)
            if segment_type == "work" and len(segment) >= 2
        ),
        None,
    )
    start_tangent = np.asarray(transfer[1], dtype=np.float64) - np.asarray(
        transfer[0], dtype=np.float64
    ) if len(transfer) >= 2 else np.zeros(2, dtype=np.float64)
    end_tangent = np.asarray(transfer[-1], dtype=np.float64) - np.asarray(
        transfer[-2], dtype=np.float64
    ) if len(transfer) >= 2 else np.zeros(2, dtype=np.float64)
    next_direction = (
        np.asarray(next_work[1], dtype=np.float64)
        - np.asarray(next_work[0], dtype=np.float64)
        if next_work is not None
        else np.zeros(2, dtype=np.float64)
    )

    def tangent_alignment(first_vector, second_vector):
        denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
        if denominator <= 1e-9:
            return None
        return float(np.dot(first_vector, second_vector) / denominator)

    start_alignment = tangent_alignment(
        start_tangent, np.asarray(work_direction, dtype=np.float64)
    )
    end_alignment = tangent_alignment(end_tangent, next_direction)
    diagnostics.update({
        "strategy": selected_strategy,
        "requested_strategy": normalized_request,
        "ordering_strategy": normalized_request,
        "geometry_radius_m": float(geometry_radius_m),
        "radius_capped_for_contour_pairing": bool(
            normalized_request == "contour"
            and geometry_radius_m < float(min_turn_radius_m) - 1e-9
        ),
        "decision": decision,
        "distance_m": float(sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(transfer, transfer[1:])
        ) * mpp),
        "retrace_distance_m": (
            float(sum(
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(retrace, retrace[1:])
            ) * mpp)
            if retrace_valid
            else 0.0
        ),
        "transverse_spacing_m": float(spacing_m),
        "inside_field_boundary": bool(inside_boundary),
        "non_degenerate_curve": bool(curved),
        "start_heading_alignment_cos": start_alignment,
        "end_heading_alignment_cos": end_alignment,
        "boundary_adjustment_reason": boundary_reason,
        "previous_headland_side": previous_headland_side,
        "next_headland_side": next_headland_side,
        "same_headland_side": same_headland_side,
    })
    if not inside_boundary:
        diagnostics["hard_reasons"].append(
            "inter-region transfer leaves field boundary"
        )
    if not curved:
        diagnostics["hard_reasons"].append(
            "inter-region transfer degraded to a straight connector"
        )
    tangent_min_cos = float(
        cfg.get("inter_region_tangent_alignment_min_cos", 0.95)
    )
    diagnostics["tangent_alignment_min_cos"] = tangent_min_cos
    if start_alignment is None or start_alignment < tangent_min_cos:
        diagnostics["hard_reasons"].append(
            "inter-region transfer is not tangent to previous work line"
        )
    if end_alignment is None or end_alignment < tangent_min_cos:
        diagnostics["hard_reasons"].append(
            "inter-region transfer is not tangent to next work line"
        )
    if same_headland_side is False:
        diagnostics["hard_reasons"].append(
            "inter-region transfer changes headland side; target region must "
            "enter from the same headland wall"
        )
    diagnostics["valid"] = not diagnostics["hard_reasons"]
    diagnostics["machine_executable"] = diagnostics["valid"]
    return transfer, diagnostics


def _connect_work_regions_independently_legacy(
    work_regions: List[Dict],
    *,
    geo=None,
    state=None,
    min_turn_radius_m: float,
    turn_strategy: str,
    config: Optional[Dict] = None,
    global_main_angle: Optional[float] = None,
    processed_mask: Optional[np.ndarray] = None,
) -> Tuple[List[List[Tuple[float, float]]], List[str], Dict, List[Dict]]:
    """Legacy regional executor retained only for historical result comparison."""
    cfg = dict(config or {})
    entry_global = getattr(state, "entry_point", None) if state is not None else None
    exit_global = getattr(state, "exit_point", None) if state is not None else None
    if entry_global is not None:
        ox = float(getattr(state, "mask_offset_x", 0) or 0)
        oy = float(getattr(state, "mask_offset_y", 0) or 0)
        entry_reference = (
            float(entry_global[0]) - ox,
            float(entry_global[1]) - oy,
        )
    else:
        entry_reference = None

    if "_route_end_reference_local" in cfg:
        end_reference_value = cfg.get("_route_end_reference_local")
        exit_reference = (
            (float(end_reference_value[0]), float(end_reference_value[1]))
            if end_reference_value is not None
            else None
        )
    elif exit_global is not None:
        ox = float(getattr(state, "mask_offset_x", 0) or 0)
        oy = float(getattr(state, "mask_offset_y", 0) or 0)
        exit_reference = (
            float(exit_global[0]) - ox,
            float(exit_global[1]) - oy,
        )
    else:
        exit_reference = None

    def region_reference_distance(item):
        if entry_reference is None:
            return float("inf")
        distances = [
            math.hypot(
                float(point[0]) - entry_reference[0],
                float(point[1]) - entry_reference[1],
            )
            for line in (item.get("work_lines") or [])
            if len(line) >= 2
            for point in (line[0], line[-1])
        ]
        return min(distances) if distances else float("inf")

    def region_end_distance(item):
        if exit_reference is None:
            return float("inf")
        distances = [
            math.hypot(
                float(point[0]) - exit_reference[0],
                float(point[1]) - exit_reference[1],
            )
            for line in (item.get("work_lines") or [])
            if len(line) >= 2
            for point in (line[0], line[-1])
        ]
        return min(distances) if distances else float("inf")

    region_reference_ordering = {
        "enabled": bool(entry_reference is not None or exit_reference is not None),
        "selected_variant": "current",
        "candidate_count": 1,
    }
    raw_regions = [dict(item) for item in (work_regions or [])]
    small_region_merges = []
    if normalize_turn_strategy(turn_strategy) == "contour" and len(raw_regions) > 1:
        # A two- or three-line physical region has no Hamiltonian ordering that
        # can visit every line without at least one adjacent-row transition.
        # Merge only that undersized region with its closest normal neighbour
        # for execution ordering; classification and source region ids remain
        # intact in diagnostics.  This is narrower than flattening the whole
        # mixed field and is the only way to satisfy the operator's explicit
        # no-adjacent-turn requirement without dropping a crop-supported line.
        while len(raw_regions) > 1:
            small_indices = [
                index
                for index, item in enumerate(raw_regions)
                if 2 <= len(item.get("work_lines") or []) <= 3
                and any(
                    other_index != index
                    and len(other.get("work_lines") or []) >= 4
                    for other_index, other in enumerate(raw_regions)
                )
            ]
            if not small_indices:
                break
            small_index = small_indices[0]
            small = raw_regions[small_index]
            small_normal = _finite_float(small.get("normal_position"))
            neighbor_candidates = []
            for index, item in enumerate(raw_regions):
                if index == small_index:
                    continue
                if len(item.get("work_lines") or []) < 4:
                    continue
                item_normal = _finite_float(item.get("normal_position"))
                normal_gap = (
                    abs(item_normal - small_normal)
                    if item_normal is not None and small_normal is not None
                    else float("inf")
                )
                neighbor_candidates.append((normal_gap, abs(index - small_index), index))
            _gap, _index_gap, neighbor_index = min(neighbor_candidates)
            neighbor = raw_regions[neighbor_index]
            members = []
            for item in (small, neighbor):
                members.extend(
                    item.get("source_regions")
                    or [{
                        "region_id": item.get("region_id"),
                        "kind": item.get("kind"),
                        "line_count": len(item.get("work_lines") or []),
                    }]
                )
            combined_lines = list(small.get("work_lines") or []) + list(
                neighbor.get("work_lines") or []
            )
            weighted_normals = [
                (
                    _finite_float(item.get("normal_position")),
                    len(item.get("work_lines") or []),
                )
                for item in (small, neighbor)
            ]
            finite_weighted = [
                (value, weight)
                for value, weight in weighted_normals
                if value is not None and weight > 0
            ]
            merged_normal = (
                sum(value * weight for value, weight in finite_weighted)
                / sum(weight for _value, weight in finite_weighted)
                if finite_weighted
                else None
            )
            source_ids = [str(item.get("region_id")) for item in members]
            merged = {
                "region_id": "+".join(source_ids),
                "kind": "contour_execution_group",
                "normal_position": merged_normal,
                "line_count": len(combined_lines),
                "work_lines": combined_lines,
                "source_regions": members,
                "source_region_ids": source_ids,
                "line_source": "nearest_region_merge_for_non_adjacent_contour",
            }
            insert_at = min(small_index, neighbor_index)
            raw_regions = [
                item
                for index, item in enumerate(raw_regions)
                if index not in {small_index, neighbor_index}
            ]
            raw_regions.insert(insert_at, merged)
            small_region_merges.append({
                "small_region_id": small.get("region_id"),
                "merged_with_region_id": neighbor.get("region_id"),
                "execution_region_id": merged["region_id"],
                "reason": "independent_2_or_3_line_contour_requires_adjacent_turn",
                "source_region_ids": source_ids,
                "line_count": len(combined_lines),
            })
    normal_positions_complete = bool(raw_regions) and all(
        _finite_float(item.get("normal_position")) is not None
        for item in raw_regions
    )
    monotonic_multi_region_order = (
        len(raw_regions) >= 3 and normal_positions_complete
    )
    if monotonic_multi_region_order:
        # The operator drives to the selected field-edge start before enabling
        # automatic tracking.  Therefore the region containing the globally
        # nearest work-line endpoint is authoritative, even when it is a
        # middle region.  After that first region, walk one adjacent normal
        # direction to its edge, then finish the remaining side.  A middle
        # start necessarily creates one later cross-over; this is preferable
        # to silently moving the route start away from the operator's marker.
        forward_regions = sorted(
            raw_regions,
            key=lambda item: (
                float(item["normal_position"]),
                str(item.get("region_id", "")),
            ),
        )

        def inter_region_reference_distance(first, second):
            distances = [
                math.hypot(
                    float(first_point[0]) - float(second_point[0]),
                    float(first_point[1]) - float(second_point[1]),
                )
                for first_line in (first.get("work_lines") or [])
                if len(first_line) >= 2
                for second_line in (second.get("work_lines") or [])
                if len(second_line) >= 2
                for first_point in (first_line[0], first_line[-1])
                for second_point in (second_line[0], second_line[-1])
            ]
            return min(distances) if distances else float("inf")

        if entry_reference is None:
            ordered_regions = forward_regions
            region_order_rule = "monotonic_normal_order_without_entry_reference"
            region_reference_ordering.update({
                "candidate_count": 1,
                "selected_variant": "stable_forward",
                "physical_adjacency_preserved": True,
            })
        else:
            nearest_region_index = min(
                range(len(forward_regions)),
                key=lambda index: (
                    region_reference_distance(forward_regions[index]),
                    index,
                ),
            )
            start = forward_regions[nearest_region_index]
            low_first = (
                [start]
                + list(reversed(forward_regions[:nearest_region_index]))
                + forward_regions[nearest_region_index + 1:]
            )
            high_first = (
                [start]
                + forward_regions[nearest_region_index + 1:]
                + list(reversed(forward_regions[:nearest_region_index]))
            )
            candidates = [("low_first", low_first), ("high_first", high_first)]

            def reference_walk_score(items):
                score = region_reference_distance(items[0])
                score += sum(
                    inter_region_reference_distance(first, second)
                    for first, second in zip(items, items[1:])
                )
                if exit_reference is not None:
                    score += region_end_distance(items[-1])
                return float(score)

            scored_candidates = [
                (reference_walk_score(items), index, name, items)
                for index, (name, items) in enumerate(candidates)
            ]
            selected_score, _, selected_name, ordered_regions = min(
                scored_candidates,
                key=lambda item: (item[0], item[1]),
            )
            region_order_rule = "nearest_entry_region_then_adjacent_normal_walk"
            region_reference_ordering.update({
                "candidate_count": len(candidates),
                "selected_variant": selected_name,
                "selected_score_px": float(selected_score),
                "nearest_entry_region_id": start.get("region_id"),
                "nearest_entry_region_normal_index": int(nearest_region_index),
                "nearest_entry_distance_px": float(
                    region_reference_distance(start)
                ),
                "candidate_scores_px": {
                    name: float(score)
                    for score, _index, name, _items in scored_candidates
                },
                "physical_adjacency_preserved": bool(
                    nearest_region_index in {0, len(forward_regions) - 1}
                ),
                "middle_start_requires_one_crossover": bool(
                    nearest_region_index not in {0, len(forward_regions) - 1}
                ),
            })
    elif entry_reference is not None:
        ordered_regions = sorted(
            raw_regions,
            key=lambda item: (
                region_reference_distance(item),
                float(item.get("normal_position"))
                if item.get("normal_position") is not None
                else float("inf"),
                str(item.get("region_id", "")),
            ),
        )
        region_order_rule = "nearest_entry_region_then_distance_tiebreak"
    else:
        ordered_regions = sorted(
            raw_regions,
            key=lambda item: (
                0 if item.get("kind") == "wide_narrow_rows" else 1,
                float(item.get("normal_position"))
                if item.get("normal_position") is not None
                else float("inf"),
                str(item.get("region_id", "")),
            ),
        )
        region_order_rule = "stable_kind_order_without_entry_reference"

    # A free exit point is a real route-order reference, not merely metadata.
    # For one or two regions, reversing the sequence is the only alternative
    # that preserves each region's internal contour and shared-wall semantics.
    if (
        not monotonic_multi_region_order
        and exit_reference is not None
        and len(ordered_regions) >= 2
    ):
        current_score = (
            region_reference_distance(ordered_regions[0])
            + region_end_distance(ordered_regions[-1])
            if entry_reference is not None
            else region_end_distance(ordered_regions[-1])
        )
        reversed_regions = list(reversed(ordered_regions))
        reversed_score = (
            region_reference_distance(reversed_regions[0])
            + region_end_distance(reversed_regions[-1])
            if entry_reference is not None
            else region_end_distance(reversed_regions[-1])
        )
        region_reference_ordering.update({
            "candidate_count": 2,
            "current_score_px": float(current_score),
            "reversed_score_px": float(reversed_score),
        })
        if reversed_score + 1e-6 < current_score:
            ordered_regions = reversed_regions
            region_order_rule = "entry_exit_reference_order"
            region_reference_ordering["selected_variant"] = "reversed"
    if len(ordered_regions) <= 1:
        region = ordered_regions[0] if ordered_regions else {"work_lines": []}
        single_region_cfg = dict(cfg)
        if exit_reference is None:
            single_region_cfg["_contour_allow_terminal_short_line"] = True
        path, types, assessment = _connect_work_lines(
            region.get("work_lines") or [],
            geo=geo,
            state=state,
            min_turn_radius_m=min_turn_radius_m,
            turn_strategy=turn_strategy,
            config=single_region_cfg,
            return_diagnostics=True,
            global_main_angle=global_main_angle,
            processed_mask=processed_mask,
        )
        assessment["region_routes"] = [{
            "region_id": region.get("region_id"),
            "kind": region.get("kind"),
            "line_count": int(region.get("line_count", 0)),
            "source_region_ids": list(
                region.get("source_region_ids") or [region.get("region_id")]
            ),
            "assessment": assessment.copy(),
        }] if ordered_regions else []
        return path, types, assessment, ordered_regions

    combined_path = []
    combined_types = []
    region_reports = []
    transfer_reports = []
    shared_contour_walls = None
    if (
        normalize_turn_strategy(turn_strategy) == "contour"
        and global_main_angle is not None
        and math.isfinite(float(global_main_angle))
    ):
        direction = np.asarray((
            math.cos(float(global_main_angle)),
            math.sin(float(global_main_angle)),
        ), dtype=np.float64)
        low_values = []
        high_values = []
        for region in ordered_regions:
            for line in region.get("work_lines") or []:
                if len(line) < 2:
                    continue
                endpoints = np.asarray((line[0], line[-1]), dtype=np.float64)
                projections = endpoints @ direction
                low_values.append(float(np.min(projections)))
                high_values.append(float(np.max(projections)))
        if low_values and high_values:
            shared_contour_walls = {
                "low": float(min(low_values)),
                "high": float(max(high_values)),
            }

    # DECISION-016: all classified regions share the same physical headland
    # lanes.  Audit every region first, then use the most restrictive safe
    # clearance on each side for the final routes.  Without this field-level
    # pass, a boundary notch in only one planting region moves that region's
    # tangent inward while the neighbouring region keeps a different tangent.
    shared_contour_clearances = None
    if shared_contour_walls is not None:
        clearance_samples = {"low": [], "high": []}
        audit_previous_endpoint = None
        for region_index, region in enumerate(ordered_regions):
            audit_cfg = dict(cfg)
            audit_cfg["_include_entry_service"] = region_index == 0
            audit_cfg["_include_exit_service"] = (
                region_index == len(ordered_regions) - 1
            )
            audit_cfg["_route_end_reference_local"] = (
                None if region_index < len(ordered_regions) - 1 else exit_reference
            )
            audit_cfg["_allow_reference_ordering_with_handoff"] = (
                region_index == len(ordered_regions) - 1
            )
            audit_cfg["_contour_allow_terminal_short_line"] = bool(
                region_index == len(ordered_regions) - 1
                and exit_reference is None
            )
            audit_cfg["_contour_forced_wall_projections"] = (
                shared_contour_walls
            )
            if audit_previous_endpoint is not None:
                audit_cfg["_route_start_reference_local"] = (
                    audit_previous_endpoint
                )
            audit_path, audit_types, audit_assessment = _connect_work_lines(
                region.get("work_lines") or [],
                geo=geo,
                state=state,
                min_turn_radius_m=min_turn_radius_m,
                turn_strategy=turn_strategy,
                config=audit_cfg,
                return_diagnostics=True,
                global_main_angle=global_main_angle,
                processed_mask=processed_mask,
            )
            safe_clearances = (
                audit_assessment.get("contour_connector", {}).get(
                    "shared_turn_clearance_m_by_side"
                )
                or {}
            )
            for side in ("low", "high"):
                value = _finite_float(safe_clearances.get(side))
                if value is not None and value >= 0.0:
                    clearance_samples[side].append(float(value))
            audit_previous_endpoint = _region_route_endpoint(
                audit_path, audit_types, first=False
            )
        shared_contour_clearances = {
            side: float(min(values))
            for side, values in clearance_samples.items()
            if values
        }
    previous_path = []
    previous_types = []
    previous_endpoint = None
    cross_region_transition_records = []
    for region_index, region in enumerate(ordered_regions):
        region_cfg = dict(cfg)
        region_cfg["_include_entry_service"] = region_index == 0
        region_cfg["_include_exit_service"] = region_index == len(ordered_regions) - 1
        region_cfg["_route_end_reference_local"] = (
            None if region_index < len(ordered_regions) - 1 else exit_reference
        )
        region_cfg["_allow_reference_ordering_with_handoff"] = (
            region_index == len(ordered_regions) - 1
        )
        region_cfg["_contour_allow_terminal_short_line"] = bool(
            region_index == len(ordered_regions) - 1
            and exit_reference is None
        )
        if shared_contour_walls is not None:
            region_cfg["_contour_forced_wall_projections"] = shared_contour_walls
        if shared_contour_clearances:
            region_cfg["_contour_forced_turn_clearance_m_by_side"] = (
                shared_contour_clearances
            )
        if previous_endpoint is not None:
            region_cfg["_route_start_reference_local"] = previous_endpoint
        region_path, region_types, region_assessment = _connect_work_lines(
            region.get("work_lines") or [],
            geo=geo,
            state=state,
            min_turn_radius_m=min_turn_radius_m,
            turn_strategy=turn_strategy,
            config=region_cfg,
            return_diagnostics=True,
            global_main_angle=global_main_angle,
            processed_mask=processed_mask,
        )
        handoff_transfer = None
        handoff_transfer_report = None
        handoff_retrace_segment = None
        handoff_candidate_label = "current"
        if (
            region_index > 0
            and normalize_turn_strategy(turn_strategy) == "contour"
            and previous_path
            and global_main_angle is not None
            and math.isfinite(float(global_main_angle))
        ):
            default_gap = _contour_region_transition_gap_m(
                previous_path,
                previous_types,
                region_path,
                region_types,
                main_angle=float(global_main_angle),
                geo=geo,
                state=state,
            )
            minimum_gap = _minimum_contour_region_transition_gap_m(
                cfg,
                state,
            )
            selected_gap = default_gap
            reversed_for_transition = False
            reverse_attempted = False
            if default_gap is not None and default_gap < minimum_gap:
                reverse_attempted = True
                reverse_cfg = dict(region_cfg)
                reverse_cfg["_contour_entry_policy"] = "farthest"
                reversed_path, reversed_types, reversed_assessment = (
                    _connect_work_lines(
                        region.get("work_lines") or [],
                        geo=geo,
                        state=state,
                        min_turn_radius_m=min_turn_radius_m,
                        turn_strategy=turn_strategy,
                        config=reverse_cfg,
                        return_diagnostics=True,
                        global_main_angle=global_main_angle,
                        processed_mask=processed_mask,
                    )
                )
                reverse_gap = _contour_region_transition_gap_m(
                    previous_path,
                    previous_types,
                    reversed_path,
                    reversed_types,
                    main_angle=float(global_main_angle),
                    geo=geo,
                    state=state,
                )
                if (
                    reverse_gap is not None
                    and (default_gap is None or reverse_gap > default_gap + 1e-6)
                ):
                    region_path = reversed_path
                    region_types = reversed_types
                    region_assessment = reversed_assessment
                    selected_gap = reverse_gap
                    reversed_for_transition = True
            if selected_gap is not None and selected_gap < minimum_gap:
                region_assessment.setdefault("hard_reasons", []).append(
                    "cross-region contour transition remains too close to the "
                    f"previous work line ({selected_gap:.2f}m < {minimum_gap:.2f}m)"
                )
                region_assessment["feasible"] = False
                region_assessment["needs_confirmation"] = True
            contour_ordering = region_assessment.setdefault(
                "contour_ordering", {}
            )
            contour_ordering.update({
                "cross_region_transition_spacing_m": selected_gap,
                "cross_region_transition_min_spacing_m": minimum_gap,
                "cross_region_transition_reverse_attempted": reverse_attempted,
                "cross_region_transition_reversed": reversed_for_transition,
            })
            cross_region_transition_records.append({
                "from_region_id": ordered_regions[region_index - 1].get(
                    "region_id"
                ),
                "to_region_id": region.get("region_id"),
                "default_spacing_m": default_gap,
                "selected_spacing_m": selected_gap,
                "minimum_spacing_m": minimum_gap,
                "reverse_attempted": reverse_attempted,
                "reversed": reversed_for_transition,
            })
        if (
            region_index > 0
            and normalize_turn_strategy(turn_strategy) == "contour"
            and previous_path
        ):
            # A region can satisfy the no-adjacent-row invariant in more than
            # one cyclic entry/orientation.  The old code selected the route
            # only by normal spacing, so a valid contour could still enter the
            # next planting block from the opposite headland wall and create a
            # long diagonal transfer.  Evaluate the available contour entry
            # policies against the actual transfer tangents and keep a same-
            # wall candidate whenever one exists.
            handoff_candidates = [
                (region_path, region_types, region_assessment, "current")
            ]
            candidate_policies = ("nearest", "farthest")
            for policy in candidate_policies:
                candidate_cfg = dict(region_cfg)
                candidate_cfg["_contour_entry_policy"] = policy
                candidate_cfg["_preserve_curve_on_boundary_failure"] = True
                candidate_path, candidate_types, candidate_assessment = (
                    _connect_work_lines(
                        region.get("work_lines") or [],
                        geo=geo,
                        state=state,
                        min_turn_radius_m=min_turn_radius_m,
                        turn_strategy=turn_strategy,
                        config=candidate_cfg,
                        return_diagnostics=True,
                        global_main_angle=global_main_angle,
                        processed_mask=processed_mask,
                    )
                )
                signature = (
                    tuple(
                        tuple(
                            (round(float(point[0]), 6), round(float(point[1]), 6))
                            for point in segment
                        )
                        for segment in candidate_path
                    ),
                    tuple(candidate_types),
                )
                if not any(
                    signature == (
                        tuple(
                            tuple(
                                (round(float(point[0]), 6), round(float(point[1]), 6))
                                for point in segment
                            )
                            for segment in existing_path
                        ),
                        tuple(existing_types),
                    )
                    for existing_path, existing_types, _existing_assessment, _label
                    in handoff_candidates
                ):
                    handoff_candidates.append(
                        (
                            candidate_path,
                            candidate_types,
                            candidate_assessment,
                            policy,
                        )
                    )

            transfer_cfg = dict(cfg)
            transfer_cfg["_preserve_curve_on_boundary_failure"] = True
            if global_main_angle is not None and math.isfinite(float(global_main_angle)):
                transfer_cfg["_global_main_angle"] = float(global_main_angle)

            scored_handoffs = []
            for candidate_index, (
                candidate_path,
                candidate_types,
                candidate_assessment,
                candidate_label,
            ) in enumerate(handoff_candidates):
                candidate_transfer, candidate_report = _build_inter_region_transfer(
                    previous_path,
                    previous_types,
                    candidate_path,
                    candidate_types,
                    geo=geo,
                    state=state,
                    min_turn_radius_m=min_turn_radius_m,
                    requested_strategy=turn_strategy,
                    config=transfer_cfg,
                    global_main_angle=global_main_angle,
                )
                transfer_options = [(
                    candidate_transfer,
                    candidate_report,
                    None,
                    candidate_label,
                )]
                if candidate_report.get("same_headland_side") is False:
                    retrace = _region_route_last_work_retrace(
                        previous_path, previous_types
                    )
                    if retrace is not None:
                        retraced_transfer, retraced_report = (
                            _build_inter_region_transfer(
                                previous_path,
                                previous_types,
                                candidate_path,
                                candidate_types,
                                geo=geo,
                                state=state,
                                min_turn_radius_m=min_turn_radius_m,
                                requested_strategy=turn_strategy,
                                config=transfer_cfg,
                                global_main_angle=global_main_angle,
                                previous_work_retrace=retrace,
                            )
                        )
                        transfer_options.append((
                            retraced_transfer,
                            retraced_report,
                            retrace,
                            f"{candidate_label}_after_retrace",
                        ))
                contour_data = candidate_assessment.get("contour_ordering", {})
                adjacent_count = int(
                    contour_data.get("adjacent_transition_count", 0) or 0
                )
                candidate_gap = None
                gap_violation = 0
                if global_main_angle is not None and math.isfinite(float(global_main_angle)):
                    candidate_gap = _contour_region_transition_gap_m(
                        previous_path,
                        previous_types,
                        candidate_path,
                        candidate_types,
                        main_angle=float(global_main_angle),
                        geo=geo,
                        state=state,
                    )
                    if (
                        candidate_gap is not None
                        and candidate_gap < minimum_gap
                    ):
                        gap_violation = 1
                for option_index, (
                    option_transfer,
                    option_report,
                    option_retrace,
                    option_label,
                ) in enumerate(transfer_options):
                    tangent_values = [
                        value
                        for value in (
                            option_report.get("start_heading_alignment_cos"),
                            option_report.get("end_heading_alignment_cos"),
                        )
                        if value is not None and math.isfinite(float(value))
                    ]
                    tangent_quality = min(tangent_values, default=-1.0)
                    same_side = option_report.get("same_headland_side")
                    same_side_rank = (
                        0 if same_side is True
                        else 1 if same_side is None
                        else 2
                    )
                    total_transfer_distance = float(
                        option_report.get("distance_m", float("inf"))
                    ) + float(option_report.get("retrace_distance_m", 0.0))
                    scored_handoffs.append({
                        "score": (
                            same_side_rank,
                            gap_violation,
                            0 if option_report.get("valid") else 1,
                            0 if option_report.get("non_degenerate_curve") else 1,
                            adjacent_count,
                            0 if option_retrace is None else 1,
                            -float(tangent_quality),
                            total_transfer_distance,
                            candidate_index,
                            option_index,
                        ),
                        "path": candidate_path,
                        "types": candidate_types,
                        "assessment": candidate_assessment,
                        "label": option_label,
                        "transfer": option_transfer,
                        "report": option_report,
                        "retrace": option_retrace,
                    })
            if scored_handoffs:
                selected_handoff = min(
                    scored_handoffs,
                    key=lambda item: item["score"],
                )
                region_path = selected_handoff["path"]
                region_types = selected_handoff["types"]
                region_assessment = selected_handoff["assessment"]
                handoff_candidate_label = selected_handoff["label"]
                handoff_transfer = selected_handoff["transfer"]
                handoff_transfer_report = selected_handoff["report"]
                handoff_retrace_segment = selected_handoff["retrace"]
                contour_ordering = region_assessment.setdefault(
                    "contour_ordering", {}
                )
                contour_ordering["handoff_entry_candidate"] = (
                    handoff_candidate_label
                )
                contour_ordering["handoff_same_headland_side"] = (
                    handoff_transfer_report.get("same_headland_side")
                )
                contour_ordering["handoff_retrace_previous_work"] = bool(
                    handoff_retrace_segment
                )
                if global_main_angle is not None and math.isfinite(float(global_main_angle)):
                    selected_gap = _contour_region_transition_gap_m(
                        previous_path,
                        previous_types,
                        region_path,
                        region_types,
                        main_angle=float(global_main_angle),
                        geo=geo,
                        state=state,
                    )
                    contour_ordering["cross_region_transition_spacing_m"] = (
                        selected_gap
                    )
                    if cross_region_transition_records:
                        cross_region_transition_records[-1].update({
                            "selected_spacing_m": selected_gap,
                            "handoff_entry_candidate": handoff_candidate_label,
                            "same_headland_side": handoff_transfer_report.get(
                                "same_headland_side"
                            ),
                        })
        if region_index > 0:
            if handoff_transfer is None or handoff_transfer_report is None:
                transfer, transfer_report = _build_inter_region_transfer(
                    previous_path,
                    previous_types,
                    region_path,
                    region_types,
                    geo=geo,
                    state=state,
                    min_turn_radius_m=min_turn_radius_m,
                    requested_strategy=turn_strategy,
                    config=cfg,
                    global_main_angle=global_main_angle,
                )
            else:
                transfer = handoff_transfer
                transfer_report = handoff_transfer_report
            transfer_report.update({
                "from_region_id": ordered_regions[region_index - 1].get("region_id"),
                "to_region_id": region.get("region_id"),
                "handoff_entry_candidate": handoff_candidate_label,
            })
            if handoff_retrace_segment:
                transfer_report["retrace_segment_index"] = len(combined_path)
                combined_path.append(handoff_retrace_segment)
                combined_types.append("turn_reverse")
            transfer_report["segment_index"] = len(combined_path)
            if transfer:
                combined_path.append(transfer)
                combined_types.append("turn_aux")
            transfer_reports.append(transfer_report)
        segment_offset = len(combined_path)
        combined_path.extend(region_path)
        combined_types.extend(region_types)
        region_reports.append({
            "region_id": region.get("region_id"),
            "kind": region.get("kind"),
            "line_count": int(region.get("line_count", 0)),
            "source_region_ids": list(
                region.get("source_region_ids") or [region.get("region_id")]
            ),
            "segment_start_index": segment_offset,
            "segment_end_index": len(combined_path) - 1,
            "assessment": region_assessment,
        })
        previous_path = region_path
        previous_types = region_types
        previous_endpoint = _region_route_endpoint(
            region_path, region_types, first=False
        )

    pass_spacings = [
        float(value)
        for report in region_reports
        for value in (report["assessment"].get("pass_spacings_m") or [])
    ]
    assessment = assess_turn_strategy(
        turn_strategy,
        pass_spacings,
        min_turn_radius_m,
    )
    hard_reasons = []
    fallback_reasons = []
    used_strategies = []
    turn_decisions = []
    boundary_adjustments = []
    non_work_violations = []
    display_only_connectors = []
    for report in region_reports:
        region_id = report["region_id"]
        item = report["assessment"]
        hard_reasons.extend(
            f"{region_id}: {reason}" for reason in item.get("hard_reasons", [])
        )
        fallback_reasons.extend(item.get("fallback_reasons", []) or [])
        used_strategies.extend(item.get("used_strategies", []) or [])
        for decision in item.get("turn_decisions", []) or []:
            bound = dict(decision)
            bound["region_id"] = region_id
            turn_decisions.append(bound)
        constraints = item.get("boundary_constraints") or {}
        boundary_adjustments.extend(constraints.get("adjustment_reasons") or [])
        non_work_violations.extend(constraints.get("non_work_violations") or [])
        display_only_connectors.extend(item.get("display_only_connectors") or [])
    for transfer in transfer_reports:
        if transfer.get("strategy"):
            used_strategies.append(transfer["strategy"])
        hard_reasons.extend(
            f"{transfer.get('from_region_id')}->{transfer.get('to_region_id')}: {reason}"
            for reason in transfer.get("hard_reasons", [])
        )
        if transfer.get("boundary_adjustment_reason"):
            boundary_adjustments.append(transfer["boundary_adjustment_reason"])
        if not transfer.get("machine_executable", False):
            display_only_connectors.append({
                "segment_index": transfer.get("segment_index"),
                "display_only_connector": True,
                "kinematically_infeasible": True,
                "reason": "; ".join(transfer.get("hard_reasons") or []),
                "role": "inter_region_transfer",
            })
    assessment.update({
        "requested_strategy": normalize_turn_strategy(turn_strategy),
        "ordering_strategy": normalize_turn_strategy(turn_strategy),
        "contour_ordering": {
            "ordering": "classified_regions_with_small_region_execution_merge",
            "algorithm_version": "mixed-region-order-v6",
            "region_order": [item.get("region_id") for item in ordered_regions],
            "region_order_rule": region_order_rule,
            "entry_reference_local": list(entry_reference)
            if entry_reference is not None else None,
            "entry_region_distances_px": {
                str(item.get("region_id")): float(region_reference_distance(item))
                for item in ordered_regions
            } if entry_reference is not None else {},
            "exit_reference_local": list(exit_reference)
            if exit_reference is not None else None,
            "exit_region_distances_px": {
                str(item.get("region_id")): float(region_end_distance(item))
                for item in ordered_regions
            } if exit_reference is not None else {},
            "reference_ordering": region_reference_ordering,
            "no_cross_region_pairing": not bool(small_region_merges),
            "small_region_execution_merges": small_region_merges,
            "shared_headland_wall_projections_px": shared_contour_walls,
            "shared_turn_clearance_m_by_side": shared_contour_clearances,
            "cross_region_transition_records": cross_region_transition_records,
        },
        "used_strategies": used_strategies,
        "fallback_reasons": fallback_reasons,
        "display_only_connectors": display_only_connectors,
        "kinematically_infeasible": bool(display_only_connectors),
        "machine_executable": not bool(hard_reasons or display_only_connectors),
        "hard_reasons": hard_reasons,
        "feasible": not hard_reasons,
        "needs_confirmation": bool(hard_reasons),
        "boundary_constraints": {
            "field_boundary_present": _field_boundary_local_polygon(state) is not None,
            "turns_adjusted": len(boundary_adjustments),
            "adjustment_reasons": boundary_adjustments,
            "non_work_violations": non_work_violations,
        },
        "pass_spacings_m": pass_spacings,
        "turn_decisions": turn_decisions,
        "endpoint_alignment": {"regions": [
            report["assessment"].get("endpoint_alignment", {})
            for report in region_reports
        ]},
        "contour_connector": {
            "enabled": normalize_turn_strategy(turn_strategy) == "contour",
            "algorithm_version": "mixed-region-contours-v1",
            "regions": [
                report["assessment"].get("contour_connector", {})
                for report in region_reports
            ],
        },
        "contour_route_mode": bool(region_reports) and all(
            report["assessment"].get("contour_route_mode", False)
            for report in region_reports
        ),
        "region_routes": region_reports,
        "region_transfers": transfer_reports,
        "service_points": {
            "entry_present": bool(
                (region_reports[0]["assessment"].get("service_points") or {}).get(
                    "entry_present", False
                )
            ),
            "exit_present": bool(
                (region_reports[-1]["assessment"].get("service_points") or {}).get(
                    "exit_present", False
                )
            ),
            "unload_count": int(
                (region_reports[-1]["assessment"].get("service_points") or {}).get(
                    "unload_count", 0
                )
            ),
            "unload_visit_count": sum(
                segment_type == "unload" for segment_type in combined_types
            ),
            "exit_route": (
                region_reports[-1]["assessment"].get("service_points") or {}
            ).get("exit_route"),
        },
    })
    return combined_path, combined_types, assessment, ordered_regions


def _connect_work_regions(
    work_regions: List[Dict],
    *,
    geo=None,
    state=None,
    min_turn_radius_m: float,
    turn_strategy: str,
    config: Optional[Dict] = None,
    global_main_angle: Optional[float] = None,
    processed_mask: Optional[np.ndarray] = None,
) -> Tuple[List[List[Tuple[float, float]]], List[str], Dict, List[Dict]]:
    """DECISION-017: generate region lines first, then connect them globally.

    Planting regions remain the provenance boundary for line density and line
    geometry.  They are deliberately *not* execution groups: every generated
    line is flattened into one field-wide set before DECISION-016 ordering,
    direction selection, shared-headland construction and turn generation.
    """
    source_regions = [dict(item) for item in (work_regions or [])]
    global_work_lines: List[List[Tuple[float, float]]] = []
    line_provenance = []
    region_generation = []
    for region in source_regions:
        region_id = region.get("region_id")
        lines = [
            list(line)
            for line in (region.get("work_lines") or [])
            if len(line) >= 2
        ]
        region_generation.append({
            "region_id": region_id,
            "kind": region.get("kind"),
            "line_source": region.get("line_source"),
            "generated_line_count": len(lines),
            "normal_position": region.get("normal_position"),
        })
        for region_line_index, line in enumerate(lines):
            line_provenance.append({
                "global_input_index": len(global_work_lines),
                "region_id": region_id,
                "region_line_index": region_line_index,
                "kind": region.get("kind"),
                "line_source": region.get("line_source"),
            })
            global_work_lines.append(line)

    global_config = dict(config or {})
    if normalize_turn_strategy(turn_strategy) == "contour":
        # DECISION-016/017: endpoint evidence is now audited across the one
        # field-wide set.  A uniquely short, boundary-blocked line may finish
        # the route, but may never sit mid-route and force a diagonal turn to
        # the next full-length line.
        global_config["_contour_allow_terminal_short_line"] = True
    full_path, segment_types, assessment = _connect_work_lines(
        global_work_lines,
        geo=geo,
        state=state,
        min_turn_radius_m=min_turn_radius_m,
        turn_strategy=turn_strategy,
        config=global_config,
        return_diagnostics=True,
        global_main_angle=global_main_angle,
        processed_mask=processed_mask,
    )
    assessment = dict(assessment or {})
    contour_ordering = dict(assessment.get("contour_ordering") or {})
    contour_ordering.update({
        "execution_scope": "all_generated_work_lines",
        "execution_topology_version": "global-after-region-generation-v1",
        "region_boundaries_affect_turns": False,
        "global_line_count": len(global_work_lines),
        "source_region_ids": [
            item.get("region_id") for item in source_regions
        ],
        "line_provenance": line_provenance,
    })
    assessment.update({
        "execution_scope": "all_generated_work_lines",
        "region_boundaries_affect_turns": False,
        "region_generation": region_generation,
        "region_routes": [],
        "region_transfers": [],
        "contour_ordering": contour_ordering,
    })
    return full_path, segment_types, assessment, source_regions


def _apply_operator_work_band_selection(
    work_regions,
    layout,
    main_angle,
    processed_mask,
    planned_mask,
    *,
    geo=None,
    state=None,
    config=None,
):
    """Bind operator work-band selection to geometry and validation scope."""
    from work_band_selection import (
        apply_work_band_selection,
        rasterize_work_band_coverage,
    )

    cfg = dict(config or {})
    explicit = "selected_work_band_ids" in cfg
    requested = cfg.get("selected_work_band_ids") if explicit else None
    selected_regions, catalog, report = apply_work_band_selection(
        work_regions,
        main_angle,
        selected_ids=requested,
    )
    selected_lines = [
        list(line)
        for region in selected_regions
        for line in (region.get("work_lines") or [])
    ]
    selected_mask = planned_mask
    if explicit and selected_lines:
        support = cfg.get("planning_support_mask")
        if not isinstance(support, np.ndarray):
            support = processed_mask
        selected_mask = rasterize_work_band_coverage(
            catalog,
            report["selected_ids"],
            shape=processed_mask.shape[:2],
            cutter_width_m=float(_harvester_specs(state)["cutter_width_m"]),
            meters_per_pixel=meters_per_pixel(processed_mask, geo, state),
            support_mask=support,
        )
    updated_layout = dict(layout or {})
    updated_layout["work_band_catalog"] = catalog
    updated_layout["work_band_selection"] = report
    updated_layout["work_regions"] = [
        {key: value for key, value in region.items() if key != "work_lines"}
        for region in selected_regions
    ]
    updated_layout["selected_work_line_count"] = len(selected_lines)
    updated_layout["planning_scope"] = (
        "operator_selected_work_bands" if explicit else "whole_field_default"
    )
    return selected_lines, selected_regions, selected_mask, updated_layout


def prepare_operator_work_band_plan(
    wide_bands,
    processed_mask,
    main_angle,
    *,
    geo=None,
    state=None,
    config=None,
    progress_callback=None,
    detected_mask=None,
    all_bands=None,
):
    """Generate the final selectable work-line catalog before turn planning."""
    cfg = Config().section("path_planning")
    cfg.update(dict(config or {}))
    validation_profile = str(cfg.get("validation_profile", "field_trial"))
    if validation_profile in {"field_trial", "machine_candidate"}:
        cfg["recover_missing_work_bands"] = False

    planning_bands, recovery = _recover_lattice_supported_work_bands(
        wide_bands,
        all_bands,
        processed_mask,
        main_angle,
        geo=geo,
        state=state,
        config=cfg,
    )
    planning_bands, boundary_filter = _filter_work_bands_to_selection(
        planning_bands,
        processed_mask,
        main_angle,
        geo=geo,
        state=state,
        config=cfg,
    )
    if boundary_filter.get("excluded"):
        kept_ids = set(boundary_filter.get("kept_band_ids") or [])
        recovery["recovered_band_ids"] = [
            band_id
            for band_id in (recovery.get("recovered_band_ids") or [])
            if band_id in kept_ids
        ]
        recovery["selection_filter"] = boundary_filter
    cfg["_selection_work_band_filter"] = boundary_filter

    planned_mask = build_band_mask(processed_mask.shape, planning_bands)
    work_lines, layout, work_regions = _prepare_work_line_layout(
        planning_bands,
        processed_mask,
        main_angle,
        geo=geo,
        state=state,
        config=cfg,
        progress_callback=progress_callback,
        detected_mask=detected_mask,
        planned_mask=planned_mask,
        reference_bands=all_bands,
        return_regions=True,
    )
    layout = dict(layout or {})
    layout["coverage_gap_recovery"] = recovery
    layout["planning_recovered"] = bool(recovery.get("recovered_band_ids"))
    work_lines, work_regions, planned_mask, layout = (
        _apply_operator_work_band_selection(
            work_regions,
            layout,
            main_angle,
            processed_mask,
            planned_mask,
            geo=geo,
            state=state,
            config=cfg,
        )
    )
    return {
        "work_lines": work_lines,
        "work_regions": work_regions,
        "planned_mask": planned_mask,
        "planning_bands": planning_bands,
        "layout": layout,
        "catalog": dict(layout.get("work_band_catalog") or {}),
        "selection": dict(layout.get("work_band_selection") or {}),
        "executed_config": cfg,
    }


def _apply_operator_centerline_selection(
    work_regions,
    layout,
    main_angle,
    processed_mask,
    planned_mask,
    *,
    geo=None,
    state=None,
    config=None,
):
    """Apply left-panel centerline checkboxes after all lines are generated."""
    from work_band_selection import (
        apply_centerline_selection,
        rasterize_centerline_coverage,
    )

    cfg = dict(config or {})
    explicit = (
        "selected_centerline_ids" in cfg
        or "selected_work_band_ids" in cfg
    )
    requested = cfg.get("selected_centerline_ids")
    if requested is None and "selected_work_band_ids" in cfg:
        requested = cfg.get("selected_work_band_ids")
    selected_regions, catalog, report = apply_centerline_selection(
        work_regions,
        main_angle,
        selected_ids=requested if explicit else None,
    )
    selected_lines = [
        list(line)
        for region in selected_regions
        for line in (region.get("work_lines") or [])
    ]
    selected_mask = planned_mask
    if explicit and selected_lines:
        support = cfg.get("planning_support_mask")
        if not isinstance(support, np.ndarray):
            support = processed_mask
        selected_mask = rasterize_centerline_coverage(
            catalog,
            report["selected_ids"],
            shape=processed_mask.shape[:2],
            cutter_width_m=float(_harvester_specs(state)["cutter_width_m"]),
            meters_per_pixel=meters_per_pixel(processed_mask, geo, state),
            support_mask=support,
        )
    updated_layout = dict(layout or {})
    updated_layout["centerline_catalog"] = catalog
    updated_layout["centerline_selection"] = report
    # Keep old cache/layout readers able to inspect the same selection without
    # reintroducing the old canvas-selection UI.
    updated_layout["work_band_catalog"] = catalog
    updated_layout["work_band_selection"] = report
    updated_layout["work_regions"] = [
        {key: value for key, value in region.items() if key != "work_lines"}
        for region in selected_regions
    ]
    updated_layout["selected_work_line_count"] = len(selected_lines)
    updated_layout["selected_centerline_count"] = len(selected_lines)
    updated_layout["planning_scope"] = (
        "operator_selected_centerlines" if explicit else "whole_field_default"
    )
    return selected_lines, selected_regions, selected_mask, updated_layout


def prepare_operator_centerline_plan(
    wide_bands,
    processed_mask,
    main_angle,
    *,
    geo=None,
    state=None,
    config=None,
    progress_callback=None,
    detected_mask=None,
    all_bands=None,
):
    """Generate every candidate centerline before connecting any turn."""
    cfg = Config().section("path_planning")
    cfg.update(dict(config or {}))
    validation_profile = str(cfg.get("validation_profile", "field_trial"))
    if validation_profile in {"field_trial", "machine_candidate"}:
        cfg["recover_missing_work_bands"] = False

    planning_bands, recovery = _recover_lattice_supported_work_bands(
        wide_bands,
        all_bands,
        processed_mask,
        main_angle,
        geo=geo,
        state=state,
        config=cfg,
    )
    planning_bands, boundary_filter = _filter_work_bands_to_selection(
        planning_bands,
        processed_mask,
        main_angle,
        geo=geo,
        state=state,
        config=cfg,
    )
    if boundary_filter.get("excluded"):
        kept_ids = set(boundary_filter.get("kept_band_ids") or [])
        recovery["recovered_band_ids"] = [
            band_id
            for band_id in (recovery.get("recovered_band_ids") or [])
            if band_id in kept_ids
        ]
        recovery["selection_filter"] = boundary_filter
    cfg["_selection_work_band_filter"] = boundary_filter

    planned_mask = build_band_mask(processed_mask.shape, planning_bands)
    work_lines, layout, work_regions = _prepare_work_line_layout(
        planning_bands,
        processed_mask,
        main_angle,
        geo=geo,
        state=state,
        config=cfg,
        progress_callback=progress_callback,
        detected_mask=detected_mask,
        planned_mask=planned_mask,
        reference_bands=all_bands,
        return_regions=True,
    )
    layout = dict(layout or {})
    field_polygon = (
        field_polygon_local(state, processed_mask.shape[:2])
        if state is not None
        else None
    )
    work_regions, centerline_boundary_filter = (
        _filter_generated_centerline_regions_to_field(
            work_regions,
            field_polygon,
        )
    )
    layout["centerline_field_boundary_filter"] = centerline_boundary_filter
    layout["work_regions"] = [
        {key: value for key, value in region.items() if key != "work_lines"}
        for region in work_regions
    ]
    layout["coverage_gap_recovery"] = recovery
    layout["planning_recovered"] = bool(recovery.get("recovered_band_ids"))
    selected_lines, selected_regions, planned_mask, layout = (
        _apply_operator_centerline_selection(
            work_regions,
            layout,
            main_angle,
            processed_mask,
            planned_mask,
            geo=geo,
            state=state,
            config=cfg,
        )
    )
    return {
        "work_lines": selected_lines,
        "work_regions": selected_regions,
        "planned_mask": planned_mask,
        "planning_bands": planning_bands,
        "layout": layout,
        "catalog": dict(layout.get("centerline_catalog") or {}),
        "selection": dict(layout.get("centerline_selection") or {}),
        "executed_config": cfg,
    }


def plan_path(wide_bands: List[Dict],
              processed_mask: np.ndarray,
              main_angle: float,
              geo=None, state=None,
              config: Optional[Dict] = None,
              progress_callback=None,
              obstacle_mask: Optional[np.ndarray] = None,
              detected_mask: Optional[np.ndarray] = None,
              planned_mask: Optional[np.ndarray] = None,
              all_bands: Optional[List[Dict]] = None) -> Dict:
    """完整的路径规划管道。

    Args:
        wide_bands: 4行带列表（已提取 centerline）
        processed_mask: 处理后的掩膜
        main_angle: 主方向角度 (rad)
        geo: GeoUtils
        state: AppState
        config: 可选配置，支持键:
            - min_turn_radius_m: 最小转弯半径 (米)
            - turn_strategy: 转弯策略
              ("bow"/"semicircle"/"pear"/"fishtail")
        progress_callback: 可选回调 callback(progress: float, stage: str)

    Returns:
        {
            "full_path": [[(x,y),...], ...],  # 偶数=工作段, 奇数=转弯段
            "geo_points": [{"lon","lat",...}, ...],
            "validation": {...},
            "is_valid": bool,
            "description": str,
        }
    """
    # DECISION-012: the selected path, pose sampling and footprint metrics
    # remain one mathematical chain; no turn-strategy ranking is performed.
    def _report(pct, stage):
        if progress_callback:
            try:
                progress_callback(pct, stage)
            except Exception:
                pass

    log = AppLogger()
    cfg_obj = Config()
    cfg = cfg_obj.section("path_planning")
    cfg.update(config or {})
    validation_profile = str(cfg.get("validation_profile", "field_trial"))
    if validation_profile in {"field_trial", "machine_candidate"}:
        # P0: automatic band invention is preview-only for the two formal
        # profiles.  Research users may opt in explicitly in their config.
        cfg["recover_missing_work_bands"] = False
    cancel_callback = cfg.get("_cancel_callback")

    def abort_if_cancelled():
        if callable(cancel_callback) and cancel_callback():
            raise RuntimeError("task cancelled")

    abort_if_cancelled()
    specs = _harvester_specs(state)
    min_turn_radius = float(
        cfg.get("min_turn_radius_m", specs.get("turn_radius_m", 1.0))
    )
    explicit_strategy = "turn_strategy" in (config or {})
    turn_strategy = str(cfg.get("turn_strategy", "bow"))
    if (
        not explicit_strategy
        and state
        and hasattr(state, "turn_strategy")
        and state.turn_strategy
    ):
        turn_strategy = state.turn_strategy
    turn_strategy = (
        "auto"
        if str(turn_strategy).strip().lower() == "auto"
        else normalize_turn_strategy(turn_strategy)
    )

    log.info(
        "=== 足迹约束路径规划开始: "
        f"割台={specs['cutter_width_m']:.2f}m, "
        f"履带={specs['track_width_m']:.2f}m, "
        f"轨距={specs['track_gauge_m']:.2f}m, "
        f"转弯={turn_strategy} ==="
    )

    _report(0.12, "生成全部候选中心线")
    abort_if_cancelled()
    prepared = prepare_operator_centerline_plan(
        wide_bands,
        processed_mask,
        main_angle,
        geo=geo,
        state=state,
        config=cfg,
        progress_callback=progress_callback,
        detected_mask=detected_mask,
        all_bands=all_bands,
    )
    work_lines = prepared["work_lines"]
    work_regions = prepared["work_regions"]
    planned_mask = prepared["planned_mask"]
    planning_bands = prepared["planning_bands"]
    layout = prepared["layout"]
    cfg = prepared["executed_config"]
    # 等距区细分线不参与田头端点校正（已在掩膜边界正确截断）
    abort_if_cancelled()
    if not work_lines:
        classification_reasons = list(
            (layout.get("planting_region_classification") or {}).get(
                "hard_reasons", []
            )
        )
        generation_reasons = list(
            layout.get("region_generation_hard_reasons") or []
        )
        hard_reasons = classification_reasons + generation_reasons
        if not hard_reasons:
            hard_reasons = ["未生成有效作业线"]
        return {
            "full_path": [],
            "segment_types": [],
            "geo_points": [],
            "validation": {"valid": False, "issues": hard_reasons},
            "is_valid": False,
            "description": "未生成有效作业线",
            "layout": layout,
            "machine_executable": False,
            "requested_strategy": turn_strategy,
            "turn_assessment": {
                "strategy": turn_strategy,
                "label": TURN_STRATEGY_LABELS.get(
                    turn_strategy,
                    turn_strategy,
                ),
                "feasible": False,
                "needs_confirmation": True,
                "hard_reasons": hard_reasons,
                "warnings": [],
            },
        }

    _report(0.76, "地头转弯候选生成")

    def evaluate_strategy(strategy_key: str) -> Dict:
        abort_if_cancelled()
        (
            candidate_segments,
            candidate_types,
            candidate_assessment,
            source_regions,
        ) = _connect_work_regions(
            work_regions,
            geo=geo,
            state=state,
            min_turn_radius_m=min_turn_radius,
            turn_strategy=strategy_key,
            config=cfg,
            global_main_angle=main_angle,
            processed_mask=processed_mask,
        )
        layout["region_generation_order"] = [
            {
                "sequence": index + 1,
                "region_id": region.get("region_id"),
                "kind": region.get("kind"),
                "line_count": int(region.get("line_count", 0)),
                "line_source": region.get("line_source"),
            }
            for index, region in enumerate(source_regions)
        ]
        # DECISION-017: region membership ends after work-line generation.
        # There is no regional execution order or inter-region transfer layer.
        layout["region_execution_order"] = []
        layout["region_order_rule"] = "not_applicable_global_turn_planning"
        layout["turn_planning_scope"] = "all_generated_work_lines"
        layout["region_boundaries_affect_turns"] = False
        candidate_assessment["user_requested_strategy"] = turn_strategy
        candidate_assessment["candidate_strategy"] = strategy_key
        for decision in candidate_assessment.get("turn_decisions", []) or []:
            decision["user_requested_strategy"] = turn_strategy
            decision["candidate_strategy"] = strategy_key
            if turn_strategy == "auto" and decision.get("reason") == "用户指定":
                decision["reason"] = "自动规则根据该处几何约束选择固定转弯实现"
        abort_if_cancelled()
        candidate_validation = validate_path(
            candidate_segments,
            obstacle_mask if obstacle_mask is not None else processed_mask,
            main_angle,
            geo,
            state,
            min_turn_radius_m=min_turn_radius,
            config=cfg,
            segment_types=candidate_types,
            detected_mask=detected_mask,
            planned_mask=planned_mask,
        )
        turn_self_crossings = sum(
            _count_polyline_self_intersections(segment)
            for index, segment in enumerate(candidate_segments)
            if (
                candidate_types[index].startswith("turn")
                if index < len(candidate_types)
                else False
            )
        )
        candidate_assessment["turn_self_crossing_count"] = turn_self_crossings
        candidate_assessment["machine_executable"] = bool(
            candidate_assessment.get("machine_executable", True)
            and not candidate_assessment.get("kinematically_infeasible", False)
        )
        # A retained outside-field curve is deliberately non-executable, but
        # it is not a degraded straight connector.  Only attach the legacy
        # display-only diagnosis when the selected geometry actually contains
        # a kinematically infeasible connector.
        if (
            not candidate_assessment["machine_executable"]
            and candidate_assessment.get("kinematically_infeasible", False)
        ):
            candidate_assessment.setdefault("hard_reasons", []).append(
                "display-only straight connector is kinematically infeasible"
            )
        if turn_self_crossings > 0:
            candidate_assessment.setdefault("hard_reasons", []).append(
                f"所选策略生成的转弯轨迹存在{turn_self_crossings}处自交"
            )
        max_outside = float(cfg.get("max_track_outside_field_pct", 2.0))
        outside_pct = float(candidate_validation.get("track_outside_field_pct", 0.0))
        if outside_pct > max_outside:
            candidate_assessment.setdefault("hard_reasons", []).append(
                f"路线履带越界率{outside_pct:.1f}%超过允许值"
                f"{max_outside:.1f}%，当前田头空间不足"
            )
        candidate_assessment["feasible"] = not candidate_assessment.get("hard_reasons")
        candidate_assessment["needs_confirmation"] = bool(candidate_assessment.get("hard_reasons"))
        if candidate_assessment.get("hard_reasons"):
            candidate_validation = dict(candidate_validation)
            candidate_validation["valid"] = False
            issues = list(candidate_validation.get("issues") or [])
            issues.extend(
                f"转弯运动学不可行: {reason}"
                for reason in candidate_assessment["hard_reasons"]
                if f"转弯运动学不可行: {reason}" not in issues
            )
            candidate_validation["issues"] = issues
        return {
            "strategy": strategy_key,
            "user_requested_strategy": turn_strategy,
            "candidate_strategy": strategy_key,
            "effective_strategies": list(
                (candidate_assessment.get("used_strategies") or [])
            ),
            "geometry_signature": hashlib.sha256(
                json.dumps(
                    {
                        "segment_types": candidate_types,
                        "path_segments": [
                            [
                                [round(float(point[0]), 6), round(float(point[1]), 6)]
                                for point in segment
                            ]
                            for segment in candidate_segments
                        ],
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "path_segments": candidate_segments,
            "segment_types": candidate_types,
            "validation": candidate_validation,
            "turn_assessment": candidate_assessment,
        }

    _report(0.84, "固定转弯策略足迹验证")
    candidate_results = []
    for strategy_key in _candidate_strategies_for_request(turn_strategy):
        abort_if_cancelled()
        candidate_results.append(evaluate_strategy(strategy_key))
    # The current scope implements and validates the selected geometry only.
    # Do not compare or rank alternative turn strategies here.
    selected_candidate = candidate_results[0] if candidate_results else evaluate_strategy(turn_strategy)
    ranked_candidates = [selected_candidate]
    path_segments = selected_candidate["path_segments"]
    segment_types = selected_candidate["segment_types"]
    validation = selected_candidate["validation"]
    turn_assessment = selected_candidate["turn_assessment"]
    turn_assessment["selected_candidate_strategy"] = selected_candidate.get("strategy")
    turn_assessment["user_requested_strategy"] = turn_strategy
    turn_assessment["candidate_strategy"] = selected_candidate.get("candidate_strategy")
    turn_assessment["effective_strategies"] = list(
        selected_candidate.get("effective_strategies") or []
    )
    turn_assessment["geometry_signature"] = selected_candidate.get("geometry_signature", "")
    turn_assessment["candidate_count"] = len(ranked_candidates)
    endpoint_alignment = turn_assessment.get("endpoint_alignment", {}) or {}
    if endpoint_alignment.get("corrected_endpoint_count", 0):
        log.info(
            "异常短作业线端点校正: "
            f"已校正 {endpoint_alignment.get('corrected_endpoint_count', 0)} 个端点, "
            f"阻止 {endpoint_alignment.get('blocked_endpoint_count', 0)} 个端点"
        )
    for decision in turn_assessment.get("turn_decisions", []) or []:
        log.info(
            f"转弯 {int(decision.get('transition_index', 0)) + 1}: "
            f"ω={float(decision.get('pass_spacing_m', 0.0)):.2f}m, "
            f"R={float(decision.get('turn_radius_m', 0.0)):.2f}m, "
            f"策略={TURN_STRATEGY_LABELS.get(str(decision.get('strategy', '')), decision.get('strategy', ''))}, "
            f"原因={decision.get('reason', '')}"
        )

    description = (
        f"履带核心重叠 {validation['track_core_overlap_pct']:.1f}%, "
        f"全田作物预计碾压 {validation['rolling_canopy_pct']:.1f}%, "
        f"主体碾压 {validation['rolling_body_pct']:.1f}%, "
        f"田头碾压 {float(validation.get('rolling_headland_pct') or 0.0):.1f}%, "
        f"作物核心碾压 {validation['rolling_core_pct']:.1f}%, "
        f"规划目标几何覆盖 {validation['planned_target_geometric_coverage_pct']:.1f}%, "
        f"作物掩膜几何覆盖 {validation['crop_mask_geometric_coverage_pct']:.1f}%, "
        f"原始识别掩膜几何覆盖 {validation['detected_mask_geometric_coverage_pct']:.1f}%, "
        f"越界 {validation['track_outside_field_pct']:.1f}%"
    )
    if validation["issues"]:
        description += "; " + "; ".join(validation["issues"])
        issue_text = " ".join(validation["issues"])
        advice = None
        if "履带与稻株核心重叠" in issue_text:
            advice = (
                "作业段履带与稻株重叠超限，请检查轨距、种植带分类"
                "或在编辑路线中横向微调作业线"
            )
        if advice:
            validation["issues"].append(advice)
            description += "; " + advice

    _report(0.94, "坐标转换")
    geo_points = path_to_geo(path_segments, geo, state, segment_types)
    display_path_segments = _path_to_global_coords(path_segments, state)
    offset_x = float(getattr(state, "mask_offset_x", 0) or 0) if state is not None else 0.0
    offset_y = float(getattr(state, "mask_offset_y", 0) or 0) if state is not None else 0.0
    segment_lengths_m = [
        sum(
            math.hypot(segment[i][0] - segment[i - 1][0], segment[i][1] - segment[i - 1][1])
            for i in range(1, len(segment))
        ) * meters_per_pixel(processed_mask, geo, state)
        for segment in path_segments
    ]
    log.info(
        f"足迹验证: valid={validation['valid']}, {description}, "
        f"总长={validation['total_length_m']:.1f}m"
    )
    _report(1.0, "完成")
    planning_factors = _build_planning_factor_report(
        state=state,
        work_lines=work_lines,
        layout=layout,
        turn_assessment=turn_assessment,
        validation=validation,
    )
    result = {
        "full_path": display_path_segments,
        "segment_types": segment_types,
        "segment_lengths_m": segment_lengths_m,
        "geo_points": geo_points,
        "validation": validation,
        "is_valid": validation["valid"],
        "description": description,
        "layout": layout,
        "planning_factors": planning_factors,
        "coordinate_space": {
            "planner_segments": "local_mask_pixel",
            "validation_segments": "local_mask_pixel",
            "display_segments": "global_display_pixel",
            "geo_points_source": "local_mask_pixel_plus_offset",
            "mask_offset_xy": [offset_x, offset_y],
        },
        "path_candidates": [
            {
                "strategy": item.get("strategy"),
                "user_requested_strategy": item.get("user_requested_strategy", turn_strategy),
                "candidate_strategy": item.get("candidate_strategy", item.get("strategy")),
                "effective_strategies": item.get("effective_strategies", []),
                "geometry_signature": item.get("geometry_signature", ""),
                "machine_executable": bool(
                    (item.get("turn_assessment") or {}).get("machine_executable", False)
                ),
                "validation": item.get("validation", {}),
                "turn_assessment": item.get("turn_assessment", {}),
            }
            for item in ranked_candidates
        ],
        "requested_strategy": turn_strategy,
        "turn_assessment": turn_assessment,
    }
    from machine_route_validator import assess_machine_readiness
    result["machine_readiness"] = assess_machine_readiness(
        result,
        execution_config=cfg,
    )
    result["machine_executable"] = bool(
        result["machine_readiness"].get("machine_executable") is True
    )
    return result

def _turn_parts_for_candidate(start, end, work_direction, turn_sign, mpp, min_turn_radius_m, selected_strategy, clearance_m):
    if selected_strategy in ("fishtail", "alpha"):
        return _turn_reverse_segments(
            start, end, work_direction, mpp, min_turn_radius_m, clearance_m,
            inner=selected_strategy == "alpha",
        )
    turn_function = TURN_STRATEGIES.get(selected_strategy, _turn_bow)
    return [(turn_function(
        start, end, work_direction, turn_sign, mpp, min_turn_radius_m, clearance_m
    ), "turn")]

def _turn_parts_length_m(parts, mpp):
    return sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for path, _segment_type in parts
        for a, b in zip(path, path[1:])
    ) * float(mpp)

def _turn_parts_have_curve(parts):
    for path, segment_type in parts:
        if str(segment_type) == "turn_reverse":
            continue
        if has_non_degenerate_curve([path]):
            return True
    return False

def _validate_fixed_headland_turn(
    base_parts, start, end, work_direction, turn_sign, selected_strategy,
    *, mpp, min_turn_radius_m, clearance_m, processed_mask, field_polygon,
    geo, state, config, validation_work_lines=None, validation_turn_index=None,
):
    """Validate one fixed strategy geometry without searching headland offsets."""
    cfg = dict(config or {})
    diagnostics = {
        "validator_version": "fixed-headland-turn-validation-v1",
        "validation_passed": False,
        "fallback_reason": "",
        "candidate_count": 0,
        "safe_candidate_count": 0,
    }
    if field_polygon is None:
        diagnostics["fallback_reason"] = "field_boundary_missing"
        return base_parts, diagnostics
    if processed_mask is None:
        diagnostics["fallback_reason"] = "processed_mask_missing"
        return base_parts, diagnostics
    headland = cfg.get("headland_mask")
    if headland is None or not np.any(np.asarray(headland) > 0):
        diagnostics["fallback_reason"] = "headland_mask_missing"
        return base_parts, diagnostics
    try:
        metric_scale = float(mpp)
    except (TypeError, ValueError):
        metric_scale = float("nan")
    if not math.isfinite(metric_scale) or metric_scale <= 0:
        diagnostics["fallback_reason"] = "metric_scale_missing"
        return base_parts, diagnostics
    shape = np.asarray(processed_mask).shape[:2]
    corridor, corridor_diag = build_turn_corridor_mask(
        shape,
        field_boundary=field_polygon,
        planning_support_mask=cfg.get("planning_support_mask"),
        headland_mask=headland,
        uncertain_residual_mask=cfg.get("uncertain_mask"),
        forbidden_mask=cfg.get("forbidden_mask"),
        endpoints=[start, end],
        corridor_radius_px=(
            2.0 * float(min_turn_radius_m) + abs(float(clearance_m)) + 3.0
        ) / metric_scale,
    )
    diagnostics["corridor"] = corridor_diag
    if not np.any(corridor > 0):
        diagnostics["fallback_reason"] = "empty_turn_corridor"
        return base_parts, diagnostics
    # The selected strategy and requested clearance define the geometry.  Do
    # not search alternative offsets here: headland/crop evidence is report
    # data and must not move a turn.  The fixed geometry only passes through
    # the existing hard-constraint validation below.
    offset = float(clearance_m)
    hard_reasons = []
    parts = base_parts
    paths = [path for path, _segment_type in parts]
    if not paths or not _turn_parts_have_curve(parts):
        hard_reasons.append("display_only_connector")
    if paths and any(
        not _segment_inside_field_boundary(path, field_polygon) for path in paths
    ):
        hard_reasons.append("outside_field")
    if paths and not polyline_inside_mask(paths, corridor):
        hard_reasons.append("corridor_violation")
    validation = {}
    local_validation_config = dict(cfg)
    if paths and not hard_reasons:
        try:
            validation_types = [segment_type for _path, segment_type in parts]
            # This pass only checks the fixed turn footprint. Whole-route crop
            # overlap and coverage checks run after all turns are assembled.
            local_validation_config.update({
                "max_track_core_overlap_pct": 100.0,
                "min_geometric_coverage_pct": 0.0,
            })
            validation = validate_footprints(
                paths,
                processed_mask,
                geo=geo,
                state=state,
                config=local_validation_config,
                harvester=_harvester_specs(state),
                segment_types=validation_types,
                headland_mask=headland,
                support_mask=cfg.get("planning_support_mask"),
                uncertain_mask=cfg.get("uncertain_mask"),
                forbidden_mask=cfg.get("forbidden_mask"),
            )
        except (TypeError, ValueError, RuntimeError):
            validation = {}
            hard_reasons.append("footprint_validation_unavailable")
    outside = _finite_float(validation.get("turn_outside_field_pct"))
    if outside is None and validation:
        outside = _finite_float(validation.get("track_outside_field_pct"))
    if (
        outside is None
        and paths
        and not any(
            not _segment_inside_field_boundary(path, field_polygon)
            for path in paths
        )
        and not hard_reasons
    ):
        hard_reasons.append("outside_field_metric_missing")
    elif outside is not None and outside > 0.0:
        hard_reasons.append("outside_field")
    forbidden = _finite_float(validation.get("turn_forbidden_overlap_area_m2"))
    if forbidden is None and validation and cfg.get("forbidden_mask") is not None:
        forbidden = _finite_float(validation.get("track_forbidden_overlap_pct"))
    if cfg.get("forbidden_mask") is not None and forbidden is None and paths:
        hard_reasons.append("forbidden_overlap_metric_missing")
    elif forbidden is not None and forbidden > 0.0:
        hard_reasons.append("forbidden_overlap")
    if paths and validation:
        formal = assess_footprint_hard_constraints(
            validation,
            local_validation_config,
        )
        validation["formal_hard_reasons"] = list(formal["hard_reasons"])
        validation["formal_validation_issues"] = list(formal["issues"])
        hard_reasons.extend(formal["hard_reasons"])
    metrics = dict(validation)
    metrics["turn_length_m"] = (
        _turn_parts_length_m(parts, metric_scale) if parts else float("inf")
    )
    candidate = {
        "candidate_id": f"{selected_strategy}:fixed:{offset:.6f}",
        "parts": parts,
        "path": paths,
        "hard_reasons": sorted(set(hard_reasons)),
        "validation_issues": list(
            validation.get("formal_validation_issues")
            or validation.get("issues")
            or []
        ),
        "metrics": metrics,
    }
    selected = validate_fixed_turn_candidate(
        candidate,
        fallback_candidate={"candidate_id": "original", "parts": base_parts},
        fallback_reason="fixed_turn_validation_failed",
    )
    diagnostics.update(selected.get("diagnostics") or {})
    diagnostics["rejected_candidates"] = selected.get("rejected_candidates", [])
    chosen = selected.get("selected")
    if chosen is None or chosen.get("candidate_id") == "original":
        return base_parts, diagnostics
    diagnostics["validation_passed"] = True
    diagnostics["selected_candidate_id"] = chosen.get("candidate_id", "")
    diagnostics["selected_metrics"] = dict(chosen.get("metrics") or {})
    return base_parts, diagnostics
