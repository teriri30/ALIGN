"""Harvester-footprint-aware work-line generation and validation."""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from harvester_geometry import (
    FOOTPRINT_MODEL,
    FOOTPRINT_REFERENCE_POINT,
    planar_geometry_m,
    transform_polygon,
)
from path_pose import sample_polyline_poses
from row_geometry import meters_per_pixel, odd_size, rotate_bound, smooth_1d


Point = Tuple[float, float]


# 用户友好的验证提示翻译字典
METRIC_TRANSLATIONS = {
    # 指标名称翻译
    "track_outside_field_pct": "履带越出田界",
    "track_forbidden_overlap_pct": "履带进入禁行区",
    "track_core_overlap_pct": "履带碾压作物核心区",
    "track_outside_support_pct": "履带在无作物支撑区",
    "track_uncertain_overlap_pct": "履带在不确定区域",
    "planned_target_geometric_coverage_pct": "割刀覆盖率",
    "rolling_canopy_pct": "预计碾压率",
    "crop_mask_geometric_coverage_pct": "作物掩膜覆盖率",
    "machine_outside_field_pct": "机身越出田界",
    "machine_forbidden_overlap_pct": "机身进入禁行区",
    # 符号翻译
    "exceeds limit": "超过限制",
    "below minimum": "低于最低要求",
}

ISSUE_SUGGESTIONS = {
    "履带越出田界": "→ 请调整路径，或扩大田块圈选范围",
    "履带进入禁行区": "→ 请拖动作业线避开禁行区",
    "机身越出田界": "→ 请调整路径，确保机身完全在田界内",
    "机身进入禁行区": "→ 请调整路径，避免机身进入禁行区",
    "割刀覆盖率": "→ 覆盖率略低，可调整作业线位置优化",
    "预计碾压率": "→ 碾压率略高，可优化转弯策略或作业线位置",
    "履带碾压作物核心区": "→ 碾压核心区略多，可调整作业线横向位置",
}


def translate_validation_issue(description: str, severity: str = "error") -> str:
    """将技术指标描述翻译为用户友好提示

    示例：
    输入："track_outside_field_pct=5.2 exceeds limit=2.0"
    输出："❌ 履带越出田界 5.2%（限制：2.0%）\n   → 请调整路径，或扩大田块圈选范围"
    """
    result = str(description)

    # 1. 替换技术术语
    for tech, user in METRIC_TRANSLATIONS.items():
        result = result.replace(tech, user)

    # 2. 美化数值格式：提取百分比值
    if "=" in result and ("超过限制" in result or "低于最低要求" in result):
        try:
            # "履带越出田界=5.2 超过限制=2.0" → "履带越出田界 5.2%（限制：2.0%）"
            parts = result.split()
            if len(parts) >= 2:
                # 提取指标和值
                metric_part = parts[0].split("=")
                if len(metric_part) == 2:
                    metric_name = metric_part[0]
                    metric_value = metric_part[1]

                    # 提取限制值
                    limit_value = None
                    for part in parts[1:]:
                        if "=" in part:
                            limit_value = part.split("=")[-1]
                            break

                    if limit_value:
                        if "超过限制" in result:
                            result = f"{metric_name} {metric_value}%（限制：{limit_value}%）"
                        elif "低于最低要求" in result:
                            result = f"{metric_name} {metric_value}%（最低要求：{limit_value}%）"
        except Exception:
            pass  # 保持原始格式

    # 3. 添加图标
    icon = "❌" if severity == "error" else "⚠️"
    result = f"{icon} {result}"

    # 4. 添加改进建议（仅对 warning）
    if severity == "warning":
        for keyword, suggestion in ISSUE_SUGGESTIONS.items():
            if keyword in result:
                result += f"\n   {suggestion}"
                break

    return result


def rasterize_forbidden_regions(shape, state=None) -> Optional[np.ndarray]:
    """Rasterize confirmed display-coordinate polygons into local mask space."""
    regions = list(getattr(state, "forbidden_regions", []) or []) if state is not None else []
    if not regions:
        return None
    height, width = int(shape[0]), int(shape[1])
    result = np.zeros((height, width), dtype=np.uint8)
    offset_x = float(getattr(state, "mask_offset_x", 0) or 0)
    offset_y = float(getattr(state, "mask_offset_y", 0) or 0)
    for region in regions:
        if len(region) < 3:
            continue
        points = np.asarray(region, dtype=np.float64)
        points[:, 0] -= offset_x
        points[:, 1] -= offset_y
        cv2.fillPoly(result, [np.round(points).astype(np.int32)], 255)
    return result if np.count_nonzero(result) else None


def _field_support(mask: np.ndarray, state=None) -> np.ndarray:
    support = np.zeros(mask.shape[:2], dtype=np.uint8)
    boundary = getattr(state, "field_boundary", None) if state is not None else None
    if boundary and len(boundary) >= 3:
        from row_geometry import sanitize_field_polygon

        # Fill the same simple ring the planner gates on: a self-intersecting
        # outline would otherwise rasterize its closing loop with even-odd fill.
        ring = sanitize_field_polygon(boundary)
        if ring is not None:
            offset_x = int(getattr(state, "mask_offset_x", 0))
            offset_y = int(getattr(state, "mask_offset_y", 0))
            points = np.asarray(ring, dtype=np.float64)
            points[:, 0] -= offset_x
            points[:, 1] -= offset_y
            cv2.fillPoly(support, [np.round(points).astype(np.int32)], 255)
            if np.count_nonzero(support) > 0:
                return support

    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None or len(points) < 3:
        return support
    hull = cv2.convexHull(points)
    cv2.fillPoly(support, [hull], 255)
    return support


def _interval_mean(cumulative: np.ndarray, start: int, end: int) -> float:
    """Mean over an interval, treating samples outside the raster as zero."""
    profile_size = max(0, int(cumulative.size) - 1)
    start, end = int(start), int(end)
    requested_width = end - start
    if requested_width <= 0 or profile_size <= 0:
        return 0.0
    clipped_start = max(0, min(start, profile_size))
    clipped_end = max(0, min(end, profile_size))
    if clipped_end <= clipped_start:
        return 0.0
    return float(cumulative[clipped_end] - cumulative[clipped_start]) / requested_width


def summarize_work_line_footprint(footprint: Dict) -> Dict:
    keys = (
        "track_core_overlap_pct",
        "planned_target_geometric_coverage_pct",
        "track_outside_field_pct",
        "track_outside_support_pct",
        "track_uncertain_overlap_pct",
        "track_forbidden_overlap_pct",
        "rolling_canopy_pct",
        "track_swept_area_m2",
        "footprint_resolution_m",
    )
    return {key: footprint.get(key) for key in keys}


def compare_work_line_footprints(
    baseline: Dict,
    optimized: Dict,
    *,
    tolerance: float = 1e-6,
) -> Dict:
    """Accept optimized work lines only when formal footprint metrics improve safely."""
    lower_is_better = (
        "track_core_overlap_pct",
        "track_outside_field_pct",
        "track_outside_support_pct",
        "track_uncertain_overlap_pct",
        "track_forbidden_overlap_pct",
    )
    higher_is_better = ("planned_target_geometric_coverage_pct",)
    required = {
        "track_core_overlap_pct",
        "planned_target_geometric_coverage_pct",
        "track_outside_field_pct",
    }
    degraded, improved, invalid = [], [], []

    for name in (*lower_is_better, *higher_is_better):
        before = baseline.get(name)
        after = optimized.get(name)
        if before is None and after is None:
            if name in required:
                invalid.append(name)
            continue
        try:
            before_value = float(before)
            after_value = float(after)
        except (TypeError, ValueError):
            invalid.append(name)
            continue
        if not math.isfinite(before_value) or not math.isfinite(after_value):
            invalid.append(name)
            continue
        delta = after_value - before_value
        if name in lower_is_better:
            if delta > tolerance:
                degraded.append(name)
            elif delta < -tolerance:
                improved.append(name)
        else:
            if delta < -tolerance:
                degraded.append(name)
            elif delta > tolerance:
                improved.append(name)

    return {
        "accepted": not invalid and not degraded and bool(improved),
        "degraded_metrics": degraded,
        "improved_metrics": improved,
        "invalid_metrics": invalid,
    }


def _choose_layout(
    crop_profile: np.ndarray,
    core_profile: np.ndarray,
    work_mpp: float,
    cutter_width_m: float,
    track_width_m: float,
    track_gauge_m: float,
    overlap_ratio: float,
) -> Tuple[List[float], Dict]:
    crop_rows = np.flatnonzero(crop_profile > max(0.03, float(np.max(crop_profile)) * 0.04))
    if crop_rows.size < 2:
        return [], {}

    y_min = float(crop_rows[0])
    y_max = float(crop_rows[-1])
    cutter_px = cutter_width_m / work_mpp
    track_px = track_width_m / work_mpp
    gauge_px = track_gauge_m / work_mpp
    base_swath = cutter_px * (1.0 - overlap_ratio)
    crop_weight = np.maximum(crop_profile, 0.0)
    crop_total = float(np.sum(crop_weight))
    if crop_total <= 1e-6:
        return [], {}

    best = None
    for swath_scale in (0.94, 1.0, 1.06):
        swath = max(cutter_px * 0.55, base_swath * swath_scale)
        phase_count = max(12, min(72, int(round(swath))))
        for phase in np.linspace(0.0, swath, phase_count, endpoint=False):
            centers = np.arange(
                y_min - cutter_px / 2.0 + phase,
                y_max + cutter_px / 2.0 + swath,
                swath,
            )
            kept = []
            cutter_union = np.zeros(crop_profile.size, dtype=bool)
            core_sum = 0.0
            soft_sum = 0.0
            track_length = 0
            for center in centers:
                cutter_start = max(0, int(math.floor(center - cutter_px / 2.0)))
                cutter_end = min(crop_profile.size, int(math.ceil(center + cutter_px / 2.0)))
                if cutter_end <= cutter_start:
                    continue
                potential = float(np.sum(crop_weight[cutter_start:cutter_end]))
                if potential < crop_total * 0.002:
                    continue
                kept.append(float(center))
                cutter_union[cutter_start:cutter_end] = True
                for track_center in (center - gauge_px / 2.0, center + gauge_px / 2.0):
                    start = max(0, int(math.floor(track_center - track_px / 2.0)))
                    end = min(crop_profile.size, int(math.ceil(track_center + track_px / 2.0)))
                    if end <= start:
                        continue
                    core_sum += float(np.sum(core_profile[start:end]))
                    soft_sum += float(np.sum(crop_profile[start:end]))
                    track_length += end - start
            if not kept:
                continue
            coverage = float(np.sum(crop_weight[cutter_union]) / crop_total)
            core_overlap = core_sum / max(track_length, 1)
            soft_overlap = soft_sum / max(track_length, 1)
            objective = (
                15.0 * (1.0 - coverage)
                + 12.0 * core_overlap
                + 2.0 * soft_overlap
                + 0.012 * len(kept)
            )
            item = (
                objective,
                kept,
                {
                    "predicted_geometric_coverage_pct": coverage * 100.0,
                    "predicted_track_core_overlap_pct": core_overlap * 100.0,
                    "predicted_track_canopy_overlap_pct": soft_overlap * 100.0,
                    "pass_count": len(kept),
                    "swath_m": swath * work_mpp,
                },
            )
            if best is None or item[0] < best[0]:
                best = item

    if best is None:
        return [], {}
    return best[1], best[2]


def _longest_true_run(values: np.ndarray) -> Optional[Tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    changes = np.flatnonzero(np.diff(values.astype(np.int8))) + 1
    bounds = np.concatenate(([0], changes, [values.size]))
    runs = [
        (int(bounds[index]), int(bounds[index + 1]))
        for index in range(len(bounds) - 1)
        if values[bounds[index]]
    ]
    return max(runs, key=lambda item: item[1] - item[0]) if runs else None


def _optimize_line(
    baseline_y: float,
    crop: np.ndarray,
    core: np.ndarray,
    support: np.ndarray,
    work_mpp: float,
    cutter_width_m: float,
    track_width_m: float,
    track_gauge_m: float,
    headland_buffer_m: float,
    config: Dict,
) -> Optional[np.ndarray]:
    cutter_px = cutter_width_m / work_mpp
    track_px = track_width_m / work_mpp
    gauge_px = track_gauge_m / work_mpp
    half_cutter = max(1, int(round(cutter_px / 2.0)))
    y0 = max(0, int(round(baseline_y)) - half_cutter)
    y1 = min(support.shape[0], int(round(baseline_y)) + half_cutter + 1)
    row_support = np.mean(support[y0:y1] > 0, axis=0) >= 0.18
    support_run = _longest_true_run(row_support)
    if support_run is None:
        return None

    start_x, end_x = support_run
    trim_px = max(0, int(round(headland_buffer_m / work_mpp)))
    if end_x - start_x > 2 * trim_px + 10:
        start_x += trim_px
        end_x -= trim_px
    if end_x - start_x < 10:
        return None

    sample_step = max(2, int(round(float(config.get("station_interval_m", 0.25)) / work_mpp)))
    window_half = max(1, int(round(float(config.get("station_window_m", 0.16)) / work_mpp / 2.0)))
    x_values = np.arange(start_x, end_x + 1, sample_step, dtype=np.int32)
    if x_values.size < 3:
        return None

    search_m = float(config.get("lateral_search_m", 0.25))
    lateral_step_m = float(config.get("lateral_step_m", 0.025))
    offset_values = np.arange(
        -search_m,
        search_m + lateral_step_m * 0.5,
        lateral_step_m,
    ) / work_mpp
    data_cost = np.zeros((x_values.size, offset_values.size), dtype=np.float64)

    for x_index, x_value in enumerate(x_values):
        x0 = max(0, int(x_value) - window_half)
        x1 = min(crop.shape[1], int(x_value) + window_half + 1)
        crop_profile = np.mean(crop[:, x0:x1] > 0, axis=1).astype(np.float64)
        core_profile = np.mean(core[:, x0:x1] > 0, axis=1).astype(np.float64)
        support_profile = np.mean(support[:, x0:x1] > 0, axis=1).astype(np.float64)
        crop_cumulative = np.concatenate(([0.0], np.cumsum(crop_profile)))
        core_cumulative = np.concatenate(([0.0], np.cumsum(core_profile)))
        support_cumulative = np.concatenate(([0.0], np.cumsum(support_profile)))

        for offset_index, offset in enumerate(offset_values):
            center = baseline_y + offset
            core_overlap = 0.0
            canopy_overlap = 0.0
            track_support = 0.0
            for track_center in (center - gauge_px / 2.0, center + gauge_px / 2.0):
                start = int(round(track_center - track_px / 2.0))
                end = int(round(track_center + track_px / 2.0)) + 1
                core_overlap += _interval_mean(core_cumulative, start, end)
                canopy_overlap += _interval_mean(crop_cumulative, start, end)
                track_support += _interval_mean(support_cumulative, start, end)
            core_overlap *= 0.5
            canopy_overlap *= 0.5
            track_support *= 0.5

            cutter_start = int(round(center - cutter_px / 2.0))
            cutter_end = int(round(center + cutter_px / 2.0)) + 1
            cutter_crop = _interval_mean(crop_cumulative, cutter_start, cutter_end)
            cutter_support = _interval_mean(support_cumulative, cutter_start, cutter_end)
            outside = 1.0 - min(track_support, cutter_support)
            normalized_offset = offset * work_mpp / max(search_m, 1e-6)
            data_cost[x_index, offset_index] = (
                25.0 * core_overlap
                + 5.0 * canopy_overlap
                + 20.0 * outside
                - 1.5 * cutter_crop
                + 0.8 * normalized_offset * normalized_offset
            )

    smooth_penalty = float(config.get("lateral_smooth_penalty", 0.12))
    dp = np.full_like(data_cost, np.inf)
    previous = np.zeros(data_cost.shape, dtype=np.int32)
    dp[0] = data_cost[0]
    indices = np.arange(offset_values.size)
    for station in range(1, x_values.size):
        for current in range(offset_values.size):
            transition = dp[station - 1] + smooth_penalty * (indices - current) ** 2
            best_previous = int(np.argmin(transition))
            previous[station, current] = best_previous
            dp[station, current] = data_cost[station, current] + transition[best_previous]

    chosen = np.zeros(x_values.size, dtype=np.int32)
    chosen[-1] = int(np.argmin(dp[-1]))
    for station in range(x_values.size - 1, 0, -1):
        chosen[station - 1] = previous[station, chosen[station]]
    offsets = offset_values[chosen]
    offsets = smooth_1d(
        offsets,
        float(config.get("path_smooth_m", 0.75))
        / max(sample_step * work_mpp, 1e-6),
    )
    return np.column_stack([x_values.astype(np.float64), baseline_y + offsets])


def _optimize_baseline_line(
    baseline: np.ndarray,
    crop: np.ndarray,
    core: np.ndarray,
    support: np.ndarray,
    forbidden: Optional[np.ndarray],
    uncertain: Optional[np.ndarray],
    work_mpp: float,
    cutter_width_m: float,
    track_width_m: float,
    track_gauge_m: float,
    config: Dict,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Optimize one row-band centerline only within its lateral search corridor."""
    values = np.asarray(baseline, dtype=np.float64)
    reference_shape = np.asarray(crop).shape[:2]
    for layer_name, layer in (
        ("core", core),
        ("support", support),
        ("forbidden", forbidden),
        ("uncertain", uncertain),
    ):
        if layer is not None and np.asarray(layer).shape[:2] != reference_shape:
            return None, {"reason": f"{layer_name}_shape_mismatch"}
    if values.ndim != 2 or values.shape[0] < 2:
        return None, {"reason": "invalid_baseline"}
    order = np.argsort(values[:, 0])
    values = values[order]
    keep = np.concatenate(([True], np.diff(values[:, 0]) > 1e-6))
    values = values[keep]
    if values.shape[0] < 2:
        return None, {"reason": "degenerate_baseline"}

    start_x, end_x = float(values[0, 0]), float(values[-1, 0])
    station_m = float(config.get("station_interval_m", 0.25))
    station_px = max(1.0, station_m / max(work_mpp, 1e-9))
    sample_count = max(2, int(math.ceil((end_x - start_x) / station_px)) + 1)
    x_values = np.linspace(start_x, end_x, sample_count)
    baseline_y = np.interp(x_values, values[:, 0], values[:, 1])

    search_m = float(config.get("lateral_search_m", 0.25))
    lateral_step_m = float(config.get("lateral_step_m", 0.025))
    if not math.isfinite(search_m) or search_m < 0.0:
        return None, {"reason": "invalid_lateral_search"}
    if not math.isfinite(lateral_step_m) or lateral_step_m <= 0.0:
        return None, {"reason": "invalid_lateral_step"}
    offset_values = np.arange(
        -search_m,
        search_m + lateral_step_m * 0.5,
        lateral_step_m,
        dtype=np.float64,
    ) / max(work_mpp, 1e-9)
    if offset_values.size == 0:
        return None, {"reason": "empty_lateral_search"}

    cutter_px = cutter_width_m / work_mpp
    track_px = track_width_m / work_mpp
    gauge_px = track_gauge_m / work_mpp
    window_half = max(
        1,
        int(round(float(config.get("station_window_m", 0.16)) / work_mpp / 2.0)),
    )
    data_cost = np.full((sample_count, offset_values.size), np.inf, dtype=np.float64)
    forbidden_mask = None if forbidden is None else np.asarray(forbidden) > 0
    uncertain_mask = None if uncertain is None else np.asarray(uncertain) > 0
    uncertain_penalty = float(config.get("lateral_uncertain_penalty", 40.0))

    for x_index, x_value in enumerate(x_values):
        x0 = max(0, int(round(x_value)) - window_half)
        x1 = min(crop.shape[1], int(round(x_value)) + window_half + 1)
        if x1 <= x0:
            continue
        crop_profile = np.mean(crop[:, x0:x1] > 0, axis=1).astype(np.float64)
        core_profile = np.mean(core[:, x0:x1] > 0, axis=1).astype(np.float64)
        support_profile = np.mean(support[:, x0:x1] > 0, axis=1).astype(np.float64)
        forbidden_profile = (
            np.mean(forbidden_mask[:, x0:x1], axis=1).astype(np.float64)
            if forbidden_mask is not None
            else np.zeros(crop.shape[0], dtype=np.float64)
        )
        uncertain_profile = (
            np.mean(uncertain_mask[:, x0:x1], axis=1).astype(np.float64)
            if uncertain_mask is not None
            else np.zeros(crop.shape[0], dtype=np.float64)
        )
        crop_cumulative = np.concatenate(([0.0], np.cumsum(crop_profile)))
        core_cumulative = np.concatenate(([0.0], np.cumsum(core_profile)))
        support_cumulative = np.concatenate(([0.0], np.cumsum(support_profile)))
        forbidden_cumulative = np.concatenate(([0.0], np.cumsum(forbidden_profile)))
        uncertain_cumulative = np.concatenate(([0.0], np.cumsum(uncertain_profile)))

        for offset_index, offset in enumerate(offset_values):
            center = float(baseline_y[x_index] + offset)
            if center < 0.0 or center >= crop.shape[0]:
                continue
            core_overlap = canopy_overlap = track_support = 0.0
            forbidden_overlap = uncertain_overlap = 0.0
            for track_center in (center - gauge_px / 2.0, center + gauge_px / 2.0):
                start = int(round(track_center - track_px / 2.0))
                end = int(round(track_center + track_px / 2.0)) + 1
                core_overlap += _interval_mean(core_cumulative, start, end)
                canopy_overlap += _interval_mean(crop_cumulative, start, end)
                track_support += _interval_mean(support_cumulative, start, end)
                forbidden_overlap += _interval_mean(forbidden_cumulative, start, end)
                uncertain_overlap += _interval_mean(uncertain_cumulative, start, end)
            core_overlap *= 0.5
            canopy_overlap *= 0.5
            track_support *= 0.5
            forbidden_overlap *= 0.5
            uncertain_overlap *= 0.5

            cutter_start = int(round(center - cutter_px / 2.0))
            cutter_end = int(round(center + cutter_px / 2.0)) + 1
            cutter_crop = _interval_mean(crop_cumulative, cutter_start, cutter_end)
            cutter_support = _interval_mean(support_cumulative, cutter_start, cutter_end)
            outside = 1.0 - min(track_support, cutter_support)
            normalized_offset = offset * work_mpp / max(search_m, lateral_step_m)
            if forbidden_overlap > 0.0:
                continue
            data_cost[x_index, offset_index] = (
                80.0 * core_overlap
                + 5.0 * canopy_overlap
                + 20.0 * outside
                + uncertain_penalty * uncertain_overlap
                - 1.0 * cutter_crop
                + 0.8 * normalized_offset * normalized_offset
            )

    if not np.all(np.any(np.isfinite(data_cost), axis=1)):
        return None, {"reason": "no_finite_lateral_candidate"}
    smooth_penalty = float(config.get("lateral_smooth_penalty", 0.12))
    dp = np.full_like(data_cost, np.inf)
    previous = np.zeros(data_cost.shape, dtype=np.int32)
    dp[0] = data_cost[0]
    indices = np.arange(offset_values.size)
    for station in range(1, sample_count):
        for current in range(offset_values.size):
            if not math.isfinite(float(data_cost[station, current])):
                continue
            transition = dp[station - 1] + smooth_penalty * (indices - current) ** 2
            best_previous = int(np.argmin(transition))
            previous[station, current] = best_previous
            dp[station, current] = data_cost[station, current] + transition[best_previous]
    if not np.any(np.isfinite(dp[-1])):
        return None, {"reason": "no_continuous_lateral_candidate"}

    chosen = np.zeros(sample_count, dtype=np.int32)
    chosen[-1] = int(np.argmin(dp[-1]))
    for station in range(sample_count - 1, 0, -1):
        chosen[station - 1] = previous[station, chosen[station]]
    offsets = offset_values[chosen]
    zero_index = int(np.argmin(np.abs(offset_values)))
    baseline_costs = data_cost[:, zero_index]
    offsets = smooth_1d(
        offsets,
        float(config.get("path_smooth_m", 0.75)) / max(station_m, 1e-6),
    )
    limit_px = search_m / max(work_mpp, 1e-9)
    offsets = np.clip(offsets, -limit_px, limit_px)
    final_indices = np.argmin(
        np.abs(offsets[:, None] - offset_values[None, :]),
        axis=1,
    )
    selected_costs = data_cost[np.arange(sample_count), final_indices]
    if not np.all(np.isfinite(selected_costs)):
        return None, {"reason": "smoothed_path_enters_invalid_candidate"}
    optimized_objective = float(np.mean(selected_costs))
    baseline_objective = (
        float(np.mean(baseline_costs))
        if np.all(np.isfinite(baseline_costs))
        else None
    )
    if (
        baseline_objective is not None
        and optimized_objective > baseline_objective + 1e-9
    ):
        return None, {
            "reason": "smoothed_path_worse_than_baseline",
            "baseline_objective": baseline_objective,
            "optimized_objective": optimized_objective,
        }
    optimized = np.column_stack([x_values, baseline_y + offsets])
    shift_m = np.abs(offsets) * work_mpp
    return optimized, {
        "mean_abs_lateral_shift_m": float(np.mean(shift_m)),
        "max_abs_lateral_shift_m": float(np.max(shift_m)),
        "station_count": int(sample_count),
        "baseline_objective": baseline_objective,
        "optimized_objective": optimized_objective,
        "objective_improvement": (
            float(baseline_objective - optimized_objective)
            if baseline_objective is not None
            else None
        ),
    }


def generate_work_lines(
    processed_mask: np.ndarray,
    main_angle: float,
    geo=None,
    state=None,
    harvester: Optional[Dict] = None,
    config: Optional[Dict] = None,
    progress_callback=None,
    baseline_lines: Optional[List[List[Point]]] = None,
) -> Tuple[List[List[Point]], Dict]:
    """Generate footprint-aware passes, optionally anchored to row-band centerlines."""
    cfg = config or {}
    specs = dict(harvester or {})
    required = ("cutter_width_m", "track_width_m", "track_gauge_m")
    missing = [key for key in required if key not in specs]
    if missing:
        raise ValueError("missing work-line geometry parameters: " + ", ".join(missing))
    cutter_width_m = float(specs["cutter_width_m"])
    track_width_m = float(specs["track_width_m"])
    track_gauge_m = float(specs["track_gauge_m"])
    headland_buffer_m = float(cfg.get("headland_buffer_m", 3.0))

    def report(value: float, stage: str):
        if progress_callback:
            try:
                progress_callback(value, stage)
            except Exception:
                pass

    raw = (np.asarray(processed_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    h, w = raw.shape[:2]
    mpp = meters_per_pixel(raw, geo, state)
    target_mpp = float(cfg.get("planning_resolution_m", 0.02))
    max_dim = int(cfg.get("planning_max_dim", 3600))
    scale = min(1.0, mpp / max(target_mpp, mpp), float(max_dim) / max(h, w))
    work = cv2.resize(
        raw,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    work = (work >= 96).astype(np.uint8) * 255
    configured_support = cfg.get("planning_support_mask")
    if (
        isinstance(configured_support, np.ndarray)
        and configured_support.shape[:2] == raw.shape[:2]
    ):
        support_full = (
            np.asarray(configured_support, dtype=np.uint8) > 0
        ).astype(np.uint8) * 255
    else:
        support_full = _field_support(raw, state)
    support_work = cv2.resize(
        support_full,
        (work.shape[1], work.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    forbidden_full = rasterize_forbidden_regions(raw.shape, state)
    forbidden_work = (
        cv2.resize(
            forbidden_full,
            (work.shape[1], work.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        if forbidden_full is not None
        else None
    )
    configured_uncertain = cfg.get("uncertain_mask")
    uncertain_work = (
        cv2.resize(
            (np.asarray(configured_uncertain, dtype=np.uint8) > 0).astype(np.uint8) * 255,
            (work.shape[1], work.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        if (
            isinstance(configured_uncertain, np.ndarray)
            and configured_uncertain.shape[:2] == raw.shape[:2]
        )
        else None
    )
    work_mpp = mpp / scale

    report(0.08, "建立农机足迹地图")
    crop_rotated, matrix = rotate_bound(work, math.degrees(main_angle), cv2.INTER_NEAREST)
    support_rotated = cv2.warpAffine(
        support_work,
        matrix,
        (crop_rotated.shape[1], crop_rotated.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    forbidden_rotated = (
        cv2.warpAffine(
            forbidden_work,
            matrix,
            (crop_rotated.shape[1], crop_rotated.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if forbidden_work is not None
        else None
    )
    uncertain_rotated = (
        cv2.warpAffine(
            uncertain_work,
            matrix,
            (crop_rotated.shape[1], crop_rotated.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if uncertain_work is not None
        else None
    )
    inverse = cv2.invertAffineTransform(matrix)
    if np.count_nonzero(support_rotated) == 0:
        support_rotated = cv2.morphologyEx(
            crop_rotated,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (
                    odd_size(1.2 / work_mpp),
                    odd_size(1.4 / work_mpp),
                ),
            ),
        )

    erosion_m = float(cfg.get("crop_core_erosion_m", 0.10))
    erosion_px = max(1, int(round(erosion_m / work_mpp)))
    core = cv2.erode(
        crop_rotated,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (odd_size(erosion_px * 2 + 1), odd_size(erosion_px * 2 + 1)),
        ),
        iterations=1,
    )

    if baseline_lines is not None:
        report(0.18, "以四行带中心线为基准建立横向搜索走廊")
        optimized_lines: List[List[Point]] = []
        line_diagnostics = []
        for index, baseline in enumerate(baseline_lines):
            values = np.asarray(baseline, dtype=np.float64)
            if values.ndim != 2 or values.shape[0] < 2:
                line_diagnostics.append({
                    "line_index": int(index),
                    "optimized": False,
                    "reason": "invalid_baseline",
                })
                continue
            baseline_work = values * scale
            baseline_rotated = cv2.transform(
                baseline_work.reshape(-1, 1, 2),
                matrix,
            ).reshape(-1, 2)
            line_rotated, diagnostics = _optimize_baseline_line(
                baseline_rotated,
                crop_rotated,
                core,
                support_rotated,
                forbidden_rotated,
                uncertain_rotated,
                work_mpp,
                cutter_width_m,
                track_width_m,
                track_gauge_m,
                cfg,
            )
            item = {"line_index": int(index), **dict(diagnostics or {})}
            if line_rotated is None or len(line_rotated) < 2:
                item["optimized"] = False
                line_diagnostics.append(item)
                continue
            line_work = cv2.transform(
                line_rotated.reshape(-1, 1, 2).astype(np.float64),
                inverse,
            ).reshape(-1, 2)
            line_full = line_work / scale
            optimized_lines.append([
                (float(x), float(y)) for x, y in line_full
            ])
            item["optimized"] = True
            line_diagnostics.append(item)
            report(
                0.18 + 0.72 * (index + 1) / max(len(baseline_lines), 1),
                f"四行带作业线横向优化 {index + 1}/{len(baseline_lines)}",
            )

        complete = len(optimized_lines) == len(baseline_lines)
        max_shift = max(
            (
                float(item.get("max_abs_lateral_shift_m", 0.0))
                for item in line_diagnostics
                if item.get("optimized")
            ),
            default=0.0,
        )
        layout = {
            "source": "four_row_band_centerlines_lateral_optimization",
            "footprint_optimization_scope": "centerline_anchored_lateral_search",
            "optimization_complete": bool(complete),
            "baseline_pass_count": len(baseline_lines),
            "optimized_pass_count": len(optimized_lines),
            "generated_pass_count": len(optimized_lines),
            "lateral_search_m": float(cfg.get("lateral_search_m", 0.25)),
            "lateral_step_m": float(cfg.get("lateral_step_m", 0.025)),
            "max_abs_lateral_shift_m": max_shift,
            "line_optimization": line_diagnostics,
            "meters_per_pixel": float(mpp),
            "planning_scale": float(scale),
            "planning_resolution_m": float(work_mpp),
            "crop_core_erosion_m": erosion_m,
        }
        if not complete:
            layout["reason"] = "incomplete_centerline_optimization"
            return [], layout
        report(0.90, "四行带作业线横向优化完成")
        return optimized_lines, layout

    support_counts = np.sum(support_rotated > 0, axis=1).astype(np.float64)
    crop_profile = np.divide(
        np.sum(crop_rotated > 0, axis=1),
        np.maximum(support_counts, 1.0),
    )
    core_profile = np.divide(
        np.sum(core > 0, axis=1),
        np.maximum(support_counts, 1.0),
    )
    crop_profile = smooth_1d(crop_profile, max(3, 0.04 / work_mpp))
    core_profile = smooth_1d(core_profile, max(3, 0.04 / work_mpp))

    report(0.18, "优化割台覆盖相位")
    centers, layout = _choose_layout(
        crop_profile,
        core_profile,
        work_mpp,
        cutter_width_m,
        track_width_m,
        track_gauge_m,
        overlap_ratio=float(cfg.get("cutter_overlap_ratio", 0.12)),
    )
    if not centers:
        return [], {"reason": "no_feasible_layout"}

    report(0.28, "逐作业线避碾优化")
    work_lines = []
    for index, center in enumerate(centers):
        line_rotated = _optimize_line(
            center,
            crop_rotated,
            core,
            support_rotated,
            work_mpp,
            cutter_width_m,
            track_width_m,
            track_gauge_m,
            headland_buffer_m,
            cfg,
        )
        if line_rotated is None or len(line_rotated) < 2:
            continue
        line_work = cv2.transform(
            line_rotated.reshape(-1, 1, 2).astype(np.float64),
            inverse,
        ).reshape(-1, 2)
        line_full = line_work / scale
        work_lines.append([(float(x), float(y)) for x, y in line_full])
        report(
            0.28 + 0.62 * (index + 1) / max(len(centers), 1),
            f"作业线优化 {index + 1}/{len(centers)}",
        )

    layout.update({
        "meters_per_pixel": float(mpp),
        "planning_scale": float(scale),
        "planning_resolution_m": float(work_mpp),
        "crop_core_erosion_m": erosion_m,
        "generated_pass_count": len(work_lines),
    })
    report(0.90, "作业线生成完成")
    return work_lines, layout


def _offset_polyline(points: np.ndarray, offset: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    result = np.zeros_like(points, dtype=np.float64)
    for index in range(len(points)):
        if index == 0:
            tangent = points[1] - points[0]
        elif index == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[index + 1] - points[index - 1]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-9:
            normal = np.array([0.0, 0.0])
        else:
            normal = np.array([-tangent[1], tangent[0]]) / norm
        result[index] = points[index] + normal * offset
    return result


def _sample_planar_poses(
    points: np.ndarray,
    spacing_px: float,
    *,
    heading_window_px: float,
    reversing: bool = False,
) -> List[Tuple[float, float, float]]:
    """Use the canonical path-pose sampler shared with animation."""
    try:
        return sample_polyline_poses(
            points,
            lambda first, second: math.hypot(
                second[0] - first[0], second[1] - first[1]
            ),
            max(0.5, float(spacing_px)),
            max(1.0, float(heading_window_px)),
            reversing=reversing,
        )
    except ValueError:
        return []


def _accumulate_pose_track_visit(
    track_pose_occupancy_count: np.ndarray,
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
) -> Tuple[int, int]:
    """Accumulate one pose using only the tracks' clipped local bounding box."""
    height, width = track_pose_occupancy_count.shape[:2]
    points = np.vstack((left_pixels, right_pixels)).astype(np.int32, copy=False)
    x0 = max(0, int(np.min(points[:, 0])))
    y0 = max(0, int(np.min(points[:, 1])))
    x1 = min(width - 1, int(np.max(points[:, 0])))
    y1 = min(height - 1, int(np.max(points[:, 1])))
    if x1 < x0 or y1 < y0:
        return 0, 0
    local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    offset = np.asarray([x0, y0], dtype=np.int32)
    cv2.fillPoly(local, [left_pixels - offset], 1)
    cv2.fillPoly(local, [right_pixels - offset], 1)
    track_pose_occupancy_count[y0:y1 + 1, x0:x1 + 1] += local.astype(np.uint32)
    return local.shape


def _accumulate_logical_segment_pass(
    track_pass_count: np.ndarray,
    segment_track_sweep: np.ndarray,
) -> None:
    """Count one logical segment once for every pixel in its union sweep.

    Pose occupancy is intentionally separate from logical passage count.  A
    segment may contain hundreds of overlapping poses, but it is still only
    one logical pass for the formal repeated-rolling metrics.  Saturating at
    the uint32 limit keeps the diagnostic map auditable even if a caller
    supplies an unusually large number of logical segments.
    """
    active = np.asarray(segment_track_sweep) > 0
    if not np.any(active):
        return
    max_count = np.iinfo(np.uint32).max
    current = np.asarray(track_pass_count[active], dtype=np.uint64)
    track_pass_count[active] = np.minimum(
        current + np.uint64(1), np.uint64(max_count)
    ).astype(np.uint32)


def _rasterize_planar_sweeps(
    shape: Tuple[int, int],
    full_path: List[List[Point]],
    segment_types: Optional[List[str]],
    geometry: Dict,
    work_mpp: float,
    config: Dict,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    List[np.ndarray], float, float, int,
]:
    track_swept = np.zeros(shape, dtype=np.uint8)
    track_pose_occupancy_count = np.zeros(shape, dtype=np.uint32)
    track_pass_count = np.zeros(shape, dtype=np.uint32)
    cutter_swept = np.zeros(shape, dtype=np.uint8)
    machine_swept = np.zeros(shape, dtype=np.uint8)
    segment_track_sweeps = [np.zeros(shape, dtype=np.uint8) for _ in full_path]
    dimensions = geometry["dimensions_m"]
    default_step_m = min(
        0.10,
        max(
            0.025,
            min(dimensions["track_width_m"], dimensions["track_length_m"]) / 3.0,
        ),
    )
    pose_step_m = float(config.get("footprint_pose_step_m", default_step_m))
    pose_step_m = max(0.01, min(0.20, pose_step_m))
    pose_heading_window_m = max(
        0.01, float(config.get("pose_heading_window_m", 0.30))
    )
    spacing_px = pose_step_m / work_mpp
    heading_window_px = pose_heading_window_m / work_mpp
    pixels_per_meter = 1.0 / work_mpp
    pose_count = 0

    for segment_index, segment in enumerate(full_path):
        if len(segment) < 2:
            continue
        segment_type = (
            segment_types[segment_index]
            if segment_types and segment_index < len(segment_types)
            else ("work" if segment_index % 2 == 0 else "turn")
        )
        poses = _sample_planar_poses(
            np.asarray(segment, dtype=np.float64),
            spacing_px,
            heading_window_px=heading_window_px,
            reversing=str(segment_type) == "turn_reverse",
        )
        pose_count += len(poses)
        previous_cutter = None
        segment_track = segment_track_sweeps[segment_index]
        for x, y, heading in poses:
            center = (x, y)
            left = transform_polygon(
                geometry["left_track"], center, heading, pixels_per_meter
            )
            right = transform_polygon(
                geometry["right_track"], center, heading, pixels_per_meter
            )
            body = transform_polygon(
                geometry["body"], center, heading, pixels_per_meter
            )
            cutter_line = transform_polygon(
                geometry["cutter_line"], center, heading, pixels_per_meter
            )
            cv2.fillPoly(machine_swept, [np.rint(body).astype(np.int32)], 255)
            left_pixels = np.rint(left).astype(np.int32)
            right_pixels = np.rint(right).astype(np.int32)
            cv2.fillPoly(segment_track, [left_pixels], 255)
            cv2.fillPoly(segment_track, [right_pixels], 255)
            cv2.fillPoly(track_swept, [left_pixels], 255)
            cv2.fillPoly(track_swept, [right_pixels], 255)
            _accumulate_pose_track_visit(
                track_pose_occupancy_count,
                left_pixels,
                right_pixels,
            )
            cutter_pixels = np.rint(cutter_line).astype(np.int32)
            cv2.line(
                machine_swept,
                tuple(cutter_pixels[0]),
                tuple(cutter_pixels[1]),
                255,
                1,
            )
            if str(segment_type) == "work":
                cv2.line(
                    cutter_swept,
                    tuple(cutter_pixels[0]),
                    tuple(cutter_pixels[1]),
                    255,
                    1,
                )
                if previous_cutter is not None:
                    swept_quad = np.asarray(
                        (
                            previous_cutter[0],
                            previous_cutter[1],
                            cutter_pixels[1],
                            cutter_pixels[0],
                        ),
                        dtype=np.int32,
                    )
                    cv2.fillPoly(cutter_swept, [swept_quad], 255)
                previous_cutter = cutter_pixels
        _accumulate_logical_segment_pass(track_pass_count, segment_track)
    return (
        track_swept, track_pose_occupancy_count, track_pass_count,
        cutter_swept, machine_swept,
        segment_track_sweeps,
        pose_step_m, pose_heading_window_m, pose_count,
    )


def validate_footprints(
    full_path: List[List[Point]],
    processed_mask: np.ndarray,
    geo=None,
    state=None,
    harvester: Optional[Dict] = None,
    config: Optional[Dict] = None,
    segment_types: Optional[List[str]] = None,
    detected_mask: Optional[np.ndarray] = None,
    planned_mask: Optional[np.ndarray] = None,
    headland_mask: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    uncertain_mask: Optional[np.ndarray] = None,
    forbidden_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Validate one canonical planar rigid footprint against semantic masks.

    DECISION-009 keeps crop overlap based on ``processed_mask`` while the
    separate support mask is used only for field-boundary support checks.
    DECISION-011 requires display and validation to share the same configured
    track, body and cutter polygons around the track-contact centre.
    DECISION-013 keeps physical rolling separate from the calibrated agronomic
    damage cost: body and headland crop pixels are both physical crop evidence,
    while only the latter may receive an explicit planning weight.
    """
    cfg = config or {}
    if headland_mask is None:
        headland_mask = cfg.get("headland_mask")
    if support_mask is None:
        support_mask = cfg.get("planning_support_mask")
    if uncertain_mask is None:
        uncertain_mask = cfg.get("uncertain_mask")
    if forbidden_mask is None:
        forbidden_mask = cfg.get("forbidden_mask")
    specs = dict(harvester or {})
    geometry = planar_geometry_m(specs)

    raw = (np.asarray(processed_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    h, w = raw.shape[:2]
    mpp = meters_per_pixel(raw, geo, state)
    max_dim = int(cfg.get("validation_max_dim", 3000))
    scale = min(1.0, float(max_dim) / max(h, w))
    body = cv2.resize(
        raw,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    body = (body >= 96).astype(np.uint8) * 255
    detected = None
    if detected_mask is not None:
        detected_raw = (
            np.asarray(detected_mask, dtype=np.uint8) > 0
        ).astype(np.uint8) * 255
        if detected_raw.shape[:2] == raw.shape[:2]:
            detected = cv2.resize(
                detected_raw,
                (body.shape[1], body.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            detected = (detected >= 96).astype(np.uint8) * 255
    planned = None
    if planned_mask is not None:
        planned_raw = (
            np.asarray(planned_mask, dtype=np.uint8) > 0
        ).astype(np.uint8) * 255
        if planned_raw.shape[:2] == raw.shape[:2]:
            planned = cv2.resize(
                planned_raw,
                (body.shape[1], body.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            planned = (planned >= 96).astype(np.uint8) * 255
    headland = np.zeros_like(body)
    headland_present = headland_mask is not None
    if headland_mask is not None:
        headland_raw = (
            np.asarray(headland_mask, dtype=np.uint8) > 0
        ).astype(np.uint8) * 255
        if headland_raw.shape[:2] != raw.shape[:2]:
            raise ValueError("headland mask shape does not match processed mask")
        headland = cv2.resize(
            headland_raw,
            (body.shape[1], body.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        headland = (headland >= 96).astype(np.uint8) * 255
        # A pixel cannot be charged as both body and headland crop.
        headland = cv2.bitwise_and(headland, cv2.bitwise_not(body))
    canopy = cv2.bitwise_or(body, headland)
    support_source = raw
    semantic_support = None
    if support_mask is not None:
        candidate_support = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if candidate_support.shape[:2] == raw.shape[:2]:
            support_source = candidate_support
            semantic_support = cv2.resize(
                candidate_support,
                (body.shape[1], body.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    elif headland_present:
        support_source = cv2.bitwise_or(
            raw,
            (np.asarray(headland_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255,
        )
        semantic_support = cv2.resize(
            support_source,
            (body.shape[1], body.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    field_support = cv2.resize(
        _field_support(support_source, state),
        (body.shape[1], body.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    uncertain = None
    if uncertain_mask is not None:
        candidate = (np.asarray(uncertain_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if candidate.shape[:2] == raw.shape[:2]:
            uncertain = cv2.resize(
                candidate,
                (body.shape[1], body.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    forbidden = None
    if forbidden_mask is not None:
        candidate = (np.asarray(forbidden_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if candidate.shape[:2] == raw.shape[:2]:
            forbidden = cv2.resize(
                candidate,
                (body.shape[1], body.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    work_mpp = mpp / scale
    erosion_m = float(cfg.get("crop_core_erosion_m", 0.10))
    erosion_px = max(1, int(round(erosion_m / work_mpp)))
    core = cv2.erode(
        body,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (odd_size(erosion_px * 2 + 1), odd_size(erosion_px * 2 + 1)),
        ),
        iterations=1,
    )

    scaled_path = [
        [(float(x) * scale, float(y) * scale) for x, y in segment]
        for segment in full_path
    ]
    (
        track_swept,
        track_pose_occupancy_count,
        track_pass_count,
        cutter_swept,
        machine_swept,
        segment_track_sweeps,
        pose_step_m,
        pose_heading_window_m,
        pose_count,
    ) = _rasterize_planar_sweeps(
        body.shape[:2], scaled_path, segment_types, geometry, work_mpp, cfg
    )

    track_area = max(int(np.count_nonzero(track_swept)), 1)
    body_area_raw = int(np.count_nonzero(body))
    headland_area_raw = int(np.count_nonzero(headland))
    canopy_area_raw = int(np.count_nonzero(canopy))
    body_area = max(body_area_raw, 1)
    headland_area = max(headland_area_raw, 1)
    canopy_area = max(canopy_area_raw, 1)
    core_area = max(int(np.count_nonzero(core)), 1)
    track_body = int(np.count_nonzero((track_swept > 0) & (body > 0)))
    track_headland = int(np.count_nonzero((track_swept > 0) & (headland > 0)))
    track_canopy = int(np.count_nonzero((track_swept > 0) & (canopy > 0)))
    track_core = int(np.count_nonzero((track_swept > 0) & (core > 0)))
    crop_mask_covered = int(np.count_nonzero((cutter_swept > 0) & (body > 0)))
    detected_crop_area = max(int(np.count_nonzero(detected)), 1) if detected is not None else 1
    detected_mask_covered = (
        int(np.count_nonzero((cutter_swept > 0) & (detected > 0)))
        if detected is not None
        else 0
    )
    planned_crop_area = max(int(np.count_nonzero(planned)), 1) if planned is not None else body_area
    planned_target_covered = (
        int(np.count_nonzero((cutter_swept > 0) & (planned > 0)))
        if planned is not None
        else crop_mask_covered
    )
    outside_field = int(
        np.count_nonzero((track_swept > 0) & (field_support == 0))
    )
    outside_support = (
        int(np.count_nonzero((track_swept > 0) & (semantic_support == 0)))
        if semantic_support is not None
        else None
    )
    uncertain_track = (
        int(np.count_nonzero((track_swept > 0) & (uncertain > 0)))
        if uncertain is not None
        else None
    )
    forbidden_track = (
        int(np.count_nonzero((track_swept > 0) & (forbidden > 0)))
        if forbidden is not None
        else None
    )
    headland_weight = float(cfg.get("headland_damage_weight", 1.0))
    if not np.isfinite(headland_weight) or not 0.0 < headland_weight <= 1.0:
        raise ValueError("headland_damage_weight must be within (0, 1]")
    pixel_area_m2 = work_mpp * work_mpp
    track_rolling_visit_once_area_m2 = float(np.count_nonzero(track_pass_count == 1) * pixel_area_m2)
    track_rolling_visit_twice_or_more_area_m2 = float(np.count_nonzero(track_pass_count >= 2) * pixel_area_m2)
    segment_metrics = []
    prior_track = np.zeros_like(track_swept)
    turn_union_exclusive = np.zeros_like(track_swept)
    turn_union_raw = np.zeros_like(track_swept)
    for segment_index, segment_sweep in enumerate(segment_track_sweeps):
        exclusive = cv2.bitwise_and(segment_sweep, cv2.bitwise_not(prior_track))
        prior_track = cv2.bitwise_or(prior_track, segment_sweep)
        segment_type = str(segment_types[segment_index]) if segment_types and segment_index < len(segment_types) else ("work" if segment_index % 2 == 0 else "turn")
        exclusive_bool = exclusive > 0
        segment_area = max(int(np.count_nonzero(exclusive_bool)), 1)
        segment_length_m = sum(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])) for a, b in zip(full_path[segment_index], full_path[segment_index][1:])) * float(mpp) if segment_index < len(full_path) else 0.0
        segment_outside = int(np.count_nonzero(exclusive_bool & (field_support == 0)))
        segment_uncertain = int(np.count_nonzero(exclusive_bool & (uncertain > 0))) if uncertain is not None else None
        segment_forbidden = int(np.count_nonzero(exclusive_bool & (forbidden > 0))) if forbidden is not None else None
        segment_record = {
            "segment_index": int(segment_index),
            "segment_type": segment_type,
            "length_m": float(segment_length_m),
            "track_swept_area_m2": float(np.count_nonzero(exclusive_bool) * pixel_area_m2),
            "track_body_overlap_area_m2": float(np.count_nonzero(exclusive_bool & (body > 0)) * pixel_area_m2),
            "track_headland_overlap_area_m2": float(np.count_nonzero(exclusive_bool & (headland > 0)) * pixel_area_m2),
            "track_uncertain_overlap_area_m2": float(segment_uncertain * pixel_area_m2) if segment_uncertain is not None else None,
            "track_forbidden_overlap_area_m2": float(segment_forbidden * pixel_area_m2) if segment_forbidden is not None else None,
            "outside_field_area_m2": float(segment_outside * pixel_area_m2),
            "outside_field_pct": float(segment_outside / segment_area * 100.0),
            "turn_cost_m2": float((np.count_nonzero(exclusive_bool & (body > 0)) + float(cfg.get("headland_damage_weight", 1.0)) * np.count_nonzero(exclusive_bool & (headland > 0))) * pixel_area_m2) if segment_type.startswith("turn") else 0.0,
        }
        segment_metrics.append(segment_record)
        segment_record["agronomic_damage_cost_m2"] = float((
            np.count_nonzero(exclusive_bool & (body > 0))
            + headland_weight * np.count_nonzero(exclusive_bool & (headland > 0))
        ) * pixel_area_m2)
        if segment_type.startswith("turn"):
            turn_union_exclusive = cv2.bitwise_or(turn_union_exclusive, exclusive)
            turn_union_raw = cv2.bitwise_or(turn_union_raw, segment_sweep)
    turn_exclusive_bool = turn_union_exclusive > 0
    turn_raw_bool = turn_union_raw > 0
    turn_headland = int(np.count_nonzero(turn_exclusive_bool & (headland > 0)))
    turn_body = int(np.count_nonzero(turn_exclusive_bool & (body > 0)))
    turn_uncertain = int(np.count_nonzero(turn_exclusive_bool & (uncertain > 0))) if uncertain is not None else None
    turn_forbidden = int(np.count_nonzero(turn_exclusive_bool & (forbidden > 0))) if forbidden is not None else None
    turn_outside = int(np.count_nonzero(turn_exclusive_bool & (field_support == 0)))
    turn_area = max(int(np.count_nonzero(turn_exclusive_bool)), 1)
    turn_repeat = int(np.count_nonzero(turn_raw_bool & (track_pass_count >= 2)))
    turn_metrics = {
        "turn_segment_metrics": [item for item in segment_metrics if item["segment_type"].startswith("turn")],
        "headland_turn_track_overlap_area_m2": float(turn_headland * pixel_area_m2),
        "headland_turn_rolling_pct": float(turn_headland / headland_area_raw * 100.0) if headland_area_raw else None,
        "turn_body_overlap_area_m2": float(turn_body * pixel_area_m2),
        "turn_uncertain_overlap_area_m2": float(turn_uncertain * pixel_area_m2) if turn_uncertain is not None else None,
        "turn_forbidden_overlap_area_m2": float(turn_forbidden * pixel_area_m2) if turn_forbidden is not None else None,
        "turn_outside_field_area_m2": float(turn_outside * pixel_area_m2),
        "turn_outside_field_pct": float(turn_outside / turn_area * 100.0),
        "turn_repeat_twice_or_more_area_m2": float(turn_repeat * pixel_area_m2),
        "turn_cost_m2": float((turn_body + float(cfg.get("headland_damage_weight", 1.0)) * turn_headland) * pixel_area_m2),
        "turn_track_swept_area_m2": float(np.count_nonzero(turn_exclusive_bool) * pixel_area_m2),
    }
    physical_rolling_area_m2 = track_canopy * pixel_area_m2
    agronomic_damage_cost_m2 = (
        track_body + headland_weight * track_headland
    ) * pixel_area_m2
    machine_area = max(int(np.count_nonzero(machine_swept)), 1)
    machine_outside_field = int(
        np.count_nonzero((machine_swept > 0) & (field_support == 0))
    )
    machine_forbidden = (
        int(np.count_nonzero((machine_swept > 0) & (forbidden > 0)))
        if forbidden is not None
        else None
    )

    result = {
        "footprint_model": FOOTPRINT_MODEL,
        "footprint_reference_point": FOOTPRINT_REFERENCE_POINT,
        "footprint_dimensions_m": dict(geometry["dimensions_m"]),
        "footprint_pose_step_m": pose_step_m,
        "footprint_pose_heading_window_m": pose_heading_window_m,
        "footprint_pose_count": pose_count,
        "segment_metrics": segment_metrics,
        "turn_segment_metrics": turn_metrics["turn_segment_metrics"],
        "turn_metrics_attribution_rule": "exclusive_first_segment",
        "footprint_repeat_count_basis": "logical_path_segment_union",
        "track_rolling_visit_once_area_m2": track_rolling_visit_once_area_m2,
        "track_rolling_visit_twice_or_more_area_m2": track_rolling_visit_twice_or_more_area_m2,
        "track_pass_count_dtype": str(track_pass_count.dtype),
        "track_pose_occupancy_count_dtype": str(track_pose_occupancy_count.dtype),
        "headland_turn_track_overlap_area_m2": turn_metrics["headland_turn_track_overlap_area_m2"],
        "headland_turn_rolling_pct": turn_metrics["headland_turn_rolling_pct"],
        "turn_body_overlap_area_m2": turn_metrics["turn_body_overlap_area_m2"],
        "turn_uncertain_overlap_area_m2": turn_metrics["turn_uncertain_overlap_area_m2"],
        "turn_forbidden_overlap_area_m2": turn_metrics["turn_forbidden_overlap_area_m2"],
        "turn_outside_field_area_m2": turn_metrics["turn_outside_field_area_m2"],
        "turn_outside_field_pct": turn_metrics["turn_outside_field_pct"],
        "turn_repeat_twice_or_more_area_m2": turn_metrics["turn_repeat_twice_or_more_area_m2"],
        "turn_cost_m2": turn_metrics["turn_cost_m2"],
        "turn_track_swept_area_m2": turn_metrics["turn_track_swept_area_m2"],
        "track_swept_area_m2": track_area * pixel_area_m2,
        "machine_swept_area_m2": machine_area * pixel_area_m2,
        "body_crop_area_m2": body_area_raw * pixel_area_m2,
        "headland_crop_area_m2": headland_area_raw * pixel_area_m2,
        "canopy_crop_area_m2": canopy_area_raw * pixel_area_m2,
        "track_body_overlap_area_m2": track_body * pixel_area_m2,
        "track_headland_overlap_area_m2": track_headland * pixel_area_m2,
        "physical_crop_rolling_area_m2": physical_rolling_area_m2,
        "headland_damage_weight": headland_weight,
        "agronomic_damage_cost_m2": agronomic_damage_cost_m2,
        "agronomic_damage_cost_basis": (
            "track_body_overlap_area_m2 + headland_damage_weight * "
            "track_headland_overlap_area_m2"
        ),
        "track_canopy_overlap_pct": track_canopy / track_area * 100.0,
        "track_body_overlap_pct": track_body / track_area * 100.0,
        "track_core_overlap_pct": track_core / track_area * 100.0,
        "rolling_body_pct": track_body / body_area * 100.0 if body_area_raw else 0.0,
        "rolling_headland_pct": (
            track_headland / headland_area * 100.0
            if headland_present and headland_area_raw
            else (0.0 if headland_present else None)
        ),
        "rolling_canopy_pct": track_canopy / canopy_area * 100.0,
        "rolling_core_pct": track_core / core_area * 100.0,
        "crop_mask_geometric_coverage_pct": crop_mask_covered / body_area * 100.0,
        "planned_target_geometric_coverage_pct": planned_target_covered / planned_crop_area * 100.0,
        "detected_mask_geometric_coverage_pct": (
            detected_mask_covered / detected_crop_area * 100.0
            if detected is not None
            else 0.0
        ),
        "track_outside_field_pct": outside_field / track_area * 100.0,
        "track_outside_support_pct": (
            outside_support / track_area * 100.0
            if outside_support is not None
            else None
        ),
        "track_uncertain_overlap_pct": (
            uncertain_track / track_area * 100.0
            if uncertain_track is not None
            else None
        ),
        "track_forbidden_overlap_pct": (
            forbidden_track / track_area * 100.0
            if forbidden_track is not None
            else None
        ),
        "field_boundary_present": bool(
            state is not None and len(getattr(state, "field_boundary", None) or []) >= 3
        ),
        "semantic_support_present": semantic_support is not None,
        "uncertain_mask_present": uncertain is not None,
        "forbidden_mask_present": forbidden is not None,
        "headland_mask_present": headland_present,
        "track_headland_overlap_pct": track_headland / track_area * 100.0,
        "machine_outside_field_pct": machine_outside_field / machine_area * 100.0,
        "machine_forbidden_overlap_pct": (
            machine_forbidden / machine_area * 100.0
            if machine_forbidden is not None
            else None
        ),
        "footprint_resolution_m": float(work_mpp),
        "crop_core_erosion_m": erosion_m,
    }
    # DECISION-015: logical segment-union pass counts, not pose occupancy,
    # support the auditable once/repeated rolling metrics.
    manual_edit_mode = bool(cfg.get("manual_edit_mode", False))
    formal = assess_footprint_hard_constraints(
        result,
        cfg,
        require_explicit_valid=False,
        manual_edit_mode=manual_edit_mode,
    )
    result.update({
        "valid": formal["valid"],
        "issues": formal["issues"],
        "errors": formal.get("errors", []),
        "warnings": formal.get("warnings", []),
        "validation_profile": formal["validation_profile"],
        "validation_limits_pct": formal["validation_limits_pct"],
    })
    return result


def assess_footprint_hard_constraints(
    footprint: Dict, config: Optional[Dict] = None, *, crossing_count: int = 0,
    require_optional_layers: bool = False,
    require_explicit_valid: bool = True,
    manual_edit_mode: bool = False,
) -> Dict:
    """Apply structural footprint checks and optional metric thresholds.

    ``track_core_overlap_pct`` remains a required finite audit metric, but the
    default ``track_core_overlap_policy=report_only`` does not reject a route
    on that metric alone.  Set the policy to ``hard_gate`` when a caller
    explicitly needs the historical threshold gate.  The separate
    ``footprint_metric_policy=report_only`` default keeps rolling/coverage
    comparisons as report fields; it does not suppress missing metrics,
    coordinate integrity, crossings, or confirmed semantic/field violations.

    When ``manual_edit_mode=True``, issues are separated into hard errors
    (severity="error") and soft warnings (severity="warning"). Hard errors
    block export; soft warnings do not. This allows users to manually edit
    paths and receive guidance without being blocked by minor issues.
    """
    cfg = dict(config or {})
    metrics = dict(footprint or {})
    profile = str(cfg.get("validation_profile", metrics.get("validation_profile", "field_trial")))
    core_overlap_policy = str(
        cfg.get("track_core_overlap_policy", "report_only")
    ).strip().lower()
    if core_overlap_policy not in {"report_only", "hard_gate"}:
        core_overlap_policy = "report_only"
    metric_policy = str(
        cfg.get("footprint_metric_policy", "report_only")
    ).strip().lower()
    if metric_policy not in {"report_only", "hard_gate"}:
        metric_policy = "report_only"
    profile_limits = {
        "field_trial": {
            "outside_field": 1.0, "outside_support": 5.0,
            "uncertain": 10.0, "forbidden": 0.0,
        },
        "machine_candidate": {
            "outside_field": 0.5, "outside_support": 2.0,
            "uncertain": 5.0, "forbidden": 0.0,
            "machine_outside_field": 0.0,
            "machine_forbidden_overlap": 0.0,
        },
    }.get(profile)
    limits = {
        "track_core_overlap_pct": cfg.get("max_track_core_overlap_pct", 8.0),
        "planned_target_geometric_coverage_pct": cfg.get("min_geometric_coverage_pct", 90.0),
        "track_outside_field_pct": (
            profile_limits["outside_field"] if profile_limits
            else cfg.get("max_track_outside_field_pct", 2.0)
        ),
        "track_outside_support_pct": (
            profile_limits["outside_support"] if profile_limits
            else cfg.get("max_track_outside_support_pct", 100.0)
        ),
        "track_uncertain_overlap_pct": (
            profile_limits["uncertain"] if profile_limits
            else cfg.get("max_track_uncertain_overlap_pct", 100.0)
        ),
        "track_forbidden_overlap_pct": (
            profile_limits["forbidden"] if profile_limits
            else cfg.get("max_track_forbidden_overlap_pct", 0.0)
        ),
    }
    required = [
        "track_core_overlap_pct",
        "planned_target_geometric_coverage_pct",
        "track_outside_field_pct",
    ]
    optional_layers = (
        ("planning_support_mask", "track_outside_support_pct"),
        ("uncertain_mask", "track_uncertain_overlap_pct"),
        ("forbidden_mask", "track_forbidden_overlap_pct"),
    )
    for config_key, metric_key in optional_layers:
        if require_optional_layers or cfg.get(config_key) is not None:
            required.append(metric_key)
    if profile == "machine_candidate":
        required.extend(("machine_outside_field_pct", "machine_forbidden_overlap_pct"))

    reasons = []
    issues = []  # Changed to list of dicts with severity
    explicit_valid = metrics.get("valid")

    def add_issue(description: str, severity: str = "error"):
        """Helper to add an issue with severity classification and translation."""
        # 翻译为用户友好描述
        translated = translate_validation_issue(description, severity)
        issues.append({
            "description": translated,
            "severity": severity,
            "raw": description  # 保留原始技术描述用于调试
        })

    # Import existing issues (maintain backward compatibility)
    for issue in (metrics.get("issues") or []):
        if isinstance(issue, dict):
            issues.append(issue)
        elif str(issue):
            add_issue(str(issue), "error")

    if explicit_valid is False:
        reasons.append("validation_invalid")
    elif explicit_valid is None and require_explicit_valid:
        reasons.append("validation_validity_metric_missing")
        add_issue("required footprint validity evidence missing: valid", "error")
    elif explicit_valid is not None and explicit_valid is not True:
        reasons.append("validation_validity_metric_invalid")
    if issues:
        reasons.append("validation_issues")

    values = {}
    for name in required:
        try:
            value = float(metrics.get(name))
        except (TypeError, ValueError):
            value = float("nan")
        values[name] = value if np.isfinite(value) else None
        if values[name] is None:
            reasons.append(f"validation_metric_missing:{name}")
            add_issue(f"required footprint metric missing: {name}", "error")

    def add_exceeded(name, severity="error"):
        """Add exceeded metric with configurable severity."""
        value = values.get(name)
        try:
            limit = float(limits[name])
        except (KeyError, TypeError, ValueError):
            limit = float("nan")
        if value is not None and np.isfinite(limit) and value > limit:
            # Hard violations:田界越界、禁行区冲突 → error
            # Soft warnings: 略高的碾压率、略低的覆盖率 → warning (manual_edit_mode only)
            is_critical = name in (
                "track_outside_field_pct",
                "track_forbidden_overlap_pct",
                "machine_outside_field_pct",
                "machine_forbidden_overlap_pct"
            )
            issue_severity = "error" if (is_critical or not manual_edit_mode) else severity

            if issue_severity == "error":
                reasons.append(f"validation_gate_exceeded:{name}")

            add_issue(
                f"{name}={value:.6g} exceeds limit={limit:.6g}",
                issue_severity
            )

    if core_overlap_policy == "hard_gate":
        add_exceeded("track_core_overlap_pct", "warning" if manual_edit_mode else "error")
    if metric_policy == "hard_gate":
        coverage = values.get("planned_target_geometric_coverage_pct")
        try:
            minimum_coverage = float(limits["planned_target_geometric_coverage_pct"])
        except (TypeError, ValueError):
            minimum_coverage = float("nan")
        if (
            coverage is not None
            and np.isfinite(minimum_coverage)
            and coverage < minimum_coverage
        ):
            issue_severity = "warning" if manual_edit_mode else "error"
            if issue_severity == "error":
                reasons.append("validation_gate_below_minimum:planned_target_geometric_coverage_pct")
            add_issue(
                f"planned_target_geometric_coverage_pct={coverage:.6g} below minimum={minimum_coverage:.6g}",
                issue_severity
            )
    for name in (
        "track_outside_field_pct", "track_outside_support_pct",
        "track_uncertain_overlap_pct", "track_forbidden_overlap_pct",
    ):
        severity = "error" if name in ("track_outside_field_pct", "track_forbidden_overlap_pct") else "warning"
        add_exceeded(name, severity)

    if profile == "machine_candidate":
        for name, limit in (("machine_outside_field_pct", 0.0), ("machine_forbidden_overlap_pct", 0.0)):
            value = values.get(name)
            if value is not None and value > limit:
                reasons.append(f"validation_gate_exceeded:{name}")
                add_issue(f"{name}={value:.6g} exceeds limit={limit:.6g}", "error")

    if int(crossing_count or 0) > 0:
        reasons.append("non_adjacent_work_line_crossing")
        add_issue(f"non-adjacent work-line crossings: {int(crossing_count)}", "error")

    hard_reasons = list(dict.fromkeys(reasons))

    # Separate errors and warnings
    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]

    return {
        "valid": not hard_reasons,
        "hard_reasons": hard_reasons,
        "issues": issues,  # Full list with severity
        "errors": errors,  # Hard errors only
        "warnings": warnings,  # Soft warnings only
        "validation_profile": profile,
        "track_core_overlap_policy": core_overlap_policy,
        "footprint_metric_policy": metric_policy,
        "validation_limits_pct": {
            **limits,
            "machine_outside_field": 0.0 if profile == "machine_candidate" else None,
            "machine_forbidden_overlap": 0.0 if profile == "machine_candidate" else None,
        },
    }
