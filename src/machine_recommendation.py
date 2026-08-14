"""Evidence-bounded harvester dimension guidance for wide/narrow-row fields."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


TRACK_WIDTH_OPTIONS_M = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
TRACK_CLEARANCE_RESERVE_M = 0.15
HEADLAND_LONGITUDINAL_RESERVE_M = 0.60
CUTTER_WIDTH_STEP_M = 0.05
CUTTER_SIDE_MARGIN_MIN_M = 0.06
CUTTER_SIDE_MARGIN_RATIO = 0.04


def _finite_positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _wide_cluster_indices(values: np.ndarray) -> np.ndarray:
    """Return indices of the larger 1-D gap cluster without forcing two classes."""
    if values.size < 3 or float(np.ptp(values)) < 0.08:
        return np.arange(values.size, dtype=np.int64)
    low, high = np.quantile(values, (0.25, 0.75))
    for _ in range(20):
        threshold = (low + high) * 0.5
        low_values = values[values < threshold]
        high_values = values[values >= threshold]
        if low_values.size == 0 or high_values.size == 0:
            return np.arange(values.size, dtype=np.int64)
        next_low = float(np.mean(low_values))
        next_high = float(np.mean(high_values))
        if abs(next_low - low) + abs(next_high - high) < 1e-6:
            low, high = next_low, next_high
            break
        low, high = next_low, next_high
    if high - low < 0.08:
        return np.arange(values.size, dtype=np.int64)
    return np.flatnonzero(values >= (low + high) * 0.5)


def _corridor_geometry(mask_result: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics = dict(mask_result.get("diagnostics") or {})
    mpp = _finite_positive(diagnostics.get("meters_per_pixel"))
    bands = []
    angle = float(mask_result.get("main_angle", 0.0) or 0.0)
    normal = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    if mpp is not None:
        for band in mask_result.get("all_bands") or []:
            if not isinstance(band, Mapping) or not band.get("centroid"):
                continue
            try:
                centroid = np.asarray(band["centroid"], dtype=np.float64)[:2]
            except (TypeError, ValueError):
                continue
            width = _finite_positive(
                band.get("classification_width_m", band.get("width_m"))
            )
            if centroid.size == 2 and np.all(np.isfinite(centroid)) and width is not None:
                bands.append((float(np.dot(centroid, normal) * mpp), width))
    bands.sort(key=lambda item: item[0])
    if len(bands) < 2:
        return {"all_clearances": [], "wide_clearances": [], "wide_centers": []}
    clearances: List[float] = []
    centers: List[float] = []
    for left, right in zip(bands, bands[1:]):
        spacing = right[0] - left[0]
        clear = spacing - 0.5 * (left[1] + right[1])
        if clear > 0:
            clearances.append(float(clear))
            centers.append(float((left[0] + right[0]) * 0.5))
    if not clearances:
        return {"all_clearances": [], "wide_clearances": [], "wide_centers": []}
    values = np.asarray(clearances, dtype=np.float64)
    indices = _wide_cluster_indices(values)
    return {
        "all_clearances": clearances,
        "wide_clearances": [clearances[int(index)] for index in indices],
        "wide_centers": [centers[int(index)] for index in indices],
    }


def _headland_depth(mask_result: Mapping[str, Any]) -> Optional[float]:
    diagnostics = dict(mask_result.get("diagnostics") or {})
    corridor = dict(diagnostics.get("headland_corridor") or {})
    depths = [
        value
        for item in corridor.get("candidates") or []
        if isinstance(item, Mapping) and item.get("decision") == "accepted"
        if (value := _finite_positive(item.get("depth_m"))) is not None
    ]
    return float(min(depths)) if depths else None


def _largest_option_not_above(limit: float) -> float:
    options = [value for value in TRACK_WIDTH_OPTIONS_M if value <= limit + 1e-9]
    if options:
        return float(max(options))
    return max(0.20, math.floor(limit / 0.05 + 1e-9) * 0.05)


def _round_up_step(value: float, step: float = CUTTER_WIDTH_STEP_M) -> float:
    return math.ceil(float(value) / step - 1e-9) * step


def _matched_cutter_width(params: Mapping[str, Any]) -> Optional[float]:
    track_width = _finite_positive(params.get("track_width_m"))
    track_gauge = _finite_positive(params.get("track_gauge_m"))
    if track_width is None or track_gauge is None:
        return None
    track_outer_width = track_gauge + track_width
    side_margin = max(
        CUTTER_SIDE_MARGIN_MIN_M,
        track_outer_width * CUTTER_SIDE_MARGIN_RATIO,
    )
    return float(_round_up_step(track_outer_width + 2.0 * side_margin))


def recommend_machine_dimensions(
    mask_result: Mapping[str, Any],
    current_params: Mapping[str, Any],
    path_result: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a conservative candidate, preserving unknown dimensions unchanged."""
    params = {
        str(key): (None if value is None else float(value))
        for key, value in current_params.items()
    }
    geometry = _corridor_geometry(mask_result or {})
    wide_clearances = geometry["wide_clearances"]
    wide_centers = geometry["wide_centers"]
    clear_width = (
        float(np.quantile(wide_clearances, 0.10)) if wide_clearances else None
    )
    track_upper = (
        max(0.0, clear_width - TRACK_CLEARANCE_RESERVE_M)
        if clear_width is not None else None
    )
    if track_upper is not None and track_upper >= 0.20:
        params["track_width_m"] = _largest_option_not_above(track_upper)

    gauge_target = None
    if len(wide_centers) >= 2:
        gauge_differences = np.diff(np.asarray(wide_centers, dtype=np.float64))
        gauge_differences = gauge_differences[
            (gauge_differences >= 1.0) & (gauge_differences <= 2.6)
        ]
        if gauge_differences.size:
            gauge_target = round(float(np.median(gauge_differences)) * 10.0) / 10.0
            params["track_gauge_m"] = gauge_target

    headland_depth = _headland_depth(mask_result or {})
    wheelbase_upper = (
        max(1.0, headland_depth - HEADLAND_LONGITUDINAL_RESERVE_M)
        if headland_depth is not None else None
    )
    if wheelbase_upper is not None:
        params["wheelbase_m"] = max(
            1.0,
            math.floor(
                min(float(params.get("wheelbase_m", 1.8)), wheelbase_upper) * 10.0
                + 1e-9
            ) / 10.0,
        )
        if "track_length_m" in params:
            params["track_length_m"] = min(
                float(params["track_length_m"]), params["wheelbase_m"]
            )

    has_machine_geometry_evidence = any(
        value is not None for value in (track_upper, gauge_target, wheelbase_upper)
    )
    matched_cutter_width = (
        _matched_cutter_width(params) if has_machine_geometry_evidence else None
    )
    if matched_cutter_width is not None:
        params["cutter_width_m"] = matched_cutter_width

    evidence_count = int(track_upper is not None) + int(wheelbase_upper is not None)
    evidence_quality = ("complete" if evidence_count == 2 else
                        "partial" if evidence_count == 1 else "insufficient")
    lines = ["智能推荐参数"]
    if clear_width is not None:
        lines.extend([
            f"- 宽通道保守净宽：{clear_width:.2f} m",
            f"- 预留 {TRACK_CLEARANCE_RESERVE_M:.2f} m 安全余量后，履带宽度影像约束上限：{track_upper:.2f} m",
            f"- 推荐履带宽度：{params.get('track_width_m', 0.0) * 100:.0f} cm",
        ])
    else:
        lines.append("- 行带几何证据不足，无法计算履带宽度上限，保留当前值。")
    if gauge_target is not None:
        lines.append(f"- 宽通道中心距估计：{gauge_target:.1f} m，作为履带中心距候选。")
    else:
        lines.append("- 可配对宽通道不足，履带中心距保留当前值。")
    if headland_depth is not None:
        lines.extend([
            f"- 两端田头保守深度：{headland_depth:.2f} m",
            f"- 扣除纵向余量后，轴距上限指导值：{wheelbase_upper:.2f} m",
            f"- 推荐轴距：{params.get('wheelbase_m', 0.0):.1f} m",
        ])
    else:
        lines.append("- 缺少可靠田头深度证据，无法计算轴距上限，保留当前值。")
    if matched_cutter_width is not None:
        outer_width = float(params["track_gauge_m"]) + float(params["track_width_m"])
        lines.append(
            f"- 自动候选割幅按履带外宽 {outer_width:.2f} m 成套匹配，"
            f"推荐有效割幅：{matched_cutter_width:.2f} m。"
        )
    lines.append("- 割刀前置距离不能由影像或行带几何推算，必须对实际收获机测量。")
    lines.extend([
        "",
        "提示：当前推荐只使用平面行带和田头几何，不含坡度、土壤或重心模型。",
        "提示：参数建议基于当前影像中的行带宽度和田头空间，请结合实际机具尺寸确认。",
    ])
    return {
        "title": "基于宽窄行几何的农机尺寸候选",
        "message": "\n".join(lines),
        "params": params,
        "limits": {
            "wide_corridor_clear_m": clear_width,
            "track_width_upper_m": track_upper,
            "track_gauge_target_m": gauge_target,
            "headland_depth_m": headland_depth,
            "wheelbase_upper_guide_m": wheelbase_upper,
            "matched_cutter_width_m": matched_cutter_width,
        },
        "evidence_quality": evidence_quality,
        "method": "mask_geometry_heuristic_v2",
    }
