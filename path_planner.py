"""
路径规划器 — 中心线串联 / 多策略转弯 / 地理坐标转换 / 路径验证 / 输出

转弯策略 (文献依据):
  - bow:         Bow turning — R < ω/2 时的弓形转弯
  - semicircle:  Semicircular turning — R ≈ ω/2 时的半圆转弯
  - pear:        Pear-shaped turning — R > ω/2 时的三圆相切转弯
  - fishtail:    Fishtail turning — 田头受限时的倒车折返
"""
import math, csv, json, os, tempfile
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from shapely.geometry import LineString, Point, Polygon
from config import Config, AppLogger
from footprint_planner import generate_work_lines, validate_footprints
from row_geometry import field_polygon_local, meters_per_pixel, smooth_1d


PATH_PLANNING_VERSION = "footprint-path-v8"


def build_band_mask(shape, bands: List[Dict]) -> np.ndarray:
    """Build a planning-only mask without modifying the optimized crop mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    contours = []
    for band in bands:
        contour = band.get("contour")
        if contour is None:
            continue
        contour = np.asarray(contour, dtype=np.int32)
        if contour.ndim == 2 and contour.shape[0] >= 3:
            contours.append(contour)
    if contours:
        cv2.fillPoly(mask, contours, 255)
    return mask


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
    return [(float(x), float(y)) for x, y in fitted]


def prepare_band_centerlines(
    wide_bands: List[Dict],
    main_angle: float,
    geo=None,
    state=None,
    config: Optional[Dict] = None,
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
        centerline = _smooth_band_centerline(
            list(band.get("centerline", [])),
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
        })

    if not candidates:
        return [], {"reason": "no_valid_four_row_centerlines"}

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

    return [item["line"] for item in accepted], {
        "source": "four_row_band_centerlines",
        "candidate_four_row_count": len(wide_bands),
        "accepted_centerline_count": len(accepted),
        "median_centerline_span_m": median_span,
        "ordered_band_ids": [item["id"] for item in accepted],
        "pass_count": len(accepted),
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
) -> Tuple[List[Tuple[float, float]], str, Optional[str]]:
    """Generate a turn, shrinking outward clearance when it would leave the selected field."""
    turn_function = TURN_STRATEGIES.get(selected_strategy, _turn_bow)
    clearances = []
    for scale in (1.0, 0.5, 0.25, 0.0):
        value = max(0.0, float(clearance_m) * scale)
        if value not in clearances:
            clearances.append(value)
    best_turn = None
    for value in clearances:
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
    return max(0.0, float(clearance_m)) / max(mpp, 1e-6)


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
                     clearance_m: float = 0.0) -> List[Tuple[float, float]]:
    """Pure semicircular turn whose radius is half the pass spacing."""
    start_v, end_v, distance, lateral, outward = _turn_frame(
        start, end, work_dir
    )
    if distance < 2.0 or lateral is None:
        return [start, end]
    physical_radius_m = distance * max(mpp, 1e-6) * 0.5
    if physical_radius_m < float(min_radius_m) - 1e-6:
        raise ValueError(
            f"semicircle radius {physical_radius_m:.3f} m is below the "
            f"minimum turn radius {float(min_radius_m):.3f} m"
        )
    clearance = _turn_clearance_px(clearance_m, mpp)
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
}

TURN_STRATEGY_LABELS = {
    "bow": "弓形转弯",
    "semicircle": "半圆转弯",
    "pear": "梨形转弯",
    "fishtail": "鱼尾折返",
    "alpha": "紧凑α折返",
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
    two_radius = 2.0 * radius
    fallback_from = None
    fallback_reason = ""

    # DECISION-010: omega/2 below R is never an executable semicircle.
    if (
        spacing < two_radius - 1e-6
        and requested in {"auto", "bow", "semicircle"}
    ):
        strategy = "fishtail" if headland_limited else "pear"
        fallback_from = None if requested == "auto" else requested
        fallback_reason = (
            f"omega={spacing:.2f}m gives omega/2={spacing * 0.5:.2f}m "
            f"below R={radius:.2f}m; using {TURN_STRATEGY_LABELS[strategy]} "
            "instead of a subminimum-radius semicircle"
        )
        return {
            "requested_strategy": requested,
            "strategy": strategy,
            "reason": fallback_reason,
            "fallback_from": fallback_from,
            "fallback_reason": fallback_reason if fallback_from else "",
            "pass_spacing_m": spacing,
            "turn_radius_m": radius,
        }

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

    return {
        "requested_strategy": requested,
        "strategy": strategy,
        "reason": reason,
        "fallback_from": fallback_from,
        "fallback_reason": fallback_reason,
        "pass_spacing_m": spacing,
        "turn_radius_m": radius,
    }


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
                hard_reasons.append(
                    f"{len(below)}处行距ω小于2R（最小{minimum:.2f}m，"
                    f"2R={two_radius:.2f}m），弓形转弯的两段R圆弧无法相切连接"
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
        "strategy": selected,
        "label": TURN_STRATEGY_LABELS.get(selected, selected),
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
    cutter_forward_offset = values.get(
        "cutter_forward_offset_m", cfg.CUTTER_FORWARD_OFFSET_M
    )
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
            low_side = projection <= midpoint
            target_projection = target_low if low_side else target_high
            delta_px = target_projection - projection
            outward = delta_px < -1e-9 if low_side else delta_px > 1e-9
            extension_m = abs(delta_px) * mpp
            side = "low" if low_side else "high"
            outlier_threshold_m = side_thresholds_m[side]
            if not outward or extension_m < outlier_threshold_m:
                continue
            target = (
                float(point[0] + direction[0] * delta_px),
                float(point[1] + direction[1] * delta_px),
            )
            blocked_reason = ""
            if extension_m > max_extension_m:
                blocked_reason = "extension_limit"
            elif field_polygon is None:
                blocked_reason = "missing_field_boundary"
            elif not _segment_inside_field_boundary([point, target], field_polygon):
                blocked_reason = "outside_field_boundary"
            if blocked_reason:
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


def _connect_work_lines(
        work_lines: List[List[Tuple[float, float]]],
        geo=None,
        state=None,
        min_turn_radius_m: float = 1.0,
        turn_strategy: str = "bow",
        config: Optional[Dict] = None,
        return_diagnostics: bool = False,
        global_main_angle: Optional[float] = None,
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

    entry_point = to_local(entry_global)
    exit_point = to_local(exit_global)
    unload_points = [to_local(point) for point in unload_globals if point]
    ordered_lines = [list(line) for line in work_lines]
    if entry_point and len(ordered_lines) >= 2:
        first_distance = min(
            math.hypot(point[0] - entry_point[0], point[1] - entry_point[1])
            for point in (ordered_lines[0][0], ordered_lines[0][-1])
        )
        last_distance = min(
            math.hypot(point[0] - entry_point[0], point[1] - entry_point[1])
            for point in (ordered_lines[-1][0], ordered_lines[-1][-1])
        )
        if last_distance < first_distance:
            ordered_lines.reverse()

    start_reversed = False
    if entry_point and ordered_lines and len(ordered_lines[0]) >= 2:
        first = ordered_lines[0]
        distance_start = math.hypot(
            first[0][0] - entry_point[0],
            first[0][1] - entry_point[1],
        )
        distance_end = math.hypot(
            first[-1][0] - entry_point[0],
            first[-1][1] - entry_point[1],
        )
        start_reversed = distance_end < distance_start

    oriented = []
    for index, line in enumerate(ordered_lines):
        points = list(line)
        if bool(index % 2) ^ start_reversed:
            points.reverse()
        if len(points) >= 2:
            oriented.append(points)

    full_path = []
    segment_types = []
    pass_spacings_m = []
    used_strategies = []
    turn_decisions = []
    fallback_reasons = []
    boundary_adjustments = []
    boundary_hard_reasons = []
    field_polygon = _field_boundary_local_polygon(state)
    alignment_angle = global_main_angle
    alignment_direction_source = "global_main_angle"
    if alignment_angle is None or not math.isfinite(float(alignment_angle)):
        weighted_cos = 0.0
        weighted_sin = 0.0
        for line in oriented:
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
    turn_anchors, endpoint_alignment = _align_short_turn_anchors(
        oriented,
        main_angle=float(alignment_angle),
        mpp=mpp,
        field_polygon=field_polygon,
        config=cfg,
    )
    endpoint_alignment["direction_source"] = alignment_direction_source

    endpoint_alignment["path_reference_shift_applied"] = False

    if entry_point and oriented:
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
        decision = _select_turn_strategy(
            requested_strategy,
            gap_m,
            min_turn_radius_m,
        )
        selected_strategy = decision["strategy"]
        turn_decisions.append({
            "transition_index": index,
            "requested_strategy": decision.get("requested_strategy"),
            "strategy": selected_strategy,
            "pass_spacing_m": float(gap_m),
            "turn_radius_m": float(min_turn_radius_m),
            "reason": decision.get("reason", ""),
            "fallback_reason": decision.get("fallback_reason", ""),
            "crop_exit": [float(line[-1][0]), float(line[-1][1])],
            "turn_start": [float(start[0]), float(start[1])],
            "turn_end": [float(end[0]), float(end[1])],
            "crop_entry": [float(next_line[0][0]), float(next_line[0][1])],
        })
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
                clearance_m,
                inner=(selected_strategy == "alpha"),
            )
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
                    full_path.append(direct)
                    segment_types.append("turn")
                    boundary_adjustments.append(
                        "倒车折返会越出框选田块，已退化为边界内端点连接"
                    )
                    boundary_hard_reasons.append(
                        "reverse turn geometry degraded to a straight segment because no "
                        "kinematically valid reverse manoeuvre fit inside the field boundary"
                    )
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
                min_turn_radius_m,
                selected_strategy,
                clearance_m,
                field_polygon,
            )
            if boundary_reason:
                boundary_adjustments.append(boundary_reason)
                if len(turn) <= 2:
                    boundary_hard_reasons.append(
                        "turn geometry degraded to a straight segment because no "
                        "kinematically valid curve fit inside the field boundary"
                    )
            full_path.append(turn)
            segment_types.append("turn")
        if math.hypot(
            end[0] - next_line[0][0],
            end[1] - next_line[0][1],
        ) > 1e-7:
            full_path.append([end, next_line[0]])
            segment_types.append("turn_approach")

    if exit_point and oriented:
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
            full_path.append([route_end, exit_point])
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
        "used_strategies": used_strategies,
        "fallback_reasons": fallback_reasons,
        "boundary_constraints": {
            "field_boundary_present": field_polygon is not None,
            "turns_adjusted": len(boundary_adjustments),
            "adjustment_reasons": boundary_adjustments,
            "non_work_violations": non_work_boundary_violations,
        },
        "pass_spacings_m": pass_spacings_m,
        "turn_decisions": turn_decisions,
        "endpoint_alignment": endpoint_alignment,
        "service_points": {
            "entry_present": entry_point is not None,
            "exit_present": exit_point is not None,
            "unload_count": len(unload_points),
            "unload_visit_count": sum(1 for item in segment_types if item == "unload"),
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

    full_path 中偶数索引为工作段，奇数索引为转弯段。
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
    crossing_count = _count_path_intersections_proper(full_path)
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
    if footprint["track_core_overlap_pct"] > max_core_overlap:
        issues.append(
            f"履带与稻株核心重叠 {footprint['track_core_overlap_pct']:.1f}% "
            f"> {max_core_overlap:.1f}%"
        )
    if footprint["planned_target_geometric_coverage_pct"] < min_geometric_coverage:
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
    machine_outside = footprint.get("machine_outside_field_pct")
    if max_machine_outside is not None:
        if machine_outside is None:
            issues.append("缺少整机平面包络田块越界校验")
        elif machine_outside > max_machine_outside:
            issues.append(
                f"整机平面包络越出田块 {machine_outside:.1f}% "
                f"> {max_machine_outside:.1f}%"
            )
    machine_forbidden = footprint.get("machine_forbidden_overlap_pct")
    if max_machine_forbidden is not None:
        if machine_forbidden is None:
            issues.append("缺少整机平面包络禁行区校验")
        elif machine_forbidden > max_machine_forbidden:
            issues.append(
                f"整机平面包络进入禁行区 {machine_forbidden:.1f}% "
                f"> {max_machine_forbidden:.1f}%"
            )
    if crossing_count > 0:
        issues.append(f"非相邻作业线交叉 {crossing_count} 处")

    result = {
        "valid": len(issues) == 0,
        "total_length_m": float(total_length_m),
        "work_length_m": float(work_length_m),
        "turn_length_m": float(turn_length_m),
        "entry_exit_length_m": float(entry_exit_length_m),
        "crossing_count": int(crossing_count),
        "issues": issues,
        "validation_profile": profile,
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
        **footprint,
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
    is_valid = (
        metrics["track_core_overlap_pct"] <= 8.0
        and metrics["crop_mask_geometric_coverage_pct"] >= 90.0
        and metrics["track_outside_field_pct"] <= 2.0
    )
    description = (
        f"履带核心重叠={metrics['track_core_overlap_pct']:.1f}%, "
        f"全田作物预计碾压={metrics['rolling_canopy_pct']:.1f}%, "
        f"作物核心碾压={metrics['rolling_core_pct']:.1f}%, "
        f"作物掩膜几何覆盖={metrics['crop_mask_geometric_coverage_pct']:.1f}%, "
        f"越界={metrics['track_outside_field_pct']:.1f}%"
    )
    return is_valid, description


def _count_path_intersections_proper(
        full_path: List[List[Tuple[float, float]]]) -> int:
    """统计非相邻工作段之间的交叉次数。

    使用线段-线段交叉检测（参数化方法），
    仅检查间隔 ≥ 2 的工作段对（跳过相邻段，它们通过转弯段连接属于正常）。
    """
    # 提取所有工作段
    work_segs = []
    for i in range(0, len(full_path), 2):
        work_segs.append((i, full_path[i]))

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


def _candidate_strategies_for_request(requested_strategy: str) -> List[str]:
    """Return executable strategies to evaluate for a user request.

    Auto should compare the full strategy family instead of trusting one early
    heuristic. Manual requests stay single-strategy, while _select_turn_strategy
    may still fallback if that strategy is geometrically impossible.
    """
    requested = str(requested_strategy or "auto").strip().lower()
    if requested == "auto":
        return ["semicircle", "bow", "pear", "fishtail", "alpha"]
    return [normalize_turn_strategy(requested)]


def _rank_path_candidates(candidates: List[Dict]) -> List[Dict]:
    """Rank candidate routes by agronomic safety first, then efficiency.

    Lower score is better. The score intentionally penalizes invalid geometry,
    crop-core rolling, field-boundary excursions and self-crossings far more
    than a modest length increase. DECISION-012 preserves physical canopy
    rolling as an unweighted measurement. DECISION-013 uses the separately
    named agronomic damage cost as the first route objective after hard
    validity constraints; physical rolling remains the next criterion.
    """
    ranked = []
    for candidate in candidates or []:
        item = dict(candidate)
        validation = item.get("validation") or {}
        assessment = item.get("turn_assessment") or {}
        segment_types = item.get("segment_types") or []
        valid_penalty = 0.0 if validation.get("valid", False) else 10000.0
        hard_penalty = 1500.0 * len(assessment.get("hard_reasons") or [])
        crossing_penalty = 1200.0 * float(assessment.get("turn_self_crossing_count", 0) or 0)
        rolling_value = validation.get("rolling_canopy_pct")
        rolling_missing = rolling_value is None
        rolling_canopy = (
            101.0 if rolling_missing else float(rolling_value or 0.0)
        )
        damage_value = validation.get("agronomic_damage_cost_m2")
        damage_missing = damage_value is None
        # Legacy/synthetic candidates may not carry the v8 metric. Falling
        # back to physical rolling preserves deterministic ordering without
        # inventing a field-calibration weight.
        agronomic_damage = (
            rolling_canopy if damage_missing else float(damage_value or 0.0)
        )
        core_overlap = float(validation.get("track_core_overlap_pct", 0.0) or 0.0)
        outside = float(validation.get("track_outside_field_pct", 0.0) or 0.0)
        outside_support = float(validation.get("track_outside_support_pct", 0.0) or 0.0)
        uncertain_overlap = float(validation.get("track_uncertain_overlap_pct", 0.0) or 0.0)
        forbidden_overlap = float(validation.get("track_forbidden_overlap_pct", 0.0) or 0.0)
        profile = str(validation.get("validation_profile", "research"))
        semantic_weight = 1.0 if profile == "research" else 4.0
        coverage = float(
            validation.get(
                "crop_mask_geometric_coverage_pct",
                validation.get("planned_target_geometric_coverage_pct", 0.0),
            ) or 0.0
        )
        total_length = float(validation.get("total_length_m", 0.0) or 0.0)
        reverse_turns = sum(1 for value in segment_types if str(value).startswith("turn_reverse"))
        unload_visits = sum(1 for value in segment_types if value == "unload")
        score = (
            valid_penalty * 100000000.0
            + hard_penalty * 1000000.0
            + crossing_penalty * 1000000.0
            + agronomic_damage * 1000000.0
            + rolling_canopy * 1000.0
            + core_overlap * 80.0
            + outside * 120.0
            + outside_support * 25.0 * semantic_weight
            + uncertain_overlap * 15.0 * semantic_weight
            + forbidden_overlap * 5000.0
            + max(0.0, 95.0 - coverage) * 25.0
            + total_length * 0.08
            + reverse_turns * 8.0
        )
        item["candidate_score"] = float(score)
        item["candidate_metrics"] = {
            "valid_penalty": valid_penalty,
            "hard_reason_count": len(assessment.get("hard_reasons") or []),
            "turn_self_crossing_count": int(assessment.get("turn_self_crossing_count", 0) or 0),
            "rolling_canopy_pct": None if rolling_missing else rolling_canopy,
            "agronomic_damage_cost_m2": (
                None if damage_missing else agronomic_damage
            ),
            "headland_damage_weight": validation.get("headland_damage_weight"),
            "track_core_overlap_pct": core_overlap,
            "track_outside_field_pct": outside,
            "track_outside_support_pct": outside_support,
            "track_uncertain_overlap_pct": uncertain_overlap,
            "track_forbidden_overlap_pct": forbidden_overlap,
            "crop_mask_geometric_coverage_pct": coverage,
            "total_length_m": total_length,
            "reverse_turns": reverse_turns,
            "unload_visits": unload_visits,
        }
        item["candidate_rank_key"] = (
            0 if validation.get("valid", False) else 1,
            len(assessment.get("hard_reasons") or []),
            int(assessment.get("turn_self_crossing_count", 0) or 0),
            1 if damage_missing else 0,
            agronomic_damage,
            1 if rolling_missing else 0,
            rolling_canopy,
            forbidden_overlap,
            outside,
            outside_support,
            uncertain_overlap,
            core_overlap,
            -coverage,
            total_length,
            str(item.get("strategy", "")),
        )
        ranked.append(item)
    return sorted(ranked, key=lambda value: value["candidate_rank_key"])


def _build_planning_factor_report(
    state=None,
    work_lines: Optional[List[List[Tuple[float, float]]]] = None,
    layout: Optional[Dict] = None,
    turn_assessment: Optional[Dict] = None,
    validation: Optional[Dict] = None,
) -> Dict:
    """Summarise the physical and agronomic factors used by route planning."""
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

def _prepare_work_line_layout(
    wide_bands: List[Dict],
    processed_mask: np.ndarray,
    main_angle: float,
    geo=None,
    state=None,
    config: Optional[Dict] = None,
    progress_callback=None,
) -> Tuple[List[List[Tuple[float, float]]], Dict]:
    """Select the declared work-line generator without silent mode fallback."""
    cfg = dict(config or {})
    mode = str(cfg.get("work_line_mode", "footprint_optimized")).strip().lower()
    if mode == "footprint_optimized":
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
            )
            layout["footprint_optimization_scope"] = "validation_only"
        if not work_lines and wide_bands:
            work_lines = [
                list(item.get("centerline", []))
                for item in wide_bands
                if len(item.get("centerline", [])) >= 2
            ]
            layout = {"fallback": "raw_band_centerlines", "footprint_optimization_scope": "validation_only"}
    elif mode == "band_centerline":
        work_lines, layout = prepare_band_centerlines(
            wide_bands,
            main_angle,
            geo=geo,
            state=state,
            config=cfg,
        )
        if not work_lines and wide_bands:
            work_lines = [
                list(item.get("centerline", []))
                for item in wide_bands
                if len(item.get("centerline", [])) >= 2
            ]
            layout = {"fallback": "raw_band_centerlines"}
    else:
        raise ValueError(f"unsupported work_line_mode: {mode}")
    layout = dict(layout or {})
    layout["work_line_mode"] = mode
    return work_lines, layout


def plan_path(wide_bands: List[Dict],
              processed_mask: np.ndarray,
              main_angle: float,
              geo=None, state=None,
              config: Optional[Dict] = None,
              progress_callback=None,
              obstacle_mask: Optional[np.ndarray] = None,
              detected_mask: Optional[np.ndarray] = None,
              planned_mask: Optional[np.ndarray] = None) -> Dict:
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

    _report(0.12, "四行带中心线筛选")
    abort_if_cancelled()
    work_lines, layout = _prepare_work_line_layout(
        wide_bands,
        processed_mask,
        main_angle,
        geo=geo,
        state=state,
        config=cfg,
        progress_callback=progress_callback,
    )
    abort_if_cancelled()
    if not work_lines:
        return {
            "full_path": [],
            "segment_types": [],
            "geo_points": [],
            "validation": {"valid": False, "issues": ["未生成有效作业线"]},
            "is_valid": False,
            "description": "未生成有效作业线",
            "layout": layout,
            "requested_strategy": turn_strategy,
            "turn_assessment": {
                "strategy": turn_strategy,
                "label": TURN_STRATEGY_LABELS.get(
                    turn_strategy,
                    turn_strategy,
                ),
                "feasible": False,
                "needs_confirmation": False,
                "hard_reasons": ["未生成有效作业线"],
                "warnings": [],
            },
        }

    _report(0.76, "地头转弯候选生成")

    def evaluate_strategy(strategy_key: str) -> Dict:
        abort_if_cancelled()
        candidate_segments, candidate_types, candidate_assessment = _connect_work_lines(
            work_lines,
            geo=geo,
            state=state,
            min_turn_radius_m=min_turn_radius,
            turn_strategy=strategy_key,
            config=cfg,
            return_diagnostics=True,
            global_main_angle=main_angle,
        )
        candidate_assessment["user_requested_strategy"] = turn_strategy
        for decision in candidate_assessment.get("turn_decisions", []) or []:
            decision["user_requested_strategy"] = turn_strategy
            decision["candidate_strategy"] = strategy_key
            if turn_strategy == "auto" and decision.get("reason") == "用户指定":
                decision["reason"] = "自动多策略评估中的候选策略满足该处几何约束"
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
            "path_segments": candidate_segments,
            "segment_types": candidate_types,
            "validation": candidate_validation,
            "turn_assessment": candidate_assessment,
        }

    _report(0.84, "多策略足迹验证")
    candidate_results = []
    for strategy_key in _candidate_strategies_for_request(turn_strategy):
        abort_if_cancelled()
        candidate_results.append(evaluate_strategy(strategy_key))
    ranked_candidates = _rank_path_candidates(candidate_results)
    selected_candidate = ranked_candidates[0] if ranked_candidates else evaluate_strategy(turn_strategy)
    path_segments = selected_candidate["path_segments"]
    segment_types = selected_candidate["segment_types"]
    validation = selected_candidate["validation"]
    turn_assessment = selected_candidate["turn_assessment"]
    turn_assessment["selected_candidate_strategy"] = selected_candidate.get("strategy")
    turn_assessment["user_requested_strategy"] = turn_strategy
    turn_assessment["candidate_score"] = selected_candidate.get("candidate_score")
    turn_assessment["candidate_metrics"] = selected_candidate.get("candidate_metrics", {})
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
                "score": item.get("candidate_score"),
                "metrics": item.get("candidate_metrics", {}),
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
    return result
