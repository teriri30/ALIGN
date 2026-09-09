"""Metric, row-aligned geometry reconstruction for crop-band masks."""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

MASK_REGULARIZATION_VERSION = 18


def odd_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 == 1 else size + 1


def meters_per_pixel(mask: np.ndarray, geo=None, state=None) -> float:
    if geo is not None and state is not None and geo.is_ready():
        ox = int(getattr(state, "mask_offset_x", 0))
        oy = int(getattr(state, "mask_offset_y", 0))
        h, w = mask.shape[:2]
        value = float(geo.meters_per_pixel(ox + w / 2, oy + h / 2))
        if 1e-5 <= value <= 5.0:
            return value
        raise ValueError(f"invalid metric scale: {value!r} m/px")
    raise ValueError(
        "verified metric scale requires ready georeferencing and image state"
    )


def field_polygon_local(
    state,
    shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Return the selected field polygon in mask-local coordinates."""
    boundary = getattr(state, "field_boundary", None) if state is not None else None
    if not boundary or len(boundary) < 3:
        return None
    ox = float(getattr(state, "mask_offset_x", 0))
    oy = float(getattr(state, "mask_offset_y", 0))
    points = np.asarray(boundary, dtype=np.float64) - np.asarray((ox, oy))
    if not np.all(np.isfinite(points)):
        return None
    h, w = shape[:2]
    if (
        np.max(points[:, 0]) < 0
        or np.max(points[:, 1]) < 0
        or np.min(points[:, 0]) >= w
        or np.min(points[:, 1]) >= h
    ):
        return None
    return points


def clip_mask_to_polygon_stripes(
    mask: np.ndarray,
    polygon: Optional[np.ndarray],
    stripe_rows: int = 1024,
) -> np.ndarray:
    """Clip a large mask without allocating another full-resolution mask."""
    if polygon is None:
        return mask
    result = mask
    height, width = result.shape[:2]
    polygon = np.asarray(polygon, dtype=np.float64)
    for y0 in range(0, height, max(64, int(stripe_rows))):
        y1 = min(height, y0 + max(64, int(stripe_rows)))
        stripe_mask = np.zeros((y1 - y0, width), dtype=np.uint8)
        shifted = polygon.copy()
        shifted[:, 1] -= y0
        cv2.fillPoly(
            stripe_mask,
            [np.round(shifted).astype(np.int32)],
            255,
        )
        cv2.bitwise_and(
            result[y0:y1],
            stripe_mask,
            dst=result[y0:y1],
        )
    return result


def clip_band_polygon(
    polygon: np.ndarray,
    field_polygon: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Raster-clip one narrow band polygon to the selected field boundary."""
    polygon = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    if field_polygon is None or len(polygon) < 3:
        return polygon
    x, y, width, height = cv2.boundingRect(polygon)
    if width <= 0 or height <= 0:
        return None
    band_roi = np.zeros((height, width), dtype=np.uint8)
    field_roi = np.zeros_like(band_roi)
    cv2.fillPoly(band_roi, [polygon - (x, y)], 255)
    shifted_field = np.asarray(field_polygon, dtype=np.float64) - (x, y)
    cv2.fillPoly(
        field_roi,
        [np.round(shifted_field).astype(np.int32)],
        255,
    )
    clipped = cv2.bitwise_and(band_roi, field_roi)
    contours, _ = cv2.findContours(
        clipped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return contour + (x, y)


def expand_polygon_px(
    polygon: np.ndarray,
    px: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Expand a polygon outward by ``px`` pixels (bbox-local dilation).

    The dilation runs on a small mask covering only the polygon bbox plus
    the expansion margin, so huge rasters never allocate a full-resolution
    mask.  The returned contour is the outer boundary of the expanded
    region.
    """
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3 or px <= 0:
        return poly
    px = int(math.ceil(px))
    bmin = np.floor(poly.min(axis=0)).astype(np.int64)
    bmax = np.ceil(poly.max(axis=0)).astype(np.int64)
    x0 = max(0, int(bmin[0] - px))
    y0 = max(0, int(bmin[1] - px))
    x1 = min(int(width), int(bmax[0] + px) + 1)
    y1 = min(int(height), int(bmax[1] + px) + 1)
    if x1 <= x0 or y1 <= y0:
        return poly
    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(local, [np.round(poly - (x0, y0)).astype(np.int32)], 255)
    kernel = 2 * px + 1
    local = cv2.dilate(local, np.ones((kernel, kernel), np.uint8))
    contours, _ = cv2.findContours(
        local,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return poly
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return contour.astype(np.float64) + (x0, y0)


def smooth_1d(values: np.ndarray, window: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values.copy()
    size = odd_size(window)
    if size > values.size:
        size = values.size if values.size % 2 == 1 else values.size - 1
    if size < 3:
        return values.copy()
    kernel = cv2.getGaussianKernel(size, max(size / 5.0, 0.8)).reshape(-1)
    padded = np.pad(values, (size // 2, size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fill_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if not np.any(good):
        return values.copy()
    indices = np.arange(values.size)
    return np.interp(indices, indices[good], values[good])


def _component_main_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size < 3:
        return 0.0
    centered_x = xs.astype(np.float64) - float(np.mean(xs))
    centered_y = ys.astype(np.float64) - float(np.mean(ys))
    xx = float(np.sum(centered_x * centered_x, dtype=np.float64))
    xy = float(np.sum(centered_x * centered_y, dtype=np.float64))
    yy = float(np.sum(centered_y * centered_y, dtype=np.float64))
    return 0.5 * math.atan2(2.0 * xy, xx - yy) % math.pi


def _component_elongation(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size < 3:
        return 1.0
    centered_x = xs.astype(np.float64) - float(np.mean(xs))
    centered_y = ys.astype(np.float64) - float(np.mean(ys))
    xx = float(np.sum(centered_x * centered_x, dtype=np.float64))
    xy = float(np.sum(centered_x * centered_y, dtype=np.float64))
    yy = float(np.sum(centered_y * centered_y, dtype=np.float64))
    radius = math.hypot(xx - yy, 2.0 * xy)
    largest = max((xx + yy + radius) * 0.5, 1e-9)
    smallest = max((xx + yy - radius) * 0.5, 1e-9)
    return float(math.sqrt(largest / smallest))


def _component_extent_along_angle(mask: np.ndarray, angle: float) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0
    direction = np.asarray((math.cos(float(angle)), math.sin(float(angle))), dtype=np.float64)
    projection = xs.astype(np.float64) * direction[0] + ys.astype(np.float64) * direction[1]
    return float(projection.max() - projection.min() + 1.0)


def _component_projection_interval(mask: np.ndarray, angle: float) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0, 0.0
    direction = np.asarray((math.cos(float(angle)), math.sin(float(angle))), dtype=np.float64)
    projection = xs.astype(np.float64) * direction[0] + ys.astype(np.float64) * direction[1]
    return float(projection.min()), float(projection.max())


def residual_mask_layers(
    raw_mask: np.ndarray,
    rebuilt_mask: np.ndarray,
    main_angle: float,
    meters_per_px: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    angle_thresh_deg: float = 60.0,
    min_area_ratio: float = 0.001,
    short_parallel_ratio: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw residuals into headland, attached body, and uncertain layers.

    DECISION-009: projection overlap is not enough to promote a residual into the
    work body. A row-parallel component must also be metrically close to the
    rebuilt body. Ambiguous parallel fragments remain visible as uncertain
    evidence instead of being mislabeled as headland or work-line body.
    """
    cfg = dict(config or {})
    raw = (np.asarray(raw_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    rebuilt = (np.asarray(rebuilt_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    residual = cv2.bitwise_and(raw, cv2.bitwise_not(rebuilt))
    headland = np.zeros_like(raw)
    body_residual = np.zeros_like(raw)
    uncertain = np.zeros_like(raw)
    if cv2.countNonZero(residual) == 0:
        return headland, body_residual, uncertain

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(residual, 8)
    threshold = math.radians(float(angle_thresh_deg))
    total = max(1, int(cv2.countNonZero(raw)))
    min_area = max(3, int(round(total * float(min_area_ratio))))
    min_parallel_extent = max(8.0, float(short_parallel_ratio) * max(raw.shape[:2]))
    has_rebuilt_body = cv2.countNonZero(rebuilt) > 0
    rebuilt_start, rebuilt_end = _component_projection_interval(rebuilt, main_angle)
    body_extension = max(3.0, float(short_parallel_ratio) * max(raw.shape[:2]) * 0.15)
    attach_m = float(cfg.get("residual_body_attach_m", 0.12))
    if meters_per_px is not None and np.isfinite(float(meters_per_px)) and float(meters_per_px) > 0:
        attach_px = max(1.0, attach_m / float(meters_per_px))
    else:
        attach_px = max(2.0, max(raw.shape[:2]) * 0.01)
    body_attachment_zone = None
    if has_rebuilt_body:
        attach_radius = max(1, int(math.ceil(attach_px)))
        body_attachment_zone = cv2.dilate(
            rebuilt,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (attach_radius * 2 + 1, attach_radius * 2 + 1),
            ),
            iterations=1,
        )
    min_attach_ratio = float(cfg.get("residual_body_min_attach_ratio", 0.02))
    body_angle_threshold = math.radians(
        float(cfg.get("residual_body_max_angle_deg", 22.5))
    )
    min_body_elongation = float(cfg.get("residual_body_min_elongation", 2.2))
    direction = np.asarray(
        (math.cos(float(main_angle)), math.sin(float(main_angle))),
        dtype=np.float64,
    )

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        label_roi = labels[y:y + height, x:x + width]
        component_selector = label_roi == label
        component = component_selector.astype(np.uint8) * 255

        def assign_component(target: np.ndarray):
            target_roi = target[y:y + height, x:x + width]
            residual_roi = residual[y:y + height, x:x + width]
            target_roi[component_selector] = residual_roi[component_selector]

        if area < min_area:
            assign_component(uncertain)
            continue

        comp_angle = _component_main_angle(component)
        diff = abs(comp_angle - float(main_angle)) % math.pi
        diff = min(diff, math.pi - diff)
        elongation = _component_elongation(component)
        parallel_extent = _component_extent_along_angle(component, main_angle)
        comp_start, comp_end = _component_projection_interval(component, main_angle)
        projection_offset = float(x * direction[0] + y * direction[1])
        comp_start += projection_offset
        comp_end += projection_offset
        overlaps_body_extent = (
            not has_rebuilt_body
            or (
                comp_end >= rebuilt_start - body_extension
                and comp_start <= rebuilt_end + body_extension
            )
        )
        if diff > threshold or not overlaps_body_extent:
            assign_component(headland)
        elif diff > body_angle_threshold or elongation < min_body_elongation:
            assign_component(uncertain)
        elif parallel_extent < min_parallel_extent or body_attachment_zone is None:
            assign_component(uncertain)
        else:
            near_body = component_selector & (
                body_attachment_zone[y:y + height, x:x + width] > 0
            )
            attach_ratio = float(np.count_nonzero(near_body) / max(area, 1))
            target = body_residual if attach_ratio >= min_attach_ratio else uncertain
            assign_component(target)
    return headland, body_residual, uncertain

def repair_row_gaps(
    mask: np.ndarray,
    main_angle: float,
    meters_per_px: float,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Close short gaps along crop-row direction without broad cross-row dilation."""
    cfg = config or {}
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(src) == 0:
        return src
    gap_m = float(cfg.get("row_gap_close_m", 2.2))
    width_m = float(cfg.get("row_gap_close_width_m", 0.04))
    k_len = odd_size(gap_m / max(float(meters_per_px), 1e-9), minimum=3)
    k_w = odd_size(width_m / max(float(meters_per_px), 1e-9), minimum=1)
    # Bound the kernel so a bad GeoTIFF scale cannot bridge across the field.
    k_len = min(k_len, max(3, int(max(src.shape[:2]) * 0.20) | 1))
    k_w = min(k_w, max(1, int(max(src.shape[:2]) * 0.01) | 1))
    if k_len <= 3:
        return src
    # DECISION-009: rotate the evidence into the row-aligned frame before the
    # close.  Rotating a long, thin kernel inside its original ``k_len x k_w``
    # canvas clips almost the entire kernel at oblique angles (the real 108°
    # case collapsed to only a few pixels), so valid internal gaps were not
    # repaired and were later removed by the raw-support constraint.
    aligned, matrix = rotate_bound(
        src, math.degrees(float(main_angle)), cv2.INTER_NEAREST
    )
    k_len = min(k_len, max(3, int(aligned.shape[1] * 0.20) | 1))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (k_len, max(1, k_w))
    )
    repaired_aligned = cv2.morphologyEx(
        aligned, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    inverse = cv2.invertAffineTransform(matrix)
    repaired = cv2.warpAffine(
        repaired_aligned,
        inverse,
        (src.shape[1], src.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Nearest-neighbour round trips can drop isolated edge pixels.  Gap repair
    # may add supported bridges but must never delete original evidence.
    return cv2.bitwise_or(src, (repaired > 0).astype(np.uint8) * 255)


def constrain_generated_body_to_raw_support(
    generated_mask: np.ndarray,
    raw_mask: np.ndarray,
    main_angle: float,
    meters_per_px: float,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Remove one-sided geometric protrusions not supported by raw mask evidence.

    New pixels are allowed only where the raw mask's own row-direction closing
    would bridge a gap. This preserves genuine short gap repairs while clipping
    geometry/closing artefacts that extend past a raw row endpoint.
    """
    cfg = dict(config or {})
    generated = (np.asarray(generated_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    raw = (np.asarray(raw_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    raw_support = repair_row_gaps(raw, main_angle, meters_per_px, cfg)
    constrained = cv2.bitwise_and(generated, raw_support)
    return constrained


def resolve_terminal_headland_overlap(
    raw_mask: np.ndarray,
    body_mask: np.ndarray,
    headland_seed_mask: np.ndarray,
    main_angle: float,
    meters_per_px: float,
    config: Optional[Dict[str, Any]] = None,
    axial_reference_mask: Optional[np.ndarray] = None,
    field_polygon: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Give strong terminal headland corridors precedence over rebuilt rows.

    Headland is initially detected from ``raw & ~rebuilt`` and therefore has
    periodic holes wherever rebuilt longitudinal bands overlap a transverse
    terminal strip.  First partition the regularized reference body into
    connected crop regions; then accept only candidates supported by trusted
    long-span terminal regions.  A second component-span gate rejects detached
    islands inside the same axial depth window.  This avoids turning an
    internal ditch, a region seam, or a triangular terminal fragment into a
    field-wide yellow corridor.
    """
    cfg = dict(config or {})
    raw = (np.asarray(raw_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    body = (np.asarray(body_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    seed = (
        (np.asarray(headland_seed_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    )
    empty = np.zeros_like(body)
    diagnostics = {
        "enabled": bool(cfg.get("headland_corridor_enabled", True)),
        "candidate_count": 0,
        "corridor_count": 0,
        "corridor_px": 0,
        "removed_body_px": 0,
        "envelope_added_px": 0,
        "rejected": {},
        "candidates": [],
        # A terminal candidate is classified against the connected regions of
        # the regularized row body before it is allowed into the semantic
        # headland layer.  Keeping this report explicit makes it possible to
        # audit why an interior region seam was rejected instead of silently
        # painting it yellow.
        "region_partition": {
            "source": "axial_reference_connected_components",
            "region_count": 0,
            "regions": [],
            "candidate_regions": [],
            "rejected_fragment_count": 0,
        },
        "corridor_component_filter": {
            "min_cross_row_span_ratio": float(cfg.get(
                "headland_corridor_min_component_span_ratio",
                cfg.get("headland_corridor_min_span_ratio", 0.45),
            )),
            "accepted_count": 0,
            "accepted_envelope_fragment_count": 0,
            "rejected_short_count": 0,
            "rejected": [],
        },
        "terminal_dense_seed": {
            "enabled": bool(cfg.get(
                "headland_terminal_dense_seed_enabled", True
            )),
            "sides_added": [],
            "sides_skipped": {},
            "fallback_sides": [],
            "selected_runs": {},
            "fallback_min_occupancy": float(np.clip(
                cfg.get("headland_terminal_dense_seed_fallback_occupancy", 0.65),
                0.0,
                1.0,
            )),
            "merge_gap_m": float(max(
                0.0,
                cfg.get("headland_terminal_dense_seed_gap_m", 1.0),
            )),
            "added_px": 0,
        },
        "terminal_shape": {
            "configured_follow_raw_terminal": bool(cfg.get(
                "headland_corridor_follow_raw_terminal", True
            )),
            "follow_raw_terminal": bool(cfg.get(
                "headland_corridor_follow_raw_terminal", True
            )),
            "follow_raw_terminal_disabled_by_separator": False,
            "separator_sides": [],
            "effective_max_body_cut_m": None,
            "effective_endpoint_margin_m": None,
            "accepted_sides": [],
            "raw_support_added_px": 0,
            "wall_source": (
                "regularized_longitudinal_body_endpoints"
                if axial_reference_mask is not None
                else "accepted_terminal_seed_envelope"
            ),
            "wall_min_axial_span_ratio": float(cfg.get(
                "headland_terminal_wall_min_axial_span_ratio", 0.72
            )),
            "wall_reference_axial_span_px": 0.0,
            "wall_trusted_row_count": 0,
            "wall_rejected_short_row_count": 0,
            "wall_fit_method": "binned_theil_sen_outlier_replacement",
            "wall_outlier_tolerance_m": float(max(
                0.0,
                cfg.get("headland_terminal_wall_outlier_tolerance_m", 0.75),
            )),
            "wall_outlier_replaced_row_count": 0,
            "wall_outlier_replaced_by_side": {},
            "field_slope_hints": {},
            "field_slope_applied_sides": [],
            "restored_band_end_trim_m": 0.0,
        },
        "axial_reference_source": (
            "regularized_longitudinal_body"
            if axial_reference_mask is not None
            else "merged_body_compatibility"
        ),
    }
    if (
        not diagnostics["enabled"]
        or body.shape != raw.shape
        or seed.shape != raw.shape
        or cv2.countNonZero(body) == 0
        or not np.isfinite(float(meters_per_px))
        or float(meters_per_px) <= 0
    ):
        return body, seed, empty, diagnostics

    seed_rotated, matrix = rotate_bound(
        seed, math.degrees(float(main_angle)), cv2.INTER_NEAREST
    )
    out_h, out_w = seed_rotated.shape[:2]
    body_rotated = cv2.warpAffine(
        body, matrix, (out_w, out_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    has_axial_reference = bool(
        axial_reference_mask is not None
        and np.asarray(axial_reference_mask).shape == body.shape
    )
    axial_reference = (
        (np.asarray(axial_reference_mask, dtype=np.uint8) > 0).astype(np.uint8)
        * 255
        if has_axial_reference
        else body
    )
    reference_rotated = cv2.warpAffine(
        axial_reference,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    raw_rotated = cv2.warpAffine(
        raw, matrix, (out_w, out_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    body_ys, body_xs = np.where(reference_rotated > 0)
    if body_xs.size == 0:
        return body, seed, empty, diagnostics

    body_x_min, body_x_max = int(body_xs.min()), int(body_xs.max())
    body_y_min, body_y_max = int(body_ys.min()), int(body_ys.max())
    body_cross_span = max(1, body_y_max - body_y_min + 1)
    terminal_depth_px = max(1, int(round(
        float(cfg.get("headland_corridor_terminal_depth_m", 3.0))
        / float(meters_per_px)
    )))

    # The regularized band polygons share a conservative axial sampling
    # interval.  In a slightly skewed field this can quantize every endpoint
    # to the same x coordinate and erase the real terminal-wall slope.  The
    # selected field polygon still preserves that physical terminal-edge
    # orientation, so retain long, cross-row polygon edges as slope hints.
    # The hint is only consumed below when the reference endpoints are
    # demonstrably quantized; genuine endpoint trends remain authoritative.
    field_slope_hints: Dict[str, Dict[str, Any]] = {}
    polygon = None
    if field_polygon is not None:
        candidate_polygon = np.asarray(field_polygon, dtype=np.float64).reshape(-1, 2)
        if (
            candidate_polygon.shape[0] >= 3
            and np.all(np.isfinite(candidate_polygon))
        ):
            polygon = cv2.transform(
                candidate_polygon.reshape(-1, 1, 2), matrix
            ).reshape(-1, 2)
    if polygon is not None:
        min_field_span_ratio = float(np.clip(
            cfg.get("headland_terminal_field_edge_min_span_ratio", 0.45),
            0.10,
            1.0,
        ))
        max_field_slope = float(max(
            0.0,
            cfg.get("headland_terminal_field_edge_max_abs_slope", 0.25),
        ))
        edge_candidates = []
        for edge_index in range(polygon.shape[0]):
            point_a = polygon[edge_index]
            point_b = polygon[(edge_index + 1) % polygon.shape[0]]
            delta_y = float(point_b[1] - point_a[1])
            if abs(delta_y) < 1e-9:
                continue
            slope = float((point_b[0] - point_a[0]) / delta_y)
            span_ratio = float(min(
                1.0,
                abs(delta_y) / max(float(body_cross_span), 1.0),
            ))
            if span_ratio < min_field_span_ratio or abs(slope) > max_field_slope:
                continue
            edge_candidates.append({
                "edge_index": int(edge_index),
                "slope_px_per_cross_row_px": slope,
                "cross_row_span_ratio": span_ratio,
                "midpoint_x_px": float((point_a[0] + point_b[0]) * 0.5),
            })
        if edge_candidates:
            start_candidate = min(
                edge_candidates, key=lambda item: item["midpoint_x_px"]
            )
            end_candidate = max(
                edge_candidates, key=lambda item: item["midpoint_x_px"]
            )
            if start_candidate["midpoint_x_px"] < body_x_min:
                field_slope_hints["start"] = start_candidate
            if end_candidate["midpoint_x_px"] > body_x_max:
                field_slope_hints["end"] = end_candidate
    diagnostics["terminal_shape"]["field_slope_hints"] = {
        side: dict(value) for side, value in field_slope_hints.items()
    }

    # A wide, non-mask strip immediately beside the regularized body is a
    # stronger terminal wall than the raw crop fragments that continue along
    # individual rows.  The latter are what produced the finger-like yellow
    # intrusions in real fields: ``follow_raw_terminal`` copied each fragment
    # past the common wall and treated it as headland.  Detect the wall from
    # cross-row occupancy, but only in a narrow endpoint neighbourhood so an
    # interior row gap cannot become a field-wide headland.
    separator_report = diagnostics["terminal_separator"] = {
        "enabled": bool(cfg.get("headland_terminal_separator_enabled", True)),
        "sides": [],
        "selected_runs": {},
        "typical_body_occupancy": 0.0,
        "max_occupancy": 0.0,
        "min_gap_px": 0,
        "endpoint_tolerance_px": 0,
    }
    separator_sides = set()
    separator_enabled = bool(separator_report["enabled"])
    if separator_enabled:
        axial_occupancy = np.mean(
            raw_rotated[body_y_min:body_y_max + 1] > 0,
            axis=0,
        ).astype(np.float64)
        separator_smooth_px = max(1.0, float(cfg.get(
            "headland_terminal_separator_smooth_m", 0.08
        )) / float(meters_per_px))
        axial_occupancy = smooth_1d(axial_occupancy, separator_smooth_px)
        body_occupancy = axial_occupancy[body_x_min:body_x_max + 1]
        body_occupancy = body_occupancy[np.isfinite(body_occupancy)]
        typical_body_occupancy = float(
            np.percentile(body_occupancy, 50.0)
            if body_occupancy.size else 0.0
        )
        configured_max_occupancy = cfg.get(
            "headland_terminal_separator_max_occupancy"
        )
        if configured_max_occupancy is None:
            max_occupancy = float(np.clip(
                typical_body_occupancy * float(cfg.get(
                    "headland_terminal_separator_occupancy_ratio", 0.82
                )),
                0.12,
                0.90,
            ))
        else:
            max_occupancy = float(np.clip(configured_max_occupancy, 0.0, 1.0))
        min_gap_px = max(3, int(round(float(cfg.get(
            "headland_terminal_separator_min_gap_m", 0.15
        )) / float(meters_per_px))))
        endpoint_tolerance_px = max(min_gap_px, int(round(float(cfg.get(
            "headland_terminal_separator_endpoint_tolerance_m", 0.60
        )) / float(meters_per_px))))
        separator_report.update({
            "typical_body_occupancy": typical_body_occupancy,
            "max_occupancy": max_occupancy,
            "min_gap_px": min_gap_px,
            "endpoint_tolerance_px": endpoint_tolerance_px,
        })

        def terminal_gap(side: str):
            search_depth = terminal_depth_px + endpoint_tolerance_px
            if side == "start":
                search_start = max(0, body_x_min - search_depth)
                search_end = min(out_w, body_x_min + endpoint_tolerance_px + 1)
            else:
                search_start = max(0, body_x_max + 1 - endpoint_tolerance_px)
                search_end = min(out_w, body_x_max + 1 + search_depth)
            if search_end <= search_start:
                return None
            local_runs = [
                (search_start + start, search_start + end)
                for start, end in _runs(
                    axial_occupancy[search_start:search_end] <= max_occupancy
                )
                if end - start >= min_gap_px
            ]

            def has_outer_terminal_support(item):
                support_width = max(
                    min_gap_px,
                    int(round(float(cfg.get(
                        "headland_terminal_separator_support_m", 0.15
                    )) / float(meters_per_px))),
                )
                if side == "start":
                    support_start = max(search_start, item[0] - support_width)
                    support_end = item[0]
                else:
                    support_start = item[1]
                    support_end = min(search_end, item[1] + support_width)
                if support_end <= support_start:
                    return False
                support = axial_occupancy[support_start:support_end]
                return bool(
                    support.size >= min_gap_px
                    and np.mean(support >= max_occupancy) >= 0.45
                )

            if side == "start":
                candidates = [
                    item for item in local_runs
                    if item[1] >= body_x_min - endpoint_tolerance_px
                    and has_outer_terminal_support(item)
                ]
                return max(candidates, key=lambda item: (item[1], item[1] - item[0])) if candidates else None
            candidates = [
                item for item in local_runs
                if item[0] <= body_x_max + 1 + endpoint_tolerance_px
                and has_outer_terminal_support(item)
            ]
            return min(candidates, key=lambda item: (item[0], -(item[1] - item[0]))) if candidates else None

        for side in ("start", "end"):
            selected = terminal_gap(side)
            if selected is None:
                continue
            separator_sides.add(side)
            separator_report["sides"].append(side)
            separator_report["selected_runs"][side] = [
                int(selected[0]), int(selected[1])
            ]

    separator_limited = bool(separator_sides)
    separator_report["effective_wall_policy"] = (
        "non_mask_terminal_gap" if separator_limited else "configured_terminal_support"
    )
    configured_body_cut_px = max(0, int(round(
        float(cfg.get("headland_corridor_max_body_cut_m", 3.0))
        / float(meters_per_px)
    )))
    configured_endpoint_margin_px = max(0, int(round(
        float(cfg.get("headland_corridor_endpoint_margin_m", 0.30))
        / float(meters_per_px)
    )))
    max_body_cut_px = 0 if separator_limited else configured_body_cut_px
    # A trusted reference wall is already the semantic boundary.  Applying an
    # additional inward margin moved the corridor about 0.3 m into visibly
    # regular work rows and made a corrected wall look wrong in the UI.  Keep
    # the legacy margin only for the compatibility path that lacks an explicit
    # longitudinal reference body.
    endpoint_margin_px = (
        0
        if separator_limited or has_axial_reference
        else configured_endpoint_margin_px
    )
    follow_raw_terminal = bool(cfg.get(
        "headland_corridor_follow_raw_terminal", True
    )) and not separator_limited
    terminal_shape = diagnostics["terminal_shape"]
    terminal_shape.update({
        "follow_raw_terminal": bool(follow_raw_terminal),
        "follow_raw_terminal_disabled_by_separator": bool(
            separator_limited and cfg.get("headland_corridor_follow_raw_terminal", True)
        ),
        "separator_sides": sorted(separator_sides),
        "configured_endpoint_margin_m": float(
            configured_endpoint_margin_px * meters_per_px
        ),
        "effective_max_body_cut_m": float(max_body_cut_px * meters_per_px),
        "effective_endpoint_margin_m": float(endpoint_margin_px * meters_per_px),
    })

    # Partition the reference body into connected crop regions first.  The
    # old implementation classified every residual pixel against one global
    # endpoint curve; a short fragment at a planting-region seam could then
    # become a yellow headland.  Here each region gets its own axial span and
    # terminal class.  Only rows belonging to a long region that actually
    # reaches the global start/end envelope can support a terminal corridor.
    region_report = diagnostics["region_partition"]
    region_count, region_labels, region_stats, _ = cv2.connectedComponentsWithStats(
        (reference_rotated > 0).astype(np.uint8), 8
    )
    region_report["region_count"] = max(0, int(region_count - 1))
    reference_spans = []
    region_records = []
    for region_id in range(1, region_count):
        region_area = int(region_stats[region_id, cv2.CC_STAT_AREA])
        if region_area <= 0:
            continue
        rx = int(region_stats[region_id, cv2.CC_STAT_LEFT])
        ry = int(region_stats[region_id, cv2.CC_STAT_TOP])
        rw = int(region_stats[region_id, cv2.CC_STAT_WIDTH])
        rh = int(region_stats[region_id, cv2.CC_STAT_HEIGHT])
        region_mask = region_labels[ry:ry + rh, rx:rx + rw] == region_id
        local_ys, local_xs = np.where(region_mask)
        if local_xs.size == 0:
            continue
        start = rx + int(local_xs.min())
        end = rx + int(local_xs.max())
        span = max(1, end - start + 1)
        reference_spans.append(float(span))
        region_records.append({
            "id": int(region_id),
            "bbox": [rx, ry, rw, rh],
            "area_px": region_area,
            "axial_start_px": int(start),
            "axial_end_px": int(end),
            "axial_span_px": int(span),
            "row_span_px": int(rh),
        })
    reference_span = float(np.percentile(reference_spans, 90.0)) if reference_spans else 0.0
    region_min_span_ratio = float(np.clip(
        cfg.get("headland_terminal_wall_min_axial_span_ratio", 0.72),
        0.25,
        1.0,
    ))
    min_region_support_ratio = float(np.clip(
        cfg.get("headland_terminal_region_min_support_ratio", 0.55),
        0.0,
        1.0,
    ))
    terminal_region_rows = {"start": set(), "end": set()}
    terminal_region_ids = {"start": [], "end": []}
    for record in region_records:
        trusted = (
            reference_span <= 0.0
            or float(record["axial_span_px"]) >= reference_span * region_min_span_ratio
        )
        near_start_region = record["axial_start_px"] <= body_x_min + terminal_depth_px
        near_end_region = record["axial_end_px"] >= body_x_max - terminal_depth_px
        classes = []
        if trusted and near_start_region:
            classes.append("start")
            terminal_region_ids["start"].append(int(record["id"]))
            terminal_region_rows["start"].update(
                range(record["bbox"][1], record["bbox"][1] + record["bbox"][3])
            )
        if trusted and near_end_region:
            classes.append("end")
            terminal_region_ids["end"].append(int(record["id"]))
            terminal_region_rows["end"].update(
                range(record["bbox"][1], record["bbox"][1] + record["bbox"][3])
            )
        record.update({
            "axial_span_ratio": float(
                min(1.0, record["axial_span_px"] / max(reference_span, 1.0))
            ),
            "trusted_long_span": bool(trusted),
            "terminal_class": classes or ["interior"],
        })
    region_report["regions"] = region_records
    region_report["terminal_region_ids"] = terminal_region_ids
    region_report["min_region_support_ratio"] = min_region_support_ratio

    # DECISION-009: a transverse terminal crop strip can be connected to the
    # longitudinal crop body, causing component PCA to hide it from the
    # residual-orientation seed.  At a body endpoint only, a dense cross-row
    # run outside the rebuilt axial envelope is independent supporting
    # evidence.  Add it as a seed; the normal span/terminal/depth gates below
    # still decide whether it is a real headland corridor.
    dense_report = diagnostics["terminal_dense_seed"]
    if dense_report["enabled"]:
        seed_depth_px = max(1, int(round(
            float(cfg.get("headland_terminal_dense_seed_depth_m", 4.0))
            / float(meters_per_px)
        )))
        min_occupancy = float(np.clip(
            cfg.get("headland_terminal_dense_seed_min_occupancy", 0.88),
            0.0,
            1.0,
        ))
        min_depth_px = max(1, int(round(
            float(cfg.get("headland_terminal_dense_seed_min_depth_m", 0.15))
            / float(meters_per_px)
        )))
        fallback_occupancy = float(np.clip(
            cfg.get("headland_terminal_dense_seed_fallback_occupancy", 0.65),
            0.0,
            min_occupancy,
        ))
        dense_gap_px = max(0, int(round(
            float(cfg.get("headland_terminal_dense_seed_gap_m", 1.0))
            / float(meters_per_px)
        )))

        def dense_runs(occupancy, threshold):
            return [
                item for item in _merge_close_runs(
                    _runs(occupancy >= threshold), dense_gap_px
                )
                if item[1] - item[0] >= min_depth_px
            ]

        def selected_run(runs, side):
            if not runs:
                return None
            return (
                max(runs, key=lambda item: item[1])
                if side == "start"
                else min(runs, key=lambda item: item[0])
            )

        for side, col_start, col_end in (
            ("start", max(0, body_x_min - seed_depth_px), body_x_min),
            ("end", body_x_max + 1, min(out_w, body_x_max + 1 + seed_depth_px)),
        ):
            if col_end <= col_start:
                dense_report["sides_skipped"][side] = "empty_search_window"
                continue
            occupancy = np.mean(
                raw_rotated[body_y_min:body_y_max + 1, col_start:col_end] > 0,
                axis=0,
            )
            strict_runs = dense_runs(occupancy, min_occupancy)
            fallback_runs = dense_runs(occupancy, fallback_occupancy)
            strict_selected = selected_run(strict_runs, side)
            fallback_selected = selected_run(fallback_runs, side)
            selected = strict_selected
            if fallback_selected is not None:
                if strict_selected is None:
                    selected = fallback_selected
                    dense_report["fallback_sides"].append(side)
                elif (
                    fallback_selected[0] <= strict_selected[1] + dense_gap_px
                    and fallback_selected[1] >= strict_selected[0] - dense_gap_px
                ):
                    selected = (
                        min(strict_selected[0], fallback_selected[0]),
                        max(strict_selected[1], fallback_selected[1]),
                    )
                    if selected != strict_selected:
                        dense_report["fallback_sides"].append(side)
            if selected is None:
                dense_report["sides_skipped"][side] = "no_dense_terminal_run"
                continue
            run_start = col_start + int(selected[0])
            run_end = col_start + int(selected[1])
            dense_report["selected_runs"][side] = [run_start, run_end]
            before = int(cv2.countNonZero(seed_rotated))
            seed_roi = seed_rotated[
                body_y_min:body_y_max + 1, run_start:run_end
            ]
            raw_roi = raw_rotated[
                body_y_min:body_y_max + 1, run_start:run_end
            ]
            seed_roi[raw_roi > 0] = 255
            added = max(0, int(cv2.countNonZero(seed_rotated)) - before)
            if added:
                dense_report["sides_added"].append(side)
                dense_report["added_px"] += added
            else:
                dense_report["sides_skipped"][side] = (
                    "orientation_seed_present"
                    if cv2.countNonZero(seed_rotated[:, col_start:col_end]) > 0
                    else "empty_dense_support"
                )

    close_m = float(cfg.get("headland_corridor_close_m", 1.2))
    close_px = odd_size(close_m / float(meters_per_px), minimum=3)
    depth_smooth_m = float(cfg.get("headland_corridor_depth_smooth_m", 0.10))
    depth_smooth_px = odd_size(
        depth_smooth_m / float(meters_per_px), minimum=1
    )
    close_px = min(close_px, max(3, int(out_h * 0.35) | 1))
    depth_smooth_px = min(depth_smooth_px, max(1, int(out_w * 0.02) | 1))
    reconstructed = cv2.morphologyEx(
        seed_rotated,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (depth_smooth_px, close_px)
        ),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (reconstructed > 0).astype(np.uint8), 8
    )
    diagnostics["candidate_count"] = max(0, int(count - 1))
    min_span_ratio = float(cfg.get("headland_corridor_min_span_ratio", 0.45))
    # ``max_body_cut_px``, ``endpoint_margin_px`` and ``follow_raw_terminal``
    # are resolved above.  A detected separator deliberately overrides the
    # permissive legacy defaults; otherwise the configured fallback remains
    # unchanged for fields without a usable non-mask wall.
    min_seed_ratio = float(cfg.get("headland_corridor_min_seed_ratio", 0.12))
    corridor_rotated = np.zeros_like(seed_rotated)
    # A real terminal strip is often broken into several raw components by
    # furrows, shadows, or a region seam. Keep the accepted, row-wise terminal
    # region separately so its boundary is continuous instead of inheriting
    # every raw hole as a false main-work extension.
    accepted_region_rotated = np.zeros_like(seed_rotated)
    accepted_seed_rotated = np.zeros_like(seed_rotated)
    follow_depth_px = max(1, int(round(
        float(cfg.get("headland_terminal_follow_max_depth_m", 4.5))
        / float(meters_per_px)
    )))
    wall_smooth_px = odd_size(
        float(cfg.get("headland_terminal_wall_smooth_m", 0.8))
        / float(meters_per_px),
        minimum=1,
    )
    region_fill = bool(cfg.get(
        "headland_corridor_fill_region",
        cfg.get("headland_corridor_fill_envelope", True),
    ))

    def reference_endpoint_curve(side: str) -> np.ndarray:
        # DECISION-009: an interior crop fragment or the boundary between two
        # planting regions is not a physical field end. Build the terminal
        # wall only from transverse rows whose longitudinal support spans most
        # of the field, then interpolate across crop-row gaps.
        values = np.full(out_h, np.nan, dtype=np.float64)
        starts = np.full(out_h, np.nan, dtype=np.float64)
        ends = np.full(out_h, np.nan, dtype=np.float64)
        spans = np.zeros(out_h, dtype=np.float64)
        for row_index in range(out_h):
            columns = np.flatnonzero(reference_rotated[row_index] > 0)
            if columns.size:
                starts[row_index] = float(columns.min())
                ends[row_index] = float(columns.max())
                spans[row_index] = float(columns.max() - columns.min() + 1)
        supported = spans > 0.0
        if not np.any(supported):
            return values
        reference_span = float(np.percentile(spans[supported], 90.0))
        min_span_ratio = float(np.clip(
            cfg.get("headland_terminal_wall_min_axial_span_ratio", 0.72),
            0.25,
            1.0,
        ))
        trusted = supported & (spans >= reference_span * min_span_ratio)
        if np.count_nonzero(trusted) < 2:
            return values
        values[trusted] = starts[trusted] if side == "start" else ends[trusted]
        shape_report = diagnostics["terminal_shape"]
        shape_report["wall_reference_axial_span_px"] = reference_span
        shape_report["wall_trusted_row_count"] = int(np.count_nonzero(trusted))
        shape_report["wall_rejected_short_row_count"] = int(np.count_nonzero(
            supported & ~trusted
        ))
        valid = np.flatnonzero(np.isfinite(values))
        if valid.size < 2:
            return values

        # A span threshold alone is insufficient: a broad planting-region
        # fragment can still span >72% of the field while ending several
        # metres inside the common terminal wall.  Interpolating those values
        # produced the recurring triangular yellow wedges.  Estimate a robust
        # global wall trend from cross-row bins and replace only endpoint rows
        # that deviate materially from that trend.  Genuine gradual/sloped
        # terminal walls remain because the Theil-Sen trend follows them.
        valid_y = valid.astype(np.float64)
        valid_values = values[valid]
        bin_count = min(24, max(4, int(math.ceil(valid.size / 50.0))))
        binned_y = []
        binned_values = []
        for indices in np.array_split(np.argsort(valid_y), bin_count):
            if indices.size == 0:
                continue
            binned_y.append(float(np.median(valid_y[indices])))
            binned_values.append(float(np.median(valid_values[indices])))
        slopes = []
        for left in range(len(binned_y)):
            for right in range(left + 1, len(binned_y)):
                delta_y = binned_y[right] - binned_y[left]
                if abs(delta_y) > 1e-9:
                    slopes.append(
                        (binned_values[right] - binned_values[left]) / delta_y
                    )
        robust_slope = float(np.median(slopes)) if slopes else 0.0
        robust_intercept = float(np.median(
            valid_values - robust_slope * valid_y
        ))
        robust_prediction = robust_slope * valid_y + robust_intercept
        residual = valid_values - robust_prediction
        residual_median = float(np.median(residual))
        residual_mad = float(np.median(np.abs(residual - residual_median)))
        quantized_mad_px = float(max(
            0.0,
            cfg.get("headland_terminal_wall_quantized_mad_px", 0.5),
        ))
        field_hint = field_slope_hints.get(side)
        field_slope_applied = bool(
            field_hint is not None and residual_mad <= quantized_mad_px
        )
        if field_slope_applied:
            robust_slope = float(field_hint["slope_px_per_cross_row_px"])
            robust_intercept = float(np.median(
                valid_values - robust_slope * valid_y
            ))
            robust_prediction = robust_slope * valid_y + robust_intercept
            # The flat endpoint samples are an artefact of the common axial
            # reconstruction interval, not independent wall observations.
            # Once quantization is proven, restore the physical field-edge
            # trend for every trusted row instead of waiting for an outlier
            # test that a flat sequence can never trigger.
            values[valid] = robust_prediction
            shape_report["field_slope_applied_sides"].append(side)
            residual = valid_values - robust_prediction
            residual_median = float(np.median(residual))
            residual_mad = float(np.median(
                np.abs(residual - residual_median)
            ))
        configured_tolerance_px = float(max(
            1.0,
            shape_report["wall_outlier_tolerance_m"] / float(meters_per_px),
        ))
        adaptive_tolerance_px = float(max(
            configured_tolerance_px,
            4.0 * 1.4826 * residual_mad,
        ))
        outlier = (
            np.zeros_like(residual, dtype=bool)
            if field_slope_applied
            else np.abs(residual - residual_median) > adaptive_tolerance_px
        )
        replaced_count = int(np.count_nonzero(outlier))
        if replaced_count:
            values[valid[outlier]] = robust_prediction[outlier]
        shape_report["wall_outlier_replaced_row_count"] += replaced_count
        shape_report["wall_outlier_replaced_by_side"][side] = replaced_count
        shape_report.setdefault("wall_robust_fit", {})[side] = {
            "slope_px_per_cross_row_px": robust_slope,
            "intercept_px": robust_intercept,
            "residual_mad_px": residual_mad,
            "effective_tolerance_px": adaptive_tolerance_px,
            "field_slope_applied": field_slope_applied,
        }
        filled = np.interp(
            np.arange(out_h, dtype=np.float64),
            valid.astype(np.float64),
            values[valid],
        )
        if wall_smooth_px > 1:
            pad = wall_smooth_px // 2
            kernel = np.ones(wall_smooth_px, dtype=np.float64) / wall_smooth_px
            filled = np.convolve(
                np.pad(filled, (pad, pad), mode="edge"),
                kernel,
                mode="valid",
            )
        restored_trim_px = float(max(
            0.0,
            cfg.get("band_end_trim_m", 0.10),
        )) / float(meters_per_px)
        if restored_trim_px > 0.0:
            filled += -restored_trim_px if side == "start" else restored_trim_px
            shape_report["restored_band_end_trim_m"] = float(
                restored_trim_px * meters_per_px
            )
        return filled

    endpoint_curves = {
        "start": reference_endpoint_curve("start"),
        "end": reference_endpoint_curve("end"),
    }

    def component_edge_curve(
        component_mask: np.ndarray,
        global_y: int,
        global_x: int,
        edge: str,
    ) -> np.ndarray:
        result = np.full(out_h, np.nan, dtype=np.float64)
        for local_y in range(component_mask.shape[0]):
            columns = np.flatnonzero(component_mask[local_y])
            if columns.size:
                result[global_y + local_y] = float(
                    global_x + (
                        columns.min() if edge == "low" else columns.max()
                    )
                )
        valid = np.flatnonzero(np.isfinite(result))
        if valid.size >= 2:
            result[valid[0]:valid[-1] + 1] = np.interp(
                np.arange(valid[0], valid[-1] + 1, dtype=np.float64),
                valid.astype(np.float64),
                result[valid],
            )
        return result

    def reject(reason: str):
        rejected = diagnostics["rejected"]
        rejected[reason] = int(rejected.get(reason, 0)) + 1

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[y:y + height, x:x + width] == label
        span_ratio = float(min(1.0, height / body_cross_span))
        near_start = x <= body_x_min + terminal_depth_px
        near_end = x + width - 1 >= body_x_max - terminal_depth_px
        near_terminal = near_start or near_end
        seed_pixels = int(np.count_nonzero(
            component & (seed_rotated[y:y + height, x:x + width] > 0)
        ))
        seed_ratio = float(seed_pixels / max(area, 1))
        candidate_rows = np.flatnonzero(np.any(component, axis=1)) + y
        reference_rows = [
            int(row_index) for row_index in candidate_rows
            if 0 <= int(row_index) < reference_rotated.shape[0]
            and np.any(reference_rotated[int(row_index)] > 0)
        ]
        candidate_region_info = {
            "bbox": [x, y, width, height],
            "region_ids": {"start": [], "end": []},
            "support_ratio": {"start": 0.0, "end": 0.0},
        }
        for side in ("start", "end"):
            side_rows = terminal_region_rows[side]
            support_ratio = (
                float(np.mean([row in side_rows for row in reference_rows]))
                if reference_rows else 0.0
            )
            candidate_region_info["support_ratio"][side] = support_ratio
            candidate_region_info["region_ids"][side] = list(
                terminal_region_ids[side]
            )
        region_report["candidate_regions"].append(candidate_region_info)
        reason = "accepted"
        if not near_terminal:
            reason = "not_terminal"
        elif span_ratio < min_span_ratio:
            reason = "insufficient_cross_row_span"
        elif seed_ratio < min_seed_ratio:
            reason = "insufficient_seed_support"
        else:
            supported_sides = [
                side for side in ("start", "end")
                if (
                    (near_start if side == "start" else near_end)
                    and candidate_region_info["support_ratio"][side]
                    >= min_region_support_ratio
                )
            ]
            if not supported_sides:
                reason = "insufficient_terminal_region_support"
            else:
                candidate_region_info["accepted_sides"] = supported_sides
        diagnostics["candidates"].append({
            "bbox": [x, y, width, height],
            "depth_m": float(width * meters_per_px),
            "cross_row_span_ratio": span_ratio,
            "seed_ratio": seed_ratio,
            "near_start": bool(near_start),
            "near_end": bool(near_end),
            "terminal_region_support": candidate_region_info["support_ratio"],
            "terminal_region_ids": candidate_region_info["region_ids"],
            "decision": reason,
        })
        if reason != "accepted":
            reject(reason)
            if reason in {
                "insufficient_terminal_region_support",
                "insufficient_cross_row_span",
            }:
                region_report["rejected_fragment_count"] += 1
            continue
        allowed_columns = np.zeros(width, dtype=bool)
        global_columns = np.arange(x, x + width)
        if near_start:
            allowed_columns |= global_columns <= body_x_min + max_body_cut_px
        if near_end:
            allowed_columns |= global_columns >= body_x_max - max_body_cut_px
        component = component & allowed_columns.reshape(1, -1)
        if not np.any(component):
            reject("outside_body_cut_depth")
            continue
        # Preserve only the seed pixels belonging to a region-qualified
        # terminal component.  Rejected residuals remain non-semantic evidence
        # instead of being painted as headland merely because they were in the
        # original orientation seed.
        accepted_seed_rotated[y:y + height, x:x + width][component] = (
            seed_rotated[y:y + height, x:x + width][component]
        )
        envelope_fill = bool(cfg.get("headland_corridor_fill_envelope", True))
        if follow_raw_terminal:
            low_edge = component_edge_curve(component, y, x, "low")
            high_edge = component_edge_curve(component, y, x, "high")
            component_rows = np.flatnonzero(np.any(component, axis=1)) + y
            before = int(cv2.countNonZero(corridor_rotated))
            accepted_sides = []
            for side, accepted in (("start", near_start), ("end", near_end)):
                if not accepted:
                    continue
                accepted_sides.append(side)
                endpoint_curve = endpoint_curves[side]
                for row_index in component_rows:
                    endpoint = endpoint_curve[row_index]
                    if not np.isfinite(endpoint):
                        endpoint = (
                            high_edge[row_index]
                            if side == "start"
                            else low_edge[row_index]
                        )
                    if side == "start":
                        # The qualified reference wall is the body/headland
                        # boundary.  Letting the candidate's inward edge win
                        # here defeats the wall fit and recreates exactly the
                        # triangular intrusion the fit rejected.  Compatibility
                        # callers without an axial reference still need the
                        # candidate edge to absorb periodic terminal islands.
                        inner = (
                            endpoint
                            if has_axial_reference
                            else max(endpoint, high_edge[row_index])
                        )
                        outer = min(low_edge[row_index], endpoint - follow_depth_px)
                        column_start = max(0, int(math.floor(outer)))
                        column_end = min(
                            out_w,
                            int(math.ceil(inner + endpoint_margin_px)) + 1,
                        )
                    else:
                        inner = (
                            endpoint
                            if has_axial_reference
                            else min(endpoint, low_edge[row_index])
                        )
                        outer = max(high_edge[row_index], endpoint + follow_depth_px)
                        column_start = max(
                            0,
                            int(math.floor(inner - endpoint_margin_px)),
                        )
                        column_end = min(out_w, int(math.ceil(outer)) + 1)
                    if column_end <= column_start:
                        continue
                    raw_row = raw_rotated[
                        row_index, column_start:column_end
                    ] > 0
                    corridor_rotated[
                        row_index, column_start:column_end
                    ][raw_row] = 255
                    if region_fill:
                        accepted_region_rotated[
                            row_index, column_start:column_end
                        ] = 255
            after = int(cv2.countNonZero(corridor_rotated))
            added = max(0, after - before)
            diagnostics["envelope_added_px"] += int(
                max(0, added - int(np.count_nonzero(component)))
            )
            shape_report = diagnostics["terminal_shape"]
            shape_report["raw_support_added_px"] += added
            for side in accepted_sides:
                if side not in shape_report["accepted_sides"]:
                    shape_report["accepted_sides"].append(side)
        elif envelope_fill:
            local_ys, local_xs = np.where(component)
            global_y0 = y + int(local_ys.min())
            global_y1 = y + int(local_ys.max()) + 1
            component_x0 = x + int(local_xs.min())
            component_x1 = x + int(local_xs.max()) + 1
            rectangles: List[Tuple[int, int, int, int]] = []
            if near_start:
                rectangles.append((
                    max(0, min(component_x0, body_x_min)),
                    min(
                        out_w,
                        max(component_x1 + endpoint_margin_px, body_x_min + 1),
                        body_x_min + max_body_cut_px + 1,
                    ),
                    global_y0,
                    global_y1,
                ))
            if near_end:
                rectangles.append((
                    max(
                        0,
                        min(component_x0 - endpoint_margin_px, body_x_max),
                        body_x_max - max_body_cut_px,
                    ),
                    min(out_w, max(component_x1, body_x_max + 1)),
                    global_y0,
                    global_y1,
                ))
            before = int(cv2.countNonZero(corridor_rotated))
            for rect_x0, rect_x1, rect_y0, rect_y1 in rectangles:
                if rect_x1 > rect_x0 and rect_y1 > rect_y0:
                    corridor_rotated[rect_y0:rect_y1, rect_x0:rect_x1] = 255
                    if region_fill:
                        accepted_region_rotated[
                            rect_y0:rect_y1, rect_x0:rect_x1
                        ] = 255
            after = int(cv2.countNonZero(corridor_rotated))
            diagnostics["envelope_added_px"] += int(
                max(0, after - before - int(np.count_nonzero(component)))
            )
        else:
            target = corridor_rotated[y:y + height, x:x + width]
            target[component] = 255
            if region_fill:
                target = accepted_region_rotated[y:y + height, x:x + width]
                target[component] = 255
        diagnostics["corridor_count"] += 1

    accepted_terminal_sides = set()
    for candidate in diagnostics["candidates"]:
        if candidate.get("decision") != "accepted":
            continue
        if candidate.get("near_start"):
            accepted_terminal_sides.add("start")
        if candidate.get("near_end"):
            accepted_terminal_sides.add("end")
    diagnostics["terminal_shape"]["accepted_sides"] = sorted(
        accepted_terminal_sides
    )

    # A morphology-closed candidate can contain a detached triangular/raw
    # island at a region seam.  Classify the resulting corridor components a
    # second time by their transverse span before warping back.  This is the
    # region-boundary guard: a terminal corridor must cover a meaningful
    # fraction of the row lattice, not merely fall inside the same axial depth
    # window as a trusted terminal seed.
    component_report = diagnostics["corridor_component_filter"]
    component_min_span_ratio = float(np.clip(
        component_report["min_cross_row_span_ratio"], 0.0, 1.0
    ))
    # Preserve every raw fragment that falls inside an accepted terminal
    # envelope.  The envelope is bounded by the component-derived terminal
    # walls, so this does not promote an interior seam outside the endpoint.
    # ``accepted_region_rotated`` is the envelope built for region-qualified
    # terminal candidates.  Keep the name consistent with the allocation
    # above; using an undefined alias here made every accepted headland
    # candidate fail before it could be returned to the semantic layers.
    corridor_rotated = cv2.bitwise_or(
        corridor_rotated, accepted_region_rotated
    )
    corridor_labels_count, corridor_labels, corridor_stats, _ = (
        cv2.connectedComponentsWithStats(
            (corridor_rotated > 0).astype(np.uint8), 8
        )
    )
    filtered_corridor = np.zeros_like(corridor_rotated)
    for component_label in range(1, corridor_labels_count):
        component_area = int(corridor_stats[component_label, cv2.CC_STAT_AREA])
        component_x = int(corridor_stats[component_label, cv2.CC_STAT_LEFT])
        component_y = int(corridor_stats[component_label, cv2.CC_STAT_TOP])
        component_width = int(corridor_stats[component_label, cv2.CC_STAT_WIDTH])
        component_height = int(corridor_stats[component_label, cv2.CC_STAT_HEIGHT])
        component_span_ratio = float(min(
            1.0,
            component_height / max(float(body_cross_span), 1.0),
        ))
        component_slice = corridor_labels[component_y:component_y + component_height,
                                          component_x:component_x + component_width] == component_label
        envelope_slice = accepted_region_rotated[
            component_y:component_y + component_height,
            component_x:component_x + component_width,
        ] > 0
        envelope_overlap = int(np.count_nonzero(component_slice & envelope_slice))
        if component_span_ratio < component_min_span_ratio and envelope_overlap == 0:
            component_report["rejected_short_count"] += 1
            component_report["rejected"].append({
                "area_px": component_area,
                "top": component_y,
                "height": component_height,
                "cross_row_span_ratio": component_span_ratio,
            })
            continue
        if component_span_ratio < component_min_span_ratio:
            component_report["accepted_envelope_fragment_count"] += 1
        filtered_corridor[corridor_labels == component_label] = 255
        component_report["accepted_count"] += 1
    corridor_rotated = filtered_corridor

    if cv2.countNonZero(corridor_rotated) == 0:
        return body, empty, empty, diagnostics

    inverse = cv2.invertAffineTransform(matrix)
    corridor = cv2.warpAffine(
        corridor_rotated,
        inverse,
        (body.shape[1], body.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    corridor = (corridor > 0).astype(np.uint8) * 255
    evidence = cv2.bitwise_or(raw, body)
    corridor = cv2.bitwise_and(corridor, evidence)
    clean_body = cv2.bitwise_and(body, cv2.bitwise_not(corridor))
    accepted_seed = cv2.warpAffine(
        accepted_seed_rotated,
        inverse,
        (body.shape[1], body.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    accepted_seed = (accepted_seed > 0).astype(np.uint8) * 255
    # Region qualification decides whether a candidate may participate; the
    # robust endpoint wall still decides where the semantic headland ends.
    # Keep accepted seed evidence only inside that final corridor envelope.
    if has_axial_reference:
        accepted_seed = cv2.bitwise_and(accepted_seed, corridor)
    resolved_headland = cv2.bitwise_or(
        accepted_seed, cv2.bitwise_and(corridor, raw)
    )
    diagnostics["corridor_px"] = int(cv2.countNonZero(corridor))
    diagnostics["removed_body_px"] = int(cv2.countNonZero(
        cv2.bitwise_and(body, corridor)
    ))
    return clean_body, resolved_headland, corridor, diagnostics


def clip_bands_to_body(
    bands: List[Dict[str, Any]],
    body_mask: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Synchronize band contours and centerlines with the resolved body mask."""
    body = (np.asarray(body_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    height, width = body.shape[:2]
    kept: List[Dict[str, Any]] = []
    dropped: List[int] = []
    for original in bands or []:
        band = dict(original)
        contour_value = band.get("contour")
        contour = np.asarray(
            contour_value if contour_value is not None else [], dtype=np.int32
        ).reshape(-1, 2)
        if contour.shape[0] < 3:
            dropped.append(int(band.get("id", -1)))
            continue
        band_mask = np.zeros_like(body)
        cv2.fillPoly(band_mask, [contour], 255)
        clipped = cv2.bitwise_and(band_mask, body)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (clipped > 0).astype(np.uint8), 8
        )
        if count <= 1:
            dropped.append(int(band.get("id", -1)))
            continue
        selected_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        selected = (labels == selected_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            dropped.append(int(band.get("id", -1)))
            continue
        clipped_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

        points = [tuple(map(float, point)) for point in band.get("centerline", [])]
        runs: List[List[Tuple[float, float]]] = []
        current: List[Tuple[float, float]] = []
        for point in points:
            px = int(round(point[0]))
            py = int(round(point[1]))
            valid = (
                0 <= px < width and 0 <= py < height and selected[py, px] > 0
            )
            if valid:
                current.append(point)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        centerline = max(runs, key=len) if runs else []
        if len(centerline) < 2:
            dropped.append(int(band.get("id", -1)))
            continue

        bx, by, bw, bh = cv2.boundingRect(clipped_contour)
        centroid = np.mean(np.asarray(centerline, dtype=np.float64), axis=0)
        band["contour"] = clipped_contour
        band["centerline"] = centerline
        band["centroid"] = (float(centroid[0]), float(centroid[1]))
        band["bbox"] = (int(bx), int(by), int(bw), int(bh))
        band["area"] = float(cv2.contourArea(clipped_contour.astype(np.float32)))
        kept.append(band)
    return kept, dropped


def residual_headland_mask(
    raw_mask: np.ndarray,
    rebuilt_mask: np.ndarray,
    main_angle: float,
    angle_thresh_deg: float = 60.0,
    min_area_ratio: float = 0.001,
) -> np.ndarray:
    """Return only headland-like residual pixels, not every missed row-band pixel."""
    headland, _body_residual, _uncertain = residual_mask_layers(
        raw_mask,
        rebuilt_mask,
        main_angle,
        angle_thresh_deg=angle_thresh_deg,
        min_area_ratio=min_area_ratio,
    )
    return headland


def repair_center_samples(
    centers: np.ndarray,
    widths: np.ndarray,
    observed: np.ndarray,
    target_width_px: float,
    sample_step: int,
    work_mpp: float,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Bridge weak observations along the local row direction."""
    centers = np.asarray(centers, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)

    provisional_widths = fill_nan(widths)
    smooth_m = float(config.get("center_outlier_smooth_m", 1.2))
    smooth_n = max(
        3,
        int(round(smooth_m / max(work_mpp * sample_step, 1e-6))),
    )
    if smooth_n % 2 == 0:
        smooth_n += 1
    local_width_reference = smooth_1d(provisional_widths, smooth_n)

    min_width_ratio = float(config.get("gap_min_width_ratio", 0.72))
    local_width_ratio = float(config.get("gap_local_width_ratio", 0.86))
    trusted = (
        observed
        & np.isfinite(centers)
        & np.isfinite(widths)
        & (widths >= max(1.0, target_width_px * min_width_ratio))
        & (widths >= local_width_reference * local_width_ratio)
    )

    provisional = fill_nan(centers)
    trend = smooth_1d(provisional, smooth_n)
    outlier_px = float(config.get("center_outlier_m", 0.10)) / max(work_mpp, 1e-6)
    trusted &= np.abs(centers - trend) <= max(2.0, outlier_px)

    if np.count_nonzero(trusted) < 4:
        trusted = observed & np.isfinite(centers)

    repaired = fill_nan(np.where(trusted, centers, np.nan))
    context_m = float(config.get("gap_center_context_m", 0.9))
    context_n = max(
        3,
        int(round(context_m / max(work_mpp * sample_step, 1e-6))),
    )
    max_shift_px = float(config.get("gap_max_lateral_shift_m", 0.05)) / max(
        work_mpp,
        1e-6,
    )

    weak = ~trusted
    run_start = None
    for index in range(len(weak) + 1):
        is_weak = index < len(weak) and weak[index]
        if is_weak and run_start is None:
            run_start = index
            continue
        if is_weak or run_start is None:
            continue

        run_end = index
        left_idx = np.flatnonzero(trusted[:run_start])
        right_idx = np.flatnonzero(trusted[run_end:]) + run_end
        left_values = centers[left_idx[-context_n:]] if left_idx.size else np.array([])
        right_values = centers[right_idx[:context_n]] if right_idx.size else np.array([])

        if left_values.size and right_values.size:
            left_center = float(np.median(left_values))
            right_center = float(np.median(right_values))
            middle = 0.5 * (left_center + right_center)
            delta = float(np.clip(
                right_center - left_center,
                -max_shift_px,
                max_shift_px,
            ))
            repaired[run_start:run_end] = np.linspace(
                middle - 0.5 * delta,
                middle + 0.5 * delta,
                run_end - run_start + 2,
            )[1:-1]
        elif left_values.size:
            repaired[run_start:run_end] = float(np.median(left_values))
        elif right_values.size:
            repaired[run_start:run_end] = float(np.median(right_values))

        run_start = None

    return repaired, trusted


def rotate_bound(
    image: np.ndarray,
    angle_deg: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate without clipping and return the affine matrix."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    out_w = int(math.ceil(h * sin_a + w * cos_a))
    out_h = int(math.ceil(h * cos_a + w * sin_a))
    matrix[0, 2] += out_w / 2.0 - center[0]
    matrix[1, 2] += out_h / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated, matrix


def estimate_row_angle(mask: np.ndarray) -> float:
    """Estimate crop-row direction from global and major-region PCA seeds."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if xs.size < 100:
        return 0.0

    step = max(1, xs.size // 400000)
    sample_x = xs[::step].astype(np.float64)
    sample_y = ys[::step].astype(np.float64)
    sample_x -= float(np.mean(sample_x))
    sample_y -= float(np.mean(sample_y))
    # Closed-form principal-axis angle for a 2D covariance matrix.  Avoid
    # np.cov/eigh here: this hot path only needs one angle, while the target
    # Windows Conda runtime can abort inside its BLAS-backed covariance call.
    # The common positive scale factor (N or N-1) cancels in atan2.
    covariance_xx = float(np.sum(sample_x * sample_x, dtype=np.float64))
    covariance_xy = float(np.sum(sample_x * sample_y, dtype=np.float64))
    covariance_yy = float(np.sum(sample_y * sample_y, dtype=np.float64))
    global_coarse = 0.5 * math.atan2(
        2.0 * covariance_xy,
        covariance_xx - covariance_yy,
    ) % math.pi
    coarse_candidates = [global_coarse]

    # Multiple separated planting regions can make the line between their
    # centroids dominate global PCA.  Major-region moments retain the row axis
    # inside each region, so include their axes as alternative search seeds.
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    ranked_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
    largest_area = cv2.contourArea(ranked_contours[0]) if ranked_contours else 0.0
    for contour in ranked_contours:
        area = cv2.contourArea(contour)
        if area < max(100.0, largest_area * 0.05):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            continue
        candidate = 0.5 * math.atan2(
            2.0 * moments["mu11"],
            moments["mu20"] - moments["mu02"],
        ) % math.pi
        if all(
            abs((candidate - existing + math.pi / 2.0) % math.pi - math.pi / 2.0)
            > math.radians(3.0)
            for existing in coarse_candidates
        ):
            coarse_candidates.append(candidate)

    coords = np.column_stack([sample_x, sample_y])
    best_angle = global_coarse
    best_score = -1.0
    for coarse in coarse_candidates:
        for angle in np.linspace(
            coarse - math.radians(12),
            coarse + math.radians(12),
            49,
        ):
            normal_x = -math.sin(angle)
            normal_y = math.cos(angle)
            projection = coords[:, 0] * normal_x + coords[:, 1] * normal_y
            bins = max(128, min(1600, int(np.ptp(projection)) + 1))
            histogram, _ = np.histogram(projection, bins=bins)
            trend = smooth_1d(histogram, max(5, bins // 80))
            high_frequency = histogram - trend
            score = float(np.std(high_frequency)) / max(float(np.mean(histogram)), 1.0)
            if score > best_score:
                best_score = score
                best_angle = angle % math.pi
    return best_angle


def _runs(binary: np.ndarray) -> List[Tuple[int, int]]:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return []
    changes = np.flatnonzero(np.diff(binary.astype(np.int8))) + 1
    bounds = np.concatenate(([0], changes, [binary.size]))
    return [
        (int(bounds[index]), int(bounds[index + 1]))
        for index in range(len(bounds) - 1)
        if binary[bounds[index]]
    ]


def _merge_close_runs(
    runs: List[Tuple[int, int]],
    max_gap: int,
) -> List[Tuple[int, int]]:
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(item) for item in merged]


def _profile_run_gates(
    runs: List[Tuple[int, int]],
    height: int,
) -> List[Tuple[int, int]]:
    """Build local occupancy gates from neighbouring profile-run edges."""
    if not runs:
        return []
    gates = []
    for index, (run_start, run_end) in enumerate(runs):
        if index == 0:
            lower = int(round(
                run_start - max(1.0, (run_end - run_start) * 0.5)
            ))
        else:
            lower = int(round((runs[index - 1][1] + run_start) * 0.5))
        if index == len(runs) - 1:
            upper = int(round(
                run_end + max(1.0, (run_end - run_start) * 0.5)
            ))
        else:
            upper = int(round((run_end + runs[index + 1][0]) * 0.5))
        gates.append((max(0, lower), min(int(height), upper)))
    return gates


def _longest_true_run(values: np.ndarray) -> Optional[Tuple[int, int]]:
    runs = _runs(values)
    return max(runs, key=lambda item: item[1] - item[0]) if runs else None


def close_boolean_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if values.size == 0 or max_gap <= 0:
        return values.copy()
    runs = _merge_close_runs(_runs(values), int(max_gap))
    result = np.zeros(values.size, dtype=bool)
    for start, end in runs:
        result[start:end] = True
    return result


def close_band_support_gaps(
    values: np.ndarray,
    internal_max_gap: int,
    terminal_max_gap: int,
    terminal_guard: int,
) -> np.ndarray:
    """Close dual-anchored band gaps with a stricter threshold near endpoints.

    DECISION-009 prevents a terminal fragment from borrowing the much larger
    internal missing-crop threshold and becoming a false row extension.
    """
    source = np.asarray(values, dtype=bool)
    runs = _runs(source)
    if len(runs) < 2:
        return source.copy()
    result = source.copy()
    guard = max(0, int(terminal_guard))
    for left, right in zip(runs, runs[1:]):
        gap_start, gap_end = left[1], right[0]
        gap = gap_end - gap_start
        terminal = left[0] <= guard or right[1] >= source.size - guard
        limit = int(terminal_max_gap if terminal else internal_max_gap)
        if gap <= max(0, limit):
            result[gap_start:gap_end] = True
    return result


def bridge_internal_support_gaps(
    values: np.ndarray,
    max_gap: int,
    terminal_guard: int = 0,
    min_adjacent_run: int = 0,
) -> np.ndarray:
    """Bridge a sufficiently supported internal gap without filling a tail.

    ``close_band_support_gaps`` deliberately uses a strict terminal rule.  A
    separate pass is needed for an interrupted crop row: both sides of the
    interruption must be real, sufficiently long support runs before they may
    be joined.  This prevents a short terminal fragment from being promoted to
    a full work row while allowing an artificial break inside the same row to
    remain one continuous path candidate.
    """
    source = np.asarray(values, dtype=bool)
    runs = _runs(source)
    if len(runs) < 2 or max_gap <= 0:
        return source.copy()
    result = source.copy()
    limit = max(0, int(max_gap))
    minimum_run = max(0, int(min_adjacent_run))
    guard = max(0, int(terminal_guard))
    for left, right in zip(runs, runs[1:]):
        gap_start, gap_end = left[1], right[0]
        if gap_end - gap_start > limit:
            continue
        if (
            left[1] - left[0] < minimum_run
            or right[1] - right[0] < minimum_run
        ):
            continue
        # A short run touching either extent is a terminal fragment.  Do not
        # bridge it merely because the gap itself is below the internal limit.
        left_is_short_terminal = (
            left[0] <= guard and left[1] - left[0] < minimum_run * 2
        )
        right_is_short_terminal = (
            right[1] >= source.size - guard
            and right[1] - right[0] < minimum_run * 2
        )
        if left_is_short_terminal or right_is_short_terminal:
            continue
        result[gap_start:gap_end] = True
    return result


def estimate_body_interval(
    foreground: np.ndarray,
    runs: List[Tuple[int, int]],
    work_mpp: float,
    config: Dict,
) -> Tuple[int, int, Dict]:
    """Find the longitudinal interval containing the repeated row lattice."""
    height, width = foreground.shape
    band_selector = np.zeros(height, dtype=bool)
    for start, end in runs:
        band_selector[start:end] = True
    gap_selector = np.zeros(height, dtype=bool)
    lattice_start = min(start for start, _ in runs)
    lattice_end = max(end for _, end in runs)
    gap_selector[lattice_start:lattice_end] = ~band_selector[lattice_start:lattice_end]
    if not np.any(band_selector) or not np.any(gap_selector):
        return 0, width, {"reason": "invalid_band_gap_selector"}

    band_signal = np.mean(foreground[band_selector], axis=0).astype(np.float64)
    gap_signal = np.mean(foreground[gap_selector], axis=0).astype(np.float64)
    # Keep the unsmoothed cross-row evidence.  The quality signals below are
    # intentionally smoothed to tolerate broken rows, but that same smoothing
    # can smear a transverse headland into the first/last body columns.  A
    # terminal guard must therefore use the local gap occupancy before the
    # longitudinal smoothing step.
    raw_band_signal = band_signal.copy()
    raw_gap_signal = gap_signal.copy()
    continuity = np.zeros(width, dtype=np.float64)
    for start, end in runs:
        continuity += np.mean(foreground[start:end], axis=0) >= float(
            config.get("body_band_presence_threshold", 0.18)
        )
    continuity /= max(len(runs), 1)

    smooth_px = max(
        3,
        float(config.get("body_quality_smooth_m", 0.45)) / work_mpp,
    )
    band_signal = smooth_1d(band_signal, smooth_px)
    gap_signal = smooth_1d(gap_signal, smooth_px)
    continuity = smooth_1d(continuity, smooth_px)
    contrast = np.clip(band_signal - gap_signal, 0.0, 1.0)

    quality = continuity * contrast
    positive = quality[quality > 0]
    if positive.size == 0:
        return 0, width, {"reason": "no_positive_body_quality"}
    quality_threshold = max(
        float(config.get("body_quality_min", 0.04)),
        min(
            float(config.get("body_quality_max", 0.18)),
            float(np.quantile(positive, 0.15)) * 0.6,
        ),
    )
    continuity_threshold = float(config.get("body_continuity_min", 0.42))
    contrast_threshold = float(config.get("body_contrast_min", 0.08))
    terminal_gap_min_occupancy = float(np.clip(
        config.get(
            "body_terminal_gap_min_occupancy",
            max(0.20, contrast_threshold * 1.5),
        ),
        0.0,
        1.0,
    ))
    terminal_gap_ratio = float(np.clip(
        config.get("body_terminal_gap_ratio", 0.65),
        0.0,
        1.0,
    ))
    # A real row tail keeps the inter-row gaps mostly empty.  A transverse
    # headland occupies both the detected row bands and their gaps.  Record
    # this as endpoint evidence, but do not remove arbitrary interior runs:
    # only overlap with the selected body's two ends is allowed to trim the
    # interval below.
    terminal_like = (
        (raw_gap_signal >= terminal_gap_min_occupancy)
        & (raw_gap_signal >= raw_band_signal * terminal_gap_ratio)
        & (continuity >= max(0.30, continuity_threshold * 0.75))
    )
    terminal_gap_close_px = max(1, int(round(
        float(config.get("body_support_extension_max_gap_m", 0.8))
        / max(work_mpp, 1e-6)
    )))
    terminal_like = close_boolean_gaps(
        terminal_like,
        terminal_gap_close_px,
    )
    body_candidate = (
        (quality >= quality_threshold)
        & (continuity >= continuity_threshold)
        & (contrast >= contrast_threshold)
    )

    close_px = odd_size(
        float(config.get("body_gap_close_m", 1.0)) / work_mpp,
        minimum=3,
    )
    candidate_u8 = body_candidate.astype(np.uint8).reshape(1, -1) * 255
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, 1)),
    )
    body_run = _longest_true_run(candidate_u8.reshape(-1) > 0)
    if body_run is None:
        return 0, width, {"reason": "no_body_run"}

    start, end = body_run
    terminal_trim_start = None
    terminal_trim_end = None
    for terminal_start, terminal_end in _runs(terminal_like):
        if terminal_start <= start < terminal_end:
            terminal_trim_start = max(
                int(start),
                int(terminal_end),
            )
        if terminal_start < end <= terminal_end:
            terminal_trim_end = min(
                int(end),
                int(terminal_start),
            )
    if terminal_trim_start is not None and terminal_trim_start < end:
        start = terminal_trim_start
    if terminal_trim_end is not None and start < terminal_trim_end:
        end = terminal_trim_end

    # The high-quality body run is good for rejecting true headlands, but weak
    # end-row pixels can still be harvestable. Extend the interval through
    # row-supported tails that keep the repeated row lattice, instead of ending
    # the work lines as soon as the strict quality score drops.
    support_min_continuity = float(config.get(
        "body_support_extension_min_continuity",
        max(0.25, continuity_threshold * 0.60),
    ))
    support_min_contrast = float(config.get(
        "body_support_extension_min_contrast",
        max(0.03, contrast_threshold * 0.50),
    ))
    support_candidate = (
        (continuity >= support_min_continuity)
        & (contrast >= support_min_contrast)
        & ~terminal_like
    )
    support_gap_px = odd_size(
        float(config.get("body_support_extension_max_gap_m", 0.8))
        / max(work_mpp, 1e-6),
        minimum=3,
    )
    support_u8 = support_candidate.astype(np.uint8).reshape(1, -1) * 255
    support_u8 = cv2.morphologyEx(
        support_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (support_gap_px, 1)),
    )
    support_runs = _runs(support_u8.reshape(-1) > 0)
    overlapping = [
        item for item in support_runs
        if item[0] <= end and item[1] >= start
    ]
    if overlapping:
        support_start, support_end = max(
            overlapping,
            key=lambda item: (min(item[1], end) - max(item[0], start), item[1] - item[0]),
        )
        start = min(start, support_start)
        end = max(end, support_end)

    extension_px = int(round(
        float(config.get("body_end_extension_m", 0.20)) / work_mpp
    ))
    start = max(0, start - extension_px)
    end = min(width, end + extension_px)
    min_length_px = int(round(
        float(config.get("body_min_length_m", 8.0)) / work_mpp
    ))
    if end - start < min_length_px:
        return 0, width, {"reason": "body_run_too_short"}

    return start, end, {
        "quality_threshold": quality_threshold,
        "support_extension_min_continuity": support_min_continuity,
        "support_extension_min_contrast": support_min_contrast,
        "terminal_guard": {
            "enabled": True,
            "min_gap_occupancy": terminal_gap_min_occupancy,
            "gap_to_band_ratio": terminal_gap_ratio,
            "closed_gap_px": terminal_gap_close_px,
            "trimmed_start_px": int(terminal_trim_start)
            if terminal_trim_start is not None else None,
            "trimmed_end_px": int(terminal_trim_end)
            if terminal_trim_end is not None else None,
            "candidate_run_count": int(len(_runs(terminal_like))),
        },
        "mean_continuity": float(np.mean(continuity[start:end])),
        "mean_contrast": float(np.mean(contrast[start:end])),
        "body_start_work_px": int(start),
        "body_end_work_px": int(end),
        "support_extended_start_px": int(start),
        "support_extended_end_px": int(end),
        "body_length_m": float((end - start) * work_mpp),
    }


def classify_band_widths(
    bands: List[Dict],
    fallback_threshold_m: float = 0.55,
    alternation_penalty: float = 0.0,
) -> Tuple[List[Dict], List[Dict], float]:
    """Classify ordered bands with width evidence and a soft sequence prior.

    alternation_penalty <= 0 disables the wide/narrow alternation prior:
    等距种植区可能出现连续 4 行带，不能盲目按“隔一个”强制交替
    （用户明确要求按检测宽度分类）。聚类前剔除离群值（田埂/边缘
    残留带，如剖面宽 12m 的带），避免阈值被拉偏。
    """
    if not bands:
        return [], [], fallback_threshold_m
    if len(bands) < 6:
        wide = [item for item in bands if item["width_m"] >= fallback_threshold_m]
        narrow = [item for item in bands if item["width_m"] < fallback_threshold_m]
        for item in wide:
            item["band_type"] = "wide"
        for item in narrow:
            item["band_type"] = "narrow"
        return wide, narrow, fallback_threshold_m

    widths = np.asarray([
        item.get("classification_width_m", item["width_m"])
        for item in bands
    ], dtype=np.float64)
    # 剔除离群值，避免聚类中心被拉偏
    median_w = float(np.median(widths))
    core = widths[widths <= max(median_w * 3.0, median_w + 2.0)]
    if len(core) < 4:
        core = widths
    center_a, center_b = np.quantile(core, [0.25, 0.75])
    for _ in range(20):
        distance_a = np.abs(core - center_a)
        distance_b = np.abs(core - center_b)
        group_a = core[distance_a <= distance_b]
        group_b = core[distance_a > distance_b]
        if group_a.size == 0 or group_b.size == 0:
            break
        new_a = float(np.mean(group_a))
        new_b = float(np.mean(group_b))
        if abs(new_a - center_a) + abs(new_b - center_b) < 1e-6:
            break
        center_a, center_b = new_a, new_b

    low, high = sorted((float(center_a), float(center_b)))
    threshold = (low + high) / 2.0
    separation = (high - low) / max(low, 1e-6)
    low_count = int(np.count_nonzero(widths < threshold))
    high_count = int(np.count_nonzero(widths >= threshold))
    if separation < 0.16 or min(low_count, high_count) < 3:
        threshold = fallback_threshold_m
        states = (widths >= threshold).astype(np.int8)
    elif alternation_penalty <= 0.0:
        # 纯宽度证据分类：等距区连续 4 行带保持 wide，不强制交替
        states = (widths >= threshold).astype(np.int8)
    else:
        scale = max((high - low) * 0.55, 0.06)
        emission = np.column_stack([
            ((widths - low) / scale) ** 2,
            ((widths - high) / scale) ** 2,
        ])
        count = len(bands)
        cost = np.full((count, 2), np.inf, dtype=np.float64)
        previous = np.zeros((count, 2), dtype=np.int8)
        cost[0] = emission[0]
        for index in range(1, count):
            for state_value in (0, 1):
                options = cost[index - 1] + np.asarray([
                    alternation_penalty if prior == state_value else 0.0
                    for prior in (0, 1)
                ])
                prior_state = int(np.argmin(options))
                previous[index, state_value] = prior_state
                cost[index, state_value] = options[prior_state] + emission[index, state_value]
        states = np.zeros(count, dtype=np.int8)
        states[-1] = int(np.argmin(cost[-1]))
        for index in range(count - 1, 0, -1):
            states[index - 1] = previous[index, states[index]]

    for item, state_value in zip(bands, states):
        item["band_type"] = "wide" if state_value else "narrow"
        item["class_width_m"] = high if state_value else low
    wide = [item for item in bands if item["band_type"] == "wide"]
    narrow = [item for item in bands if item["band_type"] == "narrow"]
    return wide, narrow, threshold


def regularize_crop_mask(
    mask: np.ndarray,
    geo=None,
    state=None,
    config: Optional[Dict] = None,
    progress_callback=None,
) -> Dict:
    """Rebuild each crop band as a smooth polygon in a row-aligned frame."""
    cfg = config or {}
    cancel_callback = cfg.get("_cancel_callback")

    def abort_if_cancelled():
        if callable(cancel_callback) and cancel_callback():
            raise RuntimeError("task cancelled")

    abort_if_cancelled()
    raw = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if np.count_nonzero(raw) < 100:
        empty_layer = np.zeros_like(raw)
        return {
            "processed_mask": raw,
            "regularized_body_mask": raw.copy(),
            "body_residual_mask": empty_layer.copy(),
            "gap_repair_added_mask": empty_layer.copy(),
            "support_clip_removed_mask": empty_layer.copy(),
            "headland_corridor_mask": empty_layer.copy(),
            "headland_mask": empty_layer.copy(),
            "uncertain_residual_mask": empty_layer.copy(),
            "neutral_support_mask": empty_layer.copy(),
            "planning_support_mask": raw.copy(),
            "main_angle": 0.0,
            "wide_bands": [],
            "narrow_bands": [],
            "all_bands": [],
            "diagnostics": {"reason": "empty_mask"},
        }

    def report(value: float, stage: str):
        if progress_callback:
            try:
                progress_callback(float(value), stage)
            except Exception:
                pass

    h, w = raw.shape[:2]
    field_polygon = field_polygon_local(state, raw.shape)
    mpp = meters_per_pixel(raw, geo, state)
    # DECISION-020: mask analysis uses the field polygon expanded outward
    # so a tightly circled field keeps the row-surrounding structure the
    # profile needs (a tight clip previously cut the model output down to
    # one connected blob and band reconstruction failed with
    # 种植带重构失败).  Bands and centrelines are still clipped back to
    # the ORIGINAL field polygon below.
    analysis_polygon = field_polygon
    if field_polygon is not None:
        analysis_expand_m = float(cfg.get("field_analysis_expand_m", 6.0))
        if analysis_expand_m > 0:
            analysis_polygon = expand_polygon_px(
                field_polygon,
                analysis_expand_m / mpp,
                h,
                w,
            )
        clip_mask_to_polygon_stripes(raw, analysis_polygon)
    max_work_dim = int(cfg.get("max_work_dim", 5000))
    scale = min(1.0, float(max_work_dim) / max(h, w))
    work = cv2.resize(
        raw,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    work = (work >= int(cfg.get("resize_threshold", 96))).astype(np.uint8) * 255
    field_work_mask = None
    if analysis_polygon is not None:
        field_work_mask = np.zeros_like(work)
        cv2.fillPoly(
            field_work_mask,
            [np.round(analysis_polygon * scale).astype(np.int32)],
            255,
        )
        cv2.bitwise_and(work, field_work_mask, dst=work)
    work_mpp = mpp / scale

    report(0.08, "稻行方向估计")
    abort_if_cancelled()
    angle = estimate_row_angle(work)
    abort_if_cancelled()
    rotated, matrix = rotate_bound(work, math.degrees(angle), cv2.INTER_NEAREST)
    inverse = cv2.invertAffineTransform(matrix)
    foreground = rotated > 0

    report(0.18, "种植带投影分析")
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            odd_size(float(cfg.get("support_close_along_m", 1.2)) / work_mpp),
            odd_size(float(cfg.get("support_close_across_m", 1.6)) / work_mpp),
        ),
    )
    support = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, support_kernel) > 0
    support_counts = support.sum(axis=1).astype(np.float64)
    row_counts = foreground.sum(axis=1).astype(np.float64)
    profile = np.divide(
        row_counts,
        np.maximum(support_counts, 1.0),
        out=np.zeros_like(row_counts),
        where=support_counts > 0,
    )
    profile = smooth_1d(
        profile,
        float(cfg.get("profile_smooth_m", 0.06)) / work_mpp,
    )

    valid = support_counts > max(10, rotated.shape[1] * 0.08)
    if not np.any(valid):
        raise RuntimeError("无法建立田块支持区域")
    profile_u8 = np.clip(profile[valid] * 255.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(
        profile_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold = float(np.clip(
        otsu / 255.0,
        float(cfg.get("profile_threshold_min", 0.25)),
        float(cfg.get("profile_threshold_max", 0.72)),
    ))
    crop_rows = profile >= threshold
    runs = _runs(crop_rows)
    runs = _merge_close_runs(
        runs,
        max_gap=max(1, int(round(float(cfg.get("merge_gap_m", 0.08)) / work_mpp))),
    )
    min_band_px = max(
        3,
        int(round(float(cfg.get("min_band_width_m", 0.16)) / work_mpp)),
    )
    runs = [(start, end) for start, end in runs if end - start >= min_band_px]
    if len(runs) < 2:
        raise RuntimeError(f"仅检测到 {len(runs)} 条稳定种植带")

    report(0.26, "田头主体区间识别")
    abort_if_cancelled()
    body_start, body_end, body_diagnostics = estimate_body_interval(
        foreground,
        runs,
        work_mpp,
        cfg,
    )
    abort_if_cancelled()

    # Re-estimate the transverse row profile inside the body only.
    body_foreground = foreground[:, body_start:body_end]
    body_support = support[:, body_start:body_end]
    body_support_counts = body_support.sum(axis=1).astype(np.float64)
    body_row_counts = body_foreground.sum(axis=1).astype(np.float64)
    body_profile = np.divide(
        body_row_counts,
        np.maximum(body_support_counts, 1.0),
        out=np.zeros_like(body_row_counts),
        where=body_support_counts > 0,
    )
    body_profile = smooth_1d(
        body_profile,
        float(cfg.get("profile_smooth_m", 0.06)) / work_mpp,
    )
    body_valid = body_support_counts > max(10, (body_end - body_start) * 0.08)
    if np.any(body_valid):
        body_profile_u8 = np.clip(
            body_profile[body_valid] * 255.0,
            0,
            255,
        ).astype(np.uint8)
        body_otsu, _ = cv2.threshold(
            body_profile_u8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        body_threshold = float(np.clip(
            body_otsu / 255.0,
            float(cfg.get("profile_threshold_min", 0.25)),
            float(cfg.get("profile_threshold_max", 0.72)),
        ))
        body_runs = _runs(body_profile >= body_threshold)
        body_runs = _merge_close_runs(
            body_runs,
            max_gap=max(
                1,
                int(round(float(cfg.get("merge_gap_m", 0.08)) / work_mpp)),
            ),
        )
        body_runs = [
            (start, end)
            for start, end in body_runs
            if end - start >= min_band_px
        ]
        if len(body_runs) >= 2:
            runs = body_runs
            threshold = body_threshold
            body_start, body_end, refined_body_diagnostics = estimate_body_interval(
                foreground,
                runs,
                work_mpp,
                cfg,
            )
            body_diagnostics.update(refined_body_diagnostics)

    # Gate each profile run by its *run edges*, not by the midpoint of the
    # neighbouring centres.  A broad uniform block next to a normal four-row
    # band can be hundreds of pixels wide; centre-based gates then include the
    # broad block tail in the neighbouring local profile and pull its median
    # away from the actual four-row centreline.  Edge midpoints keep each run's
    # local occupancy isolated while preserving a small terminal halo.
    gates = _profile_run_gates(runs, rotated.shape[0])

    profile_bands = [
        {
            "id": index + 1,
            "width_m": float((end - start) * work_mpp),
            "classification_width_m": float((end - start) * work_mpp),
        }
        for index, (start, end) in enumerate(runs)
    ]
    profile_wide, profile_narrow, width_threshold = classify_band_widths(
        profile_bands,
        fallback_threshold_m=float(cfg.get("band_width_threshold_m", 0.55)),
        # 默认关闭宽窄交替强制（用户要求按检测宽度分类，等距区可连续 4 行）
        alternation_penalty=float(cfg.get("band_alternation_penalty", 0.0)),
    )
    class_by_id = {
        item["id"]: item["band_type"]
        for item in profile_bands
    }
    class_widths = {}
    for band_type, items in (("wide", profile_wide), ("narrow", profile_narrow)):
        if items:
            class_widths[band_type] = float(np.median([
                item["width_m"] for item in items
            ]))

    sample_step = max(
        2,
        int(round(float(cfg.get("boundary_sample_m", 0.12)) / work_mpp)),
    )
    half_window = max(
        1,
        int(round(float(cfg.get("boundary_window_m", 0.08)) / work_mpp)),
    )
    xs = np.arange(body_start, body_end, sample_step, dtype=np.int32)
    rebuilt_rotated = np.zeros_like(rotated)
    bands = []
    skipped_bands = []
    filled_gap_lengths_m = []

    report(0.34, "逐带边界跟踪")
    for band_index, ((run_start, run_end), (gate_start, gate_end)) in enumerate(zip(runs, gates)):
        abort_if_cancelled()
        center_observed = np.full(xs.size, np.nan, dtype=np.float64)
        width_observed = np.full(xs.size, np.nan, dtype=np.float64)
        observed = np.zeros(xs.size, dtype=bool)

        for index, x_value in enumerate(xs):
            x0 = max(0, int(x_value) - half_window)
            x1 = min(rotated.shape[1], int(x_value) + half_window + 1)
            local_support = support[gate_start:gate_end, x0:x1]
            if local_support.size == 0 or np.mean(local_support) < 0.12:
                continue
            local = foreground[gate_start:gate_end, x0:x1]
            occupancy = local.mean(axis=1)
            local_y = np.flatnonzero(
                occupancy >= float(cfg.get("local_occupancy_threshold", 0.18))
            )
            if local_y.size < 2:
                continue
            center_observed[index] = gate_start + float(np.median(local_y))
            low = float(np.quantile(local_y, 0.04))
            high = float(np.quantile(local_y, 0.96))
            width_observed[index] = max(2.0, high - low + 1.0)
            observed[index] = True

        observed_count = int(np.count_nonzero(observed))
        observed_indices = np.flatnonzero(observed)
        observed_span_ratio = (
            float((observed_indices[-1] - observed_indices[0] + 1) / max(xs.size, 1))
            if observed_count else 0.0
        )
        if (
            observed_count < int(cfg.get("min_band_observations", 4))
            or observed_span_ratio < float(cfg.get("min_band_span_ratio", 0.35))
        ):
            skipped_bands.append({
                "id": band_index + 1,
                "observations": observed_count,
                "span_ratio": observed_span_ratio,
                "profile_width_m": float((run_end - run_start) * work_mpp),
            })
            continue

        missing_inside = ~observed.copy()
        missing_inside[:observed_indices[0]] = False
        missing_inside[observed_indices[-1] + 1:] = False
        missing_runs = _runs(missing_inside)
        longest_gap_m = max(
            ((end - start) * sample_step * work_mpp for start, end in missing_runs),
            default=0.0,
        )
        filled_gap_lengths_m.append(float(longest_gap_m))

        band_id = band_index + 1
        band_type = class_by_id.get(band_id, "wide")
        profile_width_px = float(run_end - run_start)
        class_width_px = (
            class_widths.get(band_type, profile_width_px * work_mpp) / work_mpp
        )
        class_prior = float(cfg.get("class_width_prior_blend", 0.45))
        target_width = (
            (1.0 - class_prior) * profile_width_px
            + class_prior * class_width_px
        )

        centers_i, center_trusted = repair_center_samples(
            center_observed,
            width_observed,
            observed,
            target_width,
            sample_step,
            work_mpp,
            cfg,
        )
        widths_i = fill_nan(np.where(center_trusted, width_observed, np.nan))
        center_detail = float(cfg.get("center_detail_blend", 0.65))
        trusted_indices = np.flatnonzero(
            center_trusted & np.isfinite(center_observed)
        )
        if trusted_indices.size >= 2:
            fit_x = xs[trusted_indices].astype(np.float64)
            fit_y = center_observed[trusted_indices].astype(np.float64)
            fit_x_mean = float(np.mean(fit_x))
            fit_y_mean = float(np.mean(fit_y))
            centered_x = fit_x - fit_x_mean
            denominator = float(np.sum(centered_x * centered_x, dtype=np.float64))
            if denominator > 1e-12:
                slope = float(
                    np.sum(
                        centered_x * (fit_y - fit_y_mean),
                        dtype=np.float64,
                    )
                    / denominator
                )
                intercept = fit_y_mean - slope * fit_x_mean
                row_trend = slope * xs.astype(np.float64) + intercept
            else:
                row_trend = np.full_like(
                    xs,
                    fit_y_mean,
                    dtype=np.float64,
                )
            centers_i = row_trend + center_detail * (
                centers_i - row_trend
            )

        width_detail = float(cfg.get("width_detail_blend", 0.20))
        widths_i = target_width + width_detail * (
            widths_i - float(np.nanmedian(widths_i))
        )
        width_clip = float(cfg.get("width_clip_ratio", 0.28))
        widths_i = np.clip(
            widths_i,
            target_width * (1.0 - width_clip),
            target_width * (1.0 + width_clip),
        )
        centers_i = smooth_1d(
            centers_i,
            float(cfg.get("center_smooth_m", 0.55)) / (sample_step * work_mpp),
        )
        widths_i = smooth_1d(
            widths_i,
            float(cfg.get("width_smooth_m", 0.45)) / (sample_step * work_mpp),
        )

        sample_m = max(sample_step * work_mpp, 1e-6)
        internal_gap_samples = int(
            round(float(cfg.get("band_internal_gap_close_m", 6.0)) / sample_m)
        )
        extent_seed = observed & np.isfinite(center_observed)
        trusted_seed = center_trusted & np.isfinite(center_observed)
        if np.count_nonzero(trusted_seed) >= int(cfg.get("min_band_observations", 4)):
            extent_seed = trusted_seed
        terminal_gap_samples = int(round(
            float(cfg.get(
                "band_end_gap_close_m",
                cfg.get("band_endpoint_gap_close_m", 0.35),
            )) / sample_m
        ))
        terminal_guard_samples = int(round(
            float(cfg.get("band_end_guard_m", 1.0)) / sample_m
        ))
        extent_mask = close_band_support_gaps(
            extent_seed,
            internal_max_gap=internal_gap_samples,
            terminal_max_gap=terminal_gap_samples,
            terminal_guard=terminal_guard_samples,
        )
        # A broken crop row must not become two work-line endpoints.  Bridge
        # only an internal gap whose two neighbouring support runs are both
        # substantial; the terminal pass above remains authoritative for
        # headland/tail fragments.  Bound the configured bridge by half of the
        # detected body span so a pathological value cannot fill a whole row.
        body_length_m = max(0.0, float(body_end - body_start) * work_mpp)
        configured_bridge_m = max(
            0.0,
            float(cfg.get("band_internal_gap_bridge_m", 30.0)),
        )
        bridge_limit_m = min(
            configured_bridge_m,
            body_length_m * 0.50,
        )
        terminal_fragment_min_m = max(
            0.0,
            float(cfg.get("band_terminal_fragment_min_m", 4.0)),
        )
        before_internal_bridge = extent_mask.copy()
        extent_mask = bridge_internal_support_gaps(
            extent_mask,
            max_gap=int(round(bridge_limit_m / sample_m)),
            terminal_guard=terminal_guard_samples,
            min_adjacent_run=int(round(terminal_fragment_min_m / sample_m)),
        )
        internal_bridge_added_px = int(np.count_nonzero(
            extent_mask & ~before_internal_bridge
        ))
        extent_runs = _runs(extent_mask)
        if not extent_runs:
            skipped_bands.append({
                "id": band_id,
                "reason": "no_longitudinal_support",
            })
            continue
        extent_start, extent_end = max(
            extent_runs,
            key=lambda item: (
                item[1] - item[0],
                int(np.count_nonzero(trusted_seed[item[0]:item[1]])),
            ),
        )
        trim_samples = int(round(float(cfg.get("band_end_trim_m", 0.10)) / sample_m))
        extent_start = min(extent_end, extent_start + max(0, trim_samples))
        extent_end = max(extent_start, extent_end - max(0, trim_samples))
        min_extent_samples = max(
            4,
            int(round(float(cfg.get("min_band_extent_m", 1.2)) / sample_m)),
        )
        if extent_end - extent_start < min_extent_samples:
            skipped_bands.append({
                "id": band_id,
                "reason": "short_longitudinal_support",
                "extent_m": float((extent_end - extent_start) * sample_m),
            })
            continue

        margin = max(1.0, float(cfg.get("inter_band_margin_m", 0.03)) / work_mpp)
        half_width = widths_i * 0.5
        lower = np.maximum(centers_i - half_width, gate_start + margin)
        upper = np.minimum(centers_i + half_width, gate_end - margin)
        centers_i = (lower + upper) * 0.5
        widths_i = np.maximum(2.0, upper - lower)

        x_i = xs[extent_start:extent_end].astype(np.float64)
        lower_i = lower[extent_start:extent_end]
        upper_i = upper[extent_start:extent_end]
        centers_i = centers_i[extent_start:extent_end]
        widths_i = widths_i[extent_start:extent_end]
        polygon_rotated = np.vstack([
            np.column_stack([x_i, lower_i]),
            np.column_stack([x_i[::-1], upper_i[::-1]]),
        ])

        centerline_rotated = np.column_stack([x_i, centers_i])
        centerline_work = cv2.transform(
            centerline_rotated.reshape(-1, 1, 2).astype(np.float64),
            inverse,
        ).reshape(-1, 2)
        centerline_full = centerline_work / scale

        interval_px = max(1.0, float(cfg.get("centerline_interval_m", 0.5)) / mpp)
        sampled = [centerline_full[0]]
        last_point = centerline_full[0]
        for point in centerline_full[1:]:
            if math.hypot(
                float(point[0] - last_point[0]),
                float(point[1] - last_point[1]),
            ) >= interval_px:
                sampled.append(point)
                last_point = point
        if math.hypot(
            float(centerline_full[-1, 0] - sampled[-1][0]),
            float(centerline_full[-1, 1] - sampled[-1][1]),
        ) > 1.0:
            sampled.append(centerline_full[-1])

        polygon_work = cv2.transform(
            polygon_rotated.reshape(-1, 1, 2).astype(np.float64),
            inverse,
        ).reshape(-1, 2)
        polygon_full = polygon_work / scale
        polygon_int = np.round(polygon_full).astype(np.int32)
        polygon_int = clip_band_polygon(polygon_int, field_polygon)
        if polygon_int is None or len(polygon_int) < 3:
            skipped_bands.append({
                "id": band_id,
                "reason": "outside_field_boundary",
            })
            continue
        if field_polygon is not None:
            sampled = [
                point
                for point in sampled
                if cv2.pointPolygonTest(
                    field_polygon.astype(np.float32),
                    point,
                    False,
                ) >= 0
            ]
            if len(sampled) < 2:
                skipped_bands.append({
                    "id": band_id,
                    "reason": "centerline_outside_field_boundary",
                })
                continue
        cv2.fillPoly(
            rebuilt_rotated,
            [np.round(polygon_rotated).astype(np.int32)],
            255,
        )
        bx, by, bw, bh = cv2.boundingRect(polygon_int)
        area_px = float(abs(cv2.contourArea(polygon_full.astype(np.float32))))
        centroid = np.mean(centerline_full, axis=0)

        bands.append({
            "id": band_id,
            "centroid": (float(centroid[0]), float(centroid[1])),
            "width_px": float(np.median(widths_i) / scale),
            "width_m": float(np.median(widths_i) * work_mpp),
            "profile_width_m": float(profile_width_px * work_mpp),
            "band_type": band_type,
            "area": area_px,
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "centerline": [(float(x), float(y)) for x, y in sampled],
            "contour": polygon_int,
            "observed_ratio": float(np.mean(observed)),
            "observed_span_ratio": observed_span_ratio,
            "longitudinal_extent_ratio": float((extent_end - extent_start) / max(xs.size, 1)),
            "longitudinal_extent_m": float((extent_end - extent_start) * sample_m),
            "largest_filled_gap_m": float(longest_gap_m),
            "internal_gap_bridge_limit_m": float(bridge_limit_m),
            "internal_gap_bridge_added_px": internal_bridge_added_px,
        })
        report(
            0.34 + 0.44 * (band_index + 1) / max(len(runs), 1),
            f"逐带重构 {band_index + 1}/{len(runs)}",
        )

    # Diagnostic detail in the failure message: report how many candidate
    # bands were detected and why each was skipped (observations, span
    # ratio vs min_band_span_ratio, longitudinal extent) so a failed run
    # is debuggable without re-instrumenting.
    if len(bands) < 2:
        raise RuntimeError(
            "种植带重构失败"
            f" runs={len(runs)} bands={len(bands)} skipped={len(skipped_bands)}"
            f" skip_detail={[{'id': s.get('id'), 'reason': s.get('reason'), 'obs': s.get('observations'), 'span': round(float(s.get('span_ratio', 0.0)), 3), 'extent_m': round(float(s.get('extent_m', 0.0)), 2)} for s in skipped_bands]}"
        )

    report(0.82, "几何回写")
    abort_if_cancelled()
    rebuilt_work = cv2.warpAffine(
        rebuilt_rotated,
        inverse,
        (work.shape[1], work.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rebuilt_work = (rebuilt_work >= 127).astype(np.uint8) * 255
    report(0.89, "恢复原始分辨率")
    abort_if_cancelled()
    rebuilt = cv2.resize(rebuilt_work, (w, h), interpolation=cv2.INTER_LINEAR)
    rebuilt = (rebuilt >= 127).astype(np.uint8) * 255
    clip_mask_to_polygon_stripes(rebuilt, field_polygon)

    report(0.94, "种植带分类")
    wide = [item for item in bands if item.get("band_type") == "wide"]
    narrow = [item for item in bands if item.get("band_type") == "narrow"]
    raw_bool = raw > 0
    rebuilt_bool = rebuilt > 0
    headland_mask, body_residual_mask, uncertain_residual_mask = residual_mask_layers(
        raw,
        rebuilt,
        angle,
        meters_per_px=mpp,
        config=cfg,
        angle_thresh_deg=float(cfg.get("headland_angle_thresh_deg", 60.0)),
        min_area_ratio=float(cfg.get("headland_min_area_ratio", 0.001)),
    )
    after_residual_merge = cv2.bitwise_or(rebuilt, body_residual_mask)
    after_gap_repair = repair_row_gaps(after_residual_merge, angle, mpp, cfg)
    gap_repair_added_mask = cv2.bitwise_and(
        after_gap_repair, cv2.bitwise_not(after_residual_merge)
    )
    visible_body = constrain_generated_body_to_raw_support(
        after_gap_repair, raw, angle, mpp, cfg
    )
    support_clip_removed_mask = cv2.bitwise_and(
        after_gap_repair, cv2.bitwise_not(visible_body)
    )
    visible_body, headland_mask, headland_corridor_mask, headland_corridor_diagnostics = (
        resolve_terminal_headland_overlap(
            raw,
            visible_body,
            headland_mask,
            angle,
            mpp,
            cfg,
            axial_reference_mask=rebuilt,
            field_polygon=field_polygon,
        )
    )
    body_residual_mask = cv2.bitwise_and(
        body_residual_mask, cv2.bitwise_not(headland_corridor_mask)
    )
    gap_repair_added_mask = cv2.bitwise_and(
        gap_repair_added_mask, cv2.bitwise_not(headland_corridor_mask)
    )
    uncertain_residual_mask = cv2.bitwise_and(
        uncertain_residual_mask, cv2.bitwise_not(headland_corridor_mask)
    )
    if headland_corridor_diagnostics.get("corridor_count", 0):
        bands, corridor_dropped_band_ids = clip_bands_to_body(bands, visible_body)
    else:
        corridor_dropped_band_ids = []
    wide = [item for item in bands if item.get("band_type") == "wide"]
    narrow = [item for item in bands if item.get("band_type") == "narrow"]
    visible_body_bool = visible_body > 0
    headland_bool = headland_mask > 0
    neutral_support_mask = uncertain_residual_mask
    planning_support_mask = cv2.bitwise_or(
        cv2.bitwise_or(visible_body, headland_mask),
        neutral_support_mask,
    )
    intersection = int(np.count_nonzero(raw_bool & visible_body_bool))
    union = int(np.count_nonzero(raw_bool | visible_body_bool))
    diagnostics = {
        "algorithm_version": MASK_REGULARIZATION_VERSION,
        "meters_per_pixel": float(mpp),
        "work_scale": float(scale),
        "work_meters_per_pixel": float(work_mpp),
        "profile_threshold": float(threshold),
        "band_width_threshold_m": float(width_threshold),
        "raw_area_px": int(np.count_nonzero(raw_bool)),
        "processed_area_px": int(np.count_nonzero(visible_body_bool)),
        "area_ratio": float(np.count_nonzero(visible_body_bool) / max(np.count_nonzero(raw_bool), 1)),
        "precision_to_raw": float(intersection / max(np.count_nonzero(visible_body_bool), 1)),
        "recall_to_raw": float(intersection / max(np.count_nonzero(raw_bool), 1)),
        "iou_to_raw": float(intersection / max(union, 1)),
        "headland_removed_px": int(np.count_nonzero(headland_bool)),
        "uncertain_residual_px": int(np.count_nonzero(uncertain_residual_mask)),
        "body_residual_px": int(np.count_nonzero(body_residual_mask)),
        "gap_repair_added_px": int(np.count_nonzero(gap_repair_added_mask)),
        "internal_gap_bridge_added_px": int(sum(
            int(item.get("internal_gap_bridge_added_px", 0))
            for item in bands
        )),
        "internal_gap_bridge_limit_m": float(
            max(
                (float(item.get("internal_gap_bridge_limit_m", 0.0))
                 for item in bands),
                default=0.0,
            )
        ),
        "support_clip_removed_px": int(np.count_nonzero(support_clip_removed_mask)),
        "headland_corridor_px": int(np.count_nonzero(headland_corridor_mask)),
        "headland_corridor": headland_corridor_diagnostics,
        "headland_corridor_dropped_band_ids": corridor_dropped_band_ids,
        "body_interval": body_diagnostics,
        "candidate_band_count": len(runs),
        "profile_gate_basis": "run_edge_midpoints",
        "rebuilt_band_count": len(bands),
        "skipped_bands": skipped_bands,
        "wide_profile_width_m": class_widths.get("wide"),
        "narrow_profile_width_m": class_widths.get("narrow"),
        "max_filled_gap_m": max(filled_gap_lengths_m, default=0.0),
        "field_boundary_clipped": bool(field_polygon is not None),
        "center_trend_preserved": True,
    }
    report(0.98, "结果统计")
    # DECISION-020 follow-up: the analysis polygon is expanded outward for
    # row reconstruction, but every OUTPUT layer must stay inside the
    # original user-drawn field boundary (mask never displays outside the
    # box).  Band polygons and centrelines are already clipped; this makes
    # the residual/headland/planning layers consistent with them.
    if field_polygon is not None:
        for _output_layer in (
            visible_body,
            rebuilt,
            body_residual_mask,
            gap_repair_added_mask,
            support_clip_removed_mask,
            headland_corridor_mask,
            headland_mask,
            uncertain_residual_mask,
            neutral_support_mask,
            planning_support_mask,
        ):
            clip_mask_to_polygon_stripes(_output_layer, field_polygon)
    report(1.0, "完成")
    return {
        "processed_mask": visible_body,
        "regularized_body_mask": rebuilt,
        "body_residual_mask": body_residual_mask,
        "gap_repair_added_mask": gap_repair_added_mask,
        "support_clip_removed_mask": support_clip_removed_mask,
        "headland_corridor_mask": headland_corridor_mask,
        "headland_mask": headland_mask,
        "uncertain_residual_mask": uncertain_residual_mask,
        "neutral_support_mask": neutral_support_mask,
        "planning_support_mask": planning_support_mask,
        "main_angle": float(angle),
        "wide_bands": wide,
        "narrow_bands": narrow,
        "all_bands": bands,
        "diagnostics": diagnostics,
    }
