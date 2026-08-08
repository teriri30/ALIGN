"""智能农机规划系统 - 物理运动学与拓扑规划模块"""
import math
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, Point

from config import Config, AppLogger
from state import AppState, BandInstance, Corridor, AutoPathSegment, PathPoint
from geo import GeoUtils
from row_orientation import estimate_mask_axis_fitline


def _contour_points(contour) -> np.ndarray:
    return np.asarray(contour, dtype=np.float32).reshape(-1, 2)


def _polyline_length_m(points: List[Tuple[float, float]], geo: Optional[GeoUtils]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        if geo and geo.is_ready():
            total += geo.pixel_distance_m(p1, p2)
        else:
            total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return total


def _normalize_vec(dx: float, dy: float) -> Tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return 0.0, 0.0
    return dx / norm, dy / norm


def _wrap_half_turn(angle: float) -> float:
    angle = float(angle) % math.pi
    return angle + math.pi if angle < 0.0 else angle


def _half_turn_diff(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + math.pi * 0.5) % math.pi) - math.pi * 0.5)


def _half_turn_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if ang.size == 0:
        return 0.0
    if w.size != ang.size:
        w = np.ones_like(ang, dtype=np.float64)
    valid = np.isfinite(ang) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return _wrap_half_turn(float(ang[0]))
    vec = np.sum(w[valid] * np.exp(1j * 2.0 * ang[valid]))
    if abs(vec) <= 1e-12:
        idx = int(np.argmax(w[valid]))
        return _wrap_half_turn(float(ang[valid][idx]))
    return _wrap_half_turn(0.5 * math.atan2(vec.imag, vec.real))


def _mask_meters_per_pixel(mask: np.ndarray, geo: Optional[GeoUtils], state: Optional[AppState]) -> float:
    if geo is None or state is None or not geo.is_ready():
        return 1.0
    ox = int(getattr(state, "mask_offset_x", 0))
    oy = int(getattr(state, "mask_offset_y", 0))
    h, w = mask.shape[:2]
    px = ox + w * 0.5
    py = oy + h * 0.5
    mpp = float(geo.meters_per_pixel(px, py))
    if math.isfinite(mpp) and 1e-5 <= mpp <= 5.0:
        return mpp
    raise ValueError(f"invalid metric scale: {mpp!r} m/px")


def _field_longest_edge_angle(state: Optional[AppState]) -> Optional[float]:
    pts = getattr(state, "field_boundary", None) if state is not None else None
    if not pts or len(pts) < 3:
        return None
    best_angle = None
    best_len = 0.0
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_len = math.hypot(dx, dy)
        if seg_len > best_len:
            best_len = seg_len
            best_angle = math.atan2(dy, dx)
    return _wrap_half_turn(best_angle) if best_angle is not None else None


def _crop_source_image(mask_shape: Tuple[int, int], state: Optional[AppState]) -> Optional[np.ndarray]:
    if state is None:
        return None
    img = getattr(state, "img_raw", None)
    if img is None or getattr(img, "ndim", 0) < 2:
        return None

    h, w = mask_shape[:2]
    ox = int(getattr(state, "mask_offset_x", 0))
    oy = int(getattr(state, "mask_offset_y", 0))
    img_h, img_w = img.shape[:2]
    x0 = max(0, ox)
    y0 = max(0, oy)
    x1 = min(img_w, ox + w)
    y1 = min(img_h, oy + h)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = img[y0:y1, x0:x1]
    if crop.shape[:2] == (h, w):
        return crop.copy()

    if img.ndim == 2:
        canvas = np.zeros((h, w), dtype=img.dtype)
    else:
        canvas = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
    dst_x0 = max(0, -ox)
    dst_y0 = max(0, -oy)
    canvas[dst_y0:dst_y0 + (y1 - y0), dst_x0:dst_x0 + (x1 - x0)] = crop
    return canvas


def _field_support_mask(mask_shape: Tuple[int, int], state: Optional[AppState]) -> Optional[np.ndarray]:
    pts = getattr(state, "field_boundary", None) if state is not None else None
    if not pts or len(pts) < 3:
        return None
    support = np.zeros(mask_shape[:2], dtype=np.uint8)
    ox = int(getattr(state, "mask_offset_x", 0))
    oy = int(getattr(state, "mask_offset_y", 0))
    poly = np.round(np.asarray(pts, dtype=np.float32) - np.array([ox, oy], dtype=np.float32)).astype(np.int32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return None
    cv2.fillPoly(support, [poly], 255)
    return support if np.count_nonzero(support) >= 16 else None


def _state_field_region(state: Optional[AppState]) -> Optional[Polygon]:
    pts = getattr(state, "field_boundary", None) if state is not None else None
    if not pts or len(pts) < 3:
        return None
    try:
        region = Polygon(pts)
        if state is not None and getattr(state, "img_w", 0) > 0 and getattr(state, "img_h", 0) > 0:
            img_bbox = Polygon([
                (0, 0),
                (state.img_w, 0),
                (state.img_w, state.img_h),
                (0, state.img_h),
            ])
            region = region.intersection(img_bbox)
        if region.is_empty:
            return None
        if region.geom_type == "Polygon":
            return region
        geoms = [g for g in getattr(region, "geoms", []) if g.geom_type == "Polygon"]
        return max(geoms, key=lambda g: g.area) if geoms else None
    except Exception:
        return None


def _constrain_point_to_region(point: Tuple[float, float], region: Optional[Polygon]) -> Tuple[float, float]:
    if region is None:
        return float(point[0]), float(point[1])
    try:
        pt = Point(float(point[0]), float(point[1]))
        if region.covers(pt):
            return float(point[0]), float(point[1])
        boundary = region.boundary
        proj = boundary.interpolate(boundary.project(pt))
        return float(proj.x), float(proj.y)
    except Exception:
        return float(point[0]), float(point[1])


def _constrain_polyline_to_region(points: List[Tuple[float, float]],
                                  region: Optional[Polygon],
                                  max_step_px: float = 12.0) -> List[Tuple[float, float]]:
    if region is None or len(points) < 2:
        return [(float(x), float(y)) for x, y in points]

    constrained: List[Tuple[float, float]] = []
    for idx in range(len(points) - 1):
        p1 = (float(points[idx][0]), float(points[idx][1]))
        p2 = (float(points[idx + 1][0]), float(points[idx + 1][1]))
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(math.ceil(seg_len / max(max_step_px, 1.0))))
        for step in range(steps + 1):
            if idx > 0 and step == 0:
                continue
            t = step / float(steps)
            sample = (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)
            clamped = _constrain_point_to_region(sample, region)
            if constrained and math.hypot(clamped[0] - constrained[-1][0], clamped[1] - constrained[-1][1]) < 1e-3:
                continue
            constrained.append(clamped)
    if len(constrained) < 2:
        return [(float(x), float(y)) for x, y in points]

    simplified = [constrained[0]]
    for idx in range(1, len(constrained) - 1):
        prev = simplified[-1]
        curr = constrained[idx]
        nxt = constrained[idx + 1]
        dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
        denom = dx * dx + dy * dy
        if denom > 1e-6:
            t = ((curr[0] - prev[0]) * dx + (curr[1] - prev[1]) * dy) / denom
            t = max(0.0, min(1.0, t))
            proj = (prev[0] + t * dx, prev[1] + t * dy)
            if math.hypot(curr[0] - proj[0], curr[1] - proj[1]) < 0.75:
                continue
        simplified.append(curr)
    simplified.append(constrained[-1])
    return simplified


def _support_mask_for_orientation(mask: np.ndarray, state: Optional[AppState], mpp: float) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    support = np.zeros_like(binary, dtype=np.uint8)
    field_support = _field_support_mask(mask.shape[:2], state)
    if field_support is not None:
        support = field_support.copy()

    if np.count_nonzero(support) < 128:
        nz = cv2.findNonZero(binary)
        if nz is not None and len(nz) >= 3:
            hull = cv2.convexHull(nz)
            cv2.fillConvexPoly(support, hull, 255)
        else:
            support = binary * 255

    spread_sigma = max(1.0, 0.22 / max(mpp, 1e-6))
    proximity = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=spread_sigma, sigmaY=spread_sigma,
                                 borderType=cv2.BORDER_REPLICATE)
    focused = (support > 0) & (proximity > 0.015)
    if np.count_nonzero(focused) >= max(256, int(np.count_nonzero(binary) * 0.30)):
        return focused.astype(np.uint8) * 255
    return support


def _interp_nan_1d(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    if out.size == 0:
        return out
    idx = np.arange(out.size, dtype=np.float32)
    valid = np.isfinite(out)
    if not np.any(valid):
        return out
    if np.count_nonzero(valid) == 1:
        out[:] = out[valid][0]
        return out
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def _smooth_series_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(1, -1)
    if arr.size == 0 or sigma <= 0.25:
        return arr.reshape(-1).copy()
    return cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma)).reshape(-1)


def _mask_overlap_scores(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    aa = np.asarray(a, dtype=np.uint8) > 0
    bb = np.asarray(b, dtype=np.uint8) > 0
    inter = float(np.count_nonzero(aa & bb))
    area_a = float(np.count_nonzero(aa))
    area_b = float(np.count_nonzero(bb))
    union = float(np.count_nonzero(aa | bb))
    precision = inter / max(area_b, 1.0)
    recall = inter / max(area_a, 1.0)
    iou = inter / max(union, 1.0)
    return precision, recall, iou


def _find_prominent_peaks_1d(values: np.ndarray, threshold: float, min_distance: int) -> List[int]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return []
    peaks: List[int] = []
    min_distance = max(1, int(min_distance))
    for idx in range(1, arr.size - 1):
        val = float(arr[idx])
        if val < threshold:
            continue
        if val >= float(arr[idx - 1]) and val >= float(arr[idx + 1]):
            peaks.append(idx)
    if not peaks:
        return []

    merged: List[int] = [int(peaks[0])]
    for idx in peaks[1:]:
        idx = int(idx)
        if idx - merged[-1] >= min_distance:
            merged.append(idx)
        elif arr[idx] > arr[merged[-1]]:
            merged[-1] = idx
    return merged


def _estimate_lattice_prior(mask: np.ndarray, geo: Optional[GeoUtils] = None,
                            state: Optional[AppState] = None,
                            analysis_mpp: Optional[float] = None,
                            angle_rad: Optional[float] = None) -> Optional[Dict[str, float]]:
    raw = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if np.count_nonzero(raw) < 100:
        return None

    mpp = _mask_meters_per_pixel(raw, geo, state)
    work_mpp = float(max(mpp, analysis_mpp if analysis_mpp is not None else 0.012))
    scale = min(1.0, mpp / max(work_mpp, 1e-6))
    h, w = raw.shape[:2]
    max_dim = 2800.0
    if max(h, w) * scale > max_dim:
        scale = max_dim / float(max(h, w))

    if scale < 0.999:
        sw = max(96, int(round(w * scale)))
        sh = max(96, int(round(h * scale)))
        small = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
    else:
        small = raw.copy()
        sh, sw = h, w

    field_support = _field_support_mask(raw.shape[:2], state)
    if field_support is not None:
        if scale < 0.999:
            support_small = cv2.resize(field_support, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            support_small = field_support.copy()
        support_small = (support_small > 0).astype(np.uint8) * 255
    else:
        support_small = _support_mask_for_orientation(small, state, max(work_mpp, 0.010))

    if np.count_nonzero(support_small) < 100:
        support_small = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    if np.count_nonzero(support_small) < 100:
        return None

    if angle_rad is None:
        field_angle = _field_longest_edge_angle(state)
        fit_angle = _fitline_fallback_angle(raw)
        if field_angle is not None and _half_turn_diff(field_angle, fit_angle) <= math.radians(18.0):
            angle_rad = _half_turn_mean(
                np.array([field_angle, fit_angle], dtype=np.float32),
                np.array([1.0, 1.25], dtype=np.float32),
            )
        else:
            angle_rad = fit_angle if math.isfinite(fit_angle) else (field_angle if field_angle is not None else 0.0)
    angle_rad = _wrap_half_turn(float(angle_rad))

    rot_deg = 90.0 - math.degrees(angle_rad)
    center = (sw * 0.5, sh * 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    rotated = cv2.warpAffine(small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
    rotated_support = cv2.warpAffine(support_small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)

    nz = cv2.findNonZero(rotated_support if np.count_nonzero(rotated_support) >= 64 else rotated)
    if nz is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(nz)
    pad = max(2, int(round(0.05 / max(work_mpp, 1e-6))))
    rx = max(0, rx - pad)
    ry = max(0, ry - pad)
    rw = min(sw - rx, rw + pad * 2)
    rh = min(sh - ry, rh + pad * 2)
    if rw < 16 or rh < 16:
        return None

    roi = rotated[ry:ry + rh, rx:rx + rw]
    support_roi = rotated_support[ry:ry + rh, rx:rx + rw]
    if np.count_nonzero(support_roi) < 100:
        support_roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    if np.count_nonzero(support_roi) < 100:
        return None

    row_support = np.count_nonzero(support_roi, axis=1).astype(np.float32)
    row_fill = np.count_nonzero(roi, axis=1).astype(np.float32) / np.maximum(row_support, 1.0)
    valid_rows = row_support > max(4.0, float(np.percentile(row_support[row_support > 0], 20)) * 0.45) if np.any(row_support > 0) else np.zeros(rh, dtype=bool)
    row_fill_smooth = _smooth_series_1d(row_fill, max(1.0, 0.18 / max(work_mpp, 1e-6)))
    base_margin = max(int(round(rh * 0.10)), int(round(0.55 / max(work_mpp, 1e-6))))
    base_margin = min(max(12, base_margin), max(12, rh // 4))
    fill_ref = row_fill_smooth[valid_rows] if np.any(valid_rows) else row_fill_smooth
    high_fill_thr = max(0.16, float(np.percentile(fill_ref, 82))) if fill_ref.size > 0 else 0.16

    body_y0 = int(base_margin)
    while body_y0 + 4 < rh // 2 and valid_rows[body_y0] and row_fill_smooth[body_y0] >= high_fill_thr:
        body_y0 += 1

    body_y1 = int(rh - base_margin)
    while body_y1 - 5 > rh // 2 and valid_rows[body_y1 - 1] and row_fill_smooth[body_y1 - 1] >= high_fill_thr:
        body_y1 -= 1

    if body_y1 - body_y0 < max(24, int(round(rh * 0.45))):
        body_y0 = int(base_margin)
        body_y1 = int(rh - base_margin)
    if body_y1 - body_y0 < 24:
        return None

    body_roi = roi[body_y0:body_y1]
    body_support = support_roi[body_y0:body_y1]
    body_h, body_w = body_roi.shape[:2]
    if body_h < 16 or body_w < 16 or np.count_nonzero(body_roi) < 100:
        return None

    col_support = np.count_nonzero(body_support, axis=0).astype(np.float32)
    valid_cols = col_support > max(4.0, float(np.percentile(col_support[col_support > 0], 20)) * 0.45) if np.any(col_support > 0) else np.zeros(body_w, dtype=bool)
    if np.count_nonzero(valid_cols) < max(16, int(round(body_w * 0.20))):
        valid_cols = col_support > 0
    if np.count_nonzero(valid_cols) < 16:
        return None

    col_profile = np.count_nonzero(body_roi, axis=0).astype(np.float32) / np.maximum(col_support, 1.0)
    col_profile[~valid_cols] = 0.0
    col_smooth = _smooth_series_1d(col_profile, max(1.0, 0.08 / max(work_mpp, 1e-6)))
    valid_signal = col_smooth[valid_cols]
    if valid_signal.size < 16 or not np.any(valid_signal > 1e-5):
        return None

    signal = col_smooth.copy()
    signal[~valid_cols] = 0.0
    signal[valid_cols] -= float(np.mean(valid_signal))
    window = np.hanning(signal.size).astype(np.float32) if signal.size >= 8 else np.ones_like(signal, dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(signal.size, d=1.0)
    min_pitch_px = max(6.0, 0.18 / max(work_mpp, 1e-6))
    max_pitch_px = min(float(body_w) * 0.55, 2.40 / max(work_mpp, 1e-6))
    pitch_px = 0.0
    peak_conf = 0.0
    if max_pitch_px > min_pitch_px + 2.0:
        valid_freq = (freqs > 0) & (1.0 / np.maximum(freqs, 1e-6) >= min_pitch_px) & (1.0 / np.maximum(freqs, 1e-6) <= max_pitch_px)
        if np.any(valid_freq):
            spec_roi = spectrum[valid_freq]
            freq_roi = freqs[valid_freq]
            peak_idx = int(np.argmax(spec_roi))
            peak_amp = float(spec_roi[peak_idx])
            spec_base = float(np.percentile(spec_roi, 60))
            peak_conf = peak_amp / (spec_base + 1e-6)
            if freq_roi[peak_idx] > 1e-6:
                pitch_px = float(1.0 / freq_roi[peak_idx])

    if not math.isfinite(pitch_px) or pitch_px <= 0.0:
        ac = np.correlate(signal, signal, mode="full")[signal.size - 1:]
        min_lag = max(6, int(round(min_pitch_px)))
        max_lag = min(signal.size // 2, int(round(max_pitch_px)))
        if max_lag > min_lag and ac[0] > 1e-6:
            rel = ac[min_lag:max_lag + 1] / ac[0]
            if rel.size > 0:
                best_idx = int(np.argmax(rel))
                pitch_px = float(min_lag + best_idx)
                peak_conf = max(peak_conf, float(rel[best_idx]) / max(float(np.percentile(rel, 60)), 1e-6))
    if not math.isfinite(pitch_px) or pitch_px < min_pitch_px or pitch_px > max_pitch_px:
        return None

    positive = valid_signal[valid_signal > 1e-4]
    peak_thr = max(0.015, float(np.percentile(positive, 60)) if positive.size > 0 else 0.015)
    actual_peaks = _find_prominent_peaks_1d(col_smooth, peak_thr, max(4, int(round(pitch_px * 0.55))))
    if not actual_peaks:
        actual_peaks = [int(np.argmax(col_smooth))]

    phase_step = max(0.5, pitch_px / 96.0)
    phase_grid = np.arange(0.0, max(pitch_px, phase_step), phase_step, dtype=np.float32)
    x_idx = np.arange(body_w, dtype=np.float32)
    phase_sigma = max(1.0, 0.18 * pitch_px)
    phase_scores = np.zeros(phase_grid.size, dtype=np.float32)
    for idx, phase in enumerate(phase_grid):
        dist = ((x_idx - float(phase) + 0.5 * pitch_px) % pitch_px) - 0.5 * pitch_px
        kernel = np.exp(-0.5 * np.square(dist / phase_sigma)).astype(np.float32)
        phase_scores[idx] = float(np.sum(kernel[valid_cols] * col_smooth[valid_cols]))
    if phase_scores.size == 0:
        return None
    phase_px = float(phase_grid[int(np.argmax(phase_scores))])

    while phase_px - pitch_px >= 0.0:
        phase_px -= pitch_px
    while phase_px + pitch_px < 0.0:
        phase_px += pitch_px
    while phase_px > 0.0:
        phase_px -= pitch_px
    phase_px += math.ceil(abs(phase_px) / max(pitch_px, 1e-6)) * pitch_px if phase_px < 0.0 else 0.0

    lattice_centers = []
    center_x = float(phase_px)
    while center_x - pitch_px >= 0.0:
        center_x -= pitch_px
    while center_x < body_w + pitch_px:
        if 0.0 <= center_x < body_w:
            lattice_centers.append(float(center_x))
        center_x += pitch_px
    if not lattice_centers:
        lattice_centers = [float(np.median(actual_peaks))]
    lattice_centers = np.asarray(lattice_centers, dtype=np.float32)

    diffs = np.diff(np.asarray(actual_peaks, dtype=np.float32))
    spacing_cv = float(np.std(diffs) / max(np.mean(diffs), 1e-6)) if diffs.size > 0 else 1.0

    match_radius = max(4.0, 0.35 * pitch_px)
    unmatched = set(int(p) for p in actual_peaks)
    matched_pred = 0
    residuals: List[float] = []
    for pred in lattice_centers:
        if not unmatched:
            break
        nearest = min(unmatched, key=lambda item: abs(float(item) - float(pred)))
        delta = abs(float(nearest) - float(pred))
        if delta <= match_radius:
            matched_pred += 1
            residuals.append(delta / max(pitch_px, 1e-6))
            unmatched.remove(int(nearest))
    coverage_ratio = matched_pred / max(float(lattice_centers.size), 1.0)
    lattice_residual = float(np.median(residuals)) if residuals else 1.0

    half_window = max(2, int(round(max(3.0, pitch_px * 0.22))))
    continuity_scores: List[float] = []
    for pred in lattice_centers:
        lo = max(0, int(round(float(pred) - half_window)))
        hi = min(body_w, int(round(float(pred) + half_window + 1)))
        if hi - lo < 2:
            continue
        active_rows = np.any(body_roi[:, lo:hi] > 0, axis=1)
        continuity_scores.append(float(np.count_nonzero(active_rows)) / max(float(body_h), 1.0))
    continuity_ratio = float(np.median(np.asarray(continuity_scores, dtype=np.float32))) if continuity_scores else 0.0

    margin_raw = np.zeros_like(roi)
    margin_raw[:body_y0] = roi[:body_y0]
    margin_raw[body_y1:] = roi[body_y1:]
    headland_score = 0.0
    margin_area = float(np.count_nonzero(margin_raw))
    if margin_area > 0.0:
        cc_count, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats((margin_raw > 0).astype(np.uint8), 8)
        ortho_area = 0.0
        for label in range(1, cc_count):
            area = int(cc_stats[label, cv2.CC_STAT_AREA])
            if area < 8:
                continue
            bw_box = int(cc_stats[label, cv2.CC_STAT_WIDTH])
            bh_box = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
            if bw_box >= max(10.0, bh_box * 1.30):
                ortho_area += float(area)
        headland_score = ortho_area / margin_area

    conf_term = float(np.clip((peak_conf - 1.0) / 3.0, 0.0, 1.0))
    spacing_term = 1.0 - float(np.clip(spacing_cv / 0.55, 0.0, 1.0))
    align_term = 1.0 - float(np.clip(lattice_residual / 0.35, 0.0, 1.0))
    coverage_term = float(np.clip(coverage_ratio, 0.0, 1.0))
    continuity_term = float(np.clip(continuity_ratio, 0.0, 1.0))
    headland_term = float(np.clip(headland_score, 0.0, 1.0))
    structure_score = (
        0.22 * conf_term +
        0.22 * spacing_term +
        0.18 * align_term +
        0.22 * continuity_term +
        0.10 * coverage_term +
        0.06 * headland_term
    )

    return {
        "angle_rad": float(angle_rad),
        "pitch_px": float(pitch_px),
        "phase_px": float(phase_px),
        "peak_conf": float(peak_conf),
        "spacing_cv": float(spacing_cv),
        "coverage_ratio": float(coverage_ratio),
        "lattice_residual": float(lattice_residual),
        "continuity_ratio": float(continuity_ratio),
        "headland_score": float(headland_score),
        "structure_score": float(structure_score),
        "body_y0": float(body_y0),
        "body_y1": float(body_y1),
        "stripe_count": float(lattice_centers.size),
    }


def _orientation_histogram(angles: np.ndarray, weights: np.ndarray, bins: int = 180) -> np.ndarray:
    ang = np.asarray(angles, dtype=np.float64).reshape(-1) % math.pi
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(ang) & np.isfinite(w) & (w > 0)
    hist = np.zeros(int(bins), dtype=np.float32)
    if not np.any(valid):
        return hist

    pos = ang[valid] * (float(bins) / math.pi)
    base = np.floor(pos).astype(np.int32)
    frac = pos - base
    np.add.at(hist, base % bins, (w[valid] * (1.0 - frac)).astype(np.float32))
    np.add.at(hist, (base + 1) % bins, (w[valid] * frac).astype(np.float32))

    ext = np.concatenate([hist[-3:], hist, hist[:3]])
    ext = cv2.GaussianBlur(ext.reshape(1, -1), (0, 0), sigmaX=1.2).reshape(-1)
    return ext[3:-3]


def _peak_angle_from_hist(hist: np.ndarray) -> Tuple[Optional[float], float]:
    arr = np.asarray(hist, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.any(arr > 0):
        return None, 0.0
    bins = arr.size
    peak = int(np.argmax(arr))
    peak_val = float(arr[peak])
    base = float(np.percentile(arr, 60))
    conf = peak_val / (base + 1e-6)
    offsets = np.arange(-4, 5, dtype=np.int32)
    idx = (peak + offsets) % bins
    ang = (idx.astype(np.float32) + 0.5) * (math.pi / bins)
    return _half_turn_mean(ang, arr[idx]), conf


def _fft_orientation_hist(texture: np.ndarray, support: np.ndarray, mpp: float, hp_sigma_m: float) -> Tuple[np.ndarray, float]:
    support_bool = np.asarray(support, dtype=np.uint8) > 0
    if np.count_nonzero(support_bool) < 256:
        return np.zeros(180, dtype=np.float32), 0.0

    nz = cv2.findNonZero(support_bool.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(nz)
    pad = max(6, int(round(0.20 / max(mpp, 1e-6))))
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(texture.shape[1] - x, w + pad * 2)
    h = min(texture.shape[0] - y, h + pad * 2)
    if w < 32 or h < 32:
        return np.zeros(180, dtype=np.float32), 0.0

    roi = np.asarray(texture[y:y + h, x:x + w], dtype=np.float32).copy()
    roi_support = support_bool[y:y + h, x:x + w]
    inside = roi[roi_support]
    fill = float(np.median(inside)) if inside.size else 0.0
    roi[~roi_support] = fill

    sigma = max(1.2, hp_sigma_m / max(mpp, 1e-6))
    roi_hp = roi - cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    roi_hp -= float(np.mean(roi_hp[roi_support])) if inside.size else 0.0

    soft_support = cv2.GaussianBlur(roi_support.astype(np.float32), (0, 0),
                                    sigmaX=max(1.0, min(w, h) * 0.03),
                                    sigmaY=max(1.0, min(w, h) * 0.03))
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    weighted = roi_hp * window * np.clip(soft_support, 0.0, 1.0)

    spectrum = np.fft.fftshift(np.fft.fft2(weighted))
    mag = np.log1p(np.abs(spectrum).astype(np.float32))
    yy, xx = np.indices(mag.shape, dtype=np.float32)
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    dx = xx - cx
    dy = yy - cy
    rr = np.hypot(dx, dy)
    min_r = max(4.0, min(w, h) * 0.025)
    max_r = max(min_r + 1.0, min(w, h) * 0.45)
    valid = (rr >= min_r) & (rr <= max_r)
    if not np.any(valid):
        return np.zeros(180, dtype=np.float32), 0.0

    floor = float(np.percentile(mag[valid], 70))
    weights = np.maximum(0.0, mag[valid] - floor) * np.sqrt(rr[valid] + 1e-6)
    if not np.any(weights > 0):
        return np.zeros(180, dtype=np.float32), 0.0

    angles = (np.arctan2(dy[valid], dx[valid]) + math.pi * 0.5) % math.pi
    hist = _orientation_histogram(angles, weights, bins=180)
    _, conf = _peak_angle_from_hist(hist)
    return hist, conf


def _tensor_orientation_hist(texture: np.ndarray, support: np.ndarray, mpp: float, tensor_sigma_m: float) -> Tuple[np.ndarray, float]:
    img = np.asarray(texture, dtype=np.float32)
    support_bool = np.asarray(support, dtype=np.uint8) > 0
    if np.count_nonzero(support_bool) < 256:
        return np.zeros(180, dtype=np.float32), 0.0

    hp_sigma = max(1.0, 0.06 / max(mpp, 1e-6))
    img_hp = img - cv2.GaussianBlur(img, (0, 0), sigmaX=hp_sigma, sigmaY=hp_sigma, borderType=cv2.BORDER_REPLICATE)
    gx = cv2.Sobel(img_hp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_hp, cv2.CV_32F, 0, 1, ksize=3)

    sigma = max(1.0, tensor_sigma_m / max(mpp, 1e-6))
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    orientation = (0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + math.pi * 0.5) % math.pi
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2)) / (jxx + jyy + 1e-6)
    energy = np.sqrt(jxx + jyy)

    inner = cv2.erode(support_bool.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    base_energy = energy[inner] if np.any(inner) else energy[support_bool]
    base_coherence = coherence[inner] if np.any(inner) else coherence[support_bool]
    if base_energy.size == 0:
        return np.zeros(180, dtype=np.float32), 0.0

    energy_thr = float(np.percentile(base_energy, 55))
    coherence_thr = max(0.12, float(np.percentile(base_coherence, 45)))
    valid = inner & np.isfinite(orientation) & (energy >= energy_thr) & (coherence >= coherence_thr)
    if np.count_nonzero(valid) < 256:
        valid = support_bool & np.isfinite(orientation) & (energy >= float(np.percentile(base_energy, 40))) & (coherence >= 0.08)
    if np.count_nonzero(valid) < 64:
        return np.zeros(180, dtype=np.float32), 0.0

    weights = energy[valid] * np.square(np.clip(coherence[valid], 0.0, 1.0))
    hist = _orientation_histogram(orientation[valid], weights, bins=180)
    _, conf = _peak_angle_from_hist(hist)
    return hist, conf


def _fitline_fallback_angle(mask: np.ndarray) -> float:
    return estimate_mask_axis_fitline(mask)


def _estimate_row_angle(mask: np.ndarray, geo: Optional[GeoUtils] = None, state: Optional[AppState] = None,
                        analysis_mpp: Optional[float] = None) -> float:
    raw = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(raw) < 100:
        return 0.0

    mpp = _mask_meters_per_pixel(raw, geo, state)
    orient_mpp = float(max(mpp, analysis_mpp if analysis_mpp is not None else 0.010))
    scale = min(1.0, mpp / orient_mpp)
    h, w = raw.shape[:2]
    max_dim = 3200.0
    if max(h, w) * scale > max_dim:
        scale = max_dim / float(max(h, w))
    sw = max(96, int(round(w * scale)))
    sh = max(96, int(round(h * scale)))
    small_mask = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 0.999 else raw.copy()
    _, small_mask = cv2.threshold(small_mask, 32, 255, cv2.THRESH_BINARY)

    source_img = _crop_source_image(raw.shape[:2], state)
    small_img = None
    if source_img is not None:
        interp = cv2.INTER_AREA if scale < 0.999 else cv2.INTER_LINEAR
        small_img = cv2.resize(source_img, (sw, sh), interpolation=interp) if source_img.shape[:2] != (sh, sw) or scale < 0.999 else source_img.copy()

    support = _support_mask_for_orientation(raw, state, mpp)
    small_support = cv2.resize(support, (sw, sh), interpolation=cv2.INTER_NEAREST) if scale < 0.999 else support.copy()
    if np.count_nonzero(small_support) < 256:
        small_support = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_NEAREST) if scale < 0.999 else raw.copy()

    histograms: List[Tuple[np.ndarray, float]] = []

    if small_img is not None:
        if small_img.ndim == 3 and small_img.shape[2] >= 3:
            b = small_img[:, :, 0].astype(np.float32) / 255.0
            g = small_img[:, :, 1].astype(np.float32) / 255.0
            r = small_img[:, :, 2].astype(np.float32) / 255.0
            texture = 0.7 * (2.0 * g - r - b) + 0.3 * g
        else:
            texture = small_img.astype(np.float32)
            if np.max(texture) > 1.0:
                texture /= 255.0
        for sigma_m in (0.06, 0.12):
            hist, conf = _fft_orientation_hist(texture, small_support, orient_mpp, sigma_m)
            histograms.append((hist, 1.25 * max(0.0, min(conf - 1.0, 8.0))))
        for sigma_m in (0.03, 0.07):
            hist, conf = _tensor_orientation_hist(texture, small_support, orient_mpp, sigma_m)
            histograms.append((hist, 1.60 * max(0.0, min(conf - 1.0, 8.0))))

    mask_texture = cv2.GaussianBlur((small_mask > 0).astype(np.float32), (0, 0), sigmaX=1.1, sigmaY=1.1)
    for sigma_m in (0.06, 0.12):
        hist, conf = _fft_orientation_hist(mask_texture, small_support, orient_mpp, sigma_m)
        histograms.append((hist, 0.85 * max(0.0, min(conf - 1.0, 8.0))))
    for sigma_m in (0.03, 0.07):
        hist, conf = _tensor_orientation_hist(mask_texture, small_support, orient_mpp, sigma_m)
        histograms.append((hist, 1.00 * max(0.0, min(conf - 1.0, 8.0))))

    combined = np.zeros(180, dtype=np.float32)
    for hist, weight in histograms:
        if weight <= 1e-3:
            continue
        peak, _ = _peak_angle_from_hist(hist)
        if peak is None:
            continue
        norm = hist / (float(np.max(hist)) + 1e-6)
        combined += float(weight) * norm

    peak_angle, peak_conf = _peak_angle_from_hist(combined)
    if peak_angle is not None and peak_conf >= 1.15:
        return _wrap_half_turn(peak_angle)

    field_angle = _field_longest_edge_angle(state)
    if field_angle is not None and peak_angle is not None and _half_turn_diff(field_angle, peak_angle) <= math.radians(12.0):
        return _half_turn_mean(np.array([field_angle, peak_angle], dtype=np.float32),
                               np.array([1.0, max(1.0, peak_conf)], dtype=np.float32))
    if peak_angle is not None:
        return _wrap_half_turn(peak_angle)
    if field_angle is not None:
        return field_angle
    return _fitline_fallback_angle(raw)


def _fit_whole_mask_morphology_model(mask: np.ndarray, geo: Optional[GeoUtils] = None,
                                     state: Optional[AppState] = None,
                                     analysis_mpp_floor: float = 0.012):
    raw = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if np.count_nonzero(raw) < 100:
        return None

    def _true_runs(flags: np.ndarray) -> List[Tuple[int, int]]:
        arr = np.asarray(flags, dtype=np.uint8)
        edges = np.diff(np.concatenate(([0], arr, [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def _fill_short_false_runs(flags: np.ndarray, max_gap: int) -> np.ndarray:
        out = np.asarray(flags, dtype=bool).copy()
        if max_gap <= 0 or out.size == 0:
            return out
        for start, end in _true_runs(~out):
            if start == 0 or end == out.size:
                continue
            if end - start <= max_gap:
                out[start:end] = True
        return out

    def _drop_short_true_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
        out = np.asarray(flags, dtype=bool).copy()
        if min_len <= 1 or out.size == 0:
            return out
        for start, end in _true_runs(out):
            if end - start < min_len:
                out[start:end] = False
        return out

    def _merge_close_runs(runs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
        if not runs:
            return []
        merged = [runs[0]]
        for start, end in runs[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    def _kmeans_width_centers(values: np.ndarray) -> Optional[Tuple[float, float]]:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        vals = vals[np.isfinite(vals) & (vals > 1.0)]
        if vals.size < 6:
            return None
        try:
            _, _, centers = cv2.kmeans(
                vals.reshape(-1, 1), 2, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1),
                10, cv2.KMEANS_PP_CENTERS,
            )
        except Exception:
            return None
        centers = np.sort(centers.reshape(-1).astype(np.float64))
        if centers.size != 2 or centers[0] <= 1.0:
            return None
        if centers[1] / centers[0] < 1.10:
            return None
        return float(centers[0]), float(centers[1])

    field_support = _field_support_mask(raw.shape[:2], state)
    if field_support is not None:
        raw = cv2.bitwise_and(raw, field_support)
        if np.count_nonzero(raw) < 100:
            return None

    mpp = _mask_meters_per_pixel(raw, geo, state)
    analysis_mpp = max(mpp, analysis_mpp_floor)
    scale = min(1.0, mpp / analysis_mpp)
    h, w = raw.shape[:2]
    max_dim = 3200.0
    if max(h, w) * scale > max_dim:
        scale = max_dim / float(max(h, w))

    if scale < 0.999:
        sw = max(64, int(round(w * scale)))
        sh = max(64, int(round(h * scale)))
        small = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
        if field_support is not None:
            support_small = cv2.resize(field_support, (sw, sh), interpolation=cv2.INTER_NEAREST)
            support_small = (support_small > 0).astype(np.uint8) * 255
        else:
            support_small = None
    else:
        small = raw.copy()
        sh, sw = h, w
        support_small = field_support.copy() if field_support is not None else None

    angle_rad = _estimate_row_angle(raw, geo, state, analysis_mpp=max(mpp, 0.010))
    rot_deg = 90.0 - math.degrees(angle_rad)
    center = (sw * 0.5, sh * 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    inv_rot_mat = cv2.invertAffineTransform(rot_mat)
    rotated = cv2.warpAffine(small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
    rotated_support = None
    if support_small is not None:
        rotated_support = cv2.warpAffine(support_small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)

    nz = cv2.findNonZero(rotated_support if rotated_support is not None and np.count_nonzero(rotated_support) >= 64 else rotated)
    if nz is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(nz)
    pad = max(2, int(round(0.05 / max(analysis_mpp, 1e-6))))
    rx = max(0, rx - pad)
    ry = max(0, ry - pad)
    rw = min(sw - rx, rw + pad * 2)
    rh = min(sh - ry, rh + pad * 2)
    roi = rotated[ry:ry + rh, rx:rx + rw]
    if roi.size == 0 or rw < 24 or rh < 24:
        return None

    if rotated_support is not None:
        support_roi = rotated_support[ry:ry + rh, rx:rx + rw]
    else:
        support_roi = cv2.dilate(
            roi,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(5, int(round(0.18 / max(analysis_mpp, 1e-6)))) | 1,
                 max(5, int(round(0.18 / max(analysis_mpp, 1e-6)))) | 1),
            ),
            iterations=2,
        )

    lattice_prior = _estimate_lattice_prior(raw, geo, state, analysis_mpp=analysis_mpp, angle_rad=angle_rad)
    if lattice_prior is None:
        return None
    pitch_px = float(lattice_prior.get("pitch_px", 0.0))
    phase_px = float(lattice_prior.get("phase_px", 0.0))
    lattice_conf = float(lattice_prior.get("peak_conf", 0.0))
    if not math.isfinite(pitch_px) or pitch_px < 6.0 or lattice_conf < 1.02:
        return None

    body_y0 = int(round(lattice_prior.get("body_y0", 0.0)))
    body_y1 = int(round(lattice_prior.get("body_y1", float(rh))))
    if body_y1 - body_y0 < max(24, int(round(rh * 0.45))):
        margin = max(int(round(rh * 0.10)), int(round(0.55 / max(analysis_mpp, 1e-6))))
        margin = min(max(12, margin), max(12, rh // 4))
        body_y0 = margin
        body_y1 = rh - margin
    if body_y1 - body_y0 < 24:
        return None

    body_roi = roi[body_y0:body_y1]
    body_support = support_roi[body_y0:body_y1]
    if body_roi.size == 0 or np.count_nonzero(body_roi) < 100:
        return None
    body_h, body_w = body_roi.shape[:2]

    lattice_centers = []
    seed = float(phase_px)
    while seed - pitch_px >= 0.0:
        seed -= pitch_px
    while seed < body_w + pitch_px:
        if 0.0 <= seed < body_w:
            lattice_centers.append(float(seed))
        seed += pitch_px
    if len(lattice_centers) < 8:
        return None
    lattice_centers = np.asarray(lattice_centers, dtype=np.float32)

    min_seg_width_px = max(2, int(round(0.05 / max(analysis_mpp, 1e-6))))
    merge_seg_gap_px = max(1, int(round(0.03 / max(analysis_mpp, 1e-6))))
    shift_sigma_y = max(1.0, 0.24 / max(analysis_mpp, 1e-6))
    row_left_bound = np.full(body_h, np.nan, dtype=np.float32)
    row_right_bound = np.full(body_h, np.nan, dtype=np.float32)
    shift_obs = np.full(body_h, np.nan, dtype=np.float32)
    row_runs_cache: List[List[Tuple[int, int]]] = []
    width_samples: List[List[float]] = [[] for _ in range(lattice_centers.size)]
    match_radius = max(6.0, 0.45 * pitch_px)

    for yy in range(body_h):
        support_runs = _true_runs(body_support[yy] > 0)
        if support_runs:
            row_left_bound[yy] = float(min(run[0] for run in support_runs))
            row_right_bound[yy] = float(max(run[1] for run in support_runs) - 1)

        flags = body_roi[yy] > 0
        flags = _fill_short_false_runs(flags, merge_seg_gap_px)
        flags = _drop_short_true_runs(flags, min_seg_width_px)
        runs = _merge_close_runs(_true_runs(flags), merge_seg_gap_px)
        row_runs_cache.append(runs)
        if not runs:
            continue

        centers = np.asarray([0.5 * (start + end - 1) for start, end in runs], dtype=np.float32)
        widths = np.asarray([end - start for start, end in runs], dtype=np.float32)
        if centers.size == 0 or widths.size == 0:
            continue
        deltas = ((centers - phase_px + 0.5 * pitch_px) % pitch_px) - 0.5 * pitch_px
        valid = np.isfinite(deltas) & np.isfinite(widths) & (widths > 0)
        if np.count_nonzero(valid) < 1:
            continue
        deltas = np.clip(deltas[valid], -0.45 * pitch_px, 0.45 * pitch_px)
        weights = widths[valid]
        shift_obs[yy] = float(np.average(deltas, weights=weights))

    valid_bounds = np.isfinite(row_left_bound) & np.isfinite(row_right_bound) & (row_right_bound > row_left_bound)
    if np.count_nonzero(valid_bounds) < max(8, int(round(body_h * 0.20))):
        return None
    row_left_bound = _interp_nan_1d(row_left_bound)
    row_right_bound = _interp_nan_1d(row_right_bound)

    if np.count_nonzero(np.isfinite(shift_obs)) < max(8, int(round(body_h * 0.12))):
        return None
    shift_trace = _interp_nan_1d(shift_obs)
    shift_trace = _smooth_series_1d(shift_trace, shift_sigma_y)
    shift_trace = np.clip(shift_trace, -0.40 * pitch_px, 0.40 * pitch_px)

    for yy, runs in enumerate(row_runs_cache):
        if not runs:
            continue
        shift_row = float(shift_trace[yy])
        run_centers = np.asarray([0.5 * (start + end - 1) for start, end in runs], dtype=np.float32)
        run_widths = np.asarray([end - start for start, end in runs], dtype=np.float32)
        for idx, base_center in enumerate(lattice_centers):
            target = float(base_center + shift_row)
            rel = np.abs(run_centers - target)
            if rel.size == 0:
                continue
            best = int(np.argmin(rel))
            if float(rel[best]) > match_radius:
                continue
            width_samples[idx].append(float(run_widths[best]))

    obs_widths = np.asarray(
        [float(np.median(samples)) for samples in width_samples if len(samples) >= 3],
        dtype=np.float32,
    )
    if obs_widths.size == 0:
        return None

    width_nom = np.full(lattice_centers.size, float(np.median(obs_widths)), dtype=np.float32)
    width_classes = _kmeans_width_centers(obs_widths)
    stripe_obs = np.full(lattice_centers.size, np.nan, dtype=np.float32)
    for idx, samples in enumerate(width_samples):
        if len(samples) >= 3:
            stripe_obs[idx] = float(np.median(np.asarray(samples, dtype=np.float32)))

    if width_classes is not None:
        narrow_w, wide_w = width_classes
        losses = []
        for parity in (0, 1):
            loss = 0.0
            for idx, obs in enumerate(stripe_obs):
                if not math.isfinite(float(obs)):
                    continue
                target = wide_w if ((idx + parity) % 2 == 0) else narrow_w
                loss += abs(float(obs) - target)
            losses.append(loss)
        start_parity = 0 if losses[0] <= losses[1] else 1
        for idx, obs in enumerate(stripe_obs):
            target = wide_w if ((idx + start_parity) % 2 == 0) else narrow_w
            width_nom[idx] = float(0.74 * target + 0.26 * obs) if math.isfinite(float(obs)) else float(target)
    else:
        fallback_w = float(np.median(obs_widths))
        for idx, obs in enumerate(stripe_obs):
            width_nom[idx] = float(obs) if math.isfinite(float(obs)) else fallback_w

    valid_row_count = max(1.0, float(np.count_nonzero(valid_bounds)))
    area_scale = float(np.clip(
        float(np.count_nonzero(body_roi)) / max(float(np.sum(width_nom)) * valid_row_count, 1.0),
        1.00, 1.60,
    ))
    width_nom *= area_scale

    headland_raw = np.zeros_like(roi)
    headland_raw[:body_y0] = roi[:body_y0]
    headland_raw[body_y1:] = roi[body_y1:]

    gap_guard = max(2.0, 0.08 * pitch_px)
    min_band_width = max(2.0, 0.05 / max(analysis_mpp, 1e-6))
    body_rebuilt = np.zeros_like(body_roi)
    for yy in range(body_h):
        left_bound = float(row_left_bound[yy])
        right_bound = float(row_right_bound[yy])
        if not math.isfinite(left_bound) or not math.isfinite(right_bound) or right_bound - left_bound < 4.0:
            continue
        centers_row = lattice_centers + float(shift_trace[yy])
        for idx, center_x in enumerate(centers_row):
            prev_boundary = left_bound if idx == 0 else 0.5 * (float(centers_row[idx - 1]) + float(center_x)) + gap_guard
            next_boundary = right_bound if idx + 1 == centers_row.size else 0.5 * (float(center_x) + float(centers_row[idx + 1])) - gap_guard
            if next_boundary <= prev_boundary + 1.0:
                continue
            width_eff = min(float(width_nom[idx]), next_boundary - prev_boundary + 1.0)
            if width_eff < min_band_width:
                continue
            left = max(prev_boundary, float(center_x) - 0.5 * (width_eff - 1.0))
            right = min(next_boundary, float(center_x) + 0.5 * (width_eff - 1.0))
            if right - left + 1.0 < min_band_width:
                continue
            lx = max(int(round(left)), int(math.floor(left_bound)))
            rx2 = min(int(round(right)), int(math.ceil(right_bound)))
            if rx2 >= lx:
                body_rebuilt[yy, lx:rx2 + 1] = 255

    if np.count_nonzero(body_rebuilt) < 100:
        return None

    capture_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(round(0.08 / max(analysis_mpp, 1e-6)))), max(3, int(round(0.16 / max(analysis_mpp, 1e-6))))),
    )
    capture_roi = cv2.dilate(body_rebuilt, capture_kernel, iterations=1)
    body_rebuilt = cv2.bitwise_or(body_rebuilt, cv2.bitwise_and(body_roi, capture_roi))

    final_roi = np.zeros_like(roi)
    final_roi[body_y0:body_y1] = body_rebuilt
    final_roi = cv2.bitwise_or(final_roi, headland_raw)
    final_roi = cv2.bitwise_and(final_roi, support_roi)

    clean_small = np.zeros_like(rotated)
    clean_small[ry:ry + rh, rx:rx + rw] = final_roi
    restored_small = cv2.warpAffine(clean_small, inv_rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
    if scale < 0.999:
        restored = cv2.resize(restored_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        restored = restored_small
    _, restored = cv2.threshold(restored, 127, 255, cv2.THRESH_BINARY)
    full_capture_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(round(0.14 / max(mpp, 1e-6)))), max(3, int(round(0.22 / max(mpp, 1e-6))))),
    )
    full_capture = cv2.dilate(restored, full_capture_kernel, iterations=1)
    restored = cv2.bitwise_or(restored, cv2.bitwise_and(raw, full_capture))
    if field_support is not None:
        restored = cv2.bitwise_and(restored, field_support)

    return {
        "mask": restored,
        "angle_rad": angle_rad,
        "analysis_mpp": analysis_mpp,
        "stripe_count": int(lattice_centers.size),
        "pitch_px": float(pitch_px),
        "lattice_conf": float(lattice_conf),
    }


def _fit_global_band_model(mask: np.ndarray, geo: Optional[GeoUtils] = None,
                           state: Optional[AppState] = None,
                           analysis_mpp_floor: float = 0.012):
    raw = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(raw) < 100:
        return None

    def _true_runs(flags: np.ndarray) -> List[Tuple[int, int]]:
        arr = np.asarray(flags, dtype=np.uint8)
        edges = np.diff(np.concatenate(([0], arr, [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def _fill_short_false_runs(flags: np.ndarray, max_gap: int) -> np.ndarray:
        out = np.asarray(flags, dtype=bool).copy()
        if max_gap <= 0 or out.size == 0:
            return out
        for start, end in _true_runs(~out):
            if start == 0 or end == out.size:
                continue
            if end - start <= max_gap:
                out[start:end] = True
        return out

    def _drop_short_true_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
        out = np.asarray(flags, dtype=bool).copy()
        if min_len <= 1 or out.size == 0:
            return out
        for start, end in _true_runs(out):
            if end - start < min_len:
                out[start:end] = False
        return out

    def _regularize_centers(local_centers: List[int], start: int, end: int,
                            profile: np.ndarray, spacing_px: float) -> List[int]:
        if not local_centers:
            return []
        local = sorted(int(c) for c in local_centers)
        min_dist = max(3, int(round(spacing_px * 0.55)))
        merged = []
        for c in local:
            c = max(start, min(end - 1, c))
            if not merged or c - merged[-1] >= min_dist:
                merged.append(c)
            elif profile[c] > profile[merged[-1]]:
                merged[-1] = c
        local = merged
        if spacing_px <= 0 or not local:
            return local

        filled = [local[0]]
        for prev, curr in zip(local, local[1:]):
            gap = curr - prev
            if gap > 1.65 * spacing_px:
                extra = max(0, int(round(gap / spacing_px)) - 1)
                for step_idx in range(extra):
                    target = prev + (step_idx + 1) * spacing_px
                    lo = max(start, int(round(target - 0.25 * spacing_px)))
                    hi = min(end, int(round(target + 0.25 * spacing_px)) + 1)
                    snapped = lo + int(np.argmax(profile[lo:hi])) if hi > lo else int(round(target))
                    if snapped - filled[-1] >= max(3, int(round(spacing_px * 0.45))):
                        filled.append(snapped)
            filled.append(curr)
        local = filled

        guard = 0
        while local and local[0] - start > 1.15 * spacing_px and guard < 32:
            guard += 1
            target = local[0] - spacing_px
            lo = max(start, int(round(target - 0.25 * spacing_px)))
            hi = min(end, int(round(target + 0.25 * spacing_px)) + 1)
            snapped = lo + int(np.argmax(profile[lo:hi])) if hi > lo else int(round(target))
            if snapped >= local[0] - 2:
                snapped = int(round(local[0] - spacing_px))
            local.insert(0, max(start, snapped))

        guard = 0
        while local and end - local[-1] > 1.15 * spacing_px and guard < 32:
            guard += 1
            target = local[-1] + spacing_px
            lo = max(start, int(round(target - 0.25 * spacing_px)))
            hi = min(end, int(round(target + 0.25 * spacing_px)) + 1)
            snapped = lo + int(np.argmax(profile[lo:hi])) if hi > lo else int(round(target))
            if snapped <= local[-1] + 2:
                snapped = int(round(local[-1] + spacing_px))
            local.append(min(end - 1, snapped))

        merged = []
        for c in sorted(local):
            c = max(start, min(end - 1, c))
            if not merged or c - merged[-1] >= min_dist:
                merged.append(c)
            elif profile[c] > profile[merged[-1]]:
                merged[-1] = c
        return merged

    mpp = _mask_meters_per_pixel(raw, geo, state)
    analysis_mpp = max(mpp, analysis_mpp_floor)
    scale = min(1.0, mpp / analysis_mpp)
    h, w = raw.shape[:2]
    max_dim = 3200.0
    if max(h, w) * scale > max_dim:
        scale = max_dim / float(max(h, w))

    if scale < 0.999:
        sw = max(64, int(round(w * scale)))
        sh = max(64, int(round(h * scale)))
        small = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
    else:
        small = raw.copy()
        sh, sw = h, w

    angle_rad = _estimate_row_angle(raw, geo, state, analysis_mpp=max(mpp, 0.010))
    rot_deg = 90.0 - math.degrees(angle_rad)
    center = (sw * 0.5, sh * 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    inv_rot_mat = cv2.invertAffineTransform(rot_mat)
    rotated = cv2.warpAffine(small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)

    nz = cv2.findNonZero(rotated)
    if nz is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(nz)
    pad = max(2, int(round(0.05 / max(analysis_mpp, 1e-6))))
    rx = max(0, rx - pad)
    ry = max(0, ry - pad)
    rw = min(sw - rx, rw + pad * 2)
    rh = min(sh - ry, rh + pad * 2)
    roi = rotated[ry:ry + rh, rx:rx + rw]
    if roi.size == 0 or rw < 8 or rh < 8:
        return None

    col_profile = np.count_nonzero(roi, axis=0).astype(np.float32) / max(float(rh), 1.0)
    if not np.any(col_profile > 0):
        return None
    smooth_sigma = max(1.0, 0.08 / max(analysis_mpp, 1e-6))
    col_smooth = cv2.GaussianBlur(col_profile.reshape(1, -1), (0, 0),
                                  sigmaX=smooth_sigma).reshape(-1)
    positive = col_smooth[col_smooth > 1e-4]
    if positive.size == 0:
        return None

    support_thr = max(0.015, float(np.percentile(positive, 35)) * 0.55)
    active = col_smooth >= support_thr
    active = _fill_short_false_runs(active, max(1, int(round(0.18 / max(analysis_mpp, 1e-6)))))
    active = _drop_short_true_runs(active, max(4, int(round(0.16 / max(analysis_mpp, 1e-6)))))
    active_runs = _true_runs(active)
    if not active_runs:
        return None

    spacing_px = 0.0
    signal = col_smooth.copy()
    signal[~active] = 0.0
    if np.any(active):
        signal[active] -= float(np.mean(signal[active]))
    if np.count_nonzero(active) >= 16:
        ac = np.correlate(signal, signal, mode="full")[signal.size - 1:]
        min_lag = max(6, int(round(0.25 / max(analysis_mpp, 1e-6))))
        max_lag = min(signal.size // 2, int(round(3.0 / max(analysis_mpp, 1e-6))))
        if max_lag > min_lag and ac[0] > 1e-6:
            rel = ac[min_lag:max_lag + 1] / ac[0]
            best_idx = int(np.argmax(rel))
            if rel[best_idx] > 0.08:
                spacing_px = float(min_lag + best_idx)

    peak_thr = max(support_thr * 1.35, float(np.percentile(col_smooth[active], 60)))
    candidate_peaks = []
    for xx in range(1, rw - 1):
        if not active[xx] or col_smooth[xx] < peak_thr:
            continue
        if col_smooth[xx] >= col_smooth[xx - 1] and col_smooth[xx] >= col_smooth[xx + 1]:
            candidate_peaks.append(xx)
    if not candidate_peaks:
        for start, end in active_runs:
            seg = col_smooth[start:end]
            if seg.size > 0:
                candidate_peaks.append(start + int(np.argmax(seg)))

    models = []
    total_area = float(np.count_nonzero(roi))
    max_gap_rows = max(4, int(round(0.30 / max(analysis_mpp, 1e-6))))
    min_track_len_px = max(6, int(round(0.25 / max(analysis_mpp, 1e-6))))
    search_pad_px = max(1, int(round(0.03 / max(analysis_mpp, 1e-6))))
    min_band_area_px = max(20, int(round(0.05 / max(analysis_mpp * analysis_mpp, 1e-6))))

    for run_idx, (start, end) in enumerate(active_runs):
        local_peaks = [p for p in candidate_peaks if start <= p < end]
        run_width = end - start
        if spacing_px > 0.0 and run_width > spacing_px * 0.9:
            if not local_peaks:
                count = max(1, int(round(run_width / spacing_px)))
                if count == 1:
                    local_peaks = [start + run_width // 2]
                else:
                    step = run_width / float(count)
                    local_peaks = [int(round(start + step * (i + 0.5))) for i in range(count)]
            local_peaks = _regularize_centers(local_peaks, start, end, col_smooth, spacing_px)
        elif not local_peaks:
            local_peaks = [start + run_width // 2]

        if not local_peaks:
            continue
        local_peaks = sorted(max(start, min(end - 1, int(p))) for p in local_peaks)
        boundaries = [start]
        for left_center, right_center in zip(local_peaks, local_peaks[1:]):
            lo = max(left_center, start)
            hi = min(right_center + 1, end)
            valley = lo + int(np.argmin(col_smooth[lo:hi])) if hi > lo else int(round((left_center + right_center) * 0.5))
            valley = max(lo + 1, min(hi - 1, valley)) if hi - lo > 1 else valley
            boundaries.append(valley)
        boundaries.append(end)

        for peak_idx, peak_x in enumerate(local_peaks):
            left_bound = int(boundaries[peak_idx])
            right_bound = int(boundaries[peak_idx + 1])
            if right_bound - left_bound < 2:
                continue
            search_l = max(0, left_bound - search_pad_px)
            search_r = min(rw, right_bound + search_pad_px)
            observed = np.zeros(rh, dtype=bool)
            row_left = np.full(rh, np.nan, dtype=np.float32)
            row_right = np.full(rh, np.nan, dtype=np.float32)
            row_center = np.full(rh, np.nan, dtype=np.float32)

            for yy in range(rh):
                runs = _true_runs(roi[yy, search_l:search_r] > 0)
                if not runs:
                    continue
                best = None
                best_cost = None
                for rs, re in runs:
                    left = search_l + rs
                    right = search_l + re - 1
                    center_x = 0.5 * (left + right)
                    width_px = right - left + 1.0
                    cost = abs(center_x - float(peak_x)) - 0.08 * width_px
                    if best is None or cost < best_cost:
                        best = (left, right, center_x)
                        best_cost = cost
                if best is None:
                    continue
                left, right, center_x = best
                allow_shift = max((right_bound - left_bound) * 0.85, (0.45 * spacing_px if spacing_px > 0 else 6.0))
                if abs(center_x - float(peak_x)) > allow_shift:
                    continue
                observed[yy] = True
                row_left[yy] = float(left)
                row_right[yy] = float(right)
                row_center[yy] = float(center_x)

            if np.count_nonzero(observed) < 3:
                continue
            present = _fill_short_false_runs(observed, max_gap_rows)
            present = _drop_short_true_runs(present, min_track_len_px)
            present_runs = _true_runs(present)
            if not present_runs:
                continue
            seg_start, seg_end = max(present_runs, key=lambda item: item[1] - item[0])
            observed_main = observed[seg_start:seg_end]
            if np.count_nonzero(observed_main) < 3:
                continue
            widths = (row_right - row_left + 1.0)[seg_start:seg_end][observed_main]
            centers = row_center[seg_start:seg_end][observed_main]
            if widths.size == 0 or centers.size == 0:
                continue
            width_obs = float(np.median(widths))
            center_obs = float(np.median(centers))
            if not (math.isfinite(width_obs) and math.isfinite(center_obs)):
                continue
            length_px = seg_end - seg_start
            if width_obs * max(1.0, float(length_px)) < min_band_area_px:
                continue
            models.append({
                "run": run_idx,
                "peak": float(peak_x),
                "center": float(np.clip(center_obs, left_bound, max(left_bound, right_bound - 1))),
                "width_obs": max(2.0, width_obs),
                "left_limit": float(left_bound),
                "right_limit": float(max(left_bound, right_bound - 1)),
                "y0": int(seg_start),
                "y1": int(seg_end - 1),
                "row_left": row_left[seg_start:seg_end].copy(),
                "row_right": row_right[seg_start:seg_end].copy(),
                "row_center": row_center[seg_start:seg_end].copy(),
                "observed": observed[seg_start:seg_end].copy(),
            })

    if not models:
        return None

    models.sort(key=lambda item: (item["run"], item["center"]))
    approx_area = sum(max(1.0, item["width_obs"]) * max(1, item["y1"] - item["y0"] + 1) for item in models)
    area_scale = float(np.clip(total_area / max(approx_area, 1.0), 0.92, 1.06))
    final_models = []
    trace_sigma = max(1.0, 0.08 / max(analysis_mpp, 1e-6))
    sample_step = max(1, int(round(0.02 / max(analysis_mpp, 1e-6))))
    for idx, item in enumerate(models):
        center_x = float(item["center"])
        left_limit = float(item["left_limit"])
        right_limit = float(item["right_limit"])
        prev_center = None
        next_center = None
        if idx > 0 and models[idx - 1]["run"] == item["run"]:
            prev_center = float(models[idx - 1]["center"])
        if idx + 1 < len(models) and models[idx + 1]["run"] == item["run"]:
            next_center = float(models[idx + 1]["center"])
        left_cap = center_x - left_limit
        right_cap = right_limit - center_x
        if prev_center is not None:
            left_cap = min(left_cap, center_x - 0.5 * (prev_center + center_x))
        if next_center is not None:
            right_cap = min(right_cap, 0.5 * (center_x + next_center) - center_x)
        width_max = max(2.0, 2.0 * max(0.5, min(left_cap, right_cap)) + 1.0)
        width_min = max(2.0, item["width_obs"] * 0.70)
        width_px = float(np.clip(item["width_obs"] * area_scale, width_min, max(width_min, width_max)))
        center_x = float(np.clip(center_x, left_limit, right_limit))
        left_floor = left_limit
        right_ceil = right_limit
        if prev_center is not None:
            left_floor = max(left_floor, 0.5 * (prev_center + center_x))
        if next_center is not None:
            right_ceil = min(right_ceil, 0.5 * (next_center + center_x))
        if right_ceil - left_floor < 1.0:
            continue

        obs_left = np.asarray(item["row_left"], dtype=np.float32)
        obs_right = np.asarray(item["row_right"], dtype=np.float32)
        obs_center = np.asarray(item["row_center"], dtype=np.float32)
        observed = np.asarray(item["observed"], dtype=bool)
        row_count = obs_center.size
        if row_count < 3:
            continue

        ys = np.arange(item["y0"], item["y1"] + 1, dtype=np.float32)
        obs_width = obs_right - obs_left + 1.0
        center_valid = observed & np.isfinite(obs_center)
        width_valid = observed & np.isfinite(obs_width)

        if np.count_nonzero(center_valid) >= 2:
            fit_y = ys[center_valid].astype(np.float64)
            fit_center = obs_center[center_valid].astype(np.float64)
            fit_w = np.clip(obs_width[center_valid].astype(np.float64), 1.0, None)
            coeff = np.polyfit(fit_y, fit_center, 1, w=fit_w)
            pred = np.polyval(coeff, fit_y)
            resid = fit_center - pred
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med)))
            tol = max(2.5, 2.5 * 1.4826 * mad)
            inlier = np.abs(resid - med) <= tol
            if np.count_nonzero(inlier) >= 2:
                coeff = np.polyfit(fit_y[inlier], fit_center[inlier], 1, w=fit_w[inlier])
            center_trace = np.polyval(coeff, ys.astype(np.float64)).astype(np.float32)
        else:
            center_trace = np.full(row_count, center_x, dtype=np.float32)

        if np.count_nonzero(width_valid) >= 3:
            width_base = float(np.median(obs_width[width_valid])) * area_scale
        else:
            width_base = width_px
        width_trace = np.full(row_count, float(np.clip(width_base, width_min, max(width_min, width_max))), dtype=np.float32)
        center_trace = _smooth_series_1d(center_trace, trace_sigma)
        center_trace = np.clip(center_trace, left_floor, right_ceil)
        left_trace = center_trace - 0.5 * (width_trace - 1.0)
        right_trace = center_trace + 0.5 * (width_trace - 1.0)
        left_trace = np.clip(left_trace, left_floor, right_ceil - 1.0)
        right_trace = np.clip(right_trace, left_trace + 1.0, right_ceil)

        band_area = float(np.sum(np.maximum(0.0, right_trace - left_trace + 1.0)))
        if band_area < min_band_area_px:
            continue

        sample_idx = np.arange(0, row_count, sample_step, dtype=np.int32)
        if sample_idx.size == 0 or sample_idx[-1] != row_count - 1:
            sample_idx = np.append(sample_idx, row_count - 1)
        left_pts = np.column_stack([left_trace[sample_idx], ys[sample_idx]])
        right_pts = np.column_stack([right_trace[sample_idx[::-1]], ys[sample_idx[::-1]]])
        poly = np.concatenate([left_pts, right_pts], axis=0).astype(np.float32)
        final_models.append({
            "run": item["run"],
            "center": center_x,
            "poly": poly,
        })

    if not final_models:
        return None

    rebuilt_roi = np.zeros_like(roi)
    band_infos = []
    for item in final_models:
        poly = np.asarray(item["poly"], dtype=np.float32)
        if poly.shape[0] < 4:
            continue
        cv2.fillPoly(rebuilt_roi, [np.round(poly).astype(np.int32)], 255)

        rot_poly = poly + np.array([rx, ry], dtype=np.float32)
        orig_small = cv2.transform(rot_poly.reshape(1, -1, 2), inv_rot_mat).reshape(-1, 2)
        orig_pts = (orig_small / scale) if scale < 0.999 else orig_small
        orig_pts[:, 0] = np.clip(orig_pts[:, 0], 0.0, float(w - 1))
        orig_pts[:, 1] = np.clip(orig_pts[:, 1], 0.0, float(h - 1))
        contour = np.round(orig_pts).astype(np.int32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(contour)
        (ccx, ccy), (bw, bh), _ = rect
        bx, by, bw_box, bh_box = cv2.boundingRect(contour)
        band_infos.append({
            "contour": contour,
            "centroid": (float(ccx), float(ccy)),
            "length": float(max(bw, bh)),
            "width": float(min(bw, bh)),
            "bbox": (int(bx), int(by), int(bw_box), int(bh_box)),
        })

    clean_small = np.zeros_like(rotated)
    clean_small[ry:ry + rh, rx:rx + rw] = rebuilt_roi
    restored_small = cv2.warpAffine(clean_small, inv_rot_mat, (sw, sh),
                                    flags=cv2.INTER_NEAREST, borderValue=0)
    if scale < 0.999:
        restored = cv2.resize(restored_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        restored = restored_small
    _, restored = cv2.threshold(restored, 127, 255, cv2.THRESH_BINARY)

    return {
        "mask": restored,
        "bands": band_infos,
        "angle_rad": angle_rad,
        "analysis_mpp": analysis_mpp,
    }


def _fit_parallel_stripe_model(mask: np.ndarray, geo: Optional[GeoUtils] = None,
                               state: Optional[AppState] = None,
                               analysis_mpp_floor: float = 0.012):
    raw = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(raw) < 100:
        return None

    def _true_runs(flags: np.ndarray) -> List[Tuple[int, int]]:
        arr = np.asarray(flags, dtype=np.uint8)
        edges = np.diff(np.concatenate(([0], arr, [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def _fill_short_false_runs(flags: np.ndarray, max_gap: int) -> np.ndarray:
        out = np.asarray(flags, dtype=bool).copy()
        if max_gap <= 0 or out.size == 0:
            return out
        for start, end in _true_runs(~out):
            if start == 0 or end == out.size:
                continue
            if end - start <= max_gap:
                out[start:end] = True
        return out

    def _drop_short_true_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
        out = np.asarray(flags, dtype=bool).copy()
        if min_len <= 1 or out.size == 0:
            return out
        for start, end in _true_runs(out):
            if end - start < min_len:
                out[start:end] = False
        return out

    def _merge_close_runs(runs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
        if not runs:
            return []
        merged = [runs[0]]
        for start, end in runs[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    def _regularize_centers(local_centers: List[int], start: int, end: int,
                            profile: np.ndarray, spacing_px: float) -> List[int]:
        if not local_centers:
            return []
        local = sorted(int(c) for c in local_centers)
        min_dist = max(3, int(round(spacing_px * 0.50))) if spacing_px > 0 else 3
        merged = []
        for c in local:
            c = max(start, min(end - 1, c))
            if not merged or c - merged[-1] >= min_dist:
                merged.append(c)
            elif profile[c] > profile[merged[-1]]:
                merged[-1] = c
        local = merged
        if spacing_px <= 0.0 or not local:
            return local

        filled = [local[0]]
        for prev, curr in zip(local, local[1:]):
            gap = curr - prev
            if gap > 1.60 * spacing_px:
                extra = max(0, int(round(gap / spacing_px)) - 1)
                for step_idx in range(extra):
                    target = prev + (step_idx + 1) * spacing_px
                    lo = max(start, int(round(target - 0.28 * spacing_px)))
                    hi = min(end, int(round(target + 0.28 * spacing_px)) + 1)
                    snapped = lo + int(np.argmax(profile[lo:hi])) if hi > lo else int(round(target))
                    if snapped - filled[-1] >= max(3, int(round(spacing_px * 0.42))):
                        filled.append(snapped)
            filled.append(curr)
        local = filled

        guard = 0
        while local and local[0] - start > 1.15 * spacing_px and guard < 32:
            guard += 1
            target = local[0] - spacing_px
            lo = max(start, int(round(target - 0.28 * spacing_px)))
            hi = min(end, int(round(target + 0.28 * spacing_px)) + 1)
            snapped = lo + int(np.argmax(profile[lo:hi])) if hi > lo else int(round(target))
            if snapped >= local[0] - 2:
                snapped = int(round(local[0] - spacing_px))
            local.insert(0, max(start, snapped))

        guard = 0
        while local and end - local[-1] > 1.15 * spacing_px and guard < 32:
            guard += 1
            target = local[-1] + spacing_px
            lo = max(start, int(round(target - 0.28 * spacing_px)))
            hi = min(end, int(round(target + 0.28 * spacing_px)) + 1)
            snapped = lo + int(np.argmax(profile[lo:hi])) if hi > lo else int(round(target))
            if snapped <= local[-1] + 2:
                snapped = int(round(local[-1] + spacing_px))
            local.append(min(end - 1, snapped))

        merged = []
        for c in sorted(local):
            c = max(start, min(end - 1, c))
            if not merged or c - merged[-1] >= min_dist:
                merged.append(c)
            elif profile[c] > profile[merged[-1]]:
                merged[-1] = c
        return merged

    def _kmeans_width_centers(values: np.ndarray) -> Optional[Tuple[float, float]]:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        vals = vals[np.isfinite(vals) & (vals > 1.0)]
        if vals.size < 6:
            return None
        try:
            _, _, centers = cv2.kmeans(
                vals.reshape(-1, 1), 2, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1),
                10, cv2.KMEANS_PP_CENTERS,
            )
        except Exception:
            return None
        centers = np.sort(centers.reshape(-1).astype(np.float64))
        if centers.size != 2 or centers[0] <= 1.0:
            return None
        if centers[1] / centers[0] < 1.12:
            return None
        return float(centers[0]), float(centers[1])

    field_support = _field_support_mask(raw.shape[:2], state)
    if field_support is not None:
        raw = cv2.bitwise_and(raw, field_support)
        if np.count_nonzero(raw) < 100:
            return None

    mpp = _mask_meters_per_pixel(raw, geo, state)
    analysis_mpp = max(mpp, analysis_mpp_floor)
    scale = min(1.0, mpp / analysis_mpp)
    h, w = raw.shape[:2]
    max_dim = 3200.0
    if max(h, w) * scale > max_dim:
        scale = max_dim / float(max(h, w))

    if scale < 0.999:
        sw = max(64, int(round(w * scale)))
        sh = max(64, int(round(h * scale)))
        small = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
        if field_support is not None:
            support_small = cv2.resize(field_support, (sw, sh), interpolation=cv2.INTER_AREA)
            _, support_small = cv2.threshold(support_small, 127, 255, cv2.THRESH_BINARY)
        else:
            support_small = None
    else:
        small = raw.copy()
        sh, sw = h, w
        support_small = field_support.copy() if field_support is not None else None

    angle_rad = _estimate_row_angle(raw, geo, state, analysis_mpp=max(mpp, 0.010))
    rot_deg = 90.0 - math.degrees(angle_rad)
    center = (sw * 0.5, sh * 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    inv_rot_mat = cv2.invertAffineTransform(rot_mat)
    rotated = cv2.warpAffine(small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
    rotated_support = None
    if support_small is not None:
        rotated_support = cv2.warpAffine(support_small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)

    nz = cv2.findNonZero(rotated)
    if nz is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(nz)
    pad = max(2, int(round(0.06 / max(analysis_mpp, 1e-6))))
    rx = max(0, rx - pad)
    ry = max(0, ry - pad)
    rw = min(sw - rx, rw + pad * 2)
    rh = min(sh - ry, rh + pad * 2)
    roi = rotated[ry:ry + rh, rx:rx + rw]
    if roi.size == 0 or rw < 32 or rh < 32:
        return None

    if rotated_support is not None:
        support_roi = rotated_support[ry:ry + rh, rx:rx + rw]
    else:
        proxy_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(5, int(round(0.18 / max(analysis_mpp, 1e-6)))) | 1,
             max(5, int(round(0.18 / max(analysis_mpp, 1e-6)))) | 1),
        )
        support_roi = cv2.dilate(roi, proxy_kernel, iterations=2)

    lattice_prior = _estimate_lattice_prior(raw, geo, state, analysis_mpp=analysis_mpp, angle_rad=angle_rad)
    prior_pitch_px = float(lattice_prior["pitch_px"]) if lattice_prior is not None else 0.0
    prior_phase_px = float(lattice_prior["phase_px"]) if lattice_prior is not None else 0.0
    prior_peak_conf = float(lattice_prior["peak_conf"]) if lattice_prior is not None else 0.0

    headland_margin = max(int(round(rh * 0.10)), int(round(0.60 / max(analysis_mpp, 1e-6))))
    headland_margin = min(max(16, headland_margin), max(16, rh // 4))
    if rh <= headland_margin * 2 + 12:
        return None
    body_y0 = headland_margin
    body_y1 = rh - headland_margin
    body_roi = roi[body_y0:body_y1]
    body_support = support_roi[body_y0:body_y1]
    if body_roi.size == 0 or np.count_nonzero(body_roi) < 100:
        return None

    body_h, body_w = body_roi.shape[:2]
    col_profile = np.count_nonzero(body_roi, axis=0).astype(np.float32) / max(float(body_h), 1.0)
    smooth_sigma = max(1.0, 0.08 / max(analysis_mpp, 1e-6))
    col_smooth = cv2.GaussianBlur(col_profile.reshape(1, -1), (0, 0), sigmaX=smooth_sigma).reshape(-1)
    positive = col_smooth[col_smooth > 1e-4]
    if positive.size < 16:
        return None

    pitch_px = 0.0
    signal = col_smooth.copy()
    signal -= float(np.mean(signal))
    if signal.size >= 24:
        ac = np.correlate(signal, signal, mode="full")[signal.size - 1:]
        min_lag = max(6, int(round(0.22 / max(analysis_mpp, 1e-6))))
        max_lag = min(signal.size // 2, int(round(2.2 / max(analysis_mpp, 1e-6))))
        if max_lag > min_lag and ac[0] > 1e-6:
            rel = ac[min_lag:max_lag + 1] / ac[0]
            best_idx = int(np.argmax(rel))
            if rel[best_idx] > 0.03:
                pitch_px = float(min_lag + best_idx)
    if prior_pitch_px > 0.0:
        if pitch_px <= 0.0 or prior_peak_conf >= 1.14:
            pitch_px = prior_pitch_px
        else:
            pitch_px = 0.65 * pitch_px + 0.35 * prior_pitch_px

    peak_thr = max(float(np.percentile(positive, 60)), float(np.percentile(positive, 35)) * 1.08)
    candidates = []
    for xx in range(1, body_w - 1):
        if col_smooth[xx] < peak_thr:
            continue
        if col_smooth[xx] >= col_smooth[xx - 1] and col_smooth[xx] >= col_smooth[xx + 1]:
            candidates.append(xx)
    if prior_pitch_px > 0.0 and prior_peak_conf >= 1.02:
        lattice_seed = float(prior_phase_px)
        while lattice_seed - prior_pitch_px >= 0.0:
            lattice_seed -= prior_pitch_px
        while lattice_seed < body_w + prior_pitch_px:
            if 0.0 <= lattice_seed < body_w:
                candidates.append(int(round(lattice_seed)))
            lattice_seed += prior_pitch_px
    if not candidates:
        candidates = [int(np.argmax(col_smooth))]

    merge_dist = max(4, int(round((pitch_px if pitch_px > 0 else max(24.0, body_w / 48.0)) * 0.35)))
    merged = []
    for xx in candidates:
        if not merged or xx - merged[-1] >= merge_dist:
            merged.append(xx)
        elif col_smooth[xx] > col_smooth[merged[-1]]:
            merged[-1] = xx
    candidates = merged

    if pitch_px > 0.0:
        candidates = _regularize_centers(candidates, 0, body_w, col_smooth, pitch_px)
    if len(candidates) < 8:
        return None

    if len(candidates) > 1:
        diffs = np.diff(np.asarray(candidates, dtype=np.float32))
        valid_diffs = diffs[(diffs >= max(4.0, merge_dist * 0.75)) & (diffs <= float(np.percentile(diffs, 90)) * 1.35)]
        if valid_diffs.size > 0:
            pitch_px = float(np.median(valid_diffs))
        elif pitch_px <= 0.0:
            pitch_px = float(np.median(diffs))
    if prior_pitch_px > 0.0 and math.isfinite(pitch_px):
        pitch_px = 0.72 * pitch_px + 0.28 * prior_pitch_px
    if not math.isfinite(pitch_px) or pitch_px < max(6.0, merge_dist * 1.1):
        return None

    lattice_centers = np.asarray(candidates, dtype=np.float32)
    diffs = np.diff(lattice_centers)
    pitch_cv = float(np.std(diffs) / max(np.mean(diffs), 1e-6)) if diffs.size > 0 else 1.0
    if lattice_centers.size < 12 or pitch_cv > 0.70:
        return None

    headland_raw = np.zeros_like(roi)
    margin_raw = np.zeros_like(roi)
    margin_raw[:body_y0] = roi[:body_y0]
    margin_raw[body_y1:] = roi[body_y1:]
    cc_count, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats((margin_raw > 0).astype(np.uint8), 8)
    headland_min_area = max(20, int(round(0.10 / max(analysis_mpp * analysis_mpp, 1e-6))))
    headland_min_span = max(24.0, 1.8 * pitch_px)
    for label in range(1, cc_count):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area < headland_min_area:
            continue
        bx = int(cc_stats[label, cv2.CC_STAT_LEFT])
        by = int(cc_stats[label, cv2.CC_STAT_TOP])
        bw_box = int(cc_stats[label, cv2.CC_STAT_WIDTH])
        bh_box = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
        touches_top = by < body_y0
        touches_bottom = by + bh_box > body_y1
        if not (touches_top or touches_bottom):
            continue
        if bw_box >= max(headland_min_span, bh_box * 1.35):
            headland_raw[cc_labels == label] = 255

    min_seg_width_px = max(2, int(round(0.05 / max(analysis_mpp, 1e-6))))
    merge_seg_gap_px = max(1, int(round(0.03 / max(analysis_mpp, 1e-6))))
    max_gap_rows = max(4, int(round(0.30 / max(analysis_mpp, 1e-6))))
    min_track_len_px = max(6, int(round(0.22 / max(analysis_mpp, 1e-6))))
    base_shift_px = max(3.0, 0.08 / max(analysis_mpp, 1e-6))

    row_segments: List[List[dict]] = []
    for yy in range(body_h):
        flags = body_roi[yy] > 0
        flags = _fill_short_false_runs(flags, merge_seg_gap_px)
        flags = _drop_short_true_runs(flags, min_seg_width_px)
        runs = _merge_close_runs(_true_runs(flags), merge_seg_gap_px)
        segs = []
        for start, end in runs:
            width = end - start
            if width < min_seg_width_px:
                continue
            left = float(start)
            right = float(end - 1)
            segs.append({
                "left": left,
                "right": right,
                "center": 0.5 * (left + right),
                "width": float(width),
            })
        row_segments.append(segs)

    def _new_track(seg: dict, row: int) -> dict:
        return {
            "rows": [row],
            "lefts": [float(seg["left"])],
            "rights": [float(seg["right"])],
            "last_row": int(row),
            "last_left": float(seg["left"]),
            "last_right": float(seg["right"]),
            "last_center": float(seg["center"]),
            "last_width": float(seg["width"]),
        }

    tracks: List[dict] = []
    active_tracks: List[dict] = []
    for yy, segs in enumerate(row_segments):
        survivors = []
        for track in active_tracks:
            if yy - track["last_row"] <= max_gap_rows + 1:
                survivors.append(track)
            else:
                tracks.append(track)
        active_tracks = survivors

        candidates_cost = []
        for track_idx, track in enumerate(active_tracks):
            row_gap = max(1, yy - track["last_row"])
            allow_shift = max(base_shift_px, track["last_width"] * 0.42) + row_gap * max(1.0, base_shift_px * 0.22)
            for seg_idx, seg in enumerate(segs):
                shift = abs(seg["center"] - track["last_center"])
                if shift > allow_shift:
                    continue
                overlap = max(
                    0.0,
                    min(track["last_right"], seg["right"]) - max(track["last_left"], seg["left"]) + 1.0,
                )
                width_jump = abs(seg["width"] - track["last_width"])
                cost = shift + 0.25 * width_jump - 0.60 * overlap + 0.18 * row_gap
                candidates_cost.append((cost, track_idx, seg_idx))

        matched_tracks = set()
        matched_segs = set()
        for _, track_idx, seg_idx in sorted(candidates_cost, key=lambda item: item[0]):
            if track_idx in matched_tracks or seg_idx in matched_segs:
                continue
            track = active_tracks[track_idx]
            seg = segs[seg_idx]
            track["rows"].append(int(yy))
            track["lefts"].append(float(seg["left"]))
            track["rights"].append(float(seg["right"]))
            track["last_row"] = int(yy)
            track["last_left"] = float(seg["left"])
            track["last_right"] = float(seg["right"])
            track["last_center"] = float(seg["center"])
            track["last_width"] = float(seg["width"])
            matched_tracks.add(track_idx)
            matched_segs.add(seg_idx)

        for seg_idx, seg in enumerate(segs):
            if seg_idx not in matched_segs:
                active_tracks.append(_new_track(seg, yy))

    tracks.extend(active_tracks)
    if not tracks:
        return None

    track_models = []
    for track in tracks:
        if len(track["rows"]) < 4:
            continue
        y0 = int(track["rows"][0])
        y1 = int(track["rows"][-1])
        span = y1 - y0 + 1
        if span < min_track_len_px:
            continue
        left = np.full(span, np.nan, dtype=np.float32)
        right = np.full(span, np.nan, dtype=np.float32)
        observed = np.zeros(span, dtype=bool)
        for row, lx, rx2 in zip(track["rows"], track["lefts"], track["rights"]):
            idx = int(row) - y0
            left[idx] = float(lx)
            right[idx] = float(rx2)
            observed[idx] = True
        if np.count_nonzero(observed) < 4:
            continue
        left = _interp_nan_1d(left)
        right = _interp_nan_1d(right)
        centers = 0.5 * (left + right)
        widths = np.maximum(1.0, right - left + 1.0)
        center_obs = centers[observed]
        width_obs = widths[observed]
        if center_obs.size < 4 or width_obs.size < 4:
            continue
        track_models.append({
            "y0": y0,
            "y1": y1,
            "span": span,
            "observed": observed,
            "center_trace": centers,
            "width_trace": widths,
            "base_center": float(np.median(center_obs)),
            "width_med": float(np.median(width_obs)),
            "obs_count": int(np.count_nonzero(observed)),
        })

    if len(track_models) < max(8, int(round(lattice_centers.size * 0.35))):
        return None

    track_models.sort(key=lambda item: item["base_center"])
    assignments = {idx: [] for idx in range(lattice_centers.size)}
    match_radius = max(6.0, pitch_px * 0.45)
    prev_idx = -1
    for track in track_models:
        lo = prev_idx + 1
        if lo >= lattice_centers.size:
            break
        rel = np.abs(lattice_centers[lo:] - float(track["base_center"]))
        if rel.size == 0:
            break
        best_rel = int(np.argmin(rel))
        idx = lo + best_rel
        if float(rel[best_rel]) > match_radius:
            continue
        assignments[idx].append(track)
        prev_idx = idx

    stripes = []
    for idx, base_center in enumerate(lattice_centers):
        group = assignments.get(idx, [])
        group = sorted(group, key=lambda item: (item["obs_count"], item["span"]), reverse=True)
        rep = group[0] if group else None
        width_obs = float(np.median([g["width_med"] for g in group])) if group else float("nan")
        stripes.append({
            "idx": idx,
            "base_center": float(base_center),
            "track": rep,
            "width_obs": width_obs,
        })

    obs_widths = np.asarray([item["width_obs"] for item in stripes if math.isfinite(item["width_obs"])], dtype=np.float32)
    width_classes = _kmeans_width_centers(obs_widths)
    if width_classes is not None:
        narrow_w, wide_w = width_classes
        losses = []
        for parity in (0, 1):
            loss = 0.0
            for item in stripes:
                if not math.isfinite(item["width_obs"]):
                    continue
                target = wide_w if ((item["idx"] + parity) % 2 == 0) else narrow_w
                loss += abs(float(item["width_obs"]) - target)
            losses.append(loss)
        start_parity = 0 if losses[0] <= losses[1] else 1
        for item in stripes:
            target = wide_w if ((item["idx"] + start_parity) % 2 == 0) else narrow_w
            if math.isfinite(item["width_obs"]):
                item["width_nom"] = 0.72 * target + 0.28 * float(item["width_obs"])
            else:
                item["width_nom"] = float(target)
    else:
        fallback_w = float(np.median(obs_widths)) if obs_widths.size > 0 else max(2.0, pitch_px * 0.55)
        for item in stripes:
            item["width_nom"] = float(item["width_obs"]) if math.isfinite(item["width_obs"]) else fallback_w

    shift_buckets: List[List[float]] = [[] for _ in range(body_h)]
    for item in stripes:
        track = item["track"]
        if track is None:
            continue
        span = int(track["span"])
        observed = np.asarray(track["observed"], dtype=bool)
        centers = np.asarray(track["center_trace"], dtype=np.float32)
        for local_idx in np.where(observed)[0]:
            yy = int(track["y0"]) + int(local_idx)
            shift_buckets[yy].append(float(centers[local_idx] - item["base_center"]))

    shift_obs = np.full(body_h, np.nan, dtype=np.float32)
    for yy, values in enumerate(shift_buckets):
        if values:
            shift_obs[yy] = float(np.median(np.asarray(values, dtype=np.float32)))
    if np.count_nonzero(np.isfinite(shift_obs)) < max(8, int(round(body_h * 0.18))):
        return None
    shift_idx = np.arange(body_h, dtype=np.float32)
    valid_shift = np.isfinite(shift_obs)
    if np.count_nonzero(valid_shift) >= 8:
        fit_y = shift_idx[valid_shift].astype(np.float64)
        fit_shift = shift_obs[valid_shift].astype(np.float64)
        deg = 3 if fit_y.size >= 32 else (2 if fit_y.size >= 12 else 1)
        coeff = np.polyfit(fit_y, fit_shift, deg)
        pred = np.polyval(coeff, fit_y)
        resid = fit_shift - pred
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        tol = max(2.0, 2.8 * 1.4826 * mad)
        inlier = np.abs(resid - med) <= tol
        if np.count_nonzero(inlier) >= max(6, deg + 1):
            coeff = np.polyfit(fit_y[inlier], fit_shift[inlier], deg)
        shift_trace = np.polyval(coeff, shift_idx.astype(np.float64)).astype(np.float32)
    else:
        shift_trace = _interp_nan_1d(shift_obs)
    shift_trace = _smooth_series_1d(shift_trace, max(1.0, 0.24 / max(analysis_mpp, 1e-6)))
    shift_trace = np.clip(shift_trace, -0.65 * pitch_px, 0.65 * pitch_px)

    for item in stripes:
        item["residual"] = np.zeros(body_h, dtype=np.float32)

    row_left_bound = np.full(body_h, np.nan, dtype=np.float32)
    row_right_bound = np.full(body_h, np.nan, dtype=np.float32)
    for yy in range(body_h):
        runs = _true_runs(body_support[yy] > 0)
        if not runs:
            continue
        start = min(run[0] for run in runs)
        end = max(run[1] for run in runs)
        row_left_bound[yy] = float(start)
        row_right_bound[yy] = float(end - 1)
    valid_bounds = np.isfinite(row_left_bound) & np.isfinite(row_right_bound) & (row_right_bound > row_left_bound)
    if np.count_nonzero(valid_bounds) < max(8, int(round(body_h * 0.25))):
        return None
    row_left_bound = _interp_nan_1d(row_left_bound)
    row_right_bound = _interp_nan_1d(row_right_bound)

    center_matrix = np.vstack([
        np.full(body_h, float(item["base_center"]), dtype=np.float32) + shift_trace + np.asarray(item["residual"], dtype=np.float32)
        for item in stripes
    ])
    gap_guard = max(2.0, 0.08 * pitch_px)
    min_band_width = max(2.0, 0.05 / max(analysis_mpp, 1e-6))
    body_rebuilt = np.zeros_like(body_roi)
    for yy in range(body_h):
        left_bound = float(row_left_bound[yy])
        right_bound = float(row_right_bound[yy])
        if not (math.isfinite(left_bound) and math.isfinite(right_bound) and right_bound - left_bound >= 4.0):
            continue
        centers_row = center_matrix[:, yy].astype(np.float32)
        for idx, item in enumerate(stripes):
            center_x = float(centers_row[idx])
            prev_boundary = left_bound if idx == 0 else 0.5 * (float(centers_row[idx - 1]) + center_x) + gap_guard
            next_boundary = right_bound if idx + 1 == len(stripes) else 0.5 * (center_x + float(centers_row[idx + 1])) - gap_guard
            if next_boundary <= prev_boundary + 1.0:
                continue
            width_nom = float(item["width_nom"])
            max_width = max(min_band_width, next_boundary - prev_boundary + 1.0)
            width_eff = min(width_nom, max_width)
            if width_eff < min_band_width:
                continue
            left = max(prev_boundary, center_x - 0.5 * (width_eff - 1.0))
            right = min(next_boundary, center_x + 0.5 * (width_eff - 1.0))
            if right - left + 1.0 < min_band_width:
                continue
            lx = max(int(round(left)), int(math.floor(left_bound)))
            rx2 = min(int(round(right)), int(math.ceil(right_bound)))
            if rx2 >= lx:
                body_rebuilt[yy, lx:rx2 + 1] = 255

    if np.count_nonzero(body_rebuilt) < 100:
        return None

    capture_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(round(0.08 / max(analysis_mpp, 1e-6)))), max(3, int(round(0.16 / max(analysis_mpp, 1e-6))))),
    )
    capture_roi = cv2.dilate(body_rebuilt, capture_kernel, iterations=1)
    body_rebuilt = cv2.bitwise_or(body_rebuilt, cv2.bitwise_and(body_roi, capture_roi))

    final_roi = np.zeros_like(roi)
    final_roi[body_y0:body_y1] = body_rebuilt
    final_roi = cv2.bitwise_or(final_roi, headland_raw)
    final_roi = cv2.bitwise_and(final_roi, support_roi)

    clean_small = np.zeros_like(rotated)
    clean_small[ry:ry + rh, rx:rx + rw] = final_roi
    restored_small = cv2.warpAffine(clean_small, inv_rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
    if scale < 0.999:
        restored = cv2.resize(restored_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        restored = restored_small
    _, restored = cv2.threshold(restored, 127, 255, cv2.THRESH_BINARY)
    if field_support is not None:
        restored = cv2.bitwise_and(restored, field_support)

    return {
        "mask": restored,
        "angle_rad": angle_rad,
        "analysis_mpp": analysis_mpp,
        "stripe_count": int(lattice_centers.size),
        "pitch_px": float(pitch_px),
        "pitch_cv": float(pitch_cv),
        "lattice_conf": float(prior_peak_conf),
    }


def _fit_evidence_strip_model(mask: np.ndarray, geo: Optional[GeoUtils] = None,
                              state: Optional[AppState] = None,
                              analysis_mpp_floor: float = 0.012):
    raw = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(raw) < 100:
        return None

    def _true_runs(flags: np.ndarray) -> List[Tuple[int, int]]:
        arr = np.asarray(flags, dtype=np.uint8)
        edges = np.diff(np.concatenate(([0], arr, [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def _fill_short_false_runs(flags: np.ndarray, max_gap: int) -> np.ndarray:
        out = np.asarray(flags, dtype=bool).copy()
        if max_gap <= 0 or out.size == 0:
            return out
        for start, end in _true_runs(~out):
            if start == 0 or end == out.size:
                continue
            if end - start <= max_gap:
                out[start:end] = True
        return out

    def _drop_short_true_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
        out = np.asarray(flags, dtype=bool).copy()
        if min_len <= 1 or out.size == 0:
            return out
        for start, end in _true_runs(out):
            if end - start < min_len:
                out[start:end] = False
        return out

    def _merge_close_runs(runs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
        if not runs:
            return []
        merged = [runs[0]]
        for start, end in runs[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    def _robust_poly_trace(values: np.ndarray, observed: np.ndarray, sigma: float, deg_hi: int = 2) -> np.ndarray:
        trace = _interp_nan_1d(values)
        obs_idx = np.where(observed)[0]
        if obs_idx.size < 3:
            return _smooth_series_1d(trace, sigma)
        fit_y = obs_idx.astype(np.float64)
        fit_v = values[observed].astype(np.float64)
        deg = min(deg_hi, max(1, fit_y.size - 1))
        if fit_y.size >= deg + 1:
            coeff = np.polyfit(fit_y, fit_v, deg)
            pred = np.polyval(coeff, fit_y)
            resid = fit_v - pred
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med)))
            tol = max(1.5, 2.8 * 1.4826 * mad)
            inlier = np.abs(resid - med) <= tol
            if np.count_nonzero(inlier) >= deg + 1:
                coeff = np.polyfit(fit_y[inlier], fit_v[inlier], deg)
            poly = np.polyval(coeff, np.arange(values.size, dtype=np.float64)).astype(np.float32)
            trace = 0.70 * poly + 0.30 * trace
        return _smooth_series_1d(trace, sigma)

    field_support = _field_support_mask(raw.shape[:2], state)
    if field_support is not None:
        raw = cv2.bitwise_and(raw, field_support)
        if np.count_nonzero(raw) < 100:
            return None

    mpp = _mask_meters_per_pixel(raw, geo, state)
    analysis_mpp = max(mpp, analysis_mpp_floor)
    scale = min(1.0, mpp / analysis_mpp)
    h, w = raw.shape[:2]
    max_dim = 3200.0
    if max(h, w) * scale > max_dim:
        scale = max_dim / float(max(h, w))

    if scale < 0.999:
        sw = max(64, int(round(w * scale)))
        sh = max(64, int(round(h * scale)))
        small = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
        if field_support is not None:
            support_small = cv2.resize(field_support, (sw, sh), interpolation=cv2.INTER_AREA)
            _, support_small = cv2.threshold(support_small, 127, 255, cv2.THRESH_BINARY)
        else:
            support_small = None
    else:
        small = raw.copy()
        sh, sw = h, w
        support_small = field_support.copy() if field_support is not None else None

    angle_rad = _estimate_row_angle(raw, geo, state, analysis_mpp=max(mpp, 0.010))
    rot_deg = 90.0 - math.degrees(angle_rad)
    center = (sw * 0.5, sh * 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    inv_rot_mat = cv2.invertAffineTransform(rot_mat)
    rotated = cv2.warpAffine(small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
    rotated_support = None
    if support_small is not None:
        rotated_support = cv2.warpAffine(support_small, rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)

    nz = cv2.findNonZero(rotated)
    if nz is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(nz)
    pad = max(2, int(round(0.06 / max(analysis_mpp, 1e-6))))
    rx = max(0, rx - pad)
    ry = max(0, ry - pad)
    rw = min(sw - rx, rw + pad * 2)
    rh = min(sh - ry, rh + pad * 2)
    roi = rotated[ry:ry + rh, rx:rx + rw]
    if roi.size == 0 or rw < 16 or rh < 16:
        return None

    if rotated_support is not None:
        support_roi = rotated_support[ry:ry + rh, rx:rx + rw]
    else:
        proxy_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(5, int(round(0.18 / max(analysis_mpp, 1e-6)))) | 1,
             max(5, int(round(0.18 / max(analysis_mpp, 1e-6)))) | 1),
        )
        support_roi = cv2.dilate(roi, proxy_kernel, iterations=2)

    lattice_prior = _estimate_lattice_prior(raw, geo, state, analysis_mpp=analysis_mpp, angle_rad=angle_rad)
    prior_pitch_px = float(lattice_prior["pitch_px"]) if lattice_prior is not None else 0.0
    prior_peak_conf = float(lattice_prior["peak_conf"]) if lattice_prior is not None else 0.0

    headland_margin = max(int(round(rh * 0.10)), int(round(0.55 / max(analysis_mpp, 1e-6))))
    headland_margin = min(max(12, headland_margin), max(12, rh // 4))
    if rh <= headland_margin * 2 + 12:
        return None
    body_y0 = headland_margin
    body_y1 = rh - headland_margin
    body_roi = roi[body_y0:body_y1]
    body_support = support_roi[body_y0:body_y1]
    if body_roi.size == 0 or np.count_nonzero(body_roi) < 100:
        return None

    body_h, body_w = body_roi.shape[:2]
    row_left_bound = np.full(body_h, np.nan, dtype=np.float32)
    row_right_bound = np.full(body_h, np.nan, dtype=np.float32)
    for yy in range(body_h):
        runs = _true_runs(body_support[yy] > 0)
        if not runs:
            continue
        row_left_bound[yy] = float(min(run[0] for run in runs))
        row_right_bound[yy] = float(max(run[1] for run in runs) - 1)
    valid_bounds = np.isfinite(row_left_bound) & np.isfinite(row_right_bound) & (row_right_bound > row_left_bound)
    if np.count_nonzero(valid_bounds) < max(8, int(round(body_h * 0.20))):
        return None
    row_left_bound = _interp_nan_1d(row_left_bound)
    row_right_bound = _interp_nan_1d(row_right_bound)

    min_seg_width_px = max(2, int(round(0.05 / max(analysis_mpp, 1e-6))))
    merge_seg_gap_px = max(1, int(round(0.03 / max(analysis_mpp, 1e-6))))
    max_gap_rows = max(4, int(round(0.28 / max(analysis_mpp, 1e-6))))
    min_track_len_px = max(6, int(round(0.20 / max(analysis_mpp, 1e-6))))
    track_sigma_y = max(1.0, 0.12 / max(analysis_mpp, 1e-6))
    boundary_guard_px = max(1.0, 0.02 / max(analysis_mpp, 1e-6))
    min_track_area_px = max(18, int(round(0.05 / max(analysis_mpp * analysis_mpp, 1e-6))))
    base_shift_px = max(3.0, 0.07 / max(analysis_mpp, 1e-6))

    row_segments: List[List[dict]] = []
    for yy in range(body_h):
        flags = body_roi[yy] > 0
        flags = _fill_short_false_runs(flags, merge_seg_gap_px)
        flags = _drop_short_true_runs(flags, min_seg_width_px)
        runs = _merge_close_runs(_true_runs(flags), merge_seg_gap_px)
        segs = []
        for start, end in runs:
            width = end - start
            if width < min_seg_width_px:
                continue
            left = float(start)
            right = float(end - 1)
            segs.append({
                "left": left,
                "right": right,
                "center": 0.5 * (left + right),
                "width": float(width),
            })
        row_segments.append(segs)

    def _new_track(seg: dict, row: int) -> dict:
        return {
            "rows": [row],
            "lefts": [float(seg["left"])],
            "rights": [float(seg["right"])],
            "last_row": int(row),
            "last_left": float(seg["left"]),
            "last_right": float(seg["right"]),
            "last_center": float(seg["center"]),
            "last_width": float(seg["width"]),
        }

    tracks: List[dict] = []
    active_tracks: List[dict] = []
    for yy, segs in enumerate(row_segments):
        survivors = []
        for track in active_tracks:
            if yy - track["last_row"] <= max_gap_rows + 1:
                survivors.append(track)
            else:
                tracks.append(track)
        active_tracks = survivors

        candidates = []
        for track_idx, track in enumerate(active_tracks):
            row_gap = max(1, yy - track["last_row"])
            allow_shift = max(base_shift_px, track["last_width"] * 0.42) + row_gap * max(1.0, base_shift_px * 0.20)
            for seg_idx, seg in enumerate(segs):
                shift = abs(seg["center"] - track["last_center"])
                if shift > allow_shift:
                    continue
                overlap = max(0.0, min(track["last_right"], seg["right"]) - max(track["last_left"], seg["left"]) + 1.0)
                width_jump = abs(seg["width"] - track["last_width"])
                cost = shift + 0.25 * width_jump - 0.60 * overlap + 0.20 * row_gap
                candidates.append((cost, track_idx, seg_idx))

        matched_tracks = set()
        matched_segs = set()
        for _, track_idx, seg_idx in sorted(candidates, key=lambda item: item[0]):
            if track_idx in matched_tracks or seg_idx in matched_segs:
                continue
            track = active_tracks[track_idx]
            seg = segs[seg_idx]
            track["rows"].append(int(yy))
            track["lefts"].append(float(seg["left"]))
            track["rights"].append(float(seg["right"]))
            track["last_row"] = int(yy)
            track["last_left"] = float(seg["left"])
            track["last_right"] = float(seg["right"])
            track["last_center"] = float(seg["center"])
            track["last_width"] = float(seg["width"])
            matched_tracks.add(track_idx)
            matched_segs.add(seg_idx)

        for seg_idx, seg in enumerate(segs):
            if seg_idx not in matched_segs:
                active_tracks.append(_new_track(seg, yy))

    tracks.extend(active_tracks)
    if not tracks:
        return None

    track_models = []
    for track in tracks:
        if len(track["rows"]) < 3:
            continue
        y0 = int(track["rows"][0])
        y1 = int(track["rows"][-1])
        span = y1 - y0 + 1
        if span < min_track_len_px:
            continue

        left = np.full(span, np.nan, dtype=np.float32)
        right = np.full(span, np.nan, dtype=np.float32)
        observed = np.zeros(span, dtype=bool)
        observed_mask = np.zeros((span, body_w), dtype=np.uint8)
        for row, lx, rx2 in zip(track["rows"], track["lefts"], track["rights"]):
            idx = int(row) - y0
            left[idx] = float(lx)
            right[idx] = float(rx2)
            observed[idx] = True
            lx_i = max(0, min(body_w - 1, int(round(lx))))
            rx_i = max(lx_i, min(body_w - 1, int(round(rx2))))
            observed_mask[idx, lx_i:rx_i + 1] = 255
        if np.count_nonzero(observed) < 4:
            continue

        left_obs = left.copy()
        right_obs = right.copy()
        left = _interp_nan_1d(left)
        right = _interp_nan_1d(right)
        widths = np.maximum(1.0, right - left + 1.0)
        obs_widths = widths[observed]
        if obs_widths.size == 0:
            continue
        width_med = float(np.median(obs_widths))
        if float(np.sum(obs_widths)) < min_track_area_px:
            continue

        present = _fill_short_false_runs(observed, max_gap_rows)
        present = _drop_short_true_runs(present, min_track_len_px)
        if not np.any(present):
            continue

        left_fit = _robust_poly_trace(left, observed, track_sigma_y, deg_hi=2)
        right_fit = _robust_poly_trace(right, observed, track_sigma_y, deg_hi=2)
        width_fit = np.maximum(1.0, right_fit - left_fit + 1.0)
        width_fit = np.clip(width_fit, max(1.0, width_med * 0.86), max(width_med * 1.36, width_med + 2.0))
        fit_span = np.maximum(1.0, right_fit - left_fit + 1.0)
        expand = np.maximum(0.0, 0.5 * (width_fit - fit_span))
        left_fit = left_fit - expand
        right_fit = right_fit + expand
        center_fit = 0.5 * (left_fit + right_fit)

        env_kernel_x = max(5, int(round(width_med * 1.15)))
        env_kernel_y = max(5, max_gap_rows * 3 + 1)
        envelope = cv2.dilate(observed_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (env_kernel_x, env_kernel_y)), iterations=1)

        left_global = np.full(body_h, np.nan, dtype=np.float32)
        right_global = np.full(body_h, np.nan, dtype=np.float32)
        center_global = np.full(body_h, np.nan, dtype=np.float32)
        left_global[y0:y1 + 1] = left_fit
        right_global[y0:y1 + 1] = right_fit
        center_global[y0:y1 + 1] = center_fit
        track_models.append({
            "y0": y0,
            "y1": y1,
            "span": span,
            "observed": observed,
            "present": present,
            "left_obs": left_obs,
            "right_obs": right_obs,
            "left_fit": left_fit,
            "right_fit": right_fit,
            "center_fit": center_fit,
            "width_fit": width_fit,
            "width_med": width_med,
            "envelope": envelope,
            "left_global": left_global,
            "right_global": right_global,
            "center_global": center_global,
            "base_center": float(np.median(center_fit[present])) if np.any(present) else float(np.median(center_fit)),
            "score": float(np.count_nonzero(observed)) * 2.0 + float(span),
        })

    if len(track_models) < 2:
        return None

    track_models.sort(key=lambda item: item["base_center"])
    dedup_models = []
    dedup_guard = max(4.0, float(np.median([m["width_med"] for m in track_models])) * 0.55)
    for model in track_models:
        if dedup_models and model["base_center"] - dedup_models[-1]["base_center"] < dedup_guard:
            if model["score"] > dedup_models[-1]["score"]:
                dedup_models[-1] = model
        else:
            dedup_models.append(model)
    track_models = dedup_models
    if len(track_models) < 2:
        return None

    center_bases = np.asarray([m["base_center"] for m in track_models], dtype=np.float32)
    diffs = np.diff(center_bases)
    pitch_px = float(np.median(diffs)) if diffs.size > 0 else 0.0
    if diffs.size > 3:
        valid_diffs = diffs[(diffs >= max(4.0, np.percentile(diffs, 20) * 0.8)) &
                            (diffs <= max(6.0, np.percentile(diffs, 80) * 1.2))]
        if valid_diffs.size > 0:
            pitch_px = float(np.median(valid_diffs))
    if prior_pitch_px > 0.0:
        if pitch_px <= 0.0 or prior_peak_conf >= 1.14:
            pitch_px = prior_pitch_px
        else:
            pitch_px = 0.68 * pitch_px + 0.32 * prior_pitch_px

    headland_raw = np.zeros_like(roi)
    margin_raw = np.zeros_like(roi)
    margin_raw[:body_y0] = roi[:body_y0]
    margin_raw[body_y1:] = roi[body_y1:]
    cc_count, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats((margin_raw > 0).astype(np.uint8), 8)
    headland_min_area = max(20, int(round(0.10 / max(analysis_mpp * analysis_mpp, 1e-6))))
    headland_min_span = max(24.0, 1.6 * pitch_px if pitch_px > 0.0 else 0.40 / max(analysis_mpp, 1e-6))
    for label in range(1, cc_count):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area < headland_min_area:
            continue
        bx = int(cc_stats[label, cv2.CC_STAT_LEFT])
        by = int(cc_stats[label, cv2.CC_STAT_TOP])
        bw_box = int(cc_stats[label, cv2.CC_STAT_WIDTH])
        bh_box = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
        touches_top = by < body_y0
        touches_bottom = by + bh_box > body_y1
        if not (touches_top or touches_bottom):
            continue
        if bw_box >= max(headland_min_span, bh_box * 1.35):
            headland_raw[cc_labels == label] = 255

    def _neighbor_center(models: List[dict], idx: int, yy: int, step: int) -> float:
        jj = idx + step
        while 0 <= jj < len(models):
            center_val = float(models[jj]["center_global"][yy])
            if math.isfinite(center_val):
                return center_val
            jj += step
        return float("nan")

    min_band_width = max(2.0, 0.05 / max(analysis_mpp, 1e-6))
    gap_guard = max(boundary_guard_px, 0.10 * pitch_px if pitch_px > 0.0 else boundary_guard_px)
    body_rebuilt = np.zeros_like(body_roi)
    for idx, model in enumerate(track_models):
        y0 = int(model["y0"])
        left_fit = np.asarray(model["left_fit"], dtype=np.float32)
        right_fit = np.asarray(model["right_fit"], dtype=np.float32)
        center_fit = np.asarray(model["center_fit"], dtype=np.float32)
        observed = np.asarray(model["observed"], dtype=bool)
        envelope = np.asarray(model["envelope"], dtype=np.uint8)
        present = np.asarray(model["present"], dtype=bool)
        for local_y in np.where(present)[0]:
            yy = y0 + int(local_y)
            pred_left = float(left_fit[local_y])
            pred_right = float(right_fit[local_y])
            if not (math.isfinite(pred_left) and math.isfinite(pred_right)):
                continue
            center_x = float(center_fit[local_y])
            left_bound = float(row_left_bound[yy])
            right_bound = float(row_right_bound[yy])
            left_neighbor = _neighbor_center(track_models, idx, yy, -1)
            right_neighbor = _neighbor_center(track_models, idx, yy, 1)
            if math.isfinite(left_neighbor):
                left_bound = max(left_bound, 0.5 * (left_neighbor + center_x) + gap_guard)
            if math.isfinite(right_neighbor):
                right_bound = min(right_bound, 0.5 * (center_x + right_neighbor) - gap_guard)
            if right_bound <= left_bound + 1.0:
                continue

            left = max(left_bound, pred_left)
            right = min(right_bound, pred_right)
            if right - left + 1.0 < min_band_width:
                continue

            lx = max(0, min(body_w - 1, int(round(left))))
            rx2 = max(lx, min(body_w - 1, int(round(right))))
            seg_l = lx
            seg_r = rx2
            if not observed[local_y]:
                allowed = envelope[local_y] > 0
                if not np.any(allowed):
                    continue
                allowed[:lx] = False
                allowed[rx2 + 1:] = False
                runs = _true_runs(allowed)
                if not runs:
                    continue

                def _run_score(run: Tuple[int, int]) -> Tuple[float, float]:
                    run_l = int(run[0])
                    run_r = int(run[1]) - 1
                    overlap = max(0.0, min(float(run_r), float(rx2)) - max(float(run_l), float(lx)) + 1.0)
                    center_delta = abs(0.5 * (run_l + run_r) - center_x)
                    return overlap, -center_delta

                best_run = max(runs, key=_run_score)
                seg_l = max(lx, int(best_run[0]))
                seg_r = min(rx2, int(best_run[1]) - 1)
            if seg_r - seg_l + 1 < min_band_width:
                continue
            body_rebuilt[yy, seg_l:seg_r + 1] = 255

    if np.count_nonzero(body_rebuilt) < 100:
        return None

    capture_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(round(0.08 / max(analysis_mpp, 1e-6)))), max(3, int(round(0.16 / max(analysis_mpp, 1e-6))))),
    )
    capture_roi = cv2.dilate(body_rebuilt, capture_kernel, iterations=1)
    body_rebuilt = cv2.bitwise_or(body_rebuilt, cv2.bitwise_and(body_roi, capture_roi))

    final_roi = np.zeros_like(roi)
    final_roi[body_y0:body_y1] = body_rebuilt
    final_roi = cv2.bitwise_or(final_roi, headland_raw)
    final_roi = cv2.bitwise_and(final_roi, support_roi)

    clean_small = np.zeros_like(rotated)
    clean_small[ry:ry + rh, rx:rx + rw] = final_roi
    restored_small = cv2.warpAffine(clean_small, inv_rot_mat, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
    if scale < 0.999:
        restored = cv2.resize(restored_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        restored = restored_small
    _, restored = cv2.threshold(restored, 127, 255, cv2.THRESH_BINARY)
    if field_support is not None:
        restored = cv2.bitwise_and(restored, field_support)
    restored = cv2.morphologyEx(restored, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return {
        "mask": restored,
        "angle_rad": angle_rad,
        "analysis_mpp": analysis_mpp,
        "track_count": int(len(track_models)),
        "pitch_px": float(pitch_px),
        "lattice_conf": float(prior_peak_conf),
    }


# ─── BandExtractor ──────────────────────────────────────────
def _component_principal_axis(component_mask: np.ndarray) -> Tuple[Optional[float], float, float]:
    ys, xs = np.where(np.asarray(component_mask, dtype=np.uint8) > 0)
    if ys.size < 8:
        return None, 0.0, 0.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    if centered.shape[0] < 2:
        return None, 0.0, 0.0
    cov = np.cov(centered, rowvar=False)
    if getattr(cov, "shape", None) != (2, 2):
        return None, 0.0, 0.0
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    angle = _wrap_half_turn(math.atan2(float(vec[1]), float(vec[0])))
    major = math.sqrt(max(float(vals[0]), 1e-6)) * 4.0
    minor = math.sqrt(max(float(vals[1]), 1e-6)) * 4.0 if vals.size > 1 else 1.0
    return angle, major, max(1.0, minor)


def _complete_parallel_gap_mask(mask: np.ndarray, geo: Optional[GeoUtils] = None,
                                state: Optional[AppState] = None,
                                analysis_mpp_floor: float = 0.012):
    raw = (mask > 0).astype(np.uint8) * 255
    field_support = _field_support_mask(raw.shape[:2], state)
    if field_support is None:
        return None

    support_area = float(np.count_nonzero(field_support))
    if support_area < 256.0:
        return None
    fill_ratio = float(np.count_nonzero(raw)) / max(support_area, 1.0)
    if fill_ratio < 0.55:
        return None

    raw_gap = np.where((field_support > 0) & (raw == 0), 255, 0).astype(np.uint8)
    if np.count_nonzero(raw_gap) < 100:
        return None

    gap_model = _fit_evidence_strip_model(raw_gap, geo, state, analysis_mpp_floor=analysis_mpp_floor)
    if gap_model is None or np.count_nonzero(gap_model["mask"]) < 100:
        return None

    pred_gap = gap_model["mask"].astype(np.uint8)
    combined_gap = cv2.bitwise_or(raw_gap, pred_gap)

    mpp = _mask_meters_per_pixel(raw, geo, state)
    row_angle = _estimate_row_angle(raw, geo, state, analysis_mpp=max(mpp, 0.010))
    head_angle = _wrap_half_turn(row_angle + math.pi * 0.5)
    min_comp_area_px = max(30, int(round(0.003 / max(mpp * mpp, 1e-6))))
    row_major_len_px = max(26.0, 0.25 / max(mpp, 1e-6))
    head_major_len_px = max(70.0, 0.70 / max(mpp, 1e-6))
    pred_overlap_px = max(18, int(round(0.08 / max(mpp, 1e-6))))
    row_tol = math.radians(16.0)
    head_tol = math.radians(18.0)

    cc_count, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats((combined_gap > 0).astype(np.uint8), 8)
    filtered_gap = np.zeros_like(combined_gap)
    kept_components = 0
    for label in range(1, cc_count):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area < min_comp_area_px:
            continue
        comp_mask = np.where(cc_labels == label, 255, 0).astype(np.uint8)
        overlap_pred = int(np.count_nonzero((comp_mask > 0) & (pred_gap > 0)))
        keep = overlap_pred >= pred_overlap_px
        if not keep:
            comp_angle, major, minor = _component_principal_axis(comp_mask)
            if comp_angle is not None:
                aspect = major / max(minor, 1.0)
                if aspect >= 3.2 and major >= row_major_len_px and _half_turn_diff(comp_angle, row_angle) <= row_tol:
                    keep = True
                elif major >= head_major_len_px and _half_turn_diff(comp_angle, head_angle) <= head_tol:
                    keep = True
        if keep:
            filtered_gap[comp_mask > 0] = 255
            kept_components += 1

    if np.count_nonzero(filtered_gap) < 100 or kept_components < 3:
        return None

    rebuilt = np.where((field_support > 0) & (filtered_gap == 0), 255, 0).astype(np.uint8)
    return {
        "mask": rebuilt,
        "gap_mask": filtered_gap,
        "raw_gap": raw_gap,
        "pred_gap": pred_gap,
        "fill_ratio": fill_ratio,
        "kept_components": kept_components,
    }


class BandExtractor:
    MAX_PIXELS = 5000000 

    def __init__(self, geo: GeoUtils):
        self.geo = geo
        self.log = AppLogger()
        self.cfg = Config()

    def extract(self, mask: np.ndarray, state: AppState = None) -> List[BandInstance]:
        if mask is None or np.count_nonzero(mask) < 100:
            return []

        H, W = mask.shape[:2]

        # 直扫轮廓：跳过降采样和 CC，直接在稀疏掩码上操作（快 5×）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_candidates = []
        min_area = max(20, int(self.cfg.MIN_BAND_AREA_PX * 0.5))
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            cnt = cnt.squeeze()
            if cnt.ndim < 2 or len(cnt) < 3:
                continue
            rect = cv2.minAreaRect(cnt)
            (ccx, ccy), (w, h), _ = rect
            short_edge = max(1.0, float(min(w, h)))
            aspect_ratio = float(max(w, h)) / short_edge
            if aspect_ratio < self.cfg.BAND_ASPECT_RATIO_MIN:
                continue
            raw_candidates.append({
                "contour": cnt,
                "centroid": (float(ccx), float(ccy)),
                "length": float(max(w, h)),
                "width": float(min(w, h))
            })

        if not raw_candidates:
            return []

        # Shared row-angle estimation. Prefer image texture when available,
        # fall back to mask geometry, and only use field-outline direction
        # as a weak fallback when the signal is ambiguous.
        dom_angle_rad = _estimate_row_angle(mask, self.geo, state)

        # 构造垂直于种植行的法线向量，用于投影排序
        normal_rad = dom_angle_rad + math.pi / 2
        nx, ny = math.cos(normal_rad), math.sin(normal_rad)
        if abs(nx) > abs(ny):
            if nx < 0: nx, ny = -nx, -ny
        else:
            if ny < 0: nx, ny = -nx, -ny

        # =================================================================
        # 👑 核心修复：投影排序与碎片聚类合并 (Fragment Clustering)
        # =================================================================
        for b in raw_candidates:
            b["proj"] = b["centroid"][0] * nx + b["centroid"][1] * ny
        raw_candidates.sort(key=lambda x: x["proj"])

        merged_bands = []
        if raw_candidates:
            curr_group = [raw_candidates[0]]
            median_width = float(np.median([b["width"] for b in raw_candidates]))
            merge_thresh = max(6.0, min(20.0, median_width * 1.5))
            for i in range(1, len(raw_candidates)):
                b = raw_candidates[i]
                if abs(b["proj"] - curr_group[0]["proj"]) < merge_thresh:
                    curr_group.append(b)
                else:
                    merged_bands.append(self._merge_group(curr_group, dom_angle_rad))
                    curr_group = [b]
            merged_bands.append(self._merge_group(curr_group, dom_angle_rad))

        for idx, band in enumerate(merged_bands):
            band.id = idx
        self.log.info(f"提取完成: {len(raw_candidates)} 个零散碎片 -> 物理聚类为 {len(merged_bands)} 条实体主干")
        return merged_bands

    def _merge_group(self, group: List[dict], dom_angle_rad: float) -> BandInstance:
        """将同一行的多个碎片合并为一条完整种植带"""
        all_points = np.vstack([b["contour"] for b in group])
        hull = cv2.convexHull(all_points).squeeze()
        rect = cv2.minAreaRect(hull)
        (ccx, ccy), (w, h), _ = rect
        return BandInstance(
            id=0, mask=None, contour=hull,
            centroid=(float(ccx), float(ccy)), angle=math.degrees(dom_angle_rad),
            length=float(max(w, h)), width=float(min(w, h)),
            bbox=(int(ccx - max(w, h) / 2), int(ccy - min(w, h) / 2),
                  int(max(w, h)), int(min(w, h)))
        )


# ─── CorridorAnalyzer ───────────────────────────────────────
class CorridorAnalyzer:
    def __init__(self, geo: GeoUtils, state: AppState = None):
        self.geo = geo
        self.state = state
        self.cfg = Config()
        self.log = AppLogger()

    def _harvester_specs(self) -> dict:
        hp = self.state.harvester_params if self.state else {}
        cutter_width_m = float(hp.get("cutter_width_m", self.cfg.CUTTER_WIDTH_M))
        track_width_m = float(hp.get("track_width_m", self.cfg.TRACK_WIDTH_M))
        track_gauge_m = float(hp.get("track_gauge_m", self.cfg.TRACK_GAUGE_M))
        wheelbase_m = float(hp.get("wheelbase_m", self.cfg.WHEELBASE_M))
        track_length_m = float(hp.get("track_length_m", self.cfg.TRACK_LENGTH_M))
        turn_radius_m = float(hp.get("turn_radius_m", self.cfg.TURN_RADIUS_M))
        machine_width_m = max(track_width_m * 2.0, track_gauge_m + track_width_m)
        return {
            "cutter_width_m": cutter_width_m,
            "track_width_m": track_width_m,
            "track_gauge_m": track_gauge_m,
            "wheelbase_m": wheelbase_m,
            "track_length_m": track_length_m,
            "turn_radius_m": turn_radius_m,
            "machine_width_m": machine_width_m,
        }

    def _effective_headland_buffer_m(self) -> float:
        specs = self._harvester_specs()
        return max(
            float(self.cfg.HEADLAND_BUFFER_M),
            specs["turn_radius_m"] + specs["wheelbase_m"] * 0.6 + specs["cutter_width_m"] * 0.25,
            specs["track_length_m"] * 0.75 + specs["turn_radius_m"] * 0.8,
        )

    def _clip_centerline(self, centerline: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(centerline) < 2:
            return centerline

        region = _state_field_region(self.state)
        if region is None and self.state and self.state.img_w > 0 and self.state.img_h > 0:
            region = Polygon([
                (0, 0),
                (self.state.img_w, 0),
                (self.state.img_w, self.state.img_h),
                (0, self.state.img_h),
            ])
        if region is None:
            return centerline

        clipped = LineString(centerline).intersection(region)
        if clipped.is_empty:
            return centerline

        if clipped.geom_type == "LineString":
            coords = list(clipped.coords)
        else:
            parts = [g for g in getattr(clipped, "geoms", []) if g.geom_type == "LineString"]
            if not parts:
                return centerline
            coords = list(max(parts, key=lambda g: g.length).coords)

        if len(coords) < 2:
            return centerline
        return [(float(x), float(y)) for x, y in (coords[0], coords[-1])]

    def _trim_headland(self, centerline: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(centerline) < 2:
            return centerline

        (x1, y1), (x2, y2) = centerline[0], centerline[-1]
        dx, dy = x2 - x1, y2 - y1
        seg_len_px = math.hypot(dx, dy)
        if seg_len_px <= 1e-6:
            return centerline

        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        ppm = self.geo.pixels_per_meter(mx, my) if self.geo.is_ready() else 1.0
        trim_px = max(0.0, self._effective_headland_buffer_m() * ppm)
        if seg_len_px <= trim_px * 2 + 1.0:
            return centerline

        ux, uy = dx / seg_len_px, dy / seg_len_px
        return [
            (x1 + ux * trim_px, y1 + uy * trim_px),
            (x2 - ux * trim_px, y2 - uy * trim_px),
        ]

    def analyze(self, bands: List[BandInstance]) -> List[Corridor]:
        if len(bands) < 2: return []
        corridors = []
        for i in range(len(bands) - 1):
            corridors.append(self._compute_corridor(bands[i], bands[i + 1]))
        return corridors

    def _compute_corridor(self, left: BandInstance, right: BandInstance) -> Corridor:
        """极简绝对居中线：强制平行于蓝框绝对方向，拒绝任何局部拟合偏差"""
        cx = (left.centroid[0] + right.centroid[0]) / 2.0
        cy = (left.centroid[1] + right.centroid[1]) / 2.0
        angle_rad = math.radians(left.angle)
        vx, vy = math.cos(angle_rad), math.sin(angle_rad)
        nx, ny = -vy, vx

        left_pts = _contour_points(left.contour)
        right_pts = _contour_points(right.contour)
        left_proj = left_pts[:, 0] * nx + left_pts[:, 1] * ny
        right_proj = right_pts[:, 0] * nx + right_pts[:, 1] * ny
        raw_width_px = abs(float(np.mean(right_proj) - np.mean(left_proj))) - (left.width + right.width) / 2.0

        ppm = self.geo.pixels_per_meter(cx, cy) if self.geo.is_ready() else 1.0
        cor_w = max(0.0, raw_width_px / ppm if ppm > 0 else 0.0)

        if self.state and self.state.img_w > 0 and self.state.img_h > 0:
            L = math.hypot(self.state.img_w, self.state.img_h) * 1.5
        else:
            L = 20000.0
        cl = [(cx - L * vx, cy - L * vy), (cx + L * vx, cy + L * vy)]
        cl = self._trim_headland(self._clip_centerline(cl))
        seg_len_m = _polyline_length_m(cl, self.geo)
        specs = self._harvester_specs()
        min_clearance_m = max(
            self.cfg.MIN_CORRIDOR_WIDTH_M,
            specs["track_gauge_m"] + specs["track_width_m"] * 0.2 - specs["cutter_width_m"] * 0.1,
        )
        passable = cor_w >= min_clearance_m and seg_len_m > 1.0

        return Corridor(
            left_id=left.id, right_id=right.id, centerline=cl,
            min_width_m=cor_w, avg_width_m=cor_w, passable=passable,
            risk=(not passable)
        )


def _mask_structure_metrics(mask: np.ndarray, geo: Optional[GeoUtils] = None,
                            state: Optional[AppState] = None) -> Dict[str, float]:
    prior = _estimate_lattice_prior(mask, geo, state)
    if prior is None:
        return {
            "pitch_px": 0.0,
            "spacing_cv": 1.0,
            "coverage_ratio": 0.0,
            "lattice_residual": 1.0,
            "continuity_ratio": 0.0,
            "headland_score": 0.0,
            "structure_score": 0.0,
            "peak_conf": 0.0,
            "stripe_count": 0.0,
        }
    return {
        "pitch_px": float(prior.get("pitch_px", 0.0)),
        "spacing_cv": float(prior.get("spacing_cv", 1.0)),
        "coverage_ratio": float(prior.get("coverage_ratio", 0.0)),
        "lattice_residual": float(prior.get("lattice_residual", 1.0)),
        "continuity_ratio": float(prior.get("continuity_ratio", 0.0)),
        "headland_score": float(prior.get("headland_score", 0.0)),
        "structure_score": float(prior.get("structure_score", 0.0)),
        "peak_conf": float(prior.get("peak_conf", 0.0)),
        "stripe_count": float(prior.get("stripe_count", 0.0)),
    }


def _score_smooth_candidate(raw: np.ndarray, candidate: np.ndarray,
                            geo: Optional[GeoUtils] = None,
                            state: Optional[AppState] = None) -> Dict[str, float]:
    precision, recall, iou = _mask_overlap_scores(raw, candidate)
    area_ratio = float(np.count_nonzero(candidate)) / max(float(np.count_nonzero(raw)), 1.0)
    structure = _mask_structure_metrics(candidate, geo, state)
    area_penalty = min(abs(math.log(max(area_ratio, 1e-6))) / 0.35, 1.0)
    overlap_score = (
        0.40 * precision +
        0.22 * recall +
        0.18 * iou +
        0.10 * min(area_ratio, 1.0 / max(area_ratio, 1e-6)) +
        0.10 * float(np.clip((precision + recall) * 0.5, 0.0, 1.0))
    )
    total_score = overlap_score + 0.95 * structure["structure_score"] - 0.10 * area_penalty
    return {
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "area_ratio": float(area_ratio),
        "overlap_score": float(overlap_score),
        "total_score": float(total_score),
        **structure,
    }


def smooth_mask(mask: np.ndarray, geo=None, state=None) -> np.ndarray:
    """
    Scale-aware post-processing for crop-band masks.
    The algorithm works in metric units, preserves real wide gaps, and
    rebuilds each planting band as a simpler geometric strip.
    """
    from row_geometry import regularize_crop_mask

    config = Config().section("mask_processing")
    return regularize_crop_mask(
        mask,
        geo=geo,
        state=state,
        config=config,
    )["processed_mask"]

    # Compatibility implementation retained below for fallback reference.
    if mask is None or np.count_nonzero(mask) < 100:
        return mask

    import time as _time
    _last_redraw = [0.0]

    def _progress(stage: str, pct: float, log_msg: str = ""):
        if state is not None:
            nonlocal _last_redraw
            _kw = {"smooth_stage": stage, "smooth_progress": pct}
            if log_msg and hasattr(state, 'smooth_log'):
                _log = list(getattr(state, 'smooth_log', []) or [])
                _log.append(log_msg)
                _kw["smooth_log"] = _log[-8:]
            # 节流：最多每 0.15s 触发一次重绘
            now = _time.time()
            if now - _last_redraw[0] > 0.15:
                _kw["need_redraw"] = True
                _last_redraw[0] = now
            state.safe_update(**_kw)

    raw = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    h, w = raw.shape[:2]

    # ---- 大图预降采样，避免后续步骤内存爆炸 ----
    _max_work = 2400
    _work_scale = 1.0
    if max(h, w) > _max_work:
        _work_scale = _max_work / max(h, w)
        raw_work = cv2.resize(raw, (int(w * _work_scale), int(h * _work_scale)),
                              interpolation=cv2.INTER_NEAREST)
    else:
        raw_work = raw
    wh, ww = raw_work.shape[:2]

    # ---------------------------------------------------------
    # Step 1: 先验诊断（在降采样图上估算方向/行距）
    # ---------------------------------------------------------
    _progress('方向估算', 0.02, '-> 正在计算种植带方向...')
    prior = _estimate_lattice_prior(raw_work, geo, state)
    if prior is None:
        return cv2.morphologyEx(raw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    _progress('方向估算', 0.05, 'ok 方向角估算完成')
    angle_rad = float(prior['angle_rad'])
    pitch_px_down = float(prior['pitch_px'])
    phase_px_down = float(prior['phase_px'])

    _progress('行距估算', 0.08, f'ok 行距 {pitch_px_down:.1f}px (降采样)')

    # ---- 还原 pitch/phase 到原图分辨率 ----
    mpp = _mask_meters_per_pixel(raw, geo, state)
    work_mpp = max(mpp, 0.012)
    scale = min(1.0, mpp / max(work_mpp, 1e-6))
    max_dim = 2800.0
    if max(h, w) * scale > max_dim:
        scale = max_dim / max(float(h), float(w))
    scale = max(scale, 0.01)
    pitch_px = pitch_px_down / max(scale, 1e-6)
    phase_px = phase_px_down / max(scale, 1e-6)

    vx = math.cos(angle_rad)
    vy = math.sin(angle_rad)
    nx = -vy   # 法线方向
    ny = vx

    _progress('解析准备', 0.10, f'ok 原图行距 {pitch_px:.1f}px, 方向 {math.degrees(angle_rad):.1f}°')

    # ---------------------------------------------------------
    # Step 2: 投影空间地头粗裁 (top 8% / bottom 8%) — 在降采样图上操作
    # ---------------------------------------------------------
    ys_all, xs_all = np.where(raw_work > 0)
    if len(xs_all) < 100:
        return raw

    proj_long = xs_all * vx + ys_all * vy
    min_l, max_l = proj_long.min(), proj_long.max()
    long_range = max_l - min_l
    cut_top = min_l + long_range * 0.08
    cut_bottom = max_l - long_range * 0.08
    inside = (proj_long >= cut_top) & (proj_long <= cut_bottom)
    xs_in, ys_in = xs_all[inside], ys_all[inside]
    if len(xs_in) < 100:
        return raw

    _progress('地头裁剪', 0.18, f'ok 地头裁剪: 保留中间 84% 区域 ({len(xs_in)}/{len(xs_all)} 像素)')

    # Step 3: 1D Voronoi 行分离
    _progress('行分离', 0.22, '-> 正在计算法线投影分配...')
    proj_lat = xs_in * nx + ys_in * ny
    offset = proj_lat - phase_px
    row_ids = np.round(offset / pitch_px).astype(np.int32)
    dist_to_center = np.abs(offset - row_ids.astype(np.float64) * pitch_px)

    # confusion zone: 丢弃距行中心 > 0.35×pitch 的像素
    safe_width = pitch_px * 0.35
    valid = dist_to_center <= safe_width
    row_ids_v = row_ids[valid]
    xs_v, ys_v = xs_in[valid], ys_in[valid]
    unique_rows = np.unique(row_ids_v)

    _progress('行分离完成', 0.28, f'ok 分离出 {len(unique_rows)} 行, 隔离带丢弃 {len(xs_all[inside])-len(xs_v)} 模糊像素')

    # Step 4: 逐行

    # ---------------------------------------------------------
    # Step 4: 逐行独立手术 + B 样条重构
    # ---------------------------------------------------------
    def _oriented_kernel(length, ang_rad, thickness=2):
        ksize = max(3, int(length))
        if ksize % 2 == 0:
            ksize += 1
        k = np.zeros((ksize, ksize), dtype=np.uint8)
        c = ksize // 2
        dx = int(math.cos(ang_rad) * (length / 2.0))
        dy = int(math.sin(ang_rad) * (length / 2.0))
        cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thickness)
        return k

    reconstructed = np.zeros((h, w), dtype=np.uint8)
    n_rows = len(unique_rows)
    if n_rows == 0:
        return reconstructed

    # 缓冲 padding (形态学运算需要)
    pad = max(20, int(pitch_px * 2))

    # 尝试导入 B 样条
    try:
        from scipy.interpolate import splprep, splev
        HAVE_SPLINE = True
    except ImportError:
        HAVE_SPLINE = False

    # 预计算所有行的像素索引（避免每行重复查找）
    row_pixel_groups = {}
    for ri, r_id in enumerate(unique_rows):
        m = np.where(row_ids_v == r_id)[0]
        if len(m) < 50:
            continue
        row_pixel_groups[r_id] = (ys_v[m], xs_v[m])

    n_valid = len(row_pixel_groups)
    if n_valid == 0:
        return reconstructed

    progress_step = max(1, n_valid // 10)
    progress_count = 0

    for ri, (r_id, (y_idx, x_idx)) in enumerate(row_pixel_groups.items()):
        # ---- 计算 bbox（避免创建全分辨率数组） ----
        y0, y1 = int(y_idx.min()), int(y_idx.max()) + 1
        x0, x1 = int(x_idx.min()), int(x_idx.max()) + 1
        y0_pad = max(0, y0 - pad)
        y1_pad = min(h, y1 + pad)
        x0_pad = max(0, x0 - pad)
        x1_pad = min(w, x1 + pad)
        bh = y1_pad - y0_pad
        bw = x1_pad - x0_pad

        # ---- 在 bbox 内创建行掩码（~KB ~ MB 级） ----
        row_mask = np.zeros((bh, bw), dtype=np.uint8)
        local_y = y_idx - y0_pad
        local_x = x_idx - x0_pad
        valid_local = (local_y >= 0) & (local_y < bh) & (local_x >= 0) & (local_x < bw)
        row_mask[local_y[valid_local], local_x[valid_local]] = 255

        # ---- 分级桥接 ----
        healed = row_mask.copy()
        for sc in (0.4, 0.8, 1.2):
            kl = max(5, int(pitch_px * sc))
            k = _oriented_kernel(kl, angle_rad, thickness=2)
            healed = cv2.morphologyEx(healed, cv2.MORPH_CLOSE, k)

        # ---- 取最大连通域 ----
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(healed, connectivity=8)
        if n_labels < 2:
            del row_mask, healed
            continue
        main_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        main_body = (labels == main_idx).astype(np.uint8) * 255

        # ---- 骨架 ----
        try:
            skeleton = cv2.ximgproc.thinning(main_body)
        except Exception:
            skeleton = main_body
        skel_ys, skel_xs = np.where(skeleton > 0)
        if len(skel_xs) < 10:
            del row_mask, healed, main_body, skeleton
            continue

        # 还原到全局坐标
        skel_ys_g = skel_ys + y0_pad
        skel_xs_g = skel_xs + x0_pad

        # ---- 排序 + B样条 ----
        projs = skel_xs_g * vx + skel_ys_g * vy
        order = np.argsort(projs)
        sx, sy = skel_xs_g[order].astype(np.float64), skel_ys_g[order].astype(np.float64)

        spline_pts = np.column_stack([sx, sy]).astype(np.int32)
        if HAVE_SPLINE and len(sx) >= 8:
            try:
                keep = np.ones(len(sx), dtype=bool)
                keep[1:] = (np.diff(sx) != 0) | (np.diff(sy) != 0)
                if np.sum(keep) >= 8:
                    sx_u, sy_u = sx[keep], sy[keep]
                    tck, u = splprep([sx_u, sy_u], s=len(sx_u) * 0.3)
                    u_new = np.linspace(0, 1, min(200, max(20, len(sx_u) // 2)))
                    fx, fy = splev(u_new, tck)
                    spline_pts = np.column_stack([fx, fy]).astype(np.int32)
            except Exception:
                pass

        # ---- 宽度重绘 ----
        area = float(np.count_nonzero(main_body))
        length_ = float(np.sum(np.sqrt(np.sum(np.diff(spline_pts.astype(np.float64), axis=0) ** 2, axis=1))))
        avg_width = max(2, int(area / max(length_, 1.0)))
        cv2.polylines(reconstructed, [spline_pts], isClosed=False, color=255, thickness=avg_width)

        # ---- 显式释放中间数组 ----
        del row_mask, healed, main_body, skeleton
        progress_count += 1

        # ---- 进度 + 定期 GC ----
        if progress_count % progress_step == 0 or progress_count == n_valid:
            pct = 0.30 + 0.65 * progress_count / max(n_valid, 1)
            _progress(f'逐行重构 {progress_count}/{n_valid}', pct,
                      f'ok 第 {progress_count}/{n_valid} 行: 宽度{avg_width}px')
            # 每 10% 显式回收一次内存
            if progress_count % (progress_step * 3) == 0:
                import gc; gc.collect()

    _progress('完成', 1.0)
    return reconstructed

# ─── PathPlanner: 暴力全覆盖 + 底层通讯专用直角转弯 ────────
class PathPlanner:
    def __init__(self, geo: GeoUtils, state: AppState):
        self.geo, self.state, self.cfg, self.log = geo, state, Config(), AppLogger()

    def _harvester_specs(self) -> dict:
        hp = self.state.harvester_params or {}
        cutter_width_m = float(hp.get("cutter_width_m", self.cfg.CUTTER_WIDTH_M))
        track_width_m = float(hp.get("track_width_m", self.cfg.TRACK_WIDTH_M))
        track_gauge_m = float(hp.get("track_gauge_m", self.cfg.TRACK_GAUGE_M))
        wheelbase_m = float(hp.get("wheelbase_m", self.cfg.WHEELBASE_M))
        track_length_m = float(hp.get("track_length_m", self.cfg.TRACK_LENGTH_M))
        turn_radius_m = float(hp.get("turn_radius_m", self.cfg.TURN_RADIUS_M))
        machine_width_m = max(track_width_m * 2.0, track_gauge_m + track_width_m)
        return {
            "cutter_width_m": cutter_width_m,
            "track_width_m": track_width_m,
            "track_gauge_m": track_gauge_m,
            "wheelbase_m": wheelbase_m,
            "track_length_m": track_length_m,
            "turn_radius_m": turn_radius_m,
            "machine_width_m": machine_width_m,
        }

    def _effective_headland_buffer_m(self) -> float:
        specs = self._harvester_specs()
        return max(
            float(self.cfg.HEADLAND_BUFFER_M),
            specs["turn_radius_m"] + specs["wheelbase_m"] * 0.6 + specs["cutter_width_m"] * 0.25,
            specs["track_length_m"] * 0.75 + specs["turn_radius_m"] * 0.8,
        )

    def _field_region(self) -> Optional[Polygon]:
        return _state_field_region(self.state)

    def _constrain_path_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        region = self._field_region()
        if region is None or len(points) < 2:
            return [(float(x), float(y)) for x, y in points]
        return _constrain_polyline_to_region(points, region, max_step_px=18.0)

    def _dominant_row_angle_rad(self, bands: List[BandInstance], field_boundary: Optional[List[Tuple[float, float]]] = None) -> float:
        if bands:
            band = max(bands, key=lambda b: b.length)
            ang = math.radians(float(band.angle))
            if math.sin(ang) < 0:
                ang += math.pi
            return ang
        pts = field_boundary or self.state.field_boundary
        if pts and len(pts) >= 2:
            best_len = 0.0
            best_angle = 0.0
            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                seg_len = math.hypot(dx, dy)
                if seg_len > best_len:
                    best_len = seg_len
                    best_angle = math.atan2(dy, dx)
            if math.sin(best_angle) < 0:
                best_angle += math.pi
            return best_angle
        return 0.0

    def _corridor_projection(self, corridor: Corridor) -> float:
        if len(corridor.centerline) < 2:
            return 0.0
        p1, p2 = corridor.centerline[0], corridor.centerline[-1]
        ux, uy = _normalize_vec(p2[0] - p1[0], p2[1] - p1[1])
        nx, ny = -uy, ux
        mx = (p1[0] + p2[0]) * 0.5
        my = (p1[1] + p2[1]) * 0.5
        return mx * nx + my * ny

    def _select_work_corridors(self, corridors: List[Corridor]) -> List[Corridor]:
        active = [c for c in corridors if c.passable and len(c.centerline) >= 2]
        if len(active) <= 1:
            return active

        specs = self._harvester_specs()
        swath_m = max(specs["machine_width_m"], specs["cutter_width_m"] * 0.92)
        overlap_m = max(0.10, specs["cutter_width_m"] * 0.08)
        max_step_m = max(specs["machine_width_m"], swath_m - overlap_m)
        min_step_m = max(0.60, specs["machine_width_m"] * 0.75)

        ordered = sorted(((self._corridor_projection(c), c) for c in active), key=lambda x: x[0])
        selected = [ordered[0]]
        idx = 0
        while idx < len(ordered) - 1:
            next_idx = idx + 1
            for cand_idx in range(idx + 1, len(ordered)):
                gap = ordered[cand_idx][0] - ordered[idx][0]
                if gap < min_step_m:
                    continue
                if gap <= max_step_m:
                    next_idx = cand_idx
                else:
                    break
            selected.append(ordered[next_idx])
            idx = next_idx

        return [c for _, c in selected]

    def _build_transition_points(
        self,
        current_cl: List[Tuple[float, float]],
        next_cl: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if len(current_cl) < 2 or len(next_cl) < 2:
            return [current_cl[-1], next_cl[0]]

        cp = current_cl[-1]
        np_pt = next_cl[0]
        curr_dir = _normalize_vec(cp[0] - current_cl[-2][0], cp[1] - current_cl[-2][1])
        next_dir = _normalize_vec(next_cl[1][0] - np_pt[0], next_cl[1][1] - np_pt[1])
        mx, my = (cp[0] + np_pt[0]) * 0.5, (cp[1] + np_pt[1]) * 0.5
        ppm = self.geo.pixels_per_meter(mx, my) if self.geo.is_ready() else 1.0
        lead_m = max(
            0.5,
            min(self._effective_headland_buffer_m() * 0.9, self._harvester_specs()["turn_radius_m"] + 0.5 * self._harvester_specs()["wheelbase_m"]),
        )
        lead_px = lead_m * ppm
        p_out = (cp[0] + curr_dir[0] * lead_px, cp[1] + curr_dir[1] * lead_px)
        q_out = (np_pt[0] - next_dir[0] * lead_px, np_pt[1] - next_dir[1] * lead_px)

        points = [cp]
        for pt in (p_out, q_out, np_pt):
            if math.hypot(pt[0] - points[-1][0], pt[1] - points[-1][1]) > 1e-6:
                points.append(pt)
        return points

    def clip_field_boundary(self) -> Optional[List[Tuple[float, float]]]:
        if not self.state.field_boundary or len(self.state.field_boundary) < 3: return None
        try:
            from shapely.geometry import Polygon
            user_poly = Polygon(self.state.field_boundary)
            img_bbox = Polygon([(0, 0), (self.state.img_w, 0),
                                (self.state.img_w, self.state.img_h), (0, self.state.img_h)])
            clipped = user_poly.intersection(img_bbox)
            if clipped.is_empty: return list(self.state.field_boundary)
            if clipped.geom_type == 'Polygon':
                return [(float(x), float(y)) for x, y in list(clipped.exterior.coords)[:-1]]
            elif clipped.geom_type == 'MultiPolygon':
                largest = max(clipped.geoms, key=lambda p: p.area)
                return [(float(x), float(y)) for x, y in list(largest.exterior.coords)[:-1]]
            return list(self.state.field_boundary)
        except: return list(self.state.field_boundary) if self.state.field_boundary else None

    def compute_optimal_entry_exit(self, bands: List[BandInstance],
                                     field_boundary: List[Tuple[float, float]]) -> Tuple:
        """优先选择与作业方向近似垂直的两条头地边作为进出场边。"""
        if not field_boundary or len(field_boundary) < 3: return None, None
        row_angle = self._dominant_row_angle_rad(bands, field_boundary)
        row_dir = (math.cos(row_angle), math.sin(row_angle))
        n = len(field_boundary)
        edge_candidates = []
        for i in range(n):
            p1, p2 = field_boundary[i], field_boundary[(i+1)%n]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            el = math.hypot(dx, dy)
            if el <= 1e-6:
                continue
            edge_dir = (dx / el, dy / el)
            alignment = abs(edge_dir[0] * row_dir[0] + edge_dir[1] * row_dir[1])
            edge_candidates.append((alignment, -el, i, p1, p2))

        edge_candidates.sort(key=lambda x: (x[0], x[1]))
        chosen = []
        used = set()
        for _, _, idx, p1, p2 in edge_candidates:
            if idx in used:
                continue
            chosen.append((p1, p2))
            used.add(idx)
            used.add((idx - 1) % n)
            used.add((idx + 1) % n)
            if len(chosen) == 2:
                break

        if len(chosen) >= 2:
            (a, b), (c, d) = chosen[:2]
            entry_pt = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            exit_pt = ((c[0] + d[0]) / 2, (c[1] + d[1]) / 2)
        else:
            a, b = field_boundary[0], field_boundary[1]
            c, d = field_boundary[len(field_boundary) // 2], field_boundary[(len(field_boundary) // 2 + 1) % len(field_boundary)]
            entry_pt = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            exit_pt = ((c[0] + d[0]) / 2, (c[1] + d[1]) / 2)
        return entry_pt, exit_pt

    def plan_min_rolling(self, bands: List[BandInstance], corridors: List,
                         entry_point=None, exit_point=None,
                         start_idx=0, end_idx=None) -> List[AutoPathSegment]:
        """基于农机幅宽和车体宽度选择作业通道，并生成蛇形往返路线。"""
        path = []
        if not bands or not corridors: return []

        active = self._select_work_corridors(corridors)
        if not active:
            return []

        forward = True
        first_cl = active[0].centerline
        if entry_point and len(first_cl) >= 2:
            d0 = math.hypot(first_cl[0][0] - entry_point[0], first_cl[0][1] - entry_point[1])
            d1 = math.hypot(first_cl[-1][0] - entry_point[0], first_cl[-1][1] - entry_point[1])
            forward = d0 <= d1

        for i, corridor in enumerate(active):
            cl = list(corridor.centerline)
            if not forward:
                cl.reverse()
            forward = not forward
            if len(cl) < 2:
                continue

            work_seg = AutoPathSegment()
            for pt in cl:
                work_seg.points.append(PathPoint(pixel_x=pt[0], pixel_y=pt[1]))
            work_seg.status = 1
            work_seg.segment_type = "work"
            work_seg.length_m = _polyline_length_m(cl, self.geo)
            path.append(work_seg)

            if i < len(active) - 1:
                next_corridor = active[i+1]
                next_cl = list(next_corridor.centerline)
                if forward:
                    next_cl.reverse()
                if len(next_cl) < 2:
                    continue
                turn_seg = AutoPathSegment()
                turn_pts = self._constrain_path_points(self._build_transition_points(cl, next_cl))
                turn_seg.points = [PathPoint(pixel_x=pt[0], pixel_y=pt[1]) for pt in turn_pts]
                turn_seg.status = 0
                turn_seg.segment_type = "turn"
                turn_seg.length_m = _polyline_length_m(turn_pts, self.geo)
                path.append(turn_seg)

        if entry_point and path:
            entry_seg = AutoPathSegment()
            first_pt = path[0].points[0]
            entry_pts = self._constrain_path_points([
                (entry_point[0], entry_point[1]),
                (first_pt.pixel_x, first_pt.pixel_y),
            ])
            entry_seg.points = [PathPoint(pixel_x=pt[0], pixel_y=pt[1]) for pt in entry_pts]
            entry_seg.status = 1
            entry_seg.segment_type = "entry"
            entry_seg.length_m = _polyline_length_m(
                entry_pts,
                self.geo,
            )
            path.insert(0, entry_seg)

        if exit_point and path:
            exit_seg = AutoPathSegment()
            last_pt = path[-1].points[-1]
            exit_pts = self._constrain_path_points([
                (last_pt.pixel_x, last_pt.pixel_y),
                (exit_point[0], exit_point[1]),
            ])
            exit_seg.points = [PathPoint(pixel_x=pt[0], pixel_y=pt[1]) for pt in exit_pts]
            exit_seg.status = 1
            exit_seg.segment_type = "exit"
            exit_seg.length_m = _polyline_length_m(
                exit_pts,
                self.geo,
            )
            path.append(exit_seg)

        return path

    @staticmethod
    def estimate_rolling_rate(
        bands: List[BandInstance],
        path: List[AutoPathSegment],
        field_area_m2: Optional[float] = None,
        cutter_width_m: Optional[float] = None,
        track_width_m: Optional[float] = None,
    ) -> Tuple:
        """估算碾压率和收割率。返回 (rolling_rate_pct, harvest_rate_pct)"""
        if not bands or not path:
            return 0.0, 0.0
        total_work = sum(s.length_m for s in path if s.segment_type == "work")
        total = sum(s.length_m for s in path)
        if total == 0: return 0.0, 0.0
        if field_area_m2 and field_area_m2 > 1e-6 and cutter_width_m and track_width_m:
            harvest_area_m2 = total_work * cutter_width_m
            rolling_area_m2 = total * track_width_m * 2.0
            harvest = min(100.0, harvest_area_m2 / field_area_m2 * 100.0)
            rolling = min(100.0, rolling_area_m2 / field_area_m2 * 100.0)
            return rolling, harvest
        harvest = total_work / total * 100
        rolling = max(0.0, 100.0 - harvest)
        return rolling, harvest


def diagnose_mask_params(mask: np.ndarray, geo: GeoUtils, bands_truth: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    """扫描一组基础后处理参数，给出可直接写回配置的推荐值。"""
    if mask is None or np.count_nonzero(mask) < 100:
        print("掩码为空，无法诊断。")
        return None, None

    cfg = Config()
    model_raw = cfg.section("model")
    base_area = max(200, int(cfg.MIN_BAND_AREA_PX))
    base_erode = max(0, int(model_raw.get("erode_kernel_size", cfg.ERODE_KERNEL_SIZE)))
    area_candidates = sorted({
        max(200, int(base_area * 0.25)),
        max(200, int(base_area * 0.5)),
        base_area,
        int(base_area * 1.5),
        int(base_area * 2.0),
    })
    erode_candidates = sorted({
        0,
        max(3, base_erode - 2) if base_erode > 0 else 0,
        max(3, base_erode) if base_erode > 0 else 0,
        max(3, base_erode + 2) if base_erode > 0 else 0,
    })

    best_score = float("-inf")
    best = (None, None)
    print("开始扫描后处理参数组合：")
    print(f"  erode_kernel candidates = {erode_candidates}")
    print(f"  min_band_area candidates = {area_candidates}")

    for kernel_size in erode_candidates:
        if kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            work_mask = cv2.erode(mask, kernel, iterations=max(1, cfg.ERODE_ITERATIONS))
        else:
            work_mask = mask.copy()

        for min_area in area_candidates:
            with cfg.temporary_section("path_planning", {"min_band_area_px": min_area}):
                bands = BandExtractor(geo).extract(work_mask, None)
            count = len(bands)
            if count == 0:
                score = -1e9
                avg_aspect = 0.0
            else:
                aspects = [b.length / max(1.0, b.width) for b in bands]
                avg_aspect = float(np.mean(aspects))
                score = count * 10.0 + avg_aspect
                if bands_truth is not None:
                    score -= abs(count - bands_truth) * 100.0

            print(
                f"kernel={kernel_size:>2}, min_area={min_area:>5}, "
                f"bands={count:>3}, avg_aspect={avg_aspect:>5.2f}, score={score:>8.2f}"
            )

            if score > best_score:
                best_score = score
                best = (kernel_size, min_area)

    if best[0] is None:
        print("未找到有效参数组合。")
        return None, None

    print(f"\n推荐参数: erode_kernel_size={best[0]}, min_band_area_px={best[1]}")
    return best
