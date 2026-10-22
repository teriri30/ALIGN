"""Export reproducible, component-level diagnostics for processed masks."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


DIAGNOSTIC_SCHEMA = "align.mask-diagnostic.v1"


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _binary(value: Any, shape: Tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape[:2] != shape:
        return np.zeros(shape, dtype=np.uint8)
    return (np.asarray(value) > 0).astype(np.uint8) * 255


def _projection_interval(mask: np.ndarray, angle: float) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    values = xs.astype(np.float64) * direction[0] + ys.astype(np.float64) * direction[1]
    return float(values.min()), float(values.max())


def _component_geometry(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    if xs.size < 2:
        return 0.0, 0.0, 0.0
    points = np.column_stack((xs, ys)).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major_value, minor_value = values[order[0]], values[order[1]]
    vector = vectors[:, order[0]]
    angle = math.atan2(float(vector[1]), float(vector[0])) % math.pi
    major = 4.0 * math.sqrt(max(float(major_value), 0.0))
    minor = 4.0 * math.sqrt(max(float(minor_value), 0.0))
    return angle, major, minor


def _angle_difference_deg(first: float, second: float) -> float:
    difference = abs(float(first) - float(second)) % math.pi
    return math.degrees(min(difference, math.pi - difference))


def _distance_to_reference(reference: np.ndarray) -> Optional[np.ndarray]:
    if cv2.countNonZero(reference) == 0:
        return None
    outside = (reference == 0).astype(np.uint8)
    return cv2.distanceTransform(outside, cv2.DIST_L2, 5)


def _band_points(bands: Iterable[Dict[str, Any]]) -> List[Tuple[int, np.ndarray]]:
    result = []
    for index, band in enumerate(bands or []):
        if not isinstance(band, dict):
            continue
        points = np.asarray(band.get("centerline") or [], dtype=np.float64)
        if points.ndim == 2 and points.shape[0] and points.shape[1] >= 2:
            result.append((int(band.get("id", index)), points[:, :2]))
    return result


def _nearest_band(
    centroid: Tuple[float, float], bands: List[Tuple[int, np.ndarray]], mpp: float
) -> Tuple[Optional[int], Optional[float]]:
    if not bands:
        return None, None
    point = np.asarray(centroid, dtype=np.float64)
    candidates = [
        (float(np.min(np.linalg.norm(points - point, axis=1))), band_id)
        for band_id, points in bands
    ]
    distance, band_id = min(candidates)
    return int(band_id), float(distance * mpp)


def _component_records(
    layer_name: str,
    mask: np.ndarray,
    reference: np.ndarray,
    raw: np.ndarray,
    main_angle: float,
    mpp: float,
    offset: Tuple[int, int],
    bands: List[Tuple[int, np.ndarray]],
    config: Dict[str, Any],
    geo=None,
) -> List[Dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    distance_map = _distance_to_reference(reference)
    reference_interval = _projection_interval(reference, main_angle)
    attach_m = float(config.get("residual_body_attach_m", 0.12))
    attach_px = max(1, int(math.ceil(attach_m / max(mpp, 1e-9))))
    attachment_zone = cv2.dilate(
        reference,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * attach_px + 1, 2 * attach_px + 1)),
    ) if cv2.countNonZero(reference) else np.zeros_like(reference)
    minimum_attach_ratio = float(config.get("residual_body_min_attach_ratio", 0.02))
    end_guard_m = float(config.get("band_end_guard_m", 1.0))
    records: List[Dict[str, Any]] = []

    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < 3:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        selector = labels[y:y + height, x:x + width] == label
        local_ys, local_xs = np.where(selector)
        ys = local_ys + y
        xs = local_xs + x
        component_angle, major_px, minor_px = _component_geometry(xs, ys)
        centroid_x, centroid_y = map(float, centroids[label])
        global_x = centroid_x + int(offset[0])
        global_y = centroid_y + int(offset[1])
        lon = lat = None
        if geo is not None and getattr(geo, "is_ready", lambda: False)():
            try:
                lon, lat = map(float, geo.pixel_to_lonlat(global_x, global_y))
            except Exception:
                lon = lat = None

        distance_px = (
            float(np.min(distance_map[y:y + height, x:x + width][selector]))
            if distance_map is not None else None
        )
        attach_ratio = float(np.count_nonzero(
            selector & (attachment_zone[y:y + height, x:x + width] > 0)
        ) / area_px)
        raw_support_ratio = float(np.count_nonzero(
            selector & (raw[y:y + height, x:x + width] > 0)
        ) / area_px)
        direction = np.asarray(
            (math.cos(main_angle), math.sin(main_angle)), dtype=np.float64
        )
        projections = xs.astype(np.float64) * direction[0] + ys.astype(np.float64) * direction[1]
        component_interval = (
            (float(projections.min()), float(projections.max()))
            if projections.size else None
        )
        overhang_px = 0.0
        if reference_interval is not None and component_interval is not None:
            overhang_px = max(
                0.0,
                reference_interval[0] - component_interval[0],
                component_interval[1] - reference_interval[1],
            )
        nearest_band_id, nearest_band_distance_m = _nearest_band(
            (centroid_x, centroid_y), bands, mpp
        )

        flags = []
        distance_m = distance_px * mpp if distance_px is not None else None
        if layer_name == "body_residual" and (
            distance_m is None
            or distance_m > attach_m * 1.25
            or attach_ratio < minimum_attach_ratio
        ):
            flags.append("detached_body_residual")
        if layer_name == "gap_repair_added":
            flags.append("generated_by_gap_repair")
        if layer_name == "generated_regularization" and distance_m is not None and distance_m > attach_m:
            flags.append("generated_far_from_raw_support")
        overhang_m = overhang_px * mpp
        if layer_name in {"body_residual", "gap_repair_added", "generated_regularization"} and overhang_m > end_guard_m:
            flags.append("excessive_endpoint_overhang")
        if layer_name == "uncertain_residual":
            flags.append("uncertain_residual")

        records.append({
            "component_id": f"{layer_name}:{label}",
            "source_layer": layer_name,
            "area_px": area_px,
            "area_m2": float(area_px * mpp * mpp),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": width,
            "bbox_height": height,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "global_centroid_x": global_x,
            "global_centroid_y": global_y,
            "centroid_lon": lon,
            "centroid_lat": lat,
            "orientation_deg": math.degrees(component_angle),
            "main_angle_difference_deg": _angle_difference_deg(component_angle, main_angle),
            "major_length_m": float(major_px * mpp),
            "minor_length_m": float(minor_px * mpp),
            "elongation": float(major_px / max(minor_px, 1e-9)),
            "distance_to_reference_m": distance_m,
            "attachment_ratio": attach_ratio,
            "raw_support_ratio": raw_support_ratio,
            "endpoint_overhang_m": float(overhang_m),
            "nearest_band_id": nearest_band_id,
            "nearest_band_distance_m": nearest_band_distance_m,
            "review_required": bool(flags),
            "flags": flags,
        })
    return records


def _aligned_rgb(source_image: Any, raw: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    height, width = raw.shape[:2]
    if isinstance(source_image, np.ndarray) and source_image.ndim == 3:
        if source_image.shape[:2] == (height, width):
            return np.asarray(source_image[:, :, :3], dtype=np.uint8).copy()
        x, y = map(int, offset)
        if y >= 0 and x >= 0 and y + height <= source_image.shape[0] and x + width <= source_image.shape[1]:
            return np.asarray(source_image[y:y + height, x:x + width, :3], dtype=np.uint8).copy()
    gray = np.where(raw > 0, 95, 25).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _overview_png(
    rgb: np.ndarray,
    layers: Dict[str, np.ndarray],
    records: List[Dict[str, Any]],
    max_dim: int = 2400,
) -> bytes:
    canvas = rgb.copy()
    colors = {
        "regularized_body": (50, 190, 80),
        "body_residual": (235, 55, 55),
        "gap_repair_added": (220, 45, 210),
        "headland_corridor": (45, 105, 245),
        "uncertain_residual": (245, 190, 45),
        "headland": (55, 135, 235),
    }
    for name, color in colors.items():
        mask = layers.get(name)
        if mask is None:
            continue
        selector = mask > 0
        if np.any(selector):
            canvas[selector] = (
                canvas[selector].astype(np.float32) * 0.55
                + np.asarray(color, dtype=np.float32) * 0.45
            ).astype(np.uint8)

    scale = min(1.0, float(max_dim) / max(canvas.shape[:2]))
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    review_records = [item for item in records if item["review_required"]][:200]
    for index, record in enumerate(review_records, start=1):
        x = int(round(float(record["centroid_x"]) * scale))
        y = int(round(float(record["centroid_y"]) * scale))
        cv2.circle(canvas, (x, y), 7, (255, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 0, 0), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode diagnostic overview")
    return encoded.tobytes()


def _records_csv(records: List[Dict[str, Any]]) -> str:
    fields = [
        "component_id", "source_layer", "area_px", "area_m2",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        "centroid_x", "centroid_y", "global_centroid_x", "global_centroid_y",
        "centroid_lon", "centroid_lat", "orientation_deg",
        "main_angle_difference_deg", "major_length_m", "minor_length_m",
        "elongation", "distance_to_reference_m", "attachment_ratio",
        "raw_support_ratio", "endpoint_overhang_m", "nearest_band_id",
        "nearest_band_distance_m", "review_required", "flags",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        row = dict(record)
        row["flags"] = "|".join(record.get("flags") or [])
        writer.writerow({key: row.get(key) for key in fields})
    return stream.getvalue()


def export_mask_diagnostic_bundle(
    output_path: str,
    raw_mask: np.ndarray,
    mask_result: Dict[str, Any],
    *,
    offset: Tuple[int, int] = (0, 0),
    geo=None,
    config: Optional[Dict[str, Any]] = None,
    source_image: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a ZIP bundle that preserves stages and quantifies suspicious components."""
    if not isinstance(raw_mask, np.ndarray) or raw_mask.ndim != 2:
        raise ValueError("raw mask is missing or invalid")
    result = dict(mask_result or {})
    shape = raw_mask.shape[:2]
    raw = _binary(raw_mask, shape)
    processed = _binary(result.get("processed_mask"), shape)
    if cv2.countNonZero(processed) == 0:
        raise ValueError("processed mask is missing or empty")
    regularized = _binary(result.get("regularized_body_mask"), shape)
    body_residual = _binary(result.get("body_residual_mask"), shape)
    headland = _binary(result.get("headland_mask"), shape)
    uncertain = _binary(result.get("uncertain_residual_mask"), shape)
    gap_added = _binary(result.get("gap_repair_added_mask"), shape)
    support_removed = _binary(result.get("support_clip_removed_mask"), shape)
    headland_corridor = _binary(result.get("headland_corridor_mask"), shape)
    planning_support = _binary(result.get("planning_support_mask"), shape)
    after_residual_merge = cv2.bitwise_or(regularized, body_residual)
    generated_regularization = cv2.bitwise_and(regularized, cv2.bitwise_not(raw))
    generated_final = cv2.bitwise_and(processed, cv2.bitwise_not(raw))
    layers = {
        "raw": raw,
        "regularized_body": regularized,
        "body_residual": body_residual,
        "after_residual_merge": after_residual_merge,
        "gap_repair_added": gap_added,
        "support_clip_removed": support_removed,
        "headland_corridor": headland_corridor,
        "processed": processed,
        "headland": headland,
        "uncertain_residual": uncertain,
        "planning_support": planning_support,
        "generated_regularization": generated_regularization,
        "generated_final": generated_final,
    }

    diagnostics = dict(result.get("diagnostics") or {})
    mpp = float(diagnostics.get("meters_per_pixel", 0.0) or 0.0)
    if not math.isfinite(mpp) or mpp <= 0:
        raise ValueError("valid meters_per_pixel is required for mask diagnostics")
    main_angle = float(result.get("main_angle", 0.0) or 0.0)
    cfg = dict(config or {})
    bands = _band_points(result.get("all_bands") or [])
    records: List[Dict[str, Any]] = []
    for layer_name, layer, reference in (
        ("body_residual", body_residual, regularized),
        ("gap_repair_added", gap_added, after_residual_merge),
        ("generated_regularization", generated_regularization, raw),
        ("uncertain_residual", uncertain, regularized),
        ("headland", headland, regularized),
    ):
        records.extend(_component_records(
            layer_name, layer, reference, raw, main_angle, mpp,
            offset, bands, cfg, geo=geo,
        ))
    records.sort(key=lambda item: (not item["review_required"], -float(item["area_m2"])))

    manifest = {
        "schema": DIAGNOSTIC_SCHEMA,
        "stage": "mask_optimization",
        "run_label": str((metadata or {}).get("run_label", "") or ""),
        "provenance": dict(metadata or {}),
        "mask_shape": [int(shape[0]), int(shape[1])],
        "mask_offset_xy": [int(offset[0]), int(offset[1])],
        "meters_per_pixel": mpp,
        "main_angle_deg": math.degrees(main_angle),
        "algorithm_version": diagnostics.get("algorithm_version"),
        "strength": diagnostics.get("strength"),
        "thresholds": {
            "residual_body_attach_m": float(cfg.get("residual_body_attach_m", 0.12)),
            "residual_body_min_attach_ratio": float(cfg.get("residual_body_min_attach_ratio", 0.02)),
            "band_end_guard_m": float(cfg.get("band_end_guard_m", 1.0)),
        },
        "layer_pixels": {name: int(cv2.countNonZero(mask)) for name, mask in layers.items()},
        "component_count": len(records),
        "review_component_count": sum(bool(item["review_required"]) for item in records),
        "diagnostics": diagnostics,
    }

    target = os.path.abspath(output_path)
    if not target.lower().endswith(".zip"):
        target += ".zip"
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    temporary = target + ".tmp"
    layer_stream = io.BytesIO()
    np.savez_compressed(layer_stream, **layers)
    overview = _overview_png(_aligned_rgb(source_image, raw, offset), layers, records)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
            )
            archive.writestr(
                "components.json",
                json.dumps(records, ensure_ascii=False, indent=2, default=_json_default),
            )
            archive.writestr("components.csv", "\ufeff" + _records_csv(records))
            archive.writestr("layers.npz", layer_stream.getvalue())
            archive.writestr("overview.png", overview)
            archive.writestr(
                "README.txt",
                "红色为主体残差，紫色为补缝新增，黄色为不确定残差，蓝色为田头，绿色为规则化主体。\n"
                "深蓝色为端部田头走廊；该走廊与主体的交叉像素已从主作业区移入田头层。\n"
                "components.csv 中 review_required=true 的区域需要人工复核；诊断标记不会自动删除掩膜。\n"
                "manifest.json 的 provenance 记录源影像位置、模型、推理配置、掩膜处理配置和上下游指纹。\n",
            )
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return {"path": target, **manifest}
