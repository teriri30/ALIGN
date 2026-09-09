"""Export reproducible, tile-level diagnostics for AI inference.

The inference result is only useful for debugging when it can be tied back to
the exact source raster window, model bytes, preprocessing contract and every
tile's contribution.  This module deliberately exports those facts together
with small visual artefacts; it does not change the semantic mask.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import math
import os
import zipfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


INFERENCE_DIAGNOSTIC_SCHEMA = "reui.inference-diagnostic.v1"


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalise_records(records: Optional[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    result = []
    for record in records or []:
        if isinstance(record, Mapping):
            result.append(_jsonable(dict(record)))
    return result


def _record_fields(records: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    for record in records:
        for key in record.keys():
            key = str(key)
            if key not in fields:
                fields.append(key)
    return fields


def _records_csv(records: Sequence[Mapping[str, Any]]) -> str:
    fields = _record_fields(records)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or ["tile_index"])
    writer.writeheader()
    for record in records:
        row = dict(record)
        for key, value in row.items():
            if isinstance(value, (list, tuple, dict)):
                row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        writer.writerow({key: row.get(key) for key in (fields or ["tile_index"])})
    return stream.getvalue()


def _bounded_rgb(image: np.ndarray, max_dim: int = 4096) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("source image must be an RGB array")
    source = np.asarray(source[:, :, :3], dtype=np.uint8)
    longest = max(source.shape[:2])
    if longest <= max_dim:
        return source.copy()
    scale = float(max_dim) / float(longest)
    return cv2.resize(
        source,
        (max(1, int(round(source.shape[1] * scale))), max(1, int(round(source.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    offset: Tuple[int, int],
    color: Tuple[int, int, int] = (110, 255, 130),
    alpha: float = 0.45,
) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return
    canvas_h, canvas_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    x0, y0 = map(int, offset)
    if (mask_h, mask_w) == (canvas_h, canvas_w):
        x0 = y0 = 0
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(canvas_w, x0 + mask_w), min(canvas_h, y0 + mask_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    clipped = mask[src_y0:src_y0 + dst_y1 - dst_y0, src_x0:src_x0 + dst_x1 - dst_x0]
    roi = image[dst_y0:dst_y1, dst_x0:dst_x1]
    selected = clipped > 0
    if not np.any(selected):
        return
    colour = np.asarray(color, dtype=np.float32)
    roi[selected] = (
        roi[selected].astype(np.float32) * (1.0 - float(alpha))
        + colour * float(alpha)
    ).clip(0, 255).astype(np.uint8)


def _place_mask_on_canvas(
    mask: np.ndarray,
    offset: Tuple[int, int],
    canvas_shape: Tuple[int, int],
) -> np.ndarray:
    """Place a display-space mask on a full preview canvas.

    The returned canvas is only a visualisation helper.  It deliberately
    keeps the canonical mask unchanged and applies the same crop offset used
    by the application when drawing the mask in the viewer.
    """
    canvas_h, canvas_w = [int(value) for value in canvas_shape[:2]]
    placed = np.zeros((max(1, canvas_h), max(1, canvas_w)), dtype=np.uint8)
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        return placed
    mask_h, mask_w = mask.shape[:2]
    x0, y0 = [int(value) for value in offset]
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(canvas_w, x0 + mask_w), min(canvas_h, y0 + mask_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return placed
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    placed[dst_y0:dst_y1, dst_x0:dst_x1] = mask[
        src_y0:src_y0 + (dst_y1 - dst_y0),
        src_x0:src_x0 + (dst_x1 - dst_x0),
    ]
    return placed


def _clip_mask_to_boundary(
    mask_canvas: np.ndarray,
    boundary: Optional[Sequence[Sequence[float]]],
) -> np.ndarray:
    """Clip a display-space visualisation to the selected field polygon."""
    if not isinstance(mask_canvas, np.ndarray) or mask_canvas.ndim != 2:
        return mask_canvas
    points = np.asarray(boundary or [], dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return mask_canvas
    polygon = np.zeros(mask_canvas.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon, [np.rint(points[:, :2]).astype(np.int32)], 255)
    clipped = mask_canvas.copy()
    clipped[polygon == 0] = 0
    return clipped


def _draw_field_boundary(image: np.ndarray, boundary: Optional[Sequence[Sequence[float]]]) -> None:
    points = np.asarray(boundary or [], dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return
    cv2.polylines(
        image,
        [np.rint(points[:, :2]).astype(np.int32)],
        True,
        (60, 240, 255),
        3,
        cv2.LINE_AA,
    )


def _scale_points(
    boundary: Optional[Sequence[Sequence[float]]],
    scale_x: float,
    scale_y: float,
) -> List[List[float]]:
    return [
        [float(point[0]) * float(scale_x), float(point[1]) * float(scale_y)]
        for point in (boundary or [])
        if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2
    ]


def _tile_grid_image(
    source_image: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    mask_offset: Tuple[int, int],
    mask_shape: Tuple[int, int],
    field_boundary: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    canvas = np.asarray(source_image[:, :, :3], dtype=np.uint8).copy()
    _draw_field_boundary(canvas, field_boundary)
    sx = canvas.shape[1] / max(float(mask_shape[1] + max(0, int(mask_offset[0]))), 1.0)
    sy = canvas.shape[0] / max(float(mask_shape[0] + max(0, int(mask_offset[1]))), 1.0)
    # The app's preview and mask are normally in the same display coordinate
    # system.  The fallback factors only protect diagnostic export when a
    # caller supplies a bounded preview.
    if abs(sx - 1.0) < 0.02:
        sx = 1.0
    if abs(sy - 1.0) < 0.02:
        sy = 1.0
    ox, oy = map(int, mask_offset)
    for record in records:
        window = record.get("output_window_xywh")
        if not isinstance(window, (list, tuple)) or len(window) < 4:
            continue
        x, y, width, height = [float(value) for value in window[:4]]
        left = int(round((ox + x) * sx))
        top = int(round((oy + y) * sy))
        right = int(round((ox + x + width) * sx))
        bottom = int(round((oy + y + height) * sy))
        if right <= 0 or bottom <= 0 or left >= canvas.shape[1] or top >= canvas.shape[0]:
            continue
        accepted = int(record.get("accepted_instance_count", 0) or 0)
        skipped = bool(record.get("skipped_nodata", False))
        colour = (130, 130, 130) if skipped else ((70, 220, 90) if accepted else (50, 60, 240))
        cv2.rectangle(canvas, (left, top), (right, bottom), colour, 1, cv2.LINE_AA)
    return canvas


def _tile_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accepted = [record for record in records if int(record.get("accepted_instance_count", 0) or 0) > 0]
    skipped = [record for record in records if bool(record.get("skipped_nodata", False))]
    return {
        "tile_count": int(len(records)),
        "tiles_with_accepted_instances": int(len(accepted)),
        "tiles_without_accepted_instances": int(max(0, len(records) - len(accepted) - len(skipped))),
        "skipped_nodata_tiles": int(len(skipped)),
        "accepted_instance_count": int(sum(int(record.get("accepted_instance_count", 0) or 0) for record in records)),
        "max_confidence": max(
            (float(record.get("max_confidence", 0.0) or 0.0) for record in records),
            default=0.0,
        ),
    }


def export_inference_diagnostic_bundle(
    output_path: str,
    source_image: np.ndarray,
    raw_mask: np.ndarray,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    mask_offset: Tuple[int, int] = (0, 0),
    field_boundary: Optional[Sequence[Sequence[float]]] = None,
    tile_trace: Optional[Iterable[Mapping[str, Any]]] = None,
    context_tile_trace: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Write a self-contained inference review bundle.

    The ZIP keeps the source/model identity and the tile-level evidence next
    to the mask.  It is intentionally a review artefact, not a replacement
    for the canonical project cache or for a georeferenced raster.
    """
    if not isinstance(source_image, np.ndarray) or source_image.ndim != 3:
        raise ValueError("source display image is missing")
    if not isinstance(raw_mask, np.ndarray) or raw_mask.ndim != 2:
        raise ValueError("raw inference mask is missing")
    original_source = np.asarray(source_image[:, :, :3], dtype=np.uint8)
    source = _bounded_rgb(original_source)
    raw = (np.asarray(raw_mask) > 0).astype(np.uint8) * 255
    meta = _jsonable(dict(metadata or {}))
    primary_records = _normalise_records(tile_trace)
    context_records = _normalise_records(context_tile_trace)
    offset = (int(mask_offset[0]), int(mask_offset[1]))
    field = _jsonable(field_boundary or [])
    mask_shape = [int(raw.shape[0]), int(raw.shape[1])]
    preview_scale_x = source.shape[1] / max(float(original_source.shape[1]), 1.0)
    preview_scale_y = source.shape[0] / max(float(original_source.shape[0]), 1.0)
    preview_offset = (
        int(round(offset[0] * preview_scale_x)),
        int(round(offset[1] * preview_scale_y)),
    )
    preview_boundary = _scale_points(field_boundary, preview_scale_x, preview_scale_y)
    preview_raw = raw
    if abs(preview_scale_x - 1.0) > 1e-6 or abs(preview_scale_y - 1.0) > 1e-6:
        preview_raw = cv2.resize(
            raw,
            (
                max(1, int(round(raw.shape[1] * preview_scale_x))),
                max(1, int(round(raw.shape[0] * preview_scale_y))),
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    manifest: Dict[str, Any] = {
        "schema": INFERENCE_DIAGNOSTIC_SCHEMA,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_label": str(meta.get("run_label", "") or ""),
        "stage": "inference",
        "source_image": meta.get("source_image", {}),
        "model": meta.get("model", {}),
        "image_position": {
            **dict(meta.get("image_position", {}) or {}),
            "mask_shape": mask_shape,
            "mask_offset_xy_display": [offset[0], offset[1]],
            "field_boundary_display": field,
        },
        "inference": meta.get("inference", {}),
        "preprocessing": meta.get("preprocessing", {}),
        "runtime": meta.get("runtime", {}),
        "stage_provenance": meta.get("stage_provenance", {}),
        "mask_statistics": {
            "shape": mask_shape,
            "nonzero_pixels": int(np.count_nonzero(raw)),
            "area_ratio": float(np.count_nonzero(raw) / max(raw.size, 1)),
        },
        "tile_summary": {
            "primary": _tile_summary(primary_records),
            "context": _tile_summary(context_records),
        },
        "metadata": meta,
    }

    overlay = source.copy()
    _blend_mask(overlay, preview_raw, preview_offset, (105, 255, 130), 0.45)
    _draw_field_boundary(overlay, preview_boundary)
    # Keep the raw overlay untouched for model recall debugging, and provide
    # a second display artefact that cannot be mistaken for a geospatial
    # offset when the inference crop extends beyond the selected field.
    placed_preview = _place_mask_on_canvas(
        preview_raw,
        preview_offset,
        source.shape[:2],
    )
    field_preview = _clip_mask_to_boundary(placed_preview, preview_boundary)
    field_overlay = source.copy()
    _blend_mask(field_overlay, field_preview, (0, 0), (105, 255, 130), 0.45)
    _draw_field_boundary(field_overlay, preview_boundary)
    tile_grid = _tile_grid_image(
        source,
        primary_records,
        preview_offset,
        preview_raw.shape[:2],
        preview_boundary,
    )
    files: Dict[str, bytes] = {}

    def encode_rgb(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("failed to encode inference diagnostic image")
        return encoded.tobytes()

    files["source_preview.png"] = encode_rgb(source)
    files["raw_overlay.png"] = encode_rgb(overlay)
    files["field_clipped_overlay.png"] = encode_rgb(field_overlay)
    files["tile_grid.png"] = encode_rgb(tile_grid)
    mask_stream = io.BytesIO()
    np.savez_compressed(mask_stream, raw_mask=raw)
    files["raw_mask.npz"] = mask_stream.getvalue()
    ok, encoded = cv2.imencode(".png", raw)
    if not ok:
        raise RuntimeError("failed to encode raw inference mask")
    files["raw_mask.png"] = encoded.tobytes()
    files["tile_trace.json"] = json.dumps(
        primary_records, ensure_ascii=False, indent=2, default=_json_default
    ).encode("utf-8")
    files["tile_trace.csv"] = ("\ufeff" + _records_csv(primary_records)).encode("utf-8")
    if context_records:
        files["context_tile_trace.json"] = json.dumps(
            context_records, ensure_ascii=False, indent=2, default=_json_default
        ).encode("utf-8")
        files["context_tile_trace.csv"] = (
            "\ufeff" + _records_csv(context_records)
        ).encode("utf-8")
    files["README.txt"] = (
        "reUI 推理诊断包\n\n"
        "source_preview.png：当前显示坐标下的源影像预览。\n"
        "raw_overlay.png：源影像与原始推理掩膜叠加；黄色边界为田块圈选范围。\n"
        "field_clipped_overlay.png：按田块圈选边界裁剪后的对齐效果图；仅用于显示，不修改原始掩膜。\n"
        "tile_grid.png：主切片输出窗口；绿色表示该切片至少有一个达到主置信度的实例，红色表示没有通过主置信度。\n"
        "tile_trace.csv/json：每个切片的源影像位置、输出位置、实例数、最高置信度和掩膜像素数。\n"
        "manifest.json：源影像路径/指纹、模型路径/指纹、推理配置、预处理、运行时和坐标位置。\n"
        "本包用于复盘，不会自动修改项目缓存，也不代表田间真值。\n"
    ).encode("utf-8")
    manifest["files"] = sorted(["manifest.json", *files.keys()])

    target = os.path.abspath(output_path)
    if not target.lower().endswith(".zip"):
        target += ".zip"
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    temporary = target + ".tmp"
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
            )
            for name, payload in files.items():
                archive.writestr(name, payload)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return {"path": target, **manifest}
