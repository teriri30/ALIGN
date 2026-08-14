"""Build review artifacts for a controlled field-trial route export."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import save_json_atomic
from mask_diagnostics import export_mask_diagnostic_bundle


PACKAGE_SCHEMA = "reui.field-trial-package.v1"


def _write_rgb(path: str, image: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write image: {path}")


def _write_mask(path: str, mask: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not cv2.imwrite(path, binary):
        raise OSError(f"failed to write mask: {path}")


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    offset: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.45,
):
    if not isinstance(mask, np.ndarray) or mask.ndim < 2:
        return
    image_h, image_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    x0, y0 = (0, 0) if (mask_h, mask_w) == (image_h, image_w) else offset
    dst_x0, dst_y0 = max(0, int(x0)), max(0, int(y0))
    dst_x1 = min(image_w, int(x0) + mask_w)
    dst_y1 = min(image_h, int(y0) + mask_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - int(x0), dst_y0 - int(y0)
    clipped = mask[
        src_y0:src_y0 + (dst_y1 - dst_y0),
        src_x0:src_x0 + (dst_x1 - dst_x0),
    ]
    roi = image[dst_y0:dst_y1, dst_x0:dst_x1]
    color_array = np.asarray(color, dtype=np.float32)
    for row in range(0, clipped.shape[0], 256):
        selected = clipped[row:row + 256] > 0
        if not np.any(selected):
            continue
        part = roi[row:row + 256]
        part[selected] = (
            part[selected].astype(np.float32) * (1.0 - alpha)
            + color_array * alpha
        ).clip(0, 255).astype(np.uint8)


def _points(value: Any) -> np.ndarray:
    points = np.asarray(value or [], dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=np.int32)
    return np.rint(points[:, :2]).astype(np.int32)


def _draw_polygons(image: np.ndarray, polygons: Iterable[Sequence], color, width=2):
    for polygon in polygons or []:
        points = _points(polygon)
        if len(points) >= 3:
            cv2.polylines(image, [points], True, color, width, cv2.LINE_AA)


def _render_path_preview(
    source_image: np.ndarray,
    mask_result: Mapping[str, Any],
    path_result: Mapping[str, Any],
    offset: Tuple[int, int],
    field_boundary: Sequence,
    forbidden_regions: Iterable[Sequence],
    service_points: Mapping[str, Any],
) -> np.ndarray:
    image = np.asarray(source_image).copy()
    processed = mask_result.get("processed_mask")
    if isinstance(processed, np.ndarray):
        _blend_mask(image, processed, offset, (120, 255, 140), 0.25)
    boundary = _points(field_boundary)
    if len(boundary) >= 3:
        cv2.polylines(image, [boundary], True, (70, 255, 120), 3, cv2.LINE_AA)
    _draw_polygons(image, forbidden_regions, (255, 70, 70), 3)
    segments = list(path_result.get("ordered_segments") or [])
    if not segments:
        segments.extend(dict(item, type="work") for item in path_result.get("tracks", []))
        segments.extend(dict(item, type="turn") for item in path_result.get("turns", []))
    for segment in segments:
        points = _points(segment.get("points"))
        if len(points) < 2:
            continue
        segment_type = str(segment.get("type", "work"))
        color = (70, 255, 110) if segment_type == "work" else (255, 165, 70)
        cv2.polylines(image, [points], False, color, 3, cv2.LINE_AA)
    for name, color in (("entry", (60, 220, 255)), ("exit", (255, 90, 90))):
        point = service_points.get(name)
        if point:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 7, color, -1, cv2.LINE_AA)
    for point in service_points.get("unload") or []:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 7, (255, 220, 70), -1, cv2.LINE_AA)
    return image


def export_visual_artifacts(
    output_dir: str,
    *,
    source_image: np.ndarray,
    raw_mask: np.ndarray,
    mask_result: Mapping[str, Any],
    path_result: Mapping[str, Any],
    offset: Tuple[int, int],
    field_boundary: Sequence,
    forbidden_regions: Iterable[Sequence],
    service_points: Mapping[str, Any],
    geo=None,
    mask_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    if not isinstance(source_image, np.ndarray) or source_image.ndim != 3:
        raise ValueError("source display image is missing")
    if not isinstance(raw_mask, np.ndarray) or raw_mask.ndim != 2:
        raise ValueError("raw segmentation mask is missing")
    processed = mask_result.get("processed_mask")
    if not isinstance(processed, np.ndarray) or processed.ndim != 2:
        raise ValueError("processed mask is missing")
    root = os.path.abspath(output_dir)
    os.makedirs(root, exist_ok=True)
    artifacts: Dict[str, str] = {}

    def register(relative: str) -> str:
        path = os.path.join(root, *relative.split("/"))
        artifacts[relative] = path
        return path

    _write_rgb(register("01_source_preview.png"), source_image)

    segmentation = source_image.copy()
    _blend_mask(segmentation, raw_mask, offset, (120, 255, 140))
    _write_rgb(register("02_ai_segmentation.png"), segmentation)
    del segmentation

    mask_preview = source_image.copy()
    headland = mask_result.get("headland_mask")
    uncertain = mask_result.get("uncertain_residual_mask")
    if isinstance(headland, np.ndarray):
        _blend_mask(mask_preview, headland, offset, (210, 230, 120))
    if isinstance(uncertain, np.ndarray):
        _blend_mask(mask_preview, uncertain, offset, (170, 170, 190))
    _blend_mask(mask_preview, processed, offset, (120, 255, 140))
    _write_rgb(register("03_mask_processed.png"), mask_preview)
    del mask_preview

    path_preview = _render_path_preview(
        source_image, mask_result, path_result, offset,
        field_boundary, forbidden_regions, service_points,
    )
    _write_rgb(register("04_path_planning.png"), path_preview)
    del path_preview

    _write_mask(register("masks/raw_mask.png"), raw_mask)
    _write_mask(register("masks/processed_mask.png"), processed)
    diagnostic_path = register("05_mask_diagnostics.zip")
    export_mask_diagnostic_bundle(
        diagnostic_path,
        raw_mask,
        dict(mask_result),
        offset=offset,
        geo=geo,
        config=mask_config,
        source_image=source_image,
    )
    save_json_atomic(register("06_validation.json"), {
        "validation": path_result.get("validation", {}),
        "machine_readiness": path_result.get("machine_readiness", {}),
        "turn_assessment": path_result.get("turn_assessment", {}),
        "planning_factors": path_result.get("planning_factors", {}),
    })
    readme_path = register("README.txt")
    with open(readme_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "reUI 田间试验资料包\n\n"
            "mission_field_trial.txt 为路径导入文件，使用前请按目标终端格式核对。\n"
            "同名 manifest 用于核对路径哈希、坐标规则、源影像、模型和分析参数。\n"
            "01-04 PNG 为显示坐标下的效果图，不是带地理标签的 GeoTIFF。\n"
            "masks 目录保留原始分割和处理后二值掩膜；诊断 ZIP 保存语义层和异常区域。\n"
            "原始 GeoTIFF 未复制到资料包，其绝对路径、SHA-256 和地理元数据记录在 manifest 中。\n"
        )
    return artifacts


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_package_index(output_dir: str) -> str:
    root = os.path.abspath(output_dir)
    records = []
    for directory, _subdirs, names in os.walk(root):
        for name in names:
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative == "package_index.json" or relative.endswith(".tmp"):
                continue
            records.append({
                "path": relative,
                "size_bytes": int(os.path.getsize(path)),
                "sha256": _sha256(path),
            })
    records.sort(key=lambda item: item["path"])
    target = os.path.join(root, "package_index.json")
    save_json_atomic(target, {
        "schema": PACKAGE_SCHEMA,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": records,
    })
    return target
