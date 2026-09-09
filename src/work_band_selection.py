"""Centerline catalog and operator selection helpers.

The planner still keeps the historical module name for cache compatibility,
but the user-facing object is a centerline: all candidate lines are generated
first, then the operator checks which lines may enter turn planning.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2


CATALOG_VERSION = "operator-work-band-catalog-v1"
CENTERLINE_CATALOG_VERSION = "operator-centerline-catalog-v1"


def _clean_line(line: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    points = np.asarray(list(line or []), dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] != 2
        or not np.all(np.isfinite(points))
    ):
        return []
    return [(float(x), float(y)) for x, y in points]


def _line_normal_position(line, main_angle: float) -> float:
    points = np.asarray(line, dtype=np.float64)
    normal = np.asarray(
        (-math.sin(float(main_angle)), math.cos(float(main_angle))),
        dtype=np.float64,
    )
    return float(np.median(points @ normal))


def _source_ids_for_line(region: Dict, line_index: int, line_count: int) -> List:
    declared = list(region.get("work_line_source_ids") or [])
    if len(declared) == line_count:
        value = declared[line_index]
        return list(value) if isinstance(value, (list, tuple, set)) else [value]
    band_ids = list(region.get("band_ids") or [])
    if len(band_ids) == line_count:
        return [band_ids[line_index]]
    return band_ids


def build_work_band_catalog(
    work_regions: Iterable[Dict],
    main_angle: float,
) -> Dict:
    """Number final planner work lines monotonically across the whole field."""
    records = []
    for region_order, region in enumerate(work_regions or []):
        lines = list(region.get("work_lines") or [])
        for line_index, raw_line in enumerate(lines):
            line = _clean_line(raw_line)
            if not line:
                continue
            records.append({
                "region_id": region.get("region_id"),
                "region_line_index": int(line_index),
                "region_order": int(region_order),
                "kind": region.get("kind"),
                "line_source": region.get("line_source"),
                "source_band_ids": _source_ids_for_line(
                    region, line_index, len(lines)
                ),
                "work_line": line,
                "normal_position": _line_normal_position(line, main_angle),
            })
    records.sort(key=lambda item: (
        round(float(item["normal_position"]), 6),
        item["region_order"],
        item["region_line_index"],
    ))
    for sequence, item in enumerate(records, start=1):
        item["sequence"] = int(sequence)
        item["id"] = f"work-band-{sequence:03d}"
        item["label"] = f"作业带 {sequence}"
    return {
        "algorithm_version": CATALOG_VERSION,
        "main_angle": float(main_angle),
        "count": len(records),
        "items": records,
    }


def resolve_work_band_selection(
    catalog: Dict,
    selected_ids: Optional[Iterable[str]],
) -> Tuple[List[Dict], Dict]:
    """Resolve one contiguous operator selection against the current catalog."""
    items = list((catalog or {}).get("items") or [])
    requested = list(dict.fromkeys(str(value) for value in (selected_ids or [])))
    if not requested:
        raise ValueError("至少选择一条作业带")
    by_id = {str(item.get("id")): item for item in items}
    unknown = [value for value in requested if value not in by_id]
    if unknown:
        raise ValueError("所选作业带不存在: " + "、".join(unknown))
    selected = sorted(
        (by_id[value] for value in requested),
        key=lambda item: int(item.get("sequence", 0)),
    )
    sequences = [int(item["sequence"]) for item in selected]
    return selected, {
        "algorithm_version": "operator-work-band-selection-v1",
        "selection_mode": "single" if len(selected) == 1 else "multi",
        "selected_ids": [item["id"] for item in selected],
        "selected_count": len(selected),
        "first_sequence": sequences[0],
        "last_sequence": sequences[-1],
    }


def filter_work_regions(work_regions: Iterable[Dict], selected_items: Iterable[Dict]) -> List[Dict]:
    """Keep selected region lines while preserving each region's source order."""
    selected_by_region = {}
    for item in selected_items or []:
        selected_by_region.setdefault(item.get("region_id"), {})[
            int(item["region_line_index"])
        ] = item

    filtered = []
    for region in work_regions or []:
        selected = selected_by_region.get(region.get("region_id"), {})
        if not selected:
            continue
        lines = list(region.get("work_lines") or [])
        indices = [index for index in range(len(lines)) if index in selected]
        if not indices:
            continue
        updated = dict(region)
        updated["work_lines"] = [lines[index] for index in indices]
        source_ids = [selected[index].get("source_band_ids", []) for index in indices]
        updated["work_line_source_ids"] = [
            values[0] if len(values) == 1 else list(values)
            for values in source_ids
        ]
        updated["selected_work_band_ids"] = [
            selected[index].get("id") for index in indices
        ]
        updated["line_count"] = len(indices)
        filtered.append(updated)
    return filtered


def apply_work_band_selection(
    work_regions: Iterable[Dict],
    main_angle: float,
    selected_ids: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict], Dict, Dict]:
    """Build one catalog and apply either default-all or an operator range."""
    regions = [dict(region) for region in (work_regions or [])]
    catalog = build_work_band_catalog(regions, main_angle)
    if not catalog["items"]:
        return [], catalog, {
            "algorithm_version": "operator-work-band-selection-v1",
            "selection_origin": "default_all",
            "selection_mode": "empty",
            "selected_ids": [],
            "selected_count": 0,
        }
    if selected_ids is None:
        requested = [item["id"] for item in catalog["items"]]
        origin = "default_all"
    else:
        requested = list(selected_ids)
        origin = "operator"
    selected, report = resolve_work_band_selection(catalog, requested)
    report["selection_origin"] = origin
    return filter_work_regions(regions, selected), catalog, report


def rasterize_work_band_coverage(
    catalog: Dict,
    selected_ids: Iterable[str],
    *,
    shape: Sequence[int],
    cutter_width_m: float,
    meters_per_pixel: float,
    support_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rasterize selected cutter swaths and clip them to planning support."""
    selected, _report = resolve_work_band_selection(catalog, selected_ids)
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("作业范围栅格尺寸无效")
    mpp = float(meters_per_pixel)
    cutter = float(cutter_width_m)
    if not math.isfinite(mpp) or mpp <= 0.0:
        raise ValueError("作业范围缺少有效米制比例")
    if not math.isfinite(cutter) or cutter <= 0.0:
        raise ValueError("割台宽度必须大于 0")
    coverage = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, int(round(cutter / mpp)))
    for item in selected:
        points = np.rint(np.asarray(item.get("work_line") or [], dtype=np.float64))
        if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 2:
            cv2.polylines(
                coverage,
                [points.astype(np.int32)],
                False,
                255,
                thickness,
                cv2.LINE_8,
            )
    if isinstance(support_mask, np.ndarray):
        if support_mask.shape[:2] != coverage.shape:
            raise ValueError("作业支撑层尺寸与目录不一致")
        coverage = cv2.bitwise_and(
            coverage,
            (np.asarray(support_mask) > 0).astype(np.uint8) * 255,
        )
    return coverage


def build_centerline_catalog(
    work_regions: Iterable[Dict],
    main_angle: float,
) -> Dict:
    """Build a stable catalog from every final planner centerline.

    ``work_band_catalog`` remains the serialized compatibility key, while new
    catalog items use centerline IDs and labels so the UI no longer presents
    selection as a pre-filtering "work-band" step.
    """
    legacy = build_work_band_catalog(work_regions, main_angle)
    catalog = dict(legacy)
    catalog["algorithm_version"] = CENTERLINE_CATALOG_VERSION
    items = []
    for item in legacy.get("items") or []:
        updated = dict(item)
        sequence = int(updated.get("sequence", len(items) + 1))
        updated["legacy_id"] = str(updated.get("id") or "")
        updated["id"] = f"centerline-{sequence:03d}"
        updated["label"] = f"中心线 {sequence}"
        updated["centerline"] = list(updated.get("work_line") or [])
        items.append(updated)
    catalog["items"] = items
    catalog["count"] = len(items)
    return catalog


def migrate_cached_centerline_catalog(catalog: Optional[Dict]) -> Tuple[Dict, Dict[str, str]]:
    """Migrate a pre-checkbox work-band cache to the centerline contract.

    Old project snapshots used ``work-band-NNN`` IDs and labels.  The geometry
    is still useful, but the desktop UI must expose it as centerlines and the
    next planner run must be able to resolve the old selection without
    silently changing its order.
    """
    source = dict(catalog or {})
    source_items = list(source.get("items") or [])
    if not source_items:
        return source, {}
    migrated = dict(source)
    migrated["algorithm_version"] = CENTERLINE_CATALOG_VERSION
    mapping = {}
    items = []
    for fallback_sequence, item in enumerate(source_items, start=1):
        updated = dict(item or {})
        sequence = int(updated.get("sequence", fallback_sequence) or fallback_sequence)
        old_id = str(updated.get("id") or f"work-band-{sequence:03d}")
        new_id = (
            old_id if old_id.startswith("centerline-")
            else f"centerline-{sequence:03d}"
        )
        mapping[old_id] = new_id
        updated["legacy_id"] = str(updated.get("legacy_id") or old_id)
        updated["id"] = new_id
        updated["label"] = f"中心线 {sequence}"
        updated["centerline"] = list(
            updated.get("centerline") or updated.get("work_line") or []
        )
        items.append(updated)
    migrated["items"] = items
    migrated["count"] = len(items)
    return migrated, mapping


def resolve_centerline_selection(
    catalog: Dict,
    selected_ids: Optional[Iterable[str]],
) -> Tuple[List[Dict], Dict]:
    """Resolve checkbox IDs, accepting old cached work-band IDs by sequence."""
    items = list((catalog or {}).get("items") or [])
    requested = list(dict.fromkeys(str(value) for value in (selected_ids or [])))
    if not requested:
        raise ValueError("至少选择一条中心线")
    by_id = {str(item.get("id")): item for item in items}
    by_legacy_id = {
        str(item.get("legacy_id")): item
        for item in items
        if item.get("legacy_id")
    }
    selected = []
    unknown = []
    for value in requested:
        item = by_id.get(value) or by_legacy_id.get(value)
        if item is None:
            unknown.append(value)
        elif item not in selected:
            selected.append(item)
    if unknown:
        raise ValueError("所选中心线不存在: " + "、".join(unknown))
    selected.sort(key=lambda item: int(item.get("sequence", 0)))
    sequences = [int(item.get("sequence", 0)) for item in selected]
    return selected, {
        "algorithm_version": "operator-centerline-selection-v1",
        "selection_mode": "single" if len(selected) == 1 else "multi",
        "selected_ids": [str(item.get("id")) for item in selected],
        "selected_count": len(selected),
        "first_sequence": sequences[0],
        "last_sequence": sequences[-1],
    }


def filter_centerline_regions(
    work_regions: Iterable[Dict],
    selected_items: Iterable[Dict],
) -> List[Dict]:
    """Keep selected centerlines while preserving region provenance/order."""
    selected_by_region = {}
    for item in selected_items or []:
        selected_by_region.setdefault(item.get("region_id"), {})[
            int(item["region_line_index"])
        ] = item

    filtered = []
    for region in work_regions or []:
        selected = selected_by_region.get(region.get("region_id"), {})
        if not selected:
            continue
        lines = list(region.get("work_lines") or [])
        indices = [index for index in range(len(lines)) if index in selected]
        if not indices:
            continue
        updated = dict(region)
        updated["work_lines"] = [lines[index] for index in indices]
        source_ids = [selected[index].get("source_band_ids", []) for index in indices]
        updated["work_line_source_ids"] = [
            values[0] if len(values) == 1 else list(values)
            for values in source_ids
        ]
        updated["selected_centerline_ids"] = [
            selected[index].get("id") for index in indices
        ]
        # Keep the historical field for old layout readers.
        updated["selected_work_band_ids"] = list(updated["selected_centerline_ids"])
        updated["line_count"] = len(indices)
        filtered.append(updated)
    return filtered


def apply_centerline_selection(
    work_regions: Iterable[Dict],
    main_angle: float,
    selected_ids: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict], Dict, Dict]:
    """Generate all centerlines first, then apply the checkbox selection."""
    regions = [dict(region) for region in (work_regions or [])]
    catalog = build_centerline_catalog(regions, main_angle)
    if not catalog["items"]:
        return [], catalog, {
            "algorithm_version": "operator-centerline-selection-v1",
            "selection_origin": "default_all",
            "selection_mode": "empty",
            "selected_ids": [],
            "selected_count": 0,
        }
    if selected_ids is None:
        requested = [item["id"] for item in catalog["items"]]
        origin = "default_all"
    else:
        requested = list(selected_ids)
        origin = "operator"
    selected, report = resolve_centerline_selection(catalog, requested)
    report["selection_origin"] = origin
    return filter_centerline_regions(regions, selected), catalog, report


def rasterize_centerline_coverage(
    catalog: Dict,
    selected_ids: Iterable[str],
    *,
    shape: Sequence[int],
    cutter_width_m: float,
    meters_per_pixel: float,
    support_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rasterize selected centerline cutter swaths for the validation target."""
    selected, _report = resolve_centerline_selection(catalog, selected_ids)
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("作业范围栅格尺寸无效")
    mpp = float(meters_per_pixel)
    cutter = float(cutter_width_m)
    if not math.isfinite(mpp) or mpp <= 0.0:
        raise ValueError("作业范围缺少有效米制比例")
    if not math.isfinite(cutter) or cutter <= 0.0:
        raise ValueError("割台宽度必须大于 0")
    coverage = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, int(round(cutter / mpp)))
    for item in selected:
        points = np.rint(
            np.asarray(item.get("centerline") or item.get("work_line") or [], dtype=np.float64)
        )
        if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 2:
            cv2.polylines(
                coverage,
                [points.astype(np.int32)],
                False,
                255,
                thickness,
                cv2.LINE_8,
            )
    if isinstance(support_mask, np.ndarray):
        if support_mask.shape[:2] != coverage.shape:
            raise ValueError("作业支撑层尺寸与目录不一致")
        coverage = cv2.bitwise_and(
            coverage,
            (np.asarray(support_mask) > 0).astype(np.uint8) * 255,
        )
    return coverage
