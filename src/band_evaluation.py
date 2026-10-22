# -*- coding: utf-8 -*-
"""行带识别与提线评价（离线工具）：三项指标计算与核对报告。

按批注#18 与论文口径计算（band-eval-v1）：
- 行带识别完整率 = 与人工标注真值匹配（行带对象 IoU ≥ 0.5）的识别行带数
  ÷ 真值行带总数 ×100%；
- 行带分类正确率 = 匹配成功的行带中宽/窄分类正确的比例 ×100%；
- 中心线可用率 = 通过履带通行校验且可生成作业路径的中心线数 ÷ 提线总数
  （S5 对宽行带提取的中心线总数）×100%。

管道（与桌面端"自动规划"一致，纯离线只读重算，不写任何工程文件）：
    cache/mask.npz + result.tif
      → process_mask（现行 v2 行对齐重建，config.json mask_processing）
      → path_planning 配置快照 + 语义掩膜注入（镜像 workers 的同名函数）
      → prepare_operator_centerline_plan（生成全部候选中心线，与 plan_path 同一步）
      → 逐条中心线 validate_footprints（与规划器同一函数、同一阈值）
      → 汇总三项指标 + 逐带/逐真值明细 + 附加率（可选叠加图）

真值标注：AnyLabeling/labelme JSON（多边形），标签用"宽行带/窄行带"（或
wide/narrow）。多边形坐标默认按"源影像像素坐标"解释（用 meta 的 display_scale
与 mask offset 换算到掩膜局部坐标）；若 JSON 的 imageWidth/Height 与掩膜尺寸
一致则按掩膜局部坐标直接使用；可用 --truth-basis 显式指定。

用法（game python，源码运行环境；先 unset PYTHONPATH PYTHONHOME）::

    python src/band_evaluation.py --cache-dir <项目缓存目录> --tif <result.tif> \
        [--truth <labelme.json 或目录>] [--out <输出目录>] [--iou 0.5]

输出评价包：summary.json / per_band.csv / per_truth.csv / README.md /
overlay.png（有真值时）。评价包默认写入缓存目录下独立子目录 band_eval/<时间戳>
（--out 可指定）；不修改工程代码与缓存内作业数据。

DECISION-029：行带级三项指标的独立离线复算（只读；与规划器同函数同阈值）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

BAND_EVALUATION_VERSION = "band-eval-v1"
DEFAULT_IOU_THRESHOLD = 0.5

WIDE_LABELS = {"宽行带", "宽", "wide", "wideband", "wide_band", "wide band"}
NARROW_LABELS = {"窄行带", "窄", "narrow", "narrowband", "narrow_band", "narrow band"}


# ─────────────────────────────────────────────────────────────
# 运行环境（定位 ALIGN 源码目录）
# ─────────────────────────────────────────────────────────────

def _ensure_align_src() -> str:
    """把 ALIGN/src 加入 sys.path（开发副本与工程内副本都可用）。"""
    candidates = [
        os.environ.get("ALIGN_SRC", ""),
        os.path.dirname(os.path.abspath(__file__)),
        "E:/zhlonly/project/ALIGN/src",
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "path_planner.py")):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    raise RuntimeError(
        "找不到 ALIGN 源码目录（含 path_planner.py）；请设置环境变量 ALIGN_SRC"
    )


_SRC_DIR = _ensure_align_src()


def _align_root() -> str:
    return os.path.dirname(_SRC_DIR)


def resolve_repo_config(section: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """读取仓库根 config.json 的指定段（与桌面端源码运行一致）。"""
    candidate = config_path or os.path.join(_align_root(), "config.json")
    with open(candidate, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    block = data.get(section)
    if not isinstance(block, dict) or not block:
        raise ValueError("配置文件缺少 %s 段: %s" % (section, candidate))
    return dict(block)


# ─────────────────────────────────────────────────────────────
# 输入装载（复用 centerline_evaluation 的坐标口径）
# ─────────────────────────────────────────────────────────────

def resolve_tif_path(cache_dir: str, explicit: Optional[str]) -> str:
    """解析 result.tif：显式路径 → 缓存记录路径 → 迁移后路径自动重映射。"""
    saved = {}
    state_path = os.path.join(cache_dir, "project_state.json")
    if os.path.isfile(state_path):
        saved = json.load(open(state_path, "r", encoding="utf-8"))
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    recorded = str(saved.get("tif_path") or "")
    if recorded:
        candidates.append(recorded)
        remapped = recorded.replace("\\tif周黄磊\\", "\\出差资源\\tif周黄磊\\")
        remapped = remapped.replace("/tif周黄磊/", "/出差资源/tif周黄磊/")
        if remapped != recorded:
            candidates.append(remapped)
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "找不到源影像 result.tif（尝试过: %s）；请用 --tif 指定"
        % "; ".join(candidates or ["无候选"])
    )


def load_inputs(cache_dir: str, tif_path: Optional[str]) -> Dict[str, Any]:
    from centerline_evaluation import load_project_inputs

    resolved = resolve_tif_path(cache_dir, tif_path)
    inputs = load_project_inputs(cache_dir, resolved)
    inputs["tif_resolved"] = resolved
    return inputs


# ─────────────────────────────────────────────────────────────
# 规划前配置（镜像 pyside6_app.workers 的两个函数，保持同口径）
# ─────────────────────────────────────────────────────────────

def path_planning_config_snapshot(runtime_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """镜像 workers.path_planning_config_snapshot（DECISION-013）。"""
    config = resolve_repo_config("path_planning")
    config.update(dict(runtime_config or {}))
    config.setdefault("headland_damage_weight", 1.0)
    config.setdefault("footprint_metric_policy", "report_only")
    return config


def planning_config_with_mask_evidence(config_snapshot, mask_result, state, shape) -> Dict[str, Any]:
    """镜像 workers.planning_config_with_mask_evidence（DECISION-009/013）。"""
    from footprint_planner import rasterize_forbidden_regions

    config = dict(config_snapshot or {})
    mask_result = dict(mask_result or {})
    config["headland_mask"] = mask_result.get("headland_mask")
    config["planning_support_mask"] = mask_result.get("planning_support_mask")
    config["neutral_support_mask"] = mask_result.get("neutral_support_mask")
    config["uncertain_mask"] = mask_result.get("uncertain_residual_mask")
    forbidden_confirmed = (
        getattr(state, "forbidden_regions_confirmed", False) is True
    )
    forbidden_mask = rasterize_forbidden_regions(shape, state)
    if forbidden_confirmed and forbidden_mask is None:
        forbidden_mask = np.zeros(shape[:2], dtype=np.uint8)
    config["forbidden_mask"] = forbidden_mask
    config["forbidden_mask_confirmed"] = forbidden_confirmed
    return config


# ─────────────────────────────────────────────────────────────
# 管线：处理 + 生成全部候选中心线
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    inputs: Dict[str, Any],
    mask_config: Optional[Dict[str, Any]] = None,
    path_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from mask_processor import process_mask
    from path_planner import prepare_operator_centerline_plan

    raw_mask = np.asarray(inputs["raw_mask"]).copy()
    inputs["state"].mask_raw = raw_mask
    mask_cfg = dict(mask_config) if mask_config else resolve_repo_config("mask_processing")
    processed = process_mask(
        raw_mask,
        geo=inputs["geo"],
        state=inputs["state"],
        config=dict(mask_cfg),
    )
    processed_mask = processed.get("processed_mask")
    main_angle = float(processed.get("main_angle", 0.0))
    wide_bands = list(processed.get("wide_bands") or [])
    all_bands = list(processed.get("all_bands") or [])

    cfg = path_planning_config_snapshot(path_config)
    cfg = planning_config_with_mask_evidence(
        cfg, processed, inputs["state"], np.asarray(processed_mask).shape
    )
    prepared = prepare_operator_centerline_plan(
        wide_bands,
        processed_mask,
        main_angle,
        geo=inputs["geo"],
        state=inputs["state"],
        config=cfg,
        detected_mask=raw_mask,
        all_bands=all_bands,
    )
    return {
        "raw_mask": raw_mask,
        "processed": processed,
        "processed_mask": processed_mask,
        "main_angle": main_angle,
        "wide_bands": wide_bands,
        "all_bands": all_bands,
        "path_config": cfg,
        "prepared": prepared,
    }


# ─────────────────────────────────────────────────────────────
# 逐线履带通行校验（与规划器同一函数、同一阈值）
# ─────────────────────────────────────────────────────────────

LINE_METRIC_KEYS = (
    "track_outside_field_pct",
    "track_outside_support_pct",
    "track_uncertain_overlap_pct",
    "track_forbidden_overlap_pct",
    "track_core_overlap_pct",
    "planned_target_geometric_coverage_pct",
    "rolling_canopy_pct",
)


def validate_single_line(
    line: Sequence[Sequence[float]],
    pipeline: Dict[str, Any],
    planned_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    from path_planner import _harvester_specs
    from footprint_planner import validate_footprints

    cfg = pipeline["path_config"]
    state = pipeline.get("state")
    metrics = validate_footprints(
        [[(float(x), float(y)) for x, y in line]],
        pipeline["processed_mask"],
        geo=pipeline.get("geo"),
        state=state,
        harvester=_harvester_specs(state),
        config=cfg,
        segment_types=["work"],
        detected_mask=pipeline["raw_mask"],
        planned_mask=planned_mask,
        headland_mask=cfg.get("headland_mask"),
        support_mask=cfg.get("planning_support_mask"),
        uncertain_mask=cfg.get("uncertain_mask"),
        forbidden_mask=cfg.get("forbidden_mask"),
    )
    issues = [
        str(item.get("raw") or item.get("description"))
        for item in (metrics.get("issues") or [])
        if isinstance(item, dict)
    ]
    return {
        "valid": bool(metrics.get("valid")),
        "issues": issues,
        "metrics": {key: metrics.get(key) for key in LINE_METRIC_KEYS},
    }


# ─────────────────────────────────────────────────────────────
# 真值（labelme/AnyLabeling 多边形）
# ─────────────────────────────────────────────────────────────

def normalize_band_type(label: str) -> Optional[str]:
    text = str(label or "").strip().lower()
    if not text:
        return None
    if any(token in text for token in ("窄", "narrow")):
        return "narrow"
    if any(token in text for token in ("宽", "wide")):
        return "wide"
    return None


def _labelme_files(truth_path: str) -> List[str]:
    if os.path.isdir(truth_path):
        return sorted(
            os.path.join(truth_path, name)
            for name in os.listdir(truth_path)
            if name.lower().endswith(".json")
        )
    return [truth_path]


def load_truth_polygons(
    truth_path: str,
    inputs: Dict[str, Any],
    basis: str = "auto",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """读取 labelme JSON → 掩膜局部坐标下的真值多边形列表。"""
    meta = inputs["meta"]
    offset = inputs["offset"]
    scale_x = float(meta.get("display_scale_x") or 1.0)
    scale_y = float(meta.get("display_scale_y") or 1.0)
    mask_h, mask_w = np.asarray(inputs["raw_mask"]).shape[:2]
    truths: List[Dict[str, Any]] = []
    notes: List[str] = []
    for path in _labelme_files(truth_path):
        try:
            doc = json.load(open(path, "r", encoding="utf-8"))
        except Exception as exc:
            notes.append("跳过无法解析的 JSON: %s (%s)" % (path, exc))
            continue
        shapes = doc.get("shapes")
        if not isinstance(shapes, list):
            notes.append("跳过非 labelme 格式: %s" % path)
            continue
        img_w = doc.get("imageWidth")
        img_h = doc.get("imageHeight")
        if basis == "auto":
            if img_w and img_h and abs(int(img_w) - mask_w) <= 2 and abs(int(img_h) - mask_h) <= 2:
                resolved = "mask"
            else:
                resolved = "source"
        else:
            resolved = basis
        for index, shape in enumerate(shapes):
            if str(shape.get("shape_type") or "polygon") not in ("polygon", "rectangle"):
                continue
            label = str(shape.get("label") or "")
            band_type = normalize_band_type(label)
            points = shape.get("points") or []
            if len(points) < 3:
                continue
            pts = np.asarray(points, dtype=np.float64)
            if resolved == "source":
                pts = pts / np.asarray((scale_x, scale_y), dtype=np.float64)
                pts = pts - np.asarray(offset, dtype=np.float64)
            elif resolved == "display":
                pts = pts - np.asarray(offset, dtype=np.float64)
            # mask 情况直接使用
            truths.append({
                "truth_id": "%s#%d" % (os.path.basename(path), index),
                "source_file": path,
                "label": label,
                "band_type": band_type,
                "points": pts.tolist(),
            })
        notes.append("真值文件 %s: %d 个多边形（坐标基准=%s）"
                     % (os.path.basename(path), len(shapes), resolved))
    return truths, notes


def _rasterize(points: np.ndarray, origin: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    canvas = np.zeros((size[1], size[0]), dtype=np.uint8)
    poly = np.round(points - origin).astype(np.int32)
    cv2.fillPoly(canvas, [poly], 255)
    return canvas


def _polygon_iou(points_a: np.ndarray, points_b: np.ndarray, margin: int = 2) -> float:
    """两个局部坐标多边形的 IoU（bbox 裁剪 + 栅格化）。"""
    ax0, ay0 = points_a.min(axis=0)
    ax1, ay1 = points_a.max(axis=0)
    bx0, by0 = points_b.min(axis=0)
    bx1, by1 = points_b.max(axis=0)
    x0 = int(math.floor(min(ax0, bx0))) - margin
    y0 = int(math.floor(min(ay0, by0))) - margin
    x1 = int(math.ceil(max(ax1, bx1))) + margin
    y1 = int(math.ceil(max(ay1, by1))) + margin
    w, h = max(x1 - x0, 2), max(y1 - y0, 2)
    if w * h > 4_000_000:
        return 0.0
    origin = np.asarray((x0, y0), dtype=np.float64)
    ra = _rasterize(points_a, origin, (w, h))
    rb = _rasterize(points_b, origin, (w, h))
    inter = int(np.count_nonzero((ra > 0) & (rb > 0)))
    union = int(np.count_nonzero((ra > 0) | (rb > 0)))
    return (inter / union) if union else 0.0


def match_truth_to_bands(
    truths: Sequence[Dict[str, Any]],
    bands: Sequence[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """贪心 IoU 匹配；返回 (逐真值结果, 未匹配真值 id, 未匹配行带 id)。"""
    band_records = []
    for band in bands:
        contour = band.get("contour")
        if contour is None:
            continue
        arr = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] < 3:
            continue
        band_records.append({"id": band.get("id"), "band_type": band.get("band_type"), "points": arr})

    pairs: List[Tuple[float, int, int]] = []
    for t_index, truth in enumerate(truths):
        t_points = np.asarray(truth["points"], dtype=np.float64)
        for b_index, band in enumerate(band_records):
            iou = _polygon_iou(t_points, band["points"])
            if iou > 0:
                pairs.append((iou, t_index, b_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_truth = set()
    matched_band = set()
    truth_result: Dict[int, Dict[str, Any]] = {}
    for iou, t_index, b_index in pairs:
        if t_index in matched_truth or b_index in matched_band:
            continue
        if iou < float(iou_threshold):
            break
        matched_truth.add(t_index)
        matched_band.add(b_index)
        truth_result[t_index] = {
            "matched_band_id": band_records[b_index]["id"],
            "matched_band_type": band_records[b_index]["band_type"],
            "iou": round(float(iou), 4),
        }
    rows: List[Dict[str, Any]] = []
    unmatched_truth_ids: List[str] = []
    for t_index, truth in enumerate(truths):
        match = truth_result.get(t_index)
        row = {
            "truth_id": truth["truth_id"],
            "label": truth.get("label") or "",
            "band_type": truth.get("band_type") or "",
            "matched_band_id": match["matched_band_id"] if match else None,
            "iou": match["iou"] if match else None,
            "type_correct": (
                (match["matched_band_type"] == truth.get("band_type"))
                if (match and truth.get("band_type"))
                else None
            ),
        }
        rows.append(row)
        if match is None:
            unmatched_truth_ids.append(truth["truth_id"])
    unmatched_band_ids = [
        band_records[index]["id"]
        for index in range(len(band_records))
        if index not in matched_band
    ]
    return rows, unmatched_truth_ids, unmatched_band_ids


# ─────────────────────────────────────────────────────────────
# 主评价流程
# ─────────────────────────────────────────────────────────────

def _finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _harvester_specs_snapshot(state) -> Dict[str, Any]:
    """机具几何参数快照（与规划器同源）。"""
    from path_planner import _harvester_specs

    specs = dict(_harvester_specs(state))
    return {
        key: (_finite_or_none(value) if isinstance(value, (int, float)) else value)
        for key, value in specs.items()
    }


def compute_channel_geometry(
    all_bands: Sequence[Dict[str, Any]],
    main_angle: float,
    mpp: float,
    specs: Dict[str, Any],
) -> Tuple[Dict[Any, Dict[str, Any]], Dict[str, Any]]:
    """按相邻行带边界配对推导通道净宽，并计算履带接地投影余量（对照交底书 S6(2) 判据）。

    口径与 machine_recommendation._corridor_geometry 一致：相邻两行带法向间距
    （质心投影，米制）减去两者半宽之和即为通道净宽；履带接地投影=与航向对齐的
    矩形（宽=履带宽度、按履带中心轨距对称布置），余量=通道边界到履带外/内缘的
    净距（正值=在通道内）。宽行带两侧通道均完整容纳履带投影时判为满足。
    """
    normal = np.asarray(
        (-math.sin(float(main_angle)), math.cos(float(main_angle))),
        dtype=np.float64,
    )
    items: List[Dict[str, Any]] = []
    for band in all_bands or []:
        centroid = band.get("centroid")
        if centroid is None:
            continue
        width = None
        for key in ("classification_width_m", "width_m", "profile_width_m"):
            value = _finite_or_none(band.get(key))
            if value is not None and value > 0:
                width = float(value)
                break
        if width is None:
            continue
        position = float(
            np.dot(np.asarray(centroid, dtype=np.float64)[:2], normal)
        ) * float(mpp)
        items.append({
            "id": band.get("id"),
            "band_type": band.get("band_type"),
            "pos": position,
            "width": width,
        })
    items.sort(key=lambda item: item["pos"])
    gauge_half = 0.5 * float(specs.get("track_gauge_m", 1.3))
    track_half = 0.5 * float(specs.get("track_width_m", 0.35))
    lane_inner = gauge_half - track_half
    lane_outer = gauge_half + track_half
    records: Dict[Any, Dict[str, Any]] = {}
    clears: List[float] = []
    for index, item in enumerate(items):
        if item["band_type"] != "wide":
            continue
        record: Dict[str, Any] = {"band_id": item["id"], "containment": None}
        sides: List[Dict[str, Any]] = []
        for side, neighbor in (
            ("left", items[index - 1] if index > 0 else None),
            ("right", items[index + 1] if index + 1 < len(items) else None),
        ):
            if neighbor is None:
                record[side] = None
                continue
            spacing = abs(float(neighbor["pos"]) - float(item["pos"]))
            clear = spacing - 0.5 * (float(neighbor["width"]) + float(item["width"]))
            margin_inner = lane_inner - 0.5 * float(item["width"])
            margin_outer = 0.5 * float(item["width"]) + clear - lane_outer
            side_result = {
                "neighbor_id": neighbor["id"],
                "clear_m": round(float(clear), 4),
                "margin_inner_m": round(float(margin_inner), 4),
                "margin_outer_m": round(float(margin_outer), 4),
                "contained": bool(margin_inner >= 0.0 and margin_outer >= 0.0),
            }
            record[side] = side_result
            sides.append(side_result)
            clears.append(float(clear))
        record["containment"] = bool(sides) and all(item2["contained"] for item2 in sides)
        records[item["id"]] = record
    summary = {
        "evaluated_band_count": len(records),
        "contained_band_count": sum(
            1 for record in records.values() if record.get("containment")
        ),
        "channel_clear_median_m": (
            round(float(np.median(clears)), 4) if clears else None
        ),
        "channel_clear_min_m": (round(float(np.min(clears)), 4) if clears else None),
    }
    summary["contained_pct"] = (
        round(
            100.0 * summary["contained_band_count"] / summary["evaluated_band_count"],
            3,
        )
        if summary["evaluated_band_count"]
        else None
    )
    return records, summary


def evaluate_case(
    cache_dir: str,
    tif_path: Optional[str] = None,
    truth_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    mask_config: Optional[Dict[str, Any]] = None,
    path_config: Optional[Dict[str, Any]] = None,
    truth_basis: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    started = time.perf_counter()
    if verbose:
        print("[1/4] 装载缓存与影像: %s" % cache_dir)
    inputs = load_inputs(cache_dir, tif_path)
    state_path = os.path.join(cache_dir, "project_state.json")
    saved_state = (
        json.load(open(state_path, "r", encoding="utf-8"))
        if os.path.isfile(state_path)
        else {}
    )
    inputs["state"].forbidden_regions = saved_state.get("forbidden_regions") or []
    inputs["state"].forbidden_regions_confirmed = bool(
        saved_state.get("forbidden_regions_confirmed", False)
    )

    if verbose:
        print("[2/4] 重跑掩膜处理 + 生成全部候选中心线 ...")
    pipeline = run_pipeline(inputs, mask_config=mask_config, path_config=path_config)
    pipeline["state"] = inputs["state"]
    pipeline["geo"] = inputs["geo"]

    wide_bands = pipeline["wide_bands"]
    prepared = pipeline["prepared"]
    catalog = dict(prepared.get("catalog") or {})
    items = list(catalog.get("items") or [])
    layout = dict(prepared.get("layout") or {})
    selection_filter = dict(layout.get("selection_work_band_filter") or {})
    excluded_records = list(selection_filter.get("excluded") or [])
    excluded_by_id = {
        record.get("band_id"): list(record.get("reasons") or [])
        for record in excluded_records
    }
    recovery = dict(layout.get("coverage_gap_recovery") or {})
    recovered_ids = list(recovery.get("recovered_band_ids") or [])

    if verbose:
        print("    宽行带 %d，候选中心线 %d，筛选排除 %d，自动补带 %d"
              % (len(wide_bands), len(items), len(excluded_records), len(recovered_ids)))
        print("[3/4] 逐条中心线履带通行校验（%d 条）..." % len(items))

    from path_planner import build_band_mask

    band_masks: Dict[Any, Optional[np.ndarray]] = {}
    for band in wide_bands:
        band_id = band.get("id")
        try:
            band_masks[band_id] = build_band_mask(
                pipeline["processed_mask"].shape, [band]
            )
        except Exception:
            band_masks[band_id] = None
    line_results: List[Dict[str, Any]] = []
    for number, item in enumerate(items, start=1):
        line = item.get("centerline") or item.get("work_line") or []
        if not line:
            continue
        source_ids = list(item.get("source_band_ids") or [])
        planned_mask = band_masks.get(source_ids[0]) if source_ids else None
        result = validate_single_line(line, pipeline, planned_mask=planned_mask)
        line_results.append({
            "line_id": item.get("id"),
            "source_band_ids": list(item.get("source_band_ids") or []),
            "valid": result["valid"],
            "issues": result["issues"],
            "metrics": result["metrics"],
        })
        if verbose and number % 10 == 0:
            print("    ... %d/%d" % (number, len(items)))

    # 逐带归并：一条带可对应多条线；生成即计入可用率主口径，另记录逐线校验状态
    band_rows: List[Dict[str, Any]] = []
    lines_by_band: Dict[Any, List[Dict[str, Any]]] = {}
    for line in line_results:
        for band_id in (line["source_band_ids"] or [None]):
            lines_by_band.setdefault(band_id, []).append(line)
    generated_band_count = 0
    footprint_pass_band_count = 0
    for band in wide_bands:
        band_id = band.get("id")
        lines = lines_by_band.get(band_id, [])
        if lines:
            generated_band_count += 1
        passing = [line for line in lines if line["valid"]]
        if passing:
            footprint_pass_band_count += 1
        if lines:
            status = "generated"
        elif band_id in excluded_by_id:
            status = "excluded_selection"
        else:
            status = "no_line_generated"
        best = passing[0] if passing else (lines[0] if lines else None)
        band_rows.append({
            "band_id": band_id,
            "band_type": band.get("band_type"),
            "width_m": _finite_or_none(band.get("width_m")),
            "profile_width_m": _finite_or_none(band.get("profile_width_m")),
            "status": status,
            "footprint_pass": (None if not lines else bool(passing)),
            "line_ids": ";".join(str(line["line_id"]) for line in lines),
            "exclusion": ";".join(excluded_by_id.get(band_id, [])),
            "issues": ";".join(best["issues"]) if best else "",
            **(  # 指标列
                {key: _finite_or_none((best or {}).get("metrics", {}).get(key))
                 for key in LINE_METRIC_KEYS}
                if best else {key: None for key in LINE_METRIC_KEYS}
            ),
        })

    # 通道几何（对照交底书 S6(2) 判据）：相邻行带边界配对推导净宽 + 履带投影余量
    specs_snapshot = _harvester_specs_snapshot(inputs["state"])
    channel_records, channel_summary = compute_channel_geometry(
        pipeline["all_bands"],
        pipeline["main_angle"],
        float(inputs.get("mpp") or 0.0),
        specs_snapshot,
    )
    for row in band_rows:
        record = channel_records.get(row["band_id"]) or {}
        left = record.get("left") or {}
        right = record.get("right") or {}
        row["channel_left_m"] = left.get("clear_m")
        row["channel_right_m"] = right.get("clear_m")
        row["margin_inner_m"] = left.get("margin_inner_m")
        row["margin_outer_left_m"] = left.get("margin_outer_m")
        row["margin_outer_right_m"] = right.get("margin_outer_m")
        row["channel_containment"] = record.get("containment")

    issue_counts: Dict[str, int] = {}
    for line in line_results:
        for text in line["issues"]:
            key = str(text).split("=")[0].strip() or str(text)
            issue_counts[key] = issue_counts.get(key, 0) + 1

    wide_count = len(wide_bands)
    availability_pct = (
        100.0 * generated_band_count / wide_count if wide_count else None
    )
    strict_availability_pct = (
        100.0 * footprint_pass_band_count / wide_count if wide_count else None
    )

    # 真值指标（可选）
    truth_summary: Dict[str, Any] = {
        "truth_count": 0,
        "matched_count": 0,
        "completeness_pct": None,
        "typed_matched_count": 0,
        "type_correct_count": 0,
        "classification_accuracy_pct": None,
        "unmatched_truth_ids": [],
        "unmatched_band_ids": [],
        "notes": [],
    }
    truth_rows: List[Dict[str, Any]] = []
    if truth_path:
        if verbose:
            print("[4/4] 真值比对: %s" % truth_path)
        truths, notes = load_truth_polygons(truth_path, inputs, basis=truth_basis)
        truth_summary["notes"].extend(notes)
        truth_rows, unmatched_truth, unmatched_band = match_truth_to_bands(
            truths, pipeline["all_bands"], iou_threshold
        )
        matched = [row for row in truth_rows if row["matched_band_id"] is not None]
        typed = [row for row in matched if row["band_type"]]
        correct = [row for row in typed if row["type_correct"]]
        truth_summary.update({
            "truth_count": len(truths),
            "matched_count": len(matched),
            "completeness_pct": (
                100.0 * len(matched) / len(truths) if truths else None
            ),
            "typed_matched_count": len(typed),
            "type_correct_count": len(correct),
            "classification_accuracy_pct": (
                100.0 * len(correct) / len(typed) if typed else None
            ),
            "unmatched_truth_ids": unmatched_truth,
            "unmatched_band_ids": unmatched_band,
            "iou_threshold": float(iou_threshold),
        })
    elif verbose:
        print("[4/4] 未提供真值（--truth），完整率/分类正确率记为待实测")

    summary = {
        "version": BAND_EVALUATION_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": os.path.abspath(cache_dir),
        "tif_path": inputs["tif_resolved"],
        "mpp_m_per_px": _finite_or_none(inputs.get("mpp")),
        "main_angle_deg": round(math.degrees(pipeline["main_angle"]), 3),
        "validation_profile": pipeline["path_config"].get("validation_profile"),
        "harvester_specs": specs_snapshot,
        "wide_band_count": wide_count,
        "excluded_selection_count": len(excluded_records),
        "recovered_band_count": len(recovered_ids),
        "generated_line_count": len(line_results),
        "validation_failed_line_count": sum(1 for line in line_results if not line["valid"]),
        "usable_line_count": sum(1 for line in line_results if line["valid"]),
        "generated_band_count": generated_band_count,
        "footprint_pass_band_count": footprint_pass_band_count,
        "centerline_availability_pct": (
            round(availability_pct, 3) if availability_pct is not None else None
        ),
        "centerline_availability_strict_pct": (
            round(strict_availability_pct, 3)
            if strict_availability_pct is not None else None
        ),
        "channel_containment": channel_summary,
        "validation_issue_counts": dict(sorted(issue_counts.items())),
        "truth": truth_summary,
        "runtime_s": round(time.perf_counter() - started, 2),
        "notes": [
            "主口径＝可生成作业路径（进入作业线目录）的宽行带；与规划器实际生成行为一致。",
            "严格口径＝在生成基础上逐线履带足迹校验（footprint_planner.validate_footprints，"
            "profile=%s）无门禁问题；此类问题在系统内按'诊断＋人工确认＋正式导出闸门'处理，"
            "研究导出不阻断。" % pipeline["path_config"].get("validation_profile"),
            "通道几何按相邻行带边界配对推导（法向间距−半宽之和），为冠层级实测值；"
            "农艺标称通道宽度（如45cm）可能因冠层外扩而大于可见间隙。",
            "整条路线（含转弯衔接）的最终校验以桌面端规划结果为准；本工具校验的是每条中心线自身。",
        ],
    }

    # 输出评价包
    out_dir = out_dir or os.path.join(
        cache_dir, "band_eval", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_csv(os.path.join(out_dir, "per_band.csv"), band_rows, [
        "band_id", "band_type", "width_m", "profile_width_m", "status",
        "footprint_pass", "line_ids", "exclusion", "issues",
        "channel_left_m", "channel_right_m", "margin_inner_m",
        "margin_outer_left_m", "margin_outer_right_m", "channel_containment",
        *LINE_METRIC_KEYS,
    ])
    if truth_rows:
        _write_csv(os.path.join(out_dir, "per_truth.csv"), truth_rows, [
            "truth_id", "label", "band_type", "matched_band_id", "iou", "type_correct",
        ])
    _write_readme(os.path.join(out_dir, "README.md"), summary, band_rows)
    if truth_rows:
        _write_overlay(
            os.path.join(out_dir, "overlay.png"),
            pipeline["processed_mask"],
            pipeline["all_bands"],
            truth_rows,
            inputs,
            truth_path,
            truth_basis,
        )
    summary["out_dir"] = out_dir
    if verbose:
        print("完成（%.1fs）→ %s" % (summary["runtime_s"], out_dir))
    return summary


def _write_csv(path: str, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in columns})


def _write_readme(path: str, summary: Dict[str, Any], band_rows: Sequence[Dict[str, Any]]) -> None:
    channel = dict(summary.get("channel_containment") or {})
    lines = [
        "# 行带识别与提线评价（band-eval-v1）",
        "",
        "- 案例: `%s`" % summary["cache_dir"],
        "- 影像: `%s`" % summary["tif_path"],
        "- mpp: %s m/px；主方向: %s°" % (summary["mpp_m_per_px"], summary["main_angle_deg"]),
        "",
        "## 三项指标",
        "",
        "| 指标 | 数值 | 口径 |",
        "|---|---|---|",
        "| 中心线可用率（主口径） | %s | 生成作业路径的宽行带中心线数 ÷ 宽行带（提线）总数 |" % _fmt_pct(summary["centerline_availability_pct"]),
        "| 中心线可用率（严格口径） | %s | 主口径基础上逐线履带足迹校验无门禁问题 |" % _fmt_pct(summary.get("centerline_availability_strict_pct"), "—"),
        "| 履带通道包含判据（S6-2） | %s | 按冠层级实测通道净宽与履带投影余量判定（标称45cm通道另见备注） |" % _fmt_pct(channel.get("contained_pct"), "—"),
        "| 行带识别完整率 | %s | IoU≥%s 匹配的识别行带 ÷ 真值行带总数 |" % (
            _fmt_pct(summary["truth"]["completeness_pct"]),
            summary["truth"].get("iou_threshold", DEFAULT_IOU_THRESHOLD),
        ),
        "| 行带分类正确率 | %s | 匹配对中宽窄分类正确比例（真值须带宽/窄标签） |" % _fmt_pct(summary["truth"]["classification_accuracy_pct"]),
        "",
        "## 口径说明",
        "",
        "> **中心线可用率（主口径）**：S5 对每条宽行带各提取一条中心线（提线总数）；其中实际生成了作业路径的比例即为可用率——未生成者由对象级二次筛选排除（法向余量不足、边缘异常等），逐条原因见 per_band.csv。",
        "> **严格口径**：在生成基础上，再要求逐线履带足迹校验无门禁问题（越界/无作物支撑区/不确定区/禁行区）；当前系统对这类问题按'记录＋人工确认＋正式导出闸门'处理，研究模式不阻断。",
        "",
        "## 可用率拆解",
        "",
        "- 宽行带（提线）总数: %d；生成作业路径: %d；严格通过逐线校验: %d" % (
            summary["wide_band_count"],
            summary.get("generated_band_count", 0),
            summary.get("footprint_pass_band_count", 0),
        ),
        "- 生成候选中心线: %d（含等距区细分线）；单线校验未通过: %d" % (
            summary["generated_line_count"], summary["validation_failed_line_count"],
        ),
        "- 筛选排除（法向余量/边缘异常）: %d；自动补带: %d" % (
            summary["excluded_selection_count"], summary["recovered_band_count"],
        ),
        "- 通道净宽（冠层级实测）：中位 %s m、最小 %s m（评估 %s 条宽行带）" % (
            channel.get("channel_clear_median_m"), channel.get("channel_clear_min_m"),
            channel.get("evaluated_band_count"),
        ),
        "",
        "## 备注",
        "",
    ]
    lines.extend("- %s" % note for note in summary.get("notes", []))
    if summary["truth"]["notes"]:
        lines.append("")
        lines.extend("- %s" % note for note in summary["truth"]["notes"])
    lines.append("")
    lines.append("> 逐带明细见 per_band.csv；真值对照见 per_truth.csv；机器可读见 summary.json。")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _fmt_pct(value: Optional[float], pending: str = "待实测（未提供真值）") -> str:
    return pending if value is None else "%.2f%%" % value


def _write_overlay(
    path: str,
    processed_mask: np.ndarray,
    all_bands: Sequence[Dict[str, Any]],
    truth_rows: Sequence[Dict[str, Any]],
    inputs: Dict[str, Any],
    truth_path: str,
    truth_basis: str,
) -> None:
    mask = (np.asarray(processed_mask) > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    canvas = cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)
    for band in all_bands:
        contour = band.get("contour")
        if contour is None:
            continue
        arr = np.round(np.asarray(contour, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        cv2.polylines(canvas, [arr], True, (0, 0, 255), 3)
    try:
        truths, _ = load_truth_polygons(truth_path, inputs, basis=truth_basis)
        matched_ids = {
            row["truth_id"] for row in truth_rows if row["matched_band_id"] is not None
        }
        for truth in truths:
            color = (0, 180, 0) if truth["truth_id"] in matched_ids else (255, 0, 255)
            arr = np.round(np.asarray(truth["points"], dtype=np.float64)).astype(np.int32)
            cv2.polylines(canvas, [arr], True, color, 3)
    except Exception:
        pass
    scale = min(1.0, 2000.0 / max(h, w))
    if scale < 1.0:
        canvas = cv2.resize(
            canvas,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(path, canvas)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="行带识别与提线评价（band-eval-v1）")
    parser.add_argument("--cache-dir", required=True, help="项目缓存目录")
    parser.add_argument("--tif", default=None, help="result.tif 路径（缺省自动解析/重映射）")
    parser.add_argument("--truth", default=None, help="真值 labelme JSON 文件或目录")
    parser.add_argument("--truth-basis", default="auto",
                        choices=("auto", "source", "display", "mask"),
                        help="真值坐标基准（默认 auto：按 JSON 尺寸自动判断）")
    parser.add_argument("--out", default=None, help="输出目录（默认：缓存目录下 band_eval/<时间戳>）")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD,
                        help="行带匹配 IoU 阈值（默认 0.5）")
    parser.add_argument("--quiet", action="store_true", help="仅输出结果 JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    summary = evaluate_case(
        args.cache_dir,
        tif_path=args.tif,
        truth_path=args.truth,
        out_dir=args.out,
        iou_threshold=args.iou,
        truth_basis=args.truth_basis,
        verbose=not args.quiet,
    )
    if args.quiet:
        print(json.dumps({
            key: summary[key]
            for key in (
                "wide_band_count", "generated_band_count", "generated_line_count",
                "validation_failed_line_count", "footprint_pass_band_count",
                "centerline_availability_pct", "centerline_availability_strict_pct",
                "channel_containment", "truth", "out_dir",
            )
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
