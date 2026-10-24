# -*- coding: utf-8 -*-
"""用人工标定线评价/挑选中心线算法（"算法对齐人的直觉"的离线工具）。

问题背景
--------
前几轮改中心线算法都是在"没有人工真值"的前提下调参数：换一种取中点的方法，
图上看着好了，换一块田又偏了。本模块把问题换个问法——**同一批带上，几种
候选中心线里哪一种最接近人画的线**，并且把"人为什么改这条线"归类：

- ``centered``：算法线已经与人重合，不用改；
- ``uniform``：整条线平移即可对齐（中心估计有系统偏置）；
- ``bow``：偏移沿弧长平滑变化（带真的弯，或全局直线趋势没去干净）；
- ``off_band`` / ``step``：人把线放到了另一条带上 / 偏移换挡——这是**分割**
  问题，任何"取中点"的算法都修不了。

候选（都沿同一条算法中心线的法向逐站计算，保证可比）：

- ``current``：当前算法线（含重压中线后的结果）；
- ``pre_recenter``：重压中线之前的占用率中位数线（若诊断字段存在）；
- ``mask_run_mid``：原始掩膜横截 run 的中点；
- ``distance_ridge``：原始掩膜距离变换脊线（最大内切圆圆心）；
- ``contour_chord_mid``：带轮廓法向弦的中点（行带自身的几何中心）。

输入是一个"案例包"（case bundle）：应用里点「导出标定案例」生成，包含
带宽/类型/中心线/轮廓、人工线、米制比例与 mask.npz 路径。离线跑：

    python src/centerline_calibration.py --case <case.json> [--case ...] --out <dir>

输出：``calibration_report.json`` + ``calibration_report.md``。没有人工线时
不产生任何"精度提升"结论，只报告样本数与缺口。
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import operator_paths
import reference_lines

SCHEMA = "align-calibration-case-v1"
REPORT_SCHEMA = "align-calibration-report-v1"
DEFAULT_STATION_M = 0.25
DEFAULT_SAMPLE_M = 0.25
CANDIDATE_NAMES = (
    "current",
    "pre_recenter",
    "mask_run_mid",
    "distance_ridge",
    "contour_chord_mid",
)


# ---------------------------------------------------------------- 工具
def _as_polyline(points: Any) -> np.ndarray:
    array = operator_paths.as_points(points)
    return array


def _band_width(band: Dict[str, Any]) -> float:
    for key in ("width_m", "profile_width_m", "classification_width_m"):
        try:
            value = float(band.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 1.0


def _station_frame(
    centerline: Sequence[Sequence[float]],
    mpp: float,
    station_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿中心线等距取站点，返回 (站点, 单位法向, 弧长米)。"""
    base = _as_polyline(centerline)
    if base.size == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    step_px = max(float(station_m), 0.05) / max(float(mpp), 1e-9)
    stations = operator_paths.resample_polyline(base, step_px)
    normals = operator_paths.unit_normals(base)
    # 站点重新取法向后仍是同一组点序，直接按弧长插值法向即可
    arc_base = operator_paths.arc_lengths(base)
    arc_station = operator_paths.arc_lengths(stations)
    aligned = np.zeros((len(stations), 2), dtype=np.float64)
    for axis in range(2):
        aligned[:, axis] = np.interp(arc_station, arc_base, normals[:, axis])
    norm = np.linalg.norm(aligned, axis=1)
    norm[norm <= 1e-12] = 1.0
    aligned = aligned / norm[:, None]
    return stations, aligned, arc_station * float(mpp)


def _mask_runs(
    mask: np.ndarray,
    stations: np.ndarray,
    normals: np.ndarray,
    offsets_m: np.ndarray,
    mpp: float,
) -> List[Optional[Tuple[int, int]]]:
    """逐站沿法向采样掩膜，返回包含站点的那段白 run 的 (起, 止) 下标。"""
    height, width = mask.shape[:2]
    offset_px = offsets_m / max(float(mpp), 1e-9)
    runs: List[Optional[Tuple[int, int]]] = []
    for station, normal in zip(stations, normals):
        coords = station[None, :] + offset_px[:, None] * normal[None, :]
        xs = np.rint(coords[:, 0]).astype(np.int64)
        ys = np.rint(coords[:, 1]).astype(np.int64)
        inside_frame = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if int(inside_frame.sum()) < 5:
            runs.append(None)
            continue
        values = np.zeros(offset_px.size, dtype=bool)
        values[inside_frame] = mask[ys[inside_frame], xs[inside_frame]] > 0
        indices = np.flatnonzero(values)
        if indices.size == 0:
            runs.append(None)
            continue
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        zero_index = int(np.argmin(np.abs(offset_px)))
        chosen = None
        for group in groups:
            if group[0] <= zero_index <= group[-1]:
                chosen = (int(group[0]), int(group[-1]))
                break
        if chosen is None:
            # 站点落在缝里：取离站点最近、且不太远的那段
            best_distance = None
            for group in groups:
                distance = min(abs(offset_px[group[0]]), abs(offset_px[group[-1]]))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    chosen = (int(group[0]), int(group[-1]))
            if best_distance is not None and best_distance > 0.6 / max(mpp, 1e-9):
                chosen = None
        runs.append(chosen)
    return runs


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    return cv2.distanceTransform(binary, cv2.DIST_L2, 5)


def _contour_chord_runs(
    contour: np.ndarray,
    stations: np.ndarray,
    normals: np.ndarray,
    offsets_m: np.ndarray,
    mpp: float,
    *,
    pad_px: int = 40,
) -> List[Optional[Tuple[int, int]]]:
    """在带轮廓内部沿法向取弦，返回包含站点的那段区间下标。"""
    import cv2

    if contour.shape[0] < 3:
        return [None] * len(stations)
    x0, y0 = contour.min(axis=0)
    x1, y1 = contour.max(axis=0)
    ox = int(math.floor(x0)) - pad_px
    oy = int(math.floor(y0)) - pad_px
    width = int(math.ceil(x1 - x0)) + 2 * pad_px
    height = int(math.ceil(y1 - y0)) + 2 * pad_px
    if width <= 0 or height <= 0:
        return [None] * len(stations)
    raster = np.zeros((height, width), np.uint8)
    cv2.fillPoly(raster, [np.round(contour - [ox, oy]).astype(np.int32)], 1)
    offset_px = offsets_m / max(float(mpp), 1e-9)
    runs: List[Optional[Tuple[int, int]]] = []
    for station, normal in zip(stations, normals):
        coords = station[None, :] + offset_px[:, None] * normal[None, :]
        xs = np.rint(coords[:, 0] - ox).astype(np.int64)
        ys = np.rint(coords[:, 1] - oy).astype(np.int64)
        inside_frame = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        values = np.zeros(offset_px.size, dtype=bool)
        values[inside_frame] = raster[ys[inside_frame], xs[inside_frame]] > 0
        indices = np.flatnonzero(values)
        if indices.size < 3:
            runs.append(None)
            continue
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        zero_index = int(np.argmin(np.abs(offset_px)))
        chosen = None
        for group in groups:
            if group[0] <= zero_index <= group[-1]:
                chosen = (int(group[0]), int(group[-1]))
                break
        runs.append(chosen)
    return runs


def candidate_centerlines(
    band: Dict[str, Any],
    mask: np.ndarray,
    mpp: float,
    *,
    station_m: float = DEFAULT_STATION_M,
    sample_m: float = 0.02,
) -> Dict[str, List[Tuple[float, float]]]:
    """生成同一条带上的候选中心线（掩膜局部像素）。"""
    base = _as_polyline(band.get("centerline") or band.get("work_line"))
    if base.size < 2:
        return {}
    stations, normals, _arc = _station_frame(base, mpp, station_m)
    if len(stations) < 3:
        return {}
    width = _band_width(band)
    half = max(0.2, width / 2.0 + 0.45)
    offsets_m = np.arange(-half, half + 1e-9, max(float(sample_m), 0.01))
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("掩膜必须为二维数组")

    mask_runs = _mask_runs(mask_array, stations, normals, offsets_m, mpp)
    dt = _distance_transform(mask_array)
    height, width_px = mask_array.shape[:2]

    offsets_by_candidate: Dict[str, np.ndarray] = {
        "current": np.zeros(len(stations)),
        "mask_run_mid": np.full(len(stations), np.nan),
        "distance_ridge": np.full(len(stations), np.nan),
    }
    if band.get("centerline_pre_recenter") is not None:
        pre = _as_polyline(band.get("centerline_pre_recenter"))
        if pre.size >= 2:
            # 用重压前中心线到当前站点的横向距离作为候选偏移
            arcs, offsets = operator_paths.signed_offsets(
                base, pre, step=max(station_m, 0.05) / max(mpp, 1e-9)
            )
            offsets_m_pre = offsets * float(mpp)  # signed_offsets 返回像素，统一换成米
            offsets_by_candidate["pre_recenter"] = np.interp(
                operator_paths.arc_lengths(stations),
                arcs,
                offsets_m_pre,
            ) if offsets.size else np.zeros(len(stations))

    contour = band.get("contour")
    if contour is None:
        contour_array = np.zeros((0, 2), dtype=np.float64)
    else:
        contour_array = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    contour_runs = (
        _contour_chord_runs(contour_array, stations, normals, offsets_m, mpp)
        if contour_array.shape[0] >= 3 else [None] * len(stations)
    )
    offsets_by_candidate["contour_chord_mid"] = np.full(len(stations), np.nan)

    for index, run in enumerate(mask_runs):
        if run is None:
            continue
        low, high = run
        offsets_by_candidate["mask_run_mid"][index] = (
            offsets_m[low] + offsets_m[high]
        ) / 2.0
        coords = stations[index][None, :] + (
            offsets_m[low:high + 1] / max(mpp, 1e-9)
        )[:, None] * normals[index][None, :]
        xs = np.rint(coords[:, 0]).astype(np.int64)
        ys = np.rint(coords[:, 1]).astype(np.int64)
        if (
            np.all((xs >= 0) & (xs < width_px) & (ys >= 0) & (ys < height))
            and coords.shape[0] > 0
        ):
            values = dt[ys, xs]
            peak = int(np.argmax(values))
            if float(values[peak]) > 0:
                offsets_by_candidate["distance_ridge"][index] = offsets_m[low + peak]
    for index, run in enumerate(contour_runs):
        if run is None:
            continue
        offsets_by_candidate["contour_chord_mid"][index] = (
            offsets_m[run[0]] + offsets_m[run[1]]
        ) / 2.0

    candidates: Dict[str, List[Tuple[float, float]]] = {}
    for name, offsets in offsets_by_candidate.items():
        if name not in CANDIDATE_NAMES:
            continue
        filled = _fill_nan(offsets)
        if filled is None:
            continue
        shifted = stations + normals * filled[:, None] / max(mpp, 1e-9)
        candidates[name] = [(float(x), float(y)) for x, y in shifted]
    return candidates


def _fill_nan(values: np.ndarray) -> Optional[np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return None
    if valid.all():
        return values
    index = np.arange(values.size)
    return np.interp(index, index[valid], values[valid])


# ---------------------------------------------------------------- 评价
def measure(
    candidate: Sequence[Sequence[float]],
    truth: Sequence[Sequence[float]],
    mpp: float,
    *,
    sample_m: float = DEFAULT_SAMPLE_M,
) -> Dict[str, Any]:
    """人工线 → 候选线 的 Eθ / Elat（cm），与中心线评价同口径。"""
    truth_array = _as_polyline(truth)
    candidate_array = _as_polyline(candidate)
    if truth_array.size == 0 or candidate_array.size == 0:
        return {"valid": False}
    fit = reference_lines.fit_line(truth_array)
    result = reference_lines.compare_to_polyline(
        fit, candidate_array, mpp, sample_spacing_m=sample_m
    )
    result["valid"] = True
    result["elat_mean_cm"] = float(result["elat_mean_cm"])
    result["elat_max_cm"] = float(result["elat_max_cm"])
    return result


def evaluate_band(
    band: Dict[str, Any],
    operator_points: Sequence[Sequence[float]],
    mask: np.ndarray,
    mpp: float,
    *,
    station_m: float = DEFAULT_STATION_M,
) -> Dict[str, Any]:
    """一条带：候选生成 → 与人线比 → 偏移归类。"""
    candidates = candidate_centerlines(band, mask, mpp, station_m=station_m)
    truth = _as_polyline(operator_points)
    record: Dict[str, Any] = {
        "band_id": band.get("id"),
        "band_type": band.get("band_type"),
        "width_m": _band_width(band),
        "profile_width_m": band.get("profile_width_m"),
        "operator_point_count": int(len(truth)),
        "candidates": {},
    }
    if not candidates:
        record["status"] = "no_candidate"
        return record
    best_name = None
    best_value = None
    for name, points in candidates.items():
        metrics = measure(points, truth, mpp)
        record["candidates"][name] = metrics
        if metrics.get("valid") and (
            best_value is None or metrics["elat_mean_cm"] < best_value
        ):
            best_value = metrics["elat_mean_cm"]
            best_name = name
    record["best_candidate"] = best_name
    record["best_elat_mean_cm"] = best_value
    current = record["candidates"].get("current") or {}
    if current.get("valid") and best_value is not None:
        record["improvement_cm"] = float(current["elat_mean_cm"] - best_value)
    # 人相对"当前算法线"改了多少、属于哪类问题
    arc_px, offset_px = operator_paths.signed_offsets(
        band.get("centerline") or [], truth,
        step=max(station_m, 0.05) / max(float(mpp), 1e-9),
    )
    offsets_m = offset_px * float(mpp)
    record["offset_profile"] = {
        "summary": operator_paths.profile_summary(offsets_m),
        "classification": operator_paths.classify_offset_profile(
            offsets_m, _band_width(band)
        ),
        "model": operator_paths.fit_offset_model(arc_px * float(mpp), offsets_m),
    }
    return record


# ---------------------------------------------------------------- 汇总
def _bucket(band: Dict[str, Any], median_width: float) -> str:
    band_type = str(band.get("band_type") or "unknown")
    width = _band_width(band)
    outlier = ""
    if median_width > 0 and abs(width - median_width) > 0.18 * median_width:
        outlier = "+outlier"
    return f"{band_type}{outlier}"


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """候选对比 + 分桶挑选规则（附留一验证命中率）。"""
    valid = [item for item in records if item.get("best_candidate")]
    widths = [_band_width(item) for item in records if _band_width(item) > 0]
    median_width = float(np.median(widths)) if widths else 0.0
    per_candidate: Dict[str, List[float]] = {name: [] for name in CANDIDATE_NAMES}
    wins: Dict[str, int] = {name: 0 for name in CANDIDATE_NAMES}
    for item in valid:
        for name in CANDIDATE_NAMES:
            metrics = (item.get("candidates") or {}).get(name) or {}
            if metrics.get("valid"):
                per_candidate[name].append(float(metrics["elat_mean_cm"]))
        if item.get("best_candidate"):
            wins[item["best_candidate"]] = wins.get(item["best_candidate"], 0) + 1
    candidate_summary = {}
    for name, values in per_candidate.items():
        if not values:
            continue
        candidate_summary[name] = {
            "bands": len(values),
            "median_elat_cm": float(np.median(values)),
            "mean_elat_cm": float(np.mean(values)),
            "wins": int(wins.get(name, 0)),
        }
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in valid:
        buckets.setdefault(_bucket(item, median_width), []).append(item)
    policy = {}
    for name, items in buckets.items():
        choice = _majority_choice(items)
        hits = 0
        for item in items:
            others = [other for other in items if other is not item]
            predicted = _majority_choice(others).get("candidate") if others else None
            actual = item.get("best_candidate")
            if predicted and actual:
                predicted_cm = (
                    (item.get("candidates") or {}).get(predicted) or {}
                ).get("elat_mean_cm")
                actual_cm = (
                    (item.get("candidates") or {}).get(actual) or {}
                ).get("elat_mean_cm")
                if predicted_cm is not None and actual_cm is not None:
                    hits += int(float(predicted_cm) <= float(actual_cm) + 1.0)
        policy[name] = {
            **choice,
            "bands": len(items),
            "leave_one_out_hit_rate": (hits / len(items)) if items else 0.0,
            "median_width_m": float(np.median([_band_width(item) for item in items])),
        }
    return {
        "candidate_summary": candidate_summary,
        "policy": policy,
        "bands_evaluated": len(valid),
        "bands_total": len(records),
    }


def _majority_choice(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scores: Dict[str, List[float]] = {}
    for item in items:
        for name, metrics in (item.get("candidates") or {}).items():
            if metrics.get("valid"):
                scores.setdefault(name, []).append(float(metrics["elat_mean_cm"]))
    if not scores:
        return {"candidate": None, "median_elat_cm": None}
    ranked = sorted(
        ((name, float(np.median(values))) for name, values in scores.items()),
        key=lambda pair: pair[1],
    )
    return {"candidate": ranked[0][0], "median_elat_cm": ranked[0][1]}


def problem_breakdown(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in records:
        kind = str(
            (item.get("offset_profile") or {}).get("classification", {}).get("kind")
            or "unknown"
        )
        counts[kind] = counts.get(kind, 0) + 1
    return counts


# ---------------------------------------------------------------- 案例包
def load_mask_for_case(case: Dict[str, Any]) -> np.ndarray:
    mask_path = str(case.get("mask_path") or "")
    if not mask_path or not os.path.isfile(mask_path):
        raise ValueError("案例缺少可读的 mask_path：%s" % mask_path)
    with np.load(mask_path) as payload:
        key = "mask" if "mask" in payload else list(payload.keys())[0]
        return np.asarray(payload[key])


def evaluate_case(case: Dict[str, Any], mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if case.get("schema") != SCHEMA:
        raise ValueError("案例包 schema 不匹配: %r" % (case.get("schema"),))
    mask_array = mask if mask is not None else load_mask_for_case(case)
    mpp = float(case.get("mpp") or 0.0)
    if mpp <= 0:
        raise ValueError("案例缺少有效 mpp")
    bands = {}
    for item in case.get("bands") or []:
        record = dict(item or {})
        if record.get("id") is None:
            record["id"] = record.get("band_id")
        key = record.get("id")
        if key is None:
            continue
        bands[int(key)] = record
    records = []
    for line in case.get("operator_lines") or []:
        band_key = line.get("band_id")
        band = bands.get(int(band_key)) if band_key is not None else None
        if band is None:
            records.append({
                "band_id": band_key,
                "status": "band_not_found",
            })
            continue
        records.append(evaluate_band(band, line.get("points") or [], mask_array, mpp))
    return {
        "case_id": str(case.get("case_id") or ""),
        "tif_path": str(case.get("tif_path") or ""),
        "mpp": mpp,
        "bands": records,
    }


def build_report(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    case_summaries = []
    for case in cases:
        evaluated = evaluate_case(case)
        records.extend(evaluated["bands"])
        case_summaries.append({
            "case_id": evaluated["case_id"],
            "tif_path": evaluated["tif_path"],
            "operator_lines": len(case.get("operator_lines") or []),
        })
    summary = summarize(records)
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cases": case_summaries,
        "summary": summary,
        "problem_breakdown": problem_breakdown(records),
        "bands": records,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = ["# 中心线标定报告", ""]
    lines.append("生成时间：%s" % report.get("generated_at", ""))
    lines.append("")
    lines.append("## 样本")
    lines.append("")
    lines.append("- 案例：%d 个" % len(report.get("cases") or []))
    lines.append("- 带：%d 条（有效评价 %d 条）" % (
        summary.get("bands_total", 0), summary.get("bands_evaluated", 0)
    ))
    breakdown = report.get("problem_breakdown") or {}
    if breakdown:
        lines.append("- 人工纠偏类型：" + "、".join(
            f"{kind} {count}" for kind, count in sorted(breakdown.items())
        ))
    lines.append("")
    lines.append("## 候选中心线 vs 人工线（Elat 越小越接近人）")
    lines.append("")
    lines.append("| 候选 | 带数 | 中位 Elat (cm) | 平均 Elat (cm) | 最优次数 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, item in sorted(
        (summary.get("candidate_summary") or {}).items(),
        key=lambda pair: pair[1].get("median_elat_cm", 1e9),
    ):
        lines.append("| %s | %d | %.1f | %.1f | %d |" % (
            name,
            item.get("bands", 0),
            item.get("median_elat_cm", 0.0),
            item.get("mean_elat_cm", 0.0),
            item.get("wins", 0),
        ))
    lines.append("")
    lines.append("## 分桶选择规则（留一验证）")
    lines.append("")
    lines.append("| 分桶 | 带数 | 推荐候选 | 中位 Elat (cm) | 留一命中率 |")
    lines.append("|---|---:|---|---:|---:|")
    for name, item in sorted((summary.get("policy") or {}).items()):
        lines.append("| %s | %d | %s | %.1f | %.0f%% |" % (
            name,
            item.get("bands", 0),
            item.get("candidate") or "—",
            item.get("median_elat_cm") or 0.0,
            (item.get("leave_one_out_hit_rate") or 0.0) * 100.0,
        ))
    lines.append("")
    lines.append("> 留一命中率 = 用其余带选出的候选，在该带上的 Elat 不超过该带最优候选 +1 cm 的比例。")
    lines.append("> 样本少于 3 条带的桶只作参考，不能据此定算法。")
    lines.append("")
    lines.append("## 逐带明细")
    lines.append("")
    lines.append("| 带 | 类型 | 带宽 (m) | 当前 Elat (cm) | 最优候选 | 最优 Elat (cm) | 人工纠偏类型 |")
    lines.append("|---|---|---:|---:|---|---:|---|")
    for item in report.get("bands") or []:
        candidates = item.get("candidates") or {}
        current = candidates.get("current") or {}
        kind = ((item.get("offset_profile") or {}).get("classification") or {}).get("kind", "")
        lines.append("| %s | %s | %.2f | %s | %s | %s | %s |" % (
            item.get("band_id"),
            item.get("band_type") or "",
            float(item.get("width_m") or 0.0),
            "%.1f" % current["elat_mean_cm"] if current.get("valid") else "—",
            item.get("best_candidate") or "—",
            "%.1f" % item["best_elat_mean_cm"] if item.get("best_elat_mean_cm") is not None else "—",
            kind,
        ))
    lines.append("")
    lines.append("## 结论口径")
    lines.append("")
    lines.append("- 只有存在人工标定线的带才进入统计；没有人工线时报告不产生精度结论。")
    lines.append("- `off_band` / `step` 属分割问题：换取中心算法无效，应先修掩膜/带宽分割。")
    lines.append("- 分桶规则的样本数小于 3 时，只能作为下一轮采样的方向，不能写进论文结论。")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="用人工标定线评价中心线候选")
    parser.add_argument("--case", action="append", default=[], help="案例包 JSON（可重复）")
    parser.add_argument("--out", default="", help="输出目录（默认：当前目录下 calibration_report）")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.case:
        print("至少需要一个 --case 案例包", file=sys.stderr)
        return 2
    cases = []
    for path in args.case:
        with open(path, "r", encoding="utf-8") as handle:
            cases.append(json.load(handle))
    report = build_report(cases)
    out_dir = args.out or os.path.join(os.getcwd(), "calibration_report")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "calibration_report.json")
    md_path = os.path.join(out_dir, "calibration_report.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(report))
    print(md_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
