# -*- coding: utf-8 -*-
"""人工标定作业线（operator centerline）：数据契约、几何换算与纠偏模型。

为什么要有这一层
----------------
算法可以把线放在"它认为"的带中心，但唯一能判定"人是否认可"的证据是
**人自己画/拖出来的那条线**。本模块把人工标定线当作一等公民：

- 存哪：项目缓存的 sidecar ``operator_paths.json``（不进作业状态、不进撤销
  历史，与 ``reference_lines.json`` 同级）；
- 存什么：掩膜局部像素坐标的折线 + 每次拖动的横向偏移量（可审计）+ 产物
  指纹（掩膜算法版本 / 影像 mtime / 模型哈希）；
- 怎么用：规划前按 ``(centerline_id / legacy_id / source_band_ids)`` 替换
  算法中心线，路径、栅格化、编号全部跟着走；导出的路线因此就是人认可的线。

坐标口径（与 catalog 一致）：``points`` 是**掩膜局部像素**。界面上画线用的是
全幅显示坐标，等于局部坐标 + ``mask_offset_x/y``；转换只在界面层做，本模块
的几何函数一律吃局部坐标。

偏移量口径：沿基准线弧长 ``arc_m``（米）处的**有符号横向偏移**（米），正方向
取基准线前进方向的左侧（``normal = (-ty, tx)``）。一次拖动 = 一个高斯鼓包，
多个鼓包叠加成偏移剖面，端点不移动。
"""
from __future__ import annotations

import datetime
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "operator-centerlines-v1"
FILE_NAME = "operator_paths.json"
DEFAULT_SIGMA_M = 8.0          # 单次拖动的横向作用半宽（米）
DEFAULT_PROFILE_STEP_M = 0.25  # 偏移剖面的采样间距（米）
MIN_POINTS = 2


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 基础几何
def as_points(points: Any) -> np.ndarray:
    """把任意点列转成 (n,2) float 数组；无效时返回空数组。"""
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    if not np.all(np.isfinite(array)):
        return np.zeros((0, 2), dtype=np.float64)
    return array


def arc_lengths(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = as_points(points)
    if array.size == 0:
        return np.zeros(0, dtype=np.float64)
    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def point_at_arc(
    points: Sequence[Sequence[float]], arc: float
) -> Tuple[np.ndarray, np.ndarray]:
    """返回弧长 ``arc`` 处的位置与单位切向（线性插值）。"""
    array = as_points(points)
    if array.size == 0:
        return np.zeros(2), np.array([1.0, 0.0])
    cumulative = arc_lengths(array)
    total = float(cumulative[-1])
    arc = max(0.0, min(float(arc), total))
    index = int(np.searchsorted(cumulative, arc, side="right") - 1)
    index = max(0, min(index, len(array) - 2))
    span = float(cumulative[index + 1] - cumulative[index])
    ratio = 0.0 if span <= 1e-12 else (arc - cumulative[index]) / span
    position = array[index] + (array[index + 1] - array[index]) * ratio
    tangent = array[index + 1] - array[index]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        tangent = np.array([1.0, 0.0])
    else:
        tangent = tangent / norm
    return position, tangent


def unit_normals(points: Sequence[Sequence[float]]) -> np.ndarray:
    """逐点单位法向（中心差分）；``normal = (-ty, tx)``。"""
    array = as_points(points)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    tangents = np.gradient(array, axis=0)
    norm = np.linalg.norm(tangents, axis=1)
    norm[norm <= 1e-12] = 1.0
    tangents = tangents / norm[:, None]
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def resample_polyline(
    points: Sequence[Sequence[float]], step: float
) -> np.ndarray:
    """按弧长等间隔重采样（含两端），step 与坐标同单位。"""
    array = as_points(points)
    if array.size == 0:
        return array
    cumulative = arc_lengths(array)
    total = float(cumulative[-1])
    step = max(float(step), 1e-6)
    if total <= 1e-9:
        return array[:1].copy()
    count = max(2, int(math.floor(total / step)) + 1)
    samples = np.linspace(0.0, total, count)
    out = np.zeros((count, 2), dtype=np.float64)
    for index, arc in enumerate(samples):
        position, _tangent = point_at_arc(array, float(arc))
        out[index] = position
    return out


def nearest_arc(
    points: Sequence[Sequence[float]], target: Sequence[float]
) -> Tuple[float, float]:
    """返回基准线上距 ``target`` 最近的弧长与距离（与坐标同单位）。"""
    array = as_points(points)
    if array.size == 0:
        return 0.0, float("inf")
    point = np.asarray(target, dtype=np.float64).reshape(2)
    segment_start = array[:-1]
    segment_end = array[1:]
    ab = segment_end - segment_start
    ab_len2 = np.sum(ab * ab, axis=1)
    ab_len2[ab_len2 <= 1e-12] = 1e-12
    ap = point[None, :] - segment_start
    t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
    projection = segment_start + ab * t[:, None]
    distances = np.linalg.norm(projection - point[None, :], axis=1)
    index = int(np.argmin(distances))
    cumulative = arc_lengths(array)
    arc = float(cumulative[index] + t[index] * math.sqrt(ab_len2[index]))
    return arc, float(distances[index])


def signed_offsets(
    base_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
    *,
    step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """沿基准线等间隔采样，量到目标折线的有符号横向距离。

    正方向 = 基准线前进方向的左侧。返回 ``(arc_px, offset_px)``。
    """
    base = as_points(base_points)
    target = as_points(target_points)
    if base.size == 0 or target.size == 0:
        return np.zeros(0), np.zeros(0)
    cumulative = arc_lengths(base)
    total = float(cumulative[-1])
    step = max(float(step), 1e-6)
    count = max(2, int(math.floor(total / step)) + 1)
    samples = np.linspace(0.0, total, count)

    segment_start = target[:-1]
    segment_end = target[1:]
    ab = segment_end - segment_start
    ab_len2 = np.sum(ab * ab, axis=1)
    ab_len2[ab_len2 <= 1e-12] = 1e-12

    arcs_out = np.zeros(count, dtype=np.float64)
    offsets = np.zeros(count, dtype=np.float64)
    for index, arc in enumerate(samples):
        position, tangent = point_at_arc(base, float(arc))
        ap = position[None, :] - segment_start
        t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
        projection = segment_start + ab * t[:, None]
        delta = projection - position[None, :]
        distances = np.linalg.norm(delta, axis=1)
        best = int(np.argmin(distances))
        normal = np.array([-tangent[1], tangent[0]])
        offsets[index] = float(np.dot(delta[best], normal))
        arcs_out[index] = float(arc)
    return arcs_out, offsets


def offset_profile(
    length_px: float,
    edits: Iterable[Dict[str, Any]],
    mpp: float,
    *,
    step_px: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """把拖动记录叠成一条偏移剖面。

    两类手势（与路线编辑的习惯一致）：

    - ``kind="uniform"``：整条线平移，剖面为常数（拖动一下就能对齐整条带）；
    - ``kind="local"``：局部纠弯，剖面是以 ``arc_m`` 为中心、``sigma_m`` 为
      半宽的高斯鼓包（端部截断、不外推，线端位置不动）。

    返回 ``(arc_px, offset_px)``。
    """
    length_px = max(float(length_px), 0.0)
    mpp = max(float(mpp), 1e-9)
    count = max(2, int(length_px / max(step_px, 1e-6)) + 1)
    arc_px = np.linspace(0.0, length_px, count)
    arc_m = arc_px * mpp
    profile_m = np.zeros(count, dtype=np.float64)
    for edit in edits or []:
        if not isinstance(edit, dict):
            continue
        try:
            center = float(edit.get("arc_m", 0.0))
            amount = float(edit.get("offset_m", 0.0))
            sigma = abs(float(edit.get("sigma_m", DEFAULT_SIGMA_M)))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(center) or not math.isfinite(amount):
            continue
        kind = str(edit.get("kind") or "local").strip().lower()
        if kind == "uniform":
            profile_m += amount
            continue
        sigma = max(0.25, sigma)
        profile_m += amount * np.exp(-0.5 * ((arc_m - center) / sigma) ** 2)
    return arc_px, profile_m / mpp


def apply_offset_profile(
    base_points: Sequence[Sequence[float]],
    offsets_px: np.ndarray,
    max_offset_px: Optional[float] = None,
    *,
    source_arc: Optional[np.ndarray] = None,
) -> List[Tuple[float, float]]:
    """按逐点横向偏移平移基准线；偏移量可限幅，端点弧长位置不变。

    ``source_arc`` 给出偏移量对应的弧长（像素）；缺省时按点序等距处理。
    点列被简化过时（端点之间距离不等），必须传 ``source_arc``，否则偏移
    会落在错误的弧长上。
    """
    base = as_points(base_points)
    if base.size == 0:
        return []
    offsets = np.asarray(offsets_px, dtype=np.float64).reshape(-1)
    if offsets.size != len(base):
        if offsets.size == 0:
            offsets = np.zeros(len(base), dtype=np.float64)
        elif source_arc is not None and len(source_arc) == offsets.size:
            offsets = np.interp(
                arc_lengths(base),
                np.asarray(source_arc, dtype=np.float64).reshape(-1),
                offsets,
            )
        else:
            offsets = np.interp(
                np.linspace(0.0, 1.0, len(base)),
                np.linspace(0.0, 1.0, offsets.size),
                offsets,
            )
    if max_offset_px is not None and max_offset_px > 0:
        offsets = np.clip(offsets, -float(max_offset_px), float(max_offset_px))
    normals = unit_normals(base)
    moved = base + normals * offsets[:, None]
    return [(float(x), float(y)) for x, y in moved]


def path_from_edits(
    base_points: Sequence[Sequence[float]],
    edits: Iterable[Dict[str, Any]],
    mpp: float,
    *,
    width_m: float = 1.0,
    max_shift_ratio: float = 0.45,
) -> List[Tuple[float, float]]:
    """由基准线 + 拖动记录生成人工标定线（掩膜局部像素）。"""
    base = as_points(base_points)
    if base.size == 0:
        return []
    total_px = float(arc_lengths(base)[-1])
    arc_px, profile_px = offset_profile(total_px, edits, mpp)
    profile_on_line = np.interp(arc_lengths(base), arc_px, profile_px)
    limit_px = max(0.45 * max(float(width_m), 0.2), 0.60) / max(float(mpp), 1e-9)
    return apply_offset_profile(base, profile_on_line, max_offset_px=limit_px)


# ---------------------------------------------------------------- 偏移统计
def profile_summary(offsets_m: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(offsets_m, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "abs_median_m": float(np.median(absolute)),
        "p90_m": float(np.percentile(absolute, 90)),
        "max_m": float(np.max(absolute)),
        "rms_m": float(np.sqrt(np.mean(values * values))),
    }


def fit_offset_model(arc_m: Sequence[float], offsets_m: Sequence[float]) -> Dict[str, Any]:
    """用常数 / 一次 / 二次多项式描述偏移剖面，返回残差最小的那个。

    这条信息决定"人工纠偏能不能被一个简单规则复现"：残差很小 → 平移就能
    对齐；残差很大 → 说明问题在带边界（分割），不是中心估计。
    """
    arc = np.asarray(arc_m, dtype=np.float64).reshape(-1)
    values = np.asarray(offsets_m, dtype=np.float64).reshape(-1)
    if arc.size != values.size:
        return {"kind": "insufficient", "residual_rms_m": None}
    keep = np.isfinite(arc) & np.isfinite(values)
    arc, values = arc[keep], values[keep]
    if arc.size < 4 or float(np.ptp(arc)) <= 1e-9:
        return {"kind": "insufficient", "residual_rms_m": None}
    normalized = (arc - arc.min()) / float(np.ptp(arc))
    best: Optional[Dict[str, Any]] = None
    for degree, kind in ((0, "constant"), (1, "linear"), (2, "quadratic")):
        coefficients = np.polyfit(normalized, values, degree)
        predicted = np.polyval(coefficients, normalized)
        residual = values - predicted
        rms = float(np.sqrt(np.mean(residual * residual)))
        candidate = {
            "kind": kind,
            "coeffs_m": [float(value) for value in coefficients],
            "residual_rms_m": rms,
            "max_residual_m": float(np.max(np.abs(residual))),
            "span_m": float(np.ptp(arc)),
        }
        if best is None:
            best = candidate
            continue
        # 升阶必须有实质改进（≥30%），否则保留低阶模型，避免把噪声拟成曲线。
        if rms < 0.70 * float(best["residual_rms_m"]):
            best = candidate
    return best or {"kind": "insufficient", "residual_rms_m": None}


def _best_step_split(values: np.ndarray) -> Dict[str, Any]:
    """扫描两段常数的断点，返回比单一常数好多少（用于识别"跳到邻带"）。"""
    count = int(values.size)
    if count < 8:
        return {"improvement": 0.0, "jump_m": 0.0, "break_fraction": None}
    baseline = float(np.sqrt(np.mean((values - np.median(values)) ** 2))) + 1e-9
    best = {"improvement": 0.0, "jump_m": 0.0, "break_fraction": None}
    low = max(2, int(count * 0.15))
    high = min(count - 2, int(count * 0.85))
    for split in range(low, high + 1):
        left = values[:split]
        right = values[split:]
        residual = float(np.sqrt(
            np.mean(np.concatenate([
                (left - np.median(left)) ** 2,
                (right - np.median(right)) ** 2,
            ]))
        ))
        improvement = 1.0 - residual / baseline
        jump = float(abs(np.median(right) - np.median(left)))
        if improvement > best["improvement"]:
            best = {
                "improvement": float(improvement),
                "jump_m": jump,
                "break_fraction": float(split / count),
            }
    return best


def classify_offset_profile(
    offsets_m: Sequence[float],
    width_m: float = 1.0,
) -> Dict[str, Any]:
    """把人工纠偏量归类，决定"这条带还能不能靠算法修正"。

    类别（按判据顺序）：

    - ``centered``：人工线几乎与算法线重合，不用改；
    - ``off_band``：整条线被移到另一条带上（> 0.35 倍带宽）——这是带识别/
      分割问题，平移中心线修不了，硬修只会把线压到别的稻行上；
    - ``step``：偏移沿弧长突然换挡——同一带被切断或并进了邻带；
    - ``uniform``：整条线同一个横向偏移——一次平移可解；
    - ``bow``：偏移沿弧长平滑变化——带是弯的，多项式可解；
    - ``irregular``：没有简单规律，需回看掩膜。
    """
    values = np.asarray(offsets_m, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return {"kind": "insufficient", "reason": "采样点不足"}
    width = max(float(width_m or 1.0), 0.2)
    absolute = np.abs(values)
    max_abs = float(np.max(absolute))
    median_abs = float(np.median(absolute))
    if max_abs < 0.03:
        return {"kind": "centered", "reason": "人工线几乎与算法线重合", "max_abs_m": max_abs}
    if median_abs > 0.35 * width:
        return {
            "kind": "off_band",
            "reason": "人工线整体偏离超过 0.35 倍带宽：多半落在另一条带上，属分割问题",
            "max_abs_m": max_abs,
            "median_abs_m": median_abs,
        }
    step = _best_step_split(values)
    if step["improvement"] >= 0.55 and step["jump_m"] >= 0.25 * width:
        return {
            "kind": "step",
            "reason": "偏移沿弧长换挡：同一条带被切断或并进了邻带",
            "max_abs_m": max_abs,
            "jump_m": float(step["jump_m"]),
            "break_fraction": step["break_fraction"],
        }
    spread = float(np.percentile(values, 90) - np.percentile(values, 10))
    model = fit_offset_model(np.linspace(0.0, 10.0, values.size), values)
    residual = model.get("residual_rms_m")
    # 先判"弯曲"再判"平移"：一条带如果中间拱出去 30 cm、两端回到原位，
    # 它的 p90−p10 可能刚好落在平移阈值内，但二次拟合会明显更好。
    if (
        model.get("kind") in {"linear", "quadratic"}
        and residual is not None
        and residual <= 0.30 * spread
        and spread > max(0.04, 0.12 * width)
    ):
        return {
            "kind": "bow",
            "reason": "偏移沿弧长平滑变化：带是弯的，或中心估计有系统倾斜",
            "max_abs_m": max_abs,
            "spread_m": spread,
            "model": model,
        }
    if spread <= max(0.05, 0.25 * width):
        return {
            "kind": "uniform",
            "reason": "整条线基本是同一个横向偏移",
            "max_abs_m": max_abs,
            "spread_m": spread,
        }
    return {
        "kind": "irregular",
        "reason": "偏移无简单规律，需回看该带掩膜",
        "max_abs_m": max_abs,
        "spread_m": spread,
        "model": model,
    }


# ---------------------------------------------------------------- 数据契约
def empty_payload(tif_path: str = "", artifact: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "tif_path": str(tif_path or ""),
        "artifact": dict(artifact or {}),
        "updated_at": "",
        "lines": [],
    }


def new_record(
    points: Sequence[Sequence[float]],
    *,
    centerline_id: str = "",
    legacy_id: str = "",
    source_band_ids: Optional[Sequence[Any]] = None,
    band_label: str = "",
    band_type: str = "",
    width_m: float = 0.0,
    mode: str = "drag",
    pass_index: int = 1,
    edits: Optional[Sequence[Dict[str, Any]]] = None,
    profile_m: Optional[Sequence[Sequence[float]]] = None,
    artifact: Optional[Dict[str, Any]] = None,
    operator: str = "",
    record_id: str = "",
    created_at: str = "",
) -> Dict[str, Any]:
    array = as_points(points)
    if array.size == 0:
        raise ValueError("人工标定线至少需要 2 个有效点")
    now = _now_iso()
    return {
        "id": str(record_id or ""),
        "centerline_id": str(centerline_id or ""),
        "legacy_id": str(legacy_id or ""),
        "source_band_ids": [int(value) if isinstance(value, (int, np.integer)) else value
                            for value in (source_band_ids or [])],
        "band_label": str(band_label or ""),
        "band_type": str(band_type or ""),
        "width_m": float(width_m or 0.0),
        "mode": str(mode or "drag"),
        "pass": int(pass_index),
        "operator": str(operator or ""),
        "points": [[float(x), float(y)] for x, y in array],
        "edits": [dict(edit) for edit in (edits or [])],
        "profile_m": [
            [float(pair[0]), float(pair[1])]
            for pair in (profile_m or [])
            if len(pair) >= 2
        ],
        "artifact": dict(artifact or {}),
        "created_at": created_at or now,
        "updated_at": now,
    }


def _record_keys(record: Dict[str, Any]) -> List[str]:
    keys = []
    for key in ("centerline_id", "legacy_id"):
        value = str(record.get(key) or "").strip()
        if value:
            keys.append(value)
    for value in record.get("source_band_ids") or []:
        token = str(value).strip()
        if token:
            keys.append("band:" + token)
    return keys


def find_record(
    payload: Dict[str, Any],
    *,
    centerline_id: str = "",
    legacy_id: str = "",
    source_band_ids: Optional[Sequence[Any]] = None,
    pass_index: int = 1,
) -> Optional[Dict[str, Any]]:
    wanted = []
    if centerline_id:
        wanted.append(str(centerline_id))
    if legacy_id:
        wanted.append(str(legacy_id))
    for value in source_band_ids or []:
        wanted.append("band:" + str(value))
    if not wanted:
        return None
    for record in payload.get("lines") or []:
        if not isinstance(record, dict):
            continue
        if int(record.get("pass", 1) or 1) != int(pass_index):
            continue
        if set(_record_keys(record)) & set(wanted):
            return record
    return None


def upsert_record(payload: Dict[str, Any], record: Dict[str, Any]) -> str:
    """按 (centerline_id 或 legacy_id 或来源带, pass) 原位替换，返回 replaced/appended。"""
    lines = payload.setdefault("lines", [])
    wanted = set(_record_keys(record))
    if wanted:
        for index, item in enumerate(lines):
            if not isinstance(item, dict):
                continue
            if int(item.get("pass", 1) or 1) != int(record.get("pass", 1) or 1):
                continue
            if set(_record_keys(item)) & wanted:
                record["id"] = str(item.get("id") or "") or next_record_id(payload)
                record["created_at"] = str(item.get("created_at") or record.get("created_at") or "")
                lines[index] = record
                return "replaced"
    if not record.get("id"):
        record["id"] = next_record_id(payload)
    lines.append(record)
    return "appended"


def next_record_id(payload: Dict[str, Any]) -> str:
    max_index = 0
    for item in payload.get("lines") or []:
        token = str(item.get("id") or "")
        if token.startswith("op-"):
            try:
                max_index = max(max_index, int(token[3:]))
            except ValueError:
                continue
    return "op-%04d" % (max_index + 1)


def normalize_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        schema = payload.get("schema") if isinstance(payload, dict) else None
        raise ValueError("人工标定文件 schema 不匹配: %r" % (schema,))
    out = dict(payload)
    out.setdefault("tif_path", "")
    out.setdefault("artifact", {})
    out.setdefault("updated_at", "")
    lines = out.get("lines")
    if not isinstance(lines, list):
        raise ValueError("人工标定文件缺少 lines 列表")
    clean: List[Dict[str, Any]] = []
    for item in lines:
        if not isinstance(item, dict):
            continue
        array = as_points(item.get("points"))
        if array.size == 0:
            continue
        record = dict(item)
        record["points"] = [[float(x), float(y)] for x, y in array]
        record["source_band_ids"] = list(item.get("source_band_ids") or [])
        record["edits"] = [dict(edit) for edit in (item.get("edits") or []) if isinstance(edit, dict)]
        record["pass"] = int(item.get("pass", 1) or 1)
        clean.append(record)
    out["lines"] = clean
    return out


def build_profile(
    base_points: Sequence[Sequence[float]],
    points: Sequence[Sequence[float]],
    mpp: float,
    *,
    step_m: float = DEFAULT_PROFILE_STEP_M,
) -> List[List[float]]:
    """量出（算法线 → 人工线）的偏移剖面，单位米，写进记录供离线分析。"""
    step_px = max(float(step_m), 0.05) / max(float(mpp), 1e-9)
    arc_px, offset_px = signed_offsets(base_points, points, step=step_px)
    if arc_px.size == 0:
        return []
    return [
        [float(arc * mpp), float(offset * mpp)]
        for arc, offset in zip(arc_px, offset_px)
    ]


# ---------------------------------------------------------------- 界面/接口
def from_reference_lines(
    reference_payload: Dict[str, Any],
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    artifact: Optional[Dict[str, Any]] = None,
    band_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """把"参考线标注"（显示坐标、直线拟合口径）转成人工标定线（局部坐标）。

    人工参考线本来就是人沿带描的点，只是过去只用来做评价；这里把它接进
    标定闭环：同一批桩点可以既当评价基准，又在规划时替换算法线。
    """
    records: List[Dict[str, Any]] = []
    skipped = 0
    lookup = dict(band_lookup or {})
    for item in reference_payload.get("lines") or []:
        if not isinstance(item, dict):
            continue
        array = as_points(item.get("points"))
        if array.size == 0:
            skipped += 1
            continue
        local = array - np.array([float(offset_x), float(offset_y)])
        band_id = str(item.get("band_id") or "")
        meta = lookup.get(band_id, {})
        try:
            records.append(new_record(
                local,
                centerline_id=str(meta.get("centerline_id") or ""),
                legacy_id=str(meta.get("legacy_id") or ""),
                source_band_ids=meta.get("source_band_ids") or [],
                band_label=str(meta.get("label") or item.get("band_label") or ""),
                band_type=str(meta.get("band_type") or ""),
                width_m=float(meta.get("width_m") or 0.0),
                mode="trace",
                pass_index=int(item.get("pass", 1) or 1),
                artifact=artifact,
                record_id=str(item.get("id") or "").replace("ref-", "op-"),
                created_at=str(item.get("created_at") or ""),
            ))
        except ValueError:
            skipped += 1
    return records, {"converted": len(records), "skipped": skipped}


def planner_overrides(
    payload: Dict[str, Any],
    *,
    mask_version: Optional[int] = None,
    tif_path: Optional[str] = None,
    pass_index: int = 1,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """生成规划用的替换表；版本或影像对不上的线一律不用（fail-safe）。

    Returns:
        ``({key: {"points": ..., "source": "operator", ...}}, report)``，
        key 包括 centerline id、legacy id 与 ``band:<id>``。
    """
    overrides: Dict[str, Dict[str, Any]] = {}
    report: Dict[str, Any] = {
        "algorithm_version": "operator-centerline-override-v1",
        "available": 0,
        "applied": 0,
        "stale": [],
        "reason": "",
    }
    lines = [
        item for item in (payload.get("lines") or [])
        if isinstance(item, dict) and int(item.get("pass", 1) or 1) == int(pass_index)
    ]
    report["available"] = len(lines)
    if not lines:
        report["reason"] = "no_operator_lines"
        return overrides, report
    payload_tif = str(payload.get("tif_path") or "")
    if tif_path and payload_tif and os.path.normcase(payload_tif) != os.path.normcase(str(tif_path)):
        report["reason"] = "tif_mismatch"
        report["stale"] = [str(item.get("id") or "") for item in lines]
        return overrides, report
    stored_version = payload.get("artifact", {}).get("mask_version")
    if (
        mask_version is not None
        and stored_version is not None
        and int(stored_version) != int(mask_version)
    ):
        report["reason"] = "mask_version_mismatch"
        report["stale"] = [str(item.get("id") or "") for item in lines]
        return overrides, report
    for item in lines:
        points = as_points(item.get("points"))
        if points.size == 0:
            continue
        entry = {
            "points": [[float(x), float(y)] for x, y in points],
            "source": "operator",
            "record_id": str(item.get("id") or ""),
            "band_label": str(item.get("band_label") or ""),
            "edits": len(item.get("edits") or []),
        }
        # 只用最精确的身份绑定：有中心线 id 就只认它。一条带宽带被拆成多股
        # （等距区）时，source_band_ids 会让"标一条"变成"改一片"，必须避免。
        identifiers: List[str] = []
        centerline_id = str(item.get("centerline_id") or "").strip()
        legacy_id = str(item.get("legacy_id") or "").strip()
        if centerline_id:
            identifiers.append(centerline_id)
        if legacy_id:
            identifiers.append(legacy_id)
        if not identifiers:
            identifiers.extend("band:" + str(value) for value in item.get("source_band_ids") or [])
        for key in identifiers:
            overrides[key] = entry
        report["applied"] += 1
    return overrides, report
