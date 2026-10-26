# -*- coding: utf-8 -*-
"""参考线（人工参考线）数据契约、拟合与几何度量工具。

中心线评价的核心是"提取中心线 vs 人工参考线"：人工在正射影像上沿宽行带
按桩号描点并拟合出参考线。本模块只包含参考线的纯几何与数据结构逻辑——
不依赖 Qt、不读写缓存文件——供桌面端标注模式（画线、质检、存档）与离线
评价工具（算偏差、出表出图）共同引用。

数据契约（DECISION-028，sidecar：项目缓存目录 reference_lines.json）::

    {
      "schema": "centerline-reference-lines-v1",
      "tif_path": "<项目源影像路径>",
      "station_spacing_hint_m": 5.0,
      "updated_at": "2026-10-09T12:00:00",
      "lines": [
        {
          "id": "ref-0001",
          "band_id": "centerline-003" 或 null,
          "band_label": "",
          "pass": 1,
          "operator": "",
          "created_at": "2026-10-09T12:00:00",
          "points": [[x, y], ...],          # 全幅显示坐标（界面画布口径）
          "fit": {"p1": [x, y], "p2": [x, y], "angle_rad": 1.23}
        }
      ]
    }

坐标口径：桩点为全幅显示坐标（等于掩膜局部坐标 + mask_offset）。评价前由
调用方转换为掩膜局部坐标，再与提取中心线（掩膜局部坐标）比较。
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

SCHEMA = "centerline-reference-lines-v1"
DEFAULT_STATION_SPACING_M = 5.0      # 人工画线桩距建议值（论文口径）
DEFAULT_SAMPLE_SPACING_M = 1.0       # 评价采样间距（论文口径，肖延伟式沿线采样）
MIN_STATION_POINTS = 2


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def empty_payload(tif_path: str = "", station_spacing_m: float = DEFAULT_STATION_SPACING_M) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "tif_path": str(tif_path or ""),
        "station_spacing_hint_m": float(station_spacing_m),
        "updated_at": "",
        "lines": [],
    }


def fit_line(points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """对桩点做最小二乘直线拟合（主方向角度 + 两端点 + 垂距残差）。"""
    pts = np.asarray([(float(x), float(y)) for x, y in points], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < MIN_STATION_POINTS or pts.shape[1] != 2:
        raise ValueError("参考线至少需要 2 个有效桩点")
    center = pts.mean(axis=0)
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    sxx = float(np.sum(dx * dx))
    sxy = float(np.sum(dx * dy))
    syy = float(np.sum(dy * dy))
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy) % math.pi
    ux, uy = math.cos(angle), math.sin(angle)
    t = dx * ux + dy * uy
    t0, t1 = float(np.min(t)), float(np.max(t))
    p1 = (center[0] + ux * t0, center[1] + uy * t0)
    p2 = (center[0] + ux * t1, center[1] + uy * t1)
    perp = np.abs(ux * dy - uy * dx)  # 各桩点到拟合线的垂距
    residual = float(np.sqrt(np.mean(perp * perp))) if len(pts) else 0.0
    worst = int(np.argmax(perp)) if len(pts) else 0
    return {
        "p1": [float(p1[0]), float(p1[1])],
        "p2": [float(p2[0]), float(p2[1])],
        "angle_rad": float(angle),
        "residual_rms_px": residual,
        # 最大垂距与它来自第几个桩点（1 起）。只看 RMS 会把"只有一个点跨到
        # 别的行"平均掉：58 个点里 4 个偏 114 px，RMS 只有 34 px。
        "max_deviation_px": float(perp[worst]) if len(pts) else 0.0,
        "max_deviation_station": worst + 1,
        "point_count": int(len(pts)),
    }


def line_angle(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.atan2(float(p2[1]) - float(p1[1]), float(p2[0]) - float(p1[0])) % math.pi


def angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """无向直线夹角，返回 [0, 90] 度（等价"180° 取小"口径）。"""
    d = abs((float(angle_a) - float(angle_b)) % math.pi)
    return math.degrees(min(d, math.pi - d))


def reference_fit(reference: Any) -> Dict[str, Any]:
    """从参考线记录（dict）或桩点列表得到拟合信息。"""
    if isinstance(reference, dict):
        fit = reference.get("fit") or {}
        if fit.get("p1") and fit.get("p2"):
            angle = fit.get("angle_rad")
            if angle is None:
                angle = line_angle(fit["p1"], fit["p2"])
            return {
                "p1": [float(fit["p1"][0]), float(fit["p1"][1])],
                "p2": [float(fit["p2"][0]), float(fit["p2"][1])],
                "angle_rad": float(angle),
            }
        return fit_line(reference.get("points") or [])
    return fit_line(reference)


def sample_along_line(p1: Sequence[float], p2: Sequence[float], spacing: float) -> List[Tuple[float, float]]:
    """沿线段等间隔采样（含两端点，至少 2 个点）。spacing 与坐标同单位。"""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    length = math.hypot(x2 - x1, y2 - y1)
    spacing = max(float(spacing), 1e-9)
    if length <= 1e-9:
        return [(x1, y1)]
    count = max(2, int(math.floor(length / spacing)) + 1)
    step = length / (count - 1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    return [(x1 + ux * step * i, y1 + uy * step * i) for i in range(count)]


def polyline_distances(points: Sequence[Sequence[float]], polyline: Sequence[Sequence[float]]) -> np.ndarray:
    """每个点到折线（线段集合）的最短距离，返回数组（与坐标同单位）。"""
    pts = np.asarray(points, dtype=np.float64)
    poly = np.asarray(polyline, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] != 2:
        raise ValueError("目标中心线需要至少 2 个有效点")
    if pts.size == 0:
        return np.zeros(0, dtype=np.float64)
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ab_len2 = np.sum(ab * ab, axis=1)
    ab_len2[ab_len2 <= 0] = 1e-12
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(ap * ab[None, :, :], axis=-1) / ab_len2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    diff = pts[:, None, :] - proj
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return np.min(dist, axis=1)


def compare_to_polyline(
    ref_fit: Dict[str, Any],
    target_points: Sequence[Sequence[float]],
    mpp: float,
    *,
    sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
) -> Dict[str, Any]:
    """参考线 vs 提取中心线：方向角偏差 + 平均/最大横向偏差（cm）+ 重叠率。

    横向偏差按肖延伟式口径：沿参考线等间隔采样，各采样点到提取中心线
    （折线）的最短距离；平均与最大以 cm 报告。
    重叠率 = 提取中心线在参考线纵向范围内覆盖的比例（<1 提示截断）。
    """
    mpp = float(mpp)
    if mpp <= 0:
        raise ValueError("mpp 必须为正（米/像素）")
    p1, p2 = ref_fit["p1"], ref_fit["p2"]
    target = np.asarray(target_points, dtype=np.float64)
    target_fit = fit_line(target)
    theta_deg = angle_difference_deg(ref_fit.get("angle_rad", line_angle(p1, p2)), target_fit["angle_rad"])
    samples = sample_along_line(p1, p2, float(sample_spacing_m) / mpp)
    dists_px = polyline_distances(samples, target)
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    overlap_ratio = 0.0
    if length > 1e-9:
        ux = (p2[0] - p1[0]) / length
        uy = (p2[1] - p1[1]) / length
        t_target = (target[:, 0] - p1[0]) * ux + (target[:, 1] - p1[1]) * uy
        t_min = max(0.0, float(np.min(t_target)))
        t_max = min(length, float(np.max(t_target)))
        overlap_ratio = max(0.0, min(1.0, (t_max - t_min) / length))
    cm = mpp * 100.0
    return {
        "theta_deg": float(theta_deg),
        "elat_mean_cm": float(np.mean(dists_px)) * cm,
        "elat_max_cm": float(np.max(dists_px)) * cm,
        "overlap_ratio": float(overlap_ratio),
        "sample_count": int(len(samples)),
    }


def compare_lines(
    ref_a_fit: Dict[str, Any],
    ref_b_fit: Dict[str, Any],
    mpp: float,
    *,
    sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
) -> Dict[str, Any]:
    """两遍人工参考线之间同口径偏差 = 人工判读噪声底（沿第一遍采样）。"""
    target = [list(ref_b_fit["p1"]), list(ref_b_fit["p2"])]
    return compare_to_polyline(ref_a_fit, target, mpp, sample_spacing_m=sample_spacing_m)


def line_inside_mask_fraction(ref_fit: Dict[str, Any], mask: np.ndarray, spacing_px: float) -> float:
    """参考线落在掩膜主体内的采样比例（自动质检；不做修正，只提示复核）。"""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("掩膜必须为二维数组")
    height, width = arr.shape
    samples = sample_along_line(ref_fit["p1"], ref_fit["p2"], spacing_px)
    if not samples:
        return 0.0
    inside = 0
    for x, y in samples:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height and arr[yi, xi] > 0:
            inside += 1
    return inside / len(samples)


def normalize_payload(payload: Any) -> Dict[str, Any]:
    """校验并规范化参考线文件；无效线条被跳过，拟合信息一律由桩点重算。"""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        schema = payload.get("schema") if isinstance(payload, dict) else None
        raise ValueError("参考线文件 schema 不匹配: %r" % (schema,))
    out = dict(payload)
    out.setdefault("tif_path", "")
    out.setdefault("station_spacing_hint_m", DEFAULT_STATION_SPACING_M)
    out.setdefault("updated_at", "")
    lines = out.get("lines")
    if not isinstance(lines, list):
        raise ValueError("参考线文件缺少 lines 列表")
    clean: List[Dict[str, Any]] = []
    for item in lines:
        if not isinstance(item, dict):
            continue
        points = []
        for point in item.get("points") or []:
            if len(point) >= 2:
                points.append([float(point[0]), float(point[1])])
        if len(points) < MIN_STATION_POINTS:
            continue
        record = dict(item)
        record["points"] = points
        record["fit"] = fit_line(points)
        record["pass"] = int(item.get("pass", 1) or 1)
        clean.append(record)
    out["lines"] = clean
    return out


def new_line_record(
    points: Sequence[Sequence[float]],
    *,
    band_id: Any = None,
    band_label: str = "",
    pass_index: int = 1,
    operator: str = "",
    line_id: str = "",
    created_at: str = "",
) -> Dict[str, Any]:
    pts = [[float(x), float(y)] for x, y in points]
    return {
        "id": str(line_id or ""),
        "band_id": band_id,
        "band_label": str(band_label or ""),
        "pass": int(pass_index),
        "operator": str(operator or ""),
        "created_at": created_at or _now_iso(),
        "points": pts,
        "fit": fit_line(pts),
    }


#: 相邻两段桩点的转折角阈值（180°=继续直行，0°=原路折返）。
#: 一条行带上的桩点不会在一站之内拐 40°（5 m 桩距要拐 40° 相当于转弯半径 7 m），
#: 所以超过它基本就是"换了一条行带往下画"，必须拆成两条参考线。
TURN_SPLIT_ANGLE_DEG = 40.0
#: 判断转折时，两侧线段都不得短于「中位桩距 × 该比例」，避免双击抖动被误判。
TURN_SPLIT_MIN_STEP_RATIO = 0.5
TURN_SPLIT_MIN_STEP_PX = 20.0


def turn_angles_deg(points: Sequence[Sequence[float]]) -> List[float]:
    """返回每个中间桩点处的转折角（180° 直行，0° 原路折返）。"""
    pts = np.asarray([(float(x), float(y)) for x, y in points], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return []
    vectors = np.diff(pts, axis=0)
    angles: List[float] = []
    for index in range(1, len(vectors)):
        a, b = vectors[index - 1], vectors[index]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na <= 1e-9 or nb <= 1e-9:
            angles.append(180.0)
            continue
        cos_value = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        angles.append(math.degrees(math.acos(cos_value)))
    return angles


def split_points_at_turns(
    points: Sequence[Sequence[float]],
    *,
    angle_deg: float = TURN_SPLIT_ANGLE_DEG,
    min_step_ratio: float = TURN_SPLIT_MIN_STEP_RATIO,
    min_step_px: float = TURN_SPLIT_MIN_STEP_PX,
    min_segment_points: int = 2,
) -> List[List[List[float]]]:
    """把一条桩点序列按明显转折拆成若干段（每段 ≥2 点，拐点由相邻两段共用）。

    一条参考线必须只对应一条行带：把两条行带连着画（中间直角过渡）会让
    "拟合直线 vs 桩点"整体错位，用户看到的就是"点和线错位"。

    拐点处的相邻两段共用该桩点（几何上它就是两段的公共端点）。连续几个
    桩点都超阈值时（折返式急转会同时抬升两个转折角）只取转折角最大的那个，
    否则中间那 1 个点会变成一个"单点段"被丢掉。

    ``min_segment_points`` 用于丢弃转折处的过渡桩点（默认 2 表示一段都不丢）：
    用户在田头换行带时会顺手点 1-2 个过渡点，它们既不属于上一条行带也不属于
    下一条，留着会把两侧的拟合线拉歪。
    """
    pts = [[float(x), float(y)] for x, y in points]
    if len(pts) < 3:
        return [pts] if len(pts) >= MIN_STATION_POINTS else []
    turns = turn_angles_deg(pts)
    if not turns:
        return [pts]
    steps = [
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    ]
    # 只用绝对下限判定"够长的桩距"，不按中位桩距按比例抬高下限：用户实际
    # 桩距常常很不均匀（实测中位 352 px、最大 911 px），按 0.5×中位设限会把
    # 真实转折全部滤掉（ref-0004 154 点、27 处 ≥40° 转折被判成 1 段）。
    limit = float(min_step_px)
    candidates = [
        index
        for index, turn in enumerate(turns)
        if turn >= float(angle_deg)
        and steps[index] >= limit
        and steps[index + 1] >= limit
    ]
    if not candidates:
        return [pts]
    apex_runs: List[List[int]] = []
    for index in candidates:
        if apex_runs and index == apex_runs[-1][-1] + 1:
            apex_runs[-1].append(index)
        else:
            apex_runs.append([index])
    # turns[index] 是第 (index+1) 个桩点处的转折角 → 该桩点的 0 基索引
    corners = [max(run, key=lambda k: turns[k]) + 1 for run in apex_runs]
    # 拐点归到**后一段**：它已经是新那条线带的起点，若同时算进前一段，
    # 前一段的拟合会被拐点带偏（实测 RMS 3.5 px → 19.2 px）。
    corners = [c for c in corners if 2 <= c <= len(pts) - 2]
    if not corners:
        return [pts]
    minimum = max(MIN_STATION_POINTS, int(min_segment_points))
    segments: List[List[List[float]]] = []
    start = 0
    for corner in corners:
        chunk = pts[start:corner]
        if len(chunk) >= minimum:
            segments.append(chunk)
        start = corner
    tail = pts[start:]
    if len(tail) >= minimum:
        segments.append(tail)
    return segments or [pts]


def _split_by_deviation(
    segment: Sequence[Sequence[float]],
    max_deviation: float,
    min_points: int,
    max_iterations: int,
    depth: int = 0,
) -> List[List[List[float]]]:
    """递归按"本段自己的拟合线下最大垂距"切开，直到每段都够直或点数不足。"""
    pts = [[float(x), float(y)] for x, y in segment]
    if len(pts) < min_points or depth >= max_iterations:
        return [pts]
    fit = fit_line(pts)
    p1 = np.asarray(fit["p1"], dtype=np.float64)
    vector = np.asarray(fit["p2"], dtype=np.float64) - p1
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return [pts]
    unit = vector / norm
    normal = np.asarray([-unit[1], unit[0]], dtype=np.float64)
    deviations = np.abs((np.asarray(pts, dtype=np.float64) - p1) @ normal)
    worst = int(np.argmax(deviations))
    if float(deviations[worst]) <= float(max_deviation):
        return [pts]
    if worst <= 0 or worst >= len(pts) - 1:
        # 最差点在端点上：它本身就是过渡点，丢掉端点后重试（每次少一个点，必然收敛）
        trimmed = pts[1:] if worst <= 0 else pts[:-1]
        return _split_by_deviation(
            trimmed, max_deviation, min_points, max_iterations, depth + 1
        )
    # 最差点通常是两条行带之间的过渡点：它不进任何一段
    parts: List[List[List[float]]] = []
    for chunk in (pts[:worst], pts[worst + 1:]):
        if len(chunk) >= min_points:
            parts.extend(_split_by_deviation(chunk, max_deviation, min_points, max_iterations, depth + 1))
        elif len(chunk) >= MIN_STATION_POINTS:
            parts.append(chunk)
    return parts or [pts]


def split_reference_lines(
    points: Sequence[Sequence[float]],
    *,
    angle_deg: float = TURN_SPLIT_ANGLE_DEG,
    min_step_px: float = TURN_SPLIT_MIN_STEP_PX,
    min_segment_points: int = 3,
    max_deviation_px: float = 0.0,
) -> List[List[List[float]]]:
    """一条参考线一条行带：先按转折粗分，再按"最大垂距"细分。

    用户实际画法往往是"顺着一条行带点几个桩点 → 在田头拐到下一行 → 继续点"，
    转折角不一定都超过 40°（实测 ref-0004：154 点、桩距中位 352 px，27 处转折
    里不少是渐进的），因此粗分之后还要把"一段里横跨两条行带"的部分按最大垂距
    切开。``max_deviation_px`` 为 0 时只做转折粗分。
    """
    base = split_points_at_turns(
        points,
        angle_deg=angle_deg,
        min_step_px=min_step_px,
        min_segment_points=min_segment_points,
    )
    if not max_deviation_px or float(max_deviation_px) <= 0.0:
        return base
    segments: List[List[List[float]]] = []
    for segment in base:
        segments.extend(
            _split_by_deviation(
                segment,
                float(max_deviation_px),
                max(3, int(min_segment_points)),
                16,
            )
        )
    return segments or base


def next_line_id(payload: Dict[str, Any]) -> str:
    max_index = 0
    for item in payload.get("lines") or []:
        token = str(item.get("id") or "")
        if token.startswith("ref-"):
            try:
                max_index = max(max_index, int(token[4:]))
            except ValueError:
                continue
    return "ref-%04d" % (max_index + 1)


def upsert_line(payload: Dict[str, Any], record: Dict[str, Any]) -> str:
    """按 (band_id, pass) 替换已有参考线并保留其 id；band_id 为空则追加。

    返回 "replaced" 或 "appended"。
    """
    band_id = record.get("band_id")
    pass_index = int(record.get("pass", 1) or 1)
    lines = payload.setdefault("lines", [])
    if band_id not in (None, ""):
        for index, item in enumerate(lines):
            if str(item.get("band_id")) == str(band_id) and int(item.get("pass", 1) or 1) == pass_index:
                record["id"] = str(item.get("id") or "") or next_line_id(payload)
                lines[index] = record
                return "replaced"
    if not record.get("id"):
        record["id"] = next_line_id(payload)
    lines.append(record)
    return "appended"
