# -*- coding: utf-8 -*-
"""中心线评价（离线工具）：提取中心线 vs 人工参考线。

按论文口径计算（DECISION-028）：
- 方向角偏差 Eθ：两线方向差（180° 取小，单位 °）；
- 横向偏差 Elat：沿参考线等间隔采样（默认 1 m），各采样点到提取中心线
  （折线）的最短距离；平均与最大以 cm 报告（对齐已核验文献口径）；
- 重叠率与采样数（截断等失败模式诊断列）；
- 人工重复性：同一带第 1/2 遍参考线之间的同口径偏差 = 人工判读噪声底。

用法（game python，源码运行环境）::

    python src/centerline_evaluation.py --cache-dir <项目缓存目录> \
        --tif <当前 result.tif 路径> [--out <输出目录>] [--reference <参考线json>]

默认读取项目缓存目录内的 reference_lines.json（桌面端标注模式写入）。
输出"评价包"：per_line.csv / noise_floor.csv / summary.json /
reference_lines.json / overlay.png / boxplot.png / README.md，并打包同名 zip。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reference_lines import (
    DEFAULT_SAMPLE_SPACING_M,
    compare_lines,
    compare_to_polyline,
    fit_line,
    normalize_payload,
    reference_fit,
)

CENTERLINE_EVAL_VERSION = "centerline-eval-v1"

PER_LINE_COLUMNS = [
    "line_id", "band_id", "band_type", "pass", "station_count",
    "match_distance_cm", "theta_deg", "elat_mean_cm", "elat_max_cm",
    "overlap_ratio", "sample_count", "note",
]
NOISE_COLUMNS = [
    "band_id", "line_id_pass1", "line_id_pass2",
    "theta_deg", "elat_mean_cm", "elat_max_cm",
]


# ─────────────────────────────────────────────────────────────
# 项目输入与环境
# ─────────────────────────────────────────────────────────────

def resolve_repo_config(path: Optional[str] = None) -> Dict[str, Any]:
    """mask_processing 配置：默认读仓库根 config.json（与桌面端源码运行一致）。"""
    candidate = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    )
    with open(candidate, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    section = data.get("mask_processing")
    if not isinstance(section, dict) or not section:
        raise ValueError("配置文件缺少 mask_processing 段: %s" % candidate)
    return dict(section)


def load_project_inputs(cache_dir: str, tif_path: str) -> Dict[str, Any]:
    """读取项目缓存并重建地理环境（与桌面端/探针一致的坐标口径）。"""
    from affine import Affine
    import rasterio
    from pyproj import Transformer

    from geo import GeoUtils
    from row_geometry import meters_per_pixel

    mask_path = os.path.join(cache_dir, "mask.npz")
    meta_path = os.path.join(cache_dir, "meta.json")
    state_path = os.path.join(cache_dir, "project_state.json")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError("项目缓存缺少 mask.npz: %s" % mask_path)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("项目缓存缺少 meta.json: %s" % meta_path)
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    saved_state: Dict[str, Any] = {}
    if os.path.isfile(state_path):
        saved_state = json.load(open(state_path, "r", encoding="utf-8"))
    with np.load(mask_path) as bundle:
        raw_mask = np.asarray(bundle["mask"]).copy()
        offset_x = int(bundle["offset_x"]) if "offset_x" in bundle else 0
        offset_y = int(bundle["offset_y"]) if "offset_y" in bundle else 0
    if not tif_path or not os.path.isfile(tif_path):
        fallback = str(saved_state.get("tif_path") or "")
        if fallback and os.path.isfile(fallback):
            tif_path = fallback
        else:
            raise FileNotFoundError(
                "找不到源影像（请用 --tif 指定当前 result.tif 路径）: %s"
                % (tif_path or "未提供")
            )
    with rasterio.open(tif_path) as src:
        transform, crs = src.transform, src.crs
    scale_x = float(meta.get("display_scale_x") or 1.0)
    scale_y = float(meta.get("display_scale_y") or 1.0)
    geo = GeoUtils()
    geo.set_affine(
        transform * Affine.scale(scale_x, scale_y),
        Transformer.from_crs(crs, "4326", always_xy=True),
    )
    state = SimpleNamespace(
        field_boundary=saved_state.get("field_boundary"),
        mask_offset_x=offset_x,
        mask_offset_y=offset_y,
        mask_raw=raw_mask,
        source_sha256="",
        harvester_params={},
        turn_strategy="contour",
    )
    mpp = float(meters_per_pixel(raw_mask, geo, state))
    return {
        "raw_mask": raw_mask,
        "offset": (offset_x, offset_y),
        "meta": meta,
        "saved_state": saved_state,
        "geo": geo,
        "state": state,
        "mpp": mpp,
        "tif_path": tif_path,
    }


def extract_centerlines(inputs: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """用当前 mask_processing 配置重跑掩膜处理，取全部行带及其中心线。"""
    from mask_processor import process_mask

    raw = np.asarray(inputs["raw_mask"]).copy()
    inputs["state"].mask_raw = raw
    result = process_mask(raw, geo=inputs["geo"], state=inputs["state"], config=dict(config))
    bands = list(result.get("all_bands") or [])
    return result, bands


def _band_candidates(bands: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for band in bands:
        points = band.get("centerline") or []
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        items.append({
            "id": band.get("id"),
            "band_type": band.get("band_type"),
            "width_m": band.get("width_m"),
            "profile_width_m": band.get("profile_width_m"),
            "points": arr,
        })
    return items


def _to_local(points: Sequence[Sequence[float]], offset: Tuple[int, int]) -> List[List[float]]:
    ox, oy = offset
    return [[float(x) - ox, float(y) - oy] for x, y in points]


# ─────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────

def evaluate_reference_lines(
    local_lines: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    mpp: float,
    *,
    sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
    match_max_distance_m: float = 0.8,
) -> List[Dict[str, Any]]:
    """每条参考线匹配最近的行带中心线并计算 Eθ / Elat。"""
    rows: List[Dict[str, Any]] = []
    for line in local_lines:
        fit = reference_fit(line)
        best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None
        for band in candidates:
            cmp = compare_to_polyline(fit, band["points"], mpp, sample_spacing_m=sample_spacing_m)
            if best is None or cmp["elat_mean_cm"] < best[0]:
                best = (cmp["elat_mean_cm"], band, cmp)
        matched = best is not None and (best[0] / 100.0) <= float(match_max_distance_m)
        row: Dict[str, Any] = {
            "line_id": line.get("id"),
            "band_id": None,
            "band_type": "",
            "pass": int(line.get("pass", 1) or 1),
            "station_count": len(line.get("points") or []),
            "match_distance_cm": None if best is None else round(best[0], 3),
            "theta_deg": None,
            "elat_mean_cm": None,
            "elat_max_cm": None,
            "overlap_ratio": None,
            "sample_count": None,
            "note": "",
        }
        if matched and best is not None:
            band, cmp = best[1], best[2]
            row.update({
                "band_id": band.get("id"),
                "band_type": band.get("band_type") or "",
                "theta_deg": round(cmp["theta_deg"], 4),
                "elat_mean_cm": round(cmp["elat_mean_cm"], 3),
                "elat_max_cm": round(cmp["elat_max_cm"], 3),
                "overlap_ratio": round(cmp["overlap_ratio"], 4),
                "sample_count": cmp["sample_count"],
            })
        elif best is not None:
            row["note"] = "未匹配到行带中心线（最近距离 %.1f cm）" % best[0]
        else:
            row["note"] = "没有可用的行带中心线"
        rows.append(row)
    return rows


def compute_noise_floor(
    local_lines: Sequence[Dict[str, Any]],
    mpp: float,
    *,
    sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
    match_max_distance_m: float = 1.0,
) -> List[Dict[str, Any]]:
    """第 1 遍与第 2 遍参考线之间的同口径偏差（人工判读噪声底）。"""
    pass1 = [line for line in local_lines if int(line.get("pass", 1) or 1) == 1]
    pass2 = [line for line in local_lines if int(line.get("pass", 1) or 1) == 2]
    rows: List[Dict[str, Any]] = []
    for line_a in pass1:
        fit_a = reference_fit(line_a)
        best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None
        for line_b in pass2:
            fit_b = reference_fit(line_b)
            cmp = compare_lines(fit_a, fit_b, mpp, sample_spacing_m=sample_spacing_m)
            if best is None or cmp["elat_mean_cm"] < best[0]:
                best = (cmp["elat_mean_cm"], line_b, cmp)
        if best is None or (best[0] / 100.0) > float(match_max_distance_m):
            continue
        line_b, cmp = best[1], best[2]
        rows.append({
            "band_id": line_a.get("band_id") or line_b.get("band_id") or "",
            "line_id_pass1": line_a.get("id"),
            "line_id_pass2": line_b.get("id"),
            "theta_deg": round(cmp["theta_deg"], 4),
            "elat_mean_cm": round(cmp["elat_mean_cm"], 3),
            "elat_max_cm": round(cmp["elat_max_cm"], 3),
        })
    return rows


def _mean(values: Sequence[Any]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def build_summary(
    rows: Sequence[Dict[str, Any]],
    noise_rows: Sequence[Dict[str, Any]],
    *,
    inputs: Dict[str, Any],
    cache_dir: str,
    tif_path: str,
    sample_spacing_m: float,
    match_max_distance_m: float,
    station_spacing_hint_m: float,
) -> Dict[str, Any]:
    import row_geometry

    matched = [row for row in rows if row.get("elat_mean_cm") is not None]
    noise_summary = None
    if noise_rows:
        noise_summary = {
            "pair_count": len(noise_rows),
            "theta_mean_deg": _r(_mean([row["theta_deg"] for row in noise_rows])),
            "elat_mean_cm": _r(_mean([row["elat_mean_cm"] for row in noise_rows]), 3),
            "elat_max_cm": _r(max(row["elat_max_cm"] for row in noise_rows), 3),
        }
    return {
        "version": CENTERLINE_EVAL_VERSION,
        "mask_regularization_version": int(row_geometry.MASK_REGULARIZATION_VERSION),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": cache_dir,
        "tif_path": tif_path,
        "mpp_m_per_px": _r(inputs["mpp"], 8),
        "sample_spacing_m": float(sample_spacing_m),
        "match_max_distance_m": float(match_max_distance_m),
        "station_spacing_hint_m": float(station_spacing_hint_m),
        "line_count": len(rows),
        "matched_line_count": len(matched),
        "unmatched_line_ids": [row["line_id"] for row in rows if row.get("elat_mean_cm") is None],
        "theta_mean_deg": _r(_mean([row["theta_deg"] for row in matched])),
        "elat_mean_cm": _r(_mean([row["elat_mean_cm"] for row in matched]), 3),
        "elat_mean_of_max_cm": _r(_mean([row["elat_max_cm"] for row in matched]), 3),
        "elat_max_cm": _r(max(row["elat_max_cm"] for row in matched), 3) if matched else None,
        "noise_floor": noise_summary,
        "notes": [
            "评价重跑当前 mask_processing 配置；参考线为人工研究证据（不进入作业状态）。",
            "横向偏差口径：沿参考线 1 m 等间隔采样到提取中心线的最短距离（对齐肖延伟式）。",
            "Eθ 为两条直线方向差（180° 取小）；参考线与中心线均取最小二乘拟合方向。",
            "overlap_ratio < 1 提示提取中心线在参考线范围内存在截断/缺失。",
        ],
    }


# ─────────────────────────────────────────────────────────────
# 输出（表 / 图 / 包）
# ─────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in columns})


def _apply_cjk_font(plt) -> None:
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def render_overlay_png(
    path: str,
    *,
    cache_dir: str,
    mask_local: Optional[np.ndarray],
    local_lines: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    offset: Tuple[int, int],
) -> None:
    """参考线 vs 提取中心线 叠加图（优先叠加在显示影像上）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _apply_cjk_font(plt)
    image = None
    image_path = os.path.join(cache_dir, "image_bgr.npy")
    if os.path.isfile(image_path):
        try:
            image = np.load(image_path)
        except (OSError, ValueError):
            image = None
    ox, oy = float(offset[0]), float(offset[1])
    fig, ax = plt.subplots(figsize=(13, 10), dpi=140)
    if image is not None and getattr(image, "ndim", 0) == 3:
        ax.imshow(image[..., ::-1])
        shift = np.array([ox, oy])

        def _plot(points, **kw):
            arr = np.asarray(points, dtype=np.float64) + shift
            ax.plot(arr[:, 0], arr[:, 1], **kw)

        def _scatter(points, **kw):
            arr = np.asarray(points, dtype=np.float64) + shift
            ax.scatter(arr[:, 0], arr[:, 1], **kw)
    else:
        if mask_local is not None:
            ax.imshow(mask_local, cmap="gray")

        def _plot(points, **kw):
            arr = np.asarray(points, dtype=np.float64)
            ax.plot(arr[:, 0], arr[:, 1], **kw)

        def _scatter(points, **kw):
            arr = np.asarray(points, dtype=np.float64)
            ax.scatter(arr[:, 0], arr[:, 1], **kw)

    for band in candidates:
        _plot(band["points"], color="#2ca02c", lw=1.2, alpha=0.9)
    for line in local_lines:
        fit = line["fit"]
        _plot([fit["p1"], fit["p2"]], color="#ff7f0e", lw=2.2, ls="--")
        if line.get("points"):
            _scatter(line["points"], s=14, color="#d62728", zorder=5)
    handles = [
        Line2D([], [], color="#2ca02c", lw=1.6, label="提取中心线（当前算法）"),
        Line2D([], [], color="#ff7f0e", lw=2.2, ls="--", label="人工参考线（拟合）"),
        Line2D([], [], color="#d62728", marker="o", ls="", label="桩点"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_boxplot_png(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """逐线横向偏差 / 方向角偏差箱线图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_cjk_font(plt)
    elat = [row["elat_mean_cm"] for row in rows if row.get("elat_mean_cm") is not None]
    theta = [row["theta_deg"] for row in rows if row.get("theta_deg") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), dpi=150)
    if elat:
        axes[0].boxplot(elat, widths=0.45, patch_artist=True,
                        boxprops=dict(facecolor="#cfe8cf"), medianprops=dict(color="#2ca02c"))
        axes[0].scatter(np.full(len(elat), 1.0), elat, s=14, color="#2ca02c", alpha=0.7, zorder=5)
    axes[0].set_title("逐线平均横向偏差（n=%d）" % len(elat))
    axes[0].set_ylabel("Elat mean / cm")
    axes[0].set_xticks([])
    if theta:
        axes[1].boxplot(theta, widths=0.45, patch_artist=True,
                        boxprops=dict(facecolor="#ffe3c2"), medianprops=dict(color="#ff7f0e"))
        axes[1].scatter(np.full(len(theta), 1.0), theta, s=14, color="#ff7f0e", alpha=0.7, zorder=5)
    axes[1].set_title("逐线方向角偏差（n=%d）" % len(theta))
    axes[1].set_ylabel("Eθ / °")
    axes[1].set_xticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


README_TEMPLATE = """# 中心线评价包（{version}）

- 生成时间：{generated_at}
- 项目缓存：`{cache_dir}`
- 源影像：`{tif_path}`
- 算法版本：MASK_REGULARIZATION_VERSION={mask_version}；评价工具 {version}
- 采样口径：沿参考线 {sample_spacing_m:g} m 等间隔；匹配阈 {match_max_distance_m:g} m

## 指标定义

1. **方向角偏差 Eθ（°）**：参考线与提取中心线（均取最小二乘拟合方向）的
   无向夹角（180° 取小）。
2. **横向偏差 Elat（cm）**：沿参考线等间隔采样，各采样点到提取中心线
   （折线）的最短距离；报告逐线平均与最大。
3. **重叠率 overlap_ratio**：提取中心线在参考线纵向范围内的覆盖比例；
   <1 提示截断或缺失。
4. **人工重复性（noise_floor.csv）**：同一带第 1 遍与第 2 遍参考线之间的
   同口径偏差，即人工判读噪声底；用于解释算法偏差量级。

## 文件

- `per_line.csv`：逐条参考线的指标明细（含未匹配行）。
- `noise_floor.csv`：人工重复性（若存在第 2 遍参考线）。
- `summary.json`：汇总（平均 Eθ、平均/最大横向偏差、版本与溯源信息）。
- `reference_lines.json`：本次评价使用的参考线原文（显示坐标）。
- `overlay.png` / `boxplot.png`：叠加对比图与分布箱线图。

## 复现

```bat
unset PYTHONPATH PYTHONHOME
D:/zhl/anaconda/envs/game/python.exe src/centerline_evaluation.py ^
    --cache-dir "{cache_dir}" --tif "{tif_path}"
```

> 说明：评价会以当前 mask_processing 配置重跑掩膜处理；参考线为人工研究
> 证据（reference_lines.json sidecar），不进入作业状态与撤销历史。
"""


def write_evaluation_package(
    out_dir: str,
    *,
    summary: Dict[str, Any],
    per_line_rows: Sequence[Dict[str, Any]],
    noise_rows: Sequence[Dict[str, Any]],
    ref_payload: Dict[str, Any],
    cache_dir: str,
    mask_local: Optional[np.ndarray],
    local_lines: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    offset: Tuple[int, int],
    make_figures: bool = True,
) -> List[str]:
    """写出评价包全部文件并打包 zip；返回 zip 路径（单元素列表）。"""
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "per_line.csv"), per_line_rows, PER_LINE_COLUMNS)
    if noise_rows:
        _write_csv(os.path.join(out_dir, "noise_floor.csv"), noise_rows, NOISE_COLUMNS)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "reference_lines.json"), "w", encoding="utf-8") as handle:
        json.dump(ref_payload, handle, ensure_ascii=False, indent=2)
    figure_warnings: List[str] = []
    if make_figures:
        try:
            render_overlay_png(
                os.path.join(out_dir, "overlay.png"),
                cache_dir=cache_dir,
                mask_local=mask_local,
                local_lines=local_lines,
                candidates=candidates,
                offset=offset,
            )
            render_boxplot_png(os.path.join(out_dir, "boxplot.png"), per_line_rows)
        except Exception as exc:  # 图形失败不影响数据产物
            figure_warnings.append("%s: %s" % (type(exc).__name__, exc))
    readme = README_TEMPLATE.format(
        version=summary.get("version", CENTERLINE_EVAL_VERSION),
        generated_at=summary.get("generated_at", ""),
        cache_dir=cache_dir,
        tif_path=summary.get("tif_path", ""),
        mask_version=summary.get("mask_regularization_version", ""),
        sample_spacing_m=summary.get("sample_spacing_m", DEFAULT_SAMPLE_SPACING_M),
        match_max_distance_m=summary.get("match_max_distance_m", 0.8),
    )
    if figure_warnings:
        readme += "\n## 图形生成警告\n\n- " + "\n- ".join(figure_warnings) + "\n"
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(readme)
    zip_path = out_dir.rstrip("\\/") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for root, _dirs, files in os.walk(out_dir):
            for name in sorted(files):
                full = os.path.join(root, name)
                bundle.write(full, os.path.relpath(full, out_dir))
    return [zip_path]


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def evaluate_project(
    *,
    cache_dir: str,
    tif_path: str = "",
    out_dir: str = "",
    config: Optional[Dict[str, Any]] = None,
    reference_path: str = "",
    sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
    match_max_distance_m: float = 0.8,
    make_figures: bool = True,
) -> Dict[str, Any]:
    """完整评价一个项目：读参考线 → 重跑中心线 → 计算 → 出评价包。"""
    reference_path = reference_path or os.path.join(cache_dir, "reference_lines.json")
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(
            "未找到参考线文件: %s（请先在桌面端标注参考线，或用 --reference 指定）" % reference_path
        )
    with open(reference_path, "r", encoding="utf-8") as handle:
        ref_payload = normalize_payload(json.load(handle))
    if not ref_payload.get("lines"):
        raise ValueError("参考线文件里没有有效参考线: %s" % reference_path)

    inputs = load_project_inputs(cache_dir, tif_path)
    cfg = dict(config) if config else resolve_repo_config()
    result, bands = extract_centerlines(inputs, cfg)
    candidates = _band_candidates(bands)
    if not candidates:
        raise RuntimeError("当前配置未产生任何行带中心线，无法评价")

    offset = inputs["offset"]
    local_lines: List[Dict[str, Any]] = []
    for line in ref_payload["lines"]:
        record = dict(line)
        record["points"] = _to_local(line["points"], offset)
        record["fit"] = fit_line(record["points"])
        local_lines.append(record)

    rows = evaluate_reference_lines(
        local_lines, candidates, inputs["mpp"],
        sample_spacing_m=sample_spacing_m, match_max_distance_m=match_max_distance_m,
    )
    noise_rows = compute_noise_floor(
        local_lines, inputs["mpp"],
        sample_spacing_m=sample_spacing_m, match_max_distance_m=max(1.0, match_max_distance_m),
    )
    summary = build_summary(
        rows, noise_rows,
        inputs=inputs, cache_dir=cache_dir, tif_path=inputs["tif_path"],
        sample_spacing_m=sample_spacing_m, match_max_distance_m=match_max_distance_m,
        station_spacing_hint_m=float(ref_payload.get("station_spacing_hint_m") or 0.0),
    )
    out_dir = out_dir or os.path.join(cache_dir, "centerline_eval", datetime.now().strftime("%Y%m%d-%H%M%S"))
    zip_paths = write_evaluation_package(
        out_dir,
        summary=summary,
        per_line_rows=rows,
        noise_rows=noise_rows,
        ref_payload=ref_payload,
        cache_dir=cache_dir,
        mask_local=result.get("processed_mask"),
        local_lines=local_lines,
        candidates=candidates,
        offset=offset,
        make_figures=make_figures,
    )
    return {
        "summary": summary,
        "rows": rows,
        "noise_rows": noise_rows,
        "out_dir": out_dir,
        "zip": zip_paths[0],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="中心线评价：提取中心线 vs 人工参考线")
    parser.add_argument("--cache-dir", required=True, help="项目缓存目录（含 mask.npz/meta.json/reference_lines.json）")
    parser.add_argument("--tif", default="", help="当前 result.tif 路径（用于重建地理环境）")
    parser.add_argument("--out", default="", help="输出目录（默认：缓存目录下 centerline_eval/<时间戳>）")
    parser.add_argument("--reference", default="", help="参考线 JSON 路径（默认：缓存目录 reference_lines.json）")
    parser.add_argument("--sample-spacing-m", type=float, default=DEFAULT_SAMPLE_SPACING_M)
    parser.add_argument("--match-threshold-m", type=float, default=0.8)
    parser.add_argument("--config", default="", help="mask_processing 配置文件（默认：仓库 config.json）")
    parser.add_argument("--no-figures", action="store_true", help="跳过 PNG 图形生成")
    args = parser.parse_args(argv)

    config = resolve_repo_config(args.config) if args.config else None
    outcome = evaluate_project(
        cache_dir=args.cache_dir,
        tif_path=args.tif,
        out_dir=args.out,
        config=config,
        reference_path=args.reference,
        sample_spacing_m=args.sample_spacing_m,
        match_max_distance_m=args.match_threshold_m,
        make_figures=not args.no_figures,
    )
    summary = outcome["summary"]
    print("[中心线评价] %s" % summary["version"])
    print("  参考线 %d 条（匹配 %d，未匹配 %d）"
          % (summary["line_count"], summary["matched_line_count"], len(summary["unmatched_line_ids"])))
    if summary.get("theta_mean_deg") is not None:
        print("  Eθ 平均 = %.3f° | Elat 平均 = %.2f cm | Elat 最大 = %.2f cm"
              % (summary["theta_mean_deg"], summary["elat_mean_cm"], summary["elat_max_cm"]))
    if summary.get("noise_floor"):
        nf = summary["noise_floor"]
        print("  人工重复性（%d 对）：Eθ %.3f°，Elat 平均 %.2f cm，最大 %.2f cm"
              % (nf["pair_count"], nf["theta_mean_deg"], nf["elat_mean_cm"], nf["elat_max_cm"]))
    print("  输出：%s" % outcome["out_dir"])
    print("  打包：%s" % outcome["zip"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
