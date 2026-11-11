"""ALIGN 命令层 CLI —— 阶段 1（主流程 12 条 + run-batch）+ 阶段 2（B/C/D 组 14 条 + 并发防护）。

命令行等价界面手动操作：加载影像→圈田块→识别→掩膜→中心线→选线→起点→（参数/禁行区/
参考线/标定线）→规划→模拟→导出（TXT/快照/资料包）；查询（状态/诊断/缓存/恢复）；
医生（环境自检）；维护（回滚/重置）。

设计文档：docs/命令层设计方案.md（§4.4 契约核对底表含每条命令的界面层锚点）。

铁律：
1. 只编排、不发明——全部算法调用界面同款领域函数，本文件不写几何/识别逻辑；
2. 不 import 界面主窗口/工作线程（main_window、workers 等 Qt 外壳）；仅两个例外：
   pyside6_app.task_state（Qt-free 任务状态表，query-status/rollback 复用）与
   pyside6_app.snapshot_export（快照渲染纯函数；导入会拉入 PySide6，仅启动开销）；
3. 命令边界同步强写缓存（cache.save_project_state + wait_for_project_state_saves），
   与界面共用同一缓存根（%LOCALAPPDATA%\\ALIGN\\cache\\md5(tif)[:12]），断点即缓存；
4. 并发防护：写命令持有缓存目录字节锁（cli.lock，进程退出自动释放）；
   保存前比对 project_state.json 磁盘指纹，外部（界面）改动则拒绝覆盖。

退出码：0 成功；2 前置门禁未满足（JSON 附 blocker，文案对齐界面）；3 执行失败；4 参数错误。
所有 stdout 输出为 UTF-8 JSON（单行），供 Agent/脚本直接消费。
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("ALIGN_DISABLE_AUTO_START", "1")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

EXIT_OK = 0
EXIT_GATE = 2
EXIT_FAIL = 3
EXIT_ARGS = 4

CLI_VERSION = "cli-stage1"

# run-batch 阶段顺序（等价界面九张步骤卡片的主链）
BATCH_STAGES = [
    "load-image", "set-field", "load-model", "segment", "process-mask",
    "gen-worklines", "select-worklines", "set-entry", "plan", "export-txt",
]


class GateError(RuntimeError):
    """前置门禁未满足 → 退出码 2。blocker 文案对齐界面；extra.missing 给出补齐项。"""

    def __init__(self, blocker: str, **extra):
        super().__init__(blocker)
        self.blocker = blocker
        self.extra = extra


class ParamError(ValueError):
    """参数错误 → 退出码 4。"""


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str))


def _gate(condition: bool, blocker: str, **extra) -> None:
    if not condition:
        raise GateError(blocker, **extra)


# ---------------------------------------------------------------------------
# 界面层镜像（领域层没有 Qt 无关实现，按 §4.4 锚点逐字镜像，改动须双向同步）
# ---------------------------------------------------------------------------

def _display_affine(source_affine, full_w: int, full_h: int, disp_w: int, disp_h: int):
    """镜像界面层 workers.display_affine（锚点 WK）。"""
    from affine import Affine

    scale_x = full_w / max(1, disp_w)
    scale_y = full_h / max(1, disp_h)
    return source_affine * Affine.scale(scale_x, scale_y)


def _require_metric_scale(geo, state, shape) -> float:
    """镜像界面层 workers.require_metric_scale（DECISION-003 米制尺度守门）。

    界面抛 RuntimeError 弹提示；命令层转 GateError（退出码 2），文案保持一致。
    """
    import numpy as np
    from config import Config

    if geo is None or not geo.is_ready():
        raise GateError("影像没有有效地理配准，不能执行米制掩膜处理或路径规划", missing=["地理配准"])
    if shape is None or len(shape) < 2:
        raise GateError("缺少影像状态或掩膜尺寸，不能验证地面分辨率", missing=["影像尺寸"])
    height, width = int(shape[0]), int(shape[1])
    offset_x = float(getattr(state, "mask_offset_x", 0) or 0)
    offset_y = float(getattr(state, "mask_offset_y", 0) or 0)
    value = float(geo.meters_per_pixel(offset_x + width / 2.0, offset_y + height / 2.0))
    if not np.isfinite(value) or not 1e-5 <= value <= 5.0:
        raise GateError(f"影像地面分辨率无效: {value!r} m/px", missing=["地面分辨率"])
    if hasattr(geo, "pixel_distance_m"):
        samples = [
            (offset_x + width / 2.0, offset_y + height / 2.0),
            (offset_x, offset_y),
            (offset_x + max(0, width - 2), offset_y),
            (offset_x, offset_y + max(0, height - 2)),
            (offset_x + max(0, width - 2), offset_y + max(0, height - 2)),
        ]
        means = []
        anisotropy = []
        for px, py in samples:
            dx = float(geo.pixel_distance_m((px, py), (px + 1.0, py)))
            dy = float(geo.pixel_distance_m((px, py), (px, py + 1.0)))
            if not np.isfinite(dx) or not np.isfinite(dy) or min(dx, dy) <= 0.0:
                raise GateError("影像横向或纵向地面分辨率无效", missing=["地面分辨率"])
            means.append((dx + dy) / 2.0)
            anisotropy.append(max(dx, dy) / min(dx, dy))
        geo_config = Config().section("geo")
        max_anisotropy = float(geo_config.get("max_pixel_anisotropy_ratio", 1.15))
        max_variation = float(geo_config.get("max_gsd_variation_ratio", 0.10))
        if max(anisotropy) > max_anisotropy:
            raise GateError(
                f"影像像素横纵尺度差异过大: ratio={max(anisotropy):.4f}, limit={max_anisotropy:.4f}",
                missing=["地面分辨率"],
            )
        variation = max(means) / min(means) - 1.0
        if variation > max_variation:
            raise GateError(
                f"影像范围内地面分辨率变化过大: variation={variation:.2%}, limit={max_variation:.2%}",
                missing=["地面分辨率"],
            )
        value = float(np.median(means))
    return value


def _path_config_snapshot(overrides: dict | None) -> dict:
    """镜像界面层 workers.path_planning_config_snapshot（DECISION-013，Config 口径）。"""
    from config import Config

    config = Config().section("path_planning")
    config.update(dict(overrides or {}))
    config.setdefault("headland_damage_weight", 1.0)
    config.setdefault("footprint_metric_policy", "report_only")
    return config


# ---------------------------------------------------------------------------
# 命令上下文：tif + cfg + geo + state（自缓存恢复）
# ---------------------------------------------------------------------------

class Ctx:
    """单命令上下文。restore=True 时从项目缓存恢复 AppState（缓存即断点）。"""

    def __init__(self, tif: str, restore: bool = True):
        self.tif = str(Path(tif).expanduser().resolve())
        _gate(os.path.isfile(self.tif), f"找不到影像文件：{self.tif}", missing=["影像文件"])
        from config import Config

        self.cfg = Config()
        self.saved = {}
        self.meta = {}
        self.geo = None
        self.full_w = 0
        self.full_h = 0
        self.disp_w = 0
        self.disp_h = 0
        self.crs = ""
        from state import AppState

        self.state = AppState()
        self._read_header()
        if restore:
            self._restore()
        self._state_sig = self._state_file_sig()

    def _read_header(self) -> None:
        import rasterio

        with rasterio.open(self.tif) as src:
            self.crs = src.crs.to_string() if src.crs else ""
            self._source_transform = src.transform
            self.full_w, self.full_h = int(src.width), int(src.height)

    def _display_dims(self) -> tuple[int, int]:
        meta = self.meta or {}
        try:
            sx = float(meta.get("display_scale_x") or 0.0)
            sy = float(meta.get("display_scale_y") or 0.0)
        except (TypeError, ValueError):
            sx = sy = 0.0
        if sx > 0 and sy > 0:
            return max(1, round(self.full_w / sx)), max(1, round(self.full_h / sy))
        raster_cfg = self.cfg.section("raster") or {}
        max_dim = int(raster_cfg.get("preview_max_dimension", 2600) or 2600)
        keep = min(1.0, max_dim / float(max(self.full_w, self.full_h, 1)))
        return max(1, round(self.full_w * keep)), max(1, round(self.full_h * keep))

    def _build_geo(self):
        from pyproj import Transformer
        from geo import GeoUtils

        geo = GeoUtils()
        geo.clear()
        affine = _display_affine(
            self._source_transform, self.full_w, self.full_h, self.disp_w, self.disp_h
        )
        geo.set_affine(
            affine,
            Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True),
            source_crs=self.crs,
        )
        return geo

    def _restore(self) -> None:
        import cache as cache_mod

        self.saved = cache_mod.load_project_state(self.tif) or {}
        self.meta = cache_mod.load_meta(self.tif) or {}
        self.disp_w, self.disp_h = self._display_dims()
        self.geo = self._build_geo()
        _apply_saved(self.state, self.saved, self.meta)
        self.state.safe_update(
            tif_path=self.tif,
            display_w=self.disp_w,
            display_h=self.disp_h,
            full_width=self.full_w,
            full_height=self.full_h,
        )

    def _state_file_sig(self):
        """project_state.json 的磁盘指纹（存在性 + mtime_ns + size），用于并发防护。"""
        import cache as cache_mod

        path = os.path.join(cache_mod.project_cache_dir(self.tif), "project_state.json")
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (int(st.st_mtime_ns), int(st.st_size))

    def save(self, stage: str) -> None:
        import cache as cache_mod

        # 并发防护（阶段 2）：界面/其他进程在本命令运行期间改写了缓存时拒绝覆盖。
        if self._state_file_sig() != self._state_sig:
            raise GateError(
                "缓存在本命令运行期间被外部修改（可能界面正在操作本项目）；"
                "为避免覆盖他人工作已中止。请重新执行本命令——CLI 会先恢复最新现场再执行。",
                missing=["重新执行命令"],
            )
        if not cache_mod.save_project_state(self.tif, self.state, stage=stage):
            raise RuntimeError(f"缓存写入失败（stage={stage}），已 fail-closed")
        if not cache_mod.wait_for_project_state_saves(self.tif, timeout=30.0):
            raise RuntimeError(f"缓存写入超时（stage={stage}），已 fail-closed")
        self._state_sig = self._state_file_sig()

    def resync(self) -> None:
        """落盘指纹重同步：领域代码自保存（如 inference.py 识别中途 checkpoint 直写
        project_state.json）之后调用，否则后续 save() 会把自家写入误判为外部并发修改。"""
        import cache as cache_mod

        cache_mod.wait_for_project_state_saves(self.tif, timeout=30.0)
        self._state_sig = self._state_file_sig()


def _apply_saved(state, saved: dict, meta: dict) -> None:
    """把 project_state.json 的持久化字段回填进 AppState（界面恢复链同义）。"""
    from state import AutoPathSegment, PathPoint

    if not saved:
        return
    direct_keys = (
        "field_boundary", "field_area_m2", "forbidden_regions", "forbidden_regions_confirmed",
        "current_model_name", "current_model_path", "model_sha256",
        "inference_provenance", "inference_runtime",
        "inference_done", "mask_processed", "mask_offset_x", "mask_offset_y",
        "mask_provenance", "centerline_catalog", "work_band_catalog",
        "selected_centerline_ids", "centerline_selection_report",
        "harvester_params", "harvester_param_sources", "harvester_params_confirmed",
        "turn_strategy", "turn_strategy_user_selected",
        "entry_point", "entry_point_locked", "exit_point", "exit_point_locked",
        "unload_points", "unload_point", "unload_point_locked",
        "workflow_step", "simulation_done", "export_done",
        "auto_path_planned", "auto_path_valid", "auto_path_desc",
        "path_provenance", "path_runtime", "path_validation", "path_artifact",
        "auto_path_geo", "path_points", "path_status", "path_geometry_modes",
        "last_coverage_rate", "last_track_coverage_rate", "last_turning_rate",
        "last_path_length", "last_turning_length", "last_overlap_area_m2",
        "last_overlap_point_count", "last_path_self_intersections",
        "source_sha256", "source_metadata", "raster_preprocessing", "source_identity",
        "last_rolling_rate", "last_crop_mask_geometric_coverage_pct",
        "last_planned_target_geometric_coverage_pct", "last_detected_mask_geometric_coverage_pct",
        "last_field_efficiency", "last_total_path_m", "last_work_path_m",
        "last_turn_path_m", "last_entry_exit_path_m",
    )
    for key in direct_keys:
        if key in saved and saved[key] is not None:
            setattr(state, key, saved[key])
    mask_result = saved.get("mask_result")
    if isinstance(mask_result, dict) and mask_result:
        state.mask_result = dict(mask_result)
    segments = []
    for item in saved.get("auto_path") or []:
        if not isinstance(item, dict):
            continue
        pts = [
            PathPoint(float(x), float(y))
            for x, y in (item.get("points") or [])
            if isinstance(x, (int, float)) and isinstance(y, (int, float))
        ]
        if len(pts) < 2:
            continue
        segments.append(
            AutoPathSegment(
                points=pts,
                status=int(item.get("status", 1) or 1),
                corridor_id=int(item.get("corridor_id", -1)),
                length_m=float(item.get("length_m", 0.0) or 0.0),
                segment_type=str(item.get("segment_type", "work")),
                geometry_mode=str(item.get("geometry_mode", "straight")),
            )
        )
    if segments:
        state.auto_path = segments
        state.auto_path_segments = list(segments)


def _attach_raw_mask(ctx: Ctx):
    import cache as cache_mod

    raw, _ox, _oy = cache_mod.load_mask(ctx.tif, suffix="")
    _gate(raw is not None, "未找到缓存的 AI 掩膜，请先运行 segment（作物识别）", missing=["AI 掩膜"])
    ctx.state.mask_raw = raw
    ctx.state.inference_original_mask = raw.copy()
    return raw


def _attach_processed_mask(ctx: Ctx):
    """恢复掩膜优化结果：镜像界面恢复链 MW:2330-2349 的支撑层推导。"""
    import cache as cache_mod
    import cv2
    import numpy as np

    processed, ox, oy, layers = cache_mod.load_mask_bundle(ctx.tif, suffix="_processed")
    _gate(
        processed is not None,
        "未找到缓存的掩膜优化结果，请先运行 process-mask（掩膜优化）",
        missing=["掩膜优化结果"],
    )
    mr = dict(ctx.state.mask_result or {})
    mr["processed_mask"] = processed
    for key, value in (layers or {}).items():
        if isinstance(value, np.ndarray) and value.shape[:2] == processed.shape[:2]:
            mr[key] = value
    uncertain = mr.get("uncertain_residual_mask")
    if isinstance(uncertain, np.ndarray):
        mr["neutral_support_mask"] = uncertain
    support = processed.copy()
    headland = mr.get("headland_mask")
    if isinstance(headland, np.ndarray):
        support = cv2.bitwise_or(support, headland)
    if isinstance(uncertain, np.ndarray):
        support = cv2.bitwise_or(support, uncertain)
    mr["planning_support_mask"] = support
    ctx.state.mask_result = mr
    ctx.state.mask_offset_x = ox
    ctx.state.mask_offset_y = oy
    ctx.state.mask_processed = True
    return processed


def _invalidate_from(state, stage: str) -> None:
    """镜像界面 _invalidate_analysis_from 的状态失效语义（不动缓存文件）。"""
    if stage in ("inference", "mask", "path"):
        if stage == "inference":
            state.safe_update(inference_done=False, mask_raw=None, inference_original_mask=None)
        if stage in ("inference", "mask"):
            state.safe_update(mask_processed=False, mask_result=None)
        state.safe_update(
            auto_path_planned=False, auto_path_valid=False, auto_path=None,
            auto_path_segments=None, auto_path_geo=None, auto_path_desc="",
            simulation_done=False, export_done=False,
        )


def _effective_harvester_params(cfg, state_params) -> dict:
    """镜像界面 _effective_harvester_params（MW:261-287）：属性默认 ← config.json ← state。"""
    params = {
        "cutter_width_m": float(cfg.CUTTER_WIDTH_M),
        "cutter_forward_offset_m": float(cfg.CUTTER_FORWARD_OFFSET_M),
        "track_width_m": float(cfg.TRACK_WIDTH_M),
        "track_gauge_m": float(cfg.TRACK_GAUGE_M),
        "wheelbase_m": float(cfg.WHEELBASE_M),
        "track_length_m": float(cfg.TRACK_LENGTH_M),
        "turn_radius_m": float(cfg.TURN_RADIUS_M),
        "work_speed_mps": float(cfg.WORK_SPEED_MPS),
        "turn_speed_mps": float(cfg.TURN_SPEED_MPS),
        "entry_speed_mps": float(cfg.ENTRY_SPEED_MPS),
    }
    for source in (cfg.section("harvester"), state_params or {}):
        if not isinstance(source, dict):
            continue
        for key in params:
            value = source.get(key)
            if value in (None, ""):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                params[key] = value
    return params


def _catalog_items(ctx: Ctx) -> list:
    catalog = ctx.state.centerline_catalog or {}
    return list(catalog.get("items") or [])


def _selected_ids(ctx: Ctx) -> list:
    return [str(x) for x in (ctx.state.selected_centerline_ids or [])]


def _mask_result_or_gate(ctx: Ctx) -> dict:
    mr = ctx.state.mask_result
    _gate(isinstance(mr, dict) and bool(mr), "请先完成掩膜优化", missing=["掩膜优化"])
    return mr


# ---------------------------------------------------------------------------
# 主流程命令
# ---------------------------------------------------------------------------

def do_load_image(ctx: Ctx, **_kw) -> dict:
    """load-image：等价界面「加载影像」（MW:2464→1872→WK:469）。写入初始项目状态。"""
    _gate(bool(ctx.crs), "影像缺少投影信息，无法规划路径（GUI 同门禁）", missing=["投影信息"])
    from provenance import file_sha256

    sha = file_sha256(ctx.tif)
    restored = bool(ctx.saved)
    ctx.state.safe_update(
        tif_path=ctx.tif,
        source_sha256=sha,
        source_metadata={"crs": ctx.crs, "full_width": ctx.full_w, "full_height": ctx.full_h},
        display_w=ctx.disp_w,
        display_h=ctx.disp_h,
        full_width=ctx.full_w,
        full_height=ctx.full_h,
        mask_offset_x=int(getattr(ctx.state, "mask_offset_x", 0) or 0),
        mask_offset_y=int(getattr(ctx.state, "mask_offset_y", 0) or 0),
        raster_preprocessing=ctx.cfg.section("raster_preprocessing") or {},
    )
    ctx.save("load_image")
    return {
        "tif": ctx.tif,
        "crs": ctx.crs,
        "full_size": [ctx.full_w, ctx.full_h],
        "display_size": [ctx.disp_w, ctx.disp_h],
        "source_sha256": sha,
        "cache_restored": restored,
        "workflow_step": getattr(ctx.state, "workflow_step", -1),
    }


def _parse_polygon_arg(value: str) -> list:
    if value.startswith("@"):
        value = Path(value[1:]).read_text(encoding="utf-8")
    elif value.strip().endswith(".json") and os.path.isfile(value):
        value = Path(value).read_text(encoding="utf-8")
    data = json.loads(value)
    pts = [(float(p[0]), float(p[1])) for p in data]
    if len(pts) < 3:
        raise ValueError("田块边界至少需要 3 个点")
    return pts


def _polygon_area_px(pts) -> float:
    """镜像界面 _polygon_area_px（鞋带公式）。"""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _field_area_m2(geo, pts) -> float:
    """镜像界面 _field_area_m2：pixel_polygon_area_m2 优先，失败退回形心 ppm 换算。"""
    import math

    if not pts or len(pts) < 3:
        return 0.0
    if geo is not None and geo.is_ready():
        try:
            area_m2 = float(geo.pixel_polygon_area_m2(pts))
            if math.isfinite(area_m2) and area_m2 > 0:
                return area_m2
        except Exception:
            pass
        try:
            import numpy as np

            arr = np.asarray(pts, dtype=np.float64)
            ppm = geo.pixels_per_meter(float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
            if ppm > 0:
                return _polygon_area_px(pts) / (ppm * ppm)
        except Exception:
            pass
    return 0.0


def do_set_field(ctx: Ctx, polygon: str | None = None, full_frame: bool = False, **_kw) -> dict:
    """set-field：等价界面「确认田块边界」（MW:3921→4100；RG:126）。"""
    _gate(bool(ctx.saved) or bool(ctx.state.source_sha256), "请先运行 load-image 加载影像", missing=["影像"])
    if full_frame:
        points = [(0.0, 0.0), (float(ctx.disp_w), 0.0), (float(ctx.disp_w), float(ctx.disp_h)), (0.0, float(ctx.disp_h))]
    else:
        _gate(bool(polygon), "请用 --polygon '<json>' 或 --full-frame 指定田块边界", missing=["田块边界"])
        points = _parse_polygon_arg(polygon)
    from row_geometry import sanitize_field_polygon_report

    ring, report = sanitize_field_polygon_report(points)
    _gate(
        ring is not None,
        f"田块边界不可用（{report.get('reason') or 'unknown'}），请在影像上重新圈选",
        missing=["田块边界"],
    )
    clean = [(float(p[0]), float(p[1])) for p in ring]
    area_m2 = _field_area_m2(ctx.geo, clean) or None
    had_analysis = bool(getattr(ctx.state, "inference_done", False) or getattr(ctx.state, "auto_path", None))
    if had_analysis:
        _invalidate_from(ctx.state, "inference")
    ctx.state.safe_update(
        field_boundary=list(clean),
        field_area_m2=area_m2,
        field_polygon_source="cli",
        field_polygon_prompted=True,
    )
    from workflow import WorkflowUpdater

    WorkflowUpdater.advance(ctx.state, "BOUNDARY_SET")
    ctx.save("field")
    return {
        "points": len(clean),
        "area_m2": area_m2,
        "invalidated_downstream": had_analysis,
        "full_frame": full_frame,
    }


def do_load_model(ctx: Ctx, model: str, **_kw) -> dict:
    """load-model：等价界面「加载模型」（校验可加载 + 记录身份指纹）。"""
    model_path = str(Path(model).expanduser().resolve())
    _gate(os.path.isfile(model_path), f"模型文件不存在：{model_path}", missing=["模型文件"])
    from inference import ModelEngine
    from provenance import file_sha256

    engine = ModelEngine()
    if not engine.load(model_path):
        raise RuntimeError(f"模型加载失败：{model_path}")
    sha = file_sha256(model_path)
    old_sha = getattr(ctx.state, "model_sha256", None)
    if old_sha and sha and str(old_sha) != str(sha) and getattr(ctx.state, "inference_done", False):
        _invalidate_from(ctx.state, "inference")
    ctx.state.safe_update(
        current_model_path=model_path,
        current_model_name=Path(model_path).stem,
        model_sha256=sha,
    )
    ctx.save("load_model")
    return {"model": model_path, "model_sha256": sha, "device": getattr(engine, "device", "")}


def do_segment(ctx: Ctx, timeout_s: int = 1800, **_kw) -> dict:
    """segment：等价界面「作物识别」（MW:6743→InferenceRunner；INF:705/773）。"""
    _gate(bool(ctx.state.tif_path), "请先运行 load-image 加载影像", missing=["影像"])
    boundary = list(ctx.state.field_boundary or [])
    _gate(len(boundary) >= 3, "请先设置田块边界（set-field）", missing=["田块边界"])
    model_path = getattr(ctx.state, "current_model_path", None)
    _gate(bool(model_path) and os.path.isfile(str(model_path)), "请先加载模型（load-model）", missing=["模型"])
    _require_metric_scale(ctx.geo, ctx.state, (ctx.disp_h, ctx.disp_w))

    from inference import InferenceRunner, ModelEngine

    engine = ModelEngine()
    if not engine.load(str(model_path)):
        raise RuntimeError(f"模型加载失败：{model_path}")
    runner = InferenceRunner(ctx.state, engine, ctx.geo)
    t0 = time.time()
    runner.start_from_tif(ctx.tif, (ctx.disp_h, ctx.disp_w), boundary, 1)
    while ctx.state.inference_running:
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"作物识别超时（>{timeout_s}s）")
        time.sleep(0.25)
    mask = ctx.state.mask_raw
    if mask is None or not ctx.state.inference_done:
        raise RuntimeError("作物识别失败：未产出掩膜（详见进程日志）")
    ctx.resync()  # InferenceRunner 中途自保存 project_state（INF:1568/1750），同步落盘指纹
    import numpy as np

    ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    # InferenceRunner 已自动持久化 mask.npz + project_state（INF:1521/1568），不重复写
    return {
        "mask_shape": list(mask.shape),
        "foreground_ratio": round(ratio, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "device": getattr(engine, "device", ""),
    }


def do_process_mask(ctx: Ctx, **_kw) -> dict:
    """process-mask：等价界面「掩膜优化」（MW:8618→WK:1104；MP:634）。"""
    _gate(bool(ctx.state.inference_done), "请先运行 segment（作物识别）", missing=["作物识别"])
    meta_sha = str(ctx.meta.get("model_sha256") or "")
    state_sha = str(getattr(ctx.state, "model_sha256", "") or "")
    if meta_sha and state_sha:
        _gate(meta_sha == state_sha, "当前模型与掩膜缓存不匹配，请重新运行 segment", missing=["模型一致性"])
    raw = _attach_raw_mask(ctx)
    _require_metric_scale(ctx.geo, ctx.state, raw.shape[:2])

    from mask_processor import process_mask

    t0 = time.time()
    mask_cfg = dict(ctx.cfg.section("mask_processing") or {})
    processed = process_mask(raw, geo=ctx.geo, state=ctx.state, config=mask_cfg)
    if not isinstance(processed, dict) or processed.get("processed_mask") is None:
        raise RuntimeError("掩膜优化失败：process_mask 未返回 processed_mask")
    # 溯源 + 持久化（镜像界面层 MaskProcessWorker 与 _save_processed_mask_async）
    import numpy as np
    from provenance import make_stage_record
    from row_geometry import MASK_REGULARIZATION_VERSION

    provenance_inputs = {
        "inference_fingerprint": str(
            (getattr(ctx.state, "inference_provenance", {}) or {}).get("fingerprint", "")
        ),
        "raw_mask_sha256": str(
            (getattr(ctx.state, "inference_provenance", {}) or {}).get("artifact_sha256", "")
        ),
        "mask_config": {
            key: value for key, value in mask_cfg.items() if not str(key).startswith("_")
        },
    }
    processed["provenance"] = make_stage_record(
        "mask", MASK_REGULARIZATION_VERSION, provenance_inputs, processed["processed_mask"]
    )
    processed["runtime"] = {
        "stage": "mask_processing",
        "seconds": float(time.time() - t0),
        "input_shape": [int(v) for v in raw.shape[:2]],
        "input_nonzero_pixels": int(np.count_nonzero(raw)),
    }
    ctx.state.mask_provenance = processed["provenance"]
    dense_keys = (
        "regularized_body_mask", "body_residual_mask", "gap_repair_added_mask",
        "support_clip_removed_mask", "headland_corridor_mask", "headland_mask",
        "uncertain_residual_mask",
    )
    extras = {key: processed[key] for key in dense_keys if processed.get(key) is not None}
    import cache as cache_mod

    cache_mod.save_mask(
        ctx.tif,
        processed["processed_mask"],
        offset_x=int(getattr(ctx.state, "mask_offset_x", 0) or 0),
        offset_y=int(getattr(ctx.state, "mask_offset_y", 0) or 0),
        suffix="_processed",
        extra_arrays=extras,
    )
    ctx.state.mask_result = processed
    ctx.state.mask_processed = True
    from workflow import WorkflowUpdater

    WorkflowUpdater.advance(ctx.state, "MASK_PROCESSED")
    ctx.save("mask_processed")
    proc = processed["processed_mask"]
    import numpy as np

    return {
        "mask_shape": list(proc.shape),
        "foreground_ratio": round(float(np.count_nonzero(proc)) / float(proc.size), 4) if proc.size else 0.0,
        "wide_bands": len(processed.get("wide_bands") or []),
        "narrow_bands": len(processed.get("narrow_bands") or []),
        "all_bands": len(processed.get("all_bands") or []),
        "main_angle": round(float(processed.get("main_angle", 0.0) or 0.0), 2),
    }


def do_gen_worklines(ctx: Ctx, **_kw) -> dict:
    """gen-worklines：等价界面「生成作业线」（MW:1447→WK:1188；PP:9689）。"""
    _gate(bool(ctx.state.mask_processed), "请先完成掩膜优化（process-mask）", missing=["掩膜优化"])
    processed = _attach_processed_mask(ctx)
    raw = _attach_raw_mask(ctx)
    mr = _mask_result_or_gate(ctx)
    _require_metric_scale(ctx.geo, ctx.state, processed.shape[:2])

    from path_planner import prepare_operator_centerline_plan

    config = dict(ctx.cfg.section("path_planning") or {})
    prepared = prepare_operator_centerline_plan(
        mr.get("wide_bands") or [],
        processed,
        mr.get("main_angle", 0.0) or 0.0,
        geo=ctx.geo,
        state=ctx.state,
        config=config,
        detected_mask=raw,
        all_bands=mr.get("all_bands") or [],
    )
    layout = prepared.get("layout") or {}
    catalog = prepared.get("catalog") or layout.get("centerline_catalog") or {}
    items = list(catalog.get("items") or [])
    _gate(bool(items), "未找到可用作业通道，请检查掩膜优化结果", missing=["作业线"])
    ctx.state.centerline_catalog = catalog
    ctx.save("centerline_catalog")
    return {
        "centerlines": len(items),
        "fallback": bool(prepared.get("fallback")),
        "main_angle": round(float(mr.get("main_angle", 0.0) or 0.0), 2),
    }


def do_select_worklines(ctx: Ctx, ids: str | None = None, select_all: bool = False, **_kw) -> dict:
    """select-worklines：等价界面「选择作业线→确认」（MW:1396→1372）。空选 fail-closed。"""
    items = _catalog_items(ctx)
    _gate(bool(items), "请先完成'生成作业线'，再勾选要采用的作业线。", missing=["作业线目录"])
    all_ids = [str(item.get("id")) for item in items if item.get("id") is not None]
    if select_all or ids is None:
        chosen = all_ids
    else:
        chosen = [x.strip() for x in ids.split(",") if x.strip()]
        unknown = [x for x in chosen if x not in all_ids]
        _gate(not unknown, f"作业线编号不存在：{unknown}（可选：{all_ids}）", missing=["有效编号"])
    _gate(bool(chosen), "请先至少勾选一条作业线。", missing=["作业线勾选"])
    report = {
        "selected_ids": list(chosen),
        "selection_origin": "operator",
        "total_candidates": len(all_ids),
        "source": "cli",
    }
    ctx.state.safe_update(
        selected_centerline_ids=list(chosen),
        centerline_selection_report=report,
    )
    ctx.save("centerline_selection")
    return {"selected": chosen, "total": len(all_ids)}


def do_set_entry(ctx: Ctx, auto: bool = False, point: str | None = None, **_kw) -> dict:
    """set-entry：等价界面「智能推荐起点/终点」或手动定起点（MW:6573/6620→PP:1142）。"""
    mr = _mask_result_or_gate(ctx)
    processed = mr.get("processed_mask")
    if processed is None:
        processed = _attach_processed_mask(ctx)
        mr = ctx.state.mask_result
    if auto:
        _attach_raw_mask(ctx)  # prepare_band_centerlines 内部按 state.mask_raw 取尺度参照
        from path_planner import prepare_band_centerlines

        lines, _report = prepare_band_centerlines(
            mr.get("wide_bands") or [],
            float(mr.get("main_angle", 0.0) or 0.0),
            geo=ctx.geo,
            state=ctx.state,
            config=ctx.cfg.section("path_planning"),
        )
        _gate(bool(lines), "未找到可用作业线，请先完成 掩膜优化。", missing=["作业线"])
        start = lines[0][0]
        direction = (lines[0][1][0] - start[0], lines[0][1][1] - start[1])
        import math

        norm = math.hypot(*direction)
        _gate(norm > 1e-6, "未找到可用作业线，请先完成 掩膜优化。", missing=["作业线"])
        ox = float(getattr(ctx.state, "mask_offset_x", 0) or 0.0)
        oy = float(getattr(ctx.state, "mask_offset_y", 0) or 0.0)
        ref_global = (start[0] + ox, start[1] + oy)
        px_dist = float(ctx.geo.pixel_distance_m(ref_global, (ref_global[0] + 1.0, ref_global[1])) or 0.0)
        _gate(px_dist > 0, "地理尺度无效，请检查影像投影与仿射参数", missing=["地理尺度"])
        ext_px = 1.0 / max(px_dist, 1e-6)
        entry = (start[0] - direction[0] / norm * ext_px, start[1] - direction[1] / norm * ext_px)
        if not getattr(ctx.state, "exit_point_locked", False):
            ctx.state.safe_update(exit_point=None, exit_point_locked=False)
    else:
        _gate(bool(point), "请用 --auto 或 --point x,y 指定起点", missing=["起点"])
        x_str, y_str = str(point).split(",")
        entry = (float(x_str), float(y_str))
    had_path = bool(getattr(ctx.state, "auto_path", None))
    if had_path:
        _invalidate_from(ctx.state, "path")
    ctx.state.safe_update(entry_point=entry, entry_point_locked=True)
    ctx.save("entry")
    return {"entry": [round(entry[0], 1), round(entry[1], 1)], "auto": auto, "invalidated_path": had_path}


def do_set_exit(ctx: Ctx, point: str | None = None, clear: bool = False, **_kw) -> dict:
    """set-exit：终点可选；清空与手动设置等价界面行为。"""
    if clear:
        ctx.state.safe_update(exit_point=None, exit_point_locked=False)
        ctx.save("exit")
        return {"exit": None, "locked": False}
    _gate(bool(point), "请用 --point x,y 指定终点，或 --clear 清空", missing=["终点"])
    x_str, y_str = str(point).split(",")
    exit_point = (float(x_str), float(y_str))
    if getattr(ctx.state, "auto_path", None):
        _invalidate_from(ctx.state, "path")
    ctx.state.safe_update(exit_point=exit_point, exit_point_locked=True)
    ctx.save("exit")
    return {"exit": [round(exit_point[0], 1), round(exit_point[1], 1)], "locked": True}


def do_set_unload(ctx: Ctx, points: str | None = None, clear: bool = False, **_kw) -> dict:
    """set-unload：卸粮点列表（可选）。格式 --points 'x,y;x,y'。"""
    if clear:
        ctx.state.safe_update(unload_points=[])
        ctx.save("unload")
        return {"unload_points": 0}
    _gate(bool(points), "请用 --points 'x,y;x,y' 指定卸粮点，或 --clear 清空", missing=["卸粮点"])
    pts = []
    for chunk in str(points).split(";"):
        x_str, y_str = chunk.split(",")
        pts.append((float(x_str), float(y_str)))
    if getattr(ctx.state, "auto_path", None):
        _invalidate_from(ctx.state, "path")
    ctx.state.safe_update(unload_points=pts)
    ctx.save("unload")
    return {"unload_points": len(pts)}


def _status_for_segment_type(segment_type: str) -> int:
    """镜像界面 _status_for_segment_type（MW:7011）。"""
    if segment_type == "work":
        return 1
    if segment_type == "entry":
        return 2
    if segment_type == "exit":
        return 3
    if segment_type in ("turn_reverse", "turn_aux"):
        return 4
    return 0


def _auto_segments_from_path(full_path, segment_types, segment_lengths_m):
    """镜像界面 _auto_segments_from_path（MW:8024-8062）：full_path 是分段列表，一段一 AutoPathSegment。"""
    from state import AutoPathSegment, PathPoint

    segments = []
    for index, points in enumerate(full_path or []):
        segment_type = (
            segment_types[index]
            if index < len(segment_types)
            else ("work" if index % 2 == 0 else "turn")
        )
        pts = [PathPoint(float(p[0]), float(p[1])) for p in points]
        if len(pts) < 2:
            continue
        length_m = float(segment_lengths_m[index]) if index < len(segment_lengths_m) else 0.0
        segments.append(
            AutoPathSegment(
                points=pts,
                status=_status_for_segment_type(segment_type),
                corridor_id=index,
                length_m=length_m,
                segment_type=segment_type,
                geometry_mode=("turn" if str(segment_type).startswith("turn") else "straight"),
            )
        )
    return segments


def _jsonable(value):
    """镜像界面 _manifest_jsonable（MW:7597）：numpy/非标量统一转可 JSON 序列化。"""
    import math

    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _semantic_mask_fingerprints(config):
    """镜像 WK:407-421（DECISION-001/002/009）：逐语义掩膜按内容取指纹。"""
    from provenance import artifact_sha256

    evidence = dict(config or {})
    return {
        key: artifact_sha256(evidence.get(key))
        for key in ("headland_mask", "planning_support_mask", "neutral_support_mask", "uncertain_mask", "forbidden_mask")
    }


def _attach_path_provenance(result, mask_result, state, executed_config):
    """镜像 WK:424-466（DECISION-001/013）。纯字典逻辑，因宿主在 Qt 外壳目录而内联镜像。"""
    from path_planner import PATH_PLANNING_VERSION
    from provenance import make_stage_record

    excluded = {
        "headland_mask", "planning_support_mask", "uncertain_residual_mask",
        "uncertain_mask", "forbidden_mask", "neutral_support_mask",
    }
    path_config = {
        key: value
        for key, value in dict(executed_config or {}).items()
        if not str(key).startswith("_") and key not in excluded
    }
    path_inputs = {
        "mask_fingerprint": str((dict(mask_result or {}).get("provenance", {}) or {}).get("fingerprint", "")),
        "path_config": path_config,
        "harvester": dict(getattr(state, "harvester_params", {}) or {}),
        "harvester_param_sources": dict(getattr(state, "harvester_param_sources", {}) or {}),
        "turn_strategy": str(getattr(state, "turn_strategy", "auto")),
        "entry_point": getattr(state, "entry_point", None),
        "exit_point": getattr(state, "exit_point", None),
        "unload_points": list(getattr(state, "unload_points", []) or []),
        "forbidden_regions": list(getattr(state, "forbidden_regions", []) or []),
        "forbidden_regions_confirmed": bool(getattr(state, "forbidden_regions_confirmed", False)),
        "semantic_mask_fingerprints": _semantic_mask_fingerprints(executed_config),
    }
    path_artifact = {
        "full_path": result.get("full_path", []),
        "segment_types": result.get("segment_types", []),
        "layout": result.get("layout", {}),
    }
    result["provenance"] = make_stage_record("path", PATH_PLANNING_VERSION, path_inputs, path_artifact)


def do_plan(ctx: Ctx, **_kw) -> dict:
    """plan：等价界面「路径规划」（MW:6844→PlanWorker→PP:9808）。门禁顺序对齐 MW:6859-6918。"""
    t0 = time.perf_counter()
    from band_evaluation import planning_config_with_mask_evidence
    from cache import load_operator_paths
    from operator_paths import planner_overrides
    from row_geometry import MASK_REGULARIZATION_VERSION

    # 人工标定线接入（界面 MW:6855-6869 同款；planner_overrides 内部 fail-safe，
    # 版本/影像对不上的线一律不用，返回空替换表）
    operator_payload = load_operator_paths(ctx.tif) or {}
    overrides = {}
    operator_note = ""
    if operator_payload.get("lines"):
        overrides, op_report = planner_overrides(
            operator_payload, mask_version=int(MASK_REGULARIZATION_VERSION), tif_path=ctx.tif
        )
        reason = str(op_report.get("reason") or "")
        if reason in {"mask_version_mismatch", "tif_mismatch"}:
            operator_note = "人工标定线已失效，本次规划使用算法中心线"
    _gate(ctx.state.mask_result is not None, "请先完成掩膜优化", missing=["掩膜优化"])
    _gate(bool(_catalog_items(ctx)), "请先完成'生成作业线'，再在'选择作业线'中勾选要采用的作业线。", missing=["作业线目录"])
    selected_ids = _selected_ids(ctx)
    _gate(bool(selected_ids), "请先在'选择作业线'中至少勾选一条作业线。", missing=["作业线勾选"])
    try:
        hp = _effective_harvester_params(ctx.cfg, getattr(ctx.state, "harvester_params", None))
    except (TypeError, ValueError):
        hp = None
    if hp is None:
        raise GateError(
            "缺少割刀前置距离：规划会按割刀触地点校验碾压，缺失会导致结果不可用。"
            "请先在界面或 config.json 的 harvester.cutter_forward_offset_m 实测补录；如确认不校验，显式填 0。",
            missing=["cutter_forward_offset_m"],
        )
    ctx.state.harvester_params = hp
    _gate(bool(getattr(ctx.state, "entry_point_locked", False) and getattr(ctx.state, "entry_point", None)),
          "请先设置并确认起点；终点会沿最终作业段自动延伸，卸粮点按需添加。", missing=["起点"])
    if getattr(ctx.state, "exit_point", None) is not None and not getattr(ctx.state, "exit_point_locked", False):
        ctx.state.exit_point = None
    _gate(hp.get("cutter_width_m") is not None and hp["cutter_width_m"] > 0,
          "请先配置收割机参数", missing=["割台宽度"])
    if hp.get("cutter_forward_offset_m") is None:
        raise GateError(
            "缺少割刀前置距离：规划会按割刀触地点校验碾压，缺失会导致结果不可用。"
            "请先在界面或 config.json 的 harvester.cutter_forward_offset_m 实测补录；如确认不校验，显式填 0。",
            missing=["cutter_forward_offset_m"],
        )
    processed = _attach_processed_mask(ctx)
    raw = _attach_raw_mask(ctx)
    mr = ctx.state.mask_result
    _require_metric_scale(ctx.geo, ctx.state, processed.shape[:2])
    # 规划前检查（MW:6829-6842）
    if not getattr(ctx.state, "forbidden_regions_confirmed", False):
        ctx.state.safe_update(forbidden_regions_confirmed=True)
    missing = []
    if len(getattr(ctx.state, "field_boundary", []) or []) < 3:
        missing.append("田块边界")
    import numpy as np

    if not isinstance(mr.get("planning_support_mask"), np.ndarray):
        missing.append("场地支撑层")
    if not ctx.geo.is_ready():
        missing.append("GeoTIFF 地理坐标")
    _gate(not missing, "规划前检查未通过：缺少 " + "、".join(missing) + "，请补齐后再生成田间试验路径。", missing=missing)

    selection_report = getattr(ctx.state, "centerline_selection_report", None) or {}
    runtime = {
        "min_turn_radius_m": float(hp["turn_radius_m"]),
        "turn_strategy": str(getattr(ctx.state, "turn_strategy", "auto") or "auto"),
        "service_point_mode": "reference_only",
        "include_route_finish_extension": True,
        "forbidden_mask_confirmed": bool(getattr(ctx.state, "forbidden_regions_confirmed", False)),
    }
    if selected_ids and selection_report.get("selection_origin") == "operator":
        runtime["selected_centerline_ids"] = list(selected_ids)
    # 人工标定线优先：替换表进 runtime（界面同款键 operator_centerlines）
    if overrides:
        runtime["operator_centerlines"] = overrides
    from path_planner import normalize_turn_strategy

    if normalize_turn_strategy(runtime["turn_strategy"]) == "contour":
        runtime["route_order"] = "contour"
    planning = _path_config_snapshot(runtime)
    planning = planning_config_with_mask_evidence(planning, mr, ctx.state, processed.shape)

    from path_planner import build_band_mask, global_path_to_geo, plan_path

    result = plan_path(
        mr.get("wide_bands") or [],
        processed,
        mr.get("main_angle", 0.0) or 0.0,
        ctx.geo,
        ctx.state,
        planning,
        detected_mask=raw,
        planned_mask=build_band_mask(processed.shape, mr.get("wide_bands") or []),
        all_bands=mr.get("all_bands") or [],
    )
    full_path = result.get("full_path") or []
    _gate(bool(full_path), f"路径规划失败：{result.get('description') or '未生成路径'}")
    segment_types = result.get("segment_types") or []
    segment_lengths_m = result.get("segment_lengths_m") or []
    # 界面同款（WK:1341-1348）：按 layout 实际选中集回填配置后再挂路径溯源
    layout = result.get("layout") or {}
    selection = layout.get("centerline_selection")
    if not isinstance(selection, dict):
        selection = layout.get("work_band_selection")
    if isinstance(selection, dict):
        planning["selected_centerline_ids"] = list(selection.get("selected_ids") or [])
    _attach_path_provenance(result, mr, ctx.state, planning)
    result["runtime"] = {
        "stage": "path_planning",
        "seconds": float(time.perf_counter() - t0),
        "mask_shape": [int(v) for v in processed.shape[:2]],
        "band_count": len(mr.get("wide_bands") or []),
    }
    segments = _auto_segments_from_path(full_path, segment_types, segment_lengths_m)
    # 界面同款（MW:8173-8181）：geo 就绪时以 global_path_to_geo 重算经纬度序列
    geo_points = result.get("geo_points") or []
    if ctx.geo is not None and ctx.geo.is_ready():
        geo_points = global_path_to_geo(full_path, ctx.geo, segment_types=segment_types) or []

    validation = result.get("validation") or {}
    updates = dict(
        auto_path=segments,
        auto_path_segments=list(segments),
        auto_path_geo=geo_points,
        auto_path_valid=bool(result.get("is_valid", validation.get("valid", False))),
        auto_path_desc=result.get("description", ""),
        auto_path_planned=bool(segments),
        # 控制折线（path_points/path_status/path_geometry_modes）服务于界面路线编辑器，
        # 命令层无拖拽编辑，留空；正式数据以 auto_path/auto_path_geo 为准。
        path_points=[],
        path_status=[],
        path_geometry_modes=[],
        path_validation=dict(validation),
        path_provenance=dict(result.get("provenance") or {}),
        path_runtime=dict(result.get("runtime") or {}),
        path_artifact={
            key: _jsonable(result[key])
            for key in (
                "turn_assessment", "planning_factors", "requested_strategy", "coordinate_space",
                "machine_readiness", "machine_executable", "path_candidates", "layout",
            )
            if key in result
        },
        simulation_done=False,
        export_done=False,
    )
    # 界面同款指标落盘映射（MW:8227-8240）
    for state_name, metric_name in {
        "last_rolling_rate": "rolling_canopy_pct",
        "last_crop_mask_geometric_coverage_pct": "crop_mask_geometric_coverage_pct",
        "last_planned_target_geometric_coverage_pct": "planned_target_geometric_coverage_pct",
        "last_detected_mask_geometric_coverage_pct": "detected_mask_geometric_coverage_pct",
        "last_field_efficiency": "field_efficiency_pct",
        "last_total_path_m": "total_length_m",
        "last_work_path_m": "work_length_m",
        "last_turn_path_m": "turn_length_m",
        "last_entry_exit_path_m": "entry_exit_length_m",
    }.items():
        if validation.get(metric_name) is not None:
            updates[state_name] = float(validation[metric_name])
    ctx.state.safe_update(**updates)
    from workflow import WorkflowUpdater

    WorkflowUpdater.advance(ctx.state, "PATH_PLANNED")
    ctx.save("path_planned")
    work_n = sum(1 for t in segment_types if t == "work")
    turn_n = sum(1 for t in segment_types if str(t).startswith("turn"))
    is_valid = bool(result.get("is_valid", validation.get("valid", False)))
    out = {
        "segments": len(segments),
        "work_segments": work_n,
        "turn_segments": turn_n,
        "geo_points": len(geo_points),
        "valid": is_valid,
        "description": result.get("description"),
        "turn_strategy": runtime["turn_strategy"],
        "operator_note": operator_note,
        "metrics": {
            "total_length_m": round(float(validation.get("total_length_m") or 0.0), 1),
            "work_length_m": round(float(validation.get("work_length_m") or 0.0), 1),
            "turn_length_m": round(float(validation.get("turn_length_m") or 0.0), 1),
            "rolling_canopy_pct": round(float(validation.get("rolling_canopy_pct") or 0.0), 1),
            "rolling_core_pct": round(float(validation.get("rolling_core_pct") or 0.0), 1),
            "track_core_overlap_pct": round(float(validation.get("track_core_overlap_pct") or 0.0), 1),
            "planned_target_geometric_coverage_pct": round(float(validation.get("planned_target_geometric_coverage_pct") or 0.0), 1),
            "issues": list(validation.get("issues") or []),
        },
    }
    if not is_valid:
        # 界面同款口径（MW:1659-1704）：valid=False 仅告警不拦截，路径照常落盘可导出
        out["warning"] = (result.get("description") or "路径校验未通过").strip()
    return out


def _layout_has_recovery(layout) -> bool:
    """镜像界面 _layout_has_planning_recovery（MW:7346-7357）：递归查 planning_recovered=True。"""
    def walk(value):
        if isinstance(value, dict):
            if value.get("planning_recovered") is True:
                return True
            return any(walk(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(walk(item) for item in value)
        return False

    return walk(layout)


def do_export_txt(ctx: Ctx, out: str, force: bool = False, **_kw) -> dict:
    """export-txt：等价界面「机手兼容路径导出」（MW:7410-7465 风险确认 → PP:5649）。

    界面的「导出确认」弹窗在命令层映射为 --force 预授权：不带 --force 时
    全部风险项以门禁（exit 2）列出，一条不漏。
    """
    risks, geo_points = _export_risk_gate(ctx, force)

    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    from path_planner import export_machine_compatibility_path

    export_result = export_machine_compatibility_path(geo_points, str(out_dir))
    if not export_result:
        raise RuntimeError("机手兼容路径导出失败（export_machine_compatibility_path 返回空）")
    export_diag = export_result[1] if isinstance(export_result, tuple) and len(export_result) > 1 else {}
    output_files = list(export_diag.get("output_files") or export_diag.get("files") or [])
    if not output_files:
        output_files = sorted(str(p) for p in out_dir.glob("*.txt"))
    _gate(bool(output_files), "导出后未找到输出文件，已 fail-closed")

    # 导出后回读验证（规格门槛：$PATH/$INFM/条数）
    import hashlib

    files_info = []
    for f in output_files:
        path = Path(f)
        _gate(path.is_file(), f"导出文件缺失：{f}")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        _gate(any("$PATH" in ln for ln in lines[:5]), f"导出文件缺少 $PATH 头：{path.name}")
        point_lines = [ln for ln in lines if ln.strip().upper().startswith("$PATH")]
        _gate(bool(point_lines), f"导出文件缺少路径点：{path.name}")
        files_info.append(
            {
                "file": str(path),
                "points": len(point_lines),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            }
        )
    _gate(len(point_lines) <= 5000, "路径点超过 5000，机手端无法导入，请抽稀后重试")

    ctx.state.safe_update(export_done=True)
    from workflow import WorkflowUpdater

    WorkflowUpdater.advance(ctx.state, "EXPORTED")
    ctx.save("export")
    return {"out_dir": str(out_dir), "files": files_info}


# ---------------------------------------------------------------------------
# run-batch：一键全链（等价 smoke_full_pipeline，但每阶段落盘缓存）
# ---------------------------------------------------------------------------

def _load_display_rgb(ctx: Ctx):
    """读显示网格 RGB（workers.py:531-537 界面预览同款调用口径）。

    out_shape=(h, w) 二元组、resampling=average、config=Config().section("raster_preprocessing")。
    """
    import rasterio
    from rasterio.enums import Resampling

    from config import Config
    from raster_preprocessing import read_rgb_raster

    with rasterio.open(ctx.tif) as dataset:
        rgb, _metadata = read_rgb_raster(
            dataset,
            out_shape=(ctx.disp_h, ctx.disp_w),
            resampling=Resampling.average,
            config=Config().section("raster_preprocessing"),
        )
    return rgb


def _path_result_from_state(ctx: Ctx) -> dict:
    """从缓存的 AutoPathSegment 列表重建界面 _normalise_path_result 同款归一化结果。

    段对象真身：state.AutoPathSegment(points=[PathPoint(pixel_x, pixel_y, ...)], status, ...)；
    status==1 为作业段（tracks，与 AnimationEngine._gen_report 同口径），其余为转弯/衔接。
    """
    segments = list(
        getattr(ctx.state, "auto_path_segments", None)
        or getattr(ctx.state, "auto_path", None)
        or []
    )
    tracks, turns, full_path = [], [], []
    for seg in segments:
        pts = [
            [float(p.pixel_x), float(p.pixel_y)]
            for p in (getattr(seg, "points", None) or [])
            if hasattr(p, "pixel_x") and hasattr(p, "pixel_y")
        ]
        if len(pts) < 2:
            continue
        (tracks if int(getattr(seg, "status", 1)) == 1 else turns).append({"points": pts})
        full_path.extend(pts)
    return {
        "full_path": full_path,
        "tracks": tracks,
        "turns": turns,
        "validation": getattr(ctx.state, "path_validation", {}) or {},
        "provenance": getattr(ctx.state, "path_provenance", {}) or {},
        "description": getattr(ctx.state, "auto_path_desc", "") or "",
        "runtime": getattr(ctx.state, "path_runtime", "") or "",
    }


def _export_risk_gate(ctx: Ctx, force: bool) -> tuple[list, list]:
    """镜像界面 _geo_export_payload 的风险确认链（MW:7410-7465；含转向评估/参数漂移检查）。

    返回 (风险清单, geo_points)；风险非空且未 --force 时以门禁拦截（exit 2）。
    界面的「导出确认」弹窗在命令层映射为 --force 预授权。
    """
    _gate(ctx.geo is not None and ctx.geo.is_ready(), "当前影像没有可用地理坐标，不能导出经纬度文件", missing=["地理坐标"])
    # 源文件完整性（MW:7422-7425）：加载后 tif 被改动则必须重来
    from cache import source_identity

    _gate(
        source_identity(ctx.tif) == dict(getattr(ctx.state, "source_identity", None) or {}),
        "GeoTIFF 文件在加载后已发生变化，请重新打开影像并重新规划",
        missing=["重新加载影像"],
    )
    segments = list(getattr(ctx.state, "auto_path", None) or [])
    _gate(
        bool(getattr(ctx.state, "auto_path_planned", False)) and bool(segments),
        "当前没有可导出的路径，请先生成路径",
        missing=["路径"],
    )
    geo_points = list(getattr(ctx.state, "auto_path_geo", None) or [])
    _gate(bool(geo_points), "缺少路径经纬度坐标，请重新运行 plan", missing=["路径坐标"])

    validation = dict(getattr(ctx.state, "path_validation", None) or {})
    artifact = dict(getattr(ctx.state, "path_artifact", None) or {})
    risks: list[str] = []
    if validation.get("validation_source") == "cache_snapshot" or not bool(validation.get("current_validation_complete", True)):
        risks.append("路径来自缓存快照，尚未使用当前输入重新验证")
    path_valid = bool(
        getattr(ctx.state, "auto_path_valid", False)
        and validation.get("valid", getattr(ctx.state, "auto_path_valid", False))
    )
    turn_assessment = dict(artifact.get("turn_assessment") or {})
    hard_reasons = list(turn_assessment.get("hard_reasons") or [])
    if bool(turn_assessment.get("kinematically_infeasible")):
        hard_reasons.append("路径包含仅供显示的不可执行直线连接")
    if not path_valid or hard_reasons:
        issues = list(validation.get("issues") or []) + hard_reasons
        detail = "；".join(str(item) for item in issues[:5]) or "路径未经验证"
        risks.append(f"路径未通过安全验证：{detail}")
    planned_harvester = (artifact.get("planning_factors") or {}).get("harvester") or {}
    if planned_harvester:
        current = _effective_harvester_params(ctx.cfg, getattr(ctx.state, "harvester_params", None))
        changed = []
        for key, value in planned_harvester.items():
            if key not in current:
                continue
            current_value = current[key]
            if value is None or current_value is None:
                if value is not current_value:
                    changed.append(key)
            elif abs(float(value) - float(current_value)) > 1e-9:
                changed.append(key)
        if changed:
            risks.append("当前农机参数与路径规划时不一致：" + ", ".join(changed))
    if _layout_has_recovery(artifact.get("layout")):
        risks.append("当前缓存路径包含 planning_recovered=True 的自动恢复作业带，请重新规划或逐条人工确认后再导出")
    _gate(
        not risks or force,
        "；".join(risks) + "。确认风险后加 --force 导出（等价界面「导出确认」弹窗的确认按钮）。",
        missing=["--force 风险确认"],
    )
    return risks, geo_points


def _write_export_manifest(ctx: Ctx, output_path: str, output_format: str,
                           path_result: dict, geo_points: list,
                           protocol_metadata: dict | None = None) -> str:
    """镜像界面 _write_export_manifest（MW:7722-）：导出与审计证据不可分（DECISION-001）。"""
    import cache as cache_mod
    from importlib import metadata

    from build_info import SCIENTIFIC_CODE_PATHS, load_build_info
    from provenance import APP_VERSION, code_identity, file_sha256, git_is_dirty, git_revision
    from runtime_paths import project_root as runtime_project_root

    from path_planner import PATH_PLANNING_VERSION

    versions = {}
    for package in ("numpy", "opencv-python", "PySide6", "pyproj",
                    "rasterio", "shapely", "ultralytics"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    root = runtime_project_root()
    manifest = {
        "schema": "align-export-manifest-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "channel": "cli",
        "format": output_format,
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "point_count": len(geo_points),
        "tif_path": ctx.tif,
        "source_identity": cache_mod.source_identity(ctx.tif),
        "app_version": APP_VERSION,
        "path_planning_version": PATH_PLANNING_VERSION,
        "build_info": load_build_info(),
        "git": {"revision": git_revision(root), "dirty": git_is_dirty(root)},
        "scientific_code": code_identity(root, SCIENTIFIC_CODE_PATHS),
        "config": _jsonable(ctx.cfg.snapshot()),
        "packages": versions,
        "validation": getattr(ctx.state, "path_validation", {}) or {},
        "harvester": {
            "params": _effective_harvester_params(ctx.cfg, getattr(ctx.state, "harvester_params", {}) or {}),
            "sources": getattr(ctx.state, "harvester_param_sources", {}) or {},
        },
        "protocol_metadata": dict(protocol_metadata or {}),
    }
    manifest_path = str(output_path) + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(manifest), fh, ensure_ascii=False, indent=2)
    return manifest_path


# ── B 组：参数与标注 ─────────────────────────────────────────────

def do_set_params(ctx: Ctx, params: str, **_kw) -> dict:
    """set-params：等价界面「设置→农机参数」（_apply_harvester_params MW:11438-11532）。

    R7 口径：参数不经确认门禁，来源记为 user_measured；与界面一致把生效值写进
    config.json；参数变化使已规划路径失效（足迹验证失效）。
    """
    from config import save_json_atomic

    try:
        values = json.loads(params)
    except json.JSONDecodeError as exc:
        raise ParamError(f"--params 不是合法 JSON：{exc}") from exc
    _gate(isinstance(values, dict) and values,
          "--params 必须是非空 JSON 对象",
          missing=['--params \'{"cutter_width_m": 2.0, "track_width_m": 0.45}\''])
    known = set(_effective_harvester_params(ctx.cfg, {}).keys())
    unknown = sorted(str(k) for k in values if k not in known)
    _gate(not unknown, f"未知农机参数键：{', '.join(unknown)}；允许：{', '.join(sorted(known))}",
          missing=([f"--params 中去掉 {unknown[0]}"] if unknown else []))
    cleaned = {}
    for key, val in values.items():
        try:
            cleaned[key] = float(val)
        except (TypeError, ValueError) as exc:
            raise ParamError(f"参数 {key} 的值不是数字：{val!r}") from exc
    merged = dict(getattr(ctx.state, "harvester_params", {}) or {})
    merged.update(cleaned)
    effective = _effective_harvester_params(ctx.cfg, merged)
    previous = _effective_harvester_params(
        ctx.cfg, getattr(ctx.state, "harvester_params", {}) or {})
    changed = any(abs(float(previous.get(k, 0.0)) - float(effective.get(k, 0.0))) > 1e-9
                  for k in set(previous) | set(effective))
    ctx.state.harvester_params = effective
    sources = dict(getattr(ctx.state, "harvester_param_sources", {}) or {})
    for key in cleaned:
        sources[key] = "user_measured"
    ctx.state.harvester_param_sources = sources
    ctx.state.harvester_params_confirmed = True
    snapshot = ctx.cfg.update_section("harvester", effective)
    save_json_atomic(ctx.cfg.config_path(), snapshot)
    invalidated = False
    if changed and bool(getattr(ctx.state, "auto_path_planned", False)):
        # MW:11517-11530：农机物理参数已变化，旧路径足迹验证已经失效
        _invalidate_from(ctx.state, "path")
        invalidated = True
    ctx.save("params")
    return {
        "changed": changed,
        "path_invalidated": invalidated,
        "params": effective,
        "overridden": sorted(cleaned.keys()),
    }


def _parse_polygon_arg(raw: str) -> list:
    """--polygon 载荷：[[x,y],...] 或 GeoJSON Feature/Geometry/Polygon。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParamError(f"--polygon 不是合法 JSON：{exc}") from exc
    if isinstance(data, dict):
        if data.get("type") == "Feature":
            data = data.get("geometry") or {}
        if isinstance(data, dict) and data.get("type") == "Polygon":
            data = (data.get("coordinates") or [[]])[0]
        else:
            raise ParamError("--polygon 的 GeoJSON 必须是 Polygon 或含 Polygon geometry 的 Feature")
    if not (isinstance(data, list) and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in data)):
        raise ParamError("--polygon 必须是 [[x, y], ...] 顶点数组（显示坐标）")
    return [[float(p[0]), float(p[1])] for p in data]


def do_set_forbidden(ctx: Ctx, polygon: str | None = None, clear: bool = False, **_kw) -> dict:
    """set-forbidden：等价界面禁行区确认/清除（MW:4278-4320）。

    顶点为显示坐标；必须 ≥3 点且完全落在已确认田块内；--clear 清空。
    """
    import cv2
    import numpy as np

    regions = [list(map(list, r)) for r in (getattr(ctx.state, "forbidden_regions", None) or [])]
    had_path = bool(getattr(ctx.state, "auto_path_planned", False))
    if clear:
        _gate(bool(regions), "当前没有禁行区。", missing=["禁行区"])
        ctx.state.safe_update(forbidden_regions=[], forbidden_regions_confirmed=False)
        if had_path:
            _invalidate_from(ctx.state, "path")
        ctx.save("forbidden")
        return {"cleared": len(regions), "regions": 0, "path_invalidated": had_path}
    _gate(polygon, "缺少 --polygon（或改用 --clear 清空禁行区）。", missing=["--polygon"])
    pts = _parse_polygon_arg(polygon)
    _gate(len(pts) >= 3, "禁行区至少需要 3 个顶点。", missing=["补全禁行区顶点"])
    field = list(getattr(ctx.state, "field_boundary", None) or [])
    _gate(len(field) >= 3, "请先圈选田块。", missing=["田块边界"])
    field_arr = np.asarray(field, dtype=np.float32)
    outside = [p for p in pts if cv2.pointPolygonTest(field_arr, (float(p[0]), float(p[1])), False) < 0]
    _gate(not outside, "禁行区必须完全位于已确认的田块边界内。", missing=["调整禁行区顶点"])
    regions.append([[float(x), float(y)] for x, y in pts])
    ctx.state.safe_update(forbidden_regions=regions, forbidden_regions_confirmed=True)
    if had_path:
        _invalidate_from(ctx.state, "path")
    ctx.save("forbidden")
    return {"cleared": 0, "regions": len(regions), "added_points": len(pts),
            "path_invalidated": had_path}


def _load_json_file(path: str, what: str) -> dict:
    _gate(os.path.isfile(path), f"{what}文件不存在：{path}", missing=[what])
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ParamError(f"{what}文件读取失败：{exc}") from exc


def do_import_reference(ctx: Ctx, json_path: str, **_kw) -> dict:
    """import-reference：导入参考线 sidecar（CA:139-160 save_reference_lines）。

    参考线是评价对照锚点，不参与计算链，导入不触发失效。
    """
    import cache as cache_mod

    payload = _load_json_file(json_path, "参考线 JSON")
    _gate(isinstance(payload, dict) and isinstance(payload.get("lines"), list),
          "参考线 JSON 必须是含 lines 数组的对象。", missing=['{"lines": [...]}'])
    out = cache_mod.save_reference_lines(ctx.tif, payload)
    return {"lines": len(payload["lines"]), "sidecar": out}


def do_import_operator(ctx: Ctx, json_path: str, **_kw) -> dict:
    """import-operator：导入人工标定线 sidecar（OP:33 SCHEMA；CA:195-215）。

    标定线作为规划覆盖输入（runtime.operator_centerlines），导入后使已规划路径失效。
    """
    import operator_paths

    payload = _load_json_file(json_path, "人工标定线 JSON")
    _gate(isinstance(payload, dict), "人工标定线 JSON 必须是对象。", missing=["标定线 JSON"])
    if payload.get("schema") != operator_paths.SCHEMA:
        if isinstance(payload.get("lines"), list):
            wrapped = operator_paths.empty_payload(ctx.tif, {})
            wrapped["lines"] = payload["lines"]
            payload = wrapped
        else:
            raise ParamError(
                f"人工标定线 JSON 的 schema 必须是 {operator_paths.SCHEMA}，"
                "或提供含 lines 数组的对象。")
    import cache as cache_mod

    out = cache_mod.save_operator_paths(ctx.tif, payload)
    had_path = bool(getattr(ctx.state, "auto_path_planned", False))
    if had_path:
        _invalidate_from(ctx.state, "path")
    ctx.save("operator")
    return {"lines": len(payload.get("lines") or []), "sidecar": out,
            "path_invalidated": had_path}


# ── C 组：导出与模拟 ─────────────────────────────────────────────

def _build_cli_snapshot_scene(ctx: Ctx):
    """按界面 _build_snapshot_scene（MW:8791-9093）默认口径构建快照图层（全图层可见、无标签）。

    图层顺序与颜色与界面一致；界面的图层开关在 CLI 侧固定为默认可见集。
    注意：pyside6_app.snapshot_export 会拉入 PySide6（仅启动开销，渲染本身纯函数）。
    """
    import numpy as np

    from pyside6_app.snapshot_export import SnapshotScene

    state = ctx.state
    scene = SnapshotScene()
    try:
        raw = _attach_raw_mask(ctx)
        scene.mask(raw, getattr(state, "mask_raw_offset_x", 0),
                   getattr(state, "mask_raw_offset_y", 0), (100, 210, 255))
    except Exception:
        pass
    if getattr(state, "mask_processed", False):
        _attach_processed_mask(ctx)
        mr = state.mask_result or {}
        headland = mr.get("headland_mask")
        if isinstance(headland, np.ndarray):
            scene.mask(headland, getattr(state, "mask_offset_x", 0),
                       getattr(state, "mask_offset_y", 0), (210, 230, 120))
        uncertain = mr.get("uncertain_mask")
        if isinstance(uncertain, np.ndarray):
            scene.mask(uncertain, getattr(state, "mask_offset_x", 0),
                       getattr(state, "mask_offset_y", 0), (170, 170, 190))
        processed = mr.get("processed_mask")
        if isinstance(processed, np.ndarray):
            scene.mask(processed, getattr(state, "mask_offset_x", 0),
                       getattr(state, "mask_offset_y", 0), (120, 255, 140))
    # 一条线原则：路线已存在时不再叠加中心线
    has_route = bool(getattr(state, "auto_path_planned", False))
    if not has_route:
        catalog = getattr(state, "centerline_catalog", None) or {}
        items = list(catalog.get("items") or [])
        selected = set(getattr(state, "selected_centerline_ids", None) or [])
        for item in items:
            poly = item.get("polyline") or []
            if item.get("id") in selected and len(poly) >= 2:
                pts = [(float(p[0]), float(p[1])) for p in poly]
                scene.polyline(pts, (40, 220, 80), width=5)
                scene.circle(pts[0][0], pts[0][1], 4, (120, 255, 150))
                scene.circle(pts[-1][0], pts[-1][1], 4, (120, 255, 150))
    # 参考线默认在界面为隐藏图层，CLI 默认同样不叠加
    field = [(float(p[0]), float(p[1])) for p in (getattr(state, "field_boundary", None) or [])]
    if len(field) >= 3:
        scene.polyline(field + [field[0]], (255, 140, 50), width=4)
        for x, y in field:
            scene.circle(x, y, 5, (255, 255, 0))  # MW:4065 _field_vertex_radius=5
    for idx, region in enumerate(getattr(state, "forbidden_regions", None) or [], start=1):
        pts = [(float(p[0]), float(p[1])) for p in region]
        if len(pts) >= 3:
            scene.polygon(pts, (35, 35, 220), outline=(20, 20, 255), width=4)
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            scene.label(cx, cy, f"禁行{idx}", color=(20, 20, 255))
    # 服务点（入口/出口颜色跟随路径边状态，与 _route_edit_control_point_color 同口径）
    path_status = list(getattr(state, "path_status", []) or [])
    work_color, turn_color = (0, 255, 70), (70, 120, 255)
    entry = getattr(state, "entry_point", None)
    if entry:
        color = work_color if (path_status and int(path_status[0]) == 1) else turn_color
        scene.circle(float(entry[0]), float(entry[1]), 6, color)
    exit_pt = getattr(state, "exit_point", None)
    if exit_pt:
        color = work_color if (path_status and int(path_status[-1]) == 1) else turn_color
        scene.circle(float(exit_pt[0]), float(exit_pt[1]), 6, color)
    for point in getattr(state, "unload_points", None) or []:
        scene.circle(float(point[0]), float(point[1]), 8, (255, 210, 40))
    # 规划路径：作业绿 / 转弯蓝 / 界外红（MW:9019-9093 同口径）
    if has_route:
        field_boundary = list(getattr(state, "field_boundary", None) or [])
        path_result = _path_result_from_state(ctx)
        for track in path_result["tracks"]:
            if track["points"]:
                scene.polyline([tuple(p) for p in track["points"]], work_color, width=4)
        for turn in path_result["turns"]:
            pts = turn["points"]
            if not pts:
                continue
            if field_boundary:
                inside, outside = _split_polyline_by_field(pts, field_boundary)
                for sub in inside:
                    scene.polyline([tuple(p) for p in sub], turn_color, width=3)
                for sub in outside:
                    scene.polyline([tuple(p) for p in sub], (40, 40, 255), width=3)
            else:
                scene.polyline([tuple(p) for p in pts], turn_color, width=3)
        full_path = path_result["full_path"]
        if exit_pt is None and len(full_path) >= 2:
            scene.circle(float(full_path[-1][0]), float(full_path[-1][1]), 9, (255, 80, 80))
    return scene


def _split_polyline_by_field(points, boundary):
    """镜像界面 _split_polyline_by_field_boundary：按田块边界把折线拆成界内/界外。"""
    import cv2
    import numpy as np

    if len(points) < 2 or len(boundary) < 3:
        return ([[list(p) for p in points]] if len(points) >= 2 else []), []
    field_arr = np.asarray(boundary, dtype=np.float32)

    def _inside(x, y):
        return cv2.pointPolygonTest(field_arr, (float(x), float(y)), False) >= 0

    inside_parts, outside_parts = [], []
    current = [list(map(float, points[0]))]
    current_inside = _inside(*points[0])
    for pt in points[1:]:
        pt = [float(pt[0]), float(pt[1])]
        pt_inside = _inside(*pt)
        if pt_inside != current_inside:
            (inside_parts if current_inside else outside_parts).append(current + [pt])
            current = [pt]
            current_inside = pt_inside
        else:
            current.append(pt)
    (inside_parts if current_inside else outside_parts).append(current)
    return inside_parts, outside_parts


def do_export_snapshot(ctx: Ctx, out: str, **_kw) -> dict:
    """export-snapshot：等价界面「导出全分辨率快照」（MW:8731-8777 + snapshot_export.render）。"""
    from provenance import file_sha256
    from pyside6_app.snapshot_export import render_snapshot_png

    scene = _build_cli_snapshot_scene(ctx)
    out_path = str(Path(out).expanduser().resolve())
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    info = render_snapshot_png(
        ctx.tif, scene, out_path,
        display_shape=(ctx.disp_h, ctx.disp_w), view_scale=1.0,
    )
    return {"png": info.get("path", out_path), "shape": info.get("source_shape"),
            "vectors": info.get("vector_count"), "masks": info.get("mask_count"),
            "sha256": file_sha256(info.get("path", out_path))}


def _diagnostic_metadata(ctx: Ctx, run_label: str = "路径规划", stage: str = "export") -> dict:
    """镜像界面 _diagnostic_metadata（MW:11177-11287）：源/模型/坐标溯源证据。"""
    import cache as cache_mod
    from provenance import file_sha256
    from runtime_paths import builtin_model_dir

    state = ctx.state
    source_path = str(ctx.tif or getattr(state, "tif_path", "") or "")
    source_width, source_height = int(ctx.full_w or 0), int(ctx.full_h or 0)
    display_width, display_height = int(ctx.disp_w or 0), int(ctx.disp_h or 0)
    scale_x = float(getattr(state, "display_scale_x", 0.0) or 0.0)
    scale_y = float(getattr(state, "display_scale_y", 0.0) or 0.0)
    if scale_x <= 0 and source_width and display_width:
        scale_x = float(source_width) / float(display_width)
    if scale_y <= 0 and source_height and display_height:
        scale_y = float(source_height) / float(display_height)
    scale_x = scale_x if scale_x > 0 else 1.0
    scale_y = scale_y if scale_y > 0 else 1.0

    boundary_display = [
        [float(p[0]), float(p[1])]
        for p in (getattr(state, "field_boundary", []) or [])
        if len(p) >= 2
    ]
    boundary_source = [[p[0] * scale_x, p[1] * scale_y] for p in boundary_display]
    boundary_lonlat = []
    geo = ctx.geo
    if geo is not None and geo.is_ready():
        for point in boundary_source:
            try:
                lon, lat = geo.pixel_to_lonlat(point[0], point[1])
                boundary_lonlat.append([float(lon), float(lat)])
            except Exception:
                boundary_lonlat = []
                break

    model_path = str(getattr(state, "current_model_path", "") or "")
    model_sha256 = str(getattr(state, "model_sha256", "") or "")
    if not model_sha256 and model_path and os.path.isfile(model_path):
        model_sha256 = file_sha256(model_path)
    registry = {}
    try:
        registry_path = os.path.join(str(builtin_model_dir()), "model_registry.json")
        with open(registry_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        entries = payload.get("models", payload) if isinstance(payload, dict) else {}
        if isinstance(entries, dict):
            registry = dict(entries.get(os.path.basename(model_path), {}) or {})
    except (OSError, ValueError, AttributeError):
        registry = {}
    inference_record = dict(getattr(state, "inference_provenance", {}) or {})
    inference_inputs = dict(inference_record.get("inputs") or {})
    try:
        source_identity_value = cache_mod.source_identity(source_path) if source_path else {}
    except Exception:
        source_identity_value = {}
    crop_window = inference_inputs.get("source_crop_window") or []
    metadata = {
        "run_label": str(run_label or ""),
        "stage": str(stage),
        "source_image": {
            "path": os.path.abspath(source_path) if source_path else "",
            "sha256": str(getattr(state, "source_sha256", "") or ""),
            "identity": source_identity_value,
            "source_raster_shape": [source_height, source_width],
            "display_shape": [display_height, display_width],
            "display_to_source_scale": {"x": scale_x, "y": scale_y},
            "metadata": _jsonable(getattr(state, "source_metadata", {}) or {}),
        },
        "model": {
            "name": str(getattr(state, "current_model_name", "") or ""),
            "path": os.path.abspath(model_path) if model_path else "",
            "sha256": model_sha256,
            "registry": _jsonable(registry),
        },
        "image_position": {
            "source_crop_window_xywh": _jsonable(crop_window),
            "mask_offset_xy_display": [
                int(getattr(state, "mask_offset_x", 0) or 0),
                int(getattr(state, "mask_offset_y", 0) or 0),
            ],
            "field_boundary_display": boundary_display,
            "field_boundary_source": boundary_source,
            "field_boundary_lonlat": boundary_lonlat,
        },
        "inference": {
            "algorithm_version": str(inference_record.get("algorithm_version", "") or ""),
            "config": _jsonable(
                inference_inputs.get("inference_config")
                or (ctx.cfg.section("model") if getattr(ctx, "cfg", None) is not None else {})
            ),
            "fingerprint": str(inference_record.get("fingerprint", "") or ""),
            "artifact_sha256": str(inference_record.get("artifact_sha256", "") or ""),
        },
        "preprocessing": _jsonable(getattr(state, "raster_preprocessing", {}) or {}),
        "runtime": _jsonable(getattr(state, "inference_runtime", {}) or {}),
        "stage_provenance": {
            "inference": _jsonable(inference_record),
            "mask": _jsonable(getattr(state, "mask_provenance", {}) or {}),
        },
    }
    if geo is not None and geo.is_ready() and source_width > 0 and source_height > 0:
        try:
            metadata["source_image"]["geographic_extent"] = _jsonable(
                geo.validate_raster_extent(source_width, source_height)
            )
        except Exception:
            pass
    return metadata


def do_export_package(ctx: Ctx, out: str, force: bool = False, **_kw) -> dict:
    """export-package：等价界面「导出完整试验资料包」（MW:10915-11130 三段式）。"""
    risks, geo_points = _export_risk_gate(ctx, force)
    from path_planner import export_path_format
    from provenance import file_sha256
    import trial_package

    rgb = _load_display_rgb(ctx)
    raw = _attach_raw_mask(ctx)
    _attach_processed_mask(ctx)
    base = Path(out).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    final_dir = base / f"align_trial_package_{stamp}"
    suffix = 1
    while final_dir.exists():
        suffix += 1
        final_dir = base / f"align_trial_package_{stamp}_{suffix}"
    staging = Path(tempfile.mkdtemp(prefix=".align_package_build_", dir=str(base)))
    manifest = None
    try:
        trial_package.export_visual_artifacts(
            str(staging),
            source_image=rgb,
            raw_mask=raw,
            mask_result=getattr(ctx.state, "mask_result", None) or {},
            path_result=_path_result_from_state(ctx),
            offset=(getattr(ctx.state, "mask_offset_x", 0), getattr(ctx.state, "mask_offset_y", 0)),
            field_boundary=list(getattr(ctx.state, "field_boundary", None) or []),
            forbidden_regions=list(getattr(ctx.state, "forbidden_regions", None) or []),
            service_points={
                "entry": getattr(ctx.state, "entry_point", None),
                "exit": getattr(ctx.state, "exit_point", None),
                "unload": list(getattr(ctx.state, "unload_points", None) or []),
            },
            geo=ctx.geo,
            mask_config=ctx.cfg.section("mask_processing"),
            inference_metadata=_diagnostic_metadata(ctx),
        )
        os.replace(staging, final_dir)
        # 第三段：任务路径 + 审计清单 + 包索引
        route_path = str(final_dir / "mission_field_trial.txt")
        export_path_format(geo_points, route_path)
        manifest = _write_export_manifest(
            ctx, route_path, "$PATH", _path_result_from_state(ctx), geo_points)
        trial_package.write_package_index(str(final_dir))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    from workflow import WorkflowUpdater

    WorkflowUpdater.advance(ctx.state, "EXPORTED")
    ctx.state.safe_update(export_done=True)
    ctx.save("export")
    return {
        "package_dir": str(final_dir),
        "route": route_path,
        "route_sha256": file_sha256(route_path),
        "manifest": manifest,
        "points": len(geo_points),
        "warnings": risks,
    }


def do_simulate(ctx: Ctx, frames: str | None = None, frame_stride: int = 30,
                max_frames: int = 200000, **_kw) -> dict:
    """simulate：等价界面「模拟演示」（MW:10440-10530；AnimationEngine WK_flow:87-619）。

    离线确定性复跑：注入固定步长时钟（dt=1/30s，对齐界面 33ms 定时器），
    逐帧取 get_interpolated_position() 位姿；--frames 输出离屏帧图与 frames.jsonl。
    """
    _gate(bool(getattr(ctx.state, "auto_path_planned", False)),
          "请先生成路径后再开始模拟演示。", missing=["路径规划"])
    segments = list(getattr(ctx.state, "auto_path_segments", None) or [])
    _gate(bool(segments),
          "路径点不足：当前路径点不足，无法进行模拟演示，请重新执行路径规划。",
          missing=["路径点"])
    import workflow
    from workflow import AnimationEngine

    engine = AnimationEngine(ctx.state, ctx.geo)
    engine.setup(segments)
    _gate(bool(getattr(ctx.state, "anim_total_frames", 0)),
          "路径点不足：当前路径点不足，无法进行模拟演示，请重新执行路径规划。",
          missing=["路径点"])

    class _FixedClock:
        """确定性帧时钟：每次 time() 前进固定 dt，使离屏轨迹与界面逐帧一致。"""

        def __init__(self, dt):
            self._t = 10_000.0
            self._dt = dt

        def time(self):
            self._t += self._dt
            return self._t

    real_time = workflow.time
    workflow.time = _FixedClock(1.0 / 30.0)
    renderer = None
    frames_rows = []
    finished = False
    try:
        if frames:
            renderer = _SimFrameRenderer(ctx, frames, frame_stride)
        for frame_i in range(max_frames):
            alive = engine.advance()
            pose = engine.get_interpolated_position()
            idx = int(getattr(ctx.state, "anim_frame", 0))
            speed = engine._current_speed_mps(idx)
            types = list(getattr(ctx.state, "anim_all_types", None) or ["work"])
            frames_rows.append({
                "frame": idx,
                "x": round(float(pose[0][0]), 4),
                "y": round(float(pose[0][1]), 4),
                "heading_deg": round(float(pose[1]), 4),
                "speed_mps": round(float(speed), 4),
                "segment_type": types[min(idx, len(types) - 1)] if types else "work",
                "progress": round(engine.progress_ratio(), 6),
            })
            if renderer is not None:
                renderer.draw(ctx.state, pose, frame_i)
            # 与界面 _anim_frame_tick（MW:10518-10536）同口径：引擎内部完成分支
            # 实际上不可达（anim_frame 最大 = total-1），由外层 tick 判完成。
            if (not alive) or idx >= int(getattr(ctx.state, "anim_total_frames", 0)) - 1:
                finished = True
                break
        if not finished:
            raise RuntimeError(f"模拟超过 {max_frames} 帧仍未结束，fail-closed")
    finally:
        workflow.time = real_time
    # 界面完成分支：显式 _gen_report + WorkflowUpdater.advance("ANIMATION_DONE")
    engine._gen_report()
    workflow.WorkflowUpdater.advance(ctx.state, "ANIMATION_DONE")
    ctx.state.safe_update(anim_active=False, anim_paused=False)
    report = str(getattr(ctx.state, "anim_report", "") or "")
    ctx.state.safe_update(simulation_done=True, anim_active=False)
    ctx.save("simulate")
    out = {
        "frames": len(frames_rows),
        "path_frames": int(getattr(ctx.state, "anim_total_frames", 0)),
        "total_m": round(float((engine._cum_dists[-1] if getattr(engine, "_cum_dists", None) else 0.0)), 3),
        "report": report,
    }
    if renderer is not None:
        out["frames_dir"] = renderer.finish(frames_rows, report)
    return out


class _SimFrameRenderer:
    """模拟演示的离屏帧渲染（CLI 便利化产物：显示网格底图 + 规划路径 + 农机位姿）。"""

    def __init__(self, ctx: Ctx, out_dir: str, stride: int):
        import cv2
        import numpy as np

        self._cv2 = cv2
        self.dir = str(Path(out_dir).expanduser().resolve())
        os.makedirs(self.dir, exist_ok=True)
        self.stride = max(1, int(stride))
        self.count = 0
        rgb = _load_display_rgb(ctx)
        self.base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
        path_result = _path_result_from_state(ctx)
        for track in path_result["tracks"]:
            pts = np.asarray(track["points"], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(self.base, [pts], False, (0, 255, 70), 1, cv2.LINE_AA)
        for turn in path_result["turns"]:
            pts = np.asarray(turn["points"], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(self.base, [pts], False, (70, 120, 255), 1, cv2.LINE_AA)

    def draw(self, state, pose, frame_idx: int) -> None:
        if frame_idx % self.stride != 0:
            return
        cv2 = self._cv2
        canvas = self.base.copy()
        x, y = float(pose[0][0]), float(pose[0][1])
        heading = float(pose[1])
        tip = (int(round(x + 18 * math.cos(math.radians(heading)))),
               int(round(y + 18 * math.sin(math.radians(heading)))))
        cv2.arrowedLine(canvas, (int(round(x)), int(round(y))), tip, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 6, (0, 165, 255), -1, cv2.LINE_AA)
        self.count += 1
        cv2.imwrite(os.path.join(self.dir, f"frame_{self.count:06d}.png"), canvas)

    def finish(self, rows: list, report: str) -> str:
        with open(os.path.join(self.dir, "frames.jsonl"), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(self.dir, "simulation_report.txt"), "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        return self.dir


# ── D 组：只读与运维 ─────────────────────────────────────────────

def do_query_status(ctx: Ctx, **_kw) -> dict:
    """query-status：复用界面任务状态体系（pyside6_app/task_state.py，Qt-free 直接导入）。

    derive_task_statuses 以 state.mask_raw / state.mask_result 判定识别与掩膜是否完成，
    与界面启动恢复同口径：这里先把缓存掩膜挂回 state 再推导。
    """
    from pyside6_app.task_state import TASK_LABELS, TASK_ORDER, derive_task_statuses

    if getattr(ctx.state, "inference_done", False) and getattr(ctx.state, "mask_raw", None) is None:
        _attach_raw_mask(ctx)
    if getattr(ctx.state, "mask_processed", False):
        _attach_processed_mask(ctx)
    statuses = derive_task_statuses(ctx.state, bool(getattr(ctx.state, "tif_path", "")))
    tasks = []
    for tid in TASK_ORDER:
        st = statuses[tid]
        tasks.append({
            "id": tid,
            "label": TASK_LABELS[tid],
            "status": st.status,
            "text": st.text,
            "action": st.action,
            "optional": bool(st.optional),
            "blocker": st.blocker,
        })
    pending = [t for t in tasks if not t["optional"] and t["status"] not in ("done", "disabled")]
    if pending:
        head = pending[0]
        next_action = head["action"] or f"{head['label']}（{head['id']}）"
    else:
        next_action = "全流程就绪"
    return {
        "workflow_step": getattr(ctx.state, "workflow_step", "IDLE"),
        "tasks": tasks,
        "pending_required": [t["id"] for t in pending],
        "next_action": next_action,
        "cache_dir": _cache_dir_of(ctx.tif),
    }


def _cache_dir_of(tif: str) -> str:
    import cache as cache_mod

    return cache_mod.project_cache_dir(tif)


def do_query_diagnostics(ctx: Ctx, export_zip: str | None = None, **_kw) -> dict:
    """query-diagnostics：汇总识别/掩膜/路径诊断（均可追溯到底层产物）。"""
    import numpy as np

    mask_result = getattr(ctx.state, "mask_result", None) or {}
    mask_metrics = {}
    for key, val in mask_result.items():
        if isinstance(val, np.ndarray):
            from provenance import artifact_sha256

            mask_metrics[key] = {"shape": [int(v) for v in val.shape],
                                 "nonzero": int(np.count_nonzero(val)),
                                 "sha256": artifact_sha256(val)}
        elif key in ("config", "scale_gate"):
            continue
        else:
            mask_metrics[key] = val
    diag = {
        "source": {
            "tif": ctx.tif,
            "sha256": getattr(ctx.state, "source_sha256", "") or "",
            "crs": ctx.crs,
            "full_size": [ctx.full_w, ctx.full_h],
            "display_size": [ctx.disp_w, ctx.disp_h],
        },
        "inference": {
            "done": bool(getattr(ctx.state, "inference_done", False)),
            "model": getattr(ctx.state, "current_model_path", "") or "",
            "model_sha256": getattr(ctx.state, "model_sha256", "") or "",
            "provenance": getattr(ctx.state, "inference_provenance", {}) or {},
            "runtime": getattr(ctx.state, "inference_runtime", "") or "",
        },
        "mask": {
            "processed": bool(getattr(ctx.state, "mask_processed", False)),
            "provenance": getattr(ctx.state, "mask_provenance", {}) or {},
            "offsets": [getattr(ctx.state, "mask_offset_x", 0),
                        getattr(ctx.state, "mask_offset_y", 0)],
            "metrics": mask_metrics,
        },
        "path": {
            "planned": bool(getattr(ctx.state, "auto_path_planned", False)),
            "valid": bool(getattr(ctx.state, "auto_path_valid", False)),
            "description": getattr(ctx.state, "auto_path_desc", "") or "",
            "validation": getattr(ctx.state, "path_validation", {}) or {},
            "provenance": getattr(ctx.state, "path_provenance", {}) or {},
            "runtime": getattr(ctx.state, "path_runtime", "") or "",
        },
        "harvester": {
            "params": _effective_harvester_params(
                ctx.cfg, getattr(ctx.state, "harvester_params", {}) or {}),
            "sources": getattr(ctx.state, "harvester_param_sources", {}) or {},
            "confirmed": bool(getattr(ctx.state, "harvester_params_confirmed", False)),
        },
        "workflow_step": getattr(ctx.state, "workflow_step", "IDLE"),
    }
    if export_zip:
        import zipfile

        cache_dir = _cache_dir_of(ctx.tif)
        out = str(Path(export_zip).expanduser().resolve())
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        names = [
            "project_state.json", "meta.json", "operator_paths.json",
            "reference_lines.json", "inference_report.json", "diagnostics.json",
        ]
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in names:
                path = os.path.join(cache_dir, name)
                if os.path.isfile(path):
                    zf.write(path, name)
            zf.writestr("cli_diagnostics.json",
                        json.dumps(_jsonable(diag), ensure_ascii=False, indent=2))
        diag["export_zip"] = out
    return diag


def do_query_cache(**_kw) -> dict:
    """query-cache：盘点缓存根下全部项目缓存（全局视图，不需要 --tif）。"""
    from cache import CACHE_ROOT

    rows = []
    for state_path in sorted(Path(CACHE_ROOT).glob("*/project_state.json")):
        entry = {"cache_dir": str(state_path.parent)}
        try:
            with open(state_path, encoding="utf-8") as fh:
                state = json.load(fh)
            entry["tif_path"] = state.get("tif_path", "")
            entry["workflow_step"] = state.get("workflow_step", "")
            entry["path_planned"] = bool(state.get("auto_path_planned"))
        except (OSError, json.JSONDecodeError):
            entry["tif_path"] = ""
            entry["workflow_step"] = "<损坏>"
        try:
            st = state_path.stat()
            entry["state_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            size = sum(f.stat().st_size for f in state_path.parent.rglob("*") if f.is_file())
            entry["size_mb"] = round(size / (1024 * 1024), 2)
        except OSError:
            pass
        rows.append(entry)
    rows.sort(key=lambda r: r.get("state_updated", ""), reverse=True)
    return {"cache_root": str(CACHE_ROOT), "projects": rows, "count": len(rows)}


def do_restore(ctx: Ctx, **_kw) -> dict:
    """restore：从缓存重建现场并报告（等价界面启动恢复；只读不落盘）。"""
    state = ctx.state
    catalog = getattr(state, "centerline_catalog", None) or {}
    return {
        "workflow_step": getattr(state, "workflow_step", "IDLE"),
        "tif": getattr(state, "tif_path", "") or ctx.tif,
        "display_size": [ctx.disp_w, ctx.disp_h],
        "field_points": len(getattr(state, "field_boundary", None) or []),
        "forbidden_regions": len(getattr(state, "forbidden_regions", None) or []),
        "inference_done": bool(getattr(state, "inference_done", False)),
        "mask_processed": bool(getattr(state, "mask_processed", False)),
        "centerlines": len(catalog.get("items") or []),
        "selected_worklines": len(getattr(state, "selected_centerline_ids", None) or []),
        "entry_point": getattr(state, "entry_point", None),
        "exit_point": getattr(state, "exit_point", None),
        "unload_points": len(getattr(state, "unload_points", None) or []),
        "path_planned": bool(getattr(state, "auto_path_planned", False)),
        "path_valid": bool(getattr(state, "auto_path_valid", False)),
        "path_points": len(getattr(state, "auto_path", None) or []),
        "simulation_done": bool(getattr(state, "simulation_done", False)),
        "export_done": bool(getattr(state, "export_done", False)),
        "cache_dir": _cache_dir_of(ctx.tif),
    }


def do_doctor(**_kw) -> dict:
    """doctor：CLI 环境自检（解释器 / 依赖 / 设备 / 配置 / 缓存可写）。"""
    checks = {}
    checks["python"] = {"version": sys.version.split()[0], "ok": sys.version_info >= (3, 10)}
    for mod in ("numpy", "cv2", "rasterio", "pyproj", "ultralytics"):
        try:
            m = importlib.import_module(mod)
            checks[mod] = {"version": getattr(m, "__version__", "?"), "ok": True}
        except Exception as exc:
            checks[mod] = {"ok": False, "error": str(exc)}
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        checks["torch"] = {
            "version": torch.__version__, "ok": True, "cuda": cuda,
            "device": (torch.cuda.get_device_name(0) if cuda else "cpu"),
        }
    except Exception as exc:
        checks["torch"] = {"ok": False, "error": str(exc)}
    try:
        from config import Config

        cfg = Config()
        model_section = cfg.section("model") if hasattr(cfg, "section") else {}
        checks["config"] = {"ok": True, "model": dict(model_section)}
    except Exception as exc:
        checks["config"] = {"ok": False, "error": str(exc)}
    try:
        from cache import CACHE_ROOT

        cache_root = Path(CACHE_ROOT)
        cache_root.mkdir(parents=True, exist_ok=True)
        probe = cache_root / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["cache_root"] = {"path": str(cache_root), "writable": True, "ok": True}
    except Exception as exc:
        checks["cache_root"] = {"ok": False, "error": str(exc)}
    try:
        from provenance import APP_VERSION, git_revision
        from runtime_paths import project_root

        checks["version"] = {"app": APP_VERSION, "git": git_revision(project_root()), "ok": True}
    except Exception as exc:
        checks["version"] = {"ok": False, "error": str(exc)}
    ok = all(v.get("ok") for v in checks.values())
    return {"ok": ok, "checks": checks}


# ── rollback / reset ─────────────────────────────────────────────

def _rollback_cascade(state, tid: str) -> list:
    """镜像界面 _rollback_from_task（MW:3568-3720）的状态级联（画布/计时器不适用，略）。"""
    from pyside6_app.task_state import TASK_ORDER

    idx = TASK_ORDER.index(tid)
    affected = set(TASK_ORDER[idx:])

    if "import_img" in affected:
        state.safe_update(
            tif_path="", img_h=0, img_w=0, display_h=0, display_w=0,
            downsample_factor=1, display_scale_x=1.0, display_scale_y=1.0,
            field_boundary=[], forbidden_regions=[], forbidden_regions_confirmed=False,
            mask_raw=None, inference_original_mask=None, mask_result=None,
            inference_done=False, mask_processed=False,
        )
    elif "draw_field" in affected:
        state.safe_update(
            field_boundary=[], forbidden_regions=[], forbidden_regions_confirmed=False,
            inference_done=False, mask_raw=None, inference_original_mask=None,
            mask_processed=False, mask_result=None,
        )
    elif "segment" in affected:
        state.safe_update(
            inference_done=False, mask_raw=None, inference_original_mask=None,
            mask_processed=False, mask_result=None,
        )
    elif "process" in affected:
        state.safe_update(mask_processed=False, mask_result=None)
    if any(step in affected for step in ("import_img", "draw_field", "segment", "process")):
        state.safe_update(
            centerline_catalog={}, selected_centerline_ids=[], centerline_selection_report={},
            work_band_catalog={}, selected_work_band_ids=[], work_band_selection_report={},
        )
    if "centerline_generate" in affected:
        state.safe_update(
            centerline_catalog={}, selected_centerline_ids=[], centerline_selection_report={},
            work_band_catalog={}, selected_work_band_ids=[], work_band_selection_report={},
        )
    elif "centerline_select" in affected:
        catalog = getattr(state, "centerline_catalog", None) or {}
        rollback_report = {
            "algorithm_version": "operator-centerline-selection-v1",
            "selection_origin": "rollback",
            "selected_ids": [],
            "selected_count": 0,
            "candidate_count": len(catalog.get("items") or []),
            "selection_mode": "empty",
        }
        state.safe_update(
            selected_centerline_ids=[], selected_work_band_ids=[],
            centerline_selection_report=dict(rollback_report),
            work_band_selection_report=dict(rollback_report),
        )
    if "params" in affected:
        state.safe_update(harvester_params={}, harvester_param_sources={})
    if "entry" in affected:
        state.safe_update(entry_point=None, entry_point_locked=False)
    if "exit" in affected:
        state.safe_update(exit_point=None, exit_point_locked=False)
    if "unload" in affected:
        state.safe_update(unload_points=[], unload_point=None, unload_point_locked=False)
    if "plan" in affected:
        state.safe_update(
            auto_path_planned=False, auto_path=[], auto_path_segments=[],
            auto_path_geo=[], auto_path_valid=False, auto_path_desc="",
            path_points=[], path_status=[], path_geometry_modes=[],
        )
    if "simulate" in affected:
        state.safe_update(anim_active=False, anim_paused=False, simulation_done=False)
    if "export" in affected:
        state.export_done = False
    return sorted(affected, key=TASK_ORDER.index)


def do_rollback(ctx: Ctx, step: str, **_kw) -> dict:
    """rollback：等价界面各步骤的「回退到此处」（_rollback_from_task MW:3568-3720）。"""
    from pyside6_app.task_state import TASK_ORDER

    _gate(step in TASK_ORDER,
          f"未知任务步骤：{step}；允许：{', '.join(TASK_ORDER)}",
          missing=[f"--step {TASK_ORDER[0]}"])
    affected = _rollback_cascade(ctx.state, step)
    ctx.save("rollback")
    return {"rolled_back_to": step, "cleared_stages": affected}


def do_reset(ctx: Ctx, yes: bool = False, **_kw) -> dict:
    """reset：清空本项目缓存（等价界面「从零开始」）。破坏性操作，必须 --yes。"""
    _gate(bool(yes),
          "reset 将清空本项目全部缓存（识别/掩膜/路径/标定 sidecar 一并删除），"
          "破坏性操作必须显式 --yes 确认。",
          missing=["--yes"])
    cache_dir = _cache_dir_of(ctx.tif)
    existed = os.path.isdir(cache_dir)
    if existed:
        # 本进程自己的 cli.lock 就躺在缓存目录里且句柄开着——先释放再删（Windows 句柄占用）。
        lock = getattr(ctx, "_lock", None)
        if lock is not None:
            lock.release()
        shutil.rmtree(cache_dir)
    return {"deleted": existed, "cache_dir": cache_dir}


def run_batch(ctx: Ctx, args) -> dict:

    out_dir = Path(args.out).expanduser().resolve() if args.out else (
        ROOT / "outputs" / "cli" / time.strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_after = args.stop_after
    _gate(not stop_after or stop_after in BATCH_STAGES,
          f"--stop-after 只接受：{', '.join(BATCH_STAGES)}")

    stage_runs = []
    t_all = time.time()

    def _run_stage(name, fn, **kw):
        t0 = time.time()
        result = fn(ctx, **kw)
        record = {"stage": name, "status": "ok", "seconds": round(time.time() - t0, 1), "result": result}
        stage_runs.append(record)
        _emit({"cmd": "run-batch", "stage": name, **record})
        return result

    _run_stage("load-image", do_load_image)
    if stop_after == "load-image":
        return _batch_done(out_dir, stage_runs, t_all, stop_after)
    _run_stage("set-field", do_set_field, polygon=args.field_polygon, full_frame=bool(args.full_frame))
    if stop_after == "set-field":
        return _batch_done(out_dir, stage_runs, t_all, stop_after)
    if args.model:
        _run_stage("load-model", do_load_model, model=args.model)
    _run_stage("segment", do_segment, timeout_s=args.timeout)
    if stop_after == "segment":
        return _batch_done(out_dir, stage_runs, t_all, stop_after)
    _run_stage("process-mask", do_process_mask)
    if stop_after == "process-mask":
        return _batch_done(out_dir, stage_runs, t_all, stop_after)
    _run_stage("gen-worklines", do_gen_worklines)
    if stop_after == "gen-worklines":
        return _batch_done(out_dir, stage_runs, t_all, stop_after)
    _run_stage("select-worklines", do_select_worklines, select_all=True)
    if stop_after == "select-worklines":
        return _batch_done(out_dir, stage_runs, t_all, stop_after)
    _run_stage("set-entry", do_set_entry, auto=True)
    if stop_after == "set-entry":
        return _batch_done(out_dir, stage_runs, t_all, stop_after)
    _run_stage("plan", do_plan)
    if stop_after == "plan":
        return _batch_done(out_dir, stage_runs, t_all, stop_after)
    _run_stage("export-txt", do_export_txt, out=str(out_dir), force=bool(getattr(args, "force", False)))
    return _batch_done(out_dir, stage_runs, t_all, stop_after)


def _batch_done(out_dir: Path, stage_runs: list, t_all: float, stop_after) -> dict:
    from version import APP_VERSION
    from path_planner import PATH_PLANNING_VERSION

    manifest = {
        "cli_version": CLI_VERSION,
        "app_version": APP_VERSION,
        "path_planning_version": PATH_PLANNING_VERSION,
        "stop_after": stop_after,
        "stages": stage_runs,
        "total_seconds": round(time.time() - t_all, 1),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"manifest": str(manifest_path), "stages_run": len(stage_runs), "out_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# argparse 入口
# ---------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    def error(self, message):  # 参数错误 → 退出码 4
        _emit({"status": "args_error", "error": message})
        self.print_usage(sys.stderr)
        sys.exit(EXIT_ARGS)


def _add_tif(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tif", required=True, help="GeoTIFF 影像路径（项目缓存按它定位）")


class ProjectLock:
    """CLI 进程间缓存目录互斥：msvcrt 字节锁，进程退出自动释放（铁律 4）。

    界面不参与本锁——对界面的并发防护靠 Ctx.save 的磁盘指纹检查。
    """

    def __init__(self, tif: str):
        import cache as cache_mod

        # 与 Ctx.__init__ 同款归一化（resolve）：cache 键 = md5(路径字符串)，
        # 不统一归一化会导致锁与缓存落在不同目录（混合斜杠输入时实测复现）。
        norm_tif = str(Path(tif).expanduser().resolve())
        cache_dir = cache_mod.project_cache_dir(norm_tif)
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "cli.lock")
        self._fh = None

    def acquire(self) -> None:
        import msvcrt

        self._fh = open(self.path, "a+b")
        try:
            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise GateError(
                f"另一个 CLI 进程正在操作本项目缓存（{self.path} 被锁定）；"
                "等它结束或换用另一幅影像。",
                missing=["等待其他命令结束"],
            ) from exc

    def release(self) -> None:
        if self._fh is None:
            return
        import msvcrt

        try:
            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self._fh.close()
        self._fh = None


# 会改缓存的命令需要持锁；只读命令（query-*/restore/doctor）不持锁。
_WRITE_COMMANDS = frozenset({
    "load-image", "set-field", "load-model", "segment", "process-mask",
    "gen-worklines", "select-worklines", "set-entry", "set-exit", "set-unload",
    "set-params", "set-forbidden", "import-reference", "import-operator",
    "plan", "simulate", "export-txt", "export-snapshot", "export-package",
    "rollback", "reset", "run-batch",
})
_NO_TIF_COMMANDS = frozenset({"query-cache", "doctor"})


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="align_cli",
        description="ALIGN 命令层（阶段 1）：与界面同一引擎的命令通道。详见 docs/命令层设计方案.md",
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("load-image", help="加载影像（初始化/恢复项目缓存）")
    _add_tif(p)

    p = sub.add_parser("set-field", help="设置田块边界")
    _add_tif(p)
    p.add_argument("--polygon", help="边界点 JSON：'[[x,y],...]' 或 @文件.json（显示像素坐标）")
    p.add_argument("--full-frame", action="store_true", help="整幅影像作为田块")

    p = sub.add_parser("load-model", help="加载识别模型")
    _add_tif(p)
    p.add_argument("--model", required=True, help="模型文件路径（.pt/.engine）")

    p = sub.add_parser("segment", help="作物识别（YOLO 实例分割）")
    _add_tif(p)
    p.add_argument("--timeout", type=int, default=1800, help="识别超时秒数（默认 1800）")

    p = sub.add_parser("process-mask", help="掩膜优化（行带正则化）")
    _add_tif(p)

    p = sub.add_parser("gen-worklines", help="生成作业线（中心线目录）")
    _add_tif(p)

    p = sub.add_parser("select-worklines", help="选择作业线（确认采用集）")
    _add_tif(p)
    p.add_argument("--ids", help="编号列表，逗号分隔；缺省=全选")
    p.add_argument("--all", dest="select_all", action="store_true", help="全选")

    p = sub.add_parser("set-entry", help="设置起点")
    _add_tif(p)
    p.add_argument("--auto", action="store_true", help="自动推荐（首条作业线起点外推 1m）")
    p.add_argument("--point", help="手动指定：x,y（显示像素坐标）")

    p = sub.add_parser("set-exit", help="设置/清空终点")
    _add_tif(p)
    p.add_argument("--point", help="x,y")
    p.add_argument("--clear", action="store_true")

    p = sub.add_parser("set-unload", help="设置/清空卸粮点")
    _add_tif(p)
    p.add_argument("--points", help="'x,y;x,y'")
    p.add_argument("--clear", action="store_true")

    p = sub.add_parser("plan", help="路径规划")
    _add_tif(p)

    p = sub.add_parser("export-txt", help="机手兼容 TXT 导出（含 manifest 与回读校验）")
    _add_tif(p)
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--force", action="store_true", help="预授权风险导出（等价界面「导出确认」弹窗的确认按钮）")

    p = sub.add_parser("run-batch", help="一键全链：load-image→…→export-txt")
    _add_tif(p)
    p.add_argument("--model", help="模型路径（可选；缺省沿用缓存中已加载模型）")
    p.add_argument("--field-polygon", help="田块边界 JSON；缺省配合 --full-frame")
    p.add_argument("--full-frame", action="store_true", help="整幅影像作为田块（默认）")
    p.add_argument("--out", help="输出目录（默认 outputs/cli/<时间戳>）")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--stop-after", choices=BATCH_STAGES, help="跑到指定阶段即停（断点续跑实验用）")
    p.add_argument("--force", action="store_true", help="导出阶段预授权风险（等价逐条命令 export-txt --force）")

    # ── 阶段 2：B/C/D 组命令 ──
    p = sub.add_parser("set-params", help="设置农机参数（JSON 对象，写 config.json 并使旧路径失效）")
    _add_tif(p)
    p.add_argument("--params", required=True, help="JSON 对象，如 '{\"cutter_width_m\": 2.0}'")

    p = sub.add_parser("set-forbidden", help="添加/清空禁行区（顶点为显示坐标）")
    _add_tif(p)
    p.add_argument("--polygon", help="'[[x,y],...]' 或 GeoJSON Polygon/Feature")
    p.add_argument("--clear", action="store_true", help="清空全部禁行区")

    p = sub.add_parser("import-reference", help="导入参考线 sidecar（JSON 文件）")
    _add_tif(p)
    p.add_argument("--json", dest="json_path", required=True, help="参考线 JSON 文件路径")

    p = sub.add_parser("import-operator", help="导入人工标定线 sidecar（JSON 文件，导入后需重新规划）")
    _add_tif(p)
    p.add_argument("--json", dest="json_path", required=True, help="标定线 JSON 文件路径")

    p = sub.add_parser("simulate", help="模拟演示（离线确定性复跑；可选离屏帧图）")
    _add_tif(p)
    p.add_argument("--frames", help="离屏帧图输出目录（缺省只输出轨迹统计与报告）")
    p.add_argument("--frame-stride", type=int, default=30, help="帧图抽样间隔（默认每 30 帧一张）")
    p.add_argument("--max-frames", type=int, default=200000, help="模拟帧数上限（fail-closed）")

    p = sub.add_parser("export-snapshot", help="导出全分辨率快照 PNG（图层口径同界面）")
    _add_tif(p)
    p.add_argument("--out", required=True, help="输出 PNG 路径")

    p = sub.add_parser("export-package", help="导出完整试验资料包（含审计清单与包索引）")
    _add_tif(p)
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--force", action="store_true", help="预授权风险导出（等价界面「导出确认」弹窗）")

    p = sub.add_parser("query-status", help="任务面板状态（下一步该干什么 + 阻塞原因）")
    _add_tif(p)

    p = sub.add_parser("query-diagnostics", help="诊断汇总（识别/掩膜/路径，可导出 ZIP）")
    _add_tif(p)
    p.add_argument("--export", dest="export_zip", help="同时导出诊断 ZIP 到该路径")

    p = sub.add_parser("query-cache", help="盘点缓存根下全部项目缓存（全局视图，免 --tif）")

    p = sub.add_parser("restore", help="从缓存重建现场并报告（只读）")
    _add_tif(p)

    p = sub.add_parser("doctor", help="环境自检（解释器/依赖/设备/配置/缓存，免 --tif）")

    p = sub.add_parser("rollback", help="回退到指定任务步骤（清空该步骤及下游状态）")
    _add_tif(p)
    p.add_argument("--step", required=True, help="任务步骤 id（query-status 的 id 列）")

    p = sub.add_parser("reset", help="清空本项目缓存（破坏性，必须 --yes）")
    _add_tif(p)
    p.add_argument("--yes", action="store_true", help="确认执行破坏性清空")
    return parser


def _dispatch(cmd: str, ctx: Ctx, args) -> dict:
    table = {
        "load-image": lambda: do_load_image(ctx),
        "set-field": lambda: do_set_field(ctx, polygon=args.polygon, full_frame=args.full_frame),
        "load-model": lambda: do_load_model(ctx, model=args.model),
        "segment": lambda: do_segment(ctx, timeout_s=args.timeout),
        "process-mask": lambda: do_process_mask(ctx),
        "gen-worklines": lambda: do_gen_worklines(ctx),
        "select-worklines": lambda: do_select_worklines(ctx, ids=args.ids, select_all=args.select_all),
        "set-entry": lambda: do_set_entry(ctx, auto=args.auto, point=args.point),
        "set-exit": lambda: do_set_exit(ctx, point=args.point, clear=args.clear),
        "set-unload": lambda: do_set_unload(ctx, points=args.points, clear=args.clear),
        "plan": lambda: do_plan(ctx),
        "export-txt": lambda: do_export_txt(ctx, out=args.out, force=bool(getattr(args, "force", False))),
        "set-params": lambda: do_set_params(ctx, params=args.params),
        "set-forbidden": lambda: do_set_forbidden(ctx, polygon=args.polygon, clear=args.clear),
        "import-reference": lambda: do_import_reference(ctx, json_path=args.json_path),
        "import-operator": lambda: do_import_operator(ctx, json_path=args.json_path),
        "simulate": lambda: do_simulate(ctx, frames=args.frames,
                                        frame_stride=args.frame_stride, max_frames=args.max_frames),
        "export-snapshot": lambda: do_export_snapshot(ctx, out=args.out),
        "export-package": lambda: do_export_package(ctx, out=args.out, force=bool(getattr(args, "force", False))),
        "query-status": lambda: do_query_status(ctx),
        "query-diagnostics": lambda: do_query_diagnostics(ctx, export_zip=args.export_zip),
        "query-cache": lambda: do_query_cache(),
        "restore": lambda: do_restore(ctx),
        "doctor": lambda: do_doctor(),
        "rollback": lambda: do_rollback(ctx, step=args.step),
        "reset": lambda: do_reset(ctx, yes=bool(args.yes)),
        "run-batch": lambda: run_batch(ctx, args),
    }
    return table[cmd]()


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return EXIT_OK
    t0 = time.time()
    lock = None
    try:
        # run-batch 的 set-field 缺省整幅
        if args.cmd == "run-batch" and not args.field_polygon:
            args.full_frame = True
        if args.cmd in _NO_TIF_COMMANDS:
            ctx = None
        else:
            if args.cmd in _WRITE_COMMANDS:
                lock = ProjectLock(args.tif)
                lock.acquire()
            ctx = Ctx(args.tif, restore=True)
            ctx._lock = lock  # do_reset 需要在删目录前释放本进程锁
        result = _dispatch(args.cmd, ctx, args)
        if args.cmd != "run-batch":
            _emit({"cmd": args.cmd, "status": "ok", "seconds": round(time.time() - t0, 1), "result": result})
        return EXIT_OK
    except GateError as exc:
        _emit({
            "cmd": args.cmd, "status": "gate_blocked", "blocker": exc.blocker,
            "missing": exc.extra.get("missing", []), "seconds": round(time.time() - t0, 1),
        })
        return EXIT_GATE
    except ParamError as exc:
        _emit({"cmd": args.cmd, "status": "args_error", "error": str(exc),
               "seconds": round(time.time() - t0, 1)})
        return EXIT_ARGS
    except SystemExit:
        raise
    except Exception as exc:  # 执行失败 → 退出码 3 + traceback
        traceback.print_exc(file=sys.stderr)
        _emit({"cmd": args.cmd, "status": "error", "error": f"{type(exc).__name__}: {exc}",
               "seconds": round(time.time() - t0, 1)})
        return EXIT_FAIL
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    sys.exit(main())
