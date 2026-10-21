"""全流程冒烟测试：只给正射影像，自动跑完整条链路。

给 AI / 开发者用的**外部测试工具**，不进入 reUI 系统、不注册到界面。

    python scripts/smoke_full_pipeline.py --tif data/tif/result.tif

它调用与界面完全相同的领域函数（读栅格 -> 识别 -> 掩膜优化 -> 作业线），
只是把"人点按钮"换成脚本调用，并把田块边界取成整幅影像矩形：正式作业
才需要精细圈选，测试时全图识别即可。

任何一步失败立即停止并打印该步诊断，不做静默降级、不伪造结果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 脚本模式不能触发桌面自动启动逻辑。
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")


# ---------------------------------------------------------------- 基础设施
class SmokeRun:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.report: dict = {"stages": {}, "started": time.strftime("%F %T")}

    def log(self, stage: str, message: str, level: str = "INFO") -> None:
        print(f"[{time.strftime('%H:%M:%S')}] [{level}] {stage}: {message}",
              flush=True)

    def record(self, stage: str, payload: dict) -> None:
        self.report["stages"][stage] = payload
        self.flush()

    def fail(self, stage: str, message: str, exc: BaseException | None = None) -> int:
        self.report["failed_stage"] = stage
        self.report["failure"] = message
        self.log(stage, message, "FAIL")
        if exc is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            self.report["traceback"] = traceback.format_exc()
        self.flush()
        return 1

    def flush(self) -> None:
        (self.out_dir / "smoke_report.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def full_frame_boundary(shape) -> list[list[float]]:
    """整幅影像矩形，作为"全图识别"的田块边界。

    推理与几何代码只要求一个 >= 3 点的多边形，没有"必须人工圈选"的假设。
    """
    h, w = int(shape[0]), int(shape[1])
    if h < 2 or w < 2:
        raise ValueError(f"影像尺寸无效: {shape}")
    return [[0.0, 0.0], [float(w - 1), 0.0],
            [float(w - 1), float(h - 1)], [0.0, float(h - 1)]]


def resolve_model(explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    models_dir = ROOT / "data" / "models"
    registry_path = models_dir / "model_registry.json"
    if registry_path.is_file():
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        entries = data.get("models") or data.get("items") or []
        preferred = next((e for e in entries if e.get("preferred")), None) \
            or (entries[0] if entries else None)
        if preferred:
            # registry 用 name 字段（不是 path），与 src/inference.scan_model_dir 一致
            candidate = Path(str(preferred.get("path") or preferred.get("name") or ""))
            if candidate.name:
                return candidate if candidate.is_absolute() else models_dir / candidate
    candidates = sorted(models_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("data/models 下没有可用权重")


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="reUI 全流程冒烟测试（外部工具，不属于系统功能）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tif", required=True, help="正射影像 GeoTIFF")
    parser.add_argument("--model", default="", help="模型 .pt，留空自动选")
    parser.add_argument("--out", default="", help="输出目录")
    parser.add_argument("--skip-plan", action="store_true", help="跑到作业线就停")
    parser.add_argument("--timeout", type=int, default=1800, help="单阶段超时秒数")
    args = parser.parse_args(argv)

    tif_path = Path(args.tif).expanduser().resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).expanduser().resolve() if args.out \
        else ROOT / "outputs" / "smoke" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    run = SmokeRun(out_dir)
    run.report["tif"] = str(tif_path)
    run.report["out_dir"] = str(out_dir)

    if not tif_path.is_file():
        return run.fail("准备", f"影像不存在: {tif_path}")

    try:
        import numpy as np
        import rasterio
        from config import Config
        from geo import GeoUtils
        from inference import InferenceRunner, ModelEngine
        from pyside6_app.workers import display_affine
        from raster_preprocessing import read_rgb_raster
        from state import AppState
    except Exception as exc:
        return run.fail("准备", "导入 reUI 模块失败", exc)

    # ---- 读栅格 ------------------------------------------------------
    try:
        cfg = Config()
        raster_cfg = cfg.section("raster") or {}
        max_dim = int(raster_cfg.get("preview_max_dimension", 2600) or 2600)
        with rasterio.open(tif_path) as src:
            if not src.crs:
                return run.fail("读栅格", "影像缺少投影信息，米制结果不可用")
            full_h, full_w = int(src.height), int(src.width)
            keep = min(1.0, max_dim / float(max(full_h, full_w)))
            disp_h = max(1, int(round(full_h * keep)))
            disp_w = max(1, int(round(full_w * keep)))
            rgb, _meta = read_rgb_raster(
                src,
                out_shape=(disp_h, disp_w),
                config=cfg.section("raster_preprocessing"),
            )
            transform = display_affine(src.transform, full_w, full_h, disp_w, disp_h)
            crs_value = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
        run.record("raster", {
            "full": [full_w, full_h], "display": [disp_w, disp_h],
            "crs": crs_value, "downsample": round(1.0 / keep, 3),
        })
        run.log("读栅格", f"{full_w}x{full_h} 原始 / {disp_w}x{disp_h} 显示，CRS={crs_value}")
    except Exception as exc:
        return run.fail("读栅格", "读取失败", exc)

    # ---- 地理配准 ----------------------------------------------------
    try:
        from pyproj import Transformer

        geo = GeoUtils()
        geo.clear()
        transformer = Transformer.from_crs(crs_value, "EPSG:4326", always_xy=True)
        geo.set_affine(transform, transformer, source_crs=crs_value)
        extent = geo.validate_raster_extent(disp_w, disp_h)
        run.record("georef", {"source_crs": crs_value, "extent": extent})
        run.log("地理配准", f"extent={extent['min_lon']:.6f},{extent['min_lat']:.6f} - "
                            f"{extent['max_lon']:.6f},{extent['max_lat']:.6f}")
    except Exception as exc:
        return run.fail("地理配准", "配准失败", exc)

    # ---- 模型 --------------------------------------------------------
    try:
        model_path = resolve_model(args.model)
        engine = ModelEngine()
        if not engine.load(str(model_path)):
            return run.fail("加载模型", f"加载失败: {model_path}")
        run.record("model", {"path": str(model_path)})
        run.log("加载模型", model_path.name)
    except Exception as exc:
        return run.fail("加载模型", "加载失败", exc)

    # ---- 状态（全图边界） --------------------------------------------
    boundary = full_frame_boundary((disp_h, disp_w))
    state = AppState()
    state.safe_update(
        tif_path=str(tif_path),
        field_boundary=boundary,
        current_model_path=str(model_path),
        display_w=disp_w,
        display_h=disp_h,
        full_width=full_w,
        full_height=full_h,
        mask_offset_x=0,
        mask_offset_y=0,
    )
    run.record("boundary", {"points": len(boundary),
                            "note": "整幅影像矩形（全图识别，不圈选田块）"})

    # ---- 作物识别 ----------------------------------------------------
    try:
        runner = InferenceRunner(state, engine, geo)
        runner.start_from_tif(str(tif_path), (disp_h, disp_w), boundary, 1)
        deadline = time.time() + args.timeout
        while getattr(state, "inference_running", False):
            if time.time() > deadline:
                runner.cancel()
                return run.fail("作物识别", f"超时（{args.timeout}s）")
            time.sleep(0.5)
        mask_raw = getattr(state, "mask_raw", None)
        if mask_raw is None or not getattr(state, "inference_done", False):
            return run.fail(
                "作物识别",
                f"未产出掩膜：{getattr(state, 'status_message', '') or '无输出'}",
            )
        mask_arr = np.asarray(mask_raw)
        run.record("segment", {
            "mask_shape": list(mask_arr.shape),
            "foreground_ratio": float((mask_arr > 0).mean()),
        })
        run.log("作物识别", f"掩膜 {mask_arr.shape}，"
                            f"前景占比 {float((mask_arr > 0).mean()):.4f}")
    except Exception as exc:
        return run.fail("作物识别", "识别失败", exc)

    # ---- 掩膜优化 ----------------------------------------------------
    try:
        from mask_processor import process_mask

        # 掩膜正规化需要米制尺度：geo + state（含掩膜偏移与显示尺寸）。
        processed = process_mask(
            mask_raw,
            geo=geo,
            state=state,
            config=cfg.section("mask_processing"),
        )
        if not isinstance(processed, dict) or processed.get("processed_mask") is None:
            return run.fail("掩膜优化", "未产出处理后掩膜")
        state.safe_update(mask_processed=True, mask_result=processed)
        wide = processed.get("wide_bands") or []
        all_bands = processed.get("all_bands") or []
        run.record("process", {
            "wide_bands": len(wide),
            "all_bands": len(all_bands),
        })
        run.log("掩膜优化", f"{len(wide)} 条 4 行主体区 / {len(all_bands)} 条总行带")
        if not wide:
            return run.fail("掩膜优化", "没有识别出 4 行主体区，下游无法继续")
    except Exception as exc:
        return run.fail("掩膜优化", "处理失败", exc)

    if args.skip_plan:
        run.log("完成", f"已跑通 识别 -> 掩膜优化；报告 {out_dir / 'smoke_report.json'}")
        return 0

    # ---- 作业线 ------------------------------------------------------
    try:
        from path_planner import prepare_operator_centerline_plan

        # 与 workers.CenterlineCatalogWorker 完全相同的调用形态。
        catalog = prepare_operator_centerline_plan(
            processed.get("wide_bands", []),
            np.asarray(processed["processed_mask"]),
            processed.get("main_angle", 0.0),
            geo=geo,
            state=state,
            config=cfg.section("path_planning") if hasattr(cfg, "section") else None,
            detected_mask=getattr(state, "inference_original_mask", None),
        )
        # catalog 在 layout["centerline_catalog"]，顶层没有这个键。
        centerline_catalog = dict(
            (catalog or {}).get("catalog")
            or ((catalog or {}).get("layout") or {}).get("centerline_catalog")
            or {}
        )
        items = centerline_catalog.get("items") or []
        if not items:
            return run.fail("生成作业线", "没有生成任何中心线")
        state.safe_update(
            centerline_catalog=centerline_catalog,
            selected_centerline_ids=[str(i["id"]) for i in items if i.get("id")],
        )
        run.record("centerline", {"count": len(items),
                                  "ids": [str(i.get("id")) for i in items[:5]]})
        run.log("生成作业线", f"{len(items)} 条，已全选")
    except Exception as exc:
        return run.fail("生成作业线", "生成失败", exc)

    # ---- 起点 --------------------------------------------------------
    try:
        from path_planner import prepare_band_centerlines

        lines, _ = prepare_band_centerlines(
            processed.get("wide_bands", []),
            float(processed.get("main_angle", 0.0) or 0.0),
            geo=geo,
            state=state,
            config=cfg.section("path_planning"),
        )
        if not lines:
            return run.fail("起点", "没有可用作业线端点，无法确定起点")
        # 与界面 _suggest_service_points 相同的规则：取第一条线起点并沿
        # 反方向外延 1 m 作为进场点。
        first = lines[0]
        entry = np.asarray(first[0], dtype=np.float64)
        nxt = np.asarray(first[1], dtype=np.float64)
        ppm = geo.pixels_per_meter(float(entry[0]), float(entry[1]))
        direction = entry - nxt
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            entry = entry + direction / norm * max(1.0, ppm)
        state.safe_update(entry_point=(float(entry[0]), float(entry[1])),
                          entry_point_locked=True,
                          harvester_params_confirmed=True)
        run.record("entry", {"point": [float(entry[0]), float(entry[1])]})
        run.log("起点", f"({entry[0]:.1f}, {entry[1]:.1f}) px，已确认")
    except Exception as exc:
        return run.fail("起点", "布设失败", exc)

    # ---- 路径规划 ----------------------------------------------------
    try:
        from path_planner import plan_path

        path_cfg = cfg.section("path_planning").copy()
        path_cfg["selected_centerline_ids"] = list(state.selected_centerline_ids)
        result = plan_path(
            processed.get("wide_bands", []),
            np.asarray(processed["processed_mask"]),
            float(processed.get("main_angle", 0.0) or 0.0),
            geo=geo,
            state=state,
            config=path_cfg,
            detected_mask=getattr(state, "inference_original_mask", None),
            all_bands=processed.get("all_bands") or [],
        )
        segments = result.get("full_path") or []
        validation = result.get("validation") or {}
        if not segments:
            return run.fail("路径规划", f"未生成路径：{result.get('description') or '无描述'}")
        state.safe_update(auto_path_planned=True,
                          auto_path_segments=segments,
                          auto_path_valid=bool(result.get("is_valid")))
        run.record("plan", {
            "segments": len(segments),
            "is_valid": bool(result.get("is_valid")),
            "description": str(result.get("description") or ""),
            "validation_keys": sorted(validation.keys())[:12],
        })
        run.log("路径规划", f"{len(segments)} 段，valid={result.get('is_valid')}，"
                            f"{result.get('description') or ''}")
    except Exception as exc:
        return run.fail("路径规划", "规划失败", exc)

    # ---- 导出路径文件（真正的终点：拿到可交给终端的 TXT） ------------
    try:
        from path_planner import export_machine_compatibility_path

        geo_points = result.get("geo_points") or []
        if not geo_points:
            return run.fail("路径导出", "规划结果没有经纬度点，无法导出")
        export_target = out_dir / "route"
        compat_points, export_diag = export_machine_compatibility_path(
            geo_points,
            str(export_target),
        )
        files = [Path(p) for p in (export_diag.get("output_files") or [])]
        if not files:
            return run.fail("路径导出", "导出函数没有返回任何文件")

        # 回读校验：文件必须真实存在、非空、且首行/末行符合终端样例格式
        verified = []
        for fp in files:
            if not fp.is_file():
                return run.fail("路径导出", f"声明产物不存在: {fp}")
            text = fp.read_text(encoding="utf-8")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                return run.fail("路径导出", f"产物为空: {fp.name}")
            if not lines[0].startswith("$PATH,0,"):
                return run.fail("路径导出", f"{fp.name} 首行不是 $PATH 记录: {lines[0][:60]}")
            if not lines[-1].startswith("$INFM"):
                return run.fail("路径导出", f"{fp.name} 末行不是 $INFM: {lines[-1][:60]}")
            path_records = sum(1 for ln in lines if ln.startswith("$PATH"))
            verified.append({
                "file": fp.name,
                "bytes": fp.stat().st_size,
                "path_records": path_records,
            })
        run.record("export", {
            "dir": str(out_dir),
            "file_count": len(files),
            "point_count": len(compat_points),
            "files": verified,
            "diagnostics_keys": sorted(export_diag.keys())[:12],
        })
        run.log("路径导出", f"{len(files)} 个文件，共 {len(compat_points)} 点；"
                            f"首份 {verified[0]['file']}（{verified[0]['path_records']} 条 $PATH）")
    except Exception as exc:
        return run.fail("路径导出", "导出失败", exc)

    run.log("完成", f"全链路跑通（读图→识别→掩膜→作业线→起点→规划→导出）；"
                    f"产物在 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
