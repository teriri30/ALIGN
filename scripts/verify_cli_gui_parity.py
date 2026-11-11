"""R2 契约验证：CLI 规划落盘结果 vs 界面 PlanWorker 装配链现场重算。

方法：用 CLI 自己的 Ctx 恢复项目现场（state/geo/掩膜），再调用
pyside6_app.workers 的界面同款函数链（path_planning_config_snapshot →
planning_config_with_mask_evidence → plan_path → attach_path_provenance）现场重算，
逐项对比 CLI 落盘的 path_validation。几何阶段确定性 → 容差 1e-9。
"""
import json
import os
import sys

SRC = "E:/zhlonly/project/ALIGN/src"
sys.path.insert(0, SRC)
os.environ.setdefault("ALIGN_DISABLE_AUTO_START", "1")

TIF = r"C:\Users\Administrator\AppData\Local\hermes\cache\scratch\align_cli_test\cli\result.tif"


def main() -> int:
    from align_cli import Ctx, _attach_processed_mask, _attach_raw_mask
    from pyside6_app.workers import (
        attach_path_provenance,
        path_planning_config_snapshot,
        planning_config_with_mask_evidence,
        require_metric_scale,
    )

    ctx = Ctx(TIF)
    saved = ctx.saved
    cli_validation = dict(saved.get("path_validation") or {})
    assert cli_validation, "CLI 缓存中没有 path_validation（先跑 plan）"

    # 回滚到 plan 前现场：CLI 的 plan 是在无前序路径的干净状态上跑的，
    # 带着 auto_path 复跑会走重规划分支，污染对比。
    from state import AppState
    fresh = AppState()
    for key in (
        "auto_path", "auto_path_segments", "auto_path_geo", "auto_path_planned",
        "auto_path_valid", "auto_path_desc", "path_provenance", "path_validation",
        "path_artifact", "path_runtime", "path_points", "path_status", "path_geometry_modes",
    ):
        setattr(ctx.state, key, getattr(fresh, key))

    # 界面同款恢复链（MW:2330-2349 支撑层推导在 _attach_processed_mask 内）
    _attach_processed_mask(ctx)
    raw = _attach_raw_mask(ctx)
    mr = dict(ctx.state.mask_result or {})
    processed = mr.get("processed_mask")
    assert processed is not None and raw is not None

    require_metric_scale(ctx.geo, ctx.state, processed.shape)

    # 界面 PlanWorker 装配链真身（workers.py:1266-1357）
    config = path_planning_config_snapshot({})
    config.update({
        "min_turn_radius_m": float(ctx.state.harvester_params["turn_radius_m"]),
        "turn_strategy": str(getattr(ctx.state, "turn_strategy", "auto") or "auto"),
        "service_point_mode": "reference_only",
        "include_route_finish_extension": True,
        "forbidden_mask_confirmed": bool(getattr(ctx.state, "forbidden_regions_confirmed", False)),
    })
    selection_report = getattr(ctx.state, "centerline_selection_report", None) or {}
    selected_ids = list(getattr(ctx.state, "selected_centerline_ids", None) or [])
    if selected_ids and selection_report.get("selection_origin") == "operator":
        config["selected_centerline_ids"] = selected_ids
    from path_planner import normalize_turn_strategy
    if normalize_turn_strategy(config["turn_strategy"]) == "contour":
        config["route_order"] = "contour"
    config = planning_config_with_mask_evidence(config, mr, ctx.state, processed.shape)

    from path_planner import build_band_mask, plan_path
    wide_bands = mr.get("wide_bands") or []
    result = plan_path(
        wide_bands, processed, mr.get("main_angle", 0.0) or 0.0,
        geo=ctx.geo, state=ctx.state, config=config,
        detected_mask=raw,
        planned_mask=build_band_mask(processed.shape, wide_bands),
        all_bands=mr.get("all_bands") or [],
    )
    layout = result.get("layout") or {}
    selection = layout.get("centerline_selection")
    if not isinstance(selection, dict):
        selection = layout.get("work_band_selection")
    if isinstance(selection, dict):
        config["selected_centerline_ids"] = list(selection.get("selected_ids") or [])
    attach_path_provenance(result, mr, ctx.state, config)

    # 诊断：与 CLI 持久化溯源逐项对照
    cli_inputs = (saved.get("path_provenance") or {}).get("inputs") or {}
    gui_prov = result.get("provenance") or {}
    gui_inputs = gui_prov.get("inputs") or {}
    print("== 指纹/输入对比 ==")
    print("semantic fp match:", cli_inputs.get("semantic_mask_fingerprints") == gui_inputs.get("semantic_mask_fingerprints"))
    print("  cli:", json.dumps(cli_inputs.get("semantic_mask_fingerprints")))
    print("  gui:", json.dumps(gui_inputs.get("semantic_mask_fingerprints")))
    print("mask_fingerprint:", cli_inputs.get("mask_fingerprint") == gui_inputs.get("mask_fingerprint"))
    print("entry:", cli_inputs.get("entry_point"), "|", gui_inputs.get("entry_point"))
    print("harvester match:", cli_inputs.get("harvester") == gui_inputs.get("harvester"))
    import collections
    print("cli segment_types:", dict(collections.Counter(s.get("segment_type") for s in (saved.get("auto_path") or []))))
    print("gui segment_types:", dict(collections.Counter(result.get("segment_types") or [])))

    gui_v = result.get("validation") or {}
    keys = [
        "total_length_m", "work_length_m", "turn_length_m", "entry_exit_length_m",
        "rolling_canopy_pct", "rolling_body_pct", "rolling_headland_pct",
        "rolling_core_pct", "track_core_overlap_pct",
        "planned_target_geometric_coverage_pct", "crop_mask_geometric_coverage_pct",
        "detected_mask_geometric_coverage_pct", "track_outside_field_pct",
        "field_efficiency_pct", "valid",
    ]
    diffs = []
    for key in keys:
        a, b = cli_validation.get(key), gui_v.get(key)
        if a is None and b is None:
            continue
        if isinstance(a, bool) or isinstance(b, bool):
            ok = bool(a) == bool(b)
        else:
            try:
                ok = abs(float(a) - float(b)) <= 1e-9
            except (TypeError, ValueError):
                ok = a == b
        if not ok:
            diffs.append({"key": key, "cli": a, "gui": b})
    seg_cli = len(saved.get("auto_path") or [])
    seg_gui = len(result.get("full_path") or [])
    geo_cli = len(saved.get("auto_path_geo") or [])
    geo_gui = len(result.get("geo_points") or [])

    print(json.dumps({
        "segment_count": {"cli": seg_cli, "gui": seg_gui, "match": seg_cli == seg_gui},
        "geo_points": {"cli": geo_cli, "gui": geo_gui, "match": geo_cli == geo_gui},
        "metric_keys_compared": len(keys),
        "diffs": diffs,
        "verdict": "PASS" if not diffs and seg_cli == seg_gui and geo_cli == geo_gui else "FAIL",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
