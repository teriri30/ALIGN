"""Pure task-state rules for the PySide6 workflow panel.

This module intentionally has no Qt/PySide dependency so task status can be
validated without launching the desktop UI.  The UI should render these derived
states instead of hand-maintaining labels in every action handler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


TASK_ORDER: List[str] = [
    "import_img",
    "draw_field",
    "segment",
    "process",
    "params",
    "centerline_generate",
    "centerline_select",
    "entry",
    "exit",
    "unload",
    "plan",
    "simulate",
    "export",
]

TASK_LABELS: Dict[str, str] = {
    "import_img": "影像导入",
    "draw_field": "田块圈选",
    "segment": "作物识别",
    "process": "掩膜优化",
    "params": "农机参数",
    "centerline_generate": "生成中心线",
    "centerline_select": "选择中心线",
    "entry": "起点",
    "exit": "终点",
    "unload": "卸粮点",
    "plan": "路径规划",
    "simulate": "模拟演示",
    "export": "TXT路径导出",
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "import_img": "影像、田块、识别结果、路径与导出状态都会清除。",
    "draw_field": "田块边界及其后的 作物识别、掩膜、起终点、路径、演示、导出都会清除；底图保留。",
    "segment": "作物识别结果及其后的掩膜、起终点、路径、演示、导出都会清除；底图和田块边界保留。",
    "process": "掩膜优化结果及其后的智能起终点、路径、演示、导出都会清除；原始 作物识别和田块边界保留。",
    "params": "农机参数及其后的起终点、路径、演示、导出都会回退。",
    "centerline_generate": "候选作业线目录及其后的作业线选择、起终点、路径、演示、导出都会回退；掩膜和农机参数保留。",
    "centerline_select": "作业线选择及其后的起终点、路径、演示、导出都会回退；已生成的候选作业线保留。",
    "entry": "起点及其后的可选终点、卸粮点、路径、演示、导出都会回退。",
    "exit": "可选终点及其后的卸粮点、路径、演示、导出都会回退。",
    "unload": "卸粮点及其后的路径、演示、导出都会回退；起点和终点保留。",
    "plan": "路径规划、模拟演示和导出结果会清除；影像、田块、掩膜、起终点和卸粮点保留。",
    "simulate": "仅回退模拟演示状态；已规划路径保留。",
    "export": "仅回退导出完成状态；路径和演示状态保留。",
}


@dataclass(frozen=True)
class TaskStatus:
    status: str = "pending"  # pending, available, in_progress, done, failed
    text: Optional[str] = None
    action: Optional[str] = None


def _has_harvester_base_params(params: Any) -> bool:
    return bool(
        isinstance(params, dict)
        and params.get("cutter_width_m")
        and params.get("track_width_m")
        and params.get("track_gauge_m")
        and params.get("wheelbase_m")
        and params.get("track_length_m")
        and params.get("turn_radius_m")
    )


def _has_cutter_forward_offset(params: Any) -> bool:
    return bool(isinstance(params, dict) and params.get("cutter_forward_offset_m"))


def _has_points(value: Any) -> bool:
    return bool(value)


def derive_task_statuses(state: Any, has_image: bool = False) -> Dict[str, TaskStatus]:
    """Return user-facing task statuses from the actual domain state.

    Rules follow common GIS/CAD workflow logic:
    - completed prerequisites are green;
    - exactly actionable next steps are marked available;
    - start is the required route anchor; end and unload points are optional;
    - an auto-suggested start remains pending until the user confirms it.
    """
    statuses = {tid: TaskStatus() for tid in TASK_ORDER}

    image_loaded = bool(has_image or getattr(state, "tif_path", ""))
    field_done = bool(getattr(state, "field_boundary", None))
    inference_running = bool(getattr(state, "inference_running", False))
    inference_done = bool(getattr(state, "inference_done", False) and getattr(state, "mask_raw", None) is not None)
    process_running = bool(getattr(state, "mask_processing_running", False))
    process_done = bool(getattr(state, "mask_processed", False) and getattr(state, "mask_result", None) is not None)
    plan_running = bool(getattr(state, "plan_running", False))
    harvester_params = getattr(state, "harvester_params", {})
    params_ready = _has_harvester_base_params(harvester_params)
    cutter_offset_ready = _has_cutter_forward_offset(harvester_params)
    params_done = bool(params_ready and cutter_offset_ready)
    params_confirmed = bool(getattr(state, "harvester_params_confirmed", False))
    entry = getattr(state, "entry_point", None)
    exit_point = getattr(state, "exit_point", None)
    unload_points = list(getattr(state, "unload_points", []) or [])
    path_done = bool(getattr(state, "auto_path_planned", False))
    centerline_catalog = getattr(state, "centerline_catalog", None) or getattr(
        state, "work_band_catalog", {}
    ) or {}
    centerline_items = list(centerline_catalog.get("items") or [])
    centerline_ids = {
        str(item.get("id")) for item in centerline_items if item.get("id") is not None
    }
    selected_centerlines = list(
        getattr(state, "selected_centerline_ids", None)
        or getattr(state, "selected_work_band_ids", [])
        or []
    )
    centerlines_selected = bool(
        centerline_ids.intersection(str(value) for value in selected_centerlines)
    )
    sim_done = bool(getattr(state, "simulation_done", False))
    export_done = bool(getattr(state, "export_done", False))

    statuses["import_img"] = TaskStatus("done" if image_loaded else "available", action="重选" if image_loaded else "导入")
    if image_loaded:
        statuses["draw_field"] = TaskStatus("done" if field_done else "available", action="重选" if field_done else "圈选")
    if field_done:
        statuses["segment"] = TaskStatus("done" if inference_done else "available", action="重跑" if inference_done else "运行")
    if inference_running:
        statuses["segment"] = TaskStatus("in_progress", "识别中")
    if inference_done:
        statuses["process"] = TaskStatus("done" if process_done else "available", action="重做" if process_done else "处理")
    if process_running:
        statuses["process"] = TaskStatus("in_progress", "处理中")
    if process_done:
        if params_done and params_confirmed:
            statuses["params"] = TaskStatus("done")
        elif params_done:
            statuses["params"] = TaskStatus("available", "待确认", action="确认")
        elif params_ready:
            statuses["params"] = TaskStatus("available", "缺割刀前置", action="补填")
        else:
            statuses["params"] = TaskStatus("available")

        if not centerline_items:
            statuses["centerline_generate"] = TaskStatus(
                "available", "待生成", action="生成"
            )
        else:
            statuses["centerline_generate"] = TaskStatus("done")
            selected_count = len(
                centerline_ids.intersection(str(value) for value in selected_centerlines)
            )
            statuses["centerline_select"] = TaskStatus(
                "done" if selected_count else "available",
                f"已选择 {selected_count}/{len(centerline_ids)} 条"
                if selected_count
                else "待选择",
                action="调整" if selected_count else "选择",
            )
        if not centerline_items:
            statuses["centerline_select"] = TaskStatus("pending", "先生成中心线")

        if not centerline_items or not centerlines_selected:
            statuses["entry"] = TaskStatus("pending", "先选择中心线")
            statuses["exit"] = TaskStatus("pending", "先选择中心线")
            statuses["unload"] = TaskStatus("pending", "可选；先选择中心线")
        else:
            if entry:
                locked = bool(getattr(state, "entry_point_locked", False))
                statuses["entry"] = TaskStatus(
                    "done" if locked else "available",
                    "已确认" if locked else "推荐待确认",
                    action="调整" if locked else "确认",
                )
            else:
                statuses["entry"] = TaskStatus("available", "待设置")
            if exit_point:
                locked = bool(getattr(state, "exit_point_locked", False))
                statuses["exit"] = TaskStatus(
                    "done" if locked else "available",
                    "已确认（可选）" if locked else "待确认（可选）",
                    action="调整" if locked else "确认",
                )
            else:
                statuses["exit"] = TaskStatus("done", "自动延伸", action="调整")
            if unload_points:
                statuses["unload"] = TaskStatus("done", f"{len(unload_points)} 个")
            else:
                statuses["unload"] = TaskStatus("done", "可选")

    ready_to_plan = bool(
        process_done
        and centerlines_selected
        and params_ready
        and params_confirmed
        and _has_points(entry)
        and getattr(state, "entry_point_locked", False)
    )
    if process_done and not centerline_items:
        statuses["plan"] = TaskStatus("pending", "先生成中心线")
    elif process_done and centerline_items and not centerlines_selected:
        statuses["plan"] = TaskStatus("pending", "先选择中心线")
    elif process_done and centerlines_selected and not params_confirmed:
        statuses["plan"] = TaskStatus("pending", "先确认高级设置")
    elif process_done and centerlines_selected and not getattr(
        state, "entry_point_locked", False
    ):
        statuses["plan"] = TaskStatus("pending", "先设置并确认起点")
    elif ready_to_plan:
        statuses["plan"] = TaskStatus("done" if path_done else "available", action="重算" if path_done else "生成")
    if plan_running:
        statuses["plan"] = TaskStatus("in_progress", "规划中")
    if path_done:
        statuses["simulate"] = TaskStatus("done" if sim_done else "available")
        statuses["export"] = TaskStatus("done" if export_done else "available")
    return statuses


def rollback_summary(task_id: str) -> str:
    label = TASK_LABELS.get(task_id, task_id)
    detail = TASK_DESCRIPTIONS.get(task_id, "将回退该步骤及后续结果。")
    return f"将回退到“{label}”之前。\n\n{detail}\n\n此操作可通过撤销恢复（若当前会话未关闭）。"
