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
    "segment": "AI识别",
    "process": "掩膜处理",
    "params": "农机参数",
    "entry": "起点",
    "exit": "终点",
    "unload": "卸粮点",
    "plan": "路径规划",
    "simulate": "模拟演示",
    "export": "路径导出",
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "import_img": "影像、田块、识别结果、路径与导出状态都会清除。",
    "draw_field": "田块边界及其后的 AI 识别、掩膜、起终点、路径、演示、导出都会清除；底图保留。",
    "segment": "AI 识别结果及其后的掩膜、起终点、路径、演示、导出都会清除；底图和田块边界保留。",
    "process": "掩膜处理结果及其后的智能起终点、路径、演示、导出都会清除；原始 AI 识别和田块边界保留。",
    "params": "农机参数及其后的起终点、路径、演示、导出都会回退。",
    "entry": "起点及其后的终点、卸粮点、路径、演示、导出都会回退。",
    "exit": "终点及其后的卸粮点、路径、演示、导出都会回退。",
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
    - optional unload point is not a blocker once start/end exist;
    - auto-suggested start/end are complete but labelled as recommendations.
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
    entry = getattr(state, "entry_point", None)
    exit_point = getattr(state, "exit_point", None)
    unload_points = list(getattr(state, "unload_points", []) or [])
    path_done = bool(getattr(state, "auto_path_planned", False))
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
        if params_done:
            statuses["params"] = TaskStatus("done")
        elif params_ready:
            statuses["params"] = TaskStatus("available", "缺割刀前置", action="补填")
        else:
            statuses["params"] = TaskStatus("available")
        if entry:
            text = "已确认" if getattr(state, "entry_point_locked", False) else "智能推荐"
            statuses["entry"] = TaskStatus("done", text)
        else:
            statuses["entry"] = TaskStatus("available", "待设置")
        if exit_point:
            text = "已确认" if getattr(state, "exit_point_locked", False) else "智能推荐"
            statuses["exit"] = TaskStatus("done", text)
        else:
            statuses["exit"] = TaskStatus("available", "待设置")
        if unload_points:
            statuses["unload"] = TaskStatus("done", f"{len(unload_points)} 个")
        elif entry and exit_point:
            statuses["unload"] = TaskStatus("done", "可选")
        else:
            statuses["unload"] = TaskStatus("pending", "可选")

    ready_to_plan = process_done and params_ready and _has_points(entry) and _has_points(exit_point)
    if ready_to_plan:
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
