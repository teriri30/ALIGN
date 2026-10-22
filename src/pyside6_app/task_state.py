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
    #: Optional steps never block downstream work but still change its result
    #: (exit point changes the return leg, unload points add detours), so the
    #: UI marks them instead of locking them.
    optional: bool = False
    #: What must happen first, kept separate from ``text`` so the row can show a
    #: short state ("待处理") and only surface the reason when the user clicks.
    blocker: Optional[str] = None


#: Steps that may be skipped but still affect the planned route.
#: 「高级设置」也是可选：默认参数来自配置，未确认同样能规划。
OPTIONAL_TASKS = ("exit", "unload", "params")


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

    # Each stage is only "done" while its own input still exists. Checking the
    # whole prerequisite chain (not just the stage's own flag) keeps a stale
    # artifact from surviving an upstream rollback or field change: downstream
    # work then reads pending, never done.
    image_loaded = bool(has_image or getattr(state, "tif_path", ""))
    field_done = bool(image_loaded and getattr(state, "field_boundary", None))
    inference_running = bool(getattr(state, "inference_running", False))
    # A stage is done when the image it was computed from is still loaded and
    # its own artifact still exists. The field polygon is deliberately NOT part
    # of this test: invalidation is triggered by the *action* that changes an
    # upstream artifact (``_finish_field_polygon`` calls
    # ``_invalidate_analysis_from("inference")``), so inferring staleness from
    # an empty polygon would also demote a project whose cache is only
    # partially restored. Cache entries store ``field_boundary`` and
    # ``inference_done`` together, so a mask without a polygon is a corrupt
    # archive, not a normal state.
    inference_done = bool(
        image_loaded
        and getattr(state, "inference_done", False)
        and getattr(state, "mask_raw", None) is not None
    )
    process_running = bool(getattr(state, "mask_processing_running", False))
    process_done = bool(
        inference_done
        and getattr(state, "mask_processed", False)
        and getattr(state, "mask_result", None) is not None
    )
    plan_running = bool(getattr(state, "plan_running", False))
    params_confirmed = bool(getattr(state, "harvester_params_confirmed", False))
    entry = getattr(state, "entry_point", None)
    exit_point = getattr(state, "exit_point", None)
    unload_points = list(getattr(state, "unload_points", []) or [])
    path_done = bool(getattr(state, "auto_path_planned", False))
    centerline_catalog = getattr(state, "centerline_catalog", None) or getattr(
        state, "work_band_catalog", {}
    ) or {}
    # A catalog built from an older mask is stale: treat it as absent so the
    # next step reads "regenerate" instead of silently planning on old lines.
    centerline_items = list(centerline_catalog.get("items") or []) if process_done else []
    centerline_ids = {
        str(item.get("id")) for item in centerline_items if item.get("id") is not None
    }
    selected_centerlines = (
        list(
            getattr(state, "selected_centerline_ids", None)
            or getattr(state, "selected_work_band_ids", [])
            or []
        )
        if centerline_items
        else []
    )
    centerlines_selected = bool(
        centerline_ids.intersection(str(value) for value in selected_centerlines)
    )
    selection_report = getattr(state, "centerline_selection_report", {}) or {}
    selection_origin = str(selection_report.get("selection_origin") or "")
    # "default_all" 只是生成后的自动全选建议，用户还没看过；把它当成已确认
    # 会让「选择作业线」卡片刚展开就被状态推进跳到下一步。
    # 显式确认记在 selection_confirmed（勾选动作不会把它改回 False）；
    # 老缓存没有该字段时按 origin 推断，保持既有项目行为。
    selection_confirmed = bool(
        selection_report.get(
            "selection_confirmed", selection_origin != "default_all"
        )
    )
    sim_done = bool(getattr(state, "simulation_done", False))
    export_done = bool(getattr(state, "export_done", False))

    statuses["import_img"] = TaskStatus(
        "done" if image_loaded else "available",
        action="重选" if image_loaded else "导入",
    )
    # 锁定原因必须指向"最上游那个真正缺的步骤"，不能一律说紧邻的上一步。
    # 只导入影像、还没圈选田块时，点"掩膜优化"应提示"先圈选田块"，
    # 而不是"先完成作物识别"——后者在田块都没圈的时候是误导。
    def _first_missing_blocker():
        if not image_loaded:
            return "先导入影像"
        if not field_done:
            return "先圈选田块"
        if not inference_done:
            return "先完成作物识别"
        if not process_done:
            return "先完成掩膜优化"
        return None

    if not image_loaded:
        for tid in ("draw_field", "segment", "process"):
            statuses[tid] = TaskStatus("pending", blocker="先导入影像")
    if image_loaded and not field_done and not inference_done:
        statuses["segment"] = TaskStatus("pending", blocker="先圈选田块")
    if image_loaded and not inference_done:
        statuses["process"] = TaskStatus(
            "pending", blocker=_first_missing_blocker()
        )
    if image_loaded:
        statuses["draw_field"] = TaskStatus(
            "done" if field_done else "available",
            action="重选" if field_done else "圈选",
        )
    if field_done or inference_done:
        statuses["segment"] = TaskStatus(
            "done" if inference_done else "available",
            "已识别" if inference_done else "待识别",
            action="重跑" if inference_done else "运行",
        )
    if inference_running:
        statuses["segment"] = TaskStatus("in_progress", "识别中")
    if inference_done:
        statuses["process"] = TaskStatus(
            "done" if process_done else "available",
            action="重做" if process_done else "处理",
        )
    elif not image_loaded:
        statuses["process"] = TaskStatus("pending", blocker="先导入影像")
    if process_running:
        statuses["process"] = TaskStatus("in_progress", "处理中")

    if process_done:
        # 高级设置是「可动可不动」的一步：默认参数由配置提供，未确认也能规划；
        # 确认只是把参数来源标成用户录入/实测（正式导出仍按 provenance 校验）。
        statuses["params"] = TaskStatus(
            "done",
            "已确认" if params_confirmed else "默认参数（可修改）",
            action="修改",
            optional=True,
        )
    else:
        statuses["params"] = TaskStatus(
            "pending", blocker="先完成掩膜优化", optional=True
        )

    lines_ready = bool(centerline_items and centerlines_selected)
    if not process_done:
        for tid, why in (
            ("centerline_generate", "先完成掩膜优化"),
            ("centerline_select", "先完成掩膜优化"),
            ("entry", "先完成掩膜优化"),
            ("exit", "先完成掩膜优化"),
            ("unload", "先完成掩膜优化"),
        ):
            statuses[tid] = TaskStatus(
                "pending", optional=tid in OPTIONAL_TASKS, blocker=why
            )
    else:
        if not centerline_items:
            statuses["centerline_generate"] = TaskStatus("available", "待生成", action="生成")
            statuses["centerline_select"] = TaskStatus("pending", blocker="先生成作业线")
        else:
            statuses["centerline_generate"] = TaskStatus("done", action="重新生成")
            selected_count = len(
                centerline_ids.intersection(str(value) for value in selected_centerlines)
            )
            if not selected_count:
                selection_text = "待选择"
            elif selection_confirmed:
                selection_text = f"已选择 {selected_count}/{len(centerline_ids)} 条"
            else:
                selection_text = f"已选择 {selected_count}/{len(centerline_ids)} 条（待确认）"
            statuses["centerline_select"] = TaskStatus(
                "done" if selected_count and selection_confirmed else "available",
                selection_text,
                action=(
                    "调整"
                    if selected_count and selection_confirmed
                    else ("确认" if selected_count else "选择")
                ),
            )

        if not lines_ready:
            blocker = "先生成作业线" if not centerline_items else "先选择作业线"
            statuses["entry"] = TaskStatus("pending", blocker=blocker)
            statuses["exit"] = TaskStatus("pending", optional=True, blocker=blocker)
            statuses["unload"] = TaskStatus("pending", optional=True, blocker=blocker)
        else:
            if entry:
                locked = bool(getattr(state, "entry_point_locked", False))
                statuses["entry"] = TaskStatus(
                    "done" if locked else "available",
                    "已确认" if locked else "推荐待确认",
                    action="调整" if locked else "确认",
                )
            else:
                statuses["entry"] = TaskStatus("available", "待布设起点")
            # Optional steps stay actionable (they can still change the route)
            # but are flagged optional so the card reads "选填" instead of
            # looking like unfulfilled required work.
            statuses["exit"] = TaskStatus(
                "done",
                "已设置" if exit_point else "未设置（自动延伸）",
                action="调整" if exit_point else "布设", optional=True,
            )
            # Unset optional points are a valid final state ("skipped"), not an
            # unfinished step: they read done + optional, with the text saying
            # the default that will be used instead.
            statuses["unload"] = TaskStatus(
                "done",
                f"已设置 {len(unload_points)} 个" if unload_points else "未设置",
                action="布设", optional=True,
            )

    ready_to_plan = bool(
        process_done
        and centerlines_selected
        and _has_points(entry)
        and getattr(state, "entry_point_locked", False)
    )
    if not image_loaded:
        statuses["plan"] = TaskStatus("pending", blocker="先导入影像")
    elif not field_done and not inference_done:
        # 田块都没圈时，说"先完成作物识别"是误导：识别本身就跑不了。
        statuses["plan"] = TaskStatus("pending", blocker="先圈选田块")
    elif not inference_done:
        statuses["plan"] = TaskStatus("pending", blocker="先完成作物识别")
    elif not process_done:
        statuses["plan"] = TaskStatus("pending", blocker="先完成掩膜优化")
    elif not centerline_items:
        statuses["plan"] = TaskStatus("pending", blocker="先生成作业线")
    elif not centerlines_selected:
        statuses["plan"] = TaskStatus("pending", blocker="先选择作业线")
    elif not getattr(state, "entry_point_locked", False):
        statuses["plan"] = TaskStatus("pending", blocker="先设置并确认起点")
    elif ready_to_plan:
        statuses["plan"] = TaskStatus(
            "done" if path_done else "available", action="重算" if path_done else "生成"
        )
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
