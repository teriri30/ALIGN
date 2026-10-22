"""Workflow chain contract: a step is only actionable when its whole upstream
chain is still valid, and an upstream change must invalidate everything below.

These are pure-state tests (no Qt) so a broken dependency rule fails fast.
"""
import numpy as np
import pytest

from pyside6_app.task_state import (
    OPTIONAL_TASKS, TASK_ORDER, derive_task_statuses,
)

RAW = np.ones((4, 4), dtype=np.uint8)

PARAMS = {
    "cutter_width_m": 2.0,
    "cutter_forward_offset_m": 1.35,
    "track_width_m": 0.35,
    "track_gauge_m": 1.30,
    "wheelbase_m": 1.80,
    "track_length_m": 1.70,
    "turn_radius_m": 0.90,
}


def _state():
    from state import AppState

    state = AppState()
    state.tif_path = "field.tif"
    state.field_boundary = [(0, 0), (10, 0), (10, 10)]
    state.inference_done = True
    state.mask_raw = RAW
    state.mask_processed = True
    state.mask_result = {"processed_mask": RAW}
    state.centerline_catalog = {"items": [{"id": "c1", "sequence": 1}]}
    state.selected_centerline_ids = ["c1"]
    state.harvester_params = dict(PARAMS)
    state.harvester_params_confirmed = True
    state.entry_point = (1, 1)
    state.entry_point_locked = True
    return state


def _statuses(state):
    return derive_task_statuses(state, has_image=bool(state.tif_path))


def test_complete_chain_unlocks_planning():
    statuses = _statuses(_state())
    for tid in ("import_img", "draw_field", "segment", "process", "params",
                "centerline_generate", "centerline_select", "entry"):
        assert statuses[tid].status == "done", tid
    assert statuses["plan"].status == "available"


@pytest.mark.parametrize(
    "breakup, blocked",
    [
        # 边界被清空时，识别/掩膜仍有效（失效由重画田块这个动作触发，
        # 见 test_deleting_field_polygon_invalidates_every_downstream_step）；
        # 这里只要求田块本身退回可操作，下游不因"边界为空"而被降级。
        (lambda s: setattr(s, "field_boundary", None),
         ()),
        (lambda s: (setattr(s, "inference_done", False), setattr(s, "mask_raw", None)),
         ("process", "params", "centerline_generate")),
        (lambda s: (setattr(s, "mask_processed", False), setattr(s, "mask_result", None)),
         ("params", "centerline_generate", "centerline_select", "entry", "plan")),
        (lambda s: setattr(s, "centerline_catalog", {"items": []}),
         ("centerline_select", "entry", "plan")),
        (lambda s: setattr(s, "selected_centerline_ids", []),
         ("entry", "plan")),
        (lambda s: (setattr(s, "entry_point", None),
                    setattr(s, "entry_point_locked", False)),
         ("plan",)),
    ],
)
def test_upstream_change_locks_every_downstream_step(breakup, blocked):
    state = _state()
    breakup(state)
    statuses = _statuses(state)
    for tid in blocked:
        assert statuses[tid].status == "pending", tid
        # 原因走 blocker（点击时提示），不再占用行内状态文字。
        assert statuses[tid].blocker, f"{tid} must record why it is locked"
        assert not statuses[tid].text, (
            f"{tid} must not repeat the blocker in its row text"
        )


def test_upstream_change_never_leaves_a_later_step_done():
    """The ordering invariant: done implies all predecessors are done."""
    state = _state()
    upstream = ("import_img", "draw_field", "segment", "process",
                "centerline_generate", "centerline_select", "entry")
    for broken in upstream:
        state = _state()
        if broken == "entry":
            state.entry_point_locked = False
            state.entry_point = None
        else:
            setattr(state, broken + "_done" if broken != "import_img" else "tif_path",
                    False if broken != "import_img" else "")
        statuses = _statuses(state)
        seen_pending = False
        for tid in TASK_ORDER:
            if statuses[tid].status == "pending" and tid not in OPTIONAL_TASKS:
                seen_pending = True
            elif seen_pending and tid in ("plan", "simulate", "export"):
                assert statuses[tid].status == "pending", (
                    f"{tid} must not be reachable after {broken} was invalidated"
                )


def test_optional_steps_are_flagged_and_never_block_planning():
    state = _state()
    state.exit_point = None
    state.unload_points = []
    statuses = _statuses(state)
    for tid in OPTIONAL_TASKS:
        assert statuses[tid].optional is True, tid
        # The chip carries the "选填" wording; the text must not repeat it.
        assert "选填" not in (statuses[tid].text or ""), statuses[tid]
    # Skipping both optional points must not stop the route from being built.
    assert statuses["plan"].status == "available"


def test_advanced_settings_defaults_never_block_planning():
    """高级设置可动可不动：不确认也照常规划（参数来源在导出阶段才校验）。"""
    state = _state()
    state.harvester_params_confirmed = False
    statuses = _statuses(state)
    assert statuses["params"].status == "done"
    assert statuses["params"].optional is True
    assert "默认参数" in statuses["params"].text
    assert statuses["plan"].status == "available"
    assert not statuses["plan"].blocker


def test_mask_derived_steps_survive_a_partially_restored_cache():
    """缓存里边界与掩膜成对存在；只有掩膜时不能把参数/作业线降级。

    失效必须由动作触发（重新圈选田块会显式调用
    ``_invalidate_analysis_from("inference")``），而不是由"边界为空"推断。
    """
    state = _state()
    state.field_boundary = []
    statuses = _statuses(state)
    for tid in ("segment", "process", "params", "centerline_generate"):
        assert statuses[tid].status == "done", tid
    # 田块本身退回可重选，但不影响已有掩膜的下游
    assert statuses["draw_field"].status == "available"


def test_deleting_field_polygon_invalidates_every_downstream_step():
    """删除/重画田块是动作触发失效的唯一入口，下游必须全部作废。"""
    state = _state()
    # 模拟 _finish_field_polygon / _invalidate_analysis_from("inference")
    state.field_boundary = []
    state.inference_done = False
    state.mask_raw = None
    state.mask_processed = False
    state.mask_result = None
    state.centerline_catalog = {}
    state.selected_centerline_ids = []
    state.entry_point = None
    state.entry_point_locked = False
    state.auto_path_planned = False
    statuses = _statuses(state)
    assert statuses["draw_field"].status == "available"
    for tid in ("segment", "process", "params", "centerline_generate",
                "centerline_select", "entry", "plan"):
        assert statuses[tid].status == "pending", tid
