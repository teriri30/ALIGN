"""左边任务清单与右边画布必须由同一份状态驱动（用户反复反馈的"左右不同步"）。

2026-10-26 实际反馈：缓存里 ``inference_done=True`` 而 ``mask_processed=False`` 时，
恢复流程先按状态画好识别掩膜与田块边界，随后的一致性兜底又用
``_clear_display_overlay_layers()`` 把这两层一起擦掉，结果是
「左边已经走到第 4 步掩膜优化、右边画布一片空白」。
"""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    window._tif_path = "dummy.tif"
    window.state.tif_path = "dummy.tif"
    return window


def _payload(*, processed_valid, mask_processed):
    return {
        "path": "dummy.tif",
        "saved": {
            "inference_done": True,
            "mask_processed": mask_processed,
            "field_boundary": [[0.0, 0.0], [15.0, 0.0], [15.0, 15.0], [0.0, 15.0]],
        },
        "processed_cache_valid": processed_valid,
        "raw_mask": np.full((8, 8), 255, dtype=np.uint8),
        "raw_ox": 0,
        "raw_oy": 0,
        "processed_mask": None,
        "processed_layers": {},
        "path_cache_valid": False,
    }


def test_restore_without_processed_mask_still_draws_field_and_recognition(monkeypatch):
    window = _window()
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)

    window._restore_cache_payload(_payload(processed_valid=False, mask_processed=False))

    viewer = window.image_view.viewer
    # 状态里"作物识别已完成、田块已圈选"，画布就必须画出这两层证据。
    assert window.state.inference_done is True
    assert window.state.mask_processed is False
    assert viewer.has_overlay_layer("field_boundary")
    assert viewer.has_overlay_layer("raw_mask")
    assert not viewer.has_overlay_layer("processed_mask")
    assert not viewer.has_overlay_layer("centerlines")
    assert window.canvas_state_mismatches() == []
    window.close()


def test_restore_without_processed_mask_advances_task_list_to_mask_step(monkeypatch):
    """左清单与画布讲的是同一件事：识别已完成、掩膜优化可执行。"""
    window = _window()
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)

    window._restore_cache_payload(_payload(processed_valid=False, mask_processed=False))

    rows = window.task_panel._task_widgets
    assert rows["draw_field"]._done is True
    assert rows["segment"]._done is True
    assert rows["process"]._done is False
    assert rows["process"]._action_btn.isEnabled()
    window.close()


def test_invalidating_inference_removes_the_recognition_layer(monkeypatch):
    """反向也要同步：状态里掩膜失效了，画布上就不许还留着识别掩膜。"""
    window = _window()
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)
    window._restore_cache_payload(_payload(processed_valid=False, mask_processed=False))
    viewer = window.image_view.viewer
    assert viewer.has_overlay_layer("raw_mask")

    window._invalidate_analysis_from("inference", "重跑识别")

    assert window.state.mask_raw is None
    assert not viewer.has_overlay_layer("raw_mask")
    # 田块边界属于上游证据层，不该被下游失效连带擦掉
    assert viewer.has_overlay_layer("field_boundary")
    assert window.canvas_state_mismatches() == []
    window.close()


def test_mismatch_detector_reports_a_canvas_that_lost_its_layers(monkeypatch):
    """检测器不能是空转的：手工擦掉图层后必须能报出来。"""
    window = _window()
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)
    window._restore_cache_payload(_payload(processed_valid=False, mask_processed=False))

    window._clear_display_overlay_layers()

    problems = window.canvas_state_mismatches()
    assert any("田块边界" in item for item in problems)
    assert any("掩膜" in item for item in problems)
    window.close()


def test_repair_redraws_canvas_from_state_and_keeps_the_view(monkeypatch):
    window = _window()
    monkeypatch.setattr(window, "_sync_cache_model_identity", lambda: True)
    window._restore_cache_payload(_payload(processed_valid=False, mask_processed=False))
    viewer = window.image_view.viewer
    before = viewer.capture_view_state()
    window._clear_display_overlay_layers()

    repaired = window._repair_canvas_if_state_disagrees(where="test")

    assert repaired is True
    assert viewer.has_overlay_layer("field_boundary")
    assert viewer.has_overlay_layer("raw_mask")
    assert window.canvas_state_mismatches() == []
    after = viewer.capture_view_state()
    assert after["transform"].m11() == pytest.approx(before["transform"].m11())
    # 重绘不能把用户看的位置跳走（亚像素级的边界补齐误差允许）
    assert abs(after["center"].x() - before["center"].x()) < 1.0
    assert abs(after["center"].y() - before["center"].y()) < 1.0
    window.close()
