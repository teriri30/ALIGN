# -*- coding: utf-8 -*-
"""参考线标注（DECISION-028）桌面端交互测试。

覆盖：模式进入/退出与互斥、点击加桩点、双击收线保存 sidecar、右键取消、
参考线图层绘制、第 2 遍并存（重复性）。全部离屏运行，不依赖真实影像。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow


def _window(tmp_path, monkeypatch):
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    window.state.tif_path = str(tmp_path / "field.tif")
    return window


def test_reference_mode_toggles_and_disables_field_mode(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._field_drawing = True
    window._set_reference_mode(1)
    assert window._reference_drawing is True
    assert window._field_drawing is False
    assert window.image_view.viewer.drawing_mode is True
    window._set_reference_mode(1)
    assert window._reference_drawing is False
    assert window.image_view.viewer.drawing_mode is False
    window.close()


def test_reference_mode_requires_project(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window.state.tif_path = ""
    window._set_reference_mode(1)
    assert window._reference_drawing is False
    window.close()


def test_reference_clicks_and_finish_saves_sidecar(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0), (80.0, 14.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    assert len(window._reference_pts) == 3
    ok = window._finish_reference_line()
    assert ok is True

    import cache
    import reference_lines

    payload = cache.load_reference_lines(window.state.tif_path)
    assert payload["schema"] == reference_lines.SCHEMA
    assert len(payload["lines"]) == 1
    line = payload["lines"][0]
    assert len(line["points"]) == 3
    assert line["pass"] == 1
    viewer = window.image_view.viewer
    assert viewer.has_overlay_layer("reference_lines") is True
    window.close()


def test_reference_double_click_finishes_without_extra_point(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    window._on_viewer_mouse_pressed(5.0, 5.0, 1)
    window._on_viewer_mouse_pressed(50.0, 6.0, 1)
    # 双击第二下（时间窗内、位置相近）→ 收线，不新增桩点
    window._on_viewer_mouse_pressed(50.0, 6.0, 1)
    assert window._reference_pts == []
    import cache

    payload = cache.load_reference_lines(window.state.tif_path)
    assert len(payload["lines"]) == 1
    assert len(payload["lines"][0]["points"]) == 2
    window.close()


def test_reference_right_click_cancels_current_line(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    window._on_viewer_mouse_pressed(5.0, 5.0, 1)
    window._on_viewer_mouse_pressed(0.0, 0.0, 2)
    assert window._reference_pts == []
    assert window.image_view.viewer.has_overlay_layer("interaction_reference") is False
    window.close()


def test_reference_second_pass_stored_separately(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (80.0, 14.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    assert window._finish_reference_line() is True
    window._set_reference_mode(2)
    for x, y in ((10.0, 11.0), (80.0, 15.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    assert window._finish_reference_line() is True
    import cache

    payload = cache.load_reference_lines(window.state.tif_path)
    passes = sorted(int(line["pass"]) for line in payload["lines"])
    assert passes == [1, 2]
    window.close()


def test_toggling_reference_mode_off_saves_pending_line(tmp_path, monkeypatch):
    """用户画完点直接再点菜单退出：必须自动保存，不能把线丢掉。"""
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0), (80.0, 14.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    window._set_reference_mode(1)  # 再点一次“第 1 遍”＝退出

    assert window._reference_drawing is False
    import cache

    payload = cache.load_reference_lines(window.state.tif_path)
    assert len(payload["lines"]) == 1
    assert len(payload["lines"][0]["points"]) == 3
    assert cache.load_reference_draft(window.state.tif_path, 1) == {}

    import interaction_log

    actions = [
        item["action"]
        for item in interaction_log.action_log().recent(80, action_prefix="reference.")
    ]
    assert "reference.point" in actions
    assert "reference.finish" in actions
    assert "reference.exit" in actions
    window.close()


def test_pending_reference_points_survive_a_restart(tmp_path, monkeypatch):
    """每个桩点都落盘：关掉软件再打开，未完成的线还在。"""
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    window.close()

    second = _window(tmp_path, monkeypatch)
    second._set_reference_mode(1)
    assert len(second._reference_pts) == 2
    assert second._reference_pts[0] == (10.0, 10.0)
    second.close()


def test_single_point_is_not_saved_and_draft_is_cleared(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    window._on_viewer_mouse_pressed(10.0, 10.0, 1)
    window._set_reference_mode(1)
    import cache

    assert cache.load_reference_lines(window.state.tif_path) == {}
    assert cache.load_reference_draft(window.state.tif_path, 1) == {}
    window.close()


def test_right_click_cancel_clears_the_draft(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    window._on_viewer_mouse_pressed(10.0, 10.0, 1)
    window._on_viewer_mouse_pressed(40.0, 12.0, 1)
    window._on_viewer_mouse_pressed(0.0, 0.0, 2)
    import cache

    assert cache.load_reference_draft(window.state.tif_path, 1) == {}
    window.close()


def test_finish_reference_line_menu_action_saves_without_double_click(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0), (80.0, 14.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    window._on_finish_reference_line()
    import cache

    payload = cache.load_reference_lines(window.state.tif_path)
    assert len(payload["lines"]) == 1
    assert len(payload["lines"][0]["points"]) == 3
    window.close()


def test_switching_to_second_pass_saves_pending_first_pass(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    window._set_reference_mode(2)  # 直接切第 2 遍，不能丢第 1 遍的桩点
    assert window._reference_pass == 2
    assert window._reference_pts == []
    import cache

    payload = cache.load_reference_lines(window.state.tif_path)
    assert len(payload["lines"]) == 1
    assert payload["lines"][0]["pass"] == 1
    window.close()


def test_entering_calibration_saves_pending_reference(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    window.state.centerline_catalog = {
        "items": [{
            "id": "centerline-001", "legacy_id": "work-band-001",
            "source_band_ids": [7], "sequence": 1, "label": "中心线 1",
            "centerline": [(200.0, 100.0), (200.0, 300.0)],
            "work_line": [(200.0, 100.0), (200.0, 300.0)],
        }]
    }
    window._set_calibration_mode(True)
    assert window._calibration_drawing is True
    import cache

    payload = cache.load_reference_lines(window.state.tif_path)
    assert len(payload["lines"]) == 1
    window.close()
