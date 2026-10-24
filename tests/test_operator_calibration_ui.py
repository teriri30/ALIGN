# -*- coding: utf-8 -*-
"""作业线标定（人工纠偏）的桌面端交互契约（DECISION-060）。

真跑一遍信号链：进入标定模式 → 点在线上 → 拖动 → 松手保存 sidecar →
候选目录里的线换成人工线 → 规划配置里带上 operator_centerlines。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow


class _FakeGeo:
    def is_ready(self):
        return True

    def meters_per_pixel(self, x=0.0, y=0.0):
        return 0.01

    def pixel_to_lonlat(self, x, y):
        return (0.0, 0.0)


def _window(tmp_path, monkeypatch):
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.geo = _FakeGeo()
    window._tif_rgb = np.zeros((400, 1200, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    window._tif_path = str(tmp_path / "field.tif")
    window.state.tif_path = str(tmp_path / "field.tif")
    window.state.mask_offset_x = 0
    window.state.mask_offset_y = 0
    window.state.mask_result = {
        "all_bands": [{
            "id": 7,
            "band_type": "wide",
            "width_m": 1.0,
            "centerline": [(215.0, y) for y in np.linspace(80, 320, 40)],
        }]
    }
    window.state.centerline_catalog = {
        "items": [{
            "id": "centerline-001",
            "legacy_id": "work-band-001",
            "source_band_ids": [7],
            "sequence": 1,
            "label": "中心线 1",
            "kind": "wide",
            "centerline": [(215.0, y) for y in np.linspace(80, 320, 40)],
            "work_line": [(215.0, y) for y in np.linspace(80, 320, 40)],
        }]
    }
    window.state.selected_centerline_ids = ["centerline-001"]
    return window, app


def test_calibration_requires_project_and_catalog(tmp_path, monkeypatch):
    window, _app = _window(tmp_path, monkeypatch)
    window.state.centerline_catalog = {}
    window._set_calibration_mode(True)
    assert window._calibration_drawing is False
    window.state.tif_path = ""
    window._set_calibration_mode(True)
    assert window._calibration_drawing is False
    window.close()


def test_calibration_refuses_to_start_while_route_editing(tmp_path, monkeypatch):
    window, _app = _window(tmp_path, monkeypatch)
    window._route_edit_active = True
    window._set_calibration_mode(True)
    assert window._calibration_drawing is False
    window._route_edit_active = False
    window.close()


def test_drag_whole_line_is_saved_and_replaces_the_catalog_geometry(tmp_path, monkeypatch):
    window, _app = _window(tmp_path, monkeypatch)
    window._set_calibration_mode(True)
    assert window._calibration_drawing is True

    # 点在线上 → 向右拖 30 px（法向 -x，所以这是 -0.30 m 的均匀偏移）
    window._on_viewer_mouse_pressed(215.0, 200.0, 1)
    assert window._calibration_item_id == "centerline-001"
    window._on_viewer_mouse_moved(245.0, 200.0, 1)
    window._on_viewer_mouse_released(245.0, 200.0, 1)

    import cache
    import operator_paths

    payload = cache.load_operator_paths(window.state.tif_path)
    assert payload["schema"] == operator_paths.SCHEMA
    assert len(payload["lines"]) == 1
    record = payload["lines"][0]
    assert record["centerline_id"] == "centerline-001"
    assert record["edits"][0]["kind"] == "uniform"
    assert record["edits"][0]["offset_m"] == pytest.approx(-0.30, abs=0.02)
    assert record["base_points"]

    item = window.state.centerline_catalog["items"][0]
    assert item["line_source"] == "operator"
    xs = [point[0] for point in item["centerline"]]
    assert min(xs) == pytest.approx(245.0, abs=1.5)

    # 规划配置里必须带上替换表，否则导出仍是算法线
    overrides = window._operator_overrides()
    assert overrides["centerline-001"]["points"]

    # 交互日志要留痕（含没保存项目的情况）
    import interaction_log

    actions = [item["action"] for item in interaction_log.action_log().recent(50)]
    assert "calibration.edit" in actions
    window.close()


def test_right_click_undoes_the_last_drag(tmp_path, monkeypatch):
    window, _app = _window(tmp_path, monkeypatch)
    window._set_calibration_mode(True)
    original = [tuple(point) for point in window.state.centerline_catalog["items"][0]["centerline"]]

    window._on_viewer_mouse_pressed(215.0, 200.0, 1)
    window._on_viewer_mouse_moved(245.0, 200.0, 1)
    window._on_viewer_mouse_released(245.0, 200.0, 1)
    window._on_viewer_mouse_pressed(245.0, 200.0, 2)  # 右键撤销

    item = window.state.centerline_catalog["items"][0]
    assert [tuple(point) for point in item["centerline"]] == original
    import cache

    payload = cache.load_operator_paths(window.state.tif_path)
    assert payload["lines"] == [], "撤销到空必须同时删掉 sidecar，否则重规划还是旧人线"
    window.close()


def test_undo_with_remaining_edits_rewrites_the_sidecar(tmp_path, monkeypatch):
    """撤销一次拖动后，磁盘上的记录必须与画布一致（不能只改内存）。"""
    window, _app = _window(tmp_path, monkeypatch)
    window._set_calibration_mode(True)
    # 第一次：向右 30 px（-0.30 m）；第二次：再向右 10 px（累计 -0.40 m）
    window._on_viewer_mouse_pressed(215.0, 200.0, 1)
    window._on_viewer_mouse_moved(245.0, 200.0, 1)
    window._on_viewer_mouse_released(245.0, 200.0, 1)
    window._on_viewer_mouse_pressed(245.0, 200.0, 1)
    window._on_viewer_mouse_moved(255.0, 200.0, 1)
    window._on_viewer_mouse_released(255.0, 200.0, 1)

    import cache

    before = cache.load_operator_paths(window.state.tif_path)["lines"][0]
    assert len(before["edits"]) == 2
    assert before["edits"][0]["offset_m"] == pytest.approx(-0.30, abs=0.02)

    window._on_viewer_mouse_pressed(255.0, 200.0, 2)  # 撤销第二次
    after = cache.load_operator_paths(window.state.tif_path)["lines"][0]
    assert len(after["edits"]) == 1
    assert after["edits"][0]["offset_m"] == pytest.approx(-0.30, abs=0.02)
    xs = [point[0] for point in after["points"]]
    assert min(xs) == pytest.approx(245.0, abs=1.5)
    window.close()


def test_clicking_empty_space_does_not_change_anything(tmp_path, monkeypatch):
    window, _app = _window(tmp_path, monkeypatch)
    window._set_calibration_mode(True)
    window._on_viewer_mouse_pressed(20.0, 20.0, 1)
    assert window._calibration_item_id == ""
    window._on_viewer_mouse_released(60.0, 60.0, 1)
    import cache

    assert cache.load_operator_paths(window.state.tif_path) == {}
    window.close()


def test_calibrated_line_is_drawn_on_the_canvas_from_the_sidecar(tmp_path, monkeypatch):
    """重启后画布必须画人线：画布与导出同源，不能一个算法线一个人线。"""
    window, _app = _window(tmp_path, monkeypatch)
    window._set_calibration_mode(True)
    window._on_viewer_mouse_pressed(215.0, 200.0, 1)
    window._on_viewer_mouse_moved(245.0, 200.0, 1)
    window._on_viewer_mouse_released(245.0, 200.0, 1)
    window.close()

    # 新窗口 = 重启应用：只剩磁盘上的 sidecar，目录仍是算法线
    second, _app = _window(tmp_path, monkeypatch)
    viewer = second.image_view.viewer
    captured = []
    monkeypatch.setattr(
        viewer,
        "draw_polyline",
        lambda points, **kwargs: captured.append((points, kwargs)),
        raising=False,
    )
    second._draw_centerline_overlay(viewer)
    assert captured, "应至少画出一条作业线"
    points, kwargs = captured[-1]
    from pyside6_app.main_window import OPERATOR_LINE_COLOR_BGR

    assert kwargs.get("color") == OPERATOR_LINE_COLOR_BGR
    assert min(point[0] for point in points) == pytest.approx(245.0, abs=1.5)
    second.close()
