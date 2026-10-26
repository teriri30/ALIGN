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


def test_reference_right_click_undoes_only_the_last_point(tmp_path, monkeypatch):
    """右键只撤一个桩点（用户误触右键丢过 34 个点）。"""
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0), (80.0, 14.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    window._on_viewer_mouse_pressed(0.0, 0.0, 2)

    assert len(window._reference_pts) == 2
    assert window._reference_pts[-1] == (40.0, 12.0)
    import cache

    draft = cache.load_reference_draft(window.state.tif_path, 1)
    assert [list(map(float, p)) for p in draft["points"]] == [[10.0, 10.0], [40.0, 12.0]]
    window.close()


def test_right_click_undo_works_point_by_point_and_keeps_the_line(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0), (80.0, 14.0), (120.0, 16.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    for _ in range(2):
        window._on_viewer_mouse_pressed(0.0, 0.0, 2)

    assert [p[0] for p in window._reference_pts] == [10.0, 40.0]
    window.close()


def test_ctrl_right_click_discards_the_whole_pending_line(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    monkeypatch.setattr(window, "_reference_cancel_modifier_active", staticmethod(lambda: True))
    window._set_reference_mode(1)
    for x, y in ((10.0, 10.0), (40.0, 12.0), (80.0, 14.0)):
        window._on_viewer_mouse_pressed(x, y, 1)
    window._on_viewer_mouse_pressed(0.0, 0.0, 2)

    assert window._reference_pts == []
    import cache

    assert cache.load_reference_draft(window.state.tif_path, 1) == {}
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


def test_right_click_on_an_empty_line_exits_the_mode(tmp_path, monkeypatch):
    """空线状态下右键＝退出标注，不再需要另一条入口。"""
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    assert window._reference_drawing is True

    window._on_viewer_mouse_pressed(0.0, 0.0, 2)

    assert window._reference_drawing is False
    assert window.image_view.viewer.has_overlay_layer("interaction_reference") is False
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


def test_fit_records_the_worst_station_not_just_the_rms():
    """一个点跨到别的行时 RMS 会被平均掉，必须能报出"第几个点偏了多少"。"""
    import reference_lines

    points = [[0.0, 0.0], [100.0, 2.0], [200.0, 4.0], [300.0, 120.0], [400.0, 6.0]]
    fit = reference_lines.fit_line(points)

    assert fit["point_count"] == 5
    assert fit["max_deviation_station"] == 4
    assert fit["max_deviation_px"] > 50.0
    # 最小二乘会把线往离群点方向拉，所以最大垂距一定明显大于 RMS
    assert fit["residual_rms_px"] < fit["max_deviation_px"] * 0.7


def test_saved_reference_line_connects_the_stations_it_was_drawn_through(tmp_path, monkeypatch):
    """保存后必须按桩点连成折线：只画拟合直线就是用户看到的"点和线错位"。"""
    window = _window(tmp_path, monkeypatch)
    points = [[100.0, 100.0], [110.0, 200.0], [122.0, 300.0], [134.0, 400.0]]
    monkeypatch.setattr(
        window,
        "_reference_lines_payload",
        lambda: {
            "schema": "centerline-reference-lines-v1",
            "lines": [
                {
                    "id": "ref-0001",
                    "pass": 1,
                    "points": points,
                    "fit": {
                        "p1": [100.0, 100.0],
                        "p2": [120.0, 400.0],
                        "angle_rad": 1.5,
                    },
                }
            ],
        },
    )

    window._draw_reference_lines_overlay()

    viewer = window.image_view.viewer
    items = viewer._overlay_layers.get("reference_lines") or []
    # 3 段折线 + 4 个桩点 + 1 条拟合基准虚线
    assert len(items) == (len(points) - 1) + len(points) + 1
    window.close()


def test_finish_warns_when_one_station_sits_off_the_line(tmp_path, monkeypatch):
    """桩点跨行时保存后要提示，而不是默默画一条歪线。"""
    window = _window(tmp_path, monkeypatch)
    # 0.05 m/px 的显示分辨率（接近真实工程的降采样口径）
    monkeypatch.setattr(window, "_reference_px_to_m", lambda px, x, y: float(px) * 0.05)
    messages = []
    monkeypatch.setattr(
        window.log_panel, "warn", lambda text, *a, **k: messages.append(str(text)), raising=False
    )
    window._set_reference_mode(1)
    # 一条带里有一个桩点点偏了（转折 <40°，不会被拆段），但仍应提示偏差
    for x, y in ((0.0, 0.0), (100.0, 5.0), (200.0, 35.0), (300.0, 5.0), (400.0, 0.0)):
        window._on_viewer_mouse_pressed(x, y, 1)

    assert window._finish_reference_line() is True

    assert any("偏离基准线" in text for text in messages)
    window.close()


def test_split_points_at_turns_separates_two_rows_drawn_in_one_go():
    """两条行带连着画（中间直角过渡）必须拆成两段，否则拟合线整体歪掉。"""
    import reference_lines

    points = [[0.0, 0.0], [0.0, 100.0], [0.0, 200.0], [100.0, 200.0], [200.0, 200.0]]
    segments = reference_lines.split_points_at_turns(points)

    assert [len(segment) for segment in segments] == [2, 3]
    # 拐点归到后一段，点数不丢不重
    assert sum(len(segment) for segment in segments) == len(points)


def test_split_keeps_a_single_curved_row_as_one_line():
    import reference_lines

    curved = [[0.0, 0.0], [50.0, 2.0], [100.0, 5.0], [150.0, 9.0], [200.0, 14.0], [250.0, 20.0]]

    assert len(reference_lines.split_points_at_turns(curved)) == 1


def test_finishing_two_rows_in_one_go_saves_two_reference_lines(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    messages = []
    monkeypatch.setattr(
        window.log_panel, "warn", lambda text, *a, **k: messages.append(str(text)), raising=False
    )
    window._set_reference_mode(1)
    for x, y in ((100.0, 100.0), (100.0, 200.0), (100.0, 300.0), (200.0, 300.0), (300.0, 300.0)):
        window._on_viewer_mouse_pressed(x, y, 1)

    assert window._finish_reference_line() is True

    import cache

    payload = cache.load_reference_lines(window.state.tif_path)
    assert len(payload["lines"]) == 2
    assert sorted(len(line["points"]) for line in payload["lines"]) == [2, 3]
    assert any("拆成" in text for text in messages)
    window.close()
