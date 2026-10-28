# -*- coding: utf-8 -*-
"""参考线图层的显示与清除契约（DECISION-063）。

用户口径：人工打点的线不该常驻画布——上次画的这次打开还挂着、又删不掉。
这里钉住三件事：
1. 有参考线的项目恢复后，图层行可勾选但默认不显示；
2. 进入/退出参考线标注模式时自动显示/收起；
3. 删除最后一条 / 清除全部会真的删掉 sidecar，重开不会画回来。
全部离屏运行，不依赖真实影像。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow

#: 桩点间距必须大于双击判定窗（12 px）且第二条线离第一条 > 40 px，
#: 否则会被读成“双击收线”或“近似重合替换”。
LINE_A = ((10.0, 10.0), (40.0, 12.0), (80.0, 14.0))
LINE_B = ((10.0, 210.0), (40.0, 212.0), (80.0, 214.0))


def _window(tmp_path, monkeypatch):
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    window.state.tif_path = str(tmp_path / "field.tif")
    return window


def _draw_line(window, points):
    for x, y in points:
        window._on_viewer_mouse_pressed(x, y, 1)
    assert window._finish_reference_line() is True


def test_saved_reference_lines_are_hidden_until_the_user_asks(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    _draw_line(window, LINE_A)
    window.close()

    second = _window(tmp_path, monkeypatch)
    second._sync_display_layer_availability()
    second._draw_reference_lines_overlay()

    assert second._reference_lines_present() is True
    # 图层面板里这一行可勾选（available），但默认不勾（不显示）
    assert second.image_view.display_layer_visible("reference_lines", False) is False
    items = second.image_view.viewer._overlay_layers.get("reference_lines") or []
    assert items, "参考线仍画在独立图层里，只是不可见"
    assert all(item.isVisible() is False for item in items)
    second.close()


def test_entering_reference_mode_shows_the_layer_and_exit_hides_it(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    import cache
    import reference_lines

    payload = reference_lines.empty_payload(window.state.tif_path)
    reference_lines.upsert_line(
        payload,
        reference_lines.new_line_record([(10.0, 10.0), (80.0, 14.0)], pass_index=1),
    )
    cache.save_reference_lines(window.state.tif_path, payload)
    window._reference_payload = {}
    window._reference_payload_tif = ""

    window._sync_display_layer_availability()
    assert window.image_view.display_layer_visible("reference_lines", False) is False

    window._set_reference_mode(1)
    assert window.image_view.display_layer_visible("reference_lines", False) is True

    window._set_reference_mode(1)  # 再点一次同一条菜单＝退出
    assert window._reference_drawing is False
    assert window.image_view.display_layer_visible("reference_lines", False) is False
    window.close()


def test_toggle_menu_action_shows_and_hides_without_touching_data(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    _draw_line(window, LINE_A)
    window._set_reference_mode(1)

    tif = window.state.tif_path
    window._on_toggle_reference_layer()
    assert window.image_view.display_layer_visible("reference_lines", False) is True
    window._on_toggle_reference_layer()
    assert window.image_view.display_layer_visible("reference_lines", False) is False

    import cache

    assert len(cache.load_reference_lines(tif)["lines"]) == 1
    window.close()


def test_clear_reference_lines_deletes_sidecar_and_layer(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    _draw_line(window, LINE_A)
    import cache

    tif = window.state.tif_path
    assert cache.load_reference_lines(tif)["lines"]

    monkeypatch.setattr(
        "pyside6_app.main_window.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.Yes),
    )
    window._on_clear_reference_lines()

    assert cache.load_reference_lines(tif) == {}
    assert not os.path.isfile(cache.reference_lines_path(tif))
    assert window.image_view.viewer.has_overlay_layer("reference_lines") is False
    assert window.image_view.display_layer_visible("reference_lines", False) is False
    assert window._reference_lines_present() is False
    window.close()


def test_clear_reference_lines_respects_a_cancelled_confirmation(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    _draw_line(window, LINE_A)
    import cache

    tif = window.state.tif_path
    monkeypatch.setattr(
        "pyside6_app.main_window.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.No),
    )
    window._on_clear_reference_lines()

    assert len(cache.load_reference_lines(tif)["lines"]) == 1
    window.close()


def test_delete_last_reference_line_keeps_the_rest(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._set_reference_mode(1)
    _draw_line(window, LINE_A)
    _draw_line(window, LINE_B)
    import cache

    tif = window.state.tif_path
    assert len(cache.load_reference_lines(tif)["lines"]) == 2

    window._on_delete_last_reference_line()
    assert len(cache.load_reference_lines(tif)["lines"]) == 1

    window._on_delete_last_reference_line()
    assert cache.load_reference_lines(tif) == {}
    assert not os.path.isfile(cache.reference_lines_path(tif))
    window.close()
