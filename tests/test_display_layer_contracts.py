# -*- coding: utf-8 -*-
"""显示图层默认可见性 + 标注 sidecar 删除入口的契约（DECISION-064）。

用户口径：保存过的标注不该常驻画布，也不能只能靠手工删文件才清得掉。
这里钉住五件事：
1. 每个 display layer 都必须在可用性表里有默认值规则（新增图层漏登记就红）；
2. 登记为“标注证据”的图层，有数据也只亮行、默认不画；
3. 标注 sidecar 的 save 都有配对的 delete，且 delete 真的删文件；
4. 作业线标定有批量清除入口，删空时不留空 sidecar；
5. 标注模式的进出不写用户偏好，“恢复图层默认显示”有入口。
全部离屏运行，不依赖真实影像。
"""
import ast
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from pyside6_app.image_view import DISPLAY_LAYER_KEYS
from pyside6_app import main_window as main_window_module
from pyside6_app.main_window import PERSISTED_ANNOTATION_LAYERS, MainWindow

LINE_A = ((10.0, 10.0), (40.0, 12.0), (80.0, 14.0))


def _window(tmp_path, monkeypatch):
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    window.state.tif_path = str(tmp_path / "field.tif")
    return window


def _save_reference_line(window):
    window._set_reference_mode(1)
    for x, y in LINE_A:
        window._on_viewer_mouse_pressed(x, y, 1)
    assert window._finish_reference_line() is True


def _operator_payload(tif, centerline_ids):
    import operator_paths

    payload = operator_paths.empty_payload(tif, {"mask_version": 1})
    for index, centerline_id in enumerate(centerline_ids):
        record = operator_paths.new_record(
            [(0.0, float(index)), (20.0, float(index) + 1.0)],
            centerline_id=centerline_id,
            band_label=centerline_id,
            width_m=1.0,
        )
        operator_paths.upsert_record(payload, record)
    return payload


# ---------------------------------------------------------------- 1. 默认值规则
def test_every_display_layer_has_a_default_rule():
    """可用性表必须覆盖 DISPLAY_LAYER_KEYS 全集，多一个少一个都算错。"""
    source = open(
        os.path.join(os.path.dirname(main_window_module.__file__), "main_window.py"),
        "r",
        encoding="utf-8",
    ).read()
    tree = ast.parse(source)
    keys = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_sync_display_layer_availability":
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Assign):
                continue
            if not inner.targets:
                continue
            target = inner.targets[0]
            if not isinstance(target, ast.Name) or target.id != "availability":
                continue
            if not isinstance(inner.value, ast.Dict):
                continue
            keys = [
                key.value
                for key in inner.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            break
    assert keys is not None, "可用性表 availability = {...} 不见了，默认值规则无从核对"
    assert set(keys) == set(DISPLAY_LAYER_KEYS), (
        "图层键与默认值规则不一致：缺少 %s，多出 %s"
        % (sorted(set(DISPLAY_LAYER_KEYS) - set(keys)), sorted(set(keys) - set(DISPLAY_LAYER_KEYS)))
    )


def test_annotation_layer_registry_only_holds_real_layers():
    assert PERSISTED_ANNOTATION_LAYERS, "标注图层登记表不该为空，参考线必须在内"
    assert set(PERSISTED_ANNOTATION_LAYERS) <= set(DISPLAY_LAYER_KEYS)
    assert "reference_lines" in PERSISTED_ANNOTATION_LAYERS


# ---------------------------------------------------------------- 2. 默认收起
def test_reference_lines_stay_hidden_even_when_data_exists(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    _save_reference_line(window)
    window._set_reference_mode(1)  # 退出标注
    window.close()

    # 重新打开项目：sidecar 还在，但画布必须是干净的
    second = _window(tmp_path, monkeypatch)
    second._sync_display_layer_availability()
    assert second._reference_lines_present() is True
    for key in PERSISTED_ANNOTATION_LAYERS:
        assert second.image_view._display_layer_defaults[key] is False, (
            "%s 是持久化标注证据，有数据也必须默认收起" % key
        )
        assert second.image_view.display_layer_visible(key, False) is False
    second.close()


# ---------------------------------------------------------------- 3. sidecar 配对
def test_operator_paths_sidecar_has_a_delete_pair(tmp_path, monkeypatch):
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    tif = str(tmp_path / "field.tif")
    path = cache.save_operator_paths(tif, _operator_payload(tif, ["c1", "c2"]))
    assert os.path.isfile(path)

    assert cache.delete_operator_paths(tif) is True
    assert not os.path.isfile(path)
    assert cache.load_operator_paths(tif) == {}
    assert cache.delete_operator_paths(tif) is False


def test_every_annotation_sidecar_has_a_delete_function():
    import cache

    for name in ("operator_paths", "reference_lines"):
        assert hasattr(cache, "save_" + name)
        assert hasattr(cache, "delete_" + name), (
            "%s 有 save 没有 delete：界面上“清除”之后重开还会读回旧内容" % name
        )


# ---------------------------------------------------------------- 4. 批量清除
def test_clear_all_operator_paths_deletes_the_sidecar(tmp_path, monkeypatch):
    import cache

    window = _window(tmp_path, monkeypatch)
    tif = window.state.tif_path
    payload = _operator_payload(tif, ["c1", "c2"])
    cache.save_operator_paths(tif, payload)
    window._calibration_payload = payload
    window._calibration_payload_tif = tif
    window._calibration_item_id = "c1"

    monkeypatch.setattr(
        "pyside6_app.main_window.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.Yes),
    )
    window._on_clear_operator_paths()

    assert not os.path.isfile(cache.operator_paths_path(tif))
    assert window._operator_paths_payload() == {}
    assert window._calibration_item_id == ""
    window.close()


def test_clear_all_operator_paths_respects_a_cancelled_confirmation(tmp_path, monkeypatch):
    import cache

    window = _window(tmp_path, monkeypatch)
    tif = window.state.tif_path
    payload = _operator_payload(tif, ["c1"])
    cache.save_operator_paths(tif, payload)
    window._calibration_payload = payload
    window._calibration_payload_tif = tif

    monkeypatch.setattr(
        "pyside6_app.main_window.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.No),
    )
    window._on_clear_operator_paths()

    assert os.path.isfile(cache.operator_paths_path(tif))
    window.close()


def test_delete_last_calibration_record_leaves_no_stale_record(tmp_path, monkeypatch):
    """逐条删到最后一条时，磁盘上不能再有人线记录（空 payload 与原语义一致）。"""
    import cache

    window = _window(tmp_path, monkeypatch)
    tif = window.state.tif_path
    window.state.centerline_catalog = {
        "items": [
            {
                "id": "c1",
                "centerline": [(0.0, 0.0), (20.0, 1.0)],
                "work_line": [(0.0, 0.0), (20.0, 1.0)],
            }
        ]
    }
    payload = _operator_payload(tif, ["c1"])
    cache.save_operator_paths(tif, payload)
    window._calibration_payload = payload
    window._calibration_payload_tif = tif

    assert window._delete_calibration_record("c1") is True
    assert cache.load_operator_paths(tif).get("lines") == []
    window.close()


# ---------------------------------------------------------------- 5. 显示语义
def test_mode_driven_visibility_keeps_the_user_preference_clean(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    _save_reference_line(window)
    window._set_reference_mode(1)  # 退出标注

    assert window.image_view.display_layer_user_overrides() == set()
    assert window.image_view.display_layer_visible("reference_lines", False) is False
    window.close()


def test_explicit_user_choice_survives_a_mode_round_trip(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    _save_reference_line(window)
    window._set_reference_mode(1)  # 退出标注

    window._on_toggle_reference_layer()  # 用户手动打开
    assert window.image_view.display_layer_visible("reference_lines", False) is True

    window._set_reference_mode(1)  # 进标注
    window._set_reference_mode(1)  # 退出标注
    assert window.image_view.display_layer_visible("reference_lines", False) is True, (
        "用户显式打开的图层，进出标注模式后应当仍然打开"
    )
    window.close()


def test_reset_display_layer_defaults_hides_annotation_layer(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    _save_reference_line(window)
    window._set_reference_mode(1)  # 退出标注
    window._on_toggle_reference_layer()
    assert window.image_view.display_layer_visible("reference_lines", False) is True

    window._on_reset_display_layer_defaults()
    assert window.image_view.display_layer_visible("reference_lines", False) is False
    assert window.image_view.display_layer_user_overrides() == set()
    window.close()
