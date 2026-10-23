"""菜单栏结构契约：每个功能只出现一次，归类稳定，内置测试影像可直达。"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

# shiboken 的坑：QAction.menu() 返回借用指针，临时的 QAction 包装器被回收后，
# 拿到的 QMenu 会变成 "already deleted"。这里统一把包装器钉住。
_KEEP = []


@pytest.fixture
def window():
    # 每个用例新建一个窗口：模块级 fixture 会被其它用例的 teardown 连带销毁，
    # QMenu 变成 "already deleted" 的悬空对象。
    app = QApplication.instance() or QApplication([])
    from pyside6_app.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    _KEEP.extend([win, app])
    return win


def _menus(win):
    acts = win.menuBar().actions()
    _KEEP.extend(acts)
    menus = [a.menu() for a in acts if a.menu() is not None]
    _KEEP.extend(menus)
    return menus


def _all_actions(menu, seen=None):
    seen = seen if seen is not None else []
    acts = menu.actions()
    _KEEP.extend(acts)
    for act in acts:
        sub = act.menu()
        if sub is not None:
            _KEEP.append(sub)
            _all_actions(sub, seen)
        elif act.text():
            seen.append(act)
    return seen


def test_top_level_menus_are_five_and_ordered(window):
    titles = [m.title() for m in _menus(window)]
    assert titles == ["文件", "视图", "工具", "设置", "帮助"]


def test_no_action_appears_twice(window):
    """旧版把设置/参数/模型管理同时挂在两个菜单下，重复项会互相抢快捷键。"""
    texts = [a.text() for m in _menus(window) for a in _all_actions(m)]
    dupes = {t for t in texts if texts.count(t) > 1}
    assert dupes == set(), f"重复菜单项: {sorted(dupes)}"


def test_no_duplicate_shortcuts(window):
    keys = []
    for m in _menus(window):
        for a in _all_actions(m):
            sc = a.shortcut().toString()
            if sc:
                keys.append(sc)
    dupes = {k for k in keys if keys.count(k) > 1}
    assert dupes == set(), f"重复快捷键: {sorted(dupes)}"


def test_export_items_are_grouped_under_one_submenu(window):
    fm = next(m for m in _menus(window) if m.title() == "文件")
    acts = fm.actions()
    _KEEP.extend(acts)
    export = None
    for act in acts:
        sub = act.menu()
        if sub is None:
            continue
        _KEEP.append(sub)
        if sub.title() == "导出":
            export = sub
    assert export is not None, "导出项必须收进「导出」子菜单"
    export_acts = export.actions()
    _KEEP.extend(export_acts)
    labels = [a.text() for a in export_acts if a.text()]
    # 文件菜单本身只保留打开/保存/加载/退出，导出全部下沉
    top_labels = [a.text() for a in acts if a.text()]
    assert "导出" in top_labels
    assert len(labels) >= 9
    assert not any(lbl.startswith("导出 ") for lbl in top_labels)


def test_quit_uses_ctrl_q_and_no_bare_letter_shortcuts(window):
    """单字母快捷键（旧版旋转是 "R"）会在输入框里误触，退出也不该是 [Exit]。"""
    all_acts = [a for m in _menus(window) for a in _all_actions(m)]
    shortcuts = [a.shortcut().toString() for a in all_acts if a.shortcut().toString()]
    bare = [s for s in shortcuts if len(s) == 1 and s.isalpha()]
    assert bare == [], f"不应用单字母快捷键: {bare}"
    quit_act = next(a for a in all_acts if a.text() == "退出")
    assert quit_act.shortcut().toString() == "Ctrl+Q"


def test_settings_live_in_one_place(window):
    sm = next(m for m in _menus(window) if m.title() == "设置")
    labels = [a.text() for a in sm.actions() if a.text()]
    for expected in ("系统设置…", "农机参数…", "模型管理…", "项目日志…", "系统诊断…", "打开数据目录"):
        assert expected in labels


def test_help_menu_has_manual_and_about(window):
    hm = next(m for m in _menus(window) if m.title() == "帮助")
    labels = [a.text() for a in hm.actions() if a.text()]
    assert "使用说明" in labels and "关于 ALIGN" in labels


def test_builtin_test_image_action_exists_and_loads_file(window, monkeypatch):
    from pyside6_app import main_window as mw

    fm = next(m for m in _menus(window) if m.title() == "文件")
    act = next(a for a in fm.actions() if a.text() == "打开内置测试影像")
    _KEEP.append(act)

    loaded = {}
    monkeypatch.setattr(window, "_load_tif_path", lambda p: loaded.setdefault("path", p), raising=False)
    monkeypatch.setattr(mw, "builtin_test_image", lambda: r"E:\fake\builtin.tif", raising=False)
    act.trigger()
    assert loaded, "点击菜单项后应载入内置测试影像"
