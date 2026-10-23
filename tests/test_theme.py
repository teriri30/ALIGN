import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ALIGN_DISABLE_AUTO_START", "1")

from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow
from pyside6_app.styles import THEMES, ThemeMode


def _window():
    app = QApplication.instance() or QApplication([])
    return MainWindow()


def test_toolbar_theme_toggle_returns_from_light_to_dark(monkeypatch):
    monkeypatch.setitem(sys.modules, "darkdetect", types.SimpleNamespace(isDark=lambda: False))
    window = _window()
    try:
        window._set_theme_mode(ThemeMode.LIGHT)

        window._toggle_theme()

        assert window._theme_mgr.current_mode() == ThemeMode.DARK
        assert window._theme_mgr.current_theme() == "dark"
    finally:
        window._theme_mgr.set_mode(ThemeMode.DARK)
        window.close()


def test_theme_refresh_updates_route_and_log_surfaces():
    window = _window()
    try:
        for mode, name in ((ThemeMode.LIGHT, "light"), (ThemeMode.DARK, "dark")):
            window._set_theme_mode(mode)
            colors = THEMES[name]

            assert colors["bg"] in window.route_info.styleSheet()
            assert colors["bg"] in window.route_info._list_container.styleSheet()
            assert colors["bg_darkest"] in window.log_panel.styleSheet()
    finally:
        window._theme_mgr.set_mode(ThemeMode.DARK)
        window.close()


def test_initial_theme_surfaces_match_current_manager_theme():
    window = _window()
    try:
        colors = THEMES[window._theme_mgr.current_theme()]

        assert colors["bg"] in window.route_info._list_container.styleSheet()
        assert colors["bg_darkest"] in window.log_panel.styleSheet()
    finally:
        window.close()


# ------------------------------------------------------- QSS 语法：注释别吞规则


def test_stylesheet_has_no_hash_comments():
    """QSS 只认 C 风格块注释；用 # 起的"注释"会被当成 ID 选择器，把下一条规则吞掉。"""
    from pyside6_app.styles import build_stylesheet

    css = build_stylesheet()
    offenders = [line for line in css.splitlines() if line.strip().startswith("#")]
    assert offenders == [], f"QSS 里出现 # 注释: {offenders[:3]}"


def test_stylesheet_block_comments_are_not_nested():
    """注释正文里不能再写块注释结束符，否则注释提前闭合，后面全成非法 CSS。"""
    from pyside6_app.styles import build_stylesheet

    css = build_stylesheet()
    depth = 0
    i = 0
    while i < len(css):
        if css.startswith("/*", i):
            depth += 1
            assert depth == 1, "出现嵌套块注释"
            i += 2
        elif css.startswith("*/", i):
            depth -= 1
            assert depth == 0
            i += 2
        else:
            i += 1
    assert depth == 0


def test_menu_bar_actually_gets_the_dark_background():
    """回归：菜单栏曾因 QSS 被吞而一直是系统浅色 #efefef。"""
    from PySide6.QtWidgets import QApplication, QMainWindow

    from pyside6_app.styles import build_stylesheet

    app = QApplication.instance() or QApplication([])
    win = QMainWindow()
    win.setStyleSheet(build_stylesheet())
    win.menuBar().addMenu("文件")
    win.resize(320, 120)
    win.show()
    QApplication.processEvents()
    text = win.grab().toImage().pixelColor(160, 3)
    win.close()
    assert text.lightness() < 80, f"菜单栏底色仍是浅色 {text.name()}"


def test_packaged_icon_is_shipped_for_window_icon():
    """任务栏图标靠运行时读 align.ico；spec 不打包就会退回 Qt 默认图标。"""
    from pathlib import Path

    spec = Path(__file__).resolve().parents[1] / "packaging" / "align.spec"
    text = spec.read_text(encoding="utf-8")
    assert "align.ico" in text
    assert (Path(__file__).resolve().parents[1] / "packaging" / "align.ico").is_file()
