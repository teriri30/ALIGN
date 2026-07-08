import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")

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
