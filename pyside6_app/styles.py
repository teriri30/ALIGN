
"""
styles.py — 主题系统（深色/浅色/随系统）
"""
from enum import Enum
from PySide6.QtCore import Qt, QObject, Signal

class ThemeMode(Enum):
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"

THEMES = {
    "dark": {
        "bg_darkest": "#1A1A1F", "bg_dark": "#1F1F26", "bg": "#25252D",
        "bg_light": "#2D2D38", "bg_hover": "#3A3A48", "bg_card": "#2A2A35",
        "bg_accent": "#203040", "border": "#383848", "border_light": "#48485A",
        "text": "#E8E8F0", "text_dim": "#90909E", "text_dimmer": "#5E5E6E",
        "text_bright": "#FFFFFF", "text_accent": "#60CDFF",
        "accent": "#60CDFF", "accent_hover": "#7FD8FF", "accent_pressed": "#3AAEE0",
        "accent_bg": "#1A3A4A",
        "green": "#6CCB6E", "green_bg": "#1A3A1A", "red": "#F17070",
        "red_bg": "#3A1A1A", "orange": "#F0A050", "orange_bg": "#3A2A1A",
        "yellow": "#E8D050", "purple": "#B080E0", "teal": "#50D0C0",
        "btn_default": "#333340", "btn_default_hover": "#404050",
        "btn_disabled": "#282835", "btn_disabled_text": "#5E5E6E",
        "scrollbar": "#383848", "scrollbar_hover": "#505060",
        "success_bg": "#1A3A1A", "pending_bg": "#3A2A1A",
    },
    "light": {
        "bg_darkest": "#E8E8ED", "bg_dark": "#F0F0F5", "bg": "#F5F5FA",
        "bg_light": "#FFFFFF", "bg_hover": "#E0E0E8", "bg_card": "#FFFFFF",
        "bg_accent": "#E8F0FA", "border": "#C8C8D0", "border_light": "#D8D8E0",
        "text": "#1A1A2E", "text_dim": "#606070", "text_dimmer": "#909098",
        "text_bright": "#000000", "text_accent": "#0078D4",
        "accent": "#0078D4", "accent_hover": "#106EBE", "accent_pressed": "#005A9E",
        "accent_bg": "#DEEBFA",
        "green": "#107C10", "green_bg": "#DFF6DD", "red": "#D13438",
        "red_bg": "#FDE7E9", "orange": "#D05A00", "orange_bg": "#FFF3E8",
        "yellow": "#8A6D00", "purple": "#744DA9", "teal": "#038387",
        "btn_default": "#FFFFFF", "btn_default_hover": "#E0E0E8",
        "btn_disabled": "#E0E0E0", "btn_disabled_text": "#A0A0A8",
        "scrollbar": "#C8C8D0", "scrollbar_hover": "#A0A0A8",
        "success_bg": "#DFF6DD", "pending_bg": "#FFF3E8",
    },
}

class ThemeManager(QObject):
    theme_changed = Signal(str)
    _instance = None

    def __init__(self):
        super().__init__()
        self._mode = ThemeMode.DARK
        self._current = "dark"

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ThemeManager()
        return cls._instance

    def set_mode(self, mode: ThemeMode):
        self._mode = mode
        if mode == ThemeMode.DARK:
            self._current = "dark"
        elif mode == ThemeMode.LIGHT:
            self._current = "light"
        else:
            try:
                import darkdetect
                self._current = "dark" if darkdetect.isDark() else "light"
            except ImportError:
                self._current = "light"
        self.theme_changed.emit(self._current)

    def current_theme(self):
        return self._current

    def current_mode(self):
        return self._mode

    def colors(self):
        return THEMES[self._current]

_theme = ThemeManager.instance()
COLORS = dict(_theme.colors())

def _c(n, colors=None):
    c = colors if colors is not None else COLORS
    return c.get(n, "#FFFFFF")

def build_stylesheet(colors=None):
    """动态生成全局样式表 — 根据当前主题颜色"""
    c = colors if colors is not None else COLORS
    _ = lambda n: c.get(n, "#FFFFFF")
    return f"""
QMainWindow {{ background: {_('bg_dark')}; }}
QWidget {{ color: {_('text')}; font-size: 13px; font-family: "Microsoft YaHei UI","Microsoft YaHei","SimHei","Segoe UI",sans-serif; }}
QWidget#central {{ background: {_('bg_dark')}; }}
QMenuBar {{ background: {_('bg_darkest')}; border-bottom: 1px solid {_('border')}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 14px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {_('bg_light')}; }}
QMenu {{ background: {_('bg_dark')}; border: 1px solid {_('border')}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 28px 6px 16px; border-radius: 4px; margin: 1px 2px; }}
QMenu::item:selected {{ background: {_('accent_bg')}; color: {_('text_accent')}; }}
QMenu::separator {{ height: 1px; background: {_('border')}; margin: 4px 12px; }}
QPushButton {{ background: {_('btn_default')}; color: {_('text')}; border: 1px solid {_('border')}; border-radius: 6px; padding: 6px 18px; min-height: 28px; }}
QPushButton:hover {{ background: {_('btn_default_hover')}; }}
QPushButton:disabled {{ background: {_('btn_disabled')}; color: {_('btn_disabled_text')}; }}
QPushButton#primary {{ background: {_('accent')}; color: {_('btn_default')}; font-weight: bold; border: none; border-radius: 8px; padding: 10px 24px; font-size: 14px; }}
QPushButton#primary:hover {{ background: {_('accent_hover')}; }}
QToolButton {{ background: transparent; color: {_('text')}; border: 1px solid transparent; border-radius: 6px; padding: 5px 12px; }}
QToolButton:hover {{ background: {_('bg_hover')}; }}
QToolButton:checked {{ background: {_('accent_bg')}; border-color: {_('accent')}; color: {_('text_accent')}; }}
QToolButton:disabled {{ color: {_('btn_disabled_text')}; }}
QGroupBox {{ border: 1px solid {_('border')}; border-radius: 8px; margin-top: 14px; padding: 16px 12px 12px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 14px; padding: 0 8px; }}
QLabel#title {{ font-size: 18px; font-weight: 600; color: {_('text_bright')}; }}
QLabel#dim {{ color: {_('text_dim')}; font-size: 11px; }}
QTextEdit {{ background: {_('bg_darkest')}; border: 1px solid {_('border')}; border-radius: 6px; font-family: "Cascadia Code","Consolas",monospace; font-size: 12px; padding: 6px; }}
QProgressBar {{ background: {_('bg_light')}; border: none; border-radius: 3px; text-align: center; height: 6px; }}
QProgressBar::chunk {{ background: {_('accent')}; border-radius: 3px; }}
QSlider::groove:horizontal {{ background: {_('bg_light')}; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {_('accent')}; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }}
QSlider::sub-page:horizontal {{ background: {_('accent')}; border-radius: 2px; }}
QStatusBar {{ background: {_('bg_darkest')}; border-top: 1px solid {_('border')}; font-size: 12px; color: {_('text_dim')}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {_('text_dim')}; padding: 8px 20px; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ background: {_('bg_light')}; color: {_('text')}; }}
QTabBar::tab:selected {{ color: {_('text_accent')}; border-bottom: 2px solid {_('accent')}; font-weight: 500; }}
QComboBox {{ background: {_('bg_light')}; color: {_('text')}; border: 1px solid {_('border')}; border-radius: 6px; padding: 5px 28px 5px 12px; min-height: 24px; }}
QComboBox QAbstractItemView {{ background: {_('bg')}; border: 1px solid {_('border')}; border-radius: 8px; selection-background-color: {_('accent_bg')}; }}
QSpinBox, QDoubleSpinBox {{ background: {_('bg_light')}; color: {_('text')}; border: 1px solid {_('border')}; border-radius: 6px; padding: 4px 8px; min-height: 24px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {_('scrollbar')}; border-radius: 4px; min-height: 30px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; }}
QScrollBar::handle:horizontal {{ background: {_('scrollbar')}; border-radius: 4px; min-width: 30px; margin: 2px; }}
QListWidget {{ background: {_('bg_darkest')}; border: 1px solid {_('border')}; border-radius: 8px; padding: 4px; outline: none; }}
QListWidget::item {{ padding: 8px 12px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {_('accent_bg')}; color: {_('text_accent')}; }}
QDialog {{ background: {_('bg')}; }}
QSplitter::handle {{ background: {_('border')}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QLineEdit {{ background: {_('bg_light')}; border: 1px solid {_('border')}; border-radius: 6px; padding: 6px 12px; min-height: 24px; }}
QLineEdit:focus {{ border-color: {_('accent')}; }}
QTableWidget {{ background: {_('bg_dark')}; border: 1px solid {_('border')}; border-radius: 6px; gridline-color: {_('border')}; }}
QHeaderView::section {{ background: {_('bg_darkest')}; color: {_('text_dim')}; border: none; border-bottom: 1px solid {_('border')}; padding: 8px 10px; font-weight: 500; }}
"""

# 兼容旧代码：保留 APP_STYLESHEET 作为初始值（深色主题）
APP_STYLESHEET = build_stylesheet(COLORS)
