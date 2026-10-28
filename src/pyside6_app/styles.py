
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
    # Dark palette follows the ZoneDeck convention: blue-black canvas, one
    # low-saturation panel step above it, 1px cool border, single mint accent.
    "dark": {
        "bg_darkest": "#0B0F17", "bg_dark": "#0D1420", "bg": "#111827",
        "bg_light": "#141B26", "bg_hover": "#1B2433", "bg_card": "#111827",
        "bg_accent": "#10291F", "border": "#263241", "border_light": "#334155",
        "text": "#E5E7EB", "text_dim": "#94A3B8", "text_dimmer": "#64748B",
        "text_bright": "#F8FAFC", "text_accent": "#34D399",
        "accent": "#34D399", "accent_hover": "#4ADEA8", "accent_pressed": "#22B27D",
        "accent_bg": "#10291F",
        "green": "#34D399", "green_bg": "#0F2A20", "red": "#F87171",
        "red_bg": "#2A1517", "orange": "#FBBF24", "orange_bg": "#2A2110",
        "yellow": "#FBBF24", "purple": "#A78BFA", "teal": "#2DD4BF",
        "btn_default": "#141B26", "btn_default_hover": "#1B2433", "on_accent": "#08130E",
        "btn_disabled": "#182031", "btn_disabled_text": "#4B5563",
        "scrollbar": "#263241", "scrollbar_hover": "#3A4A5E",
        "success_bg": "#0F2A20", "pending_bg": "#2A2110",
    },
    # Light palette follows the Codex++ convention: near-white canvas, white
    # cards on a cool grey page, single indigo accent.
    "light": {
        "bg_darkest": "#EEF1F6", "bg_dark": "#F7F8FA", "bg": "#FFFFFF",
        "bg_light": "#FFFFFF", "bg_hover": "#EEF1FF", "bg_card": "#FFFFFF",
        "bg_accent": "#EEF0FF", "border": "#E5E7EE", "border_light": "#D7DBE6",
        "text": "#1F2937", "text_dim": "#6B7280", "text_dimmer": "#9AA1AE",
        "text_bright": "#0F172A", "text_accent": "#3B5BFF",
        "accent": "#3B5BFF", "accent_hover": "#2F4BE0", "accent_pressed": "#2539C4",
        "accent_bg": "#EEF0FF",
        "green": "#16A34A", "green_bg": "#ECFDF3", "red": "#DC2626",
        "red_bg": "#FEF2F2", "orange": "#B45309", "orange_bg": "#FFF7ED",
        "yellow": "#8A6D00", "purple": "#744DA9", "teal": "#038387",
        "btn_default": "#FFFFFF", "btn_default_hover": "#F1F3F9", "on_accent": "#FFFFFF",
        "btn_disabled": "#F1F3F9", "btn_disabled_text": "#A8AEBB",
        "scrollbar": "#D7DBE6", "scrollbar_hover": "#B7BECD",
        "success_bg": "#ECFDF3", "pending_bg": "#FFF7ED",
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

# 全站字体族（按优先级）：数字/英文走 Times New Roman（新罗马），
# 中文回退到微软雅黑。Qt 是逐字符匹配字体族的，Times New Roman 没有中文字形，
# 所以中文自然落到下一个族，不需要手工拆分文本。
FONT_FAMILIES = (
    "Times New Roman",
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
)
FONT_FAMILY_CSS = ",".join(f'"{name}"' for name in FONT_FAMILIES)
FONT_POINT_SIZE = 10


def ui_font(point_size: int, weight=None):
    """统一字体：数字/英文 Times New Roman，中文微软雅黑。"""
    from PySide6.QtGui import QFont

    font = QFont(FONT_FAMILIES[0], point_size)
    if hasattr(font, "setFamilies"):
        font.setFamilies(list(FONT_FAMILIES))
    if weight is not None:
        font.setWeight(weight)
    return font


def build_stylesheet(colors=None):
    """动态生成全局样式表 — 根据当前主题颜色"""
    c = colors if colors is not None else COLORS
    _ = lambda n: c.get(n, "#FFFFFF")
    return f"""
QMainWindow {{ background: {_('bg_dark')}; }}
/* 字体族不写在 QSS 里：QSS 的 font-family 会覆盖 setFont，导致部分控件回退到 SimHei。
   统一由 QApplication.setFont(build_app_font()) 提供。
   注意：QSS 只认 C 风格块注释，用井号写注释会把紧随其后的规则整条吞掉；
   注释里也不能再出现块注释的结束符，否则注释会提前闭合。 */
QWidget {{ color: {_('text')}; font-size: 14px; }}
QWidget#central {{ background: {_('bg_dark')}; }}
QMenuBar {{ background: {_('bg_darkest')}; border-bottom: 1px solid {_('border')}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 14px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {_('bg_light')}; }}
QMenu {{ background: {_('bg_dark')}; border: 1px solid {_('border')}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 28px 6px 16px; border-radius: 4px; margin: 1px 2px; }}
QMenu::item:selected {{ background: {_('accent_bg')}; color: {_('text_accent')}; }}
QMenu::separator {{ height: 1px; background: {_('border')}; margin: 4px 12px; }}
QCheckBox {{ color: {_('text_dim')}; spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {_('border')}; border-radius: 3px; background: {_('bg_darkest')}; }}
QCheckBox::indicator:hover {{ border-color: {_('accent')}; }}
QCheckBox::indicator:checked {{ background: {_('accent')}; border-color: {_('accent')}; }}
QPushButton {{ background: {_('btn_default')}; color: {_('text')}; border: 1px solid {_('border')}; border-radius: 6px; padding: 6px 18px; min-height: 28px; }}
QPushButton:hover {{ background: {_('btn_default_hover')}; }}
QPushButton:disabled {{ background: {_('btn_disabled')}; color: {_('btn_disabled_text')}; }}
QPushButton#primary {{ background: {_('accent')}; color: {_('on_accent')}; font-weight: bold; border: none; border-radius: 8px; padding: 10px 24px; font-size: 15px; }}
QPushButton#primary:hover {{ background: {_('accent_hover')}; }}
QToolButton {{ background: transparent; color: {_('text')}; border: 1px solid transparent; border-radius: 6px; padding: 5px 12px; }}
QToolButton:hover {{ background: {_('bg_hover')}; }}
QToolButton:checked {{ background: {_('accent_bg')}; border-color: {_('accent')}; color: {_('text_accent')}; }}
QToolButton:disabled {{ color: {_('btn_disabled_text')}; }}
QGroupBox {{ border: 1px solid {_('border')}; border-radius: 8px; margin-top: 14px; padding: 16px 12px 12px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 14px; padding: 0 8px; }}
QLabel#title {{ font-size: 19px; font-weight: 600; color: {_('text_bright')}; }}
QLabel#dim {{ color: {_('text_dim')}; font-size: 12px; }}
QTextEdit {{ background: {_('bg_darkest')}; border: 1px solid {_('border')}; border-radius: 6px; font-family: "Cascadia Code","Consolas",monospace; font-size: 13px; padding: 6px; }}
QProgressBar {{ background: {_('bg_light')}; border: none; border-radius: 3px; text-align: center; height: 6px; }}
QProgressBar::chunk {{ background: {_('accent')}; border-radius: 3px; }}
QSlider::groove:horizontal {{ background: {_('bg_light')}; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {_('accent')}; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }}
QSlider::sub-page:horizontal {{ background: {_('accent')}; border-radius: 2px; }}
QStatusBar {{ background: {_('bg_darkest')}; border-top: 1px solid {_('border')}; font-size: 13px; color: {_('text_dim')}; }}
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
/* DECISION-067：提示框固定用浅底深字——用户口径「白色的更显眼」，
   深色主题下也保持这个对比，不跟主题走。 */
QToolTip {{ background-color: #F7F7F7; color: #1B1B1B; border: 1px solid #C9C9C9; padding: 6px 8px; border-radius: 4px; }}
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
