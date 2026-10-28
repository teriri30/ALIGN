# -*- coding: utf-8 -*-
"""悬浮提示的响应速度契约（DECISION-067）。

用户口径：「移上去一下子没反应，要加载一会才有，体验不好」。Qt 默认
`SH_ToolTip_WakeUpDelay = 700 ms`（本项目实测 693–706 ms 才弹），应用启动时
换成 `FastTooltipStyle` 后实测 152–161 ms 弹出。这里钉住三件事：
1. 唤醒延迟必须压到 200 ms 以内（再慢用户就会觉得"要等"）；
2. 离开后保持"醒着"的窗口足够长，相邻设置项之间移动能立刻再弹；
3. 启动入口真的装了这个样式，不能只在测试里成立。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle

from pyside6_app.ui_tooltips import (
    FALL_ASLEEP_DELAY_MS,
    WAKE_UP_DELAY_MS,
    FastTooltipStyle,
    install_fast_tooltips,
)


class _FakeApp:
    """只实现 `install_fast_tooltips` 用到的那两个方法，避免动全局 QApplication。"""

    def __init__(self):
        self._style = QProxyStyle("Fusion")

    def style(self):
        return self._style

    def setStyle(self, style):  # noqa: N802 - 与 Qt 接口同名
        self._style = style


def test_wake_up_delay_is_short_enough_to_feel_immediate():
    QApplication.instance() or QApplication([])
    style = FastTooltipStyle("Fusion")
    assert style.styleHint(QStyle.StyleHint.SH_ToolTip_WakeUpDelay) == WAKE_UP_DELAY_MS
    assert WAKE_UP_DELAY_MS <= 200, "实测 700 ms 被用户投诉；超过 200 ms 也算慢"
    assert (
        style.styleHint(QStyle.StyleHint.SH_ToolTip_FallAsleepDelay)
        == FALL_ASLEEP_DELAY_MS
    )
    assert FALL_ASLEEP_DELAY_MS >= 1000, "窗口太短时，移到相邻控件会重新等一次"


def test_install_is_idempotent_and_keeps_the_proxy():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    first = install_fast_tooltips(app, "Fusion")
    second = install_fast_tooltips(app, "Fusion")
    assert first is second, "重复调用不该反复换样式（每次换都会让已建控件重绘）"
    assert isinstance(app.style(), FastTooltipStyle)


def test_application_entry_installs_the_fast_tooltip_style():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "src", "app_entry.py"), encoding="utf-8") as handle:
        source = handle.read()
    assert "install_fast_tooltips(" in source, "启动入口没装快速 tooltip，改动等于没生效"
