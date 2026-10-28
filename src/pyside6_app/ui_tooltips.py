# -*- coding: utf-8 -*-
"""悬浮提示的响应速度（DECISION-067）。

Qt 默认要等约 700 ms 才弹 tooltip（`QStyle.SH_ToolTip_WakeUpDelay`），用户实测
「移上去一下子没反应，要加载一会才有」。这里用 `QProxyStyle` 把唤醒延迟压到
200 ms，并把"离开后仍算醒着"的时间拉长，让相邻设置项之间来回移动都能立刻出提示。
"""

from PySide6.QtWidgets import QProxyStyle, QStyle

#: 鼠标停下到弹出提示的等待时间。实测 150 ms 已经"像是立刻出现"，再短会在扫过一行时就乱弹。
WAKE_UP_DELAY_MS = 150
#: 离开提示后仍算"醒着"的时间：在这个窗口内移到下一个控件立刻再弹，不用重新等。
FALL_ASLEEP_DELAY_MS = 3000


class FastTooltipStyle(QProxyStyle):
    """只改 tooltip 时延，其余行为全部交回基础样式（Fusion）。"""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return WAKE_UP_DELAY_MS
        if hint == QStyle.StyleHint.SH_ToolTip_FallAsleepDelay:
            return FALL_ASLEEP_DELAY_MS
        return super().styleHint(hint, option, widget, returnData)


def install_fast_tooltips(app, base_style_name="Fusion"):
    """把应用的样式换成带快速 tooltip 的代理样式；已经装过就直接复用。

    必须在建立主窗口之前调用：`QApplication.setStyle` 会让已有控件重新 polish，
    启动阶段调用没有代价，运行中切换会出现一次重绘闪烁。
    """
    current = app.style()
    if isinstance(current, FastTooltipStyle):
        return current
    proxy = FastTooltipStyle(base_style_name)
    app.setStyle(proxy)
    return proxy
