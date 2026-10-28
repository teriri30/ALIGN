# -*- coding: utf-8 -*-
"""悬浮提示的样式、响应速度与「只弹一个」（DECISION-067）。

三类问题都归这里：
1. Qt 默认要等约 700 ms 才弹 tooltip（`QStyle.SH_ToolTip_WakeUpDelay`），
   用户实测「移上去一下子没反应」——`FastTooltipStyle` 把唤醒延迟压到 150 ms；
2. 控件和它的内部子控件（`QDoubleSpinBox` 的 `QLineEdit`）都带 tooltip 时，
   原生提示会先后各弹一次，看起来就是「两个提示叠在一起」——`HoverTip`
   吞掉原生提示，自己在鼠标停下 120 ms 后只弹一个。
"""

from PySide6.QtCore import QEvent, QObject, QPoint, QTimer
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle, QToolTip, QWidget

#: 鼠标停下到弹出提示的等待时间。实测 150 ms 已经"像是立刻出现"，再短会在扫过一行时就乱弹。
WAKE_UP_DELAY_MS = 150
#: 离开提示后仍算"醒着"的时间：在这个窗口内移到下一个控件立刻再弹，不用重新等。
FALL_ASLEEP_DELAY_MS = 3000
#: HoverTip 自己的弹出延迟：比原生快一截，又不会扫过一行就乱弹。
HOVER_DELAY_MS = 120


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


class HoverTip(QObject):
    """一个控件树上只留一个提示，且弹得快。

    做法：给控件及其所有带 tooltip 的后代装事件过滤器——
    - `QEvent.ToolTip` 一律吞掉：不吞的话控件与内部 lineEdit 会各弹一次，
      用户看到的就是两个框叠在一起；
    - `Enter` 起 120 ms 单次定时器，到点用 `QToolTip.showText` 只弹一个；
    - `Leave`/`Hide`/按下鼠标/滚轮 立即收起。
    提示框放在控件左下角外侧，避免盖住光标后触发 Leave 抖动（一闪一闪）。
    """

    def __init__(self, root, delay_ms=HOVER_DELAY_MS):
        super().__init__(root)
        self._root = root
        self._delay_ms = int(delay_ms)
        self._target = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show)
        self.install_on(root)

    def install_on(self, widget):
        """给 widget 及其所有带 tooltip 的后代装过滤器（建完控件后调用一次）。"""
        parts = [widget]
        parts.extend(part for part in widget.findChildren(QWidget) if part.toolTip())
        for part in parts:
            if part.toolTip():
                part.installEventFilter(self)
        return self

    # ------------------------------------------------------------------ 事件
    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QEvent.Type.ToolTip:
            return True  # 原生提示统一禁用，否则同一处会冒出两个框
        if kind == QEvent.Type.Enter:
            if obj.toolTip():
                self._target = obj
                self._timer.start(self._delay_ms)
        elif kind in (
            QEvent.Type.Leave,
            QEvent.Type.Hide,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Wheel,
        ):
            self._timer.stop()
            if self._target is obj:
                QToolTip.hideText()
                self._target = None
        return False

    # ------------------------------------------------------------------ 显示
    def _show(self):
        target = self._target
        if target is None:
            return
        try:
            if not target.isVisible():
                return
            text = target.toolTip()
            if not text:
                return
            anchor = target.mapToGlobal(QPoint(0, target.height() + 6))
            QToolTip.showText(anchor, text, target)
        except RuntimeError:  # 控件已销毁（对话框关掉了）
            self._target = None


def install_hover_tips(root, delay_ms=HOVER_DELAY_MS):
    """给一个对话框/面板装统一提示控制器，返回控制器（调用方要留住引用）。"""
    if QApplication.instance() is None:
        return None
    return HoverTip(root, delay_ms=delay_ms)
