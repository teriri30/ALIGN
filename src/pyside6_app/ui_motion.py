"""Spring-like motion for reUI, modelled on the two reference apps.

Both reference implementations are webview UIs, so their "Q弹" feel comes from
one CSS recipe rather than a physics engine:

* Codex++ embeds ``.dshwv-body{transform-origin:50% 100%;
  transition:transform .22s cubic-bezier(.34,1.56,.64,1)}`` — a back-out curve
  whose second control point sits above 1, i.e. an ~11 % overshoot, on a
  0.22 s transition, with the transform origin anchored to the edge the motion
  grows from.
* ZoneDeck ships a Tauri (Rust + WebView2) UI, so its panels use the same
  class of CSS transition. Its bundle is compressed, which is why only the
  ``__TAURI`` marker and a few ``transition`` fragments are visible.

Qt has no CSS transitions, so the same three ingredients are reproduced here:
short durations (100–260 ms), an overshoot curve sampled from the identical
cubic-bezier, and animation anchored to the edge that should feel elastic.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer,
    QVariantAnimation,
)
from PySide6.QtGui import (
    QColor, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import QPushButton, QToolButton, QWidget

#: Durations measured from the reference CSS (0.1 s – 0.3 s).
FAST_MS = 120
BASE_MS = 220
SLOW_MS = 300


def cubic_bezier_curve(x1: float, y1: float, x2: float, y2: float) -> QEasingCurve:
    """Return a QEasingCurve that samples the CSS ``cubic-bezier(x1,y1,x2,y2)``.

    Qt ships named curves only, so the bezier is solved directly (Newton
    iterations on x, then y). This keeps the motion identical to the CSS the
    reference apps use instead of approximating it with OutBack.
    """

    def bezier(t: float, a: float, b: float) -> float:
        # cubic bezier with implicit endpoints 0 and 1
        inv = 1.0 - t
        return 3 * inv * inv * t * a + 3 * inv * t * t * b + t * t * t

    def sample(progress: float) -> float:
        target = min(1.0, max(0.0, float(progress)))
        lo, hi = 0.0, 1.0
        for _ in range(18):
            mid = (lo + hi) / 2.0
            if bezier(mid, x1, x2) < target:
                lo = mid
            else:
                hi = mid
        return bezier((lo + hi) / 2.0, y1, y2)

    curve = QEasingCurve()
    curve.setCustomType(sample)
    return curve


#: The exact curve Codex++ uses for its elastic body transform.
BACK_OUT = cubic_bezier_curve(0.34, 1.56, 0.64, 1.0)
#: A gentler sibling used where a full overshoot would look nervous.
SOFT_OUT = cubic_bezier_curve(0.22, 1.0, 0.36, 1.0)
#: Symmetric ease used for fades.
EASE_IN_OUT = cubic_bezier_curve(0.45, 0.0, 0.2, 1.0)


def animate(widget: QWidget, prop: bytes, start: float, end: float,
            duration_ms: int = BASE_MS,
            curve: Optional[QEasingCurve] = None) -> QPropertyAnimation:
    """Animate one numeric Qt property and return the running animation."""
    anim = QPropertyAnimation(widget, prop, widget)
    anim.setDuration(int(duration_ms))
    anim.setStartValue(float(start))
    anim.setEndValue(float(end))
    anim.setEasingCurve(curve if curve is not None else BACK_OUT)
    anim.start(QPropertyAnimation.DeleteWhenStopped)
    return anim


class SpringButton(QPushButton):
    """Push button with a spring press/hover response.

    Qt widgets cannot be transformed, so the feedback is painted: a short
    inner inset plus an accent wash that peaks on press and settles back with
    the same overshoot curve. It costs one QVariantAnimation and no relayout,
    which matters because these buttons sit inside a scroll area.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._press = 0.0
        self._hover = 0.0
        self._anim: Optional[QVariantAnimation] = None
        self.setCursor(Qt.PointingHandCursor)

    # ---------------------------------------------------------------- motion --
    def _tween(self, attr: str, target: float, ms: int,
               curve: QEasingCurve) -> None:
        current = getattr(self, attr)
        anim = QVariantAnimation(self)
        anim.setDuration(int(ms))
        anim.setStartValue(float(current))
        anim.setEndValue(float(target))
        anim.setEasingCurve(curve)
        anim.valueChanged.connect(lambda v, a=attr: self._set_level(a, float(v)))
        anim.start(QVariantAnimation.DeleteWhenStopped)
        self._anim = anim

    def _set_level(self, attr: str, value: float) -> None:
        setattr(self, attr, value)
        self.update()

    # ----------------------------------------------------------------- events --
    def enterEvent(self, event):
        self._tween("_hover", 1.0, FAST_MS, SOFT_OUT)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._tween("_hover", 0.0, BASE_MS, EASE_IN_OUT)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.isEnabled():
            self._tween("_press", 1.0, 90, SOFT_OUT)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._tween("_press", 0.0, 260, BACK_OUT)
        super().mouseReleaseEvent(event)

    # --------------------------------------------------------------- painting --
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isEnabled():
            return
        press = max(0.0, min(1.4, self._press))
        hover = max(0.0, min(1.0, self._hover))
        if press <= 0.01 and hover <= 0.01:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.8, 0.8, -0.8, -0.8)
        radius = 6.0
        # press: darken inward (a cheap inset shadow)
        if press > 0.01:
            wash = QColor(0, 0, 0, int(28 * min(1.0, press)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(wash)
            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)
            painter.drawPath(path)
            # 1 px inward offset ring keeps the edge crisp while pressed
            ring = QPen(QColor(255, 255, 255, int(26 * min(1.0, press))), 1.0)
            painter.setPen(ring)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1.0, 1.0, -1.0, -1.0),
                                    radius - 1, radius - 1)
        # hover: faint accent lift on the top edge
        if hover > 0.01 and press < 0.6:
            edge = QColor(255, 255, 255, int(18 * hover))
            painter.setPen(QPen(edge, 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, radius, radius)
        painter.end()


class ChevronToggle(QToolButton):
    """Chevron button that rotates with an overshoot instead of swapping icons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0.0
        self._anim: Optional[QVariantAnimation] = None
        self.setFixedSize(24, 28)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("QToolButton{background:transparent;border:none;padding:0;margin:0;}")

    def set_expanded(self, expanded: bool, animate_motion: bool = True) -> None:
        target = 90.0 if expanded else 0.0
        if self._anim is not None:
            # DeleteWhenStopped 会让 C++ 对象在结束后被销毁，Python 包装还在；
            # 再调 stop() 会抛 "Internal C++ object already deleted"。
            try:
                self._anim.stop()
            except RuntimeError:
                pass
            self._anim = None
        if not animate_motion:
            self._angle = target
            self.update()
            return
        anim = QVariantAnimation(self)
        anim.setDuration(BASE_MS)
        anim.setStartValue(float(self._angle))
        anim.setEndValue(float(target))
        anim.setEasingCurve(BACK_OUT)
        anim.valueChanged.connect(self._set_angle)
        anim.start(QVariantAnimation.DeleteWhenStopped)
        self._anim = anim

    def _set_angle(self, value) -> None:
        self._angle = float(value)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        from pyside6_app.styles import COLORS
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.translate(self.width() / 2.0, self.height() / 2.0)
        painter.rotate(self._angle)
        pen = QPen(QColor(COLORS['text']), 2.0)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath()
        path.moveTo(-2.6, -4.6)
        path.lineTo(2.6, 0.0)
        path.lineTo(-2.6, 4.6)
        painter.drawPath(path)
        painter.end()


def tween_value(owner, attr: str, target: float, ms: int,
                curve: Optional[QEasingCurve] = None,
                on_change=None) -> Optional[QVariantAnimation]:
    """Animate a plain Python float attribute on *owner*.

    Used for hover/press levels: they are not Qt properties, so the animation
    writes through ``setattr`` and calls *on_change* to trigger a repaint.
    """
    current = float(getattr(owner, attr, 0.0))
    if abs(current - float(target)) < 1e-3:
        return None
    anim = QVariantAnimation(owner)
    anim.setDuration(int(ms))
    anim.setStartValue(current)
    anim.setEndValue(float(target))
    anim.setEasingCurve(curve if curve is not None else SOFT_OUT)

    def _apply(value):
        setattr(owner, attr, float(value))
        if on_change is not None:
            on_change()

    anim.valueChanged.connect(_apply)
    anim.start(QVariantAnimation.DeleteWhenStopped)
    holder = getattr(owner, "_tween_anims", None)
    if holder is None:
        holder = {}
        setattr(owner, "_tween_anims", holder)
    holder[attr] = anim
    return anim

class BreathingDot(QWidget):
    """Status dot that breathes while busy — Codex++ `.csw-runtime-dot[data-tone=busy]`.

    The reference rule is ``animation: csw-status-breathe 1.2s ease-in-out
    infinite`` over ``opacity .45 -> 1`` and ``scale .86 -> 1``.  Qt cannot
    transform widgets, so the pulse is painted: radius and alpha follow the
    same 1.2 s cycle, which reads as a heart-beat rather than a blink.
    """

    BUSY_MS = 1200

    def __init__(self, diameter: int = 8, parent=None):
        super().__init__(parent)
        self._diameter = int(diameter)
        self.setFixedSize(self._diameter + 6, self._diameter + 6)
        self._color = QColor("#34D399")
        self._tone = "idle"
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_tone(self, tone: str, color=None) -> None:
        """tone: idle | busy | ready | error (mirrors the reference data-tone)."""
        self._tone = str(tone)
        if color is not None:
            self._color = QColor(color)
        if self._tone == "busy" and not self._timer.isActive():
            self._timer.start()
        elif self._tone != "busy" and self._timer.isActive():
            self._timer.stop()
            self._phase = 0.0
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 16.0 / self.BUSY_MS) % 1.0
        self.update()

    def paintEvent(self, event):
        import math
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        base = self._diameter / 2.0
        if self._tone == "busy":
            # sine pair reproduces ease-in-out: scale .86 -> 1, alpha .45 -> 1
            wave = 0.5 - 0.5 * math.cos(2.0 * math.pi * self._phase)
            radius = base * (0.86 + 0.14 * wave)
            color = QColor(self._color)
            color.setAlphaF(0.45 + 0.55 * wave)
        else:
            radius = base
            color = QColor(self._color)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
        painter.end()


class ShimmerBar(QWidget):
    """Progress bar whose fill carries a travelling highlight.

    Reference: ``animation: csw-progress-text-shimmer 1.8s linear infinite``
    moving ``background-position`` to ``-100%``.  The same sweep is painted
    over the completed portion so a long-running step never looks frozen.
    """

    SWEEP_MS = 1800

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(4)
        self._ratio = 0.0
        self._phase = 0.0
        self._fill = QColor("#34D399")
        self._track = QColor(255, 255, 255, 28)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_colors(self, fill, track) -> None:
        self._fill = QColor(fill); self._track = QColor(track); self.update()

    def set_ratio(self, ratio: float) -> None:
        self._ratio = max(0.0, min(1.0, float(ratio)))
        if 0.02 < self._ratio < 0.999:
            if not self._timer.isActive():
                self._timer.start()
        elif self._timer.isActive():
            self._timer.stop(); self._phase = 0.0
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 16.0 / self.SWEEP_MS) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = rect.height() / 2.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._track)
        painter.drawRoundedRect(rect, radius, radius)
        if self._ratio <= 0.0:
            painter.end(); return
        fill_w = max(rect.height(), rect.width() * self._ratio)
        fill_rect = QRectF(rect.left(), rect.top(), fill_w, rect.height())
        painter.save()
        clip = QPainterPath(); clip.addRoundedRect(fill_rect, radius, radius)
        painter.setClipPath(clip)
        painter.fillPath(clip, self._fill)
        if self._timer.isActive():
            band_w = max(24.0, fill_w * 0.28)
            x = fill_rect.left() - band_w + (fill_w + band_w) * self._phase
            grad = QLinearGradient(x, 0.0, x + band_w, 0.0)
            top = QColor(255, 255, 255, 120)
            clear = QColor(255, 255, 255, 0)
            grad.setColorAt(0.0, clear)
            grad.setColorAt(0.5, top)
            grad.setColorAt(1.0, clear)
            painter.fillRect(QRectF(x, fill_rect.top(), band_w, fill_rect.height()), grad)
        painter.restore()
        painter.end()

class BobIcon(QWidget):
    """Icon that floats up and down while a generation is running.

    Reference: ``csw-face-generate-bob .92s cubic-bezier(.45,0,.2,1) infinite``
    translating +2px -> -4px while squashing scaleY .9 -> 1.06.
    """

    BOB_MS = 920

    def __init__(self, diameter: int = 14, parent=None):
        super().__init__(parent)
        self._d = int(diameter)
        self.setFixedSize(self._d + 10, self._d + 16)
        self._phase = 0.0
        self._running = False
        self._color = QColor("#34D399")
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_running(self, running: bool, color=None) -> None:
        self._running = bool(running)
        if color is not None:
            self._color = QColor(color)
        if self._running and not self._timer.isActive():
            self._timer.start()
        elif not self._running and self._timer.isActive():
            self._timer.stop(); self._phase = 0.0
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 16.0 / self.BOB_MS) % 1.0
        self.update()

    def paintEvent(self, event):
        import math
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # sine reproduces ease-in-out: y +2 -> -4, scaleY .9 -> 1.06
        wave = 0.5 - 0.5 * math.cos(2.0 * math.pi * self._phase)
        dy = 2.0 - 6.0 * wave
        sy = 0.9 + 0.16 * wave
        cx = self.width() / 2.0
        cy = self.height() / 2.0 - 2.0
        painter.translate(cx, cy + dy)
        painter.scale(1.0, sy)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(QPointF(0.0, 0.0), self._d / 2.0, self._d / 2.0)
        painter.end()
