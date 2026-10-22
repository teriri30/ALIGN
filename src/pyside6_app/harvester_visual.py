"""Realistic top-view harvester item for the ALIGN simulation view.

Local frame: +x forward, +y machine right, origin at the track-contact centre.
The item is painted in metres and placed with ``setPos``/``setRotation``/
``setScale(px_per_m)``; it is a *visual* model, deliberately separate from the
``planar_geometry_m`` footprint used by validation.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QLinearGradient, QPainter,
                           QPainterPath, QPen, QRadialGradient)
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget


def _rr(painter: QPainter, x: float, y: float, w: float, h: float, r: float):
    path = QPainterPath()
    path.addRoundedRect(QRectF(x, y, w, h), r, r)
    painter.drawPath(path)


def _poly(painter: QPainter, points):
    path = QPainterPath()
    first = True
    for x, y in points:
        if first:
            path.moveTo(x, y)
            first = False
        else:
            path.lineTo(x, y)
    path.closeSubpath()
    painter.drawPath(path)


class HarvesterItem(QGraphicsItem):
    """A detailed 2-D top view of the Xingguang 4LZZ-4.5Z ratoon-rice prototype.

    The double-Z model is the joint-developed low-compaction ratoon-rice
    machine, not the commercial 4LZ-4.5Z. Geometry is a photo-based estimate;
    replace body/tank lengths with measured values when available.
    """

    def __init__(self, params: Dict, parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self._phase = 0.0
        self._params = dict(params or {})

        cw = float(self._params.get("cutter_width_m", 2.0) or 2.0)
        co = float(self._params.get("cutter_forward_offset_m", 1.35) or 1.35)
        tw = float(self._params.get("track_width_m", 0.35) or 0.35)
        tg = float(self._params.get("track_gauge_m", 1.30) or 1.30)
        tl = float(self._params.get("track_length_m", 1.70) or 1.70)

        self.cw = max(1.0, cw)
        self.tw = max(0.15, tw)
        self.tg = max(0.6, tg)
        self.tl = max(0.8, tl)
        self.cutter_x = max(0.6, co)

        # visual envelope: header depth and body length are proportional to the
        # measured machine, not the 1.8 m validation footprint
        self.header_depth = min(0.95, max(0.65, self.cw * 0.40))
        self.body_front = self.cutter_x - self.header_depth
        self.body_len = 2.20
        self.body_rear = self.body_front - self.body_len
        self.body_w = max(1.25, self.tg + self.tw - 0.17)
        self.body_half = self.body_w / 2
        self.track_half = self.tl / 2
        self.gauge_half = self.tg / 2

        self.tank_front = self.body_front - 0.60
        self.tank_rear = self.tank_front - 1.05
        self.tank_w = self.body_w - 0.20
        self.cab_front = self.tank_front
        self.cab_rear = self.body_front
        # compact rear machinery: threshing/cleaning and engine share one bay
        self.engine_front = self.tank_rear
        self.engine_rear = self.body_rear

        self.reel_cx = self.cutter_x + 0.10
        self.reel_r = 0.45
        self.header_front = self.reel_cx + self.reel_r + 0.02

        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(45)

    # ------------------------------------------------------------------ Qt ---
    def boundingRect(self) -> QRectF:
        # include the shadow, divider tips and the folded unloading auger
        return QRectF(
            self.body_rear - 0.45,
            -self.cw / 2 - 0.45,
            (self.header_front - self.body_rear) + 1.60,
            self.cw + 0.90,
        )

    def set_pose(self, x: float, y: float, heading_rad: float,
                 px_per_m: float, phase_s: float) -> None:
        self.setPos(float(x), float(y))
        self.setRotation(math.degrees(float(heading_rad)))
        self.setScale(max(1e-3, float(px_per_m)))
        self._phase = float(phase_s)
        self.update()

    # -------------------------------------------------------------- painting --
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self._shadow(painter)
        self._track(painter, -self.gauge_half)
        self._track(painter, self.gauge_half)
        self._body(painter)
        self._tank(painter)
        self._track_outer(painter, -self.gauge_half)
        self._track_outer(painter, self.gauge_half)
        self._engine(painter)
        self._cab(painter)
        self._header(painter)
        self._feeder_and_auger(painter)
        self._cutter(painter)
        self._reel(painter)
        self._unloading_auger(painter)
        self._details(painter)

    # --------------------------------------------------------------- pieces ---
    def _shadow(self, painter: QPainter) -> None:
        grad = QRadialGradient(QPointF(0.05, 0.05), 3.0)
        grad.setColorAt(0.0, QColor(0, 0, 0, 80))
        grad.setColorAt(0.55, QColor(0, 0, 0, 38))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0.05, 0.05), 2.85, 1.05)

    def _track(self, painter: QPainter, yc: float) -> None:
        w = self.tw
        x0 = -self.track_half
        x1 = self.track_half
        painter.setBrush(QColor("#151515"))
        painter.setPen(QPen(QColor("#050505"), 0.045))
        _rr(painter, x0, yc - w / 2, self.tl, w, 0.055)
        painter.setBrush(QColor("#3c3c3c"))
        painter.setPen(Qt.NoPen)
        _rr(painter, x0 + 0.04, yc - w * 0.34, self.tl - 0.08, w * 0.68, 0.035)

        painter.setPen(QPen(QColor("#7d7d7d"), 0.020))
        tread = 0.068
        offset = (self._phase * 1.2) % tread
        count = int(self.tl / tread) + 1
        for i in range(count):
            x = x0 + (i * tread + offset)
            if x > x1:
                break
            painter.drawLine(QPointF(x, yc - w * 0.46),
                             QPointF(x, yc + w * 0.46))

        for x in (x0, x1):
            painter.setBrush(QColor("#2b2b2b"))
            painter.setPen(QPen(QColor("#8a8a8a"), 0.022))
            painter.drawEllipse(QPointF(x, yc), 0.105, 0.105)
            painter.setPen(QPen(QColor("#8a8a8a"), 0.016))
            for k in range(8):
                a = self._phase * 3.2 + k * math.pi / 4
                painter.drawLine(
                    QPointF(x, yc),
                    QPointF(x + math.cos(a) * 0.095, yc + math.sin(a) * 0.095),
                )
            painter.setBrush(QColor("#9a9a9a"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, yc), 0.038, 0.038)

        for x in (-0.45, 0.0, 0.45):
            painter.setBrush(QColor("#262626"))
            painter.setPen(QPen(QColor("#111"), 0.014))
            painter.drawEllipse(QPointF(x, yc), 0.068, 0.068)
            painter.setBrush(QColor("#8f8f8f"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, yc), 0.024, 0.024)

        painter.setPen(QPen(QColor("#777"), 0.020))
        painter.drawLine(QPointF(x0 + 0.08, yc), QPointF(x1 - 0.08, yc))

    def _body(self, painter: QPainter) -> None:
        grad = QLinearGradient(0, -self.body_half, 0, self.body_half)
        grad.setColorAt(0.0, QColor("#deded8"))
        grad.setColorAt(0.25, QColor("#ffffff"))
        grad.setColorAt(0.72, QColor("#fbfbf7"))
        grad.setColorAt(1.0, QColor("#e3e3dd"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#242424"), 0.050))
        _rr(painter, self.body_rear, -self.body_half,
            self.body_len, self.body_w, 0.10)

        painter.setPen(QPen(QColor(70, 70, 70, 110), 0.014))
        for x in (self.tank_front, self.tank_rear):
            painter.drawLine(QPointF(x, -self.body_half + 0.04),
                             QPointF(x, self.body_half - 0.04))
        for y in (-self.body_w * 0.25, self.body_w * 0.25):
            painter.drawLine(QPointF(self.body_rear + 0.04, y),
                             QPointF(self.body_front - 0.04, y))

    def _tank(self, painter: QPainter) -> None:
        x0 = self.tank_rear
        w = self.tank_front - self.tank_rear
        grad = QLinearGradient(0, -self.tank_w / 2, 0, self.tank_w / 2)
        grad.setColorAt(0.0, QColor("#eeeeea"))
        grad.setColorAt(0.5, QColor("#ffffff"))
        grad.setColorAt(1.0, QColor("#e2e2dc"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#242424"), 0.040))
        _rr(painter, x0, -self.tank_w / 2, w, self.tank_w, 0.06)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#c62828"))
        painter.drawRect(QRectF(x0, -self.tank_w / 2, 0.065, self.tank_w))
        painter.drawRect(QRectF(self.tank_front - 0.065, -self.tank_w / 2,
                                0.065, self.tank_w))

        painter.setPen(QPen(QColor(80, 80, 80, 90), 0.014))
        for x in (self.tank_rear + 0.22, self.tank_rear + 0.44,
                  self.tank_rear + 0.66, self.tank_rear + 0.86):
            painter.drawLine(QPointF(x, -self.tank_w / 2 + 0.06),
                             QPointF(x, self.tank_w / 2 - 0.06))

        # small red model badge on the tank lid
        painter.setBrush(QColor("#c62828"))
        painter.setPen(Qt.NoPen)
        _rr(painter, self.tank_front - 0.62, -0.13, 0.52, 0.26, 0.03)

    def _track_outer(self, painter: QPainter, yc: float) -> None:
        """Keep the outboard track run visible beside the tank."""
        half = self.tw / 2
        sign = 1 if yc >= 0 else -1
        outer = yc + sign * (half + 0.02)
        inner = yc + sign * (half - 0.17)
        y0, y1 = min(inner, outer), max(inner, outer)
        painter.setBrush(QColor("#141414"))
        painter.setPen(QPen(QColor("#050505"), 0.022))
        painter.drawRect(QRectF(-self.track_half, y0, self.tl, y1 - y0))
        painter.setPen(QPen(QColor("#6f6f6f"), 0.016))
        offset = (self._phase * 1.2) % 0.068
        x = -self.track_half + offset
        while x < self.track_half:
            painter.drawLine(QPointF(x, y0), QPointF(x, y1))
            x += 0.068
        painter.setPen(QPen(QColor(255, 255, 255, 55), 0.012))
        painter.drawLine(QPointF(-self.track_half, outer),
                         QPointF(self.track_half, outer))

    def _engine(self, painter: QPainter) -> None:
        x0 = self.engine_rear
        x1 = self.engine_front
        w = x1 - x0
        painter.setBrush(QColor("#d0d0ca"))
        painter.setPen(QPen(QColor("#333333"), 0.030))
        _rr(painter, x0, -0.68, w, 1.36, 0.05)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#c62828"))
        painter.drawRect(QRectF(x0, -0.68, 0.05, 1.36))
        painter.drawRect(QRectF(x1 - 0.05, -0.68, 0.05, 1.36))

        # rear-facing radiator grille
        painter.setPen(QPen(QColor("#6f6f69"), 0.014))
        for y in [(-0.55 + i * 0.05) for i in range(23)]:
            painter.drawLine(QPointF(x0 + 0.03, y), QPointF(x0 + 0.17, y))

        # cooling fan behind the grille
        cx = x0 + 0.38
        painter.setBrush(QColor("#efefea"))
        painter.setPen(QPen(QColor("#555555"), 0.016))
        painter.drawEllipse(QPointF(cx, 0.0), 0.16, 0.16)
        painter.setPen(QPen(QColor("#555555"), 0.014))
        for k in range(6):
            a = self._phase * 6.0 + k * math.pi / 3
            painter.drawLine(QPointF(cx, 0.0),
                             QPointF(cx + math.cos(a) * 0.14,
                                     math.sin(a) * 0.14))
        painter.setBrush(QColor("#777777"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, 0.0), 0.025, 0.025)

        # top engine-cover vents
        painter.setPen(QPen(QColor(150, 150, 144, 180), 0.012))
        for y in (-0.42, -0.22, 0.22, 0.42):
            painter.drawLine(QPointF(x0 + 0.26, y), QPointF(x1 - 0.12, y))

    def _cab(self, painter: QPainter) -> None:
        x0 = self.cab_front
        w = self.cab_rear - self.cab_front
        painter.setBrush(QColor("#e6e6e1"))
        painter.setPen(QPen(QColor("#666666"), 0.020))
        _rr(painter, x0 + 0.04, -0.50, w - 0.04, 1.00, 0.06)

        grad = QLinearGradient(0, -0.56, 0, 0.56)
        grad.setColorAt(0.0, QColor("#f7c9d4"))
        grad.setColorAt(0.5, QColor("#f2b8c6"))
        grad.setColorAt(1.0, QColor("#e3a2b4"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#b97286"), 0.034))
        _rr(painter, x0, -0.56, w, 1.12, 0.10)

        painter.setBrush(QColor("#c97f92"))
        painter.setPen(Qt.NoPen)
        _rr(painter, x0 - 0.04, -0.50, 0.07, 1.00, 0.03)
        painter.setPen(QPen(QColor(180, 110, 130, 140), 0.016))
        painter.drawLine(QPointF(x0 + w / 2, -0.54),
                         QPointF(x0 + w / 2, 0.54))
        painter.setPen(QPen(QColor(255, 255, 255, 170), 0.030))
        painter.drawLine(QPointF(x0 + 0.14, -0.38),
                         QPointF(x0 + 0.40, -0.38))

    def _header(self, painter: QPainter) -> None:
        grad = QLinearGradient(self.cutter_x, 0, self.body_front, 0)
        grad.setColorAt(0.0, QColor("#b71c1c"))
        grad.setColorAt(0.55, QColor("#d32f2f"))
        grad.setColorAt(1.0, QColor("#a51818"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#6f1010"), 0.045))
        _poly(painter, [
            (self.cutter_x, -self.cw / 2),
            (self.cutter_x, self.cw / 2),
            (self.body_front, self.body_half + 0.03),
            (self.body_front, -self.body_half - 0.03),
        ])

        painter.setBrush(QColor("#8f1414"))
        painter.setPen(QPen(QColor("#5a0d0d"), 0.025))
        for s in (-1, 1):
            _poly(painter, [
                (self.cutter_x, s * self.cw / 2),
                (self.cutter_x + 0.46, s * (self.cw / 2 + 0.02)),
                (self.cutter_x + 0.34, s * (self.cw / 2 - 0.18)),
            ])

    def _feeder_and_auger(self, painter: QPainter) -> None:
        painter.setBrush(QColor("#e3e3de"))
        painter.setPen(QPen(QColor("#444444"), 0.026))
        _rr(painter, self.body_front, -0.30,
            self.cutter_x - 0.20 - self.body_front, 0.60, 0.03)

        # crop mat being pulled rearward over the header pan
        phase_m = (self._phase * 1.4) % 0.22
        for i in range(12):
            x = self.cutter_x - 0.05 - ((i * 0.075 + phase_m) % 0.64)
            if x < self.body_front + 0.08:
                continue
            y = -0.74 + (i % 5) * 0.37
            painter.setPen(QPen(QColor(196, 184, 96, 210), 0.028))
            painter.drawLine(QPointF(x, y), QPointF(x + 0.12, y + 0.10))
            painter.setPen(QPen(QColor(224, 214, 140, 160), 0.012))
            painter.drawLine(QPointF(x + 0.02, y + 0.01),
                             QPointF(x + 0.11, y + 0.09))

        x0 = self.cutter_x - 0.57
        x1 = self.cutter_x - 0.23
        painter.setBrush(QColor("#d8d8d2"))
        painter.setPen(QPen(QColor("#666666"), 0.024))
        _rr(painter, x0, -0.90, x1 - x0, 1.80, 0.10)

        # three spiral flights rotating on the header auger
        painter.setPen(QPen(QColor(110, 110, 110, 190), 0.020))
        phase = (self._phase * 2.4) % 0.12
        for k in range(3):
            y = -0.86
            while y < 0.80:
                painter.drawLine(QPointF(x0 + 0.02 + phase + k * 0.04, y),
                                 QPointF(x0 + 0.11 + phase + k * 0.04, y + 0.10))
                y += 0.12
        painter.setPen(QPen(QColor(255, 255, 255, 120), 0.022))
        painter.drawLine(QPointF(x0 + 0.06, -0.84),
                         QPointF(x0 + 0.06, 0.84))

    def _cutter(self, painter: QPainter) -> None:
        kn = 0.055 * math.sin(self._phase * 26.0)
        # fixed guard bar
        painter.setPen(QPen(QColor("#3a0a0a"), 0.022))
        painter.drawLine(QPointF(self.cutter_x - 0.075, -self.cw / 2),
                         QPointF(self.cutter_x - 0.075, self.cw / 2))

        # reciprocating knife bar
        painter.setBrush(QColor("#8a1414"))
        painter.setPen(QPen(QColor("#4a0808"), 0.018))
        painter.drawRect(QRectF(self.cutter_x - 0.045 + kn,
                                -self.cw / 2, 0.09, self.cw))

        y = -self.cw / 2 + 0.02
        index = 0
        while y < self.cw / 2:
            painter.setBrush(QColor("#f2f2f2" if index % 2 == 0 else "#bdbdbd"))
            painter.setPen(QPen(QColor("#555555"), 0.008))
            _poly(painter, [
                (self.cutter_x + 0.020 + kn, y - 0.018),
                (self.cutter_x + 0.020 + kn, y + 0.018),
                (self.cutter_x + 0.145 + kn, y),
            ])
            y += 0.055
            index += 1

        # moving highlight makes the knife stroke readable at field scale
        painter.setPen(QPen(QColor(255, 255, 255, 190), 0.012))
        painter.drawLine(QPointF(self.cutter_x + 0.055 + kn, -self.cw / 2 + 0.02),
                         QPointF(self.cutter_x + 0.055 + kn, self.cw / 2 - 0.02))

    def _reel(self, painter: QPainter) -> None:
        r = self.reel_r
        cx = self.reel_cx
        painter.setBrush(QColor(225, 225, 225, 35))
        pen = QPen(QColor(255, 255, 255, 130), 0.014)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(
            QRectF(cx - r, -self.cw / 2 + 0.04,
                   2 * r, self.cw - 0.08), 0.04, 0.04)

        painter.setPen(QPen(QColor("#333333"), 0.014))
        painter.drawLine(QPointF(cx - r + 0.02, 0),
                         QPointF(cx + r - 0.02, 0))

        for i in range(5):
            a = self._phase * 2.4 + i * 2 * math.pi / 5
            bx = cx + r * math.cos(a)
            base = QColor("#1f1f1f")

            # faint ghost of the previous bat position -> visible rotation
            ghost = QColor(base)
            ghost.setAlpha(70)
            painter.setPen(QPen(ghost, 0.052))
            a2 = a - 0.18
            gx = cx + r * math.cos(a2)
            painter.drawLine(QPointF(gx, -0.94), QPointF(gx, 0.94))

            base.setAlpha(235)
            painter.setPen(QPen(base, 0.055))
            painter.drawLine(QPointF(bx, -0.94), QPointF(bx, 0.94))
            painter.setPen(QPen(QColor(210, 210, 210, 95), 0.012))
            painter.drawLine(QPointF(bx + 0.014, -0.94), QPointF(bx + 0.014, 0.94))
            painter.setPen(QPen(QColor(225, 225, 225, 200), 0.016))
            for ty in (-0.55, -0.28, 0.0, 0.28, 0.55):
                painter.drawLine(QPointF(bx, ty), QPointF(bx + 0.06, ty + 0.03))

            painter.setBrush(QColor(120, 120, 120, 235))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(bx, -0.94), 0.026, 0.026)
            painter.drawEllipse(QPointF(bx, 0.94), 0.026, 0.026)

        for s in (-1, 1):
            y = s * 0.955
            painter.setPen(QPen(QColor("#5a5a5a"), 0.022))
            painter.drawLine(QPointF(cx - r + 0.01, y),
                             QPointF(cx + r - 0.01, y))
            painter.setPen(QPen(QColor(90, 90, 90, 200), 0.012))
            for i in range(5):
                a = self._phase * 2.4 + i * 2 * math.pi / 5
                painter.drawLine(
                    QPointF(cx, y),
                    QPointF(cx + math.cos(a) * r, y),
                )
            a0 = self._phase * 2.4
            painter.setPen(QPen(QColor("#8a8a8a"), 0.018))
            painter.drawLine(QPointF(cx, y),
                             QPointF(cx + math.cos(a0) * r * 0.92, y))
            painter.setBrush(QColor("#5a5a5a"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx, y), 0.035, 0.035)

        painter.setPen(QPen(QColor("#444444"), 0.028))
        for s in (-1, 1):
            painter.drawLine(QPointF(cx - r, s * 0.95),
                             QPointF(self.body_front, s * 0.82))

    def _unloading_auger(self, painter: QPainter) -> None:
        y = -self.body_half - 0.12
        x0 = self.body_rear - 0.02
        x1 = -self.track_half - 0.02
        painter.setBrush(QColor("#f4f4ee"))
        painter.setPen(QPen(QColor("#777777"), 0.026))
        _rr(painter, x0, y - 0.08, x1 - x0, 0.16, 0.07)
        painter.setPen(QPen(QColor(120, 120, 120, 160), 0.014))
        painter.drawLine(QPointF(x0 + 0.04, y), QPointF(x1 - 0.02, y))

        painter.setBrush(QColor("#f2b8c6"))
        painter.setPen(QPen(QColor("#c97f92"), 0.024))
        painter.drawEllipse(QPointF(x0, y), 0.10, 0.10)
        painter.setBrush(QColor("#bdbdbd"))
        painter.setPen(QPen(QColor("#666666"), 0.020))
        painter.drawEllipse(QPointF(x1, y), 0.09, 0.09)
        painter.setPen(QPen(QColor("#666666"), 0.024))
        painter.drawLine(QPointF(x1, y), QPointF(x1, -self.body_half + 0.02))

    def _details(self, painter: QPainter) -> None:
        # exhaust and air cleaner beside the cab
        painter.setBrush(QColor("#2f2f2f"))
        painter.setPen(QPen(QColor("#111111"), 0.012))
        painter.drawEllipse(QPointF(self.cab_front + 0.22, 0.64), 0.052, 0.052)
        painter.setBrush(QColor("#111111"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(self.cab_front + 0.22, 0.64), 0.023, 0.023)

        painter.setBrush(QColor("#dcdcd6"))
        painter.setPen(QPen(QColor("#666666"), 0.014))
        painter.drawEllipse(QPointF(self.cab_front + 0.22, -0.64), 0.075, 0.075)

        painter.setPen(QPen(QColor(120, 120, 120, 180), 0.020))
        for s in (-1, 1):
            painter.drawLine(
                QPointF(self.tank_rear + 0.06, s * 0.72),
                QPointF(self.tank_front - 0.04, s * 0.72),
            )

        painter.setBrush(QColor("#c62828"))
        painter.setPen(QPen(QColor("#7f1d1d"), 0.012))
        painter.drawRect(QRectF(self.body_rear - 0.02, -0.275, 0.05, 0.55))
        for s in (-1, 1):
            painter.setBrush(QColor("#d32f2f"))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(self.body_rear - 0.05, s * 0.42 - 0.045,
                                    0.06, 0.09))
