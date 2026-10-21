"""Theme-aware outline icon set for reUI.

All geometry lives on one 24×24 grid with a single 2.0 px round-capped stroke so
glyphs keep the same optical weight at every size. The set follows the Lucide
conventions (undo-2 / redo-2, sun, moon, sliders-horizontal, chevrons, search,
refresh-cw, trash-2, info) instead of relying on characters such as ``↩`` or
``☀``: those changed shape with the installed typeface and could never be
mirror-symmetric, which is why undo/redo looked hand drawn.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

GRID = 24.0
STROKE = 2.0


def _pen(color: str, width: float = STROKE) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _icon(draw: Callable[[QPainter, str], None], color: str,
          disabled_color: Optional[str] = None, size: int = 24) -> QIcon:
    def render(c: str) -> QPixmap:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.scale(size / GRID, size / GRID)
        draw(p, c)
        p.end()
        return pm

    icon = QIcon(render(color))
    if disabled_color:
        icon.addPixmap(render(disabled_color), QIcon.Disabled)
    return icon


def _stroke(p: QPainter, color: str, width: float = STROKE) -> None:
    p.setPen(_pen(color, width))
    p.setBrush(Qt.NoBrush)


def _polyline(points: Sequence[Tuple[float, float]]) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(*points[0])
    for pt in points[1:]:
        path.lineTo(*pt)
    return path


# ----------------------------------------------------------------- undo/redo --
def _draw_undo(p: QPainter, color: str) -> None:
    """Lucide ``undo-2`` on the 24 grid: one shared path, mirrored for redo."""
    _stroke(p, color)
    # arrow head, left pointing
    p.drawPath(_polyline([(9.0, 14.0), (4.0, 9.0), (9.0, 4.0)]))
    # shaft + half-circle return
    body = QPainterPath()
    body.moveTo(4.0, 9.0)
    body.lineTo(13.0, 9.0)
    body.arcTo(QRectF(13.0, 9.0, 8.0, 8.0), 90.0, -180.0)
    body.lineTo(17.0, 17.0)
    p.drawPath(body)


def _draw_redo(p: QPainter, color: str) -> None:
    p.save()
    p.translate(GRID, 0.0)
    p.scale(-1.0, 1.0)
    _draw_undo(p, color)
    p.restore()


# ------------------------------------------------------------- theme/theme ---
def _draw_sun(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawEllipse(QPointF(12.0, 12.0), 4.2, 4.2)
    for deg in range(0, 360, 45):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        p.drawLine(QPointF(12 + 7.6 * c, 12 + 7.6 * s),
                   QPointF(12 + 10.4 * c, 12 + 10.4 * s))


def _draw_moon(p: QPainter, color: str) -> None:
    """Thin symmetric crescent: big disc minus an offset disc."""
    outer = QPainterPath()
    outer.addEllipse(QRectF(3.5, 3.5, 17.0, 17.0))
    cut = QPainterPath()
    cut.addEllipse(QRectF(9.6, -0.4, 15.0, 15.0))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawPath(outer.subtracted(cut))


def _draw_settings(p: QPainter, color: str) -> None:
    """Lucide ``sliders-horizontal``: three equal tracks, three handles."""
    for y, handle_x in ((6.0, 15.0), (12.0, 9.0), (18.0, 16.0)):
        _stroke(p, color)
        p.drawLine(QPointF(3.0, y), QPointF(21.0, y))
        p.setBrush(QColor(color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(handle_x, y), 2.2, 2.2)


# ----------------------------------------------------------------- chevrons ---
def _draw_chevron_right(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawPath(_polyline([(9.0, 18.0), (15.0, 12.0), (9.0, 6.0)]))


def _draw_chevron_down(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawPath(_polyline([(6.0, 9.0), (12.0, 15.0), (18.0, 9.0)]))


def _draw_check(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawPath(_polyline([(19.0, 7.0), (9.5, 16.5), (5.0, 12.0)]))


# ------------------------------------------------------------------ measure ---
def _draw_point(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawEllipse(QPointF(12.0, 12.0), 4.6, 4.6)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        p.drawLine(QPointF(12 + 7.4 * dx, 12 + 7.4 * dy),
                   QPointF(12 + 10.2 * dx, 12 + 10.2 * dy))


def _draw_line(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawLine(QPointF(7.0, 17.0), QPointF(17.0, 7.0))
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(7.0, 17.0), 1.9, 1.9)
    p.drawEllipse(QPointF(17.0, 7.0), 1.9, 1.9)


def _draw_area(p: QPainter, color: str) -> None:
    """Regular hexagon, vertices on the 24 grid (no free-hand polygon)."""
    _stroke(p, color)
    pts = [(12.0, 2.8), (20.0, 7.4), (20.0, 16.6),
           (12.0, 21.2), (4.0, 16.6), (4.0, 7.4)]
    p.drawPath(_polyline(pts + [pts[0]]))
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    for x, y in pts:
        p.drawEllipse(QPointF(x, y), 1.6, 1.6)


# ------------------------------------------------------------------- basics ---
def _draw_plus(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawLine(QPointF(12.0, 4.0), QPointF(12.0, 20.0))
    p.drawLine(QPointF(4.0, 12.0), QPointF(20.0, 12.0))


def _draw_minus(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawLine(QPointF(4.0, 12.0), QPointF(20.0, 12.0))


def _draw_search(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawEllipse(QPointF(10.5, 10.5), 6.5, 6.5)
    p.drawLine(QPointF(15.4, 15.4), QPointF(20.0, 20.0))


def _draw_refresh(p: QPainter, color: str) -> None:
    """Lucide ``refresh-cw``: 3/4 circle plus a tangent arrow head."""
    _stroke(p, color)
    rect = QRectF(4.0, 4.0, 16.0, 16.0)
    path = QPainterPath()
    path.arcMoveTo(rect, 40.0)
    path.arcTo(rect, 40.0, 300.0)
    p.drawPath(path)
    p.drawPath(_polyline([(12.0, 4.0), (16.4, 5.2), (15.2, 9.6)]))


def _draw_trash(p: QPainter, color: str) -> None:
    """Lucide ``trash-2`` proportions on the 24 grid."""
    _stroke(p, color)
    p.drawLine(QPointF(4.0, 6.0), QPointF(20.0, 6.0))
    p.drawPath(_polyline([(9.0, 6.0), (9.0, 3.6), (15.0, 3.6), (15.0, 6.0)]))
    p.drawPath(_polyline([(5.8, 6.0), (6.8, 20.4), (17.2, 20.4), (18.2, 6.0)]))
    p.drawLine(QPointF(10.2, 10.0), QPointF(10.2, 16.8))
    p.drawLine(QPointF(13.8, 10.0), QPointF(13.8, 16.8))


def _draw_info(p: QPainter, color: str) -> None:
    _stroke(p, color)
    p.drawEllipse(QPointF(12.0, 12.0), 8.4, 8.4)
    p.drawLine(QPointF(12.0, 11.0), QPointF(12.0, 16.4))
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(12.0, 7.8), 1.05, 1.05)


# -------------------------------------------------------------- public API ---
def undo_icon(color: str, disabled_color: Optional[str] = None) -> QIcon:
    return _icon(_draw_undo, color, disabled_color)


def redo_icon(color: str, disabled_color: Optional[str] = None) -> QIcon:
    return _icon(_draw_redo, color, disabled_color)


def theme_icon(theme_name: str, color: str) -> QIcon:
    return _icon(_draw_moon if theme_name == "light" else _draw_sun, color)


def settings_icon(color: str) -> QIcon:
    return _icon(_draw_settings, color)


def chevron_icon(expanded: bool, color: str) -> QIcon:
    return _icon(_draw_chevron_down if expanded else _draw_chevron_right, color)


def check_icon(color: str) -> QIcon:
    return _icon(_draw_check, color)


def point_icon(color: str) -> QIcon:
    return _icon(_draw_point, color)


def line_icon(color: str) -> QIcon:
    return _icon(_draw_line, color)


def area_icon(color: str) -> QIcon:
    return _icon(_draw_area, color)


def plus_icon(color: str) -> QIcon:
    return _icon(_draw_plus, color)


def minus_icon(color: str) -> QIcon:
    return _icon(_draw_minus, color)


def search_icon(color: str) -> QIcon:
    return _icon(_draw_search, color)


def refresh_icon(color: str) -> QIcon:
    return _icon(_draw_refresh, color)


def trash_icon(color: str, disabled_color: Optional[str] = None) -> QIcon:
    return _icon(_draw_trash, color, disabled_color)


def info_icon(color: str) -> QIcon:
    return _icon(_draw_info, color)
