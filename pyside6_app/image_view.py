"""
image_view.py — 可缩放/平移/覆盖绘制的图像显示控件
"""
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPixmap, QImage,
    QWheelEvent, QMouseEvent, QKeyEvent, QTransform, QFont,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem,
    QGraphicsItem,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QSlider, QComboBox, QToolButton, QStyle,
)
from pyside6_app.styles import COLORS


def _is_dark_theme():
    """根据当前主题 COLORS 判断是否为深色模式"""
    bg = COLORS.get('bg_dark', '#1F1F26')
    try:
        r = int(bg[1:3], 16)
        g = int(bg[3:5], 16)
        b = int(bg[5:7], 16)
        return (r * 299 + g * 587 + b * 114) < 38000
    except (ValueError, IndexError):
        return True


class RotationRuler(QWidget):
    """水平刻度尺旋转控件 — 拖拽即旋转视图（对齐原项目 ROTATION_DIAL）

    外观：深色圆角底板上绘制 -45°~+45° 刻度线，中心绿色指示线。
    交互：左键拖拽水平移动 → 每像素 0.25°，松手时自动吸附 90°/15°。
    """

    angle_changed = Signal(float)   # 旋转角度变化 (绝对值)

    # 绘制参数
    _PX_PER_DEG = 4.0       # 每度对应像素数
    _TICK_RANGE = 45        # 显示 ±45°
    _TICK_STEP = 5          # 刻度间距
    _MAJOR_STEP = 15        # 主刻度间距
    _SNAP_90 = 3.0          # 90° 吸附容差
    _SNAP_15 = 1.0          # 15° 吸附容差
    _DRAG_SENSITIVITY = 0.25  # 每像素旋转度数

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0.0
        self._dragging = False
        self._drag_start_x = 0
        self._drag_start_angle = 0.0
        self.setFixedHeight(32)
        self.setMinimumWidth(120)
        self.setMaximumWidth(280)
        self.setCursor(Qt.SizeHorCursor)
        self.setToolTip("左右拖拽旋转视图")
        self.setFont(QFont("Consolas", 9))

    @property
    def angle(self):
        return self._angle

    def set_angle(self, deg, snap=False):
        """设置绝对旋转角度"""
        if snap:
            deg = self._snap(deg)
        # 归一化到 [-180, 180)
        deg = ((deg + 180) % 360) - 180
        if abs(deg - self._angle) > 0.01:
            self._angle = deg
            self.update()
            self.angle_changed.emit(deg)

    @staticmethod
    def _snap(deg):
        """智能吸附：90° 倍数（3°内）> 15° 倍数（1°内）"""
        for mult in (0, 90, -90, 180, -180):
            if abs(deg - mult) < 3.0:
                return float(mult)
        nearest_15 = round(deg / 15.0) * 15.0
        if abs(deg - nearest_15) < 1.0:
            return nearest_15
        return deg

    # ── 绘制 ──

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        dark = _is_dark_theme()
        # 背景
        if dark:
            p.setPen(QPen(QColor(55, 61, 58), 1))
            p.setBrush(QColor(29, 32, 31))
        else:
            p.setPen(QPen(QColor(190, 195, 192), 1))
            p.setBrush(QColor(235, 238, 236))
        p.drawRoundedRect(1, 1, w - 2, h - 2, 5, 5)

        cx = w // 2
        tick_bottom = h - 5
        fm = p.fontMetrics()

        # 根据控件宽度动态计算可见刻度范围（消除两侧空白）
        half_range_deg = int(cx / self._PX_PER_DEG) + self._TICK_STEP
        deg_lo = int(self._angle) - half_range_deg
        deg_hi = int(self._angle) + half_range_deg
        # 对齐到 _TICK_STEP 倍数
        deg_lo = (deg_lo // self._TICK_STEP) * self._TICK_STEP
        deg_hi = (deg_hi // self._TICK_STEP + 1) * self._TICK_STEP

        for deg in range(deg_lo, deg_hi + 1, self._TICK_STEP):
            screen_x = cx + (deg - self._angle) * self._PX_PER_DEG
            if screen_x < 3 or screen_x > w - 3:
                continue
            is_major = (deg % self._MAJOR_STEP == 0)
            tick_h = 13 if is_major else 7
            if dark:
                color = QColor(165, 174, 170) if is_major else QColor(91, 99, 95)
            else:
                color = QColor(60, 68, 64) if is_major else QColor(130, 138, 134)
            p.setPen(QPen(color, 2 if is_major else 1))
            ix = int(screen_x)
            p.drawLine(ix, tick_bottom, ix, tick_bottom - tick_h)
            # 主刻度标注数字（仅在不裁切时）
            if is_major:
                lbl = f"{deg}\u00b0"
                lw = fm.horizontalAdvance(lbl)
                if screen_x - lw // 2 > 2 and screen_x + lw // 2 < w - 2:
                    p.setPen(QColor(130, 138, 134) if dark else QColor(80, 88, 84))
                    p.drawText(ix - lw // 2, tick_bottom - tick_h - 2, lbl)

        # 中心指示线（绿色）
        p.setPen(QPen(QColor(132, 160, 142) if dark else QColor(40, 140, 80), 2))
        p.drawLine(cx, 4, cx, tick_bottom)

        # 当前角度标签（顶部居中）
        angle_txt = f"{self._angle:+.1f}\u00b0"
        p.setPen(QColor(200, 210, 205) if dark else QColor(30, 40, 35))
        tw = fm.horizontalAdvance(angle_txt)
        p.drawText((w - tw) // 2, 11, angle_txt)

        p.end()

    # ── 鼠标交互 ──

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_x = e.pos().x()
            self._drag_start_angle = self._angle
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._dragging and (e.buttons() & Qt.LeftButton):
            delta_px = e.pos().x() - self._drag_start_x
            raw_angle = self._drag_start_angle + delta_px * self._DRAG_SENSITIVITY
            # 拖动时不吸附，实时显示
            self._angle = ((raw_angle + 180) % 360) - 180
            self.update()
            self.angle_changed.emit(self._angle)
            e.accept()
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.set_angle(self._angle, snap=True)  # 松手时吸附
            e.accept()
        else:
            super().mouseReleaseEvent(e)


class ImageViewer(QGraphicsView):
    mouse_moved = Signal(int, int, int)
    mouse_pressed = Signal(int, int, int)
    mouse_released = Signal(int, int, int)
    mouse_dragged = Signal(int, int, int)
    view_scale_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)
        self._overlay_group = []
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QColor(COLORS['bg_dark']))
        self._image_shape = None
        self._pixmap = None
        self._rotation_deg = 0.0
        self._pan_origin = None
        self._pan_button = None  # 哪个按钮触发的平移
        self.drawing_mode = False  # 由上层设置为 True 时禁用左键平移
        self._draw_press_pos = None    # 绘制模式按下时的屏幕坐标
        self._draw_press_scene = None  # 绘制模式按下时的场景坐标
        self._draw_dragged = False     # 是否已拖动超过阈值
        self._direct_drag_hit_test = None
        self._direct_drag_active = False
        self.setMinimumSize(200, 150)
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    def _update_scene_rect_padding(self):
        if self._pixmap is None:
            return
        scale = max(abs(self.transform().m11()), 0.001)
        image_h, image_w = self._image_shape or (self._pixmap.height(), self._pixmap.width())
        pad_x = max(image_w * 0.5, self.viewport().width() / scale)
        pad_y = max(image_h * 0.5, self.viewport().height() / scale)
        self._scene.setSceneRect(QRectF(
            -pad_x, -pad_y,
            image_w + pad_x * 2,
            image_h + pad_y * 2,
        ))

    def set_image(self, img, *, logical_shape=None):
        """DECISION-004: bound Qt texture memory without changing scene coordinates."""
        if img is None or img.size == 0:
            return
        self.clear_overlays()
        if logical_shape is None:
            logical_shape = img.shape[:2]
        from pyside6_app.workers import bounded_render_preview
        img = bounded_render_preview(img)
        logical_h, logical_w = (int(logical_shape[0]), int(logical_shape[1]))
        if min(logical_h, logical_w) <= 0:
            raise ValueError("logical image dimensions must be positive")
        self._image_shape = (logical_h, logical_w)
        pix = self._numpy_to_pixmap(img)
        if pix is None:
            return
        self._pixmap = pix
        self._pixmap_item.setPixmap(pix)
        self._pixmap_item.setTransform(QTransform().scale(
            logical_w / float(pix.width()),
            logical_h / float(pix.height()),
        ))
        self._update_scene_rect_padding()

    def clear_image(self):
        self.clear_overlays()
        self._pixmap = None
        self._image_shape = None
        self._pixmap_item.setPixmap(QPixmap())
        self._pixmap_item.setTransform(QTransform())
        self._scene.setSceneRect(QRectF())

    def _numpy_to_pixmap(self, img):
        if img.dtype != np.uint8:
            if img.max() > 0:
                img = (img / img.max() * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        if img.ndim == 2:
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        elif img.shape[2] == 3:
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        elif img.shape[2] == 4:
            qimg = QImage(img.data, w, h, w * 4, QImage.Format_RGBA8888)
        else:
            return None
        return QPixmap.fromImage(qimg)

    def fit_to_view(self):
        if self._pixmap is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._clamp_zoom()
        self._update_scene_rect_padding()
        self.view_scale_changed.emit(self.get_view_scale())

    def zoom_in(self):
        self.scale(1.25, 1.25)
        self._update_scene_rect_padding()
        self.view_scale_changed.emit(self.get_view_scale())

    def capture_view_state(self):
        """Capture the current image-space viewport for lossless image swaps."""
        center = self.mapToScene(self.viewport().rect().center())
        return {
            "transform": QTransform(self.transform()),
            "center": QPointF(float(center.x()), float(center.y())),
            "rotation_deg": float(self._rotation_deg),
        }

    def restore_view_state(self, state):
        """Restore zoom, rotation and image-space centre after an image swap."""
        if not state or self._pixmap is None:
            return
        transform = state.get("transform")
        center = state.get("center")
        if isinstance(transform, QTransform):
            self.setTransform(QTransform(transform))
        self._rotation_deg = float(state.get("rotation_deg", self._rotation_deg)) % 360
        self._clamp_zoom()
        self._update_scene_rect_padding()
        if isinstance(center, QPointF):
            self.centerOn(center)
        self.view_scale_changed.emit(self.get_view_scale())

    def zoom_out(self):
        self.scale(0.8, 0.8)
        self._update_scene_rect_padding()
        self.view_scale_changed.emit(self.get_view_scale())

    def _clamp_zoom(self):
        t = self.transform()
        sx = t.m11()
        if sx < 0.02:
            self.setTransform(QTransform().scale(0.02, 0.02))
        elif sx > 200:
            self.setTransform(QTransform().scale(200, 200))

    def get_view_scale(self):
        return self.transform().m11()

    def set_rotation(self, deg):
        old = self._rotation_deg
        self._rotation_deg = deg % 360
        self.rotate(self._rotation_deg - old)
        self._update_scene_rect_padding()
        self.view_scale_changed.emit(self.get_view_scale())

    def reset_view(self):
        self._rotation_deg = 0.0
        self.resetTransform()
        self.fit_to_view()

    def get_image_coords(self, scene_x, scene_y):
        if self._image_shape is None:
            return int(scene_x), int(scene_y)
        return int(scene_x), int(scene_y)

    # ── 鼠标事件 ──

    def set_direct_drag_hit_test(self, callback):
        """Register a scene-coordinate hit test that captures left-drag before panning."""
        self._direct_drag_hit_test = callback if callable(callback) else None

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 0.87
        self.scale(f, f)
        self._clamp_zoom()
        self._update_scene_rect_padding()
        self.view_scale_changed.emit(self.get_view_scale())
        e.accept()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_scene_rect_padding()

    def mousePressEvent(self, e):
        pos = self.mapToScene(e.pos())
        btn = {Qt.LeftButton: 1, Qt.RightButton: 2, Qt.MiddleButton: 3}.get(e.button(), 0)
        if e.button() == Qt.LeftButton and self._direct_drag_hit_test is not None:
            try:
                capture = bool(
                    self._direct_drag_hit_test(int(pos.x()), int(pos.y()))
                )
            except Exception:
                capture = False
            if capture:
                self._direct_drag_active = True
                self.setCursor(Qt.ClosedHandCursor)
                self.mouse_pressed.emit(int(pos.x()), int(pos.y()), 1)
                e.accept()
                return
        # 中键 → 始终平移
        if e.button() == Qt.MiddleButton:
            self._pan_origin = e.pos()
            self._pan_button = Qt.MiddleButton
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        # 左键 → 非绘制模式下平移
        if e.button() == Qt.LeftButton and not self.drawing_mode:
            self._pan_origin = e.pos()
            self._pan_button = Qt.LeftButton
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        # 绘制模式左键 → 记录按下位置，延迟到 release 判断是点击(加点)还是拖拽(平移)
        if e.button() == Qt.LeftButton and self.drawing_mode:
            self._draw_press_pos = e.pos()
            self._draw_press_scene = (int(pos.x()), int(pos.y()))
            self._draw_dragged = False
            self._pan_origin = e.pos()
            self._pan_button = Qt.LeftButton
            e.accept()
            return
        self.mouse_pressed.emit(int(pos.x()), int(pos.y()), btn)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        pos = self.mapToScene(e.pos())
        btn = {Qt.LeftButton: 1, Qt.RightButton: 2, Qt.MiddleButton: 3}.get(e.button(), 0)
        if e.button() == Qt.LeftButton and self._direct_drag_active:
            self._direct_drag_active = False
            self.setCursor(Qt.CrossCursor if self.drawing_mode else Qt.ArrowCursor)
            self.mouse_released.emit(int(pos.x()), int(pos.y()), 1)
            e.accept()
            return
        # 绘制模式左键释放
        if e.button() == Qt.LeftButton and self.drawing_mode:
            was_dragged = self._draw_dragged
            self._draw_press_pos = None
            self._draw_dragged = False
            self._pan_origin = None
            self._pan_button = None
            self.setCursor(Qt.CrossCursor)
            if not was_dragged and self._draw_press_scene:
                # 点击（未拖拽）→ 发送加点信号
                sx, sy = self._draw_press_scene
                self.mouse_pressed.emit(sx, sy, 1)
            self._draw_press_scene = None
            e.accept()
            return
        # 结束平移（任意按钮）
        if self._pan_origin is not None:
            self._pan_origin = None
            self._pan_button = None
            self.setCursor(Qt.ArrowCursor)
            e.accept()
            return
        self.mouse_released.emit(int(pos.x()), int(pos.y()), btn)
        super().mouseReleaseEvent(e)

    def mouseMoveEvent(self, e):
        pos = self.mapToScene(e.pos())
        btns = (1 if e.buttons() & Qt.LeftButton else 0) \
             | (2 if e.buttons() & Qt.RightButton else 0) \
             | (4 if e.buttons() & Qt.MiddleButton else 0)
        if self._direct_drag_active and (e.buttons() & Qt.LeftButton):
            self.mouse_moved.emit(int(pos.x()), int(pos.y()), btns)
            self.mouse_dragged.emit(int(pos.x()), int(pos.y()), btns)
            e.accept()
            return
        # 绘制模式左键拖拽检测
        if self.drawing_mode and self._draw_press_pos is not None and (e.buttons() & Qt.LeftButton):
            dist = (e.pos() - self._draw_press_pos).manhattanLength()
            if dist > 8:
                # 超过阈值 → 切换为平移模式
                self._draw_dragged = True
                self.setCursor(Qt.ClosedHandCursor)
                if self._pan_origin is not None:
                    delta = e.pos() - self._pan_origin
                    self.horizontalScrollBar().setValue(
                        self.horizontalScrollBar().value() - delta.x())
                    self.verticalScrollBar().setValue(
                        self.verticalScrollBar().value() - delta.y())
                self._pan_origin = e.pos()
                e.accept()
                return
        # 平移（左键或中键拖拽）
        if self._pan_origin is not None and (e.buttons() & (Qt.LeftButton | Qt.MiddleButton)):
            delta = e.pos() - self._pan_origin
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            self._pan_origin = e.pos()
            e.accept()
            return
        self.mouse_moved.emit(int(pos.x()), int(pos.y()), btns)
        if btns:
            self.mouse_dragged.emit(int(pos.x()), int(pos.y()), btns)
        super().mouseMoveEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F:
            self.fit_to_view()
        elif e.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif e.key() == Qt.Key_Minus:
            self.zoom_out()
        elif e.key() == Qt.Key_R:
            self.set_rotation(self._rotation_deg + 90)
        elif e.key() == Qt.Key_Escape:
            self.reset_view()
        else:
            super().keyPressEvent(e)

    # ── 覆盖绘制 API ──

    def clear_overlays(self):
        for i in self._overlay_group:
            self._scene.removeItem(i)
        self._overlay_group.clear()

    def remove_overlays(self, items):
        """Remove a caller-owned overlay subset without disturbing other layers."""
        for item in list(items or []):
            if item in self._overlay_group:
                self._overlay_group.remove(item)
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)

    def draw_line(self, x1, y1, x2, y2, color=(0, 200, 80), width=2, style=Qt.SolidLine):
        pen = QPen(QColor(*reversed(color) if len(color) == 3 else color), width, style)
        pen.setCosmetic(True)
        line = self._scene.addLine(x1, y1, x2, y2, pen)
        line.setZValue(25)
        self._overlay_group.append(line)
        return line

    def draw_polyline(self, pts, color=(0, 200, 80), width=2, style=Qt.SolidLine):
        if len(pts) < 2:
            return
        pen = QPen(QColor(*reversed(color)), width, style)
        pen.setCosmetic(True)
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            line = self._scene.addLine(x1, y1, x2, y2, pen)
            line.setZValue(25)
            self._overlay_group.append(line)

    def draw_polygon(self, pts, color=(0, 200, 80), outline=None, width=1, alpha=255):
        if len(pts) < 3:
            return None
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in pts])
        brush = QBrush(QColor(*reversed(color), int(alpha)))
        outline_color = outline if outline is not None else color
        pen = QPen(QColor(*reversed(outline_color)), int(width))
        pen.setCosmetic(True)
        item = self._scene.addPolygon(polygon, pen, brush)
        item.setZValue(29)
        self._overlay_group.append(item)
        return item

    def draw_circle(self, x, y, radius=4, color=(255, 255, 0), scene_scaled=False):
        brush = QBrush(QColor(*reversed(color)))
        pen_w = max(1, radius // 3)
        pen = QPen(QColor(*reversed(color)), pen_w)
        pen.setCosmetic(True)
        e = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
        e.setPen(pen)
        e.setBrush(brush)
        e.setPos(float(x), float(y))
        if not scene_scaled:
            e.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        e.setZValue(30)
        self._scene.addItem(e)
        self._overlay_group.append(e)
        return e

    def draw_text(self, x, y, text, color=(255, 255, 255), size=12):
        i = self._scene.addText(text, QFont("", size))
        i.setPos(x, y)
        i.setDefaultTextColor(QColor(*reversed(color)))
        i.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        i.setZValue(31)
        self._overlay_group.append(i)
        return i

    def draw_mask_overlay(self, mask, x=0, y=0, color=(120, 255, 140), max_dim=1400):
        """Draw a downsampled transparent mask overlay in image coordinates."""
        if mask is None or mask.size == 0:
            return None
        h, w = mask.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            sw = max(1, int(round(w * scale)))
            sh = max(1, int(round(h * scale)))
            import cv2
            small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = mask
            sh, sw = h, w
        alpha = np.clip(
            small.astype(np.float32) * (115.0 / 255.0),
            0,
            115,
        ).astype(np.uint8)
        if not np.any(alpha):
            return None
        rgba = np.zeros((sh, sw, 4), dtype=np.uint8)
        rgba[:, :, 0] = int(color[0])
        rgba[:, :, 1] = int(color[1])
        rgba[:, :, 2] = int(color[2])
        rgba[:, :, 3] = alpha
        pix = self._numpy_to_pixmap(rgba)
        if pix is None:
            return None
        item = QGraphicsPixmapItem(pix)
        item.setPos(float(x), float(y))
        item.setScale(float(w) / max(1.0, float(sw)))
        item.setZValue(10)
        self._scene.addItem(item)
        self._overlay_group.append(item)
        return item


class MetricScaleBar(QWidget):
    """动态比例尺 — 地图册线段风格：粗横线 + 端帽 + 距离标注"""

    _TARGET_PX = 100  # 目标比例尺像素长度

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ppm = 0.0
        self._view_scale = 1.0
        self._scale_m = 0.0
        self.setFixedHeight(36)
        self.setMaximumWidth(150)
        self.setMinimumWidth(70)
        self.setToolTip("比例尺")

    def update_scale(self, ppm, view_scale):
        self._ppm = ppm
        self._view_scale = max(view_scale, 0.001)
        self._recalc()
        self.update()

    def _recalc(self):
        if self._ppm <= 0 or self._view_scale <= 0:
            self._scale_m = 0
            return
        raw_m = self._TARGET_PX / (self._ppm * self._view_scale)
        self._scale_m = raw_m

    @staticmethod
    def _nice_round(val):
        if val <= 0:
            return 0
        mag = 10 ** int(f"{val:.0e}".split('e')[1]) if val >= 10 else 1.0
        norm = val / mag
        if norm < 1.5:
            nice = 1.0
        elif norm < 3.5:
            nice = 2.0
        elif norm < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return nice * mag

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        dark = _is_dark_theme()

        # 圆角背景底板
        if dark:
            p.setPen(QPen(QColor(55, 61, 58), 1))
            p.setBrush(QColor(29, 32, 31))
        else:
            p.setPen(QPen(QColor(190, 195, 192), 1))
            p.setBrush(QColor(235, 238, 236))
        p.drawRoundedRect(1, 1, w - 2, h - 2, 4, 4)

        if self._scale_m <= 0:
            p.setPen(QColor(70, 75, 72) if dark else QColor(160, 165, 162))
            p.setFont(QFont("Consolas", 8))
            p.drawText(0, 0, w, h, Qt.AlignCenter, "N/A")
            p.end()
            return

        # 计算实际绘制像素长度
        scale_px = int(round(self._scale_m * self._ppm * self._view_scale))
        scale_px = min(scale_px, w - 16)
        x0 = (w - scale_px) // 2
        x1 = x0 + scale_px
        bar_y = h - 9
        cap = 7

        # 粗横线 + 大端帽
        bar_color = QColor(226, 232, 228) if dark else QColor(40, 50, 45)
        pen = QPen(bar_color, 3)
        p.setPen(pen)
        p.drawLine(x0, bar_y, x1, bar_y)
        p.drawLine(x0, bar_y - cap, x0, bar_y + cap // 2)
        p.drawLine(x1, bar_y - cap, x1, bar_y + cap // 2)

        # 距离标签
        label = self._format_distance(self._scale_m)
        p.setPen(QColor(200, 210, 205) if dark else QColor(30, 40, 35))
        p.setFont(QFont("Consolas", 9, QFont.Bold))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label)
        p.drawText((w - tw) // 2, bar_y - cap - 3, label)

        p.end()

    @staticmethod
    def _format_distance(meters):
        if meters >= 1000:
            return f"{meters / 1000:.2f} km"
        if meters >= 10:
            return f"{meters:.1f} m"
        if meters >= 1:
            return f"{meters:.2f} m"
        return f"{meters * 100:.0f} cm"


class ImageViewerGroup(QWidget):
    snapshot_requested = Signal()
    mask_compare_requested = Signal()
    measure_tool_changed = Signal(str)
    animation_pause_requested = Signal()
    animation_seek_requested = Signal(float)
    animation_speed_requested = Signal(float)

    """带工具栏的图像查看器：适应 / 缩放 / 旋转刻度尺 / 比例尺 / 坐标显示"""

    @staticmethod
    def _tb_btn_style():
        return (
            f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
            f"border:1px solid {COLORS['border']};border-radius:4px;"
            f"padding:2px 4px;font-size:11px;min-height:20px;}}"
            f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
            f"QPushButton:pressed{{background:{COLORS['accent_bg']};}}"
        )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ppm = 0.0
        self._tb_buttons = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, 1)

        self.animation_bar = QWidget()
        self.animation_bar.setObjectName("animationBar")
        animation_layout = QHBoxLayout(self.animation_bar)
        animation_layout.setContentsMargins(6, 2, 6, 2)
        animation_layout.setSpacing(6)
        self.anim_play_btn = QToolButton()
        self.anim_play_btn.setFixedSize(74, 26)
        self.anim_play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.anim_play_btn.setText("暂停")
        self.anim_play_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.anim_play_btn.setToolTip("暂停演示")
        self.anim_play_btn.setAccessibleName("暂停或继续演示")
        self.anim_play_btn.clicked.connect(self.animation_pause_requested.emit)
        animation_layout.addWidget(self.anim_play_btn)
        self.anim_slider = QSlider(Qt.Horizontal)
        self.anim_slider.setRange(0, 1000)
        self.anim_slider.setSingleStep(1)
        self.anim_slider.setPageStep(25)
        self.anim_slider.setToolTip("拖动到指定路径位置")
        self.anim_slider.sliderMoved.connect(
            lambda value: self.animation_seek_requested.emit(value / 1000.0)
        )
        animation_layout.addWidget(self.anim_slider, 1)
        self.anim_progress_label = QLabel("0%")
        self.anim_progress_label.setFixedWidth(44)
        self.anim_progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        animation_layout.addWidget(self.anim_progress_label)
        self.anim_speed_combo = QComboBox()
        self.anim_speed_combo.setFixedWidth(72)
        self.anim_speed_combo.setToolTip("播放倍速仅影响动画播放速度；路径速度采用农机参数中的作业、转弯和进出场速度")
        for ratio in (0.25, 0.5, 1, 2, 5, 10, 20, 40, 60):
            self.anim_speed_combo.addItem(f"{ratio:g}x", float(ratio))
        self.anim_speed_combo.setCurrentIndex(self.anim_speed_combo.findData(1.0))
        self.anim_speed_combo.currentIndexChanged.connect(self._emit_animation_speed)
        animation_layout.addWidget(self.anim_speed_combo)
        layout.addWidget(self.animation_bar)
        self.animation_bar.hide()

        # ── 工具栏 ──
        tbar = QHBoxLayout()
        tbar.setSpacing(6)
        tbar.setContentsMargins(4, 2, 4, 2)

        style = self._tb_btn_style()

        # 适应
        bf = QPushButton("适应")
        bf.setFixedWidth(50)
        bf.setStyleSheet(style)
        bf.clicked.connect(self.viewer.fit_to_view)
        tbar.addWidget(bf); self._tb_buttons.append(bf)

        # 缩放
        bz = QPushButton("\uff0b")
        bz.setFixedWidth(30)
        bz.setStyleSheet(style)
        bz.clicked.connect(self.viewer.zoom_in)
        tbar.addWidget(bz); self._tb_buttons.append(bz)

        bo = QPushButton("\u2212")
        bo.setFixedWidth(30)
        bo.setStyleSheet(style)
        bo.clicked.connect(self.viewer.zoom_out)
        tbar.addWidget(bo); self._tb_buttons.append(bo)

        # 分隔
        self._tb_sep = QFrame()
        self._tb_sep.setFrameShape(QFrame.VLine)
        self._tb_sep.setStyleSheet(f"background:{COLORS['border']};max-width:1px;")
        self._tb_sep.setFixedHeight(18)
        tbar.addWidget(self._tb_sep)

        # 旋转刻度尺
        self._ruler = RotationRuler()
        self._ruler.angle_changed.connect(self._on_ruler_angle)
        tbar.addWidget(self._ruler)

        # 快速对齐按钮
        btn_h = QPushButton("水平")
        btn_h.setFixedWidth(42)
        btn_h.setStyleSheet(style)
        btn_h.setToolTip("对齐到 0°")
        btn_h.clicked.connect(lambda: self._quick_rotate(0.0))
        tbar.addWidget(btn_h); self._tb_buttons.append(btn_h)

        btn_v = QPushButton("垂直")
        btn_v.setFixedWidth(42)
        btn_v.setStyleSheet(style)
        btn_v.setToolTip("对齐到 90°")
        btn_v.clicked.connect(lambda: self._quick_rotate(90.0))
        tbar.addWidget(btn_v); self._tb_buttons.append(btn_v)

        # 重置
        brst = QPushButton("重置")
        brst.setFixedWidth(42)
        brst.setStyleSheet(style)
        brst.setToolTip("重置旋转+缩放 (Esc)")
        brst.clicked.connect(self._on_reset_view)
        tbar.addWidget(brst); self._tb_buttons.append(brst)

        sep_tools = QFrame()
        sep_tools.setFrameShape(QFrame.VLine)
        sep_tools.setStyleSheet(f"background:{COLORS['border']};max-width:1px;")
        sep_tools.setFixedHeight(18)
        tbar.addWidget(sep_tools)
        self._tb_sep_tools = sep_tools

        btn_snap = QPushButton("快照")
        btn_snap.setFixedWidth(42)
        btn_snap.setStyleSheet(style)
        btn_snap.setToolTip("保存当前屏幕快照")
        btn_snap.clicked.connect(self.snapshot_requested.emit)
        tbar.addWidget(btn_snap); self._tb_buttons.append(btn_snap)

        btn_compare = QPushButton("对比")
        btn_compare.setFixedWidth(42)
        btn_compare.setStyleSheet(style)
        btn_compare.setToolTip("切换原始影像与当前正在操作的结果")
        btn_compare.clicked.connect(self.mask_compare_requested.emit)
        tbar.addWidget(btn_compare); self._tb_buttons.append(btn_compare)

        self._measure_buttons = {}
        for text, tool, tip in [
            ("•", "point", "点选查看经纬度"),
            ("/", "line", "两点测量长度"),
            ("□", "area", "闭合多边形测量面积"),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(30)
            btn.setCheckable(True)
            btn.setStyleSheet(style)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda checked, t=tool: self._on_measure_button(t, checked))
            tbar.addWidget(btn); self._tb_buttons.append(btn)
            self._measure_buttons[tool] = btn

        # 比例尺
        self._scale_bar = MetricScaleBar()
        tbar.addWidget(self._scale_bar)

        # 坐标
        self.lbl_coord = QLabel("x:0  y:0")
        self.lbl_coord.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        tbar.addWidget(self.lbl_coord)
        tbar.addStretch()

        layout.addLayout(tbar)
        self.viewer.mouse_moved.connect(
            lambda x, y, b: self.lbl_coord.setText(f"x:{x}  y:{y}"))
        self.viewer.view_scale_changed.connect(
            lambda s: self._scale_bar.update_scale(self._ppm, s))
        self.refresh_theme()

    def _emit_animation_speed(self, index):
        value = self.anim_speed_combo.itemData(index)
        if value is not None:
            self.animation_speed_requested.emit(float(value))

    def set_animation_controls_visible(self, visible):
        self.animation_bar.setVisible(bool(visible))

    def set_animation_paused(self, paused):
        icon = QStyle.SP_MediaPlay if paused else QStyle.SP_MediaPause
        self.anim_play_btn.setIcon(self.style().standardIcon(icon))
        self.anim_play_btn.setText("继续" if paused else "暂停")
        self.anim_play_btn.setToolTip("继续演示" if paused else "暂停演示")

    def set_animation_progress(self, ratio):
        ratio = max(0.0, min(1.0, float(ratio)))
        self.anim_slider.blockSignals(True)
        self.anim_slider.setValue(int(round(ratio * 1000)))
        self.anim_slider.blockSignals(False)
        self.anim_progress_label.setText(f"{ratio * 100:.0f}%")

    def set_animation_speed(self, ratio):
        for index in range(self.anim_speed_combo.count()):
            if abs(float(self.anim_speed_combo.itemData(index)) - float(ratio)) < 1e-9:
                self.anim_speed_combo.blockSignals(True)
                self.anim_speed_combo.setCurrentIndex(index)
                self.anim_speed_combo.blockSignals(False)
                break

    def _on_ruler_angle(self, angle):
        """刻度尺角度变化 → 同步到视图"""
        self.viewer.set_rotation(angle)

    def _quick_rotate(self, deg):
        """快速对齐到指定角度"""
        self.viewer.set_rotation(deg)
        self._ruler.set_angle(deg)

    def _on_reset_view(self):
        self.viewer.reset_view()
        self._ruler.set_angle(0.0)

    def _on_measure_button(self, tool, checked):
        active = tool if checked else ""
        for name, btn in self._measure_buttons.items():
            if name != tool:
                btn.setChecked(False)
        self.measure_tool_changed.emit(active)

    def set_ppm(self, ppm):
        """设置地理比例 (pixels per meter) — 驱动比例尺显示"""
        self._ppm = ppm
        self._scale_bar.update_scale(ppm, self.viewer.get_view_scale())

    def set_image(self, img, *, logical_shape=None):
        self.viewer.set_image(img, logical_shape=logical_shape)
        self.viewer.fit_to_view()

    def clear_image(self):
        self.viewer.clear_image()
        self._ppm = 0.0
        self._scale_bar.update_scale(0.0, 1.0)
        self.lbl_coord.setText("x:0  y:0")

    def refresh_theme(self):
        """主题切换后重新应用工具栏样式"""
        style = self._tb_btn_style()
        for btn in self._tb_buttons:
            btn.setStyleSheet(style)
        if hasattr(self, '_tb_sep'):
            self._tb_sep.setStyleSheet(f"background:{COLORS['border']};max-width:1px;")
        if hasattr(self, '_tb_sep_tools'):
            self._tb_sep_tools.setStyleSheet(f"background:{COLORS['border']};max-width:1px;")
        if hasattr(self, 'lbl_coord'):
            self.lbl_coord.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        if hasattr(self, 'animation_bar'):
            self.animation_bar.setStyleSheet(
                f"QWidget#animationBar{{background:{COLORS['bg_dark']};"
                f"border-top:1px solid {COLORS['border']};}}"
            )
            self.anim_progress_label.setStyleSheet(
                f"color:{COLORS['text_dim']};font-size:11px;"
            )
        # 更新 ImageViewer 背景
        if hasattr(self.viewer, '_scene'):
            self.viewer.setBackgroundBrush(QColor(COLORS['bg_dark']))
