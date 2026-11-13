"""原图分辨率快照导出：把画布上可见的操作图层重绘到原始栅格上。

背景（DECISION-070）：旧快照走 ``viewer.grab()``——控件截图，分辨率被
屏幕尺寸和当前缩放钉死；菜单「快照 PNG…」也只在降采样显示图上画轨迹。
两条路都到不了原始栅格的分辨率。本模块与画布同一数据源、同一色号、
同一叠加顺序，把掩膜/边界/作业线/路径/服务点等图层以「显示网格坐标 ×
display_scale」重绘到从 GeoTIFF 重新读取的全分辨率 RGB 上。

坐标口径：所有 op 的坐标、线宽、字号、半径都是显示网格像素（与
viewer.draw_* 相同）；坐标统一乘 display_scale_x/y，掩膜按 INTER_LINEAR
上采样（与画布 setScale 的平滑缩放一致）。

线宽/半径/字号的口径（WYSIWYG，2026-11-10 用户验收后修正）：画布上的
笔宽是 cosmetic（屏幕像素）、圆点与文字走 ItemIgnoresTransformations，
都恒定为屏幕像素，不随缩放变化；旧实现直接乘 display_scale，等于把
「100% 缩放下的画布」重绘到原图，用户在适应窗口（fit）下看到的 5px 作业
线在快照里只剩 1.2px，被判定为“和画布差别太大”。现在改为以导出时刻的
视图比例尺为锚：``metric = display_scale / view_scale``（view_scale =
屏幕像素 / 显示网格像素）。于是把 PNG 按导出时画布的同一屏幕尺寸查看，
线宽、半径、字号与画布逐像素一致；画布缩放到 100% 导出时 metric 退化为
display_scale，与旧行为相同。

仍非 WYSIWYG 的一项：画布标签的碰撞过滤依赖当前缩放，快照在全分辨率下
按锚点画全部选中标签。农机文件预览的圆点按画布同色号重绘。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QImageWriter,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)

# 与 viewer.draw_mask_overlay 相同的默认叠加强度（115/255 ≈ 45%）。
DEFAULT_MASK_ALPHA = 115
# 掩膜按行条融合，限制单次临时内存（源图可达 15k×21k）。
MASK_STRIPE_ROWS = 1024
# PNG 压缩等级：低压缩换写入速度（效果图不差这点体积）。
PNG_COMPRESSION = 3
JPEG_QUALITY = 95


class SnapshotCancelled(RuntimeError):
    """用户请求取消时由渲染流程抛出。"""


@dataclass
class MaskOp:
    """显示网格上的二值掩膜，(x, y) 为显示网格左上角。"""

    mask: np.ndarray
    x: float = 0.0
    y: float = 0.0
    color_bgr: Tuple[int, int, int] = (120, 255, 140)
    alpha: int = DEFAULT_MASK_ALPHA


@dataclass
class PolylineOp:
    points: Sequence[Tuple[float, float]]
    color_bgr: Tuple[int, int, int] = (40, 220, 80)
    width: int = 2
    dash: bool = False


@dataclass
class PolygonOp:
    """半透明填充多边形（禁行区），outline 画边。"""

    points: Sequence[Tuple[float, float]]
    color_bgr: Tuple[int, int, int] = (35, 35, 220)
    outline_bgr: Optional[Tuple[int, int, int]] = (20, 20, 255)
    width: int = 2
    alpha: int = 75


@dataclass
class CircleOp:
    x: float
    y: float
    radius: float
    color_bgr: Tuple[int, int, int]
    pen_ratio: int = 3


@dataclass
class LabelOp:
    x: float
    y: float
    text: str
    color_bgr: Tuple[int, int, int] = (246, 248, 251)
    size: int = 10
    rotation: float = 0.0
    center: bool = True
    halo: bool = True
    bold: bool = True


@dataclass
class SnapshotScene:
    """按画布叠加顺序收集的绘制指令集。"""

    masks: List[MaskOp] = field(default_factory=list)
    vectors: List[object] = field(default_factory=list)

    def mask(
        self,
        mask: np.ndarray,
        x: float,
        y: float,
        color_bgr: Tuple[int, int, int],
        alpha: int = DEFAULT_MASK_ALPHA,
    ) -> None:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
            return
        self.masks.append(MaskOp(mask=mask, x=float(x), y=float(y),
                                 color_bgr=tuple(color_bgr), alpha=int(alpha)))

    def polyline(
        self,
        points: Sequence[Tuple[float, float]],
        color_bgr: Tuple[int, int, int],
        width: int = 2,
        dash: bool = False,
    ) -> None:
        pts = [(float(p[0]), float(p[1])) for p in points or []]
        if len(pts) < 2:
            return
        self.vectors.append(
            PolylineOp(points=pts, color_bgr=tuple(color_bgr),
                       width=int(width), dash=bool(dash))
        )

    def polygon(
        self,
        points: Sequence[Tuple[float, float]],
        color: Tuple[int, int, int],
        outline: Optional[Tuple[int, int, int]] = (20, 20, 255),
        width: int = 2,
        alpha: int = 75,
    ) -> None:
        pts = [(float(p[0]), float(p[1])) for p in points or []]
        if len(pts) < 3:
            return
        self.vectors.append(
            PolygonOp(points=pts, color_bgr=tuple(color),
                      outline_bgr=tuple(outline) if outline else None,
                      width=int(width), alpha=int(alpha))
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        color_bgr: Tuple[int, int, int],
        pen_ratio: int = 3,
    ) -> None:
        if radius <= 0:
            return
        self.vectors.append(
            CircleOp(x=float(x), y=float(y), radius=float(radius),
                     color_bgr=tuple(color_bgr), pen_ratio=int(pen_ratio))
        )

    def label(
        self,
        x: float,
        y: float,
        text: str,
        color: Tuple[int, int, int] = (246, 248, 251),
        size: int = 10,
        rotation: float = 0.0,
        center: bool = True,
        halo: bool = True,
        bold: bool = True,
    ) -> None:
        text = str(text or "")
        if not text:
            return
        self.vectors.append(
            LabelOp(x=float(x), y=float(y), text=text, color_bgr=tuple(color),
                    size=int(size), rotation=float(rotation), center=bool(center),
                    halo=bool(halo), bold=bool(bold))
        )


def _bgr_to_rgb(color_bgr: Sequence[int]) -> Tuple[int, int, int]:
    return (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))


def _qcolor(color_bgr: Sequence[int], alpha: int = 255) -> QColor:
    r, g, b = _bgr_to_rgb(color_bgr)
    return QColor(r, g, b, int(alpha))


def _blend_mask_op(
    rgb: np.ndarray, op: MaskOp, scale_x: float, scale_y: float
) -> None:
    """把一个显示网格掩膜按比例上采样并融合进 RGB（原地）。

    权重公式与 viewer.draw_mask_overlay 一致：out = dst*(1-a) + color*a，
    a = mask 值 × op.alpha / 255。按行条处理限制临时内存。
    """
    import cv2

    mask = np.asarray(op.mask)
    m_h, m_w = mask.shape[:2]
    if m_h == 0 or m_w == 0:
        return
    src_h, src_w = rgb.shape[:2]
    target_x = int(round(float(op.x) * scale_x))
    target_y = int(round(float(op.y) * scale_y))
    target_w = max(1, int(round(m_w * scale_x)))
    target_h = max(1, int(round(m_h * scale_y)))
    x0 = max(0, target_x)
    y0 = max(0, target_y)
    x1 = min(src_w, target_x + target_w)
    y1 = min(src_h, target_y + target_h)
    if x1 <= x0 or y1 <= y0:
        return
    upsampled = cv2.resize(mask, (target_w, target_h),
                           interpolation=cv2.INTER_LINEAR)
    if upsampled.shape[:2] != (y1 - y0, x1 - x0):
        upsampled = upsampled[
            max(0, y0 - target_y): y1 - target_y,
            max(0, x0 - target_x): x1 - target_x,
        ]
        if upsampled.shape[:2] != (y1 - y0, x1 - x0):
            upsampled = cv2.resize(upsampled, (x1 - x0, y1 - y0),
                                   interpolation=cv2.INTER_LINEAR)
    alpha = int(max(0, min(255, op.alpha)))
    inv = 255 - alpha
    color_rgb = np.asarray(_bgr_to_rgb(op.color_bgr), dtype=np.int32)
    for r0 in range(y0, y1, MASK_STRIPE_ROWS):
        r1 = min(r0 + MASK_STRIPE_ROWS, y1)
        sub = upsampled[r0 - y0: r1 - y0]
        region = rgb[r0:r1, x0:x1]
        if sub.shape[:2] != region.shape[:2]:
            continue
        weights = (sub.astype(np.uint16) * alpha) // 255
        if not np.any(weights):
            continue
        sel = weights > 0
        w_sel = weights[sel].astype(np.int32)
        base = region[sel].astype(np.int32)
        blended = (
            base * inv + color_rgb[None, :] * w_sel[:, None] + 127
        ) // 255
        region[sel] = blended.astype(np.uint8)


def _paint_polyline(
    painter: QPainter, op: PolylineOp,
    scale_x: float, scale_y: float, metric: float,
) -> None:
    pen = QPen(_qcolor(op.color_bgr))
    pen.setWidthF(max(1.0, float(op.width) * metric))
    if op.dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    polygon = QPolygonF(
        [QPointF(float(x) * scale_x, float(y) * scale_y) for x, y in op.points]
    )
    painter.drawPolyline(polygon)


def _paint_polygon(
    painter: QPainter, op: PolygonOp,
    scale_x: float, scale_y: float, metric: float,
) -> None:
    if op.outline_bgr is not None:
        pen = QPen(_qcolor(op.outline_bgr))
        pen.setWidthF(max(1.0, float(op.width) * metric))
        painter.setPen(pen)
    else:
        painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_qcolor(op.color_bgr, op.alpha))
    polygon = QPolygonF(
        [QPointF(float(x) * scale_x, float(y) * scale_y) for x, y in op.points]
    )
    painter.drawPolygon(polygon)


def _paint_circle(
    painter: QPainter, op: CircleOp,
    scale_x: float, scale_y: float, metric: float,
) -> None:
    radius = max(1.0, float(op.radius) * metric)
    color = _qcolor(op.color_bgr)
    pen_w = max(1, int(radius) // int(max(1, op.pen_ratio)))
    painter.setPen(QPen(color, pen_w))
    painter.setBrush(QBrush(color))
    painter.drawEllipse(
        QPointF(float(op.x) * scale_x, float(op.y) * scale_y), radius, radius
    )


def _label_font(size_px: int, bold: bool) -> QFont:
    font = QFont()
    try:
        from pyside6_app.styles import FONT_FAMILIES

        font.setFamilies(list(FONT_FAMILIES))
    except Exception:
        pass
    font.setPixelSize(max(4, int(size_px)))
    font.setBold(bool(bold))
    return font


def _paint_label(
    painter: QPainter, op: LabelOp,
    scale_x: float, scale_y: float, metric: float,
) -> None:
    font = _label_font(int(round(float(op.size) * metric)), op.bold)
    path = QPainterPath()
    path.addText(QPointF(0.0, 0.0), font, op.text)
    if path.isEmpty():
        return
    if op.center:
        rect: QRectF = path.boundingRect()
        path.translate(-rect.center().x(), -rect.center().y())
    transform = QTransform()
    transform.translate(float(op.x) * scale_x, float(op.y) * scale_y)
    if op.rotation:
        transform.rotate(float(op.rotation))
    path = transform.map(path)
    if op.halo:
        halo_pen = QPen(QColor(6, 10, 16, 205))
        halo_pen.setWidthF(max(1.0, 1.5 * metric))
        halo_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(halo_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_qcolor(op.color_bgr))
    painter.drawPath(path)


def render_snapshot_png(
    tif_path: str,
    scene: SnapshotScene,
    output_path: str,
    *,
    display_shape: Sequence[int],
    view_scale: float = 1.0,
    progress: Optional[Callable[[int, str], None]] = None,
    image_format: Optional[str] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    jpeg_quality: Optional[int] = None,
) -> dict:
    """在原始栅格上渲染快照场景并写出图像文件。

    ``view_scale``：导出时刻画布的视图比例尺（屏幕像素 / 显示网格像素，
    即 ``viewer.get_view_scale()``）。线宽、圆点半径、字号按
    ``display_scale / view_scale`` 换算，使快照与该时刻画布所见一致；
    传 1.0（默认）等于按 100% 缩放重绘，与旧行为相同。

    返回审计元数据（源/显示尺寸、缩放、op 数量），供日志与测试核对。
    ``image_format``: "PNG"（默认）或 "JPG"；后缀大小写不敏感。
    ``jpeg_quality``: 仅 JPG 生效；None 时用模块默认 JPEG_QUALITY。
    """

    def _report(pct: int, message: str) -> None:
        if progress is not None:
            progress(int(pct), str(message))

    def _check_cancelled() -> None:
        if is_cancelled is not None and is_cancelled():
            raise SnapshotCancelled("已取消快照导出")

    tif_path = str(tif_path)
    output_path = str(output_path)
    ext = output_path.lower().rsplit(".", 1)[-1] if "." in output_path else ""
    fmt = str(image_format or ("JPG" if ext in ("jpg", "jpeg") else "PNG")).upper()
    if fmt not in ("PNG", "JPG"):
        raise ValueError(f"不支持的快照格式: {fmt}")

    import rasterio

    from config import Config
    from raster_preprocessing import read_rgb_raster

    _report(5, "正在读取原始影像（全分辨率）...")
    with rasterio.open(tif_path) as src:
        source_h, source_w = int(src.height), int(src.width)
        rgb, _meta = read_rgb_raster(
            src, config=Config().section("raster_preprocessing")
        )
    # read_rgb_raster 返回 H×W×3 uint8（ascontiguousarray 保证连续）。
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.dtype != np.uint8:
        raise ValueError(f"原始影像 RGB 形状异常: {getattr(rgb, 'shape', None)}")
    rgb = np.ascontiguousarray(rgb[:, :, :3])
    if rgb.shape[0] != source_h or rgb.shape[1] != source_w:
        source_h, source_w = int(rgb.shape[0]), int(rgb.shape[1])
    _check_cancelled()

    display_h, display_w = int(display_shape[0]), int(display_shape[1])
    if display_h <= 0 or display_w <= 0:
        raise ValueError(f"显示网格尺寸非法: {display_w}×{display_h}")
    scale_x = source_w / display_w
    scale_y = source_h / display_h
    scale = (scale_x + scale_y) / 2.0
    # WYSIWYG 锚点：画布线宽/半径/字号是屏幕像素，换算到原图必须再除以
    # 导出时刻的视图比例尺，否则等于把 100% 缩放下的画布贴到原图上。
    anchor = float(view_scale or 1.0)
    if not np.isfinite(anchor) or anchor <= 1e-6:
        anchor = 1.0
    metric = scale / anchor

    if scene.masks:
        _report(35, f"正在叠加 {len(scene.masks)} 层掩膜...")
        for op in scene.masks:
            _blend_mask_op(rgb, op, scale_x, scale_y)
            _check_cancelled()

    _report(60, f"正在绘制 {len(scene.vectors)} 个矢量图层...")
    # 文字渲染需要字体数据库背后的 Qt 应用实例；脚本/测试上下文没有时在
    # 当前线程补一个（生产环境主线程已由 MainWindow 创建，这里直接复用）。
    # 必须补 QApplication 而不是 QGuiApplication：很多调用方用
    # ``QApplication.instance() or QApplication(...)`` 决定是否建应用，裸
    # QGuiApplication 会让它们跳过创建、随后构造 QWidget 直接崩。
    if QGuiApplication.instance() is None:
        try:
            from PySide6.QtWidgets import QApplication

            QApplication(["snapshot_render"])
        except Exception:
            QGuiApplication(["snapshot_render"])
    image = QImage(source_w, source_h, QImage.Format.Format_RGB888)
    if image.isNull():
        raise MemoryError(
            f"无法分配 {source_w}×{source_h} 快照画布（约 "
            f"{source_w * source_h * 3 / 1024 / 1024:.0f} MB）"
        )
    if image.bytesPerLine() == source_w * 3:
        target = np.frombuffer(image.bits(), dtype=np.uint8)
        target = target.reshape(source_h, source_w, 3)
        target[...] = rgb
    else:
        # 行有对齐填充时按 bytesPerLine 展开（RGB888 通常无填充，防御分支）。
        padded = np.frombuffer(image.bits(), dtype=np.uint8)
        padded = padded.reshape(source_h, image.bytesPerLine())
        padded[:, : source_w * 3] = rgb.reshape(source_h, source_w * 3)
    del rgb

    painter = QPainter()
    if not painter.begin(image):
        raise RuntimeError("快照画布 QPainter 启动失败")
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        for op in scene.vectors:
            if isinstance(op, PolylineOp):
                _paint_polyline(painter, op, scale_x, scale_y, metric)
            elif isinstance(op, PolygonOp):
                _paint_polygon(painter, op, scale_x, scale_y, metric)
            elif isinstance(op, CircleOp):
                _paint_circle(painter, op, scale_x, scale_y, metric)
            elif isinstance(op, LabelOp):
                _paint_label(painter, op, scale_x, scale_y, metric)
    finally:
        painter.end()
    _check_cancelled()

    _report(85, f"正在写入 {fmt} 文件...")
    if fmt == "PNG":
        from PySide6.QtGui import QImageWriter

        writer = QImageWriter(output_path, b"PNG")
        writer.setCompression(PNG_COMPRESSION)
        if not writer.write(image):
            raise OSError(f"PNG 写入失败: {output_path} ({writer.errorString()})")
    else:
        from PySide6.QtGui import QImageWriter

        writer = QImageWriter(output_path, b"JPG")
        writer.setQuality(JPEG_QUALITY if jpeg_quality is None else int(jpeg_quality))
        if not writer.write(image):
            raise OSError(f"JPEG 写入失败: {output_path} ({writer.errorString()})")
    _report(100, "")
    return {
        "path": output_path,
        "format": fmt,
        "source_shape": [source_h, source_w],
        "display_shape": [display_h, display_w],
        "display_scale": [scale_x, scale_y],
        "view_scale": anchor,
        "metric_scale": metric,
        "mask_count": len(scene.masks),
        "vector_count": len(scene.vectors),
    }


class SnapshotExportWorker(QObject):
    """后台执行原图分辨率快照导出（QThread 亲和，见 _start_background_worker）。"""

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, tif_path, scene, output_path, display_shape,
                 view_scale: float = 1.0, parent=None):
        super().__init__(parent)
        self._tif_path = str(tif_path)
        self._scene = scene
        self._output_path = str(output_path)
        self._display_shape = (int(display_shape[0]), int(display_shape[1]))
        try:
            self._view_scale = float(view_scale or 1.0)
        except (TypeError, ValueError):
            self._view_scale = 1.0
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            result = render_snapshot_png(
                self._tif_path,
                self._scene,
                self._output_path,
                display_shape=self._display_shape,
                view_scale=self._view_scale,
                progress=self._emit_progress,
                is_cancelled=lambda: self._cancelled,
            )
        except SnapshotCancelled as exc:
            self.error.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 —— 后台线程兜底，主线程统一弹错误
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(dict(result))

    def _emit_progress(self, pct: int, message: str) -> None:
        self.progress.emit(int(pct), str(message))
