"""卡片背板的「液态玻璃」实现（DECISION-048）。

Qt 没有 CSS 的 ``backdrop-filter``：``QGraphicsBlurEffect`` 只能模糊控件自身，
模糊不到它背后的东西。这里的做法是「快照 + 降采样模糊 + 独立背景层」：

1. 影像视图结束一次交互（载入/缩放/平移）后抓一张画面；
2. 先缩到 1/8 再放大回原尺寸（两遍平滑缩放 ≈ 大半径高斯，2-4 ms）；
3. 放进一个铺满面板的背景层（z 序在卡片之下），卡片半透明叠在上面。

背景层是独立子控件，不去 override 容器的 paintEvent —— 那样在
QScrollArea + 样式表背景下会与 Qt 自身的绘制冲突（实测直接段错误）。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

__all__ = ["BackdropLayer", "blur_pixmap"]


def blur_pixmap(source: QPixmap | QImage, dark: bool = True, downscale: int = 8) -> QPixmap | None:
    """把一张图压成「磨砂玻璃」：两次降采样再放大 + 一层暗色叠加。"""
    if source is None:
        return None
    image = source.toImage() if isinstance(source, QPixmap) else source
    if image.isNull() or image.width() < 8 or image.height() < 8:
        return None
    small = image.scaled(
        max(1, image.width() // downscale),
        max(1, image.height() // downscale),
        Qt.IgnoreAspectRatio,
        Qt.SmoothTransformation,
    )
    tiny = small.scaled(
        max(1, small.width() // 2),
        max(1, small.height() // 2),
        Qt.IgnoreAspectRatio,
        Qt.SmoothTransformation,
    )
    blurred = tiny.scaled(image.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    tint = QColor(11, 18, 32, 118) if dark else QColor(248, 250, 252, 150)
    tinted = QImage(blurred.size(), QImage.Format_ARGB32_Premultiplied)
    tinted.fill(Qt.transparent)
    painter = QPainter(tinted)
    painter.drawImage(0, 0, blurred)
    painter.fillRect(tinted.rect(), tint)
    painter.end()
    return QPixmap.fromImage(tinted)


class BackdropLayer(QWidget):
    """铺满父控件的模糊背景层，永远压在兄弟控件之下。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._pixmap: QPixmap | None = None
        self._opacity = 0.92
        # 跟着父控件改尺寸：用事件过滤器而不是让面板 override resizeEvent，
        # 后者会顶掉容器自己的布局逻辑（实测会段错误）。
        if parent is not None:
            parent.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QEvent.Resize and watched is self.parent():
            self.setGeometry(watched.rect())
            self.lower()
        return False

    def has_backdrop(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def set_pixmap(self, pixmap: QPixmap | None, opacity: float = 0.92) -> bool:
        self._pixmap = pixmap if (pixmap is not None and not pixmap.isNull()) else None
        self._opacity = max(0.0, min(1.0, opacity))
        self.update()
        return self.has_backdrop()

    def paintEvent(self, event):  # noqa: N802
        if not self.has_backdrop():
            return
        painter = QPainter(self)
        painter.setOpacity(self._opacity)
        painter.drawPixmap(self.rect(), self._pixmap)
        painter.end()
