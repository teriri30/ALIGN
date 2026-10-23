"""卡片背板的液态玻璃：模糊快照、背景层、面板接线（DECISION-048）。"""

from __future__ import annotations

import pytest


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _image(width: int = 240, height: int = 180):
    from PySide6.QtGui import QImage

    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(0xFF336699)
    return image


def test_blur_pixmap_produces_soft_backdrop(qapp):
    from pyside6_app.glass import blur_pixmap

    blurred = blur_pixmap(_image())
    assert blurred is not None and not blurred.isNull()
    assert (blurred.width(), blurred.height()) == (240, 180)


def test_blur_pixmap_rejects_tiny_images(qapp):
    from pyside6_app.glass import blur_pixmap

    assert blur_pixmap(_image(4, 4)) is None
    assert blur_pixmap(None) is None


def test_backdrop_layer_toggle(qapp):
    from pyside6_app.glass import BackdropLayer, blur_pixmap

    layer = BackdropLayer()
    assert layer.has_backdrop() is False
    assert layer.set_pixmap(blur_pixmap(_image())) is True
    assert layer.has_backdrop() is True
    layer.set_pixmap(None)
    assert layer.has_backdrop() is False


def test_task_panel_attaches_backdrop_and_flags_cards(qapp):
    """面板收到背板后：背景层显示、卡片切到玻璃态（半透明 + 亮边）。"""
    from pyside6_app.glass import blur_pixmap
    from pyside6_app.panels import TaskPanel

    panel = TaskPanel()
    assert panel.set_glass_backdrop(blur_pixmap(_image())) is True
    assert panel._backdrop is not None and not panel._backdrop.isHidden()
    cards = panel._step_cards
    assert cards, "面板应至少有一步卡片"
    assert all(getattr(card, "_glass_enabled", False) for card in cards)
    # 卡片真的会走玻璃分支：paintEvent 里读的就是这个标记
    assert cards[0]._glass_enabled is True

    panel.clear_glass_backdrop()
    assert all(getattr(card, "_glass_enabled", True) is False for card in cards)
    panel.deleteLater()


def test_task_panel_ignores_unusable_backdrop(qapp):
    from pyside6_app.panels import TaskPanel

    panel = TaskPanel()
    assert panel.set_glass_backdrop(None) is False
    assert panel._backdrop is None
    panel.deleteLater()


# ---------------------------------------------------------------- 背板不拉伸


def test_cover_source_rect_crops_wide_source_horizontally(qapp):
    """源图比目标宽 → 裁左右，保持目标比例（之前直接拉伸导致"奇怪条纹"）。"""
    from PySide6.QtCore import QSize

    from pyside6_app.glass import cover_source_rect

    rect = cover_source_rect(QSize(1000, 500), QSize(330, 900))
    assert rect.width() / rect.height() == pytest.approx(330 / 900, rel=1e-2)
    assert rect.left() > 0 and rect.top() == 0


def test_cover_source_rect_crops_tall_source_vertically(qapp):
    from PySide6.QtCore import QSize

    from pyside6_app.glass import cover_source_rect

    rect = cover_source_rect(QSize(400, 2000), QSize(800, 400))
    assert rect.width() / rect.height() == pytest.approx(2.0, rel=1e-2)
    assert rect.top() > 0 and rect.left() == 0


def test_cover_source_rect_handles_degenerate_sizes(qapp):
    from PySide6.QtCore import QSize

    from pyside6_app.glass import cover_source_rect

    assert cover_source_rect(QSize(0, 0), QSize(330, 900)).width() >= 0
    assert cover_source_rect(QSize(500, 500), QSize(0, 0)).width() == 500


def test_blur_is_heavier_than_before_so_rows_are_not_recognisable(qapp):
    """背板要读成柔和色晕：降采样倍数至少 12（原来 8 还能看出稻行）。"""
    from pyside6_app.glass import blur_pixmap

    assert blur_pixmap.__defaults__[1] >= 12
