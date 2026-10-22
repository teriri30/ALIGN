"""顶栏右上角按钮的可点面积与可见性（DECISION-047）。

原来的 32x28 透明按钮在深色主题下几乎看不见，用户反馈"都快看不见"。
这里把"够大、有底色、图标够亮"固化成契约，避免以后调样式时又调回去。
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def toolbar():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pyside6_app.top_bar import TopToolbar

    bar = TopToolbar()
    yield bar
    bar.deleteLater()


def test_settings_and_theme_buttons_have_a_large_hit_area(toolbar):
    for button in (toolbar._settings_btn, toolbar._theme_btn):
        assert button.width() >= 36, "按钮宽度不足，深色背景上很难点中"
        assert button.height() >= 32
        assert button.iconSize().width() >= 22, "图标太小会看不清"
        assert button.cursor().shape().name == "PointingHandCursor"


def test_icon_buttons_have_a_visible_background(toolbar):
    style = toolbar._settings_btn.styleSheet()
    assert "background" in style and "transparent" not in style.split("}")[0]
    assert "border-radius" in style
    # 图标用最亮的正文色，而不是偏暗的 text
    from pyside6_app.styles import COLORS

    assert COLORS["text_bright"] in style


def test_toolbar_is_tall_enough_for_the_buttons(toolbar):
    tallest = max(
        toolbar._settings_btn.height(),
        toolbar._theme_btn.height(),
    )
    assert toolbar.height() >= tallest + 8, "顶栏比按钮还矮，按钮会被裁掉"
