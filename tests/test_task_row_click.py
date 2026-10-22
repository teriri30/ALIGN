# -*- coding: utf-8 -*-
"""TaskRow 交互契约：pending 行可点、点击转成 clicked、灰态不被 refresh_theme 抹掉。

背景：提交 8a9978b 只解耦了「能否点击」与「是否可用」（set_status(clickable=True)），
但 TaskRow 没有 mousePressEvent，clicked 仍只由被禁用的 _action_btn 发出，
_on_task_clicked 里的 blocker 提示因此仍是死路径；同时 refresh_theme 用 isEnabled()
判灰，pending 行的 0.55 灰在闪一次提示或切主题后会复原成 1.0。
两条都不在原测试覆盖范围内（704 passed 却测不出），这里补上契约。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pyside6_app.panels import TaskRow


def _row(status="pending", enabled=False, clickable=True):
    app = QApplication.instance() or QApplication([])
    row = TaskRow({"id": "segment", "label": "图像分割", "color": "#58A6FF", "action": "开始"})
    row.set_status(status=status, enabled=enabled, clickable=clickable)
    row.resize(320, 60)
    row.show()
    app.processEvents()
    return app, row


def test_pending_row_click_emits_clicked():
    """pending 行：行收鼠标事件、操作按钮仍禁用、点击转成 clicked。"""
    app, row = _row()
    fired = []
    row.clicked.connect(fired.append)

    QTest.mouseClick(row, Qt.LeftButton)
    app.processEvents()

    assert row.isEnabled() is True
    assert row._action_btn.isEnabled() is False
    assert fired == ["segment"], "行点击没有转成 clicked，blocker 提示仍不可达"
    row.close()


def test_pending_row_keeps_dim_after_refresh_theme():
    """pending 行视觉上必须一直显灰，切主题/闪提示复原后不能变回不透明。"""
    app, row = _row()
    assert "0.55" in row.styleSheet()
    row.refresh_theme()
    assert "0.55" in row.styleSheet(), "refresh_theme 把 pending 行的灰态抹掉了"
    row.close()


def test_available_row_stays_fully_opaque():
    """可用行照旧不透明，避免修灰态时误伤。"""
    app, row = _row(status="available", enabled=True, clickable=True)
    assert "1.0" in row.styleSheet()
    row.refresh_theme()
    assert "1.0" in row.styleSheet()
    row.close()


def test_action_button_click_is_reported_separately_from_row_click():
    """操作按钮与整行点击分流：点按钮确认，点行进卡片。"""
    app, row = _row(status="available", enabled=True, clickable=True)
    row_fired, action_fired = [], []
    row.clicked.connect(row_fired.append)
    row.action_clicked.connect(action_fired.append)

    QTest.mouseClick(row._action_btn, Qt.LeftButton)
    app.processEvents()

    assert action_fired == ["segment"]
    assert row_fired == [], "点操作按钮不应再走整行点击那条路"
    row.close()


def test_centerline_selection_action_requests_confirmation():
    """「选择作业线」处于待确认时，点操作按钮要发确认信号而不是只展开卡片。"""
    from pyside6_app.panels import TaskPanel

    app = QApplication.instance() or QApplication([])
    panel = TaskPanel()
    panel.show()
    app.processEvents()
    panel.update_centerline_selection(
        {"items": [{
            "id": "centerline-001", "sequence": 1, "label": "中心线 1",
            "centerline": [(0.0, 0.0), (10.0, 0.0)],
        }]},
        ["centerline-001"],
    )
    panel.set_task_status(
        "centerline_select", "available", text="已选择 1/1 条（待确认）"
    )
    app.processEvents()

    fired = []
    panel.centerline_selection_confirm_requested.connect(lambda: fired.append(True))
    panel._on_task_action("centerline_select")
    assert fired == [True]
    panel.close()

