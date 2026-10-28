# -*- coding: utf-8 -*-
"""设置项的悬浮说明契约（DECISION-066）。

用户口径：「鼠标移动上去就有一个显示的详情，介绍这是什么东西、数值应该设置多少，
鼠标移开就消失」。踩过的坑：`QDoubleSpinBox` 的输入区是内部 `QLineEdit`，只对外层
`setToolTip` 时鼠标落在数字上根本不弹提示（用户实测"并没有看到悬浮说明"）。
这里钉住：标签、ⓘ 标记、外层控件、光标下的子控件（lineEdit）都挂同一条说明。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QLabel

from pyside6_app.dialogs import HarvesterParamsDialog, SettingsDialog

SETTINGS_FIELDS = (
    "spin_conf",
    "spin_iou",
    "spin_tile",
    "spin_interval",
    "spin_buffer",
    "spin_gap",
    "spin_end_gap",
    "spin_trim",
)

HARVESTER_FIELDS = (
    "spin_cutter",
    "spin_cutter_offset",
    "spin_track",
    "spin_gauge",
    "spin_track_length",
    "spin_turn",
    "spin_work_speed",
    "spin_turn_speed",
    "spin_entry_speed",
)


def _app():
    return QApplication.instance() or QApplication([])


def test_settings_fields_explain_themselves_under_the_cursor():
    app = _app()
    dialog = SettingsDialog({"model": {}, "path_planning": {}, "mask_processing": {}})
    dialog.show()
    app.processEvents()
    for name in SETTINGS_FIELDS:
        widget = getattr(dialog, name)
        assert widget.toolTip(), name
        point = QPoint(max(4, widget.width() // 2), max(4, widget.height() // 2))
        child = widget.childAt(point)
        assert child is not None, name
        assert child.toolTip() == widget.toolTip(), (
            "%s 的说明没挂到光标下的控件（%s 上），鼠标落在数字上不会弹提示"
            % (name, type(child).__name__)
        )
        assert len(widget.toolTip().split("\n")) >= 3, "%s 的说明太短，讲不清怎么填" % name

    # 9 = 8 个可填数值 + 不可调的「处理强度」（它也要说明为什么不可调）
    expected_marks = len(SETTINGS_FIELDS) + 1
    marks = [label for label in dialog.findChildren(QLabel) if label.text() == "ⓘ"]
    assert len(marks) == expected_marks, "每个设置项都要有一个 ⓘ 悬停标记"
    assert all(mark.toolTip() for mark in marks)
    dialog.close()


def test_harvester_params_fields_explain_themselves():
    app = _app()
    dialog = HarvesterParamsDialog({})
    dialog.show()
    app.processEvents()
    for name in HARVESTER_FIELDS:
        widget = getattr(dialog, name)
        assert widget.toolTip(), name
        assert widget.lineEdit().toolTip() == widget.toolTip(), name
    # 割刀前置距离必须强调"实测"，它是导出前的硬门槛
    assert "必须实测" in dialog.spin_cutter_offset.toolTip()
    dialog.close()
