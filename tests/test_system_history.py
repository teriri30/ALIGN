import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    return window


def test_field_polygon_finish_is_global_undo_redo_operation():
    window = _window()
    window._field_pts = [(1, 1), (30, 1), (30, 30), (1, 30)]

    window._finish_field_polygon()

    assert window.state.field_boundary == [(1, 1), (30, 1), (30, 30), (1, 30)]
    assert window.state.undo_stack
    assert window.top_toolbar._history_buttons["UNDO"].isEnabled()

    window._on_undo()
    assert window.state.field_boundary == []
    assert window.top_toolbar._history_buttons["REDO"].isEnabled()

    window._on_redo()
    assert window.state.field_boundary == [(1, 1), (30, 1), (30, 30), (1, 30)]


def test_turn_strategy_change_is_global_undo_redo_operation():
    window = _window()
    window.state.turn_strategy = "bow"

    window._on_turn_strategy_changed("semicircle")

    assert window.state.turn_strategy == "semicircle"
    assert window.state.undo_stack

    window._on_undo()
    assert window.state.turn_strategy == "bow"

    window._on_redo()
    assert window.state.turn_strategy == "semicircle"


def test_entry_exit_point_placement_is_global_undo_redo_operation():
    window = _window()
    window.state.entry_exit_mode = True
    window._entry_exit_click_idx = 0

    window._place_entry_exit_point(10, 20)

    assert window.state.entry_point == (10, 20)
    assert window.state.entry_point_locked is True
    assert window.state.undo_stack

    window._on_undo()
    assert window.state.entry_point is None
    assert window.state.entry_point_locked is False

    window._on_redo()
    assert window.state.entry_point == (10, 20)
    assert window.state.entry_point_locked is True
