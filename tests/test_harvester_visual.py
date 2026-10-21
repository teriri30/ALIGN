"""Offscreen checks for the reUI detailed harvester visual item."""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene

from pyside6_app.harvester_visual import HarvesterItem

PARAMS = {
    "cutter_width_m": 2.0,
    "cutter_forward_offset_m": 1.35,
    "track_width_m": 0.35,
    "track_gauge_m": 1.30,
    "wheelbase_m": 1.80,
    "track_length_m": 1.70,
    "turn_radius_m": 0.90,
}


def _app():
    return QApplication.instance() or QApplication([])


def test_bounding_rect_covers_the_visual_envelope():
    _app()
    item = HarvesterItem(PARAMS)
    rect = item.boundingRect()

    assert rect.width() > 5.0
    assert rect.height() > 2.6
    assert rect.left() <= -2.0
    assert rect.right() >= 1.9
    assert rect.top() <= -1.0
    assert rect.bottom() >= 1.0


def test_paint_renders_non_empty_machine_and_pose_rotates_it():
    _app()
    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 600, 600)
    item = HarvesterItem(PARAMS)
    scene.addItem(item)
    item.set_pose(300, 300, 0.0, 80.0, 1.2)

    image = QImage(600, 600, QImage.Format_ARGB32)
    image.fill(Qt.white)
    painter = QPainter(image)
    scene.render(painter)
    painter.end()

    pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape((600, 600, 4))
    non_white = int(np.count_nonzero(np.any(pixels[:, :, :3] != 255, axis=2)))
    assert non_white > 5000

    upright = item.sceneBoundingRect()
    item.set_pose(300, 300, math.pi / 2, 80.0, 2.0)
    rotated = item.sceneBoundingRect()
    assert rotated.height() > rotated.width()
    assert abs(rotated.width() - upright.height()) < 2.0


class _FakeGeo:
    def is_ready(self):
        return True

    def pixels_per_meter(self, x, y):
        return 50.0


def test_stale_scene_item_is_recreated_after_scene_clear():
    from pyside6_app.main_window import MainWindow

    _app()
    window = MainWindow()
    window.geo = _FakeGeo()
    window._tif_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    viewer = window.image_view.viewer
    scene = viewer._scene
    scene.setSceneRect(0, 0, 400, 300)

    window._update_harvester_item(100.0, 100.0, 0.0)
    first = window._harvester_item
    assert first is not None
    assert first.scene() is scene

    scene.clear()
    window._update_harvester_item(120.0, 100.0, 0.0)

    assert window._harvester_item is not None
    assert window._harvester_item is not first
    assert window._harvester_item.scene() is scene
    window.close()
