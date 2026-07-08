import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication


def test_remove_headlands_does_not_promote_long_edge_body_band_to_headland():
    from mask_processor import remove_headlands

    mask = np.zeros((100, 160), dtype=np.uint8)
    # 多条水平种植带；最左侧是一条真实种植带，不能因为位于端部/边界就被归为田头。
    for y in (18, 32, 46, 60, 74):
        mask[y:y + 5, 3:150] = 255
    # 真正田头：与主行方向近似垂直的独立连通分量。
    mask[4:84, 152:158] = 255

    body, headland = remove_headlands(mask, 0.0, angle_thresh_deg=45.0, min_area_ratio=0.01, return_headland=True)

    assert body[48, 6] == 255
    assert headland[48, 6] == 0
    assert body[40, 155] == 0
    assert headland[40, 155] == 255


def test_residual_headland_mask_filters_missed_row_band_by_orientation():
    from row_geometry import residual_headland_mask, residual_mask_layers

    raw = np.zeros((100, 160), dtype=np.uint8)
    raw[30:36, 5:150] = 255      # missed edge planting band, parallel to rows
    raw[5:85, 152:158] = 255     # true crosswise headland
    raw[10:15, 18:35] = 255      # short parallel edge replanting fragment: headland/edge layer
    rebuilt = np.zeros_like(raw)

    headland, body_residual, uncertain = residual_mask_layers(
        raw,
        rebuilt,
        0.0,
        meters_per_px=0.05,
        angle_thresh_deg=45.0,
        min_area_ratio=0.001,
    )
    headland_only = residual_headland_mask(raw, rebuilt, 0.0, angle_thresh_deg=45.0, min_area_ratio=0.001)

    assert headland[32, 30] == 0
    assert body_residual[32, 30] == 0
    assert uncertain[32, 30] == 255
    assert headland[40, 155] == 255
    assert body_residual[40, 155] == 0
    assert headland[12, 25] == 0
    assert body_residual[12, 25] == 0
    assert uncertain[12, 25] == 255
    assert np.array_equal(headland_only, headland)


def test_repair_row_gaps_closes_short_breaks_along_row_direction():
    from row_geometry import repair_row_gaps

    mask = np.zeros((60, 140), dtype=np.uint8)
    mask[28:32, 10:58] = 255
    mask[28:32, 68:120] = 255

    repaired = repair_row_gaps(mask, 0.0, meters_per_px=0.01, config={"row_gap_close_m": 0.20, "row_gap_close_width_m": 0.03})

    assert repaired[30, 63] == 255
    assert repaired[20, 63] == 0


def test_generated_body_is_constrained_to_raw_or_bidirectional_gap_support():
    from row_geometry import constrain_generated_body_to_raw_support

    raw = np.zeros((50, 130), dtype=np.uint8)
    raw[22:26, 10:45] = 255
    raw[22:26, 58:95] = 255
    generated = raw.copy()
    generated[22:26, 45:58] = 255   # valid gap fill between two raw segments
    generated[22:26, 95:118] = 255  # invalid one-sided protrusion beyond raw endpoint

    constrained = constrain_generated_body_to_raw_support(generated, raw, 0.0, 0.01, {"row_gap_close_m": 0.25})

    assert constrained[24, 50] == 255
    assert constrained[24, 110] == 0


def test_assess_turn_strategy_accepts_bow_without_unknown_reason():
    from path_planner import assess_turn_strategy

    result = assess_turn_strategy("bow", [5.0], 2.0)

    assert result["strategy"] == "bow"
    assert not any("未知转弯策略" in reason for reason in result["hard_reasons"])


def test_pyside_harvester_pose_draws_body_header_and_two_tracks(monkeypatch):
    from pyside6_app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)

    class FakeGeo:
        def is_ready(self):
            return True
        def pixels_per_meter(self, x, y):
            return 10.0

    window.geo = FakeGeo()
    window.state.harvester_params = {
        "cutter_width_m": 2.0,
        "cutter_forward_offset_m": 1.2,
        "track_width_m": 0.35,
        "track_gauge_m": 1.7,
        "wheelbase_m": 2.5,
        "track_length_m": 1.5,
    }

    polygons = []
    lines = []
    circles = []
    monkeypatch.setattr(window.image_view.viewer, "draw_polygon", lambda pts, color, outline=None, width=1: polygons.append((pts, color, outline, width)))
    monkeypatch.setattr(window.image_view.viewer, "draw_line", lambda x1, y1, x2, y2, color=(0, 200, 80), width=2, style=None: lines.append((x1, y1, x2, y2, color, width)))
    monkeypatch.setattr(window.image_view.viewer, "draw_circle", lambda x, y, radius=4, color=(255, 255, 0), **kwargs: circles.append((x, y, radius, color, kwargs)))

    window._draw_harvester_pose(50, 40, 0.0)

    assert len(polygons) >= 5  # body + cutter visual + 2 tracks + tank
    colors = [item[1] for item in polygons]
    assert (30, 150, 205) in colors
    assert (178, 132, 48) in colors
    assert colors.count((32, 36, 34)) >= 2
    assert any(item[4] == (255, 225, 120) for item in lines)  # cutter leading edge
    assert circles  # cab/center marker
    assert any(item[4].get("scene_scaled") for item in circles)
