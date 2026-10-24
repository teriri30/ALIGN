"""作业线重压中线：用行带自身轮廓的弦中点做稳健拟合。"""
import numpy as np

from row_geometry import _recenter_band_centerlines


def _slanted_band(offset_px: float, width_px: int = 70, length_px: int = 1200):
    """造一条倾斜的行带：轮廓是一个长条四边形，中线沿 y 方向略带弯曲。"""
    t = np.linspace(0, 1, 60)
    cx = np.full_like(t, 300.0)               # 直带：只考"偏心能不能被拉回"
    cy = 100 + length_px * t
    half = width_px / 2
    left = np.stack([cx - half, cy], axis=1)
    right = np.stack([cx + half, cy], axis=1)[::-1]
    contour = np.vstack([left, right]).astype(np.int32)

    line = np.stack([cx + offset_px, cy], axis=1)
    return {
        "id": 1,
        "width_m": width_px * 0.014,
        "width_px": float(width_px),
        "centerline": [(float(x), float(y)) for x, y in line[::3]],
        "contour": contour,
    }


def test_recenter_moves_an_off_center_line_back_to_the_middle():
    band = _slanted_band(offset_px=12)  # 偏 12 px ≈ 0.17 m，在 0.22×带宽的修正上限内
    before = np.asarray(band["centerline"], float)
    _recenter_band_centerlines([band], 0.014, {"centerline_recenter": True})
    after = np.asarray(band["centerline"], float)

    assert after.shape == before.shape
    def center_at(y):
        return np.full_like(np.asarray(y, dtype=float), 300.0)

    error_before = float(np.median(np.abs(before[:, 0] - center_at(before[:, 1]))))
    error_after = float(np.median(np.abs(after[:, 0] - center_at(after[:, 1]))))
    assert error_after < error_before * 0.6, "偏移没有被压回去"
    assert band.get("recenter_m", 0) > 0


def test_recenter_can_be_disabled_and_keeps_points_untouched():
    band = _slanted_band(offset_px=28)
    before = np.asarray(band["centerline"], float).copy()
    _recenter_band_centerlines([band], 0.014, {"centerline_recenter": False})
    assert np.allclose(np.asarray(band["centerline"], float), before)
