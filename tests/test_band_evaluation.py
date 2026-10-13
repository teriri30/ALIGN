# -*- coding: utf-8 -*-
"""band_evaluation 单元测试（行带识别与提线评价，DECISION-029）。

覆盖：通道几何余量数学、多边形 IoU 与真值匹配、标签归一化。
真实数据端到端由 CLI 冒烟覆盖（不是本文件的职责）。
"""
import math

import numpy as np
import pytest

import band_evaluation as be


def _band(band_id, band_type, width, pos_y, half_len=3.0):
    contour = np.asarray([
        [-half_len, pos_y - width / 2],
        [half_len, pos_y - width / 2],
        [half_len, pos_y + width / 2],
        [-half_len, pos_y + width / 2],
    ], dtype=np.float64)
    return {
        "id": band_id,
        "band_type": band_type,
        "width_m": width,
        "centroid": (0.0, float(pos_y)),
        "contour": contour,
    }


SPECS = {"track_width_m": 0.35, "track_gauge_m": 1.3,
         "cutter_width_m": 2.0, "turn_radius_m": 0.9}


def test_channel_margins_match_document_geometry():
    """宽 0.78m + 通道 0.45m + 窄 0.30m：内侧 +8.5cm、外侧 +1.5cm（交底书 +8.7/+1.3 同量级）。"""
    bands = [
        _band(1, "narrow", 0.30, -0.99),
        _band(2, "wide", 0.78, 0.0),
        _band(3, "narrow", 0.30, 0.99),
    ]
    records, summary = be.compute_channel_geometry(bands, 0.0, 1.0, SPECS)
    record = records[2]
    assert record["containment"] is True
    for side in ("left", "right"):
        assert record[side]["clear_m"] == pytest.approx(0.45, abs=1e-6)
        assert record[side]["margin_inner_m"] == pytest.approx(0.085, abs=1e-6)
        assert record[side]["margin_outer_m"] == pytest.approx(0.015, abs=1e-6)
    assert summary["contained_band_count"] == 1
    assert summary["contained_pct"] == pytest.approx(100.0)
    assert summary["channel_clear_median_m"] == pytest.approx(0.45)


def test_channel_containment_fails_when_channel_too_narrow():
    """通道净宽 0.36m < 履带外缘需求：外侧余量为负 -> 不满足。"""
    bands = [
        _band(1, "narrow", 0.30, -0.90),
        _band(2, "wide", 0.78, 0.0),
        _band(3, "narrow", 0.30, 0.90),
    ]
    records, _ = be.compute_channel_geometry(bands, 0.0, 1.0, SPECS)
    record = records[2]
    assert record["containment"] is False
    assert record["left"]["margin_outer_m"] == pytest.approx(0.39 + 0.36 - 0.825, abs=1e-6)


def test_channel_geometry_uses_normal_projection():
    """主方向 90° 时法向为 (-1, 0)：横向位置取 x 质心投影。"""
    bands = [
        _band(1, "narrow", 0.30, -0.99),
        _band(2, "wide", 0.78, 0.0),
        _band(3, "narrow", 0.30, 0.99),
    ]
    for band in bands:
        band["centroid"] = (band["centroid"][1], 0.0)  # x=pos, y=0
    records, _ = be.compute_channel_geometry(bands, math.pi / 2, 1.0, SPECS)
    assert records[2]["containment"] is True


def test_polygon_iou_and_greedy_match():
    truth_points = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    band_same = np.asarray([[1, 0], [11, 0], [11, 10], [1, 10]], dtype=float)
    band_far = np.asarray([[100, 100], [110, 100], [110, 110], [100, 110]], dtype=float)
    iou_same = be._polygon_iou(truth_points, band_same)
    # 理论 IoU=9/11≈0.818；栅格化（fillPoly 像素边界取整）会有小幅离散化偏差
    assert 0.80 <= iou_same <= 0.86
    iou_far = be._polygon_iou(truth_points, band_far)
    assert iou_far == 0.0
    truths = [{"truth_id": "t1", "label": "宽行带", "band_type": "wide",
               "points": truth_points.tolist()}]
    bands = [{"id": 1, "band_type": "wide", "contour": band_same},
             {"id": 2, "band_type": "narrow", "contour": band_far}]
    rows, unmatched_truth, unmatched_band = be.match_truth_to_bands(truths, bands, 0.5)
    assert rows[0]["matched_band_id"] == 1
    assert rows[0]["type_correct"] is True
    assert unmatched_truth == []
    assert unmatched_band == [2]


def test_match_respects_iou_threshold():
    truth_points = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    band_offset = np.asarray([[6, 0], [16, 0], [16, 10], [6, 10]], dtype=float)
    truths = [{"truth_id": "t1", "label": "窄", "band_type": "narrow",
               "points": truth_points.tolist()}]
    bands = [{"id": 1, "band_type": "narrow", "contour": band_offset}]
    rows, unmatched_truth, _ = be.match_truth_to_bands(truths, bands, 0.5)
    # IoU = 4/16 = 0.25 < 0.5
    assert rows[0]["matched_band_id"] is None
    assert unmatched_truth == ["t1"]


def test_normalize_band_type_labels():
    assert be.normalize_band_type("宽行带") == "wide"
    assert be.normalize_band_type("窄行带") == "narrow"
    assert be.normalize_band_type("Wide") == "wide"
    assert be.normalize_band_type("narrow band") == "narrow"
    assert be.normalize_band_type("作物行带") is None
    assert be.normalize_band_type("") is None

