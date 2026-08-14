import pytest

from machine_recommendation import recommend_machine_dimensions


def _band(index, y_px, width_m=0.20):
    return {
        "id": index,
        "centroid": (100.0, float(y_px)),
        "width_m": width_m,
        "classification_width_m": width_m,
    }


def test_recommendation_derives_track_limit_and_gauge_from_wide_corridors():
    mask_result = {
        "main_angle": 0.0,
        "all_bands": [
            _band(1, 20), _band(2, 90), _band(3, 180),
            _band(4, 250), _band(5, 340),
        ],
        "diagnostics": {
            "meters_per_pixel": 0.01,
            "headland_corridor": {
                "candidates": [
                    {"decision": "accepted", "depth_m": 2.6},
                    {"decision": "accepted", "depth_m": 2.8},
                ]
            },
        },
    }
    current = {
        "cutter_width_m": 2.15,
        "track_width_m": 0.40,
        "track_gauge_m": 1.70,
        "wheelbase_m": 2.40,
        "track_length_m": 2.10,
        "turn_radius_m": 1.00,
    }

    result = recommend_machine_dimensions(mask_result, current)

    assert "validated" not in result
    assert result["method"] == "mask_geometry_heuristic_v2"
    assert result["limits"]["wide_corridor_clear_m"] == pytest.approx(0.7)
    assert result["limits"]["track_width_upper_m"] == pytest.approx(0.55)
    assert result["params"]["track_width_m"] == pytest.approx(0.55)
    assert result["limits"]["track_gauge_target_m"] == pytest.approx(1.6)
    assert result["params"]["track_gauge_m"] == pytest.approx(1.6)
    assert result["limits"]["wheelbase_upper_guide_m"] == pytest.approx(2.0)
    assert result["params"]["wheelbase_m"] == pytest.approx(2.0)
    assert result["limits"]["matched_cutter_width_m"] == pytest.approx(2.35)
    assert result["params"]["cutter_width_m"] == pytest.approx(2.35)
    assert "自动候选割幅" in result["message"]
    assert "不含坡度、土壤或重心模型" in result["message"]
    assert "基于当前影像中的行带宽度和田头空间" in result["message"]
    assert "没有 DEM" not in result["message"]


def test_recommendation_scales_generated_cutter_width_with_machine_body():
    mask_result = {
        "main_angle": 0.0,
        "all_bands": [
            _band(1, 20), _band(2, 80),
        ],
        "diagnostics": {"meters_per_pixel": 0.01},
    }
    current = {
        "cutter_width_m": 2.50,
        "track_width_m": 0.40,
        "track_gauge_m": 1.00,
        "wheelbase_m": 1.80,
        "track_length_m": 1.70,
        "turn_radius_m": 0.90,
    }

    result = recommend_machine_dimensions(mask_result, current)

    outer_track_width = (
        result["params"]["track_gauge_m"] + result["params"]["track_width_m"]
    )
    assert result["params"]["cutter_width_m"] < 2.0
    assert result["params"]["cutter_width_m"] > outer_track_width
    assert result["params"]["cutter_width_m"] == pytest.approx(
        result["limits"]["matched_cutter_width_m"]
    )


def test_recommendation_does_not_invent_limits_without_geometry_evidence():
    current = {
        "cutter_width_m": 2.15,
        "track_width_m": 0.45,
        "track_gauge_m": 1.60,
        "wheelbase_m": 1.80,
        "track_length_m": 1.70,
        "turn_radius_m": 0.90,
    }

    result = recommend_machine_dimensions({}, current)

    assert result["params"] == current
    assert result["limits"]["track_width_upper_m"] is None
    assert result["limits"]["matched_cutter_width_m"] is None
    assert result["limits"]["wheelbase_upper_guide_m"] is None
    assert result["evidence_quality"] == "insufficient"
    assert "无法计算" in result["message"]
