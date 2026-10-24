# -*- coding: utf-8 -*-
"""用人工标定线评价/挑选中心线候选（DECISION-060）。"""
import numpy as np

import centerline_calibration as calibration


MPP = 0.01  # 1 cm/px


def _strip_case(offset_px=15.0):
    """1 m 宽的种植带，算法线偏 +15 cm，人工线在正中。"""
    mask = np.zeros((400, 1200), dtype=np.uint8)
    mask[60:340, 150:250] = 255
    ys = np.linspace(80, 320, 40)
    band = {
        "id": 3,
        "band_type": "wide",
        "width_m": 1.0,
        "profile_width_m": 1.0,
        "centerline": [[200.0 + offset_px, float(y)] for y in ys],
        "contour": [[150, 60], [250, 60], [250, 340], [150, 340]],
    }
    truth = [[200.0, float(y)] for y in ys]
    return mask, band, truth


def test_mask_run_mid_finds_the_strip_center_when_the_algorithm_line_is_off():
    mask, band, truth = _strip_case()
    candidates = calibration.candidate_centerlines(band, mask, MPP)
    current = calibration.measure(candidates["current"], truth, MPP)
    run_mid = calibration.measure(candidates["mask_run_mid"], truth, MPP)
    assert current["elat_mean_cm"] == 15.0
    assert run_mid["elat_mean_cm"] < 2.0


def test_contour_chord_mid_matches_the_operator_line_on_a_clean_band():
    mask, band, truth = _strip_case()
    candidates = calibration.candidate_centerlines(band, mask, MPP)
    chord = calibration.measure(candidates["contour_chord_mid"], truth, MPP)
    assert chord["elat_mean_cm"] < 1.0


def test_evaluate_band_picks_the_candidate_closest_to_the_operator():
    mask, band, truth = _strip_case()
    record = calibration.evaluate_band(band, truth, mask, MPP)
    assert record["best_candidate"] in {"mask_run_mid", "contour_chord_mid"}
    assert record["best_elat_mean_cm"] < 2.0
    assert record["improvement_cm"] > 10.0
    assert record["offset_profile"]["classification"]["kind"] == "uniform"


def test_evaluate_band_reports_off_band_when_the_operator_moved_to_the_next_band():
    mask, band, _truth = _strip_case()
    ys = np.linspace(80, 320, 40)
    far = [[200.0 - 0.9 / MPP, float(y)] for y in ys]  # 再往左 0.9 m：另一条带
    record = calibration.evaluate_band(band, far, mask, MPP)
    assert record["offset_profile"]["classification"]["kind"] == "off_band"


def test_pre_recenter_candidate_keeps_metric_units():
    """重压前中心线是像素坐标，候选偏移必须换成米，否则会放大 1/mpp 倍。"""
    mask, band, truth = _strip_case()
    pre = [[x - 0.05 / MPP, y] for x, y in band["centerline"]]  # 往 -0.05 m 挪
    band = dict(band, centerline_pre_recenter=pre)
    candidates = calibration.candidate_centerlines(band, mask, MPP)
    metrics = calibration.measure(candidates["pre_recenter"], pre, MPP)
    assert metrics["elat_mean_cm"] < 1.0
    current = calibration.measure(candidates["pre_recenter"], truth, MPP)
    assert 9.0 < current["elat_mean_cm"] < 11.0  # 15 cm - 5 cm = 10 cm


def test_case_bundle_accepts_band_id_key_from_the_desktop_export():
    mask, band, truth = _strip_case()
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    np.savez_compressed(path, mask=mask)
    try:
        case = {
            "schema": calibration.SCHEMA,
            "case_id": "case-1",
            "tif_path": "field.tif",
            "mpp": MPP,
            "mask_path": path,
            "bands": [{
                "band_id": 3,
                "band_type": "wide",
                "width_m": 1.0,
                "centerline": band["centerline"],
                "contour": band["contour"],
            }],
            "operator_lines": [{"band_id": 3, "points": truth}],
        }
        evaluated = calibration.evaluate_case(case)
        assert evaluated["bands"][0]["band_id"] == 3
        assert evaluated["bands"][0]["best_candidate"]
    finally:
        os.unlink(path)


def test_summary_ranks_candidates_and_reports_the_best_per_bucket():
    records = []
    for band_id in (1, 2, 3, 4):
        records.append({
            "band_id": band_id,
            "band_type": "wide",
            "width_m": 1.0,
            "best_candidate": "mask_run_mid" if band_id != 4 else "current",
            "best_elat_mean_cm": 1.0,
            "candidates": {
                "current": {"valid": True, "elat_mean_cm": 12.0 if band_id != 4 else 1.0},
                "mask_run_mid": {"valid": True, "elat_mean_cm": 1.0 if band_id != 4 else 9.0},
            },
        })
    summary = calibration.summarize(records)
    assert summary["bands_evaluated"] == 4
    assert summary["candidate_summary"]["mask_run_mid"]["wins"] == 3
    assert summary["policy"]["wide"]["candidate"] == "mask_run_mid"
    assert 0.0 <= summary["policy"]["wide"]["leave_one_out_hit_rate"] <= 1.0


def test_report_markdown_states_that_off_band_is_a_segmentation_problem():
    report = {
        "generated_at": "2026-10-24T00:00:00",
        "cases": [{"case_id": "c1"}],
        "summary": {"bands_total": 1, "bands_evaluated": 1, "candidate_summary": {}, "policy": {}},
        "problem_breakdown": {"off_band": 1},
        "bands": [{
            "band_id": 3, "band_type": "wide", "width_m": 1.0,
            "best_candidate": "mask_run_mid", "best_elat_mean_cm": 1.0,
            "candidates": {"current": {"valid": True, "elat_mean_cm": 12.0}},
            "offset_profile": {"classification": {"kind": "off_band"}},
        }],
    }
    text = calibration.render_markdown(report)
    assert "off_band" in text
    assert "分割问题" in text
