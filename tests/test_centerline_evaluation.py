# -*- coding: utf-8 -*-
"""中心线评价（reference_lines / centerline_evaluation）单元测试。

覆盖：参考线拟合与度量口径（Eθ / Elat / 重叠率 / 采样）、人工重复性、
参考线落膜质检、payload 校验与 upsert、匹配逻辑、评价包写出与图形渲染。
全部使用合成数据，不依赖真实影像与 Qt。
"""
import csv
import json
import math
import os
import zipfile

import numpy as np
import pytest

from reference_lines import (
    SCHEMA,
    empty_payload,
    fit_line,
    line_angle,
    angle_difference_deg,
    sample_along_line,
    polyline_distances,
    compare_to_polyline,
    compare_lines,
    line_inside_mask_fraction,
    normalize_payload,
    new_line_record,
    upsert_line,
)


def test_fit_line_is_exact_for_collinear_points():
    fit = fit_line([(0, 0), (10, 10), (20, 20), (30, 30)])
    assert math.degrees(fit["angle_rad"]) == pytest.approx(45.0, abs=1e-6)
    assert fit["residual_rms_px"] == pytest.approx(0.0, abs=1e-9)
    assert fit["p1"] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert fit["p2"] == pytest.approx([30.0, 30.0], abs=1e-6)


def test_fit_line_averages_station_noise():
    rng = np.random.default_rng(0)
    base = np.linspace(0.0, 100.0, 11)
    pts = np.column_stack([base, rng.normal(0.0, 0.5, size=len(base))])
    fit = fit_line(pts)
    # 无向直线角在 [0, π) 内取值：接近水平时可能表示为接近 π 的等价位角。
    assert angle_difference_deg(fit["angle_rad"], 0.0) < 5.0
    assert fit["residual_rms_px"] < 1.0


def test_angle_difference_wraps_to_sharp_angle():
    assert angle_difference_deg(0.0, math.radians(1.0)) == pytest.approx(1.0, abs=1e-6)
    assert angle_difference_deg(math.radians(2.0), math.radians(179.0)) == pytest.approx(3.0, abs=1e-6)


def test_polyline_distance_endpoint_and_segment():
    dist = polyline_distances([(50, 8), (-10, 0), (110, 6)], [(0, 0), (100, 0)])
    assert dist[0] == pytest.approx(8.0)
    assert dist[1] == pytest.approx(10.0)
    assert dist[2] == pytest.approx(math.hypot(10.0, 6.0))


def test_sample_along_line_spacing_and_endpoints():
    samples = sample_along_line((0, 0), (100, 0), 10.0)
    assert len(samples) == 11
    assert samples[1] == pytest.approx((10.0, 0.0), abs=1e-9)
    assert samples[-1] == pytest.approx((100.0, 0.0), abs=1e-9)


def test_compare_known_offset_and_angle():
    ref = fit_line([(0, 100), (500, 100)])
    target = np.column_stack([np.linspace(0.0, 500.0, 101), np.full(101, 95.0)])
    cmp = compare_to_polyline(ref, target, 0.01, sample_spacing_m=1.0)
    assert cmp["elat_mean_cm"] == pytest.approx(5.0, abs=1e-6)
    assert cmp["elat_max_cm"] == pytest.approx(5.0, abs=1e-6)
    assert cmp["theta_deg"] == pytest.approx(0.0, abs=1e-9)
    assert cmp["overlap_ratio"] == pytest.approx(1.0, abs=1e-9)

    angle = math.radians(2.0)
    ts = np.linspace(0.0, 500.0, 101)
    rotated = np.column_stack([ts * math.cos(angle), 100.0 + ts * math.sin(angle)])
    cmp2 = compare_to_polyline(ref, rotated, 0.01, sample_spacing_m=1.0)
    assert cmp2["theta_deg"] == pytest.approx(2.0, abs=0.01)
    assert cmp2["elat_mean_cm"] > 1.0


def test_compare_overlap_ratio_for_truncated_target():
    ref = fit_line([(0, 100), (500, 100)])
    target = np.column_stack([np.linspace(0.0, 250.0, 51), np.full(51, 100.0)])
    cmp = compare_to_polyline(ref, target, 0.01, sample_spacing_m=1.0)
    assert cmp["overlap_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert cmp["elat_mean_cm"] == pytest.approx(75.0, abs=1e-6)
    assert cmp["elat_max_cm"] == pytest.approx(250.0, abs=1e-6)


def test_compare_lines_noise_floor():
    ref_a = fit_line([(0, 0), (100, 0)])
    ref_b = fit_line([(0, 3), (100, 3)])
    cmp = compare_lines(ref_a, ref_b, 0.01, sample_spacing_m=1.0)
    assert cmp["elat_mean_cm"] == pytest.approx(3.0, abs=1e-6)
    assert cmp["elat_max_cm"] == pytest.approx(3.0, abs=1e-6)
    assert cmp["theta_deg"] == pytest.approx(0.0, abs=1e-9)


def test_line_inside_mask_fraction():
    mask = np.zeros((50, 60), dtype=np.uint8)
    mask[10:20, :50] = 255
    inside = fit_line([(2, 15), (48, 15)])
    outside = fit_line([(2, 35), (48, 35)])
    assert line_inside_mask_fraction(inside, mask, 5.0) == pytest.approx(1.0)
    assert line_inside_mask_fraction(outside, mask, 5.0) == pytest.approx(0.0)


def test_payload_upsert_replace_and_normalize():
    payload = empty_payload("T:/x.tif")
    record = new_line_record([(0, 0), (10, 10), (20, 21)], band_id="centerline-001", pass_index=1)
    assert upsert_line(payload, record) == "appended"
    assert record["id"] == "ref-0001"

    record2 = new_line_record([(0, 0), (10, 10)], band_id="centerline-001", pass_index=1)
    assert upsert_line(payload, record2) == "replaced"
    assert len(payload["lines"]) == 1
    assert payload["lines"][0]["id"] == "ref-0001"

    record3 = new_line_record([(0, 0), (10, 10)], band_id="centerline-001", pass_index=2)
    assert upsert_line(payload, record3) == "appended"
    assert len(payload["lines"]) == 2

    data = json.loads(json.dumps(payload, ensure_ascii=False))
    normalized = normalize_payload(data)
    assert normalized["schema"] == SCHEMA
    assert len(normalized["lines"]) == 2
    assert normalized["lines"][0]["fit"]["point_count"] == 2

    bad = dict(payload)
    bad["lines"] = list(payload["lines"]) + [{"id": "broken", "points": [[1, 1]]}]
    normalized_bad = normalize_payload(json.loads(json.dumps(bad, ensure_ascii=False)))
    assert len(normalized_bad["lines"]) == 2  # 单点线被跳过

    with pytest.raises(ValueError):
        normalize_payload({"schema": "other-schema", "lines": []})


def test_evaluate_reference_lines_matching_and_metrics():
    from centerline_evaluation import evaluate_reference_lines

    cand_a = {
        "id": "A", "band_type": "wide",
        "points": np.column_stack([np.linspace(0.0, 100.0, 51), np.full(51, 100.0)]),
    }
    cand_b = {
        "id": "B", "band_type": "narrow",
        "points": np.column_stack([np.linspace(0.0, 100.0, 51), np.full(51, 200.0)]),
    }
    lines = [
        {"id": "r1", "pass": 1, "points": [[0.0, 101.0], [100.0, 101.0]]},
        {"id": "r2", "pass": 1, "points": [[0.0, 201.0], [100.0, 201.0]]},
        {"id": "r3", "pass": 1, "points": [[0.0, 150.0], [100.0, 150.0]]},
    ]
    rows = evaluate_reference_lines(lines, [cand_a, cand_b], 0.01, match_max_distance_m=0.4)
    assert rows[0]["band_id"] == "A"
    assert rows[0]["elat_mean_cm"] == pytest.approx(1.0, abs=0.01)
    assert rows[0]["theta_deg"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["overlap_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert rows[1]["band_id"] == "B"
    assert rows[2]["band_id"] is None
    assert "未匹配" in rows[2]["note"]


def test_build_summary_aggregates():
    from centerline_evaluation import build_summary

    rows = [
        {"line_id": "r1", "theta_deg": 0.1, "elat_mean_cm": 2.0, "elat_max_cm": 5.0},
        {"line_id": "r2", "theta_deg": 0.3, "elat_mean_cm": 4.0, "elat_max_cm": 9.0},
        {"line_id": "r3", "theta_deg": None, "elat_mean_cm": None, "elat_max_cm": None},
    ]
    noise = [{
        "band_id": "A", "line_id_pass1": "r1", "line_id_pass2": "r1b",
        "theta_deg": 0.5, "elat_mean_cm": 2.0, "elat_max_cm": 6.0,
    }]
    summary = build_summary(
        rows, noise,
        inputs={"mpp": 0.01}, cache_dir="c", tif_path="t",
        sample_spacing_m=1.0, match_max_distance_m=0.8, station_spacing_hint_m=5.0,
    )
    assert summary["matched_line_count"] == 2
    assert summary["unmatched_line_ids"] == ["r3"]
    assert summary["theta_mean_deg"] == pytest.approx(0.2, abs=1e-6)
    assert summary["elat_mean_cm"] == pytest.approx(3.0, abs=1e-6)
    assert summary["elat_max_cm"] == pytest.approx(9.0, abs=1e-6)
    assert summary["noise_floor"]["pair_count"] == 1


def test_write_evaluation_package_tables_and_zip(tmp_path):
    from centerline_evaluation import write_evaluation_package

    rows = [{
        "line_id": "ref-0001", "band_id": "A", "band_type": "wide", "pass": 1,
        "station_count": 10, "match_distance_cm": 0.8, "theta_deg": 0.2,
        "elat_mean_cm": 3.5, "elat_max_cm": 8.1, "overlap_ratio": 1.0,
        "sample_count": 60, "note": "",
    }]
    summary = {
        "version": "centerline-eval-v1", "generated_at": "now", "tif_path": "T:/f.tif",
        "mask_regularization_version": 19, "sample_spacing_m": 1.0,
        "match_max_distance_m": 0.8, "line_count": 1, "matched_line_count": 1,
        "unmatched_line_ids": [], "theta_mean_deg": 0.2, "elat_mean_cm": 3.5,
        "elat_max_cm": 8.1, "elat_mean_of_max_cm": 8.1, "noise_floor": None,
    }
    out_dir = str(tmp_path / "pack")
    zip_paths = write_evaluation_package(
        out_dir,
        summary=summary, per_line_rows=rows, noise_rows=[],
        ref_payload=empty_payload("T:/f.tif"),
        cache_dir=str(tmp_path), mask_local=None, local_lines=[], candidates=[],
        offset=(0, 0), make_figures=False,
    )
    assert os.path.isfile(zip_paths[0])
    with zipfile.ZipFile(zip_paths[0]) as bundle:
        names = bundle.namelist()
    assert {"README.md", "per_line.csv", "summary.json", "reference_lines.json"} <= set(names)
    with open(os.path.join(out_dir, "per_line.csv"), encoding="utf-8-sig") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["elat_mean_cm"] == "3.5"
    assert parsed[0]["band_id"] == "A"


def test_render_figures_with_synthetic_data(tmp_path):
    from centerline_evaluation import render_boxplot_png, render_overlay_png

    rows = [
        {"elat_mean_cm": 2.0, "theta_deg": 0.1},
        {"elat_mean_cm": 5.0, "theta_deg": 0.3},
        {"elat_mean_cm": 9.0, "theta_deg": 0.8},
    ]
    box_path = tmp_path / "box.png"
    render_boxplot_png(str(box_path), rows)
    assert box_path.stat().st_size > 1000

    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[25:35, :] = 255
    line = {"fit": fit_line([(2, 30), (78, 30)]), "points": [[2, 30], [40, 30], [78, 30]]}
    candidate = {"points": np.column_stack([np.linspace(0.0, 79.0, 40), np.full(40, 30.0)])}
    overlay_path = tmp_path / "overlay.png"
    render_overlay_png(
        str(overlay_path),
        cache_dir=str(tmp_path),
        mask_local=mask,
        local_lines=[line],
        candidates=[candidate],
        offset=(0, 0),
    )
    assert overlay_path.stat().st_size > 1000


def test_cache_reference_lines_sidecar_roundtrip(tmp_path, monkeypatch):
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    tif = str(tmp_path / "field.tif")
    payload = empty_payload(tif)
    record = new_line_record([(0.0, 0.0), (10.0, 10.0)], band_id="centerline-001", pass_index=1)
    upsert_line(payload, record)
    path = cache.save_reference_lines(tif, payload)
    assert os.path.isfile(path)
    assert path.endswith("reference_lines.json")
    loaded = cache.load_reference_lines(tif)
    assert loaded["schema"] == payload["schema"]
    assert len(loaded["lines"]) == 1
    assert cache.project_cache_dir(tif) == os.path.dirname(path)
