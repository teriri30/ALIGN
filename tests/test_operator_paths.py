# -*- coding: utf-8 -*-
"""人工标定作业线的数据契约与几何换算（DECISION-060）。"""
import numpy as np

import operator_paths


MPP = 0.01  # 1 cm/px


def _line(x=100.0, y0=100.0, y1=6100.0, count=60):
    return [(x, float(y)) for y in np.linspace(y0, y1, count)]


def test_uniform_edit_shifts_every_station_by_the_same_offset():
    base = _line()
    moved = operator_paths.path_from_edits(
        base, [{"kind": "uniform", "offset_m": -0.12}], MPP, width_m=1.0
    )
    array = np.asarray(moved)
    # 基准线沿 +y，法向指向 -x；偏移 -0.12 m 就是往 +x 挪 12 cm
    assert np.allclose(array[:, 0], 100.0 + 0.12 / MPP, atol=0.5)
    assert np.allclose(array[:, 1], np.asarray(base)[:, 1], atol=1e-6)


def test_local_edit_moves_the_middle_and_leaves_the_ends_alone():
    base = _line()
    moved = np.asarray(operator_paths.path_from_edits(
        base,
        [{"kind": "local", "arc_m": 30.0, "offset_m": 0.20, "sigma_m": 5.0}],
        MPP,
        width_m=1.0,
    ))
    middle = moved[len(moved) // 2]
    assert abs(middle[0] - (100.0 - 0.20 / MPP)) < 2.0
    assert abs(moved[0][0] - 100.0) < 0.5
    assert abs(moved[-1][0] - 100.0) < 0.5


def test_signed_offsets_recovers_a_known_lateral_shift():
    base = _line()
    shifted = [(x - 0.10 / MPP, y) for x, y in base]
    arc, offsets = operator_paths.signed_offsets(base, shifted, step=0.25 / MPP)
    summary = operator_paths.profile_summary(offsets * MPP)
    assert abs(summary["median_m"] - 0.10) < 0.002
    assert abs(float(arc[-1]) * MPP - 60.0) < 0.1


def test_classification_separates_centered_uniform_off_band():
    centered = np.array([0.005, -0.006, 0.004, 0.002])
    uniform = np.full(20, 0.12)
    off_band = np.full(20, 0.62)
    assert operator_paths.classify_offset_profile(centered, 1.0)["kind"] == "centered"
    assert operator_paths.classify_offset_profile(uniform, 1.0)["kind"] == "uniform"
    assert operator_paths.classify_offset_profile(off_band, 1.0)["kind"] == "off_band"


def test_classification_flags_a_band_that_was_split_by_a_step():
    offsets = np.concatenate([np.full(15, 0.05), np.full(15, 0.32)])
    result = operator_paths.classify_offset_profile(offsets, 1.0)
    assert result["kind"] == "step"
    assert result["jump_m"] > 0.25


def test_upsert_replaces_the_same_band_and_keeps_the_record_id():
    payload = operator_paths.empty_payload("field.tif", {"mask_version": 21})
    first = operator_paths.new_record(
        _line(), centerline_id="centerline-003", source_band_ids=[29], width_m=1.0
    )
    assert operator_paths.upsert_record(payload, first) == "appended"
    second = operator_paths.new_record(
        _line(x=80.0), centerline_id="centerline-003", source_band_ids=[29], width_m=1.0
    )
    assert operator_paths.upsert_record(payload, second) == "replaced"
    assert len(payload["lines"]) == 1
    assert payload["lines"][0]["id"] == first["id"]


def test_planner_overrides_use_the_exact_centerline_identity():
    payload = operator_paths.empty_payload("field.tif", {"mask_version": 21})
    operator_paths.upsert_record(payload, operator_paths.new_record(
        _line(),
        centerline_id="centerline-003",
        legacy_id="work-band-003",
        source_band_ids=[29],
    ))
    overrides, report = operator_paths.planner_overrides(
        payload, mask_version=21, tif_path="field.tif"
    )
    assert report["applied"] == 1
    assert "centerline-003" in overrides
    assert "work-band-003" in overrides
    # 一条带可能被拆成多股作业线：有精确 id 时不得再用 band: 键去覆盖整片
    assert "band:29" not in overrides


def test_planner_overrides_fall_back_to_band_id_only_without_any_centerline_id():
    payload = operator_paths.empty_payload("field.tif", {"mask_version": 21})
    operator_paths.upsert_record(payload, operator_paths.new_record(
        _line(), source_band_ids=[29]
    ))
    overrides, report = operator_paths.planner_overrides(
        payload, mask_version=21, tif_path="field.tif"
    )
    assert report["applied"] == 1
    assert list(overrides) == ["band:29"]


def test_planner_overrides_refuse_a_calibration_from_another_mask_version():
    payload = operator_paths.empty_payload("field.tif", {"mask_version": 20})
    operator_paths.upsert_record(payload, operator_paths.new_record(
        _line(), centerline_id="centerline-003"
    ))
    overrides, report = operator_paths.planner_overrides(
        payload, mask_version=21, tif_path="field.tif"
    )
    assert overrides == {}
    assert report["reason"] == "mask_version_mismatch"
    assert report["stale"]


def test_reference_lines_can_be_converted_with_offset_and_band_binding():
    reference = {
        "schema": "centerline-reference-lines-v1",
        "lines": [{
            "id": "ref-0001",
            "band_id": "centerline-003",
            "points": [[110.0, 120.0], [110.0, 220.0]],
        }],
    }
    records, report = operator_paths.from_reference_lines(
        reference,
        offset_x=10.0,
        offset_y=20.0,
        band_lookup={"centerline-003": {
            "centerline_id": "centerline-003",
            "source_band_ids": [29],
            "width_m": 1.0,
        }},
    )
    assert report["converted"] == 1
    assert records[0]["centerline_id"] == "centerline-003"
    assert records[0]["source_band_ids"] == [29]
    assert records[0]["points"][0] == [100.0, 100.0]
    assert records[0]["mode"] == "trace"
