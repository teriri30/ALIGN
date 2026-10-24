# -*- coding: utf-8 -*-
"""人工标定线替换进作业区/候选目录（DECISION-060）。"""
import math

import work_band_selection as wbs


def _regions():
    return [{
        "region_id": 1,
        "kind": "wide",
        "work_lines": [[(0.0, 0.0), (0.0, 100.0)], [(10.0, 0.0), (10.0, 100.0)]],
        "work_line_source_ids": [5, 6],
    }]


def _catalog_for(regions):
    """按真实管线的写法补 selected_centerline_ids：与 work_lines 同序。"""
    catalog = wbs.build_centerline_catalog(regions, math.pi / 2)
    by_line = sorted(catalog["items"], key=lambda item: item["region_line_index"])
    regions[0]["selected_centerline_ids"] = [item["id"] for item in by_line]
    return catalog


def test_only_the_labelled_band_is_replaced_and_ids_stay_stable():
    regions = _regions()
    catalog = _catalog_for(regions)
    target = next(
        item for item in catalog["items"] if 6 in (item.get("source_band_ids") or [])
    )
    replacement = [(12.5, 0.0), (12.5, 100.0)]
    updated, updated_catalog, report = wbs.apply_operator_centerlines(
        regions, catalog, {"band:6": {"points": replacement, "record_id": "op-0001"}},
        math.pi / 2,
    )
    assert report["applied"][0]["id"] == target["id"]
    patched = next(
        item for item in updated_catalog["items"] if item["id"] == target["id"]
    )
    assert patched["centerline"] == replacement
    assert patched["line_source"] == "operator"
    # 另一条线没被"顺手"改掉
    other = next(
        item for item in updated_catalog["items"] if item["id"] != target["id"]
    )
    assert other["centerline"] == [(0.0, 0.0), (0.0, 100.0)]
    # 编号与顺序不因换线而变
    assert [item["id"] for item in updated_catalog["items"]] == [
        item["id"] for item in catalog["items"]
    ]
    assert updated[0]["work_lines"][1] == replacement
    assert updated[0]["work_lines"][0] == [(0.0, 0.0), (0.0, 100.0)]


def test_unmatched_override_is_reported_instead_of_silently_ignored():
    regions = _regions()
    catalog = _catalog_for(regions)
    _updated, _updated_catalog, report = wbs.apply_operator_centerlines(
        regions, catalog, {"band:99": {"points": [(1.0, 0.0), (1.0, 100.0)]}},
        math.pi / 2,
    )
    assert report["applied"] == []
    assert report["unmatched"] == ["band:99"]


def test_empty_overrides_is_a_no_op():
    regions = _regions()
    catalog = _catalog_for(regions)
    updated, updated_catalog, report = wbs.apply_operator_centerlines(
        regions, catalog, {}, math.pi / 2
    )
    assert report["applied"] == []
    assert updated[0]["work_lines"] == regions[0]["work_lines"]
    assert updated_catalog["items"][0]["centerline"] == catalog["items"][0]["centerline"]


def test_labelling_one_swath_does_not_drag_its_siblings_from_the_same_band():
    """等距区把一条带拆成三股：标一股不能把另外两股也改掉。"""
    regions = [{
        "region_id": 2,
        "kind": "uniform_zone",
        "work_lines": [
            [(0.0, 0.0), (0.0, 100.0)],
            [(3.0, 0.0), (3.0, 100.0)],
            [(6.0, 0.0), (6.0, 100.0)],
        ],
        "work_line_source_ids": [1, 1, 1],
    }]
    catalog = _catalog_for(regions)
    target = sorted(catalog["items"], key=lambda item: item["region_line_index"])[1]
    replacement = [(3.4, 0.0), (3.4, 100.0)]
    _updated, updated_catalog, report = wbs.apply_operator_centerlines(
        regions, catalog, {target["id"]: {"points": replacement}}, math.pi / 2
    )
    assert [entry["id"] for entry in report["applied"]] == [target["id"]]
    patched = [item for item in updated_catalog["items"] if item.get("line_source") == "operator"]
    assert len(patched) == 1
    assert patched[0]["id"] == target["id"]


def test_band_only_override_is_applied_to_a_single_line():
    """没有精确中心的描点线只能兜底匹配一次，不能套到整片作业线上。"""
    regions = [{
        "region_id": 3,
        "kind": "uniform_zone",
        "work_lines": [[(0.0, 0.0), (0.0, 100.0)], [(3.0, 0.0), (3.0, 100.0)]],
        "work_line_source_ids": [9, 9],
    }]
    catalog = _catalog_for(regions)
    _updated, _updated_catalog, report = wbs.apply_operator_centerlines(
        regions, catalog, {"band:9": {"points": [(0.2, 0.0), (0.2, 100.0)]}},
        math.pi / 2,
    )
    assert len(report["applied"]) == 1
