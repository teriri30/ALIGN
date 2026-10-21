import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow


def _region(region_id, ys, *, kind="wide_narrow_rows", source_ids=None):
    source_ids = list(source_ids or range(1, len(ys) + 1))
    return {
        "region_id": region_id,
        "kind": kind,
        "line_source": "four_row_band_centerlines",
        "band_ids": source_ids,
        "work_line_source_ids": source_ids,
        "work_lines": [[(10.0, float(y)), (90.0, float(y))] for y in ys],
    }


def test_catalog_numbers_work_bands_by_field_normal_position():
    from work_band_selection import build_work_band_catalog

    catalog = build_work_band_catalog(
        [_region("right", [30.0, 10.0], source_ids=[30, 10]),
         _region("left", [20.0], source_ids=[20])],
        main_angle=0.0,
    )

    assert [item["id"] for item in catalog["items"]] == [
        "work-band-001", "work-band-002", "work-band-003"
    ]
    assert [item["source_band_ids"] for item in catalog["items"]] == [
        [10], [20], [30]
    ]
    assert [item["label"] for item in catalog["items"]] == [
        "作业带 1", "作业带 2", "作业带 3"
    ]


def test_selection_accepts_arbitrary_multi_select():
    from work_band_selection import build_work_band_catalog, resolve_work_band_selection

    catalog = build_work_band_catalog([_region("field", [10.0, 20.0, 30.0])], 0.0)

    selected, report = resolve_work_band_selection(
        catalog, ["work-band-001", "work-band-003"]
    )
    assert [item["id"] for item in selected] == [
        "work-band-001", "work-band-003"
    ]
    assert report["selection_mode"] == "multi"

    with pytest.raises(ValueError, match="不存在"):
        resolve_work_band_selection(catalog, ["work-band-999"])
    with pytest.raises(ValueError, match="至少选择"):
        resolve_work_band_selection(catalog, [])


def test_selected_catalog_items_filter_region_lines_without_reordering():
    from work_band_selection import (
        build_work_band_catalog,
        filter_work_regions,
        resolve_work_band_selection,
    )

    regions = [_region("field", [10.0, 20.0, 30.0], source_ids=[4, 6, 8])]
    catalog = build_work_band_catalog(regions, 0.0)
    selected, _report = resolve_work_band_selection(
        catalog, ["work-band-002", "work-band-003"]
    )

    filtered = filter_work_regions(regions, selected)

    assert len(filtered) == 1
    assert filtered[0]["work_lines"] == [
        [(10.0, 20.0), (90.0, 20.0)],
        [(10.0, 30.0), (90.0, 30.0)],
    ]
    assert filtered[0]["work_line_source_ids"] == [6, 8]


def test_apply_defaults_to_all_and_records_explicit_selection_scope():
    from work_band_selection import apply_work_band_selection

    regions = [_region("field", [10.0, 20.0, 30.0], source_ids=[4, 6, 8])]

    all_regions, catalog, default_report = apply_work_band_selection(
        regions, 0.0, selected_ids=None
    )
    assert len(all_regions[0]["work_lines"]) == 3
    assert default_report["selection_origin"] == "default_all"
    assert default_report["selected_ids"] == [
        item["id"] for item in catalog["items"]
    ]

    chosen_regions, _catalog, explicit_report = apply_work_band_selection(
        regions, 0.0, selected_ids=["work-band-002"]
    )
    assert chosen_regions[0]["work_lines"] == [[(10.0, 20.0), (90.0, 20.0)]]
    assert explicit_report["selection_origin"] == "operator"
    assert explicit_report["selection_mode"] == "single"


def test_app_state_captures_operator_work_band_selection_for_undo():
    from state import AppState

    state = AppState()
    state.work_band_catalog = {"algorithm_version": "v1", "items": [{"id": "work-band-001"}]}
    state.selected_work_band_ids = ["work-band-001"]
    state.work_band_selection_report = {"selection_origin": "operator"}

    snapshot = state.capture_system_snapshot("选择作业带")
    state.work_band_catalog = {}
    state.selected_work_band_ids = []
    state.work_band_selection_report = {}
    state._restore_snapshot(snapshot)

    assert state.selected_work_band_ids == ["work-band-001"]
    assert state.work_band_selection_report["selection_origin"] == "operator"


def test_selected_work_band_coverage_uses_cutter_width_and_support_mask():
    from work_band_selection import (
        apply_work_band_selection,
        rasterize_work_band_coverage,
    )

    regions = [_region("field", [10.0, 20.0, 30.0])]
    _filtered, catalog, _report = apply_work_band_selection(
        regions, 0.0, selected_ids=["work-band-002"]
    )
    support = np.full((40, 100), 255, dtype=np.uint8)
    support[:, :15] = 0

    coverage = rasterize_work_band_coverage(
        catalog,
        ["work-band-002"],
        shape=support.shape,
        cutter_width_m=4.0,
        meters_per_pixel=1.0,
        support_mask=support,
    )

    assert np.count_nonzero(coverage[18:23, 15:91]) > 0
    assert np.count_nonzero(coverage[8:13]) == 0
    assert np.count_nonzero(coverage[28:33]) == 0
    assert np.count_nonzero(coverage[:, :15]) == 0


def test_project_state_payload_persists_selection_without_dense_mask(monkeypatch, tmp_path):
    from cache import _build_project_state_payload
    from state import AppState

    tif = tmp_path / "field.tif"
    tif.write_bytes(b"test-source")
    state = AppState()
    state.source_sha256 = "source-hash"
    state.work_band_catalog = {
        "algorithm_version": "operator-work-band-catalog-v1",
        "items": [{"id": "work-band-001", "work_line": [(1.0, 2.0), (3.0, 2.0)]}],
    }
    state.selected_work_band_ids = ["work-band-001"]
    state.work_band_selection_report = {
        "selection_origin": "operator",
        "selected_ids": ["work-band-001"],
    }
    state.mask_result = {
        "processed_mask": np.ones((2, 2), dtype=np.uint8),
        "wide_bands": [],
    }
    monkeypatch.setattr("cache.source_identity", lambda _path: {"path": "field.tif"})

    payload = _build_project_state_payload(str(tif), state, "work_band_selection")

    assert payload["selected_work_band_ids"] == ["work-band-001"]
    assert payload["work_band_selection_report"]["selection_origin"] == "operator"
    assert payload["work_band_catalog"]["items"][0]["id"] == "work-band-001"
    assert "processed_mask" not in (payload["mask_result"] or {})


def test_legacy_work_band_cache_migrates_to_centerline_ids():
    from work_band_selection import migrate_cached_centerline_catalog

    catalog, id_map = migrate_cached_centerline_catalog({
        "algorithm_version": "operator-work-band-catalog-v1",
        "items": [{
            "id": "work-band-001",
            "sequence": 1,
            "label": "作业带 1",
            "work_line": [(1.0, 2.0), (3.0, 2.0)],
        }],
    })

    assert id_map == {"work-band-001": "centerline-001"}
    assert catalog["items"][0]["id"] == "centerline-001"
    assert catalog["items"][0]["legacy_id"] == "work-band-001"
    assert catalog["items"][0]["label"] == "中心线 1"
    assert catalog["items"][0]["centerline"] == [(1.0, 2.0), (3.0, 2.0)]


def test_planner_selection_binding_filters_lines_and_validation_target():
    from path_planner import _apply_operator_work_band_selection

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    class State:
        harvester_params = {"cutter_width_m": 4.0}

    regions = [_region("field", [10.0, 20.0, 30.0], source_ids=[4, 6, 8])]
    mask = np.full((40, 100), 255, dtype=np.uint8)

    lines, selected_regions, planned_mask, layout = (
        _apply_operator_work_band_selection(
            regions,
            {},
            0.0,
            mask,
            mask,
            geo=Geo(),
            state=State(),
            config={
                "selected_work_band_ids": ["work-band-002"],
                "planning_support_mask": mask,
            },
        )
    )

    assert lines == [[(10.0, 20.0), (90.0, 20.0)]]
    assert selected_regions[0]["selected_work_band_ids"] == ["work-band-002"]
    assert layout["planning_scope"] == "operator_selected_work_bands"
    assert layout["work_band_selection"]["selected_ids"] == ["work-band-002"]
    assert np.count_nonzero(planned_mask[8:13]) == 0
    assert np.count_nonzero(planned_mask[18:23]) > 0


def test_prepare_operator_work_band_plan_uses_final_planner_geometry():
    from path_planner import prepare_operator_work_band_plan

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 40.0), (0.0, 40.0)
        ]
        harvester_params = {
            "cutter_width_m": 4.0,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "turn_radius_m": 1.0,
        }

    bands = [
        {
            "id": band_id,
            "profile_width_m": 1.0,
            "width_m": 1.0,
            "band_type": "wide",
            "centerline": [(10.0, y), (90.0, y)],
            "contour": [(8.0, y - 2.0), (92.0, y - 2.0),
                        (92.0, y + 2.0), (8.0, y + 2.0)],
        }
        for band_id, y in ((4, 10.0), (6, 20.0), (8, 30.0))
    ]
    mask = np.full((40, 100), 255, dtype=np.uint8)

    prepared = prepare_operator_work_band_plan(
        bands,
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "validation_profile": "field_trial",
            "work_line_mode": "band_centerline",
            "selected_work_band_ids": ["work-band-002"],
            "planning_support_mask": mask,
        },
        all_bands=bands,
    )

    assert prepared["catalog"]["count"] == 3
    assert len(prepared["work_lines"]) == 1
    assert prepared["selection"]["selected_ids"] == ["work-band-002"]
    assert prepared["layout"]["planning_scope"] == "operator_selected_work_bands"


def test_centerline_catalog_generates_all_lines_with_checkbox_ids():
    from work_band_selection import build_centerline_catalog

    catalog = build_centerline_catalog(
        [_region("field", [30.0, 10.0, 20.0])], main_angle=0.0
    )

    assert catalog["algorithm_version"] == "operator-centerline-catalog-v1"
    assert [item["id"] for item in catalog["items"]] == [
        "centerline-001", "centerline-002", "centerline-003"
    ]
    assert [item["label"] for item in catalog["items"]] == [
        "中心线 1", "中心线 2", "中心线 3"
    ]
    assert all(item["centerline"] for item in catalog["items"])


def test_prepare_centerline_plan_filters_only_after_catalog_generation():
    from path_planner import prepare_operator_centerline_plan

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 40.0), (0.0, 40.0)
        ]
        harvester_params = {
            "cutter_width_m": 4.0,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "turn_radius_m": 1.0,
        }

    bands = [
        {
            "id": band_id,
            "profile_width_m": 1.0,
            "width_m": 1.0,
            "band_type": "wide",
            "centerline": [(10.0, y), (90.0, y)],
            "contour": [(8.0, y - 2.0), (92.0, y - 2.0),
                        (92.0, y + 2.0), (8.0, y + 2.0)],
        }
        for band_id, y in ((4, 10.0), (6, 20.0), (8, 30.0))
    ]
    mask = np.full((40, 100), 255, dtype=np.uint8)

    prepared = prepare_operator_centerline_plan(
        bands,
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "validation_profile": "field_trial",
            "work_line_mode": "band_centerline",
            "selected_centerline_ids": ["centerline-002"],
            "planning_support_mask": mask,
        },
        all_bands=bands,
    )

    assert prepared["catalog"]["count"] == 3
    assert len(prepared["work_lines"]) == 1
    assert prepared["selection"]["selected_ids"] == ["centerline-002"]
    assert prepared["layout"]["planning_scope"] == "operator_selected_centerlines"


def test_prepare_centerline_plan_excludes_lines_outside_field_boundary():
    from path_planner import prepare_operator_centerline_plan

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 40.0), (0.0, 40.0)
        ]
        harvester_params = {
            "cutter_width_m": 4.0,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "turn_radius_m": 1.0,
        }

    bands = [
        {
            "id": band_id,
            "profile_width_m": 1.0,
            "width_m": 1.0,
            "band_type": "wide",
            "centerline": [
                (110.0 if band_id == 8 else 10.0, y),
                (190.0 if band_id == 8 else 90.0, y),
            ],
            "contour": [
                (108.0 if band_id == 8 else 8.0, y - 2.0),
                (192.0 if band_id == 8 else 92.0, y - 2.0),
                (192.0 if band_id == 8 else 92.0, y + 2.0),
                (108.0 if band_id == 8 else 8.0, y + 2.0),
            ],
        }
        for band_id, y in ((4, 10.0), (6, 20.0), (8, 30.0))
    ]
    mask = np.full((40, 200), 255, dtype=np.uint8)

    prepared = prepare_operator_centerline_plan(
        bands,
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "validation_profile": "field_trial",
            "work_line_mode": "band_centerline",
            "planning_support_mask": mask,
        },
        all_bands=bands,
    )

    assert prepared["catalog"]["count"] == 2
    assert len(prepared["work_lines"]) == 2
    boundary_filter = prepared["layout"]["centerline_field_boundary_filter"]
    assert boundary_filter["excluded_count"] == 1
    assert boundary_filter["reason"] == "excluded_lines_outside_field_boundary"
    assert all(
        max(point[0] for point in item["centerline"]) <= 100.0
        for item in prepared["catalog"]["items"]
    )


def test_task_panel_removes_old_canvas_work_band_selector():
    from pyside6_app.panels import TaskPanel

    app = QApplication.instance() or QApplication([])
    panel = TaskPanel()
    panel.show()
    app.processEvents()

    assert not hasattr(panel, "select_work_bands_btn")
    assert not hasattr(panel, "work_band_selection_requested")
    panel.close()


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    return window


def _centerline_catalog_with_three_lines():
    return {
        "algorithm_version": "operator-centerline-catalog-v1",
        "items": [
            {
                "id": "centerline-001",
                "sequence": 1,
                "label": "中心线 1",
                "line_source": "four_row_band_centerlines",
                "centerline": [(10.0, 10.0), (110.0, 10.0)],
            },
            {
                "id": "centerline-002",
                "sequence": 2,
                "label": "中心线 2",
                "line_source": "four_row_band_centerlines",
                "centerline": [(10.0, 30.0), (110.0, 30.0)],
            },
            {
                "id": "centerline-003",
                "sequence": 3,
                "label": "中心线 3",
                "line_source": "four_row_band_centerlines",
                "centerline": [(10.0, 50.0), (110.0, 50.0)],
            },
        ],
    }


def test_task_panel_lists_each_centerline_with_a_checkbox():
    from pyside6_app.panels import CenterlineSelectionRow, TaskPanel

    app = QApplication.instance() or QApplication([])
    panel = TaskPanel()
    changes = []
    panel.centerline_selection_changed.connect(
        lambda line_id, checked: changes.append((line_id, checked))
    )
    panel.update_centerline_selection(
        _centerline_catalog_with_three_lines(),
        ["centerline-001", "centerline-002", "centerline-003"],
    )
    panel.show()
    app.processEvents()

    rows = panel.findChildren(CenterlineSelectionRow)
    assert len(rows) == 3
    rows[1].checkbox.setChecked(False)
    app.processEvents()
    assert changes == [("centerline-002", False)]
    assert rows[0].checkbox.isChecked() is True
    assert rows[1].checkbox.isChecked() is False
    panel.close()


def test_unchecked_centerline_is_hidden_from_canvas_overlay():
    window = _window()
    window.state.centerline_catalog = _centerline_catalog_with_three_lines()
    window.state.selected_centerline_ids = ["centerline-001", "centerline-003"]

    class Viewer:
        def __init__(self):
            self.lines = []
            self.labels = []

        def draw_polyline(self, points, **kwargs):
            self.lines.append((points, kwargs))

        def draw_text(self, *args, **kwargs):
            self.labels.append((args, kwargs))

    viewer = Viewer()
    window._draw_centerline_overlay(viewer)

    assert len(viewer.lines) == 2
    assert all("30.0" not in str(points) for points, _kwargs in viewer.lines)
    window.close()


def test_cached_centerline_catalog_drops_lines_outside_field_boundary():
    window = _window()
    window.state.mask_raw = np.full((80, 120), 255, dtype=np.uint8)
    window.state.field_boundary = [
        (0.0, 0.0), (110.0, 0.0), (110.0, 79.0), (0.0, 79.0)
    ]
    catalog = _centerline_catalog_with_three_lines()
    catalog["items"][-1]["centerline"] = [(110.0, 50.0), (119.0, 50.0)]
    window.state.safe_update(
        centerline_catalog=catalog,
        selected_centerline_ids=[
            "centerline-001", "centerline-002", "centerline-003"
        ],
        centerline_selection_report={
            "selection_origin": "default_all",
            "selected_ids": [
                "centerline-001", "centerline-002", "centerline-003"
            ],
        },
    )

    assert window._sanitize_centerline_state_to_field() is True
    assert [
        item["id"] for item in window.state.centerline_catalog["items"]
    ] == ["centerline-001", "centerline-002"]
    assert window.state.selected_centerline_ids == [
        "centerline-001", "centerline-002"
    ]
    assert (
        window.state.centerline_selection_report[
            "centerline_field_boundary_filter"
        ]["excluded_count"]
        == 1
    )
    window.close()


def test_checkbox_change_invalidates_path_and_updates_canonical_state():
    window = _window()
    catalog = _centerline_catalog_with_three_lines()
    window.state.safe_update(
        centerline_catalog=catalog,
        selected_centerline_ids=[
            "centerline-001", "centerline-002", "centerline-003"
        ],
        auto_path_planned=True,
    )

    window._on_centerline_selection_changed("centerline-002", False)

    assert window.state.selected_centerline_ids == [
        "centerline-001", "centerline-003"
    ]
    assert window.state.selected_work_band_ids == [
        "centerline-001", "centerline-003"
    ]
    assert window.state.auto_path_planned is False
    assert window.state.centerline_selection_report["selection_mode"] == "multi"
    window.close()


def test_checkbox_change_refreshes_only_centerline_overlay(monkeypatch):
    """A checkbox edit must not redraw the base image or mask evidence."""
    window = _window()
    catalog = _centerline_catalog_with_three_lines()
    window.state.safe_update(
        centerline_catalog=catalog,
        selected_centerline_ids=[
            "centerline-001", "centerline-002", "centerline-003"
        ],
        mask_result={"processed_mask": np.ones((80, 120), dtype=np.uint8)},
        auto_path_planned=True,
    )
    refreshed = []
    monkeypatch.setattr(
        window,
        "_refresh_centerline_selection_overlay",
        lambda: refreshed.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        window,
        "_restore_visual_from_state",
        lambda: pytest.fail("centerline checkbox rebuilt the full canvas"),
    )
    monkeypatch.setattr(
        window,
        "_draw_mask_evidence_layers",
        lambda *args, **kwargs: pytest.fail("centerline checkbox redrew mask evidence"),
    )

    window._on_centerline_selection_changed("centerline-002", False)

    assert refreshed == [True]
    assert window.state.selected_centerline_ids == [
        "centerline-001", "centerline-003"
    ]
    window.close()


def test_centerline_plan_keeps_lines_when_only_the_closing_loop_is_broken():
    """Regression for 生成作业线 -> 无有效中心线 (2026-10-21 field project)."""
    from path_planner import prepare_operator_centerline_plan

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        # Bow-tie closing loop: the extra vertices straddle the first corner, as
        # when the closing double click lands a few pixels past the first vertex.
        field_boundary = [
            (10.0, 10.0), (190.0, 10.0), (190.0, 90.0), (10.0, 90.0),
            (9.0, 12.0), (12.0, 8.0),
        ]
        harvester_params = {
            "cutter_width_m": 4.0,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "turn_radius_m": 1.0,
        }

    bands = [
        {
            "id": band_id,
            "profile_width_m": 1.0,
            "width_m": 1.0,
            "band_type": "wide",
            "centerline": [(20.0, float(y)), (180.0, float(y))],
            "contour": [
                (18.0, y - 2.0), (182.0, y - 2.0),
                (182.0, y + 2.0), (18.0, y + 2.0),
            ],
        }
        for band_id, y in ((4, 30.0), (6, 50.0), (8, 70.0))
    ]
    mask = np.full((100, 200), 255, dtype=np.uint8)

    prepared = prepare_operator_centerline_plan(
        bands,
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "validation_profile": "field_trial",
            "work_line_mode": "band_centerline",
            "planning_support_mask": mask,
        },
        all_bands=bands,
    )

    assert prepared["catalog"]["count"] == 3
    geometry = prepared["layout"]["field_boundary_geometry"]
    assert geometry["repaired"] is True
    assert geometry["reason"] == "field_boundary_closing_loop_removed"
    assert prepared["layout"]["centerline_field_boundary_filter"]["kept_line_count"] == 3


def test_centerline_plan_names_an_unrepairable_field_outline():
    """An unusable outline must not be reported as 'lines outside the field'."""
    from path_planner import prepare_operator_centerline_plan

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        # A drawn "8": two comparable lobes, so the repair is ambiguous.
        field_boundary = [
            (10.0, 10.0), (190.0, 90.0), (190.0, 10.0), (10.0, 90.0),
        ]
        harvester_params = {
            "cutter_width_m": 4.0,
            "track_width_m": 0.4,
            "track_gauge_m": 1.6,
            "turn_radius_m": 1.0,
        }

    bands = [
        {
            "id": 4,
            "profile_width_m": 1.0,
            "width_m": 1.0,
            "band_type": "wide",
            "centerline": [(30.0, 50.0), (170.0, 50.0)],
            "contour": [
                (28.0, 48.0), (172.0, 48.0), (172.0, 52.0), (28.0, 52.0),
            ],
        }
    ]
    mask = np.full((100, 200), 255, dtype=np.uint8)

    prepared = prepare_operator_centerline_plan(
        bands,
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "validation_profile": "field_trial",
            "work_line_mode": "band_centerline",
            "planning_support_mask": mask,
        },
        all_bands=bands,
    )

    assert prepared["catalog"]["count"] == 0
    assert prepared["layout"]["field_boundary_geometry"]["present"] is False
    reasons = " ".join(prepared["layout"]["centerline_hard_reasons"])
    assert "自相交" in reasons
    assert "落在田块边界外" not in reasons


def test_centerline_selection_unlocks_service_points_only_after_catalog_choice():
    from state import AppState
    from pyside6_app.task_state import derive_task_statuses

    state = AppState()
    state.tif_path = "dummy.tif"
    state.field_boundary = [(0, 0), (100, 0), (100, 100)]
    state.inference_done = True
    state.mask_raw = np.ones((20, 20), dtype=np.uint8)
    state.mask_processed = True
    state.mask_result = {"processed_mask": state.mask_raw}
    state.harvester_params = {
        "cutter_width_m": 2.15,
        "cutter_forward_offset_m": 0.65,
        "track_width_m": 0.40,
        "track_gauge_m": 1.00,
        "wheelbase_m": 1.80,
        "track_length_m": 1.70,
        "turn_radius_m": 0.90,
    }
    state.harvester_params_confirmed = True
    state.centerline_catalog = {}
    state.selected_centerline_ids = []
    no_catalog = derive_task_statuses(state, has_image=True)
    assert no_catalog["centerline_generate"].status == "available"
    assert no_catalog["centerline_select"].status == "pending"
    assert no_catalog["plan"].status == "pending"

    state.centerline_catalog = _centerline_catalog_with_three_lines()
    state.entry_point = (10, 10)

    state.selected_centerline_ids = []
    before_choice = derive_task_statuses(state, has_image=True)
    assert before_choice["centerline_generate"].status == "done"
    assert before_choice["centerline_select"].status == "available"
    assert before_choice["entry"].status == "pending"
    assert before_choice["plan"].status == "pending"

    state.selected_centerline_ids = ["centerline-001"]
    state.entry_point_locked = False
    after_choice = derive_task_statuses(state, has_image=True)
    assert after_choice["centerline_select"].status == "done"
    assert after_choice["entry"].status == "available"
    assert after_choice["unload"].status == "done"
    assert after_choice["plan"].status == "pending"
