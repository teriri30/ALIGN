import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow


class _GoodGeo:
    def is_ready(self):
        return True

    def pixel_to_lonlat(self, x, y):
        return 120.0 + float(x) * 0.001, 30.0 + float(y) * 0.001

    def pixel_distance_m(self, a, b):
        return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


class _FailingGeo(_GoodGeo):
    def pixel_to_lonlat(self, x, y):
        if float(x) >= 10.0:
            raise RuntimeError("bad transform")
        return super().pixel_to_lonlat(x, y)


def test_path_to_geo_fails_instead_of_writing_fake_zero_coordinates():
    from path_planner import path_to_geo

    with pytest.raises(ValueError, match="地理坐标转换失败"):
        path_to_geo([[(0.0, 0.0), (10.0, 0.0)]], _FailingGeo(), object(), ["work"])


def test_geo_export_validation_rejects_empty_or_mismatched_points(tmp_path):
    from path_planner import export_csv, validate_geo_points

    with pytest.raises(ValueError, match="没有有效地理坐标点"):
        export_csv([], str(tmp_path / "empty.csv"))

    points = [{"lon": 120.0, "lat": 30.0, "pixel_x": 0.0, "pixel_y": 0.0}]
    with pytest.raises(ValueError, match="数量不一致"):
        validate_geo_points(points, expected_count=2)


def test_path_format_preserves_turn_segment_status(tmp_path):
    from path_planner import export_path_format

    output = tmp_path / "mission.path"
    export_path_format(
        [
            {
                "lon": 120.0,
                "lat": 30.0,
                "pixel_x": 0.0,
                "pixel_y": 0.0,
                "segment_type": "turn",
            }
        ],
        str(output),
    )

    first_record = output.read_text(encoding="utf-8").splitlines()[0]
    assert first_record.startswith("$PATH,0,0,1,30.00000000,120.00000000,")
    assert first_record.endswith(",0*")


def test_local_mask_path_exports_with_offset_but_validation_stays_local():
    from path_planner import path_to_geo, validate_path

    class Geo(_GoodGeo):
        def meters_per_pixel(self, _x, _y):
            return 1.0

    class State:
        mask_offset_x = 100
        mask_offset_y = 200
        field_boundary = [(0, 0), (59, 0), (59, 49), (0, 49)]
        harvester_params = {
            "cutter_width_m": 0.2,
            "cutter_forward_offset_m": 0.2,
            "track_width_m": 0.2,
            "track_gauge_m": 0.4,
            "wheelbase_m": 1.0,
            "track_length_m": 1.0,
            "turn_radius_m": 1.0,
        }

    local_path = [[(1.0, 2.0), (11.0, 2.0)]]
    points = path_to_geo(local_path, Geo(), State(), ["work"])
    validation = validate_path(
        local_path,
        np.ones((50, 60), dtype=np.uint8) * 255,
        0.0,
        geo=Geo(),
        state=State(),
        segment_types=["work"],
        config={
            "max_track_core_overlap_pct": 100.0,
            "min_geometric_coverage_pct": 0.0,
            "max_track_outside_field_pct": 100.0,
        },
    )

    assert [(point["pixel_x"], point["pixel_y"]) for point in points] == [
        (101.0, 202.0),
        (111.0, 202.0),
    ]
    assert validation["total_length_m"] == pytest.approx(10.0)


def test_manual_route_edit_refreshes_path_but_blocks_unvalidated_export(monkeypatch):
    """DECISION-001: manual edits must invalidate formal-export approval."""
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.geo = _GoodGeo()
    window.state.path_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]
    window.state.path_status = [1, 0]
    window._pipeline_result = {
        "path": {
            "full_path": [[(99.0, 99.0), (100.0, 100.0)]],
            "segment_types": ["work"],
            "geo_points": [
                {"lon": 1.0, "lat": 1.0, "pixel_x": 99.0, "pixel_y": 99.0},
                {"lon": 2.0, "lat": 2.0, "pixel_x": 100.0, "pixel_y": 100.0},
            ],
        }
    }

    monkeypatch.setattr(window.route_info, "update_from_auto_path", lambda _path: None)

    window._sync_auto_path_from_editable_route()
    path_result = window._normalise_path_result(window._pipeline_result["path"])
    geo_points = path_result["geo_points"]

    assert path_result["control_full_path"] == [
        [(0.0, 0.0), (10.0, 0.0)],
        [(10.0, 0.0), (10.0, 5.0)],
    ]
    assert path_result["full_path"][0] == [(0.0, 0.0), (10.0, 0.0)]
    assert len(path_result["full_path"][1]) > 2
    assert path_result["full_path"][1][0] == (10.0, 0.0)
    assert path_result["full_path"][1][-1] == (10.0, 5.0)
    assert geo_points[0]["pixel_x"] == 0.0
    assert geo_points[-1]["pixel_x"] == 10.0
    assert all(point["lon"] >= 120.0 for point in geo_points)
    with pytest.raises(ValueError, match="未通过安全验证"):
        window._geo_export_payload()


def test_manual_route_edit_can_export_current_compatibility_control_route(
    tmp_path, monkeypatch
):
    """Manual endpoint repair must reach the 0/1 compatibility file."""
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    window.geo = _GoodGeo()
    window.state.path_points = [
        (2.0, 2.0),
        (20.0, 2.0),
        (20.0, 20.0),
        (4.0, 20.0),
    ]
    window.state.path_status = [1, 0, 1]
    monkeypatch.setattr(window.route_info, "update_from_auto_path", lambda _path: None)
    monkeypatch.setattr(
        window,
        "_write_export_manifest",
        lambda path, *_args, **_kwargs: str(path) + ".manifest.json",
    )

    window._sync_auto_path_from_editable_route()
    route_path, _manifest = window._export_candidate_path(
        str(tmp_path),
        machine_compatibility=True,
        preview_only=True,
    )

    rows = [
        line
        for line in (tmp_path / "001.txt").read_text(encoding="utf-8").splitlines()
        if line.startswith("$PATH")
    ]
    assert route_path == str(tmp_path / "001.txt")
    assert len(rows) == len(window.state.path_points)
    assert [row.rstrip("*").split(",")[-1] for row in rows] == [
        "2", "2", "2", "2"
    ]
    assert window._pipeline_result["path"]["full_path"][1][0] == (20.0, 2.0)
    assert window._pipeline_result["path"]["full_path"][1][-1] == (20.0, 20.0)
    assert len(window._pipeline_result["path"]["full_path"][1]) > 2
    window.close()


def test_confirm_gate_exports_metric_failure_without_warning_block(monkeypatch):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.geo = _GoodGeo()
    path_result = {
        "full_path": [[(1.0, 1.0), (10.0, 1.0)]],
        "segment_types": ["work"],
        "ordered_segments": [{
            "points": [(1.0, 1.0), (10.0, 1.0)],
            "type": "work",
            "segment_index": 0,
        }],
        "validation": {
            "valid": False,
            "current_validation_complete": True,
            "validation_profile": "field_trial",
            "issues": ["theoretical footprint metric exceeds threshold"],
        },
        "is_valid": False,
    }
    monkeypatch.setenv("REUI_EXPORT_GATE", "confirm")
    warnings = []
    monkeypatch.setattr(window.log_panel, "warn", warnings.append)
    monkeypatch.setattr(window, "_path_result_from_state", lambda: path_result)
    monkeypatch.setattr(window, "_verify_scientific_provenance", lambda _: None)

    _result, points = window._geo_export_payload()

    assert len(points) == 2
    assert points[0]["pixel_x"] == 1.0
    assert points[1]["pixel_x"] == 10.0
    assert warnings == []
    window.close()


def test_geo_export_rejects_cached_validation_snapshot(monkeypatch):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.geo = _GoodGeo()
    window._store_path_result({
        "full_path": [[(1.0, 1.0), (10.0, 1.0)]],
        "segment_types": ["work"],
        "validation": {
            "valid": True,
            "current_validation_complete": False,
            "validation_source": "cache_snapshot",
        },
        "is_valid": True,
    })
    monkeypatch.setattr(window, "_verify_scientific_provenance", lambda _path: None)

    with pytest.raises(ValueError, match="缓存快照"):
        window._geo_export_payload()


def test_geo_export_ignores_valid_transient_pipeline_path_without_canonical_state():
    """DECISION-001: formal export never promotes a transient pipeline route."""
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.geo = _GoodGeo()
    window._pipeline_result = {
        "path": {
            "full_path": [[(1.0, 1.0), (10.0, 1.0)]],
            "segment_types": ["work"],
            "validation": {"valid": True},
            "is_valid": True,
        }
    }

    with pytest.raises(ValueError, match="没有可导出的路径"):
        window._geo_export_payload()


def test_geo_export_rejects_segment_crossing_concave_field_even_when_vertices_are_inside(monkeypatch):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    window.geo = _GoodGeo()
    window.state.field_boundary = [
        (10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (60.0, 90.0),
        (60.0, 40.0), (40.0, 40.0), (40.0, 90.0), (10.0, 90.0),
    ]
    window._store_path_result({
        "full_path": [[(20.0, 80.0), (80.0, 80.0)]],
        "segment_types": ["work"],
        "validation": {"valid": True},
        "is_valid": True,
    })
    monkeypatch.setattr(window, "_verify_scientific_provenance", lambda _path: None)

    with pytest.raises(ValueError, match="field boundary"):
        window._geo_export_payload()


def test_geo_export_rejects_valid_flag_when_route_point_leaves_field_boundary(monkeypatch):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    window.geo = _GoodGeo()
    window.state.field_boundary = [
        (10.0, 10.0),
        (90.0, 10.0),
        (90.0, 90.0),
        (10.0, 90.0),
    ]
    window._store_path_result({
        "full_path": [[(20.0, 20.0), (95.0, 20.0)]],
        "segment_types": ["work"],
        "validation": {"valid": True},
        "is_valid": True,
    })
    monkeypatch.setattr(window, "_verify_scientific_provenance", lambda _path: None)

    with pytest.raises(ValueError, match="field boundary"):
        window._geo_export_payload()
