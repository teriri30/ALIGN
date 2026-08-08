import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from trial_package import export_visual_artifacts, write_package_index
from pyside6_app.main_window import MainWindow


def _machine_ready():
    return {
        "schema": "reui.machine-readiness.v1",
        "controlled_trial_ready": True,
        "controlled_trial_blockers": [],
        "eligible_for_machine_export": True,
        "current_route_classification": "machine_export_candidate",
        "blockers": [],
        "manual_review_required": False,
        "terminal_protocol_validated": True,
    }


@pytest.mark.parametrize(
    "readiness",
    [
        {},
        {
            **_machine_ready(),
            "eligible_for_machine_export": False,
            "blockers": ["terminal_adapter_not_validated"],
        },
        {**_machine_ready(), "terminal_protocol_validated": False},
        _machine_ready(),
    ],
)
def test_candidate_path_export_fails_before_writing_without_machine_evidence(
    tmp_path, readiness
):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    output = tmp_path / "blocked.path"
    path_result = {"machine_readiness": readiness}
    points = [{
        "lat": 30.0,
        "lon": 116.0,
        "pixel_x": 1.0,
        "pixel_y": 1.0,
        "segment_index": 0,
        "point_index": 0,
        "segment_type": "work",
        "status": 1,
    }]

    with pytest.raises(ValueError, match="机器|终端协议"):
        window._export_candidate_path(
            str(output), path_result=path_result, geo_points=points
        )

    assert not output.exists()
    window.close()


def test_machine_compatibility_preview_exports_without_formal_machine_evidence(
    tmp_path, monkeypatch
):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    output = tmp_path / "compatibility-preview.path"
    path_result = {
        "machine_readiness": {},
        "validation": {"valid": True},
        "is_valid": True,
    }
    points = [
        {
            "lat": 30.0,
            "lon": 116.0,
            "pixel_x": 1.0,
            "pixel_y": 1.0,
            "segment_index": 0,
            "point_index": 0,
            "segment_type": "work",
            "status": 1,
        },
        {
            "lat": 30.001,
            "lon": 116.0,
            "pixel_x": 1.0,
            "pixel_y": 10.0,
            "segment_index": 0,
            "point_index": 1,
            "segment_type": "work",
            "status": 1,
        },
    ]
    captured = {}

    def fake_manifest(
        path,
        output_format,
        captured_result,
        captured_points,
        protocol_metadata=None,
    ):
        captured.update({
            "output_format": output_format,
            "path_result": captured_result,
            "points": captured_points,
            "protocol": protocol_metadata,
        })
        return str(path) + ".manifest.json"

    monkeypatch.setattr(window, "_write_export_manifest", fake_manifest)
    window._export_candidate_path(
        str(output),
        path_result=path_result,
        geo_points=points,
        machine_compatibility=True,
        preview_only=True,
    )

    assert output.exists()
    assert captured["output_format"] == "$PATH_MACHINE_COMPAT"
    assert captured["protocol"]["preview_only"] is True
    assert captured["protocol"]["machine_executable"] is False
    rows = [
        line
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.startswith("$PATH")
    ]
    assert len(rows) == 2
    assert all(row.rstrip("*").split(",")[-1] in {"0", "1"} for row in rows)
    window.close()


def test_machine_compatibility_export_matches_visible_compact_control_points(
    tmp_path, monkeypatch
):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    output = tmp_path / "compact-control-preview.path"
    window._tif_rgb = np.zeros((40, 40, 3), dtype=np.uint8)

    class Geo:
        def is_ready(self):
            return True

        def pixel_to_lonlat(self, x, y):
            return 116.0 + float(x) * 0.001, 30.0 + float(y) * 0.001

        def pixel_distance_m(self, first, second):
            return float(
                ((float(second[0]) - float(first[0])) ** 2
                 + (float(second[1]) - float(first[1])) ** 2) ** 0.5
            )

    window.geo = Geo()
    window.state.path_points = [
        (1.0, 1.0),
        (10.0, 1.0),
        (10.0, 10.0),
        (1.0, 10.0),
    ]
    window.state.path_status = [1, 0, 1]
    dense_points = [
        {
            "lat": 30.0 + index * 0.0001,
            "lon": 116.0,
            "pixel_x": float(index),
            "pixel_y": 1.0,
            "segment_index": 0,
            "point_index": index,
            "segment_type": "work",
            "status": 1,
        }
        for index in range(10)
    ]
    path_result = {
        "machine_readiness": {},
        "validation": {"valid": True},
        "is_valid": True,
    }
    captured = {}

    def fake_manifest(
        path,
        output_format,
        captured_result,
        captured_points,
        protocol_metadata=None,
    ):
        captured.update({
            "output_format": output_format,
            "points": captured_points,
            "protocol": protocol_metadata,
        })
        return str(path) + ".manifest.json"

    monkeypatch.setattr(window, "_write_export_manifest", fake_manifest)
    window._export_candidate_path(
        str(output),
        path_result=path_result,
        geo_points=dense_points,
        machine_compatibility=True,
        preview_only=True,
    )

    rows = [
        line
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.startswith("$PATH")
    ]
    assert len(rows) == len(window.state.path_points)
    assert [row.rstrip("*").split(",")[-1] for row in rows] == [
        "1", "0", "1", "0"
    ]
    assert captured["protocol"]["route_source"] == (
        "shared_route_control_polyline"
    )
    assert captured["protocol"]["control_route_point_count"] == 4
    assert captured["protocol"]["canonical_route_point_count"] == 10
    window.close()


def test_compatibility_payload_skips_only_formal_parameter_source_gate(monkeypatch):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)

    class Geo:
        def is_ready(self):
            return True

        def pixel_to_lonlat(self, x, y):
            return 116.0 + float(x) * 0.001, 30.0 + float(y) * 0.001

    planned_cutter_width = float(
        window._effective_harvester_params(
            getattr(window.state, "harvester_params", {}) or {}
        )["cutter_width_m"]
    )
    path_result = {
        "full_path": [[(1.0, 1.0), (10.0, 1.0)]],
        "segment_types": ["work"],
        "ordered_segments": [{
            "points": [(1.0, 1.0), (10.0, 1.0)],
            "type": "work",
        }],
        "validation": {
            "valid": True,
            "current_validation_complete": True,
            "validation_profile": "field_trial",
        },
        "is_valid": True,
        "planning_factors": {
            "harvester": {"cutter_width_m": planned_cutter_width},
            "harvester_parameter_source_status": {
                "required": ["cutter_width_m"],
                "sources": {"cutter_width_m": "default"},
            },
        },
        "layout": {},
    }
    window.geo = Geo()
    monkeypatch.setattr(window, "_path_result_from_state", lambda: path_result)
    monkeypatch.setattr(window, "_verify_scientific_provenance", lambda _: None)

    _result, points = window._geo_export_payload(formal=False)
    assert len(points) == 2

    with pytest.raises(ValueError, match="参数来源"):
        window._geo_export_payload(formal=True)
    window.close()


def test_complete_trial_visual_artifacts_include_masks_path_and_diagnostics(tmp_path):
    base = np.full((80, 120, 3), 80, dtype=np.uint8)
    raw = np.zeros((50, 70), dtype=np.uint8)
    raw[10:40, 8:60] = 255
    processed = np.zeros_like(raw)
    processed[12:38, 10:58] = 255
    headland = np.zeros_like(raw)
    headland[38:45, 10:58] = 255
    result = {
        "processed_mask": processed,
        "headland_mask": headland,
        "uncertain_residual_mask": np.zeros_like(raw),
        "diagnostics": {
            "meters_per_pixel": 0.1,
            "algorithm_version": "test",
            "strength": "standard",
        },
        "all_bands": [],
    }
    path_result = {
        "tracks": [{"points": [(10, 20), (100, 20)]}],
        "turns": [{"points": [(100, 20), (100, 60)]}],
        "validation": {"valid": True, "total_length_m": 13.0},
    }

    artifacts = export_visual_artifacts(
        str(tmp_path),
        source_image=base,
        raw_mask=raw,
        mask_result=result,
        path_result=path_result,
        offset=(20, 15),
        field_boundary=[(5, 5), (110, 5), (110, 70), (5, 70)],
        forbidden_regions=[[(45, 35), (60, 35), (60, 50), (45, 50)]],
        service_points={"entry": (10, 20), "exit": (100, 60), "unload": []},
        geo=None,
        mask_config={},
    )

    expected = {
        "01_source_preview.png",
        "02_ai_segmentation.png",
        "03_mask_processed.png",
        "04_path_planning.png",
        "masks/raw_mask.png",
        "masks/processed_mask.png",
        "05_mask_diagnostics.zip",
        "06_validation.json",
        "README.txt",
    }
    assert expected <= set(artifacts)
    for relative in expected:
        assert (tmp_path / relative).is_file()
    path_preview = cv2.imread(str(tmp_path / "04_path_planning.png"))
    assert path_preview.shape[:2] == base.shape[:2]


def test_package_index_hashes_route_manifest_and_visual_artifacts(tmp_path):
    (tmp_path / "mission_field_trial.path").write_text("$PATH,test*\n", encoding="utf-8")
    (tmp_path / "mission_field_trial.path.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preview.png").write_bytes(b"png")

    index_path = write_package_index(str(tmp_path))
    index = json.loads((tmp_path / "package_index.json").read_text(encoding="utf-8"))

    assert index_path == str(tmp_path / "package_index.json")
    assert {item["path"] for item in index["files"]} == {
        "mission_field_trial.path",
        "mission_field_trial.path.manifest.json",
        "preview.png",
    }
    assert all(len(item["sha256"]) == 64 for item in index["files"])


def test_complete_package_keeps_candidate_path_and_required_manifest_together(tmp_path, monkeypatch):
    """DECISION-001: the complete bundle cannot bypass the formal manifest."""
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    window.state.mask_raw = np.ones((40, 60), dtype=np.uint8) * 255
    window.state.mask_result = {
        "processed_mask": np.ones((40, 60), dtype=np.uint8) * 255,
        "diagnostics": {"meters_per_pixel": 0.1},
    }
    path_result = {
        "tracks": [{"points": [(1, 1), (10, 10)]}],
        "turns": [],
        "validation": {"valid": True},
        "machine_readiness": _machine_ready(),
    }
    geo_points = [
        {
            "segment_index": 0,
            "point_index": 0,
            "pixel_x": 1.0,
            "pixel_y": 1.0,
            "lat": 30.0,
            "lon": 116.0,
            "segment_type": "work",
            "status": 1,
        }
    ]
    monkeypatch.setattr(window, "_geo_export_payload", lambda: (path_result, geo_points))
    monkeypatch.setattr(
        "machine_route_validator.assess_machine_readiness",
        lambda *_args, **_kwargs: _machine_ready(),
    )
    def fake_visual_artifacts(output_dir, **_kwargs):
        preview = Path(output_dir) / "preview.png"
        preview.write_bytes(b"png")
        return {"preview.png": str(preview)}

    monkeypatch.setattr("trial_package.export_visual_artifacts", fake_visual_artifacts)

    def fake_manifest(route_path, _format, _path_result, _points):
        manifest = route_path + ".manifest.json"
        with open(manifest, "w", encoding="utf-8") as handle:
            handle.write("{}")
        return manifest

    monkeypatch.setattr(window, "_write_export_manifest", fake_manifest)

    package_dir, route_path, manifest_path = window._export_complete_trial_package(str(tmp_path))

    assert package_dir.startswith(str(tmp_path))
    assert route_path.endswith("mission_field_trial.path")
    assert manifest_path == route_path + ".manifest.json"
    assert Path(route_path).is_file()
    assert Path(manifest_path).is_file()
    assert (Path(package_dir) / "package_index.json").is_file()
    window.close()
