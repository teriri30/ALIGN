import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from trial_package import export_visual_artifacts, write_package_index
from pyside6_app.main_window import MainWindow


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
