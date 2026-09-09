import os
import json
from pathlib import Path

import pytest

from PySide6.QtWidgets import QApplication


def test_builtin_test_image_resolves_to_existing_geotiff():
    from builtin_test_assets import builtin_assets_status, builtin_test_image

    path = builtin_test_image()
    if not path:
        pytest.skip("data/tif/result.tif is an optional local demo asset")
    assert os.path.isfile(path)
    assert os.path.basename(path) == "result.tif"
    assert builtin_assets_status() == "available"


def test_builtin_test_models_include_all_registered_trained_best_weights():
    from builtin_test_assets import builtin_test_models

    models = builtin_test_models()
    assert [model["name"] for model in models] == [
        "YOLO11m-seg（内置对比）",
        "YOLO11s-seg（内置对比）",
        "YOLOv8s-seg（内置对比）",
        "YOLO11s-seg（插秧期12m）",
        "YOLO11s-seg（返青期12m）",
        "YOLO11s-seg（返青期30m）",
        "YOLO11s-seg（返青期50m）",
        "YOLO11s-seg（分蘖期12m）",
        "YOLO11s-seg（分蘖期30m）",
        "YOLO11s-seg（插秧期12m｜旧版基线）",
        "YOLO11s-seg（返青期12m｜旧版基线）",
        "YOLO11s-seg（返青期30m｜旧版基线）",
        "YOLO11s-seg（返青期50m｜旧版基线）",
        "YOLO11s-seg（分蘖期12m｜旧版基线）",
        "YOLO11s-seg（分蘖期30m｜旧版基线）",
    ]
    assert len(models) == 15
    assert all(model["path"].endswith(".pt") for model in models)
    assert all(os.path.isfile(model["path"]) for model in models)
    assert all(len(model["sha256"]) == 64 for model in models)


def test_launcher_exposes_builtin_image_independently_of_stale_recent_files(monkeypatch, tmp_path):
    import pyside6_app.launcher as launcher

    stale = tmp_path / "recent.json"
    stale.write_text('["D:/missing/result.tif"]', encoding="utf-8")
    monkeypatch.setattr(launcher, "RECENT_FILE", str(stale))

    entries = launcher.project_entries()
    if launcher.builtin_test_image():
        assert entries[0]["kind"] == "builtin_test"
        assert entries[0]["path"].endswith("result.tif")
        assert os.path.isfile(entries[0]["path"])
    else:
        assert all(entry["kind"] != "builtin_test" for entry in entries)


def test_main_window_merges_trusted_builtin_comparison_models(monkeypatch):
    from pyside6_app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow.__new__(MainWindow)
    monkeypatch.setattr(window, "_builtin_model_dir", lambda: "D:/missing/data/models")
    monkeypatch.setattr(window, "_model_registry", lambda model_dir=None: {})

    models = window._builtin_model_list()
    assert len(models) == 15
    assert all(model.get("trusted_builtin") for model in models)
    assert all(os.path.isfile(model["path"]) for model in models)
    assert window._is_trusted_builtin_model_path(models[0]["path"])
    assert not window._is_trusted_builtin_model_path("D:/untrusted/yolo11m_seg_best.pt")


def test_builtin_assets_report_partial_without_returning_personal_paths(monkeypatch, tmp_path):
    import builtin_test_assets as assets

    image = tmp_path / "result.tif"
    image.write_bytes(b"image")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "yolo11m_seg_best.pt").write_bytes(b"model")
    monkeypatch.setenv("REUI_TEST_IMAGE", str(image))
    monkeypatch.setenv("REUI_TEST_MODEL_DIR", str(model_dir))

    discovered = assets.discover_builtin_assets()

    assert discovered["status"] == "partial"
    assert discovered["image"] == str(image.resolve())
    assert [model["path"] for model in discovered["existing_models"]] == [
        str((model_dir / "yolo11m_seg_best.pt").resolve())
    ]
    expected_root = model_dir.resolve()
    assert all(Path(model["path"]).parent == expected_root for model in discovered["models"])


def test_builtin_assets_report_missing_when_no_env_or_packaged_asset_exists(monkeypatch, tmp_path):
    import builtin_test_assets as assets

    monkeypatch.delenv("REUI_TEST_IMAGE", raising=False)
    monkeypatch.delenv("REUI_TEST_MODEL_DIR", raising=False)
    monkeypatch.setattr(assets, "_DEFAULT_TEST_IMAGE", str(tmp_path / "missing.tif"))
    monkeypatch.setattr(assets, "_DEFAULT_MODEL_DIR", str(tmp_path / "missing-models"))

    discovered = assets.discover_builtin_assets()

    assert discovered["status"] == "missing"
    assert discovered["image"] == ""
    assert discovered["models"] == []


def test_builtin_assets_use_the_model_registry_manifest(monkeypatch, tmp_path):
    import builtin_test_assets as assets

    image = tmp_path / "result.tif"
    image.write_bytes(b"image")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "registry-only.pt"
    model_path.write_bytes(b"model")
    (model_dir / "model_registry.json").write_text(
        json.dumps({
            "models": [{
                "name": "registry-only.pt",
                "display_name": "Registry only",
                "sha256": "a" * 64,
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("REUI_TEST_IMAGE", str(image))
    monkeypatch.setenv("REUI_TEST_MODEL_DIR", str(model_dir))

    discovered = assets.discover_builtin_assets()

    assert discovered["models"] == [{
        "name": "Registry only",
        "path": str(model_path.resolve()),
        "sha256": "a" * 64,
        "trusted_builtin": True,
    }]
    assert discovered["status"] == "available"
