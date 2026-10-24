import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ALIGN_DISABLE_AUTO_START", "1")

import numpy as np
import pytest


def test_missing_config_uses_native_640_capture_size():
    from config import Config

    cfg = Config()
    previous_raw = dict(getattr(cfg, "_raw", {}))
    previous_loaded = getattr(cfg, "_loaded", False)
    try:
        cfg._raw = {}
        cfg._loaded = False
        assert cfg.TILE_CAPTURE_SIZE == 640
    finally:
        cfg._raw = previous_raw
        cfg._loaded = previous_loaded


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"model": {"tile_overlap": 1.0}}, "tile_overlap"),
        ({"model": {"conf_threshold": -0.1}}, "conf_threshold"),
        ({"model": {"orthomosaic_primary_mask_threshold": 1.1}}, "orthomosaic_primary_mask_threshold"),
        ({"model": {"orthomosaic_tile_edge_weight_floor": -0.1}}, "orthomosaic_tile_edge_weight_floor"),
        ({"harvester": {"track_width_m": 0.8, "track_gauge_m": 0.5}}, "track_width_m"),
        (
            {
                "harvester": {
                    "cutter_forward_offset_m": 5.0,
                    "track_length_m": 1.7,
                    "wheelbase_m": 1.8,
                }
            },
            "cutter_forward_offset_m",
        ),
        ({"mask_processing": {"strength": "unknown"}}, "strength"),
    ],
)
def test_invalid_scientific_config_is_rejected(payload, message):
    from config import ConfigValidationError, validate_config

    with pytest.raises(ConfigValidationError, match=message):
        validate_config(payload)


def test_stage_record_rejects_changed_model_or_artifact(tmp_path):
    """DECISION-002: cached stages remain bound to model and artifact hashes."""
    from provenance import ProvenanceError, array_sha256, file_sha256, make_stage_record, verify_stage_record

    model = tmp_path / "model.pt"
    model.write_bytes(b"model-a")
    mask = np.arange(16, dtype=np.uint8).reshape(4, 4)
    inputs = {"model_sha256": file_sha256(model), "conf": 0.25}
    record = make_stage_record("inference", "inference-v1", inputs, mask)

    assert verify_stage_record(record, "inference", inputs, mask)
    assert record["artifact_sha256"] == array_sha256(mask)

    model.write_bytes(b"model-b")
    changed_inputs = {"model_sha256": file_sha256(model), "conf": 0.25}
    with pytest.raises(ProvenanceError, match="input fingerprint"):
        verify_stage_record(record, "inference", changed_inputs, mask)

    changed_mask = mask.copy()
    changed_mask[0, 0] = 255
    with pytest.raises(ProvenanceError, match="artifact hash"):
        verify_stage_record(record, "inference", inputs, changed_mask)


def test_mask_stage_record_rejects_any_parameter_change():
    from provenance import ProvenanceError, make_stage_record, verify_stage_record

    mask = np.ones((8, 8), dtype=np.uint8)
    inputs = {
        "inference_fingerprint": "abc",
        "config": {"strength": "standard", "band_width_threshold_m": 0.55},
    }
    record = make_stage_record("mask", "mask-v1", inputs, mask)
    changed = {
        "inference_fingerprint": "abc",
        "config": {"strength": "standard", "band_width_threshold_m": 0.65},
    }

    with pytest.raises(ProvenanceError, match="input fingerprint"):
        verify_stage_record(record, "mask", changed, mask)


def test_uint16_preprocessing_is_per_band_deterministic_and_masks_nodata():
    """DECISION-005: raster normalization is deterministic and nodata-aware."""
    from raster_preprocessing import RASTER_PREPROCESSING_VERSION, normalise_raster_bands

    bands = np.array(
        [
            [[0, 100, 200], [300, 400, 65535]],
            [[0, 1000, 2000], [3000, 4000, 65535]],
            [[0, 10, 20], [30, 40, 65535]],
        ],
        dtype=np.uint16,
    )
    valid = np.ones_like(bands, dtype=np.uint8) * 255
    valid[:, 1, 2] = 0
    config = {"mode": "percentile_per_band", "lower_percentile": 0.0, "upper_percentile": 100.0}

    first, metadata = normalise_raster_bands(bands, valid, config)
    second, metadata_again = normalise_raster_bands(bands, valid, config)

    assert first.dtype == np.uint8
    assert first.shape == (2, 3, 3)
    assert np.array_equal(first, second)
    assert np.all(first[1, 2] == 0)
    assert metadata == metadata_again
    assert metadata["version"] == RASTER_PREPROCESSING_VERSION
    assert [item["upper"] for item in metadata["bands"]] == [400.0, 4000.0, 40.0]


def test_all_nodata_raster_window_has_a_distinct_failure_type():
    """DECISION-005: a legal all-NoData window is distinguishable from corrupt RGB."""
    from raster_preprocessing import EmptyRasterWindowError, normalise_raster_bands

    bands = np.zeros((3, 16, 16), dtype=np.uint8)
    masks = np.zeros_like(bands, dtype=np.uint8)

    with pytest.raises(EmptyRasterWindowError, match="no valid pixels"):
        normalise_raster_bands(bands, masks)


def test_legacy_export_engine_is_fail_closed(tmp_path):
    """DECISION-001: legacy exporters cannot bypass the formal export gate."""
    from export_guard import ExportEngine
    from state import AppState

    state = AppState()
    state.path_points = [(0.0, 0.0), (1.0, 1.0)]
    state.path_status = [1, 1]

    with pytest.raises(RuntimeError, match="ExportEngine"):
        ExportEngine(object(), state).export_to_file(str(tmp_path / "unsafe.path"))


def test_cache_restore_rejects_old_mask_when_model_file_changes(tmp_path, monkeypatch):
    """DECISION-002: cache restoration rejects evidence from a changed model."""
    import cache
    from config import Config
    from inference import INFERENCE_PIPELINE_VERSION, build_inference_config
    from provenance import file_sha256, make_stage_record
    from pyside6_app.workers import CacheRestoreWorker
    from state import AppState

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    source = tmp_path / "field.tif"
    source.write_bytes(b"geotiff-source")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-a")
    raw_mask = np.ones((6, 6), dtype=np.uint8)
    cfg = Config()
    inputs = {
        "source_sha256": file_sha256(source),
        "model_sha256": file_sha256(model),
        "field_boundary": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]],
        "inference_config": build_inference_config(cfg),
        "preprocessing_config": cfg.section("raster_preprocessing"),
    }
    record = make_stage_record(
        "inference", INFERENCE_PIPELINE_VERSION, inputs, raw_mask
    )
    state = AppState()
    state.safe_update(
        tif_path=str(source),
        source_sha256=file_sha256(source),
        current_model_path=str(model),
        current_model_name=model.name,
        model_sha256=file_sha256(model),
        field_boundary=inputs["field_boundary"],
        inference_done=True,
        inference_provenance=record,
    )
    cache.save_mask(str(source), raw_mask)
    cache.save_project_state(str(source), state, stage="inference")

    model.write_bytes(b"model-b")
    results = []
    worker = CacheRestoreWorker(str(source), expected_source_sha256=file_sha256(source))
    worker.finished.connect(results.append)
    worker.run()

    assert results[0]["status"] == "completed"
    restored = results[0]["result"]
    assert restored["saved"]["inference_done"] is False
    assert restored["raw_mask"] is None
    assert any(item.startswith("AI ") for item in restored["cache_validation_messages"])


def test_cache_restore_rejects_mask_before_run_edge_gate_version(tmp_path, monkeypatch):
    """DECISION-009: v13 geometry cannot survive the run-edge gate change."""
    import cache
    from config import Config
    from inference import INFERENCE_PIPELINE_VERSION, build_inference_config
    from provenance import file_sha256, make_stage_record
    from pyside6_app.workers import CacheRestoreWorker
    from row_geometry import MASK_REGULARIZATION_VERSION
    from state import AppState

    assert MASK_REGULARIZATION_VERSION == 21
    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    source = tmp_path / "field.tif"
    source.write_bytes(b"geotiff-source")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-a")
    raw_mask = np.ones((6, 6), dtype=np.uint8)
    processed_mask = raw_mask.copy()
    cfg = Config()
    inference_inputs = {
        "source_sha256": file_sha256(source),
        "model_sha256": file_sha256(model),
        "field_boundary": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]],
        "inference_config": build_inference_config(cfg),
        "preprocessing_config": cfg.section("raster_preprocessing"),
    }
    inference_record = make_stage_record(
        "inference", INFERENCE_PIPELINE_VERSION, inference_inputs, raw_mask
    )
    mask_inputs = {
        "inference_fingerprint": inference_record["fingerprint"],
        "raw_mask_sha256": inference_record["artifact_sha256"],
        "mask_config": cfg.section("mask_processing"),
    }
    legacy_mask_record = make_stage_record(
        "mask", 13, mask_inputs, processed_mask
    )
    state = AppState()
    state.safe_update(
        tif_path=str(source),
        source_sha256=file_sha256(source),
        current_model_path=str(model),
        current_model_name=model.name,
        model_sha256=file_sha256(model),
        field_boundary=inference_inputs["field_boundary"],
        inference_done=True,
        inference_provenance=inference_record,
        mask_processed=True,
        mask_result={"diagnostics": {"algorithm_version": 13}},
        mask_provenance=legacy_mask_record,
        auto_path_planned=True,
        auto_path_valid=True,
    )
    cache.save_mask(str(source), raw_mask)
    cache.save_mask(str(source), processed_mask, suffix="_processed")
    cache.save_project_state(str(source), state, stage="path")

    results = []
    worker = CacheRestoreWorker(
        str(source), expected_source_sha256=file_sha256(source)
    )
    worker.finished.connect(results.append)
    worker.run()

    assert results[0]["status"] == "completed"
    restored = results[0]["result"]
    assert restored["saved"]["inference_done"] is True
    assert restored["saved"]["mask_processed"] is False
    assert restored["saved"]["auto_path_planned"] is False
    assert restored["processed_mask"] is None
    assert any(
        "mask algorithm version changed" in item
        for item in restored["cache_validation_messages"]
    )


def test_cache_loader_never_reuses_same_shape_mask_from_another_source(tmp_path, monkeypatch):
    """DECISION-002: array shape alone never establishes cache identity."""
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cache.save_mask(str(first), np.ones((10, 10), dtype=np.uint8))

    assert cache.load_preferred_cache(str(second), image_shape=(10, 10)) is None


def test_semantic_mask_layers_roundtrip_without_entering_json_summary(tmp_path, monkeypatch):
    """DECISION-009: dense semantic layers remain atomic binary artifacts."""
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    body = np.zeros((20, 30), dtype=np.uint8)
    body[8:12, 4:26] = 255
    headland = np.zeros_like(body)
    headland[:, :3] = 255
    uncertain = np.zeros_like(body)
    uncertain[2:5, 10:15] = 255

    cache.save_mask(
        "D:/fake/source.tif",
        body,
        suffix="_processed",
        extra_arrays={
            "headland_mask": headland,
            "uncertain_residual_mask": uncertain,
        },
    )
    loaded, _x, _y, layers = cache.load_mask_bundle("D:/fake/source.tif", "_processed")
    summary = cache._mask_result_summary({
        "processed_mask": body,
        "headland_mask": headland,
        "uncertain_residual_mask": uncertain,
        "main_angle": 0.1,
    })

    assert np.array_equal(loaded, body)
    assert np.array_equal(layers["headland_mask"], headland)
    assert np.array_equal(layers["uncertain_residual_mask"], uncertain)
    assert summary == {"main_angle": 0.1}


def test_metric_processing_rejects_anisotropic_raster_pixels():
    """DECISION-003: unsupported anisotropic scale cannot enter metric processing."""
    from pyside6_app.workers import require_metric_scale

    class AnisotropicGeo:
        def is_ready(self):
            return True

        def meters_per_pixel(self, _x, _y):
            return 0.02

        def pixel_distance_m(self, first, second):
            if second[0] != first[0]:
                return 0.01
            return 0.03

    state = type("State", (), {"mask_offset_x": 0, "mask_offset_y": 0})()
    with pytest.raises(RuntimeError, match="ratio="):
        require_metric_scale(AnisotropicGeo(), state, (100, 100))


def test_builtin_model_registry_matches_present_weight_files():
    """DECISION-006: built-in model trust is pinned to verified file hashes."""
    import json
    from pathlib import Path
    from provenance import file_sha256

    model_dir = Path(__file__).resolve().parents[1] / "data" / "models"
    registry = json.loads((model_dir / "model_registry.json").read_text(encoding="utf-8"))
    assert registry["schema"] == 2
    registered_names = {item["name"] for item in registry["models"]}
    present_names = {path.name for path in model_dir.glob("*.pt")}
    assert present_names == registered_names
    for item in registry["models"]:
        weight = model_dir / item["name"]
        assert file_sha256(weight) == item["sha256"]


def test_stage_height_models_are_registered_with_complete_training_metadata():
    """DECISION-006: stage-height models are content-pinned and traceable."""
    import json
    from pathlib import Path

    model_dir = Path(__file__).resolve().parents[1] / "data" / "models"
    registry = json.loads(
        (model_dir / "model_registry.json").read_text(encoding="utf-8")
    )
    expected = {
        "yolo11s_seg_transplant_12m_best.pt": ("插秧期", 12),
        "yolo11s_seg_greenup_12m_best.pt": ("返青期", 12),
        "yolo11s_seg_greenup_30m_best.pt": ("返青期", 30),
        "yolo11s_seg_greenup_50m_best.pt": ("返青期", 50),
        "yolo11s_seg_tiller_12m_best.pt": ("分蘖期", 12),
        "yolo11s_seg_tiller_30m_best.pt": ("分蘖期", 30),
    }
    items = {item["name"]: item for item in registry["models"]}

    for filename, (stage, height_m) in expected.items():
        item = items[filename]
        assert item["display_name"] == f"YOLO11s-seg（{stage}{height_m}m）"
        assert len(item["sha256"]) == 64
        assert item["metadata_status"] == "complete"
        assert item["class_mapping"] == {"0": "rice"}
        assert item["stage"] == stage
        assert item["height_m"] == height_m
        assert item["training_preprocessing"] == {
            "slice_size": 640,
            "imgsz": 640,
            "overlap": 0.4,
            "augmentation_scope": "train_only_online",
        }
        assert item["compatibility"]["requires_manual_confirmation"] is False


def test_provenance_canonical_json_normalises_nonfinite_metadata():
    """DECISION-005: preprocessing provenance has stable canonical serialization."""
    from provenance import canonical_json_bytes

    encoded = canonical_json_bytes({"nodata": np.float32(np.nan)})
    assert encoded == b'{"nodata":null}'


def test_offline_manifest_checker_detects_route_tampering(tmp_path):
    """DECISION-001: exported route tampering is detectable offline."""
    import json
    from integrity_check import verify_manifest
    from provenance import file_sha256, make_stage_record

    output = tmp_path / "route.csv"
    output.write_text("lon,lat\n120,30\n", encoding="utf-8")
    record = make_stage_record(
        "inference", "v1", {"input": "fixed"}, np.ones((2, 2), dtype=np.uint8)
    )
    mask_record = make_stage_record("mask", "v1", {"input": "fixed"}, {"mask": 1})
    path_record = make_stage_record("path", "v1", {"input": "fixed"}, {"path": 1})
    manifest = {
        "output": {"path": str(output), "sha256": file_sha256(output)},
        "analysis": {"stage_provenance": {
            "inference": record,
            "mask": mask_record,
            "path": path_record,
        }},
    }
    manifest_path = tmp_path / "route.csv.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_manifest(manifest_path) == []
    output.write_text("lon,lat\n0,0\n", encoding="utf-8")
    assert "output SHA-256 does not match manifest" in verify_manifest(manifest_path)


def test_offline_manifest_checker_checks_every_machine_file_and_point_count(tmp_path):
    """DECISION-001: split machine files share one checked manifest contract."""
    import json
    from integrity_check import verify_manifest
    from provenance import file_sha256, make_stage_record

    first = tmp_path / "001.txt"
    second = tmp_path / "002.txt"
    infm = "$INFM,0,0.7,0.7,0.7,0.7,0,5,0*"
    first.write_text(
        "$PATH,0,0,2,30.00000000,116.00000000,0,0,2*\n"
        "$PATH,0,1,2,30.00100000,116.00000000,0,0,2*\n"
        f"{infm}\n",
        encoding="utf-8",
    )
    second.write_text(
        "$PATH,0,0,1,30.00200000,116.00000000,0,0,2*\n"
        f"{infm}\n",
        encoding="utf-8",
    )
    records = {
        stage: make_stage_record(stage, "v1", {"input": "fixed"}, {stage: 1})
        for stage in ("inference", "mask", "path")
    }
    manifest = {
        "output": {
            "path": str(first),
            "format": "$PATH_MACHINE_COMPAT",
            "sha256": file_sha256(first),
            "point_count": 3,
            "path_protocol": {
                "output_files": [str(first), str(second)],
                "points_per_file": [2, 1],
                "final_record": infm,
                "status_codes": {"file_record_status": 2},
                "output_file_integrity": [
                    {"path": str(first), "sha256": file_sha256(first), "point_count": 2},
                    {"path": str(second), "sha256": file_sha256(second), "point_count": 1},
                ],
            },
        },
        "analysis": {"stage_provenance": records},
    }
    manifest_path = tmp_path / "001.txt.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_manifest(manifest_path) == []
    second.write_text(
        "$PATH,0,0,2,30.00200000,116.00000000,0,0,2*\n"
        "$PATH,0,1,2,30.00300000,116.00000000,0,0,2*\n"
        f"{infm}\n",
        encoding="utf-8",
    )
    errors = verify_manifest(manifest_path)
    assert any("output file SHA-256 does not match manifest" in error for error in errors)
    assert any("point count does not match manifest" in error for error in errors)


def test_config_updates_are_atomic_and_temporary_overrides_are_restored():
    from config import Config, ConfigValidationError

    cfg = Config()
    original = cfg.snapshot()
    try:
        with pytest.raises(ConfigValidationError):
            cfg.update_section("model", {"tile_overlap": 1.0})
        assert cfg.snapshot() == original

        with pytest.raises(RuntimeError):
            with cfg.temporary_section("model", {"tile_overlap": 0.2}):
                assert cfg.TILE_OVERLAP == 0.2
                raise RuntimeError("stop")
        assert cfg.snapshot() == original
    finally:
        cfg.replace(original)
