import csv
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np
from affine import Affine
from pyproj import Transformer
from PySide6.QtWidgets import QApplication

from geo import GeoUtils
from pyside6_app.main_window import MainWindow
from pyside6_app.workers import bounded_render_preview, display_affine, preview_shape


def _real_geo():
    affine = Affine(0.2, 0.0, 500000.0, 0.0, -0.2, 3400000.0)
    transformer = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
    geo = GeoUtils()
    geo.set_affine(affine, transformer, source_crs="EPSG:32650")
    return geo, affine, transformer


def _window_with_path():
    import numpy as np
    from inference import INFERENCE_PIPELINE_VERSION
    from path_planner import PATH_PLANNING_VERSION
    from provenance import file_sha256, make_stage_record
    from pyside6_app.workers import (
        planning_config_with_mask_evidence,
        semantic_mask_fingerprints,
    )
    from row_geometry import MASK_REGULARIZATION_VERSION

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window.geo, _affine, _transformer = _real_geo()
    model_path = os.path.join(tempfile.gettempdir(), "reui-integrity-test-model.pt")
    with open(model_path, "wb") as handle:
        handle.write(b"test-model")
    model_sha256 = file_sha256(model_path)
    raw_mask = np.ones((40, 40), dtype=np.uint8)
    processed_mask = raw_mask.copy()
    headland_mask = np.zeros_like(processed_mask)
    headland_mask[:4, :] = 255
    planning_support_mask = processed_mask.copy()
    neutral_support_mask = np.zeros_like(processed_mask)
    uncertain_residual_mask = np.zeros_like(processed_mask)
    inference_inputs = {
        "source_sha256": "test-source-sha256",
        "model_sha256": model_sha256,
        "inference_config": {
            "capture_size": int(window.cfg.TILE_CAPTURE_SIZE),
            "overlap": float(window.cfg.TILE_OVERLAP),
            "conf": float(window.cfg.MODEL_CONF),
            "iou": float(window.cfg.MODEL_IOU),
            "batch_size": 4,
        },
        "preprocessing_config": window.cfg.section("raster_preprocessing"),
    }
    inference_record = make_stage_record(
        "inference", INFERENCE_PIPELINE_VERSION, inference_inputs, raw_mask
    )
    mask_inputs = {
        "inference_fingerprint": inference_record["fingerprint"],
        "raw_mask_sha256": inference_record["artifact_sha256"],
        "mask_config": window.cfg.section("mask_processing"),
    }
    mask_record = make_stage_record(
        "mask", MASK_REGULARIZATION_VERSION, mask_inputs, processed_mask
    )
    path_geometry = [[(10.25, 20.75), (15.5, 25.125)]]
    path_layout = {}
    harvester = window._effective_harvester_params(window.state.harvester_params)
    window.state.harvester_param_sources = {
        key: "user_measured" for key in harvester
    }
    path_config = window.cfg.section("path_planning")
    path_config.update({
        "min_turn_radius_m": float(harvester["turn_radius_m"]),
        "turn_strategy": str(window.state.turn_strategy),
        "forbidden_mask_confirmed": False,
    })
    path_inputs = {
        "mask_fingerprint": mask_record["fingerprint"],
        "path_config": path_config,
        "harvester": harvester,
        "harvester_param_sources": dict(window.state.harvester_param_sources),
        "turn_strategy": str(window.state.turn_strategy),
        "entry_point": window.state.entry_point,
        "exit_point": window.state.exit_point,
        "unload_points": list(window.state.unload_points),
        "forbidden_regions": [],
        "forbidden_regions_confirmed": False,
    }
    semantic_config = planning_config_with_mask_evidence(
        path_config,
        {
            "processed_mask": processed_mask,
            "headland_mask": headland_mask,
            "planning_support_mask": planning_support_mask,
            "neutral_support_mask": neutral_support_mask,
            "uncertain_residual_mask": uncertain_residual_mask,
        },
        window.state,
        processed_mask.shape,
    )
    path_inputs["semantic_mask_fingerprints"] = semantic_mask_fingerprints(
        semantic_config
    )
    path_record = make_stage_record(
        "path",
        PATH_PLANNING_VERSION,
        path_inputs,
        {
            "full_path": path_geometry,
            "segment_types": ["work"],
            "layout": path_layout,
        },
    )
    window.state.safe_update(
        source_sha256="test-source-sha256",
        current_model_path=model_path,
        model_sha256=model_sha256,
        mask_raw=raw_mask,
        inference_provenance=inference_record,
        mask_result={
            "processed_mask": processed_mask,
            "headland_mask": headland_mask,
            "planning_support_mask": planning_support_mask,
            "neutral_support_mask": neutral_support_mask,
            "uncertain_residual_mask": uncertain_residual_mask,
            "provenance": mask_record,
        },
        mask_provenance=mask_record,
        path_provenance=path_record,
    )
    window._store_path_result({
        "full_path": path_geometry,
        "segment_types": ["work"],
        "layout": path_layout,
        "geo_points": [
            {
                "lon": 1.0,
                "lat": 1.0,
                "pixel_x": 10.25,
                "pixel_y": 20.75,
                "segment_index": 0,
                "point_index": 0,
                "segment_type": "work",
            },
            {
                "lon": 2.0,
                "lat": 2.0,
                "pixel_x": 15.5,
                "pixel_y": 25.125,
                "segment_index": 0,
                "point_index": 1,
                "segment_type": "work",
            },
        ],
        "validation": {"valid": True},
        "is_valid": True,
        "provenance": path_record,
    })
    return window


def test_utm_conversion_outputs_wgs84_and_preserves_fractional_pixels():
    geo, affine, transformer = _real_geo()
    px, py = 10.25, 20.75

    lon, lat = geo.pixel_to_lonlat(px, py)
    source_x, source_y = affine * (px + 0.5, py + 0.5)
    expected_lon, expected_lat = transformer.transform(source_x, source_y)

    assert lon == pytest.approx(expected_lon, abs=1e-12)
    assert lat == pytest.approx(expected_lat, abs=1e-12)
    assert 116.0 < lon < 118.0
    assert 30.0 < lat < 32.0

    truncated = geo.pixel_to_lonlat(int(px), int(py))
    assert math.hypot(lon - truncated[0], lat - truncated[1]) > 1e-10


def test_physical_distance_uses_ground_geodesic_not_projected_map_units():
    to_web = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    origin_x, origin_y = to_web.transform(10.0, 60.0)
    geo = GeoUtils()
    geo.set_affine(
        Affine(1.0, 0.0, origin_x, 0.0, -1.0, origin_y),
        to_wgs84,
        source_crs="EPSG:3857",
    )

    distance = geo.pixel_distance_m((0.0, 0.0), (100.0, 0.0))

    assert 49.0 < distance < 51.0
    assert distance != pytest.approx(100.0, abs=1.0)


def test_display_affine_uses_exact_non_divisible_raster_ratio():
    source = Affine(0.3, 0.0, 500000.0, 0.0, -0.3, 3400000.0)
    full_w, full_h = 21021, 15078
    display_w, display_h = full_w // 4, full_h // 4

    transformed = display_affine(source, full_w, full_h, display_w, display_h)

    assert transformed * (display_w, display_h) == pytest.approx(
        source * (full_w, full_h), abs=1e-9
    )
    assert transformed.a != pytest.approx((source * Affine.scale(4, 4)).a, abs=1e-12)


def test_preview_shape_bounds_render_pixels_without_changing_source_footprint():
    """DECISION-003/004: UI rendering may shrink, but its affine still covers the source."""
    source = Affine(0.0035, 0.0, 447411.3937887839, 0.0, -0.0035, 3385528.6805868475)
    full_h, full_w = 32190, 22012

    display_h, display_w = preview_shape(full_h, full_w, 4096)
    transformed = display_affine(source, full_w, full_h, display_w, display_h)

    assert (display_h, display_w) == (4096, 2801)
    assert max(display_h, display_w) <= 4096
    assert transformed * (display_w, display_h) == pytest.approx(
        source * (full_w, full_h), abs=1e-9
    )


def test_low_resolution_render_keeps_logical_analysis_coordinates():
    """DECISION-004: a smaller Qt texture must not alter mask/path scene coordinates."""
    from pyside6_app.image_view import ImageViewer

    QApplication.instance() or QApplication([])
    viewer = ImageViewer()
    viewer.set_image(
        np.zeros((2, 4, 3), dtype=np.uint8),
        logical_shape=(4, 8),
    )

    bounds = viewer._pixmap_item.sceneBoundingRect()
    assert viewer._image_shape == (4, 8)
    assert bounds.width() == pytest.approx(8.0)
    assert bounds.height() == pytest.approx(4.0)


def test_bounded_render_preview_limits_texture_without_mutating_analysis_pixels():
    """DECISION-004: render decimation is a separate UI artifact."""
    analysis_rgb = np.arange(5 * 10 * 3, dtype=np.uint8).reshape(5, 10, 3)

    render_rgb = bounded_render_preview(analysis_rgb, max_dimension=4)

    assert analysis_rgb.shape == (5, 10, 3)
    assert render_rgb.shape == (2, 4, 3)
    assert max(render_rgb.shape[:2]) <= 4


def test_tif_load_commit_clears_previous_project_analysis(tmp_path, monkeypatch):
    from pyside6_app.main_window import MainWindow
    from state import AutoPathSegment

    QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "new.tif"
    source.write_bytes(b"source")
    monkeypatch.setattr(window, "_queue_cache_restore_after_current_worker", lambda _path: None)
    window.state.field_boundary = [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0)]
    window.state.mask_raw = np.ones((2, 2), dtype=np.uint8)
    window.state.mask_result = {"processed_mask": np.ones((2, 2), dtype=np.uint8)}
    window.state.inference_done = True
    window.state.mask_processed = True
    window.state.auto_path = [AutoPathSegment()]
    window.state.auto_path_planned = True
    window._pipeline_result = {"path": {"is_valid": True}}

    window._on_tif_load_done({
        "path": str(source),
        "rgb": np.zeros((5, 5, 3), dtype=np.uint8),
        "full_h": 5,
        "full_w": 5,
        "display_h": 5,
        "display_w": 5,
        "downsample": 1,
        "transformer": None,
        "transform": None,
        "crs": None,
        "source_sha256": "new-source",
        "source_metadata": {},
        "preview_preprocessing": {},
    })

    assert window.state.field_boundary == []
    assert window.state.mask_raw is None
    assert window.state.mask_result is None
    assert window.state.inference_done is False
    assert window.state.mask_processed is False
    assert window.state.auto_path == []
    assert window.state.auto_path_planned is False
    assert window._pipeline_result is None


def test_tif_loader_builds_wgs84_transformer_from_source_crs(tmp_path):
    import numpy as np
    import rasterio
    from pyside6_app.workers import TifLoadWorker

    tif_path = tmp_path / "projected.tif"
    source_affine = Affine(0.2, 0.0, 500000.0, 0.0, -0.2, 3400000.0)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=13,
        height=11,
        count=1,
        dtype="uint8",
        crs="EPSG:32650",
        transform=source_affine,
    ) as dataset:
        dataset.write(np.ones((1, 11, 13), dtype=np.uint8))

    results = []
    errors = []
    worker = TifLoadWorker(str(tif_path))
    worker.finished.connect(results.append)
    worker.error.connect(errors.append)
    worker.run()

    assert errors == []
    assert len(results) == 1
    result = results[0]
    assert result["crs"] == "EPSG:32650"
    source_x, source_y = result["transform"] * (0.5, 0.5)
    lon, lat = result["transformer"].transform(source_x, source_y)
    assert 116.0 < lon < 118.0
    assert 30.0 < lat < 32.0


def test_tif_loader_rejects_crs_without_valid_affine_geotransform(tmp_path):
    """DECISION-003: a CRS label alone cannot establish usable georeferencing."""
    import numpy as np
    import rasterio
    from pyside6_app.workers import TifLoadWorker

    tif_path = tmp_path / "identity-transform.tif"
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning):
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            width=5,
            height=5,
            count=1,
            dtype="uint8",
            crs="EPSG:32650",
            transform=Affine.identity(),
        ) as dataset:
            dataset.write(np.ones((1, 5, 5), dtype=np.uint8))

    results = []
    worker = TifLoadWorker(str(tif_path))
    worker.finished.connect(results.append)
    worker.run()

    assert results[0]["transformer"] is None
    assert "no valid affine geotransform" in results[0]["geo_error"]


def test_loading_unreferenced_image_clears_previous_georeference(tmp_path, monkeypatch):
    import numpy as np

    window = _window_with_path()
    assert window.geo.is_ready()
    source = tmp_path / "unreferenced.tif"
    source.write_bytes(b"not used by raster reader in this unit test")
    monkeypatch.setattr(window, "_start_cache_restore", lambda _path: None)

    window._on_tif_load_done({
        "path": str(source),
        "rgb": np.zeros((5, 5, 3), dtype=np.uint8),
        "full_h": 5,
        "full_w": 5,
        "display_h": 5,
        "display_w": 5,
        "downsample": 1,
        "crs": "",
        "transform": Affine.identity(),
        "transformer": None,
        "geo_error": "GeoTIFF has no CRS",
    })

    assert not window.geo.is_ready()


def test_export_payload_recomputes_stale_coordinates_from_display_path():
    window = _window_with_path()
    path_result, geo_points = window._geo_export_payload()

    assert geo_points[0]["lon"] != 1.0
    assert geo_points[1]["lat"] != 2.0
    assert path_result["geo_points"] == geo_points
    assert window.state.auto_path_geo == geo_points
    assert [(point["pixel_x"], point["pixel_y"]) for point in geo_points] == [
        (10.25, 20.75),
        (15.5, 25.125),
    ]


def test_export_rejects_forbidden_region_change_after_planning():
    window = _window_with_path()
    window.state.forbidden_regions = [[(1, 1), (3, 1), (3, 3), (1, 3)]]
    window.state.forbidden_regions_confirmed = True

    with pytest.raises(ValueError, match="input fingerprint"):
        window._geo_export_payload()


def test_export_rejects_semantic_mask_change_after_path_planning():
    """DECISION-001/002: formal output requires current semantic mask evidence."""
    window = _window_with_path()
    window.state.mask_result["planning_support_mask"][0, 0] = 0

    with pytest.raises(ValueError, match="input fingerprint"):
        window._geo_export_payload()


def test_export_rejects_service_point_change_after_path_planning():
    """DECISION-007: changed service points invalidate formal route export."""
    window = _window_with_path()
    window.state.entry_point = (5.0, 5.0)

    with pytest.raises(ValueError, match="input fingerprint"):
        window._geo_export_payload()


def test_export_rejects_model_content_change_after_inference():
    """DECISION-006: changed model bytes invalidate downstream formal output."""
    window = _window_with_path()
    with open(window.state.current_model_path, "wb") as handle:
        handle.write(b"changed-model")

    with pytest.raises(ValueError, match="input fingerprint"):
        window._geo_export_payload()


def test_export_blocks_path_points_outside_loaded_image():
    import numpy as np

    window = _window_with_path()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window._pipeline_result["path"]["full_path"] = [[(10.0, 10.0), (25.0, 10.0)]]

    with pytest.raises(ValueError, match="pixel="):
        window._geo_export_payload()


def test_export_blocks_when_loaded_geotiff_changes_on_disk(tmp_path):
    from cache import source_identity

    window = _window_with_path()
    source = tmp_path / "field.tif"
    source.write_bytes(b"original")
    window._tif_path = str(source)
    window._tif_source_identity = source_identity(str(source))
    source.write_bytes(b"changed")

    with pytest.raises(ValueError, match="GeoTIFF"):
        window._geo_export_payload()


def test_all_export_formats_round_trip_the_same_wgs84_points(tmp_path, monkeypatch):
    from path_planner import export_csv, export_json, export_kml, export_path_format
    from pyside6_app import main_window as main_window_module

    window = _window_with_path()
    _path_result, points = window._geo_export_payload()

    csv_path = tmp_path / "route.csv"
    json_path = tmp_path / "route.json"
    kml_path = tmp_path / "route.kml"
    machine_path = tmp_path / "route.path"
    geojson_path = tmp_path / "route.geojson"

    export_csv(points, str(csv_path))
    export_json(points, {"valid": True}, str(json_path))
    export_kml(points, str(kml_path))
    export_path_format(points, str(machine_path))
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(geojson_path), "GeoJSON (*.geojson)"),
    )
    monkeypatch.setattr(window.log_panel, "success", lambda _message: None)
    window._on_export_geojson()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_points = [(float(row["lon"]), float(row["lat"])) for row in csv.DictReader(handle)]
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    json_points = [(float(point["lon"]), float(point["lat"])) for point in json_data["path"]]
    assert json_data["crs"] == "EPSG:4326"

    root = ET.parse(kml_path).getroot()
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    coordinates = root.find(".//kml:coordinates", namespace).text.strip().splitlines()
    kml_points = [tuple(map(float, item.strip().split(",")[:2])) for item in coordinates]

    path_rows = [line for line in machine_path.read_text(encoding="utf-8").splitlines() if line.startswith("$PATH")]
    machine_points = []
    for row in path_rows:
        fields = row.rstrip("*").split(",")
        machine_points.append((float(fields[5]), float(fields[4])))

    geojson_data = json.loads(geojson_path.read_text(encoding="utf-8"))
    geojson_points = [
        tuple(coordinate)
        for feature in geojson_data["features"]
        for coordinate in feature["geometry"]["coordinates"]
    ]
    expected = [(point["lon"], point["lat"]) for point in points]

    for actual in (csv_points, json_points, kml_points, geojson_points):
        assert actual == pytest.approx(expected, abs=1e-12)
    for actual, reference in zip(machine_points, expected):
        assert actual[0] == pytest.approx(reference[0], abs=1e-8)
        assert actual[1] == pytest.approx(reference[1], abs=1e-8)


def test_cache_is_rejected_when_source_file_changes(tmp_path, monkeypatch):
    import cache
    from state import AppState

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    source = tmp_path / "field.tif"
    source.write_bytes(b"original geotiff header and body")
    state = AppState()
    state.tif_path = str(source)

    cache.save_project_state(str(source), state, stage="test")
    assert cache.load_project_state(str(source))["schema"] == cache.PROJECT_STATE_SCHEMA

    source.write_bytes(b"changed geotiff header and body")
    assert cache.load_project_state(str(source)) == {}


def test_legacy_export_engine_is_disabled_to_prevent_bypassing_export_gate(tmp_path):
    from export_guard import ExportEngine
    from state import AppState

    state = AppState()
    state.path_points = [(10.25, 20.75), (15.5, 25.125)]
    state.path_status = [0]
    output = tmp_path / "legacy.path"

    with pytest.raises(RuntimeError, match="ExportEngine"):
        ExportEngine(_real_geo()[0], state).export_to_file(str(output))
    assert not output.exists()
