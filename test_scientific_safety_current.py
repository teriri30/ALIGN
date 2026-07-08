import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication


class _NoGeo:
    def is_ready(self):
        return False


class _MetricGeo:
    def is_ready(self):
        return True

    def meters_per_pixel(self, _x, _y):
        return 0.01

    def pixel_to_lonlat(self, x, y):
        return 120.0 + float(x) * 1e-6, 30.0 + float(y) * 1e-6

    def pixel_distance_m(self, first, second):
        return float(np.hypot(second[0] - first[0], second[1] - first[1])) * 0.01


def test_metric_mask_pipeline_failure_is_not_silently_downgraded(monkeypatch):
    import mask_processor

    monkeypatch.setattr(
        mask_processor,
        "regularize_crop_mask",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("metric failure")),
    )

    with pytest.raises(RuntimeError, match="metric failure"):
        mask_processor.process_mask(
            np.ones((20, 20), dtype=np.uint8) * 255,
            geo=_MetricGeo(),
            state=type("State", (), {"mask_offset_x": 0, "mask_offset_y": 0})(),
            config={"strength": "standard"},
        )


def test_failed_rerun_cannot_mark_old_mask_as_new_success(monkeypatch, tmp_path):
    import model
    import rasterio
    from state import AppState

    class ImmediateThread:
        def __init__(self, target, args=(), daemon=None, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    class Engine:
        def is_loaded(self):
            return True

    monkeypatch.setattr(model.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(rasterio, "open", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("read failed")))

    state = AppState()
    state.mask_raw = np.ones((4, 4), dtype=np.uint8)
    state.inference_done = True
    state.field_boundary = [(0, 0), (3, 0), (3, 3), (0, 3)]
    runner = model.InferenceRunner(state, Engine(), _MetricGeo())

    runner.start_from_tif(str(tmp_path / "broken.tif"), (4, 4), state.field_boundary, 1)

    assert state.mask_raw is None
    assert state.inference_done is False
    assert state.inference_running is False


def test_failed_inference_tile_aborts_instead_of_becoming_background():
    """DECISION-004: a missing tile is a failed inference, not valid background."""
    from model import TiledInference

    class FailingEngine:
        def is_loaded(self):
            return True

        def predict(self, *_args, **_kwargs):
            return None

    with pytest.raises(RuntimeError, match="tile inference failed"):
        TiledInference(FailingEngine()).run(
            np.zeros((20, 20, 3), dtype=np.uint8),
            capture_sz=16,
            overlap=0.25,
            batch_size=2,
        )


def test_route_info_theme_refresh_does_not_call_task_panel_methods():
    from pyside6_app.panels import RouteInfoPanel

    app = QApplication.instance() or QApplication([])
    panel = RouteInfoPanel()
    panel.update_route({"tracks": [{"points": [(0, 0), (1, 0)]}], "turns": []})

    panel.refresh_theme()

    assert panel._summary.text().startswith("预计碾压")
    assert "作业线 1" in panel._summary.toolTip()
    panel.deleteLater()
    app.processEvents()


def test_route_info_uses_defined_rolling_metric_and_plain_forbidden_status():
    from pyside6_app.panels import RouteInfoPanel

    app = QApplication.instance() or QApplication([])
    panel = RouteInfoPanel()
    panel.update_route({
        "validation": {
            "rolling_canopy_pct": 2.5,
            "track_core_overlap_pct": 17.0,
            "track_forbidden_overlap_pct": 0.0,
        }
    })

    summary = panel._summary.text()
    assert "预计碾压 2.5%" in summary
    assert "17.0%" not in summary
    assert "禁行区 无侵入" in summary
    panel.deleteLater()
    app.processEvents()


def test_route_info_distinguishes_approach_turn_reverse_and_service_segments():
    from pyside6_app.panels import RouteInfoPanel, RouteSegmentRow

    app = QApplication.instance() or QApplication([])
    panel = RouteInfoPanel()
    panel.update_route({
        "layout": {"work_line_mode": "band_centerline"},
        "validation": {
            "validation_profile": "research",
            "validation_limits_pct": {
                "track_outside_field": 2.0,
                "track_outside_support": 100.0,
                "track_uncertain_overlap": 100.0,
                "track_forbidden_overlap": 0.0,
            },
        },
        "machine_readiness": {
            "eligible_for_machine_export": False,
            "blockers": ["footprint_optimized_work_lines_not_used"],
            "controlled_trial_blockers": ["footprint_optimized_work_lines_not_used"],
        },
        "ordered_segments": [
            {"segment_index": 0, "type": "work", "points": [(0, 0), (1, 0)]},
            {"segment_index": 1, "type": "turn_approach", "points": [(1, 0), (2, 0)]},
            {"segment_index": 2, "type": "turn", "points": [(2, 0), (2, 1)]},
            {"segment_index": 3, "type": "turn_reverse", "points": [(2, 1), (1, 1)]},
            {"segment_index": 4, "type": "exit", "points": [(1, 1), (0, 1)]},
        ]
    })

    summary = panel._summary.toolTip()
    assert "作业线 1" in summary
    assert "正式调头 1" in summary
    assert "接近 1" in summary
    assert "倒车 1" in summary
    assert "进出田/卸粮 1" in summary
    assert panel._safety.text() == "路径状态：不建议导出"
    assert "足迹优化" in panel._safety.toolTip()
    assert "支撑<=100" in panel._safety.toolTip()
    assert [row._segment_type for row in panel.findChildren(RouteSegmentRow)] == [
        "work", "turn_approach", "turn", "turn_reverse", "exit"
    ]
    panel.deleteLater()
    app.processEvents()


def test_invalid_or_manually_edited_path_is_blocked_from_export():
    from pyside6_app.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window.geo = _MetricGeo()
    window._pipeline_result = {
        "path": {
            "full_path": [[(1.0, 1.0), (2.0, 2.0)]],
            "segment_types": ["work"],
            "validation": {"valid": False, "issues": ["manual edit"]},
            "is_valid": False,
        }
    }

    with pytest.raises(ValueError, match="未通过安全验证"):
        window._geo_export_payload()


def test_metric_processing_requires_valid_georeference():
    """DECISION-003: metric algorithms fail closed without reliable georeferencing."""
    from pyside6_app.workers import require_metric_scale
    from state import AppState

    with pytest.raises(RuntimeError, match="有效地理配准"):
        require_metric_scale(_NoGeo(), AppState(), (100, 100))


def test_export_blocks_when_harvester_parameters_changed_after_planning():
    """DECISION-007: changed machine constraints invalidate planned output."""
    from pyside6_app.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window.geo = _MetricGeo()
    window.state.harvester_params = window._effective_harvester_params({})
    window.state.harvester_params["cutter_forward_offset_m"] = 1.2
    planned = dict(window.state.harvester_params)
    planned["track_width_m"] += 0.1
    window._pipeline_result = {
        "path": {
            "full_path": [[(1.0, 1.0), (2.0, 2.0)]],
            "segment_types": ["work"],
            "validation": {"valid": True, "issues": []},
            "is_valid": True,
            "planning_factors": {"harvester": planned},
        }
    }

    with pytest.raises(ValueError, match="农机参数与路径规划时不一致"):
        window._geo_export_payload()


def test_input_changes_invalidate_stale_downstream_results():
    """DECISION-007: scientific input changes clear stale downstream evidence."""
    from pyside6_app.main_window import MainWindow
    from state import AutoPathSegment

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window.state.auto_path = [AutoPathSegment()]
    window.state.auto_path_segments = [AutoPathSegment()]
    window.state.auto_path_planned = True
    window.state.auto_path_valid = True
    window.state.path_points = [(1.0, 1.0), (2.0, 2.0)]
    window.state.simulation_done = True
    window.state.export_done = True
    window._pipeline_result = {"path": {"full_path": [[(1.0, 1.0), (2.0, 2.0)]]}}

    window._invalidate_analysis_from("path", "test")

    assert window.state.auto_path == []
    assert window.state.auto_path_planned is False
    assert window.state.auto_path_valid is False
    assert window.state.path_points == []
    assert window.state.simulation_done is False
    assert window.state.export_done is False
    assert window._pipeline_result is None


def test_queued_project_save_serializes_enqueue_time_snapshot(tmp_path, monkeypatch):
    import cache
    from state import AppState

    class DeferredThread:
        targets = []

        def __init__(self, target, **_kwargs):
            self.target = target
            self.__class__.targets.append(target)

        def start(self):
            pass

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(cache.threading, "Thread", DeferredThread)
    cache._PENDING_STATE_SAVES.clear()
    cache._ACTIVE_STATE_SAVERS.clear()
    source = tmp_path / "field.tif"
    source.write_bytes(b"source")
    state = AppState()
    state.tif_path = str(source)
    state.field_area_m2 = 10.0

    cache.request_project_state_save(str(source), state, stage="queued")
    state.field_area_m2 = 99.0
    DeferredThread.targets[-1]()

    restored = cache.load_project_state(str(source))
    assert restored["field_area_m2"] == 10.0


def test_route_export_manifest_records_integrity_and_validation(tmp_path):
    import hashlib
    import json
    from pyside6_app.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    output = tmp_path / "route.csv"
    output.write_text("lon,lat\n120,30\n", encoding="utf-8")
    points = [{
        "lon": 120.0,
        "lat": 30.0,
        "pixel_x": 1.0,
        "pixel_y": 2.0,
        "segment_index": 0,
        "point_index": 0,
        "segment_type": "work",
        "status": 1,
    }]
    path_result = {
        "validation": {"valid": True, "issues": []},
        "planning_factors": {"harvester": {"track_width_m": 0.45}},
    }
    window.state.source_sha256 = "test-source-sha256"

    manifest_path = window._write_export_manifest(
        str(output), "CSV", path_result, points
    )
    manifest = json.loads(open(manifest_path, "r", encoding="utf-8").read())

    assert manifest["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["analysis"]["validation"]["valid"] is True
    assert manifest["application_version"]
    assert manifest["scientific_code"]["fingerprint"]
    assert manifest["source_image"]["sha256"] == "test-source-sha256"
    assert manifest["output"]["route_classification"] == "research_manual_review"
    assert set(manifest["analysis"]["stage_provenance"]) == {"inference", "mask", "path"}
    assert manifest["integrity"]["config_sha256"]
    assert manifest["integrity"]["route_data_sha256"]

    trial_output = tmp_path / "trial.path"
    trial_output.write_text("$PATH,0,0,1,30,120,0,0,1*\n", encoding="utf-8")
    path_result["machine_readiness"] = {"controlled_trial_ready": True}
    trial_manifest_path = window._write_export_manifest(
        str(trial_output), "$PATH", path_result, points
    )
    trial_manifest = json.loads(open(trial_manifest_path, "r", encoding="utf-8").read())
    assert trial_manifest["output"]["route_classification"] == "controlled_field_trial"
    assert "LOW_SPEED_MANUAL_SUPERVISION" in trial_manifest["output"]["usage_restriction"]
    assert trial_manifest["output"]["machine_executable"] is False
    assert trial_manifest["output"]["candidate_machine_path"] is True
    assert trial_manifest["output"]["terminal_protocol_validated"] is False
    assert trial_manifest["output"]["path_protocol"]["coordinate_order"] == [
        "latitude", "longitude"
    ]
    assert trial_manifest["output"]["path_protocol"]["point_index_base"] == 0
    assert trial_manifest["output"]["internal_path_reference_point"] == "track_contact_center"
    assert trial_manifest["output"]["footprint_model"] == "planar_rigid_footprint_v2"


def test_machine_parameter_output_is_explicitly_unvalidated_heuristic():
    """DECISION-008: machine-size advice remains labelled as an unvalidated heuristic."""
    from pyside6_app.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    recommendation = window._build_machine_recommendation({
        "full_path": [[(0.0, 0.0), (1.0, 1.0)]],
        "segment_types": ["work"],
        "validation": {},
    })

    assert recommendation["validated"] is False
    assert recommendation["method"] == "mask_geometry_heuristic_v2"
    assert "没有 DEM" in recommendation["message"]
