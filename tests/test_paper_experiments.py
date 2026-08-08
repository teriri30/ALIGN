import csv
import json
from pathlib import Path

import numpy as np


def test_default_experiment_plan_disables_unimplemented_model_comparison():
    from paper_experiments import default_experiment_plan

    plan = default_experiment_plan(["a.pt", "b.pt"])

    assert [case["id"] for case in plan["mask_cases"]] == [
        "raw_mask_geometry",
        "band_geometry_reconstruction",
        "body_residual_recovery",
        "full_mask_regularization",
    ]
    assert {case["strategy"] for case in plan["path_cases"]} == {
        "bow", "semicircle", "pear", "fishtail",
    }
    assert [case["id"] for case in plan["path_cases"]] == [
        "bow_baseline", "bow_headland_optimized",
        "semicircle_baseline", "semicircle_headland_optimized",
        "pear_baseline", "pear_headland_optimized",
        "fishtail_baseline", "fishtail_headland_optimized",
    ]
    assert all("headland_turn_optimization_enabled" in case for case in plan["path_cases"])
    assert plan["path_cases"][0]["headland_turn_optimization_enabled"] is False
    assert plan["path_cases"][1]["headland_turn_optimization_enabled"] is True
    assert plan["model_cases"] == []
    assert plan["model_comparison"] == {
        "status": "disabled",
        "reason": "independent_model_inference_not_implemented",
    }
    assert plan["comparison_policy"]["same_inputs_required"] is True


def test_export_experiment_bundle_writes_tables_figures_cases_and_manifest(tmp_path):
    from paper_experiments import export_experiment_bundle

    preview = np.zeros((24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[5:18, 4:26] = 255
    records = [
        {
            "case_id": "auto",
            "module": "path",
            "status": "completed",
            "metrics": {"rolling_canopy_pct": 1.25, "total_length_m": 80.0},
            "parameters": {"turn_strategy": "auto"},
            "preview": preview,
            "mask": mask,
        },
        {
            "case_id": "pear",
            "module": "path",
            "status": "failed",
            "error": "no feasible turn",
            "metrics": {},
            "parameters": {"turn_strategy": "pear"},
        },
    ]

    output = export_experiment_bundle(
        str(tmp_path), records,
        experiment_inputs={"source_sha256": "abc", "model_sha256": "def"},
    )

    root = Path(output)
    assert (root / "experiment_manifest.json").is_file()
    assert (root / "results_long.csv").is_file()
    assert (root / "results_wide.csv").is_file()
    assert (root / "tables" / "path_comparison.csv").is_file()
    assert (root / "figures" / "path_metrics.png").is_file()
    assert (root / "cases" / "auto" / "preview.png").is_file()
    assert (root / "cases" / "auto" / "mask.png").is_file()
    assert (root / "README.md").is_file()

    manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "reui.paper-experiment.v1"
    assert manifest["completed_count"] == 1
    assert manifest["failed_count"] == 1
    assert manifest["inputs"]["source_sha256"] == "abc"

    with (root / "results_long.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["metric"] == "rolling_canopy_pct" and row["value"] == "1.25" for row in rows)


def test_runner_isolates_state_and_preserves_failed_cases(tmp_path):
    from types import SimpleNamespace
    from paper_experiments import PaperExperimentRunner, default_experiment_plan

    raw = np.zeros((20, 24), dtype=np.uint8)
    raw[3:17, 4:20] = 255
    source = np.zeros((20, 24, 3), dtype=np.uint8)
    state = SimpleNamespace(
        harvester_params={"turn_radius_m": 0.9},
        turn_strategy="bow",
        field_boundary=[(0, 0), (23, 0), (23, 19), (0, 19)],
        forbidden_regions=[],
        forbidden_regions_confirmed=True,
        entry_point=None,
        exit_point=None,
        unload_points=[],
        mask_offset_x=0,
        mask_offset_y=0,
    )

    def process(mask, **_kwargs):
        return {
            "processed_mask": mask.copy(),
            "regularized_body_mask": mask.copy(),
            "body_residual_mask": np.zeros_like(mask),
            "main_angle": 0.0,
            "wide_bands": [{"centerline": [(5, 5), (18, 5)]}],
            "headland_mask": np.zeros_like(mask),
            "planning_support_mask": mask.copy(),
            "uncertain_residual_mask": np.zeros_like(mask),
            "diagnostics": {"iou_to_raw": 1.0},
        }

    def plan(_bands, _mask, _angle, **kwargs):
        strategy = kwargs["config"]["turn_strategy"]
        if strategy == "pear":
            raise RuntimeError("infeasible")
        return {
            "full_path": [[(5, 5), (18, 5)]],
            "segment_types": ["work"],
            "validation": {
                "valid": True,
                "rolling_canopy_pct": 1.0,
                "crop_mask_geometric_coverage_pct": 95.0,
                "total_length_m": 13.0,
            },
            "turn_assessment": {"selected_candidate_strategy": strategy},
        }

    original_strategy = state.turn_strategy
    runner = PaperExperimentRunner(
        raw_mask=raw,
        source_image=source,
        state=state,
        geo=object(),
        output_dir=str(tmp_path),
        process_mask_fn=process,
        plan_path_fn=plan,
    )
    result = runner.run(plan=default_experiment_plan([]))

    assert state.turn_strategy == original_strategy
    assert Path(result).is_dir()
    manifest = json.loads((Path(result) / "experiment_manifest.json").read_text(encoding="utf-8"))
    pear_cases = [item for item in manifest["records"] if str(item["case_id"]).startswith("pear_")]
    assert len(pear_cases) == 2
    assert all(item["status"] == "failed" for item in pear_cases)
    assert manifest["completed_count"] >= 5


def test_runner_snapshot_preserves_mask_shape_for_field_boundary_validation(tmp_path):
    from types import SimpleNamespace
    from paper_experiments import PaperExperimentRunner

    raw = np.zeros((12, 16), dtype=np.uint8)
    raw[2:10, 3:13] = 255
    state = SimpleNamespace(
        mask_raw=raw.copy(),
        field_boundary=[(0, 0), (15, 0), (15, 11), (0, 11)],
        harvester_params={"cutter_forward_offset_m": 0.8},
        turn_strategy="bow",
        forbidden_regions=[],
        forbidden_regions_confirmed=True,
        entry_point=None,
        exit_point=None,
        unload_points=[],
        mask_offset_x=0,
        mask_offset_y=0,
    )
    captured = {}

    def process(mask, **_kwargs):
        return {
            "processed_mask": mask.copy(),
            "headland_mask": np.zeros_like(mask),
            "planning_support_mask": mask.copy(),
            "uncertain_residual_mask": np.zeros_like(mask),
            "wide_bands": [],
            "main_angle": 0.0,
        }

    def plan(_bands, _mask, _angle, **kwargs):
        captured["mask_raw"] = getattr(kwargs["state"], "mask_raw", None)
        return {
            "full_path": [],
            "segment_types": [],
            "validation": {"valid": False, "issues": ["test"]},
            "turn_assessment": {},
        }

    runner = PaperExperimentRunner(
        raw_mask=raw,
        source_image=np.zeros((12, 16, 3), dtype=np.uint8),
        state=state,
        geo=object(),
        output_dir=str(tmp_path),
        process_mask_fn=process,
        plan_path_fn=plan,
    )
    runner.run(plan={
        "mask_cases": [],
        "path_cases": [{
            "id": "boundary_snapshot",
            "strategy": "bow",
            "headland_turn_optimization_enabled": True,
        }],
    })

    assert isinstance(captured["mask_raw"], np.ndarray)
    assert captured["mask_raw"].shape == raw.shape


def test_experiment_worker_emits_token_bound_export_result(tmp_path):
    from types import SimpleNamespace
    from pyside6_app.workers import PaperExperimentWorker
    from task_lifecycle import make_task_token

    state = SimpleNamespace(project_generation=1, source_sha256="source-a", model_sha256="model-a")
    token = make_task_token(state, "paper_experiment")
    runner = SimpleNamespace(run=lambda plan: str(tmp_path))
    worker = PaperExperimentWorker(
        runner, {"mask_cases": [], "path_cases": []}, token=token,
    )
    finished = []
    worker.finished.connect(finished.append)

    worker.run()

    assert finished == [{
        "task_token": token,
        "status": "completed",
        "result": {"output_dir": str(tmp_path)},
        "error": "",
    }]


def test_decision_014_experiment_menu_is_removed_from_desktop():
    """DECISION-014: offline experiments have no desktop execution entry."""
    import os
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication
    from pyside6_app.main_window import MainWindow

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["REUI_DISABLE_AUTO_START"] = "1"
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    menus = [action.text() for action in window.menuBar().actions()]
    assert "实验" not in menus
    assert not any(
        "消融实验" in action.text() for action in window.findChildren(QAction)
    )
    window.close()


def test_case_preview_places_local_mask_at_global_offset():
    from paper_experiments import _render_case_preview

    image = np.zeros((8, 10, 3), dtype=np.uint8)
    mask = np.ones((2, 3), dtype=np.uint8) * 255
    preview = _render_case_preview(
        image,
        {"processed_mask": mask},
        {"full_path": [[(5, 4), (7, 4)]]},
        (5, 3),
    )

    assert np.any(preview[3:5, 5:8])
    assert not np.any(preview[:3, :5])


def test_intermediate_ablation_metrics_are_recomputed_from_each_case_mask():
    from paper_experiments import _intermediate_mask_result

    raw = np.zeros((4, 4), dtype=np.uint8)
    raw[:2, :] = 255
    regularized = np.zeros_like(raw)
    regularized[0, :] = 255
    result = _intermediate_mask_result(
        "band_geometry_reconstruction",
        {
            "raw_mask": raw,
            "regularized_body_mask": regularized,
            "body_residual_mask": np.zeros_like(raw),
            "diagnostics": {"iou_to_raw": 0.99},
        },
        raw_mask=raw,
    )

    assert result["diagnostics"]["precision_to_raw"] == 1.0
    assert result["diagnostics"]["recall_to_raw"] == 0.5
    assert result["diagnostics"]["iou_to_raw"] == 0.5


def test_supervised_model_metrics_are_only_emitted_with_ground_truth():
    from paper_experiments import segmentation_metrics

    predicted = np.asarray([[0, 255], [0, 255]], dtype=np.uint8)
    assert segmentation_metrics(predicted, None) == {}

    truth = np.asarray([[0, 255], [255, 255]], dtype=np.uint8)
    metrics = segmentation_metrics(predicted, truth)
    assert metrics["intersection_over_union"] == 2 / 3
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 2 / 3


def test_export_experiment_bundle_writes_per_segment_path_metrics(tmp_path):
    from paper_experiments import export_experiment_bundle

    export_experiment_bundle(
        str(tmp_path),
        [{
            "case_id": "bow_baseline",
            "module": "path",
            "status": "completed",
            "metrics": {"total_length_m": 15.0},
            "segment_metrics": [
                {"segment_index": 0, "segment_type": "work", "length_m": 12.0, "track_swept_area_m2": 2.0},
                {"segment_index": 1, "segment_type": "turn", "length_m": 3.0, "track_swept_area_m2": 1.0},
            ],
        }],
    )

    path = tmp_path / "path_segments.csv"
    assert path.is_file()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["segment_type"] for row in rows] == ["work", "turn"]
    assert rows[1]["track_swept_area_m2"] == "1.0"


def test_segment_type_summary_reports_required_turn_semantics_with_zero_fill():
    from paper_experiments import _segment_type_summary_metrics

    metrics = _segment_type_summary_metrics([
        {"segment_type": "work", "length_m": 12.0, "track_swept_area_m2": 2.0},
        {"segment_type": "turn", "length_m": 3.0, "track_swept_area_m2": 1.0},
    ])

    assert metrics["work_length_m"] == 12.0
    assert metrics["turn_length_m"] == 3.0
    assert metrics["turn_approach_length_m"] == 0.0
    assert metrics["turn_reverse_length_m"] == 0.0
    assert metrics["entry_length_m"] == 0.0
    assert metrics["exit_length_m"] == 0.0
    assert metrics["turn_track_swept_area_m2"] == 1.0
    assert metrics["turn_approach_track_swept_area_m2"] == 0.0

def test_real_baseline_status_command_is_executable_and_not_placeholder():
    import json

    status_path = Path("experiments/real_yolo11m_baseline_20260722/baseline_status.json")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert "..." not in payload["command"]
    assert "ultralytics" in payload["command"]
    assert "yolo11m_seg_best.pt" in payload["command"]
    assert payload["no_inference_result_published"] is True
