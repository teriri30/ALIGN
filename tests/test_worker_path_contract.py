from types import SimpleNamespace

import pytest


def test_pipeline_and_plan_workers_share_default_plus_runtime_path_config(monkeypatch):
    """DECISION-013: both worker entry points execute one equivalent config snapshot."""
    from config import Config
    from pyside6_app.workers import PipelineWorker, PlanWorker

    defaults = {
        "headland_damage_weight": 0.75,
        "validation_profile": "field_trial",
        "max_track_core_overlap_pct": 2.0,
    }
    monkeypatch.setattr(
        Config,
        "section",
        lambda self, name: dict(defaults) if name == "path_planning" else {},
    )
    state = SimpleNamespace()
    runtime = {"min_turn_radius_m": 1.2, "turn_strategy": "dubins"}

    plan_worker = PlanWorker({}, None, state, runtime)
    pipeline_worker = PipelineWorker(
        state, None, object(), image=None, params={"path_config": runtime}
    )

    assert pipeline_worker.path_config == plan_worker.config


def test_current_mask_evidence_replaces_stale_runtime_masks(monkeypatch):
    """DECISION-013: runtime config cannot retain masks from an older route run."""
    import numpy as np
    import footprint_planner
    from pyside6_app.workers import planning_config_with_mask_evidence

    shape = (4, 5)
    current_headland = np.full(shape, 1, dtype=np.uint8)
    current_support = np.full(shape, 2, dtype=np.uint8)
    current_neutral = np.full(shape, 3, dtype=np.uint8)
    current_uncertain = np.full(shape, 4, dtype=np.uint8)
    current_forbidden = np.full(shape, 5, dtype=np.uint8)
    stale = np.full(shape, 99, dtype=np.uint8)
    mask_result = {
        "headland_mask": current_headland,
        "planning_support_mask": current_support,
        "neutral_support_mask": current_neutral,
        "uncertain_residual_mask": current_uncertain,
    }
    runtime_config = {
        "headland_mask": stale,
        "planning_support_mask": stale,
        "neutral_support_mask": stale,
        "uncertain_mask": stale,
        "forbidden_mask": stale,
        "forbidden_mask_confirmed": False,
    }
    state = SimpleNamespace(forbidden_regions_confirmed=True)
    monkeypatch.setattr(
        footprint_planner,
        "rasterize_forbidden_regions",
        lambda *_args, **_kwargs: current_forbidden,
    )

    config = planning_config_with_mask_evidence(
        runtime_config, mask_result, state, shape
    )

    assert config["headland_mask"] is current_headland
    assert config["planning_support_mask"] is current_support
    assert config["neutral_support_mask"] is current_neutral
    assert config["uncertain_mask"] is current_uncertain
    assert config["forbidden_mask"] is current_forbidden
    assert config["forbidden_mask_confirmed"] is True


def test_pipeline_worker_forwards_all_planning_mask_evidence(monkeypatch):
    """DECISION-009/013: the full pipeline must not drop semantic mask evidence."""
    import numpy as np
    import footprint_planner
    import inference
    import mask_processor
    import path_planner
    import pyside6_app.workers as workers

    shape = (4, 5)
    detected = np.full(shape, 1, dtype=np.uint8)
    processed_mask = np.full(shape, 2, dtype=np.uint8)
    planned = np.full(shape, 3, dtype=np.uint8)
    neutral = np.full(shape, 4, dtype=np.uint8)
    headland = np.full(shape, 5, dtype=np.uint8)
    uncertain = np.full(shape, 6, dtype=np.uint8)
    forbidden = np.full(shape, 7, dtype=np.uint8)
    support = np.full(shape, 8, dtype=np.uint8)
    mask_result = {
        "processed_mask": processed_mask,
        "main_angle": 12.0,
        "wide_bands": [object()],
        "neutral_support_mask": neutral,
        "headland_mask": headland,
        "uncertain_residual_mask": uncertain,
        "planning_support_mask": support,
    }
    captured = {}

    class FakeInference:
        def __init__(self, _engine):
            pass

        def run(self, *_args, **_kwargs):
            return detected

    def fake_plan_path(*args, **kwargs):
        captured.update(kwargs)
        return {"full_path": [], "segment_types": []}

    monkeypatch.setattr(inference, "TiledInference", FakeInference)
    monkeypatch.setattr(mask_processor, "process_mask", lambda *_a, **_k: mask_result)
    monkeypatch.setattr(mask_processor, "extract_centerlines", lambda *_a, **_k: [])
    monkeypatch.setattr(path_planner, "build_band_mask", lambda *_a, **_k: planned)
    monkeypatch.setattr(path_planner, "plan_path", fake_plan_path)
    monkeypatch.setattr(
        footprint_planner, "rasterize_forbidden_regions", lambda *_a, **_k: forbidden
    )
    monkeypatch.setattr(workers, "require_metric_scale", lambda *_a, **_k: 0.1)

    state = SimpleNamespace(
        inference_original_mask=detected,
        forbidden_regions_confirmed=True,
    )
    worker = workers.PipelineWorker(
        state,
        object(),
        object(),
        image=np.zeros((4, 5, 3), dtype=np.uint8),
        params={"path_config": {"turn_strategy": "dubins"}},
    )
    worker.run()

    assert captured["detected_mask"] is detected
    assert captured["planned_mask"] is planned
    assert captured["config"]["neutral_support_mask"] is neutral
    assert captured["config"]["headland_mask"] is headland
    assert captured["config"]["uncertain_mask"] is uncertain
    assert captured["config"]["forbidden_mask"] is forbidden
    assert captured["config"]["planning_support_mask"] is support


def test_pipeline_worker_does_not_promote_processed_mask_to_planning_evidence(monkeypatch):
    """DECISION-009/017: an empty four-row set must not plan the two-row mask."""
    import numpy as np
    import footprint_planner
    import inference
    import mask_processor
    import path_planner
    import pyside6_app.workers as workers

    shape = (4, 5)
    detected = np.full(shape, 1, dtype=np.uint8)
    processed_mask = np.full(shape, 255, dtype=np.uint8)
    empty_planning_mask = np.zeros(shape, dtype=np.uint8)
    mask_result = {
        "processed_mask": processed_mask,
        "main_angle": 0.0,
        "wide_bands": [],
        # This simulates a processed mask that still contains two-row bands.
        "all_bands": [{"id": 1, "band_type": "narrow"}],
    }
    captured = {}

    class FakeInference:
        def __init__(self, _engine):
            pass

        def run(self, *_args, **_kwargs):
            return detected

    def fake_plan_path(*_args, **kwargs):
        captured.update(kwargs)
        return {"full_path": [], "segment_types": []}

    monkeypatch.setattr(inference, "TiledInference", FakeInference)
    monkeypatch.setattr(mask_processor, "process_mask", lambda *_a, **_k: mask_result)
    monkeypatch.setattr(mask_processor, "extract_centerlines", lambda *_a, **_k: [])
    monkeypatch.setattr(
        path_planner,
        "build_band_mask",
        lambda shape, _bands: empty_planning_mask.copy(),
    )
    monkeypatch.setattr(path_planner, "plan_path", fake_plan_path)
    monkeypatch.setattr(
        footprint_planner,
        "rasterize_forbidden_regions",
        lambda *_a, **_k: np.zeros(shape, dtype=np.uint8),
    )
    monkeypatch.setattr(workers, "require_metric_scale", lambda *_a, **_k: 0.1)

    worker = workers.PipelineWorker(
        SimpleNamespace(
            inference_original_mask=detected,
            forbidden_regions_confirmed=False,
        ),
        object(),
        object(),
        image=np.zeros((4, 5, 3), dtype=np.uint8),
        params={"path_config": {"turn_strategy": "dubins"}},
    )
    worker.run()

    assert np.count_nonzero(captured["planned_mask"]) == 0


def test_plan_worker_does_not_promote_processed_mask_to_planning_evidence(monkeypatch):
    """The direct re-plan entry point must obey the same four-row contract."""
    import numpy as np
    import footprint_planner
    import path_planner
    import pyside6_app.workers as workers

    shape = (4, 5)
    processed_mask = np.full(shape, 255, dtype=np.uint8)
    mask_result = {
        "processed_mask": processed_mask,
        "main_angle": 0.0,
        "wide_bands": [],
        "all_bands": [{"id": 1, "band_type": "narrow"}],
        "provenance": {"fingerprint": "mask-fingerprint"},
    }
    captured = {}

    monkeypatch.setattr(
        path_planner,
        "build_band_mask",
        lambda shape, _bands: np.zeros(shape, dtype=np.uint8),
    )

    def fake_plan_path(*_args, **kwargs):
        captured.update(kwargs)
        return {"full_path": [], "segment_types": []}

    monkeypatch.setattr(path_planner, "plan_path", fake_plan_path)
    monkeypatch.setattr(
        footprint_planner,
        "rasterize_forbidden_regions",
        lambda *_a, **_k: np.zeros(shape, dtype=np.uint8),
    )
    monkeypatch.setattr(workers, "require_metric_scale", lambda *_a, **_k: 0.1)

    worker = workers.PlanWorker(
        mask_result,
        object(),
        SimpleNamespace(forbidden_regions_confirmed=False),
        {"turn_strategy": "dubins"},
    )
    worker.run()

    assert np.count_nonzero(captured["planned_mask"]) == 0


def test_pipeline_worker_completed_path_contains_verifiable_provenance(monkeypatch):
    """A completed background pipeline must already satisfy the export path evidence gate."""
    import numpy as np
    import footprint_planner
    import inference
    import mask_processor
    import path_planner
    import pyside6_app.workers as workers
    from provenance import artifact_sha256, verify_stage_record

    raw_mask = np.ones((4, 5), dtype=np.uint8)
    processed_mask = np.ones((4, 5), dtype=np.uint8) * 255
    headland_mask = np.zeros((4, 5), dtype=np.uint8)
    planning_support_mask = processed_mask.copy()
    neutral_support_mask = np.zeros((4, 5), dtype=np.uint8)
    uncertain_residual_mask = np.zeros((4, 5), dtype=np.uint8)
    forbidden_mask = np.zeros((4, 5), dtype=np.uint8)
    mask_result = {
        "processed_mask": processed_mask,
        "headland_mask": headland_mask,
        "planning_support_mask": planning_support_mask,
        "neutral_support_mask": neutral_support_mask,
        "uncertain_residual_mask": uncertain_residual_mask,
        "main_angle": 0.0,
        "wide_bands": [],
        "provenance": {"fingerprint": "mask-fingerprint"},
    }

    class FakeInference:
        def __init__(self, _engine):
            pass

        def run(self, *_args, **_kwargs):
            return raw_mask

    monkeypatch.setattr(inference, "TiledInference", FakeInference)
    monkeypatch.setattr(mask_processor, "process_mask", lambda *_a, **_k: mask_result)
    monkeypatch.setattr(mask_processor, "extract_centerlines", lambda *_a, **_k: [])
    monkeypatch.setattr(path_planner, "build_band_mask", lambda *_a, **_k: processed_mask)
    monkeypatch.setattr(
        path_planner,
        "plan_path",
        lambda *_a, **_k: {
            "full_path": [[(0.0, 0.0), (1.0, 1.0)]],
            "segment_types": ["work"],
            "layout": {
                "work_line_mode": "footprint_optimized",
                "formal_comparison": {"accepted": True},
            },
        },
    )
    monkeypatch.setattr(
        footprint_planner,
        "rasterize_forbidden_regions",
        lambda *_args, **_kwargs: forbidden_mask,
    )
    monkeypatch.setattr(workers, "require_metric_scale", lambda *_a, **_k: 0.1)

    state = SimpleNamespace(
        harvester_params={"turn_radius_m": 1.2},
        turn_strategy="dubins",
        entry_point=(0.0, 0.0),
        exit_point=(1.0, 1.0),
        unload_points=[],
        forbidden_regions=[],
        forbidden_regions_confirmed=False,
    )
    worker = workers.PipelineWorker(
        state,
        object(),
        object(),
        image=np.zeros((4, 5, 3), dtype=np.uint8),
        params={"path_config": {"min_turn_radius_m": 1.2, "turn_strategy": "dubins"}},
    )
    completed = []
    worker.finished.connect(completed.append)
    worker.run()

    path_result = completed[0]["path"]
    record = path_result["provenance"]
    artifact = {
        "full_path": path_result["full_path"],
        "segment_types": path_result["segment_types"],
        "layout": path_result["layout"],
    }
    assert record["inputs"]["mask_fingerprint"] == "mask-fingerprint"
    assert record["inputs"]["path_config"]["min_turn_radius_m"] == 1.2
    assert record["inputs"]["semantic_mask_fingerprints"] == {
        "headland_mask": artifact_sha256(headland_mask),
        "planning_support_mask": artifact_sha256(planning_support_mask),
        "neutral_support_mask": artifact_sha256(neutral_support_mask),
        "uncertain_mask": artifact_sha256(uncertain_residual_mask),
        "forbidden_mask": artifact_sha256(forbidden_mask),
    }
    verify_stage_record(record, "path", record["inputs"], artifact)

    tampered_inputs = dict(record["inputs"])
    tampered_inputs["semantic_mask_fingerprints"] = dict(
        record["inputs"]["semantic_mask_fingerprints"]
    )
    tampered_inputs["semantic_mask_fingerprints"]["planning_support_mask"] = (
        artifact_sha256(np.zeros((4, 5), dtype=np.uint8))
    )
    with pytest.raises(Exception):
        verify_stage_record(record, "path", tampered_inputs, artifact)

    tampered = dict(artifact)
    tampered["layout"] = {
        **artifact["layout"],
        "formal_comparison": {"accepted": False},
    }
    with pytest.raises(Exception):
        verify_stage_record(record, "path", record["inputs"], tampered)
