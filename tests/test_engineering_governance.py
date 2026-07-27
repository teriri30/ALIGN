import json
from pathlib import Path

import numpy as np


def test_project_session_reset_preserves_machine_and_model_but_clears_project_state():
    from pyside6_app.project_session import generation_matches, reset_project_state
    from state import AppState

    state = AppState()
    state.project_generation = 3
    state.harvester_params = {"track_width_m": 0.4}
    state.turn_strategy = "pear"
    state.current_model_name = "model.pt"
    state.field_boundary = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    state.inference_done = True

    generation = reset_project_state(state)

    assert generation == 4
    assert generation_matches(state, 4)
    assert not generation_matches(state, 3)
    assert state.harvester_params == {"track_width_m": 0.4}
    assert state.turn_strategy == "pear"
    assert state.current_model_name == "model.pt"
    assert state.field_boundary == []
    assert state.inference_done is False


def test_shared_row_orientation_is_used_without_planning_import(monkeypatch):
    import mask_processor

    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[35:45, 5:75] = 255
    monkeypatch.setattr(
        mask_processor,
        "estimate_row_orientation",
        lambda _mask, _image: 0.25,
    )

    assert mask_processor.detect_main_direction(mask) < 0.1
    assert mask_processor.detect_main_direction_refined(mask, mask) == 0.25


def test_dependency_files_share_one_core_version_source():
    root = Path(__file__).resolve().parents[1]
    core = (root / "requirements-core.txt").read_text(encoding="utf-8").splitlines()
    runtime = (root / "requirements.txt").read_text(encoding="utf-8")
    ci = (root / "requirements-ci.txt").read_text(encoding="utf-8")

    assert "-r requirements-core.txt" in runtime
    assert "-r requirements-core.txt" in ci
    for requirement in core:
        requirement = requirement.strip()
        if requirement:
            assert requirement not in runtime
            assert requirement not in ci


def test_model_registry_marks_unknown_training_metadata_explicitly():
    registry_path = Path(__file__).resolve().parents[1] / "data" / "models" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert registry["schema"] == 2
    for model in registry["models"]:
        assert model["task"] == "instance_segmentation"
        assert model["framework"] == "ultralytics"
        assert model["metadata_status"] in {"complete", "incomplete"}
        if model["metadata_status"] == "incomplete":
            assert model["compatibility"]["requires_manual_confirmation"] is True
            assert model["training_dataset_version"] is None
            assert model["class_mapping"] is None
            assert model["training_preprocessing"] is None
