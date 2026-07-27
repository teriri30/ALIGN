from types import SimpleNamespace


def test_task_token_rejects_project_switch_and_cancellation():
    from task_lifecycle import make_task_token, result_is_admissible

    state = SimpleNamespace(project_generation=3, source_sha256="source-a", model_sha256="model-a")
    token = make_task_token(state, "inference")
    assert result_is_admissible(token, state)

    state.project_generation = 4
    assert not result_is_admissible(token, state)

    state.project_generation = 3
    token.invalidate()
    assert not result_is_admissible(token, state)


def test_task_token_rejects_changed_source_or_model():
    from task_lifecycle import make_task_token, result_is_admissible

    state = SimpleNamespace(project_generation=1, source_sha256="source-a", model_sha256="model-a")
    token = make_task_token(state, "inference")
    state.source_sha256 = "source-b"
    assert not result_is_admissible(token, state)
    state.source_sha256 = "source-a"
    state.model_sha256 = "model-b"
    assert not result_is_admissible(token, state)


def test_stale_inference_runner_does_not_mutate_state_or_start_source_tif_work(monkeypatch):
    """DECISION-002: identity drift rejects a runner before it clears new-project state."""
    import numpy as np
    import inference
    from inference import InferenceRunner
    from state import AppState
    from task_lifecycle import make_task_token

    state = AppState()
    retained_mask = np.ones((2, 2), dtype=np.uint8)
    state.safe_update(
        project_generation=1,
        source_sha256="source-a",
        model_sha256="model-a",
        tif_path="project-b.tif",
        mask_raw=retained_mask,
        inference_done=True,
    )
    token = make_task_token(state, "inference")
    state.safe_update(source_sha256="source-b", model_sha256="model-b")
    started = []

    class RecordingThread:
        def __init__(self, *args, **kwargs):
            started.append((args, kwargs))

        def start(self):
            raise AssertionError("a stale inference runner must not start a worker")

    monkeypatch.setattr(inference.threading, "Thread", RecordingThread)

    InferenceRunner(state, object(), object(), token=token).start_from_tif(
        "project-a.tif",
        (10, 10),
        [(0, 0), (9, 0), (9, 9)],
        1,
    )

    assert started == []
    assert state.mask_raw is retained_mask
    assert state.inference_done is True
    assert state.inference_running is False


def test_path_cache_with_only_flat_geo_points_is_not_treated_as_editable_path():
    from pyside6_app.main_window import MainWindow

    window = MainWindow.__new__(MainWindow)
    saved = {
        "auto_path": [],
        "auto_path_geo": [{"lon": 1.0, "lat": 2.0}],
        "path_points": [[1.0, 2.0]],
    }
    assert window._restore_cached_path_geometry(saved) == {}


def test_cache_path_restore_plan_clears_derived_fields_without_canonical_segments():
    from pyside6_app.project_session import cached_path_restore_plan

    plan = cached_path_restore_plan(
        {
            'auto_path': [],
            'auto_path_geo': [{'lon': 1.0, 'lat': 2.0}],
            'path_points': [[1.0, 2.0]],
            'path_status': [1],
        },
        path_cache_valid=True,
    )

    assert plan['restore_editable_path'] is False
    assert plan['historical_path_present'] is True
    assert plan['state_updates']['auto_path'] == []
    assert plan['state_updates']['auto_path_geo'] == []
    assert plan['state_updates']['path_points'] == []
    assert plan['state_updates']['auto_path_valid'] is False


def test_turn_candidate_reports_requested_and_effective_strategy():
    from path_planner import _select_turn_strategy

    decision = _select_turn_strategy("bow", 1.0, 1.0, False)
    assert decision["requested_strategy"] == "bow"
    assert decision["candidate_strategy"] == "bow"
    assert decision["effective_strategy"] == "pear"
    assert decision["strategy"] == "pear"
    assert decision["fallback_from"] == "bow"
    assert decision["fallback_reason"]
    assert decision["geometry_signature"] == _select_turn_strategy("bow", 1.0, 1.0, False)["geometry_signature"]


def test_in_memory_inference_records_the_supplied_image_fingerprint(monkeypatch):
    """The non-GeoTIFF runner must hash its image argument, not an undefined name."""
    import numpy as np
    from inference import InferenceRunner
    from state import AppState

    state = AppState()
    state.safe_update(
        project_generation=1,
        source_sha256="source-a",
        model_sha256="",
        inference_running=True,
        tif_path="D:/fake/result.tif",
        field_boundary=[(0, 0), (9, 0), (9, 9)],
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    runner = InferenceRunner(state, object(), object())

    class FakeTiledInference:
        def __init__(self, engine):
            pass

        def run(self, *_args, **_kwargs):
            return np.ones((10, 10), dtype=np.uint8) * 255

    monkeypatch.setattr("inference.TiledInference", FakeTiledInference)
    monkeypatch.setattr("cache.save_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr("cache.save_project_state", lambda *args, **kwargs: None)

    runner._run(image, state.field_boundary)

    assert state.inference_done is True
    assert state.inference_provenance["input_fingerprint"]


def test_inference_runner_stops_cache_commits_when_its_token_is_invalidated(monkeypatch):
    import numpy as np
    from inference import InferenceRunner
    from state import AppState
    from task_lifecycle import make_task_token

    state = AppState()
    state.safe_update(
        project_generation=1,
        source_sha256="source-a",
        model_sha256="model-a",
        current_model_path="",
        tif_path="project-a.tif",
        field_boundary=[(0, 0), (3, 0), (3, 3)],
        inference_running=True,
    )
    token = make_task_token(state, "inference")
    runner = InferenceRunner(state, object(), object(), token=token)
    writes = []

    class FakeTiledInference:
        def __init__(self, engine):
            pass

        def run(self, *_args, **_kwargs):
            return np.ones((4, 4), dtype=np.uint8) * 255

    def invalidate_after_first_write(path, _mask, _x, _y, suffix="", **_kwargs):
        writes.append((path, suffix))
        token.invalidate()
        state.tif_path = "project-b.tif"
        return True

    monkeypatch.setattr("inference.TiledInference", FakeTiledInference)
    monkeypatch.setattr("cache.save_mask", invalidate_after_first_write)
    monkeypatch.setattr("cache.save_project_state", lambda *args, **kwargs: writes.append((args[0], "state")))

    runner._run(np.zeros((4, 4, 3), dtype=np.uint8), state.field_boundary)

    assert writes == [("project-a.tif", "")]
    assert state.inference_done is False


def test_inference_runner_rejects_source_or_model_identity_drift_before_cache_commit(monkeypatch):
    """DECISION-002: late inference cannot commit after source/model ownership changes."""
    import numpy as np
    from inference import InferenceRunner
    from state import AppState
    from task_lifecycle import make_task_token

    class FakeTiledInference:
        def __init__(self, engine):
            self.engine = engine

        def run(self, *_args, **_kwargs):
            return np.ones((4, 4), dtype=np.uint8) * 255

    monkeypatch.setattr("inference.TiledInference", FakeTiledInference)

    for changed_field in ("source_sha256", "model_sha256"):
        state = AppState()
        retained_mask = np.full((2, 2), 7, dtype=np.uint8)
        state.safe_update(
            project_generation=1,
            source_sha256="source-a",
            model_sha256="model-a",
            tif_path="project-a.tif",
            field_boundary=[(0, 0), (3, 0), (3, 3)],
            inference_running=True,
        )
        token = make_task_token(state, "inference")
        writes = []

        def drift_after_inference(*_args, **_kwargs):
            state.safe_update(
                **{
                    changed_field: "changed",
                    "mask_raw": retained_mask,
                    "inference_done": True,
                    "inference_running": False,
                }
            )
            return np.ones((4, 4), dtype=np.uint8) * 255

        monkeypatch.setattr("inference.TiledInference", lambda _engine: type(
            "DriftingInference", (), {"run": staticmethod(drift_after_inference)}
        )())
        monkeypatch.setattr(
            "cache.save_mask",
            lambda *args, **kwargs: writes.append((args, kwargs)),
        )
        monkeypatch.setattr(
            "cache.save_project_state",
            lambda *args, **kwargs: writes.append((args, kwargs)),
        )

        InferenceRunner(state, object(), object(), token=token)._run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            state.field_boundary,
        )

        assert writes == []
        assert state.mask_raw is retained_mask
        assert state.inference_done is True
        assert state.inference_running is False


def test_save_mask_does_not_write_meta_after_commit_callback_is_invalidated(tmp_path, monkeypatch):
    import json
    import numpy as np
    import cache

    tif_path = str(tmp_path / "project.tif")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cache, "_cache_dir", lambda _path: str(cache_dir))
    callback_calls = []

    def commit_callback():
        callback_calls.append(True)
        return len(callback_calls) == 1

    assert cache.save_mask(
        tif_path,
        np.ones((2, 2), dtype=np.uint8),
        meta={"model": "cancelled.pt"},
        commit_callback=commit_callback,
    ) is False

    meta_path = cache_dir / "meta.json"
    assert not meta_path.exists()
    assert callback_calls == [True, True]


def test_cancelled_plan_result_is_not_committed():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    window = MainWindow.__new__(MainWindow)
    state = SimpleNamespace(
        project_generation=2,
        source_sha256="source-a",
        model_sha256="model-a",
        plan_running=True,
    )
    window.state = state
    window._pipeline_result = {"path": {"sentinel": "current"}}
    window._active_plan_token = make_task_token(state, "plan")
    window._active_plan_token.invalidate()

    assert window._admit_task_completion("plan", window._active_plan_token) is False
    assert window._pipeline_result == {"path": {"sentinel": "current"}}


def test_cancelling_background_work_invalidates_active_inference_runner():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    state = SimpleNamespace(
        project_generation=2,
        source_sha256='source-a',
        model_sha256='model-a',
        inference_running=True,
        mask_processing_running=False,
        plan_running=False,
    )
    token = make_task_token(state, 'inference')
    cancelled = []
    window = MainWindow.__new__(MainWindow)
    window.state = state
    window._active_inference_token = token
    window._active_inference_runner = SimpleNamespace(cancel=lambda: cancelled.append(True))
    window._active_mask_token = None
    window._active_plan_token = None
    window._worker = None
    quit_calls = []
    window._worker_thread = SimpleNamespace(quit=lambda: quit_calls.append(True))
    window._has_background_worker = lambda: True
    window.lbl_status = SimpleNamespace(setText=lambda _text: None)
    window.log_panel = SimpleNamespace(info=lambda _text: None)
    window.btn_cancel_task = SimpleNamespace(setEnabled=lambda _enabled: None)

    window._cancel_background_worker()

    assert token.invalidated is True
    assert cancelled == [True]
    assert quit_calls == []


def test_cancelling_paper_experiment_invalidates_its_completion_token():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    state = SimpleNamespace(
        project_generation=2,
        source_sha256="source-a",
        model_sha256="model-a",
        inference_running=False,
        mask_processing_running=False,
        plan_running=False,
    )
    token = make_task_token(state, "paper_experiment")
    cancelled = []
    window = MainWindow.__new__(MainWindow)
    window.state = state
    window._active_inference_token = None
    window._active_inference_runner = None
    window._active_mask_token = None
    window._active_plan_token = None
    window._active_cache_restore_token = None
    window._active_paper_experiment_token = token
    window._worker = SimpleNamespace(cancel=lambda: cancelled.append(True))
    window._has_background_worker = lambda: True
    window.lbl_status = SimpleNamespace(setText=lambda _text: None)
    window.log_panel = SimpleNamespace(info=lambda _text: None)
    window.btn_cancel_task = SimpleNamespace(setEnabled=lambda _enabled: None)

    window._cancel_background_worker()

    assert token.invalidated is True
    assert cancelled == [True]


def test_project_or_model_switch_invalidates_the_active_inference_runner():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    state = SimpleNamespace(project_generation=2, source_sha256='source-a', model_sha256='model-a')
    token = make_task_token(state, 'inference')
    cancelled = []
    window = MainWindow.__new__(MainWindow)
    window._active_inference_token = token
    window._active_inference_runner = SimpleNamespace(cancel=lambda: cancelled.append(True))
    window._pending_segment_display = True

    window._invalidate_active_inference_runner()

    assert token.invalidated is True
    assert cancelled == [True]
    assert window._active_inference_token is None
    assert window._active_inference_runner is None
    assert window._pending_segment_display is False


def test_declining_external_model_trust_does_not_cancel_active_inference(monkeypatch, tmp_path):
    from pyside6_app.main_window import MainWindow, QMessageBox
    from task_lifecycle import make_task_token

    model_path = tmp_path / "external.pt"
    model_path.write_bytes(b"external")
    state = SimpleNamespace(project_generation=2, source_sha256="source-a", model_sha256="model-a")
    token = make_task_token(state, "inference")
    cancelled = []
    window = MainWindow.__new__(MainWindow)
    window.state = state
    window._active_inference_token = token
    window._active_inference_runner = SimpleNamespace(cancel=lambda: cancelled.append(True))
    window._pending_segment_display = True
    window._builtin_model_dir = lambda: str(tmp_path / "builtin")
    window.log_panel = SimpleNamespace(info=lambda _message: None)
    monkeypatch.setattr(
        "pyside6_app.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.No,
    )

    assert window._on_load_model_at(str(model_path)) is False
    assert token.invalidated is False
    assert cancelled == []
    assert window._active_inference_runner is not None


def test_finished_owned_inference_runner_is_released_after_natural_completion():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    state = SimpleNamespace(project_generation=2, source_sha256="source-a", model_sha256="model-a")
    token = make_task_token(state, "inference")
    runner = SimpleNamespace(is_running=lambda: False)
    window = MainWindow.__new__(MainWindow)
    window.state = state
    window._active_inference_token = token
    window._active_inference_runner = runner

    assert window._release_finished_inference_runner() is True
    assert window._active_inference_token is None
    assert window._active_inference_runner is None


def test_cancelled_cache_restore_emits_a_token_bound_cancellation_result():
    from pyside6_app.workers import CacheRestoreWorker
    from task_lifecycle import make_task_token

    state = SimpleNamespace(project_generation=2, source_sha256='source-a', model_sha256='model-a')
    token = make_task_token(state, 'cache_restore')
    token.invalidate()
    worker = CacheRestoreWorker('unused.tif', token=token)
    received = []
    worker.finished.connect(received.append)

    worker.run()

    assert len(received) == 1
    assert received[0]['status'] == 'cancelled'
    assert received[0]['task_token'] is token
    assert received[0]['result'] is None


def test_cancelled_mask_and_plan_workers_emit_token_bound_cancellation_results():
    import numpy as np
    from pyside6_app.workers import MaskProcessWorker, PlanWorker
    from task_lifecycle import make_task_token

    state = SimpleNamespace(project_generation=2, source_sha256='source-a', model_sha256='model-a')
    for task_type, worker in (
        (
            'mask',
            MaskProcessWorker(
                np.zeros((2, 2), dtype=np.uint8), object(), state, {},
                token=make_task_token(state, 'mask'),
            ),
        ),
        (
            'plan',
            PlanWorker(
                {}, object(), state, {}, token=make_task_token(state, 'plan'),
            ),
        ),
    ):
        worker.token.invalidate()
        received = []
        worker.finished.connect(received.append)

        worker.run()

        assert len(received) == 1
        assert received[0]['status'] == 'cancelled'
        assert received[0]['task_token'] is worker.token
        assert received[0]['result'] is None
        assert received[0]['task_token'].task_type == task_type


def test_cache_restore_completion_rejects_a_result_after_project_switch():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    state = SimpleNamespace(project_generation=2, source_sha256='source-a', model_sha256='model-a')
    token = make_task_token(state, 'cache_restore')
    window = MainWindow.__new__(MainWindow)
    window.state = state
    window._active_cache_restore_token = token
    window._worker_thread = None
    applied = []
    window._restore_cache_payload = lambda payload: applied.append(payload)

    state.project_generation = 3
    window._on_cache_restore_done({
        'task_token': token,
        'status': 'completed',
        'result': {'saved': {'auto_path_planned': True}},
    })

    assert applied == []


def test_cache_restore_registers_token_before_starting_its_worker():
    from pyside6_app.main_window import MainWindow

    window = MainWindow.__new__(MainWindow)
    window.state = SimpleNamespace(project_generation=2, source_sha256="source-a", model_sha256="model-a")
    observed = []
    window._start_background_worker = lambda *_args, **_kwargs: (
        observed.append(getattr(window, "_active_cache_restore_token", None)) or True
    )

    window._start_cache_restore("current.tif")

    assert observed == [window._active_cache_restore_token]


def test_failed_cache_restore_reports_error_only_for_the_owned_token():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    state = SimpleNamespace(project_generation=2, source_sha256="source-a", model_sha256="model-a")
    token = make_task_token(state, "cache_restore")
    errors = []
    statuses = []
    window = MainWindow.__new__(MainWindow)
    window.state = state
    window._active_cache_restore_token = token
    window.log_panel = SimpleNamespace(error=errors.append)
    window.lbl_status = SimpleNamespace(setText=statuses.append)

    window._on_cache_restore_done({
        "task_token": token,
        "status": "failed",
        "result": None,
        "error": "cache exploded",
    })

    assert window._active_cache_restore_token is None
    assert errors and errors[0].endswith(": cache exploded")
    assert statuses and statuses[0]


def test_segment_starts_inference_with_window_owned_task_token(monkeypatch):
    import numpy as np
    from pyside6_app.main_window import MainWindow

    captured = []

    class FakeRunner:
        def __init__(self, state, engine, geo, token=None):
            captured.append(token)

        def start_from_tif(self, *args):
            captured.append(args)

    monkeypatch.setattr('inference.InferenceRunner', FakeRunner)
    window = MainWindow.__new__(MainWindow)
    window.state = SimpleNamespace(
        project_generation=5,
        source_sha256='source-a',
        model_sha256='model-a',
        current_model_path='',
        inference_running=False,
        field_boundary=[(0, 0), (1, 0), (1, 1)],
        downsample_factor=1,
    )
    window._tif_rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    window._tif_path = 'image.tif'
    window.model_engine = SimpleNamespace(is_loaded=lambda: True)
    window.geo = object()
    window._save_system_undo = lambda _label: None
    window._invalidate_analysis_from = lambda *_args: None
    window.top_toolbar = SimpleNamespace(set_progress=lambda *_args: None)
    window.progress_bar = SimpleNamespace(show=lambda: None, setValue=lambda _value: None)
    window.task_panel = SimpleNamespace(set_task_status=lambda *_args: None)
    window.lbl_status = SimpleNamespace(setText=lambda _text: None)

    window._on_segment()

    token = captured[0]
    assert token.task_type == 'inference'
    assert token.project_generation == 5
    assert token.source_sha256 == 'source-a'
    assert token.model_sha256 == 'model-a'
    assert window._active_inference_token is token
    assert window._active_inference_runner is not None


def test_visual_restore_rebuilds_route_from_undo_owned_state(monkeypatch):
    """Undo snapshots own the route; a newer pipeline cache must not override them."""
    from pyside6_app.main_window import MainWindow
    from state import AppState, AutoPathSegment, PathPoint

    window = MainWindow.__new__(MainWindow)
    window.state = AppState()
    window.state.safe_update(
        auto_path_planned=True,
        auto_path=[
            AutoPathSegment(
                points=[PathPoint(0.0, 0.0), PathPoint(5.0, 0.0)],
                segment_type="work",
                length_m=5.0,
            )
        ],
        path_validation={"valid": False, "issues": ["undo route"]},
    )
    window._pipeline_result = {
        "path": {
            "full_path": [[(100.0, 100.0), (110.0, 100.0)]],
            "segment_types": ["work"],
            "validation": {"valid": True},
        }
    }
    shown = []
    window.top_toolbar = SimpleNamespace(refresh_actions=lambda _state: None)
    monkeypatch.setattr(window, "_show_path", lambda path: shown.append(path))

    window._restore_visual_from_state()

    assert shown[0]["full_path"] == [[(0.0, 0.0), (5.0, 0.0)]]
    assert shown[0]["validation"]["issues"] == ["undo route"]


def test_state_path_projection_never_borrows_validation_from_pipeline_cache():
    from pyside6_app.main_window import MainWindow
    from state import AppState, AutoPathSegment, PathPoint

    window = MainWindow.__new__(MainWindow)
    window.state = AppState()
    window.state.safe_update(
        auto_path_planned=True,
        auto_path=[
            AutoPathSegment(
                points=[PathPoint(0.0, 0.0), PathPoint(5.0, 0.0)],
                segment_type="work",
                length_m=5.0,
            )
        ],
        path_validation={},
    )
    window._pipeline_result = {"path": {"validation": {"valid": True}}}

    path_result = window._path_result_from_state()

    assert path_result["validation"] == {}
    assert path_result["is_valid"] is False


def test_cache_model_identity_is_historical_not_active():
    from pyside6_app.main_window import MainWindow

    window = MainWindow.__new__(MainWindow)
    restored = window._cache_model_identity_updates(
        {
            "current_model_name": "historical.pt",
            "current_model_path": "D:/models/historical.pt",
            "model_sha256": "historical-hash",
        }
    )

    assert restored == {
        "historical_model_name": "historical.pt",
        "historical_model_path": "D:/models/historical.pt",
        "historical_model_sha256": "historical-hash",
    }


def test_cache_model_identity_requires_the_loaded_engine_weight_to_match(tmp_path):
    from pyside6_app.main_window import MainWindow
    from provenance import file_sha256

    historical_path = tmp_path / "historical.pt"
    active_path = tmp_path / "active.pt"
    historical_path.write_bytes(b"historical-weight")
    active_path.write_bytes(b"active-weight")

    state = SimpleNamespace(
        historical_model_path=str(historical_path),
        historical_model_sha256=file_sha256(historical_path),
        current_model_path=str(active_path),
        model_sha256=file_sha256(active_path),
    )
    engine = SimpleNamespace(_model_path=str(active_path), is_loaded=lambda: True)

    assert MainWindow._cache_model_matches_active_engine(state, engine) is False

    state.current_model_path = str(historical_path)
    state.model_sha256 = file_sha256(historical_path)
    engine._model_path = str(historical_path)

    assert MainWindow._cache_model_matches_active_engine(state, engine) is True


def test_segment_blocks_inference_when_restored_cache_model_differs_from_active_engine(tmp_path, monkeypatch):
    """DECISION-002/006: historical cache provenance cannot authorize new inference."""
    import numpy as np
    from pyside6_app.main_window import MainWindow, QMessageBox
    from provenance import file_sha256

    historical_path = tmp_path / "historical.pt"
    active_path = tmp_path / "active.pt"
    historical_path.write_bytes(b"historical-weight")
    active_path.write_bytes(b"active-weight")
    warnings = []
    window = MainWindow.__new__(MainWindow)
    window._tif_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    window.state = SimpleNamespace(
        historical_model_path=str(historical_path),
        historical_model_sha256=file_sha256(historical_path),
        current_model_path=str(active_path),
        model_sha256=file_sha256(active_path),
    )
    window.model_engine = SimpleNamespace(
        _model_path=str(active_path),
        is_loaded=lambda: True,
    )
    monkeypatch.setattr(
        "pyside6_app.main_window.QMessageBox.warning",
        lambda *_args: warnings.append(_args),
    )

    window._on_segment()

    assert window.state.cache_model_matches_active_engine is False
    assert len(warnings) == 1


def test_failed_mask_completion_clears_only_the_matching_running_state():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    state = SimpleNamespace(
        project_generation=1,
        source_sha256="source-a",
        model_sha256="model-a",
        mask_processing_running=True,
        mask_processing_progress=0.6,
    )
    token = make_task_token(state, "mask")
    statuses = []
    errors = []
    window = MainWindow.__new__(MainWindow)
    window.state = state
    window._active_mask_token = token
    window._task_cancel_requested = False
    window.progress_bar = SimpleNamespace(hide=lambda: statuses.append("hide-progress"))
    window.btn_cancel_task = SimpleNamespace(hide=lambda: statuses.append("hide-cancel"))
    window.top_toolbar = SimpleNamespace(
        set_progress=lambda value: statuses.append(("progress", value)),
        set_worker_cancel_visible=lambda value: statuses.append(("cancel-visible", value)),
    )
    window.task_panel = SimpleNamespace(
        set_task_status=lambda *args, **kwargs: statuses.append(("task", args, kwargs))
    )
    window.lbl_status = SimpleNamespace(setText=lambda text: statuses.append(("status", text)))
    window.log_panel = SimpleNamespace(error=errors.append)
    window._apply_mask_process_result = lambda _result: errors.append("applied")

    window._on_mask_worker_done({
        "task_token": token,
        "status": "failed",
        "result": {"processed_mask": "must-not-commit"},
        "error": "mask exploded",
    })

    assert state.mask_processing_running is False
    assert state.mask_processing_progress == 0.0
    assert window._active_mask_token is None
    assert "applied" not in errors
    assert any(item[0] == "task" and item[1][0] == "process" and item[1][1] == "failed" for item in statuses)
    assert errors and errors[0].endswith(": mask exploded")


def test_failed_plan_completion_clears_only_the_matching_running_state():
    from pyside6_app.main_window import MainWindow
    from task_lifecycle import make_task_token

    state = SimpleNamespace(
        project_generation=1,
        source_sha256="source-a",
        model_sha256="model-a",
        plan_running=True,
        plan_progress=0.8,
    )
    token = make_task_token(state, "plan")
    statuses = []
    errors = []
    window = MainWindow.__new__(MainWindow)
    window.state = state
    window._active_plan_token = token
    window._task_cancel_requested = False
    window.progress_bar = SimpleNamespace(hide=lambda: statuses.append("hide-progress"))
    window.btn_cancel_task = SimpleNamespace(hide=lambda: statuses.append("hide-cancel"))
    window.top_toolbar = SimpleNamespace(
        set_progress=lambda value: statuses.append(("progress", value)),
        set_worker_cancel_visible=lambda value: statuses.append(("cancel-visible", value)),
    )
    window.task_panel = SimpleNamespace(
        set_task_status=lambda *args, **kwargs: statuses.append(("task", args, kwargs))
    )
    window.lbl_status = SimpleNamespace(setText=lambda text: statuses.append(("status", text)))
    window.log_panel = SimpleNamespace(error=errors.append)
    window._store_path_result = lambda *_args: errors.append("applied")

    window._on_plan_worker_done({
        "task_token": token,
        "status": "failed",
        "result": {"path": {"must-not-commit": True}},
        "error": "plan exploded",
    })

    assert state.plan_running is False
    assert state.plan_progress == 0.0
    assert window._active_plan_token is None
    assert "applied" not in errors
    assert any(item[0] == "task" and item[1][0] == "plan" and item[1][1] == "failed" for item in statuses)
    assert errors and errors[0].endswith(": plan exploded")

