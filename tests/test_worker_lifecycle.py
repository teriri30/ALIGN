import os
import hashlib
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ALIGN_DISABLE_AUTO_START", "1")

from PySide6.QtCore import QObject, QEventLoop, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox
from shiboken6 import isValid

from pyside6_app.main_window import MainWindow
from pyside6_app.workers import TifLoadWorker
from pyside6_app.workers import ModelLoadWorker


class _LifecycleWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int, str)

    @Slot()
    def run(self):
        self.finished.emit({"ok": True})


class _ThreadProbeWindow(MainWindow):
    plan_result_ready = Signal(dict)

    def __init__(self):
        super().__init__()
        self.completion_thread = None
        self.plan_result_ready.connect(self._on_plan_worker_done, Qt.QueuedConnection)

    @Slot(dict)
    def _on_plan_worker_done(self, result):
        self.completion_thread = QThread.currentThread()


class _RunningThreadProbe:
    def isRunning(self):
        return True


class _CancellableProbe:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def test_cancelled_tif_loader_emits_terminal_result(tmp_path):
    """DECISION-004: cancelling preview load must still terminate its QThread."""
    import numpy as np
    import rasterio
    from affine import Affine

    tif_path = tmp_path / "cancel-before-read.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=1,
        dtype="uint8",
        crs="EPSG:32650",
        transform=Affine(0.2, 0.0, 500000.0, 0.0, -0.2, 3400000.0),
    ) as dataset:
        dataset.write(np.ones((1, 3, 3), dtype=np.uint8))

    results = []
    worker = TifLoadWorker(str(tif_path))
    worker.finished.connect(results.append)
    worker.cancel()
    worker.run()

    assert results == [{"status": "cancelled", "path": str(tif_path)}]


def test_close_keeps_window_visible_while_worker_cancels():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    worker = _CancellableProbe()
    window._worker_thread = _RunningThreadProbe()
    window._worker = worker
    window.show()
    app.processEvents()
    assert window.isVisible()

    assert window.close() is False
    app.processEvents()

    assert worker.cancelled is True
    assert window._close_after_worker_cancel is True
    assert window.isVisible() is True

    window._worker_thread = None
    window._worker = None
    window._close_after_worker_cancel = False
    window.close()


def test_plan_completion_slot_runs_on_gui_thread_when_worker_emits_from_qthread():
    app = QApplication.instance() or QApplication([])
    window = _ThreadProbeWindow()
    worker = _LifecycleWorker()
    worker.finished.connect(window.plan_result_ready.emit)
    assert window._start_background_worker(worker, "test", warn_if_busy=False)

    loop = QEventLoop()
    window._worker_thread.finished.connect(loop.quit)
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    app.processEvents()

    assert window.completion_thread is app.thread()
    window.close()


def test_background_worker_uses_qt_safe_deletion_order():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    completed = []

    for _ in range(20):
        worker = _LifecycleWorker()
        worker.finished.connect(lambda result: completed.append(result["ok"]))
        assert window._start_background_worker(worker, "test", warn_if_busy=False)

        loop = QEventLoop()
        window._worker_thread.finished.connect(loop.quit)
        QTimer.singleShot(2000, loop.quit)
        loop.exec()
        app.processEvents()

        assert window._worker_thread is None
        assert window._worker is None

    assert completed == [True] * 20
    window.close()


def test_worker_qobject_is_deleted_before_its_thread_event_loop_stops():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    worker = _LifecycleWorker()
    destroyed = []
    worker.destroyed.connect(lambda: destroyed.append(True))

    assert window._start_background_worker(worker, "test", warn_if_busy=False)
    loop = QEventLoop()
    window._worker_thread.finished.connect(loop.quit)
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    app.processEvents()

    assert destroyed == [True]
    assert not isValid(worker)
    window.close()


def test_model_load_worker_verifies_hash_and_returns_a_separate_engine(tmp_path):
    model_path = tmp_path / "worker-model.pt"
    model_path.write_bytes(b"model-bytes")
    expected = hashlib.sha256(b"model-bytes").hexdigest()
    calls = []

    class FakeEngine:
        def __init__(self):
            self.loaded = False
            self._model_path = ""

        def load(self, path):
            calls.append(path)
            self.loaded = True
            self._model_path = path
            return True

        def is_loaded(self):
            return self.loaded

        def runtime_config(self):
            return {"device": "cpu", "precision": "fp32"}

        def unload(self):
            self.loaded = False

    received = []
    worker = ModelLoadWorker(
        str(model_path),
        expected_sha256=expected,
        engine_factory=FakeEngine,
    )
    worker.finished.connect(received.append)
    worker.run()

    assert received[0]["status"] == "completed"
    result = received[0]["result"]
    assert result["model_path"] == str(model_path)
    assert result["model_sha256"] == expected
    assert result["engine"].is_loaded() is True
    assert calls == [str(model_path)]


def test_model_load_worker_rejects_hash_mismatch_before_constructing_engine(tmp_path):
    model_path = tmp_path / "tampered.pt"
    model_path.write_bytes(b"tampered")
    constructed = []

    class FakeEngine:
        def __init__(self):
            constructed.append(True)

    received = []
    worker = ModelLoadWorker(
        str(model_path),
        expected_sha256="0" * 64,
        engine_factory=FakeEngine,
    )
    worker.finished.connect(received.append)
    worker.run()

    assert received[0]["status"] == "failed"
    assert "哈希" in received[0]["error"]
    assert constructed == []


def test_model_switch_commits_on_gui_thread_and_restores_selector_after_worker(
    monkeypatch, tmp_path
):
    app = QApplication.instance() or QApplication([])
    model_path = tmp_path / "gui-model.pt"
    model_path.write_bytes(b"gui-model")
    expected = hashlib.sha256(b"gui-model").hexdigest()
    sha256 = expected

    class FakeEngine:
        def __init__(self):
            self._model_path = ""
            self.loaded = False
            self.unloaded = False

        def load(self, path):
            self._model_path = path
            self.loaded = True
            return True

        def is_loaded(self):
            return self.loaded

        def unload(self):
            self.unloaded = True
            self.loaded = False

        def runtime_config(self):
            return {"device": "cpu", "precision": "fp32"}

    class ImmediateModelWorker(QObject):
        progress = Signal(int, str)
        finished = Signal(dict)
        error = Signal(str)

        def __init__(self, path, *, expected_sha256="", token=None, **_kwargs):
            super().__init__()
            self.path = str(path)
            self.expected_sha256 = expected_sha256
            self.token = token

        def cancel(self):
            self.token.invalidate()

        @Slot()
        def run(self):
            engine = FakeEngine()
            engine.load(self.path)
            self.progress.emit(10, "loading")
            self.finished.emit({
                "task_token": self.token,
                "status": "completed",
                "result": {
                    "engine": engine,
                    "model_path": self.path,
                    "model_sha256": sha256,
                    "runtime_config": {"device": "cpu", "precision": "fp32"},
                },
                "error": "",
            })

    monkeypatch.setattr("pyside6_app.main_window.ModelLoadWorker", ImmediateModelWorker)
    monkeypatch.setattr(
        "pyside6_app.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    window = MainWindow()
    window._model_list = [{"name": model_path.name, "path": str(model_path)}]
    window._builtin_model_dir = lambda: str(tmp_path / "builtin")
    window._model_registry_entry = lambda _path: {}
    window._sync_cache_model_identity = lambda: True
    window._invalidate_analysis_from = lambda *_args: None
    window._save_system_undo = lambda _label: None
    window._active_inference_runner = None

    assert window._on_load_model_at(str(model_path)) is True
    assert window._model_load_in_progress is True
    assert window.task_panel.model_combo.isEnabled() is False

    loop = QEventLoop()
    window._worker_thread.finished.connect(loop.quit)
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    app.processEvents()

    assert window._model_load_in_progress is False
    assert window.state.current_model_path == str(model_path)
    assert window.state.model_sha256 == expected
    assert window.model_engine._model_path == str(model_path)
    assert window.task_panel.model_combo.isEnabled() is True
    window.close()


def test_model_switch_is_rejected_while_inference_is_really_running(tmp_path):
    app = QApplication.instance() or QApplication([])
    model_path = tmp_path / "blocked.pt"
    model_path.write_bytes(b"blocked")
    window = MainWindow()
    window._active_inference_runner = SimpleNamespace(
        is_running=lambda: True,
        cancel=lambda: (_ for _ in ()).throw(AssertionError("must not cancel")),
    )
    messages = []
    window.log_panel.warn = messages.append

    assert window._on_load_model_at(str(model_path)) is False
    assert any("暂不能切换模型" in message for message in messages)
    window.close()


def test_cached_path_display_replay_keeps_gui_responsive():
    import json
    from pathlib import Path

    app = QApplication.instance() or QApplication([])
    cache_path = Path(__file__).resolve().parents[1] / ".hillcache" / "9a93514240d4" / "project_state.json"
    if not cache_path.is_file():
        return
    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    window = MainWindow()
    result = window._restore_cached_path_geometry(saved)
    assert result == {}
    assert not window.state.auto_path_planned
    timer_fired = []
    QTimer.singleShot(0, lambda: timer_fired.append(True))
    app.processEvents()
    assert timer_fired == [True]
    assert not saved.get("auto_path")
    window.close()


def test_path_risk_notification_is_non_modal_and_keeps_event_loop_responsive(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    modal_calls = []
    timer_fired = []

    monkeypatch.setattr(
        "pyside6_app.main_window.QMessageBox.warning",
        lambda *args, **kwargs: modal_calls.append(args),
    )
    path = {
        "validation": {"valid": False, "issues": ["履带核心重叠超限"]},
        "turn_assessment": {"hard_reasons": []},
    }
    window._warn_path_validation_issues(path)
    QTimer.singleShot(0, lambda: timer_fired.append(True))
    app.processEvents()

    assert modal_calls == []
    assert timer_fired == [True]
    assert "指标已更新" in window.lbl_status.text()
    window.close()


def test_plan_completion_defers_validation_warning_until_worker_cleanup(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    calls = []

    monkeypatch.setattr(window, "_store_path_result", lambda path, result: path)
    monkeypatch.setattr(window, "_show_path", lambda path: calls.append("draw"))
    monkeypatch.setattr(window.route_info, "update_route", lambda *args: calls.append("panel"))
    monkeypatch.setattr(window, "_sync_task_statuses", lambda: None)
    monkeypatch.setattr(window, "_path_quality_summary", lambda *args: "summary")
    monkeypatch.setattr(window, "_log_turn_strategy_result", lambda path: None)
    monkeypatch.setattr(window, "_show_validation_report", lambda *args: calls.append("report"))
    monkeypatch.setattr(window, "_warn_path_validation_issues", lambda path: calls.append("warning"))

    result = {
        "bands": [],
        "path": {
            "tracks": [], "turns": [],
            "validation": {"valid": False, "issues": ["risk"]},
        },
    }
    window._on_plan_worker_done(result)

    assert calls == ["draw", "panel", "report"]
    app.processEvents()
    assert calls == ["draw", "panel", "report", "warning"]
    window.close()
