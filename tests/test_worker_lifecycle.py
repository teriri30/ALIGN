import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")

from PySide6.QtCore import QObject, QEventLoop, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow


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
    assert "存在风险" in window.lbl_status.text()
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
    monkeypatch.setattr(window, "_log_turn_strategy_comparison", lambda path: None)
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
