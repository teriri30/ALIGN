# -*- coding: utf-8 -*-
"""交互日志（操作记录）的写入、上下文与导出契约。"""
import json

import interaction_log


def _fresh(path):
    log = interaction_log.ActionLog()
    log.set_log_path(str(path))
    log.configure(app="ALIGN", version="test", tif_path="")
    return log


def test_records_are_appended_with_context_and_sequence(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIGN_DISABLE_ACTION_LOG", raising=False)
    log = _fresh(tmp_path / "a.jsonl")
    log.configure(tif_path=r"E:\field\result.tif")
    log.record("plan.request", "path", selected=14)
    log.record("calibration.edit", "centerline-003", offset_m=-0.08)
    records = log.read()
    assert [item["action"] for item in records[:2]] == ["plan.request", "calibration.edit"]
    assert records[0]["context"]["tif_path"] == r"E:\field\result.tif"
    assert records[0]["payload"]["selected"] == 14
    assert records[1]["seq"] == records[0]["seq"] + 1
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["schema"] == interaction_log.SCHEMA


def test_recent_filters_by_action_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIGN_DISABLE_ACTION_LOG", raising=False)
    log = _fresh(tmp_path / "b.jsonl")
    for action in ("menu.action", "calibration.edit", "calibration.mode", "plan.request"):
        log.record(action, "")
    recent = log.recent(10, action_prefix="calibration.")
    assert [item["action"] for item in recent] == ["calibration.edit", "calibration.mode"]


def test_logging_never_raises_when_the_payload_is_not_json_serialisable(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIGN_DISABLE_ACTION_LOG", raising=False)
    log = _fresh(tmp_path / "c.jsonl")

    class Weird:
        def __str__(self):
            return "weird"

    record = log.record("debug.probe", "", value=Weird(), array=[Weird()])
    assert record["payload"]["value"] == "weird"
    assert log.read()[-1]["payload"]["value"] == "weird"


def test_rotation_keeps_a_backup_when_the_file_grows_past_the_limit(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIGN_DISABLE_ACTION_LOG", raising=False)
    monkeypatch.setattr(interaction_log, "MAX_BYTES", 400)
    log = _fresh(tmp_path / "d.jsonl")
    for index in range(20):
        log.record("probe.fill", "", index=index, text="x" * 40)
    assert (tmp_path / "d.jsonl").is_file()
    assert (tmp_path / "d.jsonl.1").is_file()
    assert log.read()  # 轮转后活动文件仍可读


def test_disabled_log_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIGN_DISABLE_ACTION_LOG", "1")
    log = _fresh(tmp_path / "e.jsonl")
    log.record("probe.disabled", "")
    assert not (tmp_path / "e.jsonl").exists()


def test_export_copies_the_current_log(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIGN_DISABLE_ACTION_LOG", raising=False)
    log = _fresh(tmp_path / "f.jsonl")
    log.record("probe.one", "")
    written = log.export_to(str(tmp_path / "out"))
    assert written
    assert (tmp_path / "out" / "f.jsonl").is_file()
