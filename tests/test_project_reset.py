# -*- coding: utf-8 -*-
"""「从零开始（清除本项目）」契约。

用户场景：演示前要把同一块田从头跑一遍。任务行的「回退」只清内存；缓存按 tif
路径哈希落盘，重新选择同一张影像时 CacheRestoreWorker 会把田块、掩膜、作业线、
路径和操作历史整条恢复回来。所以这里钉住四件事：
1. 磁盘清理范围：分析产物与操作历史删掉，影像解码缓存、显示参数、导出产物留下；
2. 人工标注 sidecar 默认保留，只有显式勾选才删；
3. 排队中的旧状态保存必须能被丢弃，否则删完文件又被写回来；
4. 界面上有唯一入口，执行后内存回到“刚导入影像”，影像仍然在（不必重新选文件）。
全部离屏运行，不依赖真实影像。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

# shiboken：QAction.menu() 是借用指针，包装器被回收后 QMenu 会 “already deleted”。
_KEEP = []


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", str(tmp_path / "cache"))
    return tmp_path


def _seed_project(tif_path):
    """按真实落盘结构造一个“已经跑过一遍”的项目目录。"""
    import cache

    root = cache.project_cache_dir(tif_path)
    os.makedirs(os.path.join(root, "history", "snapshots"), exist_ok=True)
    payloads = {
        "project_state.json": b"{}",
        "bands.json": b"[]",
        "mask.npz": b"mask-raw",
        "mask_best.npz": b"mask-best",
        "mask_processed.npz": b"mask-processed",
        "mask_smoothed.npz": b"mask-smoothed",
        "meta.json": b'{"display_scale_x": 4}',
        "image_bgr.npy": b"image",
        "image_bgr.json": b"{}",
        "reference_lines.json": b'{"lines": [1]}',
        "reference_draft_p1.json": b"{}",
        "operator_paths.json": b'{"lines": [1]}',
    }
    for name, payload in payloads.items():
        with open(os.path.join(root, name), "wb") as handle:
            handle.write(payload)
    with open(os.path.join(root, "history", "index.json"), "wb") as handle:
        handle.write(b"{}")
    eval_dir = os.path.join(root, "centerline_eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "per_line.csv"), "wb") as handle:
        handle.write(b"line,elat\n")
    return root


# ---------------------------------------------------------------- 1. 磁盘清理范围
def test_clear_project_cache_removes_analysis_and_keeps_image_and_annotations(cache_root):
    import cache

    tif_path = str(cache_root / "field.tif")
    root = _seed_project(tif_path)

    report = cache.clear_project_cache(tif_path)

    for name in (
        "project_state.json",
        "bands.json",
        "mask.npz",
        "mask_best.npz",
        "mask_processed.npz",
        "mask_smoothed.npz",
    ):
        assert not os.path.exists(os.path.join(root, name)), name
    assert not os.path.isdir(os.path.join(root, "history")), "操作历史必须整目录删掉"
    for name in (
        "meta.json",
        "image_bgr.npy",
        "image_bgr.json",
        "reference_lines.json",
        "reference_draft_p1.json",
        "operator_paths.json",
    ):
        assert os.path.isfile(os.path.join(root, name)), name
    assert os.path.isfile(os.path.join(root, "centerline_eval", "per_line.csv"))
    assert "history/" in report["removed"]
    assert report["removed_bytes"] > 0
    assert report["annotations_cleared"] is False
    assert report["failed"] == []


def test_clear_project_cache_only_drops_annotations_when_asked(cache_root):
    import cache

    tif_path = str(cache_root / "field.tif")
    root = _seed_project(tif_path)

    report = cache.clear_project_cache(tif_path, include_annotations=True)

    for name in (
        "reference_lines.json",
        "reference_draft_p1.json",
        "operator_paths.json",
    ):
        assert not os.path.exists(os.path.join(root, name)), name
    assert report["annotations_cleared"] is True
    # 影像侧缓存仍然保留，重开不必重新解码
    assert os.path.isfile(os.path.join(root, "meta.json"))
    assert os.path.isfile(os.path.join(root, "image_bgr.npy"))


def test_clear_project_cache_is_safe_without_a_cache_dir(cache_root):
    import cache

    report = cache.clear_project_cache(str(cache_root / "never-opened.tif"))

    assert report["removed"] == []
    assert report["removed_bytes"] == 0


# ---------------------------------------------------------------- 2. 排队保存
def test_cancel_pending_project_saves_drops_queued_payload(cache_root):
    import cache

    tif_path = str(cache_root / "field.tif")
    key = os.path.abspath(tif_path)
    with cache._STATE_SAVE_QUEUE_LOCK:
        cache._PENDING_STATE_SAVES[key] = (tif_path, {}, {}, "", None)
        cache._ACTIVE_STATE_SAVERS.discard(key)
    try:
        assert cache.cancel_pending_project_saves(tif_path) == 1
        assert cache.cancel_pending_project_saves(tif_path) == 0
    finally:
        with cache._STATE_SAVE_QUEUE_LOCK:
            cache._PENDING_STATE_SAVES.pop(key, None)
            cache._ACTIVE_STATE_SAVERS.discard(key)


# ---------------------------------------------------------------- 3. 界面入口 + 内存归零
def _window(tmp_path):
    QApplication.instance() or QApplication([])
    from pyside6_app.main_window import MainWindow

    window = MainWindow()
    _KEEP.append(window)
    return window


def test_reset_project_has_exactly_one_menu_entry(cache_root):
    window = _window(cache_root)
    acts = window.menuBar().actions()
    _KEEP.extend(acts)
    file_menu = next(a.menu() for a in acts if a.text() == "文件")
    _KEEP.append(file_menu)
    file_acts = file_menu.actions()
    _KEEP.extend(file_acts)
    labels = [a.text() for a in file_acts if a.text()]
    assert labels.count("从零开始（清除本项目）…") == 1


def test_reset_project_clears_disk_and_memory_but_keeps_the_image(cache_root, monkeypatch):
    import cache

    window = _window(cache_root)
    tif_path = str(cache_root / "field.tif")
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window._tif_rgb = rgb
    window._tif_path = tif_path
    window.image_view.set_image(rgb)
    window.state.safe_update(
        tif_path=tif_path,
        img_h=20,
        img_w=20,
        source_img_h=20,
        source_img_w=20,
        display_h=20,
        display_w=20,
        field_boundary=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        forbidden_regions=[[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]],
        entry_point=(1.0, 1.0),
        auto_path_planned=True,
        export_done=True,
        inference_done=True,
        mask_processed=True,
    )
    root = _seed_project(tif_path)

    monkeypatch.setattr(window, "_ask_reset_project", lambda *_: (True, False))
    window._on_reset_project()

    # 磁盘：处理结果与操作历史没了
    assert not os.path.exists(os.path.join(root, "project_state.json"))
    assert not os.path.exists(os.path.join(root, "mask.npz"))
    assert not os.path.isdir(os.path.join(root, "history"))
    # 人工标注默认留下
    assert os.path.isfile(os.path.join(root, "reference_lines.json"))
    # 内存：回到“刚导入影像”，但影像还在，不需要重新选文件
    assert window._tif_path == tif_path
    assert window.state.tif_path == tif_path
    assert window.state.img_w == 20 and window.state.img_h == 20
    assert window.state.field_boundary == []
    assert window.state.forbidden_regions == []
    assert window.state.entry_point is None
    assert window.state.auto_path_planned is False
    assert window.state.export_done is False
    assert window.state.inference_done is False
    assert window.state.mask_processed is False
    assert window.state.undo_stack == []
    assert window._pipeline_result is None


def test_reset_project_is_refused_while_processing(cache_root, monkeypatch):
    window = _window(cache_root)
    tif_path = str(cache_root / "field.tif")
    window._tif_path = tif_path
    window.state.safe_update(tif_path=tif_path, inference_running=True)
    root = _seed_project(tif_path)

    called = []
    monkeypatch.setattr(
        window, "_ask_reset_project", lambda *_: called.append(True) or (True, False)
    )
    window._on_reset_project()

    assert called == [], "处理中不该弹确认框"
    assert os.path.isfile(os.path.join(root, "project_state.json")), "处理中不得删缓存"
