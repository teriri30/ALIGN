"""原图分辨率快照导出（DECISION-070）的渲染与契约测试。

快照必须满足三件事：
1. 产出文件与源影像同尺寸（不做屏幕/控件分辨率截断）；
2. 掩膜按 viewer 同一混合公式叠加，基底像素不被无谓改写；
3. 图层集合跟随显示图层面板可见性，坐标/色号与画布同源。
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir("C:/Windows/Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from PySide6.QtWidgets import QApplication
from rasterio.transform import from_bounds

from pyside6_app.main_window import (
    FIELD_BOUNDARY_COLOR_BGR,
    OPERATOR_LINE_COLOR_BGR,
    MainWindow,
)
from pyside6_app.snapshot_export import (
    CircleOp,
    LabelOp,
    MaskOp,
    PolygonOp,
    PolylineOp,
    SnapshotExportWorker,
    SnapshotScene,
    render_snapshot_png,
)

# 渲染需要 Qt 应用实例（字体）；先于任何测试创建 QApplication，
# 避免渲染守卫先建裸 QGuiApplication。
QApplication.instance() or QApplication([])

BASE_RGB = (40, 80, 120)  # RGB


def _imread(path):
    """cv2.imread 读不了 Windows 非 ASCII 路径，统一走 fromfile+imdecode。"""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_tif(path, width=120, height=90):
    arr = np.zeros((height, width, 3), np.uint8)
    arr[..., 0] = BASE_RGB[0]
    arr[..., 1] = BASE_RGB[1]
    arr[..., 2] = BASE_RGB[2]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32650",
        transform=from_bounds(0, 0, 60, 45, width, height),
    ) as dst:
        for index in range(3):
            dst.write(arr[..., index], index + 1)


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    return window


def _scene_with_all_ops():
    scene = SnapshotScene()
    mask = np.zeros((30, 40), np.uint8)
    mask[5:25, 5:35] = 255
    scene.mask(mask, 0, 0, (120, 255, 140))
    scene.polyline([(2, 2), (38, 2)], (0, 0, 255), width=2)
    scene.circle(20, 15, 3, (255, 255, 0))
    scene.label(20, 22, "作业线 1", size=9)
    scene.polygon(
        [(6, 24), (12, 24), (12, 28), (6, 28)], (35, 35, 220),
        outline=(20, 20, 255), width=2,
    )
    return scene


def test_render_snapshot_png_writes_full_resolution_overlays(tmp_path):
    tif = tmp_path / "source.tif"
    out = tmp_path / "快照1.png"
    _write_tif(tif)
    scene = _scene_with_all_ops()

    meta = render_snapshot_png(str(tif), scene, str(out), display_shape=(30, 40))

    assert out.is_file()
    image = _imread(out)
    # 源分辨率（显示网格 40×30 → 原图 120×90，各 3 倍）。
    assert image.shape[:2] == (90, 120)
    assert meta["source_shape"] == [90, 120]
    assert meta["display_scale"] == [3.0, 3.0]
    # 基底未被叠加区域保持原值（RGB→BGR 读取顺序）。
    assert image[0, 0].tolist() == [BASE_RGB[2], BASE_RGB[1], BASE_RGB[0]]
    # 掩膜混合公式与 viewer 一致：out = dst*(1-a) + color*a，a=115/255。
    alpha = 115 / 255.0
    expected = [
        BASE_RGB[2] * (1 - alpha) + 120 * alpha,  # B = 原B 与 color BGR(120,255,140)→RGB(140,255,120) 的 B=120
        BASE_RGB[1] * (1 - alpha) + 255 * alpha,
        BASE_RGB[0] * (1 - alpha) + 140 * alpha,
    ]
    pixel = image[30, 30].tolist()  # display (10,10) → source (30,30)，掩膜内部
    for got, want in zip(pixel, expected):
        assert abs(got - want) <= 2, (pixel, expected)
    # 红色轨迹线存在（BGR 红 = (0,0,255)）。
    red = (
        (np.abs(image[:, :, 2].astype(int) - 255) < 40)
        & (image[:, :, 1] < 80)
        & (image[:, :, 0] < 80)
    )
    assert int(red.sum()) > 100
    # 圆点中心为纯青色（BGR (255,255,0)）。
    assert image[45, 60].tolist() == [255, 255, 0]
    # 标签文字着色（作业线 1，中心 display(20,22) → source(60,66) 附近）。
    box = image[60:72, 48:84]
    changed = int(
        np.sum(np.any(np.abs(box.astype(int) - [120, 80, 40]) > 12, axis=2))
    )
    assert changed > 30


def test_render_snapshot_png_jpg_output(tmp_path):
    tif = tmp_path / "source.tif"
    out = tmp_path / "snapshot.jpg"
    _write_tif(tif)

    meta = render_snapshot_png(
        str(tif), SnapshotScene(), str(out), display_shape=(30, 40)
    )

    assert out.is_file()
    assert meta["format"] == "JPG"
    image = _imread(out)
    assert image.shape[:2] == (90, 120)
    assert abs(int(image[0, 0][2]) - BASE_RGB[0]) <= 2


def test_render_snapshot_png_rejects_invalid_display_shape(tmp_path):
    tif = tmp_path / "source.tif"
    _write_tif(tif)
    with pytest.raises(ValueError):
        render_snapshot_png(
            str(tif), SnapshotScene(), str(tmp_path / "x.png"),
            display_shape=(0, 0),
        )


def test_snapshot_export_worker_runs_and_reports(tmp_path):
    tif = tmp_path / "source.tif"
    out = tmp_path / "快照1.png"
    _write_tif(tif)
    worker = SnapshotExportWorker(str(tif), _scene_with_all_ops(), str(out), (30, 40))
    finished, errors, progress = [], [], []

    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.progress.connect(lambda pct, msg: progress.append((pct, msg)))

    worker.run()  # 测试内同步执行（生产中由 QThread 调起）

    assert not errors, errors
    assert finished and finished[0]["source_shape"] == [90, 120]
    assert finished[0]["mask_count"] == 1
    assert progress and progress[0][0] == 5
    assert out.is_file()


def test_snapshot_export_worker_reports_cancel(tmp_path):
    tif = tmp_path / "source.tif"
    _write_tif(tif)
    worker = SnapshotExportWorker(
        str(tif), SnapshotScene(), str(tmp_path / "x.png"), (30, 40)
    )
    worker.cancel()
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors and "取消" in errors[0]


def _prepare_rich_state(window):
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 5:15] = 255
    window.state.mask_raw = mask
    window.state.mask_offset_x = 10
    window.state.mask_offset_y = 10
    window.state.field_boundary = [(2, 2), (38, 2), (38, 38), (2, 38)]
    window.state.centerline_catalog = {
        "items": [
            {"id": "w1", "label": "作业线 1", "centerline": [[3, 20], [37, 20]]}
        ]
    }
    window.state.selected_centerline_ids = ["w1"]
    window.state.auto_path_planned = True
    window.state.auto_path_segments = [
        SimpleNamespace(points=[(4, 30), (36, 30)], segment_type="work", length_m=1.0)
    ]
    window._pipeline_result = {
        "path": {"tracks": [{"points": [[4, 30], [36, 30]]}], "turns": []}
    }
    window._service_points_visible = True
    window.state.entry_point = (5, 5)
    window._sync_display_layer_availability()


def test_build_snapshot_scene_collects_visible_layers(tmp_path):
    window = _window()
    _prepare_rich_state(window)
    try:
        scene = window._build_snapshot_scene()
        masks = scene.masks
        assert any(isinstance(op, MaskOp) for op in masks)
        polylines = [op for op in scene.vectors if isinstance(op, PolylineOp)]
        # 一条线原则：路径已生成且路线层可见时，作业线折线与轨迹同一份几何，
        # 不再单独画（否则同位置叠出一条虚影边，看起来像"两条线"）。
        assert not any(op.color_bgr == (40, 220, 80) for op in polylines)
        # 田块边界（蓝）与轨迹（绿粗）都在。
        assert any(op.color_bgr == FIELD_BOUNDARY_COLOR_BGR for op in polylines)
        assert any(op.color_bgr == (0, 255, 70) and op.width == 4 for op in polylines)
        labels = [op for op in scene.vectors if isinstance(op, LabelOp)]
        # rich state 已有生成路径：作业线编号按 2026-11-10 规则自动隐藏。
        assert not any(op.text == "作业线 1" for op in labels)
        assert any(op.text == "起点" for op in labels)
        circles = [op for op in scene.vectors if isinstance(op, CircleOp)]
        assert any(op.color_bgr == (120, 255, 150) for op in circles)
        assert circles, "centerline endpoints missing"
        # 田块顶点圆半径与画布 _field_vertex_radius 同值（缩小后 5）。
        assert any(
            op.color_bgr == (255, 255, 0) and op.radius == 5 for op in circles
        )
    finally:
        window.close()


def test_build_snapshot_scene_shows_centerline_labels_before_path(tmp_path):
    window = _window()
    _prepare_rich_state(window)
    try:
        # 路径未生成（规划前/失效后）：编号必须回到画布与快照上。
        window.state.auto_path_planned = False
        window.state.auto_path_segments = []
        scene = window._build_snapshot_scene()
        labels = [op for op in scene.vectors if isinstance(op, LabelOp)]
        assert any(op.text == "作业线 1" for op in labels)
        # 折线也随路径失效恢复（路径生效时由路线层承载，见一条线原则）。
        polylines = [op for op in scene.vectors if isinstance(op, PolylineOp)]
        assert any(op.color_bgr == (40, 220, 80) for op in polylines)
    finally:
        window.close()


def test_centerline_labels_hidden_follows_path_state(tmp_path):
    window = _window()
    _prepare_rich_state(window)
    try:
        # 有生效路径 → 隐藏。
        assert window._centerline_labels_hidden() is True
        # 路径被清除（下游失效回到勾选阶段）→ 恢复。
        window.state.auto_path_planned = False
        window.state.auto_path_segments = []
        assert window._centerline_labels_hidden() is False
        # 待重新规划的虚线预览不算生效路径 → 编号保持显示。
        window._stale_path_preview = {"tracks": [{"points": [[1, 1], [2, 2]]}]}
        assert window._centerline_labels_hidden() is False
    finally:
        window.close()


def test_build_snapshot_scene_respects_layer_visibility(tmp_path):
    window = _window()
    _prepare_rich_state(window)
    try:
        window.image_view.set_display_layer_visibility(
            "centerlines", False, user=True
        )
        scene = window._build_snapshot_scene()
        polylines = [op for op in scene.vectors if isinstance(op, PolylineOp)]
        assert not any(op.color_bgr == (40, 220, 80) for op in polylines)
        labels = [op for op in scene.vectors if isinstance(op, LabelOp)]
        assert not any(op.text == "作业线 1" for op in labels)
        # 其余图层不受影响。
        assert any(op.color_bgr == FIELD_BOUNDARY_COLOR_BGR for op in polylines)
    finally:
        window.close()


def test_build_snapshot_scene_compare_mode_is_empty(tmp_path):
    window = _window()
    _prepare_rich_state(window)
    try:
        window._show_mask_before = True
        scene = window._build_snapshot_scene()
        assert not scene.masks and not scene.vectors
    finally:
        window.close()


def test_build_snapshot_scene_uses_operator_line_when_calibrated(tmp_path):
    window = _window()
    _prepare_rich_state(window)
    try:
        window._operator_payload_override = {
            "lines": [
                {
                    "id": "op1",
                    "centerline_id": "w1",
                    "points": [[3, 20], [37, 24]],
                }
            ]
        }
        # 标定线存在时，画线数据走人线（同画布 _draw_centerline_overlay 规则）。
        # 清除路径让中心线折线恢复绘制：路径生效时折线由路线层承载（一条线原则），
        # 本测试验证的是折线的数据来源与坐标偏移规则。
        window.state.auto_path_planned = False
        window.state.auto_path_segments = []
        monkey_target = window._operator_paths_payload
        window._operator_paths_payload = lambda: window._operator_payload_override
        try:
            scene = window._build_snapshot_scene()
        finally:
            window._operator_paths_payload = monkey_target
        polylines = [op for op in scene.vectors if isinstance(op, PolylineOp)]
        operator_lines = [
            op for op in polylines if op.color_bgr == OPERATOR_LINE_COLOR_BGR
        ]
        assert operator_lines, "calibrated operator line missing"
        # 标定记录存的是掩膜局部坐标，画布规则是 + mask_offset（此处 10,10）。
        assert operator_lines[0].points[-1] == (47.0, 34.0)
    finally:
        window.close()


def test_export_full_resolution_snapshot_requires_loaded_tif(tmp_path, monkeypatch):
    window = _window()
    errors = []
    monkeypatch.setattr(
        window.log_panel, "error", lambda message: errors.append(message)
    )
    try:
        window._tif_path = str(tmp_path / "missing.tif")
        window._export_full_resolution_snapshot(str(tmp_path / "快照1.png"))
        assert errors and "请先加载影像" in errors[0]
    finally:
        window.close()


def test_export_full_resolution_snapshot_end_to_end(tmp_path, monkeypatch):
    # 走真实渲染（monkeypatch 只截掉 QThread 启动，同步调 worker.run）。
    tif = tmp_path / "source.tif"
    _write_tif(tif)
    window = _window()
    _prepare_rich_state(window)
    started = {}

    def fake_start(worker, message, **kwargs):
        started["worker"] = worker
        started["message"] = message
        return True

    monkeypatch.setattr(window, "_start_background_worker", fake_start)
    window._tif_path = str(tif)
    window._snapshot_directory = str(tmp_path)
    try:
        window._export_full_resolution_snapshot(str(tmp_path / "快照1.png"))
        assert "worker" in started
        worker = started["worker"]
        worker.run()  # 同步执行后台逻辑
        out = tmp_path / "快照1.png"
        assert out.is_file()
        image = _imread(out)
        assert image.shape[:2] == (90, 120)
    finally:
        window.close()


def test_render_snapshot_png_scales_metrics_with_view_scale(tmp_path):
    """WYSIWYG：线宽/半径按 display_scale / view_scale 换算（用户 2026-11-10）。"""
    tif = tmp_path / "source.tif"
    _write_tif(tif)
    scene = SnapshotScene()
    scene.polyline([(2, 10), (38, 10)], (0, 0, 255), width=2)

    plain = render_snapshot_png(
        str(tif), scene, str(tmp_path / "plain.png"), display_shape=(30, 40)
    )
    zoomed_out = render_snapshot_png(
        str(tif), scene, str(tmp_path / "zoomed.png"),
        display_shape=(30, 40), view_scale=0.5,
    )

    # display_scale=3；view_scale=1（100% 缩放）→ 3；view_scale=0.5 → 6。
    assert plain["metric_scale"] == pytest.approx(3.0)
    assert zoomed_out["metric_scale"] == pytest.approx(6.0)

    def line_rows(pathname):
        column = _imread(pathname)[:, 60]
        red = (
            (column[:, 2] > 200) & (column[:, 1] < 90) & (column[:, 0] < 90)
        )
        return int(red.sum())

    thin, thick = line_rows(tmp_path / "plain.png"), line_rows(tmp_path / "zoomed.png")
    assert thick >= 2 * thin - 1, (thin, thick)


def test_snapshot_scene_skips_canvas_culled_centerline_labels(tmp_path):
    """快照编号只画本帧画布真正画出来的（碰撞过滤随缩放，2026-11-10）。"""
    window = _window()
    _prepare_rich_state(window)
    try:
        window.state.auto_path_planned = False
        window.state.auto_path_segments = []
        window._centerline_visible_label_ids = set()
        scene = window._build_snapshot_scene()
        labels = [op for op in scene.vectors if isinstance(op, LabelOp)]
        assert not any(op.text == "作业线 1" for op in labels)

        # 没有取过取舍结果（None）时退回画全部，避免画布未重绘就导出时丢标注。
        window._centerline_visible_label_ids = None
        scene = window._build_snapshot_scene()
        labels = [op for op in scene.vectors if isinstance(op, LabelOp)]
        assert any(op.text == "作业线 1" for op in labels)
    finally:
        window.close()


def test_export_snapshot_passes_view_scale_to_worker(tmp_path, monkeypatch):
    """导出必须带上画布当时的视图比例尺，否则快照等于 100% 缩放下的画布。"""
    tif = tmp_path / "source.tif"
    _write_tif(tif)
    window = _window()
    _prepare_rich_state(window)
    started = {}

    def fake_start(worker, message, **kwargs):
        started["worker"] = worker
        return True

    monkeypatch.setattr(window, "_start_background_worker", fake_start)
    monkeypatch.setattr(
        window.image_view.viewer, "get_view_scale", lambda: 0.25
    )
    window._tif_path = str(tif)
    try:
        window._export_full_resolution_snapshot(str(tmp_path / "快照1.png"))
        assert started["worker"]._view_scale == pytest.approx(0.25)
    finally:
        window.close()


def test_centerline_endpoint_dots_snap_to_work_segment_ends():
    """端点圆必须落在作业段端点（= 转弯衔接处），不是中心线端点。

    规划按 work_line_endpoint_extension_m 把作业段两端各外延 0.5 m，转弯弧线
    接在外延端；中心线目录不含这段外延（用户 2026-11-10 报"端点提前收住"）。
    """
    import math

    window = _window()
    _prepare_rich_state(window)
    try:
        item = window.state.centerline_catalog["items"][0]
        points = window._centerline_display_points(item)
        first, last = points[0], points[-1]
        span = math.hypot(last[0] - first[0], last[1] - first[1])
        ux = (last[0] - first[0]) / span
        uy = (last[1] - first[1]) / span
        extension = 0.02 * span  # 模拟 0.5 m 外延（比例与线长无关）
        work_start = (first[0] - ux * extension, first[1] - uy * extension)
        work_end = (last[0] + ux * extension, last[1] + uy * extension)
        segment = SimpleNamespace(
            points=[work_start, work_end],
            segment_type="work",
            length_m=1.0,
            status=1,
            geometry_mode="straight",
        )
        window.state.auto_path_planned = True
        window.state.auto_path_segments = [segment]
        window.state.auto_path = [segment]

        markers = window._centerline_endpoint_markers(
            points, window._work_segment_endpoints()
        )
        assert markers[0] == pytest.approx(work_start)
        assert markers[1] == pytest.approx(work_end)

        # 没有路径 → 保持中心线端点
        assert window._centerline_endpoint_markers(points, []) == (first, last)

        # 只有远处别的作业段 → 不许乱吸附（相邻线间距 16 m）
        far = [
            (
                (work_start[0] + 5000.0, work_start[1]),
                (work_end[0] + 5000.0, work_end[1]),
            )
        ]
        assert window._centerline_endpoint_markers(points, far) == (first, last)

        # 快照端点圆与作业段端点一致
        scene = window._build_snapshot_scene()
        dots = [
            op
            for op in scene.vectors
            if isinstance(op, CircleOp) and tuple(op.color_bgr) == (120, 255, 150)
        ]
        assert dots
        for op in dots:
            nearest = min(
                math.hypot(op.x - ref[0], op.y - ref[1])
                for ref in (work_start, work_end)
            )
            assert nearest < 1.0, nearest
    finally:
        window.close()


