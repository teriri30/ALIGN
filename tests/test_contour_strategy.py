import numpy as np
import pytest


def test_contour_order_is_outside_in_nested_sweep():
    """DECISION-016 v3: contour visits the two outside sides inward."""
    from path_planner import _contour_order_work_lines

    lines = [
        [(10.0, 10.0), (90.0, 10.0)],
        [(10.0, 20.0), (90.0, 20.0)],
        [(10.0, 30.0), (90.0, 30.0)],
        [(10.0, 40.0), (90.0, 40.0)],
    ]
    polygon = np.asarray(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]],
        dtype=np.float64,
    )

    ordered, diagnostics = _contour_order_work_lines(lines, polygon, main_angle=0.0)

    assert diagnostics["fallback_reason"] is None
    assert diagnostics["algorithm_version"] == "contour-order-v3"
    assert diagnostics["ordered_original_indices"] == [0, 3, 1, 2]
    assert ordered == [lines[index] for index in [0, 3, 1, 2]]


def test_contour_order_works_without_field_boundary():
    """DECISION-016 v3: ordering is geometric; connector construction owns the boundary gate."""
    from path_planner import _contour_order_work_lines

    lines = [[(0.0, 2.0), (10.0, 2.0)], [(0.0, 1.0), (10.0, 1.0)]]
    ordered, diagnostics = _contour_order_work_lines(lines, None)

    assert diagnostics["fallback_reason"] is None
    assert diagnostics["ordered_original_indices"] == [1, 0]
    assert ordered == [lines[1], lines[0]]


def test_contour_order_is_outside_in_nested_loop_sequence():
    """DECISION-016: 套圈按外侧两边向内交替访问，形成回字形。"""
    from path_planner import _contour_order_work_lines

    lines = [
        [(10.0, 10.0), (90.0, 10.0)],
        [(10.0, 20.0), (90.0, 20.0)],
        [(10.0, 30.0), (90.0, 30.0)],
        [(10.0, 40.0), (90.0, 40.0)],
    ]

    ordered, diagnostics = _contour_order_work_lines(
        lines, None, main_angle=0.0
    )

    assert diagnostics["fallback_reason"] is None
    assert diagnostics["ordering"] == "contour_outside_in_nested"
    assert diagnostics["algorithm_version"] == "contour-order-v3"
    assert diagnostics["ordered_original_indices"] == [0, 3, 1, 2]
    assert ordered == [lines[index] for index in [0, 3, 1, 2]]


def test_contour_order_uses_normal_axis_for_skewed_lines():
    """排序必须沿垂直于主方向的法向，而不是直接按 x/y 轴。"""
    from path_planner import _contour_order_work_lines

    # 主方向 -45°：法向位置从小到大对应 index 2, 0, 1；
    # 套圈从两侧向内取值，因此顺序为 2, 1, 0。
    lines = [
        [(10.0, 30.0), (40.0, 0.0)],
        [(20.0, 40.0), (50.0, 10.0)],
        [(0.0, 20.0), (30.0, -10.0)],
    ]
    ordered, diagnostics = _contour_order_work_lines(
        lines, None, main_angle=-0.7853981633974483
    )

    assert diagnostics["fallback_reason"] is None
    assert diagnostics["ordered_original_indices"] == [2, 1, 0]
    assert ordered == [lines[2], lines[1], lines[0]]


def test_contour_order_falls_back_on_empty_or_invalid_lines():
    from path_planner import _contour_order_work_lines

    ordered, diagnostics = _contour_order_work_lines([], None)
    assert ordered == []
    assert diagnostics["fallback_reason"] == "no_work_lines"

    bad = [[(0.0, 0.0), (float("nan"), 1.0)]]
    ordered, diagnostics = _contour_order_work_lines(bad, None, main_angle=0.0)
    assert diagnostics["fallback_reason"] == "invalid_work_line"
    assert ordered == [list(bad[0])]


def test_contour_connector_anchors_share_each_pair_end_wall():
    """DECISION-016: 同一田头的各圈共用端墙且跨度逐圈减小。"""
    from path_planner import _build_contour_connector_anchors

    lines = [
        [(10.0, 10.0), (90.0, 10.0)],
        [(90.0, 40.0), (10.0, 40.0)],
        [(10.0, 20.0), (90.0, 20.0)],
        [(90.0, 30.0), (10.0, 30.0)],
    ]
    polygon = np.asarray(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]],
        dtype=np.float64,
    )

    anchors, diagnostics = _build_contour_connector_anchors(
        lines, main_angle=0.0, field_polygon=polygon, mpp=0.1,
        config={"endpoint_extension_max_m": 4.0},
    )

    assert diagnostics["fallback_reason"] is None
    assert diagnostics["algorithm_version"] == "contour-connectors-v2"
    assert diagnostics["shared_headland_lines"] is True
    assert diagnostics["turn_distances_m"] == pytest.approx([3.0, 2.0, 1.0])
    assert diagnostics["turn_distances_nonincreasing"] is True
    assert anchors[0]["end"][0] == pytest.approx(90.0)
    assert anchors[1]["start"][0] == pytest.approx(90.0)
    assert anchors[1]["end"][0] == pytest.approx(10.0)
    assert anchors[2]["start"][0] == pytest.approx(10.0)
    assert anchors[2]["end"][0] == pytest.approx(90.0)
    assert anchors[3]["start"][0] == pytest.approx(90.0)


def test_contour_outer_loop_may_cross_more_than_four_metres():
    """横向田头跨度是田宽，不得再被旧的 4 m 上限误拒绝。"""
    from path_planner import _build_contour_connector_anchors

    lines = [
        [(10.0, 0.0), (90.0, 0.0)],
        [(90.0, 800.0), (10.0, 800.0)],
        [(10.0, 200.0), (90.0, 200.0)],
        [(90.0, 600.0), (10.0, 600.0)],
        [(10.0, 400.0), (90.0, 400.0)],
    ]
    polygon = np.asarray(
        [[0.0, -10.0], [100.0, -10.0], [100.0, 810.0], [0.0, 810.0]],
        dtype=np.float64,
    )

    anchors, diagnostics = _build_contour_connector_anchors(
        lines, main_angle=0.0, field_polygon=polygon, mpp=0.1,
        config={"contour_max_connector_m": 1.0},  # legacy key is irrelevant
    )

    assert len(anchors) == 5
    assert diagnostics["fallback_reason"] is None
    assert diagnostics["turn_distances_m"] == pytest.approx(
        [80.0, 60.0, 40.0, 20.0]
    )
    assert diagnostics["center_single_finish"] is True


def test_contour_connector_fails_closed_when_axial_extension_is_long():
    """端点沿主航线错位过大时失败关闭，不能生成旧的纵向长橙线。"""
    from path_planner import _build_contour_connector_anchors

    lines = [
        [(30.0, 10.0), (70.0, 10.0)],
        [(80.0, 40.0), (20.0, 40.0)],
    ]
    polygon = np.asarray(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]],
        dtype=np.float64,
    )

    anchors, diagnostics = _build_contour_connector_anchors(
        lines, main_angle=0.0, field_polygon=polygon, mpp=0.1,
        config={"contour_max_endpoint_extension_m": 0.5},
    )

    assert anchors == []
    assert diagnostics["fallback_reason"] == "headland_wall_extension_too_long"
    assert diagnostics["failed_extension_m"] == pytest.approx(1.0)


def test_contour_connection_failure_is_reported_in_final_diagnostics():
    """Unsupported contour geometry must not be reported as a successful route."""
    from path_planner import _connect_work_lines

    class _Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class _State:
        mask_raw = np.zeros((100, 100), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
        entry_point = None
        exit_point = None
        unload_points = []

    path, segment_types, diagnostics = _connect_work_lines(
        [
            [(30.0, 10.0), (70.0, 10.0)],
            [(80.0, 40.0), (20.0, 40.0)],
        ],
        geo=_Geo(),
        state=_State(),
        min_turn_radius_m=1.0,
        turn_strategy="contour",
        config={"contour_max_endpoint_extension_m": 0.5},
        global_main_angle=0.0,
        return_diagnostics=True,
    )

    assert path
    assert diagnostics["contour_route_mode"] is False
    assert diagnostics["contour_connector"]["fallback_reason"] == (
        "headland_wall_extension_too_long"
    )
    assert diagnostics["feasible"] is False
    assert diagnostics["machine_executable"] is False
    assert any(
        reason.startswith("contour_geometry_unavailable:")
        for reason in diagnostics["fallback_reasons"]
    )


def test_contour_route_reuses_headland_lines_and_keeps_center_single():
    """完整路线为外圈向内，田头横段重叠，奇数行中心单行收尾。"""
    from path_planner import _connect_work_lines

    class _Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class _State:
        mask_raw = np.zeros((100, 100), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)
        ]
        entry_point = None
        exit_point = None
        unload_points = []

    lines = [
        [(20.0, y), (80.0, y)]
        for y in (10.0, 30.0, 50.0, 70.0, 90.0)
    ]
    path, segment_types, diagnostics = _connect_work_lines(
        lines,
        geo=_Geo(),
        state=_State(),
        min_turn_radius_m=1.0,
        turn_strategy="contour",
        config={
            "turn_outward_clearance_m": 0.1,
            "turn_safety_margin_m": 0.1,
            "endpoint_extension_max_m": 0.0,
        },
        global_main_angle=0.0,
        return_diagnostics=True,
    )

    work = [segment for segment, kind in zip(path, segment_types) if kind == "work"]
    turns = [segment for segment, kind in zip(path, segment_types) if kind == "turn"]
    assert [segment[0][1] for segment in work] == [10.0, 90.0, 30.0, 70.0, 50.0]
    assert segment_types == ["work", "turn", "work", "turn", "work", "turn", "work", "turn", "work"]
    assert diagnostics["contour_connector"]["turn_distances_m"] == pytest.approx(
        [8.0, 6.0, 4.0, 2.0]
    )
    assert diagnostics["contour_connector"]["center_single_finish"] is True
    assert diagnostics["contour_route_mode"] is True
    assert diagnostics["fallback_reasons"] == []
    assert all(
        decision["headland_turn_optimization"].get(
            "disabled_for_contour_shared_headland"
        )
        for decision in diagnostics["turn_decisions"]
    )
    # 偶数号转弯共用右田头线，奇数号转弯共用左田头线。
    assert max(point[0] for point in turns[0]) == pytest.approx(
        max(point[0] for point in turns[2])
    )
    assert min(point[0] for point in turns[1]) == pytest.approx(
        min(point[0] for point in turns[3])
    )
    assert "connector" not in segment_types


def test_contour_near_radius_limit_uses_shared_tangent_without_pear_loop():
    """DECISION-016: the 1.75 m centre turn stays in the common tangent family."""
    from path_planner import _connect_work_lines

    class _Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class _State:
        mask_raw = np.zeros((120, 120), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (110.0, 0.0), (110.0, 110.0), (0.0, 110.0),
            (0.0, 95.0), (10.25, 95.0), (10.25, 85.0), (0.0, 85.0),
        ]
        entry_point = None
        exit_point = None
        unload_points = []

    lines = [
        [(20.0, y), (80.0, y)]
        for y in (10.0, 27.5, 45.0, 62.5, 80.0, 97.5)
    ]
    path, segment_types, diagnostics = _connect_work_lines(
        lines,
        geo=_Geo(),
        state=_State(),
        min_turn_radius_m=0.9,
        turn_strategy="contour",
        config={
            "turn_outward_clearance_m": 0.1,
            "turn_safety_margin_m": 0.1,
            "turn_radius_geometry_tolerance_m": 0.05,
            "contour_max_endpoint_extension_m": 0.0,
        },
        global_main_angle=0.0,
        return_diagnostics=True,
    )

    turns = [
        segment for segment, kind in zip(path, segment_types)
        if kind == "turn"
    ]
    assert [item["strategy"] for item in diagnostics["turn_decisions"]] == [
        "bow", "bow", "bow", "bow", "semicircle"
    ]
    centre_decision = diagnostics["turn_decisions"][-1]
    assert centre_decision["radius_shortfall_m"] == pytest.approx(0.025)
    assert "shared-tangent near-semicircle" in centre_decision["fallback_reason"]
    # Right and left turns respectively touch one common straight headland
    # tangent, including the final near-semicircle on the right.
    assert [max(point[0] for point in turns[index]) for index in (0, 2, 4)] == (
        pytest.approx([90.0, 90.0, 90.0])
    )
    assert [min(point[0] for point in turns[index]) for index in (1, 3)] == (
        pytest.approx([10.5, 10.5])
    )
    assert diagnostics["contour_connector"][
        "shared_turn_clearance_m_by_side"
    ] == pytest.approx({"low": 0.05, "high": 0.1})
    assert all(len(turn) > 2 for turn in turns)


def test_contour_entry_may_swap_outer_sides_but_never_start_at_center():
    """入口靠右时逐层交换左右，不能整体反转为中心向外。"""
    from path_planner import _connect_work_lines

    class _Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class _State:
        mask_raw = np.zeros((100, 100), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)
        ]
        entry_point = (20.0, 95.0)
        exit_point = None
        unload_points = []

    path, segment_types, diagnostics = _connect_work_lines(
        [[(20.0, y), (80.0, y)] for y in (10.0, 30.0, 50.0, 70.0, 90.0)],
        geo=_Geo(),
        state=_State(),
        min_turn_radius_m=1.0,
        turn_strategy="contour",
        config={"turn_outward_clearance_m": 0.1, "turn_safety_margin_m": 0.1},
        global_main_angle=0.0,
        return_diagnostics=True,
    )

    work = [segment for segment, kind in zip(path, segment_types) if kind == "work"]
    assert [segment[0][1] for segment in work] == [90.0, 10.0, 70.0, 30.0, 50.0]
    assert diagnostics["contour_ordering"]["outside_side_reversed_for_entry"] is True
    assert diagnostics["contour_ordering"]["ordered_original_indices"] == [4, 0, 3, 1, 2]


def test_contour_is_a_valid_ordering_strategy_and_uses_bow_turn_geometry():
    """DECISION-016: 套圈用弓形横向切线，中心窄距仍受曲率回退。"""
    from path_planner import (
        _candidate_strategies_for_request,
        assess_turn_strategy,
        normalize_turn_strategy,
    )

    assert normalize_turn_strategy("spiral") == "contour"
    assert "contour" in _candidate_strategies_for_request("auto")
    assessment = assess_turn_strategy("contour", [2.0, 2.0], 1.0)
    assert assessment["strategy"] == "contour"
    assert assessment["turn_geometry_strategy"] == "bow"
    assert assessment["hard_reasons"] == []


def test_uniform_zones_are_split_into_swath_lines():
    """DECISION-016 v3: 大片等距种植区按割幅细分多条完整作业线。

    实测 5.27 田1 中 id=1 带剖面宽 12.45m（等距种植大片），一趟割幅收不完，
    必须按割幅间距生成多条平行作业线，而不是过滤或只提一条中心线。
    """
    from path_planner import _prepare_work_line_layout
    import numpy as np

    class _Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class _State:
        mask_offset_x = 0
        mask_offset_y = 0

    # 等距区：主方向 x 轴，长 20m、法向宽 12.45m（1245px @0.01m/px）
    contour = [(0.0, 0.0), (2000.0, 0.0), (2000.0, 1245.0), (0.0, 1245.0)]
    bands = [
        {
            "id": 1, "band_type": "wide",
            "profile_width_m": 12.45, "width_m": 3.89,
            "centerline": [(100.0, 600.0), (1900.0, 600.0)],
            "contour": contour,
        },
        {
            # 相邻 4 行带离等距区足够远（>2m），内限不压缩细分范围
            "id": 2, "band_type": "wide",
            "profile_width_m": 1.04, "width_m": 1.00,
            "centerline": [(100.0, 2500.0), (1900.0, 2500.0)],
            "contour": [(0.0, 2450.0), (2000.0, 2450.0), (2000.0, 2550.0), (0.0, 2550.0)],
        },
    ]
    lines, layout = _prepare_work_line_layout(
        bands,
        np.concatenate([np.full((1245, 2000), 255, dtype=np.uint8),
                        np.zeros((1355, 2000), dtype=np.uint8)]),
        0.0,
        geo=_Geo(),
        state=_State(),
        config={
            "work_line_mode": "band_centerline",
            "uniform_zone_line_spacing_m": 1.75,
        },
    )

    # 等距区 12.45m / 1.75m ≈ 7 条 + 常规带 1 条
    assert len(lines) == 8, f"期望 8 条作业线, 实际 {len(lines)}"
    zone_report = layout["uniform_zones"]
    assert zone_report[0]["id"] == 1
    assert zone_report[0]["line_count"] == 7, zone_report
    assert zone_report[0]["line_spacing_m"] == 1.75


def test_generate_uniform_zone_lines_spacing():
    """等距区细分线间距 = 割幅（1.75m），方向与主方向平行。"""
    from path_planner import _generate_uniform_zone_lines

    class _Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class _State:
        mask_offset_x = 0
        mask_offset_y = 0

    contour = [(0.0, 0.0), (2000.0, 0.0), (2000.0, 800.0), (0.0, 800.0)]
    zone = {"id": 9, "profile_width_m": 8.0, "contour": contour}
    white_mask = np.full((800, 2000), 255, dtype=np.uint8)
    lines = _generate_uniform_zone_lines(
        zone, 0.0, 1.75, geo=_Geo(), state=_State(), mask=white_mask
    )

    assert len(lines) >= 4, f"8m / 1.75m 应约 4-5 条, 实际 {len(lines)}"
    # 方向与主方向平行（水平线）：y 恒定
    for line in lines:
        ys = {p[1] for p in line}
        assert len(ys) == 1, f"作业线应平行主方向: {line}"
        xs = [point[0] for point in line]
        assert min(xs) <= 5.0
        assert max(xs) >= 1994.0
        assert max(xs) - min(xs) >= 1990.0


def test_turn_strategy_defaults_to_contour():
    """默认转弯策略必须是套圈收割（外圈向内）。"""
    from state import AppState

    state = AppState()
    assert state.turn_strategy == "contour"
