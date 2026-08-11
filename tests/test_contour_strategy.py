import numpy as np
import pytest


def test_contour_order_avoids_adjacent_centre_transition_for_four_lines():
    """DECISION-016: the small four-line case also avoids a centre pair."""
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
    assert diagnostics["algorithm_version"] == "contour-order-v5"
    assert diagnostics["pair_count"] == 2
    assert [item["original_indices"] for item in diagnostics["corresponding_pairs"]] == [[1, 3], [0, 2]]
    assert diagnostics["ordered_original_indices"] == [1, 3, 0, 2]
    assert diagnostics["adjacent_transition_count"] == 0
    assert ordered == [lines[index] for index in [1, 3, 0, 2]]


def test_contour_order_works_without_field_boundary():
    """DECISION-016 v3: ordering is geometric; connector construction owns the boundary gate."""
    from path_planner import _contour_order_work_lines

    lines = [[(0.0, 2.0), (10.0, 2.0)], [(0.0, 1.0), (10.0, 1.0)]]
    ordered, diagnostics = _contour_order_work_lines(lines, None)

    assert diagnostics["fallback_reason"] is None
    assert diagnostics["ordered_original_indices"] == [1, 0]
    assert ordered == [lines[1], lines[0]]


def test_contour_order_is_half_split_interleaved():
    """DECISION-016: 套圈按两半交错访问，避免最后两行相邻转弯。"""
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
    assert diagnostics["ordering"] == "contour_half_split_interleaved"
    assert diagnostics["algorithm_version"] == "contour-order-v5"
    assert diagnostics["ordered_original_indices"] == [1, 3, 0, 2]
    assert ordered == [lines[index] for index in [1, 3, 0, 2]]


def test_contour_order_interleaves_even_and_odd_line_counts():
    """偶数和奇数主作业线都按两半交错，不把中心两行硬接在一起。"""
    from path_planner import _contour_order_work_lines

    lines = [
        [(0.0, float(index)), (100.0, float(index))]
        for index in range(10)
    ]

    ordered, diagnostics = _contour_order_work_lines(
        lines, None, main_angle=0.0
    )

    expected = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    assert diagnostics["ordered_original_indices"] == expected
    assert diagnostics["pair_count"] == 5
    assert [
        item["original_indices"]
        for item in diagnostics["corresponding_pairs"]
    ] == [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
    assert ordered == [lines[index] for index in expected]

    odd_lines = [
        [(0.0, float(index)), (100.0, float(index))]
        for index in range(9)
    ]
    _odd_ordered, odd_diagnostics = _contour_order_work_lines(
        odd_lines, None, main_angle=0.0
    )
    assert odd_diagnostics["ordered_original_indices"] == [0, 5, 1, 6, 2, 7, 3, 8, 4]
    assert odd_diagnostics["corresponding_pairs"][-1]["original_indices"] == [4]


def test_entry_rotation_does_not_expose_an_adjacent_centre_seam():
    """DECISION-016 v11: start-row priority must preserve the final invariant."""
    from path_planner import (
        _contour_order_work_lines,
        _select_contour_entry_rotation,
    )

    lines = [
        [(0.0, float(index)), (100.0, float(index))]
        for index in range(6)
    ]
    ordered, diagnostics = _contour_order_work_lines(
        lines, None, main_angle=0.0
    )

    selected = _select_contour_entry_rotation(
        ordered,
        diagnostics,
        entry_point=(0.0, 4.0),
        policy="nearest",
    )

    assert selected[0] == lines[4]
    assert diagnostics["algorithm_version"] == "contour-order-v6"
    assert diagnostics["adjacent_transition_count"] == 0
    assert diagnostics["final_order_validated"] is True
    assert all(
        abs(right - left) != 1
        for left, right in zip(
            diagnostics["ordered_original_indices"],
            diagnostics["ordered_original_indices"][1:],
        )
    )


def test_far_side_handoff_rotation_is_safe_for_twelve_lines():
    """DECISION-017: a far-side retry is a safe rotation, not list reversal."""
    from path_planner import (
        _contour_order_work_lines,
        _select_contour_entry_rotation,
    )

    lines = [
        [(0.0, float(index)), (100.0, float(index))]
        for index in range(12)
    ]
    ordered, diagnostics = _contour_order_work_lines(
        lines, None, main_angle=0.0
    )

    selected = _select_contour_entry_rotation(
        ordered,
        diagnostics,
        entry_point=(0.0, 11.0),
        policy="farthest",
    )

    assert selected[0] == lines[0]
    assert diagnostics["normal_order_reversed"] is True
    assert diagnostics["adjacent_transition_count"] == 0
    assert diagnostics["minimum_possible_adjacent_transition_count"] == 0


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


def test_contour_connector_can_extend_parallel_lines_into_headland_space():
    """套圈的直线端点可按配置向两侧田头统一外延。"""
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
        lines,
        main_angle=0.0,
        field_polygon=polygon,
        mpp=0.1,
        config={
            "contour_max_endpoint_extension_m": 4.0,
            "work_line_endpoint_extension_m": 0.5,
        },
    )

    assert diagnostics["fallback_reason"] is None
    assert diagnostics["configured_work_line_endpoint_extension_m"] == pytest.approx(0.5)
    assert anchors[0]["start"][0] == pytest.approx(5.0)
    assert anchors[0]["end"][0] == pytest.approx(95.0)
    assert anchors[1]["start"][0] == pytest.approx(95.0)
    assert anchors[1]["end"][0] == pytest.approx(5.0)


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


def test_boundary_truncated_pass_finishes_route_without_diagonal_connector():
    """DECISION-016 v11: a blocked long shortfall is a terminal work endpoint."""
    from path_planner import _connect_work_lines

    class _Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 1.0

    class _State:
        mask_raw = np.ones((100, 100), dtype=np.uint8) * 255
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (20.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (20.0, 100.0),
            (20.0, 60.0),
            (40.0, 60.0),
            (40.0, 40.0),
            (20.0, 40.0),
        ]
        entry_point = (20.0, 10.0)
        exit_point = None
        unload_points = []

    short_endpoint = (40.0, 50.0)
    lines = [
        [(20.0, 10.0), (80.0, 10.0)],
        [(20.0, 20.0), (80.0, 20.0)],
        [(20.0, 30.0), (80.0, 30.0)],
        [short_endpoint, (80.0, 50.0)],
        [(20.0, 70.0), (80.0, 70.0)],
    ]

    path, segment_types, diagnostics = _connect_work_lines(
        lines,
        geo=_Geo(),
        state=_State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={
            "_contour_allow_terminal_short_line": True,
            "contour_max_endpoint_extension_m": 4.0,
            "work_line_endpoint_extension_m": 0.0,
        },
        global_main_angle=0.0,
        processed_mask=_State.mask_raw,
        return_diagnostics=True,
    )

    work_segments = [
        segment
        for segment, kind in zip(path, segment_types)
        if kind == "work"
    ]
    assert len(work_segments) == len(lines)
    assert segment_types[-1] == "work"
    assert work_segments[-1][-1] == pytest.approx(short_endpoint)
    ordering = diagnostics["contour_ordering"]
    assert ordering["terminal_constraint_applied"] is True
    assert ordering["adjacent_transition_count"] == 0
    assert diagnostics["contour_connector"]["terminal_unaligned_finish"] is True


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
    assert [segment[0][1] for segment in work] == [10.0, 70.0, 30.0, 90.0, 50.0]
    assert segment_types == ["work", "turn", "work", "turn", "work", "turn", "work", "turn", "work"]
    assert diagnostics["contour_connector"]["turn_distances_m"] == pytest.approx(
        [6.0, 4.0, 6.0, 4.0]
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


def test_contour_route_uses_two_endpoint_work_chords_after_pairing():
    """套圈候选先配对，再将每条主作业带固定为起终点直线。"""
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
            (0.0, 0.0), (110.0, 0.0), (110.0, 110.0), (0.0, 110.0)
        ]
        entry_point = None
        exit_point = None
        unload_points = []

    curved_lines = [
        [(20.0, y), (50.0, y + 2.0), (80.0, y)]
        for y in (10.0, 30.0, 50.0, 70.0, 90.0)
    ]
    path, segment_types, diagnostics = _connect_work_lines(
        curved_lines,
        geo=_Geo(),
        state=_State(),
        min_turn_radius_m=1.0,
        turn_strategy="contour",
        config={
            "turn_outward_clearance_m": 0.1,
            "turn_safety_margin_m": 0.1,
            "contour_max_endpoint_extension_m": 0.0,
        },
        global_main_angle=0.0,
        return_diagnostics=True,
    )

    work = [
        segment for segment, kind in zip(path, segment_types)
        if kind == "work"
    ]
    assert all(len(segment) == 2 for segment in work)
    assert diagnostics["contour_ordering"]["work_line_geometry"] == (
        "straight_endpoint_chord"
    )


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
        "bow", "bow", "bow", "bow", "bow"
    ]
    # The half-split order keeps every transition away from the adjacent
    # centre rows while preserving one common straight headland per side.
    assert [max(point[0] for point in turns[index]) for index in (0, 2, 4)] == (
        pytest.approx([90.0, 90.0, 90.0])
    )
    assert [min(point[0] for point in turns[index]) for index in (1, 3)] == (
        pytest.approx([10.0, 10.0])
    )
    assert diagnostics["contour_connector"][
        "shared_turn_clearance_m_by_side"
    ] == pytest.approx({"low": 0.1, "high": 0.1})
    assert all(len(turn) > 2 for turn in turns)


def test_contour_entry_starts_at_the_nearest_row_pair():
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
    assert work[0][0][1] == 90.0
    assert diagnostics["contour_ordering"]["entry_priority_applied"] is True
    assert diagnostics["contour_ordering"]["ordered_original_indices"][0] == 4


def test_band_centerline_endpoint_repair_restores_internal_truncated_rows():
    from path_planner import prepare_band_centerlines

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
        mask_raw = np.zeros((500, 1400), dtype=np.uint8)
        field_boundary = [
            (0.0, 0.0), (1400.0, 0.0),
            (1400.0, 500.0), (0.0, 500.0),
        ]

    bands = [
        {
            "id": index,
            "centerline": [(100.0, 80.0 + index * 80.0),
                            (1100.0, 80.0 + index * 80.0)],
            "longitudinal_extent_m": 10.0,
            "observed_span_ratio": 1.0,
        }
        for index in range(3)
    ]
    bands.append({
        "id": 9,
        "centerline": [(500.0, 320.0), (700.0, 320.0)],
        "longitudinal_extent_m": 10.0,
        "observed_span_ratio": 0.2,
    })

    lines, diagnostics = prepare_band_centerlines(
        bands,
        0.0,
        geo=_Geo(),
        state=_State(),
        config={
            "centerline_interval_m": 1.0,
            "centerline_endpoint_repair_enabled": True,
            "centerline_endpoint_repair_completion_ratio": 0.85,
            "centerline_endpoint_repair_min_cohort": 3,
            "centerline_endpoint_repair_max_m": 20.0,
            "centerline_min_span_ratio": 0.65,
        },
    )

    assert diagnostics["endpoint_repair"]["applied"] is True
    assert diagnostics["ordered_band_ids"] == [0, 1, 2, 9]
    repaired = lines[-1]
    assert repaired[0][0] == pytest.approx(100.0)
    assert repaired[-1][0] == pytest.approx(1100.0)


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

    # 等距区必须按 ceil(12.45 / 1.75)=8 条覆盖线分配，再加常规带 1 条。
    assert len(lines) == 9, f"期望 9 条作业线, 实际 {len(lines)}"
    zone_report = layout["uniform_zones"]
    assert zone_report[0]["id"] == 1
    assert zone_report[0]["line_count"] == 8, zone_report
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


def test_uniform_zone_keeps_a_pass_when_nominal_lateral_sample_hits_a_gap():
    """交界断点不能让一条本应存在的等距作业线整体消失。"""
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
    mask = np.full((800, 2000), 255, dtype=np.uint8)
    # One nominal pass centre is deliberately non-mask; the local support
    # search should move it a short distance while retaining the pass.
    mask[396:405, :] = 0
    lines = _generate_uniform_zone_lines(
        {"id": 10, "profile_width_m": 8.0, "contour": contour},
        0.0,
        1.75,
        geo=_Geo(),
        state=_State(),
        mask=mask,
    )

    assert len(lines) == 5


def test_uniform_zone_drops_an_outer_short_mask_fragment_before_contour_ordering():
    """边界短掩膜片不能制造跨田长转弯。"""
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
    mask = np.full((800, 2000), 255, dtype=np.uint8)
    # The first outer pass has only a short supported axial run.  The other
    # nominal passes retain the complete 20 m crop span.
    mask[45:56, 200:] = 0
    diagnostics = {}
    lines = _generate_uniform_zone_lines(
        {"id": 12, "profile_width_m": 8.0, "contour": contour},
        0.0,
        1.75,
        geo=_Geo(),
        state=_State(),
        mask=mask,
        diagnostics=diagnostics,
    )

    assert diagnostics["short_edge_line_filter_applied"] is True
    assert diagnostics["short_edge_line_indices"] == [0]
    assert len(lines) == 4
    positions_m = [float(np.mean([point[1] for point in line])) * 0.01 for line in lines]
    assert np.diff(positions_m) == pytest.approx([1.75] * 3)


def test_uniform_zone_global_phase_keeps_track_gauge_spacing_after_mask_gap():
    """掩膜补偿只能改变整块相位，不能逐条改变等距作业线间距。"""
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
    mask = np.full((800, 2000), 255, dtype=np.uint8)
    mask[396:405, :] = 0
    diagnostics = {}
    lines = _generate_uniform_zone_lines(
        {"id": 11, "profile_width_m": 8.0, "contour": contour},
        0.0,
        1.75,
        geo=_Geo(),
        state=_State(),
        mask=mask,
        diagnostics=diagnostics,
    )

    positions_m = [float(np.mean([point[1] for point in line])) * 0.01 for line in lines]
    assert len(positions_m) == 5
    assert np.diff(positions_m) == pytest.approx([1.75] * 4)
    assert diagnostics["spacing_is_uniform"] is True
    assert diagnostics["actual_spacing_m"] == pytest.approx(1.75)


def test_uniform_zone_bridges_internal_axial_break_instead_of_making_a_turn_endpoint():
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
    mask = np.zeros((800, 2000), dtype=np.uint8)
    mask[:, :800] = 255
    mask[:, 1200:1900] = 255
    diagnostics = {}
    lines = _generate_uniform_zone_lines(
        {"id": 13, "profile_width_m": 8.0, "contour": contour},
        0.0,
        1.75,
        geo=_Geo(),
        state=_State(),
        mask=mask,
        diagnostics=diagnostics,
    )

    assert lines
    assert diagnostics["internal_gap_bridge_enabled"] is True
    assert min(
        float(np.hypot(
            line[-1][0] - line[0][0], line[-1][1] - line[0][1]
        ))
        for line in lines
    ) > 1500.0


def test_uniform_zone_endpoint_repair_uses_shared_headland_walls():
    from path_planner import _repair_uniform_zone_endpoints

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
        mask_raw = np.zeros((500, 1400), dtype=np.uint8)
        field_boundary = [
            (0.0, 0.0), (1400.0, 0.0),
            (1400.0, 500.0), (0.0, 500.0),
        ]

    reference_lines = [
        [(100.0, 40.0 + index * 40.0),
         (1100.0, 40.0 + index * 40.0)]
        for index in range(4)
    ]
    short_uniform = [
        [(400.0, 260.0), (700.0, 260.0)],
        [(400.0, 320.0), (700.0, 320.0)],
    ]
    repaired, diagnostics = _repair_uniform_zone_endpoints(
        short_uniform,
        {"id": 8, "longitudinal_extent_m": 10.0},
        reference_lines,
        0.0,
        geo=_Geo(),
        state=_State(),
        mask=_State.mask_raw,
        config={"centerline_endpoint_repair_max_m": 20.0},
    )

    assert diagnostics["applied"] is True
    assert all(len(line) == 2 for line in repaired)
    assert all(line[0][0] == pytest.approx(100.0) for line in repaired)
    assert all(line[-1][0] == pytest.approx(1100.0) for line in repaired)


def test_uniform_zone_default_spacing_aligns_adjacent_track_centrelines():
    """未指定等距间隔时，默认用轨距而非割幅，保证左右履带重合。"""
    from path_planner import _uniform_zone_spacing

    class _State:
        harvester_params = {"track_gauge_m": 1.3}

    spacing, basis, diagnostics = _uniform_zone_spacing({}, _State())

    assert spacing == pytest.approx(1.3)
    assert basis == "track_gauge_overlap"
    assert diagnostics["track_gauge_m"] == pytest.approx(1.3)
    assert diagnostics["adjacent_track_centerline_offset_m"] == pytest.approx(0.0)


def test_turn_strategy_defaults_to_contour():
    """默认转弯策略必须是套圈收割（外圈向内）。"""
    from state import AppState

    state = AppState()
    assert state.turn_strategy == "contour"
