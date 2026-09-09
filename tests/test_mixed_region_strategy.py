import numpy as np
import pytest


def _band(band_id, y, profile_width_m, width_m=None, height=20.0):
    half = max(0.5, height / 2.0)
    return {
        "id": band_id,
        "profile_width_m": profile_width_m,
        "width_m": profile_width_m if width_m is None else width_m,
        "centerline": [(10.0, y), (90.0, y)],
        "contour": [
            (0.0, y - half),
            (100.0, y - half),
            (100.0, y + half),
            (0.0, y + half),
        ],
    }


def test_selection_filter_keeps_only_four_complete_work_bands():
    """Small selections use context for analysis but not edge work passes."""
    from path_planner import _filter_work_bands_to_selection

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        harvester_params = {"cutter_width_m": 2.15}

    bands = [
        _band(2, 8.0, 1.02, 1.01),
        _band(4, 25.0, 1.00, 1.00),
        _band(6, 43.0, 0.96, 0.98),
        _band(8, 61.0, 0.98, 1.00),
        _band(10, 79.0, 0.98, 1.00),
        _band(11, 88.0, 2.20, 1.67),
    ]

    selected, report = _filter_work_bands_to_selection(
        bands,
        np.zeros((100, 100), dtype=np.uint8),
        main_angle=0.0,
        geo=Geo(),
        state=State(),
        config={"uniform_zone_min_width_m": 4.0},
    )

    assert [band["id"] for band in selected] == [4, 6, 8, 10]
    excluded = {item["band_id"]: item["reasons"] for item in report["excluded"]}
    assert "insufficient_cutter_normal_clearance" in excluded[2]
    assert "edge_width_outlier_below_uniform_threshold" in excluded[11]

    lines, layout, regions = __import__("path_planner")._prepare_work_line_layout(
        bands,
        np.zeros((100, 100), dtype=np.uint8),
        0.0,
        geo=Geo(),
        state=State(),
        config={"uniform_zone_min_width_m": 4.0},
        return_regions=True,
    )
    assert len(lines) == 4
    assert layout["selection_work_band_filter"]["kept_band_ids"] == [4, 6, 8, 10]
    assert [item["band_ids"] for item in regions] == [[4, 6, 8, 10]]


def test_uniform_reference_geometry_ignores_malformed_cache_values():
    """DECISION-009/017: malformed compact geometry must fail locally, not crash planning."""
    from path_planner import _uniform_zone_reference_geometry

    points, centroid, source = _uniform_zone_reference_geometry(
        {
            "contour": "not-a-point-array",
            "planning_contour": [1.0, 2.0, 3.0],
            "centerline": [(0.0, 1.0), (10.0, 1.0)],
            "centroid": "bad-centroid",
        },
        0.0,
    )

    assert source == "compact_centerline_and_mask"
    assert points.shape == (2, 2)
    assert centroid.tolist() == [5.0, 1.0]

    points, centroid, source = _uniform_zone_reference_geometry(
        {"contour": "bad", "centerline": "also-bad", "bbox": "bad"},
        0.0,
    )
    assert points is None
    assert centroid is None
    assert source == "missing_reference_geometry"


def test_region_classifier_uses_width_evidence_not_left_or_right():
    """DECISION-017: 镜像田块不能改变种植类型判定。"""
    from path_planner import _classify_planting_regions

    original = [
        _band(1, -42.0, 12.451, 3.894),
        _band(2, -38.0, 1.04, 1.00),
        _band(4, -36.0, 0.94, 0.91),
    ]
    mirrored = [
        {**item, "centerline": [(x, -y) for x, y in item["centerline"]],
         "contour": [(x, -y) for x, y in item["contour"]]}
        for item in original
    ]

    groups, report = _classify_planting_regions(original, 0.0)
    mirrored_groups, mirrored_report = _classify_planting_regions(mirrored, 0.0)

    assert report["counts"] == {
        "wide_narrow_rows": 2,
        "uniform_block": 1,
        "ambiguous": 0,
    }
    assert report["mixed_field"] is True
    assert report["automatic_planning_allowed"] is True
    assert [item["kind"] for item in groups] == [
        "wide_narrow_rows", "uniform_block"
    ]
    assert [item["classification"] for item in report["records"]] == [
        item["classification"] for item in mirrored_report["records"]
    ]
    assert [item["kind"] for item in mirrored_groups] == [
        "wide_narrow_rows", "uniform_block"
    ]


def test_region_classifier_fails_closed_inside_ambiguity_guard_band():
    """DECISION-017: 阈值附近证据不确定时不得猜测种植策略。"""
    from path_planner import _classify_planting_regions

    groups, report = _classify_planting_regions(
        [_band(9, 20.0, 2.50, 2.40)],
        0.0,
        config={
            "uniform_zone_min_width_m": 2.5,
            "planting_region_ambiguity_margin_m": 0.25,
        },
    )

    assert groups == []
    assert report["counts"]["ambiguous"] == 1
    assert report["automatic_planning_allowed"] is False
    assert report["hard_reasons"]


def test_region_classifier_resolves_borderline_profile_with_complete_wide_evidence():
    """真实缓存中的边界宽带不能把整块田误判成零作业线。"""
    from path_planner import _classify_planting_regions

    band = _band(7, 20.0, 2.449, 1.775, height=80.0)
    band.update({
        "band_type": "wide",
        "observed_span_ratio": 0.987,
        "longitudinal_extent_ratio": 0.987,
    })

    groups, report = _classify_planting_regions(
        [band],
        0.0,
        config={
            "uniform_zone_min_width_m": 2.5,
            "planting_region_ambiguity_margin_m": 0.25,
        },
    )

    assert [item["kind"] for item in groups] == ["wide_narrow_rows"]
    assert report["counts"] == {
        "wide_narrow_rows": 1,
        "uniform_block": 0,
        "ambiguous": 0,
    }
    assert report["borderline_resolved_count"] == 1
    assert report["records"][0]["reason"] == (
        "borderline_profile_resolved_by_complete_wide_band_evidence"
    )
    assert report["automatic_planning_allowed"] is True


def test_default_region_classifier_accepts_complete_25m_four_row_bands():
    """DECISION-017: the real default must not reject complete four-row bands."""
    from path_planner import _classify_planting_regions

    bands = []
    for band_id, y, profile_width_m, width_m in (
        (6, -30.0, 2.5566423, 2.5695059),
        (7, -10.0, 12.4926840, 8.0463540),
        (17, 10.0, 2.6147478, 2.5063089),
    ):
        band = _band(band_id, y, profile_width_m, width_m, height=80.0)
        band.update({
            "band_type": "wide",
            "observed_span_ratio": 1.0,
            "longitudinal_extent_ratio": 1.0,
        })
        bands.append(band)

    groups, report = _classify_planting_regions(bands, 0.0)

    assert [item["kind"] for item in groups] == [
        "wide_narrow_rows", "wide_narrow_rows", "uniform_block"
    ]
    assert [item["band_ids"] for item in groups] == [[6], [17], [7]]
    assert report["counts"] == {
        "wide_narrow_rows": 2,
        "uniform_block": 1,
        "ambiguous": 0,
    }
    assert report["automatic_planning_allowed"] is True


def test_region_classifier_splits_wide_rows_separated_by_uniform_block():
    """DECISION-017: 等距区两侧宽窄行不得合并后跨田配圈。"""
    from path_planner import _classify_planting_regions

    bands = [
        _band(1, -35.64, 2.5, 2.4),
        _band(2, -34.96, 2.5, 2.4),
        _band(3, -34.28, 2.5, 2.4),
        _band(6, -29.35, 12.4, 8.0, height=120.0),
        _band(7, -24.50, 2.5, 2.4),
        _band(8, -23.27, 2.5, 2.4),
        _band(9, -22.12, 2.5, 2.4),
    ]
    for band in bands:
        band.update({
            "band_type": "wide",
            "observed_span_ratio": 1.0,
            "longitudinal_extent_ratio": 1.0,
        })

    groups, report = _classify_planting_regions(bands, 0.0)

    assert [item["kind"] for item in groups] == [
        "wide_narrow_rows", "wide_narrow_rows", "uniform_block"
    ]
    assert [item["band_ids"] for item in groups] == [
        [1, 2, 3], [7, 8, 9], [6]
    ]
    assert report["wide_narrow_region_count"] == 2
    assert report["partition_basis"] == (
        "uniform_block_normal_position_separates_regular_rows"
    )


def test_prepare_layout_keeps_classified_region_work_lines_separate():
    """DECISION-017: 宽窄行中心线和等距区割幅线形成两个独立工作区。"""
    from path_planner import _prepare_work_line_layout

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class State:
        mask_offset_x = 0
        mask_offset_y = 0

    uniform = {
        "id": 1,
        "profile_width_m": 12.45,
        "width_m": 3.89,
        "centerline": [(100.0, 600.0), (1900.0, 600.0)],
        "contour": [
            (0.0, 0.0), (2000.0, 0.0),
            (2000.0, 1245.0), (0.0, 1245.0),
        ],
    }
    regular = {
        "id": 2,
        "profile_width_m": 1.04,
        "width_m": 1.00,
        "centerline": [(100.0, 2500.0), (1900.0, 2500.0)],
        "contour": [
            (0.0, 2450.0), (2000.0, 2450.0),
            (2000.0, 2550.0), (0.0, 2550.0),
        ],
    }
    mask = np.concatenate([
        np.full((1245, 2000), 255, dtype=np.uint8),
        np.zeros((1355, 2000), dtype=np.uint8),
    ])

    lines, layout, regions = _prepare_work_line_layout(
        [uniform, regular],
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "work_line_mode": "band_centerline",
            "uniform_zone_line_spacing_m": 1.75,
        },
        return_regions=True,
    )

    assert len(lines) == 9
    assert [item["kind"] for item in regions] == [
        "wide_narrow_rows", "uniform_block"
    ]
    assert [item["line_count"] for item in regions] == [1, 8]
    assert [item["line_source"] for item in regions] == [
        "four_row_band_centerlines", "cutter_width_uniform_swaths"
    ]
    regular_line = regions[0]["work_lines"][0]
    assert regular_line[0] == pytest.approx(regular["centerline"][0])
    assert regular_line[-1] == pytest.approx(regular["centerline"][-1])
    assert {point[1] for point in regular_line} == {2500.0}
    assert "work_lines" not in layout["work_regions"][0]
    assert layout["planting_region_classification"]["mixed_field"] is True


def test_uniform_only_field_uses_swaths_without_four_row_centerline_mode():
    """DECISION-017: 纯等距大片也必须先分类并按割幅正常生成路径线。"""
    from path_planner import _prepare_work_line_layout

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_offset_x = 0
        mask_offset_y = 0

    zone = {
        "id": 1,
        "profile_width_m": 8.0,
        "width_m": 8.0,
        "centerline": [(0.0, 40.0), (100.0, 40.0)],
        "contour": [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)],
    }
    lines, layout, regions = _prepare_work_line_layout(
        [zone],
        np.full((81, 101), 255, dtype=np.uint8),
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "work_line_mode": "footprint_optimized",
            "uniform_zone_line_spacing_m": 2.0,
        },
        return_regions=True,
    )

    assert len(lines) == 4
    assert len(regions) == 1
    assert regions[0]["kind"] == "uniform_block"
    assert regions[0]["line_source"] == "cutter_width_uniform_swaths"
    assert layout["work_line_mode"] == "uniform_block_cutter_width_swaths"


def test_compact_uniform_zone_uses_centerline_and_mask_without_dense_contour():
    """DECISION-017: UI/cache 删除稠密轮廓后，等距区仍能生成割幅线。"""
    from path_planner import _prepare_work_line_layout

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        mask_raw = None

    uniform = {
        "id": 1,
        "profile_width_m": 12.45,
        "width_m": 3.89,
        "centroid": (1000.0, 622.5),
        "bbox": (0.0, 0.0, 2000.0, 1245.0),
        "centerline": [(100.0, 622.5), (1900.0, 622.5)],
        # Deliberately no contour: this is the compact UI/cache contract.
    }
    regular = {
        "id": 2,
        "profile_width_m": 1.04,
        "width_m": 1.00,
        "centerline": [(100.0, 2500.0), (1900.0, 2500.0)],
        "contour": [
            (0.0, 2450.0), (2000.0, 2450.0),
            (2000.0, 2550.0), (0.0, 2550.0),
        ],
    }
    mask = np.concatenate([
        np.full((1245, 2000), 255, dtype=np.uint8),
        np.zeros((1355, 2000), dtype=np.uint8),
    ])

    lines, layout, regions = _prepare_work_line_layout(
        [uniform, regular],
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "work_line_mode": "band_centerline",
            "uniform_zone_line_spacing_m": 1.75,
        },
        return_regions=True,
    )

    assert len(lines) == 9
    assert [region["line_count"] for region in regions] == [1, 8]
    assert layout["uniform_zones"][0]["reference_geometry_source"] == (
        "compact_centerline_and_mask"
    )


def test_uniform_zone_expands_narrow_planning_contour_from_profile_width():
    """DECISION-009/017: measured block width must not be clipped by a narrow contour."""
    from path_planner import _prepare_work_line_layout

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        mask_raw = None

    uniform = {
        "id": 1,
        "profile_width_m": 12.45,
        "width_m": 3.89,
        "centerline": [(100.0, 622.5), (1900.0, 622.5)],
        # This reproduces the cache contract: the planning contour is only
        # the narrow semantic support stripe, while profile_width_m covers
        # the complete equal-distance block.
        "contour": [
            (0.0, 428.0), (2000.0, 428.0),
            (2000.0, 817.0), (0.0, 817.0),
        ],
    }
    regular = {
        "id": 2,
        "profile_width_m": 1.04,
        "width_m": 1.00,
        "centerline": [(100.0, 1800.0), (1900.0, 1800.0)],
        "contour": [
            (0.0, 1750.0), (2000.0, 1750.0),
            (2000.0, 1850.0), (0.0, 1850.0),
        ],
    }
    mask = np.full((2000, 2000), 255, dtype=np.uint8)

    lines, layout, regions = _prepare_work_line_layout(
        [uniform, regular],
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "work_line_mode": "band_centerline",
            "uniform_zone_line_spacing_m": 1.75,
        },
        return_regions=True,
    )

    assert [region["line_count"] for region in regions] == [1, 8]
    assert len(lines) == 9
    assert layout["uniform_zones"][0]["effective_geometry_source"] == (
        "contour_expanded_from_profile_width"
    )
    # The measured profile expands the equal-distance block, but the nearest
    # regular centreline still owns the shared boundary swath.  The guard is
    # therefore retained to prevent a cross-region duplicate pass.
    assert layout["uniform_zones"][0]["inner_limit_applied"] is True
    assert layout["uniform_zones"][0]["effective_transverse_width_m"] == pytest.approx(
        12.45
    )


def test_uniform_zone_does_not_duplicate_adjacent_regular_boundary_line():
    """DECISION-017: 相邻区域共享边界作业带，不得重复生成等距线。"""
    from path_planner import _prepare_work_line_layout

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        mask_raw = None

    uniform = {
        "id": 1,
        "profile_width_m": 12.45,
        "width_m": 3.89,
        "centerline": [(100.0, 622.5), (1900.0, 622.5)],
        "contour": [
            (0.0, 428.0), (2000.0, 428.0),
            (2000.0, 817.0), (0.0, 817.0),
        ],
    }
    regular = {
        "id": 2,
        "profile_width_m": 1.04,
        "width_m": 1.00,
        "centerline": [(100.0, 1000.0), (1900.0, 1000.0)],
        "contour": [
            (0.0, 950.0), (2000.0, 950.0),
            (2000.0, 1050.0), (0.0, 1050.0),
        ],
    }
    lines, _layout, regions = _prepare_work_line_layout(
        [uniform, regular],
        np.full((2000, 2000), 255, dtype=np.uint8),
        0.0,
        geo=Geo(),
        state=State(),
        config={
            "work_line_mode": "band_centerline",
            "uniform_zone_line_spacing_m": 1.75,
        },
        return_regions=True,
    )

    uniform_lines = regions[1]["work_lines"]
    regular_y = 1000.0
    uniform_ys = [float(np.mean([point[1] for point in line])) for line in uniform_lines]
    assert uniform_lines
    assert max(uniform_ys) <= regular_y - 100.0 + 1e-6
    assert len(lines) == len(uniform_lines) + 1


def test_ambiguous_classification_is_a_plan_path_hard_reason():
    """DECISION-017: 不确定类型在最终规划协议中也必须失败关闭。"""
    from path_planner import plan_path

    class State:
        harvester_params = {
            "cutter_width_m": 1.75,
            "cutter_forward_offset_m": 1.35,
            "track_width_m": 0.6,
            "track_gauge_m": 1.0,
            "wheelbase_m": 1.8,
            "track_length_m": 1.7,
            "turn_radius_m": 0.9,
        }
        turn_strategy = "contour"

    result = plan_path(
        [_band(9, 20.0, 2.50, 2.40)],
        np.full((100, 100), 255, dtype=np.uint8),
        0.0,
        state=State(),
        config={
            "uniform_zone_min_width_m": 2.5,
            "planting_region_ambiguity_margin_m": 0.25,
        },
    )

    assert result["is_valid"] is False
    assert result["full_path"] == []
    assert result["turn_assessment"]["needs_confirmation"] is True
    assert any(
        "classification is ambiguous" in reason
        for reason in result["turn_assessment"]["hard_reasons"]
    )


def test_mixed_region_lines_are_all_generated_before_one_global_contour_route():
    """DECISION-017: 分区只生成直线，套圈排序和转弯必须全田统一。"""
    from path_planner import _connect_work_regions

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.zeros((100, 100), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)
        ]
        entry_point = None
        exit_point = None
        unload_points = []

    regions = [
        {
            "region_id": "uniform_block_1",
            "kind": "uniform_block",
            "normal_position": 65.0,
            "line_count": 2,
            "work_lines": [
                [(20.0, 60.0), (80.0, 60.0)],
                [(20.0, 70.0), (80.0, 70.0)],
            ],
        },
        {
            "region_id": "wide_narrow_rows_1",
            "kind": "wide_narrow_rows",
            "normal_position": 15.0,
            "line_count": 2,
            "work_lines": [
                [(20.0, 10.0), (80.0, 10.0)],
                [(20.0, 20.0), (80.0, 20.0)],
            ],
        },
    ]

    path, segment_types, assessment, ordered = _connect_work_regions(
        regions,
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={
            "turn_outward_clearance_m": 0.1,
            "turn_safety_margin_m": 0.1,
            "contour_max_endpoint_extension_m": 0.0,
        },
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    assert [item["region_id"] for item in ordered] == [
        "uniform_block_1", "wide_narrow_rows_1"
    ]
    assert segment_types.count("turn_aux") == 0
    assert segment_types.count("turn_reverse") == 0
    assert "connector" not in segment_types
    work_y = [
        segment[0][1]
        for segment, segment_type in zip(path, segment_types)
        if segment_type == "work"
    ]
    assert work_y == [20.0, 70.0, 10.0, 60.0]
    ordering = assessment["contour_ordering"]
    assert ordering["execution_scope"] == "all_generated_work_lines"
    assert ordering["region_boundaries_affect_turns"] is False
    assert ordering["global_line_count"] == 4
    assert ordering["source_region_ids"] == [
        "uniform_block_1", "wide_narrow_rows_1"
    ]
    assert len(assessment["region_routes"]) == 0
    assert len(assessment["region_transfers"]) == 0
    assert assessment["machine_executable"] is True


def test_mixed_region_boundary_does_not_create_a_special_handoff():
    """DECISION-016/017: all lines use one global contour topology."""
    from path_planner import _connect_work_regions

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.zeros((220, 220), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (220.0, 0.0), (220.0, 220.0), (0.0, 220.0)
        ]
        entry_point = None
        exit_point = None
        unload_points = []

    regular_lines = [
        [(20.0, float(row)), (200.0, float(row))]
        for row in range(10, 90, 10)
    ]
    uniform_lines = [
        [(20.0, float(row)), (200.0, float(row))]
        for row in range(90, 170, 10)
    ]
    regions = [
        {
            "region_id": "wide_narrow_rows_1",
            "kind": "wide_narrow_rows",
            "normal_position": 45.0,
            "line_count": len(regular_lines),
            "work_lines": regular_lines,
        },
        {
            "region_id": "uniform_block_1",
            "kind": "uniform_block",
            "normal_position": 125.0,
            "line_count": len(uniform_lines),
            "work_lines": uniform_lines,
        },
    ]

    path, segment_types, assessment, _ordered = _connect_work_regions(
        regions,
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={"contour_max_endpoint_extension_m": 0.0},
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    ordering = assessment["contour_ordering"]
    assert ordering["execution_scope"] == "all_generated_work_lines"
    assert ordering["global_line_count"] == 16
    assert ordering["adjacent_transition_count"] == 0
    assert assessment["region_routes"] == []
    assert assessment["region_transfers"] == []
    assert segment_types.count("turn_aux") == 0
    assert segment_types.count("turn_reverse") == 0
    assert len(path) == len(segment_types)


def test_terminal_short_line_does_not_trigger_a_region_retrace():
    """DECISION-016/017: a region boundary cannot create a full-pass retrace."""
    from path_planner import _connect_work_regions

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.ones((220, 120), dtype=np.uint8) * 255
        mask_offset_x = 0
        mask_offset_y = 0
        # The right-headland notch blocks extension of the short line at
        # y=130.  The line remains final, but its blocked endpoint must not
        # use route parity to override the confirmed entry.
        field_boundary = [
            (0.0, 0.0),
            (110.0, 0.0),
            (110.0, 120.0),
            (60.0, 120.0),
            (60.0, 140.0),
            (110.0, 140.0),
            (110.0, 210.0),
            (0.0, 210.0),
        ]
        entry_point = (20.0, 10.0)
        exit_point = None
        unload_points = []
        harvester_params = {
            "track_gauge_m": 1.3,
            "cutter_width_m": 2.0,
        }

    regions = [
        {
            "region_id": "wide_narrow_rows_1",
            "kind": "wide_narrow_rows",
            "normal_position": 35.0,
            "line_count": 6,
            "work_lines": [
                # Upstream centreline endpoint order is not an execution
                # direction guarantee.  Use the opposite axial order to
                # exercise normalization before entry-side selection.
                [(80.0, float(row)), (20.0, float(row))]
                for row in (10, 20, 30, 40, 50, 60)
            ],
        },
        {
            "region_id": "uniform_block_1",
            "kind": "uniform_block",
            "normal_position": 125.0,
            "line_count": 4,
            "work_lines": [
                [
                    (60.0 if row == 130 else 80.0, float(row)),
                    (20.0, float(row)),
                ]
                for row in (100, 110, 130, 150)
            ],
        },
    ]

    path, segment_types, assessment, _ordered = _connect_work_regions(
        regions,
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={
            "contour_max_endpoint_extension_m": 0.4,
            "work_line_endpoint_extension_m": 0.0,
            "turn_outward_clearance_m": 0.1,
            "turn_safety_margin_m": 0.1,
        },
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    assert segment_types.count("work") == 10
    assert segment_types.count("turn_reverse") == 0
    assert segment_types.count("turn_aux") == 0
    assert assessment["region_routes"] == []
    assert assessment["region_transfers"] == []
    assert assessment["contour_ordering"]["global_line_count"] == 10
    work_segments = [
        segment
        for segment, segment_type in zip(path, segment_types)
        if segment_type == "work"
    ]
    # DECISION-016/017: reserving the blocked short line as the final pass may
    # only affect its local incoming turn.  It must never use even/odd pass
    # parity to reverse the whole route away from the confirmed entry.
    assert work_segments[0][0] == pytest.approx((20.0, 10.0))
    contour = assessment["contour_ordering"]
    assert contour["entry_priority_applied"] is True
    assert contour["entry_orientation_constrained_by_terminal"] is False
    assert contour["entry_selected_endpoint_local"] == pytest.approx([20.0, 10.0])
    assert contour["entry_selected_line_distance_px"] == pytest.approx(0.0)
    assert contour["entry_nearest_feasible_verified"] is True
    assert contour["entry_side_constrained_by_terminal_short_line"] is False
    assert contour[
        "entry_unconstrained_endpoint_blocked_by_terminal_constraint"
    ] is False
    assert contour["terminal_blocked_endpoint_role"] == "start"
    assert contour["terminal_finish_orientation_forced"] is False
    assert work_segments[-1][0][1] == pytest.approx(130.0)
    assert work_segments[-1][-1][1] == pytest.approx(130.0)
    assert assessment["contour_ordering"]["terminal_constraint_applied"] is True
    assert len(path) == len(segment_types)


def test_three_line_region_enters_the_global_line_set_without_execution_merge():
    """DECISION-016/017: source regions remain provenance, not route groups."""
    from path_planner import _connect_work_regions

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.ones((260, 260), dtype=np.uint8) * 255
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (240.0, 0.0), (240.0, 240.0), (0.0, 240.0)
        ]
        entry_point = (20.0, 20.0)
        exit_point = None
        unload_points = []

    large_lines = [
        [(20.0, float(row)), (220.0, float(row))]
        for row in range(20, 140, 10)
    ]
    small_lines = [
        [(20.0, float(row)), (220.0, float(row))]
        for row in (160, 170, 180)
    ]
    regions = [
        {
            "region_id": "uniform_block_1",
            "kind": "uniform_block",
            "normal_position": 75.0,
            "line_count": len(large_lines),
            "work_lines": large_lines,
        },
        {
            "region_id": "wide_narrow_rows_1",
            "kind": "wide_narrow_rows",
            "normal_position": 170.0,
            "line_count": len(small_lines),
            "work_lines": small_lines,
        },
    ]

    path, segment_types, assessment, ordered = _connect_work_regions(
        regions,
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={"contour_max_endpoint_extension_m": 0.0},
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    assert len(ordered) == 2
    assert [item["region_id"] for item in ordered] == [
        "uniform_block_1", "wide_narrow_rows_1"
    ]
    assert segment_types.count("work") == 15
    ordering = assessment["contour_ordering"]
    assert ordering["global_line_count"] == 15
    assert ordering["adjacent_transition_count"] == 0
    assert ordering["minimum_possible_adjacent_transition_count"] == 0
    assert assessment["region_transfers"] == []
    assert len(path) == len(segment_types)


def test_contour_region_transfer_never_falls_back_to_a_pear_bulb():
    """窄交界仍使用套圈弓形转场，不生成灯泡/梨形回退。"""
    from path_planner import _build_inter_region_transfer

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.zeros((100, 100), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)
        ]

    transfer, report = _build_inter_region_transfer(
        [[(20.0, 10.0), (80.0, 10.0)]],
        ["work"],
        [[(80.0, 15.0), (20.0, 15.0)]],
        ["work"],
        geo=Geo(),
        state=State(),
        min_turn_radius_m=1.0,
        requested_strategy="contour",
        config={"turn_outward_clearance_m": 0.1},
    )

    assert report["strategy"] == "bow"
    assert report["requested_strategy"] == "contour"
    assert report["radius_capped_for_contour_pairing"] is True
    assert report["valid"] is True
    assert len(transfer) > 2


def test_entry_selects_the_nearest_line_in_the_global_contour_set():
    """DECISION-017: entry selection has no region-execution stage."""
    from path_planner import _connect_work_regions

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.zeros((100, 100), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)
        ]
        entry_point = (90.0, 70.0)
        exit_point = None
        unload_points = []

    regions = [
        {
            "region_id": "wide_narrow_rows_1",
            "kind": "wide_narrow_rows",
            "normal_position": 15.0,
            "line_count": 2,
            "work_lines": [
                [(20.0, 10.0), (80.0, 10.0)],
                [(20.0, 20.0), (80.0, 20.0)],
            ],
        },
        {
            "region_id": "uniform_block_1",
            "kind": "uniform_block",
            "normal_position": 65.0,
            "line_count": 2,
            "work_lines": [
                [(20.0, 60.0), (80.0, 60.0)],
                [(20.0, 70.0), (80.0, 70.0)],
            ],
        },
    ]

    path, segment_types, assessment, ordered = _connect_work_regions(
        regions,
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={
            "turn_outward_clearance_m": 0.1,
            "turn_safety_margin_m": 0.1,
            "contour_max_endpoint_extension_m": 0.0,
        },
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    assert [item["kind"] for item in ordered] == [
        "wide_narrow_rows", "uniform_block"
    ]
    first_work = next(
        segment for segment, kind in zip(path, segment_types) if kind == "work"
    )
    assert first_work[0] == pytest.approx((80.0, 70.0))
    contour = assessment["contour_ordering"]
    assert contour["entry_priority_applied"] is True
    assert contour["entry_nearest_feasible_verified"] is True
    assert contour["entry_side_constrained_by_terminal_short_line"] is False
    assert contour["entry_selected_line_distance_px"] == pytest.approx(10.0)
    assert assessment["contour_ordering"]["execution_scope"] == (
        "all_generated_work_lines"
    )
    assert assessment["region_transfers"] == []


def test_three_regions_use_entry_only_for_the_global_line_order():
    """DECISION-017: 起点选择全田直线，不重排来源区域。"""
    from path_planner import _connect_work_regions

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.ones((220, 220), dtype=np.uint8) * 255
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (220.0, 0.0), (220.0, 220.0), (0.0, 220.0)
        ]
        # The user explicitly starts beside the middle block.  The first work
        # endpoint must therefore come from that block rather than an edge.
        entry_point = (200.0, 110.0)
        exit_point = None
        unload_points = []

    regions = []
    for region_id, kind, normal in (
        ("wide_narrow_rows_1", "wide_narrow_rows", 30.0),
        ("uniform_block_1", "uniform_block", 110.0),
        ("wide_narrow_rows_2", "wide_narrow_rows", 190.0),
    ):
        regions.append({
            "region_id": region_id,
            "kind": kind,
            "normal_position": normal,
            "line_count": 2,
            "work_lines": [
                [(20.0, normal - 5.0), (200.0, normal - 5.0)],
                [(20.0, normal + 5.0), (200.0, normal + 5.0)],
            ],
        })

    _path, _types, assessment, ordered = _connect_work_regions(
        regions,
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={"contour_max_endpoint_extension_m": 0.0},
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    assert [item["normal_position"] for item in ordered] == [30.0, 110.0, 190.0]
    assert assessment["contour_ordering"]["execution_scope"] == (
        "all_generated_work_lines"
    )
    first_work = next(
        segment for segment, kind in zip(_path, _types) if kind == "work"
    )
    assert first_work[0] == pytest.approx((200.0, 115.0))
    assert assessment["region_transfers"] == []


def test_mixed_regions_share_the_same_two_contour_headland_walls():
    """DECISION-016/017: region splitting must not split the turn lanes."""
    from path_planner import _connect_work_regions

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.ones((100, 100), dtype=np.uint8) * 255
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)
        ]
        entry_point = None
        exit_point = None
        unload_points = []

    regions = [
        {
            "region_id": "wide_narrow_rows_1",
            "kind": "wide_narrow_rows",
            "normal_position": 20.0,
            "line_count": 2,
            "work_lines": [
                [(20.0, 15.0), (80.0, 15.0)],
                [(20.0, 25.0), (80.0, 25.0)],
            ],
        },
        {
            "region_id": "uniform_block_1",
            "kind": "uniform_block",
            "normal_position": 70.0,
            "line_count": 2,
            "work_lines": [
                [(25.0, 65.0), (75.0, 65.0)],
                [(25.0, 75.0), (75.0, 75.0)],
            ],
        },
    ]

    _path, _types, assessment, _ordered = _connect_work_regions(
        regions,
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={"contour_max_endpoint_extension_m": 2.0},
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    connector = assessment["contour_connector"]
    assert connector["headland_wall_projections_px"] == {
        "low": 20.0, "high": 80.0
    }
    assert assessment["contour_ordering"]["global_line_count"] == 4
    assert assessment["region_transfers"] == []


def test_mixed_regions_share_most_restrictive_safe_turn_clearance():
    """DECISION-016: a boundary notch moves the field-wide tangent once."""
    from path_planner import _connect_work_regions

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.ones((110, 100), dtype=np.uint8) * 255
        mask_offset_x = 0
        mask_offset_y = 0
        # Only the first region has a low-headland indentation.  Both regions
        # must nevertheless use the same reduced low-side clearance.
        field_boundary = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 110.0),
            (0.0, 110.0),
            (0.0, 40.0),
            (15.0, 40.0),
            (15.0, 20.0),
            (0.0, 20.0),
        ]
        entry_point = None
        exit_point = None
        unload_points = []

    regions = [
        {
            "region_id": "wide_narrow_rows_1",
            "kind": "wide_narrow_rows",
            "normal_position": 20.0,
            "line_count": 3,
            "work_lines": [
                [(20.0, 15.0), (80.0, 15.0)],
                [(20.0, 25.0), (80.0, 25.0)],
                [(20.0, 35.0), (80.0, 35.0)],
            ],
        },
        {
            "region_id": "uniform_block_1",
            "kind": "uniform_block",
            "normal_position": 75.0,
            "line_count": 3,
            "work_lines": [
                [(20.0, 65.0), (80.0, 65.0)],
                [(20.0, 75.0), (80.0, 75.0)],
                [(20.0, 85.0), (80.0, 85.0)],
            ],
        },
    ]

    _path, _types, assessment, _ordered = _connect_work_regions(
        regions,
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="contour",
        config={
            "contour_max_endpoint_extension_m": 2.0,
            "headland_clearance_m": 0.6,
        },
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    connector = assessment["contour_connector"]
    assert connector["shared_turn_clearance_m_by_side"] == {
        "low": 0.0, "high": 0.6
    }
    assert assessment["contour_ordering"]["global_line_count"] == 6


def test_gap_recovery_interpolates_four_row_lattice_and_ignores_narrow_band():
    """DECISION-009/011/017: a two-row centre must never become a work line."""
    from path_planner import _recover_lattice_supported_work_bands

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class State:
        mask_raw = np.ones((1200, 120), dtype=np.uint8) * 255
        mask_offset_x = 0
        mask_offset_y = 0
        harvester_params = {"cutter_width_m": 1.75}

    uniform = _band(1, 50.0, 12.45, 3.89)
    regular = [
        _band(2, 200.0, 1.0),
        _band(4, 400.0, 1.0),
        _band(6, 800.0, 1.0),
        _band(8, 1000.0, 1.0),
    ]
    wrong_narrow = {
        **_band(5, 660.0, 0.52),
        "band_type": "narrow",
    }
    unrelated_narrow = {
        **_band(9, 1100.0, 0.50),
        "band_type": "narrow",
    }

    recovered, report = _recover_lattice_supported_work_bands(
        [uniform, *regular],
        [uniform, *regular, wrong_narrow, unrelated_narrow],
        State.mask_raw,
        0.0,
        geo=Geo(),
        state=State(),
        config={"validation_profile": "research", "recover_missing_work_bands": True},
    )

    assert report["recovered_band_ids"] == [-1]
    assert [item["id"] for item in recovered] == [
        1, 2, 4, 6, 8, -1
    ]
    promoted = recovered[-1]
    assert promoted["planning_recovered"] is True
    assert promoted["source_band_type"] is None
    assert promoted["planning_recovery_id"] == "planning-gap-4-6-1"
    assert {point[1] for point in promoted["centerline"]} == {600.0}
    assert report["records"][0]["ignored_nonwide_band_ids"] == [5]


def test_gap_recovery_rejects_lattice_line_without_processed_body_support():
    """DECISION-009/017: spacing alone may not invent a harvest pass."""
    from path_planner import _recover_lattice_supported_work_bands

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class State:
        mask_raw = np.zeros((1200, 120), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        harvester_params = {"cutter_width_m": 1.75}

    regular = [
        _band(2, 200.0, 1.0),
        _band(4, 400.0, 1.0),
        _band(6, 800.0, 1.0),
        _band(8, 1000.0, 1.0),
    ]
    recovered, report = _recover_lattice_supported_work_bands(
        regular,
        regular,
        State.mask_raw,
        0.0,
        geo=Geo(),
        state=State(),
        config={"validation_profile": "research", "recover_missing_work_bands": True},
    )

    assert [item["id"] for item in recovered] == [2, 4, 6, 8]
    assert report["recovered_band_ids"] == []
    assert report["records"][0]["reason"] == (
        "interpolated_four_row_line_lacks_body_support"
    )


def test_gap_recovery_uses_stable_four_row_phase_instead_of_gap_midpoint():
    """DECISION-017: an asymmetric omission follows the stable lattice phase."""
    from path_planner import _recover_lattice_supported_work_bands

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class State:
        mask_raw = np.ones((1300, 120), dtype=np.uint8) * 255
        mask_offset_x = 0
        mask_offset_y = 0
        harvester_params = {"cutter_width_m": 1.75}

    uniform = _band(1, 50.0, 12.45, 3.89)
    regular = [
        _band(2, 200.0, 1.0),
        _band(4, 390.0, 1.0),
        _band(6, 650.0, 1.0),
        _band(8, 795.0, 1.0),
        _band(10, 945.0, 1.0),
        _band(12, 1090.0, 1.0),
    ]

    recovered, report = _recover_lattice_supported_work_bands(
        [uniform, *regular],
        [uniform, *regular],
        State.mask_raw,
        0.0,
        geo=Geo(),
        state=State(),
        config={"validation_profile": "research", "recover_missing_work_bands": True},
    )

    promoted = next(item for item in recovered if item.get("planning_recovered"))
    recovered_y = float(promoted["centerline"][0][1])
    assert recovered_y == pytest.approx(501.25, abs=2.0)
    assert recovered_y != pytest.approx((390.0 + 650.0) * 0.5, abs=2.0)
    assert report["records"][0]["phase_method"] == (
        "extrapolated_from_longer_right_four_row_cohort"
    )


def test_gap_recovery_accepts_bilateral_crop_evidence_around_machine_center():
    """DECISION-017: the four-row centre may lie in an inter-row crop gap."""
    from path_planner import _recover_lattice_supported_work_bands

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.01

    class State:
        mask_offset_x = 0
        mask_offset_y = 0
        harvester_params = {"cutter_width_m": 1.75}

    mask = np.zeros((1200, 120), dtype=np.uint8)
    # The recovered machine centre is y=600 and intentionally has no crop
    # pixels.  Continuous crop evidence exists on both sides across the full
    # axial range, as happens at the centre of a four-row strip.
    mask[560:585, 8:93] = 255
    mask[615:640, 8:93] = 255
    State.mask_raw = mask
    regular = [
        _band(2, 200.0, 1.0),
        _band(4, 400.0, 1.0),
        _band(6, 800.0, 1.0),
        _band(8, 1000.0, 1.0),
    ]

    recovered, report = _recover_lattice_supported_work_bands(
        regular,
        regular,
        mask,
        0.0,
        geo=Geo(),
        state=State(),
        config={"validation_profile": "research", "recover_missing_work_bands": True},
    )

    assert report["recovered_band_ids"] == [-1]
    record = report["records"][0]
    assert record["processed_body_center_support_ratio"] == 0.0
    assert record["processed_body_support_ratio"] == 1.0
    assert record["processed_body_mean_cross_section_occupancy"] > 0.25


def test_exit_on_opposite_headland_retraces_harvested_final_work_line():
    """DECISION-015/017: exit service must not be a diagonal through the field."""
    from path_planner import _build_exit_service_route

    field = np.asarray(
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        dtype=np.float64,
    )
    route, report = _build_exit_service_route(
        [(10.0, 40.0), (50.0, 40.0), (90.0, 40.0)],
        (10.0, 50.0),
        0.0,
        0.1,
        field,
    )

    assert report["mode"] == "retrace_last_harvested_work_then_exit_headland"
    assert route[:3] == [(90.0, 40.0), (50.0, 40.0), (10.0, 40.0)]
    assert route[-1] == (10.0, 50.0)
    assert report["inside_field_boundary"] is True


def test_opposite_headland_exit_is_split_into_reverse_and_short_exit():
    """DECISION-015/017: harvested retrace and exit keep distinct semantics."""
    from path_planner import _connect_work_lines

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.zeros((100, 100), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)
        ]
        entry_point = None
        exit_point = (10.0, 50.0)
        unload_points = []

    path, segment_types, diagnostics = _connect_work_lines(
        [[(10.0, 40.0), (50.0, 40.0), (90.0, 40.0)]],
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="bow",
        config={
            "service_headland_axial_tolerance_m": 3.0,
            "service_point_mode": "connect",
        },
        return_diagnostics=True,
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    assert segment_types == ["work", "turn_reverse", "exit"]
    assert path[-2] == [(90.0, 40.0), (50.0, 40.0), (10.0, 40.0)]
    assert path[-1] == [(10.0, 40.0), (10.0, 50.0)]
    assert diagnostics["service_points"]["exit_route"]["retrace_last_work"] is True


def test_exit_near_current_work_end_does_not_add_redundant_reverse_segment():
    """A freely placed exit point must not force a last-line retrace."""
    from path_planner import _connect_work_lines

    class Geo:
        @staticmethod
        def is_ready():
            return True

        @staticmethod
        def meters_per_pixel(_x, _y):
            return 0.1

    class State:
        mask_raw = np.zeros((100, 100), dtype=np.uint8)
        mask_offset_x = 0
        mask_offset_y = 0
        field_boundary = [
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)
        ]
        entry_point = None
        exit_point = (90.0, 45.0)
        unload_points = []

    path, segment_types, diagnostics = _connect_work_lines(
        [[(10.0, 40.0), (50.0, 40.0), (90.0, 40.0)]],
        geo=Geo(),
        state=State(),
        min_turn_radius_m=0.5,
        turn_strategy="bow",
        config={
            "service_headland_axial_tolerance_m": 3.0,
            "service_point_mode": "connect",
        },
        return_diagnostics=True,
        global_main_angle=0.0,
        processed_mask=State.mask_raw,
    )

    assert segment_types == ["work", "exit"]
    assert path[-1] == [(90.0, 40.0), (90.0, 45.0)]
    assert diagnostics["service_points"]["exit_route"]["retrace_last_work"] is False
