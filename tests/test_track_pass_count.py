"""Regression tests for logical-segment track pass counting.

DECISION-015: adjacent poses inside one logical path segment are one pass;
only the union of separate logical segments can create formal repeats.
"""

import numpy as np
import pytest

from footprint_planner import validate_footprints


class _Geo:
    @staticmethod
    def is_ready():
        return True

    @staticmethod
    def meters_per_pixel(_x, _y):
        return 0.05


class _State:
    mask_offset_x = 0
    mask_offset_y = 0
    field_boundary = [(0, 0), (239, 0), (239, 239), (0, 239)]


_PARAMS = {
    "cutter_width_m": 0.8,
    "cutter_forward_offset_m": 0.5,
    "track_width_m": 0.20,
    "track_gauge_m": 0.60,
    "wheelbase_m": 0.40,
    "track_length_m": 0.40,
    "turn_radius_m": 0.20,
}


def _metric(path, segment_types, *, pose_step=0.08):
    return validate_footprints(
        path,
        np.zeros((240, 240), dtype=np.uint8),
        geo=_Geo(),
        state=_State(),
        harvester=_PARAMS,
        segment_types=segment_types,
        config={
            "validation_profile": "research",
            "footprint_pose_step_m": pose_step,
            "min_geometric_coverage_pct": 0.0,
        },
    )


def test_single_straight_pass_has_no_repeat_for_three_pose_steps():
    path = [[(30.0, 120.0), (210.0, 120.0)]]

    results = [_metric(path, ["work"], pose_step=step) for step in (0.04, 0.08, 0.16)]

    assert [item["track_rolling_visit_twice_or_more_area_m2"] for item in results] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]
    assert all(item["footprint_repeat_count_basis"] == "logical_path_segment_union" for item in results)
    assert all(item["track_pass_count_dtype"] == "uint32" for item in results)


def test_two_identical_logical_passes_repeat_once_the_single_sweep():
    path = [[(30.0, 120.0), (210.0, 120.0)]]

    single = _metric(path, ["work"])
    repeated = _metric([path[0], path[0]], ["work", "work"])

    assert repeated["track_rolling_visit_twice_or_more_area_m2"] == pytest.approx(
        single["track_swept_area_m2"], rel=0.08
    )


def test_parallel_non_intersecting_passes_have_no_repeat_area():
    result = _metric(
        [[(30.0, 70.0), (210.0, 70.0)], [(30.0, 170.0), (210.0, 170.0)]],
        ["work", "work"],
    )

    assert result["track_rolling_visit_twice_or_more_area_m2"] == pytest.approx(0.0)


def test_crossing_passes_repeat_only_in_the_crossing_region():
    crossing = _metric(
        [[(30.0, 120.0), (210.0, 120.0)], [(120.0, 30.0), (120.0, 210.0)]],
        ["work", "work"],
    )
    single = _metric([[(30.0, 120.0), (210.0, 120.0)]], ["work"])

    repeated_area = crossing["track_rolling_visit_twice_or_more_area_m2"]
    assert repeated_area > 0.0
    assert repeated_area < single["track_swept_area_m2"]


def test_one_curved_segment_is_not_repeated_by_overlapping_poses():
    result = _metric(
        [[(35.0, 120.0), (75.0, 75.0), (120.0, 120.0), (165.0, 75.0), (205.0, 120.0)]],
        ["work"],
        pose_step=0.04,
    )

    assert result["track_rolling_visit_twice_or_more_area_m2"] == pytest.approx(0.0)


def test_repeat_area_is_stable_across_pose_steps():
    path = [[(30.0, 120.0), (210.0, 120.0)]]
    values = [
        _metric([path[0], path[0]], ["work", "work"], pose_step=step)[
            "track_rolling_visit_twice_or_more_area_m2"
        ]
        for step in (0.04, 0.08, 0.16)
    ]

    assert max(values) - min(values) <= max(values) * 0.15 + 0.01


def test_logical_pass_counter_saturates_without_uint32_overflow():
    from footprint_planner import _accumulate_logical_segment_pass

    maximum = np.iinfo(np.uint32).max
    count = np.asarray([[maximum]], dtype=np.uint32)
    sweep = np.asarray([[255]], dtype=np.uint8)

    _accumulate_logical_segment_pass(count, sweep)

    assert count.dtype == np.uint32
    assert int(count[0, 0]) == maximum
