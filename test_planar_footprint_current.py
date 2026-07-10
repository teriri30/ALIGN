import numpy as np
import pytest

from footprint_planner import validate_footprints
from harvester_geometry import planar_geometry_m


def _params(track_length=1.6):
    return {
        "cutter_width_m": 2.2,
        "cutter_forward_offset_m": 1.3,
        "track_width_m": 0.4,
        "track_gauge_m": 1.6,
        "wheelbase_m": 1.9,
        "track_length_m": track_length,
        "turn_radius_m": 1.0,
    }


def test_planar_geometry_uses_the_same_configured_dimensions_as_animation():
    """DECISION-011: display and validation share one planar geometry."""
    geometry = planar_geometry_m(_params())
    left = geometry["left_track"]
    right = geometry["right_track"]
    cutter = geometry["cutter_line"]
    cutter_visual = geometry["cutter_visual"]
    cutter_visual_edge = geometry["cutter_visual_edge"]
    body = geometry["body"]

    assert np.ptp(left[:, 0]) == pytest.approx(1.6)
    assert np.ptp(left[:, 1]) == pytest.approx(0.4)
    assert abs(np.mean(left[:, 1]) - np.mean(right[:, 1])) == pytest.approx(1.6)
    assert np.ptp(cutter[:, 1]) == pytest.approx(2.2)
    assert cutter[:, 0] == pytest.approx([1.3, 1.3])
    assert cutter_visual.shape == (4, 2)
    assert np.min(cutter_visual[:, 0]) == pytest.approx(np.max(body[:, 0]))
    assert np.max(cutter_visual[:, 0]) < np.max(cutter[:, 0])
    assert cutter_visual_edge[:, 0] == pytest.approx([np.max(cutter_visual[:, 0])] * 2)
    assert np.ptp(cutter_visual[:, 1]) == pytest.approx(2.2)
    assert 0.16 <= np.ptp(cutter_visual[:, 0]) <= 0.26
    assert np.ptp(body[:, 0]) == pytest.approx(1.9)


def test_track_contact_length_changes_terminal_rolling_area():
    crop = np.zeros((180, 260), dtype=np.uint8)
    crop[75:105, 165:225] = 255
    path = [[(50.0, 90.0), (160.0, 90.0)]]
    common = {
        "cutter_width_m": 0.8,
        "cutter_forward_offset_m": 0.8,
        "track_width_m": 0.20,
        "track_gauge_m": 0.0,
        "wheelbase_m": 1.0,
        "turn_radius_m": 0.5,
    }

    short = validate_footprints(
        path, crop,
        harvester={**common, "track_length_m": 0.20},
        segment_types=["turn"],
        **_metric_context(),
    )
    long = validate_footprints(
        path, crop,
        harvester={**common, "track_length_m": 1.00},
        segment_types=["turn"],
        **_metric_context(),
    )

    assert long["rolling_canopy_pct"] > short["rolling_canopy_pct"]
    assert long["track_swept_area_m2"] > short["track_swept_area_m2"]


def test_machine_candidate_validation_rejects_machine_envelope_risk(monkeypatch):
    import path_planner

    footprint = {
        "track_core_overlap_pct": 0.0,
        "planned_target_geometric_coverage_pct": 100.0,
        "track_outside_field_pct": 0.0,
        "track_outside_support_pct": 0.0,
        "track_uncertain_overlap_pct": 0.0,
        "track_forbidden_overlap_pct": 0.0,
        "machine_outside_field_pct": 0.1,
        "machine_forbidden_overlap_pct": 0.0,
    }
    monkeypatch.setattr(path_planner, "validate_footprints", lambda *_args, **_kwargs: footprint)

    result = path_planner.validate_path(
        [[(1.0, 1.0), (2.0, 1.0)]],
        np.ones((20, 20), dtype=np.uint8) * 255,
        0.0,
        geo=_Geo(),
        state=_State(),
        segment_types=["work"],
        config={
            "validation_profile": "machine_candidate",
            "max_track_core_overlap_pct": 100.0,
            "min_geometric_coverage_pct": 0.0,
        },
    )

    assert result["valid"] is False
    assert result["machine_outside_field_pct"] == pytest.approx(0.1)
    assert result["validation_limits_pct"]["machine_outside_field"] == pytest.approx(0.0)
    assert any("整机平面包络越出田块" in issue for issue in result["issues"])


def test_validation_identifies_rigid_planar_footprint_model():
    crop = np.zeros((80, 120), dtype=np.uint8)
    crop[30:50, 10:110] = 255
    result = validate_footprints(
        [[(10.0, 40.0), (110.0, 40.0)]],
        crop,
        harvester=_params(),
        segment_types=["work"],
        **_metric_context(),
    )

    assert result["footprint_model"] == "planar_rigid_footprint_v2"
    assert result["footprint_reference_point"] == "track_contact_center"
    assert result["footprint_dimensions_m"]["track_length_m"] == pytest.approx(1.6)
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
    field_boundary = []
    harvester_params = _params()


def _metric_context():
    return {"geo": _Geo(), "state": _State()}
