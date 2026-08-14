"""DECISION-011: P0 regression tests for parameter provenance and recovery gates."""

import pytest
from PySide6.QtWidgets import QApplication


def _path_result(source="default", *, recovered=False):
    return {
        "is_valid": True,
        "validation": {
            "valid": True,
            "validation_profile": "field_trial",
            "field_boundary_present": True,
            "semantic_support_present": True,
            "forbidden_mask_present": True,
            "footprint_model": "planar_rigid_footprint_v2",
            "footprint_reference_point": "track_contact_center",
            "footprint_resolution_m": 0.05,
        },
        "planning_factors": {
            "harvester": {"track_width_m": 0.4},
            "harvester_param_sources": {"track_width_m": source},
        },
        "layout": {"planning_recovered": recovered},
    }


def test_default_parameter_source_blocks_formal_machine_readiness():
    from machine_route_validator import assess_machine_readiness

    readiness = assess_machine_readiness(_path_result("default"))

    assert readiness["controlled_trial_ready"] is False
    assert "harvester_parameter_source_unverified" in readiness[
        "controlled_trial_blockers"
    ]
    assert readiness["machine_executable"] is False


def test_cached_recovered_work_band_blocks_formal_machine_readiness():
    from machine_route_validator import assess_machine_readiness

    readiness = assess_machine_readiness(_path_result("user_measured", recovered=True))

    assert readiness["planning_recovered"] is True
    assert "planning_recovered_work_bands" in readiness["controlled_trial_blockers"]


def test_formal_profiles_force_missing_band_recovery_off():
    from path_planner import _recover_lattice_supported_work_bands

    for profile in ("field_trial", "machine_candidate"):
        recovered, report = _recover_lattice_supported_work_bands(
            [],
            [],
            None,
            0.0,
            geo=None,
            state=None,
            config={
                "validation_profile": profile,
                "recover_missing_work_bands": True,
            },
        )

        assert recovered == []
        assert report["enabled"] is False
        assert report["reason"] == "disabled"


def test_strict_formal_export_gate_rejects_defaults_and_cached_recovery():
    from pyside6_app.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    with pytest.raises(ValueError, match="参数来源"):
        window._enforce_formal_parameter_and_recovery_gate(_path_result("default"))
    with pytest.raises(ValueError, match="planning_recovered"):
        window._enforce_formal_parameter_and_recovery_gate(
            _path_result("user_measured", recovered=True)
        )
    window.close()


def test_research_export_records_parameter_and_recovery_risks_without_blocking(
    monkeypatch,
):
    from pyside6_app.main_window import MainWindow

    monkeypatch.setenv("REUI_EXPORT_GATE", "confirm")
    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._enforce_formal_parameter_and_recovery_gate(_path_result("default"))
    window._enforce_formal_parameter_and_recovery_gate(
        _path_result("user_measured", recovered=True)
    )
    window.close()


def test_route_info_keeps_parameter_provenance_out_of_default_status_ui():
    from pyside6_app.panels import RouteInfoPanel

    QApplication.instance() or QApplication([])
    panel = RouteInfoPanel()
    panel.update_route(
        {
            "planning_factors": {
                "harvester": {"track_width_m": 0.4},
                "harvester_param_sources": {"track_width_m": "default"},
            },
            "validation": {"valid": False},
        }
    )

    assert panel._safety.toolTip() == "尚未获得有效的路径校验结果"
    assert "履带宽度" not in panel._safety.toolTip()
    assert "默认值" not in panel._safety.toolTip()
