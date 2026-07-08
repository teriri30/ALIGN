from pathlib import Path


ROOT = Path(__file__).resolve().parent


DECISION_ANCHORS = {
    "DECISION-001": (
        "docs/DECISIONS.md",
        "pyside6_app/main_window.py",
        "integrity_check.py",
        "test_geo_export_safety_current.py",
    ),
    "DECISION-002": (
        "docs/DECISIONS.md",
        "cache.py",
        "provenance.py",
        "test_research_integrity_current.py",
    ),
    "DECISION-003": (
        "docs/DECISIONS.md",
        "pyside6_app/workers.py",
        "test_scientific_safety_current.py",
    ),
    "DECISION-004": (
        "docs/DECISIONS.md",
        "model.py",
        "test_reui_logic.py",
    ),
    "DECISION-005": (
        "docs/DECISIONS.md",
        "raster_preprocessing.py",
        "test_research_integrity_current.py",
    ),
    "DECISION-006": (
        "docs/DECISIONS.md",
        "pyside6_app/main_window.py",
        "test_research_integrity_current.py",
    ),
    "DECISION-007": (
        "docs/DECISIONS.md",
        "pyside6_app/main_window.py",
        "test_scientific_safety_current.py",
    ),
    "DECISION-008": (
        "docs/DECISIONS.md",
        "README.md",
        "test_scientific_safety_current.py",
    ),
    "DECISION-009": (
        "docs/DECISIONS.md",
        "row_geometry.py",
        "footprint_planner.py",
        "test_mask_strategy_current.py",
    ),
    "DECISION-010": (
        "docs/DECISIONS.md",
        "path_planner.py",
        "test_turn_endpoint_current.py",
    ),
    "DECISION-011": (
        "docs/DECISIONS.md",
        "footprint_planner.py",
        "test_planar_footprint_current.py",
    ),
    "DECISION-012": (
        "docs/DECISIONS.md",
        "path_pose.py",
        "path_planner.py",
        "workflow.py",
        "test_cross_module_math_current.py",
    ),
    "DECISION-013": (
        "docs/DECISIONS.md",
        "footprint_planner.py",
        "path_planner.py",
        "pyside6_app/workers.py",
        "test_cross_module_math_current.py",
    ),
}


def test_decision_triplets_remain_linked():
    """Every decision must remain visible in its document, code, and invariant test."""
    for decision_id, relative_paths in DECISION_ANCHORS.items():
        for relative_path in relative_paths:
            content = (ROOT / relative_path).read_text(encoding="utf-8")
            assert decision_id in content, f"{decision_id} anchor missing from {relative_path}"


def test_readme_preserves_research_prototype_boundary():
    """DECISION-008: public wording must preserve the research-prototype boundary."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "科研原型" in readme
    assert "不适用：未经人工复核直接驱动农机" in readme
