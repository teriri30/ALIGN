from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DECISION_ANCHORS = {
    "DECISION-001": (
        "docs/DECISIONS.md",
        "pyside6_app/main_window.py",
        "integrity_check.py",
        "tests/test_geo_export_safety.py",
    ),
    "DECISION-002": (
        "docs/DECISIONS.md",
        "cache.py",
        "provenance.py",
        "tests/test_research_integrity.py",
    ),
    "DECISION-003": (
        "docs/DECISIONS.md",
        "pyside6_app/workers.py",
        "tests/test_scientific_safety.py",
    ),
    "DECISION-004": (
        "docs/DECISIONS.md",
        "inference.py",
        "pyside6_app/image_view.py",
        "tests/test_reui_logic.py",
        "tests/test_geospatial_integrity.py",
    ),
    "DECISION-005": (
        "docs/DECISIONS.md",
        "raster_preprocessing.py",
        "tests/test_research_integrity.py",
    ),
    "DECISION-006": (
        "docs/DECISIONS.md",
        "pyside6_app/main_window.py",
        "tests/test_research_integrity.py",
    ),
    "DECISION-007": (
        "docs/DECISIONS.md",
        "pyside6_app/main_window.py",
        "tests/test_scientific_safety.py",
    ),
    "DECISION-008": (
        "docs/DECISIONS.md",
        "README.md",
        "tests/test_scientific_safety.py",
    ),
    "DECISION-009": (
        "docs/DECISIONS.md",
        "row_geometry.py",
        "footprint_planner.py",
        "tests/test_mask_strategy.py",
    ),
    "DECISION-010": (
        "docs/DECISIONS.md",
        "path_planner.py",
        "tests/test_turn_endpoint.py",
    ),
    "DECISION-011": (
        "docs/DECISIONS.md",
        "footprint_planner.py",
        "tests/test_planar_footprint.py",
    ),
    "DECISION-012": (
        "docs/DECISIONS.md",
        "path_pose.py",
        "path_planner.py",
        "workflow.py",
        "tests/test_cross_module_math.py",
    ),
    "DECISION-013": (
        "docs/DECISIONS.md",
        "footprint_planner.py",
        "path_planner.py",
        "pyside6_app/workers.py",
        "tests/test_cross_module_math.py",
    ),
    "DECISION-014": (
        "docs/DECISIONS.md",
        "paper_experiments.py",
        "tests/test_paper_experiments.py",
    ),
    "DECISION-015": (
        "docs/DECISIONS.md",
        "docs/DATA_CONTRACTS.md",
        "footprint_planner.py",
        "path_planner.py",
        "headland_turn_optimizer.py",
        "tests/test_headland_turn.py",
        "tests/test_headland_turn_contract.py",
        "tests/test_headland_turn_integration.py",
    ),
}


def test_decision_triplets_remain_linked():
    """Every decision must remain visible in its document, code, and invariant test."""
    for decision_id, relative_paths in DECISION_ANCHORS.items():
        for relative_path in relative_paths:
            content = (ROOT / relative_path).read_text(encoding="utf-8")
            assert decision_id in content, f"{decision_id} anchor missing from {relative_path}"


def test_readme_preserves_research_use_and_model_conditions():
    """DECISION-008: public wording states the research use and model conditions."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "科研软件" in readme
    assert "模型结果用于比较不同路径方案的几何差异" in readme
