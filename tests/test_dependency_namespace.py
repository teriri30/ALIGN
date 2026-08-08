"""Regression tests for third-party dependency namespaces."""

from pathlib import Path

import pytest


def test_coverage_resolves_to_the_third_party_package():
    import coverage

    project_root = Path(__file__).resolve().parents[1]
    module_path = Path(coverage.__file__).resolve()

    assert module_path != project_root / "legacy_route_coverage.py"
    assert module_path.parent != project_root
    assert hasattr(coverage, "Coverage")


def test_pytest_cov_can_load_when_not_explicitly_disabled(pytestconfig):
    pytest.importorskip("pytest_cov")
    plugin = pytestconfig.pluginmanager.get_plugin("_cov")
    if plugin is None:
        pytest.skip("pytest-cov is intentionally disabled by -p no:cov")
    assert plugin is not None
