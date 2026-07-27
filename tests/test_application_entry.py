from pathlib import Path


def test_application_entrypoint_uses_the_conventional_main_name():
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "main.py").is_file()
    assert not (project_root / "pyside6_main.py").exists()


def test_project_tests_are_grouped_under_the_tests_directory():
    project_root = Path(__file__).resolve().parents[1]

    assert not list(project_root.glob("test_*.py"))


def test_core_modules_use_short_role_names():
    project_root = Path(__file__).resolve().parents[1]
    required_modules = {
        "cache.py",
        "config.py",
        "coverage.py",
        "export_guard.py",
        "geo.py",
        "inference.py",
        "startup.py",
        "state.py",
        "workflow.py",
    }

    names = {path.name for path in project_root.glob("*.py")}
    assert required_modules <= names
    assert not {name for name in names if name.startswith("app_")}
