import os
from pathlib import Path


def test_application_entrypoint_uses_the_conventional_main_name():
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "src" / "main.py").is_file()
    assert not list(project_root.glob("*.py"))
    assert not (project_root / "pyside6_main.py").exists()


def test_project_tests_are_grouped_under_the_tests_directory():
    project_root = Path(__file__).resolve().parents[1]

    assert not list(project_root.glob("test_*.py"))


def test_core_modules_use_short_role_names():
    project_root = Path(__file__).resolve().parents[1]
    required_modules = {
        "cache.py",
        "config.py",
        "legacy_route_coverage.py",
        "export_guard.py",
        "geo.py",
        "inference.py",
        "startup.py",
        "state.py",
        "workflow.py",
    }

    names = {path.name for path in (project_root / "src").glob("*.py")}
    assert required_modules <= names
    assert {"app_entry.py", "frozen_main.py"} <= names


def test_startup_removes_foreign_site_packages_instead_of_only_reordering_them(monkeypatch, tmp_path):
    import startup
    import sys

    active = tmp_path / "active-site"
    user = tmp_path / "user-site"
    foreign = tmp_path / "foreign-site"
    active.mkdir()
    user.mkdir()
    foreign.mkdir()

    project = str(Path(startup.__file__).resolve().parent)
    original_path = [project, str(foreign), str(user), str(active)]
    monkeypatch.setattr(sys, "path", original_path)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([project, str(foreign)]),
    )
    monkeypatch.setattr(
        startup.sysconfig,
        "get_path",
        lambda name: str(active) if name in {"purelib", "platlib"} else "",
    )
    monkeypatch.setattr(startup.site, "getusersitepackages", lambda: str(user))

    startup._prioritize_active_environment_paths()

    # user site-packages 优先于解释器自身 site-packages（与 Python 默认
    # sys.path 语义一致），避免环境中损坏的原生扩展遮蔽 user-site 里的
    # 正常版本（如 BLAS 损坏的 numpy 导致 0xC06D007F 崩溃）。
    assert sys.path[:3] == [project, str(user), str(active)]
    assert str(foreign) not in sys.path
    assert "PYTHONPATH" not in os.environ


def test_startup_declares_inference_runtime_dependencies():
    import startup

    imports = {item[1] for item in startup.DEPENDENCIES}
    assert {"torch", "torchvision", "ultralytics"} <= imports


def test_pip_command_uses_user_site_when_environment_site_is_not_writable(monkeypatch):
    import startup

    monkeypatch.setattr(startup, "_site_directory_is_writable", lambda path: False)

    command = startup.get_pip_command(["torch==2.9.1+cu128"])

    assert "--user" in command
    assert "--extra-index-url" in command
    assert startup.PYTORCH_INDEX_URL in command
