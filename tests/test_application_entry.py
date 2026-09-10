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


def test_source_entry_only_bootstraps_the_runtime_environment():
    """DECISION-018 v2: 源码入口只做运行环境初始化，不携带依赖检测或安装。"""
    project_root = Path(__file__).resolve().parents[1]
    main_source = (project_root / "src" / "main.py").read_text(encoding="utf-8")
    assert "import startup" in main_source
    assert "ensure_environment" not in main_source

    import startup

    assert callable(startup._prioritize_active_environment_paths)
    assert not hasattr(startup, "ensure_environment")
    assert not hasattr(startup, "check_missing")
