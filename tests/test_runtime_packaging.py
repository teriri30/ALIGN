"""DECISION-018: source and frozen runtime paths remain auditable."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_user_data_paths_are_separate_from_the_bundle(monkeypatch, tmp_path):
    from runtime_paths import ensure_user_dirs, user_config_path

    user_root = tmp_path / "user-data"
    monkeypatch.setenv("REUI_USER_DATA_DIR", str(user_root))
    paths = ensure_user_dirs()

    assert user_config_path() == user_root / "config" / "config.json"
    assert paths["cache"] == user_root / "cache"
    assert paths["logs"] == user_root / "logs"
    assert paths["models"] == user_root / "models"
    assert not str(user_root).startswith(str(ROOT))
    assert all(path.is_dir() for path in paths.values())


def test_config_first_run_migrates_legacy_file_to_user_data(monkeypatch, tmp_path):
    import config as config_module
    from config import Config

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    legacy = bundle / "config.json"
    example = bundle / "config.example.json"
    legacy.write_text('{"_schema_version":"1.3","model":{}}', encoding="utf-8")
    example.write_text('{"_schema_version":"1.3","model":{}}', encoding="utf-8")
    user_root = tmp_path / "user-data"
    monkeypatch.setenv("REUI_USER_DATA_DIR", str(user_root))
    monkeypatch.delenv("REUI_CONFIG_PATH", raising=False)
    monkeypatch.setattr(config_module, "legacy_config_path", lambda: legacy)
    monkeypatch.setattr(config_module, "bundled_resource", lambda name: bundle / name)

    cfg = Config()
    assert cfg.load()
    assert Path(cfg.config_path()) == user_root / "config" / "config.json"
    assert Path(cfg.config_path()).is_file()


def test_resource_resolution_supports_exe_root_and_internal(monkeypatch, tmp_path):
    import sys
    import runtime_paths

    exe_root = tmp_path / "release"
    internal = exe_root / "_internal"
    resource = internal / "data" / "models" / "model_registry.json"
    resource.parent.mkdir(parents=True)
    resource.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_root / "reUI.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)

    assert runtime_paths.bundled_resource("data/models/model_registry.json") == resource


def test_frozen_entry_never_calls_source_environment_installer():
    source = (ROOT / "src" / "frozen_main.py").read_text(encoding="utf-8")
    assert "startup" not in source
    assert "pip" not in source
    assert "from app_entry import run_app" in source


def test_no_runtime_entry_carries_a_dependency_installer():
    """DECISION-018 v2: 任何入口都不携带依赖检测或安装界面。"""
    for relative in ("src/main.py", "src/startup.py", "src/frozen_main.py", "src/app_entry.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "tkinter" not in source, relative
        assert "pip install" not in source, relative
        assert "EnvSetupDialog" not in source, relative


def test_self_test_flag_is_handled_before_qt_startup(monkeypatch):
    import app_entry

    monkeypatch.setattr(app_entry, "run_self_test", lambda: 7)
    assert app_entry.run_app(["--self-test"]) == 7


def test_frozen_self_test_imports_qt_extension_modules():
    import app_entry

    assert {
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    } <= set(app_entry.SELF_TEST_MODULES)


def test_windows_build_isolates_pyinstaller_dll_search_path():
    script = (ROOT / "build_windows.bat").read_text(encoding="utf-8")

    assert 'set "BUILD_ORIGINAL_PATH=%PATH%"' in script
    assert 'set "PATH=%BUILD_PYINSTALLER_PATH%"' in script
    assert 'set "PATH=%BUILD_ORIGINAL_PATH%"' in script
    assert 'set "YOLO_AUTOINSTALL=false"' in script


def test_build_spec_rejects_codex_tool_runtime_dlls():
    spec = (ROOT / "packaging" / "reUI.spec").read_text(encoding="utf-8")

    assert '"codex-runtimes"' in spec
    assert "analysis.binaries = filter_host_tool_binaries" in spec


def test_version_has_one_runtime_source():
    import provenance
    from version import APP_VERSION

    assert provenance.APP_VERSION == APP_VERSION
    assert 'APP_VERSION = "0.1.0"' not in (
        ROOT / "src" / "provenance.py"
    ).read_text(encoding="utf-8")


def test_build_spec_keeps_exe_at_release_root():
    spec = (ROOT / "packaging" / "reUI.spec").read_text(encoding="utf-8")
    assert 'name="reUI"' in spec
    assert 'contents_directory="_internal"' in spec
    assert 'SOURCE_ROOT = ROOT / "src"' in spec
    assert 'SOURCE_ROOT / "frozen_main.py"' in spec

def test_windows_build_verifies_and_cleans_rebuildable_outputs():
    """build_windows.bat verifies the published EXE against the staging copy
    (logged in the build output) before removing rebuildable outputs."""
    script = (ROOT / "build_windows.bat").read_text(encoding="utf-8")
    assert "Get-FileHash" in script
    assert "[verify]" in script
    assert "REUI_CLEAN_DIST" in script
    assert "rmdir /S /Q dist" in script
