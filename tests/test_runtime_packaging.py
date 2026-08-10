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


def test_self_test_flag_is_handled_before_qt_startup(monkeypatch):
    import app_entry

    monkeypatch.setattr(app_entry, "run_self_test", lambda: 7)
    assert app_entry.run_app(["--self-test"]) == 7


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
