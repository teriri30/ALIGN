"""Actual Qt application entry shared by source and frozen launches.

DECISION-018: a frozen build must not run the source-tree dependency installer;
all runtime dependencies are already inside the release bundle.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Sequence

from build_info import load_build_info
from runtime_env import configure_geospatial_environment
from runtime_paths import (
    builtin_model_dir,
    bundled_resource,
    ensure_user_dirs,
    user_data_root,
)
from version import APP_VERSION


SELF_TEST_MODULES = (
    "numpy",
    "cv2",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "rasterio",
    "pyproj",
    "shapely",
    "scipy",
    "torch",
    "torchvision",
    "ultralytics",
)


def run_self_test() -> int:
    """Check the frozen bundle without creating a Qt window."""
    _configure_runtime_environment()
    failures: list[str] = []

    def check(name: str, callback) -> None:
        try:
            callback()
            print(f"[OK] {name}")
        except Exception as exc:  # pragma: no cover - exact native errors vary
            failures.append(f"{name}: {exc}")
            print(f"[FAIL] {name}: {exc}")

    for module_name in SELF_TEST_MODULES:
        check(f"import {module_name}", lambda name=module_name: __import__(name))

    check("config.example.json", lambda: _require_file("config.example.json"))
    registry_path = bundled_resource("data/models/model_registry.json")

    def check_registry() -> None:
        with registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        models = payload.get("models", []) if isinstance(payload, dict) else []
        if not models:
            raise RuntimeError("model registry is empty")
        missing = []
        model_dir = builtin_model_dir()
        for item in models:
            filename = str(item.get("name", "")) if isinstance(item, dict) else ""
            if filename and not (model_dir / filename).is_file():
                missing.append(filename)
        if missing:
            raise FileNotFoundError(", ".join(missing))

    check("built-in model registry", check_registry)
    check("pyproj EPSG:4326", lambda: _check_pyproj())
    check("rasterio EPSG:4326", lambda: _check_rasterio())
    check("user data directories", ensure_user_dirs)

    if failures:
        print("冻结包自检失败:")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print(f"reUI self-test passed; version={load_build_info().get('version', APP_VERSION)}")
    return 0


def _require_file(relative_path: str) -> None:
    path = bundled_resource(relative_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))


def _check_pyproj() -> None:
    from pyproj import CRS
    if CRS.from_epsg(4326).to_epsg() != 4326:
        raise RuntimeError("unexpected CRS")


def _check_rasterio() -> None:
    from rasterio.crs import CRS
    if CRS.from_epsg(4326).to_epsg() != 4326:
        raise RuntimeError("unexpected CRS")


def run_app(argv: Sequence[str] | None = None) -> int:
    """Start the Qt GUI, or run the non-GUI self-test."""
    _configure_runtime_environment()
    args = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in args:
        return run_self_test()

    from PySide6 import QtCore
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication, QMessageBox, QStyleFactory
    from pyside6_app.main_window import MainWindow
    from config import ConfigValidationError

    QtCore.QCoreApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True
    )
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication([sys.argv[0], *args])
    app.setApplicationName("智能农机规划系统")
    build_info = load_build_info()
    app.setApplicationVersion(str(build_info.get("version") or APP_VERSION))
    app.setOrganizationName("ZHL")
    app.setStyle(QStyleFactory.create("Fusion"))
    font = QFont("Microsoft YaHei UI", 10)
    if hasattr(font, "setFamilies"):
        font.setFamilies(["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Segoe UI"])
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)
    try:
        window = MainWindow()
    except ConfigValidationError as exc:
        QMessageBox.critical(
            None,
            "配置无效",
            f"配置文件包含不安全或不合法的参数，程序已停止。\n\n{exc}",
        )
        return 2
    window.show()
    if args and len(args) > 0:
        # MainWindow reads the first file argument from sys.argv for backwards
        # compatibility.  Keep it in the process argv without interpreting
        # arbitrary flags as file paths.
        pass
    return app.exec()


def _configure_runtime_environment() -> None:
    """Configure bundled geospatial data and writable third-party settings."""
    configure_geospatial_environment()
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(user_data_root() / "ultralytics"),
    )
