#!/usr/bin/env python
import os, sys
REUI_DIR = os.path.dirname(os.path.abspath(__file__))
if REUI_DIR not in sys.path: sys.path.insert(0, REUI_DIR)
try:
    from startup import ensure_environment
    if not ensure_environment(): sys.exit(0)
except Exception as exc:
    print(f"环境检查失败，程序已停止: {exc}", file=sys.stderr)
    sys.exit(1)
try:
    from PySide6 import QtCore
except ImportError:
    print("=" * 60)
    print("  需要安装 PySide6")
    print("  pip install PySide6")
    print("=" * 60)
    sys.exit(1)

def main():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication, QMessageBox, QStyleFactory
    from pyside6_app.main_window import MainWindow
    from config import ConfigValidationError
    from provenance import APP_VERSION
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("智能农机规划系统")
    app.setApplicationVersion(APP_VERSION)
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
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        pass
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
