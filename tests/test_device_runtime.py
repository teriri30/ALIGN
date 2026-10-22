"""GPU 运行时的检测、挂载、安装与移除（DECISION-045）。

这些用例跑在临时 ALIGN_USER_DATA_DIR 下（见 conftest），不会碰用户真实数据。
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

import device_runtime
from device_runtime import GpuWheelSpec

_ORIGINAL_GPU_RUNTIME_DIR = device_runtime.gpu_runtime_dir


@pytest.fixture()
def runtime_root(tmp_path, monkeypatch):
    """把 GPU 运行时目录指到临时目录。"""
    root = tmp_path / "gpu-runtime"
    monkeypatch.setattr(device_runtime, "gpu_runtime_dir", lambda: root)
    return root


@pytest.fixture()
def clean_torch(monkeypatch):
    """临时把 torch 从 sys.modules 里摘掉，测完还原。"""
    saved = sys.modules.pop("torch", None)
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["torch"] = saved


@pytest.fixture()
def clean_path():
    saved = list(sys.path)
    try:
        yield
    finally:
        sys.path[:] = saved


def _write_package(root: Path, package: str) -> None:
    (root / package).mkdir(parents=True, exist_ok=True)
    (root / package / "__init__.py").write_text("", encoding="utf-8")
    (root / package / "lib").mkdir(exist_ok=True)
    (root / package / "lib" / "cudnn.dll").write_bytes(b"x" * 128)


# ---------------------------------------------------------------- 目录与挂载


def test_gpu_runtime_dir_stays_under_user_data_root(monkeypatch, tmp_path):
    """默认实现必须落在用户数据目录下，卸载程序不会连带删掉它。"""
    default_dir = _ORIGINAL_GPU_RUNTIME_DIR
    monkeypatch.setattr(device_runtime, "user_data_root", lambda: tmp_path)
    assert default_dir() == tmp_path / device_runtime.GPU_RUNTIME_DIRNAME


def test_activate_requires_real_torch_package(runtime_root, clean_torch, clean_path):
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "torch").mkdir()
    assert device_runtime.activate() is False  # 只有空目录不算装好
    _write_package(runtime_root, "torch")
    assert device_runtime.gpu_runtime_installed() is True
    assert device_runtime.activate() is True
    assert sys.path[0] == str(runtime_root)


def test_activate_is_idempotent(runtime_root, clean_torch, clean_path):
    _write_package(runtime_root, "torch")
    assert device_runtime.activate() is True
    assert device_runtime.activate() is True
    assert sys.path.count(str(runtime_root)) == 1


def test_activate_does_nothing_when_torch_already_imported(runtime_root, clean_path, monkeypatch):
    _write_package(runtime_root, "torch")
    monkeypatch.setitem(sys.modules, "torch", object())
    assert device_runtime.activate() is False
    assert str(runtime_root) not in sys.path


def test_deactivate_removes_path_entry(runtime_root, clean_torch, clean_path):
    _write_package(runtime_root, "torch")
    device_runtime.activate()
    device_runtime.deactivate()
    assert str(runtime_root) not in sys.path


def test_runtime_size_and_remove(runtime_root):
    _write_package(runtime_root, "torch")
    size = device_runtime.gpu_runtime_size()
    assert size >= 128
    assert device_runtime.remove_gpu_runtime() is True
    assert not runtime_root.exists()
    assert device_runtime.gpu_runtime_installed() is False
    assert device_runtime.remove_gpu_runtime() is False


# ---------------------------------------------------------------- 显卡检测


def test_detect_nvidia_gpu_parses_smi_output(monkeypatch):
    class _Done:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 5060 Ti, 581.15, 16311\n"

    monkeypatch.setattr(device_runtime.subprocess, "run", lambda *a, **k: _Done())
    info = device_runtime.detect_nvidia_gpu()
    assert info["name"].startswith("NVIDIA GeForce RTX 5060")
    assert info["driver"] == "581.15"
    assert info["memory_mb"] == 16311


def test_detect_nvidia_gpu_returns_none_without_driver(monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(device_runtime.subprocess, "run", _boom)
    assert device_runtime.detect_nvidia_gpu() is None


def test_detect_nvidia_gpu_returns_none_on_error_exit(monkeypatch):
    class _Done:
        returncode = 9
        stdout = ""

    monkeypatch.setattr(device_runtime.subprocess, "run", lambda *a, **k: _Done())
    assert device_runtime.detect_nvidia_gpu() is None


# ---------------------------------------------------------------- 状态汇报


def test_torch_status_reports_error_instead_of_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    status = device_runtime.torch_status()
    assert status["available"] is False
    assert status["error"]


# ---------------------------------------------------------------- 安装流程


def _fake_wheel(package: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{package}/__init__.py", "")
        archive.writestr(f"{package}/lib/kernel.dll", b"k" * 4096)
        archive.writestr(f"{package}-0.0.0.dist-info/METADATA", "Name: x")
    return buffer.getvalue()


def test_install_extracts_package_and_writes_marker(runtime_root, monkeypatch, clean_torch, clean_path):
    wheel = _fake_wheel("torch")
    spec = GpuWheelSpec(
        package="torch",
        filename="torch-fake.whl",
        size=len(wheel),
        sha256="",
        mirrors=("https://example.invalid/torch-fake.whl",),
    )

    def fake_download(spec_, dest, progress, is_cancelled):
        dest.write_bytes(wheel)
        progress(spec_.package, len(wheel), len(wheel), 0.0)

    monkeypatch.setattr(device_runtime, "_download", fake_download)
    events = []
    installed = device_runtime.install_gpu_runtime(
        progress=lambda *args: events.append(args), wheels=(spec,)
    )

    assert installed == runtime_root
    assert (runtime_root / "torch" / "lib" / "kernel.dll").read_bytes() == b"k" * 4096
    assert not (runtime_root / "torch-0.0.0.dist-info").exists()  # 只摊开包目录
    assert (runtime_root / device_runtime.MARKER_FILENAME).is_file()
    assert events, "进度回调至少被调用一次"
    assert device_runtime.activate() is True


def test_install_cleans_up_when_download_fails(runtime_root, monkeypatch):
    spec = GpuWheelSpec("torch", "torch-fake.whl", 10, "", ("https://example.invalid/x.whl",))

    def boom(spec_, dest, progress, is_cancelled):
        raise device_runtime.InstallError("网络不通")

    monkeypatch.setattr(device_runtime, "_download", boom)
    with pytest.raises(device_runtime.InstallError):
        device_runtime.install_gpu_runtime(wheels=(spec,))
    assert not runtime_root.exists(), "失败后不能留下半装的 torch"


def test_install_stops_when_cancelled(runtime_root, monkeypatch):
    spec = GpuWheelSpec("torch", "torch-fake.whl", 10, "", ("https://example.invalid/x.whl",))
    monkeypatch.setattr(
        device_runtime, "_download", lambda spec_, dest, progress, is_cancelled: None
    )
    with pytest.raises(device_runtime.InstallCancelled):
        device_runtime.install_gpu_runtime(wheels=(spec,), is_cancelled=lambda: True)
    assert not runtime_root.exists()


def test_install_refuses_when_disk_is_tight(runtime_root, monkeypatch):
    spec = GpuWheelSpec("torch", "torch-fake.whl", 10**10, "", ("https://example.invalid/x.whl",))
    monkeypatch.setattr(device_runtime, "_disk_free_bytes", lambda path: 1000)
    with pytest.raises(device_runtime.InstallError) as excinfo:
        device_runtime.install_gpu_runtime(wheels=(spec,))
    assert "磁盘空间不足" in str(excinfo.value)


def test_wheel_specs_match_release_torch_version():
    """GPU 运行时必须与发布包内置 torch 严格同版本，否则 ABI 不匹配。"""
    release = Path(device_runtime.__file__).resolve().parents[1] / "developer" / "requirements" / "requirements-release.txt"
    text = release.read_text(encoding="utf-8")
    torch_spec = next(s for s in device_runtime.GPU_WHEELS if s.package == "torch")
    assert "torch==2.9.1+cpu" in text
    assert torch_spec.filename.startswith("torch-2.9.1+cu128-")
    assert torch_spec.mirrors[0].startswith("https://")
    assert len(torch_spec.sha256) == 64
    vision = next(s for s in device_runtime.GPU_WHEELS if s.package == "torchvision")
    assert vision.filename.startswith("torchvision-0.24.1+cu128-")


# ---------------------------------------------------------------- 界面入口


def test_settings_dialog_exposes_runtime_device_tab(monkeypatch):
    """「设置 → 运行设备」必须存在，否则用户根本找不到开关。"""
    from PySide6.QtWidgets import QApplication, QTabWidget

    QApplication.instance() or QApplication([])
    from pyside6_app import dialogs

    # 不在测试里真的去 import torch（要几秒），也不启动后台线程
    monkeypatch.setattr(dialogs._DeviceStatusWorker, "start", lambda self: None)
    dialog = dialogs.SettingsDialog({})
    tabs = dialog.findChild(QTabWidget)
    names = [tabs.tabText(i) for i in range(tabs.count())]
    assert "运行设备" in names
    dialog.close()


def _device_dialog(monkeypatch):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pyside6_app import dialogs

    monkeypatch.setattr(dialogs._DeviceStatusWorker, "start", lambda self: None)
    return dialogs.SettingsDialog({})


def test_device_tab_offers_install_when_gpu_present(monkeypatch):
    dialog = _device_dialog(monkeypatch)
    dialog._apply_device_state(
        {
            "available": True,
            "version": "2.9.1+cpu",
            "cuda_build": "",
            "cuda_available": False,
            "gpu": {"name": "NVIDIA GeForce RTX 5060 Ti", "driver": "581.15", "memory_mb": 16311},
            "installed": False,
            "size": 0,
        }
    )
    assert "可以开 GPU 加速" in dialog.lbl_device_head.text()
    assert dialog.btn_device_enable.isVisible() is False or dialog.btn_device_enable.text()
    assert "2.7 GB" in dialog.btn_device_enable.text()
    assert dialog.btn_device_remove.isHidden() is True
    dialog.close()


def test_device_tab_tells_cpu_only_users_to_do_nothing(monkeypatch):
    dialog = _device_dialog(monkeypatch)
    dialog._apply_device_state(
        {
            "available": True,
            "version": "2.9.1+cpu",
            "cuda_build": "",
            "cuda_available": False,
            "gpu": None,
            "installed": False,
            "size": 0,
        }
    )
    assert dialog.lbl_device_head.text() == "正在使用 CPU"
    assert "不需要做任何设置" in dialog.lbl_device_hint.text()
    assert dialog.btn_device_enable.isHidden() is True
    dialog.close()


def test_device_tab_reports_gpu_in_use(monkeypatch):
    dialog = _device_dialog(monkeypatch)
    dialog._apply_device_state(
        {
            "available": True,
            "version": "2.9.1+cu128",
            "cuda_build": "12.8",
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 5060 Ti",
            "gpu": {"name": "NVIDIA GeForce RTX 5060 Ti", "driver": "581.15", "memory_mb": 16311},
            "installed": True,
            "size": 4_300_000_000,
        }
    )
    assert dialog.lbl_device_head.text().startswith("正在使用 GPU")
    assert "RTX 5060 Ti" in dialog.lbl_device_detail.text()
    dialog.close()


def test_device_tab_does_not_leave_stale_buttons_after_state_change(monkeypatch):
    """先显示"可启用"，再切成"没有 N 卡"时，按钮必须收起来（曾出现残留）。"""
    dialog = _device_dialog(monkeypatch)
    dialog._apply_device_state(
        {
            "available": True,
            "version": "2.9.1+cpu",
            "cuda_build": "",
            "cuda_available": False,
            "gpu": {"name": "NVIDIA GeForce RTX 3060", "driver": "560.94", "memory_mb": 12288},
            "installed": False,
            "size": 0,
        }
    )
    assert dialog.btn_device_enable.isHidden() is False
    dialog._apply_device_state(
        {
            "available": True,
            "version": "2.9.1+cpu",
            "cuda_build": "",
            "cuda_available": False,
            "gpu": None,
            "installed": False,
            "size": 0,
        }
    )
    assert dialog.btn_device_enable.isHidden() is True
    assert dialog.btn_device_remove.isHidden() is True
    dialog.close()


def test_device_hint_keeps_room_for_three_lines(monkeypatch):
    """说明文字曾经只给一行高，后半句被裁掉。"""
    dialog = _device_dialog(monkeypatch)
    assert dialog.lbl_device_hint.minimumHeight() >= 50
    assert dialog.lbl_device_detail.minimumHeight() >= 30
    dialog.close()


# ---------------------------------------------------------------- 字体


def test_ui_font_is_times_new_roman_with_yahei_fallback():
    """数字/英文用新罗马，中文回退微软雅黑（Qt 逐字符匹配字体族）。"""
    from pyside6_app.styles import FONT_FAMILIES, FONT_FAMILY_CSS, build_stylesheet

    assert FONT_FAMILIES[0] == "Times New Roman"
    assert "Microsoft YaHei" in FONT_FAMILIES
    assert '"Times New Roman"' in FONT_FAMILY_CSS
    # 字体族只由 QApplication.setFont 提供：QSS 里再写 font-family 会压过 setFont，
    # 之前步骤标题就是这样回退成 SimHei 的（DECISION-046）。
    css = build_stylesheet()
    assert "Times New Roman" not in css

    from app_entry import build_app_font

    font = build_app_font()
    assert font.families()[0] == "Times New Roman"
    assert "Microsoft YaHei" in font.families()
