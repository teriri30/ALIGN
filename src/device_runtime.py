"""GPU 运行时（CUDA 12.8）的检测、按需下载、挂载与移除。

发布包内置的是 **CPU 版** torch（见 ``developer/requirements/requirements-release.txt``
的 ``torch==2.9.1+cpu``）：CUDA 版解压后 4 GB 以上、压缩包 2.86 GB，既进不了
GitHub Release 的 2 GiB 单文件上限，也不该让没有 NVIDIA 显卡的用户白下 2.7 GB。

因此 GPU 支持改成"按需下载、独立目录、启动时优先挂载"：

    %LOCALAPPDATA%\\ALIGN\\gpu-runtime\\torch\\...
    %LOCALAPPDATA%\\ALIGN\\gpu-runtime\\torchvision\\...

``activate()`` 在**任何 torch 导入之前**把这个目录插到 ``sys.path[0]``，
于是 ``import torch`` 拿到的是 CUDA 版；移除该目录即自动退回内置 CPU 版。
打包时 torch 是作为包目录收集进 ``_internal`` 的（不是冻结进 PYZ），
所以路径优先级对它是生效的。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from runtime_paths import user_data_root

__all__ = [
    "GpuWheelSpec",
    "GPU_WHEELS",
    "gpu_runtime_dir",
    "gpu_runtime_installed",
    "gpu_runtime_size",
    "activate",
    "deactivate",
    "detect_nvidia_gpu",
    "torch_status",
    "install_gpu_runtime",
    "remove_gpu_runtime",
    "InstallCancelled",
    "InstallError",
]


GPU_RUNTIME_DIRNAME = "gpu-runtime"
MARKER_FILENAME = ".align-gpu-runtime.json"


class InstallCancelled(RuntimeError):
    """用户取消下载/解包。"""


class InstallError(RuntimeError):
    """下载、校验或解包失败；消息面向普通用户，可直接展示。"""


@dataclass(frozen=True)
class GpuWheelSpec:
    """一个待下载的 wheel。体积与 sha256 取自 PyTorch 官方索引页。"""

    package: str          # "torch" / "torchvision"
    filename: str
    size: int             # 官方 content-length，用于进度与会话中断判断
    sha256: str
    mirrors: tuple[str, ...]

    @property
    def extract_prefix(self) -> str:
        return f"{self.package}/"

    @property
    def size_mb(self) -> float:
        return self.size / 1_000_000


_OFFICIAL = "https://download.pytorch.org/whl/cu128"
_SJTU = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128"

# Python 3.10 / Windows x64，与发布包内置的 torch 版本严格对齐：
# 版本不一致会出现 C++ ABI 不匹配（_C.pyd 导入失败）。
GPU_WHEELS: tuple[GpuWheelSpec, ...] = (
    GpuWheelSpec(
        package="torch",
        filename="torch-2.9.1+cu128-cp310-cp310-win_amd64.whl",
        size=2_862_053_390,
        sha256="64399adaa8ea0896d02cf844cba3c5dd77e769520a1af73572599e0eaa2cf551",
        mirrors=(f"{_SJTU}/torch-2.9.1%2Bcu128-cp310-cp310-win_amd64.whl",
                 f"{_OFFICIAL}/torch-2.9.1%2Bcu128-cp310-cp310-win_amd64.whl"),
    ),
    GpuWheelSpec(
        package="torchvision",
        filename="torchvision-0.24.1+cu128-cp310-cp310-win_amd64.whl",
        size=9_024_408,
        sha256="ccdf1aff65d5bd1af99c5430508c4c234b81384d78681607ce20f2dcf5407c56",
        mirrors=(f"{_SJTU}/torchvision-0.24.1%2Bcu128-cp310-cp310-win_amd64.whl",
                 f"{_OFFICIAL}/torchvision-0.24.1%2Bcu128-cp310-cp310-win_amd64.whl"),
    ),
)


def gpu_runtime_dir() -> Path:
    """GPU 运行时的安装位置（与项目数据同根，卸载程序不会删它）。"""
    return Path(user_data_root()) / GPU_RUNTIME_DIRNAME


def gpu_runtime_installed() -> bool:
    """目录里已经有可用的 torch 包才算装了。"""
    root = gpu_runtime_dir()
    return (root / "torch" / "__init__.py").is_file()


def gpu_runtime_size() -> int:
    """已安装运行时占用的字节数；没装返回 0。"""
    root = gpu_runtime_dir()
    if not root.is_dir():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def activate() -> bool:
    """把 GPU 运行时插到导入路径最前面；必须在 import torch 之前调用。

    返回 True 表示这次真的挂载了 GPU 运行时。重复调用安全。
    """
    if "torch" in sys.modules:
        # torch 已经被导入（内置 CPU 版），此时再改路径没有意义。
        return False
    if not gpu_runtime_installed():
        return False
    root = str(gpu_runtime_dir())
    if sys.path and sys.path[0] == root:
        return True
    # 去重后插到最前，保证优先于 _internal 里的 CPU 版
    sys.path = [p for p in sys.path if os.path.normcase(p) != os.path.normcase(root)]
    sys.path.insert(0, root)
    return True


def deactivate() -> None:
    """把 GPU 运行时从导入路径里摘掉（仅在 torch 尚未导入时有意义）。"""
    root = os.path.normcase(str(gpu_runtime_dir()))
    sys.path = [p for p in sys.path if os.path.normcase(p) != root]


def detect_nvidia_gpu() -> Optional[dict]:
    """用 nvidia-smi 查显卡；没有 NVIDIA 显卡或驱动时返回 None。

    不抛异常：这台机器上"没有显卡"是完全正常的情况。
    """
    if os.name != "nt" and sys.platform != "win32":
        # 目前只发 Windows 包；其它平台直接按"未检测到"处理
        pass
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    for line in (completed.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            try:
                memory_mb = int(float(parts[2]))
            except ValueError:
                memory_mb = 0
            return {
                "name": parts[0],
                "driver": parts[1],
                "memory_mb": memory_mb,
            }
    return None


def torch_status() -> dict:
    """当前进程实际加载的 torch 状态；导入失败也要给出可读结论。"""
    status = {
        "available": False,
        "version": "",
        "cuda_build": "",
        "cuda_available": False,
        "device_name": "",
        "source": "missing",
        "error": "",
    }
    try:
        import torch  # noqa: PLC0415 —— 只能运行时导入，activate() 要先跑
    except Exception as exc:  # pragma: no cover - 环境损坏时才会走到
        status["error"] = f"{type(exc).__name__}: {exc}"
        return status
    status.update(
        available=True,
        version=str(getattr(torch, "__version__", "")),
        cuda_build=str(getattr(getattr(torch, "version", None), "cuda", "") or ""),
        cuda_available=bool(torch.cuda.is_available()),
        source="gpu-runtime" if _module_from_gpu_runtime(torch) else "bundled",
    )
    if status["cuda_available"]:
        try:
            status["device_name"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover
            status["device_name"] = ""
    return status


def _module_from_gpu_runtime(module) -> bool:
    path = getattr(module, "__file__", "") or ""
    if not path:
        return False
    root = os.path.normcase(str(gpu_runtime_dir()))
    return os.path.normcase(os.path.abspath(path)).startswith(root)


def _disk_free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free
    except OSError:
        return 0


def _download(spec: GpuWheelSpec, dest: Path, progress, is_cancelled) -> None:
    """依次尝试镜像，成功后按官方 sha256 校验。"""
    errors: list[str] = []
    for url in spec.mirrors:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ALIGN-gpu-installer"})
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or spec.size)
                done = 0
                started = time.monotonic()
                last_report = 0.0
                with open(dest, "wb") as handle:
                    while True:
                        if is_cancelled():
                            raise InstallCancelled("已取消下载")
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        handle.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last_report > 0.25:
                            last_report = now
                            elapsed = max(now - started, 1e-6)
                            progress(
                                spec.package,
                                done,
                                total,
                                done / elapsed,
                            )
        except InstallCancelled:
            dest.unlink(missing_ok=True)
            raise
        except Exception as exc:  # 换下一个镜像
            errors.append(f"{url.split('/')[2]}: {type(exc).__name__}")
            dest.unlink(missing_ok=True)
            continue
        digest = _sha256_file(dest)
        if digest != spec.sha256:
            dest.unlink(missing_ok=True)
            errors.append(f"{url.split('/')[2]}: 校验值不匹配")
            continue
        progress(spec.package, spec.size, spec.size, 0.0)
        return
    raise InstallError(f"{spec.package} 下载失败（{'; '.join(errors) or '未知原因'}）")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_wheel(wheel: Path, prefix: str, target: Path, progress, is_cancelled) -> int:
    """把 wheel 里 ``prefix`` 下的文件摊到 target，返回写出的文件数。"""
    written = 0
    with zipfile.ZipFile(wheel) as archive:
        members = [m for m in archive.infolist() if m.filename.startswith(prefix)]
        total = len(members) or 1
        for index, member in enumerate(members, start=1):
            if is_cancelled():
                raise InstallCancelled("已取消解包")
            relative = member.filename[len(prefix):]
            if not relative or member.is_dir():
                continue
            destination = (target / relative).resolve()
            if not str(destination).startswith(str(target.resolve())):
                raise InstallError("压缩包内容异常（路径越界），已中止")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 512)
            written += 1
            if index % 40 == 0 or index == total:
                progress("解包", index, total, 0.0)
    return written


def install_gpu_runtime(
    progress: Callable[[str, int, int, float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    wheels: Iterable[GpuWheelSpec] = GPU_WHEELS,
) -> Path:
    """下载并安装 CUDA 运行时，返回安装目录。

    ``progress(stage, done, total, bytes_per_second)`` 用于界面进度条；
    ``is_cancelled()`` 返回 True 时抛出 :class:`InstallCancelled`。
    失败时清理干净，不会留下半装状态（下次启动不会误挂载）。
    """
    progress = progress or (lambda *args, **kwargs: None)
    is_cancelled = is_cancelled or (lambda: False)

    required = sum(spec.size for spec in wheels)
    root = gpu_runtime_dir()
    root.parent.mkdir(parents=True, exist_ok=True)
    free = _disk_free_bytes(root.parent)
    # 解包后大约膨胀 1.4 倍，留 1.8 倍余量
    if free and free < required * 1.8:
        raise InstallError(
            f"磁盘空间不足：需要约 {required * 1.8 / 1e9:.1f} GB，"
            f"当前可用 {free / 1e9:.1f} GB。请清理后再试。"
        )

    staging = Path(tempfile.mkdtemp(prefix="align-gpu-", dir=str(root.parent)))
    try:
        for spec in wheels:
            if is_cancelled():
                raise InstallCancelled("已取消")
            wheel_path = staging / spec.filename
            _download(spec, wheel_path, progress, is_cancelled)
            progress("解包", 0, 1, 0.0)
            # wheel 里的成员是 "torch/xxx"，先剥掉前缀再摊到 root/torch 下
            _extract_wheel(
                wheel_path, spec.extract_prefix, root / spec.package, progress, is_cancelled
            )
            wheel_path.unlink(missing_ok=True)

        marker = root / MARKER_FILENAME
        marker.write_text(
            '{\n  "torch": "2.9.1+cu128",\n  "torchvision": "0.24.1+cu128",'
            '  "cuda": "12.8",\n  "python": "3.10",\n  "platform": "win_amd64"\n}\n',
            encoding="utf-8",
        )
        return root
    except BaseException:
        # 半成品目录必须清掉：否则下次启动会挂载一个残缺的 torch
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def remove_gpu_runtime() -> bool:
    """删除已安装的 GPU 运行时；返回是否真的删了东西。"""
    root = gpu_runtime_dir()
    if not root.exists():
        return False
    deactivate()
    shutil.rmtree(root, ignore_errors=True)
    return not root.exists()
