#!/usr/bin/env python
"""startup.py — 源码启动的运行环境初始化。

仅使用 Python 标准库，在导入任何第三方包之前完成两件事：

1. 隔离父进程注入的 ``PYTHONPATH``，并把用户 site-packages 排在
   解释器 site-packages 之前（与 Python 默认 sys.path 语义一致）；
2. 配置 GDAL/PROJ 地理环境变量（``runtime_env``）。

DECISION-018 v2：依赖检测与一键安装界面已移除；源码入口只做上述初始化，
发布版（``align.exe``）由 ``frozen_main.py`` 进入，不导入本模块。
"""
import os
import site
import sys
import sysconfig
from pathlib import Path

from runtime_env import configure_geospatial_environment


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path or "")))


def _prioritize_active_environment_paths() -> None:
    """Isolate this interpreter from packages injected by a parent process.

    Hermes and some IDE launchers export their own ``PYTHONPATH``. Reordering
    those entries is insufficient: if this environment lacks a package, Python
    will still import a binary extension built for another Python ABI. Keep the
    project directory, then the active/user package directories, and discard
    inherited path entries from the application process.
    """
    if not sys.path:
        return
    project_dir = str(Path(__file__).resolve().parent)
    inherited = {
        _path_key(path)
        for path in str(os.environ.get("PYTHONPATH", "")).split(os.pathsep)
        if path
    }
    inherited.discard(_path_key(project_dir))
    if inherited:
        sys.path[:] = [
            path for path in sys.path
            if _path_key(path) not in inherited
        ]
    os.environ.pop("PYTHONPATH", None)

    # 用户 site-packages 优先于解释器自身的 site-packages，与 Python 默认
    # 的 sys.path 语义一致。此前把 purelib/platlib 排在 user_site 前面，
    # 导致 game 环境（conda，不可写）中 BLAS 动态库损坏的 numpy 覆盖了
    # user-site 里自带的正常 numpy，任何矩阵乘法（@ / np.linalg）都会在
    # 原生层崩溃（0xC06D007F）。
    active_paths = []
    try:
        user_site = site.getusersitepackages()
    except (AttributeError, OSError):
        user_site = ""
    if isinstance(user_site, str) and user_site:
        active_paths.append(user_site)
    for scheme in ("purelib", "platlib"):
        try:
            candidate = sysconfig.get_path(scheme)
        except (KeyError, TypeError):
            candidate = ""
        if candidate:
            active_paths.append(candidate)

    active_keys = {_path_key(path) for path in active_paths}
    if not active_keys:
        return
    first = sys.path[0]
    remaining = list(sys.path[1:])
    by_key = {_path_key(path): path for path in remaining if path}
    prioritized = []
    seen = set()
    for path in active_paths:
        key = _path_key(path)
        if key in by_key and key not in seen:
            prioritized.append(by_key[key])
            seen.add(key)
    sys.path[:] = [first] + prioritized + [
        path for path in remaining if _path_key(path) not in seen
    ]


_prioritize_active_environment_paths()
configure_geospatial_environment()
