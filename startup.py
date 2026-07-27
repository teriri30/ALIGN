#!/usr/bin/env python
"""
startup.py — 环境自检 + 一键安装模块

仅使用 Python 标准库，在导入任何第三方包之前运行。
检测核心依赖是否齐全，缺失时弹出 tkinter 对话框让用户一键安装。
"""
import importlib
from importlib import metadata
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
from runtime_env import configure_geospatial_environment


configure_geospatial_environment()

PIP_INSTALL_TIMEOUT_SECONDS = int(os.environ.get("REUI_PIP_TIMEOUT_SECONDS", "1800"))

# ── 依赖清单 ──────────────────────────────────────────────
# 版本唯一来源为 requirements-core.txt；这里只维护显示名、导入名和 conda 名。
_DEPENDENCY_METADATA = {
    "numpy": ("NumPy", "numpy", "numpy"),
    "opencv-python": ("OpenCV", "cv2", "opencv"),
    "Pillow": ("Pillow", "PIL", "pillow"),
    "PySide6": ("PySide6", "PySide6", None),
    "pyproj": ("PyProj", "pyproj", "pyproj"),
    "rasterio": ("Rasterio", "rasterio", "rasterio"),
    "shapely": ("Shapely", "shapely", "shapely"),
    "affine": ("Affine", "affine", "affine"),
    "scipy": ("SciPy", "scipy", "scipy"),
}


def _core_requirements():
    path = Path(__file__).with_name("requirements-core.txt")
    packages = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        spec = raw.strip()
        if not spec or spec.startswith("#"):
            continue
        distribution = spec.split("==", 1)[0]
        if distribution not in _DEPENDENCY_METADATA:
            continue
        display_name, import_name, conda_name = _DEPENDENCY_METADATA[distribution]
        packages.append((display_name, import_name, spec, conda_name))
    return packages


DEPENDENCIES = _core_requirements() + [
    ("Ultralytics", "ultralytics", "ultralytics==8.3.163", None),
]


def check_missing() -> list:
    """逐个 import 检测，返回缺失项列表。"""
    missing = []
    for name, import_name, pip_pkg, conda_pkg in DEPENDENCIES:
        try:
            importlib.import_module(import_name)
            if "==" in pip_pkg:
                distribution, expected = pip_pkg.split("==", 1)
                installed = metadata.version(distribution)
                if installed != expected:
                    missing.append((name, import_name, pip_pkg, conda_pkg))
        except ImportError:
            missing.append((name, import_name, pip_pkg, conda_pkg))
        except metadata.PackageNotFoundError:
            missing.append((name, import_name, pip_pkg, conda_pkg))
    return missing


def get_pip_python() -> str:
    """返回当前 Python 解释器路径，供 subprocess 调用 pip 使用。"""
    return sys.executable


def get_pip_command(packages: list[str]) -> list[str]:
    """构造 pip install 命令。"""
    cmd = [get_pip_python(), "-m", "pip", "install"]
    # 如果检测到清华镜像源环境变量或系统位于中国大陆，可添加镜像
    # 但不自动添加，由用户自行决定
    cmd.extend(packages)
    return cmd


# ── GUI 对话框 ────────────────────────────────────────────

class EnvSetupDialog(tk.Toplevel):
    """环境安装对话框"""

    PADDING = 12
    BTN_WIDTH = 14

    def __init__(self, parent: tk.Tk, missing: list):
        super().__init__(parent)
        self.parent = parent
        self.missing = missing  # [(name, import_name, pip_pkg, conda_pkg), ...]
        self.result = False  # True = 继续启动, False = 退出

        self.title("环境安装 - 智能农机规划系统")
        self.geometry("680x520")
        self.minsize(600, 450)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.transient(parent)
        self.grab_set()

        # ── 安装状态 ──
        self._installing = False
        self._install_success = False  # 是否有包安装成功（用于重启提示）
        self._install_proc = None

        self._build_ui()
        self._populate_list()
        self.center_on_parent()
        self.wait_window()

    # ── UI 构建 ──

    def _build_ui(self):
        # ── 顶部说明 ──
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=self.PADDING, pady=(self.PADDING, 4))
        ttk.Label(header, text="🔧 环境检测", font=("", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(header, text="检测到以下依赖缺失，请选择要安装的包后点击「安装选中项」",
                  wraplength=600).pack(anchor=tk.W, pady=(2, 0))

        # ── 主区域：左侧列表 + 右侧日志 ──
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=self.PADDING, pady=4)

        # 左侧：缺失包列表
        left_frame = ttk.LabelFrame(main_frame, text="缺失的依赖包", width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 6))
        left_frame.pack_propagate(False)

        self._listbox = tk.Listbox(left_frame, selectmode=tk.NONE,
                                    activestyle="none", font=("Consolas", 10))
        self._listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 全选 / 全不选按钮
        sel_frame = ttk.Frame(left_frame)
        sel_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(sel_frame, text="全选", command=self._select_all,
                   width=8).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(sel_frame, text="全不选", command=self._select_none,
                   width=8).pack(side=tk.LEFT)

        # 右侧：日志区
        right_frame = ttk.LabelFrame(main_frame, text="安装日志")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._log = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD,
                                               font=("Consolas", 9), height=10,
                                               state=tk.DISABLED, bg="#1e1e1e",
                                               fg="#d4d4d4", insertbackground="white")
        self._log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── 进度条 ──
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill=tk.X, padx=self.PADDING, pady=(0, 4))
        self._progress = ttk.Progressbar(progress_frame, mode="indeterminate")
        self._progress.pack(fill=tk.X)
        self._progress_label = ttk.Label(progress_frame, text="", font=("", 9))
        self._progress_label.pack(anchor=tk.W)

        # ── 按钮栏 ──
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=self.PADDING, pady=(0, self.PADDING))

        ttk.Button(btn_frame, text="一键安装全部", command=self._install_all,
                   width=self.BTN_WIDTH).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="安装选中项", command=self._install_selected,
                   width=self.BTN_WIDTH).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(btn_frame, text="跳过，继续启动", command=self._on_skip,
                   width=self.BTN_WIDTH).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_frame, text="退出程序", command=self._on_close,
                   width=self.BTN_WIDTH).pack(side=tk.RIGHT)

        # ── 提示信息 ──
        hint_frame = ttk.Frame(self)
        hint_frame.pack(fill=tk.X, padx=self.PADDING, pady=(0, 6))
        self._hint_label = ttk.Label(
            hint_frame,
            text="提示: 某些包（如 rasterio, pyproj）可能需要 C 编译器，"
                 "若 pip 安装失败请尝试: conda install -c conda-forge <包名>",
            wraplength=640, font=("", 8), foreground="gray")
        self._hint_label.pack(anchor=tk.W)

    def _populate_list(self):
        self._listbox.delete(0, tk.END)
        self._checkboxes = {}  # index -> (name, import_name, pip_pkg, conda_pkg, var)
        for i, (name, imp, pip_pkg, conda_pkg) in enumerate(self.missing):
            var = tk.BooleanVar(value=True)
            self._checkboxes[i] = (name, imp, pip_pkg, conda_pkg, var)
            self._listbox.insert(tk.END, f"  ☑  {name}  ({pip_pkg})")
        if not self.missing:
            self._listbox.insert(tk.END, "  ✅ 所有依赖已就绪")

    def _refresh_listbox_labels(self):
        """根据当前 checkbox 状态更新列表文字。"""
        for i in range(self._listbox.size()):
            info = self._checkboxes.get(i)
            if info is None:
                continue
            name, imp, pip_pkg, conda_pkg, var = info
            mark = "☑" if var.get() else "☐"
            status = ""
            self._listbox.delete(i)
            self._listbox.insert(i, f"  {mark}  {name}  ({pip_pkg})")
            self._listbox.selection_clear(0, tk.END)

    def _select_all(self):
        for info in self._checkboxes.values():
            info[4].set(True)
        self._refresh_listbox_labels()

    def _select_none(self):
        for info in self._checkboxes.values():
            info[4].set(False)
        self._refresh_listbox_labels()

    # ── 窗口管理 ──

    def center_on_parent(self):
        self.update_idletasks()
        pw, ph = self.parent.winfo_width(), self.parent.winfo_height()
        px, py = self.parent.winfo_x(), self.parent.winfo_y()
        w, h = self.winfo_width(), self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{max(0,x)}+{max(0,y)}")

    # ── 安装逻辑 ──

    def _log_write(self, text: str):
        """安全地在主线程向日志区追加文本。"""
        def _do():
            self._log.configure(state=tk.NORMAL)
            self._log.insert(tk.END, text)
            self._log.see(tk.END)
            self._log.configure(state=tk.DISABLED)
        self.after(0, _do)

    def _set_progress(self, active: bool, label: str = ""):
        def _do():
            if active:
                self._progress.start(15)
            else:
                self._progress.stop()
            self._progress_label.configure(text=label)
        self.after(0, _do)

    def _install_all(self):
        """安装全部缺失包。"""
        pkgs = [info[2] for info in self._checkboxes.values()]
        if pkgs:
            self._run_install(pkgs)

    def _install_selected(self):
        """安装用户选中的包。"""
        pkgs = [info[2] for info in self._checkboxes.values() if info[4].get()]
        if not pkgs:
            messagebox.showinfo("提示", "请至少勾选一个要安装的包", parent=self)
            return
        self._run_install(pkgs)

    def _run_install(self, packages: list[str]):
        """在子线程中执行 pip install。"""
        if self._installing:
            return
        self._installing = True
        self._set_progress(True, "正在安装...")

        cmd = get_pip_command(packages)
        self._log_write(f"$ {' '.join(cmd)}\n{'─' * 60}\n")

        def _worker():
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                )
                self._install_proc = proc

                def _read_stdout():
                    try:
                        for line in proc.stdout:
                            self._log_write(line)
                    except Exception:
                        pass

                reader = threading.Thread(target=_read_stdout, daemon=True)
                reader.start()
                try:
                    proc.wait(timeout=PIP_INSTALL_TIMEOUT_SECONDS)
                    reader.join(timeout=2.0)
                    returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    self._log_write(
                        f"\n[错误] pip 安装超过 {PIP_INSTALL_TIMEOUT_SECONDS} 秒，已终止。\n"
                    )
                    proc.kill()
                    reader.join(timeout=2.0)
                    returncode = -2
                self.after(0, lambda rc=returncode: self._on_install_done(rc))
            except Exception as e:
                self._log_write(f"\n[错误] 启动 pip 安装失败: {e}\n")
                self.after(0, lambda: self._on_install_done(-1))
            finally:
                if self._install_proc is proc:
                    self._install_proc = None

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _on_install_done(self, returncode: int):
        """安装完成回调。"""
        self._installing = False
        self._set_progress(False)

        if returncode == 0:
            self._install_success = True
            self._log_write(f"\n{'─' * 60}\n✅ 安装完成！重新检测依赖...\n")

            # 重新检测
            still_missing = check_missing()
            # 过滤掉不在当前对话框列表中的包
            current_missing_names = {info[1] for info in self._checkboxes.values()}
            new_missing = [(n, im, pip, conda) for n, im, pip, conda in still_missing
                           if im in current_missing_names]

            if new_missing:
                self._log_write(f"\n⚠ 以下包安装后仍不可用:\n")
                for n, im, pip, conda in new_missing:
                    hint = f"  → {n} ({im}): 尝试 conda install -c conda-forge {conda or pip}"
                    self._log_write(hint + "\n")
                self._hint_label.configure(
                    text="部分包安装失败，请尝试手动使用 conda 安装（见日志中的命令）",
                    foreground="red")
            else:
                self._log_write(f"\n✅ 所有依赖已就绪！\n")
                # 安装成功后，将按钮改为"启动程序"
                self._show_launch_ui()
        else:
            self._log_write(f"\n❌ pip 返回错误码 {returncode}，请检查日志\n")

    def _show_launch_ui(self):
        """全部安装成功后：替换按钮为启动。"""
        # 清空按钮栏重建
        for w in self.winfo_children():
            if isinstance(w, ttk.Frame):
                for child in w.winfo_children():
                    if isinstance(child, ttk.Frame):
                        # 找到按钮栏
                        for btn in child.winfo_children():
                            if isinstance(btn, ttk.Button):
                                btn.destroy()
                        ttk.Button(child, text="🚀 启动程序",
                                   command=self._on_launch,
                                   width=self.BTN_WIDTH).pack(
                            side=tk.RIGHT, padx=(6, 0))
                        break

        # 更新顶部提示
        self._hint_label.configure(
            text="✅ 环境已就绪！点击「启动程序」继续",
            foreground="green", font=("", 9, "bold"))

    # ── 按钮事件 ──

    def _on_skip(self):
        """用户选择跳过安装，直接启动。"""
        self.result = True
        self.destroy()

    def _on_close(self):
        """用户关闭窗口或点击退出。"""
        if self._installing:
            if not messagebox.askyesno("确认退出",
                    "正在安装中，确定要退出吗？", parent=self):
                return
            proc = self._install_proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.result = False
        self.destroy()

    def _on_launch(self):
        """安装完成后的启动。"""
        self.result = True
        self.destroy()


# ── 主入口 ────────────────────────────────────────────────

def ensure_environment() -> bool:
    """
    环境自检主入口。

    检测所有依赖，若全部存在则立即返回 True。
    若有缺失则弹出安装对话框，返回用户的选择：
        True  → 继续启动程序
        False → 退出程序
    """
    try:
        missing = check_missing()
    except Exception as e:
        print(f"[环境检测] check_missing 异常: {e}")
        return True  # 出错时不阻塞启动

    if not missing:
        return True

    # ── 创建隐藏根窗口作为对话框父窗口 ──
    root = tk.Tk()
    root.withdraw()
    root.title("环境安装")

    # 设置窗口图标（如果有）
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception:
            pass

    # ── 弹出对话框 ──
    dialog = EnvSetupDialog(root, missing)
    root.destroy()
    return dialog.result


if __name__ == "__main__":
    # 命令行自检模式
    missing = check_missing()
    if missing:
        print("缺失的依赖:")
        for n, imp, pip_pkg, conda_pkg in missing:
            print(f"  {n:15s}  pip install {pip_pkg}")
        print()
        ans = input("是否一键安装？[Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            pkgs = [pkg for _, _, pkg, _ in missing]
            cmd = get_pip_command(pkgs)
            print(f"\n$ {' '.join(cmd)}\n")
            subprocess.run(cmd, check=True, timeout=PIP_INSTALL_TIMEOUT_SECONDS)
            print("\n✅ 安装完成！")
        else:
            print("跳过安装。")
    else:
        print("✅ 所有依赖已就绪！")
