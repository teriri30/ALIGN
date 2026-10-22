from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QTextEdit, QListWidget, QListWidgetItem, QFormLayout, QDoubleSpinBox,
    QSpinBox, QDialogButtonBox, QTabWidget, QWidget, QFileDialog, QCheckBox,
    QComboBox, QMessageBox
)
from pyside6_app.styles import COLORS
from config import save_json_atomic, DEFAULT_HARVESTER_PARAMS
from runtime_paths import user_config_path
import device_runtime
import os


class _DeviceStatusWorker(QThread):
    """后台探测显卡与 torch 状态：导入 torch 要几秒，不能卡界面。"""

    done = Signal(dict)

    def run(self):  # noqa: D102
        info = {}
        try:
            info = device_runtime.torch_status()
        except Exception as exc:  # pragma: no cover - 环境损坏
            info = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            info["gpu"] = device_runtime.detect_nvidia_gpu()
            info["installed"] = device_runtime.gpu_runtime_installed()
            info["size"] = device_runtime.gpu_runtime_size()
        except Exception:  # pragma: no cover
            info.setdefault("gpu", None)
            info.setdefault("installed", False)
            info.setdefault("size", 0)
        self.done.emit(info)


class _GpuInstallWorker(QThread):
    """下载 + 解包 CUDA 运行时；下载在子线程，界面不冻结。"""

    progress = Signal(str, int, int, float)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):  # noqa: D102
        try:
            path = device_runtime.install_gpu_runtime(
                progress=lambda stage, done, total, speed: self.progress.emit(
                    stage, done, total, speed
                ),
                is_cancelled=lambda: self._cancel,
            )
            self.finished_ok.emit(str(path))
        except device_runtime.InstallCancelled:
            self.failed.emit("已取消，未做任何改动")
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)


class LoadingProgress(QDialog):
    def __init__(self, title="处理中", message="请稍候...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(380, 140)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        lo = QVBoxLayout(self)
        lo.setSpacing(12)
        self.lbl_message = QLabel(message)
        self.lbl_message.setStyleSheet(f"font-size:14px;color:{COLORS['text']};")
        lo.addWidget(self.lbl_message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        lo.addWidget(self.progress)
        self.lbl_detail = QLabel("")
        self.lbl_detail.setStyleSheet(f"font-size:12px;color:{COLORS['text_dim']};")
        lo.addWidget(self.lbl_detail)

    def set_progress(self, val, detail=""):
        self.progress.setValue(int(val))
        if detail:
            self.lbl_detail.setText(str(detail))


class PaperExperimentDialog(QDialog):
    """Configure a reproducible current-project paper experiment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("论文消融实验中心")
        self.resize(620, 480)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "对当前项目快照运行掩膜处理阶段对比。"
            "当前论文范围不包含转弯策略优化或策略对比；实验不会覆盖当前界面结果。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.mask_check = QCheckBox("掩膜处理对比：原始分割掩膜 vs 完整规则化")
        self.mask_check.setChecked(True)
        layout.addWidget(self.mask_check)

        truth_row = QHBoxLayout()
        self.truth_path = ""
        self.truth_label = QLabel("未提供人工真值（不计算 IoU、Dice、Precision、Recall）")
        self.truth_label.setWordWrap(True)
        truth_button = QPushButton("选择真值掩膜…")
        truth_button.clicked.connect(self._choose_truth)
        truth_row.addWidget(self.truth_label, 1)
        truth_row.addWidget(truth_button)
        layout.addLayout(truth_row)

        model_note = QLabel(
            "多模型对比：为避免把当前模型掩膜冒充其他模型结果，本入口暂不执行批量模型推理。"
            "模型间对比应在相同数据划分和原始 GeoTIFF 推理设置下单独运行。"
        )
        model_note.setWordWrap(True)
        model_note.setStyleSheet(f"color:{COLORS['text_dim']};")
        layout.addWidget(model_note)
        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("选择位置并开始")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_truth(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择人工真值二值掩膜", "", "Mask image (*.png *.tif *.tiff *.bmp)"
        )
        if path:
            self.truth_path = path
            self.truth_label.setText(path)

    def options(self):
        return {
            "include_mask": self.mask_check.isChecked(),
            # Kept for callers that read the old options shape. The current
            # paper scope intentionally does not run turn-strategy cases.
            "include_path": False,
            "ground_truth_path": self.truth_path,
        }

    def accept(self):
        if not self.mask_check.isChecked():
            QMessageBox.warning(self, "没有实验组", "请勾选掩膜处理实验。")
            return
        super().accept()


class HarvesterParamsDialog(QDialog):
    def __init__(self, params=None, parent=None, source="user_entered"):
        super().__init__(parent)
        self.setWindowTitle("农机参数设置")
        self.setFixedSize(460, 420)
        self._params = params or {}
        self._source = str(source or "user_entered")
        self._build_ui()

    def _spin(self, value, minimum, maximum, step):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setSuffix(" m")
        spin.setValue(float(value))
        spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
        return spin

    def _speed_spin(self, value):
        spin = self._spin(value, 0.05, 5.0, 0.05)
        spin.setSuffix(" m/s")
        return spin

    def _build_ui(self):
        lo = QVBoxLayout(self)
        lo.setSpacing(12)
        lbl = QLabel(
            "请设置收割机的物理参数，用于等距区域路径和理论履带碾压计算。"
            "宽窄行中心线仍以掩膜/行带几何为准。"
        )
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{COLORS['text_dim']};font-size:13px;")
        lo.addWidget(lbl)
        fm = QFormLayout()
        fm.setSpacing(8)
        fm.setContentsMargins(0, 8, 0, 8)
        self.spin_cutter = self._spin(
            self._params.get(
                "cutter_width_m", DEFAULT_HARVESTER_PARAMS["cutter_width_m"]
            ),
            0.5,
            6.0,
            0.1,
        )
        fm.addRow(QLabel("割台有效割幅:"), self.spin_cutter)
        offset_value = self._params.get("cutter_forward_offset_m")
        self.spin_cutter_offset = self._spin(
            float(offset_value) if offset_value not in (None, "") else 0.0,
            0.0,
            10.0,
            0.05,
        )
        self.spin_cutter_offset.setSpecialValueText("未设置")
        self.spin_cutter_offset.setToolTip(
            "履带接地区域几何中心到割刀切割线的水平前向距离，必须实测"
        )
        fm.addRow(QLabel("割刀前置距离:"), self.spin_cutter_offset)
        self.spin_track = self._spin(
            self._params.get(
                "track_width_m", DEFAULT_HARVESTER_PARAMS["track_width_m"]
            ),
            0.1,
            1.0,
            0.05,
        )
        fm.addRow(QLabel("履带宽度:"), self.spin_track)
        self.spin_gauge = self._spin(
            self._params.get(
                "track_gauge_m", DEFAULT_HARVESTER_PARAMS["track_gauge_m"]
            ),
            0.5,
            3.0,
            0.1,
        )
        fm.addRow(QLabel("履带中心距:"), self.spin_gauge)
        self.spin_track_length = self._spin(
            self._params.get(
                "track_length_m", DEFAULT_HARVESTER_PARAMS["track_length_m"]
            ),
            0.5,
            6.0,
            0.1,
        )
        fm.addRow(QLabel("履带接地长度:"), self.spin_track_length)
        self.spin_turn = self._spin(self._params.get("turn_radius_m", 2.0), 0.5, 6.0, 0.1)
        fm.addRow(QLabel("最小转弯半径:"), self.spin_turn)
        self.spin_work_speed = self._speed_spin(self._params.get("work_speed_mps", 1.2))
        fm.addRow(QLabel("作业速度:"), self.spin_work_speed)
        self.spin_turn_speed = self._speed_spin(self._params.get("turn_speed_mps", 0.6))
        fm.addRow(QLabel("转弯速度:"), self.spin_turn_speed)
        self.spin_entry_speed = self._speed_spin(self._params.get("entry_speed_mps", 0.8))
        fm.addRow(QLabel("进出场速度:"), self.spin_entry_speed)
        lo.addLayout(fm)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lo.addWidget(bb)

    def get_params(self):
        return {
            "cutter_width_m": self.spin_cutter.value(),
            "cutter_forward_offset_m": (
                self.spin_cutter_offset.value()
                if self.spin_cutter_offset.value() > 0
                else None
            ),
            "track_width_m": self.spin_track.value(),
            "track_gauge_m": self.spin_gauge.value(),
            "turn_radius_m": self.spin_turn.value(),
            "track_length_m": self.spin_track_length.value(),
            "work_speed_mps": self.spin_work_speed.value(),
            "turn_speed_mps": self.spin_turn_speed.value(),
            "entry_speed_mps": self.spin_entry_speed.value(),
        }
    def accept(self):
        if self.spin_cutter_offset.value() <= 0:
            QMessageBox.warning(
                self,
                "缺少实测参数",
                "请填写履带接地区域几何中心到割刀切割线的水平前向距离。\n"
                "系统不会根据机身尺寸推算该值。",
            )
            self.spin_cutter_offset.setFocus()
            return
        super().accept()


class MachineRecommendationDialog(QDialog):
    def __init__(self, recommendation, parent=None):
        super().__init__(parent)
        self.setWindowTitle("启发式农机参数候选")
        self.resize(560, 460)
        self._recommendation = recommendation or {}
        self._build_ui()

    def _build_ui(self):
        lo = QVBoxLayout(self)
        lo.setSpacing(10)
        title = QLabel(self._recommendation.get("title", "启发式农机参数候选"))
        title.setStyleSheet(f"font-size:16px;font-weight:600;color:{COLORS['text_bright']};")
        lo.addWidget(title)

        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(self._recommendation.get("message", "暂无建议。"))
        lo.addWidget(edit, 1)

        row = QHBoxLayout()
        btn_apply = QPushButton("应用候选参数")
        btn_close = QPushButton("关闭")
        btn_apply.setEnabled(
            bool(self._recommendation.get("params"))
            and self._recommendation.get("evidence_quality") != "insufficient"
        )
        if not btn_apply.isEnabled() and bool(self._recommendation.get("params")):
            btn_apply.setToolTip("当前影像证据不足，不能形成可应用的候选尺寸")
        elif btn_apply.isEnabled() and not bool(
            (self._recommendation.get("params") or {}).get("cutter_forward_offset_m")
        ):
            btn_apply.setToolTip("只应用履带、中心距、轴距等候选尺寸；割刀前置距离仍需实测")
        btn_apply.clicked.connect(self.accept)
        btn_close.clicked.connect(self.reject)
        row.addStretch()
        row.addWidget(btn_apply)
        row.addWidget(btn_close)
        lo.addLayout(row)

    def get_params(self):
        return dict(self._recommendation.get("params") or {})


def _model_display_name(model):
    path = model.get("path", "") if isinstance(model, dict) else str(model or "")
    raw_name = model.get("name", "") if isinstance(model, dict) else ""
    base = os.path.basename(path or raw_name)
    parent = os.path.basename(os.path.dirname(os.path.dirname(path))) if path else ""
    # YOLO runs usually store weights as <run>/weights/best.pt; show run · best.pt.
    if os.path.basename(os.path.dirname(path)).lower() == "weights":
        run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if run_name:
            return f"{run_name} · {base}"
    return raw_name or base or path


def _is_preferred_model(model):
    path = model.get("path", "") if isinstance(model, dict) else str(model or "")
    return os.path.basename(path).lower() == "best.pt"


def _is_last_model(model):
    path = model.get("path", "") if isinstance(model, dict) else str(model or "")
    return os.path.basename(path).lower() == "last.pt"


class ModelImportDialog(QDialog):
    """Let users review recursively discovered .pt model files before adding them."""
    def __init__(self, models, parent=None, best_only=True):
        super().__init__(parent)
        self.setWindowTitle("导入模型文件夹")
        self.resize(620, 460)
        self._models = models or []
        self._best_only = bool(best_only)
        self.selected_models = []
        self._build_ui()

    def _build_ui(self):
        lo = QVBoxLayout(self)
        tip = QLabel("已在所选文件夹及其子文件夹中找到以下 .pt 权重。默认只勾选 best.pt；last.pt 默认不选。")
        tip.setWordWrap(True)
        lo.addWidget(tip)
        self.list_widget = QListWidget()
        for model in self._models:
            name = _model_display_name(model)
            path = model.get("path", "")
            item = QListWidgetItem(f"{name}\n{path}")
            item.setToolTip(path)
            item.setData(Qt.UserRole, model)
            checked = _is_preferred_model(model) or (not self._best_only and not _is_last_model(model))
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.list_widget.addItem(item)
        lo.addWidget(self.list_widget, 1)
        row = QHBoxLayout()
        btn_all = QPushButton("全选")
        btn_none = QPushButton("全不选")
        btn_ok = QPushButton("加入选中模型")
        btn_cancel = QPushButton("取消")
        btn_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        btn_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        btn_ok.clicked.connect(self._accept_selected)
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_all); row.addWidget(btn_none); row.addStretch(); row.addWidget(btn_ok); row.addWidget(btn_cancel)
        lo.addLayout(row)

    def _set_all(self, state):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def _accept_selected(self):
        self.selected_models = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                self.selected_models.append(item.data(Qt.UserRole))
        self.accept()


class ModelManagerDialog(QDialog):
    def __init__(self, models, cur="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型管理")
        self.setFixedSize(500, 420)
        self._models = models or []
        self._current_path = cur or ""
        self.selected_path = None
        self._build_ui()

    def _build_ui(self):
        lo = QVBoxLayout(self)
        lo.setSpacing(8)
        lo.addWidget(QLabel("选择 YOLO 模型权重文件 (.pt)："))
        self.list_widget = QListWidget()
        for model in self._models:
            name = model.get("name") or os.path.basename(model.get("path", ""))
            path = model.get("path", "")
            item = QListWidgetItem(f"{name}  ({os.path.basename(path) if path else '未指定'})")
            item.setData(Qt.UserRole, path)
            if path == self._current_path:
                item.setSelected(True)
            self.list_widget.addItem(item)
        lo.addWidget(self.list_widget)
        row = QHBoxLayout()
        btn_add = QPushButton("添加模型文件")
        btn_folder = QPushButton("导入模型文件夹")
        btn_ok = QPushButton("确定")
        btn_cancel = QPushButton("取消")
        btn_add.clicked.connect(self._add_model)
        btn_folder.clicked.connect(self._add_model_folder)
        btn_ok.clicked.connect(self._accept_selected)
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_add)
        row.addWidget(btn_folder)
        row.addStretch()
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        lo.addLayout(row)

    def _add_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "PyTorch (*.pt);;All files (*)")
        if path:
            item = QListWidgetItem(f"{os.path.basename(path)}  ({os.path.basename(path)})")
            item.setData(Qt.UserRole, path)
            self.list_widget.addItem(item)
            self.list_widget.setCurrentItem(item)


    def _add_model_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模型文件夹", "")
        if not folder:
            return
        from inference import scan_model_dir
        found = scan_model_dir(folder)
        if not found:
            item = QListWidgetItem("未在所选文件夹中找到 .pt 模型")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.list_widget.addItem(item)
            return
        dlg = ModelImportDialog(found, self, best_only=True)
        if dlg.exec() == QDialog.Accepted:
            existing = {self.list_widget.item(i).data(Qt.UserRole) for i in range(self.list_widget.count())}
            first_new = None
            for model in dlg.selected_models:
                path = model.get("path", "")
                if not path or path in existing:
                    continue
                name = _model_display_name(model)
                item = QListWidgetItem(f"{name}  ({os.path.basename(path)})")
                item.setToolTip(path)
                item.setData(Qt.UserRole, path)
                self.list_widget.addItem(item)
                existing.add(path)
                if first_new is None:
                    first_new = item
            if first_new is not None:
                self.list_widget.setCurrentItem(first_new)

    def _accept_selected(self):
        item = self.list_widget.currentItem()
        if item:
            self.selected_path = item.data(Qt.UserRole)
        self.accept()


class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.resize(560, 420)
        self.setMinimumSize(520, 400)
        self._config = config or {}
        self._status_worker = None
        self._install_worker = None
        self._device_info = {}
        self._build_ui()

    def _build_ui(self):
        lo = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._general_tab(), "通用")
        tabs.addTab(self._device_tab(), "运行设备")
        tabs.addTab(self._path_tab(), "路径规划")
        tabs.addTab(self._mask_tab(), "掩膜处理")
        lo.addWidget(tabs)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lo.addWidget(bb)

    # ------------------------------------------------------------------ 运行设备
    def _device_tab(self):
        w = QWidget()
        lo = QVBoxLayout(w)
        lo.setSpacing(8)
        lo.setContentsMargins(14, 14, 14, 10)

        self.lbl_device_head = QLabel("正在检测…")
        self.lbl_device_head.setStyleSheet("font-size:15px;font-weight:600;")
        self.lbl_device_head.setWordWrap(True)
        lo.addWidget(self.lbl_device_head)

        self.lbl_device_detail = QLabel("")
        self.lbl_device_detail.setWordWrap(True)
        self.lbl_device_detail.setMinimumHeight(34)  # 两行：torch/显卡信息容易顶到右边缘
        self.lbl_device_detail.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;")
        lo.addWidget(self.lbl_device_detail)

        row = QHBoxLayout()
        self.btn_device_enable = QPushButton("启用 GPU 加速")
        self.btn_device_enable.clicked.connect(self._start_gpu_install)
        self.btn_device_remove = QPushButton("移除 GPU 运行时")
        self.btn_device_remove.clicked.connect(self._remove_gpu)
        self.btn_device_refresh = QPushButton("重新检测")
        self.btn_device_refresh.clicked.connect(self._refresh_device)
        row.addWidget(self.btn_device_enable)
        row.addWidget(self.btn_device_remove)
        row.addWidget(self.btn_device_refresh)
        row.addStretch(1)
        lo.addLayout(row)

        self.bar_device = QProgressBar()
        self.bar_device.setRange(0, 100)
        self.bar_device.setVisible(False)
        lo.addWidget(self.bar_device)

        self.lbl_device_hint = QLabel("")
        self.lbl_device_hint.setWordWrap(True)
        # 说明文字占满剩余高度：只给一行高会被裁掉后半句
        self.lbl_device_hint.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # 预留三行高度：最长的提示（有 N 卡待启用）正好三行，给少了会裁掉后半句
        self.lbl_device_hint.setMinimumHeight(56)
        self.lbl_device_hint.setMaximumHeight(88)
        self.lbl_device_hint.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;")
        lo.addWidget(self.lbl_device_hint)
        lo.addStretch(1)

        for btn in (self.btn_device_enable, self.btn_device_remove):
            btn.setVisible(False)
        self._refresh_device()
        return w

    def _refresh_device(self):
        if self._install_worker is not None and self._install_worker.isRunning():
            return
        self.lbl_device_head.setText("正在检测…")
        self.lbl_device_detail.setText("")
        self.btn_device_enable.setVisible(False)
        self.btn_device_remove.setVisible(False)
        self.btn_device_refresh.setEnabled(False)
        worker = _DeviceStatusWorker(self)
        worker.done.connect(self._apply_device_state)
        worker.finished.connect(lambda: self.btn_device_refresh.setEnabled(True))
        self._status_worker = worker
        worker.start()

    def _apply_device_state(self, info: dict):
        self._device_info = info or {}
        # 每次先全部收起，再由下面各个分支显式打开，避免上次状态残留
        self.btn_device_enable.setVisible(False)
        self.btn_device_remove.setVisible(False)
        gpu = info.get("gpu")
        installed = bool(info.get("installed"))
        size_gb = (info.get("size") or 0) / 1e9
        cuda_ready = bool(info.get("cuda_available"))

        detail_parts = []
        if info.get("available"):
            detail_parts.append(f"torch {info.get('version') or '未知'}")
            if info.get("cuda_build"):
                detail_parts.append(f"CUDA {info['cuda_build']}")
            detail_parts.append("CUDA 版" if info.get("cuda_build") else "CPU 版")
        elif info.get("error"):
            detail_parts.append(f"torch 加载失败：{info['error']}")
        if gpu:
            memory = gpu.get("memory_mb") or 0
            detail_parts.append(
                f"显卡 {gpu.get('name')}（{memory / 1024:.0f} GB，驱动 {gpu.get('driver')}）"
            )
        else:
            detail_parts.append("未检测到 NVIDIA 显卡")
        if installed:
            detail_parts.append(f"已安装 GPU 运行时 {size_gb:.1f} GB")
        self.lbl_device_detail.setText("；".join(detail_parts))

        if cuda_ready:
            self.lbl_device_head.setText(f"正在使用 GPU：{info.get('device_name') or 'CUDA 设备'}")
            self.lbl_device_hint.setText(
                "识别、掩膜和评价都会跑在显卡上，速度比 CPU 快十几倍。"
                "换回 CPU 可以点「移除 GPU 运行时」，随时能再装回来。"
            )
            self.btn_device_remove.setVisible(installed)
            return

        if gpu and not installed:
            self.lbl_device_head.setText("正在使用 CPU（检测到 NVIDIA 显卡，可以开 GPU 加速）")
            self.lbl_device_hint.setText(
                "点「启用 GPU 加速」后，程序会自己下载约 2.7 GB 的 CUDA 版 PyTorch 运行库"
                "（torch 2.9.1+cu128，从国内镜像取，装到 ALIGN 自己的数据目录），装完重启就生效。\n"
                "不装也完全能用，只是识别会慢一些。中途可以取消，取消不会留下垃圾文件。"
            )
            self.btn_device_enable.setText("启用 GPU 加速（约 2.7 GB）")
            self.btn_device_enable.setVisible(True)
            self.btn_device_remove.setVisible(installed)
            return

        if gpu and installed:
            self.lbl_device_head.setText("正在使用 CPU（GPU 运行时已装，但没能启用）")
            self.lbl_device_hint.setText(
                "常见原因是显卡驱动太旧，或这张卡不支持 CUDA 12.8。"
                "可以先更新 NVIDIA 驱动再点「重新检测」；确认用不了就点「移除 GPU 运行时」回收空间。"
            )
            self.btn_device_enable.setText("重新下载安装（约 2.7 GB）")
            self.btn_device_enable.setVisible(True)
            self.btn_device_remove.setVisible(True)
            return

        self.lbl_device_head.setText("正在使用 CPU")
        self.lbl_device_hint.setText(
            "这台电脑没有检测到 NVIDIA 显卡，CPU 模式就是最合适的选择，不需要做任何设置。"
            "程序里所有功能都能正常用。"
        )
        self.btn_device_remove.setVisible(installed)

    def _start_gpu_install(self):
        if self._install_worker is not None and self._install_worker.isRunning():
            self._install_worker.cancel()
            self.btn_device_enable.setEnabled(False)
            self.btn_device_enable.setText("正在取消…")
            return
        answer = QMessageBox.question(
            self,
            "启用 GPU 加速",
            "将下载约 2.7 GB 的 CUDA 版 PyTorch 运行库（含 CUDA 12.8 组件），安装到 ALIGN 的数据目录。\n"
            "占用磁盘约 4.3 GB，装完需要重启 ALIGN 才生效。\n\n现在开始吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        self.bar_device.setVisible(True)
        self.bar_device.setValue(0)
        self.btn_device_enable.setText("取消")
        self.btn_device_remove.setVisible(False)
        self.btn_device_refresh.setEnabled(False)

        worker = _GpuInstallWorker(self)
        worker.progress.connect(self._on_install_progress)
        worker.finished_ok.connect(self._on_install_done)
        worker.failed.connect(self._on_install_failed)
        self._install_worker = worker
        worker.start()

    def _on_install_progress(self, stage: str, done: int, total: int, speed: float):
        total = max(int(total or 0), 1)
        percent = int(min(100, done * 100 / total))
        self.bar_device.setValue(percent)
        if stage == "解包":
            self.lbl_device_hint.setText(f"正在解包（{percent}%），这一步大约 1-3 分钟，别关窗口…")
        else:
            self.lbl_device_hint.setText(
                f"正在下载 {stage}：{done / 1e6:.0f} / {total / 1e6:.0f} MB"
                f"（{speed / 1e6:.1f} MB/s）"
            )

    def _on_install_done(self, path: str):
        self.bar_device.setValue(100)
        self.bar_device.setVisible(False)
        self.btn_device_enable.setEnabled(True)
        self.btn_device_enable.setText("启用 GPU 加速（约 2.7 GB）")
        self.btn_device_refresh.setEnabled(True)
        self._install_worker = None
        QMessageBox.information(
            self,
            "装好了",
            "GPU 运行库已经装好。\n\n请关闭 ALIGN 再重新打开，"
            "之后这里会显示「正在使用 GPU」。",
        )
        self._refresh_device()

    def _on_install_failed(self, message: str):
        self.bar_device.setVisible(False)
        self.btn_device_enable.setEnabled(True)
        self.btn_device_enable.setText("启用 GPU 加速（约 2.7 GB）")
        self.btn_device_refresh.setEnabled(True)
        self._install_worker = None
        QMessageBox.warning(self, "没能装好", f"{message}\n\nCPU 模式不受影响，可以继续用。")
        self._refresh_device()

    def _remove_gpu(self):
        size_gb = (self._device_info.get("size") or device_runtime.gpu_runtime_size()) / 1e9
        answer = QMessageBox.question(
            self,
            "移除 GPU 运行时",
            f"删除已安装的 CUDA 运行库，释放约 {size_gb:.1f} GB 空间。\n"
            "ALIGN 会退回 CPU 模式，功能不受影响。继续吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        ok = device_runtime.remove_gpu_runtime()
        QMessageBox.information(
            self,
            "已移除" if ok else "没有可删的内容",
            "GPU 运行库已删除，下次启动用 CPU 模式。" if ok else "没找到已安装的 GPU 运行库。",
        )
        self._refresh_device()

    def closeEvent(self, event):  # noqa: N802
        for worker in (self._install_worker, self._status_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.wait(3000)
        super().closeEvent(event)

    def _general_tab(self):
        w = QWidget(); f = QFormLayout(w)
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.01, 0.99); self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(self._config.get("model", {}).get("conf_threshold", 0.25))
        self.spin_iou = QDoubleSpinBox(); self.spin_iou.setRange(0.01, 0.99); self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(self._config.get("model", {}).get("iou_threshold", 0.45))
        self.spin_tile = QSpinBox(); self.spin_tile.setRange(256, 2048); self.spin_tile.setSingleStep(64)
        self.spin_tile.setSuffix(" px"); self.spin_tile.setValue(self._config.get("model", {}).get("tile_capture_size", 640))
        f.addRow("置信度阈值:", self.spin_conf)
        f.addRow("IOU 阈值:", self.spin_iou)
        f.addRow("切片大小:", self.spin_tile)
        return w

    def _path_tab(self):
        w = QWidget(); f = QFormLayout(w)
        path_cfg = self._config.get("path_planning", {})
        self.spin_interval = QDoubleSpinBox(); self.spin_interval.setRange(0.1, 2.0); self.spin_interval.setSingleStep(0.1)
        self.spin_interval.setSuffix(" m"); self.spin_interval.setValue(path_cfg.get("path_point_interval_m", 0.6))
        self.spin_buffer = QDoubleSpinBox(); self.spin_buffer.setRange(0.0, 10.0); self.spin_buffer.setSingleStep(0.1)
        self.spin_buffer.setSuffix(" m"); self.spin_buffer.setValue(path_cfg.get("headland_buffer_m", 3.0))
        f.addRow("路径点间距:", self.spin_interval)
        f.addRow("田头缓冲:", self.spin_buffer)
        return w

    def _mask_tab(self):
        w = QWidget(); f = QFormLayout(w)
        self.mask_strength_value = QLabel("标准（固定）")
        self.mask_strength_value.setToolTip(
            "掩膜处理统一使用标准参数，保证不同任务的处理口径一致"
        )
        self.spin_gap = QDoubleSpinBox(); self.spin_gap.setRange(0.2, 20.0); self.spin_gap.setSingleStep(0.5)
        self.spin_gap.setSuffix(" m"); self.spin_gap.setValue(self._config.get("mask_processing", {}).get("band_internal_gap_close_m", 6.0))
        self.spin_end_gap = QDoubleSpinBox(); self.spin_end_gap.setRange(0.0, 2.0); self.spin_end_gap.setSingleStep(0.05)
        mask_cfg = self._config.get("mask_processing", {})
        self.spin_end_gap.setSuffix(" m"); self.spin_end_gap.setValue(mask_cfg.get("band_end_gap_close_m", mask_cfg.get("band_endpoint_gap_close_m", 0.35)))
        self.spin_trim = QDoubleSpinBox(); self.spin_trim.setRange(0.0, 2.0); self.spin_trim.setSingleStep(0.05)
        self.spin_trim.setSuffix(" m"); self.spin_trim.setValue(self._config.get("mask_processing", {}).get("band_end_trim_m", 0.1))
        f.addRow("处理强度:", self.mask_strength_value)
        f.addRow("行内断裂补全:", self.spin_gap)
        f.addRow("端部断裂补全:", self.spin_end_gap)
        f.addRow("端部收缩:", self.spin_trim)
        return w

    def _save(self):
        self._config.setdefault("model", {})["conf_threshold"] = self.spin_conf.value()
        self._config.setdefault("model", {})["iou_threshold"] = self.spin_iou.value()
        self._config.setdefault("model", {})["tile_capture_size"] = self.spin_tile.value()
        self._config.setdefault("path_planning", {})["path_point_interval_m"] = self.spin_interval.value()
        self._config.setdefault("path_planning", {})["headland_buffer_m"] = self.spin_buffer.value()
        # Keep the legacy config key for provenance/cache compatibility, but
        # there is only one active mask-processing profile now.
        self._config.setdefault("mask_processing", {})["strength"] = "standard"
        self._config.setdefault("mask_processing", {})["band_internal_gap_close_m"] = self.spin_gap.value()
        self._config.setdefault("mask_processing", {})["band_end_gap_close_m"] = self.spin_end_gap.value()
        self._config.setdefault("mask_processing", {})["band_end_trim_m"] = self.spin_trim.value()
        save_json_atomic(str(user_config_path()), self._config)
        self.accept()


class ProjectLogDialog(QDialog):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("项目日志")
        self.resize(700, 500)
        lo = QVBoxLayout(self)
        edit = QTextEdit(); edit.setReadOnly(True); edit.setPlainText(str(text or ""))
        lo.addWidget(edit)


class ReportDialog(QDialog):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("结果报告")
        self.resize(520, 420)
        lo = QVBoxLayout(self)
        edit = QTextEdit(); edit.setReadOnly(True); edit.setPlainText(str(data or ""))
        lo.addWidget(edit)


class DiagnosisDialog(QDialog):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("诊断信息")
        self.resize(520, 420)
        lo = QVBoxLayout(self)
        edit = QTextEdit(); edit.setReadOnly(True); edit.setPlainText(str(data or ""))
        lo.addWidget(edit)
