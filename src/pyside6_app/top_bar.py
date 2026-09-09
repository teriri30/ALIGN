"""
top_bar.py — Fluent CommandBar 顶部工具栏（无 emoji）
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QToolButton, QComboBox, QProgressBar, QFrame, QSizePolicy,
)
from pyside6_app.styles import COLORS

UI_FONT = '"Microsoft YaHei UI","Microsoft YaHei","SimHei","Segoe UI",sans-serif'

class FluentToolButton(QToolButton):
    def __init__(self, text, action_id, accent=False, parent=None):
        super().__init__(parent)
        self._action_id = action_id
        self._accent = accent
        self.setText(text)
        self.setToolTip(text)
        self.setAccessibleName(text)
        self.setMinimumHeight(32)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._update_style()
    def _update_style(self):
        if self._accent:
            self.setStyleSheet(f"QToolButton{{background:{COLORS['accent']};color:{COLORS['btn_default']};border:none;border-radius:4px;padding:5px 16px;font-size:12px;font-weight:bold;font-family:{UI_FONT};}}QToolButton:hover{{background:{COLORS['accent_hover']};}}QToolButton:disabled{{background:{COLORS['btn_disabled']};color:{COLORS['btn_disabled_text']};}}")
            pal = self.palette(); pal.setColor(QPalette.ButtonText, QColor(COLORS['btn_default'])); self.setPalette(pal)
        else:
            self.setStyleSheet(f"QToolButton{{background:{COLORS['bg_light']};color:{COLORS['text']};border:1px solid {COLORS['border']};border-radius:6px;padding:5px 14px;font-size:12px;font-family:{UI_FONT};}}QToolButton:hover{{background:{COLORS['bg_hover']};border-color:{COLORS['border_light']};}}QToolButton:disabled{{background:{COLORS['btn_disabled']};color:{COLORS['btn_disabled_text']};border-color:transparent;}}")
            pal = self.palette(); pal.setColor(QPalette.ButtonText, QColor(COLORS['text'])); self.setPalette(pal)
    def action_id(self):
        return self._action_id

class TopToolbar(QWidget):
    action_clicked = Signal(str)
    tool_changed = Signal(str)
    model_dropdown_opened = Signal()
    model_changed = Signal(str)
    turn_strategy_changed = Signal(str)
    theme_toggle = Signal()

    TURN_STRATEGY_MAP = {
        "自动": "auto", "弓形": "bow", "半圆": "semicircle",
        "梨形": "pear", "鱼尾": "fishtail", "α形": "alpha",
    }
    TURN_STRATEGY_REVERSE = {v: k for k, v in TURN_STRATEGY_MAP.items()}

    ACTION_SPECS = [
        ("打开影像","OPEN",False), ("加载模型","MODEL",False),
        ("AI识别","AI_INFER",True), ("掩膜处理","MASK_PROCESS",False),
        ("农机参数","PARAMS",False), ("起点与可选服务点","ENTRY_EXIT",False),
        ("转弯策略","TURN_STRATEGY",False), ("生成路径","PLAN",True),
        ("演示","PLAY",False), ("导出 TXT","EXPORT",False),
        ("取消任务","CANCEL_TASK",False),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        self.setObjectName("topToolbar")
        self.setStyleSheet(f"QWidget#topToolbar{{background:{COLORS['bg_dark']};border-bottom:1px solid {COLORS['border']};}}")
        self._action_buttons = {}
        self._history_buttons = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(10,2,12,2); layout.setSpacing(0)
        row1 = QHBoxLayout(); row1.setSpacing(4)
        self._tb_btns = []
        self._seps = []
        # undo/redo 放左上角
        for text, aid, tip in [("↩","UNDO","撤销上一步操作"),("↪","REDO","重做上一步撤销")]:
            btn = QToolButton(); btn.setText(text); btn.setToolTip(tip); btn.setAccessibleName(tip); btn.setFixedSize(32,28)
            btn.setStyleSheet(self._history_button_style())
            btn.clicked.connect(lambda checked,a=aid: self.action_clicked.emit(a))
            row1.addWidget(btn); self._tb_btns.append(btn); self._history_buttons[aid] = btn
        self.set_history_available(False, False)
        sep0 = QFrame(); sep0.setFrameShape(QFrame.VLine)
        sep0.setStyleSheet(f"background:{COLORS['border']};max-width:1px;max-height:24px;")
        sep0.setFixedHeight(24); row1.addWidget(sep0); self._seps.append(sep0)
        self._title_lbl = QLabel("智能农机规划系统")
        self._title_lbl.setStyleSheet(f"font-size:15px;font-weight:600;color:{COLORS['text_bright']};padding:0 8px;font-family:{UI_FONT};")
        row1.addWidget(self._title_lbl)
        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"background:{COLORS['border']};max-width:1px;max-height:24px;")
        sep.setFixedHeight(24); row1.addWidget(sep); self._seps.append(sep)
        self.lbl_filename = QLabel("")
        self.lbl_filename.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;padding:0 8px;font-family:{UI_FONT};")
        self.lbl_filename.setFixedWidth(180); row1.addWidget(self.lbl_filename)
        row1.addStretch()
        self._theme_btn = None
        # settings/theme 放右侧
        for text, aid in [("⚙","SETTINGS"),("☀","THEME")]:
            btn = QToolButton(); btn.setText(text); btn.setFixedSize(32,28)
            btn.setStyleSheet(f"QToolButton{{background:transparent;color:{COLORS['text']};border:none;border-radius:4px;font-size:14px;}}QToolButton:hover{{background:{COLORS['bg_hover']};}}")
            if aid == "THEME":
                btn.setToolTip("切换深色/浅色主题")
                btn.clicked.connect(self.theme_toggle.emit)
                self._theme_btn = btn
            else:
                btn.clicked.connect(lambda checked,a=aid: self.action_clicked.emit(a))
            row1.addWidget(btn); self._tb_btns.append(btn)
        layout.addLayout(row1)

    def set_filename(self, path):
        name = path.replace("\\","/").split("/")[-1]
        if len(name)>28: name = name[:10]+"..."+name[-7:]
        self.lbl_filename.setText(name)
    def update_theme_icon(self, theme_name):
        icons = {"dark": "☀", "light": "☾"}
        tips = {
            "dark": "当前: 深色 → 点击切换浅色",
            "light": "当前: 浅色 → 点击切换深色",
        }
        if self._theme_btn:
            self._theme_btn.setText(icons.get(theme_name, "☀"))
            self._theme_btn.setToolTip(tips.get(theme_name, "切换主题"))

    def refresh_theme(self):
        """主题切换后重新应用内联样式"""
        self.setStyleSheet(
            f"QWidget#topToolbar{{background:{COLORS['bg_dark']};border-bottom:1px solid {COLORS['border']};}}")
        # 工具栏按钮 (undo/redo/settings/theme)
        btn_style = self._history_button_style()
        for btn in getattr(self, '_tb_btns', []):
            btn.setStyleSheet(btn_style)
        # 标题和文件名
        if hasattr(self, '_title_lbl'):
            self._title_lbl.setStyleSheet(
                f"font-size:15px;font-weight:600;color:{COLORS['text_bright']};padding:0 8px;font-family:{UI_FONT};")
        if hasattr(self, 'lbl_filename'):
            self.lbl_filename.setStyleSheet(
                f"color:{COLORS['text_dim']};font-size:11px;padding:0 8px;font-family:{UI_FONT};")
        # 分隔线
        for sep in getattr(self, '_seps', []):
            sep.setStyleSheet(f"background:{COLORS['border']};max-width:1px;max-height:24px;")


    def set_worker_cancel_visible(self, visible):
        btn = self._action_buttons.get("CANCEL_TASK")
        if btn:
            btn.setVisible(bool(visible))
            btn.setEnabled(bool(visible))
    def set_action_enabled(self, aid, enabled):
        if aid in self._action_buttons: self._action_buttons[aid].setEnabled(enabled)
    def set_model_loading(self, loading):
        """Disable any toolbar model controls while weights are loading."""
        loading = bool(loading)
        button = self._action_buttons.get("MODEL")
        if button is not None:
            button.setEnabled(not loading)
        combo = getattr(self, "_model_combo", None)
        if combo is not None:
            combo.setEnabled(not loading)
    def set_action_active(self, aid, active):
        if aid in self._action_buttons:
            btn = self._action_buttons[aid]
            if active:
                btn.setStyleSheet(f"QToolButton{{background:{COLORS['accent_bg']};color:{COLORS['text_accent']};border:1px solid {COLORS['accent']};border-radius:4px;padding:5px 12px;font-size:12px;}}")
            else: btn._update_style()
    def set_progress(self, pct, msg=""):
        if not hasattr(self, 'progress_bar'):
            return
        if pct<=0: self.progress_bar.hide(); self.lbl_progress.setText("")
        else: self.progress_bar.show(); self.progress_bar.setValue(pct); self.lbl_progress.setText(msg)
    def set_history_available(self, can_undo, can_redo):
        mapping = {"UNDO": bool(can_undo), "REDO": bool(can_redo)}
        for aid, enabled in mapping.items():
            btn = self._history_buttons.get(aid)
            if btn:
                btn.setEnabled(enabled)

    @staticmethod
    def _history_button_style():
        return (
            f"QToolButton{{background:transparent;color:{COLORS['text']};border:none;"
            f"border-radius:4px;font-size:14px;font-family:{UI_FONT};}}"
            f"QToolButton:hover{{background:{COLORS['bg_hover']};}}"
            f"QToolButton:disabled{{background:transparent;color:{COLORS['btn_disabled_text']};}}"
        )

    def set_model_name(self, name):
        if "MODEL" not in self._action_buttons:
            return
        if name: d = name[:10]+"…" if len(name)>10 else name
        else: d = "加载模型"
        self._action_buttons["MODEL"].setText(d)
        if hasattr(self, '_model_combo') and name:
            self._model_combo.blockSignals(True)
            idx = self._model_combo.findText(name)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
            self._model_combo.blockSignals(False)
    def set_turn_strategy(self, strategy):
        if "TURN_STRATEGY" not in self._action_buttons:
            return
        labels = {"bow":"弓形转弯","semicircle":"半圆转弯","pear":"梨形转弯","fishtail":"鱼尾折返","auto":"自适应"}
        self._action_buttons["TURN_STRATEGY"].setText(labels.get(strategy,strategy))
        if hasattr(self, '_turn_combo') and strategy in self.TURN_STRATEGY_REVERSE:
            display_label = self.TURN_STRATEGY_REVERSE[strategy]
            self._turn_combo.blockSignals(True)
            idx = self._turn_combo.findText(display_label)
            if idx >= 0:
                self._turn_combo.setCurrentIndex(idx)
            self._turn_combo.blockSignals(False)
    def refresh_actions(self, state):
        """渐进式工具栏 — 操作按钮已移除，保留为空操作"""
        self.set_history_available(
            bool(getattr(state, 'undo_stack', [])),
            bool(getattr(state, 'redo_stack', [])),
        )
        if not self._action_buttons:
            return
        tif = bool(getattr(state, 'tif_path', None))
        hf = bool(getattr(state, 'field_boundary', None) and len(getattr(state, 'field_boundary', [])) >= 3)
        hi = bool(getattr(state, 'inference_done', False))
        hm = bool(getattr(state, 'mask_processed', False))
        hp = bool(getattr(state, 'harvester_params', False))
        he = bool(getattr(state, 'entry_point', None))
        hpl = bool(getattr(state, 'auto_path_planned', False))
        hpt = len(getattr(state, 'path_points', [])) >= 2
        ia = bool(getattr(state, 'anim_active', False))
        rn = bool(getattr(state, 'inference_running', False) or getattr(state, 'mask_processing_running', False) or getattr(state, 'plan_running', False))
        self.set_worker_cancel_visible(rn)

        # ── 可见性（渐进式披露：只显示当前步骤及下一步需要的操作） ──
        self._action_buttons["OPEN"].setVisible(True)
        # 模型：影像加载后才出现
        self._action_buttons["MODEL"].setVisible(tif and not hf and not rn)
        self._set_widget_visible(self._model_combo, tif and not hf and not rn)

        self.set_action_enabled("AI_INFER", hf and not rn)
        self._action_buttons["AI_INFER"].setVisible(tif and hf)

        self.set_action_enabled("MASK_PROCESS", hi and not rn)
        self._action_buttons["MASK_PROCESS"].setVisible(hi)

        self.set_action_enabled("PARAMS", not rn)
        self._action_buttons["PARAMS"].setVisible(hm)

        self.set_action_enabled("ENTRY_EXIT", hm and hp)
        self._action_buttons["ENTRY_EXIT"].setVisible(hm and hp)

        self.set_action_enabled("TURN_STRATEGY", hm)
        self._action_buttons["TURN_STRATEGY"].setVisible(hm)
        self._set_widget_visible(self._turn_combo, hm)

        self.set_action_enabled("PLAN", hm and hp and he and not rn)
        self._action_buttons["PLAN"].setVisible(hm and hp and he)

        self.set_action_enabled("PLAY", hpl or hpt)
        self._action_buttons["PLAY"].setVisible(hpl or hpt)

        self.set_action_enabled("EXPORT", hpl or hpt)
        self._action_buttons["EXPORT"].setVisible(hpl or hpt)

        # 播放按钮文本切换
        if ia:
            self._action_buttons["PLAY"].setText("退出演示")
        else:
            self._action_buttons["PLAY"].setText("演示")

    @staticmethod
    def _set_widget_visible(w, visible):
        """安全设置控件可见性"""
        if w is not None:
            w.setVisible(visible)

    # ── combo helpers ──────────────────────────────────────────────
    @staticmethod
    def _combo_style():
        return (
            f"QComboBox{{background:{COLORS['bg_dark']};color:{COLORS['text']};"
            f"border:1px solid {COLORS['border_light']};border-radius:3px;"
            f"padding:2px 6px;font-size:11px;min-height:20px;}}"
            f"QComboBox::drop-down{{border:none;}}"
            f"QComboBox QAbstractItemView{{background:{COLORS['bg_dark']};"
            f"color:{COLORS['text']};selection-background-color:{COLORS['accent_bg']};}}"
        )

    def _on_model_changed(self, index):
        """Emit model_changed with the selected model name (skip placeholder)."""
        if not hasattr(self, '_model_combo') or index <= 0:
            return
        name = self._model_combo.itemText(index)
        self.model_changed.emit(name)
        self.model_dropdown_opened.emit()

    def _on_turn_changed(self, index):
        """Emit turn_strategy_changed with the internal key."""
        if not hasattr(self, '_turn_combo'):
            return
        label = self._turn_combo.itemText(index)
        key = self.TURN_STRATEGY_MAP.get(label, "auto")
        self.turn_strategy_changed.emit(key)

    def populate_models(self, model_list):
        """Fill the model combo box with *model_list* (list of display names)."""
        if not hasattr(self, '_model_combo'):
            return
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItem("— 选择模型 —")
        for name in model_list:
            self._model_combo.addItem(name)
        self._model_combo.blockSignals(False)

    def set_current_model(self, name):
        """Select the model combo item matching *name*."""
        if not hasattr(self, '_model_combo'):
            return
        self._model_combo.blockSignals(True)
        idx = self._model_combo.findText(name)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)

    def set_current_strategy(self, key):
        """Select the turn-strategy combo item matching internal *key*."""
        if not hasattr(self, '_turn_combo'):
            return
        if key not in self.TURN_STRATEGY_REVERSE:
            return
        label = self.TURN_STRATEGY_REVERSE[key]
        self._turn_combo.blockSignals(True)
        idx = self._turn_combo.findText(label)
        if idx >= 0:
            self._turn_combo.setCurrentIndex(idx)
        self._turn_combo.blockSignals(False)
