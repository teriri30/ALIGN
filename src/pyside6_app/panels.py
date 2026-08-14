import math
import time
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QDrag, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QScrollArea, QFrame, QTextEdit, QToolButton, QMenu,
    QApplication,
)
from pyside6_app.styles import COLORS
from pyside6_app.task_state import TASK_ORDER

UI_FONT = '"Microsoft YaHei UI","Microsoft YaHei","SimHei","Segoe UI",sans-serif'

# 真实窗口 show() 后逐级缩窄测得：路径规划卡片展开且出现纵向滚动条时，
# 328 px 仍会产生 1 px 横向滚动，330 px 起全部控件不再横向挤压。
TASK_PANEL_COMPACT_MIN_WIDTH = 330

HARVESTER_PARAM_LABELS = {
    "cutter_width_m": "割台有效割幅",
    "cutter_forward_offset_m": "割刀前置距离",
    "track_width_m": "履带宽度",
    "track_gauge_m": "履带中心距",
    "wheelbase_m": "轴距",
    "track_length_m": "履带接地长度",
    "turn_radius_m": "最小转弯半径",
}
HARVESTER_SOURCE_LABELS = {
    "default": "默认值",
    "recommended": "推荐值",
    "user_entered": "用户填写但未实测",
    "legacy_unverified": "历史来源未记录",
    "user_measured": "用户实测",
}


class ExplicitPopupComboBox(QComboBox):
    """Ignore wheel selection unless the popup list is explicitly open.

    A planning strategy is a scientific input, not a scrollable convenience
    value.  The task panel itself sits in a scroll area, so forwarding a wheel
    event to a closed combo made ordinary panel scrolling silently alter the
    route and persist that accidental value in the project cache.
    """

    def wheelEvent(self, event):
        if self.view().isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


TASK_DEFS = [
    {"id": "import_img", "label": "影像导入", "hint": "加载正射影像 GeoTIFF 文件。", "color": "#6FA37F", "action": "导入"},
    {"id": "draw_field", "label": "田块圈选", "hint": "在影像上圈选田块范围。", "color": "#5EAF6E", "action": "圈选"},
    {"id": "segment", "label": "AI识别", "hint": "使用模型分割作物行带。", "color": "#649C7E", "action": "运行"},
    {"id": "process", "label": "掩膜处理", "hint": "优化 AI 识别掩膜并提取作物行带。", "color": "#648C82", "action": "处理"},
    {"id": "params", "label": "农机参数", "hint": "设置农机参数。", "color": "#AC7E56", "action": "设置"},
    {"id": "entry", "label": "起点（必选）", "hint": "可自由布设或拖动；规划必须从离起点最近的作业线端点开始。", "color": "#589C94", "action": "布设"},
    {"id": "exit", "label": "终点（可选）", "hint": "不设置时完成最后一条作业线即结束；设置后仅用于末段方向参考。", "color": "#C76B6B", "action": "布设"},
    {"id": "unload", "label": "卸粮点", "hint": "按需要添加一个或多个卸粮点，可为空。", "color": "#C9A64A", "action": "布设"},
    {"id": "plan", "label": "路径规划", "hint": "生成作业路线。", "color": "#589862", "action": "生成"},
    {"id": "simulate", "label": "模拟演示", "hint": "沿规划路径模拟农机运动。", "color": "#4C8C7C", "action": "演示"},
    {"id": "export", "label": "路径导出", "hint": "导出路径结果。", "color": "#4484BE", "action": "导出"},
]

STEP_GROUPS = [
    {"label": "影像导入", "tasks": ["import_img"]},
    {"label": "田块圈选", "tasks": ["draw_field"]},
    {"label": "AI识别", "tasks": ["segment"]},
    {"label": "掩膜处理", "tasks": ["process"]},
    {"label": "参数配置", "tasks": ["params"]},
    {"label": "路径规划", "tasks": ["entry", "exit", "unload", "plan"]},
    {"label": "模拟演示", "tasks": ["simulate"]},
    {"label": "路径导出", "tasks": ["export"]},
]


def _task_def(tid):
    return next((item for item in TASK_DEFS if item["id"] == tid), None)


class TaskRow(QWidget):
    clicked = Signal(str)
    delete_clicked = Signal(str)

    def __init__(self, task_def, parent=None):
        super().__init__(parent)
        self.setObjectName("taskRow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumHeight(44)
        self.setStyleSheet("#taskRow{background:transparent;}")
        self.tid = task_def["id"]
        self._color = task_def["color"]
        self._done = False
        self._status_text = "待处理"
        lo = QHBoxLayout(self)
        lo.setContentsMargins(8, 3, 8, 3)
        lo.setSpacing(7)
        self._badge = QLabel("")
        self._badge.setFixedSize(14, 14)
        self._badge.setStyleSheet(f"background:{task_def['color']};border-radius:3px;")
        lo.addWidget(self._badge)
        self._label = QLabel(task_def["label"])
        self._label.setFont(QFont("Microsoft YaHei UI", 10))
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-family:{UI_FONT};")
        self._label.setMinimumWidth(0)
        lo.addWidget(self._label, 1)
        self._status = QLabel(self._status_text)
        self._status.setFont(QFont("Microsoft YaHei UI", 9))
        self._status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._status.setMinimumWidth(0)
        self._status.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;font-family:{UI_FONT};")
        lo.addWidget(self._status)
        self._action_btn = QPushButton(task_def["action"])
        self._action_btn.setMinimumSize(58, 28)
        self._action_btn.setMaximumHeight(30)
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{task_def['color']};color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
        )
        self._action_btn.clicked.connect(lambda: self.clicked.emit(self.tid))
        lo.addWidget(self._action_btn)
        self._delete_btn = QPushButton("回退")
        self._delete_btn.setMinimumSize(52, 28)
        self._delete_btn.setMaximumHeight(30)
        self._delete_btn.setToolTip("回退到此步骤之前，并清除后续结果")
        self._delete_btn.setStyleSheet(
            "QPushButton{background:#7A4D3B;color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
            "QPushButton:hover{background:#9A654D;}"
        )
        self._delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.tid))
        lo.addWidget(self._delete_btn)

    def set_status(self, done=False, enabled=True, in_progress=False, progress_pct=-1, text=None, status="pending"):
        self._done = bool(done)
        self._in_progress = bool(in_progress)
        self._status_key = status
        self.setEnabled(bool(enabled))
        self._action_btn.setEnabled(bool(enabled))
        if text:
            display_text = str(text)
        elif done:
            display_text = "已完成"
        elif in_progress:
            display_text = f"处理中 {progress_pct}%" if progress_pct >= 0 else "处理中"
        elif status == "available":
            display_text = "可操作"
        elif status == "failed":
            display_text = "失败"
        else:
            display_text = "待处理"
        if done:
            color = COLORS.get('green', '#64D26A')
        elif in_progress or status == "available":
            color = COLORS.get('accent', '#58A6FF')
        elif status == "failed":
            color = COLORS.get('red', '#E85D63')
        else:
            color = COLORS.get('text_dim', '#8B909A')
        self._status.setText(display_text)
        self._status.setStyleSheet(f"color:{color};font-size:11px;font-family:{UI_FONT};")
        opacity = "1.0" if enabled else "0.55"
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-family:{UI_FONT};")
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{self._color};color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
            f"QPushButton:disabled{{background:{COLORS.get('bg_light', '#3A3A44')};color:{COLORS.get('text_dim', '#8B909A')};}}"
        )
        self.setStyleSheet(f"#taskRow{{background:transparent;opacity:{opacity};}}")

    def refresh_theme(self):
        status = getattr(self, "_status_key", "pending")
        in_progress = getattr(self, "_in_progress", False)
        if self._done:
            color = COLORS.get('green', '#64D26A')
        elif in_progress or status == "available":
            color = COLORS.get('accent', '#58A6FF')
        elif status == "failed":
            color = COLORS.get('red', '#E85D63')
        else:
            color = COLORS.get('text_dim', '#8B909A')
        opacity = "1.0" if self.isEnabled() else "0.55"
        self._status.setStyleSheet(f"color:{color};font-size:11px;font-family:{UI_FONT};")
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-family:{UI_FONT};")
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{self._color};color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
            f"QPushButton:disabled{{background:{COLORS.get('bg_light', '#3A3A44')};color:{COLORS.get('text_dim', '#8B909A')};}}"
        )
        self._delete_btn.setStyleSheet(
            "QPushButton{background:#7A4D3B;color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
            "QPushButton:hover{background:#9A654D;}"
        )
        self.setStyleSheet(f"#taskRow{{background:transparent;opacity:{opacity};}}")


class StepCard(QWidget):
    toggled = Signal()

    def __init__(self, title, rows, parent=None):
        super().__init__(parent)
        self.setObjectName("stepCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        # 不设置固定最小高度：折叠时由布局按表头高度决定；
        # 展开时最小高度必须覆盖内容，否则 QScrollArea 内布局会把卡片
        # 压扁，内容互相覆盖（任务行被后面的智能推荐/禁行区/调头方式盖住）。
        self._rows = rows
        self._expanded = False
        lo = QVBoxLayout(self)
        lo.setContentsMargins(7, 6, 7, 6)
        lo.setSpacing(5)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self._toggle = QToolButton(); self._toggle.setText("▸"); self._toggle.setFixedSize(24, 28)
        self._title = QLabel(title)
        self._title.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._title.setFont(QFont("Microsoft YaHei UI", 10))
        self._title.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-weight:600;font-family:{UI_FONT};")
        self._mark = QLabel("")
        self._mark.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._mark.setFixedSize(24, 28)
        self._mark.setAlignment(Qt.AlignCenter)
        head.addWidget(self._toggle); head.addWidget(self._title, 1); head.addWidget(self._mark)
        lo.addLayout(head)
        self._content = QWidget()
        self._content.setAttribute(Qt.WA_StyledBackground, True)
        self._content.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(self._content); cl.setContentsMargins(10, 2, 0, 0); cl.setSpacing(2)
        for row in rows: cl.addWidget(row)
        lo.addWidget(self._content)
        self._toggle.clicked.connect(self._toggle_expanded)
        self.set_expanded(False)
        self.refresh_state()

    def _toggle_expanded(self):
        self.set_expanded(not self._expanded)
        self.toggled.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.position().y() <= 40:
            self._toggle_expanded()
            event.accept()
            return
        super().mousePressEvent(event)

    def set_expanded(self, expanded):
        self._expanded = bool(expanded)
        self._content.setVisible(self._expanded)
        self._toggle.setText("▾" if self._expanded else "▸")
        self._sync_minimum_height()

    def _sync_minimum_height(self):
        """展开时把显式最小高度设为内容高度，折叠时回到表头高度。

        QScrollArea 内布局只按各卡片的最小高度分配纵向空间
        （QLayout::doResize 依据的也是显式最小值），如果展开的卡片
        最小高度仍是表头高度，卡片会被压扁，内容互相覆盖。
        """
        if self._expanded:
            # A child can become visible after the card was expanded (route
            # editing does this).  Drop the stale explicit minimum before
            # asking the layouts for a new hint, otherwise the QScrollArea
            # keeps the old height and clips the newly visible controls.
            self.setMinimumHeight(0)
            content_layout = self._content.layout()
            if content_layout is not None:
                content_layout.invalidate()
                content_layout.activate()
            own_layout = self.layout()
            if own_layout is not None:
                own_layout.invalidate()
                own_layout.activate()
                hint_height = own_layout.sizeHint().height()
            else:
                hint_height = self.sizeHint().height()
            self.setMinimumHeight(max(40, int(hint_height)))
        else:
            self.setMinimumHeight(40)
        self.updateGeometry()

    def refresh_content_geometry(self):
        """Recompute an expanded card after a runtime child visibility change."""
        self._sync_minimum_height()
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()

    def refresh_state(self):
        task_rows = [row for row in self._rows if hasattr(row, "_done")]
        done = bool(task_rows) and all(row._done for row in task_rows)
        self._title.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-weight:600;font-family:{UI_FONT};")
        if done:
            bg = COLORS.get("success_bg", "#1A3A1A")
            border = COLORS.get("green", "#6CCB6E")
            self._mark.setText("✓")
            self._mark.setStyleSheet(f"color:{border};font-weight:bold;font-size:14px;")
        else:
            bg = COLORS.get("bg_card", "#2A2A35")
            border = COLORS.get("border", "#383848")
            self._mark.setText("")
        self.setStyleSheet(
            f"#stepCard{{background:{bg};border:1px solid {border};"
            "border-radius:7px;margin:1px 0;}}"
            f"#stepCard QLabel{{font-family:{UI_FONT};}}"
            f"QToolButton{{background:transparent;color:{COLORS['text_dim']};border:none;font-family:{UI_FONT};}}"
            f"QToolButton:hover{{background:{COLORS['bg_hover']};border-radius:4px;}}"
        )


class TaskPanel(QWidget):
    load_tif_requested = Signal()
    load_model_requested = Signal()
    run_requested = Signal()
    segment_requested = Signal()
    process_requested = Signal()
    export_mask_diagnostics_requested = Signal()
    plan_requested = Signal()
    simulate_requested = Signal()
    params_requested = Signal()
    entry_exit_toggle = Signal(str)
    export_geojson_requested = Signal()
    export_csv_requested = Signal()
    export_img_requested = Signal()
    field_drawing_requested = Signal()
    model_changed = Signal(str)
    mask_strength_changed = Signal(str)
    turn_strategy_changed = Signal(str)
    confirm_no_forbidden_requested = Signal()
    draw_forbidden_requested = Signal()
    route_edit_requested = Signal()
    route_edit_tool_requested = Signal(str)
    recommend_service_points_requested = Signal()
    step_deleted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(TASK_PANEL_COMPACT_MIN_WIDTH)
        self._task_widgets = {}
        self._step_cards = []
        self._current_step = 0
        self._manual_step = None
        self._forbidden_count = 0
        self._forbidden_confirmed = False
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{COLORS['bg']};")
        lo = QVBoxLayout(self); lo.setContentsMargins(6, 8, 6, 8); lo.setSpacing(6)
        self._title = QLabel("任务清单")
        self._title.setFont(QFont("Microsoft YaHei UI", 12, QFont.Weight.DemiBold))
        self._title.setStyleSheet(f"font-size:16px;font-weight:600;color:{COLORS['text_bright']};font-family:{UI_FONT};")
        lo.addWidget(self._title)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        self._scroll = scroll
        body = QWidget(); bl = QVBoxLayout(body); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(4)
        body.setStyleSheet("background:transparent;")
        for idx, group in enumerate(STEP_GROUPS, start=1):
            rows = []
            for tid in group["tasks"]:
                td = _task_def(tid); row = TaskRow(td); self._task_widgets[tid] = row
                row.clicked.connect(self._on_task_clicked); row.delete_clicked.connect(self.step_deleted.emit)
                rows.append(row)
                if tid == "process":
                    self._mask_strength_label = QLabel("处理强度")
                    self._mask_strength_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;font-family:{UI_FONT};padding-left:8px;")
                    self.mask_strength_value_label = QLabel("标准（固定）")
                    self.mask_strength_value_label.setToolTip(
                        "掩膜处理统一使用标准参数，保证不同任务的处理口径一致"
                    )
                    self.mask_strength_value_label.setStyleSheet(
                        f"color:{COLORS['text']};font-size:12px;font-family:{UI_FONT};"
                        f"padding:4px 8px;border:1px solid {COLORS['border']};"
                        "border-radius:4px;"
                    )
                    rows.append(self._mask_strength_label)
                    rows.append(self.mask_strength_value_label)
                    self.mask_diagnostic_btn = QPushButton("导出诊断数据")
                    self.mask_diagnostic_btn.setToolTip("导出各处理阶段、连通区域指标和异常编号总览")
                    self.mask_diagnostic_btn.setMinimumHeight(30)
                    self.mask_diagnostic_btn.setEnabled(False)
                    self.mask_diagnostic_btn.clicked.connect(
                        self.export_mask_diagnostics_requested.emit
                    )
                    rows.append(self.mask_diagnostic_btn)
                if tid == "segment":
                    self.model_combo = QComboBox(); self.model_combo.currentTextChanged.connect(self.model_changed.emit)
                    self.model_combo.setMinimumHeight(30)
                    self.model_combo.setToolTip("选择 AI 识别模型，或从文件夹导入其他 .pt 权重")
                    self.model_combo.setStyleSheet(
                        f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-family:{UI_FONT};}}"
                    )
                    rows.append(self.model_combo)
                if tid == "plan":
                    # 任务行下方的设置区：分隔线 + 紧凑单行控件，
                    # 与上方的 4 个任务行在视觉上区分开。
                    sep = QFrame()
                    sep.setFrameShape(QFrame.HLine)
                    sep.setFixedHeight(1)
                    sep.setStyleSheet(f"color:{COLORS.get('border', '#383848')};background:{COLORS.get('border', '#383848')};")
                    rows.append(sep)
                    self.recommend_service_points_btn = QPushButton("智能推荐起点")
                    self.recommend_service_points_btn.setToolTip(
                        "推荐一个靠近作业线端部的起点；确认或拖动后，路径从离它最近的作业线端点开始"
                    )
                    self.recommend_service_points_btn.setMinimumHeight(30)
                    self.recommend_service_points_btn.clicked.connect(
                        self.recommend_service_points_requested.emit
                    )
                    rows.append(self.recommend_service_points_btn)
                    forbidden_row = QWidget()
                    forbidden_layout = QHBoxLayout(forbidden_row)
                    forbidden_layout.setContentsMargins(8, 0, 8, 0)
                    forbidden_layout.setSpacing(6)
                    self.forbidden_status_label = QLabel("禁行区：未确认")
                    self.forbidden_status_label.setStyleSheet(
                        f"color:{COLORS['orange']};font-size:12px;font-family:{UI_FONT};"
                    )
                    self.confirm_no_forbidden_btn = QPushButton("确认无")
                    self.draw_forbidden_btn = QPushButton("绘制")
                    for button in (self.confirm_no_forbidden_btn, self.draw_forbidden_btn):
                        button.setMinimumSize(52, 26)
                        button.setStyleSheet(
                            f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                            f"border:1px solid {COLORS['border']};border-radius:4px;padding:3px 8px;font-family:{UI_FONT};}}"
                            f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                        )
                    self.confirm_no_forbidden_btn.setToolTip("确认田块内没有需要避让的沟渠、水坑或障碍")
                    self.draw_forbidden_btn.setToolTip("在影像上绘制不可通行区域")
                    self.confirm_no_forbidden_btn.clicked.connect(self.confirm_no_forbidden_requested.emit)
                    self.draw_forbidden_btn.clicked.connect(self.draw_forbidden_requested.emit)
                    forbidden_layout.addWidget(self.forbidden_status_label, 1)
                    forbidden_layout.addWidget(self.confirm_no_forbidden_btn)
                    forbidden_layout.addWidget(self.draw_forbidden_btn)
                    rows.append(forbidden_row)
                    turn_row = QWidget()
                    turn_layout = QHBoxLayout(turn_row)
                    turn_layout.setContentsMargins(8, 0, 8, 0)
                    turn_layout.setSpacing(6)
                    turn_label = QLabel("调头方式")
                    turn_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;font-family:{UI_FONT};")
                    self.turn_combo = ExplicitPopupComboBox()
                    self.turn_combo.setToolTip(
                        "点击展开后选择调头/转弯方式；未展开时滚轮只滚动面板，不会改策略"
                    )
                    self.turn_combo.setMinimumHeight(28)
                    self.turn_combo.setStyleSheet(
                        f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-family:{UI_FONT};}}"
                    )
                    # DECISION-016: contour is an auditable pass-order strategy.
                    # 套圈收割为默认策略（置于首位）。
                    for label, key in [
                        ("套圈收割", "contour"), ("自动推荐", "auto"),
                        ("弓形调头", "bow"), ("半圆调头", "semicircle"),
                        ("梨形调头", "pear"), ("鱼尾折返", "fishtail"), ("紧凑 Alpha", "alpha"),
                    ]:
                        self.turn_combo.addItem(label, key)
                    # Only an explicit popup selection may change this
                    # scientific input. Programmatic synchronisation and a
                    # closed-combo wheel event must stay silent.
                    self.turn_combo.activated.connect(self._on_turn_combo_changed)
                    turn_layout.addWidget(turn_label)
                    turn_layout.addWidget(self.turn_combo, 1)
                    rows.append(turn_row)
                    self.route_edit_btn = QPushButton("编辑路线")
                    self.route_edit_btn.setToolTip(
                        "进入人工路线编辑；画布负责点线拖动，下面的路径结果负责结构操作与验证"
                    )
                    self.route_edit_btn.setMinimumHeight(30)
                    self.route_edit_btn.setStyleSheet(
                        f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:5px;padding:4px 8px;font-family:{UI_FONT};}}"
                        f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                        f"QPushButton:disabled{{color:{COLORS.get('text_dim', '#8B909A')};}}"
                    )
                    self.route_edit_btn.clicked.connect(self.route_edit_requested.emit)
                    rows.append(self.route_edit_btn)
            card = StepCard(f"{idx}. {group['label']}", rows)
            card.toggled.connect(lambda _checked=False, i=idx - 1: self._on_card_toggled(i))
            self._step_cards.append(card); bl.addWidget(card)
        bl.addStretch(); scroll.setWidget(body); lo.addWidget(scroll, 1)
        self._refresh_cards()

    def _on_task_clicked(self, tid):
        mapping = {
            "import_img": self.load_tif_requested,
            "draw_field": self.field_drawing_requested,
            "segment": self.segment_requested,
            "process": self.process_requested,
            "params": self.params_requested,
            "entry": self.entry_exit_toggle,
            "exit": self.entry_exit_toggle,
            "unload": self.entry_exit_toggle,
            "plan": self.plan_requested,
            "simulate": self.simulate_requested,
            "export": self.export_geojson_requested,
        }
        sig = mapping.get(tid)
        if sig:
            if tid in {"entry", "exit", "unload"}:
                self.entry_exit_toggle.emit(tid)
            else:
                sig.emit()

    def _refresh_cards(self):
        first = 0
        for i, group in enumerate(STEP_GROUPS):
            if any(not self._task_widgets[tid]._done for tid in group["tasks"]):
                first = i; break
        expanded = self._manual_step if self._manual_step is not None else first
        for i, card in enumerate(self._step_cards):
            card.refresh_state()
            # Accordion behaviour: only one card owns vertical space at a time.
            # Keeping both the current workflow card and a manually opened card
            # expanded can make the two content areas collide/cover each other in
            # the narrow left splitter, especially after project restore.
            card.set_expanded(i == expanded)
        self._current_step = first
        # 展开的卡片（如“路径规划”，内容较多）可能比可视区更高，
        # 滚动到展开卡片，避免步骤推进时活动步骤被挤出可视区。
        self._ensure_card_visible(expanded)

    def _on_card_toggled(self, index):
        card = self._step_cards[index]
        if card._expanded:
            self._manual_step = None if index == self._current_step else index
            for i, other in enumerate(self._step_cards):
                if i != index:
                    other.set_expanded(False)
            self._ensure_card_visible(index)
        else:
            self._manual_step = None
            self._refresh_cards()

    def _ensure_card_visible(self, index):
        """Scroll the selected accordion card into view after expansion."""
        scroll = getattr(self, "_scroll", None)
        if not scroll or not (0 <= index < len(self._step_cards)):
            return
        scroll.ensureWidgetVisible(self._step_cards[index], 0, 8)

    def set_task_status(self, tid, status, progress_pct=-1, text=None):
        row = self._task_widgets.get(tid)
        if not row: return
        row.set_status(
            done=status == "done",
            enabled=status in ("available", "in_progress", "done", "failed"),
            in_progress=status == "in_progress",
            progress_pct=progress_pct,
            text=text,
            status=status,
        )
        if not getattr(self, "_applying_task_statuses", False):
            self._refresh_cards()

    def apply_task_statuses(self, statuses):
        self._applying_task_statuses = True
        try:
            for tid in TASK_ORDER:
                st = statuses.get(tid)
                if not st:
                    continue
                self.set_task_status(tid, st.status, text=st.text)
                if st.action:
                    self.set_task_action(tid, st.action)
                else:
                    task_def = _task_def(tid)
                    if task_def:
                        self.set_task_action(tid, task_def["action"])
        finally:
            self._applying_task_statuses = False
        self._refresh_cards()

    def set_task_action(self, tid, text):
        row = self._task_widgets.get(tid)
        if row: row._action_btn.setText(str(text))


    def _on_turn_combo_changed(self, index):
        if not hasattr(self, 'turn_combo'):
            return
        key = self.turn_combo.itemData(index) or "auto"
        self.turn_strategy_changed.emit(str(key))

    def set_turn_strategy(self, key):
        if not hasattr(self, 'turn_combo'):
            return
        for i in range(self.turn_combo.count()):
            if self.turn_combo.itemData(i) == key:
                self.turn_combo.blockSignals(True)
                self.turn_combo.setCurrentIndex(i)
                self.turn_combo.blockSignals(False)
                return

    def _on_mask_strength_changed(self, index):
        # Kept as a compatibility hook for callers from older UI snapshots.
        self.mask_strength_changed.emit("standard")

    def set_mask_strength(self, key):
        if not hasattr(self, "mask_strength_value_label"):
            return
        self.mask_strength_value_label.setText("标准（固定）")

    def set_forbidden_state(self, count, confirmed):
        if not hasattr(self, "forbidden_status_label"):
            return
        count = int(count or 0)
        self._forbidden_count = count
        self._forbidden_confirmed = bool(confirmed)
        if count:
            text = f"禁行区：已绘制 {count} 处"
            color = COLORS["red"]
        elif confirmed:
            text = "禁行区：已确认无"
            color = COLORS["green"]
        else:
            text = "禁行区：未确认"
            color = COLORS["orange"]
        self.forbidden_status_label.setText(text)
        self.forbidden_status_label.setStyleSheet(
            f"color:{color};font-size:12px;font-family:{UI_FONT};"
        )

    def set_mask_diagnostic_available(self, available):
        if hasattr(self, "mask_diagnostic_btn"):
            self.mask_diagnostic_btn.setEnabled(bool(available))

    def set_model_options(self, models, current=""):
        if not hasattr(self, 'model_combo'): return
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in models or []:
            name = model.get("name") or model.get("path") or str(model)
            path = model.get("path", name) if isinstance(model, dict) else str(model)
            self.model_combo.addItem(name, path)
            if path == current or name == current:
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self.model_combo.addItem("导入模型文件夹...", "__IMPORT_MODEL_FOLDER__")
        self.model_combo.blockSignals(False)

    def populate_models(self, models, current=""):
        normalized = [{"name": str(item), "path": str(item)} for item in (models or [])]
        self.set_model_options(normalized, current)

    def set_model_path(self, name):
        if hasattr(self, 'model_combo'):
            idx = self.model_combo.findText(name)
            if idx >= 0: self.model_combo.setCurrentIndex(idx)

    def refresh_theme(self):
        self.setStyleSheet(f"background:{COLORS['bg']};")
        self._title.setStyleSheet(f"font-size:16px;font-weight:600;color:{COLORS['text_bright']};font-family:{UI_FONT};")
        combo_style = (
            f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
            f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-family:{UI_FONT};}}"
        )
        if hasattr(self, "_mask_strength_label"):
            self._mask_strength_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;font-family:{UI_FONT};padding-left:8px;")
        for combo_name in ("model_combo", "turn_combo"):
            combo = getattr(self, combo_name, None)
            if combo:
                combo.setStyleSheet(combo_style)
        for button_name in (
            "route_edit_btn", "mask_diagnostic_btn",
            "recommend_service_points_btn",
        ):
            button = getattr(self, button_name, None)
            if button:
                button.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                f"border:1px solid {COLORS['border']};border-radius:5px;padding:4px 8px;font-family:{UI_FONT};}}"
                f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                f"QPushButton:disabled{{color:{COLORS.get('text_dim', '#8B909A')};}}"
                )
        for button_name in ("confirm_no_forbidden_btn", "draw_forbidden_btn"):
            button = getattr(self, button_name, None)
            if button:
                button.setStyleSheet(
                    f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                    f"border:1px solid {COLORS['border']};border-radius:4px;padding:3px 8px;font-family:{UI_FONT};}}"
                    f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                )
        self.set_forbidden_state(self._forbidden_count, self._forbidden_confirmed)
        for row in self.findChildren(TaskRow):
            row.refresh_theme()
        for card in self._step_cards:
            card.refresh_state()

    def set_route_editing_active(self, active):
        active = bool(active)
        if hasattr(self, "route_edit_btn"):
            self.route_edit_btn.setText("完成编辑" if active else "编辑路线")

    def set_route_edit_mode(self, mode):
        mode = str(mode or "select")
        mapping = {
            "select": getattr(self, "route_select_btn", None),
            "add_work": getattr(self, "route_add_work_btn", None),
            "add_point": getattr(self, "route_add_point_btn", None),
        }
        for key, button in mapping.items():
            if button is not None:
                button.setChecked(key == mode)


class RouteSegmentRow(QWidget):
    clicked = Signal(int)
    edit_clicked = Signal(int, int)
    edit_action_requested = Signal(str, int, int)
    reorder_requested = Signal(int, int)
    _WORK_MIME = "application/x-reui-work-line-index"

    def __init__(self, index, label, distance_m, is_work=True, parent=None,
                 segment_type=None, point_count=None, edit_meta=None,
                 display_number=None):
        super().__init__(parent); self._index = index; self._is_work = is_work
        self._segment_type = str(segment_type or ("work" if is_work else "turn"))
        self._edit_meta = dict(edit_meta or {})
        self._edit_selected = False
        self._drag_start = None
        self.setMinimumHeight(28)
        self.setAcceptDrops(bool(self._edit_meta and self._is_work))
        lo = QHBoxLayout(self); lo.setContentsMargins(8, 3, 8, 3); lo.setSpacing(8)
        badge = {
            "work": "线", "turn_approach": "接", "turn": "转",
            "turn_reverse": "倒", "turn_aux": "辅", "entry": "进",
            "exit": "出", "unload": "卸",
        }.get(self._segment_type, "段")
        self._badge = QLabel(badge); self._badge.setFixedSize(24, 18); self._badge.setAlignment(Qt.AlignCenter)
        number = index + 1 if display_number is None else int(display_number)
        self._label = QLabel(f"{label} {number:02d}")
        self._point_count = QLabel(f"{int(point_count)}点") if point_count is not None else None
        self._dist = QLabel(f"{distance_m:.1f} m")
        lo.addWidget(self._badge)
        lo.addWidget(self._label, 1)
        if self._point_count is not None:
            lo.addWidget(self._point_count)
        lo.addWidget(self._dist)
        if self._edit_meta:
            self.setToolTip(
                "单击联动画布；右击编辑线段"
                + ("；拖动卡片调整作业顺序" if self._is_work else "")
            )
        self.refresh_theme()

    def refresh_theme(self):
        color = {
            "work": "#529C62", "turn_approach": "#4C7FA8", "turn": "#BA9648",
            "turn_reverse": "#A85A5A", "turn_aux": "#8A6E9E", "entry": "#4F8790",
            "exit": "#6A7780", "unload": "#8C7353",
        }.get(self._segment_type, "#6A7780")
        self._badge.setStyleSheet(f"background:{color};color:white;border-radius:2px;font-size:10px;")
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:12px;")
        if self._point_count is not None:
            self._point_count.setStyleSheet(
                f"color:{COLORS['text_dim']};font-size:10px;"
            )
        self._dist.setStyleSheet(f"color:{COLORS['text_dim']};font-family:Consolas;")
        self.setStyleSheet(
            f"RouteSegmentRow{{background:{COLORS['bg_accent'] if self._edit_selected else COLORS['bg']};"
            f"border:1px solid {COLORS['accent'] if self._edit_selected else 'transparent'};"
            "border-radius:4px;}}"
        )

    def set_edit_selected(self, selected):
        self._edit_selected = bool(selected)
        self.refresh_theme()

    def available_edit_actions(self):
        if not self._edit_meta:
            return []
        actions = ["toggle_segment", "add_waypoint", "validate_route"]
        if int(self._edit_meta.get("node_end", -1)) - int(
            self._edit_meta.get("node_start", -1)
        ) >= 2:
            actions.append("delete_waypoint")
        if self._is_work:
            actions.extend(["add_work_after", "reverse_work", "delete_work"])
        return actions

    def trigger_edit_action(self, action, node_index=None):
        if not self._edit_meta:
            return
        if action == "delete_waypoint" and node_index is not None:
            first = last = int(node_index)
        else:
            first = int(self._edit_meta.get("edge_start", -1))
            last = int(self._edit_meta.get("edge_end", first))
        self.edit_action_requested.emit(str(action), first, last)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position().toPoint()
            if self._edit_meta:
                self.edit_clicked.emit(
                    int(self._edit_meta.get("edge_start", -1)),
                    int(self._edit_meta.get("edge_end", -1)),
                )
            else:
                self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        work_index = int(self._edit_meta.get("work_index", -1))
        if (
            self._drag_start is not None
            and work_index >= 0
            and event.buttons() & Qt.LeftButton
            and (event.position().toPoint() - self._drag_start).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(self._WORK_MIME, str(work_index).encode("ascii"))
            drag.setMimeData(mime)
            drag.setPixmap(self.grab())
            drag.exec(Qt.MoveAction)
            self._drag_start = None
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event):
        if self._is_work and event.mimeData().hasFormat(self._WORK_MIME):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        target = int(self._edit_meta.get("work_index", -1))
        try:
            source = int(bytes(event.mimeData().data(self._WORK_MIME)).decode("ascii"))
        except (TypeError, ValueError, UnicodeDecodeError):
            event.ignore()
            return
        if source >= 0 and target >= 0 and source != target:
            self.reorder_requested.emit(source, target)
            event.acceptProposedAction()
            return
        event.ignore()

    def contextMenuEvent(self, event):
        if not self._edit_meta:
            super().contextMenuEvent(event)
            return
        menu = QMenu(self)
        status = int(self._edit_meta.get("status", 1))
        toggle = menu.addAction("切换为转弯" if status == 1 else "切换为作业")
        toggle.triggered.connect(lambda: self.trigger_edit_action("toggle_segment"))
        add_point = menu.addAction("在中点添加航点")
        add_point.triggered.connect(lambda: self.trigger_edit_action("add_waypoint"))
        node_start = int(self._edit_meta.get("node_start", -1))
        node_end = int(self._edit_meta.get("node_end", -1))
        internal_nodes = list(range(node_start + 1, node_end))
        if internal_nodes:
            delete_menu = menu.addMenu("删除内部航点")
            for node_index in internal_nodes:
                action = delete_menu.addAction(f"航点 {node_index + 1}")
                action.triggered.connect(
                    lambda _checked=False, value=node_index:
                    self.trigger_edit_action("delete_waypoint", value)
                )
        if self._is_work:
            menu.addSeparator()
            add_work = menu.addAction("在此后新增作业线")
            add_work.triggered.connect(lambda: self.trigger_edit_action("add_work_after"))
            reverse_work = menu.addAction("反向当前作业线")
            reverse_work.triggered.connect(lambda: self.trigger_edit_action("reverse_work"))
            delete_work = menu.addAction("删除整条作业线")
            delete_work.triggered.connect(lambda: self.trigger_edit_action("delete_work"))
        menu.addSeparator()
        validate = menu.addAction("验证当前人工路线")
        validate.triggered.connect(lambda: self.trigger_edit_action("validate_route"))
        menu.exec(event.globalPos())


class RouteInfoPanel(QWidget):
    segment_selected = Signal(int)
    service_point_selected = Signal(str, int)
    edit_segment_selected = Signal(int, int)
    edit_action_requested = Signal(str, int, int)
    work_reorder_requested = Signal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent); self._segments = []; self._build_ui()
    def _build_ui(self):
        lo = QVBoxLayout(self); lo.setContentsMargins(0,0,0,0); lo.setSpacing(0)
        self._title_hdr = QLabel("路径结果"); self._title_hdr.setStyleSheet(f"font-size:13px;font-weight:600;color:{COLORS['text']};padding:8px 14px 2px;")
        self._summary = QLabel("作业线 0 / 转弯段 0")
        self._summary.setWordWrap(True)
        self._summary.setMinimumWidth(0)
        self._summary.setStyleSheet(f"font-size:11px;color:{COLORS['text_dim']};padding:2px 14px 4px;")
        self._safety = QLabel("路径状态：未评估")
        self._safety.setWordWrap(True)
        self._safety.setStyleSheet(f"font-size:11px;color:{COLORS['orange']};padding:2px 14px 6px;")
        self._edit_hint = QLabel("单击联动画布｜右击编辑或验证｜拖动作业线排序")
        self._edit_hint.setWordWrap(True)
        self._edit_hint.setStyleSheet(
            f"font-size:10px;color:{COLORS['text_accent']};padding:0 14px 6px;"
        )
        self._edit_hint.setVisible(False)
        lo.addWidget(self._title_hdr); lo.addWidget(self._summary); lo.addWidget(self._safety)
        lo.addWidget(self._edit_hint)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        self._scroll = sc
        self._list_container = QWidget(); self._list_layout = QVBoxLayout(self._list_container); self._list_layout.setContentsMargins(8,0,8,0); self._list_layout.setSpacing(2)
        self._empty_label = QLabel("尚无路径结果。"); self._empty_label.setStyleSheet(f"font-size:12px;color:{COLORS['text_dimmer']};padding:8px;")
        self._list_layout.addWidget(self._empty_label); self._list_layout.addStretch(); sc.setWidget(self._list_container); lo.addWidget(sc,1)
    def _clear_list(self):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0); w = item.widget()
            if w and w != self._empty_label: w.deleteLater()
        self._segments.clear()
        self._list_layout.addWidget(self._empty_label)
        self._list_layout.addStretch()
    def _add_row(self, row):
        self._segments.append(row)
        self._list_layout.insertWidget(max(0, self._list_layout.count()-1), row)

    def _set_editable(self, editable):
        self._edit_hint.setVisible(bool(editable))
    @staticmethod
    def _segment_label(segment_type):
        return {
            "work": "作业线", "turn_approach": "转弯接近", "turn": "正式调头",
            "turn_reverse": "倒车段", "turn_aux": "调头辅助", "entry": "进田段",
            "exit": "出田段", "unload": "卸粮段",
        }.get(str(segment_type), "其他路段")
    def update_route(self, data, validation=None):
        self._set_editable(False)
        data = data or {}
        tracks = data.get("tracks", []); turns = data.get("turns", [])
        validation = validation or data.get("validation", {}) or {}
        ordered = list(data.get("ordered_segments") or [])
        if not ordered:
            ordered = [dict(segment, type=segment.get("type", "work")) for segment in tracks]
            ordered.extend(
                dict(segment, type=segment.get("type", "turn")) for segment in turns
            )
        self._clear_list(); self._empty_label.setVisible(not ordered)
        counts = {}
        for fallback_index, segment in enumerate(ordered):
            segment_type = str(segment.get("type", "turn"))
            counts[segment_type] = counts.get(segment_type, 0) + 1
            segment_index = int(segment.get("segment_index", fallback_index))
            length = float(segment.get("length", self._calc_distance(segment.get("points", []))) or 0.0)
            row = RouteSegmentRow(
                segment_index, self._segment_label(segment_type), length,
                segment_type == "work", segment_type=segment_type,
            )
            row.clicked.connect(self.segment_selected.emit); self._add_row(row)
        detail_parts = [f"作业线 {counts.get('work', 0)}", f"正式调头 {counts.get('turn', 0)}"]
        if counts.get("turn_approach"): detail_parts.append(f"接近 {counts['turn_approach']}")
        if counts.get("turn_reverse"): detail_parts.append(f"倒车 {counts['turn_reverse']}")
        service_count = sum(counts.get(key, 0) for key in ("entry", "exit", "unload"))
        if service_count: detail_parts.append(f"进出田/卸粮 {service_count}")
        total = float(validation.get("total_length_m", data.get("total_length_m", 0.0)) or 0.0)
        work = float(validation.get("work_length_m", data.get("work_length_m", 0.0)) or 0.0)
        turn = float(validation.get("turn_length_m", data.get("turn_length_m", 0.0)) or 0.0)
        coverage = validation.get(
            "planned_target_geometric_coverage_pct",
            validation.get("crop_mask_geometric_coverage_pct"),
        )
        rolling = validation.get("rolling_canopy_pct")
        outside = validation.get("track_outside_field_pct")
        forbidden = validation.get("track_forbidden_overlap_pct")
        efficiency = validation.get("field_efficiency_pct", validation.get("field_efficiency"))
        if total > 0: detail_parts.append(f"总长 {total:.1f}m")
        if work > 0: detail_parts.append(f"作业 {work:.1f}m")
        if turn > 0: detail_parts.append(f"转弯 {turn:.1f}m")
        if coverage is not None: detail_parts.append(f"覆盖 {float(coverage):.1f}%")
        if rolling is not None: detail_parts.append(f"碾压 {float(rolling):.1f}%")
        if efficiency is not None: detail_parts.append(f"效率 {float(efficiency):.1f}%")
        issues = validation.get("issues") or []
        if issues: detail_parts.append(f"风险 {len(issues)}处")
        if forbidden is None:
            forbidden_text = "禁行区 未确认"
        elif abs(float(forbidden)) <= 1e-9:
            forbidden_text = "禁行区 无侵入"
        else:
            forbidden_text = f"禁行区 {float(forbidden):.1f}%"
        main_parts = [
            f"预计碾压 {float(rolling):.1f}%" if rolling is not None else "预计碾压 --",
            f"作物掩膜覆盖 {float(coverage):.1f}%" if coverage is not None else "作物掩膜覆盖 --",
            f"越界 {float(outside):.1f}%" if outside is not None else "越界 --",
            forbidden_text,
            f"总长 {total:.1f}m" if total > 0 else "总长 --",
        ]
        self._summary.setText(" / ".join(main_parts))
        self._summary.setToolTip(" / ".join(detail_parts))
        readiness = data.get("machine_readiness", {}) or {}
        trial_ready = bool(readiness.get("controlled_trial_ready", False))
        if not bool(validation.get("valid", False)):
            execution_label = "路径校验未通过"
        elif trial_ready:
            execution_label = "路径校验通过"
        else:
            execution_label = "待补充试验记录"
        self._safety.setText(f"路径状态：{execution_label}")
        blockers = list(readiness.get("controlled_trial_blockers") or [])
        blocker_labels = {
            "path_validation_failed": "路径验证未通过",
            "turn_hard_errors_present": "调头轨迹存在硬错误",
            "footprint_optimized_work_lines_not_used": "未启用足迹优化",
            "field_trial_validation_profile_not_used": "未启用田间试验检查",
            "confirmed_field_boundary_missing": "田块边界未确认",
            "semantic_support_missing": "场地支撑层缺失",
            "confirmed_forbidden_map_missing": "禁行区尚未确认",
            "planning_resolution_too_coarse": "规划分辨率不足",
            "footprint_contract_mismatch": "足迹模型或路径参考点不一致",
            "harvester_parameter_source_unverified": "农机参数来源未完成实测",
            "planning_recovered_work_bands": "包含自动恢复作业带",
        }
        readable_blockers = [blocker_labels.get(value, value) for value in blockers]
        limits = validation.get("validation_limits_pct", {}) or {}
        details = [f"待补充: {'；'.join(readable_blockers)}" if blockers else "田间试验检查已通过"]
        planning_factors = data.get("planning_factors", {}) or {}
        source_map = dict(
            planning_factors.get("harvester_param_sources")
            or (readiness.get("harvester_parameter_source_status", {}) or {}).get(
                "sources"
            )
            or {}
        )
        unverified_sources = [
            f"{HARVESTER_PARAM_LABELS.get(key, key)}（{HARVESTER_SOURCE_LABELS.get(source, source)}）"
            for key, source in source_map.items()
            if source != "user_measured"
        ]
        if unverified_sources:
            details.append("参数来源未完成实测：" + "、".join(unverified_sources))
        if limits:
            details.append(
                "阈值(%): 田界<={track_outside_field:g}, 支撑<={track_outside_support:g}, "
                "不确定<={track_uncertain_overlap:g}, 禁行<={track_forbidden_overlap:g}".format(**limits)
            )
        self._safety.setToolTip("\n".join(details))

    def update_editable_route(self, records):
        """DECISION-019: render direct editors for the shared control route."""
        records = [dict(record) for record in (records or [])]
        self._set_editable(True)
        self._clear_list()
        self._empty_label.setVisible(not records)
        work_count = sum(int(record.get("status", 1)) == 1 for record in records)
        turn_count = len(records) - work_count
        total_length = sum(float(record.get("length_m", 0.0) or 0.0) for record in records)
        display_counts = {"work": 0, "turn": 0}
        for fallback_index, record in enumerate(records):
            segment_index = int(record.get("segment_index", fallback_index))
            segment_type = str(record.get("type", "work"))
            display_counts[segment_type] = display_counts.get(segment_type, 0) + 1
            row = RouteSegmentRow(
                segment_index,
                self._segment_label(segment_type),
                float(record.get("length_m", 0.0) or 0.0),
                segment_type == "work",
                segment_type=segment_type,
                point_count=int(record.get("point_count", 2) or 2),
                edit_meta=record,
                display_number=record.get(
                    "display_number", display_counts[segment_type]
                ),
            )
            row.edit_clicked.connect(self.edit_segment_selected.emit)
            row.edit_action_requested.connect(self.edit_action_requested.emit)
            row.reorder_requested.connect(self.work_reorder_requested.emit)
            self._add_row(row)
        self._summary.setText(
            f"作业线 {work_count} / 转弯段 {turn_count} / 总长 {total_length:.1f}m"
        )
        self._summary.setToolTip("人工控制路线；卡片和画布共享同一几何与顺序")
        self._safety.setText("路径状态：手动编辑后需重新验证")
        self._safety.setToolTip("右击任一路径卡片，选择“验证当前人工路线”重新计算指标")

    def set_edit_selection(self, selected_type, selected_index):
        try:
            selected_index = int(selected_index)
        except (TypeError, ValueError):
            selected_index = -1
        for row in self._segments:
            meta = row._edit_meta
            if not meta:
                row.set_edit_selected(False)
                continue
            node_start = int(meta.get("node_start", -1))
            node_end = int(meta.get("node_end", -1))
            edge_start = int(meta.get("edge_start", -1))
            edge_end = int(meta.get("edge_end", -1))
            selected = (
                selected_type == "edge" and edge_start <= selected_index <= edge_end
            ) or (
                selected_type == "node" and node_start <= selected_index <= node_end
            )
            row.set_edit_selected(selected)
    def update_from_auto_path(self, auto_path):
        self._set_editable(False)
        self._clear_list(); self._empty_label.setVisible(not bool(auto_path))
        counts = {}; total = work_len = turn_len = 0.0
        for idx, seg in enumerate(auto_path or []):
            typ = str(getattr(seg, 'segment_type', 'work')); is_work = typ == 'work'
            counts[typ] = counts.get(typ, 0) + 1
            length = float(getattr(seg, 'length_m', 0.0) or 0.0); total += length
            if is_work: work_len += length
            elif typ.startswith("turn"): turn_len += length
            row = RouteSegmentRow(idx, self._segment_label(typ), length, is_work, segment_type=typ)
            row.clicked.connect(self.segment_selected.emit); self._add_row(row)
        parts = [f"作业线 {counts.get('work', 0)}", f"正式调头 {counts.get('turn', 0)}"]
        if counts.get("turn_approach"): parts.append(f"接近 {counts['turn_approach']}")
        if counts.get("turn_reverse"): parts.append(f"倒车 {counts['turn_reverse']}")
        if total > 0: parts.append(f"总长 {total:.1f}m")
        if work_len > 0: parts.append(f"作业 {work_len:.1f}m")
        if turn_len > 0: parts.append(f"转弯 {turn_len:.1f}m")
        self._summary.setText(" / ".join(parts))
        self._safety.setText("路径状态：手动编辑后需重新验证")
        self._safety.setToolTip("手动编辑后必须重新规划并完成全部验证")
    def update_service_points(self, entry_point=None, exit_point=None, unload_points=None):
        self._set_editable(False)
        self._clear_list(); items=[]
        if entry_point: items.append(("entry",0,"起点",True))
        if exit_point: items.append(("exit",0,"终点",False))
        for i,_ in enumerate(unload_points or []): items.append(("unload",i,f"卸粮点 {i+1}",False))
        self._empty_label.setVisible(not items)
        for row_idx,(kind,point_idx,label,is_work) in enumerate(items):
            row=RouteSegmentRow(row_idx,label,0.0,is_work); row.clicked.connect(lambda _r,k=kind,i=point_idx: self.service_point_selected.emit(k,i)); self._add_row(row)
        self._summary.setText(f"起点 / 可选终点与卸粮点 {len(items)}")
        self._safety.setText("路径状态：尚未规划")
        self._safety.setToolTip("")
    def clear_info(self):
        self._set_editable(False)
        self._clear_list(); self._empty_label.setVisible(True); self._summary.setText("作业线 0 / 转弯段 0")
        self._safety.setText("路径状态：未评估"); self._safety.setToolTip("")
    def refresh_theme(self):
        self.setStyleSheet(f"background:{COLORS['bg']};")
        self._scroll.setStyleSheet(
            f"QScrollArea{{background:{COLORS['bg']};border:none;}}"
        )
        self._scroll.viewport().setStyleSheet(f"background:{COLORS['bg']};")
        self._list_container.setStyleSheet(f"background:{COLORS['bg']};")
        self._title_hdr.setStyleSheet(f"font-size:13px;font-weight:600;color:{COLORS['text']};padding:8px 14px 2px;")
        self._summary.setStyleSheet(f"font-size:11px;color:{COLORS['text_dim']};padding:2px 14px 4px;")
        self._safety.setStyleSheet(f"font-size:11px;color:{COLORS['orange']};padding:2px 14px 6px;")
        self._edit_hint.setStyleSheet(
            f"font-size:10px;color:{COLORS['text_accent']};padding:0 14px 6px;"
        )
        self._empty_label.setStyleSheet(f"font-size:12px;color:{COLORS['text_dimmer']};padding:8px;")
        for row in self.findChildren(RouteSegmentRow):
            row.refresh_theme()
    @staticmethod
    def _calc_distance(points):
        pts = [(float(p[0]), float(p[1])) for p in points or [] if len(p) >= 2]
        return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) for i in range(1,len(pts)))


class LogPanel(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent); self.setReadOnly(True); self.setMinimumHeight(120)
        self.refresh_theme()
    def refresh_theme(self):
        self.setStyleSheet(
            f"QTextEdit{{background:{COLORS['bg_darkest']};color:{COLORS['text']};"
            f"border:1px solid {COLORS['border']};border-radius:6px;"
            "font-family:'Cascadia Code','Consolas',monospace;font-size:12px;padding:6px;}"
        )
    def _append(self, level, msg):
        ts = time.strftime("%H:%M:%S")
        self.append(f"[{ts}] {msg}")
    def info(self, m): self._append("INFO", m)
    def success(self, m): self._append("SUCCESS", m)
    def warn(self, m): self._append("WARN", m)
    def error(self, m): self._append("ERROR", m)


class ParamPanel(QWidget):
    params_changed = Signal(); load_tif_requested = Signal(); load_model_requested = Signal(); run_requested = Signal()
    export_geojson_requested = Signal(); export_csv_requested = Signal(); export_img_requested = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tif_path = ""
        self.model_path = ""
    def set_tif_path(self, p):
        self.tif_path = str(p or "")
    def set_model_path(self, p):
        self.model_path = str(p or "")


class ToolPanel(QWidget):
    tool_changed = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tool = ""
    def set_tool(self, tool):
        self.current_tool = str(tool or "")
        self.tool_changed.emit(self.current_tool)


class StatusSteps(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_index = 0
    def set_current(self, idx):
        self.current_index = int(idx or 0)
