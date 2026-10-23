import math
import time
from PySide6.QtCore import (
    Qt, Signal, QMimeData, QSize, QEasingCurve, QParallelAnimationGroup,
    QPropertyAnimation, QRectF, QTimer,
)
from PySide6.QtGui import QDrag, QFont, QPainter, QColor, QPen, QBrush, QPainterPath
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QScrollArea, QFrame, QTextEdit, QToolButton, QMenu,
    QApplication, QCheckBox, QSizePolicy,
)
from pyside6_app.styles import COLORS, ui_font, FONT_FAMILY_CSS
from pyside6_app.task_state import TASK_ORDER
from pyside6_app.ui_icons import chevron_icon, check_icon
from pyside6_app.ui_motion import (
    BACK_OUT, BobIcon, BreathingDot, ChevronToggle, ShimmerBar,
    SpringButton, tween_value,
)

# 字体统一由 QApplication.setFont(build_app_font()) 提供（数字新罗马 + 中文微软雅黑）。
# 不要在这里或控件样式表里再写 font-family：QSS 的字体族会压过 setFont，
# 之前就是因为它，步骤标题回退成了 SimHei。

# 真实窗口 show() 后逐级缩窄测得：高级设置或路径规划卡片展开且出现纵向滚动条时，
# 328 px 仍会产生 1 px 横向滚动，330 px 起全部控件不再横向挤压。
TASK_PANEL_COMPACT_MIN_WIDTH = 330

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
    {"id": "import_img", "label": "影像导入", "hint": "加载航拍正射影像 GeoTIFF 文件（必须含地理坐标）。", "color": "#6FA37F", "action": "导入"},
    {"id": "draw_field", "label": "田块圈选", "hint": "在影像上圈选田块作业范围。", "color": "#5EAF6E", "action": "圈选"},
    {"id": "segment", "label": "作物识别", "hint": "使用训练好的模型识别作物种植带（像素级分割）。", "color": "#649C7E", "action": "运行"},
    {"id": "process", "label": "掩膜优化", "hint": "对识别掩膜去噪、聚合，提取清晰的种植带几何。", "color": "#648C82", "action": "处理"},
    {"id": "params", "label": "农机参数", "hint": "设置收割机幅宽、轮距、转弯半径等参数。", "color": "#AC7E56", "action": "设置"},
    {"id": "centerline_generate", "label": "生成作业线", "hint": "从识别出的种植带生成收割路线（作业线）。", "color": "#5EAF8A", "action": "生成"},
    {"id": "centerline_select", "label": "选择作业线", "hint": "勾选要采用的作业线；每条4行种植带对应一条作业线。", "color": "#4F9D8E", "action": "选择"},
    {"id": "entry", "label": "起点（必选）", "hint": "可自由布设或拖动；规划必须从离起点最近的作业线端点开始。", "color": "#589C94", "action": "布设"},
    {
        "id": "exit",
        "label": "终点（选填）",
        "hint": "默认沿最终作业段延伸并写入终端文件；可在画布上拖动调整。",
        "color": "#C76B6B",
        "action": "调整",
        "optional": True,
    },
    {"id": "unload", "label": "卸粮点（选填）", "hint": "按需要添加一个或多个卸粮点，可为空。", "color": "#C9A64A", "action": "布设", "optional": True},
    {"id": "plan", "label": "路径规划", "hint": "生成完整收割路径（含转弯、进出场）。", "color": "#589862", "action": "生成"},
    {"id": "simulate", "label": "模拟演示", "hint": "沿规划路径模拟农机运动，检验可行性。", "color": "#4C8C7C", "action": "演示"},
    {
        "id": "export",
        "label": "TXT路径导出",
        "hint": "按终端 $PATH 格式导出路径文件（001.txt、002.txt）；GeoJSON 在文件菜单单独导出。",
        "color": "#4484BE",
        "action": "导出",
    },
]

STEP_GROUPS = [
    {"label": "影像导入", "tasks": ["import_img"]},
    {"label": "田块圈选", "tasks": ["draw_field"]},
    {"label": "作物识别", "tasks": ["segment"]},
    {"label": "掩膜优化", "tasks": ["process"]},
    {"label": "生成作业线", "tasks": ["centerline_generate"]},
    {"label": "选择作业线", "tasks": ["centerline_select"]},
    # 高级设置是可选步骤（默认参数即来自配置），不再作为规划前置条件，
    # 因此放回流程末尾附近，避免让人以为"必须先设置"。
    {"label": "路径规划", "tasks": ["entry", "exit", "unload", "plan"]},
    {"label": "高级设置", "tasks": ["params"]},
    {"label": "模拟演示", "tasks": ["simulate"]},
    {"label": "TXT路径导出", "tasks": ["export"]},
]


def _task_def(tid):
    return next((item for item in TASK_DEFS if item["id"] == tid), None)


class TaskRow(QWidget):
    clicked = Signal(str)
    #: 只有「操作按钮」被按下时发这个信号；行本身只发 ``clicked``。
    #: 两者分开，是因为「选择作业线」这类步骤点行只应该展开卡片，
    #: 点按钮才应该确认选择（否则用户想打开卡片看一眼就直接被确认了）。
    action_clicked = Signal(str)
    delete_clicked = Signal(str)

    def __init__(self, task_def, parent=None):
        super().__init__(parent)
        self.setObjectName("taskRow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#taskRow{background:transparent;}")
        self.tid = task_def["id"]
        self._color = task_def["color"]
        self._done = False
        self._status_key = "pending"
        self._in_progress = False
        self._status_text = "待处理"
        self._blocker = ""
        self._flash = ""           # 点击后短暂显示的阻塞原因
        self._flash_until = 0.0    # monotonic 时间戳
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)
        # 两行结构：第一行标题，第二行"状态 + 按钮"。
        # 之前把按钮放到第三行，44 px 的行高装不下，按钮被卡片边界裁掉；
        # 现在固定两行，行高由 sizeHint 决定，所有子控件都在布局里。
        self._badge = QLabel("")
        self._badge.setFixedSize(14, 14)
        self._badge.setVisible(False)   # replaced by the breathing status dot
        self._label = QLabel(task_def["label"])
        self._label.setFont(ui_font(10))
        self._optional = bool(task_def.get("optional", False))
        # Optional steps carry their marking in the label itself ("（选填）")
        # plus an accent tone: a separate chip overflowed the compact panel.
        self._label.setStyleSheet(
            f"color:{COLORS.get('text_accent', COLORS['text']) if self._optional else COLORS['text']};"
            f"font-size:14px;"
        )
        self._label.setMinimumWidth(0)
        self._optional_tip = "选填步骤：可以跳过，但一旦设置会改变生成的路径结果"
        self._label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self._status = QLabel(self._status_text)
        self._status.setFont(ui_font(9))
        self._status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._status.setMinimumWidth(0)
        self._status.setStyleSheet(
            f"color:{COLORS['text_dim']};font-size:12px;"
        )

        self._action_btn = SpringButton(task_def["action"])
        self._action_btn.setFixedSize(58, 28)
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['accent']};color:{COLORS['on_accent']};"
            f"border:none;border-radius:6px;font-size:13px;font-weight:600;padding:0;"
            f"}}"
            f"QPushButton:hover{{background:{COLORS['accent_hover']};}}"
        )
        self._action_btn.clicked.connect(lambda: self.action_clicked.emit(self.tid))

        self._delete_btn = SpringButton("回退")
        self._delete_btn.setFixedSize(52, 28)
        self._delete_btn.setToolTip("回退到此步骤之前，并清除后续结果")
        self._delete_btn.setStyleSheet(self._secondary_button_style())
        self._delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.tid))

        self._busy = BreathingDot(8)
        self._busy.setVisible(False)
        self._progress = ShimmerBar()
        # 进度条位置常驻（4 px）：运行时只把轨道显示出来，不改变行高，
        # 这样卡片高度从头到尾是稳定的，不会出现"进度条一出现就压住下一行"。
        self._progress.set_idle(True)
        self._progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        title_row.addWidget(self._busy)
        title_row.addWidget(self._label, 1)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)
        action_row.addWidget(self._status, 1)
        action_row.addWidget(self._action_btn)
        action_row.addWidget(self._delete_btn)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(6)
        outer.addLayout(title_row)
        outer.addLayout(action_row)
        outer.addWidget(self._progress)
        # 行高由布局实际 sizeHint 决定：进度条显示/隐藏时高度会变，
        # 写死一个数值就会出现"进度条压住按钮"或"按钮被裁"。
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._sync_row_height()
        self._update_badge()

    def _sync_row_height(self):
        """按当前子控件（含进度条）的实际高度重算行高。"""
        lay = self.layout()
        if lay is None:
            return
        lay.invalidate()
        lay.activate()
        target = max(56, int(lay.sizeHint().height()))
        if target == int(getattr(self, "_synced_height", 0) or 0):
            return
        self._synced_height = target
        self.setMinimumHeight(target)
        self._notify_card_height_change()

    def _notify_card_height_change(self):
        """行高变了，让外层折叠卡片按真实内容重算高度。

        卡片展开时会把内容区的 maximumHeight 固定成「展开那一刻」的高度。
        进度条是运行中才出现的，行因此变高；不同步卡片的话，卡片仍按旧高度
        分配空间，进度条就会压住下面的「处理强度」标签（截图里的文字遮挡）。
        """
        parent = self.parentWidget()
        depth = 0
        while parent is not None and depth < 8:
            refresh = getattr(parent, "refresh_content_geometry", None)
            if callable(refresh):
                refresh()
                return
            parent = parent.parentWidget()
            depth += 1

    def mousePressEvent(self, event):
        """整行可点：pending 行的操作按钮是禁用的，点击必须由行本身转成 clicked，
        否则 _on_task_clicked 里的 blocker 提示永远走不到（P1）。"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.tid)
        super().mousePressEvent(event)

    def flash_blocker(self, message):
        """在一行内就地显示"为什么现在不能做"，2.6 秒后自动复原。"""
        self._flash = str(message or "")
        if not self._flash:
            return
        self._flash_timer.start(2600)
        self._render_flash()

    def _clear_flash(self):
        self._flash = ""
        self._render_flash()

    def _render_flash(self):
        if self._flash:
            self._status.setText(self._flash)
            self._status.setStyleSheet(
                f"color:{COLORS.get('orange', '#E8A33D')};font-size:12px;"
                f"font-weight:600;"
            )
        else:
            self._status.setText(self._status_text)
            self.refresh_theme()

    def set_status(self, done=False, enabled=True, in_progress=False, progress_pct=-1, text=None, status="pending", clickable=None):
        if clickable is None:
            clickable = enabled
        # 状态没变就直接返回：勾选作业线时每次都会重算全表状态，13 行各写一遍
        # 样式表（Qt 每次都要重新解析 CSS）是勾选卡顿的主要来源。
        signature = (
            bool(done), bool(enabled), bool(clickable), bool(in_progress),
            int(progress_pct), text, str(status),
        )
        if signature == getattr(self, "_status_signature", None) and not self._flash:
            return
        self._status_signature = signature
        self._done = bool(done)
        self._in_progress = bool(in_progress)
        self._status_key = status
        self._progress_pct = progress_pct
        # 「能不能点」与「看起来是否可用」是两件事：pending 行可点，但不该因此失去灰态。
        # 灰按 enabled 记下来，refresh_theme 不能再拿 isEnabled() 反推（P1 附带缺陷）。
        self._enabled_visual = bool(enabled)
        self.setEnabled(bool(clickable))
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
        if getattr(self, "_optional", False):
            self.setToolTip(getattr(self, "_optional_tip", ""))
            label_color = COLORS.get("text_accent", COLORS["text"])
        else:
            label_color = COLORS["text"]
        self._label.setStyleSheet(
            f"color:{label_color};font-size:14px;"
        )
        self._status_text = display_text
        if not self._flash:
            self._status.setText(display_text)
            self._status.setStyleSheet(
                f"color:{color};font-size:12px;"
            )
        opacity = "1.0" if enabled else "0.55"
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['accent']};color:{COLORS['on_accent']};"
            f"border:none;border-radius:6px;font-size:13px;font-weight:600;padding:0;"
            f"}}"
            f"QPushButton:hover{{background:{COLORS['accent_hover']};}}"
            f"QPushButton:disabled{{background:{COLORS.get('bg_light', '#3A3A44')};"
            f"color:{COLORS.get('text_dim', '#8B909A')};}}"
        )
        self._update_badge()
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
        opacity = "1.0" if getattr(self, "_enabled_visual", True) else "0.55"
        self._status.setStyleSheet(f"color:{color};font-size:12px;")
        label_color = (
            COLORS.get("text_accent", COLORS["text"])
            if getattr(self, "_optional", False)
            else COLORS["text"]
        )
        self._label.setStyleSheet(
            f"color:{label_color};font-size:14px;"
        )
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['accent']};color:{COLORS['on_accent']};"
            f"border:none;border-radius:6px;font-size:13px;font-weight:600;padding:0;"
            f"}}"
            f"QPushButton:hover{{background:{COLORS['accent_hover']};}}"
            f"QPushButton:disabled{{background:{COLORS.get('bg_light', '#3A3A44')};"
            f"color:{COLORS.get('text_dim', '#8B909A')};}}"
        )
        self._delete_btn.setStyleSheet(self._secondary_button_style())
        self._update_badge()
        self.setStyleSheet(f"#taskRow{{background:transparent;opacity:{opacity};}}")

    def _update_badge(self):
        """Status dot: green done, accent in progress, dim pending."""
        if getattr(self, "_done", False):
            color = COLORS.get("green", "#6CCB6E")
        elif getattr(self, "_in_progress", False) or getattr(self, "_status_key", "") == "available":
            color = COLORS.get("accent", "#60CDFF")
        elif getattr(self, "_status_key", "") == "failed":
            color = COLORS.get("red", "#F17070")
        else:
            color = COLORS.get("border_light", "#48485A")
        self._badge.setStyleSheet(
            f"background:{color};border-radius:4px;")
        busy = bool(
            getattr(self, "_in_progress", False)
            or getattr(self, "_status_key", "") in ("available", "running"))
        dot = getattr(self, "_busy", None)
        if dot is not None:
            dot.setVisible(True)
            if getattr(self, "_done", False) or getattr(self, "_status_key", "") == "success":
                dot.set_tone("ready", COLORS.get("green", "#34D399"))
            elif getattr(self, "_status_key", "") == "failed":
                dot.set_tone("error", COLORS.get("red", "#E85D63"))
            elif busy:
                dot.set_tone("busy", COLORS.get("accent", "#34D399"))
            else:
                dot.set_tone("idle", COLORS.get("border_light", "#334155"))
        bar = getattr(self, "_progress", None)
        if bar is not None:
            pct = int(getattr(self, "_progress_pct", -1))
            bar.set_idle(pct < 0)
            if pct >= 0:
                bar.set_ratio(pct / 100.0 if pct > 1 else float(pct))
            bar.set_colors(COLORS.get("accent", "#34D399"),
                           COLORS.get("border", "#263241"))
            # 位置常驻，行高不随进度条出现而变化（见 __init__ 的说明）。

    @staticmethod
    def _secondary_button_style():
        """Neutral secondary action: same geometry as the primary button,
        outline style instead of a saturated brown fill."""
        return (
            f"QPushButton{{background:transparent;color:{COLORS['text_dim']};"
            f"border:1px solid {COLORS['border_light']};border-radius:6px;"
            f"font-size:13px;font-weight:600;padding:0;}}"
            f"QPushButton:hover{{background:{COLORS['bg_hover']};color:{COLORS['text']};"
            f"border-color:{COLORS['accent']};}}"
            f"QPushButton:disabled{{color:{COLORS['btn_disabled_text']};"
            f"border-color:{COLORS['border']};}}"
        )


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
        self._hover = 0.0
        self._active = False
        self._done = False
        self.setMouseTracking(True)
        lo = QVBoxLayout(self)
        # 浮岛式卡片：内外都要留白，堆太密看着累（用户明确要求）。
        lo.setContentsMargins(12, 10, 12, 10)
        lo.setSpacing(8)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        # Rotating chevron instead of swapping two icons, so the expand/collapse
        # reads as one continuous motion.
        self._toggle = ChevronToggle()
        self._title = QLabel(title)
        self._title.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._title.setFont(ui_font(10))
        self._title.setStyleSheet(f"color:{COLORS['text']};font-size:14px;font-weight:600;")
        # Status dot on the header so a collapsed card still reports its state.
        self._status_dot = BreathingDot(8)
        self._status_dot.set_tone("idle", COLORS.get("border_light", "#334155"))
        self._status_dot.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._mark = QLabel("")
        self._mark.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._mark.setFixedSize(16, 28)
        self._mark.setAlignment(Qt.AlignCenter)
        self._mark.setVisible(False)
        head.addWidget(self._toggle)
        head.addWidget(self._status_dot)
        head.addWidget(self._title, 1)
        head.addWidget(self._mark)
        lo.addLayout(head)
        # 内容区是独立一层：垂直方向 Fixed + maximumHeight 动画，
        # 这样表头永远在卡片顶部不动，内容从 0 高度向下长出来。
        # 参考 qt-collapsible-section（Qt Wiki 的经典 Section 实现）：
        # 内容区与 Section 本体的 maximumHeight 并行动画。
        self._content = QWidget()
        self._content.setAttribute(Qt.WA_StyledBackground, True)
        self._content.setStyleSheet("background:transparent;")
        self._content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        cl = QVBoxLayout(self._content); cl.setContentsMargins(12, 8, 4, 6); cl.setSpacing(8)
        for row in rows: cl.addWidget(row)
        self._content.setMaximumHeight(0)
        self._content.setVisible(False)
        lo.addWidget(self._content)
        # 展开/收起用并行动画组：内容区高度 + 卡片 min/max 高度同时变。
        self._expand_group = QParallelAnimationGroup(self)
        self._expand_duration = 220
        self._expand_group.finished.connect(self._on_expand_finished)
        self._toggle.clicked.connect(self._toggle_expanded)
        self.set_expanded(False, animate=False)
        self.refresh_state()

    def _toggle_expanded(self):
        """只发信号，由 TaskPanel 统一决定展开还是收起。

        旧实现先 `set_expanded(not self._expanded)` 再 emit，于是
        `_on_card_toggled` 读到的是**已经切换过**的状态，又把它切回去，
        表现为"点表头没反应"。
        """
        self.toggled.emit()

    def mousePressEvent(self, event):
        # 只有点在表头区域才切换展开状态；点内容区不应误收起。
        if event.button() == Qt.LeftButton and event.position().y() <= self._collapsed_height():
            self._toggle_expanded()
            event.accept()
            return
        super().mousePressEvent(event)

    # ------------------------------------------------------------------ 高度
    def _collapsed_height(self):
        """折叠高度 = 表头那一行的高度（含卡片内边距）。"""
        layout = self.layout()
        if layout is None:
            return 40
        head = layout.itemAt(0)
        head_h = 28
        if head is not None and head.layout() is not None:
            head_h = head.layout().sizeHint().height()
        m = layout.contentsMargins()
        return max(40, int(head_h + m.top() + m.bottom()))

    def _content_height(self):
        """内容区完整展开时的高度（含它自己的内边距）。"""
        cl = self._content.layout()
        if cl is None:
            return 0
        cl.invalidate()
        cl.activate()
        return max(0, int(cl.sizeHint().height()))

    def set_expanded(self, expanded, animate=True):
        expanded = bool(expanded)
        changed = expanded != self._expanded
        self._expanded = expanded
        self._toggle.set_expanded(expanded, animate_motion=changed)
        if not changed or not animate or not self.isVisible():
            self._apply_expanded_state(expanded)
            return
        self._animate_expansion(expanded)

    def _apply_expanded_state(self, expanded):
        """无动画地把卡片切到目标高度（初始化和打断时用）。"""
        collapsed = self._collapsed_height()
        content_h = self._content_height()
        self._expand_group.stop()
        if expanded:
            self._content.setVisible(True)
            self._content.setMaximumHeight(content_h)
            self.setMinimumHeight(collapsed + content_h)
            self.setMaximumHeight(collapsed + content_h)
        else:
            self._content.setMaximumHeight(0)
            self._content.setVisible(False)
            self.setMinimumHeight(collapsed)
            self.setMaximumHeight(collapsed)
        self.updateGeometry()
        self.update()

    def _animate_expansion(self, expanded):
        """表头不动，内容从 0 高度向下展开（或反向收起）。

        三个动画并行：
          * 内容区 maximumHeight 0 <-> 内容高度
          * 卡片 minimumHeight / maximumHeight 折叠高度 <-> 折叠+内容高度
        表头在布局第一行，内容在第二行，所以标题的相对位置全程不变。
        """
        collapsed = self._collapsed_height()
        content_h = self._content_height()
        target_h = collapsed + content_h
        group = self._expand_group
        group.stop()
        for anim in list(group.animationAt(i) for i in range(group.animationCount())):
            group.removeAnimation(anim)

        if expanded:
            self._content.setVisible(True)
            start_content = max(0, self._content.maximumHeight())
            start_total = max(collapsed, self.minimumHeight())
        else:
            start_content = max(0, self._content.height())
            start_total = max(collapsed, self.height())

        content_anim = QPropertyAnimation(self._content, b"maximumHeight", self)
        content_anim.setStartValue(start_content)
        content_anim.setEndValue(content_h if expanded else 0)
        content_anim.setDuration(self._expand_duration)
        group.addAnimation(content_anim)

        for prop in (b"minimumHeight", b"maximumHeight"):
            anim = QPropertyAnimation(self, prop, self)
            anim.setStartValue(start_total)
            anim.setEndValue(target_h if expanded else collapsed)
            anim.setDuration(self._expand_duration)
            group.addAnimation(anim)

        # 标准缓动：内容高度本身就在变，再加过冲会让底部先空一块再收回。
        for i in range(group.animationCount()):
            group.animationAt(i).setEasingCurve(QEasingCurve.OutCubic)
        group.start()

    def _on_expand_finished(self):
        """收起动画结束后把内容彻底隐藏，避免它继续参与布局。"""
        if not self._expanded:
            self._content.setVisible(False)
            self._content.setMaximumHeight(0)
        else:
            # 展开过程中子控件高度可能变化（进度条出现、选填分组展开），
            # 动画目标是开始时算的旧高度；结束后按真实内容收口一次。
            self._sync_minimum_height()
        self.updateGeometry()
        self.update()

    def _sync_minimum_height(self):
        """无动画地把高度对齐到当前展开状态（运行时子控件可见性变化后调用）。"""
        self._apply_expanded_state(self._expanded)

    def refresh_content_geometry(self):
        """Recompute an expanded card after a runtime child visibility change."""
        self._sync_minimum_height()
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()

    def _chrome(self):
        """Colours for the painted card surface (see ``paintEvent``).

        Mirrors the reference sidebar rows: the row itself is transparent and
        two layers do the work — a 3 px accent rail marking the current step
        and a soft tinted surface that lights up on hover or selection.
        """
        done = bool(getattr(self, "_done", False))
        if done:
            # Done cards keep the plain surface AND no rail: a coloured rail
            # plus tinted fill plus coloured border read as a second, skewed
            # layer behind the card. Only the header check marks completion.
            rail = COLORS.get("bg_card", "#111827")
            surface = COLORS.get("bg_card", "#111827")
            border = COLORS.get("border", "#263241")
        elif getattr(self, "_active", False):
            # Only a rail + border mark the current step; the surface stays the
            # plain card colour. Blending a solid accent_bg (teal) over the card
            # produced a visibly different block that read as a second layer.
            rail = COLORS.get("accent", "#34D399")
            surface = COLORS.get("bg_card", "#111827")
            border = COLORS.get("accent", "#34D399")
        else:
            rail = COLORS.get("border_light", "#334155")
            surface = COLORS.get("bg_card", "#111827")
            border = COLORS.get("border", "#263241")
        return rail, surface, border

    def paintEvent(self, event):
        """Paint card chrome with a rounded clip so a hover fill cannot spill."""
        from PySide6.QtGui import QBrush, QPainterPath
        rail, surface, border = self._chrome()
        hover = max(0.0, min(1.0, getattr(self, "_hover", 0.0)))
        active = bool(getattr(self, "_active", False)) or bool(getattr(self, "_done", False))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 1.5, -0.5, -1.5)
        radius = 8.0
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        painter.setClipPath(clip)

        fill = QColor(surface)
        if hover > 0.01 and not active:
            lift = QColor(COLORS.get("bg_hover", "#1B2433"))
            fill = QColor(
                int(fill.red() + (lift.red() - fill.red()) * hover),
                int(fill.green() + (lift.green() - fill.green()) * hover),
                int(fill.blue() + (lift.blue() - fill.blue()) * hover),
            )
        # Always opaque: a translucent card let the scroll canvas bleed through
        # and read as a skewed shadow behind the rounded rect.
        fill.setAlpha(255)
        painter.fillPath(clip, QBrush(fill))

        painter.setClipping(False)
        # 纯色底上卡片边缘要看得出来：静息描边从 90 提到 130（用户反馈"边缘不清晰"）
        edge = QColor(border)
        edge.setAlpha(int(130 + 80 * hover) if not active else 205)
        painter.setPen(QPen(edge, 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, radius, radius)

        # 3 px status rail on the left edge, inside the rounded corner.
        rail_color = QColor(rail)
        if not active:
            rail_color.setAlpha(int(70 + 150 * hover))
        rail_rect = QRectF(rect.left() + 0.5, rect.top() + 3.0,
                           3.0, max(0.0, rect.height() - 6.0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(rail_color)
        painter.drawRoundedRect(rail_rect, 1.5, 1.5)
        painter.end()

    def enterEvent(self, event):
        self._set_hover(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_hover(0.0)
        super().leaveEvent(event)

    def _set_hover(self, target):
        """Hover lights up with the same curve as the reference rows (160 ms)."""
        from pyside6_app.ui_motion import EASE_IN_OUT, tween_value
        if not self.isVisible():
            self._hover = target
            self.update()
            return
        tween_value(self, "_hover", target, 160, EASE_IN_OUT,
                    lambda: self.update())

    def refresh_state(self, force=False):
        """Card emphasis follows the workflow state, not the task colour.

        Active card: accent 3 px rail on the left edge. Completed card: mint
        check and a hairline success border. Idle card: plain surface, no
        ornament, so ten stacked cards stay quiet until they matter.

        ``force`` 供换主题时使用：状态没变但颜色变了，必须重刷样式。
        """
        # A row may report completion either through _done or through the
        # status key "done"/"success"; checking only _done let a completed card
        # sit on "not done", re-firing the completion sweep on every refresh.
        task_rows = [
            row for row in self._rows
            if hasattr(row, "_done") or hasattr(row, "_status_key")
        ]
        done = bool(task_rows) and all(
            bool(getattr(row, "_done", False))
            or str(getattr(row, "_status_key", "")) in ("done", "success")
            for row in task_rows
        )
        active = bool(getattr(self, "_active", False)) and not done
        self._done = done
        self._was_done = done
        dot = getattr(self, "_status_dot", None)
        tone_color = self.collapsed_status_dot() if dot is not None else None
        # 状态刷新在勾选作业线时会被整表重算；样式没变就不要再写一遍
        # setStyleSheet（Qt 每次都要重新解析 CSS，10 张卡片就是几十毫秒）。
        key = (bool(done), bool(active), tone_color)
        if not force and key == getattr(self, "_refresh_key", None):
            return
        self._refresh_key = key
        self._title.setStyleSheet(
            f"color:{COLORS['text_bright'] if active else COLORS['text']};"
            f"font-size:14px;font-weight:{600 if active else 500};"
            f"")
        if done:
            border = COLORS.get("green", "#34D399")
            self._mark.setPixmap(check_icon(border).pixmap(QSize(14, 14)))
            self._mark.setVisible(True)
        else:
            self._mark.clear()
            self._mark.setVisible(False)
        # Collapsed cards hide their rows, so the header owns the status dot.
        if dot is not None and tone_color is not None:
            dot.set_tone(tone_color[0], tone_color[1])
        # 完成只靠对勾 + 状态点表达，不再播放扫光（用户要求只要交互反馈）。
        # Keep the card's own flag in sync: TaskPanel._step_cards[i]._done is
        # read elsewhere (activation logic, tests) and used to drift.
        # Chrome (surface, border, rail) is painted in paintEvent so the hover
        # fill and the rail can share one rounded clip.
        self.setStyleSheet(
            f"#stepCard{{background:transparent;border:none;}}"
            f"#stepCard QLabel{{}}"
            f"QToolButton{{background:transparent;color:{COLORS['text_dim']};"
            f"border:none;}}"
            f"QToolButton:hover{{background:{COLORS['bg_hover']};border-radius:4px;}}"
        )
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def collapsed_status_dot(self):
        """Return (tone, color) summarising the rows for the collapsed header.

        Collapsed cards hide their rows, so a running or failed row would
        otherwise be invisible; the header carries the dot instead.
        """
        # Only real task rows carry a status; labels/combos live in the same
        # list and must not be counted as pending work.
        rows = [r for r in self._rows if hasattr(r, "_status_key")]
        if not rows:
            return "idle", COLORS.get("border_light", "#334155")
        keys = [str(getattr(r, "_status_key", "pending")) for r in rows]
        if "failed" in keys:
            return "error", COLORS.get("red", "#E85D63")
        if any(getattr(r, "_in_progress", False) for r in rows):
            return "busy", COLORS.get("accent", "#34D399")
        # Callers pass either done=True or status="success"; accept both.
        finished = [
            getattr(r, "_done", False) or str(getattr(r, "_status_key", "")) == "success"
            for r in rows
        ]
        if all(finished):
            return "ready", COLORS.get("green", "#34D399")
        if "available" in keys:
            return "busy", COLORS.get("accent", "#34D399")
        return "idle", COLORS.get("border_light", "#334155")

    def set_active(self, active):
        """Mark this card as the workflow's current step."""
        if getattr(self, "_active", False) == bool(active):
            return
        self._active = bool(active)
        self.refresh_state()


class _OptionalHeader(QWidget):
    """二级折叠分组的表头（整行可点，暗色文字）。"""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class OptionalRowGroup(QWidget):
    """选填行的二级折叠分组：默认收起，表头暗色，展开后露出子行。

    「终点」「卸粮点」是选填项，和必选的「起点」平铺在一张卡片里时，两整行
    会让人误以为必须填。折进二级分组后，卡片只展示必选流程；表头用暗色
    文字报告当前取值（未设置/已设置），需要改的时候再展开。
    """

    def __init__(self, title, rows, parent=None):
        super().__init__(parent)
        self._rows = [row for row in rows if row is not None]
        self._expanded = False
        self._title_text = str(title)
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(2)

        self._header = _OptionalHeader()
        self._header.setObjectName("optionalGroupHeader")
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(6, 2, 6, 2)
        hl.setSpacing(4)
        self._toggle = ChevronToggle()
        self._toggle.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._title = QLabel(self._title_text)
        self._title.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._summary = QLabel("")
        self._summary.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(self._toggle)
        hl.addWidget(self._title, 1)
        hl.addWidget(self._summary)
        self._header.clicked.connect(self.toggle)
        lo.addWidget(self._header)

        self._content = QWidget()
        cl = QVBoxLayout(self._content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        for row in self._rows:
            cl.addWidget(row)
        self._content.setVisible(False)
        lo.addWidget(self._content)
        self.refresh_theme()

    # ---------------------------------------------------------------- state
    def is_expanded(self):
        return bool(self._expanded)

    def toggle(self):
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        changed = expanded != self._expanded
        self._expanded = expanded
        self._toggle.set_expanded(expanded, animate_motion=changed)
        self._content.setVisible(expanded)
        if changed:
            self._notify_card_height_change()

    def set_summary(self, text):
        text = str(text or "")
        if self._summary.text() != text:
            self._summary.setText(text)

    def _notify_card_height_change(self):
        parent = self.parentWidget()
        depth = 0
        while parent is not None and depth < 8:
            refresh = getattr(parent, "refresh_content_geometry", None)
            if callable(refresh):
                refresh()
                return
            parent = parent.parentWidget()
            depth += 1

    def refresh_theme(self):
        self._header.setStyleSheet(
            "#optionalGroupHeader{background:transparent;border-radius:5px;}"
            f"#optionalGroupHeader:hover{{background:{COLORS.get('bg_hover', '#1B2433')};}}"
        )
        self._title.setStyleSheet(
            f"color:{COLORS.get('text_dim', '#8B909A')};font-size:12px;"
            f""
        )
        self._summary.setStyleSheet(
            f"color:{COLORS.get('text_dim', '#8B909A')};font-size:11px;"
            f""
        )
        self._toggle.setStyleSheet(
            "QToolButton{background:transparent;border:none;padding:0;margin:0;}"
        )


class TaskPanel(QWidget):
    load_tif_requested = Signal()
    load_model_requested = Signal()
    run_requested = Signal()
    segment_requested = Signal()
    export_inference_diagnostics_requested = Signal()
    process_requested = Signal()
    export_mask_diagnostics_requested = Signal()
    plan_requested = Signal()
    simulate_requested = Signal()
    params_requested = Signal()
    entry_exit_toggle = Signal(str)
    export_machine_compatibility_requested = Signal()
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
    centerline_generate_requested = Signal()
    centerline_selection_changed = Signal(str, bool)
    centerline_selection_confirm_requested = Signal()
    step_deleted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._backdrop = None
        self.setMinimumWidth(TASK_PANEL_COMPACT_MIN_WIDTH)
        self._task_widgets = {}
        self._step_cards = []
        self._current_step = 0
        self._manual_step = None
        self._user_collapsed = False
        self._forbidden_count = 0
        self._forbidden_confirmed = False
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{COLORS['bg']};")
        lo = QVBoxLayout(self); lo.setContentsMargins(8, 10, 8, 10); lo.setSpacing(6)
        self._title = QLabel("任务清单")
        self._title.setFont(ui_font(12, QFont.Weight.DemiBold))
        self._title.setStyleSheet(f"font-size:17px;font-weight:600;color:{COLORS['text_bright']};")
        lo.addWidget(self._title)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        self._scroll = scroll
        body = QWidget(); bl = QVBoxLayout(body); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(10)
        body.setStyleSheet("background:transparent;")
        for idx, group in enumerate(STEP_GROUPS, start=1):
            rows = []
            optional_rows = []
            for tid in group["tasks"]:
                td = _task_def(tid); row = TaskRow(td); self._task_widgets[tid] = row
                row.clicked.connect(self._on_task_clicked)
                row.action_clicked.connect(self._on_task_action)
                row.delete_clicked.connect(self.step_deleted.emit)
                # 选填行折进二级分组（默认收起、暗色表头），必选流程保持一屏可见。
                if tid in ("exit", "unload"):
                    optional_rows.append(row)
                    continue
                rows.append(row)
                if tid == "process":
                    self._mask_strength_label = QLabel("处理强度")
                    self._mask_strength_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:13px;padding-left:8px;")
                    self.mask_strength_value_label = QLabel("标准（固定）")
                    self.mask_strength_value_label.setToolTip(
                        "掩膜处理统一使用标准参数，保证不同任务的处理口径一致"
                    )
                    self.mask_strength_value_label.setStyleSheet(
                        f"color:{COLORS['text']};font-size:13px;"
                        f"padding:4px 8px;border:1px solid {COLORS['border']};"
                        "border-radius:4px;"
                    )
                    rows.append(self._mask_strength_label)
                    rows.append(self.mask_strength_value_label)
                    self.mask_diagnostic_btn = QPushButton("导出掩膜优化诊断")
                    self.mask_diagnostic_btn.setToolTip("导出掩膜处理各阶段、连通区域指标和异常编号总览")
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
                        f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;}}"
                    )
                    rows.append(self.model_combo)
                    self.inference_diagnostic_btn = QPushButton("导出识别诊断")
                    self.inference_diagnostic_btn.setToolTip(
                        "导出源影像位置、模型、切片网格、置信度和原始推理掩膜"
                    )
                    self.inference_diagnostic_btn.setMinimumHeight(30)
                    self.inference_diagnostic_btn.setEnabled(False)
                    self.inference_diagnostic_btn.clicked.connect(
                        self.export_inference_diagnostics_requested.emit
                    )
                    rows.append(self.inference_diagnostic_btn)
                if tid == "centerline_select":
                    self.centerline_selector = CenterlineSelectionPanel()
                    self.centerline_selector.centerline_selection_changed.connect(
                        self.centerline_selection_changed.emit
                    )
                    rows.append(self.centerline_selector)
                if tid == "entry":
                    self.recommend_service_points_btn = QPushButton("智能推荐起点")
                    self.recommend_service_points_btn.setToolTip(
                        "推荐一个靠近作业线端部的起点；确认或拖动后，路径从离它最近的作业线端点开始"
                    )
                    self.recommend_service_points_btn.setMinimumHeight(30)
                    self.recommend_service_points_btn.clicked.connect(
                        self.recommend_service_points_requested.emit
                    )
                    rows.append(self.recommend_service_points_btn)
                if tid == "params":
                    # 高级设置单独放在“高级设置”折叠卡中，默认不挤占
                    # 基础的中心线选择、起终点和路径规划流程。
                    sep = QFrame()
                    sep.setFrameShape(QFrame.HLine)
                    sep.setFixedHeight(1)
                    sep.setStyleSheet(f"color:{COLORS.get('border', '#383848')};background:{COLORS.get('border', '#383848')};")
                    rows.append(sep)
                    forbidden_row = QWidget()
                    forbidden_layout = QHBoxLayout(forbidden_row)
                    forbidden_layout.setContentsMargins(8, 0, 8, 0)
                    forbidden_layout.setSpacing(6)
                    self.forbidden_status_label = QLabel("禁行区：未确认")
                    self.forbidden_status_label.setStyleSheet(
                        f"color:{COLORS['orange']};font-size:13px;"
                    )
                    self.confirm_no_forbidden_btn = QPushButton("确认无")
                    self.draw_forbidden_btn = QPushButton("绘制")
                    for button in (self.confirm_no_forbidden_btn, self.draw_forbidden_btn):
                        button.setMinimumSize(52, 26)
                        button.setStyleSheet(
                            f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                            f"border:1px solid {COLORS['border']};border-radius:4px;padding:3px 8px;}}"
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
                    turn_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:13px;")
                    self.turn_combo = ExplicitPopupComboBox()
                    self.turn_combo.setToolTip(
                        "点击展开后选择调头/转弯方式；未展开时滚轮只滚动面板，不会改策略"
                    )
                    self.turn_combo.setMinimumHeight(28)
                    self.turn_combo.setStyleSheet(
                        f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;}}"
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
                        "进入人工路线编辑；画布负责点线拖动和加点，下面的路径结果负责结构操作、直线/转弯切换与验证"
                    )
                    self.route_edit_btn.setMinimumHeight(30)
                    self.route_edit_btn.setStyleSheet(
                        f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:5px;padding:4px 8px;}}"
                        f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                        f"QPushButton:disabled{{color:{COLORS.get('text_dim', '#8B909A')};}}"
                    )
                    self.route_edit_btn.clicked.connect(self.route_edit_requested.emit)
                    rows.append(self.route_edit_btn)
                    self.route_edit_action_row = QWidget()
                    route_edit_action_layout = QHBoxLayout(
                        self.route_edit_action_row
                    )
                    route_edit_action_layout.setContentsMargins(8, 0, 8, 0)
                    route_edit_action_layout.setSpacing(6)
                    self.route_add_point_btn = QPushButton("新增点")
                    self.route_add_point_btn.setCheckable(True)
                    self.route_add_point_btn.setToolTip(
                        "在线段上单击可插入控制点；空白处单击会从当前路线末端追加控制点"
                    )
                    self.route_toggle_status_btn = QPushButton("作业/转弯")
                    self.route_toggle_status_btn.setToolTip(
                        "把当前选中的整条线段在作业状态与转弯状态之间切换"
                    )
                    self.route_toggle_geometry_btn = QPushButton("直线/转弯")
                    self.route_toggle_geometry_btn.setToolTip(
                        "把当前选中的点—点连接切换为直线或曲线转弯；不改变作业语义"
                    )
                    for button in (
                        self.route_add_point_btn,
                        self.route_toggle_status_btn,
                        self.route_toggle_geometry_btn,
                    ):
                        button.setMinimumHeight(28)
                        button.setStyleSheet(
                            f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                            f"border:1px solid {COLORS['border']};border-radius:4px;padding:3px 8px;}}"
                            f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                            f"QPushButton:checked{{background:{COLORS['accent']};color:white;}}"
                        )
                    self.route_add_point_btn.clicked.connect(
                        lambda: self.route_edit_tool_requested.emit("add_point")
                    )
                    self.route_toggle_status_btn.clicked.connect(
                        lambda: self.route_edit_tool_requested.emit("toggle_status")
                    )
                    self.route_toggle_geometry_btn.clicked.connect(
                        lambda: self.route_edit_tool_requested.emit("toggle_geometry")
                    )
                    route_edit_action_layout.addWidget(
                        self.route_add_point_btn, 1
                    )
                    route_edit_action_layout.addWidget(
                        self.route_toggle_status_btn, 1
                    )
                    route_edit_action_layout.addWidget(
                        self.route_toggle_geometry_btn, 1
                    )
                    self.route_edit_action_row.setVisible(False)
                    rows.append(self.route_edit_action_row)
            if optional_rows:
                self.optional_group = OptionalRowGroup(
                    "选填：终点与卸粮点", optional_rows
                )
                rows.append(self.optional_group)
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
            "centerline_generate": self.centerline_generate_requested,
            "entry": self.entry_exit_toggle,
            "exit": self.entry_exit_toggle,
            "unload": self.entry_exit_toggle,
            "plan": self.plan_requested,
            "simulate": self.simulate_requested,
            # The task-panel export is the terminal-facing numbered TXT
            # workflow.  GeoJSON remains an explicit record export in the
            # File menu so clicking the main workflow cannot silently produce
            # a different artifact than the operator requested.
            "export": self.export_machine_compatibility_requested,
        }
        if tid == "centerline_select":
            row = self._task_widgets.get(tid)
            if row is not None and getattr(row, "_blocker", ""):
                row.flash_blocker(row._blocker)
                return
            self._expand_card_for_task(tid)
            return
        row = self._task_widgets.get(tid)
        if row is not None and getattr(row, "_blocker", ""):
            # 上一步未完成：不执行动作，只在这一行就地提示原因，几秒后复原。
            # 这样"先完成X"只出现在用户真正想往前走的那个位置，而不是
            # 每一行都复述一遍。
            row.flash_blocker(row._blocker)
            return
        if getattr(row, "_status_key", "") == "pending":
            row.flash_blocker(getattr(row, "_blocker", "") or "请先完成前置步骤")
            return
        sig = mapping.get(tid)
        if sig:
            if tid in {"entry", "exit", "unload"}:
                self.entry_exit_toggle.emit(tid)
            else:
                sig.emit()

    def _on_task_action(self, tid):
        """操作按钮与整行点击分流。

        「选择作业线」行只有点按钮（确认）才算采纳当前勾选；点行本身只展开
        卡片，方便先看一眼、改几条再确认。
        """
        if tid == "centerline_select":
            row = self._task_widgets.get(tid)
            if row is not None and getattr(row, "_blocker", ""):
                row.flash_blocker(row._blocker)
                return
            if row is not None and getattr(row, "_status_key", "") == "available":
                self.centerline_selection_confirm_requested.emit()
                return
            self._expand_card_for_task(tid)
            return
        self._on_task_clicked(tid)

    def _expand_card_for_task(self, tid):
        for index, group in enumerate(STEP_GROUPS):
            if tid in group["tasks"]:
                self._manual_step = index
                self._refresh_cards()
                return

    def _refresh_cards(self):
        first = 0
        for i, group in enumerate(STEP_GROUPS):
            if any(not self._task_widgets[tid]._done for tid in group["tasks"]):
                first = i; break
        # 状态刷新只更新状态与高亮，**不覆盖用户的展开/收起选择**。
        # 旧实现在这里无条件 set_expanded(i == expanded)，于是用户刚收起的
        # 卡片会在下一次状态推送时被强行展开，表现为"点了没反应"。
        step_changed = first != getattr(self, "_last_first_step", -1)
        self._last_first_step = first
        for i, card in enumerate(self._step_cards):
            card.set_active(i == first)
            card.refresh_state()
        self._current_step = first
        # 步骤推进（某一步完成）时自动把焦点交给下一步：旧步骤收起、新步骤
        # 展开。这里必须清掉用户手动选择，否则推进后仍停留在旧卡片上，
        # 用户还得手动关一个再开一个。
        if step_changed:
            self._manual_step = None
            self._user_collapsed = False
            for i, card in enumerate(self._step_cards):
                card.set_expanded(i == first)
            self._ensure_card_visible(first)

    def _on_card_toggled(self, index):
        """点击表头：展开这张卡，或把它收起来。

        旧实现只在"已展开"分支里收起**其他**卡片，唯独不收自己，所以展开
        之后再点同一张卡没有任何反应。收起后还要记住用户意图，否则下一次
        状态刷新会把当前步骤卡片又自动展开。
        """
        card = self._step_cards[index]
        if card._expanded:
            card.set_expanded(False)
            self._manual_step = None
            self._user_collapsed = True
            return
        self._user_collapsed = False
        self._manual_step = index
        for i, other in enumerate(self._step_cards):
            if i != index:
                other.set_expanded(False)
        card.set_expanded(True)
        self._ensure_card_visible(index)

    def _ensure_card_visible(self, index):
        """Scroll the selected accordion card into view after expansion."""
        scroll = getattr(self, "_scroll", None)
        if not scroll or not (0 <= index < len(self._step_cards)):
            return
        scroll.ensureWidgetVisible(self._step_cards[index], 0, 8)

    def set_task_status(self, tid, status, progress_pct=-1, text=None,
                        optional=None, blocker=None):
        row = self._task_widgets.get(tid)
        if not row: return
        if optional is not None:
            row._optional = bool(optional)
        row._blocker = str(blocker or "")
        row.set_status(
            done=status == "done",
            enabled=status in ("available", "in_progress", "done", "failed"),
            clickable=True,
            in_progress=status == "in_progress",
            progress_pct=progress_pct,
            text=text,
            status=status,
        )
        if not getattr(self, "_applying_task_statuses", False):
            self._sync_optional_summary()
            self._refresh_cards()

    def apply_task_statuses(self, statuses):
        self._applying_task_statuses = True
        try:
            for tid in TASK_ORDER:
                st = statuses.get(tid)
                if not st:
                    continue
                text = st.text
                if tid == "plan" and st.status == "pending" and st.blocker:
                    # 路径规划只有一个按钮，把「为什么还不能规划」直接写在这一行，
                    # 不用等用户点一下才闪出来（选好起点却规划不了就是这么来的）。
                    text = st.blocker
                self.set_task_status(
                    tid, st.status, text=text,
                    optional=bool(getattr(st, "optional", False)),
                    blocker=getattr(st, "blocker", None),
                )
                if st.action:
                    self.set_task_action(tid, st.action)
                else:
                    task_def = _task_def(tid)
                    if task_def:
                        self.set_task_action(tid, task_def["action"])
        finally:
            self._applying_task_statuses = False
        self._sync_optional_summary()
        self._refresh_cards()

    def _sync_optional_summary(self):
        """选填分组表头显示当前取值，收起时也能一眼看到有没有设。"""
        group = getattr(self, "optional_group", None)
        if group is None:
            return
        done_parts = []
        exit_row = self._task_widgets.get("exit")
        if exit_row is not None and str(getattr(exit_row, "_status_text", "")).startswith("已设置"):
            done_parts.append("终点已设")
        unload_row = self._task_widgets.get("unload")
        if unload_row is not None and str(getattr(unload_row, "_status_text", "")).startswith("已设置"):
            done_parts.append("卸粮点已设")
        group.set_summary("、".join(done_parts) if done_parts else "均未设置")

    def set_task_action(self, tid, text):
        row = self._task_widgets.get(tid)
        if row and row._action_btn.text() != str(text):
            row._action_btn.setText(str(text))

    def update_centerline_selection(self, catalog, selected_ids=None):
        if hasattr(self, "centerline_selector"):
            self.centerline_selector.update_centerline_selection(catalog, selected_ids)

    def sync_centerline_selection(self, selected_ids=None):
        """只同步勾选状态，不重建行（勾选/取消的高频路径）。"""
        if hasattr(self, "centerline_selector"):
            self.centerline_selector.sync_selection(selected_ids)

    def clear_centerline_selection(self):
        if hasattr(self, "centerline_selector"):
            self.centerline_selector.clear_centerline_selection()

    def set_centerline_checked(self, centerline_id, checked):
        if hasattr(self, "centerline_selector"):
            self.centerline_selector.set_centerline_checked(centerline_id, checked)


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
            f"color:{color};font-size:13px;"
        )

    def set_mask_diagnostic_available(self, available):
        if hasattr(self, "mask_diagnostic_btn"):
            self.mask_diagnostic_btn.setEnabled(bool(available))

    def set_inference_diagnostic_available(self, available):
        if hasattr(self, "inference_diagnostic_btn"):
            self.inference_diagnostic_btn.setEnabled(bool(available))

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

    def set_model_loading(self, loading):
        """Lock model selection and AI start while a weight is deserialised."""
        loading = bool(loading)
        if hasattr(self, "model_combo"):
            self.model_combo.setEnabled(not loading)
            self.model_combo.setToolTip(
                "模型正在后台加载，请等待完成..."
                if loading
                else "选择 AI 识别模型，或从文件夹导入其他 .pt 权重"
            )
        row = self._task_widgets.get("segment")
        if row is not None and loading:
            # ``_sync_task_statuses`` may refresh the row while the worker is
            # active; the main window reapplies this lock after that refresh.
            row._action_btn.setEnabled(False)
        elif row is not None:
            row._action_btn.setEnabled(row.isEnabled())

    def populate_models(self, models, current=""):
        normalized = [{"name": str(item), "path": str(item)} for item in (models or [])]
        self.set_model_options(normalized, current)

    def set_model_path(self, name):
        if hasattr(self, 'model_combo'):
            idx = self.model_combo.findText(name)
            if idx >= 0: self.model_combo.setCurrentIndex(idx)

    def refresh_theme(self):
        self.setStyleSheet(f"background:{COLORS['bg']};")
        self._title.setStyleSheet(f"font-size:17px;font-weight:600;color:{COLORS['text_bright']};")
        combo_style = (
            f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
            f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;}}"
        )
        if hasattr(self, "_mask_strength_label"):
            self._mask_strength_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:13px;padding-left:8px;")
        for combo_name in ("model_combo", "turn_combo"):
            combo = getattr(self, combo_name, None)
            if combo:
                combo.setStyleSheet(combo_style)
        for button_name in (
            "route_edit_btn", "mask_diagnostic_btn", "inference_diagnostic_btn",
            "recommend_service_points_btn",
        ):
            button = getattr(self, button_name, None)
            if button:
                button.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                f"border:1px solid {COLORS['border']};border-radius:5px;padding:4px 8px;}}"
                f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                f"QPushButton:disabled{{color:{COLORS.get('text_dim', '#8B909A')};}}"
                )
        for button_name in ("confirm_no_forbidden_btn", "draw_forbidden_btn"):
            button = getattr(self, button_name, None)
            if button:
                button.setStyleSheet(
                    f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                    f"border:1px solid {COLORS['border']};border-radius:4px;padding:3px 8px;}}"
                    f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                )
        self.set_forbidden_state(self._forbidden_count, self._forbidden_confirmed)
        for row in self.findChildren(TaskRow):
            row.refresh_theme()
        if hasattr(self, "centerline_selector"):
            self.centerline_selector.refresh_theme()
        group = getattr(self, "optional_group", None)
        if group is not None:
            group.refresh_theme()
        for card in self._step_cards:
            card.refresh_state(force=True)

    def set_route_editing_active(self, active):
        active = bool(active)
        if hasattr(self, "route_edit_btn"):
            self.route_edit_btn.setText("完成编辑" if active else "编辑路线")
        if hasattr(self, "route_edit_action_row"):
            self.route_edit_action_row.setVisible(active)
        if not active and hasattr(self, "route_add_point_btn"):
            self.route_add_point_btn.setChecked(False)
        for card in self._step_cards:
            if hasattr(self, "route_edit_btn") and self.route_edit_btn in getattr(
                card, "_rows", []
            ):
                card.refresh_content_geometry()
                break

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
    _WORK_MIME = "application/x-align-work-line-index"

    def __init__(self, index, label, distance_m, is_work=True, parent=None,
                 segment_type=None, point_count=None, edit_meta=None,
                 display_number=None):
        super().__init__(parent); self._index = index; self._is_work = is_work
        self._segment_type = str(segment_type or ("work" if is_work else "turn"))
        self._edit_meta = dict(edit_meta or {})
        self._edit_selected = False
        self._hover = 0.0
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
        self._geometry = None
        if self._edit_meta:
            mode = str(self._edit_meta.get("geometry_mode") or "straight")
            self._geometry = QLabel("转弯" if mode == "turn" else "直线")
        self._dist = QLabel(f"{distance_m:.1f} m")
        lo.addWidget(self._badge)
        lo.addWidget(self._label, 1)
        if self._point_count is not None:
            lo.addWidget(self._point_count)
        if self._geometry is not None:
            lo.addWidget(self._geometry)
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
        self._badge.setStyleSheet(f"background:{color};color:white;border-radius:2px;font-size:11px;")
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:13px;")
        if self._point_count is not None:
            self._point_count.setStyleSheet(
                f"color:{COLORS['text_dim']};font-size:11px;"
            )
        if self._geometry is not None:
            self._geometry.setStyleSheet(
                f"color:{COLORS['accent']};font-size:11px;"
            )
        self._dist.setFont(ui_font(9))
        self._dist.setStyleSheet(f"color:{COLORS['text_dim']};")
        # Surface is painted (paintEvent) so hover can fade in like the
        # reference rows; QSS only carries the text colours.
        self.setStyleSheet(
            f"RouteSegmentRow{{background:transparent;border:none;}}"
        )
        self.update()

    def paintEvent(self, event):
        selected = bool(getattr(self, "_edit_selected", False))
        hover = max(0.0, min(1.0, getattr(self, "_hover", 0.0)))
        if selected:
            fill = QColor(COLORS.get("bg_accent", "#10291F"))
        else:
            base = QColor(COLORS.get("bg", "#0B0F17"))
            lift = QColor(COLORS.get("bg_hover", "#1B2433"))
            fill = QColor(
                int(base.red() + (lift.red() - base.red()) * hover),
                int(base.green() + (lift.green() - base.green()) * hover),
                int(base.blue() + (lift.blue() - base.blue()) * hover),
            )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath(); path.addRoundedRect(rect, 4.0, 4.0)
        painter.fillPath(path, fill)
        edge = QColor(COLORS.get("accent", "#34D399"))
        edge.setAlpha(200 if selected else int(70 * hover))
        if selected or hover > 0.02:
            painter.setPen(QPen(edge, 1.0)); painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, 4.0, 4.0)
        painter.end()

    def enterEvent(self, event):
        self._set_hover(1.0); super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_hover(0.0); super().leaveEvent(event)

    def _set_hover(self, target):
        if not hasattr(self, "_hover"):
            self._hover = 0.0
        tween_value(self, "_hover", target, 160, None, lambda: self.update())

    def set_edit_selected(self, selected):
        self._edit_selected = bool(selected)
        self.refresh_theme()

    def available_edit_actions(self):
        if not self._edit_meta:
            return []
        actions = [
            "toggle_segment",
            "toggle_geometry",
            "add_waypoint",
            "validate_route",
        ]
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
        geometry_mode = str(self._edit_meta.get("geometry_mode") or "straight")
        geometry_toggle = menu.addAction(
            "切换为直线"
            if geometry_mode == "turn"
            else "切换为转弯"
        )
        geometry_toggle.triggered.connect(
            lambda: self.trigger_edit_action("toggle_geometry")
        )
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


class CenterlineSelectionRow(QWidget):
    """One detected centerline with an explicit planning checkbox."""

    toggled = Signal(str, bool)

    #: 行内只显示来源的中文短名：原始标识符（four_row_band_centerlines 等）
    #: 挤在每行右侧，读起来像日志，和"这是哪条线"无关。
    _SOURCE_DISPLAY = {
        "four_row_band_centerlines": "宽窄行",
        "cutter_width_uniform_swaths": "等距区",
        "nearest_region_merge_for_non_adjacent_contour": "合并区",
    }

    def __init__(self, item, checked=True, parent=None):
        super().__init__(parent)
        item = dict(item or {})
        self.centerline_id = str(item.get("id") or "")
        sequence = int(item.get("sequence", 0) or 0)
        label = str(item.get("label") or f"中心线 {sequence}")
        raw_source = str(item.get("line_source") or item.get("kind") or "")
        source = self._SOURCE_DISPLAY.get(raw_source, "")
        self.setMinimumHeight(28)
        self._hover = 0.0
        self.setMouseTracking(True)
        lo = QHBoxLayout(self)
        lo.setContentsMargins(8, 2, 8, 2)
        lo.setSpacing(6)
        self.checkbox = QCheckBox()
        self.checkbox.setFixedWidth(22)
        self.checkbox.setToolTip("勾选后参与路径；取消勾选后隐藏并不录用")
        self.checkbox.setChecked(bool(checked))
        self.checkbox.toggled.connect(
            lambda value: self.toggled.emit(self.centerline_id, bool(value))
        )
        self._label = QLabel(label)
        self._label.setMinimumWidth(0)
        self._label.setStyleSheet(
            f"font-size:13px;color:{COLORS['text']};"
        )
        self._source = QLabel(source)
        self._source.setMinimumWidth(0)
        self._source.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._source.setToolTip(raw_source)
        self._source.setVisible(bool(source))
        self._source.setStyleSheet(
            f"font-size:11px;color:{COLORS['text_dim']};"
        )
        lo.addWidget(self.checkbox)
        lo.addWidget(self._label, 1)
        lo.addWidget(self._source)

    def paintEvent(self, event):
        hover = max(0.0, min(1.0, getattr(self, "_hover", 0.0)))
        if hover <= 0.01:
            return
        base = QColor(COLORS.get("bg", "#0B0F17"))
        lift = QColor(COLORS.get("bg_hover", "#1B2433"))
        fill = QColor(
            int(base.red() + (lift.red() - base.red()) * hover),
            int(base.green() + (lift.green() - base.green()) * hover),
            int(base.blue() + (lift.blue() - base.blue()) * hover),
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 4.0, 4.0)
        painter.fillPath(path, fill)
        painter.end()

    def enterEvent(self, event):
        tween_value(self, "_hover", 1.0, 160, None, lambda: self.update())
        super().enterEvent(event)

    def leaveEvent(self, event):
        tween_value(self, "_hover", 0.0, 160, None, lambda: self.update())
        super().leaveEvent(event)

    def set_checked(self, checked):
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(bool(checked))
        self.checkbox.blockSignals(False)

    def refresh_theme(self):
        self._label.setStyleSheet(
            f"font-size:13px;color:{COLORS['text']};"
        )
        self._source.setStyleSheet(
            f"font-size:11px;color:{COLORS['text_dim']};"
        )


class CenterlineSelectionPanel(QWidget):
    """Independent workflow control for selecting detected centerlines."""

    centerline_selection_changed = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._centerline_rows = {}
        self.setObjectName("centerlineSelectionPanel")
        lo = QVBoxLayout(self)
        lo.setContentsMargins(8, 0, 8, 2)
        lo.setSpacing(3)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._count_label = QLabel("已选择 0/0 条")
        self._count_label.setStyleSheet(
            f"font-size:13px;color:{COLORS['text_accent']};"
        )
        header.addWidget(self._count_label)
        header.addStretch()
        lo.addLayout(header)

        self._hint = QLabel(
            "一条种植带对应一条作业线（复选框）；取消勾选只移除该带，底图和其他路线保持不动。"
        )
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            f"font-size:11px;color:{COLORS['text_accent']};"
        )
        lo.addWidget(self._hint)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setMaximumHeight(160)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(1)
        self._scroll.setWidget(self._container)
        lo.addWidget(self._scroll)
        self._empty_label = QLabel("尚未生成中心线，请先点击上一步“生成中心线”。")
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(
            f"font-size:12px;color:{COLORS['text_dim']};"
        )
        lo.addWidget(self._empty_label)
        self.clear_centerline_selection()

    def _clear_rows(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._centerline_rows.clear()

    def clear_centerline_selection(self):
        self._clear_rows()
        self._count_label.setText("已选择 0/0 条")
        self._scroll.setVisible(False)
        self._empty_label.setVisible(True)

    def update_centerline_selection(self, catalog, selected_ids=None):
        items = list((catalog or {}).get("items") or [])
        self._clear_rows()
        if not items:
            self.clear_centerline_selection()
            return
        selected = {str(value) for value in (selected_ids or [])}
        selected_count = 0
        for item in items:
            row = CenterlineSelectionRow(
                item,
                checked=str(item.get("id")) in selected,
            )
            selected_count += int(row.checkbox.isChecked())
            row.toggled.connect(self._on_row_toggled)
            self._centerline_rows[str(item.get("id"))] = row
            self._layout.addWidget(row)
        self._layout.addStretch()
        self._count_label.setText(f"已选择 {selected_count}/{len(items)} 条")
        self._scroll.setMinimumHeight(min(160, 30 * len(items) + 4))
        self._scroll.setVisible(True)
        self._empty_label.setVisible(False)

    def _on_row_toggled(self, centerline_id, checked):
        self._update_count()
        self.centerline_selection_changed.emit(centerline_id, bool(checked))

    def _update_count(self):
        total = len(self._centerline_rows)
        selected_count = sum(
            int(row.checkbox.isChecked()) for row in self._centerline_rows.values()
        )
        self._count_label.setText(f"已选择 {selected_count}/{total} 条")

    def sync_selection(self, selected_ids=None):
        """只同步勾选状态与计数，不销毁/重建行控件。

        勾选一条作业线时调用 update_centerline_selection 会把 14 行全部
        deleteLater 再重建，滚动位置也会回到顶部——点一下卡一下就是这么来的。
        """
        if not self._centerline_rows:
            return
        selected = {str(value) for value in (selected_ids or [])}
        for centerline_id, row in self._centerline_rows.items():
            row.set_checked(centerline_id in selected)
        self._update_count()

    def set_centerline_checked(self, centerline_id, checked):
        row = self._centerline_rows.get(str(centerline_id))
        if row is not None:
            row.set_checked(checked)

    def refresh_theme(self):
        self._count_label.setStyleSheet(
            f"font-size:13px;color:{COLORS['text_accent']};"
        )
        self._hint.setStyleSheet(
            f"font-size:11px;color:{COLORS['text_accent']};"
        )
        self._empty_label.setStyleSheet(
            f"font-size:12px;color:{COLORS['text_dim']};"
        )
        self._container.setStyleSheet(f"background:{COLORS['bg']};")
        self._scroll.setStyleSheet(
            f"QScrollArea{{background:{COLORS['bg']};border:none;}}"
        )
        for row in self._centerline_rows.values():
            row.refresh_theme()


class RouteInfoPanel(QWidget):
    segment_selected = Signal(int)
    service_point_selected = Signal(str, int)
    edit_segment_selected = Signal(int, int)
    edit_action_requested = Signal(str, int, int)
    work_reorder_requested = Signal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._segments = []
        self._build_ui()
    def _build_ui(self):
        lo = QVBoxLayout(self); lo.setContentsMargins(0,0,0,0); lo.setSpacing(0)
        self._title_hdr = QLabel("路径结果"); self._title_hdr.setStyleSheet(f"font-size:14px;font-weight:600;color:{COLORS['text']};padding:8px 14px 2px;")
        self._summary = QLabel("作业段 0 / 转弯段 0")
        self._summary.setWordWrap(True)
        self._summary.setMinimumWidth(0)
        self._summary.setStyleSheet(f"font-size:12px;color:{COLORS['text_dim']};padding:2px 14px 4px;")
        self._safety = QLabel("路径状态：未评估")
        self._safety.setWordWrap(True)
        self._safety.setStyleSheet(f"font-size:12px;color:{COLORS['orange']};padding:2px 14px 6px;")
        self._edit_hint = QLabel("单击联动画布｜右击编辑或验证｜拖动作业线排序")
        self._edit_hint.setWordWrap(True)
        self._edit_hint.setStyleSheet(
            f"font-size:11px;color:{COLORS['text_accent']};padding:0 14px 6px;"
        )
        self._edit_hint.setVisible(False)
        lo.addWidget(self._title_hdr); lo.addWidget(self._summary); lo.addWidget(self._safety)
        lo.addWidget(self._edit_hint)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        self._scroll = sc
        self._list_container = QWidget(); self._list_layout = QVBoxLayout(self._list_container); self._list_layout.setContentsMargins(8,0,8,0); self._list_layout.setSpacing(2)
        self._empty_label = QLabel("尚无路径结果。"); self._empty_label.setStyleSheet(f"font-size:13px;color:{COLORS['text_dimmer']};padding:8px;")
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
        validation_passed = validation.get("valid") is True
        execution_label = (
            "路径校验通过" if validation_passed else "路径校验未通过"
        )
        self._safety.setText(f"路径状态：{execution_label}")
        if validation_passed:
            details = ["已完成当前路径的几何与足迹检查"]
        elif issues:
            details = ["请调整路径后重新校验"]
            details.extend(str(issue) for issue in issues[:3])
        else:
            details = ["尚未获得有效的路径校验结果"]
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
            f"作业段 {work_count} / 转弯段 {turn_count} / 总长 {total_length:.1f} m"
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
        self._clear_list(); self._empty_label.setVisible(True); self._summary.setText("作业段 0 / 转弯段 0")
        self._safety.setText("路径状态：未评估"); self._safety.setToolTip("")
    def refresh_theme(self):
        self.setStyleSheet(f"background:{COLORS['bg']};")
        self._scroll.setStyleSheet(
            f"QScrollArea{{background:{COLORS['bg']};border:none;}}"
        )
        self._scroll.viewport().setStyleSheet(f"background:{COLORS['bg']};")
        self._list_container.setStyleSheet(f"background:{COLORS['bg']};")
        self._title_hdr.setStyleSheet(f"font-size:14px;font-weight:600;color:{COLORS['text']};padding:8px 14px 2px;")
        self._summary.setStyleSheet(f"font-size:12px;color:{COLORS['text_dim']};padding:2px 14px 4px;")
        self._safety.setStyleSheet(f"font-size:12px;color:{COLORS['orange']};padding:2px 14px 6px;")
        self._edit_hint.setStyleSheet(
            f"font-size:11px;color:{COLORS['text_accent']};padding:0 14px 6px;"
        )
        self._empty_label.setStyleSheet(f"font-size:13px;color:{COLORS['text_dimmer']};padding:8px;")
        for row in self.findChildren(RouteSegmentRow):
            row.refresh_theme()
    @staticmethod
    def _calc_distance(points):
        pts = [(float(p[0]), float(p[1])) for p in points or [] if len(p) >= 2]
        return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) for i in range(1,len(pts)))


class LogPanel(QWidget):
    """运行记录：分类、去重、可折叠明细（DECISION-049）。

    改之前是一个裸 QTextEdit，184 条调用各写各的，同一件事有 2–4 种说法，
    哈希/EPSG 这类明细和"该做什么"的提示混在一起。现在按级别着色 + 同文案
    合并 ×N + 明细默认收起。
    """

    MAX_ENTRIES = 400
    MERGE_WINDOW_S = 6.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setObjectName("logPanel")
        self._entries = []          # [(ts_float, level, text, key, repeat)]
        self._show_details = False
        self._view = QTextEdit()
        self._view.setReadOnly(True)
        self._view.setObjectName("logView")
        self._hint = QLabel("运行记录")
        self._chk_detail = QCheckBox("详情")
        self._chk_detail.setToolTip("显示参数、路径、缓存指纹等技术明细")
        self._chk_detail.toggled.connect(self.set_show_details)
        self._btn_clear = QPushButton("清空")
        self._btn_clear.setFixedHeight(22)
        self._btn_clear.clicked.connect(self._confirm_clear)
        head = QHBoxLayout()
        head.setContentsMargins(2, 0, 2, 0)
        head.setSpacing(6)
        head.addWidget(self._hint)
        head.addSpacing(8)
        head.addWidget(self._chk_detail)
        head.addSpacing(8)
        head.addWidget(self._btn_clear)
        head.addStretch(1)
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(4)
        lo.addLayout(head)
        lo.addWidget(self._view, 1)
        self.refresh_theme()

    # ------------------------------------------------------------------ 主题
    def refresh_theme(self):
        self._hint.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;")
        self._chk_detail.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;")
        self._btn_clear.setStyleSheet(
            f"QPushButton{{background:transparent;color:{COLORS['text_dim']};"
            f"border:1px solid {COLORS['border']};border-radius:5px;padding:0 8px;font-size:12px;}}"
            f"QPushButton:hover{{background:{COLORS['bg_hover']};color:{COLORS['text']};}}"
        )
        self.setStyleSheet(
            f"QWidget#logPanel{{background:{COLORS['bg_darkest']};color:{COLORS['text']};}}"
        )
        self._view.setStyleSheet(
            f"QTextEdit#logView{{background:{COLORS['bg_darkest']};color:{COLORS['text']};"
            f"border:1px solid {COLORS['border']};border-radius:6px;"
            "font-family:'Cascadia Code','Consolas',monospace;font-size:13px;padding:6px;}"
        )

    # ------------------------------------------------------------------ 状态
    def set_show_details(self, show: bool):
        self._show_details = bool(show)
        self._render()

    def _confirm_clear(self):
        from PySide6.QtWidgets import QMessageBox

        if not self._entries:
            return
        answer = QMessageBox.question(
            self, "清空运行记录", "清空后无法恢复，确定吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.clear()

    def clear(self):
        self._entries.clear()
        self._view.clear()

    def entry_count(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ 记录
    def _append(self, level, msg):
        from ui_messages import detail_key, normalize

        text = normalize(msg)
        if not text:
            return
        if level == "detail" and not self._show_details:
            # 明细照常入队（打开开关即可回看），只是不渲染
            pass
        now = time.time()
        key = detail_key(text)
        last = self._entries[-1] if self._entries else None
        if last and last[3] == key and now - last[0] <= self.MERGE_WINDOW_S:
            self._entries[-1] = (last[0], last[1], text, key, last[4] + 1)
        else:
            self._entries.append((now, level, text, key, 1))
            if len(self._entries) > self.MAX_ENTRIES:
                self._entries = self._entries[-self.MAX_ENTRIES:]
        self._render()

    def info(self, m): self._append("info", m)
    def success(self, m): self._append("ok", m)
    def warn(self, m): self._append("warn", m)
    def error(self, m): self._append("error", m)
    def detail(self, m): self._append("detail", m)

    # ------------------------------------------------------------------ 渲染
    def _colors(self):
        return {
            "info": COLORS["text"],
            "ok": COLORS.get("green", "#34D399"),
            "warn": COLORS.get("orange", "#F5A524"),
            "error": COLORS.get("red", "#E85D63"),
            "detail": COLORS["text_dim"],
        }

    def _render(self):
        from ui_messages import GLYPHS

        colors = self._colors()
        rows = []
        for ts, level, text, _key, repeat in self._entries:
            if level == "detail" and not self._show_details:
                continue
            stamp = time.strftime("%H:%M:%S", time.localtime(ts))
            glyph = GLYPHS.get(level, "·")
            color = colors.get(level, COLORS["text"])
            suffix = f"  ×{repeat}" if repeat > 1 else ""
            rows.append(
                f'<div style="margin:1px 0;">'
                f'<span style="color:{COLORS["text_dim"]};">{stamp}</span> '
                f'<span style="color:{color};">{glyph}</span> '
                f'<span style="color:{color if level != "info" else COLORS["text"]};">'
                f'{self._escape(text)}</span>'
                f'<span style="color:{COLORS["text_dim"]};">{suffix}</span></div>'
            )
        self._view.setHtml("".join(rows))
        bar = self._view.verticalScrollBar()
        bar.setValue(bar.maximum())

    @staticmethod
    def _escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


class StatusStrip(QWidget):
    """底部状态条：步骤 / 阶段 / 提示 / 指标 四段各就各位（DECISION-049）。

    改之前只有一句自由文本，步骤、进度、提示、"该做什么"全挤在同一个 QLabel 里，
    读起来散。现在固定四段，缺省段自动隐藏，保持一行高度。
    """

    TONES = {
        "busy": "accent",
        "ready": "green",
        "error": "red",
        "warn": "orange",
        "idle": "text_dim",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tone = "idle"
        self.dot = BreathingDot(7)
        self.lbl_step = QLabel("")
        self.lbl_phase = QLabel("")
        self.lbl_hint = QLabel("")
        self.lbl_metrics = QLabel("")
        self.lbl_metrics_divider = QLabel("│")
        self._sep_step = QLabel("│")
        self._sep_hint = QLabel("│")
        self._separators = (self._sep_step, self._sep_hint, self.lbl_metrics_divider)
        for lbl in (self.lbl_step, self.lbl_hint, self.lbl_metrics, *self._separators):
            lbl.setVisible(False)
        lo = QHBoxLayout(self)
        lo.setContentsMargins(2, 0, 4, 0)
        lo.setSpacing(8)
        lo.addWidget(self.dot)
        lo.addWidget(self.lbl_step)
        lo.addWidget(self._sep_step)
        lo.addWidget(self.lbl_phase)
        lo.addWidget(self._sep_hint)
        lo.addWidget(self.lbl_hint)
        lo.addStretch(1)
        lo.addWidget(self.lbl_metrics_divider)
        lo.addWidget(self.lbl_metrics)
        self.refresh_theme()

    def refresh_theme(self):
        self.lbl_step.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;")
        self.lbl_phase.setStyleSheet(f"color:{COLORS['text']};")
        self.lbl_hint.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;")
        self.lbl_metrics.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;")
        for sep in self._separators:
            sep.setStyleSheet(f"color:{COLORS['text_dim']};opacity:0.45;")
        self.set_tone(self._tone)

    # ------------------------------------------------------------------ 写入
    def setText(self, text):  # noqa: N802 - 兼容旧调用 lbl_status.setText(...)
        self.set_phase(text)

    def set_phase(self, text, tone=None):
        from ui_messages import normalize

        self.lbl_phase.setText(normalize(text))
        if tone:
            self.set_tone(tone)

    def set_step(self, text):
        from ui_messages import normalize

        value = normalize(text)
        self.lbl_step.setText(value)
        self.lbl_step.setVisible(bool(value))
        self._sep_step.setVisible(bool(value))

    def set_hint(self, text, tone="idle"):
        from ui_messages import normalize

        value = normalize(text)
        self.lbl_hint.setText(value)
        self.lbl_hint.setVisible(bool(value))
        self._sep_hint.setVisible(bool(value))
        if value:
            color = COLORS.get(
                {"warn": "orange", "error": "red", "ready": "green"}.get(tone, "text_dim"),
                COLORS["text_dim"],
            )
            self.lbl_hint.setStyleSheet(f"color:{color};font-size:12px;")

    def set_metrics(self, items):
        """items: dict 或 [(label, value)]；空值自动省略。"""
        from ui_messages import normalize

        pairs = items.items() if isinstance(items, dict) else (items or [])
        parts = [f"{normalize(k)} {normalize(v)}" for k, v in pairs if str(v).strip()]
        text = " · ".join(parts)
        self.lbl_metrics.setText(text)
        self.lbl_metrics.setVisible(bool(text))
        self.lbl_metrics_divider.setVisible(bool(text) and self.lbl_hint.isVisible())

    def set_tone(self, tone):
        self._tone = tone or "idle"
        color = COLORS.get(self.TONES.get(self._tone, "text_dim"), COLORS["text_dim"])
        self.dot.set_tone(self._tone, color)

    def text(self):  # 兼容旧代码 lbl_status.text()
        return self.lbl_phase.text()

    def phase_text(self):
        return self.lbl_phase.text()


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
