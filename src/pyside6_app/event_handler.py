"""
event_handler.py — PySide6 事件协调器

在 PySide6 架构中作为统一的鼠标/键盘事件协调层。
MainWindow 在处理完田块绘制、起终点与卸粮点布设等交互后，将事件转发到此处，
用于：坐标跟踪、状态查询、以及可选的 cv2 EventHandler 兼容转发。
"""
import time
from typing import Optional, Tuple, Set

from PySide6.QtCore import QObject, Signal, Qt


class EventBridge(QObject):
    """Qt 事件协调器 — 跟踪鼠标/键盘状态，可选转发到 cv2 EventHandler。

    职责：
    1. 始终维护鼠标坐标、按钮状态（供外部查询）
    2. 双击检测（用于田块多边形闭合等）
    3. 如果设置了 cv2 EventHandler（_handler），将 Qt 事件转换为 cv2 格式转发
    """

    # PySide6 信号
    status_changed = Signal(str)
    need_redraw = Signal()
    workflow_changed = Signal(str)
    double_clicked = Signal(int, int)  # x, y — 双击事件

    # 双击检测参数
    _DBLCLICK_DIST = 15   # 像素
    _DBLCLICK_TIME = 0.4  # 秒

    def __init__(self, state, geo):
        super().__init__()
        self.state = state
        self.geo = geo

        # 可选的 cv2 EventHandler（延迟设置）
        self._handler = None
        self._renderer = None
        self._runner = None
        self._model_engine = None

        # ── 鼠标状态（始终维护） ──
        self._last_xy: Tuple[int, int] = (0, 0)
        self._buttons_down: Set[int] = set()

        # ── 双击检测 ──
        self._last_click_xy: Tuple[int, int] = (0, 0)
        self._last_click_time: float = 0.0
        self._last_click_button: int = 0

    # ── 设置方法 ──

    def set_handler(self, handler):
        """设置 ui.EventHandler 实例（可选，用于 cv2 兼容转发）。"""
        self._handler = handler

    def set_renderer(self, renderer):
        self._renderer = renderer

    # ── 状态查询 ──

    def get_last_xy(self) -> Tuple[int, int]:
        return self._last_xy

    def is_button_down(self, button: int) -> bool:
        """查询某按钮是否按下 (1=左, 2=右, 3=中)。"""
        return button in self._buttons_down

    # ── Qt 鼠标事件处理 ──

    def on_mouse_pressed(self, x: int, y: int, button: int):
        """鼠标按下 — 更新状态 + 双击检测 + 可选 cv2 转发。"""
        # 始终更新状态
        self._buttons_down.add(button)
        self._last_xy = (x, y)

        # 双击检测
        now = time.time()
        if (button == self._last_click_button
                and button == 1  # 只检测左键双击
                and now - self._last_click_time < self._DBLCLICK_TIME
                and abs(x - self._last_click_xy[0]) < self._DBLCLICK_DIST
                and abs(y - self._last_click_xy[1]) < self._DBLCLICK_DIST):
            self.double_clicked.emit(x, y)
            self._last_click_time = 0.0  # 重置，避免连续触发
        else:
            self._last_click_xy = (x, y)
            self._last_click_time = now
            self._last_click_button = button

        # 可选 cv2 转发
        if self._handler is not None:
            self._forward_mouse_press_cv2(x, y, button)

    def on_mouse_released(self, x: int, y: int, button: int):
        """鼠标释放。"""
        self._buttons_down.discard(button)
        self._last_xy = (x, y)

        if self._handler is not None:
            self._forward_mouse_release_cv2(x, y, button)

    def on_mouse_moved(self, x: int, y: int, buttons: int):
        """鼠标移动。"""
        self._last_xy = (x, y)

        if self._handler is not None:
            from cv2 import EVENT_MOUSEMOVE
            flags = 0
            if buttons & 1: flags |= 1   # EVENT_FLAG_LBUTTON
            if buttons & 2: flags |= 2   # EVENT_FLAG_RBUTTON
            if buttons & 4: flags |= 4   # EVENT_FLAG_MBUTTON
            self._handler.mouse_callback(EVENT_MOUSEMOVE, x, y, flags, None)

    def on_mouse_dragged(self, x: int, y: int, buttons: int):
        """鼠标拖拽（移动 + 按下）。"""
        self._last_xy = (x, y)
        if self._handler is not None:
            self.on_mouse_moved(x, y, buttons)

    # ── cv2 转发辅助 ──

    def _qt_buttons_to_flags(self, qt_buttons) -> int:
        """Qt 按钮掩码 → cv2 flags。"""
        from cv2 import (EVENT_FLAG_LBUTTON, EVENT_FLAG_RBUTTON,
                         EVENT_FLAG_MBUTTON)
        flags = 0
        if qt_buttons & Qt.LeftButton:
            flags |= EVENT_FLAG_LBUTTON
        if qt_buttons & Qt.RightButton:
            flags |= EVENT_FLAG_RBUTTON
        if qt_buttons & Qt.MiddleButton:
            flags |= EVENT_FLAG_MBUTTON
        return flags

    def _forward_mouse_press_cv2(self, x: int, y: int, button: int):
        """将按下事件转为 cv2 格式转发给 handler。"""
        from cv2 import EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN, EVENT_MBUTTONDOWN
        evt = {1: EVENT_LBUTTONDOWN, 2: EVENT_RBUTTONDOWN,
               3: EVENT_MBUTTONDOWN}.get(button, EVENT_LBUTTONDOWN)
        flags = self._qt_buttons_to_flags(
            {1: Qt.LeftButton, 2: Qt.RightButton, 3: Qt.MiddleButton}.get(
                button, Qt.NoButton))
        self._handler.mouse_callback(evt, x, y, flags, None)

    def _forward_mouse_release_cv2(self, x: int, y: int, button: int):
        """将释放事件转为 cv2 格式转发给 handler。"""
        from cv2 import EVENT_LBUTTONUP, EVENT_RBUTTONUP, EVENT_MBUTTONUP
        evt = {1: EVENT_LBUTTONUP, 2: EVENT_RBUTTONUP,
               3: EVENT_MBUTTONUP}.get(button, EVENT_LBUTTONUP)
        self._handler.mouse_callback(evt, x, y, 0, None)

    # ── Qt 键盘事件 → cv2 key ──

    def on_key_pressed(self, key: int, text: str) -> bool:
        """Qt 键盘事件 → 调用 keyboard_callback（如已设置）。"""
        if self._handler is None:
            return True
        cv2_key = self._qt_key_to_cv2(key)
        if cv2_key is not None:
            return self._handler.keyboard_callback(cv2_key)
        return True

    @staticmethod
    def _qt_key_to_cv2(qt_key: int) -> Optional[int]:
        """Qt 键码 → cv2.waitKey() 兼容值。"""
        mapping = {
            Qt.Key_Space: ord(' '),
            Qt.Key_Plus: ord('+'),
            Qt.Key_Equal: ord('='),
            Qt.Key_Minus: ord('-'),
            Qt.Key_Delete: 255,
            Qt.Key_Escape: 27,
            Qt.Key_Return: 13,
            Qt.Key_Enter: 13,
            Qt.Key_Left: 2424832,
            Qt.Key_Up: 2490368,
            Qt.Key_Right: 2555904,
            Qt.Key_Down: 2621440,
            Qt.Key_F: ord('f'),
            Qt.Key_R: ord('r'),
            Qt.Key_S: ord('s'),
            Qt.Key_O: ord('o'),
            Qt.Key_M: ord('m'),
            Qt.Key_P: ord('p'),
            Qt.Key_Z: ord('z'),
            Qt.Key_Y: ord('y'),
        }
        return mapping.get(qt_key)

    # ── 操作转发 ──

    def handle_pending_action(self):
        """处理待执行操作。"""
        if self._handler:
            self._handler.handle_pending_action()
            self.need_redraw.emit()

    def check_pending(self) -> bool:
        """是否有待处理操作。"""
        return bool(getattr(self.state, 'pending_action', None))
