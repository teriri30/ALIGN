"""通用工具函数"""
import math
from typing import Dict, Any, Tuple
from PIL import ImageFont
import numpy as np

_font_cache: Dict[int, Any] = {}


def get_chinese_font(size: int = 16):
    """获取抗锯齿中文 TrueType 字体，带缓存"""
    key = (size, 'chinese')
    if key in _font_cache:
        return _font_cache[key]

    # Windows 常用中文字体完整路径（按首选项排序）
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑 — 最佳
        "C:/Windows/Fonts/msyhbd.ttc",   # 微软雅黑粗体
        "C:/Windows/Fonts/deng.ttf",     # 等线
        "C:/Windows/Fonts/simhei.ttf",   # 黑体
        "C:/Windows/Fonts/SimSun.ttc",   # 宋体
    ]

    for path in font_candidates:
        try:
            ft = ImageFont.truetype(path, size)
            _font_cache[key] = ft
            return ft
        except Exception:
            continue

    # 最后的回退：用 PIL 默认字体但尝试放大（避免位图模糊）
    try:
        ft = ImageFont.truetype("arial.ttf", size)
        _font_cache[key] = ft
        return ft
    except Exception:
        ft = ImageFont.load_default()
        _font_cache[key] = ft
        return ft


def point_to_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    line_mag = math.hypot(dx, dy)
    if line_mag == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (line_mag * line_mag)
    if t < 0:
        return math.hypot(px - x1, py - y1)
    if t > 1:
        return math.hypot(px - x2, py - y2)
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def polygon_area(pts: np.ndarray) -> float:
    return 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) -
                     np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
