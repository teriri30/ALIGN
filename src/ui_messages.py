"""界面文案的唯一出口：级别、句式、去重键与固定模板（DECISION-049）。

为什么要有这个模块
------------------
改之前 184 条日志调用散在 `main_window.py` 里，同一件事有 2–4 种说法
（"当前有任务正在运行，暂不能切换模型"有 4 个变体），还混进了"智能推荐""已成功"
这类词，以及哈希、EPSG、缓存版本这些不该出现在面向用户位置的细节。

三条规则
--------
1. **一句话只讲一件事**：阶段（正在做什么）／结果（得到什么，带数字）／
   提示（下一步）／异常（出了什么事 + 怎么办）／明细（参数、路径、哈希）。
2. **句式固定**：
   - 阶段  `正在<动词><对象>`          "正在识别作物行带"
   - 结果  `<对象> <数量><单位>`        "作业线 14 条"
   - 完成  `<对象>已<动作>`            "作业线已生成"
   - 提示  `<前置>后才能<动作>`        "设置起点后才能规划"
   - 异常  `<对象><动作>失败：<原因> · <怎么办>`
3. **不用**：智能／一键／自动优化／已成功／请稍候／抱歉／感叹号。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class Level:
    """日志级别。DETAIL 只进"详情"，默认不显示在交互栏。"""

    INFO = "info"
    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    DETAIL = "detail"


# 级别 → 前缀字形（配合主题色，见 LogPanel.refresh_theme）
GLYPHS = {
    Level.INFO: "·",
    Level.OK: "✓",
    Level.WARN: "!",
    Level.ERROR: "×",
    Level.DETAIL: "⋯",
}

_BANNED = ("智能", "一键", "自动优化", "已成功", "请稍候", "抱歉", "！！")
_BANNED_REPLACEMENTS = {"智能推荐": "推荐", "系统智能": "系统", "已成功": ""}


def normalize(text: str) -> str:
    """统一文案：去营销词、压缩空白、去掉句尾重复标点。"""
    value = str(text or "")
    for bad, good in _BANNED_REPLACEMENTS.items():
        value = value.replace(bad, good)
    for bad in _BANNED:
        value = value.replace(bad, "")
    # 多行压成单行：交互栏只有一行的高度，长句靠模板拆
    value = re.sub(r"\s*\n+\s*", " ", value)
    value = re.sub(r"[ \t]{2,}", " ", value).strip()
    value = value.replace("！！", "！").replace("！", "")
    value = re.sub(r"[。．.]{2,}$", "。", value)
    return value.strip()


def detail_key(text: str) -> str:
    """去重键。

    只对"进度型"文案（以"正在"开头或含百分比）抹掉数字，让
    "正在识别：12%" 与 "正在识别：47%" 合并成一条 ×N；
    结果型文案（"作业线 14 条"）保持原样，数量变了就是新事件。
    """
    value = normalize(text)
    if value.startswith("正在") or "%" in value:
        return re.sub(r"\d+(?:\.\d+)?", "#", value)
    return value


# --------------------------------------------------------------------- 结果类
def bands_summary(main_count: int, edge_count: int = 0) -> str:
    """"主体区 15 条 · 边缘区 14 条"。"""
    parts = [f"主体区 {int(main_count)} 条"]
    if int(edge_count) > 0:
        parts.append(f"边缘区 {int(edge_count)} 条")
    return " · ".join(parts)


def mask_done(main_count: int, edge_count: int = 0) -> str:
    return f"掩膜优化完成 · {bands_summary(main_count, edge_count)}。下一步：生成作业线"


def centerlines_done(count: int) -> str:
    return f"作业线已生成 · 共 {int(count)} 条。下一步：勾选需要的作业线"


def selection_confirmed(selected: int, total: int) -> str:
    return f"作业线已确认 · {int(selected)}/{int(total)} 条。下一步：设置起点"


def origin_suggested(mode: str = "reference") -> str:
    if mode == "geometry":
        return "起点已推荐 · 可拖动调整；终点随最后一段自动延伸"
    return "起点已推荐 · 可拖动调整；终点按需在画布上设置"


def project_restored(next_step: str = "") -> str:
    """项目缓存恢复后的状态条结论：现在的状态 + 下一步。

    「正在读取逐步操作历史…」这类进度文案必须被结论替换掉，否则恢复完成后
    状态条一直停在过程中间，用户会以为程序卡住了（2026-10-26）。
    """
    step = normalize(next_step or "")
    return f"已恢复最近项目 · {step}" if step else "已恢复最近项目"


def path_done(work_lines: int, turns: int, length_m: float) -> str:
    return (
        f"路径已生成 · 作业段 {int(work_lines)} 段，转弯 {int(turns)} 处，"
        f"总长 {float(length_m):.1f} m。下一步：导出或模拟演示"
    )


def export_done(path_name: str) -> str:
    return f"已导出 {path_name}"


# --------------------------------------------------------------------- 前置类
def task_busy(action: str) -> str:
    """所有"有任务在跑"统一成这一句，不再一个动作一个说法。"""
    return f"有任务在运行 · 等它结束再{action}"


def need_first(step: str, action: str) -> str:
    return f"先{step}才能{action}"


def not_ready(step: str) -> str:
    return f"{step}还没完成 · 完成后才能继续"


def missing_cache(what: str, rerun: str) -> str:
    return f"没有{what}缓存 · 重新{rerun}一次"


# --------------------------------------------------------------------- 异常类
def failed(what: str, reason: str = "", hint: str = "") -> str:
    text = f"{what}失败"
    reason = normalize(reason)
    if reason:
        text += f"：{reason}"
    if hint:
        text += f" · {hint}"
    return text


@dataclass(frozen=True)
class Entry:
    """一条待显示的消息。"""

    level: str
    text: str

    @property
    def glyph(self) -> str:
        return GLYPHS.get(self.level, "·")


def make(level: str, text: str) -> Entry:
    return Entry(level, normalize(text))
