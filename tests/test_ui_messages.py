"""交互栏文案与状态条的结构（DECISION-049）。

改前的症状：184 条日志调用各写各的，同一件事 2–4 种说法，营销词和哈希混在
提示里，底部只有一句自由文本。这些契约把"分类 + 统一句式 + 明细收起"固定下来。
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------- 文案规范


def test_normalize_strips_marketing_words_and_newlines():
    import ui_messages as m

    assert m.normalize("系统智能推荐起点；\n\n请确认") == "系统推荐起点； 请确认"
    assert "智能" not in m.normalize("已智能推荐起点")
    assert "请稍候" not in m.normalize("正在识别，请稍候")
    assert m.normalize("  多   空格  ") == "多 空格"


def test_message_templates_are_single_sentence():
    import ui_messages as m

    assert m.bands_summary(15, 14) == "主体区 15 条 · 边缘区 14 条"
    assert m.bands_summary(15, 0) == "主体区 15 条"
    assert m.mask_done(15, 14).startswith("掩膜优化完成 · ")
    assert "下一步" in m.mask_done(15, 14)
    assert m.centerlines_done(14) == "作业线已生成 · 共 14 条。下一步：勾选需要的作业线"
    assert m.selection_confirmed(14, 14) == "作业线已确认 · 14/14 条。下一步：设置起点"
    assert m.task_busy("切换模型") == "有任务在运行 · 等它结束再切换模型"
    assert m.need_first("设置起点", "规划") == "先设置起点才能规划"


def test_failed_message_carries_reason_and_hint():
    import ui_messages as m

    text = m.failed("掩膜优化", "显存不足", "降低切片大小后重试")
    assert text == "掩膜优化失败：显存不足 · 降低切片大小后重试"


def test_detail_key_ignores_numbers_for_dedup():
    import ui_messages as m

    assert m.detail_key("正在识别：12%") == m.detail_key("正在识别：87%")
    # 结果型文案数量变了就是新事件，不能合并
    assert m.detail_key("作业线 14 条") != m.detail_key("作业线 15 条")


# --------------------------------------------------------------- 日志面板


def test_log_panel_merges_repeats_and_hides_details(qapp):
    from pyside6_app.panels import LogPanel

    panel = LogPanel()
    panel.info("正在识别作物行带")
    panel.info("正在识别作物行带")
    assert panel.entry_count() == 1, "6 秒内同文案应合并，而不是刷两行"
    panel.detail("坐标摘要：EPSG:4326")
    assert "EPSG" not in panel._view.toPlainText(), "明细默认不显示"
    panel.set_show_details(True)
    assert "EPSG" in panel._view.toPlainText()


def test_log_panel_marks_levels_with_glyphs(qapp):
    from pyside6_app.panels import LogPanel

    panel = LogPanel()
    panel.success("作业线已生成")
    panel.warn("有任务在运行")
    panel.error("掩膜优化失败：显存不足")
    text = panel._view.toPlainText()
    assert "✓" in text and "!" in text and "×" in text


def test_log_panel_caps_entries(qapp):
    from pyside6_app.panels import LogPanel

    panel = LogPanel()
    for i in range(LogPanel.MAX_ENTRIES + 30):
        panel.info("事件 " + chr(0x4E00 + i) + " 号")
    assert panel.entry_count() == LogPanel.MAX_ENTRIES


# --------------------------------------------------------------- 状态条


def test_status_strip_sections(qapp):
    from pyside6_app.panels import StatusStrip

    strip = StatusStrip()
    strip.setText("掩膜优化完成（15 条主体区），正在刷新预览...")
    assert strip.text().startswith("掩膜优化完成")
    strip.set_step("第 4 步 · 掩膜优化")
    assert strip.lbl_step.isHidden() is False and "第 4 步" in strip.lbl_step.text()
    strip.set_hint("下一步：生成作业线", "warn")
    assert strip.lbl_hint.isHidden() is False
    strip.set_metrics({"主体区": "15 条", "边缘区": "14 条"})
    assert "主体区 15 条" in strip.lbl_metrics.text()
    strip.set_hint("")
    assert strip.lbl_hint.isHidden() is True


def test_status_strip_normalizes_text(qapp):
    from pyside6_app.panels import StatusStrip

    strip = StatusStrip()
    strip.setText("系统智能推荐起点")
    assert strip.text() == "系统推荐起点"
