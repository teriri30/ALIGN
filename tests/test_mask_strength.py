"""The mask pipeline exposes one deterministic processing profile."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ALIGN_DISABLE_AUTO_START", "1")


def test_legacy_mask_strength_is_canonicalized_to_standard():
    from config import validate_config

    for legacy in ("light", "standard", "strong", "very_strong"):
        normalized = validate_config({"mask_processing": {"strength": legacy}})
        assert normalized["mask_processing"]["strength"] == "standard"


def test_mask_profile_does_not_apply_removed_strength_overrides():
    from mask_processor import _apply_strength_profile

    normalized = _apply_strength_profile({"strength": "very_strong"})

    assert normalized["strength"] == "standard"
    assert "support_close_along_m" not in normalized
    assert "body_end_extension_m" not in normalized


def test_task_panel_shows_fixed_mask_strength_without_a_combo():
    from PySide6.QtWidgets import QApplication
    from pyside6_app.panels import TaskPanel

    QApplication.instance() or QApplication([])
    panel = TaskPanel()

    assert not hasattr(panel, "mask_strength_combo")
    assert panel.mask_strength_value_label.text() == "标准（固定）"
