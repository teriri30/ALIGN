"""Shared pytest environment setup."""
import os

from runtime_env import configure_geospatial_environment


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")
# Automated suites must always run under the deterministic fail-closed export
# gate, regardless of the local config.json export.gate_mode setting.
os.environ.setdefault("REUI_EXPORT_GATE", "fail_closed")
configure_geospatial_environment()
