"""Shared pytest environment setup."""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from runtime_env import configure_geospatial_environment


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ALIGN_DISABLE_AUTO_START", "1")
# The managed test runner cannot write to its synthetic LOCALAPPDATA path.
# Keep test state isolated without changing the production default used by a
# normal Windows user (DECISION-018).
if not os.environ.get("ALIGN_USER_DATA_DIR"):
    _pytest_user_data = tempfile.mkdtemp(prefix="align-pytest-")
    os.environ["ALIGN_USER_DATA_DIR"] = _pytest_user_data
    atexit.register(shutil.rmtree, _pytest_user_data, ignore_errors=True)
# Automated suites must always run under the deterministic fail-closed export
# gate, regardless of the local config.json export.gate_mode setting.
os.environ.setdefault("ALIGN_EXPORT_GATE", "fail_closed")
configure_geospatial_environment()
