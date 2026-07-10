#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
PYTHON_BIN="${REUI_PYTHON:-/d/zhl/anaconda/envs/game/python.exe}"

"$PYTHON_BIN" -m compileall -q .
"$PYTHON_BIN" -m pytest "$@"
