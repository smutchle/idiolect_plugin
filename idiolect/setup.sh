#!/usr/bin/env bash
# One-time setup: creates a private venv in ~/.idiolect/venv (outside the plugin, so it
# survives plugin updates) and installs the spaCy English model.
# Works on Linux, macOS and Windows (Git Bash).
set -euo pipefail
HOME_DIR="${IDIOLECT_HOME:-$HOME/.idiolect}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME_DIR/venv"
mkdir -p "$HOME_DIR"

venv_python() {   # venv layout differs: bin/python (POSIX) vs Scripts/python.exe (Windows)
  if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"
  elif [ -x "$VENV/Scripts/python.exe" ]; then echo "$VENV/Scripts/python.exe"
  else echo ""; fi
}

find_system_python() {   # python3 is often missing on Windows; try the usual launchers
  for c in python3 python "py -3"; do
    if $c -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

export PYTHONUTF8=1
if command -v uv >/dev/null 2>&1; then
  [ -n "$(venv_python)" ] || uv venv --python 3.11 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install -r "$HERE/requirements.txt"
  INSTALL=(env VIRTUAL_ENV="$VENV" uv pip install)
else
  if [ -z "$(venv_python)" ]; then
    SYS_PY="$(find_system_python)" || { echo "idiolect: need Python 3.10+ (python3, python or py -3) or uv on PATH" >&2; exit 1; }
    $SYS_PY -m venv "$VENV"
  fi
  PY="$(venv_python)"
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install -r "$HERE/requirements.txt"
  INSTALL=("$PY" -m pip install)
fi
PY="$(venv_python)"
[ -n "$PY" ] || { echo "idiolect: venv creation failed at $VENV" >&2; exit 1; }
SPACY_MV=$("$PY" -c "import spacy; print('.'.join(spacy.__version__.split('.')[:2]))")
"$PY" -c "import en_core_web_sm" 2>/dev/null || "${INSTALL[@]}" \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-${SPACY_MV}.0/en_core_web_sm-${SPACY_MV}.0-py3-none-any.whl"
"$PY" -c "import spacy; spacy.load('en_core_web_sm')"
echo "idiolect: environment ready at $VENV"
