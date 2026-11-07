#!/usr/bin/env bash
# One-time setup: creates a private venv in ~/.idiolect/venv (outside the plugin,
# so it survives plugin updates) and installs the spaCy English model.
set -euo pipefail
HOME_DIR="${IDIOLECT_HOME:-$HOME/.idiolect}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HOME_DIR"
if command -v uv >/dev/null 2>&1; then
  [ -x "$HOME_DIR/venv/bin/python" ] || uv venv --python 3.11 "$HOME_DIR/venv"
  VIRTUAL_ENV="$HOME_DIR/venv" uv pip install -r "$HERE/requirements.txt"
  INSTALL=(env VIRTUAL_ENV="$HOME_DIR/venv" uv pip install)
else
  [ -x "$HOME_DIR/venv/bin/python" ] || python3 -m venv "$HOME_DIR/venv"
  "$HOME_DIR/venv/bin/pip" install -r "$HERE/requirements.txt"
  INSTALL=("$HOME_DIR/venv/bin/pip" install)
fi
SPACY_MV=$("$HOME_DIR/venv/bin/python" -c "import spacy; print('.'.join(spacy.__version__.split('.')[:2]))")
"$HOME_DIR/venv/bin/python" -c "import en_core_web_sm" 2>/dev/null || "${INSTALL[@]}" \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-${SPACY_MV}.0/en_core_web_sm-${SPACY_MV}.0-py3-none-any.whl"
"$HOME_DIR/venv/bin/python" -c "import spacy; spacy.load('en_core_web_sm'); print('idiolect: environment ready at $HOME_DIR/venv')"
