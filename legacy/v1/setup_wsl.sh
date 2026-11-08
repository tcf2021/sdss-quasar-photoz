#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv first using the README, then rerun this script.'
  exit 1
fi
mode="${1:-cuda}"
case "$mode" in
  cuda) index='https://download.pytorch.org/whl/cu128' ;;
  cpu) index='https://download.pytorch.org/whl/cpu' ;;
  *) echo 'Usage: bash setup_wsl.sh [cuda|cpu]'; exit 2 ;;
esac
uv python install 3.11
if [[ ! -d .venv ]]; then
  uv venv --python 3.11 --seed .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a separate Python 3.11 .venv"'
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch==2.9.0 --index-url "$index"
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > environment-lock.txt
.venv/bin/python project.py check-env --device "$mode"
echo 'Environment ready. Run: source .venv/bin/activate'
