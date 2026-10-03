#!/usr/bin/env bash
# Step 1a: create the Python environment.
#   bash setup_env.sh
# Uses conda (env "greenvl", Python 3.11) if available, otherwise a .venv with python3.11/3.12.
# Writes requirements.lock.txt with the exact versions installed, for the methods section.
set -euo pipefail
cd "$(dirname "$0")"
ENV_NAME="${ENV_NAME:-greenvl}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
  ACTIVATE="conda activate $ENV_NAME"
else
  PY="$(command -v python3.11 || command -v python3.12 || true)"
  if [ -z "$PY" ]; then echo "Need conda or python3.11/3.12 on PATH." >&2; exit 1; fi
  "$PY" -m venv .venv
  source .venv/bin/activate
  ACTIVATE="source $(pwd)/.venv/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > requirements.lock.txt
python -c "import torch; print('torch', torch.__version__, '| MPS available:', torch.backends.mps.is_available())"
echo
echo "Done. Activate with:  $ACTIVATE"
