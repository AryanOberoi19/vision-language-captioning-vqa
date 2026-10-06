#!/bin/zsh
# Step 11: local demo (captions and answers at any precision, with the energy of each request), in this Terminal
# window. Double-click in Finder, or run: open scripts/demo.command
# The password starts the CPU and DRAM energy counters (press Ctrl+C at the prompt to skip: GPU energy only).
# The browser opens at http://127.0.0.1:7860 when the FP32 model is loaded (~1 min). Stop: Ctrl+C here.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
"$PY" -c "import gradio" 2>/dev/null || { echo "Installing gradio (once)"; "$PY" -m pip install -q gradio || exit 1; }
trap : INT  # Ctrl+C at the password prompt skips it; Ctrl+C later stops the demo
echo "Demo, $(date '+%d %b %H:%M'). Enter your Mac password for CPU and DRAM energy (Ctrl+C to skip)."
sudo -v || echo "Continuing with GPU energy only."
"$PY" scripts/14_demo.py "$@"
