#!/bin/zsh
# Step 6: reference run, one more epoch per start (then scored on Karpathy val), in this Terminal window.
# Double-click in Finder, or run: open scripts/reference_run.command
# Pause: Ctrl+C (the run saves at the current step). Resume: open this file again.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/reference
trap : INT  # Ctrl+C reaches Python; this shell keeps waiting until the run has saved and exited
echo "Reference run, $(date '+%d %b %H:%M'). Enter your Mac password to start the energy meter."
sudo -v || exit 1
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/07_reference_run.py "$@" 2>&1 | tee -ia results/reference/console.log
