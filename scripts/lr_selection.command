#!/bin/zsh
# Step 4b: start or resume the learning-rate selection in this Terminal window.
# Double-click in Finder, or run: open scripts/lr_selection.command
# Pause: Ctrl+C (the run finishes its current step and saves). Resume: open this file again.
# Progress at any time: python scripts/progress.py
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/lr_selection
trap : INT  # Ctrl+C reaches Python; this shell keeps waiting until the run has saved and exited
echo "Learning-rate selection, $(date '+%d %b %H:%M'). Enter your Mac password to start the energy meter."
sudo -v || exit 1
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/04_select_lr.py 2>&1 | tee -ia results/lr_selection/console.log
