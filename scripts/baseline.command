#!/bin/zsh
# CNN-LSTM baseline (Show and Tell over ResNet-50 features): features, training, test evaluation, inference cost,
# in this Terminal window. Double-click in Finder, or run: open scripts/baseline.command
# Password once (energy meter), then ~1.5-2 h. Pause: Ctrl+C (finished epochs and steps are kept).
# Resume: open this file again.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/baseline
trap : INT  # Ctrl+C reaches Python; this shell keeps waiting until it has saved and exited
echo "CNN-LSTM baseline, $(date '+%d %b %H:%M'). Enter your Mac password to start the energy meter."
sudo -v || exit 1
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/13_baseline.py "$@" 2>&1 | tee -ia results/baseline/console.log
