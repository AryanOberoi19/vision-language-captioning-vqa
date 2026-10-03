#!/bin/zsh
# Step 7: inference cost of the reference model (energy, latency, memory), in this Terminal window.
# Double-click in Finder, or run: open scripts/inference.command
# Close other apps and keep the Mac on AC power. Pause: Ctrl+C (finished passes are kept). Resume: open this file again.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/inference
trap : INT  # Ctrl+C reaches Python; this shell keeps waiting until it has saved and exited
echo "Inference measurement, $(date '+%d %b %H:%M'). Enter your Mac password to start the energy meter."
sudo -v || exit 1
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/08_measure_inference.py "$@" 2>&1 | tee -ia results/inference/console.log
