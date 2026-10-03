#!/bin/zsh
# Step 8b: energy, latency and memory of every inference configuration, in this Terminal window.
# Double-click in Finder, or run: open scripts/variants_energy.command
# Close other apps, keep the Mac on AC power and leave it alone while it runs.
# Pause: Ctrl+C (finished windows are kept). Resume: open this file again.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/variants
trap : INT  # Ctrl+C reaches Python; this shell keeps waiting until it has saved and exited
echo "Configuration energy, $(date '+%d %b %H:%M'). Enter your Mac password to start the energy meter."
sudo -v || exit 1
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/10_measure_variants.py "$@" 2>&1 | tee -ia results/variants/console_energy.log
