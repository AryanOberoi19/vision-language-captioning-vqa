#!/bin/zsh
# Encoder check (ViT-L/14 vs ViT-B/32 pilot) in this Terminal window, ~2 h.
# Double-click in Finder, or run: open scripts/encoder_check.command
# Pause: Ctrl+C (the step in progress saves). Resume: open this file again.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/encoder_check
trap : INT  # Ctrl+C reaches Python; this shell keeps waiting until the step has saved and exited
echo "Encoder check, $(date '+%d %b %H:%M'). Enter your Mac password to start the energy meter."
sudo -v || exit 1
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/06_encoder_check.py 2>&1 | tee -ia results/encoder_check/console.log
