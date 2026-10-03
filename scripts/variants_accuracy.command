#!/bin/zsh
# Step 8a: accuracy of every inference configuration (precision on encoder / decoder / both; greedy and beam 5),
# on the full Karpathy test split, in this Terminal window. Double-click in Finder, or run:
#   open scripts/variants_accuracy.command
# Pause: Ctrl+C (finished steps are kept). Resume: open this file again.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/variants
trap : INT  # Ctrl+C reaches Python; this shell keeps waiting until it has saved and exited
echo "Configuration accuracy, $(date '+%d %b %H:%M'). Enter your Mac password to start the energy meter."
sudo -v || exit 1
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/09_evaluate_variants.py "$@" 2>&1 | tee -ia results/variants/console_accuracy.log
