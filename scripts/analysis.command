#!/bin/zsh
# Step 10: analysis of the inference configurations (frontiers, hallucination, retention, Grad-CAM, batch check),
# in this Terminal window. Double-click in Finder, or run: open scripts/analysis.command
# No password needed (nothing measures energy). ~45-60 min. Re-running reuses the cached per-image scores.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/analysis
echo "Analysis, $(date '+%d %b %H:%M')."
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/11_analysis.py "$@" 2>&1 | tee -ia results/analysis/console.log
