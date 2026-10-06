#!/bin/zsh
# VizWiz-Captions zero-shot check (reference + frontier configurations), in this Terminal window.
# Double-click in Finder, or run: open scripts/vizwiz.command
# No password needed. ~1 h. Pause: Ctrl+C (finished steps are kept). Resume: open this file again.
cd "${0:A:h}/.." || exit 1
PY=/opt/anaconda3/envs/greenvl/bin/python
[ -x "$PY" ] || PY=python
export PYTHONUNBUFFERED=1
mkdir -p results/vizwiz
trap : INT  # Ctrl+C reaches Python; this shell keeps waiting until it has saved and exited
echo "VizWiz zero-shot check, $(date '+%d %b %H:%M')."
caffeinate -i -w $$ &  # keeps the Mac awake while this window runs (closing the lid still sleeps it)
"$PY" scripts/12_vizwiz.py "$@" 2>&1 | tee -ia results/vizwiz/console.log
