#!/usr/bin/env python3
"""Encoder check before the reference run (agreed with Aryan, 2 Oct 2026; rule fixed before any ViT-L/14 result).

    open scripts/encoder_check.command          # Terminal window: password once, then ~2 h
    sudo -v && python scripts/06_encoder_check.py

Only one configuration will be trained, so its encoder is chosen first. ViT-L/14 costs about the same to train as
ViT-B/32 (features are cached once), but about 3x more per caption and 7x more per answer at inference on the Mac
(encoder 27 vs 414 images/s). It is chosen only if its accuracy gain is clearly worth that.

1. Cache ViT-L/14 features for the COCO Karpathy sets (~70 min, resumable by shard).
2. Train the ViT-B/32 pilot's exact configuration on them: COCO + VQA v2, LoRA rank 8, lr 1e-3, seed 0, batch 64,
   4,428 steps (a quarter epoch). Same data order as the B/32 pilot.
3. Score both pilots on Karpathy val (scripts/05_evaluate.py, beam 3).

Rule: ViT-L/14 is chosen if its validation CIDEr is higher by at least 5.0 points or its VQA accuracy by at least
2.0 points, and neither metric is lower than ViT-B/32's by more than 1.0 CIDEr / 0.5 VQA point. Otherwise ViT-B/32
stays, and the reference run continues from the B/32 pilot. The winning pilot is the first quarter epoch of the
reference run.

Pausing: Ctrl+C; the step in progress saves and this script stops. Open encoder_check.command again to resume.
Results: results/encoder_check/summary.md and summary.json.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl import paths  # noqa: E402
from greenvl.energy import KEEPER_ENV, EnergyMeter  # noqa: E402
from greenvl.lock import RunLock  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
OUT = paths.RESULTS / "encoder_check"
PAUSED = 130
SETS = ("coco_train", "coco_val", "coco_test")
STEPS = 4428
CHECKPOINT = f"step_{STEPS:06d}.pt"
RUNS = {"ViT-B/32": "coco_ViT-B-32_lora8_lr1e-3_s0", "ViT-L/14": "coco_ViT-L-14_lora8_lr1e-3_s0"}
GAIN = {"CIDEr": 5.0, "VQA": 2.0}       # points that make ViT-L/14 worth its inference cost
TOLERANCE = {"CIDEr": 1.0, "VQA": 0.5}  # largest drop allowed on the other metric


def run_child(cmd: list[str]) -> int:
    """Run one step to completion; on Ctrl+C wait for the child to save and exit instead of killing it."""
    child = subprocess.Popen(cmd)
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue


def step(cmd: list[str], what: str):
    print(f"\n>>> {what}  ({datetime.now():%H:%M})", flush=True)
    rc = run_child([sys.executable, *map(str, cmd)])
    if rc == PAUSED:
        print("\nPaused: progress is saved. To resume, open scripts/encoder_check.command.", flush=True)
        sys.exit(PAUSED)
    if rc != 0:
        sys.exit(f"{what} failed (exit {rc}); finished work is kept. Run this script again after fixing it.")


def pilot_entry(run: str) -> dict | None:
    log = paths.RESULTS / "runs" / run / "log.jsonl"
    if not log.exists():
        return None
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    done = [e for e in entries if e["type"] == "partial" and e["step"] >= STEPS]
    return done[-1] if done else None


def eval_path(run: str) -> Path:
    return paths.RESULTS / "eval" / run / f"{Path(CHECKPOINT).stem}_val_beam3.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--allow-gpu-only-energy", action="store_true",
                    help="continue even if CPU and DRAM energy cannot be measured (no sudo)")
    args = ap.parse_args()

    RunLock(OUT / ".lock")
    meter = EnergyMeter("mps")
    if meter.available and not meter.counters_live and not args.allow_gpu_only_energy:
        sys.exit("CPU and DRAM energy counters are not live. Run `sudo -v` and start this script again "
                 "(or pass --allow-gpu-only-energy).")
    if meter.keeper_pid:
        os.environ[KEEPER_ENV] = str(meter.keeper_pid)  # every step reuses this powermetrics
    try:
        if pilot_entry(RUNS["ViT-B/32"]) is None:
            sys.exit(f"The ViT-B/32 pilot ({RUNS['ViT-B/32']}, {STEPS:,} steps) is missing.")
        folder = paths.FEATURES / "ViT-L-14"
        missing = [s for s in SETS if not (folder / f"{s}.pt").exists()]
        if missing:
            step([SCRIPTS / "02_extract_features.py", "--encoder", "ViT-L/14", "--sets", *missing,
                  "--idle-seconds", args.idle_seconds], f"caching ViT-L/14 features for {', '.join(missing)}")
        run = RUNS["ViT-L/14"]
        if pilot_entry(run) is None:
            step([SCRIPTS / "03_train.py", "--data", "coco", "--run-name", run, "--encoder", "ViT-L/14",
                  "--adaptation", "lora", "--rank", 8, "--lr", 0.001, "--epochs", 1, "--batch", 64, "--seed", 0,
                  "--save-epochs", "all", "--idle-seconds", args.idle_seconds, "--max-steps", STEPS],
                 f"ViT-L/14 pilot, {STEPS:,} steps")
    finally:
        meter.close()
    for enc, run in RUNS.items():
        if not eval_path(run).exists():
            step([SCRIPTS / "05_evaluate.py", "--run", run, "--checkpoint", CHECKPOINT, "--split", "val"],
                 f"evaluating the {enc} pilot on Karpathy val")
    write_summary()


def write_summary():
    rows = {}
    for enc, run in RUNS.items():
        r = json.loads(eval_path(run).read_text())
        p = pilot_entry(run)
        c, v = r["caption"], r["vqa"]
        rows[enc] = {"run": run, "CIDEr": 100 * c["CIDEr"]["value"], "CIDEr_ci95": [100 * x for x in c["CIDEr"]["ci95"]],
                     "VQA": 100 * v["accuracy"]["value"], "VQA_ci95": [100 * x for x in v["accuracy"]["ci95"]],
                     "BLEU-4": 100 * c["BLEU-4"]["value"], "SPICE": 100 * c["SPICE"]["value"] if "SPICE" in c else None,
                     "CLIPScore": 100 * c["CLIPScore"]["value"], "CHAIR_i": 100 * c["CHAIR_i"]["value"],
                     "val_loss_caption": p["val_loss_caption"], "val_loss_vqa": p["val_loss_vqa"],
                     "train_seconds": p["train_seconds"],
                     "train_wh": p["train_energy"]["total_j"] / 3600 if p.get("train_energy") else None}
    b, l = rows["ViT-B/32"], rows["ViT-L/14"]
    delta = {m: l[m] - b[m] for m in ("CIDEr", "VQA")}
    gain = any(delta[m] >= GAIN[m] for m in GAIN)
    no_loss = all(delta[m] >= -TOLERANCE[m] for m in TOLERANCE)
    chosen = "ViT-L/14" if gain and no_loss else "ViT-B/32"
    summary = {"rule": f"ViT-L/14 if CIDEr +{GAIN['CIDEr']} or VQA +{GAIN['VQA']} points, and neither lower by more "
                       f"than {TOLERANCE['CIDEr']} CIDEr / {TOLERANCE['VQA']} VQA; else ViT-B/32",
               "steps": STEPS, "pilots": rows, "delta": delta, "chosen": chosen,
               "reference_run": RUNS[chosen], "time": datetime.now().isoformat(timespec="seconds")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    def f(x, d=1):
        return "-" if x is None else f"{x:.{d}f}"
    lines = [f"# Encoder check ({STEPS:,}-step pilots, Karpathy val, beam 3)\n", f"Rule: {summary['rule']}.\n",
             "| | ViT-B/32 | ViT-L/14 | Difference |", "|---|---|---|---|"]
    for m, d in (("CIDEr", 1), ("VQA", 1), ("BLEU-4", 1), ("SPICE", 1), ("CLIPScore", 1), ("CHAIR_i", 1),
                 ("val_loss_caption", 4), ("val_loss_vqa", 4), ("train_seconds", 0), ("train_wh", 2)):
        diff = None if b[m] is None or l[m] is None else l[m] - b[m]
        lines.append(f"| {m} | {f(b[m], d)} | {f(l[m], d)} | {'' if diff is None else f'{diff:+.{d}f}'} |")
    lines.append(f"\nChosen: **{chosen}**; the reference run continues {RUNS[chosen]}.")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPaused. To resume, open scripts/encoder_check.command.", flush=True)
        sys.exit(PAUSED)
