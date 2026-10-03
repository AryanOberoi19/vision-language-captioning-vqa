#!/usr/bin/env python3
"""Step 6: the reference run, one epoch per start (agreed with Aryan, 2 Oct 2026).

    open scripts/reference_run.command          # Terminal window: password once, then one more epoch (~1.7-2.3 h)
    sudo -v && python scripts/07_reference_run.py [--encoder ViT-L/14] [--seed 0] [--epochs N]

Each start trains the run to one more full epoch than it has (or finishes an epoch that was paused), then scores the
new epoch_NN.pt on Karpathy val with every metric (scripts/05_evaluate.py, beam 3) and prints the table of all
epochs so far. Training goes through scripts/03_train.py, so stopping after any epoch and starting again gives the
same weights as one uninterrupted run (constant learning rate after warm-up; optimiser, scheduler, random state and
data order are restored from last.pt). Validation decoding runs only between epochs, outside the measured training
energy.

Configuration (scope decision of 2 Oct; encoder from the encoder check): COCO + VQA v2, CLIP ViT-L/14, ClipCap MLP,
GPT-2 small, LoRA rank 8 per head, lr 1e-3, AdamW, weight decay 0.01, warm-up 0.2 epoch then constant, clip 1.0,
batch 64, fp32. Seed 0 continues its encoder-check pilot (the first 4,428 steps).

Pausing: Ctrl+C; the run saves at the current step and this script stops. Open reference_run.command again to resume.
Results: results/reference/<run>.md and .json.
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
OUT = paths.RESULTS / "reference"
PAUSED = 130
PILOT = "step_004428.pt"
# Proposed stopping rule (not yet agreed): stop after the first epoch that adds less than both of these.
STOP_GAIN = {"CIDEr": 1.0, "VQA": 0.5}


def run_name(encoder: str, seed: int) -> str:
    return f"coco_{encoder.replace('/', '-')}_lora8_lr1e-3_s{seed}"


def run_child(cmd: list[str]) -> int:
    """Run one step to completion; on Ctrl+C wait for the child to save and exit instead of killing it."""
    child = subprocess.Popen(cmd)
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue


def step(cmd: list, what: str):
    print(f"\n>>> {what}  ({datetime.now():%d %b %H:%M})", flush=True)
    rc = run_child([sys.executable, *map(str, cmd)])
    if rc == PAUSED:
        print("\nPaused: progress is saved. To resume, open scripts/reference_run.command.", flush=True)
        sys.exit(PAUSED)
    if rc != 0:
        sys.exit(f"{what} failed (exit {rc}); finished work is kept. Run this script again after fixing it.")


def log_entries(run: str) -> list[dict]:
    path = paths.RESULTS / "runs" / run / "log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def finished_epochs(run: str) -> dict[int, dict]:
    return {e["epoch"]: e for e in log_entries(run) if e["type"] == "epoch"}


def eval_file(run: str, ckpt: str) -> Path:
    return paths.RESULTS / "eval" / run / f"{Path(ckpt).stem}_val_beam3.json"


def evaluate(run: str, ckpt: str):
    if not eval_file(run, ckpt).exists():
        step([SCRIPTS / "05_evaluate.py", "--run", run, "--checkpoint", ckpt, "--split", "val"],
             f"scoring {ckpt} on Karpathy val")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="ViT-L/14", choices=["ViT-L/14", "ViT-B/32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="train to this many epochs (default: one more)")
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--allow-gpu-only-energy", action="store_true",
                    help="continue even if CPU and DRAM energy cannot be measured (no sudo)")
    args = ap.parse_args()
    run = run_name(args.encoder, args.seed)
    target = args.epochs or len(finished_epochs(run)) + 1

    RunLock(OUT / ".lock")
    meter = EnergyMeter("mps")
    if meter.available and not meter.counters_live and not args.allow_gpu_only_energy:
        sys.exit("CPU and DRAM energy counters are not live. Run `sudo -v` and start this script again "
                 "(or pass --allow-gpu-only-energy).")
    if meter.keeper_pid:
        os.environ[KEEPER_ENV] = str(meter.keeper_pid)
    try:
        if len(finished_epochs(run)) < target:
            step([SCRIPTS / "03_train.py", "--data", "coco", "--run-name", run, "--encoder", args.encoder,
                  "--adaptation", "lora", "--rank", 8, "--lr", 0.001, "--epochs", target, "--batch", 64,
                  "--seed", args.seed, "--save-epochs", "all", "--idle-seconds", args.idle_seconds],
                 f"{run}: training to {target} epoch{'s' if target > 1 else ''}")
    finally:
        meter.close()
    for e in sorted(finished_epochs(run)):
        evaluate(run, f"epoch_{e:02d}.pt")
    write_summary(run)


def write_summary(run: str):
    rows = []
    pilot = [e for e in log_entries(run) if e["type"] == "partial" and e["checkpoint"] == PILOT]
    points = ([(0.25, PILOT, pilot[-1])] if pilot and eval_file(run, PILOT).exists() else []) + \
             [(e + 1, f"epoch_{e:02d}.pt", x) for e, x in sorted(finished_epochs(run).items())
              if eval_file(run, f"epoch_{e:02d}.pt").exists()]
    for epochs, ckpt, entry in points:
        r = json.loads(eval_file(run, ckpt).read_text())
        c, v = r["caption"], r["vqa"]
        energy = entry.get("train_energy") or {}
        rows.append({"epochs": epochs, "checkpoint": ckpt, "CIDEr": 100 * c["CIDEr"]["value"],
                     "VQA": 100 * v["accuracy"]["value"], "BLEU-4": 100 * c["BLEU-4"]["value"],
                     "SPICE": 100 * c["SPICE"]["value"] if "SPICE" in c else None,
                     "CHAIR_i": 100 * c["CHAIR_i"]["value"], "CHAIR_s": 100 * c["CHAIR_s"]["value"],
                     "length": c["mean_length_words"], "val_loss_caption": entry["val_loss_caption"],
                     "val_loss_vqa": entry["val_loss_vqa"], "train_hours": entry["train_seconds"] / 3600,
                     "train_wh": energy["total_j"] / 3600 if energy.get("total_j") else None,
                     "sessions": entry.get("sessions")})
    epochs_only = [r for r in rows if r["epochs"] >= 1]
    verdict = None
    if len(epochs_only) >= 2:
        a, b = epochs_only[-2], epochs_only[-1]
        gains = {m: b[m] - a[m] for m in STOP_GAIN}
        stop = all(gains[m] < STOP_GAIN[m] for m in STOP_GAIN)
        verdict = {"gains": gains, "stop": stop}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run}.json").write_text(json.dumps({"run": run, "rows": rows, "proposed_stop": verdict,
                                                  "time": datetime.now().isoformat(timespec="seconds")}, indent=1))

    def f(x, d=1):
        return "-" if x is None else f"{x:.{d}f}"
    lines = [f"# {run} (Karpathy val, beam 3)\n",
             "| Epochs | CIDEr | VQA | BLEU-4 | SPICE | CHAIR_i | CHAIR_s | Words | Val loss cap / VQA | Train h | Train Wh |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['epochs']:g} | {f(r['CIDEr'])} | {f(r['VQA'])} | {f(r['BLEU-4'])} | {f(r['SPICE'])} | "
                     f"{f(r['CHAIR_i'])} | {f(r['CHAIR_s'])} | {f(r['length'])} | "
                     f"{r['val_loss_caption']:.4f} / {r['val_loss_vqa']:.4f} | {f(r['train_hours'], 2)} | "
                     f"{f(r['train_wh'], 1)} |")
    lines.append("\nTrain h and Wh are per epoch. Epoch 1's include the pilot's 4,428 steps (row 0.25), summed over "
                 "the sessions that trained them.")
    if verdict:
        g = verdict["gains"]
        lines.append(f"\nLast epoch added CIDEr {g['CIDEr']:+.1f}, VQA {g['VQA']:+.1f}. Proposed stopping rule "
                     f"(stop when CIDEr < +{STOP_GAIN['CIDEr']} and VQA < +{STOP_GAIN['VQA']}): "
                     f"{'stop' if verdict['stop'] else 'continue'}.")
    (OUT / f"{run}.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPaused. To resume, open scripts/reference_run.command.", flush=True)
        sys.exit(PAUSED)
