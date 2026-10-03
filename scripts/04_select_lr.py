#!/usr/bin/env python3
"""Learning-rate selection before the training grid, run end to end (plan agreed 29 Sep 2026).

    sudo -v && python scripts/04_select_lr.py        # all stages, ~3 h, resumable: re-run to continue
    open scripts/lr_selection.command                # the same in a new Terminal window (or double-click it)
    python scripts/04_select_lr.py --plan            # show progress and what is left, run nothing

Stage A  Flickr8k, reference configuration (ViT-B/32, LoRA rank 8), learning rates 3e-5, 1e-4 and 3e-4 for
         3 epochs each. If the best is at an end of that range, the next rate beyond it is added. The two rates
         with the lowest validation loss after 3 epochs are extended to 6 epochs; the chosen rate is the one whose
         lowest validation loss over those 6 epochs is lower.
Stage B  Flickr8k, full fine-tuning of GPT-2 at the chosen rate and at the next lower rate, 3 epochs each.
         Reported, not acted on: whether full fine-tuning keeps the shared rate (§5 as written) or gets its own
         is Aryan's decision.
Stage C  COCO + VQA v2, both heads, a quarter epoch at the chosen rate and at the next lower rate (ViT-B/32 COCO
         features are cached first if missing). The lower rate replaces the chosen one only if its validation loss
         is lower by more than 0.02 on either task, or the chosen rate failed with a non-finite loss. The pilot at
         the rate that is kept is the first quarter epoch of reference run seed 0; that run continues from it.

Stage A and B runs keep only their newest weights file (development runs; saves ~5 GB). The script refuses to
start with less than 12 GB free.

Pausing: Ctrl+C. The run in progress finishes its current step, saves, and this script stops; run it again to
resume (sudo -v first). Feature caching resumes from its last saved shard of 8,192 images.

The rules are fixed here, before any result is seen. Every run is an ordinary scripts/03_train.py run in
results/runs/ (energy included). This script starts one background powermetrics for all of them, so sudo is
needed once, at the start. Results: results/lr_selection/summary.md and summary.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl import paths  # noqa: E402
from greenvl.energy import KEEPER_ENV, EnergyMeter  # noqa: E402
from greenvl.lock import RunLock  # noqa: E402

GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
STAGE_A_RATES = [3e-5, 1e-4, 3e-4]
MARGIN = 0.02          # validation-loss difference (nats per token) that counts as "better"
RISE = 0.01            # a final-epoch rise above the best earlier epoch larger than this marks a rate unstable
PILOT_FRACTION = 0.25  # of one COCO epoch
ENCODER = "ViT-B/32"
NEEDED_GB = 12         # all stages write ~8.5 GB (full fine-tuning checkpoints are the bulk); keep a margin

SCRIPTS = Path(__file__).resolve().parent
TRAIN = Path(os.environ.get("GREENVL_TRAIN_SCRIPT", SCRIPTS / "03_train.py"))
EXTRACT = Path(os.environ.get("GREENVL_EXTRACT_SCRIPT", SCRIPTS / "02_extract_features.py"))
OUT = paths.RESULTS / "lr_selection"
PAUSED = 130  # exit code of a step stopped with Ctrl+C after saving its progress


def run_child(cmd: list[str]) -> int:
    """Run one step to completion. Ctrl+C reaches the child as well, which saves its progress and exits; this
    process waits for that instead of killing the child mid-save."""
    child = subprocess.Popen(cmd)
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue


def paused():
    print("\nPaused: progress is saved. To resume, open scripts/lr_selection.command "
          "(or run `sudo -v && python scripts/04_select_lr.py`).", flush=True)
    sys.exit(PAUSED)


def fmt(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def name_a(lr):
    return f"f8k_lr{fmt(lr)}"


def name_b(lr):
    return f"f8k_full_lr{fmt(lr)}"


def name_c(lr):
    return f"coco_{ENCODER.replace('/', '-')}_lora8_lr{fmt(lr)}_s0"


def lower(lr):
    i = GRID.index(lr)
    return GRID[i - 1] if i > 0 else lr / 3


# ---------------------------------------------------------------- run state

def read_run(name: str) -> dict:
    """Validation losses and failures of one run, from its log.jsonl (later entries win)."""
    path = paths.RESULTS / "runs" / name / "log.jsonl"
    state = {"name": name, "epochs": {}, "partial": None, "errors": []}
    if not path.exists():
        return state
    for line in path.read_text().splitlines():
        x = json.loads(line)
        if x["type"] == "epoch":
            state["epochs"][x["epoch"]] = x
        elif x["type"] == "partial":
            state["partial"] = x
        elif x["type"] == "error":
            state["errors"].append(x["error"])
    return state


def curve(state: dict, task: str = "caption") -> list[float]:
    return [state["epochs"][e][f"val_loss_{task}"] for e in sorted(state["epochs"])]


def nonfinite(state: dict) -> bool:
    return any("non-finite" in e for e in state["errors"])


def unstable(state: dict) -> bool:
    c = curve(state)
    return nonfinite(state) or (len(c) >= 2 and c[-1] > min(c[:-1]) + RISE)


class Runner:
    def __init__(self, args, plan_only: bool):
        self.args, self.plan_only, self.pending = args, plan_only, []

    def train(self, name: str, lr: float, data: str, epochs: int, adaptation: str = "lora",
              max_steps: int | None = None, keep: str = "all") -> dict:
        """Bring a run to `epochs` full epochs (or to `max_steps`), resuming or extending it as needed."""
        state = read_run(name)
        if max_steps:
            done = state["epochs"] or (state["partial"] and state["partial"]["step"] >= max_steps)
        else:
            done = len(state["epochs"]) >= epochs
        if done or nonfinite(state):
            return state
        cmd = [sys.executable, str(TRAIN), "--data", data, "--run-name", name, "--lr", str(lr),
               "--epochs", str(epochs), "--adaptation", adaptation, "--encoder", ENCODER,
               "--batch", str(self.args.batch), "--idle-seconds", str(self.args.idle_seconds),
               "--save-epochs", keep]
        if max_steps:
            cmd += ["--max-steps", str(max_steps)]
        if self.plan_only:
            self.pending.append(" ".join(cmd[1:]))
            return state
        print(f"\n>>> {name}: lr {fmt(lr)}, {adaptation}, {data}, "
              + (f"{max_steps:,} steps" if max_steps else f"{epochs} epochs"), flush=True)
        rc = run_child(cmd)
        if rc == PAUSED:
            paused()
        state = read_run(name)
        if rc != 0 and not nonfinite(state):
            sys.exit(f"{name} stopped with an error ({state['errors'][-1] if state['errors'] else f'exit {rc}'}). "
                     "Fix it and run this script again; finished work is kept.")
        return state


# ---------------------------------------------------------------- stages

def stage_a(run: Runner, summary: dict):
    s, e = run.args.short_epochs, run.args.long_epochs
    rates = list(STAGE_A_RATES)
    states = {lr: run.train(name_a(lr), lr, "flickr8k", s, keep="last") for lr in rates}
    if run.plan_only and run.pending:
        return None

    def at_short(lr):
        c = curve(states[lr])
        return c[s - 1] if len(c) >= s else float("inf")

    ranked = sorted((lr for lr in rates if not unstable(states[lr])), key=at_short)
    if ranked and ranked[0] in (min(rates), max(rates)):  # best at an end of the range: look one step further
        i = GRID.index(ranked[0]) + (1 if ranked[0] == max(rates) else -1)
        extra = GRID[i] if 0 <= i < len(GRID) else None
        if extra is not None:
            rates.append(extra)
            states[extra] = run.train(name_a(extra), extra, "flickr8k", s, keep="last")
            if run.plan_only and run.pending:
                return None
            ranked = sorted((lr for lr in rates if not unstable(states[lr])), key=at_short)
    if not ranked:
        sys.exit("Every learning rate was unstable on Flickr8k; nothing to choose from.")
    top = ranked[:2]
    for lr in top:
        states[lr] = run.train(name_a(lr), lr, "flickr8k", e, keep="last")
    if run.plan_only and run.pending:
        return None
    chosen = min(top, key=lambda lr: min(curve(states[lr])))
    other = [lr for lr in top if lr != chosen]
    summary["stage_a"] = {
        "short_epochs": s, "long_epochs": e,
        "rule": f"rank by validation loss after {s} epochs, extend the best two to {e}, choose the lower minimum",
        "runs": {fmt(lr): {"run": name_a(lr), "val_loss_caption": curve(states[lr]), "unstable": unstable(states[lr])}
                 for lr in sorted(rates)},
        "extended": [fmt(lr) for lr in top],
        "chosen": fmt(chosen),
        "margin_over_runner_up": (round(min(curve(states[other[0]])) - min(curve(states[chosen])), 4)
                                  if other else None),
    }
    return chosen


def stage_b(run: Runner, summary: dict, chosen: float):
    s = run.args.short_epochs
    rates = [chosen, lower(chosen)]
    states = {lr: run.train(name_b(lr), lr, "flickr8k", s, adaptation="full", keep="last") for lr in rates}
    if run.plan_only and run.pending:
        return
    best = {lr: (min(curve(states[lr])) if curve(states[lr]) else float("inf")) for lr in rates}
    prefers_lower = best[rates[1]] < best[rates[0]] - MARGIN
    summary["stage_b"] = {
        "rule": f"full fine-tuning at the chosen and next lower rate, {s} epochs; lower counts as better by > {MARGIN}",
        "runs": {fmt(lr): {"run": name_b(lr), "val_loss_caption": curve(states[lr]), "unstable": unstable(states[lr])}
                 for lr in rates},
        "full_finetuning_prefers_lower_rate": prefers_lower,
        "note": ("Full fine-tuning did better at the lower rate: decide whether it keeps the shared rate (§5 as "
                 "written) or gets its own." if prefers_lower else
                 "The shared rate is not worse for full fine-tuning; §5 can stay as written."),
    }


def pilot_steps(batch: int) -> int:
    pairs = json.loads((paths.PROCESSED / "summary.json").read_text())["coco"]["train_pairs"]
    steps_per_epoch = round((pairs // batch) / 0.5)  # captions and VQA batches drawn with equal probability
    return max(1, round(steps_per_epoch * PILOT_FRACTION))


def ensure_coco_features(run: Runner):
    folder = paths.FEATURES / ENCODER.replace("/", "-")
    missing = [s for s in ("coco_train", "coco_val", "coco_test") if not (folder / f"{s}.pt").exists()]
    if not missing:
        return
    cmd = [sys.executable, str(EXTRACT), "--encoder", ENCODER, "--sets", *missing,
           "--idle-seconds", str(run.args.idle_seconds)]
    if run.plan_only:
        run.pending.append(" ".join(cmd[1:]))
        return
    print(f"\n>>> caching {ENCODER} features for {', '.join(missing)}", flush=True)
    rc = run_child(cmd)
    if rc == PAUSED:
        paused()
    if rc != 0:
        sys.exit("Feature extraction failed; run this script again to retry.")


def stage_c(run: Runner, summary: dict, chosen: float):
    ensure_coco_features(run)
    if run.plan_only and run.pending:
        return None
    steps = pilot_steps(run.args.batch)
    rates = [chosen, lower(chosen)]
    states = {lr: run.train(name_c(lr), lr, "coco", 1, max_steps=steps) for lr in rates}
    if run.plan_only and run.pending:
        return None

    def val(lr, task):
        p = states[lr]["partial"]
        return p[f"val_loss_{task}"] if p else float("inf")

    better = {t: val(rates[1], t) < val(rates[0], t) - MARGIN for t in ("caption", "vqa")}
    switch = nonfinite(states[chosen]) or any(better.values())
    kept = rates[1] if switch else rates[0]
    summary["stage_c"] = {
        "rule": f"{steps:,}-step pilot on COCO + VQA v2; switch to the lower rate if it is better by > {MARGIN} on "
                "either task or the chosen rate fails",
        "steps": steps,
        "runs": {fmt(lr): {"run": name_c(lr), "val_loss_caption": val(lr, "caption"), "val_loss_vqa": val(lr, "vqa"),
                           "failed": nonfinite(states[lr])} for lr in rates},
        "lower_rate_better": better,
        "final_rate": fmt(kept),
        "reference_run": name_c(kept),
    }
    return kept


# ---------------------------------------------------------------- report

def write_summary(summary: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# Learning-rate selection\n", f"Updated {summary['updated']}.\n"]
    if "stage_a" in summary:
        a = summary["stage_a"]
        lines += ["## Stage A: Flickr8k, LoRA r8, ViT-B/32\n", f"Rule: {a['rule']}.\n",
                  "| Rate | Validation loss per epoch | Unstable |", "|---|---|---|"]
        lines += [f"| {lr} | {', '.join(f'{v:.4f}' for v in r['val_loss_caption'])} | {r['unstable']} |"
                  for lr, r in a["runs"].items()]
        lines += ["", f"Extended to {a['long_epochs']} epochs: {', '.join(a['extended'])}. Chosen: **{a['chosen']}**"
                  + (f" (lower minimum by {a['margin_over_runner_up']})" if a["margin_over_runner_up"] is not None else "")
                  + ".\n"]
    if "stage_b" in summary:
        b = summary["stage_b"]
        lines += ["## Stage B: Flickr8k, full fine-tuning\n", "| Rate | Validation loss per epoch |", "|---|---|"]
        lines += [f"| {lr} | {', '.join(f'{v:.4f}' for v in r['val_loss_caption'])} |" for lr, r in b["runs"].items()]
        lines += ["", b["note"] + "\n"]
    if "stage_c" in summary:
        c = summary["stage_c"]
        lines += [f"## Stage C: COCO + VQA v2 pilot, {c['steps']:,} steps\n",
                  "| Rate | Caption validation loss | VQA validation loss |", "|---|---|---|"]
        lines += [f"| {lr} | {r['val_loss_caption']:.4f} | {r['val_loss_vqa']:.4f} |" for lr, r in c["runs"].items()]
        lines += ["", f"Final learning rate: **{c['final_rate']}**. Reference run seed 0 continues from "
                      f"`{c['reference_run']}`.\n"]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="show progress and the runs still to do; run nothing")
    ap.add_argument("--stages", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--short-epochs", type=int, default=3)
    ap.add_argument("--long-epochs", type=int, default=6)
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--ignore-disk-check", action="store_true", help=f"start even with less than {NEEDED_GB} GB free")
    ap.add_argument("--allow-gpu-only-energy", action="store_true",
                    help="continue even if CPU and DRAM energy cannot be measured (no sudo)")
    args = ap.parse_args()

    run = Runner(args, plan_only=args.plan)
    meter = None
    if not args.plan:
        RunLock(OUT / ".lock")
        free_gb = shutil.disk_usage(paths.RESULTS.parent if not paths.RESULTS.exists() else paths.RESULTS).free / 1e9
        if free_gb < NEEDED_GB and not args.ignore_disk_check:
            sys.exit(f"Only {free_gb:.0f} GB free on this disk; these runs need about {NEEDED_GB} GB with a margin. "
                     "Free some space (or pass --ignore-disk-check).")
        meter = EnergyMeter("mps")
        if meter.available and not meter.counters_live and not args.allow_gpu_only_energy:
            sys.exit("CPU and DRAM energy counters are not live. Run `sudo -v` and start this script again "
                     "(or pass --allow-gpu-only-energy).")
        if meter.keeper_pid:
            os.environ[KEEPER_ENV] = str(meter.keeper_pid)  # child runs reuse this powermetrics

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    try:
        chosen = stage_a(run, summary) if "A" in args.stages else None
        if chosen is None and "stage_a" in summary:
            chosen = float(summary["stage_a"]["chosen"])
        if chosen is not None and not (run.plan_only and run.pending):
            if "B" in args.stages:
                stage_b(run, summary, chosen)
            if "C" in args.stages and not (run.plan_only and run.pending):
                stage_c(run, summary, chosen)
    finally:
        if meter:
            meter.close()

    if run.plan_only:
        print("Next runs:" if run.pending else "Nothing left to run.")
        for cmd in run.pending:
            print("  python", cmd)
        if run.pending:
            print("(Later runs depend on these results and are decided when they finish.)")
        return
    summary["updated"] = datetime.now().isoformat(timespec="seconds")
    write_summary(summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C between steps: nothing is in progress
        paused()
