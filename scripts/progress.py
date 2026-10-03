#!/usr/bin/env python3
"""Progress of the learning-rate selection (04_select_lr.py): what is running, how much is done and left, and an
estimate of the time to finish. Reads files only (no torch), so it can run at any time, alongside training.

    python scripts/progress.py

Runs not decided yet (which two rates are extended, the Stage B and C rates) are counted with the sizes the rules
fix. Speeds are measured from the logs where a run of that kind exists, otherwise estimated (see DEFAULTS).
"""
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl import paths  # noqa: E402

_spec = importlib.util.spec_from_file_location("select_lr", Path(__file__).with_name("04_select_lr.py"))
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

BATCH = 64
SHORT, LONG = 3, 6
# Seconds per step, per validation, and per process start (loading plus the 60 s idle window) until measured.
DEFAULTS = {
    "f8k_lora": {"step": 0.46, "val": 14, "start": 90},   # measured on f8k_lr1e-4, 29 Sep 2026
    "f8k_full": {"step": 0.70, "val": 14, "start": 90},   # guess: full fine-tuning ~1.5x LoRA
    "coco_lora": {"step": 0.45, "val": 150, "start": 180},  # guess until the first pilot logs steps
}
IMAGES_PER_S = 107.0  # ViT-B/32 feature caching, Flickr8k, 29 Sep 2026
FEATURE_SETS = {"coco_train": 113_287, "coco_val": 5_000, "coco_test": 5_000}
STALE_S = 300  # no new log line for this long while "running" is worth a mention


def log_entries(name):
    path = paths.RESULTS / "runs" / name / "log.jsonl"
    if not path.exists():
        return [], None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()], path.stat().st_mtime


def steps_done(entries):
    return max((x["step"] for x in entries if x["type"] in ("step", "epoch", "partial")), default=0)


def measured(kind, names):
    """Seconds per step and per validation from the runs of one kind, where logged."""
    step, val = [], []
    for name in names:
        entries, _ = log_entries(name)
        step += [BATCH / x["samples_per_s"] for x in entries if x["type"] == "step" and x.get("samples_per_s")][-20:]
        val += [x["val_seconds"] for x in entries if x["type"] in ("epoch", "partial") and x.get("val_seconds")]
    d = dict(DEFAULTS[kind])
    if step:
        d["step"], d["step_measured"] = statistics.median(step), True
    if val:
        d["val"] = statistics.median(val)
    return d


def run_item(name, kind, target_steps, validations, label=None):
    entries, mtime = log_entries(name)
    done = steps_done(entries)
    vals_done = sum(x["type"] in ("epoch", "partial") for x in entries)
    return {"label": label or name, "run": name, "kind": kind, "target": target_steps, "done": min(done, target_steps),
            "val_left": max(0, validations - vals_done), "mtime": mtime, "entries": entries, "count": 1}


def placeholder(label, kind, target_steps, validations, count):
    return {"label": label, "run": None, "kind": kind, "target": target_steps, "done": 0, "val_left": validations,
            "mtime": None, "entries": [], "count": count}


def plan():
    """Items in the order the driver runs them, following its rules on the results so far."""
    cfg = paths.RESULTS / "runs" / sel.name_a(1e-4) / "config.json"
    spe = json.loads(cfg.read_text())["steps_per_epoch"] if cfg.exists() else 30_000 // BATCH
    items, chosen = [], None
    rates = list(sel.STAGE_A_RATES)
    states = {lr: sel.read_run(sel.name_a(lr)) for lr in rates}
    short_done = lambda lr: len(states[lr]["epochs"]) >= SHORT or sel.nonfinite(states[lr])  # noqa: E731
    top = None
    if all(short_done(lr) for lr in rates):
        at_short = lambda lr: sel.curve(states[lr])[SHORT - 1] if len(sel.curve(states[lr])) >= SHORT else 1e9  # noqa
        ranked = sorted((lr for lr in rates if not sel.unstable(states[lr])), key=at_short)
        if ranked and ranked[0] in (min(rates), max(rates)):
            i = sel.GRID.index(ranked[0]) + (1 if ranked[0] == max(rates) else -1)
            if 0 <= i < len(sel.GRID):
                rates.append(sel.GRID[i])
                states[sel.GRID[i]] = sel.read_run(sel.name_a(sel.GRID[i]))
                if short_done(sel.GRID[i]):
                    ranked = sorted((lr for lr in rates if not sel.unstable(states[lr])), key=at_short)
                else:
                    ranked = None
        if ranked:
            top = ranked[:2]
    for lr in rates:
        extended = top is not None and lr in top
        epochs = LONG if extended else SHORT
        label = f"A  {sel.name_a(lr)}" + (f" (extended to {LONG} epochs)" if extended else "")
        items.append(run_item(sel.name_a(lr), "f8k_lora", epochs * spe, epochs, label))
    if top is None:
        items.append(placeholder(f"A  best two rates extended to {LONG} epochs", "f8k_lora", 2 * (LONG - SHORT) * spe,
                                 2 * (LONG - SHORT), 2))
    elif all(len(sel.read_run(sel.name_a(lr))["epochs"]) >= LONG for lr in top):
        chosen = min(top, key=lambda lr: min(sel.curve(sel.read_run(sel.name_a(lr)))))

    if chosen is not None:
        for lr in (chosen, sel.lower(chosen)):
            items.append(run_item(sel.name_b(lr), "f8k_full", SHORT * spe, SHORT, f"B  {sel.name_b(lr)}"))
    else:
        items.append(placeholder("B  full fine-tuning at 2 rates", "f8k_full", 2 * SHORT * spe, 2 * SHORT, 2))

    folder = paths.FEATURES / sel.ENCODER.replace("/", "-")
    missing = [s for s in FEATURE_SETS if not (folder / f"{s}.pt").exists()]
    if missing:
        done = 0
        for s in missing:
            parts = sorted((folder / f"{s}.parts").glob("*.pt"))
            done += min(len(parts) * 8192, FEATURE_SETS[s])
        mtimes = [p.stat().st_mtime for s in missing for p in (folder / f"{s}.parts").glob("*.pt")]
        items.append({"label": "C  cache COCO ViT-B/32 features", "run": None, "kind": "features",
                      "target": sum(FEATURE_SETS[s] for s in missing), "done": done, "val_left": 0,
                      "mtime": max(mtimes) if mtimes else None, "entries": [], "count": 1})

    pilot = sel.pilot_steps(BATCH)
    if chosen is not None:
        for lr in (chosen, sel.lower(chosen)):
            items.append(run_item(sel.name_c(lr), "coco_lora", pilot, 1, f"C  {sel.name_c(lr)} ({pilot:,}-step pilot)"))
    else:
        items.append(placeholder(f"C  COCO + VQA pilots at 2 rates ({pilot:,} steps each)", "coco_lora", 2 * pilot, 2, 2))
    return items


def seconds_left(item, speed, active):
    if item["kind"] == "features":
        left = (item["target"] - item["done"]) / IMAGES_PER_S
        return left + (0 if active else DEFAULTS["f8k_lora"]["start"])
    s = speed[item["kind"]]
    left = (item["target"] - item["done"]) * s["step"] + item["val_left"] * s["val"]
    if left > 0 and not active:
        left += s["start"] * item["count"]
    return left


def fmt_duration(sec):
    m = round(sec / 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def driver_status():
    lock = sel.OUT / ".lock"
    if not lock.exists():
        return False, None
    try:
        pid = int(lock.read_text().strip() or 0)
    except (OSError, ValueError):
        return False, None
    from greenvl.lock import _alive
    return bool(pid) and _alive(pid), pid


def main():
    items = plan()
    names = {k: [i["run"] for i in items if i["kind"] == k and i["run"]] for k in DEFAULTS}
    speed = {k: measured(k, names[k]) for k in DEFAULTS}
    running, pid = driver_status()
    now = time.time()

    # The active item: the most recently written log (or feature shard) that is not complete.
    open_items = [i for i in items if i["mtime"] and (i["done"] < i["target"] or i["val_left"])]
    active = max(open_items, key=lambda i: i["mtime"]) if running and open_items else None
    if running and active is None:
        active = next((i for i in items if i["done"] < i["target"] or i["val_left"]), None)

    total = sum(i["target"] for i in items if i["kind"] != "features")
    done = sum(i["done"] for i in items if i["kind"] != "features")
    runs_total = sum(i["count"] for i in items if i["kind"] != "features")
    runs_done = sum(1 for i in items if i["run"] and i["kind"] != "features" and i["done"] >= i["target"]
                    and not i["val_left"])
    left = sum(seconds_left(i, speed, i is active) for i in items)

    state = f"running (process {pid})" if running else "not running (paused or finished)"
    print(f"Learning-rate selection: {state}")
    if active is not None:
        a = active
        if a["kind"] == "features":
            print(f"Now:  {a['label']}: {a['done']:,} of {a['target']:,} images saved "
                  f"({100 * a['done'] / a['target']:.0f}%), about {fmt_duration(seconds_left(a, speed, True))} left")
        else:
            steps = [x for x in a["entries"] if x["type"] == "step"]
            rate = f", {steps[-1]['samples_per_s']:.0f} samples/s" if steps else ""
            epoch = f"epoch {steps[-1]['epoch'] + 1}, " if steps else ""
            print(f"Now:  {a['label']}: {epoch}step {a['done']:,} of {a['target']:,} "
                  f"({100 * a['done'] / a['target']:.0f}%){rate}; this run needs about "
                  f"{fmt_duration(seconds_left(a, speed, True))} more")
        if a["mtime"] and now - a["mtime"] > STALE_S:
            print(f"      (no new log line for {fmt_duration(now - a['mtime'])}: validating, loading, or stalled)")
    feats = next((i for i in items if i["kind"] == "features"), None)
    print(f"Done: {done:,} of {total:,} training steps ({100 * done / total:.0f}%); {runs_done} of {runs_total} runs "
          "complete; " + (f"COCO features {feats['done']:,} of {feats['target']:,} images" if feats
                          else "COCO features cached"))
    if left > 0:
        finish = datetime.now() + timedelta(seconds=left)
        guessed = [k for k, v in speed.items() if not v.get("step_measured") and any(
            i["kind"] == k and i["done"] < i["target"] for i in items)]
        print(f"Left: about {fmt_duration(left)}" + (f", finishing around {finish:%H:%M}" if running else
                                                     " of running time")
              + (f" (speed of {', '.join(guessed)} estimated until measured)" if guessed else ""))
    else:
        print("Left: nothing; see results/lr_selection/summary.md")
    print("Runs:")
    for i in items:
        if i["kind"] == "features":
            continue
        mark = "done" if i["run"] and i["done"] >= i["target"] and not i["val_left"] else (
            "now " if i is active else "    ")
        vals = [x.get("val_loss_caption") for x in i["entries"] if x["type"] == "epoch"]
        tail = f"  val {' '.join(f'{v:.3f}' for v in vals)}" if vals else ""
        print(f"  {mark} {i['label']}: {i['done']:,}/{i['target']:,} steps{tail}")


if __name__ == "__main__":
    main()
