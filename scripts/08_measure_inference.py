#!/usr/bin/env python3
"""Step 7: inference cost of a trained model, end to end, one image or question at a time (methodology §8).

    open scripts/inference.command                    # Terminal window: password once, then the reference measurement
    sudo -v && python scripts/08_measure_inference.py [--n 500] [--repeats 5]
    python scripts/08_measure_inference.py --n 8 --repeats 1 --warmup 2 --idle-seconds 5 --allow-gpu-only-energy  # smoke

What is measured, per unit (a precision and decoding setting, and a task: captions or VQA answers):
  - Energy per caption / per answer: SoC + DRAM energy from the IOReport counters over a pass of N items at batch 1,
    minus idle power measured for --idle-seconds just before the pass, divided by N. Also reported with idle
    included, and split into the encoder phase and the decoding phase.
  - Latency per item (median and 95th percentile), split the same way.
  - Peak memory: Metal driver memory sampled after every item, and the process's peak resident memory.
  - Model size per component, and GFLOPs per image / caption / answer (computed once, on the CPU, in fp32).

Protocol: the N test images (captions) and N test questions on N different images (answers) are a fixed random
sample (seed 0), saved in results/inference/subset_<split>_<n>.json. Their files are read once at the start so every
pass reads them from the OS file cache. Each pass is split in two phases measured separately: phase 1 reads and
encodes all N images, phase 2 generates the N captions or answers from those embeddings; per-item energy is the sum.
Each unit loads its model, runs --warmup items unmeasured, measures idle power, then runs its pass. Units are
repeated --repeats times, in a random order within each repetition. Captions and answers of the first repetition
are compared with scripts/05_evaluate.py's batched outputs for the same checkpoint (agreement rate).

Results: results/inference/<run>/<checkpoint>/measurements.jsonl (one line per unit and repetition) and summary.md.
Pausing: Ctrl+C stops after the current pass; finished passes are kept and skipped when the same command runs again.
"""
import argparse
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl.device import empty_cache, get_device  # noqa: E402  (first: sets MPS memory limits)

import torch  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.energy import KEEPER_ENV, EnergyMeter, awake_clock  # noqa: E402
from greenvl.inference import DECODING, count_flops, load_pipeline, model_size_mb  # noqa: E402
from greenvl.lock import RunLock  # noqa: E402

REFERENCE_RUN = "coco_ViT-L-14_lora8_lr1e-3_s0"
REFERENCE_CHECKPOINT = "epoch_02.pt"  # highest validation CIDEr of the 3-epoch reference run (results/reference/)


# ---------------------------------------------------------------- subset

def subset(split: str, n: int) -> dict:
    """Fixed random sample of N images (captions) and N questions on N distinct images (answers)."""
    path = paths.RESULTS / "inference" / f"subset_{split}_{n}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(0)
    images = json.loads((paths.PROCESSED / "coco_karpathy.json").read_text())[split]
    files = {r["image_id"]: r["file"] for r in images}
    cap = sorted(rng.sample(sorted(files), n))
    questions = json.loads((paths.PROCESSED / f"vqa_{split}_questions.json").read_text())["questions"]
    by_image = {}
    for q in sorted(questions, key=lambda q: q["question_id"]):
        by_image.setdefault(q["image_id"], []).append(q)
    vqa_images = sorted(rng.sample(sorted(by_image), n))
    vqa = [rng.choice(by_image[i]) for i in vqa_images]
    out = {"split": split, "n": n, "seed": 0,
           "caption": [{"image_id": i, "file": files[i]} for i in cap],
           "vqa": [{"question_id": q["question_id"], "image_id": q["image_id"], "file": files[q["image_id"]],
                    "question": q["question"]} for q in vqa]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def warm_file_cache(items: list[dict]) -> float:
    t0 = time.perf_counter()
    for it in items:
        (paths.COCO_IMAGES / it["file"]).read_bytes()
    return time.perf_counter() - t0


# ---------------------------------------------------------------- one pass

def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def metal_mb(dev) -> float | None:
    return torch.mps.driver_allocated_memory() / 2**20 if dev.type == "mps" else None


def run_pass(pipe, items, task, decoding, meter, idle_w):
    """Phase 1 encodes every item's image, phase 2 generates every caption or answer; both at batch 1."""
    dev = pipe.device
    peak = 0.0
    enc_t, dec_t, embs, outputs = [], [], [], []
    if meter.available:
        meter.begin("encode")
    t_phase = awake_clock()
    for it in items:
        t0 = time.perf_counter()
        embs.append(pipe.encode(paths.COCO_IMAGES / it["file"]))
        enc_t.append(time.perf_counter() - t0)
        peak = max(peak, metal_mb(dev) or 0.0)
    enc_s = awake_clock() - t_phase
    enc_e = meter.end("encode") if meter.available else None

    if meter.available:
        meter.begin("decode")
    t_phase = awake_clock()
    for it, emb in zip(items, embs):
        t0 = time.perf_counter()
        outputs.append(pipe.caption(emb, decoding) if task == "caption" else pipe.answer(emb, it["question"]))
        dec_t.append(time.perf_counter() - t0)
        peak = max(peak, metal_mb(dev) or 0.0)
    dec_s = awake_clock() - t_phase
    dec_e = meter.end("decode") if meter.available else None

    n = len(items)
    total = [a + b for a, b in zip(enc_t, dec_t)]

    def per_item(e, s):
        if not e or e.get("total_j") is None:
            return None, None
        above = (e["total_j"] - idle_w * s) / n if idle_w is not None else None
        return e["total_j"] / n, above

    enc_j, enc_above = per_item(enc_e, enc_s)
    dec_j, dec_above = per_item(dec_e, dec_s)
    ms = lambda xs, q: 1000 * percentile(xs, q)  # noqa: E731
    return {
        "n": n, "seconds": {"encode": enc_s, "decode": dec_s},
        "energy": {"encode": enc_e, "decode": dec_e},
        "j_per_item": None if enc_j is None else enc_j + dec_j,
        "j_per_item_above_idle": None if enc_above is None else enc_above + dec_above,
        "j_per_item_split": {"encode": enc_j, "decode": dec_j, "encode_above_idle": enc_above,
                             "decode_above_idle": dec_above},
        "latency_ms": {"median": ms(total, 0.5), "p95": ms(total, 0.95), "mean": 1000 * statistics.mean(total),
                       "encode_median": ms(enc_t, 0.5), "decode_median": ms(dec_t, 0.5)},
        "peak_metal_mb": peak if dev.type == "mps" else None,
        "outputs": outputs,
    }


# ---------------------------------------------------------------- reference outputs (agreement check)

def reference_outputs(run: str, ckpt: str, split: str, task: str, decoding: str) -> dict:
    stem = Path(ckpt).stem
    d = paths.RESULTS / "eval" / run
    if task == "caption":
        f = d / f"{stem}_{split}_{decoding}_captions.json"
        return {c["image_id"]: c["caption"] for c in json.loads(f.read_text())["captions"]} if f.exists() else {}
    f = d / f"{stem}_{split}_answers.json"
    return {a["question_id"]: a["answer"] for a in json.loads(f.read_text())["answers"]} if f.exists() else {}


def agreement(items, outputs, ref, task) -> float | None:
    key = "image_id" if task == "caption" else "question_id"
    pairs = [(o, ref[it[key]]) for it, o in zip(items, outputs) if it[key] in ref]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


# ---------------------------------------------------------------- environment record

def system_state() -> dict:
    state = {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.platform()}
    if sys.platform == "darwin":
        for key, cmd in (("power", ["pmset", "-g", "batt"]), ("power_settings", ["pmset", "-g"])):
            try:
                state[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
        settings = state.pop("power_settings", "")
        # macOS energy mode: "powermode" 0 automatic, 1 low power, 2 high power (older macOS: "lowpowermode" 0/1)
        state["energy_mode"] = next((ln.strip() for ln in settings.splitlines()
                                     if "powermode" in ln.lower()), None)
        state["on_ac_power"] = "AC Power" in state.get("power", "")
    return state


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=REFERENCE_RUN)
    ap.add_argument("--checkpoint", default=REFERENCE_CHECKPOINT)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tasks", nargs="+", default=["caption", "vqa"], choices=["caption", "vqa"])
    ap.add_argument("--decoding", nargs="+", default=["beam3"], choices=list(DECODING), help="caption decoding(s)")
    ap.add_argument("--precision", nargs="+", default=["fp32"])
    ap.add_argument("--n", type=int, default=500, help="items per pass")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10, help="unmeasured items before each pass")
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--no-flops", action="store_true")
    ap.add_argument("--label", default=None, help="results folder name (default: the checkpoint name); e.g. _smoke")
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-gpu-only-energy", action="store_true",
                    help="continue even if CPU and DRAM energy cannot be measured (no sudo)")
    args = ap.parse_args()

    dev = get_device(args.device)
    out_dir = paths.RESULTS / "inference" / args.run / (args.label or Path(args.checkpoint).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    RunLock(out_dir / ".lock")
    log_path = out_dir / "measurements.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            if r["n"] == args.n and r["split"] == args.split:
                done.add((r["precision"], r["task"], r["decoding"], r["repeat"]))

    meter = EnergyMeter(dev.type)
    if meter.available and not meter.counters_live and not args.allow_gpu_only_energy:
        sys.exit("CPU and DRAM energy counters are not live. Run `sudo -v` and start this script again "
                 "(or pass --allow-gpu-only-energy).")
    if meter.keeper_pid:
        os.environ[KEEPER_ENV] = str(meter.keeper_pid)

    sub = subset(args.split, args.n)
    print(f"subset: {args.n} images for captions, {args.n} questions; file cache warmed in "
          f"{warm_file_cache(sub['caption'] + sub['vqa']):.1f} s", flush=True)
    units = [(p, "caption", d) for p in args.precision for d in args.decoding if "caption" in args.tasks]
    units += [(p, "vqa", "greedy") for p in args.precision if "vqa" in args.tasks]
    system = system_state()
    if sys.platform == "darwin" and not system.get("on_ac_power"):
        print("WARNING: not on AC power; measurements should run on AC power.", flush=True)

    try:
        for rep in range(args.repeats):
            order = units[:]
            random.Random(1000 + rep).shuffle(order)
            for pos, (precision, task, decoding) in enumerate(order):
                if (precision, task, decoding, rep) in done:
                    continue
                items = sub[task]
                print(f"\n>>> repeat {rep + 1}/{args.repeats}: {precision} {task} {decoding} "
                      f"({datetime.now():%H:%M})", flush=True)
                pipe = load_pipeline(args.run, args.checkpoint, dev, precision)
                for it in items[: args.warmup]:  # unmeasured: compiles kernels, fills caches
                    emb = pipe.encode(paths.COCO_IMAGES / it["file"])
                    pipe.caption(emb, decoding) if task == "caption" else pipe.answer(emb, it["question"])
                idle = meter.measure_idle(args.idle_seconds) if meter.available and args.idle_seconds > 0 else None
                idle_w = idle["watts"] if idle else None
                r = run_pass(pipe, items, task, decoding, meter, idle_w)
                ref = reference_outputs(args.run, args.checkpoint, args.split, task, decoding)
                record = {"time": datetime.now().isoformat(timespec="seconds"), "run": args.run,
                          "checkpoint": args.checkpoint, "split": args.split, "precision": precision, "task": task,
                          "decoding": decoding, "repeat": rep, "position": pos, "n": args.n, "batch": 1,
                          "idle": idle, "idle_w": idle_w, "counters_live": meter.counters_live,
                          "agreement_with_eval": agreement(items, r["outputs"], ref, task),
                          "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
                          if sys.platform == "darwin" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10,
                          "model_size_mb": model_size_mb(pipe), "system": system,
                          **{k: v for k, v in r.items() if k != "outputs"}}
                if rep == 0:
                    record["outputs"] = r["outputs"]
                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                j = record["j_per_item_above_idle"]
                print(f"    {r['latency_ms']['median']:.0f} ms median, {r['latency_ms']['p95']:.0f} ms p95"
                      + (f", {j:.3f} J per {'caption' if task == 'caption' else 'answer'} above idle" if j else "")
                      + (f", agreement with evaluation {record['agreement_with_eval']:.1%}"
                         if record["agreement_with_eval"] is not None else ""), flush=True)
                del pipe
                empty_cache(dev)
    finally:
        meter.close()

    flops_path = out_dir / "flops.json"
    if not args.no_flops and not flops_path.exists():
        print("\ncounting FLOPs on the CPU (fp32, 3 images)", flush=True)
        pipe = load_pipeline(args.run, args.checkpoint, dev, "fp32")
        k = sub["vqa"][:3]
        flops = count_flops(pipe, [paths.COCO_IMAGES / it["file"] for it in k], [it["question"] for it in k],
                            "beam3")
        flops_path.write_text(json.dumps(flops, indent=1))
    write_summary(out_dir, args.n, args.split)


def write_summary(out_dir: Path, n: int, split: str):
    rows = {}
    for line in (out_dir / "measurements.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["n"] == n and r["split"] == split:
            rows.setdefault((r["precision"], r["task"], r["decoding"]), []).append(r)
    flops = json.loads((out_dir / "flops.json").read_text()) if (out_dir / "flops.json").exists() else {}

    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return "-"
        return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"

    lines = [f"# Inference cost, batch 1, {n} {split} items per pass (mean ± sd over repeats)\n",
             "| Precision | Task | Decoding | Repeats | J/item above idle | J/item incl. idle | encoder J | decoding J "
             "| Latency median ms | p95 ms | Peak Metal MB | Size MB | Agreement |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (p, task, d), rs in sorted(rows.items()):
        lat = [r["latency_ms"]["median"] for r in rs]
        p95 = [r["latency_ms"]["p95"] for r in rs]
        agree = next((r.get("agreement_with_eval") for r in rs if r["repeat"] == 0), None)
        lines.append(
            f"| {p} | {task} | {d} | {len(rs)} | {ms([r['j_per_item_above_idle'] for r in rs])} | "
            f"{ms([r['j_per_item'] for r in rs])} | {ms([r['j_per_item_split']['encode_above_idle'] for r in rs])} | "
            f"{ms([r['j_per_item_split']['decode_above_idle'] for r in rs])} | {statistics.mean(lat):.0f} | "
            f"{statistics.mean(p95):.0f} | {max(r['peak_metal_mb'] or 0 for r in rs):.0f} | "
            f"{rs[0]['model_size_mb']['total']:.0f} | {'-' if agree is None else f'{agree:.1%}'} |")
    if flops:
        lines.append(f"\nGFLOPs (fp32, CPU count over {flops['images']} images): encoder {flops['encoder_gflops']:.1f} "
                     f"per image; caption decoding ({flops['decoding']}) {flops['caption_decode_gflops']:.2f}; "
                     f"answer decoding {flops['answer_decode_gflops']:.2f}.")
    lines.append("\nEncoder and decoding columns are per item, above idle. Agreement: share of first-repeat outputs "
                 "identical to the batched evaluation's (scripts/05_evaluate.py).")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPaused: finished passes are saved. Run the same command to continue.", flush=True)
        sys.exit(130)
