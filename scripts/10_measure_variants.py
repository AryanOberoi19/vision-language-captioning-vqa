#!/usr/bin/env python3
"""Step 8b: inference cost of every configuration, measured component by component (methodology §8; design agreed
with Aryan on 3 Oct: 3 repetitions, more can be added later).

    open scripts/variants_energy.command              # Terminal window: password once, then ~2 h
    sudo -v && python scripts/10_measure_variants.py [--repeats 3]
    python scripts/10_measure_variants.py --label epoch_02_t4        # second platform: colab/colab_t4.ipynb
    python scripts/10_measure_variants.py --summary-only
    python scripts/10_measure_variants.py --n 8 --repeats 1 --pause 1 --idle-seconds 5 --warmup 2 \\
        --label _smoke --allow-gpu-only-energy        # code check

Configurations: greenvl/precision.py (FP32 reference; FP16, INT8, NF4 on the encoder, the decoder or both) with
captions at beam 3, plus captions with greedy and beam 5 decoding at FP32. Items: the step-7 subset, N Karpathy test
images for captions and N test questions on N other images (results/inference/subset_test_<N>.json).

Windows, 26 per repetition, in a random order (seed 1000 + repetition):
  - 4 encoder windows: the encoder at FP32, FP16, INT8 and NF4 encodes all 2N subset images, one at a time from
    the image file (read, preprocess, encode). The encoder does the same work for either task (step 7: 0.957 J
    per image in the caption passes, 0.955 J in the answer passes), so one window per precision serves both.
  - 22 decoding windows: each configuration's decoder (mapping network + GPT-2 + adapters at its precision)
    generates the N captions and the N answers, one at a time, from the embeddings its own encoder precision
    produces; plus FP32 captions with greedy and with beam 5.
  An idle window (--idle-seconds, re-measured when disturbed) opens each repetition. Before every window the model
  runs --warmup items unmeasured, then the Mac rests for --pause seconds (cool-down, unmeasured).
Per configuration and repetition: energy per caption (answer) = its encoder window's J per image + its decoding
window's J per item. Step 7 measured the same two phases as separate windows and added them the same way. Latency
per item = that item's encoding time + its decoding time. Above idle subtracts the repetition's idle power over
each window's duration. Every repetition contains FP32, so each configuration is also reported relative to the
FP32 of its own repetition (robust to differences between sessions or days).

All 8 model parts (4 encoders, 4 decoders) stay loaded for the session. The embeddings of the 2N images at each
encoder precision are computed once (unmeasured, batch 1) and saved; the decoding windows use them.
Memory with a configuration's encoder and decoder both loaded is measured once per configuration in a fresh process
(--memory CONFIG: 20 captions and 20 answers), before the parts are loaded: GPU memory held by tensors, the Metal
driver's total, and the process's resident memory.

Results: results/variants/<run>/<checkpoint>/ (energy.jsonl: one line per window and per idle window; memory.json;
energy.md; summary.md adds the accuracy of 09_evaluate_variants.py). Resuming: windows already in energy.jsonl are
skipped and a resumed repetition gets a new idle window. --repeats 5 later adds repetitions 4 and 5.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl.device import empty_cache, get_device, synchronize  # noqa: E402  (first: sets MPS memory limits)

import torch  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.data import load_tokenizer  # noqa: E402
from greenvl.decode import answer_questions, generate_captions  # noqa: E402
from greenvl.energy import KEEPER_ENV, EnergyMeter, awake_clock  # noqa: E402
from greenvl.inference import component_sizes, encode_image, load_pipeline  # noqa: E402
from greenvl.lock import RunLock  # noqa: E402
from greenvl.measure import (CO2_G_PER_KWH, agreement, mean_sd, measure_idle_checked, metal_mb,  # noqa: E402
                             peak_rss_mb, percentile, reference_outputs, subset, system_state, tensor_mb,
                             top_processes, warm_file_cache)
from greenvl.precision import CONFIGS, PRECISIONS, REFERENCE, load_decoder, load_encoder  # noqa: E402

RUN = "coco_ViT-L-14_lora8_lr1e-3_s0"
CHECKPOINT = "epoch_02.pt"
SPLIT = "test"
MEMORY_ITEMS = 20
WINDOW_ATTEMPTS = 3  # a window whose counters read impossibly (suspect_reading) is measured again


def windows() -> list[dict]:
    w = [{"kind": "encode", "precision": p} for p in PRECISIONS]
    for c in CONFIGS:
        w.append({"kind": "decode", "config": c, "task": "caption", "decoding": "beam3"})
        w.append({"kind": "decode", "config": c, "task": "vqa", "decoding": "greedy"})
    w += [{"kind": "decode", "config": REFERENCE, "task": "caption", "decoding": d} for d in ("greedy", "beam5")]
    for x in w:
        x["key"] = (f"encode:{x['precision']}" if x["kind"] == "encode"
                    else f"decode:{x['config']}:{x['task']}:{x['decoding']}")
    return w


# ---------------------------------------------------------------- one window

def run_encode(encoder, processor, precision, items, dev):
    lat = []
    for it in items:
        t0 = time.perf_counter()
        encode_image(encoder, processor, paths.COCO_IMAGES / it["file"], dev, precision)
        lat.append(time.perf_counter() - t0)
    return lat, None


def run_decode(model, tok, task, decoding, items, embs, dev):
    lat, outputs = [], []
    with torch.inference_mode():
        for it, emb in zip(items, embs):
            t0 = time.perf_counter()
            if task == "caption":
                out = generate_captions(model, emb[None], decoding, batch_size=1, tokenizer=tok)[0]
            else:
                out = answer_questions(model, emb[None], [it["question"]], batch_size=1, tokenizer=tok)[0]
            synchronize(dev)
            lat.append(time.perf_counter() - t0)
            outputs.append(out)
    return lat, outputs


def suspect_reading(energy, seconds, meter) -> str | None:
    """A reading that cannot be right: the GPU counter at exactly 0 J over several seconds of model work (seen once
    on 3 Oct: every counter read 0 for a 10 s NF4 extraction window; repeats read normally), or CPU / DRAM at 0 J
    while powermetrics keeps those counters live."""
    if not energy or seconds < 5:
        return None
    if energy.get("gpu_j") == 0:
        return "GPU counter read 0 J"
    if meter.counters_live and (energy.get("cpu_j") == 0 or energy.get("dram_j") == 0):
        return "CPU or DRAM counter read 0 J"
    return None


def measured(meter, fn):
    """Run fn() inside one energy window; (fn's result, energy dict or None, awake seconds)."""
    if meter.available:
        meter.begin("window")
    t0 = awake_clock()
    result = fn()
    seconds = awake_clock() - t0
    energy = meter.end("window") if meter.available else None
    return result, energy, seconds


# ---------------------------------------------------------------- memory check (fresh process per configuration)

def memory_check(args, dev, sub):
    """Memory of one configuration as deployed: encoder and decoder loaded together, then 20 captions and 20
    answers. The conversion to reduced precision happens on the CPU and leaves freed blocks behind (CPU heap, MPS
    allocator cache) that a deployment loading an already-converted model would not have, so the cache is emptied
    after loading and the process's current memory, not its lifetime peak, is reported."""
    import gc

    import psutil

    pipe = load_pipeline(args.run, args.checkpoint, dev, args.memory)
    gc.collect()
    empty_cache(dev)
    tensors = lambda: tensor_mb(dev) or 0.0  # noqa: E731
    loaded = {"tensor_mb": tensors(), "metal_mb": metal_mb(dev) or 0.0}
    peak = dict(loaded)
    k = min(MEMORY_ITEMS, args.n)
    for task in ("caption", "vqa"):
        for it in sub[task][:k]:
            emb = pipe.encode(paths.COCO_IMAGES / it["file"])
            pipe.caption(emb, "beam3") if task == "caption" else pipe.answer(emb, it["question"])
            peak["tensor_mb"] = max(peak["tensor_mb"], tensors())
            peak["metal_mb"] = max(peak["metal_mb"], metal_mb(dev) or 0.0)
    gpu = dev.type in ("mps", "cuda")
    out = {"config": args.memory, "items": 2 * k, "loaded": loaded, "device": str(dev),
           "peak_tensor_mb": peak["tensor_mb"] if gpu else None,
           "peak_metal_mb": peak["metal_mb"] if gpu else None,
           "rss_mb": psutil.Process().memory_info().rss / 2**20, "lifetime_peak_rss_mb": peak_rss_mb(),
           "size_mb": component_sizes(pipe.encoder, pipe.model)}
    print("MEMORY " + json.dumps(out), flush=True)


def memory_checks(args, out_dir: Path):
    path = out_dir / "memory.json"
    mem = json.loads(path.read_text()) if path.exists() else {}
    for c in CONFIGS:
        if c in mem:
            continue
        print(f"memory check: {c}", flush=True)
        cmd = [sys.executable, __file__, "--memory", c, "--run", args.run, "--checkpoint", args.checkpoint,
               "--n", str(args.n)] + (["--device", args.device] if args.device else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("MEMORY ")), None)
        if r.returncode != 0 or line is None:
            sys.exit(f"memory check for {c} failed:\n{r.stderr[-2000:]}")
        mem[c] = json.loads(line[len("MEMORY "):])
        path.write_text(json.dumps(mem, indent=1))
    return mem


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--n", type=int, default=500, help="captions and answers per window (images: 2N)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5, help="unmeasured items before each window")
    ap.add_argument("--pause", type=float, default=20, help="unmeasured rest before each window (s)")
    ap.add_argument("--idle-seconds", type=float, default=60, help="idle window at the start of each repetition")
    ap.add_argument("--label", default=None, help="results folder name (default: the checkpoint name)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--memory", default=None, choices=list(CONFIGS), help=argparse.SUPPRESS)
    ap.add_argument("--allow-gpu-only-energy", action="store_true",
                    help="continue even if CPU and DRAM energy cannot be measured (no sudo)")
    args = ap.parse_args()
    out_dir = paths.RESULTS / "variants" / args.run / (args.label or Path(args.checkpoint).stem)
    if args.summary_only:
        return write_summary(out_dir, args)
    dev = get_device(args.device)
    sub = subset(SPLIT, args.n)
    if args.memory:
        return memory_check(args, dev, sub)

    out_dir.mkdir(parents=True, exist_ok=True)
    RunLock(out_dir / ".lock_energy")
    log_path = out_dir / "energy.jsonl"
    records = [json.loads(ln) for ln in log_path.read_text().splitlines()] if log_path.exists() else []
    done = {(r["repeat"], r["key"]) for r in records
            if r["type"] == "window" and r["subset_n"] == args.n and not r.get("suspect")}

    meter = EnergyMeter(dev.type)
    if meter.available and not meter.counters_live and not args.allow_gpu_only_energy:
        sys.exit("CPU and DRAM energy counters are not live. Run `sudo -v` and start this script again "
                 "(or pass --allow-gpu-only-energy).")
    if meter.keeper_pid:
        os.environ[KEEPER_ENV] = str(meter.keeper_pid)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    system = system_state()
    if sys.platform == "darwin" and not system.get("on_ac_power"):
        print("WARNING: not on AC power; measurements should run on AC power.", flush=True)

    def log(rec):
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    try:
        caption_items, vqa_items = sub["caption"], sub["vqa"]
        images = caption_items + vqa_items  # encoder windows: captions' images first, then the questions' images
        print(f"subset: {args.n} images for captions, {args.n} questions; file cache warmed in "
              f"{warm_file_cache(images):.1f} s", flush=True)
        memory_checks(args, out_dir)

        print("loading the 4 encoders and 4 decoders", flush=True)
        decoders, encoders, quantized = {}, {}, {}
        for p in PRECISIONS:
            decoders[p], cfg, _, quantized[f"decoder_{p}"] = load_decoder(args.run, args.checkpoint, p, dev)
        for p in PRECISIONS:
            enc, proc, quantized[f"encoder_{p}"] = load_encoder(cfg["encoder"], p, dev)
            encoders[p] = (enc, proc)
        tok = load_tokenizer(cfg["decoder"])
        sizes = {"encoder": {p: component_sizes(encoders[p][0], None)["encoder"] for p in PRECISIONS},
                 "mapper": {p: component_sizes(None, decoders[p])["mapper"] for p in PRECISIONS},
                 "decoder": {p: component_sizes(None, decoders[p])["decoder"] for p in PRECISIONS}}
        log({"type": "session", "session": session, "time": datetime.now().isoformat(timespec="seconds"),
             "run": args.run, "checkpoint": args.checkpoint, "split": SPLIT, "n": args.n, "repeats": args.repeats,
             "warmup": args.warmup, "pause": args.pause, "idle_seconds": args.idle_seconds, "sizes_mb": sizes,
             "quantized_layers": quantized, "counters_live": meter.counters_live, "energy_backend": meter.backend,
             "device": str(dev), "system": system})

        emb_dir = out_dir / "embeddings"
        emb_dir.mkdir(exist_ok=True)
        embs = {}
        for p in PRECISIONS:  # unmeasured; also warms every encoder
            f = emb_dir / f"{p}_n{args.n}.pt"
            if f.exists():
                embs[p] = torch.load(f)
            else:
                print(f"embeddings of the {len(images)} images at {p} (unmeasured)", flush=True)
                e = torch.stack([encode_image(*encoders[p], paths.COCO_IMAGES / it["file"], dev, p).cpu()
                                 for it in images])
                torch.save(e, f)
                embs[p] = e

        idle_history = []
        for rep in range(args.repeats):
            order = windows()
            random.Random(1000 + rep).shuffle(order)
            if all((rep, w["key"]) in done for w in order):
                continue
            print(f"\n=== repetition {rep + 1}/{args.repeats} ({datetime.now():%H:%M})", flush=True)
            idle = measure_idle_checked(meter, args.idle_seconds, idle_history)
            idle_w = idle["watts"] if idle else None
            log({"type": "idle", "session": session, "repeat": rep, "idle": idle,
                 "time": datetime.now().isoformat(timespec="seconds")})
            if idle:
                print(f"idle {idle_w:.3f} W", flush=True)
            for pos, w in enumerate(order):
                if (rep, w["key"]) in done:
                    continue
                if w["kind"] == "encode":
                    p = w["precision"]
                    enc, proc = encoders[p]
                    run_encode(enc, proc, p, images[: args.warmup], dev)
                    work = lambda: run_encode(enc, proc, p, images, dev)  # noqa: E731
                    n = len(images)
                else:
                    enc_p, dec_p = CONFIGS[w["config"]]
                    items = caption_items if w["task"] == "caption" else vqa_items
                    e = embs[enc_p][: args.n] if w["task"] == "caption" else embs[enc_p][args.n:]
                    e = e.to(dev, non_blocking=False)
                    model = decoders[dec_p]
                    run_decode(model, tok, w["task"], w["decoding"], items[: args.warmup], e[: args.warmup], dev)
                    work = lambda: run_decode(model, tok, w["task"], w["decoding"], items, e, dev)  # noqa: E731
                    n = len(items)
                suspects = []
                for attempt in range(1, WINDOW_ATTEMPTS + 1):
                    time.sleep(args.pause)
                    (lat, outputs), energy, seconds = measured(meter, work)
                    reason = suspect_reading(energy, seconds, meter)
                    if not reason:
                        break
                    suspects.append(reason)
                    print(f"    {w['key']}: {reason}; "
                          + ("keeping this reading" if attempt == WINDOW_ATTEMPTS else "measuring the window again"),
                          flush=True)
                total = energy["total_j"] if energy and energy.get("total_j") is not None else None
                rec = {"type": "window", "session": session, "time": datetime.now().isoformat(timespec="seconds"),
                       "repeat": rep, "position": pos, **w, "subset_n": args.n, "n": n, "seconds": seconds,
                       "energy": energy,
                       "idle_w": idle_w, "j_per_item": total / n if total is not None else None,
                       "j_per_item_above_idle": (total - idle_w * seconds) / n
                       if total is not None and idle_w is not None else None,
                       "latency_ms": [round(1000 * x, 3) for x in lat],
                       "attempts": attempt, "suspect_readings": suspects, "suspect": bool(reason),
                       "background_after": top_processes()}
                if outputs is not None and rep == 0:
                    rec["outputs"] = outputs
                log(rec)
                if not rec["suspect"]:  # a window still suspect after every attempt is measured again on resume
                    done.add((rep, w["key"]))
                j = rec["j_per_item_above_idle"]
                print(f"  [{pos + 1:2d}/{len(order)}] {w['key']:<34} {statistics.median(lat) * 1000:6.0f} ms median"
                      + (f", {j:.3f} J/item above idle" if j is not None else ""), flush=True)
    finally:
        meter.close()
    write_summary(out_dir, args)


# ---------------------------------------------------------------- summary

def write_summary(out_dir: Path, args):
    log_path = out_dir / "energy.jsonl"
    if not log_path.exists():
        print(f"no measurements in {out_dir}")
        return
    recs = [json.loads(ln) for ln in log_path.read_text().splitlines()]
    sessions = [r for r in recs if r["type"] == "session"]
    nvml = bool(sessions) and sessions[-1].get("energy_backend") == "nvml"  # a cloud GPU: board energy only
    win = {}
    for r in recs:
        if r["type"] == "window" and r["subset_n"] == args.n and not r.get("suspect"):
            win[(r["repeat"], r["key"])] = r  # a re-measured window replaces the earlier one
    mem_path = out_dir / "memory.json"
    mem = json.loads(mem_path.read_text()) if mem_path.exists() else {}
    sizes = sessions[-1]["sizes_mb"] if sessions else None
    reps = sorted({k[0] for k in win})
    n = args.n

    rows = []
    for w in windows():
        if w["kind"] != "decode":
            continue
        enc_p, dec_p = CONFIGS[w["config"]]
        offset = 0 if w["task"] == "caption" else n
        per_rep = []
        for rep in reps:
            d, e = win.get((rep, w["key"])), win.get((rep, f"encode:{enc_p}"))
            ref_d = win.get((rep, f"decode:{REFERENCE}:{w['task']}:beam3" if w["task"] == "caption"
                             else f"decode:{REFERENCE}:vqa:greedy"))
            ref_e = win.get((rep, f"encode:{CONFIGS[REFERENCE][0]}"))
            if not d or not e:
                continue
            x = {"repeat": rep}
            if d["j_per_item"] is not None and e["j_per_item"] is not None:
                x["j"] = e["j_per_item"] + d["j_per_item"]
                if ref_d and ref_e and ref_d["j_per_item"] is not None:
                    x["rel_fp32"] = x["j"] / (ref_e["j_per_item"] + ref_d["j_per_item"])
                for c in ("cpu_j", "gpu_j", "dram_j"):
                    x[c] = (e["energy"].get(c) or 0.0) / e["n"] + (d["energy"].get(c) or 0.0) / d["n"]
            if d["j_per_item_above_idle"] is not None and e["j_per_item_above_idle"] is not None:
                x["j_above"] = e["j_per_item_above_idle"] + d["j_per_item_above_idle"]
                x["enc_above"], x["dec_above"] = e["j_per_item_above_idle"], d["j_per_item_above_idle"]
            item_lat = [a + b for a, b in zip(e["latency_ms"][offset: offset + n], d["latency_ms"])]
            x["lat_median"], x["lat_p95"] = percentile(item_lat, 0.5), percentile(item_lat, 0.95)
            x["enc_median"], x["dec_median"] = percentile(e["latency_ms"][offset: offset + n], 0.5), \
                percentile(d["latency_ms"], 0.5)
            per_rep.append(x)
        if not per_rep:
            continue
        first = win.get((0, w["key"]))
        agree = None
        if first and first.get("outputs"):
            items = subset(SPLIT, n)[w["task"]]
            ref = reference_outputs(args.run, args.checkpoint, SPLIT, w["task"], w["decoding"], w["config"])
            agree = agreement(items, first["outputs"], ref, w["task"])
        size = (sizes["encoder"][enc_p] + sizes["mapper"][dec_p] + sizes["decoder"][dec_p]) if sizes else None
        measured_setting = w["task"] == "vqa" or w["decoding"] == "beam3"  # what the memory check ran
        m = (mem.get(w["config"]) or {}) if measured_setting else {}
        row = {"config": w["config"], "task": w["task"], "decoding": w["decoding"], "repeats": len(per_rep),
               "size_mb": size, "agreement_b1_vs_b64": agree, "memory_mb": m.get("peak_tensor_mb"),
               "metal_driver_mb": m.get("peak_metal_mb")}
        for k in ("j", "j_above", "rel_fp32", "enc_above", "dec_above", "cpu_j", "gpu_j", "dram_j", "lat_median",
                  "lat_p95", "enc_median", "dec_median"):
            row[k] = mean_sd([x.get(k) for x in per_rep])
        if row["j"][0] is not None and not nvml:  # the grid factor is India's; a cloud GPU's location is unknown
            row["g_co2e_per_1000"] = row["j"][0] * 1000 / 3.6e6 * CO2_G_PER_KWH
        rows.append(row)
    (out_dir / "energy.json").write_text(json.dumps({"n": n, "repetitions": reps, "rows": rows,
                                                     "time": datetime.now().isoformat(timespec="seconds")}, indent=1))

    def ms(t, d=3):
        m, s = t
        return "-" if m is None else (f"{m:.{d}f} ± {s:.{d}f}" if s is not None else f"{m:.{d}f}")

    def f0(x):
        return "-" if x is None else f"{x:.0f}"

    def pct(x):
        return "-" if x is None else f"{x:.1%}"
    lines = [f"# Inference cost per configuration, batch 1, {n} {SPLIT} items per task, "
             f"{len(reps)} repetition{'s' if len(reps) != 1 else ''} (mean ± sd)\n",
             "| Configuration | Task | Decoding | J/item above idle | J/item incl. idle | vs FP32 | encoder J | "
             "decoding J | Latency median ms | p95 ms | Size MiB | Memory MB | Same output as batch 64 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['task']} | {r['decoding']} | {ms(r['j_above'])} | {ms(r['j'])} | "
                     f"{ms(r['rel_fp32'], 2)} | {ms(r['enc_above'])} | {ms(r['dec_above'])} | "
                     f"{f0(r['lat_median'][0])} | {f0(r['lat_p95'][0])} | {f0(r['size_mb'])} | "
                     f"{f0(r['memory_mb'])} | {pct(r['agreement_b1_vs_b64'])} |")
    if not nvml:  # NVML reads the GPU board alone, so there is nothing to split
        lines += ["\n## Energy by component (J per item incl. idle, mean over repetitions)\n",
                  "| Configuration | Task | Decoding | CPU | GPU | DRAM | g CO2e per 1,000 items |",
                  "|---|---|---|---|---|---|---|"]
    for r in rows:
        if nvml or r["cpu_j"][0] is None:
            continue
        tot = r["cpu_j"][0] + r["gpu_j"][0] + r["dram_j"][0]
        lines.append(f"| {r['config']} | {r['task']} | {r['decoding']} | "
                     + " | ".join(f"{r[c][0]:.3f} ({r[c][0] / tot:.0%})" for c in ("cpu_j", "gpu_j", "dram_j"))
                     + f" | {r.get('g_co2e_per_1000', 0):.2f} |")
    lines.append("\nPer item = the configuration's encoder window (J per image) + its decoding window (J per item), "
                 "same repetition. vs FP32: ratio to the FP32 reference of the same repetition and task. Encoder and "
                 "decoding columns are above idle. Size: parameters and buffers as stored, quantization scales "
                 "included. Memory: GPU memory held by tensors (weights, caches, activations) with encoder and decoder "
                 "loaded together, highest value after each of 20 captions and 20 answers, fresh process; "
                 + ("the CUDA allocator's reserved total" if nvml else "the Metal driver's total (allocated in large chunks)")
                 + " is in memory.json. Same output as batch 64: share of first-repetition outputs identical to "
                 "05_evaluate.py's (batch 64) for the same configuration. "
                 + ("Energy covers the GPU board only (NVML; no CPU or DRAM counters on this platform); no CO2e is "
                    "given, since the data centre's grid intensity and PUE are unknown."
                    if nvml else f"Energy covers the SoC (CPU, GPU) and DRAM; CO2e uses J incl. idle at "
                    f"{CO2_G_PER_KWH} g/kWh, PUE 1."))
    (out_dir / "energy.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    joined_summary(out_dir, rows)


def joined_summary(out_dir: Path, rows: list[dict]):
    """Accuracy (09_evaluate_variants.py) and cost side by side: the data for the RQ1 frontier."""
    acc_path = out_dir / "accuracy.json"
    if not acc_path.exists() or not rows:
        return
    acc = {r["variant"]: r for r in json.loads(acc_path.read_text())["rows"]}
    cost = {(r["config"], r["task"], r["decoding"]): r for r in rows}

    def v(m):
        return "-" if not m else f"{m['value']:.1f}"

    def j(r):
        return "-" if not r or r["j"][0] is None else f"{r['j'][0]:.3f}"

    def f0(x):
        return "-" if x is None else f"{x:.0f}"
    lines = ["# Accuracy and cost per configuration (Karpathy test accuracy; cost at batch 1, J incl. idle)\n",
             "| Configuration | CIDEr | CHAIR_i | VQA | J/caption | J/answer | Latency/caption ms | Size MiB | "
             "Memory MB |", "|---|---|---|---|---|---|---|---|---|"]
    for name, a in acc.items():
        dec = a["decoding"]
        c = cost.get((a["config"], "caption", dec))
        q = cost.get((a["config"], "vqa", "greedy")) if dec == "beam3" else None
        lat = c["lat_median"][0] if c else None
        size = c["size_mb"] if c else None
        peak = c["memory_mb"] if c else None
        lines.append(f"| {name} | {v(a.get('CIDEr'))} | {v(a.get('CHAIR_i'))} | {v(a.get('VQA'))} | {j(c)} | {j(q)} | "
                     f"{f0(lat)} | {f0(size)} | {f0(peak)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPaused: finished windows are saved. Open scripts/variants_energy.command to resume.", flush=True)
        sys.exit(130)
