#!/usr/bin/env python3
"""CNN-LSTM baseline (methodology §5): Show and Tell over cached ResNet-50 features, trained once as an external
reference point at the low-cost end of the captioning frontier. Design agreed with Aryan on 4-5 Oct.

    open scripts/baseline.command                   # Terminal window: password once, then ~1.5-2 h
    sudo -v && python scripts/13_baseline.py
    python scripts/13_baseline.py --summary-only
    python scripts/13_baseline.py --label _smoke --train-limit 20000 --epochs 2 --val-limit 200 --test-limit 200 \\
        --n 8 --repeats 1 --pause 1 --idle-seconds 5 --warmup 2      # code check (features are extracted in full)

Steps, each skipped when its output exists (Ctrl+C pauses; open the launcher again to resume):
  1. ResNet-50 features (torchvision IMAGENET1K_V2, frozen, 2048-d average pool) of the COCO Karpathy train, val and
     test images with 02_extract_features.py: the same extraction code, energy log and idle window as the CLIP
     features, so the extraction energy is counted the same way.
  2. Training (--part train): all 5 captions per training image (566,747 pairs), teacher forcing, cross-entropy,
     Adam lr 5e-4 x 0.8 every 3 epochs, batch 128, dropout 0.5, gradient norm clipped at 1.0 (as for the main
     model), seed 0, 20 epochs. Words: the evaluation's PTB tokenisation of the training captions, words seen at least
     5 times, captions cut at 16 words. After every epoch, validation CIDEr on the 5,000 Karpathy val images with
     beam 3 (the main model's selection rule); the best epoch is kept. Energy per epoch over the training steps, idle
     power once per session (as in 03_train.py). A stopped epoch is repeated on resume and counted once.
  3. Test evaluation (--part test): the best epoch on the 5,000 Karpathy test images with beam 3, scored by the same
     code as 05_evaluate.py: BLEU-4, CIDEr, SPICE, CLIPScore (the disjoint scorer), CHAIR, 95 % bootstrap intervals.
     No VQA: the baseline has no VQA head.
  4. Inference cost (--part energy): the step-10 design on the 500 step-7 caption images, batch 1. Four windows per
     repetition in random order: ResNet-50 encoding, Show-and-Tell beam-3 decoding, and the FP32 reference's encoder
     and decoder windows, so every repetition has its own FP32 to compare against and the FP32 values can be checked
     against step 10's (different session). Idle once per repetition, warm-up, 20 s pause, suspect-reading retry.
     Memory of the baseline (encoder + decoder loaded, 20 captions) in a fresh process; size as stored.
  5. Summary: results/baseline/<name>/summary.md and .json, next to the FP32 reference model.
"""
import argparse
import importlib.util
import json
import math
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.data import FeatureStore, caption_rows, load_tokenizer  # noqa: E402
from greenvl.energy import KEEPER_ENV, EnergyMeter, awake_clock  # noqa: E402
from greenvl.inference import encode_image, tensor_bytes  # noqa: E402
from greenvl.lock import RunLock  # noqa: E402
from greenvl.measure import (CO2_G_PER_KWH, mean_sd, measure_idle_checked, metal_mb, peak_rss_mb,  # noqa: E402
                             percentile, subset, system_state, tensor_mb, top_processes, warm_file_cache)
from greenvl.metrics import CLIPScorer, caption_scores, cider, ptb_tokenize  # noqa: E402
from greenvl.model import IMAGE_MODELS, SCORER  # noqa: E402
from greenvl.precision import REFERENCE, load_decoder, load_encoder  # noqa: E402
from greenvl.showtell import ENCODER, MAX_WORDS, ShowTell, Vocab, beam_search, generate, parameter_count, \
    sequences  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
NAME = "showtell_resnet50_s0"
REF_RUN, REF_CKPT = "coco_ViT-L-14_lora8_lr1e-3_s0", "epoch_02.pt"  # the FP32 reference model (steps 6-10)
EPOCHS, BATCH, LR, DECAY, DECAY_EVERY, CLIP, SEED = 20, 128, 5e-4, 0.8, 3, 1.0, 0
DECODING = "beam3"  # selection and test decoding, as for the main model
SPLIT = "test"
MEMORY_ITEMS = 20
PAUSED = 130


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def references(split: str) -> dict:
    d = json.loads((paths.PROCESSED / f"coco_refs_{split}.json").read_text())
    refs = {}
    for a in d["annotations"]:
        refs.setdefault(a["image_id"], []).append(a["caption"])
    return refs


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def load_model(out: Path, which: str, dev) -> tuple[ShowTell, Vocab, dict]:
    vocab = Vocab.load(out / "vocab.json")
    state = torch.load(out / f"{which}.pt", map_location="cpu")
    model = ShowTell(len(vocab))
    model.load_state_dict(state["weights"])
    return model.to(dev).eval(), vocab, state


def require_counters(meter, allow: bool):
    if meter.available and not meter.counters_live and not allow:
        sys.exit("CPU and DRAM energy counters are not live. Run `sudo -v` and start this script again "
                 "(or pass --allow-gpu-only-energy).")


# ---------------------------------------------------------------- step 2: training

def part_train(args, out: Path):
    RunLock(out / ".lock_train")
    dev = get_device(args.device)
    torch.manual_seed(SEED)
    last_path = out / "last.pt"
    resume = torch.load(last_path, map_location="cpu", weights_only=False) if last_path.exists() else None
    if resume and resume["epoch"] >= args.epochs:
        print(f"training finished ({resume['epoch']} epochs; best epoch {resume['best_epoch']})")
        return

    # ---- data: PTB tokenisation of the training captions (cached), vocabulary, fixed-length sequences
    t0 = time.perf_counter()
    rows = caption_rows("coco", "train")
    tok_path = paths.RESULTS / "baseline" / "coco_train_ptb.json"  # shared by every baseline run
    if tok_path.exists():
        tokenized = {int(k): v for k, v in json.loads(tok_path.read_text()).items()}
    else:
        print(f"PTB tokenisation of {sum(len(r['captions']) for r in rows):,} training captions", flush=True)
        tokenized = ptb_tokenize({r["image_id"]: r["captions"] for r in rows})
        write_json(tok_path, {str(k): v for k, v in tokenized.items()})
    store = FeatureStore(ENCODER, "coco", ["coco_train"])
    pairs = [(store.index[r["image_id"]], c) for r in rows for c in tokenized[r["image_id"]]]
    if args.train_limit:
        pairs = random.Random(SEED).sample(pairs, args.train_limit)
    if (out / "vocab.json").exists():
        vocab = Vocab.load(out / "vocab.json")
    else:
        vocab = Vocab.build([c for _, c in pairs])
        vocab.save(out / "vocab.json")
    inp, tgt = sequences(vocab, [c for _, c in pairs])
    cut = sum(len(c.split()) > MAX_WORDS for _, c in pairs)
    unk = float((tgt == vocab.unk).sum() / (tgt != vocab.pad).sum())
    img = torch.tensor([i for i, _ in pairs])
    feats, inp, tgt, img = store.feats.to(dev), inp.long().to(dev), tgt.long().to(dev), img.to(dev)
    N = len(pairs)
    steps = math.ceil(N / args.batch)
    val_refs = references("val")
    val_ids = sorted(val_refs)[: args.val_limit] if args.val_limit else sorted(val_refs)
    val_store = FeatureStore(ENCODER, "coco", ["coco_val"])
    val_feats = torch.stack([val_store.get(i) for i in val_ids])
    val_gts = ptb_tokenize({i: val_refs[i] for i in val_ids})
    print(f"data ready in {time.perf_counter() - t0:.0f} s: {N:,} caption pairs, vocabulary {len(vocab):,}, "
          f"{cut / N:.1%} captions cut at {MAX_WORDS} words, <unk> {unk:.2%} of target words, {steps:,} steps/epoch",
          flush=True)

    # ---- model, optimiser, resume
    model = ShowTell(len(vocab)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    start, best = 0, {"epoch": None, "cider": -1.0}
    if resume:
        model.load_state_dict(resume["weights"])
        opt.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["rng"])
        start, best = resume["epoch"], {"epoch": resume["best_epoch"], "cider": resume["best_cider"]}
        print(f"resuming at epoch {start} (best so far: epoch {best['epoch']}, val CIDEr {100 * best['cider']:.1f})")
    cfg = {"name": out.name, "encoder": ENCODER, "decoder": "Show and Tell LSTM", "parameters": parameter_count(model),
           "vocab": len(vocab), "pairs": N, "epochs": args.epochs, "batch": args.batch, "lr": LR, "decay": DECAY,
           "decay_every": DECAY_EVERY, "clip": CLIP, "seed": SEED, "selection": f"CIDEr, validation, {DECODING}",
           "val_images": len(val_ids), "train_limit": args.train_limit}
    write_json(out / "config.json", cfg)
    print(f"device {dev}; {cfg['parameters']:,} parameters", flush=True)

    meter = EnergyMeter(dev.type)
    require_counters(meter, args.allow_gpu_only_energy)
    idle = meter.measure_idle(args.idle_seconds) if meter.available and args.idle_seconds > 0 else None
    idle_w = idle["watts"] if idle else None
    log_f = (out / "log.jsonl").open("a")

    def log(entry):
        log_f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")
        log_f.flush()

    log({"type": "start", "from_epoch": start, "device": str(dev), "energy_backend": meter.backend,
         "energy_counters_live": meter.counters_live, "idle_window": idle})
    epoch = start
    try:
        for epoch in range(start, args.epochs):
            lr = LR * DECAY ** (epoch // DECAY_EVERY)
            for g in opt.param_groups:
                g["lr"] = lr
            perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED * 1000 + epoch)).to(dev)
            model.train()
            total, count, t_log, seen = 0.0, 0, time.perf_counter(), 0
            if meter.available:
                meter.begin("train")
            t_ep = awake_clock()
            for s in range(steps):
                idx = perm[s * args.batch: (s + 1) * args.batch]
                logits = model(feats[img[idx]], inp[idx])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt[idx].reshape(-1),
                                       ignore_index=vocab.pad)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {s}")
                total += value
                count += 1
                seen += len(idx)
                if (s + 1) % args.log_every == 0:
                    dt = time.perf_counter() - t_log
                    print(f"epoch {epoch} step {s + 1:,}/{steps:,} loss {total / count:.3f} | "
                          f"{seen / dt:.0f} pairs/s, lr {lr:.2e}", flush=True)
                    t_log, seen = time.perf_counter(), 0
            synchronize(dev)
            seconds = awake_clock() - t_ep
            energy = meter.end("train") if meter.available else None
            above = (energy["total_j"] - idle_w * seconds if energy and energy.get("total_j") is not None
                     and idle_w is not None else None)

            t_val = time.perf_counter()
            caps = generate(model, vocab, val_feats, DECODING)
            score, _ = cider(val_gts, ptb_tokenize({i: [c] for i, c in zip(val_ids, caps)}))
            val_seconds = time.perf_counter() - t_val
            if score > best["cider"]:
                best = {"epoch": epoch, "cider": score}
                torch.save({"weights": model.state_dict(), "epoch": epoch, "val_cider": score}, out / "best.pt")
            entry = {"type": "epoch", "epoch": epoch, "lr": lr, "train_loss": round(total / count, 4),
                     "train_seconds": round(seconds, 1), "train_energy": energy,
                     "train_energy_above_idle_j": round(above, 1) if above is not None else None,
                     "val_cider": round(score, 4), "val_seconds": round(val_seconds, 1),
                     "val_example": caps[0], "best_epoch": best["epoch"]}
            log(entry)
            tmp = last_path.with_suffix(".tmp")
            torch.save({"weights": model.state_dict(), "optimizer": opt.state_dict(), "rng": torch.get_rng_state(),
                        "epoch": epoch + 1, "best_epoch": best["epoch"], "best_cider": best["cider"]}, tmp)
            tmp.replace(last_path)
            wh = f", {energy['total_j'] / 3600:.2f} Wh" if energy and energy.get("total_j") is not None else ""
            print(f"== epoch {epoch}: train loss {total / count:.3f}, {seconds / 60:.1f} min{wh} | val CIDEr "
                  f"{100 * score:.1f} ({val_seconds:.0f} s){' (best)' if best['epoch'] == epoch else ''} | "
                  f"\"{caps[0]}\"", flush=True)
            empty_cache(dev)
    except KeyboardInterrupt:
        if meter.available:
            try:
                meter.end("train")
            except Exception:  # noqa: BLE001  (the window may already be closed)
                pass
        log({"type": "stopped", "epoch": epoch})
        print(f"\nstopped during epoch {epoch}; it is repeated when the training resumes.", flush=True)
        sys.exit(PAUSED)
    finally:
        meter.close()
        log_f.close()


# ---------------------------------------------------------------- step 3: test evaluation

def part_test(args, out: Path):
    ev = load_script("05_evaluate.py")
    dev = get_device(args.device)
    model, vocab, state = load_model(out, "best", dev)
    refs = references(SPLIT)
    ids = sorted(refs)[: args.test_limit] if args.test_limit else sorted(refs)
    store = FeatureStore(ENCODER, "coco", [f"coco_{SPLIT}"])
    feats = torch.stack([store.get(i) for i in ids])
    t0 = time.perf_counter()
    caps = dict(zip(ids, generate(model, vocab, feats, DECODING)))
    seconds = time.perf_counter() - t0
    scorer = CLIPScorer(IMAGE_MODELS[SCORER][0], dev)
    scorer.name = SCORER
    embeds = ev.scorer_embeddings("coco", SPLIT, ids)
    res, per_image, keys = caption_scores({i: refs[i] for i in ids}, caps, spice_on=not args.no_spice,
                                          clip_scorer=scorer, image_embeds=embeds)
    chair, gt = ev.chair_ground_truth(SPLIT, ids)
    ch = chair.score({i: caps[i] for i in keys}, gt)
    mentions = np.array([ch["per_caption"][i]["mentions"] for i in keys], dtype=np.float64)
    hall = np.array([ch["per_caption"][i]["hallucinated"] for i in keys], dtype=np.float64)
    res["CHAIR_i"] = {"value": ch["CHAIR_i"], "ci95": ev.ratio_ci(hall, mentions)}
    res["CHAIR_s"] = {"value": ch["CHAIR_s"], "ci95": ev.mean_ci((hall > 0).astype(np.float64))}
    res["object_mentions"] = ch["object_mentions"]
    res["decode_seconds"] = round(seconds, 1)
    stem = f"{SPLIT}_{DECODING}"
    write_json(out / f"{stem}_captions.json", {"decoding": DECODING, "seconds": round(seconds, 1), "captions": [
        {"image_id": i, "caption": caps[i],
         "hallucinated": [w for w, _ in ch["per_caption"][i]["hallucinated_words"]]} for i in ids]})
    result = {"run": out.name, "checkpoint": f"best.pt (epoch {state['epoch']})", "data": "coco", "split": SPLIT,
              "decoding": DECODING, "precision": REFERENCE, "limit": args.test_limit,
              "time": datetime.now().isoformat(timespec="seconds"), "device": str(dev), "caption": res}
    write_json(out / f"{stem}.json", result)
    ev.print_summary(result)


# ---------------------------------------------------------------- step 4: inference cost

def windows() -> list[dict]:
    return [{"kind": "encode", "model": "baseline", "key": "encode:resnet50"},
            {"kind": "decode", "model": "baseline", "key": f"decode:showtell:caption:{DECODING}"},
            {"kind": "encode", "model": "reference", "key": "encode:fp32"},
            {"kind": "decode", "model": "reference", "key": f"decode:fp32:caption:{DECODING}"}]


def run_showtell(model, vocab, embs, dev):
    lat, outputs = [], []
    with torch.inference_mode():
        for e in embs:
            t0 = time.perf_counter()
            out = vocab.decode(beam_search(model, e[None], 3, vocab)[0])
            synchronize(dev)
            lat.append(time.perf_counter() - t0)
            outputs.append(out)
    return lat, outputs


def memory_check(args, out: Path, dev):
    """Memory of the baseline as deployed (ResNet-50 and the LSTM loaded, then 20 captions), fresh process; the
    step-10 memory check, for the FP32 reference see results/variants/.../memory.json."""
    import gc

    import psutil

    items = subset(SPLIT, args.n)["caption"][:MEMORY_ITEMS]
    encoder, processor, _ = load_encoder(ENCODER, REFERENCE, dev)
    model, vocab, _ = load_model(out, "best", dev)
    gc.collect()
    empty_cache(dev)
    tensors = lambda: tensor_mb(dev) or 0.0  # noqa: E731
    peak = {"tensor_mb": tensors(), "metal_mb": metal_mb(dev) or 0.0}
    loaded = dict(peak)
    with torch.inference_mode():
        for it in items:
            e = encode_image(encoder, processor, paths.COCO_IMAGES / it["file"], dev, REFERENCE)
            beam_search(model, e[None], 3, vocab)
            peak["tensor_mb"] = max(peak["tensor_mb"], tensors())
            peak["metal_mb"] = max(peak["metal_mb"], metal_mb(dev) or 0.0)
    gpu = dev.type in ("mps", "cuda")
    res = {"items": len(items), "loaded": loaded, "peak_tensor_mb": peak["tensor_mb"] if gpu else None,
           "peak_metal_mb": peak["metal_mb"] if gpu else None,
           "rss_mb": psutil.Process().memory_info().rss / 2**20, "lifetime_peak_rss_mb": peak_rss_mb(),
           "size_mb": {"encoder": tensor_bytes(encoder) / 2**20, "decoder": tensor_bytes(model) / 2**20}}
    print("MEMORY " + json.dumps(res), flush=True)


def part_energy(args, out: Path):
    m10 = load_script("10_measure_variants.py")
    RunLock(out / ".lock_energy")
    dev = get_device(args.device)
    mem_path = out / "memory.json"
    if not mem_path.exists():
        print("memory check: baseline", flush=True)
        r = subprocess.run([sys.executable, __file__, "--part", "memory", "--label", args.label or "", "--n",
                            str(args.n)] + (["--device", args.device] if args.device else []),
                           capture_output=True, text=True)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("MEMORY ")), None)
        if r.returncode != 0 or line is None:
            sys.exit(f"memory check failed:\n{r.stderr[-2000:]}")
        write_json(mem_path, json.loads(line[len("MEMORY "):]))

    log_path = out / "energy.jsonl"
    records = [json.loads(ln) for ln in log_path.read_text().splitlines()] if log_path.exists() else []
    done = {(r["repeat"], r["key"]) for r in records
            if r["type"] == "window" and r["subset_n"] == args.n and not r.get("suspect")}
    meter = EnergyMeter(dev.type)
    require_counters(meter, args.allow_gpu_only_energy)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    system = system_state()
    if sys.platform == "darwin" and not system.get("on_ac_power"):
        print("WARNING: not on AC power; measurements should run on AC power.", flush=True)

    def log(rec):
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    try:
        items = subset(SPLIT, args.n)["caption"]
        print(f"subset: {args.n} caption images (step 7/10); file cache warmed in {warm_file_cache(items):.1f} s",
              flush=True)
        ref_model, cfg, _, _ = load_decoder(REF_RUN, REF_CKPT, REFERENCE, dev)
        tok = load_tokenizer(cfg["decoder"])
        encoders = {"reference": load_encoder(cfg["encoder"], REFERENCE, dev)[:2],
                    "baseline": load_encoder(ENCODER, REFERENCE, dev)[:2]}
        base_model, vocab, state = load_model(out, "best", dev)
        sizes = {"reference": {"encoder": tensor_bytes(encoders["reference"][0]) / 2**20,
                               "decoder": tensor_bytes(ref_model) / 2**20},
                 "baseline": {"encoder": tensor_bytes(encoders["baseline"][0]) / 2**20,
                              "decoder": tensor_bytes(base_model) / 2**20}}
        log({"type": "session", "session": session, "time": datetime.now().isoformat(timespec="seconds"),
             "baseline": out.name, "baseline_epoch": state["epoch"], "reference": [REF_RUN, REF_CKPT], "n": args.n,
             "repeats": args.repeats, "warmup": args.warmup, "pause": args.pause, "idle_seconds": args.idle_seconds,
             "sizes_mb": sizes, "counters_live": meter.counters_live, "system": system})
        embs = {}
        for m in ("reference", "baseline"):  # unmeasured; also warms both encoders
            f = out / "embeddings" / f"{m}_n{args.n}.pt"
            if f.exists():
                embs[m] = torch.load(f)
            else:
                print(f"{m} embeddings of the {len(items)} images (unmeasured)", flush=True)
                embs[m] = torch.stack([encode_image(*encoders[m], paths.COCO_IMAGES / it["file"], dev, REFERENCE).cpu()
                                       for it in items])
                f.parent.mkdir(parents=True, exist_ok=True)
                torch.save(embs[m], f)

        idle_history = []
        for rep in range(args.repeats):
            order = windows()
            random.Random(2000 + rep).shuffle(order)
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
                m = w["model"]
                if w["kind"] == "encode":
                    enc, proc = encoders[m]
                    m10.run_encode(enc, proc, REFERENCE, items[: args.warmup], dev)
                    work = lambda: m10.run_encode(enc, proc, REFERENCE, items, dev)  # noqa: E731
                else:
                    e = embs[m].to(dev)
                    if m == "baseline":
                        run_showtell(base_model, vocab, e[: args.warmup], dev)
                        work = lambda: run_showtell(base_model, vocab, e, dev)  # noqa: E731
                    else:
                        m10.run_decode(ref_model, tok, "caption", DECODING, items[: args.warmup], e[: args.warmup], dev)
                        work = lambda: m10.run_decode(ref_model, tok, "caption", DECODING, items, e, dev)  # noqa: E731
                suspects = []
                for attempt in range(1, m10.WINDOW_ATTEMPTS + 1):
                    time.sleep(args.pause)
                    (lat, outputs), energy, seconds = m10.measured(meter, work)
                    reason = m10.suspect_reading(energy, seconds, meter)
                    if not reason:
                        break
                    suspects.append(reason)
                    print(f"    {w['key']}: {reason}; " + ("keeping this reading" if attempt == m10.WINDOW_ATTEMPTS
                                                           else "measuring the window again"), flush=True)
                n = len(items)
                total = energy["total_j"] if energy and energy.get("total_j") is not None else None
                rec = {"type": "window", "session": session, "time": datetime.now().isoformat(timespec="seconds"),
                       "repeat": rep, "position": pos, **w, "subset_n": args.n, "n": n, "seconds": seconds,
                       "energy": energy, "idle_w": idle_w, "j_per_item": total / n if total is not None else None,
                       "j_per_item_above_idle": (total - idle_w * seconds) / n
                       if total is not None and idle_w is not None else None,
                       "latency_ms": [round(1000 * x, 3) for x in lat], "attempts": attempt,
                       "suspect_readings": suspects, "suspect": bool(reason), "background_after": top_processes()}
                if outputs is not None and rep == 0:
                    rec["outputs"] = outputs
                log(rec)
                if not rec["suspect"]:
                    done.add((rep, w["key"]))
                j = rec["j_per_item_above_idle"]
                print(f"  [{pos + 1}/{len(order)}] {w['key']:<32} {statistics.median(lat) * 1000:6.0f} ms median"
                      + (f", {j:.4f} J/item above idle" if j is not None else ""), flush=True)
    finally:
        meter.close()


# ---------------------------------------------------------------- step 5: summary

def cost_rows(out: Path, n: int) -> dict:
    """Per model: J per caption (incl. idle and above idle), ratio to the same repetition's FP32, latency."""
    path = out / "energy.jsonl"
    if not path.exists():
        return {}
    win = {}
    for r in (json.loads(ln) for ln in path.read_text().splitlines()):
        if r["type"] == "window" and r["subset_n"] == n and not r.get("suspect"):
            win[(r["repeat"], r["key"])] = r
    reps = sorted({k[0] for k in win})
    keys = {"baseline": ("encode:resnet50", f"decode:showtell:caption:{DECODING}"),
            "reference": ("encode:fp32", f"decode:fp32:caption:{DECODING}")}
    rows = {}
    for m, (ek, dk) in keys.items():
        per = []
        for rep in reps:
            e, d = win.get((rep, ek)), win.get((rep, dk))
            re_, rd = win.get((rep, keys["reference"][0])), win.get((rep, keys["reference"][1]))
            if not e or not d or e["j_per_item"] is None or d["j_per_item"] is None:
                continue
            x = {"j": e["j_per_item"] + d["j_per_item"], "enc_j": e["j_per_item"], "dec_j": d["j_per_item"]}
            if e["j_per_item_above_idle"] is not None and d["j_per_item_above_idle"] is not None:
                x["j_above"] = e["j_per_item_above_idle"] + d["j_per_item_above_idle"]
                x["enc_above"], x["dec_above"] = e["j_per_item_above_idle"], d["j_per_item_above_idle"]
            if re_ and rd and re_["j_per_item"] is not None and rd["j_per_item"] is not None:
                x["rel_fp32"] = x["j"] / (re_["j_per_item"] + rd["j_per_item"])
                if "j_above" in x and re_["j_per_item_above_idle"] is not None:
                    x["rel_fp32_above"] = x["j_above"] / (re_["j_per_item_above_idle"] + rd["j_per_item_above_idle"])
            for c in ("cpu_j", "gpu_j", "dram_j"):
                x[c] = (e["energy"].get(c) or 0.0) / e["n"] + (d["energy"].get(c) or 0.0) / d["n"]
            lat = [a + b for a, b in zip(e["latency_ms"], d["latency_ms"])]
            x["lat_median"], x["lat_p95"] = percentile(lat, 0.5), percentile(lat, 0.95)
            if re_ and rd:
                ref_lat = [a + b for a, b in zip(re_["latency_ms"], rd["latency_ms"])]
                x["rel_lat_median"] = x["lat_median"] / percentile(ref_lat, 0.5)
            x["enc_median"], x["dec_median"] = percentile(e["latency_ms"], 0.5), percentile(d["latency_ms"], 0.5)
            per.append(x)
        if per:
            rows[m] = {k: mean_sd([x.get(k) for x in per]) for k in per[0]} | {"repeats": len(per)}
    return rows


def training_totals(log_path: Path, key_energy="train_energy") -> dict:
    """Sum over the last entry of each epoch: time, energy incl. idle, above idle."""
    if not log_path.exists():
        return {}
    epochs = {}
    for r in (json.loads(ln) for ln in log_path.read_text().splitlines()):
        if r.get("type") == "epoch":
            epochs[r["epoch"]] = r
    sec = sum(r.get("train_seconds") or 0 for r in epochs.values())
    j = [((r.get(key_energy) or {}).get("total_j")) for r in epochs.values()]
    above = [r.get("train_energy_above_idle_j") for r in epochs.values()]
    return {"epochs": len(epochs), "seconds": sec, "wh": sum(j) / 3600 if j and None not in j else None,
            "wh_above_idle": sum(above) / 3600 if above and None not in above else None, "per_epoch": epochs}


def extraction_wh(encoder_slug: str) -> tuple[float | None, float | None]:
    """Energy of the COCO feature extraction (train, val, test), incl. and above idle, from 02_extract_features.py's
    logs (newest per set)."""
    sets = {}
    for f in sorted((paths.RESULTS / "features").glob(f"extract_{encoder_slug}_2*.json")):
        for s, e in json.loads(f.read_text()).get("sets", {}).items():
            if s.startswith("coco_") and e.get("energy"):
                above = e["j_per_image_above_idle"] * e["images"] if e.get("j_per_image_above_idle") is not None \
                    else None
                sets[s] = (e["energy"]["total_j"], above)
    if len(sets) != 3:
        return None, None
    above = [a for _, a in sets.values()]
    return sum(t for t, _ in sets.values()) / 3600, (sum(above) / 3600 if None not in above else None)


def counters_live(out: Path) -> dict:
    """Whether CPU and DRAM energy was measured (powermetrics running) in each measured part; None if not run."""
    def read(path):
        return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []
    res = {}
    starts = [r for r in read(out / "log.jsonl") if r.get("type") == "start"]
    res["training"] = all(r.get("energy_counters_live") for r in starts) if starts else None
    sessions = [r for r in read(out / "energy.jsonl") if r.get("type") == "session"]
    res["inference cost"] = all(r.get("counters_live") for r in sessions) if sessions else None
    sets = {}
    for f in sorted((paths.RESULTS / "features").glob(f"extract_{ENCODER}_2*.json")):
        for k, e in json.loads(f.read_text()).get("sets", {}).items():
            sets[k] = (e.get("energy") or {}).get("counters_live")
    res["feature extraction"] = all(sets.values()) if sets else None
    return res


def write_summary(args, out: Path):
    test = json.loads((out / f"{SPLIT}_{DECODING}.json").read_text()) if (out / f"{SPLIT}_{DECODING}.json").exists() \
        else None
    ref_eval = paths.RESULTS / "eval" / REF_RUN / f"{Path(REF_CKPT).stem}_{SPLIT}_{DECODING}.json"
    ref = json.loads(ref_eval.read_text()) if ref_eval.exists() else None
    train = training_totals(out / "log.jsonl")
    ref_train = training_totals(paths.RESULTS / "runs" / REF_RUN / "log.jsonl")
    cost = cost_rows(out, args.n)
    step10 = paths.RESULTS / "variants" / REF_RUN / Path(REF_CKPT).stem / "energy.json"
    s10 = next((r for r in json.loads(step10.read_text())["rows"] if r["config"] == REFERENCE
                and r["task"] == "caption" and r["decoding"] == DECODING), None) if step10.exists() else None
    mem = json.loads((out / "memory.json").read_text()) if (out / "memory.json").exists() else {}
    s10_mem = paths.RESULTS / "variants" / REF_RUN / Path(REF_CKPT).stem / "memory.json"
    ref_mem = json.loads(s10_mem.read_text()).get(REFERENCE, {}) if s10_mem.exists() else {}
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    ref_cfg = json.loads((paths.RESULTS / "runs" / REF_RUN / "config.json").read_text())

    def m(r, k):
        x = (r or {}).get("caption", {}).get(k)
        return "-" if not x else (f"{100 * x['value']:.1f} [{100 * x['ci95'][0]:.1f}, {100 * x['ci95'][1]:.1f}]"
                                  if "ci95" in x else f"{100 * x['value']:.1f}")

    def ms(t, d=3):
        if not t or t[0] is None:
            return "-"
        return f"{t[0]:.{d}f} ± {t[1]:.{d}f}" if t[1] is not None else f"{t[0]:.{d}f}"

    def f1(x, d=1):
        return "-" if x is None else f"{x:.{d}f}"
    b, r = cost.get("baseline", {}), cost.get("reference", {})
    epochs = train.get("per_epoch") or {}
    best = max(epochs.values(), key=lambda e: e["val_cider"]) if epochs else None
    head = (f"Karpathy {SPLIT}, {DECODING}. Baseline: frozen ResNet-50, {cfg.get('parameters', 0):,} trained "
            "parameters (LSTM decoder)")
    if best:
        head += f", best epoch {best['epoch']} of {len(epochs)} by validation CIDEr ({100 * best['val_cider']:.1f})"
    head += (f". Reference: CLIP ViT-L/14 + mapping network + GPT-2 with LoRA, "
             f"{ref_cfg.get('parameters', {}).get('trainable_total', 0):,} trained parameters.\n")
    lines = [f"# CNN-LSTM baseline ({out.name}) vs the FP32 reference model ({REF_RUN} {REF_CKPT})\n", head,
             "## Accuracy\n", "| Model | CIDEr [CI] | BLEU-4 | SPICE | CLIPScore | CHAIR_i | CHAIR_s | Words |",
             "|---|---|---|---|---|---|---|---|"]
    for name, x in (("Show and Tell (ResNet-50 + LSTM)", test), ("FP32 reference", ref)):
        lines.append(f"| {name} | {m(x, 'CIDEr')} | {m(x, 'BLEU-4')} | {m(x, 'SPICE')} | {m(x, 'CLIPScore')} | "
                     f"{m(x, 'CHAIR_i')} | {m(x, 'CHAIR_s')} | "
                     f"{f1((x or {}).get('caption', {}).get('mean_length_words'))} |")
    lines += ["\n## Inference cost (batch 1, J per caption, mean ± sd over repetitions, this session)\n",
              "| Model | J above idle | J incl. idle | vs FP32 (incl. idle) | encoder J | decoding J | Latency median ms "
              "| p95 ms | Size MiB | Memory MB |", "|---|---|---|---|---|---|---|---|---|---|"]
    if b:
        size = sum(mem.get("size_mb", {}).values()) if mem else None
        lines.append(f"| Show and Tell | {ms(b.get('j_above'), 4)} | {ms(b.get('j'), 4)} | {ms(b.get('rel_fp32'), 3)} | "
                     f"{ms(b.get('enc_above'), 4)} | {ms(b.get('dec_above'), 4)} | {f1(b['lat_median'][0], 0)} | "
                     f"{f1(b['lat_p95'][0], 0)} | {f1(size, 0)} | {f1(mem.get('peak_tensor_mb'), 0)} |")
    if r:
        lines.append(f"| FP32 reference | {ms(r.get('j_above'), 4)} | {ms(r.get('j'), 4)} | 1 | "
                     f"{ms(r.get('enc_above'), 4)} | {ms(r.get('dec_above'), 4)} | {f1(r['lat_median'][0], 0)} | "
                     f"{f1(r['lat_p95'][0], 0)} | {f1(s10['size_mb'] if s10 else None, 0)} | "
                     f"{f1(ref_mem.get('peak_tensor_mb'), 0)} |")
    if s10 and r:
        lines.append(f"\nCheck against step 10 (other session): FP32 reference {ms(tuple(s10['j_above']), 4)} J above "
                     f"idle there, {ms(r.get('j_above'), 4)} J here; {ms(tuple(s10['j']), 4)} vs "
                     f"{ms(r.get('j'), 4)} J incl. idle.")
    ext_b, ext_r = extraction_wh("ResNet-50"), extraction_wh(ref_cfg["encoder"].replace("/", "-"))
    lines += ["\n## Training cost\n", "| Model | Feature extraction Wh (above idle) | Training Wh (above idle) | "
              "Training h | Epochs | Wh per epoch |", "|---|---|---|---|---|---|"]
    for name, t, ext in (("Show and Tell", train, ext_b), ("FP32 reference", ref_train, ext_r)):
        if t:
            per = t["wh"] / t["epochs"] if t.get("wh") is not None and t["epochs"] else None
            lines.append(f"| {name} | {f1(ext[0])} ({f1(ext[1])}) | {f1(t.get('wh'))} ({f1(t.get('wh_above_idle'))}) | "
                         f"{f1(t['seconds'] / 3600, 2)} | {t['epochs']} | {f1(per)} |")
    lines.append("\nIncl. idle: everything the counters recorded; above idle: minus the session's idle power over the "
                 "same time, which also removes a constant background load such as macOS indexing.")
    if train.get("per_epoch"):
        lines += ["\n## Baseline epochs\n", "| Epoch | lr | Train loss | Val CIDEr | Train min | Wh |",
                  "|---|---|---|---|---|---|"]
        for e in sorted(train["per_epoch"]):
            x = train["per_epoch"][e]
            wh = (x.get("train_energy") or {}).get("total_j")
            lines.append(f"| {e} | {x['lr']:.2e} | {x['train_loss']:.3f} | {100 * x['val_cider']:.1f} | "
                         f"{x['train_seconds'] / 60:.1f} | {f1(wh / 3600 if wh is not None else None, 2)} |")
    gpu_only = [what for what, live in counters_live(out).items() if live is False]
    if gpu_only:
        lines.append(f"\nWARNING: CPU and DRAM counters were not live for {', '.join(gpu_only)}: those energies are GPU "
                     "only and not comparable with the reference's. Run again with the password (baseline.command).")
    lines.append(f"\nEnergy covers the SoC (CPU, GPU) and DRAM. CO2e at {CO2_G_PER_KWH} g/kWh: multiply Wh by "
                 f"{CO2_G_PER_KWH / 1000:.3f} g. Training Wh: the training steps only (validation decoding excluded), "
                 "as for the reference model; the reference's feature extraction covers its COCO train, val and test "
                 "features at FP32.")
    write_json(out / "summary.json", {"baseline": out.name, "test": test and test["caption"], "cost": cost,
                                      "step10_fp32": s10, "memory": mem, "training": {k: v for k, v in train.items()
                                                                                      if k != "per_epoch"},
                                      "reference_training": {k: v for k, v in ref_train.items() if k != "per_epoch"},
                                      "extraction_wh": {"baseline": ext_b[0], "reference": ext_r[0]},
                                      "extraction_wh_above_idle": {"baseline": ext_b[1], "reference": ext_r[1]},
                                      "time": datetime.now().isoformat(timespec="seconds")})
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------- orchestration

def run_child(cmd: list) -> int:
    child = subprocess.Popen([sys.executable, *map(str, cmd)])
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue


def step(cmd: list, what: str):
    print(f"\n>>> {what}  ({datetime.now():%d %b %H:%M})", flush=True)
    rc = run_child(cmd)
    if rc in (PAUSED, -2):
        print("\nPaused: progress is saved. To resume, open scripts/baseline.command.", flush=True)
        sys.exit(PAUSED)
    if rc != 0:
        sys.exit(f"{what} failed (exit {rc}); finished work is kept. Run this script again after fixing it.")


def trained(out: Path, epochs: int) -> bool:
    p = out / "last.pt"
    return p.exists() and torch.load(p, map_location="cpu", weights_only=False)["epoch"] >= epochs


def energy_done(out: Path, n: int, repeats: int) -> bool:
    p = out / "energy.jsonl"
    if not p.exists() or not (out / "memory.json").exists():
        return False
    got = {(r["repeat"], r["key"]) for r in map(json.loads, p.read_text().splitlines())
           if r["type"] == "window" and r["subset_n"] == n and not r.get("suspect")}
    return all((rep, w["key"]) in got for rep in range(repeats) for w in windows())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["train", "test", "energy", "memory"], default=None,
                    help="one step only (default: all steps in order)")
    ap.add_argument("--label", default=None, help="results folder suffix, e.g. _smoke")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--train-limit", type=int, default=None, help="random N caption pairs (code check)")
    ap.add_argument("--val-limit", type=int, default=None, help="first N validation images (code check)")
    ap.add_argument("--test-limit", type=int, default=None, help="first N test images (code check)")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--no-spice", action="store_true")
    ap.add_argument("--n", type=int, default=500, help="caption images per energy window")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--pause", type=float, default=20)
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-gpu-only-energy", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out = paths.RESULTS / "baseline" / (NAME + (args.label or ""))
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        return write_summary(args, out)
    if args.part:
        dev = get_device(args.device)
        return {"train": part_train, "test": part_test, "energy": part_energy,
                "memory": lambda a, o: memory_check(a, o, dev)}[args.part](args, out)

    RunLock(out / ".lock")
    passthrough = [f"--{k.replace('_', '-')}={v}" for k, v in vars(args).items()
                   if k in ("label", "epochs", "batch", "train_limit", "val_limit", "test_limit", "log_every", "n",
                            "repeats", "warmup", "pause", "idle_seconds", "device") and v is not None]
    passthrough += [f"--{k.replace('_', '-')}" for k in ("no_spice", "allow_gpu_only_energy") if getattr(args, k)]
    meter = EnergyMeter("mps")
    require_counters(meter, args.allow_gpu_only_energy)
    if meter.keeper_pid:
        os.environ[KEEPER_ENV] = str(meter.keeper_pid)  # the steps reuse it: the password is needed once
    try:
        missing = [s for s in ("coco_train", "coco_val", "coco_test")
                   if not (paths.FEATURES / ENCODER / f"{s}.pt").exists()]
        if missing:
            step([SCRIPTS / "02_extract_features.py", "--encoder", ENCODER, "--sets", *missing, "--idle-seconds",
                  args.idle_seconds], f"{ENCODER} features of {', '.join(missing)}")
        if not trained(out, args.epochs):
            step([__file__, "--part", "train", *passthrough], f"training, {args.epochs} epochs")
        if not (out / f"{SPLIT}_{DECODING}.json").exists():
            step([__file__, "--part", "test", *passthrough], f"Karpathy {SPLIT} evaluation, {DECODING}")
        if not energy_done(out, args.n, args.repeats):
            step([__file__, "--part", "energy", *passthrough], f"inference cost, {args.repeats} repetitions")
    finally:
        meter.close()
    write_summary(args, out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPaused: finished steps are saved. Open scripts/baseline.command to resume.", flush=True)
        sys.exit(PAUSED)
