#!/usr/bin/env python3
"""Step 1: check the environment and benchmark the machine that will train and measure.

    python scripts/00_check_env.py              # full check (downloads GPT-2, GPT-2 medium, three CLIP encoders)
    python scripts/00_check_env.py --quick      # smoke test: tiny batches, a few steps

Reports hardware, library versions, which precisions and quantization kernels run on the device, whether
energy counters are readable, training and encoder throughput, and what the training grid costs in hours
and energy. Writes results/env/env_report_<timestamp>.json.
"""
import argparse
import gc
import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from transformers import GPT2Config

from greenvl import paths
from greenvl.device import empty_cache, get_device, memory_mb, synchronize
from greenvl.energy import EnergyMeter
from greenvl.model import DECODERS, ENCODERS, TASKS, CaptionVQAModel, build_decoder

PACKAGES = ["torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes", "zeus-apple-silicon",
            "codecarbon", "pycocotools", "pycocoevalcap", "numpy", "pillow", "scipy", "pandas"]
CO2_G_PER_KWH = 727  # CEA, Indian grid, FY 2023-24 (methodology §8)

# Architecture-exact configs for --random-init (offline runs). Throughput does not depend on the weights.
GPT2_CONFIGS = {"gpt2": dict(), "gpt2-medium": dict(n_embd=1024, n_layer=24, n_head=16)}
CLIP_CONFIGS = {
    "ViT-B/32": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
                     patch_size=32, projection_dim=512),
    "ViT-B/16": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
                     patch_size=16, projection_dim=512),
    "ViT-L/14": dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
                     patch_size=14, projection_dim=768),
}
AMP = {"fp32": None, "bf16": torch.bfloat16}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def err(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"


# ---------------------------------------------------------------- system

def system_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    if sys.platform == "darwin":
        info["macos"] = platform.mac_ver()[0]
        info["chip"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = sh(["sysctl", "-n", "hw.memsize"])
        info["ram_gb"] = round(int(mem) / 2**30, 1) if mem else None
        info["cpu_performance_cores"] = sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        info["cpu_efficiency_cores"] = sh(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
        gpu = re.search(r"Total Number of Cores:\s*(\d+)", sh(["system_profiler", "SPDisplaysDataType"]) or "")
        info["gpu_cores"] = int(gpu.group(1)) if gpu else None
        batt = sh(["pmset", "-g", "batt"]) or ""
        info["power_source"] = "AC" if "AC Power" in batt else ("battery" if "Battery Power" in batt else None)
        # pmset reports lowpowermode (older macOS) or powermode (0 automatic, 1 low power, 2 high power)
        modes = dict(re.findall(r"^\s*(lowpowermode|powermode|highpowermode)\s+(\d+)", sh(["pmset", "-g"]) or "", re.M))
        info["pmset_power_modes"] = modes or None
        info["low_power_mode"] = (modes.get("lowpowermode") == "1" or modes.get("powermode") == "1") if modes else None
    else:
        try:
            import psutil

            info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        except ImportError:
            pass
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


def package_versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


# ---------------------------------------------------------------- device capabilities

def device_checks(dev):
    out = {"device": str(dev), "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
           "cuda_available": torch.cuda.is_available(), "dtypes": {}}
    for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        try:
            a = torch.randn(256, 256, device=dev, dtype=dt)
            (a @ a).sum().item()
            out["dtypes"][name] = "ok"
        except Exception as e:
            out["dtypes"][name] = err(e)
    try:
        lin = nn.Linear(64, 64).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            y = lin(torch.randn(8, 64, device=dev))
        out["autocast_bf16"] = "ok" if y.dtype == torch.bfloat16 else f"ran, output dtype {y.dtype}"
    except Exception as e:
        out["autocast_bf16"] = err(e)
    return out


def time_call(fn, dev, iters=30):
    fn()
    synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(dev)
    return (time.perf_counter() - t0) / iters * 1000


def bnb_layer_checks(dev):
    """Can bitsandbytes NF4 / INT8 layers run here, how accurate, and how fast versus an FP16 layer."""
    try:
        import bitsandbytes as bnb
    except Exception as e:
        return {"import": err(e)}
    res = {"version": getattr(bnb, "__version__", None)}
    torch.manual_seed(0)
    ref = nn.Linear(768, 3072)
    x = torch.randn(64, 768)
    with torch.no_grad():
        y_ref = ref(x)
    for target in dict.fromkeys([dev.type, "cpu"]):
        d = torch.device(target)
        r = {}
        try:
            base = nn.Linear(768, 3072).to(d, torch.float16)
            base.load_state_dict(ref.state_dict())
            xin = x.to(d, torch.float16)
            with torch.no_grad():
                r["fp16_linear_ms"] = round(time_call(lambda: base(xin), d), 3)
        except Exception as e:
            r["fp16_linear"] = err(e)
            xin = x.to(d, torch.float16)
        for kind in ["nf4", "int8"]:
            try:
                if kind == "nf4":
                    layer = bnb.nn.Linear4bit(768, 3072, compute_dtype=torch.float16, quant_type="nf4")
                    layer.weight = bnb.nn.Params4bit(ref.weight.data.clone(), requires_grad=False, quant_type="nf4")
                else:
                    layer = bnb.nn.Linear8bitLt(768, 3072, has_fp16_weights=False, threshold=6.0)
                    layer.weight = bnb.nn.Int8Params(ref.weight.data.clone(), requires_grad=False, has_fp16_weights=False)
                layer.bias = nn.Parameter(ref.bias.data.clone().half(), requires_grad=False)
                layer = layer.to(d)
                with torch.no_grad():
                    y = layer(xin)
                    rel = ((y.float().cpu() - y_ref).norm() / y_ref.norm()).item()
                    r[kind] = {"rel_error": round(rel, 4), "ms": round(time_call(lambda: layer(xin), d), 3)}
            except Exception as e:
                r[kind] = err(e)
        res[target] = r
    return res


def bnb_transformers_check(dev):
    """The path the precision factor will actually use: GPT-2 loaded through transformers' BitsAndBytesConfig."""
    try:
        from transformers import BitsAndBytesConfig, GPT2LMHeadModel
    except Exception as e:
        return {"import": err(e)}
    out = {}
    specs = {
        "nf4": dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
        "int8": dict(load_in_8bit=True),
    }
    for target in dict.fromkeys([dev.type, "cpu"]):
        for kind, spec in specs.items():
            key = f"{target}/{kind}"
            try:
                m = GPT2LMHeadModel.from_pretrained(DECODERS["gpt2"], quantization_config=BitsAndBytesConfig(**spec),
                                                    device_map={"": target})
                ids = torch.tensor([[464, 3290, 318]], device=target)
                with torch.no_grad():
                    m(input_ids=ids)
                out[key] = {"ok": True, "footprint_mb": round(m.get_memory_footprint() / 2**20, 1)}
                del m
            except Exception as e:
                out[key] = err(e)
            gc.collect()
    return out


# ---------------------------------------------------------------- benchmarks

def bench_decoder(dev, decoder, adaptation, clip_dim, batch, text_len, steps, warmup, amp, random_init, meter,
                  idle_w, train=True):
    cfg = GPT2Config(**GPT2_CONFIGS[decoder]) if random_init else None
    model = CaptionVQAModel(build_decoder(decoder, adaptation, config=cfg), clip_dim).to(dev)
    model.train(train)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4) if train else None
    vocab = model.decoder.config.vocab_size
    feats = torch.randn(batch, clip_dim, device=dev)
    ids = torch.randint(0, vocab, (batch, text_len), device=dev)
    mask = torch.ones_like(ids)
    labels = ids.clone()
    amp_dtype = AMP[amp]

    def step(i):
        model.set_task(TASKS[i % 2])
        with torch.autocast(device_type=dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            if train:
                loss = model(feats, ids, mask, labels)
            else:
                with torch.no_grad():
                    loss = model(feats, ids, mask, labels)
        if train:
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        return loss

    for i in range(warmup):
        step(i)
    synchronize(dev)
    if meter.available:
        meter.begin("bench")
    t0 = time.perf_counter()
    for i in range(steps):
        loss = step(i)
    synchronize(dev)
    dt = time.perf_counter() - t0
    energy = meter.end("bench") if meter.available else None
    n = steps * batch
    out = {"samples_per_s": round(n / dt, 1), "ms_per_step": round(dt / steps * 1000, 1),
           "memory_mb": memory_mb(dev), "loss_finite": bool(torch.isfinite(loss).item()),
           "trainable_params": model.parameter_counts()["trainable_total"]}
    if energy and energy["total_j"] is not None:
        out["j_per_sample"] = round(energy["total_j"] / n, 4)
        out["watts"] = round(energy["total_j"] / energy["seconds"], 2)
        out["j_per_sample_by_component"] = {k[:-2]: (round(v / n, 5) if v is not None else None)
                                            for k, v in energy.items() if k.endswith("_j") and k != "total_j"}
        if idle_w is not None:
            out["j_per_sample_above_idle"] = round((energy["total_j"] - idle_w * energy["seconds"]) / n, 4)
    del model, opt
    gc.collect()
    empty_cache(dev)
    return out


def bench_encoder(dev, name, batch, steps, warmup, dtype, random_init):
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    if random_init:
        model = CLIPVisionModelWithProjection(CLIPVisionConfig(image_size=224, **CLIP_CONFIGS[name]))
    else:
        model = CLIPVisionModelWithProjection.from_pretrained(ENCODERS[name][0])
    model = model.to(dev, dtype).eval()
    x = torch.randn(batch, 3, 224, 224, device=dev, dtype=dtype)
    with torch.inference_mode():
        for _ in range(warmup):
            model(pixel_values=x)
        synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(steps):
            emb = model(pixel_values=x).image_embeds
        synchronize(dev)
    dt = time.perf_counter() - t0
    out = {"images_per_s": round(steps * batch / dt, 1), "embed_dim": emb.shape[-1]}
    del model
    gc.collect()
    empty_cache(dev)
    return out


# ---------------------------------------------------------------- estimates

def dataset_sizes():
    """Training-set sizes from step 2's summary if it exists, else the counts in datasets.md."""
    summary = paths.PROCESSED / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        return s["coco"]["train_pairs"], s["vqa"]["train_questions"], s["feature_images_total"], "processed/summary.json"
    return 566_747, 443_757, 139_037, "fallback constants"


def grid_estimate(train, enc, epochs, seeds):
    """Hours and energy for the one-factor-at-a-time training grid (methodology Table 2), before the focused grid."""
    n_cap, n_vqa, n_images, source = dataset_sizes()
    samples = 2 * n_cap  # one epoch = one pass over caption pairs, with an equal number of VQA batches
    est = {"source": source, "caption_pairs": n_cap, "vqa_questions": n_vqa, "samples_per_epoch": samples,
           "epochs": epochs, "seeds": seeds, "by_precision": {}}
    for amp in AMP:
        def tp(key):
            r = train.get(f"{key}/{amp}")
            return r["samples_per_s"] if isinstance(r, dict) else None

        def jps(key):
            r = train.get(f"{key}/{amp}")
            return r.get("j_per_sample_above_idle") if isinstance(r, dict) else None

        lora, frozen, full, teacher, teacher_fwd = (tp(k) for k in
                                                     ["gpt2/lora", "gpt2/frozen", "gpt2/full", "gpt2-medium/lora", "gpt2-medium/forward"])
        student = 1 / (1 / lora + 1 / teacher_fwd) if lora and teacher_fwd else None
        rows = [
            ("LoRA r8 reference, ViT-B/16, ViT-L/14, r4, r16", 5 * seeds, lora, jps("gpt2/lora")),
            ("Frozen GPT-2", seeds, frozen, jps("gpt2/frozen")),
            ("Full fine-tuning", seeds, full, jps("gpt2/full")),
            ("Distilled student (+ teacher forward)", seeds, student,
             (jps("gpt2/lora") or 0) + (jps("gpt2-medium/forward") or 0) or None),
            ("Teacher: GPT-2 medium, LoRA r8", 1, teacher, jps("gpt2-medium/lora")),
        ]
        table, total_h, total_kwh = [], 0.0, 0.0
        for label, runs, rate, j in rows:
            h = runs * epochs * samples / rate / 3600 if rate else None
            kwh = runs * epochs * samples * j / 3.6e6 if j else None
            table.append({"configs": label, "runs": runs, "hours": round(h, 1) if h else None,
                          "kwh": round(kwh, 2) if kwh else None})
            total_h += h or 0
            total_kwh += kwh or 0
        est["by_precision"][amp] = {"rows": table, "total_hours": round(total_h, 1), "total_days": round(total_h / 24, 1),
                                    "total_kwh": round(total_kwh, 2) if total_kwh else None,
                                    "total_co2_kg_at_india_grid": round(total_kwh * CO2_G_PER_KWH / 1000, 2) if total_kwh else None}
    est["feature_extraction_hours"] = {k: round(n_images / v["images_per_s"] / 3600, 2)
                                       for k, v in enc.items() if isinstance(v, dict)}
    return est


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--text-len", type=int, default=24, help="caption / question+answer tokens after the prefix")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10, help="planning figure for the estimate (ClipCap's COCO schedule)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--idle-seconds", type=float, default=10)
    ap.add_argument("--skip-bench", action="store_true", help="only system, device, quantization and energy checks")
    ap.add_argument("--random-init", action="store_true", help="offline: random weights, exact architectures")
    ap.add_argument("--quick", action="store_true", help="smoke test with tiny batches")
    args = ap.parse_args()
    if args.quick:
        args.batch, args.steps, args.warmup, args.text_len, args.idle_seconds = 4, 2, 1, 8, 1

    dev = get_device(args.device)
    report = {"timestamp": datetime.now().isoformat(timespec="seconds"), "args": vars(args)}

    print("== System")
    report["system"] = system_info()
    report["packages"] = package_versions()
    for k, v in {**report["system"], **report["packages"]}.items():
        print(f"  {k:24s} {v}")
    if report["system"].get("power_source") == "battery" or report["system"].get("low_power_mode"):
        print("  WARNING: on battery or Low Power Mode. Benchmarks and energy readings need AC power, Low Power Mode off.")

    print("== Device")
    report["device"] = device_checks(dev)
    print(json.dumps(report["device"], indent=2))

    print("== Energy counters")
    meter = EnergyMeter(dev.type)
    idle = meter.measure_idle(args.idle_seconds) if meter.available else None
    idle_w = idle["watts"] if idle else None
    report["energy"] = {"backend": meter.backend, "error": meter.error, "idle_w": idle_w, "idle_window": idle}
    print(f"  backend {meter.backend}  idle {idle_w} W  {meter.error or ''}")
    if idle:
        print("  idle by component (W): " + ", ".join(
            f"{k[:-2]} {v / idle['seconds']:.3f}" if v is not None else f"{k[:-2]} n/a"
            for k, v in idle.items() if k.endswith("_j") and k != "total_j"))

    print("== bitsandbytes layers (NF4, INT8)")
    report["bnb_layers"] = bnb_layer_checks(dev)
    print(json.dumps(report["bnb_layers"], indent=2))

    if not args.skip_bench:
        if not args.random_init:
            print("== bitsandbytes through transformers (GPT-2)")
            report["bnb_transformers"] = bnb_transformers_check(dev)
            print(json.dumps(report["bnb_transformers"], indent=2))

        print(f"== Training throughput (batch {args.batch}, {args.text_len} text tokens + 10 prefix)")
        train = {}
        jobs = [("gpt2", "lora", True), ("gpt2", "frozen", True), ("gpt2", "full", True),
                ("gpt2-medium", "lora", True), ("gpt2-medium", "lora", False)]
        for decoder, adaptation, is_train in jobs:
            for amp in AMP:
                key = f"{decoder}/{adaptation if is_train else 'forward'}/{amp}"
                try:
                    train[key] = bench_decoder(dev, decoder, adaptation, 512, args.batch, args.text_len, args.steps,
                                               args.warmup, amp, args.random_init, meter, idle_w, train=is_train)
                except Exception as e:
                    train[key] = err(e)
                print(f"  {key:28s} {train[key]}")
        report["train_throughput"] = train

        print("== Encoder throughput (inference, batch 64 unless --quick)")
        enc = {}
        for name in ENCODERS:
            for dname, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
                key = f"{name}/{dname}"
                try:
                    enc[key] = bench_encoder(dev, name, min(args.batch, 64), max(args.steps // 2, 2), args.warmup, dt,
                                             args.random_init)
                except Exception as e:
                    enc[key] = err(e)
                print(f"  {key:16s} {enc[key]}")
        report["encoder_throughput"] = enc

        est = grid_estimate(train, {k: v for k, v in enc.items() if k.endswith("/fp32")}, args.epochs, args.seeds)
        report["estimate"] = est
        print(f"== Training grid estimate ({args.epochs} epochs x {est['samples_per_epoch']:,} samples, "
              f"{args.seeds} seeds; excludes focused grid and CNN-LSTM)")
        for amp, e in est["by_precision"].items():
            print(f"  [{amp}] total {e['total_hours']} h = {e['total_days']} days, {e['total_kwh']} kWh, "
                  f"{e['total_co2_kg_at_india_grid']} kg CO2e")
            for row in e["rows"]:
                print(f"      {row['configs']:48s} runs {row['runs']:2d}  {row['hours']} h  {row['kwh']} kWh")
        print(f"  feature extraction (all images, fp32, model time only): {est['feature_extraction_hours']} h")

    out_dir = paths.RESULTS / "env"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"env_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
