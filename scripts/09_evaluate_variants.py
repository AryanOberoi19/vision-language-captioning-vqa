#!/usr/bin/env python3
"""Step 8a: accuracy of every inference configuration of the trained model (methodology §6, Table 2; scope of 2 Oct,
design agreed with Aryan on 3 Oct). Nothing is retrained.

    open scripts/variants_accuracy.command          # Terminal window: password once, then ~1.5-2 h
    sudo -v && python scripts/09_evaluate_variants.py
    python scripts/09_evaluate_variants.py --summary-only

Configurations (greenvl/precision.py), all of coco_ViT-L-14_lora8_lr1e-3_s0/epoch_02.pt:
  - fp32: the reference (captions beam 3, answers greedy), scored in step 6b;
  - enc_, dec_ and both_ x fp16, int8, nf4: the encoder, the decoder (mapping network + GPT-2 + adapters) or both
    at reduced precision, captions beam 3;
  - captions with greedy and beam 5 decoding at FP32 (answers are always greedy, so captions only).
Steps, each skipped when its output exists, so a stopped run resumes where it was:
  1. Karpathy test features (5,000 images) with the encoder at FP16, INT8 and NF4: 02_extract_features.py
     --precision, batch 64; time and energy per image are logged in results/features/.
  2. Every configuration scored on the full Karpathy test split by 05_evaluate.py --precision (BLEU-4, CIDEr,
     SPICE, CLIPScore, CHAIR, VQA accuracy by answer type, VQA-CE, 95 % bootstrap intervals).
  3. results/variants/<run>/<checkpoint>/accuracy.md and .json: one row per configuration, with each head's
     retention ratio against FP32 (methodology §9, RQ3).

Evaluation runs at batch 64 as in step 6b. LLM.int8 picks its outlier columns per batch, so INT8 outputs can differ
slightly between batch 64 and the batch-1 runs of 10_measure_variants.py; that script reports how often they agree.
Pausing: Ctrl+C; finished steps are kept and the step in progress resumes (feature shards, decoded text).
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
from greenvl.precision import (CONFIGS, PRECISIONS, REFERENCE, VARIANTS, config_tag, feature_name,  # noqa: E402
                               variant_name)

SCRIPTS = Path(__file__).resolve().parent
RUN = "coco_ViT-L-14_lora8_lr1e-3_s0"
CHECKPOINT = "epoch_02.pt"
SPLIT = "test"
PAUSED = 130
def eval_path(run: str, ckpt: str, config: str, decoding: str) -> Path:
    return paths.RESULTS / "eval" / run / f"{config_tag(Path(ckpt).stem, config)}_{SPLIT}_{decoding}.json"


def scored(run: str, ckpt: str, config: str, decoding: str) -> bool:
    """Scored on the whole split (a quick --limit check does not count and is overwritten)."""
    path = eval_path(run, ckpt, config, decoding)
    return path.exists() and json.loads(path.read_text()).get("limit") is None


def features_path(encoder: str, precision: str) -> Path:
    return paths.FEATURES / feature_name(encoder, precision).replace("/", "-") / f"coco_{SPLIT}.pt"


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
        print("\nPaused: progress is saved. To resume, open scripts/variants_accuracy.command.", flush=True)
        sys.exit(PAUSED)
    if rc != 0:
        sys.exit(f"{what} failed (exit {rc}); finished work is kept. Run this script again after fixing it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--idle-seconds", type=float, default=60, help="idle window before each feature extraction")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--allow-gpu-only-energy", action="store_true",
                    help="continue even if CPU and DRAM energy cannot be measured (no sudo)")
    args = ap.parse_args()
    out_dir = paths.RESULTS / "variants" / args.run / Path(args.checkpoint).stem
    if args.summary_only:
        return write_summary(args.run, args.checkpoint, out_dir)

    encoder = json.loads((paths.RESULTS / "runs" / args.run / "config.json").read_text())["encoder"]
    RunLock(out_dir / ".lock_accuracy")
    meter = EnergyMeter("mps")
    if meter.available and not meter.counters_live and not args.allow_gpu_only_energy:
        sys.exit("CPU and DRAM energy counters are not live. Run `sudo -v` and start this script again "
                 "(or pass --allow-gpu-only-energy).")
    if meter.keeper_pid:
        os.environ[KEEPER_ENV] = str(meter.keeper_pid)
    try:
        for p in PRECISIONS[1:]:
            if not features_path(encoder, p).exists():
                step([SCRIPTS / "02_extract_features.py", "--encoder", encoder, "--precision", p,
                      "--sets", f"coco_{SPLIT}", "--idle-seconds", args.idle_seconds],
                     f"Karpathy {SPLIT} features, {encoder} at {p.upper()}")
    finally:
        meter.close()
    for config, decoding in VARIANTS:
        if scored(args.run, args.checkpoint, config, decoding):
            continue
        cmd = [SCRIPTS / "05_evaluate.py", "--run", args.run, "--checkpoint", args.checkpoint, "--split", SPLIT,
               "--precision", config, "--decoding", decoding]
        if decoding != "beam3":
            cmd += ["--tasks", "caption"]
        step(cmd, f"scoring {variant_name(config, decoding)} on Karpathy {SPLIT}")
    write_summary(args.run, args.checkpoint, out_dir)


# ---------------------------------------------------------------- summary

def latest_extraction(encoder: str, precision: str) -> dict | None:
    """Batch-64 extraction log of the test features at this precision (newest), from results/features/."""
    slug = feature_name(encoder, precision).replace("/", "-")
    for f in sorted((paths.RESULTS / "features").glob(f"extract_{slug}_2*.json"), reverse=True):
        d = json.loads(f.read_text())
        if d.get("precision", "fp32") == precision and f"coco_{SPLIT}" in d.get("sets", {}):
            return {**d["sets"][f"coco_{SPLIT}"], "quantized_layers": d.get("quantized_layers")}
    return None


def write_summary(run: str, ckpt: str, out_dir: Path):
    rows = []
    for config, decoding in VARIANTS:
        path = eval_path(run, ckpt, config, decoding)
        if not scored(run, ckpt, config, decoding):
            continue
        r = json.loads(path.read_text())
        c, v = r.get("caption", {}), r.get("vqa")
        row = {"variant": variant_name(config, decoding), "config": config, "decoding": decoding,
               "encoder_precision": CONFIGS[config][0], "decoder_precision": CONFIGS[config][1]}
        for k in ("BLEU-4", "CIDEr", "SPICE", "CLIPScore", "CHAIR_i", "CHAIR_s"):
            if k in c:
                row[k] = {"value": 100 * c[k]["value"], "ci95": [100 * x for x in c[k]["ci95"]]}
        row["words"] = c.get("mean_length_words")
        row["caption_decode_seconds_b64"] = c.get("decode_seconds")
        if v:
            row["VQA"] = {"value": 100 * v["accuracy"]["value"], "ci95": [100 * x for x in v["accuracy"]["ci95"]]}
            row["VQA_by_type"] = {t: 100 * m["value"] for t, m in v["by_answer_type"].items()}
            if "vqa_ce" in v:
                row["VQA-CE"] = {"value": 100 * v["vqa_ce"]["value"],
                                 "ci95": [100 * x for x in v["vqa_ce"]["ci95"]]}
            row["answer_decode_seconds_b64"] = v.get("decode_seconds")
        rows.append(row)
    ref = next((r for r in rows if r["variant"] == REFERENCE), None)
    for r in rows:  # retention ratios R_h(c) = m_h(c) / m_h(c0) (methodology §9, RQ3)
        if ref and "CIDEr" in r:
            r["R_cap"] = r["CIDEr"]["value"] / ref["CIDEr"]["value"]
        if ref and "VQA" in r:
            r["R_vqa"] = r["VQA"]["value"] / ref["VQA"]["value"]
        if "R_cap" in r and "R_vqa" in r:
            r["delta_R"] = r["R_vqa"] - r["R_cap"]
    encoder = json.loads((paths.RESULTS / "runs" / run / "config.json").read_text())["encoder"]
    extraction = {p: latest_extraction(encoder, p) for p in PRECISIONS}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accuracy.json").write_text(json.dumps(
        {"run": run, "checkpoint": ckpt, "split": SPLIT, "rows": rows, "feature_extraction_b64": extraction,
         "time": datetime.now().isoformat(timespec="seconds")}, indent=1))

    def ci(m):
        return "-" if m is None else f"{m['value']:.1f} [{m['ci95'][0]:.1f}, {m['ci95'][1]:.1f}]"

    def num(x, d=1):
        return "-" if x is None else f"{x:.{d}f}"
    lines = [f"# Accuracy per configuration, {run} {ckpt}, Karpathy {SPLIT} (95 % intervals)\n",
             "| Configuration | Encoder | Decoder | Captions | CIDEr | BLEU-4 | SPICE | CLIPScore | CHAIR_i | CHAIR_s "
             "| Words | VQA | yes/no | number | other | VQA-CE | R_cap | R_vqa | ΔR |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r.get("VQA_by_type", {})
        lines.append(
            f"| {r['variant']} | {r['encoder_precision']} | {r['decoder_precision']} | {r['decoding']} | "
            f"{ci(r.get('CIDEr'))} | {num(r.get('BLEU-4', {}).get('value'))} | {num(r.get('SPICE', {}).get('value'))} | "
            f"{num(r.get('CLIPScore', {}).get('value'))} | {ci(r.get('CHAIR_i'))} | "
            f"{num(r.get('CHAIR_s', {}).get('value'))} | {num(r.get('words'))} | {ci(r.get('VQA'))} | "
            f"{num(t.get('yes/no'))} | {num(t.get('number'))} | {num(t.get('other'))} | {ci(r.get('VQA-CE'))} | "
            f"{num(r.get('R_cap'), 3)} | {num(r.get('R_vqa'), 3)} | {num(r.get('delta_R'), 3)} |")
    lines += ["\nR_cap = CIDEr / CIDEr(FP32), R_vqa = VQA accuracy / VQA accuracy(FP32), ΔR = R_vqa - R_cap "
              "(methodology §9, RQ3; only configurations that change both heads enter the test).",
              "\n## Encoder at batch 64 (test feature extraction, incl. JPEG decoding)\n",
              "| Encoder precision | Quantized layers | Images/s | J/image above idle |", "|---|---|---|---|"]
    for p, e in extraction.items():
        if e:
            lines.append(f"| {p} | {e.get('quantized_layers') if e.get('quantized_layers') is not None else 0} | "
                         f"{e.get('images_per_s')} | {e.get('j_per_image_above_idle', '-')} |")
    (out_dir / "accuracy.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPaused: finished steps are saved. Open scripts/variants_accuracy.command to resume.", flush=True)
        sys.exit(130)
