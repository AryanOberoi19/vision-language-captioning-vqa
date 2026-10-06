#!/usr/bin/env python3
"""VizWiz-Captions zero-shot check (methodology §4: the reference and the frontier configurations, as a check
against the assistive use case; agreed with Aryan, 4 Oct). Captions only; nothing is trained.

    open scripts/vizwiz.command                     # ~1 h, no password
    python scripts/12_vizwiz.py --summary-only
    python scripts/12_vizwiz.py --configs fp32 --limit 100     # code check (extracts the FP32 features in full)

Configurations: FP32 (reference) and the step-10 frontier configurations: FP16 on both parts (energy), FP32 with
greedy decoding (latency), NF4 and INT8 on both parts (size and memory).
Steps, each skipped when its output exists:
  1. ViT-L/14 features of the 7,750 VizWiz val images at FP32, FP16, INT8 and NF4 (02_extract_features.py
     --precision, batch 64). The energy of these runs is not part of the study, so no password is needed.
  2. Each configuration's captions for the 7,542 images that keep at least one reference after the precanned and
     rejected captions are removed (step 2), scored by 05_evaluate.py --data vizwiz: BLEU-4, CIDEr, SPICE,
     CLIPScore and length with 95 % bootstrap intervals. No CHAIR: VizWiz has no object annotations.
  3. results/vizwiz/<run>/<checkpoint>/summary.md and .json: VizWiz next to Karpathy test per configuration, and
     each configuration's CIDEr difference to FP32 on VizWiz with a paired 95 % bootstrap interval (the step-10
     rule); figures/vizwiz_examples: six images (seed 0) with the FP32 caption and two references.
"""
import argparse
import json
import random
import subprocess
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.analysis import boot_mean, bootstrap_weights, ci95  # noqa: E402
from greenvl.lock import RunLock  # noqa: E402
from greenvl.metrics import cider, ptb_tokenize  # noqa: E402
from greenvl.precision import CONFIGS, REFERENCE, config_tag, feature_name, variant_name  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
RUN = "coco_ViT-L-14_lora8_lr1e-3_s0"
CHECKPOINT = "epoch_02.pt"
PAUSED = 130
# the reference and the step-10 frontier configurations (results/analysis/.../frontiers_m4pro.md)
VIZWIZ_VARIANTS = [(REFERENCE, "beam3"), ("both_fp16", "beam3"), (REFERENCE, "greedy"), ("both_nf4", "beam3"),
                   ("both_int8", "beam3")]
WHY = {REFERENCE: "reference", "both_fp16": "energy frontier", f"{REFERENCE}_greedy": "latency frontier",
       "both_nf4": "size / memory frontier", "both_int8": "size / memory frontier"}
SURFACE, INK, INK2 = "#ffffff", "#0b0b0b", "#52514e"


def eval_path(run, ckpt, config, decoding, data="vizwiz_val") -> Path:
    return paths.RESULTS / "eval" / run / f"{config_tag(Path(ckpt).stem, config)}_{data}_{decoding}.json"


def scored(run, ckpt, config, decoding) -> bool:
    p = eval_path(run, ckpt, config, decoding)
    return p.exists() and json.loads(p.read_text()).get("limit") is None


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
    if rc in (PAUSED, -2):  # 05_evaluate.py ends on SIGINT (-2) rather than 130
        print("\nPaused: progress is saved. To resume, open scripts/vizwiz.command.", flush=True)
        sys.exit(PAUSED)
    if rc != 0:
        sys.exit(f"{what} failed (exit {rc}); finished work is kept. Run this script again after fixing it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--configs", nargs="+", default=None,
                    help="variant names to evaluate (default: all five), e.g. fp32 both_fp16 fp32_greedy")
    ap.add_argument("--limit", type=int, default=None, help="first N images (code check; outputs are overwritten "
                                                            "by the full run)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out_dir = paths.RESULTS / "vizwiz" / args.run / Path(args.checkpoint).stem
    variants = [(c, d) for c, d in VIZWIZ_VARIANTS if not args.configs or variant_name(c, d) in args.configs]
    if args.summary_only:
        return write_summary(args.run, args.checkpoint, out_dir)

    encoder = json.loads((paths.RESULTS / "runs" / args.run / "config.json").read_text())["encoder"]
    RunLock(paths.RESULTS / "vizwiz" / ".lock")
    for p in dict.fromkeys(CONFIGS[c][0] for c, _ in variants):
        f = paths.FEATURES / feature_name(encoder, p).replace("/", "-") / "vizwiz_val.pt"
        if not f.exists():
            step([SCRIPTS / "02_extract_features.py", "--encoder", encoder, "--precision", p, "--sets", "vizwiz_val",
                  "--idle-seconds", 0], f"VizWiz val features, {encoder} at {p.upper()}")
    for c, d in variants:
        if args.limit is None and scored(args.run, args.checkpoint, c, d):
            continue
        cmd = [SCRIPTS / "05_evaluate.py", "--run", args.run, "--checkpoint", args.checkpoint, "--data", "vizwiz",
               "--split", "val", "--precision", c, "--decoding", d, "--tasks", "caption"]
        step(cmd + (["--limit", args.limit] if args.limit else []), f"VizWiz captions, {variant_name(c, d)}")
    if args.limit is None:
        write_summary(args.run, args.checkpoint, out_dir)


# ---------------------------------------------------------------- summary

def references(family: str, split: str) -> dict:
    d = json.loads((paths.PROCESSED / f"{family}_refs_{split}.json").read_text())
    refs = defaultdict(list)
    for a in d["annotations"]:
        refs[a["image_id"]].append(a["caption"])
    return dict(refs)


def write_summary(run: str, ckpt: str, out_dir: Path):
    done = [(c, d) for c, d in VIZWIZ_VARIANTS if scored(run, ckpt, c, d)]
    if not done:
        print("no complete VizWiz evaluations yet")
        return
    refs = references("vizwiz", "val")
    ids = sorted(refs)
    caps = {}
    for c, d in done:
        p = eval_path(run, ckpt, c, d).with_name(eval_path(run, ckpt, c, d).stem + "_captions.json")
        caps[variant_name(c, d)] = {x["image_id"]: x["caption"] for x in json.loads(p.read_text())["captions"]}
    print(f"PTB tokenisation for {len(caps)} configurations", flush=True)
    tok = ptb_tokenize({**{("g", i): refs[i] for i in ids}, **{(v, i): [caps[v][i]] for v in caps for i in ids}})
    gts = {i: tok[("g", i)] for i in ids}
    per = {v: cider(gts, {i: tok[(v, i)] for i in ids})[1] for v in caps}
    W = bootstrap_weights(len(ids))
    boot = {v: 100 * boot_mean(per[v], W) for v in per}

    rows = []
    for c, d in done:
        v = variant_name(c, d)
        vz = json.loads(eval_path(run, ckpt, c, d).read_text())["caption"]
        coco_p = eval_path(run, ckpt, c, d, data="test")
        coco = json.loads(coco_p.read_text())["caption"] if coco_p.exists() else {}
        row = {"variant": v, "why": WHY.get(v, ""), "images": vz["images"]}
        for k in ("BLEU-4", "CIDEr", "SPICE", "CLIPScore"):
            if k in vz:
                row[k] = {"value": 100 * vz[k]["value"], "ci95": [100 * x for x in vz[k]["ci95"]]}
            if k in coco:
                row[f"coco_{k}"] = 100 * coco[k]["value"]
        row["words"] = vz.get("mean_length_words")
        if v != REFERENCE and REFERENCE in boot:
            diff = boot[v] - boot[REFERENCE]
            row["CIDEr_vs_fp32"] = {"value": float(100 * (per[v].mean() - per[REFERENCE].mean())), "ci95": ci95(diff)}
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"run": run, "checkpoint": ckpt, "images": len(ids),
                                                     "rows": rows, "time": datetime.now().isoformat(timespec="seconds")},
                                                    indent=1))

    def ci(m):
        return "-" if not m else f"{m['value']:.1f} [{m['ci95'][0]:.1f}, {m['ci95'][1]:.1f}]"

    def n1(x):
        return "-" if x is None else f"{x:.1f}"
    lines = [f"# VizWiz-Captions val, zero-shot ({run} {ckpt}; {len(ids):,} images with references)\n",
             "Trained on COCO + VQA v2 only. Karpathy test values in brackets for comparison. CIDEr vs FP32: paired 95 % "
             "bootstrap interval over the VizWiz images (the step-10 rule).\n",
             "| Configuration | Why it is here | CIDEr [CI] (COCO) | BLEU-4 (COCO) | SPICE (COCO) | CLIPScore (COCO) | Words "
             "| CIDEr vs FP32 [CI] |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        vs = r.get("CIDEr_vs_fp32")
        lines.append(f"| {r['variant']} | {r['why']} | {ci(r.get('CIDEr'))} ({n1(r.get('coco_CIDEr'))}) | "
                     f"{n1((r.get('BLEU-4') or {}).get('value'))} ({n1(r.get('coco_BLEU-4'))}) | "
                     f"{n1((r.get('SPICE') or {}).get('value'))} ({n1(r.get('coco_SPICE'))}) | "
                     f"{n1((r.get('CLIPScore') or {}).get('value'))} ({n1(r.get('coco_CLIPScore'))}) | {n1(r.get('words'))} | "
                     + (f"{vs['value']:+.1f} [{vs['ci95'][0]:+.1f}, {vs['ci95'][1]:+.1f}] |" if vs else "- |"))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    if REFERENCE in caps:
        examples_figure(refs, caps[REFERENCE], out_dir)


def examples_figure(refs: dict, caps: dict, out_dir: Path, n: int = 6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    files = {r["image_id"]: r["file"] for r in json.loads((paths.PROCESSED / "vizwiz_val_images.json").read_text())}
    ids = sorted(refs)
    pick = sorted(random.Random(0).sample(ids, n))
    fig, axes = plt.subplots(2, 3, figsize=(10, 8.4))
    for ax, i in zip(axes.flat, pick):
        with Image.open(paths.VIZWIZ_IMAGES / files[i]) as im:
            im = im.convert("RGB")
            im.thumbnail((512, 512))
            ax.imshow(np.asarray(im))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        text = ["Model: " + "\n".join(textwrap.wrap(caps[i], 40))]
        text += ["Ref: " + "\n".join(textwrap.wrap(r, 40)) for r in refs[i][:2]]
        ax.set_xlabel("\n".join(text), fontsize=7.5, color=INK, loc="left")
    fig.suptitle("VizWiz-Captions val, zero-shot: FP32 captions and two human references (6 random images)",
                 x=0.02, ha="left", fontsize=10, color=INK)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    d = out_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"vizwiz_examples.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPaused: finished steps are saved. Open scripts/vizwiz.command to resume.", flush=True)
        sys.exit(130)
