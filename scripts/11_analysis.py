#!/usr/bin/env python3
"""Step 10: analysis of the inference configurations (methodology §9; plan agreed with Aryan, 3 Oct 2026).

    open scripts/analysis.command                               # every part, ~45-60 min, no password
    python scripts/11_analysis.py --parts frontier rq3 batch    # selected parts
    python scripts/11_analysis.py --limit 200 --label _smoke --no-spice --attribution-images 2   # code check
    python scripts/11_analysis.py --platform t4 --platform-name "Colab T4" --parts frontier batch \
        --energy results/variants/<run>/epoch_02_t4/energy.json      # second platform (colab/colab_t4.ipynb)
    python scripts/11_analysis.py --parts frontier --energy-basis incl         # energy including idle (*_inclidle)

Inputs: the step-8 evaluation outputs (results/eval/<run>/, Karpathy test) and cost measurements
(results/variants/<run>/<checkpoint>/energy.json, memory.json, energy.jsonl).

Parts (results/analysis/<run>/<checkpoint>/):
  frontier  RQ1. Accuracy-cost frontiers on one platform: CIDEr vs energy per caption (primary); SPICE and CHAIR_i
            vs energy; CIDEr vs latency, size and memory; VQA accuracy vs energy, latency and size. Energy is per item
            above idle (methodology §8). Dominance (agreed 3 Oct; with one training seed the seed-based rule of §9
            cannot be used): a dominates b if it is not worse on either axis and better on at least one. An accuracy
            difference counts when the paired 95 % bootstrap interval of the difference (both configurations scored
            on the same resampled test images; 1,000 resamples) excludes zero. A cost difference counts when it
            exceeds the pooled standard deviation over the repetitions (energy, latency), or 1 % (size and memory,
            which do not vary between repetitions).
  rq2       Hallucination. CHAIR_i and CHAIR_s by caption length (<= 8, 9-10, >= 11 words); visual-grounding score of
            every object word, hallucinated vs correctly mentioned, each configuration's model at its own precision
            scoring its own captions.
  rq3       Retention R_cap = CIDEr / CIDEr(FP32), R_vqa likewise, ΔR = R_vqa - R_cap for the 9 precision
            configurations (decoding applies to captions only); Wilcoxon signed-rank test across them; bootstrap
            intervals with images resampled for both heads together. Grounding score of every answer.
  attribution  Where the model looks, for 8 fixed test images (seed 0): two object words of each caption and the
            answer to one question, FP32, qualitative only. Occlusion maps (a 28 x 28 px window greyed out at every
            14 px step; how much the word's log-probability drops). The methodology names Grad-CAM; gradient-based
            maps (Grad-CAM at several layers, gradient x activation, attention relevance) all concentrate on a few
            high-norm background tokens that CLIP ViT-L/14 uses as global stores (Darcet et al., 2024), so occlusion
            is used and attribution_methods.png shows the comparison.
  batch     Batch 1 (the energy runs) vs batch 64 (the evaluation): accuracy of both on the same 500 captions and
            500 answers, per configuration.

CNN-LSTM baseline (13_baseline.py), when its results exist: one more point on the caption frontiers of the Mac (not on
the VQA ones, it has no VQA head; not on another platform's, where it was not measured). Its accuracy is computed from its saved captions like every configuration's. Its energy
and latency were measured in another session, next to the FP32 reference, so they enter as the ratio to that
session's FP32 (per repetition) times the step-10 FP32 value; the sd scales the same way. Size and memory are
absolute. --no-baseline leaves it out.

Grounding score (§9): g(y) = log p(y | context, I) - log p(y | context, Ī), with I the image's own visual prefix and Ī
the mean prefix over 10,000 training images (seed 0; FP32 features through each configuration's own mapping network).
Words spanning several tokens score the sum over their tokens; an answer scores the sum over its tokens.
"""
import argparse
import json
import random
import sys
import textwrap
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl.device import empty_cache, get_device  # noqa: E402  (first: sets MPS memory limits)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.analysis import (boot_mean, boot_ratio, bootstrap_weights, caption_tokens, ci95, mean_prefix,  # noqa: E402
                              object_mentions, token_logprobs, word_count)
from greenvl.chair import Chair  # noqa: E402
from greenvl.data import MAX_ANSWER_TOKENS, FeatureStore, load_tokenizer  # noqa: E402
from greenvl.decode import question_prompt  # noqa: E402
from greenvl.metrics import cider, ptb_tokenize, spice  # noqa: E402
from greenvl.precision import (CONFIGS, REFERENCE, VARIANTS, config_tag, feature_name, load_decoder,  # noqa: E402
                               variant_name)

RUN = "coco_ViT-L-14_lora8_lr1e-3_s0"
CHECKPOINT = "epoch_02.pt"
SPLIT = "test"
LENGTH_BINS = [("<= 8 words", 0, 8), ("9-10 words", 9, 10), (">= 11 words", 11, 10**6)]
PREFIX_SAMPLE = 10_000
CORRECT_ANSWER = 0.9  # VQA accuracy at or above this counts as a correct answer (wrong: 0)

# Figures: chart surface, ink and the precision palette (dataviz skill reference palette, light mode; slots 1-3
# validated all-pairs on white: CVD ΔE 9.2, normal-vision 24.0). FP32 is the reference and is drawn in ink.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PRECISION_COLOR = {"fp32": INK, "fp16": "#2a78d6", "int8": "#eb6834", "nf4": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834"]  # two-series charts (correct vs hallucinated)
HEAT = "#eb6834"  # Grad-CAM overlay: one hue, opacity carries the magnitude
BASELINE = "showtell"  # the CNN-LSTM baseline (13_baseline.py)
DEFAULT_PLATFORM = "m4pro"
# Energy basis of the frontiers: above idle (methodology §8's E_out, the default) or including idle (--energy-basis
# incl): at batch 1 a GPU that stays in its performance state while the model is loaded (the T4: ~33 W) spends most of
# each item's energy on idle draw, which a deployment serving one image at a time pays.
ENERGY_BASES = {"above": ("j_above", "rel_fp32_above", "above idle"), "incl": ("j", "rel_fp32", "incl. idle")}
ENERGY_KEY = "j_above"
BASELINE_DIR = paths.RESULTS / "baseline" / "showtell_resnet50_s0"
BASELINE_COLOR = "#4a3aa7"  # reference palette slot 7 (violet): a different model family, not a precision


# ---------------------------------------------------------------- names and encodings

def display_name(variant: str) -> str:
    if variant == BASELINE:
        return "Show and Tell"
    if variant == REFERENCE:
        return "FP32"
    if variant.startswith(REFERENCE + "_"):
        return "FP32 " + {"greedy": "greedy", "beam5": "beam 5"}[variant.split("_", 1)[1]]
    target, prec = variant.split("_")
    return f"{ {'enc': 'Encoder', 'dec': 'Decoder', 'both': 'Both'}[target]} {prec.upper()}"


def variant_style(variant: str) -> tuple[str, str]:
    """(colour, marker): colour = precision, marker = what it is applied to (or the decoding variant)."""
    if variant == BASELINE:
        return BASELINE_COLOR, "*"
    if variant == REFERENCE:
        return PRECISION_COLOR["fp32"], "s"
    if variant.startswith(REFERENCE + "_"):
        return PRECISION_COLOR["fp32"], {"greedy": "D", "beam5": "P"}[variant.split("_", 1)[1]]
    target, prec = variant.split("_")
    return PRECISION_COLOR[prec], {"enc": "^", "dec": "v", "both": "o"}[target]


# ---------------------------------------------------------------- data

class Data:
    """Step-8 evaluation outputs for one run and checkpoint on Karpathy test."""

    def __init__(self, run: str, checkpoint: str, limit: int | None, baseline: bool = True):
        self.run, self.checkpoint, self.stem = run, checkpoint, Path(checkpoint).stem
        self.eval_dir = paths.RESULTS / "eval" / run
        cfg = json.loads((paths.RESULTS / "runs" / run / "config.json").read_text())
        self.encoder, self.decoder = cfg["encoder"], cfg["decoder"]
        refs = defaultdict(list)
        for a in json.loads((paths.PROCESSED / f"coco_refs_{SPLIT}.json").read_text())["annotations"]:
            refs[a["image_id"]].append(a["caption"])
        self.refs = dict(refs)
        self.all_image_ids = sorted(self.refs)
        self.image_ids = self.all_image_ids[:limit] if limit else self.all_image_ids
        self.index = {i: k for k, i in enumerate(self.image_ids)}
        self.variants = [variant_name(c, d) for c, d in VARIANTS
                         if (self.eval_dir / f"{config_tag(self.stem, c)}_{SPLIT}_{d}_captions.json").exists()]
        self.captions, self.summary = {}, {}
        for c, d in VARIANTS:
            v = variant_name(c, d)
            if v not in self.variants:
                continue
            tag = config_tag(self.stem, c)
            caps = json.loads((self.eval_dir / f"{tag}_{SPLIT}_{d}_captions.json").read_text())["captions"]
            self.captions[v] = {x["image_id"]: x["caption"] for x in caps}
            self.summary[v] = json.loads((self.eval_dir / f"{tag}_{SPLIT}_{d}.json").read_text())
        self.extra = []  # the CNN-LSTM baseline: caption frontiers only (RQ2, RQ3 and the batch check skip it)
        base_caps = BASELINE_DIR / f"{SPLIT}_beam3_captions.json"
        if baseline and base_caps.exists() and (BASELINE_DIR / f"{SPLIT}_beam3.json").exists():
            self.extra = [BASELINE]
            self.captions[BASELINE] = {x["image_id"]: x["caption"] for x in json.loads(base_caps.read_text())["captions"]}
            self.summary[BASELINE] = json.loads((BASELINE_DIR / f"{SPLIT}_beam3.json").read_text())
        self.answers = {}
        for c in CONFIGS:
            f = self.eval_dir / f"{config_tag(self.stem, c)}_{SPLIT}_answers.json"
            if f.exists():
                self.answers[c] = {a["question_id"]: (a["answer"], a["accuracy"])
                                   for a in json.loads(f.read_text())["answers"]}
        keep = set(self.image_ids)
        qs = json.loads((paths.PROCESSED / f"vqa_{SPLIT}_questions.json").read_text())["questions"]
        self.questions = sorted((q for q in qs if q["image_id"] in keep), key=lambda q: q["question_id"])
        self.anns = {a["question_id"]: a for a in
                     json.loads((paths.PROCESSED / f"vqa_{SPLIT}_annotations.json").read_text())["annotations"]}
        self.chair = Chair(paths.DATA_ROOT / "chair" / "synonyms.txt")
        gt = json.loads((paths.RESULTS / "eval" / f"_chair_gt_coco_{SPLIT}.json").read_text())  # 05_evaluate's cache
        self.gt = {int(k): set(v) for k, v in gt.items()}

    def config_of(self, variant: str) -> tuple[str, str]:
        if variant == BASELINE:
            return BASELINE, "beam3"
        return next((c, d) for c, d in VARIANTS if variant_name(c, d) == variant)


# ---------------------------------------------------------------- per-image metrics (cached)

def per_image_metrics(D: Data, out_dir: Path, spice_on: bool):
    """({variant: {CIDEr, SPICE (if on), hall, mentions, words}}, {config: (VQA num, den)}), arrays over D.image_ids.
    CIDEr and CHAIR are recomputed from the saved captions (checked against the evaluation's values); SPICE runs once
    per configuration (~1 min each) and is cached."""
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    n = len(D.image_ids)
    path = cache / f"per_image_n{n}.npz"
    scored = D.variants + D.extra
    M = {v: {} for v in scored}
    if path.exists():
        z = np.load(path)
        for key in z.files:
            v, metric = key.split("|")
            if v in M:
                M[v][metric] = z[key]
    for v in scored:
        f = cache / f"spice_{v}_n{n}.npy"
        if f.exists():
            M[v]["SPICE"] = np.load(f)
    need = [v for v in scored if "CIDEr" not in M[v] or (spice_on and "SPICE" not in M[v])]
    if need:
        print(f"PTB tokenisation for {len(need)} configurations", flush=True)
        tok = ptb_tokenize({**{("g", i): D.refs[i] for i in D.image_ids},
                            **{(v, i): [D.captions[v][i]] for v in need for i in D.image_ids}})
        gts = {i: tok[("g", i)] for i in D.image_ids}
    for v in need:
        res = {i: tok[(v, i)] for i in D.image_ids}
        if "CIDEr" not in M[v]:
            score, M[v]["CIDEr"] = cider(gts, res)
            ref_val = D.summary[v]["caption"]["CIDEr"]["value"]
            if n == len(D.all_image_ids) and abs(score - ref_val) > 1e-6:
                print(f"  WARNING {v}: CIDEr {score:.6f} differs from the evaluation's {ref_val:.6f}", flush=True)
            hall, ment, words, mismatch = np.zeros(n), np.zeros(n), np.zeros(n), 0
            for k, i in enumerate(D.image_ids):
                cap = D.captions[v][i]
                ms = object_mentions(D.chair, cap)
                mismatch += [m[1] for m in ms] != D.chair.caption_objects(cap)[1]
                ment[k], hall[k], words[k] = len(ms), sum(m[1] not in D.gt[i] for m in ms), word_count(cap)
            if mismatch:
                print(f"  WARNING {v}: object mentions differ from CHAIR's in {mismatch} captions", flush=True)
            M[v].update({"hall": hall, "mentions": ment, "words": words})
        if spice_on and "SPICE" not in M[v]:
            print(f"  SPICE {display_name(v)} ({datetime.now():%H:%M})", flush=True)
            _, M[v]["SPICE"] = spice(gts, res)
            np.save(cache / f"spice_{v}_n{n}.npy", M[v]["SPICE"])
    np.savez(path, **{f"{v}|{m}": a for v, d in M.items() for m, a in d.items() if m != "SPICE"})
    vqa = {}
    for c, ans in D.answers.items():
        num, den = np.zeros(n), np.zeros(n)
        for q in D.questions:
            k = D.index[q["image_id"]]
            num[k] += ans[q["question_id"]][1]
            den[k] += 1
        vqa[c] = (num, den)
    return M, vqa


# ---------------------------------------------------------------- figures: shared styling

def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
        "text.color": INK, "pdf.fonttype": 42})
    return plt


def save(fig, out_dir: Path, name: str):
    d = out_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def legend_handles(variants: list[str]):
    from matplotlib.lines import Line2D

    handles = []
    for v in variants:
        color, marker = variant_style(v)
        handles.append(Line2D([], [], ls="none", marker=marker, ms=7, mfc=color, mec=color, label=display_name(v)))
    handles.append(Line2D([], [], ls="none", marker="o", ms=7, mfc=SURFACE, mec=INK2, mew=1.4,
                          label="hollow = dominated"))
    return handles


def variant_legend(ax, variants: list[str], loc="upper center", anchor=(0.5, -0.16), ncol=4):
    ax.legend(handles=legend_handles(variants), loc=loc, bbox_to_anchor=anchor, ncol=ncol, handletextpad=0.3,
              columnspacing=1.2)


def figure_legend(fig, variants: list[str], ncol=7):
    fig.legend(handles=legend_handles(variants), loc="lower center", ncol=ncol, handletextpad=0.3,
               columnspacing=1.2)


# ---------------------------------------------------------------- RQ1: frontiers

def load_costs(energy_path: Path) -> dict:
    """{(config, task, decoding): row} from 10_measure_variants.py's energy.json."""
    rows = json.loads(energy_path.read_text())["rows"]
    return {(r["config"], r["task"], r["decoding"]): r for r in rows}


def baseline_costs(costs: dict, basis: str = "above") -> dict:
    """The CNN-LSTM baseline's cost row in step-10 terms (see the module docstring): energy and latency as the ratio
    to its own session's FP32 times the step-10 FP32 value. Empty if its measurements are missing."""
    path = BASELINE_DIR / "summary.json"
    fp32 = costs.get((REFERENCE, "caption", "beam3"))
    if not path.exists() or fp32 is None:
        return {}
    s = json.loads(path.read_text())
    b, mem = (s.get("cost") or {}).get("baseline"), s.get("memory") or {}
    key, rel, _ = ENERGY_BASES[basis]
    if not b or not b.get(rel) or b[rel][0] is None:
        return {}

    def scaled(rel, ref):
        if not rel or rel[0] is None:
            return [None, None]
        return [rel[0] * ref[0], rel[1] * ref[0] if rel[1] is not None else None]
    return {(BASELINE, "caption", "beam3"): {
        "config": BASELINE, "task": "caption", "decoding": "beam3",
        key: scaled(b[rel], fp32[key]),
        "lat_median": scaled(b.get("rel_lat_median"), fp32["lat_median"]),
        "size_mb": sum((mem.get("size_mb") or {}).values()) or None, "memory_mb": mem.get("peak_tensor_mb"),
        "measured": {k: b.get(k) for k in ("j_above", "j", "rel_fp32_above", "lat_median", "rel_lat_median")}}}


def dominance(points: list[dict], higher_is_better: bool, cost_kind: str) -> None:
    """Fills point['dominated_by'] and point['frontier'] (see the module docstring for the rule)."""
    for a in points:
        a["dominated_by"] = []
    for a in points:
        for b in points:
            if a is b:
                continue
            lo, hi = ci95(a["boot"] - b["boot"])
            if not higher_is_better:
                lo, hi = -hi, -lo  # positive = a better
            acc_better, acc_worse = lo > 0, hi < 0
            if cost_kind in ("energy", "latency"):
                thr = ((a["cost_sd"] or 0) ** 2 / 2 + (b["cost_sd"] or 0) ** 2 / 2) ** 0.5
            else:
                thr = 0.01 * max(a["cost"], b["cost"])
            saving = b["cost"] - a["cost"]  # positive = a cheaper
            cost_better, cost_worse = saving > thr, saving < -thr
            if not acc_worse and not cost_worse and (acc_better or cost_better):
                b["dominated_by"].append(a["variant"])
    for p in points:
        p["frontier"] = not p["dominated_by"]


FRONTIERS = [  # name, task, accuracy metric, higher is better, cost kind, axis labels
    ("captions_energy", "caption", "CIDEr", True, "energy", "Energy per caption above idle (J)", "CIDEr"),
    ("captions_spice_energy", "caption", "SPICE", True, "energy", "Energy per caption above idle (J)", "SPICE"),
    ("captions_chair_energy", "caption", "CHAIR_i", False, "energy", "Energy per caption above idle (J)",
     "CHAIR_i (lower is better)"),
    ("captions_latency", "caption", "CIDEr", True, "latency", "Median latency per caption (ms)", "CIDEr"),
    ("captions_size", "caption", "CIDEr", True, "size", "Model size (MiB)", "CIDEr"),
    ("captions_memory", "caption", "CIDEr", True, "memory", "GPU memory held by tensors (MB)", "CIDEr"),
    ("answers_energy", "vqa", "VQA", True, "energy", "Energy per answer above idle (J)", "VQA accuracy"),
    ("answers_latency", "vqa", "VQA", True, "latency", "Median latency per answer (ms)", "VQA accuracy"),
    ("answers_size", "vqa", "VQA", True, "size", "Model size (MiB)", "VQA accuracy"),
]


def frontier_points(spec, D, M, vqa, W, costs) -> list[dict]:
    name, task, metric, higher, kind = spec[:5]
    pts = []
    for v in D.variants + D.extra:
        c, d = D.config_of(v)
        if task == "vqa" and (d != "beam3" or c not in vqa):
            continue
        row = costs.get((c, task, "greedy" if task == "vqa" else d))
        if row is None:
            continue
        if metric == "VQA":
            num, den = vqa[c]
            value, boot = num.sum() / den.sum(), boot_ratio(num, den, W)
        elif metric == "CHAIR_i":
            value = M[v]["hall"].sum() / max(M[v]["mentions"].sum(), 1)
            boot = boot_ratio(M[v]["hall"], M[v]["mentions"], W)
        else:
            if metric not in M[v]:
                return []
            value, boot = M[v][metric].mean(), boot_mean(M[v][metric], W)
        cost, sd = {"energy": tuple(row[ENERGY_KEY]), "latency": tuple(row["lat_median"]),
                    "size": (row["size_mb"], None), "memory": (row.get("memory_mb"), None)}[kind]
        if cost is None:
            continue
        pts.append({"variant": v, "value": 100 * value, "boot": 100 * boot, "ci95": ci95(100 * boot),
                    "cost": cost, "cost_sd": sd})
    dominance(pts, higher, kind)
    return pts


def plot_frontier(ax, pts, spec, label_frontier=True):
    name, task, metric, higher, kind, xlabel, ylabel = spec
    for p in pts:
        color, marker = variant_style(p["variant"])
        ax.errorbar(p["cost"], p["value"], xerr=p["cost_sd"] or 0,
                    yerr=[[p["value"] - p["ci95"][0]], [p["ci95"][1] - p["value"]]],
                    fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=0, zorder=2)
        ax.plot(p["cost"], p["value"], ls="none", marker=marker, ms=8, mew=1.6, mec=color,
                mfc=color if p["frontier"] else SURFACE, zorder=4)
    front = sorted((p for p in pts if p["frontier"]), key=lambda p: p["cost"])
    if len(front) > 1:
        ax.plot([p["cost"] for p in front], [p["value"] for p in front], color=INK2, lw=1.0, zorder=3)
    if label_frontier:
        lo, hi = ax.get_xlim()
        for p in front:
            near_left = (p["cost"] - lo) < 0.12 * (hi - lo)  # keep the label clear of the y tick labels
            ax.annotate(display_name(p["variant"]), (p["cost"], p["value"]), textcoords="offset points",
                        xytext=(6, 8) if near_left else (0, 10), ha="left" if near_left else "center",
                        va="bottom", fontsize=8, color=INK, zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def part_frontier(D, args, out_dir, M, vqa, W):
    global ENERGY_KEY
    plt = setup_matplotlib()
    ENERGY_KEY, _, basis_label = ENERGY_BASES[args.energy_basis]
    specs = [tuple(x.replace("above idle", basis_label) if isinstance(x, str) else x for x in spec)
             for spec in FRONTIERS]
    costs = load_costs(args.energy)
    if args.platform == DEFAULT_PLATFORM:  # the baseline was measured next to FP32 on the Mac only
        costs.update(baseline_costs(costs, args.energy_basis))
    slug = args.platform + ("" if args.energy_basis == "above" else "_inclidle")
    results = {}
    lines = [f"# RQ1: accuracy-cost frontiers, {args.platform_name} ({D.run} {D.checkpoint}, Karpathy {SPLIT})\n",
             "Dominance: a configuration dominates another if it is not worse on either axis and better on at least "
             "one. Accuracy differences count when the paired 95 % bootstrap interval of the difference excludes zero "
             "(1,000 resamples of the test images, shared by all configurations); cost differences when they exceed "
             "the pooled sd over the repetitions (energy, latency) or 1 % (size, memory). "
             + ("Energy above idle (methodology §8). " if args.energy_basis == "above" else
                "Energy including idle (the device's idle draw during each item counts; methodology §8 uses above "
                "idle, frontiers_<platform>.md). ")
             + "Accuracy in points (x 100).\n"]
    if args.platform != DEFAULT_PLATFORM:
        lines.append(f"Cost measured on {args.platform_name} ({args.energy}); accuracy is the Mac's full Karpathy-test "
                     f"evaluation at batch 64. batch_check_{args.platform}.md scores this platform's batch-1 outputs on "
                     "the 500-item subset against it.\n")
    if (BASELINE, "caption", "beam3") in costs:
        lines.append("Show and Tell (CNN-LSTM baseline, 13_baseline.py) was measured in another session next to the "
                     "FP32 reference; its energy and latency are its ratio to that FP32 (per repetition) times the "
                     "FP32 values here. Captions only.\n")
    for spec in specs:
        pts = frontier_points(spec, D, M, vqa, W, costs)
        if not pts:
            continue
        name, task, metric, higher, kind, xlabel, ylabel = spec
        results[name] = [{k: p[k] for k in ("variant", "value", "ci95", "cost", "cost_sd", "frontier",
                                            "dominated_by")} for p in pts]
        unit = {"energy": "J", "latency": "ms", "size": "MiB", "memory": "MB"}[kind]
        lines += [f"\n## {metric} vs {kind} ({'captions' if task == 'caption' else 'answers'})\n",
                  f"Frontier: {', '.join(display_name(p['variant']) for p in sorted(pts, key=lambda p: p['cost']) if p['frontier'])}\n",
                  f"| Configuration | {metric} [95 % CI] | {kind} ({unit}) | On frontier | Dominated by |",
                  "|---|---|---|---|---|"]
        for p in sorted(pts, key=lambda p: p["cost"]):
            sd = f" ± {p['cost_sd']:.3f}" if p["cost_sd"] is not None and kind == "energy" else \
                (f" ± {p['cost_sd']:.1f}" if p["cost_sd"] is not None else "")
            cost = f"{p['cost']:.3f}" if kind == "energy" else f"{p['cost']:.0f}"
            lines.append(f"| {display_name(p['variant'])} | {p['value']:.1f} [{p['ci95'][0]:.1f}, {p['ci95'][1]:.1f}] | "
                         f"{cost}{sd} | {'yes' if p['frontier'] else ''} | "
                         f"{', '.join(display_name(x) for x in p['dominated_by'])} |")
    # summary ratio (§9): accuracy per joule; FLOPs do not change with precision, so a per-GFLOP ratio cannot separate
    # the precision configurations and is not reported for them.
    lines += [f"\n## Accuracy per joule ({basis_label})\n", "| Configuration | CIDEr per J | VQA accuracy per J |",
              "|---|---|---|"]
    ce = {p["variant"]: p for p in results.get("captions_energy", [])}
    ae = {p["variant"]: p for p in results.get("answers_energy", [])}
    for v in D.variants + D.extra:
        c = ce.get(v)
        a = ae.get(v)
        lines.append(f"| {display_name(v)} | {c['value'] / c['cost']:.1f} |" if c else f"| {display_name(v)} | - |")
        lines[-1] += f" {a['value'] / a['cost']:.1f} |" if a else " - |"
    (out_dir / f"frontiers_{slug}.json").write_text(json.dumps(
        {"platform": args.platform_name, "energy_file": str(args.energy), "energy_basis": args.energy_basis,
         "frontiers": results,
         "time": datetime.now().isoformat(timespec="seconds")}, indent=1))
    (out_dir / f"frontiers_{slug}.md").write_text("\n".join(lines) + "\n")

    # primary figure
    by = {s[0]: s for s in specs}
    if "captions_energy" in results:
        pts = frontier_points(by["captions_energy"], D, M, vqa, W, costs)
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        plot_frontier(ax, pts, by["captions_energy"])
        ax.set_title(f"Captioning: accuracy vs energy per caption ({args.platform_name}, batch 1"
                     + ("" if args.energy_basis == "above" else ", incl. idle") + ")", loc="left")
        variant_legend(ax, [p["variant"] for p in pts], anchor=(0.5, -0.15))
        save(fig, out_dir, f"frontier_captions_energy_{slug}")
        plt.close(fig)
    # secondary figures
    panels = [s for s in ("captions_spice_energy", "captions_chair_energy", "captions_latency", "captions_size")
              if s in results]
    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
        for ax, s in zip(axes.flat, panels):
            plot_frontier(ax, frontier_points(by[s], D, M, vqa, W, costs), by[s])
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(f"Captioning: secondary frontiers ({args.platform_name})", x=0.06, ha="left", fontsize=11)
        figure_legend(fig, [p["variant"] for p in frontier_points(by[panels[0]], D, M, vqa, W, costs)], ncol=7)
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))
        save(fig, out_dir, f"frontier_captions_secondary_{slug}")
        plt.close(fig)
    panels = [s for s in ("answers_energy", "answers_size") if s in results]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), squeeze=False)
        for ax, s in zip(axes[0], panels):
            plot_frontier(ax, frontier_points(by[s], D, M, vqa, W, costs), by[s])
        fig.suptitle(f"VQA: accuracy vs cost ({args.platform_name}, batch 1)", x=0.06, ha="left", fontsize=11)
        figure_legend(fig, [p["variant"] for p in frontier_points(by[panels[0]], D, M, vqa, W, costs)], ncol=6)
        fig.tight_layout(rect=(0, 0.13, 1, 0.95))
        save(fig, out_dir, f"frontier_answers_{slug}")
        plt.close(fig)
    print("\n".join(lines[:2] + [ln for ln in lines if ln.startswith("Frontier:") or ln.startswith("\n## ")]))


# ---------------------------------------------------------------- grounding score (shared by RQ2 and RQ3)

class Decoders:
    """Loads one decoder precision at a time, with its mean visual prefix Ī (cached on disk)."""

    def __init__(self, D, dev, out_dir, n_prefix=PREFIX_SAMPLE):
        self.D, self.dev, self.cache = D, dev, out_dir / "cache"
        self.n_prefix = n_prefix
        self.precision = self.model = self.ibar = None
        self._train = None

    def train_sample(self) -> torch.Tensor:
        if self._train is None:
            store = FeatureStore(self.D.encoder, "coco", ["coco_train"])
            idx = sorted(random.Random(0).sample(range(len(store.feats)), min(self.n_prefix, len(store.feats))))
            self._train = store.feats[idx]
        return self._train

    def get(self, precision: str):
        if precision != self.precision:
            self.model = None
            empty_cache(self.dev)
            self.model, _, _, _ = load_decoder(self.D.run, self.D.checkpoint, precision, self.dev)
            f = self.cache / f"mean_prefix_{precision}_n{self.n_prefix}.pt"
            if f.exists():
                self.ibar = torch.load(f)
            else:
                self.ibar = mean_prefix(self.model, self.train_sample()).cpu()
                torch.save(self.ibar, f)
            self.precision = precision
        return self.model, self.ibar.to(self.dev)


def test_features(D, encoder_precision: str) -> FeatureStore:
    return FeatureStore(feature_name(D.encoder, encoder_precision), "coco", [f"coco_{SPLIT}"])


def caption_grounding(D, dec: Decoders, variant: str, tok) -> dict:
    """Per object mention in the variant's captions: image index, hallucinated?, grounding score."""
    c, _ = D.config_of(variant)
    enc_p, dec_p = CONFIGS[c]
    model, ibar = dec.get(dec_p)
    model.set_task("caption")
    store = test_features(D, enc_p)
    caps = [D.captions[variant][i] for i in D.image_ids]
    enc = [caption_tokens(tok, cap) for cap in caps]
    seqs = [e[0] for e in enc]
    feats = torch.stack([store.get(i) for i in D.image_ids])
    lp_i = token_logprobs(model, seqs, [0] * len(seqs), tok.eos_token_id, feats=feats)
    lp_b = token_logprobs(model, seqs, [0] * len(seqs), tok.eos_token_id, prefix=ibar)
    img, hall, g, all_tokens = [], [], [], []
    for k, (i, cap, (ids, offsets)) in enumerate(zip(D.image_ids, caps, enc)):
        gt = (lp_i[k] - lp_b[k])[: len(offsets)]  # caption tokens, without the end-of-text token
        all_tokens.append(float(gt.mean()) if len(gt) else np.nan)
        for word, node, s, e in object_mentions(D.chair, cap):
            toks = [t for t, (a, b) in enumerate(offsets) if a < e and b > s]
            if toks:
                img.append(k)
                hall.append(node not in D.gt[i])
                g.append(float(gt[toks].sum()))
    return {"img": np.array(img, dtype=np.int64), "hall": np.array(hall, dtype=bool),
            "g": np.array(g, dtype=np.float64), "token_mean": np.array(all_tokens)}


def group_means(rec: dict, mask: np.ndarray, n_images: int, W: np.ndarray):
    """(mean, bootstrap replicates) of rec['g'] over the mentions selected by mask, images resampled."""
    s = np.bincount(rec["img"][mask], weights=rec["g"][mask], minlength=n_images)
    c = np.bincount(rec["img"][mask], minlength=n_images).astype(np.float64)
    return s.sum() / max(c.sum(), 1), boot_ratio(s, c, W)


# ---------------------------------------------------------------- RQ2

def part_rq2(D, args, out_dir, M, W, dev):
    plt = setup_matplotlib()
    n = len(D.image_ids)
    lines = [f"# RQ2: hallucination ({D.run} {D.checkpoint}, Karpathy {SPLIT}, {n:,} images)\n",
             "## CHAIR by caption length\n",
             "Captions binned by length so that shorter, more generic captions cannot pass for less hallucination "
             "(methodology §9). n = captions in the bin.\n",
             "| Configuration | Words (mean) | " + " | ".join(f"{b[0]}: n / CHAIR_s / CHAIR_i" for b in LENGTH_BINS) + " |",
             "|---|---|" + "---|" * len(LENGTH_BINS)]
    chair_bins = {}
    for v in D.variants:
        m = M[v]
        cells, chair_bins[v] = [], {}
        for label, lo, hi in LENGTH_BINS:
            sel = (m["words"] >= lo) & (m["words"] <= hi)
            k = int(sel.sum())
            cs = float((m["hall"][sel] > 0).mean()) if k else float("nan")
            ci = float(m["hall"][sel].sum() / max(m["mentions"][sel].sum(), 1)) if k else float("nan")
            chair_bins[v][label] = {"captions": k, "CHAIR_s": cs, "CHAIR_i": ci}
            cells.append(f"{k} / {100 * cs:.1f} / {100 * ci:.1f}")
        lines.append(f"| {display_name(v)} | {m['words'].mean():.2f} | " + " | ".join(cells) + " |")

    tok = load_tokenizer(D.decoder)
    dec = Decoders(D, dev, out_dir)
    ground = {}
    order = sorted(D.variants, key=lambda v: CONFIGS[D.config_of(v)[0]][1])  # group by decoder precision
    for v in order:
        t0 = time.perf_counter()
        rec = caption_grounding(D, dec, v, tok)
        hall = rec["hall"]
        gc, bc = group_means(rec, ~hall, n, W)
        gh, bh = group_means(rec, hall, n, W)
        ground[v] = {"rec": rec, "correct": (gc, bc), "hallucinated": (gh, bh)}
        print(f"  grounding {display_name(v)}: {len(hall):,} object mentions ({hall.sum():,} hallucinated), "
              f"mean g correct {gc:.2f}, hallucinated {gh:.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    dec.model = None
    empty_cache(dev)

    ref = ground.get(REFERENCE)
    lines += ["\n## Visual-grounding score of object words\n",
              "g = log p(word | caption so far, image) - log p(word | caption so far, mean image); natural log, summed "
              "over the word's tokens. Each configuration scores its own captions at its own precision. Intervals: "
              "95 % bootstrap over images. A hallucinated-word score closer to zero than the correct-word score means "
              "hallucinated objects lean on the language prior rather than the image (H2).\n",
              "| Configuration | Mentions (hallucinated) | g correct [CI] | g hallucinated [CI] | Difference [CI] | "
              "Share g <= 0: correct / hallucinated | g hallucinated vs FP32 [CI] |", "|---|---|---|---|---|---|---|"]
    out = {}
    for v in D.variants:
        G = ground[v]
        rec, (gc, bc), (gh, bh) = G["rec"], G["correct"], G["hallucinated"]
        diff = bh - bc
        le0_c = float((rec["g"][~rec["hall"]] <= 0).mean()) if (~rec["hall"]).any() else float("nan")
        le0_h = float((rec["g"][rec["hall"]] <= 0).mean()) if rec["hall"].any() else float("nan")
        vs = (gh - ref["hallucinated"][0], ci95(bh - ref["hallucinated"][1])) if ref and v != REFERENCE else None
        out[v] = {"mentions": int(len(rec["g"])), "hallucinated": int(rec["hall"].sum()),
                  "g_correct": gc, "g_correct_ci95": ci95(bc), "g_hallucinated": gh, "g_hallucinated_ci95": ci95(bh),
                  "difference": gh - gc, "difference_ci95": ci95(diff), "share_le0_correct": le0_c,
                  "share_le0_hallucinated": le0_h, "g_token_mean": float(np.nanmean(rec["token_mean"])),
                  "g_hallucinated_vs_fp32": vs}
        lines.append(f"| {display_name(v)} | {len(rec['g']):,} ({int(rec['hall'].sum()):,}) | "
                     f"{gc:.2f} [{ci95(bc)[0]:.2f}, {ci95(bc)[1]:.2f}] | {gh:.2f} [{ci95(bh)[0]:.2f}, {ci95(bh)[1]:.2f}] | "
                     f"{gh - gc:.2f} [{ci95(diff)[0]:.2f}, {ci95(diff)[1]:.2f}] | {100 * le0_c:.0f} % / {100 * le0_h:.0f} % | "
                     + (f"{vs[0]:+.2f} [{vs[1][0]:+.2f}, {vs[1][1]:+.2f}] |" if vs else "- |"))
    (out_dir / "rq2.json").write_text(json.dumps({"chair_by_length": chair_bins, "grounding": out,
                                                  "prefix_sample": PREFIX_SAMPLE,
                                                  "time": datetime.now().isoformat(timespec="seconds")}, indent=1))
    (out_dir / "rq2.md").write_text("\n".join(lines) + "\n")

    # figure: dot plot, one row per configuration, correct vs hallucinated with intervals
    vs_ = list(reversed(D.variants))
    fig, ax = plt.subplots(figsize=(6.6, 0.38 * len(vs_) + 1.4))
    for k, v in enumerate(vs_):
        for j, (key, color, off) in enumerate((("correct", SERIES[0], 0.12), ("hallucinated", SERIES[1], -0.12))):
            m, b = ground[v][key]
            lo, hi = ci95(b)
            ax.plot([lo, hi], [k + off] * 2, color=MUTED, lw=0.9, zorder=2)
            ax.plot(m, k + off, ls="none", marker="o", ms=7, mfc=color, mec=SURFACE, mew=1.5, zorder=3,
                    label=("correctly mentioned objects", "hallucinated objects")[j] if k == 0 else None)
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    ax.set_yticks(range(len(vs_)))
    ax.set_yticklabels([display_name(v) for v in vs_])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Grounding score g (nats): log p(word | image) - log p(word | mean image)")
    ax.set_title("Object words: how much the image raised their probability", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1 - 1.2 / (0.38 * len(vs_) + 1.4) * 0.3), ncol=2)
    save(fig, out_dir, "rq2_grounding")
    plt.close(fig)
    print("\n".join(lines[-len(D.variants) - 2:]))


# ---------------------------------------------------------------- RQ3

def answer_grounding(D, dec: Decoders, config: str, tok) -> dict:
    """Per question with a non-empty answer: image index, accuracy, answer type, grounding score of the answer."""
    enc_p, dec_p = CONFIGS[config]
    model, ibar = dec.get(dec_p)
    model.set_task("vqa")
    store = test_features(D, enc_p)
    ans = D.answers[config]
    qs = [q for q in D.questions if ans[q["question_id"]][0].strip()]
    seqs, starts = [], []
    for q in qs:
        prompt = question_prompt(tok, q["question"])
        a = tok(f" {ans[q['question_id']][0]}", add_special_tokens=False)["input_ids"][:MAX_ANSWER_TOKENS]
        seqs.append(prompt + a)
        starts.append(len(prompt))
    feats = torch.stack([store.get(q["image_id"]) for q in qs])
    lp_i = token_logprobs(model, seqs, starts, tok.eos_token_id, feats=feats, batch_size=128)
    lp_b = token_logprobs(model, seqs, starts, tok.eos_token_id, prefix=ibar, batch_size=128)
    return {"img": np.array([D.index[q["image_id"]] for q in qs], dtype=np.int64),
            "acc": np.array([ans[q["question_id"]][1] for q in qs]),
            "type": np.array([D.anns[q["question_id"]]["answer_type"] for q in qs]),
            "g": np.array([float((a - b).sum()) for a, b in zip(lp_i, lp_b)]),
            "empty": len(D.questions) - len(qs)}


def part_rq3(D, args, out_dir, M, vqa, W, dev):
    from scipy.stats import wilcoxon

    plt = setup_matplotlib()
    n = len(D.image_ids)
    configs = [c for c in CONFIGS if c != REFERENCE and c in vqa and c in M]
    cap = {c: (M[c]["CIDEr"].mean(), boot_mean(M[c]["CIDEr"], W)) for c in [REFERENCE, *configs]}
    ans = {c: (vqa[c][0].sum() / vqa[c][1].sum(), boot_ratio(*vqa[c], W)) for c in [REFERENCE, *configs]}
    rows, d_points, d_boot = [], [], []
    for c in configs:
        r_cap, b_cap = cap[c][0] / cap[REFERENCE][0], cap[c][1] / cap[REFERENCE][1]
        r_vqa, b_vqa = ans[c][0] / ans[REFERENCE][0], ans[c][1] / ans[REFERENCE][1]
        rows.append({"config": c, "R_cap": r_cap, "R_cap_ci95": ci95(b_cap), "R_vqa": r_vqa,
                     "R_vqa_ci95": ci95(b_vqa), "delta_R": r_vqa - r_cap, "delta_R_ci95": ci95(b_vqa - b_cap)})
        d_points.append(r_vqa - r_cap)
        d_boot.append(b_vqa - b_cap)
    d_points = np.array(d_points)
    mean_boot = np.mean(d_boot, axis=0)
    test = wilcoxon(d_points, alternative="two-sided") if len(d_points) >= 2 else None
    lines = [f"# RQ3: retention of VQA vs captioning ({D.run} {D.checkpoint}, Karpathy {SPLIT})\n",
             "R_cap = CIDEr / CIDEr(FP32), R_vqa = VQA accuracy / VQA accuracy(FP32), ΔR = R_vqa - R_cap. Only the "
             "precision configurations change both heads (methodology §9). Intervals: 95 % bootstrap over test images, "
             "captions and questions of an image resampled together.\n",
             "| Configuration | R_cap [CI] | R_vqa [CI] | ΔR [CI] |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {display_name(r['config'])} | {r['R_cap']:.4f} [{r['R_cap_ci95'][0]:.4f}, {r['R_cap_ci95'][1]:.4f}] | "
                     f"{r['R_vqa']:.4f} [{r['R_vqa_ci95'][0]:.4f}, {r['R_vqa_ci95'][1]:.4f}] | "
                     f"{r['delta_R']:+.4f} [{r['delta_R_ci95'][0]:+.4f}, {r['delta_R_ci95'][1]:+.4f}] |")
    lines.append(f"\nMean ΔR over the {len(rows)} configurations: {d_points.mean():+.4f} "
                 f"[{ci95(mean_boot)[0]:+.4f}, {ci95(mean_boot)[1]:+.4f}]. "
                 + (f"Wilcoxon signed-rank test (two-sided, n = {len(d_points)}): W = {test.statistic:.1f}, "
                    f"p = {test.pvalue:.3f}." if test else ""))

    tok = load_tokenizer(D.decoder)
    dec = Decoders(D, dev, out_dir)
    grounding = {}
    for c in sorted([REFERENCE, *configs], key=lambda c: CONFIGS[c][1]):
        t0 = time.perf_counter()
        rec = answer_grounding(D, dec, c, tok)
        s = np.bincount(rec["img"], weights=rec["g"], minlength=n)
        k = np.bincount(rec["img"], minlength=n).astype(np.float64)
        entry = {"rec": rec, "all": (s.sum() / k.sum(), boot_ratio(s, k, W))}
        for name, mask in (("correct", rec["acc"] >= CORRECT_ANSWER), ("wrong", rec["acc"] == 0),
                           *[(t, rec["type"] == t) for t in ("yes/no", "number", "other")]):
            s2 = np.bincount(rec["img"][mask], weights=rec["g"][mask], minlength=n)
            k2 = np.bincount(rec["img"][mask], minlength=n).astype(np.float64)
            entry[name] = (s2.sum() / max(k2.sum(), 1), boot_ratio(s2, k2, W), int(mask.sum()))
        grounding[c] = entry
        print(f"  answer grounding {display_name(c)}: {len(rec['g']):,} answers, mean g {entry['all'][0]:.2f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    dec.model = None
    empty_cache(dev)
    ref = grounding[REFERENCE]
    lines += ["\n## Visual-grounding score of answers\n",
              "g(a) = log p(answer | question, image) - log p(answer | question, mean image), summed over the answer's "
              f"tokens. Correct: VQA accuracy >= {CORRECT_ANSWER}; wrong: 0. A head that loses visual evidence under "
              "compression shows a lower g than FP32.\n",
              "| Configuration | g all [CI] | vs FP32 [CI] | g correct | g wrong | yes/no | number | other |",
              "|---|---|---|---|---|---|---|---|"]
    out = {}
    for c in [REFERENCE, *configs]:
        e = grounding[c]
        vs = None if c == REFERENCE else (e["all"][0] - ref["all"][0], ci95(e["all"][1] - ref["all"][1]))
        out[c] = {"g_all": e["all"][0], "g_all_ci95": ci95(e["all"][1]), "g_vs_fp32": vs,
                  **{f"g_{k}": e[k][0] for k in ("correct", "wrong", "yes/no", "number", "other")},
                  **{f"n_{k}": e[k][2] for k in ("correct", "wrong", "yes/no", "number", "other")},
                  "empty_answers": e["rec"]["empty"]}
        lines.append(f"| {display_name(c)} | {e['all'][0]:.2f} [{ci95(e['all'][1])[0]:.2f}, {ci95(e['all'][1])[1]:.2f}] | "
                     + (f"{vs[0]:+.3f} [{vs[1][0]:+.3f}, {vs[1][1]:+.3f}] | " if vs else "- | ")
                     + " | ".join(f"{e[k][0]:.2f}" for k in ("correct", "wrong", "yes/no", "number", "other")) + " |")
    (out_dir / "rq3.json").write_text(json.dumps(
        {"retention": rows, "mean_delta_R": float(d_points.mean()), "mean_delta_R_ci95": ci95(mean_boot),
         "wilcoxon": {"statistic": float(test.statistic), "p": float(test.pvalue), "n": len(d_points)} if test else None,
         "answer_grounding": out, "time": datetime.now().isoformat(timespec="seconds")}, indent=1))
    (out_dir / "rq3.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6.0, 0.4 * len(rows) + 1.3))
    for k, r in enumerate(reversed(rows)):
        color, marker = variant_style(r["config"])
        ax.plot(r["delta_R_ci95"], [k, k], color=MUTED, lw=0.9, zorder=2)
        ax.plot(r["delta_R"], k, ls="none", marker=marker, ms=8, mfc=color, mec=SURFACE, mew=1.5, zorder=3)
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([display_name(r["config"]) for r in reversed(rows)])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ΔR = R_vqa - R_cap (relative to FP32; > 0: VQA keeps more than captioning)")
    ax.set_title("Retention under reduced precision, VQA vs captioning", loc="left")
    save(fig, out_dir, "rq3_retention")
    plt.close(fig)
    print("\n".join(lines))


# ---------------------------------------------------------------- Grad-CAM

def gradcam_map(pipe, image_path: Path, task: str, seq: list[int], targets: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(224 x 224 RGB image as the encoder sees it, 16 x 16 Grad-CAM map in [0, 1]) for the summed log-probability of
    seq[j], j in targets, under teacher forcing. Target layer: the input of ViT-L/14's last block (its first layer
    norm); the output of the last block reaches the embedding only through the class token, so its patch tokens
    carry no gradient."""
    from PIL import Image

    dev, model, enc = pipe.device, pipe.model, pipe.encoder
    with Image.open(image_path) as im:
        pixels = pipe.processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"].to(dev)
    pixels.requires_grad_(True)
    store = {}

    def hook(_m, _i, o):
        o.retain_grad()
        store["a"] = o

    handle = enc.vision_model.encoder.layers[-1].layer_norm1.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            model.set_task(task)
            emb = enc(pixel_values=pixels).image_embeds
            base = model.base_lm()
            k = model.prefix_len
            ids = torch.tensor([seq], device=dev)
            hidden = base.transformer(inputs_embeds=torch.cat([model.prefix(emb), base.transformer.wte(ids)], 1)
                                      ).last_hidden_state[0]
            lp = torch.log_softmax(base.lm_head(hidden[[k + j - 1 for j in targets]]).float(), -1)
            lp.gather(1, torch.tensor([seq[j] for j in targets], device=dev)[:, None]).sum().backward()
    finally:
        handle.remove()
    a, g = store["a"][0, 1:].detach().float(), store["a"].grad[0, 1:].float()
    cam = torch.relu(a @ g.mean(0))
    side = int(round(cam.numel() ** 0.5))
    cam = cam.reshape(side, side)
    cam = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()
    mean = torch.tensor(pipe.processor.image_mean).view(3, 1, 1)
    std = torch.tensor(pipe.processor.image_std).view(3, 1, 1)
    img = (pixels.detach()[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return img, cam


OCCLUSION_WINDOW, OCCLUSION_STRIDE = 28, 14  # pixels: 2 x 2 patches, moved by one patch


@torch.no_grad()
def occlusion_drops(pipe, pixels: torch.Tensor, jobs: list[tuple[str, list[int], list[list[int]]]]):
    """For every window position, how much greying out that window lowers the log-probability of each target:
    jobs = [(task, token sequence, [target token indices, ...])]. Returns ({(job, target): 224 x 224 map of the mean
    drop over the windows covering each pixel}, {(job, target): log-probability with the full image})."""
    from scipy.ndimage import gaussian_filter

    enc, model, tok = pipe.encoder, pipe.model, pipe.tokenizer
    side = pixels.shape[-1]
    pos = list(range(0, side - OCCLUSION_WINDOW + 1, OCCLUSION_STRIDE))
    batch = [pixels]
    for y in pos:
        for x in pos:
            m = pixels.clone()
            m[:, :, y: y + OCCLUSION_WINDOW, x: x + OCCLUSION_WINDOW] = 0  # the processor's mean colour
            batch.append(m)
    allpx = torch.cat(batch)
    embs = torch.cat([enc(pixel_values=allpx[i: i + 32]).image_embeds for i in range(0, len(allpx), 32)]).float()
    maps, full = {}, {}
    for j, (task, seq, targets) in enumerate(jobs):
        model.set_task(task)
        lps = token_logprobs(model, [seq] * len(embs), [0] * len(embs), tok.eos_token_id, feats=embs)
        for t, tgt in enumerate(targets):
            lp = np.array([x[tgt].sum() for x in lps])
            drop = lp[0] - lp[1:]
            heat, cnt, k = np.zeros((side, side)), np.zeros((side, side)), 0
            for y in pos:
                for x in pos:
                    heat[y: y + OCCLUSION_WINDOW, x: x + OCCLUSION_WINDOW] += drop[k]
                    cnt[y: y + OCCLUSION_WINDOW, x: x + OCCLUSION_WINDOW] += 1
                    k += 1
            maps[(j, t)] = gaussian_filter(np.clip(heat / np.maximum(cnt, 1), 0, None), sigma=OCCLUSION_STRIDE / 2)
            full[(j, t)] = float(lp[0])
    return maps, full


def artifact_tokens(pipe, pixels: torch.Tensor, k: int = 3) -> list[tuple[int, int, float]]:
    """The k patch tokens with the largest hidden-state norm at the input of ViT-L/14's last block: (row, col, norm
    / median norm). CLIP ViT-L/14 stores global information in a few such high-norm tokens on low-detail background
    (Darcet et al., 2024); gradient-based maps concentrate on them."""
    with torch.no_grad():
        hs = pipe.encoder.vision_model(pixel_values=pixels, output_hidden_states=True).hidden_states[-2][0, 1:]
    norms = hs.norm(dim=-1).float().cpu().numpy()
    side = int(round(len(norms) ** 0.5))
    return [(int(t) // side, int(t) % side, float(norms[t] / np.median(norms))) for t in np.argsort(-norms)[:k]]


def overlay(ax, img: np.ndarray, heat: np.ndarray):
    """Greyscale image (so the heat colour is never confused with the image's own colours) and a one-hue overlay
    whose opacity carries the value."""
    from matplotlib.colors import to_rgb

    grey = np.repeat(img.mean(-1, keepdims=True), 3, -1) * 0.75 + 0.25
    h = heat / heat.max() if heat.max() > 0 else heat
    ax.imshow(grey)
    ax.imshow(np.concatenate([np.broadcast_to(np.array(to_rgb(HEAT)), h.shape + (3,)), (0.85 * h)[..., None]], -1))


def part_attribution(D, args, out_dir, dev):
    from matplotlib.patches import Rectangle
    from PIL import Image

    from greenvl.inference import load_pipeline

    plt = setup_matplotlib()
    pipe = load_pipeline(D.run, D.checkpoint, dev, REFERENCE)
    tok = pipe.tokenizer
    files = {r["image_id"]: r["file"] for r in json.loads((paths.PROCESSED / "coco_karpathy.json").read_text())[SPLIT]}
    first_q = {}
    for q in D.questions:
        first_q.setdefault(q["image_id"], q)
    ids = list(D.image_ids)
    random.Random(0).shuffle(ids)
    n_hall = max(1, args.attribution_images // 4)
    clean, halluc = [], []
    for i in ids:
        ms = object_mentions(D.chair, D.captions[REFERENCE][i])
        if i not in first_q or len({m[1] for m in ms}) < 2:
            continue
        if any(m[1] not in D.gt[i] for m in ms):
            if len(halluc) < n_hall:
                halluc.append(i)
        elif len(clean) < args.attribution_images - n_hall:
            clean.append(i)
        if len(clean) + len(halluc) >= args.attribution_images:
            break
    rows = []
    mean = torch.tensor(pipe.processor.image_mean).view(3, 1, 1)
    std = torch.tensor(pipe.processor.image_std).view(3, 1, 1)
    for i in clean + halluc:
        cap = D.captions[REFERENCE][i]
        seq, offsets = caption_tokens(tok, cap)
        words, seen = [], set()
        for m in sorted(object_mentions(D.chair, cap), key=lambda m: m[1] in D.gt[i]):  # hallucinated first
            if m[1] not in seen:
                seen.add(m[1])
                words.append(m)
        words = words[:2]
        word_targets = [[t for t, (a, b) in enumerate(offsets) if a < e and b > s] for _, _, s, e in words]
        q = first_q[i]
        a_text = D.answers[REFERENCE][q["question_id"]][0]
        prompt = question_prompt(tok, q["question"])
        a_ids = tok(f" {a_text}", add_special_tokens=False)["input_ids"][:MAX_ANSWER_TOKENS]
        with Image.open(paths.COCO_IMAGES / files[i]) as im:
            pixels = pipe.processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"].to(dev)
        t0 = time.perf_counter()
        maps, full = occlusion_drops(pipe, pixels, [("caption", seq, word_targets),
                                                    ("vqa", prompt + a_ids, [list(range(len(prompt), len(prompt) + len(a_ids)))])])
        img = (pixels[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        cells = [{"kind": "word", "word": cap[s:e], "hallucinated": node not in D.gt[i], "map": maps[(0, t)],
                  "logprob": full[(0, t)], "max_drop": float(maps[(0, t)].max())}
                 for t, (_, node, s, e) in enumerate(words)]
        cells.append({"kind": "answer", "question": q["question"], "answer": a_text, "map": maps[(1, 0)],
                      "logprob": full[(1, 0)], "max_drop": float(maps[(1, 0)].max())})
        rows.append({"image_id": i, "caption": cap, "img": img, "pixels": pixels, "cells": cells,
                     "word_targets": word_targets, "seq": seq, "artifacts": artifact_tokens(pipe, pixels)})
        print(f"  occlusion maps, image {i} ({time.perf_counter() - t0:.0f} s): {cap!r}", flush=True)

    per_fig = 4
    for f in range(0, len(rows), per_fig):
        part = rows[f: f + per_fig]
        fig, axes = plt.subplots(len(part), 4, figsize=(9.2, 2.85 * len(part)), squeeze=False)
        for r, row in enumerate(part):
            axes[r][0].imshow(row["img"])
            axes[r][0].set_xlabel("\n".join(textwrap.wrap(row["caption"], 34)), fontsize=8, color=INK)
            word_cells = [x for x in row["cells"] if x["kind"] == "word"]
            for c in range(1, 4):
                ax = axes[r][c]
                cell = row["cells"][-1] if c == 3 else (word_cells[c - 1] if c - 1 < len(word_cells) else None)
                if cell is None:
                    ax.set_visible(False)
                    continue
                overlay(ax, row["img"], cell["map"])
                scale = f"max drop {cell['max_drop']:.2f} nats"
                if cell["kind"] == "word":
                    label = f'"{cell["word"]}"' + (" (hallucinated)" if cell["hallucinated"] else "") + f"\n{scale}"
                else:
                    label = "\n".join(textwrap.wrap(f"Q: {cell['question']}", 36)) + f"\nA: {cell['answer']} ({scale})"
                ax.set_xlabel(label, fontsize=8, color=INK)
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
        for c, title in enumerate(("Image (as encoded)", "Object word 1", "Object word 2", "Answer")):
            axes[0][c].set_title(title, fontsize=9, color=INK2)
        fig.suptitle("Occlusion: where hiding a 28 x 28 px region lowers the word's probability (FP32)", x=0.02,
                     ha="left", fontsize=10)
        fig.tight_layout()
        save(fig, out_dir, f"attribution_{f // per_fig + 1}")
        plt.close(fig)

    # why not gradients: Grad-CAM vs occlusion for the first two images, high-norm tokens marked
    comp = rows[:2]
    if comp:
        fig, axes = plt.subplots(len(comp), 3, figsize=(7.0, 3.0 * len(comp)), squeeze=False)
        for r, row in enumerate(comp):
            cell = next(x for x in row["cells"] if x["kind"] == "word")
            img, cam = gradcam_map(pipe, paths.COCO_IMAGES / files[row["image_id"]], "caption", row["seq"],
                                   row["word_targets"][0])
            cam_up = torch.nn.functional.interpolate(torch.tensor(cam)[None, None], size=img.shape[:2],
                                                     mode="bilinear", align_corners=False)[0, 0].numpy()
            axes[r][0].imshow(row["img"])
            for rr, cc, _ in row["artifacts"]:
                axes[r][0].add_patch(Rectangle((cc * 14, rr * 14), 14, 14, fill=False, ec=SERIES[0], lw=1.4))
            overlay(axes[r][1], row["img"], cam_up)
            overlay(axes[r][2], row["img"], cell["map"])
            axes[r][0].set_xlabel(f'"{cell["word"]}"; boxes: 3 highest-norm tokens\n'
                                  f"({', '.join(f'{x[2]:.0f}x' for x in row['artifacts'])} the median norm)", fontsize=8)
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
        for c, title in enumerate(("Image", "Grad-CAM (last block)", "Occlusion")):
            axes[0][c].set_title(title, fontsize=9, color=INK2)
        fig.tight_layout(h_pad=2.0)
        save(fig, out_dir, "attribution_methods")
        plt.close(fig)

    meta = [{"image_id": row["image_id"], "caption": row["caption"], "artifact_tokens": row["artifacts"],
             "cells": [{k: v for k, v in c.items() if k != "map"} for c in row["cells"]]} for row in rows]
    (out_dir / "attribution.json").write_text(json.dumps(
        {"method": f"occlusion, {OCCLUSION_WINDOW} px window, stride {OCCLUSION_STRIDE} px, mean-colour fill; map = "
                   "mean drop in log-probability over the windows covering each pixel, Gaussian-smoothed",
         "images": meta, "seed": 0}, indent=1))
    del pipe
    empty_cache(dev)


# ---------------------------------------------------------------- batch 1 vs batch 64

def part_batch(D, args, out_dir):
    from greenvl.vqa_metrics import question_accuracy

    path = args.energy.with_name("energy.jsonl")
    sub = json.loads((paths.RESULTS / "inference" / f"subset_{SPLIT}_500.json").read_text())
    recs = [json.loads(ln) for ln in path.read_text().splitlines()]
    windows = {r["key"]: r for r in recs if r["type"] == "window" and r["kind"] == "decode" and r.get("outputs")
               and r["repeat"] == 0}
    cap_ids = [it["image_id"] for it in sub["caption"]]
    entries = []
    for key, r in sorted(windows.items()):
        if r["task"] != "caption":
            continue
        v = variant_name(r["config"], r["decoding"])
        if v in D.captions:
            entries.append((v, r["outputs"]))
    print(f"PTB tokenisation for the batch-1 check ({len(entries)} configurations)", flush=True)
    tok = ptb_tokenize({**{("g", i): D.refs[i] for i in cap_ids},
                        **{(v, "b1", i): [o] for v, outs in entries for i, o in zip(cap_ids, outs)},
                        **{(v, "b64", i): [D.captions[v][i]] for v, _ in entries for i in cap_ids}})
    gts = {i: tok[("g", i)] for i in cap_ids}
    rows = {}
    for v, outs in entries:
        r = {"identical": float(np.mean([o == D.captions[v][i] for i, o in zip(cap_ids, outs)]))}
        for b, caps in (("b1", dict(zip(cap_ids, outs))), ("b64", {i: D.captions[v][i] for i in cap_ids})):
            r[f"CIDEr_{b}"] = 100 * cider(gts, {i: tok[(v, b, i)] for i in cap_ids})[0]
            ms = [object_mentions(D.chair, caps[i]) for i in cap_ids]
            hall = sum(m[1] not in D.gt[i] for i, mm in zip(cap_ids, ms) for m in mm)
            r[f"CHAIR_i_{b}"] = 100 * hall / max(sum(len(mm) for mm in ms), 1)
        rows[v] = r
    for key, w in sorted(windows.items()):
        if w["task"] != "vqa" or w["config"] not in D.answers:
            continue
        r = rows.setdefault(w["config"], {})
        b64 = D.answers[w["config"]]
        qids = [it["question_id"] for it in sub["vqa"]]
        r["answers_identical"] = float(np.mean([o == b64[q][0] for q, o in zip(qids, w["outputs"])]))
        r["VQA_b1"] = 100 * float(np.mean([question_accuracy(o, D.anns[q]) for q, o in zip(qids, w["outputs"])]))
        r["VQA_b64"] = 100 * float(np.mean([b64[q][1] for q in qids]))
    other = args.platform != DEFAULT_PLATFORM
    lines = [f"# Batch 1 (energy runs{', ' + args.platform_name if other else ''}) vs batch 64 (evaluation, "
             f"{'MacBook Pro M4 Pro' if other else 'same device'}), same 500 test images and 500 questions\n",
             "Accuracy of the first-repetition outputs of 10_measure_variants.py against the evaluation's outputs for "
             "the same items. CIDEr's document frequencies come from the 500 images' references, so these values are "
             "not comparable with the full-test CIDEr; only the batch-1 vs batch-64 difference is.\n",
             "| Configuration | Captions identical | CIDEr b64 | CIDEr b1 | Δ | CHAIR_i b64 | CHAIR_i b1 | "
             "Answers identical | VQA b64 | VQA b1 | Δ |", "|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, d=1, pct=False):
        return "-" if x is None else (f"{100 * x:.1f} %" if pct else f"{x:.{d}f}")
    for v in D.variants:
        r = rows.get(v)
        if not r:
            continue
        dc = r["CIDEr_b1"] - r["CIDEr_b64"] if "CIDEr_b1" in r else None
        dv = r["VQA_b1"] - r["VQA_b64"] if "VQA_b1" in r else None
        lines.append(f"| {display_name(v)} | {f(r.get('identical'), pct=True)} | {f(r.get('CIDEr_b64'))} | "
                     f"{f(r.get('CIDEr_b1'))} | {f(dc, 2)} | {f(r.get('CHAIR_i_b64'))} | {f(r.get('CHAIR_i_b1'))} | "
                     f"{f(r.get('answers_identical'), pct=True)} | {f(r.get('VQA_b64'))} | {f(r.get('VQA_b1'))} | "
                     f"{f(dv, 2)} |")
    suffix = f"_{args.platform}" if other else ""
    (out_dir / f"batch_check{suffix}.json").write_text(json.dumps(rows, indent=1))
    (out_dir / f"batch_check{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- main

PARTS = ("frontier", "rq2", "rq3", "attribution", "batch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    ap.add_argument("--platform", default=DEFAULT_PLATFORM, help="short name for the cost measurements (file names)")
    ap.add_argument("--platform-name", default="MacBook Pro M4 Pro", help="name in titles and reports")
    ap.add_argument("--energy", type=Path, default=None,
                    help="energy.json of 10_measure_variants.py (default: results/variants/<run>/<checkpoint>/)")
    ap.add_argument("--limit", type=int, default=None, help="first N test images (code check)")
    ap.add_argument("--label", default=None, help="results subfolder (default: the checkpoint name)")
    ap.add_argument("--no-spice", action="store_true")
    ap.add_argument("--attribution-images", type=int, default=8)
    ap.add_argument("--no-baseline", action="store_true", help="leave the CNN-LSTM baseline off the frontiers")
    ap.add_argument("--energy-basis", choices=list(ENERGY_BASES), default="above",
                    help="frontier energy above idle (default, methodology §8) or including idle (files *_inclidle)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    stem = Path(args.checkpoint).stem
    args.energy = args.energy or paths.RESULTS / "variants" / args.run / stem / "energy.json"
    out_dir = paths.RESULTS / "analysis" / args.run / (args.label or stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = get_device(args.device)
    t0 = time.perf_counter()
    D = Data(args.run, args.checkpoint, args.limit, baseline=not args.no_baseline)
    print(f"{len(D.variants)} configurations{' + the CNN-LSTM baseline' if D.extra else ''}, {len(D.image_ids):,} test images, {len(D.questions):,} questions "
          f"-> {out_dir}", flush=True)
    W = bootstrap_weights(len(D.image_ids))
    need_metrics = {"frontier", "rq2", "rq3"} & set(args.parts)
    M, vqa = per_image_metrics(D, out_dir, spice_on=not args.no_spice and "frontier" in args.parts) \
        if need_metrics else (None, None)
    for part in PARTS:
        if part not in args.parts:
            continue
        print(f"\n>>> {part} ({datetime.now():%H:%M})", flush=True)
        if part == "frontier":
            part_frontier(D, args, out_dir, M, vqa, W)
        elif part == "rq2":
            part_rq2(D, args, out_dir, M, W, dev)
        elif part == "rq3":
            part_rq3(D, args, out_dir, M, vqa, W, dev)
        elif part == "attribution":
            part_attribution(D, args, out_dir, dev)
        elif part == "batch":
            part_batch(D, args, out_dir)
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
