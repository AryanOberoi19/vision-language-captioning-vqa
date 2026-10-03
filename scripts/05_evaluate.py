#!/usr/bin/env python3
"""Step 5: decode and score a trained run (methodology §7).

    python scripts/05_evaluate.py --run coco_ViT-B-32_lora8_lr1e-3_s0 --select        # validation CIDEr per epoch
    python scripts/05_evaluate.py --run coco_ViT-B-32_lora8_lr1e-3_s0 --split test    # the selected checkpoint, all metrics
    python scripts/05_evaluate.py --run NAME --checkpoint step_004428.pt --split val --limit 500 --no-spice   # quick look

Captioning on the run's Karpathy split (COCO or Flickr8k): BLEU-4, CIDEr (primary), SPICE, CLIPScore with the
out-of-grid scorer, CHAIR_i and CHAIR_s (COCO) and mean caption length. VQA (COCO runs): official accuracy on the
split's questions, by answer type, and on the VQA-CE counterexamples (test split). Every value has a 95% bootstrap
interval over images. --decoding greedy | beam3 (reference) | beam5 applies to captions; answers are greedy.

--select decodes the validation images with every epoch_NN.pt of the run and records CIDEr; the checkpoint with the
highest is the run's selected checkpoint (methodology §5), used by default afterwards.

Writes results/eval/<run>/<checkpoint>_<split>_<decoding>.json, with the captions and answers beside it. Generated
text is reused when the same checkpoint, split and decoding are scored again (--redecode forces decoding).
"""
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl.device import get_device  # noqa: E402  (first: sets MPS memory limits)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.data import FeatureStore  # noqa: E402
from greenvl.decode import DECODING, answer_questions, generate_captions, load_run, weights_files  # noqa: E402
from greenvl.metrics import CLIPScorer, caption_scores, cider, mean_ci, ptb_tokenize, ratio_ci  # noqa: E402
from greenvl.model import IMAGE_MODELS, SCORER  # noqa: E402

EVAL = paths.RESULTS / "eval"
SELECTION_DECODING = "beam3"


# ---------------------------------------------------------------- data

def feature_sets(family: str, split: str) -> list[str]:
    return [f"coco_{split}"] if family == "coco" else ["flickr8k"]


def references(family: str, split: str) -> dict:
    """{image_id: [reference captions]} from Datasets/processed/<family>_refs_<split>.json (COCO format)."""
    d = json.loads((paths.PROCESSED / f"{family}_refs_{split}.json").read_text())
    refs = defaultdict(list)
    for a in d["annotations"]:
        refs[a["image_id"]].append(a["caption"])
    return dict(refs)


def load_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ---------------------------------------------------------------- captions

def decode_captions(model, store, image_ids, decoding, out_path, redecode, batch):
    cached = load_json(out_path) if not redecode else None
    if cached and cached.get("decoding") == decoding and [c["image_id"] for c in cached["captions"]] == image_ids:
        return {c["image_id"]: c["caption"] for c in cached["captions"]}, cached.get("seconds")
    feats = torch.stack([store.get(i) for i in image_ids])
    t0 = time.perf_counter()
    caps = generate_captions(model, feats, decoding, batch)
    seconds = time.perf_counter() - t0
    write_json(out_path, {"decoding": decoding, "seconds": round(seconds, 1),
                          "captions": [{"image_id": i, "caption": c} for i, c in zip(image_ids, caps)]})
    return dict(zip(image_ids, caps)), seconds


def scorer_embeddings(family, split, image_ids):
    """CLIPScore image embeddings, extracted once with 02_extract_features.py if missing."""
    sets = feature_sets(family, split)
    folder = paths.FEATURES / SCORER.replace("/", "-")
    missing = [s for s in sets if not (folder / f"{s}.pt").exists()]
    if missing:
        print(f"extracting {SCORER} image embeddings for {', '.join(missing)} (once)", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("02_extract_features.py")), "--encoder", SCORER,
                        "--sets", *missing, "--idle-seconds", "0"], check=True)
    store = FeatureStore(SCORER, family, sets)
    return torch.stack([store.get(i) for i in image_ids])


def chair_ground_truth(split, image_ids):
    from greenvl.chair import Chair

    chair = Chair(paths.DATA_ROOT / "chair" / "synonyms.txt")
    cache = EVAL / f"_chair_gt_coco_{split}.json"
    gt = load_json(cache)
    if gt is None:
        all_ids = sorted(references("coco", split))
        ann = paths.DATA_ROOT / "coco" / "annotations"
        gt_sets = chair.ground_truth(all_ids, ann / "instances_val2014.json", ann / "captions_val2014.json")
        write_json(cache, {str(k): sorted(v) for k, v in gt_sets.items()})
        gt = load_json(cache)
    return chair, {i: set(gt[str(i)]) for i in image_ids}


def score_captions(args, cfg, model, run_eval, stem, image_ids, device):
    family = cfg["data"]
    refs = references(family, args.split)
    store = FeatureStore(cfg["encoder"], family, feature_sets(family, args.split))
    caps, seconds = decode_captions(model, store, image_ids, args.decoding, run_eval / f"{stem}_captions.json",
                                    args.redecode, args.batch)
    scorer = embeds = None
    if not args.no_clipscore:
        scorer = CLIPScorer(IMAGE_MODELS[SCORER][0], device)
        scorer.name = SCORER
        embeds = scorer_embeddings(family, args.split, image_ids)
    out, per_image, keys = caption_scores({i: refs[i] for i in image_ids}, caps, spice_on=not args.no_spice,
                                          clip_scorer=scorer, image_embeds=embeds)
    out["decode_seconds"] = round(seconds, 1) if seconds is not None else None
    if family == "coco":
        chair, gt = chair_ground_truth(args.split, image_ids)
        ch = chair.score({i: caps[i] for i in keys}, gt)
        mentions = np.array([ch["per_caption"][i]["mentions"] for i in keys], dtype=np.float64)
        hall = np.array([ch["per_caption"][i]["hallucinated"] for i in keys], dtype=np.float64)
        out["CHAIR_i"] = {"value": ch["CHAIR_i"], "ci95": ratio_ci(hall, mentions)}
        out["CHAIR_s"] = {"value": ch["CHAIR_s"], "ci95": mean_ci((hall > 0).astype(np.float64))}
        out["object_mentions"] = ch["object_mentions"]
        detail = load_json(run_eval / f"{stem}_captions.json")
        for c in detail["captions"]:
            c["hallucinated"] = [w for w, _ in ch["per_caption"][c["image_id"]]["hallucinated_words"]]
        write_json(run_eval / f"{stem}_captions.json", detail)
    return out


# ---------------------------------------------------------------- VQA

def score_vqa(args, cfg, model, run_eval, ckpt_stem, image_ids):
    from greenvl.vqa_metrics import question_accuracy

    split = args.split
    questions = json.loads((paths.PROCESSED / f"vqa_{split}_questions.json").read_text())["questions"]
    anns = {a["question_id"]: a for a in
            json.loads((paths.PROCESSED / f"vqa_{split}_annotations.json").read_text())["annotations"]}
    keep = set(image_ids)
    questions = sorted((q for q in questions if q["image_id"] in keep), key=lambda q: q["question_id"])
    path = run_eval / f"{ckpt_stem}_{split}_answers.json"
    cached = load_json(path) if not args.redecode else None
    qids = [q["question_id"] for q in questions]
    if cached and [a["question_id"] for a in cached["answers"]] == qids:
        answers, seconds = [a["answer"] for a in cached["answers"]], cached.get("seconds")
    else:
        store = FeatureStore(cfg["encoder"], "coco", feature_sets("coco", split))
        feats = torch.stack([store.get(q["image_id"]) for q in questions])
        t0 = time.perf_counter()
        answers = answer_questions(model, feats, [q["question"] for q in questions], args.batch)
        seconds = time.perf_counter() - t0
    acc = np.array([question_accuracy(a, anns[q]) for a, q in zip(answers, qids)])
    write_json(path, {"seconds": round(seconds, 1) if seconds else seconds,
                      "answers": [{"question_id": q, "answer": a, "accuracy": round(float(s), 4)}
                                  for q, a, s in zip(qids, answers, acc)]})

    img_index = {i: k for k, i in enumerate(sorted(keep))}

    def grouped(mask):
        num, den = np.zeros(len(img_index)), np.zeros(len(img_index))
        for q, s, m in zip(questions, acc, mask):
            if m:
                num[img_index[q["image_id"]]] += s
                den[img_index[q["image_id"]]] += 1
        return {"value": float(num.sum() / max(den.sum(), 1)), "ci95": ratio_ci(num, den), "questions": int(den.sum())}

    out = {"questions": len(qids), "accuracy": grouped([True] * len(qids)),
           "decode_seconds": round(seconds, 1) if seconds else seconds, "by_answer_type": {}}
    for t in sorted({anns[q]["answer_type"] for q in qids}):
        out["by_answer_type"][t] = grouped([anns[q]["answer_type"] == t for q in qids])
    if split == "test":
        ce = set(json.loads((paths.PROCESSED / "vqace_test_qids.json").read_text()))
        out["vqa_ce"] = grouped([q in ce for q in qids])
    return out


# ---------------------------------------------------------------- modes

def select(args, device):
    run_dir = paths.RESULTS / "runs" / args.run
    files = [p for p in weights_files(run_dir) if p.name.startswith("epoch_")] or weights_files(run_dir)
    run_eval = EVAL / args.run
    sel_path = run_eval / "selection.json"
    sel = load_json(sel_path, {"metric": f"CIDEr, validation, {SELECTION_DECODING}", "scores": {}})
    cfg = json.loads((run_dir / "config.json").read_text())
    refs = references(cfg["data"], "val")
    image_ids = sorted(refs)[: args.limit] if args.limit else sorted(refs)
    if args.limit:
        sel["limit"] = args.limit
    gts = ptb_tokenize({i: refs[i] for i in image_ids})
    for f in files:
        if f.name in sel["scores"]:
            continue
        model, cfg, _ = load_run(args.run, f.name, device)
        store = FeatureStore(cfg["encoder"], cfg["data"], feature_sets(cfg["data"], "val"))
        caps, seconds = decode_captions(model, store, image_ids, SELECTION_DECODING,
                                        run_eval / f"{f.stem}_val_{SELECTION_DECODING}_captions.json", args.redecode,
                                        args.batch)
        res = ptb_tokenize({i: [caps[i]] for i in image_ids})
        score, _ = cider(gts, res)
        sel["scores"][f.name] = round(score, 4)
        print(f"{f.name}: validation CIDEr {100 * score:.1f} ({seconds:.0f} s decoding)", flush=True)
        del model
        write_json(sel_path, sel)
    sel["best"] = max(sel["scores"], key=sel["scores"].get)
    sel["updated"] = datetime.now().isoformat(timespec="seconds")
    write_json(sel_path, sel)
    print(f"selected {sel['best']} (validation CIDEr {100 * sel['scores'][sel['best']]:.1f}) -> {sel_path}")


def evaluate(args, device):
    run_eval = EVAL / args.run
    checkpoint = args.checkpoint or (load_json(run_eval / "selection.json") or {}).get("best")
    model, cfg, ckpt = load_run(args.run, checkpoint, device)
    family = cfg["data"]
    image_ids = sorted(references(family, args.split))
    if args.limit:
        image_ids = image_ids[: args.limit]
    stem = f"{ckpt.stem}_{args.split}_{args.decoding}"
    result = {"run": args.run, "checkpoint": ckpt.name, "split": args.split, "decoding": args.decoding,
              "limit": args.limit, "time": datetime.now().isoformat(timespec="seconds"), "device": str(device)}
    tasks = args.tasks or (["caption", "vqa"] if "vqa" in cfg["tasks"] else ["caption"])
    if "caption" in tasks:
        result["caption"] = score_captions(args, cfg, model, run_eval, stem, image_ids, device)
    if "vqa" in tasks:
        result["vqa"] = score_vqa(args, cfg, model, run_eval, ckpt.stem, image_ids)
    write_json(run_eval / f"{stem}.json", result)
    print_summary(result)
    print(f"-> {run_eval / (stem + '.json')}")


def print_summary(r):
    def f(m, scale=100):
        return f"{scale * m['value']:.1f} [{scale * m['ci95'][0]:.1f}, {scale * m['ci95'][1]:.1f}]" if "ci95" in m \
            else f"{scale * m['value']:.1f}"
    print(f"\n{r['run']} {r['checkpoint']} {r['split']} ({r['decoding']})")
    c = r.get("caption")
    if c:
        parts = [f"{k} {f(c[k])}" for k in ("BLEU-4", "CIDEr", "SPICE", "CLIPScore", "CHAIR_i", "CHAIR_s") if k in c]
        print(f"  captions ({c['images']:,} images): " + " | ".join(parts) + f" | length {c['mean_length_words']:.1f} words")
    v = r.get("vqa")
    if v:
        types = ", ".join(f"{t} {f(m)}" for t, m in v["by_answer_type"].items())
        print(f"  VQA ({v['questions']:,} questions): accuracy {f(v['accuracy'])} | {types}"
              + (f" | VQA-CE {f(v['vqa_ce'])} ({v['vqa_ce']['questions']:,})" if "vqa_ce" in v else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--select", action="store_true", help="validation CIDEr for every epoch checkpoint")
    ap.add_argument("--checkpoint", default=None, help="file in the run folder (default: selected, else newest)")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--decoding", choices=list(DECODING), default="beam3")
    ap.add_argument("--tasks", nargs="+", choices=["caption", "vqa"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N images of the split (quick checks)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-spice", action="store_true")
    ap.add_argument("--no-clipscore", action="store_true")
    ap.add_argument("--redecode", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    select(args, device) if args.select else evaluate(args, device)


if __name__ == "__main__":
    main()
