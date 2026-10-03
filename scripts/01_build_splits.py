#!/usr/bin/env python3
"""Step 2: build the training splits and evaluation subsets (datasets.md, "Subsets to build after download").

    python scripts/01_build_splits.py

Reads Datasets/ and writes Datasets/processed/:
  coco_karpathy.json              {split: [{image_id, file, captions}]}; train = Karpathy train + restval
  flickr8k_karpathy.json          same format; official 6,000 / 1,000 / 1,000
  coco_refs_{val,test}.json       COCO-format references for pycocoevalcap, from captions_val2014.json
  flickr8k_refs_{val,test}.json   COCO-format references from the Karpathy file
  vqa_train.json                  train2014 questions: question_id, image_id, question, answer, answer_type
  vqa_{val,test}_questions.json   official format: val2014 questions on the Karpathy val / test images
  vqa_{val,test}_annotations.json
  vqace_test_qids.json            VQA-CE counterexample question ids on the Karpathy test images
  vizwiz_val_images.json          all VizWiz val images
  vizwiz_refs_val.json            references without precanned and rejected captions
  summary.json                    counts, compared with those recorded in datasets.md

The VQA training target is the most common of the ten human answers (multiple_choice_answer).
"""
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvl import paths  # noqa: E402

D = paths.DATA_ROOT
OUT = paths.PROCESSED

# Counts recorded in datasets.md (checked 21 Sep 2026).
EXPECTED = {
    "coco.train_images": 113_287, "coco.val_images": 5_000, "coco.test_images": 5_000,
    "flickr8k.train_images": 6_000, "flickr8k.val_images": 1_000, "flickr8k.test_images": 1_000,
    "vqa.train_questions": 443_757, "vqa.val_questions_all": 214_354,
    "vqa.test_questions": 26_280, "vqa.val_questions": 26_729,
    "vqace.all": 63_298, "vqace.test": 7_801,
    "vizwiz.images": 7_750, "vizwiz.captions": 38_750, "vizwiz.precanned": 5_167, "vizwiz.rejected": 964,
    "vizwiz.images_without_refs": 208,
}


def load(path):
    with open(path) as f:
        return json.load(f)


def save(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w") as f:
        json.dump(obj, f)
    print(f"  wrote {name}")


def clean(text):
    return " ".join(text.split())


def coco_refs(image_ids, anns, file_names, description):
    ids = set(image_ids)
    return {
        "info": {"description": description},
        "licenses": [],
        "type": "captions",
        "images": [{"id": i, "file_name": file_names[i]} for i in image_ids],
        "annotations": [{"image_id": a["image_id"], "id": a["id"], "caption": a["caption"]}
                        for a in anns if a["image_id"] in ids],
    }


def karpathy_splits(name, file_of):
    data = load(D / "karpathy" / f"dataset_{name}.json")
    splits = {"train": [], "val": [], "test": []}
    for im in data["images"]:
        split = "train" if im["split"] in ("train", "restval") else im["split"]
        splits[split].append({
            "image_id": im["cocoid"] if name == "coco" else im["imgid"],
            "file": file_of(im),
            "captions": [clean(s["raw"]) for s in im["sentences"]],
            **({"sentids": im["sentids"]} if name != "coco" else {}),
        })
    return splits


def check_files(entries, root):
    """Every listed image exists on disk (one directory listing per folder, not one stat per file)."""
    listing = {}
    missing = 0
    for e in entries:
        folder, fname = os.path.split(e["file"])
        if folder not in listing:
            listing[folder] = set(os.listdir(root / folder))
        missing += fname not in listing[folder]
    return missing


def main():
    counts = {}

    print("COCO Karpathy split")
    coco = karpathy_splits("coco", lambda im: f"{im['filepath']}/{im['filename']}")
    for s, rows in coco.items():
        counts[f"coco.{s}_images"] = len(rows)
    counts["coco.train_pairs"] = sum(len(r["captions"]) for r in coco["train"])
    counts["coco.missing_files"] = check_files([r for rows in coco.values() for r in rows], paths.COCO_IMAGES)
    save(coco, "coco_karpathy.json")

    caps_val = load(D / "coco" / "annotations" / "captions_val2014.json")
    names = {im["id"]: im["file_name"] for im in caps_val["images"]}
    for s in ("val", "test"):
        ids = [r["image_id"] for r in coco[s]]
        assert all(i in names for i in ids), f"Karpathy {s} image missing from val2014"
        refs = coco_refs(ids, caps_val["annotations"], names, f"COCO val2014 captions on the Karpathy {s} images")
        counts[f"coco.{s}_ref_captions"] = len(refs["annotations"])
        save(refs, f"coco_refs_{s}.json")
    del caps_val

    print("Flickr8k Karpathy split")
    f8 = karpathy_splits("flickr8k", lambda im: im["filename"])
    for s, rows in f8.items():
        counts[f"flickr8k.{s}_images"] = len(rows)
    counts["flickr8k.train_pairs"] = sum(len(r["captions"]) for r in f8["train"])
    missing, have = 0, set(os.listdir(paths.FLICKR8K_IMAGES))
    missing = sum(r["file"] not in have for rows in f8.values() for r in rows)
    counts["flickr8k.missing_files"] = missing
    save(f8, "flickr8k_karpathy.json")
    for s in ("val", "test"):
        rows = f8[s]
        save({
            "info": {"description": f"Flickr8k {s} references (Karpathy file)"}, "licenses": [], "type": "captions",
            "images": [{"id": r["image_id"], "file_name": r["file"]} for r in rows],
            "annotations": [{"image_id": r["image_id"], "id": sid, "caption": c}
                            for r in rows for sid, c in zip(r["sentids"], r["captions"])],
        }, f"flickr8k_refs_{s}.json")

    test_ids = {r["image_id"] for r in coco["test"]}
    val_ids = {r["image_id"] for r in coco["val"]}
    del coco
    gc.collect()

    print("VQA v2 train")
    q = {x["question_id"]: x for x in load(D / "vqav2" / "v2_OpenEnded_mscoco_train2014_questions.json")["questions"]}
    anns = load(D / "vqav2" / "v2_mscoco_train2014_annotations.json")["annotations"]
    train_rows = [{
        "question_id": a["question_id"],
        "image_id": a["image_id"],
        "question": q[a["question_id"]]["question"],
        "answer": a["multiple_choice_answer"],
        "answer_type": a["answer_type"],
    } for a in anns]
    del anns, q
    gc.collect()
    counts["vqa.train_questions"] = len(train_rows)
    counts["vqa.train_images"] = len({r["image_id"] for r in train_rows})
    counts["vqa.train_images_in_karpathy_test"] = len({r["image_id"] for r in train_rows} & test_ids)
    save(train_rows, "vqa_train.json")
    del train_rows
    gc.collect()

    print("VQA v2 val -> Karpathy val / test")
    qv = load(D / "vqav2" / "v2_OpenEnded_mscoco_val2014_questions.json")
    av = load(D / "vqav2" / "v2_mscoco_val2014_annotations.json")
    counts["vqa.val_questions_all"] = len(qv["questions"])
    for s, ids in (("test", test_ids), ("val", val_ids)):
        qs = [x for x in qv["questions"] if x["image_id"] in ids]
        an = [x for x in av["annotations"] if x["image_id"] in ids]
        assert len(qs) == len(an)
        counts[f"vqa.{s}_questions"] = len(qs)
        save({**{k: v for k, v in qv.items() if k != "questions"}, "questions": qs}, f"vqa_{s}_questions.json")
        save({**{k: v for k, v in av.items() if k != "annotations"}, "annotations": an}, f"vqa_{s}_annotations.json")
        if s == "test":
            test_qids = {x["question_id"] for x in qs}
    del qv, av
    gc.collect()

    print("VQA-CE")
    ce = load(D / "vqa_ce" / "counterexamples.json")
    counts["vqace.all"] = len(ce)
    ce_test = sorted(set(ce) & test_qids)
    counts["vqace.test"] = len(ce_test)
    save(ce_test, "vqace_test_qids.json")

    print("VizWiz-Captions val")
    vw = load(D / "vizwiz" / "annotations" / "val.json")
    counts["vizwiz.images"] = len(vw["images"])
    counts["vizwiz.captions"] = len(vw["annotations"])
    counts["vizwiz.precanned"] = sum(a["is_precanned"] for a in vw["annotations"])
    counts["vizwiz.rejected"] = sum(a["is_rejected"] for a in vw["annotations"])
    usable = [a for a in vw["annotations"] if not a["is_precanned"] and not a["is_rejected"]]
    with_refs = {a["image_id"] for a in usable}
    counts["vizwiz.images_without_refs"] = sum(im["id"] not in with_refs for im in vw["images"])
    have = set(os.listdir(paths.VIZWIZ_IMAGES))
    counts["vizwiz.missing_files"] = sum(im["file_name"] not in have for im in vw["images"])
    save([{"image_id": im["id"], "file": im["file_name"]} for im in vw["images"]], "vizwiz_val_images.json")
    ref_images = [im for im in vw["images"] if im["id"] in with_refs]
    save({
        "info": {"description": "VizWiz-Captions val, precanned and rejected captions removed"}, "licenses": [],
        "type": "captions",
        "images": [{"id": im["id"], "file_name": im["file_name"]} for im in ref_images],
        "annotations": [{"image_id": a["image_id"], "id": a["id"], "caption": a["caption"]} for a in usable],
    }, "vizwiz_refs_val.json")

    counts["feature_images_total"] = (counts["coco.train_images"] + counts["coco.val_images"] + counts["coco.test_images"]
                                      + counts["flickr8k.train_images"] + counts["flickr8k.val_images"]
                                      + counts["flickr8k.test_images"] + counts["vizwiz.images"])

    print("\nCounts vs datasets.md")
    ok = True
    for k, v in counts.items():
        exp = EXPECTED.get(k)
        must_be_zero = k.endswith("missing_files") or k.endswith("_in_karpathy_test")
        status = ("OK" if v == exp else "MISMATCH") if exp is not None else ("OK" if v == 0 else "CHECK") if must_be_zero else ""
        ok &= status in ("OK", "")
        print(f"  {k:40s} {v:>9,}  {status}{'' if exp is None or v == exp else f' (expected {exp:,})'}")

    summary = {
        "all_checks_passed": ok,
        "counts": counts,
        "coco": {"train_pairs": counts["coco.train_pairs"]},
        "vqa": {"train_questions": counts["vqa.train_questions"]},
        "feature_images_total": counts["feature_images_total"],
        "vqa_train_target": "multiple_choice_answer (most common of the ten human answers)",
    }
    save(summary, "summary.json")
    print("\nAll checks passed." if ok else "\nSome checks failed - see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
