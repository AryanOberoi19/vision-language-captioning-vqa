#!/usr/bin/env python3
"""Step 3: cache CLIP image embeddings. The encoder is frozen, so every image is encoded once (methodology §3).

    python scripts/02_extract_features.py --encoder ViT-B/32
    python scripts/02_extract_features.py --encoder all
    python scripts/02_extract_features.py --encoder ViT-B/32 --limit 64      # smoke test, writes to features/_smoke
    python scripts/02_extract_features.py --encoder ViT-L/14 --precision int8 --sets coco_test   # step 8

Sets: coco_train, coco_val and coco_test (Karpathy 113,287 / 5,000 / 5,000), flickr8k (all 8,000), vizwiz_val
(7,750). 139,037 images per encoder.

Writes Datasets/features/<encoder>/<set>.pt = {"ids", "feats"}: projected CLIP image embeddings in float32,
not normalised (the model normalises). Preprocessing is the encoder's own Hugging Face image processor.
--precision fp16 | int8 | nf4 runs the encoder at that precision (greenvl/precision.py; step 8) and writes to
Datasets/features/<encoder>_<precision>/.
Logs time and energy per set to results/features/ (methodology §8: the one-time extraction is reported against
the energy of re-encoding every epoch). Measures idle power for --idle-seconds first; close other apps.

Stopping and resuming: finished sets are kept. The set in progress is saved in shards of 8,192 images
(<set>.parts/ beside the output); running the command again continues after the last complete shard, and the
set's time and energy are the sum over its shards (each measured once, in the session that kept it).
"""
import argparse
import importlib.metadata as md
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl.device import get_device, synchronize  # first: sets MPS memory limits

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.energy import EnergyMeter, awake_clock
from greenvl.model import ENCODERS, IMAGE_MODELS, SCORER
from greenvl.precision import PRECISIONS, compute_dtype, feature_name, load_encoder

SETS = ["coco_train", "coco_val", "coco_test", "flickr8k", "vizwiz_val"]
SHARD = 8192  # images per saved shard
SESSION = datetime.now().strftime("%Y%m%d_%H%M%S")


def encoder_slug(name: str) -> str:
    return name.replace("/", "-")


def load_set(name: str):
    """(image root, [(image_id, relative file)]) for one image set."""
    P = paths.PROCESSED
    if name in ("coco_train", "coco_val", "coco_test"):
        rows = json.loads((P / "coco_karpathy.json").read_text())[name.split("_")[1]]
        return paths.COCO_IMAGES, [(r["image_id"], r["file"]) for r in rows]
    if name == "flickr8k":
        data = json.loads((P / "flickr8k_karpathy.json").read_text())
        return paths.FLICKR8K_IMAGES, [(r["image_id"], r["file"]) for s in ("train", "val", "test") for r in data[s]]
    if name == "vizwiz_val":
        rows = json.loads((P / "vizwiz_val_images.json").read_text())
        return paths.VIZWIZ_IMAGES, [(r["image_id"], r["file"]) for r in rows]
    raise ValueError(name)


class ImageSet(Dataset):
    """Decodes and preprocesses in DataLoader workers (module-level so macOS spawn can pickle it)."""

    def __init__(self, root, items, processor):
        self.root, self.items, self.processor = Path(root), items, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image_id, rel = self.items[i]
        with Image.open(self.root / rel) as im:
            pixels = self.processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return pixels, image_id


def load_shards(parts: Path) -> list[dict]:
    """Complete shards in order; stops at the first gap (a shard file is written atomically)."""
    shards, end = [], 0
    for f in sorted(parts.glob("*.pt")):
        d = torch.load(f)
        if d["start"] != end:
            break
        shards.append(d)
        end += len(d["ids"])
    return shards


def extract(model, processor, root, items, dev, batch, workers, parts: Path, meter, idle_w, dtype=torch.float32):
    """Encode items in order, saving a shard every SHARD images; resumes after the last complete shard."""
    parts.mkdir(parents=True, exist_ok=True)
    shards = load_shards(parts)
    done = sum(len(d["ids"]) for d in shards)
    if done:
        print(f"    resuming after {done:,} of {len(items):,} images ({len(shards)} shards)", flush=True)
    loader = DataLoader(ImageSet(root, items[done:], processor), batch_size=batch, num_workers=workers,
                        shuffle=False, persistent_workers=False)
    buf_ids, buf_feats = [], []

    def begin():
        if meter.available:
            meter.begin("shard")
        return awake_clock()

    def flush(t0):
        nonlocal done
        synchronize(dev)
        energy = meter.end("shard") if meter.available else None
        seconds = awake_clock() - t0
        ids, feats = torch.cat(buf_ids), torch.cat(buf_feats)
        above = (energy["total_j"] - idle_w * seconds if energy and energy["total_j"] is not None
                 and idle_w is not None else None)
        shard = {"start": done, "ids": ids, "feats": feats, "seconds": seconds, "energy": energy,
                 "above_idle_j": above, "session": SESSION}
        tmp = parts / f"{done:07d}.tmp"
        torch.save(shard, tmp)
        tmp.replace(parts / f"{done:07d}.pt")
        shards.append(shard)
        done += len(ids)
        buf_ids.clear()
        buf_feats.clear()

    t0 = begin()
    with torch.inference_mode():
        for pixels, image_ids in loader:
            emb = model(pixel_values=pixels.to(dev, dtype)).image_embeds
            buf_feats.append(emb.float().cpu())
            buf_ids.append(image_ids)
            if sum(len(x) for x in buf_ids) >= SHARD:
                flush(t0)
                t0 = begin()
    if buf_ids:
        flush(t0)
    elif meter.available:
        meter.end("shard")
    return shards


def combine(shards: list[dict]) -> dict:
    """Ids, features, time and energy of a set from its shards."""
    out = {"ids": torch.cat([d["ids"] for d in shards]), "feats": torch.cat([d["feats"] for d in shards]),
           "seconds": sum(d["seconds"] for d in shards), "sessions": len({d["session"] for d in shards})}
    energies = [d["energy"] for d in shards]
    if all(e is not None for e in energies):
        e = {"seconds": out["seconds"]}
        for k in energies[0]:
            if k.endswith("_j"):
                e[k] = None if energies[0][k] is None else sum(x[k] for x in energies)
        stale = sorted({c for x in energies for c in x.get("stale_counters", [])})
        if stale:
            e["stale_counters"] = stale
        e["counters_live"] = all(x.get("counters_live", True) for x in energies)
        out["energy"] = e
    else:
        out["energy"] = None
    above = [d["above_idle_j"] for d in shards]
    out["above_idle_j"] = sum(above) if all(a is not None for a in above) else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=list(IMAGE_MODELS) + ["all"],
                    help=f"a grid encoder, all three, or the CLIPScore scorer ({SCORER})")
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--precision", default="fp32", choices=PRECISIONS, help="encoder precision (step 8)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--device", default=None)
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--limit", type=int, default=None, help="first N images per set; output goes to features/_smoke")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dev = get_device(args.device)
    encoders = list(ENCODERS) if args.encoder == "all" else [args.encoder]
    if args.precision != "fp32" and SCORER in encoders:
        sys.exit("the CLIPScore scorer always runs in fp32")
    feat_root = paths.FEATURES / "_smoke" if args.limit else paths.FEATURES
    meter = EnergyMeter(dev.type)
    print(f"device {dev}, energy backend {meter.backend or meter.error}")
    idle = meter.measure_idle(args.idle_seconds) if meter.available and args.idle_seconds > 0 else None
    idle_w = idle["watts"] if idle else None
    if idle:
        print(f"idle power {idle_w:.3f} W over {idle['seconds']:.0f} s")

    for name in encoders:
        fname = feature_name(name, args.precision)
        out_dir = feat_root / encoder_slug(fname)
        out_dir.mkdir(parents=True, exist_ok=True)
        model, processor, quantized = load_encoder(name, args.precision, dev)
        log = {"encoder": name, "precision": args.precision, "quantized_layers": quantized,
               "hf_id": IMAGE_MODELS[name][0], "device": str(dev), "batch": args.batch,
               "workers": args.workers, "timestamp": datetime.now().isoformat(timespec="seconds"),
               "versions": {p: md.version(p) for p in ("torch", "transformers", "pillow")},
               "idle_window": idle, "sets": {}}
        res_dir = paths.RESULTS / "features"
        res_dir.mkdir(parents=True, exist_ok=True)
        log_path = res_dir / f"extract_{encoder_slug(fname)}{'_smoke' if args.limit else ''}_{SESSION}.json"
        for s in args.sets:
            path = out_dir / f"{s}.pt"
            if path.exists() and not args.overwrite:
                print(f"  {fname} {s}: exists, skipping ({path})")
                continue
            root, items = load_set(s)
            if args.limit:
                items = items[: args.limit]
            parts = out_dir / f"{s}.parts"
            if args.overwrite and parts.exists():
                shutil.rmtree(parts)
            r = combine(extract(model, processor, root, items, dev, args.batch, args.workers, parts, meter, idle_w,
                                compute_dtype(args.precision)))
            ids, feats, seconds, energy = r["ids"], r["feats"], r["seconds"], r["energy"]

            assert ids.tolist() == [i for i, _ in items], "missing, duplicate or misordered ids"
            assert torch.isfinite(feats).all(), "non-finite embeddings"
            tmp = path.with_suffix(".tmp")
            torch.save({"ids": ids, "feats": feats, "encoder": name, "precision": args.precision,
                        "hf_id": IMAGE_MODELS[name][0]}, tmp)
            tmp.replace(path)
            shutil.rmtree(parts)

            entry = {"images": len(items), "seconds": round(seconds, 1), "images_per_s": round(len(items) / seconds, 1),
                     "dim": feats.shape[1], "norm_mean": round(feats.norm(dim=1).mean().item(), 3), "file": str(path),
                     "sessions": r["sessions"]}
            if energy and energy["total_j"] is not None:
                entry["energy"] = energy
                entry["j_per_image"] = round(energy["total_j"] / len(items), 4)
                if r["above_idle_j"] is not None:
                    entry["j_per_image_above_idle"] = round(r["above_idle_j"] / len(items), 4)
            log["sets"][s] = entry
            log_path.write_text(json.dumps(log, indent=2, default=str))  # after every set, so a stop loses none
            print(f"  {fname} {s}: {len(items):,} images, {seconds:.0f} s, {entry['images_per_s']} img/s, "
                  f"{entry.get('j_per_image_above_idle', entry.get('j_per_image'))} J/image above idle")
        del model
    print("done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped; complete shards are saved. Run the same command to continue.", flush=True)
        sys.exit(130)
