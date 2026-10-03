#!/usr/bin/env python3
"""Step 4: train the mapping network and the adapter sets (methodology §5).

    python scripts/03_train.py --data flickr8k --run-name f8k_lr1e-4 --lr 1e-4 --epochs 5
    python scripts/03_train.py --data coco --run-name ref_s0 --seed 0
    python scripts/03_train.py --data flickr8k --run-name smoke --max-steps 20 --idle-seconds 0   # smoke test

--data flickr8k trains the captioning head only (Flickr8k has no VQA questions); --data coco trains both heads
on the COCO Karpathy train split and all VQA v2 train2014 questions, one task per batch with equal probability.

Writes results/runs/<run-name>/:
  config.json   all settings
  log.jsonl     step logs (loss per task, learning rate, samples/s), epoch summaries (train and validation loss
                per task, time, energy), the idle-power window
  epoch_NN.pt   trained weights only, one per epoch, for checkpoint selection by validation CIDEr (step 5)
  last.pt       weights, optimiser, scheduler, random state and position; re-running the same command resumes
                from it. Written every --save-every steps, at every epoch end, and when the run stops.

Pausing: Ctrl+C (or SIGTERM, or closing the Terminal window) finishes the current step, saves last.pt and exits with code 130; a second Ctrl+C
stops at once and the next run resumes from the previous save. Resuming restores the weights, optimiser, learning
rate, random state and position in the data order, so the finished run matches an uninterrupted one. An epoch that
spans several sessions has its time, energy and training loss added up across them (stored in last.pt between
sessions); its summary records how many sessions it took, and any time the computer slept inside it.

Adding epochs: with the default constant learning rate after warm-up, re-running a run with a larger --epochs
(or a larger or no --max-steps) continues it from where it stopped, and the result is identical to having asked
for that much training from the start (same data order, same random state). A --max-steps stop inside an epoch
saves step_NNNNNN.pt and a "partial" log entry with validation losses. A --schedule linear run decays to zero over
its planned epochs, so it cannot be extended; train it again under a new name instead.
"""
import argparse
import json
import itertools
import math
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenvl.device import empty_cache, get_device, memory_mb, synchronize  # first: sets MPS memory limits

import torch  # noqa: E402

from greenvl import paths  # noqa: E402
from greenvl.data import (CaptionDataset, FeatureStore, TaskMixer, VQADataset, caption_rows, load_tokenizer,
                          make_loader, vqa_rows)
from greenvl.energy import EnergyMeter, awake_clock
from greenvl.lock import RunLock
from greenvl.model import ADAPTATIONS, DECODERS, ENCODERS, TASKS, CaptionVQAModel, build_decoder

AMP = {"fp32": None, "bf16": torch.bfloat16}
# Settings that must match for a run to resume.
RESUME_KEYS = ["data", "tasks", "encoder", "decoder", "adaptation", "rank", "epochs", "batch", "lr", "weight_decay",
               "warmup_epochs", "schedule", "clip", "amp", "seed", "max_steps"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, choices=["flickr8k", "coco"])
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=None,
                    help="default: caption for flickr8k, caption and vqa for coco")
    ap.add_argument("--encoder", default="ViT-B/32", choices=list(ENCODERS))
    ap.add_argument("--decoder", default="gpt2", choices=list(DECODERS))
    ap.add_argument("--adaptation", default="lora", choices=ADAPTATIONS)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=float, default=0.2,
                    help="linear warm-up over this share of one epoch (independent of --epochs)")
    ap.add_argument("--schedule", choices=["constant", "linear"], default="constant",
                    help="after warm-up: constant (runs can be extended) or linear decay to zero over --epochs")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping")
    ap.add_argument("--amp", choices=list(AMP), default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000, help="steps between resumable checkpoints")
    ap.add_argument("--val-batches", type=int, default=0, help="validation batches per task per epoch (0 = all)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (smoke tests)")
    ap.add_argument("--save-epochs", choices=["all", "last"], default="all",
                    help="keep every epoch_NN.pt (for checkpoint selection) or only the latest (development runs)")
    ap.add_argument("--empty-cache-every", type=int, default=100,
                    help="release the MPS allocator's cached memory every N steps (0 = never)")
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true", help="ignore an existing run directory and start fresh")
    args = ap.parse_args()
    if args.tasks is None:
        args.tasks = ["caption"] if args.data == "flickr8k" else list(TASKS)
    if args.data == "flickr8k" and "vqa" in args.tasks:
        ap.error("Flickr8k has no VQA questions; use --data coco for the VQA head")
    if args.run_name is None:
        args.run_name = (f"{args.data}_{args.encoder.replace('/', '-')}_{args.adaptation}"
                         f"{args.rank if args.adaptation == 'lora' else ''}_s{args.seed}")
    return args


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad or "lora_" in n}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def load_trainable(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = {n for n, p in model.named_parameters() if p.requires_grad or "lora_" in n}
    assert not unexpected and not (expected - set(state)), "checkpoint does not match this configuration"


def lr_lambda(total, warmup, schedule):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def rng_state(dev):
    s = {"torch": torch.get_rng_state()}
    if dev.type == "mps":
        s["mps"] = torch.mps.get_rng_state()
    elif dev.type == "cuda":
        s["cuda"] = torch.cuda.get_rng_state()
    return s


def set_rng_state(dev, s):
    torch.set_rng_state(s["torch"])
    if dev.type == "mps" and "mps" in s:
        torch.mps.set_rng_state(s["mps"])
    elif dev.type == "cuda" and "cuda" in s:
        torch.cuda.set_rng_state(s["cuda"])


class StopRequest:
    """Ctrl+C, SIGTERM or SIGHUP (Terminal window closed) while training: finish the current step, save, exit.
    A second one stops at once, and then nothing is saved (the step in progress may be half applied; the previous
    last.pt stays)."""

    def __init__(self):
        self.requested = self.forced = False
        self.armed = False  # outside the training loop a signal stops at once: nothing unsaved is lost
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGHUP, self._handle)

    def _handle(self, signum, frame):
        if self.armed and not self.requested:
            self.requested = True
            print("\nstopping after the current step (Ctrl+C again to stop at once) ...", flush=True)
            return
        self.forced = self.armed
        raise KeyboardInterrupt


@torch.no_grad()
def validate(model, loaders, dev, amp_dtype, max_batches, stop=None):
    model.eval()
    out = {}
    for task, loader in loaders.items():
        model.set_task(task)
        total, tokens = 0.0, 0
        for i, b in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if stop is not None and stop.requested:
                raise KeyboardInterrupt  # validation is repeated on resume
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                loss = model(b["feats"], b["input_ids"], b["attention_mask"], b["labels"])
            n = int((b["labels"] != -100).sum())  # the loss is a mean over these tokens; weight batches by it
            total += loss.item() * n
            tokens += n
        out[task] = total / max(tokens, 1)
    model.train()
    return out


def main():
    args = parse_args()
    stop = StopRequest()
    run_dir = paths.RESULTS / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    RunLock(run_dir / ".lock")
    cfg = {k: v for k, v in vars(args).items()}
    last_path = run_dir / "last.pt"

    resume = None
    if last_path.exists() and not args.overwrite:
        resume = torch.load(last_path, weights_only=False)
        keys = [k for k in RESUME_KEYS if not (k in ("epochs", "max_steps") and args.schedule == "constant")]
        diff = {k: (resume["config"].get(k), cfg[k]) for k in keys if resume["config"].get(k) != cfg[k]}
        if diff:
            sys.exit(f"{run_dir} holds a run with different settings {diff}; use another --run-name or --overwrite")
        if args.epochs < resume["epoch"]:
            sys.exit(f"{args.run_name} already has {resume['epoch']} epochs; its epoch_NN.pt files hold each one")

    torch.manual_seed(args.seed)
    dev = get_device(args.device)
    amp_dtype = AMP[args.amp]

    # ------------------------------------------------ data
    t0 = time.perf_counter()
    tokenizer = load_tokenizer(args.decoder)
    store = FeatureStore(args.encoder, args.data)
    train_sets, val_sets = {}, {}
    if "caption" in args.tasks:
        train_sets["caption"] = CaptionDataset(caption_rows(args.data, "train"), tokenizer, store)
        val_sets["caption"] = CaptionDataset(caption_rows(args.data, "val"), tokenizer, store)
    if "vqa" in args.tasks:
        train_sets["vqa"] = VQADataset(vqa_rows("train"), tokenizer, store)
        val_sets["vqa"] = VQADataset(vqa_rows("val"), tokenizer, store)
    pad = tokenizer.pad_token_id
    mixer = TaskMixer(train_sets, {t: 1 / len(args.tasks) for t in args.tasks}, args.batch, pad, args.seed)
    val_loaders = {t: make_loader(d, args.batch, pad, shuffle=False) for t, d in val_sets.items()}
    total_steps = args.epochs * mixer.steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, round(args.warmup_epochs * mixer.steps_per_epoch))
    if resume and resume["global_step"] >= total_steps and resume.get("finished"):
        print(f"{args.run_name} already has {resume['global_step']:,} steps ({resume['epoch']} full epochs); "
              "raise --epochs or --max-steps to continue it")
        return
    print(f"data ready in {time.perf_counter() - t0:.0f} s: " + ", ".join(f"{t} {len(d):,}" for t, d in train_sets.items())
          + f"; {mixer.steps_per_epoch:,} steps/epoch, {total_steps:,} steps total, warm-up {warmup:,}")

    # ------------------------------------------------ model and optimiser
    model = CaptionVQAModel(build_decoder(args.decoder, args.adaptation, args.rank), store.dim).to(dev)
    model.train()
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total_steps, warmup, args.schedule))
    counts = model.parameter_counts()
    print(f"device {dev}; trainable {counts['trainable_total']:,} of {counts['total']:,} parameters")

    start_epoch, skip, global_step = 0, 0, 0
    if resume:
        load_trainable(model, resume["weights"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        set_rng_state(dev, resume["rng"])
        start_epoch, skip, global_step = resume["epoch"], resume["step_in_epoch"], resume["global_step"]
        verb = "extending" if resume.get("finished") else "resuming"
        print(f"{verb} {args.run_name} at epoch {start_epoch}, step {skip} of the epoch ({global_step:,} overall), "
              f"training to {args.epochs} epochs")
    else:
        (run_dir / "log.jsonl").write_text("")
    (run_dir / "config.json").write_text(json.dumps({**cfg, "parameters": counts, "total_steps": total_steps,
                                                    "warmup_steps": warmup,
                                                    "steps_per_epoch": mixer.steps_per_epoch}, indent=2))
    log_file = open(run_dir / "log.jsonl", "a")

    def log(entry):
        log_file.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")
        log_file.flush()

    def save_last(epoch, step_in_epoch, finished=False, carry=None):
        """carry: time, energy and loss totals of an epoch that is not finished yet (see below)."""
        tmp = last_path.with_suffix(".tmp")
        torch.save({"weights": trainable_state(model), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "rng": rng_state(dev), "epoch": epoch, "step_in_epoch": step_in_epoch, "global_step": global_step,
                    "config": cfg, "finished": finished, "carry": carry}, tmp)
        tmp.replace(last_path)  # atomic: a stop during the write leaves the previous last.pt intact

    meter = EnergyMeter(dev.type)
    idle = meter.measure_idle(args.idle_seconds) if meter.available and args.idle_seconds > 0 else None
    log({"type": "start", "resumed": bool(resume), "from_epoch": start_epoch, "to_epochs": args.epochs,
         "device": str(dev), "energy_backend": meter.backend, "energy_counters_live": meter.counters_live,
         "energy_warning": meter.warning, "idle_window": idle})
    idle_w = idle["watts"] if idle else None

    # ------------------------------------------------ per-epoch totals across sessions
    # Training time and energy are measured in windows that close at every save of last.pt. The totals of the
    # current epoch up to that save go into last.pt ("carry"), so an epoch finished in a later session reports the
    # whole epoch. Steps after the last save are repeated on resume and counted once, in the session that keeps them.
    def new_totals(epoch):
        return {"epoch": epoch, "seconds": 0.0, "energy": None, "above_idle_j": 0.0, "slept_seconds": 0.0,
                "sums": {t: [0.0, 0] for t in args.tasks}, "sessions": 1}

    win = {"open": False}

    def open_window():
        if meter.available:
            meter.begin("train")
        win.update(open=True, t=awake_clock(), wall=time.time())

    def close_window(tot):
        if not win["open"]:
            return
        win["open"] = False
        synchronize(dev)
        seconds = awake_clock() - win["t"]
        tot["seconds"] += seconds
        slept = (time.time() - win["wall"]) - seconds
        if slept > 2:
            tot["slept_seconds"] += slept
        if not meter.available:
            tot["above_idle_j"] = None
            return
        e = meter.end("train")
        tot["energy"] = tot["energy"] or {}
        for k, v in e.items():
            if k.endswith("_j"):
                tot["energy"][k] = None if v is None else (tot["energy"].get(k) or 0.0) + v
        if e.get("stale_counters"):
            tot["stale_counters"] = sorted(set(tot.get("stale_counters", [])) | set(e["stale_counters"]))
        if not e.get("counters_live", True):
            tot["counters_live"] = False
        if tot["above_idle_j"] is not None and idle_w is not None and e.get("total_j") is not None:
            tot["above_idle_j"] += e["total_j"] - idle_w * seconds
        else:
            tot["above_idle_j"] = None

    def energy_summary(tot):
        if tot["energy"] is None:
            return None
        out = {"seconds": round(tot["seconds"], 1),
               **{k: (round(v, 3) if v is not None else None) for k, v in tot["energy"].items()},
               "counters_live": tot.get("counters_live", True)}
        if tot.get("stale_counters"):
            out["stale_counters"] = tot["stale_counters"]
        return out

    # ------------------------------------------------ training
    carry = resume.get("carry") if resume else None
    position = [start_epoch, skip]  # last completed (epoch, step within epoch); saved if the run stops
    tot = None
    stop.armed = True
    try:
        for epoch in range(start_epoch, args.epochs):
            if carry and carry["epoch"] == epoch:  # this epoch was started in an earlier session
                tot = carry
                tot["sessions"] += 1
            else:
                tot = new_totals(epoch)
            carry = None
            sums = tot["sums"]
            window = {t: [0.0, 0] for t in args.tasks}
            # Everything was trained before a stop during the final validation: validate and finish only.
            exhausted = global_step >= total_steps
            done = exhausted and bool(args.max_steps)
            batches = iter(()) if exhausted else mixer.epoch(epoch, skip)
            first = next(batches, None)  # replays the part of an interrupted epoch already trained, unmeasured
            seen, t_log = 0, time.perf_counter()
            open_window()
            for step, task, b in itertools.chain([first] if first else [], batches):
                model.set_task(task)
                b = {k: v.to(dev) for k, v in b.items()}
                with torch.autocast(dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    loss = model(b["feats"], b["input_ids"], b["attention_mask"], b["labels"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                position = [epoch, step + 1]
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite loss at step {global_step} ({task})")
                for acc in (sums, window):
                    acc[task][0] += value
                    acc[task][1] += 1
                seen += b["input_ids"].shape[0]

                if global_step % args.log_every == 0:
                    dt = time.perf_counter() - t_log
                    entry = {"type": "step", "epoch": epoch, "step": global_step, "lr": sched.get_last_lr()[0],
                             "samples_per_s": round(seen / dt, 1), "memory_mb": memory_mb(dev),
                             **{f"loss_{t}": round(s_ / n, 4) for t, (s_, n) in window.items() if n}}
                    log(entry)
                    eta_h = (total_steps - global_step) * dt / args.log_every / 3600
                    mem = f", {entry['memory_mb'] / 1024:.1f} GB" if entry["memory_mb"] else ""
                    print(f"epoch {epoch} step {global_step:,}/{total_steps:,} "
                          + " ".join(f"{k[5:]} {v:.3f}" for k, v in entry.items() if k.startswith("loss_"))
                          + f" | {entry['samples_per_s']} samples/s{mem}, lr {entry['lr']:.2e}, ETA {eta_h:.1f} h")
                    window = {t: [0.0, 0] for t in args.tasks}
                    seen, t_log = 0, time.perf_counter()
                if global_step % args.save_every == 0:
                    close_window(tot)
                    save_last(epoch, step + 1, carry=tot)
                    open_window()
                if args.empty_cache_every and global_step % args.empty_cache_every == 0:
                    empty_cache(dev)
                if args.max_steps and global_step >= args.max_steps:
                    done = True
                    break
                if stop.requested:
                    raise KeyboardInterrupt
            close_window(tot)
            skip = 0

            t_val = time.perf_counter()
            val = validate(model, val_loaders, dev, amp_dtype, args.val_batches, stop)
            val_seconds = time.perf_counter() - t_val
            empty_cache(dev)
            partial = done and position[1] < mixer.steps_per_epoch  # stopped by --max-steps inside the epoch
            ckpt = f"step_{global_step:06d}.pt" if partial else f"epoch_{epoch:02d}.pt"
            torch.save(trainable_state(model), run_dir / ckpt)
            if args.save_epochs == "last":  # development runs: keep only the newest weights file
                for old in [*run_dir.glob("epoch_*.pt"), *run_dir.glob("step_*.pt")]:
                    if old.name != ckpt:
                        old.unlink()
            energy = energy_summary(tot)
            summary = {"type": "partial" if partial else "epoch", "epoch": epoch, "step": global_step,
                       "step_in_epoch": position[1], "checkpoint": ckpt, "train_seconds": round(tot["seconds"], 1),
                       "val_seconds": round(val_seconds, 1), "memory_mb": memory_mb(dev),
                       **{f"train_loss_{t}": round(s_ / n, 4) for t, (s_, n) in sums.items() if n},
                       **{f"val_loss_{t}": round(v, 4) for t, v in val.items()}, "train_energy": energy,
                       "sessions": tot["sessions"]}
            if tot["above_idle_j"] is not None and energy is not None:
                summary["train_energy_above_idle_j"] = round(tot["above_idle_j"], 1)
            if tot["slept_seconds"]:
                summary["slept_seconds"] = round(tot["slept_seconds"], 1)
            log(summary)
            print(f"== epoch {epoch}{f' (partial, step {global_step:,})' if partial else ''}: "
                  + " ".join(f"{k} {v}" for k, v in summary.items()
                                                 if k.startswith(("train_loss", "val_loss")))
                  + f" | train {tot['seconds'] / 60:.1f} min"
                  + (f", {energy['total_j'] / 3600:.2f} Wh" if energy and energy.get("total_j") is not None else "")
                  + (f", {tot['sessions']} sessions" if tot["sessions"] > 1 else "")
                  + (f" (WARNING: {'/'.join(energy['stale_counters'])} energy counters did not advance)"
                     if energy and energy.get("stale_counters") else ""), flush=True)
            if partial:
                save_last(*position, finished=True, carry=tot)  # a longer run continues this epoch
            else:
                save_last(epoch + 1, 0, finished=done or epoch == args.epochs - 1)
                position = [epoch + 1, 0]
                tot = None
            if done:
                break
    except (KeyboardInterrupt, Exception) as e:
        if tot is not None:
            close_window(tot)
        saved = not stop.forced
        if saved:
            save_last(*position, carry=tot if tot is not None and tot["epoch"] == position[0] else None)
        log({"type": "error", "error": f"{type(e).__name__}: {e}", "position": position, "global_step": global_step,
             "saved": saved, "memory_mb": memory_mb(dev), "traceback": traceback.format_exc()[-4000:]})
        if saved:
            print(f"\nstopped at epoch {position[0]}, step {position[1]:,} of the epoch; progress saved to last.pt. "
                  "Re-run the same command to resume.", flush=True)
        else:
            print("\nstopped at once; the next run resumes from the previous save of last.pt.", flush=True)
        if isinstance(e, KeyboardInterrupt):
            sys.exit(130)
        raise
    finally:
        stop.armed = False
        meter.close()

    log({"type": "end", "global_step": global_step})
    log_file.close()
    print(f"finished: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped before training started; nothing to save.", flush=True)
        sys.exit(130)
