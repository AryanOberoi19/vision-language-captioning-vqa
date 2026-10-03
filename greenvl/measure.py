"""Shared pieces of the inference-cost measurements (scripts/08_measure_inference.py, scripts/10_measure_variants.py;
methodology §8): the fixed test subset, idle power with a disturbance check, latency percentiles, the agreement
check against the batched evaluation, and the record of the machine's state."""
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import paths
from .precision import config_tag

CO2_G_PER_KWH = 727  # CEA, Indian grid, FY 2023-24 (methodology §8), as in scripts/00_check_env.py

IDLE_ATTEMPTS = 3
IDLE_ABS_W = 1.0       # with no clean window yet, idle power above this counts as disturbed
IDLE_REL = 3.0         # later: disturbed if above IDLE_REL x the session's median clean idle ...
IDLE_MARGIN_W = 0.5    # ... and more than IDLE_MARGIN_W above it


# ---------------------------------------------------------------- subset

def subset(split: str, n: int) -> dict:
    """Fixed random sample of N images (captions) and N questions on N distinct images (answers)."""
    path = paths.RESULTS / "inference" / f"subset_{split}_{n}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(0)
    images = json.loads((paths.PROCESSED / "coco_karpathy.json").read_text())[split]
    files = {r["image_id"]: r["file"] for r in images}
    cap = sorted(rng.sample(sorted(files), n))
    questions = json.loads((paths.PROCESSED / f"vqa_{split}_questions.json").read_text())["questions"]
    by_image = {}
    for q in sorted(questions, key=lambda q: q["question_id"]):
        by_image.setdefault(q["image_id"], []).append(q)
    vqa_images = sorted(rng.sample(sorted(by_image), n))
    vqa = [rng.choice(by_image[i]) for i in vqa_images]
    out = {"split": split, "n": n, "seed": 0,
           "caption": [{"image_id": i, "file": files[i]} for i in cap],
           "vqa": [{"question_id": q["question_id"], "image_id": q["image_id"], "file": files[q["image_id"]],
                    "question": q["question"]} for q in vqa]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def warm_file_cache(items: list[dict]) -> float:
    t0 = time.perf_counter()
    for it in items:
        (paths.COCO_IMAGES / it["file"]).read_bytes()
    return time.perf_counter() - t0


# ---------------------------------------------------------------- statistics

def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean_sd(xs: list) -> tuple[float | None, float | None]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


def metal_mb(dev) -> float | None:
    return torch.mps.driver_allocated_memory() / 2**20 if dev.type == "mps" else None


# ---------------------------------------------------------------- reference outputs (agreement check)

def reference_outputs(run: str, ckpt: str, split: str, task: str, decoding: str, config: str = "fp32") -> dict:
    """Outputs of scripts/05_evaluate.py (batch 64) for the same checkpoint and configuration."""
    tag = config_tag(Path(ckpt).stem, config)
    d = paths.RESULTS / "eval" / run
    if task == "caption":
        f = d / f"{tag}_{split}_{decoding}_captions.json"
        return {c["image_id"]: c["caption"] for c in json.loads(f.read_text())["captions"]} if f.exists() else {}
    f = d / f"{tag}_{split}_answers.json"
    return {a["question_id"]: a["answer"] for a in json.loads(f.read_text())["answers"]} if f.exists() else {}


def agreement(items, outputs, ref, task) -> float | None:
    key = "image_id" if task == "caption" else "question_id"
    pairs = [(o, ref[it[key]]) for it, o in zip(items, outputs) if it[key] in ref]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


# ---------------------------------------------------------------- idle power

def measure_idle_checked(meter, seconds: float, history: list[float]) -> dict | None:
    """Idle power. macOS background jobs (Spotlight, Photos analysis) can run during an idle window and inflate it
    (2.2 W instead of ~0.2 W on 3 Oct), which would make above-idle energy too low. A window is disturbed if it
    exceeds IDLE_ABS_W (no clean window yet) or both IDLE_REL x and IDLE_MARGIN_W above the median of this
    session's clean windows; it is then measured again, up to IDLE_ATTEMPTS times, and the lowest is used if all are
    disturbed. Every attempt is recorded."""
    if not meter.available or seconds <= 0:
        return None
    attempts = []
    for _ in range(IDLE_ATTEMPTS):
        r = meter.measure_idle(seconds)
        r["background"] = top_processes()
        attempts.append(r)
        base = statistics.median(history) if history else None
        limit = IDLE_ABS_W if base is None else max(IDLE_REL * base, base + IDLE_MARGIN_W)
        r["disturbed"] = r["watts"] > limit
        if not r["disturbed"]:
            break
        last = len(attempts) == IDLE_ATTEMPTS
        print(f"    idle window disturbed ({r['watts']:.2f} W > {limit:.2f} W); "
              + ("using the lowest of the attempts" if last else "measuring again"), flush=True)
    best = min(attempts, key=lambda a: a["watts"])
    if not best["disturbed"]:
        history.append(best["watts"])
    return {**best, "attempts": len(attempts), "all_attempts_w": [round(a["watts"], 4) for a in attempts]}


def top_processes(k: int = 3) -> list[str]:
    """The k processes using most CPU right now (other than this one), for the record."""
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu=,comm="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "python" not in parts[1].lower():
            try:
                rows.append((float(parts[0]), Path(parts[1]).name))
            except ValueError:
                pass
    return [f"{name} {cpu:.0f}%" for cpu, name in sorted(rows, reverse=True)[:k]]


# ---------------------------------------------------------------- environment record

def system_state() -> dict:
    import importlib.metadata as md

    state = {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.platform()}
    for pkg in ("transformers", "bitsandbytes", "peft"):
        try:
            state[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    if sys.platform == "darwin":
        for key, cmd in (("power", ["pmset", "-g", "batt"]), ("power_settings", ["pmset", "-g"])):
            try:
                state[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
        settings = state.pop("power_settings", "")
        # macOS energy mode: "powermode" 0 automatic, 1 low power, 2 high power (older macOS: "lowpowermode" 0/1)
        state["energy_mode"] = next((ln.strip() for ln in settings.splitlines()
                                     if "powermode" in ln.lower()), None)
        state["on_ac_power"] = "AC Power" in state.get("power", "")
    return state


def peak_rss_mb() -> float:
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10
