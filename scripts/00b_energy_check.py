#!/usr/bin/env python3
"""Step 1b: check that the energy counters read sensibly, component by component.

    sudo -v && python scripts/00b_energy_check.py                   # counters as the other scripts read them
    sudo -v && python scripts/00b_energy_check.py --powermetrics    # also compare with Apple's powermetrics
    python scripts/00b_energy_check.py --no-keeper                  # raw counters, no background powermetrics

Runs a short first window, three idle windows, a CPU load, a GPU load and a final idle window, and prints
mean power per component for each. Idle power should be clearly above zero; under load the working
component should dominate. With --powermetrics, Apple's own tool samples the same windows (needs sudo,
so run `sudo -v` first). Close other apps and keep the Mac on AC power.
Writes results/env/energy_check_<timestamp>.json.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from greenvl import paths
from greenvl.device import get_device, synchronize
from greenvl.energy import APPLE_COMPONENTS, EnergyMeter


def cpu_load(seconds, dev):
    a = torch.randn(1024, 1024)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a @ a


def gpu_load(seconds, dev):
    a = torch.randn(4096, 4096, device=dev)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a @ a
        synchronize(dev)


def idle(seconds, dev):
    time.sleep(seconds)


def powermetrics_start(seconds):
    return subprocess.Popen(
        ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power", "-i", "1000", "-n",
         str(max(int(seconds), 1))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def powermetrics_read(proc):
    out, err = proc.communicate(timeout=60)
    vals = {}
    for name, mw in re.findall(r"^(CPU|GPU|ANE|Combined) Power[^:]*:\s*([\d.]+)\s*mW", out, re.M):
        vals.setdefault(name.lower(), []).append(float(mw) / 1000)
    if not vals:
        return {"error": (err or out).strip()[-300:]}
    return {k: round(statistics.mean(v), 3) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--powermetrics", action="store_true", help="compare with sudo powermetrics (run `sudo -v` first)")
    ap.add_argument("--no-keeper", action="store_true",
                    help="do not run the background powermetrics that keeps CPU and DRAM counters advancing")
    args = ap.parse_args()

    dev = get_device()
    meter = EnergyMeter(dev.type, keep_counters_live=not args.no_keeper)
    if not meter.available:
        sys.exit(f"No energy counters: {meter.error}")

    s = args.seconds
    phases = [("first window", idle, 2), ("idle 1", idle, s), ("idle 2", idle, s), ("idle 3", idle, s),
              ("CPU load", cpu_load, s), (f"GPU load ({dev.type})", gpu_load, s), ("idle 4", idle, s)]
    comps = list(APPLE_COMPONENTS)
    print(f"device {dev}, backend {meter.backend}, CPU/DRAM counters kept live: {meter.counters_live}\n")
    print(f"{'window':18s}{'total W':>9s}" + "".join(f"{c + ' W':>11s}" for c in comps)
          + ("   | powermetrics W (cpu, gpu, ane, combined)" if args.powermetrics else ""))

    rows = []
    cum0 = meter.cumulative()
    for label, fn, secs in phases:
        pm = powermetrics_start(secs) if args.powermetrics else None
        meter.begin(label)
        fn(secs, dev)
        r = meter.end(label)
        row = {"window": label, "seconds": round(r["seconds"], 2), "total_w": r["total_j"] / r["seconds"],
               **{f"{c}_w": (r[f"{c}_j"] / r["seconds"] if r.get(f"{c}_j") is not None else None) for c in comps}}
        if pm:
            row["powermetrics_w"] = powermetrics_read(pm)
        rows.append(row)
        line = f"{label:18s}{row['total_w']:9.3f}" + "".join(
            f"{row[f'{c}_w']:11.3f}" if row[f"{c}_w"] is not None else f"{'n/a':>11s}" for c in comps)
        if pm:
            p = row["powermetrics_w"]
            line += "   | " + (", ".join(str(p.get(k)) for k in ("cpu", "gpu", "ane", "combined"))
                               if "error" not in p else p["error"])
        print(line)
    cum1 = meter.cumulative()

    window_sum = sum(r["total_w"] * r["seconds"] for r in rows)
    cum_delta = cum1["total_j"] - cum0["total_j"]
    print(f"\nsum of windows {window_sum:.1f} J vs cumulative counter delta {cum_delta:.1f} J "
          f"(the delta also covers the gaps between windows, so it should be slightly larger)")

    idle_w = [r["total_w"] for r in rows if r["window"].startswith("idle")]
    verdict = {
        "idle_mean_w": round(statistics.mean(idle_w), 3),
        "idle_plausible": statistics.mean(idle_w) > 0.05,
        "components_missing": [c for c in comps if all(r[f"{c}_w"] is None for r in rows)],
        "cpu_load_raises_cpu": (rows[4]["cpu_w"] or 0) > 5 * max(rows[1]["cpu_w"] or 0, 0.01),
        "gpu_load_raises_gpu": (rows[5]["gpu_w"] or 0) > 5 * max(rows[1]["gpu_w"] or 0, 0.01),
    }
    print(json.dumps(verdict, indent=2))

    out_dir = paths.RESULTS / "env"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"energy_check_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"rows": rows, "cumulative_delta_j": cum_delta, "window_sum_j": window_sum,
                               "verdict": verdict}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
