"""Energy readings from the platform's own counters.

Apple Silicon: IOReport "Energy Model" counters via zeus-apple-silicon. Apple derives these from a power model
(utilisation, frequency, voltage), the same source powermetrics reports.
NVIDIA: NVML's cumulative board energy counter (Volta and newer).
Anything else: no reading. Estimates from rated TDP are never substituted (methodology §8).

Measurement boundary on Apple Silicon: SoC (CPU, GPU, GPU SRAM, ANE) plus DRAM. Display, SSD, fans and
power-adapter losses are outside it.

On the M4 Pro under macOS 27 the CPU and DRAM counters do not advance unless a privileged sampler is running
(29 Sep 2026: 0 J for CPU and DRAM over 10-75 s windows, even under a 6 W CPU load; the GPU counter advanced
normally). With `sudo powermetrics` running they advance and agree with powermetrics within ~3-4 % under load
(27 Sep 2026). So on Apple Silicon this meter keeps a low-rate powermetrics process running in the background.
It needs cached sudo credentials: run `sudo -v` shortly before a script that measures energy. A driver that
launches several scripts (04_select_lr.py) starts one keeper and passes its process id in GREENVL_KEEPER_PID, so
the child scripts reuse it and sudo is needed only once.

Window lengths use a clock that stops while the computer sleeps (lid closed), so a sleep inside a window never
counts as run time or as idle power to subtract; the sleep itself is reported as `slept_seconds`.
"""
import atexit
import os
import subprocess
import time

APPLE_COMPONENTS = {"cpu": "cpu_total_mj", "gpu": "gpu_mj", "gpu_sram": "gpu_sram_mj", "ane": "ane_mj", "dram": "dram_mj"}
POWERMETRICS = "powermetrics --samplers cpu_power,gpu_power,ane_power -i 500"
# Runs as root under sudo. The shell watches this Python process and stops powermetrics when it exits, even if
# Python is killed without running its exit handlers, so no root process is left behind.
KEEPER = ("{pm} >/dev/null 2>&1 & pm=$!; trap 'kill $pm 2>/dev/null; exit 0' TERM INT HUP; "
          "while kill -0 {parent} 2>/dev/null && kill -0 $pm 2>/dev/null; do sleep 1; done; kill $pm 2>/dev/null")
KEEPER_ENV = "GREENVL_KEEPER_PID"
SETTLE_SECONDS = 3.0  # after powermetrics starts, energy accumulated while the counters were frozen is released
SLEEP_REPORT_S = 2.0  # wall-clock minus awake time above this is reported as sleep
NO_SUDO = ("CPU and DRAM energy will not be measured: powermetrics could not start ({}). Run `sudo -v` in this "
           "Terminal and start the script again within a few minutes.")


if hasattr(time, "CLOCK_UPTIME_RAW"):  # macOS: does not advance while the system sleeps
    def awake_clock() -> float:
        return time.clock_gettime(time.CLOCK_UPTIME_RAW)
else:  # Linux: CLOCK_MONOTONIC also stops during suspend
    awake_clock = time.monotonic


class EnergyMeter:
    def __init__(self, device_type: str, keep_counters_live: bool = True):
        self.backend, self.error, self.warning = None, None, None
        self._pm = None
        self._external_keeper = None
        try:
            if device_type == "cuda":
                import pynvml

                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml = pynvml
                self.backend = "nvml"
            else:
                from zeus_apple_silicon import AppleEnergyMonitor

                self._monitor = AppleEnergyMonitor()
                self.backend = "apple_ioreport"
        except Exception as e:  # no counters on this platform
            self.error = f"{type(e).__name__}: {e}"
        self._t0, self._e0 = {}, {}
        if self.backend == "apple_ioreport":
            if keep_counters_live:
                self._start_powermetrics()
            # Discard one window so the energy released when the counters start moving is not counted anywhere.
            self._monitor.begin_window("_settle", restart=True)
            time.sleep(1.0)
            self._monitor.end_window("_settle")

    # ------------------------------------------------ powermetrics keeper

    def _start_powermetrics(self):
        pid = os.environ.get(KEEPER_ENV)
        if pid and _alive(int(pid)):
            self._external_keeper = int(pid)  # a parent process already keeps the counters live
            return
        try:
            ok = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            ok, why = False, type(e).__name__
        else:
            why = "no cached sudo credentials"
        if not ok:
            self.warning = NO_SUDO.format(why)
            print(f"WARNING: {self.warning}")
            return
        self._pm = subprocess.Popen(["sudo", "-n", "sh", "-c", KEEPER.format(pm=POWERMETRICS, parent=os.getpid())],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)
        time.sleep(SETTLE_SECONDS)
        if self._pm.poll() is not None:
            self.warning = NO_SUDO.format("powermetrics exited straight away")
            print(f"WARNING: {self.warning}")
            self._pm = None

    @property
    def counters_live(self) -> bool:
        if self.backend == "nvml":
            return True
        if self._external_keeper is not None:
            return _alive(self._external_keeper)
        return self._pm is not None and self._pm.poll() is None

    @property
    def keeper_pid(self) -> int | None:
        """Process id to pass to child scripts in GREENVL_KEEPER_PID."""
        if self._pm is not None and self._pm.poll() is None:
            return self._pm.pid
        return self._external_keeper

    def close(self):
        if self._pm is not None and self._pm.poll() is None:
            self._pm.terminate()  # sudo relays the signal to the shell, whose trap stops powermetrics
            try:
                self._pm.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._pm.kill()
        self._pm = None

    # ------------------------------------------------ windows

    @property
    def available(self) -> bool:
        return self.backend is not None

    @staticmethod
    def _apple_parts(metrics) -> dict:
        """Joules per component; None where the chip does not report that component."""
        out = {}
        for name, attr in APPLE_COMPONENTS.items():
            v = getattr(metrics, attr, None)
            out[f"{name}_j"] = v / 1000 if v is not None else None
        out["total_j"] = sum(v for v in out.values() if v is not None)
        return out

    def begin(self, label: str) -> None:
        if self.backend == "apple_ioreport":
            self._monitor.begin_window(label, restart=True)
        elif self.backend == "nvml":
            self._e0[label] = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        self._t0[label] = (awake_clock(), time.time())

    def end(self, label: str) -> dict:
        """Energy in joules for the window, split by component where the platform reports it."""
        t0, wall0 = self._t0.pop(label)
        seconds = awake_clock() - t0
        out = {"seconds": seconds}
        slept = (time.time() - wall0) - seconds
        if slept > SLEEP_REPORT_S:
            out["slept_seconds"] = round(slept, 1)
        if self.backend == "apple_ioreport":
            out.update(self._apple_parts(self._monitor.end_window(label)))
            out["counters_live"] = self.counters_live
            # CPU and DRAM draw power whenever the machine is on, so a zero over several seconds means those
            # counters did not advance; flag it, never hide it.
            stale = [c for c in ("cpu", "dram") if seconds > 5 and out.get(f"{c}_j") == 0]
            if stale:
                out["stale_counters"] = stale
        elif self.backend == "nvml":
            e1 = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            out["gpu_j"] = (e1 - self._e0.pop(label)) / 1000
            out["total_j"] = out["gpu_j"]
        else:
            out["total_j"] = None
        return out

    def cumulative(self) -> dict | None:
        """Counter values since an arbitrary fixed point (Apple only); deltas cross-check the windows."""
        if self.backend == "apple_ioreport":
            return self._apple_parts(self._monitor.get_cumulative_energy())
        return None

    def measure_idle(self, seconds: float) -> dict | None:
        """One idle window: energy per component and mean total power. Methodology §8 uses 60 s before each run."""
        if not self.available:
            return None
        self.begin("_idle")
        time.sleep(seconds)
        r = self.end("_idle")
        r["watts"] = r["total_j"] / r["seconds"]
        return r

    def idle_power_w(self, seconds: float) -> float | None:
        r = self.measure_idle(seconds)
        return r["watts"] if r else None


def _alive(pid: int) -> bool:
    """True if the process exists (it may belong to root, which only makes kill() refuse permission)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
