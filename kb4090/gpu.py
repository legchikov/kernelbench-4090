"""GPU lock, L2 flushing and telemetry for a single shared GPU.

On one RTX 4090, several agents (or EAT team members) share the card. Only one process may
*time* at once, otherwise each measurement includes someone else's kernels. Every timing run
takes an exclusive file lock first. Correctness checks can still run concurrently.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path
from typing import Any

_NVML = None


def _nvml() -> Any:
    global _NVML
    if _NVML is None:
        try:
            import pynvml  # provided by the nvidia-ml-py package

            pynvml.nvmlInit()
            _NVML = pynvml
        except Exception:
            _NVML = False
    return _NVML or None


class GpuLock:
    """Exclusive, cross-process lock for timing on one GPU (``with GpuLock(0, dir): ...``)."""

    def __init__(self, gpu_index: int, lock_dir: str | Path, timeout_s: float = 3600) -> None:
        self.path = Path(lock_dir).expanduser() / f"kb4090-gpu{gpu_index}.lock"
        self.timeout_s = timeout_s
        self.waited_s = 0.0
        self._fh = None

    def __enter__(self) -> GpuLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")  # noqa: SIM115 - held for the lifetime of the lock
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start > self.timeout_s:
                    raise TimeoutError(f"waited {self.timeout_s}s for {self.path}") from None
                time.sleep(0.5)
        self.waited_s = time.monotonic() - start
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"{os.getpid()}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


class L2Flusher:
    """Evict L2 before each timed iteration by writing a buffer much larger than L2 (72 MB on a 4090).

    The buffer is allocated once, so flushing doesn't churn the caching allocator.
    """

    def __init__(self, nbytes: int, device: Any) -> None:
        import torch

        self.buf = torch.empty(max(nbytes // 4, 1), dtype=torch.int32, device=device)

    def __call__(self) -> None:
        self.buf.zero_()


def telemetry(gpu_index: int) -> dict:
    """Clocks, temperature, power and throttle reasons. Uses NVML if available, else nvidia-smi."""
    nv = _nvml()
    if nv:
        try:
            h = nv.nvmlDeviceGetHandleByIndex(gpu_index)
            out = {
                "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                "mem_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "util_pct": nv.nvmlDeviceGetUtilizationRates(h).gpu,
                "mem_used_mb": nv.nvmlDeviceGetMemoryInfo(h).used / 2**20,
            }
            getter = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
            if getter:
                out["throttle_mask"] = int(getter(h))
                out["throttle_reasons"] = _decode_throttle(nv, out["throttle_mask"])
            return out
        except Exception as exc:  # pragma: no cover - depends on the driver
            return {"error": f"nvml: {exc}"}
    return _smi_query(gpu_index, ["clocks.sm", "clocks.mem", "temperature.gpu", "power.draw",
                                  "utilization.gpu", "memory.used"])


def _decode_throttle(nv: Any, mask: int) -> list[str]:
    names = {
        "GpuIdle": "idle",
        "ApplicationsClocksSetting": "app_clocks",
        "SwPowerCap": "sw_power_cap",
        "HwSlowdown": "hw_slowdown",
        "SyncBoost": "sync_boost",
        "SwThermalSlowdown": "sw_thermal",
        "HwThermalSlowdown": "hw_thermal",
        "HwPowerBrakeSlowdown": "hw_power_brake",
        "DisplayClockSetting": "display_clocks",
    }
    found = []
    for suffix, label in names.items():
        for prefix in ("nvmlClocksEventReason", "nvmlClocksThrottleReason"):
            bit = getattr(nv, prefix + suffix, None)
            if bit is not None and mask & int(bit):
                found.append(label)
                break
    return found


def _smi_query(gpu_index: int, fields: list[str]) -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
    except Exception as exc:
        return {"error": f"nvidia-smi: {exc}"}
    values = [v.strip() for v in out.split(",")]
    return dict(zip(fields, values, strict=False))


def other_compute_pids(gpu_index: int) -> list[int]:
    """PIDs of other processes running compute work on this GPU (best effort)."""
    nv = _nvml()
    if not nv:
        return []
    try:
        h = nv.nvmlDeviceGetHandleByIndex(gpu_index)
        procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
        me = os.getpid()
        return sorted({p.pid for p in procs if p.pid != me})
    except Exception:
        return []


def wait_for_idle(gpu_index: int, max_wait_s: float, util_pct: float) -> dict:
    """Wait (up to max_wait_s) until no other process is busy on the GPU. Returns what was seen."""
    start = time.monotonic()
    seen: dict = {"waited_s": 0.0, "idle": None, "others": []}
    while True:
        others = other_compute_pids(gpu_index)
        tel = telemetry(gpu_index)
        util = tel.get("util_pct")
        try:
            util = float(util) if util is not None else None
        except (TypeError, ValueError):
            util = None
        busy = bool(others) and (util is None or util > util_pct)
        seen.update({"waited_s": time.monotonic() - start, "others": others, "util_pct": util})
        if not busy:
            seen["idle"] = True
            return seen
        if time.monotonic() - start > max_wait_s:
            seen["idle"] = False
            return seen
        time.sleep(2.0)
