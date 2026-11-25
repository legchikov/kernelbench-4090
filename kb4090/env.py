"""Environment fingerprint: everything that can change a timing, recorded with every result.

Publish the fingerprint with your numbers. "Speedup on a 4090" means little without the driver,
CUDA, PyTorch version, clocks and whether the card was also driving a display.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, kernelbench_root

CLOCK_MARKER = "/var/tmp/kb4090-locks/clocks-gpu{idx}.json"

# Environment variables that change numerics or kernel selection. Recorded with every fingerprint.
NUMERICS_ENV = ["NVIDIA_TF32_OVERRIDE", "TRITON_DEFAULT_FP_FUSION", "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_LAUNCH_BLOCKING", "PYTORCH_CUDA_ALLOC_CONF", "TORCH_CUDNN_V8_API_DISABLED"]


def _run(cmd: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True, cwd=cwd).stdout.strip()
    except Exception:
        return None


def _git_state(path: Path) -> dict:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=path)
    dirty = _run(["git", "status", "--porcelain"], cwd=path)
    return {"commit": sha, "dirty": bool(dirty) if dirty is not None else None}


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def fingerprint(rig: dict, with_torch: bool = True) -> dict:
    idx = int(rig.get("gpu_index", 0))
    fp: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wsl": is_wsl(),
        "kb4090": _git_state(REPO_ROOT),
    }
    kb_root = kernelbench_root(rig)
    if kb_root and kb_root.exists():
        kb = _git_state(kb_root)
        kb["pinned"] = rig.get("kernelbench_commit")
        kb["matches_pin"] = (kb["commit"] == kb["pinned"]) if kb["pinned"] else None
        fp["kernelbench"] = kb
    else:
        fp["kernelbench"] = {"error": f"not found at {kb_root}"}

    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,persistence_mode,display_active,"
                "power.limit,clocks.max.sm,clocks.max.mem,pci.bus_id", "--format=csv,noheader", "-i", str(idx)])
    if smi:
        keys = ["name", "driver", "persistence_mode", "display_active", "power_limit", "max_sm_clock",
                "max_mem_clock", "pci_bus_id"]
        fp["gpu"] = dict(zip(keys, [v.strip() for v in smi.split(",")], strict=False))
    else:
        fp["gpu"] = {"error": "nvidia-smi not available"}
    nvcc = _run(["nvcc", "--version"])
    fp["nvcc"] = nvcc.splitlines()[-1] if nvcc else None

    fp["numerics_env"] = {k: os.environ.get(k) for k in NUMERICS_ENV}
    marker = Path(CLOCK_MARKER.format(idx=idx))
    fp["clocks_locked"] = json.loads(marker.read_text()) if marker.exists() else None

    if with_torch:
        try:
            import torch

            fp["torch"] = torch.__version__
            fp["torch_cuda"] = torch.version.cuda
            fp["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                fp["compute_capability"] = f"{cap[0]}.{cap[1]}"
                fp["device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            fp["torch"] = f"unavailable: {exc}"
        try:
            import triton

            fp["triton"] = triton.__version__
        except Exception:
            fp["triton"] = None
    fp["env_id"] = env_id(fp)
    return fp


def env_id(fp: dict) -> str:
    """Short id over the parts of the fingerprint that should never change within one study."""
    stable = {
        "gpu": {k: fp.get("gpu", {}).get(k) for k in ("name", "driver", "power_limit")},
        "torch": fp.get("torch"),
        "torch_cuda": fp.get("torch_cuda"),
        "triton": fp.get("triton"),
        "kernelbench": fp.get("kernelbench", {}).get("commit"),
        "clocks_locked": fp.get("clocks_locked"),
        "numerics_env": fp.get("numerics_env"),
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:12]


def warnings_for(fp: dict, hw: dict) -> list[str]:
    out = []
    if fp.get("wsl"):
        out.append("Running under WSL: Nsight Compute and the PyTorch profiler (CUPTI) may be limited; "
                   "kernel attribution can be unavailable. Native Linux is recommended for published numbers.")
    gpu = fp.get("gpu", {})
    if str(gpu.get("display_active", "")).lower() == "enabled":
        out.append("This GPU is driving a display; compositor work adds timing noise. Run headless if you can.")
    if str(gpu.get("persistence_mode", "")).lower() == "disabled":
        out.append("Persistence mode is off (sudo nvidia-smi -pm 1).")
    if not fp.get("clocks_locked"):
        out.append("Clocks are not locked (scripts/lock_clocks.sh). Boost behaviour will add run-to-run variance.")
    cc = fp.get("compute_capability")
    if cc and hw.get("compute_capability") and cc != str(hw["compute_capability"]):
        out.append(f"Compute capability {cc} does not match the hardware file ({hw['compute_capability']}).")
    kb = fp.get("kernelbench", {})
    if kb.get("matches_pin") is False:
        out.append(f"KernelBench checkout {kb.get('commit')} != pinned {kb.get('pinned')}.")
    if kb.get("dirty"):
        out.append("KernelBench checkout has local changes.")
    if fp.get("kb4090", {}).get("dirty"):
        out.append("This repo has uncommitted changes; results won't map to a commit.")
    if not hw.get("verified", False):
        out.append(f"Hardware peaks in {hw.get('_source')} are unverified. Check them against NVIDIA's "
                   "Ada whitepaper and set verified: true before publishing speed-of-light numbers.")
    if os.environ.get("NVIDIA_TF32_OVERRIDE"):
        out.append("NVIDIA_TF32_OVERRIDE is set in the environment; it changes FP32 numerics globally.")
    return out
