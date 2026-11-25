"""Runtime anti-cheating checks that static analysis can't do.

* Kernel attribution: profile one forward pass and check that the candidate actually launched a
  kernel of its own, rather than only cuBLAS/cuDNN/ATen kernels (the "defined but not called" and
  "PyTorch fallback" hacks). A kernel counts as *custom* if it never appears when the reference
  model runs and doesn't look like a library kernel.
* Harness integrity: candidate code runs in our process, so we snapshot the functions and global
  flags the harness relies on and verify nothing was swapped out.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

# Fallback only, used when the candidate declares no kernel names we can match (see classify_kernels).
LIBRARY_KERNEL_PATTERNS = [
    r"^void at::", r"^at::", r"cublas", r"cudnn", r"cutlass", r"xmma", r"^(ampere|sm\d+|volta|turing|ada)_",
    r"^Memcpy", r"^Memset", r"^memcpy", r"^memset", r"nchwToNhwc", r"nhwcToNchw", r"implicit_convolve",
    r"^void (gemmk1|gemmSN|gemv2|gemvx|splitKreduce|dot_kernel|asum_kernel|scal_kernel)", r"^gemv2",
    r"^void cub::", r"^void thrust::", r"^void flash_", r"^triton_(poi|red|per|tem)_",
]
_LIB_RE = re.compile("|".join(LIBRARY_KERNEL_PATTERNS))


def kernel_base_name(name: str) -> str:
    """'void ns::relu_kernel<float>(float const*, ...)' -> 'relu_kernel'."""
    n = name.strip()
    if n.startswith("void "):
        n = n[5:]
    for sep in ("(", "<"):
        n = n.split(sep, 1)[0]
    return n.split("::")[-1].strip()


def profile_kernels(fn: Callable[[], Any]) -> dict:
    """Run ``fn`` once under the profiler and list the GPU kernels it launched."""
    from torch.profiler import ProfilerActivity, profile

    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        events = prof.key_averages()
    except Exception as exc:  # CUPTI can be unavailable (e.g. some WSL setups)
        return {"available": False, "error": str(exc), "kernels": []}
    kernels = []
    for e in events:
        dev = str(getattr(e, "device_type", ""))
        t_us = getattr(e, "self_device_time_total", None)
        if t_us is None:
            t_us = getattr(e, "self_cuda_time_total", 0.0)
        if "CUDA" in dev and t_us:
            kernels.append({"name": e.key, "time_us": float(t_us), "calls": int(e.count)})
    kernels.sort(key=lambda k: -k["time_us"])
    return {"available": bool(kernels), "kernels": kernels}


def classify_kernels(candidate: dict, reference: dict, declared: list[str] | None = None) -> dict:
    """Split the candidate's kernels into custom vs library.

    Primary rule: a kernel is custom if its base name is a ``__global__`` function or ``@triton.jit``
    function declared in the candidate's own source (``declared``). Only if the candidate declares
    nothing we can match do we fall back to "not seen in the reference run and not a known library
    kernel name".
    """
    ref_names = {k["name"] for k in reference.get("kernels", [])}
    declared_set = set(declared or [])
    total = sum(k["time_us"] for k in candidate.get("kernels", []))
    custom = []
    library = []
    for k in candidate.get("kernels", []):
        if declared_set:
            base = kernel_base_name(k["name"])
            # Older Triton versions append a specialisation suffix (relu_kernel_0d1d2d).
            is_custom = base in declared_set or any(base.startswith(d + "_") for d in declared_set)
        else:
            is_custom = k["name"] not in ref_names and not _LIB_RE.search(k["name"])
        (custom if is_custom else library).append({**k, "custom": is_custom})
    custom_time = sum(k["time_us"] for k in custom)
    return {
        "profiler_available": candidate.get("available", False),
        "method": "declared_names" if declared_set else "fallback_heuristic",
        "declared": sorted(declared_set),
        "n_kernels": len(candidate.get("kernels", [])),
        "n_custom": len(custom),
        "custom_fraction": custom_time / total if total else 0.0,
        "custom": custom[:20],
        "library": library[:20],
        "total_us": total,
    }


def _flag_state() -> dict:
    return {
        "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
    }


_CRITICAL = {
    "torch.allclose": lambda: torch.allclose,
    "torch.equal": lambda: torch.equal,
    "torch.isfinite": lambda: torch.isfinite,
    # torch.manual_seed is NOT listed: the PyTorch profiler itself rewraps it on first use.
    "torch.cuda.synchronize": lambda: torch.cuda.synchronize,
    "torch.cuda.Event": lambda: torch.cuda.Event,
    "torch.cuda.Event.record": lambda: torch.cuda.Event.record,
    "torch.cuda.Event.elapsed_time": lambda: torch.cuda.Event.elapsed_time,
    "torch.Tensor.__sub__": lambda: torch.Tensor.__sub__,
    "torch.Tensor.abs": lambda: torch.Tensor.abs,
    "torch.Tensor.to": lambda: torch.Tensor.to,
    "torch.Tensor.sum": lambda: torch.Tensor.sum,
    "torch.nn.Module.__call__": lambda: torch.nn.Module.__call__,
    "time.perf_counter": lambda: time.perf_counter,
    "time.monotonic": lambda: time.monotonic,
}


def prewarm_harness(device: torch.device) -> None:
    """Run the profiler and FLOP counter once before snapshotting, so their own lazy patches of
    torch internals are not mistaken for tampering by the candidate."""
    from torch.utils.flop_counter import FlopCounterMode

    a = torch.ones(4, 4, device=device)
    profile_kernels(lambda: a @ a)
    with FlopCounterMode(display=False):
        _ = a @ a


class HarnessIntegrity:
    def __init__(self) -> None:
        self.ids = {name: id(get()) for name, get in _CRITICAL.items()}
        self.flags = _flag_state()

    def verify(self) -> list[str]:
        changed = [name for name, get in _CRITICAL.items() if id(get()) != self.ids[name]]
        now = _flag_state()
        changed += [f"flag:{k}" for k, v in self.flags.items() if now.get(k) != v]
        return changed


def thread_count() -> int:
    return threading.active_count()
