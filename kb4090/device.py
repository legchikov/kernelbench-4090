"""Device helpers. Real runs always use CUDA.

``KB4090_DEVICE=cpu`` exists only so the whole pipeline can be exercised in CI without a GPU
(tests/test_pipeline_cpu.py). CPU timings mean nothing and must never be published.
"""

from __future__ import annotations

import os
import time

import torch

_perf_counter = time.perf_counter


def get_device() -> torch.device:
    return torch.device(os.environ.get("KB4090_DEVICE", "cuda:0"))


def is_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"


def setup(device: torch.device) -> None:
    if is_cuda(device):
        torch.cuda.set_device(device)


def sync(device: torch.device) -> None:
    """Device-wide synchronise: waits for work on *every* stream, not just the current one."""
    if is_cuda(device):
        torch.cuda.synchronize(device)


def empty_cache(device: torch.device) -> None:
    if is_cuda(device):
        torch.cuda.empty_cache()


def reset_peak_memory(device: torch.device) -> None:
    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**30 if is_cuda(device) else None


class Timer:
    """CUDA-event pair on the current stream (or a host clock in CPU debug mode)."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = is_cuda(device)
        if self.cuda:
            self._s = torch.cuda.Event(enable_timing=True)
            self._e = torch.cuda.Event(enable_timing=True)
        self._t0 = self._t1 = 0.0

    def start(self) -> None:
        if self.cuda:
            self._s.record()
        else:
            self._t0 = _perf_counter()

    def stop(self) -> None:
        if self.cuda:
            self._e.record()
        else:
            self._t1 = _perf_counter()

    def elapsed_ms(self) -> float:
        return self._s.elapsed_time(self._e) if self.cuda else (self._t1 - self._t0) * 1e3
