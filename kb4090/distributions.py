"""Hidden input distributions.

KernelBench tasks build their inputs in ``get_inputs()`` with ``torch.rand``/``torch.randn``. Some
tasks then impose structure (softmax, triu, symmetric matrices...). To change the *values* while
keeping that structure, we patch the random generators while ``get_inputs()`` runs and transform
each raw draw before the task code sees it:

  base   unchanged
  x3     values x 3       (catches kernels that assume a value range)
  x0p01  values x 0.01    (catches kernels that are only accurate at one scale)
  neg    values x -1      (catches sign-specialised kernels, e.g. a "ReLU" that returns its input)

This follows the four-distribution idea of KernelBench-Verified (facebookresearch/kernel_bench_verified).
A distribution is only *applicable* when the reference output stays finite; e.g. ``neg`` makes
``log`` tasks produce NaN, so it is skipped for them and recorded as not applicable.

Integer generators (``torch.randint``) and constants (``torch.ones``, ``torch.arange``) are untouched.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

SCALES: dict[str, float | None] = {"base": None, "x3": 3.0, "x0p01": 0.01, "neg": -1.0}
_PATCHED = ("rand", "randn", "rand_like", "randn_like")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)  # seeds CPU and every CUDA device


@contextmanager
def patched_rng(dist: str, device: str | torch.device | None) -> Iterator[None]:
    """Transform every float tensor produced by torch.rand/randn(_like) inside the block."""
    if dist not in SCALES:
        raise KeyError(f"unknown distribution {dist!r}; choose from {sorted(SCALES)}")
    scale = SCALES[dist]
    originals = {name: getattr(torch, name) for name in _PATCHED}

    def make(name: str, orig: Any) -> Any:
        is_like = name.endswith("_like")

        def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
            if device is not None and not is_like and kwargs.get("device") is None and kwargs.get("out") is None:
                kwargs["device"] = device
            t = orig(*args, **kwargs)
            if scale is not None and t.is_floating_point():
                with torch.no_grad():
                    t.mul_(scale)
            return t

        wrapper.__wrapped__ = orig  # type: ignore[attr-defined]
        return wrapper

    try:
        for name, orig in originals.items():
            setattr(torch, name, make(name, orig))
        yield
    finally:
        for name, orig in originals.items():
            setattr(torch, name, orig)


def _to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list | tuple):
        return type(x)(_to_device(v, device) for v in x)
    return x


def generate_inputs(task_module: Any, seed: int, dist: str, device: torch.device) -> tuple[list, dict]:
    """Call the task's get_inputs() under a seed and distribution, and move the result to ``device``.

    We first try to create tensors directly on the GPU (fast, no host copy of multi-GB tensors).
    If the task mixes devices inside get_inputs() and that fails, we fall back to the CPU.
    """
    meta = {"seed": seed, "dist": dist, "gen_device": str(device)}
    try:
        seed_all(seed)
        with patched_rng(dist, device):
            inputs = list(task_module.get_inputs())
    except RuntimeError:
        meta["gen_device"] = "cpu"
        seed_all(seed)
        with patched_rng(dist, None):
            inputs = list(task_module.get_inputs())
    return [_to_device(x, device) for x in inputs], meta


def perturb_inplace(inputs: list, step: int) -> None:
    """Change input *values* between timing iterations without allocating.

    Alternating x1.25 / x0.8 keeps the scale stable over many iterations and preserves structure
    (sign, zeros, triangularity, symmetry). A kernel that caches its output by pointer or shape
    then returns stale results, which the timing spot-checks catch.
    """
    factor = 1.25 if step % 2 == 0 else 0.8
    with torch.no_grad():
        for x in inputs:
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                x.mul_(factor)
