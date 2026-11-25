"""Count the work a task requires, for the speed-of-light check (see roofline.py).

FLOPs come from torch.utils.flop_counter (matmul, conv, attention; element-wise ops count as 0,
which only makes the bound looser, never wrong). Bytes are the compulsory DRAM traffic under a
cold L2: every input, parameter and buffer read once, every output written once.

That byte count can be too high for tasks whose inputs have structure a kernel may legitimately
exploit (e.g. triangular matrices, where half the input is known zeros). For those, set
``sol_margin`` lower in configs/tasks.yaml. A SOL violation only marks a result SUSPECT (for human
review), never INVALID.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def count_flops(fn: Callable[[], Any]) -> int | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as counter:
            fn()
        return int(counter.get_total_flops())
    except Exception:
        return None


def tensor_bytes(obj: Any) -> int:
    seen: set[int] = set()
    total = 0

    def walk(x: Any) -> None:
        nonlocal total
        if isinstance(x, torch.Tensor):
            key = x.untyped_storage().data_ptr() if x.device.type != "meta" else id(x)
            if key not in seen:
                seen.add(key)
                total += x.numel() * x.element_size()
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(obj)
    return total


def model_state_bytes(model: torch.nn.Module) -> int:
    return tensor_bytes(list(model.parameters()) + list(model.buffers()))
