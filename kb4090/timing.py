"""Interleaved, cold-cache timing with a stream guard.

For every timed iteration:
  1. ``pre`` hook (set numeric flags, seed), then perturb the shared inputs in place
  2. flush L2 (write 256 MB)
  3. synchronise the whole device
  4. record a CUDA event, run the forward, record an event, synchronise the whole device again

We keep two clocks per iteration:
  * the CUDA-event time on the current stream (the standard KernelBench metric), and
  * host wall time around a device-wide synchronise, which also covers work on other streams.
If the host time, net of the measured fixed overhead, is far above the event time, the candidate is
running work off the timed stream (KernelBench's non_default_stream_kernel.py), and the stream guard
fires.

Contenders (baselines and the candidate) take turns in rotating order within each round, so drift
from temperature or clocks hits all of them alike (the "interleaved controls" from the Sybil
inference-optimisation method).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .device import Timer, sync
from .stats import summarise

_perf_counter = time.perf_counter  # captured before any candidate code is imported


@dataclass
class Contender:
    name: str
    fn: Callable[[], Any]
    pre: Callable[[], None] | None = None
    post: Callable[[], None] | None = None
    spot_check: Callable[[Any], dict] | None = None
    event_ms: list[float] = field(default_factory=list)
    host_ms: list[float] = field(default_factory=list)
    spot_results: list[dict] = field(default_factory=list)


def measure_host_overhead(device: torch.device, n: int = 50) -> float:
    """Median (host - event) time for a trivial kernel: the fixed cost of our timing loop."""
    x = torch.zeros(1, device=device)
    gaps = []
    for _ in range(n + 5):
        sync(device)
        timer = Timer(device)
        t0 = _perf_counter()
        timer.start()
        x.add_(1.0)
        timer.stop()
        sync(device)
        t1 = _perf_counter()
        gaps.append((t1 - t0) * 1e3 - timer.elapsed_ms())
    gaps = sorted(gaps[5:])
    return gaps[len(gaps) // 2]


def _one_iteration(c: Contender, device: torch.device, flush: Callable[[], None] | None,
                   perturb: Callable[[], None] | None) -> Any:
    if c.pre:
        c.pre()
    if perturb:
        perturb()
    if flush:
        flush()
    sync(device)
    timer = Timer(device)
    t0 = _perf_counter()
    timer.start()
    out = c.fn()
    timer.stop()
    sync(device)
    t1 = _perf_counter()
    if c.post:
        c.post()
    c.event_ms.append(timer.elapsed_ms())
    c.host_ms.append((t1 - t0) * 1e3)
    return out


def warmup(c: Contender, device: torch.device, iters: int, min_ms: float, max_s: float) -> None:
    start = _perf_counter()
    done = 0
    while True:
        if c.pre:
            c.pre()
        out = c.fn()
        sync(device)
        del out
        if c.post:
            c.post()
        done += 1
        elapsed = _perf_counter() - start
        if (done >= iters and elapsed * 1e3 >= min_ms) or elapsed > max_s:
            return


def run_interleaved(
    contenders: list[Contender],
    device: torch.device,
    rounds: int,
    iters_per_round: int,
    flush: Callable[[], None] | None,
    perturb_step: Callable[[int], None] | None,
    spot_checks: int,
    warmup_iters: int,
    min_warmup_ms: float,
    max_warmup_s: float,
) -> dict:
    for c in contenders:
        warmup(c, device, warmup_iters, min_warmup_ms, max_warmup_s)

    # Spread spot checks over the run: (round, iteration) positions for contenders that have one.
    positions = set()
    if spot_checks > 0 and rounds > 0:
        for k in range(spot_checks):
            r = min(rounds - 1, (k * rounds) // spot_checks)
            positions.add((r, iters_per_round - 1 - (k % max(1, iters_per_round))))

    step = 0
    n = len(contenders)
    for r in range(rounds):
        order = contenders[r % n:] + contenders[: r % n]
        for c in order:
            for i in range(iters_per_round):
                perturb = (lambda s=step: perturb_step(s)) if perturb_step else None
                step += 1
                out = _one_iteration(c, device, flush, perturb)
                if c.spot_check is not None and (r, i) in positions:
                    try:
                        c.spot_results.append({"round": r, "iter": i, **c.spot_check(out)})
                    except torch.cuda.OutOfMemoryError:
                        c.spot_results.append({"round": r, "iter": i, "ok": None, "reason": "skipped: out of memory"})
                del out
    return {c.name: {"event_ms": c.event_ms, "host_ms": c.host_ms, "spot_checks": c.spot_results}
            for c in contenders}


def stream_guard(event_ms: list[float], host_ms: list[float], overhead_ms: float,
                 ratio: float, slack_ms: float) -> dict:
    ev = summarise(event_ms)["median"]
    host = summarise(host_ms)["median"]
    net = max(0.0, host - overhead_ms)
    fired = net > ratio * ev + slack_ms
    return {"event_median_ms": ev, "host_median_ms": host, "host_net_ms": net,
            "overhead_ms": overhead_ms, "fired": bool(fired)}
