"""Output comparison and input fingerprints, written to stay within 24 GB.

Some KernelBench tasks have multi-GB tensors (level1/19_ReLU is 4096 x 393216 fp32, 6.4 GB per
tensor), so we never materialise ``(cand - ref)`` in one go: comparisons run in chunks and the
reference output lives on the host while the candidate runs.

A candidate passes a comparison when, for every output tensor:
  * it is a plain torch.Tensor (not a lazy subclass) with the reference's shape and dtype,
  * it is finite wherever the reference is finite,
  * every element satisfies |c - r| <= atol + rtol * |r|   (KernelBench's allclose rule), and
  * the relative L2 error ||c - r|| / ||r|| <= rel_l2_max   (stops tiny outputs, e.g. inputs x0.01,
    from passing trivially under atol=1e-2).
"""

from __future__ import annotations

import math
from typing import Any

import torch

CHUNK_ELEMS = 1 << 25  # 32M elements per chunk (~1.5 GB of float64 temporaries)


def flatten(obj: Any) -> list:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, list | tuple):
        out: list = []
        for v in obj:
            out.extend(flatten(v))
        return out
    if isinstance(obj, dict):
        out = []
        for k in sorted(obj):
            out.extend(flatten(obj[k]))
        return out
    return [obj]


def is_plain_tensor(t: Any) -> bool:
    return type(t) in (torch.Tensor, torch.nn.Parameter)


def all_finite(obj: Any) -> bool:
    for t in flatten(obj):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            for chunk in _chunks(t):
                if not bool(torch.isfinite(chunk).all()):
                    return False
    return True


def to_host(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, list | tuple):
        return type(obj)(to_host(v) for v in obj)
    if isinstance(obj, dict):
        return {k: to_host(v) for k, v in obj.items()}
    return obj


def nbytes_list(obj: Any) -> list[int]:
    return [t.numel() * t.element_size() for t in flatten(obj) if isinstance(t, torch.Tensor) and t.is_cuda]


def poison_allocator(sizes: list[int], device: torch.device) -> None:
    """Fill freed allocator blocks with 0xFF bytes (NaN for fp32/fp16/bf16).

    A candidate that returns ``torch.empty_like(...)`` would otherwise be handed the block that
    just held the reference output, and "pass" without computing anything. KernelBench's
    result_reuse_kernel.py adversarial test exercises exactly this.
    """
    if device.type != "cuda":
        return
    blocks = []
    try:
        for size in sizes:
            b = torch.empty(size, dtype=torch.uint8, device=device)
            b.fill_(0xFF)
            blocks.append(b)
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        pass
    finally:
        del blocks


def _chunks(t: torch.Tensor, n: int = CHUNK_ELEMS):
    flat = t.reshape(-1)
    for i in range(0, flat.numel(), n):
        yield flat[i : i + n]


def fingerprint(t: Any) -> tuple:
    """Cheap, deterministic fingerprint of an input, to detect in-place mutation."""
    if not isinstance(t, torch.Tensor):
        return ("obj", repr(t)[:200])
    total = 0.0
    abs_total = 0.0
    for chunk in _chunks(t.detach()):
        c = chunk.to(torch.float64) if (chunk.is_floating_point() or chunk.dtype == torch.bool) else chunk.to(torch.int64)
        total += float(c.sum())
        abs_total += float(c.abs().sum())
    flat = t.detach().reshape(-1)
    step = max(1, flat.numel() // 4096)
    sample = flat[::step][:4096].to("cpu").clone()
    return (tuple(t.shape), str(t.dtype), total, abs_total, sample)


def same_fingerprint(a: tuple, b: tuple) -> bool:
    if a[0] == "obj" or b[0] == "obj":
        return a == b
    if a[:2] != b[:2]:
        return False
    if not (_same_float(a[2], b[2]) and _same_float(a[3], b[3])):
        return False
    return bool(torch.equal(a[4], b[4]))


def _same_float(x: float, y: float) -> bool:
    return (math.isnan(x) and math.isnan(y)) or x == y


def compare(ref: Any, cand: Any, atol: float, rtol: float, rel_l2_max: float) -> dict:
    ref_flat, cand_flat = flatten(ref), flatten(cand)
    if len(ref_flat) != len(cand_flat):
        return {"ok": False, "reason": f"output structure differs: {len(ref_flat)} vs {len(cand_flat)} items"}
    per_output = []
    for i, (r, c) in enumerate(zip(ref_flat, cand_flat, strict=True)):
        if not isinstance(r, torch.Tensor):
            ok = r == c
            per_output.append({"index": i, "ok": bool(ok), "reason": None if ok else "non-tensor output differs"})
            continue
        if not isinstance(c, torch.Tensor):
            per_output.append({"index": i, "ok": False, "reason": f"expected a tensor, got {type(c).__name__}"})
            continue
        if not is_plain_tensor(c):
            per_output.append({"index": i, "ok": False, "reason": f"output is a tensor subclass ({type(c).__name__})"})
            continue
        if tuple(r.shape) != tuple(c.shape):
            per_output.append({"index": i, "ok": False, "reason": f"shape {tuple(c.shape)} != {tuple(r.shape)}"})
            continue
        if r.dtype != c.dtype:
            per_output.append({"index": i, "ok": False, "reason": f"dtype {c.dtype} != {r.dtype}"})
            continue
        per_output.append({"index": i, **_tensor_stats(r, c, atol, rtol, rel_l2_max)})
    ok = all(p["ok"] for p in per_output)
    worst = max((p.get("max_abs", 0.0) or 0.0 for p in per_output), default=0.0)
    rel = max((p.get("rel_l2", 0.0) or 0.0 for p in per_output), default=0.0)
    reason = next((p["reason"] for p in per_output if not p["ok"]), None)
    return {"ok": ok, "reason": reason, "max_abs": worst, "rel_l2": rel, "outputs": per_output}


def _tensor_stats(r: torch.Tensor, c: torch.Tensor, atol: float, rtol: float, rel_l2_max: float) -> dict:
    device = c.device
    exact = not (r.is_floating_point() or r.is_complex())
    mismatches = 0
    nonfinite = 0
    max_abs = 0.0
    sq_err = 0.0
    sq_ref = 0.0
    for rc, cc in zip(_chunks(r), _chunks(c), strict=True):
        rc = rc.to(device)
        if exact:
            mismatches += int((rc != cc).sum())
            continue
        r64, c64 = rc.to(torch.float64), cc.to(torch.float64)
        r_fin = torch.isfinite(r64)
        c_bad = (~torch.isfinite(c64)) & r_fin
        nonfinite += int(c_bad.sum())
        diff = (c64 - r64).abs()
        diff = torch.where(r_fin, diff, torch.zeros_like(diff))
        tol = atol + rtol * r64.abs()
        mismatches += int(((diff > tol) & r_fin).sum()) + int(c_bad.sum())
        finite_diff = diff[torch.isfinite(diff)]
        if finite_diff.numel():
            max_abs = max(max_abs, float(finite_diff.max()))
        sq_err += float((torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff)) ** 2).sum())
        sq_ref += float((torch.where(r_fin, r64, torch.zeros_like(r64)) ** 2).sum())
    rel_l2 = math.sqrt(sq_err) / math.sqrt(sq_ref) if sq_ref > 0 else (0.0 if sq_err == 0 else float("inf"))
    ok = mismatches == 0 and (exact or sq_ref == 0 or rel_l2 <= rel_l2_max)
    reason = None
    if nonfinite:
        reason = f"{nonfinite} non-finite values where the reference is finite"
    elif mismatches:
        reason = f"{mismatches} elements outside atol={atol}, rtol={rtol}"
    elif not ok:
        reason = f"relative L2 error {rel_l2:.3g} > {rel_l2_max}"
    return {
        "ok": ok,
        "reason": reason,
        "mismatches": mismatches,
        "nonfinite": nonfinite,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "numel": r.numel(),
    }
