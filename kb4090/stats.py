"""Small, dependency-light statistics helpers. No PyTorch here."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def summarise(xs: Sequence[float]) -> dict:
    a = np.asarray(list(xs), dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.percentile(a, [10, 25, 50, 75, 90])
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {
        "n": int(a.size),
        "median": float(q50),
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else float("nan"),
        "p10": float(q10),
        "p25": float(q25),
        "p75": float(q75),
        "p90": float(q90),
        "iqr": float(q75 - q25),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def bootstrap_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Median(numerator) / median(denominator) with a percentile bootstrap CI.

    For speedups pass baseline times as the numerator and candidate times as the denominator.
    """
    num = np.asarray(list(numerator), dtype=np.float64)
    den = np.asarray(list(denominator), dtype=np.float64)
    if num.size == 0 or den.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    point = float(np.median(num) / np.median(den))
    idx_n = rng.integers(0, num.size, size=(n_boot, num.size))
    idx_d = rng.integers(0, den.size, size=(n_boot, den.size))
    ratios = np.median(num[idx_n], axis=1) / np.median(den[idx_d], axis=1)
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def geomean(xs: Sequence[float]) -> float:
    vals = list(xs)
    if not vals:
        return float("nan")
    if any(x <= 0 for x in vals):
        raise ValueError("geomean needs strictly positive values")
    return float(math.exp(sum(math.log(x) for x in vals) / len(vals)))


def min_detectable_speedup(aa_ratio_ci: dict) -> float:
    """From an A/A test (same code timed twice), the smallest speedup you can tell apart from noise."""
    lo, hi = aa_ratio_ci["lo"], aa_ratio_ci["hi"]
    return float(max(1.0 / lo if lo > 0 else float("inf"), hi))
