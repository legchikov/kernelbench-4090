"""Speed-of-light (roofline) arithmetic. Pure Python so it can be unit-tested without a GPU.

A kernel can never finish faster than the larger of:
  * compute time  = FLOPs / peak FLOP rate
  * memory time   = bytes that must cross DRAM / peak DRAM bandwidth
If a measured time beats this bound, the measurement or the kernel is wrong.
"""

from __future__ import annotations


def speed_of_light(
    flops: float | None,
    bytes_moved: float,
    peak_tflops: float,
    bandwidth_gbs: float,
) -> dict:
    compute_ms = (flops or 0.0) / (peak_tflops * 1e12) * 1e3 if peak_tflops else 0.0
    memory_ms = bytes_moved / (bandwidth_gbs * 1e9) * 1e3 if bandwidth_gbs else 0.0
    bound = "compute" if compute_ms >= memory_ms else "memory"
    intensity = (flops or 0.0) / bytes_moved if bytes_moved else float("inf")
    ridge = (peak_tflops * 1e12) / (bandwidth_gbs * 1e9) if bandwidth_gbs else float("inf")
    return {
        "sol_ms": max(compute_ms, memory_ms),
        "compute_ms": compute_ms,
        "memory_ms": memory_ms,
        "bound": bound,
        "arithmetic_intensity": intensity,
        "ridge_point": ridge,
        "flops": flops,
        "bytes": bytes_moved,
        "peak_tflops": peak_tflops,
        "bandwidth_gbs": bandwidth_gbs,
    }


def peak_for(hw: dict, key: str) -> float:
    peaks = hw.get("peaks_tflops", {})
    value = peaks.get(key)
    if value is None:
        raise KeyError(f"hardware file {hw.get('_source')} has no peaks_tflops.{key}")
    return float(value)
