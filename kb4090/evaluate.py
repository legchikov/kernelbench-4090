"""Evaluation pipeline. Runs inside a fresh subprocess per candidate (see runner.py / worker.py),
so a crashing kernel or a sticky CUDA error can't poison the next evaluation.

Stages (fail fast, most severe verdict wins):
  1. static checks                      -> INVALID on any error
  2. import candidate (compiles)        -> FAIL_COMPILE
  3. construct models, sync weights     -> FAIL_RUNTIME
  4. correctness: seeds x distributions -> FAIL_INCORRECT / FAIL_RUNTIME / INVALID (input mutation,
                                           threads, harness tamper)
  5. kernel attribution (profiler)      -> INVALID if no custom kernel ran
  6. timing under the GPU lock          -> INVALID if stream guard or spot checks fire
  7. speed of light                     -> SUSPECT if faster than physically possible

Verdicts: PASS, SUSPECT (needs a human look; excluded from headline numbers until reviewed),
FAIL_COMPILE, FAIL_RUNTIME, FAIL_INCORRECT, FAIL_TIMEOUT, INVALID (cheating detected),
HARNESS_ERROR (our fault or the task's; excluded from statistics).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
import traceback
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from . import compare as cmp
from .device import empty_cache, get_device, peak_memory_gb, reset_peak_memory, setup, sync
from .distributions import generate_inputs, perturb_inplace, seed_all
from .gpu import GpuLock, L2Flusher, other_compute_pids, telemetry, wait_for_idle
from .roofline import peak_for, speed_of_light
from .runtime_checks import HarnessIntegrity, classify_kernels, prewarm_harness, profile_kernels, thread_count
from .sol import count_flops, model_state_bytes, tensor_bytes
from .stats import bootstrap_ratio_ci, summarise
from .timing import Contender, measure_host_overhead, run_interleaved, stream_guard

SEVERITY = ["INVALID", "FAIL_COMPILE", "FAIL_RUNTIME", "FAIL_INCORRECT", "FAIL_TIMEOUT",
            "HARNESS_ERROR", "SUSPECT", "PASS"]


class Verdict:
    def __init__(self) -> None:
        self.reasons: list[dict] = []
        self.warnings: list[str] = []

    def add(self, verdict: str, reason: str) -> None:
        assert verdict in SEVERITY, verdict
        self.reasons.append({"verdict": verdict, "reason": reason})

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def failed(self) -> bool:
        return any(r["verdict"] != "SUSPECT" for r in self.reasons)

    def final(self) -> str:
        if not self.reasons:
            return "PASS"
        return min((r["verdict"] for r in self.reasons), key=SEVERITY.index)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path: str | Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_strict_flags() -> None:
    """Ground truth is strict FP32: no TF32 anywhere, deterministic cuDNN algorithm choice off."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False


@contextmanager
def tf32(enabled: bool):
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    p = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = m
        torch.backends.cudnn.allow_tf32 = c
        torch.set_float32_matmul_precision(p)


def _tail(text: str, n: int = 4000) -> str:
    return text if len(text) <= n else "...\n" + text[-n:]


def _sync_weights(ref: torch.nn.Module, cand: torch.nn.Module) -> bool:
    """Copy reference weights into the candidate when parameter names and shapes match exactly.

    KernelBench relies on identical seeding and init order; copying removes that fragility
    for candidates that keep the reference's parameter layout.
    """
    rs, cs = ref.state_dict(), cand.state_dict()
    if not rs or set(rs) != set(cs) or any(rs[k].shape != cs[k].shape for k in rs):
        return False
    cand.load_state_dict(rs)
    return True


def _record_exception(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)[:2000], "traceback": _tail(traceback.format_exc())}


def _forward(model: torch.nn.Module, inputs: list, fwd_seed: int) -> Any:
    seed_all(fwd_seed)  # tasks like level3/35_LSTM draw random states inside forward()
    with torch.no_grad():
        return model(*inputs)


# ---------------------------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------------------------
def evaluate_candidate(task_path: str, candidate_path: str, rig: dict, hw: dict, task_meta: dict,
                       mode: str = "official") -> dict:
    from .static_check import check_source

    started = time.time()
    v = Verdict()
    res: dict[str, Any] = {
        "kind": "eval", "mode": mode, "task_id": task_meta.get("id"), "task_path": str(task_path),
        "task_sha256": sha256_file(task_path), "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path), "started": started,
    }
    device = get_device()
    setup(device)
    set_strict_flags()
    policy = rig["policy"]
    corr_cfg = rig["correctness"] if mode == "official" else {**rig["correctness"], **rig["dev"]}

    def finish() -> dict:
        res["verdict"] = v.final()
        res["reasons"] = v.reasons
        res["warnings"] = v.warnings
        res["elapsed_s"] = time.time() - started
        return res

    # 1. static ---------------------------------------------------------------------------------
    st = check_source(Path(candidate_path).read_text(encoding="utf-8"), policy)
    res["static"] = st
    for w in st["warnings"]:
        v.warn(f"static:{w['rule']}: {w['message']}")
    if st["errors"]:
        v.add("INVALID", "static: " + "; ".join(f"{e['rule']} ({e['message']})" for e in st["errors"]))
        return finish()

    # 2. load reference, then import (compile) the candidate --------------------------------------
    ref_mod = load_module(task_path, "kb4090_task_ref")
    init_inputs = list(ref_mod.get_init_inputs())
    prewarm_harness(device)
    integrity = HarnessIntegrity()
    threads_before_import = thread_count()
    t0 = time.perf_counter()
    try:
        cand_mod = load_module(candidate_path, "kb4090_candidate")
    except Exception as exc:
        res["compile_error"] = _record_exception(exc)
        v.add("FAIL_COMPILE", f"{type(exc).__name__}: {str(exc)[:300]}")
        return finish()
    res["compile_s"] = time.perf_counter() - t0
    tampered = integrity.verify()
    if tampered:
        v.add("INVALID", f"harness_tamper at import: {tampered}")
        return finish()
    if thread_count() > threads_before_import:
        v.add("INVALID", "candidate started threads at import")
        return finish()
    if not hasattr(cand_mod, "ModelNew"):
        v.add("INVALID", "candidate defines no ModelNew")
        return finish()

    # 3. construct ------------------------------------------------------------------------------
    base_seed = int(rig.get("seed", 42))
    seed_all(base_seed)
    ref_model = ref_mod.Model(*init_inputs).to(device)
    try:
        seed_all(base_seed)
        cand_model = cand_mod.ModelNew(*init_inputs).to(device)
    except Exception as exc:
        res["runtime_error"] = _record_exception(exc)
        v.add("FAIL_RUNTIME", f"constructing ModelNew: {type(exc).__name__}: {str(exc)[:300]}")
        return finish()
    res["weights_synced"] = _sync_weights(ref_model, cand_model)

    # First call: JIT compilation (Triton) and library initialisation happen here, not in the trials.
    import threading

    names_before = {t.name for t in threading.enumerate()}
    try:
        warm_inputs, _ = generate_inputs(ref_mod, base_seed + 3, "base", device)
        _ = _forward(cand_model, warm_inputs, base_seed + 4)
        sync(device)
        del _, warm_inputs
    except Exception as exc:
        res["runtime_error"] = _record_exception(exc)
        v.add("FAIL_RUNTIME", f"first ModelNew.forward raised {type(exc).__name__}: {str(exc)[:300]}")
        return finish()
    new_threads = sorted({t.name for t in threading.enumerate()} - names_before)
    if new_threads:
        v.warn(f"threads appeared during the first forward (compilation?): {new_threads}")

    # 4. correctness ----------------------------------------------------------------------------
    corr = _correctness(ref_mod, ref_model, cand_model, corr_cfg, base_seed, device, v, integrity)
    res["correctness"] = corr if mode == "official" else {k: corr[k] for k in ("passed", "trials", "summary")}
    if v.failed:
        return finish()

    # 5. kernel attribution ---------------------------------------------------------------------
    inputs, _ = generate_inputs(ref_mod, base_seed + 7, "base", device)
    fwd_seed = base_seed + 11
    ref_prof = profile_kernels(lambda: _forward(ref_model, inputs, fwd_seed))
    cand_prof = profile_kernels(lambda: _forward(cand_model, inputs, fwd_seed))
    attribution = classify_kernels(cand_prof, ref_prof, st.get("declared_kernels"))
    res["kernels"] = attribution
    if not attribution["profiler_available"]:
        v.warn("profiler unavailable: could not verify that a custom kernel ran")
    else:
        if policy["require_custom_kernel"] and attribution["n_custom"] == 0:
            v.add("INVALID", "no custom kernel ran (only library/ATen kernels were launched)")
        elif attribution["custom_fraction"] < float(policy["min_custom_gpu_fraction"]):
            v.add("INVALID", f"custom kernels are only {attribution['custom_fraction']:.1%} of GPU time")
        elif attribution["custom_fraction"] < float(policy["warn_custom_gpu_fraction"]):
            v.warn(f"custom kernels are only {attribution['custom_fraction']:.1%} of GPU time")
    if integrity.verify():
        v.add("INVALID", f"harness_tamper during forward: {integrity.verify()}")
    if v.failed:
        return finish()

    # 6. timing ---------------------------------------------------------------------------------
    timing_cfg = rig["timing"] if mode == "official" else {**rig["timing"], **{
        k: rig["dev"][k] for k in ("rounds", "iters_per_round") if k in rig["dev"]}}
    tres = _time_against_baselines(ref_model, cand_model, inputs, fwd_seed, rig, timing_cfg, task_meta,
                                   device, corr_cfg, with_candidate=True)
    res["timing"] = tres
    cand_t = tres["contenders"]["candidate"]
    if tres.get("contended"):
        v.warn(f"other processes used the GPU during timing: {tres.get('others')}")
    sg = cand_t["stream_guard"]
    if timing_cfg["stream_guard"]["enabled"] and sg["fired"]:
        v.add("INVALID", f"stream guard: host time {sg['host_net_ms']:.3f} ms vs event time "
                         f"{sg['event_median_ms']:.3f} ms (work running off the timed stream?)")
    bad_spots = [s for s in cand_t["spot_checks"] if s.get("ok") is False]
    if bad_spots:
        v.add("INVALID", f"output during timing did not match the reference ({bad_spots[0].get('reason')}); "
                         "the kernel may be caching or skipping work")

    # 7. speed of light -------------------------------------------------------------------------
    sol = _speed_of_light(ref_model, inputs, fwd_seed, hw)
    res["sol"] = sol
    margin = float(task_meta.get("sol_margin") or policy["sol_margin"])
    base_medians = {k: c["summary"]["median"] for k, c in tres["contenders"].items() if k != "candidate"}
    if sol.get("guard"):
        guard_ms = sol["guard"]["sol_ms"] * margin
        if any(m < guard_ms for m in base_medians.values()):
            sol["unreliable"] = True
            v.warn("a PyTorch baseline beat the speed-of-light bound: the SOL model is wrong for this task")
        elif cand_t["summary"]["median"] < guard_ms:
            v.add("SUSPECT", f"candidate median {cand_t['summary']['median']:.4f} ms is below the "
                             f"speed-of-light bound {sol['guard']['sol_ms']:.4f} ms x {margin}")
        if sol.get("efficiency"):
            res["pct_of_sol"] = sol["efficiency"]["sol_ms"] / cand_t["summary"]["median"]

    # Headline numbers --------------------------------------------------------------------------
    res["speedups"] = tres["speedups"]
    res["headline_baseline"] = rig["headline_baseline"]
    head = tres["speedups"].get(rig["headline_baseline"])
    res["headline_speedup"] = head["point"] if head else None
    res["telemetry"] = tres.get("telemetry")
    if any("thermal" in r or "power" in r for r in (tres.get("telemetry", {}).get("after", {}) or {})
           .get("throttle_reasons", [])):
        v.warn("GPU reported thermal/power throttling during timing")
    return finish()


def _correctness(ref_mod: Any, ref_model: torch.nn.Module, cand_model: torch.nn.Module, cfg: dict,
                 base_seed: int, device: torch.device, v: Verdict, integrity: HarnessIntegrity) -> dict:
    trials = []
    for k in range(int(cfg["seeds"])):
        seed = base_seed + 1000 * (k + 1)
        for dist in cfg["distributions"]:
            trial: dict[str, Any] = {"seed": seed, "dist": dist}
            inputs, meta = generate_inputs(ref_mod, seed, dist, device)
            trial["gen_device"] = meta["gen_device"]
            ref_out = _forward(ref_model, inputs, seed + 1)
            sync(device)
            if not cmp.all_finite(ref_out):
                trial.update(applicable=False, ok=None, reason="reference output not finite")
                trials.append(trial)
                del ref_out, inputs
                continue
            ref_host = cmp.to_host(ref_out)
            sizes = cmp.nbytes_list(ref_out)
            del ref_out
            cmp.poison_allocator(sizes, device)
            before = [cmp.fingerprint(x) for x in inputs]
            threads_before = thread_count()
            try:
                cand_out = _forward(cand_model, inputs, seed + 1)
                sync(device)
            except torch.cuda.OutOfMemoryError as exc:
                trial.update(applicable=True, ok=False, reason="out of memory", error=_record_exception(exc))
                trials.append(trial)
                v.add("FAIL_RUNTIME", f"out of memory in ModelNew.forward (seed {seed}, {dist})")
                break
            except Exception as exc:
                trial.update(applicable=True, ok=False, reason="exception", error=_record_exception(exc))
                trials.append(trial)
                v.add("FAIL_RUNTIME", f"ModelNew.forward raised {type(exc).__name__}: {str(exc)[:300]}")
                break
            if thread_count() > threads_before:
                v.add("INVALID", "ModelNew.forward started threads")
            after = [cmp.fingerprint(x) for x in inputs]
            if not all(cmp.same_fingerprint(a, b) for a, b in zip(before, after, strict=True)):
                v.add("INVALID", "ModelNew.forward modified its inputs in place")
            tampered = integrity.verify()
            if tampered:
                v.add("INVALID", f"harness_tamper during forward: {tampered}")
            c = cmp.compare(ref_host, cand_out, float(cfg["atol"]), float(cfg["rtol"]), float(cfg["rel_l2_max"]))
            trial.update(applicable=True, ok=c["ok"], reason=c["reason"], max_abs=c["max_abs"], rel_l2=c["rel_l2"])
            trials.append(trial)
            del cand_out, ref_host, inputs
            empty_cache(device)
            if v.failed:
                break
        if v.failed:
            break

    applicable = [t for t in trials if t.get("applicable")]
    failed = [t for t in applicable if t.get("ok") is False]
    by_dist: dict[str, dict] = {}
    for t in trials:
        d = by_dist.setdefault(t["dist"], {"applicable": 0, "passed": 0, "not_applicable": 0})
        if t.get("applicable"):
            d["applicable"] += 1
            d["passed"] += int(bool(t.get("ok")))
        else:
            d["not_applicable"] += 1
    if not any(t["dist"] == "base" and t.get("applicable") for t in trials) and not v.failed:
        v.add("HARNESS_ERROR", "reference output is not finite on the base distribution")
    if failed and not v.failed:
        hidden_only = all(t["dist"] != "base" for t in failed)
        dists = sorted({t["dist"] for t in failed})
        msg = f"wrong output on {len(failed)}/{len(applicable)} trials ({', '.join(dists)})"
        if hidden_only:
            msg += "; passes the standard inputs but fails transformed ones (specialised to the test inputs?)"
        v.add("FAIL_INCORRECT", msg)
    return {
        "passed": bool(applicable) and not failed,
        "hidden_only_failure": bool(failed) and all(t["dist"] != "base" for t in failed),
        "summary": by_dist,
        "trials": trials,
    }


def _time_against_baselines(ref_model: torch.nn.Module, cand_model: torch.nn.Module | None, inputs: list,
                            fwd_seed: int, rig: dict, tcfg: dict, task_meta: dict, device: torch.device,
                            corr_cfg: dict, with_candidate: bool, extra_contenders: list | None = None) -> dict:
    gpu_index = int(rig["gpu_index"])
    contenders: list[Contender] = []

    def seeded(flag: bool):
        def pre() -> None:
            torch.set_float32_matmul_precision("high" if flag else "highest")  # stable API
            torch.backends.cuda.matmul.allow_tf32 = flag                        # legacy flags, kept in sync
            torch.backends.cudnn.allow_tf32 = flag
            seed_all(fwd_seed)

        return pre

    def restore() -> None:
        set_strict_flags()

    def fwd(model: torch.nn.Module):
        def run() -> Any:
            with torch.no_grad():
                return model(*inputs)

        return run

    for name in rig["baselines"]:
        if name == "eager_fp32":
            contenders.append(Contender(name, fwd(ref_model), pre=seeded(False), post=restore))
        elif name == "eager_tf32":
            contenders.append(Contender(name, fwd(ref_model), pre=seeded(True), post=restore))
        elif name == "compile":
            compiled = torch.compile(ref_model)
            contenders.append(Contender(name, fwd(compiled), pre=seeded(False), post=restore))
        else:
            raise ValueError(f"unknown baseline {name!r}")
    contenders.extend(extra_contenders or [])

    if with_candidate and cand_model is not None:
        def spot(out: Any) -> dict:
            ref_out = _forward(ref_model, inputs, fwd_seed)
            c = cmp.compare(ref_out, out, float(corr_cfg["atol"]), float(corr_cfg["rtol"]),
                            float(corr_cfg["rel_l2_max"]))
            del ref_out
            return {"ok": c["ok"], "reason": c["reason"], "max_abs": c["max_abs"]}

        contenders.append(Contender("candidate", fwd(cand_model),
                                    pre=seeded(bool(rig["policy"]["candidate_tf32"])), post=restore,
                                    spot_check=spot if tcfg.get("spot_checks", 0) else None))

    perturb = None
    if tcfg.get("perturb_inputs", True) and task_meta.get("perturb", True):
        def perturb(step: int) -> None:
            perturb_inplace(inputs, step)

    out: dict[str, Any] = {}
    with GpuLock(gpu_index, rig["lock_dir"]) as lock:
        out["lock_wait_s"] = lock.waited_s
        idle = wait_for_idle(gpu_index, float(tcfg["wait_for_idle_s"]), float(tcfg["idle_util_pct"]))
        out["idle_check"] = idle
        tel_before = telemetry(gpu_index)
        flusher = L2Flusher(int(tcfg["l2_flush_bytes"]), device) if tcfg.get("l2_flush", True) else None
        overhead = measure_host_overhead(device)
        raw = run_interleaved(
            contenders, device, int(tcfg["rounds"]), int(tcfg["iters_per_round"]), flusher, perturb,
            int(tcfg.get("spot_checks", 0)) if with_candidate else 0, int(tcfg["warmup_iters"]),
            float(tcfg["min_warmup_ms"]), float(tcfg["max_warmup_s"]),
        )
        others_after = other_compute_pids(gpu_index)
        tel_after = telemetry(gpu_index)
        del flusher
    set_strict_flags()
    out["telemetry"] = {"before": tel_before, "after": tel_after}
    out["contended"] = bool(idle.get("idle") is False or others_after)
    out["others"] = sorted(set(idle.get("others", [])) | set(others_after))
    out["host_overhead_ms"] = overhead
    sg_cfg = tcfg["stream_guard"]
    out["contenders"] = {}
    for name, r in raw.items():
        out["contenders"][name] = {
            "summary": summarise(r["event_ms"]),
            "host_summary": summarise(r["host_ms"]),
            "stream_guard": stream_guard(r["event_ms"], r["host_ms"], overhead, float(sg_cfg["ratio"]),
                                         float(sg_cfg["slack_ms"])),
            "spot_checks": r["spot_checks"],
            "event_ms": r["event_ms"],
        }
    out["speedups"] = {}
    if "candidate" in raw:
        for name in rig["baselines"]:
            out["speedups"][name] = bootstrap_ratio_ci(raw[name]["event_ms"], raw["candidate"]["event_ms"])
    lo, hi = tcfg.get("runtime_window_ms", [1.0, 15.0])
    out["in_runtime_window"] = {name: lo <= c["summary"]["median"] <= hi for name, c in out["contenders"].items()}
    return out


def _speed_of_light(ref_model: torch.nn.Module, inputs: list, fwd_seed: int, hw: dict) -> dict:
    seed_all(fwd_seed)
    with torch.no_grad():
        flops = count_flops(lambda: ref_model(*inputs))
        ref_out = _forward(ref_model, inputs, fwd_seed)
    nbytes = tensor_bytes(inputs) + model_state_bytes(ref_model) + tensor_bytes(ref_out)
    del ref_out
    sol_cfg = hw.get("sol", {})
    out: dict[str, Any] = {"flops": flops, "bytes": nbytes}
    try:
        bw = float(hw["dram_bandwidth_gbs"]) * float(sol_cfg.get("bandwidth_derate", 1.0))
        out["guard"] = speed_of_light(flops, nbytes, peak_for(hw, sol_cfg.get("guard_peak", "fp16_tensor_fp16_acc")), bw)
        out["efficiency"] = speed_of_light(flops, nbytes, peak_for(hw, sol_cfg.get("efficiency_peak", "tf32_tensor")), bw)
    except (KeyError, TypeError, ValueError) as exc:
        out["error"] = f"hardware file incomplete: {exc}"
    return out


# ---------------------------------------------------------------------------------------------
# Baseline and noise-floor modes (no candidate)
# ---------------------------------------------------------------------------------------------
def evaluate_baseline(task_path: str, rig: dict, hw: dict, task_meta: dict) -> dict:
    device = get_device()
    setup(device)
    set_strict_flags()
    res: dict[str, Any] = {"kind": "baseline", "task_id": task_meta.get("id"), "task_path": str(task_path),
                           "task_sha256": sha256_file(task_path), "started": time.time()}
    ref_mod = load_module(task_path, "kb4090_task_ref")
    base_seed = int(rig.get("seed", 42))
    reset_peak_memory(device)
    try:
        seed_all(base_seed)
        ref_model = ref_mod.Model(*ref_mod.get_init_inputs()).to(device)
        inputs, meta = generate_inputs(ref_mod, base_seed + 7, "base", device)
        res["input_bytes"] = tensor_bytes(inputs)
        res["gen_device"] = meta["gen_device"]
        fwd_seed = base_seed + 11
        out = _forward(ref_model, inputs, fwd_seed)
        sync(device)
        res["output_bytes"] = tensor_bytes(out)
        res["output_finite"] = cmp.all_finite(out)
        del out
        res["peak_mem_gb"] = peak_memory_gb(device)
        res["sol"] = _speed_of_light(ref_model, inputs, fwd_seed, hw)
        res["timing"] = _time_against_baselines(ref_model, None, inputs, fwd_seed, rig, rig["timing"], task_meta,
                                                device, rig["correctness"], with_candidate=False)
        meds = {k: c["summary"]["median"] for k, c in res["timing"]["contenders"].items()}
        res["medians_ms"] = meds
        if "eager_fp32" in meds and "eager_tf32" in meds:
            res["tf32_gain"] = meds["eager_fp32"] / meds["eager_tf32"]
        if res["sol"].get("efficiency"):
            res["pct_of_sol"] = {k: res["sol"]["efficiency"]["sol_ms"] / m for k, m in meds.items()}
        res["verdict"] = "OK"
    except torch.cuda.OutOfMemoryError as exc:
        res["verdict"] = "OOM"
        res["error"] = _record_exception(exc)
        res["peak_mem_gb"] = peak_memory_gb(device)
    except Exception as exc:
        res["verdict"] = "ERROR"
        res["error"] = _record_exception(exc)
    return res


def evaluate_noise(task_path: str, rig: dict, hw: dict, task_meta: dict) -> dict:
    """A/A test: time the same eager FP32 reference as two contenders. Any 'speedup' is noise."""
    device = get_device()
    setup(device)
    set_strict_flags()
    res: dict[str, Any] = {"kind": "noise", "task_id": task_meta.get("id"), "task_path": str(task_path),
                           "started": time.time()}
    ref_mod = load_module(task_path, "kb4090_task_ref")
    base_seed = int(rig.get("seed", 42))
    seed_all(base_seed)
    ref_model = ref_mod.Model(*ref_mod.get_init_inputs()).to(device)
    inputs, _ = generate_inputs(ref_mod, base_seed + 7, "base", device)
    fwd_seed = base_seed + 11

    def run() -> Any:
        with torch.no_grad():
            return ref_model(*inputs)

    def pre() -> None:
        seed_all(fwd_seed)

    rig_aa = {**rig, "baselines": []}
    extra = [Contender("A", run, pre=pre), Contender("B", run, pre=pre)]
    t = _time_against_baselines(ref_model, None, inputs, fwd_seed, rig_aa, rig["timing"], task_meta, device,
                                rig["correctness"], with_candidate=False, extra_contenders=extra)
    a = t["contenders"]["A"]["event_ms"]
    b = t["contenders"]["B"]["event_ms"]
    ci = bootstrap_ratio_ci(a, b)
    from .stats import min_detectable_speedup

    res.update(
        timing=t,
        aa_ratio=ci,
        min_detectable_speedup=min_detectable_speedup(ci),
        cv_a=t["contenders"]["A"]["summary"]["cv"],
        cv_b=t["contenders"]["B"]["summary"]["cv"],
        verdict="OK",
    )
    return res
