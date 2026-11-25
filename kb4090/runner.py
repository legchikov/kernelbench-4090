"""Run the evaluation worker in a fresh, isolated subprocess.

* ``python -I``: the candidate's directory and PYTHONPATH can't shadow modules (install this repo
  with ``pip install -e .`` so the worker can import kb4090).
* One process per evaluation: a kernel that corrupts the CUDA context only takes itself down.
* Per-candidate TORCH_EXTENSIONS_DIR: no stale builds shared between candidates.
* TORCH_CUDA_ARCH_LIST=8.9: compile for the 4090 only (faster builds).
* Optional sandbox prefix (e.g. scripts/sandbox.sh --no-net ...) from configs/rig.yaml.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import REPO_ROOT, resolve_path


def _sha12(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def run_worker(
    mode: str,
    task: dict,
    rig: dict,
    hw: dict,
    out_path: str | Path,
    candidate: str | Path | None = None,
    timeout_s: float | None = None,
) -> dict:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_s = float(timeout_s or rig.get("eval_timeout_s", 1800))

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # make CUDA indices match nvidia-smi / NVML
    env["CUDA_VISIBLE_DEVICES"] = str(rig["gpu_index"])
    env["TORCH_CUDA_ARCH_LIST"] = str(rig["build"]["torch_cuda_arch_list"])
    ext_root = Path(rig["build"]["extensions_root"]).expanduser()
    build_dir = ext_root / (_sha12(candidate) if candidate else "baseline")
    build_dir.mkdir(parents=True, exist_ok=True)
    env["TORCH_EXTENSIONS_DIR"] = str(build_dir)
    env["TRITON_CACHE_DIR"] = str(build_dir / "triton")  # per-candidate, and writable inside a sandbox
    env.pop("NVIDIA_TF32_OVERRIDE", None)  # never let the environment change FP32 numerics
    Path(rig["lock_dir"]).expanduser().mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kb4090-") as tmp:
        rig_json = Path(tmp) / "rig.json"
        hw_json = Path(tmp) / "hw.json"
        meta_json = Path(tmp) / "task.json"
        rig_json.write_text(json.dumps(rig, default=str))
        hw_json.write_text(json.dumps(hw, default=str))
        meta_json.write_text(json.dumps(task, default=str))
        cmd = [sys.executable, "-I", "-m", "kb4090.worker", "--mode", mode, "--task", str(task["abs_path"]),
               "--out", str(out_path), "--rig-json", str(rig_json), "--hw-json", str(hw_json),
               "--task-meta-json", str(meta_json)]
        if candidate:
            cmd += ["--candidate", str(Path(candidate).resolve())]
        prefix = rig.get("sandbox_prefix")
        if prefix:
            cmd = [str(resolve_path(p, REPO_ROOT)) if i == 0 else
                   p.format(build_dir=build_dir, out_dir=out_path.parent, lock_dir=rig["lock_dir"], tmp=tmp)
                   for i, p in enumerate(prefix)] + cmd

        log_path = out_path.with_suffix(".log")
        started = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                rc = None

    log_tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
    if rc is None:
        res = {"verdict": "FAIL_TIMEOUT", "reasons": [{"verdict": "FAIL_TIMEOUT",
                                                       "reason": f"evaluation exceeded {timeout_s:.0f}s"}]}
    elif out_path.exists():
        res = json.loads(out_path.read_text())
    else:
        hint = ""
        if "No module named kb4090" in log_tail:
            hint = " (install this repo with `pip install -e .`; the worker runs with python -I)"
        verdict = "FAIL_RUNTIME" if candidate and not hint else "HARNESS_ERROR"
        res = {"verdict": verdict,
               "reasons": [{"verdict": verdict, "reason": f"worker exited with code {rc} "
                            f"without writing a result{hint}"}]}
    res.setdefault("task_id", task.get("id"))
    res["wall_s"] = time.time() - started
    res["log_path"] = str(log_path)
    res["log_tail"] = log_tail[-1500:]
    out_path.write_text(json.dumps(res, indent=1, default=str))
    return res
