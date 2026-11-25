"""Configuration loading.

Everything that changes a measurement lives in a YAML file under ``configs/`` or ``hardware/``,
and every result records a hash of the exact configuration that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DEFAULT_RIG: dict[str, Any] = {
    "gpu_index": 0,
    "hardware": "rtx4090",
    "kernelbench_root": "../KernelBench",
    "kernelbench_commit": None,
    "results_dir": "results",
    "runs_dir": "runs",
    "lock_dir": "/var/tmp/kb4090-locks",
    "seed": 42,
    "correctness": {
        "seeds": 5,
        "distributions": ["base", "x3", "x0p01", "neg"],
        "atol": 1e-2,
        "rtol": 1e-2,
        "rel_l2_max": 1e-2,
    },
    "dev": {
        "seeds": 2,
        "distributions": ["base"],
        "rounds": 3,
        "iters_per_round": 10,
    },
    "timing": {
        "warmup_iters": 5,
        "min_warmup_ms": 200.0,
        "max_warmup_s": 600.0,
        "rounds": 5,
        "iters_per_round": 20,
        "l2_flush": True,
        "l2_flush_bytes": 256 * 1024 * 1024,
        "perturb_inputs": True,
        "spot_checks": 3,
        "wait_for_idle_s": 120,
        "idle_util_pct": 10,
        "stream_guard": {"enabled": True, "ratio": 1.5, "slack_ms": 0.05},
        "runtime_window_ms": [1.0, 15.0],
    },
    "baselines": ["eager_fp32", "eager_tf32"],
    "headline_baseline": "eager_tf32",
    "policy": {
        "allowed_backends": ["cuda", "triton"],
        "require_custom_kernel": True,
        "min_custom_gpu_fraction": 0.0,
        "warn_custom_gpu_fraction": 0.2,
        "sol_margin": 0.9,
        "candidate_tf32": False,
        "static": {},
        "env_allowlist": ["TORCH_CUDA_ARCH_LIST", "MAX_JOBS"],
    },
    "build": {
        "torch_cuda_arch_list": "8.9",
        "extensions_root": "~/.cache/kb4090/torch_extensions",
    },
    "eval_timeout_s": 1800,
    "sandbox_prefix": None,
}


def deep_merge(base: dict, override: dict | None) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_rig(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the rig configuration: defaults <- configs/rig.yaml <- optional overlay <- overrides."""
    rig_path = Path(path or os.environ.get("KB4090_RIG") or REPO_ROOT / "configs" / "rig.yaml")
    rig = deep_merge(DEFAULT_RIG, load_yaml(rig_path) if rig_path.exists() else {})
    overlay = os.environ.get("KB4090_RIG_OVERLAY")
    if overlay:
        rig = deep_merge(rig, load_yaml(overlay))
        rig["_overlay"] = str(overlay)
    rig = deep_merge(rig, overrides)
    rig["_source"] = str(rig_path)
    return rig


def config_hash(cfg: Any) -> str:
    """Stable short hash of a config, ignoring private keys (those starting with '_')."""

    def strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in sorted(obj.items()) if not str(k).startswith("_")}
        if isinstance(obj, list | tuple):
            return [strip(v) for v in obj]
        return obj

    blob = json.dumps(strip(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def kernelbench_root(rig: dict) -> Path | None:
    value = os.environ.get("KB_ROOT") or rig.get("kernelbench_root")
    return resolve_path(value) if value else None


def load_hardware(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = REPO_ROOT / "hardware" / f"{name_or_path}.yaml"
    hw = load_yaml(p)
    hw["_source"] = str(p)
    return hw


TASK_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "perturb": True,
    "sol_margin": None,
    "notes": "",
}


def load_tasks(path: str | Path | None = None) -> list[dict]:
    p = Path(path or os.environ.get("KB4090_TASKS") or REPO_ROOT / "configs" / "tasks.yaml")
    data = load_yaml(p)
    tasks = []
    seen: set[str] = set()
    for raw in data.get("tasks", []):
        for key in ("id", "level", "path"):
            if key not in raw:
                raise ValueError(f"task entry {raw!r} in {p} is missing '{key}'")
        if raw["id"] in seen:
            raise ValueError(f"duplicate task id {raw['id']!r} in {p}")
        seen.add(raw["id"])
        tasks.append({**TASK_DEFAULTS, **raw})
    return tasks


def resolve_task_path(task: dict, rig: dict) -> Path:
    """Find a task file: absolute path, path relative to this repo, or path inside KernelBench/."""
    raw = Path(os.path.expandvars(str(task["path"]))).expanduser()
    if raw.is_absolute():
        return raw
    local = REPO_ROOT / raw
    if local.exists():
        return local.resolve()
    root = kernelbench_root(rig)
    if root is None:
        raise FileNotFoundError(f"set kernelbench_root in configs/rig.yaml or KB_ROOT to find {raw}")
    return (root / "KernelBench" / raw).resolve()


def find_task(ref: str, tasks: list[dict], rig: dict) -> dict:
    """Look a task up by id, or wrap an arbitrary path as an ad-hoc task."""
    for task in tasks:
        if task["id"] == ref:
            return {**task, "abs_path": str(resolve_task_path(task, rig))}
    p = Path(ref).expanduser()
    if p.exists():
        return {**TASK_DEFAULTS, "id": p.stem, "level": None, "path": str(p), "abs_path": str(p.resolve())}
    raise KeyError(f"unknown task {ref!r}: not an id in configs/tasks.yaml and not an existing file")


def load_agents(path: str | Path | None = None) -> dict:
    p = Path(path or os.environ.get("KB4090_AGENTS") or REPO_ROOT / "configs" / "agents.yaml")
    data = load_yaml(p)
    data["_source"] = str(p)
    return data
