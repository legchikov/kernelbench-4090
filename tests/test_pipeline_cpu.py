"""End-to-end pipeline tests in CPU debug mode (KB4090_DEVICE=cpu).

These exercise the evaluation logic (verdicts, hidden distributions, input-mutation and integrity
checks, spot checks, the subprocess worker) without a GPU. Kernel-specific defences (allocator
poisoning, stream guard, kernel attribution) need CUDA and are covered by `kb4090 redteam`.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kb4090.config import deep_merge, load_hardware, load_rig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TASK = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.rand(64, 256)]

def get_init_inputs():
    return []
'''

HEADER = "import torch\nimport torch.nn as nn\n\n"
CANDIDATES = {
    "good": "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x.clamp_min(0)\n",
    "identity": "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x.clone()\n",
    "inplace": "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x.clamp_(min=0)\n",
    "memo": ("class ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.c = {}\n"
             "    def forward(self, x):\n        k = x.data_ptr()\n        if k not in self.c:\n"
             "            self.c = {k: x.clamp_min(0)}\n        return self.c[k]\n"),
    "timer_patch": ("_ev = getattr(getattr(torch, 'cuda'), 'Event')\nsetattr(_ev, 'elapsed_time', lambda s, e: 0.001)\n"
                    "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x.clamp_min(0)\n"),
    "tf32_flag": ("torch.backends.cuda.matmul.allow_tf32 = True\n"
                  "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x.clamp_min(0)\n"),
    "raises": "class ModelNew(nn.Module):\n    def forward(self, x):\n        raise RuntimeError('boom')\n",
    "compile_fail": "raise RuntimeError('nvcc failed')\nclass ModelNew(nn.Module):\n    pass\n",
    "wrong_shape": "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x.clamp_min(0).flatten()\n",
}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KB4090_DEVICE", "cpu")
    task = tmp_path / "task.py"
    task.write_text(TASK)
    paths = {}
    for name, body in CANDIDATES.items():
        p = tmp_path / f"{name}.py"
        p.write_text(HEADER + body)
        paths[name] = p
    rig = deep_merge(load_rig(ROOT / "configs" / "rig.yaml"), {
        "lock_dir": str(tmp_path / "locks"),
        "correctness": {"seeds": 2},
        "timing": {"rounds": 2, "iters_per_round": 3, "warmup_iters": 1, "min_warmup_ms": 0, "l2_flush": False,
                   "wait_for_idle_s": 0, "spot_checks": 2},
        "policy": {"require_custom_kernel": False, "static": {"no_kernel_backend": "off", "torch_ops": "off"}},
        "build": {"extensions_root": str(tmp_path / "ext")},
    })
    meta = {"id": "cpu-relu", "abs_path": str(task), "perturb": True, "sol_margin": None}
    return rig, load_hardware("rtx4090"), meta, paths


def run(env, name):
    from kb4090.evaluate import evaluate_candidate

    rig, hw, meta, paths = env
    return evaluate_candidate(meta["abs_path"], str(paths[name]), rig, hw, meta)


def test_good_candidate_passes(env):
    res = run(env, "good")
    assert res["verdict"] == "PASS", res["reasons"]
    assert set(res["speedups"]) == {"eager_fp32", "eager_tf32"}
    assert res["correctness"]["summary"]["neg"]["passed"] == 2
    assert all(s["ok"] for s in res["timing"]["contenders"]["candidate"]["spot_checks"])


def test_identity_fails_only_on_hidden_distribution(env):
    res = run(env, "identity")
    assert res["verdict"] == "FAIL_INCORRECT"
    assert res["correctness"]["hidden_only_failure"] is True


def test_inplace_mutation_is_invalid(env):
    res = run(env, "inplace")
    assert res["verdict"] == "INVALID"
    assert any("modified its inputs" in r["reason"] for r in res["reasons"])


def test_memoisation_is_not_pass(env):
    assert run(env, "memo")["verdict"] in {"FAIL_INCORRECT", "INVALID", "SUSPECT"}


def test_runtime_integrity_catches_obfuscated_patch(env):
    res = run(env, "timer_patch")
    assert res["static"]["errors"] == []  # static check misses it by design
    assert res["verdict"] == "INVALID" and "harness_tamper" in res["reasons"][0]["reason"]


def test_static_error_stops_early(env):
    res = run(env, "tf32_flag")
    assert res["verdict"] == "INVALID" and "numeric_flag_tamper" in res["reasons"][0]["reason"]
    assert "correctness" not in res


def test_runtime_compile_and_shape_failures(env):
    assert run(env, "raises")["verdict"] == "FAIL_RUNTIME"
    assert run(env, "compile_fail")["verdict"] == "FAIL_COMPILE"
    assert run(env, "wrong_shape")["verdict"] == "FAIL_INCORRECT"


def test_dev_mode_hides_hidden_distributions(env):
    from kb4090.evaluate import evaluate_candidate

    rig, hw, meta, paths = env
    res = evaluate_candidate(meta["abs_path"], str(paths["identity"]), rig, hw, meta, mode="dev")
    assert res["verdict"] == "PASS"  # dev feedback only uses base inputs...
    assert set(res["correctness"]["summary"]) == {"base"}  # ...and says nothing about the others


def test_baseline_and_noise_modes(env):
    from kb4090.evaluate import evaluate_baseline, evaluate_noise

    rig, hw, meta, _ = env
    b = evaluate_baseline(meta["abs_path"], rig, hw, meta)
    assert b["verdict"] == "OK" and set(b["medians_ms"]) == {"eager_fp32", "eager_tf32"}
    assert b["sol"]["guard"]["bound"] == "memory"
    n = evaluate_noise(meta["abs_path"], rig, hw, meta)
    assert n["verdict"] == "OK" and n["min_detectable_speedup"] >= 1.0


def test_subprocess_worker_roundtrip(env, tmp_path):
    from kb4090.runner import run_worker

    rig, hw, meta, paths = env
    good = run_worker("official", meta, rig, hw, tmp_path / "out" / "good.json", candidate=paths["good"])
    assert good["verdict"] == "PASS", good.get("log_tail")
    bad = run_worker("official", meta, rig, hw, tmp_path / "out" / "identity.json", candidate=paths["identity"])
    assert bad["verdict"] == "FAIL_INCORRECT"
