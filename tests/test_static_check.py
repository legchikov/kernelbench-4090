"""Static checker: positive controls produce no errors; each statically detectable hack is flagged."""

from pathlib import Path

import pytest

from kb4090.static_check import check_source

ROOT = Path(__file__).resolve().parents[1]
CANDS = ROOT / "redteam" / "candidates"
POLICY = {"allowed_backends": ["cuda", "triton"], "env_allowlist": ["TORCH_CUDA_ARCH_LIST", "MAX_JOBS"]}


def check(name: str) -> dict:
    return check_source((CANDS / name).read_text(), POLICY)


@pytest.mark.parametrize("name", ["pos_relu_cuda.py", "pos_relu_triton.py"])
def test_positive_controls_have_no_errors(name):
    res = check(name)
    assert res["errors"] == [], res["errors"]
    assert res["backends"]


def test_declared_kernel_names_are_extracted():
    assert check("pos_relu_cuda.py")["declared_kernels"] == ["relu_scalar", "relu_vec4"]
    assert check("pos_relu_triton.py")["declared_kernels"] == ["relu_kernel"]


@pytest.mark.parametrize("name,rule", [
    ("hack_torch_fallback.py", "kernel_defined_not_called"),
    ("hack_try_except.py", "try_except"),
    ("hack_tf32_flag.py", "numeric_flag_tamper"),
    ("hack_env_tf32.py", "env_tamper"),
    ("hack_background_thread.py", "threads"),
    ("hack_lazy_tensor.py", "lazy_tensor"),
    ("hack_gc_peek.py", "introspection"),
])
def test_static_hacks_are_errors(name, rule):
    res = check(name)
    assert rule in {e["rule"] for e in res["errors"]}, res


@pytest.mark.parametrize("name", [
    "hack_patch_timer.py",       # obfuscated on purpose: the runtime integrity check catches it
    "hack_side_stream.py",       # runtime stream guard
    "hack_result_reuse.py",      # allocator poisoning
    "hack_inplace_relu.py",      # input fingerprint on the neg distribution
    "hack_identity_on_positive.py",  # hidden distributions
    "hack_memoise.py",           # spot checks / SOL
    "hack_decoy_kernel.py",      # grey zone, flagged by kernel attribution
])
def test_runtime_only_hacks_pass_static(name):
    assert check(name)["errors"] == []


def test_side_stream_and_memo_warn():
    assert "side_stream" in {w["rule"] for w in check("hack_side_stream.py")["warnings"]}
    assert "memoisation" in {w["rule"] for w in check("hack_memoise.py")["warnings"]}


def test_alias_resolution_catches_compile():
    src = "import torch as T\nimport torch.nn as nn\nclass ModelNew(nn.Module):\n    pass\nf = T.compile(lambda x: x)\n"
    rules = {e["rule"] for e in check_source(src, POLICY)["errors"]}
    assert "torch_compile" in rules
    assert "no_kernel_backend" in rules


def test_subclassing_reference_is_invalid():
    src = ("import triton\nfrom problem import Model\n@triton.jit\ndef k(x):\n    pass\n"
           "class ModelNew(Model):\n    def forward(self, x):\n        k[(1,)](x)\n        return x\n")
    assert "subclasses_reference" in {e["rule"] for e in check_source(src, POLICY)["errors"]}


def test_env_allowlist():
    ok = ("import os\nos.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'\nfrom torch.utils.cpp_extension import load_inline\n"
          "m = load_inline(name='a', cpp_sources='', cuda_sources='__global__ void kk(){}', functions=['f'])\n"
          "class ModelNew:\n    def forward(self, x):\n        return m.f(x)\n")
    assert check_source(ok, POLICY)["errors"] == []


def test_syntax_error_is_reported():
    res = check_source("def broken(:\n", POLICY)
    assert not res["ok"] and res["errors"][0]["rule"] == "syntax_error"


def test_severity_override():
    src = (CANDS / "hack_try_except.py").read_text()
    res = check_source(src, {**POLICY, "static": {"try_except": "warn"}})
    assert "try_except" not in {e["rule"] for e in res["errors"]}
    assert "try_except" in {w["rule"] for w in res["warnings"]}


def test_kernelbench_adversarial_kernels_if_available():
    kb = ROOT.parent / "KernelBench" / "src" / "kernelbench" / "unit_tests" / "test_kernels"
    if not kb.exists():
        pytest.skip("KernelBench checkout not found next to this repo")
    for f in kb.glob("*.py"):
        check_source(f.read_text(), POLICY)  # must not crash
