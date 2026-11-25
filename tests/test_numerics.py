"""Comparison, distributions, fingerprints and FLOP counting, on CPU."""

import types

import pytest

torch = pytest.importorskip("torch")

from kb4090 import compare as cmp  # noqa: E402
from kb4090.distributions import generate_inputs, patched_rng, perturb_inplace  # noqa: E402
from kb4090.sol import count_flops, tensor_bytes  # noqa: E402

TOL = dict(atol=1e-2, rtol=1e-2, rel_l2_max=1e-2)


def test_identical_outputs_pass():
    x = torch.randn(1000)
    assert cmp.compare(x, x.clone(), **TOL)["ok"]


def test_small_error_passes_large_fails():
    x = torch.randn(1000)
    assert cmp.compare(x, x + 1e-4, **TOL)["ok"]
    res = cmp.compare(x, x + 0.5, **TOL)
    assert not res["ok"] and "outside" in res["reason"]


def test_rel_l2_catches_tiny_outputs():
    # Outputs of size ~0.001 all pass atol=1e-2 even if completely wrong; rel-L2 must catch it.
    ref = torch.rand(10000) * 1e-3
    wrong = torch.rand(10000) * 1e-3
    res = cmp.compare(ref, wrong, **TOL)
    assert not res["ok"] and "relative L2" in res["reason"]


def test_nan_where_reference_finite_fails():
    ref = torch.ones(10)
    cand = ref.clone()
    cand[3] = float("nan")
    res = cmp.compare(ref, cand, **TOL)
    assert not res["ok"] and "non-finite" in res["reason"]


def test_dtype_shape_structure_and_subclass():
    ref = torch.ones(4, 4)
    assert "dtype" in cmp.compare(ref, ref.half(), **TOL)["reason"]
    assert "shape" in cmp.compare(ref, torch.ones(16), **TOL)["reason"]
    assert "structure" in cmp.compare((ref, ref), ref, **TOL)["reason"]

    class Sub(torch.Tensor):
        pass

    assert "subclass" in cmp.compare(ref, ref.clone().as_subclass(Sub), **TOL)["reason"]


def test_integer_outputs_must_match_exactly():
    ref = torch.arange(10)
    cand = ref.clone()
    cand[0] = 1
    assert not cmp.compare(ref, cand, **TOL)["ok"]


def test_chunking_matches_unchunked(monkeypatch):
    ref = torch.randn(10007)
    cand = ref + torch.randn(10007) * 1e-3
    full = cmp.compare(ref, cand, **TOL)
    monkeypatch.setattr(cmp, "CHUNK_ELEMS", 1000)
    chunked = cmp.compare(ref, cand, **TOL)
    assert full["ok"] == chunked["ok"]
    assert abs(full["rel_l2"] - chunked["rel_l2"]) < 1e-9


def test_fingerprint_detects_mutation():
    x = torch.rand(1000)
    fp = cmp.fingerprint(x)
    assert cmp.same_fingerprint(fp, cmp.fingerprint(x))
    x[500] += 1.0
    assert not cmp.same_fingerprint(fp, cmp.fingerprint(x))


def _task():
    mod = types.ModuleType("t")

    def get_inputs():
        a = torch.rand(8, 8)
        return [torch.triu(a), torch.randint(0, 5, (4,)), torch.softmax(torch.randn(3, 5), dim=-1), 7]

    mod.get_inputs = get_inputs
    return mod


def test_distributions_transform_floats_and_keep_structure():
    base, _ = generate_inputs(_task(), 1, "base", torch.device("cpu"))
    neg, _ = generate_inputs(_task(), 1, "neg", torch.device("cpu"))
    assert torch.equal(neg[0], -base[0])           # values negated, triangular structure kept
    assert torch.equal(torch.triu(neg[0]), neg[0])
    assert torch.equal(neg[1], base[1])            # randint untouched
    assert torch.allclose(neg[2].sum(-1), torch.ones(3))  # softmax still sums to 1
    assert neg[3] == 7
    x3, _ = generate_inputs(_task(), 1, "x3", torch.device("cpu"))
    assert torch.allclose(x3[0], 3 * base[0])


def test_patched_rng_restores_generators():
    orig = torch.rand
    with patched_rng("x3", None):
        assert torch.rand is not orig
    assert torch.rand is orig


def test_perturb_is_stable_over_pairs():
    x = torch.rand(100)
    x0 = x.clone()
    perturb_inplace([x], 0)
    assert not torch.allclose(x, x0)
    perturb_inplace([x], 1)
    assert torch.allclose(x, x0, rtol=1e-6)


def test_flop_counting_and_bytes():
    a, b = torch.rand(64, 32), torch.rand(32, 16)
    assert count_flops(lambda: a @ b) == 2 * 64 * 32 * 16
    assert tensor_bytes([a, b, a]) == (64 * 32 + 32 * 16) * 4  # shared storage counted once


def test_kernel_attribution_by_declared_names():
    from kb4090.runtime_checks import classify_kernels, kernel_base_name

    assert kernel_base_name("void ns::relu_vec4<float>(float4 const*, float4*, long)") == "relu_vec4"
    assert kernel_base_name("relu_kernel") == "relu_kernel"
    cand = {"available": True, "kernels": [
        {"name": "relu_vec4(float4 const*, float4*, long)", "time_us": 90.0, "calls": 1},
        {"name": "relu_kernel_0d1d2d", "time_us": 5.0, "calls": 1},
        {"name": "void at::native::vectorized_elementwise_kernel<4, ...>", "time_us": 5.0, "calls": 1},
        {"name": "sgemm_tiled(float const*)", "time_us": 0.0, "calls": 1},
    ]}
    res = classify_kernels(cand, {"kernels": []}, declared=["relu_vec4", "relu_kernel", "sgemm_tiled"])
    assert res["n_custom"] == 3 and res["method"] == "declared_names"
    assert res["custom_fraction"] == 0.95
    # Without declared names, the fallback must not call a user kernel named like a library one "library"
    # just because it contains "gemm" (only cuBLAS/ATen-style prefixes count).
    fb = classify_kernels({"available": True, "kernels": [{"name": "sgemm_tiled(float const*)", "time_us": 1.0,
                                                           "calls": 1}]}, {"kernels": []})
    assert fb["n_custom"] == 1
