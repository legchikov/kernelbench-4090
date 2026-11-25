# Methodology

This is what the harness measures, why, and where it can still be wrong. Link to it from every post.

## 1. What is held fixed

- **Hardware and software**: one RTX 4090, locked graphics clock, persistence mode on, ideally not
  driving a display. Driver, CUDA, PyTorch and Triton versions are recorded in an environment
  fingerprint (`kb4090 env`). Every ledger row carries its `env_id` and a hash of `configs/rig.yaml`.
- **Benchmark**: KernelBench at a pinned commit, and a frozen task subset (`configs/tasks.yaml`).
  Prune the subset once, after Phase 1, then never change it within a study.
- **Budget**: the same dollar budget per task for every system being compared, with three or more
  runs per (system, task).

## 2. Correctness

Ground truth is the reference `Model` in strict FP32 (TF32 off for matmul and cuDNN).

For each of `seeds` (default 5) and each distribution:

| Distribution | Transform of every float drawn by `torch.rand`/`randn` inside `get_inputs()` |
|---|---|
| `base` | none |
| `x3` | × 3 |
| `x0p01` | × 0.01 |
| `neg` | × −1 |

Transforming the raw draws, rather than the final inputs, keeps whatever structure the task adds
afterwards (softmax rows still sum to 1, triangular matrices stay triangular). A distribution only
applies if the reference output stays finite; otherwise it is recorded as not applicable.

A candidate output passes when, for every output tensor: it is a plain `torch.Tensor` with the
reference's shape and dtype, it is finite wherever the reference is, every element satisfies
`|c − r| ≤ atol + rtol·|r|` (KernelBench's rule, `atol = rtol = 1e-2`), and the relative L2 error
is ≤ `1e-2`. The relative-L2 check matters for `x0p01`, where `atol = 1e-2` alone would accept
almost anything.

Before the candidate runs on each trial:
- the reference output is copied to the host and its GPU blocks are refilled with `0xFF` (NaN),
  so a candidate returning `torch.empty` can't inherit the answer;
- inputs are fingerprinted (float64 sums plus a 4096-element sample) and checked again afterwards;
- the RNG is reseeded identically, because some tasks (e.g. `level3/35_LSTM`) draw random states
  inside `forward()`.

Reference weights are copied into the candidate when parameter names and shapes match, which
removes KernelBench's dependence on identical initialisation order.

## 3. Did the candidate's own kernel run?

One forward pass of each model runs under the PyTorch profiler. A kernel counts as **custom** if its
base name matches a `__global__` function in the candidate's CUDA source or a `@triton.jit`
function in its Python. With the default policy at least one custom kernel must run, and the
custom share of GPU time is reported. KernelBench allows partial replacement, so a small share
is only a warning by default (see the `hack_decoy_kernel` grey-zone case).

## 4. Timing

- All contenders (eager FP32, eager TF32, optionally `torch.compile`, and the candidate) share the
  same input tensors and take turns in rounds, with the order rotated each round.
- Each iteration: set flags and seed → rescale inputs in place (×1.25 / ×0.8 alternating) → flush
  L2 (write 256 MiB) → device-wide sync → CUDA events around `forward` → device-wide sync.
- The headline number is the CUDA-event median. Speedup = median(baseline) / median(candidate),
  with a 95% bootstrap CI.
- **Stream guard**: host wall time across the device-wide sync, minus the measured fixed overhead,
  must not exceed 1.5 × event time + 0.05 ms. Work on another stream shows up in host time but not
  in event time.
- **Spot checks**: three timed candidate outputs are compared with the reference on the same
  (rescaled) inputs. A memoised kernel returns a stale answer and fails.
- Timing holds an exclusive per-GPU file lock and first waits for the GPU to be idle. Other
  processes seen during timing are recorded as contention.
- KernelBench v0.1 sized Level 1-2 problems to run 1-15 ms on an H100. Results record whether each
  baseline falls in that window on the 4090.

## 5. Speed of light

`FLOPs` come from `torch.utils.flop_counter` (matmul, conv, attention; element-wise ops count as 0)
and `bytes` are the compulsory DRAM traffic under a cold L2 (inputs, parameters and buffers read
once, outputs written once). The bound is `max(FLOPs / peak, bytes / bandwidth)`.

- The **guard** uses the most generous peak (FP16 tensor cores with FP16 accumulate), so it can
  only flag the physically impossible. A candidate below `0.9 ×` the guard is `SUSPECT`. If a
  PyTorch baseline beats the guard, the bound is wrong for that task and is ignored.
- **% of SOL** in tables uses the TF32 peak, which is what an FP32-correct kernel can realistically use.
- Peaks come from `hardware/rtx4090.yaml`, which stays marked unverified until checked against
  NVIDIA's whitepaper.

## 6. Reporting rules

1. Headline: PASS-only speedups vs eager TF32. Non-PASS runs count as 1.0.
2. Per task: the median of runs. Best-of-n is shown, labelled, and never used as the headline.
3. Across tasks: geometric mean of per-task medians, `fast_p` for p ∈ {1.0, 1.2, 1.5, 2.0} on medians.
4. Always publish: the env fingerprint, the rig hash, the A/A noise floor for the tasks shown, and
   the red-team table for the exact harness commit.
5. SUSPECT results need a written review (`kb4090 review --row-id … --note …`) before they count.
   Look at the kernel, profile it with Nsight Compute, and explain where the time went.
6. Negative results get published too.

## 7. Known gaps

- Correctness is checked on the task's shapes only. Shape-specialised kernels are allowed, as in
  KernelBench.
- The fingerprint can in principle miss a mutation that preserves both sums and every sampled
  element. In practice the `neg` distribution makes in-place writes visible.
- A candidate that hides work in a library call the reference also uses still passes attribution
  if it also runs a declared kernel. The custom-time share is the signal to read.
- Agents in the same EAT team can still run ad-hoc GPU work outside the lock; final numbers come
  from `eat-final` on an idle GPU.
