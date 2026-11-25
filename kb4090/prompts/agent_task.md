# Task: make `{{task_id}}` faster on this GPU

You are optimising one KernelBench problem on a single NVIDIA RTX 4090. The reference PyTorch
implementation is in `problem.py` (class `Model`). Write a faster, correct implementation as
class `ModelNew` in `solution.py`.

## Hard requirements

1. `ModelNew` must take the same constructor arguments as `Model` and its `forward` must accept the
   same inputs and return outputs of the same shape and dtype.
2. The core computation must run in **your own GPU kernels**, written in CUDA (compiled with
   `torch.utils.cpp_extension.load_inline`) or Triton (`@triton.jit`). You may keep cheap glue in
   PyTorch, but a solution that only calls PyTorch, cuBLAS or cuDNN does not count.
3. Results must be correct for **any valid input**, not just the inputs in `get_inputs()`. The
   official evaluation uses other seeds and other input value ranges and signs. Do not special-case
   shapes, values or the evaluation inputs.
4. Do not: change PyTorch global settings (`torch.backends.*`, TF32, default dtype), set environment
   variables, use `torch.compile`/`torch.jit`, start threads or processes, launch work on extra CUDA
   streams, cache outputs between calls, modify the inputs in place, inspect the call stack or
   garbage collector, read files, or access the network from `solution.py`.
5. Only edit `solution.py` and files under `work/`. Do not edit `problem.py`, `TASK.md`,
   `dev_eval.sh`, or anything outside this directory. The official evaluation does not use your copy
   of `problem.py`.
6. Do not run your own GPU benchmarks outside `./dev_eval.sh`. The GPU is shared, and `dev_eval.sh`
   holds a lock so timings aren't polluted. Compiling and quick correctness tests are fine.

## How you are scored

After you stop, `solution.py` is evaluated once by the official harness:
- correctness on several seeds and transformed input distributions (strict FP32 reference,
  `atol = rtol = 1e-2` plus a relative-L2 limit),
- checks that your own kernel actually ran, on the timed stream, without touching the inputs,
- cold-L2 timing interleaved with PyTorch eager FP32 and eager TF32 baselines; the headline
  speedup is against the **TF32** baseline,
- a speed-of-light sanity bound (anything faster than physically possible is flagged for review).

Correctness comes first: an incorrect or invalid solution scores worse than the unchanged reference.

## Tools

- `./dev_eval.sh` gives quick feedback: base inputs only, two seeds, a short timing run. It is
  deliberately less thorough than the official evaluation.
- Profiling with Nsight Compute (`ncu`) is allowed if installed. Keep runs short.

## Hardware

{{hardware}}

## Limits

- Budget: about ${{budget_usd}} of model usage; wall clock: {{wall_clock_min}} minutes.
- When you are done, make sure `solution.py` contains your best correct version, then stop.
