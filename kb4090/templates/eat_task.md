## Hard Requirements (MUST be in every agent's prompt)

**Task.** Make {{N_TASKS}} KernelBench problems faster on one NVIDIA RTX 4090 by replacing their
PyTorch implementations with your own GPU kernels. For each problem, the reference is class `Model`
in the reference file below; write class `ModelNew` in the matching file under `solutions/`.

{{TASK_LIST}}

**Goal.** Maximise the score printed by `bash eat/eval.sh` (last line `SCORE: <x>`): the geometric
mean, over all problems, of the verified speedup against PyTorch eager with TF32 enabled. An
unchanged solution counts as 1.0. **Any incorrect, invalid or suspicious solution makes the whole
score 0**, so never leave a broken file in `solutions/`.

**Read first.** This file, then each reference file above, then `CLAUDE.md`.

**Files.**
- You may edit only `solutions/*.py` and files inside your own working copy `work_<role>/`.
- Never edit anything under `eat/`, `CLAUDE.md`, `.gitignore`, or the reference files.
- `ModelNew` must keep `Model`'s constructor arguments and `forward` inputs, and return outputs of
  the same shape and dtype.

**What counts as a solution.**
- The core computation must run in your own kernels: CUDA via `torch.utils.cpp_extension.load_inline`,
  or Triton via `@triton.jit`. Pure PyTorch/cuBLAS/cuDNN solutions are rejected.
- Banned in `solutions/*.py`: changing `torch.backends.*`/TF32/default dtype, setting environment
  variables, `torch.compile`/`torch.jit`, threads or processes, extra CUDA streams, caching outputs
  between calls, modifying inputs in place, `try/except` fallbacks, inspecting the call stack or
  garbage collector, reading files, network access.

**Honesty.** Results must hold for any valid input. The official eval uses unseen seeds and
transformed input values (different scales and signs) and checks that your kernel actually ran on
the timed stream. Do not special-case shapes, values or the evaluation's inputs. Do not try to
read, infer or game the evaluation. A result that beats the hardware's speed of light is
treated as a bug, not a win.

**Resources.**
- GPU {{GPU}} only. It is shared by the whole team.
- **All performance measurement goes through `bash eat/eval.sh` or
  `kb4090 dev-eval --task <reference file> --candidate <your file>`.** Both hold a GPU lock so
  timings aren't polluted. Never run your own timing loops or benchmarks. Compiling and short
  correctness tests in your own scripts are fine.
- Time limits: `kb4090 dev-eval` takes 1-3 minutes per problem; `bash eat/eval.sh` takes up to
  about {{N_TASKS}} x 5 minutes. Keep every other command under 10 minutes.

**Sources.** You may read public documentation (CUDA, PTX, Triton, PyTorch, CUTLASS) and general
optimisation write-ups. Do not look up or copy published solutions to these specific KernelBench
problems.

**Git.** Commit locally if the harness asks you to. Never push to any remote.

## Shared Environment

- All agents share this directory. Each agent works in its own copy `work_<role>/` (copy the
  relevant `solutions/<problem>.py` there and iterate on it).
- Only the agent that ships a new best for a problem copies it into `solutions/<problem>.py`, after
  `kb4090 dev-eval` passes on it, and adds one line to `work_results.md`: problem, role, dev-eval
  speedup, one-sentence description of the change.
- Example dev check:
  `kb4090 dev-eval --task {{FIRST_TASK_PATH}} --candidate work_<role>/{{FIRST_TASK_ID}}.py`

### Hardware

{{HARDWARE}}
