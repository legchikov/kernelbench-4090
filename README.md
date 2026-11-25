# kernelbench-4090

**How far do coding agents get on GPU kernel optimisation with a single consumer GPU, when the
numbers are measured so they hold up to scrutiny?**

This repo is a hardened evaluation harness and experiment kit for
[KernelBench](https://github.com/ScalingIntelligence/KernelBench) on one NVIDIA RTX 4090 (Ada,
sm_89). It compares:

- a **single agent** (Claude Code, DeepSeek Harness) at a fixed dollar budget,
- an **agent team** that a supervisor evolves ([EAT, Evolving Agent Teams](https://github.com/caoshiyi/EAT),
  Cao et al., UC Berkeley Sky Computing Lab), at the same budget,

on a frozen subset of 24 KernelBench problems, with every result going through the same official
evaluation.

> **Status: v0.1.** CPU unit tests and the end-to-end pipeline tests pass in CI, and the red-team
> CUDA kernels compile for sm_89. None of the GPU paths have run on a real RTX 4090 yet. The
> first job is `kb4090 redteam` on the card: until every case behaves as expected, no number from
> this repo should be published.

## Why another harness?

Fast-looking KernelBench results are cheap. Agents find ways round the measurement: launching
work on another CUDA stream, returning the reference's freed output buffer, specialising to the
test inputs, flipping TF32 on, caching results. The original fp32 reference also leaves TF32
off, which inflates matmul-heavy speedups. This harness:

| Defence | What it catches |
|---|---|
| AST static checks (+ KernelBench's own regex checker if installed) | PyTorch fallbacks, `try/except` fallbacks, `torch.compile`, global flag or env tampering, threads, lazy tensor subclasses, introspection, file/network access |
| Kernel attribution with the PyTorch profiler | "Kernel defined but never called": the candidate must run a `__global__`/`@triton.jit` kernel it declared itself |
| Four input distributions (`base`, `×3`, `×0.01`, `×−1`), applied to the raw random draws so task structure is preserved | Kernels specialised to the test inputs (e.g. a "ReLU" that returns its input) |
| Allocator poisoning (freed blocks filled with NaN) | Returning uninitialised memory that happens to hold the reference's output |
| Input fingerprints before and after `forward` | In-place mutation of inputs |
| Runtime integrity snapshot of timing and comparison functions and global flags | Monkey-patching, including obfuscated patches the static check can't see |
| Stream guard: CUDA-event time vs host time across a device-wide sync | Work launched off the timed stream |
| In-place input perturbation between timing iterations, plus spot checks | Output caching / memoisation |
| Speed-of-light bound (roofline from FLOPs and compulsory DRAM bytes) | Anything faster than physically possible → `SUSPECT`, for human review |
| TF32 baseline as the headline, strict FP32 as ground truth | Speedups that are really just TF32 |
| Interleaved contenders, cold L2, a GPU lock, clock locking, an environment fingerprint | Drift, contention between agents sharing one GPU, irreproducible setups |

Every defence has a red-team case in `redteam/` that must be caught, plus two honest kernels that
must pass. See [docs/methodology.md](docs/methodology.md).

## Quick start

```bash
# 0. PyTorch for your driver first (example: CUDA 12.8 wheels), then this repo
pip install torch --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/ScalingIntelligence/KernelBench ../KernelBench
git -C ../KernelBench checkout 423217d9fda91e0c2d67e4a43bf62f96f6d104f1   # pinned in configs/rig.yaml
pip install -e ".[gpu,plot,dev]"

# 1. Rig
sudo scripts/lock_clocks.sh 0 2400      # pick a clock your card sustains; see the script header
kb4090 env                              # fingerprint + warnings (WSL, display attached, unverified specs...)

# 2. Prove the harness before trusting it
kb4090 redteam                          # every hack caught, both positive controls pass
pytest -m gpu -s                        # same thing, as a test

# 3. Phase 1 data
kb4090 baseline --tasks all             # eager FP32/TF32 timings, memory, SOL per task
kb4090 noise --tasks all                # A/A test: the smallest speedup you can detect
kb4090 tables                           # results/tables/*.md and *.csv

# 4. Agents at a matched budget (3 runs each, median reported)
kb4090 run-agent --agent claude_code --tasks level1 --runs 3 --budget 10
kb4090 run-agent --agent dsh         --tasks level1 --runs 3 --budget 10
kb4090 tables                           # + results/charts/cost_vs_speedup.png

# 5. EAT on the same tasks
kb4090 prepare-eat --tasks L2-076-gemm-add-relu,L2-099-matmul-gelu-smax --out ../eat-run1 --budget 60
#   ...launch EAT as its SKILL.md describes, then:
kb4090 eat-final --repo ../eat-run1 --cost-usd <from eat/log/BUDGET.md> --budget 60
```

Evaluate a single kernel by hand: `kb4090 eval --task L1-019-relu --candidate my_relu.py`.

## Verdicts

| Verdict | Meaning | Counts as |
|---|---|---|
| `PASS` | correct on every seed and distribution, own kernel ran, timing clean | its speedup |
| `SUSPECT` | faster than the speed-of-light bound | 1.0 until a human review (`kb4090 review`) |
| `FAIL_COMPILE` / `FAIL_RUNTIME` / `FAIL_INCORRECT` / `FAIL_TIMEOUT` | what it says | 1.0 (you keep the reference) |
| `INVALID` | a cheating pattern was detected | 1.0, and worth a look |
| `HARNESS_ERROR` | our fault or the task's | excluded |

Speedups are reported against **PyTorch eager with TF32 enabled**, per task as the **median of
runs** (best-of-n is shown separately and labelled). Across tasks: geometric mean and KernelBench's
`fast_p` computed on medians.

## Layout

```
kb4090/            harness: static_check, distributions, compare, timing, runtime_checks, sol, evaluate,
                   runner (subprocess per eval), agents, eat, summarise, cli
configs/           rig.yaml (all measurement settings), tasks.yaml (the subset), agents.yaml, rig.eat.yaml
hardware/          rtx4090.yaml (spec-sheet peaks, marked unverified until you check them)
redteam/           tasks/, candidates/ (2 positive controls, 13 hacks), expected.yaml
scripts/           lock_clocks.sh, unlock_clocks.sh, sandbox.sh (bubblewrap)
docs/              methodology, roadmap, post plan, DeepSeek Harness supervisor design note
results/ledger/    append-only JSONL: evals, agent runs, baselines, noise, reviews, red-team
```

## Safety

Agents run headless with permission prompts disabled, and candidate kernels are arbitrary code.
`run-agent` refuses to start without a sandbox unless you pass `--no-sandbox-i-understand`. Use
`scripts/sandbox.sh` (bubblewrap) or, better, a disposable VM/container with no credentials
besides the one API key the agent needs. Evaluation workers can run with no network
(`sandbox_prefix` in `configs/rig.yaml`).

## Known limitations

- **Hardware peaks are unverified** (`hardware/rtx4090.yaml`): check them against NVIDIA's Ada
  whitepaper before publishing speed-of-light or "% of SOL" numbers.
- **DeepSeek Harness is a developer preview.** Its headless JSON event schema and whether its
  usage numbers are per-step or cumulative must be checked on one run (`configs/agents.yaml`).
  Budgets are enforced by our token meter, priced with numbers you fill in.
- **Costs are estimates**: Claude Code's `total_cost_usd` is a client-side estimate; check the
  provider's billing for totals you publish.
- **EAT drives Claude Code only**; its in-loop score on one shared GPU is noisy. Publish the
  `eat-final` numbers, which rerun the full evaluation on an idle GPU.
- **The speed-of-light byte count can be too high** for inputs with exploitable structure
  (e.g. triangular matrices). It only ever produces `SUSPECT`, never `INVALID`; tune `sol_margin`
  per task.
- KernelBench's fp32 tolerance (`atol = rtol = 1e-2`) is kept for comparability, with an extra
  relative-L2 check.

## Credits

- KernelBench (Ouyang, Guo et al., Stanford Scaling Intelligence Lab), MIT. Its adversarial
  kernels and static checker are used directly from your checkout, not copied.
- KernelBench-Verified (facebookresearch/kernel_bench_verified): the TF32 baseline and
  multi-distribution idea.
- EAT, Evolving Agent Teams (Cao, Mao, Li, Gonzalez, Stoica), Apache-2.0.
- The measurement method follows the Sybil Solutions inference-optimisation skill: compute the
  ceiling, change one thing at a time, use interleaved controls, log every result including failures.

MIT licence.
