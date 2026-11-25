# Roadmap (about 10 weeks at 8-10 hours a week)

Tick boxes as you go. Each phase ends with something you can post.

## Phase 1: rig and baselines (weeks 1-2)

- [ ] Native Linux if at all possible (WSL limits CUPTI/Nsight). Note the GPU, driver, CUDA, PyTorch.
- [ ] Check every number in `hardware/rtx4090.yaml` against NVIDIA's Ada whitepaper and
      `torch.cuda.get_device_properties(0)`. Set `verified: true` and fill in `source`.
- [ ] Find a sustainable clock (`nvidia-smi -q -d CLOCK,PERFORMANCE` under load) and lock it.
- [ ] `kb4090 env`: clear every warning you can.
- [ ] `kb4090 redteam`: all cases behave as expected. Fix the harness before anything else if not.
- [ ] `kb4090 baseline --tasks all`, `kb4090 noise --tasks all`, `kb4090 tables`.
- [ ] Prune `configs/tasks.yaml`: drop tasks that OOM in 24 GB or sit far outside the 1-15 ms window,
      and write down why. **Freeze the list** (commit it and tag `tasks-v1`).
- **Post 1:** the speed-of-light table: which KernelBench tasks are compute-bound vs memory-bound
  on a 4090, how close PyTorch already is, and how much TF32 buys on GeForce Ada (expected to be
  small, but measure it).

## Phase 2: the harness (weeks 2-3)

- [ ] Write one more hack of your own that the suite doesn't catch yet, then add the defence and
      the case. This is the most valuable thing you can contribute.
- [ ] Run `kb4090 eval` on a hand-written kernel for one task and profile it with `ncu`. You should
      be able to explain every kernel you post.
- **Post 2:** "I tried to cheat my own benchmark", with the red-team table, one hack explained
  in depth, and why each defence exists.

## Phase 3: single agents at a matched budget (weeks 4-5)

- [ ] Fill in DeepSeek prices in `configs/agents.yaml`. On one short run, check whether `step_end`
      usage is per-step or cumulative, and set `usage_is_cumulative`.
- [ ] Set up the sandbox (`scripts/sandbox.sh` or a VM). Test that it blocks what it should.
- [ ] Level 1 first: `--runs 3 --budget <same for both>` for Claude Code and dsh.
- [ ] Read every INVALID and SUSPECT transcript. They are your best material.
- **Post 3:** dollars per verified speedup, open harness vs closed, with medians and failure
  modes, not just the best run.

## Phase 4: EAT (weeks 6-8)

- [ ] Pick the 5-8 tasks where single agents plateaued (Level 2/3).
- [ ] `kb4090 prepare-eat`, review `eat/task.md` yourself, run `bash eat/eval.sh` three times to see
      the noise, then launch EAT as its SKILL.md says, with `EAT_GPUS` set to your one GPU.
- [ ] Watch the supervisor's team configurations in `eat/log/`: which roles does it invent on Ada?
- [ ] `kb4090 eat-final` on an idle GPU, with the cost from `eat/log/BUDGET.md`.
- **Post 4:** does the team effect from the paper replicate on a different benchmark and a
  consumer GPU, at a matched budget, over several runs?

## Phase 5: stretch (weeks 9-10)

- [ ] Port the supervisor loop to DeepSeek Harness as a plugin (see `dsh-supervisor-design.md`).
- [ ] Upstream contributions: 4090 baseline timings to KernelBench; an issue or PR to EAT on
      single-GPU operation; your new hack and defence to KernelBench's adversarial tests.
- **Post 5:** the write-up, with the repo, the ledger, transcripts and every kernel.
