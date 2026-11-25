# Design note: an EAT-style supervisor on DeepSeek Harness (Phase 5, not implemented)

EAT runs its teams on Claude Code. DeepSeek Harness (`dsh`) is MIT-licensed and built so that the
model adapter, tool registry, sandbox, scheduler and agent loop are all plugins, and it can call
other coding agents as sub-agents. That makes it a natural place for an open supervisor loop.
This note is a plan, not code: the dsh plugin API is in developer preview and will change, so
read its docs at the version you pin before starting.

## What to port

EAT's loop, per its README and SKILL.md:

1. The supervisor reflects on the last team's output and the shared learnings file.
2. It emits a team configuration: roles, prompts, phases, and file/GPU ownership.
3. The harness launches the team phase by phase, runs the official eval, and records cost and score.
4. Repeat while budget remains. Each iteration ends with a commit carrying the score.

## Mapping onto dsh

| EAT concept | dsh piece |
|---|---|
| supervisor agent | a dsh session with a "supervisor" system prompt and read access to `eat/log/` |
| team configuration | JSON emitted by the supervisor and validated against a schema we define |
| team member | a dsh sub-session per role (or Claude Code/Codex as a sub-agent, for mixed teams) |
| official eval | a tool plugin wrapping `bash eat/eval.sh` (unchanged: reuse `kb4090 prepare-eat`) |
| budget | a scheduler plugin that meters `step_end` usage and stops spawning at the budget |
| shared file system | the same repo layout EAT uses, so runs are comparable |

## Experiments it enables

- Same supervisor prompt, different worker models (DeepSeek V4 vs Claude) at a matched budget.
- Open-weight workers served locally. The 4090 is busy benchmarking, so serve them on another machine.

## Risks

- Breaking changes in dsh: pin the version and record it in the env fingerprint.
- Comparability with EAT: keep its prompts and team-configuration schema as close as possible,
  and document every difference.
