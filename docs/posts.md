# Post plan

Each post: one claim, one chart or table, the caveat, a link to the exact commit. Tag authors only
when you have something specific for them.

| # | Claim to test | Artefacts | Command |
|---|---|---|---|
| 1 | Where the 4090 sits against the roofline for 24 KernelBench tasks; how much TF32 buys on GeForce Ada | `results/tables/baselines.md`, `noise.md` | `kb4090 baseline --tasks all && kb4090 noise --tasks all && kb4090 tables` |
| 2 | 13 known ways to fake a KernelBench speedup, and the check that catches each | `results/tables/redteam.md` | `kb4090 redteam` |
| 3 | Verified speedup per dollar: Claude Code vs DeepSeek Harness, matched budget, median of 3 | `results/tables/agents.md`, `results/charts/cost_vs_speedup.png` | `kb4090 run-agent ...; kb4090 tables` |
| 4 | Does a supervisor-evolved team beat a single agent at the same budget on Ada? | the above plus EAT's `eat/log/` team configs | `kb4090 eat-final ...; kb4090 tables` |
| 5 | Full write-up | repo + ledger + transcripts | - |

## Checklist before posting any number

- [ ] Red-team suite passes at this commit (link the table).
- [ ] Hardware peaks verified, if the post shows SOL or % of SOL.
- [ ] Clocks locked; the env fingerprint is linked.
- [ ] The speedup's CI lower bound is above the A/A noise floor for that task.
- [ ] Headline vs eager **TF32**, median of runs; best-of-n only if labelled.
- [ ] Every SUSPECT reviewed, with a note.
- [ ] Costs labelled as estimates; budgets identical across systems.
- [ ] You can explain *why* the best kernel is faster (Nsight Compute: memory throughput,
      occupancy, instruction mix).
- [ ] Failures and INVALIDs reported, not hidden.

## Things that make posts credible

- Show a failure next to every win.
- Put the noise floor on the chart.
- "Here's how I could be wrong" beats "here's how great this is".
- Release the transcripts. People learn most from watching where agents go wrong.
