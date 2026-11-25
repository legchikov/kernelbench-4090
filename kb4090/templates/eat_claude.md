# Read this first

The rules for this repository are in `eat/task.md`. They are mandatory for every agent.

- The score comes only from `bash eat/eval.sh`. Never edit anything in `eat/`.
- Measure performance only with `bash eat/eval.sh` or `kb4090 dev-eval`; both take a GPU lock.
- Work in your own `work_<role>/` copy; only ship a correct, dev-eval-passing file to `solutions/`.
- A broken file in `solutions/` makes the team's score 0.
