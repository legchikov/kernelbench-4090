"""Command-line interface: ``kb4090 <command>``. Run ``kb4090 -h`` for the list."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import ledger
from .config import (
    REPO_ROOT,
    config_hash,
    find_task,
    load_agents,
    load_hardware,
    load_rig,
    load_tasks,
    resolve_path,
)


def _ctx(args: argparse.Namespace) -> tuple[dict, dict, list[dict]]:
    rig = load_rig(getattr(args, "rig", None))
    if getattr(args, "gpu", None) is not None:
        rig["gpu_index"] = args.gpu
    hw = load_hardware(rig["hardware"])
    tasks = load_tasks(getattr(args, "tasks_file", None))
    return rig, hw, tasks


def _results_dir(rig: dict) -> Path:
    return resolve_path(rig["results_dir"])


def _select_tasks(ref: str | None, tasks: list[dict], rig: dict) -> list[dict]:
    if not ref or ref == "all":
        return [find_task(t["id"], tasks, rig) for t in tasks if t.get("enabled", True)]
    if ref.startswith("level"):
        lvl = int(ref.replace("level", ""))
        return [find_task(t["id"], tasks, rig) for t in tasks if t.get("level") == lvl and t.get("enabled", True)]
    return [find_task(x.strip(), tasks, rig) for x in ref.split(",") if x.strip()]


def _env_stamp(rig: dict, hw: dict) -> dict:
    from .env import fingerprint

    fp = fingerprint(rig, with_torch=False)
    return {"env_id": fp["env_id"], "rig_hash": config_hash({k: v for k, v in rig.items() if k != "gpu_index"}),
            "hw": hw.get("name"), "kb4090_commit": fp["kb4090"]["commit"], "kb4090_dirty": fp["kb4090"]["dirty"]}


def _print_eval(res: dict, verbose: bool = False) -> None:
    print(f"verdict: {res.get('verdict')}")
    for r in res.get("reasons", []):
        print(f"  - {r['verdict']}: {r['reason']}")
    for w in res.get("warnings", [])[:10]:
        print(f"  ! {w}")
    sp = res.get("speedups") or {}
    for name, ci in sp.items():
        print(f"  speedup vs {name}: {ci['point']:.3f}x  (95% CI {ci['lo']:.3f}-{ci['hi']:.3f})")
    timing = (res.get("timing") or {}).get("contenders", {})
    for name, c in timing.items():
        s = c["summary"]
        print(f"  {name:>11}: median {s['median']:.4f} ms  (IQR {s['iqr']:.4f}, n={s['n']})")
    if res.get("pct_of_sol") is not None:
        print(f"  at {res['pct_of_sol']:.1%} of the speed-of-light bound (TF32 peak / DRAM bandwidth)")
    k = res.get("kernels")
    if k and k.get("profiler_available"):
        print(f"  custom kernels: {k['n_custom']} ({k['custom_fraction']:.0%} of GPU time)")
    if verbose and res.get("log_tail"):
        print("--- worker log (tail) ---\n" + res["log_tail"])


# ---------------------------------------------------------------------------------------------
def cmd_env(args: argparse.Namespace) -> int:
    from .env import fingerprint, warnings_for

    rig, hw, _ = _ctx(args)
    fp = fingerprint(rig)
    print(json.dumps(fp, indent=2, default=str))
    warns = warnings_for(fp, hw)
    for w in warns:
        print(f"WARNING: {w}", file=sys.stderr)
    out = _results_dir(rig) / "env" / f"{fp['env_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**fp, "warnings": warns}, indent=2, default=str))
    print(f"saved {out}", file=sys.stderr)
    return 0


def cmd_eval(args: argparse.Namespace, mode: str = "official") -> int:
    from .runner import run_worker

    rig, hw, tasks = _ctx(args)
    task = find_task(args.task, tasks, rig)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if getattr(args, "out", None) else \
        _results_dir(rig) / "evals" / task["id"] / f"{stamp}-{Path(args.candidate).stem}.json"
    if mode == "dev":
        out = Path(args.candidate).resolve().parent / ".dev_eval" / f"{stamp}.json"
    res = run_worker(mode, task, rig, hw, out, candidate=args.candidate)
    _print_eval(res, verbose=getattr(args, "verbose", False) or res.get("verdict") in {"FAIL_COMPILE", "FAIL_RUNTIME"})
    if mode == "official" and not args.no_ledger:
        ledger.append(_results_dir(rig), "eval", {**res, **_env_stamp(rig, hw), "label": args.label})
    print(f"full result: {out}")
    return 0 if res.get("verdict") == "PASS" else 1


def cmd_baseline(args: argparse.Namespace, mode: str = "baseline") -> int:
    from .runner import run_worker

    rig, hw, tasks = _ctx(args)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if mode == "baseline" and args.compile:
        rig = {**rig, "baselines": list(dict.fromkeys([*rig["baselines"], "compile"]))}
    for task in _select_tasks(args.tasks, tasks, rig):
        out = _results_dir(rig) / mode / stamp / f"{task['id']}.json"
        print(f"[{mode}] {task['id']} ...", flush=True)
        res = run_worker(mode, task, rig, hw, out)
        ledger.append(_results_dir(rig), mode, {**res, **_env_stamp(rig, hw)})
        if mode == "baseline":
            meds = res.get("medians_ms") or {}
            print(f"    {res.get('verdict')}: " + ", ".join(f"{k} {v:.3f} ms" for k, v in meds.items())
                  + (f", peak mem {res['peak_mem_gb']:.1f} GB" if res.get("peak_mem_gb") else ""))
        else:
            print(f"    A/A ratio {res.get('aa_ratio', {}).get('point', float('nan')):.4f}, "
                  f"min detectable speedup {res.get('min_detectable_speedup', float('nan')):.3f}x")
    from .summarise import write_all

    print(json.dumps(write_all(_results_dir(rig)), indent=1))
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    from .summarise import write_all

    rig, _, _ = _ctx(args)
    print(json.dumps(write_all(_results_dir(rig)), indent=1))
    return 0


def cmd_redteam(args: argparse.Namespace) -> int:
    from .config import load_yaml
    from .runner import run_worker

    rig, hw, tasks = _ctx(args)
    spec = load_yaml(REPO_ROOT / "redteam" / "expected.yaml")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows, misses = [], 0
    from .config import kernelbench_root

    kb_root = str(kernelbench_root(rig) or "")
    for case in spec["cases"]:
        if args.only and args.only not in case["candidate"]:
            continue
        cand = resolve_path(case["candidate"].replace("{kb_root}", kb_root))
        task_ref = case["task"].replace("{kb_root}", kb_root)
        if not cand.exists():
            print(f"skip {case['candidate']}: not found (check kernelbench_root in configs/rig.yaml)")
            continue
        try:
            task = find_task(str(resolve_path(task_ref)) if "/" in task_ref else task_ref, tasks, rig)
        except KeyError as exc:
            print(f"skip {case['candidate']}: {exc}")
            continue
        res = run_worker("official", task, rig, hw, _results_dir(rig) / "redteam" / stamp / f"{cand.stem}.json",
                         candidate=cand)
        got = res.get("verdict")
        reasons = " | ".join(r["reason"] for r in res.get("reasons", []))[:200]
        ok = got in case["expect"]
        if ok and case.get("expect_warning"):
            ok = any(case["expect_warning"] in w for w in res.get("warnings", []))
            reasons = reasons or next((w for w in res.get("warnings", []) if case["expect_warning"] in w), "")
        misses += not ok
        rows.append({"candidate": cand.stem, "what": case["what"], "expected": "/".join(case["expect"]),
                     "got": got, "caught": "yes" if ok else "**NO**", "why": reasons})
        print(f"{'ok  ' if ok else 'MISS'} {cand.stem:<34} expected {'/'.join(case['expect']):<28} got {got}")
        ledger.append(_results_dir(rig), "redteam", {**res, **_env_stamp(rig, hw), "case": case})
    from .summarise import markdown

    md = "# Red-team results\n\nEach row is a candidate kernel that tries a known trick. 'caught' means the " \
         "harness gave the expected verdict.\n\n" + markdown(rows, ["candidate", "what", "expected", "got", "caught", "why"])
    out = _results_dir(rig) / "tables" / "redteam.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"\n{len(rows) - misses}/{len(rows)} behaved as expected. Table: {out}")
    return 0 if misses == 0 else 1


def cmd_run_agent(args: argparse.Namespace) -> int:
    from .agents import launch, prepare_workspace, snapshot_solution
    from .runner import run_worker

    rig, hw, tasks = _ctx(args)
    agents = load_agents(args.agents_file)
    if args.agent not in agents.get("agents", {}):
        print(f"unknown agent {args.agent!r}; configured: {sorted(agents.get('agents', {}))}", file=sys.stderr)
        return 2
    cfg = {**agents.get("defaults", {}), **agents["agents"][args.agent]}
    if args.model:
        cfg["model"] = args.model
    budget = float(args.budget if args.budget is not None else cfg.get("budget_usd", 10.0))
    wall = float(args.wall_clock_min * 60 if args.wall_clock_min else cfg.get("wall_clock_s", 3600))
    if not cfg.get("sandbox_prefix") and not args.no_sandbox_i_understand:
        print("Refusing to start: this agent runs with permission prompts disabled and no sandbox_prefix is "
              "configured in configs/agents.yaml. Configure scripts/sandbox.sh (recommended), run inside a "
              "disposable VM/container, or pass --no-sandbox-i-understand.", file=sys.stderr)
        return 2
    for task in _select_tasks(args.tasks, tasks, rig):
        for k in range(args.runs):
            run_dir = resolve_path(rig["runs_dir"]) / args.agent / task["id"] / f"{time.strftime('%Y%m%d-%H%M%S')}-run{k}"
            prepare_workspace(task, run_dir, budget, args.agent, hw, wall)
            print(f"[{args.agent}] {task['id']} run {k}: {run_dir}", flush=True)
            meta = launch(args.agent, cfg, run_dir, budget, wall, rig)
            (run_dir / "agent_meta.json").write_text(json.dumps(meta, indent=1, default=str))
            final = snapshot_solution(run_dir)
            res = run_worker("official", task, rig, hw, run_dir / "official_eval.json", candidate=final)
            row = {**meta, "task_id": task["id"], "run_idx": k, "run_dir": str(run_dir), "verdict": res.get("verdict"),
                   "headline_speedup": res.get("headline_speedup"), "speedups": res.get("speedups"),
                   "reasons": res.get("reasons"), "pct_of_sol": res.get("pct_of_sol"),
                   "candidate_sha256": res.get("candidate_sha256"), **_env_stamp(rig, hw)}
            ledger.append(_results_dir(rig), "agent_run", row)
            cost = f"${meta['cost_usd']:.2f}" if meta.get("cost_usd") is not None else "cost unknown"
            sp = f"{res['headline_speedup']:.3f}x" if res.get("headline_speedup") else "-"
            print(f"    exit={meta['exit_reason']} {cost} -> {res.get('verdict')} {sp}")
    return 0


def cmd_prepare_eat(args: argparse.Namespace) -> int:
    from .eat import prepare_eat_repo

    rig, hw, tasks = _ctx(args)
    selected = _select_tasks(args.tasks, tasks, rig)
    out = prepare_eat_repo(selected, Path(args.out).expanduser().resolve(), rig, hw, args.budget)
    print(f"EAT task repo written to {out}\nNext:\n  cd {out}\n  # edit eat/env.sh to activate your venv\n"
          "  git init && git add -A && git commit -m 'EAT task setup'\n"
          "  bash eat/eval.sh   # run 2-3 times to see the noise before launching\n"
          "  # then follow EAT's SKILL.md to launch (EAT_GPUS is set in eat/env.sh)")
    return 0


def cmd_eat_score(args: argparse.Namespace) -> int:
    from .eat import eat_score

    rig, hw, _ = _ctx(args)
    res = eat_score(Path(args.repo).resolve(), rig, hw, verbose_hidden=args.verbose_hidden)
    for line in res["lines"]:
        print(line)
    print(f"SCORE: {res['score']:.4f}")
    return 0


def cmd_eat_final(args: argparse.Namespace) -> int:
    import os

    from .runner import run_worker

    os.environ.pop("KB4090_RIG_OVERLAY", None)
    rig, hw, _ = _ctx(args)
    repo = Path(args.repo).resolve()
    meta = json.loads((repo / "eat" / "tasks.json").read_text())
    n = len(meta)
    out_dir = repo / ".work" / "kb4090-final" / time.strftime("%Y%m%d-%H%M%S")
    for task_id, m in meta.items():
        sol = repo / "solutions" / f"{task_id}.py"
        task = {"id": task_id, "abs_path": m["task_path"], "perturb": m.get("perturb", True),
                "sol_margin": m.get("sol_margin")}
        res = run_worker("official", task, rig, hw, out_dir / f"{task_id}.json", candidate=sol)
        row = {"agent": "eat", "model": args.model, "task_id": task_id, "run_idx": args.run_idx,
               "run_dir": str(repo), "verdict": res.get("verdict"), "headline_speedup": res.get("headline_speedup"),
               "speedups": res.get("speedups"), "reasons": res.get("reasons"), "pct_of_sol": res.get("pct_of_sol"),
               "cost_usd": (args.cost_usd / n) if args.cost_usd is not None else None,
               "cost_source": "eat_budget_md_split_evenly" if args.cost_usd is not None else None,
               "budget_usd": (args.budget / n) if args.budget else None, **_env_stamp(rig, hw)}
        ledger.append(_results_dir(rig), "agent_run", row)
        print(f"{task_id}: {res.get('verdict')} {res.get('headline_speedup')}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    rig, _, _ = _ctx(args)
    ledger.append(_results_dir(rig), "review", {"target_row_id": args.row_id, "new_verdict": args.verdict,
                                                "note": args.note})
    print("review recorded")
    return 0


# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kb4090", description="Hardened KernelBench experiments on an RTX 4090")
    ap.add_argument("--rig", help="rig config (default configs/rig.yaml or $KB4090_RIG)")
    ap.add_argument("--tasks-file", help="task list (default configs/tasks.yaml)")
    ap.add_argument("--gpu", type=int, help="override gpu_index")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="print and save the environment fingerprint, with warnings")

    p = sub.add_parser("eval", help="official evaluation of one candidate (appends to the ledger)")
    p.add_argument("--task", required=True, help="task id from configs/tasks.yaml or a path")
    p.add_argument("--candidate", required=True)
    p.add_argument("--out")
    p.add_argument("--label", default=None)
    p.add_argument("--no-ledger", action="store_true")
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("dev-eval", help="quick feedback for agents (base inputs only, short timing)")
    p.add_argument("--task", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--verbose", action="store_true")

    for name, help_ in [("baseline", "time PyTorch eager FP32/TF32 per task, plus speed of light"),
                        ("noise", "A/A noise-floor test per task")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--tasks", default="all", help="'all', 'level1', or comma-separated ids")
        if name == "baseline":
            p.add_argument("--compile", action="store_true", help="also time torch.compile (slow)")

    sub.add_parser("tables", help="rebuild Markdown/CSV tables and the chart from the ledger")

    p = sub.add_parser("redteam", help="run the red-team suite: every known hack must be caught")
    p.add_argument("--only", help="substring filter on candidate path")

    p = sub.add_parser("run-agent", help="run a coding agent on tasks at a fixed budget, then evaluate")
    p.add_argument("--agent", required=True, help="name from configs/agents.yaml")
    p.add_argument("--tasks", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--budget", type=float, help="USD per run (default from agents.yaml)")
    p.add_argument("--model")
    p.add_argument("--wall-clock-min", type=float)
    p.add_argument("--agents-file")
    p.add_argument("--no-sandbox-i-understand", action="store_true")

    p = sub.add_parser("prepare-eat", help="write an EAT task repository for a set of tasks")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=200.0, help="EAT budget you plan to use (for the docs)")

    p = sub.add_parser("eat-score", help="(used by eat/eval.sh) score an EAT repo; prints SCORE: <x>")
    p.add_argument("--repo", required=True)
    p.add_argument("--verbose-hidden", action="store_true", help="show which hidden distribution failed")

    p = sub.add_parser("eat-final", help="full official evaluation of an EAT repo's final solutions")
    p.add_argument("--repo", required=True)
    p.add_argument("--cost-usd", type=float, help="total spend from eat/log/BUDGET.md")
    p.add_argument("--budget", type=float)
    p.add_argument("--model", default="opus")
    p.add_argument("--run-idx", type=int, default=0)

    p = sub.add_parser("review", help="record a human review of a ledger row (e.g. SUSPECT -> PASS)")
    p.add_argument("--row-id", required=True)
    p.add_argument("--verdict", required=True, choices=["PASS", "INVALID", "FAIL_INCORRECT", "SUSPECT"])
    p.add_argument("--note", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "env": cmd_env,
        "eval": cmd_eval,
        "dev-eval": lambda a: cmd_eval(a, mode="dev"),
        "baseline": cmd_baseline,
        "noise": lambda a: cmd_baseline(a, mode="noise"),
        "tables": cmd_tables,
        "redteam": cmd_redteam,
        "run-agent": cmd_run_agent,
        "prepare-eat": cmd_prepare_eat,
        "eat-score": cmd_eat_score,
        "eat-final": cmd_eat_final,
        "review": cmd_review,
    }
    if args.cmd == "dev-eval":
        args.no_ledger, args.label, args.out = True, None, None
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
