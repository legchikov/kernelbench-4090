"""Bridge to EAT (Evolving Agent Teams, github.com/caoshiyi/EAT).

EAT optimises "a program against a fixed eval". Its SKILL.md asks the target repository for:
  eat/task.md  rules injected into every agent prompt; must start with
               "## Hard Requirements (MUST be in every agent's prompt)" then "## Shared Environment"
  eat/eval.sh  official eval of the committed state; prints "SCORE: <x>" (higher is better) and
               returns 0 on any correctness failure
  eat/env.sh   optional, sourced first (EAT_GPUS etc.)
plus .work/ and eat/log/.supervisor_* in .gitignore, then a git commit.

``prepare_eat_repo`` writes all of that for a set of KernelBench tasks. ``eat_score`` is what
eat/eval.sh runs: a geometric mean of verified speedups over the tasks, where an unchanged stub
counts as 1.0 and any FAIL/INVALID/SUSPECT makes the whole score 0. Hidden-distribution details are
not printed, so the team gets a verdict but not a map of the held-out tests.

During a run the in-loop scores are noisy: several agents share one GPU. The authoritative number
is ``kb4090 eat-final`` after the run, with the full official evaluation on an idle GPU.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .agents import _hardware_facts, stub_from_reference
from .config import PACKAGE_ROOT, REPO_ROOT
from .stats import geomean


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_eat_repo(tasks: list[dict], out_dir: Path, rig: dict, hw: dict, budget_usd: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "eat").mkdir()
    sol_dir = out_dir / "solutions"
    sol_dir.mkdir()
    stubs = {}
    task_lines = []
    for t in tasks:
        src = Path(t["abs_path"]).read_text(encoding="utf-8")
        sol = sol_dir / f"{t['id']}.py"
        sol.write_text(stub_from_reference(src))
        stubs[t["id"]] = {"stub_sha256": _sha(sol), "task_path": t["abs_path"], "perturb": t.get("perturb", True),
                          "sol_margin": t.get("sol_margin")}
        task_lines.append(f"- `{t['id']}`: reference `{t['abs_path']}` -> edit `solutions/{t['id']}.py`")
    (out_dir / "eat" / "tasks.json").write_text(json.dumps(stubs, indent=1))

    tmpl = PACKAGE_ROOT / "templates"
    gpu = str(rig["gpu_index"])
    values = {
        "{{TASK_LIST}}": "\n".join(task_lines),
        "{{N_TASKS}}": str(len(tasks)),
        "{{GPU}}": gpu,
        "{{HARDWARE}}": _hardware_facts(hw),
        "{{BUDGET}}": f"{budget_usd:.0f}",
        "{{RIG}}": str(rig.get("_source", REPO_ROOT / "configs" / "rig.yaml")),
        "{{OVERLAY}}": str(REPO_ROOT / "configs" / "rig.eat.yaml"),
        "{{FIRST_TASK_PATH}}": tasks[0]["abs_path"] if tasks else "",
        "{{FIRST_TASK_ID}}": tasks[0]["id"] if tasks else "",
    }
    for name, dest in [("eat_task.md", "eat/task.md"), ("eat_eval.sh", "eat/eval.sh"),
                       ("eat_env.sh", "eat/env.sh"), ("eat_claude.md", "CLAUDE.md"),
                       ("eat_gitignore", ".gitignore")]:
        text = (tmpl / name).read_text(encoding="utf-8")
        for k, v in values.items():
            text = text.replace(k, v)
        (out_dir / dest).write_text(text)
    (out_dir / "eat" / "eval.sh").chmod(0o755)
    return out_dir


def eat_score(repo: Path, rig: dict, hw: dict, mode: str = "official", verbose_hidden: bool = False) -> dict:
    """Score every solution; used by eat/eval.sh. Returns a dict with 'score' and per-task lines."""
    from .runner import run_worker

    meta = json.loads((repo / "eat" / "tasks.json").read_text())
    out_root = repo / ".work" / "kb4090" / time.strftime("%Y%m%d-%H%M%S")
    lines, speedups, zero = [], [], False
    results = {}
    for task_id, m in meta.items():
        sol = repo / "solutions" / f"{task_id}.py"
        if not sol.exists():
            lines.append(f"task {task_id}: MISSING solutions/{task_id}.py")
            zero = True
            continue
        if _sha(sol) == m["stub_sha256"]:
            lines.append(f"task {task_id}: UNCHANGED (counts as 1.000x)")
            speedups.append(1.0)
            continue
        task = {"id": task_id, "abs_path": m["task_path"], "perturb": m.get("perturb", True),
                "sol_margin": m.get("sol_margin")}
        res = run_worker(mode, task, rig, hw, out_root / f"{task_id}.json", candidate=sol)
        results[task_id] = res
        verdict = res.get("verdict")
        if verdict == "PASS" and res.get("headline_speedup"):
            s = float(res["headline_speedup"])
            speedups.append(s)
            lines.append(f"task {task_id}: PASS {s:.3f}x vs {res.get('headline_baseline')}")
        else:
            zero = True
            reason = (res.get("reasons") or [{}])[0].get("reason", "")
            if not verbose_hidden and verdict == "FAIL_INCORRECT":
                reason = "wrong output on at least one held-out input"
            lines.append(f"task {task_id}: {verdict} - {reason[:300]}")
    score = 0.0 if zero or not speedups else geomean(speedups)
    return {"score": score, "lines": lines, "results": results, "out_dir": str(out_root)}
