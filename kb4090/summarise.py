"""Turn the ledger into tables (Markdown + CSV) and one chart for posts.

Conventions (state them in every post):
  * Per run, the *effective speedup* is the headline speedup (vs PyTorch eager TF32) if the official
    verdict is PASS, else 1.0: a failed or invalid kernel means you keep the reference.
  * SUSPECT results count as 1.0 until a human review row upgrades them (``kb4090 review``).
  * Per task we report the MEDIAN over runs (and the best, labelled as such).
  * Across tasks: geometric mean of per-task medians, and fast_p = share of tasks whose median
    run is a PASS with speedup > p (KernelBench's fast_p, computed on medians, not best-of-n).
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from . import ledger
from .stats import geomean

FAST_P = (1.0, 1.2, 1.5, 2.0)


def _apply_reviews(rows: list[dict], reviews: list[dict]) -> list[dict]:
    by_target = {r["target_row_id"]: r for r in reviews}
    out = []
    for row in rows:
        rv = by_target.get(row.get("row_id"))
        if rv:
            row = {**row, "verdict": rv["new_verdict"], "review_note": rv.get("note")}
        out.append(row)
    return out


def effective_speedup(row: dict) -> float:
    if row.get("verdict") == "PASS" and row.get("headline_speedup"):
        return float(row["headline_speedup"])
    return 1.0


def agent_rows(results_dir: str | Path) -> list[dict]:
    rows = list(ledger.read(results_dir, "agent_run"))
    return _apply_reviews(rows, list(ledger.read(results_dir, "review")))


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    per_task: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        per_task[(r.get("agent"), r.get("model"), r.get("task_id"))].append(r)
    task_table = []
    for (agent, model, task), runs in sorted(per_task.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        eff = [effective_speedup(r) for r in runs]
        costs = [r["cost_usd"] for r in runs if r.get("cost_usd") is not None]
        verdicts = [r.get("verdict") for r in runs]
        median_run = sorted(runs, key=effective_speedup)[len(runs) // 2]
        task_table.append({
            "agent": agent, "model": model, "task_id": task, "runs": len(runs),
            "passed": sum(v == "PASS" for v in verdicts),
            "suspect": sum(v == "SUSPECT" for v in verdicts),
            "median_speedup": statistics.median(eff),
            "best_speedup": max(eff),
            "median_is_pass": median_run.get("verdict") == "PASS",
            "median_cost_usd": statistics.median(costs) if costs else None,
            "total_cost_usd": sum(costs) if costs else None,
            "verdicts": ",".join(sorted(set(str(v) for v in verdicts))),
        })
    per_agent: dict[tuple, list[dict]] = defaultdict(list)
    for t in task_table:
        per_agent[(t["agent"], t["model"])].append(t)
    agent_table = []
    for (agent, model), tasks in sorted(per_agent.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        meds = [t["median_speedup"] for t in tasks]
        runs = sum(t["runs"] for t in tasks)
        costs = [t["total_cost_usd"] for t in tasks if t["total_cost_usd"] is not None]
        row = {
            "agent": agent, "model": model, "tasks": len(tasks), "runs": runs,
            "geomean_speedup": geomean(meds),
            "pass_rate": sum(t["passed"] for t in tasks) / runs if runs else 0.0,
            "total_cost_usd": sum(costs) if costs else None,
            "cost_per_task_usd": (sum(costs) / len(tasks)) if costs else None,
        }
        for p in FAST_P:
            row[f"fast_{p:g}"] = sum(t["median_is_pass"] and t["median_speedup"] > p for t in tasks) / len(tasks)
        agent_table.append(row)
    return task_table, agent_table


def _fmt(v: object) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def markdown(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_no data yet_\n"
    head = "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n"
    return head + "".join("| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |\n" for r in rows)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def latest_by_task(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for r in rows:
        if r.get("task_id") and (r["task_id"] not in latest or r.get("recorded", 0) > latest[r["task_id"]].get("recorded", 0)):
            latest[r["task_id"]] = r
    return [latest[k] for k in sorted(latest)]


def baseline_table(results_dir: str | Path) -> list[dict]:
    out = []
    for r in latest_by_task(list(ledger.read(results_dir, "baseline"))):
        meds = r.get("medians_ms") or {}
        sol = (r.get("sol") or {}).get("efficiency") or {}
        out.append({
            "task_id": r.get("task_id"), "status": r.get("verdict"),
            "eager_fp32_ms": meds.get("eager_fp32"), "eager_tf32_ms": meds.get("eager_tf32"),
            "tf32_gain": r.get("tf32_gain"), "gflop": (sol.get("flops") or 0) / 1e9 if sol else None,
            "mbytes": (sol.get("bytes") or 0) / 1e6 if sol else None, "bound": sol.get("bound"),
            "sol_ms": sol.get("sol_ms"), "pct_sol_tf32": (r.get("pct_of_sol") or {}).get("eager_tf32"),
            "peak_mem_gb": r.get("peak_mem_gb"),
            "in_1_15ms_window": (r.get("timing") or {}).get("in_runtime_window", {}).get("eager_fp32"),
        })
    return out


def noise_table(results_dir: str | Path) -> list[dict]:
    out = []
    for r in latest_by_task(list(ledger.read(results_dir, "noise"))):
        ci = r.get("aa_ratio") or {}
        out.append({"task_id": r.get("task_id"), "cv_a": r.get("cv_a"), "cv_b": r.get("cv_b"),
                    "aa_ratio": ci.get("point"), "aa_lo": ci.get("lo"), "aa_hi": ci.get("hi"),
                    "min_detectable_speedup": r.get("min_detectable_speedup")})
    return out


# Categorical slots 1-4 of the dataviz reference palette (validated: CVD + normal-vision pass on
# white; contrast below 3:1 for slots 2-4, so every point carries a direct text label).
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#191919", "#666666", "#e6e6e6"


def cost_vs_speedup_chart(agent_table: list[dict], path: Path) -> Path | None:
    """One point per agent/model: x = dollars per task, y = geomean verified speedup."""
    pts = [r for r in agent_table if r.get("cost_per_task_usd") is not None]
    if not pts:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=200)
    ax.axhline(1.0, color=TEXT_SECONDARY, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.995, 1.0, "PyTorch eager TF32 = 1.0×", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=8, color=TEXT_SECONDARY)
    for i, r in enumerate(pts):
        color = SERIES[i % len(SERIES)] if i < len(SERIES) else "#999999"
        label = f"{r['agent']} ({r['model']})" if r.get("model") else str(r["agent"])
        ax.scatter([r["cost_per_task_usd"]], [r["geomean_speedup"]], s=90, color=color,
                   edgecolor="white", linewidth=2, zorder=3, label=label)
        ax.annotate(f"{label}\n{r['geomean_speedup']:.2f}× · pass {r['pass_rate']:.0%}",
                    (r["cost_per_task_usd"], r["geomean_speedup"]), xytext=(8, 6),
                    textcoords="offset points", fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlabel("Mean spend per task (USD, client-side estimates)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Geomean verified speedup (median of runs)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Verified speedup vs spend on KernelBench subset, RTX 4090", loc="left", fontsize=11,
                 color=TEXT_PRIMARY)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ymin = min(0.9, min(r["geomean_speedup"] for r in pts) * 0.95)
    ax.set_ylim(bottom=ymin)
    ax.set_xlim(left=0)
    if len(pts) >= 2:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_all(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    out_dir = results_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    base = baseline_table(results_dir)
    if base:
        cols = list(base[0].keys())
        (out_dir / "baselines.md").write_text("# PyTorch baselines and speed of light on RTX 4090\n\n" + markdown(base, cols))
        write_csv(base, out_dir / "baselines.csv")
        written["baselines"] = str(out_dir / "baselines.md")
    noise = noise_table(results_dir)
    if noise:
        (out_dir / "noise.md").write_text("# Noise floor (A/A tests)\n\n" + markdown(noise, list(noise[0].keys())))
        write_csv(noise, out_dir / "noise.csv")
        written["noise"] = str(out_dir / "noise.md")
    rows = agent_rows(results_dir)
    if rows:
        tasks, agents = aggregate(rows)
        conventions = "Conventions" + (__doc__ or "").split("Conventions", 1)[1]
        text = "# Agent results (official evaluation)\n\n```\n" + conventions.strip() + "\n```\n\n"
        text += "## Per agent\n\n" + markdown(agents, list(agents[0].keys()))
        text += "\n## Per task\n\n" + markdown(tasks, list(tasks[0].keys()))
        (out_dir / "agents.md").write_text(text)
        write_csv(agents, out_dir / "agents.csv")
        write_csv(tasks, out_dir / "agent_tasks.csv")
        written["agents"] = str(out_dir / "agents.md")
        chart = cost_vs_speedup_chart(agents, results_dir / "charts" / "cost_vs_speedup.png")
        if chart:
            written["chart"] = str(chart)
    return written
