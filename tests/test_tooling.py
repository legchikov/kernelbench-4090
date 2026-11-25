"""Pure-Python tooling: config, stats, roofline, agent metering, summaries, EAT repo generation."""

import json
from pathlib import Path

import pytest

from kb4090 import ledger
from kb4090.agents import UsageMeter, build_command, find_usage, stub_from_reference
from kb4090.config import config_hash, deep_merge, load_hardware, load_rig, load_tasks
from kb4090.roofline import speed_of_light
from kb4090.stats import bootstrap_ratio_ci, geomean, min_detectable_speedup, summarise
from kb4090.summarise import aggregate, effective_speedup, write_all

ROOT = Path(__file__).resolve().parents[1]


def test_configs_load_and_tasks_are_unique():
    rig = load_rig(ROOT / "configs" / "rig.yaml")
    assert rig["headline_baseline"] in rig["baselines"]
    tasks = load_tasks(ROOT / "configs" / "tasks.yaml")
    assert len({t["id"] for t in tasks}) == len(tasks) >= 20
    assert {t["level"] for t in tasks} == {1, 2, 3}
    hw = load_hardware("rtx4090")
    assert hw["compute_capability"] == "8.9"
    assert hw["sol"]["guard_peak"] in hw["peaks_tflops"]


def test_task_paths_exist_in_kernelbench_if_present():
    kb = ROOT.parent / "KernelBench" / "KernelBench"
    if not kb.exists():
        pytest.skip("KernelBench checkout not found next to this repo")
    for t in load_tasks(ROOT / "configs" / "tasks.yaml"):
        assert (kb / t["path"]).exists(), t["path"]


def test_deep_merge_and_hash():
    a = {"x": {"y": 1, "z": 2}, "w": 3}
    b = deep_merge(a, {"x": {"y": 5}})
    assert b == {"x": {"y": 5, "z": 2}, "w": 3} and a["x"]["y"] == 1
    assert config_hash(a) == config_hash({**a, "_private": 1}) != config_hash(b)


def test_stats():
    s = summarise([1, 2, 3, 4, 100])
    assert s["median"] == 3 and s["n"] == 5
    ci = bootstrap_ratio_ci([2.0] * 50, [1.0] * 50)
    assert ci["point"] == pytest.approx(2.0) and ci["lo"] == pytest.approx(2.0)
    assert geomean([1, 4]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        geomean([1, 0])
    assert min_detectable_speedup({"lo": 0.98, "hi": 1.03}) == pytest.approx(1.03)


def test_roofline():
    # 4096^3 matmul at 82.6 TFLOPS: compute-bound, ~1.66 ms
    sol = speed_of_light(2 * 4096**3, 3 * 4096 * 4096 * 4, 82.6, 1008)
    assert sol["bound"] == "compute" and sol["sol_ms"] == pytest.approx(1.664, rel=1e-3)
    # 1 GiB in + 1 GiB out at 1008 GB/s: memory-bound, ~2.13 ms
    sol = speed_of_light(0, 2 * 2**30, 82.6, 1008)
    assert sol["bound"] == "memory" and sol["sol_ms"] == pytest.approx(2.130, rel=1e-3)


def test_claude_meter_reads_result_event():
    m = UsageMeter("claude_stream_json", None)
    m.feed(json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 5}}}))
    m.feed(json.dumps({"type": "result", "subtype": "success", "total_cost_usd": 1.25, "num_turns": 7,
                       "usage": {"input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 20}}))
    assert m.cost_usd == 1.25 and m.cost_source == "claude_code_estimate"
    assert m.tokens == {"input": 100, "cached": 50, "output": 20}


def test_dsh_meter_prices_tokens_and_does_not_double_count():
    prices = {"input": 1.0, "cached_input": 0.1, "output": 2.0, "input_includes_cached": True}
    m = UsageMeter("dsh_json", prices)
    ev = {"type": "step_end", "usage": {"prompt_tokens": 1_000_000, "prompt_cache_hit_tokens": 400_000,
                                        "prompt_cache_miss_tokens": 600_000, "completion_tokens": 500_000}}
    m.feed(json.dumps(ev))
    assert m.tokens == {"input": 1_000_000, "cached": 400_000, "output": 500_000}
    # 0.6M uncached x $1 + 0.4M cached x $0.1 + 0.5M out x $2 = 0.6 + 0.04 + 1.0
    assert m.cost_usd == pytest.approx(1.64)


def test_find_usage_nested_and_missing():
    assert find_usage({"a": {"b": [{"input_tokens": 3, "output_tokens": 4}]}}) == {"input": 3, "cached": 0, "output": 4}
    assert find_usage({"type": "text", "text": "hi"}) is None


def test_build_command_formats_placeholders():
    cfg = {"command": ["claude", "-p", "--model", "{model}", "--max-budget-usd", "{budget_usd}"], "prompt_via": "arg"}
    cmd = build_command(cfg, {"model": "opus", "budget_usd": "5.00", "prompt": "go"})
    assert cmd == ["claude", "-p", "--model", "opus", "--max-budget-usd", "5.00", "go"]


def test_stub_renames_model():
    src = "class Model(nn.Module):\n    def __init__(self):\n        super(Model, self).__init__()\n"
    stub = stub_from_reference(src)
    assert "class ModelNew(" in stub and "super(ModelNew, self)" in stub


def _row(agent, task, verdict, speedup, cost):
    return {"agent": agent, "model": "m", "task_id": task, "verdict": verdict, "headline_speedup": speedup,
            "cost_usd": cost}


def test_aggregate_uses_medians_and_counts_failures_as_one():
    rows = [_row("a", "t1", "PASS", 2.0, 1.0), _row("a", "t1", "PASS", 1.5, 1.0), _row("a", "t1", "FAIL_INCORRECT", None, 1.0),
            _row("a", "t2", "INVALID", 9.0, 2.0)]
    assert effective_speedup(rows[3]) == 1.0
    tasks, agents = aggregate(rows)
    t1 = next(t for t in tasks if t["task_id"] == "t1")
    assert t1["median_speedup"] == 1.5 and t1["best_speedup"] == 2.0 and t1["passed"] == 2
    a = agents[0]
    assert a["geomean_speedup"] == pytest.approx((1.5 * 1.0) ** 0.5)
    assert a["fast_1"] == 0.5 and a["fast_2"] == 0.0


def test_ledger_reviews_and_tables(tmp_path):
    r = ledger.append(tmp_path, "agent_run", _row("a", "t1", "SUSPECT", 3.0, 1.0))
    ledger.append(tmp_path, "agent_run", _row("b", "t1", "PASS", 1.2, 2.0))
    ledger.append(tmp_path, "review", {"target_row_id": r["row_id"], "new_verdict": "PASS", "note": "checked with ncu"})
    written = write_all(tmp_path)
    text = Path(written["agents"]).read_text()
    assert "Conventions" in text and "| a | m | 1 |" in text


def test_prepare_eat_repo(tmp_path):
    from kb4090.eat import prepare_eat_repo

    task_file = ROOT / "redteam" / "tasks" / "relu.py"
    tasks = [{"id": "relu", "abs_path": str(task_file), "perturb": True, "sol_margin": None}]
    rig = load_rig(ROOT / "configs" / "rig.yaml")
    out = prepare_eat_repo(tasks, tmp_path / "eatrepo", rig, load_hardware("rtx4090"), 200)
    task_md = (out / "eat" / "task.md").read_text()
    assert task_md.startswith("## Hard Requirements (MUST be in every agent's prompt)")
    assert "## Shared Environment" in task_md and "{{" not in task_md
    assert "SCORE" in (out / "eat" / "eval.sh").read_text()
    assert ".work/" in (out / ".gitignore").read_text()
    meta = json.loads((out / "eat" / "tasks.json").read_text())
    assert (out / "solutions" / "relu.py").exists() and meta["relu"]["stub_sha256"]
