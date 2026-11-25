"""Run a coding agent (Claude Code, DeepSeek Harness, ...) on one task at a fixed dollar budget.

Each run gets its own workspace:
  runs/<agent>/<task_id>/run<k>/
    TASK.md        the full task statement and rules (rendered from kb4090/prompts/agent_task.md)
    CLAUDE.md, AGENTS.md   pointers to TASK.md (each harness reads its own file)
    problem.py     read-only copy of the reference (the official eval uses the canonical file)
    solution.py    starting point; the agent edits this
    dev_eval.sh    quick feedback: base inputs only, fewer seeds, timing under the GPU lock
    transcript.jsonl, agent_meta.json, final_solution.py, official_eval.json

The official evaluation runs after the agent exits, on a snapshot of solution.py, with the hidden
distributions, and is the only number that goes in the ledger.

Budgets:
  * Claude Code enforces ``--max-budget-usd`` itself, and reports ``total_cost_usd`` (a client-side
    estimate) in its final ``result`` event.
  * DeepSeek Harness (``dsh --profile headless --json``) has no budget flag that we know of. We read
    token usage from its event stream, price it with the per-million-token prices you put in
    configs/agents.yaml, and kill the run when the estimate reaches the budget.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .config import PACKAGE_ROOT, REPO_ROOT, resolve_path

USAGE_ALIASES = {
    "input": ("input_tokens", "prompt_tokens", "inputTokens", "promptTokens", "prompt_cache_miss_tokens"),
    "cached": ("cache_read_input_tokens", "cached_tokens", "prompt_cache_hit_tokens", "cacheReadTokens",
               "cacheReadInputTokens", "cache_read_tokens"),
    "output": ("output_tokens", "completion_tokens", "outputTokens", "completionTokens"),
}


class UsageMeter:
    """Accumulates token usage and cost from an agent's JSON event stream."""

    def __init__(self, parser: str, prices: dict | None, cumulative: bool = False) -> None:
        self.parser = parser
        self.prices = prices or {}
        self.cumulative = cumulative
        self.tokens = {"input": 0, "cached": 0, "output": 0}
        self.cost_usd: float | None = None
        self.cost_source: str | None = None
        self.result_event: dict | None = None
        self.events = 0

    def feed(self, line: str) -> None:
        line = line.strip()
        if not line or not line.startswith("{"):
            return
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return
        self.events += 1
        if self.parser == "claude_stream_json":
            self._claude(ev)
        elif self.parser == "dsh_json":
            self._dsh(ev)

    def _claude(self, ev: dict) -> None:
        if ev.get("type") == "result":
            self.result_event = ev
            if ev.get("total_cost_usd") is not None:
                self.cost_usd = float(ev["total_cost_usd"])
                self.cost_source = "claude_code_estimate"
            usage = ev.get("usage") or {}
            self.tokens = {
                "input": int(usage.get("input_tokens", 0) or 0) + int(usage.get("cache_creation_input_tokens", 0) or 0),
                "cached": int(usage.get("cache_read_input_tokens", 0) or 0),
                "output": int(usage.get("output_tokens", 0) or 0),
            }

    def _dsh(self, ev: dict) -> None:
        if ev.get("type") == "final":
            self.result_event = ev
        usage = find_usage(ev)
        if not usage:
            return
        if self.cumulative:
            self.tokens = {k: max(self.tokens[k], usage.get(k, 0)) for k in self.tokens}
        else:
            for k in self.tokens:
                self.tokens[k] += usage.get(k, 0)
        self.cost_usd = self.price(self.tokens)
        self.cost_source = "token_prices" if self.cost_usd is not None else None

    def price(self, tokens: dict) -> float | None:
        p_in, p_out = self.prices.get("input"), self.prices.get("output")
        if p_in is None or p_out is None:
            return None
        p_cached = self.prices.get("cached_input", p_in)
        uncached = max(0, tokens["input"] - tokens["cached"]) if self.prices.get("input_includes_cached") \
            else tokens["input"]
        return (uncached * p_in + tokens["cached"] * p_cached + tokens["output"] * p_out) / 1e6


def find_usage(obj: Any) -> dict | None:
    """Find a token-usage dict anywhere inside an event (schemas differ between harnesses)."""
    if isinstance(obj, dict):
        hits = {}
        for kind, aliases in USAGE_ALIASES.items():
            for a in aliases:  # first alias wins, so prompt_tokens + prompt_cache_miss_tokens isn't double-counted
                if isinstance(obj.get(a), int | float) and not isinstance(obj.get(a), bool):
                    hits[kind] = int(obj[a])
                    break
        if "input" in hits or "output" in hits:
            return {k: hits.get(k, 0) for k in ("input", "cached", "output")}
        for v in obj.values():
            found = find_usage(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_usage(v)
            if found:
                return found
    return None


def stub_from_reference(src: str) -> str:
    header = (
        "# Starting point: this is the reference implementation renamed to ModelNew.\n"
        "# Replace the PyTorch computation with your own CUDA (torch.utils.cpp_extension.load_inline)\n"
        "# or Triton kernel(s). Keep the constructor and forward() signatures unchanged.\n\n"
    )
    return header + src.replace("class Model(", "class ModelNew(").replace("super(Model,", "super(ModelNew,")


def render(template: str, **values: Any) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def _hardware_facts(hw: dict) -> str:
    peaks = hw.get("peaks_tflops", {})
    lines = [
        f"- GPU: {hw.get('name')} ({hw.get('arch')}, compute capability {hw.get('compute_capability')}, "
        f"{hw.get('sms')} SMs, {hw.get('memory_gb')} GB)",
        f"- DRAM bandwidth: ~{hw.get('dram_bandwidth_gbs')} GB/s; L2: {hw.get('l2_mb')} MB",
        "- Peak dense TFLOPS (unverified spec-sheet values): "
        + ", ".join(f"{k} {v}" for k, v in peaks.items() if v is not None),
    ]
    for note in hw.get("agent_notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines)


def prepare_workspace(task: dict, run_dir: Path, budget_usd: float, agent_name: str, hw: dict,
                      wall_clock_s: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    src = Path(task["abs_path"]).read_text(encoding="utf-8")
    (run_dir / "problem.py").write_text(src)
    (run_dir / "solution.py").write_text(stub_from_reference(src))
    template = (PACKAGE_ROOT / "prompts" / "agent_task.md").read_text(encoding="utf-8")
    task_md = render(template, task_id=task["id"], task_path=task["abs_path"], budget_usd=f"{budget_usd:.2f}",
                     wall_clock_min=int(wall_clock_s // 60), hardware=_hardware_facts(hw), agent=agent_name)
    (run_dir / "TASK.md").write_text(task_md)
    pointer = "Read TASK.md in this directory first and follow it exactly. Only edit solution.py and files under work/.\n"
    (run_dir / "CLAUDE.md").write_text(pointer)
    (run_dir / "AGENTS.md").write_text(pointer)
    (run_dir / "work").mkdir()
    dev = run_dir / "dev_eval.sh"
    dev.write_text("#!/usr/bin/env bash\n# Quick feedback on solution.py. Takes the GPU lock for timing.\n"
                   "set -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
                   f"exec kb4090 dev-eval --task {json.dumps(task['abs_path'])} --candidate solution.py \"$@\"\n")
    dev.chmod(0o755)


def build_command(agent_cfg: dict, values: dict) -> list[str]:
    cmd = [str(part).format(**values) for part in agent_cfg["command"]]
    if agent_cfg.get("prompt_via", "arg") == "arg":
        cmd.append(values["prompt"])
    prefix = agent_cfg.get("sandbox_prefix")
    if prefix:
        cmd = [str(resolve_path(prefix[0], REPO_ROOT))] + [str(p).format(**values) for p in prefix[1:]] + cmd
    return cmd


def _reader(stream: Any, q: queue.Queue) -> None:
    for line in iter(stream.readline, ""):
        q.put(line)
    q.put(None)


def launch(agent_name: str, agent_cfg: dict, run_dir: Path, budget_usd: float, wall_clock_s: float,
           rig: dict) -> dict:
    prices = agent_cfg.get("prices_usd_per_mtok") or {}
    native_budget = bool(agent_cfg.get("native_budget", False))
    meter = UsageMeter(agent_cfg.get("output", "text"), prices, bool(agent_cfg.get("usage_is_cumulative", False)))
    if not native_budget and meter.price({"input": 1, "cached": 0, "output": 1}) is None:
        raise ValueError(
            f"agent '{agent_name}' has no native budget and no prices in configs/agents.yaml "
            "(prices_usd_per_mtok.input/output). Fill in current prices so the budget can be enforced.")
    values = {
        "model": agent_cfg.get("model", ""),
        "budget_usd": f"{budget_usd:.2f}",
        "workdir": str(run_dir),
        "lock_dir": str(Path(rig["lock_dir"]).expanduser()),
        "home": str(Path.home()),
        "repo": str(REPO_ROOT),
        "prompt": agent_cfg.get("prompt", "Read TASK.md in the current directory and complete the task."),
    }
    cmd = build_command(agent_cfg, values)
    env = os.environ.copy()
    env.update({k: str(v).format(**values) for k, v in (agent_cfg.get("env") or {}).items()})
    env["KB4090_RIG"] = rig.get("_source", "")
    if rig.get("_overlay"):
        env["KB4090_RIG_OVERLAY"] = rig["_overlay"]
    env.pop("NVIDIA_TF32_OVERRIDE", None)

    transcript = open(run_dir / "transcript.jsonl", "w", encoding="utf-8")  # noqa: SIM115
    stderr_log = open(run_dir / "agent_stderr.log", "w", encoding="utf-8")  # noqa: SIM115
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE, stderr=stderr_log,
                            stdin=subprocess.PIPE if agent_cfg.get("prompt_via") == "stdin" else subprocess.DEVNULL,
                            text=True, start_new_session=True)
    if agent_cfg.get("prompt_via") == "stdin":
        assert proc.stdin is not None
        proc.stdin.write(values["prompt"])
        proc.stdin.close()
    q: queue.Queue = queue.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()

    exit_reason = "completed"
    while True:
        try:
            line = q.get(timeout=1.0)
        except queue.Empty:
            line = ""
        if line is None:
            break
        if line:
            transcript.write(line)
            meter.feed(line)
        if time.time() - started > wall_clock_s:
            exit_reason = "wall_clock_limit"
            _terminate(proc)
            break
        if not native_budget and meter.cost_usd is not None and meter.cost_usd >= budget_usd:
            exit_reason = "budget_guard"
            _terminate(proc)
            break
    rc = proc.wait()
    transcript.close()
    stderr_log.close()
    if exit_reason == "completed" and meter.result_event and meter.result_event.get("subtype"):
        exit_reason = str(meter.result_event["subtype"])
    return {
        "agent": agent_name,
        "model": agent_cfg.get("model"),
        "command": cmd[:-1] if agent_cfg.get("prompt_via", "arg") == "arg" else cmd,
        "return_code": rc,
        "exit_reason": exit_reason,
        "wall_s": time.time() - started,
        "budget_usd": budget_usd,
        "cost_usd": meter.cost_usd,
        "cost_source": meter.cost_source,
        "tokens": meter.tokens,
        "num_turns": (meter.result_event or {}).get("num_turns"),
        "events": meter.events,
    }


def _terminate(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def snapshot_solution(run_dir: Path) -> Path:
    final = run_dir / "final_solution.py"
    shutil.copyfile(run_dir / "solution.py", final)
    return final
