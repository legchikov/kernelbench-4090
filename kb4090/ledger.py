"""Append-only JSONL results ledger. Every row carries the env id and config hash it was made with.

Never edit rows by hand. If a result turns out to be wrong, append a correction row
(``kind: review``) that references the original row's ``row_id``; summaries apply reviews.
"""

from __future__ import annotations

import fcntl
import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

LEDGERS = {
    "eval": "evals.jsonl",
    "agent_run": "agent_runs.jsonl",
    "baseline": "baselines.jsonl",
    "noise": "noise.jsonl",
    "review": "reviews.jsonl",
    "redteam": "redteam.jsonl",
}

_HEAVY_KEYS = {"event_ms"}


def _slim(obj: object) -> object:
    """Drop per-iteration arrays from rows (the full JSON stays next to the run)."""
    if isinstance(obj, dict):
        return {k: _slim(v) for k, v in obj.items() if k not in _HEAVY_KEYS}
    if isinstance(obj, list):
        return [_slim(v) for v in obj]
    return obj


def append(results_dir: str | Path, kind: str, row: dict) -> dict:
    path = Path(results_dir) / "ledger" / LEDGERS[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"row_id": uuid.uuid4().hex[:12], "recorded": time.time(), "kind": kind, **_slim(row)}
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(row, default=str) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)
    return row


def read(results_dir: str | Path, kind: str) -> Iterator[dict]:
    path = Path(results_dir) / "ledger" / LEDGERS[kind]
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
