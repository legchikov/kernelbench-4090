"""Subprocess entry point: ``python -I -m kb4090.worker --mode official --task T --candidate C --out R``.

Always writes a JSON result (even on a crash inside the harness) so the parent can tell a broken
kernel from a broken harness. Use runner.run_worker() rather than calling this directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _json_default(obj: object) -> object:
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() <= 16 else f"<tensor {tuple(obj.shape)}>"
        if isinstance(obj, torch.dtype | torch.device):
            return str(obj)
    except Exception:
        pass
    return str(obj)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kb4090.worker")
    ap.add_argument("--mode", required=True, choices=["official", "dev", "baseline", "noise"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--candidate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig-json", required=True)
    ap.add_argument("--hw-json", required=True)
    ap.add_argument("--task-meta-json", required=True)
    args = ap.parse_args(argv)

    rig = json.loads(Path(args.rig_json).read_text())
    hw = json.loads(Path(args.hw_json).read_text())
    meta = json.loads(Path(args.task_meta_json).read_text())
    started = time.time()
    try:
        from . import evaluate as ev

        if args.mode in ("official", "dev"):
            if not args.candidate:
                raise SystemExit("--candidate is required for official/dev mode")
            res = ev.evaluate_candidate(args.task, args.candidate, rig, hw, meta, mode=args.mode)
        elif args.mode == "baseline":
            res = ev.evaluate_baseline(args.task, rig, hw, meta)
        else:
            res = ev.evaluate_noise(args.task, rig, hw, meta)
    except BaseException as exc:  # noqa: BLE001 - we must always write a result
        res = {
            "kind": args.mode, "task_id": meta.get("id"), "task_path": args.task,
            "candidate_path": args.candidate, "verdict": "HARNESS_ERROR",
            "reasons": [{"verdict": "HARNESS_ERROR", "reason": f"{type(exc).__name__}: {str(exc)[:500]}"}],
            "traceback": traceback.format_exc()[-6000:], "started": started,
        }
    Path(args.out).write_text(json.dumps(res, default=_json_default, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
