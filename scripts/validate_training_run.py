#!/usr/bin/env python3
"""Check durable training-run structure without judging model quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = ("manifest.json", "config.json", "STATUS.md", "NEXT_ACTIONS.md", "queue.jsonl")
REQUIRED_DIRS = ("checkpoints", "eval", "artifacts", "logs", "locks", "data", "failure-bank")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure, do not crash
        return {"_error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.expanduser().resolve()
    errors = []
    warnings = []
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            errors.append(f"missing file: {name}")
    for name in REQUIRED_DIRS:
        if not (root / name).is_dir():
            errors.append(f"missing directory: {name}")

    manifest = read_json(root / "manifest.json") if (root / "manifest.json").is_file() else {}
    for key in ("run_id", "task", "method", "dataset", "config_hash", "seed", "status"):
        if key not in manifest:
            errors.append(f"manifest missing key: {key}")
    if manifest.get("dataset", {}).get("fingerprint") in (None, "", "pending"):
        warnings.append("dataset fingerprint is not finalized")
    if manifest.get("vocab_hash") in (None, "", "pending"):
        warnings.append("vocab hash is not finalized")
    lifecycle = manifest.get("lifecycle", {})
    if lifecycle.get("branch_state") not in {None, "active", "promising", "plateaued", "dead", "superseded", "promoted"}:
        errors.append("invalid lifecycle.branch_state")
    tracking = manifest.get("tracking", {})
    if tracking.get("provider", "local") != "local" and not (root / "events.jsonl").is_file():
        warnings.append("external tracking configured without local events.jsonl fallback")

    checkpoints = list((root / "checkpoints").glob("*") if (root / "checkpoints").is_dir() else [])
    eval_files = list((root / "eval").glob("*") if (root / "eval").is_dir() else [])
    artifacts = list((root / "artifacts").glob("*") if (root / "artifacts").is_dir() else [])
    if manifest.get("status") in {"running", "checkpointed", "evaluating", "promoted"} and not checkpoints:
        errors.append("active/completed run has no checkpoint")
    if manifest.get("status") in {"evaluating", "validated", "promoted", "rejected"} and not eval_files:
        errors.append("evaluated run has no eval output")
    if manifest.get("status") == "promoted" and not artifacts:
        errors.append("promoted run has no artifact")

    result = {"run_dir": str(root), "ok": not errors, "errors": errors, "warnings": warnings,
              "checkpoint_files": len(checkpoints), "eval_files": len(eval_files), "artifact_files": len(artifacts)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
