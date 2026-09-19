#!/usr/bin/env python3
"""Exercise the local-first run contract without a model or network."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([sys.executable, *args], cwd=ROOT, text=True, capture_output=True)
    if result.returncode != expect:
        raise AssertionError(f"command failed ({result.returncode} != {expect}): {args}\n{result.stdout}\n{result.stderr}")
    return result


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mercury-skill-self-test-") as temp:
        base = Path(temp)
        run_dir = base / "run"
        run("scripts/scaffold_run.py", str(run_dir), "--task", "self-test", "--method", "sft", "--device", "cpu")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["tracking"]["provider"] == "local"
        assert manifest["lifecycle"]["branch_state"] == "active"

        checkpoint = run_dir / "checkpoints" / "smoke.pt"
        checkpoint.write_bytes(b"deterministic-test-artifact")
        run("scripts/track_run.py", str(run_dir), "--state", "running", "--event", "smoke", "--step", "1", "--loss", "1.0")
        report = base / "eval.json"
        report.write_text(json.dumps({"report_type": "summary", "decision": "validated", "metrics": {"smoke": 1.0}}), encoding="utf-8")
        run("scripts/promote_artifact.py", str(run_dir), str(checkpoint), "--eval-report", str(report), "--decision", "validated", "--confirm")
        validation = run("scripts/validate_training_run.py", str(run_dir))
        assert json.loads(validation.stdout)["ok"] is True
        comparison = run("scripts/compare_runs.py", str(run_dir), "--format", "json")
        assert json.loads(comparison.stdout)[0]["status"] == "validated"
    print("Mercury Skill self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
