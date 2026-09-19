#!/usr/bin/env python3
"""Compare local training runs without inventing missing metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    else:
        result[prefix] = value
    return result


def best_eval(root: Path) -> dict[str, Any]:
    reports = sorted((root / "eval").glob("*.json"))
    candidates = []
    for path in reports:
        report = read_json(path)
        if report:
            candidates.append({"file": str(path), **report})
    if not candidates:
        return {}
    # Prefer an explicitly marked summary, otherwise use the newest report.
    for report in reversed(candidates):
        if report.get("summary") is True or report.get("report_type") == "summary":
            return report
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    rows = []
    for raw in args.run_dirs:
        root = raw.expanduser().resolve()
        manifest = read_json(root / "manifest.json")
        report = best_eval(root)
        metrics = flatten(report.get("metrics", report.get("results", {})), "metrics")
        rows.append({
            "run_dir": str(root),
            "run_id": manifest.get("run_id", root.name),
            "status": manifest.get("status", "unknown"),
            "method": manifest.get("method", "unknown"),
            "seed": manifest.get("seed"),
            "device": manifest.get("device"),
            "dataset_fingerprint": manifest.get("dataset", {}).get("fingerprint"),
            "config_hash": manifest.get("config_hash"),
            "eval_file": report.get("file"),
            "metrics": metrics,
        })
    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print("| run | status | method | seed | device | dataset fingerprint | metrics |")
    print("| --- | --- | --- | ---: | --- | --- | --- |")
    for row in rows:
        metric_text = ", ".join(f"{key.removeprefix('metrics.')}={value}" for key, value in row["metrics"].items()) or "(missing)"
        print(f"| `{row['run_id']}` | `{row['status']}` | `{row['method']}` | {row['seed'] if row['seed'] is not None else '?'} | `{row['device'] or '?'}` | `{row['dataset_fingerprint'] or 'pending'}` | {metric_text} |")
    print("\n註：缺少 eval report 的 run 會顯示 `(missing)`，不會被推測或補值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
