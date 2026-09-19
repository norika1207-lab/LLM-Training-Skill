#!/usr/bin/env python3
"""Append durable run events and update a small progress snapshot."""
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else default
    except (OSError, json.JSONDecodeError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--state", default=None, help="queued/running/checkpointed/evaluating/etc.")
    parser.add_argument("--event", default="heartbeat")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--loss", type=float)
    parser.add_argument("--metric", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--checkpoint")
    parser.add_argument("--message", default="")
    args = parser.parse_args()

    root = args.run_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"run directory does not exist: {root}")
    timestamp = now()
    metrics: dict[str, object] = {}
    for item in args.metric:
        if "=" not in item:
            raise SystemExit(f"invalid metric (expected NAME=VALUE): {item}")
        name, raw = item.split("=", 1)
        try:
            value: object = float(raw)
        except ValueError:
            value = raw
        metrics[name] = value

    event = {
        "at": timestamp,
        "event": args.event,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    for name, value in (("state", args.state), ("epoch", args.epoch), ("batch", args.batch),
                        ("step", args.step), ("loss", args.loss), ("checkpoint", args.checkpoint),
                        ("message", args.message or None)):
        if value is not None:
            event[name] = value
    if metrics:
        event["metrics"] = metrics

    events_path = root / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    progress = load_json(root / "progress.json", {})
    progress.update({"updated_at": timestamp, "host": event["host"], "pid": event["pid"]})
    for key in ("state", "epoch", "batch", "step", "loss", "checkpoint"):
        if key in event:
            progress[key] = event[key]
    if metrics:
        progress.setdefault("metrics", {}).update(metrics)
    (root / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = root / "manifest.json"
    if args.state and manifest_path.exists():
        manifest = load_json(manifest_path, {})
        manifest["status"] = args.state
        manifest["last_event_at"] = timestamp
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
