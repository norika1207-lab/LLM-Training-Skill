#!/usr/bin/env python3
"""Run one safe, resumable command from an append-only JSONL queue."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_events(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("job_id"):
            latest[item["job_id"]] = item
    return latest


def append_event(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), **payload}, ensure_ascii=False, sort_keys=True) + "\n")


def acquire_lock(lock: Path) -> bool:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "at": now()}, handle)
        return True
    except FileExistsError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path, help="JSONL jobs; each job needs job_id, run_dir, command")
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()
    queue = args.queue.expanduser().resolve()
    events_path = (args.events or queue.with_name("worker-events.jsonl")).expanduser().resolve()
    processed = 0
    while True:
        states = load_events(events_path)
        jobs = []
        for line in queue.read_text(encoding="utf-8").splitlines() if queue.exists() else []:
            try:
                job = json.loads(line)
            except json.JSONDecodeError:
                continue
            if job.get("job_id") and job.get("job_id") not in states and job.get("state", "queued") == "queued":
                jobs.append(job)
        if not jobs:
            if args.once or (args.max_jobs and processed >= args.max_jobs):
                return 0
            time.sleep(args.poll_seconds)
            continue
        job = jobs[0]
        job_id = job["job_id"]
        root = Path(job["run_dir"]).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock = root / "locks" / f"{job_id}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        if not acquire_lock(lock):
            append_event(events_path, {"job_id": job_id, "state": "blocked", "reason": "resource lock exists"})
            continue
        append_event(events_path, {"job_id": job_id, "state": "running", "run_dir": str(root), "command": job.get("command")})
        log_path = root / "logs" / f"worker-{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = job.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list) or not command:
            append_event(events_path, {"job_id": job_id, "state": "blocked", "reason": "command must be a non-empty list or shell-free string"})
            lock.unlink(missing_ok=True)
            continue
        try:
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run([str(part) for part in command], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
            state = "checkpointed" if result.returncode == 0 else "rejected"
            append_event(events_path, {"job_id": job_id, "state": state, "returncode": result.returncode, "log": str(log_path)})
        finally:
            lock.unlink(missing_ok=True)
        processed += 1
        if args.once or (args.max_jobs and processed >= args.max_jobs):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
