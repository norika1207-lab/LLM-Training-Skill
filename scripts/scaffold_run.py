#!/usr/bin/env python3
"""Create a durable, non-destructive training run skeleton."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--base-model", default="unknown")
    parser.add_argument("--device", default="preflight-required")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.run_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.force:
        raise SystemExit(f"refusing non-empty run directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "eval", "artifacts", "logs", "locks", "data", "failure-bank"):
        (root / name).mkdir(exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    config = {
        "task": args.task,
        "method": args.method,
        "base_model": args.base_model,
        "device": args.device,
        "seed": args.seed,
        "created_at": now,
    }
    config_text = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (root / "config.json").write_text(config_text, encoding="utf-8")
    manifest = {
        "run_id": root.name,
        "task": args.task,
        "method": args.method,
        "base_model": args.base_model,
        "dataset": {"train_rows": 0, "valid_rows": 0, "sealed_rows": 0, "fingerprint": "pending"},
        "config_hash": sha256_text(config_text),
        "vocab_hash": "pending",
        "seed": args.seed,
        "device": args.device,
        "checkpoint_policy": {"every_batches": 20, "keep_best": 3},
        "gates": {"valid_metric": "pending", "product_metric": "pending", "max_regression": 0.01},
        "status": "queued",
        "created_at": now,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "queue.jsonl").write_text(json.dumps({"state": "queued", "at": now, "run_id": root.name}) + "\n", encoding="utf-8")
    (root / "STATUS.md").write_text(f"# {root.name}\n\n- 狀態：queued\n- 建立時間：{now}\n- 下一步：完成 preflight 與 smoke\n", encoding="utf-8")
    (root / "NEXT_ACTIONS.md").write_text("# Next actions\n\n1. 完成硬體／依賴 preflight。\n2. 驗證資料 manifest 與 fingerprint。\n3. 跑 smoke，確認第一個 checkpoint 可載入並可評估。\n", encoding="utf-8")
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
