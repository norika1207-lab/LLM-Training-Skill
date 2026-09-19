# Continuous Training Runbook

## Run directory

```text
run-YYYYMMDD-HHMMSS/
├── manifest.json
├── STATUS.md
├── NEXT_ACTIONS.md
├── queue.jsonl
├── preflight.json
├── config.json
├── data/
│   ├── train.manifest.json
│   ├── valid.manifest.json
│   ├── sealed.manifest.json
│   └── failure-bank.jsonl
├── checkpoints/
├── eval/
├── artifacts/
├── logs/
└── locks/
```

## Minimal manifest

```json
{
  "run_id": "receipt-student-20260919-120000",
  "task": "field_extraction",
  "base_model": "…",
  "method": "sft|lora|distill|ranker|classifier|surgery",
  "dataset": {"train_rows": 0, "valid_rows": 0, "sealed_rows": 0, "fingerprint": "…"},
  "config_hash": "…",
  "vocab_hash": "…",
  "seed": 17,
  "device": "cuda:0",
  "checkpoint_policy": {"every_batches": 20, "keep_best": 3},
  "gates": {"valid_metric": "…", "product_metric": "…", "max_regression": 0.01},
  "status": "queued"
}
```

## Queue state transitions

```text
queued → preflight → smoke → running → checkpointed → evaluating
                                  ├──────────────→ blocked
                                  └──────────────→ failed
evaluating → promoted | rejected
rejected → queued (only with a changed hypothesis/config)
blocked → queued (only after blocker is resolved)
```

Do not retry an identical failed run indefinitely. A retry must state what changed: device, input length, batch size, data cleaning, architecture, objective, or dependency. Keep the failed run and its log.

## Worker scheduling rules

1. Preflight each worker immediately before launch. Do not trust an old hardware report.
2. One exclusive heavy job per GPU. Use a lock file containing run id, PID, host, device and start time.
3. Small CPU data validation, manifest generation and report rendering may run in parallel with a GPU job, but never two jobs that compete for the same disk or RAM budget.
4. A download, gated model, or unreachable node is a queue blocker, not a training success. Continue independent local jobs.
5. For remote workers, ship a self-contained task bundle with relative paths, exact command, expected output schema, checksum and return location. Do not rely on a live chat session.
6. A watcher must verify more than PID existence: read heartbeat, last batch, last checkpoint mtime, device utilization, exit code and artifact validity.
7. On completion, the watcher runs evaluation before starting the next dependent rung. On failure, it records the reason and starts only independent jobs.

## Resume rules

Resume only when all of these match: base model identity, dataset fingerprint, tokenizer/vocab hash, config hash, optimizer compatibility and device/runtime assumptions. If new data introduces new vocabulary, start a new run from the base model or implement and test vocabulary expansion. Never silently resume a checkpoint with a different dataset or tokenizer.

On resume, load the latest valid checkpoint, then verify with a small deterministic batch before full continuation. If the latest checkpoint is corrupt, use the newest checkpoint whose checksum and load test pass. Keep the original run id in the child manifest and add `resumed_from`.

## Stopping and notification

Stop a run when it is complete, invalid, demonstrably non-progressing, unsafe for the host, or blocked by missing authority/data. Do not stop merely because it is slow if checkpoints and validation are healthy. When using an external heartbeat or automation, stay silent for unchanged healthy progress and notify only on meaningful state change, completion, failure, or required action.
