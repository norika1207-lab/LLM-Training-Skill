#!/usr/bin/env python3
"""Create an immutable release manifest after an explicit validated decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--eval-report", required=True, type=Path)
    parser.add_argument("--decision", choices=("validated", "promoted"), required=True)
    parser.add_argument("--active-name", default="active")
    parser.add_argument("--confirm", action="store_true", help="required for changing manifest status and active pointer")
    args = parser.parse_args()

    root = args.run_dir.expanduser().resolve()
    artifact = args.artifact.expanduser().resolve()
    report_path = args.eval_report.expanduser().resolve()
    if not args.confirm:
        raise SystemExit("refusing promotion without --confirm")
    if not artifact.is_file():
        raise SystemExit(f"artifact is not a file: {artifact}")
    report = read_json(report_path)
    if not report:
        raise SystemExit(f"eval report is missing or invalid: {report_path}")
    if report.get("decision") not in {"validated", "promoted"}:
        raise SystemExit("eval report must contain decision=validated or decision=promoted")
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest:
        raise SystemExit(f"missing manifest: {manifest_path}")
    if manifest.get("status") in {"rejected", "blocked", "contaminated"}:
        raise SystemExit(f"cannot promote run in status={manifest.get('status')}")

    release_dir = root / "artifacts" / "releases"
    release_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    stored_eval = eval_dir / report_path.name
    if report_path.resolve() != stored_eval.resolve():
        shutil.copy2(report_path, stored_eval)
    digest = sha256(artifact)
    target = release_dir / f"{artifact.stem}-{digest[:12]}{artifact.suffix}"
    if not target.exists():
        shutil.copy2(artifact, target)
    release = {
        "release_id": target.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.get("run_id", root.name),
        "source_artifact": str(artifact),
        "artifact": str(target),
        "artifact_sha256": digest,
        "artifact_bytes": target.stat().st_size,
        "eval_report": str(stored_eval),
        "eval_decision": report.get("decision"),
        "metrics": report.get("metrics", report.get("results", {})),
        "dataset": manifest.get("dataset", {}),
        "config_hash": manifest.get("config_hash"),
        "vocab_hash": manifest.get("vocab_hash"),
        "seed": manifest.get("seed"),
        "device": manifest.get("device"),
    }
    release_path = release_dir / f"{target.stem}.release-manifest.json"
    release_path.write_text(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.decision == "promoted":
        active = root / "artifacts" / args.active_name
        pointer = active.with_suffix(active.suffix + ".json")
        pointer.write_text(json.dumps({"release_manifest": str(release_path), "artifact": str(target), "sha256": digest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["status"] = "promoted"
        manifest["active_release"] = str(release_path)
    else:
        manifest["status"] = "validated"
        manifest["validated_release"] = str(release_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
