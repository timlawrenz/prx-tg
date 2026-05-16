#!/usr/bin/env python3
"""Audit an experiment directory for completeness and reproducibility.

Usage:
    python scripts/audit_experiment.py experiments/2026-05-13_2035
    python scripts/audit_experiment.py experiments/vast_sync_2026-05-16_0530
"""

import argparse
import json
import sys
from pathlib import Path


def check(label, ok, detail=""):
    status = "✓" if ok else "✗"
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok


def audit(exp_dir):
    d = Path(exp_dir)
    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT AUDIT: {d}")
    print(f"{'='*60}\n")

    # 1. metadata.json exists
    meta_path = d / "metadata.json"
    has_meta = meta_path.exists()
    if check("metadata.json exists", has_meta):
        passed += 1
        meta = json.loads(meta_path.read_text())

        # 2. git not dirty
        dirty = meta.get("git_dirty", True)
        if check("git clean at run time", not dirty,
                 f"commit={meta.get('git_commit','?')[:8]}"):
            passed += 1
        else:
            failed += 1

        # 3. git commit recorded
        if check("git commit recorded", bool(meta.get("git_commit")),
                 meta.get("git_commit", "?")[:8]):
            passed += 1
        else:
            failed += 1

        # 4. config path recorded
        if check("config source recorded", bool(meta.get("config_path") or meta.get("config_source"))):
            passed += 1
        else:
            failed += 1
    else:
        failed += 1

    # 5. frozen config
    cfg_path = d / "config.yaml"
    if check("config.yaml frozen copy", cfg_path.exists()):
        passed += 1
    else:
        failed += 1

    # 6. checkpoints
    ckpt_dir = d / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    has_final = any("final" in c.name for c in ckpts)
    if check(f"checkpoints present", len(ckpts) > 0,
             f"{len(ckpts)} files, final={'yes' if has_final else 'NO'}"):
        passed += 1
    else:
        failed += 1

    # 7. training log
    has_log = (d / "training_log.jsonl").exists() or (d / "training.log").exists()
    if check("training log present", has_log):
        passed += 1
    else:
        failed += 1

    # 8. tensorboard
    tb_dir = d / "tensorboard"
    tb_files = list(tb_dir.rglob("events.*")) if tb_dir.exists() else []
    if check("tensorboard events", len(tb_files) > 0, f"{len(tb_files)} files"):
        passed += 1
    else:
        failed += 1

    # 9. validation outputs
    val_dir = d / "validation" if (d / "validation").exists() else d / "validation_outputs"
    val_results = list(val_dir.rglob("results.json")) if val_dir.exists() else []
    if check("validation results.json", len(val_results) > 0, f"{len(val_results)} files"):
        passed += 1
    else:
        failed += 1

    # Summary
    total = passed + failed
    print(f"\n  Score: {passed}/{total} checks passed")
    if failed == 0:
        print("  ✓  Experiment is REPRODUCIBLE and COMPLETE")
    elif failed <= 2:
        print("  ⚠  Experiment has minor gaps — review above")
    else:
        print("  ✗  Experiment has significant gaps — NOT reproducible")
    print("")

    return failed


def main():
    parser = argparse.ArgumentParser(description="Audit experiment directory")
    parser.add_argument("exp_dir", nargs="+", help="Experiment directory/directories to audit")
    args = parser.parse_args()

    total_failures = 0
    for d in args.exp_dir:
        total_failures += audit(d)

    sys.exit(1 if total_failures > 0 else 0)


if __name__ == "__main__":
    main()
