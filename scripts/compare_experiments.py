#!/usr/bin/env python3
"""Compare two experiment runs: metrics, configs, and collateral inventory.

Usage:
    python scripts/compare_experiments.py experiments/run_A experiments/run_B
    python scripts/compare_experiments.py experiments/vast_sync_2026-05-16_0530 experiments/vast_sync_arm_e_2026-05-16_0525
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_yaml(path):
    try:
        return yaml.safe_load(path.read_text())
    except Exception:
        return None


def flatten(d, prefix=""):
    """Flatten a nested dict into dotted keys."""
    items = {}
    if not isinstance(d, dict):
        return {prefix: d}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten(v, key))
        else:
            items[key] = v
    return items


def diff_configs(a, b):
    """Return keys that differ between two flattened configs."""
    fa, fb = flatten(a), flatten(b)
    all_keys = sorted(set(fa) | set(fb))
    diffs = []
    for k in all_keys:
        va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
        if str(va) != str(vb):
            diffs.append((k, va, vb))
    return diffs


def inventory(exp_dir):
    """Count collateral files."""
    d = Path(exp_dir)
    inv = {}
    for subdir in ["checkpoints", "tensorboard", "validation", "visual_debug"]:
        p = d / subdir
        if p.exists():
            files = list(p.rglob("*"))
            inv[subdir] = len([f for f in files if f.is_file()])
        else:
            inv[subdir] = 0
    inv["has_metadata"] = (d / "metadata.json").exists()
    inv["has_config"] = (d / "config.yaml").exists()
    inv["has_training_log"] = (d / "training_log.jsonl").exists() or (d / "training.log").exists()
    return inv


def find_latest_metrics(exp_dir):
    """Find latest validation results.json."""
    d = Path(exp_dir)
    for val_dir in [d / "validation", d / "validation_outputs"]:
        if val_dir.exists():
            step_dirs = sorted(val_dir.glob("step*"))
            if step_dirs:
                rj = step_dirs[-1] / "results.json"
                if rj.exists():
                    return load_json(rj), step_dirs[-1].name
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Compare two experiment runs")
    parser.add_argument("exp_a", help="Path to experiment A")
    parser.add_argument("exp_b", help="Path to experiment B")
    args = parser.parse_args()

    a, b = Path(args.exp_a), Path(args.exp_b)
    if not a.exists():
        print(f"ERROR: {a} not found"); sys.exit(1)
    if not b.exists():
        print(f"ERROR: {b} not found"); sys.exit(1)

    # ── Metadata ──
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPARISON")
    print(f"  A: {a}")
    print(f"  B: {b}")
    print(f"{'='*70}\n")

    meta_a = load_json(a / "metadata.json")
    meta_b = load_json(b / "metadata.json")

    print("── Metadata ──")
    for label, meta in [("A", meta_a), ("B", meta_b)]:
        if meta:
            commit = meta.get("git_commit", "?")[:8]
            dirty = "DIRTY" if meta.get("git_dirty") else "clean"
            name = meta.get("experiment_name", meta.get("config_path", "?"))
            print(f"  {label}: {name}  commit={commit} ({dirty})  {meta.get('timestamp','?')}")
        else:
            print(f"  {label}: no metadata.json")

    # ── Config diff ──
    cfg_a = load_yaml(a / "config.yaml")
    cfg_b = load_yaml(b / "config.yaml")

    if cfg_a and cfg_b:
        diffs = diff_configs(cfg_a, cfg_b)
        print(f"\n── Config Differences ({len(diffs)} keys) ──")
        if diffs:
            max_key = max(len(d[0]) for d in diffs)
            for key, va, vb in diffs:
                print(f"  {key:<{max_key}}  A={va}  B={vb}")
        else:
            print("  (identical)")
    else:
        print("\n── Config Differences ──")
        print(f"  Could not load configs (A={'ok' if cfg_a else 'missing'}, B={'ok' if cfg_b else 'missing'})")

    # ── Metrics ──
    metrics_a, step_a = find_latest_metrics(a)
    metrics_b, step_b = find_latest_metrics(b)

    print(f"\n── Validation Metrics ──")
    if metrics_a or metrics_b:
        all_keys = sorted(set(
            list(flatten(metrics_a or {}).keys()) +
            list(flatten(metrics_b or {}).keys())
        ))
        print(f"  {'metric':<40} {'A ('+str(step_a)+')':<20} {'B ('+str(step_b)+')':<20}")
        print(f"  {'─'*40} {'─'*20} {'─'*20}")
        fa = flatten(metrics_a or {})
        fb = flatten(metrics_b or {})
        for k in all_keys:
            va = fa.get(k, "-")
            vb = fb.get(k, "-")
            marker = ""
            try:
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    marker = " ◄" if va < vb else (" ►" if vb < va else "")
            except Exception:
                pass
            print(f"  {k:<40} {str(va):<20} {str(vb):<20}{marker}")
    else:
        print("  No validation results found in either experiment.")

    # ── Collateral inventory ──
    inv_a = inventory(a)
    inv_b = inventory(b)

    print(f"\n── Collateral Inventory ──")
    print(f"  {'artifact':<25} {'A':<10} {'B':<10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10}")
    for key in sorted(set(list(inv_a.keys()) + list(inv_b.keys()))):
        va = inv_a.get(key, "-")
        vb = inv_b.get(key, "-")
        print(f"  {key:<25} {str(va):<10} {str(vb):<10}")

    print("")


if __name__ == "__main__":
    main()
