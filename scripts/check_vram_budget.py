#!/usr/bin/env python3
"""Does the arm's free-VRAM guard still cover what the trainer actually needs?

The launcher refuses to claim while free VRAM < LF_MIN_FREE_VRAM_GB. That guard is a
per-arm constant, and constants rot: a config change (batch, resolution, precision,
gradient_checkpointing) moves the footprint while the guard stays put. This script
re-derives the verdict from the run's OWN measurements, so the check is a command
rather than something someone has to remember.

Inputs — all measured, none assumed:
  * tensor peak    torch.cuda.max_memory_allocated(), logged to tensorboard as
                   memory/peak_vram_gb. It is a high-water mark and is never reset
                   mid-run, so the LAST value is the run's peak; in-train validation
                   happens in-process and is therefore included.
  * card overhead  nvidia-smi per-process used_memory minus that tensor peak (CUDA
                   context + allocator cache). Default 0.65 GiB, measured 2026-09-21
                   on this 4090 arm; override with --overhead.

Verdict:
  guard >= footprint + MARGIN_GIB  -> PASS
  guard <  footprint + MARGIN_GIB  -> FAIL  (a launch can OOM)
  guard >  2x footprint            -> WARN  (over-reservation starves co-tenants —
                                      the failure that cost a 7-hour idle window)

Usage:
  scripts/check_vram_budget.py --arm-env ~/.config/prx-tg/arms/<arm>.env \
                               --run-dir experiments/<arm>/runs/<timestamp>

Run it with the project venv (it needs tensorboard):
  .venv/bin/python3 scripts/check_vram_budget.py ...
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

MARGIN_GIB = 2.0
DEFAULT_OVERHEAD_GIB = 0.65
TB_TAG = "memory/peak_vram_gb"


def guard_from_env(path: str) -> float | None:
    """Read LF_MIN_FREE_VRAM_GB from an arm env file, ignoring commented lines.

    Takes the LAST assignment, and warns on duplicates: systemd applies an
    EnvironmentFile in order, so a trailing duplicate silently overrides an earlier
    line. Found live 2026-09-21 — a documented `=14` was overridden by a stale `=20`
    further down the file, and the launcher kept refusing at 20 with nothing erroring.
    """
    try:
        text = open(path).read()
    except OSError as exc:
        print(f"cannot read {path}: {exc}")
        return None
    found: list[tuple[int, float]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        m = re.match(r"\s*(?:export\s+)?LF_MIN_FREE_VRAM_GB\s*=\s*([0-9.]+)", line)
        if m:
            found.append((n, float(m.group(1))))
    if not found:
        return None
    if len(found) > 1:
        print(f"WARN: LF_MIN_FREE_VRAM_GB assigned {len(found)}x in "
              f"{os.path.basename(path)} at lines {[n for n, _ in found]} — "
              f"the LAST wins ({found[-1][1]:g}); delete the stale line(s)")
    return found[-1][1]


def tensor_peak(run_dir: str) -> tuple[float | None, int]:
    """Highest memory/peak_vram_gb across the run's tensorboard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not importable — run this with the project venv (.venv/bin/python3)")
        return None, 0
    files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    best, seen = None, 0
    for f in sorted(files):
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
            if TB_TAG in ea.Tags().get("scalars", []):
                val = ea.Scalars(TB_TAG)[-1].value
                best = val if best is None else max(best, val)
                seen += 1
        except Exception as exc:
            print(f"  (skipped {os.path.basename(f)[:48]}: {type(exc).__name__})")
    return best, seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-env", required=True, help="arm env file holding LF_MIN_FREE_VRAM_GB")
    ap.add_argument("--run-dir", required=True, help="run dir containing tensorboard/")
    ap.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD_GIB,
                    help=f"card-level overhead above the tensor peak (default {DEFAULT_OVERHEAD_GIB} GiB)")
    ap.add_argument("--margin", type=float, default=MARGIN_GIB)
    args = ap.parse_args()

    guard = guard_from_env(args.arm_env)
    if guard is None:
        print(f"FAIL: no LF_MIN_FREE_VRAM_GB found in {args.arm_env}")
        return 1
    peak, n = tensor_peak(args.run_dir)
    if peak is None:
        print(f"FAIL: no {TB_TAG} in {args.run_dir} (event files read: {n})")
        return 1

    footprint = peak + args.overhead
    margin = guard - footprint
    print(f"guard        : {guard:.2f} GiB free  (from {args.arm_env})")
    print(f"tensor peak  : {peak:.2f} GiB       (tb {TB_TAG}, {n} event file(s))")
    print(f"card overhead: {args.overhead:.2f} GiB")
    print(f"footprint    : {footprint:.2f} GiB  (what the card must have free)")
    print(f"margin       : {margin:+.2f} GiB  (need >= {args.margin:.2f})")

    if margin < args.margin:
        print("VERDICT: FAIL — a launch can OOM; raise LF_MIN_FREE_VRAM_GB")
        return 1
    if guard > 2 * footprint:
        print("VERDICT: WARN — over-reserved; co-tenants starve while the GPU sits idle")
        return 0
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
