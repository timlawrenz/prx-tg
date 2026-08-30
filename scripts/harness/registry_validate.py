#!/usr/bin/env python3
"""Validate the prx-tg avenue registry (Research Loop, Phase 1b, M1).

Stdlib-only. Usage:
    .venv/bin/python3 scripts/harness/registry_validate.py research/avenues/registry.json

Exit codes:
    0 = valid (warnings allowed)
    1 = hard error (registry must not enter a tick)

Mirrors the stratum-hq harness doctrine: the registry is the source of truth;
every invariant here is enforced BEFORE any tick touches it.
"""
import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = 1
STATES = {"registered", "active", "terminal_validated", "terminal_falsified", "blocked"}
LEVELS = {"prior": {"high", "med", "low"},
          "measurability": {"high", "med", "low"},
          "cost": {"low", "med", "high"}}
DECLARATION_FIELDS = [
    "scope", "differs_from", "output_semantics", "provenance", "abstention",
    "qualification_gate", "expected_gpu_hours", "config", "arm_issue",
]
KNOWN_GATES = {
    "default_champion_gate", "stability_only", "band_containment",
    "phase4_gated",  # placeholder until gates_calibration.json registers the G-suite
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("registry", type=Path)
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    reg = json.loads(args.registry.read_text())

    # Top level
    if reg.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("champion", "selection_progress", "exploration", "falsification",
                "cost_tiers_gpu_hours", "calibration_file", "candidates"):
        if key not in reg:
            errors.append(f"missing top-level key: {key}")
    champ = reg.get("champion", {})
    for k in ("slug", "checkpoint"):
        if not champ.get(k):
            errors.append(f"champion.{k} missing")
    if not reg.get("candidates"):
        errors.append("candidates list empty — registry cannot be terminal at M1")
    if reg.get("selection_progress") is None or not isinstance(reg["selection_progress"], int):
        errors.append("selection_progress must be an int")

    # Frozen calibration must exist BEFORE the first verdict (HARKing guard).
    # At M1 it legitimately does not exist yet: warning, not error — becomes a
    # hard error inside tick.py --write.
    cal = Path(reg.get("calibration_file", ""))
    if not cal.is_file():
        warnings.append(f"calibration file not yet frozen: {cal} "
                        "(hard error at tick --write time, expect M2 to produce it)")

    cands = reg.get("candidates", [])
    ids = [c.get("id") for c in cands]
    if len(ids) != len(set(ids)):
        errors.append("duplicate candidate id")
    actives = [c for c in cands if c.get("state") == "active"]
    if len(actives) > 1:
        errors.append(f"one-active invariant violated: {len(actives)} active arms "
                      f"([{', '.join(c.get('id', '?') for c in actives)}])")

    for c in cands:
        cid = c.get("id", "?")
        if c.get("state") not in STATES:
            errors.append(f"{cid}: unknown state {c.get('state')!r}")
        for dim in LEVELS:
            if c.get(dim) not in LEVELS[dim]:
                errors.append(f"{cid}: {dim} must be one of {sorted(LEVELS[dim])}")
        if not isinstance(c.get("strikes"), int) or c["strikes"] < 0:
            errors.append(f"{cid}: strikes must be int >= 0")
        strike_limit = reg.get("falsification", {}).get("strike_limit", 0)
        if c.get("strikes", 0) >= strike_limit and c.get("state") not in {
                "terminal_falsified", "blocked"}:
            warnings.append(f"{cid}: strikes >= strike_limit but not terminal_falsified")
        if not isinstance(c.get("evidence_parts"), list):
            errors.append(f"{cid}: evidence_parts must be a list")
        if not isinstance(c.get("verdicts"), list):
            errors.append(f"{cid}: verdicts must be a list")
        decl = c.get("declaration", {})
        for f in DECLARATION_FIELDS:
            if f not in decl:
                errors.append(f"{cid}: declaration missing field: {f}")
        if decl.get("qualification_gate") not in KNOWN_GATES:
            errors.append(f"{cid}: qualification_gate {decl.get('qualification_gate')!r} "
                          "not in gate register (new gates must be registered+calibrated first)")
        cfg = decl.get("config", "")
        if cfg and not Path(cfg).is_file():
            warnings.append(f"{cid}: config {cfg} not on disk yet "
                            "(hard error in propose.py; registered seed may ship without it)")
        if decl.get("arm_issue") in (None, 0):
            warnings.append(f"{cid}: arm_issue=0 — issue created at proposal time "
                            "on github.com/timlawrenz/prx-tg")
        if not isinstance(decl.get("expected_gpu_hours"), (int, float)) or \
                decl.get("expected_gpu_hours", 0) <= 0:
            errors.append(f"{cid}: expected_gpu_hours must be a positive number")

    for w in warnings:
        print(f"WARN {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"ERROR {e}", file=sys.stderr)
        print(f"{len(errors)} error(s), {len(warnings)} warning(s)", file=sys.stderr)
        return 1
    print(f"OK registry valid: {len(cands)} candidates, champion={champ.get('slug')}, "
          f"{len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())