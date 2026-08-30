#!/usr/bin/env python3
"""M3 — deterministic research-loop tick (stratum-ffhq harness pattern).

One tick = load registry → check one active arm against frozen gate calibration
+ gate JSONL (+ blind-review votes) → strike / falsify / validate → select next
arm by EIG arithmetic → atomically write the registry → emit a verdict object.

The LLM/agent NEVER computes verdicts: it relays this script's JSON output.
USAGE:
    .venv/bin/python3 scripts/harness/tick.py \
        --registry research/avenues/registry.json \
        --gates <gates-output-dir> [--votes <votes.jsonl>] [--write]

Without --write the tick is a dry run (verdict computed, nothing written).
Exit codes: 0 ok (incl. no-op), 2 input error, 3 registry mutated mid-tick.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

PRIOR_W = {"high": 1.0, "med": 0.6, "low": 0.2}
MEAS_W = {"high": 1.0, "med": 0.6, "low": 0.3}
COST_P = {"low": 0.0, "med": 0.15, "high": 0.35}
STRIKE_W = 0.45
ACTIONABLE = {"registered", "active"}
TERMINAL = {"terminal_validated", "terminal_falsified"}


def wilson_lb(wins: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half


def in_band(value: float, band: dict) -> bool:
    return band["p5"] <= value <= band["p95"]


def score_table(registry: dict, established_parts: set) -> list[dict]:
    """EIG scores for all actionable candidates; sorted (-eig, id) — fully deterministic."""
    novelty = registry["exploration"]["novelty_bonus"]
    cands = [c for c in registry["candidates"] if c["state"] in ACTIONABLE]
    table = []
    for c in cands:
        base = PRIOR_W[c["prior"]] * MEAS_W[c["measurability"]] \
            - COST_P[c["cost"]] - c.get("strikes", 0) * STRIKE_W
        bonus = novelty if set(c.get("evidence_parts", [])) - established_parts else 0.0
        table.append({"id": c["id"], "base": round(base, 4),
                      "novelty_bonus": round(bonus, 4),
                      "eig": round(base + bonus, 4),
                      "strikes": c.get("strikes", 0),
                      "prior": c["prior"], "cost": c["cost"]})
    table.sort(key=lambda r: (-r["eig"], r["id"]))
    return table


def select_next(registry: dict) -> tuple[str | None, str, list[dict]]:
    """(chosen_id | None, selected_via, score_table). Deterministic; ties by id."""
    established = set()
    for c in registry["candidates"]:
        if c["state"] == "terminal_validated":
            established |= set(c.get("evidence_parts", []))
    table = score_table(registry, established)
    if not table:
        return None, "none", []
    every_n = registry["exploration"]["every_n"]
    index = int(registry.get("selection_progress", 0))
    explore_slot = every_n > 0 and (index + 1) % every_n == 0
    if explore_slot:
        # ε-greedy: force the lowest-prior actionable proposal; ties by id
        cands = [c for c in registry["candidates"] if c["state"] in ACTIONABLE]
        chosen = min(cands, key=lambda c: (PRIOR_W[c["prior"]], c["id"]))
        return chosen["id"], "explore", table
    return table[0]["id"], "exploit", table


def eval_gates(gates_path: Path, calib: dict) -> dict:
    """Read gates JSONL, return {gate_id: {"value":.., "in_band":..}} plus meta."""
    if not gates_path.is_file():
        raise ValueError(f"gates JSONL missing: {gates_path}")
    out = {}
    for line in gates_path.read_text().strip().splitlines():
        rec = json.loads(line)
        gid = rec["gate_id"]
        band = calib["gates"].get(gid)
        if band is None:
            raise ValueError(f"gate {gid} not present in calibration (unregistered gate)")
        out[gid] = {"value": rec["value"], "n": rec.get("n"),
                    "in_band": in_band(rec["value"], band),
                    "band": [band["p5"], band["p95"]]}
    return out


def load_votes(votes_path: Path) -> dict | None:
    if votes_path is None or not votes_path.is_file():
        return None
    return json.loads(votes_path.read_text())


def verdict_for(arm: dict, g: dict, votes: dict | None, registry: dict) -> dict:
    gate = arm["declaration"]["qualification_gate"]
    out_of_band = {k: v for k, v in g.items() if not v["in_band"]}

    if gate == "phase4_gated":
        return {"kind": "needs_human", "reason": "phase4_gated requires owner ruling"}
    if gate == "stability_only":
        return {"kind": "needs_human", "reason": "stability gate thresholds not calibrated"}
    if gate == "band_containment":
        return {"kind": "validated" if not out_of_band else "not_better",
                "out_of_band": sorted(out_of_band)}

    # default_champion_gate (and anything else): gates first, then blind review
    if out_of_band:
        # Gate failure is the arm's fault → strikes regardless of vote state.
        return {"kind": "not_better", "out_of_band": sorted(out_of_band)}
    if votes is None:
        return {"kind": "needs_human", "reason": "blind-review votes required, none provided",
                "out_of_band": sorted(out_of_band)}
    n = votes["wins"] + votes["losses"] + votes["ties"]
    w_eff = votes["wins"] + 0.5 * votes["ties"]
    lb = wilson_lb(w_eff, n)
    passes_votes = n > 0 and lb > 0.5
    champion_measured = bool(registry["champion"].get("measured"))
    if not passes_votes:
        return {"kind": "not_better", "wilson_lb": round(lb, 4),
                "note": "blind win-rate CI lower bound <= 0.5"}
    # validated (bootstrap: first measured champion waives the >=1 gain clause)
    return {"kind": "validated", "wilson_lb": round(lb, 4),
            "bootstrap_waived_gain": not champion_measured}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--gates", type=Path, help="gates.jsonl file (or dir containing it)")
    ap.add_argument("--votes", type=Path)
    ap.add_argument("--checkpoint", default=None,
                    help="path of the checkpoint this verdict was measured on (recorded in champion)")
    ap.add_argument("--calibration", default=None, type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    reg_path = args.registry
    original = reg_path.read_bytes()
    registry = json.loads(original.decode())

    cal_path = args.calibration or Path(registry.get("calibration_file", ""))
    if not cal_path.is_file():
        print("ERROR frozen calibration missing — M2 artifact required before any verdict",
              file=sys.stderr)
        return 2
    calib = json.loads(cal_path.read_text())

    actives = [c for c in registry["candidates"] if c["state"] == "active"]
    if len(actives) > 1:
        print("ERROR one-active invariant violated before tick", file=sys.stderr)
        return 2

    verdict = {"registry": str(reg_path), "write": args.write}

    if not actives:
        # Nothing active → selection path (activate or no-op)
        if not any(c["state"] == "registered" for c in registry["candidates"]):
            verdict.update({"next_action": "none", "note": "no actionable arms"})
        else:
            chosen, via, table = select_next(registry)
            registry["selection_progress"] = int(registry.get("selection_progress", 0)) + 1
            for c in registry["candidates"]:
                if c["id"] == chosen:
                    c["state"] = "active"
            verdict.update({"next_action": "activate", "activated": chosen,
                            "selected_via": via, "score_table": table})
    else:
        arm = actives[0]
        gates_path = args.gates
        if gates_path is None:
            print("ERROR --gates required when an arm is active", file=sys.stderr)
            return 2
        if gates_path.is_dir():
            gates_path = gates_path / "gates.jsonl"
        g = eval_gates(gates_path, calib)
        votes = load_votes(args.votes)
        v = verdict_for(arm, g, votes, registry)
        limit = registry["falsification"]["strike_limit"]
        arm["verdicts"].append({**v, "gates": {k: vv["value"] for k, vv in g.items()}})

        if v["kind"] == "needs_human":
            verdict.update({"next_action": "needs_human", "arm": arm["id"], "reason": v["reason"]})
        elif v["kind"] == "validated":
            arm["state"] = "terminal_validated"
            registry["champion"] = {
                "slug": arm["id"],
                "checkpoint": args.checkpoint or registry["champion"]["checkpoint"],
                "measured": {k: vv["value"] for k, vv in g.items()},
            }
            verdict.update({"next_action": "validate_then_select", "arm": arm["id"],
                            "verdict": v})
            chosen, via, table = select_next(registry)
            if chosen:
                registry["selection_progress"] = int(registry.get("selection_progress", 0)) + 1
                for c in registry["candidates"]:
                    if c["id"] == chosen:
                        c["state"] = "active"
                verdict.update({"activated": chosen, "selected_via": via,
                                "score_table": table})
        else:  # not_better → strike
            arm["strikes"] += 1
            verdict.update({"next_action": "research_pending", "arm": arm["id"],
                            "strike": arm["strikes"], "verdict": v})
            if arm["strikes"] >= limit:
                arm["state"] = "terminal_falsified"
                verdict["next_action"] = "falsified_then_select"
                chosen, via, table = select_next(registry)
                if chosen:
                    registry["selection_progress"] = int(registry.get("selection_progress", 0)) + 1
                    for c in registry["candidates"]:
                        if c["id"] == chosen:
                            c["state"] = "active"
                    verdict.update({"activated": chosen, "selected_via": via,
                                    "score_table": table})

    # Atomic write guard: refuse if the registry mutated mid-tick
    if args.write:
        now = reg_path.read_bytes()
        if hashlib.sha256(now).hexdigest() != hashlib.sha256(original).hexdigest():
            print("ERROR registry changed on disk mid-tick — refusing write", file=sys.stderr)
            return 3
        tmp = reg_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registry, indent=2))
        tmp.replace(reg_path)

    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())