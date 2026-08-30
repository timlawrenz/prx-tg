#!/usr/bin/env python3
"""Blind-review aggregator (Phase 1).

Reads a pool.json + one or more votes.jsonl (one per rater) and emits:
  - win rates for model B vs A on ab pairs (the headline the tick consumes)
  - Wilson CI lower bound (reusing tick.wilson_lb — same math, one source)
  - CALIBRATION SELF-CHECK: real photos must be chosen as real in >=95% of
    calibration pairs, or the whole session is invalid and discarded
  - per-rater stats + pairwise agreement
Writes tick-consumable votes.json: {"wins", "losses", "ties", ...}.

USAGE:
    .venv/bin/python3 scripts/harness/review_aggregate.py \
        --pool <pool.json> --votes <v1.jsonl> [--votes <v2.jsonl> ...] \
        --out research/avenues/votes/<arm>/votes.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick import wilson_lb  # noqa: E402

CALIBRATION_MIN_REAL_WIN_RATE = 0.95


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--votes", required=True, action="append", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    pool = json.loads(args.pool.read_text())
    pair_map = {p["pair_id"]: p for p in pool["pairs"]}

    votes: list[dict] = []
    raters = set()
    for vp in args.votes:
        for line in vp.read_text().strip().splitlines():
            v = json.loads(line)
            if v["pair_id"] not in pair_map:
                print(f"ERROR unknown pair_id {v['pair_id']}", file=sys.stderr)
                return 2
            votes.append(v)
            raters.add(v.get("rater", "?"))
    if not votes:
        print("ERROR no votes", file=sys.stderr)
        return 2

    # Calibration self-check first: real must win >=95% where present
    cal_n = 0
    cal_real_wins = 0
    for v in votes:
        p = pair_map[v["pair_id"]]
        if p["kind"] != "calibration":
            continue
        cal_n += 1
        left_real = p["left_is_real"]
        chose_real = (v["choice"] == "L" and left_real) or (v["choice"] == "R" and not left_real)
        cal_real_wins += int(chose_real)
    cal_rate = cal_real_wins / cal_n if cal_n else 1.0
    calibration_ok = cal_rate >= CALIBRATION_MIN_REAL_WIN_RATE

    # Headline: B wins over A on ab pairs (ties count half)
    wins = losses = ties = 0
    for v in votes:
        p = pair_map[v["pair_id"]]
        if p["kind"] != "ab":
            continue
        b_left = p["left_model"] == "b"
        chose_left = v["choice"] == "L"
        chose_b = (chose_left and b_left) or (not chose_left and p["right_model"] == "b")
        if v["choice"] == "T":
            ties += 1
        elif chose_b:
            wins += 1
        else:
            losses += 1
    n_ab = wins + losses + ties
    lb = wilson_lb(wins + 0.5 * ties, n_ab)

    # Per-rater agreement on overlapping ab pairs (raw %)
    by_rater: dict[str, dict] = {}
    for v in votes:
        if pair_map[v["pair_id"]]["kind"] != "ab":
            continue
        r = by_rater.setdefault(v.get("rater", "?"), {"L": 0, "R": 0, "T": 0})
        r[v["choice"]] += 1
    agreement = None
    rids = sorted(by_rater)
    if len(rids) >= 2:
        agree = total = 0
        for v0 in votes:
            if pair_map[v0["pair_id"]]["kind"] != "ab":
                continue
            for v1 in votes:
                if v1["rater"] != v0["rater"] and v1["pair_id"] == v0["pair_id"]:
                    agree += int(v0["choice"] == v1["choice"])
                    total += 1
        agreement = round(agree / total, 4) if total else None

    result = {
        # tick-consumable shape:
        "wins": wins, "losses": losses, "ties": ties,
        "n_pairs": n_ab,
        "wilson_lb": round(lb, 4),
        "win_rate_b": round((wins + 0.5 * ties) / n_ab, 4) if n_ab else None,
        "calibration": {
            "n": cal_n, "real_win_rate": round(cal_rate, 4),
            "min_required": CALIBRATION_MIN_REAL_WIN_RATE,
            "pass": calibration_ok,
        },
        "raters": sorted(raters),
        "per_rater": {r: dict(v) for r, v in by_rater.items()},
        "inter_rater_agreement": agreement,
        "session_valid": calibration_ok,
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    print(json.dumps(result, indent=2))
    print(f"session_valid={calibration_ok} wilson_lb={lb:.4f} -> {out}", file=sys.stderr)
    return 0 if calibration_ok else 4


if __name__ == "__main__":
    sys.exit(main())