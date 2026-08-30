#!/usr/bin/env python3
"""Blind-review vote recorder (Phase 1).

Human rater CLI: shows each pair's rendered PNG path (open it yourself in any
image viewer), asks Left/Right/Tie/Skip. The JSON ground truth in pool.json is
NEVER printed — blindness by construction. Appends votes.jsonl (idempotent per
pair_id+rater: re-running overwrites that pair's vote).

USAGE (one run per rater per block):
    .venv/bin/python3 scripts/harness/review_vote.py \
        --pool research/avenues/review/realism/pool.json \
        --rater tim --out research/avenues/votes/<arm>/tim.jsonl
"""
import argparse
import json
import sys
from pathlib import Path


def ask(pair) -> str:
    print(f"\n{pair['image']}")
    print("[L]eft  [R]ight  [T]ie  [S]kip")
    while True:
        sys.stdout.write("> ")
        sys.stdout.flush()
        raw = sys.stdin.readline().strip().upper()
        if raw in ("L", "R", "T", "S"):
            return raw
        print("answer L/R/T/S")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--rater", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    pool = json.loads(args.pool.read_text())
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.is_file():
        for line in out.read_text().strip().splitlines():
            v = json.loads(line)
            existing[v["pair_id"]] = v

    with open(out, "a") as f:
        for pair in pool["pairs"]:
            if pair["pair_id"] in existing:
                print(f"skip {pair['pair_id']} (already voted)")
                continue
            choice = ask(pair)
            if choice == "S":
                print("skipped")
                continue
            rec = {"pair_id": pair["pair_id"], "rater": args.rater, "choice": choice}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"recorded: {choice}")
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())