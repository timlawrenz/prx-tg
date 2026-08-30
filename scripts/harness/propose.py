#!/usr/bin/env python3
"""M4 — gated idea registration for the research-loop avenue registry.

Mirrors the stratum-ffhq harness's `propose-dimensions`: each candidate needs
the full declaration, and hard rejection rules protect the registry from
unpayable arms. Verdict objects come from THIS script — never hand-edited.

USAGE:
    .venv/bin/python3 scripts/harness/propose.py \
        --registry research/avenues/registry.json \
        --candidates <candidates.json> \
        [--require-new-evidence-part] [--write]

Rejection rules (each a hard error → exit 1, nothing written):
  1. missing any declaration field (scope/output_semantics/provenance/
     abstention/qualification_gate/expected_gpu_hours/config/arm_issue/differs_from)
  2. config file missing on disk (AGENTS.md: named config, frozen into experiment dir)
  3. qualification_gate not in the shared gate register (a trick is payable only
     if an existing gate can measure it; new gates must be registered +
     calibrated FIRST)
  4. duplicate / pre-existing candidate id
  5. --require-new-evidence-part AND no NEW evidence part (nothing beyond what
     terminal_validated arms already established)
  6. malformed enum (prior/measurability/cost) or non-positive expected_gpu_hours
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_register import DECLARATION_FIELDS, KNOWN_GATES, LEVELS  # noqa: E402


def rejections(candidate: dict, registry: dict, require_new: bool) -> list[str]:
    reasons: list[str] = []
    cid = candidate.get("id", "?")
    if not cid or cid == "?":
        reasons.append("missing id")
    if any(c["id"] == cid for c in registry.get("candidates", [])):
        reasons.append(f"id already exists: {cid}")
    for dim in ("prior", "measurability", "cost"):
        if candidate.get(dim) not in LEVELS[dim]:
            reasons.append(f"{cid}: {dim} must be one of {sorted(LEVELS[dim])}")
    decl = candidate.get("declaration", {})
    if not isinstance(decl, dict):
        reasons.append(f"{cid}: declaration must be an object")
        return reasons
    for f in DECLARATION_FIELDS:
        if f not in decl:
            reasons.append(f"{cid}: declaration missing field: {f}")
    gate = decl.get("qualification_gate")
    if gate not in KNOWN_GATES:
        reasons.append(f"{cid}: qualification_gate {gate!r} not in gate register")
    cfg = decl.get("config", "")
    if not cfg or not Path(cfg).is_file():
        reasons.append(f"{cid}: config {cfg!r} not on disk")
    if not isinstance(decl.get("expected_gpu_hours"), (int, float)) or \
            decl.get("expected_gpu_hours", 0) <= 0:
        reasons.append(f"{cid}: expected_gpu_hours must be a positive number")
    if candidate.get("state", "registered") != "registered":
        reasons.append(f"{cid}: new candidates must enter as 'registered'")
    if require_new:
        established = set()
        for c in registry.get("candidates", []):
            if c.get("state") == "terminal_validated":
                established |= set(c.get("evidence_parts", []))
        if not (set(candidate.get("evidence_parts", [])) - established):
            reasons.append(f"{cid}: no NEW evidence part (require-new-evidence-part)")
    return reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--require-new-evidence-part", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    original = args.registry.read_bytes()
    registry = json.loads(original.decode())
    incoming = json.loads(args.candidates.read_text())
    if not isinstance(incoming, list):
        incoming = [incoming]

    report = {"accepted": [], "rejected": []}
    for cand in incoming:
        reasons = rejections(cand, registry, args.require_new_evidence_part)
        if reasons:
            report["rejected"].append({"id": cand.get("id", "?"), "reasons": reasons})
        else:
            # normalize into registry shape
            cand.setdefault("state", "registered")
            cand.setdefault("strikes", 0)
            cand.setdefault("evidence_parts", [])
            cand.setdefault("verdicts", [])
            registry["candidates"].append(cand)
            report["accepted"].append(cand.get("id"))

    print(json.dumps(report, indent=2))

    if report["rejected"]:
        print(f"{len(report['rejected'])} rejected, {len(report['accepted'])} accepted",
              file=sys.stderr)
        return 1
    if args.write:
        if hashlib.sha256(args.registry.read_bytes()).hexdigest() != \
                hashlib.sha256(original).hexdigest():
            print("ERROR registry changed on disk mid-proposal — refusing write",
                  file=sys.stderr)
            return 3
        tmp = args.registry.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registry, indent=2))
        tmp.replace(args.registry)
        print(f"wrote {len(report['accepted'])} candidate(s) to {args.registry}",
              file=sys.stderr)
    else:
        print(f"dry-run: {len(report['accepted'])} candidate(s) would be accepted",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())