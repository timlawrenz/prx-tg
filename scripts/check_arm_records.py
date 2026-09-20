#!/usr/bin/env python3
"""check_arm_records.py — enforce the AGENTS.md §0 arm-record contract.

Deterministic verifier, no LLM. The skill's own rule: never let the model that
produced a result be the thing that certifies it.

Contract (AGENTS.md §0):
  - every slug arm on the NAS has a git-tracked soft trio at
    experiment-configs/{slug}/ (README.md, config.yaml, provenance.yaml)
  - provenance.yaml's `branch` and `tags` resolve in git
  - the canonical run_dir exists
  - `status: planned` arms have never run, so they owe no run_dirs (they still owe
    branch/tags/mode/gate) — never invent a run path to satisfy the checker
  - a concluded arm's `ledger_anchor` resolves in docs/EXPERIMENTS_AND_RESULTS.md
  - a concluded arm has a `blog_brief` that exists
  - soft files are NOT sitting on the NAS side (where git cannot see them)

Usage:
  python scripts/check_arm_records.py                  # full check
  python scripts/check_arm_records.py --allow-missing  # migration window:
                                                       # only broken links fail
  python scripts/check_arm_records.py --summary        # counts only
  python scripts/check_arm_records.py --json           # machine-readable

Exit 0 = contract satisfied. Non-zero = STOP; do not claim the arm is concluded.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"          # symlink -> NAS (hard)
CONFIGS = REPO / "experiment-configs"       # git (soft)
LEDGER = REPO / "docs" / "EXPERIMENTS_AND_RESULTS.md"
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}$")
CONCLUDED = {"GO", "GO-with-caveat", "PIVOT", "PARK", "KILL", "PASS", "FAIL"}
SOFT_TRIO = ("README.md", "config.yaml", "provenance.yaml")
REQUIRED_KEYS = ("arm", "mode", "hypothesis", "falsified_if", "pre_registered_gate")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def git_ok(*args: str) -> bool:
    return git(*args).returncode == 0


def tracked(path: Path) -> bool:
    try:
        rel = str(path.relative_to(REPO))
    except ValueError:
        return False
    return git_ok("ls-files", "--error-unmatch", rel)


def load_yaml(path: Path):
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required:  pip install pyyaml")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def is_arm_dir(d: Path) -> bool:
    return d.is_dir() and not TS_RE.match(d.name)


def scan():
    if not EXPERIMENTS.exists():
        sys.exit(f"experiments/ not found at {EXPERIMENTS}")
    arms = sorted(d.name for d in EXPERIMENTS.iterdir() if is_arm_dir(d))
    legacy = sorted(d.name for d in EXPERIMENTS.iterdir()
                    if d.is_dir() and TS_RE.match(d.name))
    return arms, legacy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true",
                    help="migration window: not-yet-migrated arms warn instead of fail")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    arms, legacy = scan()
    errors: list[str] = []
    missing: list[str] = []
    warns: list[str] = []
    passed: list[str] = []

    # --- structural guard: experiments/ must stay git-invisible -------------
    if not EXPERIMENTS.is_symlink():
        warns.append("experiments/ is not a symlink — soft files there could be "
                     "committed by accident; verify the .gitignore still excludes it")

    ledger_text = LEDGER.read_text() if LEDGER.exists() else ""
    if not ledger_text:
        errors.append(f"ledger not found or empty: {LEDGER.relative_to(REPO)}")

    record_slugs = []
    if CONFIGS.exists():
        record_slugs = sorted(p.name for p in CONFIGS.iterdir()
                              if p.is_dir() and p.name != ".git")

    # validate EVERY record, whether or not the arm has a NAS directory: an
    # inference-only / feasibility arm has no experiments/{slug}/ but still owes
    # a complete chain (this is how vae-ceiling went unchecked).
    for slug in sorted(set(arms) | set(record_slugs)):
        if slug not in arms:
            warns.append(f"{slug}: record exists in experiment-configs/ but no "
                         f"experiments/{slug}/ arm on the NAS (inference-only or stale "
                         f"record) — links validated below anyway")
        prov = CONFIGS / slug / "provenance.yaml"

        # 1. soft trio present in git?
        if not prov.exists():
            missing.append(slug)
            continue
        if not tracked(prov):
            errors.append(f"{slug}: {prov.relative_to(REPO)} exists but is NOT tracked by git")
            continue
        for name in SOFT_TRIO:
            f = CONFIGS / slug / name
            if not f.exists():
                errors.append(f"{slug}: soft trio incomplete — missing {name}")
            elif not tracked(f):
                errors.append(f"{slug}: {name} present but not tracked by git")

        # 2. regression detector: soft files left on the NAS side
        for name in SOFT_TRIO:
            if (EXPERIMENTS / slug / name).exists():
                warns.append(f"{slug}: {name} also exists under experiments/ (NAS) — "
                             f"git cannot see it; the git copy is authoritative")

        # 3. provenance content
        try:
            prov_data = load_yaml(prov)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{slug}: provenance.yaml does not parse — {exc}")
            continue

        for key in REQUIRED_KEYS:
            if not prov_data.get(key):
                errors.append(f"{slug}: provenance missing required key `{key}`")

        branch = prov_data.get("branch")
        if not branch:
            errors.append(f"{slug}: provenance missing `branch` — no experiment branch "
                          f"exists for this arm; create one, or record why it cannot")
        elif not git_ok("rev-parse", "--verify", branch):
            errors.append(f"{slug}: branch `{branch}` does not resolve in git")

        tags = prov_data.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if not git_ok("rev-parse", "--verify", f"refs/tags/{tag}"):
                errors.append(f"{slug}: tag `{tag}` does not resolve in git")

        # An arm declared `status: planned` has never been run, so it owes no run
        # dirs — inventing a path just to satisfy the checker would be fabrication.
        # It still owes branch/tags/mode/gate (checked above), and warns that the
        # gate must be frozen before the first launch.
        planned = str(prov_data.get("status") or "").lower() == "planned"
        run_dirs = prov_data.get("run_dirs") or []
        if isinstance(run_dirs, str):
            run_dirs = [run_dirs]
        if not run_dirs and not planned:
            errors.append(f"{slug}: provenance has no `run_dirs`")
        elif not run_dirs and planned:
            warns.append(f"{slug}: declared `status: planned` (never run) so it has no "
                         f"run_dirs — freeze `pre_registered_gate` before the first launch")
        for rd in run_dirs:
            if not (REPO / rd).exists():
                errors.append(f"{slug}: run_dir `{rd}` does not exist")

        # 4. concluded arms must close the chain
        verdict = str(prov_data.get("verdict") or "active")
        if verdict in CONCLUDED:
            anchor = prov_data.get("ledger_anchor")
            if not anchor:
                errors.append(f"{slug}: verdict `{verdict}` but no `ledger_anchor`")
            elif anchor not in ledger_text:
                errors.append(f"{slug}: ledger_anchor `{anchor}` not found in the ledger")

            brief = prov_data.get("blog_brief")
            if not brief:
                errors.append(f"{slug}: verdict `{verdict}` but no `blog_brief`")
            elif not (REPO / brief).exists():
                errors.append(f"{slug}: blog_brief `{brief}` does not exist")

            if not prov_data.get("conclusions"):
                errors.append(f"{slug}: verdict `{verdict}` but `conclusions` is empty")

        passed.append(slug)

    # --- report -------------------------------------------------------------
    if args.json:
        print(json.dumps({"errors": errors, "missing": missing, "warnings": warns,
                          "passed": passed, "legacy_dirs": len(legacy),
                          "arms": len(arms)}, indent=2))
    elif args.summary:
        print(f"arms on NAS: {len(arms)}   legacy timestamp dirs: {len(legacy)}")
        print(f"records ok: {len(passed)}   missing: {len(missing)}   errors: {len(errors)}")
    else:
        print(f"arm-record check — {len(arms)} slug arms, {len(legacy)} legacy dirs\n")
        if errors:
            print(f"ERRORS ({len(errors)}) — broken links, must be fixed:")
            for e in errors:
                print(f"  ✗ {e}")
            print()
        if missing:
            print(f"MISSING ({len(missing)}) — arm has no git-tracked record yet:")
            for m in missing:
                print(f"  · {m}")
            print()
        if warns:
            print(f"WARNINGS ({len(warns)}):")
            for w in warns:
                print(f"  ! {w}")
            print()
        if passed:
            print(f"OK ({len(passed)}): {', '.join(passed)}")
        if legacy:
            print(f"\nNOTE: {len(legacy)} legacy timestamp-named dirs under experiments/ "
                  f"(pre-governance sprawl). Map them to arms or mark them legacy.")

    if errors:
        return 1
    if missing and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
