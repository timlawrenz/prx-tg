# Avenues Registry — prx-tg Research Loop

Part of Phase 1b (`../../.hermes/plans/2026-08-30_photorealism-gates-blind-review.md`).
Transplanted from the stratum-ffhq autonomous research harness — process, not build.

## Files

| File | Role |
|---|---|
| `registry.json` | Source of truth for arm candidates, champion, selection state. Docs lag it — registry wins. |
| `gates_calibration.json` | Frozen G0 band thresholds from the real-FFHQ reference set. Produced by M2; **frozen before the first verdict, never recomputed mid-sweep**. Absent at M1 (warning only). |
| `issues/` | Per-arm note files until GitHub issues are created (`arm_issue` = real issue number on github.com/timlawrenz/prx-tg). |

## Candidate lifecycle

```
registered → active → terminal_validated
                   → terminal_falsified   (strike_limit reached)
registered → blocked                     (policy/authority gate — never selected)
```

- Exactly 0 or 1 `active` arms at any time (one-active invariant).
- A NOT_BETTER verdict below the falsification limit records a strike and keeps
  the SAME arm active; the selector only moves on validate/falsify.
- Verdicts are written by `tick.py` from gate JSONL only — never hand-typed.

## Validation

```bash
.venv/bin/python3 scripts/harness/registry_validate.py research/avenues/registry.json
```

Exit 1 = must fix before any tick; warnings are advisory (configs not yet
frozen for seeded candidates, calibration pending at M1).