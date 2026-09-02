# Project Status — prx-tg

**Last updated:** 2026-09-01 (evening)
**Phase / status:** Phase 1b — Arm `dip-conv-head` 10k **COMPLETE, verdict GO** (all 4 pre-registered gates PASS). Review workflow now has a web UI. Next: decide champion promotion / next arm via tick.

## Current state

- **Arm `dip-conv-head` 10k DONE — verdict GO (PASS all 4 gates).**
  LPIPS 0.7251 (≤0.746 ✓), loss ~0.011 (≤0.0135 ✓), blind 10k-vs-5k win-rate 0.846
  (CI LB 0.5776 > 0.5 ✓), calibration 1.0 ✓. FaceConf 0.488→0.707 (+45%).
  Clean run, zero NaN, `checkpoint_step0010000.pt` saved.
- **Blind-review web UI shipped** (`scripts/harness/review_server.py`, 0.0.0.0:8765):
  neutral-URL A/B picker, idempotent votes.jsonl, reusable for any pool.
- **QC fixes** (commit 22b9b67): eval loaders head_type parity, adapter mask-length
  regression + 2 tests, resume provenance git fields.
- **GitHub issue chain**: #2 (QC) #3 (blind review) #4 (ledger verdict) — ledger
  entry written. Cron agent `prx-tg-dip-10k-research-agent` ready (gateway-gated).
- **Remaining true gap: photorealism.** 10k renders are sharply improved but still
  painterly — the primary success criterion is still un-met.

## Immediate blockers / next action

1. Promote `dip-conv-head` 10k to champion? (tick decision / user call)
2. Run next avenue candidate via tick (registry: gamma2-noise-scale, irepa-upgrade, ...)
3. For autonomous drive: `hermes gateway start` (cron agent + issue chain dormant otherwise)
