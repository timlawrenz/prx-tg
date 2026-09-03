# Project Status — prx-tg

**Last updated:** 2026-09-02
**Phase / status:** Phase 1b — Arm `dip-conv-head` 10k **COMPLETE, verdict GO-with-caveat** (criteria a/b/d PASS; G0 criterion NOT_VALIDATED → strike 1/3). Review workflow now has a web UI. Next: decide champion promotion / next arm via tick.

## Current state

- **Arm `dip-conv-head` 10k DONE — verdict GO-with-caveat.**
  LPIPS 0.7251 (≤0.746 ✓), loss ~0.011 (≤0.0135 ✓), blind 10k-vs-5k win-rate 0.846
  (CI LB 0.5776 > 0.5 ✓), calibration 1.0 ✓. FaceConf 0.488→0.707 (+45%).
  Clean run, zero NaN, `checkpoint_step0010000.pt` saved.
  **Caveat — G0 NOT_VALIDATED:** 3/6 G0 gates out-of-band at 10k (g0a noise floor,
  g0d p50/p95 local contrast, all below frozen real-FFHQ band) → strike 1/3 in
  registry (f6c079b). Photorealism still un-met.
- **Next arm ACTIVE:** `gamma2-noise-scale` training on GPU (tmux `gamma2-10k`,
  started 2026-09-02 ~16:16, log /tmp/gamma2_10k.log) — tick-selected to attack
  the g0a failing gate.
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
