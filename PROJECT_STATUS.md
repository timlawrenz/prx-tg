# Project Status — prx-tg

**Last updated:** 2026-09-01
**Phase / status:** ACTIVE — Phase 1b Research Loop. Arm `dip-conv-head` in 10k continuation (~74%, ETA ~10h). Blind-review + QC workflow automated via GitHub issues + cron agent.

## Current state

- **Arm `dip-conv-head` 10k continuation RUNNING** (tmux `dip-10k`, resumed from
  `checkpoint_step0005000.pt`, budget extended 5000→10000). Loss 0.0113 (vs 0.0135
  at 5k — still improving), GPU 87%, ETA ~10h. 5k run completed clean: loss 0.0135,
  LPIPS 0.746, zero NaN, all instrumentation fired.
- **QC fixes shipped + verified** (commit 22b9b67): eval loaders now pass `head_type`
  (dip) so quality-metrics works; adapter mask-length regression fixed + 2 unit tests;
  resume provenance records git commit. Quality metrics re-run on step-5000 OK
  (aesthetic 4.64, clip 0.107, faceconf 0.488).
- **Blind review started.** Batch 1: dip-5k beat old May baseline 8–0 (low bar —
  baseline near-blob, pool retired as uninformative). Key finding: `text_only` eval
  is frontal-only by construction (no pose input); `dino_swap` mode DOES show pose
  variation. 10k review will use temporal 5k-vs-10k comparison.
- **10k gate PRE-REGISTERED** (provenance.yaml, 2026-09-01, before results):
  PASS if blind win-rate CI LB > 0.5 (10k vs 5k) AND LPIPS 10k <= 0.746 AND no G0
  regression AND loss 10k <= 0.0135.
- **Autonomous workflow wired:** GitHub issues #2 (QC) → #3 (blind review, human
  gate) → #4 (ledger verdict). Cron agent `prx-tg-dip-10k-research-agent` created
  (every 30m) — NOTE: gateway down, will fire once `hermes gateway start`.
  tmux watcher `dip-watch` builds the 10k review pool automatically on completion.

## Immediate blockers / next action

1. Wait for 10k training to finish (~10h). Watcher auto-builds review pool.
2. When done: run QC (issue #2), then tim votes on 10k-vs-5k pairs (issue #3,
   human gate), then ledger verdict (issue #4).
3. Gateway must be started for the cron agent to drive the chain:
   `hermes gateway start` (or rely on tmux watcher + manual).

## Headline result so far

- dip-conv-head 5k: clean convergent run, faces structurally correct but painterly —
  **not photorealistic** (faceconf 0.488, aesthetic 4.64). 10k gate pre-registered.
- No checkpoint yet produces photo-realistic portraits — photorealism remains the
  primary success criterion.
