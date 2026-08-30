# Project Status — prx-tg

**Last updated:** 2026-08-30
**Phase / status:** ACTIVE — Phase 1b Research Loop (photorealism gates + blind review + avenue registry). No training arms active.

## Current state

- **Arm `dip-conv-head` RUNNING** (`experiments/dip-conv-head/runs/2026-08-30_1449`,
  FP8, effective batch 256, 5,000 steps). Launch blockers fixed en route:
  pose2/133 joint-count collision, mixed-enrichment collate KeyError, and
  stream-gated loader (~28MB→~7MB per sample). Verdict via tick when done.
- **Photorealism is the yardstick.** No checkpoint yet produces photo-realistic
  portraits. All future arms judged by deterministic G-gates vs the real-FFHQ
  calibration + blind human review — not eyeball collages.
- **Arm O (`zg-token-basis-cfg-guard`) PASS** — monotonic dim0→yaw at ≥2
  checkpoints, no collapse through 5k. Release checkpoint in `release/`.
- **Research-loop M1–M3 + M4-adjacent shipped:** avenues registry + tick +
  calibration + gate producer (`scripts/harness/`). Calibration re-frozen
  2026-08-30 with mask-free gate variants (g0a/g0b/g0c/g0d `_full`).
- Stratum2 FFHQ enrichment ~4% (pose2/seg2); captions: `caption.txt`+`t5_hidden`
  only — caption2/t52 are unreliable stratum2 artifacts (do not use).

## Immediate next action

M4: `propose.py` gated idea registration. Then Phase 1: `run_gates.py` producer
+ blind-review voting tool → first photorealism deficit report against the release
checkpoints. All pure-files/no-GPU until an arm is greenlit.

## Headline result so far

- Best-scoring architecture for generation quality: no-DINO-patches (Arm M, 7k validation).
- Arm O: yaw control proven stable (first and only controlled-pose PASS).
- Photorealism: **0 passes** — the deficit report (post-M2) will be the first measured baseline.

## Key open questions

1. Do the G0 photorealism bands separate real FFHQ from current model output? (M2 deficit report)
2. What is the γ=2 noise-scale arm's band movement? (first tick candidate, Phase 2)
3. Blind-review rater pool + eval suite size — still open (plan §Open decisions).
4. Enrichment priority: caption-quality pass (VLM spatial-facts) vs pose2 completion — recommendation stands: caption-first for photorealism.