# Project Status — prx-tg

**Last updated:** 2026-08-30
**Phase / status:** ACTIVE — Phase 1b Research Loop (photorealism gates + blind review + avenue registry). No training arms active.

## Current state

- **Photorealism is the yardstick.** No checkpoint yet produces photo-realistic
  portraits. All future arms are judged by deterministic G-gates versus a real-FFHQ
  reference distribution plus a blind human-preference review — not eyeball collages.
- **Arm O (`zg-token-basis-cfg-guard`) recorded PASS** (2026-08-30, close-out of the
  2026-07-17 pivot): monotonic dim0→yaw at ≥2 checkpoints, no collapse through 5k
  steps. Ledger updated; release checkpoint at
  `release/prx_tg_armo_cfgguard_step5000_bf16.safetensors`.
- **Research-loop M1–M3 shipped (2026-08-30):** avenues registry (7 candidates) +
  validator; G0 gate calibration frozen from 1,000 real FFHQ images
  (`research/avenues/gates_calibration.json`); deterministic tick state machine
  with 18/18 unit tests. Selection demo: first tick activates `dip-conv-head`
  (id-tiebreak vs `gamma2-noise-scale`, both EIG 1.1).
- Stratum2 FFHQ enrichment continues on the 4090 (pose2/seg2 ~4% coverage); the GPU
  is NOT free — training arms need explicit greenlight.

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