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
- **Research-loop M1 shipped:** `research/avenues/registry.json` (7 seeded candidates),
  `scripts/harness/registry_validate.py`, plan at
  `.hermes/plans/2026-08-30_photorealism-gates-blind-review.md` (Phase 1 + 1b).
- Stratum2 FFHQ enrichment continues on the 4090 (pose2/seg2 ~4% coverage); the GPU
  is NOT free — training arms need explicit greenlight.

## Immediate next action

M2: gate calibration pipeline — real-FFHQ reference-set stats → frozen
`research/avenues/gates_calibration.json`. Then M3: `scripts/harness/tick.py`
state machine + unit tests. Both pure-files, no GPU.

## Headline result so far

- Best-scoring architecture for generation quality: no-DINO-patches (Arm M, 7k validation).
- Arm O: yaw control proven stable (first and only controlled-pose PASS).
- Photorealism: **0 passes** — the deficit report (post-M2) will be the first measured baseline.

## Key open questions

1. Do the G0 photorealism bands separate real FFHQ from current model output? (M2 deficit report)
2. What is the γ=2 noise-scale arm's band movement? (first tick candidate, Phase 2)
3. Blind-review rater pool + eval suite size — still open (plan §Open decisions).
4. Enrichment priority: caption-quality pass (VLM spatial-facts) vs pose2 completion — recommendation stands: caption-first for photorealism.