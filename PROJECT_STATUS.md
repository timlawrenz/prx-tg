# Project Status — prx-tg

**Last updated:** 2026-09-21
**Phase / status:** PIVOT — Arm `pixel-posttrain` **CANCELLED (tainted)**, `latent-first-pretrain` **BLOCKED (tainted, never gated)**. New gated redesign approved in principle; Stage A (forensic baseline) starting.

## Current state

- **Arm PP cancelled as tainted (2026-09-21, Tim).** P1 had no official gate, and
  P1/PP were diagnosed as ignoring text conditioning while memorizing DINOv3 CLS
  tokens as per-image keys into the training set. PP run `2026-09-20_1413` stopped
  at step 2198/10000. Neither checkpoint may warm-start any future arm.
- **Structural grounding for the diagnosis:** CLS is a per-image unique key present
  ~45% of steps; text+CLS co-occur only ~40%; there is **no train/val/test split**
  (validation reconstructs training images, so recon LPIPS rewards memorization);
  the two instruments that would have caught it (`run_text_manip`, `run_dino_swap`)
  exist but were disabled in P1's config.
- **Plan v2 approved in principle** (`.hermes/plans/2026-09-22_0000-p1-pp-gated-redesign.md`):
  Stage A forensic baseline → Stage B data prep + train/val/never-touched-test split
  → Stage C gated P1 (optional dropout-profile ablation) → Stage D gated PP →
  Stage E eidolon renderer.
- **Decisions settled:** experiments stay at 10k scale (50k reserved for finals);
  hegre split to be **vertical (person-level holdout)** when it binds at Stage E;
  hegre excluded from all P1/PP training.

## Immediate blockers / next action

1. **Stage A forensic baseline (running now):** A1 text-manip + A2 dino-swap +
   A4 novel-CLS on P1-10k and Arm J-35k; A3 memorization probe (new script).
2. Record Stage A verdict in the ledger; numbers become the baselines for
   Stage C/D gates.
3. Stage B: seeded 68k/1k/1k split manifest + fixed probe panel (git artifacts).

## Headline result so far

Dip-conv-head 10k: GO-with-caveat (LPIPS 0.7251, blind win-rate 0.846; G0
photorealism NOT validated — strike 1/3). Photorealism still un-met.
The P1/PP taint verdict now supersedes the warm-start path: the next backbone
must come from a **gated** P1.
