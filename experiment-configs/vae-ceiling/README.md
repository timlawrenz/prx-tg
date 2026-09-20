# vae-ceiling

**Status:** retrospective record — run 2026-09-10, recorded 2026-09-19. `verdict: pending`.

## Hypothesis

The FLUX AE (`/mnt/models/vae/ae.safetensors`, the AE that produced the precomputed
`AbstractPhil/ffhq_flux_latents_repaired` set) reconstructs real FFHQ faces faithfully
enough that a latent-first model could reach photorealism *without* a pixel-space
post-training phase.

## Falsified if

Encode→decode of a **real** photo through the AE fails the photoreal band — i.e. the
reconstruction is visibly washed out, low-contrast or mushy. In that case the
autoencoder itself, not the latent model, is the binding constraint on the ceiling.

## What it found

**Falsified.** The AE at its best scale reconstructs a recognizable but clearly
non-photoreal face: luminance lifted 0.44 → 0.86, contrast roughly halved. The
precomputed `_repaired` latent set is additionally **512², not 1024²**, so it cannot
support a 1024² latent-first model at all.

See `research/results/vae_ceiling/RESULTS.md` for the full write-up, the corrected
summary table (2026-09-19), and the metric jsonl.

## Why this arm exists (differs-from)

It is not an ablation — it is a **feasibility gate** run *before* committing GPU time to
`latent-first-pretrain`. It exists because the precomputed latent set's true resolution
was unknown and its reconstruction quality unmeasured. It is the reason
`latent-first-pretrain` encodes latents at 1024² itself and why the **pixel post-training
phase (Part 2) is first-class rather than optional**.

## Record status — read this before trusting the fields

This arm predates the AGENTS.md §0 contract (it was run 2026-09-10), so several required
links were never created. Where a link is still missing it is recorded as an **absent
fact**, not filled in retroactively; the two added on 2026-09-19 are marked as such:

| Field | State |
|---|---|
| `branch` | `exp/vae-ceiling` — created 2026-09-19 (seeded from main) as the home for brief 05; the arm itself ran with none |
| `tags` | `arm/vae-ceiling/registered` — created 2026-09-19 on `exp/vae-ceiling` |
| `ledger_anchor` | **none** — this arm has no entry in `docs/EXPERIMENTS_AND_RESULTS.md` |
| `pre_registered_gate` | **never pre-registered** — the numbers came first |
| `mode` | **undeclared** — no confirmatory/exploratory mode was set before the run |
| `git_commit` | **unknown** — no commit was recorded with the run |
| metrics | real, computed by `scripts/vae_ceiling_test.py`; two summary figures were mislabeled and corrected 2026-09-19 |

`python scripts/check_arm_records.py` now passes for this arm: the branch resolves and the
tag exists. What remains open is the `queued_rerun` work in `provenance.yaml` — the script
fix, the real-photo `lum`/`contrast` capture, the re-run, and the ledger entry that would
move the verdict off `pending`.