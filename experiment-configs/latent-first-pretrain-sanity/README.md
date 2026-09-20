# Arm: latent-first-pretrain-sanity

**Status:** control harness for the deterministic validation seeding (2026-09-20) ·
**verdict: active** · gate frozen before the run (see provenance).

## What this arm is for

This is not a quality arm — it is an **instrumentation and reproducibility harness**
for the latent-first path (FLUX-AE latent mode end-to-end: latent dataloader,
`in_channels=16` model, latent decode via `production/flux_ae.py`, sampler and
validation in latent space).

It exists to answer one question, and it now has a second job:

1. **Original job (2026-09-12):** a 60-step lifecycle check that every instrumented
   hook actually fires inside the run window — `visual_debug` at 20/40/60,
   validation at 30/60, checkpoints at 30/60. This exists because a 50-step sanity
   run once missed an interval-250 debug hook and died at step 250.
2. **New job (2026-09-20):** serve as the **control** for the deterministic
   validation seeding. Validation fires at 30 *and* 60, with a checkpoint at each, so
   the same checkpoint can be validated in-train and then re-run standalone. If the
   standalone re-run does not reproduce the in-train numbers, the seeding change is
   rejected and reverted.

## Hypothesis

The deterministic validation seeding added 2026-09-20 (`VALIDATION_NOISE_SEED_BASE +
sample index`, drawn from a **local** `torch.Generator`) makes in-train validation
results reproducible by a standalone re-run against the same checkpoint, without
perturbing the training RNG stream.

## Why this matters (the finding that motivated it)

The `latent-first-pretrain` arm's per-step validation LPIPS values were **not
seed-controlled**. `torch.manual_seed` appeared only inside `run_divergence_test`,
which a `self_guidance: true` arm skips entirely, so reconstruction / text-only /
dino-swap / text-manip all drew from the **global RNG state** — a function of the
entire training history. Two consequences:

- a standalone re-run of a finished checkpoint could not reproduce the in-train
  numbers (measured: recon LPIPS 0.806 standalone vs 0.524 in-train, pixel
  correlation only +0.22…+0.55), so a "recovered" final validation was impossible;
- the arm's own trend across checkpoints mixed model change with RNG drift, so the
  apparent post-8000 regression was not cleanly attributable to the model.

## Differs from

`latent-first-pretrain` — **same architecture, data and latent mode**; only the
budget and instrumentation cadence are reduced so the whole lifecycle fits in one
short window:

| knob | latent-first-pretrain | this arm |
|---|---|---|
| `training.total_steps` | 10000 | 60 |
| `training.warmup_steps` | 500 | 10 |
| `training.grad_accumulation_steps` | 64 (eff. 256) | 16 (eff. 64) |
| `training.ema_warmup_steps` | 1000 | 10 |
| `training.repa.decay_start/end` | 4000 / 8000 | 20 / 40 |
| `validation.interval_steps` | 2500 | 30 |
| `validation.visual_debug_interval` | 500 | 20 |
| `checkpoint.save_every` | 500 | 30 |

## Expected outcome

Run completes 60/60 with no NaN; all hooks fire in-window; two
`results.json` files exist containing all four tests; and a standalone re-run of
`checkpoint_step0000060.pt` reproduces the in-train step-60 numbers to within
numerical tolerance. Anything less means the seeding change is wrong.

## Prior runs in this arm

`runs/2026-09-12_0939`, `2026-09-12_1054`, `2026-09-12_1125`, `2026-09-17_1134` —
all pre-date the seeding change and therefore carry **unseeded** validation numbers.
They are retained as history; only runs from 2026-09-20 onward are seed-controlled.
