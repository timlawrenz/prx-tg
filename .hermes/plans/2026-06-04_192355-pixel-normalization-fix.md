# Plan: Resolve Issue #5a — Pixel Normalization (`[0,1]` vs `[-1,1]`) in Pixel-Space Flow Matching

**Date:** 2026-06-04
**Author:** Hermes (planning session, prx-tg profile)
**Status:** Plan only — no code changed in this session.
**Workspace:** `/home/tim/source/activity/prx-tg`

---

## 1. Goal

Determine whether the pixel-space flow-matching pipeline trains on mis-scaled
pixel data (data in `[0,1]` interpolated against `N(0,1)` noise), quantify the
impact, fix it consistently across train/sample/validate, and decide how this
affects the claims in the paper (`paper/prx-tg-ablation.tex`) and the May-13 /
May-23 blog posts.

This issue was raised in the May-13 blog post ("Pixel scaling … the dataloader
yields `[0,1]` RGB pixels, which slightly biases the flow-matching objective").
This plan **confirms it is real in code** and lays out the resolution.

---

## 2. Confirmed findings (read-only investigation, this session)

The bug is **present and confirmed**, not hypothetical:

- `production/data.py:119,129` — dataloader returns `image_data` as
  `(3,H,W) float`, **range `[0,1]`**, with no centering. Comment explicitly says
  `range [0, 1]`.
- `production/train.py:478` — `x0 = batch['image_data']` → the `[0,1]` tensor is
  used directly as the flow-matching data endpoint.
- `production/train.py:243` — `z1 = torch.randn_like(x0)` → noise endpoint is
  `N(0,1)` (≈ `[-3,3]`, mean 0).
- `production/train.py:249` — `zt = (1 - t) * x0 + t * z1` interpolates between a
  **`[0,1]`, mean≈0.5 data distribution** and a **zero-mean Gaussian**.
- `production/train.py:263` — `v_target = z1 - x0`.

**Why this is a defect:** Rectified-flow / flow-matching assumes the two
endpoints are distributionally compatible (both ~zero-mean, comparable scale).
Here the data endpoint is offset (mean ≈ +0.5) and half the dynamic range of the
noise. The learned velocity field must absorb a constant DC offset, the
`v_target` is biased, and the signal-to-noise schedule across `t` is skewed.
Standard SD3/rectified-flow pixel pipelines center data to `[-1,1]` (mean 0)
precisely to avoid this.

**Inference side is inconsistent too** (must be fixed together):
- `production/sample.py:371` and `production/visual_debug.py:109` take model
  output, `clamp(0,1)`, then `* 2 - 1` for downstream consumers — i.e. they
  assume the model predicts in `[0,1]`.
- `production/validate.py:256-257,558-559` convert GT `[0,1] → [-1,1]` only for
  the LPIPS comparison, not for the model I/O path.
- Several docstrings in `train.py` / `sample.py` still say "latents" and
  reference a Flux VAE decoder (`train.py:67`, `compute()` lines 85-120) — stale
  from the pre-pixel-space era; a source of confusion that should be cleaned up.

**Net:** training, sampling, and validation each make slightly different
assumptions about pixel range. The fix must make all three consistent.

---

## 3. Proposed approach

Adopt the standard convention: **operate the diffusion process in `[-1,1]`**,
centering pixel data at ingest and de-centering only at the final image-write
boundary.

Two viable options; **Option A recommended.**

### Option A (recommended): center at the dataloader boundary
- In `production/data.py`, convert `image_data` from `[0,1]` to `[-1,1]`
  (`x = x * 2 - 1`) at the point it is returned, for both the training
  dataset and `ValidationDataset`.
- Then `x0` is mean≈0, matching `z1 ~ N(0,1)`.
- Audit every consumer to remove the now-redundant `* 2 - 1` conversions
  (`sample.py:371`, `visual_debug.py:109`) and confirm GT-handling in
  `validate.py` no longer double-converts.
- Decode/write path: only the final PIL conversion does `[-1,1] → [0,1]`
  (`sample.py:200`, `visual_debug.py:177` already do this).

### Option B: center inside the training step only
- Leave the dataloader at `[0,1]`; insert `x0 = x0 * 2 - 1` at `train.py:478`
  and mirror it everywhere the model is invoked (validation, visual debug,
  sampler input/output).
- Rejected as default: more scattered edit sites, easier to miss one, and the
  dataloader contract stays misleading.

### Decision required (open question O1)
Whether `[-1,1]` (recommended, matches SD3/Flux convention) or zero-mean
unit-variance per-channel standardization. `[-1,1]` is simpler and standard;
per-channel stats are marginally better-conditioned but add a stats-computation
step and a stored normalization constant. **Recommend `[-1,1]`.**

---

## 4. Step-by-step plan

1. **Reproduce / quantify the bias (spike, no production change).**
   - Write a throwaway script under `paper/` or `/tmp` that loads one batch,
     prints `x0.min/max/mean/std`, builds `zt` at several `t`, and prints
     `v_target` stats. Confirm mean offset and asymmetric SNR numerically.
   - Record the numbers — they become the justification for the fix and the
     paper caveat.

2. **Implement the normalization fix (Option A).**
   - Edit `production/data.py`: center `image_data` to `[-1,1]` in `__getitem__`
     / sample-processing for both training and validation datasets. Update the
     `# range [0, 1]` comments to `[-1, 1]`.
   - Grep for every site that assumed `[0,1]` model I/O and reconcile:
     `production/train.py` (x0 use, any LPIPS `compute`), `production/sample.py`
     (lines 369-371, 200), `production/visual_debug.py` (lines 108-113, 177),
     `production/validate.py` (lines 256-257, 558-559).
   - Remove stale VAE/latent docstrings in `train.py` `compute()` and
     `sample.py` so the pixel-space contract is unambiguous.

3. **Sanity training run (governance: AGENTS.md requires a short run for
   `train.py`/`model.py` changes).**
   - Commit/stash all other changes first (`git_dirty: false` is mandatory).
   - Run a 50–100 step sanity run on the 7k subset; confirm no NaN/divergence in
     first 100 steps (`velocity_norm_warning: 10.0`, `grad_norm_warning: 10.0`).
   - Eyeball `visual_debug` output and a couple of validation reconstructions to
     confirm images are not inverted/clipped.

4. **Short A/B comparison (old `[0,1]` vs new `[-1,1]`).**
   - Run a short controlled comparison (e.g. 1000–1500 steps) from the same
     seed/config, changing only the normalization, to measure the effect on
     reconstruction/text LPIPS and loss-curve shape.
   - Use the existing validation harness
     (`scripts/run_checkpoint_validation.py`) for comparable LPIPS.

5. **Decide paper framing based on the A/B result.**
   - If impact is small → add a one-paragraph **Limitations** caveat to
     `paper/prx-tg-ablation.tex` stating all A–I arms trained with `[0,1]`
     data centering, that this introduces a known DC-offset bias, and that a
     corrected-normalization run is future work.
   - If impact is large → the A–I numbers may need re-validation before
     publication; escalate before submitting. (This is the reason #5a was
     prioritized first.)

6. **Update collateral.**
   - Update `README.md` known-limitations / status if needed.
   - Fix the May-23 blog's Arm D table (separate already-known issue: it used
     step-500 Arm D numbers vs step-5000 Arm G — see §6 open items).
   - Record the resolved finding in OpenViking under
     `viking://resources/projects/prx-tg/`.

---

## 5. Files likely to change

| File | Change |
|------|--------|
| `production/data.py` | Center pixels `[0,1]→[-1,1]` in train + `ValidationDataset`; fix comments |
| `production/train.py` | Reconcile `x0` range assumption; clean stale VAE/latent docstrings in `compute()` |
| `production/sample.py` | Remove redundant `*2-1` (line 371); verify decode path (line 200) |
| `production/visual_debug.py` | Remove redundant `*2-1` (line 109); verify denorm (line 177) |
| `production/validate.py` | Verify GT `[0,1]→[-1,1]` conversions (256-257, 558-559) don't double-convert |
| `paper/prx-tg-ablation.tex` | Add Limitations caveat (or trigger re-validation) per §4.5 |
| `README.md` | Update known limitations / status if warranted |
| (spike script) | Throwaway quantification script — not committed to production |

---

## 6. Tests / validation (run locally BEFORE any commit)

- **Unit-level:** assert dataloader output range is `[-1,1]` (min ≥ -1, max ≤ 1,
  mean ≈ 0) for a sampled batch; assert a round-trip
  `pixel→[-1,1]→PIL→[0,1]` is visually correct.
- **Smoke:** existing `scripts/validate_training_flow.py` and
  `scripts/validate_overfit.py` (overfit a single sample; LPIPS should drop
  below the memorization threshold) pass with the new normalization.
- **Sanity run:** 50–100 steps, no NaN, velocity/grad norms within warnings.
- **A/B:** short run comparison with LPIPS via `run_checkpoint_validation.py`.
- **GPU coordination:** the RTX 4090 ("game") is shared and may be running the
  big training run. **Ask before launching any GPU job; coordinate a pause.**
  Sanity/A-B runs should slot into a free window.
- **Governance (AGENTS.md):** commit/stash so `git_dirty: false`; do not modify
  `production/` while another training run is active; use a named config.

---

## 7. Risks, tradeoffs, open questions

- **R1 — Invalidation risk:** if the `[0,1]` bias materially helped or hurt
  specific arms unevenly, the A–I comparison could be partly confounded. The
  A/B test in §4.4 is what tells us whether the paper needs only a caveat or a
  re-run. This is the highest-stakes unknown and the reason to do #5a first.
- **R2 — Inference mismatch:** if any consumer's `*2-1` is missed, outputs will
  look washed out / inverted. Mitigation: exhaustive grep checklist in §4.2 plus
  the visual sanity check.
- **R3 — Checkpoint compatibility:** existing checkpoints were trained on
  `[0,1]`; loading them into a `[-1,1]` inference path will look wrong. Document
  that old checkpoints require the old path; new runs use the new path.
- **O1 (open):** `[-1,1]` vs per-channel standardization — recommend `[-1,1]`.
- **O2 (open):** Does AsymFlow's PCA projection (`apply_asymflow_projection`,
  `train.py:256`) assume a particular noise/data scale? Re-check that the
  subspace projection is unaffected by re-centering data (it projects `z1`, not
  `x0`, so likely fine — but verify).
- **O3 (open):** Confirm with author whether captions are also processed by
  CLIP (blog says "CLIP and T5"; paper/config say T5-Large only). Not part of
  #5a but surfaced during review; resolve separately.

---

## 8. Recommendation

Do §4.1 (quantify) and §4.2 (fix, Option A) first, then the §4.4 A/B run during
a coordinated GPU window. The A/B result is the decision point: small impact →
caveat in the paper and proceed; large impact → hold publication and re-validate
affected arms. Treat the normalization fix as a prerequisite for the production
"big run" regardless of the paper outcome.
