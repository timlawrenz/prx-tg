# VAE Ceiling Experiment — Results (2026-09-10)

## Question
If we do latent-first (Scheme B), the VAE's reconstruction quality sets the ceiling on
what the *latent* model can ever produce. Does the FLUX AE (the one that made the
precomputed `AbstractPhil/ffhq_flux_latents_repaired`) clear our photoreal gate — or is
the autoencoder itself the binding constraint before we commit any training?

## Method (GPU-safe)
- Real FFHQ raw 1024² PNGs (deterministic subset: indices 0–15)
- FLUX AE (`/mnt/models/vae/ae.safetensors`, loaded via ComfyUI — auto-VRAM-managed,
  batched to 1, `expandable_segments:True`, verified 23.6GB free throughout)
- 3 scale conventions tested at full 1024² (the AE's latent scale ambiguity)
- SDXL VAE (`/mnt/models/vae/SDXL/sdxl_vae.safetensors`) as independent 2nd AE
- **LPIPS (AlexNet) at 256px + visual inspection at full res**

## Results (n=16, LPIPS mean / min / max; lower = better)

All rows now quoted at the **same** scale convention (s = ×2.76932, the scale that
minimises LPIPS — see correction note below):

| Model | mean | min | max | note |
|---|---|---|---|---|
| FLUX AE @1024², s=2.769 | **0.242** | 0.177 | 0.297 | best scale |
| FLUX AE @1024², s=1 | 0.303 | 0.212 | 0.356 | |
| FLUX AE @512² (precomputed latents), s=2.769 | **0.520** | 0.463 | 0.586 | the `_repaired` HF set is **512²**, not 1024² |
| SDXL VAE @1024² | 0.346 | 0.278 | 0.416 | |

Other scale conventions on the 512² precomputed set, for completeness: s=0.3611 →
mean 0.817 (min 0.695, max 0.877); s=1 → mean 0.706 (min 0.599, max 0.763).

### ⚠ Correction (2026-09-19)
This table originally read **0.706 / 0.599 / 0.763** for the 512² row. That figure is
`flux_s1_lpips` — the 512² decode at scale **×1.0** — while every other row quoted the
**best** scale (×2.76932). The row therefore compared the *worst* scale of the 512² set
against the *best* scale of the 1024² set, overstating the 512² penalty by ~0.19 LPIPS.

Recomputed from the stored `ceiling_metrics.jsonl` (16 rows, **unchanged**), the
like-for-like 512² figure at the matching scale is **0.520 / 0.463 / 0.586**.
The conclusion direction is unaffected — 0.520 is still more than double 0.242 — but
the previously published number was wrong.

Root cause: `scripts/vae_ceiling_test.py`'s own summary loop iterates over
`flux_s2.7689` / `flux_full1024_s2.7689`, while the actual stored keys are
`..._s2.76932`. The two best-scale rows therefore printed nothing, and the summary
numbers had to be picked by hand from the jsonl. **The metrics jsonl itself is
unaffected — only the summary table was wrong.** The script fix is queued below.

## Visual (full-res, cross-checked)
- **FLUX AE @1024² best-scale:** recognizable face, but **washed out, low contrast,
  mushy detail** — eyelashes/hair/skin texture smoothed away. Verdict: *plastic/painterly,
  not photoreal*, despite good LPIPS (LPIPS under-weights the brightness/exposure shift).
- **SDXL VAE:** brighter, color cast (pink), some detail loss; not photoreal either.

### ⚠ Correction (2026-09-19) — the luminance figures
This section originally read "**washed out (mean-lum 0.85 vs real 0.47)**". The luminance
of the *real* photos is **not stored in `ceiling_metrics.jsonl`** — it has no `real_lum`
key — so that comparison was not reproducible from the record. Both sides have now been
measured directly from the saved PNGs in `samples/` (indices 0, 3, 7, 12):

| | mean luminance | mean contrast |
|---|---|---|
| real photos (4 saved samples) | **0.437** | 0.239 |
| FLUX AE @1024² s=2.769 decode (same 4) | **0.855** | 0.154 |

The jsonl's `flux_full1024_s2.76932_lum` over all 16 rows gives **0.871**, consistent
with the 0.855 measured on the 4 saved samples. The original claim was therefore
**directionally correct and numerically close**, but the "real 0.47" side was never a
recorded measurement. Note also that `flux_s2.76932_lum` — the *512² decode* — averages
**0.457**; it is possible the original "0.47" was that value, misread as the real photos'.
Storing a real-photo `lum`/`contrast` per row is queued below.

## Conclusions
1. **The precomputed `AbstractPhil/ffhq_flux_latents_repaired` is NOT a 1024² dataset.**
   It stores 512² latents (16×64×64). Re-verified externally on 2026-09-19 by querying the
   HF datasets-server directly: the latent array is `(16, 64, 64)` → 64×8 = **512²**.
   Using it for a 1024² latent-first model would cap training at 512² and reconstructs
   poorly (LPIPS 0.520 like-for-like; 0.706 at the ×1.0 scale) — **do not use this set.**
2. **Both FLUX AE and SDXL VAE, at their best, are mushy/washed at full res.** Neither
   clears the photoreal band on FFHQ faces even for *encode→decode of a real photo*.
3. **The VAE ceiling is real and material.** If we trained a perfect latent model, the
   decoded output would still be a softened, brightness-lifted face. This does NOT kill
   Scheme B (Z-Image/L2P/AsymFlow all post-train in pixel space precisely to recover this
   detail — the latent phase is for *semantics*, the pixel phase for *texture*), but it
   means: **(a)** the pixel post-training phase is not optional — it is the phase that
   actually renders photoreal texture; **(b)** our G0-style gate must compare against
   *decoded-real-latents*, not against raw photos, or we will falsely blame the VAE for
   what is a latent-model failure; **(c)** the brightness lift (0.437→0.855 measured) is a
   decode convention issue to control for (scale/offset), not a fatal artifact.

## What this changes
- `latent-first-pretrain` is **still on the table**, but the plan must:
  1. Encode latents at **1024² ourselves** (FLUX AE via /mnt/models/vae/ae.safetensors,
     or a better AE) — never use the 512² `_repaired` set.
  2. Budget a **pixel post-training phase** as first-class (that's where photorealism
     comes from; the latent phase alone cannot deliver it through the mushy decode).
  3. Include a **decoded-latents baseline** in every gate (real photo → encode → decode
     vs generated photo → encode → decode), so "is the VAE mushing it" is measured
     separately from "did the model produce a bad latent".

## Queued follow-up — next GPU window
Tracked in `experiment-configs/vae-ceiling/provenance.yaml` under `queued_rerun`:

1. Fix `scripts/vae_ceiling_test.py` summary loop: `flux_s2.7689` → `flux_s2.76932`,
   `flux_full1024_s2.7689` → `flux_full1024_s2.76932`.
2. Store real-photo luminance/contrast per row (`real_lum`, `real_contrast`) so the
   washed-out comparison is reproducible from the record.
3. Re-run `python scripts/vae_ceiling_test.py --n 16` (~6 min, 16 images, batch 1) and
   confirm the corrected table reproduces from the regenerated jsonl.
4. Then write the ledger entry, create the arm tag, and decide this arm's branch home —
   this arm currently has **no branch, no tag, no ledger entry**.

## Artifacts
- Metrics: `research/results/vae_ceiling/ceiling_metrics.jsonl` (16 rows, full detail)
- Sample images (real vs flux-best vs sdxl for idx 0,3,7,12): `research/results/vae_ceiling/samples/`
- Script (reproducible): `scripts/vae_ceiling_test.py`
- Arm record: `experiment-configs/vae-ceiling/{README.md,provenance.yaml}`
- GPU hygiene: 23.6GB free before/after, 0% util at exit, no training disturbed.
- **Versioning caveat:** `research/results/` is currently gitignored, so none of the
  above is in git. Per AGENTS.md §0.1 this is soft data and must reach the repo.