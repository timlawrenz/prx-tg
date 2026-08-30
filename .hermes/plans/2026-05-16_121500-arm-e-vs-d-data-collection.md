# Plan: Arm E vs Arm D — Training Data Collection & Paper Preparation

**Created:** 2026-05-16  
**Status:** Planning only — no execution yet

---

## Goal

Collect, consolidate, and locally archive all training data needed to write a paper comparing **Arm E** (seg spatial loss weighting) vs **Arm D** (full-stack baseline) from the current prx-tg training run.

---

## What We Are Comparing

| | Arm D | Arm E |
|---|---|---|
| **Key difference** | No seg weighting | Seg weight map (face 2×, bg 0.5×, `normalize=True`) |
| **Training steps** | 5000 | 5000 (in progress, ~step 1724 at time of writing) |
| **Checkpoint source** | NAS: `vast_final_2026-05-12_0902/experiments/` | Vast.ai: `/workspace/experiments/2026-05-15_2003/` |
| **Training log** | NAS: `…/experiments/training_log.jsonl` | Vast.ai: same path (live) |
| **Validation data** | NAS: `…/experiments/validation/step*/results.json` | Vast.ai: same path (live) |
| **Config** | NAS: `…/experiments/config.yaml` | Vast.ai: `scripts/arm_e_config.yaml` |

---

## Data Inventory

### Arm D — already on NAS (complete)

All data lives at `/mnt/nas-ai-models/training-data/prx-tg/vast_final_2026-05-12_0902/experiments/`:

| Asset | Path | Status |
|---|---|---|
| Config | `config.yaml` | ✅ present |
| Metadata | `metadata.json` | ✅ present |
| Training log (JSONL) | `training_log.jsonl` | ✅ present |
| TensorBoard | `tensorboard/` | ✅ present |
| Checkpoints | `checkpoints/checkpoint_step*.pt` | ✅ present |
| Validation results | `validation/step{500,1000,…,5000}/results.json` | ✅ all 10 steps |
| Validation images | `validation/step*/reconstruction/`, `text_only/`, `dino_swap/`, `text_manip/` | ✅ present |

**Arm D is complete. No collection action needed.**

---

### Arm E — in progress on Vast.ai

Training runs on Vast.ai at `/workspace/experiments/2026-05-15_2003/`.  
Live cron sync is running but only syncs every few hours; the sync dirs are stale.

| Asset | Source path (Vast.ai) | NAS landing zone | Status |
|---|---|---|---|
| Config | `/workspace/prx-tg/scripts/arm_e_config.yaml` | collected by existing sync | ✅ present in sync dirs |
| Metadata | `.../metadata.json` | present in sync dirs | ✅ |
| Training log (JSONL) | `.../training_log.jsonl` | partial (112 lines as of this morning) | ⚠️ stale — needs final pull |
| TensorBoard | `.../tensorboard/` | partial | ⚠️ needs final pull |
| Checkpoints | `.../checkpoints/checkpoint_step*.pt` | only 1600 & 1700 synced so far | ⚠️ needs full sync at completion |
| Validation results | `.../validation/step*/results.json` | only 1700 synced | ⚠️ needs ongoing + final sync |
| Validation images | `.../validation/step*/reconstruction/`, etc. | only 1700 synced | ⚠️ needs ongoing + final sync |
| armE.log (nohup log) | `/workspace/armE.log` | not synced | ⚠️ collect at end |

**Expected validation checkpoints:** steps 1700, 1800, …, 5000 (every 100 steps per config).  
At ~54s/step, training completes around **Sunday ~18:00 local**.

---

## Locally-Generated Data (Post-Hoc)

The Vast.ai-side `validate.py` generates validation at each checkpoint during training.  
However, we also need **standardized, comparable** validation runs that use the **exact same seed, prompts, and sample indices** for both arms. The Vast.ai on-the-fly validation may not be bit-for-bit identical to local runs.

### What needs to be generated locally after collection:

1. **Full validation suite at each checkpoint** (steps 500–5000, every 500 steps)  
   — Run locally via `scripts/run_validate.py` (or equivalent) for each Arm E checkpoint  
   — Use same fixed validation seed and sample set as used for Arm D  
   — Produces: `reconstruction LPIPS`, `text-only LPIPS`, `DINO swap`, `text manipulation LPIPS diff`

2. **Qualitative image grid** (side-by-side Arm D vs Arm E at matched steps)  
   — For the paper: reconstruction images from same 5–10 sample indices at steps 1000, 2000, 3000, 4000, 5000  
   — Script to generate: not yet written

3. **Loss curve plots**  
   — Parse both `training_log.jsonl` files  
   — Plot: `loss`, `repa_loss`, `repa_weight`, `grad_norm`, `lr` vs step  
   — Overlay Arm D and Arm E on the same axes

4. **LPIPS curve across training**  
   — Aggregate `results.json` files for each arm across all checkpoint steps  
   — Plot: `reconstruction LPIPS` and `text-only LPIPS` vs step for both arms

5. **Text manipulation LPIPS diff curve**  
   — `mean_lpips_difference` vs step for both arms  
   — This is the primary metric for "text responsiveness"

---

## Collection Plan — Step by Step

### Phase 1: Final sync from Vast.ai (after training completes ~Sunday 18:00)

1. `rsync -avz -e "ssh -p 13840" root@ssh3.vast.ai:/workspace/experiments/2026-05-15_2003/ /mnt/nas-ai-models/training-data/prx-tg/arm_e_final/`
2. `rsync -avz -e "ssh -p 13840" root@ssh3.vast.ai:/workspace/armE.log /mnt/nas-ai-models/training-data/prx-tg/arm_e_final/`
3. Verify checkpoint count: should be ~35 files (every 100 steps from 1600 → 5000, plus earlier ones)
4. Verify validation dirs: should be `step1700/` through `step5000/` at minimum

**Canonical NAS destination:** `/mnt/nas-ai-models/training-data/prx-tg/arm_e_final/`

### Phase 2: Standardized local re-validation (Arm E checkpoints)

For each Arm E checkpoint at steps 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000:
1. Run `validate.py` locally (Strix Halo, ROCm) using the same script used for Arm D
2. Use identical fixed seed + sample indices as Arm D validation
3. Save output to `scripts/validation_outputs/arm_e/step{NNNN}/results.json`

**Note:** Step 1500 local validation for Arm E already done — results in `scripts/validation_outputs/step0001500/`.

### Phase 3: Generate paper figures

1. **Loss curves script** — `scripts/plot_loss_curves.py`  
   - Inputs: `arm_d/training_log.jsonl`, `arm_e_final/training_log.jsonl`  
   - Outputs: `figures/loss_curve_comparison.png`

2. **LPIPS trajectory script** — `scripts/plot_lpips_trajectory.py`  
   - Inputs: `validation/step*/results.json` for both arms  
   - Outputs: `figures/lpips_reconstruction.png`, `figures/lpips_text_only.png`, `figures/text_manip_lpips.png`

3. **Image grid script** — `scripts/generate_comparison_grid.py`  
   - Inputs: validation image dirs for matched steps and sample indices  
   - Outputs: `figures/reconstruction_grid_step{N}.png`

---

## Canonical Directory Layout (post-collection)

```
/mnt/nas-ai-models/training-data/prx-tg/
  arm_d/                          # symlink or copy of vast_final_2026-05-12_0902/experiments/
    config.yaml
    training_log.jsonl
    validation/step{500..5000}/results.json
    checkpoints/
  arm_e_final/                    # final rsync from Vast.ai
    config.yaml
    training_log.jsonl
    armE.log
    validation/step{1700..5000}/results.json
    checkpoints/

/home/tim/source/activity/prx-tg/
  scripts/
    validation_outputs/
      arm_d/step{500..5000}/results.json    # already on NAS, may symlink
      arm_e/step{1500..5000}/results.json   # local re-validation
    figures/                                # generated paper figures
      loss_curve_comparison.png
      lpips_reconstruction.png
      lpips_text_only.png
      text_manip_lpips.png
      reconstruction_grid_step*.png
```

---

## Key Metrics for the Paper

### Primary claim
> Seg spatial loss weighting improves text controllability (text manipulation LPIPS diff ↑) without degrading reconstruction fidelity (reconstruction LPIPS ≈ Arm D).

### Metrics table (to be filled)

| Metric | Arm D (step 5000) | Arm E (step 5000) | Δ |
|---|---|---|---|
| Reconstruction LPIPS | 0.9352 | TBD | TBD |
| Text-only LPIPS | 0.9593 | TBD | TBD |
| Text manipulation LPIPS diff | 0.4663 | TBD | TBD |
| Mean grad norm | TBD | ~0.30–0.38 | TBD |

*Arm D step 5000 values sourced from `vast_final_2026-05-12_0902/experiments/validation/step0005000/results.json`.*

---

## Risks & Open Questions

- **Validation comparability:** Arm D validation was run on-GPU during Vast.ai training; Arm E step 1500 was run locally (ROCm). Small numerical differences expected but not meaningful. The re-validation in Phase 2 uses the same local environment for both, which is the fair comparison baseline.
- **Vast.ai instance stability:** Mitigated by `vast-experiment-sync` cron job (every 3h), which rsyncs the full experiments dir to NAS. If the instance is reclaimed, we lose at most ~3h of data.
- **Checkpoint gaps:** The 3h cron sync should keep NAS reasonably current. All checkpoints will be on NAS within one sync cycle of being written.
- **arm_e_config.yaml uses `$STRATUM_DIR`:** Local re-validation needs `STRATUM_DIR=/path/to/stratum` set to point to local NAS stratum data at `/mnt/nas-ai-models/training-data/ffhq/stratum/`.

---

## Open Questions to Resolve Before Execution

1. Do we want to re-validate Arm D locally too (for parity), or trust the Vast.ai-generated Arm D validation numbers?
2. What checkpoint spacing for the paper? Every 500 steps (10 points) or every 1000 steps (5 points)?
3. Do we want a FID score? That requires a separate batch generation pass, not covered by current `validate.py`.
4. Is the Vast.ai cron sync job still active and healthy? Should verify before Sunday.
