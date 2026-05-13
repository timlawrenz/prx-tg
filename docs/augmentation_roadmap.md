# Data Augmentation Roadmap — Full Stack (Arm D)

**Status:** Pre-implementation planning  
**Baseline:** Full Stack checkpoint from ablation study (Muon + TREAD + REPA, 5000 steps)  
**Tagged:** `v1.0-ablation-study`

---

## Context

The ablation study established Full Stack as the canonical architecture. The study review
identified data augmentation as a primary lever for the next improvement cycle — but with
explicit traps that must be respected before any augmentation is switched on.

---

## Current Augmentation State

From `production/config.yaml`:

```yaml
data:
  # horizontal_flip_prob: 0.5  # DISABLED — see below
  # NOTE: Caption swapping (left↔right) is NOT supported because T5 embeddings
  # are pre-computed. Caption text is unused during training (only T5 embeddings matter).
```

Horizontal flip is disabled because **T5 embeddings are pre-computed** and cannot be
re-computed per augmented sample at training time. Any augmentation that changes spatial
semantics requires corresponding updates to all conditioning signals.

---

## Augmentation Candidates

### 1. Horizontal Flip (blocked — requires index remapping)

**Benefit:** Doubles effective dataset size at zero data cost.

**Blocker (Index Swap Pitfall):** Flipping the pixel tensor is insufficient. DWPose uses
133 whole-body keypoints with named, asymmetric indices (e.g., `LEFT_EYE` ≠ `RIGHT_EYE`).
Flipping the image without remapping indices trains the model to associate the "right-side"
text token with "left-side" pixel data, corrupting cross-attention.

**Required work before enabling:**
- [ ] Build a DWPose 133-keypoint left↔right index swap map
- [ ] Apply swap to `joint_positions` tensor in the dataloader augmentation path
- [ ] Confirm T5 embeddings don't embed left/right spatial semantics (audit captions in `faces7k`)
- [ ] If captions do embed left/right: either drop flip or regenerate T5 embeddings on-the-fly

**Files to change:** `production/dataset.py` (or equivalent dataloader), `production/config.yaml`

---

### 2. Random Crop with Keypoint Masking (blocked — requires visibility masking)

**Benefit:** Exposes the model to partial face compositions; improves generalization beyond
tightly-framed FFHQ crops.

**Blocker (Truncation Trap):** Random cropping causes some keypoints to fall outside the
frame. The DWPose estimator must not be allowed to interpolate or "guess" coordinates
beyond pixel boundaries. Keypoints outside the crop bounding box must receive `visibility=0`
and must be masked out of the spatial conditioning loss.

**Required work before enabling:**
- [ ] Add crop-aware visibility masking in the dataloader
- [ ] Confirm `pose_confidence_threshold: 0.05` already zeroes out low-confidence joints
- [ ] Validate that the spatial conditioning path correctly respects `[NULL_POSE]` tokens
- [ ] Unit test: crop an image, assert out-of-frame joints show `visibility=0` in batch

---

### 3. Color Jitter / Photometric Augmentation (low risk)

**Benefit:** Reduces sensitivity to FFHQ-specific color grading. FFHQ images share lighting
characteristics (professional studio-style) that may limit generalization.

**Risk:** Low — color jitter does not affect spatial conditioning or T5 embeddings.

**Required work:**
- [ ] Add `color_jitter_prob`, `brightness`, `contrast`, `saturation` knobs to `config.yaml`
- [ ] Implement in dataloader; apply only to pixel tensor (not DINO embeddings, which are pre-computed)
- [ ] Note: DINO embeddings are pre-computed at ingest. Color jitter applied at train time will
      create a mismatch between the pixel and DINO conditioning. Options:
      - (a) Disable color jitter (safe)
      - (b) Re-compute DINO embeddings on-the-fly (expensive: ~200ms per sample)
      - (c) Apply jitter only to pixel/loss path, not to DINO conditioning (mixed signal)
- [ ] **Decision needed before implementation**

---

### 4. Resolution Schedule Tuning (active, no blocker)

The current schedule trains 0–500 steps at 0.25× scale, 500–3000 at 0.5×, 3000–6000 at 1.0×.
At 5000 steps the model barely had time at full resolution.

**For the next run (longer training):**
- [ ] Profile what fraction of compute is spent at each resolution
- [ ] Consider compressing the 0.25× phase (burn-in is fast, scale up sooner)
- [ ] Consider extending full-resolution phase to ≥50% of total steps

---

## REPA Stage-Wise Termination (not augmentation, but adjacent)

The study review flags this as a **critical trap** for extended training:

> REPA operates under a phenomenon categorized as "works until it doesn't". After the
> burn-in phase, the teacher's embeddings become a restrictive "straitjacket" that penalizes
> fine-grained detail generation.

**Recommendation (HASTE methodology):**
- Decay `repa.weight` from 0.5 → 0.0 over steps 1000–2000 (first ~20–30% of a 6000-step run)
- [ ] Add `repa.decay_start_step` and `repa.decay_end_step` to `config.yaml`
- [ ] Implement linear decay schedule in training loop

---

## Priority Order

| Priority | Item | Risk | Blocker |
|----------|------|------|---------|
| 1 | REPA stage-wise decay | Low | None — code change only |
| 2 | Color jitter (pixel-only, no DINO) | Medium | Decision on DINO mismatch |
| 3 | Horizontal flip + index swap | High | DWPose 133-pt remap table |
| 4 | Random crop + visibility mask | High | Dataloader audit |
| 5 | Resolution schedule tuning | Low | Profile first |

---

## Reference

- `docs/ablation/ablation_study_review.md` — Sections 7.3 (augmentation traps), 6.2 (HASTE/REPA)
- `production/config.yaml` — current training configuration (Full Stack)
- `production/train_production.py` — training loop
- `v1.0-ablation-study` — git tag of archived ablation codebase
