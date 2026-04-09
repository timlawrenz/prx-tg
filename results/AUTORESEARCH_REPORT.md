# Autoresearch Tournament Report

**Branch**: `autoresearch/20260330`  
**Duration**: March 30 – April 9, 2026  
**Hardware**: RTX 2070 SUPER (8GB VRAM), single GPU, float16  
**Base config**: `production/config_turbo_2070.yaml`  
**Total experiments**: 18 completed, 2 interrupted, 1 parse-bug crash  

---

## Executive Summary

Starting from the default config (LR=3e-4, warmup=500, wd=0.03, mask_ratio=0.75), the tournament found that **learning rate was the dominant knob**, yielding a 48% reduction in training loss. The key insight: with warmup_steps=500 equal to the step budget, LR never reached its target — raising LR and shortening warmup gave the model a proper warmup→peak→decay schedule.

**Best configuration found**:
- `training.optimizer.lr=4e-3` (was 3e-4, **13× higher**)
- `training.warmup_steps=200` (was 500)
- `training.optimizer.weight_decay=0.01` (was 0.03)
- `training.maskdit.mask_ratio=0.6` (was 0.75, found in Phase C)

**Best checkpoint**: `experiments/2026-04-06_2150/checkpoints/checkpoint_final.pt`  
(2000 total steps, reconstruction_lpips=1.2959)

---

## Phase A: Convergence Speed (val_loss, 500 steps from scratch)

**Goal**: Find config that converges fastest in limited steps.  
**Metric**: val_loss (training loss at final step)  
**Experiments**: 10 completed

| # | Config | val_loss | vs baseline | status |
|---|--------|----------|-------------|--------|
| 1 | baseline (LR 3e-4) | 0.4528 | — | keep |
| 2 | LR 1e-4 | 0.7326 | +62% worse | discard |
| 3 | LR 5e-4 | 0.3514 | -22% | keep |
| 4 | LR 1e-3 | 0.3072 | -32% | keep |
| 5 | LR 2e-3 | 0.2665 | -41% | keep |
| 6 | LR 4e-3 | 0.2610 | -42% | keep |
| 7 | LR 4e-3 + warmup 200 | 0.2388 | -47% | keep |
| 8 | LR 4e-3 + warmup 100 | 0.2433 | -46% | discard |
| 9 | **LR 4e-3 + warmup 200 + wd 0.01** | **0.2369** | **-48%** | **⭐ winner** |

### Phase A Insights

1. **LR sweep was monotonically improving** (1e-4 → 4e-3), driven by the warmup interaction: with warmup=step_budget, higher target LR = steeper ramp = more effective learning.
2. **Warmup=200 was the sweet spot**: lets LR peak at step 200 then decay for 300 steps (proper schedule), vs warmup=500 where LR never peaks.
3. **Warmup=100 slightly worse**: too aggressive a ramp, not enough warmup.
4. **Weight decay 0.01**: marginal 1% improvement over 0.03.
5. Diminishing returns hit at LR=4e-3 (only 2% gain over 2e-3).

---

## Phase B: Image Quality (reconstruction_lpips, 500 additional steps)

**Goal**: Tune for actual image quality from Phase A's best checkpoint.  
**Metric**: reconstruction_lpips (lower = better)  
**Resume from**: Phase A winner at step 500  
**Experiments**: 4 completed (all discards)

| # | Config | recon_lpips | text_only_lpips | status |
|---|--------|-------------|-----------------|--------|
| 1 | baseline | 1.3075 | 0.9692 | keep |
| 2 | mask_ratio 0.6 | 1.3082 | 0.9694 | discard |
| 3 | ema_decay 0.999 | 1.3081 | 0.9693 | discard |
| 4 | tread_routing 0.5 | 1.3081 | 0.9694 | discard |
| 5 | LR 8e-3 | 1.3079 | 0.9668 | discard |

### Phase B Insights

1. **500 additional steps was insufficient** to differentiate configs — all experiments landed within ±0.0007 of baseline.
2. Transitioned to Phase C after 4 consecutive discards with clear insight: need more training steps.

---

## Phase C: Extended Training (reconstruction_lpips, 1000 additional steps)

**Goal**: Same as B but with 2× the step budget.  
**Metric**: reconstruction_lpips (lower = better)  
**Resume from**: Phase B best (= Phase A winner) at step 1000  
**Experiments**: 2 completed, 1 interrupted

| # | Config | recon_lpips | text_only_lpips | status |
|---|--------|-------------|-----------------|--------|
| 1 | baseline | 1.2985 | 0.9633 | keep |
| 2 | **mask_ratio 0.6** | **1.2959** | 0.9638 | **⭐ winner** |
| 3 | LR 8e-3 | 1.2977 | **0.9577** | keep |
| 4 | tread_routing 0.5 | — | — | interrupted |

### Phase C Insights

1. **1000 additional steps showed real differentiation** — Phase C baseline improved 0.7% over Phase B (vs ~0% variance in Phase B).
2. **mask_ratio=0.6 is the reconstruction winner** — less masking = more visible context = better reconstructions.
3. **LR=8e-3 is the text_only winner** — higher LR improves text-conditioned generation quality.
4. Different knobs optimize different quality dimensions.

---

## Recommended Config Changes

Apply to `production/config_turbo_2070.yaml`:

```yaml
training:
  warmup_steps: 200          # was 500
  optimizer:
    lr: 0.004                 # was 0.0003
    weight_decay: 0.01        # was 0.03
  maskdit:
    mask_ratio: 0.6           # was 0.75
```

These changes are baked into `production/config_phase_b.yaml` (except mask_ratio).

---

## Bugs & Workarounds Discovered

1. **Checkpoint path bug** (`autoresearch.py:291-304`): `evaluate_val_loss_standalone()` looks for checkpoints relative to CWD instead of experiment dir. Every run reports "crash" — required manual TSV fixes.

2. **Resume overwrites experiment dir** (`train_production.py:130-136`): `--resume-from` infers experiment directory from checkpoint path and writes into it. Workaround: copy checkpoint to neutral location (`checkpoints/phase_X_best/`).

3. **Step budget is absolute** (`autoresearch.py:128`): `--step-budget N` sets `total_steps=N`, not additional steps. For resumed runs, must add the existing step count (e.g., step_budget=1000 for 500 additional steps from step 500).

4. **Multi-change separator**: `--changes` splits on whitespace, not commas. Use `"key1=val1 key2=val2"`.

---

## File Index

| File | Description |
|------|-------------|
| `results/results_phase_A.tsv` | Phase A results (10 experiments) |
| `results/results_phase_B.tsv` | Phase B results (5 experiments) |
| `results/results_phase_C.tsv` | Phase C results (3 experiments) |
| `results/AUTORESEARCH_REPORT.md` | This report |
| `production/config_phase_b.yaml` | Config with Phase A winners baked in |
| `checkpoints/phase_a_best/` | Phase A best checkpoint (gitignored) |
| `checkpoints/phase_b_best/` | Phase B best checkpoint (gitignored) |
| `experiments/2026-04-06_2150/` | Overall best experiment (Phase C mask_ratio=0.6) |
