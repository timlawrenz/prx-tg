## Context

Before committing to full-scale training (400M+ parameters, 60k images, weeks of GPU time on 4090), we must validate the architecture works correctly at minimal scale. The Nano DiT tests fundamental assumptions: Does dual conditioning (DINOv3 + T5) work? Does flow matching converge? Do adaLN-Zero and cross-attention layers interact correctly? Catching bugs here saves weeks of wasted full-scale training.

Current State:
- WebDataset validation shards created (100 samples, 4 buckets, ~300MB)
- Stage 2 embeddings ready (DINOv3, VAE, T5 hidden states, attention masks)
- the-plan.md Part B specifies validation objectives

Constraints:
- 24GB 4090 VRAM budget
- 512×512 training resolution (64×64 latents) for fast iteration
- Must complete in 2-4 hours (~5k steps)
- Must prove reconstruction, DINO control, and text control before proceeding to Part C

## Goals / Non-Goals

**Goals:**
- Validate Nano DiT architecture (12L/384H/6A) learns to memorize 100 images
- Prove DINOv3 conditioning controls composition/lighting via adaLN-Zero
- Prove T5 conditioning controls spatial details via cross-attention
- Confirm flow matching loss and logit-normal sampling converge correctly
- Establish baseline training dynamics before scaling to 400M parameters

**Non-Goals:**
- Production-quality image generation (this is a sanity check, not deployment)
- Full 60k dataset training (that's Part C)
- Multi-GPU or distributed training (single 4090 sufficient)
- Advanced sampling techniques (50 Euler steps adequate for validation)
- Hyperparameter tuning (use the-plan.md defaults from Part D)

## Decisions

### Decision: 512×512 training resolution (64×64 latents)
**Rationale:** Fast iteration (2-4 hours for 5k steps on 4090) while preserving enough spatial resolution to test cross-attention and composition. 1024×1024 would take 4× longer without adding validation signal.

**Alternatives considered:**
- 256×256: Too low resolution, loses spatial control testing
- 1024×1024: 4× slower, unnecessary for validation
- Mixed resolutions: Complicates batch logic, not needed for 100-image overfit

**Trade-off:** Can't test full-resolution details (hair texture), but validates architecture correctness which is the goal.

### Decision: Nano model size (12L/384H/6A) ≈ 30-50M params
**Rationale:** Small enough to train fast (5k steps in 2-4 hours) but large enough to memorize 100 images and test conditioning interactions. 12 layers sufficient to test multi-layer adaLN and cross-attention propagation.

**Alternatives considered:**
- Tiny (6L/256H): Too small, might succeed via overfitting without proving architecture
- Medium (24L/512H): Too slow, defeats purpose of fast validation
- Same as production (32L/1024H): Would take days, defeats purpose

**Trade-off:** Can't test scaling behaviors, but proves core architecture works.

### Decision: Independent CFG dropout (10% both, 10% text-only, 10% DINO-only, 70% both)
**Rationale:** Forces model to learn both conditioning signals independently. Matches the-plan.md Part D section 4B. Without this, model might ignore one signal.

**Alternatives considered:**
- Standard CFG (only unconditional dropout): Doesn't guarantee both signals are learned
- Higher dropout rates: Would slow convergence on 100-image overfit

**Trade-off:** Slightly slower initial convergence, but proves dual conditioning works correctly.

### Decision: Logit-normal timestep sampling (mean=0, std=1)
**Rationale:** Concentrates training on mid-range timesteps where structure forms. Matches the-plan.md Part D section 2. Flow matching benefits more from this than uniform sampling.

**Alternatives considered:**
- Uniform [0, 1]: Over-samples easy/hard extremes, under-samples critical middle range
- Different logit-normal params: (0, 1) is standard for flow matching

**Trade-off:** None - this is best practice for flow matching.

### Decision: EMA decay warmup from 0 → 0.9999 over 5k steps
**Rationale:** Prevents EMA from copying noisy initial weights. Matches the-plan.md Part E section 4. EMA starts tracking only after model shows learning signal.

**Alternatives considered:**
- Fixed 0.9999 from start: EMA would be 99.99% random init for first 10k steps
- No EMA: Can't evaluate smooth weights vs noisy training weights

**Trade-off:** None - warmup is critical for from-scratch training.

### Decision: AdamW (β1=0.9, β2=0.95, wd=0.03, lr=3e-4 → 1e-6)
**Rationale:** Follows the-plan.md Part D section 1 exactly. β2=0.95 (not 0.999) adapts faster for generative models from scratch. 5k-step warmup stabilizes adaLN layers.

**Alternatives considered:**
- Standard Adam β2=0.999: Too slow to adapt for from-scratch
- SGD: Doesn't work for transformers from scratch
- Higher peak LR: Would cause instability in early steps

**Trade-off:** None - these are proven hyperparameters from Flux/SD3 training.

### Decision: Horizontal flip augmentation with caption swap (left ↔ right)
**Rationale:** Doubles effective dataset size (100 → 200 samples) while preserving spatial caption correctness. Matches the-plan.md Part D section 3.

**Alternatives considered:**
- No augmentation: Model might overfit exact training poses
- Random crop: Conflicts with bucketing, loses composition
- Color jitter: Destroys lighting signal we're trying to learn

**Trade-off:** Slightly more complex dataloader logic, but critical for spatial control testing.

### Decision: Validation every 1000 steps (5 validation runs total)
**Rationale:** Frequent enough to catch early failures, infrequent enough to not slow training significantly. Each validation run takes ~5-10 minutes (3 tests × 5 samples × 50 steps × 3 forward passes).

**Alternatives considered:**
- Every 500 steps: Slows training too much
- Only at end: Might miss early failure modes

**Trade-off:** ~5-10% training time overhead, but necessary to monitor convergence.

### Decision: Three validation tests (reconstruction, DINO swap, text manip)
**Rationale:** Each tests a different failure mode:
- Reconstruction: Model memorizes images (baseline capability)
- DINO swap: adaLN-Zero controls composition independently
- Text manip: Cross-attention responds to caption changes

**Alternatives considered:**
- Only reconstruction: Doesn't prove dual conditioning works
- More tests: Diminishing returns, slows validation

**Trade-off:** None - these three are minimal sufficient set.

### Decision: Use Flux VAE decoder for validation (not training)
**Rationale:** Need to decode latents to images for visual inspection. Only used during validation (5 runs × 15 images = 75 decodes total), not training loop.

**Alternatives considered:**
- SDXL VAE: Lower quality, might miss visual artifacts
- No decoding: Can't visually verify results

**Trade-off:** Slight validation time overhead (~30 seconds per run), but necessary for interpretability.

## Risks / Trade-offs

**[Risk: 100 images too few to test generalization]**
→ **Mitigation:** This is intentional - we're testing overfitting/memorization ability, not generalization. Part C handles generalization.

**[Risk: 512×512 resolution hides full-resolution bugs]**
→ **Mitigation:** Acceptable - we're validating architecture correctness, not final quality. Part C trains at native resolutions.

**[Risk: Nano model succeeds but full model fails due to depth]**
→ **Mitigation:** 12 layers sufficient to test adaLN propagation and cross-attention depth. If Nano fails, full model will definitely fail.

**[Risk: Flow matching numerics differ at scale]**
→ **Mitigation:** Loss function and sampling are identical between Nano and full model. Numerics should transfer.

**[Risk: Validation metrics are subjective (visual inspection)]**
→ **Mitigation:** LPIPS provides quantitative reconstruction metric. Visual inspection is qualitative backup.

**[Trade-off: No gradient accumulation]**
→ Batch size 8-16 fits in 24GB for Nano model. Full model (Part C) will need accumulation, but Nano doesn't.

**[Trade-off: No mixed precision (bfloat16)]**
→ Nano model trains in float32 for simplicity. Full model must use bfloat16 (the-plan.md strict rule). If float32 Nano fails, bfloat16 full model will fail.

## Migration Plan

**Phase 1: Implementation (1-2 days)**
1. Implement Nano DiT architecture (model.py)
2. Implement validation dataloader (data.py)
3. Implement training loop (train.py)
4. Implement validation tests (validate.py)

**Phase 2: Smoke Test (30 minutes)**
1. Run 10 training steps, verify no crashes
2. Check loss is finite and decreasing
3. Verify checkpoints save correctly

**Phase 3: Full Validation Run (2-4 hours)**
1. Train for 5k steps
2. Monitor loss curve (should decrease monotonically)
3. Run validation at steps 1k, 2k, 3k, 4k, 5k
4. Visual inspection of generated images

**Phase 4: Go/No-Go Decision**
- **GO (proceed to Part C)** if:
  - Loss converges (final loss < 10% of initial loss)
  - Reconstruction LPIPS < 0.2
  - DINO swap shows composition transfer
  - Text manip shows spatial control
- **NO-GO (debug architecture)** if any metric fails

**Rollback Strategy:**
- Validation is read-only (no prod impact)
- Can delete validation/ directory and restart
- No dependencies on Part C

## Open Questions

None - design is complete and ready for implementation. All decisions based on proven practices from the-plan.md.
