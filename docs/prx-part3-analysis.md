# PRX Part 3 → prx-tg: Research Analysis

## Problem Statement

PRX Part 3 presents the culmination of Photoroom's diffusion research: a **24-hour speedrun** combining all their best tricks from Parts 1 & 2 into a single training recipe. Our prx-tg model was directly inspired by Parts 1 & 2. This analysis identifies what's new in Part 3, compares it to our current approach, and recommends what to adopt.

Source: https://huggingface.co/blog/Photoroom/prx-part3

## Article Summary

PRX Part 3 trained a text-to-image model from scratch on **32 H200 GPUs** (~$1,500 total) using:
- **X-prediction in pixel space** (no VAE)
- **Perceptual losses** (LPIPS + DINO)
- **TREAD token routing** (skip 50% of tokens through middle blocks)
- **REPA alignment** (DINOv3 at block 8, weight 0.5)
- **Muon optimizer** (for 2D parameters; Adam for the rest)
- **8.7M synthetic images** (3 public datasets, re-captioned with Gemini 2.5 Flash)
- **Two-stage schedule**: 512px (100k steps, batch 1024) → 1024px (20k steps, batch 512)

---

## Technique-by-Technique Comparison

### 1. 🔴 X-Prediction / Pixel-Space Training (MAJOR CHANGE)

**Article**: Predicts clean image x₀ directly in pixel space. Patch size 32 with 256-dim bottleneck. No VAE at all. Sequence length: 256 tokens at 512px, 1024 tokens at 1024px.

**Us**: V-prediction (velocity) in Flux VAE latent space. Patch size 2. Sequence length: 1024 tokens at 512px (64×64 latent / 2).

**Assessment**: This is the article's most fundamental departure. Going to pixel space eliminates the VAE entirely, simplifies the pipeline, and unlocks perceptual losses trivially. **However**, this is a complete architectural rewrite — different patch sizes, different token counts, different loss target, different data pipeline (no more pre-computed VAE latents).

**Recommendation**: **Monitor but don't adopt yet.** Our pre-computed VAE latent pipeline is already built and working. The pixel-space approach is compelling but would require rebuilding from scratch. Consider this for a future "v2" if latent-space quality plateaus.

---

### 2. 🟡 Perceptual Losses — LPIPS + DINO (HIGH VALUE, MODERATE EFFORT)

**Article**: On top of flow matching loss, adds:
- **LPIPS loss** (weight 0.1): Low-level perceptual similarity
- **DINO perceptual loss** (weight 0.01): Semantic similarity via DINOv2 features

Applied on pooled full images at all noise levels. "Small overhead, consistent quality boost."

**Us**: Pure MSE flow matching loss only.

**Assessment**: This is the **single most impactful idea** we could adopt. The catch: in latent space, computing LPIPS/DINO losses requires decoding predicted latents back to pixels via the VAE decoder each training step. This is expensive on a single 4090.

**Possible approaches for us**:
1. **Latent-space perceptual loss**: Define a perceptual loss directly on the predicted vs. target latents using a small discriminator or feature extractor trained on latent space. Avoids decoding overhead entirely.
2. **Periodic pixel-space loss**: Decode to pixels every N steps (e.g., every 10 accumulation steps) and compute perceptual loss. Amortizes the cost.
3. **Lightweight DINO-on-latents**: Since our DINOv3 embeddings are pre-computed for each image, we could add a loss that encourages the model's intermediate representations to align with the DINOv3 features — which is basically REPA (see #4 below).

**Recommendation**: **Adopt REPA first** (see below) as it gives us representation alignment without decoding. Consider latent-space perceptual losses as a follow-up experiment.

---

### 3. 🟡 TREAD Token Routing (HIGH VALUE, MODERATE EFFORT)

**Article**: Randomly selects 50% of tokens to bypass blocks 2 through (N-1). Routed tokens re-enter at the penultimate block. Uses self-guidance (dense vs. routed conditional prediction) instead of vanilla unconditional CFG.

**Us**: All tokens pass through all 18 blocks. No routing.

**Assessment**: TREAD effectively **halves the compute** for the middle blocks. On our single 4090, where each training step is expensive, this could be transformative — potentially doubling throughput or allowing larger batch sizes. The self-guidance modification to CFG is important (standard CFG degrades with routing).

**Key considerations for our setup**:
- We have 18 blocks; routing from block 2 to block 17 means 15 blocks process only half the tokens
- Self-guidance replaces unconditional CFG with dense-vs-sparse comparison — changes our sampling code too
- Implementation is relatively self-contained (routing logic in the transformer forward pass)

**Recommendation**: **Strong candidate for adoption.** The throughput gain alone justifies the implementation cost. Start with a validation experiment on the small model.

---

### 4. 🟢 REPA — Representation Alignment (HIGH VALUE, LOW EFFORT) ✅ IMPLEMENTED

**Article**: Adds alignment loss between transformer hidden states (at block 8) and DINOv3 teacher features. Weight 0.5. Only computed on non-routed tokens (if using TREAD).

**Us**: We already have DINOv3 embeddings (CLS + patches) pre-computed and use them for conditioning, but we have **no alignment loss** that encourages internal representations to match DINOv3.

**Assessment**: This is the **easiest win**. We already load DINOv3 patch features for every training sample. REPA adds a single linear projection at block 8 (or whichever middle block we choose) and a cosine similarity loss between the projected hidden states and the DINOv3 features. No new data needed, no architectural upheaval.

**Implementation** (completed on `repa` branch):
1. Added `repa_proj = nn.Linear(hidden_size, dino_patch_dim)` at configurable block (default: `depth // 2`)
2. Cosine similarity loss between projected hidden states and raw DINOv3 patch features
3. Loss weight 0.5, respects padding mask for variable-length patches
4. Adds ~786K parameters (0.2% of model), minimal compute overhead

**Recommendation**: ✅ **Adopted as first experiment.**

---

### 5. 🟢 Muon Optimizer (MODERATE VALUE, LOW EFFORT)

**Article**: Muon for 2D parameters (matrices), Adam for everything else. Both at lr=1e-4 with momentum=0.95, Nesterov momentum, 5 Newton-Schulz steps.

**Us**: AdamW for everything, lr=3e-4, betas=(0.9, 0.95).

**Assessment**: Muon showed "clear improvement over Adam" in their Part 2 experiments. The dual-optimizer setup is straightforward — split parameters by dimensionality and use different optimizers for each group. The `muon_fsdp_2` implementation exists as a library, though we'd need the non-FSDP version for single-GPU.

**Key considerations**:
- Muon is designed for matrix-shaped parameters — applies Newton-Schulz orthogonalization
- Our AdamW with lr=3e-4 may be over-tuned for our specific setup; Muon at 1e-4 is a different regime
- Need to verify single-GPU Muon implementation exists (the FSDP version is multi-GPU)

**Recommendation**: **Adopt as a low-risk experiment.** Swap optimizer, keep everything else constant, compare loss curves. Easy A/B test.

---

### 6. 🟡 EMA Configuration (LOW VALUE, TRIVIAL EFFORT)

**Article**: smoothing=0.999, update every 10 batches, start from step 0.

**Us**: decay=0.9999, update every step, warmup from 0 over 500 steps.

**Assessment**: Minor difference. Their more aggressive EMA (0.999 vs 0.9999) means the EMA tracks the training weights more closely, which might be better for shorter training runs. Updating every 10 batches vs. every step is a minor compute saving. Our warmup approach is actually more conservative and arguably better for from-scratch training.

**Recommendation**: **Keep current approach.** Our warmup strategy is sound. If we switch to Muon and train faster, consider testing 0.999 decay.

---

### 7. 🟡 Training Schedule: 512→1024 (MODERATE VALUE, MODERATE EFFORT)

**Article**: Start at 512px for 100k steps, then fine-tune at 1024px for 20k steps. Drop REPA during the 1024 stage.

**Us**: Train at native bucket resolutions (704-1344px) from the start.

**Assessment**: Starting at lower resolution and fine-tuning up is a proven technique for faster convergence — the model learns structure at 512px (cheaper per step) then learns high-frequency detail at 1024px. We currently train at full resolution from step 0, which is more expensive per step.

**Key consideration**: Our bucket-aware data pipeline supports multiple resolutions already. We could add a "resolution phase" that starts with only 1024×1024 (or 512×512-equivalent) buckets and adds larger buckets later. This requires config changes, not code changes.

**Recommendation**: **Consider for next training run.** Add resolution scheduling to the config (start with smaller buckets, add larger ones at step N).

---

### 8. 🔴 Dataset Scale & Synthetic Data (HIGH VALUE, HIGH EFFORT)

**Article**: 8.7M images from 3 synthetic datasets (Flux-generated, FLUX-Reason, Midjourney). Re-captioned with Gemini 2.5 Flash.

**Us**: ~15k curated real images (targeting 86k). Captioned with Gemma3:27b.

**Assessment**: The scale gap is enormous: 8.7M vs 15k (580× difference). The article explicitly states remaining issues are "undertraining artifacts and limited data diversity." We're far more data-limited. However, our dataset is curated for a vertical domain, which partially compensates.

**Ideas**:
- **Use the same public datasets** (Flux-generated, FLUX-Reason-6M, midjourney-v6-llava) for pre-training, then fine-tune on our curated vertical data
- **Re-caption** our images with Gemini 2.5 Flash for consistency
- **Synthetic augmentation**: Generate additional training images using Flux/SDXL in our domain

**Recommendation**: **Strategic priority.** Supplementing with synthetic pre-training data is the highest-leverage dataset improvement. Consider a two-phase approach: pre-train on public synthetic data, then fine-tune on our vertical dataset.

---

### 9. 🟡 Self-Guidance for CFG (MODERATE VALUE, DEPENDS ON TREAD)

**Article**: Instead of standard CFG (conditional vs. unconditional), uses dense-vs-routed conditional predictions. This is specifically because TREAD routing degrades vanilla CFG quality.

**Us**: Standard dual CFG with unconditional baseline (3 forward passes: uncond, text-only, DINO-only).

**Assessment**: Only relevant if we adopt TREAD. Standard CFG works fine without routing. Self-guidance is a paired requirement with TREAD.

**Recommendation**: **Adopt together with TREAD** if we implement token routing.

---

## Priority Ranking

| Priority | Technique | Impact | Effort | Risk |
|----------|-----------|--------|--------|------|
| **1** | REPA (DINOv3 alignment) | High | Low | Low |
| **2** | Muon Optimizer | Moderate-High | Low | Low |
| **3** | TREAD Token Routing | High | Moderate | Medium |
| **4** | Synthetic Data Pre-training | Very High | High | Low |
| **5** | Resolution Scheduling | Moderate | Low | Low |
| **6** | Perceptual Losses | High | High | Medium |
| **7** | Pixel-Space Training | Very High | Very High | High |

---

## Recommended Experiment Sequence

### Phase 1: Quick Wins (Current Architecture)
1. ✅ **Add REPA loss** at block `depth // 2` with DINOv3 alignment (weight 0.5)
2. **Try Muon optimizer** for 2D parameters (A/B test vs. AdamW)
3. **Test resolution scheduling** (512px start → full resolution)

### Phase 2: Throughput Optimization
4. **Implement TREAD routing** (50% tokens, blocks 2→17)
5. **Switch to self-guidance CFG** (required for TREAD)

### Phase 3: Data & Architecture
6. **Pre-train on public synthetic data** before fine-tuning on vertical dataset
7. **Evaluate pixel-space training** as potential v2 architecture

---

## Key Papers

- [Li and He, 2025](https://arxiv.org/abs/2511.13720) — X-prediction / pixel-space formulation
- [Ma et al.](https://arxiv.org/abs/2602.02493) — PixelGen: perceptual losses for pixel diffusion
- [Krause et al., 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Krause_TREAD_Token_Routing_for_Efficient_Architecture-agnostic_Diffusion_Training_ICCV_2025_paper.pdf) — TREAD token routing
- [Yu et al., 2024](https://arxiv.org/abs/2410.06940) — REPA representation alignment
- [Krause et al., 2025](https://arxiv.org/abs/2601.01608) — Self-guidance for token-sparse models
- [Siméoni et al., 2025](https://arxiv.org/abs/2508.10104) — DINOv3 (teacher for REPA)
- [Muon optimizer](https://github.com/samsja/muon_fsdp_2) — FSDP implementation
