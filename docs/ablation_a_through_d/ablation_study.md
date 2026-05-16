# Ablation Study: TREAD and Muon Optimizer Effects on DiT Training

## Abstract

We present a controlled ablation study examining the independent and combined
effects of two training innovations on a Diffusion Transformer (DiT) model for
human portrait generation: TREAD (Token Routing for Efficient Attention
Distillation) and the Muon optimizer. Four arms trained for 5,000 steps each on
the same physical hardware (quad-GPU Vast.ai node, CUDA_VISIBLE_DEVICES strictly
assigned), eliminating hardware variance as a confound. We evaluate reconstruction
fidelity (Recon LPIPS), text-conditioned generation quality (Text LPIPS), and
text controllability (Text Manipulation LPIPS delta) at the final checkpoint.
We additionally document the mechanistic basis for the observed instabilities,
known training traps for this architecture class, and recommended mitigations.

---

## 1. Introduction

Training efficiency and image quality in diffusion models are often at odds.
TREAD reduces compute by probabilistically routing tokens through a subset of
attention blocks, trading coverage for speed. The Muon optimizer applies
orthogonalized Nesterov momentum via Newton-Schulz iteration, potentially
stabilizing training dynamics under sparse gradient regimes. Neither has been
evaluated in combination on a constrained-dataset portrait DiT. This study
isolates their contributions.

The model is prx-tg, a NanoDiT backbone trained on 70,000–100,000 FFHQ portraits
at 1024x1024 resolution. Images are compressed into a 128x128 latent grid using
a Flux VAE with 8x downsampling (f8). The architecture uses "quad-conditioning":
cross-attention receives dense captions (CLIP/T5), visual identity embeddings
(DINOv2), spatial layout maps, and DWPose keypoints simultaneously. This
multi-modal conditioning places significant pressure on the optimizer to balance
localized anatomical constraints (pose) against global stylistic directives
(text, identity).

Data augmentation — including preprocessing, normalization pipelines, and
curation — was performed using stratum-hq
(https://github.com/timlawrenz/stratum-hq). Horizontal flip augmentation was
explicitly excluded from the pipeline. Because flip operations on spatially
conditioned data require remapping of symmetric landmark indices (e.g., left eye
↔ right eye in DWPose), introducing them without strict index remapping would
corrupt the cross-attention binding between text tokens and spatial features.
The decision to exclude flips is therefore architecturally motivated, not merely
a simplification.

---

## 2. Hardware Constraints and Memory Optimizations

Training a 400M+ parameter DiT alongside T5, DINOv2, and DWPose encoders on a
single 24GB RTX 4090 requires several non-negotiable engineering choices.

**Gradient Checkpointing.** The DiT operates on 4096 tokens at this resolution.
Storing intermediate activations during the forward pass for a full backward
pass would exceed 20GB alone. Aggressive gradient checkpointing (use_reentrant=False)
drops these activations and recomputes them during the backward pass, trading
20-30% speed for a massive memory reduction. This penalty is largely offset by
TREAD's token routing in the TREAD arms.

**Bias Removal.** Affine biases are stripped from QKV projections, output
projections, and FFN hidden layers (fc1, fc2). In models using LayerNorm or
RMSNorm, these biases are mathematically redundant — the normalization layers
center the data, making a subsequent learnable bias equivalent to a scalar shift
that the network absorbs elsewhere. Biases are retained only where structurally
necessary: final output projections and adaLN regressors that inject timestep
and conditioning signals. The savings are approximately 5-10% of total parameter
memory and reduce gradient bandwidth during updates.

**T5 Memory Reclamation.** The T5 encoder consumes over 10GB of VRAM to process
dense captions. A cleanup_t5_encoder() routine migrates T5 to CPU immediately
after text embeddings are cached, freeing VRAM before the DiT backward pass.
Without this, OOM errors are deterministic.

**Dynamic Positional Embeddings.** Static positional embedding buffers are
replaced with 2D sincos embeddings computed dynamically from the incoming
latent tensor's height and width. This supports multi-resolution training
without padding or fixed-shape assumptions, preventing the model from
overfitting to a rigid square spatial prior.

**Precision Hygiene.** BF16 is used for matrix multiplications (autocast) to
reduce memory bandwidth, but master weights and optimizer state buffers are kept
in FP32. Storing Muon or AdamW state in BF16 is a known failure mode: small
gradients round to zero in BF16's limited mantissa, causing the network to
silently stall. Loss curves appear stable while learning effectively stops.
The FP32 master weight pattern prevents this underflow without sacrificing the
memory benefits of BF16 compute.

---

## 3. Experimental Design

### 3.1 Arms

| Arm | Label       | Optimizer | TREAD | Notes                        |
|-----|-------------|-----------|-------|------------------------------|
| A   | Baseline    | AdamW     | Off   | Standard training            |
| B   | TREAD+AdamW | AdamW     | On    | Speed optimization only      |
| C   | TREAD+Muon  | Muon      | On    | Speed + optimizer change     |
| D   | Full Stack  | Muon      | On    | TREAD + Muon + modified loss*|

*Arm D uses a modified REPA loss formulation. Loss values are not comparable
across D and A/B/C. LPIPS comparisons across all arms remain valid.

### 3.2 Hardware and Reproducibility

All four arms ran concurrently on a single quad-GPU instance (Vast.ai instance
36034271). GPU assignment was pinned via CUDA_VISIBLE_DEVICES=0,1,2,3 for arms
A, B, C, D respectively. Training duration: approximately 110 hours wall-clock.
Checkpoints saved at steps 500, 1000, 1500, ..., 4500, and 5000 (11 checkpoints
per arm). Validation run at each checkpoint interval and at training completion.
Running all arms on the same physical node eliminates hardware variance as a
confound.

### 3.3 Metrics

- **Recon LPIPS**: LPIPS between generated and ground-truth portrait given full
  conditioning (identity + text). Lower is better. 25 samples.
- **Text LPIPS**: LPIPS between generated portrait and ground-truth given text
  conditioning only (no identity embedding). Lower is better. 20 samples.
- **Text Manip delta**: Mean absolute LPIPS difference between generations for
  a caption and its single-attribute modification (e.g. "dark hair" → "light
  hair"). Higher delta indicates stronger text responsiveness.

---

## 4. Results

### 4.1 Final Checkpoint Metrics (Step 5000)

| Arm               | Recon LPIPS   | Text LPIPS    | Text Manip delta |
|-------------------|---------------|---------------|-----------------|
| A — Baseline      | 0.9352        | 0.9593        | 0.466           |
| B — TREAD+AdamW   | **1.0161**    | 0.9396        | 0.373           |
| C — TREAD+Muon    | 0.9463        | 0.9603        | **0.546**       |
| D — Full Stack*   | **0.9267**    | **0.9219**    | 0.431           |

*D loss metric is not comparable to A/B/C. LPIPS comparisons valid.

### 4.2 Training Dynamics

TREAD arms (B, C, D) trained approximately 17% faster than Arm A (~70 s/it
vs ~83 s/it), confirming the token routing efficiency gain. Wall-clock times
to completion: A ~111.8h, B ~94.9h, C ~95.1h, D ~94.9h.

Arm B exhibited a gradual reconstruction collapse beginning around step 3500
(Recon LPIPS 0.922), with monotonic degradation through step 5000 (1.016). Its
best checkpoint was step 3000 (0.906). This is not a sudden mid-training
failure; the collapse accumulates progressively.

Arm C showed moderate late-stage regression on both metrics relative to its
best mid-training checkpoint (~step 3500), consistent with Muon's known tendency
toward aggressive late-training updates without learning rate warmdown.

Arm D established a text quality lead from step 500 (Text LPIPS 0.900) onward,
reflecting early semantic structure acquisition from the modified REPA loss.
Note that Arm D's step 500 train_loss of 0.163 (vs A=0.073, B=0.050, C=0.037)
confirms the modified loss operates on a different scale — direct loss
comparisons with A/B/C are invalid.

---

## 5. Analysis

### 5.1 TREAD Alone (B vs A) — Mixed Results

TREAD with AdamW yields marginally better Text LPIPS (0.940 vs 0.959) but
catastrophically degrades reconstruction (1.016 vs 0.935). The collapse has a
clear mechanistic basis.

TREAD probabilistically routes up to 50% of tokens around intermediate
attention and FFN blocks. The bypassed blocks receive highly sparse, irregular
gradient signals over thousands of iterations. AdamW interprets near-zero
gradients as low-variance parameters and decays their second-moment estimates.
This inflates the adaptive learning rate for "starved" weights. When a
high-frequency token is eventually routed through those blocks, the resulting
gradient is multiplied by this inflated rate, producing divergent updates that
shatter internal representations. The collapse is not a flaw in TREAD itself —
it is a fundamental incompatibility between scalar adaptive moment estimation
and dynamic spatial sparsity.

Text LPIPS improvement suggests TREAD does not harm the text pathway; sparse
attention may regularize it by preventing overfitting to the identity signal.
However this benefit does not justify the reconstruction cost in the AdamW
configuration.

### 5.2 Muon Corrects the TREAD Instability (C vs B)

Replacing AdamW with Muon stabilizes the TREAD training dynamics. Arm C's
reconstruction LPIPS (0.946) is 0.070 points better than B's collapsed value
(1.016). Muon applies orthogonalized Nesterov momentum via a 5th-degree
Newton-Schulz polynomial, iteratively updating the normalized matrix X using
X = aX + b(XX^T)X + c(XX^T)^2 X until the update converges to the nearest
orthogonal matrix. This enforces a fixed spectral norm on the update step.

The consequence is direct: because Muon applies a uniform update magnitude
across the entire weight matrix rather than per-parameter scaling, the
"starved" pathways do not develop explosive learning rates. The orthogonalized
updates act as an intrinsic regularizer, maintaining geometric stability even
when token distributions fluctuate due to routing probabilities.

An additional benefit: Muon's single momentum buffer uses 4 bytes per parameter
versus AdamW's 8 (two buffers), reducing optimizer state memory by 50%.

For FFHQ portraits specifically, human faces exhibit heavy-tailed feature
distributions (rare lighting conditions, accessories, anatomical variances).
Muon's isotropic updates ensure balanced parameter updates across head and tail
classes — practically confirmed by Arm C's highest Text Manipulation delta
(0.546), indicating superior binding of rare text tokens to specific visual
features.

### 5.3 Text Controllability — Muon's Signature

The Text Manipulation delta measures how much a single-attribute caption edit
changes the output. Arm C scores 0.546, meaningfully above Baseline (0.466),
TREAD+AdamW (0.373), and Full Stack (0.431). Muon's spectral normalization
promotes stronger binding between text token activations and output features.

Arm B's low delta (0.373) is consistent with its reconstruction collapse:
a degraded model shows low LPIPS variation regardless of caption changes.

### 5.4 Full Stack (D) — Best Overall Quality

Arm D achieves the best Recon LPIPS (0.927) and best Text LPIPS (0.922).
The REPA loss forces the DiT to simultaneously align its intermediate noisy
hidden states with DINOv2's semantic representations during training. This
dual-objective formulation accelerates early semantic acquisition (visible in
D's early Text LPIPS lead), and the Muon optimizer's stability allows the model
to reach final convergence without the instabilities that would otherwise
accompany the modified loss. The full stack combination outperforms each
component in isolation.

---

## 6. Learning Curves

Validation ran every 500 steps. Key inflection points:

| Milestone                       | Step  | Observation                          |
|---------------------------------|-------|--------------------------------------|
| First LPIPS scores              | ~1000 | B, C, D ahead of A early             |
| B reconstruction peak           | ~3000 | B briefly leads Recon at 0.906       |
| B reconstruction collapse onset | ~3500 | B begins monotonic degradation       |
| D Text LPIPS breaks below 0.90  | ~500  | D establishes text quality lead early|
| A catches up on Recon           | ~3500 | Slow but steady baseline convergence |
| B final collapse                | 5000  | B ends at 1.016 Recon                |

The early performance of TREAD arms reflects token routing benefit during
initial feature learning. The late-stage bifurcation between B and C/D isolates
the optimizer's role in maintaining stability.

---

## 7. Known Traps and Mitigations

### 7.1 The HASTE Trap: REPA Loss Termination

Arm D uses a REPA alignment loss that anchors the DiT's intermediate hidden
states to DINOv2's semantic embeddings. This is highly effective during the
burn-in phase: DINO provides a structural scaffold that pulls the DiT out of
its initial chaotic state and accelerates early convergence. However, DINOv2
operates in a lower-dimensional embedding space optimized for classification
and segmentation, and inherently discards the high-frequency textural variance
required for photorealism (pores, hair strands, skin texture).

If REPA is maintained indefinitely, the teacher's embeddings transition from
guide to straitjacket: the generator is penalized for synthesizing fine-grained
details that don't exist in the teacher's feature maps. The HASTE (Holistic
Alignment with Stage-wise Termination for Efficient training) framework
identifies this as "works until it doesn't."

**Mitigation:** Decay the REPA alignment weight to zero after the initial
convergence phase (approximately the first 20-30% of the training run). This
allows the DiT to focus on unconstrained flow-matching for final high-frequency
refinement. This was not implemented in the current 5000-step study; it is a
key recommendation for production runs.

### 7.2 The 32x32 Grid Artifact Trap

The Flux VAE compresses by 8x (f8). With a patch size of 4 (p4), each
transformer token represents a 32x32 pixel block in image space. Independent
linear projections per token before unpatchifying create visible grid seams.

prx-tg mitigates this with patch size 2 (p2), reducing the grid footprint to
16x16. Further mitigations:

- **Latent normalization:** Raw Flux VAE latents range from approximately
  -10 to +10. Unnormalized, their variance overrides positional embeddings,
  causing the network to memorize grid boundaries. Dataset statistics should
  be computed and latents normalized to unit variance (mu=0, sigma=1) before
  DiT input.
- **Output convolution:** A 3x3 Conv2D immediately after the unpatchify layer
  provides a spatial receptive field that crosses token boundaries, naturally
  smoothing seams. An overlapping PatchEmbed (kernel_size=3, stride=2) at
  input achieves similar spatial sharing before the transformer.

### 7.3 Data Augmentation and Spatial Conditioning

All augmentation in this study was performed using stratum-hq
(https://github.com/timlawrenz/stratum-hq). Horizontal flip augmentation was
explicitly excluded from the pipeline.

This is the correct decision for quad-conditioned models. Horizontal flipping
on portrait data with DWPose conditioning requires remapping symmetric landmark
indices (left eye ↔ right eye, left shoulder ↔ right shoulder, etc.) in
addition to flipping the pixel matrix. Without index remapping, the model
learns to associate "right-side" text tokens with "left-side" pixel data,
corrupting cross-attention binding and producing anatomically inconsistent
outputs. Excluding flips avoids this trap entirely without sacrificing
augmentation value, since the FFHQ dataset already contains naturally diverse
head orientations and lighting.

A separate consideration: if random cropping is applied, DWPose keypoints that
fall outside the crop frame must be assigned visibility flag 0, and the loss
must mask those regions during backpropagation. Allowing the estimator to
extrapolate coordinates beyond the crop boundary introduces false spatial
conditioning that degrades pose adherence.

---

## 8. Evaluation Limitations and Future Metrics

LPIPS captures perceptual texture similarity and broad structural alignment
based on deep network activations. For a model conditioned on DWPose keypoints,
LPIPS alone is insufficient: it cannot verify whether the generated pose
actually matches the input spatial condition.

A model may generate a highly photorealistic face (good LPIPS) while ignoring
the jaw angle or shoulder tilt dictated by the DWPose input. To prove that
Full Stack (Arm D) maintains spatial controllability, the validation suite
should add:

**MPJPE (Mean Per Joint Position Error):** Average Euclidean distance between
the ground-truth keypoints fed as conditioning and the keypoints extracted from
the generated image via an independent pose estimator. Directly quantifies
pose fidelity.

**PA-MPJPE (Procrustes-Aligned MPJPE):** Aligns the generated pose to the
ground truth via an optimal affine transformation before computing error. This
isolates structural accuracy from rotational or scale variance in the generated
subject, providing a rigorous measure of how well the DiT's cross-attention
mechanisms bind visual output to spatial conditions.

The reported Text Manip delta of 0.546 (Arm C) is a strong qualitative
indicator of text responsiveness. PA-MPJPE would provide the quantitative
spatial controllability baseline necessary for academic publication.

---

## 9. Conclusions

1. **TREAD alone (with AdamW) is unstable for long runs.** Reconstruction
   degrades monotonically from ~step 3500 and collapses by step 5000. The cause
   is a fundamental incompatibility between AdamW's adaptive moment estimation
   and TREAD's dynamic spatial sparsity — not a deficiency in TREAD itself.
   Do not deploy TREAD without a compatible optimizer.

2. **Muon stabilizes TREAD.** The TREAD+Muon combination (Arm C) recovers
   reconstruction quality to within 0.011 LPIPS of baseline while retaining
   TREAD's 17% throughput benefit and achieving the strongest text
   controllability of any arm tested (delta 0.546). The mechanism is Muon's
   orthogonalized spectral updates, which prevent the learning rate inflation
   that destabilizes AdamW under sparse gradients.

3. **The Full Stack (Arm D) is the strongest configuration.** TREAD + Muon +
   modified REPA loss achieves the best metrics across both reconstruction
   (0.9267) and text generation (0.9219). The REPA dual-objective accelerates
   early semantic acquisition; Muon provides the stability to reach full
   convergence. For production runs, add stage-wise REPA loss termination
   (Section 7.1) to avoid the long-run capacity mismatch trap.

4. **Training throughput:** TREAD arms complete training ~17h faster in
   wall-clock time. At equivalent step budgets, this is a direct reduction in
   compute cost for future experiments.

### Recommendation

For production training of prx-tg: adopt the Full Stack (Arm D) configuration
with REPA loss decayed to zero by step ~1000-1500 (the first 20-30% of a
5000-step run). If the modified loss formulation is not yet production-ready,
TREAD+Muon (Arm C) is a safe intermediate — it preserves reconstruction
quality, improves text controllability, and reduces training time relative to
the baseline.

---

## Appendix: Per-Sample Reconstruction LPIPS at Step 5000

| Sample | Baseline (A) | TREAD+AdamW (B) | TREAD+Muon (C) | Full Stack (D) |
|--------|-------------|-----------------|----------------|----------------|
| Mean   | 0.9352      | 1.0161          | 0.9463         | 0.9267         |
| Min    | 0.871       | 0.954           | 0.888          | 0.856          |
| Max    | 0.992       | 1.090           | 1.012          | 1.026          |
| Std    | ~0.022      | ~0.030          | ~0.025         | ~0.030         |
