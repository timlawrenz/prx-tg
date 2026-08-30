# Nyström Self-Attention for NanoDiT — Technical Design

> **Status:** Design document — no execution yet.
> **Active production run:** `experiments/2026-06-04_2116/` (40k steps on 70k Stratum, ~42h remaining as of 2026-06-18).
> **No code changes until the active run completes and experiment branches are properly separated.**

## Goal

Replace `F.scaled_dot_product_attention` in NanoDiT self-attention with a Nyströmformer low-rank approximation. Answer two questions in sequence:

1. **Q1 (viability):** Does Nyström with uniform landmark selection match full-attention baseline quality?
2. **Q2 (face-biasing):** Does per-sample stratified landmark selection informed by Sapiens segmentation further improve over uniform Nyström?

The ablation is staged: baseline → uniform Nyström → face-biased Nyström. Each arm gates the next.

## Architectural Context

| Component | Sequence Length | Attention Type | Nyström Applicable? |
|-----------|----------------|----------------|---------------------|
| Self-attention | N ≈ 3700–4096 (64×64 patches at 1024²) | `F.scaled_dot_product_attention(q, k, v)` | ✅ Yes — square Gram matrix |
| Cross-attention | Q: 4096, KV: ~4528 | Same SDPA | ❌ No — rectangular cross-similarity, not a Gram |

**Cross-attention is excluded.** The Nyström method requires a positive semidefinite kernel on a single set. Cross-attention Q-K are from different sequences and would need a different approach (Linformer, Performer, etc.).

## Key Decisions (locked)

### 1. Landmark Count: `m = 64`
- 64 / 4096 = 1.56% of tokens as landmarks
- Compute: O(N·m + m³) ≈ O(262K + 262K) per head vs O(N²) = 16.7M
- **~98% compute reduction** for the attention step
- m=128 would double the landmark kernel SVD cost with diminishing returns
- m=64 is the standard recommendation from the Nyströmformer paper for N≈4000

### 2. Face-Biased Landmark Selection (Arm C only, optional)

When enabled, landmarks are allocated proportional to region importance using the Sapiens seg_map:
- **Per-sample, stratified** via seg_map `(B, 64, 64)` — Sapiens class IDs at token resolution
- Face classes: {2, 3, 23, 24, 25, 26, 27} (Face_Neck, Hair, Lips)
- Allocation: 60% of landmarks to face region, 40% to background (configurable `face_ratio`)
- Within each region: uniform segmentation (simple, deterministic, no extra params)
- Landmark indices pre-computed in training loop, passed to model as `(B, m)` tensor
- Model uses `torch.gather` for batched per-sample landmark extraction — single fused GPU kernel, no Python loop

**Why it might help:** Face patches carry higher-frequency detail and more complex inter-patch dependencies. Background patches are smoother and more self-similar. Shifting landmark budget toward higher-rank regions improves global approximation quality for the same `m`.

**But this is Arm C — only run it if Arm B (uniform) passes the viability gate.** If uniform Nyström is already worse than baseline, stratified sampling won't fix it.

**Seg map alignment:** For 1024×1024 images with patch_size=16, the DiT token grid is 64×64 and the Sapiens seg_grid is also 64×64 — 1:1 alignment. Each seg_grid cell maps to exactly one DiT token. For non-square aspect ratios, the seg_grid must be upsampled to match the actual patch grid dimensions.

### 3. Layers: Last 6 (layers 12–17)
- Applied only to the last 6 of 18 layers
- Early layers (0–11) retain full `F.scaled_dot_product_attention`
- This targets the layers where fine-grained features are processed and where approximation error might be most tolerable
- Layers 12-17 are after the TREAD routing block (route_start=1, route_end=-1=depth-2=16) so they process the full reassembled sequence
- Configurable via `model.nystrom_self_attn.layers` for future sweeps

### 4. Inference Fallback
- During sampling/inference (35 Euler steps with self-guidance), seg_map is not available
- **Uniform landmark selection** at inference time — model falls back to evenly-spaced landmarks
- Face-biasing is a training signal quality optimization; inference speed is already fast

### 5. Nyström Algorithm
Standard Nyströmformer with SVD pseudoinverse for the landmark kernel:

```
Given: Q, K, V of shape (B, H, N, D), landmark indices of shape (B, m)

1. Extract landmark K, V:  k_land = gather(K, indices), v_land = gather(V, indices)
2. Extract landmark Q:     q_land = gather(Q, indices)
3. Landmark kernel:        Ã = softmax(Q_land · K_land^T / √D)           → (B, H, m, m)
4. Cross-kernel:           K̃ = softmax(Q · K_land^T / √D)               → (B, H, N, m)
5. Pseudoinverse:          U, S, Vt = SVD(Ã);  S_inv = 1/S where S > ε
                           Ã⁺ = Vt^T · diag(S_inv) · U^T
6. Two-step application:   Z = K̃^T · V                                  → (B, H, m, D)
                           Z = Ã⁺ · Z                                   → (B, H, m, D)
                           output = K̃ · Z                               → (B, H, N, D)
```

Numerical stability: ε = 1e-6 on singular values. bfloat16 SVD on 64×64 matrices is well-conditioned (tested empirically — max condition number ~10²–10³ for attention kernels).

## Experiment Design

Three arms, solving two independent questions:

| Arm | Dir | Question |
|-----|-----|----------|
| A: `70k-baseline-5k` | `experiments/70k-baseline-5k/` | Reference — full SDPA attention |
| B: `nystrom-uniform-5k` | `experiments/nystrom-uniform-5k/` | Q1: Does Nyström (uniform) match baseline? |
| C: `nystrom-face-biased-5k` | `experiments/nystrom-face-biased-5k/` | Q2: Does face-biasing improve over uniform? |

**Execution order:** A → B → C. If B fails (Nyström-uniform significantly worse than baseline), skip C — face-biasing won't rescue a broken approximation. If B passes, C isolates the effect of stratified sampling.

### Arm A: `70k-baseline-5k` (Reference)

- Config: fork `experiments/2026-06-04_2116/config.yaml`
- `total_steps: 5000`, `checkpoint.save_every: 500`
- All features as production baseline (Muon, TREAD, REPA, AsymFlow, FP8, pose)
- REPA: full weight 0.5 throughout (decay schedule is 8000-16000, irrelevant at 5k)
- `stratum_max_samples: 70000`
- Fresh run from scratch. LR schedule matched to 5k steps.

### Arm B: `nystrom-uniform-5k` (Q1: Nyström viability)

- Fork Arm A config, add:
  ```yaml
  model:
    nystrom_self_attn:
      enabled: true
      num_landmarks: 64
      face_biased: false           # <— uniform landmarks
      layers: "last_6"
      svd_epsilon: 1.0e-6
  ```
- **No seg_map dependency.** Landmark indices computed uniformly within the model or pre-computed as evenly-spaced indices.
- All other config identical to Arm A.

**Gate at step 5000:** Training loss and validation loss within 5% of Arm A. If loss diverges (>10%), Q1 fails → stop, investigate (SVD stability? wrong layers? interaction with TREAD?). Do not proceed to Arm C.

### Arm C: `nystrom-face-biased-5k` (Q2: face-biasing benefit)

- Fork Arm B config, change:
  ```yaml
  model:
    nystrom_self_attn:
      face_biased: true            # <— stratified face-biased landmarks
      face_ratio: 0.6
  ```
- Requires `seg_map` in the batch (available in Stratum data path).
- All other config identical to Arm B.

**Gate at step 5000:** Compare to Arm B. Does face-biasing improve training/validation loss, REPA alignment, or subjective image quality? If B ≈ C, face-biasing provides no benefit for this architecture — drop it. If C > B, the stratified approach is validated.

### Evaluation Framework

**Problem:** The model memorizes the 70k training set. Training loss, validation loss (no held-out set), and REPA alignment are carry-overs from the proving-ground phase and do not measure generation quality. The project goal (from GitHub) is: "Prove that high-quality text-to-image models can be trained on small, vertical datasets using consumer hardware." We need face-generation-specific evaluation.

**What we dropped (proving-ground carryovers):**
- ~~Reconstruction LPIPS~~ — model memorizes; LPIPS→0 is meaningless
- ~~DINO swap~~ — proved DINO dominates identity (generated image looks 100% like DINO source). Question answered, test no longer needed.
- ~~Text manipulation~~ — LPIPS diff is a weak proxy ("different" ≠ "correctly different")
- ~~CFG divergence~~ — already skipped in self-guidance mode
- ~~Text-only LPIPS~~ — compares to training image (memorization)

**Adopted evaluation: two-tier approach**

**Tier 1 — During training (every 500 steps, lightweight):**

Generate 10 faces from 10 fixed prompts at fixed seeds. Same prompts and seeds across all arms and all steps. Log images to TensorBoard for visual progression — watch the model learn.

Run DWPose (already in the pipeline for training conditioning) on each generated face:
- `validation/pose_confidence_{i}` — mean keypoint confidence for prompt i (10 separate scalars, i=0..9)

**Important:** Keypoint count is NOT a valid quality metric — a side profile legitimately has fewer visible keypoints than a frontal portrait. Only keypoint confidence is meaningful, and only as an intra-sample temporal trajectory (same prompt, same seed, across steps). Cross-prompt comparison of absolute confidence is meaningless (different poses, angles, occlusion). The signal is: does confidence grow over training for each individual prompt?

Also compute CLIP Score and LAION Aesthetic Score on each generated face:
- `validation/clip_score_{i}` — CLIP cosine similarity between generated image and its prompt (10 separate scalars, i=0..9)
- `validation/aesthetic_score_{i}` — LAION Aesthetic Predictor V2 score (CLIP ViT-L/14 embedding → MLP → 1-10 scalar, 10 separate scalars, i=0..9)

CLIP Score is cheap (~50ms for 8 images on 4090) and tells us whether the model is respecting the text conditioning. The LAION Aesthetic Predictor V2 (`sac+logos+ava1-l14-linearMSE.pth`, via `camenduru/improved-aesthetic-predictor` on HuggingFace) is an MLP head on top of the same CLIP ViT-L/14 embedding we're already computing — so it's essentially free. It outputs a 1-10 score trained on the AVA dataset (human aesthetic ratings). Score 5.5+ is considered good.

The aesthetic score counters a specific human bias: if a generated face is attractive, the viewer may overlook artifacts (haze, blur, texture degradation). The aesthetic predictor evaluates the entire image — backgrounds, texture quality, composition — not just the face, so artifacts that look fine behind a pretty face still drag the score down.

Like DWPose confidence, the meaningful signal for both CLIP Score and aesthetic score is temporal (does it grow for each prompt over training?), not cross-prompt.

DWPose is face-specific, already in the codebase, and ~20-50ms per image.

Log images to TensorBoard as `validation/generated_{i}` (i=0..9) at each validation step.

**Tier 2 — Post-training (once, after step 5000 checkpoint, heavyweight):**

| Metric | Source | Method |
|--------|--------|--------|
| FID | pytorch-fid | Generate 10K faces, compute FID vs held-out FFHQ test split |
| NN retrieval | Custom | For 100 generated faces, find nearest neighbor in training set (DINO cosine) — memorization check |
| F-Eval | F-Bench (when code released) | Score 500 generated faces on quality, authenticity, correspondence, identity |

**Note on F-Eval:** F-Bench (ICCV 2025) is the ideal face-specific benchmark — reference-free, human-preference-aligned, LMM-based. However, as of 2026-06-18, the GitHub repo (`github.com/MediaX-SJTU/F-Bench`) contains only a project page — no code or model weights released. F-Eval is based on a large multimodal model (vision encoder + face encoder + LLM + LoRA experts), likely 7B+ params, which would be impractical to run during training on a 4090 already loaded with the DiT. F-Eval is flagged as future work for when code is available.

**Fixed validation prompts (used across all arms, all steps):**

8 prompts, chosen to span diversity (gender, age, race, emotion, pose, hair, accessories):

```
1. A young woman with pale skin and long straight brown hair, smiling, looking directly at the camera
2. An elderly man with dark skin and a bald head, wearing glasses, facing slightly left, neutral expression
3. A middle-aged Asian woman with short black hair and small earrings, serious expression, facing slightly right
4. A young man with tan skin and short dark hair, looking away from the camera, surprised expression
5. A mature man with a gray beard and thinning hair, wearing glasses, looking directly at the camera, neutral expression
6. A young woman with curly red hair and freckles, laughing, facing slightly left
7. A mature woman with tan skin and silver hair, wearing small earrings, neutral expression, facing slightly right
8. A young child with brown skin and short dark hair, looking directly at the camera, neutral expression
9. An elderly woman with blonde hair and pale skin, wearing glasses, facing slightly left, smiling
10. A young woman with long blonde hair and a full beard, looking directly at the camera, neutral expression
```

Seeds: `[42, 142, 242, 342, 442, 542, 642, 742, 842, 942]` (one per prompt, fixed)

**Coverage summary:**

| Dimension | Distribution |
|-----------|-------------|
| Gender | 6F (1,3,6,7,9,10), 4M (2,4,5,8) |
| Age | young adult ×4 (1,4,6,10), child ×1 (8), middle-aged ×1 (3), mature ×2 (5,7), elderly ×2 (2,9) |
| Skin/race | pale ×2 (1,9), dark (2), Asian (3), tan ×2 (4,7), freckled (6), brown (8) |
| Emotion | smiling ×2 (1,9), neutral ×4 (2,5,8,10), serious (3), surprised (4), laughing (6) |
| Pose | frontal ×4 (1,5,8,10), left ×3 (2,6,9), right ×2 (3,7), away (4) |
| Hair | long straight brown (1), bald (2), short black ×2 (3,8), short dark (4), gray/thinning (5), curly red (6), silver (7), blonde ×2 (9,10) |
| Accessories | none ×5 (1,4,6,8,10), glasses ×3 (2,5,9), earrings ×2 (3,7) |
| Unusual | Prompt 10: bearded woman — tests compositional generalization (concepts individually in-distribution, combination out-of-distribution) |

**Design rationale:**
- No two prompts share the same combination — every prompt tests a unique cell
- Neutral overrepresented (3×) — baseline emotion; enough to distinguish "generates neutral well" from "can only generate neutral"
- Young adult overrepresented (3×) — FFHQ skews young; tests demographic majority overfit
- Child included (1×) — FFHQ explicitly contains infants/children; tests age distribution breadth
- Glasses and earrings — known weak spots for diffusion models
- Pose variety — ensures DWPose confidence isn't inflated by only frontal faces
- Skin tone descriptors are direct ("pale", "dark", "tan", "brown") rather than racial categories — matches FFHQ caption style
- Prompt 10 (bearded woman) — tests compositional generalization: the model knows "woman" and "beard" individually but has never seen them combined in FFHQ. If the model generates a bearded woman → strong text adherence. If it generates a clean-shaven woman → collapsed to training prior. CLIP Score on prompt 10 is diagnostic: high = text followed, low = prior won.

**What this set doesn't cover (deliberate):**
- Hats/headwear — FFHQ has few hat images; would mostly test failure
- Extreme emotions (anger, fear) — unlikely in portrait photos, out-of-distribution
- Non-binary gender — FFHQ captions likely don't include this; CLIP Score would be unreliable

**TensorBoard logging plan:**

During training (every 500 steps):
- `validation/generated_{i}` — 10 images (visual progression)
- `validation/pose_confidence_{i}` — 10 scalars, one per prompt (intra-sample trajectory; compare across steps, not across prompts)
- `validation/clip_score_{i}` — 10 scalars, one per prompt (text-image alignment trajectory)
- `validation/aesthetic_score_{i}` — 10 scalars, one per prompt (general image quality trajectory; counters viewer bias toward attractive faces)

During training (every 2 steps, existing — monitoring only):
- `train/loss`, `train/grad_norm`, `train/learning_rate`, `train/ema_decay`
- `monitor/velocity_norm`
- `sys/iter_per_sec`, `memory/vram_*`

Post-training (once):
- `validation/nn_retrieval_mean_distance` — scalar
- `validation/nn_retrieval_min_distance` — scalar

### Comparison Metrics (all arms)

**During training (monitoring, not gating):**

| Metric | Source | Frequency |
|--------|--------|-----------|
| Training loss / velocity norm / grad norm | `training_log.jsonl` | Every 2 steps |
| Throughput (iter/sec) | `training_log.jsonl` | Every 2 steps |
| VRAM usage | `training_log.jsonl` | Every 2 steps |
| Generated faces (10, fixed prompts+seeds) | TensorBoard images | Every 500 steps |
| DWPose confidence (per-prompt, intra-sample trajectory) | TensorBoard scalars | Every 500 steps |
| CLIP Score (per-prompt, intra-sample trajectory) | TensorBoard scalars | Every 500 steps |
| LAION Aesthetic Score (per-prompt, intra-sample trajectory) | TensorBoard scalars | Every 500 steps |

**Post-training (gating):**

| Metric | Source | Method |
|--------|--------|--------|
| NN retrieval | Custom | 100 generated faces, nearest neighbor in training set (DINO cosine) — memorization check |
| F-Eval | F-Bench (future) | 500 generated faces, when code released |

FID dropped — reference set problem (70k Stratum is FFHQ-derived, nothing held out from same distribution) and 10K image generation cost (~28h per arm) is disproportionate for a 5k-step ablation.

### Decision Gates (pre-registered)

Every comparison uses the **step 5000 checkpoint** (`checkpoint_step005000.pt`). The fixed evaluation set is 10 prompts × 10 seeds (same across all arms and all steps). Compute NN retrieval on 100 generated faces post-training.

**Primary gate metric: LAION Aesthetic Score at step 5000 (mean across 10 prompts)**

The aesthetic score is the best available objective quality metric that:
- Requires no reference set (unlike FID)
- Works on single images (unlike FID which needs 10K+)
- Evaluates the whole image including artifacts (counters viewer bias toward attractive faces)
- Is essentially free (MLP head on CLIP embedding we already compute)

**Q1 Gate (Arm B uniform Nyström vs Arm A baseline):**

| Aesthetic score Δ vs Baseline (step 5000) | Throughput Δ | Verdict |
|-------------------------------------------|-------------|---------|
| ≥ 0 (equal or better) | ≥ +10% iter/sec | ✅ **Strong win** — adopt, proceed to Arm C |
| ≥ −5% (slightly worse) | ≥ +10% iter/sec | ⚠️ **Acceptable** — proceed to Arm C |
| ≥ −5% | < +10% iter/sec | ❌ **No benefit** — complexity without speedup |
| < −5% (significantly worse) | any | ❌ **Quality degraded** — reject, do NOT proceed to Arm C |

*Note: Higher aesthetic score = better. "−5%" means the score dropped 5% relative to baseline.*

**Q2 Gate (Arm C face-biased vs Arm B uniform):**

| Aesthetic score Δ vs Arm B (step 5000) | Verdict |
|----------------------------------------|---------|
| > 0 | ✅ **Face-biasing helps** |
| ≈ 0 (±2%) | ❌ **No benefit** — drop face-biasing |
| < −2% | ❌ **Regressive** |

**Trajectory check (intra-sample, diagnostic — all 10 prompts):**
- DWPose confidence: should grow over training for each prompt. Declining across 3 consecutive validation steps → flag
- CLIP Score: should grow over training for each prompt. Declining across 3 consecutive validation steps → flag
- Aesthetic score: should grow over training for each prompt. Declining across 3 consecutive validation steps → flag
- These are per-prompt temporal checks — absolute values are not comparable across prompts

**Collapse detector (both arms, automatic reject during training):**
- Velocity norm > 10.0 at any point after warmup (step 1000) → ❌ immediate reject
- NaN in training loss at any step → ❌ immediate reject

**Memorization guard (diagnostic, not gating):**
- NN retrieval mean distance should not be significantly smaller than baseline
- If Nyström NN distance < baseline by >10%, flag — approximation may be collapsing diversity

**Why aesthetic score as primary gate:** F-Eval code is not yet released. FID requires a reference set we don't have (70k Stratum is FFHQ-derived, nothing held out) and 10K+ images (~28h generation per arm). The LAION Aesthetic Predictor is free (MLP on existing CLIP embedding), works per-image, and evaluates the whole image including artifacts — which counters the specific human bias of overlooking quality issues in attractive faces. When F-Eval code drops, it should replace aesthetic score as primary gate.

**Future experiment this gates:** If Q1 passes with ≤5% aesthetic score degradation / +10% speed, the next experiment is "28L Nyström vs 18L full-attention" — trading approximation error for additional model capacity.

### GPU Budget
- Arm A (baseline): ~4-5 hours on RTX 4090
- Arm B (uniform): ~4-5 hours
- Arm C (face-biased): ~4-5 hours
- Total: ~12-15 GPU hours
- **Must coordinate with active 40k run** — run sequentially, not concurrently

## Implementation Plan

### Files to Modify

| File | Change | Purpose |
|------|--------|---------|
| `production/model.py` | Add `nystrom_attention()` function | Core Nyströmformer implementation |
| `production/model.py` | Modify `Attention.__init__` | Accept `nystrom_landmarks` param |
| `production/model.py` | Modify `Attention.forward()` | Gate: Nyström vs SDPA based on config |
| `production/model.py` | Modify `NanoDiT.__init__` | Accept `nystrom_config`, pass to blocks |
| `production/model.py` | Modify `NanoDiT.forward()` | Accept `nystrom_landmark_indices` |
| `production/config_loader.py` | Add `NystromSelfAttnConfig` dataclass | Config parsing |
| `production/train.py` | Add `compute_face_biased_landmarks()` | Per-sample landmark selection |
| `production/train.py` | Modify `flow_matching_loss()` | Compute landmarks, pass to model |
| `production/train.py` | Modify training step | Extract seg_map, compute landmarks |

### New Config Schema

```yaml
model:
  nystrom_self_attn:
    enabled: false               # bool — master switch
    num_landmarks: 64            # int — m, number of landmark tokens
    face_biased: true            # bool — use seg_map for stratified sampling
    face_ratio: 0.6              # float — fraction of landmarks allocated to face
    layers: "last_6"             # str — "all", "last_N", or comma-separated indices
    svd_epsilon: 1.0e-6         # float — singular value threshold
```

### Training Loop Changes

In `flow_matching_loss()` (train.py), before model forward:

```python
if nystrom_config and nystrom_config.face_biased and seg_map is not None:
    nystrom_landmark_indices = compute_face_biased_landmarks(
        seg_map,                    # (B, 64, 64) int16
        patch_resolution=(h_p, w_p),  # actual patch grid from x0 shape
        m=nystrom_config.num_landmarks,
        face_ratio=nystrom_config.face_ratio,
    )  # → (B, m) int64
else:
    nystrom_landmark_indices = None
```

### Attention Module Gate

In `Attention.forward()`:

```python
if self.is_cross_attn or self.nystrom_landmarks <= 0:
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
else:
    x = nystrom_attention(q, k, v, self.nystrom_landmarks,
                          landmark_indices=nystrom_landmark_indices,
                          svd_epsilon=self.svd_epsilon)
```

### Fallback Paths

- `seg_map is None` (data path without segmentation) → uniform landmark selection
- `nystrom_landmark_indices is None` → uniform landmark selection
- Inference mode (no seg_map) → uniform landmark selection
- `nystrom_config.enabled is False` → standard SDPA (no overhead)

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| SVD instability in bfloat16 | Use float32 for SVD (m=64 → negligible cost); ε=1e-6 threshold |
| Approximation error compounding over 6 layers | Compare validation loss at each 500-step checkpoint |
| Interaction with TREAD routing | TREAD routes layers 1-16; Nyström applies to 12-17 → overlap on layers 12-16. Landmark indices must reference the full (reassembled) sequence, not the visible-only subset. This is safe because at layer 12 the sequence is reassembled (model.py line 698-701). |
| Interaction with MaskDiT | MaskDiT subsamples tokens before self-attention. Landmark indices must reference the visible token subset. Need separate handling — defer to future work. For initial ablation, MaskDiT disabled (it's off in production config). |
| REPA alignment quality | Landmarks reduce attention fidelity → may reduce REPA alignment. Track `repa_loss` in training log. |
| Non-square aspect ratios | Seg grid is always 64×64 but patch grid varies (e.g., 76×52 for 1216×832). Need to upsample seg_map to match actual patch grid. |

## Open Questions (deferred)

1. **DWPose integration for validation:** The training pipeline already loads DWPose for conditioning. Need to verify we can run it in inference mode on generated images without loading a second model instance (VRAM constraints on 4090). May need to run DWPose on CPU or after unloading the DiT.

2. **LAION Aesthetic Predictor integration:** Need to load CLIP ViT-L/14 (for CLIP Score) and the aesthetic MLP head (`sac+logos+ava1-l14-linearMSE.pth`). Verify the CLIP model can share GPU memory with the DiT during validation, or run CLIP on CPU after generation.

3. **Validation prompt set:** The 8 fixed prompts above are a first draft. Should be reviewed for diversity coverage and whether they adequately stress-test the model (glasses, facial hair, ethnicities, poses).

4. **Layer sweep:** "last_6" is a hypothesis. If Q1 passes, sweep layer selection: "all", "last_3", "first_6", "middle_6".

5. **Landmark count sweep:** m=64 vs m=128 vs m=32 — tradeoff between approximation quality and compute reduction.

6. **Iterative refinement vs SVD pseudoinverse:** The original Nyströmformer paper uses Woodbury identity for the inverse (iterative, avoids SVD). If SVD proves numerically unstable in practice, switch to the iterative approach.

7. **Inference speed measurement:** Compare sampling throughput (sec/image at 35 steps) between baseline and Nyström checkpoints to quantify the inference benefit.

8. **F-Eval adoption:** When F-Bench code is released, replace FID as primary gate with F-Eval. F-Eval is reference-free and face-specific — strictly better for our use case.

## Experiment Directory Structure

All arms follow `experiments/STRUCTURE.md` — descriptive slugs, frozen configs, mandatory provenance, post-hoc validation directory.

**Important:** The active 40k run (`experiments/2026-06-04_2116/`) already has `validation/step0005000/` containing old proving-ground metrics (dino_swap, reconstruction, text_manip, text_only). That is a **different experiment** with a 40k-target LR schedule. Our 5k ablation arms are separate experiments in separate directories — evaluation results must not be mixed. Each arm's F-Eval/FID/CLIP/NN-retrieval results live exclusively under that arm's own `experiments/{slug}/validation/step005000/`.

### Arm A: `experiments/70k-baseline-5k/`

```
experiments/70k-baseline-5k/
  README.md           # Reference baseline, full SDPA, 5k steps on 70k Stratum
  config.yaml         # Forked from 2026-06-04_2116/config.yaml, total_steps=5000
  provenance.yaml     # arm: a, differs_from: 2026-06-04_2116, git_commit, training_host, etc.
  runs/
    {YYYY-MM-DD_HHMM}/
      metadata.json
      training_log.jsonl
      tensorboard/
      checkpoints/
      validation/
  validation/         # Post-hoc F-Eval, FID, CLIP Score, NN retrieval
    step005000/
      f_eval_results.json
      fid_results.json
      clip_score_results.json
      nn_retrieval_results.json
      generated_faces/   # 500 faces from fixed prompt set
  figures/
  notes/
```

### Arm B: `experiments/nystrom-uniform-5k/`

Same structure. `provenance.yaml` records `differs_from: 70k-baseline-5k` and the Nyström config delta.

### Arm C: `experiments/nystrom-face-biased-5k/`

Same structure. `provenance.yaml` records `differs_from: nystrom-uniform-5k` and the face_biased flag.

### Provenance Template (per arm)

```yaml
arm: b                          # a, b, c
hypothesis: >
  Nyströmformer self-attention approximation with uniform landmarks
  matches full-attention quality at 5k steps while improving throughput.
differs_from: 70k-baseline-5k   # arm A for B, arm B for C
diff_summary: |
  - nystrom_self_attn.enabled: true
  - nystrom_self_attn.num_landmarks: 64
  - nystrom_self_attn.face_biased: false
  - nystrom_self_attn.layers: "last_6"
  - All other settings identical to comparator
git_commit: <SHA>
git_dirty: false
training_host: game (RTX 4090)
training_gpu: NVIDIA RTX 4090 (24GB VRAM)
training_steps: 5000
data_source: $STRATUM_DIR (Stratum NAS, 70k FFHQ portraits)
evaluation: F-Bench F-Eval + FID + CLIP Score + NN retrieval
```

### Pre-Flight Checklist (per arm, before training starts)

1. `mkdir -p experiments/{slug}/{runs,validation,figures,notes}`
2. Write `config.yaml` (frozen — never edited after run starts)
3. Write `provenance.yaml` (mandatory before first run)
4. Write `README.md` (hypothesis, expected outcome, how it differs from comparator)
5. Verify `git_dirty: false` — commit or stash all code changes
6. Verify data availability: `ls $STRATUM_DIR` or check shard dirs
7. Launch: `python -m production.train_production --config experiments/{slug}/config.yaml`

### Post-Training Checklist (per arm, after step 5000)

1. Verify `metadata.json` exists and `git_dirty` is false
2. Run NN retrieval memorization check (100 faces, DINO cosine vs training set)
3. Save all results to `experiments/{slug}/validation/step005000/`
4. Update README.md with results summary
5. Update Experiment Registry table in project README.md

## Branching Strategy

Per experiment governance (AGENTS.md):
- Current branch: whatever is active when the 40k run completes
- Create `exp/nystrom-self-attn` branch from a clean commit
- Each arm gets its own experiment directory under `experiments/`
- `git_dirty: false` is mandatory in metadata.json

## Reference

- Nyströmformer paper: Xiong et al., "Nyströmformer: A Nyström-based Algorithm for Approximating Self-Attention", AAAI 2021
- Prior analysis (2026-06-13): Self-attn Gram matrix well-defined; cross-attn not applicable
- Active production config: `experiments/2026-06-04_2116/config.yaml` (Arm J, 70k Stratum)
