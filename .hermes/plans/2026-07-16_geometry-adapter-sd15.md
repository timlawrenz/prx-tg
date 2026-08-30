# Geometry-Only Adapter for SD 1.5 — Implementation Plan

> **Goal:** Prove that prx-tg's per-dim geometry token basis can control face pose in a frozen open-source T2I model by injecting z_g tokens into cross-attention alongside CLIP text embeddings. No identity conditioning — geometry-only. SD 1.5 is the target (open weights, cross-attention architecture, better faces than SD2).
>
> **Environment verified (2026-07-17):** SD 1.5 loads ✓ (UNet 860M, cross_attn_dim=768, text encoder hidden=768, VAE latent channels=4). SD 2.1 is gated on HF and unavailable locally. SD 1.5 is the confirmed target. Diffusers upgraded to 0.39.0 to fix transformers 5.9.0 compatibility. NAS has 170+ SD1.5-based finetunes but no base checkpoint file — pipeline loads from HF cache.

**Status:** Plan phase. Not yet implemented.
**Repo:** `/home/tim/source/activity/prx-tg`
**Branch:** `main` (new experiment, no code changes to production/)
**Slug:** `geometry-adapter-sd15`
**Experiment dir:** `experiments/geometry-adapter-sd15/`

---

## Hypothesis

A lightweight adapter (~630K params) that encodes z_g (50-dim disentangled geometry vector) into CLIP-aligned cross-attention tokens can steer a frozen SD 1.5 UNet to control face pose — specifically, varying `z_g[0]` (yaw, R²=0.996) should produce visible head rotation in generated faces, while keeping identity/text unchanged.

**Success criterion:** Generate the same seed + same text prompt, vary `z_g[0]` across [-2, -1, 0, 1, 2], and observe monotonic head yaw rotation in the output images.

**Stop condition (from memory):** If training fails to converge or collapse recurs, stop the from-scratch approach and use the frozen SD1.5 backbone instead. This experiment IS that pivot.

---

## Architecture

### Adapter (`geometry_adapter.py` — new file)

```
GeometryAdapter(nn.Module)  ~630K params
══════════════════════════════════════════

Inputs:
  z_g              (B, 50)   float32  — geometry vector
  cfg_drop_geo     (B,)      bool     — CFG mask

Trainable parameters:
  geometry_proj: Sequential(
    Linear(1, 768, bias=True),    # 1,536
    GELU(),
    Linear(768, 768, bias=True),  # 590,592
  )                                # total: ~592K

  geo_basis: Parameter(1, 50, 768) # 38,400
    (per-dim learned token identity, σ=0.02 init)

  null_geometry: Parameter(1, 50)  # 50 (learned null vector)

Forward:
  1. CFG guard: if all(drop_geo) → z_g = null_geometry
  2. z_g_expanded = z_g.unsqueeze(-1)                     # (B, 50, 1)
  3. geo_cond = geometry_proj(z_g_expanded)                # (B, 50, 768)
  4. if not all(drop_geo):
       geo_cond = geo_cond + geo_basis                     # (B, 50, 768)
  5. return geo_cond                                       # (B, 50, 768)
```

### Integration into SD 1.5 UNet

SD 1.5 UNet has cross-attention at 5 resolution levels:
- Down block 2 (48×48), Down block 3 (24×24)
- Mid block (12×12)
- Up block 2 (24×24), Up block 3 (48×48)

Each has `SpatialTransformer` with `cross_attn(Q=spatial_features, K,V=encoder_hidden_states)`.

**Injection:** Replace `encoder_hidden_states` (CLIP text embeddings, (B, 77, 768)) with concatenated `[text_emb, geo_tokens]` → `(B, 77+50, 768)`. Same tokens go to all 5 cross-attention layers.

**No architectural changes to the UNet.** The adapter produces tokens the UNet's existing cross-attention can consume. The only modification is in the forward call: concatenate before passing to the UNet.

**No per-block projections, no zero-convs.** Matches prx-tg's design — a single shared adapter encoder produces tokens consumed identically by all blocks.

---

## Data Landscape

### Data Summary (NAS Survey 2026-07-17)

| Dataset | Images | z_g | Captions | DWPose | Format |
|---------|--------|-----|----------|--------|--------|
| **FFHQ stratum** | 70,000 | ✅ (50,) | ✅ VLM captions | ✅ (133,3) | Flat dirs, `pixel.npy` |
| **Hegre corpus** | 31,668 | ✅ (50,) | ❌ persona-only | ❌ | Flat dirs, `pixel.npy` |
| **hegre-faces/v1** | 384,012 | ✅ 161K | ✅ 113K+ VLM | ✅ 113K+ | Persona/shoot/face JPGs |

**Training target: 183K+ images** with aligned (image, caption, z_g) triples across FFHQ (70K) and hegre-faces (113K+ with captions).

### Primary: FFHQ Stratum (70,000 images)

Located at `/mnt/nas-ai-models/training-data/ffhq/stratum/`.

Standard stratum format — flat per-sample directories with `pixel.npy`, `z_g.npy`, `caption.txt`, `pose.npy`. Loads directly via existing `StratumDataset` with a new adapter mode.

### Primary: hegre-faces/v1 (113K+ with captions + z_g)

Located at `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/`.

**Tree structure** (separate parallel trees, matched by persona/shoot/face name):

```
hegre-faces/v1/
├── faces/{persona}/{shoot}/{face}.jpg        ← 384,012 JPG face crops
├── stratum/{persona}/{shoot}/{face}/          ← caption.txt, pose.npy, DINO, T5
│   └── caption.txt                            ← VLM-generated, rich physical descriptions
├── zg/faces/{persona}/{shoot}/{face}.npy      ← 160,805 z_g vectors (50,)
├── auraface/faces/{persona}/{shoot}/{face}.npy
└── lda/faces/{persona}/{shoot}/{face}.npy
```

**Sample caption:** *"A Caucasian female with a slender build is positioned in profile facing right against a solid medium gray background. Her hair is long, dark brown, and layered... Eyes are blue-grey..."* — These are detailed, pose-aware VLM descriptions.

**Key: 113,446 entries** in `stratum_approved_list.txt` have full stratum processing (caption + DINO + T5 + pose). The z_g set is larger (160K) because extraction is cheaper.

**No `pixel.npy`** — images are JPGs. For SD1.5: load JPG → resize 512×512 → VAE encode (or pre-encode to latents once, ~3.6GB for 113K images).

### Data Loading: Reusing prx-tg Infrastructure

prx-tg's data loading has three reusable patterns:

**1. `StratumDataset` adapter dispatch** (`production/data_stratum.py`):
```python
class StratumDataset(IterableDataset):
    def __init__(self, stratum_dir, adapter_name="stratum", ...):
        # adapter_name controls which _load_* method is called
        if adapter_name == "eidolon":
            self._load = self._load_sample_eidolon   # identity + geometry + stub fields
        else:
            self._load = self._load_sample            # DINO + T5 + pose + seg + caption
```

**Plan: Add `adapter_name="sd15_geometry"`** → new `_load_sample_sd15()` method that loads only `pixel.npy`, `z_g.npy`, `caption.txt`. This is a ~15-line addition. Reuses all the caching, shuffling, batching, and worker infrastructure.

**2. `MultiSourceStratumDataset` weighted interleaving** (`production/data_stratum.py`):
```python
sources = [
    (StratumDataset(dir=FFHQ,  adapter_name="sd15_geometry", ...), weight=2.3),
    (StratumDataset(dir=HEGRE, adapter_name="sd15_geometry", ...), weight=1.0),
]
loader = MultiSourceStratumDataset(sources)
```

Each iteration picks a source at random (weighted) and yields its next batch. All sources must share `batch_size`, `target_latent_size`, `adapter_name`. This is the same pattern Arm O uses for FFHQ+hegre interleaving.

**3. YAML config integration** — the `stratum_dirs` block in `config.yaml` maps directly to `get_multi_stratum_dataloader()`:
```yaml
data:
  source: "stratum"
  stratum_dirs:
    - dir: "/mnt/nas-ai-models/training-data/ffhq/stratum"
      weight: 2.3
    - dir: "/mnt/nas-ai-models/training-data/eidolon/hegre_corpus"
      weight: 1.0
```

### Recommended Two-Phase Data Strategy

**Phase 1 (immediate): FFHQ only via `adapter_name="sd15_geometry"` on existing `StratumDataset`.** 70K images, flat dirs, `pixel.npy` + `z_g.npy` + `caption.txt` all in one place. No new dataset class needed. Gets the experiment running in under an hour of coding. Batch dict:
```python
{'image_data': (B,3,512,512), 'z_g': (B,50), 'caption': list[str]}
```

**Phase 2 (follow-up): Add hegre-faces via new `HegreFacesStratumDataset` class** that handles the persona/shoot/face tree. Builds a flat sample index from the three parallel trees (faces/, stratum/, zg/faces/). Plugs into the same `MultiSourceStratumDataset` interleaving pattern. Unlocks 113K+ additional images.

Alternatively: pre-process hegre-faces into standard stratum format (JPG→pixel.npy, copy caption.txt, symlink z_g.npy) so it feeds into the same `StratumDataset` with no new code.

---

## Training

### Loop (per step)

```python
# 1. Sample batch
z_0    = batch['latent']                    # (B, 4, 64, 64) — VAE latents
z_g    = batch['z_g']                       # (B, 50)
text   = batch['caption']                   # list[str]

# 2. CLIP text encoding (offline or cached)
text_emb = clip_encode(text)                # (B, 77, 768)
null_text_emb = clip_encode("")             # (1, 77, 768)

# 3. Noise
ε = torch.randn_like(z_0)
t = logit_normal_sample(B)                  # (B,) — timestep
z_t = t*z_0 + (1-t)*ε                       # rectified flow (or use SD's native noise schedule)

# Actually, SD1.5 uses standard DDPM noise schedule. Use that.
# noise = randn_like(z_0)
# timestep = randint(0, 1000)  
# z_t = add_noise(z_0, noise, timestep)

# 4. Geometry adapter
geo_tokens = adapter(z_g)                   # (B, 50, 768)

# 5. CFG dropout (during training)
rand = torch.rand(B)
uncond_idx = rand < 0.10                    # 10%: drop both text and geo
geo_drop_idx = (rand >= 0.10) & (rand < 0.30)  # 20%: drop geo only

# Apply drops
text_emb[uncond_idx] = null_text_emb
geo_tokens[uncond_idx | geo_drop_idx] = adapter(null_geometry)

# 6. Concatenate and forward
encoder_hidden_states = torch.cat([text_emb, geo_tokens], dim=1)  # (B, 127, 768)
ε_pred = frozen_unet(z_t, timestep, encoder_hidden_states=encoder_hidden_states).sample

# 7. Loss
loss = F.mse_loss(ε_pred, ε)
loss.backward()  # gradients only flow through adapter
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Not Muon — adapter weights are (1,50,768), not 2D matrices |
| Learning rate | 1e-4 | Lower than prx-tg (3e-4) — small adapter on frozen giant |
| LR schedule | Cosine to 1e-6 | Standard |
| Batch size | 8 | Fits on 24GB VRAM with frozen UNet |
| Training steps | 10,000 | ~1 epoch over 70K images at batch 8 |
| CFG p_uncond | 0.10 | Drop text+geo |
| CFG p_geo_drop | 0.20 | Drop geo only (text live) |
| Precision | bfloat16 | SD1.5 native |

### Training Cost Estimate

- SD1.5 UNet forward: ~0.3s per batch on RTX 4090 (BF16)
- VAE encode: ~0.05s (or pre-encoded, so 0s)
- Phase 1 (70K FFHQ): 10,000 steps × 0.35s = ~1 hour
- Phase 2 (+113K hegre-faces): 20,000 steps × 0.35s = ~2 hours
- Plus data loading overhead: 2-4 hours total for full dataset

---

## Inference (CFG Sampling)

### 3-Pass Dual CFG

```python
# Pre-compute
text_emb   = clip_encode(prompt)                    # (1, 77, 768)
null_text  = clip_encode("")                        # (1, 77, 768)
geo_tokens = adapter(target_z_g)                    # (1, 50, 768)
null_geo   = adapter(null_geometry)                 # (1, 50, 768)

# Pass 1: Unconditional
ctx_uncond = cat([null_text, null_geo])             # (1, 127, 768)
ε_uncond = unet(z_t, t, encoder_hidden_states=ctx_uncond)

# Pass 2: Text-only
ctx_text = cat([text_emb, null_geo])                # (1, 127, 768)
ε_text = unet(z_t, t, encoder_hidden_states=ctx_text)

# Pass 3: Geometry-only
ctx_geo = cat([null_text, geo_tokens])              # (1, 127, 768)
ε_geo = unet(z_t, t, encoder_hidden_states=ctx_geo)

# Combine
ε = ε_uncond + text_scale*(ε_text - ε_uncond) + geo_scale*(ε_geo - ε_uncond)
```

### Sampler

Use standard DDIM sampler (50 steps) or DPMSolver++ (20 steps). SD1.5's native scheduler.

### Geometry Sweep

For validation, generate a grid varying a single z_g dimension:
```
prompt: "a studio portrait of a person"
seed: 42
z_g[0] (yaw):  [-2.0, -1.0, 0.0, 1.0, 2.0]
z_g[1:] = 0 (neutral for all other dims)
text_scale: 7.5
geo_scale:  sweep [1.0, 2.0, 3.0, 5.0]
```

Expected: as `geo_scale` increases, head yaw should become more pronounced. As `z_g[0]` varies, yaw should change monotonically.

---

## Evaluation

### Primary Metric: Qualitative Geometry Sweep

Generate a 5×4 grid (5 z_g[0] values × 4 geo_scales) and visually verify yaw progression. This is the core success/failure signal.

### Secondary Metrics

| Metric | What it measures | How |
|--------|-----------------|-----|
| **DWPose yaw delta** | Actual head yaw vs z_g[0] value | Run DWPose on generated images, compare yaw with z_g[0] |
| **CLIP image similarity** | Does text prompt still work? | CLIP score between generated image and prompt |
| **FID on FFHQ val** | Image quality preservation | FID between 1K generated images and FFHQ validation |
| **Identity consistency** | Does face identity change with z_g? | Face recognition embedding distance across z_g sweep |

### Success Gate

**PASS if:** DWPose yaw correlates with z_g[0] with R² > 0.5 AND visible monotonic yaw change in the sweep grid AND text following is preserved (CLIP score within 10% of baseline SD1.5).

**FAIL if:** No visible yaw change at any geo_scale, OR text following degrades significantly, OR faces collapse/artifact.

---

## Implementation Tasks

### Task 0: Verify Environment

```bash
cd /home/tim/source/activity/prx-tg
python -c "import diffusers; print(f'diffusers {diffusers.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB')"
```

**Gate:** diffusers >= 0.30, CUDA available, >= 20GB VRAM.

### Task 1: Load and Verify SD 1.5

```python
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
).to('cuda')

# Verify UNet cross-attention dim
unet = pipe.unet
assert unet.config.cross_attention_dim == 768  # matches our adapter

# Verify text encoder hidden dim
text_enc = pipe.text_encoder
assert text_enc.config.hidden_size == 768

# Quick inference test
pipe("a portrait photo", num_inference_steps=5).images[0].save('/tmp/sd15_test.png')
```

**Gate:** SD1.5 loads, runs, produces recognizable output.

### Task 2: Create Experiment Directory

```bash
mkdir -p experiments/geometry-adapter-sd15/
mkdir -p experiments/geometry-adapter-sd15/runs/
```

Create `experiments/geometry-adapter-sd15/config.yaml` with all hyperparameters.
Create `experiments/geometry-adapter-sd15/README.md` with hypothesis and expected outcome.
Create `experiments/geometry-adapter-sd15/provenance.yaml`.

### Task 3: Build GeometryAdapter (`geometry_adapter.py`)

Implement `GeometryAdapter` class as specified in the Architecture section above. Standalone file, no prx-tg dependencies. Tests:
```python
adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
z_g = torch.randn(2, 50)
tokens = adapter(z_g)  # (2, 50, 768)
assert tokens.shape == (2, 50, 768)

# CFG drop test
tokens_dropped = adapter(z_g, cfg_drop_geo=torch.tensor([True, True]))
assert torch.allclose(tokens_dropped, adapter(torch.zeros(2, 50)))
```

### Task 4: Build Data Pipeline

**Phase 1 (immediate):** Add `adapter_name="sd15_geometry"` to `StratumDataset`.

In `production/data_stratum.py`, add a `_load_sample_sd15()` method and dispatch it:

```python
def _load_sample_sd15(self, d: Path) -> dict:
    """Minimal loader: pixel + z_g + caption for SD1.5 geometry adapter."""
    pixel   = np.load(d / 'pixel.npy')          # (3, H, W) f16
    z_g     = np.load(d / 'z_g.npy')            # (50,) f32
    caption = (d / 'caption.txt').read_text().strip()
    
    image_data = _resize_image(pixel, 512)       # → 512×512 for SD1.5
    
    return {
        'image_data': image_data,                # (3, 512, 512) f32
        'z_g': torch.from_numpy(z_g).float(),    # (50,)
        'caption': caption,
    }
```

Then in `__init__`, add the dispatch branch:
```python
if self.adapter_name == "sd15_geometry":
    self._load = self._load_sample_sd15
elif self.adapter_name == "eidolon":
    ...
```

And update `_collate()` to handle the `z_g` field (already present from eidolon path):
```python
if 'z_g' in batch[0]:
    result['z_g'] = torch.stack([s['z_g'] for s in batch])
```

**Phase 2 (follow-up):** Create `HegreFacesStratumDataset` for the persona/shoot/face tree structure. Or pre-process hegre-faces JPGs → `pixel.npy` + symlink `z_g.npy` so it feeds into the same `StratumDataset` with `adapter_name="sd15_geometry"`.

**Pre-encoding (optional):** Pre-encode all images to SD1.5 VAE latents and save as `.npy`. For 70K images at 4×64×64×float16 = 32KB each, total ≈ 2.2GB. Worth doing for iteration speed. Script: `scripts/preencode_latents.py`.

### Task 5: Build Training Loop

File: `train_geometry_adapter.py`
- Loads frozen UNet (BF16), CLIP text encoder, VAE
- Builds `GeometryAdapter` (FP32 for training)
- CFG dropout logic
- AdamW optimizer
- Standard epsilon-prediction loss
- Checkpointing every 1000 steps
- TensorBoard logging

### Task 6: Build Evaluation Pipeline

File: `eval_geometry_adapter.py`
- Geometry sweep grid generator
- DWPose yaw extraction
- CLIP score computation
- FID computation
- Output collage generation

### Task 7: Train (10,000 steps)

```bash
python train_geometry_adapter.py \
    --config experiments/geometry-adapter-sd15/config.yaml \
    --output-dir experiments/geometry-adapter-sd15/runs/$(date +%Y-%m-%d_%H%M)/
```

Estimated runtime: ~2-3 hours on RTX 4090.

### Task 8: Evaluate

```bash
python eval_geometry_adapter.py \
    --checkpoint experiments/geometry-adapter-sd15/runs/<ts>/adapter_step10000.pt \
    --output-dir experiments/geometry-adapter-sd15/validation/
```

### Task 9: Document Results

Update `experiments/geometry-adapter-sd15/README.md` with:
- Sweep collage
- DWPose yaw vs z_g[0] scatter plot
- CLIP score preservation
- Pass/fail verdict with evidence

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **Frozen UNet ignores geometry tokens** | Medium | If loss doesn't improve: try (a) initialize adapter output near CLIP embedding mean, (b) LoRA-unfreeze cross-attn K,V projections |
| **Geometry tokens overwhelm text** | Low | 50 tokens vs 77 text tokens. If CLIP score degrades, reduce `geo_scale` or compress z_g via PCA to 16 dims |
| **z_g distribution mismatch** | Low | FFHQ has diverse poses. If artifacts at extreme z_g values, add noise to z_g during training (z_g + N(0,0.05)) |
| **Adapter doesn't learn (loss flatlines)** | Low-Medium | Check: is CFG drop actually working? Are geometry tokens carrying signal? Try: train without CFG initially, add CFG after loss drops |
| **SD1.5 VAE latent space is incompatible** | Very Low | Standard 4×64×64 latent space. Our noise/loss are standard epsilon prediction. |
| **Text encoder null embedding** | Low | Use `clip_encode("")` which produces mean-pooled null embedding, matching SD1.5 training |
| **hegre-faces tree structure adds complexity** | Low-Medium | Phase 1 uses flat FFHQ only — no tree complexity. Phase 2: either write `HegreFacesStratumDataset` (~100 lines) or pre-process JPGs into `pixel.npy` format (one-time script) |

---

## Design Decisions (Documented)

1. **Geometry-only, not identity+geometry.** Identity is the base model's problem. The adapter only steers pose. This eliminates the identity collapse risk from prx-tg's eidolon experiments.

2. **SD 1.5 not SD 2.** SD2 is gated on HF. SD1.5 is open, has better face quality, and identical cross-attention architecture. Same hidden dim (768).

3. **3-pass CFG, not 2-pass.** Independent control over text strength (prompt following) and geometry strength (pose adherence). Users can dial them separately.

4. **Shared adapter, no per-block projections.** Matches prx-tg's proven design. The model's existing cross-attention learns to use the same tokens differently at different resolutions.

5. **FFHQ Phase 1, hegre-faces Phase 2.** FFHQ (70K images) via a new `adapter_name="sd15_geometry"` on existing `StratumDataset` gets the experiment running immediately. hegre-faces (113K+ with captions + z_g) added as Phase 2 via a new `HegreFacesStratumDataset` class or pre-processing into stratum format. Both feed into the same `MultiSourceStratumDataset` weighted interleaving pattern already used by Arm O. Total: 183K+ images available.

6. **No zero-init output gate.** Unlike ControlNet's spatial injection, cross-attention tokens that are zero vectors naturally produce zero output — the model ignores them until they carry signal. The adapter doesn't need a zero-init gate because it's not injecting into a residual pathway.

---

## Relationship to prx-tg Collapse Problem

This experiment serves a dual purpose:

1. **Primary:** Prove the geometry adapter concept works on a frozen backbone.
2. **Diagnostic:** If the same adapter architecture (per-dim basis, shared MLP, no gate) works on frozen SD1.5 but collapses in from-scratch prx-tg, it isolates the prx-tg collapse to the from-scratch training dynamics, not the adapter design.

If SD1.5 adapter works → geometry token mechanism is valid → prx-tg's collapse is a training stability issue, not an architecture bug.
If SD1.5 adapter also fails → the geometry token approach may need a fundamentally different injection mechanism (e.g., per-block gated cross-attention).
