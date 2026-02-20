# Implementation Plan: DINOv3 Spatial Patches Integration

## ⚠️ CRITICAL BUG DISCOVERED

**Current preprocessing BREAKS spatial alignment!**

Line 245 in `scripts/generate_approved_image_dataset.py`:
```python
transforms.CenterCrop(224)  # ❌ WRONG: Only processes center of image
```

**Problem**: For 1216×832 image, center-crop gives 224×224 from center → DINO patches represent only the center, but VAE latents represent the full image → spatial misalignment!

**Solution**: Dynamic resolution (preserve aspect ratio, perfect alignment)
- Feed images at multiples of 14 (e.g., 1218×826 → 87×59 patch grid)
- Each bucket gets its own patch grid size
- **Verified**: DINOv3 supports this natively via `processor(size={"height": H, "width": W}, do_center_crop=False)`

## Problem Statement

Current architecture uses only DINO CLS token (1024-dim global vector) for visual conditioning, resulting in "blobs in wrong places" - correct colors/style but poor spatial layout. DINOv3 extracts spatial patch embeddings that are currently computed but **spatially misaligned due to center-crop bug**.

**Goal**: Fix preprocessing AND integrate DINO patches to provide spatially-aligned conditioning.

## Architecture Decisions (Confirmed)

### Method: Concatenated Cross-Attention with CLS Fallback + Dynamic Resolution

- **Single cross-attention module** per DiT block (simpler than parallel)
- **Variable-length sequences**: T5 text [512] + DINO_CLS [1] + DINO_patches [**variable**]
- **Bucket-specific patch counts**:
  - 1024×1024 → 1022×1022 DINO → 73×73 patches = 5329 + 1 CLS = **5842 total tokens**
  - 1216×832 → 1218×826 DINO → 87×59 patches = 5133 + 1 CLS = **5646 total tokens**
  - etc. (each bucket has different patch count)
- All projected to same hidden_size before concatenation

**Rationale**: 
- **Simpler than parallel**: Fewer parameters, single attention mechanism
- **Perfect spatial alignment**: No center-crop, preserve aspect ratio
- **CLS as fallback**: Attention heads can route to global CLS when local patches aren't needed
- **Dual CLS usage**: 
  1. adaLN modulation (controls entire block via scale/shift)
  2. Global token in cross-attention (fallback for spatial routing)
- **Efficient routing**: Model learns to attend to text semantics, global style, or specific spatial patches as needed

### Open Questions - RESOLVED ✓

**1. Patches mask: Should we mask invalid patches?**
- ✅ **No**. All patches + CLS are always valid (no padding in spatial dimension)
- Mask implementation: `torch.cat([t5_mask, torch.ones(B, num_dino_tokens)], dim=1)`
- T5 portion handles padding, DINO portion always unmasked
- **Note**: `num_dino_tokens` varies per bucket (CLS + variable patches)

**2. CFG scaling: Do patches need a separate CFG scale?**
- ✅ **No**. Standard equation works: `v_pred = v_uncond + s_text * Δv_text + s_dino * Δv_dino`
- CLS and patches dropped together during training → unified "visual semantic" signal at inference

**3. CLS positioning: [T5, CLS, patches] or [CLS, T5, patches]?**
- ✅ **Either works** (transformers are permutation invariant)
- Choose: **[T5, CLS, patches]** (clean and logical)

**4. Positional encoding for patches: Add 2D pos encoding?**
- ✅ **MVP: No**. DINOv3 embeddings already contain absolute positional info from ViT's first layer
- **Dynamic resolution**: Patches inherently encode their 2D position in the original image
- **Phase 2 (if needed)**: Add tiny learnable 2D sinusoidal embedding if model struggles with variable-grid→64×64 mapping after 20k steps

**5. CRITICAL: Spatial alignment bug**
- ✅ **FIXED via dynamic resolution**
- Current preprocessing uses `CenterCrop(224)` → misalignment
- Solution: Round each bucket to nearest multiple of 14, feed full aspect-corrected image to DINO
- Example: 1216×832 bucket → 1218×826 DINO input → 87×59 patch grid → perfect alignment with VAE latents

### Component Specifications

1. **T5 Text Embeddings**: 
   - Shape: [512, 4096] (unchanged)
   - Projected to [512, hidden_size] via existing `text_proj`

2. **DINO CLS Token**:
   - Shape: [1, 1024]
   - Projected to [1, hidden_size] via existing `dino_proj`
   - **Dual usage**:
     - adaLN modulation: `dino_proj(CLS) + timestep_emb` (existing)
     - Cross-attention: Prepended to patch sequence as global fallback token

3. **DINO Patches (Variable-Length)**:
   - Shape: **[num_patches, 1024]** where `num_patches` depends on bucket
   - **Bucket-specific dimensions** (rounded to nearest multiple of 14):
     ```
     1024×1024 → 1022×1022 DINO → 73×73 = 5329 patches
     1216×832  → 1218×826  DINO → 87×59 = 5133 patches
     832×1216  → 826×1218  DINO → 59×87 = 5133 patches
     1280×768  → 1274×770  DINO → 91×55 = 5005 patches
     768×1280  → 770×1274  DINO → 55×91 = 5005 patches
     1344×704  → 1344×700  DINO → 96×50 = 4800 patches
     704×1344  → 700×1344  DINO → 50×96 = 4800 patches
     ```
   - Projected to [num_patches, hidden_size] via new `dino_patch_proj`
   - **Spatial alignment preserved**: No center-crop, patches map directly to image regions
   
4. **Combined Cross-Attention Sequence (Variable-Length)**:
   - Concatenate: `[text_proj(T5), dino_proj(CLS), dino_patch_proj(patches)]`
   - Final shape: **`[512 + 1 + num_patches, hidden_size]`** (varies per bucket)
   - Example: 1024×1024 bucket → `[512 + 1 + 5329, 768] = [5842, 768]`
   - Model learns to route attention between text concepts, global style, and spatially-aligned features

5. **CFG Dropout Strategy**:
   - Keep current config: 10% uncond, 30% text-only, 5% DINO-only, 55% both
   - When dropping DINO: drop CLS (from both adaLN and cross-attn) AND patches together
   - When dropping text: drop T5 tokens, keep CLS + patches
   - Use learned null embeddings for dropped signals
   - **Key insight**: Dropping DINO forces model to generate spatial layout from text alone

5. **Training Start**:
   - Attempt warm-start from current checkpoint
   - New parameters (cross-attn-2, dino_patch_proj, null_dino_patches) initialized randomly
   - May need to reduce learning rate initially for stability

## Implementation Plan

### Phase 1: Data Pipeline (4-6 hours)

**Goal**: Fix spatial alignment bug AND load variable-length DINO patches during training.

#### 1.0 ⚠️ FIX CRITICAL BUG: DINO Preprocessing
**File**: `scripts/generate_approved_image_dataset.py`

**CRITICAL**: Current preprocessing breaks spatial alignment! Must fix before regenerating patches.

- [ ] **Remove/replace `preprocess_vit_like` function** (lines 238-250):
  ```python
  # OLD (BROKEN):
  transforms.Resize(size + 32)
  transforms.CenterCrop(size)  # ❌ LOSES SPATIAL ALIGNMENT
  
  # NEW (FIXED):
  # Use processor with dynamic resolution, no center-crop
  ```

- [ ] **Update `compute_dinov3_patches()` and `compute_dinov3_both()`** (lines ~308-385):
  ```python
  def compute_dinov3_patches(dino, device, image, target_width, target_height):
      """Extract DINOv3 patches at bucket-specific resolution.
      
      Args:
          target_width, target_height: Bucket dimensions (e.g., 1216, 832)
      
      Returns:
          numpy array of shape (num_patches, 1024) where:
              num_patches = (dino_h // 14) * (dino_w // 14)
              dino_h, dino_w = nearest multiples of 14 to target dimensions
      """
      # Round to nearest multiple of 14
      dino_h = round(target_height / 14) * 14
      dino_w = round(target_width / 14) * 14
      
      # Use processor with dynamic size, no center-crop
      inputs = dino["processor"](
          images=image,
          size={"height": dino_h, "width": dino_w},
          do_center_crop=False,  # ✅ PRESERVE SPATIAL ALIGNMENT
          do_resize=True,
          return_tensors="pt"
      )
      
      inputs = {k: v.to(device) for k, v in inputs.items()}
      with torch.no_grad():
          outputs = dino["model"](**inputs)
      
      # Extract patches (skip CLS at index 0, exclude register tokens)
      # Output shape: (1, num_tokens, 1024) where num_tokens = 1 CLS + patches + registers
      num_patches = (dino_h // 14) * (dino_w // 14)
      patches = outputs.last_hidden_state[0, 1:num_patches+1, :]  # (num_patches, 1024)
      
      return patches.detach().cpu().float().numpy()
  ```

- [ ] **Update call sites** to pass bucket dimensions:
  ```python
  # In main processing loop:
  bucket_w, bucket_h = ...  # Extract from record['aspect_bucket']
  patches = compute_dinov3_patches(dino, device, pil_image, bucket_w, bucket_h)
  ```

**Time estimate**: ~1-2 hours to fix preprocessing

#### 1.0.1 DELETE OLD (MISALIGNED) PATCHES
**CRITICAL**: Existing patches were computed with center-crop → spatially misaligned!

- [ ] **Delete old patches directory**:
  ```bash
  # Backup for reference (optional)
  mv data/derived/dinov3_patches data/derived/dinov3_patches_OLD_MISALIGNED
  
  # Or just delete
  rm -rf data/derived/dinov3_patches
  ```

- [ ] **Regenerate ALL patches** with fixed preprocessing:
  ```bash
  python scripts/generate_approved_image_dataset.py \
    --pass-filter dinov3 \
    --device cuda
  ```
  
**Time estimate**: ~1-2 hours (depends on dataset size)

#### 1.1 Update Shard Creation Script
**File**: `scripts/create_webdataset_shards.py`

**After** patches are regenerated with correct spatial alignment:

- [ ] Add `dinov3_patches` directory to checked directories (line ~259)
- [ ] Add patches path to `iter_ready_records()`:
  ```python
  dino_patches_dir = derived_dir / "dinov3_patches"
  
  # In completeness check (line ~149):
  dino_patch_path = dino_patches_dir / f"{image_id}.npy"
  if not (...and dino_patch_path.is_file()):
      counters.skipped_incomplete += 1
      continue
  
  # In yield dict (line ~160):
  yield {
      ...
      "dino_path": dino_path,
      "dino_patch_path": dino_patch_path,  # NEW
      ...
  }
  ```

- [ ] Add patches to tar writing (line ~228):
  ```python
  add_file(tf, f"{image_id}.dinov3.npy", s["dino_path"])
  add_file(tf, f"{image_id}.dinov3_patches.npy", s["dino_patch_path"])  # NEW
  add_file(tf, f"{image_id}.vae.npy", s["vae_path"])
  ```

- [ ] **Regenerate shards** with correctly-aligned patches:
  ```bash
  # Backup old shards
  mv data/shards/10000 data/shards/10000_no_patches_backup
  
  # Create new shards with patches
  python scripts/create_webdataset_shards.py \
    --output-dir data/shards/10000 \
    --shard-size 1000 \
    --overwrite
  ```

**Time estimate**: ~30 min to modify script, ~1 hour to regenerate shards

#### 1.2 Update WebDataset Loading
**File**: `production/data.py`

- [ ] Add `dinov3_patches.npy` to sample keys in WebDataset pipeline
- [ ] Handle **variable-length patches** (shape varies per bucket!)
- [ ] Add normalization for patches (use same DINO stats as CLS)
- [ ] Update batch structure to include patches
- [ ] **IMPORTANT**: Batch size MUST be 1 for variable-length sequences (already the case)

**Changes needed**:
```python
# In decode_sample():
sample['dinov3_patches'] = np.load(...)  # Shape: (num_patches, 1024) - VARIABLE!

# Normalization (use same DINO stats):
if 'dinov3_patches' in sample:
    patches = sample['dinov3_patches']  # (num_patches, 1024)
    patches = (patches - dino_mean) / dino_std  # Apply DINO normalization
    sample['dinov3_patches'] = patches

# In collate (batch_size=1, so no padding needed):
batch['dinov3_patches'] = torch.from_numpy(patches).to(dtype)  # (1, num_patches, 1024)
```

**Note**: Variable-length sequences mean we CANNOT batch multiple samples together without padding. This is fine - our batch_size=1 with gradient accumulation already handles this.

#### 1.3 Verify Patch Availability
**Script**: Quick verification

- [ ] Count how many training samples have patches on disk
  ```bash
  find data/derived/dinov3_patches -name "*.npy" | wc -l
  ```
  Expected: Should match number of images in dataset.

- [ ] **Verify patch shapes are variable and correct**:
  ```bash
  python3 << 'EOF'
  import numpy as np
  from pathlib import Path
  
  patches_dir = Path("data/derived/dinov3_patches")
  
  # Sample a few patches and check shapes
  for i, p in enumerate(list(patches_dir.glob("*.npy"))[:10]):
      arr = np.load(p)
      print(f"{p.name}: {arr.shape}")
      if i >= 9:
          break
  
  # Expected: Different shapes like (5329, 1024), (5133, 1024), etc.
  EOF
  ```

- [ ] Ensure patches exist for validation set
- [ ] After regenerating shards, verify they include patches:
  ```bash
  # List contents of first shard
  tar -tf data/shards/10000/bucket_1024x1024/shard-000000.tar | grep patches | head -5
  ```
  Should see: `{image_id}.dinov3_patches.npy` files

---

### Phase 2: Model Architecture Changes (1-2 hours)

**Goal**: Modify NanoDiT to support concatenated cross-attention with CLS fallback.

#### 2.1 Update DiTBlock Class
**File**: `production/model.py` (lines ~190-245)

**Current structure**:
```
Self-Attention (latents → latents)
Cross-Attention (latents → T5 text only)
MLP
```

**New structure**:
```
Self-Attention (latents → latents)
Cross-Attention (latents → [T5 + CLS + patches] concatenated)
MLP
```

**Changes**:
- [ ] **No new attention module needed** - reuse existing `self.cross_attn`
- [ ] Update forward signature to accept `c_dino_patches` (variable-length!)
- [ ] Concatenate `c_text + c_dino_cls + c_dino_patches` before cross-attention
- [ ] **Dynamic mask concatenation** (patch count varies per sample)

**Pseudocode**:
```python
class DiTBlock:
    # No changes to __init__ - same 6-param adaLN
    
    def forward(self, x, c_dino, c_text, text_mask, c_dino_cls_token, c_patches):
        # c_text: (B, 512, hidden_size)
        # c_dino_cls_token: (B, 1, hidden_size)  
        # c_patches: (B, num_patches, hidden_size) - VARIABLE LENGTH!
        # Concatenate cross-attention sequence
        # Combined_context: (B, 512 + 1 + num_patches, hidden_size) - VARIABLE!
        combined_context = torch.cat([c_text, c_dino_cls_token, c_patches], dim=1)
        
        # Concatenate masks: T5 mask + all-ones for DINO
        # num_dino_tokens = 1 (CLS) + num_patches (varies per bucket)
        B = x.shape[0]
        num_dino_tokens = 1 + c_patches.shape[1]
        
        if text_mask is not None:
            dino_mask = torch.ones(B, num_dino_tokens, device=text_mask.device, dtype=text_mask.dtype)
            combined_mask = torch.cat([text_mask, dino_mask], dim=1)  # (B, 512 + num_dino_tokens)
        else:
            combined_mask = None
        
        # Single cross-attention to combined sequence
        x = x + self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca),
            context=combined_context,
            mask=combined_mask
        )
        
        # MLP
        x = x + self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x
```

**Note**: Variable-length is handled naturally by concatenation - no padding needed since batch_size=1!

#### 2.2 Update NanoDiT Class
**File**: `production/model.py` (lines ~248-435)

- [ ] Add `dino_patch_dim=1024` parameter to `__init__`
- [ ] Add `self.dino_patch_proj = nn.Linear(dino_patch_dim, hidden_size)`
- [ ] **IMPORTANT**: null embedding must handle variable length:
  ```python
  # DON'T do this (fixed size):
  # self.null_dino_patches = nn.Parameter(torch.zeros(1, 196, dino_patch_dim))
  
  # DO this (single null token, broadcast to any length):
  self.null_dino_patch_token = nn.Parameter(torch.zeros(1, 1, dino_patch_dim))
  # Usage: null_patches = self.null_dino_patch_token.expand(B, num_patches, -1)
  ```
- [ ] Update `forward()` signature to accept `dino_patches` (B, num_patches, 1024) - variable!
- [ ] Project patches: `patches_cond = self.dino_patch_proj(dino_patches)`
- [ ] **Prepare CLS for cross-attention**: `dino_cls_token = dino_cond.unsqueeze(1)` (add seq dim)
- [ ] **Concatenate for cross-attention**: `[text_cond, dino_cls_token, patches_cond]`
- [ ] Handle CFG dropout for DINO (drops both CLS and patches)
- [ ] Pass concatenated sequence to all blocks

**Key insight**: `dino_proj(CLS)` is computed once and used twice:
1. For adaLN: `dino_cond = self.dino_proj(dino_emb) + t_emb`  (B, hidden_size)
2. For cross-attn: `dino_cls_token = dino_cond.unsqueeze(1)` (B, 1, hidden_size)

**CFG Dropout Logic** (variable-length aware):
**CFG Dropout Logic** (variable-length aware):
```python
# Get number of patches for this sample (varies per bucket)
B, num_patches, _ = dino_patches.shape

# When dropping DINO: drop both CLS and patches
if cfg_drop_both is not None:
    dino_emb = torch.where(cfg_drop_both.unsqueeze(1), self.null_dino, dino_emb)
    # Broadcast null token to match patch count
    null_patches = self.null_dino_patch_token.expand(B, num_patches, -1)
    dino_patches = torch.where(
        cfg_drop_both.unsqueeze(1).unsqueeze(2),
        null_patches,
        dino_patches
    )

if cfg_drop_dino is not None:
    dino_emb = torch.where(cfg_drop_dino.unsqueeze(1), self.null_dino, dino_emb)
    # Broadcast null token to match patch count
    null_patches = self.null_dino_patch_token.expand(B, num_patches, -1)
    dino_patches = torch.where(
        cfg_drop_dino.unsqueeze(1).unsqueeze(2),
        null_patches,
        dino_patches
    )

# Project (happens after CFG dropout)
dino_cond = self.dino_proj(dino_emb) + t_emb  # (B, hidden_size)
dino_cls_token = dino_cond.unsqueeze(1)  # (B, 1, hidden_size)
text_cond = self.text_proj(text_emb)  # (B, 512, hidden_size)
patches_cond = self.dino_patch_proj(dino_patches)  # (B, num_patches, hidden_size)
```

#### 2.3 Initialize New Parameters
**Strategy**: Normal init for new projections

- [ ] Normal init for `dino_patch_proj` (like existing projections)
- [ ] Zero-initialize `null_dino_patch_token` (single token, learned during training)
- [ ] **No new attention modules** - reusing existing cross_attn (simpler!)

---

### Phase 3: Training Loop Integration (1 hour)

**Goal**: Pass variable-length patches through training loop with CFG dropout.

#### 3.1 Update Trainer Forward Pass
**File**: `production/train.py` (flow_matching_loss function, lines ~34-102)

- [ ] Extract `dinov3_patches` from batch dict
- [ ] Pass `dino_patches` to model forward call
- [ ] Ensure shapes match: `(B, num_patches, 1024)` - **variable length!**

**Changes**:
```python
def flow_matching_loss(model, batch, ...):
    dinov3 = batch['dinov3']  # (B, 1024) - existing
    dinov3_patches = batch['dinov3_patches']  # (B, num_patches, 1024) - NEW, VARIABLE!
    
    v_pred = model(
        x_t, t, 
        dino_emb=dinov3, 
        dino_patches=dinov3_patches,  # NEW
        text_emb=t5_hidden,
        ...
    )
```

#### 3.2 Verify CFG Dropout
**File**: `production/train.py` (lines ~60-85)

- [ ] Confirm CFG masks apply to both DINO CLS and patches
- [ ] Test that `cfg_drop_dino` correctly nullifies patches during training

---

### Phase 4: Validation & Sampling (1-2 hours)

**Goal**: Support patches in validation and inference.

#### 4.1 Update Validation Tests
**File**: `production/validate.py`

- [ ] Load patches from validation dataloader
- [ ] Pass patches to model during validation forward passes
- [ ] Reconstruction test: use original image's patches
- [ ] DINO swap test: swap patches along with CLS token
- [ ] Text-only test: use null patches when dropping DINO

#### 4.2 Update Sampler
**File**: `production/sample.py` (EulerSampler, lines ~10-96)

- [ ] Accept `dino_patches` parameter (optional, defaults to null)
- [ ] Prepare `dino_cls_token` from CLS for concatenation
- [ ] Support CFG with patches (concatenated sequence):
  - Uncond: null CLS + null patches + null text
  - Text-only: null CLS + null patches + text
  - DINO-only: real CLS + real patches + null text
  - Full: real CLS + real patches + text

**Note**: Existing dual CFG formula should work with concatenated sequence - no changes needed to CFG math, just pass concatenated context.

#### 4.3 Update Visual Debug
**File**: `production/visual_debug.py`

- [ ] Load patches for debug samples
- [ ] Pass to sampler during quick generation

---

### Phase 5: Testing & Debugging (2-3 hours)

**Goal**: Verify implementation before committing to long training run.

#### 5.1 Unit Tests

- [ ] **Shape test**: Verify all tensor shapes through forward pass
  ```python
  # Test: batch_size=2, different aspect ratios
  x = torch.randn(2, 16, 64, 64)  # VAE latents
  t = torch.randn(2)
  dino_cls = torch.randn(2, 1024)
  dino_patches = torch.randn(2, 196, 1024)
  text = torch.randn(2, 512, 4096)
  
  v = model(x, t, dino_cls, dino_patches, text, ...)
  assert v.shape == (2, 16, 64, 64)
  ```

- [ ] **CFG dropout test**: Verify null embeddings are applied correctly
  ```python
  cfg_drop_dino = torch.tensor([True, False])
  v = model(..., cfg_drop_dino=cfg_drop_dino)
  # Check that null_dino_patches is used for batch[0]
  ```

- [ ] **Gradient flow test**: Ensure patches receive gradients through concatenated cross-attn
  ```python
  loss = v.sum()
  loss.backward()
  assert model.dino_patch_proj.weight.grad is not None
  assert model.blocks[0].cross_attn.kv.weight.grad is not None
  ```

#### 5.2 Warm-Start Checkpoint Loading

**Challenge**: New parameters (dino_patch_proj, null_dino_patches) don't exist in old checkpoint.

**Strategy**:
- [ ] Modify checkpoint loading to skip missing keys with warning
- [ ] Verify old parameters load correctly (existing projections, cross_attn, etc.)
- [ ] Initialize new parameters with proper init strategy

**Code**:
```python
# In train.py load_checkpoint():
state_dict = checkpoint['model']
model_state = model.state_dict()

# Load matching keys, skip new ones
for name, param in state_dict.items():
    if name in model_state:
        if param.shape == model_state[name].shape:
            model_state[name].copy_(param)
        else:
            print(f"Shape mismatch for {name}: checkpoint {param.shape} vs model {model_state[name].shape}")
    else:
        print(f"Skipping {name} (not in new model)")

# Warn about new parameters
for name in model_state.keys():
    if name not in state_dict:
        print(f"New parameter {name} will be randomly initialized")
```

**Expected new parameters**: 
- `dino_patch_proj.weight`, `dino_patch_proj.bias`
- `null_dino_patches`

#### 5.3 Short Training Run (100 steps)

- [ ] Run 100 training steps on full dataset
- [ ] Monitor:
  - Loss should be similar to pre-patch baseline (~1.0-1.5)
  - Velocity norm should be healthy (~1.0-2.0)
  - Gradient norm should be reasonable (<5.0)
- [ ] Check for NaN/Inf in concatenated cross-attention
- [ ] Verify CFG dropout is working (log which samples use null patches)
- [ ] Verify CLS appears in cross-attention context (shape should be 709 tokens)

#### 5.4 Visual Validation

- [ ] Generate 4 samples at step 100
- [ ] Compare to baseline (pre-patches) at similar step
- [ ] Look for:
  - Better spatial structure (edges, faces in correct positions)
  - No degradation in color/style accuracy
  - No obvious artifacts from patches cross-attn

---

### Phase 6: Configuration & Documentation (30 min)

#### 6.1 Update Config
**File**: `production/config.yaml`

- [ ] Add `use_dino_patches: true` flag (for future flexibility)
- [ ] Document patches shape in comments
- [ ] Note warm-start from checkpoint without patches

#### 6.2 Update README
**File**: `README.md`

- [ ] Update architecture section to mention parallel cross-attention
- [ ] Update "Known Limitations" - remove "no spatial conditioning"
- [ ] Add note about patches in "Future Work" → "Current Work"

#### 6.3 Code Comments

- [ ] Add docstrings to new cross_attn_patches
- [ ] Document CFG dropout behavior with patches
- [ ] Explain adaLN parameter count increase (6→8)

---

### Phase 7: Full Training (Ongoing)

#### 7.1 Resume Training

- [ ] Warm-start from current checkpoint (~10k steps, 768 hidden)
- [ ] Initial learning rate: reduce by 2× for stability (1.5e-4 instead of 3e-4)
- [ ] Monitor first 5k steps closely for instability

#### 7.2 Monitoring

Watch for:
- **Loss trajectory**: Should continue decreasing from current value
- **Velocity norm**: Should stay 0.5-3.0 range
- **Attention patterns**: Could visualize what tokens each latent patch attends to (text vs CLS vs patches)
- **Gradient norms**: Watch dino_patch_proj gradients (shouldn't be zero or exploding)

#### 7.3 Validation Milestones

- **5k steps**: Quick check (visual_debug)
- **10k steps**: Full validation suite
  - Reconstruction LPIPS should improve vs baseline
  - DINO swap should show better spatial transfer
  - Text manipulation should remain strong
- **20k steps**: Compare to baseline at same step count
- **50k+ steps**: Assess if spatial layout is improving

---

## Risk Assessment

### High Risk
1. **Information Leakage**: Patches contain exact spatial layout
   - **Mitigation**: CFG dropout (20% text-only) forces model to learn text→spatial correlation
   - **Test**: Generate from pure text (null patches) - should produce reasonable layouts

2. **Training Instability**: Adding new cross-attention mid-training
   - **Mitigation**: Warm-start + reduced LR + zero-init output layers
   - **Test**: Monitor first 1k steps for NaN/Inf

### Medium Risk
3. **Model Ignoring Text**: Patches are so informative, model ignores text
   - **Mitigation**: Current dropout (30% text-only) forces text-only generation practice
   - **Additional benefit**: CLS in cross-attn provides fallback, shouldn't overpower text
   - **Test**: Text manipulation validation should show strong text conditioning

4. **Checkpoint Compatibility**: Warm-start might fail or cause issues
   - **Mitigation**: Graceful handling of missing keys, option to start fresh
   - **Fallback**: If unstable after 1k steps, restart from step 0

5. **CLS Redundancy**: CLS in both adaLN and cross-attn might be wasteful
   - **Counter**: Serves different purposes (global modulation vs attention routing)
   - **Benefit**: Provides fallback token for attention when patches aren't needed
   - **Monitor**: Check if CLS token is actually attended to during training

### Low Risk
6. **Memory Issues**: 196 patches adds ~380MB per batch
   - **Current**: ~5GB VRAM usage with batch_size=1, gradient accumulation
   - **Estimate**: Patches add <1GB → should fit in 24GB VRAM
   - **Advantage**: Concatenation uses less memory than parallel cross-attention
   - **Mitigation**: Reduce gradient accumulation if needed

---

## Success Criteria

### Minimum Viable (MVP)
- [ ] Training runs without errors for 10k steps
- [ ] Loss trajectory similar to baseline
- [ ] Visual_debug shows no major degradation

### Good Success
- [ ] Reconstruction LPIPS improves by >10% vs baseline
- [ ] Spatial layout noticeably better ("blobs" in correct places)
- [ ] Text conditioning remains strong

### Excellent Success
- [ ] Reconstruction LPIPS <0.4 by 50k steps
- [ ] Generated images from pure text show reasonable spatial layout
- [ ] DINO swap transfers spatial structure accurately
- [ ] Model can generate from text-only (null patches) with acceptable quality

---

## Timeline Estimate

**Development**: 12-18 hours (updated for dynamic resolution + preprocessing fix)
- Phase 1 (Data): 4-6h (**CRITICAL**: includes fixing preprocessing bug + regenerating all patches + shards!)
  - Fix preprocessing: 1-2h
  - Regenerate patches: 1-2h
  - Update shard script: 0.5h
  - Regenerate shards: 1-1.5h
- Phase 2 (Model): 1-2h (simpler - no second cross-attn module, variable-length handled naturally)
- Phase 3 (Training): 1h
- Phase 4 (Validation): 1-2h (need to handle variable-length in validation too)
- Phase 5 (Testing): 2-3h
- Phase 6 (Docs): 0.5h

**Debugging buffer**: +4h (variable-length sequences, preprocessing verification)

**Total before training**: ~16-22 hours

**Training**: 
- 100 steps test: 5 min
- 10k steps: ~7 hours (at 1.6 it/s)
- 50k steps: ~35 hours

**Total to first assessment**: ~3-4 days

---

## Critical Path & Risks

### ⚠️ CRITICAL: Spatial Alignment Bug
**Status**: Discovered, not yet fixed
**Impact**: HIGH - All existing patches are spatially misaligned
**Required**: Must fix preprocessing AND regenerate all patches before any training
**Time**: ~3-4 hours minimum (preprocessing fix + patch regeneration)

### Risk: Variable-Length Sequences
**Mitigation**: batch_size=1 already, so no padding needed
**Testing**: Verify different buckets produce correct patch counts

### Risk: Memory Usage
**Issue**: Larger cross-attention sequences (5842 vs 709 tokens for 1024×1024)
**Mitigation**: Gradient checkpointing already enabled, batch_size=1
**Monitoring**: Watch GPU memory during initial training

### Risk: Warm-Start Compatibility
**Issue**: New parameters (dino_patch_proj, null_patch_token) won't exist in checkpoint
**Mitigation**: Load matching parameters, initialize new ones randomly
**Fallback**: Train from scratch if warm-start causes instability

---

## Open Questions - RESOLVED ✓

(See earlier section for all resolved questions)

## Notes

- DINOv3 patches are pre-computed and stored in `data/derived/dinov3_patches/`
- Patches are 14×14 grid from 224×224 DINO input (16×16 pixel patches)
- Each patch represents ~16×16 pixels of the 224px DINO view
- After bucketing, this corresponds to different coverage of the actual image
- The model will need to learn this mapping implicitly

---

**Status**: Ready for implementation
**Last Updated**: 2026-02-20
**Next Step**: Begin Phase 1 - Data Pipeline
