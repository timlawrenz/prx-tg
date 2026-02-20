# Implementation Plan: DINOv3 Spatial Patches Integration

## Problem Statement

Current architecture uses only DINO CLS token (1024-dim global vector) for visual conditioning, resulting in "blobs in wrong places" - correct colors/style but poor spatial layout. DINOv3 extracts 196 spatial patch embeddings (14×14 grid) that are currently computed but unused.

**Goal**: Integrate DINO patches into the architecture to provide spatial conditioning while maintaining text-driven generation capability.

## Architecture Decisions (Confirmed)

### Method: Concatenated Cross-Attention with CLS Fallback (Hybrid Approach)
- **Single cross-attention module** per DiT block (simpler than parallel)
- **Concatenate sequence**: T5 text [512] + DINO_CLS [1] + DINO_patches [196] = **709 tokens total**
- All projected to same hidden_size before concatenation

**Rationale**: 
- **Simpler than parallel**: Fewer parameters, single attention mechanism
- **CLS as fallback**: Attention heads can route to global CLS when local patches aren't needed (e.g., smooth/uniform regions)
- **Dual CLS usage**: 
  1. adaLN modulation (controls entire block via scale/shift)
  2. Global token in cross-attention (fallback for spatial routing)
- **Efficient routing**: Model learns to attend to text semantics, global style, or specific spatial patches as needed

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

3. **DINO Patches**:
   - Shape: [196, 1024] (native DINOv3 14×14 grid)
   - Projected to [196, hidden_size] via new `dino_patch_proj`
   
4. **Combined Cross-Attention Sequence**:
   - Concatenate: `[text_proj(T5), dino_proj(CLS), dino_patch_proj(patches)]`
   - Final shape: `[709, hidden_size]` where 709 = 512 + 1 + 196
   - Model learns to route attention between text concepts, global style, and spatial features

4. **CFG Dropout Strategy**:
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

### Phase 1: Data Pipeline (1-2 hours)

**Goal**: Load DINO patches during training without breaking existing flow.

#### 1.1 Update WebDataset Loading
**File**: `production/data.py`

- [ ] Add `dinov3_patches.npy` to sample keys in WebDataset pipeline
- [ ] Handle missing patches gracefully (skip samples if patches don't exist)
- [ ] Add normalization for patches (same stats as CLS, or compute separately)
- [ ] Update collate function to include patches in batch

**Changes needed**:
```python
# In decode_sample() or similar:
sample['dinov3_patches'] = np.load(...)  # Shape: (196, 1024)

# In collate or transform:
patches = torch.from_numpy(patches).to(dtype)  # (B, 196, 1024)
```

#### 1.2 Verify Patch Availability
**Script**: Quick verification

- [ ] Count how many training samples have patches on disk
- [ ] Ensure patches exist for validation set
- [ ] If missing: run `scripts/generate_approved_image_dataset.py` to compute patches

**Command**:
```bash
find data/derived/dinov3_patches -name "*.npy" | wc -l
```

Expected: Should match number of images in dataset.

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
- [ ] Update forward signature to accept `c_dino_patches` (separate from `c_text`)
- [ ] Concatenate `c_text + c_dino_cls + c_dino_patches` before cross-attention
- [ ] Handle combined mask: T5 has padding mask, DINO (CLS+patches) always valid

**Pseudocode**:
```python
class DiTBlock:
    # No changes to __init__ - same 6-param adaLN
    
    def forward(self, x, c_dino, c_text, text_mask, c_dino_cls_token, c_patches):
        shift_msa, scale_msa, shift_ca, scale_ca, shift_mlp, scale_mlp = \
            self.adaLN_modulation(c_dino).chunk(6, dim=1)
        
        # Self-attention
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # Concatenate cross-attention context
        # c_text: (B, 512, hidden_size)
        # c_dino_cls_token: (B, 1, hidden_size)  
        # c_patches: (B, 196, hidden_size)
        combined_context = torch.cat([c_text, c_dino_cls_token, c_patches], dim=1)
        # combined_context: (B, 709, hidden_size)
        
        # Concatenate masks: text mask (B, 512) + all-valid for DINO (B, 197)
        if text_mask is not None:
            dino_mask = torch.ones(B, 197, device=text_mask.device, dtype=text_mask.dtype)
            combined_mask = torch.cat([text_mask, dino_mask], dim=1)  # (B, 709)
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

**Note**: This is **simpler** than parallel approach - no new attention module, no extra LayerNorm, same 6-param adaLN!

#### 2.2 Update NanoDiT Class
**File**: `production/model.py` (lines ~248-435)

- [ ] Add `dino_patch_dim=1024` parameter to `__init__`
- [ ] Add `self.dino_patch_proj = nn.Linear(dino_patch_dim, hidden_size)`
- [ ] Add `self.null_dino_patches = nn.Parameter(torch.zeros(1, 196, dino_patch_dim))`
- [ ] Update `forward()` signature to accept `dino_patches` (B, 196, 1024)
- [ ] Project patches: `patches_cond = self.dino_patch_proj(dino_patches)`
- [ ] **Prepare CLS for cross-attention**: `dino_cls_token = dino_cond.unsqueeze(1)` (add seq dim)
- [ ] **Concatenate for cross-attention**: `[text_cond, dino_cls_token, patches_cond]`
- [ ] Handle CFG dropout for DINO (drops both CLS and patches)
- [ ] Pass concatenated sequence to all blocks

**Key insight**: `dino_proj(CLS)` is computed once and used twice:
1. For adaLN: `dino_cond = self.dino_proj(dino_emb) + t_emb`  (B, hidden_size)
2. For cross-attn: `dino_cls_token = dino_cond.unsqueeze(1)` (B, 1, hidden_size)

**CFG Dropout Logic**:
```python
# When dropping DINO: drop both CLS and patches
if cfg_drop_both is not None:
    dino_emb = torch.where(cfg_drop_both.unsqueeze(1), self.null_dino, dino_emb)
    dino_patches = torch.where(
        cfg_drop_both.unsqueeze(1).unsqueeze(2),
        self.null_dino_patches.expand(B, -1, -1),
        dino_patches
    )

if cfg_drop_dino is not None:
    dino_emb = torch.where(cfg_drop_dino.unsqueeze(1), self.null_dino, dino_emb)
    dino_patches = torch.where(
        cfg_drop_dino.unsqueeze(1).unsqueeze(2),
        self.null_dino_patches.expand(B, -1, -1),
        dino_patches
    )

# Project (happens after CFG dropout)
dino_cond = self.dino_proj(dino_emb) + t_emb  # (B, hidden_size)
dino_cls_token = dino_cond.unsqueeze(1)  # (B, 1, hidden_size)
text_cond = self.text_proj(text_emb)  # (B, 512, hidden_size)
patches_cond = self.dino_patch_proj(dino_patches)  # (B, 196, hidden_size)
```

#### 2.3 Initialize New Parameters
**Strategy**: Normal init for new projections

- [ ] Normal init for `dino_patch_proj` (like existing projections)
- [ ] Zero-initialize `null_dino_patches` (learned during training)
- [ ] **No new attention modules** - reusing existing cross_attn (simpler!)

---

### Phase 3: Training Loop Integration (1 hour)

**Goal**: Pass patches through training loop with CFG dropout.

#### 3.1 Update Trainer Forward Pass
**File**: `production/train.py` (flow_matching_loss function, lines ~34-102)

- [ ] Extract `dinov3_patches` from batch dict
- [ ] Pass `dino_patches` to model forward call
- [ ] Ensure shapes match: `(B, 196, 1024)`

**Changes**:
```python
def flow_matching_loss(model, batch, ...):
    dinov3 = batch['dinov3']  # (B, 1024) - existing
    dinov3_patches = batch['dinov3_patches']  # (B, 196, 1024) - NEW
    
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

**Development**: 6-10 hours (reduced from parallel approach)
- Phase 1 (Data): 1-2h
- Phase 2 (Model): 1-2h (simpler - no second cross-attn module!)
- Phase 3 (Training): 1h
- Phase 4 (Validation): 1-2h
- Phase 5 (Testing): 2-3h
- Phase 6 (Docs): 0.5h

**Debugging buffer**: +3h (fewer edge cases than parallel)

**Total before training**: ~9-13 hours

**Training**: 
- 100 steps test: 5 min
- 10k steps: ~7 hours (at 1.6 it/s)
- 50k steps: ~35 hours

**Total to first assessment**: ~2-3 days

---

## Open Questions

1. **Patches mask**: Should we mask invalid patches? (Unlikely with 14×14 grid, all valid)
2. **Attention analysis**: Should we log which tokens get attended (text vs CLS vs patches)?
3. **CFG scaling**: Do patches need separate CFG scale? (Probably not - they're part of DINO signal)
4. **CLS positioning**: Should CLS be at start [CLS, T5, patches] or middle [T5, CLS, patches]?
   - **Suggestion**: Middle position [T5, CLS, patches] - CLS bridges text and spatial
5. **Positional encoding for patches**: Add 2D pos encoding to patches before projection?
   - **Current**: Patches already have implicit spatial structure from 14×14 grid
   - **Consideration**: Could add learned 2D pos embed for explicit spatial info

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
