# DWPose Conditioning Integration Guide

This directory contains complete documentation for integrating DWPose conditioning into the NanoDiT model training pipeline.

## Documentation Files

### 1. **DWPOSE_QUICK_REFERENCE.md** (Start Here! ⭐)
- **Purpose**: High-level overview and checklist
- **Contents**:
  - Architecture overview
  - Integration checklist organized by file
  - Expected data formats
  - Implementation strategy (Integrated vs Independent)
  - File modification summary
  - Testing checklist
- **Best for**: Quick lookup, planning implementation, understanding dependencies

### 2. **DWPOSE_INTEGRATION_GUIDE.txt** (Detailed Reference)
- **Purpose**: Complete line-by-line implementation guide
- **Contents**:
  - All exact line numbers from source files
  - 10+ lines of context for each section
  - Complete function signatures
  - CFG dropout logic with exact code snippets
  - Data loading pipeline details
  - WebDataset tar file structure
  - Configuration options
  - Summary integration table
- **Best for**: Implementation, copying code snippets, understanding exact changes

## Quick Navigation

### By Component:

**Model Architecture** (`production/model.py`)
- Add projection layers (lines 328-330, 353-355)
- Update forward signature (lines 420-422)
- Implement CFG dropout (lines 454-475)
- Project conditioning tokens (lines 486-495)
- Pass to blocks (lines 516, 524, 536, 542)

**Training Loop** (`production/train.py`)
- CFG dropout sampling (lines 206-224)
- Batch unpacking (lines 598-605)
- Loss function calls (lines 609-619)
- Function signatures (lines 179, 234-240)

**Data Loading** (`production/data.py`)
- Load from WebDataset (lines 262-266)
- Process sample (lines 284-292)
- Collate function (lines 299-330)
- BucketAwareDataLoader (lines 425-488)

**Inference Sampling** (`production/sample.py`)
- Sampler signature (lines 23-37)
- Dual CFG implementation (lines 92-117)
- ValidationSampler (lines 287-299)

**Data Creation** (`scripts/create_webdataset_shards.py`)
- File requirements (lines 114-155)
- Tar file writing (lines 244-262)

**Configuration** (`production/config.yaml`)
- CFG dropout section (lines 36-42)
- Sampling scales (lines 118-123)

## Key Concepts

### Current Architecture
- **Conditioning**: DINO (CLS + patches) + T5 text
- **Projection**: All conditioning → hidden_size (384 or 768)
- **Token assembly**: Concatenated for cross-attention
- **CFG dropout**: Mutually exclusive categorical (10% uncond, 30% text-only, 5% cls-only, 5% patches-only, 50% full)
- **Null embeddings**: Learned parameters for each conditioning type
- **Inference**: Dual CFG with 3 model passes OR self-guidance with 2 passes

### Integration Options

**Option A: Integrated with DINO** (Simpler)
- DWPose drops when DINO drops, kept when DINO kept
- No new CFG category
- 3 inference passes (same as current)
- Less compute overhead

**Option B: Independent Control** (More Flexible)
- Add `p_dwpose_only` CFG category
- Independent dropout control
- 4+ inference passes for full combinations
- More conditioning control

## Data Format

### Expected Input Tensors
```
DINO CLS:           (B, 1024)
DINO patches:       (B, num_patches, 1024)  — VARIABLE LENGTH
T5 text:            (B, 512, 1024)
T5 mask:            (B, 512) binary
DWPose:             (B, seq_len_pose, pose_dim)  — assume variable
DWPose mask:        (B, seq_len_pose) binary
```

### After Projection to hidden_size
```
DINO CLS:           (B, 384)
DINO patches:       (B, num_patches, 384)
T5 conditioning:    (B, 512, 384)
DWPose conditioning: (B, seq_len_pose, 384)
```

### Null Embeddings
```
null_dino:                (1, 1024)
null_dino_patch_token:    (1, 1, 1024)
null_text:                (1, 1, 1024)
null_dwpose:              (1, 1, pose_dim)  — NEW
```

## Implementation Phases

### Phase 1: Model Architecture (1-2 hours)
1. Add `dwpose_proj` and `null_dwpose` to `__init__`
2. Update forward signature with new parameters
3. Implement CFG dropout logic
4. Project and pass DWPose to blocks

### Phase 2: Training Integration (1 hour)
1. Update `flow_matching_loss()` signature
2. Add/choose CFG dropout mask sampling strategy
3. Unpack from batch and pass through training loop
4. Update config with probabilities/scales

### Phase 3: Data Pipeline (2-3 hours)
1. Load `dwpose.npy` from WebDataset
2. Implement variable-length padding in collate
3. Update `create_webdataset_shards.py` for tar creation
4. Test data loading with real/dummy data

### Phase 4: Inference & Testing (2-3 hours)
1. Update sampler signatures
2. Implement CFG passes (3 vs 4+)
3. Integration and validation tests
4. Ablation studies

## Testing Strategy

1. **Unit Tests**: Forward pass with dummy inputs
2. **Data Integration**: Load & process embeddings
3. **Training**: Loss computation, gradient flow, checkpoints
4. **Inference**: CFG passes, sampling quality
5. **Ablation**: Measure conditioning effectiveness

## Configuration Changes

### Minimal (Integrated Strategy)
```yaml
# config.yaml — No changes needed to cfg_dropout
# Model: Add dwpose_dim parameter
```

### Full (Independent Strategy)
```yaml
# config.yaml
training:
  cfg_dropout:
    p_uncond: 0.05
    p_text_only: 0.25
    p_dino_only: 0.05
    p_dwpose_only: 0.10
    # etc. sum to 100%

sampling:
  dwpose_scale: 2.0
```

## Important Notes

1. **Variable-Length Handling**: If DWPose sequences vary in length (like DINO patches), apply padding in `collate_fn()` and track with binary mask

2. **CFG Dropout Pattern**: Uses `torch.where()` with unsqueeze() for broadcasting:
   - Masks are (B,) boolean
   - Unsqueeze for proper broadcasting to (B, 1, ...) or (B, 1, 1, ...)

3. **Timestep Embedding**: Current code adds timestep to DINO CLS. Decide for DWPose:
   - Add timestep? Or keep raw pose representation?

4. **Block Interface**: Check `DiTBlock.forward()` signature before adding new parameters

5. **REPA/TREAD Compatibility**: Should work as-is since DWPose is just another conditioning sequence

## DiTBlock Modification Needed

You'll need to check and potentially modify `DiTBlock.forward()` to accept:
- `dwpose_cond: (B, seq_len_pose, hidden_size)`
- `dwpose_mask: (B, seq_len_pose)` (optional)

Current block likely concatenates all conditioning tensors for cross-attention, so DWPose should fit into that pattern naturally.

## Files Modified

| File | Lines | Changes | Priority |
|------|-------|---------|----------|
| `production/model.py` | 9 locations | ~10-15 new | High |
| `production/train.py` | 5 locations | ~15-20 new | High |
| `production/data.py` | 4 locations | ~15-20 new | High |
| `production/sample.py` | 3 locations | ~10-20 new | Medium |
| `scripts/create_webdataset_shards.py` | 2 locations | ~5-10 new | Medium |
| `production/config.yaml` | 2 sections | ~2-5 new | Low |

## Validation Points

- [ ] Model loads and forward pass works
- [ ] CFG masks applied correctly
- [ ] Batch loading with padding works
- [ ] Loss computation valid
- [ ] EMA updates include new params
- [ ] Checkpoints save/load properly
- [ ] Sampling with CFG passes correct shapes
- [ ] WebDataset shards created with dwpose files
- [ ] Validation metrics show pose conditioning effect

## Questions to Answer During Implementation

1. What is `pose_dim` for DWPose embeddings?
2. Should timestep be added to DWPose conditioning?
3. Choose: Integrated vs Independent CFG control?
4. How long is the DWPose sequence (fixed or variable)?
5. Is there a separate DWPose attention mask, or always fully valid?
6. Should REPA loss also include pose alignment?

## References

- Original file paths: `/home/tim/source/activity/prx-tg/production/`
- Line numbers reference current codebase state
- All code snippets are from active source files

---

**Last Updated**: March 2024
**Status**: Analysis Complete, Ready for Implementation
**Estimated Implementation Time**: 6-8 hours

For detailed line-by-line guidance, see `DWPOSE_INTEGRATION_GUIDE.txt`
