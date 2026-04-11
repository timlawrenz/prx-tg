# DWPose Conditioning Integration — Quick Reference

## Overview
This document summarizes the exact integration points for adding DWPose conditioning to the NanoDiT model. All line numbers and code snippets are from the production codebase.

## Key Facts
- **Model**: NanoDiT with hidden_size = 384 (config) or 768 (production)
- **Current Conditioning**: DINO (CLS + patches) + T5 text
- **CFG Dropout**: Mutually exclusive categorical (10% uncond, 30% text-only, 5% cls-only, 5% patches-only, 50% full)
- **Token Assembly**: All conditioning projected to hidden_size, concatenated for cross-attention
- **Inference**: Dual CFG with 3 model passes (uncond, text-only, dino-only) OR self-guidance

---

## Integration Checklist

### 1. Model Architecture (`production/model.py`)

#### 1.1 Add Projection Layer & Null Embedding
- **Location**: Lines 328-330 (projections), 353-355 (null embeddings)
- **Action**:
  ```python
  # After line 330:
  self.dwpose_proj = nn.Linear(dwpose_dim, hidden_size, bias=True)
  
  # After line 355:
  self.null_dwpose = nn.Parameter(torch.zeros(1, 1, dwpose_dim))
  ```

#### 1.2 Update Forward Signature
- **Location**: Lines 420-422
- **Add parameters**: `dwpose_emb=None, dwpose_mask=None, cfg_drop_dwpose=None`

#### 1.3 Add CFG Dropout Logic
- **Location**: After line 475 (after cfg_drop_text logic)
- **Pattern**:
  ```python
  if cfg_drop_dwpose is not None:
      null_dwpose = self.null_dwpose.expand(B, dwpose_emb.shape[1], -1)
      dwpose_emb = torch.where(
          cfg_drop_dwpose.unsqueeze(1).unsqueeze(2),
          null_dwpose,
          dwpose_emb
      )
  ```

#### 1.4 Project Conditioning
- **Location**: After line 495
- **Code**: `dwpose_cond = self.dwpose_proj(dwpose_emb)  # (B, seq_len_pose, hidden_size)`

#### 1.5 Pass to Blocks
- **Locations**: Lines 516, 524, 536, 542
- **Add**: `dwpose_cond` and `dwpose_mask=dwpose_mask` to all `self.blocks[i]()` calls

---

### 2. Training Loop (`production/train.py`)

#### 2.1 CFG Dropout Sampling
- **Location**: Lines 206-224 in `flow_matching_loss()`
- **Options**:
  - **Option 1 (Integrated)**: `drop_dwpose = drop_dino_cls | drop_dino`
  - **Option 2 (Independent)**: Add `p_dwpose_only` category to categorical sampling

#### 2.2 Batch Unpacking
- **Location**: Lines 598-605 in `train_step()`
- **Add**:
  ```python
  dwpose_emb = batch['dwpose_embedding'].to(self.device)
  dwpose_mask = batch.get('dwpose_mask')
  if dwpose_mask is not None:
      dwpose_mask = dwpose_mask.to(self.device)
  ```

#### 2.3 Function Signatures
- **`flow_matching_loss()` (line 179)**: Add `dwpose_emb=None, dwpose_mask=None`
- **Model forward call (lines 234-240)**: Add `dwpose_emb=dwpose_emb, dwpose_mask=dwpose_mask, cfg_drop_dwpose=drop_dwpose`

---

### 3. Data Loading (`production/data.py`)

#### 3.1 Load from WebDataset
- **Location**: Lines 262-266 in `process_sample()`
- **Add**:
  ```python
  dwpose_emb = sample['dwpose.npy']  # (seq_len_pose, pose_dim)
  dwpose_mask = sample.get('dwpose_mask.npy')
  ```

#### 3.2 Return Dictionary
- **Location**: Lines 284-292
- **Add**:
  ```python
  'dwpose_embedding': torch.from_numpy(dwpose_emb).float(),
  'dwpose_mask': torch.from_numpy(dwpose_mask).long() if dwpose_mask is not None else None,
  ```

#### 3.3 Variable-Length Padding (if needed)
- **Location**: `collate_fn()` method (lines 299-330)
- **Pattern**: Same as `dinov3_patches` — find max length, pad shorter sequences with zeros, create binary mask
- **Return**:
  ```python
  'dwpose_embedding': torch.stack(padded_dwpose),  # (B, max_dwpose_len, pose_dim)
  'dwpose_mask': torch.stack(dwpose_masks),        # (B, max_dwpose_len)
  ```

---

### 4. Inference Sampling (`production/sample.py`)

#### 4.1 Sampler Signature
- **Location**: Lines 23-37 in `EulerSampler.sample()`
- **Add**: `dwpose_emb, dwpose_mask, dwpose_scale=2.0` (if separate control)

#### 4.2 Dual CFG Implementation
- **Location**: Lines 92-117
- **Current**: 3 model passes (uncond, text-only, dino-only)
- **Options**:
  - **Option 1 (Integrated)**: Modify v_dino pass to keep pose when DINO kept
  - **Option 2 (Independent, 4+ passes)**: Add separate v_dwpose pass with `cfg_drop_dwpose=zeros`
    ```python
    v_pred = v_uncond + text_scale*(v_text - v_uncond) + dino_scale*(v_dino - v_uncond) + dwpose_scale*(v_dwpose - v_uncond)
    ```

#### 4.3 ValidationSampler Signature
- **Location**: Lines 287-299 in `generate()`
- **Add**: `dwpose_emb, dwpose_mask, dwpose_scale=None`

---

### 5. WebDataset Tar Creation (`scripts/create_webdataset_shards.py`)

#### 5.1 File Requirements
- **Location**: Lines 114-155 in `iter_ready_records()`
- **Add**:
  ```python
  dwpose_dir = derived_dir / "dwpose"
  dwpose_path = dwpose_dir / f"{image_id}.npy"
  
  # In existence check (line 155):
  if not (... and dwpose_path.is_file()):
  ```

#### 5.2 Add Files to Tar
- **Location**: Lines 244-262 in `write_shards()`
- **Add** (after line 260):
  ```python
  add_file(tf, f"{image_id}.dwpose.npy", s["dwpose_path"])
  if "dwpose_mask_path" in s:
      add_file(tf, f"{image_id}.dwpose_mask.npy", s["dwpose_mask_path"])
  ```

---

### 6. Configuration (`production/config.yaml`)

#### 6.1 CFG Dropout Probabilities
- **Location**: Lines 36-42 in `training.cfg_dropout`
- **Current**:
  ```yaml
  cfg_dropout:
    p_uncond: 0.1
    p_text_only: 0.3
    p_dino_cls_only: 0.05
    p_dino_patches_only: 0.05
    # Remaining 50% keeps all signals
  ```
- **Action**: If independent control, add `p_dwpose_only` and adjust all to sum to 100%

#### 6.2 Sampling Scales
- **Location**: Lines 118-123 in `sampling`
- **Add** (if separate control):
  ```yaml
  dwpose_scale: 2.0
  ```

---

## File Summary

| File | Component | Line Range | Key Variables |
|------|-----------|-----------|---|
| `model.py` | Projection layers | 328-330 | `dwpose_proj` |
| `model.py` | Null embeddings | 353-355 | `null_dwpose` |
| `model.py` | Forward signature | 420-422 | `dwpose_emb, dwpose_mask, cfg_drop_dwpose` |
| `model.py` | CFG dropout | 454-475 | Mask application with `torch.where` |
| `model.py` | Token projection | 486-495 | `dwpose_cond` |
| `model.py` | Block calls | 516, 524, 536, 542 | Pass `dwpose_cond, dwpose_mask` |
| `train.py` | CFG sampling | 206-224 | `drop_dwpose` mask |
| `train.py` | Batch unpacking | 598-605 | `dwpose_emb, dwpose_mask` |
| `train.py` | Loss function | 179, 234-240 | Function signature and forward call |
| `data.py` | Data loading | 262-266 | Load `dwpose.npy` from sample |
| `data.py` | Return dict | 284-292 | Add `dwpose_embedding, dwpose_mask` |
| `data.py` | Collate | 299-330 | Padding logic (if variable-length) |
| `sample.py` | Sampler signature | 23-37 | `dwpose_emb, dwpose_mask, dwpose_scale` |
| `sample.py` | CFG passes | 92-117 | 3-4+ model passes with pose control |
| `create_webdataset_shards.py` | File check | 114-155 | Check `dwpose_path.is_file()` |
| `create_webdataset_shards.py` | Tar writing | 244-262 | `add_file(...dwpose.npy...)` |
| `config.yaml` | CFG dropout | 36-42 | Add `p_dwpose_only` if independent |
| `config.yaml` | Sampling | 118-123 | Add `dwpose_scale` if independent |

---

## Expected Data Format

### Input Embeddings
- **DINO CLS**: (B, 1024)
- **DINO patches**: (B, num_patches, 1024) — **VARIABLE LENGTH**
- **T5 text**: (B, 512, 1024)
- **T5 mask**: (B, 512) binary
- **DWPose**: (B, seq_len_pose, pose_dim) — assume similar variable length
- **DWPose mask**: (B, seq_len_pose) binary (if variable)

### After Projection to hidden_size=384
- **DINO CLS token**: (B, 1, 384)
- **DINO patches**: (B, num_patches, 384)
- **T5 conditioning**: (B, 512, 384)
- **DWPose conditioning**: (B, seq_len_pose, 384)

### Null Embeddings
- **null_dino**: (1, 1024) → projected to (1, 384)
- **null_dino_patch_token**: (1, 1, 1024) → expanded to (B, num_patches, 1024) → projected to (B, num_patches, 384)
- **null_text**: (1, 1, 1024) → expanded to (B, 512, 1024) → projected to (B, 512, 384)
- **null_dwpose**: (1, 1, pose_dim) → expanded to (B, seq_len_pose, pose_dim) → projected to (B, seq_len_pose, 384)

---

## Implementation Strategy

### Phase 1: Basic Integration (Model)
1. Add `dwpose_proj` and `null_dwpose` to `__init__`
2. Update forward signature
3. Add CFG dropout logic
4. Project and pass to blocks

### Phase 2: Training Integration
1. Update `flow_matching_loss()` signature
2. Add CFG dropout mask sampling (choose: integrated or independent)
3. Unpack from batch and pass through training loop
4. Update config with CFG probabilities

### Phase 3: Data Pipeline
1. Load `dwpose.npy` from WebDataset samples
2. Add to batch dict
3. Implement variable-length padding in collate
4. Update `create_webdataset_shards.py` to create tar files with pose data

### Phase 4: Inference
1. Update sampler signatures
2. Implement CFG passes (3 for integrated, 4+ for independent)
3. Test sampling with pose conditioning

---

## Notes

- **Timestep Addition**: In the current code, timestep is added to DINO CLS conditioning (line 492). For DWPose, decide if timestep should be added (probably not, to avoid confusing the pose representation).
  
- **REPA/TREAD Compatibility**: DWPose should work with REPA and TREAD as-is, since it's just another conditioning sequence passed to blocks. If you want REPA loss to also align DWPose to something, you'd need to add additional projection in REPA section.

- **Variable Length Handling**: If DWPose sequences vary in length across the dataset (like patches do), apply padding in collate function and track with binary mask.

- **CFG Strategy Choice**:
  - **Integrated**: Drop DWPose whenever DINO is dropped (simpler, fewer passes)
  - **Independent**: Add separate CFG category for DWPose (more control, more compute)

---

## Testing Checklist

- [ ] Model forward pass with dwpose inputs
- [ ] CFG dropout mask generation and application
- [ ] Batch loading with variable-length padding
- [ ] Training loss computation with dwpose
- [ ] EMA updates with new projections
- [ ] Checkpoint save/load with dwpose parameters
- [ ] Sampling with dual CFG (3 or 4+ passes)
- [ ] WebDataset tar creation with dwpose files
- [ ] Validation generation with pose conditioning

---

**Full detailed guide available in: `DWPOSE_INTEGRATION_GUIDE.txt`**
