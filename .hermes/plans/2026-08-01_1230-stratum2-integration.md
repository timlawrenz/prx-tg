# Stratum2 (Sapiens 2) Integration into prx-tg — Technical Design

> **Status:** Code implemented and tested. Arm B config frozen. Awaiting sanity test and training launch.
> **Branch:** `exp/stratum2-integration` (commits 10dd28c, 7dd1f47)
> **No training run started until sanity test passes and user approves.**

## Goal

Integrate the five new Sapiens 2 artifacts from the stratum2 pipeline (`pose2.npy`, `seg2.npy`, `normal2.npy`, `pointmap.npy`, `matting.npy`) into prx-tg's conditioning, loss weighting, and data quality pipeline. The objective is to improve the model's ability to generate high-quality photorealistic humans from text prompts, with better facial geometry, body structure understanding, and edge rendering.

## Current Context

### What prx-tg is today

prx-tg is a pixel-space DiT (238.8M params, hidden 768, depth 18) trained on 70k FFHQ portraits from the Stratum v1 dataset. It uses:

- **Conditioning (StratumAdapter):** T5-Large text hidden states (512×1024) + DINOv3 CLS (1024) + DINOv3 patches (3696×1024) + DWPose 133-keypoint COCO-WholeBody pose — all assembled into a cross-attention sequence
- **Loss weighting:** Optional Sapiens v1 28-class segmentation-based spatial weighting (currently disabled — ablation concluded it was confounded)
- **CFG dropout:** 7-stream categorical dropout (uncond, text-only, dino-cls-only, dino-patches-only, drop-pose, pose-only, all-present)
- **Training optimizations:** Muon optimizer, TREAD token routing, REPA alignment, AsymFlow, FP8 precision
- **Inference:** Text-only generation via CFG dropout (pose and DINO dropped, text guidance scale 3.0)

### What stratum2 adds

The stratum2 pipeline (running on Strix Halo, Sapiens 2 5B models) produces these artifacts per image, verified on sample `0jkbuyws5tk2x9bxo5ui24h1o8b9`:

| Artifact | Model | Shape | Dtype | Key advantage over v1 |
|----------|-------|-------|-------|------------------------|
| `pose2.npy` | Sapiens 2 + DETR | (1, 308, 3) | float32 | 274 face landmarks (vs 68), absolute pixel coords |
| `seg2.npy` | Sapiens 2 Seg 5B | (H, W) | uint8 | 29-class (adds Tongue), full original resolution |
| `normal2.npy` | Sapiens 2 Normal 5B | (H, W, 3) | float16 | Surface normals at full resolution (vs bucket) |
| `pointmap.npy` | Sapiens 2 Pointmap 5B | (H, W, 3) | float16 | Metric 3D XYZ in camera frame (replaces relative depth) |
| `matting.npy` | Sapiens 2 Matting 1B | (H, W) | float16 | Soft alpha matte, sub-pixel hair/edge boundaries |

### Critical taxonomy difference: v1 (28-class) vs v2 (29-class)

The Sapiens 1 and Sapiens 2 segmentation models use **different class ID mappings**. This is a breaking change that must be handled in the code:

| Body part | v1 (Sapiens 1, 28-class) | v2 (Sapiens 2, DOME_CLASSES_29) |
|-----------|--------------------------|----------------------------------|
| Background | 0 | 0 |
| Face_Neck | 2 | 3 |
| Hair | 3 | 4 |
| Lower_Lip | 23 | 24 |
| Upper_Lip | 24 | 25 |
| Lower_Teeth | 25 | 26 |
| Upper_Teeth | 26 | 27 |
| Tongue | 27 | 28 |

All face-related classes shift by +1. Background remains 0. The current `build_seg_weight` hardcodes `FACE_CLASSES = {2, 3, 23, 24, 25, 26, 27}` — this will silently misweight v2 segmentation maps.

### Data source decision

The 70k FFHQ Stratum v1 dataset (`/mnt/nas-ai-models/training-data/ffhq/stratum/`) does **not** have stratum2 artifacts. The stratum2 artifacts exist only in the Crawlr dataset (`/mnt/nas-ai-models/training-data/crawlr/stratum/`).

**Two paths forward:**
1. **Re-process FFHQ with stratum2** — run `stratum2 process` on the 70k FFHQ images to generate all v2 artifacts. This requires ~95+ GB RAM (5B models) and significant time on Strix Halo.
2. **Train on Crawlr dataset** — the curated, approved photos with exactly one person per image. Likely much smaller than 70k, but higher quality and purpose-built for human photo generation.

**Recommendation:** Start with Crawlr (path 2) for the initial ablation. The dataset is already enriched with stratum2 artifacts. If the ablation shows promise, re-process FFHQ with stratum2 for a full-scale training run. This plan assumes the Crawlr dataset is available and sufficiently large for a 5k-step ablation.

## Architectural Reasoning

### Why each stratum2 artifact matters for T2I human generation

**pose2 (308 keypoints):** The current 133-keypoint DWPose has 68 face landmarks. Sapiens 2 has 274 face landmarks — individual eye corners, eyelid contours, lip vermilion borders, nose bridge, nostrils, jaw line, ear folds. During training, the model learns the relationship between text descriptions and facial geometry. 274 landmarks teach it far more precise face anatomy than 68. At inference, pose is dropped via CFG — but the learned text→geometry mapping transfers to text-only generation.

**seg2 (29-class):** The existing `build_seg_weight` uses v1's 28-class taxonomy. Sapiens 2 adds class 28 (Tongue) and provides segmentation at full original resolution (vs bucket resolution). The mouth interior (teeth, tongue, lips) is where face generation most visibly fails. Upgraded loss weighting can prioritize these regions.

**pointmap (metric 3D XYZ):** The current model has no explicit 3D understanding. DINOv3 patches provide implicit 2D appearance layout, and pose gives sparse 2D joint positions. Pointmap provides per-pixel metric XYZ coordinates — the actual 3D surface of the body in camera space. This captures body shape information that 2D pose cannot: chest depth, nose projection, shoulder curve, hip contour.

**normal2 (surface normals):** Surface normals encode how light interacts with the body surface. For photorealistic humans, understanding that cheekbones catch specular highlights, that concave eye sockets are shadowed, that skin has subsurface scattering — normals are the direct geometric signal for this.

**matting (alpha matte):** Most valuable as a data quality tool and edge-aware loss signal. The soft alpha boundary captures hair strand edges and body silhouettes — exactly where diffusion models produce artifacts.

### Why NOT to add certain signals

- **Don't add matting, normals, or pointmap as pixel-space input channels.** The model operates in pixel space (3 channels, RGB). Adding conditioning channels to the input changes the architecture and requires training from scratch.
- **Don't use seg2 as a conditioning input** (cross-attention tokens). Segmentation is a *consequence* of the body, not a *cause*. Conditioning on seg at training and dropping it at inference creates a train-test gap. Use it only for loss weighting.
- **Don't use normals as an auxiliary output** (multi-task prediction head). That's a different architecture competing with the primary flow-matching objective at this model scale (238.8M params).
- **Don't add all new modalities simultaneously.** Each new CFG stream dilutes the training signal. Start with pose2+seg2 (Tier 1), validate, then add 3D geometry (Tier 2) as a controlled ablation.

### Design principle: conditioning that teaches, then drops away

All new conditioning signals follow the same pattern as existing pose/DINO:
1. **Training:** The signal is present, teaching the model the relationship between text descriptions and body geometry/structure/lighting.
2. **Inference:** The signal is dropped via CFG. The model generates from text alone, but the learned text→geometry mapping transfers.
3. **CFG dropout:** Each new modality gets its own dropout stream, so the model learns to generate without it.

## Proposed Approach: Four-Phase Integration

Each phase is independently testable as an ablation arm. Each arm gates the next.

| Phase | Change | New files loaded | Code touched | Expected benefit |
|-------|--------|-------------------|--------------|------------------|
| **1** | pose2 (308 kp) + seg2 (29-class loss weight) | `pose2.npy`, `seg2.npy` | `data_stratum.py`, `config_loader.py`, `train.py` | Better facial geometry, mouth/teeth rendering |
| **2** | 3D geometry tokens (pointmap + normals combined) | `pointmap.npy`, `normal2.npy` | `data_stratum.py`, `adapters.py`, `config_loader.py`, `train.py` | Body depth, skin lighting |
| **3** | Matting edge-aware loss | `matting.npy` | `data_stratum.py`, `train.py` | Hair/silhouette edge quality |
| **4** | Full-scale training run | All stratum2 artifacts | Config only | Production model |

---

## Phase 1: pose2 + seg2 (Direct Drop-In Upgrades)

### 1.1 pose2.npy — 308-Keypoint Pose

**Current state:** The data loader (`data_stratum.py` line 151) loads `pose.npy` — DWPose 133 keypoints in normalized `[-1, 1]` coordinates. The `StratumAdapter` parameterizes `num_pose_joints` and handles arbitrary joint counts via `nn.Embedding(num_pose_joints, hidden_size)`.

**What changes:**

1. **Data loader:** Load `pose2.npy` instead of `pose.npy`. The array shape is `(1, 308, 3)` — one person, 308 keypoints. The coordinates are in **absolute pixel space** `[0, W] × [0, H]`, not normalized. Must normalize to `[-1, 1]`:
   ```python
   pose2 = np.load(d / 'pose2.npy')  # (1, 308, 3) float32
   pose2 = pose2[0]  # (308, 3) — single person
   # Normalize x, y from pixel space to [-1, 1]
   meta = json.loads((d / 'metadata.json').read_text())
   W, H = meta['width'], meta['height']
   pose2[:, 0] = (pose2[:, 0] / W) * 2.0 - 1.0  # x: [0, W] → [-1, 1]
   pose2[:, 1] = (pose2[:, 1] / H) * 2.0 - 1.0  # y: [0, H] → [-1, 1]
   # Confidence (pose2[:, 2]) is already in [0, ~2.0] — clip to [0, 1]
   pose2[:, 2] = pose2[:, 2].clip(0, 1)
   ```

2. **Config:** `model.num_pose_joints: 308` (was 133). Also update `adapter.num_pose_joints: 308`.

3. **Adapter:** No structural change. The MLP (`pose_proj`) takes `pose_dim=3` regardless of joint count. The `pose_joint_embed = nn.Embedding(num_pose_joints, hidden_size)` resizes automatically. The CFG dropout logic is identical — `cfg_drop_pose` drops all pose tokens.

4. **Collate:** The collate function already stacks `pose_keypoints` via `torch.stack`. Shape changes from `(B, 133, 3)` to `(B, 308, 3)`. No code change needed.

5. **Fallback for datasets without pose2:** If `pose2.npy` doesn't exist, fall back to `pose.npy` with a warning. This allows mixing v1 and v2 datasets during transition.

**Verification:** The 308-keypoint pose from the reviewed image has 289/308 keypoints with confidence > 0.5 (mean confidence 0.792). The Sapiens 2 pose model uses a DETR person detector, so it correctly identifies single-person images.

### 1.2 seg2.npy — 29-Class Segmentation Loss Weighting

**Current state:** The data loader (line 159) loads `seg.npy` (Sapiens v1, 28-class). The `build_seg_weight` function (train.py line 399) uses `FACE_CLASSES = {2, 3, 23, 24, 25, 26, 27}` with the v1 taxonomy. Seg weight is currently **disabled** in production config (`seg_weight.enabled: false`) because the ablation concluded spatial weighting was confounded.

**What changes:**

1. **Data loader:** Load `seg2.npy` instead of `seg.npy`. The shape is `(H, W)` at full original resolution (e.g., 1778×936), not bucket resolution. The downsampling to token grid via `F.interpolate(..., mode='nearest')` still works — the token grid size is derived from `target_latent_size // 16`.

2. **Class ID mapping:** Update `build_seg_weight` to use the v2 29-class taxonomy:
   ```python
   # Sapiens 2 DOME_CLASSES_29 taxonomy
   FACE_SKIN_CLASSES_V2 = {3}           # Face_Neck
   HAIR_CLASS_V2 = {4}                  # Hair
   MOUTH_CLASSES_V2 = {24, 25, 26, 27, 28}  # Lower_Lip, Upper_Lip, Lower_Teeth, Upper_Teeth, Tongue
   ```

3. **Granular weighting:** The v2 taxonomy enables finer-grained weighting than v1. Proposed weights:
   ```python
   mouth_weight: 4.0       # Teeth, tongue, lips — highest failure region
   face_skin_weight: 3.0   # Face_Neck — second priority
   hair_weight: 2.0         # Hair — edge complexity
   bg_weight: 0.5           # Background — de-emphasize
   other_weight: 1.0        # Everything else
   ```

4. **Config schema:** Extend `SegWeightConfig` to support the v2 class IDs and granular weights. Add a `taxonomy_version` field to select v1 vs v2 class mappings:
   ```python
   @dataclass
   class SegWeightConfig:
       enabled: bool = False
       taxonomy_version: int = 2       # 1 = Sapiens v1 28-class, 2 = Sapiens v2 29-class
       mouth_weight: float = 4.0       # New: lips, teeth, tongue
       face_skin_weight: float = 3.0   # Renamed from face_weight
       hair_weight: float = 2.0        # New: separate hair weighting
       bg_weight: float = 0.5
       other_weight: float = 1.0
       normalize: bool = True
       # Deprecated v1 fields (kept for backward compat)
       face_weight: float = 2.0
   ```

5. **`build_seg_weight` update:** Branch on `taxonomy_version`:
   ```python
   if cfg.taxonomy_version == 2:
       weight = torch.full_like(seg_f, cfg.other_weight)
       weight[seg_f == 0] = cfg.bg_weight
       weight[seg_f == 3] = cfg.face_skin_weight      # Face_Neck
       weight[seg_f == 4] = cfg.hair_weight             # Hair
       for cls in {24, 25, 26, 27, 28}:                 # Mouth interior
           weight[seg_f == cls] = cfg.mouth_weight
   else:  # v1 backward compat
       FACE_CLASSES = {2, 3, 23, 24, 25, 26, 27}
       weight = torch.full_like(seg_f, cfg.other_weight)
       weight[seg_f == 0] = cfg.bg_weight
       for cls in FACE_CLASSES:
           weight[seg_f == cls] = cfg.face_weight
   ```

6. **Fallback:** If `seg2.npy` doesn't exist, fall back to `seg.npy` with `taxonomy_version=1`.

### 1.3 Files to Modify (Phase 1)

| File | Change | Lines affected |
|------|--------|----------------|
| `production/data_stratum.py` | Load `pose2.npy` with normalization, load `seg2.npy` | `_load_sample()` ~line 146-176 |
| `production/config_loader.py` | Extend `SegWeightConfig` with v2 fields, add `taxonomy_version` | `SegWeightConfig` ~line 146 |
| `production/config_loader.py` | Update `ModelConfig.num_pose_joints` default to 308 | `ModelConfig` ~line 26 |
| `production/config_loader.py` | Update `AdapterConfig.num_pose_joints` default to 308 | `AdapterConfig` ~line 49 |
| `production/train.py` | Update `build_seg_weight` with v2 class mapping | `build_seg_weight` ~line 399 |
| `experiments/stratum2-pose-seg/config.yaml` | New experiment config | New file |

---

## Phase 2: 3D Geometry Conditioning (pointmap + normals)

### 2.1 Combined Geometry Tokens

**Concept:** Downsample the pointmap and normal maps to a coarse spatial grid (e.g., 16×16 = 256 tokens). Each token carries a 6D vector: 3D XYZ position (from pointmap) + 3D surface orientation (from normals). Project through an MLP to `hidden_size`, add learned positional embeddings, and concatenate into the cross-attention sequence alongside DINOv3 patches and pose tokens.

**Why combined, not separate:** The model already has 7 CFG streams. Adding 2 more (pointmap, normals separately) would dilute the training signal. Combining them into a single "3D geometry" modality with one CFG dropout stream (`cfg_drop_geometry_3d`) keeps the stream count at 8.

**Token count:** 256 tokens (16×16 grid). This is modest relative to the 3696 DINOv3 patch tokens already in the sequence.

### 2.2 Data Loader Changes

```python
# In _load_sample():
pointmap = np.load(d / 'pointmap.npy')   # (H, W, 3) float16, metric XYZ
normal2 = np.load(d / 'normal2.npy')     # (H, W, 3) float16, unit vectors
seg2 = np.load(d / 'seg2.npy')           # (H, W) uint8, for foreground mask

# Combine into 6D per-pixel: [X, Y, Z, Nx, Ny, Nz]
geometry_6d = np.concatenate([pointmap, normal2], axis=-1)  # (H, W, 6)

# Mask background using seg2 > 0
fg_mask = (seg2 > 0)
geometry_6d[~fg_mask] = 0.0

# Downsample to 16×16 grid via area interpolation
geo_t = torch.from_numpy(geometry_6d.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # (1, 6, H, W)
geo_grid = F.interpolate(geo_t, size=(16, 16), mode='area').squeeze(0).permute(1, 2, 0)  # (16, 16, 6)
geo_grid = geo_grid.reshape(256, 6)  # (256, 6) — flattened spatial grid
```

**Normalization:** Pointmap XYZ is in meters (typical range: ±1m for body, 2-3m for camera distance). Normals are unit vectors [-1, 1]. Both should be normalized:
```python
# Pointmap: divide by a typical scene scale (e.g., 3.0m) to get roughly [-1, 1]
geo_grid[:, :3] = geo_grid[:, :3] / 3.0
# Normals: already in [-1, 1], no change needed
```

### 2.3 Adapter Changes

Add to `StratumAdapter.__init__()`:

```python
# 3D geometry conditioning (pointmap + normals combined)
self.geometry_3d_enabled = geometry_3d_enabled  # bool, from config
self.geometry_3d_grid_size = 16  # 16×16 = 256 tokens
self.geometry_3d_proj = nn.Sequential(
    nn.Linear(6, hidden_size, bias=True),
    nn.GELU(),
    nn.Linear(hidden_size, hidden_size, bias=True),
)
self.geometry_3d_pos_embed = nn.Parameter(
    torch.randn(1, 256, hidden_size) * 0.02
)
self.null_geometry_3d = nn.Parameter(torch.zeros(1, 256, 6))
```

In `StratumAdapter.forward()`:

```python
geometry_3d = kwargs.get('geometry_3d')  # (B, 256, 6) or None
if geometry_3d is not None and self.geometry_3d_enabled:
    geometry_3d = self.apply_cfg_drop_source(
        geometry_3d, kwargs.get('cfg_drop_geometry_3d'),
        self.null_geometry_3d)
    geo_tokens = self.geometry_3d_proj(geometry_3d)  # (B, 256, hidden)
    geo_tokens = geo_tokens + self.geometry_3d_pos_embed
    # Append to patches_cond
    if patches_cond is not None:
        patches_cond = torch.cat([patches_cond, geo_tokens], dim=1)
        # Extend mask
        geo_mask = torch.ones(B, 256, device=..., dtype=...)
        dino_patches_mask = torch.cat([dino_patches_mask, geo_mask], dim=1)
    else:
        patches_cond = geo_tokens
        dino_patches_mask = torch.ones(B, 256, device=..., dtype=...)
```

### 2.4 CFG Dropout Changes

Add `cfg_drop_geometry_3d` to the 7-stream CFG dropout in `flow_matching_loss()`:

```python
# New stream: drop 3D geometry
p_drop_geometry_3d = cfg_probs.get('p_drop_geometry_3d', 0.10)

# In the dropout logic:
# geometry_3d is dropped when:
#   - drop_both (unconditional)
#   - cat_drop_geometry_3d (new dedicated stream)
# geometry_3d is kept in all other streams (text-only, dino-cls-only, etc.)
# This means the model learns to generate without 3D geometry from text alone
drop_geometry_3d = drop_both | cat_drop_geometry_3d
model_kwargs['cfg_drop_geometry_3d'] = drop_geometry_3d
```

**Updated CFG stream allocation (8-stream):**

| Stream | Probability | Signals kept |
|--------|-------------|--------------|
| Unconditional | 10% | None |
| Text-only | 25% | T5 text |
| DINO CLS only | 5% | DINOv3 CLS |
| DINO patches only | 5% | DINOv3 patches |
| Drop pose | 10% | Text + DINO, no pose |
| Pose only | 5% | Pose only |
| **Drop 3D geometry** | **10%** | **All except 3D geometry** |
| All present | **30%** | All signals |

*Note: Probabilities must sum to 1.0. The "all present" share drops from 40% to 30% to accommodate the new stream.*

### 2.5 Files to Modify (Phase 2)

| File | Change |
|------|--------|
| `production/data_stratum.py` | Load `pointmap.npy`, `normal2.npy`, combine into 6D grid |
| `production/adapters.py` | Add `geometry_3d_proj`, `geometry_3d_pos_embed`, `null_geometry_3d` to `StratumAdapter` |
| `production/config_loader.py` | Add `geometry_3d_enabled` to `AdapterConfig`, `p_drop_geometry_3d` to `CFGDropoutConfig` |
| `production/train.py` | Add `cfg_drop_geometry_3d` to CFG dropout logic |
| `experiments/stratum2-geometry/config.yaml` | New experiment config |

---

## Phase 3: Matting Edge-Aware Loss

### 3.1 Concept

Use the alpha matte (`matting.npy`) to boost loss weight at boundary pixels — the soft transition zone where `0.1 < alpha < 0.9`. These are hair edges, body silhouettes, and skin-clothing boundaries where the model most often produces artifacts.

### 3.2 Implementation

In `flow_matching_loss()`, alongside `build_seg_weight()`:

```python
def build_matting_edge_weight(matting, x0_shape, edge_boost=2.0, edge_low=0.1, edge_high=0.9):
    """Build spatial loss weight from alpha matte boundary zone."""
    B, C, H, W = x0_shape
    # matting: (B, TG, TG) downsampled to token grid, or (B, H, W) full res
    # Upsample to pixel grid
    m = matting.unsqueeze(1)  # (B, 1, H_m, W_m)
    m = F.interpolate(m, size=(H, W), mode='bilinear', align_corners=False)
    # Boundary zone: soft alpha region
    edge = (m > edge_low) & (m < edge_high)  # (B, 1, H, W)
    # Boost: base 1.0 + edge_boost at boundaries
    weight = torch.ones_like(m)
    weight[edge] = 1.0 + edge_boost
    return weight
```

**Combined with seg weight:** Both weight maps can be multiplied:
```python
if use_seg_weight and use_matting_edge:
    seg_w = build_seg_weight(seg_map, x0.shape, seg_weight_config)
    matting_w = build_matting_edge_weight(matting, x0.shape, matting_config)
    total_weight = seg_w * matting_w
    # Re-normalize
    total_weight = total_weight / total_weight.mean(dim=[1,2,3], keepdim=True).clamp(min=1e-6)
    loss = (sq_err * total_weight).mean()
```

### 3.3 Data Quality Filter

Beyond loss weighting, matting serves as a **data quality gate**. The stratum-hq skill documents that Sapiens seg collapses on tight face crops (<30% foreground). Use matting to flag/quarantine images:

```python
# In data loader, skip samples with insufficient foreground
fg_ratio = (matting > 0.1).mean()
if fg_ratio < 0.05:  # Less than 5% foreground — likely seg collapse
    print(f"[StratumDataset] WARNING: low foreground ({fg_ratio:.1%}) in {d.name}")
    # Option: skip, or load but flag for edge-loss exclusion
```

### 3.4 Files to Modify (Phase 3)

| File | Change |
|------|--------|
| `production/data_stratum.py` | Load `matting.npy`, pass through collate |
| `production/train.py` | Add `build_matting_edge_weight()`, integrate into loss |
| `production/config_loader.py` | Add `MattingEdgeConfig` dataclass |
| `experiments/stratum2-matting/config.yaml` | New experiment config |

---

## Phase 4: Full-Scale Training Run

Once the best-performing configuration from Phases 1-3 is identified, run a full-scale training run (40k+ steps) on the full Crawlr dataset (or re-processed FFHQ with stratum2) with all winning features enabled.

This phase requires:
1. Re-processing the full dataset with stratum2 if using FFHQ (significant compute on Strix Halo)
2. Creating a production config with all validated features
3. Following the AGENTS.md experiment governance (frozen config, provenance, git clean)

---

## Experiment Design

Four arms, each testing one incremental change. Each arm gates the next.

| Arm | Dir | Question | New signals vs baseline |
|-----|-----|----------|------------------------|
| A: `stratum2-baseline` | `experiments/stratum2-baseline/` | Reference — current production model on Crawlr + stratum2 data (v1 pose/seg only, no v2 features) | None — control |
| B: `stratum2-pose-seg` | `experiments/stratum2-pose-seg/` | Does 308-kp pose + 29-class seg loss improve face quality? | pose2, seg2 |
| C: `stratum2-geometry` | `experiments/stratum2-geometry/` | Does 3D geometry conditioning improve body structure? | pose2, seg2, pointmap, normals |
| D: `stratum2-matting` | `experiments/stratum2-matting/` | Does matting edge-aware loss improve edge quality? | pose2, seg2, pointmap, normals, matting |

**Execution order:** A → B → C → D. Each arm gates the next.

### Arm A: `stratum2-baseline` (Reference)

- Config: fork current production `config.yaml`
- `total_steps: 5000`, `checkpoint.save_every: 500`
- Data: Crawlr stratum2 dataset, but **load v1 artifacts only** (`pose.npy`, `seg.npy`)
- `num_pose_joints: 133`, `seg_weight.enabled: false` (matching current production)
- Fresh run from scratch. LR schedule matched to 5k steps.
- Purpose: Establish a baseline on the Crawlr dataset with the current model.

### Arm B: `stratum2-pose-seg` (Phase 1)

- Fork Arm A config, change:
  ```yaml
  model:
    num_pose_joints: 308
  adapter:
    num_pose_joints: 308
  training:
    seg_weight:
      enabled: true
      taxonomy_version: 2
      mouth_weight: 4.0
      face_skin_weight: 3.0
      hair_weight: 2.0
      bg_weight: 0.5
      other_weight: 1.0
      normalize: true
  ```
- Data: Load `pose2.npy` (308 kp, normalized to [-1, 1]) and `seg2.npy` (29-class)
- All other config identical to Arm A.

**Gate at step 5000:** Compare aesthetic score trajectory and DWPose confidence trajectory against Arm A. Does 308-kp pose + 29-class seg improve face quality? If B ≈ A, the added keypoints don't help at this scale — may need more training steps. If B < A, the coordinate normalization or class mapping may be wrong.

### Arm C: `stratum2-geometry` (Phase 2)

- Fork Arm B config, add:
  ```yaml
  adapter:
    geometry_3d_enabled: true
  training:
    cfg_dropout:
      p_drop_geometry_3d: 0.10
      # Adjust remaining probabilities to sum to 1.0
      p_uncond: 0.10
      p_text_only: 0.25
      p_dino_cls_only: 0.05
      p_dino_patches_only: 0.05
      p_drop_pose: 0.10
      p_pose_only: 0.05
      # p_drop_geometry_3d: 0.10 (new)
      # All present: 0.30 (reduced from 0.40)
  ```
- Data: Load `pointmap.npy` and `normal2.npy`, combine into 256-token 6D grid
- Adapter: `StratumAdapter` extended with `geometry_3d_proj`, `geometry_3d_pos_embed`

**Gate at step 5000:** Compare against Arm B. Does 3D geometry conditioning improve body structure and lighting? Check aesthetic score trajectory and look for improved body proportions in generated images. If C ≈ B, 3D geometry doesn't help at this model scale — may need larger model or more steps.

### Arm D: `stratum2-matting` (Phase 3)

- Fork Arm C config, add:
  ```yaml
  training:
    matting_edge:
      enabled: true
      edge_boost: 2.0
      edge_low: 0.1
      edge_high: 0.9
  ```
- Data: Load `matting.npy`, compute edge-aware loss weight

**Gate at step 5000:** Compare against Arm C. Does matting edge-aware loss improve hair and silhouette quality? This is best assessed visually (hair strand rendering, body outline sharpness) since the effect may be subtle and not captured by aesthetic score alone.

### Evaluation Framework

Reuse the existing two-tier validation suite from the Nyström plan (`.hermes/plans/2026-06-18_nystrom-self-attention.md`):

**Tier 1 (during training, every 500 steps):**
- Generate 10 faces from 10 fixed prompts at fixed seeds
- DWPose confidence trajectory (per-prompt, intra-sample temporal)
- CLIP Score trajectory (per-prompt, text alignment)
- LAION Aesthetic Score trajectory (per-prompt, primary gate metric)
- Log images to TensorBoard

**Tier 2 (post-training, once):**
- NN retrieval memorization check (100 faces, DINO cosine vs training set)

**Primary gate metric:** LAION Aesthetic Score at step 5000 (mean across 10 prompts).

**Fixed validation prompts:** Reuse the 10 prompts from the Nyström plan (spanning gender, age, race, emotion, pose, hair, accessories).

**Gate criteria:**

| Arm comparison | Aesthetic score Δ | Verdict |
|----------------|-------------------|---------|
| B vs A | ≥ +2% | ✅ Pose2+seg2 helps — proceed to C |
| B vs A | ±2% | ⚠️ No improvement — investigate, then proceed to C |
| B vs A | < −2% | ❌ Regression — debug normalization/class mapping |
| C vs B | ≥ +2% | ✅ 3D geometry helps — proceed to D |
| C vs B | ±2% | ⚠️ No benefit — skip D, conclude at B |
| D vs C | ≥ +1% | ✅ Matting edge loss helps |
| D vs C | < +1% | ❌ No benefit — drop matting loss |

*Note: The thresholds are tighter (2%) than the Nyström plan (5%) because these are conditioning improvements, not architectural approximations — we expect either clear improvement or no change, not degradation.*

**Collapse detector (all arms, automatic reject):**
- Velocity norm > 10.0 after warmup (step 1000) → immediate reject
- NaN in training loss at any step → immediate reject

### GPU Budget

- Arm A (baseline): ~4-5 hours on RTX 4090 (5k steps)
- Arm B (pose+seg): ~4-5 hours
- Arm C (geometry): ~5-6 hours (slightly more cross-attention tokens)
- Arm D (matting): ~5-6 hours
- Total: ~18-22 GPU hours
- **Must coordinate with any active training runs — run sequentially**

---

## New Config Schema

### Full config for Arm D (all features enabled):

```yaml
model:
  hidden_size: 768
  depth: 18
  num_heads: 12
  patch_size: 16
  mlp_ratio: 4.0
  in_channels: 3
  prediction_type: x_prediction
  input_size: 1024
  num_pose_joints: 308           # CHANGED: 133 → 308 (Sapiens 2)
  pose_confidence_threshold: 0.05

adapter:
  name: stratum
  num_pose_joints: 308           # CHANGED: must match model
  geometry_3d_enabled: true      # NEW: 3D geometry cross-attention tokens

training:
  total_steps: 5000
  warmup_steps: 500
  batch_size: 4
  grad_accumulation_steps: 64

  optimizer:
    type: Muon
    lr: 3.0e-4
    min_lr: 1.0e-6
    betas: [0.9, 0.95]
    weight_decay: 0.03
    eps: 1.0e-8
    muon:
      momentum: 0.95
      nesterov: true
      ns_steps: 5
      adjust_lr_fn: match_rms_adamw

  grad_clip: 1.0
  ema_decay: 0.9999
  ema_warmup_steps: 1000

  cfg_dropout:
    p_uncond: 0.10
    p_text_only: 0.25
    p_dino_cls_only: 0.05
    p_dino_patches_only: 0.05
    p_drop_pose: 0.10
    p_pose_only: 0.05
    p_drop_geometry_3d: 0.10    # NEW: drop 3D geometry stream
    # All present: 0.30

  repa:
    enabled: true
    weight: 0.5
    block_index: -1
    loss_type: cosine
    decay_start_step: 1000      # Scaled for 5k steps
    decay_end_step: 2500

  tread:
    enabled: true
    routing_probability: 0.5
    route_start: 1
    route_end: -1
    self_guidance: true
    guidance_scale: 3.0

  timestep_sampling: logit_normal
  logit_normal_loc: 0.0
  logit_normal_scale: 1.0

  gradient_checkpointing: false
  mixed_precision: true
  precision: bfloat16           # Use bf16 for ablation (FP8 adds complexity)

  asymflow:
    enabled: true
    rank: 8

  seg_weight:
    enabled: true                # ENABLED for stratum2
    taxonomy_version: 2         # NEW: Sapiens 2 29-class
    mouth_weight: 4.0           # NEW: lips, teeth, tongue
    face_skin_weight: 3.0       # NEW: face/neck skin
    hair_weight: 2.0            # NEW: hair
    bg_weight: 0.5
    other_weight: 1.0
    normalize: true

  matting_edge:                  # NEW: edge-aware loss
    enabled: true
    edge_boost: 2.0
    edge_low: 0.1
    edge_high: 0.9

  dino_patches:
    spatial_window_radius: null

data:
  source: stratum
  stratum_dir: "/mnt/nas-ai-models/training-data/crawlr/stratum"
  stratum_max_samples: null       # Use all available Crawlr samples
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true

sampling:
  num_steps: 35
  text_scale: 3.0
  dino_scale: 2.5
  self_guidance: true
  guidance_scale: 3.0

validation:
  enabled: true
  interval_steps: 500
  num_samples: 25
  quality_metrics:
    enabled: true
    num_prompts: 10

checkpoint:
  save_every: 500
  keep_last_n: 10
  save_optimizer: true

logging:
  log_every: 2
  monitor_velocity_norm: true
  monitor_grad_norm: true
  velocity_norm_warning: 10.0
  grad_norm_warning: 10.0
```

---

## Implementation Plan

### Task 1: Extend config_loader.py with stratum2 fields

**Objective:** Add all new config fields needed for stratum2 integration.
**Files:** `production/config_loader.py`

1. Update `ModelConfig.num_pose_joints` default: 133 → 308
2. Add `geometry_3d_enabled: bool = False` to `AdapterConfig`
3. Update `AdapterConfig.num_pose_joints` default: 133 → 308
4. Extend `SegWeightConfig` with v2 fields: `taxonomy_version`, `mouth_weight`, `face_skin_weight`, `hair_weight`
5. Add `p_drop_geometry_3d: float = 0.10` to `CFGDropoutConfig`
6. Add `MattingEdgeConfig` dataclass
7. Add `matting_edge: MattingEdgeConfig` to `TrainingConfig`
8. Update `CFGDropoutConfig.to_dict()` to include `p_drop_geometry_3d`

### Task 2: Update data_stratum.py to load stratum2 artifacts

**Objective:** Load pose2.npy, seg2.npy, pointmap.npy, normal2.npy, matting.npy with proper normalization.
**Files:** `production/data_stratum.py`

1. In `_load_sample()`:
   - Load `pose2.npy` with fallback to `pose.npy`
   - Normalize pose2 coordinates from pixel space to [-1, 1] using metadata width/height
   - Load `seg2.npy` with fallback to `seg.npy`
   - Load `pointmap.npy` and `normal2.npy`, combine into 6D grid (256 tokens)
   - Load `matting.npy` (downsampled to token grid)
2. Update `_collate()` to include `geometry_3d` and `matting` in batch dict
3. Add foreground quality check (skip samples with < 5% foreground in matting)

### Task 3: Update build_seg_weight in train.py for v2 taxonomy

**Objective:** Support both v1 (28-class) and v2 (29-class) segmentation taxonomies.
**Files:** `production/train.py`

1. Add `taxonomy_version` branching in `build_seg_weight()`
2. Implement v2 class mapping with granular weights (mouth, face_skin, hair)
3. Maintain backward compatibility with v1 configs

### Task 4: Add 3D geometry conditioning to StratumAdapter

**Objective:** Add geometry_3d_proj, positional embeddings, and CFG dropout to the adapter.
**Files:** `production/adapters.py`

1. Add `geometry_3d_enabled` parameter to `StratumAdapter.__init__()`
2. Add `geometry_3d_proj` MLP, `geometry_3d_pos_embed`, `null_geometry_3d` parameters
3. In `forward()`, accept `geometry_3d` kwarg, apply CFG dropout, project, add pos embed, append to cross-attention sequence
4. Update mask assembly to include geometry tokens

### Task 5: Add geometry_3d CFG dropout to flow_matching_loss

**Objective:** Add the 8th CFG dropout stream for 3D geometry.
**Files:** `production/train.py`

1. Add `p_drop_geometry_3d` to the categorical dropout logic
2. Compute `drop_geometry_3d` mask
3. Pass `cfg_drop_geometry_3d` to model_kwargs
4. Adjust probability bookkeeping (all-present share drops from 40% to 30%)

### Task 6: Add matting edge-aware loss

**Objective:** Weight loss higher at alpha matte boundary pixels.
**Files:** `production/train.py`

1. Add `build_matting_edge_weight()` function
2. Integrate with existing seg_weight in `flow_matching_loss()`
3. Multiply seg weight × matting weight, re-normalize

### Task 7: Create experiment configs and directories

**Objective:** Create all four arm configs with frozen configs and provenance.
**Files:** `experiments/stratum2-baseline/`, `experiments/stratum2-pose-seg/`, `experiments/stratum2-geometry/`, `experiments/stratum2-matting/`

1. Create directory structure per AGENTS.md
2. Write frozen `config.yaml` for each arm
3. Write `provenance.yaml` with git commit, diff summary
4. Write `README.md` with hypothesis and expected outcome

### Task 8: Sanity test — 50-step run

**Objective:** Verify the code changes don't crash and produce reasonable loss values.
**Files:** Run on RTX 4090

1. Run Arm B config with `total_steps: 50` on a small subset of Crawlr data
2. Verify no NaN, no crash, loss decreasing
3. Check generated validation images at step 50
4. Commit code changes only after sanity test passes

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **pose2 coordinate normalization wrong** | Model receives garbage pose → training diverges | Verify normalized coordinates match [-1, 1] range on sample data before training; check against pixel→normalized formula |
| **seg2 class ID mismatch** | Loss weights applied to wrong body parts | Unit test `build_seg_weight` with known v2 class IDs; verify Face_Neck=3, Hair=4, Mouth=24-28 |
| **308-keypoint embedding too large** | 308 joint embeddings × 768 hidden = 236K params (vs 133 × 768 = 102K) | Negligible — 0.1% of model params. No mitigation needed. |
| **3D geometry tokens add noise** | Model spends capacity learning to predict 3D structure instead of pixels | CFG dropout (10%) forces text-only generation; REPA alignment still dominates geometric learning |
| **CFG stream dilution** | 8 streams means less "all-present" training (30% vs 40%) | Monitor training loss trajectory; if loss is higher than baseline at same step, reduce p_drop_geometry_3d to 0.05 |
| **Crawlr dataset too small** | Model memorizes quickly, can't learn generalizable features | Check dataset size before training; if < 5k images, consider re-processing FFHQ with stratum2 |
| **Pointmap scale varies across images** | Different camera distances produce different XYZ ranges | Normalize by per-sample max Z (or fixed 3.0m divisor); verify on sample data |
| **Normal2 background not masked** | Background normals are noise, pollute geometry tokens | Already handled: seg2 foreground mask applied before downsampling |
| **Mixed v1/v2 datasets** | Some images have v1 artifacts only, others have v2 | Fallback logic in data loader: if v2 file missing, load v1 with v1 taxonomy. Log which samples used v1 vs v2. |
| **Matting edge loss over-weights boundaries** | Model focuses on edges at expense of interior quality | edge_boost=2.0 is conservative; monitor interior quality in generated images. Can reduce to 1.5 or disable. |
| **Memory increase from 308 pose tokens** | 308 vs 133 tokens in cross-attention sequence | 175 extra tokens × 768 hidden = ~0.5MB per batch — negligible vs 3696 DINOv3 patch tokens already present |
| **stratum2 dataset enrichment incomplete** | Some images have seg2 but not normal2/pointmap | Data loader skips missing files with warning; geometry_3d falls back to None (no geometry tokens added) |

---

## Open Questions (deferred)

1. **Crawlr dataset size:** How many approved images are in the Crawlr dataset? If < 5k, the 5k-step ablation may overfit immediately. Need to check `ls /mnt/nas-ai-models/training-data/crawlr/approved/ | wc -l` before starting.

2. **FFHQ re-processing timeline:** If Crawlr is too small, how long would re-processing 70k FFHQ images with stratum2 take on Strix Halo? The 5B models run at ~0.4-0.6 img/s per pass, and there are 5 passes (seg2, pose2, normal2, pointmap, matting). Rough estimate: 70k / 0.5 × 5 passes = ~700k seconds ≈ 8 days. This is a significant investment.

3. **Pointmap normalization:** Should we normalize per-sample (divide by per-sample max Z) or use a fixed scene scale (3.0m)? Per-sample is more robust to camera distance variation but loses absolute scale information. Fixed scale preserves depth ordering across samples. Need to experiment.

4. **Geometry grid resolution:** 16×16 = 256 tokens is a first guess. Should we try 8×8 (64 tokens, cheaper) or 24×24 (576 tokens, more detailed)? Trade-off between conditioning detail and cross-attention cost.

5. **Mouth weight tuning:** 4.0 for mouth classes is aggressive. Should we sweep 2.0, 4.0, 8.0 to find the optimal emphasis? Or is 4.0 a reasonable default?

6. **REPA interaction:** Does the 308-keypoint pose interfere with REPA alignment? REPA aligns hidden states with DINOv3 patches — adding more pose tokens changes the cross-attention distribution. Monitor `repa_loss` trajectory.

7. **Inference geometry:** At inference, all conditioning is dropped via CFG (text-only generation). But the model has learned geometry-conditioned representations. Does this transfer to better text-only generation, or does it create a distribution gap? The 10% `p_drop_geometry_3d` stream should bridge this, but needs empirical validation.

8. **Combined seg×matting weight normalization:** When both seg_weight and matting_edge are enabled, the combined weight map may have extreme values at face-boundary pixels (face_weight=3.0 × edge_boost=2.0 = 6.0). Should we clip or just rely on per-sample normalization?

---

## Experiment Directory Structure

All arms follow `docs/experiment-structure.md` — descriptive slugs, frozen configs, mandatory provenance.

### Pre-Flight Checklist (per arm)

1. `mkdir -p experiments/{slug}/{runs,validation,figures,notes}`
2. Write `config.yaml` (frozen — never edited after run starts)
3. Write `provenance.yaml` (mandatory before first run)
4. Write `README.md` (hypothesis, expected outcome, how it differs from comparator)
5. Verify `git_dirty: false` — commit or stash all code changes
6. Verify data availability: `ls /mnt/nas-ai-models/training-data/crawlr/stratum/ | wc -l`
7. Launch: `PYTORCH_ALLOC_CONF=expandable_segments:True python -m production.train_production --config experiments/{slug}/config.yaml`

### Post-Training Checklist (per arm)

1. Verify `metadata.json` exists and `git_dirty` is false
2. Run NN retrieval memorization check (100 faces, DINO cosine vs training set)
3. Save all results to `experiments/{slug}/validation/step005000/`
4. Update README.md with results summary
5. Update Experiment Registry table in project README.md

### Provenance Template

```yaml
arm: b
hypothesis: >
  308-keypoint Sapiens 2 pose + 29-class segmentation loss weighting
  improves face generation quality over 133-keypoint DWPose baseline.
differs_from: stratum2-baseline
diff_summary: |
  - model.num_pose_joints: 133 → 308
  - training.seg_weight.enabled: false → true
  - training.seg_weight.taxonomy_version: 2
  - Data: pose2.npy + seg2.npy (Sapiens 2) instead of pose.npy + seg.npy (v1)
  - All other settings identical to comparator
git_commit: <SHA>
git_dirty: false
training_host: game (RTX 4090)
training_gpu: NVIDIA RTX 4090 (24GB VRAM)
training_steps: 5000
data_source: Crawlr stratum2 dataset
evaluation: Aesthetic score + CLIP Score + DWPose confidence + NN retrieval
```

## Branching Strategy

Per experiment governance (AGENTS.md):
- Create `exp/stratum2-integration` branch from a clean commit
- All code changes (Tasks 1-6) committed on this branch
- Experiment configs (Task 7) committed on this branch
- Sanity test (Task 8) run on this branch before launching any training
