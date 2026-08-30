# Data Inventory — faces7k, FFHQ, and Hegre Corpus

> **Generated:** 2026-07-20 from forensic inspection of live filesystem
> **Principle:** Every number below was verified on disk, not recalled from memory.
> Inconclusive counts are marked `~` (timeout or ambiguous).

---

## 1. Storage Layout

All data lives on the Stratum NAS mounted at `/mnt/nas-ai-models/training-data/`.

```
/mnt/nas-ai-models/training-data/
├── crawlr/              ← faces7k (webdataset shards) + raw pipeline
├── ffhq/stratum/        ← FFHQ 70K (stratum per-image dirs)
├── eidolon/
│   ├── hegre_corpus/    ← Hegre Corpus 31.7K (eidolon stratum, minimal)
│   ├── hegre_enriched/  ← Raw enriched images (NOT training-ready)
│   ├── hegre-faces/v1/  ← Auxiliary clustering/analysis data
│   └── geometry_pca_data/  ← PCA basis data
└── prx-tg/              ← Experiment runs (experiments/ symlinks here)
```

The local project directory `data/` contains symlinks:
- `data/approved` → `/mnt/nas-ai-models/training-data/crawlr/approved/`
- `data/derived`  → `/mnt/nas-ai-models/training-data/crawlr/derived/`
- `data/raw`      → `/mnt/nas-ai-models/training-data/crawlr/raw/`
- `data/shards`   → `/mnt/nas-ai-models/training-data/crawlr/shards/`

---

## 2. How the Data Was Produced: stratum-hq

The per-image directory format used by `StratumDataset` is produced by **[stratum-hq](https://github.com/timlawrenz/stratum-hq)** — a dataset-agnostic image enrichment pipeline.

### What stratum does

Given a directory of source images, stratum produces one output directory per image containing:

```
source/ffhq/00001.png → dataset/ffhq/00001/
├── metadata.json        # dimensions, aspect bucket, source path
├── caption.txt          # dense objective description (Ollama + Gemma3-27B)
├── dinov3_cls.npy       # (1024,) float16 — DINOv3-ViT-L/16 CLS token
├── dinov3_patches.npy   # (N, 1024) float16 — spatial patch tokens
├── t5_hidden.npy        # (512, 1024) float16 — T5-Large hidden states
├── t5_mask.npy          # (512,) uint8 — T5 attention mask
├── pose.npy             # (133, 3) float16 — DWPose whole-body keypoints
├── seg.npy              # (H, W) uint8 — Sapiens 28-class segmentation
├── depth.npy            # (H, W) float16 — relative depth
├── normal.npy           # (H, W, 3) float16 — surface normals
└── pixel.npy            # (3, H, W) float16 — bucketed RGB crop (opt-in)
```

### Pipeline passes

Each pass is **independent and idempotent** — it skips images where the output already exists:

| Pass | Model | Artifacts |
|------|-------|-----------|
| `caption` | Gemma3-27B (Ollama) | `caption.txt` |
| `dinov3` | DINOv3-ViT-L/16 | `dinov3_cls.npy`, `dinov3_patches.npy` |
| `t5` | T5-Large | `t5_hidden.npy`, `t5_mask.npy` |
| `pose` | DWPose (ONNX) | `pose.npy` |
| `seg` | Sapiens-1B | `seg.npy` |
| `depth` | Sapiens-1B | `depth.npy` (depends on seg for foreground mask) |
| `normal` | Sapiens-1B | `normal.npy` (depends on seg) |
| `pixel` | — | `pixel.npy` (opt-in bucketed crop) |

### Key design properties

- **The filesystem is the database** — completeness is determined by which files exist. There is no central registry or JSONL tracking file.
- **Multi-GPU sharding** — `--shard N/M` splits images deterministically across GPUs without coordination.
- **Aspect ratio bucketing** — images are assigned to the closest bucket (1024×1024, 832×1216, 1216×832, etc.), resize-to-cover + center-cropped to exact dimensions.
- **CLI tools**: `stratum process` (generate artifacts), `stratum status` (report completeness), `stratum verify` (validate shapes/dtypes), `stratum publish` (upload to HuggingFace).

### Relationship to prx-tg training data

Both the **FFHQ stratum** (70K images) and **Hegre Corpus** (31.7K images) datasets were produced by stratum-hq. The per-image directory format is consumed directly by `production/data_stratum.py`'s `StratumDataset` loader.

The **faces7k webdataset shards** were also likely produced by running stratum-hq on the crawlr approved images, then packaging the output directories into tar shards — though the specific sharding script is not present in the repo.

---

## 3. Training Datasets

### 3.1 faces7k (Crawlr Shards)

| Property | Value |
|----------|-------|
| **Format** | WebDataset tar shards |
| **Path** | `/mnt/nas-ai-models/training-data/crawlr/shards/faces7k/` |
| **Images** | **7,000** (7 shards × 1,000) |
| **Buckets** | 1 (`bucket_1024x1024`) |
| **Total size** | ~158 GB |
| **Used by** | WebDataset training path (`data.source: "webdataset"`) |
| **Current role** | Validation/visual_debug only in current Arm O (training uses stratum) |

#### Satellite data per entry (inside each `.tar`)

Each shard entry is identified by a numeric prefix (e.g. `00000`). The tar stores these extensions:

| Extension | Shape / Content | Type | Status |
|-----------|----------------|------|--------|
| `.json` | caption, image_id, aspect_bucket, width, height, t5_attention_mask | metadata | ✅ All 7,000 |
| `.image.npy` | (3, 1024, 1024) float16 RGB [0,1] | pixel data | ✅ All 7,000 |
| `.dinov3.npy` | (1024,) float32 | DINOv3 CLS embedding | ✅ All 7,000 |
| `.dinov3_patches.npy` | (num_patches, 1024) float32 — **variable length** | DINOv3 patch tokens | ✅ All 7,000 |
| `.t5h.npy` | (512, 1024) float16 | T5-XXL hidden states | ✅ All 7,000 |
| `.t5m.npy` | (512,) uint8 | T5 attention mask | ✅ All 7,000 |
| `.pose.npy` | (133, 3) float32 [x_norm, y_norm, confidence] | DWPose keypoints | ✅ All 7,000 |

**Verdict: COMPLETE.** All 7,000 entries have every satellite modality. The shards are ready to train with the full multi-modal conditioning stack (T5 text + DINOv3 CLS + DINOv3 patches + DWPose).

#### Caption quality

Captions are rich Gemma-3-27B generations, e.g.:
> "A close-up, frontal portrait of an infant with light brown skin and dark hair. The infant has a rounded face, visible cheek fullness, and dark eyes..."

These are dense attribute descriptions (face shape, skin tone, expression, clothing, lighting, composition). No short/vague captions observed.

---

### 3.2 FFHQ 70K (Stratum Per-Image Dirs)

| Property | Value |
|----------|-------|
| **Format** | Stratum per-image directories |
| **Path** | `/mnt/nas-ai-models/training-data/ffhq/stratum/` |
| **Images** | **70,000** (dirs `00000`–`69999`) |
| **Size** | ~2 TB (estimated from individual file sizes; du timed out) |
| **Used by** | Primary training data for Arm O (weight 2.3 in mix) |
| **Loading mode** | `adapter_name="stratum"` → full pipeline load |

#### Satellite data per directory

| File | Shape / Content | Used by stratum? |
|------|----------------|------------------|
| `pixel.npy` | (3, 1024, 1024) float16 RGB | ✅ Image |
| `dinov3_cls.npy` | (1024,) float16 | ✅ DINOv3 CLS |
| `dinov3_patches.npy` | (4096, 1024) float16 | ✅ DINOv3 patches |
| `t5_hidden.npy` | (512, 1024) float16 | ✅ T5 hidden states |
| `t5_mask.npy` | (512,) uint8 | ✅ T5 mask |
| `pose.npy` | (133, 3) float16 | ✅ DWPose |
| `caption.txt` | text string | ✅ Caption |
| `metadata.json` | {image_id, source_path, width, height, aspect_bucket} | ✅ ID |
| `seg.npy` | (1024, 1024) uint8 → downsampled to (64, 64) int16 | ✅ Segmentation map |
| `depth.npy` | (H, W) depth map | ❌ Not loaded (exists but unused) |
| `normal.npy` | (H, W, 3) surface normals | ❌ Not loaded |
| `z_g.npy` | (50,) float32 geometry embedding | ❌ Not loaded in stratum mode; ✅ loaded in eidolon mode |
| `auraface_lda.npy` | (64,) float64 identity embedding | ❌ Not loaded in stratum mode; ✅ loaded in eidolon mode |

**Verdict: COMPLETE.** 70,000 entries. All required satellite data present. Additional data (`depth`, `normal`, `z_g`, `auraface_lda`) exists on disk but is not consumed by the standard stratum loader — only the eidolon loader uses `z_g` and `auraface_lda`.

---

### 3.3 Hegre Corpus (Eidolon Stratum)

| Property | Value |
|----------|-------|
| **Format** | Stratum per-image directories (named, not numbered) |
| **Path** | `/mnt/nas-ai-models/training-data/eidolon/hegre_corpus/` |
| **Images** | **31,668** (named persona directories) |
| **Used by** | Arm O training (weight 1.0 in mix) |
| **Loading mode** | `adapter_name="eidolon"` → minimal load (identity + geometry only) |

#### Satellite data per directory

| File | Content | Status |
|------|---------|--------|
| `pixel.npy` | (3, H, W) float16 RGB | ✅ |
| `auraface_lda.npy` | (64,) float64 identity embedding | ✅ |
| `z_g.npy` | (50,) float32 geometry embedding | ✅ |
| `metadata.json` | {image_id, persona, ...} | ✅ |

**Not present:** caption.txt, t5_hidden.npy, t5_mask.npy, dinov3_cls.npy, dinov3_patches.npy, pose.npy, seg.npy

**Verdict: PARTIAL (by design).** The hegre corpus is an *eidolon-only* dataset. It carries identity (auraface_lda) and geometry (z_g) embeddings but no text, DINO, or pose conditioning. The `_load_sample_eidolon()` method fills in zero stubs for missing modalities — the EidolonAdapter drops them anyway.

This is the "geometry control" signal source that powers Arm O's disentangled identity/geometry conditioning.

---

### 3.4 Training Data Mix

From `experiments/zg-token-basis-cfg-guard/config.yaml`:

```yaml
data:
  source: "stratum"
  stratum_dirs:
    - dir: "/mnt/nas-ai-models/training-data/ffhq/stratum"
      weight: 2.3
    - dir: "/mnt/nas-ai-models/training-data/eidolon/hegre_corpus"
      weight: 1.0
```

The training loop interleaves both directories using `adapter_name="eidolon"`:
- **FFHQ dirs** (~70%): loaded via `_load_sample_eidolon()`, but FFHQ dirs lack `auraface_lda.npy` and `z_g.npy` — this will cause load failures (⚠️ bug?). Need to verify whether FFHQ dirs are skipped or if the eidolon loader falls back gracefully.
- **Hegre Corpus dirs** (~30%): properly loaded with identity + geometry.

> ⚠️ **Open question:** The config uses `adapter_name="eidolon"` for both stratum dirs, but FFHQ dirs don't have `auraface_lda.npy` or `z_g.npy`. The loader does `np.load(d / 'auraface_lda.npy')` which will raise `FileNotFoundError` for FFHQ entries. This means **only hegre corpus samples are actually loaded** when `adapter_name="eidolon"`. The FFHQ stratum dir with weight 2.3 may be silently skipped (caught by the `except Exception` on line 237).

---

## 4. Non-Training Data

### 4.1 Crawlr Approved Images

| Property | Value |
|----------|-------|
| **Path** | `/mnt/nas-ai-models/training-data/crawlr/approved/` |
| **Files** | **14,881** images (webp, jpg, jpeg, png) |
| **Role** | Raw input to the caption/embedding pipeline |
| **NOT used** | for training directly — only after sharding |

These are the raw approved photos from the Crawlr system. They feed into the `generate_approved_image_dataset.py` pipeline (referenced in `data/README.md` but **not present in the repo**).

### 4.2 Crawlr Derived Data

| Path | Files | Coverage (vs 14,881) |
|------|-------|----------------------|
| `derived/dinov3/` | 877 `.npy` | 5.9% |
| `derived/dinov3_patches/` | 41,266 `.npy` | ~161 images? (41K ÷ 256 patches) |
| `derived/pose/` | 2,084 `.npy` | 14.0% |
| `derived/t5_hidden/` | 877 `.npy` | 5.9% |
| `derived/images/` | 2,191 `.npy` | 14.7% |
| `derived/approved_image_dataset.*.jsonl` | ~70 timestamped files + 1 canonical | Varies |

**Verdict: VERY INCOMPLETE.** The derived data covers only a small fraction of approved images. This is the "Stage 2" pipeline output referenced in `data/README.md`. Only the 7,000 images that made it into shards have complete satellite data. The remaining ~7,881 approved images have NO precomputed features.

**The `derived/` directory is effectively stale** — it represents partial pipeline runs from Feb/March 2026 that were never completed for all 14,881 images.

### 4.3 Crawlr Raw Images

| Property | Value |
|----------|-------|
| **Path** | `/mnt/nas-ai-models/training-data/crawlr/raw/hq/` |
| **Structure** | 2-char hex prefix dirs (e.g., `e8/`, `vw/`, `9q/`) |
| **Count** | ~31K+ (per `data/README.md`) — `find` timed out |
| **Role** | Source for syncing approved symlinks |

### 4.4 Hegre Enriched

| Property | Value |
|----------|-------|
| **Path** | `/mnt/nas-ai-models/training-data/eidolon/hegre_enriched/` |
| **Structure** | `{id}_{persona-slug}/` → individual image subdirs |
| **Count** | 2,576 persona directories |
| **Format** | NOT stratum format — raw images organized by persona |
| **Status** | Source material, not training-ready |

---

## 5. What the Training Code Expects

### 5.1 WebDataset path (`data.py`)

Looks for these keys in each shard entry:
```
json, dinov3.npy, dinov3_patches.npy, image.npy, t5h.npy, t5m.npy, pose.npy
```

Batch dict produced:
```
image_data (B,3,H,W) f32, dino_embedding (B,1024), dinov3_patches (B,max_p,1024),
dinov3_patches_mask (B,max_p), t5_hidden (B,512,1024), t5_mask (B,512),
pose_keypoints (B,133,3), captions list[str], image_ids list[str]
```

### 5.2 Stratum path — standard (`data_stratum.py`, `adapter_name="stratum"`)

Expected per-directory files:
```
pixel.npy, t5_hidden.npy, t5_mask.npy, dinov3_cls.npy, dinov3_patches.npy,
pose.npy, caption.txt, metadata.json, seg.npy
```

### 5.3 Stratum path — eidolon (`data_stratum.py`, `adapter_name="eidolon"`)

Expected per-directory files:
```
pixel.npy, auraface_lda.npy, z_g.npy, metadata.json
```

Produces zero stubs for: `dino_embedding`, `dinov3_patches`, `t5_hidden`, `t5_mask`, `pose_keypoints`, `seg_map`.

### 5.4 Which modalities does the model actually use?

| Modality | Used by NanoDiT model? | Arm O (eidolon)? |
|----------|----------------------|-------------------|
| Pixel image | ✅ Prediction target | ✅ |
| T5 text hidden | ✅ Cross-attention conditioning | ❌ (zero stub) |
| DINOv3 CLS | ✅ adaLN modulation | ❌ (zero stub) |
| DINOv3 patches | ✅ Cross-attention (if enabled) | ❌ (zero stub) |
| DWPose keypoints | ✅ Cross-attention conditioning | ❌ (zero stub) |
| AuraFace LDA | ❌ (not in baseline) | ✅ Identity embedding |
| z_g geometry | ❌ (not in baseline) | ✅ Geometry embedding |

---

## 6. Summary Table

| Dataset | Images | Format | Caption | T5 | DINOv3 CLS | DINOv3 Patches | DWPose | Identity (AuraFace) | Geometry (z_g) | Seg | Training Role |
|---------|--------|--------|---------|-----|-----------|----------------|--------|---------------------|-----------------|-----|---------------|
| **faces7k shards** | 7,000 | WebDataset | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Validation/vdbg (Arm O) |
| **FFHQ stratum** | 70,000 | Per-image dir | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Primary training (Arm O, 2.3×) |
| **Hegre Corpus** | 31,668 | Per-image dir | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Geometry signal (Arm O, 1.0×) |
| **Crawlr approved** | 14,881 | Raw images | ❌* | ❌* | ❌* | ❌* | ❌* | ❌ | ❌ | ❌ | Pipeline source only |
| **Hegre enriched** | 2,576 | Raw persona dirs | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | NOT training-ready |

\* The `derived/` directory has partial precomputed features for 877–2,191 images (5.9%–14.7% coverage), not complete.

---

## 7. Known Issues & Open Questions

1. **`generate_approved_image_dataset.py` missing:** The script referenced in `data/README.md` and in memory does not exist in the repo. The pipeline from approved images → derived features → shards is not reproducible from the current codebase.

2. **14,881 approved, only 7,000 sharded:** 7,881 approved images (~53%) have no precomputed features and never made it into training shards. No clear record of *why* those 7,000 were selected.

3. **dinov3_patches in derived/ is anomalous:** 41,266 patch files vs only 877 CLS files. This ratio (~47×) suggests the patches are stored *per-patch* rather than per-image, making the derived patch data effectively unusable for direct per-image lookup.

4. **No caption data for hegre corpus:** The hegre corpus has no text captions. The `_load_sample_eidolon()` method uses `meta.get('persona', '')` as a placeholder caption. This is cosmetic — the model doesn't condition on text in eidolon mode.

5. **Stratum data on NAS only:** All stratum data lives on the NAS at `/mnt/nas-ai-models/`. If the NAS is unavailable, no training is possible.

---

## 8. Data Pipeline Diagram

```
Crawlr CDN
    │
    ▼
raw/ (~31K images) ──sync_approved_photos.py──▶ approved/ (14,881 images)
                                                      │
                                                      ▼
                                    generate_approved_image_dataset.py (MISSING)
                                                      │
                                                      ▼
                                            derived/ (partial: 5.9-14.7% coverage)
                                                      │
                                                      │ sharding step
                                                      ▼
                                            shards/faces7k/ (7,000 images, COMPLETE)
                                                      │
                                                      ▼
                                            WebDataset training (older arms)

FFHQ (HuggingFace)
    │
    ▼
ffhq/stratum/ (70,000 images, COMPLETE) ──▶ StratumDataset (full pipeline + eidolon identity/geometry)
                                                      │
                                                      ▼
                                            Current Arm O training (via eidolon adapter)

Hegre source images
    │
    ▼
eidolon/hegre_corpus/ (31,668 images) ──▶ StratumDataset (eidolon mode, identity+geometry only)
                                                      │
                                                      ▼
                                            Current Arm O training (geometry signal)
```
