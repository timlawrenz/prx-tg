## Context

Stage 1 generated a monolithic JSONL file (`data/derived/approved_image_dataset.jsonl`) containing inline DINOv3 embeddings (1024 floats each). This works for ~1,444 records but doesn't scale to 60k images. Adding VAE latents (16×64×64) and T5 hidden states (77×1024) would bloat the file to ~20GB and make it unmaintainable.

Stage 2 migrates to a hybrid format: lightweight JSONL metadata + separate .npy files organized by embedding type. This enables independent regeneration, easier debugging, and prepares for Stage 3 WebDataset sharding.

**Current State:**
- Existing script: `scripts/generate_approved_image_dataset.py` (Stage 1 complete)
- Existing data: `data/derived/approved_image_dataset.jsonl` (~2GB, 1444+ records)
- Fields per record: `image_path`, `dinov3_embedding` (inline array), `caption`, `t5_attention_mask`, `height`, `width`

**Constraints:**
- Must not lose existing DINOv3 embeddings or captions (expensive to regenerate)
- Must handle partial migration gracefully (script crashes, Ctrl+C)
- ROCm environment (AMD GPU) - some models may have compatibility issues
- Disk space: ~18GB needed for Stage 2 outputs

## Goals / Non-Goals

**Goals:**
- Migrate existing JSONL to Stage 2 format without data loss
- Generate VAE latents for all images using Flux VAE encoder
- Generate T5 hidden states for all captions
- Extract DINOv3 embeddings to separate .npy files
- Add aspect ratio bucketing (record bucket assignment in JSONL)
- Idempotent operation (skip existing .npy files, resume from crash)
- Shrink JSONL from ~2GB to ~5-10MB

**Non-Goals:**
- Stage 3 WebDataset sharding (separate change)
- Training code (out of scope)
- Dataset normalization statistics (compute after Stage 2 complete)
- Validation set split (do later)
- Image resizing to bucket dimensions (Stage 3 preprocessing)

## Decisions

### Decision 1: Three-Pass Architecture (Optional)

**Choice:** Support processing in three independent passes: (1) DINOv3 extraction, (2) VAE latent generation, (3) T5 hidden state generation. Default behavior loads all models at once.

**Rationale:**
- **Default (no --pass flag)**: Load all models simultaneously for maximum speed (~30-40GB VRAM)
- **Separate passes (--pass flag)**: Allow low-VRAM systems (8-24GB) to process one stage at a time
- Each pass is idempotent (checks for existing .npy files)
- Clear separation of concerns for debugging

**VRAM Requirements:**
- **All models at once**: ~30-40GB VRAM (DINOv3 ~8GB + Gemma 27B ~20GB + FLUX VAE ~4GB + T5-Large ~8GB)
- **DINOv3 pass only**: ~10-15GB VRAM
- **VAE pass only**: ~3-4GB VRAM
- **T5 pass only**: ~5-8GB VRAM

**Alternatives Considered:**
- Single pass only: Would exclude users with <40GB VRAM
- On-demand model loading per record: Too slow, constant model load/unload overhead

### Decision 2: Aspect Ratio Bucketing Strategy

**Choice:** Define 7 predefined buckets at 1024px equivalent area with 64-pixel modulus:
- 1024×1024 (square, ratio 1.0)
- 832×1216 (portrait, ratio ~0.68)
- 1216×832 (landscape, ratio ~1.46)
- 768×1280 (tall portrait, ratio 0.6)
- 1280×768 (wide landscape, ratio ~1.67)
- 704×1344 (very tall, ratio ~0.52)
- 1344×704 (very wide, ratio ~1.91)

Assign each image to the bucket with the closest aspect ratio. Store bucket assignment in `aspect_bucket` field (e.g., "1024x1024").

**Rationale:**
- 1024px baseline matches Flux training resolution
- All dimensions divisible by 64 (VAE downsampling factor of 8 × transformer patch size of 8)
- Covers common photo aspect ratios (3:2, 4:3, 16:9, portrait/landscape)
- 7 buckets balance flexibility vs batch dimension variety

**Alternatives Considered:**
- Dynamic bucketing (arbitrary resolutions): Complicates Stage 3, harder to debug
- Crop to square: Destroys composition, violates "high quality" requirement
- Single resolution (1024×1024): Would require padding/cropping, wastes capacity

### Decision 3: VAE Latent Encoding

**Choice:** Use Flux VAE encoder (`black-forest-labs/FLUX.1-dev` VAE component) to encode images to latents. Save as float16 to `data/derived/vae_latents/{image_id}.npy`.

**Format:** Shape (16, H//8, W//8) where H,W are original image dimensions (not bucket dimensions).

**Rationale:**
- Flux VAE is the target model's encoder (must match training architecture)
- Encode at original resolution (resizing happens in Stage 3 dataloader)
- float16 saves 50% disk space vs float32 with negligible precision loss
- 16 channels × 64×64 spatial × 2 bytes = 131KB per 1024×1024 image

**Alternatives Considered:**
- SD VAE: Wrong architecture (only 4 channels, different latent space)
- Encode at bucket resolution now: Premature optimization, loses flexibility
- float32: 2× disk space for no benefit (VAE latents aren't that precise)

### Decision 4: T5 Hidden State Generation

**Choice:** Load full T5-Large encoder (not just tokenizer). Encode captions to `last_hidden_state`, save as float16 to `data/derived/t5_hidden/{image_id}.npy`.

**Format:** Shape (77, 1024) - 77 tokens (CLIP/SD convention), 1024-dim T5-Large hidden size.

**Rationale:**
- DiT models consume full hidden state sequence (not just pooled embedding)
- Already have captions and attention masks from Stage 1 (reuse them)
- float16 reduces size from 316KB to 158KB per image
- 77 tokens × 1024-dim matches SD/CLIP convention

**Alternatives Considered:**
- T5-Base (smaller): Less expressive (768-dim vs 1024-dim), not worth the savings
- T5-XXL (larger): Marginal quality gain, 4× memory, slower encoding
- Recompute on-the-fly during training: 100× slower, defeats preprocessing purpose

### Decision 5: File Naming and Directory Structure

**Choice:** Use `image_id` as filename base (derived from `image_path` basename without extension).

Directory structure:
```
data/derived/
  approved_image_dataset.jsonl           # Metadata only (~5-10MB)
  dinov3/
    o2jdtq9uz05whadn9jc8y4mz46xg.npy    # (1024,) float32
  vae_latents/
    o2jdtq9uz05whadn9jc8y4mz46xg.npy    # (16, H//8, W//8) float16
  t5_hidden/
    o2jdtq9uz05whadn9jc8y4mz46xg.npy    # (77, 1024) float16
```

**Rationale:**
- `image_id` is unique, stable identifier (based on filename)
- Flat directory structure (no subdirectories) - simpler, ~60k files per dir is manageable
- Extension-less filenames in source → consistent image_id format
- `.npy` extension for binary files (NumPy standard)

**Alternatives Considered:**
- Nested directories (hash-based sharding): Premature optimization, complicates lookup
- HDF5/Zarr single file: Random access still slow, harder to regenerate individual embeddings
- Keep embeddings inline: Doesn't scale, defeats purpose of Stage 2

### Decision 6: JSONL Format Migration Strategy

**Choice:** Enrichment approach with format version field:
1. Add `image_id` field (basename without extension)
2. Add `aspect_bucket` field (e.g., "1024x1024")
3. Add `format_version` field (value: 2)
4. Remove `dinov3_embedding` array field
5. Keep all other fields unchanged (`caption`, `t5_attention_mask`, `height`, `width`)

**Migration Logic:**
- Load existing JSONL into memory (dict keyed by image_path)
- For each record:
  - If `format_version == 2` and all .npy files exist → skip
  - Otherwise: enrich with missing fields, write .npy files as needed
- Write enriched records to `.tmp` file during processing
- Final: atomic merge of .tmp into main JSONL

**Rationale:**
- Reuses existing enrichment infrastructure from Stage 1 (incremental saves, Ctrl+C handling)
- `format_version` field enables future migrations
- Removing inline embedding reduces JSONL size by ~95%
- Preserves expensive data (captions, existing embeddings)

**Alternatives Considered:**
- Fresh regeneration: Wastes compute (1444 existing captions = ~2 hours of Gemma inference)
- Keep both inline and external embeddings: Redundant, wastes disk space
- Separate Stage 1/Stage 2 files: Confusing, hard to track canonical version

### Decision 7: Idempotency and Resumption

**Choice:** Check for existing .npy files before generating embeddings. Use file existence as completion signal.

**Logic per record:**
```python
needs_dinov3 = not (dinov3_dir / f"{image_id}.npy").exists()
needs_vae = not (vae_dir / f"{image_id}.npy").exists()
needs_t5 = not (t5_dir / f"{image_id}.npy").exists()
needs_migration = record.get("format_version") != 2
```

Only load models and process if any needs_* flag is True.

**Rationale:**
- File existence check is fast (no need for separate index file)
- Supports partial migration (process subset of images)
- Safe for crashes/Ctrl+C (work is preserved in .npy files)
- Aligns with Stage 1 enrichment pattern

**Alternatives Considered:**
- Separate completion index file (.idx): Redundant with filesystem metadata
- Database tracking: Overkill for batch processing
- Always regenerate: Defeats idempotency goal

### Decision 8: Batch Processing and Progress Tracking

**Choice:** Process in batches of 4 images (reuse existing batch size). Track four counters:
- `migrated_new`: Records converted to format v2 + all embeddings generated
- `enriched`: Records with partial embeddings filled in
- `extracted`: Records where only DINOv3 extraction happened (already had format v2)
- `skipped`: Records fully complete (format v2 + all .npy files exist)

Display: `progress: X migrated, Y enriched, Z extracted, W skipped (total: N) rate=R/s`

**Rationale:**
- Batch size 4 matches Stage 1 (good GPU utilization without OOM)
- Four counters give clear visibility into what work was done
- Rate calculation helps estimate completion time
- Consistent with existing progress reporting pattern

## Risks / Trade-offs

### Risk: Flux VAE Model Compatibility with ROCm
**Issue:** Flux models are newer, may have ROCm-specific bugs or missing kernels.

**Mitigation:**
- Test VAE encoding on single image before batch processing
- Have fallback to SD3 VAE if Flux VAE fails (different latent space, but better than nothing)
- Log clear error messages with model name and error details

### Risk: 18GB Disk Space Exhaustion
**Issue:** Stage 2 outputs consume ~18GB. Disk might fill during processing.

**Mitigation:**
- Check available disk space before starting (abort if < 20GB free)
- Process in chunks (e.g., 10k images at a time) if space is tight
- Provide `--output-base-dir` flag to write to different disk if needed

### Risk: Partial Migration State Confusion
**Issue:** Script crashes midway → some records v1, some v2 → unclear what's safe to delete.

**Mitigation:**
- Never delete original JSONL until migration 100% complete
- Use `format_version` field to identify migration state
- Provide `--verify` flag to scan JSONL and check .npy files consistency

### Risk: T5-Large Model Memory Footprint
**Issue:** T5-Large encoder is ~3GB model, might cause OOM with other models loaded.

**Mitigation:**
- Use three-pass architecture (only one model loaded at a time)
- Explicit `del model; torch.cuda.empty_cache()` between passes
- Allow `--passes` flag to run specific passes independently if needed

### Trade-off: External Files vs Inline
**Benefit:** 95% smaller JSONL, independent regeneration, faster iteration
**Cost:** More files (180k .npy files for 60k images), slightly more complex loading
**Assessment:** Worth it - filesystem handles 180k files fine, Stage 3 consolidates into tars

### Trade-off: Float16 Precision Loss
**Benefit:** 50% disk savings (~9GB saved)
**Cost:** Minor precision loss in VAE latents and T5 hidden states
**Assessment:** Acceptable - neural network training is noise-tolerant, float16 is standard for diffusion models

## Migration Plan

**Phase 1: Validation (Manual)**
1. Test on 10 sample images
2. Verify .npy files are correctly shaped and not NaN/Inf
3. Verify JSONL format v2 records have correct fields

**Phase 2: DINOv3 Extraction Pass (Safe)**
1. Run script with `--pass dinov3` (extract existing embeddings)
2. This pass cannot fail (data already exists in JSONL)
3. Creates `data/derived/dinov3/` directory

**Phase 3: VAE Latent Generation (Slow)**
1. Run script with `--pass vae`
2. Expected time: ~60k images × 0.5s = 8-10 hours
3. Creates `data/derived/vae_latents/` directory
4. Checkpoint: Verify 5-10 random .npy files are valid

**Phase 4: T5 Hidden State Generation (Moderate)**
1. Run script with `--pass t5`
2. Expected time: ~60k images × 0.2s = 3-4 hours
3. Creates `data/derived/t5_hidden/` directory

**Phase 5: JSONL Migration (Fast)**
1. Run script without pass restriction (or `--pass migrate`)
2. Reads all records, adds `image_id`, `aspect_bucket`, `format_version`
3. Removes `dinov3_embedding` inline array
4. Atomic merge to final JSONL

**Rollback Strategy:**
- Keep original JSONL as `approved_image_dataset.jsonl.stage1.backup`
- If migration fails, restore backup and delete .npy directories
- If Stage 2 proves problematic, can regenerate inline format from .npy files

## Open Questions

1. **Flux VAE model exact name?** Is it `black-forest-labs/FLUX.1-dev` VAE component, or separate model ID?
   - **Resolution needed before implementation**

2. **VAE latent normalization now or later?** Do we compute dataset mean/std in this change, or separate analysis step?
   - **Recommendation:** Separate step after Stage 2 complete (need all latents to compute stats)

3. **Aspect bucket validation?** Should script verify bucket assignments are reasonable (no extreme outliers)?
   - **Recommendation:** Log warning for aspect ratios outside [0.4, 2.5] range, but don't fail

4. **Concurrent processing?** Should we use multiprocessing for I/O-bound tasks (loading images)?
   - **Recommendation:** Start single-threaded, add parallelism if too slow (GPT handles model inference)
