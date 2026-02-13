## Why

Stage 1 (JSONL with embeddings inline) is complete but not suitable for training at scale. Storing large embeddings directly in JSONL creates three problems: (1) the file becomes unwieldy (~2GB for 60k images), (2) we cannot regenerate individual embedding types without reprocessing everything, and (3) we need additional embeddings (VAE latents, T5 hidden states) that will bloat the file to ~20GB+. Stage 2 moves to a hybrid format with lightweight JSONL metadata and separate .npy files by embedding type, enabling independent regeneration and preparing for Stage 3 WebDataset sharding.

## What Changes

- **Add `image_id` field** to all JSONL records (basename of `image_path` without extension)
- **Extract DINOv3 embeddings** from existing JSONL records and write to `data/derived/dinov3/{image_id}.npy` (1024 floats, float32, ~4KB each)
- **Remove `dinov3_embedding` array** from JSONL records (replaced by `image_id` reference)
- **Generate VAE latents** by encoding images through Flux VAE encoder and write to `data/derived/vae_latents/{image_id}.npy` (16×64×64, float16, ~131KB each)
- **Generate T5 hidden states** by encoding captions through T5-Large and write to `data/derived/t5_hidden/{image_id}.npy` (77×1024, float16, ~158KB each)
- **Add `aspect_bucket` field** to JSONL records (e.g., "1024x1024", "832x1216") based on aspect ratio bucketing
- **Preserve existing fields** (`caption`, `t5_attention_mask`, `height`, `width`) in JSONL without modification
- **Idempotent operation:** Script detects existing .npy files and JSONL fields, skips completed work, allows resumption

## Capabilities

### New Capabilities
- `stage-2-storage`: Defines the hybrid storage format with metadata JSONL and separate embedding directories, including directory structure, file naming conventions, aspect ratio bucketing, and idempotent processing rules.

### Modified Capabilities
- `approved-image-embeddings`: Changes output format from monolithic JSONL to hybrid format. Adds VAE latent encoding, T5 hidden state encoding, and aspect ratio bucketing. Removes inline DINOv3 embedding from JSONL (replaced by external .npy file).

## Impact

**Files Modified:**
- `scripts/generate_approved_image_dataset.py` - Add VAE/T5 encoding, aspect bucketing, .npy file I/O, JSONL format migration

**Data Migration:**
- Existing `data/derived/approved_image_dataset.jsonl` (~2GB, 1444+ records) will be enriched in-place
- Creates three new directories: `data/derived/dinov3/`, `data/derived/vae_latents/`, `data/derived/t5_hidden/`
- Final JSONL size reduces to ~5-10MB (metadata only)
- Total Stage 2 storage: ~17MB (metadata + DINOv3) + ~8GB (VAE) + ~9.5GB (T5) ≈ **18GB for 60k images**

**Dependencies:**
- Requires Flux VAE model (likely `black-forest-labs/FLUX.1-dev` or similar) for latent encoding
- Already has T5-Large tokenizer (from Stage 1) - need full T5-Large model for hidden states
- Torch, transformers, numpy, PIL (already present)

**Backwards Compatibility:**
- **BREAKING:** Code depending on inline `dinov3_embedding` field in JSONL will break
- Migration is one-way (no automatic rollback to Stage 1 format)
- Script must handle partial migration state gracefully (some records migrated, others not)
