## Why

The current dataset generation script (`scripts/generate_approved_image_dataset.py`) produces JSONL files with `image_path`, `dinov3_embedding`, and `caption` fields. However, the updated data pipeline (Stage 2 in `the-plan.md`) requires additional metadata fields: `t5_attention_mask`, `height`, and `width`. These fields are needed for T5 text encoding and aspect ratio bucketing in downstream processing. Since many captions have already been generated (expensive Gemma-3-27B inference), we must enrich the existing dataset without regenerating DINOv3 embeddings or captions.

## What Changes

- Make the script **idempotent by default**: automatically detect existing JSONL entries and enrich them with missing fields
- When output file exists, read existing entries and:
  - Preserve all existing fields (`image_path`, `dinov3_embedding`, `caption`)
  - Add missing fields if not present
  - Skip re-computation of expensive operations (DINOv3, Gemma captioning)
- Compute and add three new fields to each JSONL entry:
  - `t5_attention_mask`: Array of 1s/0s indicating valid token positions (computed by tokenizing the caption with T5)
  - `height`: Image height in pixels (read from image file)
  - `width`: Image width in pixels (read from image file)
- Skip entries that already have all required fields (fully idempotent)
- Update progress reporting to distinguish "processing new", "enriching existing", and "skipping complete"
- The `--no-resume` flag will clear the output and regenerate everything (breaking change: now more aggressive)

## Capabilities

### New Capabilities

None - this enhances existing functionality.

### Modified Capabilities

- `approved-image-embeddings`: Adds metadata fields (`t5_attention_mask`, `height`, `width`) to the JSONL output format. Adds enrichment mode to support adding fields to existing datasets without regenerating expensive embeddings.

## Impact

**Code:**
- `scripts/generate_approved_image_dataset.py`: Add enrichment mode logic, T5 tokenizer loading, image dimension reading
- `scripts/requirements-approved-image-embeddings.txt`: May need to ensure T5 tokenizer is available (already included via `transformers`)

**Data:**
- `data/derived/approved_image_dataset.jsonl`: Existing file will be read, enriched with new fields, and rewritten (backup recommended)
- JSONL format changes from 3 fields to 6 fields (backward compatible - old entries work, just missing metadata)

**Dependencies:**
- Requires T5 tokenizer (already available via `transformers>=4.50.0`)
- Requires `PIL` for reading image dimensions (already installed)

**Workflow:**
- Users with existing datasets run the same command: `python scripts/generate_approved_image_dataset.py --output data/derived/approved_image_dataset.jsonl` (automatically enriches existing entries)
- New datasets generated with the same command will include all 6 fields from the start
- Use `--no-resume` to force full regeneration (deletes output and starts fresh)
