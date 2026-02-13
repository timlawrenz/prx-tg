## Context

The existing `scripts/generate_approved_image_dataset.py` generates JSONL files with three fields: `image_path`, `dinov3_embedding`, and `caption`. Stage 2 of the data pipeline (per `the-plan.md`) requires additional metadata: `t5_attention_mask`, `height`, and `width`. 

**Current state:**
- Script uses a sidecar `.idx` file to track completed images and skip reprocessing
- DINOv3 embeddings (~5 sec/image on GPU) and Gemma captions (~30 sec/image on GPU) are expensive
- Users have already generated datasets with thousands of entries

**Problem:**
- Adding new required fields would force users to regenerate entire datasets (hours/days of GPU time)
- No mechanism to add fields to existing records without recomputing embeddings/captions

**Constraints:**
- Must preserve backward compatibility with existing JSONL files
- T5 tokenizer and PIL are already available in the environment
- Metadata computation is fast (~0.01 sec/image) compared to embeddings/captions

## Goals / Non-Goals

**Goals:**
- Make script fully idempotent: running twice produces same result
- Automatically detect and enrich incomplete records with missing fields
- Skip expensive operations (DINOv3, Gemma) when data already exists
- Add three new fields: `t5_attention_mask`, `height`, `width`
- Update progress reporting to show enrichment status

**Non-Goals:**
- Changing the JSONL format structure (remains one JSON object per line)
- Modifying existing fields (preserve `image_path`, `dinov3_embedding`, `caption` exactly as-is)
- Parallel processing or distributed execution (single-process is sufficient)
- Creating separate Stage 2 `.npy` files (that's a future script)

## Decisions

### Decision 1: Two-pass architecture (read existing → write enriched)

**Chosen approach:** Read entire existing JSONL into memory as a dict `{image_path: record}`, enrich in-place, write back atomically.

**Alternatives considered:**
- **Stream rewrite (read line, enrich, write line):** Risky if script crashes mid-write (corrupts output file). Would need temp file + atomic rename.
- **Separate enrichment script:** More code to maintain, confusing UX (two commands for same dataset).

**Rationale:** 
- 60k records × ~2KB each = 120MB in memory (trivial on modern systems)
- Atomic write via temp file prevents corruption
- Simpler logic: all records in memory, easy to detect missing fields

**Implementation:**
```python
# Load existing records
existing_records = {}
if output_path.exists():
    with output_path.open("r") as f:
        for line in f:
            record = json.loads(line)
            existing_records[record["image_path"]] = record

# Process each image
for image_path in iter_images(input_dir):
    if image_path in existing_records:
        record = existing_records[image_path]
        # Enrich with missing fields
    else:
        # Generate full record (DINOv3 + Gemma + metadata)
        record = {...}
    
    all_records.append(record)

# Atomic write
with temp_file.open("w") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")
temp_file.rename(output_path)
```

### Decision 2: Field presence detection

**Chosen approach:** Check if field exists and is non-empty (not `None`, not empty list).

**Logic:**
```python
needs_dino = "dinov3_embedding" not in record or not record["dinov3_embedding"]
needs_caption = "caption" not in record or not record["caption"]
needs_t5_mask = "t5_attention_mask" not in record or record["t5_attention_mask"] is None
needs_dimensions = "height" not in record or "width" not in record
```

**Rationale:** Handles both missing keys (old format) and explicitly null values (failed computation).

### Decision 3: T5 tokenizer initialization

**Chosen approach:** Load T5-Large tokenizer once at startup, reuse for all captions.

**Model:** `t5-large` (standard T5 used in diffusion models, not T5-XXL or other variants)

**Tokenization:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-large")
tokens = tokenizer(
    caption,
    max_length=77,           # Standard CLIP/T5 sequence length for diffusion
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
attention_mask = tokens["attention_mask"][0].tolist()  # Convert to list for JSON
```

**Rationale:** 
- T5-Large is fast (~10ms per caption)
- 77 tokens is diffusion standard (matches CLIP, Stable Diffusion conventions)
- Attention mask is small (~308 bytes as JSON array)

### Decision 4: Image dimension reading

**Chosen approach:** Use PIL to read image headers (no full decode).

**Implementation:**
```python
from PIL import Image

with Image.open(image_path) as img:
    width, height = img.size
```

**Rationale:** 
- PIL header reading is instant (<1ms per image)
- No need to decode full image into memory
- Already have PIL dependency

### Decision 5: Remove sidecar `.idx` file

**Chosen approach:** Deprecate the `.idx` file. Use the JSONL itself as the source of truth for completed images.

**Rationale:**
- With two-pass architecture, we load the full JSONL anyway
- `.idx` becomes redundant and could get out of sync
- Simpler: one file to track instead of two

**Migration:** Existing `.idx` files are ignored (no breaking change).

### Decision 6: Progress reporting granularity

**Chosen approach:** Three status categories with counts:
- `processed_new`: Full pipeline (DINOv3 + Gemma + metadata)
- `enriched`: Added metadata only (image already had embeddings/caption)
- `skipped`: All fields already present

**Output format:**
```
Processing: 150 new, 4800 enriched, 100 skipped (total: 5050/5500)
```

**Rationale:** Users need to understand what work is happening (expensive vs cheap operations).

## Risks / Trade-offs

### Risk: Memory usage for large datasets
**Scenario:** 100k images × 2KB/record = 200MB in memory  
**Mitigation:** Acceptable for modern systems. If this becomes an issue (500k+ images), switch to streaming architecture with temp file.

### Risk: Corrupted JSONL files
**Scenario:** Script crashes mid-write, output file is truncated  
**Mitigation:** Use atomic write (write to `.tmp` file, rename on success). Original file preserved if crash occurs.

### Risk: T5 tokenizer version mismatch
**Scenario:** Different `transformers` versions produce different attention masks  
**Mitigation:** Pin `transformers>=4.50.0` in requirements. Document T5 version in output (future: add metadata header).

### Trade-off: Reading entire JSONL into memory
**Benefit:** Simple, atomic writes, easy field detection  
**Cost:** ~200MB RAM for 100k images (negligible)  
**Decision:** Accept the cost for simplicity

### Trade-off: In-place enrichment vs separate output
**Benefit:** Users don't need to manage multiple dataset versions  
**Cost:** Can't compare before/after easily  
**Decision:** In-place is preferred. Users can backup manually if needed.

## Migration Plan

**For existing users:**
1. Backup existing JSONL: `cp data/derived/approved_image_dataset.jsonl data/derived/approved_image_dataset.jsonl.bak`
2. Run script normally: `python scripts/generate_approved_image_dataset.py --output data/derived/approved_image_dataset.jsonl`
3. Script automatically:
   - Loads existing records
   - Detects missing fields (`t5_attention_mask`, `height`, `width`)
   - Enriches records without recomputing embeddings/captions
   - Writes back atomically
4. Verify: Check that DINOv3/caption fields are unchanged, new fields added

**Rollback:** Restore from backup if issues occur.

**For new users:** Script generates all 6 fields on first run (no difference in behavior).

## Open Questions

None - design is fully specified.
