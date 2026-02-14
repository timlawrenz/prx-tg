## Context

The training pipeline requires packaging Stage 2 embeddings (metadata JSONL + separate .npy files across three directories) into WebDataset tar shards. The current structure has ~60k+ individual files causing severe filesystem I/O bottlenecks during training due to random seeks and inode lookups. WebDataset is the industry-standard solution (used by LAION, Stability AI) that enables sequential streaming (10-100× faster), batched loading, and cloud-native distribution.

Current State:
- Stage 2 output: `data/derived/approved_image_dataset.jsonl` (metadata) + `dinov3/*.npy` + `vae_latents/*.npy` + `t5_hidden/*.npy`
- 1732 complete samples ready, 32k+ total images in pipeline
- Part B (Minimal Validation) requires creating a 100-sample shard for testing

Constraints:
- Must preserve .npy files verbatim (float16/float32 precision critical)
- Must maintain aspect bucket grouping (training requires uniform batch dimensions)
- Must support creating both small validation sets (100-500 samples) and full training sets (60k+ samples)

## Goals / Non-Goals

**Goals:**
- Package Stage 2 embeddings into WebDataset tar format grouped by aspect bucket
- Support configurable shard size (default 1000 samples/shard) and sample limits
- Preserve all embedding data without modification or re-encoding
- Enable deterministic output with optional shuffling for validation set diversity
- Provide clear progress reporting and dry-run mode for verification

**Non-Goals:**
- Training dataloader implementation (handled separately by PyTorch/WebDataset library)
- Image resizing or aspect bucket assignment (already done in Stage 2)
- Embedding regeneration or validation (Stage 2 responsibility)
- Distributed sharding across multiple machines (single-machine script sufficient)

## Decisions

### Decision: Use Python tarfile module instead of WebDataset library for creation
**Rationale:** The WebDataset library is optimized for *reading* shards during training. For *creating* shards, Python's built-in `tarfile` module provides direct control over tar format and avoids an unnecessary dependency during data preprocessing. WebDataset has specific naming conventions ({key}.{ext}) that we can easily follow with tarfile.addfile().

**Alternatives considered:**
- Use WebDataset's `ShardWriter`: Adds dependency for preprocessing, less control over tar format
- Use system `tar` command: Requires shell calls, harder to handle edge cases, less portable

**Trade-off:** Slightly more verbose code (manual tarfile API calls) but zero additional dependencies and full control.

### Decision: Load all ready samples into RAM before sharding
**Rationale:** Enables --shuffle mode and --limit without multiple passes. With 60k samples × ~300 bytes metadata each = ~18MB RAM, this is negligible compared to model training requirements (40GB+ VRAM).

**Alternatives considered:**
- Stream JSONL and write shards in single pass: Cannot shuffle, harder to apply limits across buckets
- Two-pass approach (scan then write): Extra I/O, more complex

**Trade-off:** Slight memory overhead (~20-50MB for full dataset) but simpler code and enables shuffle/limit features.

### Decision: Group by aspect bucket, then shard within each bucket
**Rationale:** Training requires batches with uniform dimensions. Grouping by bucket ensures a dataloader can stream a single bucket's shards sequentially without needing to handle mixed resolutions. This matches the structure in Part C of the-plan.md.

**Alternatives considered:**
- Single global shard sequence with mixed buckets: Training dataloader would need complex filtering/bucketing logic
- Separate JSONL per bucket: More files to manage, doesn't solve I/O bottleneck

**Trade-off:** More directories (one per bucket) but training code is simpler and faster.

### Decision: Store T5 attention mask as separate .npy file instead of inline JSON
**Rationale:** WebDataset convention uses separate files per data type. Storing the 77-int mask as .npy enables zero-copy loading during training and matches the pattern for other embeddings (dinov3.npy, vae.npy, t5h.npy).

**Alternatives considered:**
- Keep mask in JSON: Training dataloader needs to deserialize JSON + convert list to numpy, slower
- Inline mask into t5h.npy: Breaks shape assumptions, requires custom loading logic

**Trade-off:** One additional file per sample (77 bytes as uint8), but consistent pattern and faster training loads.

### Decision: Use zero-padded shard naming (shard-000000.tar)
**Rationale:** Ensures lexicographic sort matches numerical order for stable glob patterns in training code (e.g., `glob('bucket_*/shard-*.tar')`). Six digits supports up to 1M shards per bucket (1B samples @ 1000/shard).

**Alternatives considered:**
- Plain numbers (shard-0.tar, shard-1.tar): Breaks lexicographic sort (shard-10.tar < shard-2.tar)
- UUID-based names: Harder to determine shard count, not human-readable

**Trade-off:** Slightly longer filenames, but standard pattern across ML datasets.

### Decision: Default shard size of 1000 samples
**Rationale:** Balances file handle efficiency (fewer file opens during training) with granularity for shuffling. At ~300KB per sample (metadata + embeddings), 1000 samples = ~300MB tar file, which is manageable for cloud storage and local buffering.

**Alternatives considered:**
- 100 samples/shard: Too many file handles during training (600 shards for 60k samples)
- 10k samples/shard: Large tars (3GB+), harder to shuffle, slower random access for validation

**Trade-off:** Sweet spot for most use cases; configurable via --shard-size for edge cases.

## Risks / Trade-offs

**[Risk: T5 mask conversion uint8 overflow]**
→ **Mitigation:** Masks are strictly 0 or 1 (validated in Stage 2 spec), uint8 is sufficient. Script validates mask values before conversion.

**[Risk: Tar file corruption during write]**
→ **Mitigation:** Write to .tmp suffix, rename on success. If interrupted, incomplete .tmp files are ignored. Use --overwrite to regenerate corrupted shards.

**[Risk: Large buckets cause memory pressure during shuffle]**
→ **Mitigation:** With 60k samples @ ~300 bytes metadata = 18MB RAM, not a concern. If dataset grows 10×, still <200MB. Document this assumption in script help text.

**[Risk: User overwrites existing shards accidentally]**
→ **Mitigation:** Default behavior refuses to overwrite. Requires explicit --overwrite flag. Dry-run mode allows verification before real run.

**[Risk: WebDataset library version incompatibility during training]**
→ **Mitigation:** Tar format is stable (POSIX standard). WebDataset documentation confirms compatibility across versions. Pin webdataset version in training requirements.txt.

**[Trade-off: No compression]**
→ Tar files are uncompressed for faster sequential reads during training. Storage cost: ~90GB for 60k samples. Compression (gzip/zstd) would reduce to ~60GB but add decompression overhead during training. Training speed is higher priority.

**[Trade-off: No incremental updates]**
→ Once shards are created, adding new samples requires regenerating affected shards. This is acceptable for batch workflows (Stage 2 finishes before Stage 3). Real-time updates not required.

## Migration Plan

**Initial Deployment:**
1. Run script with --dry-run --limit 100 to verify Stage 2 readiness
2. Create validation shard: `--limit 100 --shuffle --output-dir data/shards/validation`
3. Verify tar contents: `tar -tf data/shards/validation/bucket_1024x1024/shard-000000.tar | head`
4. Test with WebDataset loader (minimal Python snippet)
5. Create full training shards: `--output-dir data/shards/train`

**Rollback Strategy:**
- Shards are pure artifacts (no state changes to Stage 2 data)
- Delete `data/shards/` directory to roll back
- Re-run script with different parameters if needed

**Data Validation:**
- Compare sample counts: `tar -tf shard-*.tar | grep '\.json$' | wc -l` should match ready_records from Stage 2
- Spot-check .npy files load correctly: `np.load(tarfile.extractfile('abc123.vae.npy'))`

## Open Questions

None - design is complete and ready for implementation.
