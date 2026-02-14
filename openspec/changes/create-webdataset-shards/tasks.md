## 1. Script Setup

- [x] 1.1 Create `scripts/create_webdataset_shards.py` with shebang and docstring
- [x] 1.2 Add argument parser with all CLI flags (input-jsonl, derived-dir, output-dir, limit, shard-size, shuffle, seed, bucket, overwrite, dry-run, progress-every)
- [x] 1.3 Add Counters dataclass for tracking (total_records, ready_records, skipped_incomplete, written_samples, written_shards)
- [x] 1.4 Add helper functions (eprint, is_mask_ok, shard_name)

## 2. Core Reading Logic

- [x] 2.1 Implement iter_ready_records() generator to scan JSONL and validate readiness
- [x] 2.2 Add validation for caption (non-empty string), t5_attention_mask (77-length list of 0/1), image_id, aspect_bucket
- [x] 2.3 Add filesystem checks for dinov3/*.npy, vae_latents/*.npy, t5_hidden/*.npy
- [x] 2.4 Add bucket filtering logic when --bucket is specified
- [x] 2.5 Add progress reporting during JSONL scan (every N ready records)

## 3. Shard Writing Logic

- [x] 3.1 Implement add_bytes() helper to add bytes to tarfile with TarInfo
- [x] 3.2 Implement add_file() helper to add .npy files to tarfile from Path
- [x] 3.3 Implement write_shards() function that creates bucket directories and tar files
- [x] 3.4 Add JSON metadata serialization ({image_id}.json with required fields)
- [x] 3.5 Add T5 attention mask conversion to uint8 .npy ({image_id}.t5m.npy)
- [x] 3.6 Add copying of existing .npy files (dinov3, vae, t5_hidden) with proper arcnames
- [x] 3.7 Add shard-size chunking logic (split samples into shards of max N samples)
- [x] 3.8 Add zero-padded shard naming (shard-000000.tar, shard-000001.tar, etc.)

## 4. Processing Pipeline

- [x] 4.1 Implement main() function with input validation (check JSONL exists, check derived subdirs exist)
- [x] 4.2 Add loading of all ready records into RAM via list(iter_ready_records())
- [x] 4.3 Add shuffle logic using random.Random(seed) when --shuffle is specified
- [x] 4.4 Add limit logic to slice ready list to first N samples when --limit > 0
- [x] 4.5 Add grouping logic (organize ready samples by aspect_bucket into dict)
- [x] 4.6 Add per-bucket shard writing loop
- [x] 4.7 Add final summary reporting (total_records, ready_records, skipped_incomplete, written_samples, written_shards)

## 5. Error Handling & Safety

- [x] 5.1 Add FileExistsError check for existing shard files (unless --overwrite)
- [x] 5.2 Add graceful handling of malformed JSONL lines (skip with warning)
- [x] 5.3 Add dry-run mode that increments counters without writing files
- [x] 5.4 Add input validation error messages (missing JSONL, missing derived dirs)

## 6. Documentation & Testing

- [x] 6.1 Update `scripts/README.md` with create_webdataset_shards.py section
- [x] 6.2 Add usage examples (dry-run, validation shard, full training shards)
- [x] 6.3 Add smoke test command (--limit 2 --dry-run)
- [x] 6.4 Add .gitignore entry for `data/shards/` (already covered by data/ entry)
- [x] 6.5 Document output structure (bucket_{WxH}/shard-NNNNNN.tar) and tar entry format

## 7. Validation

- [x] 7.1 Run smoke test: `python3 scripts/create_webdataset_shards.py --limit 2 --dry-run`
- [x] 7.2 Create 100-sample validation shard with --shuffle --limit 100
- [x] 7.3 Verify tar contents with `tar -tf shard-000000.tar | head -n 20`
- [x] 7.4 Verify .npy files load correctly from tar (extract and np.load)
- [x] 7.5 Verify JSON metadata contains required fields (image_id, aspect_bucket, caption, etc.)
