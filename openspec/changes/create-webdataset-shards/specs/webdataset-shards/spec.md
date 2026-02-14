## Purpose

Package Stage 2 derived embeddings (metadata JSONL + separate .npy files) into WebDataset tar shards grouped by aspect ratio bucket for efficient sequential streaming during training.

## ADDED Requirements

### Requirement: Load Stage 2 metadata and validate readiness
The system SHALL read the Stage 2 metadata JSONL file and identify "ready" samples where all required components exist: caption, t5_attention_mask (77-length list of 0s and 1s), image_id, aspect_bucket, and corresponding .npy files (dinov3, vae_latents, t5_hidden).

#### Scenario: Complete sample is identified as ready
- **WHEN** a JSONL record has caption, t5_attention_mask (77 valid ints), image_id, aspect_bucket, and all three .npy files exist
- **THEN** the system includes this sample in the ready set

#### Scenario: Incomplete sample is skipped
- **WHEN** a JSONL record is missing caption, has invalid mask, or any .npy file is missing
- **THEN** the system skips this sample and increments skipped_incomplete counter

#### Scenario: Malformed JSONL line is skipped gracefully
- **WHEN** a JSONL line cannot be parsed as JSON
- **THEN** the system logs a warning, increments skipped_incomplete counter, and continues processing

### Requirement: Group samples by aspect ratio bucket
The system SHALL organize ready samples by their aspect_bucket field value before creating shards.

#### Scenario: Samples are grouped by bucket
- **WHEN** ready samples have aspect_bucket values "1024x1024" and "832x1216"
- **THEN** the system creates two separate groups, one per bucket

#### Scenario: Bucket filter restricts output
- **WHEN** user specifies --bucket 1024x1024 and ready samples exist in multiple buckets
- **THEN** only samples with aspect_bucket="1024x1024" are included in output shards

### Requirement: Create tar shards with WebDataset naming convention
The system SHALL write tar files containing WebDataset entries where each sample has five files: {image_id}.json (metadata), {image_id}.dinov3.npy, {image_id}.vae.npy, {image_id}.t5h.npy (T5 hidden states), {image_id}.t5m.npy (T5 attention mask).

#### Scenario: Tar entry contains all five files
- **WHEN** a sample with image_id="abc123" is written to a shard
- **THEN** the tar contains abc123.json, abc123.dinov3.npy, abc123.vae.npy, abc123.t5h.npy, abc123.t5m.npy

#### Scenario: JSON metadata includes required fields
- **WHEN** the .json file for a sample is written
- **THEN** it contains at minimum: image_id, aspect_bucket, caption, image_path, height, width

#### Scenario: T5 attention mask is saved as .npy
- **WHEN** the t5_attention_mask from JSONL (77-length list) is written
- **THEN** it is saved as {image_id}.t5m.npy in uint8 format

### Requirement: Organize shards by bucket in output directory
The system SHALL write shards to {output_dir}/bucket_{aspect_bucket}/shard-NNNNNN.tar where NNNNNN is a zero-padded shard index starting from 000000.

#### Scenario: Bucket directory is created
- **WHEN** writing shards for aspect_bucket="832x1216"
- **THEN** the system creates directory {output_dir}/bucket_832x1216/

#### Scenario: Shard files use zero-padded naming
- **WHEN** writing the first, second, and 100th shard
- **THEN** filenames are shard-000000.tar, shard-000001.tar, shard-000099.tar

### Requirement: Limit samples per shard
The system SHALL write at most N samples per tar shard where N is configurable (default 1000). When a bucket has more than N samples, the system SHALL create multiple shards.

#### Scenario: Single shard for small bucket
- **WHEN** a bucket has 800 ready samples and shard_size=1000
- **THEN** the system writes one shard containing all 800 samples

#### Scenario: Multiple shards for large bucket
- **WHEN** a bucket has 2500 ready samples and shard_size=1000
- **THEN** the system writes three shards: shard-000000.tar (1000), shard-000001.tar (1000), shard-000002.tar (500)

### Requirement: Support sample count limit
The system SHALL support a --limit flag to write at most N total ready samples across all buckets, useful for creating small validation datasets.

#### Scenario: Limit restricts total output
- **WHEN** 5000 ready samples exist and --limit 100 is specified
- **THEN** the system writes shards containing exactly 100 samples total

#### Scenario: Limit applies after shuffle
- **WHEN** --shuffle and --limit 100 are both specified
- **THEN** the system shuffles all ready samples first, then selects the first 100

### Requirement: Support shuffle mode
The system SHALL support a --shuffle flag that randomly reorders ready samples before applying limits and sharding, using a configurable random seed for reproducibility.

#### Scenario: Shuffle produces different order
- **WHEN** --shuffle is specified
- **THEN** the order of samples in shards differs from the JSONL order

#### Scenario: Seed ensures reproducibility
- **WHEN** --shuffle --seed 42 is run twice
- **THEN** both runs produce identical sample order

### Requirement: Prevent accidental overwrite of existing shards
The system SHALL refuse to overwrite existing shard tar files unless --overwrite flag is provided.

#### Scenario: Existing shard blocks write
- **WHEN** a shard tar file already exists and --overwrite is not specified
- **THEN** the system raises FileExistsError and stops

#### Scenario: Overwrite flag allows replacement
- **WHEN** a shard tar file already exists and --overwrite is specified
- **THEN** the system deletes the old file and writes the new shard

### Requirement: Support dry-run mode
The system SHALL support a --dry-run flag that simulates shard creation without writing files, reporting what would be written.

#### Scenario: Dry run does not write files
- **WHEN** --dry-run is specified
- **THEN** the system reports written_samples and written_shards counters without creating tar files

#### Scenario: Dry run validates inputs
- **WHEN** --dry-run is specified and input JSONL is missing
- **THEN** the system reports error before attempting to scan samples

### Requirement: Progress reporting
The system SHALL report progress periodically, including counts of total_records scanned, ready_records found, and skipped_incomplete records.

#### Scenario: Progress during scan
- **WHEN** scanning JSONL with progress_every=500
- **THEN** the system prints progress every 500 ready records found

#### Scenario: Final summary
- **WHEN** processing completes
- **THEN** the system prints final counts: total_records, ready_records, skipped_incomplete, written_samples, written_shards

### Requirement: Preserve existing .npy files verbatim
The system SHALL copy .npy files (dinov3, vae_latents, t5_hidden) into tar shards without modification or re-encoding.

#### Scenario: NPY files are copied as-is
- **WHEN** a .npy file contains float16 VAE latents with shape (16, 128, 96)
- **THEN** the .npy file in the tar shard has identical bytes and loads with the same shape and dtype
