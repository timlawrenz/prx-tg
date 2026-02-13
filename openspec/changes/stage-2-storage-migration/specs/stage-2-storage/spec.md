## Purpose

Define the Stage 2 hybrid storage format for image embeddings, combining lightweight JSONL metadata with separate binary files organized by embedding type. This format enables independent regeneration of embedding types, easier debugging, and prepares for Stage 3 WebDataset sharding.

## Requirements

### Requirement: Directory structure organization
The system SHALL organize Stage 2 outputs into a hybrid structure with metadata JSONL and separate directories per embedding type.

#### Scenario: Output directories are created
- **WHEN** Stage 2 processing begins
- **THEN** the system creates `data/derived/dinov3/`, `data/derived/vae_latents/`, and `data/derived/t5_hidden/` directories if they do not exist

#### Scenario: JSONL metadata is stored separately from embeddings
- **WHEN** Stage 2 processing completes
- **THEN** metadata exists in `data/derived/approved_image_dataset.jsonl`
- **AND** binary embeddings exist in separate `.npy` files under type-specific directories

### Requirement: File naming convention
The system SHALL use `image_id` as the base filename for all embedding files, where `image_id` is derived from the original image filename without extension.

#### Scenario: Image ID is derived consistently
- **WHEN** an image path is `data/approved/abc123xyz.jpg`
- **THEN** the `image_id` is `abc123xyz`

#### Scenario: Embedding files use image_id naming
- **WHEN** an image with `image_id = abc123xyz` is processed
- **THEN** DINOv3 embedding is saved as `data/derived/dinov3/abc123xyz.npy`
- **AND** VAE latent is saved as `data/derived/vae_latents/abc123xyz.npy`
- **AND** T5 hidden state is saved as `data/derived/t5_hidden/abc123xyz.npy`

### Requirement: JSONL metadata format
The system SHALL emit JSONL records containing metadata fields but NOT inline embeddings. Each record MUST include `image_id` for cross-referencing binary files.

#### Scenario: Metadata record contains required fields
- **WHEN** a Stage 2 record is written
- **THEN** the record contains `image_id`, `image_path`, `caption`, `t5_attention_mask`, `height`, `width`, `aspect_bucket`, and `format_version`

#### Scenario: Format version identifies Stage 2 records
- **WHEN** a Stage 2 record is written
- **THEN** the `format_version` field equals `2`

#### Scenario: Inline embeddings are removed
- **WHEN** a Stage 2 record is written
- **THEN** the record does NOT contain a `dinov3_embedding` field
- **AND** the record does NOT contain inline arrays for VAE or T5 embeddings

### Requirement: DINOv3 embedding storage
The system SHALL store DINOv3 embeddings as individual `.npy` files with shape (1024,) and dtype float32.

#### Scenario: DINOv3 file has correct shape
- **WHEN** a DINOv3 embedding is saved
- **THEN** loading the `.npy` file yields a NumPy array with shape (1024,)
- **AND** the array dtype is float32

#### Scenario: DINOv3 file size is approximately 4KB
- **WHEN** a DINOv3 embedding is saved
- **THEN** the file size is approximately 4KB (1024 floats × 4 bytes)

### Requirement: VAE latent storage
The system SHALL store VAE latents as individual `.npy` files with shape (16, H//8, W//8) and dtype float16, where H and W are the original image dimensions.

#### Scenario: VAE latent has correct shape
- **WHEN** an image with dimensions 1024×768 is encoded
- **THEN** the VAE latent file has shape (16, 128, 96)
- **AND** the array dtype is float16

#### Scenario: VAE latent file size is approximately correct
- **WHEN** a 1024×1024 image is encoded
- **THEN** the VAE latent file size is approximately 131KB (16 × 64 × 64 × 2 bytes)

### Requirement: T5 hidden state storage
The system SHALL store T5 hidden states as individual `.npy` files with shape (77, 1024) and dtype float16.

#### Scenario: T5 hidden state has correct shape
- **WHEN** a caption is encoded with T5-Large
- **THEN** the hidden state file has shape (77, 1024)
- **AND** the array dtype is float16

#### Scenario: T5 hidden state file size is approximately 158KB
- **WHEN** a T5 hidden state is saved
- **THEN** the file size is approximately 158KB (77 × 1024 × 2 bytes)

### Requirement: Aspect ratio bucketing
The system SHALL assign each image to one of seven predefined aspect ratio buckets and record the assignment in the JSONL metadata.

#### Scenario: Bucket assignment is based on aspect ratio
- **WHEN** an image has aspect ratio (width/height) closest to 1.0
- **THEN** the `aspect_bucket` field is set to `"1024x1024"`

#### Scenario: All images are assigned to a bucket
- **WHEN** any image is processed
- **THEN** the record contains an `aspect_bucket` field with one of: `"1024x1024"`, `"832x1216"`, `"1216x832"`, `"768x1280"`, `"1280x768"`, `"704x1344"`, or `"1344x704"`

#### Scenario: Bucket dimensions are divisible by 64
- **WHEN** an image is assigned to any aspect bucket
- **THEN** both width and height of the bucket are divisible by 64

### Requirement: Idempotent processing
The system SHALL detect existing `.npy` files and skip regeneration, allowing safe resumption after interruption.

#### Scenario: Existing embedding file is not regenerated
- **WHEN** a `.npy` file already exists for an embedding type and image_id
- **THEN** the system skips regenerating that embedding
- **AND** the existing file is preserved unchanged

#### Scenario: Partial completion can be resumed
- **WHEN** processing is interrupted and some images have all embeddings while others have none
- **THEN** rerunning the system completes only the missing embeddings
- **AND** existing embeddings are not recomputed

### Requirement: Atomic JSONL updates
The system SHALL write JSONL updates to a temporary file and perform atomic merges to prevent data loss during interruption.

#### Scenario: Updates are written to temporary file first
- **WHEN** Stage 2 processing is in progress
- **THEN** new records are appended to a `.tmp` file
- **AND** the main JSONL file is not modified until final merge

#### Scenario: Graceful shutdown merges temporary file
- **WHEN** the process receives SIGINT (Ctrl+C) during Stage 2 processing
- **THEN** the system merges the `.tmp` file into the main JSONL
- **AND** the `.tmp` file is deleted
- **AND** all work up to the interruption point is preserved

### Requirement: Storage efficiency
The system SHALL reduce JSONL file size by removing inline embeddings while maintaining total storage within reasonable bounds.

#### Scenario: JSONL size is reduced significantly
- **WHEN** Stage 2 migration completes for existing Stage 1 data
- **THEN** the JSONL file size is reduced by at least 90%

#### Scenario: Total Stage 2 storage is predictable
- **WHEN** 60,000 images are processed through Stage 2
- **THEN** total storage is approximately 18GB (240MB DINOv3 + 8GB VAE + 9.5GB T5 + 10MB JSONL)
