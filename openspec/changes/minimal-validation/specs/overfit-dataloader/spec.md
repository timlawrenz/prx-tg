## Purpose

Load 100-image validation dataset from WebDataset shards with 512x512 resizing for fast overfitting experiments.

## ADDED Requirements

### Requirement: Load samples from WebDataset validation shards
The system SHALL read tar files from `data/shards/validation/bucket_*/shard-*.tar` and extract samples with all five components (json, dinov3.npy, vae.npy, t5h.npy, t5m.npy).

#### Scenario: Samples are loaded from all buckets
- **WHEN** the dataloader is initialized with validation shards directory
- **THEN** it discovers and loads samples from all bucket subdirectories (1024x1024, 832x1216, 768x1280, 1216x832)

#### Scenario: Each sample contains required components
- **WHEN** a sample is yielded from the dataloader
- **THEN** it contains keys: vae_latent (tensor), t5_hidden (tensor), t5_mask (tensor), dino_embedding (tensor), caption (string), image_id (string)

### Requirement: Resize VAE latents to 512x512 equivalent
The system SHALL resize VAE latents to match 512×512 input resolution (64×64 latent space) regardless of original aspect ratio, using bilinear interpolation.

#### Scenario: Square image resized to 64x64 latent
- **WHEN** a 1024×1024 image VAE latent (16×128×128) is loaded
- **THEN** it is resized to (16×64×64) using bilinear interpolation

#### Scenario: Portrait image resized to 64x64 latent
- **WHEN** an 832×1216 image VAE latent (16×104×152) is loaded
- **THEN** it is resized to (16×64×64) using bilinear interpolation

#### Scenario: Landscape image resized to 64x64 latent
- **WHEN** a 1216×832 image VAE latent (16×152×104) is loaded
- **THEN** it is resized to (16×64×64) using bilinear interpolation

### Requirement: Apply horizontal flip augmentation
The system SHALL randomly flip images horizontally with 50% probability and correspondingly swap "left" ↔ "right" in captions.

#### Scenario: Flip is applied with probability 0.5
- **WHEN** the dataloader is configured with flip augmentation
- **THEN** approximately 50% of samples are horizontally flipped

#### Scenario: Caption is updated when flipped
- **WHEN** a sample with caption "woman looking left" is flipped
- **THEN** the caption becomes "woman looking right"

#### Scenario: Non-spatial captions are unchanged
- **WHEN** a sample with caption "woman in red dress" is flipped
- **THEN** the caption remains "woman in red dress"

### Requirement: Batch samples with consistent shapes
The system SHALL batch samples such that all VAE latents in a batch have identical spatial dimensions (64×64 after resizing).

#### Scenario: Batch has uniform latent shapes
- **WHEN** a batch of size 8 is created
- **THEN** all 8 VAE latents have shape (16, 64, 64)

#### Scenario: T5 sequences have uniform length
- **WHEN** a batch of size 8 is created
- **THEN** all 8 T5 hidden state tensors have shape (77, 1024)

### Requirement: Shuffle samples for diversity
The system SHALL shuffle the 100-sample dataset between epochs to ensure diverse mini-batch composition.

#### Scenario: Order changes between epochs
- **WHEN** two consecutive epochs are iterated
- **THEN** the sample order differs between epochs

#### Scenario: All samples appear in each epoch
- **WHEN** one full epoch is completed
- **THEN** all 100 samples have been yielded exactly once

### Requirement: Convert numpy arrays to PyTorch tensors
The system SHALL convert all .npy arrays (VAE latents, DINOv3, T5 hidden, T5 mask) to PyTorch tensors with correct dtype.

#### Scenario: VAE latent is float32 tensor
- **WHEN** a VAE latent .npy (float16) is loaded
- **THEN** it is converted to torch.float32 tensor

#### Scenario: DINOv3 embedding is float32 tensor
- **WHEN** a DINOv3 .npy (float32) is loaded
- **THEN** it is converted to torch.float32 tensor

#### Scenario: T5 hidden states are float32 tensor
- **WHEN** T5 hidden .npy (float16) is loaded
- **THEN** it is converted to torch.float32 tensor

#### Scenario: T5 mask is boolean tensor
- **WHEN** T5 mask .npy (uint8) is loaded
- **THEN** it is converted to torch.bool tensor
