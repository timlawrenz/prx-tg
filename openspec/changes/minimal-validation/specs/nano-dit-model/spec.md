## Purpose

Define a minimal DiT architecture for validation testing before full-scale training. The Nano DiT proves dual conditioning (DINOv3 + T5) and flow matching work correctly on a 100-image overfit dataset.

## ADDED Requirements

### Requirement: Model architecture parameters
The system SHALL implement a Diffusion Transformer with 12 layers, 384 hidden dimensions, and 6 attention heads, designed for fast overfitting validation on 100 images at 512x512 resolution.

#### Scenario: Model size is appropriate for validation
- **WHEN** the Nano DiT model is instantiated
- **THEN** it has approximately 30-50M parameters (small enough for fast training, large enough to memorize 100 images)

#### Scenario: Architecture supports dual conditioning
- **WHEN** the model forward pass is called with DINOv3 and T5 embeddings
- **THEN** both conditioning signals are incorporated without errors

### Requirement: DINOv3 conditioning via adaLN-Zero
The system SHALL project DINOv3 embeddings (1024-dim) to adaLN-Zero modulation parameters (scale and shift) that modulate LayerNorm in each transformer block.

#### Scenario: DINOv3 projection layer exists
- **WHEN** the model is initialized
- **THEN** a projection layer from 1024 → (2 × hidden_dim) exists for adaLN modulation

#### Scenario: adaLN-Zero initialized to zero
- **WHEN** the model is initialized
- **THEN** the adaLN modulation projection weights and biases are zero (ensures identity function at step 0)

#### Scenario: DINOv3 modulates all transformer blocks
- **WHEN** a forward pass is executed with a DINOv3 embedding
- **THEN** each of the 12 transformer blocks receives scale and shift parameters derived from the DINOv3 embedding

### Requirement: T5 conditioning via cross-attention
The system SHALL incorporate T5 hidden states (77 × 1024) as keys and values in cross-attention layers within each transformer block.

#### Scenario: Cross-attention layers exist
- **WHEN** the model is initialized
- **THEN** each transformer block contains a cross-attention layer with num_heads=6

#### Scenario: T5 attention mask is applied
- **WHEN** a forward pass is executed with T5 hidden states and attention mask
- **THEN** the cross-attention correctly masks padded tokens (mask value 0)

#### Scenario: Self-attention and cross-attention are distinct
- **WHEN** examining a transformer block
- **THEN** self-attention operates on latent features, cross-attention operates on T5 keys/values

### Requirement: Patch embedding and positional encoding
The system SHALL embed VAE latents (16 channels × 64 × 64 at 512x512 input) into patches with 2D sinusoidal positional embeddings.

#### Scenario: Patch size supports 512x512 resolution
- **WHEN** input is 16 × 64 × 64 VAE latent
- **THEN** patch embedding produces sequence length appropriate for 64×64 spatial grid with patch_size=2 (resulting in 32×32 = 1024 patches)

#### Scenario: Positional embeddings are 2D sinusoidal
- **WHEN** the model computes positional embeddings for a 64×64 latent grid
- **THEN** embeddings use fixed 2D sinusoidal encoding (not learnable, generalizes across resolutions)

### Requirement: Flow matching velocity prediction
The system SHALL output a velocity vector v that predicts the direction from noisy latent z_t toward clean latent z_0, compatible with rectified flow training.

#### Scenario: Output matches input latent shape
- **WHEN** the model forward pass receives a latent of shape (B, 16, H, W)
- **THEN** the output velocity v has shape (B, 16, H, W)

#### Scenario: No activation function on output
- **WHEN** the model is defined
- **THEN** the final output projection has no activation (linear output for velocity prediction)

### Requirement: Independent classifier-free guidance dropout
The system SHALL support independent dropout of DINOv3 and T5 conditioning during training with configurable probabilities.

#### Scenario: Unconditional mode (both dropped)
- **WHEN** forward pass is called with cfg_drop_both=True
- **THEN** model uses null embeddings for both DINOv3 and T5

#### Scenario: Text-only mode (DINO dropped)
- **WHEN** forward pass is called with cfg_drop_dino=True
- **THEN** model uses null DINOv3 embedding but real T5 embedding

#### Scenario: DINO-only mode (text dropped)
- **WHEN** forward pass is called with cfg_drop_text=True
- **THEN** model uses real DINOv3 embedding but null T5 embedding

#### Scenario: Fully conditioned mode
- **WHEN** forward pass is called with no dropout flags
- **THEN** model uses both real DINOv3 and T5 embeddings
