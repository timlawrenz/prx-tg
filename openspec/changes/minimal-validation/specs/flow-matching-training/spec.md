## Purpose

Implement rectified flow training loop with logit-normal timestep sampling, EMA tracking, and independent CFG dropout for dual conditioning validation.

## ADDED Requirements

### Requirement: Rectified flow velocity prediction loss
The system SHALL train the model to predict velocity v = (z_0 - z_1) / (t_1 - t_0) where z_0 is clean latent, z_1 is noise, and t is uniformly sampled timestamp.

#### Scenario: Loss is MSE between predicted and target velocity
- **WHEN** a training step computes loss
- **THEN** loss = MSE(v_pred, v_target) where v_target = z_0 - z_t

#### Scenario: Velocity is normalized by timestep
- **WHEN** computing target velocity
- **THEN** v_target accounts for the interpolation path from z_t to z_0

### Requirement: Logit-normal timestep sampling
The system SHALL sample timesteps t from logit-normal distribution with mean=0.0 and std=1.0, then clamp to [0.001, 0.999].

#### Scenario: Timesteps are concentrated in middle range
- **WHEN** 1000 timesteps are sampled
- **THEN** median is near 0.5 and distribution density is higher around 0.3-0.7 than at extremes

#### Scenario: Extreme timesteps are avoided
- **WHEN** timesteps are sampled
- **THEN** no timestep is exactly 0.0 or 1.0 (clamped to [0.001, 0.999])

### Requirement: Independent CFG dropout during training
The system SHALL randomly drop conditioning signals with configurable probabilities: p_drop_both=0.1, p_drop_text=0.1, p_drop_dino=0.1, p_keep_both=0.7.

#### Scenario: Unconditional samples are 10% of batches
- **WHEN** training for 1000 steps
- **THEN** approximately 100 steps use null embeddings for both DINOv3 and T5

#### Scenario: Text-only samples are 10% of batches
- **WHEN** training for 1000 steps
- **THEN** approximately 100 steps use null DINOv3 but real T5

#### Scenario: DINO-only samples are 10% of batches
- **WHEN** training for 1000 steps
- **THEN** approximately 100 steps use real DINOv3 but null T5

#### Scenario: Fully conditioned samples are 70% of batches
- **WHEN** training for 1000 steps
- **THEN** approximately 700 steps use both real DINOv3 and T5

### Requirement: EMA model tracking with warmup
The system SHALL maintain an exponential moving average of model weights with decay warming up from 0.0 to 0.9999 over first 5000 steps.

#### Scenario: EMA decay starts at 0.0
- **WHEN** training begins at step 0
- **THEN** EMA decay is 0.0 (EMA weights = current weights)

#### Scenario: EMA decay increases linearly
- **WHEN** training reaches step 2500
- **THEN** EMA decay is approximately 0.5

#### Scenario: EMA decay reaches target at 5000 steps
- **WHEN** training reaches step 5000
- **THEN** EMA decay is 0.9999

#### Scenario: EMA decay stays at target after warmup
- **WHEN** training reaches step 10000
- **THEN** EMA decay is still 0.9999

### Requirement: AdamW optimizer with weight decay
The system SHALL use AdamW optimizer with betas=(0.9, 0.95), weight_decay=0.03, epsilon=1e-8.

#### Scenario: Optimizer is AdamW
- **WHEN** the optimizer is instantiated
- **THEN** it is torch.optim.AdamW with specified hyperparameters

#### Scenario: Weight decay is applied
- **WHEN** optimizer step is executed
- **THEN** weight decay penalty is applied to parameters

### Requirement: Learning rate schedule with warmup and cosine decay
The system SHALL use peak learning rate 3e-4 with linear warmup over first 5000 steps, followed by cosine decay to 1e-6.

#### Scenario: Learning rate starts near zero
- **WHEN** training begins at step 0
- **THEN** learning rate is near 0 (warmup start)

#### Scenario: Learning rate reaches peak after warmup
- **WHEN** training reaches step 5000
- **THEN** learning rate is 3e-4

#### Scenario: Learning rate decays after warmup
- **WHEN** training reaches step 7500 (halfway through decay)
- **THEN** learning rate is between 3e-4 and 1e-6

#### Scenario: Learning rate reaches minimum
- **WHEN** training reaches final step
- **THEN** learning rate is 1e-6

### Requirement: Gradient clipping
The system SHALL clip gradient norm to max value of 1.0 to prevent instability.

#### Scenario: Gradients are clipped
- **WHEN** gradient norm exceeds 1.0
- **THEN** gradients are scaled to have norm = 1.0

#### Scenario: Small gradients are not affected
- **WHEN** gradient norm is 0.5
- **THEN** gradients are not modified

### Requirement: Checkpoint saving
The system SHALL save model checkpoint (both training and EMA weights) every 1000 steps and at the end of training.

#### Scenario: Checkpoint saved every 1000 steps
- **WHEN** training reaches step 1000, 2000, 3000, etc.
- **THEN** a checkpoint file is written with both model and EMA weights

#### Scenario: Final checkpoint saved
- **WHEN** training completes at step 5000
- **THEN** a final checkpoint is written

#### Scenario: Checkpoint includes optimizer state
- **WHEN** a checkpoint is saved
- **THEN** it includes model weights, EMA weights, optimizer state, and step number

### Requirement: Progress logging
The system SHALL log training metrics every 100 steps including loss, learning rate, gradient norm, and step time.

#### Scenario: Metrics logged every 100 steps
- **WHEN** training reaches step 100, 200, 300, etc.
- **THEN** metrics are printed to stderr

#### Scenario: Loss is logged
- **WHEN** metrics are logged
- **THEN** current step loss value is included

#### Scenario: Learning rate is logged
- **WHEN** metrics are logged
- **THEN** current learning rate is included
