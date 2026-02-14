"""Configuration for Nano DiT validation training."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_size: int = 64  # Latent spatial size (64x64 for 512x512 images)
    patch_size: int = 2
    in_channels: int = 16  # Flux VAE latent channels
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dino_dim: int = 1024
    text_dim: int = 1024


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    batch_size: int = 8
    total_steps: int = 10000
    warmup_steps: int = 2000  # Warm up for first 20% of training
    peak_lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.03
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    
    # CFG dropout probabilities
    p_drop_both: float = 0.1
    p_drop_text: float = 0.1
    p_drop_dino: float = 0.1
    
    # Logging and checkpointing
    checkpoint_every: int = 1000
    log_every: int = 50


@dataclass
class SamplingConfig:
    """Sampling configuration for validation."""
    num_steps: int = 50
    text_scale: float = 3.0
    dino_scale: float = 2.0
    lpips_net: str = 'alex'


@dataclass
class PathsConfig:
    """File paths configuration."""
    data_dir: str = 'data/shards/100'  # 100-sample validation shards
    output_dir: str = 'output'
    checkpoint_dir: str = 'checkpoints'
    validation_dir: str = 'validation'
    
    def __post_init__(self):
        # Convert strings to Path objects
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.validation_dir = Path(self.validation_dir)


@dataclass
class ValidationConfig:
    """Validation test configuration."""
    frequency: int = 1000  # Run validation every N steps
    
    # Test sample indices (defined in validate.py)
    # These are just placeholders; actual indices are in validate.py
    num_reconstruction_samples: int = 100
    num_dino_swap_pairs: int = 5
    num_text_manip_cases: int = 5


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = None
    training: TrainingConfig = None
    sampling: SamplingConfig = None
    paths: PathsConfig = None
    validation: ValidationConfig = None
    
    # Device
    device: str = 'cuda'
    
    def __post_init__(self):
        # Initialize sub-configs if not provided
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.sampling is None:
            self.sampling = SamplingConfig()
        if self.paths is None:
            self.paths = PathsConfig()
        if self.validation is None:
            self.validation = ValidationConfig()


def get_default_config():
    """Get default configuration."""
    return Config()


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    
    print("Model Config:")
    print(f"  Layers: {config.model.depth}")
    print(f"  Hidden size: {config.model.hidden_size}")
    print(f"  Attention heads: {config.model.num_heads}")
    
    print("\nTraining Config:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Total steps: {config.training.total_steps}")
    print(f"  Peak LR: {config.training.peak_lr}")
    
    print("\nSampling Config:")
    print(f"  Steps: {config.sampling.num_steps}")
    print(f"  Text scale: {config.sampling.text_scale}")
    print(f"  DINO scale: {config.sampling.dino_scale}")
    
    print("\nPaths:")
    print(f"  Data: {config.paths.data_dir}")
    print(f"  Checkpoints: {config.paths.checkpoint_dir}")
    print(f"  Validation: {config.paths.validation_dir}")
