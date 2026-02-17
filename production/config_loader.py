"""Configuration loading and validation for production training.

Loads YAML config and provides structured access to all hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    depth: int = 12
    hidden_size: int = 384
    num_heads: int = 6
    patch_size: int = 2
    mlp_ratio: float = 4.0
    in_channels: int = 16
    input_size: int = 64


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "AdamW"
    lr: float = 3e-4
    min_lr: float = 1e-6
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.03
    eps: float = 1e-8


@dataclass
class CFGDropoutConfig:
    """CFG dropout probabilities (mutually exclusive)."""
    p_uncond: float = 0.1        # Drop both
    p_text_only: float = 0.1     # Drop DINO only
    p_dino_only: float = 0.1     # Drop text only
    
    def to_dict(self):
        """Convert to dict for compatibility with training code."""
        return {
            'p_drop_both': self.p_uncond,
            'p_drop_text': self.p_dino_only,  # Note: swap! text dropout means drop text
            'p_drop_dino': self.p_text_only,  # dino dropout means drop dino
        }


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    total_steps: int = 50000
    warmup_steps: int = 5000
    batch_size: int = 8
    grad_accumulation_steps: int = 32
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 5000
    
    cfg_dropout: CFGDropoutConfig = field(default_factory=CFGDropoutConfig)
    
    timestep_sampling: Literal["uniform", "logit_normal"] = "logit_normal"
    logit_normal_loc: float = 0.0
    logit_normal_scale: float = 1.0
    
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    precision: Literal["float32", "bfloat16"] = "bfloat16"


@dataclass
class DataConfig:
    """Data loading configuration."""
    shard_base_dir: str = "data/shards"
    buckets: List[str] = field(default_factory=lambda: [
        "1024x1024", "832x1216", "1216x832", "768x1280", "1280x768"
    ])
    bucket_sampling: Literal["proportional", "uniform"] = "proportional"
    
    horizontal_flip_prob: float = 0.5
    swap_caption_lr: bool = True
    
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class SamplingConfig:
    """Sampling configuration for validation."""
    num_steps: int = 50
    text_scale: float = 3.0
    dino_scale: float = 2.0


@dataclass
class ValidationConfig:
    """Validation configuration."""
    enabled: bool = True
    interval_steps: int = 5000
    num_samples: int = 100
    output_dir: str = "validation_outputs"
    
    run_reconstruction: bool = True
    run_dino_swap: bool = True
    run_text_manip: bool = True
    
    visual_debug_interval: int = 1000
    visual_debug_num_samples: int = 4
    visual_debug_dir: str = "visual_debug"


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    save_every: int = 5000
    keep_last_n: int = 3
    output_dir: str = "checkpoints"
    save_optimizer: bool = True


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    log_every: int = 50
    log_file: str = "training.log"
    
    monitor_velocity_norm: bool = True
    monitor_grad_norm: bool = True
    velocity_norm_warning: float = 10.0
    grad_norm_warning: float = 10.0


@dataclass
class PathConfig:
    """Path configuration."""
    vae_path: str = "models/vae.safetensors"
    t5_path: str = "models/t5xxl_fp16.safetensors"


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object with all settings
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Recursively build dataclass instances
    def build_dataclass(cls, data):
        if data is None:
            return cls()
        
        # Get field types
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        
        for key, value in data.items():
            if key in field_types:
                field_type = field_types[key]
                # Check if field type is a dataclass
                if hasattr(field_type, '__dataclass_fields__'):
                    kwargs[key] = build_dataclass(field_type, value)
                else:
                    kwargs[key] = value
        
        return cls(**kwargs)
    
    return build_dataclass(Config, config_dict)


def save_config(config: Config, config_path: str | Path):
    """Save configuration to YAML file.
    
    Args:
        config: Config object
        config_path: Path to save YAML file
    """
    import dataclasses
    
    def dataclass_to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {
                key: dataclass_to_dict(value)
                for key, value in dataclasses.asdict(obj).items()
            }
        return obj
    
    config_dict = dataclass_to_dict(config)
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Test config loading
    config = load_config("config.yaml")
    print("Config loaded successfully!")
    print(f"Model: {config.model.depth} layers, {config.model.hidden_size} hidden")
    print(f"Training: {config.training.total_steps} steps, lr={config.training.optimizer.lr}")
    print(f"Data buckets: {config.data.buckets}")
