"""Production training entry point.

Usage:
    python -m production.train_production
    python -m production.train_production --config custom_config.yaml
    python -m production.train_production --resume checkpoints/checkpoint_step010000.pt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import shutil
import json
import subprocess

import torch

from .config_loader import load_config
from .model import NanoDiT
from .data import get_production_dataloader
from .train import ProductionTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Production DiT Training')
    
    parser.add_argument(
        '--config',
        type=str,
        default='production/config.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU index (if multiple GPUs available)'
    )
    
    return parser.parse_args()


def print_gpu_memory(device):
    """Print current GPU memory usage."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def print_config_summary(config):
    """Print configuration summary."""
    print("\n" + "="*60)
    print("PRODUCTION DIT TRAINING")
    print("="*60)
    print("\nModel:")
    print(f"  Layers: {config.model.depth}")
    print(f"  Hidden size: {config.model.hidden_size}")
    print(f"  Heads: {config.model.num_heads}")
    print(f"  Patch size: {config.model.patch_size}")
    print("\nTraining:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Grad accumulation: {config.training.grad_accumulation_steps}")
    print(f"  Effective batch: {config.training.batch_size * config.training.grad_accumulation_steps}")
    print(f"  Total steps: {config.training.total_steps}")
    print(f"  Warmup: {config.training.warmup_steps} steps")
    print(f"  Peak LR: {config.training.optimizer.lr}")
    print(f"  Precision: {config.training.precision}")
    print("\nData:")
    print(f"  Shard dir: {config.data.shard_base_dir}")
    print(f"  Buckets: {len(config.data.buckets)}")
    print(f"  Flip prob: {config.data.horizontal_flip_prob}")
    print("\nCheckpoints:")
    print(f"  Save every: {config.checkpoint.save_every} steps")
    print(f"  Output: {config.checkpoint.output_dir}")
    print("\nValidation:")
    if config.validation.enabled:
        print(f"  Every {config.validation.interval_steps} steps")
        print(f"  Output: {config.validation.output_dir}")
    else:
        print("  Disabled")
    print("="*60 + "\n")


def create_experiment_dir(config_path):
    """Create timestamped experiment directory and save metadata.
    
    Creates: experiments/YYYY-MM-DD_HHMM/
    Saves: config.yaml, metadata.json (git commit, timestamp, command)
    
    Args:
        config_path: Path to config file
    
    Returns:
        experiment_dir: Path to created directory
    """
    # Create timestamp: 2026-02-15_1130
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    
    # Create experiment directory
    exp_dir = Path('experiments') / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config file
    config_dest = exp_dir / 'config.yaml'
    shutil.copy2(config_path, config_dest)
    print(f"Saved config to: {config_dest}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config_path': str(config_path),
        'command': ' '.join(sys.argv),
    }
    
    # Try to get git commit hash
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        metadata['git_commit'] = git_hash
        
        # Check for uncommitted changes
        git_status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        metadata['git_dirty'] = bool(git_status)
        if git_status:
            metadata['warning'] = 'Uncommitted changes present'
    except:
        metadata['git_commit'] = 'unknown'
        metadata['git_dirty'] = False
    
    # Save metadata
    metadata_path = exp_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    return exp_dir


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Create experiment directory and save config
    print("\n" + "="*60)
    print("EXPERIMENT TRACKING")
    print("="*60)
    experiment_dir = create_experiment_dir(args.config)
    print(f"Experiment directory: {experiment_dir}")
    print("="*60 + "\n")
    
    # Setup device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(device)
            print(f"Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    print_config_summary(config)
    
    # Create output directories
    Path(config.checkpoint.output_dir).mkdir(parents=True, exist_ok=True)
    if config.validation.enabled:
        Path(config.validation.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Compute latent statistics if normalization is enabled
    from production.data import USE_LATENT_NORMALIZATION, load_or_compute_latent_stats
    if USE_LATENT_NORMALIZATION:
        print("\n" + "="*60)
        print("VAE LATENT NORMALIZATION")
        print("="*60)
        stats = load_or_compute_latent_stats(
            shard_dir=config.data.shard_base_dir,
            num_samples=1000
        )
        # Update module-level constants
        import production.data as data_module
        data_module.FLUX_LATENT_MEAN = stats['mean']
        data_module.FLUX_LATENT_STD = stats['std']
        print(f"Normalization enabled:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print("="*60 + "\n")
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = get_production_dataloader(config, device)
    print(f"Dataloader created")
    
    # Create model
    print("Creating model...")
    model = NanoDiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
    if config.training.gradient_checkpointing:
        print(f"Gradient checkpointing: ENABLED (trades ~20-30% speed for 3-4x memory savings)")
    
    # Create validation function if enabled
    validate_fn = None
    if config.validation.enabled:
        from .validate import create_validation_fn
        print("Creating validation function...")
        # Note: TensorBoard writer will be passed from trainer after it's created
        validate_fn_factory = lambda tb_writer: create_validation_fn(
            shard_dir=config.data.shard_base_dir,
            output_dir=config.validation.output_dir,
            tensorboard_writer=tb_writer,
        )
    
    # Create visual debugging function (if enabled)
    visual_debug_fn = None
    if config.validation.visual_debug_interval > 0:
        print("Creating visual debug function...")
        from .visual_debug import create_visual_debug_fn
        # Note: TensorBoard writer will be passed from trainer after it's created
        visual_debug_fn_factory = lambda tb_writer: create_visual_debug_fn(
            shard_dir=config.data.shard_base_dir,
            output_dir=config.validation.visual_debug_dir,
            num_samples=config.validation.visual_debug_num_samples,
            text_scale=config.sampling.text_scale,
            dino_scale=config.sampling.dino_scale,
            num_steps=config.sampling.num_steps,
            device=device,
            tensorboard_writer=tb_writer,
        )
    
    # Create trainer
    print("Creating trainer...")
    trainer = ProductionTrainer(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
        experiment_name=experiment_dir.name,  # Pass experiment name for TensorBoard
    )
    
    # Now create actual validation and visual debug functions with TensorBoard writer
    if config.validation.enabled:
        validate_fn = validate_fn_factory(trainer.writer)
    if config.validation.visual_debug_interval > 0:
        visual_debug_fn = visual_debug_fn_factory(trainer.writer)
    
    print_gpu_memory(device)
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...\n")
    try:
        trainer.train(
            validate_fn=validate_fn,
            visual_debug_fn=visual_debug_fn,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Saving checkpoint...")
        trainer.save_checkpoint(
            Path(config.checkpoint.output_dir) / 'checkpoint_interrupt.pt'
        )
        print(f"Checkpoint saved to: {config.checkpoint.output_dir}/checkpoint_interrupt.pt")
        print("You can resume with: --resume checkpoints/checkpoint_interrupt.pt")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    else:
        # Training completed normally
        trainer.save_checkpoint(
            Path(config.checkpoint.output_dir) / 'checkpoint_final.pt'
        )
        print("Training complete!")
    finally:
        print_gpu_memory(device)


if __name__ == "__main__":
    main()
