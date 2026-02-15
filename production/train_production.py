"""Production training entry point.

Usage:
    python -m production.train_production
    python -m production.train_production --config custom_config.yaml
    python -m production.train_production --resume checkpoints/checkpoint_step010000.pt
"""

import argparse
import signal
import sys
from pathlib import Path

import torch

from .config_loader import load_config
from .model import NanoDiT
from .data import get_production_dataloader
from .train import ProductionTrainer


class GracefulInterrupt:
    """Handle Ctrl+C gracefully."""
    
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)
    
    def handle_interrupt(self, signum, frame):
        print("\n\nInterrupt received, saving checkpoint...")
        self.interrupted = True


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


def main():
    """Main training function."""
    args = parse_args()
    interrupt_handler = GracefulInterrupt()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
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
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
    
    # Create validation function if enabled
    validate_fn = None
    if config.validation.enabled:
        from .validate import create_validation_fn
        print("Creating validation function...")
        validate_fn = create_validation_fn(
            shard_dir=config.data.shard_base_dir,
            output_dir=config.validation.output_dir,
        )
    
    # Create trainer
    print("Creating trainer...")
    trainer = ProductionTrainer(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
    )
    
    print_gpu_memory(device)
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...\n")
    try:
        trainer.train(validate_fn=validate_fn)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        # Final checkpoint
        if not interrupt_handler.interrupted:
            trainer.save_checkpoint(
                Path(config.checkpoint.output_dir) / 'checkpoint_final.pt'
            )
            print("Training complete!")
        
        print_gpu_memory(device)


if __name__ == "__main__":
    main()
