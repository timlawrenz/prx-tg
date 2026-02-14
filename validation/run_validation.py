"""Main entry point for Nano DiT validation training."""

import argparse
import signal
import sys
import torch

from .model import NanoDiT
from .data import get_validation_dataloader
from .train import Trainer
from .validate import create_validation_fn
from .config import get_default_config


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
    parser = argparse.ArgumentParser(description='Nano DiT Validation Training')
    
    # Paths
    parser.add_argument('--data-dir', type=str, default='data/shards/100',
                        help='Path to validation shards')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--validation-dir', type=str, default='validation',
                        help='Directory for validation outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--total-steps', type=int, default=5000,
                        help='Total training steps')
    parser.add_argument('--peak-lr', type=float, default=3e-4,
                        help='Peak learning rate')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (if multiple GPUs available)')
    
    # Misc
    parser.add_argument('--no-validation', action='store_true',
                        help='Skip validation tests (faster for testing)')
    
    return parser.parse_args()


def print_gpu_memory(device):
    """Print current GPU memory usage."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def main():
    """Main training function."""
    args = parse_args()
    
    # Get default config and override with args
    config = get_default_config()
    config.paths.data_dir = args.data_dir
    config.paths.checkpoint_dir = args.checkpoint_dir
    config.paths.validation_dir = args.validation_dir
    config.training.batch_size = args.batch_size
    config.training.total_steps = args.total_steps
    config.training.peak_lr = args.peak_lr
    
    # Setup device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{args.gpu}')
            print(f"Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    config.device = str(device)
    
    # Print configuration
    print("\n" + "="*60)
    print("NANO DIT VALIDATION TRAINING")
    print("="*60)
    print(f"\nModel:")
    print(f"  Layers: {config.model.depth}")
    print(f"  Hidden size: {config.model.hidden_size}")
    print(f"  Heads: {config.model.num_heads}")
    print(f"\nTraining:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Total steps: {config.training.total_steps}")
    print(f"  Peak LR: {config.training.peak_lr}")
    print(f"  Warmup: {config.training.warmup_steps} steps")
    print(f"\nData:")
    print(f"  Shard dir: {config.paths.data_dir}")
    print(f"\nOutput:")
    print(f"  Checkpoints: {config.paths.checkpoint_dir}")
    print(f"  Validation: {config.paths.validation_dir}")
    print("="*60 + "\n")
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = get_validation_dataloader(
        shard_dir=str(config.paths.data_dir),
        batch_size=config.training.batch_size,
        shuffle=True,
        flip_prob=0.5,
    )
    
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
        dino_dim=config.model.dino_dim,
        text_dim=config.model.text_dim,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    
    # Create validation function (if not disabled)
    if args.no_validation:
        print("Validation disabled")
        validation_fn = None
    else:
        print("Creating validation function...")
        validation_fn = create_validation_fn(
            dataloader,
            output_dir=str(config.paths.validation_dir)
        )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        device=device,
        total_steps=config.training.total_steps,
        warmup_steps=config.training.warmup_steps,
        peak_lr=config.training.peak_lr,
        min_lr=config.training.min_lr,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
        ema_decay=config.training.ema_decay,
        cfg_probs={
            'p_drop_both': config.training.p_drop_both,
            'p_drop_text': config.training.p_drop_text,
            'p_drop_dino': config.training.p_drop_dino,
        },
        checkpoint_every=config.training.checkpoint_every,
        log_every=config.training.log_every,
        checkpoint_dir=str(config.paths.checkpoint_dir),
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Print initial GPU memory
    print_gpu_memory(device)
    
    # Setup graceful interrupt
    interrupt = GracefulInterrupt()
    
    # Override trainer's train method to handle interrupt
    original_train = trainer.train
    
    def train_with_interrupt(validate_fn=None):
        try:
            original_train(validate_fn=validate_fn)
        except KeyboardInterrupt:
            pass
        finally:
            if interrupt.interrupted:
                print("Saving final checkpoint due to interrupt...")
                trainer.save_checkpoint(
                    config.paths.checkpoint_dir / 'checkpoint_interrupted.pt'
                )
    
    trainer.train = train_with_interrupt
    
    # Start training
    print("\nStarting training...\n")
    trainer.train(validate_fn=validation_fn)
    
    # Final GPU memory
    print_gpu_memory(device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
