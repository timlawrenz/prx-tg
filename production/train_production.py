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


def create_experiment_dir(config_path, resume_path=None):
    """Create timestamped experiment directory and save metadata, or reuse existing.
    
    If resume_path is provided (e.g., experiments/2026-02-22_1200/checkpoints/checkpoint.pt),
    it will reuse the existing 'experiments/2026-02-22_1200' directory.
    
    Creates: experiments/YYYY-MM-DD_HHMM/
    Saves: config.yaml, metadata.json (git commit, timestamp, command)
    
    Args:
        config_path: Path to config file
        resume_path: Optional path to checkpoint being resumed
    
    Returns:
        experiment_dir: Path to created/reused directory
    """
    if resume_path:
        # Expected resume_path: experiments/2026-02-22_1200/checkpoints/checkpoint.pt
        # Get the parent directory of the 'checkpoints' directory
        try:
            exp_dir = Path(resume_path).parent.parent
            if exp_dir.name.startswith('202') and exp_dir.parent.name == 'experiments':
                print(f"Resuming existing experiment: {exp_dir}")
                
                # Append resume info to metadata
                metadata_path = exp_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if 'resumes' not in metadata:
                        metadata['resumes'] = []
                    
                    metadata['resumes'].append({
                        'timestamp': datetime.now().isoformat(),
                        'checkpoint': str(resume_path),
                        'command': ' '.join(sys.argv)
                    })
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                return exp_dir
        except Exception as e:
            print(f"Warning: Could not resolve experiment dir from resume path: {e}")
            print("Falling back to creating new experiment directory.")
    
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
    experiment_dir = create_experiment_dir(args.config, args.resume)
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
    
    # Override config paths to use experiment directory
    config.checkpoint.output_dir = str(experiment_dir / 'checkpoints')
    config.validation.output_dir = str(experiment_dir / 'validation')
    config.validation.visual_debug_dir = str(experiment_dir / 'visual_debug')
    
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
    
    # Resolve REPA block index
    repa_block_idx = None
    if config.training.repa.enabled:
        idx = config.training.repa.block_index
        repa_block_idx = config.model.depth // 2 if idx < 0 else idx
        print(f"REPA enabled: alignment at block {repa_block_idx}, weight {config.training.repa.weight}")
    
    # Resolve TREAD route range
    tread_route_start = None
    tread_route_end = None
    tread_routing_prob = 0.5
    if config.training.tread.enabled:
        tread_route_start = config.training.tread.route_start
        end = config.training.tread.route_end
        tread_route_end = config.model.depth - 2 if end < 0 else end
        tread_routing_prob = config.training.tread.routing_probability
        print(f"TREAD enabled: routing {tread_routing_prob*100:.0f}% tokens past blocks {tread_route_start}-{tread_route_end}")
    
    model = NanoDiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
        repa_block_idx=repa_block_idx,
        tread_route_start=tread_route_start,
        tread_route_end=tread_route_end,
        tread_routing_prob=tread_routing_prob,
        bottleneck_size=config.model.bottleneck_size,
        num_pose_joints=config.model.num_pose_joints,
        pose_confidence_threshold=config.model.pose_confidence_threshold,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
    if config.training.gradient_checkpointing:
        print(f"Gradient checkpointing: ENABLED (trades ~20-30% speed for 3-4x memory savings)")
    opt_type = config.training.optimizer.type
    if opt_type == 'Muon':
        n2d = sum(p.numel() for p in model.parameters() if p.requires_grad and p.ndim == 2)
        n_other = trainable_params - n2d
        print(f"Optimizer: Muon (hybrid) — {n2d/1e6:.1f}M params Muon, {n_other/1e6:.3f}M params AdamW")
    else:
        print(f"Optimizer: {opt_type}")
    if config.sampling.self_guidance:
        print(f"Self-guidance CFG: ENABLED (scale={config.sampling.guidance_scale})")
    else:
        print(f"Dual CFG: text_scale={config.sampling.text_scale}, dino_scale={config.sampling.dino_scale}")
    res_phases = config.training.get_resolution_phases()
    if res_phases:
        phase_strs = [f"{p.scale}x until step {p.until_step}" for p in res_phases]
        print(f"Resolution schedule: {', '.join(phase_strs)}")
    if config.training.perceptual.enabled:
        p = config.training.perceptual
        print(f"Perceptual loss (LPIPS): weight={p.lpips_weight}, every {p.every_n_microsteps} micro-steps, crop={p.crop_size}")
    
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
            text_scale=config.sampling.text_scale,
            dino_scale=config.sampling.dino_scale,
            num_steps=config.sampling.num_steps,
            self_guidance=config.sampling.self_guidance,
            guidance_scale=config.sampling.guidance_scale,
            prediction_type=config.model.prediction_type,
            get_resolution_scale=lambda: getattr(dataloader, 'resolution_scale', 1.0),
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
            self_guidance=config.sampling.self_guidance,
            guidance_scale=config.sampling.guidance_scale,
            get_resolution_scale=lambda: getattr(dataloader, 'resolution_scale', 1.0),
            prediction_type=config.model.prediction_type,
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
