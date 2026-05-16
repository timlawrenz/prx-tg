#!/usr/bin/env python3
"""Run full validation suite on a specific saved checkpoint."""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from production.config_loader import load_config
from production.model import NanoDiT
from production.train import EMAModel
from production.validate import create_validation_fn

def main():
    parser = argparse.ArgumentParser(description="Run validation on a saved checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt file")
    parser.add_argument("--step", type=int, default=None, help="Step number (inferred from filename if not provided)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Determine step number
    step = args.step
    if step is None:
        import re
        match = re.search(r'step(\d+)', args.checkpoint)
        if match:
            step = int(match.group(1))
        else:
            print("Could not infer step from filename, defaulting to 0")
            step = 0
    print(f"Validating for step: {step}")
    
    # Determine experiment directory for outputs
    exp_dir = Path(args.config).parent
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Create model
    print("Creating model architecture...")
    kwargs = {
        'input_size': config.model.input_size,
        'patch_size': config.model.patch_size,
        'in_channels': config.model.in_channels,
        'hidden_size': config.model.hidden_size,
        'depth': config.model.depth,
        'num_heads': config.model.num_heads,
        'mlp_ratio': config.model.mlp_ratio,
        'use_gradient_checkpointing': False,
    }
    
    # Handle optional REPA and TREAD settings
    if hasattr(config.training, 'repa') and getattr(config.training.repa, 'enabled', False):
        kwargs['repa_block_idx'] = getattr(config.training.repa, 'block_index', -1)
    if hasattr(config.training, 'tread') and getattr(config.training.tread, 'enabled', False):
        kwargs['tread_route_start'] = getattr(config.training.tread, 'route_start', 1)
        kwargs['tread_route_end'] = getattr(config.training.tread, 'route_end', -1)
        kwargs['tread_routing_prob'] = getattr(config.training.tread, 'routing_probability', 0.5)

    model = NanoDiT(**kwargs)
    
    # Create EMA container
    ema = EMAModel(
        model, 
        decay=config.training.ema_decay, 
        warmup_steps=config.training.ema_warmup_steps
    )
    
    # Load weights
    print("Loading weights...")
    model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    
    model = model.to(device)
    for param in ema.ema_params.values():
        param.data = param.data.to(device)
    
    model.eval()
    
    # Create TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_dir = exp_dir / 'tensorboard'
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    # Create validation function
    print("Creating validation function...")
    validate_fn = create_validation_fn(
        shard_dir=config.data.shard_base_dir,
        output_dir=str(exp_dir / config.validation.output_dir),
        tensorboard_writer=writer,
        text_scale=config.sampling.text_scale,
        dino_scale=config.sampling.dino_scale,
        num_steps=config.sampling.num_steps,
        prediction_type=getattr(config.model, 'prediction_type', 'v_prediction'),
        self_guidance=getattr(config.sampling, 'self_guidance', False),
        guidance_scale=getattr(config.sampling, 'guidance_scale', 3.0),
        source=getattr(config.data, 'source', 'webdataset'),
        stratum_dir=getattr(config.data, 'stratum_dir', '/mnt/nas-ai-models/training-data/ffhq/stratum'),
    )

    # Run validation
    print(f"\n{'='*60}")
    print(f"RUNNING VALIDATION FOR STEP {step}")
    print(f"{'='*60}")
    print("="*60)
    
    with torch.no_grad():
        validate_fn(model, ema, step, device)
        
    writer.close()
        
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
