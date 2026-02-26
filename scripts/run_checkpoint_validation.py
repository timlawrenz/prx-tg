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
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    
    # Create model
    print("Creating model architecture...")
    model = NanoDiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        use_gradient_checkpointing=False, # Not needed for inference
    )
    
    # Create EMA container
    ema = EMAModel(
        model, 
        decay=config.training.ema_decay, 
        warmup_steps=config.training.ema_warmup_steps
    )
    
    # Load weights
    print("Loading weights...")
    model.load_state_dict(ckpt['model_state'])
    ema.load_state_dict(ckpt['ema_state'])
    
    model = model.to(device)
    for param in ema.ema_params.values():
        param.data = param.data.to(device)
    
    model.eval()
    
    # Create validation function
    print("Creating validation suite...")
    validate_fn = create_validation_fn(
        config=config,
        device=device,
        experiment_dir=exp_dir
    )
    
    # Run validation
    print("\n" + "="*60)
    print(f"RUNNING VALIDATION FOR STEP {step}")
    print("="*60)
    
    with torch.no_grad():
        validate_fn(model, ema, step, device)
        
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
