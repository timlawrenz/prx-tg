#!/usr/bin/env python3
"""Run validation on a specific eidolon checkpoint with the fixed sampler."""
import sys, torch, re, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from production.config_loader import load_config
from production.model import NanoDiT
from production.train import EMAModel
from production.validate import create_validation_fn

def main():
    config_path = 'experiments/eidolon-conditioning/config.yaml'
    checkpoint_path = 'experiments/eidolon-conditioning/runs/2026-06-30_2202/checkpoints/checkpoint_step0003000.pt'
    step = 3000

    print("=== Loading config ===")
    config = load_config(config_path)

    print("=== Creating model ===")
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
    if hasattr(config.training, 'repa') and getattr(config.training.repa, 'enabled', False):
        kwargs['repa_block_idx'] = getattr(config.training.repa, 'block_index', -1)
    if hasattr(config.training, 'tread') and getattr(config.training.tread, 'enabled', False):
        kwargs['tread_route_start'] = getattr(config.training.tread, 'route_start', 1)
        kwargs['tread_route_end'] = getattr(config.training.tread, 'route_end', -1)
        kwargs['tread_routing_prob'] = getattr(config.training.tread, 'routing_probability', 0.5)

    # Add adapter kwargs from config
    adapter_cfg = getattr(config, 'adapter', None)
    if adapter_cfg:
        adapter_kwargs = {
            'name': adapter_cfg.name,
            'identity_dim': getattr(adapter_cfg, 'identity_dim', 64),
            'z_g_dim': getattr(adapter_cfg, 'z_g_dim', 50),
        }
        kwargs['adapter_kwargs'] = adapter_kwargs
    model = NanoDiT(**kwargs)
    print(f"  Model created: {sum(p.numel() for p in model.parameters()):,} params")

    print("=== Loading checkpoint ===")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])

    ema = EMAModel(model, decay=config.training.ema_decay, warmup_steps=config.training.ema_warmup_steps)
    ema.load_state_dict(ckpt['ema'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    model = model.to(device)
    for param in ema.ema_params.values():
        param.data = param.data.to(device)
    model.eval()

    # Sampling defaults (eidolon config doesn't have a sampling section)
    text_scale = getattr(config, 'sampling', None)
    text_scale = getattr(text_scale, 'text_scale', 3.0) if text_scale else 3.0
    
    dino_scale = getattr(config, 'sampling', None)  
    dino_scale = getattr(dino_scale, 'dino_scale', 2.0) if dino_scale else 2.0
    
    num_steps = getattr(config, 'sampling', None)
    num_steps = getattr(num_steps, 'num_steps', 50) if num_steps else 50
    
    prediction_type = getattr(config.model, 'prediction_type', 'v_prediction')
    stratum_dir = getattr(config.data, 'stratum_dir', '/mnt/nas-ai-models/training-data/ffhq/stratum')

    print(f"  text_scale={text_scale}, dino_scale={dino_scale}, num_steps={num_steps}")
    print(f"  prediction_type={prediction_type}")
    print(f"  stratum_dir={stratum_dir}")

    exp_dir = Path(config_path).parent
    output_dir = str(exp_dir / 'validation')

    print("=== Creating validation function ===")
    validate_fn = create_validation_fn(
        shard_dir=None,
        output_dir=output_dir,
        text_scale=text_scale,
        dino_scale=dino_scale,
        num_steps=num_steps,
        prediction_type=prediction_type,
        source='stratum',
        stratum_dir=stratum_dir,
        adapter_name='eidolon',
    )

    print("=== Running validation ===")
    with torch.no_grad():
        validate_fn(model, ema, step, device)

    print("\nDone!")

if __name__ == '__main__':
    main()
