#!/usr/bin/env python3
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from production.config_loader import load_config
from production.model import NanoDiT
from production.train import EMAModel
from production.validate import create_validation_fn

def strip_orig_mod(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    step = args.step if args.step is not None else 0
    exp_dir = Path(args.config).parent
    
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
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

    model = NanoDiT(**kwargs)
    ema = EMAModel(model, decay=config.training.ema_decay, warmup_steps=config.training.ema_warmup_steps)
    
    model.load_state_dict(strip_orig_mod(ckpt['model']))
    ema.load_state_dict(strip_orig_mod(ckpt['ema']))
    
    model = model.to(device)
    for param in ema.ema_params.values():
        param.data = param.data.to(device)
    
    model.eval()
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(exp_dir / 'tensorboard_temp'))
    
    validate_fn = create_validation_fn(
        shard_dir=config.data.shard_base_dir,
        output_dir=str(exp_dir / "validation_custom"),
        tensorboard_writer=writer,
        text_scale=config.sampling.text_scale,
        dino_scale=config.sampling.dino_scale,
        num_steps=config.sampling.num_steps,
        prediction_type=getattr(config.model, 'prediction_type', 'v_prediction'),
        self_guidance=False,
        guidance_scale=getattr(config.sampling, 'guidance_scale', 3.0),
        source=getattr(config.data, 'source', 'webdataset'),
        stratum_dir=getattr(config.data, 'stratum_dir', '/mnt/nas-ai-models/training-data/ffhq/stratum'),
    )

    with torch.no_grad():
        validate_fn(model, ema, step, device)
        
    writer.close()

if __name__ == "__main__":
    main()
