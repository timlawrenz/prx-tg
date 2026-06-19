#!/usr/bin/env python3
import argparse
import torch
import sys
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from production.config_loader import load_config
from production.model import NanoDiT
from production.train import EMAModel
from production.validate import ValidationSampler
from torchvision.utils import save_image
from production.validate import ValidationRunner

def strip_orig_mod(state_dict):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
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
    
    ema.copy_to(model)
    model = model.to(device)
    model.eval()

    class DummyLoader:
        def __iter__(self):
            return iter([])
            
    runner = ValidationRunner(
        model,
        ema,
        DummyLoader(),
        device=device,
        output_dir=exp_dir / "validation_custom",
        prediction_type=getattr(config.model, 'prediction_type', 'v_prediction'),
    )

    sampler = ValidationSampler(
        model,
        None, 
        device=device,
        num_steps=config.sampling.num_steps,
        text_scale=3.0,
        dino_scale=0.0, 
        self_guidance=False, 
        prediction_type=getattr(config.model, 'prediction_type', 'v_prediction'),
    )

    out_dir = exp_dir / "normal_prompts"
    out_dir.mkdir(exist_ok=True)

    prompts = [
        "A professional headshot of a middle-aged woman with curly brown hair, wearing a navy blazer, smiling warmly.",
        "A cinematic portrait of a young man with a slight beard, wearing a casual grey t-shirt, soft studio lighting.",
        "A close-up photograph of an elderly woman with kind eyes and deep wrinkles, wearing round glasses.",
        "A passport photo of a teenage boy with short blonde hair, freckles, and a neutral expression.",
        "A candid photo of a smiling woman with dark skin and a shaved head, wearing silver hoop earrings."
    ]

    print("Generating 'normal' portrait prompts...")
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt}")
        text_emb, text_mask = runner.encode_caption(prompt)
        text_emb = text_emb.to(device)
        text_mask = text_mask.to(device)
        
        dino_emb = torch.zeros(1, 1024, device=device)
        dino_patches = torch.zeros(1, 256, 1024, device=device)
        
        with torch.no_grad():
            gen_image = sampler.generate(
                dino_emb,
                dino_patches,
                text_emb,
                text_mask,
                latent_size=getattr(model, 'input_size', 128),
                self_guidance=False,
                dino_scale=0.0,
                text_scale=3.5 
            )[0] 
        
        gen_image = (gen_image + 1.0) / 2.0
        gen_image = gen_image.clamp(0, 1)
        save_path = out_dir / f"normal_{i:02d}.png"
        save_image(gen_image, save_path)
        
        with open(out_dir / f"normal_{i:02d}.txt", "w") as f:
            f.write(prompt)

    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()