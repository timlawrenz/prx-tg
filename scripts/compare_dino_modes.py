#!/usr/bin/env python3
"""Compare text-only vs DINO-only vs text+DINO generation on real validation samples."""
import argparse, torch, sys, json, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from production.config_loader import load_config
from production.model import NanoDiT
from production.train import EMAModel
from production.validate import ValidationSampler, ValidationRunner
from production.sample import tensor_to_pil
from production.data import get_deterministic_validation_dataloader


def strip_orig_mod(state_dict):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/dino_comparison")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=35)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"Checkpoint step: {ckpt['step']}")

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
    if hasattr(config.training, 'repa') and config.training.repa.enabled:
        kwargs['repa_block_idx'] = config.training.repa.block_index
    if hasattr(config.training, 'tread') and config.training.tread.enabled:
        kwargs['tread_route_start'] = config.training.tread.route_start
        kwargs['tread_route_end'] = config.training.tread.route_end
        kwargs['tread_routing_prob'] = config.training.tread.routing_probability

    model = NanoDiT(**kwargs)
    ema = EMAModel(model, decay=config.training.ema_decay,
                   warmup_steps=config.training.ema_warmup_steps)
    model.load_state_dict(strip_orig_mod(ckpt['model']))
    ema.load_state_dict(strip_orig_mod(ckpt['ema']))
    ema.copy_to(model)
    model = model.to(device).eval()

    pred_type = getattr(config.model, 'prediction_type', 'v_prediction')

    # Load validation data with real DINO embeddings
    stratum_dir = os.environ.get('STRATUM_DIR', '')
    if not stratum_dir or stratum_dir == '':
        stratum_dir = '/mnt/nas-ai-models/training-data/ffhq/stratum'
    print(f"Stratum dir: {stratum_dir}")

    loader = get_deterministic_validation_dataloader(
        shard_dir=None,
        batch_size=1,
        target_latent_size=config.model.input_size,
        source='stratum',
        stratum_dir=stratum_dir,
    )

    # Collect samples
    samples = []
    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break
        samples.append(batch)
        print(f"Loaded sample {i}: caption={batch.get('caption', 'N/A')[:80]}")

    # Create sampler (dual CFG mode for flexible control)
    sampler = ValidationSampler(
        model, None, device=device, num_steps=args.steps,
        text_scale=3.0, dino_scale=2.5,
        self_guidance=False, prediction_type=pred_type,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    results = []

    for idx, sample in enumerate(samples):
        dino_emb = sample['dino_embedding'].to(device)
        dino_patches = sample['dinov3_patches'].to(device)
        text_emb = sample['t5_hidden'].to(device)
        text_mask = sample['t5_mask'].to(device)
        caption = sample.get('captions', ['N/A'])[0]
        image_id = sample.get('image_ids', [f'sample_{idx}'])[0]

        print(f"\n--- Sample {idx}: {caption[:80]} ---")

        # Mode A: Text-only (dino_scale=0)
        torch.manual_seed(42)
        with torch.no_grad():
            img_text_only = sampler.generate(
                dino_emb, dino_patches, text_emb, text_mask,
                latent_size=config.model.input_size,
                self_guidance=False, text_scale=6.0, dino_scale=0.0,
            )[0]

        # Mode B: DINO-only (text_scale=0)
        torch.manual_seed(42)
        with torch.no_grad():
            img_dino_only = sampler.generate(
                dino_emb, dino_patches, text_emb, text_mask,
                latent_size=config.model.input_size,
                self_guidance=False, text_scale=0.0, dino_scale=5.0,
            )[0]

        # Mode C: Text + DINO (balanced)
        torch.manual_seed(42)
        with torch.no_grad():
            img_both = sampler.generate(
                dino_emb, dino_patches, text_emb, text_mask,
                latent_size=config.model.input_size,
                self_guidance=False, text_scale=3.0, dino_scale=2.5,
            )[0]

        # Save individual images
        for mode, img in [("text_only", img_text_only), ("dino_only", img_dino_only), ("both", img_both)]:
            pil = tensor_to_pil(img)
            path = out_dir / f"sample{idx}_{mode}.png"
            pil.save(path)

        results.append({
            'idx': idx, 'image_id': image_id, 'caption': caption,
            'text_only': str(out_dir / f"sample{idx}_text_only.png"),
            'dino_only': str(out_dir / f"sample{idx}_dino_only.png"),
            'both': str(out_dir / f"sample{idx}_both.png"),
        })

    # Save results manifest
    (out_dir / "manifest.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone. {len(results)} samples × 3 modes saved to {out_dir}/")


if __name__ == "__main__":
    main()
