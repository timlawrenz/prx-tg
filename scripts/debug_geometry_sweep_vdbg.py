#!/usr/bin/env python3
"""Geometry sweep in visual_debug style — full CFG, EulerSampler directly.

Reproduces the visual_debug sampling path but with z_g dimension sweeping.
"""
import sys, torch
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from production.config_loader import load_config
from production.model import NanoDiT
from production.train import EMAModel
from production.sample import EulerSampler, tensor_to_pil
from production.data import get_deterministic_validation_dataloader


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = 'experiments/eidolon-conditioning/config.yaml'
CHECKPOINT = 'experiments/eidolon-conditioning/runs/2026-06-30_2202/checkpoints/checkpoint_step0004500.pt'
STEP = 4500
OUTPUT_DIR = Path('experiments/eidolon-conditioning/runs/2026-06-30_2202/_vdbg_geometry_sweep')

# Sweep settings (matching validation runner defaults)
SWEEP_DIM = 0
SWEEP_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
SWEEP_SAMPLE_INDICES = [10, 30, 50]

# CFG scales (visual_debug uses full scales)
TEXT_SCALE = 3.0   # identity in eidolon
DINO_SCALE = 2.0   # geometry in eidolon
NUM_STEPS = 50


def create_image_collage(images, spacing=10):
    """Create a horizontal collage from torch tensors in [-1, 1]."""
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img_np = (img.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
        else:
            pil_img = img
        pil_images.append(pil_img)

    max_height = max(img.height for img in pil_images)
    total_width = sum(img.width for img in pil_images) + spacing * (len(pil_images) - 1)
    collage = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    x_offset = 0
    for pil_img in pil_images:
        y_offset = (max_height - pil_img.height) // 2
        collage.paste(pil_img, (x_offset, y_offset))
        x_offset += pil_img.width + spacing

    return collage


def main():
    print(f"=== Geometry sweep — visual debug style (step {STEP}) ===")
    print(f"    Config: {CONFIG}")
    print(f"    Checkpoint: {CHECKPOINT}")
    print(f"    CFG: identity_scale={TEXT_SCALE}, geometry_scale={DINO_SCALE}")

    config = load_config(CONFIG)

    # ── Build model ────────────────────────────────────────────────────────────
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
    adapter_cfg = getattr(config, 'adapter', None)
    if adapter_cfg:
        kwargs['adapter_kwargs'] = {
            'name': adapter_cfg.name,
            'identity_dim': getattr(adapter_cfg, 'identity_dim', 64),
            'z_g_dim': getattr(adapter_cfg, 'z_g_dim', 50),
        }
    model = NanoDiT(**kwargs)
    print(f"    Model: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("=== Loading checkpoint ===")
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    model = model.to(device)
    # CRITICAL: visual_debug uses self.ema.model which IS the training model
    # (post-optimizer weights), NOT the EMA-smoothed params.
    # Using ema.copy_to() gives EMA-smoothed weights which produce blockier
    # output at step 4500. Use raw training weights to match visual_debug quality.
    model.eval()

    # ── Load samples from stratum (matching visual_debug's dataloader) ────────
    print("=== Loading validation samples from stratum ===")
    loader = get_deterministic_validation_dataloader(
        shard_dir=None,
        batch_size=1,
        target_latent_size=config.model.input_size,
        source='stratum',
        stratum_dir=config.data.stratum_dir,
        adapter_name='eidolon',
    )

    # Pull enough samples for all sweep indices
    max_idx = max(SWEEP_SAMPLE_INDICES)
    samples = []
    data_iter = iter(loader)
    sample_count = 0
    while sample_count <= max_idx:
        batch = next(data_iter)
        for i in range(batch['image_data'].shape[0]):
            if sample_count <= max_idx:
                samples.append({
                    'identity_emb': batch['identity_emb'][i],
                    'geometry_emb': batch['geometry_emb'][i],
                    'image_id': batch['image_ids'][i],
                })
                sample_count += 1
    print(f"    Loaded {len(samples)} samples (up to idx {max_idx})")

    # ── Run geometry sweep ────────────────────────────────────────────────────
    sampler = EulerSampler(num_steps=NUM_STEPS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_idx in SWEEP_SAMPLE_INDICES:
        sample = samples[sample_idx]
        identity_emb = sample['identity_emb'].unsqueeze(0).to(device)  # (1, 64)
        base_geometry = sample['geometry_emb'].unsqueeze(0).to(device)  # (1, 50)

        print(f"\n--- Sample {sample_idx} (image_id={sample['image_id']}) ---")

        sweep_images = []

        for val in SWEEP_VALUES:
            modified_geometry = base_geometry.clone()
            modified_geometry[:, SWEEP_DIM] = val

            with torch.no_grad():
                output = sampler.sample(
                    model=model,
                    shape=(1, 3, 1024, 1024),  # pixel-space full res
                    identity_emb=identity_emb,
                    geometry_emb=modified_geometry,
                    device=device,
                    text_scale=TEXT_SCALE,
                    dino_scale=DINO_SCALE,
                    prediction_type='x_prediction',
                )

            # Convert like visual_debug: [0,1] → [-1,1] for tensor_to_pil
            img_tensor = output[0].clamp(0, 1) * 2 - 1  # (3, 1024, 1024)
            sweep_images.append(img_tensor)

            # Save individual image
            pil_img = tensor_to_pil(img_tensor)
            filename = f"sample{sample_idx:02d}_dim{SWEEP_DIM}_val{val:+.1f}.png"
            pil_img.save(OUTPUT_DIR / filename)
            print(f"    z_g[{SWEEP_DIM}]={val:+.1f} → {filename}")

        # Create and save collage
        collage = create_image_collage(sweep_images, spacing=10)
        collage_path = OUTPUT_DIR / f"sample{sample_idx:02d}_sweep_collage.png"
        collage.save(collage_path)
        print(f"    Collage → {collage_path}")

    print(f"\n=== Done! {len(SWEEP_SAMPLE_INDICES)} samples swept ===")
    print(f"    Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
