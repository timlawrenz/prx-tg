"""Visual debugging utilities for production training.

Quick image generation during training to monitor visual quality.
Separate from full validation suite for faster iteration.
"""

import torch
from pathlib import Path
from .sample import EulerSampler, load_vae_decoder, decode_latents, tensor_to_pil


def create_visual_debug_fn(
    shard_dir,
    output_dir,
    num_samples=4,
    text_scale=3.0,
    dino_scale=2.0,
    num_steps=50,
    device='cuda'
):
    """Create visual debugging function for training loop.
    
    Args:
        shard_dir: Path to shard directory (for loading samples)
        output_dir: Output directory for visual debug images
        num_samples: Number of images to generate
        text_scale: CFG text scale
        dino_scale: CFG DINO scale
        num_steps: Number of sampling steps
        device: Device to run on
    
    Returns:
        debug_fn: Function that takes (model, step) and generates images
    """
    # Load VAE decoder
    vae = load_vae_decoder(device=device)
    sampler = EulerSampler(num_steps=num_steps)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load a fixed set of samples for consistent monitoring
    debug_samples = _load_debug_samples(shard_dir, num_samples, device)
    
    @torch.no_grad()
    def debug_fn(model, step):
        """Generate visual debug images at current training step."""
        model.eval()
        
        # Create step directory
        step_dir = output_path / f"step{step:07d}"
        step_dir.mkdir(exist_ok=True)
        
        # Generate images for each debug sample
        for idx, sample in enumerate(debug_samples):
            # Extract conditioning
            dino_emb = sample['dino'].unsqueeze(0)  # (1, 1024)
            text_emb = sample['text_emb'].unsqueeze(0)  # (1, 77, 1024)
            text_mask = sample['text_mask'].unsqueeze(0)  # (1, 77)
            caption = sample['caption']
            
            # Sample latents
            latent_shape = (1, 16, 64, 64)  # Fixed for now (Stage 1)
            latents = sampler.sample(
                model=model,
                shape=latent_shape,
                dino_emb=dino_emb,
                text_emb=text_emb,
                text_mask=text_mask,
                device=device,
                text_scale=text_scale,
                dino_scale=dino_scale,
            )
            
            # Decode to image (64x64 latent -> 512x512 image via 8x VAE upsampling)
            images = decode_latents(vae, latents)
            img_tensor = images[0]  # (3, 512, 512) in [-1, 1]
            pil_img = tensor_to_pil(img_tensor)
            
            # Save with caption truncated for filename
            safe_caption = caption[:50].replace(' ', '_').replace('/', '_')
            filename = f"sample{idx:02d}_{safe_caption}.png"
            pil_img.save(step_dir / filename)
        
        print(f"Visual debug: Generated {num_samples} images at step {step} -> {step_dir}")
        model.train()
    
    return debug_fn


def _load_debug_samples(shard_dir, num_samples, device):
    """Load a fixed set of samples for consistent visual debugging.
    
    Args:
        shard_dir: Path to shard directory
        num_samples: Number of samples to load
        device: Device to load to
    
    Returns:
        samples: List of dicts with keys: dino, text_emb, text_mask, caption
    """
    from .data import ValidationDataset
    
    # Create dataset (same as training)
    dataset = ValidationDataset(
        shard_dir=shard_dir,
        flip_prob=0.0,  # No flip for debug samples
        target_latent_size=64,  # Fixed for now (Stage 1)
        batch_size=1,
    )
    
    # Load first N samples directly from iterator (yields batched dicts)
    samples = []
    for idx, batch in enumerate(dataset):
        if idx >= num_samples:
            break
        
        # Extract single sample from batch (batch_size=1)
        # collate_fn returns: dino_embedding, t5_hidden, t5_mask, captions (list), image_ids (list)
        sample = {
            'dino': batch['dino_embedding'][0].to(device),  # (1024,)
            'text_emb': batch['t5_hidden'][0].to(device),  # (77, 1024)
            'text_mask': batch['t5_mask'][0].to(device),  # (77,)
            'caption': batch['captions'][0],
        }
        samples.append(sample)
    
    return samples
