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
    device='cuda',
    tensorboard_writer=None,
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
        tensorboard_writer: Optional TensorBoard SummaryWriter for logging
    
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
        
        # Collect all generated images for collage
        generated_images = []
        captions_text = []
        
        # Generate images for each debug sample
        for idx, sample in enumerate(debug_samples):
            # Extract conditioning
            dino_emb = sample['dino'].unsqueeze(0)  # (1, 1024)
            text_emb = sample['text_emb'].unsqueeze(0)  # (1, 512, 1024)
            text_mask = sample['text_mask'].unsqueeze(0)  # (1, 512)
            caption = sample['caption']
            
            # Sample latents
            latent_shape = (1, 16, 128, 128)  # 1024x1024 target resolution
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
            
            # Decode to image (128x128 latent -> 1024x1024 image via 8x VAE upsampling)
            images = decode_latents(vae, latents)
            img_tensor = images[0]  # (3, 1024, 1024) in [-1, 1]
            
            # Save individual image to disk
            pil_img = tensor_to_pil(img_tensor)
            safe_caption = caption[:50].replace(' ', '_').replace('/', '_')
            filename = f"sample{idx:02d}_{safe_caption}.png"
            pil_img.save(step_dir / filename)
            
            # Collect for collage
            generated_images.append(img_tensor)
            captions_text.append(f"Sample {idx}: {caption[:80]}")
        
        # Create collage for TensorBoard (if available)
        if tensorboard_writer is not None:
            # Import create_image_collage from validate module
            import numpy as np
            from PIL import Image
            
            # Create horizontal collage
            collage_array = create_image_collage_from_tensors(generated_images, spacing=10)
            
            # Save collage to disk
            collage_img = Image.fromarray(collage_array)
            collage_img.save(step_dir / "collage.png")
            
            # Log to TensorBoard
            collage_tensor = torch.from_numpy(collage_array).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            tensorboard_writer.add_image(
                'visual_debug/collage',
                collage_tensor,
                global_step=step,
                dataformats='CHW'
            )
            
            # Log captions as text
            caption_text = "\n\n".join(captions_text)
            tensorboard_writer.add_text(
                'visual_debug/captions',
                caption_text,
                global_step=step
            )
        
        print(f"Visual debug: Generated {num_samples} images at step {step} -> {step_dir}")
        model.train()
    
    return debug_fn


def create_image_collage_from_tensors(images, spacing=10):
    """Create a horizontal collage from torch tensors.
    
    Args:
        images: List of torch tensors (C, H, W) in [-1, 1]
        spacing: Pixels between images
    
    Returns:
        collage_array: numpy array (H, W, 3) in [0, 255] uint8
    """
    import numpy as np
    from PIL import Image
    
    # Convert all to PIL Images
    pil_images = []
    for img in images:
        # Denormalize from [-1, 1] to [0, 1]
        img_np = (img.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        # Convert to (H, W, C) uint8
        img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_images.append(pil_img)
    
    # Calculate collage dimensions
    max_height = max(img.height for img in pil_images)
    total_width = sum(img.width for img in pil_images) + spacing * (len(pil_images) - 1)
    
    # Create white background
    collage = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    
    # Paste images
    x_offset = 0
    for pil_img in pil_images:
        # Center vertically
        y_offset = (max_height - pil_img.height) // 2
        collage.paste(pil_img, (x_offset, y_offset))
        x_offset += pil_img.width + spacing
    
    # Convert to numpy for TensorBoard
    collage_array = np.array(collage)  # (H, W, 3) uint8
    
    return collage_array


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
    
    # Create dataset with deterministic sampling
    dataset = ValidationDataset(
        shard_dir=shard_dir,
        flip_prob=0.0,  # No flip for debug samples
        target_latent_size=128,  # 1024x1024 target resolution
        batch_size=1,
        shuffle=False,  # No shuffle for deterministic order
        deterministic=True,  # Set seeds for reproducible sampling
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
            'text_emb': batch['t5_hidden'][0].to(device),  # (512, 1024)
            'text_mask': batch['t5_mask'][0].to(device),  # (512,)
            'caption': batch['captions'][0],
        }
        samples.append(sample)
    
    return samples
