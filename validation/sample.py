"""Sampling utilities for Nano DiT validation."""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path


class EulerSampler:
    """Euler sampler for rectified flow models."""
    
    def __init__(self, num_steps=50):
        """
        Args:
            num_steps: number of denoising steps
        """
        self.num_steps = num_steps
        # Uniform timesteps from 1.0 to 0.0
        self.timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        dino_emb,
        text_emb,
        text_mask,
        device='cuda',
        text_scale=3.0,
        dino_scale=2.0,
    ):
        """Sample from model using Euler integration with dual CFG.
        
        Args:
            model: NanoDiT model
            shape: (B, C, H, W) output shape
            dino_emb: (B, 1024) DINOv3 embeddings
            text_emb: (B, 77, 1024) T5 hidden states
            text_mask: (B, 77) T5 attention mask
            device: torch device
            text_scale: CFG scale for text conditioning
            dino_scale: CFG scale for DINO conditioning
        
        Returns:
            latents: (B, C, H, W) sampled latents
        """
        B = shape[0]
        
        # Start from pure noise (t=1.0)
        zt = torch.randn(shape, device=device)
        
        timesteps = self.timesteps.to(device)
        
        for i in range(self.num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            
            t_batch = torch.full((B,), t_curr, device=device)
            
            # Three forward passes for dual CFG
            # 1. Unconditional (both dropped)
            v_uncond = model(
                zt, t_batch, dino_emb, text_emb, text_mask,
                cfg_drop_both=torch.ones(B, dtype=torch.bool, device=device),
            )
            
            # 2. Text-only (DINO dropped)
            v_text = model(
                zt, t_batch, dino_emb, text_emb, text_mask,
                cfg_drop_dino=torch.ones(B, dtype=torch.bool, device=device),
            )
            
            # 3. DINO-only (text dropped)
            v_dino = model(
                zt, t_batch, dino_emb, text_emb, text_mask,
                cfg_drop_text=torch.ones(B, dtype=torch.bool, device=device),
            )
            
            # Dual CFG combination
            # v = v_uncond + text_scale * (v_text - v_uncond) + dino_scale * (v_dino - v_uncond)
            v_pred = v_uncond + text_scale * (v_text - v_uncond) + dino_scale * (v_dino - v_uncond)
            
            # Euler integration: z_{t-dt} = z_t + v * dt
            zt = zt + v_pred * dt
        
        return zt


def load_vae_decoder(device='cuda'):
    """Load Flux VAE decoder for latent decoding.
    
    Returns:
        decoder: VAE decoder model
    """
    try:
        from diffusers import AutoencoderKL
        
        # Load Flux VAE (same as used in embedding generation)
        vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="vae",
            torch_dtype=torch.float16,
        ).to(device)
        vae.eval()
        
        return vae
    except Exception as e:
        print(f"Error loading VAE decoder: {e}")
        print("Make sure you have diffusers installed and HuggingFace access")
        raise


@torch.no_grad()
def decode_latents(vae, latents):
    """Decode VAE latents to RGB images.
    
    Args:
        vae: VAE decoder model
        latents: (B, 16, H, W) latent tensors
    
    Returns:
        images: (B, 3, H*8, W*8) RGB images in [-1, 1]
    """
    # Flux VAE uses 8x spatial compression
    # latents: (B, 16, 64, 64) -> images: (B, 3, 512, 512)
    
    # Convert to half precision for faster decoding
    latents = latents.half()
    
    # Decode
    images = vae.decode(latents).sample
    
    return images


def tensor_to_pil(tensor):
    """Convert tensor to PIL image.
    
    Args:
        tensor: (C, H, W) tensor in [-1, 1]
    
    Returns:
        image: PIL Image
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    array = tensor.cpu().float().numpy()
    array = (array * 255).astype(np.uint8)
    
    # Convert CHW to HWC
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    
    return Image.fromarray(array)


def save_images(images, save_dir, prefix='sample', image_ids=None):
    """Save batch of images to directory.
    
    Args:
        images: (B, 3, H, W) tensor
        save_dir: Path to save directory
        prefix: filename prefix
        image_ids: optional list of image IDs for filenames
    
    Returns:
        saved_paths: list of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, img_tensor in enumerate(images):
        if image_ids is not None:
            filename = f"{prefix}_{image_ids[i]}.png"
        else:
            filename = f"{prefix}_{i:04d}.png"
        
        filepath = save_dir / filename
        
        pil_img = tensor_to_pil(img_tensor)
        pil_img.save(filepath)
        
        saved_paths.append(str(filepath))
    
    return saved_paths


class ValidationSampler:
    """High-level sampler for validation tests."""
    
    def __init__(
        self,
        model,
        vae,
        device='cuda',
        num_steps=50,
        text_scale=3.0,
        dino_scale=2.0,
    ):
        """
        Args:
            model: NanoDiT model
            vae: VAE decoder
            device: torch device
            num_steps: number of sampling steps
            text_scale: CFG scale for text
            dino_scale: CFG scale for DINO
        """
        self.model = model
        self.vae = vae
        self.device = device
        self.sampler = EulerSampler(num_steps=num_steps)
        self.text_scale = text_scale
        self.dino_scale = dino_scale
    
    @torch.no_grad()
    def generate(
        self,
        dino_emb,
        text_emb,
        text_mask,
        latent_size=64,
        batch_size=None,
    ):
        """Generate images from conditioning.
        
        Args:
            dino_emb: (B, 1024) DINOv3 embeddings
            text_emb: (B, 77, 1024) T5 hidden states
            text_mask: (B, 77) T5 attention mask
            latent_size: spatial size of latents (64 = 512x512 images)
            batch_size: override batch size (default: from embeddings)
        
        Returns:
            images: (B, 3, 512, 512) RGB images
        """
        self.model.eval()
        
        if batch_size is None:
            batch_size = dino_emb.shape[0]
        
        # Ensure everything is on correct device
        dino_emb = dino_emb.to(self.device)
        text_emb = text_emb.to(self.device)
        text_mask = text_mask.to(self.device)
        
        # Sample latents
        shape = (batch_size, 16, latent_size, latent_size)
        latents = self.sampler.sample(
            self.model,
            shape,
            dino_emb,
            text_emb,
            text_mask,
            device=self.device,
            text_scale=self.text_scale,
            dino_scale=self.dino_scale,
        )
        
        # Decode to images
        images = decode_latents(self.vae, latents)
        
        return images


if __name__ == "__main__":
    # Test sampler
    print("Testing Euler sampler...")
    
    sampler = EulerSampler(num_steps=10)
    print(f"Timesteps: {sampler.timesteps}")
    print(f"Number of steps: {sampler.num_steps}")
    
    # Test timestep spacing
    dts = sampler.timesteps[:-1] - sampler.timesteps[1:]
    print(f"Step sizes (dt): min={dts.min():.4f}, max={dts.max():.4f}, mean={dts.mean():.4f}")
    
    print("✓ Sampler test passed")
