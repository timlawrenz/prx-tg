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
        dino_patches,
        text_emb,
        text_mask,
        device='cuda',
        text_scale=3.0,
        dino_scale=2.0,
        self_guidance=False,
        guidance_scale=3.0,
        prediction_type="v_prediction",
    ):
        """Sample from model using Euler integration with dual CFG or self-guidance.
        
        Supports:
        - v_prediction: model outputs velocity v, integrate directly
        - x_prediction: model outputs x0, derive velocity v = (zt - x0) / t
        
        Args:
            model: NanoDiT model
            shape: (B, C, H, W) output shape
            dino_emb: (B, 1024) DINOv3 CLS embeddings
            dino_patches: (B, num_patches, 1024) DINOv3 spatial patches
            text_emb: (B, 500, 1024) T5 hidden states
            text_mask: (B, 500) T5 attention mask
            device: torch device
            text_scale: CFG scale for text conditioning (dual CFG mode)
            dino_scale: CFG scale for DINO conditioning (dual CFG mode)
            self_guidance: if True, use TREAD self-guidance instead of dual CFG
            guidance_scale: self-guidance scale (self-guidance mode only)
            prediction_type: "v_prediction" or "x_prediction"
        
        Returns:
            output: (B, C, H, W) sampled data (latents or pixels)
        """
        B = shape[0]
        
        # Start from pure noise (t=1.0)
        zt = torch.randn(shape, device=device)
        
        timesteps = self.timesteps.to(device)
        
        for i in range(self.num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr  # Negative (moving from 1.0 to 0.0)
            
            t_batch = torch.full((B,), t_curr, device=device)
            
            if self_guidance:
                # Self-guidance: 2 passes (dense vs routed conditional)
                # Pass 1: Dense (all tokens, conditional, no routing)
                v_dense = model(
                    zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
                    tread_enabled=False,
                )
                
                # Pass 2: Routed (50% tokens, conditional, with routing)
                v_routed = model(
                    zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
                    tread_enabled=True,
                )
                
                # Self-guidance combination
                v_pred = v_routed + guidance_scale * (v_dense - v_routed)
            else:
                # Dual CFG: 3 passes (unconditional, text-only, DINO-only)
                # 1. Unconditional (all dropped)
                v_uncond = model(
                    zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
                    cfg_drop_text=torch.ones(B, dtype=torch.bool, device=device),
                    cfg_drop_dino_cls=torch.ones(B, dtype=torch.bool, device=device),
                    cfg_drop_dino_patches=torch.ones(B, dtype=torch.bool, device=device),
                )
                
                # 2. Text-only (DINO CLS and patches dropped)
                v_text = model(
                    zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
                    cfg_drop_text=torch.zeros(B, dtype=torch.bool, device=device),
                    cfg_drop_dino_cls=torch.ones(B, dtype=torch.bool, device=device),
                    cfg_drop_dino_patches=torch.ones(B, dtype=torch.bool, device=device),
                )
                
                # 3. DINO-only (text dropped, DINO CLS and patches kept)
                v_dino = model(
                    zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
                    cfg_drop_text=torch.ones(B, dtype=torch.bool, device=device),
                    cfg_drop_dino_cls=torch.zeros(B, dtype=torch.bool, device=device),
                    cfg_drop_dino_patches=torch.zeros(B, dtype=torch.bool, device=device),
                )
                
                # Dual CFG combination
                v_pred = v_uncond + text_scale * (v_text - v_uncond) + dino_scale * (v_dino - v_uncond)
            
            # Derive velocity for Euler integration
            if prediction_type == "x_prediction":
                # Model outputs x0_pred; derive velocity: v = (zt - x0) / t
                t_val = t_curr.clamp(min=0.05)
                v_pred_euler = (zt - v_pred) / t_val
            else:
                v_pred_euler = v_pred
            
            # Euler integration: z_{t+dt} = z_t + v * dt
            # dt is negative, v points toward noise, so we move toward data
            zt = zt + v_pred_euler * dt
        
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
        
        # Enable slicing and tiling to save memory
        vae.enable_slicing()
        vae.enable_tiling()
        
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
        latents: (B, 16, H, W) latent tensors (normalized)
    
    Returns:
        images: (B, 3, H*8, W*8) RGB images in [-1, 1]
    """
    from .data import denormalize_vae_latent
    
    # Flux VAE uses 8x spatial compression
    # latents: (B, 16, 64, 64) -> images: (B, 3, 512, 512)
    
    # Denormalize latents before decoding (if normalization is enabled)
    latents = denormalize_vae_latent(latents)
    
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
        self_guidance=False,
        guidance_scale=3.0,
        prediction_type="v_prediction",
    ):
        """
        Args:
            model: NanoDiT model
            vae: VAE decoder (can be None for pixel-space)
            device: torch device
            num_steps: number of sampling steps
            text_scale: CFG scale for text (dual CFG mode)
            dino_scale: CFG scale for DINO (dual CFG mode)
            self_guidance: use self-guidance instead of dual CFG
            guidance_scale: self-guidance scale
            prediction_type: "v_prediction" or "x_prediction"
        """
        self.model = model
        self.vae = vae
        self.device = device
        self.sampler = EulerSampler(num_steps=num_steps)
        self.text_scale = text_scale
        self.dino_scale = dino_scale
        self.self_guidance = self_guidance
        self.guidance_scale = guidance_scale
        self.prediction_type = prediction_type
    
    @torch.no_grad()
    def generate(
        self,
        dino_emb,
        dino_patches,
        text_emb,
        text_mask,
        latent_size=None,
        batch_size=None,
        text_scale=None,
        dino_scale=None,
        self_guidance=None,
        guidance_scale=None,
    ):
        """Generate images from conditioning.
        
        Args:
            dino_emb: (B, 1024) DINOv3 CLS embeddings
            dino_patches: (B, num_patches, 1024) DINOv3 spatial patches (variable length!)
            text_emb: (B, 500, 1024) T5 hidden states
            text_mask: (B, 500) T5 attention mask
            latent_size: spatial size of latents (64 = 512x512 images)
            batch_size: override batch size (default: from embeddings)
            text_scale: override text CFG scale (dual CFG mode, default: use sampler's)
            dino_scale: override dino CFG scale (dual CFG mode, default: use sampler's)
            self_guidance: override self-guidance mode (default: use sampler's)
            guidance_scale: override guidance scale (default: use sampler's)
        
        Returns:
            images: (B, 3, 512, 512) RGB images
        """
        self.model.eval()
        
        if batch_size is None:
            batch_size = dino_emb.shape[0]
        
        # Ensure everything is on correct device
        dino_emb = dino_emb.to(self.device)
        dino_patches = dino_patches.to(self.device)
        text_emb = text_emb.to(self.device)
        text_mask = text_mask.to(self.device)
        
        if latent_size is None:
            latent_size = getattr(self.model, 'input_size', 64)

        use_self_guidance = self_guidance if self_guidance is not None else self.self_guidance
        
        # Determine shape based on prediction type
        pixel_space = self.prediction_type == "x_prediction"
        in_channels = 3 if pixel_space else 16
        if pixel_space:
            # For pixel-space, latent_size is the pixel dimension (e.g., 1024)
            # If it looks like a latent size (small), scale up
            spatial_size = latent_size if latent_size > 64 else latent_size * 8
        else:
            spatial_size = latent_size

        # Sample
        shape = (batch_size, in_channels, spatial_size, spatial_size)
        if use_self_guidance:
            output = self.sampler.sample(
                self.model, shape, dino_emb, dino_patches, text_emb, text_mask,
                device=self.device,
                self_guidance=True,
                guidance_scale=guidance_scale if guidance_scale is not None else self.guidance_scale,
                prediction_type=self.prediction_type,
            )
        else:
            cfg_text_scale = text_scale if text_scale is not None else self.text_scale
            cfg_dino_scale = dino_scale if dino_scale is not None else self.dino_scale
            output = self.sampler.sample(
                self.model, shape, dino_emb, dino_patches, text_emb, text_mask,
                device=self.device,
                text_scale=cfg_text_scale, dino_scale=cfg_dino_scale,
                prediction_type=self.prediction_type,
            )
        
        if pixel_space:
            # Output is already RGB [0,1] — clamp and return
            images = output.clamp(0, 1)
        else:
            # Decode latents to images via VAE
            images = decode_latents(self.vae, output)
        
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
