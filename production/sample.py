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
        device='cuda',
        text_scale=3.0,
        dino_scale=2.0,
        self_guidance=False,
        guidance_scale=3.0,
        prediction_type="v_prediction",
        **adapter_kwargs,
    ):
        """Sample from model using Euler integration with dual CFG or self-guidance.
        
        Supports:
        - v_prediction: model outputs velocity v, integrate directly
        - x_prediction: model outputs x0, derive velocity v = (zt - x0) / t
        
        Args:
            model: NanoDiT model
            shape: (B, C, H, W) output shape
            device: torch device
            text_scale: CFG scale for text conditioning (dual CFG mode)
            dino_scale: CFG scale for DINO conditioning (dual CFG mode)
            self_guidance: if True, use TREAD self-guidance instead of dual CFG
            guidance_scale: self-guidance scale (self-guidance mode only)
            prediction_type: "v_prediction" or "x_prediction"
            **adapter_kwargs: forwarded to model() -> adapter.forward()
                StratumAdapter: dino_emb, text_emb, dino_patches, text_mask,
                    dino_patches_mask, pose_kpts, cfg_drop_dino, cfg_drop_text,
                    cfg_drop_dino_patches, cfg_drop_pose
                EidolonAdapter: identity_emb, geometry_emb,
                    cfg_drop_identity, cfg_drop_geometry
        
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
                    zt, t_batch,
                    tread_enabled=False,
                    **adapter_kwargs,
                )
                
                # Pass 2: Routed (50% tokens, conditional, with routing)
                v_routed = model(
                    zt, t_batch,
                    tread_enabled=True,
                    **adapter_kwargs,
                )
                
                # Self-guidance combination
                v_pred = v_routed + guidance_scale * (v_dense - v_routed)
            else:
                # Dual CFG: 3 passes (unconditional, text-only, DINO-only)
                # Construct 3 conditioning variants by overriding cfg_drop masks
                drop_all = torch.ones(B, dtype=torch.bool, device=device)
                keep_all = torch.zeros(B, dtype=torch.bool, device=device)
                
                # Build unconditional kwargs: drop everything
                uncond_kwargs = dict(adapter_kwargs)
                if 'cfg_drop_dino' in adapter_kwargs or 'dino_emb' in adapter_kwargs:
                    uncond_kwargs['cfg_drop_dino'] = drop_all
                    uncond_kwargs['cfg_drop_text'] = drop_all
                    uncond_kwargs['cfg_drop_dino_patches'] = drop_all
                if 'pose_kpts' in adapter_kwargs:
                    uncond_kwargs['cfg_drop_pose'] = drop_all
                if 'cfg_drop_identity' in adapter_kwargs or 'identity_emb' in adapter_kwargs:
                    uncond_kwargs['cfg_drop_identity'] = drop_all
                    uncond_kwargs['cfg_drop_geometry'] = drop_all
                
                # 1. Unconditional
                v_uncond = model(zt, t_batch, **uncond_kwargs)
                
                # Build text-only kwargs: keep text, drop DINO/pose/identity/geometry
                text_kwargs = dict(adapter_kwargs)
                if 'cfg_drop_dino' in adapter_kwargs or 'dino_emb' in adapter_kwargs:
                    text_kwargs['cfg_drop_text'] = keep_all
                    text_kwargs['cfg_drop_dino'] = drop_all
                    text_kwargs['cfg_drop_dino_patches'] = drop_all
                if 'pose_kpts' in adapter_kwargs:
                    text_kwargs['cfg_drop_pose'] = drop_all
                if 'cfg_drop_identity' in adapter_kwargs or 'identity_emb' in adapter_kwargs:
                    text_kwargs['cfg_drop_identity'] = drop_all
                    text_kwargs['cfg_drop_geometry'] = drop_all
                
                # 2. Text-only
                v_text = model(zt, t_batch, **text_kwargs)
                
                # Build DINO-only kwargs: keep DINO, drop text/pose/identity/geometry
                dino_kwargs = dict(adapter_kwargs)
                if 'cfg_drop_dino' in adapter_kwargs or 'dino_emb' in adapter_kwargs:
                    dino_kwargs['cfg_drop_text'] = drop_all
                    dino_kwargs['cfg_drop_dino'] = keep_all
                    dino_kwargs['cfg_drop_dino_patches'] = keep_all
                if 'pose_kpts' in adapter_kwargs:
                    dino_kwargs['cfg_drop_pose'] = drop_all
                if 'cfg_drop_identity' in adapter_kwargs or 'identity_emb' in adapter_kwargs:
                    dino_kwargs['cfg_drop_identity'] = drop_all
                    dino_kwargs['cfg_drop_geometry'] = drop_all
                
                # 3. DINO-only
                v_dino = model(zt, t_batch, **dino_kwargs)
                
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
        latent_size=None,
        batch_size=None,
        text_scale=None,
        dino_scale=None,
        self_guidance=None,
        guidance_scale=None,
        **adapter_kwargs,
    ):
        """Generate images from conditioning.
        
        Args:
            latent_size: spatial size of latents (64 = 512x512 images)
            batch_size: override batch size (default: from kwargs)
            text_scale: override text CFG scale (dual CFG mode, default: use sampler's)
            dino_scale: override dino CFG scale (dual CFG mode, default: use sampler's)
            self_guidance: override self-guidance mode (default: use sampler's)
            guidance_scale: override guidance scale (default: use sampler's)
            **adapter_kwargs: forwarded to model() -> adapter.forward()
                StratumAdapter: dino_emb, text_emb, dino_patches, text_mask,
                    dino_patches_mask, pose_kpts, cfg_drop_dino, cfg_drop_text,
                    cfg_drop_dino_patches, cfg_drop_pose
                EidolonAdapter: identity_emb, geometry_emb,
                    cfg_drop_identity, cfg_drop_geometry
        
        Returns:
            images: (B, 3, 512, 512) RGB images
        """
        self.model.eval()
        
        # Move all tensor values in adapter_kwargs to the correct device
        device_kwargs = {}
        for k, v in adapter_kwargs.items():
            if isinstance(v, torch.Tensor):
                device_kwargs[k] = v.to(self.device)
            else:
                device_kwargs[k] = v
        
        # Detect batch size from the first tensor-like adapter kwarg
        if batch_size is None:
            for v in device_kwargs.values():
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    batch_size = v.shape[0]
                    break
            if batch_size is None:
                raise ValueError("Cannot determine batch_size from adapter_kwargs — no tensor found")
        
        if latent_size is None:
            latent_size = getattr(self.model, 'input_size', 128)

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
                self.model, shape,
                device=self.device,
                self_guidance=True,
                guidance_scale=guidance_scale if guidance_scale is not None else self.guidance_scale,
                prediction_type=self.prediction_type,
                **device_kwargs,
            )
        else:
            cfg_text_scale = text_scale if text_scale is not None else self.text_scale
            cfg_dino_scale = dino_scale if dino_scale is not None else self.dino_scale
            output = self.sampler.sample(
                self.model, shape,
                device=self.device,
                text_scale=cfg_text_scale, dino_scale=cfg_dino_scale,
                prediction_type=self.prediction_type,
                **device_kwargs,
            )
        
        if pixel_space:
            # Output is RGB [0,1] from sampler — convert to [-1,1] for consistency
            # (tensor_to_pil, LPIPS, and all consumers expect [-1,1])
            images = output.clamp(0, 1) * 2 - 1
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
