"""FLUX-AE decoder for latent-space validation/sampling (latent-first arms).

The on-disk AE at paths.vae_path (/mnt/models/vae/ae.safetensors) is a
comfy-format **AutoencodingEngine** (raw 16-ch latents, NO quant/post_quant
convs, LDM-style key names). diffusers cannot load it directly:
  - AutoencoderKL.from_single_file silently mis-maps weights (MSE ~0.25 at
    every latent scale) because the key names don't line up without the
    converter.
  - The standard AutoencoderKL architecture insists on post_quant_conv, which
    this AE does not have — decoding raw latents through a randomly-init
    post_quant_conv destroys the output (mean ~0, std ~0.4 vs src ~0.5/0.2).

Working recipe (verified 2026-09-12, decode MSE 0.0001 vs source pixel.npy):
  1. AutoencoderKL.from_config(FLUX vae config from HF repo config.json)
  2. Load on-disk ae.safetensors, inject IDENTITY quant_conv/post_quant_conv
     (the AE is raw-latent; identity convs reproduce the no-op the comfy
     AutoencodingEngine has), convert LDM->diffusers via
     diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_vae_checkpoint
  3. vae.decode(z) returns [-1,1]; map to [0,1] with (x+1)/2 for pixel-space
     consumers (tensor_to_pil, LPIPS, validation collages).

NOTE on channels: this AE's encoder conv_out is (32, 512, 3, 3) — a 32-ch
encoder output split via the double_z convention. The LATENT is 16 channels
(decoder conv_in (512, 16, 3, 3), encode produced (16,128,128)). Identity
quant conv is (16,16,1,1).
"""
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint
from safetensors.torch import load_file


def _flux_vae_config() -> dict:
    """Return the FLUX.1-dev VAE config (block_out_channels [128,256,512,512] etc).

    Loaded from the HF repo config.json (small, ~1KB — no weight download).
    Falls back to a hardcoded copy if HF is unreachable.
    """
    hardcoded = {
        "_class_name": "AutoencoderKL",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": ["DownEncoderBlock2D"] * 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 16,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "sample_size": 1024,
        "scaling_factor": 1.0,
        "force_upcast": False,
        "mid_block_add_attention": True,
    }
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("black-forest-labs/FLUX.1-dev", "vae/config.json")
        cfg = json.load(open(p))
        cfg["_class_name"] = "AutoencoderKL"
        return cfg
    except Exception:
        return hardcoded


def load_flux_ae_decoder(device: str = "cuda", vae_path: str = "/mnt/models/vae/ae.safetensors"):
    """Load the on-disk FLUX AE as a diffusers AutoencoderKL decoder (raw latents).

    Returns a VAE whose .decode(z) maps (B,16,H,W) latent -> (B,3,8H,8W) in [-1,1].
    Identity quant/post_quant convs reproduce the raw-latent AutoencodingEngine.
    """
    cfg = _flux_vae_config()
    vae = AutoencoderKL.from_config(cfg)
    sd = load_file(vae_path)

    zch = cfg.get("latent_channels", 16)
    # Inject identity quant/post-quant convs (the AE has none — raw latents).
    # diffusers architecture REQUIRES these layers; identity == no-op == the
    # comfy AutoencodingEngine behaviour (DiagGaussian regularizer = pass-through).
    sd["quant_conv.weight"] = torch.eye(zch).reshape(zch, zch, 1, 1).float()
    sd["quant_conv.bias"] = torch.zeros(zch).float()
    sd["post_quant_conv.weight"] = torch.eye(zch).reshape(zch, zch, 1, 1).float()
    sd["post_quant_conv.bias"] = torch.zeros(zch).float()

    renamed = convert_ldm_vae_checkpoint(sd, cfg)
    missing, unexpected = vae.load_state_dict(renamed, strict=False)
    # quant/post_quant appear as 'unexpected' if the config didn't create them,
    # or as part of the arch. If 'missing' is non-empty something is structurally off.
    if missing:
        raise RuntimeError(f"FLUX AE decoder missing {len(missing)} keys: {missing[:5]}")

    vae.eval()
    vae.to(device)
    return vae


@torch.no_grad()
def decode_latents(vae, latents):
    """Decode FLUX latents (B,16,H,W) -> images (B,3,8H,8W) in [-1,1] (consumer range)."""
    latents = latents.to(vae.device, dtype=vae.dtype)
    out = vae.decode(latents).sample  # (B,3,H*8,W*8) in [-1,1]
    return out


if __name__ == "__main__":
    # Quick self-test: decode one real latent, compare to source pixel.npy.
    import sys
    latent = np.load("/mnt/nas-ai-models/training-data/ffhq/stratum/00042/flux_latent.npy")
    src = np.load("/mnt/nas-ai-models/training-data/ffhq/stratum/00042/pixel.npy").astype(np.float32)
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    vae = load_flux_ae_decoder(device=dev)
    z = torch.from_numpy(latent[:, :64, :64]).unsqueeze(0).float()
    out = decode_latents(vae, z)
    im = ((out[0].cpu().numpy() + 1.0) / 2.0)
    src_crop = src[:, :512, :512]
    print("out", tuple(out.shape), "mean", round(float(im.mean()), 3), "std", round(float(im.std()), 3))
    print("src ", tuple(src_crop.shape), "mean", round(float(src_crop.mean()), 3), "std", round(float(src_crop.std()), 3))
    print("mse ", round(float(((im - src_crop) ** 2).mean()), 5))
    from PIL import Image
    Image.fromarray((np.clip(im.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)).save("/tmp/flux_ae_decode_selftest.png")
    print("saved /tmp/flux_ae_decode_selftest.png")