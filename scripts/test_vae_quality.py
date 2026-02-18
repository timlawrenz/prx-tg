#!/usr/bin/env python3
"""
Test VAE reconstruction quality by decoding training latents directly.

This bypasses the DiT model to see if the VAE itself can produce sharp images.
If VAE reconstructions are blurry, that's your quality ceiling.

Usage:
    python scripts/test_vae_quality.py --checkpoint checkpoints/step_074000.pt --num_samples 16
"""

import argparse
import torch
import json
import tarfile
import io
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add production to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))


def load_vae(device):
    """Load Flux VAE"""
    from diffusers import AutoencoderKL
    
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device)
    vae.eval()
    return vae


def load_samples_from_shards(shard_dir: Path, num_samples: int):
    """Load samples from WebDataset shards"""
    samples = []
    
    # Find all tar files in any bucket subdirectory
    tar_files = list(shard_dir.glob('**/shard-*.tar'))
    if not tar_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")
    
    print(f"  Found {len(tar_files)} shard files")
    
    for tar_path in tar_files:
        if len(samples) >= num_samples:
            break
            
        with tarfile.open(tar_path, 'r') as tar:
            # Group files by key
            files_by_key = {}
            for member in tar.getmembers():
                if member.isfile():
                    # Extract key from filename (e.g., "abc123.json" -> "abc123")
                    # Handle extensions like ".vae.npy", ".t5h.npy", etc.
                    key = member.name.split('.')[0]
                    if key not in files_by_key:
                        files_by_key[key] = {}
                    files_by_key[key][member.name] = member
            
            # Process complete samples
            for key, files in files_by_key.items():
                if len(samples) >= num_samples:
                    break
                
                # Check if we have all required files
                json_file = None
                vae_file = None
                
                for filename in files.keys():
                    if filename.endswith('.json'):
                        json_file = filename
                    elif filename.endswith('.vae.npy'):
                        vae_file = filename
                
                if not (json_file and vae_file):
                    continue
                
                # Load the sample data
                sample = {'key': key}
                
                # Load metadata
                json_data = tar.extractfile(files[json_file]).read()
                sample['metadata'] = json.loads(json_data)
                
                # Load VAE latent
                vae_data = tar.extractfile(files[vae_file]).read()
                sample['vae_latent'] = np.load(io.BytesIO(vae_data))
                
                samples.append(sample)
    
    return samples[:num_samples]


def decode_latents(vae, latents, device):
    """Decode VAE latents to images"""
    with torch.no_grad():
        latents = latents.to(device)
        # Flux VAE expects latents in normalized range
        # Training code applies normalize_vae_latent which scales by 0.13025
        # Reverse it: latent_normalized * 0.13025 -> divide by 0.13025
        latents = latents / 0.13025
        
        images = vae.decode(latents).sample  # Returns [B, 3, H, W] in [-1, 1]
        # Convert to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        # Convert to numpy [B, H, W, 3]
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype(np.uint8)
    return images


def load_original_image(image_path: Path, target_size=None):
    """Load and optionally resize original image"""
    img = Image.open(image_path).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)


def create_comparison_grid(originals, reconstructions, grid_cols=4):
    """Create a grid showing original | VAE reconstruction pairs"""
    n_samples = len(originals)
    grid_rows = (n_samples + grid_cols - 1) // grid_cols
    
    # Each cell shows [original | reconstruction]
    cell_h, cell_w = originals[0].shape[:2]
    grid_h = grid_rows * cell_h
    grid_w = grid_cols * cell_w * 2  # *2 for side-by-side
    
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    for idx in range(n_samples):
        row = idx // grid_cols
        col = idx % grid_cols
        
        y = row * cell_h
        x_orig = col * cell_w * 2
        x_recon = x_orig + cell_w
        
        # Place original and reconstruction side by side
        grid[y:y+cell_h, x_orig:x_orig+cell_w] = originals[idx]
        grid[y:y+cell_h, x_recon:x_recon+cell_w] = reconstructions[idx]
    
    return grid


def main():
    parser = argparse.ArgumentParser(description='Test VAE reconstruction quality')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to test')
    parser.add_argument('--shard_dir', type=str, default='data/shards/5000', help='Directory containing shard files')
    parser.add_argument('--output_dir', type=str, default='validation_outputs', help='Output directory')
    parser.add_argument('--grid_cols', type=int, default=4, help='Columns in output grid')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load VAE
    print("Loading VAE...")
    vae = load_vae(device=device)
    vae.eval()
    
    # Load samples from shards
    shard_dir = Path(args.shard_dir)
    print(f"\nLoading {args.num_samples} samples from {shard_dir}")
    samples = load_samples_from_shards(shard_dir, args.num_samples)
    
    if not samples:
        print("ERROR: No samples found!")
        return
    
    print(f"Loaded {len(samples)} samples")
    
    originals = []
    reconstructions = []
    
    print("\nDecoding latents...")
    for sample in tqdm(samples):
        # Get latent
        latent = sample['vae_latent']  # [16, H, W]
        latent = torch.from_numpy(latent).unsqueeze(0)  # [1, 16, H, W]
        
        # Decode
        recon_images = decode_latents(vae, latent, device)  # [1, H*8, W*8, 3]
        recon_img = recon_images[0]
        
        # Load original image (resize to match reconstruction)
        metadata = sample['metadata']
        orig_path = Path(metadata['image_path'])
        
        if not orig_path.exists():
            print(f"  Warning: original not found: {orig_path}, using black placeholder")
            orig_img = np.zeros_like(recon_img)
        else:
            orig_img = load_original_image(orig_path, target_size=(recon_img.shape[1], recon_img.shape[0]))
        
        originals.append(orig_img)
        reconstructions.append(recon_img)
    
    # Create comparison grid
    print(f"\nCreating comparison grid ({len(originals)} samples)...")
    grid = create_comparison_grid(originals, reconstructions, grid_cols=args.grid_cols)
    
    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f'vae_quality_step{checkpoint.get("step", "unknown")}.png'
    Image.fromarray(grid).save(output_path)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Layout: Each row shows {args.grid_cols} pairs of [original | VAE reconstruction]")
    print(f"  If VAE reconstructions are blurry, that's your quality ceiling.")
    print(f"  If VAE reconstructions are sharp, the DiT is the bottleneck.")


if __name__ == '__main__':
    main()
