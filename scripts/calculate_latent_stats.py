"""Calculate Mean and Std for Flux VAE latents in WebDataset shards."""

import argparse
from pathlib import Path
import numpy as np
import webdataset as wds
from tqdm import tqdm


def calculate_stats(shard_dir, num_samples=1000):
    """Calculate mean and std of VAE latents.
    
    Uses global statistics (single scalar mean/std) to preserve channel
    relationships. This is the standard approach for latent diffusion models.
    
    Args:
        shard_dir: Path to directory containing bucket_*/shard-*.tar files
        num_samples: Number of samples to process for statistics
    """
    path = Path(shard_dir)
    # Find all .tar files across all buckets
    shards = list(path.glob('bucket_*/shard-*.tar'))
    if not shards:
        print(f"No shards found in {shard_dir}")
        print(f"Expected pattern: {shard_dir}/bucket_*/shard-*.tar")
        return

    print(f"Found {len(shards)} shards")
    print(f"Calculating stats on first {num_samples} samples...")
    print()
    
    # Create dataset
    dataset = (
        wds.WebDataset([str(s) for s in shards])
        .decode()
        .to_tuple("vae.npy")
    )
    
    # Accumulate pixel values
    # We compute global scalar mean/std to preserve channel relationships
    # (standard practice for latent diffusion models like Stable Diffusion)
    pixels = []
    
    count = 0
    for (vae,) in tqdm(dataset, desc="Loading samples"):
        # vae is (16, H, W) numpy array
        # Flatten to (16 * H * W)
        pixels.append(vae.flatten())
        count += 1
        if count >= num_samples:
            break
            
    if not pixels:
        print("No samples found.")
        return

    # Concatenate all pixels from all images
    # shape: (N_samples * 16 * H * W, )
    print("\nConcatenating data...")
    all_pixels = np.concatenate(pixels)
    
    print("Computing statistics...")
    # Convert to float64 to avoid overflow in variance computation
    # (sum of squares can overflow with 500M+ values in float32/float16)
    all_pixels_f64 = all_pixels.astype(np.float64)
    mean = np.mean(all_pixels_f64)
    std = np.std(all_pixels_f64)
    
    print("\n" + "="*60)
    print("FLUX VAE LATENT STATISTICS")
    print("="*60)
    print(f"Samples processed: {count}")
    print(f"Total values: {len(all_pixels):,}")
    print(f"Min value: {all_pixels.min():.6f}")
    print(f"Max value: {all_pixels.max():.6f}")
    print("-" * 60)
    print(f"FLUX_LATENT_MEAN = {mean:.6f}")
    print(f"FLUX_LATENT_STD = {std:.6f}")
    print("-" * 60)
    print("\nTo use these values, add to production/data.py:")
    print()
    print("# Flux VAE latent normalization")
    print(f"FLUX_LATENT_MEAN = {mean:.6f}")
    print(f"FLUX_LATENT_STD = {std:.6f}")
    print()
    print("def normalize_vae_latent(latent):")
    print("    '''Normalize VAE latent to zero mean, unit variance.'''")
    print("    return (latent - FLUX_LATENT_MEAN) / FLUX_LATENT_STD")
    print()
    print("def denormalize_vae_latent(latent):")
    print("    '''Denormalize VAE latent back to original scale.'''")
    print("    return latent * FLUX_LATENT_STD + FLUX_LATENT_MEAN")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mean and std for Flux VAE latents"
    )
    parser.add_argument(
        "shard_dir",
        help="Path to shard directory (e.g. data/shards/4000 or data/shards/validation)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of samples to process (default: 1000)"
    )
    args = parser.parse_args()
    
    calculate_stats(args.shard_dir, args.n)
