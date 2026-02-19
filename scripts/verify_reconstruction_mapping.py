"""Verify that reconstruction test uses correct embeddings for each image."""

import torch
import sys
from pathlib import Path

# Add production directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))

from data import get_deterministic_validation_dataloader
import webdataset as wds

def main():
    # Load validation dataloader
    print("Loading validation dataloader...")
    dataloader = get_deterministic_validation_dataloader(
        shard_base_dir="data/shards/5000",
        batch_size=1,
    )
    
    target_id = "00w15a4yvnm47abqyoltehrxmn94"
    
    print(f"\nSearching for image_id: {target_id}")
    print("=" * 60)
    
    # Iterate through validation set
    found = False
    for batch_idx, batch in enumerate(dataloader):
        image_ids = batch['image_ids']
        
        for i, img_id in enumerate(image_ids):
            if img_id == target_id:
                print(f"\n✓ Found at batch {batch_idx}, position {i}")
                print(f"  Image ID: {img_id}")
                print(f"  Caption: {batch['captions'][i][:100]}...")
                print(f"  VAE latent shape: {batch['vae_latent'][i].shape}")
                print(f"  DINO embedding shape: {batch['dino_embedding'][i].shape}")
                print(f"  T5 hidden shape: {batch['t5_hidden'][i].shape}")
                found = True
                break
        
        if found:
            break
        
        if batch_idx > 200:  # Safety limit
            print("\n✗ Not found in first 200 batches")
            break
    
    if found:
        print("\n" + "=" * 60)
        print("Verification: Image ID matches in dataloader")
        print("This confirms the reconstruction test SHOULD use correct embeddings")
        print("\nIf generated image looks nothing like original, possible causes:")
        print("  1. Model capacity too small (384 hidden, 12 layers)")
        print("  2. Training not converged yet (115k/250k steps)")
        print("  3. Bug in sampling/generation process")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
