"""Check if training data has compositional diversity or bias."""
import torch
from pathlib import Path
from production.config_loader import load_config
from production.data import get_production_dataloader

config = load_config('production/config.yaml')
loader = get_production_dataloader(config, device='cpu')

print("Sampling 50 images from training data...")
print("Computing spatial variance (high = diverse, low = repeated pattern)")

variances = []
for i, batch in enumerate(loader):
    if i >= 50:
        break
    
    # Get VAE latent (B, 16, 64, 64)
    latent = batch['vae_latent'][0]  # First in batch
    
    # Compute spatial variance (how much does each position vary?)
    # High variance = diverse content at that position
    # Low variance = similar content at that position
    spatial_var = latent.var(dim=0).mean().item()  # Average across channels
    variances.append(spatial_var)
    
    if i < 5:
        print(f"  Sample {i}: spatial_var = {spatial_var:.6f}")

import numpy as np
variances = np.array(variances)
print(f"\nSpatial variance stats:")
print(f"  Mean: {variances.mean():.6f}")
print(f"  Std:  {variances.std():.6f}")
print(f"  Min:  {variances.min():.6f}")
print(f"  Max:  {variances.max():.6f}")

# Check if samples are identical
print(f"\nCoefficient of variation: {variances.std() / variances.mean():.4f}")
print("  (>0.2 = good diversity, <0.05 = possible repetition)")

# Also check image_ids for uniqueness
print("\nChecking image_id uniqueness...")
ids = []
for i, batch in enumerate(loader):
    if i >= 50:
        break
    ids.append(batch['image_ids'][0])

unique_ids = len(set(ids))
print(f"Unique image_ids: {unique_ids}/50")
if unique_ids == 50:
    print("✅ All different images")
elif unique_ids < 25:
    print(f"❌ Only {unique_ids} unique images - REPETITION DETECTED")
    print(f"   Repeated IDs: {[id for id in ids if ids.count(id) > 1][:10]}")
else:
    print(f"⚠️  Some repetition: {unique_ids} unique / 50 total")
