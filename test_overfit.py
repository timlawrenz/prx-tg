"""Test if model can overfit on a tiny dataset."""
import torch
from pathlib import Path
from production.config_loader import load_config
from production.model import NanoDiT
from production.data import get_production_dataloader
from production.sample import ValidationSampler
import lpips
lpips_fn = lpips.LPIPS(net="alex")
import torch.nn.functional as F

print("="*60)
print("OVERFITTING TEST")
print("="*60)
print("Goal: Prove model can memorize by training on just 50 images")
print()

# Load model
config = load_config('production/config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NanoDiT(
    hidden_size=config.model.hidden_size,
    depth=config.model.depth,
    num_heads=config.model.num_heads,
    patch_size=config.model.patch_size,
).to(device)

# Load checkpoint
ckpt_path = 'checkpoints/checkpoint_step0020000.pt'
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
print(f"Loaded step: {ckpt['step']}")
print()

# Get ONE sample from training data
loader = get_production_dataloader(config, device=device)
sample = next(iter(loader))

print("Test sample:")
print(f"  VAE latent: {sample['vae_latent'].shape}")
print(f"  DINO patches: {sample['dinov3_patches'].shape}")
print(f"  Caption: {sample['captions'][0][:80]}...")
print()

# Reconstruct this EXACT sample
print("Reconstructing training sample (should be perfect if memorized)...")
sampler = ValidationSampler(
    model=model,
    num_steps=config.sampling.num_steps,
    text_scale=config.sampling.text_scale,
    dino_scale=config.sampling.dino_scale,
    device=device
)

# Get conditioning
dino_emb = sample['dino_embedding'].to(device)
dino_patches = sample['dinov3_patches'].to(device)
text_emb = sample['t5_hidden'].to(device)
text_mask = sample['t5_mask'].to(device)
caption = sample['captions'][0]

# Generate
model.eval()
with torch.no_grad():
    generated = sampler.generate(
        batch_size=1,
        dino_emb=dino_emb,
        dino_patches=dino_patches,
        text_emb=text_emb,
        text_mask=text_mask,
        height=sample['vae_latent'].shape[2],
        width=sample['vae_latent'].shape[3],
    )

# Compute LPIPS
original = sample['vae_latent'].to(device)
lpips_score = lpips_fn(original, generated).item()

print(f"LPIPS: {lpips_score:.4f}")
print()
print("Expected:")
print("  LPIPS < 0.3: Model memorized this sample ✅")
print("  LPIPS > 0.8: Model NOT memorizing ❌")
print()

if lpips_score < 0.3:
    print("✅ SUCCESS: Model can memorize!")
    print("   Problem is likely: need more data, longer training, or better generalization")
elif lpips_score > 0.8:
    print("❌ FAILURE: Model NOT learning properly!")
    print("   Possible causes:")
    print("   1. Sampling process broken (forward/backward mismatch)")
    print("   2. VAE encoding/decoding mismatch")
    print("   3. Conditioning not being used")
    print("   4. Flow matching implementation bug")
    
# Save comparison
from torchvision.utils import save_image
output_dir = Path('test_overfit_output')
output_dir.mkdir(exist_ok=True)

# Denormalize from [-1, 1] to [0, 1] if needed
def denorm(x):
    return (x + 1) / 2

save_image(denorm(original[0]), output_dir / 'original.png')
save_image(denorm(generated[0]), output_dir / 'reconstructed.png')
print(f"\nSaved comparison to: {output_dir}/")
print(f"  original.png")
print(f"  reconstructed.png")
