"""Dataloader for validation shards."""

import re
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import webdataset as wds


def swap_left_right(caption):
    """Swap 'left' and 'right' in caption for horizontal flip augmentation."""
    # Temporary placeholder to avoid double-swapping
    caption = re.sub(r'\bleft\b', '<LEFT>', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\bright\b', '<RIGHT>', caption, flags=re.IGNORECASE)
    caption = re.sub(r'<LEFT>', 'right', caption, flags=re.IGNORECASE)
    caption = re.sub(r'<RIGHT>', 'left', caption, flags=re.IGNORECASE)
    return caption


def resize_vae_latent(latent, target_size=64):
    """Resize VAE latent to target spatial size using bilinear interpolation.
    
    NOTE: This dataset stores VAE latents at their original resolution
    (variable sizes matching the input images). Bilinear interpolation is
    used to resize them to a uniform target_size for batching.
    
    While resizing VAE latents is "nonphysical" relative to the VAE manifold,
    it's a practical necessity for handling variable-resolution datasets.
    
    Args:
        latent: (C, H, W) numpy array
        target_size: int, target spatial size (default 64 for 512x512 images)
    
    Returns:
        resized: (C, target_size, target_size) torch tensor
    """
    # Convert to torch and add batch dimension
    latent_t = torch.from_numpy(latent).unsqueeze(0)  # (1, C, H, W)
    
    # Resize using bilinear interpolation if needed
    if latent_t.shape[2] != target_size or latent_t.shape[3] != target_size:
        latent_t = F.interpolate(
            latent_t,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
    
    return latent_t.squeeze(0)  # (C, H, W)


class ValidationDataset:
    """Dataset loader for WebDataset validation shards."""
    
    def __init__(
        self,
        shard_dir,
        batch_size=8,
        shuffle=True,
        flip_prob=0.5,
        target_latent_size=64,
    ):
        """
        Args:
            shard_dir: Path to validation shards (e.g., 'data/shards/validation')
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle between epochs
            flip_prob: Probability of horizontal flip augmentation
            target_latent_size: Target spatial size for VAE latents (64 = 512x512 image)
        """
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flip_prob = flip_prob
        self.target_latent_size = target_latent_size
        
        # Find all shard files across buckets
        self.shard_files = sorted(self.shard_dir.glob('bucket_*/shard-*.tar'))
        if not self.shard_files:
            raise ValueError(f"No shard files found in {shard_dir}")
        
        print(f"Found {len(self.shard_files)} shard files")
    
    def process_sample(self, sample):
        """Process a raw WebDataset sample into model inputs.
        
        Args:
            sample: dict with keys: __key__, json, dinov3.npy, vae.npy, t5h.npy, t5m.npy
        
        Returns:
            dict with keys: vae_latent, dino_embedding, t5_hidden, t5_mask, caption, image_id
        """
        # Parse metadata (webdataset already decoded JSON)
        metadata = sample['json']
        image_id = metadata['image_id']
        caption = metadata['caption']
        
        # Load embeddings (webdataset already decoded .npy files to numpy arrays)
        dino_emb = sample['dinov3.npy']  # (1024,)
        vae_latent = sample['vae.npy']  # (16, H, W)
        t5_hidden = sample['t5h.npy']  # (77, 1024)
        t5_mask = sample['t5m.npy']  # (77,)
        
        # Apply horizontal flip augmentation
        if random.random() < self.flip_prob:
            vae_latent = np.flip(vae_latent, axis=2).copy()  # Flip width dimension
            caption = swap_left_right(caption)
        
        # Resize VAE latent to target size
        vae_latent = resize_vae_latent(vae_latent, self.target_latent_size)
        
        # Convert to tensors
        return {
            'vae_latent': vae_latent.float(),  # (16, 64, 64)
            'dino_embedding': torch.from_numpy(dino_emb).float(),  # (1024,)
            't5_hidden': torch.from_numpy(t5_hidden).float(),  # (77, 1024)
            't5_mask': torch.from_numpy(t5_mask).long(),  # (77,)
            'caption': caption,
            'image_id': image_id,
        }
    
    def collate_fn(self, batch):
        """Collate batch of samples into batched tensors."""
        return {
            'vae_latent': torch.stack([s['vae_latent'] for s in batch]),
            'dino_embedding': torch.stack([s['dino_embedding'] for s in batch]),
            't5_hidden': torch.stack([s['t5_hidden'] for s in batch]),
            't5_mask': torch.stack([s['t5_mask'] for s in batch]),
            'captions': [s['caption'] for s in batch],
            'image_ids': [s['image_id'] for s in batch],
        }
    
    def create_dataloader(self):
        """Create WebDataset dataloader.
        
        If shuffle=True: Creates infinite dataloader with repeat() for training
        If shuffle=False: Creates finite dataloader for deterministic validation
        """
        # Convert Path objects to strings for webdataset
        shard_urls = [str(f) for f in self.shard_files]
        
        # Create WebDataset pipeline
        dataset = (
            wds.WebDataset(shard_urls, shardshuffle=1000 if self.shuffle else False)
            .decode()
            .map(self.process_sample)
            .batched(self.batch_size, collation_fn=self.collate_fn)
        )
        
        # Make infinite by repeating (only for training)
        if self.shuffle:
            dataset = dataset.repeat()
        
        return dataset
    
    def __iter__(self):
        """Iterate over batches."""
        dataloader = self.create_dataloader()
        return iter(dataloader)


def get_validation_dataloader(
    shard_dir='data/shards/validation',
    batch_size=8,
    shuffle=True,
    flip_prob=0.5,
    target_latent_size=64,
):
    """Convenience function to create validation dataloader for training.
    
    Args:
        shard_dir: Path to validation shards
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle between epochs
        flip_prob: Probability of horizontal flip augmentation
        target_latent_size: Target spatial size for VAE latents
    
    Returns:
        iterable dataloader yielding batches
    """
    dataset = ValidationDataset(
        shard_dir=shard_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        flip_prob=flip_prob,
        target_latent_size=target_latent_size,
    )
    return dataset


def get_deterministic_validation_dataloader(
    shard_dir='data/shards/validation',
    batch_size=1,
    target_latent_size=64,
):
    """Create deterministic validation dataloader for consistent testing.
    
    This dataloader:
    - Does NOT shuffle (stable sample ordering)
    - Does NOT flip (no augmentation)
    - Does NOT repeat (finite, single pass)
    
    Use this for validation tests where you need consistent sample indices
    across different training steps (reconstruction LPIPS, DINO swap, etc.)
    
    Args:
        shard_dir: Path to validation shards
        batch_size: Number of samples per batch (default 1 for validation)
        target_latent_size: Target spatial size for VAE latents
    
    Returns:
        iterable dataloader yielding batches
    """
    dataset = ValidationDataset(
        shard_dir=shard_dir,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: no shuffle for deterministic ordering
        flip_prob=0.0,   # CRITICAL: no augmentation for consistency
        target_latent_size=target_latent_size,
    )
    return dataset


def get_production_dataloader(config, device='cuda'):
    """Create production dataloader from config.
    
    Args:
        config: Config object from config_loader
        device: torch device (for logging purposes)
        
    Returns:
        iterable dataloader yielding batches
    """
    from .config_loader import Config
    
    # For now, use single bucket validation dataset
    # TODO Stage 3: implement multi-bucket sampling
    data_cfg = config.data
    training_cfg = config.training
    
    # Use first bucket for initial testing
    if data_cfg.buckets:
        first_bucket = data_cfg.buckets[0]
        shard_dir = f"{data_cfg.shard_base_dir}/{first_bucket}"
    else:
        shard_dir = data_cfg.shard_base_dir
    
    print(f"  Shard dir: {shard_dir}")
    print(f"  Batch size: {training_cfg.batch_size}")
    print(f"  Flip prob: {data_cfg.horizontal_flip_prob}")
    print(f"  TODO: Multi-bucket sampling (Stage 3)")
    
    dataset = ValidationDataset(
        shard_dir=shard_dir,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        flip_prob=data_cfg.horizontal_flip_prob,
        target_latent_size=config.model.input_size,
    )
    
    return dataset


if __name__ == "__main__":
    # Test dataloader
    print("Testing validation dataloader...")
    
    dataloader = get_validation_dataloader(
        shard_dir='data/shards/validation',
        batch_size=4,
        shuffle=True,
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"VAE latent shape: {batch['vae_latent'].shape}")
    print(f"DINO embedding shape: {batch['dino_embedding'].shape}")
    print(f"T5 hidden shape: {batch['t5_hidden'].shape}")
    print(f"T5 mask shape: {batch['t5_mask'].shape}")
    print(f"Number of captions: {len(batch['captions'])}")
    print(f"Sample caption: {batch['captions'][0][:80]}...")
    print("✓ Dataloader test passed")
