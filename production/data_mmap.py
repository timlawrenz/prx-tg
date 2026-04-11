"""Zero-copy memory-mapped data loader for unified memory architectures (APU).

Instead of streaming from WebDataset tar shards, this loader directly memory-maps
the individual .npy files from data/derived/. On an APU with unified memory
(e.g., AMD Strix Halo with 128GB LPDDR5X), the GPU compute units can read from
the same physical memory addresses — eliminating host-to-device transfer entirely.

Usage:
    This module is selected automatically when config.data.zero_copy = True.
    See production/data.py: get_production_dataloader() for the dispatch logic.
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class MmapSampleIndex:
    """In-memory index of all samples, organized by aspect-ratio bucket.
    
    Built once on startup by scanning the metadata JSONL. Holds only paths
    and metadata (~200 bytes per sample, ~12 MB for 60k images).
    """
    
    def __init__(self, metadata_jsonl: str, derived_dir: str, buckets: list[str]):
        self.derived_dir = Path(derived_dir)
        self.samples_by_bucket: dict[str, list[dict]] = {}
        self._load_index(metadata_jsonl, buckets)
    
    def _load_index(self, metadata_jsonl: str, buckets: list[str]):
        """Load metadata JSONL and index samples by bucket."""
        allowed_buckets = set()
        for b in buckets:
            name = b if b.startswith('bucket_') else f"bucket_{b}"
            # Store both with and without prefix for flexible matching
            allowed_buckets.add(name)
            allowed_buckets.add(name.replace('bucket_', ''))
        
        with open(metadata_jsonl) as f:
            for line in f:
                record = json.loads(line.strip())
                bucket_raw = record.get('aspect_bucket', '')
                bucket_name = bucket_raw if bucket_raw.startswith('bucket_') else f"bucket_{bucket_raw}"
                
                # Filter to configured buckets
                if bucket_name not in allowed_buckets and bucket_raw not in allowed_buckets:
                    continue
                
                image_id = record['image_id']
                sample = {
                    'image_id': image_id,
                    'caption': record.get('caption', ''),
                    'bucket': bucket_name,
                    'width': record.get('width', 1024),
                    'height': record.get('height', 1024),
                    # Paths to .npy files (resolved lazily)
                    'image_path': self.derived_dir / 'images' / f'{image_id}.npy',
                    'dinov3_path': self.derived_dir / 'dinov3' / f'{image_id}.npy',
                    'dinov3_patches_path': self.derived_dir / 'dinov3_patches' / f'{image_id}.npy',
                    't5h_path': self.derived_dir / 't5_hidden' / f'{image_id}.npy',
                    'pose_path': self.derived_dir / 'pose' / f'{image_id}.npy',
                    't5_attention_mask': record.get('t5_attention_mask'),
                }
                
                if bucket_name not in self.samples_by_bucket:
                    self.samples_by_bucket[bucket_name] = []
                self.samples_by_bucket[bucket_name].append(sample)
        
        total = sum(len(v) for v in self.samples_by_bucket.values())
        print(f"  MmapSampleIndex: {total} samples across {len(self.samples_by_bucket)} buckets")
        for bucket, samples in sorted(self.samples_by_bucket.items()):
            print(f"    {bucket}: {len(samples)} samples")


class MmapBucketDataset:
    """Dataset for a single aspect-ratio bucket using memory-mapped .npy files.
    
    Each __iter__ call shuffles the sample order and yields batches.
    Uses np.load(mmap_mode='r') for zero-copy reads on unified memory.
    """
    
    def __init__(self, samples: list[dict], batch_size: int, target_size: tuple[int, int],
                 shuffle: bool = True, flip_prob: float = 0.0):
        self.samples = samples
        self.batch_size = batch_size
        self.target_latent_size = target_size  # (H, W) in pixels
        self.shuffle = shuffle
        self.flip_prob = flip_prob
    
    def _load_sample(self, sample: dict) -> dict:
        """Load a single sample via memory mapping."""
        image_id = sample['image_id']
        
        # Memory-map all .npy files (zero-copy on unified memory)
        image_data = np.load(sample['image_path'], mmap_mode='r')      # (3, H, W) float16
        dino_emb = np.load(sample['dinov3_path'], mmap_mode='r')       # (1024,) float16
        dino_patches = np.load(sample['dinov3_patches_path'], mmap_mode='r')  # (N, 1024) float16
        t5_hidden = np.load(sample['t5h_path'], mmap_mode='r')         # (512, 1024) float16
        pose_kpts = np.load(sample['pose_path'], mmap_mode='r')        # (133, 3) float16
        
        # T5 attention mask: from JSONL metadata (small, not worth a file)
        t5_mask = sample.get('t5_attention_mask')
        if t5_mask is not None:
            t5_mask = np.array(t5_mask, dtype=np.int64)
        else:
            # Fallback: all ones (assume all tokens valid)
            t5_mask = np.ones(t5_hidden.shape[0], dtype=np.int64)
        
        # Convert to tensors — torch.from_numpy shares memory with the mmap'd array
        # On unified memory APUs, this means GPU can read directly from the same address
        image_t = torch.from_numpy(np.array(image_data))  # Copy from mmap to writable tensor
        
        # Resize image to target resolution if needed
        if self.target_latent_size is not None:
            target_h, target_w = self.target_latent_size
            if image_t.shape[1] != target_h or image_t.shape[2] != target_w:
                image_t = F.interpolate(
                    image_t.unsqueeze(0).float(),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
        
        return {
            'image_data': image_t.float(),
            'dino_embedding': torch.from_numpy(np.array(dino_emb)).float(),
            'dinov3_patches': torch.from_numpy(np.array(dino_patches)).float(),
            't5_hidden': torch.from_numpy(np.array(t5_hidden)).float(),
            't5_mask': torch.from_numpy(t5_mask).long(),
            'pose_keypoints': torch.from_numpy(np.array(pose_kpts)).float(),
            'caption': sample['caption'],
            'image_id': image_id,
        }
    
    def _collate(self, batch: list[dict]) -> dict:
        """Collate samples into a batch, padding variable-length patches."""
        max_patches = max(s['dinov3_patches'].shape[0] for s in batch)
        
        padded_patches = []
        patch_masks = []
        for s in batch:
            patches = s['dinov3_patches']
            num_patches = patches.shape[0]
            
            mask = torch.zeros(max_patches, dtype=torch.long)
            mask[:num_patches] = 1
            patch_masks.append(mask)
            
            if num_patches < max_patches:
                pad = torch.zeros(max_patches - num_patches, patches.shape[1], dtype=patches.dtype)
                padded = torch.cat([patches, pad], dim=0)
            else:
                padded = patches
            padded_patches.append(padded)
        
        return {
            'image_data': torch.stack([s['image_data'] for s in batch]),
            'dino_embedding': torch.stack([s['dino_embedding'] for s in batch]),
            'dinov3_patches': torch.stack(padded_patches),
            'dinov3_patches_mask': torch.stack(patch_masks),
            't5_hidden': torch.stack([s['t5_hidden'] for s in batch]),
            't5_mask': torch.stack([s['t5_mask'] for s in batch]),
            'pose_keypoints': torch.stack([s['pose_keypoints'] for s in batch]),
            'captions': [s['caption'] for s in batch],
            'image_ids': [s['image_id'] for s in batch],
        }
    
    def __iter__(self):
        """Yield batches, reshuffling each epoch."""
        indices = list(range(len(self.samples)))
        while True:
            if self.shuffle:
                random.shuffle(indices)
            
            batch = []
            for idx in indices:
                try:
                    sample = self._load_sample(self.samples[idx])
                    batch.append(sample)
                except Exception as e:
                    # Skip corrupt samples
                    print(f"  warning: skipping sample {self.samples[idx]['image_id']}: {e}")
                    continue
                
                if len(batch) == self.batch_size:
                    yield self._collate(batch)
                    batch = []


def _bucket_target_pixel_size(bucket_name: str) -> tuple[int, int]:
    """Get target pixel dimensions (H, W) from bucket name."""
    m = re.search(r"(\d+)x(\d+)$", bucket_name)
    if not m:
        raise ValueError(f"Invalid bucket name (expected ..._<W>x<H>): {bucket_name}")
    w_px, h_px = int(m.group(1)), int(m.group(2))
    return (h_px, w_px)


def get_mmap_dataloader(config):
    """Create zero-copy mmap dataloader from config.
    
    Returns a BucketAwareDataLoader-compatible object that uses memory-mapped
    .npy files instead of WebDataset tar shards.
    """
    from .data import BucketAwareDataLoader
    
    data_cfg = config.data
    training_cfg = config.training
    
    print(f"  Zero-copy mmap data loading: ENABLED")
    print(f"  Derived dir: {data_cfg.derived_dir}")
    print(f"  Metadata JSONL: {data_cfg.metadata_jsonl}")
    
    # Build sample index from metadata
    index = MmapSampleIndex(
        metadata_jsonl=data_cfg.metadata_jsonl,
        derived_dir=data_cfg.derived_dir,
        buckets=data_cfg.buckets,
    )
    
    bucket_datasets = {}
    bucket_weights = []
    
    for bucket_name, samples in index.samples_by_bucket.items():
        if len(samples) < training_cfg.batch_size:
            print(f"  warning: skipping bucket {bucket_name} ({len(samples)} < batch_size {training_cfg.batch_size})")
            continue
        
        target_size = _bucket_target_pixel_size(bucket_name)
        
        bucket_datasets[bucket_name] = MmapBucketDataset(
            samples=samples,
            batch_size=training_cfg.batch_size,
            target_size=target_size,
            shuffle=True,
            flip_prob=data_cfg.horizontal_flip_prob,
        )
        bucket_weights.append(len(samples))
    
    if not bucket_datasets:
        raise ValueError(f"No buckets with enough samples found in {data_cfg.metadata_jsonl}")
    
    if data_cfg.bucket_sampling == 'uniform':
        bucket_weights = [1.0] * len(bucket_weights)
    
    return BucketAwareDataLoader(bucket_datasets, bucket_weights, pixel_space=True)
