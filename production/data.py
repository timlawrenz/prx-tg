"""Dataloader for pixel-space WebDataset shards."""

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


def resize_image(image, target_size=1024):
    """Resize image to target spatial size using bilinear interpolation.

    Args:
        image: (C, H, W) numpy array
        target_size: int or (H, W) tuple in pixels

    Returns:
        resized: (C, H, W) torch tensor
    """
    image_t = torch.from_numpy(image).unsqueeze(0)  # (1, C, H, W)

    if isinstance(target_size, tuple):
        target_h, target_w = target_size
    else:
        target_h, target_w = target_size, target_size

    if image_t.shape[2] != target_h or image_t.shape[3] != target_w:
        image_t = F.interpolate(
            image_t,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )

    return image_t.squeeze(0)  # (C, H, W)


class ValidationDataset:
    """Dataset loader for pixel-space WebDataset shards."""

    def __init__(
        self,
        shard_dir,
        batch_size=8,
        shuffle=True,
        flip_prob=0.5,
        target_latent_size=64,
        shard_files=None,
        deterministic=False,
    ):
        """
        Args:
            shard_dir: Path to shards (e.g., 'data/shards/faces7k/bucket_1024x1024')
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle between epochs
            flip_prob: Probability of horizontal flip augmentation
            target_latent_size: Target spatial size in pixels
            deterministic: If True, set seeds for reproducible sample ordering
        """
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flip_prob = flip_prob
        self.target_latent_size = target_latent_size

        # Set seeds for deterministic behavior
        if deterministic:
            import random
            import numpy as np
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)

        # Find shard files
        if shard_files is None:
            self.shard_files = sorted(self.shard_dir.glob('bucket_*/shard-*.tar'))
        else:
            self.shard_files = [Path(f) for f in shard_files]
            self.shard_files = sorted(self.shard_files)
        if not self.shard_files:
            raise ValueError(f"No shard files found in {shard_dir}")

        print(f"Found {len(self.shard_files)} shard files")

    def process_sample(self, sample):
        """Process a raw WebDataset sample into model inputs.

        Args:
            sample: dict with keys: __key__, json, dinov3.npy, dinov3_patches.npy,
                    image.npy, t5h.npy, t5m.npy, pose.npy

        Returns:
            dict with keys: image_data, dino_embedding, dinov3_patches, t5_hidden,
                            t5_mask, pose_keypoints, caption, image_id
        """
        # Parse metadata (webdataset already decoded JSON)
        metadata = sample['json']
        image_id = metadata['image_id']
        caption = metadata['caption']

        # Load embeddings (webdataset already decoded .npy files to numpy arrays)
        dino_emb = sample['dinov3.npy']         # (1024,)
        dino_patches = sample['dinov3_patches.npy']  # (num_patches, 1024) - variable length!
        t5_hidden = sample['t5h.npy']           # (512, 1024) - T5-XXL supports 512 tokens
        t5_mask = sample['t5m.npy']             # (512,)
        pose_kpts = sample['pose.npy']          # (133, 3) — [x_norm, y_norm, confidence] per joint

        # Load pixel-space RGB image
        image_data = sample['image.npy']        # (3, H, W) float16, range [0, 1]

        # Resize to target size if specified
        if self.target_latent_size is None:
            image_data = torch.from_numpy(image_data)
        else:
            image_data = resize_image(image_data, self.target_latent_size)

        # Convert to tensors
        return {
            'image_data': image_data.float(),                          # (3, H, W) RGB [0, 1]
            'dino_embedding': torch.from_numpy(dino_emb).float(),      # (1024,)
            'dinov3_patches': torch.from_numpy(dino_patches).float(),  # (num_patches, 1024) - VARIABLE!
            't5_hidden': torch.from_numpy(t5_hidden).float(),          # (512, 1024)
            't5_mask': torch.from_numpy(t5_mask).long(),               # (512,)
            'pose_keypoints': torch.from_numpy(pose_kpts).float(),     # (133, 3)
            'caption': caption,
            'image_id': image_id,
        }

    def collate_fn(self, batch):
        """Collate batch of samples into batched tensors.

        Pads variable-length dinov3_patches to the max length in the batch.
        """
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
            'dinov3_patches': torch.stack(padded_patches),       # (B, max_patches, 1024)
            'dinov3_patches_mask': torch.stack(patch_masks),     # (B, max_patches)
            't5_hidden': torch.stack([s['t5_hidden'] for s in batch]),
            't5_mask': torch.stack([s['t5_mask'] for s in batch]),
            'pose_keypoints': torch.stack([s['pose_keypoints'] for s in batch]),  # (B, 133, 3)
            'captions': [s['caption'] for s in batch],
            'image_ids': [s['image_id'] for s in batch],
        }

    def create_dataloader(self):
        """Create WebDataset dataloader.

        If shuffle=True: Creates infinite dataloader with repeat() for training
        If shuffle=False: Creates finite dataloader for deterministic validation
        """
        shard_urls = [str(f) for f in self.shard_files]

        dataset = (
            wds.WebDataset(shard_urls, shardshuffle=1000 if self.shuffle else False, handler=wds.warn_and_continue)
            .decode(handler=wds.warn_and_continue)
            .map(self.process_sample, handler=wds.warn_and_continue)
            .batched(self.batch_size, collation_fn=self.collate_fn, partial=False)
        )

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
        target_latent_size: Target spatial size in pixels

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
    source="webdataset",
    stratum_dir="/workspace/stratum",
):
    """Create deterministic validation dataloader for consistent testing.

    This dataloader:
    - Does NOT shuffle (stable sample ordering)
    - Does NOT flip (no augmentation)
    - Does NOT repeat (finite, single pass)
    - Sets seeds for reproducible sample selection

    Args:
        shard_dir: Path to validation shards
        batch_size: Number of samples per batch (default 1 for validation)
        target_latent_size: Target spatial size in pixels

    Returns:
        iterable dataloader yielding batches
    """
    if source == "stratum":
        from .data_stratum import StratumDataset
        print(f"  Validation data source: stratum")
        print(f"  Validation Stratum dir: {stratum_dir}")
        return StratumDataset(
            stratum_dir=stratum_dir,
            batch_size=batch_size,
            shuffle=False,  # deterministic
            target_latent_size=target_latent_size,
            max_samples=100,  # use first 100 for validation
        )
    
    dataset = ValidationDataset(
        shard_dir=shard_dir,
        batch_size=batch_size,
        shuffle=False,      # CRITICAL: no shuffle for deterministic ordering
        flip_prob=0.0,      # CRITICAL: no augmentation for consistency
        target_latent_size=target_latent_size,
        deterministic=True, # CRITICAL: set seeds for reproducible sampling
    )
    return dataset


class BucketAwareDataLoader:
    """Sample whole batches from a single aspect-ratio bucket."""

    def __init__(self, bucket_datasets, bucket_weights):
        self.bucket_datasets = bucket_datasets  # dict[name -> ValidationDataset]
        self.bucket_names = list(bucket_datasets.keys())
        self.bucket_weights = bucket_weights
        self._logged = set()
        self._resolution_scale = 1.0
        # Store original (full-res) target sizes per bucket
        self._base_target_sizes = {
            name: ds.target_latent_size for name, ds in bucket_datasets.items()
        }

    @property
    def resolution_scale(self):
        return self._resolution_scale

    @resolution_scale.setter
    def resolution_scale(self, scale):
        """Update resolution scale, adjusting target size on all bucket datasets."""
        if scale == self._resolution_scale:
            return
        self._resolution_scale = scale
        for name, ds in self.bucket_datasets.items():
            base = self._base_target_sizes[name]
            if isinstance(base, tuple):
                h = max(32, (int(base[0] * scale) // 32) * 32)
                w = max(32, (int(base[1] * scale) // 32) * 32)
                ds.target_latent_size = (h, w)
            else:
                ds.target_latent_size = max(32, (int(base * scale) // 32) * 32)
        self._logged.clear()  # Re-log shapes at new resolution

    @property
    def batch_size(self):
        """Current micro-batch size (same across all bucket datasets)."""
        first = next(iter(self.bucket_datasets.values()), None)
        return first.batch_size if first else 1

    @batch_size.setter
    def batch_size(self, bs):
        """Update batch size on all bucket datasets. Takes effect on next __iter__ cycle."""
        for ds in self.bucket_datasets.values():
            ds.batch_size = bs
        self._needs_reload = True

    def __iter__(self):
        iters = {name: iter(ds) for name, ds in self.bucket_datasets.items()}
        bucket_names = list(self.bucket_names)
        bucket_weights = list(self.bucket_weights)
        while bucket_names:
            bucket = random.choices(bucket_names, weights=bucket_weights, k=1)[0]
            try:
                batch = next(iters[bucket])
            except StopIteration:
                iters[bucket] = iter(self.bucket_datasets[bucket])
                try:
                    batch = next(iters[bucket])
                except StopIteration:
                    # Bucket has too few samples for a full batch (partial=False); remove it.
                    idx = bucket_names.index(bucket)
                    bucket_names.pop(idx)
                    bucket_weights.pop(idx)
                    iters.pop(bucket, None)
                    print(f"  warning: removing bucket with insufficient samples: {bucket}")
                    continue
            if bucket not in self._logged:
                self._logged.add(bucket)
                print(f"  Bucket {bucket}: batch image_data {tuple(batch['image_data'].shape)}")
            batch['bucket'] = bucket
            yield batch

        raise RuntimeError("No buckets available for sampling (all exhausted or empty)")


def _normalize_bucket_name(name: str) -> str:
    return name if name.startswith('bucket_') else f"bucket_{name}"


def _bucket_target_pixel_size(bucket_name: str) -> tuple[int, int]:
    """Get target pixel dimensions (H, W) from bucket name."""
    m = re.search(r"(\d+)x(\d+)$", bucket_name)
    if not m:
        raise ValueError(f"Invalid bucket name (expected ..._<W>x<H>): {bucket_name}")
    w_px, h_px = int(m.group(1)), int(m.group(2))
    return (h_px, w_px)


def get_production_dataloader(config, device='cuda'):
    """Create production dataloader from config.

    Dispatches on data.source:
      "webdataset" (default) — bucket-aware loader from shard tars
      "stratum"              — flat loader from per-image stratum dirs
    """
    from .config_loader import Config

    data_cfg = config.data
    training_cfg = config.training

    if getattr(data_cfg, 'source', 'webdataset') == 'stratum':
        from .data_stratum import get_stratum_dataloader
        print(f"  Data source: stratum")
        print(f"  Stratum dir: {data_cfg.stratum_dir}")
        print(f"  Batch size: {training_cfg.batch_size}")
        return get_stratum_dataloader(
            stratum_dir=data_cfg.stratum_dir,
            batch_size=training_cfg.batch_size,
            shuffle=True,
            target_latent_size=config.model.input_size,
            max_samples=data_cfg.stratum_max_samples,
        )

    shard_dir = Path(data_cfg.shard_base_dir)

    print(f"  Shard base dir: {shard_dir}")
    print(f"  Buckets (configured): {len(data_cfg.buckets)}")
    print(f"  Batch size: {training_cfg.batch_size}")
    print(f"  Flip prob: {data_cfg.horizontal_flip_prob}")
    print(f"  Bucket-aware batching: ENABLED")
    print(f"  Bucket sampling: {data_cfg.bucket_sampling}")

    bucket_datasets = {}
    bucket_weights = []

    for bucket in data_cfg.buckets:
        bucket_name = _normalize_bucket_name(bucket)
        shard_files = sorted((shard_dir / bucket_name).glob('shard-*.tar'))
        if not shard_files:
            print(f"  warning: skipping bucket with no shards: {bucket_name}")
            continue

        target_size = _bucket_target_pixel_size(bucket_name)

        bucket_datasets[bucket_name] = ValidationDataset(
            shard_dir=str(shard_dir / bucket_name),
            shard_files=shard_files,
            batch_size=training_cfg.batch_size,
            shuffle=True,
            flip_prob=data_cfg.horizontal_flip_prob,
            target_latent_size=target_size,
        )
        bucket_weights.append(len(shard_files))

    if not bucket_datasets:
        raise ValueError(f"No shard files found for any configured bucket in {shard_dir}")

    if data_cfg.bucket_sampling == 'uniform':
        bucket_weights = [1.0 for _ in bucket_weights]

    return BucketAwareDataLoader(bucket_datasets, bucket_weights)


if __name__ == "__main__":
    # Test dataloader
    print("Testing validation dataloader...")

    dataloader = get_validation_dataloader(
        shard_dir='data/shards/validation',
        batch_size=4,
        shuffle=True,
    )

    batch = next(iter(dataloader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Image data shape: {batch['image_data'].shape}")
    print(f"DINO embedding shape: {batch['dino_embedding'].shape}")
    print(f"T5 hidden shape: {batch['t5_hidden'].shape}")
    print(f"T5 mask shape: {batch['t5_mask'].shape}")
    print(f"Number of captions: {len(batch['captions'])}")
    print(f"Sample caption: {batch['captions'][0][:80]}...")
    print("✓ Dataloader test passed")
