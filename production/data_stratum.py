"""Dataloader for stratum-hq per-image directory format.

Each sample lives in its own directory, e.g.:
    /mnt/nas-ai-models/training-data/ffhq/stratum/00000/
        pixel.npy          (3, 1024, 1024)  float16  [0, 1]
        t5_hidden.npy      (512, 1024)      float16
        t5_mask.npy        (512,)           uint8
        dinov3_cls.npy     (1024,)          float16
        dinov3_patches.npy (4096, 1024)     float16
        pose.npy           (133, 3)         float16
        caption.txt        str
        metadata.json      {image_id, source_path, width, height, aspect_bucket}

The batch dict returned is identical to the WebDataset path so train.py needs
no changes:
    image_data          (B, 3, H, W)       float32
    dino_embedding      (B, 1024)          float32
    dinov3_patches      (B, max_patches, 1024)  float32
    dinov3_patches_mask (B, max_patches)   long
    t5_hidden           (B, 512, 1024)     float32
    t5_mask             (B, 512)           long
    pose_keypoints      (B, 133, 3)        float32
    captions            list[str]
    image_ids           list[str]
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _resize_image(pixel: np.ndarray, target_size) -> torch.Tensor:
    """Resize (3, H, W) float16 numpy array to target_size via bilinear.

    Args:
        pixel: (3, H, W) numpy array
        target_size: int or (H, W) tuple in pixels

    Returns:
        (3, H, W) float32 torch tensor
    """
    t = torch.from_numpy(pixel.astype(np.float32)).unsqueeze(0)  # (1, 3, H, W)
    if isinstance(target_size, tuple):
        h, w = target_size
    else:
        h = w = target_size
    if t.shape[2] != h or t.shape[3] != w:
        t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    return t.squeeze(0)  # (3, H, W)


def _collate(batch: list[dict]) -> dict:
    """Collate a list of per-sample dicts into a batched dict.

    dinov3_patches are variable-length and are zero-padded to the longest in
    the batch; a boolean mask is added.
    """
    max_patches = max(s['dinov3_patches'].shape[0] for s in batch)

    padded_patches, patch_masks = [], []
    for s in batch:
        p = s['dinov3_patches']
        n = p.shape[0]
        mask = torch.zeros(max_patches, dtype=torch.long)
        mask[:n] = 1
        patch_masks.append(mask)
        if n < max_patches:
            pad = torch.zeros(max_patches - n, p.shape[1], dtype=p.dtype)
            p = torch.cat([p, pad], dim=0)
        padded_patches.append(p)

    return {
        'image_data':          torch.stack([s['image_data']      for s in batch]),
        'dino_embedding':      torch.stack([s['dino_embedding']   for s in batch]),
        'dinov3_patches':      torch.stack(padded_patches),
        'dinov3_patches_mask': torch.stack(patch_masks),
        't5_hidden':           torch.stack([s['t5_hidden']        for s in batch]),
        't5_mask':             torch.stack([s['t5_mask']          for s in batch]),
        'pose_keypoints':      torch.stack([s['pose_keypoints']   for s in batch]),
        'seg_map':             torch.stack([s['seg_map']          for s in batch]),
        'captions':            [s['caption']   for s in batch],
        'image_ids':           [s['image_id']  for s in batch],
    }


class StratumDataset:
    """Infinite shuffled dataloader over a stratum-hq directory tree.

    Compatible with BucketAwareDataLoader — exposes .target_latent_size and
    .batch_size attributes and implements __iter__ yielding collated batches.
    """

    def __init__(
        self,
        stratum_dir: str,
        batch_size: int = 4,
        shuffle: bool = True,
        target_latent_size=1024,
        num_workers: int = 0,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            stratum_dir: Root directory containing per-image subdirs (00000, 00001, …)
            batch_size: Samples per batch
            shuffle: Randomise sample order each epoch
            target_latent_size: Resize pixel.npy to this spatial size (int or (H,W))
            num_workers: Reserved for future DataLoader integration; ignored for now
            max_samples: If set, only iterate up to this many samples instead of 70000
        """
        self.stratum_dir = Path(stratum_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_latent_size = target_latent_size

        # Generate paths directly from the known naming convention (00000–69999).
        # Avoids scandir/iterdir over NAS which can block for several seconds on
        # 70k entries. Corrupt/missing dirs are handled gracefully in _load_sample.
        limit = max_samples if max_samples is not None else 70000
        self._dirs = [self.stratum_dir / f"{i:05d}" for i in range(limit)]
        print(f"[StratumDataset] {len(self._dirs)} samples in {self.stratum_dir}")

    @property
    def resolution_scale(self):
        # Tie latent target to network patch size dynamically. For pixel space, 
        # the model patch_size dictates the alignment requirement.
        return 1.0

    @resolution_scale.setter
    def resolution_scale(self, value):
        pass  # Handled statically per batch context.

    # ------------------------------------------------------------------

    def _load_sample(self, d: Path) -> dict:
        """Load one sample directory into a per-sample dict of tensors."""
        pixel      = np.load(d / 'pixel.npy')           # (3, H, W) f16
        t5_hidden  = np.load(d / 't5_hidden.npy')       # (512, 1024) f16
        t5_mask    = np.load(d / 't5_mask.npy')         # (512,) uint8
        dino_cls   = np.load(d / 'dinov3_cls.npy')      # (1024,) f16
        dino_pat   = np.load(d / 'dinov3_patches.npy')  # (4096, 1024) f16
        pose       = np.load(d / 'pose.npy')            # (133, 3) f16
        caption    = (d / 'caption.txt').read_text().strip()
        meta       = json.loads((d / 'metadata.json').read_text())

        image_data = _resize_image(pixel, self.target_latent_size)

        # Seg map: load uint8 (1024×1024), downsample to token grid (nearest-neighbor).
        # patch_size=16, input=1024px → token grid = 64×64.
        seg_raw = np.load(d / 'seg.npy')                # (H, W) uint8
        seg_t   = torch.from_numpy(seg_raw.astype(np.int16)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        token_size = self.target_latent_size // 16 if isinstance(self.target_latent_size, int) else 64
        seg_grid = F.interpolate(
            seg_t.float(), size=(token_size, token_size), mode='nearest'
        ).squeeze(0).squeeze(0).to(torch.int16)         # (TG, TG) int16

        return {
            'image_data':     image_data,                                       # (3, H, W) f32
            'dino_embedding': torch.from_numpy(dino_cls.astype(np.float32)),   # (1024,)
            'dinov3_patches': torch.from_numpy(dino_pat.astype(np.float32)),   # (4096, 1024)
            't5_hidden':      torch.from_numpy(t5_hidden.astype(np.float32)),  # (512, 1024)
            't5_mask':        torch.from_numpy(t5_mask.astype(np.int64)),      # (512,)
            'pose_keypoints': torch.from_numpy(pose.astype(np.float32)),       # (133, 3)
            'seg_map':        seg_grid,                                         # (TG, TG) int16
            'caption':        caption,
            'image_id':       meta.get('image_id', d.name),
        }

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        """Yield collated batches indefinitely (infinite, shuffled)."""
        dirs = list(self._dirs)
        while True:
            if self.shuffle:
                random.shuffle(dirs)
            batch_buf = []
            for d in dirs:
                try:
                    sample = self._load_sample(d)
                except Exception as e:
                    print(f"[StratumDataset] skipping {d.name}: {e}")
                    continue
                batch_buf.append(sample)
                if len(batch_buf) == self.batch_size:
                    yield _collate(batch_buf)
                    batch_buf = []
            # tail samples dropped (same behaviour as WebDataset partial=False)


def get_stratum_dataloader(
    stratum_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    target_latent_size=1024,
    max_samples: Optional[int] = None,
) -> StratumDataset:
    """Create a StratumDataset dataloader.

    Returns a StratumDataset instance which is directly iterable and
    compatible with the existing training loop (same batch dict keys as
    the WebDataset path).

    Args:
        stratum_dir: Root of per-image stratum dirs
        batch_size: Samples per batch
        shuffle: Randomise order each pass
        target_latent_size: Resize pixel to this size (int or (H, W))
        max_samples: If set, only iterate up to this many samples
    """
    return StratumDataset(
        stratum_dir=stratum_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        target_latent_size=target_latent_size,
        max_samples=max_samples,
    )
