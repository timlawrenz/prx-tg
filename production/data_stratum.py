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

    pose_keypoints may have different joint counts (133 for v1, 308 for v2)
    across samples. They are zero-padded to the longest, with a
    pose_joints_mask indicating valid joints.
    """
    max_patches = max(s['dinov3_patches'].shape[0] for s in batch)
    max_joints = max(s['pose_keypoints'].shape[0] for s in batch)

    padded_patches, patch_masks = [], []
    padded_poses = []
    pose_masks = []
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

        # Pad pose to max_joints
        pose = s['pose_keypoints']   # (N_joints, 3)
        nj = pose.shape[0]
        pose_mask = torch.zeros(max_joints, dtype=torch.long)
        pose_mask[:nj] = 1
        pose_masks.append(pose_mask)
        if nj < max_joints:
            pose_pad = torch.zeros(max_joints - nj, 3, dtype=pose.dtype)
            pose = torch.cat([pose, pose_pad], dim=0)
        padded_poses.append(pose)

    result = {
        'image_data':          torch.stack([s['image_data']      for s in batch]),
        'dino_embedding':      torch.stack([s['dino_embedding']   for s in batch]),
        'dinov3_patches':      torch.stack(padded_patches),
        'dinov3_patches_mask': torch.stack(patch_masks),
        't5_hidden':           torch.stack([s['t5_hidden']        for s in batch]),
        't5_mask':             torch.stack([s['t5_mask']          for s in batch]),
        'pose_keypoints':      torch.stack(padded_poses),
        'pose_mask':           torch.stack(pose_masks),
        'seg_map':             torch.stack([s['seg_map']          for s in batch]),
        'captions':            [s['caption']   for s in batch],
        'image_ids':           [s['image_id']  for s in batch],
    }
    if 'geometry_3d' in batch[0]:
        result['geometry_3d'] = torch.stack([s['geometry_3d'] for s in batch])
    if 'matting' in batch[0]:
        result['matting'] = torch.stack([s['matting'] for s in batch])
    if 'identity_emb' in batch[0]:
        result['identity_emb'] = torch.stack([s['identity_emb'] for s in batch])
    if 'geometry_emb' in batch[0]:
        result['geometry_emb'] = torch.stack([s['geometry_emb'] for s in batch])
    return result


from torch.utils.data import IterableDataset

class StratumDataset(IterableDataset):
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
        adapter_name: str = "stratum",
    ):
        """
        Args:
            stratum_dir: Root directory containing per-image subdirs (00000, 00001, …)
            batch_size: Samples per batch
            shuffle: Randomise sample order each epoch
            target_latent_size: Resize pixel.npy to this spatial size (int or (H,W))
            num_workers: Reserved for future DataLoader integration; ignored for now
            max_samples: If set, only iterate up to this many samples instead of 70000
            adapter_name: "stratum" (default) or "eidolon" — controls which extra
                          embeddings are loaded (eidolon loads auraface_lda + z_g)
        """
        self.stratum_dir = Path(stratum_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_latent_size = target_latent_size
        self.adapter_name = adapter_name

        # Scan directory for available samples (stable across rebuilds)
        existing = sorted([
            d for d in self.stratum_dir.iterdir()
            if d.is_dir() and (d / "pixel.npy").exists()
        ])
        if max_samples is not None and len(existing) > max_samples:
            existing = existing[:max_samples]
        self._dirs = existing
        print(f"[StratumDataset] {len(self._dirs)} samples in {self.stratum_dir}")

    # ------------------------------------------------------------------

    def _load_sample(self, d: Path) -> dict:
        """Load one sample directory into a per-sample dict of tensors.

        Supports both stratum v1 and stratum2 artifacts. When v2 files
        (pose2.npy, seg2.npy) are present, they are preferred. Otherwise
        the v1 fallbacks (pose.npy, seg.npy) are used.
        """
        pixel      = np.load(d / 'pixel.npy')           # (3, H, W) f16
        t5_hidden  = np.load(d / 't5_hidden.npy')       # (512, 1024) f16
        t5_mask    = np.load(d / 't5_mask.npy')         # (512,) uint8
        dino_cls   = np.load(d / 'dinov3_cls.npy')      # (1024,) f16
        dino_pat   = np.load(d / 'dinov3_patches.npy')  # (4096, 1024) f16
        caption    = (d / 'caption.txt').read_text().strip()
        meta       = json.loads((d / 'metadata.json').read_text())

        image_data = _resize_image(pixel, self.target_latent_size)

        # ── Pose: prefer pose2.npy (Sapiens 2, 308 kp) over pose.npy (v1, 133 kp)
        pose2_path = d / 'pose2.npy'
        if pose2_path.exists():
            pose2 = np.load(pose2_path)              # (1, 308, 3) or (N, 308, 3) f32
            pose2 = pose2[0]                          # (308, 3) — take first person
            # pose2 coordinates are in absolute pixel space [0, W] × [0, H].
            # Normalize to [-1, 1] using original image dimensions from metadata.
            W_orig, H_orig = meta['width'], meta['height']
            pose2[:, 0] = (pose2[:, 0] / W_orig) * 2.0 - 1.0   # x: [0, W] → [-1, 1]
            pose2[:, 1] = (pose2[:, 1] / H_orig) * 2.0 - 1.0   # y: [0, H] → [-1, 1]
            # Confidence (pose2[:, 2]) can exceed 1.0 — clip to [0, 1]
            pose2[:, 2] = np.clip(pose2[:, 2], 0.0, 1.0)
            pose = pose2.astype(np.float32)
            num_joints = 308
        else:
            pose = np.load(d / 'pose.npy').astype(np.float32)  # (133, 3) f16→f32
            num_joints = 133

        # ── Segmentation: prefer seg2.npy (Sapiens 2, 29-class) over seg.npy (v1, 28-class)
        seg2_path = d / 'seg2.npy'
        if seg2_path.exists():
            seg_raw = np.load(seg2_path)               # (H, W) uint8, 29-class
        else:
            seg_raw = np.load(d / 'seg.npy')           # (H, W) uint8, 28-class

        # Downsample seg to token grid (nearest-neighbor).
        # patch_size=16, input=1024px → token grid = 64×64.
        seg_t   = torch.from_numpy(seg_raw.astype(np.int16)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        token_size = self.target_latent_size // 16 if isinstance(self.target_latent_size, int) else 64
        seg_grid = F.interpolate(
            seg_t.float(), size=(token_size, token_size), mode='nearest'
        ).squeeze(0).squeeze(0).to(torch.int16)         # (TG, TG) int16

        # ── 3D geometry: pointmap + normals combined (optional, Phase 2)
        geometry_3d = None
        pointmap_path = d / 'pointmap.npy'
        normal2_path = d / 'normal2.npy'
        if pointmap_path.exists() and normal2_path.exists() and seg2_path.exists():
            pointmap = np.load(pointmap_path)           # (H, W, 3) f16, metric XYZ
            normal2  = np.load(normal2_path)            # (H, W, 3) f16, unit vectors
            seg2_full = seg_raw                          # already loaded above

            # Combine into 6D: [X, Y, Z, Nx, Ny, Nz]
            geometry_6d = np.concatenate([pointmap, normal2], axis=-1)  # (H, W, 6)

            # Mask background using seg2 > 0
            fg_mask = (seg2_full > 0)
            geometry_6d[~fg_mask] = 0.0

            # Downsample to 16×16 grid via area interpolation
            geo_t = torch.from_numpy(geometry_6d.astype(np.float32))  # (H, W, 6)
            geo_t = geo_t.permute(2, 0, 1).unsqueeze(0)  # (1, 6, H, W)
            geo_grid = F.interpolate(geo_t, size=(16, 16), mode='area')  # (1, 6, 16, 16)
            geo_grid = geo_grid.squeeze(0).permute(1, 2, 0).reshape(256, 6)  # (256, 6)

            # Normalize pointmap XYZ by a fixed scene scale (3.0m)
            geo_grid[:, :3] = geo_grid[:, :3] / 3.0
            # Normals are already in [-1, 1]

            geometry_3d = geo_grid  # (256, 6) float32

        # ── Matting alpha (optional, Phase 3)
        matting = None
        matting_path = d / 'matting.npy'
        if matting_path.exists():
            matting_raw = np.load(matting_path)         # (H, W) f16, alpha [0,1]
            matting_t = torch.from_numpy(matting_raw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            matting_grid = F.interpolate(matting_t, size=(token_size, token_size), mode='area'
                                         ).squeeze(0).squeeze(0)  # (TG, TG) float32
            matting = matting_grid

        sample = {
            'image_data':     image_data,                                       # (3, H, W) f32
            'dino_embedding': torch.from_numpy(dino_cls.astype(np.float32)),   # (1024,)
            'dinov3_patches': torch.from_numpy(dino_pat.astype(np.float32)),   # (4096, 1024)
            't5_hidden':      torch.from_numpy(t5_hidden.astype(np.float32)),  # (512, 1024)
            't5_mask':        torch.from_numpy(t5_mask.astype(np.int64)),      # (512,)
            'pose_keypoints': torch.from_numpy(pose),                          # (N_joints, 3) f32
            'num_pose_joints': num_joints,                                       # 133 or 308
            'seg_map':        seg_grid,                                         # (TG, TG) int16
            'caption':        caption,
            'image_id':       meta.get('image_id', d.name),
        }
        if geometry_3d is not None:
            sample['geometry_3d'] = geometry_3d
        if matting is not None:
            sample['matting'] = matting
        return sample

    def _load_sample_eidolon(self, d: Path) -> dict:
        """Load minimal eidolon sample (pixel + identity + geometry only).

        Does NOT load T5, DINO, pose, or seg — the EidolonAdapter drops them.
        Returns zero-filled stubs for fields the collate function expects.
        """
        pixel       = np.load(d / 'pixel.npy')           # (3, H, W) f16
        identity_emb = np.load(d / 'auraface_lda.npy')   # (64,) float64
        geometry_emb = np.load(d / 'z_g.npy')            # (50,) float32
        meta        = json.loads((d / 'metadata.json').read_text())

        image_data = _resize_image(pixel, self.target_latent_size)

        return {
            'image_data':     image_data,
            'identity_emb':   torch.from_numpy(identity_emb).float(),
            'geometry_emb':   torch.from_numpy(geometry_emb).float(),
            # Stubs for collate compatibility (not used by EidolonAdapter)
            'dino_embedding': torch.zeros(1024),
            'dinov3_patches': torch.zeros(1, 1024),
            't5_hidden':      torch.zeros(1, 1024),
            't5_mask':        torch.zeros(1, dtype=torch.int64),
            'pose_keypoints': torch.zeros(133, 3),
            'seg_map':        torch.zeros(64, 64, dtype=torch.int16),
            'caption':        meta.get('persona', ''),
            'image_id':       meta.get('image_id', d.name),
        }

    def _load(self, d: Path) -> dict:
        """Dispatch to the correct loader based on adapter_name."""
        if self.adapter_name == "eidolon":
            return self._load_sample_eidolon(d)
        return self._load_sample(d)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        """Yield collated batches indefinitely (infinite, shuffled)."""
        import math
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_dirs = list(self._dirs)
        else:
            per_worker = int(math.ceil(len(self._dirs) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_dirs = self._dirs[worker_id * per_worker:(worker_id + 1) * per_worker]

        while True:
            dirs = list(worker_dirs)
            if self.shuffle:
                random.shuffle(dirs)
            batch_buf = []
            for d in dirs:
                try:
                    sample = self._load(d)
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
    adapter_name: str = "stratum",
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
        adapter_name: "stratum" (default) or "eidolon" — controls which extra
                      embeddings are loaded
    """
    return StratumDataset(
        stratum_dir=stratum_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        target_latent_size=target_latent_size,
        max_samples=max_samples,
        adapter_name=adapter_name,
    )
