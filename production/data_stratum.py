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
    if all('geometry_3d' in s for s in batch):
        result['geometry_3d'] = torch.stack([s['geometry_3d'] for s in batch])
    if all('matting' in s for s in batch):
        result['matting'] = torch.stack([s['matting'] for s in batch])
    if all('identity_emb' in s for s in batch):
        result['identity_emb'] = torch.stack([s['identity_emb'] for s in batch])
    if all('geometry_emb' in s for s in batch):
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
        require_pose2: bool = False,
        prefer_pose2: bool = False,
        latent_mode: bool = False,
        load_dino_patches: bool = True,
        load_seg: bool = True,
        load_geometry_3d: bool = True,
        load_matting: bool = False,
        expected_basis_fingerprint: Optional[str] = None,
        allow_unstamped_identity: bool = False,
        exclude_dirs: Optional[set] = None,
        exclude_personas: Optional[set] = None,
    ):
        """
        Args:
            stratum_dir: Root directory containing per-image subdirs (00000, 00001, …)
            batch_size: Samples per batch
            shuffle: Randomise sample order each epoch
            target_latent_size: Resize pixel.npy to this spatial size (int or (H,W)).
                          IGNORED in latent_mode — FLUX latents are (16,128,128) already.
            num_workers: Reserved for future DataLoader integration; ignored for now
            max_samples: If set, only iterate up to this many samples instead of 70000
            adapter_name: "stratum" (default) or "eidolon" — controls which extra
                          embeddings are loaded (eidolon loads auraface_lda + z_g)
            require_pose2: If True, skip directories without pose2.npy.
                           Use for pose2 ablation to ensure consistent 308kp conditioning.
            prefer_pose2: If True, load pose2.npy (308kp) and filter to dirs that have it.
                          If False (default), ALWAYS load pose.npy (133kp) — the adapter
                          is built for exactly one joint count; never mix within a run.
            latent_mode: If True, read flux_latent.npy (16,128,128) instead of
                          pixel.npy (3,1024,1024) as image_data — latent-space training
                          (FLUX-AE 8x). Image augmentation on latents is NOT applied.
            load_dino_patches: Load dinov3_patches.npy (8.4MB/sample). False when
                          training.dino_patches.enabled=false — the model never sees
                          them; stub (1,1024) keeps the collate contract.
            load_seg: Load seg.npy/seg2.npy (1MB) — False when seg_weight and
                          matting_edge are both disabled.
            load_geometry_3d: Load pointmap+normal2 (12.6MB when present) — False
                          when the geometry_3d CFG stream is disabled.
            load_matting: Load matting.npy (2MB) — True only for matting_edge arms.
            expected_basis_fingerprint: For adapter_name="eidolon": the LDA basis
                          fingerprint this run expects (see the dataset dir's
                          BASIS_FINGERPRINT.json). A mismatch is refused — see
                          _assert_identity_basis.
            allow_unstamped_identity: Downgrade a missing/unverifiable basis stamp
                          from an error to a warning. Does NOT excuse a measured
                          scale mismatch.
        """
        self.stratum_dir = Path(stratum_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_latent_size = target_latent_size
        self.adapter_name = adapter_name
        self.require_pose2 = require_pose2
        self.prefer_pose2 = prefer_pose2
        self.latent_mode = latent_mode
        self.load_dino_patches = load_dino_patches
        self.load_seg = load_seg
        self.load_geometry_3d = load_geometry_3d
        self.load_matting = load_matting
        self.expected_basis_fingerprint = expected_basis_fingerprint
        self.allow_unstamped_identity = allow_unstamped_identity
        # Identity-level holdout. `exclude_dirs` matches a sample directory name
        # (FFHQ: the identity IS the dir), `exclude_personas` matches the part
        # before `--` (hegre: one dir per shot of a persona).
        self.exclude_dirs = exclude_dirs or set()
        self.exclude_personas = exclude_personas or set()

        # Refuse a mixed-basis identity slot before any sample is read (eidolon only).
        _assert_identity_basis(self.stratum_dir, self.adapter_name,
                               expected_basis_fingerprint, allow_unstamped_identity)

        # Scan directory for available samples (stable across rebuilds).
        # latent_mode gates on flux_latent.npy — images lacking the latent are skipped
        # entirely (never mixed with pixel fallback).
        img_file = "flux_latent.npy" if latent_mode else "pixel.npy"
        # The eidolon adapter needs its identity + geometry sidecars on EVERY
        # sample it is offered. Without this the scan counts dirs that the loader
        # then throws on, so the printed sample count is not what the run
        # consumes (40 FFHQ dirs carry a latent but no auraface_lda.npy).
        needs_eidolon = (self.adapter_name == "eidolon")
        existing = sorted([
            d for d in self.stratum_dir.iterdir()
            if d.is_dir() and (d / img_file).exists()
            and (not require_pose2 or (d / "pose2.npy").exists())
            and (not prefer_pose2 or (d / "pose2.npy").exists())
            and (not needs_eidolon
                 or ((d / "auraface_lda.npy").exists() and (d / "z_g.npy").exists()))
        ])
        # ── identity-level holdout: drop excluded identities BEFORE any cap ──
        n_scanned = len(existing)
        if self.exclude_dirs or self.exclude_personas:
            existing = [d for d in existing if not self._is_excluded(d)]
            n_excluded = n_scanned - len(existing)
            print(f"[StratumDataset] holdout: excluded {n_excluded} of {n_scanned} dirs "
                  f"({len(self.exclude_dirs)} dir ids, {len(self.exclude_personas)} persona ids)")
            # Fail closed if nothing matched: it means the manifest does not
            # describe THIS tree, and training on the holdout silently is the one
            # failure this mechanism exists to prevent.
            if n_excluded == 0:
                raise RuntimeError(
                    f"holdout excluded 0 of {n_scanned} dirs under {self.stratum_dir} — "
                    f"the manifest does not describe this tree (wrong root, or the tree "
                    f"was rebuilt). Refusing to train on the holdout.")
        if max_samples is not None and len(existing) > max_samples:
            existing = existing[:max_samples]
        self._dirs = existing
        print(f"[StratumDataset] {len(self._dirs)} samples in {self.stratum_dir}"
              + (f" (latent_mode=True)" if latent_mode else "")
              + (f" (require_pose2=True)" if require_pose2 else ""))

    # ------------------------------------------------------------------

    @staticmethod
    def _persona_of(name: str) -> Optional[str]:
        """Persona id from a sample dir name, or None for a flat (FFHQ) tree."""
        return name.split("--", 1)[0] if "--" in name else None

    def _is_excluded(self, d: Path) -> bool:
        if d.name in self.exclude_dirs:
            return True
        persona = self._persona_of(d.name)
        return persona is not None and persona in self.exclude_personas

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
        if self.load_dino_patches:
            dino_pat = np.load(d / 'dinov3_patches.npy')  # (4096, 1024) f16 — 8.4MB
        else:
            # Patches disabled in this arm — stub keeps the collate contract
            # without paying the per-sample read+cast (the model never sees them).
            dino_pat = np.zeros((1, 1024), dtype=np.float16)
        caption    = (d / 'caption.txt').read_text().strip()
        meta       = json.loads((d / 'metadata.json').read_text())

        if self.latent_mode:
            # Latent-space training: image_data = FLUX-AE latent (16,128,128) f16.
            # Latents are already at target resolution — no resize, no augmentation.
            image_data = torch.from_numpy(np.load(d / 'flux_latent.npy').astype(np.float32))
        else:
            image_data = _resize_image(pixel, self.target_latent_size)

        # ── Pose: serve EXACTLY ONE joint count per run — the adapter is built
        # for a single num_pose_joints. prefer_pose2 (True only when the model
        # config declares 308 joints) loads pose2.npy; otherwise v1 pose.npy
        # (133kp) is always used, so mixed-enrichment datasets never inject a
        # 308-kp tensor into a 133-kp model.
        if self.prefer_pose2:
            pose2 = np.load(d / 'pose2.npy')              # (1, 308, 3) or (N, 308, 3) f32
            pose2 = pose2[0]                              # (308, 3) — take first person
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
        # Skip entirely when neither seg_weight nor matting_edge consumes it.
        token_size = self.target_latent_size // 16 if isinstance(self.target_latent_size, int) else 64
        if self.load_seg:
            seg2_path = d / 'seg2.npy'
            if seg2_path.exists():
                seg_raw = np.load(seg2_path)           # (H, W) uint8, 29-class
            else:
                seg_raw = np.load(d / 'seg.npy')       # (H, W) uint8, 28-class
            seg_t   = torch.from_numpy(seg_raw.astype(np.int16)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            seg_grid = F.interpolate(
                seg_t.float(), size=(token_size, token_size), mode='nearest'
            ).squeeze(0).squeeze(0).to(torch.int16)     # (TG, TG) int16
        else:
            seg_grid = torch.zeros(token_size, token_size, dtype=torch.int16)
            seg2_path = d / 'seg2.npy'  # referenced by the geometry_3d gate below

        # ── 3D geometry: pointmap + normals combined (optional, Phase 2)
        geometry_3d = None
        pointmap_path = d / 'pointmap.npy'
        normal2_path = d / 'normal2.npy'
        if (self.load_geometry_3d and pointmap_path.exists()
                and normal2_path.exists() and seg2_path.exists()):
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
        if self.load_matting and matting_path.exists():
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
        """Load minimal eidolon sample (identity + geometry + image).

        Does NOT load T5, DINO, pose, or seg — the EidolonAdapter drops them.
        Returns zero-filled stubs for fields the collate function expects.

        Honors `latent_mode`: the image slot is either the FLUX-AE latent
        (16,128,128) read directly, or the resized pixel tensor. Getting this
        wrong feeds a 3-channel image to a 16-channel patch embed, so the two
        paths must not be conflated (and reading pixel.npy when the latent is
        all we need wastes 6.3 MB of NAS I/O per sample).
        """
        identity_emb = np.load(d / 'auraface_lda.npy')   # (64,) float64|float32
        geometry_emb = np.load(d / 'z_g.npy')            # (50,) float32
        meta        = json.loads((d / 'metadata.json').read_text())

        if self.latent_mode:
            # Latents are already at target resolution — no resize, no augmentation.
            image_data = torch.from_numpy(
                np.load(d / 'flux_latent.npy').astype(np.float32))   # (16,128,128)
        else:
            pixel = np.load(d / 'pixel.npy')             # (3, H, W) f16
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
    require_pose2: bool = False,
    prefer_pose2: bool = False,
    latent_mode: bool = False,
    load_dino_patches: bool = True,
    load_seg: bool = True,
    load_geometry_3d: bool = True,
    load_matting: bool = False,
    expected_basis_fingerprint: Optional[str] = None,
    allow_unstamped_identity: bool = False,
    exclude_dirs: Optional[set] = None,
    exclude_personas: Optional[set] = None,
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
        require_pose2: If True, skip directories without pose2.npy.
    """
    return StratumDataset(
        stratum_dir=stratum_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        target_latent_size=target_latent_size,
        max_samples=max_samples,
        adapter_name=adapter_name,
        require_pose2=require_pose2,
        prefer_pose2=prefer_pose2,
        latent_mode=latent_mode,
        load_dino_patches=load_dino_patches,
        load_seg=load_seg,
        load_geometry_3d=load_geometry_3d,
        load_matting=load_matting,
        expected_basis_fingerprint=expected_basis_fingerprint,
        allow_unstamped_identity=allow_unstamped_identity,
        exclude_dirs=exclude_dirs,
        exclude_personas=exclude_personas,
    )


# Unit-norm refit coords measure ~1.0; raw coords ~153; pre-refit coords ~0.35.
_IDENTITY_NORM_BAND = (0.5, 2.0)


def _assert_identity_basis(
    stratum_dir: Path,
    adapter_name: str,
    expected_fingerprint: Optional[str],
    allow_unstamped: bool,
) -> None:
    """Refuse to feed the identity slot from a differently-basis'd dataset.

    The 64-d AuraFace-LDA vector is only meaningful together with the basis it
    was projected through. A refit changes both direction and magnitude, so a
    slot built from one basis, mixed with arms trained on another, yields a
    condition the model cannot use — and nothing else notices, because every
    path returns a valid-looking (64,) tensor. That is the defect class that
    produced the mixed-basis Eidolon arms.

    Two checks, because they fail differently:
      1. the directory stamp's basis_fingerprint must match the config's declared
         expectation — catches a renamed or stale tree;
      2. the MEASURED norm of a sample of identity vectors must sit in the
         unit-norm band — catches a raw-coords (~153) or pre-refit (~0.35) tree
         by reading the tensor, not by trusting a convention string.

    `allow_unstamped` downgrades (1) to a warning. It deliberately does NOT
    excuse (2): a measured scale mismatch is a definite confound, not a
    bookkeeping gap.
    """
    if adapter_name != "eidolon":
        return

    label = "[StratumDataset] IDENTITY BASIS"

    def _fail(msg: str) -> None:
        if allow_unstamped:
            print(f"[StratumDataset] WARNING: {msg}  "
                  f"(proceeding: data.allow_unstamped_identity=true)")
        else:
            raise RuntimeError(
                f"{msg}  Fix: set data.basis_fingerprint to the dataset's "
                f"BASIS_FINGERPRINT.json value, or set "
                f"data.allow_unstamped_identity=true to accept this knowingly "
                f"(geometry-only runs of concluded arms only).")

    stamp_path = stratum_dir / "BASIS_FINGERPRINT.json"
    found, conv = None, None
    if stamp_path.exists():
        try:
            stamp = json.loads(stamp_path.read_text())
            found = stamp.get("basis_fingerprint")
            conv = stamp.get("projection_convention")
        except Exception as e:               # unreadable stamp = no evidence
            _fail(f"{label} UNGUARDED: {stamp_path} is unreadable ({e}).")

    if not found:
        _fail(f"{label} UNGUARDED: {stratum_dir} carries no BASIS_FINGERPRINT.json, "
              f"so the auraface_lda slot cannot be verified.")
    elif expected_fingerprint and found != expected_fingerprint:
        raise RuntimeError(
            f"{label} MISMATCH: {stratum_dir} is stamped {found} but the config "
            f"declares data.basis_fingerprint={expected_fingerprint}. The identity "
            f"vectors are in a different basis (direction AND magnitude) than this "
            f"run expects — refusing to train on a mixed-basis identity slot.")
    elif not expected_fingerprint:
        _fail(f"{label} UNGUARDED: {stratum_dir} is stamped {found} ({conv}) but the "
              f"config declares no data.basis_fingerprint to check it against.")
    else:
        print(f"{label} stamp OK: {found} ({conv})")

    # (2) measured scale, independent of the stamp and of its wording.
    sample_dirs = []
    try:
        for d in sorted(stratum_dir.iterdir()):
            if (d / "auraface_lda.npy").exists():
                sample_dirs.append(d)
                if len(sample_dirs) >= 8:
                    break
    except OSError as e:
        _fail(f"{label} UNGUARDED: cannot list {stratum_dir} ({e}).")
        return

    if not sample_dirs:
        _fail(f"{label} UNGUARDED: no auraface_lda.npy under {stratum_dir}.")
        return

    norms = []
    for d in sample_dirs:
        try:
            norms.append(float(np.linalg.norm(np.load(d / "auraface_lda.npy"))))
        except Exception:
            continue
    if not norms:
        _fail(f"{label} UNGUARDED: sampled auraface_lda.npy files were unreadable.")
        return

    lo, hi = _IDENTITY_NORM_BAND
    med = float(np.median(norms))
    if not (lo <= med <= hi):
        raise RuntimeError(
            f"{label} SCALE MISMATCH: median norm {med:.3f} over {len(norms)} sampled "
            f"identity vectors in {stratum_dir} falls outside the unit-norm band "
            f"[{lo}, {hi}]. Unit-norm refit coords are ~1.0, raw coords ~153, "
            f"pre-refit coords ~0.35. The DiT identity slot requires unit-norm "
            f"vectors — refusing to train at a {med:.3f} scale.")
    print(f"{label} scale OK: median norm {med:.3f} over {len(norms)} sampled vectors")


# ======================================================================
# Identity-level holdout + multi-source weighted interleaving
# ======================================================================

def load_holdout_exclusions(manifest_path: str) -> dict:
    """Read the LOCKED identity-holdout manifest into exclusion sets.

    Returns {"dirs": {...}, "personas": {...}}. Each list's recorded sha256 is
    verified before use: the manifest is a frozen artifact, and one silently
    regenerated under an already-gated run would change the training set behind
    the gate's back.

    The digest convention must match scripts/build_identity_holdout.py exactly:
    sha256 over each element encoded with a trailing newline, in list order.
    """
    import hashlib
    import json

    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"holdout manifest not found: {p}")
    m = json.loads(p.read_text())

    def _digest(items):
        h = hashlib.sha256()
        for it in items:
            h.update(it.encode() + b"\n")
        return h.hexdigest()

    out = {"dirs": set(), "personas": set()}
    for section, key in (("ffhq", "dirs"), ("hegre", "personas")):
        block = m.get(section) or {}
        items = block.get("holdout") or []
        recorded = block.get("holdout_sha256")
        if recorded and _digest(items) != recorded:
            raise ValueError(
                f"holdout manifest {p} section `{section}` does NOT match its "
                f"recorded sha256 (recorded {str(recorded)[:16]}, computed "
                f"{_digest(items)[:16]}). The manifest drifted since it was locked. "
                f"Regenerate it deliberately and re-derive the gate — do not train "
                f"on a silent change to the holdout.")
        if items:
            out[key] |= set(items)

    if not out["dirs"] and not out["personas"]:
        raise ValueError(f"holdout manifest {p} declares no exclusions — refusing.")
    return out


class MultiStratumDataset:
    """Weighted interleaving of several per-image stratum roots.

    One epoch emits samples from every root, interleaved in random order, with
    each root's share set by its weight. `weight` is a relative DRAW SHARE, not a
    cap:

        k   = max_i(len(list_i) / w_i)     # the binding root
        N_i = round(k * w_i)               # draws for root i

    so the configured ratio holds EXACTLY while no root is under-used: the root
    that binds gets one full pass, and shorter roots are recycled (reshuffled
    each cycle) to top up their share. That is deliberate — the scarce root here
    is the one carrying the mechanism under test (hegre is the only multi-shot
    source), so its share must not collapse to its natural size ratio.

    Each root keeps its OWN identity-basis guard and holdout exclusion, because
    the trees carry different units (FFHQ: identity == dir; hegre: dir == one
    shot of a persona).
    """

    def __init__(self, datasets: list, weights: list, batch_size: int = 4,
                 shuffle: bool = True, seed: int = 1234):
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must be the same length")
        if any(w <= 0 for w in weights):
            raise ValueError(f"weights must be positive, got {weights}")
        for ds in datasets:
            if len(ds._dirs) == 0:
                raise RuntimeError(
                    f"root {ds.stratum_dir} yielded 0 samples — refusing to start "
                    f"with an empty source. A silently empty root is how a "
                    f"multi-source arm quietly becomes a single-source arm.")
        self.datasets = datasets
        self.weights = [float(w) for w in weights]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        lens = [len(d._dirs) for d in datasets]
        k = max(l / w for l, w in zip(lens, self.weights))
        draws = [int(round(k * w)) for w in self.weights]
        print("[MultiStratumDataset] weighted interleaving:")
        for ds, w, l, n in zip(datasets, self.weights, lens, draws):
            print(f"    {Path(ds.stratum_dir).name:18s} weight={w:<6g} "
                  f"available={l:<7d} draws/epoch={n:<7d} recycle={n/l:.2f}x")
        print(f"    epoch = {sum(draws)} samples -> {sum(draws) // batch_size} batches")

    def _epoch_schedule(self):
        """(root_index, dir_path) pairs for one epoch, order shuffled."""
        import random
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        lens = [len(d._dirs) for d in self.datasets]
        k = max(l / w for l, w in zip(lens, self.weights))
        sched = []
        for i, (l, w) in enumerate(zip(lens, self.weights)):
            need, picked = int(round(k * w)), []
            while len(picked) < need:
                order = list(range(l))
                rng.shuffle(order)
                picked.extend(order[:need - len(picked)])
            sched.extend((i, self.datasets[i]._dirs[j]) for j in picked)
        if self.shuffle:
            rng.shuffle(sched)
        return sched

    def __iter__(self):
        buf = []
        for ri, d in self._epoch_schedule():
            try:
                buf.append(self.datasets[ri]._load(d))
            except Exception as e:
                print(f"[MultiStratumDataset] skipping {d.name}: {e}")
                continue
            if len(buf) == self.batch_size:
                yield _collate(buf)
                buf = []
        # tail dropped — same behaviour as StratumDataset (partial=False)

    def __len__(self):
        lens = [len(d._dirs) for d in self.datasets]
        k = max(l / w for l, w in zip(lens, self.weights))
        return sum(int(round(k * w)) for w in self.weights) // self.batch_size


def get_multi_stratum_dataloader(roots, batch_size: int = 4, shuffle: bool = True,
                                 seed: int = 1234, **kwargs) -> MultiStratumDataset:
    """Build a MultiStratumDataset from [{"dir": ..., "weight": ...}, ...].

    Every remaining kwarg is passed through to each per-root StratumDataset, so
    the guards (basis fingerprint, holdout exclusions, latent_mode) apply per
    root rather than once for the whole run.
    """
    if not roots:
        raise ValueError("stratum_dirs is empty")
    datasets, weights = [], []
    for entry in roots:
        if isinstance(entry, str):
            root, w = entry, 1.0
        elif isinstance(entry, dict):
            root = entry.get("dir") or entry.get("path")
            w = float(entry.get("weight", 1.0))
        else:
            raise ValueError(f"stratum_dirs entry must be a str or dict, got {entry!r}")
        if not root:
            raise ValueError(f"stratum_dirs entry declares no dir: {entry}")
        datasets.append(StratumDataset(stratum_dir=root, batch_size=batch_size, **kwargs))
        weights.append(w)
    return MultiStratumDataset(datasets, weights, batch_size=batch_size,
                               shuffle=shuffle, seed=seed)

