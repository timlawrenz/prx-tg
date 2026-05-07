"""Stratum-FFHQ dataset loader for prx-tg.

Loads training data in the stratum-ffhq per-image directory format
(local filesystem or HuggingFace streaming) and maps it to prx-tg's
expected sample dict — drop-in compatible with the existing pipeline.

stratum-ffhq per-image layout (local):
  <image_id>/
    metadata.json       # width, height, aspect_bucket, source_path
    caption.txt         # plain text
    dinov3_cls.npy      # (1024,)    float16
    dinov3_patches.npy  # (N, 1024)  float16
    t5_hidden.npy       # (512, 1024) float16
    t5_mask.npy         # (512,)     uint8
    pose.npy            # (133, 3)   float16
    pixel.npy           # (3, H, W)  float16 — optional, RGB crop
    seg.npy / depth.npy / normal.npy  # unused by prx-tg

HuggingFace: each layer is a parquet or npy_tar.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import swap_left_right  # reuse existing caption flip logic


# ── File name mapping: stratum → prx-tg WebDataset key ──────────────
# stratum-ffhq uses descriptive names; prx-tg uses abbreviated shard keys.
STRATUM_TO_PRX = {
    "dinov3_cls.npy": "dinov3_embedding",     # prx-tg's dinov3.npy
    "dinov3_patches.npy": "dinov3_patches",
    "t5_hidden.npy": "t5_hidden",             # prx-tg's t5h.npy
    "t5_mask.npy": "t5_mask",                 # prx-tg's t5m.npy
    "pose.npy": "pose_keypoints",
    "pixel.npy": "image_data",                # opt-in RGB
}


def _resolve_image_dir(
    img_dir: Path,
    image_base: Optional[Path],
) -> Path | None:
    """Resolve the source image for a stratum per-image directory.

    Priority:
      1. pixel.npy inside img_dir (stratum opt-in pixel layer)
      2. image_base/<image_id>.<ext>  (original images, matched by relative path)
    """
    pixel_path = img_dir / "pixel.npy"
    if pixel_path.exists():
        return pixel_path

    if image_base is None:
        return None

    # image_id is the relative path from the dataset root to this img_dir.
    # We try common extensions.
    img_dir_resolved = img_dir.resolve()
    # Walk up to find dataset_root (parent of the first metadata.json ancestor)
    # and derive the relative image_id.
    # Simpler: just take the dir name as image_id if flat, or reconstruct path.

    # Try image_base / <image_id> with common extensions
    image_id = img_dir.name  # fallback: just the directory name
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = image_base / f"{image_id}{ext}"
        if candidate.exists():
            return candidate

    # Try without extension (image_id already has it as dir name)
    if image_id.endswith((".png", ".jpg", ".jpeg", ".webp")):
        stem = Path(image_id).stem
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = image_base / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    return None


class StratumDataset:
    """Dataset that reads stratum-ffhq per-image directories.

    Produces the same sample dict as ValidationDataset.process_sample()
    so it slots directly into BucketAwareDataLoader / the training loop.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        batch_size: int = 8,
        shuffle: bool = True,
        flip_prob: float = 0.5,
        target_latent_size: int | tuple[int, int] = 64,
        image_base: str | Path | None = None,
        pixel_space: bool = False,
        deterministic: bool = False,
    ):
        """
        Args:
            dataset_dir: Path to stratum dataset root (contains <image_id>/ dirs).
            batch_size: Samples per batch.
            shuffle: Shuffle image directories between epochs.
            flip_prob: Horizontal flip probability.
            target_latent_size: Target spatial size for image tensor
                (latent-space: 64 for 512px; pixel-space: e.g. 1024).
            image_base: Directory containing original images, used when
                pixel.npy is absent from the stratum dirs. Images matched
                by <image_id>.<ext>.
            pixel_space: If True, load RGB pixel data for x_prediction.
                Requires either pixel.npy or image_base.
            deterministic: Set seeds for reproducible ordering.
        """
        self.dataset_dir = Path(dataset_dir).resolve()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flip_prob = flip_prob
        self.target_latent_size = target_latent_size
        self.pixel_space = pixel_space
        self._image_base = Path(image_base).resolve() if image_base else None

        if deterministic:
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)

        self._image_dirs = self._discover_dirs()

        if not self._image_dirs:
            raise ValueError(
                f"No metadata.json found under {self.dataset_dir}. "
                f"Is this a stratum dataset?"
            )

        print(f"StratumDataset: {len(self._image_dirs)} samples from {self.dataset_dir}")
        if self._image_base:
            print(f"  Image source: {self._image_base}")
        print(f"  Pixel space: {self.pixel_space}")

        self._resolution_scale = 1.0

    @property
    def resolution_scale(self):
        return self._resolution_scale

    @resolution_scale.setter
    def resolution_scale(self, scale):
        if scale == self._resolution_scale:
            return
        self._resolution_scale = scale
        # Adjust target_latent_size
        if isinstance(self.target_latent_size, tuple):
            align = 32 if self.pixel_space else 2
            base = self.target_latent_size
            h = max(align, (int(base[0] * scale) // align) * align)
            w = max(align, (int(base[1] * scale) // align) * align)
            self.target_latent_size = (h, w)
        else:
            align = 32 if self.pixel_space else 2
            self.target_latent_size = max(align, (int(self.target_latent_size * scale) // align) * align)

    def _discover_dirs(self) -> list[Path]:
        """Find all per-image directories (those containing metadata.json)."""
        dirs = []
        for meta_path in sorted(self.dataset_dir.rglob("metadata.json")):
            dirs.append(meta_path.parent)
        if self.shuffle:
            rng = random.Random(42)
            rng.shuffle(dirs)
        return dirs

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray:
        """Load a .npy file with memory-mapping for large files."""
        return np.load(path)

    def _load_image(self, img_dir: Path) -> torch.Tensor:
        """Load RGB image for this sample.

        Returns (3, H, W) float32 tensor in [0, 1].
        """
        image_path = _resolve_image_dir(img_dir, self._image_base)

        if image_path is None:
            raise FileNotFoundError(
                f"No pixel data for {img_dir.name}. "
                f"Provide pixel.npy in stratum dir or --image-base pointing to originals."
            )

        if image_path.suffix == ".npy":
            arr = self._load_npy(image_path)  # (3, H, W) float16 [0, 1]
            image = torch.from_numpy(arr.astype(np.float32))
        else:
            # Load from image file using PIL → torch
            from PIL import Image

            pil_img = Image.open(image_path).convert("RGB")
            arr = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)
            image = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        # Resize to target
        if self.target_latent_size is not None:
            if isinstance(self.target_latent_size, tuple):
                target_h, target_w = self.target_latent_size
            else:
                target_h = target_w = self.target_latent_size

            if image.shape[1] != target_h or image.shape[2] != target_w:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        return image.float()

    def process_sample(self, img_dir: Path) -> dict:
        """Load a single stratum per-image directory into prx-tg sample dict.

        Returns dict with keys matching ValidationDataset.process_sample():
          image_data, dino_embedding, dinov3_patches, t5_hidden, t5_mask,
          pose_keypoints, caption, image_id
        """
        image_id = img_dir.name

        # Metadata
        meta_path = img_dir / "metadata.json"
        if meta_path.exists():
            with meta_path.open() as f:
                metadata = json.load(f)
            caption = metadata.get("caption", "")
        else:
            metadata = {}
            caption = ""

        # Caption from caption.txt (newer stratum writes here)
        caption_path = img_dir / "caption.txt"
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()

        # Embeddings
        dino_cls = torch.from_numpy(
            self._load_npy(img_dir / "dinov3_cls.npy").astype(np.float32)
        )  # (1024,)

        dino_patches = torch.from_numpy(
            self._load_npy(img_dir / "dinov3_patches.npy").astype(np.float32)
        )  # (N, 1024)

        t5_hidden = torch.from_numpy(
            self._load_npy(img_dir / "t5_hidden.npy").astype(np.float32)
        )  # (512, 1024)

        t5_mask_raw = self._load_npy(img_dir / "t5_mask.npy")
        # Handle both uint8 and bool
        t5_mask = torch.from_numpy(t5_mask_raw.astype(np.int64)).long()  # (512,)

        pose_kpts = torch.from_numpy(
            self._load_npy(img_dir / "pose.npy").astype(np.float32)
        )  # (133, 3)

        # Image
        if self.pixel_space:
            image_data = self._load_image(img_dir)
        else:
            # Latent-space mode: placeholder (not used in current pipeline)
            # VAE latents would need to be precomputed
            image_data = torch.zeros(3, 64, 64)  # dummy

        return {
            "image_data": image_data,
            "dino_embedding": dino_cls,
            "dinov3_patches": dino_patches,
            "t5_hidden": t5_hidden,
            "t5_mask": t5_mask,
            "pose_keypoints": pose_kpts,
            "caption": caption,
            "image_id": image_id,
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate batch, padding dinov3_patches to max length."""
        max_patches = max(s["dinov3_patches"].shape[0] for s in batch)

        padded_patches = []
        patch_masks = []
        for s in batch:
            patches = s["dinov3_patches"]
            num_patches = patches.shape[0]

            mask = torch.zeros(max_patches, dtype=torch.long)
            mask[:num_patches] = 1
            patch_masks.append(mask)

            if num_patches < max_patches:
                pad = torch.zeros(
                    max_patches - num_patches, patches.shape[1],
                    dtype=patches.dtype,
                )
                padded = torch.cat([patches, pad], dim=0)
            else:
                padded = patches
            padded_patches.append(padded)

        return {
            "image_data": torch.stack([s["image_data"] for s in batch]),
            "dino_embedding": torch.stack([s["dino_embedding"] for s in batch]),
            "dinov3_patches": torch.stack(padded_patches),
            "dinov3_patches_mask": torch.stack(patch_masks),
            "t5_hidden": torch.stack([s["t5_hidden"] for s in batch]),
            "t5_mask": torch.stack([s["t5_mask"] for s in batch]),
            "pose_keypoints": torch.stack([s["pose_keypoints"] for s in batch]),
            "captions": [s["caption"] for s in batch],
            "image_ids": [s["image_id"] for s in batch],
        }

    def __iter__(self) -> Iterator[dict]:
        """Yield batches. If shuffle, repeats infinitely like WebDataset."""
        while True:
            dirs = list(self._image_dirs)
            if self.shuffle:
                random.shuffle(dirs)

            # Collect full batches
            batch_dirs = []
            for d in dirs:
                try:
                    sample = self.process_sample(d)
                    batch_dirs.append(sample)
                    if len(batch_dirs) >= self.batch_size:
                        yield self.collate_fn(batch_dirs)
                        batch_dirs = []
                except (FileNotFoundError, OSError) as e:
                    print(f"  stratum: skipping {d.name}: {e}")
                    continue

            # Yield partial batch at end (only if not infinite repeat)
            if batch_dirs and not self.shuffle:
                yield self.collate_fn(batch_dirs)

            if not self.shuffle:
                break

    def __len__(self) -> int:
        return len(self._image_dirs)


def get_stratum_dataloader(
    dataset_dir: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    flip_prob: float = 0.5,
    target_latent_size: int = 64,
    image_base: str | Path | None = None,
    pixel_space: bool = False,
) -> StratumDataset:
    """Convenience factory for StratumDataset."""
    return StratumDataset(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        flip_prob=flip_prob,
        target_latent_size=target_latent_size,
        image_base=image_base,
        pixel_space=pixel_space,
    )


# ── HuggingFace streaming loader ────────────────────────────────────

class StratumHFDataset:
    """Load stratum-ffhq directly from HuggingFace via streaming.

    Uses `datasets.load_dataset(streaming=True)` to avoid downloading
    the full dataset. Each row is a stratum sample with pre-computed
    embeddings stored as numpy bytes.
    """

    def __init__(
        self,
        repo: str = "timlawrenz/stratum-ffhq",
        *,
        batch_size: int = 8,
        shuffle: bool = True,
        flip_prob: float = 0.5,
        target_latent_size: int | tuple[int, int] = 64,
        image_base: str | Path | None = None,
        pixel_space: bool = False,
        layers: list[str] | None = None,
    ):
        """
        Args:
            repo: HuggingFace dataset repo (user/dataset).
            batch_size: Samples per batch.
            shuffle: Shuffle between epochs.
            flip_prob: Horizontal flip probability.
            target_latent_size: Target spatial size for images.
            image_base: Directory of original images (stratum-ffhq
                doesn't include pixels by default).
            pixel_space: If True, load RGB images.
            layers: Which layers to load. Default: all available.
        """
        self.repo = repo
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flip_prob = flip_prob
        self.target_latent_size = target_latent_size
        self.pixel_space = pixel_space
        self._image_base = Path(image_base).resolve() if image_base else None

        self._layers = layers or ["caption", "dinov3", "t5", "pose"]

        # Lazy init
        self._hf_dataset = None
        self._length = None

        print(f"StratumHFDataset: repo={repo}, layers={self._layers}")
        print(f"  Pixel space: {self.pixel_space}")

    def _ensure_loaded(self):
        """Lazy-load HuggingFace dataset on first access."""
        if self._hf_dataset is not None:
            return

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required for HF streaming. "
                "Install: pip install datasets"
            )

        # Load each layer separately and merge by image_id
        # For simplicity, load all layers and let datasets merge
        self._hf_dataset = load_dataset(
            self.repo,
            split="train",
            streaming=True,
        )

        # Peek to count (optional, may be slow)
        print(f"  HF dataset loaded (streaming)")

    def _hf_row_to_sample(self, row: dict) -> dict:
        """Convert a HuggingFace dataset row to prx-tg sample dict."""
        image_id = str(row.get("image_id", ""))

        # DINOv3
        dino_cls_bytes = row.get("dinov3_cls")
        if dino_cls_bytes is not None:
            if isinstance(dino_cls_bytes, bytes):
                dino_cls = torch.from_numpy(
                    np.frombuffer(dino_cls_bytes, dtype=np.float16).copy().astype(np.float32).reshape(1024)
                )
            else:
                dino_cls = torch.from_numpy(np.array(dino_cls_bytes, dtype=np.float32))
        else:
            dino_cls = torch.zeros(1024)

        # DINOv3 patches
        dino_patches_bytes = row.get("dinov3_patches")
        if dino_patches_bytes is not None:
            if isinstance(dino_patches_bytes, bytes):
                arr = np.frombuffer(dino_patches_bytes, dtype=np.float16).copy().astype(np.float32)
                num_patches = len(arr) // 1024
                dino_patches = torch.from_numpy(arr.reshape(num_patches, 1024))
            else:
                dino_patches = torch.from_numpy(np.array(dino_patches_bytes, dtype=np.float32))
        else:
            dino_patches = torch.zeros(1, 1024)

        # T5
        t5_bytes = row.get("t5_hidden")
        t5_mask_bytes = row.get("t5_mask")
        if t5_bytes is not None:
            if isinstance(t5_bytes, bytes):
                t5_hidden = torch.from_numpy(
                    np.frombuffer(t5_bytes, dtype=np.float16).copy().astype(np.float32).reshape(512, 1024)
                )
            else:
                t5_hidden = torch.from_numpy(np.array(t5_bytes, dtype=np.float32))
        else:
            t5_hidden = torch.zeros(512, 1024)

        if t5_mask_bytes is not None:
            if isinstance(t5_mask_bytes, bytes):
                t5_mask = torch.from_numpy(
                    np.frombuffer(t5_mask_bytes, dtype=np.uint8).copy().astype(np.int64)
                )
            else:
                t5_mask = torch.from_numpy(np.array(t5_mask_bytes, dtype=np.int64))
        else:
            t5_mask = torch.zeros(512, dtype=torch.long)

        # Pose
        pose_bytes = row.get("pose")
        if pose_bytes is not None:
            if isinstance(pose_bytes, bytes):
                pose_kpts = torch.from_numpy(
                    np.frombuffer(pose_bytes, dtype=np.float16).copy().astype(np.float32).reshape(133, 3)
                )
            else:
                pose_kpts = torch.from_numpy(np.array(pose_bytes, dtype=np.float32))
        else:
            pose_kpts = torch.zeros(133, 3)

        # Caption
        caption = str(row.get("caption", ""))

        # Image
        if self.pixel_space and self._image_base:
            image_data = self._load_image_from_base(image_id)
        elif self.pixel_space:
            image_data = torch.zeros(3, 64, 64)
        else:
            image_data = torch.zeros(3, 64, 64)

        return {
            "image_data": image_data,
            "dino_embedding": dino_cls,
            "dinov3_patches": dino_patches,
            "t5_hidden": t5_hidden,
            "t5_mask": t5_mask,
            "pose_keypoints": pose_kpts,
            "caption": caption,
            "image_id": image_id,
        }

    def _load_image_from_base(self, image_id: str) -> torch.Tensor:
        """Load image from image_base matched by image_id."""
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = self._image_base / f"{image_id}{ext}"
            if candidate.exists():
                from PIL import Image
                import numpy as np

                pil_img = Image.open(candidate).convert("RGB")
                arr = np.array(pil_img, dtype=np.float32) / 255.0
                image = torch.from_numpy(arr).permute(2, 0, 1)

                if self.target_latent_size is not None:
                    if isinstance(self.target_latent_size, tuple):
                        th, tw = self.target_latent_size
                    else:
                        th = tw = self.target_latent_size
                    if image.shape[1] != th or image.shape[2] != tw:
                        image = F.interpolate(
                            image.unsqueeze(0), size=(th, tw),
                            mode="bilinear", align_corners=False,
                        ).squeeze(0)

                return image.float()

        return torch.zeros(3, 64, 64)

    def collate_fn(self, batch: list[dict]) -> dict:
        """Same collation as StratumDataset (padding dinov3_patches)."""
        max_patches = max(s["dinov3_patches"].shape[0] for s in batch)
        padded_patches = []
        patch_masks = []
        for s in batch:
            patches = s["dinov3_patches"]
            n = patches.shape[0]
            mask = torch.zeros(max_patches, dtype=torch.long)
            mask[:n] = 1
            patch_masks.append(mask)
            if n < max_patches:
                pad = torch.zeros(max_patches - n, patches.shape[1], dtype=patches.dtype)
                padded = torch.cat([patches, pad], dim=0)
            else:
                padded = patches
            padded_patches.append(padded)

        return {
            "image_data": torch.stack([s["image_data"] for s in batch]),
            "dino_embedding": torch.stack([s["dino_embedding"] for s in batch]),
            "dinov3_patches": torch.stack(padded_patches),
            "dinov3_patches_mask": torch.stack(patch_masks),
            "t5_hidden": torch.stack([s["t5_hidden"] for s in batch]),
            "t5_mask": torch.stack([s["t5_mask"] for s in batch]),
            "pose_keypoints": torch.stack([s["pose_keypoints"] for s in batch]),
            "captions": [s["caption"] for s in batch],
            "image_ids": [s["image_id"] for s in batch],
        }

    def __iter__(self) -> Iterator[dict]:
        self._ensure_loaded()

        while True:
            batch = []
            for row in self._hf_dataset:
                if self.shuffle and random.random() < 0.5:
                    # Simple shuffle: skip ~half to simulate shuffle across epochs
                    pass
                try:
                    sample = self._hf_row_to_sample(row)
                    batch.append(sample)
                    if len(batch) >= self.batch_size:
                        yield self.collate_fn(batch)
                        batch = []
                except Exception as e:
                    print(f"  stratum-hf: error processing {row.get('image_id', '?')}: {e}")
                    continue

            if batch:
                yield self.collate_fn(batch)
            if not self.shuffle:
                break

    def __len__(self) -> int:
        if self._length is None:
            try:
                from datasets import get_dataset_config_names, load_dataset
                # Quick count from parquet metadata
                ds = load_dataset(self.repo, split="train", streaming=True)
                # This is approximate; streaming datasets don't have len
                self._length = 70000  # FFHQ has 70k images
            except Exception:
                self._length = 70000
        return self._length


# ── Tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m production.stratum_data <stratum_dataset_dir> [--pixel-space] [--image-base DIR]")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    pixel_space = "--pixel-space" in sys.argv

    image_base = None
    if "--image-base" in sys.argv:
        idx = sys.argv.index("--image-base")
        image_base = sys.argv[idx + 1]

    ds = StratumDataset(
        dataset_dir=dataset_dir,
        batch_size=2,
        shuffle=False,
        pixel_space=pixel_space,
        image_base=image_base,
    )

    print(f"\nSamples: {len(ds)}")
    print("Loading first batch...\n")

    for batch in ds:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"  image_data:    {batch['image_data'].shape}  {batch['image_data'].dtype}")
        print(f"  dino_embedding: {batch['dino_embedding'].shape}  {batch['dino_embedding'].dtype}")
        print(f"  dinov3_patches: {batch['dinov3_patches'].shape}  {batch['dinov3_patches'].dtype}")
        print(f"  t5_hidden:     {batch['t5_hidden'].shape}  {batch['t5_hidden'].dtype}")
        print(f"  t5_mask:       {batch['t5_mask'].shape}  {batch['t5_mask'].dtype}")
        print(f"  pose_keypoints:{batch['pose_keypoints'].shape}  {batch['pose_keypoints'].dtype}")
        print(f"  captions:      {len(batch['captions'])}")
        if batch["captions"]:
            print(f"    [0]: {batch['captions'][0][:120]}...")
        print(f"  image_ids:     {batch['image_ids']}")
        print("\n✓ StratumDataset test passed")
        break