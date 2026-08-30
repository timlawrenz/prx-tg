"""Tests for sd15_geometry adapter mode on StratumDataset.

Tests:
  - sd15_loader_produces_correct_batch_fields: batch has image_data, z_g, caption
  - sd15_loader_resizes_to_target: image is resized to target_latent_size
  - sd15_loader_z_g_shape: z_g is (batch, 50) float32
  - sd15_collate_stacks_z_g: collate correctly stacks z_g vectors
  - sd15_loader_cache: .stratum_cache is created

Uses temporary directories with mock stratum data — no NAS access needed.
"""

import json
import os
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import patch


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_stratum_sample(root: Path, name: str, img_size=1024, caption="a test portrait"):
    """Create a minimal stratum sample directory with pixel.npy + z_g.npy + caption.txt."""
    d = root / name
    d.mkdir(parents=True)

    h = w = img_size
    pixel = np.random.rand(3, h, w).astype(np.float16)
    np.save(d / "pixel.npy", pixel)

    z_g = np.random.randn(50).astype(np.float32)
    np.save(d / "z_g.npy", z_g)

    (d / "caption.txt").write_text(caption)
    (d / "metadata.json").write_text(json.dumps({"image_id": name}))

    return d


def _make_mock_stratum(tmp_path: Path, n_samples=8, img_size=1024):
    """Create a mock stratum directory with n_samples."""
    for i in range(n_samples):
        _make_stratum_sample(tmp_path, f"{i:05d}", img_size=img_size)
    return tmp_path


# ── Tests ────────────────────────────────────────────────────────────────────

class TestStratumDatasetSD15:
    """Test the sd15_geometry adapter mode on StratumDataset."""

    def test_sd15_loader_produces_correct_batch_fields(self):
        """sd15_geometry loader should yield batches with image_data, z_g, caption."""
        from production.data_stratum import StratumDataset

        with tempfile.TemporaryDirectory() as td:
            root = _make_mock_stratum(Path(td), n_samples=8)

            ds = StratumDataset(
                stratum_dir=str(root),
                batch_size=4,
                shuffle=False,
                target_latent_size=512,
                adapter_name="sd15_geometry",
            )

            # Get first batch
            batch = next(iter(ds))

            assert "image_data" in batch, "Batch missing image_data"
            assert "z_g" in batch, "Batch missing z_g"
            assert "captions" in batch, "Batch missing captions"
            assert "image_ids" in batch, "Batch missing image_ids"

            # Shape checks
            assert batch["image_data"].shape == (4, 3, 512, 512), \
                f"Expected (4, 3, 512, 512), got {batch['image_data'].shape}"
            assert batch["z_g"].shape == (4, 50), \
                f"Expected (4, 50), got {batch['z_g'].shape}"
            assert len(batch["captions"]) == 4
            assert all(isinstance(c, str) for c in batch["captions"])

    def test_sd15_loader_z_g_dtype(self):
        """z_g should be float32."""
        from production.data_stratum import StratumDataset

        with tempfile.TemporaryDirectory() as td:
            root = _make_mock_stratum(Path(td), n_samples=4)

            ds = StratumDataset(
                stratum_dir=str(root),
                batch_size=2,
                shuffle=False,
                target_latent_size=512,
                adapter_name="sd15_geometry",
            )

            batch = next(iter(ds))
            assert batch["z_g"].dtype == torch.float32, \
                f"Expected float32, got {batch['z_g'].dtype}"

    def test_sd15_loader_resizes_image(self):
        """Images should be resized to target_latent_size (512)."""
        from production.data_stratum import StratumDataset

        with tempfile.TemporaryDirectory() as td:
            root = _make_mock_stratum(Path(td), n_samples=4, img_size=1024)

            ds = StratumDataset(
                stratum_dir=str(root),
                batch_size=2,
                shuffle=False,
                target_latent_size=512,
                adapter_name="sd15_geometry",
            )

            batch = next(iter(ds))
            assert batch["image_data"].shape[2] == 512, "Image height should be 512"
            assert batch["image_data"].shape[3] == 512, "Image width should be 512"

    def test_sd15_loader_image_range(self):
        """Image data should be in [0, 1] range (float32)."""
        from production.data_stratum import StratumDataset

        with tempfile.TemporaryDirectory() as td:
            root = _make_mock_stratum(Path(td), n_samples=4)

            ds = StratumDataset(
                stratum_dir=str(root),
                batch_size=2,
                shuffle=False,
                target_latent_size=512,
                adapter_name="sd15_geometry",
            )

            batch = next(iter(ds))
            img = batch["image_data"]
            assert img.min() >= 0.0, f"Image min below 0: {img.min()}"
            assert img.max() <= 1.0, f"Image max above 1: {img.max()}"

    def test_sd15_loader_shuffle(self):
        """Shuffled loader should produce different ordering across epochs."""
        from production.data_stratum import StratumDataset

        with tempfile.TemporaryDirectory() as td:
            root = _make_mock_stratum(Path(td), n_samples=16)

            ds = StratumDataset(
                stratum_dir=str(root),
                batch_size=4,
                shuffle=True,
                target_latent_size=512,
                adapter_name="sd15_geometry",
            )

            # Get captions from first 2 batches of first pass
            it = iter(ds)
            batch1 = next(it)
            captions1_a = batch1["captions"]
            batch2 = next(it)
            captions1_b = batch2["captions"]

            # Second pass
            it = iter(ds)
            batch1_v2 = next(it)
            captions2 = batch1_v2["captions"]

            # With shuffle, the first batch of two epochs should differ
            # (16 samples, batch 4 — only 4 batches, some overlap possible but unlikely)
            all_same = all(a == b for a, b in zip(captions1_a, captions2))
            assert not all_same, \
                "Shuffled iterations produced identical ordering (unlikely with 16 samples)"

    def test_sd15_loader_basic_init(self):
        """StratumDataset with sd15_geometry mode initializes without error."""
        from production.data_stratum import StratumDataset

        with tempfile.TemporaryDirectory() as td:
            root = _make_mock_stratum(Path(td), n_samples=1)

            ds = StratumDataset(
                stratum_dir=str(root),
                batch_size=2,
                shuffle=False,
                target_latent_size=512,
                adapter_name="sd15_geometry",
            )

            assert ds.adapter_name == "sd15_geometry"
            assert ds.batch_size == 2

    def test_sd15_loader_skips_missing_z_g(self):
        """Samples without z_g.npy should be skipped gracefully."""
        from production.data_stratum import StratumDataset

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Create 2 valid, 1 missing z_g, and 1 more valid (4 total)
            _make_stratum_sample(root, "00000")
            _make_stratum_sample(root, "00001")
            os.remove(root / "00001" / "z_g.npy")  # this one should be skipped
            _make_stratum_sample(root, "00002")
            _make_stratum_sample(root, "00003")

            ds = StratumDataset(
                stratum_dir=str(root),
                batch_size=2,
                shuffle=False,
                target_latent_size=512,
                adapter_name="sd15_geometry",
            )

            # Should yield batches from the 3 valid samples
            batch = next(iter(ds))
            assert len(batch["captions"]) == 2, \
                f"Expected batch of 2, got {len(batch['captions'])}"
            # The skipped sample (00001) should not appear
            for cid in batch["image_ids"]:
                assert cid != "00001", f"Skipped sample appeared in batch: {cid}"

    def test_collate_includes_z_g_field(self):
        """_collate should correctly stack z_g vectors."""
        from production.data_stratum import _collate

        batch = [
            {
                "image_data": torch.randn(3, 512, 512),
                "z_g": torch.randn(50),
                "caption": "test",
                "image_id": "00000",
            },
            {
                "image_data": torch.randn(3, 512, 512),
                "z_g": torch.randn(50),
                "caption": "test2",
                "image_id": "00001",
            },
        ]

        # Add stub fields that _collate expects (from stratum adapter)
        for s in batch:
            s["dino_embedding"] = torch.zeros(1024)
            s["dinov3_patches"] = torch.zeros(1, 1024)
            s["t5_hidden"] = torch.zeros(1, 1024)
            s["t5_mask"] = torch.zeros(1, dtype=torch.int64)
            s["pose_keypoints"] = torch.zeros(133, 3)
            s["seg_map"] = torch.zeros(64, 64, dtype=torch.int16)

        result = _collate(batch)

        assert "z_g" in result, "z_g missing from collated batch"
        assert result["z_g"].shape == (2, 50), \
            f"Expected (2, 50), got {result['z_g'].shape}"
