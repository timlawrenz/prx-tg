"""Regression tests for visual_debug._load_debug_samples key aliasing.

Bug (2026-08-30): StratumDataset returns conditioning under keys
t5_hidden/t5_mask/pose_keypoints, but _load_debug_samples only aliased
dino_embedding/dinov3_patches/captions. debug_fn() reads text_emb/text_mask/
pose_kpts → KeyError at the first visual_debug tick (step 250) — which a
50-step sanity run never reached. These tests pin the alias contract so any
backend that changes batch keys fails loudly here instead of at step 250
of a production run.
"""

import numpy as np
import torch

from production import visual_debug


class _FakeValLoader:
    """Yields one batch dict in StratumDataset._collate format."""

    def __iter__(self):
        yield {
            "image_data": torch.rand(4, 3, 64, 64),
            "dino_embedding": torch.rand(4, 1024),
            "dinov3_patches": torch.rand(4, 16, 1024),
            "dinov3_patches_mask": torch.ones(4, 16, dtype=torch.long),
            "t5_hidden": torch.rand(4, 512, 1024),
            "t5_mask": torch.ones(4, 512, dtype=torch.long),
            "pose_keypoints": torch.rand(4, 133, 3),
            "seg_map": torch.zeros(4, 4, 4, dtype=torch.int16),
            "captions": [f"caption {i}" for i in range(4)],
            "image_ids": [f"{i:05d}" for i in range(4)],
        }
        # single batch
        return


def _make_stratum_loader():
    from production import data as data_mod

    def fake_dataloader(**kwargs):
        return _FakeValLoader()

    old = data_mod.get_deterministic_validation_dataloader
    data_mod.get_deterministic_validation_dataloader = fake_dataloader
    return old


def test_debug_samples_have_all_sampler_keys(monkeypatch):
    """Every key debug_fn() reads must exist after _load_debug_samples."""
    monkeypatch.setattr(
        "production.data.get_deterministic_validation_dataloader",
        lambda **kwargs: _FakeValLoader(),
    )
    samples = visual_debug._load_debug_samples(
        shard_dir="unused",
        num_samples=4,
        device="cpu",
        source="stratum",
        stratum_dir="/unused",
        adapter_name="stratum",
    )
    assert len(samples) == 4
    for s in samples:
        # Keys read directly in visual_debug.debug_fn() (stratum adapter branch)
        for key in ("dino", "dino_patches", "text_emb", "text_mask", "pose_kpts"):
            assert key in s, f"missing sampler key: {key}"
        # get() with fallback — caption must resolve to a string
        assert isinstance(s.get("caption", ""), str)
        # shapes match what debug_fn unsqueezes to (B=1, ...)
        assert s["text_emb"].shape == (512, 1024)
        assert s["text_mask"].shape == (512,)
        assert s["pose_kpts"].shape == (133, 3)
        assert s["dino"].shape == (1024,)
        assert s["dino_patches"].ndim == 2 and s["dino_patches"].shape[1] == 1024


def test_alias_does_not_clobber_existing_keys(monkeypatch):
    """Pre-computed aliases must not overwrite real batch content."""
    monkeypatch.setattr(
        "production.data.get_deterministic_validation_dataloader",
        lambda **kwargs: _FakeValLoader(),
    )
    samples = visual_debug._load_debug_samples(
        shard_dir="unused",
        num_samples=4,
        device="cpu",
        source="stratum",
        stratum_dir="/unused",
        adapter_name="stratum",
    )
    for i, s in enumerate(samples):
        assert s["text_emb"] is s["t5_hidden"] or torch.equal(
            s["text_emb"], s["t5_hidden"]
        )
        assert s["pose_kpts"].shape == s["pose_keypoints"].shape
        assert s["caption"] == f"caption {i}"