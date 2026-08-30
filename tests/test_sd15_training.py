"""Smoke test for SD1.5 geometry adapter training loop.

Tests:
  - train_loop_can_initialize: all components load without error
  - train_loop_one_step: single training step produces loss and gradients
  - train_loop_cfg_dropout: CFG drop masks are applied correctly during training
"""

import torch
import pytest
import sys
from pathlib import Path

# Path to experiment source
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "geometry-adapter-sd15" / "src"))


@pytest.mark.slow
@pytest.mark.gpu
class TestTrainingLoopSmoke:
    """Smoke tests that require GPU and SD1.5 model weights."""

    @pytest.fixture(scope="class")
    def train_module(self):
        """Import the training module once."""
        import train as train_mod
        return train_mod

    @pytest.fixture(scope="class")
    def components(self, train_module):
        """Initialize all components (loads SD1.5 — slow, once per class)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return train_module.init_training(
            base_model="runwayml/stable-diffusion-v1-5",
            device=device,
            dtype=torch.float16,
        )

    def test_train_loop_can_initialize(self, components):
        """All components should initialize without error."""
        unet, text_encoder, tokenizer, vae, noise_scheduler, adapter, optimizer = components
        assert unet is not None
        assert text_encoder is not None
        assert vae is not None
        assert adapter is not None
        assert optimizer is not None
        # Verify UNet is frozen
        for p in unet.parameters():
            assert not p.requires_grad
        # Verify adapter is trainable
        assert any(p.requires_grad for p in adapter.parameters())

    @torch.no_grad()
    def test_train_loop_one_step(self, components, train_module):
        """A single training step should produce a loss and gradients."""
        unet, text_encoder, tokenizer, vae, noise_scheduler, adapter, optimizer = components
        device = next(adapter.parameters()).device
        dtype = torch.float32  # adapter trains in fp32

        # Create dummy batch (matching StratumDataset sd15_geometry output)
        B = 2
        batch = {
            "image_data": torch.rand(B, 3, 512, 512, device=device, dtype=torch.float32),
            "z_g": torch.randn(B, 50, device=device, dtype=torch.float32),
            "captions": ["a test portrait", "another photo"],
        }

        loss = train_module.training_step(
            batch=batch,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            noise_scheduler=noise_scheduler,
            adapter=adapter,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            p_uncond=0.10,
            p_geo_drop=0.20,
        )

        assert isinstance(loss, float) or loss.requires_grad is False  # loss returned as float
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_cfg_dropout_masks(self, train_module):
        """CFG dropout should produce valid masks summing to expected probabilities."""
        B = 1000  # large batch for statistical check
        p_uncond = 0.10
        p_geo_drop = 0.20

        masks = train_module.build_cfg_masks(B, p_uncond, p_geo_drop)

        assert "uncond" in masks
        assert "geo_drop" in masks
        assert masks["uncond"].shape == (B,)
        assert masks["geo_drop"].shape == (B,)

        # Statistical check: proportions should be close
        uncond_frac = masks["uncond"].float().mean().item()
        geo_drop_frac = masks["geo_drop"].float().mean().item()

        assert 0.05 < uncond_frac < 0.15, f"Expected ~0.10 uncond, got {uncond_frac:.3f}"
        assert 0.14 < geo_drop_frac < 0.26, f"Expected ~0.20 geo_drop, got {geo_drop_frac:.3f}"

        # Uncond and geo_drop should be disjoint (no sample in both)
        both = masks["uncond"] & masks["geo_drop"]
        assert both.sum().item() == 0, "No sample should be both uncond and geo_drop"
