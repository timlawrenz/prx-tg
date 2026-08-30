"""Tests for geometry adapter evaluation pipeline.

Tests (non-GPU / fast):
  - generate_sweep_grid creates correct layout
  - load_adapter_checkpoint restores weights
  - z_g_sweep_values produces correct grid
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "geometry-adapter-sd15" / "src"))


class TestEvalUtilities:
    """Test evaluation utility functions (no GPU needed)."""

    def test_z_g_sweep_values(self):
        """Sweep should vary one dimension while keeping others at zero."""
        import eval_geometry_adapter as ev

        values = ev.build_sweep_z_g(
            dim=0,
            sweep_range=(-2.0, 2.0),
            n_steps=5,
            z_g_dim=50,
        )
        assert values.shape == (5, 50)
        # dim0 should vary
        expected = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert torch.allclose(values[:, 0], expected)
        # All other dims should be zero
        assert torch.allclose(values[:, 1:], torch.zeros(5, 49))
        assert values.dtype == torch.float32

    def test_z_g_sweep_different_dim(self):
        """Sweep should work on any dimension."""
        import eval_geometry_adapter as ev

        values = ev.build_sweep_z_g(dim=3, sweep_range=(-1.0, 1.0), n_steps=3, z_g_dim=50)
        assert values.shape == (3, 50)
        assert torch.allclose(values[:, 3], torch.tensor([-1.0, 0.0, 1.0]))

    def test_load_adapter_checkpoint(self, tmp_path):
        """Should load adapter weights from a checkpoint file."""
        from geometry_adapter import GeometryAdapter
        import eval_geometry_adapter as ev

        # Create adapter, save checkpoint
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        ckpt = tmp_path / "adapter_step001000.pt"
        torch.save({"adapter_state_dict": adapter.state_dict(), "step": 1000}, ckpt)

        # Load
        loaded = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        ev.load_adapter_checkpoint(loaded, str(ckpt))

        # Weights should match
        for (n1, p1), (n2, p2) in zip(
            adapter.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter {n1} mismatch after load"

    def test_load_adapter_checkpoint_no_token_basis(self, tmp_path):
        """Should load a checkpoint saved without token_basis into compatible model."""
        from geometry_adapter import GeometryAdapter
        import eval_geometry_adapter as ev

        # Save without basis
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=False)
        ckpt = tmp_path / "adapter_no_basis.pt"
        torch.save({"adapter_state_dict": adapter.state_dict(), "step": 1000}, ckpt)

        # Load into without-basis model (should work)
        loaded_false = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=False)
        ev.load_adapter_checkpoint(loaded_false, str(ckpt))
        assert not loaded_false.token_basis

    def test_sweep_grid_layout(self):
        """Sweep grid should have correct number of rows/cols."""
        import eval_geometry_adapter as ev

        n_values = 5
        n_prompts = 3
        rows, cols = ev.grid_layout(n_values, n_prompts)
        assert rows == n_values  # one row per z_g value
        assert cols == n_prompts  # one column per prompt
