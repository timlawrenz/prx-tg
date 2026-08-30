"""Tests for GeometryAdapter — RED phase (tests written before implementation).

Tests:
  - shapes_in_zero_shot_batch: output shape matches (B, z_g_dim, hidden_size)
  - cfg_guard_drops_geometry: when all cfg_drop_geo=True, output matches null_geometry
  - basis_added_when_geo_live: when token_basis=True and no drop, tokens differ per dim
  - no_basis_symmetric_tokens: without token_basis, same z_g value produces same token
  - null_geometry_is_learned: null_geometry is a Parameter, not hardcoded zeros
"""

import torch
import pytest
import sys
from pathlib import Path

# Add the experiment src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "geometry-adapter-sd15" / "src"))

# This import will fail until we implement the module — that's the RED phase.
from geometry_adapter import GeometryAdapter


class TestGeometryAdapterShapes:
    """Test output tensor shapes."""

    def test_shapes_in_zero_shot_batch(self):
        """Output should be (batch, z_g_dim, hidden_size)."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        z_g = torch.randn(4, 50)
        tokens = adapter(z_g)
        assert tokens.shape == (4, 50, 768), f"Expected (4, 50, 768), got {tokens.shape}"

    def test_single_sample(self):
        """Single sample batch should work."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        z_g = torch.randn(1, 50)
        tokens = adapter(z_g)
        assert tokens.shape == (1, 50, 768)

    def test_different_hidden_sizes(self):
        """Adapter should work with any hidden_size."""
        for h in [384, 512, 768, 1024]:
            adapter = GeometryAdapter(hidden_size=h, z_g_dim=50, token_basis=True)
            tokens = adapter(torch.randn(2, 50))
            assert tokens.shape == (2, 50, h), f"Failed for hidden_size={h}"


class TestGeometryAdapterCFGGuard:
    """Test CFG dropout behavior."""

    def test_cfg_guard_drops_all_geometry(self):
        """When cfg_drop_geo is all True, output should match null_geometry with no basis."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        z_g = torch.randn(2, 50)
        drop_mask = torch.tensor([True, True])

        tokens_dropped = adapter(z_g, cfg_drop_geo=drop_mask)

        # Reference: manually pass null_geometry through with drop mask
        null_zg = adapter.null_geometry.expand(2, -1)
        tokens_ref = adapter(null_zg, cfg_drop_geo=drop_mask)

        assert torch.allclose(tokens_dropped, tokens_ref, atol=1e-6), \
            "Dropped random z_g should equal null_geometry passed through same path"

    def test_cfg_guard_mixed_batch(self):
        """Mixed drop mask: dropped sample should NOT carry basis (per-sample masking)."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        z_g = torch.randn(2, 50)
        drop_mask = torch.tensor([True, False])

        tokens = adapter(z_g, cfg_drop_geo=drop_mask)

        # Sample 0 (dropped) should not equal sample 1 (live)
        assert not torch.allclose(tokens[0], tokens[1], atol=1e-3), \
            "Dropped sample [0] should differ from live sample [1]"

        # Sample 0 should match null_geometry output (no basis contamination)
        null_zg = adapter.null_geometry.expand(1, -1)
        null_tokens = adapter(null_zg, cfg_drop_geo=torch.tensor([True]))
        assert torch.allclose(tokens[0], null_tokens[0], atol=1e-4), \
            "Dropped sample should match clean null output (no basis leak)"

    def test_no_drop_preserves_signal(self):
        """Without CFG drop, different z_g should produce different tokens."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        z_g_a = torch.randn(1, 50)
        z_g_b = z_g_a.clone()
        z_g_b[0, 0] += 2.0  # large change on dim0

        tokens_a = adapter(z_g_a, cfg_drop_geo=None)
        tokens_b = adapter(z_g_b, cfg_drop_geo=None)

        # Tokens should differ when z_g differs
        assert not torch.allclose(tokens_a, tokens_b, atol=1e-2), \
            "Different z_g inputs should produce different tokens"


class TestGeometryAdapterTokenBasis:
    """Test per-dimension token identity (geo_basis)."""

    def test_basis_creates_distinguishable_tokens(self):
        """With token_basis=True, each of the 50 positions should have distinct tokens."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        z_g = torch.zeros(1, 50)  # all zeros — only basis should differentiate

        tokens = adapter(z_g, cfg_drop_geo=None)  # (1, 50, 768)

        # Check that all 50 positions are distinct
        for i in range(50):
            for j in range(i + 1, 50):
                assert not torch.allclose(tokens[0, i], tokens[0, j], atol=1e-4), \
                    f"Positions {i} and {j} should differ with token_basis=True"

    def test_no_basis_symmetric_tokens(self):
        """Without token_basis, same z_g value should produce identical tokens."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=False)
        z_g = torch.zeros(1, 50)  # all same value

        tokens = adapter(z_g, cfg_drop_geo=None)  # (1, 50, 768)

        # All positions should be nearly identical (same input, shared MLP, no basis)
        ref = tokens[0, 0]
        for i in range(1, 50):
            assert torch.allclose(tokens[0, i], ref, atol=1e-6), \
                f"Position {i} should match position 0 without token_basis"

    def test_basis_not_added_when_cfg_dropped(self):
        """When all geometry is dropped, basis should NOT be added — all 50 positions identical."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        z_g = torch.randn(1, 50)
        drop_mask = torch.tensor([True])

        tokens = adapter(z_g, cfg_drop_geo=drop_mask)  # (1, 50, 768)

        # With basis skipped, all 50 positions from the shared MLP should be identical
        # (null_geometry is a single value repeated across dims)
        ref = tokens[0, 0]
        for i in range(1, 50):
            assert torch.allclose(tokens[0, i], ref, atol=1e-6), \
                f"Position {i} should match position 0 when basis is skipped"


class TestGeometryAdapterNullGeometry:
    """Test null_geometry parameter."""

    def test_null_geometry_is_learned_parameter(self):
        """null_geometry should be a nn.Parameter, not a plain tensor."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        assert isinstance(adapter.null_geometry, torch.nn.Parameter), \
            "null_geometry must be a nn.Parameter"

    def test_null_geometry_shape(self):
        """null_geometry should be (1, z_g_dim)."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        assert adapter.null_geometry.shape == (1, 50)

    def test_null_geometry_defaults_to_zeros(self):
        """null_geometry should initialize as zeros."""
        adapter = GeometryAdapter(hidden_size=768, z_g_dim=50, token_basis=True)
        assert torch.allclose(adapter.null_geometry, torch.zeros(1, 50), atol=1e-6), \
            "null_geometry should initialize to zeros"
