"""Tests for production/adapters.py — neutral conditioning adapter interface."""

import torch
import pytest
from production.adapters import ConditioningOutput, ConditioningAdapter


# ── Task 1: ConditioningAdapter base class ──────────────────────────────────

class TestConditioningOutput:
    def test_creates_dataclass_with_expected_fields(self):
        """ConditioningOutput is a dataclass with global_cond, sequence_cond, sequence_mask."""
        B, S, H = 2, 64, 768
        global_cond = torch.randn(B, H)
        sequence_cond = torch.randn(B, S, H)
        sequence_mask = torch.ones(B, S)

        out = ConditioningOutput(
            global_cond=global_cond,
            sequence_cond=sequence_cond,
            sequence_mask=sequence_mask,
        )

        assert out.global_cond is global_cond
        assert out.sequence_cond is sequence_cond
        assert out.sequence_mask is sequence_mask

    def test_sequence_mask_can_be_none(self):
        """sequence_mask is Optional — None is valid."""
        out = ConditioningOutput(
            global_cond=torch.randn(2, 768),
            sequence_cond=torch.randn(2, 64, 768),
            sequence_mask=None,
        )
        assert out.sequence_mask is None


class TestConditioningAdapterBase:
    def test_forward_raises_not_implemented(self):
        """Base ConditioningAdapter.forward raises NotImplementedError (abstract)."""
        adapter = ConditioningAdapter(hidden_size=768)
        with pytest.raises(NotImplementedError):
            adapter()

    def test_hidden_size_stored(self):
        """Constructor stores hidden_size."""
        adapter = ConditioningAdapter(hidden_size=512)
        assert adapter.hidden_size == 512

    def test_is_nn_module(self):
        """ConditioningAdapter is a torch.nn.Module."""
        import torch.nn as nn
        adapter = ConditioningAdapter(hidden_size=768)
        assert isinstance(adapter, nn.Module)


class TestApplyCfgDropSource:
    def test_no_drop_mask_returns_original(self):
        """When drop_mask is None, source_emb is returned unchanged."""
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        null = torch.zeros(1, 2)
        result = ConditioningAdapter.apply_cfg_drop_source(source, None, null)
        assert torch.equal(result, source)

    def test_drop_mask_replaces_with_null(self):
        """When drop_mask is True for a batch element, that row becomes null."""
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        null = torch.tensor([[-1.0, -1.0]])
        drop_mask = torch.tensor([True, False])
        result = ConditioningAdapter.apply_cfg_drop_source(source, drop_mask, null)
        expected = torch.tensor([[-1.0, -1.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_drop_mask_partial_batch(self):
        """Only masked rows are replaced."""
        B, D = 4, 8
        source = torch.ones(B, D)
        null = torch.zeros(1, D)
        drop_mask = torch.tensor([True, False, True, False])
        result = ConditioningAdapter.apply_cfg_drop_source(source, drop_mask, null)
        assert torch.equal(result[0], torch.zeros(D))
        assert torch.equal(result[1], torch.ones(D))
        assert torch.equal(result[2], torch.zeros(D))
        assert torch.equal(result[3], torch.ones(D))

    def test_drop_mask_3d_tensor(self):
        """Works with 3D tensors (sequence conditioning)."""
        B, S, D = 2, 5, 8
        source = torch.ones(B, S, D)
        null = torch.zeros(1, S, D)
        drop_mask = torch.tensor([True, False])
        result = ConditioningAdapter.apply_cfg_drop_source(source, drop_mask, null)
        assert result.shape == (B, S, D)
        assert (result[0] == 0).all()
        assert (result[1] == 1).all()


# ── Task 2: StratumAdapter ──────────────────────────────────────────────────

class TestStratumAdapterShape:
    """Shape tests for StratumAdapter."""

    def test_output_shapes_basic(self):
        """Produces correct shapes for global_cond, sequence_cond, sequence_mask."""
        from production.adapters import StratumAdapter
        B, H = 2, 768
        T_len, P_len = 77, 256

        adapter = StratumAdapter(
            hidden_size=H, dino_patches_enabled=True,
            dino_dim=1024, text_dim=1024, dino_patch_dim=1024,
        )

        dino_emb = torch.randn(B, 1024)
        text_emb = torch.randn(B, T_len, 1024)
        text_mask = torch.ones(B, T_len)
        dino_patches = torch.randn(B, P_len, 1024)
        dino_patches_mask = torch.ones(B, P_len)

        out = adapter(
            dino_emb=dino_emb,
            text_emb=text_emb,
            text_mask=text_mask,
            dino_patches=dino_patches,
            dino_patches_mask=dino_patches_mask,
        )

        assert out.global_cond.shape == (B, H)
        assert out.sequence_cond.shape == (B, T_len + 1 + P_len, H)
        assert out.sequence_mask.shape == (B, T_len + 1 + P_len)

    def test_without_patches(self):
        """When dino_patches not provided, sequence is [text, CLS] only."""
        from production.adapters import StratumAdapter
        B, H = 2, 768
        T_len = 77

        adapter = StratumAdapter(
            hidden_size=H, dino_patches_enabled=True,
            dino_dim=1024, text_dim=1024, dino_patch_dim=1024,
        )

        dino_emb = torch.randn(B, 1024)
        text_emb = torch.randn(B, T_len, 1024)
        text_mask = torch.ones(B, T_len)

        out = adapter(
            dino_emb=dino_emb,
            text_emb=text_emb,
            text_mask=text_mask,
        )

        assert out.global_cond.shape == (B, H)
        assert out.sequence_cond.shape == (B, T_len + 1, H)
        assert out.sequence_mask.shape == (B, T_len + 1)

    def test_with_t_emb(self):
        """t_emb is added to global_cond."""
        from production.adapters import StratumAdapter
        B, H = 2, 768

        adapter = StratumAdapter(
            hidden_size=H, dino_patches_enabled=False,
            dino_dim=1024, text_dim=1024,
        )

        dino_emb = torch.zeros(B, 1024)
        text_emb = torch.randn(B, 32, 1024)
        text_mask = torch.ones(B, 32)
        t_emb = torch.ones(B, H)

        out_no_t = adapter(dino_emb=dino_emb, text_emb=text_emb, text_mask=text_mask)
        out_with_t = adapter(dino_emb=dino_emb, text_emb=text_emb, text_mask=text_mask, t_emb=t_emb)

        # t_emb is added, so with_t should differ from without
        assert not torch.allclose(out_no_t.global_cond, out_with_t.global_cond)


class TestStratumAdapterCFG:
    """CFG dropout tests for StratumAdapter."""

    def test_dino_cfg_drop(self):
        """DINO CFG drop replaces with null."""
        from production.adapters import StratumAdapter
        B, H = 2, 768

        adapter = StratumAdapter(
            hidden_size=H, dino_patches_enabled=False,
            dino_dim=1024, text_dim=1024,
        )

        dino_emb = torch.randn(B, 1024)
        text_emb = torch.randn(B, 32, 1024)
        text_mask = torch.ones(B, 32)

        out_clean = adapter(dino_emb=dino_emb, text_emb=text_emb, text_mask=text_mask)
        out_drop = adapter(dino_emb=dino_emb, text_emb=text_emb, text_mask=text_mask,
                           cfg_drop_dino=torch.tensor([True, False]))

        # First item should differ (null), second should match
        assert not torch.allclose(out_drop.global_cond[0], out_clean.global_cond[0])
        assert torch.allclose(out_drop.global_cond[1], out_clean.global_cond[1])

    def test_text_cfg_drop(self):
        """Text CFG drop replaces with null."""
        from production.adapters import StratumAdapter
        B, H = 2, 768

        adapter = StratumAdapter(
            hidden_size=H, dino_patches_enabled=False,
            dino_dim=1024, text_dim=1024,
        )

        dino_emb = torch.randn(B, 1024)
        text_emb = torch.randn(B, 32, 1024)
        text_mask = torch.ones(B, 32)

        out_clean = adapter(dino_emb=dino_emb, text_emb=text_emb, text_mask=text_mask)
        out_drop = adapter(dino_emb=dino_emb, text_emb=text_emb, text_mask=text_mask,
                           cfg_drop_text=torch.tensor([True, False]))

        assert not torch.allclose(out_drop.sequence_cond[0], out_clean.sequence_cond[0])
        assert torch.allclose(out_drop.sequence_cond[1], out_clean.sequence_cond[1])


class TestStratumAdapterPoolingGuard:
    """dino_pool_factor is not ported — must raise NotImplementedError."""

    def test_pooling_guard(self):
        """Passing dino_pool_factor raises NotImplementedError."""
        from production.adapters import StratumAdapter
        with pytest.raises(NotImplementedError):
            StratumAdapter(
                hidden_size=768,
                dino_dim=1024, text_dim=1024,
                dino_patches_enabled=True,
                dino_pool_factor=2,
            )


class TestStratumAdapterMaskRegression:
    """Verify combined mask matches what the old DiTBlock produced."""

    def _old_ditblock_combined_mask(self, text_mask, patches_mask):
        """Replicate the mask assembly that DiTBlock._forward_impl used to do."""
        B = text_mask.shape[0]
        cls_mask = torch.ones(B, 1, device=text_mask.device, dtype=text_mask.dtype)
        if patches_mask is not None:
            patches_mask = patches_mask.to(device=text_mask.device, dtype=text_mask.dtype)
            return torch.cat([text_mask, cls_mask, patches_mask], dim=1)
        else:
            return torch.cat([text_mask, cls_mask], dim=1)

    def test_mask_without_patches_matches_old_behavior(self):
        """Mask = [text_mask, ones(B,1)] — identical to old DiTBlock."""
        from production.adapters import StratumAdapter
        B, H = 2, 768
        T_len = 77

        adapter = StratumAdapter(
            hidden_size=H, dino_patches_enabled=False,
            dino_dim=1024, text_dim=1024,
        )

        dino_emb = torch.randn(B, 1024)
        text_emb = torch.randn(B, T_len, 1024)
        text_mask = torch.ones(B, T_len)
        text_mask[0, 40:] = 0  # variable length text

        out = adapter(dino_emb=dino_emb, text_emb=text_emb, text_mask=text_mask)

        expected = self._old_ditblock_combined_mask(text_mask, None)
        assert torch.equal(out.sequence_mask, expected)

    def test_mask_with_patches_matches_old_behavior(self):
        """Mask = [text_mask, ones(B,1), patches_mask] — identical to old DiTBlock."""
        from production.adapters import StratumAdapter
        B, H = 2, 768
        T_len, P_len = 77, 256

        adapter = StratumAdapter(
            hidden_size=H, dino_patches_enabled=True,
            dino_dim=1024, text_dim=1024, dino_patch_dim=1024,
        )

        dino_emb = torch.randn(B, 1024)
        text_emb = torch.randn(B, T_len, 1024)
        text_mask = torch.ones(B, T_len)
        text_mask[0, 30:] = 0  # shorter text
        dino_patches = torch.randn(B, P_len, 1024)
        patches_mask = torch.ones(B, P_len)
        patches_mask[0, 200:] = 0  # shorter patches

        out = adapter(
            dino_emb=dino_emb, text_emb=text_emb, text_mask=text_mask,
            dino_patches=dino_patches, dino_patches_mask=patches_mask,
        )

        expected = self._old_ditblock_combined_mask(text_mask, patches_mask)
        assert torch.equal(out.sequence_mask, expected)

    def test_none_text_mask_produces_none_cross_mask(self):
        """When text_mask is None, the old DiTBlock produced None — adapter should too."""
        from production.adapters import StratumAdapter
        B, H = 2, 768

        adapter = StratumAdapter(
            hidden_size=H, dino_patches_enabled=False,
            dino_dim=1024, text_dim=1024,
        )

        dino_emb = torch.randn(B, 1024)
        text_emb = torch.randn(B, 32, 1024)

        out = adapter(dino_emb=dino_emb, text_emb=text_emb)

        assert out.sequence_mask is None


# ── Task 3: EidolonAdapter ──────────────────────────────────────────────────

class TestEidolonAdapterShape:
    """Shape tests for EidolonAdapter."""

    def test_output_shapes(self):
        """Produces correct shapes: global_cond (B,H), sequence_cond (B,50,H), mask (B,50)."""
        from production.adapters import EidolonAdapter
        B, H = 2, 768
        identity_dim = 64
        z_g_dim = 50

        adapter = EidolonAdapter(
            hidden_size=H, identity_dim=identity_dim, z_g_dim=z_g_dim,
        )

        identity_emb = torch.randn(B, identity_dim)
        geometry_emb = torch.randn(B, z_g_dim)

        out = adapter(identity_emb=identity_emb, geometry_emb=geometry_emb)

        assert out.global_cond.shape == (B, H)
        assert out.sequence_cond.shape == (B, z_g_dim, H)
        assert out.sequence_mask.shape == (B, z_g_dim)
        assert (out.sequence_mask == 1).all()

    def test_with_t_emb(self):
        """t_emb is added to identity_cond for global conditioning."""
        from production.adapters import EidolonAdapter
        B, H = 2, 768

        adapter = EidolonAdapter(hidden_size=H, identity_dim=64, z_g_dim=50)

        identity_emb = torch.zeros(B, 64)
        geometry_emb = torch.randn(B, 50)
        t_emb = torch.ones(B, H)

        out_no_t = adapter(identity_emb=identity_emb, geometry_emb=geometry_emb)
        out_with_t = adapter(identity_emb=identity_emb, geometry_emb=geometry_emb, t_emb=t_emb)

        # t_emb is added, so global_cond should differ
        assert not torch.allclose(out_no_t.global_cond, out_with_t.global_cond)
        # But sequence_cond should be unaffected (geometry only)
        assert torch.allclose(out_no_t.sequence_cond, out_with_t.sequence_cond)


class TestEidolonAdapterCFG:
    """CFG dropout tests for EidolonAdapter."""

    def test_identity_cfg_drop(self):
        """Identity CFG drop replaces with null."""
        from production.adapters import EidolonAdapter
        B, H = 2, 768

        adapter = EidolonAdapter(hidden_size=H, identity_dim=64, z_g_dim=50)

        identity_emb = torch.randn(B, 64)
        geometry_emb = torch.randn(B, 50)

        out_clean = adapter(identity_emb=identity_emb, geometry_emb=geometry_emb)
        out_drop = adapter(
            identity_emb=identity_emb, geometry_emb=geometry_emb,
            cfg_drop_identity=torch.tensor([True, False]),
        )

        # First item should differ (null identity), second should match
        assert not torch.allclose(out_drop.global_cond[0], out_clean.global_cond[0])
        assert torch.allclose(out_drop.global_cond[1], out_clean.global_cond[1])

    def test_geometry_cfg_drop(self):
        """Geometry CFG drop replaces geometry tokens with null."""
        from production.adapters import EidolonAdapter
        B, H = 2, 768

        adapter = EidolonAdapter(hidden_size=H, identity_dim=64, z_g_dim=50)

        identity_emb = torch.randn(B, 64)
        geometry_emb = torch.randn(B, 50)

        out_clean = adapter(identity_emb=identity_emb, geometry_emb=geometry_emb)
        out_drop = adapter(
            identity_emb=identity_emb, geometry_emb=geometry_emb,
            cfg_drop_geometry=torch.tensor([True, False]),
        )

        # First item's sequence_cond should differ (null geometry), second should match
        assert not torch.allclose(out_drop.sequence_cond[0], out_clean.sequence_cond[0])
        assert torch.allclose(out_drop.sequence_cond[1], out_clean.sequence_cond[1])
        # Global cond should be unaffected (identity not dropped)
        assert torch.allclose(out_drop.global_cond, out_clean.global_cond)

    def test_both_cfg_drop(self):
        """Both identity and geometry dropped simultaneously."""
        from production.adapters import EidolonAdapter
        B, H = 2, 768

        adapter = EidolonAdapter(hidden_size=H, identity_dim=64, z_g_dim=50)

        identity_emb = torch.randn(B, 64)
        geometry_emb = torch.randn(B, 50)

        out_clean = adapter(identity_emb=identity_emb, geometry_emb=geometry_emb)
        out_drop = adapter(
            identity_emb=identity_emb, geometry_emb=geometry_emb,
            cfg_drop_identity=torch.tensor([True, False]),
            cfg_drop_geometry=torch.tensor([True, False]),
        )

        # First item: both dropped → differs from clean
        assert not torch.allclose(out_drop.global_cond[0], out_clean.global_cond[0])
        assert not torch.allclose(out_drop.sequence_cond[0], out_clean.sequence_cond[0])
        # Second item: neither dropped → matches clean
        assert torch.allclose(out_drop.global_cond[1], out_clean.global_cond[1])
        assert torch.allclose(out_drop.sequence_cond[1], out_clean.sequence_cond[1])
