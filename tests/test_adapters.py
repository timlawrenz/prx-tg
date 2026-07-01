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
