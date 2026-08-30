"""Unit tests for the DiP-style output head (dip-conv-head arm, tick 1)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from production.model import NanoDiT, DiPHead


def _small_model(head_type, input_size=64, patch_size=8, hidden=64, depth=2, heads=4):
    return NanoDiT(
        input_size=input_size, patch_size=patch_size, in_channels=3,
        hidden_size=hidden, depth=depth, num_heads=heads, mlp_ratio=4.0,
        use_gradient_checkpointing=False, head_type=head_type,
    )


def test_default_head_is_legacy_linear():
    m = _small_model("linear")
    assert m.head_type == "linear"
    assert isinstance(m.output_conv, torch.nn.Conv2d)


def test_dip_head_is_dip():
    m = _small_model("dip")
    assert m.head_type == "dip"
    assert isinstance(m.output_conv, DiPHead)


def test_unknown_head_type_rejected():
    with pytest.raises(ValueError, match="unknown head_type"):
        _small_model("transformer")


def test_dip_head_forward_shape_and_finite():
    m = _small_model("dip")
    m.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = m(
            x, torch.tensor([0.5]),
            dino_emb=torch.randn(1, 1024),
            text_emb=torch.randn(1, 500, 1024),
            text_mask=torch.ones(1, 500, dtype=torch.bool),
            dino_patches=torch.randn(1, 64, 1024),
            dino_patches_mask=torch.ones(1, 64, dtype=torch.bool),
            pose_kpts=torch.randn(1, 133, 3),
        )
    assert out.shape == (1, 3, 64, 64)
    assert torch.isfinite(out).all()


def test_dip_head_zero_init_final():
    """Final layer zero-init => head output is all zeros at init (identity residual)."""
    m = _small_model("dip")
    assert (m.output_conv.up2.weight == 0).all()
    assert (m.output_conv.up2.bias == 0).all()


def test_dip_head_resolution_contract():
    """All production bucket sizes must be divisible by 4 (down/up contract)."""
    for h, w in [(1024, 1024), (1216, 832), (1280, 768), (1344, 704),
                 (704, 1344), (768, 1280), (832, 1216)]:
        assert h % 4 == 0 and w % 4 == 0, (h, w)


def test_dip_head_param_count_sane():
    m = _small_model("dip")
    head_params = sum(p.numel() for p in m.output_conv.parameters())
    assert head_params < 200_000  # shallow by design; ~150k with mid=32


def test_legacy_linear_output_zero_init_preserved():
    m = _small_model("linear")
    assert (m.output_conv.weight == 0).all()
    assert (m.output_conv.bias == 0).all()
