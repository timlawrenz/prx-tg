"""Unit tests for StratumDataset._collate optional-key handling.

The batch-4 mixed-enrichment crash (KeyError: 'geometry_3d', 2026-08-30) is the
regression this guards: optional keys must be included only when ALL samples in
the batch carry them (batch[0]-only checks mix 308/133-style artifacts).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))

import torch  # noqa: E402
from data_stratum import _collate  # noqa: E402


def _sample(geometry_3d=True, matting=True):
    s = {
        'image_data': torch.zeros(3, 16, 16),
        'dino_embedding': torch.zeros(1024),
        'dinov3_patches': torch.zeros(64, 1024),
        't5_hidden': torch.zeros(512, 1024),
        't5_mask': torch.zeros(512, dtype=torch.long),
        'pose_keypoints': torch.zeros(133, 3),
        'seg_map': torch.zeros(16, 16, dtype=torch.long),
        'caption': 'c', 'image_id': 'i',
    }
    if geometry_3d:
        s['geometry_3d'] = torch.zeros(256, 6)
    if matting:
        s['matting'] = torch.zeros(16, 16)
    return s


def test_uniform_batch_keeps_keys():
    r = _collate([_sample(), _sample()])
    assert 'geometry_3d' in r and 'matting' in r
    assert r['geometry_3d'].shape == (2, 256, 6)


def test_mixed_batch_omits_optional_keys():
    # batch[0] has geometry_3d, batch[1] does NOT — must not raise, key omitted
    r = _collate([_sample(geometry_3d=True), _sample(geometry_3d=False)])
    assert 'geometry_3d' not in r
    assert 'matting' in r  # uniform for matting -> kept
    assert r['image_data'].shape == (2, 3, 16, 16)


def test_none_have_keys():
    r = _collate([_sample(geometry_3d=False, matting=False)] * 3)
    assert 'geometry_3d' not in r and 'matting' not in r
    assert r['pose_keypoints'].shape == (3, 133, 3)


def test_pose_padding_mixed_joints_still_pads_but_pose2_is_config_gated():
    """Padding handles shape variance; the 308-vs-133 mix itself is prevented by
    prefer_pose2 upstream (config-driven), not by the collate."""
    s = [_sample(), _sample()]
    s[0]['pose_keypoints'] = torch.zeros(60, 3)  # ragged (won't happen with the flag, but pads safely)
    r = _collate(s)
    assert r['pose_keypoints'].shape == (2, 133, 3)
    assert r['pose_mask'][0].sum() == 60
    assert r['pose_mask'][1].sum() == 133