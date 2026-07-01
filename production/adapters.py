"""Neutral conditioning adapters for the DiT."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConditioningOutput:
    global_cond: torch.Tensor          # (B, hidden_size)
    sequence_cond: torch.Tensor        # (B, S, hidden_size)
    sequence_mask: Optional[torch.Tensor]  # (B, S) or None


class ConditioningAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    @staticmethod
    def apply_cfg_drop_source(source_emb, drop_mask, null_emb):
        if drop_mask is not None:
            null = null_emb.expand(source_emb.shape[0], *source_emb.shape[1:])
            source_emb = torch.where(
                drop_mask.view(-1, *([1] * (source_emb.ndim - 1))), null, source_emb)
        return source_emb

    def forward(self, **kwargs) -> ConditioningOutput:
        raise NotImplementedError
