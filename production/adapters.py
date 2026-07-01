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


# ── StratumAdapter: exact DINO+T5+pose replica ──────────────────────────────

class StratumAdapter(ConditioningAdapter):
    """Regression-gate adapter that exactly replicates current DINO+T5+pose behavior.

    Owns all projectors (dino_proj, dino_patch_proj, text_proj, pose_proj,
    pose_joint_embed) and all source-space null parameters.  Produces a
    concatenated cross-attention sequence and a combined mask — the DiTBlock
    receives the result as a single neutral sequence_cond + sequence_mask.
    """

    def __init__(
        self,
        hidden_size: int,
        dino_dim: int = 1024,
        dino_patch_dim: int = 1024,
        text_dim: int = 1024,
        dino_patches_enabled: bool = True,
        num_pose_joints: int = 133,
        pose_dim: int = 3,
        pose_confidence_threshold: float = 0.05,
        dino_pool_factor: Optional[int] = None,
    ):
        super().__init__(hidden_size)

        if dino_pool_factor is not None:
            raise NotImplementedError(
                "dino_pool_factor pooling is not ported to StratumAdapter")

        self.dino_patches_enabled = dino_patches_enabled
        self.num_pose_joints = num_pose_joints
        self.pose_confidence_threshold = pose_confidence_threshold

        # Conditioning projections (bias=True matching original NanoDiT)
        self.dino_proj = nn.Linear(dino_dim, hidden_size, bias=True)
        self.dino_patch_proj = nn.Linear(dino_patch_dim, hidden_size, bias=True)
        self.text_proj = nn.Linear(text_dim, hidden_size, bias=True)

        # Pose conditioning: MLP projector + learned joint-type embeddings
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.pose_joint_embed = nn.Embedding(num_pose_joints, hidden_size)

        # Source-space null embeddings for CFG dropout
        self.null_dino = nn.Parameter(torch.zeros(1, dino_dim))
        self.null_dino_patch_token = nn.Parameter(torch.zeros(1, 1, dino_patch_dim))
        self.null_text = nn.Parameter(torch.zeros(1, 1, text_dim))
        # null_pose is in post-projection space (matching original NanoDiT design)
        self.null_pose = nn.Parameter(torch.zeros(1, num_pose_joints, hidden_size))

    def forward(self, **kwargs) -> ConditioningOutput:
        dino_emb = kwargs["dino_emb"]                       # (B, dino_dim)
        text_emb = kwargs["text_emb"]                       # (B, T_len, text_dim)
        text_mask = kwargs.get("text_mask")                  # (B, T_len) or None
        dino_patches = kwargs.get("dino_patches")           # (B, P_len, dino_patch_dim) or None
        dino_patches_mask = kwargs.get("dino_patches_mask") # (B, P_len) or None
        t_emb = kwargs.get("t_emb")                         # (B, hidden_size) or None
        pose_kpts = kwargs.get("pose_kpts")                 # (B, 133, 3) or None

        B = dino_emb.shape[0]

        # ── CFG dropout (source-space) ────────────────────────────────────
        dino_emb = self.apply_cfg_drop_source(
            dino_emb, kwargs.get("cfg_drop_dino"), self.null_dino)

        text_emb = self.apply_cfg_drop_source(
            text_emb, kwargs.get("cfg_drop_text"), self.null_text)

        if dino_patches is not None:
            dino_patches = self.apply_cfg_drop_source(
                dino_patches, kwargs.get("cfg_drop_dino_patches"),
                self.null_dino_patch_token)

        # ── Global conditioning (adaLN) ───────────────────────────────────
        dino_cond = self.dino_proj(dino_emb)           # (B, hidden_size)
        if t_emb is not None:
            dino_cond = dino_cond + t_emb
        global_cond = dino_cond

        # ── Sequence conditioning (cross-attention) ───────────────────────
        text_cond = self.text_proj(text_emb)            # (B, T_len, hidden_size)
        dino_cls_token = global_cond.unsqueeze(1)       # (B, 1, hidden_size)

        # Build combined context: [text, CLS, (patches)]
        if dino_patches is not None and self.dino_patches_enabled:
            patches_cond = self.dino_patch_proj(dino_patches)  # (B, P_len, hidden_size)
        else:
            patches_cond = None

        # ── Pose conditioning ─────────────────────────────────────────────
        if pose_kpts is not None:
            cfg_drop_pose = kwargs.get("cfg_drop_pose")
            null_pose_expanded = self.null_pose.expand(B, -1, -1)  # (B, 133, hidden)

            if cfg_drop_pose is not None:
                pose_proj = self.pose_proj(pose_kpts) + self.pose_joint_embed.weight
                conf = pose_kpts[:, :, 2]
                low_conf = conf < self.pose_confidence_threshold
                pose_proj = torch.where(low_conf.unsqueeze(2), null_pose_expanded, pose_proj)
                pose_tokens = torch.where(
                    cfg_drop_pose.unsqueeze(1).unsqueeze(2),
                    null_pose_expanded, pose_proj,
                )
            else:
                pose_tokens = self.pose_proj(pose_kpts) + self.pose_joint_embed.weight
                conf = pose_kpts[:, :, 2]
                low_conf = conf < self.pose_confidence_threshold
                pose_tokens = torch.where(low_conf.unsqueeze(2), null_pose_expanded, pose_tokens)

            # Append pose tokens to patches_cond
            if patches_cond is not None:
                patches_cond = torch.cat([patches_cond, pose_tokens], dim=1)
                if dino_patches_mask is not None:
                    pose_mask = torch.ones(B, self.num_pose_joints,
                                           device=dino_patches_mask.device,
                                           dtype=dino_patches_mask.dtype)
                    dino_patches_mask = torch.cat([dino_patches_mask, pose_mask], dim=1)
            else:
                patches_cond = pose_tokens
                if text_mask is not None:
                    dino_patches_mask = torch.ones(B, self.num_pose_joints,
                                                   device=text_mask.device,
                                                   dtype=text_mask.dtype)

        # ── Assemble combined sequence ────────────────────────────────────
        if patches_cond is not None:
            combined_context = torch.cat([text_cond, dino_cls_token, patches_cond], dim=1)
        else:
            combined_context = torch.cat([text_cond, dino_cls_token], dim=1)

        # ── Assemble combined mask (replicating old DiTBlock logic) ───────
        if text_mask is not None:
            cls_mask = torch.ones(B, 1, device=text_mask.device, dtype=text_mask.dtype)
            if dino_patches_mask is not None:
                dino_patches_mask = dino_patches_mask.to(
                    device=text_mask.device, dtype=text_mask.dtype)
                cross_mask = torch.cat([text_mask, cls_mask, dino_patches_mask], dim=1)
            else:
                cross_mask = torch.cat([text_mask, cls_mask], dim=1)
        else:
            cross_mask = None

        return ConditioningOutput(
            global_cond=global_cond,
            sequence_cond=combined_context,
            sequence_mask=cross_mask,
        )


# ── EidolonAdapter: AuraFace-LDA (global) + z_g (sequence) ──────────────────

class EidolonAdapter(ConditioningAdapter):
    """Identity-driven conditioning: AuraFace-LDA for global, z_g for sequence.

    Drops T5 text and DINO patches entirely.  Sequence length is just 50 tokens
    (one per z_g component), providing a ~99% reduction in cross-attention FLOPs.
    """

    def __init__(self, hidden_size: int, identity_dim: int = 64, z_g_dim: int = 50):
        super().__init__(hidden_size)
        self.identity_proj = nn.Linear(identity_dim, hidden_size, bias=True)
        self.geometry_proj = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.null_identity = nn.Parameter(torch.zeros(1, identity_dim))
        self.null_geometry = nn.Parameter(torch.zeros(1, z_g_dim))
        self.z_g_dim = z_g_dim

    def forward(self, **kwargs) -> ConditioningOutput:
        identity_emb = kwargs["identity_emb"]
        geometry_emb = kwargs["geometry_emb"]
        t_emb = kwargs.get("t_emb")

        identity_emb = self.apply_cfg_drop_source(
            identity_emb, kwargs.get("cfg_drop_identity"), self.null_identity)
        geometry_emb = self.apply_cfg_drop_source(
            geometry_emb, kwargs.get("cfg_drop_geometry"), self.null_geometry)

        identity_cond = self.identity_proj(identity_emb)
        global_cond = identity_cond + (t_emb if t_emb is not None else 0)
        geometry_cond = self.geometry_proj(geometry_emb.unsqueeze(-1))  # (B, 50, hidden)
        B = geometry_cond.shape[0]

        return ConditioningOutput(
            global_cond=global_cond,
            sequence_cond=geometry_cond,
            sequence_mask=torch.ones(B, self.z_g_dim, device=geometry_cond.device),
        )
