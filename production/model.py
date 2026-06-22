"""Nano DiT model for validation testing.

Architecture: 12 layers, 384 hidden dim, 6 attention heads (~40M params)
Conditioning:
  - DINOv3 CLS (1024) → adaLN-Zero (global style/timing)
  - T5 text (500×1024) → Cross-Attention (semantic concepts)
  - DINOv3 patches (~3880×1024) → Cross-Attention (spatial layout)
  
Cross-attention receives concatenated sequence: [T5, DINO_CLS, DINO_patches]
where DINO_CLS serves as a global fallback token and patches provide spatial alignment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """Apply adaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vectors using sinusoidal encoding."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * 1000.0, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    """Embed image patches (VAE latents or raw pixels) into hidden dimension."""
    
    def __init__(self, patch_size=16, in_channels=3, hidden_size=384, bottleneck_size=0):
        super().__init__()
        self.patch_size = patch_size
        self.bottleneck_size = bottleneck_size
        if bottleneck_size > 0:
            self.proj = nn.Conv2d(in_channels, bottleneck_size, kernel_size=patch_size, stride=patch_size)
            self.expand = nn.Linear(bottleneck_size, hidden_size)
        else:
            self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
            self.expand = None

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, bottleneck or hidden_size, H/ps, W/ps)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        if self.expand is not None:
            x = self.expand(x)  # (B, H*W, hidden_size)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: embedding dimension (must be even)
        grid_size: int or tuple (H, W)
    
    Returns:
        pos_embed: (H*W, embed_dim)
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size
    
    grid_h_coords = torch.arange(grid_h, dtype=torch.float32)
    grid_w_coords = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h_coords, grid_w_coords, indexing='ij')
    grid = torch.stack(grid, dim=0)  # (2, H, W)
    grid = grid.reshape(2, -1).T  # (H*W, 2)
    
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sinusoidal embeddings from position grid.
    
    Args:
        embed_dim: output dimension for each position
        pos: (M, 2) array of positions
    
    Returns:
        emb: (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega /= embed_dim / 4.
    omega = 1. / 10000**omega  # (embed_dim/4,)
    
    # pos: (M, 2) with H and W coordinates
    # Generate embeddings separately for H and W, then concatenate
    out_h = torch.einsum('m,d->md', pos[:, 0], omega)  # (M, embed_dim/4)
    out_w = torch.einsum('m,d->md', pos[:, 1], omega)  # (M, embed_dim/4)
    
    # Apply sin/cos to both
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (M, embed_dim/2)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (M, embed_dim/2)
    
    # Concatenate H and W embeddings
    emb = torch.cat([emb_h, emb_w], dim=1)  # (M, embed_dim)
    return emb


class Attention(nn.Module):
    """Multi-head attention (self or cross)."""
    
    def __init__(self, dim, num_heads=6, qkv_bias=False, is_cross_attn=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.is_cross_attn = is_cross_attn
        
        if is_cross_attn:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, context=None, mask=None):
        """
        Args:
            x: (B, N, C) query tokens
            context: (B, M, C) for cross-attention, None for self-attention
            mask: (B, M) attention mask for cross-attention (1=attend, 0=ignore)
                  OR (B, 1, N, M) per-query-token mask for spatial windowing
        """
        B, N, C = x.shape
        
        if self.is_cross_attn:
            assert context is not None
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(context).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            M = context.shape[1]
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            M = N
        
        # Use memory-efficient scaled dot product attention
        if mask is not None:
            if mask.dim() == 4:
                # Already a 4D per-query-token mask: (B, 1, N, M)
                attn_mask = mask.bool()
            else:
                # 2D mask: (B, M) → broadcast to (B, 1, 1, M)
                attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            x = F.scaled_dot_product_attention(q, k, v)
            
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    """MLP block: hidden_dim -> mlp_dim -> hidden_dim"""
    
    def __init__(self, hidden_dim, mlp_dim=None):
        super().__init__()
        mlp_dim = mlp_dim or hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(mlp_dim, hidden_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero (DINO) and cross-attention (T5)."""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, is_cross_attn=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, is_cross_attn=True)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        
        # adaLN modulation (6 params: scale/shift for norm1, norm2, norm3)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Zero-init the adaLN gate (critical for stability)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def _forward_impl(self, x, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask=None, x_mask=None, spatial_cross_mask=None):
        """Internal forward implementation for checkpointing."""
        # Get adaLN modulation parameters from DINOv3
        shift_msa, scale_msa, shift_ca, scale_ca, shift_mlp, scale_mlp = \
            self.adaLN_modulation(c_dino).chunk(6, dim=1)
        
        # Self-attention with adaLN
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask=x_mask)
        
        # Concatenate cross-attention sequence: [T5 text, DINO CLS, DINO patches]
        combined_context = torch.cat([c_text, c_dino_cls_token, c_patches], dim=1)
        
        if spatial_cross_mask is not None:
            # Use pre-computed 4D spatial mask directly: (B, 1, N, M)
            cross_mask = spatial_cross_mask
        else:
            # Build 2D combined mask from text + patches masks
            B = x.shape[0]
            
            if text_mask is not None:
                cls_mask = torch.ones(B, 1, device=text_mask.device, dtype=text_mask.dtype)
                
                if patches_mask is None:
                    patches_mask = torch.ones(B, c_patches.shape[1], device=text_mask.device, dtype=text_mask.dtype)
                else:
                    patches_mask = patches_mask.to(device=text_mask.device, dtype=text_mask.dtype)
                    
                cross_mask = torch.cat([text_mask, cls_mask, patches_mask], dim=1)  # (B, seq + 1 + num_patches)
            else:
                cross_mask = None
        
        # Cross-attention to combined sequence with adaLN
        x = x + self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca),
            context=combined_context,
            mask=cross_mask
        )
        
        # MLP with adaLN
        x = x + self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        
        return x

    def forward(self, x, c_dino, c_text, text_mask=None, c_dino_cls_token=None, c_patches=None, patches_mask=None, x_mask=None, spatial_cross_mask=None):
        """
        Args:
            x: (B, N, C) latent tokens
            c_dino: (B, C) DINOv3 conditioning for adaLN
            c_text: (B, M, C) T5 conditioning
            text_mask: (B, M) attention mask for T5
            c_dino_cls_token: (B, 1, C) DINO CLS token for cross-attention
            c_patches: (B, num_patches, C) DINO patch tokens for cross-attention (variable length)
            patches_mask: (B, num_patches) attention mask for DINO patches
            x_mask: (B, N) attention mask for self-attention
            spatial_cross_mask: (B, 1, N, M) per-query-token cross-attention mask (optional)
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask, x_mask, spatial_cross_mask, use_reentrant=False
            )
        else:
            return self._forward_impl(x, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask, x_mask, spatial_cross_mask)

class MaskDiTDecoder(nn.Module):
    """Lightweight decoder for MaskDiT masked token reconstruction.
    
    Receives the full token sequence (visible encoder outputs + learned mask tokens)
    and produces predictions for all positions. Uses the same DiTBlock architecture
    but with fewer layers.
    """
    
    def __init__(self, hidden_size, num_heads, depth=4, mlp_ratio=4.0, use_checkpoint=False):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_checkpoint=use_checkpoint)
            for _ in range(depth)
        ])
    
    def forward(self, x_visible, visible_idx, masked_idx, N_total, pos_embed,
                c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask=None):
        """
        Args:
            x_visible: (B, N_vis, C) encoder output for visible tokens
            visible_idx: (N_vis,) indices of visible tokens
            masked_idx: (N_mask,) indices of masked tokens
            N_total: int, total number of tokens
            pos_embed: (1, N_total, C) positional embeddings for all tokens
            c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask: conditioning
        
        Returns:
            x_full: (B, N_total, C) decoded output for all token positions
        """
        B, _, C = x_visible.shape
        
        # Create full sequence: visible outputs + mask tokens
        mask_tokens = self.mask_token.expand(B, len(masked_idx), -1)
        
        # Add positional embeddings to mask tokens
        mask_tokens = mask_tokens + pos_embed[:, masked_idx]
        
        # Assemble full sequence
        x_full = torch.empty(B, N_total, C, device=x_visible.device, dtype=x_visible.dtype)
        x_full[:, visible_idx] = x_visible
        x_full[:, masked_idx] = mask_tokens
        
        # Run through decoder blocks
        for block in self.blocks:
            x_full = block(x_full, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask)
        
        return x_full


class NanoDiT(nn.Module):
    """Nano DiT: 12L, 384H, 6A for validation testing."""
    
    def __init__(
        self,
        input_size=64,  # Latent spatial size - IGNORED for dynamic pos embed
        patch_size=16,
        in_channels=3,  # RGB pixel channels
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dino_dim=1024,
        dino_patch_dim=1024,
        text_dim=1024,
        use_gradient_checkpointing=False,
        repa_block_idx=None,
        tread_route_start=None,
        tread_route_end=None,
        tread_routing_prob=0.5,
        bottleneck_size=0,
        num_pose_joints=133,
        pose_dim=3,
        pose_confidence_threshold=0.05,
        maskdit_enabled=False,
        maskdit_mask_ratio=0.75,
        maskdit_decoder_depth=4,
        spatial_window_radius=None,   # DINO patch cross-attn spatial window (None = all patches)
    ):
        super().__init__()
        self.input_size = input_size  # For backward compatibility, but not used
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.repa_block_idx = repa_block_idx
        self.tread_route_start = tread_route_start
        self.tread_route_end = tread_route_end
        self.tread_routing_prob = tread_routing_prob
        self.tread_enabled = tread_route_start is not None and tread_route_end is not None
        self.spatial_window_radius = spatial_window_radius
        self.num_pose_joints = num_pose_joints
        self.pose_confidence_threshold = pose_confidence_threshold
        
        # Patch embedding
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, bottleneck_size=bottleneck_size)
        
        # NOTE: Positional embedding is now generated dynamically in forward()
        # to support variable aspect ratios from bucketed training
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Conditioning projections (keep bias for these)
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
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_checkpoint=use_gradient_checkpointing)
            for _ in range(depth)
        ])
        
        # REPA projection (projects hidden states to DINOv3 patch feature space)
        if repa_block_idx is not None:
            self.repa_proj = nn.Linear(hidden_size, dino_patch_dim, bias=False)
        
        # MaskDiT decoder for masked training
        self.maskdit_enabled = maskdit_enabled
        self.maskdit_mask_ratio = maskdit_mask_ratio
        if maskdit_enabled:
            self.maskdit_decoder = MaskDiTDecoder(
                hidden_size, num_heads, depth=maskdit_decoder_depth,
                mlp_ratio=mlp_ratio, use_checkpoint=use_gradient_checkpointing,
            )
        
        # Output layers (keep bias for final projection)
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)
        
        # Local refinement convolution to smooth patch boundaries
        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        
        # Initialize weights
        self.initialize_weights()
        
        # Learnable null embeddings for CFG
        self.null_dino = nn.Parameter(torch.zeros(1, dino_dim))
        self.null_dino_patch_token = nn.Parameter(torch.zeros(1, 1, dino_patch_dim))
        self.null_text = nn.Parameter(torch.zeros(1, 1, text_dim))
        # [NULL_POSE]: learned token the model sees when pose is dropped during CFG
        self.null_pose = nn.Parameter(torch.zeros(1, num_pose_joints, hidden_size))
    
    def get_pos_embed(self, h, w, device):
        """Generate 2D sinusoidal positional embeddings for given spatial size.
        
        Args:
            h: height in patches
            w: width in patches
            device: torch device
        
        Returns:
            pos_embed: (1, h*w, hidden_size)
        """
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, (h, w))
        pos_embed = pos_embed.to(device).float().unsqueeze(0)
        return pos_embed

    def _build_spatial_cross_mask(self, N_latent, h_patches, w_patches, N_dino, N_text, has_pose, pad_ctx, device, dtype):
        """Build a 4D per-query-token cross-attention mask for spatial windowing.
        
        Each latent token attends to:
        - All text tokens (always)
        - CLS token (always)
        - DINO patches within self.spatial_window_radius of its spatial position
        - All pose tokens (always)
        
        Returns:
            mask: (1, 1, N_latent, M_context) where True = attend
        """
        r = self.spatial_window_radius
        
        # Estimate DINO grid size from patch count
        # DINOv3 uses ~14px patches internally, but the exact grid varies.
        # Approximate with sqrt to maintain aspect ratio.
        dino_h = int((N_dino * h_patches / w_patches) ** 0.5)
        dino_w = N_dino // dino_h
        while dino_h * dino_w < N_dino:
            dino_w += 1
        while dino_h * dino_w > N_dino + dino_w:
            dino_h -= 1
        
        # Latent token grid positions
        lat_rows = torch.arange(h_patches, device=device).unsqueeze(1).expand(h_patches, w_patches).flatten()  # (N_latent,)
        lat_cols = torch.arange(w_patches, device=device).unsqueeze(0).expand(h_patches, w_patches).flatten()
        
        # Map each latent token to DINO grid position (linear scaling)
        dino_rows = (lat_rows.float() * dino_h / h_patches).long().clamp(0, dino_h - 1)
        dino_cols = (lat_cols.float() * dino_w / w_patches).long().clamp(0, dino_w - 1)
        
        # Build DINO patch mask: (N_latent, N_dino) — True if DINO patch is within radius
        dino_patch_mask = torch.zeros(N_latent, N_dino, device=device, dtype=torch.bool)
        
        for i in range(N_latent):
            dr, dc = dino_rows[i].item(), dino_cols[i].item()
            r_min = max(0, dr - r)
            r_max = min(dino_h - 1, dr + r)
            c_min = max(0, dc - r)
            c_max = min(dino_w - 1, dc + r)
            
            # Gather DINO patch indices in the spatial window
            for drr in range(r_min, r_max + 1):
                for dcc in range(c_min, c_max + 1):
                    idx = drr * dino_w + dcc
                    if idx < N_dino:
                        dino_patch_mask[i, idx] = True
        
        # Total context: text + CLS + dino_patches + pose(optional)
        N_pose = self.num_pose_joints if has_pose else 0
        M_total = N_text + 1 + N_dino + N_pose
        
        # Full mask: (1, 1, N_latent, M_total)
        full_mask = torch.zeros(1, 1, N_latent, M_total, device=device, dtype=torch.bool)
        
        # Text tokens: all True
        full_mask[:, :, :, :N_text] = True
        # CLS token: True
        full_mask[:, :, :, N_text] = True
        # DINO patches: spatial window
        full_mask[:, :, :, N_text + 1:N_text + 1 + N_dino] = dino_patch_mask.unsqueeze(0).unsqueeze(0)
        # Pose tokens: all True
        if N_pose > 0:
            full_mask[:, :, :, N_text + 1 + N_dino:N_text + 1 + N_dino + N_pose] = True
        
        # Context padding (Rule of 16): always False (don't attend to padding)
        if pad_ctx > 0:
            full_mask = F.pad(full_mask, (0, pad_ctx), value=False)
        
        return full_mask

    def initialize_weights(self):
        # Standard initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize patch embedding like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        if self.x_embedder.expand is not None:
            nn.init.xavier_uniform_(self.x_embedder.expand.weight)
            nn.init.constant_(self.x_embedder.expand.bias, 0)
        
        # Keep final_proj normally initialized (xavier from _basic_init)
        # DO NOT zero-init - that would kill all gradients!
        
        # Zero-init output conv for training stability (residual starts at zero)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        
        # Xavier-init REPA projection if present
        if self.repa_block_idx is not None:
            nn.init.xavier_uniform_(self.repa_proj.weight)

    def unpatchify(self, x, h, w):
        """Convert patch tokens back to spatial latents.
        
        Args:
            x: (B, N, patch_size^2 * C) where N = h * w
            h: height in patches
            w: width in patches
        
        Returns:
            latents: (B, C, H, W) where H = h * patch_size, W = w * patch_size
        """
        B = x.shape[0]
        x = x.reshape(B, h, w, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        latents = x.reshape(B, self.in_channels, h * self.patch_size, w * self.patch_size)
        return latents

    def forward(self, x, t, dino_emb, text_emb, dino_patches=None, text_mask=None, dino_patches_mask=None,
                cfg_drop_text=None, cfg_drop_dino_cls=None, cfg_drop_dino_patches=None,
                pose_kpts=None, cfg_drop_pose=None,
                return_repa_hidden=False, tread_enabled=None, maskdit_enabled=None):
        """
        Args:
            x: (B, C, H, W) noisy latents (H and W can vary for different aspect ratios)
            t: (B,) timesteps
            dino_emb: (B, 1024) DINOv3 CLS embeddings
            text_emb: (B, seq_len, 1024) T5 hidden states (seq_len=500 for full captions)
            dino_patches: (B, num_patches, 1024) DINOv3 spatial patches (VARIABLE LENGTH!)
            text_mask: (B, seq_len) T5 attention mask (1=valid, 0=padding)
            dino_patches_mask: (B, num_patches) DINOv3 patches attention mask
            cfg_drop_text: (B,) bool mask for dropping text
            cfg_drop_dino_cls: (B,) bool mask for dropping DINO CLS
            cfg_drop_dino_patches: (B,) bool mask for dropping DINO patches
            pose_kpts: (B, 133, 3) pose keypoints [x_norm, y_norm, confidence]
            cfg_drop_pose: (B,) bool mask for dropping pose (swaps to [NULL_POSE])
            return_repa_hidden: if True, return (v_pred, repa_hidden, tread_visible_idx, maskdit_info) tuple
            tread_enabled: override for TREAD routing (None = use self.training when configured)
            maskdit_enabled: override for MaskDiT masking (None = use self.maskdit_enabled when training)
        
        Returns:
            v: (B, C, H, W) predicted velocity
            -- or if return_repa_hidden=True --
            (v, repa_hidden, tread_visible_idx, maskdit_info): tuple with extra info
            maskdit_info is None when masking is off, or dict with visible_idx, masked_idx, 
            x0_pred_masked (decoder output for masked positions) when masking is on
        """
        B, C, H, W = x.shape
        
        # Get number of patches (varies per bucket)
        if dino_patches is not None:
            num_patches = dino_patches.shape[1]
        else:
            # Default: use null patches for debugging/fallback
            num_patches = 3880  # Approximate typical count
            dino_patches = self.null_dino_patch_token.expand(B, num_patches, -1)
        
        # Apply CFG dropout independently
        if cfg_drop_dino_cls is not None:
            dino_emb = torch.where(
                cfg_drop_dino_cls.unsqueeze(1),
                self.null_dino.expand(B, -1),
                dino_emb
            )
            
        if cfg_drop_dino_patches is not None:
            null_patches = self.null_dino_patch_token.expand(B, num_patches, -1)
            dino_patches = torch.where(
                cfg_drop_dino_patches.unsqueeze(1).unsqueeze(2),
                null_patches,
                dino_patches
            )
        
        if cfg_drop_text is not None:
            text_emb = torch.where(
                cfg_drop_text.unsqueeze(1).unsqueeze(2),
                self.null_text.expand(B, text_emb.shape[1], -1),
                text_emb
            )
        
        # Embed inputs with dynamic positional encoding
        x = self.x_embedder(x)  # (B, N, hidden_size) where N = (H//patch_size) * (W//patch_size)
        
        # Generate positional embeddings based on actual spatial dimensions
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        pos_embed = self.get_pos_embed(h_patches, w_patches, x.device)
        x = x + pos_embed

        # Dynamic Tensor Masking for latent sequence (Rule of 32)
        original_S = x.shape[1]
        
        # In compiled mode, modulo arithmetic on sequence shapes creates SymInts. 
        # Using simple // instead of math functions to keep it clean for Dynamo
        pad_latent = (32 - (original_S % 32)) % 32
        
        x_mask = None
        # Unconditional padding to avoid branching on SymInts
        x = F.pad(x, (0, 0, 0, pad_latent))
        x_mask = torch.cat([
            torch.ones(B, original_S, device=x.device, dtype=torch.bool),
            torch.zeros(B, pad_latent, device=x.device, dtype=torch.bool)
        ], dim=1)
        
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        
        # Project conditioning (after CFG dropout)
        dino_cond = self.dino_proj(dino_emb) + t_emb  # (B, hidden_size) - for adaLN
        dino_cls_token = dino_cond.unsqueeze(1)  # (B, 1, hidden_size) - for cross-attention
        text_cond = self.text_proj(text_emb)  # (B, seq_len, hidden_size)
        patches_cond = self.dino_patch_proj(dino_patches)  # (B, num_patches, hidden_size)
        
        # Pose conditioning: project keypoints and add joint-type embeddings
        if pose_kpts is not None:
            if cfg_drop_pose is not None:
                null_pose_expanded = self.null_pose.expand(B, -1, -1)
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
                null_pose_expanded = self.null_pose.expand(B, -1, -1)
                pose_tokens = torch.where(low_conf.unsqueeze(2), null_pose_expanded, pose_tokens)
            
            patches_cond = torch.cat([patches_cond, pose_tokens], dim=1)
            if dino_patches_mask is not None:
                pose_mask = torch.ones(B, self.num_pose_joints, device=dino_patches_mask.device, dtype=dino_patches_mask.dtype)
                dino_patches_mask = torch.cat([dino_patches_mask, pose_mask], dim=1)
        
        # Dynamic Tensor Masking for context sequence (Rule of 16)
        S_ctx = text_cond.shape[1] + 1 + patches_cond.shape[1]
        
        # When compiled dynamically, this forces pad_ctx to be an int (no unbacked SymInt issue)
        # since it's just Python modulo arithmetic on known symbolic shapes.
        # But we must avoid conditional branching on SymInts. F.pad handles SymInts fine.
        
        pad_ctx = (16 - (S_ctx % 16)) % 16
        
        # We must avoid 'if pad_ctx > 0:' when pad_ctx is a SymInt.
        # Instead, we just unconditionally pad. F.pad with 0 padding is a no-op anyway.
        patches_cond = F.pad(patches_cond, (0, 0, 0, pad_ctx))
        if text_mask is None:
            text_mask = torch.ones(B, text_cond.shape[1], device=text_cond.device, dtype=torch.long)
        if dino_patches_mask is None:
            original_patches_len = patches_cond.shape[1] - pad_ctx
            dino_patches_mask = torch.ones(B, original_patches_len, device=patches_cond.device, dtype=text_mask.dtype)
        dino_patches_mask = F.pad(dino_patches_mask, (0, pad_ctx), value=0)
        
        # Build spatial window cross-attention mask if configured
        spatial_cross_mask = None
        if self.spatial_window_radius is not None:
            N_latent_orig = original_S  # pre-padding latent token count
            N_dino_orig = dino_patches.shape[1]  # pre-append DINO patch count
            N_text = text_cond.shape[1]
            has_pose = pose_kpts is not None
            
            # Build mask BEFORE context padding — the mask covers the raw context.
            # DiTBlock will cat [text, CLS, patches] and the mask dimensions must match.
            # Note: patches_cond already includes pose + context padding.
            # We build the mask for the FULL context including pose and padding.
            spatial_cross_mask = self._build_spatial_cross_mask(
                N_latent_orig, h_patches, w_patches, N_dino_orig, N_text, has_pose, pad_ctx,
                x.device, torch.bool
            )
        
        # MaskDiT: randomly mask image tokens during training
        use_maskdit = self.maskdit_enabled and (maskdit_enabled if maskdit_enabled is not None else self.training)
        maskdit_info = None
        N_total = x.shape[1]
        
        if use_maskdit:
            N_mask = int(N_total * self.maskdit_mask_ratio)
            N_vis = N_total - N_mask
            
            perm = torch.randperm(N_total, device=x.device)
            visible_idx = perm[:N_vis].sort().values
            masked_idx = perm[N_vis:].sort().values
            
            # Keep only visible tokens for encoder
            x_full_before_mask = x  # save for decoder pos_embed
            x = x[:, visible_idx]  # (B, N_vis, C)
        
            # When compiled dynamically, returning dicts with tensors can cause graph breaks
            # or unbacked symbol issues. Use a structured tuple or handle it carefully.
            maskdit_info = {
                'visible_idx': visible_idx,
                'masked_idx': masked_idx,
                'N_total': int(N_total),
                'pos_embed': pos_embed,
            }
        
        # Determine if TREAD routing is active
        use_tread = self.tread_enabled and (tread_enabled if tread_enabled is not None else self.training)
        
        # Transformer blocks with optional TREAD routing
        repa_hidden = None
        tread_visible_idx = None
        
        if use_tread:
            N = x.shape[1]
            # When compiled dynamically, int() casting of a float product can create unbacked symbols
            # Use integer arithmetic if routing probability is a clean fraction like 0.5
            if self.tread_routing_prob == 0.5:
                N_visible = N // 2
            else:
                N_visible = N - int(N * self.tread_routing_prob)
            
            perm = torch.randperm(N, device=x.device)
            visible_idx = perm[:N_visible].sort().values
            routed_idx = perm[N_visible:].sort().values
            tread_visible_idx = visible_idx
            
            # Sub-sample the x_mask for middle blocks
            visible_x_mask = x_mask[:, visible_idx] if x_mask is not None else None
            
            for i in range(self.tread_route_start):
                x = self.blocks[i](x, dino_cond, text_cond, text_mask, dino_cls_token, patches_cond, patches_mask=dino_patches_mask, x_mask=x_mask, spatial_cross_mask=spatial_cross_mask)
            
            routed_tokens = x[:, routed_idx]
            x = x[:, visible_idx]
            
            for i in range(self.tread_route_start, self.tread_route_end + 1):
                x = self.blocks[i](x, dino_cond, text_cond, text_mask, dino_cls_token, patches_cond, patches_mask=dino_patches_mask, x_mask=visible_x_mask, spatial_cross_mask=spatial_cross_mask)
                if return_repa_hidden and i == self.repa_block_idx:
                    repa_hidden = self.repa_proj(x)
            
            full_x = torch.empty(x.shape[0], N, x.shape[2], device=x.device, dtype=x.dtype)
            full_x[:, visible_idx] = x
            full_x[:, routed_idx] = routed_tokens
            x = full_x
            
            for i in range(self.tread_route_end + 1, len(self.blocks)):
                x = self.blocks[i](x, dino_cond, text_cond, text_mask, dino_cls_token, patches_cond, patches_mask=dino_patches_mask, x_mask=x_mask, spatial_cross_mask=spatial_cross_mask)
                if return_repa_hidden and i == self.repa_block_idx:
                    repa_hidden = self.repa_proj(x)
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, dino_cond, text_cond, text_mask, dino_cls_token, patches_cond, patches_mask=dino_patches_mask, x_mask=x_mask, spatial_cross_mask=spatial_cross_mask)
                if return_repa_hidden and i == self.repa_block_idx:
                    repa_hidden = self.repa_proj(x)
        
        # MaskDiT decoder: reconstruct full token sequence from visible encoder outputs
        if use_maskdit and maskdit_info is not None:
            x = self.maskdit_decoder(
                x, maskdit_info['visible_idx'], maskdit_info['masked_idx'],
                maskdit_info['N_total'], maskdit_info['pos_embed'],
                dino_cond, text_cond, text_mask, dino_cls_token, patches_cond, dino_patches_mask,
            )
            
        # Slice off padding tokens if added
        # Using unconditional slicing based on original_S instead of `if pad_latent > 0`
        x = x[:, :original_S, :]
        
        # Output projection
        x = self.final_norm(x)
        x = self.final_proj(x)
        x = self.unpatchify(x, h_patches, w_patches)  # (B, C, H, W)
        
        # Smooth patch boundaries with residual connection
        x = x + self.output_conv(x)
        
        if return_repa_hidden:
            return x, repa_hidden, tread_visible_idx, maskdit_info
        return x


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model instantiation
    model = NanoDiT(
        input_size=64,
        patch_size=2,
        in_channels=16,
        hidden_size=384,
        depth=12,
        num_heads=6,
    )
    
    print(f"Model parameters: {count_parameters(model) / 1e6:.1f}M")
    
    # Test forward pass
    B = 2
    x = torch.randn(B, 16, 64, 64)
    t = torch.rand(B)
    dino = torch.randn(B, 1024)
    text = torch.randn(B, 512, 1024)  # Updated to 512 tokens
    mask = torch.ones(B, 512)
    
    with torch.no_grad():
        v = model(x, t, dino, text, text_mask=mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {v.shape}")
    
    # Test with pose conditioning
    pose = torch.randn(B, 133, 3)
    pose[:, :, 2] = torch.rand(B, 133)  # Confidence in [0, 1]
    with torch.no_grad():
        v_pose = model(x, t, dino, text, pose_kpts=pose, text_mask=mask)
    print(f"Output shape (with pose): {v_pose.shape}")
    print("✓ Model test passed")
