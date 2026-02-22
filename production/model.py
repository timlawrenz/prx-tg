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
    """Embed VAE latents into patches."""
    
    def __init__(self, patch_size=2, in_channels=16, hidden_size=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 16, H, W)
        x = self.proj(x)  # (B, hidden_size, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, hidden_size)
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
            # mask: (B, M) -> (B, 1, 1, M) boolean mask for SDPA
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

    def _forward_impl(self, x, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask=None):
        """Internal forward implementation for checkpointing."""
        # Get adaLN modulation parameters from DINOv3
        shift_msa, scale_msa, shift_ca, scale_ca, shift_mlp, scale_mlp = \
            self.adaLN_modulation(c_dino).chunk(6, dim=1)
        
        # Self-attention with adaLN
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # Concatenate cross-attention sequence: [T5 text, DINO CLS, DINO patches]
        # c_text: (B, 500, hidden_size)
        # c_dino_cls_token: (B, 1, hidden_size)
        # c_patches: (B, num_patches, hidden_size) - VARIABLE LENGTH!
        combined_context = torch.cat([c_text, c_dino_cls_token, c_patches], dim=1)
        
        # Concatenate masks: T5 mask + CLS mask (1) + patches mask
        B = x.shape[0]
        
        if text_mask is not None:
            # CLS is always valid
            cls_mask = torch.ones(B, 1, device=text_mask.device, dtype=text_mask.dtype)
            
            # Patches mask (if provided), otherwise assume all patches valid
            if patches_mask is None:
                patches_mask = torch.ones(B, c_patches.shape[1], device=text_mask.device, dtype=text_mask.dtype)
            else:
                patches_mask = patches_mask.to(device=text_mask.device, dtype=text_mask.dtype)
                
            combined_mask = torch.cat([text_mask, cls_mask, patches_mask], dim=1)  # (B, seq + 1 + num_patches)
        else:
            combined_mask = None
        
        # Cross-attention to combined sequence with adaLN
        x = x + self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca),
            context=combined_context,
            mask=combined_mask
        )
        
        # MLP with adaLN
        x = x + self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        
        return x

    def forward(self, x, c_dino, c_text, text_mask=None, c_dino_cls_token=None, c_patches=None, patches_mask=None):
        """
        Args:
            x: (B, N, C) latent tokens
            c_dino: (B, C) DINOv3 conditioning for adaLN
            c_text: (B, M, C) T5 conditioning
            text_mask: (B, M) attention mask for T5
            c_dino_cls_token: (B, 1, C) DINO CLS token for cross-attention
            c_patches: (B, num_patches, C) DINO patch tokens for cross-attention (variable length)
            patches_mask: (B, num_patches) attention mask for DINO patches
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask, use_reentrant=False
            )
        else:
            return self._forward_impl(x, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask)


class NanoDiT(nn.Module):
    """Nano DiT: 12L, 384H, 6A for validation testing."""
    
    def __init__(
        self,
        input_size=64,  # Latent spatial size (64x64 for 512x512 images) - IGNORED for dynamic pos embed
        patch_size=2,
        in_channels=16,  # Flux VAE latent channels
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dino_dim=1024,
        dino_patch_dim=1024,
        text_dim=1024,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        self.input_size = input_size  # For backward compatibility, but not used
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Patch embedding
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        
        # NOTE: Positional embedding is now generated dynamically in forward()
        # to support variable aspect ratios from bucketed training
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Conditioning projections (keep bias for these)
        self.dino_proj = nn.Linear(dino_dim, hidden_size, bias=True)
        self.dino_patch_proj = nn.Linear(dino_patch_dim, hidden_size, bias=True)
        self.text_proj = nn.Linear(text_dim, hidden_size, bias=True)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_checkpoint=use_gradient_checkpointing)
            for _ in range(depth)
        ])
        
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
        
        # Keep final_proj normally initialized (xavier from _basic_init)
        # DO NOT zero-init - that would kill all gradients!
        
        # Zero-init output conv for training stability (residual starts at zero)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

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
                cfg_drop_text=None, cfg_drop_dino_cls=None, cfg_drop_dino_patches=None):
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
        
        Returns:
            v: (B, C, H, W) predicted velocity
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
        
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        
        # Project conditioning (after CFG dropout)
        # Timestep is added to DINO conditioning (which goes through adaLN-Zero)
        # This ensures timestep information reaches all blocks, even when DINO is dropped
        # (null_dino still gets projected and adds t_emb)
        dino_cond = self.dino_proj(dino_emb) + t_emb  # (B, hidden_size) - for adaLN
        dino_cls_token = dino_cond.unsqueeze(1)  # (B, 1, hidden_size) - for cross-attention
        text_cond = self.text_proj(text_emb)  # (B, seq_len, hidden_size)
        patches_cond = self.dino_patch_proj(dino_patches)  # (B, num_patches, hidden_size)
        
        # Transformer blocks (with concatenated cross-attention)
        for block in self.blocks:
            x = block(x, dino_cond, text_cond, text_mask, dino_cls_token, patches_cond, patches_mask=dino_patches_mask)
        
        # Output projection
        x = self.final_norm(x)
        x = self.final_proj(x)
        x = self.unpatchify(x, h_patches, w_patches)  # (B, C, H, W)
        
        # Smooth patch boundaries with residual connection
        x = x + self.output_conv(x)
        
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
        v = model(x, t, dino, text, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {v.shape}")
    print("✓ Model test passed")
