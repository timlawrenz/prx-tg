"""Nano DiT model for validation testing.

Architecture: 12 layers, 384 hidden dim, 6 attention heads (~30-50M params)
Dual conditioning: DINOv3 (1024) → adaLN-Zero, T5 (77×1024) → Cross-Attention
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
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
    
    def __init__(self, dim, num_heads=6, qkv_bias=True, is_cross_attn=False):
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
        
        self.proj = nn.Linear(dim, dim)

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
        
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, M) -> (B, num_heads, N, M)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided (for cross-attention with padding)
        if mask is not None:
            # mask: (B, M) -> (B, 1, 1, M) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    """MLP block: hidden_dim -> mlp_dim -> hidden_dim"""
    
    def __init__(self, hidden_dim, mlp_dim=None):
        super().__init__()
        mlp_dim = mlp_dim or hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero (DINO) and cross-attention (T5)."""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, is_cross_attn=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(hidden_size, num_heads=num_heads, is_cross_attn=True)
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

    def forward(self, x, c_dino, c_text, text_mask=None):
        """
        Args:
            x: (B, N, C) latent tokens
            c_dino: (B, C) DINOv3 conditioning
            c_text: (B, M, C) T5 conditioning
            text_mask: (B, M) attention mask for T5
        """
        # Get adaLN modulation parameters from DINOv3
        shift_msa, scale_msa, shift_ca, scale_ca, shift_mlp, scale_mlp = \
            self.adaLN_modulation(c_dino).chunk(6, dim=1)
        
        # Self-attention with adaLN
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # Cross-attention with adaLN
        x = x + self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca),
            context=c_text,
            mask=text_mask
        )
        
        # MLP with adaLN
        x = x + self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        
        return x


class NanoDiT(nn.Module):
    """Nano DiT: 12L, 384H, 6A for validation testing."""
    
    def __init__(
        self,
        input_size=64,  # Latent spatial size (64x64 for 512x512 images)
        patch_size=2,
        in_channels=16,  # Flux VAE latent channels
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dino_dim=1024,
        text_dim=1024,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        # Patch embedding
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        num_patches = (input_size // patch_size) ** 2
        
        # Positional embedding (fixed 2D sincos)
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        pos_embed = get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Conditioning projections
        self.dino_proj = nn.Linear(dino_dim, hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)
        
        # Initialize weights
        self.initialize_weights()
        
        # Learnable null embeddings for CFG
        self.null_dino = nn.Parameter(torch.zeros(1, dino_dim))
        self.null_text = nn.Parameter(torch.zeros(1, 1, text_dim))

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
        
        # Zero-out final projection (adaLN-Zero pattern for output)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, x):
        """Convert patch tokens back to spatial latents.
        
        Args:
            x: (B, N, patch_size^2 * C)
        
        Returns:
            latents: (B, C, H, W)
        """
        B = x.shape[0]
        H = W = self.input_size // self.patch_size
        x = x.reshape(B, H, W, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        latents = x.reshape(B, self.in_channels, H * self.patch_size, W * self.patch_size)
        return latents

    def forward(self, x, t, dino_emb, text_emb, text_mask=None, 
                cfg_drop_both=None, cfg_drop_dino=None, cfg_drop_text=None):
        """
        Args:
            x: (B, C, H, W) noisy latents
            t: (B,) timesteps
            dino_emb: (B, 1024) DINOv3 embeddings
            text_emb: (B, 77, 1024) T5 hidden states
            text_mask: (B, 77) T5 attention mask (1=valid, 0=padding)
            cfg_drop_both: (B,) bool mask for unconditional
            cfg_drop_dino: (B,) bool mask for text-only
            cfg_drop_text: (B,) bool mask for dino-only
        
        Returns:
            v: (B, C, H, W) predicted velocity
        """
        B = x.shape[0]
        
        # Apply CFG dropout
        if cfg_drop_both is not None:
            dino_emb = torch.where(
                cfg_drop_both.unsqueeze(1),
                self.null_dino.expand(B, -1),
                dino_emb
            )
            text_emb = torch.where(
                cfg_drop_both.unsqueeze(1).unsqueeze(2),
                self.null_text.expand(B, text_emb.shape[1], -1),
                text_emb
            )
        
        if cfg_drop_dino is not None:
            dino_emb = torch.where(
                cfg_drop_dino.unsqueeze(1),
                self.null_dino.expand(B, -1),
                dino_emb
            )
        
        if cfg_drop_text is not None:
            text_emb = torch.where(
                cfg_drop_text.unsqueeze(1).unsqueeze(2),
                self.null_text.expand(B, text_emb.shape[1], -1),
                text_emb
            )
        
        # Embed inputs
        x = self.x_embedder(x) + self.pos_embed  # (B, N, hidden_size)
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        dino_cond = self.dino_proj(dino_emb) + t_emb  # (B, hidden_size)
        text_cond = self.text_proj(text_emb)  # (B, 77, hidden_size)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, dino_cond, text_cond, text_mask)
        
        # Output projection
        x = self.final_norm(x)
        x = self.final_proj(x)
        x = self.unpatchify(x)  # (B, C, H, W)
        
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
    text = torch.randn(B, 77, 1024)
    mask = torch.ones(B, 77)
    
    with torch.no_grad():
        v = model(x, t, dino, text, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {v.shape}")
    print("✓ Model test passed")
