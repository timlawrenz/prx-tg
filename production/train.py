"""Training loop for Nano DiT validation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from tqdm import tqdm
import json
import random
import numpy as np


class PerceptualLossModule:
    """Computes LPIPS perceptual loss on VAE-decoded latent crops.
    
    Lazily loads a frozen VAE decoder and LPIPS network. Computes loss on
    random spatial crops of latents to keep memory usage constant.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self._vae = None
        self._lpips_fn = None
    
    def _ensure_loaded(self):
        """Lazily load VAE decoder and LPIPS network on first use."""
        if self._vae is not None:
            return
        
        from diffusers import AutoencoderKL
        import lpips
        
        vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="vae",
            torch_dtype=torch.float16,
        ).to(self.device)
        vae.eval()
        vae.enable_slicing()
        vae.enable_tiling()
        # Free encoder (we only need decoder)
        del vae.encoder
        for p in vae.parameters():
            p.requires_grad_(False)
        self._vae = vae
        
        self._lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self._lpips_fn.eval()
        for p in self._lpips_fn.parameters():
            p.requires_grad_(False)
    
    def compute(self, x0, x0_hat, crop_size=32):
        """Compute LPIPS loss between decoded latent crops.
        
        Args:
            x0: (B, C, H, W) ground truth latents (normalized)
            x0_hat: (B, C, H, W) predicted latents (with grad from v_pred)
            crop_size: spatial size of latent crop
        
        Returns:
            lpips_loss: scalar tensor with gradient
        """
        self._ensure_loaded()
        from .data import denormalize_vae_latent
        
        B, C, H, W = x0.shape
        cs = min(crop_size, H, W)
        
        # Random crop (same location for x0 and x0_hat)
        top = random.randint(0, H - cs)
        left = random.randint(0, W - cs)
        x0_crop = x0[:, :, top:top+cs, left:left+cs]
        x0_hat_crop = x0_hat[:, :, top:top+cs, left:left+cs]
        
        # Denormalize for VAE decode
        x0_crop = denormalize_vae_latent(x0_crop)
        x0_hat_crop = denormalize_vae_latent(x0_hat_crop)
        
        # Decode ground truth (no grad needed)
        with torch.no_grad():
            pixels_gt = self._vae.decode(x0_crop.half()).sample.float()
        
        # Decode prediction (grad flows through x0_hat_crop → v_pred)
        pixels_pred = self._vae.decode(x0_hat_crop.half()).sample.float()
        
        # Clamp to [-1, 1] for LPIPS
        pixels_gt = pixels_gt.clamp(-1, 1)
        pixels_pred = pixels_pred.clamp(-1, 1)
        
        # LPIPS expects float32
        loss = self._lpips_fn(pixels_gt, pixels_pred).mean()
        return loss


def logit_normal_sample(size, mean=0.0, std=1.0, device='cpu'):
    """Sample timesteps from logit-normal distribution.
    
    Args:
        size: tuple, shape of output
        mean: float, mean of underlying normal distribution
        std: float, std of underlying normal distribution
        device: torch device
    
    Returns:
        t: timesteps in range [0.001, 0.999]
    """
    # Sample from normal distribution
    z = torch.randn(size, device=device) * std + mean
    # Apply logistic function: sigma(z) = 1 / (1 + exp(-z))
    t = torch.sigmoid(z)
    # Clamp to avoid extreme values
    t = torch.clamp(t, min=0.001, max=0.999)
    return t


def compute_repa_loss(repa_hidden, dino_patches, dino_patches_mask, loss_type="cosine", visible_idx=None):
    """Compute REPA alignment loss between projected hidden states and DINOv3 patches.
    
    Handles size mismatches between latent token grid and DINOv3 patch grid
    by truncating to the shorter sequence (both are raster-order from the
    same aspect ratio, so spatial correspondence is preserved).
    
    When TREAD routing is active, repa_hidden contains only visible tokens.
    visible_idx maps them back to spatial positions for correct alignment.
    
    Args:
        repa_hidden: (B, N_latent, dino_dim) projected hidden states from REPA block
        dino_patches: (B, N_dino, dino_dim) raw DINOv3 patch features (teacher signal)
        dino_patches_mask: (B, N_dino) mask for valid (non-padded) patches, or None
        loss_type: "cosine" or "mse"
        visible_idx: (N_visible,) indices of visible tokens when TREAD is active, or None
    
    Returns:
        loss: scalar REPA alignment loss
    """
    # When TREAD is active, index DINOv3 patches to match visible token positions.
    # visible_idx is in [0, N_latent) but N_dino may be smaller, so filter first.
    if visible_idx is not None:
        N_dino = dino_patches.shape[1]
        valid_mask = visible_idx < N_dino
        valid_idx = visible_idx[valid_mask]
        dino_patches = dino_patches[:, valid_idx]
        if dino_patches_mask is not None:
            dino_patches_mask = dino_patches_mask[:, valid_idx]
        # Keep only the corresponding repa_hidden tokens
        repa_hidden = repa_hidden[:, valid_mask]
    
    N_latent = repa_hidden.shape[1]
    N_dino = dino_patches.shape[1]
    N = min(N_latent, N_dino)
    
    repa_hidden = repa_hidden[:, :N]
    dino_patches = dino_patches[:, :N]
    if dino_patches_mask is not None:
        dino_patches_mask = dino_patches_mask[:, :N]
    
    if loss_type == "cosine":
        # Normalize for cosine similarity
        h_norm = F.normalize(repa_hidden, dim=-1)
        t_norm = F.normalize(dino_patches, dim=-1)
        # Per-token cosine similarity: 1 = perfect alignment
        cos_sim = (h_norm * t_norm).sum(dim=-1)  # (B, N)
        per_token_loss = 1.0 - cos_sim
    else:
        per_token_loss = F.mse_loss(repa_hidden, dino_patches, reduction='none').mean(dim=-1)  # (B, N)
    
    if dino_patches_mask is not None:
        # Zero out loss for padded tokens and normalize by valid count
        per_token_loss = per_token_loss * dino_patches_mask.float()
        num_valid = dino_patches_mask.float().sum()
        if num_valid > 0:
            return per_token_loss.sum() / num_valid
        return per_token_loss.sum() * 0.0  # all padded — return zero with grad
    
    return per_token_loss.mean()


def flow_matching_loss(model, x0, dino_emb, dino_patches, text_emb, text_mask, cfg_probs, dino_patches_mask=None, return_v_pred=False, repa_config=None, tread_config=None, perceptual_module=None, perceptual_config=None, micro_step=0, prediction_type="v_prediction", t_clamp_min=0.05):
    """Compute flow matching loss with mutually exclusive CFG dropout.
    
    Supports two prediction modes:
    - v_prediction: model predicts velocity v = z1 - x0 (original)
    - x_prediction: model predicts clean data x0, converted to v-space for MSE
      with t clamped >= t_clamp_min (Li & He 2025)
    
    Args:
        model: NanoDiT model
        x0: (B, C, H, W) clean data (latents or pixels at t=0)
        dino_emb: (B, 1024) DINOv3 CLS embeddings
        dino_patches: (B, num_patches, 1024) DINOv3 spatial patches
        text_emb: (B, seq_len, 1024) T5 hidden states
        text_mask: (B, seq_len) T5 attention mask
        cfg_probs: dict with CFG dropout probabilities
        dino_patches_mask: (B, num_patches) mask for padding
        return_v_pred: bool, if True return (loss, v_pred, repa_loss, lpips_loss)
        repa_config: optional REPAConfig
        tread_config: optional TREADConfig
        perceptual_module: optional PerceptualLossModule for LPIPS loss
        perceptual_config: optional PerceptualLossConfig
        micro_step: current micro-step (for every-N gating)
        prediction_type: "v_prediction" or "x_prediction"
        t_clamp_min: minimum t for x→v conversion (avoids div-by-zero)
    
    Returns:
        loss: scalar tensor, or (loss, v_pred, repa_loss, lpips_loss) if return_v_pred=True
    """
    B = x0.shape[0]
    device = x0.device
    
    # Sample timesteps from logit-normal distribution
    t = logit_normal_sample((B,), device=device)
    
    # Sample noise z1 ~ N(0, I)
    z1 = torch.randn_like(x0)
    
    # Linear interpolation: z_t = (1-t) * x0 + t * z1
    # At t=0: zt = x0 (clean data)
    # At t=1: zt = z1 (pure noise)
    t_expanded = t.view(B, 1, 1, 1)
    zt = (1 - t_expanded) * x0 + t_expanded * z1
    
    # Rectified flow target: velocity field v_t = d(z_t)/dt = z1 - x0
    # This points from data (x0) towards noise (z1)
    # Integrating forward in time: z_t -> z_{t+dt} moves toward noise
    # Integrating backward in time (sampling): z_t -> z_{t-dt} moves toward data
    v_target = z1 - x0
    
    # Mutually exclusive CFG dropout (categorical sampling)
    # Sample one random number per batch item and threshold it
    rand = torch.rand(B, device=device)
    p_both = cfg_probs['p_uncond']
    p_text = cfg_probs['p_text_only']
    p_dino_cls = cfg_probs['p_dino_cls_only']
    p_dino_patches = cfg_probs['p_dino_patches_only']
    
    # Assign to exclusive categories
    drop_both = rand < p_both
    drop_dino = (rand >= p_both) & (rand < p_both + p_text)
    drop_text_and_patches = (rand >= p_both + p_text) & (rand < p_both + p_text + p_dino_cls)
    drop_text_and_cls = (rand >= p_both + p_text + p_dino_cls) & (rand < p_both + p_text + p_dino_cls + p_dino_patches)
    # Remainder (50% by default) has all conditionings present
    
    # Combine masks for specific components
    # We want to drop text when: drop_both OR drop_text_and_patches OR drop_text_and_cls
    drop_text = drop_both | drop_text_and_patches | drop_text_and_cls
    
    # We want to drop DINO CLS when: drop_both OR drop_dino OR drop_text_and_cls
    drop_dino_cls = drop_both | drop_dino | drop_text_and_cls
    
    # We want to drop DINO patches when: drop_both OR drop_dino OR drop_text_and_patches
    drop_dino_patches_mask = drop_both | drop_dino | drop_text_and_patches
    
    # Determine if we need REPA hidden states
    use_repa = repa_config is not None and repa_config.enabled
    
    # TREAD: enable routing during training
    tread_enabled = tread_config is not None and tread_config.enabled
    
    # Predict velocity (with DINO patches)
    model_output = model(
        zt, t, dino_emb, text_emb, dino_patches, text_mask, dino_patches_mask=dino_patches_mask,
        cfg_drop_text=drop_text,
        cfg_drop_dino_cls=drop_dino_cls,
        cfg_drop_dino_patches=drop_dino_patches_mask,
        return_repa_hidden=use_repa,
        tread_enabled=tread_enabled,
    )
    
    if use_repa:
        v_pred, repa_hidden, tread_visible_idx = model_output
    else:
        v_pred = model_output
    
    # Compute MSE loss based on prediction type
    if prediction_type == "x_prediction":
        # Model predicts x0; convert both prediction and target to v-space
        # v = (zt - x0) / max(t, t_clamp_min)  — clamp avoids div-by-zero
        t_clamped = t.clamp(min=t_clamp_min).view(B, 1, 1, 1)
        # v_pred is actually x0_pred from model; convert to v-space
        x0_pred = v_pred  # model output is x0 in x-prediction mode
        v_pred_converted = (zt - x0_pred) / t_clamped
        v_target_converted = (zt - x0) / t_clamped
        loss = F.mse_loss(v_pred_converted, v_target_converted)
    else:
        # Standard v-prediction: MSE on velocity directly
        loss = F.mse_loss(v_pred, v_target)
    
    # REPA alignment loss
    repa_loss = None
    if use_repa and repa_hidden is not None:
        repa_loss = compute_repa_loss(
            repa_hidden, dino_patches, dino_patches_mask, repa_config.loss_type,
            visible_idx=tread_visible_idx if tread_enabled else None,
        )
        loss = loss + repa_config.weight * repa_loss
    
    # Perceptual (LPIPS) loss — computed every N micro-steps
    lpips_loss = None
    use_perceptual = (
        perceptual_module is not None
        and perceptual_config is not None
        and perceptual_config.enabled
        and micro_step % perceptual_config.every_n_microsteps == 0
    )
    if use_perceptual:
        if prediction_type == "x_prediction":
            # In x-prediction, model output IS x0_pred — use directly for perceptual loss
            # (pixel-space: no VAE decode needed, but PerceptualLossModule handles that)
            x0_hat = v_pred  # x0_pred from model
        else:
            # Reconstruct predicted clean latent: x0_hat = zt - t * v_pred
            x0_hat = zt - t_expanded * v_pred
        lpips_loss = perceptual_module.compute(x0, x0_hat, perceptual_config.crop_size)
        loss = loss + perceptual_config.lpips_weight * lpips_loss
    
    if return_v_pred:
        return loss, v_pred, repa_loss, lpips_loss
    return loss


class EMAModel:
    """Exponential Moving Average of model weights."""
    
    def __init__(self, model, decay=0.9999, warmup_steps=5000):
        """
        Args:
            model: PyTorch model
            decay: EMA decay rate (target after warmup)
            warmup_steps: number of steps to warm up decay from 0 to target
        """
        self.model = model
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        
        # Initialize EMA parameters
        self.ema_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.data.clone()
    
    def get_decay(self):
        """Get current EMA decay with linear warmup."""
        if self.step < self.warmup_steps:
            return self.step / self.warmup_steps * self.target_decay
        return self.target_decay
    
    def update(self):
        """Update EMA parameters."""
        decay = self.get_decay()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    self.ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        
        self.step += 1
    
    def copy_to(self, model):
        """Copy EMA parameters to model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.ema_params:
                    param.data.copy_(self.ema_params[name])
    
    def state_dict(self):
        """Get state dict for saving."""
        return {
            'ema_params': self.ema_params,
            'step': self.step,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.ema_params = state_dict['ema_params']
        self.step = state_dict['step']


def get_lr_schedule(step, warmup_steps, total_steps, peak_lr, min_lr):
    """Get learning rate with linear warmup and cosine decay.
    
    Args:
        step: current training step
        warmup_steps: number of warmup steps
        total_steps: total training steps
        peak_lr: peak learning rate
        min_lr: minimum learning rate
    
    Returns:
        lr: learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return peak_lr * step / warmup_steps
    else:
        # Cosine decay
        # Avoid division by zero if warmup_steps == total_steps
        if total_steps <= warmup_steps:
            return peak_lr
        
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(progress, 1.0)  # Clamp to [0, 1]
        return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


class Trainer:
    """Training orchestration for Nano DiT."""
    
    def __init__(
        self,
        model,
        dataloader,
        device='cuda',
        total_steps=5000,
        warmup_steps=5000,
        peak_lr=3e-4,
        min_lr=1e-6,
        weight_decay=0.03,
        grad_clip=1.0,
        ema_decay=0.9999,
        cfg_probs=None,
        grad_accumulation_steps=1,
        checkpoint_every=1000,
        log_every=50,
        checkpoint_dir='checkpoints',
        optimizer_config=None,
    ):
        """
        Args:
            model: NanoDiT model
            dataloader: iterable dataloader
            device: torch device
            total_steps: total training steps
            warmup_steps: LR warmup steps
            peak_lr: peak learning rate
            min_lr: minimum learning rate
            weight_decay: AdamW weight decay
            grad_clip: gradient clipping norm
            ema_decay: target EMA decay
            cfg_probs: dict with CFG dropout probabilities
            grad_accumulation_steps: number of gradient accumulation steps
            checkpoint_every: checkpoint save frequency
            log_every: logging frequency
            checkpoint_dir: directory for saving checkpoints
            optimizer_config: OptimizerConfig object (if None, uses AdamW defaults)
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.grad_clip = grad_clip
        self.grad_accumulation_steps = grad_accumulation_steps
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # CFG dropout probabilities
        self.cfg_probs = cfg_probs or {
            'p_uncond': 0.1,
            'p_text_only': 0.1,
            'p_dino_cls_only': 0.1,
            'p_dino_patches_only': 0.1,
        }
        
        # Optimizer setup
        self.optimizer_type = getattr(optimizer_config, 'type', 'AdamW') if optimizer_config else 'AdamW'
        self.optimizer_muon = None
        self.optimizer_adam = None
        
        if self.optimizer_type == 'Muon':
            self._create_muon_optimizer(model, optimizer_config, peak_lr, weight_decay)
        else:
            self.optimizer_adam = torch.optim.AdamW(
                model.parameters(),
                lr=peak_lr,
                betas=tuple(optimizer_config.betas) if optimizer_config else (0.9, 0.95),
                weight_decay=weight_decay,
                eps=optimizer_config.eps if optimizer_config else 1e-8,
            )
        
        # EMA
        self.ema = EMAModel(model, decay=ema_decay, warmup_steps=warmup_steps)
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        # REPA config (set by ProductionTrainer if enabled)
        self.repa_config = None
        
        # TREAD config (set by ProductionTrainer if enabled)
        self.tread_config = None
        
        # Logging
        self.log_file = self.checkpoint_dir.parent / 'training_log.jsonl'
    
    def _create_muon_optimizer(self, model, optimizer_config, peak_lr, weight_decay):
        """Create hybrid Muon (2D params) + AdamW (non-2D params) optimizer."""
        muon_params = []
        adam_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        muon_cfg = optimizer_config.muon
        self.optimizer_muon = torch.optim.Muon(
            muon_params,
            lr=peak_lr,
            weight_decay=weight_decay,
            momentum=muon_cfg.momentum,
            nesterov=muon_cfg.nesterov,
            ns_steps=muon_cfg.ns_steps,
            adjust_lr_fn=muon_cfg.adjust_lr_fn,
        )
        
        if adam_params:
            self.optimizer_adam = torch.optim.AdamW(
                adam_params,
                lr=peak_lr,
                betas=tuple(optimizer_config.betas),
                weight_decay=weight_decay,
                eps=optimizer_config.eps,
            )
        
        self._muon_param_count = sum(p.numel() for p in muon_params)
        self._adam_param_count = sum(p.numel() for p in adam_params)
    
    def _step_optimizers(self):
        """Step all active optimizers."""
        if self.optimizer_muon is not None:
            self.optimizer_muon.step()
        if self.optimizer_adam is not None:
            self.optimizer_adam.step()
    
    def _zero_grad_optimizers(self):
        """Zero gradients on all active optimizers."""
        if self.optimizer_muon is not None:
            self.optimizer_muon.zero_grad()
        if self.optimizer_adam is not None:
            self.optimizer_adam.zero_grad()
    
    def _set_lr_optimizers(self, lr):
        """Set learning rate on all active optimizers."""
        if self.optimizer_muon is not None:
            for pg in self.optimizer_muon.param_groups:
                pg['lr'] = lr
        if self.optimizer_adam is not None:
            for pg in self.optimizer_adam.param_groups:
                pg['lr'] = lr
    
    def _update_resolution_schedule(self):
        """Check and apply resolution scale based on current step.
        
        Uses self.resolution_phases (set by ProductionTrainer) to determine
        the correct scale for the current step. Updates the dataloader's
        resolution_scale property when the phase changes.
        """
        phases = getattr(self, 'resolution_phases', None)
        if not phases:
            return
        
        # Find the current phase
        scale = 1.0
        for phase in phases:
            if self.step < phase.until_step:
                scale = phase.scale
                break
        
        current = getattr(self, '_current_resolution_scale', None)
        if current != scale:
            if hasattr(self.dataloader, 'resolution_scale'):
                self.dataloader.resolution_scale = scale
                self._current_resolution_scale = scale
                if current is not None:
                    print(f"\n  Resolution schedule: scale {current} → {scale} at step {self.step}")
    
    def train_step(self, batch):
        """Execute one training step.
        
        Args:
            batch: dict with vae_latent, dino_embedding, dinov3_patches, t5_hidden, t5_mask
        
        Returns:
            dict with loss and grad_norm
        """
        self.model.train()
        
        # Move batch to device
        x0 = batch['vae_latent'].to(self.device)
        dino_emb = batch['dino_embedding'].to(self.device)
        dino_patches = batch['dinov3_patches'].to(self.device)  # (B, num_patches, 1024)
        text_emb = batch['t5_hidden'].to(self.device)
        text_mask = batch['t5_mask'].to(self.device)
        dino_patches_mask = batch.get('dinov3_patches_mask')
        if dino_patches_mask is not None:
            dino_patches_mask = dino_patches_mask.to(self.device)
        
        # Compute loss (with velocity prediction for monitoring)
        loss, v_pred, repa_loss, lpips_loss = flow_matching_loss(
            self.model, x0, dino_emb, dino_patches, text_emb, text_mask, self.cfg_probs, 
            dino_patches_mask=dino_patches_mask, return_v_pred=True,
            repa_config=self.repa_config,
            tread_config=self.tread_config,
            perceptual_module=getattr(self, 'perceptual_module', None),
            perceptual_config=getattr(self, 'perceptual_config', None),
            micro_step=getattr(self, 'micro_step', 0),
            prediction_type=getattr(self, 'prediction_type', 'v_prediction'),
            t_clamp_min=getattr(self, 't_clamp_min', 0.05),
        )
        
        # Scale loss by accumulation steps
        loss = loss / self.grad_accumulation_steps
        
        # Backward (accumulate gradients)
        loss.backward()
        
        # Increment micro step counter
        if not hasattr(self, 'micro_step'):
            self.micro_step = 0
        self.micro_step += 1
        
        # Only step optimizer every accumulation_steps
        is_accumulation_step = self.micro_step % self.grad_accumulation_steps == 0
        
        # Always compute LR for reporting (even if not stepping)
        lr = get_lr_schedule(
            self.step, self.warmup_steps, self.total_steps,
            self.peak_lr, self.min_lr
        )
        
        # Compute grad norm for reporting
        grad_norm = sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
        
        # Collect per-layer gradient norms for monitoring patch learning
        layer_grad_norms = {}
        if hasattr(self.model, 'dino_patch_proj') and self.model.dino_patch_proj.weight.grad is not None:
            layer_grad_norms['patch_proj'] = self.model.dino_patch_proj.weight.grad.norm().item()
        if hasattr(self.model, 'text_proj') and self.model.text_proj.weight.grad is not None:
            layer_grad_norms['text_proj'] = self.model.text_proj.weight.grad.norm().item()
        if hasattr(self.model, 'dino_proj') and self.model.dino_proj.weight.grad is not None:
            layer_grad_norms['dino_proj'] = self.model.dino_proj.weight.grad.norm().item()
        
        if is_accumulation_step:
            # Gradient clipping
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            
            # Apply LR and step all optimizers
            self._set_lr_optimizers(lr)
            self._step_optimizers()
            self._zero_grad_optimizers()
            
            # EMA update (only when optimizer steps)
            self.ema.update()
        
        # Collect VRAM metrics
        metrics = {
            'loss': loss.item() * self.grad_accumulation_steps,  # Report unscaled loss
            'grad_norm': grad_norm,
            'lr': lr,
        }
        
        # Add per-layer gradient norms
        for layer_name, grad_norm_val in layer_grad_norms.items():
            metrics[f'grad_norm/{layer_name}'] = grad_norm_val
        
        # REPA loss logging
        if repa_loss is not None:
            metrics['repa_loss'] = repa_loss.item()
        
        # LPIPS perceptual loss logging
        if lpips_loss is not None:
            metrics['lpips_loss'] = lpips_loss.item()
        
        # Every 100 steps, add weight statistics for monitoring parameter evolution
        # Note: self.step gets incremented AFTER train_step returns, so check (self.step + 1)
        if is_accumulation_step and (self.step + 1) % 100 == 0:
            # Handle DataParallel wrapper
            model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Debug: always log that we're trying
            metrics['debug/weights_check'] = 1.0
            
            if hasattr(model, 'dino_patch_proj'):
                w = model.dino_patch_proj.weight.data
                metrics['weights/patch_proj_std'] = w.std().item()
                metrics['weights/patch_proj_mean'] = w.mean().item()
            else:
                metrics['debug/no_patch_proj'] = 1.0
                
            if hasattr(model, 'text_proj'):
                w = model.text_proj.weight.data
                metrics['weights/text_proj_std'] = w.std().item()
            else:
                metrics['debug/no_text_proj'] = 1.0
                
            if hasattr(model, 'null_dino_patch_token'):
                w = model.null_dino_patch_token.data
                metrics['weights/null_patch_token_norm'] = w.norm().item()
            else:
                metrics['debug/no_null_token'] = 1.0
        
        # Add velocity norm monitoring (collapse detection)
        if self.monitor_velocity:
            # Compute RMS (root mean square) magnitude: should stabilize near 1.0
            # Collapse indicators: v_norm → 0 or v_norm >> 100
            v_norm = v_pred.detach().pow(2).mean().sqrt().item()
            metrics['velocity_norm'] = v_norm
            
            # Warning for potential collapse
            if v_norm > self.velocity_warning or v_norm < 0.01:
                print(f"⚠️  WARNING: velocity_norm = {v_norm:.4f} (expected ~1.0, collapse if <0.01 or >{self.velocity_warning})")
        
        # Add GPU memory metrics if using CUDA
        if self.device.type == 'cuda':
            metrics['vram_allocated_gb'] = torch.cuda.memory_allocated(self.device) / 1024**3
            metrics['vram_reserved_gb'] = torch.cuda.memory_reserved(self.device) / 1024**3
        
        return metrics, is_accumulation_step
    
    def save_checkpoint(self, path=None):
        """Save training checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f'checkpoint_step{self.step:07d}.pt'
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer_type': self.optimizer_type,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
            }
        }
        
        # Save optimizer state(s)
        if self.optimizer_type == 'Muon':
            checkpoint['optimizer_muon'] = self.optimizer_muon.state_dict()
            if self.optimizer_adam is not None:
                checkpoint['optimizer_adam'] = self.optimizer_adam.state_dict()
        else:
            checkpoint['optimizer'] = self.optimizer_adam.state_dict()
        
        # Save CUDA RNG state if available
        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state()
        
        torch.save(checkpoint, path)
        
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        self.ema.load_state_dict(checkpoint['ema'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        # Load optimizer state(s) with format detection
        saved_type = checkpoint.get('optimizer_type', 'AdamW')
        if saved_type == self.optimizer_type:
            # Same optimizer type — load directly
            if self.optimizer_type == 'Muon':
                self.optimizer_muon.load_state_dict(checkpoint['optimizer_muon'])
                if self.optimizer_adam is not None and 'optimizer_adam' in checkpoint:
                    self.optimizer_adam.load_state_dict(checkpoint['optimizer_adam'])
            else:
                self.optimizer_adam.load_state_dict(checkpoint.get('optimizer', checkpoint.get('optimizer_adam')))
        else:
            # Optimizer type changed — skip state, will start fresh
            print(f"  WARNING: Checkpoint has {saved_type} optimizer but current config uses {self.optimizer_type}")
            print(f"  Optimizer state will start fresh (model weights and EMA loaded OK)")
        
        # Restore RNG states if available (older checkpoints may not have them)
        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            
            # Convert to CPU ByteTensor if needed (may be CUDA or numpy)
            torch_state = rng_state['torch']
            if isinstance(torch_state, torch.Tensor):
                torch_state = torch_state.cpu().to(dtype=torch.uint8)
            else:
                torch_state = torch.ByteTensor(torch_state)
            torch.set_rng_state(torch_state)
            
            if 'cuda' in rng_state and torch.cuda.is_available():
                cuda_state = rng_state['cuda']
                if isinstance(cuda_state, torch.Tensor):
                    cuda_state = cuda_state.cpu().to(dtype=torch.uint8)
                else:
                    cuda_state = torch.ByteTensor(cuda_state)
                torch.cuda.set_rng_state(cuda_state)
        
        print(f"Loaded checkpoint from {path} (step {self.step})")
    
    def log(self, metrics):
        """Log metrics to file and console."""
        metrics['step'] = self.step
        metrics['epoch'] = self.epoch
        
        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def train(self, validate_fn=None, visual_debug_fn=None):
        """Run full training loop.
        
        Args:
            validate_fn: optional function(model, ema, step, device) for validation
            visual_debug_fn: optional function(model, step) for visual debugging
        """
        print(f"Starting training for {self.total_steps} steps")
        print(f"Warmup: {self.warmup_steps} steps")
        print(f"Peak LR: {self.peak_lr}, Min LR: {self.min_lr}")
        print(f"CFG probs: {self.cfg_probs}")
        
        pbar = tqdm(total=self.total_steps, initial=self.step, desc='Training')
        
        data_iter = iter(self.dataloader)
        
        accum_loss = 0.0
        
        while self.step < self.total_steps:
            # Update resolution schedule if configured
            self._update_resolution_schedule()
            
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            
            # Training step
            metrics, is_step = self.train_step(batch)
            accum_loss += metrics['loss']
            
            if is_step:
                self.step += 1
                
                # Average loss over accumulation steps
                metrics['loss'] = accum_loss / self.grad_accumulation_steps
                accum_loss = 0.0
                
                # Logging
                if self.step % self.log_every == 0:
                    self.log(metrics)
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'grad': f"{metrics['grad_norm']:.2f}",
                        'lr': f"{metrics['lr']:.2e}",
                    })
                
                # Visual debugging (more frequent than full validation)
                if visual_debug_fn is not None:
                    # Check if visual_debug_interval is configured (ProductionTrainer only)
                    interval = getattr(self, 'visual_debug_interval', 0)
                    if interval > 0 and self.step % interval == 0:
                        visual_debug_fn(self.ema.model if self.ema else self.model, self.step)
                
                # Checkpointing
                if self.step % self.checkpoint_every == 0:
                    self.save_checkpoint()
                    
                    # Run validation if provided
                    if validate_fn is not None:
                        # Free training memory before validation
                        torch.cuda.empty_cache()
                        validate_fn(self.model, self.ema, self.step, self.device)
                        # Clean up after validation
                        torch.cuda.empty_cache()
                        # Put model back in train mode
                        self.model.train()
                
                pbar.update(1)
        
        pbar.close()
        
        # Final checkpoint
        self.save_checkpoint(self.checkpoint_dir / 'checkpoint_final.pt')
        print("Training complete!")


class ProductionTrainer(Trainer):
    """Production trainer with config-driven initialization.
    
    Extends the base Trainer to accept a Config object instead of
    individual parameters, enabling cleaner initialization and better
    separation of concerns.
    """
    
    def __init__(self, model, dataloader, config, device='cuda', experiment_name=None):
        """Initialize from config object.
        
        Args:
            model: NanoDiT model
            dataloader: iterable dataloader
            config: Config object from config_loader
            device: torch device
            experiment_name: Optional experiment name for TensorBoard run name
        """
        from .config_loader import Config
        
        # Extract config values
        training = config.training
        checkpoint_cfg = config.checkpoint
        logging_cfg = config.logging
        
        # Call parent init with extracted values
        super().__init__(
            model=model,
            dataloader=dataloader,
            device=device,
            total_steps=training.total_steps,
            warmup_steps=training.warmup_steps,
            peak_lr=training.optimizer.lr,
            min_lr=training.optimizer.min_lr,
            weight_decay=training.optimizer.weight_decay,
            grad_clip=training.grad_clip,
            ema_decay=training.ema_decay,
            cfg_probs=training.cfg_dropout.to_dict(),
            grad_accumulation_steps=training.grad_accumulation_steps,
            checkpoint_every=checkpoint_cfg.save_every,
            log_every=logging_cfg.log_every,
            checkpoint_dir=checkpoint_cfg.output_dir,
            optimizer_config=training.optimizer,
        )
        
        # Store additional config for production features
        self.config = config
        self.experiment_name = experiment_name or "default"
        self.ema_warmup_steps = training.ema_warmup_steps
        self.timestep_sampling = training.timestep_sampling
        self.logit_normal_loc = training.logit_normal_loc
        self.logit_normal_scale = training.logit_normal_scale
        
        # Monitoring flags
        self.monitor_velocity = logging_cfg.monitor_velocity_norm
        self.monitor_grad = logging_cfg.monitor_grad_norm
        self.velocity_warning = logging_cfg.velocity_norm_warning
        self.grad_warning = logging_cfg.grad_norm_warning
        
        # Visual debugging interval
        self.visual_debug_interval = config.validation.visual_debug_interval
        
        # REPA config
        self.repa_config = training.repa if training.repa.enabled else None
        
        # TREAD config
        self.tread_config = training.tread if training.tread.enabled else None
        
        # Resolution schedule
        self.resolution_phases = training.get_resolution_phases()
        self._current_resolution_scale = None  # Will be set on first step
        
        # Perceptual loss (LPIPS)
        if training.perceptual.enabled:
            self.perceptual_module = PerceptualLossModule(device=device)
            self.perceptual_config = training.perceptual
        else:
            self.perceptual_module = None
            self.perceptual_config = None
        
        # Prediction type (v_prediction or x_prediction)
        self.prediction_type = config.model.prediction_type
        self.t_clamp_min = config.model.t_clamp_min
        
        # Gradient accumulation state
        self.accum_steps = 0
        self.accum_loss = 0.0
        
        # TensorBoard logging
        try:
            from torch.utils.tensorboard import SummaryWriter
            # Use experiment directory root for tensorboard (not inside checkpoints/)
            exp_dir = Path(checkpoint_cfg.output_dir).parent  # checkpoints/ -> experiment/
            tensorboard_dir = exp_dir / 'tensorboard'
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"  TensorBoard logging: {tensorboard_dir}")
        except ImportError:
            print("  TensorBoard not available (install: pip install tensorboard)")
            self.writer = None
        
        print(f"ProductionTrainer initialized:")
        print(f"  Gradient accumulation: {self.grad_accumulation_steps} steps")
        print(f"  Effective batch size: {training.batch_size * self.grad_accumulation_steps}")
        print(f"  Timestep sampling: {self.timestep_sampling}")
        print(f"  EMA warmup: {self.ema_warmup_steps} steps")
    
    def log(self, metrics):
        """Log metrics to file, console, and TensorBoard.
        
        Overrides parent log() to add TensorBoard support.
        """
        # Call parent logging (JSONL file)
        super().log(metrics)
        
        # Log to TensorBoard
        if self.writer is not None:
            step = metrics.get('step', self.step)
            
            # Training metrics
            if 'loss' in metrics:
                self.writer.add_scalar('train/loss', metrics['loss'], step)
            if 'grad_norm' in metrics:
                self.writer.add_scalar('train/grad_norm', metrics['grad_norm'], step)
            if 'lr' in metrics:
                self.writer.add_scalar('train/learning_rate', metrics['lr'], step)
            
            # Per-layer gradient norms (for monitoring patch learning)
            for key, value in metrics.items():
                if key.startswith('grad_norm/'):
                    layer_name = key.split('/', 1)[1]
                    self.writer.add_scalar(f'gradients/{layer_name}', value, step)
            
            # Weight statistics (for monitoring parameter evolution)
            for key, value in metrics.items():
                if key.startswith('weights/'):
                    weight_name = key.split('/', 1)[1]
                    self.writer.add_scalar(f'weights/{weight_name}', value, step)
            
            # GPU memory metrics
            if 'vram_allocated_gb' in metrics:
                self.writer.add_scalar('memory/vram_allocated_gb', metrics['vram_allocated_gb'], step)
            if 'vram_reserved_gb' in metrics:
                self.writer.add_scalar('memory/vram_reserved_gb', metrics['vram_reserved_gb'], step)
            
            # Additional monitoring
            if 'velocity_norm' in metrics:
                self.writer.add_scalar('monitor/velocity_norm', metrics['velocity_norm'], step)
            if 'ema_decay' in metrics:
                self.writer.add_scalar('train/ema_decay', metrics['ema_decay'], step)
            if 'repa_loss' in metrics:
                self.writer.add_scalar('train/repa_loss', metrics['repa_loss'], step)
            if 'lpips_loss' in metrics:
                self.writer.add_scalar('train/lpips_loss', metrics['lpips_loss'], step)
    
    def __del__(self):
        """Cleanup: close TensorBoard writer."""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    import torch.nn.functional as F
    
    # Test timestep sampling
    print("Testing logit-normal sampling...")
    t = logit_normal_sample((1000,))
    print(f"Median: {t.median():.3f}")
    print(f"Min: {t.min():.3f}, Max: {t.max():.3f}")
    print(f"In [0.3, 0.7]: {((t >= 0.3) & (t <= 0.7)).sum().item() / 1000:.2%}")
    print("✓ Timestep sampling test passed")
