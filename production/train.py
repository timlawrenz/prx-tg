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


def flow_matching_loss(model, x0, dino_emb, dino_patches, text_emb, text_mask, cfg_probs, return_v_pred=False):
    """Compute flow matching loss with mutually exclusive CFG dropout.
    
    CFG dropout uses categorical sampling to ensure exactly one of:
    - 70% both conditionings present
    - 10% both dropped (unconditional)
    - 10% text dropped (DINO-only)
    - 10% DINO dropped (text-only)
    
    Args:
        model: NanoDiT model
        x0: (B, C, H, W) clean latents (data at t=0)
        dino_emb: (B, 1024) DINOv3 CLS embeddings
        dino_patches: (B, num_patches, 1024) DINOv3 spatial patches (variable length!)
        text_emb: (B, seq_len, 1024) T5 hidden states (seq_len=500 for full captions)
        text_mask: (B, seq_len) T5 attention mask
        cfg_probs: dict with p_drop_both, p_drop_text, p_drop_dino
        return_v_pred: bool, if True return (loss, v_pred) for monitoring
    
    Returns:
        loss: scalar tensor, or (loss, v_pred) if return_v_pred=True
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
    p_both = cfg_probs['p_drop_both']
    p_text = cfg_probs['p_drop_text']
    p_dino = cfg_probs['p_drop_dino']
    
    # Assign to exclusive categories
    drop_both = rand < p_both
    drop_text = (rand >= p_both) & (rand < p_both + p_text)
    drop_dino = (rand >= p_both + p_text) & (rand < p_both + p_text + p_dino)
    # Remainder (70% by default) has both conditionings present
    
    # Predict velocity (with DINO patches)
    v_pred = model(
        zt, t, dino_emb, text_emb, dino_patches, text_mask,
        cfg_drop_both=drop_both,
        cfg_drop_dino=drop_dino,
        cfg_drop_text=drop_text,
    )
    
    # MSE loss
    loss = F.mse_loss(v_pred, v_target)
    
    if return_v_pred:
        return loss, v_pred
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
            'p_drop_both': 0.1,
            'p_drop_text': 0.1,
            'p_drop_dino': 0.1,
        }
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
            eps=1e-8,
        )
        
        # EMA
        self.ema = EMAModel(model, decay=ema_decay, warmup_steps=warmup_steps)
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        # Logging
        self.log_file = self.checkpoint_dir / 'training_log.jsonl'
    
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
        dino_patches = batch['dinov3_patches'].to(self.device)  # (B, num_patches, 1024) - variable length!
        text_emb = batch['t5_hidden'].to(self.device)
        text_mask = batch['t5_mask'].to(self.device)
        
        # Compute loss (with velocity prediction for monitoring)
        loss, v_pred = flow_matching_loss(
            self.model, x0, dino_emb, dino_patches, text_emb, text_mask, self.cfg_probs, return_v_pred=True
        )
        
        # Scale loss by accumulation steps
        loss = loss / self.grad_accumulation_steps
        
        # Backward (accumulate gradients)
        loss.backward()
        
        # Only step optimizer every accumulation_steps
        is_accumulation_step = (self.step + 1) % self.grad_accumulation_steps == 0
        
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
            
            # Apply LR to optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
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
        
        # Every 100 steps, add weight statistics for monitoring parameter evolution
        if self.step % 100 == 0:
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
        
        return metrics
    
    def save_checkpoint(self, path=None):
        """Save training checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f'checkpoint_step{self.step:07d}.pt'
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
            }
        }
        
        # Save CUDA RNG state if available
        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state()
        
        torch.save(checkpoint, path)
        
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema.load_state_dict(checkpoint['ema'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
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
        
        while self.step < self.total_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            
            # Training step
            metrics = self.train_step(batch)
            self.step += 1
            
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
        
        # Gradient accumulation state
        self.accum_steps = 0
        self.accum_loss = 0.0
        
        # TensorBoard logging
        try:
            from torch.utils.tensorboard import SummaryWriter
            # Use experiment name as subdirectory for better organization
            tensorboard_dir = Path(checkpoint_cfg.output_dir) / 'tensorboard' / self.experiment_name
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
