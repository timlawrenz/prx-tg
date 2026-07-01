"""Validation tests for Nano DiT."""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import lpips
import numpy as np
from PIL import Image

from .sample import ValidationSampler, load_vae_decoder, save_images


# Fixed test sample indices for consistency across validation runs
# These will be selected from the validation dataset
RECONSTRUCTION_TEST_INDICES = list(range(0, 100, 4))  # 25 samples evenly spaced

DINO_SWAP_TEST_PAIRS = [
    (5, 42),   # Pair 1
    (12, 78),  # Pair 2
    (23, 56),  # Pair 3
    (34, 91),  # Pair 4
    (47, 63),  # Pair 5
]

# Text manipulation test: find common patterns in captions and swap them
TEXT_MANIP_REPLACEMENTS = [
    ('slender', 'muscular'),
    ('pale', 'tan'),
    ('woman', 'person'),
    ('long', 'short'),
    ('dark', 'light'),
    ('young', 'mature'),
    ('facing right', 'facing left'),
    ('directly at', 'away from'),
]

TEXT_MANIP_TEST_INDICES = [8, 15, 27, 39, 51]

# Text-only generation test: pure text-to-image (no DINO guidance)
# This simulates future text-to-image usage without DINO embeddings
TEXT_ONLY_TEST_INDICES = list(range(0, 100, 5))  # 20 samples evenly spaced


def create_image_collage(images, labels=None, spacing=10):
    """Create a horizontal collage from multiple tensors or PIL images.
    
    Args:
        images: List of torch tensors (C, H, W) in [-1, 1] or PIL Images
        labels: Optional list of text labels for each image
        spacing: Pixels between images
    
    Returns:
        collage_array: numpy array (H, W, 3) in [0, 255] uint8
    """
    # Convert all to PIL Images
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            # Denormalize from [-1, 1] to [0, 1]
            img_np = (img.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            # Convert to (H, W, C) uint8
            img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
        else:
            pil_img = img
        pil_images.append(pil_img)
    
    # Calculate collage dimensions
    max_height = max(img.height for img in pil_images)
    total_width = sum(img.width for img in pil_images) + spacing * (len(pil_images) - 1)
    
    # Create white background
    collage = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    
    # Paste images
    x_offset = 0
    for pil_img in pil_images:
        # Center vertically
        y_offset = (max_height - pil_img.height) // 2
        collage.paste(pil_img, (x_offset, y_offset))
        x_offset += pil_img.width + spacing
    
    # Convert to numpy for TensorBoard
    collage_array = np.array(collage)  # (H, W, 3) uint8
    
    return collage_array


class ValidationRunner:
    """Run validation tests on Nano DiT model."""
    
    def __init__(
        self,
        model,
        ema,
        dataloader,
        device='cuda',
        output_dir='validation',
        lpips_net='alex',
        tensorboard_writer=None,
        text_scale=3.0,
        dino_scale=2.0,
        num_steps=50,
        self_guidance=False,
        guidance_scale=3.0,
        prediction_type="v_prediction",
    ):
        """
        Args:
            model: NanoDiT model (training weights)
            ema: EMA model
            dataloader: validation dataloader
            device: torch device
            output_dir: directory for validation outputs
            lpips_net: LPIPS network ('alex' or 'vgg')
            tensorboard_writer: Optional TensorBoard SummaryWriter
            prediction_type: "v_prediction" or "x_prediction"
        """
        self.model = model
        self.ema = ema
        self.dataloader = dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.tb_writer = tensorboard_writer
        self.prediction_type = prediction_type

        # Sampling CFG scales for validation
        self.text_scale = text_scale
        self.dino_scale = dino_scale
        self.num_steps = num_steps
        self.self_guidance = self_guidance
        self.guidance_scale = guidance_scale
        
        # Load VAE decoder (not needed for pixel-space)
        if prediction_type == "x_prediction":
            print("Pixel-space mode: skipping VAE decoder load")
            self.vae = None
        else:
            print("Loading VAE decoder...")
            self.vae = load_vae_decoder(device=device)
        
        # Load LPIPS metric
        print(f"Loading LPIPS metric ({lpips_net})...")
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
        
        # T5 encoder/tokenizer for text manipulation (lazy loaded)
        self.t5_encoder = None
        self.t5_tokenizer = None
        
        # Cache validation samples for consistent testing
        self.validation_samples = None
    
    def load_validation_samples(self):
        """Load and cache all validation samples."""
        if self.validation_samples is not None:
            return self.validation_samples
        
        print("Loading validation samples...")
        samples = []
        
        # Get enough samples to cover all test indices
        max_idx = max(
            max(RECONSTRUCTION_TEST_INDICES),
            max(max(pair) for pair in DINO_SWAP_TEST_PAIRS),
            max(TEXT_MANIP_TEST_INDICES),
        )
        
        data_iter = iter(self.dataloader)
        sample_count = 0
        
        while sample_count <= max_idx:
            batch = next(data_iter)
            batch_size = batch['image_data'].shape[0]
            
            for i in range(batch_size):
                if sample_count <= max_idx:
                    sample = {
                        'image_data': batch['image_data'][i].cpu(),
                        'image_id': batch['image_ids'][i],
                    }
                    # Stratum fields (always present)
                    if 'dino_embedding' in batch:
                        sample['dino_embedding'] = batch['dino_embedding'][i].cpu()
                    if 'dinov3_patches' in batch:
                        sample['dinov3_patches'] = batch['dinov3_patches'][i].cpu()
                    if 't5_hidden' in batch:
                        sample['t5_hidden'] = batch['t5_hidden'][i].cpu()
                    if 't5_mask' in batch:
                        sample['t5_mask'] = batch['t5_mask'][i].cpu()
                    if 'captions' in batch:
                        sample['caption'] = batch['captions'][i]
                    # Eidolon fields
                    if 'identity_emb' in batch:
                        sample['identity_emb'] = batch['identity_emb'][i].cpu()
                    if 'geometry_emb' in batch:
                        sample['geometry_emb'] = batch['geometry_emb'][i].cpu()
                    samples.append(sample)
                    sample_count += 1
        
        self.validation_samples = samples
        print(f"Loaded {len(samples)} validation samples")
        return samples
    
    def _get_adapter_name(self):
        """Detect which adapter the model is using."""
        from production.adapters import EidolonAdapter
        if isinstance(self.model.adapter, EidolonAdapter):
            return "eidolon"
        return "stratum"
    
    def _build_conditioning_kwargs(self, sample, batch_dim=True):
        """Build adapter-specific conditioning kwargs from a cached sample.
        
        Args:
            sample: cached validation sample dict
            batch_dim: if True, add batch dimension via .unsqueeze(0)
            
        Returns:
            dict of kwargs for sampler.generate(**kwargs)
        """
        adapter_name = self._get_adapter_name()
        kwargs = {}
        
        if adapter_name == "stratum":
            dino_emb = sample['dino_embedding']
            dino_patches = sample['dinov3_patches']
            text_emb = sample['t5_hidden']
            text_mask = sample['t5_mask']
            if batch_dim:
                dino_emb = dino_emb.unsqueeze(0)
                dino_patches = dino_patches.unsqueeze(0)
                text_emb = text_emb.unsqueeze(0)
                text_mask = text_mask.unsqueeze(0)
            kwargs['dino_emb'] = dino_emb
            kwargs['dino_patches'] = dino_patches
            kwargs['text_emb'] = text_emb
            kwargs['text_mask'] = text_mask
        elif adapter_name == "eidolon":
            identity_emb = sample['identity_emb']
            geometry_emb = sample['geometry_emb']
            if batch_dim:
                identity_emb = identity_emb.unsqueeze(0)
                geometry_emb = geometry_emb.unsqueeze(0)
            kwargs['identity_emb'] = identity_emb
            kwargs['geometry_emb'] = geometry_emb
        else:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        return kwargs
    
    def _build_batch_conditioning_kwargs(self, batch_samples):
        """Build adapter-specific conditioning kwargs from a list of cached samples.
        
        Args:
            batch_samples: list of cached validation sample dicts
            
        Returns:
            dict of kwargs for sampler.generate(**kwargs)
        """
        adapter_name = self._get_adapter_name()
        kwargs = {}
        
        if adapter_name == "stratum":
            kwargs['dino_emb'] = torch.stack([s['dino_embedding'] for s in batch_samples])
            kwargs['dino_patches'] = torch.stack([s['dinov3_patches'] for s in batch_samples])
            kwargs['text_emb'] = torch.stack([s['t5_hidden'] for s in batch_samples])
            kwargs['text_mask'] = torch.stack([s['t5_mask'] for s in batch_samples])
        elif adapter_name == "eidolon":
            kwargs['identity_emb'] = torch.stack([s['identity_emb'] for s in batch_samples])
            kwargs['geometry_emb'] = torch.stack([s['geometry_emb'] for s in batch_samples])
        
        return kwargs
    
    def _get_latent_size(self):
        """Compute latent size from current resolution scale.
        
        Returns the latent spatial size that matches the current training
        resolution, so validation generates at the same resolution the model
        was trained at.
        """
        base_size = getattr(self.model, 'input_size', 128)
        latent_size = base_size
        return latent_size
    
    def run_reconstruction_test(self, step, sampler, latent_size=None):
        """Test 1: Generate from original conditioning, measure LPIPS.
        
        For stratum: uses DINO + T5 conditioning.
        For eidolon: uses identity_emb + geometry_emb conditioning.
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
            latent_size: spatial size for generation (respects resolution schedule)
        
        Returns:
            dict with lpips_scores and mean_lpips
        """
        print(f"Running reconstruction test (step {step})...")
        
        # Move LPIPS to GPU for this test
        self.lpips_fn.to(self.device)
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'reconstruction'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        lpips_scores = []
        
        # Generate in batches for efficiency (reduced batch_size for 1024x1024 to avoid OOM)
        batch_size = 1
        
        for i in tqdm(range(0, len(RECONSTRUCTION_TEST_INDICES), batch_size), desc='Reconstruction'):
            batch_indices = RECONSTRUCTION_TEST_INDICES[i:i+batch_size]
            
            # Gather batch
            batch_samples = [samples[idx] for idx in batch_indices]
            gt_images_raw = torch.stack([s['image_data'] for s in batch_samples])
            image_ids = [s['image_id'] for s in batch_samples]
            
            # Build adapter-specific conditioning kwargs
            cond_kwargs = self._build_batch_conditioning_kwargs(batch_samples)
            
            # Generate images at current training resolution
            gen_images = sampler.generate(
                latent_size=latent_size,
                text_scale=self.text_scale/2.0,
                dino_scale=self.dino_scale/2.0,
                **cond_kwargs,
            )
            
            # Resize GT to match generation resolution
            if latent_size is not None:
                gt_h, gt_w = gt_images_raw.shape[2], gt_images_raw.shape[3]
                if gt_h != latent_size or gt_w != latent_size:
                    gt_images_raw = F.interpolate(
                        gt_images_raw, size=(latent_size, latent_size),
                        mode='bilinear', align_corners=False
                    )
            
            # Convert ground truth [0,1] → [-1,1] for LPIPS comparison
            gt_images = gt_images_raw.to(self.device) * 2 - 1
            
            # Compute LPIPS
            for j in range(len(batch_indices)):
                lpips_val = self.lpips_fn(
                    gen_images[j:j+1],
                    gt_images[j:j+1]
                ).item()
                lpips_scores.append(lpips_val)
            
            # Save generated images
            save_images(
                gen_images,
                output_dir,
                prefix=f'recon',
                image_ids=image_ids
            )
            
            # Clear GPU memory after each batch
            del cond_kwargs, gen_images, gt_images
            torch.cuda.empty_cache()
        
        mean_lpips = sum(lpips_scores) / len(lpips_scores)
        
        # Move LPIPS back to CPU to save GPU memory
        self.lpips_fn.to('cpu')
        torch.cuda.empty_cache()
        
        return {
            'lpips_scores': lpips_scores,
            'mean_lpips': mean_lpips,
            'num_samples': len(lpips_scores),
        }
    
    def run_dino_swap_test(self, step, sampler, latent_size=None):
        """Test 2: Swap DINO embeddings, keep original captions. Stratum only.
        
        For each pair (A, B):
        - Generate with caption_A + dino_A (reference)
        - Generate with caption_A + dino_B (swapped)
        - Generate with caption_B + dino_B (reference)
        
        This lets you visually compare:
        - captionA_dinoA vs captionA_dinoB to see DINO's effect
        - captionA_dinoB vs captionB_dinoB to see caption's effect
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
            latent_size: spatial size for generation (respects resolution schedule)
        
        Returns:
            dict with test results
        """
        adapter_name = self._get_adapter_name()
        if adapter_name != "stratum":
            return {'skipped': True, 'reason': f'DINO swap not applicable to {adapter_name} adapter'}
        
        print(f"Running DINO swap test (step {step})...")
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'dino_swap'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for pair_idx, (idx_a, idx_b) in enumerate(DINO_SWAP_TEST_PAIRS):
            sample_a = samples[idx_a]
            sample_b = samples[idx_b]
            
            # Build kwargs for each variant using the helper
            kwargs_a = self._build_conditioning_kwargs(sample_a, batch_dim=True)
            kwargs_b = self._build_conditioning_kwargs(sample_b, batch_dim=True)
            
            # 1. Reference: A's conditioning
            gen_a_ref = sampler.generate(
                latent_size=latent_size,
                text_scale=self.text_scale/2.0,
                dino_scale=self.dino_scale/2.0,
                **kwargs_a,
            )[0]  # (3, H, W)
            
            # 2. Swapped: A's text + B's DINO (swap CLS + patches together)
            kwargs_swap = dict(kwargs_a)
            kwargs_swap['dino_emb'] = kwargs_b['dino_emb']
            kwargs_swap['dino_patches'] = kwargs_b['dino_patches']
            gen_a_swap = sampler.generate(
                latent_size=latent_size,
                text_scale=self.text_scale/2.0,
                dino_scale=self.dino_scale/2.0,
                **kwargs_swap,
            )[0]  # (3, H, W)
            
            # 3. Reference: B's conditioning
            gen_b_ref = sampler.generate(
                latent_size=latent_size,
                text_scale=self.text_scale/2.0,
                dino_scale=self.dino_scale/2.0,
                **kwargs_b,
            )[0]  # (3, H, W)
            
            # Create collage: [A_ref | A_swap | B_ref]
            collage_array = create_image_collage([gen_a_ref, gen_a_swap, gen_b_ref])
            
            # Save collage as file
            collage_img = Image.fromarray(collage_array)
            collage_path = output_dir / f'pair{pair_idx}_collage.png'
            collage_img.save(collage_path)
            
            # Log to TensorBoard (if available)
            if self.tb_writer is not None:
                collage_tensor = torch.from_numpy(collage_array).permute(2, 0, 1)  # (3, H, W)
                self.tb_writer.add_image(
                    f'validation/dino_swap_pair{pair_idx}',
                    collage_tensor,
                    global_step=step,
                    dataformats='CHW'
                )
            
            results.append({
                'pair_idx': pair_idx,
                'idx_a': idx_a,
                'idx_b': idx_b,
                'caption_a': sample_a.get('caption', '(no caption)')[:100] + '...',
                'caption_b': sample_b.get('caption', '(no caption)')[:100] + '...',
            })
            
            # Clear GPU memory
            del gen_a_ref, gen_a_swap, gen_b_ref
            torch.cuda.empty_cache()
        
        return {
            'num_pairs': len(DINO_SWAP_TEST_PAIRS),
            'results': results,
        }
    
    def run_divergence_test(self, step, sampler, latent_size=None):
        """Test 2b: CFG Divergence Test - prove text and DINO are independent.
        
        From the-plan.md Part E.2:
        Generate from the SAME starting noise with two extreme CFG settings:
        - Pass 1: scale_text=4.0, scale_dino=0.0 (text-only, no DINO)
        - Pass 2: scale_text=0.0, scale_dino=2.0 (DINO-only, no text)
        
        This proves that text and DINO conditioning are truly independent.
        Stratum only.
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
            latent_size: spatial size for generation (respects resolution schedule)
        
        Returns:
            dict with test results
        """
        adapter_name = self._get_adapter_name()
        if adapter_name != "stratum":
            return {'skipped': True, 'reason': f'Divergence test not applicable to {adapter_name} adapter'}
        
        print(f"Running CFG divergence test (step {step})...")
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'divergence'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a small subset for this test
        test_indices = [5, 23, 47, 78, 91]
        results = []
        
        for test_idx, sample_idx in enumerate(test_indices):
            sample = samples[sample_idx]
            kwargs = self._build_conditioning_kwargs(sample, batch_dim=True)
            
            # Generate from same noise with extreme CFG scales
            # Fix random seed for reproducibility
            torch.manual_seed(42 + test_idx)
            
            # Generate: text-only (scale_text=4.0, scale_dino=0.0)
            gen_text_only = sampler.generate(
                latent_size=latent_size,
                text_scale=4.0,
                dino_scale=0.0,
                **kwargs,
            )[0]  # (3, H, W)
            
            # Reset seed to same starting noise
            torch.manual_seed(42 + test_idx)
            
            # Generate: DINO-only (scale_text=0.0, scale_dino=2.0)
            gen_dino_only = sampler.generate(
                latent_size=latent_size,
                text_scale=0.0,
                dino_scale=2.0,
                **kwargs,
            )[0]  # (3, H, W)
            
            # Reset seed again for reference generation with both
            torch.manual_seed(42 + test_idx)
            
            # Generate: both (default scales)
            gen_both = sampler.generate(
                latent_size=latent_size,
                **kwargs,
            )[0]  # (3, H, W)
            
            # Create collage: [text-only | DINO-only | both]
            collage_array = create_image_collage([gen_text_only, gen_dino_only, gen_both])
            
            # Save collage
            collage_img = Image.fromarray(collage_array)
            collage_path = output_dir / f'sample{test_idx}_divergence.png'
            collage_img.save(collage_path)
            
            # Log to TensorBoard
            if self.tb_writer is not None:
                collage_tensor = torch.from_numpy(collage_array).permute(2, 0, 1)  # (3, H, W)
                self.tb_writer.add_image(
                    f'validation/divergence_sample{test_idx}',
                    collage_tensor,
                    global_step=step,
                    dataformats='CHW'
                )
            
            results.append({
                'sample_idx': sample_idx,
                'caption': sample.get('caption', '(no caption)')[:100] + '...',
            })
            
            # Clear GPU memory
            del gen_text_only, gen_dino_only, gen_both
            torch.cuda.empty_cache()
        
        return {
            'num_samples': len(test_indices),
            'results': results,
        }
    
    def run_text_only_test(self, step, sampler, latent_size=None):
        """Test 2c: Generate from text ONLY (no DINO guidance). Stratum only.
        
        Simulates pure text-to-image generation for future use cases.
        Sets dino_scale=0.0 to disable DINO conditioning entirely.
        
        This answers: "How well would this model work as a text-to-image model
        without DINO embeddings?"
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
            latent_size: spatial size for generation (respects resolution schedule)
        
        Returns:
            dict with lpips_scores and mean_lpips
        """
        adapter_name = self._get_adapter_name()
        if adapter_name != "stratum":
            return {'skipped': True, 'reason': f'Text-only test not applicable to {adapter_name} adapter'}
        
        print(f"Running text-only generation test (step {step})...")
        
        # Move LPIPS to GPU for this test
        self.lpips_fn.to(self.device)
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'text_only'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        lpips_scores = []
        
        # Generate in batches
        batch_size = 1
        
        for i in tqdm(range(0, len(TEXT_ONLY_TEST_INDICES), batch_size), desc='Text-only generation'):
            batch_indices = TEXT_ONLY_TEST_INDICES[i:i+batch_size]
            
            # Gather batch
            batch_samples = [samples[idx] for idx in batch_indices]
            gt_images_raw = torch.stack([s['image_data'] for s in batch_samples])
            image_ids = [s['image_id'] for s in batch_samples]
            
            # Build adapter-specific conditioning kwargs
            cond_kwargs = self._build_batch_conditioning_kwargs(batch_samples)
            
            # Generate images with TEXT ONLY (dino_scale=0.0 disables DINO CLS + patches)
            # Force dual-CFG path even in self-guidance mode
            gen_images = sampler.generate(
                latent_size=latent_size,
                self_guidance=False,
                dino_scale=0.0,  # Zero DINO influence
                text_scale=3.0,  # Normal text guidance
                **cond_kwargs,
            )
            
            # Resize GT to match generation resolution
            if latent_size is not None:
                gt_h, gt_w = gt_images_raw.shape[2], gt_images_raw.shape[3]
                if gt_h != latent_size or gt_w != latent_size:
                    gt_images_raw = F.interpolate(
                        gt_images_raw, size=(latent_size, latent_size),
                        mode='bilinear', align_corners=False
                    )
            
            # Convert ground truth [0,1] → [-1,1] for LPIPS comparison
            gt_images = gt_images_raw.to(self.device) * 2 - 1
            
            # Compute LPIPS
            for j in range(len(batch_indices)):
                lpips_val = self.lpips_fn(
                    gen_images[j:j+1],
                    gt_images[j:j+1]
                ).item()
                lpips_scores.append(lpips_val)
            
            # Save generated images
            save_images(
                gen_images,
                output_dir,
                prefix=f'text_only',
                image_ids=image_ids
            )
            
            # Clear GPU memory after each batch
            del cond_kwargs, gen_images, gt_images
            torch.cuda.empty_cache()
        
        mean_lpips = sum(lpips_scores) / len(lpips_scores)
        
        # Move LPIPS back to CPU to save GPU memory
        self.lpips_fn.to('cpu')
        torch.cuda.empty_cache()
        
        return {
            'lpips_scores': lpips_scores,
            'mean_lpips': mean_lpips,
            'num_samples': len(lpips_scores),
        }
    
    def load_t5_encoder(self):
        """Lazy load T5 encoder and tokenizer for text manipulation test."""
        if self.t5_encoder is None:
            print("Loading T5 encoder for text manipulation...")
            from transformers import T5EncoderModel, AutoTokenizer
            
            # Try loading from local cache first, fallback to network
            try:
                self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-large", local_files_only=True)
            except Exception:
                self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")
                
            try:
                self.t5_encoder = T5EncoderModel.from_pretrained(
                    "t5-large", 
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
            except Exception:
                self.t5_encoder = T5EncoderModel.from_pretrained(
                    "t5-large", 
                    torch_dtype=torch.float16
                )
            self.t5_encoder.to(self.device)
            self.t5_encoder.eval()
        
        return self.t5_encoder, self.t5_tokenizer
    
    def encode_caption(self, caption):
        """Encode a caption with T5 to get hidden states and attention mask.
        
        Returns:
            tuple: (hidden_states, attention_mask) as torch tensors in float32
                   hidden_states shape: (1, 512, 1024)
                   attention_mask shape: (1, 512)
        """
        encoder, tokenizer = self.load_t5_encoder()
        
        # Tokenize with T5's max length (512 tokens, not CLIP's 77)
        inputs = tokenizer(
            caption,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state.float()  # (1, 512, 1024) in float32
        
        return hidden_states, attention_mask
    
    def run_text_manip_test(self, step, sampler, latent_size=None):
        """Test 3: Modify caption, keep DINO embedding. Stratum only.
        
        NOT applicable to eidolon (no T5). Skip when adapter_name == "eidolon".
        
        Dynamically finds text to replace from a list of common patterns.
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
            latent_size: spatial size for generation (respects resolution schedule)
        
        Returns:
            dict with test results and LPIPS comparison scores
        """
        adapter_name = self._get_adapter_name()
        if adapter_name == "eidolon":
            return {'skipped': True, 'reason': 'Text manipulation not applicable to eidolon (no T5)'}
        
        print(f"Running text manipulation test (step {step})...")
        
        # Move LPIPS to GPU for this test
        self.lpips_fn.to(self.device)
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'text_manip'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for idx in TEXT_MANIP_TEST_INDICES:
            sample = samples[idx]
            original_caption = sample['caption']
            
            # Find first matching pattern in this caption
            original_text = None
            modified_text = None
            for orig, mod in TEXT_MANIP_REPLACEMENTS:
                if orig in original_caption.lower():
                    # Case-sensitive search for replacement
                    if orig in original_caption:
                        original_text = orig
                        modified_text = mod
                        break
                    # Try capitalized version
                    elif orig.capitalize() in original_caption:
                        original_text = orig.capitalize()
                        modified_text = mod.capitalize()
                        break
            
            if original_text is None:
                print(f"Warning: No matching text patterns found in caption at idx {idx}")
                print(f"Caption: {original_caption[:200]}...")
                continue
            
            # Modified caption (string replace)
            modified_caption = original_caption.replace(original_text, modified_text)
            
            # Build kwargs for original conditioning
            kwargs_orig = self._build_conditioning_kwargs(sample, batch_dim=True)
            
            # Generate with original caption
            gen_orig = sampler.generate(
                latent_size=latent_size,
                text_scale=self.text_scale/2.0,
                dino_scale=self.dino_scale/2.0,
                **kwargs_orig,
            )[0]  # (3, H, W)
            
            # Re-encode modified caption with T5
            text_emb_mod, text_mask_mod = self.encode_caption(modified_caption)
            
            # Build kwargs with modified text
            kwargs_mod = dict(kwargs_orig)
            kwargs_mod['text_emb'] = text_emb_mod
            kwargs_mod['text_mask'] = text_mask_mod
            
            # Generate with modified caption
            gen_mod = sampler.generate(
                latent_size=latent_size,
                text_scale=self.text_scale/2.0,
                dino_scale=self.dino_scale/2.0,
                **kwargs_mod,
            )[0]  # (3, H, W)
            
            # Compute LPIPS between original and modified generations
            lpips_val = self.lpips_fn(gen_orig.unsqueeze(0), gen_mod.unsqueeze(0)).item()
            
            # Create collage: [original | modified]
            collage_array = create_image_collage([gen_orig, gen_mod])
            
            # Save collage as file
            collage_img = Image.fromarray(collage_array)
            collage_path = output_dir / f'sample{idx}_collage.png'
            collage_img.save(collage_path)
            
            # Log to TensorBoard (if available)
            if self.tb_writer is not None:
                collage_tensor = torch.from_numpy(collage_array).permute(2, 0, 1)  # (3, H, W)
                self.tb_writer.add_image(
                    f'validation/text_manip_{idx}',
                    collage_tensor,
                    global_step=step,
                    dataformats='CHW'
                )
                # Add caption as text
                self.tb_writer.add_text(
                    f'validation/text_manip_{idx}_caption',
                    f"Original: {original_caption[:100]}\nModified: {modified_caption[:100]}",
                    global_step=step
                )
            
            results.append({
                'idx': idx,
                'original_caption': original_caption,
                'modified_caption': modified_caption,
                'original_text': original_text,
                'modified_text': modified_text,
                'lpips_difference': lpips_val,
            })
            
            # Clear GPU memory
            del kwargs_orig, kwargs_mod, text_emb_mod, text_mask_mod
            del gen_orig, gen_mod
            torch.cuda.empty_cache()
        
        # Unload T5 encoder to free GPU memory
        if self.t5_encoder is not None:
            self.t5_encoder.to('cpu')  # Move to CPU before deletion
            del self.t5_encoder
            del self.t5_tokenizer
            self.t5_encoder = None
            self.t5_tokenizer = None
            torch.cuda.empty_cache()
            print("T5 encoder unloaded from GPU")
        
        # Move LPIPS back to CPU
        self.lpips_fn.to('cpu')
        torch.cuda.empty_cache()
        
        # Compute mean LPIPS difference
        if results:
            mean_lpips_diff = sum(r['lpips_difference'] for r in results) / len(results)
        else:
            mean_lpips_diff = 0.0
        
        return {
            'num_cases': len(TEXT_MANIP_TEST_INDICES),
            'num_successful': len(results),
            'mean_lpips_difference': mean_lpips_diff,
            'results': results,
        }
    
    # ── Eidolon-specific validation tests ────────────────────────────────────
    
    EIDOLON_SWAP_TEST_PAIRS = [
        (5, 42),   # Pair 1
        (12, 78),  # Pair 2
        (23, 56),  # Pair 3
        (34, 91),  # Pair 4
        (47, 63),  # Pair 5
    ]
    
    def run_eidolon_identity_swap_test(self, step, sampler, latent_size=None):
        """Eidolon Test: Identity swap — same geometry_emb, different identity_emb.
        
        For each pair (A, B):
        - Generate with identity_A + geometry_A (reference A)
        - Generate with identity_B + geometry_A (identity swapped)
        - Generate with identity_B + geometry_B (reference B)
        
        Visual comparison: A_ref vs A_swap should look different (different identity).
        External metric: AuraFace cosine should distinguish.
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
            latent_size: spatial size for generation
        
        Returns:
            dict with test results
        """
        print(f"Running eidolon identity swap test (step {step})...")
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'eidolon_identity_swap'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for pair_idx, (idx_a, idx_b) in enumerate(self.EIDOLON_SWAP_TEST_PAIRS):
            sample_a = samples[idx_a]
            sample_b = samples[idx_b]
            
            kwargs_a = self._build_conditioning_kwargs(sample_a, batch_dim=True)
            kwargs_b = self._build_conditioning_kwargs(sample_b, batch_dim=True)
            
            # 1. Reference: identity_A + geometry_A
            gen_a_ref = sampler.generate(
                latent_size=latent_size,
                **kwargs_a,
            )[0]  # (3, H, W)
            
            # 2. Identity swap: identity_B + geometry_A
            kwargs_swap = dict(kwargs_a)
            kwargs_swap['identity_emb'] = kwargs_b['identity_emb']
            gen_a_swap = sampler.generate(
                latent_size=latent_size,
                **kwargs_swap,
            )[0]  # (3, H, W)
            
            # 3. Reference: identity_B + geometry_B
            gen_b_ref = sampler.generate(
                latent_size=latent_size,
                **kwargs_b,
            )[0]  # (3, H, W)
            
            # Create collage: [A_ref | A_swap | B_ref]
            collage_array = create_image_collage([gen_a_ref, gen_a_swap, gen_b_ref])
            
            collage_img = Image.fromarray(collage_array)
            collage_path = output_dir / f'pair{pair_idx}_collage.png'
            collage_img.save(collage_path)
            
            if self.tb_writer is not None:
                collage_tensor = torch.from_numpy(collage_array).permute(2, 0, 1)
                self.tb_writer.add_image(
                    f'validation/eidolon_identity_swap_pair{pair_idx}',
                    collage_tensor,
                    global_step=step,
                    dataformats='CHW'
                )
            
            results.append({
                'pair_idx': pair_idx,
                'idx_a': idx_a,
                'idx_b': idx_b,
            })
            
            del gen_a_ref, gen_a_swap, gen_b_ref
            torch.cuda.empty_cache()
        
        return {
            'num_pairs': len(self.EIDOLON_SWAP_TEST_PAIRS),
            'results': results,
        }
    
    EIDOLON_GEOMETRY_SWEEP_INDICES = [10, 30, 50]
    EIDOLON_GEOMETRY_SWEEP_DIM = 0  # which z_g dimension to sweep
    EIDOLON_GEOMETRY_SWEEP_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    def run_eidolon_geometry_sweep_test(self, step, sampler, latent_size=None):
        """Eidolon Test: Geometry sweep — fix identity, sweep one z_g dimension.
        
        Fixes identity_emb, sweeps one z_g dimension through multiple values.
        DWPose should track pose change, AuraFace should hold (same identity).
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
            latent_size: spatial size for generation
        
        Returns:
            dict with test results
        """
        print(f"Running eidolon geometry sweep test (step {step})...")
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'eidolon_geometry_sweep'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        sweep_dim = self.EIDOLON_GEOMETRY_SWEEP_DIM
        
        for sample_idx in self.EIDOLON_GEOMETRY_SWEEP_INDICES:
            sample = samples[sample_idx]
            base_kwargs = self._build_conditioning_kwargs(sample, batch_dim=True)
            identity_emb = base_kwargs['identity_emb'].clone()
            base_geometry = base_kwargs['geometry_emb'].clone()
            
            sweep_images = []
            sweep_labels = []
            
            for val in self.EIDOLON_GEOMETRY_SWEEP_VALUES:
                modified_geometry = base_geometry.clone()
                modified_geometry[:, sweep_dim] = val
                
                sweep_kwargs = {
                    'identity_emb': identity_emb,
                    'geometry_emb': modified_geometry,
                }
                
                gen = sampler.generate(
                    latent_size=latent_size,
                    **sweep_kwargs,
                )[0]  # (3, H, W)
                
                sweep_images.append(gen)
                sweep_labels.append(f'z_g[{sweep_dim}]={val:.1f}')
            
            # Create collage of all sweep values
            collage_array = create_image_collage(sweep_images, labels=sweep_labels)
            
            collage_img = Image.fromarray(collage_array)
            collage_path = output_dir / f'sample{sample_idx}_sweep.png'
            collage_img.save(collage_path)
            
            if self.tb_writer is not None:
                collage_tensor = torch.from_numpy(collage_array).permute(2, 0, 1)
                self.tb_writer.add_image(
                    f'validation/eidolon_geometry_sweep_{sample_idx}',
                    collage_tensor,
                    global_step=step,
                    dataformats='CHW'
                )
            
            results.append({
                'sample_idx': sample_idx,
                'sweep_dim': sweep_dim,
                'sweep_values': self.EIDOLON_GEOMETRY_SWEEP_VALUES,
            })
            
            del sweep_images, sweep_kwargs
            torch.cuda.empty_cache()
        
        return {
            'num_samples': len(self.EIDOLON_GEOMETRY_SWEEP_INDICES),
            'sweep_dim': sweep_dim,
            'results': results,
        }
    
    def run_validation(self, step):
        """Run all validation tests, dispatched by adapter type.
        
        Args:
            step: current training step
        
        Returns:
            dict with all validation results
        """
        # Compute latent size from resolution schedule
        latent_size = self._get_latent_size()
        base_size = getattr(self.model, 'input_size', 128)
        scale = latent_size / base_size if base_size > 0 else 1.0
        adapter_name = self._get_adapter_name()
        
        print(f"\n{'='*60}")
        print(f"VALIDATION AT STEP {step}")
        print(f"  Adapter: {adapter_name}")
        if latent_size != base_size:
            print(f"  Resolution scale: {scale:.2f}x ({latent_size}×{latent_size} latent → {latent_size*8}px)")
        print(f"{'='*60}\n")
        
        # Free up memory before validation
        torch.cuda.empty_cache()
        
        # Backup current model weights to CPU (save GPU memory)
        print("Backing up model weights to CPU...")
        model_backup = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        # Load EMA weights into main model (in-place)
        print("Loading EMA weights for validation...")
        self.ema.copy_to(self.model)
        self.model.eval()
        
        # Create sampler using main model (now with EMA weights)
        sampler = ValidationSampler(
            self.model,
            self.vae,
            device=self.device,
            num_steps=self.num_steps,
            text_scale=self.text_scale,
            dino_scale=self.dino_scale,
            self_guidance=self.self_guidance,
            guidance_scale=self.guidance_scale,
            prediction_type=self.prediction_type,
        )
        
        try:
            # Reconstruction test runs for ALL adapters
            results = {
                'step': step,
                'latent_size': latent_size,
                'adapter': adapter_name,
                'reconstruction': self.run_reconstruction_test(step, sampler, latent_size=latent_size),
            }
            
            if adapter_name == "stratum":
                # Stratum-specific tests
                if self.self_guidance:
                    results['divergence'] = {'skipped': True, 'reason': 'self-guidance mode'}
                else:
                    results['divergence'] = self.run_divergence_test(step, sampler, latent_size=latent_size)
                
                # Text-only test (uses dual-CFG path)
                results['text_only'] = self.run_text_only_test(step, sampler, latent_size=latent_size)
                
                # DINO swap and text manipulation tests
                results['dino_swap'] = self.run_dino_swap_test(step, sampler, latent_size=latent_size)
                results['text_manip'] = self.run_text_manip_test(step, sampler, latent_size=latent_size)
                
            elif adapter_name == "eidolon":
                # Eidolon-specific tests
                results['dino_swap'] = {'skipped': True, 'reason': 'eidolon adapter'}
                results['divergence'] = {'skipped': True, 'reason': 'eidolon adapter'}
                results['text_only'] = {'skipped': True, 'reason': 'eidolon adapter (no text)'}
                results['text_manip'] = {'skipped': True, 'reason': 'eidolon adapter (no T5)'}
                results['eidolon_identity_swap'] = self.run_eidolon_identity_swap_test(
                    step, sampler, latent_size=latent_size)
                results['eidolon_geometry_sweep'] = self.run_eidolon_geometry_sweep_test(
                    step, sampler, latent_size=latent_size)
        finally:
            # Clean up sampler to prevent memory leak
            del sampler
            
            # Restore original weights
            print("Restoring training weights...")
            self.model.load_state_dict(model_backup)
            self.model.train()
            
            # Delete backup and clear cache
            del model_backup
            torch.cuda.empty_cache()
        
        # Save results
        results_file = self.output_dir / f'step{step:07d}' / 'results.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('validation/reconstruction_lpips', results['reconstruction']['mean_lpips'], step)
            if not results.get('text_only', {}).get('skipped'):
                self.tb_writer.add_scalar('validation/text_only_lpips', results['text_only']['mean_lpips'], step)
        
        # Print summary
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"  Adapter: {adapter_name}")
        if self.self_guidance:
            print(f"  (self-guidance mode, scale={self.guidance_scale})")
        if latent_size != base_size:
            print(f"  (resolution: {scale:.2f}x, {latent_size*8}px)")
        print(f"{'='*60}")
        print(f"Reconstruction LPIPS: {results['reconstruction']['mean_lpips']:.4f} ({results['reconstruction']['num_samples']} samples)")
        
        if adapter_name == "stratum":
            if results.get('text_only', {}).get('skipped'):
                print(f"Text-only LPIPS: SKIPPED")
            else:
                print(f"Text-only LPIPS: {results['text_only']['mean_lpips']:.4f} ({results['text_only']['num_samples']} samples)")
            if results.get('divergence', {}).get('skipped'):
                print(f"CFG Divergence: SKIPPED")
            else:
                print(f"CFG Divergence: {results['divergence']['num_samples']} samples")
            if results.get('dino_swap', {}).get('skipped'):
                print(f"DINO Swap: SKIPPED")
            else:
                print(f"DINO Swap: {results['dino_swap']['num_pairs']} pairs")
            if results.get('text_manip', {}).get('skipped'):
                print(f"Text Manip: SKIPPED")
            else:
                print(f"Text Manip: {results['text_manip']['num_successful']}/{results['text_manip']['num_cases']} cases, mean LPIPS diff: {results['text_manip']['mean_lpips_difference']:.4f}")
        
        elif adapter_name == "eidolon":
            if results.get('eidolon_identity_swap', {}).get('skipped'):
                print(f"Identity Swap: SKIPPED")
            else:
                print(f"Identity Swap: {results['eidolon_identity_swap']['num_pairs']} pairs")
            if results.get('eidolon_geometry_sweep', {}).get('skipped'):
                print(f"Geometry Sweep: SKIPPED")
            else:
                print(f"Geometry Sweep: {results['eidolon_geometry_sweep']['num_samples']} samples, dim={results['eidolon_geometry_sweep']['sweep_dim']}")
        
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}\n")
        
        return results


def create_validation_fn(
    shard_dir,
    output_dir='validation',
    tensorboard_writer=None,
    text_scale=3.0,
    dino_scale=2.0,
    num_steps=50,
    self_guidance=False,
    guidance_scale=3.0,
    prediction_type="v_prediction",
    source="webdataset",
    stratum_dir="/workspace/stratum",
    adapter_name="stratum",
):
    """Create validation function for training loop.
    
    IMPORTANT: This creates its own deterministic dataloader internally,
    separate from the training dataloader. This ensures:
    - Sample indices are stable across validation runs
    - No shuffle (same idx always means same image)
    - Finite iteration (no repeat)
    
    Args:
        shard_dir: Path to validation shards
        output_dir: Output directory for validation results
        tensorboard_writer: Optional TensorBoard SummaryWriter
        self_guidance: Use self-guidance instead of dual CFG
        guidance_scale: Self-guidance scale
        prediction_type: "v_prediction" or "x_prediction"
    
    Returns:
        validation_fn(model, ema, step, device)
    """
    from .data import get_deterministic_validation_dataloader
    
    pixel_space = prediction_type == "x_prediction"
    runner = None
    val_dataloader = None

    def validation_fn(model, ema, step, device):
        nonlocal runner, val_dataloader

        # Create deterministic validation dataloader (first call only)
        # Always load at full resolution
        if val_dataloader is None:
            if source == "stratum":
                print(f"Creating deterministic validation dataloader from stratum_dir: {stratum_dir}...")
            else:
                print(f"Creating deterministic validation dataloader from shard_dir: {shard_dir}...")
            val_dataloader = get_deterministic_validation_dataloader(
                shard_dir=shard_dir,
                batch_size=1,  # Process one at a time for validation
                target_latent_size=getattr(model, 'input_size', 128),
                source=source,
                stratum_dir=stratum_dir,
                adapter_name=adapter_name,
            )
        
        if runner is None:
            runner = ValidationRunner(
                model, ema, val_dataloader, device, output_dir,
                tensorboard_writer=tensorboard_writer,
                text_scale=text_scale,
                dino_scale=dino_scale,
                num_steps=num_steps,
                self_guidance=self_guidance,
                guidance_scale=guidance_scale,
                prediction_type=prediction_type,
            )
        
        runner.run_validation(step)
    
    return validation_fn


if __name__ == "__main__":
    print("Validation test configuration:")
    print(f"Reconstruction: {len(RECONSTRUCTION_TEST_INDICES)} samples")
    print(f"DINO swap: {len(DINO_SWAP_TEST_PAIRS)} pairs")
    print(f"Text manipulation: {len(TEXT_MANIP_TEST_INDICES)} indices")
