"""Validation tests for Nano DiT."""

import json
import torch
from pathlib import Path
from tqdm import tqdm
import lpips

from .sample import ValidationSampler, load_vae_decoder, save_images


# Fixed test sample indices for consistency across validation runs
# These will be selected from the validation dataset
RECONSTRUCTION_TEST_INDICES = list(range(100))  # All 100 samples

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
    ):
        """
        Args:
            model: NanoDiT model (training weights)
            ema: EMA model
            dataloader: validation dataloader
            device: torch device
            output_dir: directory for validation outputs
            lpips_net: LPIPS network ('alex' or 'vgg')
        """
        self.model = model
        self.ema = ema
        self.dataloader = dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Load VAE decoder
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
            batch_size = batch['vae_latent'].shape[0]
            
            for i in range(batch_size):
                if sample_count <= max_idx:
                    samples.append({
                        'vae_latent': batch['vae_latent'][i],
                        'dino_embedding': batch['dino_embedding'][i],
                        't5_hidden': batch['t5_hidden'][i],
                        't5_mask': batch['t5_mask'][i],
                        'caption': batch['captions'][i],
                        'image_id': batch['image_ids'][i],
                    })
                    sample_count += 1
        
        self.validation_samples = samples
        print(f"Loaded {len(samples)} validation samples")
        return samples
    
    def run_reconstruction_test(self, step, sampler):
        """Test 1: Generate from original caption + DINO, measure LPIPS.
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
        
        Returns:
            dict with lpips_scores and mean_lpips
        """
        print(f"Running reconstruction test (step {step})...")
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'reconstruction'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        lpips_scores = []
        
        # Generate in batches for efficiency (smaller batch for validation to avoid OOM)
        batch_size = 2
        
        for i in tqdm(range(0, len(RECONSTRUCTION_TEST_INDICES), batch_size), desc='Reconstruction'):
            batch_indices = RECONSTRUCTION_TEST_INDICES[i:i+batch_size]
            
            # Gather batch
            batch_samples = [samples[idx] for idx in batch_indices]
            
            dino_emb = torch.stack([s['dino_embedding'] for s in batch_samples])
            text_emb = torch.stack([s['t5_hidden'] for s in batch_samples])
            text_mask = torch.stack([s['t5_mask'] for s in batch_samples])
            gt_latents = torch.stack([s['vae_latent'] for s in batch_samples])
            image_ids = [s['image_id'] for s in batch_samples]
            
            # Generate images
            gen_images = sampler.generate(dino_emb, text_emb, text_mask)
            
            # Decode ground truth latents for comparison
            gt_images = sampler.vae.decode(gt_latents.half().to(self.device)).sample
            
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
            del dino_emb, text_emb, text_mask, gt_latents, gen_images, gt_images
            torch.cuda.empty_cache()
        
        mean_lpips = sum(lpips_scores) / len(lpips_scores)
        
        return {
            'lpips_scores': lpips_scores,
            'mean_lpips': mean_lpips,
            'num_samples': len(lpips_scores),
        }
    
    def run_dino_swap_test(self, step, sampler):
        """Test 2: Swap DINO embeddings, keep original captions.
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
        
        Returns:
            dict with test results
        """
        print(f"Running DINO swap test (step {step})...")
        
        samples = self.load_validation_samples()
        output_dir = self.output_dir / f'step{step:07d}' / 'dino_swap'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for pair_idx, (idx_a, idx_b) in enumerate(DINO_SWAP_TEST_PAIRS):
            sample_a = samples[idx_a]
            sample_b = samples[idx_b]
            
            # A's caption + B's DINO
            dino_emb = sample_b['dino_embedding'].unsqueeze(0)
            text_emb = sample_a['t5_hidden'].unsqueeze(0)
            text_mask = sample_a['t5_mask'].unsqueeze(0)
            
            gen_images = sampler.generate(dino_emb, text_emb, text_mask)
            
            # Save with descriptive name
            save_images(
                gen_images,
                output_dir,
                prefix=f'pair{pair_idx}_captionA{idx_a}_dinoB{idx_b}',
                image_ids=None
            )
            
            results.append({
                'pair_idx': pair_idx,
                'caption_source': idx_a,
                'dino_source': idx_b,
                'caption': sample_a['caption'],
            })
            
            # Clear GPU memory
            del dino_emb, text_emb, text_mask, gen_images
            torch.cuda.empty_cache()
        
        return {
            'num_pairs': len(DINO_SWAP_TEST_PAIRS),
            'results': results,
        }
    
    def load_t5_encoder(self):
        """Lazy load T5 encoder and tokenizer for text manipulation test."""
        if self.t5_encoder is None:
            print("Loading T5 encoder for text manipulation...")
            from transformers import T5EncoderModel, AutoTokenizer
            
            self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")
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
        """
        encoder, tokenizer = self.load_t5_encoder()
        
        # Tokenize
        inputs = tokenizer(
            caption,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state.float()  # (1, 77, 1024) in float32
        
        return hidden_states, attention_mask
    
    def run_text_manip_test(self, step, sampler):
        """Test 3: Modify caption, keep DINO embedding.
        
        Dynamically finds text to replace from a list of common patterns.
        
        Args:
            step: current training step
            sampler: ValidationSampler instance
        
        Returns:
            dict with test results and LPIPS comparison scores
        """
        print(f"Running text manipulation test (step {step})...")
        
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
            
            dino_emb = sample['dino_embedding'].unsqueeze(0)
            
            # Generate with original caption
            text_emb_orig = sample['t5_hidden'].unsqueeze(0)
            text_mask_orig = sample['t5_mask'].unsqueeze(0)
            gen_orig = sampler.generate(dino_emb, text_emb_orig, text_mask_orig)
            
            # Re-encode modified caption with T5
            text_emb_mod, text_mask_mod = self.encode_caption(modified_caption)
            
            # Generate with modified caption
            gen_mod = sampler.generate(dino_emb, text_emb_mod, text_mask_mod)
            
            # Compute LPIPS between original and modified generations
            lpips_val = self.lpips_fn(gen_orig, gen_mod).item()
            
            # Save both versions
            save_images(
                gen_orig,
                output_dir,
                prefix=f'sample{idx}_original',
                image_ids=None
            )
            save_images(
                gen_mod,
                output_dir,
                prefix=f'sample{idx}_modified',
                image_ids=None
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
            del dino_emb, text_emb_orig, text_mask_orig, text_emb_mod, text_mask_mod
            del gen_orig, gen_mod
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
    
    def run_validation(self, step):
        """Run all validation tests.
        
        Args:
            step: current training step
        
        Returns:
            dict with all validation results
        """
        print(f"\n{'='*60}")
        print(f"VALIDATION AT STEP {step}")
        print(f"{'='*60}\n")
        
        # Copy EMA weights to a temporary model for evaluation
        eval_model = type(self.model)(
            input_size=self.model.input_size,
            patch_size=self.model.patch_size,
            in_channels=self.model.in_channels,
            hidden_size=384,
            depth=12,
            num_heads=self.model.num_heads,
        ).to(self.device)
        
        self.ema.copy_to(eval_model)
        eval_model.eval()
        
        # Create sampler
        sampler = ValidationSampler(
            eval_model,
            self.vae,
            device=self.device,
            num_steps=50,
            text_scale=3.0,
            dino_scale=2.0,
        )
        
        # Run tests
        results = {
            'step': step,
            'reconstruction': self.run_reconstruction_test(step, sampler),
            'dino_swap': self.run_dino_swap_test(step, sampler),
            'text_manip': self.run_text_manip_test(step, sampler),
        }
        
        # Save results
        results_file = self.output_dir / f'step{step:07d}' / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Reconstruction LPIPS: {results['reconstruction']['mean_lpips']:.4f}")
        print(f"DINO swap: {results['dino_swap']['num_pairs']} pairs generated")
        print(f"Text manipulation: {results['text_manip']['num_successful']}/{results['text_manip']['num_cases']} cases, "
              f"mean LPIPS diff: {results['text_manip']['mean_lpips_difference']:.4f}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}\n")
        
        return results


def create_validation_fn(shard_dir, output_dir='validation'):
    """Create validation function for training loop.
    
    IMPORTANT: This creates its own deterministic dataloader internally,
    separate from the training dataloader. This ensures:
    - Sample indices are stable across validation runs
    - No shuffle (same idx always means same image)
    - No augmentation (no flips)
    - Finite iteration (no repeat)
    
    Args:
        shard_dir: Path to validation shards
        output_dir: Output directory for validation results
    
    Returns:
        validation_fn(model, ema, step, device)
    """
    from .data import get_deterministic_validation_dataloader
    
    runner = None
    val_dataloader = None
    
    def validation_fn(model, ema, step, device):
        nonlocal runner, val_dataloader
        
        # Create deterministic validation dataloader (first call only)
        if val_dataloader is None:
            print(f"Creating deterministic validation dataloader from {shard_dir}...")
            val_dataloader = get_deterministic_validation_dataloader(
                shard_dir=shard_dir,
                batch_size=1,  # Process one at a time for validation
            )
        
        if runner is None:
            runner = ValidationRunner(
                model, ema, val_dataloader, device, output_dir
            )
        
        runner.run_validation(step)
    
    return validation_fn


if __name__ == "__main__":
    print("Validation test configuration:")
    print(f"Reconstruction: {len(RECONSTRUCTION_TEST_INDICES)} samples")
    print(f"DINO swap: {len(DINO_SWAP_TEST_PAIRS)} pairs")
    print(f"Text manipulation: {len(TEXT_MANIP_TEST_INDICES)} indices")
