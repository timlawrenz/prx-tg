#!/usr/bin/env python3
"""Auto-tuner: find optimal batch_size & grad_accumulation_steps per resolution phase.

Uses a Micro Genetic Algorithm to maximize measured samples/second at each resolution,
respecting GPU memory limits. Bigger batch != faster — the GA finds the true throughput
peak by measuring actual it/s, not by maximizing batch size.

Usage:
    python scripts/auto_tune.py --config production/config_turbo.yaml
    python scripts/auto_tune.py --config production/config_turbo.yaml --generations 12
    python scripts/auto_tune.py --config production/config_turbo.yaml --output tuned_config.yaml
"""

import argparse
import copy
import gc
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml


# ---------------------------------------------------------------------------
# Tuning candidate
# ---------------------------------------------------------------------------

@dataclass
class TuningCandidate:
    """One point in the search space."""
    batch_size: int = 1
    grad_accumulation_steps: int = 32
    gradient_checkpointing: bool = False

    def genes(self) -> list:
        return [self.batch_size, self.grad_accumulation_steps, self.gradient_checkpointing]

    @staticmethod
    def from_genes(genes: list) -> "TuningCandidate":
        return TuningCandidate(
            batch_size=genes[0],
            grad_accumulation_steps=genes[1],
            gradient_checkpointing=genes[2],
        )


@dataclass
class BenchmarkResult:
    candidate: TuningCandidate
    samples_per_sec: float = 0.0
    iter_per_sec: float = 0.0
    peak_vram_gb: float = 0.0
    oom: bool = False


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
GRAD_ACCUM = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CHECKPOINTING = [True, False]


def random_candidate(max_batch: int = 64) -> TuningCandidate:
    bs = random.choice([b for b in BATCH_SIZES if b <= max_batch])
    ga = random.choice(GRAD_ACCUM)
    gc_ = random.choice(CHECKPOINTING)
    return TuningCandidate(batch_size=bs, grad_accumulation_steps=ga, gradient_checkpointing=gc_)


def mutate(c: TuningCandidate, max_batch: int = 64) -> TuningCandidate:
    genes = c.genes()
    idx = random.randint(0, 2)
    if idx == 0:
        pool = [b for b in BATCH_SIZES if b <= max_batch]
        genes[0] = random.choice(pool)
    elif idx == 1:
        genes[1] = random.choice(GRAD_ACCUM)
    else:
        genes[2] = not genes[2]
    return TuningCandidate.from_genes(genes)


def crossover(a: TuningCandidate, b: TuningCandidate) -> TuningCandidate:
    ga, gb = a.genes(), b.genes()
    child = [ga[i] if random.random() < 0.5 else gb[i] for i in range(3)]
    return TuningCandidate.from_genes(child)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs short training bursts and measures throughput with synthetic data."""

    def __init__(self, config, device: torch.device, warmup_steps: int = 8, timed_steps: int = 16):
        self.config = config
        self.device = device
        self.warmup_steps = warmup_steps
        self.timed_steps = timed_steps

        # Build model once (reused across all benchmarks)
        self.model = self._build_model()
        self.model.train()

        # Determine AMP settings
        precision = getattr(config.training, 'precision', 'float32')
        self.use_amp = config.training.mixed_precision and precision != 'float32'
        self.amp_dtype = torch.float16 if precision == 'float16' else torch.bfloat16

    def _build_model(self):
        from production.model import NanoDiT
        mc = self.config.model
        tc = self.config.training

        tread_route_start, tread_route_end, tread_routing_prob = None, None, 0.0
        if tc.tread.enabled:
            tread_route_start = tc.tread.route_start
            tread_route_end = tc.tread.route_end
            tread_routing_prob = tc.tread.routing_probability

        repa_block_idx = tc.repa.block_index if tc.repa.enabled else None

        maskdit_enabled = tc.maskdit.enabled
        maskdit_mask_ratio = tc.maskdit.mask_ratio
        maskdit_decoder_depth = tc.maskdit.decoder_depth

        return NanoDiT(
            input_size=mc.input_size,
            patch_size=mc.patch_size,
            in_channels=mc.in_channels,
            hidden_size=mc.hidden_size,
            depth=mc.depth,
            num_heads=mc.num_heads,
            mlp_ratio=mc.mlp_ratio,
            use_gradient_checkpointing=False,  # overridden per candidate
            repa_block_idx=repa_block_idx,
            tread_route_start=tread_route_start,
            tread_route_end=tread_route_end,
            tread_routing_prob=tread_routing_prob,
            bottleneck_size=mc.bottleneck_size,
            num_pose_joints=mc.num_pose_joints,
            pose_confidence_threshold=mc.pose_confidence_threshold,
            maskdit_enabled=maskdit_enabled,
            maskdit_mask_ratio=maskdit_mask_ratio,
            maskdit_decoder_depth=maskdit_decoder_depth,
        ).to(self.device)

    def _make_synthetic_batch(self, batch_size: int, h_px: int, w_px: int) -> dict:
        """Create a synthetic batch matching the model's expected inputs."""
        mc = self.config.model
        ps = mc.patch_size
        C = mc.in_channels  # pixel-space: 3, latent-space: 4

        return {
            'image_data': torch.randn(batch_size, C, h_px, w_px, device=self.device),
            'dino_embedding': torch.randn(batch_size, 1024, device=self.device),
            'dinov3_patches': torch.randn(batch_size, 256, 1024, device=self.device),
            't5_hidden': torch.randn(batch_size, 77, 2048, device=self.device),
            't5_mask': torch.ones(batch_size, 77, dtype=torch.bool, device=self.device),
            'pose_keypoints': torch.randn(batch_size, mc.num_pose_joints, 3, device=self.device),
            'bucket': 'synthetic',
        }

    def _run_micro_step(self, batch: dict, optimizer, grad_scaler) -> float:
        """Run one forward + backward pass. Returns loss scalar."""
        from production.train import flow_matching_loss

        tc = self.config.training
        maskdit_cfg = tc.maskdit if tc.maskdit.enabled else None
        tread_enabled = tc.tread.enabled

        ctx = torch.amp.autocast('cuda', enabled=self.use_amp,
                                 dtype=self.amp_dtype if self.use_amp else torch.float32)

        with ctx:
            result = flow_matching_loss(
                self.model,
                batch['image_data'],
                batch['dino_embedding'],
                batch['dinov3_patches'],
                batch['t5_hidden'],
                batch['t5_mask'],
                cfg_probs={'p_uncond': 0.0, 'p_text_only': 0.0, 'p_dino_only': 0.0},
                return_v_pred=False,
                tread_enabled=tread_enabled,
                maskdit_config=maskdit_cfg,
            )
            # flow_matching_loss returns scalar when return_v_pred=False
            loss = result

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item()

    def benchmark(self, candidate: TuningCandidate, h_px: int, w_px: int) -> BenchmarkResult:
        """Benchmark a single candidate at a given resolution. Returns BenchmarkResult."""
        result = BenchmarkResult(candidate=candidate)

        # Apply gradient checkpointing setting
        if hasattr(self.model, 'use_gradient_checkpointing'):
            self.model.use_gradient_checkpointing = candidate.gradient_checkpointing

        # Create optimizer (lightweight, recreated per benchmark)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        grad_scaler = torch.amp.GradScaler('cuda') if (self.use_amp and self.amp_dtype == torch.float16) else None

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        gc.collect()

        try:
            batch = self._make_synthetic_batch(candidate.batch_size, h_px, w_px)

            # Warmup (not timed — lets CUDA JIT compile kernels)
            for _ in range(self.warmup_steps):
                optimizer.zero_grad(set_to_none=True)
                self._run_micro_step(batch, optimizer, grad_scaler)
                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

            torch.cuda.synchronize()

            # Timed run
            start = time.perf_counter()
            for _ in range(self.timed_steps):
                optimizer.zero_grad(set_to_none=True)
                self._run_micro_step(batch, optimizer, grad_scaler)
                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            result.iter_per_sec = self.timed_steps / elapsed
            result.samples_per_sec = (self.timed_steps * candidate.batch_size) / elapsed
            result.peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                result.oom = True
                result.samples_per_sec = 0.0
                result.iter_per_sec = 0.0
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise
        finally:
            del optimizer
            if grad_scaler is not None:
                del grad_scaler
            torch.cuda.empty_cache()

        return result


# ---------------------------------------------------------------------------
# Micro-GA
# ---------------------------------------------------------------------------

class MicroGA:
    """Small-population genetic algorithm for throughput optimization."""

    def __init__(
        self,
        runner: BenchmarkRunner,
        h_px: int,
        w_px: int,
        population_size: int = 12,
        generations: int = 8,
        elitism: int = 2,
        mutation_rate: float = 0.20,
        tournament_k: int = 3,
        max_batch: int = 64,
    ):
        self.runner = runner
        self.h_px = h_px
        self.w_px = w_px
        self.pop_size = population_size
        self.generations = generations
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.max_batch = max_batch

    def _tournament_select(self, pop: List[BenchmarkResult]) -> TuningCandidate:
        pool = random.sample(pop, min(self.tournament_k, len(pop)))
        best = max(pool, key=lambda r: r.samples_per_sec)
        return copy.deepcopy(best.candidate)

    def run(self, seeds: Optional[List[TuningCandidate]] = None) -> List[BenchmarkResult]:
        """Run the GA. Returns all evaluated results sorted by fitness (best first)."""
        all_results: List[BenchmarkResult] = []

        # Initial population
        population: List[TuningCandidate] = []
        if seeds:
            population.extend(seeds[:self.pop_size])
        while len(population) < self.pop_size:
            population.append(random_candidate(self.max_batch))

        for gen in range(self.generations):
            # Evaluate
            gen_results = []
            for i, cand in enumerate(population):
                res = self.runner.benchmark(cand, self.h_px, self.w_px)
                gen_results.append(res)
                all_results.append(res)
                status = "OOM" if res.oom else f"{res.samples_per_sec:.1f} samp/s, {res.peak_vram_gb:.2f}GB"
                print(f"    gen {gen+1}/{self.generations} | #{i+1:2d} "
                      f"bs={cand.batch_size:<3d} ga={cand.grad_accumulation_steps:<4d} "
                      f"ckpt={'Y' if cand.gradient_checkpointing else 'N'} → {status}")

            # Sort by fitness
            gen_results.sort(key=lambda r: r.samples_per_sec, reverse=True)
            best = gen_results[0]
            print(f"    ── gen {gen+1} best: bs={best.candidate.batch_size} ga={best.candidate.grad_accumulation_steps} "
                  f"ckpt={'Y' if best.candidate.gradient_checkpointing else 'N'} "
                  f"→ {best.samples_per_sec:.1f} samp/s")

            if gen == self.generations - 1:
                break

            # Elitism: carry top individuals forward
            next_gen = [copy.deepcopy(r.candidate) for r in gen_results[:self.elitism]]

            # Fill rest with crossover + mutation
            while len(next_gen) < self.pop_size:
                p1 = self._tournament_select(gen_results)
                p2 = self._tournament_select(gen_results)
                child = crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = mutate(child, self.max_batch)
                # Clamp batch_size to feasible range
                child.batch_size = min(child.batch_size, self.max_batch)
                next_gen.append(child)

            population = next_gen

        all_results.sort(key=lambda r: r.samples_per_sec, reverse=True)
        return all_results


# ---------------------------------------------------------------------------
# OOM boundary finder (binary search)
# ---------------------------------------------------------------------------

def find_max_batch(runner: BenchmarkRunner, h_px: int, w_px: int) -> int:
    """Binary search for the largest batch_size that doesn't OOM."""
    lo, hi = 0, len(BATCH_SIZES) - 1
    max_ok = BATCH_SIZES[0]

    while lo <= hi:
        mid = (lo + hi) // 2
        bs = BATCH_SIZES[mid]
        cand = TuningCandidate(batch_size=bs, grad_accumulation_steps=1, gradient_checkpointing=False)
        res = runner.benchmark(cand, h_px, w_px)
        if res.oom:
            hi = mid - 1
        else:
            max_ok = bs
            lo = mid + 1
        print(f"  OOM probe bs={bs:<3d}: {'OOM' if res.oom else f'OK ({res.peak_vram_gb:.2f}GB)'}")

    # Also check with gradient checkpointing — may unlock larger batches
    for bs in BATCH_SIZES:
        if bs <= max_ok:
            continue
        cand = TuningCandidate(batch_size=bs, grad_accumulation_steps=1, gradient_checkpointing=True)
        res = runner.benchmark(cand, h_px, w_px)
        if res.oom:
            break
        max_ok = bs
        print(f"  OOM probe bs={bs:<3d} +ckpt: OK ({res.peak_vram_gb:.2f}GB)")

    return max_ok


# ---------------------------------------------------------------------------
# Resolution → pixel size helper
# ---------------------------------------------------------------------------

def scale_to_pixel_size(scale: float, base_size: int = 1024, patch_size: int = 16) -> Tuple[int, int]:
    """Convert a resolution scale factor to aligned pixel dimensions."""
    align = 32  # pixel-space alignment
    size = max(align, (int(base_size * scale) // align) * align)
    return (size, size)


# ---------------------------------------------------------------------------
# Auto-tuner orchestrator
# ---------------------------------------------------------------------------

class AutoTuner:
    """Runs per-resolution throughput optimization."""

    def __init__(self, config, device: torch.device, pop_size: int = 12,
                 generations: int = 8, warmup_steps: int = 8, timed_steps: int = 16):
        self.config = config
        self.device = device
        self.pop_size = pop_size
        self.generations = generations
        self.runner = BenchmarkRunner(config, device, warmup_steps, timed_steps)

    def tune_phase(self, scale: float) -> BenchmarkResult:
        """Find optimal config for a single resolution phase."""
        h, w = scale_to_pixel_size(scale, self.config.model.input_size,
                                   self.config.model.patch_size)
        print(f"\n{'='*60}")
        print(f"  Resolution: {h}×{w} (scale={scale})")
        print(f"{'='*60}")

        # Phase 1: find OOM boundary
        print(f"\n  Phase 1: Finding OOM boundary...")
        max_batch = find_max_batch(self.runner, h, w)
        print(f"  Max feasible batch_size: {max_batch}")

        # Phase 2: GA search
        print(f"\n  Phase 2: Micro-GA search (pop={self.pop_size}, gen={self.generations})...")

        # Seed population with heuristic points spread across the range
        seeds = []
        # Current config as a baseline
        seeds.append(TuningCandidate(
            batch_size=self.config.training.batch_size,
            grad_accumulation_steps=self.config.training.grad_accumulation_steps,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
        ))
        # Max batch with minimal accumulation
        seeds.append(TuningCandidate(batch_size=max_batch, grad_accumulation_steps=1))
        # Half max batch
        half_max = max(1, max_batch // 2)
        seeds.append(TuningCandidate(batch_size=half_max, grad_accumulation_steps=2))
        # Small batch, high accumulation
        seeds.append(TuningCandidate(batch_size=1, grad_accumulation_steps=64))
        # With checkpointing
        seeds.append(TuningCandidate(batch_size=max_batch, grad_accumulation_steps=1, gradient_checkpointing=True))

        ga = MicroGA(
            runner=self.runner,
            h_px=h, w_px=w,
            population_size=self.pop_size,
            generations=self.generations,
            max_batch=max_batch,
        )

        results = ga.run(seeds=seeds)
        return results[0]  # best

    def tune_all(self) -> dict:
        """Tune all resolution phases. Returns {scale: BenchmarkResult}."""
        phases = self.config.training.get_resolution_phases()
        if not phases:
            print("No resolution schedule defined — tuning at full scale only.")
            phases = [type('Phase', (), {'scale': 1.0, 'until_step': 99999})]

        results = {}
        for phase in phases:
            best = self.tune_phase(phase.scale)
            results[phase.scale] = best

        return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results: dict, config):
    """Print a summary table of tuning results."""
    phases = config.training.get_resolution_phases()

    print(f"\n{'='*70}")
    print(f"  AUTO-TUNE RESULTS")
    print(f"{'='*70}")
    print(f"  {'Scale':<8} {'Resolution':<12} {'batch':<7} {'accum':<7} {'ckpt':<6} {'samp/s':<10} {'VRAM GB'}")
    print(f"  {'-'*8} {'-'*12} {'-'*7} {'-'*7} {'-'*6} {'-'*10} {'-'*7}")

    for phase in phases:
        r = results.get(phase.scale)
        if r is None:
            continue
        h, w = scale_to_pixel_size(phase.scale, config.model.input_size, config.model.patch_size)
        c = r.candidate
        print(f"  {phase.scale:<8.3f} {h}×{w:<8} {c.batch_size:<7d} "
              f"{c.grad_accumulation_steps:<7d} {'Y' if c.gradient_checkpointing else 'N':<6} "
              f"{r.samples_per_sec:<10.1f} {r.peak_vram_gb:.2f}")

    # Comparison with original config
    orig_bs = config.training.batch_size
    orig_ga = config.training.grad_accumulation_steps
    print(f"\n  Original config: batch_size={orig_bs}, grad_accumulation={orig_ga} (all phases)")
    print(f"{'='*70}")


def generate_yaml_snippet(results: dict, config) -> str:
    """Generate a resolution_schedule YAML snippet with per-phase overrides."""
    phases = config.training.get_resolution_phases()
    lines = ["resolution_schedule:"]
    for phase in phases:
        r = results.get(phase.scale)
        lines.append(f"  - until_step: {phase.until_step}")
        lines.append(f"    scale: {phase.scale}")
        if r and not r.oom:
            c = r.candidate
            lines.append(f"    batch_size: {c.batch_size}")
            lines.append(f"    grad_accumulation_steps: {c.grad_accumulation_steps}")
    return "\n".join(lines)


def write_tuned_config(results: dict, config, config_path: str, output_path: str):
    """Write a complete config with tuned per-phase overrides."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    phases = config.training.get_resolution_phases()
    tuned_schedule = []
    for phase in phases:
        entry = {'until_step': phase.until_step, 'scale': phase.scale}
        r = results.get(phase.scale)
        if r and not r.oom:
            c = r.candidate
            entry['batch_size'] = c.batch_size
            entry['grad_accumulation_steps'] = c.grad_accumulation_steps
        tuned_schedule.append(entry)

    raw.setdefault('training', {})['resolution_schedule'] = tuned_schedule

    with open(output_path, 'w') as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Tuned config written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Auto-tune batch_size & grad_accumulation_steps per resolution phase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda', help='Device (default: cuda)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--population', type=int, default=12, help='GA population size')
    parser.add_argument('--generations', type=int, default=8, help='GA generations')
    parser.add_argument('--warmup-steps', type=int, default=8, help='Warmup micro-steps per evaluation')
    parser.add_argument('--timed-steps', type=int, default=16, help='Timed micro-steps per evaluation')
    parser.add_argument('--output', type=str, default=None, help='Write tuned config to this path')
    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda':
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("WARNING: Running on CPU — results won't reflect GPU throughput")

    # Load config
    from production.config_loader import load_config
    config = load_config(args.config)

    print(f"\nModel: {config.model.hidden_size}h / {config.model.depth}L "
          f"({sum(p.numel() for p in AutoTuner(config, device, 1, 1).runner.model.parameters()) / 1e6:.1f}M params)")
    print(f"Precision: {config.training.precision}")
    print(f"MaskDiT: {'ON' if config.training.maskdit.enabled else 'OFF'}")
    print(f"TREAD: {'ON' if config.training.tread.enabled else 'OFF'}")

    # Run tuner
    tuner = AutoTuner(
        config, device,
        pop_size=args.population,
        generations=args.generations,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
    )
    results = tuner.tune_all()

    # Output
    print_results(results, config)
    print(f"\nYAML snippet:\n")
    print(generate_yaml_snippet(results, config))

    if args.output:
        write_tuned_config(results, config, args.config, args.output)


if __name__ == '__main__':
    main()
