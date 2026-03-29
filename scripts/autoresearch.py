#!/usr/bin/env python3
"""Autoresearch: autonomous experiment loop for DiT training.

Inspired by karpathy/autoresearch. The loop:
1. Apply proposed config changes
2. Auto-tune throughput for this config (~2 min)
3. Train for a fixed time budget
4. Evaluate validation loss
5. Keep or discard based on val_loss improvement

Usage (called by an AI agent via program.md):
    python scripts/autoresearch.py run \
        --base-config production/config_turbo.yaml \
        --changes "training.maskdit.mask_ratio=0.8" \
        --description "increase mask ratio from 0.75 to 0.8" \
        --time-budget 15

    python scripts/autoresearch.py baseline \
        --base-config production/config_turbo.yaml \
        --time-budget 15

    python scripts/autoresearch.py results
"""

import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


RESULTS_FILE = "results.tsv"
TSV_HEADER = "commit\tval_loss\tsamp_per_sec\tpeak_vram_gb\tstatus\tdescription\n"


def apply_nested_change(d, key_path, value):
    """Apply a dotted key path to a nested dict. e.g. 'training.maskdit.mask_ratio' = 0.8"""
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Auto-convert types
    if isinstance(value, str):
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
    d[keys[-1]] = value


def parse_changes(changes_str):
    """Parse 'key=value key2=value2' into a list of (key, value) tuples."""
    if not changes_str:
        return []
    pairs = []
    for part in changes_str.split():
        if '=' not in part:
            continue
        key, val = part.split('=', 1)
        pairs.append((key, val))
    return pairs


def make_experiment_config(base_config_path, changes, time_budget, output_path):
    """Create an experiment config with changes applied + time budget set."""
    with open(base_config_path) as f:
        raw = yaml.safe_load(f)

    # Apply user changes
    for key, val in changes:
        apply_nested_change(raw, key, val)

    # Set time budget
    raw.setdefault('training', {})['time_budget_minutes'] = time_budget

    # Disable checkpointing and validation for speed
    raw.setdefault('checkpoint', {})['save_every'] = 999999
    raw.setdefault('validation', {})['enabled'] = False
    raw.setdefault('validation', {})['visual_debug_interval'] = 0

    with open(output_path, 'w') as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    return output_path


def auto_tune_config(config_path, device='cuda:0'):
    """Run quick auto-tune and apply results to config."""
    import torch
    from production.config_loader import load_config
    from scripts.auto_tune import quick_tune, apply_tune_results

    config = load_config(config_path)
    dev = torch.device(device)
    torch.cuda.set_device(dev)

    print("  Auto-tuning throughput...", flush=True)
    results = quick_tune(config, dev)

    for scale, r in sorted(results.items()):
        res = int(config.model.input_size * scale)
        print(f"    {res}×{res}: bs={r['batch_size']}, ga={r['grad_accumulation_steps']}, "
              f"{r['samples_per_sec']:.1f} samp/s")

    apply_tune_results(config_path, results, config_path)
    # Free GPU memory used by tuner
    torch.cuda.empty_cache()
    import gc; gc.collect()


def run_training(config_path, device='cuda:0'):
    """Run training and capture results. Returns (val_loss, peak_vram_gb, samp_per_sec)."""
    gpu_idx = device.split(':')[1] if ':' in device else '0'

    cmd = [
        sys.executable, '-m', 'production.train_production',
        '--config', str(config_path),
        '--gpu', gpu_idx,
    ]

    env = os.environ.copy()
    env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    print(f"  Training with config: {config_path}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(Path(__file__).parent.parent))

    output = result.stdout + result.stderr

    # Parse final loss from training output
    val_loss = None
    peak_vram = 0.0
    samp_per_sec = 0.0

    for line in output.split('\n'):
        # Look for the last loss= in tqdm output
        if 'loss' in line and "'" in line:
            try:
                import re
                m = re.search(r"'loss':\s*'([0-9.]+)'", line)
                if m:
                    val_loss = float(m.group(1))
            except (ValueError, AttributeError):
                pass

    # Get peak VRAM
    try:
        import torch
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    except Exception:
        pass

    if result.returncode != 0:
        print(f"  CRASH: {result.stderr[-500:]}", flush=True)
        return None, peak_vram, samp_per_sec

    return val_loss, peak_vram, samp_per_sec


def evaluate_val_loss_standalone(config_path, device='cuda:0'):
    """Load the model from training and compute validation loss."""
    import torch
    from production.config_loader import load_config
    from production.model import NanoDiT
    from production.train import evaluate_val_loss
    from production.data import create_dataloader

    config = load_config(config_path)
    dev = torch.device(device)

    # Find the latest checkpoint
    ckpt_dir = Path(config.checkpoint.output_dir)
    ckpt_path = ckpt_dir / 'checkpoint_final.pt'
    if not ckpt_path.exists():
        # Try to find any checkpoint
        ckpts = sorted(ckpt_dir.glob('checkpoint_*.pt'))
        if ckpts:
            ckpt_path = ckpts[-1]
        else:
            print("  No checkpoint found — cannot evaluate")
            return None

    # Load model
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    mc = config.model
    tc = config.training

    model = NanoDiT(
        input_size=mc.input_size, patch_size=mc.patch_size, in_channels=mc.in_channels,
        hidden_size=mc.hidden_size, depth=mc.depth, num_heads=mc.num_heads,
        mlp_ratio=mc.mlp_ratio,
        repa_block_idx=tc.repa.block_index if tc.repa.enabled else None,
        tread_route_start=tc.tread.route_start if tc.tread.enabled else None,
        tread_route_end=tc.tread.route_end if tc.tread.enabled else None,
        tread_routing_prob=tc.tread.routing_probability if tc.tread.enabled else 0.0,
        bottleneck_size=mc.bottleneck_size,
        num_pose_joints=mc.num_pose_joints,
        pose_confidence_threshold=mc.pose_confidence_threshold,
        maskdit_enabled=tc.maskdit.enabled,
        maskdit_mask_ratio=tc.maskdit.mask_ratio,
        maskdit_decoder_depth=tc.maskdit.decoder_depth,
    ).to(dev)

    # Load weights (try EMA first, fall back to model)
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'], strict=False)
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Create validation dataloader
    from production.data import BucketAwareDataLoader
    dataloader = BucketAwareDataLoader(
        shard_base_dir=config.data.shard_base_dir,
        bucket_configs=config.data.buckets,
        batch_size=1,
        horizontal_flip_prob=0.0,
    )

    result = evaluate_val_loss(model, dataloader, config, dev, num_batches=50)
    print(f"  Val loss: {result['val_loss']:.6f} ({result['num_samples']} samples)")
    return result['val_loss']


def git_short_hash():
    """Get current git short hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def log_result(commit, val_loss, samp_per_sec, peak_vram, status, description):
    """Append a result to results.tsv."""
    results_path = Path(RESULTS_FILE)
    if not results_path.exists():
        results_path.write_text(TSV_HEADER)

    val_str = f"{val_loss:.6f}" if val_loss is not None else "0.000000"
    vram_str = f"{peak_vram:.1f}" if peak_vram else "0.0"
    samp_str = f"{samp_per_sec:.1f}" if samp_per_sec else "0.0"

    with open(results_path, 'a') as f:
        f.write(f"{commit}\t{val_str}\t{samp_str}\t{vram_str}\t{status}\t{description}\n")

    print(f"\n  Logged: {status} | val_loss={val_str} | {description}")


def read_best_val_loss():
    """Read the best val_loss from results.tsv (among 'keep' entries)."""
    results_path = Path(RESULTS_FILE)
    if not results_path.exists():
        return None
    best = None
    for line in results_path.read_text().strip().split('\n')[1:]:  # skip header
        parts = line.split('\t')
        if len(parts) >= 4 and parts[3] == 'keep':
            try:
                val = float(parts[1])
                if val > 0 and (best is None or val < best):
                    best = val
            except ValueError:
                pass
    return best


def cmd_run(args):
    """Run a single experiment."""
    changes = parse_changes(args.changes)
    if not changes:
        print("ERROR: --changes is required. Format: 'key=value key2=value2'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {args.description or 'unnamed'}")
    print(f"  Changes: {args.changes}")
    print(f"  Time budget: {args.time_budget} min")
    print(f"{'='*60}\n")

    # Create experiment config
    exp_config = Path(tempfile.mktemp(suffix='.yaml', prefix='autoresearch_'))
    try:
        make_experiment_config(args.base_config, changes, args.time_budget, exp_config)

        # Auto-tune throughput
        auto_tune_config(str(exp_config), device=f'cuda:{args.gpu}')

        # Git commit the config change for tracking
        # (We don't actually commit — the agent handles git)

        # Run training
        val_loss, peak_vram, samp_per_sec = run_training(str(exp_config), device=f'cuda:{args.gpu}')

        # Evaluate validation loss
        if val_loss is None:
            val_loss = evaluate_val_loss_standalone(str(exp_config), device=f'cuda:{args.gpu}')

        # Decide keep/discard
        if val_loss is None:
            status = 'crash'
        else:
            best = read_best_val_loss()
            if best is None or val_loss < best:
                status = 'keep'
            else:
                status = 'discard'

        # Log
        commit = git_short_hash()
        desc = args.description or args.changes
        log_result(commit, val_loss, samp_per_sec, peak_vram, status, desc)

        # Print structured output for agent parsing
        print(f"\n---")
        print(f"val_loss: {val_loss:.6f}" if val_loss else "val_loss: CRASH")
        print(f"peak_vram_gb: {peak_vram:.1f}")
        print(f"status: {status}")
        print(f"description: {desc}")
        print(f"---")

    finally:
        if exp_config.exists():
            exp_config.unlink()


def cmd_baseline(args):
    """Run baseline (no changes) to establish reference val_loss."""
    print(f"\n{'='*60}")
    print(f"  BASELINE RUN")
    print(f"  Config: {args.base_config}")
    print(f"  Time budget: {args.time_budget} min")
    print(f"{'='*60}\n")

    exp_config = Path(tempfile.mktemp(suffix='.yaml', prefix='autoresearch_'))
    try:
        make_experiment_config(args.base_config, [], args.time_budget, exp_config)
        auto_tune_config(str(exp_config), device=f'cuda:{args.gpu}')
        val_loss, peak_vram, samp_per_sec = run_training(str(exp_config), device=f'cuda:{args.gpu}')
        if val_loss is None:
            val_loss = evaluate_val_loss_standalone(str(exp_config), device=f'cuda:{args.gpu}')

        commit = git_short_hash()
        status = 'keep' if val_loss is not None else 'crash'
        log_result(commit, val_loss, samp_per_sec, peak_vram, status, 'baseline')

        print(f"\n---")
        print(f"val_loss: {val_loss:.6f}" if val_loss else "val_loss: CRASH")
        print(f"peak_vram_gb: {peak_vram:.1f}")
        print(f"status: {status}")
        print(f"description: baseline")
        print(f"---")

    finally:
        if exp_config.exists():
            exp_config.unlink()


def cmd_results(args):
    """Print results.tsv in a readable format."""
    results_path = Path(RESULTS_FILE)
    if not results_path.exists():
        print("No results yet. Run 'baseline' first.")
        return

    print(results_path.read_text())
    best = read_best_val_loss()
    if best is not None:
        print(f"\nBest val_loss: {best:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Autoresearch: autonomous DiT experiment loop')
    sub = parser.add_subparsers(dest='command')

    # run
    p_run = sub.add_parser('run', help='Run an experiment with config changes')
    p_run.add_argument('--base-config', type=str, required=True)
    p_run.add_argument('--changes', type=str, required=True,
                       help='Config changes: "key=value key2=value2"')
    p_run.add_argument('--description', type=str, default='')
    p_run.add_argument('--time-budget', type=float, default=15,
                       help='Training time budget in minutes (default: 15)')
    p_run.add_argument('--gpu', type=int, default=0)

    # baseline
    p_base = sub.add_parser('baseline', help='Run baseline (no changes)')
    p_base.add_argument('--base-config', type=str, required=True)
    p_base.add_argument('--time-budget', type=float, default=15)
    p_base.add_argument('--gpu', type=int, default=0)

    # results
    sub.add_parser('results', help='Show experiment results')

    args = parser.parse_args()
    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'baseline':
        cmd_baseline(args)
    elif args.command == 'results':
        cmd_results(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
