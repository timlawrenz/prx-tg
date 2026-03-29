#!/usr/bin/env python3
"""Autoresearch: multi-phase tournament experiment loop for DiT training.

Inspired by karpathy/autoresearch. Multi-phase tournament:
  Phase A: N experiments from scratch (val_loss metric)
  Phase B: N experiments resuming from A's best (reconstruction_lpips metric)
  Phase C: N experiments resuming from B's best ...

Each experiment:
1. Apply proposed config changes
2. Auto-tune throughput for this config (~2 min)
3. Train for a fixed step budget (deterministic, not time-based)
4. Evaluate metric (val_loss or reconstruction_lpips)
5. Keep or discard based on improvement

Usage (called by an AI agent via program.md):
    # Phase A: explore from scratch
    python scripts/autoresearch.py baseline --phase A \
        --base-config production/config_turbo_2070.yaml --step-budget 1500

    python scripts/autoresearch.py run --phase A \
        --base-config production/config_turbo_2070.yaml \
        --changes "training.optimizer.lr=1e-4" \
        --description "lower LR" --step-budget 1500

    # Find Phase A's best checkpoint
    python scripts/autoresearch.py best-checkpoint --phase A

    # Phase B: exploit from Phase A's best
    python scripts/autoresearch.py run --phase B \
        --base-config production/config_turbo_2070.yaml \
        --resume-from experiments/.../checkpoint_final.pt \
        --changes "training.maskdit.mask_ratio=0.8" \
        --description "higher masking" --step-budget 1500 \
        --metric reconstruction_lpips

    python scripts/autoresearch.py results --phase A
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


RESULTS_DIR = Path("results")
TSV_HEADER = "commit\tval_loss\treconstruction_lpips\ttext_only_lpips\tsamp_per_sec\tpeak_vram_gb\tstatus\tdescription\tcheckpoint_path\n"

VALID_METRICS = ('val_loss', 'reconstruction_lpips', 'text_only_lpips')

# Lower is better for all metrics
METRIC_COLUMNS = {
    'val_loss': 1,
    'reconstruction_lpips': 2,
    'text_only_lpips': 3,
}


def results_file_for_phase(phase):
    """Get the results TSV path for a given phase."""
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / f"results_phase_{phase}.tsv"


def apply_nested_change(d, key_path, value):
    """Apply a dotted key path to a nested dict. e.g. 'training.maskdit.mask_ratio' = 0.8"""
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
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


def make_experiment_config(base_config_path, changes, output_path,
                           step_budget=None, time_budget=None):
    """Create an experiment config with changes applied + budget set.

    When step_budget is set:
      - training.total_steps = step_budget
      - time_budget_minutes = 0 (disabled)
      - checkpoint.save_every and validation.interval_steps sync to step_budget
      - validation.enabled = True (needed for LPIPS metrics in later phases)
    When time_budget is set (legacy):
      - training.time_budget_minutes = time_budget
      - validation disabled for speed
    """
    with open(base_config_path) as f:
        raw = yaml.safe_load(f)

    for key, val in changes:
        apply_nested_change(raw, key, val)

    training = raw.setdefault('training', {})
    checkpoint = raw.setdefault('checkpoint', {})
    validation = raw.setdefault('validation', {})

    if step_budget is not None:
        training['total_steps'] = step_budget
        training['time_budget_minutes'] = 0

        # Sync checkpoint and validation intervals:
        # save at halfway and at end; validate at end
        save_interval = max(step_budget // 2, 1)
        checkpoint['save_every'] = save_interval
        validation['enabled'] = True
        validation['interval_steps'] = step_budget
        validation['num_samples'] = 25
        validation['visual_debug_interval'] = save_interval
        validation['visual_debug_num_samples'] = 4
    elif time_budget is not None:
        training['time_budget_minutes'] = time_budget
        checkpoint['save_every'] = 999999
        validation['enabled'] = False
        validation['visual_debug_interval'] = 0
    else:
        raise ValueError("Either step_budget or time_budget must be set")

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
    torch.cuda.empty_cache()
    import gc; gc.collect()


def run_training(config_path, device='cuda:0', resume_from=None):
    """Run training subprocess. Returns (val_loss, peak_vram_gb, samp_per_sec, experiment_dir)."""
    gpu_idx = device.split(':')[1] if ':' in device else '0'

    cmd = [
        sys.executable, '-m', 'production.train_production',
        '--config', str(config_path),
        '--gpu', gpu_idx,
    ]
    if resume_from:
        cmd.extend(['--resume', str(resume_from)])

    env = os.environ.copy()
    env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    print(f"  Training with config: {config_path}", flush=True)
    if resume_from:
        print(f"  Resuming from: {resume_from}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            cwd=str(Path(__file__).parent.parent))

    output = result.stdout + result.stderr

    val_loss = None
    peak_vram = 0.0
    samp_per_sec = 0.0
    experiment_dir = None

    for line in output.split('\n'):
        # Parse training loss from tqdm output
        if 'loss' in line and "'" in line:
            try:
                m = re.search(r"'loss':\s*'([0-9.]+)'", line)
                if m:
                    val_loss = float(m.group(1))
            except (ValueError, AttributeError):
                pass
        # Parse experiment directory
        if 'Experiment directory:' in line:
            m = re.search(r'Experiment directory:\s*(.+)', line)
            if m:
                experiment_dir = m.group(1).strip()

    try:
        import torch
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    except Exception:
        pass

    if result.returncode != 0:
        print(f"  CRASH: {result.stderr[-500:]}", flush=True)
        return None, peak_vram, samp_per_sec, experiment_dir

    return val_loss, peak_vram, samp_per_sec, experiment_dir


def find_checkpoint(experiment_dir=None, config_path=None):
    """Find the final checkpoint from a training run."""
    if experiment_dir:
        ckpt_dir = Path(experiment_dir) / 'checkpoints'
    elif config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        ckpt_dir = Path(raw.get('checkpoint', {}).get('output_dir', 'checkpoints'))
    else:
        ckpt_dir = Path('checkpoints')

    final = ckpt_dir / 'checkpoint_final.pt'
    if final.exists():
        return str(final)
    ckpts = sorted(ckpt_dir.glob('checkpoint_step*.pt'))
    if ckpts:
        return str(ckpts[-1])
    return None


def extract_lpips_from_experiment(experiment_dir):
    """Extract LPIPS metrics from validation results.json in an experiment directory.

    Returns dict with keys: reconstruction_lpips, text_only_lpips (or None if not found).
    """
    if not experiment_dir:
        return {'reconstruction_lpips': None, 'text_only_lpips': None}

    val_dir = Path(experiment_dir) / 'validation_outputs'
    if not val_dir.exists():
        # Try config-relative path
        val_dir = Path('validation_outputs')

    if not val_dir.exists():
        return {'reconstruction_lpips': None, 'text_only_lpips': None}

    # Find the latest step's results.json
    step_dirs = sorted(val_dir.glob('step*'), key=lambda p: p.name)
    if not step_dirs:
        return {'reconstruction_lpips': None, 'text_only_lpips': None}

    results_json = step_dirs[-1] / 'results.json'
    if not results_json.exists():
        return {'reconstruction_lpips': None, 'text_only_lpips': None}

    try:
        with open(results_json) as f:
            data = json.load(f)
        recon_lpips = data.get('reconstruction', {}).get('mean_lpips')
        text_lpips = data.get('text_only', {}).get('mean_lpips')
        return {
            'reconstruction_lpips': recon_lpips,
            'text_only_lpips': text_lpips,
        }
    except (json.JSONDecodeError, KeyError):
        return {'reconstruction_lpips': None, 'text_only_lpips': None}


def evaluate_val_loss_standalone(config_path, device='cuda:0'):
    """Load the model from training and compute validation loss."""
    import torch
    from production.config_loader import load_config
    from production.model import NanoDiT
    from production.train import evaluate_val_loss

    config = load_config(config_path)
    dev = torch.device(device)

    ckpt_path = find_checkpoint(config_path=config_path)
    if not ckpt_path:
        print("  No checkpoint found — cannot evaluate")
        return None

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

    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'], strict=False)
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

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


def log_result(phase, commit, val_loss, reconstruction_lpips, text_only_lpips,
               samp_per_sec, peak_vram, status, description, checkpoint_path):
    """Append a result to the phase results file."""
    results_path = results_file_for_phase(phase)
    if not results_path.exists():
        results_path.write_text(TSV_HEADER)

    def fmt(v, decimals=6):
        return f"{v:.{decimals}f}" if v is not None else ""

    row = '\t'.join([
        commit,
        fmt(val_loss),
        fmt(reconstruction_lpips, 4),
        fmt(text_only_lpips, 4),
        fmt(samp_per_sec, 1),
        fmt(peak_vram, 1),
        status,
        description,
        checkpoint_path or '',
    ])

    with open(results_path, 'a') as f:
        f.write(row + '\n')

    metric_str = f"val_loss={fmt(val_loss)}"
    if reconstruction_lpips is not None:
        metric_str += f" recon_lpips={fmt(reconstruction_lpips, 4)}"
    print(f"\n  Logged: {status} | {metric_str} | {description}")


def read_best_result(phase, metric='val_loss'):
    """Read the best metric value from a phase results file (among 'keep' entries).

    Returns (best_value, best_checkpoint_path) or (None, None).
    """
    results_path = results_file_for_phase(phase)
    if not results_path.exists():
        return None, None

    col_idx = METRIC_COLUMNS.get(metric)
    if col_idx is None:
        raise ValueError(f"Unknown metric: {metric}. Valid: {VALID_METRICS}")

    best_val = None
    best_ckpt = None

    for line in results_path.read_text().strip().split('\n')[1:]:
        parts = line.split('\t')
        if len(parts) < 9:
            continue
        status = parts[6]
        if status != 'keep':
            continue
        try:
            val = float(parts[col_idx])
            if val > 0 and (best_val is None or val < best_val):
                best_val = val
                best_ckpt = parts[8] if parts[8] else None
        except (ValueError, IndexError):
            pass

    return best_val, best_ckpt


def _get_budget_str(args):
    """Format budget string for display."""
    if hasattr(args, 'step_budget') and args.step_budget:
        return f"{args.step_budget} steps"
    elif hasattr(args, 'time_budget') and args.time_budget:
        return f"{args.time_budget} min"
    return "unset"


def _run_experiment(args, changes, description, is_baseline=False):
    """Core experiment runner shared by cmd_run and cmd_baseline."""
    phase = args.phase
    metric = getattr(args, 'metric', 'val_loss')
    resume_from = getattr(args, 'resume_from', None)
    device = f'cuda:{args.gpu}'

    print(f"\n{'='*60}")
    if is_baseline:
        print(f"  BASELINE RUN (Phase {phase})")
    else:
        print(f"  EXPERIMENT (Phase {phase}): {description}")
        print(f"  Changes: {args.changes}")
    print(f"  Budget: {_get_budget_str(args)}")
    print(f"  Metric: {metric}")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    print(f"{'='*60}\n")

    exp_config = Path(tempfile.mktemp(suffix='.yaml', prefix='autoresearch_'))
    try:
        step_budget = getattr(args, 'step_budget', None)
        time_budget = getattr(args, 'time_budget', None)
        make_experiment_config(args.base_config, changes, exp_config,
                               step_budget=step_budget, time_budget=time_budget)

        auto_tune_config(str(exp_config), device=device)

        val_loss, peak_vram, samp_per_sec, experiment_dir = run_training(
            str(exp_config), device=device, resume_from=resume_from)

        # Fallback: evaluate val_loss standalone if not captured from training output
        if val_loss is None:
            val_loss = evaluate_val_loss_standalone(str(exp_config), device=device)

        # Extract LPIPS metrics from validation outputs
        lpips = extract_lpips_from_experiment(experiment_dir)
        reconstruction_lpips = lpips['reconstruction_lpips']
        text_only_lpips = lpips['text_only_lpips']

        # Find checkpoint path for this experiment
        checkpoint_path = find_checkpoint(experiment_dir=experiment_dir, config_path=str(exp_config))

        # Decide keep/discard based on selected metric
        if metric == 'val_loss':
            metric_val = val_loss
        elif metric == 'reconstruction_lpips':
            metric_val = reconstruction_lpips
        elif metric == 'text_only_lpips':
            metric_val = text_only_lpips
        else:
            metric_val = val_loss

        if metric_val is None:
            status = 'crash'
        elif is_baseline:
            status = 'keep'
        else:
            best_val, _ = read_best_result(phase, metric)
            if best_val is None or metric_val < best_val:
                status = 'keep'
            else:
                status = 'discard'

        commit = git_short_hash()
        log_result(phase, commit, val_loss, reconstruction_lpips, text_only_lpips,
                   samp_per_sec, peak_vram, status, description, checkpoint_path)

        # Structured output for agent parsing
        print(f"\n---")
        print(f"phase: {phase}")
        print(f"val_loss: {val_loss:.6f}" if val_loss is not None else "val_loss: CRASH")
        if reconstruction_lpips is not None:
            print(f"reconstruction_lpips: {reconstruction_lpips:.4f}")
        if text_only_lpips is not None:
            print(f"text_only_lpips: {text_only_lpips:.4f}")
        print(f"peak_vram_gb: {peak_vram:.1f}")
        print(f"status: {status}")
        print(f"checkpoint: {checkpoint_path or 'NONE'}")
        print(f"description: {description}")
        print(f"---")

    finally:
        if exp_config.exists():
            exp_config.unlink()


def cmd_run(args):
    """Run a single experiment."""
    changes = parse_changes(args.changes)
    if not changes:
        print("ERROR: --changes is required. Format: 'key=value key2=value2'")
        sys.exit(1)
    desc = args.description or args.changes
    _run_experiment(args, changes, desc, is_baseline=False)


def cmd_baseline(args):
    """Run baseline (no changes) to establish reference metric."""
    _run_experiment(args, [], 'baseline', is_baseline=True)


def cmd_results(args):
    """Print results for a phase in a readable format."""
    results_path = results_file_for_phase(args.phase)
    if not results_path.exists():
        print(f"No results yet for phase {args.phase}. Run 'baseline --phase {args.phase}' first.")
        return

    print(f"Phase {args.phase} results:")
    print(results_path.read_text())

    for metric in VALID_METRICS:
        best_val, best_ckpt = read_best_result(args.phase, metric)
        if best_val is not None:
            print(f"  Best {metric}: {best_val:.6f}" + (f" → {best_ckpt}" if best_ckpt else ""))


def cmd_best_checkpoint(args):
    """Print the best checkpoint path from a phase."""
    metric = args.metric
    best_val, best_ckpt = read_best_result(args.phase, metric)

    if best_val is None:
        print(f"No 'keep' results found for phase {args.phase} with metric {metric}.")
        sys.exit(1)

    if not best_ckpt:
        print(f"Best {metric}={best_val:.6f} in phase {args.phase}, but no checkpoint path recorded.")
        sys.exit(1)

    # Structured output for agent parsing
    print(f"---")
    print(f"phase: {args.phase}")
    print(f"metric: {metric}")
    print(f"best_value: {best_val:.6f}")
    print(f"checkpoint: {best_ckpt}")
    print(f"---")


def _add_common_args(parser):
    """Add args shared between run and baseline."""
    parser.add_argument('--base-config', type=str, required=True)
    parser.add_argument('--phase', type=str, default='A',
                        help='Experiment phase (default: A)')
    parser.add_argument('--gpu', type=int, default=0)

    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--step-budget', type=int, default=None,
                        help='Training step budget (recommended). Deterministic, syncs with validation.')
    budget.add_argument('--time-budget', type=float, default=None,
                        help='Training time budget in minutes (legacy). Non-deterministic.')

    parser.add_argument('--resume-from', type=str, default=None,
                        help='Checkpoint path to resume from (for Phase B+)')
    parser.add_argument('--metric', type=str, default='val_loss',
                        choices=VALID_METRICS,
                        help='Metric for keep/discard (default: val_loss)')


def main():
    parser = argparse.ArgumentParser(
        description='Autoresearch: multi-phase tournament experiment loop for DiT training')
    sub = parser.add_subparsers(dest='command')

    # run
    p_run = sub.add_parser('run', help='Run an experiment with config changes')
    _add_common_args(p_run)
    p_run.add_argument('--changes', type=str, required=True,
                       help='Config changes: "key=value key2=value2"')
    p_run.add_argument('--description', type=str, default='')

    # baseline
    p_base = sub.add_parser('baseline', help='Run baseline (no changes)')
    _add_common_args(p_base)

    # results
    p_results = sub.add_parser('results', help='Show experiment results for a phase')
    p_results.add_argument('--phase', type=str, default='A')

    # best-checkpoint
    p_best = sub.add_parser('best-checkpoint', help='Print best checkpoint from a phase')
    p_best.add_argument('--phase', type=str, required=True)
    p_best.add_argument('--metric', type=str, default='val_loss',
                        choices=VALID_METRICS)

    args = parser.parse_args()
    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'baseline':
        cmd_baseline(args)
    elif args.command == 'results':
        cmd_results(args)
    elif args.command == 'best-checkpoint':
        cmd_best_checkpoint(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
