# AGENTS.md — prx-tg Experiment Governance

> Rules for any AI agent (Hermes, Claude Code, Codex, Cursor) working in this repo.
> Violations of the MUST rules below invalidate the run.

## Project Overview

prx-tg is a pixel-space DiT (Diffusion Transformer) for face generation. It trains a NanoDiT model with TREAD routing, REPA alignment, Muon optimizer, and optional segmentation weighting. Training runs on Vast.ai GPUs; data lives on a Stratum NAS.

## Repository Layout

```
production/           # Core training code (model, train, validate, config_loader)
scripts/              # Experiment configs, helper scripts, autoresearch
experiments/          # Run outputs: checkpoints, tensorboard, validation, metadata
  <timestamp>/        # Each run gets a timestamped directory
    metadata.json     # Git commit, config path, command, dirty flag
    config.yaml       # Frozen config snapshot (copied at run start)
    checkpoints/      # Model checkpoints (step*.pt, final.pt)
    tensorboard/      # TensorBoard logs
    validation/       # Validation outputs (images, results.json)
    visual_debug/     # Debug visualizations
research/             # Ratiocinator specs and results (when initialized)
  specs/              # Experiment YAML specs for ratiocinator fleet
  results/            # Collected metrics and analysis
.hermes/plans/        # Hermes Agent execution plans
```

## Experiment Execution Rules

### Before Any Training Run

1. **MUST** commit or stash all code changes. `git_dirty: true` in metadata.json means the run is not reproducible.
2. **MUST** use a named config file (not inline overrides). The config is frozen into the experiment directory.
3. **MUST** record which arm/experiment this run belongs to (use `--experiment-name` or the config filename convention: `arm_{letter}_config.yaml`).
4. **MUST** verify data availability before launching (`ls $STRATUM_DIR` or check shard dirs).

### Config File Conventions

- Base config: `production/config.yaml` (Arm D — full stack baseline)
- Arm variants: `scripts/arm_{letter}_config.yaml`
- Each arm config **MUST** have a comment header explaining:
  - What it tests (ablation hypothesis)
  - How it differs from the baseline
  - Expected outcome

### Naming Conventions

| Entity | Format | Example |
|--------|--------|---------|
| Arm config | `arm_{letter}_config.yaml` | `arm_e_config.yaml` |
| Experiment dir | `<YYYY-MM-DD_HHMM>` (auto) | `2026-05-13_2035` |
| Vast sync dir | `vast_sync_<arm>_<date>_<time>` | `vast_sync_arm_e_2026-05-16_0525` |
| Ratiocinator spec | `research/specs/<name>.yaml` | `research/specs/arm_e_vs_d.yaml` |
| Checkpoint | `checkpoint_step{N}.pt` | `checkpoint_step2500.pt` |

### During a Run

- **MUST NOT** modify code in `production/` while a training run is active.
- **MUST** monitor for NaN/divergence in the first 100 steps (check training.log or tensorboard).
- **SHOULD** sync checkpoints from Vast.ai periodically (`scripts/sync_vast.sh` or rsync).

### After a Run

1. **MUST** verify `metadata.json` exists and `git_dirty` is false.
2. **MUST** run validation on the final checkpoint if not run during training:
   ```bash
   python scripts/run_checkpoint_validation.py \
     --config experiments/<dir>/config.yaml \
     --checkpoint experiments/<dir>/checkpoints/checkpoint_final.pt
   ```
3. **MUST** collect metrics into a comparison table before claiming one arm beats another.
4. **SHOULD** sync tensorboard logs for visual comparison.

## Ratiocinator Integration

When using `ratiocinator fleet` for parallel runs:

1. Spec files go in `research/specs/`.
2. The spec **MUST** pin `repo.commit` to a specific SHA (not just branch HEAD).
3. Budget caps **MUST** be set (`budget.max_dollars`, `budget.train_timeout_s`).
4. Metrics protocol is `json_line` — training code outputs `METRICS:{...}` lines.

## Collateral Locations

| Artifact | Location | Retention |
|----------|----------|-----------|
| Checkpoints | `experiments/<run>/checkpoints/` | Keep last 10 per run |
| Validation images | `experiments/<run>/validation/` | Keep all |
| TensorBoard | `experiments/<run>/tensorboard/` | Keep all |
| Training log | `experiments/<run>/training_log.jsonl` | Keep all |
| Frozen config | `experiments/<run>/config.yaml` | Keep all |
| Metadata | `experiments/<run>/metadata.json` | Keep all |
| Vast.ai syncs | `experiments/vast_sync_*` | Archive after analysis |
| Autoresearch TSV | `results/results_phase_*.tsv` | Keep all |
| Ratiocinator results | `research/results/` | Keep all |

## Code Review Rules

- Changes to `production/model.py` or `production/train.py` **MUST** be tested with a short sanity run (50-100 steps) before committing.
- Config changes that alter loss computation (seg_weight, repa, tread) require a new arm letter.
- Hyperparameter tweaks within an existing arm (lr, batch_size) use the same arm letter but a new run.

## Data Pipeline

- Source: Stratum NAS at `$STRATUM_DIR`
- Shards: Pre-bucketed WebDataset shards at `data/shards/faces7k/`
- Buckets: Multiple aspect ratios (1024×1024, 1216×832, etc.)
- **MUST NOT** modify shard data during training.

## GPU / Vast.ai Rules

- Always use `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- Pin CUDA version in Docker image (currently `cuda12.8`).
- Sync experiment directory from Vast.ai before destroying the instance.
- Use `gradient_checkpointing: true` for memory efficiency.
