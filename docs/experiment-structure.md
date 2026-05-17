# Experiment Directory Structure

> **Note:** The `experiments/` directory is a symlink to the Stratum NAS
> (`/mnt/nas-ai-models/training-data/prx-tg`). Experiment data lives on
> the NAS; this document lives in git as the governance reference.

Every arm gets a canonical directory under `experiments/`. This is the single source of truth for that arm's data, config, provenance, and results.

## Naming

Experiment directories use **descriptive slugs** (lowercase, hyphens): `full-stack-baseline`, `seg-weight-spatial`, `no-repa-ablation`. The slug should convey what the arm tests in 2-4 words. Legacy single-letter aliases (arm_d, arm_e) can be recorded in `provenance.yaml` for cross-referencing older notes.

## Template

```
experiments/
  {slug}/
    README.md                    # What this arm tests, hypothesis, expected outcome
    config.yaml                  # Frozen config (copied at run start, never edited after)
    provenance.yaml              # Machine-readable provenance (see below)
    runs/                        # One subdir per training run attempt
      {YYYY-MM-DD_HHMM}/        # Timestamped (auto-created by train_production.py)
        metadata.json            # Git commit, command, dirty flag (auto-generated)
        training_log.jsonl       # Step-by-step metrics
        tensorboard/             # TensorBoard event files
        checkpoints/             # checkpoint_step{N}.pt, checkpoint_final.pt
        validation/              # On-GPU validation during training
          step{NNNN}/
            results.json
            reconstruction/
            text_only/
            dino_swap/
            text_manip/
        nohup.log                # Raw training stdout/stderr
    validation/                  # Post-hoc local re-validation (standardized)
      step{NNNN}/
        results.json
        reconstruction/
        text_only/
    figures/                     # Generated comparison plots for this arm
    notes/                       # Free-form analysis notes, observations
```

## provenance.yaml

Machine-readable record of where this arm's data came from and how it was produced.

```yaml
arm: e
legacy_alias: arm_e
hypothesis: >
  Seg spatial loss weighting improves text controllability
  without degrading reconstruction fidelity.
differs_from: d   # baseline arm for comparison
diff_summary: |
  - seg_weight_map enabled (face 2×, bg 0.5×, normalize=True)
  - all other settings identical to Arm D
git_commit: 0513760fe8f42e944e1258f9675466f86a598866
git_dirty: true   # be honest
training_host: vast.ai
training_gpu: RTX 4090
training_steps: 5000
data_source: $STRATUM_DIR (Stratum NAS, FFHQ WebDataset shards)
evacuated_from: ssh3.vast.ai:13840:/workspace/prx-tg/experiments/2026-05-17_0043/
evacuated_at: 2026-05-18T00:00:00Z   # filled after final sync
cron_job: vast-experiment-sync        # how data was evacuated
```

## Rules

1. **One arm = one directory.** No more `vast_sync_arm_e_*`, `vast_backup_*`, `ablation_arm_e/` sprawl.
2. **Config is frozen at run start.** The source config lives in `experiments/{slug}/config.yaml`. The training script copies it into `runs/{timestamp}/config.yaml` — that copy is the immutable record.
3. **Multiple runs are fine.** False starts, restarts, hyperparameter tweaks within the same arm all go under `runs/`. The README notes which run is canonical.
4. **Post-hoc validation goes in `validation/`**, not inside `runs/`. This is the standardized, comparable data (same seed, same samples, same local hardware for all arms).
5. **Raw Vast.ai data lands in `runs/`.** The sync cron job targets `experiments/{slug}/runs/`.
6. **Provenance is mandatory.** `provenance.yaml` is filled before the first run starts. Update `evacuated_at` after final sync.
7. **Register every new arm** in the [Experiment Registry](../README.md#experiment-registry) table in the README.

## Migration from current layout

| Current location | Move to |
|---|---|
| `scripts/arm_e_config.yaml` | `experiments/seg-weight-spatial/config.yaml` |
| `experiments/vast_sync_*/experiments/2026-05-17_0043/` | `experiments/seg-weight-spatial/runs/2026-05-17_0043/` |
| `experiments/autotune_arm_e_*.log` | `experiments/seg-weight-spatial/notes/` |
| `experiments/autotune_arm_e_*_provenance.md` | `experiments/seg-weight-spatial/provenance.yaml` (convert) |
| `experiments/vast_final_2026-05-12_0902/` | `experiments/full-stack-baseline/runs/2026-05-12_0902/` |
| `experiments/ablations/config_D_full_stack.yaml` | `experiments/full-stack-baseline/config.yaml` |
| `experiments/ablations/` (A, B, C configs) | `experiments/minimal-baseline/`, `tread-adamw/`, `tread-muon/` respectively |
| `sync_vast_arm_e.sh` (repo root) | delete (replaced by cron job) |
| `experiments/ablation_arm_e/` (empty) | delete |

## Creating a new arm

```bash
# 1. Create the directory
mkdir -p experiments/no-repa-ablation/{runs,validation,figures,notes}

# 2. Write the config (start from baseline, modify)
cp experiments/full-stack-baseline/config.yaml experiments/no-repa-ablation/config.yaml
# edit experiments/no-repa-ablation/config.yaml

# 3. Write provenance.yaml
cat > experiments/no-repa-ablation/provenance.yaml << 'EOF'
arm: f
legacy_alias: arm_f
hypothesis: ...
differs_from: full-stack-baseline
diff_summary: |
  - ...
git_commit: $(git rev-parse HEAD)
training_host: vast.ai
training_gpu: ...
training_steps: 5000
data_source: $STRATUM_DIR
EOF

# 4. Write README.md
cat > experiments/no-repa-ablation/README.md << 'EOF'
# Arm F: <title>
...
EOF

# 5. Launch training (pointing to the arm's config)
python -m production.train_production --config experiments/no-repa-ablation/config.yaml

# 6. Add a row to the Experiment Registry table in README.md

# 7. Update sync cron if needed (current job syncs all of experiments/)
```
