# AGENTS.md — prx-tg Experiment Governance

> Rules for any AI agent (Hermes, Claude Code, Codex, Cursor) working in this repo.
> Violations of the MUST rules below invalidate the run.

---

# 0. READ THIS FIRST — where the method comes from, and the laws that bind it

The **method** is generic and lives in the `scientific-experiment-structure` skill.
Load it before starting any experiment:

```
skill_view(name='scientific-experiment-structure')
```

That skill is **shared with other projects — do not edit it to add prx-tg
specifics.** This file is the mapping layer: it turns the generic method into
prx-tg's concrete paths and laws. Where the two disagree, **this file wins.**

| Generic rule (in the skill) | prx-tg law (concrete, in this repo) |
|---|---|
| "Separate data from governance" | Soft data in git; hard data under the `experiments/` symlink → §0.1 |
| "One arm = one directory" | `experiments/{slug}/` (NAS, hard) **and** `experiment-configs/{slug}/` (git, soft) → §0.2 |
| "Provenance is mandatory" | `experiment-configs/{slug}/provenance.yaml`, committed, with the link fields in §0.3 |
| "Pre-register the gate before results" | `pre_registered_gate` + `falsified_if` + `mode` in provenance.yaml, committed **before** the first run |
| "Register every new arm" | `docs/EXPERIMENT_TREE.md` **and** a row in the `README.md` Experiment Registry |
| "Write the ledger entry" | `docs/EXPERIMENTS_AND_RESULTS.md`, including numbered `conclusions:` |
| "Adversarial pass before any PASS" | the checklist block inside the ledger entry; verdict is `PENDING` if any box is unchecked |
| "KILL → tombstone" | `docs/DISCONTINUATION_NOTICE*.md` |
| "Tag the conclusion" | `arm/{slug}/concluded-{GO\|PIVOT\|PARK\|KILL}` → §0.4 |
| "Report the result publicly" | `docs/blog/brief-{NN}-{slug}.md` → §0.5 |

## 0.1 The one rule that makes everything else possible

**GitHub holds ALL the soft data. The NAS holds the hard data.**

- **Soft — MUST be in git:** configs, provenance, per-arm READMEs, ledgers, trees,
  governance docs, metric/JSON summaries, plots, blog briefs, papers.
- **Hard — NAS only:** checkpoints and weights, tensorboard event files, validation
  images, raw training logs, datasets.

`experiments` is a symlink to `/mnt/nas-ai-models/training-data/prx-tg`, and it is
committed to git as a **symlink blob**:

```
$ git ls-files -s experiments
120000 b34cfe29d5f42370742a300d439cd518d26d86fa 0	experiments
```

Mode `120000` is a symlink. **Git cannot see inside it — ever.** Anything written
under `experiments/` is invisible to GitHub permanently. A config, a provenance
file or an arm README placed there is lost to every future reader, human or agent.

This is not hypothetical: it is how this project lost the provenance of **21 of its
37 arms**, and why 24 of them have no ledger entry anyone can find.

## 0.2 Arm soft home — `experiment-configs/{slug}/`

Every arm's soft trio lives **here, in git**:

```
experiment-configs/{slug}/
├── README.md          # hypothesis, differs-from, expected outcome
├── config.yaml        # canonical config, frozen at run start
└── provenance.yaml    # machine-readable provenance + the link fields in §0.3
```

`experiments/{slug}/` on the NAS keeps only the hard artifacts:
`runs/`, `checkpoints/`, `tensorboard/`, `validation/`, logs.

**Precedent:** `experiment-configs/stratum2-pose2/` (2026-08-01) is the only arm
that did this correctly. The convention was never written down anywhere, so every
arm after it regressed to putting soft files on the NAS. Do not repeat that.

## 0.3 `provenance.yaml` — required fields

The generic template from the skill, **plus the four link fields** that let a future
reader reconstruct the whole chain:

```yaml
arm: dip-conv-head
arm_letter: DCH
aliases: [dip-conv-head-10k]
mode: confirmatory            # confirmatory | exploratory — set BEFORE the first run
hypothesis: >                 # must be falsifiable
  Extending the DiP conv-head arm from 5k to 10k steps moves the model toward
  photorealism.
falsified_if: >               # the outcome that proves the hypothesis WRONG
  LPIPS does not improve, or the G0 physics gates regress.
pre_registered_gate: "PASS if blind win-rate CI LB > 0.5 AND LPIPS@10k <= 0.746 AND no G0 regression AND loss <= 0.0135."
differs_from: dip-conv-head
diff_summary: |
  - budget extended 5000 -> 10000 (same arm)
  - resumed from checkpoint_step0005000.pt

# --- the four link fields (prx-tg addition) ---
branch: exp/stratum2-integration          # the arm's REAL branch — record it, never rename history
tags: [arm/dip-conv-head/concluded-GO]    # "Tag A"
conclusions:                              # numbered 1..N — the conclusions a reader can cite
  - "5k->10k extension improved facerealism: recon LPIPS 0.746 -> 0.7251, face-conf +45%"
  - "Blind human review: 11W/2L over 13 index-matched pairs; calibration 7/7"
  - "Photorealism NOT met: 3/6 G0 physics gates sit below the real-FFHQ band"
ledger_anchor: "dip-conv-head 10k continuation"   # heading text in docs/EXPERIMENTS_AND_RESULTS.md
blog_brief: docs/blog/brief-03-dip-conv-head-10k.md
blog_post: null                           # URL once published

# --- run provenance ---
git_commit: 0209a2634473319b34e4f1c9749e8c175b0d41a1
git_dirty: false
training_host: game
training_gpu: RTX 4090
training_steps: 10000
data_snapshot: "FFHQ stratum 70k portraits"
agent_model: null                         # LLM that drove the run, when agent-assisted
agent_model_snapshot: null
run_dirs:                                 # canonical run FIRST
  - experiments/dip-conv-head/runs/2026-08-30_1449
superseded_by: null
supersedes: null
```

Validate before committing:

```bash
python -c "import yaml; yaml.safe_load(open('experiment-configs/{slug}/provenance.yaml'))"
```

## 0.4 Tags — so "Tag A tried this" is a real, resolvable thing

| Tag | When | Example |
|---|---|---|
| `arm/{slug}/registered` | at arm creation | `arm/dip-conv-head/registered` |
| `arm/{slug}/concluded-{GO\|PIVOT\|PARK\|KILL}` | at verdict | `arm/dip-conv-head/concluded-GO` |
| `arm/{slug}/data-{YYYY-MM-DD_HHMM}` | at a data snapshot worth pinning | `arm/faces70k-fp8/data-2026-06-19_1015` |

**Branch per arm:** `arm/{slug}` for new work. Existing branches keep their real
names — record the actual branch in `provenance.yaml`; **never rename history to
fit a convention.**

## 0.5 The chain we must be able to reconstruct

For any arm, any agent must be able to answer the following **from git alone**:

> "Branch/Tag A tried this thing, came up with this data, we had conclusions 1, 2
> and 3, and we wrote this blog post about it."

| Link | Artifact |
|---|---|
| Branch/Tag A | `provenance.yaml: branch` + `tags` |
| "tried this thing" | `provenance.yaml: hypothesis` / `falsified_if` / `pre_registered_gate`, `README.md` |
| "came up with this data" | `provenance.yaml: run_dirs` (NAS paths) + `git_commit` |
| "conclusions 1, 2, 3" | `provenance.yaml: conclusions` ↔ `docs/EXPERIMENTS_AND_RESULTS.md` (`ledger_anchor`) |
| "wrote this blog post" | `provenance.yaml: blog_brief` + `blog_post` |

If any link is missing, the arm is **not concluded** — it is `PENDING`.

## 0.6 Verification — run this before claiming anything

```bash
python scripts/check_arm_records.py
```

Non-zero exit = **stop.** It is deterministic code, not an LLM (the skill's rule:
never let the model that produced a result be the thing that certifies it). It
fails when:

- a slug directory on the NAS has no `experiment-configs/{slug}/provenance.yaml`
- a `branch` or `tag` recorded in provenance does not resolve in git
- the canonical `run_dir` does not exist
- `verdict != active` but `ledger_anchor` does not resolve in the ledger
- a concluded arm has no `blog_brief`
- the soft trio exists on the NAS but not in git

During the migration window, `--allow-missing` downgrades *not-yet-migrated* arms
to warnings while still failing on broken links.

## 0.7 MUST rules agents have actually broken here

1. **MUST NOT** write `config.yaml`, `provenance.yaml` or an arm `README.md` under
   `experiments/` — git cannot see them. Use `experiment-configs/{slug}/`.
2. **MUST** commit `experiment-configs/{slug}/` (trio + gate) **before** the first
   training run.
3. **MUST** set `mode:` (`confirmatory` | `exploratory`) before the first run.
   A peeked result is exploratory, permanently.
4. **MUST** write numbered `conclusions:` in provenance.yaml and the matching
   ledger entry after the verdict.
5. **MUST** tag the conclusion (`arm/{slug}/concluded-...`).
6. **MUST NOT** create a new top-level directory for experiment tracking. The
   structure above already exists — **extend it, do not add a fifteenth standard.**

---

## Project Overview

prx-tg is a pixel-space DiT (Diffusion Transformer) for face generation. It trains a
NanoDiT model with TREAD routing, REPA alignment, Muon optimizer, and optional
segmentation weighting. Training runs on Vast.ai GPUs and on the local 4090; data
lives on a Stratum NAS.

## Repository Layout

```
production/              # Core training code (model, train, validate, config_loader)
scripts/                 # Helper scripts, autoresearch, harness, check_arm_records.py
experiment-configs/      # GIT-TRACKED soft per-arm data  <-- arm configs + provenance live HERE
  {slug}/
    README.md            # Hypothesis, differs-from, expected outcome
    config.yaml          # Canonical frozen config
    provenance.yaml      # Machine-readable provenance + link fields (§0.3)
experiments/             # SYMLINK -> Stratum NAS. HARD data only.
  {slug}/
    runs/<YYYY-MM-DD_HHMM>/   # checkpoints, tensorboard, validation images, logs
docs/                    # Git-tracked governance + documentation
  EXPERIMENTS_AND_RESULTS.md   # Permanent ledger (conclusions live here)
  EXPERIMENT_TREE.md           # Active / Concluded / TBD map
  experiment-structure.md      # Detailed layout + provenance template
  blog/                        # Blog briefs for lawrenz.com (brief-{NN}-{slug}.md)
research/                # Ratiocinator specs, avenues registry, results
  avenues/registry.json  # Arm states + strikes (committed by policy)
  results/               # Experiment writeups + metrics (soft — should be committed)
release/                 # Stripped release weights (hard — NAS/local only)
.hermes/plans/           # Hermes Agent execution plans (committed by policy)
```

## Experiment Execution Rules

### Before Any Training Run

1. **MUST** commit or stash all code changes. `git_dirty: true` in metadata.json means the run is not reproducible.
2. **MUST** use a named config file (not inline overrides), stored at `experiment-configs/{slug}/config.yaml`.
3. **MUST** record which arm this run belongs to (via the `experiment-configs/{slug}/` directory).
4. **MUST** verify data availability before launching (`ls $STRATUM_DIR` or check shard dirs).
5. **MUST** have the soft trio committed (§0.2) and the pre-registered gate written before launching.

### Config File Conventions

- Base config: `experiment-configs/full-stack-baseline/config.yaml` (Arm D — full stack baseline)
- Arm variants: `experiment-configs/{slug}/config.yaml`
- Each arm config **MUST** have a comment header explaining:
  - What it tests (ablation hypothesis)
  - How it differs from the baseline
  - Expected outcome

### Naming Conventions

| Entity | Format | Example |
|--------|--------|---------|
| Arm soft dir (git) | `experiment-configs/{slug}/` | `experiment-configs/seg-weight-spatial/` |
| Arm config (git) | `experiment-configs/{slug}/config.yaml` | `experiment-configs/seg-weight-spatial/config.yaml` |
| Arm provenance (git) | `experiment-configs/{slug}/provenance.yaml` | `experiment-configs/seg-weight-spatial/provenance.yaml` |
| Arm hard dir (NAS) | `experiments/{slug}/` | `experiments/seg-weight-spatial/` |
| Run dir (NAS) | `experiments/{slug}/runs/<YYYY-MM-DD_HHMM>/` | `experiments/seg-weight-spatial/runs/2026-05-17_0043/` |
| Blog brief (git) | `docs/blog/brief-{NN}-{slug}.md` | `docs/blog/brief-02-dino-patches-counterproductive.md` |
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
     --config experiment-configs/{slug}/config.yaml \
     --checkpoint experiments/{slug}/runs/<timestamp>/checkpoints/checkpoint_final.pt
   ```
3. **MUST** run the adversarial pass and write the ledger entry with numbered `conclusions:`.
4. **MUST** update `experiment-configs/{slug}/provenance.yaml` (verdict, tags, conclusions, links).
5. **MUST** tag the conclusion (§0.4).
6. **SHOULD** sync tensorboard logs for visual comparison.
7. **MUST** update the Experiment Registry table in `README.md` with final status.
8. **MUST** run `python scripts/check_arm_records.py` and get a zero exit.

## Ratiocinator Integration

When using `ratiocinator fleet` for parallel runs:

1. Spec files go in `research/specs/`.
2. The spec **MUST** pin `repo.commit` to a specific SHA (not just branch HEAD).
3. Budget caps **MUST** be set (`budget.max_dollars`, `budget.train_timeout_s`).
4. Metrics protocol is `json_line` — training code outputs `METRICS:{...}` lines.

## Collateral Locations

| Artifact | Location | In git? | Retention |
|----------|----------|---------|-----------|
| Frozen config | `experiment-configs/{slug}/config.yaml` | ✅ yes | Keep all |
| Provenance | `experiment-configs/{slug}/provenance.yaml` | ✅ yes | Keep all |
| Arm README | `experiment-configs/{slug}/README.md` | ✅ yes | Keep all |
| Blog brief | `docs/blog/brief-{NN}-{slug}.md` | ✅ yes | Keep all |
| Ledger entry | `docs/EXPERIMENTS_AND_RESULTS.md` | ✅ yes | Keep all |
| Checkpoints | `experiments/{slug}/runs/<ts>/checkpoints/` | ❌ NAS | Keep last 10 per run |
| Validation images | `experiments/{slug}/runs/<ts>/validation/` | ❌ NAS | Keep all |
| TensorBoard | `experiments/{slug}/runs/<ts>/tensorboard/` | ❌ NAS | Keep all |
| Training log | `experiments/{slug}/runs/<ts>/training_log.jsonl` | ❌ NAS | Keep all |
| Metadata | `experiments/{slug}/runs/<ts>/metadata.json` | ❌ NAS | Keep all |
| Ratiocinator results | `research/results/` | ⚠️ whitelist needed | Keep all |

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
- Shared-GPU scheduling: the 4090 is arbitrated by the file-based scheduler at
  `/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py`. See the
  `gpu-resource-scheduler` skill. Never hold the GPU for a whole multi-day run —
  reserve only to the next natural break.
