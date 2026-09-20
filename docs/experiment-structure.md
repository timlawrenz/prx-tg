# Experiment Directory Structure — the detail behind `AGENTS.md` §0

> The generic method lives in the `scientific-experiment-structure` skill.
> `AGENTS.md` §0 is prx-tg's law; **this document is the detail**. Where they
> disagree, `AGENTS.md` §0 wins.

## 1. The split everything depends on

`experiments/` is a **symlink** to the Stratum NAS
(`/mnt/nas-ai-models/training-data/prx-tg`), and it is committed to git as a
symlink blob:

```
$ git ls-files -s experiments
120000 b34cfe29d5f42370742a300d439cd518d26d86fa 0	experiments
```

Mode `120000` is a symlink — **git cannot see inside it.** The repo is therefore
split in two:

| Kind | What | Where | In git? |
|---|---|---|---|
| **Soft** | config, provenance, arm README, notes, conclusions, ledger, briefs | `experiment-configs/{slug}/` | ✅ yes |
| **Hard** | checkpoints/weights, tensorboard event files, validation images, raw logs, datasets | `experiments/{slug}/` | ❌ NAS only |

A `config.yaml`, a `provenance.yaml` or an arm `README.md` written under
`experiments/` is **beyond git's reach permanently**. This is not hypothetical: it
is how 21 of the current 37 arms lost their provenance, and why 24 have no ledger
entry anyone can find.

## 2. Soft layout (git) — `experiment-configs/{slug}/`

```
experiment-configs/
  {slug}/
    README.md        # hypothesis, differs-from, expected outcome, record status
    config.yaml      # canonical config; frozen at run start; never edited after
    provenance.yaml  # machine-readable record + the four link fields (§3)
  vae-ceiling/       # backfilled 2026-09-19 (retrospective record)
  stratum2-pose2/    # the original precedent (2026-08-01)
```

## 3. Hard layout (NAS) — `experiments/{slug}/`

```
experiments/
  {slug}/
    runs/{YYYY-MM-DD_HHMM}/       # one dir per run attempt (auto-created by train_production.py)
      metadata.json               # git commit, command, git_dirty flag (auto-generated)
      training_log.jsonl           # step-by-step metrics
      tensorboard/                 # event files
      checkpoints/                 # checkpoint_step{N}.pt, checkpoint_final.pt
      validation/step{NNNNNN}/     # in-training validation
      config.yaml                  # the immutable copy the trainer wrote — fine here, it is hard
      nohup.log
    validation/                   # post-hoc standardized re-validation
      {YYYY-MM-DD_HHMM}/          # scoped to run timestamp
        step{NNNNNN}/
          results.json
          reconstruction/
          text_only/
          evaluation_results.json  # CLIP + aesthetic + DWPose metrics (per checkpoint)
          collage.png              # generated sample grid for this checkpoint
    metric_trajectory.csv          # aggregated quality-metric trajectory across checkpoints
    metric_graphs.png              # trajectory plots (aesthetic, CLIP, face confidence)
    progression_timeline.png       # checkpoint collage timeline
    visual_debug/  figures/  notes/  quality_metrics/
```

## 4. `provenance.yaml` — the full template

The generic template plus the **four link fields** that close the chain:

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

# --- the four link fields ---
branch: exp/stratum2-integration          # the arm's REAL branch; never rename history
tags: [arm/dip-conv-head/concluded-GO]
conclusions:                              # numbered 1..N — what a reader can cite
  - "5k->10k extension improved facerealism: recon LPIPS 0.746 -> 0.7251, face-conf +45%"
  - "Blind human review: 11W/2L over 13 index-matched pairs; calibration 7/7"
  - "Photorealism NOT met: 3/6 G0 physics gates sit below the real-FFHQ band"
ledger_anchor: "dip-conv-head 10k continuation"   # heading text in docs/EXPERIMENTS_AND_RESULTS.md
blog_brief: docs/blog/brief-03-dip-conv-head-10k.md
blog_post: null

# --- run provenance ---
git_commit: 0209a2634473319b34e4f1c9749e8c175b0d41a1
git_dirty: false
training_host: game           # game | strix | vast.ai
training_gpu: RTX 4090
training_steps: 10000
data_snapshot: "FFHQ stratum 70k portraits"
agent_model: null             # LLM that drove the run, when agent-assisted
run_dirs:                     # canonical run FIRST
  - experiments/dip-conv-head/runs/2026-08-30_1449
verdict: active               # active | GO | PIVOT | PARK | KILL | pending
superseded_by: null
supersedes: null
```

Validate before committing:

```bash
python -c "import yaml; yaml.safe_load(open('experiment-configs/{slug}/provenance.yaml'))"
```

**Retrospective arms** (run before 2026-09-19, when this contract existed): record
missing links as **absent facts** — `branch: null`, `tags: []`, a `ledger_anchor: null`
plus a note — do **not** invent them. See `experiment-configs/vae-ceiling/` for the
worked example, including a `queued_rerun` block for outstanding repairs.

## 5. The chain, field by field

Any agent must be able to answer this **from git alone**:

> "Branch/Tag A tried this thing, came up with this data, we had conclusions 1, 2 and
> 3, and we wrote this blog post about it."

| Link | Artifact |
|---|---|
| Branch/Tag A | `provenance.yaml: branch` + `tags` |
| "tried this thing" | `provenance.yaml: hypothesis` / `falsified_if` / `pre_registered_gate`, `README.md` |
| "came up with this data" | `provenance.yaml: run_dirs` (NAS paths) + `git_commit` |
| "conclusions 1, 2, 3" | `provenance.yaml: conclusions` ↔ `EXPERIMENTS_AND_RESULTS.md` (`ledger_anchor`) |
| "wrote this blog post" | `provenance.yaml: blog_brief` + `blog_post` |

If any link is missing, the arm is **not concluded** — it is `PENDING`.

## 6. Rules

1. **One arm = one slug.** No `vast_sync_arm_e_*`, `vast_backup_*`, `ablation_arm_e/`
   sprawl. Legacy timestamp-named dirs get mapped to an arm or marked legacy.
2. **The soft trio lives in git** (`experiment-configs/{slug}/`), never on the NAS.
3. **Config is frozen at run start.** The canonical copy is
   `experiment-configs/{slug}/config.yaml`; the trainer's own copy under
   `runs/{timestamp}/` is the immutable hard record.
4. **Provenance is written and committed BEFORE the first run**, including
   `falsified_if`, `pre_registered_gate` and `mode`.
5. **Multiple runs are fine** — false starts and restarts go under `runs/`. The
   README names the canonical run (listed first in `run_dirs`).
6. **Post-hoc validation goes in `experiments/{slug}/validation/`**, not inside a
   run dir: that is the standardized, comparable data.
7. **Every arm gets a ledger entry** in `docs/EXPERIMENTS_AND_RESULTS.md` with
   numbered `conclusions:`, plus a row in the README Experiment Registry.
8. **Tag the verdict**: `arm/{slug}/concluded-{GO|PIVOT|PARK|KILL}` beyond
   `arm/{slug}/registered`.
9. **One blog brief per verdict / phase milestone**, at
   `docs/blog/brief-{NN}-{slug}.md`, and it lives on **that experiment's branch** —
   a brief has no home without its branch.
10. **No new top-level directory** for experiment tracking. Extend this structure.

## 7. Creating a new arm — correct sequence

```bash
SLUG=no-repa-ablation

# 1. SOFT TRIO FIRST, in git — before any GPU time
mkdir -p experiment-configs/$SLUG
cp experiment-configs/full-stack-baseline/config.yaml experiment-configs/$SLUG/config.yaml
$EDITOR experiment-configs/$SLUG/config.yaml       # comment header: what it tests /
                                                   # differs-how / expected outcome
$EDITOR experiment-configs/$SLUG/provenance.yaml   # hypothesis, falsified_if,
                                                   # pre_registered_gate, mode
$EDITOR experiment-configs/$SLUG/README.md
git add experiment-configs/$SLUG
git commit -m "arm($SLUG): pre-register gate + config"
git tag arm/$SLUG/registered

# 2. HARD dir on the NAS (runs/ is created by the trainer)
mkdir -p experiments/$SLUG

# 3. Launch — point at the git-tracked config, not a NAS copy
python -m production.train_production --config experiment-configs/$SLUG/config.yaml

# 4. Register the arm
#    - docs/EXPERIMENT_TREE.md
#    - README.md Experiment Registry row
```

## 8. Closing an arm

1. Confirm `metadata.json` exists and `git_dirty: false`.
2. Run the adversarial pass; write the ledger entry with numbered `conclusions:`.
3. Fill `verdict`, `conclusions`, `run_dirs`, `ledger_anchor`, `blog_brief` in
   `provenance.yaml`.
4. `git tag arm/{slug}/concluded-{VERDICT}`.
5. Write the brief on the arm's branch.
6. Update the README Experiment Registry row.
7. **Verify:**

```bash
python scripts/check_arm_records.py          # non-zero exit = STOP, not concluded
```

## History — what this document used to say

Until 2026-09-19 this document described `experiments/{slug}/` as "the single source
of truth for that arm's data, config, provenance, and results", and its
"Creating a new arm" recipe contained:

```bash
cat > experiments/no-repa-ablation/provenance.yaml << 'EOF'
```

Both put soft governance files where git cannot reach them, and both are the direct
cause of the missing-provenance arms. The instruction is corrected above; the old
migration table that routed configs and provenance into `experiments/` has been
removed for the same reason.