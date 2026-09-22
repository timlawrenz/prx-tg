# Research Plan v2 — P1 and PP as separately gated processes (foundation for the Eidolon renderer)

**Date:** 2026-09-21 (v2 — supersedes `2026-09-21_2320-eidolon-visualization-engine.md`)
**Status:** APPROVED-IN-PRINCIPLE 2026-09-21; Stage A executing. Decision answers: D1=yes (Stage A first); D2=train/val/test split; D3=experiments at 10k scale (50k reserved for finals); D4=explained (dropout ablation); D5=vertical (person-level) holdout when hegre binds at Stage E, excluded from P1/PP training; additional-dataset offer noted for the train side.
**Verified live state at writing:** PP run `2026-09-20_1413` is stopped (no trainer process, last log step 2198, GPU free).

---

## 0. Premise change (Tim, 2026-09-21)

- **Arm PP is cancelled and tainted.** It warm-started from a P1 that had no
  official gate, and both P1 and PP were "basically ignoring text conditioning
  and almost entirely memorizing CLS tokens as keys into the training dataset."
  No downstream arm may build on either checkpoint.
- **P1 becomes its own gated research process** — with real, pre-registered
  gates that measure conditioning usage and memorization, possibly an ablation.
- The plan **starts at data preparation and segmentation**, then P1 (data,
  dropout, gates), then PP (same questions: what data, what dropout, gates).
- **Budget is an open variable** (50k-step runs existed in earlier phases —
  the envelope is not the 10k that P1/PP used).

## 1. Why the taint diagnosis is structurally credible (grounding, from code)

| Mechanism | Evidence |
|---|---|
| DINOv3 CLS is a per-image ~unique key (1024-d, one per training image). Whenever present, CLS→image is learnable by lookup; generalization is not required by the loss. | data layout, `StratumDataset` |
| CFG dropout gives the model CLS on ~45% of steps, text on ~65%, **both on only ~40%** (`train.py:300-329`, P1 config profile: p_uncond .10 / p_text_only .25 / p_dino_cls_only .05 / p_dino_patches_only .05 / p_drop_pose .10 / p_pose_only .05). When both are present, CLS is the easier signal — text is free to be ignored. | computed marginals from `production/train.py` |
| **There is no train/val/test split anywhere in the pipeline.** Validation reconstructs *training* images — recon LPIPS rewards memorization and cannot detect it. | grep: no `split`/`holdout` logic in `production/` |
| The two instruments that would have caught this — `run_text_manip` (does swapping caption attributes move the output?) and `run_dino_swap` (does swapping CLS move the output?) — **exist in `production/validate.py` but were disabled in P1's config** (`run_dino_swap: false`, `run_text_manip: false`). | P1 `config.yaml:132-133` |

Nothing here contradicts the diagnosis; the instrumentation gap explains why it
was never measured. Stage A converts the diagnosis into numbers before any
redesign is gated against it.

## 2. Stage plan

```
Stage A  Forensic baseline (eval-only, no training, ~1-2 GPU-h)
Stage B  Data preparation & segmentation (no GPU training)
Stage C  P1 gated process (gates + optional dropout ablation → full P1 re-run)
Stage D  PP gated process (data/dropout/gates, warm-start from the GATED P1)
Stage E  Eidolon renderer arm (downstream; only sketched — depends on C/D)
```

### Stage A — Forensic baseline: put numbers on the taint (eval-only)

Purpose: confirm/quantify the failure mode with artifact-grade evidence and
produce the **baseline numbers the redesign gates against**. All probes reuse
existing instruments where possible, on the P1 step-10000 checkpoint, with the
Arm J champion (faces70k-fp8 step-35k) as reference.

- **A1 Text-binding probe:** enable `run_text_manip` against P1-10k and Arm J —
  swap caption attributes (slender→muscular, pale→tan, …) at fixed seed/CLS,
  measure output delta. Diagnosis predicts ≈0 for P1; Arm J establishes what
  "load-bearing text" scores on this instrument.
- **A2 CLS-binding probe:** `run_dino_swap` — swap CLS between pairs at fixed
  seed/text. Expected strong on both (that part is not the disease).
- **A3 Memorization probe:** generate with exact training conditioning
  (text+CLS+pose of training image i, fixed seeds) → nearest-neighbor retrieval
  against the training set (AuraFace/DINO embedding NN + LPIPS to source).
  Metric: NN-is-source hit rate + distance distribution vs a matched
  novel-conditioning control.
- **A4 Novel-CLS probe:** interpolate/average CLS vectors (never seen) → is the
  output a novel blend or the nearest training image? This is the
  generalization question in its purest form (and the one the eidolon product
  ultimately depends on: sliders and priors emit *novel* vectors).
- **Deliverable:** `research/results/` forensic report + metric jsonl; verdict
  recorded in the ledger. Eval-only, exploratory — it calibrates instruments,
  it does not PASS/FAIL any training arm.

### Stage B — Data preparation & segmentation (the new foundation)

**B1 Corpus & satellite audit (FFHQ stratum 70k).** Readdir-only coverage
counts per artifact: `pixel`, `flux_latent`, `t5_hidden`, `dinov3_cls`,
`dinov3_patches`, `pose`, `seg`/`seg2`, `auraface_lda`, `z_g`. The arm's
`data_snapshot` records real numbers; scan gates on artifact existence
(Pitfall 16 discipline).

**B2 Split design — the currently missing piece.**
- Deterministic (seeded, committed manifest in git) **train / val / test**
  split of the 70k, stratified across the 7 aspect-ratio buckets.
  Proposal: 68k train / 1k val / 1k **never-touched test** (touched once, at
  the very end of a gated arm — evaluation-design gate #3).
- Val becomes the home of *all* recon-style validation (recon LPIPS moves OFF
  training images — this is the single change that makes memorization
  measurable in-training).
- Implementation: a split manifest (`data/splits/v1.json`) consumed by
  `StratumDataset` and every eval loader (train, validate, visual_debug,
  quality metrics, gates) — one source of truth, greppable.

**B3 Fixed probe panel (temporal-trajectory instrument).** A committed panel
of conditioning tuples (val images, captions, seeds, CLS vectors, plus the
novel-CLS interpolations from A4) evaluated at every checkpoint — intra-sample
tracking per user preference, replacing ad-hoc collage reading.

**B4 Seg-mask policy decision.** Whether `seg_weight` (taxonomy v2) is in
scope for the P1/PP redesign or stays off; if in scope, measure seg2 coverage
first (it was ~3.4k/70k in August — a coverage number, not a vibe). Flagged
as an explicit decision point (D4 below), not silently inherited.

**B5 Hegre contract (R1, carried from v1).** Hegre is excluded from all P1/PP
training; it remains the held-out cross-shoot identity substrate for the
Stage-E renderer gate. (FFHQ auraface is per-image; hegre's is
persona-averageable — mixing them is a conditioning confound regardless.)

**B6 Conditioning contract per stream, written down before P1 gates.**
Per stream (T5 text, DINO CLS, DINO patches, pose): role (semantic / identity
/ spatial / structure), marginal presence under the dropout profile, null
embedding, eval CFG combination — and the rule that **every CFG combination
used at eval has non-zero probability in training** (eval-regime = train-regime).

### Stage C — P1 as a gated research process

**C0 Gate instruments first (null-hypothesis discipline).** The instruments
are A1-A4 run on the val split: text-binding delta, CLS-binding delta,
memorization hit-rate, novel-CLS novelty score. Pre-register thresholds
*against the Stage-A baselines* (e.g., "text-binding ≥ Arm J's value",
"memorization hit-rate ≤ x vs baseline y") — instrument validation on
known-bad (P1-10k) and known-better (Arm J) models comes before any arm gate
rides on them.

**C1 Optional ablation: conditioning-profile arms (the "possibly ablation").**
Question: which dropout profile makes text load-bearing instead of
CLS-memorized? Single-variable arms at a *short* budget (proposal: 3-5k
steps, latent space, measured regime), candidates:
- **P1-D1 — raised text-only share** (e.g., p_text_only 0.25→0.45, CLS
  presence ↓): forces text to carry semantics more often.
- **P1-D2 — CLS-scarce** (p_dino_cls_only high + reduced all-kept share):
  CLS present ~20-25% — CLS becomes a sometimes-signal the model cannot lean
  on as a primary key.
- **P1-D3 — scheduled dropout** (CLS presence annealed over training):
  structure first via CLS, semantics forced later.
Gate: the C0 instruments at matched steps + structural quality on decoded val
samples + no-collapse. Exploratory-registered arms (screening), findings feed
the confirmatory P1 re-run.

**C2 Confirmatory P1 re-run (`latent-first-pretrain-v2`).**
- Gate frozen before launch, drafted from: text-binding ≥ baseline+δ,
  memorization hit-rate ≤ ceiling, novel-CLS generalization, decoded
  structural quality (G0-style gates on decoded val), no collapse.
- **Budget is a decision variable, not an inheritance.** Sizing options:
  10k (old, ~97h), 20k (~194h), 50k (~486h ≈ 20 days continuous 4090 —
  weeks of calendar under the 1k-segment policy). Decision point D3: pick
  after the C1 ablation shows the regime, and state the extrapolation factor
  honestly (feasibility-before-registration).

### Stage D — PP as a gated research process

Same questions, answered explicitly this time:
- **What data:** FFHQ train split only (B2), pixels; recon validation on the
  val split; hegre excluded (B5).
- **What dropout:** decided from C1's result — the pixel phase inherits the
  conditioning contract, it doesn't silently re-use the champion's profile.
- **Warm-start source:** the **gated** P1-v2 checkpoint only (never the
  tainted one); warm-start accounting verified on CPU (182/188 precedent).
- **Gates (draft):** conditioning-binding instruments (C0) must not regress
  vs the P1-v2 anchor; memorization ceiling holds in pixel space; G0 photoreal
  gates vs frozen calibration (the PP-criterion-3 lineage); quality vs Arm J
  champion at matched steps; no collapse.
- **Budget:** open, same 50k-step envelope discussion as C2. Pixel space is
  cheaper per step (~19 s/it champion-proven → 50k ≈ 264h).

### Stage E — Eidolon renderer (downstream, sketched)

Once PP-v2 is gated: the eidolon arm (AuraFace-LDA identity + z_g geometry,
per-dim basis, asymmetric CFG — Arm O's proven recipe) warm-started from
PP-v2, with the v1 plan's G1-G5 gates (identity retention vs mismatched-vector
null, yaw binding, disentanglement, quality, no collapse) and the hegre
cross-shoot persistence gate (eval-only, persona-level bootstrap). Written as
a separate plan when C/D produce numbers — registering it now would gate on
imaginary baselines.

## 3. Execution order and governance

1. Stage A is read-only/eval — it can start immediately on approval (GPU is
   free; scheduler claim per standing rule).
2. Stage B's split manifest + probe panel are git artifacts (soft data) —
   committed before any training; every loader grepped to consume them.
3. Each training stage gets the full §0 trio + tree registration +
   `check_arm_records.py` green **before** launch.
4. **Stale-governance repair needed now** (approval item): PROJECT_STATUS.md
   still says "gamma2 training on GPU" and predates PP's cancellation; the
   registry still holds `pixel-posttrain`. Record PP's cancellation/taint
   (state: blocked or terminal, verdict noting taint — not a science verdict)
   so no future tick reactivates it.

## 4. Decision points for Tim

- **D1 — Stage A first?** Confirm the forensic baseline runs before redesign
  (my recommendation: yes — cheap, and it makes every later gate honest). Or
  is the diagnosis taken as established and we skip to Stage B?
- **D2 — "Segmentation" reading.** I've read it as dataset splitting
  (train/val/never-touched-test + probe panel, B2/B3). If you also meant
  seg-mask loss weighting (B4), confirm scope.
- **D3 — Budget envelope.** Is ~50k steps per phase the real ceiling
  (≈486h latent / ≈264h pixel continuous — weeks of calendar under segments)?
  Or a different cap? Deferred-final until C1 ablation results, but I need
  the outer bound for registration.
- **D4 — C1 ablation scope.** Run the dropout-profile screening arms (D1/D2/D3,
  3-5k steps each) before committing to the confirmatory P1-v2? Or pick a
  profile by argument and go straight to C2?
- **D5 — Hegre exclusion (R1)** still holds for P1/PP (yes/no).

## 5. Risks / honest notes

- **The taint may be partly architectural, not just dropout:** a 1024-d
  per-image key at 45% presence may be memorizable at ANY feasible dropout
  rate. The C1 ablation is designed to detect exactly this (if D1/D2/D3 all
  fail the memorization ceiling, the lever is the CLS stream itself —
  dimensionality reduction, noising, or removal — which is a bigger redesign
  and must be surfaced, not patched around).
- **Text may be weak because captions are weak** (T5-large, short Gemma
  captions — Pitfall 15 + tangential research). If C1 shows text-binding
  can't reach Arm J's level under any dropout, the fix is the caption/encoder
  regeneration, not the schedule.
- 50k-step budgets multiply every silent-failure mode in the pitfalls list;
  all lifecycle hooks, deterministic pixel-stat divergence detection, and
  non-fatal instrumentation are mandatory, not optional.
