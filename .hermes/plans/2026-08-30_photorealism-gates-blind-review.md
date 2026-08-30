# Photorealism Gates + Blind Review — prx-tg Quality Line

**Date:** 2026-08-30
**Status:** proposal (pre-registration to follow per arm)
**Trigger:** "We have not yet produced a model that creates a portrait that is anywhere
close to photo realistic." → The evaluation target is photorealism. Measurement must
precede training; no further arms are judged by eyeball alone.

## Hard facts anchoring this plan (verified 2026-08-30)

- 70,000 FFHQ stratum dirs, full v1 satellites: `caption.txt` + `t5_hidden.npy`
  (t5-large, 512×1024), dinov3 CLS+patches, `pose.npy` (133kp), `seg.npy`, z_g,
  auraface_lda. **`caption2.txt`/`t52_hidden.npy` are stratum2 experiment artifacts —
  NOT reliable captions. Do not use.**
- hegre corpus 37,011 dirs: pixel + auraface_lda + z_g only (no captions/T5).
- Stratum2 pose2/seg2 present in ~4% of FFHQ (2,958 / 3,373 dirs) and still being
  enriched on the 4090. Missing stratum2 artifacts CAN be produced on demand.
- Best-performing architecture per own ablations: no DINO patches (Arm M) — but only
  validated on the 7k subset; never run at 70k scale.
- Arm O (`zg-token-basis-cfg-guard`) finished 5,000 steps 2026-07-23 with its
  pre-registered gate satisfied (yaw sweep + no collapse); verdict never recorded.
  Governance close-out pending.
- Release checkpoints: `prx_tg_armj_faces70k_step35000_bf16.safetensors`,
  `prx_tg_armk_dino_step5000_bf16.safetensors` (in `release/`).
- GPU context: 4090 currently busy with stratum enrichment (do not disturb);
  training arms must be cleared with tim first.

## Goal restated

Produce a model that generates **photo-realistic portraits**. Photorealism is
established by (a) deterministic automatic gates tested against a real-image
reference distribution, and (b) a pre-registered blind human preference test.
A model is "photorealistic" only when it passes the gates AND wins the blind
review. Nothing else counts.

## Phase 0 — Governance close-out (no GPU)

1. Record Arm O PASS in `docs/EXPERIMENTS_AND_RESULTS.md` + tree
   (verdict: PASS on pre-registered gate; evidence: step-5000 sweep yaw rotation
   monotonic, recon LPIPS 0.819, no collapse).
2. Update `PROJECT_STATUS.md` (6 weeks stale).
3. Commit all untracked governance docs + scripts (several runs logged
   `git_dirty: true`).
4. Strip Arm O final checkpoint → browsable BF16 safetensors (existing recipe).

## Phase 1 — Build the measurement, then measure (no training)

Deliverable: `scripts/eval_gates/` producing `results.jsonl` per checkpoint, and
`scripts/blind_review/` generating the human-vote pool. Retro-run both on the
release checkpoints + a real-FFHQ reference set → first **photorealism deficit
report** (the honest baseline). PASS thresholds are calibrated from the real-image
distribution BEFORE any model is judged (null-baseline rule); thresholds are not
read off the first model results.

## Automatic gate suite (deterministic, no LLM certifying LLM)

Every gate gets: vacuous-pass guard, detector self-check vs a known-bad corpus
(flux_nsfw), null baseline, and per-gate unit test of the metric code.

### G0 — Photorealism distribution gates (vs ~2k real FFHQ reference)

| Gate | Measurement | Why it catches fakes |
|---|---|---|
| G0a sensor-noise floor | wavelet high-pass residual std in flat skin regions (seg mask); log-std vs real band [p5,p95] | real cameras always have sensor noise; predicted x0 has essentially zero → "too clean / plastic" |
| G0b spectral slope | log-log power-spectrum slope of luminance, 0.1–0.9×nyquist | GAN/DiT outputs show characteristic spectral deviations |
| G0c skin texture energy | bandpass (0.25–0.75×nyquist) energy inside face skin mask, normalized | catches "airbrushed/doll-smooth" signature |
| G0d local contrast | histogram of 8×8 RMS contrast in midtone skin regions | catches flat AI-gradient tone mapping |

### G1 — Structural plausibility (extend pony-ft gate list, finalized)

- Hand/finger count (on hand prompts, DWPose 21-pt; vacuous-pass: ≥1 hand found)
- Face count + symmetry (front-facing prompts; mirror displacement ≤ tolerance)
- Anthropometry: joint angles + L/R ratios within ~2σ of real distributions
- Upgradable to 308-kp pose2 checks once pose2 enrichment covers the eval sources
- Determinism sanity: same prompt+seed twice, structural match required

### G2 — Text compliance (already partially built)

- CLIP score (exists in quality_metrics); attribute-binding manipulation delta
  (existing text-manip test) auto-measured.

### G3 — Mode-collapse / memorization

- Density & coverage (Naeem et al.) on DINOv3 CLS embeddings of eval outputs vs
  real reference (sample-efficient vs FID; ~200+ generations). Directly tests the
  old question of whether long FFHQ training overfits/memorizes.

## Blind human-preference review process

Goal: human photorealism + quality preference with zero label leakage.

1. **Pool construction:** fixed eval suite — never-trained prompts × fixed seeds ×
   candidate checkpoints, mixed with **hidden real photos** from a held-out FFHQ
   slice (calibration ladders).
2. **Pairing:** randomized pairs, random left/right placement, filenames opaque;
   render at identical resolution/exposure scale.
3. **Two separate question blocks** (separate sessions, never mixed):
   **Q1 "which looks more like a real photograph?"** (photorealism) and
   **Q2 "which is better quality?"** (quality) — kept apart because attractive-but-
   fake faces corrupt realism judgment (the known evaluation bias).
4. **Raters:** ≥3 humans; inter-rater agreement reported (Cohen's κ). Any
   disagreement larger than noise → that prompt subset is exploratory, revisit.
5. **Calibration self-check:** real photos must be chosen as "more real" against
   model images in ≥95% of pairs where they appear, or the display/rater pipeline
   is invalid and results are discarded.
6. **Pre-registered decision rule:** candidate PASSes the blind review only if its
   realism win-rate vs the release baseline has a binomial-CI lower bound > 0.5,
   and PASSES G0–G3. Vote records as JSONL (mirrors the persona-search
   `gen/vote-archive` pattern; reuse the existing voting machinery where possible).
7. **Reviewers see only shuffled pairs; the recorder knows the truth only after all
   votes are in.** No mid-review reveals.

## Phase 2 — First measured training arms (one variable each, on 70k stratum)

After the deficit report exists, cheapest-first, each gated by G0–G3 + blind review:

1. **γ=2 noise scale** (one-line `flow_matching_loss` change, Z-Image-calibrated
   for pixel space). Baseline: current γ=1 on the same 70k config. — cheapest
2. **DiP-style conv head** vs linear head (Z-Image: removes grid artifacts)
3. **iREPA** (Conv2d projection + spatial norm) — proven FID gains
4. **QK-RMSNorm** — attention stability (Ideogram 4)

All run at 5k steps on the 70k stratum no-DINO-patch config for comparability;
each arm's gates are pre-registered before launch.

## Phase 3 — Data levers (enrichment/captioning, no architecture change)

- Spatial-fact caption enrichment pass (H3 lesson: captions verbalize the same
  structural facts the conditioning encodes) — requires a VLM run over FFHQ;
  feasible on existing stratum pipeline.
- hegre corpus captioning pass if the aesthetic target needs non-FFHQ data.
- Self-generated + real 1:1 mix (Z-Image) if adaptation quality stalls.

## Phase 4 — Structural escalation (only if Phase 2–3 plateau below photoreal band)

- Latent-first pretraining → pixel post-training (Z-Image headline result; the
  quantified fix for slow pixel-from-scratch convergence). Pipeline change;
  decision to be made with measured Phase 2/3 numbers in hand, not before.
- Scale: 0.8B backbone on rented GPUs remains the last lever, per known
  industrial training economics.

## Open decisions (tim)

1. Greenlight Phase 0 + Phase 1 build (no GPU needed; ~1 session of work).
2. Eval suite size: recommend 30 prompts × 8 seeds (~240 images/arm) to make G3
   statistically usable.
3. Rater pool: who are the 2+ additional raters?
4. **Enrichment priority:** pose2 to 100% FFHQ (feeds G1 at 308kp) vs caption-quality
   pass first (feeds the text stream)? With photorealism-first framing, the
   caption pass is the stronger first move.

---

# Phase 1b — Research Loop (stratum-ffhq harness transfer)

**Status:** APPROVED 2026-08-30 (tim). Scoped from the stratum-ffhq autonomous
research harness (`stratum-research-harness` skill, worktree
`stratum-hq-stage-b-experiment`) — transplant the *process*, not the build.
prx-tg's atoms are multi-hour 5k-step training arms, not 40-minute review runs,
so cadence is **event-driven** (tick on run completion), not wall-clock.

## Design rules carried over verbatim

1. **Atomic deterministic ticks.** Verdicts are computed by code from gate
   JSONL files, never transcribed from memory or eyeball. A `--write` tick
   refuses to run if the registry mutated mid-tick ("changed on disk").
2. **One-active invariant.** Exactly one `active` arm at any time; a NOT_BETTER
   verdict below the falsification limit records a strike and keeps the SAME
   arm active. Two active arms = hard failure (maps to exactly one training
   arm on the 4090).
3. **Selection is arithmetic.** EIG = prior × measurability − cost −
   strikes × 0.45, with the harness weights (prior high 1.0/med 0.6/low 0.2;
   measurability high 1.0/med 0.6/low 0.3; cost low 0/med 0.15/high 0.35).
   Plus ε-greedy forced exploration slot every N selections and a novelty
   bonus for candidates naming a NEW measurable evidence part (a G-gate
   axis no terminal arm has established). Ties broken by id; full score
   table recorded every selection.
4. **Blocked ≠ active.** Policy-gated arms (needs own ruling, needs human
   votes) are `blocked`, and the selector skips them while evidence-side
   feeders stay autonomous.
5. **No-op cycles are correct.** The loop honestly says "nothing actionable"
   instead of manufacturing churn.
6. **Producer ≠ certifier.** The gate suite (deterministic code) certifies;
   the run produces; the tick reads files.

## Registry — `research/avenues/registry.json` (schema v1)

```json
{
  "schema_version": 1,
  "champion": {
    "slug": "faces70k-fp8",
    "checkpoint": "release/prx_tg_armj_faces70k_step35000_bf16.safetensors",
    "measured": {}
  },
  "selection_progress": 0,
  "exploration": {"every_n": 5, "novelty_bonus": 0.25},
  "falsification": {"strike_limit": 3},
  "cost_tiers_gpu_hours": {"low": 30, "med": 60},
  "candidates": [
    {
      "id": "gamma2-noise-scale",
      "state": "registered",
      "prior": "high", "measurability": "high", "cost": "med",
      "strikes": 0,
      "evidence_parts": ["g0a_sensor_noise_floor", "g0b_spectral_slope"],
      "declaration": {
        "scope": "one-line: replace γ=1 rectified-flow noise scale with γ=2",
        "differs_from": "champion",
        "output_semantics": "what must change + which gate measures it",
        "provenance": "docs/tangential-research.md 2026-08-18 Z-Image-Turbo §1",
        "abstention": "prediction horizon / known limits",
        "qualification_gate": "blind win-rate CI>0.5 vs champion AND ≥1 G0 band gain AND no G1 regression",
        "expected_gpu_hours": 30,
        "config": "experiments/{slug}/config.yaml",
        "arm_issue": 0
      },
      "verdicts": []
    }
  ]
}
```

Candidate states: `registered` → `active` → `terminal_validated` |
`terminal_falsified` | `blocked`.

**Cost tiers** are derived from measured run rates (Arm I ≈ 26h FP8 / 5k steps
on 7k; 70k-stratum rate to be measured at first launch, not estimated by fiat).

## Tick data flow

1. `run_gates.py` — producer: checkpoint → `gates/{arm}/{step}/gates.jsonl`
   (all automatic gates; existing quality-metrics scripts compose into this).
2. Blind review (human vote session, when the arm's gate requires it) →
   `votes.jsonl`.
3. `tick.py --registry ... --gates <dir> [--votes <file>] --write`:
   * changed-on-disk guard → load registry
   * gate checks vs **frozen calibration file** (`gates_calibration.json`,
     band thresholds derived from the real-FFHQ reference set — frozen BEFORE
     the first verdict, never recomputed mid-sweep; a moving band is HARKing)
   * verdict object: validated / not_better(+strike) / falsified / needs_human
   * one-active invariant + strike path (third strike → falsify → select next)
   * champion advancement only via pre-registered rule (see registry qual gate)
   * selection: EIG arithmetic + explore slot + novelty bonus, full score table
   * atomic tmp+rename write; emits verdict JSON — the agent relays it and
     writes the ledger entry from it, never its own numbers
4. Ledger reconciliation remains a separate, explicit task (docs lag registry —
   harness doctrine).

## Proposal gate — `propose.py`

`propose.py --registry ... --candidates <json> [--require-new-evidence-part] --write`

Rejection rules (each a hard error):
- missing any declaration field (scope, output_semantics, provenance,
  abstention, qualification_gate, expected_gpu_hours, config)
- **redundant evidence**: declares no measurable axis beyond what terminal
  arms already established → rejected (stops attribute-tagger spam and
  unresolvable arms)
- `config` file missing on disk (AGENTS.md: named config, frozen into
  experiment dir)
- no `arm_issue` (routing state — **the repo HAS a remote:
  github.com/timlawrenz/prx-tg**, so `arm_issue` = a real GitHub issue number,
  created at proposal time, mirroring the harness's label-sync mandate)
- unregistered gate in qualification_gate (a trick is payable only if an
  existing gate can measure it; new gates must be registered + calibrated
  FIRST — this is the mechanism that stops attractive-mask artifacts at the
  selection level)

## Human holds (the only three)

1. GPU budget authorization for runs beyond the pre-registered cap
   ("let me know BEFORE you train" rule, process-ified).
2. Blind-review voting sessions (inherently human) → tick resumes on votes.
3. Phase-4 architectural escalations (latent-first pretraining, scale).

## Build order (est. ~2 agent-days, no GPU until M5)

| Step | Artifact | Est. |
|---|---|---|
| M1 | `research/avenues/registry.json` schema + `validate_registry()` + seed from tangential-research/TBD candidates | ~½d |
| M2 | gate calibration pipeline: reference-set stats → frozen `gates_calibration.json` | ~½d |
| M3 | `tick.py` state machine + unit tests (strike path, one-active invariant, explore slot, novelty bonus, third-strike falsify, mid-tick mutation refusal — the harness's regression suite names these tests) | ~½d |
| M4 | `propose.py` + rejection rules + tests | ~⅓d |
| M5 | event-driven strategist cron in THIS profile (poll run-completion → tick-ready marker → relay verdict JSON; prompt registers proposals via propose.py; never hand-types verdicts), modeled on the harness's observer/tick-ready pattern | after M1–M3 verified |

M1–M4 are pure files/no-GPU and can start immediately.