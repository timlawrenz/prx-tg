# prx-tg Blog Briefs — for the lawrenz.com blog writer

Briefs for turning prx-tg scientific progress into lawrenz.com posts. **These are
research-side briefs, not published posts.** The blog writer converts each brief
into a post; nothing here goes live directly.

## Cadence (decided 2026-09-19)

**ONE post per arm verdict (`[CONCLUDED — GO/KILL]`) or phase milestone.** Not per
training segment — 1k segment breaks are internal checkpoints, not conclusions.

## Where prx-tg sits

prx-tg belongs to a group with **eidolon / morphometrics / stratum-hq / bt-net**,
whose shared theme is *"Getting a grip on human identity."* prx-tg is the
**visualization part** of that pipeline — the renderer that turns the group's
identity and geometry work into images. Posts must sit inside that group
narrative and cross-link to the sister projects; they should not read as a
standalone model-training diary.

## Source artifacts (where a "step" comes from)

| Artifact | What it holds |
|---|---|
| `docs/EXPERIMENTS_AND_RESULTS.md` | Per-arm entries with pre-registered gates, Empirical Evidence, Adversarial Pass, Verdict, Artifacts |
| `docs/EXPERIMENT_TREE.md` | Active / Concluded / TBD branches — which we follow vs cut off |
| `research/avenues/registry.json` | Arm states (active/blocked/registered) + strike counts |
| `research/results/` | Research syntheses and standalone experiments |

## Series state

Publication order is **chronological** (the narrative builds). I worked the
backlog backward from the most recent step.

| # | Brief | Step date | Verdict | Status |
|---|---|---|---|---|
| 01 | `brief-01-fp8-scaling-to-70k.md` | 2026-06 | GO (scaling) | brief ready |
| 02 | `brief-02-dino-patches-counterproductive.md` | 2026-06-19 | GO — counterintuitive | brief ready |
| 03 | `brief-03-dip-conv-head-10k.md` | 2026-09-01 | GO-with-caveat, strike 1/3 | brief ready |
| 04 | `brief-04-gamma2-noise-scale.md` | 2026-09-10 | NOT_BETTER + code post-mortem | brief ready |
| 05 | `brief-05-vae-ceiling.md` | 2026-09-10 | Constraint discovered | brief ready |
| 06 | `brief-06-latent-first-pivot.md` | 2026-09-12 → | in flight (Part 1 of 2) | brief ready |

**Unblogged before this series:** the last prx-tg post is
`2026-06-17-introducing-stratum-ffhq-multi-modal-enriched-face-dataset.md`. Also
`_posts/draft_prx_tg_seg_weighting.md` ("The Perils of Spatial Loss Weighting in
Pixel-Space Diffusion") exists in the blog repo and was never published — decide
whether it joins this series or is retired.

## Non-negotiables for every post

1. **Visuals: actual faces first.** Tim's preference is explicit — **prefer real
   face renders over diagrams.** Lead each post with model output, and where the
   point is a comparison, pair model output against a real photo. A diagram is
   supporting material for an architecture or two-stage-plan point — never the
   lead image. Every image needs `{: style="width: 100%;"}`.
2. **Human framing first.** A plain-English account of what we tried and
   concluded *before* any metric appears. A non-scientist should be able to
   follow the first three paragraphs.
3. **An explicit "what's next" section.** Every post ends by pointing at the
   next step.
4. **The goal stays visible.** A text-to-image model producing high-quality human
   photos, good **both with and without pose conditioning** — no-pose is the more
   frequent real-world use case (primary product surface); with-pose is the more
   scientifically interesting one.
5. **prx-tg is a TWO-PART program.** Part 1 = latent-space semantic pretrain
   (current `latent-first-pretrain`). Part 2 = pixel-space post-training, which is
   where photorealism comes from. **A Part-1 post must never imply the end product
   has been characterised.**
6. **Numbers discipline.** Never lead with LPIPS. Give the sample size `n` with
   every number. Keep latent-space and pixel-space values clearly separated —
   they are not comparable. Never present a within-noise move as a result
   (recon LPIPS at n=4 is a small sample; treat sub-0.05 as noise).

## Visuals — verified paths (faces, checked to exist 2026-09-19)

Every brief has real face images on disk. Use these; do not settle for a diagram.

| Brief | Face visuals (verified) |
|---|---|
| 01 FP8@70k | `release/demo_output/` — 7 renders: `demo_seed137_armj_cfg3.png`, `demo_seed137_armj_ncfg.png`, `demo_seed137_armj_selfg3.png`, `demo_seed137_cfg1.png`, `demo_seed137_cfg3.png`, `demo_seed137_cfg5.png`, `demo_seed137_ncfg.png` |
| 02 DINO K/L/M | **Controlled 3-way comparison at matched step + prompts** — `experiments/spatial-window-baseline/runs/2026-06-20_1125/validation/step0005000/text_only/` (20), `experiments/spatial-window-2/runs/2026-06-22_0010/validation/step0005000/text_only/` (20), `experiments/no-dino-patch-ablation/runs/2026-06-23_1110/validation/step0005000/text_only/` (20). Same prompt index across arms = a fair K/L/M figure. ⚠️ `visual_debug/` is **empty** in all three arms — the faces live under `runs/<ts>/validation/`. |
| 03 dip-conv-head | `experiments/dip-conv-head/runs/2026-08-30_1449/validation/step0010000/{text_only,reconstruction,dino_swap,text_manip}/`; plus **the actual blind-review pairs** — `research/avenues/review_10k/realism/pairs/realism_ab_*.png` (30 files). These pairs *are* the artifact the post is about. |
| 04 gamma2 | `experiments/gamma2-noise-scale/runs/2026-09-04_1440/visual_debug/step0010000/` — `collage.png` + `sample00..sample03_*.png` (5 files). Pair the step-10000 face with a real photo to make the "too dark" point. |
| 05 VAE ceiling | `research/results/vae_ceiling/samples/` — real-vs-decoded triples: `000_real.png`, `000_flux_full1024_s2.76932.png`, `000_sdxl.png` (also `003_*`, `007_*`). The ideal figure: same face, real → FLUX decode → SDXL decode. |
| 06 latent-first | `experiments/latent-first-pretrain/runs/2026-09-13_2225/validation/step0009000/text_only/` (20 faces) + `reconstruction/` (4). Decoded, text-driven output from Part 1. |

Also available across all six: the per-step `reconstruction/` sets for
before/after timelines (same prompt index at successive steps).

### Image compliance — mandatory, read before adding any face

- **Real FFHQ photographs** (brief 03's calibration pairs, brief 05's `*_real.png`,
  any output-vs-real figure) require attribution: **CC BY-NC-SA 4.0 by NVIDIA
  Corporation**, in a compact `<small>` line (not a blockquote), **listing the
  FFHQ image IDs in full**.
- **Never include images from the proprietary multi-shoot dataset.** Only FFHQ.
- **Never write the name "Hegre"** in public material — "proprietary dataset".
- **`stratum-ffhq` is Tim's product, not NVIDIA's.** Link to the HuggingFace
  dataset and the `stratum-hq` repo on first mention.
- Model *outputs* are not FFHQ images, but any figure that pairs output with a
  real photo pulls the attribution requirement in.
- **Scope discipline:** do not present Part 2 or any unbuilt pipeline stage as
  existing. Part 2 is the *next step*, described as planned.

## Narrative arc convention (Tim checks this explicitly)

Every narrative block gets its own **six-beat bullet arc**:

```
- **Hook** — the question or tension that makes this block worth reading
- **Setup** — what we actually did
- **Evidence** — the measured facts, with n
- **Turn** — what surprised us / what we got wrong
- **Meaning** — what it changes about the plan
- **Handoff** — link to the next block
```

A block with thesis prose but no arc bullets is incomplete — the arcs are the
draft skeleton.
