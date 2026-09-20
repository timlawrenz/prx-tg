# Brief 06 — Part 1 of 2: teaching the model what a face is, cheaply

**Step:** `latent-first-pretrain` — Scheme B Phase 1
**Date:** launched 2026-09-12; **in flight** (step 9,136 / 10,000 as of 2026-09-19)
**Verdict:** none yet — this is a *plan-and-approach* post, not a result
**Source:** `experiments/latent-first-pretrain/config.yaml` (header comment),
`research/results/vae_ceiling/RESULTS.md`
**Suggested tags:** `research`, `prx-tg`, `diffusion`, `machine-learning`
**Suggested thumbnail:** **an actual face** — a decoded text-driven sample from
step 9,000 (see the Visuals table in `README.md`). Do NOT use a diagram as the
lead image.

---

## Where this sits

prx-tg is the visualization part of the group's *"Getting a grip on human
identity"* work. This post is where the group's renderer adopts the two-stage
design that the identity projects imply: understand the structure cheaply first,
then spend the expensive compute only on the part that needs it. It cross-links
to the VAE-ceiling post (which forced the split) and to stratum-hq (whose 70k
enriched dataset feeds both stages).

## Plain-English summary (write this first, before any metric)

The previous post ended with a hard limit: the compressor at the front of the
pipeline softens faces no matter what. So we split the job in two.

**Part 1 — learn what a face *is*.** Work in the compressed space, where images
are ~12× smaller and training is cheap. This phase is not trying to produce a
beautiful photograph; it is trying to learn structure, anatomy, and the
relationship between a text description and a face.

**Part 2 — make it look real.** Once the model understands faces, retrain it in
full pixel space to render the texture — pores, hair, the way light sits on skin.
That is where photorealism is expected to come from.

This post is about Part 1, which is running now. It is deliberately a
*what-we-are-testing and why* post: Part 1 produces numbers about semantics, and
we have been explicit with ourselves that those numbers are **not** a proven
predictor of the final picture. We will not pretend otherwise.

---

## Block 1 — Why split the job

- **Hook** — The compressor ceiling from the previous experiment meant any
  perfect model would still output a softened face.
- **Setup** — Rather than accept that, separate the *semantic* work from the
  *textural* work and do them in different spaces.
- **Evidence** — This is what the published pixel-space models do: they
  post-train in pixel space specifically to recover texture. The two-stage design
  is a response to the measured ceiling, not a stylistic choice.
- **Turn** — The counterintuitive part is that the *cheap* phase comes first. It
  feels backwards to learn in a lossy space and then re-learn in the accurate one.
- **Meaning** — Semantics are expensive to learn per-pixel and cheap to learn in
  a compressed representation; texture is the reverse. Split accordingly.
- **Handoff** — So what exactly is Part 1?

## Block 2 — What Part 1 actually is

- **Hook** — "Latent pretraining" is jargon. Unpack it.
- **Setup** — Keep the same transformer body as the champion (768 wide, 18 layers,
  12 heads) and the same 70k dataset and conditioning stack, but change the
  *space*: **16-channel latents at 128×128** instead of 3-channel pixels at
  1024×1024 — roughly 12× smaller per image.
- **Evidence** — The token grid is deliberately preserved: a patch size of 2 over
  a 128×128 latent gives a **64×64 token grid, identical to the pixel model's
  16-pixel patch grid.** That keeps the representation-alignment step (REPA)
  comparable between phases — the latent hidden states are aligned against the
  same DINOv3 patch grid as before.
- **Turn** — The model predicts the **clean latent** directly (`x_prediction`)
  rather than a velocity, which changes what the loss is even asking for.
- **Meaning** — Phase 1 is a *transfer-friendly* experiment: same body, same
  conditioning, same alignment geometry — only the space changes.
- **Handoff** — Which means the comparison against the pixel-space champion is
  controlled. That matters.

## Block 3 — Keeping the comparison honest

- **Hook** — A phase change that also changes five other things proves nothing.
- **Setup** — Differs from the champion in exactly: `in_channels` (3 → 16),
  `patch_size` (16 → 2), prediction target (velocity → clean latent x0), data
  source (pixel shards → precomputed `flux_latent.npy`), and `latent_space: true`
  so the samplers decode through the autoencoder.
- **Evidence** — Same 70k data, same conditioning stack, same effective batch
  (4 × 64 accumulation = 256), same optimizer family (Muon at 3e-4), same
  alignment block.
- **Turn** — The honest limit: **a latent-space LPIPS is not comparable to a
  pixel-space LPIPS.** The number will look different and that difference is not
  a result.
- **Meaning** — Say this in the post explicitly. It is the single easiest way for
  a reader (or an author) to draw a false conclusion.
- **Handoff** — Here's what we're watching, and how.

## Block 4 — What we watch, and how we refuse to fool ourselves

- **Hook** — What would count as progress in a phase whose output is not the
  final product?
- **Setup** — Validation every 500 steps: reconstruction from real latents,
  text-only generation (the actual goal: text → face), a DINO-swap test, and a
  text-manipulation test that asks whether changing the prompt actually changes
  the image.
- **Evidence** — **n is small and must be stated:** reconstruction n=4,
  text-only n=20, text-manipulation n=5. Our own convention treats sub-0.05
  differences on such sets as within noise.
- **Turn** — Because of that, we now also **stop at every 1,000-step boundary and
  look at the images**, rather than trusting the numbers alone.
- **Meaning** — **Do not report a within-noise move as a result.** If the last
  segment's metrics were flat, the post should say they were flat.
- **Handoff** — Which brings in the other half of the goal: pose.

## Block 5 — Two modes, one model

- **Hook** — The model must be good in two different usage modes, and only one of
  them is the common case.
- **Setup** — Pose conditioning is already wired in: 133 pose joints, with pose
  dropped on 10% of training steps and pose-only conditioning on 5% — a deliberate
  dropout schedule so the model works with *and* without a pose given.
- **Evidence** — The more frequent real-world use case is **without** pose: a
  plain text-to-image request. The scientifically interesting case is **with**
  pose, where geometry is controlled explicitly.
- **Turn** — This makes it possible to win the interesting case while quietly
  degrading the common one — so **a pose-conditioned win that costs no-pose
  quality is not a win.**
- **Meaning** — Both regimes get evaluated, always. (Also worth noting as a group
  thread: the sister face-vectors work is where pose and identity were
  disentangled into separate channels in the first place.)
- **Handoff** — And a practical note about how this work is actually scheduled.

## Block 6 — Sharing one GPU, in 1,000-step slices

- **Hook** — A 10,000-step run at ~37 s/step is about 4 days of a single GPU.
- **Setup** — Instead of reserving the machine for the whole run, the work is now
  scheduled **only until the next 1,000-step boundary**, then released.
- **Evidence** — At each boundary the trainer is stopped cleanly, a checkpoint is
  written, and the GPU is handed back to a queue shared with the group's other
  projects. If nobody else has asked for it, the run simply claims it again and
  continues.
- **Turn** — This costs restarts, and it means the run finishes later in
  wall-clock terms than an uninterrupted one would.
- **Meaning** — It buys two things: other projects are never starved by a
  multi-day hold, and the team gets a **mandatory visual review every 1,000
  steps** instead of one look at the end. (This is the same instinct as the
  original arm's daily-interrupt schedule — now formalised.)
- **Handoff** — Next: Part 1's result, then Part 2 — the phase where
  photorealism is actually expected.

---

## Caveats to carry into the post — read these before drafting

- **This is IN FLIGHT.** As of 2026-09-19 the run is at step 9,136 / 10,000. Do
  not write it as a completed result. Frame it as approach + what we're testing.
- **Part-1 LPIPS is NOT a validated predictor of final quality.** The correlation
  is a strong hunch, explicitly not a guarantee. Never rank checkpoints or imply
  the end product is characterised from these numbers.
- **Latent-space ≠ pixel-space.** Do not compare this run's LPIPS to the champion's
  pixel-space numbers anywhere in the post.
- **Sample sizes are small:** recon n=4, text-only n=20, text-manip n=5. Sub-0.05
  is noise.
- **Pose is a dual-mode requirement**, not an add-on. No-pose is the more frequent
  use case; with-pose is the more interesting one. Evaluate both.
- Two-part structure: **Part 1 buys semantics and structure. Photorealism comes
  from Part 2.** Say so, and don't let the post imply otherwise.
- If the metrics moved within noise over the reported window, **say they were
  flat**. Honest flatness is better than a manufactured trend.

## Artifacts

- Config + header comment: `experiments/latent-first-pretrain/config.yaml`
- Run: `experiments/latent-first-pretrain/runs/2026-09-13_2225/`
  (checkpoints every 500 steps, `validation/step*/results.json`, `visual_debug/`)
- Segment-break artifacts: `validation/step0008500/`, `step0009000/` (results.json
  + decoded sample images for the visual review)
- Precomputed latents: `/mnt/nas-ai-models/training-data/ffhq/stratum/*/flux_latent.npy`
  (fp16, 16×128×128), generated by `scripts/generate_flux_latents.py`
- Motivation: `research/results/vae_ceiling/RESULTS.md`
