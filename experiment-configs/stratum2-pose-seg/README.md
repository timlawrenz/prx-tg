# Arm B: Stratum2 Pose2 + Seg2

## Hypothesis

308-keypoint Sapiens 2 pose (274 face landmarks vs 68 in DWPose) combined with 29-class segmentation loss weighting (granular mouth/teeth/tongue weighting) improves face generation quality on FFHQ.

## What Changed

- **Pose:** `pose2.npy` (308 keypoints, Sapiens 2 + DETR) replaces `pose.npy` (133 keypoints, DWPose). Coordinates normalized from absolute pixel space to [-1, 1].
- **Seg loss:** `seg2.npy` (29-class DOME_CLASSES_29) replaces `seg.npy` (28-class). New taxonomy enables granular weighting: mouth=4.0, face_skin=3.0, hair=2.0, bg=0.5.
- **Fallback:** Images without v2 artifacts automatically use v1 data (133kp pose, 28-class seg).

## Expected Outcome

- Improved facial geometry in generated faces (more precise eye, lip, nose rendering)
- Improved mouth/teeth rendering (the highest-failure region for face generation)
- Aesthetic score trajectory should be above baseline by step 5000

## How it Differs from Baseline

Same dataset (FFHQ 70k), same model architecture (238.8M params), same training hyperparameters (Muon, TREAD, REPA, AsymFlow). Only the conditioning signals (pose2, seg2) and loss weighting (29-class with granular weights) are changed.

## Status

- [x] Code implemented (commit 10dd28c)
- [x] Config frozen
- [x] Provenance recorded
- [ ] Sanity test (50 steps)
- [ ] Training run
- [ ] Evaluation
