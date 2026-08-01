# pose2 — 308-Keypoint Sapiens 2 Pose (Single-Variable Ablation)

## Hypothesis

308-keypoint Sapiens 2 pose (274 face landmarks vs 68 in DWPose) improves face generation quality on FFHQ. More granular facial landmarks teach the model finer eye, lip, nose, and jaw geometry during training, which transfers to text-only generation at inference.

## What Changed (Single Variable)

- **Pose:** `pose2.npy` (308 keypoints, Sapiens 2 + DETR) replaces `pose.npy` (133 keypoints, DWPose). Coordinates normalized from absolute pixel space to [-1, 1].
- `num_pose_joints`: 133 → 308 (model + adapter)

## What Did NOT Change

- seg_weight: disabled (same as production)
- geometry_3d: not loaded, not enabled
- matting_edge: not loaded, not enabled
- All hyperparameters: identical to production config

## Fallback Behavior

Images without `pose2.npy` use `pose.npy` (133kp). The model's `pose_joint_embed` has 308 entries; v1 images use the first 133. This is a known confound — for a fully clean ablation, re-run once pose2 enrichment completes on all 70k images.

## Status

- [x] Code implemented (commit 10dd28c)
- [x] Config frozen (single variable: pose2 only)
- [x] Provenance recorded
- [ ] Sanity test
- [ ] Training run
- [ ] Evaluation
