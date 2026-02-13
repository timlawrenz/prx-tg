## Why

`data/raw/` contains ~1M image files stored without extensions, and we need a reproducible way to materialize the subset of *approved* photos (from `https://crawlr.lawrenz.com/photos.json` pagination) as normal image filenames with correct extensions.

## What Changes

- Add a script that:
  - Pages through `https://crawlr.lawrenz.com/photos.json?page=N` until an empty result.
  - For each returned item, reads `filename` and checks for `data/raw/<filename>`.
  - Detects the image type from file bytes (e.g., jpg/png) and determines an extension.
  - Creates/updates a symlink at `data/approved/<filename>.<ext>` pointing to `data/raw/<filename>`.
- Ensure the script is safe to re-run (idempotent) and efficient for large datasets (no full scan of `data/raw`).

## Capabilities

### New Capabilities
- `approved-photo-symlinks`: Maintain `data/approved/` as a symlinked, extension-correct view of the approved photo list.

### Modified Capabilities

## Impact

- Filesystem: creates many symlinks under `data/approved/`.
- Network: downloads ~60k records via a paginated JSON API.
- Runtime considerations: must handle large volumes efficiently (stream pages; constant-memory processing; avoid `ls`/directory scans of `data/raw`).
