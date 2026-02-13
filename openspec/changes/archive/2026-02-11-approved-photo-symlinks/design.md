## Context

We have a large local corpus of images at `data/raw/` (~1M files) stored *without* file extensions. Separately, there is a paginated API of **approved** photos at `https://crawlr.lawrenz.com/photos.json` where each item includes a `filename` that matches a raw file on disk.

We need an efficient, repeatable process to create an “approved view” under `data/approved/` containing symlinks named with the correct file extension (e.g., `data/approved/<filename>.jpg -> data/raw/<filename>`), without scanning `data/raw/`.

Constraints:
- The approved list is ~60k items (API pagination until an empty response).
- `data/raw/` is too large to enumerate repeatedly.
- Image type is guaranteed to be something like JPEG/PNG (possibly others), but raw filenames have no extension.

## Goals / Non-Goals

**Goals:**
- Stream the approved-photo API page-by-page and process each photo independently.
- For each approved `filename`:
  - If `data/raw/<filename>` exists, detect its image type via file “magic bytes” and choose an extension.
  - Create or update a symlink at `data/approved/<filename>.<ext>` pointing to `data/raw/<filename>`.
- Make the process idempotent and safe to re-run.
- Keep runtime and memory bounded (no global in-memory set of all raw files).

**Non-Goals:**
- Removing “stale” symlinks in `data/approved/` that are no longer approved (unless explicitly added later as an option).
- Renaming/moving raw files or mutating `data/raw/`.
- Downloading image bytes from `exportable_url` (we rely on local `data/raw/`).

## Decisions

1) **Implementation as a standalone script (no full project build tooling)**
- Implement as a small script (likely Python) that uses only the standard library for HTTP + JSON + filesystem operations.
- Rationale: the repo currently has no established application runtime; a single script keeps operational complexity low.

2) **Pagination strategy**
- Start at page 1 and increment `page=N` until the API returns an empty JSON array.
- Add retry/backoff for transient HTTP failures (timeouts, 5xx, 429).
- Optional flags for operability:
  - `--start-page N` / `--end-page N`
  - `--limit M` (for smoke tests)
  - `--dry-run`

3) **File existence checks without scanning `data/raw/`**
- For each approved filename, test existence directly: `data/raw/<filename>`.
- Rationale: O(approved) `stat()` calls is acceptable (~60k) and avoids O(raw) directory enumeration (~1M).

4) **Image type detection (magic bytes)**
- Read a small prefix (e.g., first 16–32 bytes) from the raw file and detect type by signatures:
  - JPEG: `FF D8 FF`
  - PNG: `89 50 4E 47 0D 0A 1A 0A`
  - GIF: `47 49 46 38`
  - WEBP: `RIFF....WEBP`
- Map detected types to extensions: `jpg`, `png`, `gif`, `webp`.
- If unknown: record and skip (or put into an `unknown` report).
- Rationale: avoids external deps; fast; deterministic.

5) **Symlink creation/update semantics**
- Ensure `data/approved/` exists.
- Target path: `data/raw/<filename>`.
- Link path: `data/approved/<filename>.<ext>`.
- Idempotency rules:
  - If link path doesn’t exist: create symlink.
  - If link path exists and already points to the correct target: do nothing.
  - If link path exists but points elsewhere (or is a regular file): replace it.
  - If an old link exists with a different extension for the same base filename, leave it alone for now (see Open Questions) unless we choose to “normalize” later.
- Prefer creating **relative symlinks** (e.g., `../raw/<filename>`) to improve portability if the repo is moved.

## Risks / Trade-offs

- **[API rate limiting / transient failures] → Mitigation:** retry with exponential backoff; allow resuming via `--start-page`.
- **[Unknown/rare image formats] → Mitigation:** add a small, extendable signature table; log unknowns for follow-up.
- **[Partial runs] → Mitigation:** idempotent symlink updates; reruns converge to the desired state.
- **[Filesystem performance creating many symlinks] → Mitigation:** avoid extra syscalls; optionally batch logs; keep per-item work minimal.

## Migration Plan

- Run the script once to populate `data/approved/`.
- Re-run anytime the approved list changes.
- Rollback is simply removing `data/approved/` symlinks (does not affect `data/raw/`).

## Open Questions

- Pruning is out of scope for this change for now (potential future addition via `--prune`).
- If a base filename already has a symlink with the wrong extension from a prior run, should we remove/replace the old extension variant automatically?
