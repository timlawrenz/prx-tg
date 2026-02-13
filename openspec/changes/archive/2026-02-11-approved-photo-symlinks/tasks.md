## 1. Script scaffolding

- [x] 1.1 Add a runnable script (e.g., `scripts/sync_approved_photos.py`) and ensure it can be invoked from repo root
- [x] 1.2 Ensure `data/approved/` is created if missing

## 2. Approved list ingestion

- [x] 2.1 Implement pagination over `https://crawlr.lawrenz.com/photos.json?page=N` until an empty JSON array
- [x] 2.2 Parse each item and extract `filename` (ignore other fields)
- [x] 2.3 Add basic retry/backoff for transient HTTP failures (timeouts/5xx/429)

## 3. Raw file checks (no directory scan)

- [x] 3.1 For each approved `filename`, check existence of `data/raw/<filename>` via direct path lookup (no `ls`/full directory enumeration)
- [x] 3.2 Skip symlink creation when the raw file is missing

## 4. Image type detection

- [x] 4.1 Implement magic-byte detection for at least JPEG and PNG (optionally GIF/WEBP) and map to extensions (`jpg`, `png`, ...)
- [x] 4.2 If an image type is unknown/unhandled, skip creating a symlink for that file

## 5. Symlink creation/update (idempotent)

- [x] 5.1 Create relative symlinks: `data/approved/<filename>.<ext>` -> `../raw/<filename>`
- [x] 5.2 If the symlink already exists and points to the correct target, leave it unchanged
- [x] 5.3 If the destination path exists but is wrong (wrong target or regular file), replace it with the correct symlink

## 6. Verification / smoke checks

- [x] 6.1 Add a small smoke-run mode (e.g., `--limit` or `--end-page`) to validate behavior without processing all pages
- [x] 6.2 Manually verify on a few known filenames that the created link has the correct extension and points to the raw file

## 7. Documentation

- [x] 7.1 Document how to run the script and the supported flags (and explicitly note that pruning is not performed in this change)

## 8. Progress reporting

- [x] 8.1 Add progress reporting (current page + periodic counters)
