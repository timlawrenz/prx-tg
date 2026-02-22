#!/usr/bin/env python3
"""Sync approved photos to `data/approved/`.

Fetches the paginated approved photo list from:
  https://crawlr.lawrenz.com/photos.json?page=N

For each item:
1. Checks for raw file at `data/raw/<filename>` (no extension)
2. If missing, downloads from `exportable_url` to `data/raw/<filename>`
3. Detects file type from magic bytes
4. Creates/updates symlink: data/approved/<filename>.<ext> -> ../raw/<filename>

Notes:
- Automatically downloads missing files from exportable_url (CDN)
- In dry-run mode, downloads are skipped (only reported)
- Use --no-prune to skip removal of stale symlinks and their derived data
- Does NOT enumerate `data/raw/`; uses per-filename path lookups

Examples:
  python3 scripts/sync_approved_photos.py --dry-run --end-page 1 --limit 20
  python3 scripts/sync_approved_photos.py --start-page 10 --end-page 12
  python3 scripts/sync_approved_photos.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://crawlr.lawrenz.com/photos.json"


@dataclass
class Counters:
    processed: int = 0
    missing_raw: int = 0
    downloaded: int = 0
    download_failed: int = 0
    unknown_type: int = 0
    symlink_created: int = 0
    symlink_updated: int = 0
    symlink_unchanged: int = 0
    stale_symlinks: int = 0
    stale_records: int = 0
    stale_npy: int = 0


def build_page_url(base_url: str, page: int) -> str:
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}page={page}"


def fetch_json_array(url: str, timeout_s: float, retries: int) -> list[Any]:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "approved-photo-symlinks/1.0"})
            with urlopen(req, timeout=timeout_s) as resp:
                body = resp.read()
            data = json.loads(body)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array from {url}, got {type(data).__name__}")
            return data
        except HTTPError as e:
            last_err = e
            retryable = e.code in (429, 500, 502, 503, 504)
            if not retryable or attempt >= retries:
                raise

            retry_after = None
            try:
                retry_after = int(e.headers.get("Retry-After", ""))
            except Exception:
                retry_after = None

            backoff = min(60, 2**attempt)
            if retry_after is not None:
                backoff = max(backoff, retry_after)
            time.sleep(backoff)
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            if attempt >= retries:
                raise
            time.sleep(min(60, 2**attempt))

    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def download_raw_file(url: str, dest_path: str, timeout: float) -> bool:
    """
    Download raw file from URL to dest_path.
    
    Args:
        url: Source URL to download from
        dest_path: Destination file path
        timeout: HTTP timeout in seconds
    
    Returns:
        True on success, False on any error
    """
    try:
        req = Request(url, headers={"User-Agent": "approved-photo-symlinks/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        
        with open(dest_path, 'wb') as f:
            f.write(data)
        
        return True
    except HTTPError as e:
        print(f"warning: download failed (HTTP {e.code}): {url}", file=sys.stderr)
        return False
    except URLError as e:
        print(f"warning: download failed (network error): {url} - {e.reason}", file=sys.stderr)
        return False
    except TimeoutError:
        print(f"warning: download timed out: {url}", file=sys.stderr)
        return False
    except OSError as e:
        print(f"warning: failed to write file {dest_path}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"warning: unexpected error downloading {url}: {e}", file=sys.stderr)
        return False


def detect_extension(path: str) -> str | None:
    with open(path, "rb") as f:
        head = f.read(32)

    if head.startswith(b"\xFF\xD8\xFF"):
        return "jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"GIF8"):
        return "gif"
    if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"

    return None


def ensure_symlink(link_path: str, target_rel: str, target_abs: str, dry_run: bool) -> str:
    link_dir = os.path.dirname(link_path)

    if os.path.islink(link_path):
        existing_rel = os.readlink(link_path)
        existing_abs = os.path.abspath(os.path.join(link_dir, existing_rel))
        if os.path.normpath(existing_abs) == os.path.normpath(target_abs):
            return "unchanged"

    existed = os.path.lexists(link_path)
    if existed and not os.path.islink(link_path) and os.path.isdir(link_path):
        raise IsADirectoryError(link_path)

    if existed:
        if dry_run:
            return "updated"
        os.unlink(link_path)

    if dry_run:
        return "created" if not existed else "updated"

    os.symlink(target_rel, link_path)
    return "created" if not existed else "updated"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create extension-correct symlinks for approved photos.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for photos.json (default: %(default)s)")
    p.add_argument("--start-page", type=int, default=1, help="First page to fetch (default: %(default)s)")
    p.add_argument("--end-page", type=int, default=None, help="Last page to fetch (inclusive)")
    p.add_argument("--limit", type=int, default=None, help="Stop after processing N photo entries (smoke test)")
    p.add_argument(
        "--stop-after-links",
        type=int,
        default=None,
        help="Stop after creating/updating N symlinks (useful for quick verification)",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not modify filesystem; just report actions")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds (default: %(default)s)")
    p.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print a progress line every N processed items (0 to disable)",
    )
    p.add_argument("--verbose", action="store_true", help="Print each symlink create/update action")
    p.add_argument("--retries", type=int, default=5, help="Retries per page fetch (default: %(default)s)")
    p.add_argument("--raw-dir", default=os.path.join("data", "raw"), help="Raw images directory")
    p.add_argument("--approved-dir", default=os.path.join("data", "approved"), help="Approved symlinks directory")
    p.add_argument("--no-prune", dest="prune", action="store_false", help="Skip removal of stale symlinks/records/embeddings after scan")
    p.set_defaults(prune=True)
    p.add_argument("--derived-dir", default=os.path.join("data", "derived"), help="Derived data directory (for pruning)")
    p.add_argument("--jsonl", default=os.path.join("data", "derived", "approved_image_dataset.jsonl"), help="JSONL dataset path (for pruning)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.prune and (args.end_page is not None or args.limit is not None):
        print("error: pruning requires a full scan; use --no-prune with --end-page or --limit", file=sys.stderr)
        return 2

    raw_dir = os.path.abspath(args.raw_dir)
    approved_dir = os.path.abspath(args.approved_dir)

    if not os.path.isdir(raw_dir):
        print(f"error: raw dir not found: {raw_dir}", file=sys.stderr)
        return 2

    os.makedirs(approved_dir, exist_ok=True)

    counters = Counters()
    approved_filenames: set[str] = set()
    failed_pages: int = 0

    page = args.start_page
    while True:
        if args.end_page is not None and page > args.end_page:
            break

        url = build_page_url(args.base_url, page)
        print(f"page {page}: fetching {url}", file=sys.stderr)
        try:
            items = fetch_json_array(url, timeout_s=args.timeout, retries=args.retries)
        except Exception as e:
            print(f"warning: page {page}: fetch failed: {e}", file=sys.stderr)
            failed_pages += 1
            page += 1
            continue
        if not items:
            print(f"page {page}: empty; stopping", file=sys.stderr)
            break
        print(f"page {page}: {len(items)} items", file=sys.stderr)

        for item in items:
            if args.limit is not None and counters.processed >= args.limit:
                return summarize_and_exit(counters)

            counters.processed += 1
            if args.progress_every and (counters.processed % args.progress_every == 0):
                print(
                    f"progress: processed={counters.processed} downloaded={counters.downloaded} "
                    f"missing_raw={counters.missing_raw} download_failed={counters.download_failed} "
                    f"unknown_type={counters.unknown_type} created={counters.symlink_created} updated={counters.symlink_updated}",
                    file=sys.stderr,
                )

            filename = None
            exportable_url = None
            if isinstance(item, dict):
                filename = item.get("filename")
                exportable_url = item.get("exportable_url")
            if not filename:
                continue

            approved_filenames.add(filename)

            raw_path = os.path.join(raw_dir, filename)
            
            # Download missing raw file if possible
            if not os.path.exists(raw_path):
                # Check if we have a URL to download from
                if not exportable_url:
                    counters.missing_raw += 1
                    continue
                
                # Validate URL
                if not isinstance(exportable_url, str) or not exportable_url.startswith("https://"):
                    if args.verbose:
                        print(f"warning: invalid exportable_url for {filename}: {exportable_url}", file=sys.stderr)
                    counters.missing_raw += 1
                    continue
                
                # Skip download in dry-run mode
                if args.dry_run:
                    if args.verbose:
                        print(f"dry-run: would download {filename} from {exportable_url}")
                    continue
                
                # Download the file
                if args.verbose:
                    print(f"downloading: {filename} from {exportable_url}", file=sys.stderr)
                
                if not download_raw_file(exportable_url, raw_path, timeout=args.timeout):
                    counters.download_failed += 1
                    continue
                
                counters.downloaded += 1
                if args.verbose:
                    size_kb = os.path.getsize(raw_path) / 1024
                    print(f"downloaded: {filename} ({size_kb:.1f} KB)", file=sys.stderr)

            ext = detect_extension(raw_path)
            if ext is None:
                counters.unknown_type += 1
                continue

            link_name = f"{filename}.{ext}"
            link_path = os.path.join(approved_dir, link_name)
            target_rel = os.path.relpath(raw_path, start=approved_dir)

            status = ensure_symlink(link_path, target_rel=target_rel, target_abs=raw_path, dry_run=args.dry_run)
            if status == "created":
                counters.symlink_created += 1
            elif status == "updated":
                counters.symlink_updated += 1
            else:
                counters.symlink_unchanged += 1

            if args.verbose and status in ("created", "updated"):
                print(f"{status}: {os.path.relpath(link_path)} -> {target_rel}")

            if args.stop_after_links is not None and (counters.symlink_created + counters.symlink_updated) >= args.stop_after_links:
                return summarize_and_exit(counters)

        page += 1

    if args.prune:
        if failed_pages > 0:
            print(
                f"warning: skipping prune — {failed_pages} page(s) failed to fetch; run again when all pages succeed",
                file=sys.stderr,
            )
        else:
            stale_sym, stale_rec, stale_npy = prune_stale(
                approved_filenames=approved_filenames,
                approved_dir=approved_dir,
                derived_dir=os.path.abspath(args.derived_dir),
                jsonl_path=os.path.abspath(args.jsonl),
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            counters.stale_symlinks = stale_sym
            counters.stale_records = stale_rec
            counters.stale_npy = stale_npy

    return summarize_and_exit(counters)


def prune_stale(
    approved_filenames: set[str],
    approved_dir: str,
    derived_dir: str,
    jsonl_path: str,
    dry_run: bool,
    verbose: bool,
) -> tuple[int, int, int]:
    """Remove symlinks, JSONL records, and .npy files for images no longer in the approved set."""
    stale_symlinks = 0
    stale_records = 0
    stale_npy = 0

    stale_stems: list[str] = []
    for entry in os.scandir(approved_dir):
        if not entry.is_symlink():
            continue
        stem = os.path.splitext(entry.name)[0]
        if stem not in approved_filenames:
            stale_stems.append(stem)
            if verbose:
                print(f"stale symlink: {entry.path}", file=sys.stderr)
            if not dry_run:
                os.unlink(entry.path)
            stale_symlinks += 1

    if not stale_stems:
        return stale_symlinks, stale_records, stale_npy

    stale_set = set(stale_stems)

    if os.path.isfile(jsonl_path):
        kept_lines: list[str] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("id") in stale_set:
                        stale_records += 1
                        if verbose:
                            print(f"stale record: {record.get('id')}", file=sys.stderr)
                        continue
                except json.JSONDecodeError:
                    pass
                kept_lines.append(line)
        if stale_records > 0 and not dry_run:
            tmp_path = jsonl_path + ".prune-tmp"
            with open(tmp_path, "w") as f:
                for line in kept_lines:
                    f.write(line + "\n")
            os.replace(tmp_path, jsonl_path)

    npy_subdirs = ["dinov3", "vae_latents", "t5_hidden"]
    for stem in stale_stems:
        for subdir in npy_subdirs:
            npy_path = os.path.join(derived_dir, subdir, stem + ".npy")
            if os.path.isfile(npy_path):
                if verbose:
                    print(f"stale npy: {npy_path}", file=sys.stderr)
                if not dry_run:
                    os.unlink(npy_path)
                stale_npy += 1

    return stale_symlinks, stale_records, stale_npy


def summarize_and_exit(c: Counters) -> int:
    lines = [
        "Summary:",
        f"  processed:        {c.processed}",
        f"  downloaded:       {c.downloaded}",
        f"  download failed:  {c.download_failed}",
        f"  missing raw:      {c.missing_raw}",
        f"  unknown type:     {c.unknown_type}",
        f"  symlink created:  {c.symlink_created}",
        f"  symlink updated:  {c.symlink_updated}",
        f"  symlink unchanged:{c.symlink_unchanged}",
    ]
    if c.stale_symlinks or c.stale_records or c.stale_npy:
        lines += [
            f"  stale symlinks:   {c.stale_symlinks}",
            f"  stale records:    {c.stale_records}",
            f"  stale npy files:  {c.stale_npy}",
        ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130)
