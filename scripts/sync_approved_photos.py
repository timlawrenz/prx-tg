#!/usr/bin/env python3
"""Sync approved photos to `data/approved/`.

Fetches the paginated approved photo list from:
  https://crawlr.lawrenz.com/photos.json?page=N

For each item, checks for a matching raw file at `data/raw/<filename>` (no extension),
detects its type from magic bytes, then creates/updates a symlink:
  data/approved/<filename>.<ext> -> ../raw/<filename>

Notes:
- Pruning (removing stale links) is intentionally NOT implemented in this change.
- This script does NOT enumerate `data/raw/`; it does per-filename path lookups.

Examples:
  python3 scripts/sync_approved_photos.py --dry-run --end-page 1 --limit 20
  python3 scripts/sync_approved_photos.py --start-page 10 --end-page 12
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
    unknown_type: int = 0
    symlink_created: int = 0
    symlink_updated: int = 0
    symlink_unchanged: int = 0


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
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    raw_dir = os.path.abspath(args.raw_dir)
    approved_dir = os.path.abspath(args.approved_dir)

    if not os.path.isdir(raw_dir):
        print(f"error: raw dir not found: {raw_dir}", file=sys.stderr)
        return 2

    os.makedirs(approved_dir, exist_ok=True)

    counters = Counters()

    page = args.start_page
    while True:
        if args.end_page is not None and page > args.end_page:
            break

        url = build_page_url(args.base_url, page)
        print(f"page {page}: fetching {url}", file=sys.stderr)
        items = fetch_json_array(url, timeout_s=args.timeout, retries=args.retries)
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
                    f"progress: processed={counters.processed} missing_raw={counters.missing_raw} "
                    f"unknown_type={counters.unknown_type} created={counters.symlink_created} updated={counters.symlink_updated}",
                    file=sys.stderr,
                )

            filename = None
            if isinstance(item, dict):
                filename = item.get("filename")
            if not filename:
                continue

            raw_path = os.path.join(raw_dir, filename)
            if not os.path.exists(raw_path):
                counters.missing_raw += 1
                continue

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

    return summarize_and_exit(counters)


def summarize_and_exit(c: Counters) -> int:
    print(
        "\n".join(
            [
                "Summary:",
                f"  processed:        {c.processed}",
                f"  missing raw:      {c.missing_raw}",
                f"  unknown type:     {c.unknown_type}",
                f"  symlink created:  {c.symlink_created}",
                f"  symlink updated:  {c.symlink_updated}",
                f"  symlink unchanged:{c.symlink_unchanged}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130)
