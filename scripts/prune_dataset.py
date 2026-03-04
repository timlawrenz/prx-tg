#!/usr/bin/env python3
"""
Prune orphaned data from the approved image dataset.

Removes:
- Broken symlinks in data/approved/ (pointing to missing raw files)
- Orphaned JSONL records (no corresponding symlink in approved/)
- Orphaned .npy files in derived directories
"""

import argparse
import json
import os
import sys
from pathlib import Path


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def compute_image_id(image_path: Path) -> str:
    """Extract image_id from path (basename without extension)."""
    return image_path.stem


def find_broken_symlinks(approved_dir: Path, verbose: bool = False) -> list[Path]:
    """
    Find all broken symlinks in approved directory.
    
    A symlink is broken if it points to a non-existent file.
    
    Args:
        approved_dir: Path to data/approved directory
        verbose: Enable verbose logging
    
    Returns:
        List of broken symlink paths
    """
    broken = []
    
    if not approved_dir.exists():
        eprint(f"warning: approved directory does not exist: {approved_dir}")
        return broken
    
    eprint(f"Scanning symlinks in {approved_dir}...")
    count = 0
    
    for item in approved_dir.iterdir():
        if item.is_symlink():
            count += 1
            if count % 5000 == 0:
                eprint(f"  Scanned {count} symlinks, found {len(broken)} broken...")
            
            # Check if target exists
            try:
                if not item.exists():
                    broken.append(item)
                    if verbose:
                        eprint(f"verbose: found broken symlink: {item}")
            except OSError as e:
                # Handle cases where symlink is completely invalid
                broken.append(item)
                if verbose:
                    eprint(f"verbose: found invalid symlink: {item} ({e})")
    
    eprint(f"  Scanned {count} total symlinks")
    return broken


def find_orphaned_records(jsonl_path: Path, approved_dir: Path, verbose: bool = False) -> list[dict]:
    """
    Find JSONL records with no corresponding symlink in approved/.
    
    Args:
        jsonl_path: Path to JSONL file
        approved_dir: Path to data/approved directory
        verbose: Enable verbose logging
    
    Returns:
        List of orphaned record dictionaries (each contains at least image_id)
    """
    orphaned = []
    
    if not jsonl_path.exists():
        eprint(f"warning: JSONL file does not exist: {jsonl_path}")
        return orphaned
    
    eprint(f"Checking JSONL records against symlinks...")
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 500 == 0:
                eprint(f"  Checked {line_num} records, found {len(orphaned)} orphaned...")
            
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                eprint(f"warning: malformed JSON at line {line_num}: {e}")
                continue
            
            # Get image_id from record
            image_id = record.get('image_id')
            if not image_id:
                # Fallback: extract from image_path if available
                image_path = record.get('image_path')
                if image_path:
                    image_id = compute_image_id(Path(image_path))
                else:
                    eprint(f"warning: record at line {line_num} has no image_id or image_path")
                    continue
            
            # Check if any symlink with this image_id exists in approved/
            # Need to check all possible extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                symlink_path = approved_dir / f"{image_id}{ext}"
                if symlink_path.exists():
                    found = True
                    break
            
            if not found:
                orphaned.append({'image_id': image_id, 'line_num': line_num})
                if verbose:
                    eprint(f"verbose: found orphaned record: {image_id} (line {line_num})")
    
    eprint(f"  Checked {line_num} total records")
    return orphaned


def delete_broken_symlinks(symlinks: list[Path], dry_run: bool = False, verbose: bool = False) -> int:
    """
    Delete broken symlinks from approved directory.
    
    Args:
        symlinks: List of symlink paths to delete
        dry_run: If True, only report what would be deleted
        verbose: Enable verbose logging
    
    Returns:
        Count of successfully deleted symlinks
    """
    deleted = 0
    
    for symlink in symlinks:
        if dry_run:
            eprint(f"dry-run: would delete broken symlink: {symlink}")
            deleted += 1
        else:
            try:
                symlink.unlink()
                deleted += 1
                if verbose:
                    eprint(f"verbose: deleted broken symlink: {symlink}")
            except OSError as e:
                eprint(f"error: failed to delete symlink {symlink}: {e}")
    
    return deleted


def delete_orphaned_npy_files(image_ids: list[str], derived_dir: Path, dry_run: bool = False, verbose: bool = False) -> dict[str, int]:
    """
    Delete orphaned .npy files for given image_ids.
    
    Args:
        image_ids: List of image IDs to remove
        derived_dir: Path to data/derived directory
        dry_run: If True, only report what would be deleted
        verbose: Enable verbose logging
    
    Returns:
        Dictionary with counts per directory: {'dinov3': N, 'vae_latents': M, 't5_hidden': K}
    """
    counts = {
        'dinov3': 0,
        'dinov3_patches': 0,
        'vae_latents': 0,
        't5_hidden': 0
    }
    
    subdirs = {
        'dinov3': derived_dir / 'dinov3',
        'dinov3_patches': derived_dir / 'dinov3_patches',
        'vae_latents': derived_dir / 'vae_latents',
        't5_hidden': derived_dir / 't5_hidden'
    }
    
    for image_id in image_ids:
        for key, subdir in subdirs.items():
            npy_path = subdir / f"{image_id}.npy"
            
            if npy_path.exists():
                if dry_run:
                    eprint(f"dry-run: would delete {npy_path}")
                    counts[key] += 1
                else:
                    try:
                        npy_path.unlink()
                        counts[key] += 1
                        if verbose:
                            eprint(f"verbose: deleted {npy_path}")
                    except OSError as e:
                        eprint(f"error: failed to delete {npy_path}: {e}")
    
    return counts


def prune_jsonl(orphaned_ids: set[str], jsonl_path: Path, dry_run: bool = False, verbose: bool = False) -> int:
    """
    Remove orphaned records from JSONL file.
    
    Uses atomic file replacement: writes to .tmp file, then renames.
    
    Args:
        orphaned_ids: Set of image IDs to remove
        jsonl_path: Path to JSONL file
        dry_run: If True, only report what would be removed
        verbose: Enable verbose logging
    
    Returns:
        Count of removed records
    """
    removed = 0
    
    if not jsonl_path.exists():
        eprint(f"warning: JSONL file does not exist: {jsonl_path}")
        return removed
    
    if dry_run:
        # Just count what would be removed
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    image_id = record.get('image_id')
                    if image_id and image_id in orphaned_ids:
                        removed += 1
                        if verbose:
                            eprint(f"dry-run: would remove record: {image_id}")
                except json.JSONDecodeError:
                    continue
        return removed
    
    # Atomic update: write to temp file
    tmp_path = jsonl_path.with_suffix('.jsonl.prune-tmp')
    kept = 0
    
    try:
        with open(jsonl_path, 'r') as infile, open(tmp_path, 'w') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    image_id = record.get('image_id')
                    
                    if image_id and image_id in orphaned_ids:
                        removed += 1
                        if verbose:
                            eprint(f"verbose: removing record: {image_id}")
                    else:
                        outfile.write(line + '\n')
                        kept += 1
                except json.JSONDecodeError as e:
                    # Keep malformed lines (don't silently drop)
                    eprint(f"warning: keeping malformed JSON line: {e}")
                    outfile.write(line + '\n')
                    kept += 1
        
        # Atomic replace
        tmp_path.replace(jsonl_path)
        
        if verbose:
            eprint(f"verbose: JSONL updated - kept {kept} records, removed {removed} records")
    
    except Exception as e:
        eprint(f"error: failed to prune JSONL: {e}")
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    
    return removed


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prune orphaned data from approved image dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run to see what would be deleted
  python scripts/prune_dataset.py --dry-run --verbose
  
  # Actually delete orphaned data
  python scripts/prune_dataset.py
  
  # Custom paths
  python scripts/prune_dataset.py --approved-dir /path/to/approved --derived-dir /path/to/derived

Warning: This script permanently deletes data. Always run with --dry-run first!
        """
    )
    
    parser.add_argument(
        '--approved-dir',
        type=Path,
        default=Path('data/approved'),
        help='Path to approved symlinks directory (default: data/approved)'
    )
    
    parser.add_argument(
        '--derived-dir',
        type=Path,
        default=Path('data/derived'),
        help='Path to derived data directory (default: data/derived)'
    )
    
    parser.add_argument(
        '--jsonl',
        type=Path,
        default=Path('data/derived/approved_image_dataset.jsonl'),
        help='Path to JSONL file (default: data/derived/approved_image_dataset.jsonl)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    # Validation
    if not args.approved_dir.exists():
        eprint(f"error: approved directory does not exist: {args.approved_dir}")
        return 1
    
    if not args.derived_dir.exists():
        eprint(f"error: derived directory does not exist: {args.derived_dir}")
        return 1
    
    if not args.jsonl.exists():
        eprint(f"error: JSONL file does not exist: {args.jsonl}")
        return 1
    
    # Print header
    mode = "DRY-RUN MODE" if args.dry_run else "DELETION MODE"
    eprint(f"\n{'='*60}")
    eprint(f"Dataset Pruning Script - {mode}")
    eprint(f"{'='*60}")
    eprint(f"Approved dir: {args.approved_dir}")
    eprint(f"Derived dir:  {args.derived_dir}")
    eprint(f"JSONL file:   {args.jsonl}")
    eprint(f"{'='*60}\n")
    
    if args.dry_run:
        eprint("⚠️  DRY-RUN: No data will be deleted\n")
    else:
        eprint("⚠️  WARNING: This will permanently delete data!\n")
    
    # Phase 1: Discovery
    eprint("Phase 1: Discovering orphaned data...")
    broken_symlinks = find_broken_symlinks(args.approved_dir, args.verbose)
    orphaned_records = find_orphaned_records(args.jsonl, args.approved_dir, args.verbose)
    
    # Combine image IDs from both sources
    broken_ids = {compute_image_id(link) for link in broken_symlinks}
    orphaned_ids = {record['image_id'] for record in orphaned_records}
    all_orphaned_ids = broken_ids | orphaned_ids
    
    eprint(f"\nDiscovery Results:")
    eprint(f"  Broken symlinks:    {len(broken_symlinks)}")
    eprint(f"  Orphaned records:   {len(orphaned_records)}")
    eprint(f"  Total unique IDs:   {len(all_orphaned_ids)}")
    
    if not broken_symlinks and not orphaned_records:
        eprint(f"\n✅ No orphaned data found. Dataset is clean!")
        return 0
    
    eprint()
    
    # Phase 2: Deletion
    if not args.dry_run:
        eprint("Phase 2: Deleting orphaned data...")
    else:
        eprint("Phase 2: Simulating deletion (dry-run)...")
    
    # Delete broken symlinks
    deleted_symlinks = delete_broken_symlinks(broken_symlinks, args.dry_run, args.verbose)
    
    # Delete .npy files
    npy_counts = delete_orphaned_npy_files(list(all_orphaned_ids), args.derived_dir, args.dry_run, args.verbose)
    
    # Prune JSONL
    removed_records = prune_jsonl(all_orphaned_ids, args.jsonl, args.dry_run, args.verbose)
    
    # Summary
    eprint(f"\n{'='*60}")
    eprint("Summary:")
    eprint(f"{'='*60}")
    eprint(f"  Broken symlinks deleted:     {deleted_symlinks}")
    eprint(f"  JSONL records removed:       {removed_records}")
    eprint(f"  DINOv3 embeddings deleted:   {npy_counts['dinov3']}")
    eprint(f"  VAE latents deleted:         {npy_counts['vae_latents']}")
    eprint(f"  T5 hidden states deleted:    {npy_counts['t5_hidden']}")
    eprint(f"  Total .npy files deleted:    {sum(npy_counts.values())}")
    eprint(f"{'='*60}\n")
    
    if args.dry_run:
        eprint("✅ Dry-run complete. Run without --dry-run to actually delete.")
    else:
        eprint("✅ Pruning complete.")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        eprint("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        eprint(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
