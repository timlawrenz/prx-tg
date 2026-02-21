#!/usr/bin/env python3
"""
Validate DINOv3 patches and delete any with incorrect counts.

This script checks all patches against their expected counts based on
the /16 rounding formula. Any patches with incorrect counts (likely
generated with the old /14 rounding bug) are deleted so they can be
regenerated.

Usage:
    python scripts/validate_and_clean_patches.py [--dry-run] [--verbose]
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def calculate_expected_patch_count(width, height):
    """Calculate expected patch count using /16 rounding."""
    dino_w = round(width / 16) * 16
    dino_h = round(height / 16) * 16
    return (dino_h // 16) * (dino_w // 16)


def main():
    parser = argparse.ArgumentParser(description="Validate and clean DINOv3 patches")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--verbose", action="store_true",
                       help="Show all files checked, not just problematic ones")
    args = parser.parse_args()
    
    # Paths
    dataset_file = Path("data/derived/approved_image_dataset.jsonl")
    patch_dir = Path("data/derived/dinov3_patches")
    
    if not dataset_file.exists():
        print(f"❌ Dataset file not found: {dataset_file}")
        return 1
    
    if not patch_dir.exists():
        print(f"❌ Patch directory not found: {patch_dir}")
        return 1
    
    print(f"{'=' * 80}")
    print("DINOv3 Patch Validation and Cleanup")
    print(f"{'=' * 80}\n")
    
    # Load dataset metadata
    print(f"Loading metadata from {dataset_file}...")
    records = []
    with open(dataset_file) as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    print(f"Loaded {len(records)} records\n")
    
    # Statistics
    stats = {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'missing': 0,
        'by_bucket': defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0}),
    }
    
    files_to_delete = []
    
    # Check each record
    for record in records:
        image_id = record['image_id']
        bucket = record['aspect_bucket']
        
        # Parse bucket dimensions
        width, height = map(int, bucket.split('x'))
        expected_count = calculate_expected_patch_count(width, height)
        
        # Check if patch file exists
        patch_file = patch_dir / f"{image_id}.npy"
        
        stats['total'] += 1
        stats['by_bucket'][bucket]['total'] += 1
        
        if not patch_file.exists():
            stats['missing'] += 1
            if args.verbose:
                print(f"⏳ MISSING: {image_id} ({bucket})")
            continue
        
        # Load and check patch count
        try:
            patches = np.load(patch_file)
            actual_count = patches.shape[0]
            
            if actual_count == expected_count:
                stats['correct'] += 1
                stats['by_bucket'][bucket]['correct'] += 1
                if args.verbose:
                    print(f"✅ OK: {image_id} ({bucket}): {actual_count} patches")
            else:
                stats['incorrect'] += 1
                stats['by_bucket'][bucket]['incorrect'] += 1
                files_to_delete.append((patch_file, image_id, bucket, expected_count, actual_count))
                print(f"❌ WRONG: {image_id} ({bucket}): {actual_count} (expected {expected_count})")
        
        except Exception as e:
            print(f"⚠️  ERROR loading {image_id}: {e}")
            files_to_delete.append((patch_file, image_id, bucket, expected_count, None))
    
    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}\n")
    
    print(f"Total files checked:  {stats['total']}")
    print(f"  ✅ Correct:         {stats['correct']}")
    print(f"  ❌ Incorrect:       {stats['incorrect']}")
    print(f"  ⏳ Missing:         {stats['missing']}")
    print()
    
    # Per-bucket breakdown
    print("By bucket:")
    print(f"  {'Bucket':<12}  {'Total':>6}  {'Correct':>8}  {'Incorrect':>10}")
    print(f"  {'-' * 50}")
    for bucket in sorted(stats['by_bucket'].keys()):
        b_stats = stats['by_bucket'][bucket]
        print(f"  {bucket:<12}  {b_stats['total']:>6}  {b_stats['correct']:>8}  {b_stats['incorrect']:>10}")
    
    # Delete incorrect files
    if files_to_delete:
        print(f"\n{'=' * 80}")
        print(f"{'Action Required' if not args.dry_run else 'Dry Run'}")
        print(f"{'=' * 80}\n")
        
        print(f"Found {len(files_to_delete)} incorrect patch files\n")
        
        if args.dry_run:
            print("🔍 DRY RUN: Would delete these files:\n")
            for patch_file, image_id, bucket, expected, actual in files_to_delete[:10]:
                print(f"  - {image_id}.npy ({bucket}): {actual} → should be {expected}")
            if len(files_to_delete) > 10:
                print(f"  ... and {len(files_to_delete) - 10} more")
            print(f"\nRun without --dry-run to actually delete these files")
        else:
            print("🗑️  Deleting incorrect patch files...\n")
            deleted_count = 0
            for patch_file, image_id, bucket, expected, actual in files_to_delete:
                try:
                    patch_file.unlink()
                    deleted_count += 1
                    if args.verbose:
                        print(f"  Deleted: {image_id}.npy")
                except Exception as e:
                    print(f"  ⚠️  Failed to delete {image_id}.npy: {e}")
            
            print(f"\n✅ Deleted {deleted_count} patch files")
            print(f"\n📝 Next step: Run the regeneration script:")
            print(f"   python scripts/generate_approved_image_dataset.py --verbose")
    else:
        print(f"\n✅ All existing patches have correct counts!")
        if stats['missing'] > 0:
            print(f"\n📝 Note: {stats['missing']} patches are missing and need generation:")
            print(f"   python scripts/generate_approved_image_dataset.py --verbose")
    
    return 0


if __name__ == "__main__":
    exit(main())
