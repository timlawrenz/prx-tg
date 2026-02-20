# Regenerating DINO Patches After Spatial Alignment Fix

## Critical Context

**A spatial alignment bug was discovered and fixed in the DINO preprocessing pipeline.**

### The Problem
- **Before**: `transforms.CenterCrop(224)` processed only the center of images
- **Impact**: For 1216×832 images, DINO saw only center 224×224, but VAE encoded full image
- **Result**: Patches spatially misaligned with VAE latents → model can't learn correct correlations

### The Fix
- **After**: Dynamic resolution with no center-crop
- **Method**: Round bucket dims to nearest multiple of 14, feed full aspect-correct image
- **Result**: Perfect spatial alignment, variable-length patches per bucket

## Patch Counts Per Bucket

After regeneration, patches will have these shapes:

```
Bucket          DINO Input    Patch Grid    Total Patches
1024×1024   →   1022×1022  →  73×73      =  5329
1216×832    →   1218×826   →  87×59      =  5133
832×1216    →   826×1218   →  59×87      =  5133
1280×768    →   1274×770   →  91×55      =  5005
768×1280    →   770×1274   →  55×91      =  5005
1344×704    →   1344×700   →  96×50      =  4800
704×1344    →   700×1344   →  50×96      =  4800
```

## Safe Regeneration Steps

### 1. Backup Old Patches (Optional)
```bash
# If you want to keep old patches for comparison
mv data/derived/dinov3_patches data/derived/dinov3_patches_OLD_MISALIGNED

# Or just delete them (they're broken anyway)
rm -rf data/derived/dinov3_patches
```

### 2. Regenerate Patches with Fixed Script
```bash
# The script will automatically use the new dynamic resolution
python scripts/generate_approved_image_dataset.py \
  --input-dir data/approved \
  --output-jsonl data/dataset.jsonl \
  --pass-filter dinov3 \
  --device cuda \
  --verbose
```

**What this does:**
- Loads each image according to its `aspect_bucket` in the JSONL
- Rounds bucket dimensions to nearest multiple of 14
- Feeds aspect-correct image to DINO (no center-crop!)
- Saves variable-length patches to `data/derived/dinov3_patches/{image_id}.npy`

### 3. Verify Patch Shapes
```bash
# Check a few patches to confirm variable lengths
python3 << 'EOF'
import numpy as np
from pathlib import Path

patches_dir = Path("data/derived/dinov3_patches")

print("Sample patch shapes (should vary by bucket):")
for i, p in enumerate(list(patches_dir.glob("*.npy"))[:10]):
    arr = np.load(p)
    print(f"  {p.name}: {arr.shape}")
    if i >= 9:
        break

# Expected: Different shapes like (5329, 1024), (5133, 1024), etc.
EOF
```

Expected output:
```
Sample patch shapes (should vary by bucket):
  image_001.npy: (5329, 1024)  # 1024×1024 bucket
  image_002.npy: (5133, 1024)  # 1216×832 bucket
  image_003.npy: (5133, 1024)  # 832×1216 bucket
  ...
```

### 4. Update Shards (After Full Training Completes)

**DON'T do this yet** - wait until current training finishes!

When ready to integrate patches into training:
1. Update `scripts/create_webdataset_shards.py` (see implementation plan)
2. Regenerate all shards with new patches
3. Update `production/data.py` to load variable-length patches
4. Update model architecture for dynamic sequences

## Verification Checklist

- [ ] Old patches backed up or deleted
- [ ] Script runs without errors
- [ ] New patches created in `data/derived/dinov3_patches/`
- [ ] Patch counts match expected values (see table above)
- [ ] Shapes vary across buckets (not all 196×1024)
- [ ] File sizes reasonable (~20-42 KB per file depending on bucket)

## Next Steps

After regeneration completes:
1. ✅ Patches are now spatially aligned with VAE latents
2. ⏳ Wait for current training to complete
3. ⏳ Implement Phase 1.1-1.3 of the integration plan (shard updates)
4. ⏳ Implement Phase 2-6 (model architecture, training integration)

## Troubleshooting

**Q: Script says "aspect_bucket not found"**
A: Make sure `data/dataset.jsonl` has `aspect_bucket` field for all records. Run with `--pass-filter migrate` first if needed.

**Q: All patches have same shape**
A: Check that bucket dimensions are being passed correctly. Add `--verbose` to see processing details.

**Q: Out of memory**
A: Reduce batch size in script or process images one at a time.

## Important Notes

- Old patches (196×1024) are **spatially misaligned** and unusable
- New patches vary in count per bucket (perfect alignment)
- Don't mix old and new patches - regenerate ALL
- Shards don't include patches yet - that's a separate step
