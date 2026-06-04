| I | `fp8_native` | ✓ | Muon | ✓ | FP8 via torchao, dynamic tensor masking | ✅ Done | 2026-06 | G |

### Key Comparisons
- **G → I**: `sys/iter_per_sec` & `memory/peak_vram_gb` — Native FP8 via `torchao` trained successfully! The model fits into 24GB VRAM and trains in **26 hours** (down from 56h). Perceptual quality remains highly competitive with BF16 (Recon LPIPS 0.906 vs 0.900, Text-only LPIPS 0.909 vs 0.920), while text controllability actually improved (Text Manip Diff 0.504 vs 0.485). FP8 represents a >2x speedup with negligible perceptual degradation.
