#!/bin/bash
# Wrapper script to run generate_approved_image_dataset.py with proper environment variables

# Silence Unsloth torchvision warning
export UNSLOTH_SKIP_TORCHVISION_CHECK=1

# Use HF_HOME instead of deprecated TRANSFORMERS_CACHE
# export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ROCm tuning for optimal performance (optional, only if using ROCm)
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_ENABLED=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# Run the script with all arguments passed through
python scripts/generate_approved_image_dataset.py "$@"
