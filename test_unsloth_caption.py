#!/usr/bin/env python3
"""Quick test of Unsloth Gemma3 captioning integration."""

import sys
import torch
from pathlib import Path
from PIL import Image

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Import the functions we modified
from generate_approved_image_dataset import (
    load_caption_pipeline,
    generate_captions,
    GEMMA_MODEL_ID
)

def main():
    print("=" * 60)
    print("Testing Unsloth Gemma3 Caption Integration")
    print("=" * 60)
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (this will be slow)")
    
    print(f"Model: {GEMMA_MODEL_ID}")
    print()
    
    # Find test images
    approved_dir = Path("data/approved")
    if not approved_dir.exists():
        print(f"Error: {approved_dir} not found")
        return 1
    
    test_images = sorted(approved_dir.glob("*.jpg"))[:3]
    if not test_images:
        print(f"Error: No images found in {approved_dir}")
        return 1
    
    print(f"Testing with {len(test_images)} images:")
    for img_path in test_images:
        print(f"  - {img_path.name}")
    print()
    
    # Load model
    print("Loading Unsloth Gemma3 model...")
    try:
        caption_pipe = load_caption_pipeline(device, GEMMA_MODEL_ID)
        print("✓ Model loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate captions
    print("Generating captions...")
    images = [Image.open(p).convert("RGB") for p in test_images]
    
    try:
        captions = generate_captions(caption_pipe, images, max_new_tokens=256)
        print("✓ Captions generated successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to generate captions: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display results
    print("Results:")
    print("=" * 60)
    for img_path, caption in zip(test_images, captions):
        print(f"\nImage: {img_path.name}")
        print(f"Caption: {caption}")
        print()
    
    print("=" * 60)
    print("✓ Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
