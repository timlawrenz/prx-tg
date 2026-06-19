import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def create_metric_graphs(exp_dir):
    csv_path = exp_dir / "metric_trajectory.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Aesthetic Score (Target > 5.0)
    ax1.plot(df['step'], df['aesthetic_score'], marker='o', color='purple', linewidth=2)
    ax1.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='Good (>5.0)')
    ax1.set_title('LAION Aesthetic Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (1-10)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # CLIP Score
    ax2.plot(df['step'], df['clip_score'], marker='o', color='green', linewidth=2)
    ax2.set_title('CLIP Score (Text-Image Alignment)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cosine Similarity')
    ax2.grid(True, alpha=0.3)
    
    # Face Confidence
    ax3.plot(df['step'], df['face_confidence'], marker='o', color='blue', linewidth=2)
    ax3.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    ax3.set_title('DWPose Face Confidence', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Training Steps', fontsize=11)
    ax3.set_ylabel('Mean Confidence')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    out_path = exp_dir / "metric_graphs.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved metric graphs to {out_path}")


def create_progression_collage(exp_dir):
    # Select key milestones
    milestones = [2500, 5000, 10000, 20000, 30000, 32500]
    
    collages = []
    labels = []
    
    for step in milestones:
        collage_path = exp_dir / "validation_evals" / f"step{step:07d}" / "collage.png"
        if collage_path.exists():
            img = Image.open(collage_path)
            collages.append(img)
            labels.append(f"Step {step:,}")
            
    if not collages:
        print("No collages found.")
        return
        
    # Assume all collages are same width. Resize if slightly off.
    base_w = collages[0].width
    base_h = collages[0].height
    
    # Add space for labels (50px per image)
    label_h = 50
    total_h = (base_h + label_h) * len(collages)
    
    master = Image.new('RGB', (base_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(master)
    
    y_offset = 0
    for img, label in zip(collages, labels):
        # Draw label
        draw.text((10, y_offset + 15), label, fill=(0, 0, 0))
        y_offset += label_h
        
        # Paste image
        if img.width != base_w:
            img = img.resize((base_w, base_h))
        master.paste(img, (0, y_offset))
        y_offset += base_h
        
    out_path = exp_dir / "progression_timeline.png"
    master.save(out_path)
    print(f"Saved progression timeline to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_evaluations.py <experiment_dir>")
        sys.exit(1)
        
    exp_dir = Path(sys.argv[1])
    create_metric_graphs(exp_dir)
    create_progression_collage(exp_dir)
