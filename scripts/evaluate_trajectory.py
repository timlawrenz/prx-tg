#!/usr/bin/env python3
"""
Run evaluate_checkpoint.py across all available checkpoints in an experiment.
Gathers the results and creates a CSV of the metric trajectories.
"""

import os
import sys
import glob
import json
import torch
import pandas as pd
from pathlib import Path

# Fix python parsing path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_checkpoint import run_evaluation, load_aesthetic_predictor
import open_clip
from scripts.dwpose_onnx import DWPoseDetector
from scripts.txt2img import T5Encoder

def evaluate_all_checkpoints(experiment_dir):
    exp_path = Path(experiment_dir)
    checkpoints_dir = exp_path / "checkpoints"
    out_dir_base = exp_path / "validation_evals"
    
    # Find all checkpoint_step*.pt
    checkpoints = sorted(glob.glob(str(checkpoints_dir / "checkpoint_step*.pt")))
    
    if not checkpoints:
        print(f"No regular step checkpoints found in {checkpoints_dir}")
        sys.exit(1)
        
    print(f"Found {len(checkpoints)} checkpoints to evaluate.")
    
    # --- PRELOAD MODELS ---
    print("Pre-loading evaluation models to save time across runs...")
    device = 'cuda'
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device=device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
    aesthetic_model = load_aesthetic_predictor(device)
    print("Loading DWPose (ONNX CPU)...")
    dwpose = DWPoseDetector(device="cpu")
    print("Loading T5 Encoder (CUDA)...")
    t5 = T5Encoder(torch.device(device))
    t5._load() # explicitly load to warm up
    # ----------------------

    trajectory_data = []
    config_path = str(exp_path / "config.yaml")
    
    for ckpt_path in checkpoints:
        ckpt_name = Path(ckpt_path).stem
        # Extract step number from "checkpoint_step0002500"
        step = int(ckpt_name.replace("checkpoint_step", ""))
        
        out_dir = out_dir_base / f"step{step:07d}"
        
        # Check if already evaluated
        results_file = out_dir / "evaluation_results.json"
        
        if not results_file.exists():
            print(f"\n[{step}] Running evaluation...")
            try:
                run_evaluation(
                    ckpt_path, config_path, str(out_dir), device='cuda',
                    clip_model=clip_model, clip_preprocess=clip_preprocess,
                    clip_tokenizer=clip_tokenizer, aesthetic_model=aesthetic_model,
                    dwpose=dwpose, t5=t5
                )
            except Exception as e:
                print(f"Failed to evaluate {ckpt_path}: {e}")
                continue
        else:
            print(f"\n[{step}] Found existing evaluation results. Skipping generation.")
            
        # Read the aggregated summary
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
            
            summary = data.get("summary", {})
            trajectory_data.append({
                "step": step,
                "aesthetic_score": summary.get("mean_aesthetic_score", 0),
                "clip_score": summary.get("mean_clip_score", 0),
                "face_confidence": summary.get("mean_dwpose_face_conf", 0)
            })
        except Exception as e:
            print(f"Error reading results for step {step}: {e}")
            
    # Save CSV
    if trajectory_data:
        df = pd.DataFrame(trajectory_data)
        df = df.sort_values(by="step")
        csv_path = exp_path / "metric_trajectory.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nTrajectory CSV saved to {csv_path}")
        print("\nTrajectory Summary:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_trajectory.py <experiment_dir>")
        sys.exit(1)
        
    evaluate_all_checkpoints(sys.argv[1])
