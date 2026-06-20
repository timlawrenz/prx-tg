#!/usr/bin/env python3
"""
Robust Validation Evaluator for PRX-TG.

Evaluates a specific checkpoint on 10 fixed prompts to measure:
1. DWPose Confidence (face coherence & spatial stability)
2. CLIP Score (text-image alignment)
3. LAION Aesthetic Score (general quality & artifact detection)

Creates a collage and outputs a JSON metrics file.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
try:
    import open_clip
except ImportError:
    print("FATAL: open_clip_torch is required. (pip install open_clip_torch)")
    sys.exit(1)

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production.config_loader import load_config
from production.sample import ValidationSampler
from production.validate import create_image_collage
from scripts.dwpose_onnx import DWPoseDetector
from scripts.txt2img import T5Encoder, tensor_to_pil, load_model

# -----------------------------------------------------------------------------
# Fixed Prompt Set (10 prompts)
# -----------------------------------------------------------------------------
FIXED_PROMPTS = [
    "A young woman with pale skin and long straight brown hair, smiling, looking directly at the camera",
    "An elderly man with dark skin and a bald head, wearing glasses, facing slightly left, neutral expression",
    "A middle-aged Asian woman with short black hair and small earrings, serious expression, facing slightly right",
    "A young man with tan skin and short dark hair, looking away from the camera, surprised expression",
    "A mature man with a gray beard and thinning hair, wearing glasses, looking directly at the camera, neutral expression",
    "A young woman with curly red hair and freckles, laughing, facing slightly left",
    "A mature woman with tan skin and silver hair, wearing small earrings, neutral expression, facing slightly right",
    "A young child with brown skin and short dark hair, looking directly at the camera, neutral expression",
    "An elderly woman with blonde hair and pale skin, wearing glasses, facing slightly left, smiling",
    "A young woman with long blonde hair and a full beard, looking directly at the camera, neutral expression"
]

FIXED_SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]

# -----------------------------------------------------------------------------
# Aesthetic Predictor Model
# -----------------------------------------------------------------------------
class MLPUpdate(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layers(x)

def load_aesthetic_predictor(device):
    print("  Loading LAION Aesthetic Predictor V2...")
    model = MLPUpdate(768)
    cached_path = hf_hub_download(
        repo_id="camenduru/improved-aesthetic-predictor", 
        filename="sac+logos+ava1-l14-linearMSE.pth"
    )
    state_dict = torch.load(cached_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Main Evaluator Logic
# -----------------------------------------------------------------------------
def run_evaluation(checkpoint_path, config_path, output_dir, device='cuda', clip_model=None, clip_preprocess=None, clip_tokenizer=None, aesthetic_model=None, dwpose=None, t5=None):
    print(f"--- PRX-TG Checkpoint Evaluator ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output Dir: {output_dir}\n")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load Evaluation Models (if not provided) ---
    if clip_model is None:
        print("[1/3] Loading Evaluation Models...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device=device)
        clip_model.eval()
        clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
        aesthetic_model = load_aesthetic_predictor(device)
        print("  Loading DWPose (ONNX CPU)...")
        dwpose = DWPoseDetector(device="cpu")
    else:
        print("[1/3] Using pre-loaded Evaluation Models...")
    
    # --- 2. Load Generation Models ---
    import yaml
    
    print("\n[2/3] Loading Generation Models...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model = load_model(checkpoint_path, config, torch.device(device))
    model.eval()
    
    if t5 is None:
        device_obj = torch.device(device)
        t5 = T5Encoder(device_obj)
    
    mc = config.get("model", {})
    sc = config.get("sampling", {})
    
    sampler = ValidationSampler(
        model,
        vae=None,
        device=device,
        num_steps=sc.get("num_steps", 35),
        text_scale=sc.get("text_scale", 3.0),
        dino_scale=0.0, # Pure Text-to-image mode
        self_guidance=False, # MUST BE FALSE for dino_scale=0 to work properly via Dual CFG
        guidance_scale=sc.get("guidance_scale", 3.0),
        prediction_type=mc.get("prediction_type", "x_prediction")
    )

    # --- 3. Run Generation & Evaluation ---
    print("\n[3/3] Generating & Evaluating 10 Samples...")
    
    results = {}
    pil_images = []
    
    for i, (prompt, seed) in enumerate(zip(FIXED_PROMPTS, FIXED_SEEDS)):
        t_start = time.time()
        
        # Generation
        torch.manual_seed(seed)
        text_emb, text_mask = t5.encode(prompt)
        
        # Pass dummy DINO
        b, _, h, w = 1, 3, 1024, 1024
        dino_emb = torch.zeros(1, 1024, device=device)
        dino_patches = torch.zeros(1, 256, 1024, device=device)
        
        gen_tensor = sampler.generate(
            dino_emb=dino_emb,
            dino_patches=dino_patches,
            text_emb=text_emb,
            text_mask=text_mask,
            latent_size=h,
            self_guidance=False, # Override instance setting to be safe
            dino_scale=0.0
        )
        
        pil_img = tensor_to_pil(gen_tensor[0])
        pil_images.append(pil_img)
        
        # --- Evaluate: DWPose ---
        # convert to uint8 HWC BGR for cv2/onnx
        img_cv2 = np.array(pil_img)[:, :, ::-1] 
        keypoints, scores, bboxes = dwpose(img_cv2, single_person=True)
        
        if len(scores) > 0:
            face_scores = scores[0][23:91] # COCO-WholeBody face keypoints are 23-90
            mean_face_conf = float(np.mean(face_scores))
            total_face_kp = int(np.sum(face_scores > 0.1))
        else:
            mean_face_conf = 0.0
            total_face_kp = 0
            
        # --- Evaluate: CLIP & Aesthetic ---
        clip_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
        text_tokens = clip_tokenizer(prompt).to(device)
        
        with torch.no_grad():
            img_features = clip_model.encode_image(clip_input)
            text_features = clip_model.encode_text(text_tokens)
            
            img_features /= img_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            clip_score = float((img_features @ text_features.T).item())
            
            aes_score = float(aesthetic_model(img_features).item())
            
        results[f"prompt_{i}"] = {
            "prompt": prompt,
            "seed": seed,
            "clip_score": clip_score,
            "aesthetic_score": aes_score,
            "dwpose_mean_face_conf": mean_face_conf,
            "dwpose_face_kp_count": total_face_kp,
            "time_sec": time.time() - t_start
        }
        
        print(f"  [{i+1}/10] Aesthetic: {aes_score:.2f} | CLIP: {clip_score:.3f} | FaceConf: {mean_face_conf:.3f}")

    # Summary
    mean_aes = np.mean([r["aesthetic_score"] for r in results.values()])
    mean_clip = np.mean([r["clip_score"] for r in results.values()])
    mean_face_conf = np.mean([r["dwpose_mean_face_conf"] for r in results.values()])
    
    results["summary"] = {
        "mean_aesthetic_score": mean_aes,
        "mean_clip_score": mean_clip,
        "mean_dwpose_face_conf": mean_face_conf
    }
    
    print(f"\n--- SUMMARY ---")
    print(f"Mean Aesthetic Score: {mean_aes:.2f}")
    print(f"Mean CLIP Score:      {mean_clip:.3f}")
    print(f"Mean Face Confidence: {mean_face_conf:.3f}")
    
    # Outputs
    with open(out_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save individual prompt images (for TensorBoard)
    for i, pil_img in enumerate(pil_images):
        pil_img.save(out_path / f"prompt_{i:02d}.png")
        
    collage_img = Image.fromarray(create_image_collage(pil_images, spacing=20))
    collage_img.save(out_path / "collage.png")
    
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PRX-TG checkpoint.")
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--config", help="Optional path to config.yaml", default=None)
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    # If config not provided, assume it's in the checkpoint's parent parent directory
    config_path = args.config
    if not config_path:
        cp = Path(args.checkpoint).resolve()
        config_path = str(cp.parent.parent / "config.yaml")

    run_evaluation(args.checkpoint, config_path, args.output_dir)
