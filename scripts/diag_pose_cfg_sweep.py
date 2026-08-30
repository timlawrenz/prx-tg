#!/usr/bin/env python3
"""
DIAGNOSTIC: does the DiT bind z_g dim0 (=yaw, |corr|=0.97, ~14deg/sigma) to output pose?
Separates two hypotheses for the "always frontal" failure:
  (H1) AuraFace identity-stream pose-leak lets the model cheat pose -> ignores z_g.
       Prediction: head rotates under GEOMETRY-ONLY CFG (identity dropped) but not full CFG.
  (H2) Undertraining / no binding at 5k steps.
       Prediction: head stays frontal even under geometry-only CFG + wide sweep.

Noise-matched (same seed before every gen) so differences are purely conditioning.
"""
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))
from production.config_loader import load_config
from production.model import NanoDiT
from production.sample import EulerSampler, tensor_to_pil
from production.data import get_deterministic_validation_dataloader

RUNS = "/mnt/nas-ai-models/training-data/prx-tg"
CKPTS = [
  ("hegre-geometry", "experiments/hegre-geometry/config.yaml",
   f"{RUNS}/hegre-geometry/runs/2026-07-05_2053/checkpoints/checkpoint_final.pt",
   f"{RUNS}/hegre-geometry/runs/2026-07-05_2053/_diag_pose_cfg"),
  ("eidolon-conditioning", "experiments/eidolon-conditioning/config.yaml",
   f"{RUNS}/eidolon-conditioning/runs/2026-06-30_2202/checkpoints/checkpoint_final.pt",
   f"{RUNS}/eidolon-conditioning/runs/2026-06-30_2202/_diag_pose_cfg"),
  # Arm N: per-dim z_g token basis (geometry_token_basis=true). The config carries
  # the flag, so run_ckpt builds the model correctly and loads geo_basis. Runs are
  # skipped automatically until checkpoint_final.pt exists.
  ("zg-token-basis", "experiments/zg-token-basis/config.yaml",
   f"{RUNS}/zg-token-basis/runs/2026-07-08_1327/checkpoints/checkpoint_final.pt",
   f"{RUNS}/zg-token-basis/runs/2026-07-08_1327/_diag_pose_cfg"),
]
SWEEP_DIM = 0
SWEEP_VALUES = [-3.0, -1.5, 0.0, 1.5, 3.0]      # ~ +/-42 deg yaw at 14 deg/sigma
SAMPLE_INDICES = [10, 30, 50]
CFG_CONFIGS = [("fullcfg_id3_geo2", 3.0, 2.0), ("geoonly_id0_geo4", 0.0, 4.0)]
NUM_STEPS = 50
SEED = 1234

def grid(rows, spacing=8, labels_left=None):
    """rows: list of list of (3,H,W) tensors in [-1,1]. Build a grid image."""
    pil_rows=[]
    for r in rows:
        imgs=[]
        for img in r:
            a=(img.cpu().numpy()*0.5+0.5).clip(0,1)
            imgs.append(Image.fromarray((a.transpose(1,2,0)*255).astype(np.uint8)))
        pil_rows.append(imgs)
    h=pil_rows[0][0].height; w=pil_rows[0][0].width
    ncol=max(len(r) for r in pil_rows); nrow=len(pil_rows)
    W=ncol*w+(ncol-1)*spacing; H=nrow*h+(nrow-1)*spacing
    canvas=Image.new('RGB',(W,H),(255,255,255))
    for ri,r in enumerate(pil_rows):
        for ci,im in enumerate(r):
            canvas.paste(im,(ci*(w+spacing), ri*(h+spacing)))
    return canvas

def run_ckpt(name, cfg_path, ckpt_path, out_dir):
    if not Path(ckpt_path).exists():
        print(f"\n######## {name} ######## SKIP: checkpoint not found yet: {ckpt_path}")
        return
    print(f"\n######## {name} ########\n  ckpt={ckpt_path}")
    config=load_config(cfg_path)
    kw=dict(input_size=config.model.input_size, patch_size=config.model.patch_size,
            in_channels=config.model.in_channels, hidden_size=config.model.hidden_size,
            depth=config.model.depth, num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio, use_gradient_checkpointing=False)
    ad=getattr(config,'adapter',None)
    if ad: kw['adapter_kwargs']={'name':ad.name,'identity_dim':getattr(ad,'identity_dim',64),
                                 'z_g_dim':getattr(ad,'z_g_dim',50),
                                 'geometry_token_basis':getattr(ad,'geometry_token_basis',False)}
    model=NanoDiT(**kw)
    ckpt=torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])   # training weights, NOT ema (matches visual_debug)
    dev=torch.device('cuda'); model=model.to(dev).eval()

    loader=get_deterministic_validation_dataloader(shard_dir=None, batch_size=1,
        target_latent_size=config.model.input_size, source='stratum',
        stratum_dir=config.data.stratum_dir, adapter_name='eidolon')
    maxi=max(SAMPLE_INDICES); samples=[]; it=iter(loader); cnt=0
    while cnt<=maxi:
        b=next(it)
        for i in range(b['image_data'].shape[0]) if 'image_data' in b else range(b['identity_emb'].shape[0]):
            if cnt<=maxi:
                samples.append({'identity_emb':b['identity_emb'][i],'geometry_emb':b['geometry_emb'][i],
                                'image_id':b['image_ids'][i]}); cnt+=1
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sampler=EulerSampler(num_steps=NUM_STEPS)
    for si in SAMPLE_INDICES:
        s=samples[si]
        idv=s['identity_emb'].unsqueeze(0).to(dev); baseg=s['geometry_emb'].unsqueeze(0).to(dev)
        print(f"  -- sample {si} (id={s['image_id']}) base z_g[0]={float(baseg[0,0]):+.2f}")
        rows=[]
        for cname, id_scale, geo_scale in CFG_CONFIGS:
            row=[]
            for val in SWEEP_VALUES:
                g=baseg.clone(); g[:,SWEEP_DIM]=val
                torch.manual_seed(SEED)          # noise-matched across ALL gens
                with torch.no_grad():
                    o=sampler.sample(model=model, shape=(1,3,1024,1024),
                        identity_emb=idv, geometry_emb=g, device=dev,
                        text_scale=id_scale, dino_scale=geo_scale, prediction_type='x_prediction')
                t=o[0].clamp(0,1)*2-1; row.append(t)
                tensor_to_pil(t).save(out/f"s{si:02d}_{cname}_dim0_{val:+.1f}.png")
            rows.append(row)
            print(f"     {cname}: done")
        g_img=grid(rows)
        p=out/f"s{si:02d}_GRID_rows=[full,geoonly]_cols=dim0[{','.join(f'{v:+.1f}' for v in SWEEP_VALUES)}].png"
        g_img.save(p); print(f"     GRID -> {p}")
    del model; torch.cuda.empty_cache()

if __name__=='__main__':
    for c in CKPTS:
        run_ckpt(*c)
    print("\nALL DONE.")
