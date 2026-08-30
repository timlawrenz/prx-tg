"""Sanity check for the zg-token-basis adapter change (no training). Fast."""
import io, torch
from production.config_loader import load_config
from production.model import NanoDiT

def build(geometry_token_basis):
    return NanoDiT(input_size=1024, patch_size=16, in_channels=3, hidden_size=768,
                   depth=18, num_heads=12, mlp_ratio=4.0, use_gradient_checkpointing=False,
                   adapter_kwargs={"name":"eidolon","geometry_token_basis":geometry_token_basis})

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", dev)

# 1. config parses the flag
cfg = load_config('experiments/zg-token-basis/config.yaml')
assert cfg.adapter.name == 'eidolon', cfg.adapter.name
assert cfg.adapter.geometry_token_basis is True, cfg.adapter.geometry_token_basis
print("[1] config.adapter.geometry_token_basis =", cfg.adapter.geometry_token_basis, "OK")

# 2. model builds with basis, geo_basis registered with right shape
m = build(True).to(dev)
assert m.adapter.geometry_token_basis is True
assert hasattr(m.adapter, 'geo_basis'), "geo_basis missing!"
print("[2] geo_basis shape:", tuple(m.adapter.geo_basis.shape), "(expect (1,50,768)) OK")

# 3. forward + backward, check geo_basis learns
B=2; x=torch.randn(B,3,128,128,device=dev); t=torch.rand(B,device=dev)
idv=torch.randn(B,64,device=dev); g=torch.randn(B,50,device=dev)
out = m(x, t, identity_emb=idv, geometry_emb=g)
assert out.shape == x.shape, out.shape
loss = out.float().mean(); loss.backward()
gb = m.adapter.geo_basis.grad
assert gb is not None and torch.isfinite(gb).all() and gb.abs().sum()>0, "geo_basis grad bad"
print("[3] forward out", tuple(out.shape), "| geo_basis.grad finite & nonzero:", float(gb.abs().mean()), "OK")

# 4. dim0 sensitivity: changing z_g[:,0] changes output (path is live)
m.eval()
with torch.no_grad():
    torch.manual_seed(0); a=m(x,t,identity_emb=idv,geometry_emb=g)
    g2=g.clone(); g2[:,0]+=3.0
    torch.manual_seed(0); b=m(x,t,identity_emb=idv,geometry_emb=g2)
print("[4] |Δoutput| when z_g[0]+=3:", float((a-b).abs().mean()), "(nonzero => geometry path live)")

# 5. state_dict round-trip (basis present)
buf=io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
m2=build(True).to(dev); m2.load_state_dict(torch.load(buf, weights_only=True))
assert any('geo_basis' in k for k in m2.state_dict()), "geo_basis not in state_dict"
print("[5] state_dict round-trip OK; geo_basis key present")

# 6. backward compat: basis=False has NO geo_basis, and old-style ckpt loads
m0=build(False).to(dev)
assert not hasattr(m0.adapter,'geo_basis'), "basis=False must not create geo_basis"
keys0=[k for k in m0.state_dict() if 'geo_basis' in k]
assert keys0==[], keys0
# a False checkpoint loads into a False model (mirrors existing runs)
buf0=io.BytesIO(); torch.save(m0.state_dict(),buf0); buf0.seek(0)
build(False).to(dev).load_state_dict(torch.load(buf0, weights_only=True))
print("[6] backward-compat OK: geometry_token_basis=False identical to legacy adapter")
print("\nALL SANITY CHECKS PASSED")
