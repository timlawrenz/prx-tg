import torch
import torch.nn.functional as F
import time
import math

def nystrom_attention(q, k, v, m=64):
    B, H, N, D = q.shape
    scale = D ** -0.5
    
    # 1. Landmark selection (uniform stepped for simplicity of benchmark)
    # Using slice is actually slightly FASTER than gather, giving Nystrom the benefit of the doubt
    step = N // m
    q_land = q[:, :, step//2::step]  # (B, H, m, D)
    k_land = k[:, :, step//2::step]
    
    # Reviewer's note: Q_land * K_land^T is not PSD. 
    # Standard Nystrom uses K_land * K_land^T for the Gram matrix, but let's test the plan's math
    # Or rather, the exact Nystromformer math: A_tilde = softmax(Q_land @ K_land^T)
    attn_land = F.softmax((q_land @ k_land.transpose(-2, -1)) * scale, dim=-1)  # (B, H, m, m)
    attn_cross_q = F.softmax((q @ k_land.transpose(-2, -1)) * scale, dim=-1)    # (B, H, N, m)
    attn_cross_k = F.softmax((q_land @ k.transpose(-2, -1)) * scale, dim=-1)    # (B, H, m, N)
    
    # SVD pseudoinverse in float32 for stability
    # (Doing this in bfloat16 often causes cuSOLVER to fail or be wildly unstable)
    attn_land_f32 = attn_land.float()
    U, S, Vt = torch.linalg.svd(attn_land_f32)
    S_inv = torch.where(S > 1e-6, 1.0 / S, 0.0)
    attn_land_inv = (Vt.transpose(-2, -1) * S_inv.unsqueeze(-2)) @ U.transpose(-2, -1)
    attn_land_inv = attn_land_inv.to(q.dtype)
    
    # Final multiplication: K_cross_q @ A_inv @ K_cross_k @ V
    # Order matters for FLOPs:
    # 1. k_cross_k @ V -> (B, H, m, D)
    # 2. A_inv @ (K_cross_k @ V) -> (B, H, m, D)
    # 3. K_cross_q @ (...) -> (B, H, N, D)
    step1 = attn_cross_k @ v
    step2 = attn_land_inv @ step1
    out = attn_cross_q @ step2
    
    return out

def benchmark(fn, q, k, v, name, warmup=20, iters=100, **kwargs):
    # Warmup
    for _ in range(warmup):
        out = fn(q, k, v, **kwargs)
        loss = out.sum()
        loss.backward()
        
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(iters):
        out = fn(q, k, v, **kwargs)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Clear grads to avoid OOM
    q.grad = None
    k.grad = None
    v.grad = None
    
    avg_ms = ((end_time - start_time) / iters) * 1000
    print(f"{name: <25} | {avg_ms:.2f} ms per fwd+bwd step")

def main():
    B, H, N, D = 4, 12, 4096, 64
    print(f"Benchmarking with B={B}, H={H}, N={N}, D={D}, dtype=bfloat16")
    print("-" * 50)
    
    # Initialize tensors
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    # 1. SDPA (FlashAttention-2)
    def sdpa_baseline(q, k, v):
        # is_causal=False is default
        return F.scaled_dot_product_attention(q, k, v)
        
    benchmark(sdpa_baseline, q, k, v, "SDPA (FlashAttention-2)")
    
    # 2. Nystrom m=64
    benchmark(nystrom_attention, q, k, v, "Nystrom (m=64)", m=64)
    
    # 3. Nystrom m=128
    benchmark(nystrom_attention, q, k, v, "Nystrom (m=128)", m=128)

if __name__ == "__main__":
    main()
