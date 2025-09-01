# fused_tree_attention_triton.py
# FlashAttention-style streaming softmax with tree/causal mask in Triton.
# Usage:
#   python fused_tree_attention_triton.py --mode sanity
#   python fused_tree_attention_triton.py --mode bench --B 1 --H 32 --Lq 128 --Lk 1024 --D 128
#
# Drop-in wrapper:
#   out = tree_attention_fused(q, k, v, mask=mask, scale=None,
#                              BLOCK_M=64, BLOCK_N=64, BLOCK_D=64)

import argparse
import math
import time
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_tree_attn_kernel(
    # Pointers
    q_ptr, k_ptr, v_ptr, mask_ptr, o_ptr,
    # Sizes
    B, H, Lq, Lk, D, MH,
    # Strides (elements)
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_ob, stride_oh, stride_om, stride_od,
    # Scale
    scale,
    # Tiles
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Grid = (ceil_div(Lq, BLOCK_M), B*H)
    Q/K/V: [B,H,L,D], mask: [B,MH,Lq,Lk] additive (0 or -inf), O: [B,H,Lq,D] **fp32**
    """

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    # rows for this query block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < Lq

    # streaming-softmax state (fp32)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    for start_n in range(0, Lk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < Lk

        # ---- S = Q K^T (BM x BN), fp32 ----
        S = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for start_d in range(0, D, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < D

            q_blk = (q_ptr + b*stride_qb + h*stride_qh
                     + offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qd)
            k_blk = (k_ptr + b*stride_kb + h*stride_kh
                     + offs_n[:, None]*stride_kn + offs_d[None, :]*stride_kd)

            q = tl.load(q_blk, mask=(m_mask[:, None] & d_mask[None, :]), other=0.0).to(tl.float32)
            k = tl.load(k_blk, mask=(n_mask[:, None] & d_mask[None, :]), other=0.0).to(tl.float32)

            S += tl.dot(q, tl.trans(k))

        S = S * scale

        # mask add (broadcast heads if MH==1 by zero stride_mh set at launch)
        m_blk = (mask_ptr + b*stride_mb + h*stride_mh
                 + offs_m[:, None]*stride_mm + offs_n[None, :]*stride_mn)
        M = tl.load(m_blk, mask=(m_mask[:, None] & n_mask[None, :]), other=-float("inf")).to(tl.float32)
        S = S + M

        # tile softmax stats
        m_ij = tl.max(S, axis=1)                  # [BM]
        P    = tl.exp(S - m_ij[:, None])          # [BM,BN]
        l_ij = tl.sum(P, axis=1)                  # [BM]

        # streaming update
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i  - m_new)
        beta  = tl.exp(m_ij - m_new)
        denom = alpha * l_i + beta * l_ij + EPS

        # *** FIX: include l_i in the old-output scale ***
        scale_old = (alpha * l_i) / denom         # [BM]
        scale_new = (beta)        / denom         # [BM]

        m_i = m_new
        l_i = denom

        # O = O * scale_old + (P @ V) * scale_new   (all fp32)
        for start_d in range(0, D, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < D

            o_blk = (o_ptr + b*stride_ob + h*stride_oh
                     + offs_m[:, None]*stride_om + offs_d[None, :]*stride_od)
            v_blk = (v_ptr + b*stride_vb + h*stride_vh
                     + offs_n[:, None]*stride_vn + offs_d[None, :]*stride_vd)

            O = tl.load(o_blk, mask=(m_mask[:, None] & d_mask[None, :]), other=0.0).to(tl.float32)
            V = tl.load(v_blk, mask=(n_mask[:, None] & d_mask[None, :]), other=0.0).to(tl.float32)

            PV = tl.dot(P, V)  # [BM,BD]
            O  = O * scale_old[:, None] + PV * scale_new[:, None]

            tl.store(o_blk, O, mask=(m_mask[:, None] & d_mask[None, :]))


def tree_attention_fused(
    q: torch.Tensor,  # [B,H,Lq,D]
    k: torch.Tensor,  # [B,H,Lk,D]
    v: torch.Tensor,  # [B,H,Lk,D]
    mask: Optional[torch.Tensor] = None,  # [B,1 or H,Lq,Lk], additive (0 or -inf)
    scale: Optional[float] = None,
    *,
    BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_D: int = 64, eps: float = 1e-6,
) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    B,H,Lq,D = q.shape
    assert k.shape == (B,H,k.shape[2],D) and v.shape == (B,H,k.shape[2],D)
    Lk = k.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if mask is None:
        mask = torch.zeros((B,1,Lq,Lk), device=q.device, dtype=torch.float32)
    else:
        mask = mask.to(torch.float32)
        assert mask.shape == (B, mask.shape[1], Lq, Lk) and mask.shape[1] in (1, H)

    MH = mask.shape[1]
    # *** FIX 2: keep O as fp32 inside the kernel, cast back on return ***
    out_fp32 = torch.zeros((B,H,Lq,D), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(Lq, BLOCK_M), B*H)
    _fused_tree_attn_kernel[grid](
        q, k, v, mask, out_fp32,
        B, H, Lq, Lk, D, MH,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mask.stride(0), mask.stride(1) if MH == H else 0, mask.stride(2), mask.stride(3),
        out_fp32.stride(0), out_fp32.stride(1), out_fp32.stride(2), out_fp32.stride(3),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, EPS=eps,
    )
    return out_fp32.to(q.dtype)


# ---------------------------
# reference / tests / bench
# ---------------------------

def _sdpa_ref(q, k, v, mask, scale):
    B,H,Lq,D = q.shape
    Lk = k.shape[2]
    qf = q.reshape(B*H, Lq, D)
    kf = k.reshape(B*H, Lk, D)
    vf = v.reshape(B*H, Lk, D)
    if mask.shape[1] == 1:
        mf = mask.expand(B,H,Lq,Lk).reshape(B*H, Lq, Lk)
    else:
        mf = mask.reshape(B*H, Lq, Lk)
    out = torch.nn.functional.scaled_dot_product_attention(
        qf.to(torch.float32), kf.to(torch.float32), vf.to(torch.float32),
        attn_mask=mf, dropout_p=0.0, is_causal=False, scale=scale
    )
    return out.reshape(B,H,Lq,D)


def _make_random_mask(B,H,Lq,Lk, p_block=0.05, per_head=False, device="cuda"):
    base = torch.full((B,1,Lq,Lk), 0.0, device=device, dtype=torch.float32)
    i = torch.arange(Lq, device=device)[:, None]
    j = torch.arange(Lk, device=device)[None, :]
    causal = (j > i).to(torch.float32) * (-1e9)
    base = base + causal
    if p_block > 0:
        rnd = (torch.rand(B,1,Lq,Lk, device=device) < p_block)
        base[rnd] = -1e9
    if per_head:
        base = base.expand(B,H,Lq,Lk).clone()
    return base  # [B,1 or H,Lq,Lk]


def run_sanity():
    torch.manual_seed(0)
    device = "cuda"
    cases = [
        dict(B=1,H=8, Lq=64,  Lk=256, D=64,  dtype=torch.float16, per_head=False),
        dict(B=1,H=8, Lq=96,  Lk=128, D=128, dtype=torch.float16, per_head=True),
        dict(B=2,H=4, Lq=33,  Lk=77,  D=80,  dtype=torch.bfloat16, per_head=False),
        dict(B=1,H=16,Lq=128, Lk=512, D=64,  dtype=torch.float16, per_head=True),
    ]
    for cfg in cases:
        B,H,Lq,Lk,D = cfg["B"],cfg["H"],cfg["Lq"],cfg["Lk"],cfg["D"]
        q = torch.randn(B,H,Lq,D, device=device, dtype=cfg["dtype"])
        k = torch.randn(B,H,Lk,D, device=device, dtype=cfg["dtype"])
        v = torch.randn(B,H,Lk,D, device=device, dtype=cfg["dtype"])
        mask  = _make_random_mask(B,H,Lq,Lk, p_block=0.05, per_head=cfg["per_head"], device=device)
        scale = 1.0 / math.sqrt(D)

        out_ref   = _sdpa_ref(q, k, v, mask, scale).to(q.dtype)
        out_fused = tree_attention_fused(q, k, v, mask=mask, scale=scale,
                                         BLOCK_M=64, BLOCK_N=64, BLOCK_D=64)

        atol = 2e-2 if cfg["dtype"] == torch.float16 else 3e-2
        rtol = 2e-2 if cfg["dtype"] == torch.float16 else 3e-2
        ok   = torch.allclose(out_fused, out_ref, atol=atol, rtol=rtol)
        diff = (out_fused - out_ref).abs().max().item()
        print(f"[sanity] {cfg} -> ok={ok}, max_diff={diff:.4e}")
        if not ok:
            raise AssertionError("Sanity comparison failed.")
    print("[sanity] all good ✅")


def run_bench(B=1,H=32,Lq=128,Lk=1024,D=128,dtype="float16",
              BLOCK_M=64,BLOCK_N=64,BLOCK_D=64,iters=50,seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    q = torch.randn(B,H,Lq,D, device=device, dtype=dt)
    k = torch.randn(B,H,Lk,D, device=device, dtype=dt)
    v = torch.randn(B,H,Lk,D, device=device, dtype=dt)
    mask = _make_random_mask(B,H,Lq,Lk, p_block=0.0, per_head=False, device=device)
    scale = 1.0 / math.sqrt(D)

    # warmup
    for _ in range(10):
        _sdpa_ref(q, k, v, mask, scale)
        tree_attention_fused(q, k, v, mask=mask, scale=scale,
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _sdpa_ref(q, k, v, mask, scale)
    torch.cuda.synchronize(); t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(iters):
        tree_attention_fused(q, k, v, mask=mask, scale=scale,
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)
    torch.cuda.synchronize(); t3 = time.perf_counter()

    sdpa_ms  = (t1 - t0) * 1e3 / iters
    fused_ms = (t3 - t2) * 1e3 / iters
    print(f"[bench] PyTorch SDPA : {sdpa_ms:7.3f} ms/iter")
    print(f"[bench] Fused Triton : {fused_ms:7.3f} ms/iter  (BM={BLOCK_M}, BN={BLOCK_N}, BD={BLOCK_D})")
    print(f"[bench] speedup x{sdpa_ms / fused_ms: .2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "bench"], default="sanity")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--Lq", type=int, default=128)
    ap.add_argument("--Lk", type=int, default=1024)
    ap.add_argument("--D",  type=int, default=128)
    ap.add_argument("--dtype", choices=["float16","bfloat16","float32"], default="float16")
    ap.add_argument("--BM", type=int, default=64)
    ap.add_argument("--BN", type=int, default=64)
    ap.add_argument("--BD", type=int, default=64)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "sanity":
        run_sanity()
    else:
        run_bench(B=args.B,H=args.H,Lq=args.Lq,Lk=args.Lk,D=args.D,
                  dtype=args.dtype,BLOCK_M=args.BM,BLOCK_N=args.BN,BLOCK_D=args.BD,
                  iters=args.iters,seed=args.seed)


if __name__ == "__main__":
    main()
