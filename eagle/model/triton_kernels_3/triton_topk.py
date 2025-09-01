# triton_topk_fixed2.py
# Stage-1 Triton tile top-K (no register arrays! direct masked stores), Stage-2 PyTorch reduce.
# Includes sanity tests & benchmark.

import math
import argparse
import time
from typing import Tuple
import torch

try:
    import triton
    import triton.language as tl
except Exception as e:
    raise SystemExit("Triton is required. Install with `pip install triton`.") from e


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


@triton.jit
def _topk_stage1_kernel(
    scores_ptr,                           # *float* [B,S,V]
    cand_vals_ptr, cand_idx_ptr,          # *float32*, *int32* [B,S,NB,K]
    B, S, V,
    stride_sb, stride_ss, stride_sv,      # strides of scores
    stride_vb, stride_vs, stride_vnb, stride_vk,   # strides of cand_vals
    stride_ib, stride_is, stride_inb, stride_ik,   # strides of cand_idx
    K_ACTUAL,                             # runtime K (<= K_CAP)
    K_CAP: tl.constexpr,                  # compile-time Po2 register loop count
    BLOCK_V: tl.constexpr,                # compile-time Po2 tile over vocab
):
    """
    For each (b, s, nb), select K_CAP winners within a vocab tile of size BLOCK_V, by
    iterating 'argmax + knock-out'. We directly write each iteration's winner to the
    candidate tensors, but only for k < K_ACTUAL (runtime predicate). No register arrays.
    """
    pid_b  = tl.program_id(0)
    pid_s  = tl.program_id(1)
    pid_nb = tl.program_id(2)

    # tile indices
    offs_v = pid_nb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    base = (pid_b * stride_sb) + (pid_s * stride_ss) + offs_v * stride_sv
    vals = tl.load(scores_ptr + base, mask=mask_v, other=-float('inf')).to(tl.float32)

    ar = tl.arange(0, BLOCK_V)

    out_base_v = (pid_b * stride_vb) + (pid_s * stride_vs) + (pid_nb * stride_vnb)
    out_base_i = (pid_b * stride_ib) + (pid_s * stride_is) + (pid_nb * stride_inb)

    # pick winners; write directly with runtime predicate (k < K_ACTUAL)
    for k in range(K_CAP):
        vmax = tl.max(vals, axis=0)            # scalar
        arg  = tl.argmax(vals, axis=0)         # scalar in [0, BLOCK_V)
        gidx = (pid_nb * BLOCK_V + arg).to(tl.int32)
        # predicate: only write first K_ACTUAL winners
        do_store = k < K_ACTUAL
        tl.store(cand_vals_ptr + out_base_v + k * stride_vk, vmax, mask=do_store)
        tl.store(cand_idx_ptr  + out_base_i + k * stride_ik, gidx, mask=do_store)
        # knock out picked position
        vals = tl.where(ar == arg, -float('inf'), vals)


def triton_topk(scores: torch.Tensor, K: int, BLOCK_V: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stage-1 (Triton): per-block (BLOCK_V) top-K, writing K winners per block to [B,S,NB,K].
    Stage-2 (PyTorch): reduce [NB*K] -> K per (B,S).

    Args
    ----
    scores : [B,S,V] float16/float32 CUDA tensor
    K      : number of top elements per (B,S)
    BLOCK_V: vocab tile size (power of two: 512/1024/2048/4096)

    Returns
    -------
    vals : [B,S,K] same dtype as scores
    idx  : [B,S,K] int32 indices into vocab
    """
    assert scores.is_cuda and scores.ndim == 3
    B, S, V = scores.shape
    assert 1 <= K <= V
    assert (BLOCK_V & (BLOCK_V - 1)) == 0, "BLOCK_V must be a power of two"

    K_CAP = next_pow2(K)           # compile-time register loop count
    NB = math.ceil(V / BLOCK_V)    # number of vocab tiles

    # Candidate buffers [B,S,NB,K]
    cand_vals = torch.empty((B, S, NB, K), device=scores.device, dtype=torch.float32)
    cand_idx  = torch.empty((B, S, NB, K), device=scores.device, dtype=torch.int32)

    sb, ss, sv = scores.stride()
    cvb, cvs, cvnb, cvk = cand_vals.stride()
    cib, cis, cinb, cik = cand_idx.stride()

    grid1 = (B, S, NB)
    _topk_stage1_kernel[grid1](
        scores, cand_vals, cand_idx,
        B, S, V,
        sb, ss, sv,
        cvb, cvs, cvnb, cvk,
        cib, cis, cinb, cik,
        K,                             # K_ACTUAL runtime
        K_CAP=K_CAP,                   # constexpr
        BLOCK_V=BLOCK_V,               # constexpr
    )

    # Stage-2 reduce: [B,S,NB,K] -> [B,S,K]
    cand_vals_flat = cand_vals.view(B, S, NB * K)
    cand_idx_flat  = cand_idx.view(B, S, NB * K)

    top_vals, top_pos = torch.topk(cand_vals_flat, k=K, dim=-1)          # fp32
    top_idx = torch.gather(cand_idx_flat, dim=-1, index=top_pos)         # int32

    return top_vals.to(dtype=scores.dtype), top_idx


# ---------------------------
# Reference + testers
# ---------------------------

def torch_topk_ref(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(x, k, dim=-1)

def check_correctness(B=2, S=3, V=10000, K=10, dtype="float16", block_v=2048, seed=0) -> None:
    torch.manual_seed(seed)
    dt = torch.float16 if dtype == "float16" else torch.float32
    x = torch.randn(B, S, V, device="cuda", dtype=dt)

    v_ref, i_ref = torch_topk_ref(x, K)
    v_trt, i_trt = triton_topk(x, K, BLOCK_V=block_v)

    # Compare via gathered values (tie-agnostic)
    g_ref = torch.gather(x.float(), -1, i_ref.long())
    g_trt = torch.gather(x.float(), -1, i_trt.long())

    ok_vals = torch.allclose(g_trt, g_ref, rtol=1e-3, atol=1e-3)
    if not ok_vals:
        max_diff = (g_trt - g_ref).abs().max().item()
        raise AssertionError(f"Value mismatch: max abs diff={max_diff}")

    # Optional: check non-increasing order along K
    ref_sorted = (g_ref[..., 1:] <= g_ref[..., :-1] + 1e-6).all().item()
    trt_sorted = (g_trt[..., 1:] <= g_trt[..., :-1] + 1e-6).all().item()
    print(f"[sanity] OK values={ok_vals}, sorted_ref={ref_sorted}, sorted_triton={trt_sorted}")

def benchmark(B=2, S=2, V=128_256, K=10, dtype="float16", block_v=2048, iters=50, seed=0) -> None:
    torch.manual_seed(seed)
    dt = torch.float16 if dtype == "float16" else torch.float32
    x = torch.randn(B, S, V, device="cuda", dtype=dt)

    # warmup
    for _ in range(10):
        torch_topk_ref(x, K)
        triton_topk(x, K, BLOCK_V=block_v)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        v_ref, i_ref = torch_topk_ref(x, K)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(iters):
        v_trt, i_trt = triton_topk(x, K, BLOCK_V=block_v)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    torch_ms  = (t1 - t0) * 1e3 / iters
    triton_ms = (t3 - t2) * 1e3 / iters

    g_ref = torch.gather(x.float(), -1, i_ref.long())
    g_trt = torch.gather(x.float(), -1, i_trt.long())
    ok_vals = torch.allclose(g_trt, g_ref, rtol=1e-3, atol=1e-3)

    print(f"[bench] torch.topk:  {torch_ms:7.3f} ms/iter")
    print(f"[bench] triton_topk: {triton_ms:7.3f} ms/iter  (BLOCK_V={block_v}, K_CAP={next_pow2(K)})")
    print(f"[bench] speedup x{torch_ms / triton_ms: .2f}, correctness={ok_vals}")


def main():
    parser = argparse.ArgumentParser(description="Triton Top-K (stage-1 masked stores + PyTorch reduce)")
    parser.add_argument("--mode", choices=["sanity", "bench"], default="sanity")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S", type=int, default=3)
    parser.add_argument("--V", type=int, default=10000)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--block_v", type=int, default=2048, help="power-of-two vocab tile (e.g., 1024/2048/4096)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    if args.mode == "sanity":
        print(f"Running sanity: B={args.B}, S={args.S}, V={args.V}, K={args.K}, dtype={args.dtype}, BLOCK_V={args.block_v}")
        check_correctness(B=args.B, S=args.S, V=args.V, K=args.K, dtype=args.dtype, block_v=args.block_v, seed=args.seed)
    else:
        print(f"Running bench:  B={args.B}, S={args.S}, V={args.V}, K={args.K}, dtype={args.dtype}, BLOCK_V={args.block_v}, iters={args.iters}")
        benchmark(B=args.B, S=args.S, V=args.V, K=args.K, dtype=args.dtype, block_v=args.block_v, iters=args.iters, seed=args.seed)

if __name__ == "__main__":
    main()
