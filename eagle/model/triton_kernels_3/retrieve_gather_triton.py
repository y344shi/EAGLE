# retrieve_gather_triton.py
# Triton kernel for "retrieve gather":
#   src: [B, N, F] (or [N, F]) and indices: [L, D]  --> out: [B, L, D, F]
# Sentinel index -1 is treated as "pad": output zeros for that slot.
#
# This is a drop-in GPU replacement for:
#   logits = tree_logits[0, retrieve_indices]                      # [L, D, V]
#   retrieve_hidden = hidden_state_new[:, retrieve_indices]        # [B, L, D, H]
#
# Notes:
# - Works with float16/bfloat16/float32 (and other element dtypes).
# - Supports arbitrary strides; for best perf, keep F (last dim) contiguous.
# - Autotunes BLOCK_F for feature dimension.
'''
# sanity
python retrieve_gather_triton.py --mode sanity

# bench logits-like (V large)
python retrieve_gather_triton.py --mode bench --case logits --B 1 --N 60 --F 128256 --L 42 --D 5 --dtype fp16

# bench hidden-like (H moderately large)
python retrieve_gather_triton.py --mode bench --case hidden --B 1 --N 60 --F 12288 --L 42 --D 6 --dtype fp16
'''
import argparse
import time
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_F': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_F': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_F': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_F': 512}, num_warps=8, num_stages=2),
    ],
    key=['F']
)
@triton.jit
def _gather_rows_kernel(
    src_ptr,         # *T* [B, N, F]
    idx_ptr,         # *int64* [L, D]
    dst_ptr,         # *T* [B, L, D, F]
    # sizes
    B, N, F, L, D,
    # strides (elements)
    stride_sb, stride_sn, stride_sf,    # src
    stride_il, stride_id,               # idx
    stride_ob, stride_ol, stride_od, stride_of,  # dst
    # launch coords
    # grid = (B, L, D)
    BLOCK_F: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_d = tl.program_id(2)

    # Bounds check (split to avoid chained boolean error)
    if pid_b >= B:
        return
    if pid_l >= L:
        return
    if pid_d >= D:
        return

    # Load index (int64); clamp for pointer safety
    idx = tl.load(idx_ptr + pid_l * stride_il + pid_d * stride_id).to(tl.int64)
    valid_row = (idx >= 0) & (idx < N)
    idx_safe  = tl.minimum(tl.maximum(idx, 0), N - 1)

    # Compute base pointers
    # src row = src[b, idx, :]
    # dst row = dst[b, l, d, :]
    src_row = src_ptr + pid_b * stride_sb + idx_safe * stride_sn
    dst_row = dst_ptr + pid_b * stride_ob + pid_l * stride_ol + pid_d * stride_od

    # Guard: idx in [0, N)
    valid_row = (idx >= 0) & (idx < N)

    # Copy along F in blocks
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask_f = offs < F
        vals = tl.load(src_row + offs * stride_sf, mask=(mask_f & valid_row), other=0)
        tl.store(dst_row + offs * stride_of, vals, mask=mask_f)


def _ensure_3d(src: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Ensure src is [B, N, F]; return (maybe_unsqueezed_src, added_batch_dim: bool)."""
    if src.ndim == 2:
        src = src.unsqueeze(0)  # [1, N, F]
        return src, True
    elif src.ndim == 3:
        return src, False
    else:
        raise ValueError(f"src must be 2D or 3D, got {src.shape}")


@torch.no_grad()
def retrieve_gather_triton(
    src: torch.Tensor,        # [B,N,F] or [N,F]
    indices: torch.Tensor,    # [L,D] (int64), may contain -1 as pad
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gather rows from `src` along the N-dimension using `indices`, producing [B,L,D,F].
    Positions with index == -1 are zero-filled.
    """
    assert src.is_cuda and indices.is_cuda, "inputs must be on CUDA"
    src3, squeezed = _ensure_3d(src)
    B, N, F = src3.shape
    assert indices.ndim == 2 and indices.dtype in (torch.int64, torch.long)
    L, D = indices.shape

    # Allocate output
    if out is None:
        out = torch.empty((B, L, D, F), device=src.device, dtype=src.dtype)
    else:
        assert out.shape == (B, L, D, F) and out.device == src.device and out.dtype == src.dtype

    # Strides (elements)
    stride_sb, stride_sn, stride_sf = src3.stride()
    stride_il, stride_id = indices.stride()
    stride_ob, stride_ol, stride_od, stride_of = out.stride()

    grid = (B, L, D)
    _gather_rows_kernel[grid](
        src3, indices, out,
        B, N, F, L, D,
        stride_sb, stride_sn, stride_sf,
        stride_il, stride_id,
        stride_ob, stride_ol, stride_od, stride_of,
    )
    return out if not squeezed else out[0]  # squeeze B if original was 2D


# Convenience wrappers for EAGLE call-sites -------------------------------

@torch.no_grad()
def retrieve_logits_triton(tree_logits: torch.Tensor, retrieve_indices: torch.Tensor) -> torch.Tensor:
    """
    tree_logits: [1,N,V] or [N,V]
    retrieve_indices: [L,D]
    returns: [L,D,V]
    """
    # Make logits [N,V]
    if tree_logits.ndim == 3:
        assert tree_logits.shape[0] == 1
        src = tree_logits[0]
    elif tree_logits.ndim == 2:
        src = tree_logits
    else:
        raise ValueError(f"tree_logits must be [1,N,V] or [N,V], got {tree_logits.shape}")
    return retrieve_gather_triton(src, retrieve_indices)  # -> [L,D,V]


@torch.no_grad()
def retrieve_hidden_triton(hidden_state_new: torch.Tensor, retrieve_indices: torch.Tensor) -> torch.Tensor:
    """
    hidden_state_new: [B,N,H]
    retrieve_indices: [L,D]
    returns: [B,L,D,H]
    """
    assert hidden_state_new.ndim == 3
    return retrieve_gather_triton(hidden_state_new, retrieve_indices)


# ------------------- Reference, sanity, benchmark ------------------------

def _torch_gather_ref(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Reference that matches Triton semantics for -1 pads:
    - When idx==-1, fill zeros (instead of PyTorch’s wrap-to-last behavior).
    """
    src3, squeezed = _ensure_3d(src)
    B, N, F = src3.shape
    L, D = indices.shape
    out = torch.zeros((B, L, D, F), device=src.device, dtype=src.dtype)

    valid = indices >= 0
    if valid.any():
        # Build a mask and copy only valid rows
        # We'll flatten (L,D) to one list of positions to avoid fancy slicing pitfalls
        pos = valid.nonzero(as_tuple=False)  # [M, 2] where columns are (l, d)
        l_idx = pos[:, 0]
        d_idx = pos[:, 1]
        row_idx = indices[l_idx, d_idx]  # in [0,N)
        # Gather per batch (loop over B small; B=1 or a few in EAGLE)
        for b in range(B):
            out[b, l_idx, d_idx] = src3[b, row_idx]
    return out if not squeezed else out[0]


def _rand_indices(L, D, N, pad_prob=0.1, seed=0, device="cuda"):
    torch.manual_seed(seed)
    idx = torch.randint(0, N, (L, D), device=device, dtype=torch.long)
    if pad_prob > 0:
        mask = (torch.rand(L, D, device=device) < pad_prob)
        idx[mask] = -1
    return idx


def run_sanity():
    device = "cuda"
    torch.manual_seed(0)

    # Case 1: logits-like
    N, V = 60, 128256
    L, D = 42, 5
    logits = torch.randn(1, N, V, device=device, dtype=torch.float16)
    idx = _rand_indices(L, D, N, pad_prob=0.2, seed=1, device=device)

    ref = _torch_gather_ref(logits[0], idx)       # [L,D,V]
    trt = retrieve_logits_triton(logits, idx)     # [L,D,V]
    max_diff = (ref.float() - trt.float()).abs().max().item()
    ok = torch.allclose(ref.float(), trt.float(), atol=1e-3, rtol=1e-3)
    print(f"[sanity/logits] ok={ok}, max_diff={max_diff:.3e}")

    # Case 2: hidden-like
    B, N, H = 1, 60, 12288
    L, D = 37, 6
    hidden = torch.randn(B, N, H, device=device, dtype=torch.float16)
    idx = _rand_indices(L, D, N, pad_prob=0.15, seed=2, device=device)

    ref = _torch_gather_ref(hidden, idx)          # [B,L,D,H]
    trt = retrieve_hidden_triton(hidden, idx)     # [B,L,D,H]
    max_diff = (ref.float() - trt.float()).abs().max().item()
    ok = torch.allclose(ref.float(), trt.float(), atol=1e-3, rtol=1e-3)
    print(f"[sanity/hidden] ok={ok}, max_diff={max_diff:.3e}")

    if not ok:
        raise AssertionError("Sanity failed.")


def run_bench(case="logits", B=1, N=60, F=128256, L=42, D=5, dtype="fp16", iters=200, seed=0):
    device = "cuda"
    torch.manual_seed(seed)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

    if case == "logits":
        src = torch.randn(1, N, F, device=device, dtype=dt)  # [1,N,V]
        idx = _rand_indices(L, D, N, pad_prob=0.2, seed=1, device=device)
        ref_fn = lambda: _torch_gather_ref(src[0], idx)
        trt_fn = lambda: retrieve_logits_triton(src, idx)
        label = "logits"
    else:
        src = torch.randn(B, N, F, device=device, dtype=dt)  # [B,N,H]
        idx = _rand_indices(L, D, N, pad_prob=0.15, seed=2, device=device)
        ref_fn = lambda: _torch_gather_ref(src, idx)
        trt_fn = lambda: retrieve_hidden_triton(src, idx)
        label = "hidden"

    # correctness once
    ref = ref_fn()
    trt = trt_fn()
    assert torch.allclose(ref.float(), trt.float(), atol=1e-3, rtol=1e-3)

    # warmup
    for _ in range(10):
        ref_fn(); trt_fn()
    torch.cuda.synchronize()

    # ref timing
    t0 = time.perf_counter()
    for _ in range(iters):
        ref_fn()
    torch.cuda.synchronize(); t1 = time.perf_counter()

    # triton timing
    t2 = time.perf_counter()
    for _ in range(iters):
        trt_fn()
    torch.cuda.synchronize(); t3 = time.perf_counter()

    ref_ms = (t1 - t0) * 1e3 / iters
    trt_ms = (t3 - t2) * 1e3 / iters

    print(f"[bench/{label}] B={B} N={N} F={F} L={L} D={D} dtype={dtype}")
    print(f"[bench/{label}] PyTorch : {ref_ms:7.3f} ms/iter")
    print(f"[bench/{label}] Triton  : {trt_ms:7.3f} ms/iter  (speedup x{ref_ms / trt_ms: .2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "bench"], default="sanity")
    ap.add_argument("--case", choices=["logits", "hidden"], default="logits")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--N", type=int, default=60)
    ap.add_argument("--F", type=int, default=128256)  # V or H
    ap.add_argument("--L", type=int, default=42)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "sanity":
        run_sanity()
    else:
        run_bench(case=args.case, B=args.B, N=args.N, F=args.F,
                  L=args.L, D=args.D, dtype=args.dtype,
                  iters=args.iters, seed=args.seed)


if __name__ == "__main__":
    main()
