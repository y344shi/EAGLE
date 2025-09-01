# kv_scatter_triton.py
# Triton KV-cache scatter/copy + tests & benchmark (fixed boolean guard)
# Usage:
#   python kv_scatter_triton.py --mode sanity
#   python kv_scatter_triton.py --mode bench --H 32 --T 4096 --D 128 --L 64

import argparse
import time
from typing import Sequence
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _kv_scatter_copy_kernel_4d(
    src_ptr, dst_ptr,          # [Layers, H, T, D]
    sel_ptr,                   # [L] int32 timesteps to copy
    prev_len,                  # scalar int: write begins at prev_len
    Layers, H, T, D,
    stride_sl, stride_sh, stride_st, stride_sd,
    stride_dl, stride_dh, stride_dt, stride_dd,
    stride_sel,
    BLOCK_D: tl.constexpr,
):
    # grid = (Layers * H, L)
    pid = tl.program_id(0)
    l = tl.program_id(1)

    layer = pid // H
    h = pid % H

    # scalar loads
    t_src = tl.load(sel_ptr + l * stride_sel, mask=True, other=0).to(tl.int32)
    t_dst = prev_len + l

    ok_src = (t_src >= 0) & (t_src < T)
    ok_dst = (t_dst >= 0) & (t_dst < T)
    in_bounds = ok_src & ok_dst

    if in_bounds:
        offs = tl.arange(0, BLOCK_D)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + offs
            m = d < D
            src_ix = layer * stride_sl + h * stride_sh + t_src * stride_st + d * stride_sd
            dst_ix = layer * stride_dl + h * stride_dh + t_dst * stride_dt + d * stride_dd
            val = tl.load(src_ptr + src_ix, mask=m, other=0)
            tl.store(dst_ptr + dst_ix, val, mask=m)

def kv_scatter_copy(
    kv: torch.Tensor,
    select_indices: torch.Tensor,
    prev_len: int,
    block_d: int = 128,
) -> None:
    """
    In-place copy for a KV tensor of shape [Layers, H, T, D] or [H, T, D].
    Copies rows kv[..., select_indices, :] -> kv[..., prev_len:prev_len+L, :].
    """
    assert kv.is_cuda, "KV must be CUDA"
    assert select_indices.is_cuda, "indices must be CUDA"
    assert kv.ndim in (3, 4), "KV must be [H,T,D] or [Layers,H,T,D]"

    if kv.ndim == 3:
        kv = kv.unsqueeze(0) # Treat as [1, H, T, D]

    Layers, H, T, D = kv.shape
    sl, sh, st, sd = kv.stride()
    dl, dh, dt, dd = kv.stride()

    sel = select_indices
    if sel.dtype != torch.int32:
        sel = sel.to(torch.int32)
    stride_sel = sel.stride(0)
    L = sel.shape[0]

    grid = (Layers * H, L)
    _kv_scatter_copy_kernel_4d[grid](
        kv, kv, sel,
        prev_len,
        Layers, H, T, D,
        sl, sh, st, sd,
        dl, dh, dt, dd,
        stride_sel,
        BLOCK_D=block_d,
    )


def kv_update_list_triton(
    past_key_values_data_list: Sequence[torch.Tensor],
    select_indices: torch.Tensor,
    prev_len: int,
    block_d: int = 128,
) -> None:
    """
    Drop-in replacement for the slow Python copy loop in update_inference_inputs().
    Works for each KV tensor in the list ([H,T,D] or [1,H,T,D], CUDA).
    """
    for kv in past_key_values_data_list:
        assert kv.is_cuda, "All KV tensors must be CUDA"
        kv_scatter_copy(kv, select_indices, prev_len, block_d=block_d)


# ---- sanity & bench ----

def _sanity_once(H=8, T=2048, D=128, L=17, dtype=torch.float16, layout4d=False, seed=0):
    torch.manual_seed(seed)
    kv = torch.randn((H, T, D), device="cuda", dtype=dtype)
    if layout4d:
        kv = kv.unsqueeze(0)  # [1,H,T,D]

    prev_len = max(L + 8, 32)
    assert prev_len < T
    sel = torch.randint(0, prev_len, (L,), device="cuda", dtype=torch.int32)

    ref = kv.clone()
    if layout4d:
        ref[0, :, prev_len:prev_len + L, :] = ref[0, :, sel.long(), :]
    else:
        ref[:, prev_len:prev_len + L, :] = ref[:, sel.long(), :]

    out = kv.clone()
    kv_scatter_copy(out, sel, prev_len, block_d=128)

    ok = torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)
    tag = "[4D]" if layout4d else "[3D]"
    print(f"[sanity]{tag} H={H} T={T} D={D} L={L} -> {ok}")
    if not ok:
        diff = (out - ref).abs().max().item()
        raise AssertionError(f"sanity failed (max diff {diff})")


def run_sanity():
    for layout4d in (False, True):
        _sanity_once(H=8,  T=1024, D=128, L=17,  dtype=torch.float16, layout4d=layout4d, seed=1)
        _sanity_once(H=32, T=2048, D=64,  L=33,  dtype=torch.float16, layout4d=layout4d, seed=2)
        _sanity_once(H=8,  T=4096, D=256, L=7,   dtype=torch.float32, layout4d=layout4d, seed=3)
    print("[sanity] all good ✅")


def run_bench(H=32, T=4096, D=128, L=64, dtype="float16", iters=200, seed=0):
    torch.manual_seed(seed)
    dt = torch.float16 if dtype == "float16" else torch.float32
    kv = torch.randn((H, T, D), device="cuda", dtype=dt)
    prev_len = max(64, L + 8)
    sel = torch.randint(0, prev_len, (L,), device="cuda", dtype=torch.int32)

    # warmup
    for _ in range(10):
        kv_scatter_copy(kv, sel, prev_len, block_d=128)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        kv_scatter_copy(kv, sel, prev_len, block_d=128)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    kv2 = kv.clone()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(iters):
        kv2[:, prev_len:prev_len + L, :] = kv2[:, sel.long(), :]
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    triton_ms = (t1 - t0) * 1e3 / iters
    torch_ms  = (t3 - t2) * 1e3 / iters
    print(f"[bench] Triton KV copy: {triton_ms:7.3f} ms/iter")
    print(f"[bench] Torch  KV copy: {torch_ms:7.3f} ms/iter")
    print(f"[bench] speedup x{torch_ms / triton_ms: .2f}")


def main():
    ap = argparse.ArgumentParser("KV cache scatter/copy (Triton)")
    ap.add_argument("--mode", choices=["sanity", "bench"], default="sanity")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if not _HAS_TRITON:
        raise SystemExit("Triton not found. `pip install triton`")

    if args.mode == "sanity":
        run_sanity()
    else:
        run_bench(H=args.H, T=args.T, D=args.D, L=args.L, dtype=args.dtype, iters=args.iters, seed=args.seed)


if __name__ == "__main__":
    main()
