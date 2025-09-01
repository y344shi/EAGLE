# evaluate_posterior_triton_fused.py
# Fused greedy evaluate_posterior for EAGLE:
#   * One Triton kernel per LEAF computes the greedy argmax over V across all steps
#     and accumulates the prefix accept length without materializing an [L,D] winners tensor.
#   * Host then argmaxes across leaves and returns softmax at [best_leaf, accept_len].
'''How to run
Sanity:
python evaluate_posterior_triton_fused.py --mode sanity --L 42 --D 5 --V 128256 --dtype fp16

Benchmark:
python evaluate_posterior_triton_fused.py --mode bench  --L 42 --D 5 --V 128256 --dtype fp16 --iters 500
'''
import argparse
import time
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# -------------------------------
# Triton kernel: fused greedy pass
# -------------------------------
# Inputs:
#   logits:    [L, D, V]  (fp16/bf16/fp32)
#   cands:     [L, D]     (int64; -1 treated as immediate mismatch)
# Output:
#   acc_len:   [L]        (int32 / int64 on host)
#
# For each leaf 'l', we loop steps s=0..D-2:
#   - argmax over vocab of logits[l, s, :]
#   - compare to cands[l, s+1]
#   - accumulate accept length until first mismatch
# We avoid early 'break' for portable performance: keep an 'alive' flag and
# accumulate conditionally. D is small, so wasted work is tiny.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_V': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 4096}, num_warps=8, num_stages=2),
    ],
    key=['V']
)
@triton.jit
def _eval_post_greedy_kernel(
    logits_ptr,          # *float* [L, D, V]
    cands_ptr,           # *int64* [L, D]
    acc_len_ptr,         # *int32* [L]
    L, D, V,
    # strides (elements)
    stride_l, stride_d, stride_v,
    stride_cl, stride_cd,
    BLOCK_V: tl.constexpr,
):
    l = tl.program_id(0)
    if l >= L:
        return

    acc_len = tl.zeros((), tl.int32)
    alive   = tl.full((), 1, tl.int32)   # 1 means we’re still matching prefix

    # loop steps s=0..D-2; compare to cands[:, s+1]
    if D > 1:
        for s in range(0, D - 1):
            # pointer to logits[l, s, :]
            base = logits_ptr + l * stride_l + s * stride_d

            # argmax over vocab
            best_val = tl.full((), -float("inf"), tl.float32)
            best_idx = tl.full((), 0, tl.int32)
            for start in range(0, V, BLOCK_V):
                offs = start + tl.arange(0, BLOCK_V)
                mask = offs < V
                vals = tl.load(base + offs * stride_v, mask=mask, other=-float("inf")).to(tl.float32)
                tile_max = tl.max(vals, axis=0)       # scalar
                tile_arg = tl.argmax(vals, axis=0)    # scalar in [0, BLOCK_V)
                take = tile_max > best_val
                best_val = tl.where(take, tile_max, best_val)
                best_idx = tl.where(take, (start + tile_arg).to(tl.int32), best_idx)

            # candidate token for next position (s+1)
            cand = tl.load(cands_ptr + l * stride_cl + (s + 1) * stride_cd).to(tl.int32)
            # valid if not -1
            valid = cand != (-1)
            # this step is a match iff valid & (winner == cand)
            matched = (best_idx == cand) & valid
            # accumulate if still alive (prefix)
            step_inc = tl.where((alive == 1) & matched, 1, 0)
            acc_len += step_inc
            # stay alive only if still matching
            alive = tl.where((alive == 1) & matched, 1, 0)

    tl.store(acc_len_ptr + l, acc_len)


# -------------------------------
# Public API
# -------------------------------

@torch.no_grad()
def evaluate_posterior_triton(
    logits: torch.Tensor,       # [L, D, V]
    candidates: torch.Tensor,   # [L, D]
    logits_processor=None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Greedy evaluate_posterior on GPU (fused).
    Returns:
      best_candidate (scalar int64),
      accept_length (scalar int64),
      sample_p (1D probs over V at [best, accept_length]) or None if logits_processor is given
    """
    # Keep host sampling path for temperature/top-p processors to match HF behavior exactly.
    if logits_processor is not None:
        L, D, V = logits.shape
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        adjustflag = False
        for i in range(1, D):
            if i != accept_length:
                break
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            seen = set()
            for j in range(L):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = int(x.item())
                    if xi in seen or xi == -1:
                        continue
                    seen.add(xi)
                    r = torch.rand((), device=logits.device)
                    px = gtp[xi]
                    if r <= px:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        best_candidate = j
                        accept_length += 1
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != D:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        return (torch.tensor(best_candidate, device=logits.device),
                torch.tensor(accept_length - 1, device=logits.device),
                sample_p)

    assert logits.ndim == 3 and candidates.ndim == 2, "logits [L,D,V], candidates [L,D]"
    L, D, V = logits.shape
    assert candidates.shape == (L, D)
    assert logits.is_cuda and candidates.is_cuda

    # 1) fused kernel computes per-leaf accept length
    acc_len_i32 = torch.empty((L,), device=logits.device, dtype=torch.int32)
    stride_l, stride_d, stride_v = logits.stride()
    stride_cl, stride_cd = candidates.stride()

    grid = (L,)
    _eval_post_greedy_kernel[grid](
        logits, candidates, acc_len_i32,
        L, D, V,
        stride_l, stride_d, stride_v,
        stride_cl, stride_cd,
    )

    acc_len = acc_len_i32.to(torch.int64)

    # 2) pick best leaf
    best_idx = torch.argmax(acc_len)
    best_len = acc_len[best_idx]

    # 3) return next-step prob distribution at [best_idx, best_len]
    # (compute softmax in fp32 for stability)
    sample_p = torch.softmax(logits[best_idx, best_len].to(torch.float32), dim=-1)
    return best_idx, best_len, sample_p


# -------------------------------
# Reference & tests
# -------------------------------

@torch.no_grad()
def _evaluate_posterior_ref(logits: torch.Tensor, candidates: torch.Tensor):
    """Pure PyTorch reference (greedy)."""
    L, D, V = logits.shape
    if D > 1:
        winners = torch.argmax(logits[:, :-1, :], dim=-1)  # [L, D-1]
        eq = (winners == candidates[:, 1:])
        acc = torch.cumprod(eq.to(torch.int64), dim=1).sum(dim=1)  # [L]
    else:
        acc = torch.zeros((L,), device=logits.device, dtype=torch.int64)
    best = torch.argmax(acc)
    best_len = acc[best]
    sample_p = torch.softmax(logits[best, best_len], dim=-1)
    return best, best_len, sample_p


def _rand_logits_cands(L, D, V, dtype, device, pad_prob=0.1, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(L, D, V, device=device, dtype=dtype)
    cands  = torch.randint(low=0, high=V, size=(L, D), device=device, dtype=torch.int64)
    if pad_prob > 0 and D > 1:
        pad_mask = (torch.rand(L, D, device=device) < pad_prob)
        cands[pad_mask] = -1
    return logits, cands


def run_sanity(L=42, D=5, V=128256, dtype=torch.float16, device="cuda", seed=0):
    logits, cands = _rand_logits_cands(L, D, V, dtype, device, pad_prob=0.1, seed=seed)
    b_ref, len_ref, p_ref = _evaluate_posterior_ref(logits, cands)
    b_trt, len_trt, p_trt = evaluate_posterior_triton(logits, cands)

    ok_idx = int(b_ref.item()) == int(b_trt.item())
    ok_len = int(len_ref.item()) == int(len_trt.item())
    diff = (p_ref.float() - p_trt.float()).abs().max().item()
    ok_p = torch.allclose(p_ref.float(), p_trt.float(), atol=1e-3, rtol=1e-3)

    print(f"[sanity] idx_ok={ok_idx}, len_ok={ok_len}, max|Δp|={diff:.3e}, p_ok={ok_p}")
    if not (ok_idx and ok_len and ok_p):
        raise AssertionError("Sanity failed.")


def run_bench(L=42, D=5, V=128256, dtype="fp16", iters=500, seed=0, device="cuda"):
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    logits, cands = _rand_logits_cands(L, D, V, dt, device, pad_prob=0.1, seed=seed)

    # verify once
    b_ref, len_ref, p_ref = _evaluate_posterior_ref(logits, cands)
    b_trt, len_trt, p_trt = evaluate_posterior_triton(logits, cands)
    assert int(b_ref) == int(b_trt) and int(len_ref) == int(len_trt)
    assert torch.allclose(p_ref.float(), p_trt.float(), atol=1e-3, rtol=1e-3)
    # warmup
    for _ in range(10):
        _evaluate_posterior_ref(logits, cands)
        evaluate_posterior_triton(logits, cands)
    torch.cuda.synchronize()

    # reference timing
    t0 = time.perf_counter()
    for _ in range(iters):
        _evaluate_posterior_ref(logits, cands)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # triton timing
    t2 = time.perf_counter()
    for _ in range(iters):
        evaluate_posterior_triton(logits, cands)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    ref_ms = (t1 - t0) * 1e3 / iters
    trt_ms = (t3 - t2) * 1e3 / iters
    print(f"[bench] shapes L={L}, D={D}, V={V}, dtype={dtype}")
    print(f"[bench] PyTorch ref : {ref_ms:7.3f} ms/iter")
    print(f"[bench] Triton impl : {trt_ms:7.3f} ms/iter  (speedup x{ref_ms / trt_ms: .2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "bench"], default="sanity")
    ap.add_argument("--L", type=int, default=42)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--V", type=int, default=128256)
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    if args.mode == "sanity":
        run_sanity(L=args.L, D=args.D, V=args.V,
                   dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype],
                   device=device, seed=args.seed)
    else:
        run_bench(L=args.L, D=args.D, V=args.V, dtype=args.dtype, iters=args.iters, seed=args.seed, device=device)


if __name__ == "__main__":
    main()
