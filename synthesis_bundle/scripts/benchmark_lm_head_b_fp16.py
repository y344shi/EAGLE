#!/usr/bin/env python3
"""Section 1.2 benchmark: LM head B microbenchmark + stage timing proxy.

This is a GPU proxy benchmark (V100 here) to establish bottleneck evidence before FPGA optimization.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


def measure_kernel(fn, iterations: int, warmup: int = 30) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_us = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start_ev.record()
        fn()
        end_ev.record()
        end_ev.synchronize()
        times_us.append(start_ev.elapsed_time(end_ev) * 1000.0)

    return statistics.mean(times_us), statistics.pstdev(times_us)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/section1_lm_head_benchmark.csv"))
    parser.add_argument("--iter", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vocab", type=int, default=128256)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--topk", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda"
    dtype = torch.float16

    # Stage tensors (proxy for drafter decode path).
    emb = torch.randn(args.vocab, args.hidden, device=device, dtype=dtype)
    w_attn = torch.randn(args.hidden, args.hidden, device=device, dtype=dtype)
    w_up = torch.randn(args.hidden, 14336, device=device, dtype=dtype)
    w_down = torch.randn(14336, args.hidden, device=device, dtype=dtype)

    wa = torch.randn(args.hidden, args.rank, device=device, dtype=dtype)
    wb = torch.randn(args.vocab, args.rank, device=device, dtype=dtype)
    w_lm = torch.randn(args.vocab, args.hidden, device=device, dtype=dtype)

    token_id = torch.tensor([1234], device=device, dtype=torch.long)
    hidden = emb[token_id].clone()

    # 1) Mandatory B-only microbenchmark: 128 x 128k projection.
    a_vec = torch.randn(1, args.rank, device=device, dtype=dtype)

    def b_only():
        _ = torch.matmul(a_vec, wb.t())

    b_mean_us, b_std_us = measure_kernel(b_only, args.iter)
    b_read_bytes = wb.numel() * 2 + a_vec.numel() * 2
    b_eff_gbps = (b_read_bytes / (b_mean_us * 1e-6)) / 1e9

    # 2) Stage breakdown proxy (embedding/layer/LM-A/B/C/top-k).
    stage_data: dict[str, list[float]] = {
        "embedding_us": [],
        "transformer_layers_us": [],
        "lm_head_a_us": [],
        "lm_head_b_us": [],
        "lm_head_c_us": [],
        "topk_us": [],
        "total_us": [],
    }

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(30):
        # Warmup full path
        x = emb[token_id].clone()
        h = torch.matmul(x, w_attn)
        u = torch.matmul(h, w_up)
        d = torch.matmul(F.silu(u), w_down)
        x2 = h + d
        a = torch.matmul(x2, wa)
        b = torch.matmul(a, wb.t())
        cand = torch.topk(b, args.topk, dim=-1).indices
        sel = w_lm[cand[0]]
        c = torch.matmul(x2, sel.t())
        _ = torch.topk(c, 1, dim=-1)
    torch.cuda.synchronize()

    for _ in range(args.iter):
        # embedding
        start_ev.record()
        x = emb[token_id].clone()
        end_ev.record(); end_ev.synchronize()
        stage_data["embedding_us"].append(start_ev.elapsed_time(end_ev) * 1000.0)

        # transformer (proxy single-layer compute)
        start_ev.record()
        h = torch.matmul(x, w_attn)
        u = torch.matmul(h, w_up)
        d = torch.matmul(F.silu(u), w_down)
        x2 = h + d
        end_ev.record(); end_ev.synchronize()
        stage_data["transformer_layers_us"].append(start_ev.elapsed_time(end_ev) * 1000.0)

        # lm head A
        start_ev.record()
        a = torch.matmul(x2, wa)
        end_ev.record(); end_ev.synchronize()
        stage_data["lm_head_a_us"].append(start_ev.elapsed_time(end_ev) * 1000.0)

        # lm head B
        start_ev.record()
        b = torch.matmul(a, wb.t())
        end_ev.record(); end_ev.synchronize()
        stage_data["lm_head_b_us"].append(start_ev.elapsed_time(end_ev) * 1000.0)

        # lm head C (candidate gather + re-score)
        start_ev.record()
        cand = torch.topk(b, args.topk, dim=-1).indices
        sel = w_lm[cand[0]]
        c = torch.matmul(x2, sel.t())
        end_ev.record(); end_ev.synchronize()
        stage_data["lm_head_c_us"].append(start_ev.elapsed_time(end_ev) * 1000.0)

        # final top-k
        start_ev.record()
        _ = torch.topk(c, 1, dim=-1)
        end_ev.record(); end_ev.synchronize()
        stage_data["topk_us"].append(start_ev.elapsed_time(end_ev) * 1000.0)

    for i in range(args.iter):
        total = (
            stage_data["embedding_us"][i]
            + stage_data["transformer_layers_us"][i]
            + stage_data["lm_head_a_us"][i]
            + stage_data["lm_head_b_us"][i]
            + stage_data["lm_head_c_us"][i]
            + stage_data["topk_us"][i]
        )
        stage_data["total_us"].append(total)

    means = {k: statistics.mean(v) for k, v in stage_data.items()}
    b_percent_total = 100.0 * means["lm_head_b_us"] / means["total_us"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["gpu_name", torch.cuda.get_device_name(0)])
        w.writerow(["dtype", str(dtype)])
        w.writerow(["vocab", args.vocab])
        w.writerow(["hidden", args.hidden])
        w.writerow(["rank", args.rank])
        w.writerow(["topk", args.topk])
        w.writerow(["b_only_latency_mean_us", f"{b_mean_us:.3f}"])
        w.writerow(["b_only_latency_std_us", f"{b_std_us:.3f}"])
        w.writerow(["b_hbm_read_bytes_per_token", b_read_bytes])
        w.writerow(["b_effective_gbps", f"{b_eff_gbps:.3f}"])
        for key in [
            "embedding_us",
            "transformer_layers_us",
            "lm_head_a_us",
            "lm_head_b_us",
            "lm_head_c_us",
            "topk_us",
            "total_us",
        ]:
            w.writerow([f"stage_mean_{key}", f"{means[key]:.3f}"])
        w.writerow(["lm_head_b_percent_total", f"{b_percent_total:.2f}"])

    print(f"Wrote: {args.output}")
    print(f"B-only latency: {b_mean_us:.3f} us (std {b_std_us:.3f} us)")
    print(f"B read bytes/token: {b_read_bytes}")
    print(f"B effective bandwidth: {b_eff_gbps:.3f} GB/s")
    print("Stage breakdown (mean us):")
    print(f"  Embedding        {means['embedding_us']:.3f}")
    print(f"  Transformer      {means['transformer_layers_us']:.3f}")
    print(f"  LM head A        {means['lm_head_a_us']:.3f}")
    print(f"  LM head B        {means['lm_head_b_us']:.3f}")
    print(f"  LM head C        {means['lm_head_c_us']:.3f}")
    print(f"  Top-K            {means['topk_us']:.3f}")
    print(f"  Total            {means['total_us']:.3f}")
    print(f"LM head B % total: {b_percent_total:.2f}%")


if __name__ == "__main__":
    main()
