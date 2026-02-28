#!/usr/bin/env python3
"""Measure host->device transfer characteristics as a proxy when QDMA is unavailable.

Outputs CSV with per-size latency and effective GB/s.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch


def has_qdma_device() -> bool:
    dev = Path("/dev")
    if not dev.exists():
        return False
    for p in dev.iterdir():
        name = p.name
        if "qdma" in name or "xocl" in name or "xclmgmt" in name:
            return True
    return False


def measure_copy(size_bytes: int, iterations: int, warmup: int) -> tuple[float, float, float]:
    numel = size_bytes
    host = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
    device = torch.empty(numel, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        device.copy_(host, non_blocking=True)
    torch.cuda.synchronize()

    lat_us = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start_ev.record()
        device.copy_(host, non_blocking=True)
        end_ev.record()
        end_ev.synchronize()
        ms = start_ev.elapsed_time(end_ev)
        lat_us.append(ms * 1000.0)

    mean_us = statistics.mean(lat_us)
    std_us = statistics.pstdev(lat_us)
    gbps = (size_bytes / (mean_us * 1e-6)) / 1e9
    return mean_us, std_us, gbps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/section1_qdma_hbm_proxy.csv"),
        help="CSV output path",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iter-large", type=int, default=120)
    parser.add_argument("--iter-small", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for proxy measurement.")

    torch.cuda.init()
    gpu_name = torch.cuda.get_device_name(0)

    sizes = [
        ("small_4KB", 4 * 1024, args.iter_small),
        ("small_64KB", 64 * 1024, args.iter_small),
        ("large_1MB", 1 * 1024 * 1024, args.iter_large),
        ("large_16MB", 16 * 1024 * 1024, args.iter_large),
    ]

    rows = []
    for label, size, iters in sizes:
        mean_us, std_us, gbps = measure_copy(size, iters, args.warmup)
        rows.append(
            {
                "label": label,
                "size_bytes": size,
                "iterations": iters,
                "latency_mean_us": f"{mean_us:.3f}",
                "latency_std_us": f"{std_us:.3f}",
                "effective_gbps": f"{gbps:.3f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "size_bytes",
                "iterations",
                "latency_mean_us",
                "latency_std_us",
                "effective_gbps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    qdma_present = has_qdma_device()
    print(f"GPU: {gpu_name}")
    print(f"QDMA device visible: {qdma_present}")
    print(f"Wrote: {args.output}")
    for r in rows:
        print(
            f"{r['label']}: mean={r['latency_mean_us']} us, "
            f"std={r['latency_std_us']} us, bw={r['effective_gbps']} GB/s"
        )


if __name__ == "__main__":
    main()
