#!/usr/bin/env python3
import argparse
import math
import os
import struct
import subprocess
import tempfile
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


MAGIC = 0x53415454  # "SATT"


@contextmanager
def sdpa_backend_ctx(backend: str):
    if backend == "flash":
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            yield
    elif backend == "math":
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            yield
    elif backend == "efficient":
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            yield
    else:
        # auto: let PyTorch pick best available backend
        yield


def build_cli(bundle_dir: str) -> str:
    exe_path = os.path.join(bundle_dir, "fused_online_attention_pwl_cli")
    src_path = os.path.join(bundle_dir, "fused_online_attention_pwl_cli.cpp")
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-I.",
        src_path,
        "-o",
        exe_path,
    ]
    subprocess.run(cmd, cwd=bundle_dir, check=True)
    return exe_path


def write_input(path: str, head_dim: int, seq_len: int, padded_len: int,
                q: np.ndarray, k: np.ndarray, v: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Iiii", MAGIC, head_dim, seq_len, padded_len))
        f.write(q.astype(np.float32, copy=False).tobytes(order="C"))
        f.write(k.astype(np.float32, copy=False).reshape(-1).tobytes(order="C"))
        f.write(v.astype(np.float32, copy=False).reshape(-1).tobytes(order="C"))


def read_output(path: str, head_dim: int) -> np.ndarray:
    out = np.fromfile(path, dtype=np.float32)
    if out.size != head_dim:
        raise RuntimeError(f"Unexpected output length: got {out.size}, expected {head_dim}")
    return out


def next_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def run_case(cli_exe: str,
             head_dim: int,
             seq_len: int,
             padded_len: int,
             backend: str,
             device: str,
             dtype: torch.dtype,
             seed: int) -> tuple[float, float]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    q = torch.randn((head_dim,), generator=g, device=device, dtype=dtype)
    k = torch.randn((seq_len, head_dim), generator=g, device=device, dtype=dtype)
    v = torch.randn((seq_len, head_dim), generator=g, device=device, dtype=dtype)

    q4 = q.view(1, 1, 1, head_dim)
    k4 = k.view(1, 1, seq_len, head_dim)
    v4 = v.view(1, 1, seq_len, head_dim)

    with sdpa_backend_ctx(backend):
        o_torch = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=None, dropout_p=0.0, is_causal=False
        )
    o_torch = o_torch.view(head_dim).float().cpu().numpy()

    q_np = q.float().cpu().numpy()
    k_np = k.float().cpu().numpy()
    v_np = v.float().cpu().numpy()

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.bin")
        out_path = os.path.join(td, "out.bin")
        write_input(in_path, head_dim, seq_len, padded_len, q_np, k_np, v_np)
        subprocess.run([cli_exe, in_path, out_path], check=True)
        o_hls = read_output(out_path, head_dim)

    abs_err = np.abs(o_hls - o_torch)
    return float(abs_err.max()), float(abs_err.mean())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare fused_online_attention_pwl HLS kernel output vs PyTorch SDPA (Flash/Math)."
    )
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--min-seq", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--pad-multiple", type=int, default=8)
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "flash", "math", "efficient"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-abs-thresh", type=float, default=0.08)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    cli_exe = build_cli(bundle_dir)

    # Probe backend once so we can emit a clear message early.
    try:
        _ = run_case(
            cli_exe=cli_exe,
            head_dim=args.head_dim,
            seq_len=max(args.min_seq, 8),
            padded_len=next_multiple(max(args.min_seq, 8), args.pad_multiple),
            backend=args.backend,
            device=args.device,
            dtype=dtype,
            seed=args.seed,
        )
    except RuntimeError as e:
        msg = str(e)
        if args.backend == "flash" and "No available kernel" in msg:
            print("Flash backend unavailable on this device/runtime.")
            print("Typical cause: Flash in PyTorch requires SM80+; current GPU may be older.")
        raise

    rng = np.random.default_rng(args.seed)
    max_abs = 0.0
    mean_abs_acc = 0.0

    for i in range(args.cases):
        seq_len = int(rng.integers(args.min_seq, args.max_seq + 1))
        padded_len = next_multiple(seq_len, args.pad_multiple)
        case_seed = args.seed + i * 17 + seq_len

        case_max, case_mean = run_case(
            cli_exe=cli_exe,
            head_dim=args.head_dim,
            seq_len=seq_len,
            padded_len=padded_len,
            backend=args.backend,
            device=args.device,
            dtype=dtype,
            seed=case_seed,
        )
        max_abs = max(max_abs, case_max)
        mean_abs_acc += case_mean

        print(
            f"[case {i+1:02d}/{args.cases}] seq={seq_len:3d} pad={padded_len:3d} "
            f"max_abs={case_max:.6f} mean_abs={case_mean:.6f}"
        )

    mean_abs = mean_abs_acc / max(args.cases, 1)
    print("\nSummary")
    print(f"  backend   : {args.backend}")
    print(f"  device    : {args.device}")
    print(f"  dtype     : {args.dtype}")
    print(f"  head_dim  : {args.head_dim}")
    print(f"  cases     : {args.cases}")
    print(f"  max_abs   : {max_abs:.6f}")
    print(f"  mean_abs  : {mean_abs:.6f}")
    print(f"  threshold : {args.max_abs_thresh:.6f}")

    if math.isnan(max_abs) or max_abs > args.max_abs_thresh:
        print("[FAIL] Error exceeds threshold.")
        return 1
    print("[PASS] HLS kernel matches PyTorch SDPA within threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

