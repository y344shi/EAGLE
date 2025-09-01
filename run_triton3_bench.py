#!/usr/bin/env python3
"""
run_triton_bench.py
Small wrapper that adds project root to PYTHONPATH and launches the benchmark.  
python run_triton3_bench.py \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --ea_model_path   yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --prompt "Once upon a time" \
    --max_new_tokens 256 \
    --compare \
    --runs 3 --warmup 1 --print_output

"""

import sys
from pathlib import Path
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser("Run EAGLE Triton launcher with sane defaults")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ea_model_path",   type=str, default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compare", action="store_true", default=True)
    parser.add_argument("--triton_only", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["float16","fp16","bf16","bfloat16","float32","fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--print_output", action="store_true")
    parser.add_argument("--no-trace", action="store_true", help="Disable verbose tracing")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler for one run and print a summary table.")
    args = parser.parse_args()

    # Add project root to sys.path so imports work when running from anywhere
    project_root = Path(__file__).resolve().parent
    os.environ.setdefault("PYTHONPATH", str(project_root))

    bench = project_root / "eagle/model/triton_kernels_3/eagle_triton_launch.py"
    cmd = [
        sys.executable, str(bench),
        "--base_model_path", args.base_model_path,
        "--ea_model_path",   args.ea_model_path,
        "--prompt", args.prompt,
        "--max_new_tokens", str(args.max_new_tokens),
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--torch_dtype", args.torch_dtype,
        "--device_map", args.device_map,
    ]
    if args.compare:      cmd.append("--compare")
    if args.triton_only:  cmd.append("--triton_only")
    if args.print_output: cmd.append("--print_output")
    if args.no_trace:     cmd.append("--no-trace")
    if args.profile:      cmd.append("--profile")

    print("Running:", " ".join(cmd))
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()