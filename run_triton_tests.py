#!/usr/bin/env python
"""
Script to run the Triton kernel tests with proper Python path setup.
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EAGLE Triton kernel tests")
    parser.add_argument("--base_model_path", type=str, default='meta-llama/Llama-3.1-8B-Instruct', help="Path to the base model")
    parser.add_argument("--ea_model_path", type=str, default='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B', help="Path to the EAGLE model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--compare", action="store_true", help="Compare PyTorch and Triton implementations")
    
    args = parser.parse_args()
    
    # Build command
    cmd = [sys.executable, "eagle/model/triton_kernels_2/test_triton_kernels.py"]
    cmd.extend(["--base_model_path", args.base_model_path])
    cmd.extend(["--ea_model_path", args.ea_model_path])
    cmd.extend(["--prompt", args.prompt])
    cmd.extend(["--max_new_tokens", str(args.max_new_tokens)])
    if args.compare:
        cmd.append("--compare")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)