"""
Benchmark script for EAGLE Triton kernels.
"""

import torch
import time
import argparse
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels.ea_model_patch import patch_eagle_model, unpatch_eagle_model


def benchmark_generation(model, tokenizer, prompt, max_new_tokens=100, num_runs=3, use_triton=True):
    """
    Benchmark text generation with the given model.
    
    Args:
        model: The EAGLE model
        tokenizer: The tokenizer
        prompt: Text prompt to generate from
        max_new_tokens: Maximum number of new tokens to generate
        num_runs: Number of runs to average over
        use_triton: Whether to use Triton kernels
        
    Returns:
        tuple: (average_time, tokens_per_second)
    """
    # Apply Triton patches if requested
    if use_triton:
        model = patch_eagle_model(model)
    
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base_model.device)
    
    # Warm-up run
    with torch.no_grad():
        _ = model.eagenerate(input_ids, max_new_tokens=10)
    
    # Benchmark runs
    times = []
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.eagenerate(input_ids, max_new_tokens=max_new_tokens)
        end_time = time.time()
        times.append(end_time - start_time)
        
        # Print progress
        print(f"Run {i+1}/{num_runs}: {times[-1]:.2f} seconds")
    
    # Calculate metrics
    avg_time = np.mean(times)
    tokens_generated = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_second = tokens_generated / avg_time
    
    # Remove patches if applied
    if use_triton:
        unpatch_eagle_model(model)
    
    return avg_time, tokens_per_second


def run_benchmarks(model_path, prompt, max_new_tokens=100, num_runs=3):
    """
    Run benchmarks comparing PyTorch and Triton implementations.
    
    Args:
        model_path: Path to the model
        prompt: Text prompt to generate from
        max_new_tokens: Maximum number of new tokens to generate
        num_runs: Number of runs to average over
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = EaModel.from_pretrained(
        base_model_path=model_path,
        ea_model_path=model_path,
        device_map="auto",
    )
    tokenizer = model.get_tokenizer()
    
    # Benchmark PyTorch implementation
    print("\n" + "=" * 80)
    print("Benchmarking PyTorch implementation...")
    print("=" * 80)
    pytorch_time, pytorch_tps = benchmark_generation(
        model, tokenizer, prompt, max_new_tokens, num_runs, use_triton=False
    )
    
    # Benchmark Triton implementation
    print("\n" + "=" * 80)
    print("Benchmarking Triton implementation...")
    print("=" * 80)
    triton_time, triton_tps = benchmark_generation(
        model, tokenizer, prompt, max_new_tokens, num_runs, use_triton=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("Benchmark Results:")
    print(f"PyTorch: {pytorch_time:.2f} seconds, {pytorch_tps:.2f} tokens/second")
    print(f"Triton: {triton_time:.2f} seconds, {triton_tps:.2f} tokens/second")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    print("=" * 80)
    
    # Plot results
    labels = ['PyTorch', 'Triton']
    times = [pytorch_time, triton_time]
    tps = [pytorch_tps, triton_tps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot generation time
    ax1.bar(labels, times, color=['blue', 'orange'])
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Generation Time')
    
    # Plot tokens per second
    ax2.bar(labels, tps, color=['blue', 'orange'])
    ax2.set_ylabel('Tokens per second')
    ax2.set_title('Generation Speed')
    
    plt.tight_layout()
    plt.savefig('eagle_triton_benchmark.png')
    print("Benchmark plot saved to eagle_triton_benchmark.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark EAGLE Triton kernels")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs to average over")
    
    args = parser.parse_args()
    
    run_benchmarks(args.model_path, args.prompt, args.max_new_tokens, args.num_runs)