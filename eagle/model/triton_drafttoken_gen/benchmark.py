"""
benchmark.py — Benchmark the Triton drafter against the original implementation.
"""

import time
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from eagle.model.configs import EConfig
from eagle.model.cnets import Model
from eagle.model.triton_drafttoken_gen.frontier_api import FrontierConfig, frontier_generate
from eagle.model.triton_drafttoken_gen.frontier_integration import register_triton_backend


def setup_model(model_size: str = "8B") -> Tuple[Model, Dict[str, int]]:
    """
    Set up an EA model for benchmarking.
    
    Args:
        model_size: Size of the model to use ("8B" or "70B")
        
    Returns:
        Tuple of (model, dims) where dims is a dictionary of model dimensions
    """
    # Create a local config instead of loading from HuggingFace
    config = EConfig()
    
    # Set model dimensions based on size
    if model_size == "8B":
        config.hidden_size = 4096
        config.intermediate_size = 11008
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.vocab_size = 32000
        config.draft_vocab_size = 32000
    elif model_size == "70B":
        config.hidden_size = 8192
        config.intermediate_size = 28672
        config.num_attention_heads = 64
        config.num_key_value_heads = 8
        config.vocab_size = 32000
        config.draft_vocab_size = 32000
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Create model
    model = Model(
        config, 
        load_emb=False, 
        total_tokens=60, 
        depth=5, 
        top_k=10, 
        threshold=1.0
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Extract dimensions
    dims = {
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "draft_vocab_size": config.draft_vocab_size,
        "num_attention_heads": config.num_attention_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
    }
    
    return model, dims


def benchmark_original(
    model: Model, 
    hidden_states: torch.Tensor, 
    input_ids: torch.Tensor, 
    n_runs: int = 10
) -> Tuple[float, List[float]]:
    """
    Benchmark the original EA implementation.
    
    Args:
        model: The EA model
        hidden_states: Input hidden states
        input_ids: Input token IDs
        n_runs: Number of runs to average over
        
    Returns:
        Tuple of (average_time, all_times)
    """
    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            model.topK_genrate(hidden_states, input_ids, model.lm_head, None)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            model.topK_genrate(hidden_states, input_ids, model.lm_head, None)
        
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # ms
    
    return sum(times) / len(times), times


def benchmark_frontier_api(
    model: Model, 
    hidden_states: torch.Tensor, 
    input_ids: torch.Tensor, 
    backend: str,
    n_runs: int = 10,
    use_fallback: bool = False,
) -> Tuple[float, List[float]]:
    """
    Benchmark the frontier API implementation.
    
    Args:
        model: The EA model
        hidden_states: Input hidden states
        input_ids: Input token IDs
        backend: Backend to use ("ea_layer" or "triton")
        n_runs: Number of runs to average over
        use_fallback: Whether to use the PyTorch fallback for the Triton backend
        
    Returns:
        Tuple of (average_time, all_times)
    """
    # Register the Triton backend
    register_triton_backend()
    
    # Create frontier config
    cfg = FrontierConfig(
        total_token=model.total_tokens,
        depth=model.depth,
        top_k=model.top_k,
        vocab_size=model.config.vocab_size,
        hidden_size=model.config.hidden_size,
        use_concat_taps=True,
        use_fc_align=True,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    
    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            frontier_generate(
                cfg,
                features_concat=hidden_states,
                backend=backend,
                ea_layer=model,
                input_ids=input_ids,
                use_fallback=use_fallback,
            )
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            frontier_generate(
                cfg,
                features_concat=hidden_states,
                backend=backend,
                ea_layer=model,
                input_ids=input_ids,
                use_fallback=use_fallback,
            )
        
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # ms
    
    return sum(times) / len(times), times


def run_benchmarks(
    model_size: str = "8B",
    n_runs: int = 10,
    plot: bool = True,
) -> Dict[str, float]:
    """
    Run benchmarks for different implementations.
    
    Args:
        model_size: Size of the model to use ("8B" or "70B")
        n_runs: Number of runs to average over
        plot: Whether to plot the results
        
    Returns:
        Dictionary of average times for each implementation
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return {}
    
    # Set up model
    model, dims = setup_model(model_size)
    
    # Create inputs
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden_states = torch.randn(1, 3 * dims["hidden_size"], device=device, dtype=dtype)
    input_ids = torch.randint(0, dims["vocab_size"], (1, 10), device=device)
    
    # Run benchmarks
    print(f"Running benchmarks for {model_size} model ({n_runs} runs each)...")
    
    print("Benchmarking original implementation...")
    orig_avg, orig_times = benchmark_original(model, hidden_states, input_ids, n_runs)
    print(f"  Average time: {orig_avg:.2f} ms")
    
    print("Benchmarking frontier API with ea_layer backend...")
    ea_avg, ea_times = benchmark_frontier_api(model, hidden_states, input_ids, "ea_layer", n_runs)
    print(f"  Average time: {ea_avg:.2f} ms")
    
    print("Benchmarking frontier API with triton backend (fallback)...")
    triton_fb_avg, triton_fb_times = benchmark_frontier_api(
        model, hidden_states, input_ids, "triton", n_runs, use_fallback=True
    )
    print(f"  Average time: {triton_fb_avg:.2f} ms")
    
    print("Benchmarking frontier API with triton backend...")
    triton_avg, triton_times = benchmark_frontier_api(
        model, hidden_states, input_ids, "triton", n_runs, use_fallback=False
    )
    print(f"  Average time: {triton_avg:.2f} ms")
    
    # Collect results
    results = {
        "Original": orig_avg,
        "Frontier API (ea_layer)": ea_avg,
        "Frontier API (triton fallback)": triton_fb_avg,
        "Frontier API (triton)": triton_avg,
    }
    
    # Plot results
    if plot:
        plt.figure(figsize=(10, 6))
        plt.boxplot([orig_times, ea_times, triton_fb_times, triton_times], 
                   labels=["Original", "Frontier API\n(ea_layer)", 
                           "Frontier API\n(triton fallback)", "Frontier API\n(triton)"])
        plt.ylabel("Time (ms)")
        plt.title(f"EAGLE Draft Token Generation Performance ({model_size} model)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(f"drafter_benchmark_{model_size}.png")
        print(f"Plot saved to drafter_benchmark_{model_size}.png")
    
    return results


def main():
    """
    Main function for the benchmark script.
    """
    parser = argparse.ArgumentParser(description="Benchmark the Triton drafter")
    parser.add_argument("--model-size", type=str, default="8B", choices=["8B", "70B"],
                        help="Size of the model to use")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs to average over")
    parser.add_argument("--no-plot", action="store_true",
                        help="Don't plot the results")
    args = parser.parse_args()
    
    run_benchmarks(args.model_size, args.runs, not args.no_plot)


if __name__ == "__main__":
    main()