import torch
import time
import argparse
import numpy as np
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from transformers import AutoTokenizer
    from eagle.model.ea_model import EaModel
    from integration import optimize_eagle_with_triton
except ImportError:
    print("Warning: Could not import required modules. Make sure transformers is installed and eagle module is in the Python path.")

def benchmark_generation(model, tokenizer, prompt, max_length, num_runs=5):
    """
    Benchmark text generation with the given model.
    
    Parameters:
        model: EAGLE model
        tokenizer: tokenizer for the model
        prompt: input prompt for generation
        max_length: maximum length of generated text
        num_runs: number of runs for averaging
        
    Returns:
        avg_time: average time per run in seconds
        tokens_per_second: average tokens generated per second
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    # Warm-up run
    with torch.no_grad():
        _ = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            do_sample=False,
            use_cache=True
        )
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                do_sample=False,
                use_cache=True
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate metrics
    avg_time = np.mean(times)
    tokens_generated = outputs.shape[1] - input_length
    tokens_per_second = tokens_generated / avg_time
    
    return avg_time, tokens_per_second

def benchmark_attention(model, batch_size=1, seq_len=512, head_dim=64, num_heads=32, num_runs=100):
    """
    Benchmark attention computation.
    
    Parameters:
        model: EAGLE model
        batch_size: batch size for the benchmark
        seq_len: sequence length for the benchmark
        head_dim: dimension of each attention head
        num_heads: number of attention heads
        num_runs: number of runs for averaging
        
    Returns:
        avg_time_original: average time per run for original implementation
        avg_time_triton: average time per run for Triton implementation
    """
    # Create random inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    
    # Get attention implementation
    attention_layer = model.base_model.model.layers[0].self_attn
    
    # Benchmark original implementation
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = attention_layer.forward(q, k, v)
    
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time_original = (end_time - start_time) / num_runs
    
    # Optimize model with Triton kernels
    model, original_methods = optimize_eagle_with_triton(model)
    
    # Benchmark Triton implementation
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = attention_layer.forward(q, k, v)
    
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time_triton = (end_time - start_time) / num_runs
    
    # Restore original methods
    for layer in model.base_model.model.layers:
        layer.self_attn.forward = original_methods['attention']
    
    return avg_time_original, avg_time_triton

def benchmark_kv_cache(model, batch_size=1, seq_len=512, head_dim=64, num_heads=32, num_runs=100):
    """
    Benchmark KV cache operations.
    
    Parameters:
        model: EAGLE model
        batch_size: batch size for the benchmark
        seq_len: sequence length for the benchmark
        head_dim: dimension of each attention head
        num_heads: number of attention heads
        num_runs: number of runs for averaging
        
    Returns:
        avg_time_original: average time per run for original implementation
        avg_time_triton: average time per run for Triton implementation
    """
    # Create random inputs
    k_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    v_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    k_new = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    v_new = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda")
    current_length = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    
    # Define original implementation (simplified)
    def original_append_to_kv_cache(k_cache, v_cache, k_new, v_new, current_length):
        batch_size = k_cache.shape[0]
        for b in range(batch_size):
            curr_len = current_length[b].item()
            k_cache[b, :, curr_len:curr_len+1] = k_new[b]
            v_cache[b, :, curr_len:curr_len+1] = v_new[b]
            current_length[b] += 1
    
    # Benchmark original implementation
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            original_append_to_kv_cache(k_cache.clone(), v_cache.clone(), k_new, v_new, current_length.clone())
    
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time_original = (end_time - start_time) / num_runs
    
    # Import Triton implementation
    from eagle.model.triton_kernels import triton_append_to_kv_cache
    
    # Benchmark Triton implementation
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            triton_append_to_kv_cache(k_cache.clone(), v_cache.clone(), k_new, v_new, current_length.clone())
    
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time_triton = (end_time - start_time) / num_runs
    
    return avg_time_original, avg_time_triton

def main():
    parser = argparse.ArgumentParser(description="Benchmark EAGLE model with Triton kernels")
    parser.add_argument("--model", type=str, default="EAGLE-3B", help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Triton is a language and compiler for writing highly efficient custom CUDA kernels.", help="Input prompt for generation")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs for averaging")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for attention and KV cache benchmarks")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for attention and KV cache benchmarks")
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EaModel.from_pretrained(args.model)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA for benchmarking")
    else:
        print("CUDA not available, using CPU (benchmarks will be slow)")
    
    # Benchmark text generation
    print("\n=== Text Generation Benchmark ===")
    print(f"Prompt: {args.prompt}")
    print(f"Max length: {args.max_length}")
    print(f"Number of runs: {args.num_runs}")
    
    # Original model
    print("\nBenchmarking original model...")
    avg_time_original, tokens_per_second_original = benchmark_generation(
        model, tokenizer, args.prompt, args.max_length, args.num_runs
    )
    
    # Optimize model with Triton kernels
    print("\nOptimizing model with Triton kernels...")
    model, original_methods = optimize_eagle_with_triton(model)
    
    # Optimized model
    print("\nBenchmarking optimized model...")
    avg_time_triton, tokens_per_second_triton = benchmark_generation(
        model, tokenizer, args.prompt, args.max_length, args.num_runs
    )
    
    # Print results
    print("\n=== Text Generation Results ===")
    print(f"Original model: {avg_time_original:.4f} seconds, {tokens_per_second_original:.2f} tokens/second")
    print(f"Triton-optimized model: {avg_time_triton:.4f} seconds, {tokens_per_second_triton:.2f} tokens/second")
    print(f"Speedup: {avg_time_original / avg_time_triton:.2f}x")
    
    # Benchmark attention computation
    print("\n=== Attention Computation Benchmark ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    
    avg_time_original, avg_time_triton = benchmark_attention(
        model, args.batch_size, args.seq_len, 64, 32, 100
    )
    
    print("\n=== Attention Computation Results ===")
    print(f"Original implementation: {avg_time_original * 1000:.4f} ms")
    print(f"Triton implementation: {avg_time_triton * 1000:.4f} ms")
    print(f"Speedup: {avg_time_original / avg_time_triton:.2f}x")
    
    # Benchmark KV cache operations
    print("\n=== KV Cache Operations Benchmark ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    
    avg_time_original, avg_time_triton = benchmark_kv_cache(
        model, args.batch_size, args.seq_len, 64, 32, 100
    )
    
    print("\n=== KV Cache Operations Results ===")
    print(f"Original implementation: {avg_time_original * 1000:.4f} ms")
    print(f"Triton implementation: {avg_time_triton * 1000:.4f} ms")
    print(f"Speedup: {avg_time_original / avg_time_triton:.2f}x")


if __name__ == "__main__":
    # Check if Triton is available
    try:
        import triton
        main()
    except ImportError:
        print("Triton is not installed. Please install it with:")
        print("pip install triton")
        print("For more information, visit: https://github.com/openai/triton")