import os
import sys
import time
import torch
import argparse
import unittest

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from test_kernels import TestTritonKernels

def run_all_tests(verbose=True, detailed_report=False):
    """
    Run all tests and generate a report.
    
    Parameters:
        verbose: Whether to print detailed test output
        detailed_report: Whether to generate a detailed performance report
    
    Returns:
        success: Whether all tests passed
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTritonKernels)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Generate report
    print("\n" + "=" * 80)
    print("TRITON KERNELS TEST REPORT")
    print("=" * 80)
    
    # Summary
    print(f"\nTotal tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Failures and errors
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
            if detailed_report:
                print(traceback)
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
            if detailed_report:
                print(traceback)
    
    # Performance report
    if detailed_report:
        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)
        
        # Create a new instance for performance testing
        test_instance = TestTritonKernels()
        test_instance.setUp()
        
        # Test attention with different batch sizes and sequence lengths
        print("\nAttention Performance:")
        batch_sizes = [1, 2, 4, 8]
        seq_lengths = [16, 32, 64, 128, 256, 512]
        
        print(f"{'Batch Size':<10} {'Seq Length':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # Create random inputs
                q = torch.randn(batch_size, test_instance.num_heads, seq_len, test_instance.head_dim, device=test_instance.device)
                k = torch.randn(batch_size, test_instance.num_heads, seq_len, test_instance.head_dim, device=test_instance.device)
                v = torch.randn(batch_size, test_instance.num_heads, seq_len, test_instance.head_dim, device=test_instance.device)
                scale = 1.0 / (test_instance.head_dim ** 0.5)
                
                # PyTorch implementation
                def pytorch_attention(q, k, v, scale):
                    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=test_instance.device), diagonal=1).bool()
                    scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                    attn_weights = torch.softmax(scores, dim=-1)
                    output = torch.matmul(attn_weights, v)
                    return output
                
                # Warm-up
                _ = pytorch_attention(q, k, v, scale)
                from eagle.model.triton_kernels.attention import triton_attention
                _ = triton_attention(q, k, v, scale)
                
                # Benchmark PyTorch
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(10):
                    _ = pytorch_attention(q, k, v, scale)
                torch.cuda.synchronize()
                pytorch_time = (time.time() - start_time) * 100  # ms
                
                # Benchmark Triton
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(10):
                    _ = triton_attention(q, k, v, scale)
                torch.cuda.synchronize()
                triton_time = (time.time() - start_time) * 100  # ms
                
                # Print results
                speedup = pytorch_time / triton_time if triton_time > 0 else float('inf')
                print(f"{batch_size:<10} {seq_len:<10} {pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:<10.2f}")
        
        # Test KV cache with different batch sizes and sequence lengths
        print("\nKV Cache Performance:")
        max_seq_lengths = [64, 128, 256, 512, 1024]
        
        print(f"{'Batch Size':<10} {'Max Seq Len':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        for batch_size in batch_sizes:
            for max_seq_len in max_seq_lengths:
                # Create random inputs
                k_cache = torch.randn(batch_size, test_instance.num_heads, max_seq_len, test_instance.head_dim, device=test_instance.device)
                v_cache = torch.randn(batch_size, test_instance.num_heads, max_seq_len, test_instance.head_dim, device=test_instance.device)
                k_new = torch.randn(batch_size, test_instance.num_heads, 1, test_instance.head_dim, device=test_instance.device)
                v_new = torch.randn(batch_size, test_instance.num_heads, 1, test_instance.head_dim, device=test_instance.device)
                current_length = torch.randint(0, max_seq_len - 1, (batch_size,), dtype=torch.int32, device=test_instance.device)
                
                # PyTorch implementation
                def pytorch_append_to_kv_cache(k_cache, v_cache, k_new, v_new, current_length):
                    k_cache_copy = k_cache.clone()
                    v_cache_copy = v_cache.clone()
                    current_length_copy = current_length.clone()
                    
                    for b in range(batch_size):
                        curr_len = current_length_copy[b].item()
                        k_cache_copy[b, :, curr_len:curr_len+1] = k_new[b]
                        v_cache_copy[b, :, curr_len:curr_len+1] = v_new[b]
                        current_length_copy[b] += 1
                    
                    return k_cache_copy, v_cache_copy, current_length_copy
                
                # Warm-up
                _ = pytorch_append_to_kv_cache(k_cache.clone(), v_cache.clone(), k_new, v_new, current_length.clone())
                from eagle.model.triton_kernels.kv_cache import triton_append_to_kv_cache
                k_cache_triton = k_cache.clone()
                v_cache_triton = v_cache.clone()
                current_length_triton = current_length.clone()
                triton_append_to_kv_cache(k_cache_triton, v_cache_triton, k_new, v_new, current_length_triton)
                
                # Benchmark PyTorch
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(10):
                    _ = pytorch_append_to_kv_cache(k_cache.clone(), v_cache.clone(), k_new, v_new, current_length.clone())
                torch.cuda.synchronize()
                pytorch_time = (time.time() - start_time) * 100  # ms
                
                # Benchmark Triton
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(10):
                    k_cache_triton = k_cache.clone()
                    v_cache_triton = v_cache.clone()
                    current_length_triton = current_length.clone()
                    triton_append_to_kv_cache(k_cache_triton, v_cache_triton, k_new, v_new, current_length_triton)
                torch.cuda.synchronize()
                triton_time = (time.time() - start_time) * 100  # ms
                
                # Print results
                speedup = pytorch_time / triton_time if triton_time > 0 else float('inf')
                print(f"{batch_size:<10} {max_seq_len:<10} {pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:<10.2f}")
        
        # Test top-k with different batch sizes and vocabulary sizes
        print("\nTop-K Performance:")
        vocab_sizes = [10000, 32000, 50000]
        k_values = [5, 10, 50, 100]
        
        print(f"{'Batch Size':<10} {'Vocab Size':<10} {'K':<5} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
        print("-" * 65)
        
        for batch_size in batch_sizes:
            for vocab_size in vocab_sizes:
                for k in k_values:
                    # Create random inputs
                    logits = torch.randn(batch_size, vocab_size, device=test_instance.device)
                    
                    # PyTorch implementation
                    def pytorch_compute_topk(logits, k):
                        return torch.topk(logits, k)
                    
                    # Warm-up
                    _ = pytorch_compute_topk(logits, k)
                    from eagle.model.triton_kernels.tree_decoding import triton_compute_topk
                    _ = triton_compute_topk(logits, k)
                    
                    # Benchmark PyTorch
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(10):
                        _ = pytorch_compute_topk(logits, k)
                    torch.cuda.synchronize()
                    pytorch_time = (time.time() - start_time) * 100  # ms
                    
                    # Benchmark Triton
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(10):
                        _ = triton_compute_topk(logits, k)
                    torch.cuda.synchronize()
                    triton_time = (time.time() - start_time) * 100  # ms
                    
                    # Print results
                    speedup = pytorch_time / triton_time if triton_time > 0 else float('inf')
                    print(f"{batch_size:<10} {vocab_size:<10} {k:<5} {pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:<10.2f}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Run tests for Triton kernels")
    parser.add_argument("--verbose", action="store_true", help="Print detailed test output")
    parser.add_argument("--detailed-report", action="store_true", help="Generate detailed performance report")
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, tests will run on CPU and may be slow")
    else:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    success = run_all_tests(args.verbose, args.detailed_report)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Check if Triton is available
    try:
        import triton
        main()
    except ImportError:
        print("Triton is not installed. Please install it with:")
        print("pip install triton")
        print("For more information, visit: https://github.com/openai/triton")