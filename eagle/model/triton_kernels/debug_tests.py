import torch
import numpy as np
import time
import traceback
import sys
import os

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def debug_attention():
    """Debug attention computation."""
    print("\n=== Debugging Attention Computation ===")
    
    # Set up test parameters
    batch_size = 2
    num_heads = 8
    seq_len = 16
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    scale = 1.0 / np.sqrt(head_dim)
    
    # Try to import and run Triton attention
    try:
        from attention import triton_attention
        print("Imported triton_attention successfully")
        
        # Try to run the function
        triton_output = triton_attention(q, k, v, scale)
        print("Successfully ran triton_attention")
    except Exception as e:
        print(f"Error in triton_attention: {e}")
        print("\nTraceback:")
        traceback.print_exc()

def debug_kv_cache():
    """Debug KV cache operations."""
    print("\n=== Debugging KV Cache Operations ===")
    
    # Set up test parameters
    batch_size = 2
    num_heads = 8
    max_seq_len = 64
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random inputs for append_to_kv_cache
    k_cache = torch.randn(batch_size, num_heads, max_seq_len, head_dim, device=device)
    v_cache = torch.randn(batch_size, num_heads, max_seq_len, head_dim, device=device)
    k_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
    v_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
    current_length = torch.tensor([0, 1], dtype=torch.int32, device=device)
    
    # Try to import and run Triton append_to_kv_cache
    try:
        from kv_cache import triton_append_to_kv_cache
        print("Imported triton_append_to_kv_cache successfully")
        
        # Try to run the function
        k_cache_copy = k_cache.clone()
        v_cache_copy = v_cache.clone()
        current_length_copy = current_length.clone()
        triton_append_to_kv_cache(k_cache_copy, v_cache_copy, k_new, v_new, current_length_copy)
        print("Successfully ran triton_append_to_kv_cache")
    except Exception as e:
        print(f"Error in triton_append_to_kv_cache: {e}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Create random inputs for retrieve_from_kv_cache
    indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32, device=device)
    
    # Try to import and run Triton retrieve_from_kv_cache
    try:
        from kv_cache import triton_retrieve_from_kv_cache
        print("Imported triton_retrieve_from_kv_cache successfully")
        
        # Try to run the function
        k_out, v_out = triton_retrieve_from_kv_cache(k_cache, v_cache, indices)
        print("Successfully ran triton_retrieve_from_kv_cache")
    except Exception as e:
        print(f"Error in triton_retrieve_from_kv_cache: {e}")
        print("\nTraceback:")
        traceback.print_exc()

def debug_tree_decoding():
    """Debug tree decoding operations."""
    print("\n=== Debugging Tree Decoding Operations ===")
    
    # Set up test parameters
    batch_size = 2
    vocab_size = 32000
    total_tokens = 8
    seq_len = 5
    num_candidates = 3
    input_len = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Debug compute_topk
    print("\nDebugging compute_topk")
    logits = torch.randn(batch_size, vocab_size, device=device)
    k = 5
    
    try:
        from tree_decoding import triton_compute_topk
        print("Imported triton_compute_topk successfully")
        
        # Try to run the function
        values, indices = triton_compute_topk(logits, k)
        print("Successfully ran triton_compute_topk")
    except Exception as e:
        print(f"Error in triton_compute_topk: {e}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Debug compute_tree_mask
    print("\nDebugging compute_tree_mask")
    parents = torch.tensor([
        [0, 0, 1, 1, 2, 3, 4, 5],  # First batch
        [0, 0, 0, 1, 2, 2, 3, 4]   # Second batch
    ], dtype=torch.int32, device=device)
    
    try:
        from tree_decoding import triton_compute_tree_mask
        print("Imported triton_compute_tree_mask successfully")
        
        # Try to run the function
        tree_mask = triton_compute_tree_mask(parents, total_tokens)
        print("Successfully ran triton_compute_tree_mask")
    except Exception as e:
        print(f"Error in triton_compute_tree_mask: {e}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Debug evaluate_posterior
    print("\nDebugging evaluate_posterior")
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    candidates = torch.zeros((batch_size, num_candidates, seq_len), dtype=torch.int32, device=device)
    
    try:
        from tree_decoding import triton_evaluate_posterior
        print("Imported triton_evaluate_posterior successfully")
        
        # Try to run the function
        best_candidate, accept_length = triton_evaluate_posterior(logits, candidates)
        print("Successfully ran triton_evaluate_posterior")
    except Exception as e:
        print(f"Error in triton_evaluate_posterior: {e}")
        print("\nTraceback:")
        traceback.print_exc()
    
    # Debug update_inputs
    print("\nDebugging update_inputs")
    input_ids = torch.randint(0, vocab_size, (batch_size, input_len), dtype=torch.int32, device=device)
    best_candidate = torch.tensor([0, 1], dtype=torch.int32, device=device)
    accept_length = torch.tensor([2, 3], dtype=torch.int32, device=device)
    
    try:
        from tree_decoding import triton_update_inputs
        print("Imported triton_update_inputs successfully")
        
        # Try to run the function
        output_ids = triton_update_inputs(input_ids, candidates, best_candidate, accept_length)
        print("Successfully ran triton_update_inputs")
    except Exception as e:
        print(f"Error in triton_update_inputs: {e}")
        print("\nTraceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # Check if Triton is available
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
        
        # Run debug functions
        debug_attention()
        debug_kv_cache()
        debug_tree_decoding()
    except ImportError:
        print("Triton is not installed. Please install it with:")
        print("$ pip install triton")
    
    print("\n\n=== Done Debugging ===")
