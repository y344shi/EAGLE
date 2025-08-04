import torch
import numpy as np
import unittest
import time
from typing import Tuple, List, Optional
import sys
import os

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestTritonKernels(unittest.TestCase):
    """Test suite for Triton-optimized kernels."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        # Set default tolerance for numerical comparisons
        self.rtol = 1e-4  # relative tolerance
        self.atol = 1e-5  # absolute tolerance
        
        # Set default test dimensions
        self.batch_size = 2
        self.num_heads = 8
        self.seq_len = 16
        self.head_dim = 64
        self.vocab_size = 32000
    
    def test_attention(self):
        """Test attention computation."""
        print("\n=== Testing Attention Computation ===")

        if self.device.type != "cuda":
            self.skipTest("Triton attention requires CUDA")
        
        # Create random inputs
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        scale = 1.0 / np.sqrt(self.head_dim)

        # PyTorch implementation
        def pytorch_attention(q, k, v, scale):
            # [batch_size, num_heads, seq_len_q, seq_len_k]
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale

            # Apply causal mask (for autoregressive generation)
            causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=self.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Apply softmax
            attn_weights = torch.softmax(scores, dim=-1)

            # Compute weighted sum
            output = torch.matmul(attn_weights, v)
            return output

        # Run PyTorch implementation
        start_time = time.time()
        pytorch_output = pytorch_attention(q, k, v, scale)
        pytorch_time = time.time() - start_time
        print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")

        # Run Triton implementation
        from eagle.model.triton_kernels.attention import triton_attention
        start_time = time.time()
        triton_output = triton_attention(q, k, v, scale)
        triton_time = time.time() - start_time

        # Check correctness
        self.assertTrue(torch.allclose(pytorch_output, triton_output, rtol=self.rtol, atol=self.atol),
                        "Attention outputs do not match")

        # Print performance comparison
        print(f"Triton time: {triton_time * 1000:.4f} ms")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")

        # Test with different sequence lengths
        for seq_len in [32, 64, 128]:
            print(f"\nTesting with sequence length {seq_len}")
            q = torch.randn(self.batch_size, self.num_heads, seq_len, self.head_dim, device=self.device)
            k = torch.randn(self.batch_size, self.num_heads, seq_len, self.head_dim, device=self.device)
            v = torch.randn(self.batch_size, self.num_heads, seq_len, self.head_dim, device=self.device)

            # Adjust causal mask for PyTorch implementation
            def pytorch_attention_adjusted(q, k, v, scale):
                scores = torch.matmul(q, k.transpose(-1, -2)) * scale
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v)
                return output

            # Run PyTorch implementation
            start_time = time.time()
            pytorch_output = pytorch_attention_adjusted(q, k, v, scale)
            pytorch_time = time.time() - start_time
            print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")

            # Run Triton implementation
            start_time = time.time()
            triton_output = triton_attention(q, k, v, scale)
            triton_time = time.time() - start_time

            # Check correctness
            self.assertTrue(torch.allclose(pytorch_output, triton_output, rtol=self.rtol, atol=self.atol),
                            f"Attention outputs do not match for sequence length {seq_len}")

            # Print performance comparison
            print(f"Triton time: {triton_time * 1000:.4f} ms")
            print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    
    def test_kv_cache(self):
        """Test KV cache operations."""
        print("\n=== Testing KV Cache Operations ===")
        
        # Test append_to_kv_cache
        print("\nTesting append_to_kv_cache")
        
        # Create random inputs
        max_seq_len = 64
        k_cache = torch.randn(self.batch_size, self.num_heads, max_seq_len, self.head_dim, device=self.device)
        v_cache = torch.randn(self.batch_size, self.num_heads, max_seq_len, self.head_dim, device=self.device)
        k_new = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim, device=self.device)
        v_new = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim, device=self.device)
        current_length = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        
        # PyTorch implementation
        def pytorch_append_to_kv_cache(k_cache, v_cache, k_new, v_new, current_length):
            k_cache_copy = k_cache.clone()
            v_cache_copy = v_cache.clone()
            current_length_copy = current_length.clone()
            
            for b in range(self.batch_size):
                curr_len = current_length_copy[b].item()
                k_cache_copy[b, :, curr_len:curr_len+1] = k_new[b]
                v_cache_copy[b, :, curr_len:curr_len+1] = v_new[b]
                current_length_copy[b] += 1
            
            return k_cache_copy, v_cache_copy, current_length_copy
        
        # Run PyTorch implementation
        start_time = time.time()
        k_cache_pytorch, v_cache_pytorch, current_length_pytorch = pytorch_append_to_kv_cache(
            k_cache.clone(), v_cache.clone(), k_new, v_new, current_length.clone()
        )
        pytorch_time = time.time() - start_time
        print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")
        
        # Run Triton implementation
        from eagle.model.triton_kernels.kv_cache import triton_append_to_kv_cache
        start_time = time.time()
        k_cache_triton = k_cache.clone()
        v_cache_triton = v_cache.clone()
        current_length_triton = current_length.clone()
        triton_append_to_kv_cache(k_cache_triton, v_cache_triton, k_new, v_new, current_length_triton)
        triton_time = time.time() - start_time
        
        # Check correctness
        self.assertTrue(torch.allclose(k_cache_pytorch, k_cache_triton, rtol=self.rtol, atol=self.atol),
                        "KV cache key outputs do not match")
        self.assertTrue(torch.allclose(v_cache_pytorch, v_cache_triton, rtol=self.rtol, atol=self.atol),
                        "KV cache value outputs do not match")
        self.assertTrue(torch.all(current_length_pytorch == current_length_triton),
                        "KV cache length outputs do not match")
        
        # Print performance comparison
        print(f"Triton time: {triton_time * 1000:.4f} ms")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")
        
        # Test retrieve_from_kv_cache
        print("\nTesting retrieve_from_kv_cache")
        
        # Create random inputs
        indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32, device=self.device)
        
        # PyTorch implementation
        def pytorch_retrieve_from_kv_cache(k_cache, v_cache, indices):
            batch_size, num_heads, _, head_dim = k_cache.shape
            out_seq_len = indices.shape[1]
            
            k_out = torch.zeros((batch_size, num_heads, out_seq_len, head_dim), 
                               dtype=k_cache.dtype, device=k_cache.device)
            v_out = torch.zeros((batch_size, num_heads, out_seq_len, head_dim), 
                               dtype=v_cache.dtype, device=v_cache.device)
            
            for b in range(batch_size):
                for s in range(out_seq_len):
                    idx = indices[b, s].item()
                    k_out[b, :, s] = k_cache[b, :, idx]
                    v_out[b, :, s] = v_cache[b, :, idx]
            
            return k_out, v_out
        
        # Run PyTorch implementation
        start_time = time.time()
        k_out_pytorch, v_out_pytorch = pytorch_retrieve_from_kv_cache(k_cache, v_cache, indices)
        pytorch_time = time.time() - start_time
        print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")
        
        # Run Triton implementation
        from eagle.model.triton_kernels.kv_cache import triton_retrieve_from_kv_cache
        start_time = time.time()
        k_out_triton, v_out_triton = triton_retrieve_from_kv_cache(k_cache, v_cache, indices)
        triton_time = time.time() - start_time
        
        # Check correctness
        self.assertTrue(torch.allclose(k_out_pytorch, k_out_triton, rtol=self.rtol, atol=self.atol),
                        "KV cache retrieval key outputs do not match")
        self.assertTrue(torch.allclose(v_out_pytorch, v_out_triton, rtol=self.rtol, atol=self.atol),
                        "KV cache retrieval value outputs do not match")
        
        # Print performance comparison
        print(f"Triton time: {triton_time * 1000:.4f} ms")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    
    def test_compute_topk(self):
        """Test top-k computation."""
        print("\n=== Testing Top-K Computation ===")
        
        # Create random inputs
        logits = torch.randn(self.batch_size, self.vocab_size, device=self.device)
        k = 5
        
        # PyTorch implementation
        def pytorch_compute_topk(logits, k):
            return torch.topk(logits, k)
        
        # Run PyTorch implementation
        start_time = time.time()
        values_pytorch, indices_pytorch = pytorch_compute_topk(logits, k)
        pytorch_time = time.time() - start_time
        print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")
        
        # Run Triton implementation
        from eagle.model.triton_kernels.tree_decoding import triton_compute_topk
        start_time = time.time()
        values_triton, indices_triton = triton_compute_topk(logits, k)
        triton_time = time.time() - start_time
        
        # Check correctness
        self.assertTrue(torch.allclose(values_pytorch, values_triton, rtol=self.rtol, atol=self.atol),
                        "Top-k values do not match")
        self.assertTrue(torch.all(indices_pytorch == indices_triton),
                        "Top-k indices do not match")
        
        # Print performance comparison
        print(f"Triton time: {triton_time * 1000:.4f} ms")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")
        
        # Test with different k values
        for k in [10, 50, 100]:
            print(f"\nTesting with k={k}")
            
            # Run PyTorch implementation
            start_time = time.time()
            values_pytorch, indices_pytorch = pytorch_compute_topk(logits, k)
            pytorch_time = time.time() - start_time
            print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")
            
            # Run Triton implementation
            start_time = time.time()
            values_triton, indices_triton = triton_compute_topk(logits, k)
            triton_time = time.time() - start_time
            
            # Check correctness
            self.assertTrue(torch.allclose(values_pytorch, values_triton, rtol=self.rtol, atol=self.atol),
                            f"Top-k values do not match for k={k}")
            self.assertTrue(torch.all(indices_pytorch == indices_triton),
                            f"Top-k indices do not match for k={k}")
            
            # Print performance comparison
            print(f"Triton time: {triton_time * 1000:.4f} ms")
            print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    
    def test_tree_mask(self):
        """Test tree mask computation."""
        print("\n=== Testing Tree Mask Computation ===")
        
        # Create random inputs
        total_tokens = 8
        parents = torch.tensor([
            [0, 0, 1, 1, 2, 3, 4, 5],  # First batch
            [0, 0, 0, 1, 2, 2, 3, 4]   # Second batch
        ], dtype=torch.int32, device=self.device)
        
        # PyTorch implementation
        def pytorch_compute_tree_mask(parents, total_tokens):
            batch_size = parents.shape[0]
            tree_mask = torch.zeros((batch_size, total_tokens, total_tokens), 
                                   dtype=torch.int32, device=parents.device)
            
            # Initialize: each token can see itself
            for b in range(batch_size):
                for t in range(total_tokens):
                    tree_mask[b, t, t] = 1
            
            # For each token, determine which previous tokens it can attend to
            for b in range(batch_size):
                for t in range(1, total_tokens):
                    parent = parents[b, t].item()
                    if parent == 0:
                        # Root token (0) only sees itself
                        continue
                    else:
                        # Copy parent's mask
                        for i in range(total_tokens):
                            if tree_mask[b, parent, i] == 1:
                                tree_mask[b, t, i] = 1
            
            return tree_mask
        
        # Run PyTorch implementation
        start_time = time.time()
        tree_mask_pytorch = pytorch_compute_tree_mask(parents, total_tokens)
        pytorch_time = time.time() - start_time
        print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")
        
        # Run Triton implementation
        from eagle.model.triton_kernels.tree_decoding import triton_compute_tree_mask
        start_time = time.time()
        tree_mask_triton = triton_compute_tree_mask(parents, total_tokens)
        triton_time = time.time() - start_time
        
        # Check correctness
        self.assertTrue(torch.all(tree_mask_pytorch == tree_mask_triton),
                        "Tree masks do not match")
        
        # Print performance comparison
        print(f"Triton time: {triton_time * 1000:.4f} ms")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")
        
        # Test with different tree sizes
        for total_tokens in [16, 32]:
            print(f"\nTesting with tree size {total_tokens}")
            
            # Create random parents
            parents = torch.randint(0, total_tokens, (self.batch_size, total_tokens), 
                                   dtype=torch.int32, device=self.device)
            # Ensure valid tree structure
            for b in range(self.batch_size):
                parents[b, 0] = 0  # Root is its own parent
                for t in range(1, total_tokens):
                    parents[b, t] = torch.randint(0, t, (1,)).item()
            
            # Run PyTorch implementation
            start_time = time.time()
            tree_mask_pytorch = pytorch_compute_tree_mask(parents, total_tokens)
            pytorch_time = time.time() - start_time
            print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")
            
            # Run Triton implementation
            start_time = time.time()
            tree_mask_triton = triton_compute_tree_mask(parents, total_tokens)
            triton_time = time.time() - start_time
            
            # Check correctness
            self.assertTrue(torch.all(tree_mask_pytorch == tree_mask_triton),
                            f"Tree masks do not match for tree size {total_tokens}")
            
            # Print performance comparison
            print(f"Triton time: {triton_time * 1000:.4f} ms")
            print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    
    def test_evaluate_posterior(self):
        """Test posterior probability evaluation."""
        print("\n=== Testing Posterior Probability Evaluation ===")
        
        # Create random inputs
        seq_len = 5
        num_candidates = 3
        logits = torch.randn(self.batch_size, seq_len, self.vocab_size, device=self.device)
        
        # Create candidates with some matching the max logits
        candidates = torch.zeros((self.batch_size, num_candidates, seq_len), 
                                dtype=torch.int32, device=self.device)
        
        # For testing, make some candidates match the max logits
        for b in range(self.batch_size):
            for s in range(seq_len):
                max_idx = torch.argmax(logits[b, s]).item()
                # First candidate: all tokens match
                candidates[b, 0, s] = max_idx
                # Second candidate: first half match
                candidates[b, 1, s] = max_idx if s < seq_len // 2 else (max_idx + 1) % self.vocab_size
                # Third candidate: only first token matches
                candidates[b, 2, s] = max_idx if s == 0 else (max_idx + 2) % self.vocab_size
        
        # PyTorch implementation
        def pytorch_evaluate_posterior(logits, candidates):
            batch_size, seq_len, vocab_size = logits.shape
            _, num_candidates, _ = candidates.shape
            
            best_candidate = torch.zeros(batch_size, dtype=torch.int32, device=logits.device)
            accept_length = torch.zeros(batch_size, dtype=torch.int32, device=logits.device)
            
            for b in range(batch_size):
                max_accept_len = 0
                max_candidate_idx = 0
                
                for c in range(num_candidates):
                    accept_len = 0
                    
                    for s in range(seq_len):
                        candidate_token = candidates[b, c, s].item()
                        max_logit_idx = torch.argmax(logits[b, s]).item()
                        
                        if max_logit_idx == candidate_token:
                            accept_len += 1
                        else:
                            break
                    
                    if accept_len > max_accept_len:
                        max_accept_len = accept_len
                        max_candidate_idx = c
                
                best_candidate[b] = max_candidate_idx
                accept_length[b] = max_accept_len
            
            return best_candidate, accept_length
        
        # Run PyTorch implementation
        start_time = time.time()
        best_candidate_pytorch, accept_length_pytorch = pytorch_evaluate_posterior(logits, candidates)
        pytorch_time = time.time() - start_time
        print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")
        
        # Run Triton implementation
        from eagle.model.triton_kernels.tree_decoding import triton_evaluate_posterior
        start_time = time.time()
        best_candidate_triton, accept_length_triton = triton_evaluate_posterior(logits, candidates)
        triton_time = time.time() - start_time
        
        # Check correctness
        self.assertTrue(torch.all(best_candidate_pytorch == best_candidate_triton),
                        "Best candidate indices do not match")
        self.assertTrue(torch.all(accept_length_pytorch == accept_length_triton),
                        "Accept lengths do not match")
        
        # Print performance comparison
        print(f"Triton time: {triton_time * 1000:.4f} ms")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")
        
        # Verify expected results
        print("\nVerifying expected results:")
        print(f"Best candidates: {best_candidate_pytorch}")
        print(f"Accept lengths: {accept_length_pytorch}")
        
        # Expected results:
        # - First batch: best candidate should be 0 with accept_length = seq_len
        # - Second batch: best candidate should be 0 with accept_length = seq_len
        self.assertTrue(torch.all(best_candidate_pytorch == 0),
                        "Best candidate should be 0 (all tokens match)")
        self.assertTrue(torch.all(accept_length_pytorch == seq_len),
                        f"Accept length should be {seq_len} (all tokens match)")
    
    def test_update_inputs(self):
        """Test input sequence update."""
        print("\n=== Testing Input Sequence Update ===")

        if self.device.type != "cuda":
            self.skipTest("Triton update_inputs requires CUDA")
        
        # Create random inputs
        input_len = 4
        seq_len = 5
        num_candidates = 3
        
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, input_len), 
                                 dtype=torch.int32, device=self.device)
        candidates = torch.randint(0, self.vocab_size, (self.batch_size, num_candidates, seq_len), 
                                  dtype=torch.int32, device=self.device)
        best_candidate = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        accept_length = torch.tensor([2, 3], dtype=torch.int32, device=self.device)
        
        # PyTorch implementation
        def pytorch_update_inputs(input_ids, candidates, best_candidate, accept_length):
            batch_size, input_len = input_ids.shape
            max_accept_len = int(accept_length.max().item()) + 1
            
            output_ids = torch.zeros((batch_size, input_len + max_accept_len), 
                                    dtype=input_ids.dtype, device=input_ids.device)
            
            for b in range(batch_size):
                # Copy input tokens
                output_ids[b, :input_len] = input_ids[b]
                
                # Copy accepted tokens from the best candidate
                best_idx = best_candidate[b].item()
                acc_len = accept_length[b].item()
                
                for i in range(acc_len + 1):
                    output_ids[b, input_len + i] = candidates[b, best_idx, i]
            
            return output_ids
        
        # Run PyTorch implementation
        start_time = time.time()
        output_ids_pytorch = pytorch_update_inputs(input_ids, candidates, best_candidate, accept_length)
        pytorch_time = time.time() - start_time
        print(f"PyTorch time: {pytorch_time * 1000:.4f} ms")
        
        # Run Triton implementation
        from eagle.model.triton_kernels.tree_decoding import triton_update_inputs
        start_time = time.time()
        output_ids_triton = triton_update_inputs(input_ids, candidates, best_candidate, accept_length)
        triton_time = time.time() - start_time
        
        # Check correctness
        self.assertTrue(torch.all(output_ids_pytorch == output_ids_triton),
                        "Updated input sequences do not match")
        
        # Print performance comparison
        print(f"Triton time: {triton_time * 1000:.4f} ms")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")
        
        # Verify expected results
        print("\nVerifying expected results:")
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output_ids_pytorch.shape}")
        
        # Expected results:
        # - First batch: input_ids + candidates[0, 0, :3]
        # - Second batch: input_ids + candidates[1, 1, :4]
        for b in range(self.batch_size):
            best_idx = best_candidate[b].item()
            acc_len = accept_length[b].item()
            
            # Check input tokens
            self.assertTrue(torch.all(output_ids_pytorch[b, :input_len] == input_ids[b]),
                            f"Input tokens do not match for batch {b}")
            
            # Check accepted tokens
            for i in range(acc_len + 1):
                self.assertEqual(output_ids_pytorch[b, input_len + i].item(), 
                                candidates[b, best_idx, i].item(),
                                f"Accepted token {i} does not match for batch {b}")


if __name__ == "__main__":
    # Check if Triton is available
    try:
        import triton
        unittest.main()
    except ImportError:
        print("Triton is not installed. Please install it with:")
        print("pip install triton")
        print("For more information, visit: https://github.com/openai/triton")