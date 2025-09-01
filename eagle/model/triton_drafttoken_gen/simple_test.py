"""
Simple test script for the Triton drafter that doesn't depend on pytest.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import the drafter module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eagle.model.triton_drafttoken_gen.drafter import (
    DrafterConfig, Weights, Buffers, launch_drafter,
    streaming_topk_lm_head_ref
)

def test_shapes():
    """Test that the drafter produces outputs with the correct shapes."""
    print("=== Testing drafter shapes ===")
    
    # Create a small config for testing
    cfg = DrafterConfig(
        H_ea=128,
        V=1024,
        n_head=4,
        head_dim=32,
        K=4,
        TOPK=4,
        DEPTH=3,
        T_max=16,
    )
    
    # Create dummy inputs
    device = torch.device("cpu")
    X_concat = torch.randn(1, 1, cfg.H_ea, device=device, dtype=torch.float32)
    
    # Initialize buffers
    bufs = Buffers(
        Kbuf=torch.zeros(cfg.T_max, cfg.H_ea, device=device, dtype=torch.float32),
        Vbuf=torch.zeros(cfg.T_max, cfg.H_ea, device=device, dtype=torch.float32),
        pos_id=torch.zeros(cfg.T_max, device=device, dtype=torch.long),
        parents=torch.full((cfg.T_max,), -1, device=device, dtype=torch.long),
        frontier_idx=torch.zeros(cfg.K, device=device, dtype=torch.long),
        scores=torch.zeros(cfg.K, device=device, dtype=torch.float32),
        next_frontier_idx=torch.empty(cfg.K, device=device, dtype=torch.long),
        next_scores=torch.empty(cfg.K, device=device, dtype=torch.float32),
        next_tokens=torch.empty(cfg.K, device=device, dtype=torch.long),
    )
    
    # Initialize weights (not used in fallback mode)
    weights = Weights()
    
    # Run the drafter in fallback mode
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = launch_drafter(
        cfg, {"X_concat": X_concat}, weights, bufs, fallback=True
    )
    
    # Check shapes
    # The fallback implementation may have different shapes for different outputs
    # Let's just print them and make sure they're reasonable
    print(f"Draft tokens shape: {draft_tokens.shape}")
    print(f"Tree mask shape: {tree_mask.shape}")
    print(f"Tree position IDs shape: {tree_position_ids.shape}")
    print(f"Retrieve indices shape: {retrieve_indices.shape}")
    
    # Check that the shapes are reasonable
    assert draft_tokens.shape[0] == 1, "Batch size should be 1"
    assert draft_tokens.shape[1] <= cfg.T_max, "Number of tokens should not exceed T_max"
    assert tree_mask.shape[0] == 1 and tree_mask.shape[1] == 1, "Tree mask should have batch and head dimensions of 1"
    assert tree_mask.shape[2] == tree_mask.shape[3], "Tree mask should be square"
    assert tree_position_ids.numel() > 0, "Tree position IDs should not be empty"
    assert retrieve_indices.shape[1] <= cfg.DEPTH + 1, "Retrieve indices should have at most DEPTH+1 columns"
    
    # retrieve_indices shape depends on the tree structure, but should have the right number of columns
    assert retrieve_indices.shape[1] == cfg.DEPTH + 1, f"Expected retrieve_indices shape (R, {cfg.DEPTH + 1}), got {retrieve_indices.shape}"
    
    print("✅ Shape test passed")
    return True

def test_streaming_topk():
    """Test the streaming top-k implementation."""
    print("\n=== Testing streaming top-k ===")
    
    # Create random inputs
    H = 128
    V = 1024
    topk = 10
    
    device = torch.device("cpu")
    h = torch.randn(H, device=device, dtype=torch.float32)
    W = torch.randn(V, H, device=device, dtype=torch.float32)
    
    # Run the reference implementation
    ref_vals, ref_idxs = streaming_topk_lm_head_ref(h, W, topk)
    
    # Compute the full logits for verification
    full_logits = W @ h
    torch_vals, torch_idxs = torch.topk(full_logits, k=topk)
    
    # Check that the reference implementation matches torch.topk
    max_diff = torch.max(torch.abs(ref_vals - torch_vals)).item()
    assert max_diff < 1e-5, f"Reference top-k doesn't match torch.topk: max diff = {max_diff}"
    
    print("✅ Streaming top-k test passed")
    return True

def main():
    """Run all tests."""
    test_shapes()
    test_streaming_topk()
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()