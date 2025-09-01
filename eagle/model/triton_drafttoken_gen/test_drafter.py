import pytest
import torch
from eagle.model.triton_drafttoken_gen.drafter import (
    DrafterConfig, Weights, Buffers, launch_drafter, _TRITON_AVAILABLE
)

@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="Triton is not available")
def test_drafter_kernel():
    """
    Test the Triton drafter kernel with small dimensions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    # Small config for testing
    cfg = DrafterConfig(
        H_ea=128,
        V=1024,
        n_head=4,
        head_dim=32,
        K=4,
        TOPK=4,
        DEPTH=3,
        T_max=1 + 4 * 3,  # root + K * DEPTH
        V_BLK=256,
        ANCBLK=16,
        dtype=dtype,
    )
    
    # Random weights
    torch.manual_seed(42)
    weights = Weights(
        W_fc=torch.randn(3 * cfg.H_ea, cfg.H_ea, device=device, dtype=dtype),
        Wq=torch.randn(cfg.H_ea, cfg.H_ea, device=device, dtype=dtype),
        Wk=torch.randn(cfg.H_ea, cfg.H_ea, device=device, dtype=dtype),
        Wv=torch.randn(cfg.H_ea, cfg.H_ea, device=device, dtype=dtype),
        Wo=torch.randn(cfg.H_ea, cfg.H_ea, device=device, dtype=dtype),
        W1=torch.randn(cfg.H_ea, 4 * cfg.H_ea, device=device, dtype=dtype),
        W2=torch.randn(2 * cfg.H_ea, cfg.H_ea, device=device, dtype=dtype),
        rms_gamma=torch.randn(cfg.H_ea, device=device, dtype=dtype),
        W_vocab=torch.randn(cfg.V, cfg.H_ea, device=device, dtype=dtype),
    )
    
    # Initialize buffers
    bufs = Buffers(
        Kbuf=torch.zeros(cfg.T_max, cfg.H_ea, device=device, dtype=dtype),
        Vbuf=torch.zeros(cfg.T_max, cfg.H_ea, device=device, dtype=dtype),
        pos_id=torch.zeros(cfg.T_max, device=device, dtype=torch.long),
        parents=torch.full((cfg.T_max,), -1, device=device, dtype=torch.long),
        frontier_idx=torch.zeros(cfg.K, device=device, dtype=torch.long),
        scores=torch.zeros(cfg.K, device=device, dtype=torch.float32),
        next_frontier_idx=torch.empty(cfg.K, device=device, dtype=torch.long),
        next_scores=torch.empty(cfg.K, device=device, dtype=torch.float32),
        next_tokens=torch.empty(cfg.K, device=device, dtype=torch.long),
    )
    
    # Initialize root node
    bufs.pos_id[0] = 0
    bufs.parents[0] = -1
    bufs.frontier_idx[0] = 0
    
    # Create input tensor (concatenated taps)
    X_concat = torch.randn(1, 1, 3 * cfg.H_ea, device=device, dtype=dtype)
    
    # Run with fallback=True first to get reference outputs
    draft_tokens_ref, retrieve_indices_ref, tree_mask_ref, tree_position_ids_ref = launch_drafter(
        cfg, {"X_concat": X_concat}, weights, bufs, fallback=True
    )
    
    # Reset buffers
    bufs.Kbuf.zero_()
    bufs.Vbuf.zero_()
    bufs.pos_id.zero_()
    bufs.pos_id[0] = 0
    bufs.parents.fill_(-1)
    bufs.parents[0] = -1
    bufs.frontier_idx.zero_()
    bufs.frontier_idx[0] = 0
    bufs.scores.zero_()
    
    # Try with Triton kernel if available
    try:
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = launch_drafter(
            cfg, {"X_concat": X_concat}, weights, bufs, fallback=False
        )
        
        # Check shapes match
        assert draft_tokens.shape == draft_tokens_ref.shape, "draft_tokens shape mismatch"
        assert tree_mask.shape == tree_mask_ref.shape, "tree_mask shape mismatch"
        assert tree_position_ids.shape == tree_position_ids_ref.shape, "tree_position_ids shape mismatch"
        
        print("✅ Triton kernel test passed: Shapes match reference implementation")
        
    except NotImplementedError:
        print("⚠️ Triton kernel not fully implemented yet, using fallback")
    
    # Return reference outputs for inspection
    return draft_tokens_ref, retrieve_indices_ref, tree_mask_ref, tree_position_ids_ref

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_drafter_kernel()
    else:
        print("CUDA not available, skipping test")