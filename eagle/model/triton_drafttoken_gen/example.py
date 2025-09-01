"""
Example usage of the Triton drafter for EAGLE.

This file demonstrates how to use the Triton drafter in different scenarios:
1. Direct integration with an EA model
2. Through the frontier API
3. Using the low-level API
"""

import torch
from eagle.model.ea_model import EaModel
from eagle.model.triton_drafttoken_gen.drafter import DrafterConfig, Weights, Buffers, launch_drafter
from eagle.model.triton_drafttoken_gen.ea_integration import patch_ea_model_with_triton_drafter
from eagle.model.triton_drafttoken_gen.frontier_api import FrontierConfig, frontier_generate
from eagle.model.triton_drafttoken_gen.frontier_integration import register_triton_backend


def example_direct_integration():
    """Example of direct integration with an EA model."""
    print("=== Direct Integration Example ===")
    
    # Load your EA model (placeholder)
    model = EaModel.from_pretrained("llama-3-8b-eagle")
    
    # Patch the model to use the Triton drafter
    model = patch_ea_model_with_triton_drafter(model, use_triton=True)
    
    # Use the model as usual
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
    outputs = model.generate(input_ids, max_length=20)
    
    print(f"Generated output shape: {outputs.shape}")
    return model


def example_frontier_api(model):
    """Example of using the frontier API."""
    print("\n=== Frontier API Example ===")
    
    # Register the Triton backend
    register_triton_backend()
    
    # Create a frontier config
    cfg = FrontierConfig(
        total_token=60,
        depth=5,
        top_k=10,
        vocab_size=32000,
        hidden_size=4096,
    )
    
    # Create dummy features
    features = torch.randn(1, 1, 3 * 4096, device=model.device, dtype=torch.float16)
    
    # Generate the frontier using the Triton backend
    frontier = frontier_generate(
        cfg,
        features_concat=features,
        backend="triton",
        ea_layer=model.ea_layer,
    )
    
    print(f"Frontier draft_tokens shape: {frontier.draft_tokens.shape}")
    print(f"Frontier retrieve_indices shape: {frontier.retrieve_indices.shape}")
    return frontier


def example_low_level_api(device="cuda"):
    """Example of using the low-level API."""
    print("\n=== Low-Level API Example ===")
    
    # Create configuration
    cfg = DrafterConfig(
        H_ea=128,  # Small for example
        V=1024,    # Small for example
        n_head=4,
        head_dim=32,
        K=5,
        TOPK=5,
        DEPTH=3,
        T_max=16,  # root + K * DEPTH
    )
    
    # Initialize weights with random values
    weights = Weights(
        W_fc=torch.randn(3 * cfg.H_ea, cfg.H_ea, device=device, dtype=torch.float16),
        Wq=torch.randn(cfg.H_ea, cfg.H_ea, device=device, dtype=torch.float16),
        Wk=torch.randn(cfg.H_ea, cfg.H_ea, device=device, dtype=torch.float16),
        Wv=torch.randn(cfg.H_ea, cfg.H_ea, device=device, dtype=torch.float16),
        Wo=torch.randn(cfg.H_ea, cfg.H_ea, device=device, dtype=torch.float16),
        W1=torch.randn(cfg.H_ea, 4 * cfg.H_ea, device=device, dtype=torch.float16),
        W2=torch.randn(2 * cfg.H_ea, cfg.H_ea, device=device, dtype=torch.float16),
        rms_gamma=torch.randn(cfg.H_ea, device=device, dtype=torch.float16),
        W_vocab=torch.randn(cfg.V, cfg.H_ea, device=device, dtype=torch.float16),
    )
    
    # Initialize buffers
    bufs = Buffers(
        Kbuf=torch.zeros(cfg.T_max, cfg.H_ea, device=device, dtype=torch.float16),
        Vbuf=torch.zeros(cfg.T_max, cfg.H_ea, device=device, dtype=torch.float16),
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
    
    # Create dummy features
    X_concat = torch.randn(1, 1, 3 * cfg.H_ea, device=device, dtype=torch.float16)
    
    # Launch the drafter (using fallback=True for CPU)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = launch_drafter(
        cfg, {"X_concat": X_concat}, weights, bufs, fallback=True
    )
    
    print(f"draft_tokens shape: {draft_tokens.shape}")
    print(f"retrieve_indices shape: {retrieve_indices.shape}")
    print(f"tree_mask shape: {tree_mask.shape}")
    print(f"tree_position_ids shape: {tree_position_ids.shape}")
    
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        try:
            # Try to run all examples
            model = example_direct_integration()
            frontier = example_frontier_api(model)
            outputs = example_low_level_api(device)
        except Exception as e:
            print(f"Error running examples with CUDA: {e}")
            print("Falling back to CPU examples...")
            outputs = example_low_level_api("cpu")
    else:
        # Run only the low-level API example with CPU fallback
        print("CUDA not available, running CPU example only")
        outputs = example_low_level_api("cpu")