"""
frontier_integration.py — Integration of Triton drafter with the frontier API.

This file provides a backend for the frontier API that uses the Triton drafter.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import torch

from eagle.model.triton_drafttoken_gen.drafter import (
    DrafterConfig, Weights, Buffers, launch_drafter, _TRITON_AVAILABLE
)
from eagle.model.triton_drafttoken_gen.frontier_api import (
    FrontierConfig, FrontierOutput
)


def _extract_weights_from_ea_layer(ea_layer) -> Weights:
    """
    Extract weights from an EA layer for use in the Triton drafter.
    
    Args:
        ea_layer: The EA layer to extract weights from
        
    Returns:
        Weights object containing the extracted weights
    """
    device = next(ea_layer.parameters()).device
    dtype = next(ea_layer.parameters()).dtype
    
    # Get weights from EA model
    fc = ea_layer.fc
    midlayer = ea_layer.midlayer
    lm_head = ea_layer.lm_head
    norm = ea_layer.norm
    
    # Extract attention weights
    Wq = midlayer.self_attn.q_proj.weight
    Wk = midlayer.self_attn.k_proj.weight
    Wv = midlayer.self_attn.v_proj.weight
    Wo = midlayer.self_attn.o_proj.weight
    
    # Extract MLP weights
    W1_gate = midlayer.mlp.gate_proj.weight
    W1_up = midlayer.mlp.up_proj.weight
    W1 = torch.cat([W1_gate, W1_up], dim=0)  # Combine for SwiGLU
    W2 = midlayer.mlp.down_proj.weight
    
    # Extract normalization weights
    rms_gamma = midlayer.post_attention_layernorm.weight
    
    # Extract LM head weights
    W_vocab = lm_head.weight
    
    return Weights(
        W_fc=fc.weight,
        Wq=Wq,
        Wk=Wk,
        Wv=Wv,
        Wo=Wo,
        W1=W1,
        W2=W2,
        rms_gamma=rms_gamma,
        W_vocab=W_vocab,
    )


def _backend_triton_drafter(
    cfg: FrontierConfig,
    *,
    features_concat: torch.Tensor,
    ea_layer=None,
    input_ids: Optional[torch.Tensor] = None,
    logits_processor=None,
    use_fallback: bool = False,
) -> FrontierOutput:
    """
    Backend that uses the Triton drafter to generate the frontier.
    
    Args:
        cfg: FrontierConfig with generation parameters
        features_concat: Concatenated features tensor
        ea_layer: EA layer (used to extract weights)
        input_ids: Input token IDs (not used by the Triton drafter)
        logits_processor: Optional logits processor
        use_fallback: Whether to use the PyTorch fallback implementation
        
    Returns:
        FrontierOutput with the generated frontier
    """
    device = features_concat.device
    dtype = features_concat.dtype
    
    # Create drafter config from frontier config
    drafter_cfg = DrafterConfig(
        H_ea=cfg.hidden_size,
        V=cfg.vocab_size,
        n_head=ea_layer.config.num_attention_heads if hasattr(ea_layer, "config") else 32,
        head_dim=cfg.hidden_size // ea_layer.config.num_attention_heads if hasattr(ea_layer, "config") else 32,
        K=cfg.top_k,
        TOPK=cfg.top_k,
        DEPTH=cfg.depth,
        T_max=cfg.total_token,
        use_fc=cfg.use_fc_align,
        use_concat_taps=cfg.use_concat_taps,
        dtype=dtype,
    )
    
    # Extract weights from EA layer
    weights = _extract_weights_from_ea_layer(ea_layer)
    
    # Initialize buffers
    bufs = Buffers(
        Kbuf=torch.zeros(cfg.total_token, cfg.hidden_size, device=device, dtype=dtype),
        Vbuf=torch.zeros(cfg.total_token, cfg.hidden_size, device=device, dtype=dtype),
        pos_id=torch.zeros(cfg.total_token, device=device, dtype=torch.long),
        parents=torch.full((cfg.total_token,), -1, device=device, dtype=torch.long),
        frontier_idx=torch.zeros(cfg.top_k, device=device, dtype=torch.long),
        scores=torch.zeros(cfg.top_k, device=device, dtype=torch.float32),
        next_frontier_idx=torch.empty(cfg.top_k, device=device, dtype=torch.long),
        next_scores=torch.empty(cfg.top_k, device=device, dtype=torch.float32),
        next_tokens=torch.empty(cfg.top_k, device=device, dtype=torch.long),
    )
    
    # Initialize root node
    bufs.pos_id[0] = 0
    bufs.parents[0] = -1
    bufs.frontier_idx[0] = 0
    
    # Launch drafter
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = launch_drafter(
        drafter_cfg,
        {"X_concat": features_concat},
        weights,
        bufs,
        fallback=use_fallback or not _TRITON_AVAILABLE,
    )
    
    return FrontierOutput(draft_tokens, retrieve_indices, tree_mask, tree_position_ids)


def register_triton_backend():
    """
    Register the Triton drafter backend with the frontier API.
    
    This function monkey-patches the frontier_api module to add the "triton" backend.
    """
    from eagle.model.triton_drafttoken_gen import frontier_api
    
    # Add the Triton backend to the frontier_generate function
    original_frontier_generate = frontier_api.frontier_generate
    
    def patched_frontier_generate(
        cfg: FrontierConfig,
        *,
        features_concat: Optional[torch.Tensor] = None,
        features_low: Optional[torch.Tensor] = None,
        features_mid: Optional[torch.Tensor] = None,
        features_high: Optional[torch.Tensor] = None,
        backend: str = "stub",
        ea_layer: Optional[object] = None,
        input_ids: Optional[torch.Tensor] = None,
        logits_processor: Optional[object] = None,
        use_fallback: bool = False,
    ) -> FrontierOutput:
        """
        Generate the EAGLE draft frontier using either a stub backend (shape-only),
        the real EA layer backend, or the Triton drafter backend.
        
        Args
        ----
        cfg : FrontierConfig
        features_concat : [1, L, hidden or 3*H_tap]  if use_concat_taps=True
        features_low/mid/high : [1, L, H_tap]        if use_concat_taps=False
        backend : "stub" | "ea_layer" | "triton"
        ea_layer : required when backend="ea_layer" or backend="triton"
        input_ids : required when backend="ea_layer"
        logits_processor : optional (sampler)
        use_fallback : whether to use the PyTorch fallback for the Triton backend
        
        Returns
        -------
        FrontierOutput(draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
        """
        if backend == "triton":
            assert ea_layer is not None, "ea_layer required for backend='triton'"
            if cfg.use_concat_taps:
                assert features_concat is not None, "features_concat required when use_concat_taps=True"
                return _backend_triton_drafter(
                    cfg, 
                    features_concat=features_concat, 
                    ea_layer=ea_layer,
                    input_ids=input_ids, 
                    logits_processor=logits_processor,
                    use_fallback=use_fallback,
                )
            else:
                assert all(t is not None for t in (features_low, features_mid, features_high)), \
                    "features_low/mid/high required when use_concat_taps=False"
                features_concat = torch.cat([features_low, features_mid, features_high], dim=-1)
                return _backend_triton_drafter(
                    cfg, 
                    features_concat=features_concat, 
                    ea_layer=ea_layer,
                    input_ids=input_ids, 
                    logits_processor=logits_processor,
                    use_fallback=use_fallback,
                )
        else:
            return original_frontier_generate(
                cfg,
                features_concat=features_concat,
                features_low=features_low,
                features_mid=features_mid,
                features_high=features_high,
                backend=backend,
                ea_layer=ea_layer,
                input_ids=input_ids,
                logits_processor=logits_processor,
            )
    
    # Replace the original function with the patched one
    frontier_api.frontier_generate = patched_frontier_generate


def example_usage():
    """
    Example of how to use the Triton drafter with the frontier API.
    """
    from eagle.model.triton_drafttoken_gen.frontier_api import FrontierConfig, frontier_generate
    
    # Register the Triton backend
    register_triton_backend()
    
    # Create a frontier config
    cfg = FrontierConfig(
        total_token=60,
        depth=5,
        top_k=10,
        vocab_size=32000,
        hidden_size=4096,
        use_concat_taps=True,
        use_fc_align=True,
        device="cuda",
    )
    
    # Create dummy inputs
    features_concat = torch.randn(1, 1, 3 * 4096, device="cuda", dtype=torch.float16)
    
    # Create a dummy EA layer (in practice, you would use a real one)
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    config = EConfig.from_pretrained('config.json')
    ea_layer = Model(config, load_emb=False)
    ea_layer.to("cuda")
    
    # Generate the frontier using the Triton backend
    frontier = frontier_generate(
        cfg,
        features_concat=features_concat,
        backend="triton",
        ea_layer=ea_layer,
        use_fallback=True,  # Use PyTorch fallback for testing
    )
    
    print(f"Generated {frontier.draft_tokens.shape[1]} draft tokens")
    print(f"Tree mask shape: {frontier.tree_mask.shape}")
    print(f"Tree position IDs: {frontier.tree_position_ids}")
    print(f"Retrieve indices shape: {frontier.retrieve_indices.shape}")
    
    return frontier


if __name__ == "__main__":
    if torch.cuda.is_available():
        example_usage()
    else:
        print("CUDA not available, skipping example")