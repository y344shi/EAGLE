"""
Integration example showing how to use the Triton drafter in the EA layer.
"""

import torch
from typing import Dict, Optional, Tuple

from eagle.model.cnets import Model
from eagle.model.configs import EConfig
from eagle.model.triton_drafttoken_gen.drafter import (
    DrafterConfig, Weights, Buffers, launch_drafter, _TRITON_AVAILABLE
)

class TritonDrafterIntegration:
    """
    Integration class for using the Triton drafter in the EA layer.
    """
    
    def __init__(self, ea_model: Model, use_triton: bool = True):
        """
        Initialize the Triton drafter integration.
        
        Args:
            ea_model: The EA model to integrate with
            use_triton: Whether to use the Triton kernel or fallback to PyTorch
        """
        self.ea_model = ea_model
        self.use_triton = use_triton and _TRITON_AVAILABLE
        self.config = ea_model.config
        
        # Extract dimensions from EA model
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.draft_vocab_size = self.config.draft_vocab_size
        
        # Create drafter config
        self.drafter_config = DrafterConfig(
            H_ea=self.hidden_size,
            V=self.draft_vocab_size,
            n_head=self.config.num_attention_heads,
            head_dim=self.hidden_size // self.config.num_attention_heads,
            K=ea_model.top_k,
            TOPK=ea_model.top_k,
            DEPTH=ea_model.depth,
            T_max=ea_model.total_tokens + 1,  # +1 for root
            use_fc=True,
            use_concat_taps=True,
        )
        
        # Extract weights from EA model
        self.weights = self._extract_weights()
        
        # Initialize buffers (will be created on first use)
        self.buffers = None
    
    def _extract_weights(self) -> Weights:
        """
        Extract weights from the EA model for use in the Triton drafter.
        """
        device = next(self.ea_model.parameters()).device
        dtype = next(self.ea_model.parameters()).dtype
        
        # Get weights from EA model
        fc = self.ea_model.fc
        midlayer = self.ea_model.midlayer
        lm_head = self.ea_model.lm_head
        norm = self.ea_model.norm
        
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
    
    def _initialize_buffers(self, device):
        """
        Initialize buffers for the drafter.
        """
        cfg = self.drafter_config
        dtype = next(self.ea_model.parameters()).dtype
        
        self.buffers = Buffers(
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
        self.buffers.pos_id[0] = 0
        self.buffers.parents[0] = -1
        self.buffers.frontier_idx[0] = 0
    
    def draft_tokens(
        self, 
        hidden_states: torch.Tensor, 
        input_ids: torch.Tensor,
        logits_processor=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate draft tokens using the Triton drafter.
        
        Args:
            hidden_states: Hidden states from the base model
            input_ids: Input token IDs
            logits_processor: Optional logits processor
        
        Returns:
            Tuple of (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
        """
        device = hidden_states.device
        
        # Initialize buffers if needed
        if self.buffers is None:
            self._initialize_buffers(device)
        
        # Reset buffers for new generation
        self.buffers.Kbuf.zero_()
        self.buffers.Vbuf.zero_()
        self.buffers.pos_id.zero_()
        self.buffers.pos_id[0] = 0
        self.buffers.parents.fill_(-1)
        self.buffers.parents[0] = -1
        self.buffers.frontier_idx.zero_()
        self.buffers.frontier_idx[0] = 0
        self.buffers.scores.zero_()
        
        # Prepare input tensor (concatenated taps)
        X_concat = hidden_states.unsqueeze(0)  # [1, L, 3*H]
        
        # Launch drafter
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = launch_drafter(
            self.drafter_config,
            {"X_concat": X_concat},
            self.weights,
            self.buffers,
            fallback=not self.use_triton,
        )
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

def patch_ea_model_with_triton_drafter(model: Model, use_triton: bool = True) -> Model:
    """
    Patch an EA model to use the Triton drafter.
    
    Args:
        model: The EA model to patch
        use_triton: Whether to use the Triton kernel or fallback to PyTorch
    
    Returns:
        The patched model
    """
    # Create integration
    integration = TritonDrafterIntegration(model, use_triton=use_triton)
    
    # Save original method
    original_topK_generate = model.topK_genrate
    
    # Define patched method
    def patched_topK_generate(self, hidden_states, input_ids, head, logits_processor):
        if use_triton and _TRITON_AVAILABLE:
            return integration.draft_tokens(hidden_states, input_ids, logits_processor)
        else:
            return original_topK_generate(hidden_states, input_ids, head, logits_processor)
    
    # Apply patch
    model.topK_genrate = patched_topK_generate.__get__(model, Model)
    
    return model

def main():
    """
    Example usage of the Triton drafter integration.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Load EA model
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    model.to('cuda')
    
    # Patch model with Triton drafter
    model = patch_ea_model_with_triton_drafter(model, use_triton=True)
    
    # Generate some random inputs
    hidden_states = torch.randn(1, 3 * config.hidden_size, device='cuda')
    input_ids = torch.randint(0, config.vocab_size, (1, 10), device='cuda')
    
    # Generate draft tokens
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.topK_genrate(
        hidden_states, input_ids, model.lm_head, None
    )
    
    print(f"Generated {draft_tokens.shape[1]} draft tokens")
    print(f"Tree mask shape: {tree_mask.shape}")
    print(f"Tree position IDs: {tree_position_ids}")
    print(f"Retrieve indices shape: {retrieve_indices.shape}")

if __name__ == "__main__":
    main()