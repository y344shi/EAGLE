import torch
import warnings

from .attention import triton_attention
from .kv_cache import (
    triton_append_to_kv_cache,
    triton_retrieve_from_kv_cache,
)
from .tree_decoding import (
    triton_compute_topk,
    triton_compute_tree_mask,
    triton_evaluate_posterior,
    triton_update_inputs,
)

def triton_attention_with_fallback(q, k, v, scale=None, mask=None):
    """Backward compatible wrapper calling :func:`triton_attention`."""
    return triton_attention(q, k, v, mask=mask, scale=scale)


def triton_append_to_kv_cache_with_fallback(k_cache, v_cache, k_new, v_new, current_length):
    """Wrapper for :func:`triton_append_to_kv_cache`."""
    return triton_append_to_kv_cache(k_cache, v_cache, k_new, v_new, current_length)


def triton_retrieve_from_kv_cache_with_fallback(k_cache, v_cache, indices):
    """Wrapper for :func:`triton_retrieve_from_kv_cache`."""
    return triton_retrieve_from_kv_cache(k_cache, v_cache, indices)


def triton_compute_topk_with_fallback(logits, k):
    """Wrapper for :func:`triton_compute_topk`."""
    return triton_compute_topk(logits, k)


def triton_compute_tree_mask_with_fallback(parents, total_tokens):
    """Wrapper for :func:`triton_compute_tree_mask`."""
    return triton_compute_tree_mask(parents, total_tokens)


def triton_evaluate_posterior_with_fallback(logits, candidates):
    """Wrapper for :func:`triton_evaluate_posterior`."""
    return triton_evaluate_posterior(logits, candidates)


def triton_update_inputs_with_fallback(input_ids, candidates, best_candidate, accept_length):
    """Wrapper for :func:`triton_update_inputs`."""
    return triton_update_inputs(input_ids, candidates, best_candidate, accept_length)


def optimize_eagle_with_triton(model):
    """
    Optimize an EAGLE model with Triton kernels.
    
    Parameters:
        model: EAGLE model instance
        
    Returns:
        model: Optimized EAGLE model
        original_methods: Dictionary of original methods for restoring
    """
    original_methods = {}
    
    # Store original methods
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'layers'):
        # Replace attention computation in LLaMA-style models
        for layer in model.base_model.model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'forward'):
                if 'attention' not in original_methods:
                    original_methods['attention'] = layer.self_attn.forward
                
                # Create a new forward method that uses Triton attention
                def new_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
                    # Extract query, key, value projections from the original implementation
                    q = self.q_proj(hidden_states)
                    k = self.k_proj(hidden_states)
                    v = self.v_proj(hidden_states)
                    
                    # Reshape for multi-head attention
                    batch_size, seq_len, _ = hidden_states.shape
                    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    
                    # Use accelerated Triton attention
                    attn_output = triton_attention(q, k, v)
                    
                    # Reshape back
                    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                    
                    # Apply output projection
                    attn_output = self.o_proj(attn_output)
                    
                    return attn_output, None, past_key_value
                
                # Bind the new method to the layer
                import types
                layer.self_attn.forward = types.MethodType(new_forward, layer.self_attn)
    
    # Replace KV cache operations if present
    if hasattr(model, 'append_to_kv_cache'):
        original_methods['append_to_kv_cache'] = model.append_to_kv_cache
        model.append_to_kv_cache = lambda k_cache, v_cache, k_new, v_new, current_length: triton_append_to_kv_cache(k_cache, v_cache, k_new, v_new, current_length)
    
    if hasattr(model, 'retrieve_from_kv_cache'):
        original_methods['retrieve_from_kv_cache'] = model.retrieve_from_kv_cache
        model.retrieve_from_kv_cache = lambda k_cache, v_cache, indices: triton_retrieve_from_kv_cache(k_cache, v_cache, indices)
    
    # Replace tree-based token generation operations if present
    if hasattr(model, 'compute_topk'):
        original_methods['compute_topk'] = model.compute_topk
        model.compute_topk = lambda logits, k: triton_compute_topk(logits, k)
    
    if hasattr(model, 'compute_tree_mask'):
        original_methods['compute_tree_mask'] = model.compute_tree_mask
        model.compute_tree_mask = lambda parents, total_tokens: triton_compute_tree_mask(parents, total_tokens)
    
    if hasattr(model, 'evaluate_posterior'):
        original_methods['evaluate_posterior'] = model.evaluate_posterior
        model.evaluate_posterior = lambda logits, candidates: triton_evaluate_posterior(logits, candidates)
    
    if hasattr(model, 'update_inputs'):
        original_methods['update_inputs'] = model.update_inputs
        model.update_inputs = lambda input_ids, candidates, best_candidate, accept_length: triton_update_inputs(input_ids, candidates, best_candidate, accept_length)
    
    return model, original_methods