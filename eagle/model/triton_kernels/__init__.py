"""
Triton-optimized kernels for the EAGLE model.

This package provides Triton-optimized CUDA kernels for the EAGLE model.
"""

# Import direct implementations from modules
from .attention import triton_attention
from .kv_cache import triton_append_to_kv_cache, triton_retrieve_from_kv_cache
from .tree_decoding import (
    triton_compute_topk,
    triton_compute_tree_mask,
    triton_evaluate_posterior,
    triton_update_inputs,
)
from .fused_kernels import fused_attention_kv_tree

# Import integration utilities
from .integration import (
    triton_attention_with_fallback,
    triton_append_to_kv_cache_with_fallback,
    triton_retrieve_from_kv_cache_with_fallback,
    triton_compute_topk_with_fallback,
    triton_compute_tree_mask_with_fallback,
    triton_evaluate_posterior_with_fallback,
    triton_update_inputs_with_fallback,
    optimize_eagle_with_triton,
)

__all__ = [
    # Direct implementations
    'triton_attention',
    'triton_append_to_kv_cache',
    'triton_retrieve_from_kv_cache',
    'triton_compute_topk',
    'triton_compute_tree_mask',
    'triton_evaluate_posterior',
    'triton_update_inputs',
    'fused_attention_kv_tree',
    
    # Integration utilities
    'triton_attention_with_fallback',
    'triton_append_to_kv_cache_with_fallback',
    'triton_retrieve_from_kv_cache_with_fallback',
    'triton_compute_topk_with_fallback',
    'triton_compute_tree_mask_with_fallback',
    'triton_evaluate_posterior_with_fallback',
    'triton_update_inputs_with_fallback',
    'optimize_eagle_with_triton',
]

# Check if Triton is available
try:
    import triton
    HAS_TRITON = True
except ImportError:
    import warnings
    warnings.warn(
        "Triton is not installed. To install Triton, run: pip install triton"
    )
    HAS_TRITON = False
