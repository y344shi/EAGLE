"""
Triton kernels for EAGLE model.
"""

from .tree_attention import tree_attention, tree_decoding_triton
from .topk_expand import topk_expand_triton, topk_generate_triton
from .posterior_eval import evaluate_posterior_triton
from .kv_block_copy import kv_block_copy_triton, update_inference_inputs_triton
from .mask_preparation import prepare_decoder_attention_mask_triton
from .integration import triton_kernels, TritonEagleKernels
from .ea_model_patch import patch_eagle_model, unpatch_eagle_model

__all__ = [
    'tree_attention',
    'tree_decoding_triton',
    'topk_expand_triton',
    'topk_generate_triton',
    'evaluate_posterior_triton',
    'kv_block_copy_triton',
    'update_inference_inputs_triton',
    'prepare_decoder_attention_mask_triton',
    'triton_kernels',
    'TritonEagleKernels',
    'patch_eagle_model',
    'unpatch_eagle_model',
]