"""
Patch for ea_model.py to use Triton kernels.

This module provides functions to patch the EAGLE model to use Triton kernels
for improved performance.
"""

import torch
from .integration import triton_kernels


def patch_eagle_model(model):
    """
    Patch an EAGLE model to use Triton kernels.
    
    Args:
        model: The EAGLE model to patch
    """
    # Store original functions
    original_tree_decoding = model.tree_decoding
    original_evaluate_posterior = model.evaluate_posterior
    original_update_inference_inputs = model.update_inference_inputs
    
    # Patch Model._prepare_decoder_attention_mask
    original_prepare_decoder_attention_mask = model.base_model.model._prepare_decoder_attention_mask
    
    def patched_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Use Triton implementation if on CUDA and tree_mask is available
        if inputs_embeds.is_cuda and hasattr(self, "tree_mask") and self.tree_mask is not None:
            return triton_kernels.prepare_decoder_attention_mask(
                attention_mask, inputs_embeds, past_key_values_length, self.tree_mask
            )
        # Fall back to original implementation
        return original_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)
    
    # Patch ea_layer.topK_generate
    original_topk_generate = model.ea_layer.topK_genrate
    
    def patched_topk_generate(hidden_states, input_ids, head, logits_processor):
        # Use Triton implementation if on CUDA
        if hidden_states.is_cuda:
            return triton_kernels.topk_generate(
                hidden_states,
                input_ids,
                head,
                logits_processor,
                model.ea_layer.total_tokens,
                model.ea_layer.depth,
                model.ea_layer.top_k,
                model.ea_layer.threshold,
                model.ea_layer.stable_kv if hasattr(model.ea_layer, "stable_kv") else None,
                model.ea_layer.d2t if hasattr(model.ea_layer, "d2t") else None,
            )
        # Fall back to original implementation
        return original_topk_generate(hidden_states, input_ids, head, logits_processor)
    
    # Apply patches
    model.base_model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: patched_prepare_decoder_attention_mask(
        model.base_model.model, *args, **kwargs
    )
    
    model.ea_layer.topK_genrate = lambda *args, **kwargs: patched_topk_generate(*args, **kwargs)
    
    # Replace utility functions
    import eagle.model.utils as utils
    
    # Store original functions
    utils.original_tree_decoding = utils.tree_decoding
    utils.original_evaluate_posterior = utils.evaluate_posterior
    utils.original_update_inference_inputs = utils.update_inference_inputs
    
    # Patch functions
    def patched_tree_decoding(model, *args, **kwargs):
        if args[1].is_cuda:  # tree_candidates
            return triton_kernels.tree_decoding(model, *args, **kwargs)
        return utils.original_tree_decoding(model, *args, **kwargs)
    
    def patched_evaluate_posterior(logits, candidates, logits_processor):
        if logits.is_cuda:
            return triton_kernels.evaluate_posterior(logits, candidates, logits_processor)
        return utils.original_evaluate_posterior(logits, candidates, logits_processor)
    
    def patched_update_inference_inputs(*args, **kwargs):
        if args[0].is_cuda:  # input_ids
            return triton_kernels.update_inference_inputs(*args, **kwargs)
        return utils.original_update_inference_inputs(*args, **kwargs)
    
    # Apply patches
    utils.tree_decoding = patched_tree_decoding
    utils.evaluate_posterior = patched_evaluate_posterior
    utils.update_inference_inputs = patched_update_inference_inputs
    
    return model


def unpatch_eagle_model(model):
    """
    Remove Triton kernel patches from an EAGLE model.
    
    Args:
        model: The patched EAGLE model
    """
    import eagle.model.utils as utils
    
    # Restore original functions if they exist
    if hasattr(utils, "original_tree_decoding"):
        utils.tree_decoding = utils.original_tree_decoding
        delattr(utils, "original_tree_decoding")
    
    if hasattr(utils, "original_evaluate_posterior"):
        utils.evaluate_posterior = utils.original_evaluate_posterior
        delattr(utils, "original_evaluate_posterior")
    
    if hasattr(utils, "original_update_inference_inputs"):
        utils.update_inference_inputs = utils.original_update_inference_inputs
        delattr(utils, "original_update_inference_inputs")
    
    # Restore original model methods
    model.base_model.model._prepare_decoder_attention_mask = model.base_model.model.__class__._prepare_decoder_attention_mask.__get__(
        model.base_model.model, model.base_model.model.__class__
    )
    
    # Restore original ea_layer.topK_generate
    from eagle.model.cnets import Model
    model.ea_layer.topK_genrate = Model.topK_genrate.__get__(model.ea_layer, Model)
    
    return model