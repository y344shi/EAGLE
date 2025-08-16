"""
Patch for ea_model.py to use Triton kernels.

This module provides functions to patch the EAGLE model to use Triton kernels
for improved performance. It safely binds patched methods and falls back to
original implementations when Triton/CUDA is not available.
"""

from __future__ import annotations

import types
import torch
from typing import Any, Tuple

from .integration import triton_kernels
from .timers import time_block, inc
import eagle.model.utils as utils


def patch_eagle_model(model):
    """
    Patch an EAGLE model to use Triton kernels.

    Args:
        model: The EAGLE model instance to patch
    """
    # 1) Patch LlamaModel._prepare_decoder_attention_mask at the instance level
    inst = model.base_model.model
    original_prepare_unbound = inst.__class__._prepare_decoder_attention_mask

    def _patched_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Use Triton version only when CUDA and tree_mask exists
        try:
            if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.is_cuda and getattr(self, "tree_mask", None) is not None:
                inc("mask_prepare_triton")
                with time_block("mask_prepare_triton"):
                    return triton_kernels.prepare_decoder_attention_mask(
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        past_kv_len=past_key_values_length,
                        tree_mask=self.tree_mask,
                    )
        except Exception:
            # Fall-through to original on any error
            pass
        # Call original unbound method with explicit self
        return original_prepare_unbound(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

    inst._prepare_decoder_attention_mask = types.MethodType(_patched_prepare_decoder_attention_mask, inst)

    # 2) Patch ea_layer.topK_genrate
    ea = model.ea_layer
    original_topk_unbound = ea.__class__.topK_genrate

    def _patched_topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        try:
            if isinstance(hidden_states, torch.Tensor) and hidden_states.is_cuda:
                inc("topk_generate_triton")
                with time_block("topk_generate_triton"):
                    return triton_kernels.topk_generate(
                        hidden_states=hidden_states,
                        input_ids=input_ids,
                        head=head,
                        logits_processor=logits_processor,
                        total_tokens=getattr(self, "total_tokens", None),
                        depth=getattr(self, "depth", None),
                        top_k=getattr(self, "top_k", None),
                        threshold=getattr(self, "threshold", None),
                        stable_kv=getattr(self, "stable_kv", None),
                        d2t=getattr(self, "d2t", None),
                    )
        except Exception:
            pass
        return original_topk_unbound(self, hidden_states, input_ids, head, logits_processor)

    ea.topK_genrate = types.MethodType(_patched_topK_genrate, ea)

    # 3) Patch utils functions (module-level)
    if not hasattr(utils, "original_tree_decoding"):
        utils.original_tree_decoding = utils.tree_decoding
    if not hasattr(utils, "original_evaluate_posterior"):
        utils.original_evaluate_posterior = utils.evaluate_posterior
    if not hasattr(utils, "original_update_inference_inputs"):
        utils.original_update_inference_inputs = utils.update_inference_inputs

    def _patched_tree_decoding(model_obj, tree_candidates, past_kv, tree_position_ids, input_ids, retrieve_indices):
        try:
            if isinstance(tree_candidates, torch.Tensor) and tree_candidates.is_cuda:
                inc("tree_decoding_triton")
                with time_block("tree_decoding_triton"):
                    return triton_kernels.tree_decoding(
                        model_obj, tree_candidates, past_kv, tree_position_ids, input_ids, retrieve_indices
                    )
        except Exception:
            pass
        return utils.original_tree_decoding(model_obj, tree_candidates, past_kv, tree_position_ids, input_ids, retrieve_indices)

    def _patched_evaluate_posterior(logits, candidates, logits_processor):
        try:
            if isinstance(logits, torch.Tensor) and logits.is_cuda:
                inc("evaluate_posterior_triton")
                with time_block("evaluate_posterior_triton"):
                    return triton_kernels.evaluate_posterior(logits, candidates, logits_processor)
        except Exception:
            pass
        return utils.original_evaluate_posterior(logits, candidates, logits_processor)

    def _patched_update_inference_inputs(
        input_ids: torch.Tensor,
        candidates: torch.Tensor,
        best_candidate: torch.Tensor,
        accept_length: torch.Tensor,
        retrieve_indices: torch.Tensor,
        logits_processor,
        new_token: int,
        past_key_values_data_list,
        current_length_data: torch.Tensor,
        model_obj,
        hidden_state_new: torch.Tensor,
        sample_p: torch.Tensor,
    ) -> Tuple[Any, ...]:
        try:
            if isinstance(input_ids, torch.Tensor) and input_ids.is_cuda:
                inc("update_inference_inputs_triton")
                with time_block("update_inference_inputs_triton"):
                    return triton_kernels.update_inference_inputs(
                        input_ids=input_ids,
                        candidates=candidates,
                        best_candidate=best_candidate,
                        accept_length=accept_length,
                        retrieve_indices=retrieve_indices,
                        logits_processor=logits_processor,
                        new_token=new_token,
                        past_key_values_data_list=past_key_values_data_list,
                        current_length_data=current_length_data,
                        model=model_obj,
                        hidden_state_new=hidden_state_new,
                        sample_p=sample_p,
                    )
        except Exception:
            pass
        return utils.original_update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            past_key_values_data_list,
            current_length_data,
            model_obj,
            hidden_state_new,
            sample_p,
        )

    utils.tree_decoding = _patched_tree_decoding
    utils.evaluate_posterior = _patched_evaluate_posterior
    utils.update_inference_inputs = _patched_update_inference_inputs

    return model


def unpatch_eagle_model(model):
    """
    Remove Triton kernel patches from an EAGLE model.

    Args:
        model: The patched EAGLE model
    """
    inst = model.base_model.model
    inst._prepare_decoder_attention_mask = types.MethodType(
        inst.__class__._prepare_decoder_attention_mask, inst
    )

    from eagle.model.cnets import Model as _EA_Model
    model.ea_layer.topK_genrate = types.MethodType(_EA_Model.topK_genrate, model.ea_layer)

    if hasattr(utils, "original_tree_decoding"):
        utils.tree_decoding = utils.original_tree_decoding
        delattr(utils, "original_tree_decoding")

    if hasattr(utils, "original_evaluate_posterior"):
        utils.evaluate_posterior = utils.original_evaluate_posterior
        delattr(utils, "original_evaluate_posterior")

    if hasattr(utils, "original_update_inference_inputs"):
        utils.update_inference_inputs = utils.original_update_inference_inputs
        delattr(utils, "original_update_inference_inputs")

    return model