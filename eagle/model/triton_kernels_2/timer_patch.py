"""
Timer-only patch for EAGLE PyTorch paths.

This patch instruments the PyTorch (baseline) EAGLE functions with timing,
without changing their implementations. It lets us measure EAGLE-only time
on baseline and compare to Triton.

Patches:
- LlamaModel._prepare_decoder_attention_mask (timed when tree_mask exists)
- ea_layer.topK_genrate
- utils.tree_decoding
- utils.evaluate_posterior
- utils.update_inference_inputs
"""

from __future__ import annotations

import types
import torch
from typing import Any, Tuple

from .timers import time_block, inc
import eagle.model.utils as utils


def patch_eagle_timers_only(model):
    """
    Patch an EAGLE model to time PyTorch EAGLE portions only (no Triton).
    """
    # 1) Time LlamaModel._prepare_decoder_attention_mask
    inst = model.base_model.model
    original_prepare_unbound = inst.__class__._prepare_decoder_attention_mask

    def _timed_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        try:
            if getattr(self, "tree_mask", None) is not None:
                inc("mask_prepare_baseline")
                with time_block("mask_prepare_baseline"):
                    return original_prepare_unbound(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)
        except Exception:
            pass
        return original_prepare_unbound(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

    inst._prepare_decoder_attention_mask = types.MethodType(_timed_prepare_decoder_attention_mask, inst)

    # 2) Time ea_layer.topK_genrate
    ea = model.ea_layer
    original_topk_unbound = ea.__class__.topK_genrate

    def _timed_topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        inc("topk_generate_baseline")
        with time_block("topk_generate_baseline"):
            return original_topk_unbound(self, hidden_states, input_ids, head, logits_processor)

    ea.topK_genrate = types.MethodType(_timed_topK_genrate, ea)

    # 3) Time utils functions
    if not hasattr(utils, "baseline_original_tree_decoding"):
        utils.baseline_original_tree_decoding = utils.tree_decoding
    if not hasattr(utils, "baseline_original_evaluate_posterior"):
        utils.baseline_original_evaluate_posterior = utils.evaluate_posterior
    if not hasattr(utils, "baseline_original_update_inference_inputs"):
        utils.baseline_original_update_inference_inputs = utils.update_inference_inputs

    def _timed_tree_decoding(model_obj, tree_candidates, past_kv, tree_position_ids, input_ids, retrieve_indices):
        inc("tree_decoding_baseline")
        with time_block("tree_decoding_baseline"):
            return utils.baseline_original_tree_decoding(model_obj, tree_candidates, past_kv, tree_position_ids, input_ids, retrieve_indices)

    def _timed_evaluate_posterior(logits, candidates, logits_processor):
        inc("evaluate_posterior_baseline")
        with time_block("evaluate_posterior_baseline"):
            return utils.baseline_original_evaluate_posterior(logits, candidates, logits_processor)

    def _timed_update_inference_inputs(
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
    ):
        inc("update_inference_inputs_baseline")
        with time_block("update_inference_inputs_baseline"):
            return utils.baseline_original_update_inference_inputs(
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

    utils.tree_decoding = _timed_tree_decoding
    utils.evaluate_posterior = _timed_evaluate_posterior
    utils.update_inference_inputs = _timed_update_inference_inputs

    return model


def unpatch_eagle_timers_only(model):
    """
    Remove the timer-only patch and restore original PyTorch functions.
    """
    inst = model.base_model.model
    inst._prepare_decoder_attention_mask = types.MethodType(
        inst.__class__._prepare_decoder_attention_mask, inst
    )

    from eagle.model.cnets import Model as _EA_Model
    model.ea_layer.topK_genrate = types.MethodType(_EA_Model.topK_genrate, model.ea_layer)

    if hasattr(utils, "baseline_original_tree_decoding"):
        utils.tree_decoding = utils.baseline_original_tree_decoding
        delattr(utils, "baseline_original_tree_decoding")

    if hasattr(utils, "baseline_original_evaluate_posterior"):
        utils.evaluate_posterior = utils.baseline_original_evaluate_posterior
        delattr(utils, "baseline_original_evaluate_posterior")

    if hasattr(utils, "baseline_original_update_inference_inputs"):
        utils.update_inference_inputs = utils.baseline_original_update_inference_inputs
        delattr(utils, "baseline_original_update_inference_inputs")

    return model