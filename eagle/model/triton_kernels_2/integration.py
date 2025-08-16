import torch
from typing import List, Tuple, Optional

from .tree_attention import tree_attention, tree_decoding_triton
from .topk_expand import topk_expand_triton, topk_generate_triton
from .posterior_eval import evaluate_posterior_triton
from .kv_block_copy import kv_block_copy_triton, update_inference_inputs_triton
from .mask_preparation import prepare_decoder_attention_mask_triton


class TritonEagleKernels:
    """
    Integration class for all Triton kernels used in EAGLE model.
    """
    
    @staticmethod
    def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
    ):
        """
        Triton implementation of tree_decoding function.
        """
        return tree_decoding_triton(
            model,
            tree_candidates,
            past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices,
        )
    
    @staticmethod
    def topk_generate(
        hidden_states,
        input_ids,
        head,
        logits_processor,
        total_tokens,
        depth,
        top_k,
        threshold,
        stable_kv=None,
        d2t=None,
    ):
        """
        Triton implementation of topK_generate function.
        """
        return topk_generate_triton(
            hidden_states,
            input_ids,
            head,
            logits_processor,
            total_tokens,
            depth,
            top_k,
            threshold,
            stable_kv,
            d2t,
        )
    
    @staticmethod
    def evaluate_posterior(
        logits,
        candidates,
        logits_processor=None,
    ):
        """
        Triton implementation of evaluate_posterior function.
        """
        return evaluate_posterior_triton(
            logits,
            candidates,
            logits_processor,
        )
    
    @staticmethod
    def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p,
    ):
        """
        Triton implementation of update_inference_inputs function.
        """
        return update_inference_inputs_triton(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            past_key_values_data_list,
            current_length_data,
            model,
            hidden_state_new,
            sample_p,
        )
    
    @staticmethod
    def prepare_decoder_attention_mask(
        attention_mask,
        inputs_embeds,
        past_len,
        tree_mask=None,
    ):
        """
        Triton implementation of _prepare_decoder_attention_mask function.
        """
        return prepare_decoder_attention_mask_triton(
            attention_mask,
            inputs_embeds,
            past_len,
            tree_mask,
        )


# Create a singleton instance for easy access
triton_kernels = TritonEagleKernels()