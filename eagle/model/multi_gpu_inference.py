import torch
import time
import argparse
import os
import json
from transformers import AutoTokenizer

from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor

class MultiGpuEaModel(EaModel):
    """
    Extension of EaModel that supports placing the draft model on a separate GPU.
    """
    
    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            draft_device=None,  # Device for the draft model
    ):
        super().__init__(
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
        )
        
        # Set up draft model device
        self.base_device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        self.draft_device = draft_device if draft_device is not None else self.base_device
        self.cross_device = (self.draft_device != self.base_device)
        
        # Move draft model to specified device
        if self.cross_device:
            print(f"Moving draft model from {self.base_device} to {self.draft_device}")
            self.ea_layer.to(self.draft_device)
            
            # Set up cross-device communication flags
            self.ea_layer.diff_device = True
            if hasattr(self.ea_layer, "headweight"):
                self.ea_layer.headweight = self.base_model.lm_head.weight.clone().to(self.draft_device)
            else:
                self.ea_layer.layer_device = self.draft_device
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        # Handle cross-device communication if needed
        if self.cross_device and self.use_eagle3:
            # Move hidden states to draft model device for processing
            if output_orig:
                return outputs, orig, hidden_states
            else:
                return outputs, hidden_states
        else:
            if output_orig:
                return outputs, orig, hidden_states
            else:
                return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
            
        # Create CUDA streams for overlapping operations if using multiple GPUs
        if self.cross_device:
            base_stream = torch.cuda.Stream(device=self.base_device)
            draft_stream = torch.cuda.Stream(device=self.draft_device)
        
        # Avoid modifying the input_ids in-place
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # Prefill
        if self.cross_device:
            with torch.cuda.stream(base_stream):
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = self._initialize_tree_multi_gpu(
                    input_ids, past_key_values, logits_processor
                )
        else:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor
            )
        
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        
        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask

            if self.cross_device:
                # Move draft tokens to base model device
                draft_tokens_base = draft_tokens.to(self.base_device)
                
                # Target model forward, get logits
                logits, hidden_state_new, outputs = self._tree_decoding_multi_gpu(
                    draft_tokens_base,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                )
            else:
                draft_tokens = draft_tokens.to(input_ids.device)
                logits, hidden_state_new, outputs = tree_decoding(
                    self,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                )
            
            # Prepare candidates
            draft_tokens_padded = torch.cat((draft_tokens.to(self.base_device), padding), dim=1)
            candidates = draft_tokens_padded[0, retrieve_indices.to(self.base_device)]
            
            # Verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
        
            # Adjusting the input sequence, draft model forward
            if self.cross_device:
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = self._update_inference_inputs_multi_gpu(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    hidden_state_new,
                    sample_p
                )
            else:
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    hidden_state_new,
                    sample_p
                )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
                
        # Synchronize streams before returning if using multiple GPUs
        if self.cross_device:
            torch.cuda.synchronize(self.base_device)
            torch.cuda.synchronize(self.draft_device)
            
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx
    
    def _initialize_tree_multi_gpu(self, input_ids, past_key_values, logits_processor):
        """Multi-GPU version of initialize_tree function"""
        outputs, orig, hidden_states = self(
            input_ids, past_key_values=past_key_values, output_orig=True
        )

        if logits_processor is not None:
            logits = orig[:, -1]
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(orig[:, -1])
            token = token[None, None]
        
        input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

        # Handle cross-device operations for EAGLE3
        if self.use_eagle3:
            if outputs["hidden_states"][0].device != self.draft_device:
                # Move hidden states to draft model device
                hidden_states_draft = [x.to(self.draft_device) for x in outputs["hidden_states"]]
                hidden_states = torch.cat(hidden_states_draft, dim=-1)
            else:
                hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
        
        # Generate draft tokens on the draft model device
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
            hidden_states, 
            input_ids.to(self.draft_device), 
            self.base_model.lm_head.to(self.draft_device) if self.cross_device else self.base_model.lm_head,
            logits_processor
        )
        
        # Move results back to base model device if needed
        if self.cross_device:
            tree_mask = tree_mask.to(self.base_device)
            tree_position_ids = tree_position_ids.to(self.base_device)
            retrieve_indices = retrieve_indices.to(self.base_device)
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token
    
    def _tree_decoding_multi_gpu(
            self,
            tree_candidates,
            past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices,
    ):
        """Multi-GPU version of tree_decoding function"""
        position_ids = tree_position_ids + input_ids.shape[1]
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        
        outputs, tree_logits, hidden_state = self(
            tree_candidates,
            output_orig=True,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        if self.use_eagle3:
            if outputs["hidden_states"][0].device != self.draft_device:
                # Move hidden states to draft model device for processing
                hidden_states_draft = [x.to(self.draft_device) for x in outputs["hidden_states"]]
                hidden_state = torch.cat(hidden_states_draft, dim=-1)
            else:
                hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

        # Get logits for the retrieve indices
        logits = tree_logits[0, retrieve_indices]
        
        return logits, hidden_state, outputs
    
    def _update_inference_inputs_multi_gpu(
            self,
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            past_key_values_data_list,
            current_length_data,
            hidden_state_new,
            sample_p
    ):
        """Multi-GPU version of update_inference_inputs function"""
        prev_input_len = input_ids.shape[1]
        # Map the best candidate indices to the original indices in the sequence
        select_indices = (
                retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
        )
        # Append the tokens from the best candidate to the input sequence
        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
        )
        
        # Update the past key values based on the selected tokens
        for past_key_values_data in past_key_values_data_list:
            tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
            # Destination tensor where the relevant past information will be stored
            dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
            # Copy relevant past information from the source to the destination
            dst.copy_(tgt, non_blocking=True)

        # Update the current length tensor
        current_length_data.fill_(prev_input_len + tgt.shape[-2])

        # Move hidden states to draft model device
        retrieve_hidden_state_new = hidden_state_new.to(self.draft_device)
        retrieve_hidden_state_new = retrieve_hidden_state_new[:, retrieve_indices.to(self.draft_device)]
        
        accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
        
        # Sample next token
        prob = sample_p
        if logits_processor is not None:
            token = torch.multinomial(prob, 1)
            token = token[None]
        else:
            token = torch.argmax(prob)
            token = token[None, None]
        
        # Move token to input_ids device
        token = token.to(input_ids.device)
        
        # Generate new draft tokens
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((input_ids, token), dim=1).to(self.draft_device),
            head=self.base_model.lm_head.to(self.draft_device) if self.cross_device else self.base_model.lm_head,
            logits_processor=logits_processor
        )

        # Move results back to base model device if needed
        if self.cross_device:
            tree_mask = tree_mask.to(self.base_device)
            tree_position_ids = tree_position_ids.to(self.base_device)
            retrieve_indices = retrieve_indices.to(self.base_device)

        new_token += accept_length + 1

        return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, None, token

@torch.no_grad()
def load_model_multi_gpu(
    base_model_path,
    ea_model_path,
    use_eagle3=True,
    use_multi_gpu=False,
    base_device="cuda:0",
    draft_device="cuda:1",
    **kwargs
):
    """
    Load the EAGLE model with optional multi-GPU support.
    
    Args:
        base_model_path: Path to the base model
        ea_model_path: Path to the EAGLE model
        use_eagle3: Whether to use EAGLE3 (True) or EAGLE2 (False)
        use_multi_gpu: Whether to use multiple GPUs
        base_device: Device for the base model
        draft_device: Device for the draft model
        **kwargs: Additional arguments for model loading
    
    Returns:
        An EaModel or MultiGpuEaModel instance
    """
    from transformers import AutoConfig
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
    from eagle.model.modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
    from eagle.model.modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
    
    # Determine model type
    Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
    
    # Load base model on specified device
    if Type == 'LlamaForCausalLM':
        base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path, device_map=base_device, **kwargs
        )
    elif Type == 'Qwen2ForCausalLM':
        base_model = KVQwen2ForCausalLM.from_pretrained(
            base_model_path, device_map=base_device, **kwargs
        )
    else:
        base_model = KVMixtralForCausalLM.from_pretrained(
            base_model_path, device_map=base_device, **kwargs
        )
    
    # Load draft model configuration
    configpath = os.path.join(ea_model_path, "config.json")
    if not os.path.exists(configpath):
        configpath = hf_hub_download(ea_model_path, "config.json")
    
    # Load draft model weights
    try:
        load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path, map_location="cpu")
    except:
        from safetensors.torch import load_file
        load_model_path = os.path.join(ea_model_path, "model.safetensors")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
        ea_layer_state_dict = load_file(load_model_path)
    
    # Extract model parameters from kwargs
    total_token = kwargs.pop("total_token", 60)
    depth = kwargs.pop("depth", 7)
    top_k = kwargs.pop("top_k", 10)
    threshold = kwargs.pop("threshold", 1.0)
    
    # Create model with specified device placement
    if use_multi_gpu:
        model = MultiGpuEaModel(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            draft_device=draft_device
        )
    else:
        model = EaModel(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )
    
    return model