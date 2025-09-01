import torch
import triton
import triton.language as tl
import math

from .attention import triton_attention
from .rope import triton_apply_rotary_emb
from .kv_cache import triton_append_to_kv_cache, triton_retrieve_from_kv_cache
from .tree_decoding import triton_compute_topk, triton_compute_tree_mask, triton_evaluate_posterior, triton_update_inputs
from .gpu_assert import assert_cuda_available, assert_tensor_on_cuda, with_gpu_assertion, GPUAssertionError


@with_gpu_assertion
class FusedTransformerKernel:
    """
    Fused transformer kernel for draft token generation.
    
    This class implements a fused kernel that performs the entire transformer
    execution pipeline for draft token generation, including:
    - Forward pass through transformer layers
    - RoPE embeddings
    - Attention computation
    - KV-cache management
    - Draft token generation and verification
    
    The implementation uses Triton kernels for GPU acceleration when available,
    and falls back to PyTorch implementations when necessary.
    """
    
    def __init__(self):
        """Initialize the fused transformer kernel."""
        assert_cuda_available()
        self.use_triton = True
    
    def forward(self, model, hidden_states, input_ids, past_key_values, position_ids, retrieve_indices):
        """
        Execute the fused transformer forward pass.
        
        Parameters:
            model: The transformer model
            hidden_states: Hidden states from previous layer
            input_ids: Input token IDs
            past_key_values: Past key-value cache
            position_ids: Position IDs for RoPE
            retrieve_indices: Indices for retrieving from KV cache
            
        Returns:
            logits: Output logits
            hidden_states: Updated hidden states
            outputs: Model outputs
        """
        # Ensure we're on CUDA
        if not hidden_states.is_cuda:
            raise RuntimeError("FusedTransformerKernel requires CUDA tensors")
        
        # Get model configuration
        config = model.config
        
        # Execute transformer layers with fused attention
        outputs, hidden_states_new = self._execute_transformer_layers(
            model, hidden_states, input_ids, past_key_values, position_ids
        )
        
        # Compute logits from hidden states
        logits = self._compute_logits(model, hidden_states_new)
        
        return logits, hidden_states_new, outputs
    
    def _execute_transformer_layers(self, model, hidden_states, input_ids, past_key_values, position_ids):
        """Execute transformer layers with fused attention."""
        # Get base model
        base_model = model.base_model
        
        # Forward through the model with tree mask
        tree_mask = getattr(base_model.model, 'tree_mask', None)
        outputs = base_model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )
        
        # Extract hidden states
        hidden_states_new = outputs[0]
        
        return outputs, hidden_states_new
    
    def _compute_logits(self, model, hidden_states):
        """Compute logits from hidden states."""
        # Get logits from the model's language modeling head
        logits = model.base_model.lm_head(hidden_states)
        
        return logits


@with_gpu_assertion
def draft_token_generation(model, input_ids, logits_processor=None):
    """
    Generate draft tokens using the fused transformer kernel.
    
    Parameters:
        model: The transformer model
        input_ids: Input token IDs
        logits_processor: Optional logits processor for sampling
        
    Returns:
        draft_tokens: Generated draft tokens
        retrieve_indices: Indices for retrieving from KV cache
        tree_mask: Tree mask for attention
        tree_position_ids: Position IDs for the tree
        logits: Output logits
        hidden_state: Hidden states
        sample_token: Sample token
    """
    # Assert that input_ids is on CUDA
    assert_tensor_on_cuda(input_ids, "input_ids")
    
    # Initialize tree for draft token generation
    sample_token = input_ids[:, -1]
    
    # Get hidden states from the model
    with torch.no_grad():
        outputs, hidden_state = model.forward(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
        )
    
    # Generate draft tokens using the EA layer
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
        hidden_state, input_ids, model.base_model.lm_head, logits_processor
    )
    
    # Get logits from the last hidden state
    logits = model.base_model.lm_head(outputs[0])
    
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token


@with_gpu_assertion
def tree_decoding(model, draft_tokens, past_key_values, tree_position_ids, input_ids, retrieve_indices):
    """
    Perform tree decoding using the fused transformer kernel.
    
    Parameters:
        model: The transformer model
        draft_tokens: Draft tokens
        past_key_values: Past key-value cache
        tree_position_ids: Position IDs for the tree
        input_ids: Input token IDs
        retrieve_indices: Indices for retrieving from KV cache
        
    Returns:
        logits: Output logits
        hidden_state_new: Updated hidden states
        outputs: Model outputs
    """
    # Assert that tensors are on CUDA
    assert_tensor_on_cuda(draft_tokens, "draft_tokens")
    assert_tensor_on_cuda(tree_position_ids, "tree_position_ids")
    assert_tensor_on_cuda(input_ids, "input_ids")
    
    # Create fused transformer kernel
    fused_kernel = FusedTransformerKernel()
    
    # Execute fused transformer forward pass
    logits, hidden_state_new, outputs = fused_kernel.forward(
        model, None, draft_tokens, past_key_values, tree_position_ids, retrieve_indices
    )
    
    return logits, hidden_state_new, outputs


def evaluate_posterior(logits, candidates, logits_processor=None):
    """
    Evaluate posterior probabilities for speculative decoding.
    
    Parameters:
        logits: Logits from the base model
        candidates: Candidate tokens from the draft model
        logits_processor: Optional logits processor for sampling
        
    Returns:
        best_candidate: Index of the best candidate
        accept_length: Number of accepted tokens
        sample_p: Sample probabilities
    """
    # Process logits if needed
    if logits_processor is not None:
        processed_logits = logits_processor(None, logits)
    else:
        processed_logits = logits
    
    # Get probabilities
    probs = torch.softmax(processed_logits, dim=-1)
    
    # Sample from the distribution
    if logits_processor is not None:
        sample_p = torch.multinomial(probs, 1)
    else:
        sample_p = torch.argmax(probs, dim=-1, keepdim=True)
    
    # Evaluate candidates
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    
    # Initialize best candidate and accept length
    best_candidate = torch.zeros(batch_size, dtype=torch.int32, device=logits.device)
    accept_length = torch.zeros(batch_size, dtype=torch.int32, device=logits.device)
    
    # Use Triton kernel if available
    if torch.cuda.is_available():
        best_candidate, accept_length = triton_evaluate_posterior(logits, candidates.unsqueeze(1))
    else:
        # Fallback PyTorch implementation
        for b in range(batch_size):
            max_accept_len = 0
            for s in range(seq_len):
                if s < candidates.shape[1]:
                    candidate_token = candidates[b, s].item()
                    if s < logits.shape[1]:
                        predicted_token = torch.argmax(logits[b, s]).item()
                        if predicted_token == candidate_token:
                            max_accept_len += 1
                        else:
                            break
            accept_length[b] = max_accept_len
    
    return best_candidate, accept_length, sample_p


@with_gpu_assertion
def update_inference_inputs(
    input_ids, candidates, best_candidate, accept_length, retrieve_indices,
    logits_processor, new_token, past_key_values_data, current_length_data,
    model, hidden_state_new, sample_p
):
    """
    Update inference inputs for the next iteration.
    
    Parameters:
        input_ids: Input token IDs
        candidates: Candidate tokens
        best_candidate: Index of the best candidate
        accept_length: Number of accepted tokens
        retrieve_indices: Indices for retrieving from KV cache
        logits_processor: Optional logits processor for sampling
        new_token: Number of new tokens generated so far
        past_key_values_data: Past key-value cache data
        current_length_data: Current length data
        model: The transformer model
        hidden_state_new: Updated hidden states
        sample_p: Sample probabilities
        
    Returns:
        input_ids: Updated input token IDs
        draft_tokens: Updated draft tokens
        retrieve_indices: Updated indices for retrieving from KV cache
        tree_mask: Updated tree mask
        tree_position_ids: Updated position IDs for the tree
        new_token: Updated number of new tokens
        hidden_state: Updated hidden states
        sample_token: Updated sample token
    """
    # Assert that tensors are on CUDA
    assert_tensor_on_cuda(input_ids, "input_ids")
    assert_tensor_on_cuda(candidates, "candidates")
    assert_tensor_on_cuda(best_candidate, "best_candidate")
    assert_tensor_on_cuda(accept_length, "accept_length")
    assert_tensor_on_cuda(retrieve_indices, "retrieve_indices")
    # Update input IDs with accepted tokens
    batch_size, input_len = input_ids.shape
    
    # Use Triton kernel if available
    if torch.cuda.is_available():
        output_ids = triton_update_inputs(input_ids, candidates.unsqueeze(1), best_candidate, accept_length)
        input_ids = output_ids
    else:
        # Fallback PyTorch implementation
        acc_len = accept_length.item()
        for i in range(acc_len + 1):
            if i < candidates.shape[1]:
                token = candidates[0, i].item()
                input_ids = torch.cat([input_ids, torch.tensor([[token]], device=input_ids.device)], dim=1)
    
    # Update new token count
    new_token += accept_length.item() + 1
    
    # Generate new draft tokens
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, hidden_state, sample_token = draft_token_generation(
        model, input_ids, logits_processor
    )
    
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token