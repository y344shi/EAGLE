import torch
import triton
import triton.language as tl
from typing import List, Optional


@triton.jit
def _kv_block_copy_kernel(
    # Pointers to matrices
    src_ptr, dst_ptr,
    # Dimensions
    batch_size, num_heads, src_seq_len, dst_seq_len, head_dim,
    # Source and destination indices
    src_start, dst_start, copy_len,
    # Strides
    stride_b, stride_h, stride_s, stride_d,
    # Block sizes
    BLOCK_SIZE_S: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
):
    """
    Kernel for copying blocks of KV cache.
    """
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_block = tl.program_id(2)
    
    # Compute offsets
    batch_offset = pid_batch * stride_b
    head_offset = pid_head * stride_h
    
    # Compute block start and end
    block_size = BLOCK_SIZE_D
    block_start = pid_block * block_size
    block_end = min(block_start + block_size, copy_len)
    
    # Skip if block is out of bounds
    if block_start >= copy_len:
        return
    
    # Copy each element in the block
    for i in range(block_start, block_end):
        src_seq_idx = src_start + i
        dst_seq_idx = dst_start + i
        
        # Skip if indices are out of bounds
        if src_seq_idx >= src_seq_len or dst_seq_idx >= dst_seq_len:
            continue
        
        src_offset = batch_offset + head_offset + src_seq_idx * stride_s
        dst_offset = batch_offset + head_offset + dst_seq_idx * stride_s
        
        # Copy values for this position
        for d in range(0, head_dim, BLOCK_SIZE_D):
            d_offsets = tl.arange(0, BLOCK_SIZE_D)
            d_mask = (d + d_offsets) < head_dim
            
            # Load from source
            values = tl.load(
                src_ptr + src_offset + (d + d_offsets) * stride_d,
                mask=d_mask,
                other=0.0
            )
            
            # Store to destination
            tl.store(
                dst_ptr + dst_offset + (d + d_offsets) * stride_d,
                values,
                mask=d_mask
            )


def kv_block_copy_triton(
    past_kv_data: torch.Tensor,
    select_indices: torch.Tensor,
    dst_range: Optional[torch.Tensor] = None,
) -> None:
    """
    Copy blocks of KV cache using Triton.
    
    Args:
        past_kv_data: KV cache tensor of shape [batch_size, num_heads, seq_len, head_dim]
        select_indices: Indices to select from source of shape [copy_len]
        dst_range: Optional destination range of shape [2] (start, length)
    """
    batch_size, num_heads, seq_len, head_dim = past_kv_data.shape
    copy_len = select_indices.shape[0]
    
    # Default destination range is at the end of the sequence
    if dst_range is None:
        dst_start = seq_len - copy_len
        dst_len = copy_len
    else:
        dst_start = dst_range[0].item()
        dst_len = dst_range[1].item()
    
    # If not on CUDA, fall back to PyTorch implementation
    if not past_kv_data.is_cuda:
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(copy_len):
                    src_idx = select_indices[i].item()
                    dst_idx = dst_start + i
                    
                    if src_idx >= 0 and src_idx < seq_len and dst_idx < seq_len:
                        past_kv_data[b, h, dst_idx] = past_kv_data[b, h, src_idx]
        return
    
    # Compute strides
    stride_b, stride_h, stride_s, stride_d = past_kv_data.stride()
    
    # Determine block sizes
    BLOCK_SIZE_S = min(32, copy_len)
    BLOCK_SIZE_D = min(32, head_dim)
    
    # Launch kernel
    grid = (batch_size, num_heads, triton.cdiv(copy_len, BLOCK_SIZE_S))
    _kv_block_copy_kernel[grid](
        past_kv_data, past_kv_data,
        batch_size, num_heads, seq_len, seq_len, head_dim,
        select_indices, dst_start, copy_len,
        stride_b, stride_h, stride_s, stride_d,
        BLOCK_SIZE_S=BLOCK_SIZE_S, BLOCK_SIZE_D=BLOCK_SIZE_D,
    )


def update_inference_inputs_triton(
    input_ids: torch.Tensor,
    candidates: torch.Tensor,
    best_candidate: torch.Tensor,
    accept_length: torch.Tensor,
    retrieve_indices: torch.Tensor,
    logits_processor,
    new_token: int,
    past_key_values_data_list: List[torch.Tensor],
    current_length_data: torch.Tensor,
    model,
    hidden_state_new: torch.Tensor,
    sample_p: torch.Tensor,
) -> tuple:
    """
    Triton implementation of update_inference_inputs function.
    
    This function replaces the PyTorch implementation in utils.py.
    """
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
        # Use Triton kernel for block copy
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        dst_range = torch.tensor([prev_input_len, tgt.shape[-2]], device=past_key_values_data.device)
        kv_block_copy_triton(past_key_values_data, select_indices.to(past_key_values_data.device), dst_range)
    
    # Update the current length tensor
    current_length_data.fill_(prev_input_len + select_indices.shape[0])
    
    # Get hidden states for the selected path
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    
    # Sample next token
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    
    # Generate next draft tokens
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
        accept_hidden_state_new,
        input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
        head=model.base_model.lm_head,
        logits_processor=logits_processor
    )
    
    new_token += accept_length + 1
    
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, None, token
