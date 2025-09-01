import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _prepare_decoder_attention_mask_kernel(
    # Pointers to matrices
    attention_mask_ptr, tree_mask_ptr, output_mask_ptr,
    # Dimensions
    batch_size, num_heads, seq_len, kv_seq_len, tree_size,
    # Strides
    stride_ab, stride_as, stride_ak,
    stride_tb, stride_th, stride_ts, stride_tk,
    stride_ob, stride_oh, stride_os, stride_ok,
    # Offsets
    tree_offset_s, tree_offset_k,
    # Constants
    min_value: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Kernel for preparing decoder attention mask with tree mask.
    """
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_block_m = tl.program_id(2)
    pid_block_n = tl.program_id(3)
    
    # Compute block start and end
    block_start_m = pid_block_m * BLOCK_SIZE_M
    block_end_m = min(block_start_m + BLOCK_SIZE_M, seq_len)
    
    block_start_n = pid_block_n * BLOCK_SIZE_N
    block_end_n = min(block_start_n + BLOCK_SIZE_N, kv_seq_len)
    
    # Skip if block is out of bounds
    if block_start_m >= seq_len or block_start_n >= kv_seq_len:
        return
    
    # Process each element in the block
    for m in range(block_start_m, block_end_m):
        for n in range(block_start_n, block_end_n):
            # Compute output offset
            output_offset = (
                pid_batch * stride_ob +
                pid_head * stride_oh +
                m * stride_os +
                n * stride_ok
            )
            
            # Initialize with attention mask value
            mask_value = tl.load(
                attention_mask_ptr +
                pid_batch * stride_ab +
                m * stride_as +
                n * stride_ak
            )
            
            # Apply tree mask if in tree region
            if m >= tree_offset_s and m < tree_offset_s + tree_size and n >= tree_offset_k and n < tree_offset_k + tree_size:
                tree_m = m - tree_offset_s
                tree_n = n - tree_offset_k
                
                tree_mask_value = tl.load(
                    tree_mask_ptr +
                    pid_batch * stride_tb +
                    pid_head * stride_th +
                    tree_m * stride_ts +
                    tree_n * stride_tk
                )
                
                # Apply tree mask: if tree_mask is 0, set to min_value
                if tree_mask_value == 0:
                    mask_value = min_value
            
            # Store result
            tl.store(output_mask_ptr + output_offset, mask_value)


def prepare_decoder_attention_mask_triton(
    attention_mask: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_len: int,
    tree_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Prepare decoder attention mask with tree mask using Triton.
    
    Args:
        attention_mask: Attention mask of shape [batch_size, seq_len]
        inputs_embeds: Input embeddings of shape [batch_size, seq_len, hidden_size]
        past_len: Length of past key values
        tree_mask: Optional tree mask of shape [batch_size, 1, tree_size, tree_size]
        
    Returns:
        Combined attention mask of shape [batch_size, 1, seq_len, seq_len + past_len]
    """
    batch_size, seq_length = attention_mask.shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    
    # Create causal mask
    causal_mask = torch.full(
        (seq_length, seq_length),
        torch.finfo(dtype).min,
        device=device
    )
    causal_mask_cond = torch.arange(seq_length, device=device)
    causal_mask.masked_fill_(causal_mask_cond < (causal_mask_cond + 1).view(seq_length, 1), 0)
    
    # Add past_len to causal mask if needed
    if past_len > 0:
        causal_mask = torch.cat(
            [torch.zeros(seq_length, past_len, dtype=dtype, device=device), causal_mask],
            dim=-1
        )
    
    # Expand causal mask
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length + past_len)
    
    # Expand attention mask
    expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length + past_len)
    expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
    
    # Combine masks
    combined_mask = causal_mask + expanded_mask
    
    # Apply tree mask if provided
    if tree_mask is not None:
        tree_size = tree_mask.shape[-1]
        
        # If not on CUDA, fall back to PyTorch implementation
        if not combined_mask.is_cuda:
            # Apply tree mask to the bottom-right corner of the combined mask
            combined_mask[:, :, -tree_size:, -tree_size:][tree_mask == 0] = torch.finfo(dtype).min
            return combined_mask
        
        # Compute strides
        stride_ab, stride_as, stride_ak = expanded_mask.stride()
        stride_tb, stride_th, stride_ts, stride_tk = tree_mask.stride()
        stride_ob, stride_oh, stride_os, stride_ok = combined_mask.stride()
        
        # Determine block sizes
        BLOCK_SIZE_M = min(32, seq_length)
        BLOCK_SIZE_N = min(32, seq_length + past_len)
        
        # Launch kernel
        grid = (batch_size, 1, triton.cdiv(seq_length, BLOCK_SIZE_M), triton.cdiv(seq_length + past_len, BLOCK_SIZE_N))
        _prepare_decoder_attention_mask_kernel[grid](
            expanded_mask, tree_mask, combined_mask,
            batch_size, 1, seq_length, seq_length + past_len, tree_size,
            stride_ab, stride_as, stride_ak,
            stride_tb, stride_th, stride_ts, stride_tk,
            stride_ob, stride_oh, stride_os, stride_ok,
            seq_length - tree_size, seq_length + past_len - tree_size,
            torch.finfo(dtype).min,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    
    return combined_mask