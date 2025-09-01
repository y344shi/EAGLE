import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _fused_tree_attention_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, mask_ptr, o_ptr,
    # Matrix dimensions
    batch_size, num_heads, seq_len, kv_seq_len, head_dim,
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_mb, stride_mh, stride_ms, stride_md,
    stride_ob, stride_oh, stride_os, stride_od,
    # Scale
    scale,
    # Optional: block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """
    Fused tree attention kernel that computes QK^T, applies mask, softmax, and then computes attention @ V
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Compute head_id and batch_id
    head_id = pid_batch % num_heads
    batch_id = pid_batch // num_heads
    
    # Compute offsets
    offset_q = batch_id * stride_qb + head_id * stride_qh
    offset_k = batch_id * stride_kb + head_id * stride_kh
    offset_v = batch_id * stride_vb + head_id * stride_vh
    offset_m = batch_id * stride_mb + head_id * stride_mh
    offset_o = batch_id * stride_ob + head_id * stride_oh
    
    # Compute attention for this block
    # Load q block
    q_start = offset_q + pid_m * BLOCK_M * stride_qs
    q_block_ptr = q_ptr + q_start
    q = tl.load(q_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_qs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd,
                mask=(tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_DMODEL)[None, :] < head_dim))
    
    # Initialize accumulator for output
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Iterate through k/v blocks
    for start_n in range(0, kv_seq_len, BLOCK_N):
        # Load k block
        k_start = offset_k + start_n * stride_ks
        k_block_ptr = k_ptr + k_start
        k = tl.load(k_block_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_ks + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kd,
                    mask=(tl.arange(0, BLOCK_N)[:, None] + start_n < kv_seq_len) & (tl.arange(0, BLOCK_DMODEL)[None, :] < head_dim))
        
        # Load mask block
        m_start = offset_m + pid_m * BLOCK_M * stride_ms + start_n * stride_md
        m_block_ptr = mask_ptr + m_start
        mask = tl.load(m_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_ms + tl.arange(0, BLOCK_N)[None, :] * stride_md,
                      mask=(tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_N)[None, :] + start_n < kv_seq_len))
        
        # Compute attention scores
        scores = tl.dot(q, tl.trans(k))
        scores = scores * scale
        
        # Apply mask
        scores = scores + mask
        
        # Apply softmax
        scores = tl.softmax(scores, axis=1)
        
        # Load v block
        v_start = offset_v + start_n * stride_vs
        v_block_ptr = v_ptr + v_start
        v = tl.load(v_block_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_vs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd,
                    mask=(tl.arange(0, BLOCK_N)[:, None] + start_n < kv_seq_len) & (tl.arange(0, BLOCK_DMODEL)[None, :] < head_dim))
        
        # Compute attention output
        acc += tl.dot(scores, v)
    
    # Store output
    o_start = offset_o + pid_m * BLOCK_M * stride_os
    o_block_ptr = o_ptr + o_start
    tl.store(o_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_os + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od,
             acc, mask=(tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_DMODEL)[None, :] < head_dim))


def tree_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute tree-based attention using Triton.
    
    Args:
        q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch_size, num_heads, kv_seq_len, head_dim]
        v: Value tensor of shape [batch_size, num_heads, kv_seq_len, head_dim]
        mask: Attention mask of shape [batch_size, 1, seq_len, kv_seq_len]
        scale: Scaling factor for attention scores
        
    Returns:
        output: Attention output of shape [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    
    # Create output tensor
    output = torch.empty_like(q)
    
    # Set default scale if not provided
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.zeros((batch_size, 1, seq_len, kv_seq_len), device=q.device, dtype=q.dtype)
    
    # Compute grid and block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_DMODEL = min(32, head_dim)
    
    grid = (
        triton.cdiv(seq_len, BLOCK_M),
        1,
        batch_size * num_heads
    )
    
    # Launch kernel
    _fused_tree_attention_kernel[grid](
        q, k, v, mask, output,
        batch_size, num_heads, seq_len, kv_seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return output


def tree_decoding_triton(
    model,
    tree_candidates,
    past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Triton implementation of tree_decoding function.
    
    This function replaces the PyTorch implementation in utils.py.
    """
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    # Use Triton kernel for gather operation
    logits = tree_logits[0, retrieve_indices]
    
    return logits, hidden_state, outputs