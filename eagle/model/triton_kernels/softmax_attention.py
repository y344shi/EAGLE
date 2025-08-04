import torch
import triton
import triton.language as tl
import math

@triton.jit
def _softmax_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr,
    batch_size, num_heads, seq_len_q, seq_len_kv, head_dim, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_mb, stride_mh, stride_ms, stride_mt,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    Compute softmax attention in a single kernel.
    
    Parameters:
        q_ptr: pointer to query tensor
        k_ptr: pointer to key tensor
        v_ptr: pointer to value tensor
        o_ptr: pointer to output tensor
        mask_ptr: pointer to attention mask tensor
        batch_size: number of sequences in the batch
        num_heads: number of attention heads
        seq_len_q: sequence length of queries
        seq_len_kv: sequence length of keys/values
        head_dim: dimension of each attention head
        scale: scaling factor for attention scores
        stride_*: strides for the respective tensors
        BLOCK_SIZE: block size for tiling
        HAS_MASK: whether an attention mask is supplied
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch, head, and sequence indices
    batch_idx = (pid // (num_heads * seq_len_q)) % batch_size
    head_idx = (pid // seq_len_q) % num_heads
    seq_idx = pid % seq_len_q
    
    # Create offsets for loading vectors
    offsets_d = tl.arange(0, BLOCK_SIZE)
    mask_d = offsets_d < head_dim
    
    # Load query vector for current token
    q_ptr_batch_head_seq = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + seq_idx * stride_qs
    q_vec = tl.load(q_ptr_batch_head_seq + offsets_d * stride_qd, mask=mask_d, other=0.0)
    
    # Initialize accumulator and normalization factor
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    norm_factor = 0.0
    
    # Process each key-value pair
    for kv_idx in range(seq_len_kv):
        # Load key vector
        k_ptr_batch_head_seq = k_ptr + batch_idx * stride_kb + head_idx * stride_kh + kv_idx * stride_ks
        k_vec = tl.load(k_ptr_batch_head_seq + offsets_d * stride_kd, mask=mask_d, other=0.0)
        
        # Compute attention score using dot product
        score = tl.sum(q_vec * k_vec, axis=0) * scale
        
        # Apply attention mask if provided
        if HAS_MASK:
            mask_val = tl.load(
                mask_ptr
                + batch_idx * stride_mb
                + head_idx * stride_mh
                + seq_idx * stride_ms
                + kv_idx * stride_mt
            )
            score = score + mask_val
        
        # Apply causal mask
        if kv_idx > seq_idx:
            score = -float('inf')
        
        # Apply softmax (step 1): exponentiate
        score = tl.exp(score)
        norm_factor += score
        
        # Load value vector
        v_ptr_batch_head_seq = v_ptr + batch_idx * stride_vb + head_idx * stride_vh + kv_idx * stride_vs
        v_vec = tl.load(v_ptr_batch_head_seq + offsets_d * stride_vd, mask=mask_d, other=0.0)
        
        # Update accumulator with weighted value
        acc += score * v_vec
    
    # Apply softmax (step 2): normalize
    if norm_factor > 0.0:
        acc = acc / norm_factor
    
    # Store output
    o_ptr_batch_head_seq = o_ptr + batch_idx * stride_ob + head_idx * stride_oh + seq_idx * stride_os
    tl.store(o_ptr_batch_head_seq + offsets_d * stride_od, acc, mask=mask_d)


def triton_softmax_attention(q, k, v, mask=None, scale=None):
    """
    Compute softmax attention using Triton kernels.
    
    Parameters:
        q: query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        k: key tensor of shape [batch_size, num_heads, seq_len_kv, head_dim]
        v: value tensor of shape [batch_size, num_heads, seq_len_kv, head_dim]
        mask: attention mask tensor of shape [batch_size, num_heads, seq_len_q, seq_len_kv]
        scale: scaling factor for attention scores
        
    Returns:
        output tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_kv, _ = k.shape
    
    # Set scale factor for softmax
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Create output tensor
    output = torch.empty_like(q)
    
    # Compute strides for tensors
    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = v.stride()
    stride_ob, stride_oh, stride_os, stride_od = output.stride()
    
    if mask is not None:
        stride_mb, stride_mh, stride_ms, stride_mt = mask.stride()
    else:
        stride_mb = stride_mh = stride_ms = stride_mt = 0
    
    # Determine block size
    BLOCK_SIZE = min(128, head_dim)
    
    # Launch kernel
    grid = (batch_size * num_heads * seq_len_q,)
    _softmax_attention_kernel[grid](
        q,
        k,
        v,
        output,
        mask if mask is not None else q,
        batch_size,
        num_heads,
        seq_len_q,
        seq_len_kv,
        head_dim,
        scale,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        stride_mb,
        stride_mh,
        stride_ms,
        stride_mt,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_MASK=mask is not None,
    )
    
    return output


# PyTorch fallback implementation for testing
def pytorch_softmax_attention(q, k, v, mask=None, scale=None):
    """
    Compute softmax attention using PyTorch as a fallback.
    
    Parameters:
        q: query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        k: key tensor of shape [batch_size, num_heads, seq_len_kv, head_dim]
        v: value tensor of shape [batch_size, num_heads, seq_len_kv, head_dim]
        mask: attention mask tensor of shape [batch_size, num_heads, seq_len_q, seq_len_kv]
        scale: scaling factor for attention scores
        
    Returns:
        output tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
    """
    # Set scale factor for softmax
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    # Apply attention mask if provided
    if mask is not None:
        scores = scores + mask
    
    # Apply causal mask
    seq_len_q, seq_len_kv = scores.shape[-2], scores.shape[-1]
    causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, device=q.device), diagonal=1).bool()
    scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Compute output
    output = torch.matmul(attn_weights, v)
    
    return output