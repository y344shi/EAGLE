import torch
import triton
import triton.language as tl
import math

@triton.jit
def _apply_rotary_emb_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, position_ids_ptr,
    q_out_ptr, k_out_ptr,
    batch_size, num_heads, seq_len, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_cb, stride_cs, stride_cd,
    stride_sb, stride_ss, stride_sd,
    stride_pb, stride_ps,
    stride_qob, stride_qoh, stride_qos, stride_qod,
    stride_kob, stride_koh, stride_kos, stride_kod,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Parameters:
        q_ptr: pointer to query tensor
        k_ptr: pointer to key tensor
        cos_ptr: pointer to cosine tensor
        sin_ptr: pointer to sine tensor
        position_ids_ptr: pointer to position IDs tensor
        q_out_ptr: pointer to output query tensor
        k_out_ptr: pointer to output key tensor
        batch_size: number of sequences in the batch
        num_heads: number of attention heads
        seq_len: sequence length
        head_dim: dimension of each attention head
        stride_*: strides for the respective tensors
        BLOCK_SIZE: block size for tiling
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch, head, and sequence indices
    batch_idx = (pid // (num_heads * seq_len)) % batch_size
    head_idx = (pid // seq_len) % num_heads
    seq_idx = pid % seq_len
    
    # Get position ID for this sequence position
    pos_id = tl.load(position_ids_ptr + batch_idx * stride_pb + seq_idx * stride_ps)
    
    # Create offsets for loading vectors
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    
    # Load query and key vectors
    q_ptr_batch_head_seq = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + seq_idx * stride_qs
    k_ptr_batch_head_seq = k_ptr + batch_idx * stride_kb + head_idx * stride_kh + seq_idx * stride_ks
    
    q = tl.load(q_ptr_batch_head_seq + offsets * stride_qd, mask=mask, other=0.0)
    k = tl.load(k_ptr_batch_head_seq + offsets * stride_kd, mask=mask, other=0.0)
    
    # Load cos and sin for this position
    cos_ptr_pos = cos_ptr + pos_id * stride_cs
    sin_ptr_pos = sin_ptr + pos_id * stride_ss
    
    cos = tl.load(cos_ptr_pos + offsets * stride_cd, mask=mask, other=1.0)
    sin = tl.load(sin_ptr_pos + offsets * stride_sd, mask=mask, other=0.0)
    
    # Compute half indices for rotation
    half_dim = head_dim // 2
    half_mask = offsets < half_dim
    
    # Apply rotary embeddings
    # For the first half of dimensions
    q_first_half = tl.load(q_ptr_batch_head_seq + offsets * stride_qd, mask=half_mask, other=0.0)
    q_second_half = tl.load(q_ptr_batch_head_seq + (offsets + half_dim) * stride_qd, mask=half_mask, other=0.0)
    
    k_first_half = tl.load(k_ptr_batch_head_seq + offsets * stride_kd, mask=half_mask, other=0.0)
    k_second_half = tl.load(k_ptr_batch_head_seq + (offsets + half_dim) * stride_kd, mask=half_mask, other=0.0)
    
    # Load cos and sin for the first half
    cos_first_half = tl.load(cos_ptr_pos + offsets * stride_cd, mask=half_mask, other=1.0)
    sin_first_half = tl.load(sin_ptr_pos + offsets * stride_sd, mask=half_mask, other=0.0)
    
    # Compute rotated values
    q_rot_first = q_first_half * cos_first_half - q_second_half * sin_first_half
    q_rot_second = q_second_half * cos_first_half + q_first_half * sin_first_half
    
    k_rot_first = k_first_half * cos_first_half - k_second_half * sin_first_half
    k_rot_second = k_second_half * cos_first_half + k_first_half * sin_first_half
    
    # Store rotated query and key
    q_out_ptr_batch_head_seq = q_out_ptr + batch_idx * stride_qob + head_idx * stride_qoh + seq_idx * stride_qos
    k_out_ptr_batch_head_seq = k_out_ptr + batch_idx * stride_kob + head_idx * stride_koh + seq_idx * stride_kos
    
    # Store first half
    tl.store(q_out_ptr_batch_head_seq + offsets * stride_qod, q_rot_first, mask=half_mask)
    tl.store(q_out_ptr_batch_head_seq + (offsets + half_dim) * stride_qod, q_rot_second, mask=half_mask)
    
    tl.store(k_out_ptr_batch_head_seq + offsets * stride_kod, k_rot_first, mask=half_mask)
    tl.store(k_out_ptr_batch_head_seq + (offsets + half_dim) * stride_kod, k_rot_second, mask=half_mask)


def triton_apply_rotary_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors using Triton kernels.
    
    Parameters:
        q: query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k: key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        cos: cosine tensor of shape [seq_len, head_dim]
        sin: sine tensor of shape [seq_len, head_dim]
        position_ids: position IDs tensor of shape [batch_size, seq_len]
        
    Returns:
        q_embed: rotary embedded query tensor
        k_embed: rotary embedded key tensor
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create output tensors
    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)
    
    # Compute strides for tensors
    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k.stride()
    stride_cb, stride_cs, stride_cd = cos.stride()
    stride_sb, stride_ss, stride_sd = sin.stride()
    stride_pb, stride_ps = position_ids.stride()
    stride_qob, stride_qoh, stride_qos, stride_qod = q_embed.stride()
    stride_kob, stride_koh, stride_kos, stride_kod = k_embed.stride()
    
    # Determine block size
    BLOCK_SIZE = min(128, head_dim)
    
    # Launch kernel
    grid = (batch_size * num_heads * seq_len,)
    _apply_rotary_emb_kernel[grid](
        q, k, cos, sin, position_ids,
        q_embed, k_embed,
        batch_size, num_heads, seq_len, head_dim,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_cb, stride_cs, stride_cd,
        stride_sb, stride_ss, stride_sd,
        stride_pb, stride_ps,
        stride_qob, stride_qoh, stride_qos, stride_qod,
        stride_kob, stride_koh, stride_kos, stride_kod,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return q_embed, k_embed


# PyTorch fallback implementation for testing
def pytorch_apply_rotary_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings using PyTorch as a fallback.
    
    Parameters:
        q: query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k: key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        cos: cosine tensor of shape [seq_len, head_dim]
        sin: sine tensor of shape [seq_len, head_dim]
        position_ids: position IDs tensor of shape [batch_size, seq_len]
        
    Returns:
        q_embed: rotary embedded query tensor
        k_embed: rotary embedded key tensor
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create output tensors
    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)
    
    # Apply rotary embeddings
    for b in range(batch_size):
        for s in range(seq_len):
            # Get position ID for this sequence position
            pos_id = position_ids[b, s].item()
            
            # Get cos and sin for this position
            cos_pos = cos[pos_id]
            sin_pos = sin[pos_id]
            
            # Apply to all heads for this batch and sequence position
            for h in range(num_heads):
                q_vec = q[b, h, s]
                k_vec = k[b, h, s]
                
                # Split into two halves
                half_dim = head_dim // 2
                q_first_half = q_vec[:half_dim]
                q_second_half = q_vec[half_dim:]
                k_first_half = k_vec[:half_dim]
                k_second_half = k_vec[half_dim:]
                
                # Get cos and sin for the first half
                cos_first_half = cos_pos[:half_dim]
                sin_first_half = sin_pos[:half_dim]
                
                # Compute rotated values
                q_rot_first = q_first_half * cos_first_half - q_second_half * sin_first_half
                q_rot_second = q_second_half * cos_first_half + q_first_half * sin_first_half
                
                k_rot_first = k_first_half * cos_first_half - k_second_half * sin_first_half
                k_rot_second = k_second_half * cos_first_half + k_first_half * sin_first_half
                
                # Combine rotated values
                q_embed[b, h, s, :half_dim] = q_rot_first
                q_embed[b, h, s, half_dim:] = q_rot_second
                k_embed[b, h, s, :half_dim] = k_rot_first
                k_embed[b, h, s, half_dim:] = k_rot_second
    
    return q_embed, k_embed