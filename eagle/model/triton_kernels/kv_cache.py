import torch
import triton
import triton.language as tl

@triton.jit
def _append_to_kv_cache_kernel(
    k_cache, v_cache, k_new, v_new, current_length,
    stride_kcb, stride_kch, stride_kcn, stride_kck,
    stride_vcb, stride_vch, stride_vcn, stride_vck,
    stride_knb, stride_knh, stride_knn, stride_knk,
    stride_vnb, stride_vnh, stride_vnn, stride_vnk,
    batch_size, num_heads, new_seq_len, head_dim, max_seq_len,
    BLOCK_K: tl.constexpr,
):
    """
    Append new key-value pairs to the KV cache.
    
    Parameters:
        k_cache: key cache tensor
        v_cache: value cache tensor
        k_new: new key tensor to append
        v_new: new value tensor to append
        current_length: tensor containing current length of each sequence
        stride_*: strides for the respective tensors
        batch_size: number of sequences in the batch
        num_heads: number of attention heads
        new_seq_len: length of new sequence to append
        head_dim: dimension of each attention head
        max_seq_len: maximum sequence length supported by the cache
        BLOCK_K: block size for tiling along head dimension
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch, head, and sequence indices
    batch_idx = (pid // (num_heads * new_seq_len)) % batch_size
    head_idx = (pid // new_seq_len) % num_heads
    seq_idx = pid % new_seq_len
    
    # Get current length for this batch
    curr_len = tl.load(current_length + batch_idx)
    
    # Check if we're within bounds
    if curr_len + seq_idx >= max_seq_len:
        return
    
    # Compute pointers
    k_cache_ptr = k_cache + batch_idx * stride_kcb + head_idx * stride_kch + (curr_len + seq_idx) * stride_kcn
    v_cache_ptr = v_cache + batch_idx * stride_vcb + head_idx * stride_vch + (curr_len + seq_idx) * stride_vcn
    k_new_ptr = k_new + batch_idx * stride_knb + head_idx * stride_knh + seq_idx * stride_knn
    v_new_ptr = v_new + batch_idx * stride_vnb + head_idx * stride_vnh + seq_idx * stride_vnn
    
    # Copy values one by one
    for k_idx in range(0, head_dim):
        # Load new key and value
        k_val = tl.load(k_new_ptr + k_idx * stride_knk)
        v_val = tl.load(v_new_ptr + k_idx * stride_vnk)
        
        # Store to cache
        tl.store(k_cache_ptr + k_idx * stride_kck, k_val)
        tl.store(v_cache_ptr + k_idx * stride_vck, v_val)


@triton.jit
def _update_length_kernel(
    current_length, new_seq_len,
    batch_size,
    BLOCK: tl.constexpr,
):
    """
    Update the current length tensor after appending to the KV cache.
    
    Parameters:
        current_length: tensor containing current length of each sequence
        new_seq_len: length of new sequence appended
        batch_size: number of sequences in the batch
        BLOCK: block size for tiling
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch indices
    batch_start = pid * BLOCK
    batch_end = min(batch_start + BLOCK, batch_size)
    
    # Update lengths
    for b in range(batch_start, batch_end):
        curr_len = tl.load(current_length + b)
        tl.store(current_length + b, curr_len + new_seq_len)


def triton_append_to_kv_cache(k_cache, v_cache, k_new, v_new, current_length):
    """
    Append new key-value pairs to the KV cache using PyTorch as a fallback.
    
    Parameters:
        k_cache: key cache tensor of shape [batch_size, num_heads, max_seq_len, head_dim]
        v_cache: value cache tensor of shape [batch_size, num_heads, max_seq_len, head_dim]
        k_new: new key tensor to append of shape [batch_size, num_heads, new_seq_len, head_dim]
        v_new: new value tensor to append of shape [batch_size, num_heads, new_seq_len, head_dim]
        current_length: tensor containing current length of each sequence of shape [batch_size]
        
    Returns:
        None (updates k_cache, v_cache, and current_length in-place)
    """
    batch_size, num_heads, max_seq_len, head_dim = k_cache.shape
    _, _, new_seq_len, _ = k_new.shape
    
    # For each batch element
    for b in range(batch_size):
        curr_len = current_length[b].item()
        
        # Check if we have enough space in the cache
        if curr_len + new_seq_len > max_seq_len:
            # If not enough space, only copy what fits
            copy_len = max(0, max_seq_len - curr_len)
            if copy_len > 0:
                k_cache[b, :, curr_len:curr_len+copy_len, :] = k_new[b, :, :copy_len, :]
                v_cache[b, :, curr_len:curr_len+copy_len, :] = v_new[b, :, :copy_len, :]
                current_length[b] += copy_len
        else:
            # Copy new keys and values to cache
            k_cache[b, :, curr_len:curr_len+new_seq_len, :] = k_new[b, :, :, :]
            v_cache[b, :, curr_len:curr_len+new_seq_len, :] = v_new[b, :, :, :]
            current_length[b] += new_seq_len


@triton.jit
def _retrieve_from_kv_cache_kernel(
    k_cache, v_cache, k_out, v_out, indices,
    stride_kcb, stride_kch, stride_kcn, stride_kck,
    stride_vcb, stride_vch, stride_vcn, stride_vck,
    stride_kob, stride_koh, stride_kon, stride_kok,
    stride_vob, stride_voh, stride_von, stride_vok,
    stride_ib, stride_in,
    batch_size, num_heads, out_seq_len, head_dim,
    BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Retrieve key-value pairs from the KV cache based on indices.
    
    Parameters:
        k_cache: key cache tensor
        v_cache: value cache tensor
        k_out: output key tensor
        v_out: output value tensor
        indices: indices to retrieve from cache
        stride_*: strides for the respective tensors
        batch_size: number of sequences in the batch
        num_heads: number of attention heads
        out_seq_len: length of output sequence
        head_dim: dimension of each attention head
        BLOCK_*: block sizes for tiling
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch, head, and sequence indices
    batch_idx = (pid // (num_heads * out_seq_len)) % batch_size
    head_idx = (pid // out_seq_len) % num_heads
    seq_idx = pid % out_seq_len
    
    # Get index to retrieve
    idx = tl.load(indices + batch_idx * stride_ib + seq_idx * stride_in)
    
    # Compute pointers
    k_cache_ptr = k_cache + batch_idx * stride_kcb + head_idx * stride_kch + idx * stride_kcn
    v_cache_ptr = v_cache + batch_idx * stride_vcb + head_idx * stride_vch + idx * stride_vcn
    k_out_ptr = k_out + batch_idx * stride_kob + head_idx * stride_koh + seq_idx * stride_kon
    v_out_ptr = v_out + batch_idx * stride_vob + head_idx * stride_voh + seq_idx * stride_von
    
    # Load and store in blocks along the head dimension
    for k in range(0, head_dim, BLOCK_K):
        k_size = min(BLOCK_K, head_dim - k)
        offs_k = tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_size
        
        # Load from cache
        k_vals = tl.load(k_cache_ptr + offs_k * stride_kck, mask=mask_k)
        v_vals = tl.load(v_cache_ptr + offs_k * stride_vck, mask=mask_k)
        
        # Store to output
        tl.store(k_out_ptr + offs_k * stride_kok, k_vals, mask=mask_k)
        tl.store(v_out_ptr + offs_k * stride_vok, v_vals, mask=mask_k)


def triton_retrieve_from_kv_cache(k_cache, v_cache, indices):
    """
    Retrieve key-value pairs from the KV cache based on indices using PyTorch as a fallback.
    
    Parameters:
        k_cache: key cache tensor of shape [batch_size, num_heads, max_seq_len, head_dim]
        v_cache: value cache tensor of shape [batch_size, num_heads, max_seq_len, head_dim]
        indices: indices to retrieve from cache of shape [batch_size, out_seq_len]
        
    Returns:
        k_out: retrieved key tensor of shape [batch_size, num_heads, out_seq_len, head_dim]
        v_out: retrieved value tensor of shape [batch_size, num_heads, out_seq_len, head_dim]
    """
    batch_size, num_heads, max_seq_len, head_dim = k_cache.shape
    _, out_seq_len = indices.shape
    
    # Create output tensors
    k_out = torch.empty((batch_size, num_heads, out_seq_len, head_dim), 
                        dtype=k_cache.dtype, device=k_cache.device)
    v_out = torch.empty((batch_size, num_heads, out_seq_len, head_dim), 
                        dtype=v_cache.dtype, device=v_cache.device)
    
    # For each batch element
    for b in range(batch_size):
        for s in range(out_seq_len):
            # Get index to retrieve
            idx = indices[b, s].item()
            
            # Ensure index is valid
            if idx < 0 or idx >= max_seq_len:
                continue
                
            # Copy from cache to output
            k_out[b, :, s, :] = k_cache[b, :, idx, :]
            v_out[b, :, s, :] = v_cache[b, :, idx, :]
    
    return k_out, v_out
