import torch
import math

def triton_attention(q, k, v, scale=None):
    """
    Compute attention using PyTorch as a fallback.
    
    Parameters:
        q: query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        k: key tensor of shape [batch_size, num_kv_heads, seq_len_kv, head_dim]
        v: value tensor of shape [batch_size, num_kv_heads, seq_len_kv, head_dim]
        scale: scaling factor for attention scores
        
    Returns:
        output tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, num_kv_heads, seq_len_kv, _ = k.shape
    
    # Handle grouped query attention (e.g., for LLaMA)
    num_kv_groups = num_heads // num_kv_heads
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
    
    # Set scale factor for softmax
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, device=q.device), diagonal=1).bool()
    attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # Compute output
    output = torch.matmul(attn_weights, v)
    
    return output


