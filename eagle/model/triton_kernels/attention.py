import torch

from .softmax_attention import (
    triton_softmax_attention,
    pytorch_softmax_attention,
)


def triton_attention(q, k, v, scale=None):
    """
    Compute attention using Triton kernels when possible.

    Falls back to the PyTorch implementation on non-CUDA devices.

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

    # Use Triton implementation when running on CUDA; otherwise fall back to PyTorch
    if q.is_cuda:
        return triton_softmax_attention(q, k, v, scale=scale)

    return pytorch_softmax_attention(q, k, v, scale=scale)


