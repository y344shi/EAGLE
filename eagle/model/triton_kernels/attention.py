"""High level attention wrapper.

This module originally provided a pure PyTorch implementation as a
fallback while Triton kernels were under development.  The optimized
softmax attention kernel now lives in :mod:`softmax_attention` and is
used directly here to keep the public API stable.
"""

from .softmax_attention import triton_softmax_attention


def triton_attention(q, k, v, scale=None, mask=None):
    """Compute attention using the accelerated Triton kernel.

    Parameters
    ----------
    q, k, v:
        Query, key and value tensors of shape ``[batch, heads, seq, dim]``.
    mask: optional
        Attention mask tensor broadcastable to the attention matrix.
    scale: optional
        Scaling factor applied to the attention scores before softmax.

    Returns
    -------
    torch.Tensor
        Attention output of shape ``[batch, heads, seq, dim]``.
    """

    return triton_softmax_attention(q, k, v, mask=mask, scale=scale)



