"""Placeholder for fused speculative decoding mega kernel.

This module sketches the interface for a future Triton kernel that fuses
several steps of the speculative decoding pipeline:

- Draft-token generation
- Candidate verification against the base model
- KV-cache updates
- Input sequence updates

The actual kernel implementation is left as future work and will require
coordinated tiling across sequence, head and vocabulary dimensions.
"""

import torch
import triton
import triton.language as tl


def triton_mega_kernel(*args, **kwargs):
    """Placeholder function for future fused kernel."""
    raise NotImplementedError("Mega kernel is not yet implemented")
