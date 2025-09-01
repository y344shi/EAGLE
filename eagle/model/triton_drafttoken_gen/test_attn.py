import pytest
import torch
import torch.nn.functional as F

from eagle.model.triton_drafttoken_gen.drafter import (
    single_query_flashattn_ref,
    single_query_flashattn_triton,
    _TRITON_AVAILABLE
)

@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="Triton is not available")
@pytest.mark.parametrize("H, N_CTX, N_HEAD", [
    (128, 256, 4),
    (256, 512, 8),
])
def test_single_query_attn_triton_matches_ref(H, N_CTX, N_HEAD):
    device = torch.device("cuda")
    dtype = torch.float16
    head_dim = H // N_HEAD
    scale = head_dim ** -0.5

    torch.manual_seed(1)
    q = torch.randn(H, device=device, dtype=dtype)
    K_cache = torch.randn(N_CTX, H, device=device, dtype=dtype)
    V_cache = torch.randn(N_CTX, H, device=device, dtype=dtype)

    # Reference implementation
    ref_out = single_query_flashattn_ref(
        q, K_cache, V_cache, scale=scale, attn_mask_len=N_CTX
    )

    # Triton implementation (to be implemented)
    # This will fail until the triton kernel is written
    triton_out = single_query_flashattn_triton(
        q.unsqueeze(0), K_cache, V_cache, N_HEAD
    )

    # Compare results
    assert torch.allclose(ref_out, triton_out.squeeze(0), atol=1e-2, rtol=0)