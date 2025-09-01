# Unit test for streaming LM-head top-k reference helper.
# Run: pytest -q eagle/model/triton_drafttoken_gen/test_lmhead_topk.py

import torch
import pytest

from eagle.model.triton_drafttoken_gen.drafter import streaming_topk_lm_head_ref, streaming_topk_lm_head_triton, _TRITON_AVAILABLE

@pytest.mark.parametrize("H,V,topk,V_BLK", [
    (128, 1024, 5, 256),
    (256, 4096, 10, 512),
])
def test_streaming_topk_ref_matches_full(H, V, topk, V_BLK):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    torch.manual_seed(0)
    h = torch.randn(H, device=device, dtype=dtype)
    W = torch.randn(V, H, device=device, dtype=dtype)

    ref_vals, ref_idx = streaming_topk_lm_head_ref(h, W, topk, V_BLK=V_BLK, acc_dtype=torch.float32)
    # Full logits (reference)
    logits = (W.float() @ h.float())
    full_vals, full_idx = torch.topk(logits, k=topk)

    # Compare sets (order should match topk order)
    assert torch.allclose(ref_vals, full_vals, rtol=1e-4, atol=1e-3)
    assert torch.equal(ref_idx, full_idx)

@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="Triton is not available")
@pytest.mark.parametrize("H,V,topk,V_BLK,H_BLK", [
    (128, 1024, 5, 256, 64),
    (256, 4096, 10, 512, 128),
])
def test_streaming_topk_triton_matches_ref(H, V, topk, V_BLK, H_BLK):
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(0)
    h = torch.randn(H, device=device, dtype=dtype)
    W = torch.randn(V, H, device=device, dtype=dtype)

    # Reference implementation
    ref_vals, ref_idx = streaming_topk_lm_head_ref(h, W, topk, V_BLK=V_BLK, acc_dtype=torch.float32)

    # Triton implementation
    triton_vals, triton_idx = streaming_topk_lm_head_triton(h, W, topk, V_BLK=V_BLK, H_BLK=H_BLK)

    # The bubble sort in Triton might not be perfectly stable, so we check for value and index set equality.
    # Sort both by index to create a canonical order for comparison.
    ref_idx_sorted, ref_order = torch.sort(ref_idx)
    ref_vals_sorted = ref_vals[ref_order]

    triton_idx_sorted, triton_order = torch.sort(triton_idx)
    triton_vals_sorted = triton_vals[triton_order]
    
    assert torch.equal(ref_idx_sorted, triton_idx_sorted), "Indices do not match"
    assert torch.allclose(ref_vals_sorted, triton_vals_sorted, rtol=1e-3, atol=1e-2), "Values do not match"
