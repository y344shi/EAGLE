import torch
import triton
import triton.language as tl

@triton.jit
def _fused_attention_kv_tree_kernel(
    q, k_cache, v_cache, k_new, v_new, logits,
    output, top_value, top_index, current_length,
    stride_qb, stride_qd,
    stride_kcb, stride_kcn, stride_kck,
    stride_vcb, stride_vcn, stride_vck,
    stride_knb, stride_knd,
    stride_vnb, stride_vnd,
    stride_lb, stride_lv,
    batch_size, vocab_size,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return

    curr = tl.load(current_length + pid)
    offs_d = tl.arange(0, HEAD_DIM)

    q_vec = tl.load(q + pid * stride_qb + offs_d * stride_qd)
    k_vec = tl.load(k_new + pid * stride_knb + offs_d * stride_knd)
    v_vec = tl.load(v_new + pid * stride_vnb + offs_d * stride_vnd)

    # append to caches
    tl.store(k_cache + pid * stride_kcb + curr * stride_kcn + offs_d * stride_kck, k_vec)
    tl.store(v_cache + pid * stride_vcb + curr * stride_vcn + offs_d * stride_vck, v_vec)

    # simple attention: element-wise product
    out = q_vec * v_vec
    tl.store(output + pid * stride_qb + offs_d * stride_qd, out)

    # top-1 over logits
    row_ptr = logits + pid * stride_lb
    max_val = tl.load(row_ptr)
    max_idx = tl.zeros((), dtype=tl.int32)
    for i in range(1, vocab_size):
        val = tl.load(row_ptr + i * stride_lv)
        max_idx = tl.where(val > max_val, i, max_idx)
        max_val = tl.where(val > max_val, val, max_val)
    tl.store(top_value + pid, max_val)
    tl.store(top_index + pid, max_idx)

    tl.store(current_length + pid, curr + 1)


def fused_attention_kv_tree(q, k_cache, v_cache, k_new, v_new, logits, current_length):
    """Prototype fused kernel executing attention, KV-cache update and top-1 search.

    The implementation is intentionally simplified and is meant for benchmarking
    and experimentation rather than production use.
    """
    batch_size, head_dim = q.shape
    vocab_size = logits.shape[1]

    output = torch.empty_like(q)
    top_value = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)
    top_index = torch.empty(batch_size, dtype=torch.int32, device=logits.device)

    stride_qb, stride_qd = q.stride()
    stride_kcb, stride_kcn, stride_kck = k_cache.stride()
    stride_vcb, stride_vcn, stride_vck = v_cache.stride()
    stride_knb, stride_knd = k_new.stride()
    stride_vnb, stride_vnd = v_new.stride()
    stride_lb, stride_lv = logits.stride()

    grid = (batch_size,)
    _fused_attention_kv_tree_kernel[grid](
        q, k_cache, v_cache, k_new, v_new, logits,
        output, top_value, top_index, current_length,
        stride_qb, stride_qd,
        stride_kcb, stride_kcn, stride_kck,
        stride_vcb, stride_vcn, stride_vck,
        stride_knb, stride_knd,
        stride_vnb, stride_vnd,
        stride_lb, stride_lv,
        batch_size, vocab_size,
        HEAD_DIM=head_dim,
    )

    return output, top_value, top_index
