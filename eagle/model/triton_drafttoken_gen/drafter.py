# Copyright (c) 2025
# Triton "drafter" megakernel skeleton for EA draft token generation.
# This file provides:
#  - Data classes (DrafterConfig, Weights, Buffers)
#  - Triton megakernel stub (_drafter_kernel) for future fusion
#  - Reference helpers (PyTorch) for modular tests:
#       * streaming_topk_lm_head_ref: blockwise LM-head + streaming top-k
#       * single_query_flashattn_ref: SDPA-based single-query attention
#  - launch_drafter(): fallback path that emits shape-correct buffers
#
# Place under: eagle/model/triton_drafttoken_gen/drafter.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


# ------------------------------
# Configuration & typed bundles
# ------------------------------

@dataclass
class DrafterConfig:
    # Model dims
    H_ea: int                 # EA hidden size (post-alignment)
    V: int                    # Vocab size (lm_head rows)
    n_head: int               # number of attention heads
    head_dim: int             # H_ea / n_head
    # Drafting parameters
    K: int                    # frontier width (top-k per depth)
    TOPK: int                 # top-k per-parent expansions (often == K)
    DEPTH: int                # number of drafting depths
    T_max: int                # max #packed nodes (root + drafted nodes)
    # Tiling (tune later)
    V_BLK: int = 2048         # vocabulary tile for lm_head matmul
    ANCBLK: int = 64          # ancestor tile for attention reduction
    # Path options
    use_fc: bool = False      # use fc align for 3*tap hidden → H_ea
    use_concat_taps: bool = True  # inputs come already concatenated
    dtype: torch.dtype = torch.float16
    acc_dtype: torch.dtype = torch.float32


@dataclass
class Weights:
    # Optional: fc align (for Low/Mid/High taps)
    W_fc: Optional[torch.Tensor] = None        # [3H, H_ea] or None

    # One-layer decoder weights (EA layer)
    Wq: Optional[torch.Tensor] = None          # [H_ea, H_ea]
    Wk: Optional[torch.Tensor] = None          # [H_ea, H_ea]
    Wv: Optional[torch.Tensor] = None          # [H_ea, H_ea]
    Wo: Optional[torch.Tensor] = None          # [H_ea, H_ea]
    # MLP (GEGLU / SiLU+Linear): keep generic two-linears for now
    W1: Optional[torch.Tensor] = None          # [H_ea, H_mid]
    W2: Optional[torch.Tensor] = None          # [H_mid, H_ea]
    # Norm params
    rms_gamma: Optional[torch.Tensor] = None   # [H_ea]

    # LM head
    W_vocab: Optional[torch.Tensor] = None     # [V, H_ea]


@dataclass
class Buffers:
    # K/V buffers laid out by packed node_id in rows of H_ea (or [T, n_head, head_dim])
    Kbuf: torch.Tensor       # [T_max, H_ea] or [T_max, n_head, head_dim]
    Vbuf: torch.Tensor       # [T_max, H_ea] or [T_max, n_head, head_dim]

    # Tree metadata for packed order
    pos_id: torch.Tensor     # [T_max], int32/64; pos_id[node] == depth of node
    parents: torch.Tensor    # [T_max], int32/64; parent[node] is packed index of parent

    # Frontier state (current breadth)
    frontier_idx: torch.Tensor  # [K], int32/64; packed node ids to expand
    scores: torch.Tensor        # [K], float (log-probs)

    # Outputs (allocated by host, filled by kernel)
    next_frontier_idx: torch.Tensor  # [K]
    next_scores: torch.Tensor        # [K]
    next_tokens: torch.Tensor        # [K]

    # Optional scratch (allocated once)
    cand_token: Optional[torch.Tensor] = None  # [K*TOPK]
    cand_score: Optional[torch.Tensor] = None  # [K*TOPK]
    cand_parent: Optional[torch.Tensor] = None # [K*TOPK]


# ------------------------------
# Reference helpers (PyTorch)
# ------------------------------

def streaming_topk_lm_head_ref(
    h: torch.Tensor,           # [..., H]
    W_vocab: torch.Tensor,     # [V, H]
    topk: int,
    V_BLK: int = 2048,
    *, acc_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming top-k over vocab without materializing full logits:
      - iterate tiles of size V_BLK
      - compute logits tile = h @ W[v0:v0+V_BLK]^T
      - merge into running top-k
    Supports h of shape [H] or [B,H] with B==1; returns (values, indices).
    """
    assert h.ndim in (1, 2), "h must be [H] or [1,H]"
    if h.ndim == 2:
        assert h.shape[0] == 1
        h_vec = h[0]
    else:
        h_vec = h
    H = h_vec.numel()
    V = W_vocab.shape[0]
    device = h_vec.device

    vals = torch.full((topk,), -float("inf"), dtype=acc_dtype, device=device)
    idxs = torch.full((topk,), -1, dtype=torch.long, device=device)

    for v0 in range(0, V, V_BLK):
        v1 = min(v0 + V_BLK, V)
        W_blk = W_vocab[v0:v1]                            # [v1-v0, H]
        logits_blk = (W_blk.to(acc_dtype) @ h_vec.to(acc_dtype))  # [v1-v0]
        # Merge blk candidates with current heap
        # Concatenate current heap with new block; take topk once
        cand_vals = torch.cat([vals, logits_blk], dim=0)
        cand_idxs = torch.cat([idxs, torch.arange(v0, v1, device=device)], dim=0)
        new_vals, order = torch.topk(cand_vals, k=min(topk, cand_vals.numel()))
        new_idxs = cand_idxs[order]
        # Keep the heap size == topk
        vals[:new_vals.numel()] = new_vals
        idxs[:new_idxs.numel()] = new_idxs
        # If new_vals smaller than topk (first iter), pad remains -inf / -1

    return vals, idxs


def single_query_flashattn_ref(
    q: torch.Tensor,      # [H] or [1,H]
    K: torch.Tensor,      # [N, H]
    V: torch.Tensor,      # [N, H]
    *, scale: float,
    attn_mask_len: Optional[int] = None,
    acc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Reference single-query attention via torch SDPA:
      - q_len = 1, key_len = attn_mask_len or N
      - returns [H]
    """
    if q.ndim == 1:
        q = q[None, None, :]  # [1,1,H]
    else:
        q = q[None, :, :]     # [1,1,H] (assuming [1,H])
    klen = attn_mask_len if attn_mask_len is not None else K.shape[0]
    Kt = K[:klen][None, :, :]  # [1,klen,H]
    Vt = V[:klen][None, :, :]  # [1,klen,H]
    out = F.scaled_dot_product_attention(
        q.to(acc_dtype), Kt.to(acc_dtype), Vt.to(acc_dtype),
        attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )  # [1,1,H]
    return out[0, 0].to(q.dtype)


# ------------------------------
# Kernel (skeleton)
# ------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _streaming_topk_kernel(
        h_ptr, W_ptr,
        output_vals_ptr, output_idx_ptr,
        V: tl.constexpr, H: tl.constexpr,
        TOPK: tl.constexpr,
        V_BLK_SIZE: tl.constexpr, H_BLK_SIZE: tl.constexpr,
        # strides
        h_stride, w_stride_v, w_stride_h
    ):
        """
        Computes top-k logits for a single hidden state vector `h` against a large weight matrix `W`.
        Uses register-based sorting to find top-k elements.
        """
        # This kernel runs in a single thread block. Parallelism is over the vocab blocks.
        pid = tl.program_id(0)

        # state for top-k, held in registers
        top_k_vals = tl.full((TOPK,), -float('inf'), dtype=tl.float32)
        top_k_idxs = tl.full((TOPK,), -1, dtype=tl.int32)

        # loop over vocabulary in blocks
        v_range = tl.arange(0, V_BLK_SIZE)
        for v_start in range(0, V, V_BLK_SIZE):
            v_offs = v_start + v_range
            v_mask = v_offs < V

            # compute logits for the block: h @ W_blk.T
            logits_acc = tl.zeros((V_BLK_SIZE,), dtype=tl.float32)
            h_range = tl.arange(0, H_BLK_SIZE)
            for h_start in range(0, H, H_BLK_SIZE):
                h_offs = h_start + h_range
                h_mask = h_offs < H
                h_block = tl.load(h_ptr + h_offs * h_stride, mask=h_mask, other=0.0)

                w_offs = v_offs[:, None] * w_stride_v + h_offs[None, :] * w_stride_h
                w_mask = v_mask[:, None] & h_mask[None, :]
                w_block = tl.load(W_ptr + w_offs, mask=w_mask, other=0.0)
                
                logits_acc += tl.sum(w_block * h_block[None, :], axis=1)

            # Merge with current top-k (simplified)
            for v_idx in range(V_BLK_SIZE):
                if v_mask[v_idx]:
                    val = logits_acc[v_idx]
                    
                    # Check if this value belongs in top-k
                    if val > top_k_vals[TOPK-1]:
                        # Find insertion point
                        insert_idx = TOPK - 1
                        while insert_idx > 0 and val > top_k_vals[insert_idx-1]:
                            insert_idx -= 1
                        
                        # Shift elements down
                        for j in range(TOPK-1, insert_idx, -1):
                            top_k_vals[j] = top_k_vals[j-1]
                            top_k_idxs[j] = top_k_idxs[j-1]
                        
                        # Insert new value
                        top_k_vals[insert_idx] = val
                        top_k_idxs[insert_idx] = v_start + v_idx

        # after the loop, write the final top-k to output
        out_offs = tl.arange(0, TOPK)
        tl.store(output_vals_ptr + out_offs, top_k_vals)
        tl.store(output_idx_ptr + out_offs, top_k_idxs)

    def streaming_topk_lm_head_triton(h: torch.Tensor, W: torch.Tensor, topk: int, V_BLK: int = 1024, H_BLK: int = 64):
        """
        Host wrapper for the Triton-based streaming top-k.
        """
        V, H = W.shape
        device = h.device
        
        assert h.shape[0] == H, "h must be a vector of size H"
        assert h.ndim == 1, "h must be a 1D vector"

        topk_vals = torch.empty(topk, device=device, dtype=torch.float32)
        topk_idx = torch.empty(topk, device=device, dtype=torch.int32)
        
        # Use a single program instance
        grid = (1,)
        
        # Launch the kernel
        _streaming_topk_kernel[grid](
            h, W,
            topk_vals, topk_idx,
            V=V, H=H,
            TOPK=topk,
            V_BLK_SIZE=V_BLK, H_BLK_SIZE=H_BLK,
            h_stride=h.stride(0), w_stride_v=W.stride(0), w_stride_h=W.stride(1)
        )
        
        return topk_vals, topk_idx.long()

    @triton.jit
    def _single_query_attn_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_ctx, stride_k_head, stride_k_dim,
        stride_v_ctx, stride_v_head, stride_v_dim,
        stride_o_batch, stride_o_head, stride_o_dim,
        n_ctx,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Triton kernel for single-query FlashAttention.
        Computes attention for a single query vector against a context of key-value pairs.
        Each program instance computes attention for one head.
        """
        # Program IDs
        pid_head = tl.program_id(0)
        pid_batch = tl.program_id(1)

        # Pointers for the current head and batch item
        q_ptr_hb = Q_ptr + pid_batch * stride_q_batch + pid_head * stride_q_head
        k_ptr_hb = K_ptr + pid_head * stride_k_head
        v_ptr_hb = V_ptr + pid_head * stride_v_head
        o_ptr_hb = O_ptr + pid_batch * stride_o_batch + pid_head * stride_o_head

        # Load query vector for the head into SRAM
        q_offs = tl.arange(0, HEAD_DIM)
        q_1d = tl.load(q_ptr_hb + q_offs * stride_q_dim)
        q = tl.reshape(q_1d, (1, HEAD_DIM))

        # Initialize accumulator, max logit, and sum of exps
        acc = tl.zeros([1, HEAD_DIM], dtype=tl.float32)
        m_i = -float('inf')
        l_i = 0.0
        scale = (HEAD_DIM) ** -0.5

        # Loop over key-value blocks
        for start_n in range(0, n_ctx, BLOCK_N):
            # --- Load K and V blocks ---
            k_offs = start_n + tl.arange(0, BLOCK_N)
            k_mask = k_offs < n_ctx
            
            k_ptr_block = k_ptr_hb + k_offs[:, None] * stride_k_ctx + q_offs[None, :] * stride_k_dim
            v_ptr_block = v_ptr_hb + k_offs[:, None] * stride_v_ctx + q_offs[None, :] * stride_v_dim
            
            k = tl.load(k_ptr_block, mask=k_mask[:, None], other=0.0)
            v = tl.load(v_ptr_block, mask=k_mask[:, None], other=0.0)

            # --- Compute attention scores (element-wise) ---
            # q is [1, HEAD_DIM], k is [BLOCK_N, HEAD_DIM]
            # We need to compute (q @ k.T). This is sum(q * k) over HEAD_DIM
            s_unmasked = tl.sum(q * k, axis=1) * scale # [BLOCK_N]
            s = tl.where(k_mask, s_unmasked, -float('inf'))

            # --- Online softmax update ---
            m_j = tl.max(s, 0)
            p_j = tl.exp(s - m_j)
            l_j = tl.sum(p_j, 0)

            m_new = tl.maximum(m_i, m_j)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_j - m_new)

            l_new = alpha * l_i + beta * l_j
            
            # --- Update accumulator (element-wise) ---
            # p_j is [BLOCK_N], v is [BLOCK_N, HEAD_DIM]
            # We need p_j.T @ v, which is an outer product style update
            p_j_casted = p_j.to(v.dtype)
            acc_new_num = tl.sum(p_j_casted[:, None] * v, axis=0) # [HEAD_DIM]
            
            acc = (l_i * alpha * acc + beta * acc_new_num[None, :]) / l_new
            
            l_i = l_new
            m_i = m_new

        # Write output to global memory
        acc_1d = tl.reshape(acc, (HEAD_DIM,))
        o_offs = tl.arange(0, HEAD_DIM)
        tl.store(o_ptr_hb + o_offs * stride_o_dim, acc_1d)

    def single_query_flashattn_triton(
        q: torch.Tensor,      # [1, H]
        K_cache: torch.Tensor, # [N_CTX, H]
        V_cache: torch.Tensor, # [N_CTX, H]
        n_head: int,
    ) -> torch.Tensor:
        """
        Host wrapper for the Triton-based single-query attention.
        """
        Z, H = q.shape
        N_CTX, _ = K_cache.shape
        assert Z == 1, "Batch size must be 1 for single-query attention"
        
        head_dim = H // n_head
        assert head_dim * n_head == H, "H must be divisible by n_head"

        o = torch.empty_like(q)

        # Reshape for per-head processing
        q_reshaped = q.view(Z, n_head, head_dim)
        K_reshaped = K_cache.view(N_CTX, n_head, head_dim)
        V_reshaped = V_cache.view(N_CTX, n_head, head_dim)
        o_reshaped = o.view(Z, n_head, head_dim)

        grid = (n_head, Z)
        
        # Use heuristics for block size
        BLOCK_N = 64 if head_dim <= 64 else 32

        _single_query_attn_kernel[grid](
            q_reshaped, K_reshaped, V_reshaped, o_reshaped,
            q_reshaped.stride(0), q_reshaped.stride(1), q_reshaped.stride(2),
            K_reshaped.stride(0), K_reshaped.stride(1), K_reshaped.stride(2),
            V_reshaped.stride(0), V_reshaped.stride(1), V_reshaped.stride(2),
            o_reshaped.stride(0), o_reshaped.stride(1), o_reshaped.stride(2),
            n_ctx=N_CTX,
            HEAD_DIM=head_dim,
            BLOCK_N=BLOCK_N,
        )
        return o

    @triton.jit
    def _drafter_kernel(
        # Pointers
        X_ptr,                        # taps or already-aligned input hidden for *current* nodes (implementation-defined)
        W_fc_ptr, use_fc: tl.constexpr,
        Wq_ptr, Wk_ptr, Wv_ptr, Wo_ptr,
        W1_ptr, W2_ptr, rms_gamma_ptr,
        W_vocab_ptr,
        Kbuf_ptr, Vbuf_ptr,
        pos_id_ptr, parents_ptr,
        frontier_idx_ptr, scores_ptr,
        next_frontier_idx_ptr, next_scores_ptr, next_tokens_ptr,
        cand_tok_ptr, cand_scr_ptr, cand_par_ptr,
        # Dims / strides
        H: tl.constexpr, V: tl.constexpr, T_max: tl.constexpr,
        K: tl.constexpr, TOPK: tl.constexpr, DEPTH: tl.constexpr,
        V_BLK: tl.constexpr, ANCBLK: tl.constexpr,
        n_head: tl.constexpr, head_dim: tl.constexpr,
        # Strides
        X_stride_b, X_stride_s, X_stride_h,
        W_fc_stride_i, W_fc_stride_o,
        Wq_stride_i, Wq_stride_o, Wk_stride_i, Wk_stride_o, Wv_stride_i, Wv_stride_o, Wo_stride_i, Wo_stride_o,
        W1_stride_i, W1_stride_o, W2_stride_i, W2_stride_o,
        W_vocab_stride_v, W_vocab_stride_h,
        Kbuf_stride_t, Kbuf_stride_h, Vbuf_stride_t, Vbuf_stride_h,
        # Other params
        eps: float = 1e-6,
        use_concat_taps: tl.constexpr = True,
    ):
        """
        Triton megakernel for EA draft token generation.
        
        Each program instance handles one sequence, looping over DEPTH steps.
        """
        pid = tl.program_id(0)  # One PI per sequence
        
        # Constants
        sqrt_head_dim = tl.math.sqrt(float(head_dim))
        
        # Initialize node counter (starts at 1 for root)
        node_counter = 1
        
        # Loop over depths
        for d in range(DEPTH):
            # Load current frontier nodes and scores
            frontier_offsets = tl.arange(0, K)
            frontier_mask = frontier_offsets < K
            frontier_nodes = tl.load(frontier_idx_ptr + frontier_offsets, mask=frontier_mask, other=0)
            frontier_scores = tl.load(scores_ptr + frontier_offsets, mask=frontier_mask, other=-float('inf'))
            
            # 1) Ingest taps + align (optional)
            # For each frontier node, load its input hidden state
            for k in range(K):
                node_idx = frontier_nodes[k]
                
                # Load input hidden for the current node
                h_in = tl.zeros([H], dtype=tl.float32)
                
                if use_concat_taps:
                    # Load concatenated taps directly
                    h_offs = tl.arange(0, H)
                    h_in = tl.load(X_ptr + node_idx * X_stride_s + h_offs * X_stride_h)
                else:
                    # Load separate taps and concatenate
                    # This would need to be implemented based on how taps are stored
                    pass
                
                # Apply fc alignment if needed
                if use_fc:
                    h_aligned = tl.zeros([H], dtype=tl.float32)
                    for h_in_start in range(0, 3*H, H):
                        for h_out in range(H):
                            dot_prod = 0.0
                            for h_in_offset in range(H):
                                h_in_idx = h_in_start + h_in_offset
                                if h_in_idx < 3*H:  # Boundary check
                                    w = tl.load(W_fc_ptr + h_in_idx * W_fc_stride_i + h_out * W_fc_stride_o)
                                    x = tl.load(X_ptr + node_idx * X_stride_s + h_in_idx * X_stride_h)
                                    dot_prod += w * x
                            h_aligned[h_out] += dot_prod
                    h_in = h_aligned
                
                # 2) QKV + ROPE + cache write
                # Compute Q, K, V projections
                q = tl.zeros([H], dtype=tl.float32)
                k = tl.zeros([H], dtype=tl.float32)
                v = tl.zeros([H], dtype=tl.float32)
                
                # Compute Q projection
                for h_out in range(H):
                    dot_prod = 0.0
                    for h_in_offset in range(H):
                        w = tl.load(Wq_ptr + h_in_offset * Wq_stride_i + h_out * Wq_stride_o)
                        dot_prod += w * h_in[h_in_offset]
                    q[h_out] = dot_prod
                
                # Compute K projection
                for h_out in range(H):
                    dot_prod = 0.0
                    for h_in_offset in range(H):
                        w = tl.load(Wk_ptr + h_in_offset * Wk_stride_i + h_out * Wk_stride_o)
                        dot_prod += w * h_in[h_in_offset]
                    k[h_out] = dot_prod
                
                # Compute V projection
                for h_out in range(H):
                    dot_prod = 0.0
                    for h_in_offset in range(H):
                        w = tl.load(Wv_ptr + h_in_offset * Wv_stride_i + h_out * Wv_stride_o)
                        dot_prod += w * h_in[h_in_offset]
                    v[h_out] = dot_prod
                
                # Apply ROPE to Q and K
                pos = tl.load(pos_id_ptr + node_idx)
                
                # Reshape q, k for per-head processing
                q_heads = tl.reshape(q, (n_head, head_dim))
                k_heads = tl.reshape(k, (n_head, head_dim))
                
                # Apply ROPE (simplified implementation)
                for h in range(n_head):
                    for i in range(0, head_dim, 2):
                        if i + 1 < head_dim:  # Ensure we have pairs
                            freq = 1.0 / (10000.0 ** (i / head_dim))
                            val = pos * freq
                            
                            # Apply rotation to q
                            q0 = q_heads[h, i]
                            q1 = q_heads[h, i+1]
                            q_heads[h, i] = q0 * tl.math.cos(val) - q1 * tl.math.sin(val)
                            q_heads[h, i+1] = q0 * tl.math.sin(val) + q1 * tl.math.cos(val)
                            
                            # Apply rotation to k
                            k0 = k_heads[h, i]
                            k1 = k_heads[h, i+1]
                            k_heads[h, i] = k0 * tl.math.cos(val) - k1 * tl.math.sin(val)
                            k_heads[h, i+1] = k0 * tl.math.sin(val) + k1 * tl.math.cos(val)
                
                # Reshape back
                q = tl.reshape(q_heads, (H,))
                k = tl.reshape(k_heads, (H,))
                
                # Write K, V to cache for this node
                k_offs = tl.arange(0, H)
                v_offs = tl.arange(0, H)
                tl.store(Kbuf_ptr + node_idx * Kbuf_stride_t + k_offs * Kbuf_stride_h, k)
                tl.store(Vbuf_ptr + node_idx * Vbuf_stride_t + v_offs * Vbuf_stride_h, v)
                
                # 3) Tree attention (FlashAttention-style)
                # For this node, attend to all its ancestors (implicit mask)
                n_ancestors = pos + 1  # Including self
                
                # Initialize accumulators for attention
                m_i = -float('inf')  # Running max for numerical stability
                l_i = 0.0  # Sum of exp(score - max_score)
                attn_out = tl.zeros([H], dtype=tl.float32)
                
                # Process ancestors in blocks
                for anc_start in range(0, n_ancestors, ANCBLK):
                    anc_end = min(anc_start + ANCBLK, n_ancestors)
                    anc_len = anc_end - anc_start
                    
                    # Load K, V blocks for these ancestors
                    anc_offsets = tl.arange(anc_start, anc_end)
                    anc_mask = anc_offsets < n_ancestors
                    
                    # Process by head for better register usage
                    for h in range(n_head):
                        h_start = h * head_dim
                        h_end = (h + 1) * head_dim
                        
                        # Extract query vector for this head
                        q_head = q[h_start:h_end]
                        
                        # Load K, V for this head and ancestor block
                        k_block = tl.zeros([anc_len, head_dim], dtype=tl.float32)
                        v_block = tl.zeros([anc_len, head_dim], dtype=tl.float32)
                        
                        for i in range(anc_len):
                            if anc_mask[i]:
                                anc_idx = anc_offsets[i]
                                for j in range(head_dim):
                                    k_idx = h_start + j
                                    k_block[i, j] = tl.load(Kbuf_ptr + anc_idx * Kbuf_stride_t + k_idx * Kbuf_stride_h)
                                    v_block[i, j] = tl.load(Vbuf_ptr + anc_idx * Vbuf_stride_t + k_idx * Vbuf_stride_h)
                        
                        # Compute attention scores
                        scores = tl.zeros([anc_len], dtype=tl.float32)
                        for i in range(anc_len):
                            if anc_mask[i]:
                                dot_prod = 0.0
                                for j in range(head_dim):
                                    dot_prod += q_head[j] * k_block[i, j]
                                scores[i] = dot_prod / sqrt_head_dim
                            else:
                                scores[i] = -float('inf')
                        
                        # Find max score in this block for numerical stability
                        m_block = -float('inf')
                        for i in range(anc_len):
                            m_block = tl.maximum(m_block, scores[i])
                        
                        # Update running max and rescale previous contributions
                        m_new = tl.maximum(m_i, m_block)
                        scale_old = tl.math.exp(m_i - m_new)
                        scale_new = tl.math.exp(m_block - m_new)
                        
                        # Compute attention weights and weighted sum
                        l_block = 0.0
                        attn_head = tl.zeros([head_dim], dtype=tl.float32)
                        
                        for i in range(anc_len):
                            if anc_mask[i]:
                                weight = tl.math.exp(scores[i] - m_block)
                                l_block += weight
                                for j in range(head_dim):
                                    attn_head[j] += weight * v_block[i, j]
                        
                        # Update accumulators with rescaling
                        l_new = scale_old * l_i + scale_new * l_block
                        
                        # Update output for this head
                        for j in range(head_dim):
                            old_val = attn_out[h_start + j]
                            new_val = scale_old * old_val + scale_new * attn_head[j]
                            attn_out[h_start + j] = new_val / l_new if l_new > 0 else 0.0
                        
                        # Update running max and sum
                        m_i = m_new
                        l_i = l_new
                
                # 4) Apply output projection (Wo)
                attn_proj = tl.zeros([H], dtype=tl.float32)
                for h_out in range(H):
                    dot_prod = 0.0
                    for h_in_offset in range(H):
                        w = tl.load(Wo_ptr + h_in_offset * Wo_stride_i + h_out * Wo_stride_o)
                        dot_prod += w * attn_out[h_in_offset]
                    attn_proj[h_out] = dot_prod
                
                # Add residual connection
                hidden = h_in + attn_proj
                
                # RMSNorm
                rms_norm_hidden = tl.zeros([H], dtype=tl.float32)
                variance = 0.0
                for i in range(H):
                    variance += hidden[i] * hidden[i]
                variance = variance / H + eps
                inv_std = 1.0 / tl.math.sqrt(variance)
                
                for i in range(H):
                    gamma = tl.load(rms_gamma_ptr + i)
                    rms_norm_hidden[i] = hidden[i] * inv_std * gamma
                
                # MLP (SwiGLU or similar)
                # First projection (W1)
                mlp_inter = tl.zeros([4*H], dtype=tl.float32)  # Assuming 4x expansion
                for h_out in range(4*H):
                    dot_prod = 0.0
                    for h_in_offset in range(H):
                        w = tl.load(W1_ptr + h_in_offset * W1_stride_i + h_out * W1_stride_o)
                        dot_prod += w * rms_norm_hidden[h_in_offset]
                    mlp_inter[h_out] = dot_prod
                
                # Apply activation (SiLU/Swish)
                for i in range(2*H):
                    gate = mlp_inter[i]
                    mlp_inter[i] = gate * (1.0 / (1.0 + tl.math.exp(-gate)))  # SiLU
                
                # Element-wise multiply gate and linear parts
                for i in range(2*H):
                    mlp_inter[i] = mlp_inter[i] * mlp_inter[i + 2*H]
                
                # Second projection (W2)
                mlp_out = tl.zeros([H], dtype=tl.float32)
                for h_out in range(H):
                    dot_prod = 0.0
                    for h_in_offset in range(2*H):
                        w = tl.load(W2_ptr + h_in_offset * W2_stride_i + h_out * W2_stride_o)
                        dot_prod += w * mlp_inter[h_in_offset]
                    mlp_out[h_out] = dot_prod
                
                # Add residual
                hidden = hidden + mlp_out
                
                # 5) LM-head matmul + streaming top-k
                # Initialize top-k tracking
                topk_vals = tl.zeros([TOPK], dtype=tl.float32)
                topk_idxs = tl.zeros([TOPK], dtype=tl.int32)
                
                for i in range(TOPK):
                    topk_vals[i] = -float('inf')
                    topk_idxs[i] = -1
                
                # Process vocabulary in blocks
                for v_start in range(0, V, V_BLK):
                    v_end = min(v_start + V_BLK, V)
                    v_len = v_end - v_start
                    
                    # Load vocab block
                    logits_block = tl.zeros([v_len], dtype=tl.float32)
                    
                    # Compute logits for this vocab block
                    for v_idx in range(v_len):
                        dot_prod = 0.0
                        for h_idx in range(H):
                            w = tl.load(W_vocab_ptr + (v_start + v_idx) * W_vocab_stride_v + h_idx * W_vocab_stride_h)
                            dot_prod += w * hidden[h_idx]
                        logits_block[v_idx] = dot_prod
                    
                    # Convert to log-probs (simplified softmax)
                    # In practice, we'd need a proper softmax, but for top-k we can use raw logits
                    
                    # Merge with current top-k (simplified)
                    for v_idx in range(v_len):
                        val = logits_block[v_idx]
                        
                        # Check if this value belongs in top-k
                        if val > topk_vals[TOPK-1]:
                            # Find insertion point
                            insert_idx = TOPK - 1
                            while insert_idx > 0 and val > topk_vals[insert_idx-1]:
                                insert_idx -= 1
                            
                            # Shift elements down
                            for j in range(TOPK-1, insert_idx, -1):
                                topk_vals[j] = topk_vals[j-1]
                                topk_idxs[j] = topk_idxs[j-1]
                            
                            # Insert new value
                            topk_vals[insert_idx] = val
                            topk_idxs[insert_idx] = v_start + v_idx
                
                # Store top-k tokens and scores for this node
                for i in range(TOPK):
                    tl.store(cand_tok_ptr + k * TOPK + i, topk_idxs[i])
                    # Add parent score to get cumulative score
                    tl.store(cand_scr_ptr + k * TOPK + i, topk_vals[i] + frontier_scores[k])
                    tl.store(cand_par_ptr + k * TOPK + i, node_idx)
            
            # 6) Global K-way prune
            # We have K*TOPK candidates, need to select top K
            cand_range = tl.arange(0, K*TOPK)
            cand_mask = cand_range < K*TOPK
            
            # Load all candidate scores
            all_scores = tl.zeros([K*TOPK], dtype=tl.float32)
            for i in range(K*TOPK):
                if cand_mask[i]:
                    all_scores[i] = tl.load(cand_scr_ptr + i)
                else:
                    all_scores[i] = -float('inf')
            
            # Find top K scores (simplified selection)
            next_scores_local = tl.zeros([K], dtype=tl.float32)
            next_indices_local = tl.zeros([K], dtype=tl.int32)
            
            for i in range(K):
                next_scores_local[i] = -float('inf')
                next_indices_local[i] = -1
            
            # Simple selection algorithm
            for i in range(K*TOPK):
                if cand_mask[i]:
                    val = all_scores[i]
                    if val > next_scores_local[K-1]:
                        # Find insertion point
                        insert_idx = K - 1
                        while insert_idx > 0 and val > next_scores_local[insert_idx-1]:
                            insert_idx -= 1
                        
                        # Shift elements down
                        for j in range(K-1, insert_idx, -1):
                            next_scores_local[j] = next_scores_local[j-1]
                            next_indices_local[j] = next_indices_local[j-1]
                        
                        # Insert new value
                        next_scores_local[insert_idx] = val
                        next_indices_local[insert_idx] = i
            
            # 7) Update state for next iteration
            # Store next frontier information
            for i in range(K):
                if i < K:
                    cand_idx = next_indices_local[i]
                    if cand_idx >= 0 and cand_idx < K*TOPK:
                        # Get token, parent for this candidate
                        token = tl.load(cand_tok_ptr + cand_idx)
                        parent = tl.load(cand_par_ptr + cand_idx)
                        score = next_scores_local[i]
                        
                        # Store in next frontier
                        tl.store(next_frontier_idx_ptr + i, node_counter)
                        tl.store(next_scores_ptr + i, score)
                        tl.store(next_tokens_ptr + i, token)
                        
                        # Update tree metadata
                        tl.store(parents_ptr + node_counter, parent)
                        tl.store(pos_id_ptr + node_counter, d + 1)  # depth = current depth + 1
                        
                        # Increment node counter
                        node_counter += 1
            
            # Swap frontier pointers for next iteration
            # In practice, this would be done by the host code
            
            # Synchronize before next iteration
            tl.debug_barrier()


# ------------------------------
# Host wrapper
# ------------------------------

def launch_drafter(
    cfg: DrafterConfig,
    X_concat_or_three: Dict[str, torch.Tensor],
    weights: Weights,
    bufs: Buffers,
    fallback: bool = True,
):
    """
    Launch the Triton drafter megakernel for one drafting *phase* (DEPTH steps),
    or run a PyTorch fallback that produces *shape-correct* outputs.

    Returns
    -------
    draft_tokens : torch.LongTensor [1, T]
    retrieve_indices : torch.LongTensor [R, C]
    tree_mask : torch.FloatTensor [1, 1, T, T]  (0/1 mask)
    tree_position_ids : torch.LongTensor [T]
    """
    device = bufs.Kbuf.device
    T = cfg.T_max

    # --- basic shape checks ---
    assert bufs.Kbuf.shape[0] >= T and bufs.Vbuf.shape[0] >= T, "K/V buf rows < T_max"
    assert bufs.frontier_idx.numel() == cfg.K and bufs.scores.numel() == cfg.K, "frontier shape mismatch"

    if _TRITON_AVAILABLE and not fallback:
        # Prepare scratch buffers if not provided
        if bufs.cand_token is None:
            bufs.cand_token = torch.empty(cfg.K * cfg.TOPK, dtype=torch.long, device=device)
        if bufs.cand_score is None:
            bufs.cand_score = torch.empty(cfg.K * cfg.TOPK, dtype=cfg.acc_dtype, device=device)
        if bufs.cand_parent is None:
            bufs.cand_parent = torch.empty(cfg.K * cfg.TOPK, dtype=torch.long, device=device)
        
        # Get input tensor
        if "X_concat" in X_concat_or_three:
            X = X_concat_or_three["X_concat"]
            use_concat_taps = True
        else:
            # For now, we only support concatenated taps
            raise NotImplementedError("Separate taps not yet supported in Triton kernel")
        
        # Prepare grid
        grid = (1,)  # One program instance per sequence
        
        # Launch kernel
        _drafter_kernel[grid](
            # Pointers
            X, 
            weights.W_fc if weights.W_fc is not None else torch.empty(0, device=device),
            weights.Wq, weights.Wk, weights.Wv, weights.Wo,
            weights.W1, weights.W2, weights.rms_gamma,
            weights.W_vocab,
            bufs.Kbuf, bufs.Vbuf,
            bufs.pos_id, bufs.parents,
            bufs.frontier_idx, bufs.scores,
            bufs.next_frontier_idx, bufs.next_scores, bufs.next_tokens,
            bufs.cand_token, bufs.cand_score, bufs.cand_parent,
            # Dims / strides
            H=cfg.H_ea, V=cfg.V, T_max=cfg.T_max,
            K=cfg.K, TOPK=cfg.TOPK, DEPTH=cfg.DEPTH,
            V_BLK=cfg.V_BLK, ANCBLK=cfg.ANCBLK,
            n_head=cfg.n_head, head_dim=cfg.head_dim,
            # Strides
            X_stride_b=X.stride(0) if X.ndim > 2 else 0,
            X_stride_s=X.stride(-2),
            X_stride_h=X.stride(-1),
            W_fc_stride_i=weights.W_fc.stride(0) if weights.W_fc is not None else 0,
            W_fc_stride_o=weights.W_fc.stride(1) if weights.W_fc is not None else 0,
            Wq_stride_i=weights.Wq.stride(0),
            Wq_stride_o=weights.Wq.stride(1),
            Wk_stride_i=weights.Wk.stride(0),
            Wk_stride_o=weights.Wk.stride(1),
            Wv_stride_i=weights.Wv.stride(0),
            Wv_stride_o=weights.Wv.stride(1),
            Wo_stride_i=weights.Wo.stride(0),
            Wo_stride_o=weights.Wo.stride(1),
            W1_stride_i=weights.W1.stride(0),
            W1_stride_o=weights.W1.stride(1),
            W2_stride_i=weights.W2.stride(0),
            W2_stride_o=weights.W2.stride(1),
            W_vocab_stride_v=weights.W_vocab.stride(0),
            W_vocab_stride_h=weights.W_vocab.stride(1),
            Kbuf_stride_t=bufs.Kbuf.stride(0),
            Kbuf_stride_h=bufs.Kbuf.stride(1),
            Vbuf_stride_t=bufs.Vbuf.stride(0),
            Vbuf_stride_h=bufs.Vbuf.stride(1),
            # Other params
            eps=1e-6,
            use_concat_taps=use_concat_taps,
        )
        
        # After kernel execution, we need to build the tree structure from the results
        # This is similar to the fallback path but uses the kernel-populated buffers
        
        # Get the number of nodes actually used (could be less than T_max)
        valid_nodes = torch.sum(bufs.parents >= 0).item()
        
        # Build draft_tokens from next_tokens
        draft_tokens = torch.full((1, valid_nodes), -1, dtype=torch.long, device=device)
        draft_tokens[0, 0] = 0  # Root token
        
        # Copy tokens from next_tokens for each depth
        token_idx = 1
        for d in range(cfg.DEPTH):
            for k in range(cfg.K):
                if token_idx < valid_nodes:
                    draft_tokens[0, token_idx] = bufs.next_tokens[k + d * cfg.K]
                    token_idx += 1
        
        # Build tree mask from parents
        tree_mask = torch.zeros((valid_nodes, valid_nodes), dtype=cfg.dtype, device=device)
        for i in range(valid_nodes):
            p = i
            while p >= 0:
                tree_mask[i, p] = 1.0
                p = bufs.parents[p].item()
        tree_mask = tree_mask.unsqueeze(0).unsqueeze(0)
        
        # Get tree position ids
        tree_position_ids = bufs.pos_id[:valid_nodes].clone()
        
        # Build retrieve indices
        # Find leaf nodes (nodes that are not parents of any other node)
        is_parent = torch.zeros(valid_nodes, dtype=torch.bool, device=device)
        for i in range(1, valid_nodes):
            parent_idx = bufs.parents[i].item()
            if parent_idx >= 0:
                is_parent[parent_idx] = True
        
        leaf_nodes = torch.nonzero(~is_parent, as_tuple=False).flatten()
        max_depth = torch.max(tree_position_ids).item() + 1
        
        retrieve_indices = torch.full((leaf_nodes.numel(), max_depth), -1, dtype=torch.long, device=device)
        for i, leaf in enumerate(leaf_nodes):
            node = leaf
            depth = tree_position_ids[node].item()
            retrieve_indices[i, depth] = node
            
            while node > 0:
                parent = bufs.parents[node].item()
                parent_depth = tree_position_ids[parent].item()
                retrieve_indices[i, parent_depth] = parent
                node = parent
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    # ------------------------------
    # Fallback: produce shape-correct outputs (stub)
    # ------------------------------
    draft_tokens = torch.full((1, T), -1, dtype=torch.long, device=device)
    tree_pos = torch.zeros((T,), dtype=torch.long, device=device)
    parents = torch.full((T,), -1, dtype=torch.long, device=device)

    # root
    draft_tokens[0, 0] = 0
    parents[0] = -1
    tree_pos[0] = 0

    # breadth-then-depth pack
    node = 1
    for d in range(1, cfg.DEPTH + 1):
        for k in range(cfg.K):
            if node >= T:
                break
            draft_tokens[0, node] = (1000 + d * 17 + k) % cfg.V
            parents[node] = 0 if d == 1 else (1 + (d - 2) * cfg.K + (k % cfg.K))
            tree_pos[node] = d
            node += 1
        if node >= T:
            break

    # Build tree_mask [1,1,T_eff,T_eff] from parents
    T_eff = node
    mask = torch.zeros((T_eff, T_eff), dtype=cfg.dtype, device=device)
    for i in range(T_eff):
        p = i
        while p >= 0:
            mask[i, p] = 1.0
            p = parents[p].item()
    tree_mask = mask.unsqueeze(0).unsqueeze(0).to(cfg.dtype)

    # Retrieve indices from leaves
    children_count = torch.zeros((T_eff,), dtype=torch.int32, device=device)
    for i in range(1, T_eff):
        p = parents[i].item()
        if p >= 0:
            children_count[p] += 1
    leaves = torch.nonzero(children_count == 0, as_tuple=False).flatten().tolist()
    if 0 in leaves:
        leaves.remove(0)

    paths: List[List[int]] = []
    max_depth = 1
    for leaf in leaves:
        chain = []
        p = leaf
        while p >= 0:
            chain.append(p)
            p = parents[p].item()
        chain.reverse()
        paths.append(chain)
        max_depth = max(max_depth, len(chain))

    R = max(1, len(paths))
    C = max_depth
    retrieve_indices = torch.full((R, C), -1, dtype=torch.long, device=device)
    for r, chain in enumerate(paths):
        retrieve_indices[r, :len(chain)] = torch.tensor(chain, dtype=torch.long, device=device)

    tree_position_ids = tree_pos.clone()
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids