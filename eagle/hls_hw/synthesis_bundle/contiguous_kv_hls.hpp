#ifndef TMAC_CONTIGUOUS_KV_HLS_HPP
#define TMAC_CONTIGUOUS_KV_HLS_HPP

// Contiguous KV cache: write new tokens to HBM, gather prefix + ancestors + self.
// Replaces kv_cache_manager (URAM/HBM hot-cold tiering) and paged-attention indirection.
//
// HBM layout (batch_size=1):
//   [0 .. prefix_len-1]                          : prefix KV from target model
//   [prefix_len + layer * max_tree_width + t]     : draft layer `layer`, tree slot `t`
//
// max_tree_width is the fixed stride per layer (= compile-time TREE_WIDTH for now).
// Each layer may use fewer than max_tree_width slots; unused slots are padding.
//
// Each token occupies VECS_PER_KV_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W vectors.

#include "tmac_utils.hpp"
#include "eagle_tier1_top.hpp"

namespace tmac {
namespace hls {

constexpr int kMaxDraftDepth = 16;

// ---------------------------------------------------------------------------
// Write TREE_WIDTH new K/V tokens to contiguous HBM.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int NUM_KV_HEADS>
void contiguous_kv_write(
    hls_stream<vec_t<VEC_W>>& k_in,   // post-RoPE K, TREE_WIDTH tokens token-major
    hls_stream<vec_t<VEC_W>>& v_in,   // V projection, TREE_WIDTH tokens token-major
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    int write_base_token               // = prefix_len + current_depth * max_tree_width
) {
#pragma HLS INLINE off
    constexpr int VECS_PER_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
    static_assert((NUM_KV_HEADS * HEAD_DIM) % VEC_W == 0, "KV width must align to VEC_W");

write_token_loop:
    for (int t = 0; t < TREE_WIDTH; ++t) {
        const int base = (write_base_token + t) * VECS_PER_TOKEN;
    write_vec_loop:
        for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS PIPELINE II=1
            vec_t<VEC_W> kv = k_in.read();
            vec_t<VEC_W> vv = v_in.read();
            hbm_k[base + v] = kv;
            hbm_v[base + v] = vv;
        }
    }
}

// ---------------------------------------------------------------------------
// Gather prefix + ancestor + self KV from contiguous HBM.
//
// Traces each query's ancestor chain internally using parent_indices_per_layer:
//   parent_indices_per_layer[l * max_tree_width + t] = slot in layer l that is
//   the parent of slot t at layer l+1.
//
// max_tree_width is the fixed stride in both the HBM layout and parent_indices array.
// Output per query: (prefix_len + current_depth + 1) tokens of KV history.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int NUM_KV_HEADS, int MAX_DEPTH>
void contiguous_kv_gather(
    const vec_t<VEC_W>* hbm_k,
    const vec_t<VEC_W>* hbm_v,
    int prefix_len,
    int current_depth,
    int max_tree_width,
    const int* parent_indices_per_layer,  // [MAX_DEPTH * max_tree_width]
    hls_stream<vec_t<VEC_W>> k_out[TREE_WIDTH],
    hls_stream<vec_t<VEC_W>> v_out[TREE_WIDTH]
) {
#pragma HLS INLINE off
    constexpr int VECS_PER_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
    static_assert((NUM_KV_HEADS * HEAD_DIM) % VEC_W == 0, "KV width must align to VEC_W");

    // Phase 1: Stream prefix tokens (shared across all queries, broadcast).
prefix_token_loop:
    for (int p = 0; p < prefix_len; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=1 avg=128 max=2048
        const int base = p * VECS_PER_TOKEN;
    prefix_vec_loop:
        for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS PIPELINE II=1
            vec_t<VEC_W> kv = hbm_k[base + v];
            vec_t<VEC_W> vv = hbm_v[base + v];
        prefix_broadcast_loop:
            for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS UNROLL
                k_out[t].write(kv);
                v_out[t].write(vv);
            }
        }
    }

    // Phase 2: Stream ancestor tokens (per-query, trace chain backward).
    // For each query t at current_depth d, trace from layer d-1 back to layer 0.
ancestor_query_loop:
    for (int t = 0; t < TREE_WIDTH; ++t) {
        int slot = t;
    ancestor_layer_loop:
        for (int l = current_depth - 1; l >= 0; --l) {
#pragma HLS LOOP_TRIPCOUNT min=0 avg=2 max=16
            int parent_slot = parent_indices_per_layer[l * max_tree_width + slot];
            int token_idx = prefix_len + l * max_tree_width + parent_slot;
            int base = token_idx * VECS_PER_TOKEN;
        ancestor_vec_loop:
            for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS PIPELINE II=1
                k_out[t].write(hbm_k[base + v]);
                v_out[t].write(hbm_v[base + v]);
            }
            slot = parent_slot;
        }
    }

    // Phase 3: Stream self tokens (one per query, just written by contiguous_kv_write).
self_token_loop:
    for (int t = 0; t < TREE_WIDTH; ++t) {
        int token_idx = prefix_len + current_depth * max_tree_width + t;
        int base = token_idx * VECS_PER_TOKEN;
    self_vec_loop:
        for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS PIPELINE II=1
            k_out[t].write(hbm_k[base + v]);
            v_out[t].write(hbm_v[base + v]);
        }
    }
}

// ---------------------------------------------------------------------------
// Combined write + gather: single DATAFLOW stage that writes new K/V to HBM
// then gathers prefix + ancestors + self for all TREE_WIDTH queries.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int NUM_KV_HEADS, int MAX_DEPTH>
void contiguous_kv_write_and_gather(
    hls_stream<vec_t<VEC_W>>& k_in,
    hls_stream<vec_t<VEC_W>>& v_in,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    int prefix_len,
    int current_depth,
    int max_tree_width,
    const int* parent_indices_per_layer,  // [MAX_DEPTH * max_tree_width]
    hls_stream<vec_t<VEC_W>> k_out[TREE_WIDTH],
    hls_stream<vec_t<VEC_W>> v_out[TREE_WIDTH]
) {
#pragma HLS INLINE off
    // Write first (so self-token is in HBM for gather phase 3).
    contiguous_kv_write<HEAD_DIM, NUM_KV_HEADS>(
        k_in, v_in, hbm_k, hbm_v,
        prefix_len + current_depth * max_tree_width);

    // Then gather.
    contiguous_kv_gather<HEAD_DIM, NUM_KV_HEADS, MAX_DEPTH>(
        hbm_k, hbm_v, prefix_len, current_depth,
        max_tree_width, parent_indices_per_layer, k_out, v_out);
}

} // namespace hls
} // namespace tmac

#endif // TMAC_CONTIGUOUS_KV_HLS_HPP
