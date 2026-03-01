#ifndef TMAC_COST_DRAFT_TREE_KV_CACHE_HLS_HPP
#define TMAC_COST_DRAFT_TREE_KV_CACHE_HLS_HPP

#include <cstdint>

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

// Gather parent-visible KV history (tree-shaped index/mask field) into per-candidate streams.
// This bridges controller outputs:
//   - kv_indices [batch, max_tree_width, max_input_size]
//   - kv_mask    [batch, max_tree_width, max_input_size]
//   - kv_lens    [batch, max_tree_width]
// to attention-ready token-major K/V streams per candidate.
template <int HEAD_DIM, int NUM_KV_HEADS, int MAX_TREE_WIDTH, int MAX_INPUT_SIZE>
void cdt_tree_kv_cache_gather_hls(
    const vec_t<VEC_W>* hbm_k_buffer,       // [kv_cache_tokens, VECS_PER_TOKEN]
    const vec_t<VEC_W>* hbm_v_buffer,       // [kv_cache_tokens, VECS_PER_TOKEN]
    const int32_t* kv_indices,              // [batch, max_tree_width, max_input_size]
    const bool* kv_mask,                    // [batch, max_tree_width, max_input_size]
    const int* kv_lens,                     // [batch, max_tree_width]
    int batch_idx,
    int batch_size,
    int width,
    int max_tree_width,
    int max_input_size,
    int kv_cache_tokens,
    hls_stream<vec_t<VEC_W>> k_hist_streams[MAX_TREE_WIDTH],
    hls_stream<vec_t<VEC_W>> v_hist_streams[MAX_TREE_WIDTH],
    int query_seq_lens[MAX_TREE_WIDTH]      // out: emitted token counts per query
) {
#pragma HLS INLINE off
    static_assert((NUM_KV_HEADS * HEAD_DIM) % VEC_W == 0,
                  "NUM_KV_HEADS*HEAD_DIM must align to VEC_W");
    static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive");
    static_assert(NUM_KV_HEADS > 0, "NUM_KV_HEADS must be positive");
    static_assert(MAX_TREE_WIDTH > 0, "MAX_TREE_WIDTH must be positive");
    static_assert(MAX_INPUT_SIZE > 0, "MAX_INPUT_SIZE must be positive");

    constexpr int VECS_PER_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;

    int use_tree_width = max_tree_width;
    if (use_tree_width < 1) use_tree_width = 1;
    if (use_tree_width > MAX_TREE_WIDTH) use_tree_width = MAX_TREE_WIDTH;

    int use_width = width;
    if (use_width < 0) use_width = 0;
    if (use_width > use_tree_width) use_width = use_tree_width;

    int use_input_size = max_input_size;
    if (use_input_size < 1) use_input_size = 1;
    if (use_input_size > MAX_INPUT_SIZE) use_input_size = MAX_INPUT_SIZE;

    if (batch_size <= 0) {
    zero_lens_loop:
        for (int q = 0; q < MAX_TREE_WIDTH; ++q) {
#pragma HLS PIPELINE II = 1
            query_seq_lens[q] = 0;
        }
        return;
    }

    int use_b = batch_idx;
    if (use_b < 0) use_b = 0;
    if (use_b >= batch_size) use_b = batch_size - 1;

    if (kv_cache_tokens < 0) kv_cache_tokens = 0;

query_loop:
    for (int q = 0; q < MAX_TREE_WIDTH; ++q) {
        int emitted = 0;

        if (q < use_width) {
            const int len_idx = use_b * use_tree_width + q;
            int req_len = kv_lens[len_idx];
            if (req_len < 0) req_len = 0;
            if (req_len > use_input_size) req_len = use_input_size;

            const int list_base = (use_b * use_tree_width + q) * use_input_size;

        pos_loop:
            for (int p = 0; p < MAX_INPUT_SIZE; ++p) {
                if (p >= req_len || p >= use_input_size) {
                    continue;
                }
                const int flat = list_base + p;
                if (!kv_mask[flat]) {
                    continue;
                }

                const int32_t tok_idx = kv_indices[flat];
                if (tok_idx < 0 || tok_idx >= kv_cache_tokens) {
                    continue;
                }

                const int token_off = tok_idx * VECS_PER_TOKEN;
            vec_loop:
                for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS PIPELINE II = 1
                    k_hist_streams[q].write(hbm_k_buffer[token_off + v]);
                    v_hist_streams[q].write(hbm_v_buffer[token_off + v]);
                }
                ++emitted;
            }
        }

        query_seq_lens[q] = emitted;
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_KV_CACHE_HLS_HPP
