#ifndef TMAC_COST_DRAFT_TREE_SCORE_HLS_HPP
#define TMAC_COST_DRAFT_TREE_SCORE_HLS_HPP

#include <cstdint>

namespace tmac {
namespace hls {

constexpr int kCdtSortWidth = 64;
constexpr float kCdtPadScore = -1e10f;

// Tripcount policy for HLS synthesis latency estimation.
constexpr int kCdtScoreTcBatch = 1;
constexpr int kCdtScoreTcTopK = 8;
constexpr int kCdtScoreTcTreeWidth = 4;
constexpr int kCdtScoreTcTotalTopK = kCdtScoreTcTreeWidth * kCdtScoreTcTopK; // 32
constexpr int kCdtScoreTcHidden = 4096;

inline void e4d_bitonic_sort(float scores[kCdtSortWidth],
                             int64_t indices[kCdtSortWidth],
                             int valid_count) {
#pragma HLS INLINE
bitonic_size:
    for (int size = 2; size <= kCdtSortWidth; size <<= 1) {
#pragma HLS loop_tripcount min=6 max=6 avg=6
    bitonic_stride:
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma HLS loop_tripcount min=1 max=6 avg=(1+6)/2
        bitonic_tid:
            for (int tid = 0; tid < kCdtSortWidth / 2; ++tid) {
#pragma HLS loop_tripcount min=kCdtSortWidth/2 max=kCdtSortWidth/2 avg=kCdtSortWidth/2
#pragma HLS PIPELINE II = 1
                const int i = ((tid / stride) * (stride * 2)) + (tid % stride);
                const int j = i + stride;
                if (i < valid_count && j < valid_count) {
                    const float score_i = scores[i];
                    const float score_j = scores[j];
                    const bool dir = ((i & size) == 0);
                    if ((dir && score_i < score_j) || (!dir && score_i > score_j)) {
                        scores[i] = score_j;
                        scores[j] = score_i;
                        const int64_t tmp = indices[i];
                        indices[i] = indices[j];
                        indices[j] = tmp;
                    }
                }
            }
        }
    }
}

inline int64_t e4d_hot_token_lookup(
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    int64_t token) {
#pragma HLS INLINE
    if (hot_token_id == nullptr || hot_token_vocab_size <= 0) {
        return token;
    }
    if (token < 0 || token >= hot_token_vocab_size) {
        return token;
    }
    return hot_token_id[token];
}

inline void e4d_score_core(
    const float* topk_probas_sampling,   // [batch_size, tree_width * node_top_k]
    const int64_t* topk_tokens_sampling, // [batch_size, tree_width * node_top_k] (optional)
    const float* last_layer_scores,      // [batch_size, tree_width]
    const float* input_hidden_states,    // [batch_size, tree_width, hidden_size]
    const int64_t* hot_token_id,         // [hot_token_vocab_size] (optional)
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    float* curr_layer_scores,            // [batch_size, tree_width * node_top_k]
    float* sort_layer_scores,            // [batch_size, tree_width * node_top_k]
    int64_t* sort_layer_indices,         // [batch_size, tree_width * node_top_k]
    int64_t* cache_topk_indices,         // [batch_size, node_top_k]
    int64_t* parent_indices_in_layer,    // [batch_size, node_top_k]
    float* output_hidden_states,         // [batch_size, node_top_k, hidden_size]
    int64_t* remapped_topk_tokens_sampling, // [batch_size, tree_width * node_top_k] (optional)
    int64_t* output_tokens               // [batch_size, node_top_k] (optional)
) {
#pragma HLS INLINE off
// #pragma HLS BIND_STORAGE variable=topk_probas_sampling type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=topk_tokens_sampling type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=last_layer_scores type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=input_hidden_states type=ram_2p impl=bram
// #pragma HLS ARRAY_PARTITION variable=input_hidden_states type=cyclic factor=16 dim=1
// #pragma HLS BIND_STORAGE variable=curr_layer_scores type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=sort_layer_scores type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=sort_layer_indices type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=cache_topk_indices type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=parent_indices_in_layer type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=output_hidden_states type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=remapped_topk_tokens_sampling type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=output_tokens type=ram_2p impl=bram
    const int total_topk = tree_width * node_top_k;
    if (total_topk > kCdtSortWidth || total_topk <= 0) {
        return;
    }

batch_loop:
    for (int b = 0; b < batch_size; ++b) {
#pragma HLS loop_tripcount min=kCdtScoreTcBatch max=kCdtScoreTcBatch avg=kCdtScoreTcBatch
        float s_scores[kCdtSortWidth];
        int64_t s_indices[kCdtSortWidth];
        int64_t s_tokens[kCdtSortWidth];
// #pragma HLS ARRAY_PARTITION variable = s_scores complete
// #pragma HLS ARRAY_PARTITION variable = s_indices complete
// #pragma HLS ARRAY_PARTITION variable = s_tokens complete

    init_shared:
        for (int i = 0; i < kCdtSortWidth; ++i) {
#pragma HLS UNROLL
            s_scores[i] = kCdtPadScore;
            s_indices[i] = i;
            s_tokens[i] = 0;
        }

    score_loop:
        for (int tid = 0; tid < total_topk; ++tid) {
#pragma HLS loop_tripcount min=kCdtScoreTcTotalTopK max=kCdtScoreTcTotalTopK avg=kCdtScoreTcTotalTopK
#pragma HLS PIPELINE II = 1
            const int parent_node_idx = tid / node_top_k;
            const int flat_idx = b * total_topk + tid;
            const float score = topk_probas_sampling[flat_idx] *
                                last_layer_scores[b * tree_width + parent_node_idx];
            s_scores[tid] = score;
            curr_layer_scores[flat_idx] = score;

            if (topk_tokens_sampling != nullptr) {
                int64_t token = topk_tokens_sampling[flat_idx];
                if (use_hot_token_id) {
                    token = e4d_hot_token_lookup(hot_token_id, hot_token_vocab_size, token);
                }
                s_tokens[tid] = token;
                if (remapped_topk_tokens_sampling != nullptr) {
                    remapped_topk_tokens_sampling[flat_idx] = token;
                }
            }
        }

        e4d_bitonic_sort(s_scores, s_indices, total_topk);

    write_sorted_loop:
        for (int tid = 0; tid < total_topk; ++tid) {
#pragma HLS loop_tripcount min=kCdtScoreTcTotalTopK max=kCdtScoreTcTotalTopK avg=kCdtScoreTcTotalTopK
#pragma HLS PIPELINE II = 1
            sort_layer_scores[b * total_topk + tid] = s_scores[tid];
            sort_layer_indices[b * total_topk + tid] = s_indices[tid];
            if (tid < node_top_k) {
                const int64_t best_idx = s_indices[tid];
                cache_topk_indices[b * node_top_k + tid] =
                    static_cast<int64_t>(cumu_count) + best_idx;
                parent_indices_in_layer[b * node_top_k + tid] =
                    best_idx / node_top_k;
                if (output_tokens != nullptr && topk_tokens_sampling != nullptr) {
                    output_tokens[b * node_top_k + tid] = s_tokens[best_idx];
                }
            }
        }

    gather_hidden_loop:
        for (int k = 0; k < node_top_k; ++k) {
#pragma HLS loop_tripcount min=kCdtScoreTcTopK max=kCdtScoreTcTopK avg=kCdtScoreTcTopK
            int64_t parent_idx = parent_indices_in_layer[b * node_top_k + k];
            if (parent_idx < 0) parent_idx = 0;
            if (parent_idx >= tree_width) parent_idx = tree_width - 1;

            const int64_t src_base =
                (static_cast<int64_t>(b) * tree_width + parent_idx) * hidden_size;
            const int64_t dst_base =
                (static_cast<int64_t>(b) * node_top_k + k) * hidden_size;

        copy_hidden_dim:
            for (int h = 0; h < hidden_size; ++h) {
#pragma HLS loop_tripcount min=kCdtScoreTcHidden max=kCdtScoreTcHidden avg=kCdtScoreTcHidden
#pragma HLS PIPELINE II = 1
                output_hidden_states[dst_base + h] = input_hidden_states[src_base + h];
            }
        }
    }
}

// Backward-compatible API used by the existing testbench/flow.
inline void e4d_score(
    const float* topk_probas_sampling,   // [batch_size, tree_width * node_top_k]
    const float* last_layer_scores,      // [batch_size, tree_width]
    const float* input_hidden_states,    // [batch_size, tree_width, hidden_size]
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    float* curr_layer_scores,            // [batch_size, tree_width * node_top_k]
    float* sort_layer_scores,            // [batch_size, tree_width * node_top_k]
    int64_t* sort_layer_indices,         // [batch_size, tree_width * node_top_k]
    int64_t* cache_topk_indices,         // [batch_size, node_top_k]
    int64_t* parent_indices_in_layer,    // [batch_size, node_top_k]
    float* output_hidden_states          // [batch_size, node_top_k, hidden_size]
) {
#pragma HLS INLINE
    e4d_score_core(
        topk_probas_sampling,
        nullptr,
        last_layer_scores,
        input_hidden_states,
        nullptr,
        0,
        false,
        batch_size,
        node_top_k,
        tree_width,
        hidden_size,
        cumu_count,
        curr_layer_scores,
        sort_layer_scores,
        sort_layer_indices,
        cache_topk_indices,
        parent_indices_in_layer,
        output_hidden_states,
        nullptr,
        nullptr);
}

// New API for multi-candidate adaptation: carries token path and optional hot-token remap.
inline void e4d_score_with_tokens(
    const float* topk_probas_sampling,   // [batch_size, tree_width * node_top_k]
    const int64_t* topk_tokens_sampling, // [batch_size, tree_width * node_top_k]
    const float* last_layer_scores,      // [batch_size, tree_width]
    const float* input_hidden_states,    // [batch_size, tree_width, hidden_size]
    const int64_t* hot_token_id,         // [hot_token_vocab_size]
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    float* curr_layer_scores,            // [batch_size, tree_width * node_top_k]
    float* sort_layer_scores,            // [batch_size, tree_width * node_top_k]
    int64_t* sort_layer_indices,         // [batch_size, tree_width * node_top_k]
    int64_t* cache_topk_indices,         // [batch_size, node_top_k]
    int64_t* parent_indices_in_layer,    // [batch_size, node_top_k]
    float* output_hidden_states,         // [batch_size, node_top_k, hidden_size]
    int64_t* remapped_topk_tokens_sampling, // [batch_size, tree_width * node_top_k]
    int64_t* output_tokens               // [batch_size, node_top_k]
) {
#pragma HLS INLINE
    e4d_score_core(
        topk_probas_sampling,
        topk_tokens_sampling,
        last_layer_scores,
        input_hidden_states,
        hot_token_id,
        hot_token_vocab_size,
        use_hot_token_id,
        batch_size,
        node_top_k,
        tree_width,
        hidden_size,
        cumu_count,
        curr_layer_scores,
        sort_layer_scores,
        sort_layer_indices,
        cache_topk_indices,
        parent_indices_in_layer,
        output_hidden_states,
        remapped_topk_tokens_sampling,
        output_tokens);
}

// Backward-compatible wrappers used by existing testbenches/docs.
inline void cost_draft_tree_layer_score_hls(
    const float* topk_probas_sampling,
    const float* last_layer_scores,
    const float* input_hidden_states,
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    float* curr_layer_scores,
    float* sort_layer_scores,
    int64_t* sort_layer_indices,
    int64_t* cache_topk_indices,
    int64_t* parent_indices_in_layer,
    float* output_hidden_states) {
#pragma HLS INLINE
    e4d_score(
        topk_probas_sampling,
        last_layer_scores,
        input_hidden_states,
        batch_size,
        node_top_k,
        tree_width,
        hidden_size,
        cumu_count,
        curr_layer_scores,
        sort_layer_scores,
        sort_layer_indices,
        cache_topk_indices,
        parent_indices_in_layer,
        output_hidden_states);
}

inline void cost_draft_tree_layer_score_hls_with_tokens(
    const float* topk_probas_sampling,
    const int64_t* topk_tokens_sampling,
    const float* last_layer_scores,
    const float* input_hidden_states,
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    float* curr_layer_scores,
    float* sort_layer_scores,
    int64_t* sort_layer_indices,
    int64_t* cache_topk_indices,
    int64_t* parent_indices_in_layer,
    float* output_hidden_states,
    int64_t* remapped_topk_tokens_sampling,
    int64_t* output_tokens) {
#pragma HLS INLINE
    e4d_score_with_tokens(
        topk_probas_sampling,
        topk_tokens_sampling,
        last_layer_scores,
        input_hidden_states,
        hot_token_id,
        hot_token_vocab_size,
        use_hot_token_id,
        batch_size,
        node_top_k,
        tree_width,
        hidden_size,
        cumu_count,
        curr_layer_scores,
        sort_layer_scores,
        sort_layer_indices,
        cache_topk_indices,
        parent_indices_in_layer,
        output_hidden_states,
        remapped_topk_tokens_sampling,
        output_tokens);
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_SCORE_HLS_HPP
