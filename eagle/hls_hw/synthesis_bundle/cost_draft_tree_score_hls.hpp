#ifndef TMAC_COST_DRAFT_TREE_SCORE_HLS_HPP
#define TMAC_COST_DRAFT_TREE_SCORE_HLS_HPP

#include <cstdint>

namespace tmac {
namespace hls {

constexpr int kCdtSortWidth = 64;
constexpr float kCdtPadScore = -1e10f;

inline void cdt_bitonic_sort_64(float scores[kCdtSortWidth],
                                int64_t indices[kCdtSortWidth],
                                int valid_count) {
#pragma HLS INLINE
bitonic_size:
    for (int size = 2; size <= kCdtSortWidth; size <<= 1) {
    bitonic_stride:
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
        bitonic_tid:
            for (int tid = 0; tid < kCdtSortWidth / 2; ++tid) {
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

// HLS mapping for CostDraftTree draft_tree_layer_gen_kernel score path.
inline void cost_draft_tree_layer_score_hls(
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
#pragma HLS INLINE off
    const int total_topk = tree_width * node_top_k;
    if (total_topk > kCdtSortWidth || total_topk <= 0) {
        return;
    }

batch_loop:
    for (int b = 0; b < batch_size; ++b) {
        float s_scores[kCdtSortWidth];
        int64_t s_indices[kCdtSortWidth];
#pragma HLS ARRAY_PARTITION variable = s_scores complete
#pragma HLS ARRAY_PARTITION variable = s_indices complete

    init_shared:
        for (int i = 0; i < kCdtSortWidth; ++i) {
#pragma HLS UNROLL
            s_scores[i] = kCdtPadScore;
            s_indices[i] = i;
        }

    score_loop:
        for (int tid = 0; tid < total_topk; ++tid) {
#pragma HLS PIPELINE II = 1
            const int parent_node_idx = tid / node_top_k;
            const int flat_idx = b * total_topk + tid;
            const float score = topk_probas_sampling[flat_idx] *
                                last_layer_scores[b * tree_width + parent_node_idx];
            s_scores[tid] = score;
            curr_layer_scores[flat_idx] = score;
        }

        cdt_bitonic_sort_64(s_scores, s_indices, total_topk);

    write_sorted_loop:
        for (int tid = 0; tid < total_topk; ++tid) {
#pragma HLS PIPELINE II = 1
            sort_layer_scores[b * total_topk + tid] = s_scores[tid];
            sort_layer_indices[b * total_topk + tid] = s_indices[tid];
            if (tid < node_top_k) {
                const int64_t best_idx = s_indices[tid];
                cache_topk_indices[b * node_top_k + tid] =
                    static_cast<int64_t>(cumu_count) + best_idx;
                parent_indices_in_layer[b * node_top_k + tid] =
                    best_idx / node_top_k;
            }
        }

    gather_hidden_loop:
        for (int k = 0; k < node_top_k; ++k) {
            int64_t parent_idx = parent_indices_in_layer[b * node_top_k + k];
            if (parent_idx < 0) parent_idx = 0;
            if (parent_idx >= tree_width) parent_idx = tree_width - 1;

            const int64_t src_base =
                (static_cast<int64_t>(b) * tree_width + parent_idx) * hidden_size;
            const int64_t dst_base =
                (static_cast<int64_t>(b) * node_top_k + k) * hidden_size;

        copy_hidden_dim:
            for (int h = 0; h < hidden_size; ++h) {
#pragma HLS PIPELINE II = 1
                output_hidden_states[dst_base + h] = input_hidden_states[src_base + h];
            }
        }
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_SCORE_HLS_HPP
