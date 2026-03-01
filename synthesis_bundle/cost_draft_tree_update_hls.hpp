#ifndef TMAC_COST_DRAFT_TREE_UPDATE_HLS_HPP
#define TMAC_COST_DRAFT_TREE_UPDATE_HLS_HPP

#include <cstdint>

namespace tmac {
namespace hls {

constexpr float kCdtUpdatePadScore = -1e10f;
constexpr int kCdtUpdateMergeMax = 256;

// Tripcount policy for HLS synthesis latency estimation.
constexpr int kCdtUpdateTcBatch = 1;
constexpr int kCdtUpdateTcTopK = 8;
constexpr int kCdtUpdateTcTreeWidth = 4;
constexpr int kCdtUpdateTcTotalTopK = kCdtUpdateTcTreeWidth * kCdtUpdateTcTopK; // 32
constexpr int kCdtUpdateTcMaxVerifyNum = 64;

inline int e4d_min(int a, int b) {
#pragma HLS INLINE
    return (a < b) ? a : b;
}

inline int64_t e4d_safe_index(int64_t idx, int64_t low, int64_t high, int64_t fallback) {
#pragma HLS INLINE
    if (idx < low || idx >= high) {
        return fallback;
    }
    return idx;
}

// HLS mapping for update_cumu_draft_state kernel path.
void e4d_update_state(
    const float* topk_probas,         // [batch_size, tree_width * node_top_k]
    const int64_t* topk_tokens,       // [batch_size, tree_width * node_top_k]
    const float* sorted_scores,       // [batch_size, tree_width * node_top_k]
    const int64_t* sorted_indexs,     // [batch_size, tree_width * node_top_k]
    const int64_t* parent_indexs,     // [batch_size, node_top_k]
    const int64_t* topk_indexs,       // [batch_size, tree_width]
    int batch_size,
    int node_top_k,
    int tree_width,
    int cumu_count,
    int verify_num,
    int curr_depth,
    int max_node_count,
    int max_verify_num,
    int64_t* cumu_tokens,             // [batch_size, max_node_count]
    float* cumu_scores,               // [batch_size, max_node_count]
    int64_t* cumu_deltas,             // [batch_size, max_node_count]
    int64_t* prev_indexs,             // [batch_size, max_node_count]
    int64_t* next_indexs,             // [batch_size, max_node_count]
    int64_t* side_indexs,             // [batch_size, max_node_count]
    float* output_scores,             // [batch_size, node_top_k]
    int64_t* output_tokens,           // [batch_size, node_top_k]
    float* work_scores,               // [batch_size, max_verify_num + node_top_k]
    float* sort_scores                // [batch_size, max_verify_num]
) {
#pragma HLS INLINE off
    if (batch_size <= 0 || node_top_k <= 0 || tree_width <= 0 ||
        max_node_count <= 0 || max_verify_num <= 0) {
        return;
    }

    const int num_new_tokens = tree_width * node_top_k;
    if (num_new_tokens <= 0) {
        return;
    }
    if (verify_num <= 0) {
        verify_num = 1;
    }
    if (verify_num > max_verify_num) {
        verify_num = max_verify_num;
    }

batch_loop:
    for (int b = 0; b < batch_size; ++b) {
#pragma HLS loop_tripcount min=kCdtUpdateTcBatch max=kCdtUpdateTcBatch avg=kCdtUpdateTcBatch
        const int topk_offset = b * num_new_tokens;
        const int parent_offset = b * node_top_k;
        const int topk_indexs_offset = b * tree_width;
        const int output_offset = b * node_top_k;
        const int node_offset = b * max_node_count;
        const int verify_offset = b * max_verify_num;
        const int work_offset = b * (max_verify_num + node_top_k);

        // 1) Update output_scores and output_tokens.
    output_topk_loop:
        for (int i = 0; i < node_top_k; ++i) {
#pragma HLS loop_tripcount min=kCdtUpdateTcTopK max=kCdtUpdateTcTopK avg=kCdtUpdateTcTopK
#pragma HLS PIPELINE II = 1
            output_scores[output_offset + i] = sorted_scores[topk_offset + i];

            int64_t parent_idx = parent_indexs[parent_offset + i];
            parent_idx = e4d_safe_index(parent_idx, 0, tree_width, 0);

            int64_t original_idx = sorted_indexs[topk_offset + i];
            if (original_idx < 0) {
                original_idx = 0;
            }
            const int64_t child_idx = original_idx % node_top_k;
            const int64_t tok_idx =
                topk_offset + parent_idx * node_top_k + child_idx;
            output_tokens[output_offset + i] = topk_tokens[tok_idx];
        }

        // 2) Update cumulative tensors and in-layer links.
        const int start = cumu_count;
    update_new_nodes_loop:
        for (int i = 0; i < num_new_tokens; ++i) {
#pragma HLS loop_tripcount min=kCdtUpdateTcTotalTopK max=kCdtUpdateTcTotalTopK avg=kCdtUpdateTcTotalTopK
#pragma HLS PIPELINE II = 1
            const int global_idx = start + i;
            if (global_idx < max_node_count) {
                cumu_tokens[node_offset + global_idx] = topk_tokens[topk_offset + i];
                cumu_scores[node_offset + global_idx] = topk_probas[topk_offset + i] * 0.9999f;
                cumu_deltas[node_offset + global_idx] = curr_depth;

                const int parent_node_idx_in_tree = i / node_top_k;
                prev_indexs[node_offset + global_idx] =
                    topk_indexs[topk_indexs_offset + parent_node_idx_in_tree];

                next_indexs[node_offset + global_idx] = -1;
                const int child_idx_in_node = i % node_top_k;
                side_indexs[node_offset + global_idx] =
                    (child_idx_in_node == node_top_k - 1) ? -1 : (global_idx + 1);
            }
        }

        // 3) Update parent next pointers.
    update_parent_next_loop:
        for (int i = 0; i < tree_width; ++i) {
#pragma HLS loop_tripcount min=kCdtUpdateTcTreeWidth max=kCdtUpdateTcTreeWidth avg=kCdtUpdateTcTreeWidth
#pragma HLS PIPELINE II = 1
            const int64_t parent_global_idx = topk_indexs[topk_indexs_offset + i];
            if (parent_global_idx >= 0 && parent_global_idx < max_node_count) {
                next_indexs[node_offset + parent_global_idx] = start + i * node_top_k;
            }
        }

        // 4a) Update work_scores prefix.
        const int work_size_0 = e4d_min(verify_num, cumu_count);
    work_scores_old_loop:
        for (int i = 0; i < work_size_0; ++i) {
#pragma HLS loop_tripcount min=1 max=kCdtUpdateTcMaxVerifyNum avg=(1+kCdtUpdateTcMaxVerifyNum)/2
#pragma HLS PIPELINE II = 1
            work_scores[work_offset + i] = sort_scores[verify_offset + i];
        }
    work_scores_new_loop:
        for (int i = 0; i < node_top_k; ++i) {
#pragma HLS loop_tripcount min=kCdtUpdateTcTopK max=kCdtUpdateTcTopK avg=kCdtUpdateTcTopK
#pragma HLS PIPELINE II = 1
            work_scores[work_offset + work_size_0 + i] = output_scores[output_offset + i];
        }

        // 4b) Merge two descending segments into sort_scores prefix.
        // left: sort_scores[:work_size_0], right: sorted_scores[:num_new_tokens]
        const int work_size_1 = e4d_min(verify_num, cumu_count + num_new_tokens);

        if (work_size_1 > kCdtUpdateMergeMax) {
            // Keep behavior defined under compile-time bound.
            continue;
        }

        float merged_top[kCdtUpdateMergeMax];
#pragma HLS ARRAY_PARTITION variable = merged_top cyclic factor = 8

        int ia = 0;
        int ib = 0;
    merge_loop:
        for (int i = 0; i < work_size_1; ++i) {
#pragma HLS loop_tripcount min=1 max=kCdtUpdateTcMaxVerifyNum avg=(1+kCdtUpdateTcMaxVerifyNum)/2
//#pragma HLS PIPELINE II = 1
            const bool has_a = (ia < work_size_0);
            const bool has_b = (ib < num_new_tokens);
            const float a = has_a ? sort_scores[verify_offset + ia] : kCdtUpdatePadScore;
            const float bscore = has_b ? sorted_scores[topk_offset + ib] : kCdtUpdatePadScore;

            if (has_a && (!has_b || a >= bscore)) {
                merged_top[i] = a;
                ++ia;
            } else {
                merged_top[i] = bscore;
                ++ib;
            }
        }

    write_sort_scores_loop:
        for (int i = 0; i < work_size_1; ++i) {
#pragma HLS loop_tripcount min=1 max=kCdtUpdateTcMaxVerifyNum avg=(1+kCdtUpdateTcMaxVerifyNum)/2
#pragma HLS PIPELINE II = 1
            sort_scores[verify_offset + i] = merged_top[i];
        }
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_UPDATE_HLS_HPP
