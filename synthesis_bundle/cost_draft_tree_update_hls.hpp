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
);

// Backward-compatible API for existing TBs/scripts.
inline void cost_draft_tree_update_state_hls(
    const float* topk_probas,
    const int64_t* topk_tokens,
    const float* sorted_scores,
    const int64_t* sorted_indexs,
    const int64_t* parent_indexs,
    const int64_t* topk_indexs,
    int batch_size,
    int node_top_k,
    int tree_width,
    int cumu_count,
    int verify_num,
    int curr_depth,
    int max_node_count,
    int max_verify_num,
    int64_t* cumu_tokens,
    float* cumu_scores,
    int64_t* cumu_deltas,
    int64_t* prev_indexs,
    int64_t* next_indexs,
    int64_t* side_indexs,
    float* output_scores,
    int64_t* output_tokens,
    float* work_scores,
    float* sort_scores) {
#pragma HLS INLINE
    e4d_update_state(
        topk_probas,
        topk_tokens,
        sorted_scores,
        sorted_indexs,
        parent_indexs,
        topk_indexs,
        batch_size,
        node_top_k,
        tree_width,
        cumu_count,
        verify_num,
        curr_depth,
        max_node_count,
        max_verify_num,
        cumu_tokens,
        cumu_scores,
        cumu_deltas,
        prev_indexs,
        next_indexs,
        side_indexs,
        output_scores,
        output_tokens,
        work_scores,
        sort_scores);
}

// Backward-compatible alias used by existing TBs/scripts.
inline void cost_draft_tree_update_state_hls(
    const float* topk_probas,
    const int64_t* topk_tokens,
    const float* sorted_scores,
    const int64_t* sorted_indexs,
    const int64_t* parent_indexs,
    const int64_t* topk_indexs,
    int batch_size,
    int node_top_k,
    int tree_width,
    int cumu_count,
    int verify_num,
    int curr_depth,
    int max_node_count,
    int max_verify_num,
    int64_t* cumu_tokens,
    float* cumu_scores,
    int64_t* cumu_deltas,
    int64_t* prev_indexs,
    int64_t* next_indexs,
    int64_t* side_indexs,
    float* output_scores,
    int64_t* output_tokens,
    float* work_scores,
    float* sort_scores) {
    e4d_update_state(
        topk_probas,
        topk_tokens,
        sorted_scores,
        sorted_indexs,
        parent_indexs,
        topk_indexs,
        batch_size,
        node_top_k,
        tree_width,
        cumu_count,
        verify_num,
        curr_depth,
        max_node_count,
        max_verify_num,
        cumu_tokens,
        cumu_scores,
        cumu_deltas,
        prev_indexs,
        next_indexs,
        side_indexs,
        output_scores,
        output_tokens,
        work_scores,
        sort_scores);
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_UPDATE_HLS_HPP
