#ifndef TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP
#define TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP

#include <cstdint>

#include "cost_draft_tree_controller_hls.hpp"
#include "cost_draft_tree_kv_cache_hls.hpp"
#include "cost_draft_tree_score_hls.hpp"
#include "cost_draft_tree_update_hls.hpp"

namespace tmac {
namespace hls {

constexpr int kCdtFusedMaxBatch = 128;
constexpr int kCdtFusedMaxNodeTopK = 16;

inline void cdt_prepare_selected_cache_locs(
    const int32_t* selected_cache_locs,
    const int64_t* cache_topk_indices,
    int batch_size,
    int node_top_k,
    int32_t* child_cache_locs // [batch_size, node_top_k]
) {
#pragma HLS INLINE off
prep_cache_loc_loop:
    for (int b = 0; b < batch_size; ++b) {
    prep_cache_loc_inner:
        for (int i = 0; i < node_top_k; ++i) {
#pragma HLS PIPELINE II = 1
            const int idx = b * node_top_k + i;
            int32_t v = -1;
            if (selected_cache_locs != nullptr) {
                v = selected_cache_locs[idx];
            } else {
                int64_t src = cache_topk_indices[idx];
                if (src > 0x7fffffffLL) {
                    src = 0x7fffffffLL;
                }
                if (src < -1) {
                    src = -1;
                }
                v = static_cast<int32_t>(src);
            }
            child_cache_locs[idx] = v;
        }
    }
}

// Gather controller-generated parent-visible KV listings into per-candidate K/V streams.
template <int HEAD_DIM, int NUM_KV_HEADS, int MAX_TREE_WIDTH, int MAX_INPUT_SIZE>
inline void cost_draft_tree_tree_kv_cache_gather_hls(
    const vec_t<VEC_W>* hbm_k_buffer,       // [kv_cache_tokens, VECS_PER_TOKEN]
    const vec_t<VEC_W>* hbm_v_buffer,       // [kv_cache_tokens, VECS_PER_TOKEN]
    const int32_t* controller_kv_indices,   // [batch, max_tree_width, max_input_size]
    const bool* controller_kv_mask,         // [batch, max_tree_width, max_input_size]
    const int* controller_kv_lens,          // [batch, max_tree_width]
    int batch_idx,
    int batch_size,
    int tree_width,
    int max_tree_width,
    int max_input_size,
    int kv_cache_tokens,
    hls_stream<vec_t<VEC_W>> k_hist_streams[MAX_TREE_WIDTH],
    hls_stream<vec_t<VEC_W>> v_hist_streams[MAX_TREE_WIDTH],
    int query_seq_lens[MAX_TREE_WIDTH]      // out: valid gathered tokens per query
) {
#pragma HLS INLINE off
    cdt_tree_kv_cache_gather_hls<HEAD_DIM, NUM_KV_HEADS, MAX_TREE_WIDTH, MAX_INPUT_SIZE>(
        hbm_k_buffer,
        hbm_v_buffer,
        controller_kv_indices,
        controller_kv_mask,
        controller_kv_lens,
        batch_idx,
        batch_size,
        tree_width,
        max_tree_width,
        max_input_size,
        kv_cache_tokens,
        k_hist_streams,
        v_hist_streams,
        query_seq_lens);
}

// Fused step wiring for one draft-tree layer in HLS:
// 1) score/sort + parent pick,
// 2) cumulative state update,
// 3) controller frontier expansion,
// 4) parent-only KV listing/mask generation.
inline void cost_draft_tree_fused_step_hls(
    // Score inputs
    const float* topk_probas_sampling,      // [batch, tree_width * node_top_k]
    const int64_t* topk_tokens_sampling,    // [batch, tree_width * node_top_k]
    const float* last_layer_scores,         // [batch, tree_width]
    const float* input_hidden_states,       // [batch, tree_width, hidden]
    const int64_t* hot_token_id,            // [vocab]
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,

    // Update-state inputs
    const int64_t* topk_indexs_prev,        // [batch, tree_width]
    const bool* input_tree_mask,            // [batch, tree_width, input_count - 1]

    // Controller inputs
    const int64_t* controller_frontier_in,  // [batch, max_tree_width]
    const int32_t* prefix_kv_locs,          // [batch, max_prefix_len]
    const int* prefix_lens,                 // [batch]
    const int32_t* selected_cache_locs,     // [batch, node_top_k] optional, fallback=cache_topk

    // Shared dims
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    int input_count,
    int verify_num,
    int curr_depth,

    // Capacity dims
    int max_input_size,
    int max_node_count,
    int max_verify_num,
    int max_tree_width,
    int max_prefix_len,
    int parent_width,
    int next_tree_width,

    // Persistent legacy draft state (update kernel output)
    int64_t* cumu_tokens,                   // [batch, max_node_count]
    float* cumu_scores,                     // [batch, max_node_count]
    int64_t* cumu_deltas,                   // [batch, max_node_count]
    int64_t* prev_indexs,                   // [batch, max_node_count]
    int64_t* next_indexs,                   // [batch, max_node_count]
    int64_t* side_indexs,                   // [batch, max_node_count]
    float* output_scores,                   // [batch, node_top_k]
    int64_t* output_tokens,                 // [batch, node_top_k]
    float* work_scores,                     // [batch, max_verify_num + node_top_k]
    float* sort_scores,                     // [batch, max_verify_num]
    bool* output_tree_mask,                 // [batch, node_top_k, max_input_size + 1]

    // Persistent controller state
    int* controller_node_count,             // [batch]
    int64_t* controller_node_token_ids,     // [batch, max_node_count]
    int64_t* controller_node_parent_ids,    // [batch, max_node_count]
    int64_t* controller_node_first_child_ids, // [batch, max_node_count]
    int64_t* controller_node_last_child_ids,  // [batch, max_node_count]
    int64_t* controller_node_next_sibling_ids, // [batch, max_node_count]
    int64_t* controller_node_depths,        // [batch, max_node_count]
    int32_t* controller_node_cache_locs,    // [batch, max_node_count]

    // Fused outputs
    float* output_hidden_states,            // [batch, node_top_k, hidden]
    int64_t* cache_topk_indices,            // [batch, node_top_k]
    int64_t* controller_frontier_out,       // [batch, max_tree_width]
    int32_t* controller_kv_indices,         // [batch, max_tree_width, max_input_size]
    bool* controller_kv_mask,               // [batch, max_tree_width, max_input_size]
    int* controller_kv_lens,                // [batch, max_tree_width]
    int64_t* controller_frontier_tokens,    // [batch, max_tree_width]
    int64_t* controller_frontier_parent_ids,// [batch, max_tree_width]
    int64_t* controller_frontier_depths,    // [batch, max_tree_width]
    int32_t* controller_frontier_cache_locs,// [batch, max_tree_width]

    // Optional debug outputs (can be nullptr)
    float* dbg_curr_layer_scores,           // [batch, tree_width * node_top_k]
    float* dbg_sort_layer_scores,           // [batch, tree_width * node_top_k]
    int64_t* dbg_sort_layer_indices,        // [batch, tree_width * node_top_k]
    int64_t* dbg_parent_indices_in_layer,   // [batch, node_top_k]
    int64_t* dbg_remapped_topk_tokens       // [batch, tree_width * node_top_k]
) {
#pragma HLS INLINE off
    const int total_topk = tree_width * node_top_k;
    if (batch_size <= 0 || batch_size > kCdtFusedMaxBatch) {
        return;
    }
    if (node_top_k <= 0 || node_top_k > kCdtFusedMaxNodeTopK) {
        return;
    }
    if (total_topk <= 0 || total_topk > kCdtSortWidth) {
        return;
    }
    if (parent_width <= 0 || parent_width > max_tree_width) {
        return;
    }
    if (next_tree_width < 0 || next_tree_width > max_tree_width) {
        return;
    }

    // Inter-stage buffers (fixed upper bounds for synthesis).
    float s_curr_layer_scores[kCdtFusedMaxBatch * kCdtSortWidth];
    float s_sort_layer_scores[kCdtFusedMaxBatch * kCdtSortWidth];
    int64_t s_sort_layer_indices[kCdtFusedMaxBatch * kCdtSortWidth];
    int64_t s_parent_indices_in_layer[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
    int64_t s_remapped_topk_tokens[kCdtFusedMaxBatch * kCdtSortWidth];
    int64_t s_selected_output_tokens[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
    int32_t s_child_cache_locs[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable = s_curr_layer_scores type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_sort_layer_scores type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_sort_layer_indices type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_parent_indices_in_layer type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_remapped_topk_tokens type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_selected_output_tokens type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_child_cache_locs type = ram_2p impl = bram

#pragma HLS DATAFLOW

    // Stage 1: score/sort + parent selection + hidden gather.
    cost_draft_tree_layer_score_hls_with_tokens(
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
        s_curr_layer_scores,
        s_sort_layer_scores,
        s_sort_layer_indices,
        cache_topk_indices,
        s_parent_indices_in_layer,
        output_hidden_states,
        s_remapped_topk_tokens,
        s_selected_output_tokens);

    // Stage 2: update cumulative draft state.
    cost_draft_tree_update_state_hls(
        topk_probas_sampling,
        s_remapped_topk_tokens,
        s_sort_layer_scores,
        s_sort_layer_indices,
        s_parent_indices_in_layer,
        topk_indexs_prev,
        input_tree_mask,
        batch_size,
        node_top_k,
        tree_width,
        input_count,
        cumu_count,
        verify_num,
        curr_depth,
        max_input_size,
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
        sort_scores,
        output_tree_mask);

    // Stage 3: controller frontier expansion.
    cdt_prepare_selected_cache_locs(
        selected_cache_locs,
        cache_topk_indices,
        batch_size,
        node_top_k,
        s_child_cache_locs);

    cdt_controller_expand_frontier(
        controller_frontier_in,
        s_parent_indices_in_layer,
        s_selected_output_tokens,
        s_child_cache_locs,
        batch_size,
        parent_width,
        next_tree_width,
        max_tree_width,
        max_node_count,
        controller_node_count,
        controller_frontier_out,
        controller_node_token_ids,
        controller_node_parent_ids,
        controller_node_first_child_ids,
        controller_node_last_child_ids,
        controller_node_next_sibling_ids,
        controller_node_depths,
        controller_node_cache_locs);

    // Stage 4: parent-only KV list/mask for next frontier.
    cdt_controller_build_parent_visible_kv(
        controller_frontier_out,
        prefix_kv_locs,
        prefix_lens,
        batch_size,
        next_tree_width,
        max_tree_width,
        max_prefix_len,
        max_input_size,
        max_node_count,
        controller_node_parent_ids,
        controller_node_cache_locs,
        controller_kv_indices,
        controller_kv_mask,
        controller_kv_lens,
        nullptr);

    cdt_controller_export_frontier(
        controller_frontier_out,
        batch_size,
        next_tree_width,
        max_tree_width,
        max_node_count,
        controller_node_token_ids,
        controller_node_parent_ids,
        controller_node_depths,
        controller_node_cache_locs,
        controller_frontier_tokens,
        controller_frontier_parent_ids,
        controller_frontier_depths,
        controller_frontier_cache_locs);

    // Optional debug copies.
    if (dbg_curr_layer_scores != nullptr || dbg_sort_layer_scores != nullptr ||
        dbg_sort_layer_indices != nullptr || dbg_remapped_topk_tokens != nullptr) {
    dbg_flat_loop:
        for (int b = 0; b < batch_size; ++b) {
        dbg_flat_inner:
            for (int t = 0; t < total_topk; ++t) {
#pragma HLS PIPELINE II = 1
                const int idx = b * total_topk + t;
                if (dbg_curr_layer_scores != nullptr) {
                    dbg_curr_layer_scores[idx] = s_curr_layer_scores[idx];
                }
                if (dbg_sort_layer_scores != nullptr) {
                    dbg_sort_layer_scores[idx] = s_sort_layer_scores[idx];
                }
                if (dbg_sort_layer_indices != nullptr) {
                    dbg_sort_layer_indices[idx] = s_sort_layer_indices[idx];
                }
                if (dbg_remapped_topk_tokens != nullptr) {
                    dbg_remapped_topk_tokens[idx] = s_remapped_topk_tokens[idx];
                }
            }
        }
    }

    if (dbg_parent_indices_in_layer != nullptr) {
    dbg_parent_loop:
        for (int b = 0; b < batch_size; ++b) {
        dbg_parent_inner:
            for (int i = 0; i < node_top_k; ++i) {
#pragma HLS PIPELINE II = 1
                dbg_parent_indices_in_layer[b * node_top_k + i] =
                    s_parent_indices_in_layer[b * node_top_k + i];
            }
        }
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP
