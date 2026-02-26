#ifndef TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP
#define TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP

#include <cstdint>

#include "cost_draft_tree_controller_hls.hpp"
#include "cost_draft_tree_score_hls.hpp"
#include "cost_draft_tree_update_hls.hpp"
#include "eagle_tier1_lm_top.hpp"

namespace tmac {
namespace hls {

constexpr int kCdtFusedMaxBatch = 128;
constexpr int kCdtFusedMaxNodeTopK = 16;

// Fused step wiring for one draft-tree layer in HLS:
// 1) score/sort + parent pick + hidden gather,
// 2) cumulative state update (cumu_tokens, prev/next/side_indexs, work/sort_scores).
// KV management uses contiguous HBM ancestor-chain; no tree-mask or controller needed.
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
    const int64_t* topk_indexs_prev,        // [batch, tree_width] global cumu indices

    // Shared dims
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    int verify_num,
    int curr_depth,

    // Capacity dims
    int max_node_count,
    int max_verify_num,

    // Persistent legacy draft state (in/out)
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

    // Fused outputs
    float* output_hidden_states,            // [batch, node_top_k, hidden]
    int64_t* cache_topk_indices,            // [batch, node_top_k]

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

    // Inter-stage buffers (fixed upper bounds for synthesis).
    float s_curr_layer_scores[kCdtFusedMaxBatch * kCdtSortWidth];
    float s_sort_layer_scores[kCdtFusedMaxBatch * kCdtSortWidth];
    int64_t s_sort_layer_indices[kCdtFusedMaxBatch * kCdtSortWidth];
    int64_t s_parent_indices_in_layer[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
    int64_t s_remapped_topk_tokens[kCdtFusedMaxBatch * kCdtSortWidth];
#pragma HLS BIND_STORAGE variable = s_curr_layer_scores type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_sort_layer_scores type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_sort_layer_indices type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_parent_indices_in_layer type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_remapped_topk_tokens type = ram_2p impl = bram

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
        nullptr);   // output_tokens not needed (no controller)

    // Stage 2: update cumulative draft state.
    cost_draft_tree_update_state_hls(
        topk_probas_sampling,
        s_remapped_topk_tokens,
        s_sort_layer_scores,
        s_sort_layer_indices,
        s_parent_indices_in_layer,
        topk_indexs_prev,
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

// Fixed tree-width policy helper for orchestrator usage.
// The policy keeps tree_width/verify_num unchanged and never stops early.
struct CdtFixedWidthPolicyHls {
    inline void operator()(
        int depth,
        int batch_size,
        int curr_tree_width,
        int node_top_k,
        int max_tree_width,
        int curr_verify_num,
        const float* work_scores,
        int max_verify_num,
        int* next_tree_width,
        int* next_verify_num,
        bool* stop_signal) const {
#pragma HLS INLINE
        (void)depth;
        (void)batch_size;
        (void)node_top_k;
        (void)max_tree_width;
        (void)work_scores;
        (void)max_verify_num;
        if (next_tree_width != nullptr) {
            *next_tree_width = curr_tree_width;
        }
        if (next_verify_num != nullptr) {
            *next_verify_num = curr_verify_num;
        }
        if (stop_signal != nullptr) {
            *stop_signal = false;
        }
    }
};

inline void cdt_copy_frontier_for_next_depth_hls(
    const int64_t* frontier_src,  // [batch, max_tree_width]
    int batch_size,
    int max_tree_width,
    int64_t* frontier_dst         // [batch, max_tree_width]
) {
#pragma HLS INLINE off
    if (frontier_src == nullptr || frontier_dst == nullptr || frontier_src == frontier_dst) {
        return;
    }

copy_frontier_loop_b:
    for (int b = 0; b < batch_size; ++b) {
    copy_frontier_loop_i:
        for (int i = 0; i < max_tree_width; ++i) {
#pragma HLS PIPELINE II = 1
            frontier_dst[b * max_tree_width + i] = frontier_src[b * max_tree_width + i];
        }
    }
}

// Wire per-layer outputs into the next layer's inputs.
// Selects first next_tree_width entries from node_top_k outputs and
// feeds selected hidden states and global indices back into the next SLM call.
inline void cdt_prepare_next_layer_inputs_hls(
    const float* output_scores,           // [batch, node_top_k]
    const int64_t* output_tokens,         // [batch, node_top_k]
    const float* output_hidden_states,    // [batch, node_top_k, hidden]
    const int64_t* cache_topk_indices,    // [batch, node_top_k]
    int batch_size,
    int node_top_k,
    int hidden_size,
    int next_tree_width,
    int max_tree_width,
    int64_t* next_input_tokens,           // packed [batch, next_tree_width]
    float* next_last_layer_scores,        // packed [batch, next_tree_width]
    float* next_input_hidden_states,      // packed [batch, next_tree_width, hidden]
    int64_t* next_topk_indexs_prev        // packed [batch, next_tree_width]
) {
#pragma HLS INLINE off
    if (batch_size <= 0 || node_top_k <= 0 || hidden_size <= 0 || max_tree_width <= 0) {
        return;
    }
    if (next_input_tokens == nullptr || next_last_layer_scores == nullptr ||
        next_input_hidden_states == nullptr || next_topk_indexs_prev == nullptr) {
        return;
    }

    int use_width = cdt_clamp_int(next_tree_width, 0, max_tree_width);
    use_width = cdt_clamp_int(use_width, 0, node_top_k);

next_layer_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
    next_layer_slot_loop:
        for (int t = 0; t < use_width; ++t) {
            const int token_dst = b * use_width + t;

            int64_t prev_idx_v = -1;
            if (cache_topk_indices != nullptr) {
                prev_idx_v = cache_topk_indices[b * node_top_k + t];
            }

            next_input_tokens[token_dst] = output_tokens[b * node_top_k + t];
            next_last_layer_scores[token_dst] = output_scores[b * node_top_k + t];
            next_topk_indexs_prev[token_dst] = prev_idx_v;

            const int64_t hidden_dst_base =
                (static_cast<int64_t>(b) * use_width + t) * hidden_size;

            if (output_hidden_states != nullptr) {
                const int64_t hidden_src_base =
                    (static_cast<int64_t>(b) * node_top_k + t) * hidden_size;
            next_layer_hidden_copy_loop:
                for (int h = 0; h < hidden_size; ++h) {
#pragma HLS PIPELINE II = 1
                    next_input_hidden_states[hidden_dst_base + h] =
                        output_hidden_states[hidden_src_base + h];
                }
            } else {
            next_layer_hidden_zero_loop:
                for (int h = 0; h < hidden_size; ++h) {
#pragma HLS PIPELINE II = 1
                    next_input_hidden_states[hidden_dst_base + h] = 0.0f;
                }
            }
        }
    }
}

// Run EAGLE4 SLM forward + LM-head top-k for one draft depth.
// The SLM path owns top-k candidate generation; outputs are packed to fused-step layout.
inline void cdt_run_eagle4_slm_topk_hls(
    const float* input_hidden_states,          // packed [batch, tree_width, hidden]
    int batch_size,
    int tree_width,
    int hidden_size,
    int node_top_k,
    const pack512* w_q,     const float* s_q,
    const pack512* w_k,     const float* s_k,
    const pack512* w_v,     const float* s_v,
    const pack512* w_o,     const float* s_o,
    const pack512* w_gate,  const float* gate_scales,
    const pack512* w_up,    const float* up_scales,
    const pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& rope_cfg,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    const uint16_t* efficient_lm_head_down_proj_weight,
    const int32_t* efficient_lm_head_qweight_row_major,
    const uint16_t* efficient_lm_head_scales_row_major,
    const int32_t* efficient_lm_head_qzeros,
    const int32_t* efficient_lm_head_g_idx,
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    int prefix_len,
    int current_depth,
    const int* parent_indices_per_layer,
    float* reasoning_hidden_states_out,        // packed [batch, tree_width, hidden]
    float* topk_probas_sampling_out,           // packed [batch, tree_width * node_top_k]
    int64_t* topk_tokens_sampling_out          // packed [batch, tree_width * node_top_k]
) {
#pragma HLS INLINE off
    if (input_hidden_states == nullptr || reasoning_hidden_states_out == nullptr ||
        topk_probas_sampling_out == nullptr || topk_tokens_sampling_out == nullptr) {
        return;
    }
    if (batch_size <= 0 || tree_width <= 0 || hidden_size <= 0 || node_top_k <= 0) {
        return;
    }
    if (tree_width > TREE_WIDTH || hidden_size > HIDDEN || node_top_k > kCdtFusedMaxNodeTopK ||
        node_top_k > kEagle4LmTopKMax) {
        return;
    }

slm_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
        hls::stream<vec_t<VEC_W>> hidden_in_stream("cdt_hidden_in_stream");
        hls::stream<vec_t<VEC_W>> embed_in_stream("cdt_embed_in_stream");

    slm_stream_token_loop:
        for (int t = 0; t < TREE_WIDTH; ++t) {
        slm_stream_hidden_vec_loop:
            for (int hv = 0; hv < HIDDEN / VEC_W; ++hv) {
#pragma HLS PIPELINE II = 1
                vec_t<VEC_W> hidden_vec;
                vec_t<VEC_W> embed_vec;
#pragma HLS ARRAY_PARTITION variable = hidden_vec complete
#pragma HLS ARRAY_PARTITION variable = embed_vec complete

            slm_stream_lane_loop:
                for (int lane = 0; lane < VEC_W; ++lane) {
#pragma HLS UNROLL
                    const int h = hv * VEC_W + lane;
                    float v = 0.0f;
                    if (t < tree_width && h < hidden_size) {
                        const int64_t src =
                            (static_cast<int64_t>(b) * tree_width + t) * hidden_size + h;
                        v = input_hidden_states[src];
                    }
                    hidden_vec[lane] = v;
                    // Embedding stream fallback: reuse hidden stream when embed lookup is external.
                    embed_vec[lane] = v;
                }

                hidden_in_stream.write(hidden_vec);
                embed_in_stream.write(embed_vec);
            }
        }

        int best_id = -1;
        float best_score = 0.0f;
        float reasoning_state[TREE_WIDTH * HIDDEN];
        int candidate_indices[TREE_WIDTH * kCdtFusedMaxNodeTopK];
        float gathered_logits[TREE_WIDTH * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable = reasoning_state type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = candidate_indices type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = gathered_logits type = ram_2p impl = bram

        eagle_tier1_lm_top_eagle4(
            hidden_in_stream,
            embed_in_stream,
            &best_id,
            &best_score,
            w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales, w_up, up_scales,
            w_down, down_scales,
            hidden_norm_gamma, embed_norm_gamma, post_attn_norm_gamma, final_norm_gamma,
            rope_cfg,
            hbm_k, hbm_v,
            efficient_lm_head_down_proj_weight,
            efficient_lm_head_qweight_row_major,
            efficient_lm_head_scales_row_major,
            efficient_lm_head_qzeros,
            efficient_lm_head_g_idx,
            lm_head_weight,
            efficient_lm_rank,
            efficient_lm_vocab_size,
            node_top_k,
            reasoning_state,
            candidate_indices,
            gathered_logits,
            prefix_len,
            current_depth,
            parent_indices_per_layer);

    slm_copy_hidden_loop_t:
        for (int t = 0; t < tree_width; ++t) {
        slm_copy_hidden_loop_h:
            for (int h = 0; h < hidden_size; ++h) {
#pragma HLS PIPELINE II = 1
                const int64_t dst =
                    (static_cast<int64_t>(b) * tree_width + t) * hidden_size + h;
                const int64_t src = static_cast<int64_t>(t) * HIDDEN + h;
                reasoning_hidden_states_out[dst] = reasoning_state[src];
            }
        }

    slm_copy_topk_loop_t:
        for (int t = 0; t < tree_width; ++t) {
        slm_copy_topk_loop_k:
            for (int k = 0; k < node_top_k; ++k) {
#pragma HLS PIPELINE II = 1
                const int src = t * node_top_k + k;
                const int dst = b * (tree_width * node_top_k) + src;
                topk_tokens_sampling_out[dst] = static_cast<int64_t>(candidate_indices[src]);
                topk_probas_sampling_out[dst] = gathered_logits[src];
            }
        }
    }
}

// Multi-layer orchestrator:
//   per depth:
//     1) run EAGLE4 SLM forward + LM-head top-k (`eagle_tier1_lm_top_eagle4`),
//     2) run one fused tree step (score + update),
//     3) wire fused outputs into next-layer inputs (hidden recurrence + index carry),
//     4) repeat until tree_depth or stop.
//
// KV management uses contiguous HBM ancestor-chain (parent_indices_accum);
// no controller node graph or tree mask is needed.
//
// WidthPolicy contract:
//   void operator()(
//       int depth, int batch_size, int curr_tree_width, int node_top_k, int max_tree_width,
//       int curr_verify_num, const float* work_scores, int max_verify_num,
//       int* next_tree_width, int* next_verify_num, bool* stop_signal);
template <typename WidthPolicy>
inline void cost_draft_tree_multilayer_orchestrator_hls(
    WidthPolicy width_policy,
    int tree_depth,
    int curr_depth_start,

    // Step-working buffers (in/out across depths)
    int64_t* step_input_tokens,           // packed [batch, tree_width]
    float* step_input_hidden_states,      // packed [batch, tree_width, hidden]
    float* step_last_layer_scores,        // packed [batch, tree_width]
    int64_t* step_topk_indexs_prev,       // packed [batch, tree_width]
    float* step_topk_probas_sampling,     // packed [batch, tree_width * node_top_k]
    int64_t* step_topk_tokens_sampling,   // packed [batch, tree_width * node_top_k]

    // EAGLE4 SLM + LM-head weights/buffers
    const pack512* w_q,     const float* s_q,
    const pack512* w_k,     const float* s_k,
    const pack512* w_v,     const float* s_v,
    const pack512* w_o,     const float* s_o,
    const pack512* w_gate,  const float* gate_scales,
    const pack512* w_up,    const float* up_scales,
    const pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& rope_cfg,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    const uint16_t* efficient_lm_head_down_proj_weight,
    const int32_t* efficient_lm_head_qweight_row_major,
    const uint16_t* efficient_lm_head_scales_row_major,
    const int32_t* efficient_lm_head_qzeros,
    const int32_t* efficient_lm_head_g_idx,
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,

    // Contiguous KV context
    int prefix_len,

    // Hot-token remap config
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,

    // Runtime dims (in/out)
    int batch_size,
    int node_top_k,
    int hidden_size,
    int* io_tree_width,
    int* io_verify_num,
    int* io_cumu_count,

    // Capacity dims
    int max_node_count,
    int max_verify_num,
    int max_tree_width,

    // Persistent legacy state
    int64_t* cumu_tokens,
    float* cumu_scores,
    int64_t* cumu_deltas,
    int64_t* prev_indexs,
    int64_t* next_indexs,
    int64_t* side_indexs,
    float* output_scores,
    int64_t* output_tokens,
    float* work_scores,
    float* sort_scores,

    // Per-step fused outputs / scratch
    float* output_hidden_states,          // [batch, node_top_k, hidden]
    int64_t* cache_topk_indices,          // [batch, node_top_k]

    // Optional debug outputs (final executed step contents)
    float* dbg_curr_layer_scores,
    float* dbg_sort_layer_scores,
    int64_t* dbg_sort_layer_indices,
    int64_t* dbg_parent_indices_in_layer,
    int64_t* dbg_remapped_topk_tokens,

    // Optional loop outputs
    int* executed_depths,
    bool* stopped_early
) {
#pragma HLS INLINE off
    if (tree_depth <= 0 || batch_size <= 0 || node_top_k <= 0 || hidden_size <= 0) {
        return;
    }
    if (io_tree_width == nullptr || io_verify_num == nullptr || io_cumu_count == nullptr) {
        return;
    }
    if (step_input_tokens == nullptr || step_input_hidden_states == nullptr ||
        step_last_layer_scores == nullptr || step_topk_indexs_prev == nullptr ||
        step_topk_probas_sampling == nullptr || step_topk_tokens_sampling == nullptr ||
        output_hidden_states == nullptr || cache_topk_indices == nullptr ||
        output_scores == nullptr || output_tokens == nullptr) {
        return;
    }
    if (w_q == nullptr || s_q == nullptr || w_k == nullptr || s_k == nullptr ||
        w_v == nullptr || s_v == nullptr || w_o == nullptr || s_o == nullptr ||
        w_gate == nullptr || gate_scales == nullptr || w_up == nullptr || up_scales == nullptr ||
        w_down == nullptr || down_scales == nullptr || hidden_norm_gamma == nullptr ||
        embed_norm_gamma == nullptr || post_attn_norm_gamma == nullptr ||
        final_norm_gamma == nullptr || hbm_k == nullptr || hbm_v == nullptr ||
        efficient_lm_head_down_proj_weight == nullptr ||
        efficient_lm_head_qweight_row_major == nullptr ||
        efficient_lm_head_scales_row_major == nullptr || lm_head_weight == nullptr) {
        return;
    }
    if (hidden_size > HIDDEN || max_tree_width > TREE_WIDTH ||
        node_top_k > kCdtFusedMaxNodeTopK || node_top_k > kEagle4LmTopKMax) {
        return;
    }

    int curr_tree_width = cdt_clamp_int(*io_tree_width, 0, max_tree_width);
    curr_tree_width = cdt_clamp_int(curr_tree_width, 0, node_top_k);
    int curr_verify_num = cdt_clamp_int(*io_verify_num, 1, max_verify_num);
    int curr_cumu_count = cdt_clamp_int(*io_cumu_count, 0, max_node_count);

    int depth_done = 0;
    bool stopped = false;

    // Accumulated parent indices: parent_indices_accum[l * TREE_WIDTH + t] =
    //   slot in layer l that is the parent of slot t at layer l+1.
    // Indexed by absolute depth (curr_depth_start + d), stride TREE_WIDTH.
    int parent_indices_accum[kCdtControllerMaxDepth * TREE_WIDTH];
#pragma HLS BIND_STORAGE variable = parent_indices_accum type = ram_2p impl = bram
    for (int i = 0; i < kCdtControllerMaxDepth * TREE_WIDTH; ++i) {
#pragma HLS PIPELINE II = 1
        parent_indices_accum[i] = 0;
    }

    // Internal scratch: receives parent slot indices from the fused step on every depth.
    // Using a dedicated buffer (always non-null) guarantees the step writes parent indices
    // even when the caller does not supply a debug output pointer.
    int64_t s_parent_scratch[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable = s_parent_scratch type = ram_2p impl = bram

orchestrator_depth_loop:
    for (int d = 0; d < tree_depth; ++d) {
        if (curr_tree_width <= 0) {
            stopped = true;
            break;
        }

        int next_tree_width = curr_tree_width;
        int next_verify_num = curr_verify_num;
        bool stop_signal = false;

        width_policy(
            d,
            batch_size,
            curr_tree_width,
            node_top_k,
            max_tree_width,
            curr_verify_num,
            work_scores,
            max_verify_num,
            &next_tree_width,
            &next_verify_num,
            &stop_signal);

        next_tree_width = cdt_clamp_int(next_tree_width, 0, max_tree_width);
        next_tree_width = cdt_clamp_int(next_tree_width, 0, node_top_k);
        next_verify_num = cdt_clamp_int(next_verify_num, 1, max_verify_num);

        // Stage A/B: SLM forward and LM-head top-k for current frontier.
        const int current_depth = curr_depth_start + d;
        cdt_run_eagle4_slm_topk_hls(
            step_input_hidden_states,
            batch_size,
            curr_tree_width,
            hidden_size,
            node_top_k,
            w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales, w_up, up_scales,
            w_down, down_scales,
            hidden_norm_gamma, embed_norm_gamma, post_attn_norm_gamma, final_norm_gamma,
            rope_cfg,
            hbm_k, hbm_v,
            efficient_lm_head_down_proj_weight,
            efficient_lm_head_qweight_row_major,
            efficient_lm_head_scales_row_major,
            efficient_lm_head_qzeros,
            efficient_lm_head_g_idx,
            lm_head_weight,
            efficient_lm_rank,
            efficient_lm_vocab_size,
            prefix_len,
            current_depth,
            parent_indices_accum,
            step_input_hidden_states,
            step_topk_probas_sampling,
            step_topk_tokens_sampling);

        // Stage C: fused score/update for one depth.
        cost_draft_tree_fused_step_hls(
            step_topk_probas_sampling,
            step_topk_tokens_sampling,
            step_last_layer_scores,
            step_input_hidden_states,
            hot_token_id,
            hot_token_vocab_size,
            use_hot_token_id,
            step_topk_indexs_prev,
            batch_size,
            node_top_k,
            curr_tree_width,
            hidden_size,
            curr_cumu_count,
            curr_verify_num,
            curr_depth_start + d,
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
            output_hidden_states,
            cache_topk_indices,
            dbg_curr_layer_scores,
            dbg_sort_layer_scores,
            dbg_sort_layer_indices,
            s_parent_scratch,        // always non-null: step always writes parent indices here
            dbg_remapped_topk_tokens);

        // Always accumulate parent indices for contiguous KV gather at next depth.
        // s_parent_scratch holds [batch, node_top_k] parent slot indices written by the step above.
        // We use batch-0 entries (multi-batch KV ancestor support deferred to a future pass).
        // Bounds-guard against curr_depth_start overflow of parent_indices_accum.
        if (current_depth >= 0 && current_depth < kCdtControllerMaxDepth) {
        accum_parent_loop:
            for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS PIPELINE II = 1
                int parent_slot = 0;
                if (t < node_top_k) {
                    int64_t v = s_parent_scratch[t];  // batch-0 entry
                    parent_slot = (v >= 0 && v < max_tree_width) ? static_cast<int>(v) : 0;
                }
                parent_indices_accum[current_depth * TREE_WIDTH + t] = parent_slot;
            }
        }
        // Forward to caller's debug output if requested.
        if (dbg_parent_indices_in_layer != nullptr) {
        dbg_parent_fwd_loop_b:
            for (int b = 0; b < batch_size; ++b) {
            dbg_parent_fwd_loop_i:
                for (int i = 0; i < node_top_k; ++i) {
#pragma HLS PIPELINE II = 1
                    dbg_parent_indices_in_layer[b * node_top_k + i] =
                        s_parent_scratch[b * node_top_k + i];
                }
            }
        }

        curr_cumu_count += curr_tree_width * node_top_k;
        if (curr_cumu_count > max_node_count) {
            curr_cumu_count = max_node_count;
        }

        ++depth_done;
        if (d + 1 >= tree_depth || stop_signal || next_tree_width <= 0) {
            stopped = stop_signal || (next_tree_width <= 0);
            break;
        }

        // Stage D: recurrence wiring for next SLM call.
        cdt_prepare_next_layer_inputs_hls(
            output_scores,
            output_tokens,
            output_hidden_states,
            cache_topk_indices,
            batch_size,
            node_top_k,
            hidden_size,
            next_tree_width,
            max_tree_width,
            step_input_tokens,
            step_last_layer_scores,
            step_input_hidden_states,
            step_topk_indexs_prev);

        curr_tree_width = next_tree_width;
        curr_verify_num = next_verify_num;
    }

    *io_tree_width = curr_tree_width;
    *io_verify_num = curr_verify_num;
    *io_cumu_count = curr_cumu_count;
    if (executed_depths != nullptr) {
        *executed_depths = depth_done;
    }
    if (stopped_early != nullptr) {
        *stopped_early = stopped;
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP
