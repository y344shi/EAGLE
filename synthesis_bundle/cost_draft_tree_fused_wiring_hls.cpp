#include "cost_draft_tree_fused_wiring_hls.hpp"
#ifndef __SYNTHESIS__
#include <cstdio>
#endif

namespace tmac {
namespace hls {

// Fused step wiring for one draft-tree layer in HLS:
// 1) score/sort + parent pick + hidden gather,
// 2) cumulative state update (cumu_tokens, prev/next/side_indexs, work/sort_scores).
// KV management uses contiguous HBM ancestor-chain; no tree-mask or controller needed.
void e4d_fused_step(
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
        s_curr_layer_scores,
        s_sort_layer_scores,
        s_sort_layer_indices,
        cache_topk_indices,
        s_parent_indices_in_layer,
        output_hidden_states,
        s_remapped_topk_tokens,
        nullptr);   // output_tokens not needed (no controller)

    // Stage 2: update cumulative draft state.
    e4d_update_state(
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
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
        dbg_flat_inner:
            for (int t = 0; t < total_topk; ++t) {
#pragma HLS loop_tripcount min=kTcTotalTopK max=kTcTotalTopK avg=kTcTotalTopK
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
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
        dbg_parent_inner:
            for (int i = 0; i < node_top_k; ++i) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
#pragma HLS PIPELINE II = 1
                dbg_parent_indices_in_layer[b * node_top_k + i] =
                    s_parent_indices_in_layer[b * node_top_k + i];
            }
        }
    }
}

// Select top-k from logits and output softmax probabilities for those winners.
// If candidate_indices is provided, winner indices are remapped to real vocab token IDs.
void e4d_softmax_topk(
    const float* logits,                 // [batch, logits_width]
    const int64_t* candidate_indices,    // [batch, logits_width] optional
    int batch_size,
    int logits_width,
    int node_top_k,
    float* topk_probas_out,              // [batch, node_top_k]
    int64_t* topk_tokens_out             // [batch, node_top_k]
) {
#pragma HLS INLINE off
    if (logits == nullptr || topk_probas_out == nullptr || topk_tokens_out == nullptr) {
        return;
    }
    if (batch_size <= 0 || batch_size > kCdtFusedMaxBatch || logits_width <= 0 ||
        node_top_k <= 0 || node_top_k > kCdtFusedMaxNodeTopK) {
        return;
    }

topk_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
        float best_logits[kCdtFusedMaxNodeTopK];
        int best_indices[kCdtFusedMaxNodeTopK];
#pragma HLS ARRAY_PARTITION variable = best_logits complete
#pragma HLS ARRAY_PARTITION variable = best_indices complete

    topk_init_loop:
        for (int k = 0; k < node_top_k; ++k) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
            best_logits[k] = -std::numeric_limits<float>::infinity();
            best_indices[k] = -1;
        }

        float max_logit = -std::numeric_limits<float>::infinity();
    topk_scan_loop:
        for (int i = 0; i < logits_width; ++i) {
#pragma HLS loop_tripcount min=kTcLogitsWidth max=kTcLogitsWidth avg=kTcLogitsWidth
#pragma HLS PIPELINE II = 1
            const float v = logits[b * logits_width + i];
            if (v > max_logit) {
                max_logit = v;
            }

            int min_pos = 0;
            float min_val = best_logits[0];
        topk_find_min_loop:
            for (int k = 1; k < node_top_k; ++k) {
#pragma HLS loop_tripcount min=kTcTopK-1 max=kTcTopK-1 avg=kTcTopK-1
                if (best_logits[k] < min_val) {
                    min_val = best_logits[k];
                    min_pos = k;
                }
            }
            if (v > min_val) {
                best_logits[min_pos] = v;
                best_indices[min_pos] = i;
            }
        }

    topk_sort_loop_i:
        for (int i = 0; i < node_top_k; ++i) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
            int best_pos = i;
            float best_val = best_logits[i];
        topk_sort_loop_j:
            for (int j = i + 1; j < node_top_k; ++j) {
#pragma HLS loop_tripcount min=1 max=kTcTopK-1 avg=(1+kTcTopK-1)/2
                if (best_logits[j] > best_val) {
                    best_val = best_logits[j];
                    best_pos = j;
                }
            }
            if (best_pos != i) {
                const float tmp_v = best_logits[i];
                best_logits[i] = best_logits[best_pos];
                best_logits[best_pos] = tmp_v;
                const int tmp_i = best_indices[i];
                best_indices[i] = best_indices[best_pos];
                best_indices[best_pos] = tmp_i;
            }
        }

        float sum_exp = 0.0f;
    topk_sumexp_loop:
        for (int i = 0; i < logits_width; ++i) {
#pragma HLS loop_tripcount min=kTcLogitsWidth max=kTcLogitsWidth avg=kTcLogitsWidth
#pragma HLS PIPELINE II = 1
            sum_exp += std::exp(logits[b * logits_width + i] - max_logit);
        }
        const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    topk_write_loop:
        for (int k = 0; k < node_top_k; ++k) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
#pragma HLS PIPELINE II = 1
            int idx = best_indices[k];
            if (idx < 0 || idx >= logits_width) {
                idx = 0;
            }
            topk_probas_out[b * node_top_k + k] =
                std::exp(best_logits[k] - max_logit) * inv_sum;

            int64_t tok = idx;
            if (candidate_indices != nullptr) {
                tok = candidate_indices[b * logits_width + idx];
            }
            topk_tokens_out[b * node_top_k + k] = tok;
        }
    }
}

void e4d_copy_frontier(
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
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
    copy_frontier_loop_i:
        for (int i = 0; i < max_tree_width; ++i) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
#pragma HLS PIPELINE II = 1
            frontier_dst[b * max_tree_width + i] = frontier_src[b * max_tree_width + i];
        }
    }
}

// Wire per-layer outputs into the next layer's inputs.
// Selects first next_tree_width entries from node_top_k outputs and
// feeds selected hidden states and global indices back into the next SLM call.
void e4d_prep_next_inputs(
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

    int use_width = e4d_clamp_int(next_tree_width, 0, max_tree_width);
    use_width = e4d_clamp_int(use_width, 0, node_top_k);

next_layer_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
    next_layer_slot_loop:
        for (int t = 0; t < use_width; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
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
#pragma HLS loop_tripcount min=HIDDEN max=HIDDEN avg=HIDDEN
#pragma HLS PIPELINE II = 1
                    next_input_hidden_states[hidden_dst_base + h] =
                        output_hidden_states[hidden_src_base + h];
                }
            } else {
            next_layer_hidden_zero_loop:
                for (int h = 0; h < hidden_size; ++h) {
#pragma HLS loop_tripcount min=HIDDEN max=HIDDEN avg=HIDDEN
#pragma HLS PIPELINE II = 1
                    next_input_hidden_states[hidden_dst_base + h] = 0.0f;
                }
            }
        }
    }
}

// Run EAGLE4 SLM forward + LM-head top-k for one draft depth.
// The SLM path owns top-k candidate generation; outputs are packed to fused-step layout.
void e4d_slm_topk(
    const float* input_hidden_states,          // packed [batch, tree_width, hidden]
    const float* input_embed_states,           // packed [batch, tree_width, hidden] or nullptr (falls back to hidden)
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
    const float rope_cos_vals[HEAD_DIM / 2],
    const float rope_sin_vals[HEAD_DIM / 2],
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    const uint16_t efficient_lm_head_down_proj_weight[tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax],
    const int32_t efficient_lm_head_qweight_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks],
    const uint16_t efficient_lm_head_scales_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_qzeros[tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_g_idx[tmac::hls::kEagle4LmRankMax],
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
#pragma HLS BIND_STORAGE variable=parent_indices_per_layer type=ram_2p impl=bram
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
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
        hls_stream<vec_t<VEC_W>> hidden_in_stream("cdt_hidden_in_stream");
        hls_stream<vec_t<VEC_W>> embed_in_stream("cdt_embed_in_stream");

    slm_stream_token_loop:
        for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
        slm_stream_hidden_vec_loop:
            for (int hv = 0; hv < HIDDEN / VEC_W; ++hv) {
#pragma HLS loop_tripcount min=HIDDEN/VEC_W max=HIDDEN/VEC_W avg=HIDDEN/VEC_W
#pragma HLS PIPELINE II = 1
                vec_t<VEC_W> hidden_vec;
                vec_t<VEC_W> embed_vec;
#pragma HLS ARRAY_PARTITION variable = hidden_vec complete
#pragma HLS ARRAY_PARTITION variable = embed_vec complete

            slm_stream_lane_loop:
                for (int lane = 0; lane < VEC_W; ++lane) {
#pragma HLS UNROLL
                    const int h = hv * VEC_W + lane;
                    float hv_val = 0.0f;
                    float ev_val = 0.0f;
                    if (t < tree_width && h < hidden_size) {
                        const int64_t src =
                            (static_cast<int64_t>(b) * tree_width + t) * hidden_size + h;
                        hv_val = input_hidden_states[src];
                        ev_val = (input_embed_states != nullptr)
                                     ? input_embed_states[src]
                                     : hv_val;
                    }
                    hidden_vec[lane] = hv_val;
                    embed_vec[lane] = ev_val;
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
            rope_cos_vals, rope_sin_vals,
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
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
        slm_copy_hidden_loop_h:
            for (int h = 0; h < hidden_size; ++h) {
#pragma HLS loop_tripcount min=HIDDEN max=HIDDEN avg=HIDDEN
#pragma HLS PIPELINE II = 1
                const int64_t dst =
                    (static_cast<int64_t>(b) * tree_width + t) * hidden_size + h;
                const int64_t src = static_cast<int64_t>(t) * HIDDEN + h;
                reasoning_hidden_states_out[dst] = reasoning_state[src];
            }
        }

    slm_copy_topk_loop_t:
        for (int t = 0; t < tree_width; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
            float max_logit = -1.0e30f;
        slm_topk_max_loop_k:
            for (int k = 0; k < node_top_k; ++k) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
#pragma HLS PIPELINE II = 1
                const int src = t * node_top_k + k;
                const float v = gathered_logits[src];
                if (v > max_logit) {
                    max_logit = v;
                }
            }

            float exp_vals[kCdtFusedMaxNodeTopK];
#pragma HLS ARRAY_PARTITION variable = exp_vals complete
            float sum_exp = 0.0f;
        slm_topk_exp_loop_k:
            for (int k = 0; k < node_top_k; ++k) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
#pragma HLS PIPELINE II = 1
                const int src = t * node_top_k + k;
                const float e = std::exp(gathered_logits[src] - max_logit);
                exp_vals[k] = e;
                sum_exp += e;
            }
            const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

        slm_copy_topk_loop_k:
            for (int k = 0; k < node_top_k; ++k) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
#pragma HLS PIPELINE II = 1
                const int src = t * node_top_k + k;
                const int dst = b * (tree_width * node_top_k) + src;
                topk_tokens_sampling_out[dst] = static_cast<int64_t>(candidate_indices[src]);
                topk_probas_sampling_out[dst] = exp_vals[k] * inv_sum;
            }
        }
    }
}

// Multi-layer orchestrator:
//   optional InitialLoop (PyTorch draft_InitialLoop parity):
//     0) from previous-verify logits -> softmax+top-k (or caller-provided initial top-k),
//     1) run one fused tree step with tree_width=1,
//     2) call width policy for depth-0 and seed the first recurrent frontier.
//   recurrent loop per depth:
//     1) run EAGLE4 SLM forward + LM-head top-k (`eagle_tier1_lm_top_eagle4`),
//     2) run one fused tree step (score + update),
//     3) wire fused outputs into next-layer inputs (hidden recurrence + index carry),
//     4) repeat until tree_depth or stop.
//
// KV management uses contiguous HBM ancestor-chain (parent_indices_accum);
// no controller node graph or tree mask is needed.
//
// Policy schedule contract:
//   If use_policy_schedule=true and schedule arrays are non-null, depth-wise policy comes from:
//     policy_next_tree_width[depth], policy_next_verify_num[depth], policy_stop_signal[depth].
//   Otherwise, fixed policy is used: keep curr_tree_width/curr_verify_num and never stop.
void e4d_apply_policy(
    int depth,
    int batch_size,
    int curr_tree_width,
    int node_top_k,
    int max_tree_width,
    int curr_verify_num,
    const float* work_scores,
    int max_verify_num,
    const int* policy_next_tree_width,
    const int* policy_next_verify_num,
    const int* policy_stop_signal,
    int policy_depth,
    bool use_policy_schedule,
    int* next_tree_width,
    int* next_verify_num,
    bool* stop_signal) {
#pragma HLS INLINE
    (void)batch_size;
    (void)node_top_k;
    (void)max_tree_width;
    (void)work_scores;
    (void)max_verify_num;
    int scheduled_tree_width = curr_tree_width;
    int scheduled_verify_num = curr_verify_num;
    bool scheduled_stop = false;
    if (use_policy_schedule && policy_next_tree_width != nullptr &&
        policy_next_verify_num != nullptr && policy_stop_signal != nullptr &&
        depth >= 0 && depth < policy_depth) {
        scheduled_tree_width = policy_next_tree_width[depth];
        scheduled_verify_num = policy_next_verify_num[depth];
        scheduled_stop = (policy_stop_signal[depth] != 0);
    }
    if (next_tree_width != nullptr) {
        *next_tree_width = scheduled_tree_width;
    }
    if (next_verify_num != nullptr) {
        *next_verify_num = scheduled_verify_num;
    }
    if (stop_signal != nullptr) {
        *stop_signal = scheduled_stop;
    }
}

// Draft-prefill projection: [batch=1, 3H] -> [batch=1, H] using existing LUTMAC dense projection.
void e4d_prefill_fc(
    const float* input_hidden_states_3h,   // [batch, 3 * hidden]
    const pack512* fc_weight,              // packed [3*hidden -> hidden]
    const float* fc_scales,                // grouped scales
    int batch_size,
    int hidden_size,
    float* projected_hidden_states) {      // [batch, hidden]
#pragma HLS INLINE off
    if (input_hidden_states_3h == nullptr || fc_weight == nullptr ||
        fc_scales == nullptr || projected_hidden_states == nullptr) {
        return;
    }
    // Current pipeline milestone assumes batch_size=1 and hidden_size==HIDDEN.
    if (batch_size != 1 || hidden_size != HIDDEN) {
        return;
    }

    hls_stream<vec_t<VEC_W>> s_fc_in("s_fc_in");
    hls_stream<vec_t<VEC_W>> s_fc_out("s_fc_out");

prefill_fc_stream_in_loop:
    for (int hv = 0; hv < (HIDDEN * 3) / VEC_W; ++hv) {
#pragma HLS loop_tripcount min=(HIDDEN*3)/VEC_W max=(HIDDEN*3)/VEC_W avg=(HIDDEN*3)/VEC_W
#pragma HLS PIPELINE II=1
        vec_t<VEC_W> v;
#pragma HLS ARRAY_PARTITION variable=v complete
    prefill_fc_stream_in_lane:
        for (int lane = 0; lane < VEC_W; ++lane) {
#pragma HLS UNROLL
            const int idx = hv * VEC_W + lane;
            v[lane] = input_hidden_states_3h[idx];
        }
        s_fc_in.write(v);
    }

    dense_projection_production_scaled_batched<0, 1, HIDDEN * 3, HIDDEN, 128, TMAC_USE_TMAC_QKV>(
        s_fc_in,
        s_fc_out,
        fc_weight,
        fc_scales);

prefill_fc_stream_out_loop:
    for (int hv = 0; hv < HIDDEN / VEC_W; ++hv) {
#pragma HLS loop_tripcount min=HIDDEN/VEC_W max=HIDDEN/VEC_W avg=HIDDEN/VEC_W
#pragma HLS PIPELINE II=1
        vec_t<VEC_W> v = s_fc_out.read();
    prefill_fc_stream_out_lane:
        for (int lane = 0; lane < VEC_W; ++lane) {
#pragma HLS UNROLL
            const int idx = hv * VEC_W + lane;
            projected_hidden_states[idx] = v[lane];
        }
    }
}

// Build recurrent embed states from token IDs using fp16 embed table.
// Returns false if any token ID is invalid for the embed vocab.
bool e4d_lookup_recurrent_embed_states(
    const int64_t* step_input_tokens,          // packed [batch, tree_width]
    int batch_size,
    int tree_width,
    int hidden_size,
    const uint16_t* draft_embed_tokens_weight, // fp16 [kEagle4FullVocab, hidden]
    float* embed_states_out                     // packed [batch, tree_width, hidden]
) {
#pragma HLS INLINE off
    if (step_input_tokens == nullptr || draft_embed_tokens_weight == nullptr ||
        embed_states_out == nullptr) {
        return false;
    }
    if (batch_size <= 0 || tree_width <= 0 || hidden_size <= 0) {
        return false;
    }

lookup_embed_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
    lookup_embed_token_loop:
        for (int t = 0; t < tree_width; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
            const int64_t token_id = step_input_tokens[b * tree_width + t];
            if (token_id < 0 || token_id >= kEagle4FullVocab) {
                return false;
            }
            const int64_t emb_row = token_id * hidden_size;
            const int64_t dst_row = (static_cast<int64_t>(b) * tree_width + t) * hidden_size;
        lookup_embed_hidden_loop:
            for (int h = 0; h < hidden_size; ++h) {
#pragma HLS loop_tripcount min=HIDDEN max=HIDDEN avg=HIDDEN
#pragma HLS PIPELINE II = 1
                const uint16_t w = draft_embed_tokens_weight[emb_row + h];
                embed_states_out[dst_row + h] = eagle4_fp16_to_float(w);
            }
        }
    }
    return true;
}

void eagle4_draft_impl(
    int tree_depth,
    int curr_depth_start,

    // Optional policy schedule (depth-indexed).
    const int* policy_next_tree_width,   // [policy_depth] optional
    const int* policy_next_verify_num,   // [policy_depth] optional
    const int* policy_stop_signal,       // [policy_depth] optional (0/1)
    int policy_depth,
    bool use_policy_schedule,

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
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>* rope_cfg_table,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    const uint16_t efficient_lm_head_down_proj_weight[tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax],
    const int32_t efficient_lm_head_qweight_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks],
    const uint16_t efficient_lm_head_scales_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_qzeros[tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_g_idx[tmac::hls::kEagle4LmRankMax],
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    const uint16_t* draft_embed_tokens_weight,

    // Contiguous KV context
    int prefix_len,
    bool enable_accepted_kv_compact,
    const int64_t* accepted_draft_node_ids,
    int accepted_draft_node_count,
    int64_t* node_to_hbm_slot,

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
    bool* stopped_early,

    // Optional InitialLoop (disabled by default).
    // If enabled:
    //   - use caller-provided initial_topk_* if present,
    //   - otherwise compute initial_topk_* from initial_logits (+ optional candidate remap).
    // initial_hidden_states must be [batch, hidden] unless prefill-stage is enabled.
    bool enable_initial_loop,// = false,
    const float* initial_logits,// = nullptr,             // [batch, initial_logits_width]
    const int64_t* initial_candidate_indices,// = nullptr,// [batch, initial_logits_width] optional
    int initial_logits_width,// = 0,
    const float* initial_topk_probas,// = nullptr,        // [batch, node_top_k] optional
    const int64_t* initial_topk_tokens,// = nullptr,      // [batch, node_top_k] optional
    const float* initial_hidden_states,// = nullptr       // [batch, hidden]

    // Optional draft-prefill stage (3H -> H projection + one SLM forward).
    bool enable_prefill_stage,// = false
    const float* prefill_input_hidden_states_3h,// = nullptr // [batch, 3 * hidden]
    const float* prefill_input_embed_states,// = nullptr      // [batch, hidden]
    const pack512* prefill_fc_weight,// = nullptr             // packed [3*hidden -> hidden]
    const float* prefill_fc_scales// = nullptr                // grouped scales
) {
#pragma HLS INLINE off
    if (tree_depth <= 0 || batch_size <= 0 || batch_size > kHlsHiddenBatch ||
        node_top_k <= 0 || hidden_size <= 0) {
        return;
    }
    if (curr_depth_start < 0 || curr_depth_start + tree_depth > kCdtControllerMaxDepth) {
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
        embed_norm_gamma == nullptr || post_attn_norm_gamma == nullptr || rope_cfg_table == nullptr ||
        final_norm_gamma == nullptr || hbm_k == nullptr || hbm_v == nullptr ||
        efficient_lm_head_down_proj_weight == nullptr ||
        efficient_lm_head_qweight_row_major == nullptr ||
        efficient_lm_head_scales_row_major == nullptr || lm_head_weight == nullptr ||
        draft_embed_tokens_weight == nullptr) {
        return;
    }
    if (hidden_size > HIDDEN || max_tree_width > TREE_WIDTH ||
        node_top_k > kCdtFusedMaxNodeTopK || node_top_k > kEagle4LmTopKMax) {
        return;
    }
    if (enable_accepted_kv_compact) {
        if (accepted_draft_node_count < 0 ||
            accepted_draft_node_count > max_verify_num ||
            accepted_draft_node_count > kContiguousKvMaxAccepted) {
            return;
        }
        if (accepted_draft_node_count > 0 &&
            (accepted_draft_node_ids == nullptr || node_to_hbm_slot == nullptr)) {
            return;
        }
    }

    const int accepted_compact_count =
        (enable_accepted_kv_compact && accepted_draft_node_count > 0)
            ? accepted_draft_node_count
            : 0;
    // Conservative upper-bound for compact/write slot validation.
    const int max_hbm_token_count =
        prefix_len + max_node_count +
        (curr_depth_start + tree_depth + 1) * max_tree_width + kContiguousKvMaxAccepted;
    if (max_hbm_token_count <= 0) {
        return;
    }

    int64_t accepted_draft_kv_slots[kContiguousKvMaxAccepted];
#pragma HLS BIND_STORAGE variable=accepted_draft_kv_slots type=ram_2p impl=bram
map_accept_nodes_loop:
    for (int i = 0; i < kContiguousKvMaxAccepted; ++i) {
#pragma HLS loop_tripcount min=1 max=kContiguousKvMaxAccepted avg=kContiguousKvMaxAccepted/2
#pragma HLS PIPELINE II = 1
        accepted_draft_kv_slots[i] = -1;
    }
    if (accepted_compact_count > 0) {
    map_accept_nodes_loop_used:
        for (int i = 0; i < kContiguousKvMaxAccepted; ++i) {
#pragma HLS loop_tripcount min=1 max=kContiguousKvMaxAccepted avg=kContiguousKvMaxAccepted/2
#pragma HLS PIPELINE II = 1
            if (i >= accepted_compact_count) {
                break;
            }
            const int64_t draft_node_id = accepted_draft_node_ids[i];
            if (draft_node_id < 0 || draft_node_id >= max_node_count) {
                return;
            }
            const int64_t mapped_slot = node_to_hbm_slot[draft_node_id];
            if (mapped_slot < 0 || mapped_slot >= max_hbm_token_count) {
                return;
            }
            accepted_draft_kv_slots[i] = mapped_slot;
        }
    }
    if (accepted_compact_count > 0) {
        const bool compact_ok = contiguous_kv_compact_accepted<
            HEAD_DIM, NUM_KV_HEADS, kContiguousKvMaxAccepted>(
                hbm_k,
                hbm_v,
                prefix_len,
                accepted_draft_kv_slots,
                accepted_compact_count,
                max_hbm_token_count);
        if (!compact_ok) {
            return;
        }
    }
    const int effective_prefix_len = prefix_len + accepted_compact_count;
    if (effective_prefix_len < 0 || effective_prefix_len >= max_hbm_token_count) {
        return;
    }
    if (node_to_hbm_slot != nullptr) {
    node_map_init_loop:
        for (int i = 0; i < kHlsMaxNodeCount; ++i) {
#pragma HLS loop_tripcount min=kTcMaxNodeCount max=kTcMaxNodeCount avg=kTcMaxNodeCount
#pragma HLS PIPELINE II = 1
            if (i < max_node_count) {
                node_to_hbm_slot[i] = -1;
            }
        }
    }

    const bool run_prefill_stage =
        enable_prefill_stage &&
        enable_initial_loop &&
        prefill_input_hidden_states_3h != nullptr &&
        prefill_input_embed_states != nullptr &&
        prefill_fc_weight != nullptr &&
        prefill_fc_scales != nullptr;
    if (enable_prefill_stage && !run_prefill_stage) {
        return;
    }

    const float* initial_hidden_states_ptr = initial_hidden_states;
    const float* initial_topk_probas_ptr = initial_topk_probas;
    const int64_t* initial_topk_tokens_ptr = initial_topk_tokens;

    const bool has_initial_topk =
        run_prefill_stage ||
        (initial_topk_probas_ptr != nullptr && initial_topk_tokens_ptr != nullptr);
    const bool has_initial_logits = (initial_logits != nullptr && initial_logits_width > 0);
    const bool run_initial_loop = enable_initial_loop && (has_initial_topk || has_initial_logits);

    if (enable_initial_loop &&
        (!run_initial_loop || (!run_prefill_stage && initial_hidden_states_ptr == nullptr))) {
        return;
    }

    int curr_tree_width = e4d_clamp_int(*io_tree_width, 0, max_tree_width);
    curr_tree_width = e4d_clamp_int(curr_tree_width, 0, node_top_k);
    int curr_verify_num = e4d_clamp_int(*io_verify_num, 1, max_verify_num);
    int curr_cumu_count = e4d_clamp_int(*io_cumu_count, 0, max_node_count);

    int depth_done = 0;
    bool stopped = false;

    // Accumulated parent indices: parent_indices_accum[l * TREE_WIDTH + t] =
    //   slot in layer l that is the parent of slot t at layer l+1.
    // Indexed by absolute depth (curr_depth_start + d), stride TREE_WIDTH.
    int parent_indices_accum[kCdtControllerMaxDepth * TREE_WIDTH];
#pragma HLS BIND_STORAGE variable = parent_indices_accum type = ram_2p impl = bram
    for (int i = 0; i < kCdtControllerMaxDepth * TREE_WIDTH; ++i) {
#pragma HLS loop_tripcount min=kTcParentAccumSize max=kTcParentAccumSize avg=kTcParentAccumSize
#pragma HLS PIPELINE II = 1
        parent_indices_accum[i] = 0;
    }

    // Internal scratch: receives parent slot indices from the fused step on every depth.
    // Using a dedicated buffer (always non-null) guarantees the step writes parent indices
    // even when the caller does not supply a debug output pointer.
    int64_t s_parent_scratch[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable = s_parent_scratch type = ram_2p impl = bram
    float s_initial_topk_probas[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
    int64_t s_initial_topk_tokens[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable = s_initial_topk_probas type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = s_initial_topk_tokens type = ram_2p impl = bram

    // ── BRAM staging: small working arrays ───────────────────────────────
    int64_t bram_step_input_tokens[kCdtFusedMaxBatch * TREE_WIDTH];
#pragma HLS BIND_STORAGE variable=bram_step_input_tokens type=ram_2p impl=bram
    float   bram_step_last_layer_scores[kCdtFusedMaxBatch * TREE_WIDTH];
#pragma HLS BIND_STORAGE variable=bram_step_last_layer_scores type=ram_2p impl=bram
    int64_t bram_step_topk_indexs_prev[kCdtFusedMaxBatch * TREE_WIDTH];
#pragma HLS BIND_STORAGE variable=bram_step_topk_indexs_prev type=ram_2p impl=bram
    float   bram_step_topk_probas_sampling[kCdtFusedMaxBatch * TREE_WIDTH * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable=bram_step_topk_probas_sampling type=ram_2p impl=bram
    int64_t bram_step_topk_tokens_sampling[kCdtFusedMaxBatch * TREE_WIDTH * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable=bram_step_topk_tokens_sampling type=ram_2p impl=bram
    float   bram_output_scores[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable=bram_output_scores type=ram_2p impl=bram
    int64_t bram_output_tokens[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable=bram_output_tokens type=ram_2p impl=bram
    float   bram_work_scores[kCdtFusedMaxBatch * (kCdtFusedMaxBatch + kCdtFusedMaxNodeTopK)];
#pragma HLS BIND_STORAGE variable=bram_work_scores type=ram_2p impl=bram
    float   bram_sort_scores[kCdtFusedMaxBatch * kCdtFusedMaxBatch];
#pragma HLS BIND_STORAGE variable=bram_sort_scores type=ram_2p impl=bram
    int64_t bram_cache_topk_indices[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable=bram_cache_topk_indices type=ram_2p impl=bram
    float bram_rope_cos_vals[HEAD_DIM / 2];
#pragma HLS BIND_STORAGE variable=bram_rope_cos_vals type=ram_2p impl=bram
    float bram_rope_sin_vals[HEAD_DIM / 2];
#pragma HLS BIND_STORAGE variable=bram_rope_sin_vals type=ram_2p impl=bram
    float s_prefill_hidden_projected[kHlsHiddenBatch * HIDDEN];
    float s_prefill_embed_states[kHlsHiddenBatch * HIDDEN];
    float s_prefill_reasoning_hidden[kHlsHiddenBatch * HIDDEN];
    float s_prefill_topk_probas[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
    int64_t s_prefill_topk_tokens[kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK];
#pragma HLS BIND_STORAGE variable=s_prefill_hidden_projected type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_prefill_embed_states type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_prefill_reasoning_hidden type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_prefill_topk_probas type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_prefill_topk_tokens type=ram_2p impl=bram

    // ── BRAM staging: hidden state arrays (kHlsHiddenBatch=1 active) ─────
    float bram_step_input_hidden_states[kHlsHiddenBatch * TREE_WIDTH * HIDDEN];
#pragma HLS BIND_STORAGE variable=bram_step_input_hidden_states type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=bram_step_input_hidden_states type=cyclic factor=16 dim=1
    float bram_step_input_embed_states[kHlsHiddenBatch * TREE_WIDTH * HIDDEN];
#pragma HLS BIND_STORAGE variable=bram_step_input_embed_states type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=bram_step_input_embed_states type=cyclic factor=16 dim=1
    float bram_output_hidden_states[kHlsHiddenBatch * kCdtFusedMaxNodeTopK * HIDDEN];
#pragma HLS BIND_STORAGE variable=bram_output_hidden_states type=ram_2p impl=bram

    // ── URAM staging: cumu/index arrays (grow each depth) ────────────────
    int64_t uram_cumu_tokens[kCdtFusedMaxBatch * kHlsMaxNodeCount];
#pragma HLS BIND_STORAGE variable=uram_cumu_tokens type=ram_2p impl=uram
    float   uram_cumu_scores[kCdtFusedMaxBatch * kHlsMaxNodeCount];
#pragma HLS BIND_STORAGE variable=uram_cumu_scores type=ram_2p impl=uram
    int64_t uram_cumu_deltas[kCdtFusedMaxBatch * kHlsMaxNodeCount];
#pragma HLS BIND_STORAGE variable=uram_cumu_deltas type=ram_2p impl=uram
    int64_t uram_prev_indexs[kCdtFusedMaxBatch * kHlsMaxNodeCount];
#pragma HLS BIND_STORAGE variable=uram_prev_indexs type=ram_2p impl=uram
    int64_t uram_next_indexs[kCdtFusedMaxBatch * kHlsMaxNodeCount];
#pragma HLS BIND_STORAGE variable=uram_next_indexs type=ram_2p impl=uram
    int64_t uram_side_indexs[kCdtFusedMaxBatch * kHlsMaxNodeCount];
#pragma HLS BIND_STORAGE variable=uram_side_indexs type=ram_2p impl=uram

    // BRAM staging: efficient LM-head quantized weights (prefetched once at entry)
    constexpr int kLmGroupSize = 64;
    constexpr int kLmMaxInPacks = kEagle4LmRankMax / 8;
    constexpr int kLmMaxGroups = (kEagle4LmRankMax + kLmGroupSize - 1) / kLmGroupSize;
    constexpr int kLmMaxVocabPacked = (kLmTcVocab + 7) / 8;

    int32_t bram_efficient_qweight[kLmTcVocab * kLmMaxInPacks];
    uint16_t bram_efficient_scales[kLmTcVocab * kLmMaxGroups];
    int32_t bram_efficient_qzeros[kLmMaxVocabPacked * kLmMaxGroups];
    int32_t bram_efficient_g_idx[kEagle4LmRankMax];
#pragma HLS BIND_STORAGE variable=bram_efficient_qweight type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=bram_efficient_scales type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=bram_efficient_qzeros type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=bram_efficient_qweight type=cyclic factor=128 dim=1
#pragma HLS ARRAY_PARTITION variable=bram_efficient_scales type=cyclic factor=128 dim=1
#pragma HLS ARRAY_PARTITION variable=bram_efficient_qzeros type=cyclic factor=128 dim=1
#pragma HLS ARRAY_PARTITION variable=bram_efficient_g_idx type=complete dim=1

    const int lm_in_packs = efficient_lm_rank / 8;
    const int lm_groups = (efficient_lm_rank + kLmGroupSize - 1) / kLmGroupSize;
    const int lm_vocab_packed = (efficient_lm_vocab_size + 7) / 8;

    int loop_start_depth = 0;
    const int step_flat  = batch_size * max_tree_width;
    const int topk_flat  = batch_size * max_tree_width * node_top_k;
    const int node_flat  = batch_size * max_node_count;
    const int hid_step   = batch_size * max_tree_width * hidden_size;
    const int hid_out    = batch_size * node_top_k * hidden_size;
    const int work_flat  = batch_size * (curr_verify_num + node_top_k);

    // ── Pre-loop: AXI → BRAM/URAM (one-time prefetch) ────────────────────
    {
    load_lm_qweight:
        for (int i = 0; i < efficient_lm_vocab_size * lm_in_packs; ++i) {
            #pragma HLS loop_tripcount min=1 max=kLmTcVocab*(kEagle4LmRankMax/8) avg=kLmTcVocab*(kEagle4LmRankMax/8)
            #pragma HLS PIPELINE II=1
            bram_efficient_qweight[i] = efficient_lm_head_qweight_row_major[i];
        }
    load_lm_scales:
        for (int i = 0; i < efficient_lm_vocab_size * lm_groups; ++i) {
            #pragma HLS loop_tripcount min=1 max=kLmTcVocab*((kEagle4LmRankMax+kLmGroupSize-1)/kLmGroupSize) avg=kLmTcVocab*((kEagle4LmRankMax+kLmGroupSize-1)/kLmGroupSize)
            #pragma HLS PIPELINE II=1
            bram_efficient_scales[i] = efficient_lm_head_scales_row_major[i];
        }
        if (efficient_lm_head_qzeros != nullptr) {
        load_lm_qzeros:
            for (int i = 0; i < lm_vocab_packed * lm_groups; ++i) {
                #pragma HLS loop_tripcount min=1 max=(kLmTcVocab+7)/8*((kEagle4LmRankMax+kLmGroupSize-1)/kLmGroupSize) avg=(kLmTcVocab+7)/8*((kEagle4LmRankMax+kLmGroupSize-1)/kLmGroupSize)
                #pragma HLS PIPELINE II=1
                bram_efficient_qzeros[i] = efficient_lm_head_qzeros[i];
            }
        }
    load_lm_gidx:
        for (int i = 0; i < kEagle4LmRankMax; ++i) {
            #pragma HLS loop_tripcount min=kEagle4LmRankMax max=kEagle4LmRankMax avg=kEagle4LmRankMax
            #pragma HLS PIPELINE II=1
            if (i < efficient_lm_rank) {
                bram_efficient_g_idx[i] =
                    (efficient_lm_head_g_idx != nullptr) ? efficient_lm_head_g_idx[i] : (i / kLmGroupSize);
            } else {
                bram_efficient_g_idx[i] = 0;
            }
        }

    load_step_tokens:
        for (int i = 0; i < step_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxStepFlat avg=kTcStepFlat
            #pragma HLS PIPELINE II=1
            bram_step_input_tokens[i] = step_input_tokens[i];
        }
    load_step_scores:
        for (int i = 0; i < step_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxStepFlat avg=kTcStepFlat
            #pragma HLS PIPELINE II=1
            bram_step_last_layer_scores[i] = step_last_layer_scores[i]; 
        }
    load_step_prev:
        for (int i = 0; i < step_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxStepFlat avg=kTcStepFlat
            #pragma HLS PIPELINE II=1
            bram_step_topk_indexs_prev[i] = step_topk_indexs_prev[i]; 
        }
    load_step_probas:
        for (int i = 0; i < topk_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxTopkFlat avg=kTcTopkFlat
            #pragma HLS PIPELINE II=1
            bram_step_topk_probas_sampling[i] = step_topk_probas_sampling[i];
        }
    load_step_toks:
        for (int i = 0; i < topk_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxTopkFlat avg=kTcTopkFlat
            #pragma HLS PIPELINE II=1
            bram_step_topk_tokens_sampling[i] = step_topk_tokens_sampling[i];
        }
    load_hid_in:
        for (int i = 0; i < hid_step; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxHidStep avg=kTcHidStep
            #pragma HLS PIPELINE II=1
            bram_step_input_hidden_states[i] = step_input_hidden_states[i];
        }
    load_work_scores:
        for (int i = 0; i < work_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxWorkFlat avg=kTcWorkFlat
            #pragma HLS PIPELINE II=1
            bram_work_scores[i] = work_scores[i];
        }
    load_rope_cos:
        for (int i = 0; i < HEAD_DIM / 2; ++i) {
            #pragma HLS loop_tripcount min=HEAD_DIM/2 max=HEAD_DIM/2 avg=HEAD_DIM/2
            #pragma HLS PIPELINE II=1
            bram_rope_cos_vals[i] = rope_cfg_table->cos_vals[i];
        }
    load_rope_sin:
        for (int i = 0; i < HEAD_DIM / 2; ++i) {
            #pragma HLS loop_tripcount min=HEAD_DIM/2 max=HEAD_DIM/2 avg=HEAD_DIM/2
            #pragma HLS PIPELINE II=1
            bram_rope_sin_vals[i] = rope_cfg_table->sin_vals[i];
        }
    load_cumu_tokens:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
            //#pragma HLS PIPELINE II=1
            uram_cumu_tokens[i] = cumu_tokens[i];
        }
    load_cumu_scores:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
            //#pragma HLS PIPELINE II=1
            uram_cumu_scores[i] = cumu_scores[i];
        }
    load_cumu_deltas:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
           // #pragma HLS PIPELINE II=1
            uram_cumu_deltas[i] = cumu_deltas[i];
        }
    load_prev_idx:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
            //#pragma HLS PIPELINE II=1
            uram_prev_indexs[i]  = prev_indexs[i]; 
        }
    load_next_idx:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
            //#pragma HLS PIPELINE II=1
            uram_next_indexs[i]  = next_indexs[i]; 
        }
    load_side_idx:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
            //#pragma HLS PIPELINE II=1
            uram_side_indexs[i]  = side_indexs[i]; 
        }
    }

    if (run_prefill_stage) {
    prefill_topk_zero_loop:
        for (int i = 0; i < kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK; ++i) {
#pragma HLS loop_tripcount min=kTcInitScratchSize max=kTcInitScratchSize avg=kTcInitScratchSize
#pragma HLS PIPELINE II=1
            s_prefill_topk_probas[i] = 0.0f;
            s_prefill_topk_tokens[i] = 0;
        }
    prefill_embed_copy_loop:
        for (int i = 0; i < batch_size * hidden_size; ++i) {
#pragma HLS loop_tripcount min=HIDDEN max=HIDDEN avg=HIDDEN
#pragma HLS PIPELINE II=1
            s_prefill_embed_states[i] = prefill_input_embed_states[i];
        }

        // Project [batch, 3H] captured target features into draft hidden width.
        e4d_prefill_fc(
            prefill_input_hidden_states_3h,
            prefill_fc_weight,
            prefill_fc_scales,
            batch_size,
            hidden_size,
            s_prefill_hidden_projected);

        // Prefill SLM uses the shared rope table (single config for current pipeline).
        e4d_slm_topk(
            s_prefill_hidden_projected,
            s_prefill_embed_states,
            batch_size,
            1,
            hidden_size,
            node_top_k,
            w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales, w_up, up_scales,
            w_down, down_scales,
            hidden_norm_gamma, embed_norm_gamma, post_attn_norm_gamma, final_norm_gamma,
            bram_rope_cos_vals, bram_rope_sin_vals,
            hbm_k, hbm_v,
            efficient_lm_head_down_proj_weight,
            bram_efficient_qweight,
            bram_efficient_scales,
            bram_efficient_qzeros,
            bram_efficient_g_idx,
            lm_head_weight,
            efficient_lm_rank,
            efficient_lm_vocab_size,
            effective_prefix_len,
            curr_depth_start,
            parent_indices_accum,
            s_prefill_reasoning_hidden,
            s_prefill_topk_probas,
            s_prefill_topk_tokens);
        if (node_to_hbm_slot != nullptr && max_node_count > 0) {
            const int root_slot = effective_prefix_len + curr_depth_start * max_tree_width;
            if (root_slot >= 0 && root_slot < max_hbm_token_count) {
                node_to_hbm_slot[0] = root_slot;
            }
        }
        initial_hidden_states_ptr = s_prefill_reasoning_hidden;
        initial_topk_probas_ptr = s_prefill_topk_probas;
        initial_topk_tokens_ptr = s_prefill_topk_tokens;
    }

    if (run_initial_loop) {
        if (has_initial_topk) {
            // Copy external data into local scratch so the pointer below is always
            // a known local object (HLS cannot synthesize conditional pointer aliasing).
        init_topk_copy_loop:
            for (int i = 0; i < kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK; ++i) {
#pragma HLS loop_tripcount min=kTcInitScratchSize max=kTcInitScratchSize avg=kTcInitScratchSize
#pragma HLS PIPELINE II = 1
                s_initial_topk_probas[i] = initial_topk_probas_ptr[i];
                s_initial_topk_tokens[i] = initial_topk_tokens_ptr[i];
            }
        } else {
            e4d_softmax_topk(
                initial_logits,
                initial_candidate_indices,
                batch_size,
                initial_logits_width,
                node_top_k,
                s_initial_topk_probas,
                s_initial_topk_tokens);
        }

        float initial_last_layer_scores[kCdtFusedMaxBatch];
        int64_t initial_topk_indexs_prev[kCdtFusedMaxBatch];
#pragma HLS ARRAY_PARTITION variable = initial_last_layer_scores complete
#pragma HLS ARRAY_PARTITION variable = initial_topk_indexs_prev complete
    init_seed_loop:
        for (int b = 0; b < batch_size; ++b) {
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
#pragma HLS PIPELINE II = 1
            initial_last_layer_scores[b] = 1.0f;
            initial_topk_indexs_prev[b] = 0;  // root global index
        }

        // InitialLoop Stage-0 parity:
        // topk from previous-verify logits updates cumulative state with tree_width=1.
        e4d_fused_step(
            s_initial_topk_probas,
            s_initial_topk_tokens,
            initial_last_layer_scores,
            initial_hidden_states_ptr,
            hot_token_id,
            hot_token_vocab_size,
            use_hot_token_id,
            initial_topk_indexs_prev,
            batch_size,
            node_top_k,
            1,  // initial tree_width is fixed to root expansion
            hidden_size,
            curr_cumu_count,
            curr_verify_num,
            curr_depth_start + 1,  // align with Python depth labeling for first generated layer
            max_node_count,
            max_verify_num,
            uram_cumu_tokens,
            uram_cumu_scores,
            uram_cumu_deltas,
            uram_prev_indexs,
            uram_next_indexs,
            uram_side_indexs,
            bram_output_scores,
            bram_output_tokens,
            bram_work_scores,
            bram_sort_scores,
            bram_output_hidden_states,
            bram_cache_topk_indices,
            dbg_curr_layer_scores,
            dbg_sort_layer_scores,
            dbg_sort_layer_indices,
            s_parent_scratch,
            dbg_remapped_topk_tokens);

        // Record root-parent mapping for first recurrent SLM call.
        if (curr_depth_start >= 0 && curr_depth_start < kCdtControllerMaxDepth) {
        init_parent_accum_loop:
            for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
#pragma HLS PIPELINE II = 1
                int parent_slot = 0;
                if (t < node_top_k) {
                    const int64_t v = s_parent_scratch[t];
                    parent_slot = (v >= 0 && v < max_tree_width) ? static_cast<int>(v) : 0;
                }
                parent_indices_accum[curr_depth_start * TREE_WIDTH + t] = parent_slot;
            }
        }
        if (dbg_parent_indices_in_layer != nullptr) {
        init_dbg_parent_fwd_loop_b:
            for (int b = 0; b < batch_size; ++b) {
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
            init_dbg_parent_fwd_loop_i:
                for (int i = 0; i < node_top_k; ++i) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
#pragma HLS PIPELINE II = 1
                    dbg_parent_indices_in_layer[b * node_top_k + i] =
                        s_parent_scratch[b * node_top_k + i];
                }
            }
        }

        curr_cumu_count += node_top_k;
        if (curr_cumu_count > max_node_count) {
            curr_cumu_count = max_node_count;
        }
        ++depth_done;

        // Python parity: depth-0 width policy is evaluated after the initial cumulative update.
        int next_tree_width = curr_tree_width;
        int next_verify_num = curr_verify_num;
        bool stop_signal = false;
        e4d_apply_policy(
            0,
            batch_size,
            1,
            node_top_k,
            max_tree_width,
            curr_verify_num,
            bram_work_scores,
            max_verify_num,
            policy_next_tree_width,
            policy_next_verify_num,
            policy_stop_signal,
            policy_depth,
            use_policy_schedule,
            &next_tree_width,
            &next_verify_num,
            &stop_signal);

        next_tree_width = e4d_clamp_int(next_tree_width, 0, max_tree_width);
        next_tree_width = e4d_clamp_int(next_tree_width, 0, node_top_k);
        next_verify_num = e4d_clamp_int(next_verify_num, 1, max_verify_num);
        curr_tree_width = next_tree_width;
        curr_verify_num = next_verify_num;

        if (tree_depth <= 1 || stop_signal || next_tree_width <= 0) {
            stopped = stop_signal || (next_tree_width <= 0);
            goto orchestrator_finalize;
        }

        // Stage-1 parity: select first tree_width frontier for the first recurrent SLM forward.
        e4d_prep_next_inputs(
            bram_output_scores,
            bram_output_tokens,
            bram_output_hidden_states,
            bram_cache_topk_indices,
            batch_size,
            node_top_k,
            hidden_size,
            next_tree_width,
            max_tree_width,
            bram_step_input_tokens,
            bram_step_last_layer_scores,
            bram_step_input_hidden_states,
            bram_step_topk_indexs_prev);

        loop_start_depth = 1;
    }

orchestrator_depth_loop:
    for (int d = loop_start_depth; d < tree_depth; ++d) {
#pragma HLS loop_tripcount min=kTcDepth max=kTcDepth avg=kTcDepth
        if (curr_tree_width <= 0) {
            stopped = true;
            break;
        }

        int next_tree_width = curr_tree_width;
        int next_verify_num = curr_verify_num;
        bool stop_signal = false;
        if (!run_initial_loop) {
            e4d_apply_policy(
                d,
                batch_size,
                curr_tree_width,
                node_top_k,
                max_tree_width,
                curr_verify_num,
                bram_work_scores,
                max_verify_num,
                policy_next_tree_width,
                policy_next_verify_num,
                policy_stop_signal,
                policy_depth,
                use_policy_schedule,
                &next_tree_width,
                &next_verify_num,
                &stop_signal);

            next_tree_width = e4d_clamp_int(next_tree_width, 0, max_tree_width);
            next_tree_width = e4d_clamp_int(next_tree_width, 0, node_top_k);
            next_verify_num = e4d_clamp_int(next_verify_num, 1, max_verify_num);
        }

        // Stage A/B: SLM forward and LM-head top-k for current frontier.
        const int current_depth = curr_depth_start + d;
        const bool embed_ok = e4d_lookup_recurrent_embed_states(
            bram_step_input_tokens,
            batch_size,
            curr_tree_width,
            hidden_size,
            draft_embed_tokens_weight,
            bram_step_input_embed_states);
        if (!embed_ok) {
            return;
        }
        e4d_slm_topk(
            bram_step_input_hidden_states,
            bram_step_input_embed_states,
            batch_size,
            curr_tree_width,
            hidden_size,
            node_top_k,
            w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales, w_up, up_scales,
            w_down, down_scales,
            hidden_norm_gamma, embed_norm_gamma, post_attn_norm_gamma, final_norm_gamma,
            bram_rope_cos_vals, bram_rope_sin_vals,
            hbm_k, hbm_v,
            efficient_lm_head_down_proj_weight,
            bram_efficient_qweight,
            bram_efficient_scales,
            bram_efficient_qzeros,
            bram_efficient_g_idx,
            lm_head_weight,
            efficient_lm_rank,
            efficient_lm_vocab_size,
            effective_prefix_len,
            current_depth,
            parent_indices_accum,
            bram_step_input_hidden_states,
            bram_step_topk_probas_sampling,
            bram_step_topk_tokens_sampling);
        if (node_to_hbm_slot != nullptr) {
            const int slot_base = effective_prefix_len + current_depth * max_tree_width;
        update_node_to_hbm_loop:
            for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
#pragma HLS PIPELINE II = 1
                if (t < curr_tree_width) {
                    const int64_t node_id = bram_step_topk_indexs_prev[t];  // batch=1
                    const int slot = slot_base + t;
                    if (node_id >= 0 && node_id < max_node_count &&
                        slot >= 0 && slot < max_hbm_token_count) {
                        node_to_hbm_slot[node_id] = slot;
                    }
                }
            }
        }

        // Stage C: fused score/update for one depth.
        e4d_fused_step(
            bram_step_topk_probas_sampling,
            bram_step_topk_tokens_sampling,
            bram_step_last_layer_scores,
            bram_step_input_hidden_states,
            hot_token_id,
            hot_token_vocab_size,
            use_hot_token_id,
            bram_step_topk_indexs_prev,
            batch_size,
            node_top_k,
            curr_tree_width,
            hidden_size,
            curr_cumu_count,
            curr_verify_num,
            run_initial_loop ? (curr_depth_start + d + 1) : (curr_depth_start + d),
            max_node_count,
            max_verify_num,
            uram_cumu_tokens,
            uram_cumu_scores,
            uram_cumu_deltas,
            uram_prev_indexs,
            uram_next_indexs,
            uram_side_indexs,
            bram_output_scores,
            bram_output_tokens,
            bram_work_scores,
            bram_sort_scores,
            bram_output_hidden_states,
            bram_cache_topk_indices,
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
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
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
#pragma HLS loop_tripcount min=kTcBatch max=kTcBatch avg=kTcBatch
            dbg_parent_fwd_loop_i:
                for (int i = 0; i < node_top_k; ++i) {
#pragma HLS loop_tripcount min=kTcTopK max=kTcTopK avg=kTcTopK
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
        if (run_initial_loop) {
            e4d_apply_policy(
                d,
                batch_size,
                curr_tree_width,
                node_top_k,
                max_tree_width,
                curr_verify_num,
                bram_work_scores,
                max_verify_num,
                policy_next_tree_width,
                policy_next_verify_num,
                policy_stop_signal,
                policy_depth,
                use_policy_schedule,
                &next_tree_width,
                &next_verify_num,
                &stop_signal);

            next_tree_width = e4d_clamp_int(next_tree_width, 0, max_tree_width);
            next_tree_width = e4d_clamp_int(next_tree_width, 0, node_top_k);
            next_verify_num = e4d_clamp_int(next_verify_num, 1, max_verify_num);
        }
        if (d + 1 >= tree_depth || stop_signal || next_tree_width <= 0) {
            curr_tree_width = next_tree_width;
            curr_verify_num = next_verify_num;
            stopped = stop_signal || (next_tree_width <= 0);
            break;
        }

        // Stage D: recurrence wiring for next SLM call.
        e4d_prep_next_inputs(
            bram_output_scores,
            bram_output_tokens,
            bram_output_hidden_states,
            bram_cache_topk_indices,
            batch_size,
            node_top_k,
            hidden_size,
            next_tree_width,
            max_tree_width,
            bram_step_input_tokens,
            bram_step_last_layer_scores,
            bram_step_input_hidden_states,
            bram_step_topk_indexs_prev);

        curr_tree_width = next_tree_width;
        curr_verify_num = next_verify_num;
    }

orchestrator_finalize:
    *io_tree_width = curr_tree_width;
    *io_verify_num = curr_verify_num;
    *io_cumu_count = curr_cumu_count;
    if (executed_depths != nullptr) {
        *executed_depths = depth_done;
    }
    if (stopped_early != nullptr) {
        *stopped_early = stopped;
    }

    // ── Post-loop: BRAM/URAM → AXI (one-time writeback) ─────────────────
    {
    wb_step_tokens:
        for (int i = 0; i < step_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxStepFlat avg=kTcStepFlat
            #pragma HLS PIPELINE II=1
            step_input_tokens[i] = bram_step_input_tokens[i]; 
        }
    wb_step_scores:
        for (int i = 0; i < step_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxStepFlat avg=kTcStepFlat
            #pragma HLS PIPELINE II=1
            step_last_layer_scores[i] = bram_step_last_layer_scores[i]; 
        }
    wb_step_prev:
        for (int i = 0; i < step_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxStepFlat avg=kTcStepFlat
            #pragma HLS PIPELINE II=1
            step_topk_indexs_prev[i] = bram_step_topk_indexs_prev[i]; 
        }
    wb_step_probas:
        for (int i = 0; i < topk_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxTopkFlat avg=kTcTopkFlat
            #pragma HLS PIPELINE II=1
            step_topk_probas_sampling[i] = bram_step_topk_probas_sampling[i]; 
        }
    wb_step_toks:
        for (int i = 0; i < topk_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxTopkFlat avg=kTcTopkFlat
            #pragma HLS PIPELINE II=1
            step_topk_tokens_sampling[i] = bram_step_topk_tokens_sampling[i]; 
        }
    wb_hid_in:
        for (int i = 0; i < hid_step; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxHidStep avg=kTcHidStep
            #pragma HLS PIPELINE II=1
            step_input_hidden_states[i] = bram_step_input_hidden_states[i]; 
        }
    wb_output_scores:
        for (int i = 0; i < batch_size * node_top_k; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxOutputFlat avg=kTcOutputFlat
            #pragma HLS PIPELINE II=1
            output_scores[i] = bram_output_scores[i]; 
        }
    wb_output_tokens:
        for (int i = 0; i < batch_size * node_top_k; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxOutputFlat avg=kTcOutputFlat
            #pragma HLS PIPELINE II=1
            output_tokens[i] = bram_output_tokens[i]; 
        }
    wb_work_scores:
        for (int i = 0; i < work_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxWorkFlat avg=kTcWorkFlat
            #pragma HLS PIPELINE II=1
            work_scores[i] = bram_work_scores[i];
        }
    wb_sort_scores:
        for (int i = 0; i < batch_size * curr_verify_num; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxSortFlat avg=kTcSortFlat
            #pragma HLS PIPELINE II=1
            sort_scores[i] = bram_sort_scores[i]; 
        }
    wb_cache_topk:
        for (int i = 0; i < batch_size * node_top_k; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxOutputFlat avg=kTcOutputFlat
            #pragma HLS PIPELINE II=1
            cache_topk_indices[i] = bram_cache_topk_indices[i]; 
        }
    wb_hid_out:
        for (int i = 0; i < hid_out; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxHidOut avg=kTcHidOut
            #pragma HLS PIPELINE II=1
            output_hidden_states[i] = bram_output_hidden_states[i]; 
        }
    wb_cumu_tokens:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
           // #pragma HLS PIPELINE II=1
            cumu_tokens[i] = uram_cumu_tokens[i]; 
        }
    wb_cumu_scores:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
           // #pragma HLS PIPELINE II=1
            cumu_scores[i] = uram_cumu_scores[i];
        }
    wb_cumu_deltas:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
           // #pragma HLS PIPELINE II=1
            cumu_deltas[i] = uram_cumu_deltas[i];
        }
    wb_prev_idx:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
           // #pragma HLS PIPELINE II=1
            prev_indexs[i] = uram_prev_indexs[i];
        }
    wb_next_idx:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
            //#pragma HLS PIPELINE II=1
            next_indexs[i] = uram_next_indexs[i];
        }
    wb_side_idx:
        for (int i = 0; i < node_flat; ++i) {
            #pragma HLS loop_tripcount min=1 max=kMaxNodeFlat avg=kTcNodeFlat
           // #pragma HLS PIPELINE II=1
            side_indexs[i] = uram_side_indexs[i];
        }
    }
}

} // namespace hls
} // namespace tmac

void eagle4_draft(
    int tree_depth,
    int curr_depth_start,
    const int* policy_next_tree_width,
    const int* policy_next_verify_num,
    const int* policy_stop_signal,
    int policy_depth,
    bool use_policy_schedule,
    int64_t* step_input_tokens,
    float* step_input_hidden_states,
    float* step_last_layer_scores,
    int64_t* step_topk_indexs_prev,
    float* step_topk_probas_sampling,
    int64_t* step_topk_tokens_sampling,
    const tmac::hls::pack512* w_q,     const float* s_q,
    const tmac::hls::pack512* w_k,     const float* s_k,
    const tmac::hls::pack512* w_v,     const float* s_v,
    const tmac::hls::pack512* w_o,     const float* s_o,
    const tmac::hls::pack512* w_gate,  const float* gate_scales,
    const tmac::hls::pack512* w_up,    const float* up_scales,
    const tmac::hls::pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const tmac::hls::RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>* rope_cfg_table,
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_k,
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_v,
    const uint16_t efficient_lm_head_down_proj_weight[tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax],
    const int32_t efficient_lm_head_qweight_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks],
    const uint16_t efficient_lm_head_scales_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_qzeros[tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups],
    const int32_t efficient_lm_head_g_idx[tmac::hls::kEagle4LmRankMax],
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    const uint16_t* draft_embed_tokens_weight,
    int prefix_len,
    bool enable_accepted_kv_compact,
    const int64_t* accepted_draft_node_ids,
    int accepted_draft_node_count,
    int64_t* node_to_hbm_slot,
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
    int batch_size,
    int node_top_k,
    int hidden_size,
    int* io_tree_width,
    int* io_verify_num,
    int* io_cumu_count,
    int max_node_count,
    int max_verify_num,
    int max_tree_width,
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
    float* output_hidden_states,
    int64_t* cache_topk_indices,
    float* dbg_curr_layer_scores,
    float* dbg_sort_layer_scores,
    int64_t* dbg_sort_layer_indices,
    int64_t* dbg_parent_indices_in_layer,
    int64_t* dbg_remapped_topk_tokens,
    int* executed_depths,
    bool* stopped_early,
    bool enable_initial_loop,
    const float* initial_logits,
    const int64_t* initial_candidate_indices,
    int initial_logits_width,
    const float* initial_topk_probas,
    const int64_t* initial_topk_tokens,
    const float* initial_hidden_states,
    bool enable_prefill_stage,
    const float* prefill_input_hidden_states_3h,
    const float* prefill_input_embed_states,
    const tmac::hls::pack512* prefill_fc_weight,
    const float* prefill_fc_scales) {
#pragma HLS INLINE off
#pragma HLS INTERFACE m_axi port=policy_next_tree_width offset=slave bundle=gmem_policy
#pragma HLS INTERFACE m_axi port=policy_next_verify_num offset=slave bundle=gmem_policy
#pragma HLS INTERFACE m_axi port=policy_stop_signal offset=slave bundle=gmem_policy

#pragma HLS INTERFACE m_axi port=step_input_tokens offset=slave bundle=gmem_step0
#pragma HLS INTERFACE m_axi port=step_input_hidden_states offset=slave bundle=gmem_step1
#pragma HLS INTERFACE m_axi port=step_last_layer_scores offset=slave bundle=gmem_step2
#pragma HLS INTERFACE m_axi port=step_topk_indexs_prev offset=slave bundle=gmem_step3
#pragma HLS INTERFACE m_axi port=step_topk_probas_sampling offset=slave bundle=gmem_step4
#pragma HLS INTERFACE m_axi port=step_topk_tokens_sampling offset=slave bundle=gmem_step5

#pragma HLS INTERFACE m_axi port=w_q offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=s_q offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=w_k offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=s_k offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=w_v offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=s_v offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=w_o offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=s_o offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=w_gate offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=gate_scales offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=w_up offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=up_scales offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=w_down offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=down_scales offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=hidden_norm_gamma offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=embed_norm_gamma offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=post_attn_norm_gamma offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=final_norm_gamma offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=rope_cfg_table offset=slave bundle=gmem_cfg
#pragma HLS INTERFACE m_axi port=hbm_k offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=hbm_v offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=accepted_draft_node_ids offset=slave bundle=gmem_kv_compact
#pragma HLS INTERFACE m_axi port=node_to_hbm_slot offset=slave bundle=gmem_kv_compact

#pragma HLS INTERFACE m_axi port=efficient_lm_head_down_proj_weight offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=efficient_lm_head_qweight_row_major offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=efficient_lm_head_scales_row_major offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=efficient_lm_head_qzeros offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=efficient_lm_head_g_idx offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=lm_head_weight offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=draft_embed_tokens_weight offset=slave bundle=gmem_embed
#pragma HLS INTERFACE m_axi port=hot_token_id offset=slave bundle=gmem_hot

#pragma HLS INTERFACE m_axi port=io_tree_width offset=slave bundle=gmem_io
#pragma HLS INTERFACE m_axi port=io_verify_num offset=slave bundle=gmem_io
#pragma HLS INTERFACE m_axi port=io_cumu_count offset=slave bundle=gmem_io

#pragma HLS INTERFACE m_axi port=cumu_tokens offset=slave bundle=gmem_state0
#pragma HLS INTERFACE m_axi port=cumu_scores offset=slave bundle=gmem_state1
#pragma HLS INTERFACE m_axi port=cumu_deltas offset=slave bundle=gmem_state2
#pragma HLS INTERFACE m_axi port=prev_indexs offset=slave bundle=gmem_state3
#pragma HLS INTERFACE m_axi port=next_indexs offset=slave bundle=gmem_state4
#pragma HLS INTERFACE m_axi port=side_indexs offset=slave bundle=gmem_state5
#pragma HLS INTERFACE m_axi port=output_scores offset=slave bundle=gmem_state6
#pragma HLS INTERFACE m_axi port=output_tokens offset=slave bundle=gmem_state7
#pragma HLS INTERFACE m_axi port=work_scores offset=slave bundle=gmem_state8
#pragma HLS INTERFACE m_axi port=sort_scores offset=slave bundle=gmem_state9
#pragma HLS INTERFACE m_axi port=output_hidden_states offset=slave bundle=gmem_state10
#pragma HLS INTERFACE m_axi port=cache_topk_indices offset=slave bundle=gmem_state11

#pragma HLS INTERFACE m_axi port=dbg_curr_layer_scores offset=slave bundle=gmem_dbg0
#pragma HLS INTERFACE m_axi port=dbg_sort_layer_scores offset=slave bundle=gmem_dbg1
#pragma HLS INTERFACE m_axi port=dbg_sort_layer_indices offset=slave bundle=gmem_dbg2
#pragma HLS INTERFACE m_axi port=dbg_parent_indices_in_layer offset=slave bundle=gmem_dbg3
#pragma HLS INTERFACE m_axi port=dbg_remapped_topk_tokens offset=slave bundle=gmem_dbg4
#pragma HLS INTERFACE m_axi port=executed_depths offset=slave bundle=gmem_dbg5
#pragma HLS INTERFACE m_axi port=stopped_early offset=slave bundle=gmem_dbg5

#pragma HLS INTERFACE m_axi port=initial_logits offset=slave bundle=gmem_init0
#pragma HLS INTERFACE m_axi port=initial_candidate_indices offset=slave bundle=gmem_init1
#pragma HLS INTERFACE m_axi port=initial_topk_probas offset=slave bundle=gmem_init2
#pragma HLS INTERFACE m_axi port=initial_topk_tokens offset=slave bundle=gmem_init3
#pragma HLS INTERFACE m_axi port=initial_hidden_states offset=slave bundle=gmem_init4
#pragma HLS INTERFACE m_axi port=prefill_input_hidden_states_3h offset=slave bundle=gmem_prefill0
#pragma HLS INTERFACE m_axi port=prefill_input_embed_states offset=slave bundle=gmem_prefill1
#pragma HLS INTERFACE m_axi port=prefill_fc_weight offset=slave bundle=gmem_prefill2
#pragma HLS INTERFACE m_axi port=prefill_fc_scales offset=slave bundle=gmem_prefill3

#pragma HLS INTERFACE s_axilite port=tree_depth bundle=control
#pragma HLS INTERFACE s_axilite port=curr_depth_start bundle=control
#pragma HLS INTERFACE s_axilite port=policy_depth bundle=control
#pragma HLS INTERFACE s_axilite port=use_policy_schedule bundle=control
#pragma HLS INTERFACE s_axilite port=efficient_lm_rank bundle=control
#pragma HLS INTERFACE s_axilite port=efficient_lm_vocab_size bundle=control
#pragma HLS INTERFACE s_axilite port=prefix_len bundle=control
#pragma HLS INTERFACE s_axilite port=enable_accepted_kv_compact bundle=control
#pragma HLS INTERFACE s_axilite port=accepted_draft_node_count bundle=control
#pragma HLS INTERFACE s_axilite port=hot_token_vocab_size bundle=control
#pragma HLS INTERFACE s_axilite port=use_hot_token_id bundle=control
#pragma HLS INTERFACE s_axilite port=batch_size bundle=control
#pragma HLS INTERFACE s_axilite port=node_top_k bundle=control
#pragma HLS INTERFACE s_axilite port=hidden_size bundle=control
#pragma HLS INTERFACE s_axilite port=max_node_count bundle=control
#pragma HLS INTERFACE s_axilite port=max_verify_num bundle=control
#pragma HLS INTERFACE s_axilite port=max_tree_width bundle=control
#pragma HLS INTERFACE s_axilite port=enable_initial_loop bundle=control
#pragma HLS INTERFACE s_axilite port=enable_prefill_stage bundle=control
#pragma HLS INTERFACE s_axilite port=initial_logits_width bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    tmac::hls::eagle4_draft_impl(
        tree_depth,
        curr_depth_start,
        policy_next_tree_width,
        policy_next_verify_num,
        policy_stop_signal,
        policy_depth,
        use_policy_schedule,
        step_input_tokens,
        step_input_hidden_states,
        step_last_layer_scores,
        step_topk_indexs_prev,
        step_topk_probas_sampling,
        step_topk_tokens_sampling,
        w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales, w_up, up_scales, w_down,
        down_scales,
        hidden_norm_gamma,
        embed_norm_gamma,
        post_attn_norm_gamma,
        final_norm_gamma,
        rope_cfg_table,
        hbm_k,
        hbm_v,
        efficient_lm_head_down_proj_weight,
        efficient_lm_head_qweight_row_major,
        efficient_lm_head_scales_row_major,
        efficient_lm_head_qzeros,
        efficient_lm_head_g_idx,
        lm_head_weight,
        efficient_lm_rank,
        efficient_lm_vocab_size,
        draft_embed_tokens_weight,
        prefix_len,
        enable_accepted_kv_compact,
        accepted_draft_node_ids,
        accepted_draft_node_count,
        node_to_hbm_slot,
        hot_token_id,
        hot_token_vocab_size,
        use_hot_token_id,
        batch_size,
        node_top_k,
        hidden_size,
        io_tree_width,
        io_verify_num,
        io_cumu_count,
        max_node_count,
        max_verify_num,
        max_tree_width,
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
        dbg_parent_indices_in_layer,
        dbg_remapped_topk_tokens,
        executed_depths,
        stopped_early,
        enable_initial_loop,
        initial_logits,
        initial_candidate_indices,
        initial_logits_width,
        initial_topk_probas,
        initial_topk_tokens,
        initial_hidden_states,
        enable_prefill_stage,
        prefill_input_hidden_states_3h,
        prefill_input_embed_states,
        prefill_fc_weight,
        prefill_fc_scales);
}

void eagle4_draft_bs1_fixed(
    int tree_depth,
    int curr_depth_start,
    int prefix_len,
    int fixed_tree_width,
    int fixed_verify_num,
    int* io_cumu_count,
    int* final_tree_width,
    int* final_verify_num,
    int64_t* step_input_tokens,
    float* step_input_hidden_states,
    float* step_last_layer_scores,
    int64_t* step_topk_indexs_prev,
    float* step_topk_probas_sampling,
    int64_t* step_topk_tokens_sampling,
    const tmac::hls::pack512* w_q,     const float* s_q,
    const tmac::hls::pack512* w_k,     const float* s_k,
    const tmac::hls::pack512* w_v,     const float* s_v,
    const tmac::hls::pack512* w_o,     const float* s_o,
    const tmac::hls::pack512* w_gate,  const float* gate_scales,
    const tmac::hls::pack512* w_up,    const float* up_scales,
    const tmac::hls::pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const tmac::hls::RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>* rope_cfg_table,
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_k,
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_v,
    const uint16_t* efficient_lm_head_down_proj_weight,
    const int32_t* efficient_lm_head_qweight_row_major,
    const uint16_t* efficient_lm_head_scales_row_major,
    const int32_t* efficient_lm_head_qzeros,
    const int32_t* efficient_lm_head_g_idx,
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
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
    float* output_hidden_states,
    int64_t* cache_topk_indices,
    float* dbg_curr_layer_scores,
    float* dbg_sort_layer_scores,
    int64_t* dbg_sort_layer_indices,
    int64_t* dbg_parent_indices_in_layer,
    int64_t* dbg_remapped_topk_tokens,
    int* executed_depths,
    bool* stopped_early,
    bool enable_initial_loop,
    const float* initial_logits,
    const int64_t* initial_candidate_indices,
    int initial_logits_width,
    const float* initial_topk_probas,
    const int64_t* initial_topk_tokens,
    const float* initial_hidden_states) {
#pragma HLS INLINE off
    const int batch_size = tmac::hls::kE4dTransitionBatchSize;
    const int node_top_k = tmac::hls::kE4dTransitionNodeTopK;
    const int hidden_size = tmac::hls::kE4dTransitionHiddenSize;
    const int max_node_count = tmac::hls::kE4dTransitionMaxNodeCount;
    const int max_verify_num = tmac::hls::kE4dTransitionMaxVerifyNum;
    const int max_tree_width = tmac::hls::kE4dTransitionMaxTreeWidth;

    int io_tree_width_local = tmac::hls::e4d_clamp_int(fixed_tree_width, 1, max_tree_width);
    io_tree_width_local = tmac::hls::e4d_clamp_int(io_tree_width_local, 1, node_top_k);
    int io_verify_num_local = tmac::hls::e4d_clamp_int(fixed_verify_num, 1, max_verify_num);

    int io_cumu_count_local = 0;
    if (io_cumu_count != nullptr) {
        io_cumu_count_local = *io_cumu_count;
    }
    io_cumu_count_local = tmac::hls::e4d_clamp_int(io_cumu_count_local, 0, max_node_count);

    tmac::hls::eagle4_draft_impl(
        tree_depth,
        curr_depth_start,
        nullptr,
        nullptr,
        nullptr,
        0,
        false,
        step_input_tokens,
        step_input_hidden_states,
        step_last_layer_scores,
        step_topk_indexs_prev,
        step_topk_probas_sampling,
        step_topk_tokens_sampling,
        w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales, w_up, up_scales, w_down,
        down_scales,
        hidden_norm_gamma,
        embed_norm_gamma,
        post_attn_norm_gamma,
        final_norm_gamma,
        rope_cfg_table,
        hbm_k,
        hbm_v,
        efficient_lm_head_down_proj_weight,
        efficient_lm_head_qweight_row_major,
        efficient_lm_head_scales_row_major,
        efficient_lm_head_qzeros,
        efficient_lm_head_g_idx,
        lm_head_weight,
        efficient_lm_rank,
        efficient_lm_vocab_size,
        prefix_len,
        hot_token_id,
        hot_token_vocab_size,
        use_hot_token_id,
        batch_size,
        node_top_k,
        hidden_size,
        &io_tree_width_local,
        &io_verify_num_local,
        &io_cumu_count_local,
        max_node_count,
        max_verify_num,
        max_tree_width,
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
        dbg_parent_indices_in_layer,
        dbg_remapped_topk_tokens,
        executed_depths,
        stopped_early,
        enable_initial_loop,
        initial_logits,
        initial_candidate_indices,
        initial_logits_width,
        initial_topk_probas,
        initial_topk_tokens,
        initial_hidden_states);

    if (io_cumu_count != nullptr) {
        *io_cumu_count = io_cumu_count_local;
    }
    if (final_tree_width != nullptr) {
        *final_tree_width = io_tree_width_local;
    }
    if (final_verify_num != nullptr) {
        *final_verify_num = io_verify_num_local;
    }
}
