#include "eagle_tier1_lm_top.hpp"
#include <limits>

// Super-wrapper: Tier1 transformer -> 8-way LM head (single token, batch slot 0).
namespace {

void sink_reasoning_stream(hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& reasoning_stream,
                           float* reasoning_state_out) {
#pragma HLS INLINE off
    for (int t = 0; t < tmac::hls::TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=tmac::hls::TREE_WIDTH max=tmac::hls::TREE_WIDTH
        for (int i = 0; i < tmac::hls::HIDDEN / tmac::hls::VEC_W; ++i) {
#pragma HLS loop_tripcount min=tmac::hls::HIDDEN/tmac::hls::VEC_W max=tmac::hls::HIDDEN/tmac::hls::VEC_W
#pragma HLS PIPELINE II=1
            auto v = reasoning_stream.read();
            for (int j = 0; j < tmac::hls::VEC_W; ++j) {
#pragma HLS UNROLL
                reasoning_state_out[t * tmac::hls::HIDDEN + i * tmac::hls::VEC_W + j] = v[j];
            }
        }
    }
}

void collect_logits_stream(
    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& logits_stream,
    float logits_hidden[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmHiddenMax]) {
#pragma HLS INLINE off
    for (int t = 0; t < tmac::hls::TREE_WIDTH; t++) {
#pragma HLS loop_tripcount min=tmac::hls::TREE_WIDTH max=tmac::hls::TREE_WIDTH
        for (int i = 0; i < tmac::hls::HIDDEN / tmac::hls::VEC_W; ++i) {
#pragma HLS loop_tripcount min=tmac::hls::HIDDEN/tmac::hls::VEC_W max=tmac::hls::HIDDEN/tmac::hls::VEC_W
    #pragma HLS PIPELINE II=1
            auto v = logits_stream.read();
            for (int j = 0; j < tmac::hls::VEC_W; ++j) {
    #pragma HLS UNROLL
                logits_hidden[t][i * tmac::hls::VEC_W + j] = v[j];
            }
        }
    }
}
} // namespace

void eagle_tier1_lm_top(hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& hidden_in_stream,
                        hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& embed_in_stream,
                        int* best_id,
                        float* best_score,
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
                        const tmac::hls::RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>& rope_cfg,
                        tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_k,
                        tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_v,
                        const wide_vec_t* lm_w0,
                        const wide_vec_t* lm_w1,
                        const wide_vec_t* lm_w2,
                        const wide_vec_t* lm_w3,
                        const wide_vec_t* lm_w4,
                        const wide_vec_t* lm_w5,
                        const wide_vec_t* lm_w6,
                        const wide_vec_t* lm_w7,
                        float* reasoning_state_out,
                        const uint16_t* efficient_lm_head_down_proj_weight,
                        const int32_t* efficient_lm_head_qweight_row_major,
                        const uint16_t* efficient_lm_head_scales_row_major,
                        const int32_t* efficient_lm_head_qzeros,
                        const int32_t* efficient_lm_head_g_idx,
                        const uint16_t* lm_head_weight,
                        int efficient_lm_rank,
                        int efficient_lm_vocab_size,
                        int efficient_lm_num_candidates,
                        int* candidate_indices_out,
                        float* gathered_logits_out,
                        int prefix_len,
                        int current_depth,
                        const int* parent_indices_per_layer) {

    // Integration note:
    // Prefer EAGLE4 efficient LM-head path when all required buffers/dims are provided.
    const bool use_eagle4_lm_head =
        efficient_lm_head_down_proj_weight != nullptr &&
        efficient_lm_head_qweight_row_major != nullptr &&
        efficient_lm_head_scales_row_major != nullptr &&
        lm_head_weight != nullptr &&
        efficient_lm_rank > 0 &&
        efficient_lm_vocab_size > 0 &&
        efficient_lm_num_candidates > 0;
    if (use_eagle4_lm_head) {
        eagle_tier1_lm_top_eagle4(
            hidden_in_stream,
            embed_in_stream,
            best_id,
            best_score,
            w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales, w_up, up_scales, w_down, down_scales,
            hidden_norm_gamma, embed_norm_gamma, post_attn_norm_gamma, final_norm_gamma,
            rope_cfg, hbm_k, hbm_v,
            efficient_lm_head_down_proj_weight,
            efficient_lm_head_qweight_row_major,
            efficient_lm_head_scales_row_major,
            efficient_lm_head_qzeros,
            efficient_lm_head_g_idx,
            lm_head_weight,
            efficient_lm_rank,
            efficient_lm_vocab_size,
            efficient_lm_num_candidates,
            reasoning_state_out,
            candidate_indices_out,
            gathered_logits_out,
            prefix_len,
            current_depth,
            parent_indices_per_layer);
        return;
    }

#if TMAC_ALLOW_LEGACY_LM_HEAD8WAY
    // Legacy compatibility path (Eagle3 dense 8-way LM head).
#pragma HLS DATAFLOW

    // EAGLE4 parity block
    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>> reasoning_out("reasoning_out");
    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>> logits_out("logits_out");
#pragma HLS STREAM variable=reasoning_out depth=64
#pragma HLS STREAM variable=logits_out depth=64
    eagle_tier1_top_eagle4_l0(hidden_in_stream, embed_in_stream, reasoning_out, logits_out,
                              w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales,
                              w_up, up_scales, w_down, down_scales, hidden_norm_gamma, embed_norm_gamma,
                              post_attn_norm_gamma, final_norm_gamma, rope_cfg, hbm_k, hbm_v,
                              prefix_len, current_depth, parent_indices_per_layer);

    TokenOutput lm_result{};
    lm_head_8way_top(lm_w0, lm_w1, lm_w2, lm_w3, lm_w4, lm_w5, lm_w6, lm_w7,
                     logits_out, lm_result);
    sink_reasoning_stream(reasoning_out, reasoning_state_out);

    *best_id = lm_result.best_id[0];
    *best_score = lm_result.best_score[0];
#else
    // Legacy path intentionally disabled by default while integrating EAGLE4 LM-head end-to-end.
    // Build with -DTMAC_ALLOW_LEGACY_LM_HEAD8WAY=1 for Eagle3 regression.
    *best_id = -1;
    *best_score = -std::numeric_limits<float>::infinity();
    (void)lm_w0; (void)lm_w1; (void)lm_w2; (void)lm_w3;
    (void)lm_w4; (void)lm_w5; (void)lm_w6; (void)lm_w7;
#endif
}

void eagle_tier1_lm_top_eagle4(hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& hidden_in_stream,
                               hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& embed_in_stream,
                               int* best_id,
                               float* best_score,
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
                               const tmac::hls::RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>& rope_cfg,
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
                               int efficient_lm_num_candidates,
                               float* reasoning_state_out,
                               int* candidate_indices_out,
                               float* gathered_logits_out,
                               int prefix_len,
                               int current_depth,
                               const int* parent_indices_per_layer) {
    int rank = efficient_lm_rank;
    int vocab = efficient_lm_vocab_size;
    int topk = efficient_lm_num_candidates;
    if (rank <= 0 || rank > tmac::hls::kEagle4LmRankMax ||
        vocab <= 0 || topk <= 0 || topk > tmac::hls::kEagle4LmTopKMax) {
        *best_id = -1;
        *best_score = -std::numeric_limits<float>::infinity();
        return;
    }

    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>> reasoning_out("reasoning_out");
    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>> logits_out("logits_out");
#pragma HLS STREAM variable=reasoning_out depth=64
#pragma HLS STREAM variable=logits_out depth=64

    eagle_tier1_top_eagle4_l0(hidden_in_stream, embed_in_stream, reasoning_out, logits_out,
                              w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o, w_gate, gate_scales,
                              w_up, up_scales, w_down, down_scales, hidden_norm_gamma, embed_norm_gamma,
                              post_attn_norm_gamma, final_norm_gamma, rope_cfg, hbm_k, hbm_v,
                              prefix_len, current_depth, parent_indices_per_layer);

    sink_reasoning_stream(reasoning_out, reasoning_state_out);

    float logits_hidden[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmHiddenMax];
    float low_rank[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmRankMax];
    int candidate_indices[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmTopKMax];
    float candidate_scores[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmTopKMax];
    float gathered_logits[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmTopKMax];
    int topk_tokens[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmTopKMax];
    float topk_probas[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmTopKMax];
#pragma HLS ARRAY_PARTITION variable=logits_hidden cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=low_rank cyclic factor=16 dim=2

    collect_logits_stream(logits_out, logits_hidden);

    tmac::hls::eagle4_lm_down_project(
        logits_hidden,
        efficient_lm_head_down_proj_weight,
        low_rank,
        tmac::hls::HIDDEN,
        rank);

    tmac::hls::eagle4_lm_candidate_logits_row4(
        low_rank,
        efficient_lm_head_qweight_row_major,
        efficient_lm_head_scales_row_major,
        efficient_lm_head_qzeros,
        efficient_lm_head_g_idx,
        rank,
        vocab,
        128,
        nullptr,
        topk,
        candidate_indices,
        candidate_scores);

    tmac::hls::eagle4_lm_gather_dot_fp16(
        logits_hidden,
        lm_head_weight,
        candidate_indices,
        gathered_logits,
        tmac::hls::HIDDEN,
        topk);

    tmac::hls::eagle4_lm_softmax_topk(
        candidate_indices,
        gathered_logits,
        topk,
        topk_tokens,
        topk_probas,
        best_id,
        best_score);

    // Output: TREE_WIDTH * topk entries, laid out [t0_c0, t0_c1, ..., t1_c0, t1_c1, ...]
    if (candidate_indices_out != nullptr) {
        for (int t = 0; t < tmac::hls::TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=tmac::hls::TREE_WIDTH max=tmac::hls::TREE_WIDTH
            for (int i = 0; i < topk; ++i) {
#pragma HLS loop_tripcount min=tmac::hls::kEagle4LmTopKMax max=tmac::hls::kEagle4LmTopKMax
#pragma HLS PIPELINE II=1
                candidate_indices_out[t * topk + i] = topk_tokens[t][i];
            }
        }
    }
    if (gathered_logits_out != nullptr) {
        for (int t = 0; t < tmac::hls::TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=tmac::hls::TREE_WIDTH max=tmac::hls::TREE_WIDTH
            for (int i = 0; i < topk; ++i) {
#pragma HLS loop_tripcount min=tmac::hls::kEagle4LmTopKMax max=tmac::hls::kEagle4LmTopKMax
#pragma HLS PIPELINE II=1
                gathered_logits_out[t * topk + i] = topk_probas[t][i];
            }
        }
    }
}
