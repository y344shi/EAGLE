#ifndef TMAC_EAGLE_TIER1_LM_TOP_HPP
#define TMAC_EAGLE_TIER1_LM_TOP_HPP

#include <cstdint>

#include "eagle_tier1_top.hpp"
#include "eagle4_lm_head_hls.hpp"

// LM-head compatibility modes:
// - EAGLE4 efficient LM-head path is the integration target.
// - Legacy Eagle3 lm_head_8way path is kept only for regression builds.
#ifndef TMAC_ALLOW_LEGACY_LM_HEAD8WAY
#define TMAC_ALLOW_LEGACY_LM_HEAD8WAY 0
#endif

#if TMAC_ALLOW_LEGACY_LM_HEAD8WAY
#include "lm_head_8way.hpp"
#else
// Compatibility placeholder when legacy lm_head_8way is disabled.
struct wide_vec_t {
    uint16_t data[32];
};
#endif

// Super-wrapper: EAGLE4 layer-0 parity tier1 block -> 8-way LM head (single token, batch slot 0).
// `reasoning_state_out` stores the post-residual reasoning stream (HIDDEN floats) for next-step recurrence.
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
                        // EAGLE4 efficient LM-head integration inputs (optional for transitional wiring).
                        const uint16_t efficient_lm_head_down_proj_weight[tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax] = nullptr, // fp16 [rank, hidden]
                        const int32_t efficient_lm_head_qweight_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks] = nullptr, // int32 [vocab, rank/8]
                        const uint16_t efficient_lm_head_scales_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups] = nullptr, // fp16 [vocab, rank/group]
                        const int32_t efficient_lm_head_qzeros[tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups] = nullptr,      // packed int32 [ceil(vocab/8), rank/group]
                        const int32_t efficient_lm_head_g_idx[tmac::hls::kEagle4LmRankMax] = nullptr,                                   // optional [rank]
                        const uint16_t* lm_head_weight = nullptr,                       // fp16 [vocab, hidden]
                        int efficient_lm_rank = 0,
                        int efficient_lm_vocab_size = 0,
                        int efficient_lm_num_candidates = 0,
                        int* candidate_indices_out = nullptr,                           // optional [num_candidates]
                        float* gathered_logits_out = nullptr,                           // optional [num_candidates]
                        int prefix_len = 0,
                        int current_depth = 0,
                        const int* parent_indices_per_layer = nullptr);

// EAGLE4 parity wrapper with efficient LM-head path:
//   1) SLM forward (TREE_WIDTH tokens)
//   2) hidden -> low-rank projection (per token)
//   3) candidate scoring via GPTQ row-major (per token)
//   4) gather-dot over lm_head.weight for selected candidates (per token)
//   5) softmax over gathered logits -> topk probabilities (per token)
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
                               const float rope_cos_vals[tmac::hls::HEAD_DIM / 2],
                               const float rope_sin_vals[tmac::hls::HEAD_DIM / 2],
                               tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_k,
                               tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_v,
                                const uint16_t efficient_lm_head_down_proj_weight[tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax], // fp16 [rank, hidden]
                                const int32_t efficient_lm_head_qweight_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks], // int32 [vocab, rank/8]
                                const uint16_t efficient_lm_head_scales_row_major[tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups], // fp16 [vocab, rank/group]
                                const int32_t efficient_lm_head_qzeros[tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups],      // packed int32 [ceil(vocab/8), rank/group] or nullptr
                                const int32_t efficient_lm_head_g_idx[tmac::hls::kEagle4LmRankMax],                                   // optional [rank]
                               const uint16_t* lm_head_weight,                      // fp16 [vocab, hidden]
                               int efficient_lm_rank,
                               int efficient_lm_vocab_size,
                               int efficient_lm_num_candidates,
                               float* reasoning_state_out,                          // [TREE_WIDTH * HIDDEN]
                               int* candidate_indices_out,      // [TREE_WIDTH * num_candidates] token IDs
                               float* gathered_logits_out,      // [TREE_WIDTH * num_candidates] softmax probabilities
                               int prefix_len,
                               int current_depth,
                               const int* parent_indices_per_layer);

#endif // TMAC_EAGLE_TIER1_LM_TOP_HPP
