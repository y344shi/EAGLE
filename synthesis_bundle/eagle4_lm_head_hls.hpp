#ifndef TMAC_EAGLE4_LM_HEAD_HLS_HPP
#define TMAC_EAGLE4_LM_HEAD_HLS_HPP

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <hls_half.h>
#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

constexpr int kEagle4LmHiddenMax = 4096;
constexpr int kEagle4LmRankMax = 256;
constexpr int kEagle4LmTopKMax = 1024;

// Tripcount constants for C synthesis latency estimation (only for values without existing named constants).
constexpr int kLmTcQpackFactor = 8;   // int4 values per int32
constexpr int kLmTcVocab       = 32000; // typical vocab size

inline float eagle4_fp16_to_float(uint16_t h) {
#pragma HLS INLINE
    half fp16_val;
    std::memcpy(&fp16_val, &h, sizeof(half));
    return static_cast<float>(fp16_val);
}

inline int eagle4_lowest_slot(const float* scores, int topk) {
    int min_pos = 0;
    float min_val = scores[0];
    for (int i = 1; i < topk; ++i) {
#pragma HLS loop_tripcount min=1 max=kEagle4LmTopKMax avg=(1+kEagle4LmTopKMax)/2
        if (scores[i] < min_val) {
            min_val = scores[i];
            min_pos = i;
        }
    }
    return min_pos;
}

void eagle4_lm_down_project(
    const float logits_hidden[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmHiddenMax],
    const uint16_t* down_proj_weight,  // fp16, [rank, hidden_dim]
    float low_rank[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmRankMax],
    int hidden_dim,
    int rank);

void eagle4_lm_candidate_logits_row4(
    const float low_rank[TREE_WIDTH][kEagle4LmRankMax],
    const int32_t* qweight_row_major,
    const uint16_t* scales_row_major,
    const int32_t* qzeros_packed,
    const int32_t* g_idx,
    int rank,
    int vocab,
    int group_size,
    float* logits_out,
    int topk,
    int topk_indices[TREE_WIDTH][kEagle4LmTopKMax],
    float topk_scores[TREE_WIDTH][kEagle4LmTopKMax]);

void eagle4_lm_gather_dot_fp16(
    const float hidden[TREE_WIDTH][kEagle4LmHiddenMax],
    const uint16_t* lm_head_weight,
    const int candidate_indices[TREE_WIDTH][kEagle4LmTopKMax],
    float gathered_logits[TREE_WIDTH][kEagle4LmTopKMax],
    int hidden_dim,
    int num_candidates);

// Per-token softmax over gathered logits, producing probabilities for all topk candidates.
// This is the Gap-2 fix: replaces best_of_candidates with softmax probabilities
// that feed into the tree expansion fused step.
void eagle4_lm_softmax_topk(
    const int candidate_indices[TREE_WIDTH][kEagle4LmTopKMax],
    const float gathered_logits[TREE_WIDTH][kEagle4LmTopKMax],
    int num_candidates,
    int topk_tokens_out[TREE_WIDTH][kEagle4LmTopKMax],
    float topk_probas_out[TREE_WIDTH][kEagle4LmTopKMax],
    int* best_id,
    float* best_score);

} // namespace hls
} // namespace tmac

#endif // TMAC_EAGLE4_LM_HEAD_HLS_HPP
