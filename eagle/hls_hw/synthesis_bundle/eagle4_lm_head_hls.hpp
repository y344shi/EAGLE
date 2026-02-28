#ifndef TMAC_EAGLE4_LM_HEAD_HLS_HPP
#define TMAC_EAGLE4_LM_HEAD_HLS_HPP

#include <cstdint>
#include <cstring>
#include <limits>
#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

constexpr int kEagle4LmHiddenMax = 4096;
constexpr int kEagle4LmRankMax = 256;
constexpr int kEagle4LmTopKMax = 1024;

float e4_f16(uint16_t h);

int e4_min_slot(const float* scores, int topk);

void e4_lm_down(
    const float logits_hidden[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmHiddenMax],
    const uint16_t* down_proj_weight,  // fp16, [rank, hidden_dim]
    float low_rank[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmRankMax],                   // [rank]
    int hidden_dim,
    int rank);

void e4_lm_cand4(
    const float low_rank[TREE_WIDTH][kEagle4LmRankMax],  // [TREE_WIDTH, rank]
    const int32_t* qweight_row_major,    // [vocab, rank/8]
    const uint16_t* scales_row_major,    // fp16, [rank/group_size, vocab]
    const int32_t* qzeros_packed,        // int32 packed zeros, [rank/group_size, ceil(vocab/8)] or nullptr
    const int32_t* g_idx,                // optional [rank], maps input channel -> group id
    int rank,
    int vocab,
    int group_size,
    float* logits_out,                   // [vocab] or nullptr (unused for tree expansion)
    int topk,
    int topk_indices[TREE_WIDTH][kEagle4LmTopKMax],   // [TREE_WIDTH, topk]
    float topk_scores[TREE_WIDTH][kEagle4LmTopKMax]);

void e4_lm_gdot(
    const float hidden[TREE_WIDTH][kEagle4LmHiddenMax],  // [TREE_WIDTH, hidden_dim]
    const uint16_t* lm_head_weight,      // fp16, [vocab, hidden_dim]
    const int candidate_indices[TREE_WIDTH][kEagle4LmTopKMax],  // [TREE_WIDTH, num_candidates]
    float gathered_logits[TREE_WIDTH][kEagle4LmTopKMax],        // [TREE_WIDTH, num_candidates]
    int hidden_dim,
    int num_candidates);

// Per-token softmax over gathered logits, producing probabilities for all topk candidates.
// This is the Gap-2 fix: replaces best_of_candidates with softmax probabilities
// that feed into the tree expansion fused step.
void e4_lm_softmax(
    const int candidate_indices[TREE_WIDTH][kEagle4LmTopKMax],   // [TREE_WIDTH, topk]
    const float gathered_logits[TREE_WIDTH][kEagle4LmTopKMax],   // [TREE_WIDTH, topk]
    int num_candidates,
    int topk_tokens_out[TREE_WIDTH][kEagle4LmTopKMax],           // [TREE_WIDTH, topk] token IDs
    float topk_probas_out[TREE_WIDTH][kEagle4LmTopKMax],         // [TREE_WIDTH, topk] probabilities
    int* best_id,                                                 // overall best (backward compat)
    float* best_score);

} // namespace hls
} // namespace tmac

#endif // TMAC_EAGLE4_LM_HEAD_HLS_HPP
