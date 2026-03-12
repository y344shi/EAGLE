#ifndef TMAC_EAGLE4_LM_HEAD_HLS_HPP
#define TMAC_EAGLE4_LM_HEAD_HLS_HPP

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#ifdef __SYNTHESIS__
#include <hls_half.h>
#endif
#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

constexpr int kEagle4LmHiddenMax = 4096;
constexpr int kEagle4LmRankMax = 256;
constexpr int kEagle4LmTopKMax = 16;
constexpr int kEagle4LmGroupSize = 128;

// Tripcount constants for C synthesis latency estimation (only for values without existing named constants).
constexpr int kLmTcQpackFactor = 8;   // int4 values per int32
constexpr int kLmTcVocab       = 128256; // Llama-3.1-8B full vocab
constexpr int kLmMaxInPacks = kEagle4LmRankMax / kLmTcQpackFactor;
constexpr int kLmMaxGroups = (kEagle4LmRankMax + kEagle4LmGroupSize - 1) / kEagle4LmGroupSize;
constexpr int kLmMaxVocabPacked = (kLmTcVocab + 7) / 8;

inline float eagle4_fp16_to_float(uint16_t h) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
    half fp16_val;
    std::memcpy(&fp16_val, &h, sizeof(half));
    return static_cast<float>(fp16_val);
#else
    // Software IEEE-754 fp16 decode for desktop compilation.
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp5 = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    float result;
    if (exp5 == 0) {
        // subnormal or zero
        result = std::ldexp(static_cast<float>(frac), -24);
    } else if (exp5 == 0x1f) {
        // inf / nan
        result = (frac == 0) ? std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::quiet_NaN();
    } else {
        result = std::ldexp(static_cast<float>(frac + 1024), exp5 - 25);
    }
    return sign ? -result : result;
#endif
}

inline int eagle4_lowest_slot(const float* scores, int topk) {
#pragma HLS INLINE
    int min_pos = 0;
    float min_val = scores[0];
    for (int i = 1; i < kEagle4LmTopKMax; ++i) {
#pragma HLS UNROLL
        if (i < topk && scores[i] < min_val) {
            min_val = scores[i];
            min_pos = i;
        }
    }
    return min_pos;
}

void eagle4_lm_down_project(
    const float logits_hidden[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmHiddenMax],
    const uint16_t down_proj_weight[kEagle4LmRankMax * kEagle4LmHiddenMax],  // fp16, [rank, hidden_dim]
    float low_rank[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmRankMax],
    int hidden_dim,
    int rank);

void eagle4_lm_candidate_logits_row4(
    const float low_rank[TREE_WIDTH][kEagle4LmRankMax],
    const int32_t qweight_row_major[kLmTcVocab * kLmMaxInPacks],
    const uint16_t scales_row_major[kLmTcVocab * kLmMaxGroups],    // expected [vocab, rank/group_size] (transposed/output-major)
    const int32_t qzeros_packed[kLmMaxVocabPacked * kLmMaxGroups], // expected [ceil(vocab/8), rank/group_size] (transposed/output-major)
    const int32_t g_idx[kEagle4LmRankMax],
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
