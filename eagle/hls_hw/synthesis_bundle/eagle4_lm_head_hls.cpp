#include "eagle4_lm_head_hls.hpp"
#include <cmath>
#include <hls_half.h>

namespace tmac {
namespace hls {

// Tripcount constants for C synthesis latency estimation (only for values without existing named constants).
constexpr int kLmTcQpackFactor = 8;   // int4 values per int32
constexpr int kLmTcVocab       = 32000; // typical vocab size

float eagle4_fp16_to_float(uint16_t h) {
    #pragma HLS INLINE
    
    half fp16_val;
    // Safely copy the 16 bits into the half-precision variable
    std::memcpy(&fp16_val, &h, sizeof(half));
    
    // Value-cast the half to a full 32-bit float
    return static_cast<float>(fp16_val);
}

int e4_min_slot(const float* scores, int topk) {
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

void e4_lm_down(
    const float logits_hidden[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmHiddenMax],
    const uint16_t* down_proj_weight,  // fp16, [rank, hidden_dim]
    float low_rank[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmRankMax],                   // [rank]
    int hidden_dim,
    int rank) {

    for (int r = 0; r < rank; ++r) {
#pragma HLS loop_tripcount min=kEagle4LmRankMax max=kEagle4LmRankMax avg=kEagle4LmRankMax
        const size_t row_base = static_cast<size_t>(r) * static_cast<size_t>(hidden_dim);
        for (int t = 0; t < TREE_WIDTH; t++) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
            float acc = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
#pragma HLS loop_tripcount min=kEagle4LmHiddenMax max=kEagle4LmHiddenMax avg=kEagle4LmHiddenMax
                const float w = e4_f16(down_proj_weight[row_base + static_cast<size_t>(h)]);
                acc += logits_hidden[t][h] * w;
            }
            low_rank[t][r] = acc;
        }
    }
}

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
    float topk_scores[TREE_WIDTH][kEagle4LmTopKMax]) { // [TREE_WIDTH, topk]
    const int in_packs = rank / 8;
    const int groups = rank / group_size;
    const int vocab_packed = (vocab + 7) / 8;
    const bool keep_topk = (topk > 0 && topk_indices != nullptr && topk_scores != nullptr);

    if (keep_topk) {
        for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
            for (int i = 0; i < topk; ++i) {
#pragma HLS loop_tripcount min=1 max=kEagle4LmTopKMax avg=(1+kEagle4LmTopKMax)/2
                topk_indices[t][i] = -1;
                topk_scores[t][i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    for (int o = 0; o < vocab; ++o) {
#pragma HLS loop_tripcount min=kLmTcVocab max=kLmTcVocab avg=kLmTcVocab
        // Dequantize weights for this vocab entry (shared across all tree candidates)
        float dequant_w[kEagle4LmRankMax];
        for (int p = 0; p < in_packs; ++p) {
#pragma HLS loop_tripcount min=kEagle4LmRankMax/8 max=kEagle4LmRankMax/8 avg=kEagle4LmRankMax/8
            const int k_base = p * 8;
            const int32_t packed = qweight_row_major[static_cast<size_t>(o) * static_cast<size_t>(in_packs) + p];
            for (int j = 0; j < 8; ++j) {
#pragma HLS loop_tripcount min=kLmTcQpackFactor max=kLmTcQpackFactor avg=kLmTcQpackFactor
                const int k = k_base + j;
                const int g = (g_idx != nullptr) ? g_idx[k] : (k / group_size);
                const float scale = e4_f16(
                    scales_row_major[static_cast<size_t>(g) * static_cast<size_t>(vocab) + o]);

                int zero = 8;
                if (qzeros_packed != nullptr) {
                    const int32_t z =
                        qzeros_packed[static_cast<size_t>(g) * static_cast<size_t>(vocab_packed) + (o >> 3)];
                    zero = ((z >> ((o & 7) * 4)) & 0xF) + 1;
                }

                const int raw = (packed >> (j * 4)) & 0xF;
                dequant_w[k] = static_cast<float>(raw - zero) * scale;
            }
        }

        // Compute dot product for each tree candidate
        for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
            float acc = 0.0f;
            for (int k = 0; k < rank; ++k) {
#pragma HLS loop_tripcount min=kEagle4LmRankMax max=kEagle4LmRankMax avg=kEagle4LmRankMax
                acc += dequant_w[k] * low_rank[t][k];
            }

            if (keep_topk) {
                const int min_pos = e4_min_slot(topk_scores[t], topk);
                if (acc > topk_scores[t][min_pos]) {
                    topk_scores[t][min_pos] = acc;
                    topk_indices[t][min_pos] = o;
                }
            }
        }
    }
}

void e4_lm_gdot(
    const float hidden[TREE_WIDTH][kEagle4LmHiddenMax],  // [TREE_WIDTH, hidden_dim]
    const uint16_t* lm_head_weight,      // fp16, [vocab, hidden_dim]
    const int candidate_indices[TREE_WIDTH][kEagle4LmTopKMax],  // [TREE_WIDTH, num_candidates]
    float gathered_logits[TREE_WIDTH][kEagle4LmTopKMax],        // [TREE_WIDTH, num_candidates]
    int hidden_dim,
    int num_candidates) {
    for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
        for (int c = 0; c < num_candidates; ++c) {
#pragma HLS loop_tripcount min=kEagle4LmTopKMax max=kEagle4LmTopKMax avg=kEagle4LmTopKMax
            const int tok = candidate_indices[t][c];
            const size_t row_base = static_cast<size_t>(tok) * static_cast<size_t>(hidden_dim);
            float acc = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
#pragma HLS loop_tripcount min=kEagle4LmHiddenMax max=kEagle4LmHiddenMax avg=kEagle4LmHiddenMax
                const float w = e4_f16(lm_head_weight[row_base + static_cast<size_t>(h)]);
                acc += hidden[t][h] * w;
            }
            gathered_logits[t][c] = acc;
        }
    }
}

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
    float* best_score) {                                          // overall best score
    int global_best_tok = -1;
    float global_best_val = -std::numeric_limits<float>::infinity();

    for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
        // Find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < num_candidates; ++c) {
#pragma HLS loop_tripcount min=kEagle4LmTopKMax max=kEagle4LmTopKMax avg=kEagle4LmTopKMax
            if (gathered_logits[t][c] > max_val) {
                max_val = gathered_logits[t][c];
            }
        }

        // Compute exp(logit - max) and sum
        float sum_exp = 0.0f;
        for (int c = 0; c < num_candidates; ++c) {
#pragma HLS loop_tripcount min=kEagle4LmTopKMax max=kEagle4LmTopKMax avg=kEagle4LmTopKMax
            float e = std::exp(gathered_logits[t][c] - max_val);
            topk_probas_out[t][c] = e;
            sum_exp += e;
        }

        // Normalize to probabilities and copy token IDs
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int c = 0; c < num_candidates; ++c) {
#pragma HLS loop_tripcount min=kEagle4LmTopKMax max=kEagle4LmTopKMax avg=kEagle4LmTopKMax
            topk_probas_out[t][c] *= inv_sum;
            topk_tokens_out[t][c] = candidate_indices[t][c];
        }

        // Track global best (backward compat)
        if (max_val > global_best_val) {
            global_best_val = max_val;
            // Find which candidate has max_val
            for (int c = 0; c < num_candidates; ++c) {
#pragma HLS loop_tripcount min=kEagle4LmTopKMax max=kEagle4LmTopKMax avg=kEagle4LmTopKMax
                if (gathered_logits[t][c] == max_val) {
                    global_best_tok = candidate_indices[t][c];
                    break;
                }
            }
        }
    }

    *best_id = global_best_tok;
    *best_score = global_best_val;
}

} // namespace hls
} // namespace tmac
