#ifndef TMAC_EAGLE4_LM_HEAD_HLS_HPP
#define TMAC_EAGLE4_LM_HEAD_HLS_HPP

#include <cstdint>
#include <cstring>
#include <limits>

namespace tmac {
namespace hls {

constexpr int kEagle4LmHiddenMax = 4096;
constexpr int kEagle4LmRankMax = 256;
constexpr int kEagle4LmTopKMax = 1024;

inline float eagle4_fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

inline int eagle4_lowest_slot(const float* scores, int topk) {
    int min_pos = 0;
    float min_val = scores[0];
    for (int i = 1; i < topk; ++i) {
        if (scores[i] < min_val) {
            min_val = scores[i];
            min_pos = i;
        }
    }
    return min_pos;
}

inline void eagle4_lm_down_project(
    const float logits_hidden[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmHiddenMax],
    const uint16_t* down_proj_weight,  // fp16, [rank, hidden_dim]
    float low_rank[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmRankMax],                   // [rank]
    int hidden_dim,
    int rank) {
    for (int r = 0; r < rank; ++r) {
        const size_t row_base = static_cast<size_t>(r) * static_cast<size_t>(hidden_dim);
        for (int t = 0; t < TREE_WIDTH; t++) {
            float acc = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                const float w = eagle4_fp16_to_float(down_proj_weight[row_base + static_cast<size_t>(h)]);
                acc += logits_hidden[t][h] * w;
            }
            low_rank[t][r] = acc;
        }
    }
}

inline void eagle4_lm_candidate_logits_row4(
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
            for (int i = 0; i < topk; ++i) {
                topk_indices[t][i] = -1;
                topk_scores[t][i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    for (int o = 0; o < vocab; ++o) {
        // Dequantize weights for this vocab entry (shared across all tree candidates)
        float dequant_w[kEagle4LmRankMax];
        for (int p = 0; p < in_packs; ++p) {
            const int k_base = p * 8;
            const int32_t packed = qweight_row_major[static_cast<size_t>(o) * static_cast<size_t>(in_packs) + p];
            for (int j = 0; j < 8; ++j) {
                const int k = k_base + j;
                const int g = (g_idx != nullptr) ? g_idx[k] : (k / group_size);
                const float scale = eagle4_fp16_to_float(
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
            float acc = 0.0f;
            for (int k = 0; k < rank; ++k) {
                acc += dequant_w[k] * low_rank[t][k];
            }

            if (keep_topk) {
                const int min_pos = eagle4_lowest_slot(topk_scores[t], topk);
                if (acc > topk_scores[t][min_pos]) {
                    topk_scores[t][min_pos] = acc;
                    topk_indices[t][min_pos] = o;
                }
            }
        }
    }
}

inline void eagle4_lm_gather_dot_fp16(
    const float hidden[TREE_WIDTH][kEagle4LmHiddenMax],  // [TREE_WIDTH, hidden_dim]
    const uint16_t* lm_head_weight,      // fp16, [vocab, hidden_dim]
    const int candidate_indices[TREE_WIDTH][kEagle4LmTopKMax],  // [TREE_WIDTH, num_candidates]
    float gathered_logits[TREE_WIDTH][kEagle4LmTopKMax],        // [TREE_WIDTH, num_candidates]
    int hidden_dim,
    int num_candidates) {
    for (int t = 0; t < TREE_WIDTH; ++t) {
        for (int c = 0; c < num_candidates; ++c) {
            const int tok = candidate_indices[t][c];
            const size_t row_base = static_cast<size_t>(tok) * static_cast<size_t>(hidden_dim);
            float acc = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                const float w = eagle4_fp16_to_float(lm_head_weight[row_base + static_cast<size_t>(h)]);
                acc += hidden[t][h] * w;
            }
            gathered_logits[t][c] = acc;
        }
    }
}

// Per-token softmax over gathered logits, producing probabilities for all topk candidates.
// This is the Gap-2 fix: replaces best_of_candidates with softmax probabilities
// that feed into the tree expansion fused step.
inline void eagle4_lm_softmax_topk(
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
        // Find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < num_candidates; ++c) {
            if (gathered_logits[t][c] > max_val) {
                max_val = gathered_logits[t][c];
            }
        }

        // Compute exp(logit - max) and sum
        float sum_exp = 0.0f;
        for (int c = 0; c < num_candidates; ++c) {
            float e = std::exp(gathered_logits[t][c] - max_val);
            topk_probas_out[t][c] = e;
            sum_exp += e;
        }

        // Normalize to probabilities and copy token IDs
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int c = 0; c < num_candidates; ++c) {
            topk_probas_out[t][c] *= inv_sum;
            topk_tokens_out[t][c] = candidate_indices[t][c];
        }

        // Track global best (backward compat)
        if (max_val > global_best_val) {
            global_best_val = max_val;
            // Find which candidate has max_val
            for (int c = 0; c < num_candidates; ++c) {
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

#endif // TMAC_EAGLE4_LM_HEAD_HLS_HPP
