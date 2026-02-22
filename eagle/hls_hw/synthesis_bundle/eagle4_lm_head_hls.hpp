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
    const float* hidden,               // [hidden_dim]
    const uint16_t* down_proj_weight,  // fp16, [rank, hidden_dim]
    float* low_rank,                   // [rank]
    int hidden_dim,
    int rank) {
    for (int r = 0; r < rank; ++r) {
        float acc = 0.0f;
        const size_t row_base = static_cast<size_t>(r) * static_cast<size_t>(hidden_dim);
        for (int h = 0; h < hidden_dim; ++h) {
            const float w = eagle4_fp16_to_float(down_proj_weight[row_base + static_cast<size_t>(h)]);
            acc += hidden[h] * w;
        }
        low_rank[r] = acc;
    }
}

inline void eagle4_lm_candidate_logits_row4(
    const float* low_rank,               // [rank]
    const int32_t* qweight_row_major,    // [vocab, rank/8]
    const uint16_t* scales_row_major,    // fp16, [rank/group_size, vocab]
    const int32_t* qzeros_packed,        // int32 packed zeros, [rank/group_size, ceil(vocab/8)] or nullptr
    const int32_t* g_idx,                // optional [rank], maps input channel -> group id
    int rank,
    int vocab,
    int group_size,
    float* logits_out,                   // [vocab] or nullptr
    int topk,
    int* topk_indices,                   // [topk] or nullptr
    float* topk_scores) {                // [topk] or nullptr
    const int in_packs = rank / 8;
    const int groups = rank / group_size;
    const int vocab_packed = (vocab + 7) / 8;
    const bool keep_topk = (topk > 0 && topk_indices != nullptr && topk_scores != nullptr);

    if (keep_topk) {
        for (int i = 0; i < topk; ++i) {
            topk_indices[i] = -1;
            topk_scores[i] = -std::numeric_limits<float>::infinity();
        }
    }

    for (int o = 0; o < vocab; ++o) {
        float acc = 0.0f;

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
                    // GPTQ stores packed qzeros with an implicit +1 during dequant usage.
                    zero = ((z >> ((o & 7) * 4)) & 0xF) + 1;
                }

                const int raw = (packed >> (j * 4)) & 0xF;
                const float w = static_cast<float>(raw - zero) * scale;
                acc += w * low_rank[k];
            }
        }

        if (logits_out != nullptr) {
            logits_out[o] = acc;
        }

        if (keep_topk) {
            const int min_pos = eagle4_lowest_slot(topk_scores, topk);
            if (acc > topk_scores[min_pos]) {
                topk_scores[min_pos] = acc;
                topk_indices[min_pos] = o;
            }
        }
    }
}

inline void eagle4_lm_gather_dot_fp16(
    const float* hidden,                 // [hidden_dim]
    const uint16_t* lm_head_weight,      // fp16, [vocab, hidden_dim]
    const int* candidate_indices,        // [num_candidates]
    float* gathered_logits,              // [num_candidates]
    int hidden_dim,
    int num_candidates) {
    for (int c = 0; c < num_candidates; ++c) {
        const int tok = candidate_indices[c];
        const size_t row_base = static_cast<size_t>(tok) * static_cast<size_t>(hidden_dim);
        float acc = 0.0f;
        for (int h = 0; h < hidden_dim; ++h) {
            const float w = eagle4_fp16_to_float(lm_head_weight[row_base + static_cast<size_t>(h)]);
            acc += hidden[h] * w;
        }
        gathered_logits[c] = acc;
    }
}

inline void eagle4_lm_best_of_candidates(
    const int* candidate_indices,        // [num_candidates]
    const float* gathered_logits,        // [num_candidates]
    int num_candidates,
    int* best_id,
    float* best_score) {
    int best_tok = -1;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < num_candidates; ++c) {
        if (gathered_logits[c] > best_val) {
            best_val = gathered_logits[c];
            best_tok = candidate_indices[c];
        }
    }
    *best_id = best_tok;
    *best_score = best_val;
}

} // namespace hls
} // namespace tmac

#endif // TMAC_EAGLE4_LM_HEAD_HLS_HPP
