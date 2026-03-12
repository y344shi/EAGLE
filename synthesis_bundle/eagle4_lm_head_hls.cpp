#include "eagle4_lm_head_hls.hpp"

namespace tmac {
namespace hls {

void eagle4_lm_down_project(
    const float logits_hidden[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmHiddenMax],
    const uint16_t down_proj_weight[kEagle4LmRankMax * kEagle4LmHiddenMax],  // fp16, [rank, hidden_dim]
    float low_rank[tmac::hls::TREE_WIDTH][tmac::hls::kEagle4LmRankMax],                   // [rank]
    int hidden_dim,
    int rank) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=logits_hidden type=cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=low_rank type=complete dim=0

    for (int r = 0; r < rank; ++r) {
#pragma HLS loop_tripcount min=kEagle4LmRankMax max=kEagle4LmRankMax avg=kEagle4LmRankMax
        const size_t row_base = static_cast<size_t>(r) * static_cast<size_t>(hidden_dim);
        for (int t = 0; t < TREE_WIDTH; t++) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
            float acc = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
#pragma HLS loop_tripcount min=kEagle4LmHiddenMax max=kEagle4LmHiddenMax avg=kEagle4LmHiddenMax
                const float w = eagle4_fp16_to_float(down_proj_weight[row_base + static_cast<size_t>(h)]);
                acc += logits_hidden[t][h] * w;
            }
            low_rank[t][r] = acc;
        }
    }
}

void eagle4_lm_candidate_logits_row4(
    const float low_rank[TREE_WIDTH][kEagle4LmRankMax],  // [TREE_WIDTH, rank]
    const int32_t qweight_row_major[kLmTcVocab * kLmMaxInPacks],    // [vocab, rank/8]
    const uint16_t scales_row_major[kLmTcVocab * kLmMaxGroups],     // fp16, [vocab, rank/group_size] (transposed/output-major)
    const int32_t qzeros_packed[kLmMaxVocabPacked * kLmMaxGroups],  // int32 packed zeros, [ceil(vocab/8), rank/group_size] (transposed/output-major)
    const int32_t g_idx[kEagle4LmRankMax],                          // optional [rank], maps input channel -> group id
    int rank,
    int vocab,
    int group_size,
    float* logits_out,                   // [vocab] or nullptr (unused for tree expansion)
    int topk,
    int topk_indices[TREE_WIDTH][kEagle4LmTopKMax],   // [TREE_WIDTH, topk]
    float topk_scores[TREE_WIDTH][kEagle4LmTopKMax]) { // [TREE_WIDTH, topk]
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=low_rank type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=topk_indices type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=topk_scores type=complete dim=0
#pragma HLS BIND_STORAGE variable=qweight_row_major type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=scales_row_major type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=qzeros_packed type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=qweight_row_major type=cyclic factor=128 dim=1
#pragma HLS ARRAY_PARTITION variable=scales_row_major type=cyclic factor=128 dim=1
#pragma HLS ARRAY_PARTITION variable=qzeros_packed type=cyclic factor=128 dim=1
#pragma HLS ARRAY_PARTITION variable=g_idx type=complete dim=1
    const int in_packs = rank / 8;
    const int groups = (rank + group_size - 1) / group_size;
    const bool keep_topk = (topk > 0 && topk_indices != nullptr && topk_scores != nullptr);

    if (keep_topk) {
        for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
            for (int i = 0; i < kEagle4LmTopKMax; ++i) {
#pragma HLS UNROLL
                if (i < topk) {
                    topk_indices[t][i] = -1;
                    topk_scores[t][i] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    for (int o = 0; o < vocab; ++o) {
#pragma HLS loop_tripcount min=kLmTcVocab max=kLmTcVocab avg=kLmTcVocab
#pragma HLS PIPELINE off
        // Dequantize weights for this vocab entry (shared across all tree candidates).
        // Weight/scale/zero tensors are expected to be preloaded in BRAM at entrypoint.
        float dequant_w[kEagle4LmRankMax];
#pragma HLS ARRAY_PARTITION variable=dequant_w complete dim=0
        for (int p = 0; p < kEagle4LmRankMax / 8; ++p) {
#pragma HLS UNROLL
            const int k_base = p * 8;
            int32_t packed = 0;
            if (p < in_packs) {
                packed = qweight_row_major[static_cast<size_t>(o) * static_cast<size_t>(in_packs) + p];
            }
            for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL
                const int k = k_base + j;
                if (k < rank) {
                    const int g = (g_idx != nullptr) ? g_idx[k] : (k / group_size);
                    const float scale = eagle4_fp16_to_float(
                        scales_row_major[static_cast<size_t>(o) * static_cast<size_t>(groups) + g]);

                    int zero = 8;
                    if (qzeros_packed != nullptr) {
                        const int32_t z =
                            qzeros_packed[static_cast<size_t>(o >> 3) * static_cast<size_t>(groups) + g];
                        zero = ((z >> ((o & 7) * 4)) & 0xF) + 1;
                    }

                    const int raw = (packed >> (j * 4)) & 0xF;
                    dequant_w[k] = static_cast<float>(raw - zero) * scale;
                } else {
                    dequant_w[k] = 0.0f;
                }
            }
        }

        // Compute dot product for each tree candidate
        for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
//#pragma HLS PIPELINE off
#pragma HLS UNROLL
            float acc = 0.0f;
            for (int k = 0; k < kEagle4LmRankMax; ++k) {
#pragma HLS UNROLL
                if (k < rank) {
                    acc += dequant_w[k] * low_rank[t][k];
                }
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

void eagle4_lm_gather_dot_fp16(
    const float hidden[TREE_WIDTH][kEagle4LmHiddenMax],  // [TREE_WIDTH, hidden_dim]
    const uint16_t* lm_head_weight,      // fp16, [vocab, hidden_dim]
    const int candidate_indices[TREE_WIDTH][kEagle4LmTopKMax],  // [TREE_WIDTH, num_candidates]
    float gathered_logits[TREE_WIDTH][kEagle4LmTopKMax],        // [TREE_WIDTH, num_candidates]
    int hidden_dim,
    int num_candidates) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=hidden type=cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=candidate_indices type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=gathered_logits type=complete dim=0
    for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
        for (int c = 0; c < num_candidates; ++c) {
#pragma HLS loop_tripcount min=kEagle4LmTopKMax max=kEagle4LmTopKMax avg=kEagle4LmTopKMax
            const int tok = candidate_indices[t][c];
            const size_t row_base = static_cast<size_t>(tok) * static_cast<size_t>(hidden_dim);
            float acc = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
#pragma HLS loop_tripcount min=kEagle4LmHiddenMax max=kEagle4LmHiddenMax avg=kEagle4LmHiddenMax
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
void eagle4_lm_softmax_topk(
    const int candidate_indices[TREE_WIDTH][kEagle4LmTopKMax],   // [TREE_WIDTH, topk]
    const float gathered_logits[TREE_WIDTH][kEagle4LmTopKMax],   // [TREE_WIDTH, topk]
    int num_candidates,
    int topk_tokens_out[TREE_WIDTH][kEagle4LmTopKMax],           // [TREE_WIDTH, topk] token IDs
    float topk_probas_out[TREE_WIDTH][kEagle4LmTopKMax],         // [TREE_WIDTH, topk] probabilities
    int* best_id,                                                 // overall best (backward compat)
    float* best_score) {                                          // overall best score
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=candidate_indices type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=gathered_logits type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=topk_tokens_out type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=topk_probas_out type=complete dim=0
    int global_best_tok = -1;
    float global_best_val = -std::numeric_limits<float>::infinity();

    for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
        // Find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < kEagle4LmTopKMax; ++c) {
#pragma HLS UNROLL
            if (c < num_candidates && gathered_logits[t][c] > max_val) {
                max_val = gathered_logits[t][c];
            }
        }

        // Compute exp(logit - max) and sum
        float sum_exp = 0.0f;
        for (int c = 0; c < kEagle4LmTopKMax; ++c) {
#pragma HLS UNROLL
            if (c < num_candidates) {
                float e = std::exp(gathered_logits[t][c] - max_val);
                topk_probas_out[t][c] = e;
                sum_exp += e;
            }
        }

        // Normalize to probabilities and copy token IDs
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int c = 0; c < kEagle4LmTopKMax; ++c) {
#pragma HLS UNROLL
            if (c < num_candidates) {
                topk_probas_out[t][c] *= inv_sum;
                topk_tokens_out[t][c] = candidate_indices[t][c];
            }
        }

        // Track global best (backward compat)
        if (max_val > global_best_val) {
            global_best_val = max_val;
            // Find which candidate has max_val
            for (int c = 0; c < kEagle4LmTopKMax; ++c) {
#pragma HLS UNROLL
                if (c < num_candidates && gathered_logits[t][c] == max_val) {
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
