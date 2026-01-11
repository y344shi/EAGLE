#ifndef TMAC_ATTENTION_SOLVER_HPP
#define TMAC_ATTENTION_SOLVER_HPP

// Streaming attention solver for one head using online softmax.
// - Stationary Q held on-chip.
// - Streams K/V history from the KV cache manager.
// - Computes scaled dot, online softmax, and context in a single pass.

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

using tmac::hls::VEC_W;
using tmac::hls::vec_t;
using tmac::hls::hls_stream;

// HEAD_DIM must be divisible by VEC_W.
template <int HEAD_DIM>
// Optional padded_len lets us mirror Marlin's padded decode length; masked tokens contribute zero.
void attention_solver(hls_stream<vec_t<VEC_W>>& q_stream,     // one head query
                      hls_stream<vec_t<VEC_W>>& k_hist,       // history K
                      hls_stream<vec_t<VEC_W>>& v_hist,       // history V
                      hls_stream<vec_t<VEC_W>>& context_out,  // context
                      int seq_len,
                      int padded_len = -1) {
#pragma HLS INTERFACE axis port = q_stream
#pragma HLS INTERFACE axis port = k_hist
#pragma HLS INTERFACE axis port = v_hist
#pragma HLS INTERFACE axis port = context_out
#pragma HLS INTERFACE s_axilite port = seq_len bundle = control
#pragma HLS INTERFACE s_axilite port = padded_len bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static_assert(HEAD_DIM % VEC_W == 0, "HEAD_DIM must align to VEC_W");

    const int total_len = (padded_len > 0) ? padded_len : seq_len;

    // Stationary Q buffer
    float q_buffer[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = q_buffer cyclic factor = VEC_W

    // Load Q
    for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<VEC_W> chunk = q_stream.read();
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
            q_buffer[i * VEC_W + j] = chunk[j];
        }
    }

    // Online softmax state
    float m_prev = -1e30f;
    float d_prev = 0.0f;

    float o_buffer[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = o_buffer cyclic factor = VEC_W
    for (int i = 0; i < HEAD_DIM; ++i) {
#pragma HLS UNROLL
        o_buffer[i] = 0.0f;
    }

    const float scale = 1.0f / ::hls::sqrt(static_cast<float>(HEAD_DIM));

    // Process history tokens
    token_loop:
    for (int t = 0; t < total_len; ++t) {
        float partial_scores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float v_local[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = v_local cyclic factor = VEC_W

        if (t < seq_len) {
            // Dot(Q, K_t) while latching V_t
            for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
                vec_t<VEC_W> k_chunk = k_hist.read();
                vec_t<VEC_W> v_chunk = v_hist.read();
                for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                    const int idx = i * VEC_W + j;
                    partial_scores[j & 0x3] += q_buffer[idx] * k_chunk[j];
                    v_local[idx] = v_chunk[j];
                }
            }
        } else {
            // Padded/masked token: zero contribution, very negative score
            for (int i = 0; i < HEAD_DIM; ++i) {
#pragma HLS UNROLL
                v_local[i] = 0.0f;
            }
            partial_scores[0] = partial_scores[1] = partial_scores[2] = partial_scores[3] = -1e9f;
        }

        float score = (partial_scores[0] + partial_scores[1]) +
                      (partial_scores[2] + partial_scores[3]);
        score *= scale;

        // Online softmax update
        const float m_new = (score > m_prev) ? score : m_prev;
        const float corr = ::hls::exp(m_prev - m_new);
        const float new_term = ::hls::exp(score - m_new);
        const float d_new = d_prev * corr + new_term;

        // Update context
        for (int i = 0; i < HEAD_DIM; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor = VEC_W
            o_buffer[i] = o_buffer[i] * corr + v_local[i] * new_term;
        }

        m_prev = m_new;
        d_prev = d_new;
    }

    const float inv_d = 1.0f / d_prev;

    // Stream out normalized context
    for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<VEC_W> out_chunk;
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
            out_chunk[j] = o_buffer[i * VEC_W + j] * inv_d;
        }
        context_out.write(out_chunk);
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_ATTENTION_SOLVER_HPP
