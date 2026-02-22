#ifndef TMAC_FUSED_ONLINE_ATTENTION_PWL_HPP
#define TMAC_FUSED_ONLINE_ATTENTION_PWL_HPP

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

using tmac::hls::VEC_W;
using tmac::hls::hls_stream;
using tmac::hls::vec_t;

constexpr float kLog2e = 1.4426950408889634f;
constexpr int kExp2Clamp = 15;

// Piecewise linear approximation of 2^f over f in [0, 1).
inline float exp2_frac_pwl_4seg(float frac) {
#pragma HLS INLINE
    if (frac < 0.25f) return 0.75785828f * frac + 1.0f;
    if (frac < 0.50f) return 0.90002775f * frac + 0.96445763f;
    if (frac < 0.75f) return 1.07031631f * frac + 0.87931335f;
    return 1.27282763f * frac + 0.72742941f;
}

// Approximate exp(x) by converting to base-2 and using a small PWL for the fractional part.
// This mirrors the paper's hardware-oriented softmax path and avoids full transcendental units.
inline float exp_softmax_pwl(float x) {
#pragma HLS INLINE
    float x2 = x * kLog2e; // e^x = 2^(x*log2(e))
    if (x2 >= 0.0f) x2 = 0.0f;
    if (x2 <= -static_cast<float>(kExp2Clamp)) return 0.0f;

    int int_part = static_cast<int>(x2);
    if (static_cast<float>(int_part) > x2) --int_part; // floor for negative inputs
    if (int_part < -kExp2Clamp) return 0.0f;

    const float frac = x2 - static_cast<float>(int_part);
    const float frac_term = exp2_frac_pwl_4seg(frac);

    // 2^int_part for int_part in [-15, 0]
    static const float kPow2Neg[kExp2Clamp + 1] = {
        1.0f,
        0.5f,
        0.25f,
        0.125f,
        0.0625f,
        0.03125f,
        0.015625f,
        0.0078125f,
        0.00390625f,
        0.001953125f,
        0.0009765625f,
        0.00048828125f,
        0.000244140625f,
        0.0001220703125f,
        0.00006103515625f,
        0.000030517578125f,
    };

    return frac_term * kPow2Neg[-int_part];
}

// Fused single-head decode attention.
// - Q is loaded once and held stationary.
// - K/V history is streamed token-by-token.
// - Score, online softmax state, and context update are fused in one pass.
template <int HEAD_DIM>
void fused_online_attention_pwl(hls_stream<vec_t<VEC_W>>& q_stream,
                              hls_stream<vec_t<VEC_W>>& k_hist,
                              hls_stream<vec_t<VEC_W>>& v_hist,
                              hls_stream<vec_t<VEC_W>>& context_out,
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

    float q_buffer[HEAD_DIM];
    float ctx_acc[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = q_buffer cyclic factor = VEC_W
#pragma HLS ARRAY_PARTITION variable = ctx_acc cyclic factor = VEC_W

    const int vec_chunks = HEAD_DIM / VEC_W;

load_q:
    for (int i = 0; i < vec_chunks; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<VEC_W> chunk = q_stream.read();
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
            q_buffer[i * VEC_W + j] = chunk[j];
            ctx_acc[i * VEC_W + j] = 0.0f;
        }
    }

    float m_prev = -1e30f;
    float d_prev = 0.0f;
    const float scale = 1.0f / ::hls::sqrt(static_cast<float>(HEAD_DIM));

token_loop:
    for (int t = 0; t < total_len; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 avg=1024 max=4096
        float partial_score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float v_local[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = v_local cyclic factor = VEC_W

        if (t < seq_len) {
dot_and_load:
            for (int i = 0; i < vec_chunks; ++i) {
#pragma HLS PIPELINE II = 1
                vec_t<VEC_W> k_chunk = k_hist.read();
                vec_t<VEC_W> v_chunk = v_hist.read();
                for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                    const int idx = i * VEC_W + j;
                    partial_score[j & 0x3] += q_buffer[idx] * k_chunk[j];
                    v_local[idx] = v_chunk[j];
                }
            }
        } else {
pad_token:
            for (int i = 0; i < HEAD_DIM; ++i) {
#pragma HLS PIPELINE II = 1
                v_local[i] = 0.0f;
            }
        }

        float score = (partial_score[0] + partial_score[1]) +
                      (partial_score[2] + partial_score[3]);
        if (t >= seq_len) score = -1e9f;
        score *= scale;

        const float m_new = (score > m_prev) ? score : m_prev;
        const float corr = exp_softmax_pwl(m_prev - m_new);
        const float new_term = exp_softmax_pwl(score - m_new);
        const float d_new = d_prev * corr + new_term;

update_ctx:
        for (int i = 0; i < vec_chunks; ++i) {
#pragma HLS PIPELINE II = 1
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                const int idx = i * VEC_W + j;
                ctx_acc[idx] = ctx_acc[idx] * corr + v_local[idx] * new_term;
            }
        }

        m_prev = m_new;
        d_prev = d_new;
    }

    const float inv_d = (d_prev > 0.0f) ? (1.0f / d_prev) : 0.0f;

write_ctx:
    for (int i = 0; i < vec_chunks; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<VEC_W> out_chunk{};
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
            out_chunk[j] = ctx_acc[i * VEC_W + j] * inv_d;
        }
        context_out.write(out_chunk);
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_FUSED_ONLINE_ATTENTION_PWL_HPP
