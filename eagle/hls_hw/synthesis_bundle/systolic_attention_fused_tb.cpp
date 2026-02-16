#include "systolic_attention_fused.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using tmac::hls::VEC_W;
using tmac::hls::exp_softmax_pwl;
using tmac::hls::hls_stream;
using tmac::hls::systolic_attention_fused;
using tmac::hls::vec_t;

template <int HEAD_DIM>
std::vector<float> attention_reference_exact(const std::vector<float>& q,
                                             const std::vector<float>& k_hist,
                                             const std::vector<float>& v_hist,
                                             int seq_len,
                                             int padded_len) {
    const int total_len = (padded_len > 0) ? padded_len : seq_len;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    std::vector<float> scores(total_len, -1e9f);
    for (int t = 0; t < seq_len; ++t) {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += q[d] * k_hist[t * HEAD_DIM + d];
        }
        scores[t] = dot * scale;
    }

    float max_score = scores[0];
    for (int t = 1; t < total_len; ++t) {
        if (scores[t] > max_score) max_score = scores[t];
    }

    std::vector<float> numer(total_len, 0.0f);
    float denom = 0.0f;
    for (int t = 0; t < total_len; ++t) {
        numer[t] = std::exp(scores[t] - max_score);
        denom += numer[t];
    }

    std::vector<float> out(HEAD_DIM, 0.0f);
    if (denom == 0.0f) return out;
    for (int t = 0; t < seq_len; ++t) {
        const float w = numer[t] / denom;
        for (int d = 0; d < HEAD_DIM; ++d) {
            out[d] += w * v_hist[t * HEAD_DIM + d];
        }
    }
    return out;
}

template <int HEAD_DIM>
std::vector<float> attention_reference_online_pwl(const std::vector<float>& q,
                                                  const std::vector<float>& k_hist,
                                                  const std::vector<float>& v_hist,
                                                  int seq_len,
                                                  int padded_len) {
    const int total_len = (padded_len > 0) ? padded_len : seq_len;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    float m_prev = -1e30f;
    float d_prev = 0.0f;
    std::vector<float> out(HEAD_DIM, 0.0f);

    for (int t = 0; t < total_len; ++t) {
        float score = -1e9f;
        if (t < seq_len) {
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += q[d] * k_hist[t * HEAD_DIM + d];
            }
            score = dot * scale;
        }

        const float m_new = (score > m_prev) ? score : m_prev;
        const float corr = exp_softmax_pwl(m_prev - m_new);
        const float new_term = exp_softmax_pwl(score - m_new);
        const float d_new = d_prev * corr + new_term;

        if (t < seq_len) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                out[d] = out[d] * corr + v_hist[t * HEAD_DIM + d] * new_term;
            }
        } else {
            for (int d = 0; d < HEAD_DIM; ++d) {
                out[d] = out[d] * corr;
            }
        }

        m_prev = m_new;
        d_prev = d_new;
    }

    const float inv_d = (d_prev > 0.0f) ? (1.0f / d_prev) : 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) out[d] *= inv_d;
    return out;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

template <int HEAD_DIM>
std::vector<float> run_kernel(const std::vector<float>& q,
                              const std::vector<float>& k_hist,
                              const std::vector<float>& v_hist,
                              int seq_len,
                              int padded_len) {
    hls_stream<vec_t<VEC_W>> q_stream("q_stream");
    hls_stream<vec_t<VEC_W>> k_stream("k_stream");
    hls_stream<vec_t<VEC_W>> v_stream("v_stream");
    hls_stream<vec_t<VEC_W>> out_stream("out_stream");

    for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
        vec_t<VEC_W> chunk{};
        for (int j = 0; j < VEC_W; ++j) chunk[j] = q[i * VEC_W + j];
        q_stream.write(chunk);
    }

    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
            vec_t<VEC_W> k_chunk{};
            vec_t<VEC_W> v_chunk{};
            for (int j = 0; j < VEC_W; ++j) {
                const int idx = t * HEAD_DIM + i * VEC_W + j;
                k_chunk[j] = k_hist[idx];
                v_chunk[j] = v_hist[idx];
            }
            k_stream.write(k_chunk);
            v_stream.write(v_chunk);
        }
    }

    systolic_attention_fused<HEAD_DIM>(
        q_stream, k_stream, v_stream, out_stream, seq_len, padded_len);

    std::vector<float> out(HEAD_DIM, 0.0f);
    for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
        vec_t<VEC_W> chunk = out_stream.read();
        for (int j = 0; j < VEC_W; ++j) out[i * VEC_W + j] = chunk[j];
    }
    return out;
}

int main() {
    constexpr int HEAD_DIM = 128;
    constexpr int SEQ_LEN = 23;
    constexpr int PADDED_LEN = 24;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(HEAD_DIM);
    std::vector<float> k_hist(SEQ_LEN * HEAD_DIM);
    std::vector<float> v_hist(SEQ_LEN * HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; ++i) q[i] = dist(rng);
    for (int i = 0; i < SEQ_LEN * HEAD_DIM; ++i) {
        k_hist[i] = dist(rng);
        v_hist[i] = dist(rng);
    }

    const std::vector<float> hw = run_kernel<HEAD_DIM>(q, k_hist, v_hist, SEQ_LEN, PADDED_LEN);
    const std::vector<float> ref_online =
        attention_reference_online_pwl<HEAD_DIM>(q, k_hist, v_hist, SEQ_LEN, PADDED_LEN);
    const std::vector<float> ref_exact =
        attention_reference_exact<HEAD_DIM>(q, k_hist, v_hist, SEQ_LEN, PADDED_LEN);

    const float err_hw_online = max_abs_diff(hw, ref_online);
    const float err_hw_exact = max_abs_diff(hw, ref_exact);

    std::printf("max|HW - online(PWL)| = %.8f\n", err_hw_online);
    std::printf("max|HW - exact(exp) | = %.8f\n", err_hw_exact);

    const bool pass_impl = err_hw_online < 1e-5f;
    const bool pass_approx = err_hw_exact < 7e-2f;

    if (!pass_impl) {
        std::printf("[FAIL] Kernel mismatch against equivalent online-PWL reference.\n");
        return 1;
    }
    if (!pass_approx) {
        std::printf("[FAIL] Approximation error exceeds threshold vs exact softmax.\n");
        return 1;
    }

    std::printf("[PASS] SystolicAttention fused kernel C-sim checks passed.\n");
    return 0;
}

