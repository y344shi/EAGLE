#ifndef TMAC_RMS_NORM_STREAM_HPP
#define TMAC_RMS_NORM_STREAM_HPP

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

using tmac::hls::VEC_W;
using tmac::hls::vec_t;
using tmac::hls::hls_stream;

// Streaming RMSNorm: assumes gamma length = hidden_dim; eps fixed.
template <int HIDDEN_DIM>
void rms_norm_stream(hls_stream<vec_t<VEC_W>>& in_stream,
                     hls_stream<vec_t<VEC_W>>& out_stream,
                     const float* gamma,
                     float eps = 1e-6f) {
#pragma HLS INTERFACE axis port = in_stream
#pragma HLS INTERFACE axis port = out_stream
#pragma HLS INTERFACE s_axilite port = gamma bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static_assert(HIDDEN_DIM % VEC_W == 0, "HIDDEN_DIM must be divisible by VEC_W");

    float buf[HIDDEN_DIM];
#pragma HLS ARRAY_PARTITION variable = buf cyclic factor = VEC_W

    // Load and accumulate sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < HIDDEN_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 2
        vec_t<VEC_W> v = in_stream.read();
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
            float x = v[j];
            buf[i * VEC_W + j] = x;
            sum_sq += x * x;
        }
    }
    float scale = ::hls::sqrt(sum_sq / HIDDEN_DIM + eps);
    // Normalize and apply gamma
    for (int i = 0; i < HIDDEN_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<VEC_W> out;
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
            int idx = i * VEC_W + j;
            out[j] = buf[idx] * gamma[idx] / scale;
        }
        out_stream.write(out);
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_RMS_NORM_STREAM_HPP
