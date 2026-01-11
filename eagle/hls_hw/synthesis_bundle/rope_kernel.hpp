#ifndef TMAC_ROPE_KERNEL_HPP
#define TMAC_ROPE_KERNEL_HPP

// Optimized RoPE kernel with host fallbacks.
// - CPU precomputes cos/sin for the current position and passes them in cfg arrays.
// - Streams Q/K as vec16 chunks, buffers one head locally, rotates pairs, streams out.
// - VEC_W fixed at 16 to match the rest of the pipeline.

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

using tmac::hls::VEC_W;
using tmac::hls::vec_t;
using tmac::hls::hls_stream;

template <int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM>
struct RopeConfig {
    float cos_vals[HEAD_DIM / 2];
    float sin_vals[HEAD_DIM / 2];
};

template <int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM>
void rope_apply_stream(hls_stream<vec_t<VEC_W>>& q_in,
                       hls_stream<vec_t<VEC_W>>& q_out,
                       hls_stream<vec_t<VEC_W>>& k_in,
                       hls_stream<vec_t<VEC_W>>& k_out,
                       const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& cfg) {
#pragma HLS INTERFACE axis port = q_in
#pragma HLS INTERFACE axis port = q_out
#pragma HLS INTERFACE axis port = k_in
#pragma HLS INTERFACE axis port = k_out
#pragma HLS INTERFACE s_axilite port = cfg bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    constexpr int HALF_DIM = HEAD_DIM / 2;
    float q_buf[NUM_HEADS][HEAD_DIM];
    float k_buf[NUM_KV_HEADS][HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = q_buf complete dim = 2
#pragma HLS ARRAY_PARTITION variable = k_buf complete dim = 2

    // Load Q
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
            vec_t<VEC_W> chunk = q_in.read();
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                q_buf[h][i * VEC_W + j] = chunk[j];
            }
        }
    }
    // Load K
    for (int h = 0; h < NUM_KV_HEADS; ++h) {
        for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
            vec_t<VEC_W> chunk = k_in.read();
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                k_buf[h][i * VEC_W + j] = chunk[j];
            }
        }
    }

    // Rotate Q
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int i = 0; i < HALF_DIM; ++i) {
#pragma HLS PIPELINE II = 1
            const float a = q_buf[h][i];
            const float b = q_buf[h][i + HALF_DIM];
            const float c = cfg.cos_vals[i];
            const float s = cfg.sin_vals[i];
            q_buf[h][i] = a * c - b * s;
            q_buf[h][i + HALF_DIM] = a * s + b * c;
        }
    }
    // Rotate K
    for (int h = 0; h < NUM_KV_HEADS; ++h) {
        for (int i = 0; i < HALF_DIM; ++i) {
#pragma HLS PIPELINE II = 1
            const float a = k_buf[h][i];
            const float b = k_buf[h][i + HALF_DIM];
            const float c = cfg.cos_vals[i];
            const float s = cfg.sin_vals[i];
            k_buf[h][i] = a * c - b * s;
            k_buf[h][i + HALF_DIM] = a * s + b * c;
        }
    }

    // Stream out Q
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
            vec_t<VEC_W> chunk;
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                chunk[j] = q_buf[h][i * VEC_W + j];
            }
            q_out.write(chunk);
        }
    }
    // Stream out K
    for (int h = 0; h < NUM_KV_HEADS; ++h) {
        for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
            vec_t<VEC_W> chunk;
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                chunk[j] = k_buf[h][i * VEC_W + j];
            }
            k_out.write(chunk);
        }
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_ROPE_KERNEL_HPP
