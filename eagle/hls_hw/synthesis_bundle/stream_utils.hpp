#ifndef TMAC_STREAM_UTILS_HPP
#define TMAC_STREAM_UTILS_HPP

// Small helper kernels for stream plumbing: duplication, addition, and SwiGLU-style SiLU*up.

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

using tmac::hls::VEC_W;
using tmac::hls::vec_t;
using tmac::hls::hls_stream;

// Scale stream by scalar.
template <int W>
void stream_scale(hls_stream<vec_t<W>>& in,
                  hls_stream<vec_t<W>>& out,
                  float scale,
                  int elements) {
#pragma HLS INLINE off
    for (int i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<W> v = in.read();
        vec_t<W> r;
        for (int j = 0; j < W; ++j) {
#pragma HLS UNROLL
            r[j] = v[j] * scale;
        }
        out.write(r);
    }
}

// Duplicate a fixed number of elements to two outputs.
template <int W>
void stream_dup(hls_stream<vec_t<W>>& in,
                hls_stream<vec_t<W>>& out0,
                hls_stream<vec_t<W>>& out1,
                int elements) {
#pragma HLS INLINE off
    for (int i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<W> v = in.read();
        out0.write(v);
        out1.write(v);
    }
}

// Duplicate to three outputs.
template <int W>
void stream_trip(hls_stream<vec_t<W>>& in,
                 hls_stream<vec_t<W>>& out0,
                 hls_stream<vec_t<W>>& out1,
                 hls_stream<vec_t<W>>& out2,
                 int elements) {
#pragma HLS INLINE off
    for (int i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<W> v = in.read();
        out0.write(v);
        out1.write(v);
        out2.write(v);
    }
}

// Elementwise add of two streams.
template <int W>
void stream_add(hls_stream<vec_t<W>>& in0,
                hls_stream<vec_t<W>>& in1,
                hls_stream<vec_t<W>>& out,
                int elements) {
#pragma HLS INLINE off
    for (int i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<W> a = in0.read();
        vec_t<W> b = in1.read();
        vec_t<W> r;
        for (int j = 0; j < W; ++j) {
#pragma HLS UNROLL
            r[j] = a[j] + b[j];
        }
        out.write(r);
    }
}

// SiLU(gate) * up stream (SwiGLU style).
template <int W>
void silu_mul_stream(hls_stream<vec_t<W>>& gate,
                     hls_stream<vec_t<W>>& up,
                     hls_stream<vec_t<W>>& out,
                     int elements) {
#pragma HLS INLINE off
    for (int i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II = 1
        vec_t<W> g = gate.read();
        vec_t<W> u = up.read();
        vec_t<W> r;
        for (int j = 0; j < W; ++j) {
#pragma HLS UNROLL
            float sig = 1.0f / (1.0f + ::hls::exp(-g[j]));
            r[j] = (g[j] * sig) * u[j];
        }
        out.write(r);
    }
}

// Simple pass-through (copy) for a fixed number of elements.
template <int W>
void stream_passthrough(hls_stream<vec_t<W>>& in,
                        hls_stream<vec_t<W>>& out,
                        int elements) {
#pragma HLS INLINE off
    for (int i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II = 1
        out.write(in.read());
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_STREAM_UTILS_HPP
