#ifndef TMAC_DEEP_PIPELINE_LUTMAC_HPP
#define TMAC_DEEP_PIPELINE_LUTMAC_HPP

#include <array>
#include <cstdint>
#include <cstring>

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

using tmac::hls::pack512;
using tmac::hls::VEC_W;
using tmac::hls::vec_t;
using tmac::hls::hls_stream;

// Extract raw 4-bit weight at lane idx from pack512 (0..15). The real value is (raw - 8).
inline uint8_t get_w4_raw(const pack512& p, int idx) {
#ifdef __SYNTHESIS__
    const int bit = idx * 4;
    ap_uint<4> raw = p.range(bit + 3, bit);
    return static_cast<uint8_t>(raw);
#else
    const int byte_idx = idx >> 1;
    const bool high = idx & 1;
    uint8_t b = p.bytes[byte_idx];
    return static_cast<uint8_t>(high ? (b >> 4) & 0xF : b & 0xF);
#endif
}

inline int8_t decode_w4(uint8_t raw) { return static_cast<int8_t>(raw) - 8; }

// LUT-MAC: multiply 16-lane activation vector by packed INT4 weights using LUT selection.
template <int SCALE_EXP>
void lut_mac_tile(const vec_t<VEC_W>& a_vec,
                  const pack512& w_pkt,
                  vec_t<VEC_W>& acc_vec) {
#pragma HLS INLINE
    // Precompute positive multipliers for each lane once.
    for (int lane = 0; lane < VEC_W; ++lane) {
#pragma HLS UNROLL
        const float a_raw = a_vec[lane];
        const float a_scaled = a_raw * (1.0f / static_cast<float>(1 << SCALE_EXP));

        // Build small LUT for |w| in [0..7]; sign handled after selection.
        float lut_pos[9];
#pragma HLS ARRAY_PARTITION variable = lut_pos complete
        lut_pos[0] = 0.0f;
        lut_pos[1] = a_scaled;
        lut_pos[2] = a_scaled * 2.0f;     // shift left 1
        lut_pos[4] = a_scaled * 4.0f;     // shift left 2
        lut_pos[3] = lut_pos[1] + lut_pos[2];
        lut_pos[5] = lut_pos[1] + lut_pos[4];
        lut_pos[6] = lut_pos[2] + lut_pos[4];
        lut_pos[7] = lut_pos[3] + lut_pos[4];

        lut_pos[8] = lut_pos[4] + lut_pos[4];

        const uint8_t w_raw = get_w4_raw(w_pkt, lane);
        const int8_t w = decode_w4(w_raw);
        const uint8_t mag = static_cast<uint8_t>(w < 0 ? -w : w);
        float prod = lut_pos[mag];
        if (w < 0) prod = -prod;
        acc_vec[lane] += prod;
    }
}

// Broadcast variant: one scalar activation times OUT_W weights (one packet) accumulated into OUT_W outputs.
template <int SCALE_EXP, int OUT_W, bool ENABLE_TMAC = true>
void lut_mac_broadcast(float a_scalar,
                       const pack512& w_pkt,
                       vec_t<OUT_W>& acc_vec) {
#pragma HLS INLINE
    // Precompute LUT for this scalar
    const float a_scaled = a_scalar * (1.0f / static_cast<float>(1 << SCALE_EXP));
    float lut_pos[9];
#pragma HLS ARRAY_PARTITION variable = lut_pos complete
    lut_pos[0] = 0.0f;
    lut_pos[1] = a_scaled;
    lut_pos[2] = a_scaled * 2.0f;
    lut_pos[4] = a_scaled * 4.0f;
    lut_pos[3] = lut_pos[1] + lut_pos[2];
    lut_pos[5] = lut_pos[1] + lut_pos[4];
    lut_pos[6] = lut_pos[2] + lut_pos[4];
    lut_pos[7] = lut_pos[3] + lut_pos[4];
    lut_pos[8] = lut_pos[4] + lut_pos[4];

    if constexpr (ENABLE_TMAC) {
        for (int lane = 0; lane < OUT_W; ++lane) {
#pragma HLS UNROLL factor = 16
            const uint8_t w_raw = get_w4_raw(w_pkt, lane);
            const int8_t w = decode_w4(w_raw);
            const uint8_t mag = static_cast<uint8_t>(w < 0 ? -w : w);
            float prod = lut_pos[mag];
            if (w < 0) prod = -prod;
            acc_vec[lane] += prod;
        }
    } else {
        // DSP path: direct multiply
        for (int lane = 0; lane < OUT_W; ++lane) {
#pragma HLS UNROLL factor = 16
            const uint8_t w_raw = get_w4_raw(w_pkt, lane);
            const int8_t w = decode_w4(w_raw);
            acc_vec[lane] += a_scaled * static_cast<float>(w);
        }
    }
}

// Streaming dense projection (production): broadcast one scalar per cycle, hide FP add latency with interleaved accumulators.
// Weight layout: weights[k] holds OUT_W INT4 weights for input scalar k (broadcast).
template <int SCALE_EXP, int INPUT_DIM, int OUT_W = 128, bool ENABLE_TMAC = true>
void dense_projection_production(hls_stream<vec_t<VEC_W>>& a_stream,
                                 hls_stream<vec_t<VEC_W>>& c_stream,
                                 const pack512* weights) {
#pragma HLS INTERFACE axis port = a_stream
#pragma HLS INTERFACE axis port = c_stream
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = gmem depth = 1024
#pragma HLS INTERFACE s_axilite port = weights bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static_assert(INPUT_DIM % VEC_W == 0, "INPUT_DIM must be multiple of VEC_W");

    vec_t<OUT_W> acc_banks[4];
#pragma HLS ARRAY_PARTITION variable = acc_banks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc_banks cyclic factor = 16 dim = 2

    // init accumulators
    for (int b = 0; b < 4; ++b) {
        for (int i = 0; i < OUT_W; ++i) {
#pragma HLS UNROLL
            acc_banks[b][i] = 0.0f;
        }
    }

    vec_t<VEC_W> current_input_chunk{};
    // flattened loop over all input scalars
    for (int k = 0; k < INPUT_DIM; ++k) {
#pragma HLS PIPELINE II = 1
        if ((k & (VEC_W - 1)) == 0) {
            current_input_chunk = a_stream.read();
        }
        const float a_scalar = current_input_chunk[k & (VEC_W - 1)];
        const pack512 w_pkt = weights[k];
        const int bank = k & 0x3; // k % 4
        lut_mac_broadcast<SCALE_EXP, OUT_W, ENABLE_TMAC>(a_scalar, w_pkt, acc_banks[bank]);
    }

    // reduce banks and stream out
    const int out_chunks = OUT_W / VEC_W;
    for (int oc = 0; oc < out_chunks; ++oc) {
        vec_t<VEC_W> out_vec;
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
            const int lane = oc * VEC_W + j;
            float sum = acc_banks[0][lane] + acc_banks[1][lane] + acc_banks[2][lane] +
                        acc_banks[3][lane];
            out_vec[j] = sum;
        }
        c_stream.write(out_vec);
    }
}

// Variant with per-group, per-lane floating scales.
// Layout expectation (from pack_all_4bit.py):
//   weights: tile-major, then input, then lane -> weights[t * INPUT_DIM + k] holds 128 lanes.
//   scales : packed as scales[g][tile][lane] where g = input_idx / GROUP_SIZE.
// ENABLE_TMAC toggles LUT-MAC (true) vs direct DSP multiply (false) for ease of A/B testing.
template <int SCALE_EXP, int INPUT_DIM, int OUT_W = 128, int GROUP_SIZE = 128, bool ENABLE_TMAC = false>
void dense_projection_production_scaled(hls_stream<vec_t<VEC_W>>& a_stream,
                                        hls_stream<vec_t<VEC_W>>& c_stream,
                                        const pack512* weights,
                                        const float* scales) {
#pragma HLS INTERFACE axis port = a_stream
#pragma HLS INTERFACE axis port = c_stream
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = gmem depth = 1024
#pragma HLS INTERFACE m_axi port = scales offset = slave bundle = gmem depth = 1024
#pragma HLS INTERFACE s_axilite port = weights bundle = control
#pragma HLS INTERFACE s_axilite port = scales bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static_assert(INPUT_DIM % VEC_W == 0, "INPUT_DIM must be multiple of VEC_W");
    static_assert(GROUP_SIZE % VEC_W == 0, "GROUP_SIZE must be multiple of VEC_W");
    static_assert(OUT_W % 128 == 0, "OUT_W must be multiple of tile size 128");

    constexpr int TILE = 128;
    constexpr int TILES = OUT_W / TILE;

    vec_t<TILE> acc_banks[TILES][4];
#pragma HLS ARRAY_PARTITION variable = acc_banks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc_banks complete dim = 2
#pragma HLS ARRAY_PARTITION variable = acc_banks cyclic factor = 16 dim = 3

    for (int t = 0; t < TILES; ++t) {
        for (int b = 0; b < 4; ++b) {
            for (int i = 0; i < TILE; ++i) {
#pragma HLS UNROLL
                acc_banks[t][b][i] = 0.0f;
            }
        }
    }

    vec_t<VEC_W> current_input_chunk{};
    for (int k = 0; k < INPUT_DIM; ++k) {
#pragma HLS PIPELINE II = 1
        if ((k & (VEC_W - 1)) == 0) {
            current_input_chunk = a_stream.read();
        }
        const float a_scalar = current_input_chunk[k & (VEC_W - 1)];
        const int group = k / GROUP_SIZE;

        const float a_scaled_pow2 = (1.0f / static_cast<float>(1 << SCALE_EXP));
        const float a_scaled = a_scalar * a_scaled_pow2;
        float lut_pos[9];
#pragma HLS ARRAY_PARTITION variable = lut_pos complete
        lut_pos[0] = 0.0f;
        lut_pos[1] = a_scaled;
        lut_pos[2] = a_scaled * 2.0f;
        lut_pos[4] = a_scaled * 4.0f;
        lut_pos[3] = lut_pos[1] + lut_pos[2];
        lut_pos[5] = lut_pos[1] + lut_pos[4];
        lut_pos[6] = lut_pos[2] + lut_pos[4];
        lut_pos[7] = lut_pos[3] + lut_pos[4];
        lut_pos[8] = lut_pos[4] + lut_pos[4];

        for (int t = 0; t < TILES; ++t) {
#pragma HLS UNROLL
            const pack512 w_pkt = weights[t * INPUT_DIM + k];
            const float* scale_tile = scales + (group * TILES + t) * TILE;
            const int bank = k & 0x3;

            if (ENABLE_TMAC) {
                for (int lane = 0; lane < TILE; ++lane) {
#pragma HLS UNROLL factor = 16
                    const uint8_t w_raw = get_w4_raw(w_pkt, lane);
                    const int8_t w = decode_w4(w_raw);
                    const uint8_t mag = static_cast<uint8_t>(w < 0 ? -w : w);
                    float prod = lut_pos[mag];
                    if (w < 0) prod = -prod;
                    prod *= scale_tile[lane];
                    acc_banks[t][bank][lane] += prod;
                }
            } else {
                for (int lane = 0; lane < TILE; ++lane) {
#pragma HLS UNROLL factor = 16
                    const uint8_t w_raw = get_w4_raw(w_pkt, lane);
                    const int8_t w = decode_w4(w_raw);
                    const float prod = a_scalar * a_scaled_pow2 * static_cast<float>(w) * scale_tile[lane];
                    acc_banks[t][bank][lane] += prod;
                }
            }
        }
    }

    // Write out accumulated tiles
    for (int t = 0; t < TILES; ++t) {
        for (int oc = 0; oc < TILE / VEC_W; ++oc) {
            vec_t<VEC_W> out_vec;
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                const int lane = oc * VEC_W + j;
                float sum = acc_banks[t][0][lane] + acc_banks[t][1][lane] + acc_banks[t][2][lane] +
                            acc_banks[t][3][lane];
                out_vec[j] = sum;
            }
            c_stream.write(out_vec);
        }
    }
}

// Variant that reads scales in the original CPU layout: scales[group * OUT_DIM_TOTAL + out_idx].
template <int SCALE_EXP, int INPUT_DIM, int OUT_W = 128, int GROUP_SIZE = 128, int OUT_DIM_TOTAL = 4096>
void dense_projection_production_scaled_raw(hls_stream<vec_t<VEC_W>>& a_stream,
                                            hls_stream<vec_t<VEC_W>>& c_stream,
                                            const pack512* weights,
                                            const float* scales,
                                            int tile_base) {
#pragma HLS INTERFACE axis port = a_stream
#pragma HLS INTERFACE axis port = c_stream
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = gmem depth = 1024
#pragma HLS INTERFACE m_axi port = scales offset = slave bundle = gmem depth = 1024
#pragma HLS INTERFACE s_axilite port = weights bundle = control
#pragma HLS INTERFACE s_axilite port = scales bundle = control
#pragma HLS INTERFACE s_axilite port = tile_base bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static_assert(INPUT_DIM % VEC_W == 0, "INPUT_DIM must be multiple of VEC_W");
    static_assert(GROUP_SIZE % VEC_W == 0, "GROUP_SIZE must be multiple of VEC_W");

    vec_t<OUT_W> acc_banks[4];
#pragma HLS ARRAY_PARTITION variable = acc_banks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc_banks cyclic factor = 16 dim = 2

    for (int b = 0; b < 4; ++b) {
        for (int i = 0; i < OUT_W; ++i) {
#pragma HLS UNROLL
            acc_banks[b][i] = 0.0f;
        }
    }

    vec_t<VEC_W> current_input_chunk{};
    for (int k = 0; k < INPUT_DIM; ++k) {
#pragma HLS PIPELINE II = 1
        if ((k & (VEC_W - 1)) == 0) {
            current_input_chunk = a_stream.read();
        }
        const float a_scalar = current_input_chunk[k & (VEC_W - 1)];
        const pack512 w_pkt = weights[k];
        const int bank = k & 0x3;
        const int group = k / GROUP_SIZE;

        const float a_scaled = a_scalar * (1.0f / static_cast<float>(1 << SCALE_EXP));
        float lut_pos[9];
#pragma HLS ARRAY_PARTITION variable = lut_pos complete
        lut_pos[0] = 0.0f;
        lut_pos[1] = a_scaled;
        lut_pos[2] = a_scaled * 2.0f;
        lut_pos[4] = a_scaled * 4.0f;
        lut_pos[3] = lut_pos[1] + lut_pos[2];
        lut_pos[5] = lut_pos[1] + lut_pos[4];
        lut_pos[6] = lut_pos[2] + lut_pos[4];
        lut_pos[7] = lut_pos[3] + lut_pos[4];
        lut_pos[8] = lut_pos[4] + lut_pos[4];

        for (int lane = 0; lane < OUT_W; ++lane) {
#pragma HLS UNROLL factor = 16
            const uint8_t w_raw = get_w4_raw(w_pkt, lane);
            const int8_t w = decode_w4(w_raw);
            const uint8_t mag = static_cast<uint8_t>(w < 0 ? -w : w);
            float prod = lut_pos[mag];
            if (w < 0) prod = -prod;
            const int out_idx = tile_base + lane;
            prod *= scales[group * OUT_DIM_TOTAL + out_idx];
            acc_banks[bank][lane] += prod;
        }
    }

    const int out_chunks = OUT_W / VEC_W;
    for (int oc = 0; oc < out_chunks; ++oc) {
        vec_t<VEC_W> out_vec;
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
            const int lane = oc * VEC_W + j;
            float sum = acc_banks[0][lane] + acc_banks[1][lane] + acc_banks[2][lane] +
                        acc_banks[3][lane];
            out_vec[j] = sum;
        }
        c_stream.write(out_vec);
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_DEEP_PIPELINE_LUTMAC_HPP
