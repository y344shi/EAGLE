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

// Extract raw 4-bit weight at lane idx from pack512 (0..127). The real value is (raw - 8).
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

template <int SCALE_EXP>
inline float scale_pow2(float x) {
#pragma HLS INLINE
    return x * (1.0f / static_cast<float>(1 << SCALE_EXP));
}

inline void build_lut_pos(float a_scaled, float lut_pos[9]) {
#pragma HLS INLINE
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
}

// Build a full raw-index LUT for INT4 code space (0..15), where real_w = raw - 8.
// This maps directly from packed weight nibble to product and avoids per-lane sign/mag decode.
inline void build_lut_raw16(float a_scaled, float lut_raw16[16]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = lut_raw16 complete
    float lut_pos[9];
    build_lut_pos(a_scaled, lut_pos);

    // raw 0..7  => w -8..-1
    lut_raw16[0] = -lut_pos[8];
    lut_raw16[1] = -lut_pos[7];
    lut_raw16[2] = -lut_pos[6];
    lut_raw16[3] = -lut_pos[5];
    lut_raw16[4] = -lut_pos[4];
    lut_raw16[5] = -lut_pos[3];
    lut_raw16[6] = -lut_pos[2];
    lut_raw16[7] = -lut_pos[1];

    // raw 8..15 => w 0..7
    lut_raw16[8] = lut_pos[0];
    lut_raw16[9] = lut_pos[1];
    lut_raw16[10] = lut_pos[2];
    lut_raw16[11] = lut_pos[3];
    lut_raw16[12] = lut_pos[4];
    lut_raw16[13] = lut_pos[5];
    lut_raw16[14] = lut_pos[6];
    lut_raw16[15] = lut_pos[7];
}

// LUT-MAC: multiply 16-lane activation vector by packed INT4 weights using LUT selection.
template <int SCALE_EXP>
void lut_mac_tile(const vec_t<VEC_W>& a_vec,
                  const pack512& w_pkt,
                  vec_t<VEC_W>& acc_vec) {
#pragma HLS INLINE
    // Precompute positive multipliers for each lane once.
    for (int lane = 0; lane < VEC_W; ++lane) {
#pragma HLS UNROLL
        const float a_scaled = scale_pow2<SCALE_EXP>(a_vec[lane]);

        // Build small LUT for |w| in [0..7]; sign handled after selection.
        float lut_pos[9];
        build_lut_pos(a_scaled, lut_pos);

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
    static_assert(OUT_W <= 128, "pack512 stores at most 128 int4 weights");
    static_assert(OUT_W % VEC_W == 0, "OUT_W must align to VEC_W");

    // Precompute LUT for this scalar
    const float a_scaled = scale_pow2<SCALE_EXP>(a_scalar);
    float lut_raw16[16];
    build_lut_raw16(a_scaled, lut_raw16);

    if constexpr (ENABLE_TMAC) {
        for (int lane = 0; lane < OUT_W; ++lane) {
#pragma HLS UNROLL factor = 16
            const uint8_t w_raw = get_w4_raw(w_pkt, lane);
            acc_vec[lane] += lut_raw16[w_raw];
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
    static_assert(OUT_W <= 128, "pack512 stores at most 128 int4 weights");
    static_assert(OUT_W % VEC_W == 0, "OUT_W must align to VEC_W");

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

template <int SCALE_EXP, int INPUT_DIM, int OUT_DIM = 128, int GROUP_SIZE = 128, bool ENABLE_TMAC = false>
void dense_projection_production_scaled(hls_stream<vec_t<VEC_W>>& a_stream,
                                        hls_stream<vec_t<VEC_W>>& c_stream,
                                        const pack512* weights,
                                        const float* scales) {
#pragma HLS INTERFACE axis port = a_stream
#pragma HLS INTERFACE axis port = c_stream
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = gmem0 depth = 1024
#pragma HLS INTERFACE m_axi port = scales offset = slave bundle = gmem1 depth = 1024
#pragma HLS INTERFACE s_axilite port = weights bundle = control
#pragma HLS INTERFACE s_axilite port = scales bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static_assert(INPUT_DIM % VEC_W == 0, "INPUT_DIM must be multiple of VEC_W");
    static_assert(GROUP_SIZE % VEC_W == 0, "GROUP_SIZE must be multiple of VEC_W");
    static_assert(INPUT_DIM % GROUP_SIZE == 0, "INPUT_DIM must be multiple of GROUP_SIZE");
    static_assert(OUT_DIM % 128 == 0, "OUT_W must be multiple of tile size 128");

    constexpr int TILE = 128;
    constexpr int TILES = OUT_DIM / TILE;
    constexpr int NUM_GROUPS = INPUT_DIM / GROUP_SIZE;
    static float a_buffer[INPUT_DIM];
#pragma HLS BIND_STORAGE variable=a_buffer type=ram_2p impl=bram

    // buffers to hold weights for one tile
    static pack512 weights_tile_bram[INPUT_DIM];
#pragma HLS BIND_STORAGE variable=weights_tile_bram type=ram_2p impl=bram
    static float scales_tile_bram[NUM_GROUPS][TILE];
#pragma HLS BIND_STORAGE variable=scales_tile_bram type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=scales_tile_bram cyclic factor=VEC_W dim=2

    // Accumulators for ONE tile. Reset for each tile.
    float acc_banks[4][TILE];
#pragma HLS ARRAY_PARTITION variable=acc_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_banks cyclic factor=VEC_W dim=2

    vec_t<VEC_W> current_input_chunk;
ingest_a_loop:
    for (int k = 0; k < INPUT_DIM; k += VEC_W) {
        current_input_chunk = a_stream.read();
        for (int j = 0; j < VEC_W; ++j) {
#pragma HLS PIPELINE II=1
            a_buffer[k + j] = current_input_chunk[j];
        }
    }

    float lut_raw16[16];
#pragma HLS ARRAY_PARTITION variable=lut_raw16 complete

tile_loop:
    for (int t = 0; t < TILES; ++t) {
    init_acc_loop:
        for (int b = 0; b < 4; ++b) {
            for (int i = 0; i < TILE; ++i) {
#pragma HLS UNROLL factor=VEC_W
                acc_banks[b][i] = 0.0f;
            }
        }

    load_weights_loop:
        for (int k = 0; k < INPUT_DIM; ++k) {
#pragma HLS PIPELINE II=1
            weights_tile_bram[k] = weights[t * INPUT_DIM + k];
        }

    load_scales_loop:
        for (int g = 0; g < NUM_GROUPS; ++g) {
            for (int l = 0; l < TILE; ++l) {
#pragma HLS PIPELINE II=1
                // HBM address: scales are grouped, then tiled, then by lane
                int hbm_addr = (g * TILES + t) * TILE + l;
                scales_tile_bram[g][l] = scales[hbm_addr];
            }
        }

        // --- B: COMPUTE for the current tile using on-chip data ---
    compute_group_loop:
        for (int g = 0; g < NUM_GROUPS; ++g) {
        compute_k_in_group_loop:
            for (int kg = 0; kg < GROUP_SIZE; ++kg) {
#pragma HLS PIPELINE II=1
            const int k = g * GROUP_SIZE + kg;
            const float a_scalar = a_buffer[k]; // Read from on-chip buffer
            const int bank = k & 0x3;

            // Fetch weights and scales for this 'k' from their respective BRAMs
            const pack512 w_pkt = weights_tile_bram[k];

            // Pre-calculate LUT for TMAC if enabled
            if constexpr (ENABLE_TMAC) {
                build_lut_raw16(scale_pow2<SCALE_EXP>(a_scalar), lut_raw16);
            }

        compute_lane_loop:
            for (int lane = 0; lane < TILE; ++lane) {
#pragma HLS UNROLL factor=VEC_W
                const uint8_t w_raw = get_w4_raw(w_pkt, lane);
                const float scale_val = scales_tile_bram[g][lane];

                float prod;
                if constexpr (ENABLE_TMAC) {
                    prod = lut_raw16[w_raw] * scale_val;
                } else {
                    const int8_t w = decode_w4(w_raw);
                    prod = scale_pow2<SCALE_EXP>(a_scalar) * static_cast<float>(w) * scale_val;
                }
                acc_banks[bank][lane] += prod;
            }
        }
        }

    store_oc_loop:
        for (int oc = 0; oc < TILE / VEC_W; ++oc) {
#pragma HLS PIPELINE II=1
            vec_t<VEC_W> out_vec;
        store_j_loop:
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                const int lane = oc * VEC_W + j;
                float sum = acc_banks[0][lane] + acc_banks[1][lane] + acc_banks[2][lane] +
                            acc_banks[3][lane];
                out_vec[j] = sum;
            }
            c_stream.write(out_vec);
        }
    } // End of tile_loop
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
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = gmem0 depth = 1024
#pragma HLS INTERFACE m_axi port = scales offset = slave bundle = gmem1 depth = 1024
#pragma HLS INTERFACE s_axilite port = weights bundle = control
#pragma HLS INTERFACE s_axilite port = scales bundle = control
#pragma HLS INTERFACE s_axilite port = tile_base bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static_assert(INPUT_DIM % VEC_W == 0, "INPUT_DIM must be multiple of VEC_W");
    static_assert(GROUP_SIZE % VEC_W == 0, "GROUP_SIZE must be multiple of VEC_W");
    static_assert(OUT_W <= 128, "pack512 stores at most 128 int4 weights");
    static_assert(OUT_W % VEC_W == 0, "OUT_W must align to VEC_W");

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

        const float a_scaled = scale_pow2<SCALE_EXP>(a_scalar);
        float lut_raw16[16];
        build_lut_raw16(a_scaled, lut_raw16);

        for (int lane = 0; lane < OUT_W; ++lane) {
#pragma HLS UNROLL factor = 16
            const uint8_t w_raw = get_w4_raw(w_pkt, lane);
            float prod = lut_raw16[w_raw];
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
