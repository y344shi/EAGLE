#ifndef TMAC_KV_CACHE_MANAGER_HPP
#define TMAC_KV_CACHE_MANAGER_HPP

// Hybrid hot/cold KV cache manager.
// - Hot tier: URAM for fast reuse of recent tokens (e.g., first 16K tokens).
// - Cold tier: HBM fallback when sequence exceeds hot capacity.
// - Interfaces: AXIS for K/V in and history out; AXI-MM for HBM buffers; AXI-Lite for control.

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

using tmac::hls::VEC_W;
using tmac::hls::vec_t;
using tmac::hls::hls_stream;
constexpr int MAX_URAM_TOKENS = 16384; // Hot tier capacity (tokens)

// HEAD_DIM * NUM_KV_HEADS must be divisible by VEC_W.
template <int HEAD_DIM, int NUM_KV_HEADS>
void kv_cache_manager(hls_stream<vec_t<VEC_W>>& k_in_stream,   // new token K (post-RoPE)
                      hls_stream<vec_t<VEC_W>>& v_in_stream,   // new token V (post-RoPE)
                      hls_stream<vec_t<VEC_W>>& k_hist_stream, // history out to attention
                      hls_stream<vec_t<VEC_W>>& v_hist_stream, // history out to attention
                      vec_t<VEC_W>* hbm_k_buffer,               // cold tier K (AXI-MM)
                      vec_t<VEC_W>* hbm_v_buffer,               // cold tier V (AXI-MM)
                      int current_length,                       // valid tokens in cache
                      bool write_enable,                        // append new token?
                      bool read_enable) {                       // stream history?
#pragma HLS INTERFACE axis port = k_in_stream
#pragma HLS INTERFACE axis port = v_in_stream
#pragma HLS INTERFACE axis port = k_hist_stream
#pragma HLS INTERFACE axis port = v_hist_stream
#pragma HLS INTERFACE m_axi port = hbm_k_buffer offset = slave bundle = gmem0 depth = 1024
#pragma HLS INTERFACE m_axi port = hbm_v_buffer offset = slave bundle = gmem1 depth = 1024
#pragma HLS INTERFACE s_axilite port = current_length bundle = control
#pragma HLS INTERFACE s_axilite port = write_enable bundle = control
#pragma HLS INTERFACE s_axilite port = read_enable bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    constexpr int VECS_PER_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
    static_assert((NUM_KV_HEADS * HEAD_DIM) % VEC_W == 0, "KV width must align to VEC_W");

    // Hot cache in URAM
    static vec_t<VEC_W> k_uram[MAX_URAM_TOKENS][VECS_PER_TOKEN];
    static vec_t<VEC_W> v_uram[MAX_URAM_TOKENS][VECS_PER_TOKEN];
#pragma HLS BIND_STORAGE variable = k_uram type = ram_2p impl = uram
#pragma HLS BIND_STORAGE variable = v_uram type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = k_uram cyclic factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = v_uram cyclic factor = 2 dim = 2

    // Write new token
    if (write_enable) {
        const int idx = current_length;
        for (int i = 0; i < VECS_PER_TOKEN; ++i) {
#pragma HLS PIPELINE II = 1
            vec_t<VEC_W> k_val = k_in_stream.read();
            vec_t<VEC_W> v_val = v_in_stream.read();
            // Always mirror to HBM for debug/inspection even when hot lives in URAM.
            const int off = idx * VECS_PER_TOKEN + i;
            hbm_k_buffer[off] = k_val;
            hbm_v_buffer[off] = v_val;

            if (idx < MAX_URAM_TOKENS) {
                k_uram[idx][i] = k_val;
                v_uram[idx][i] = v_val;
            }
        }
    }

    // Read history (loop fission: hot then cold). Include the just-written token when write_enable is true.
    if (read_enable) {
        const int total_len = current_length + (write_enable ? 1 : 0);
        const int hot_count = (total_len < MAX_URAM_TOKENS) ? total_len : MAX_URAM_TOKENS;
        const int cold_count = total_len - hot_count;

        // Hot tier: URAM, deterministic latency, II=1
    read_hot:
        for (int t = 0; t < hot_count; ++t) {
#pragma HLS LOOP_TRIPCOUNT max = MAX_URAM_TOKENS
            for (int i = 0; i < VECS_PER_TOKEN; ++i) {
#pragma HLS PIPELINE II = 1
                vec_t<VEC_W> k_val = k_uram[t][i];
                vec_t<VEC_W> v_val = v_uram[t][i];
                k_hist_stream.write(k_val);
                v_hist_stream.write(v_val);
            }
        }

        // Cold tier: HBM burst
        if (cold_count > 0) {
        read_cold:
            for (int t = MAX_URAM_TOKENS; t < total_len; ++t) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 1024
                const int base_off = t * VECS_PER_TOKEN;
                for (int i = 0; i < VECS_PER_TOKEN; ++i) {
#pragma HLS PIPELINE II = 1
                    const int off = base_off + i;
                    vec_t<VEC_W> k_val = hbm_k_buffer[off];
                    vec_t<VEC_W> v_val = hbm_v_buffer[off];
                    k_hist_stream.write(k_val);
                    v_hist_stream.write(v_val);
                }
            }
        }
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_KV_CACHE_MANAGER_HPP
