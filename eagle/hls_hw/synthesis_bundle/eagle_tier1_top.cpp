#include "eagle_tier1_top.hpp"

// Include implementation headers here
#include "attention_solver.hpp"
#include "deep_pipeline_lutmac.hpp"
#include "kv_cache_manager.hpp"
#include "rms_norm_stream.hpp"

namespace tmac {
namespace hls {

// Internal constants
constexpr float RESIDUAL_SCALE = 1.4f / 5.7445626465380286f; // sqrt(33)
constexpr int MAX_CTX = 2048; 

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// FUNCTION IMPLEMENTATION
void eagle_tier1_top(
    hls_stream<vec_t<VEC_W>>& in_stream,
    hls_stream<vec_t<VEC_W>>& out_stream,
    const pack512* w_q,     const float* s_q,
    const pack512* w_k,     const float* s_k,
    const pack512* w_v,     const float* s_v,
    const pack512* w_o,     const float* s_o,
    const pack512* w_gate,  const float* gate_scales,
    const pack512* w_up,    const float* up_scales,
    const pack512* w_down,  const float* down_scales,
    const float* norm1_gamma,
    const float* norm2_gamma,
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& rope_cfg,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    int seq_len,
    int current_length
) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE m_axi port=w_q offset=slave bundle=gmem0 depth=4096
#pragma HLS INTERFACE m_axi port=s_q offset=slave bundle=gmem0 depth=4096
#pragma HLS INTERFACE m_axi port=w_k offset=slave bundle=gmem1 depth=1024
#pragma HLS INTERFACE m_axi port=s_k offset=slave bundle=gmem1 depth=1024
#pragma HLS INTERFACE m_axi port=w_v offset=slave bundle=gmem2 depth=1024
#pragma HLS INTERFACE m_axi port=s_v offset=slave bundle=gmem2 depth=1024
#pragma HLS INTERFACE m_axi port=w_o offset=slave bundle=gmem3 depth=4096
#pragma HLS INTERFACE m_axi port=s_o offset=slave bundle=gmem3 depth=4096
#pragma HLS INTERFACE m_axi port=w_gate offset=slave bundle=gmem4 depth=4096
#pragma HLS INTERFACE m_axi port=gate_scales offset=slave bundle=gmem4 depth=4096
#pragma HLS INTERFACE m_axi port=w_up offset=slave bundle=gmem5 depth=4096
#pragma HLS INTERFACE m_axi port=up_scales offset=slave bundle=gmem5 depth=4096
#pragma HLS INTERFACE m_axi port=w_down offset=slave bundle=gmem6 depth=4096
#pragma HLS INTERFACE m_axi port=down_scales offset=slave bundle=gmem6 depth=4096
#pragma HLS INTERFACE m_axi port=norm1_gamma offset=slave bundle=gmem7 depth=4096
#pragma HLS INTERFACE m_axi port=norm2_gamma offset=slave bundle=gmem7 depth=4096
#pragma HLS INTERFACE m_axi port=hbm_k offset=slave bundle=gmem8 depth=16384
#pragma HLS INTERFACE m_axi port=hbm_v offset=slave bundle=gmem9 depth=16384
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=current_length bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    constexpr int VECS_PER_Q = HEAD_DIM / VEC_W;                       
    constexpr int VECS_PER_KV_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W; 
    constexpr int HEADS_PER_KV = NUM_HEADS / NUM_KV_HEADS;             

    // Streams
    static hls_stream<vec_t<VEC_W>> s_in_attn("s_in_attn");
    static hls_stream<vec_t<VEC_W>> s_in_resid("s_in_resid");
    static hls_stream<vec_t<VEC_W>> s_norm("s_norm");
    static hls_stream<vec_t<VEC_W>> s_q_in("s_q_in"), s_k_in("s_k_in"), s_v_in("s_v_in");
    static hls_stream<vec_t<VEC_W>> s_q_proj("s_q_proj"), s_k_proj("s_k_proj"), s_v_proj("s_v_proj");
    static hls_stream<vec_t<VEC_W>> s_q_rot("s_q_rot"), s_k_rot("s_k_rot");
    static hls_stream<vec_t<VEC_W>> s_k_hist_raw("s_k_hist_raw"), s_v_hist_raw("s_v_hist_raw");
    static hls_stream<vec_t<VEC_W>> s_context("s_context");
    static hls_stream<vec_t<VEC_W>> s_o_proj("s_o_proj"), s_o_scaled("s_o_scaled");
    static hls_stream<vec_t<VEC_W>> s_res1("s_res1"), s_res1_norm_in("s_res1_norm_in"), s_res1_skip("s_res1_skip");
    static hls_stream<vec_t<VEC_W>> s_ffn_norm("s_ffn_norm"), s_gate_in("s_gate_in"), s_up_in("s_up_in");
    static hls_stream<vec_t<VEC_W>> s_gate_vec("s_gate_vec"), s_up_vec("s_up_vec"), s_swiglu("s_swiglu"), s_down("s_down");

    // Stage 0: split raw input for residual and norm path
    stream_dup<VEC_W>(in_stream, s_in_attn, s_in_resid, HIDDEN / VEC_W);

    // Stage 1: RMSNorm for attention
    rms_norm_stream<HIDDEN>(s_in_attn, s_norm, norm1_gamma, 1e-6f);

    // Stage 2: triplicate to Q/K/V branches
    stream_trip<VEC_W>(s_norm, s_q_in, s_k_in, s_v_in, HIDDEN / VEC_W);

    // Stage 4: projections (broadcast-packed weights)
    dense_projection_production_scaled<0, HIDDEN, HIDDEN>(s_q_in, s_q_proj, w_q, s_q);
    dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_k_in, s_k_proj, w_k, s_k);
    dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_v_in, s_v_proj, w_v, s_v);

    // Stage 5: RoPE on Q/K
    rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>(s_q_proj, s_q_rot, s_k_proj, s_k_rot, rope_cfg);

    // Stage 6: KV cache append and stream history once
    kv_cache_manager<HEAD_DIM, NUM_KV_HEADS>(s_k_rot, s_v_proj, s_k_hist_raw, s_v_hist_raw, hbm_k, hbm_v,
                                             current_length, true, true);

    // Stage 7: Buffer Q for all heads (post-RoPE)
    float q_buf[NUM_HEADS][HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = q_buf complete dim = 2
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int i = 0; i < VECS_PER_Q; ++i) {
#pragma HLS PIPELINE II = 1
            vec_t<VEC_W> chunk = s_q_rot.read();
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                q_buf[h][i * VEC_W + j] = chunk[j];
            }
        }
    }

    // Stage 8: Buffer KV history once into URAM then replay for each head.
    const int hist_len = current_length + 1; // includes the newly appended token
    static vec_t<VEC_W> k_hist_buf[MAX_CTX][VECS_PER_KV_TOKEN];
    static vec_t<VEC_W> v_hist_buf[MAX_CTX][VECS_PER_KV_TOKEN];
#pragma HLS BIND_STORAGE variable = k_hist_buf type = ram_2p impl = uram
#pragma HLS BIND_STORAGE variable = v_hist_buf type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = k_hist_buf cyclic factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = v_hist_buf cyclic factor = 2 dim = 2

    buffer_hist_loop:
    for (int t = 0; t < hist_len; ++t) {
        for (int v = 0; v < VECS_PER_KV_TOKEN; ++v) {
#pragma HLS PIPELINE II = 1
            k_hist_buf[t][v] = s_k_hist_raw.read();
            v_hist_buf[t][v] = s_v_hist_raw.read();
        }
    }

    // Multi-head attention, replaying buffered history
head_loop:
    for (int h = 0; h < NUM_HEADS; ++h) {
#pragma HLS LOOP_TRIPCOUNT max = NUM_HEADS
        const int kv_idx = h / HEADS_PER_KV; // 0 or 1

        hls_stream<vec_t<VEC_W>> q_head_stream("q_head_stream");
        hls_stream<vec_t<VEC_W>> k_head_stream("k_head_stream");
        hls_stream<vec_t<VEC_W>> v_head_stream("v_head_stream");
        hls_stream<vec_t<VEC_W>> ctx_head_stream("ctx_head_stream");

        // Stream Q for this head
        for (int i = 0; i < VECS_PER_Q; ++i) {
#pragma HLS PIPELINE II = 1
            vec_t<VEC_W> chunk;
            for (int j = 0; j < VEC_W; ++j) {
#pragma HLS UNROLL
                chunk[j] = q_buf[h][i * VEC_W + j];
            }
            q_head_stream.write(chunk);
        }

        // Replay buffered K/V for this head (filter KV head block)
    replay_hist:
        for (int t = 0; t < hist_len; ++t) {
            for (int v = 0; v < VECS_PER_KV_TOKEN; ++v) {
#pragma HLS PIPELINE II = 1
                const int block = v / VECS_PER_Q; // which KV head
                if (block == kv_idx) {
                    k_head_stream.write(k_hist_buf[t][v]);
                    v_head_stream.write(v_hist_buf[t][v]);
                }
            }
        }

        // Run attention for this head
        const int padded_len = ((hist_len + 127) / 128) * 128;
        attention_solver<HEAD_DIM>(q_head_stream, k_head_stream, v_head_stream, ctx_head_stream, hist_len, padded_len);

        // Append context
        for (int i = 0; i < VECS_PER_Q; ++i) {
#pragma HLS PIPELINE II = 1
            s_context.write(ctx_head_stream.read());
        }
    }

    // Stage 9: Output projection (use TMAC kernel for quantized weights)
    dense_projection_production_scaled<0, HIDDEN, HIDDEN, 128, false>(s_context, s_o_proj, w_o, s_o);

    // Stage 10: scale attention output and residual add (skip uses raw input)
    stream_scale<VEC_W>(s_o_proj, s_o_scaled, RESIDUAL_SCALE, HIDDEN / VEC_W);
    stream_add<VEC_W>(s_o_scaled, s_in_resid, s_res1, HIDDEN / VEC_W);

    // Duplicate for FFN residual
    stream_dup<VEC_W>(s_res1, s_res1_norm_in, s_res1_skip, HIDDEN / VEC_W);

    // Stage 11: RMSNorm before FFN
    rms_norm_stream<HIDDEN>(s_res1_norm_in, s_ffn_norm, norm2_gamma, 1e-6f);

    // Stage 12: Gate/Up projections
    stream_dup<VEC_W>(s_ffn_norm, s_gate_in, s_up_in, HIDDEN / VEC_W);
    dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE, 128, false>(s_gate_in, s_gate_vec, w_gate, gate_scales);
    dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE, 128, false>(s_up_in, s_up_vec, w_up, up_scales);

    // Stage 13: SiLU * Up
    silu_mul_stream<VEC_W>(s_gate_vec, s_up_vec, s_swiglu, INTERMEDIATE / VEC_W);

    // Stage 14: Down projection AND Final Residual Add
    dense_projection_production_scaled<0, INTERMEDIATE, HIDDEN, 128, false>(s_swiglu, s_down, w_down, down_scales);
    
    // FIX: Combine FFN output (s_down) with the residual skip (s_res1_skip)
    stream_add<VEC_W>(s_down, s_res1_skip, out_stream, HIDDEN / VEC_W);
}

} // namespace hls
} // namespace tmac