#include "eagle_tier1_top.hpp"

namespace tmac {
namespace hls {

// Internal constants
constexpr int VECS_PER_Q = HEAD_DIM / VEC_W;                       
constexpr int VECS_PER_KV_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W; 
constexpr int HEADS_PER_KV = NUM_HEADS / NUM_KV_HEADS;  

static_assert(NUM_HEADS % NUM_KV_HEADS == 0, "NUM_HEADS must be divisible by NUM_KV_HEADS");
static_assert(HIDDEN == NUM_HEADS * HEAD_DIM, "HIDDEN must equal NUM_HEADS * HEAD_DIM");

// q_rot stream order is head-major (h0 vecs, h1 vecs, ...). Route each chunk to its head.
void distribute_q_heads(hls_stream<vec_t<VEC_W>>& s_q_rot, hls_stream<vec_t<VEC_W>> q_head_streams[NUM_HEADS]) {
#pragma HLS INLINE off
head_loop:
    for (int h = 0; h < NUM_HEADS; ++h) {
    vec_loop:
        for (int i = 0; i < VECS_PER_Q; ++i) {
#pragma HLS PIPELINE II = 1
            q_head_streams[h].write(s_q_rot.read());
        }
    }
}

void broadcast_kv_heads(
    hls_stream<vec_t<VEC_W>>& s_k_hist_raw,
    hls_stream<vec_t<VEC_W>>& s_v_hist_raw,
    hls_stream<vec_t<VEC_W>> k_head_streams[NUM_HEADS],
    hls_stream<vec_t<VEC_W>> v_head_streams[NUM_HEADS],
    int hist_len
){
#pragma HLS INLINE off
token_loop:
    for (int t = 0; t < hist_len; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 avg=MAX_CTX/2 max=MAX_CTX
    kv_vec_loop:
        for (int v = 0; v < VECS_PER_KV_TOKEN; ++v) {
#pragma HLS PIPELINE II = 1
            int kvh = v / VECS_PER_Q;
            vec_t<VEC_W> ek = s_k_hist_raw.read();
            vec_t<VEC_W> ev = s_v_hist_raw.read();
            
        dup_head_loop:
            for (int h = 0; h < HEADS_PER_KV; h++) {
#pragma HLS UNROLL
                k_head_streams[kvh * HEADS_PER_KV + h].write(ek);
                v_head_streams[kvh * HEADS_PER_KV + h].write(ev);
            }
        }
    }
}

void grouped_query_attention(
    hls_stream<vec_t<VEC_W>> q_head_streams[NUM_HEADS],
    hls_stream<vec_t<VEC_W>> k_head_streams[NUM_HEADS],
    hls_stream<vec_t<VEC_W>> v_head_streams[NUM_HEADS],
    hls_stream<vec_t<VEC_W>> ctx_head_streams[NUM_HEADS],
    int hist_len,
    int padded_len
) {
#pragma HLS INLINE off
    for (int h = 0; h < NUM_HEADS; ++h) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT max = NUM_HEADS
#if TMAC_ATTN_SOLVER_MODE == 0
        attention_solver<HEAD_DIM>(q_head_streams[h], k_head_streams[h], v_head_streams[h], ctx_head_streams[h], hist_len, padded_len);
#elif TMAC_ATTN_SOLVER_MODE == 1
        fused_online_attention_pwl<HEAD_DIM>(q_head_streams[h], k_head_streams[h], v_head_streams[h], ctx_head_streams[h], hist_len, padded_len);
#else
        if (hist_len >= TMAC_ATTN_FUSED_SWITCH_LEN) {
            fused_online_attention_pwl<HEAD_DIM>(q_head_streams[h], k_head_streams[h], v_head_streams[h], ctx_head_streams[h], hist_len, padded_len);
        } else {
            attention_solver<HEAD_DIM>(q_head_streams[h], k_head_streams[h], v_head_streams[h], ctx_head_streams[h], hist_len, padded_len);
        }
#endif
    }
}

void collect_ctx(hls_stream<vec_t<VEC_W>>& s_context, hls_stream<vec_t<VEC_W>> ctx_head_streams[NUM_HEADS]) {
#pragma HLS INLINE off
head_loop:
    for (int i = 0; i < NUM_HEADS; i++) {
    vec_loop:
        for (int j = 0; j < VECS_PER_Q; ++j) {
#pragma HLS PIPELINE II = 1
            s_context.write(ctx_head_streams[i].read());
        }
    }
}

void concat_embed_hidden(hls_stream<vec_t<VEC_W>>& s_embed_norm,
                         hls_stream<vec_t<VEC_W>>& s_hidden_norm,
                         hls_stream<vec_t<VEC_W>>& s_attn_cat) {
#pragma HLS INLINE off
embed_loop:
    for (int i = 0; i < HIDDEN / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
        s_attn_cat.write(s_embed_norm.read());
    }
hidden_loop:
    for (int i = 0; i < HIDDEN / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
        s_attn_cat.write(s_hidden_norm.read());
    }
}

void split_down_2hs(hls_stream<vec_t<VEC_W>>& s_down_2hs,
                    hls_stream<vec_t<VEC_W>>& s_to_logits,
                    hls_stream<vec_t<VEC_W>>& s_for_reasoning) {
#pragma HLS INLINE off
to_logits_loop:
    for (int i = 0; i < HIDDEN / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
        s_to_logits.write(s_down_2hs.read());
    }
reasoning_loop:
    for (int i = 0; i < HIDDEN / VEC_W; ++i) {
#pragma HLS PIPELINE II = 1
        s_for_reasoning.write(s_down_2hs.read());
    }
}

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
#pragma HLS INTERFACE m_axi port=w_q offset=slave bundle=gmem0 depth=TMAC_W_Q_DEPTH
#pragma HLS INTERFACE m_axi port=s_q offset=slave bundle=gmem0 depth=TMAC_S_Q_DEPTH
#pragma HLS INTERFACE m_axi port=w_k offset=slave bundle=gmem1 depth=TMAC_W_KV_DEPTH
#pragma HLS INTERFACE m_axi port=s_k offset=slave bundle=gmem1 depth=TMAC_S_KV_DEPTH
#pragma HLS INTERFACE m_axi port=w_v offset=slave bundle=gmem2 depth=TMAC_W_KV_DEPTH
#pragma HLS INTERFACE m_axi port=s_v offset=slave bundle=gmem2 depth=TMAC_S_KV_DEPTH
#pragma HLS INTERFACE m_axi port=w_o offset=slave bundle=gmem3 depth=TMAC_W_O_DEPTH
#pragma HLS INTERFACE m_axi port=s_o offset=slave bundle=gmem3 depth=TMAC_S_O_DEPTH
#pragma HLS INTERFACE m_axi port=w_gate offset=slave bundle=gmem4 depth=TMAC_W_GATE_UP_DEPTH
#pragma HLS INTERFACE m_axi port=gate_scales offset=slave bundle=gmem4 depth=TMAC_S_GATE_UP_DEPTH
#pragma HLS INTERFACE m_axi port=w_up offset=slave bundle=gmem5 depth=TMAC_W_GATE_UP_DEPTH
#pragma HLS INTERFACE m_axi port=up_scales offset=slave bundle=gmem5 depth=TMAC_S_GATE_UP_DEPTH
#pragma HLS INTERFACE m_axi port=w_down offset=slave bundle=gmem6 depth=TMAC_W_DOWN_DEPTH
#pragma HLS INTERFACE m_axi port=down_scales offset=slave bundle=gmem6 depth=TMAC_S_DOWN_DEPTH
#pragma HLS INTERFACE m_axi port=norm1_gamma offset=slave bundle=gmem7 depth=TMAC_HIDDEN_SIZE
#pragma HLS INTERFACE m_axi port=norm2_gamma offset=slave bundle=gmem7 depth=TMAC_HIDDEN_SIZE
#pragma HLS INTERFACE m_axi port=hbm_k offset=slave bundle=gmem8 depth=TMAC_KV_CACHE_DEPTH
#pragma HLS INTERFACE m_axi port=hbm_v offset=slave bundle=gmem9 depth=TMAC_KV_CACHE_DEPTH
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=current_length bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW          

    // Streams
    hls_stream<vec_t<VEC_W>> s_in_attn("s_in_attn");
    hls_stream<vec_t<VEC_W>> s_in_resid("s_in_resid");
    hls_stream<vec_t<VEC_W>> s_norm("s_norm");
    hls_stream<vec_t<VEC_W>> s_q_in("s_q_in"), s_k_in("s_k_in"), s_v_in("s_v_in");
    hls_stream<vec_t<VEC_W>> s_q_proj("s_q_proj"), s_k_proj("s_k_proj"), s_v_proj("s_v_proj");
    hls_stream<vec_t<VEC_W>> s_q_rot("s_q_rot"), s_k_rot("s_k_rot");
    hls_stream<vec_t<VEC_W>> s_k_hist_raw("s_k_hist_raw"), s_v_hist_raw("s_v_hist_raw");
    hls_stream<vec_t<VEC_W>> s_context("s_context");
    hls_stream<vec_t<VEC_W>> s_o_proj("s_o_proj"), s_o_scaled("s_o_scaled");
    hls_stream<vec_t<VEC_W>> s_res1("s_res1"), s_res1_norm_in("s_res1_norm_in"), s_res1_skip("s_res1_skip");
    hls_stream<vec_t<VEC_W>> s_ffn_norm("s_ffn_norm"), s_gate_in("s_gate_in"), s_up_in("s_up_in");
    hls_stream<vec_t<VEC_W>> s_gate_vec("s_gate_vec"), s_up_vec("s_up_vec"), s_swiglu("s_swiglu"), s_down("s_down");

    hls_stream<vec_t<VEC_W>> q_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> k_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> v_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> ctx_head_streams[NUM_HEADS];

    // Stage 0: split raw input for residual and norm path
    stream_dup<VEC_W>(in_stream, s_in_attn, s_in_resid, HIDDEN / VEC_W);

    // Stage 1: RMSNorm for attention
    rms_norm_stream<HIDDEN>(s_in_attn, s_norm, norm1_gamma, 1e-6f);

    // Stage 2: triplicate to Q/K/V branches
    stream_trip<VEC_W>(s_norm, s_q_in, s_k_in, s_v_in, HIDDEN / VEC_W);

    // Stage 4: projections (broadcast-packed weights)
    dense_projection_production_scaled<0, HIDDEN, HIDDEN, 128, TMAC_USE_TMAC_QKV>(s_q_in, s_q_proj, w_q, s_q);
    dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM, 128, TMAC_USE_TMAC_QKV>(s_k_in, s_k_proj, w_k, s_k);
    dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM, 128, TMAC_USE_TMAC_QKV>(s_v_in, s_v_proj, w_v, s_v);

    // Stage 5: RoPE on Q/K
    rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>(s_q_proj, s_q_rot, s_k_proj, s_k_rot, rope_cfg);

    // Stage 6: KV cache append and stream history once
    kv_cache_manager<HEAD_DIM, NUM_KV_HEADS>(s_k_rot, s_v_proj, s_k_hist_raw, s_v_hist_raw, hbm_k, hbm_v,
                                             current_length, true, true);

    const int hist_len = current_length + 1; // includes the newly appended token
    const int padded_len = ((hist_len + 127) / 128) * 128;

    distribute_q_heads(s_q_rot, q_head_streams);
    broadcast_kv_heads(s_k_hist_raw, s_v_hist_raw, k_head_streams, v_head_streams, hist_len);
    grouped_query_attention(q_head_streams, k_head_streams, v_head_streams, ctx_head_streams, hist_len, padded_len);
    collect_ctx(s_context, ctx_head_streams);

    // Stage 9: Output projection (use TMAC kernel for quantized weights)
    dense_projection_production_scaled<0, HIDDEN, HIDDEN, 128, TMAC_USE_TMAC_O>(s_context, s_o_proj, w_o, s_o);

    // Stage 10: scale attention output and residual add (skip uses raw input)
    stream_scale<VEC_W>(s_o_proj, s_o_scaled, RESIDUAL_SCALE, HIDDEN / VEC_W);
    stream_add<VEC_W>(s_o_scaled, s_in_resid, s_res1, HIDDEN / VEC_W);

    // Duplicate for FFN residual
    stream_dup<VEC_W>(s_res1, s_res1_norm_in, s_res1_skip, HIDDEN / VEC_W);

    // Stage 11: RMSNorm before FFN
    rms_norm_stream<HIDDEN>(s_res1_norm_in, s_ffn_norm, norm2_gamma, 1e-6f);

    // Stage 12: Gate/Up projections
    stream_dup<VEC_W>(s_ffn_norm, s_gate_in, s_up_in, HIDDEN / VEC_W);
    dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE, 128, TMAC_USE_TMAC_FFN>(s_gate_in, s_gate_vec, w_gate, gate_scales);
    dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE, 128, TMAC_USE_TMAC_FFN>(s_up_in, s_up_vec, w_up, up_scales);

    // Stage 13: SiLU * Up
    silu_mul_stream<VEC_W>(s_gate_vec, s_up_vec, s_swiglu, INTERMEDIATE / VEC_W);

    // Stage 14: Down projection AND Final Residual Add
    dense_projection_production_scaled<0, INTERMEDIATE, HIDDEN, 128, TMAC_USE_TMAC_FFN>(s_swiglu, s_down, w_down, down_scales);
    
    // FIX: Combine FFN output (s_down) with the residual skip (s_res1_skip)
    stream_add<VEC_W>(s_down, s_res1_skip, out_stream, HIDDEN / VEC_W);
}

void eagle_tier1_top_eagle4_l0(
    hls_stream<vec_t<VEC_W>>& hidden_in_stream,
    hls_stream<vec_t<VEC_W>>& embed_in_stream,
    hls_stream<vec_t<VEC_W>>& reasoning_out_stream,
    hls_stream<vec_t<VEC_W>>& logits_norm_out_stream,
    const pack512* w_q,     const float* s_q,
    const pack512* w_k,     const float* s_k,
    const pack512* w_v,     const float* s_v,
    const pack512* w_o,     const float* s_o,
    const pack512* w_gate,  const float* gate_scales,
    const pack512* w_up,    const float* up_scales,
    const pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& rope_cfg,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    int seq_len,
    int current_length
) {
#pragma HLS INTERFACE axis port=hidden_in_stream
#pragma HLS INTERFACE axis port=embed_in_stream
#pragma HLS INTERFACE axis port=reasoning_out_stream
#pragma HLS INTERFACE axis port=logits_norm_out_stream
#pragma HLS INTERFACE m_axi port=w_q offset=slave bundle=gmem0 depth=TMAC_W_Q_DEPTH
#pragma HLS INTERFACE m_axi port=s_q offset=slave bundle=gmem0 depth=TMAC_S_Q_DEPTH
#pragma HLS INTERFACE m_axi port=w_k offset=slave bundle=gmem1 depth=TMAC_W_KV_DEPTH
#pragma HLS INTERFACE m_axi port=s_k offset=slave bundle=gmem1 depth=TMAC_S_KV_DEPTH
#pragma HLS INTERFACE m_axi port=w_v offset=slave bundle=gmem2 depth=TMAC_W_KV_DEPTH
#pragma HLS INTERFACE m_axi port=s_v offset=slave bundle=gmem2 depth=TMAC_S_KV_DEPTH
#pragma HLS INTERFACE m_axi port=w_o offset=slave bundle=gmem3 depth=TMAC_W_O_DEPTH
#pragma HLS INTERFACE m_axi port=s_o offset=slave bundle=gmem3 depth=TMAC_S_O_DEPTH
#pragma HLS INTERFACE m_axi port=w_gate offset=slave bundle=gmem4 depth=TMAC_W_GATE_UP_DEPTH
#pragma HLS INTERFACE m_axi port=gate_scales offset=slave bundle=gmem4 depth=TMAC_S_GATE_UP_DEPTH
#pragma HLS INTERFACE m_axi port=w_up offset=slave bundle=gmem5 depth=TMAC_W_GATE_UP_DEPTH
#pragma HLS INTERFACE m_axi port=up_scales offset=slave bundle=gmem5 depth=TMAC_S_GATE_UP_DEPTH
#pragma HLS INTERFACE m_axi port=w_down offset=slave bundle=gmem6 depth=TMAC_W_DOWN_DEPTH
#pragma HLS INTERFACE m_axi port=down_scales offset=slave bundle=gmem6 depth=TMAC_S_DOWN_DEPTH
#pragma HLS INTERFACE m_axi port=hidden_norm_gamma offset=slave bundle=gmem7 depth=TMAC_HIDDEN_SIZE
#pragma HLS INTERFACE m_axi port=embed_norm_gamma offset=slave bundle=gmem7 depth=TMAC_HIDDEN_SIZE
#pragma HLS INTERFACE m_axi port=post_attn_norm_gamma offset=slave bundle=gmem7 depth=TMAC_HIDDEN_SIZE
#pragma HLS INTERFACE m_axi port=final_norm_gamma offset=slave bundle=gmem7 depth=TMAC_HIDDEN_SIZE
#pragma HLS INTERFACE m_axi port=hbm_k offset=slave bundle=gmem8 depth=TMAC_KV_CACHE_DEPTH
#pragma HLS INTERFACE m_axi port=hbm_v offset=slave bundle=gmem9 depth=TMAC_KV_CACHE_DEPTH
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=current_length bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    hls_stream<vec_t<VEC_W>> s_hidden_norm_in("s_hidden_norm_in");
    hls_stream<vec_t<VEC_W>> s_hidden_residual("s_hidden_residual");
    hls_stream<vec_t<VEC_W>> s_hidden_norm("s_hidden_norm");
    hls_stream<vec_t<VEC_W>> s_embed_norm("s_embed_norm");
    hls_stream<vec_t<VEC_W>> s_attn_cat("s_attn_cat");
    hls_stream<vec_t<VEC_W>> s_q_in("s_q_in"), s_k_in("s_k_in"), s_v_in("s_v_in");
    hls_stream<vec_t<VEC_W>> s_q_proj("s_q_proj"), s_k_proj("s_k_proj"), s_v_proj("s_v_proj");
    hls_stream<vec_t<VEC_W>> s_q_rot("s_q_rot"), s_k_rot("s_k_rot");
    hls_stream<vec_t<VEC_W>> s_k_hist_raw("s_k_hist_raw"), s_v_hist_raw("s_v_hist_raw");
    hls_stream<vec_t<VEC_W>> s_context("s_context");
    hls_stream<vec_t<VEC_W>> s_o_proj("s_o_proj");
    hls_stream<vec_t<VEC_W>> s_post_attn_residual("s_post_attn_residual");
    hls_stream<vec_t<VEC_W>> s_post_attn_norm_in("s_post_attn_norm_in");
    hls_stream<vec_t<VEC_W>> s_post_attn_residual_for_add("s_post_attn_residual_for_add");
    hls_stream<vec_t<VEC_W>> s_post_attn_norm("s_post_attn_norm");
    hls_stream<vec_t<VEC_W>> s_gate_in("s_gate_in"), s_up_in("s_up_in");
    hls_stream<vec_t<VEC_W>> s_gate_vec("s_gate_vec"), s_up_vec("s_up_vec"), s_swiglu("s_swiglu");
    hls_stream<vec_t<VEC_W>> s_down_2hs("s_down_2hs");
    hls_stream<vec_t<VEC_W>> s_to_logits_raw("s_to_logits_raw");
    hls_stream<vec_t<VEC_W>> s_for_reasoning("s_for_reasoning");

    hls_stream<vec_t<VEC_W>> q_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> k_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> v_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> ctx_head_streams[NUM_HEADS];

    // Stage 0: duplicate hidden input for norm and residual branches
    stream_dup<VEC_W>(hidden_in_stream, s_hidden_norm_in, s_hidden_residual, HIDDEN / VEC_W);

    // Stage 1: independent RMSNorm on hidden and embed branches
    rms_norm_stream<HIDDEN>(s_hidden_norm_in, s_hidden_norm, hidden_norm_gamma, RMS_EPS);
    rms_norm_stream<HIDDEN>(embed_in_stream, s_embed_norm, embed_norm_gamma, RMS_EPS);

    // Stage 2: layer-0 attention input concat [embed_norm || hidden_norm] (2H)
    concat_embed_hidden(s_embed_norm, s_hidden_norm, s_attn_cat);

    // Stage 3: triplicate for Q/K/V paths
    stream_trip<VEC_W>(s_attn_cat, s_q_in, s_k_in, s_v_in, QKV_INPUT / VEC_W);

    // Stage 4: Q/K/V projections with 2H input
    dense_projection_production_scaled<0, QKV_INPUT, HIDDEN, 128, TMAC_USE_TMAC_QKV>(s_q_in, s_q_proj, w_q, s_q);
    dense_projection_production_scaled<0, QKV_INPUT, NUM_KV_HEADS * HEAD_DIM, 128, TMAC_USE_TMAC_QKV>(s_k_in, s_k_proj, w_k, s_k);
    dense_projection_production_scaled<0, QKV_INPUT, NUM_KV_HEADS * HEAD_DIM, 128, TMAC_USE_TMAC_QKV>(s_v_in, s_v_proj, w_v, s_v);

    // Stage 5: RoPE on Q/K
    rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>(s_q_proj, s_q_rot, s_k_proj, s_k_rot, rope_cfg);

    // Stage 6: KV cache append + history stream
    kv_cache_manager<HEAD_DIM, NUM_KV_HEADS>(s_k_rot, s_v_proj, s_k_hist_raw, s_v_hist_raw, hbm_k, hbm_v,
                                             current_length, true, true);

    const int hist_len = current_length + 1;
    const int padded_len = ((hist_len + 127) / 128) * 128;

    // Stage 7: grouped query attention
    distribute_q_heads(s_q_rot, q_head_streams);
    broadcast_kv_heads(s_k_hist_raw, s_v_hist_raw, k_head_streams, v_head_streams, hist_len);
    grouped_query_attention(q_head_streams, k_head_streams, v_head_streams, ctx_head_streams, hist_len, padded_len);
    collect_ctx(s_context, ctx_head_streams);

    // Stage 8: output projection
    dense_projection_production_scaled<0, HIDDEN, HIDDEN, 128, TMAC_USE_TMAC_O>(s_context, s_o_proj, w_o, s_o);

    // Stage 9: post-attn residual + post-attn RMSNorm (returns both residual and normalized stream)
    stream_add<VEC_W>(s_o_proj, s_hidden_residual, s_post_attn_residual, HIDDEN / VEC_W);
    stream_dup<VEC_W>(s_post_attn_residual, s_post_attn_norm_in, s_post_attn_residual_for_add, HIDDEN / VEC_W);
    rms_norm_stream<HIDDEN>(s_post_attn_norm_in, s_post_attn_norm, post_attn_norm_gamma, RMS_EPS);

    // Stage 10: FFN gate/up + SiLU
    stream_dup<VEC_W>(s_post_attn_norm, s_gate_in, s_up_in, HIDDEN / VEC_W);
    dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE, 128, TMAC_USE_TMAC_FFN>(s_gate_in, s_gate_vec, w_gate, gate_scales);
    dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE, 128, TMAC_USE_TMAC_FFN>(s_up_in, s_up_vec, w_up, up_scales);
    silu_mul_stream<VEC_W>(s_gate_vec, s_up_vec, s_swiglu, INTERMEDIATE / VEC_W);

    // Stage 11: down-proj to 2H and split (to_logits, for_reasoning)
    dense_projection_production_scaled<0, INTERMEDIATE, DOWN_OUTPUT, 128, TMAC_USE_TMAC_FFN>(s_swiglu, s_down_2hs, w_down, down_scales);
    split_down_2hs(s_down_2hs, s_to_logits_raw, s_for_reasoning);

    // Stage 12: final norm on logits stream and residual add on reasoning stream
    rms_norm_stream<HIDDEN>(s_to_logits_raw, logits_norm_out_stream, final_norm_gamma, RMS_EPS);
    stream_add<VEC_W>(s_for_reasoning, s_post_attn_residual_for_add, reasoning_out_stream, HIDDEN / VEC_W);
}

} // namespace hls
} // namespace tmac
