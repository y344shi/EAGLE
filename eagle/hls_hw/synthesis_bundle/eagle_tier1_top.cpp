#include "eagle_tier1_top.hpp"

namespace tmac {
namespace hls {

// Internal constants
constexpr int VECS_PER_Q = HEAD_DIM / VEC_W;                       
constexpr int VECS_PER_KV_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W; 
constexpr int HEADS_PER_KV = NUM_HEADS / NUM_KV_HEADS;  

static_assert(NUM_HEADS % NUM_KV_HEADS == 0, "NUM_HEADS must be divisible by NUM_KV_HEADS");
static_assert(HIDDEN == NUM_HEADS * HEAD_DIM, "HIDDEN must equal NUM_HEADS * HEAD_DIM");

// q_rot stream order is token major then head-major (h0 vecs, h1 vecs, ...). Route each chunk to its head.
void distribute_q_heads(hls_stream<vec_t<VEC_W>>& s_q_rot, hls_stream<vec_t<VEC_W>> q_head_streams[TREE_WIDTH][NUM_HEADS]) {
#pragma HLS INLINE off
for (int t = 0; t < TREE_WIDTH; t++) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
head_loop:
    for (int h = 0; h < NUM_HEADS; ++h) {
#pragma HLS loop_tripcount min=NUM_HEADS max=NUM_HEADS avg=NUM_HEADS
    vec_loop:
        for (int i = 0; i < VECS_PER_Q; ++i) {
#pragma HLS loop_tripcount min=VECS_PER_Q max=VECS_PER_Q avg=VECS_PER_Q
#pragma HLS PIPELINE II = 1
            q_head_streams[t][h].write(s_q_rot.read());
        }
    }
}
}

void broadcast_kv_heads(
    hls_stream<vec_t<VEC_W>> s_k_hist_raw[TREE_WIDTH],
    hls_stream<vec_t<VEC_W>> s_v_hist_raw[TREE_WIDTH],
    hls_stream<vec_t<VEC_W>> k_head_streams[TREE_WIDTH][NUM_HEADS],
    hls_stream<vec_t<VEC_W>> v_head_streams[TREE_WIDTH][NUM_HEADS],
    int hist_len
){
#pragma HLS INLINE off
    for (int t = 0; t < TREE_WIDTH; t++) {
#pragma HLS UNROLL
    token_loop:
        for (int _ = 0; _ < hist_len; ++_) {
    #pragma HLS LOOP_TRIPCOUNT min=1 avg=MAX_CTX/2 max=MAX_CTX
        kv_vec_loop:
            for (int v = 0; v < VECS_PER_KV_TOKEN; ++v) {
    #pragma HLS loop_tripcount min=VECS_PER_KV_TOKEN max=VECS_PER_KV_TOKEN avg=VECS_PER_KV_TOKEN
    #pragma HLS PIPELINE II = 1
                int kvh = v / VECS_PER_Q;
                vec_t<VEC_W> ek = s_k_hist_raw[t].read();
                vec_t<VEC_W> ev = s_v_hist_raw[t].read();
                
            dup_head_loop:
                for (int h = 0; h < HEADS_PER_KV; h++) {
    #pragma HLS UNROLL
                    k_head_streams[t][kvh * HEADS_PER_KV + h].write(ek);
                    v_head_streams[t][kvh * HEADS_PER_KV + h].write(ev);
                }
            }
        }
    }
}

void grouped_query_attention(
    hls_stream<vec_t<VEC_W>> q_head_streams[TREE_WIDTH][NUM_HEADS],
    hls_stream<vec_t<VEC_W>> k_head_streams[TREE_WIDTH][NUM_HEADS],
    hls_stream<vec_t<VEC_W>> v_head_streams[TREE_WIDTH][NUM_HEADS],
    hls_stream<vec_t<VEC_W>> ctx_head_streams[TREE_WIDTH][NUM_HEADS],
    int hist_len,
    int padded_len
) {
#pragma HLS INLINE off
for (int t = 0; t < TREE_WIDTH; t++) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT max = TREE_WIDTH
    for (int h = 0; h < NUM_HEADS; ++h) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT max = NUM_HEADS
#if TMAC_ATTN_SOLVER_MODE == 0
        attention_solver<HEAD_DIM>(q_head_streams[t][h], k_head_streams[t][h], v_head_streams[t][h], ctx_head_streams[t][h], hist_len, padded_len);
#elif TMAC_ATTN_SOLVER_MODE == 1
        fused_online_attention_pwl<HEAD_DIM>(q_head_streams[t][h], k_head_streams[t][h], v_head_streams[t][h], ctx_head_streams[t][h], hist_len, padded_len);
#else
        if (hist_len >= TMAC_ATTN_FUSED_SWITCH_LEN) {
            fused_online_attention_pwl<HEAD_DIM>(q_head_streams[t][h], k_head_streams[t][h], v_head_streams[t][h], ctx_head_streams[t][h], hist_len, padded_len);
        } else {
            attention_solver<HEAD_DIM>(q_head_streams[t][h], k_head_streams[t][h], v_head_streams[t][h], ctx_head_streams[t][h], hist_len, padded_len);
        }
#endif
    }
}
}

void collect_ctx(hls_stream<vec_t<VEC_W>>& s_context, hls_stream<vec_t<VEC_W>> ctx_head_streams[TREE_WIDTH][NUM_HEADS]) {
#pragma HLS INLINE off
    for (int t = 0; t < TREE_WIDTH; t++) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
    head_loop:
        for (int i = 0; i < NUM_HEADS; i++) {
#pragma HLS loop_tripcount min=NUM_HEADS max=NUM_HEADS avg=NUM_HEADS
        vec_loop:
            for (int j = 0; j < VECS_PER_Q; ++j) {
#pragma HLS loop_tripcount min=VECS_PER_Q max=VECS_PER_Q avg=VECS_PER_Q
    #pragma HLS PIPELINE II = 1
                s_context.write(ctx_head_streams[t][i].read());
            }
        }
    }
}

void concat_embed_hidden(hls_stream<vec_t<VEC_W>>& s_embed_norm,
                         hls_stream<vec_t<VEC_W>>& s_hidden_norm,
                         hls_stream<vec_t<VEC_W>>& s_attn_cat) {
#pragma HLS INLINE off
    constexpr int VECS_PER_H = HIDDEN / VEC_W;
token_cat_loop:
    for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
embed_loop:
        for (int i = 0; i < VECS_PER_H; ++i) {
#pragma HLS loop_tripcount min=HIDDEN/VEC_W max=HIDDEN/VEC_W avg=HIDDEN/VEC_W
#pragma HLS PIPELINE II = 1
        s_attn_cat.write(s_embed_norm.read());
    }
hidden_loop:
        for (int i = 0; i < VECS_PER_H; ++i) {
#pragma HLS loop_tripcount min=HIDDEN/VEC_W max=HIDDEN/VEC_W avg=HIDDEN/VEC_W
#pragma HLS PIPELINE II = 1
        s_attn_cat.write(s_hidden_norm.read());
        }
    }
}

void split_down_2hs(hls_stream<vec_t<VEC_W>>& s_down_2hs,
                    hls_stream<vec_t<VEC_W>>& s_to_logits,
                    hls_stream<vec_t<VEC_W>>& s_for_reasoning) {
#pragma HLS INLINE off
    constexpr int VECS_PER_H = HIDDEN / VEC_W;
token_split_loop:
    for (int t = 0; t < TREE_WIDTH; ++t) {
#pragma HLS loop_tripcount min=TREE_WIDTH max=TREE_WIDTH avg=TREE_WIDTH
to_logits_loop:
        for (int i = 0; i < VECS_PER_H; ++i) {
#pragma HLS loop_tripcount min=HIDDEN/VEC_W max=HIDDEN/VEC_W avg=HIDDEN/VEC_W
#pragma HLS PIPELINE II = 1
        s_to_logits.write(s_down_2hs.read());
    }
reasoning_loop:
        for (int i = 0; i < VECS_PER_H; ++i) {
#pragma HLS loop_tripcount min=HIDDEN/VEC_W max=HIDDEN/VEC_W avg=HIDDEN/VEC_W
#pragma HLS PIPELINE II = 1
        s_for_reasoning.write(s_down_2hs.read());
        }
    }
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
    int prefix_len,
    int current_depth,
    const int* parent_indices_per_layer
) {

#pragma HLS DATAFLOW
    hls_stream<vec_t<VEC_W>> s_hidden_norm_in("s_hidden_norm_in");
    hls_stream<vec_t<VEC_W>> s_hidden_residual("s_hidden_residual");
    hls_stream<vec_t<VEC_W>> s_hidden_norm("s_hidden_norm");
    hls_stream<vec_t<VEC_W>> s_embed_norm("s_embed_norm");
    hls_stream<vec_t<VEC_W>> s_attn_cat("s_attn_cat");
    hls_stream<vec_t<VEC_W>> s_q_in("s_q_in"), s_k_in("s_k_in"), s_v_in("s_v_in");
    hls_stream<vec_t<VEC_W>> s_q_proj("s_q_proj"), s_k_proj("s_k_proj"), s_v_proj("s_v_proj");
    hls_stream<vec_t<VEC_W>> s_q_rot("s_q_rot"), s_k_rot("s_k_rot");
    hls_stream<vec_t<VEC_W>> s_k_hist_raw[TREE_WIDTH];
    hls_stream<vec_t<VEC_W>> s_v_hist_raw[TREE_WIDTH];
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

    hls_stream<vec_t<VEC_W>> q_head_streams[TREE_WIDTH][NUM_HEADS];
    hls_stream<vec_t<VEC_W>> k_head_streams[TREE_WIDTH][NUM_HEADS];
    hls_stream<vec_t<VEC_W>> v_head_streams[TREE_WIDTH][NUM_HEADS];
    hls_stream<vec_t<VEC_W>> ctx_head_streams[TREE_WIDTH][NUM_HEADS];

    // Stage 0: duplicate hidden input for norm and residual branches
    stream_dup<VEC_W>(hidden_in_stream, s_hidden_norm_in, s_hidden_residual, NUM_CHUNKS);

    // Stage 1: independent RMSNorm on hidden and embed branches
    rms_norm_stream<HIDDEN, TREE_WIDTH>(s_hidden_norm_in, s_hidden_norm, hidden_norm_gamma, RMS_EPS);
    rms_norm_stream<HIDDEN, TREE_WIDTH>(embed_in_stream, s_embed_norm, embed_norm_gamma, RMS_EPS);

    // Stage 2: layer-0 attention input concat [embed_norm || hidden_norm] (2H)
    concat_embed_hidden(s_embed_norm, s_hidden_norm, s_attn_cat);

    // Stage 3: triplicate for Q/K/V paths
    stream_trip<VEC_W>(s_attn_cat, s_q_in, s_k_in, s_v_in, NUM_CHUNKS * 2);

    // Stage 4: Q/K/V projections with 2H input
    dense_projection_production_scaled_batched<0, TREE_WIDTH, QKV_INPUT, HIDDEN, 128, TMAC_USE_TMAC_QKV>(s_q_in, s_q_proj, w_q, s_q);
    dense_projection_production_scaled_batched<0, TREE_WIDTH, QKV_INPUT, NUM_KV_HEADS * HEAD_DIM, 128, TMAC_USE_TMAC_QKV>(s_k_in, s_k_proj, w_k, s_k);
    dense_projection_production_scaled_batched<0, TREE_WIDTH, QKV_INPUT, NUM_KV_HEADS * HEAD_DIM, 128, TMAC_USE_TMAC_QKV>(s_v_in, s_v_proj, w_v, s_v);

    // Stage 5: RoPE on Q/K
    rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, TREE_WIDTH>(s_q_proj, s_q_rot, s_k_proj, s_k_rot, rope_cfg);

    // Stage 6: Write new KV to contiguous HBM, then gather prefix + ancestors + self.
    contiguous_kv_write_and_gather<HEAD_DIM, NUM_KV_HEADS, kMaxDraftDepth>(
        s_k_rot, s_v_proj, hbm_k, hbm_v,
        prefix_len, current_depth, TREE_WIDTH, parent_indices_per_layer,
        s_k_hist_raw, s_v_hist_raw);

    const int hist_len = prefix_len + current_depth + 1;
    const int padded_len = ((hist_len + 127) / 128) * 128;

    // Stage 7: grouped query attention
    distribute_q_heads(s_q_rot, q_head_streams);
    broadcast_kv_heads(s_k_hist_raw, s_v_hist_raw, k_head_streams, v_head_streams, hist_len);
    grouped_query_attention(q_head_streams, k_head_streams, v_head_streams, ctx_head_streams, hist_len, padded_len);
    collect_ctx(s_context, ctx_head_streams);

    // Stage 8: output projection
    dense_projection_production_scaled_batched<0, TREE_WIDTH, HIDDEN, HIDDEN, 128, TMAC_USE_TMAC_O>(s_context, s_o_proj, w_o, s_o);

    // Stage 9: post-attn residual + post-attn RMSNorm (returns both residual and normalized stream)
    stream_add<VEC_W>(s_o_proj, s_hidden_residual, s_post_attn_residual, NUM_CHUNKS);
    stream_dup<VEC_W>(s_post_attn_residual, s_post_attn_norm_in, s_post_attn_residual_for_add, NUM_CHUNKS);
    rms_norm_stream<HIDDEN, TREE_WIDTH>(s_post_attn_norm_in, s_post_attn_norm, post_attn_norm_gamma, RMS_EPS);

    // Stage 10: FFN gate/up + SiLU
    stream_dup<VEC_W>(s_post_attn_norm, s_gate_in, s_up_in, NUM_CHUNKS);
    dense_projection_production_scaled_batched<0, TREE_WIDTH, HIDDEN, INTERMEDIATE, 128, TMAC_USE_TMAC_FFN>(s_gate_in, s_gate_vec, w_gate, gate_scales);
    dense_projection_production_scaled_batched<0, TREE_WIDTH, HIDDEN, INTERMEDIATE, 128, TMAC_USE_TMAC_FFN>(s_up_in, s_up_vec, w_up, up_scales);
    silu_mul_stream<VEC_W>(s_gate_vec, s_up_vec, s_swiglu, (TREE_WIDTH * INTERMEDIATE) / VEC_W);

    // Stage 11: down-proj to 2H and split (to_logits, for_reasoning)
    dense_projection_production_scaled_batched<0, TREE_WIDTH, INTERMEDIATE, DOWN_OUTPUT, 128, TMAC_USE_TMAC_FFN>(s_swiglu, s_down_2hs, w_down, down_scales);
    split_down_2hs(s_down_2hs, s_to_logits_raw, s_for_reasoning);

    // Stage 12: final norm on logits stream and residual add on reasoning stream
    rms_norm_stream<HIDDEN, TREE_WIDTH>(s_to_logits_raw, logits_norm_out_stream, final_norm_gamma, RMS_EPS);
    stream_add<VEC_W>(s_for_reasoning, s_post_attn_residual_for_add, reasoning_out_stream, NUM_CHUNKS);
}

} // namespace hls
} // namespace tmac
