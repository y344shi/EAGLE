#include "eagle_tier1_lm_top.hpp"

// Super-wrapper: Tier1 transformer -> 8-way LM head (single token, batch slot 0).
void eagle_tier1_lm_top(hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& in_stream,
                        int* best_id,
                        float* best_score,
                        const tmac::hls::pack512* w_q,     const float* s_q,
                        const tmac::hls::pack512* w_k,     const float* s_k,
                        const tmac::hls::pack512* w_v,     const float* s_v,
                        const tmac::hls::pack512* w_o,     const float* s_o,
                        const tmac::hls::pack512* w_gate,  const float* gate_scales,
                        const tmac::hls::pack512* w_up,    const float* up_scales,
                        const tmac::hls::pack512* w_down,  const float* down_scales,
                        const float* norm1_gamma,
                        const float* norm2_gamma,
                        const tmac::hls::RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>& rope_cfg,
                        tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_k,
                        tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_v,
                        const wide_vec_t* lm_w0,
                        const wide_vec_t* lm_w1,
                        const wide_vec_t* lm_w2,
                        const wide_vec_t* lm_w3,
                        const wide_vec_t* lm_w4,
                        const wide_vec_t* lm_w5,
                        const wide_vec_t* lm_w6,
                        const wide_vec_t* lm_w7,
                        int seq_len,
                        int current_length) {
#pragma HLS INTERFACE axis port=in_stream
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
#pragma HLS INTERFACE m_axi port=lm_w0 bundle=gmem_lm0 depth=1200000
#pragma HLS INTERFACE m_axi port=lm_w1 bundle=gmem_lm1 depth=1200000
#pragma HLS INTERFACE m_axi port=lm_w2 bundle=gmem_lm2 depth=1200000
#pragma HLS INTERFACE m_axi port=lm_w3 bundle=gmem_lm3 depth=1200000
#pragma HLS INTERFACE m_axi port=lm_w4 bundle=gmem_lm4 depth=1200000
#pragma HLS INTERFACE m_axi port=lm_w5 bundle=gmem_lm5 depth=1200000
#pragma HLS INTERFACE m_axi port=lm_w6 bundle=gmem_lm6 depth=1200000
#pragma HLS INTERFACE m_axi port=lm_w7 bundle=gmem_lm7 depth=1200000
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=current_length bundle=control
#pragma HLS INTERFACE s_axilite port=best_id bundle=control
#pragma HLS INTERFACE s_axilite port=best_score bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    // Tier1 block
    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>> block_out("block_out");
#pragma HLS STREAM variable=block_out depth=64
    eagle_tier1_top(in_stream, block_out, w_q, s_q, w_k, s_k, w_v, s_v, w_o, s_o,
                    w_gate, gate_scales, w_up, up_scales, w_down, down_scales,
                    norm1_gamma, norm2_gamma, rope_cfg, hbm_k, hbm_v, seq_len, current_length);

    TokenOutput lm_result{};
    lm_head_8way_top(lm_w0, lm_w1, lm_w2, lm_w3, lm_w4, lm_w5, lm_w6, lm_w7,
                     block_out, lm_result);

    *best_id = lm_result.best_id[0];
    *best_score = lm_result.best_score[0];
}
