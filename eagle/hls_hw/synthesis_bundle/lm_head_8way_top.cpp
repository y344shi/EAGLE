#include "lm_head_8way.hpp"

// Broadcaster: Reads once, writes to 8 streams
void broadcast_hidden(
    //const dtype_in* src,
    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& src,
    hidden_stream_t& s0, hidden_stream_t& s1, hidden_stream_t& s2, hidden_stream_t& s3,
    hidden_stream_t& s4, hidden_stream_t& s5, hidden_stream_t& s6, hidden_stream_t& s7
) {
    #pragma HLS INLINE off
    hls::vector<float, 16> v;

    broadcast_loop:
    for(int i = 0; i < H_DIM * T_BATCH; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=163840 max=163840
        #pragma HLS PIPELINE II=1

        dtype_in val = (dtype_in)0.0;
        
        if (i % T_BATCH == 0) {
            const int h_idx = i / T_BATCH;
            const int v_idx = h_idx % tmac::hls::VEC_W;
            
            if (v_idx == 0) {
                v = src.read();
            }

            val = v[v_idx];
        }
        
        s0.write(val); s1.write(val); s2.write(val); s3.write(val);
        s4.write(val); s5.write(val); s6.write(val); s7.write(val);
    }
}

void global_reduce(TokenOutput out[NUM_ENGINES], TokenOutput& final_out) {
    #pragma HLS INLINE off 

    reduce_batch: for(int t=0; t<T_BATCH; t++) {
        #pragma HLS PIPELINE II=1
        
        float global_best_score = -1e9f;
        int   global_best_id    = -1;

        reduce_engines: for(int e=0; e<NUM_ENGINES; e++) {
            #pragma HLS UNROLL
            float score = out[e].best_score[t];
            int   id    = out[e].best_id[t];

            if (score > global_best_score) {
                global_best_score = score;
                global_best_id    = id;
            }
        }
        final_out.best_score[t] = global_best_score;
        final_out.best_id[t]    = global_best_id;
    }
}

void lm_head_8way_top(
    const wide_vec_t* weights_0,
    const wide_vec_t* weights_1,
    const wide_vec_t* weights_2,
    const wide_vec_t* weights_3,
    const wide_vec_t* weights_4,
    const wide_vec_t* weights_5,
    const wide_vec_t* weights_6,
    const wide_vec_t* weights_7,
    //const dtype_in* hidden_in,
    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& hidden_in,
    TokenOutput& final_output
) {
    #pragma HLS INTERFACE m_axi port=weights_0 bundle=gmem0 depth=100000 latency=64 num_read_outstanding=32
    #pragma HLS INTERFACE m_axi port=weights_1 bundle=gmem1 depth=100000
    #pragma HLS INTERFACE m_axi port=weights_2 bundle=gmem2 depth=100000
    #pragma HLS INTERFACE m_axi port=weights_3 bundle=gmem3 depth=100000
    #pragma HLS INTERFACE m_axi port=weights_4 bundle=gmem4 depth=100000
    #pragma HLS INTERFACE m_axi port=weights_5 bundle=gmem5 depth=100000
    #pragma HLS INTERFACE m_axi port=weights_6 bundle=gmem6 depth=100000
    #pragma HLS INTERFACE m_axi port=weights_7 bundle=gmem7 depth=100000
    
    #pragma HLS INTERFACE m_axi port=hidden_in bundle=gmem_h offset=slave depth=163840
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS AGGREGATE variable=final_output compact=auto

    #pragma HLS DATAFLOW

    // Streams for Broadcasting
    hidden_stream_t s_h[8];
    #pragma HLS STREAM variable=s_h depth=128
    #pragma HLS ARRAY_PARTITION variable=s_h complete dim=1

    // PIPO buffer for results (No 'static' to allow dataflow)
    TokenOutput partial_results[8];
    #pragma HLS ARRAY_PARTITION variable=partial_results complete dim=1

    // 1. Broadcast Hidden State
    broadcast_hidden(hidden_in, s_h[0], s_h[1], s_h[2], s_h[3], s_h[4], s_h[5], s_h[6], s_h[7]);

    // 2. Parallel Engines
    lm_head_engine_stream(weights_0, s_h[0], partial_results[0], VOCAB_SLICE, 0);
    lm_head_engine_stream(weights_1, s_h[1], partial_results[1], VOCAB_SLICE, 1);
   // lm_head_engine_stream(weights_2, s_h[2], partial_results[2], VOCAB_SLICE, 2);
   // lm_head_engine_stream(weights_3, s_h[3], partial_results[3], VOCAB_SLICE, 3);
   // lm_head_engine_stream(weights_4, s_h[4], partial_results[4], VOCAB_SLICE, 4);
   // lm_head_engine_stream(weights_5, s_h[5], partial_results[5], VOCAB_SLICE, 5);
   // lm_head_engine_stream(weights_6, s_h[6], partial_results[6], VOCAB_SLICE, 6);
   // lm_head_engine_stream(weights_7, s_h[7], partial_results[7], VOCAB_SLICE, 7);

    // 3. Reduction
    global_reduce(partial_results, final_output);
}
