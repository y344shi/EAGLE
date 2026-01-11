#include "lm_head_8way.hpp"

// Single Compute Engine (Stream Based)
void lm_head_engine_stream(
    const wide_vec_t* weights_in, 
    hidden_stream_t& hidden_stream,    
    TokenOutput& output,
    int vocab_size_slice
) {
    // Latency optimized AXI settings for high-bandwidth streaming
    #pragma HLS INTERFACE m_axi port=weights_in bundle=gmem0 offset=slave depth=100000 latency=64 num_read_outstanding=32 max_read_burst_length=32
    
    // Internal Stream Interface (FIFO)
    #pragma HLS INTERFACE ap_fifo port=hidden_stream
    
    #pragma HLS INTERFACE s_axilite port=vocab_size_slice bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS AGGREGATE variable=output compact=auto

    // Local URAM Buffer
    static dtype_in hidden_buf[H_DIM][T_BATCH];
    #pragma HLS BIND_STORAGE variable=hidden_buf type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=hidden_buf complete dim=2 

    // 1. LOAD HIDDEN STATES (From Stream)
    // This consumes the broadcasted data immediately
    load_h_outer: for(int k = 0; k < H_DIM; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
        
        load_h_inner: for(int t = 0; t < T_BATCH; t++) {
            #pragma HLS LOOP_TRIPCOUNT min=40 max=40
            #pragma HLS PIPELINE II=1
            // Pop from stream
            hidden_buf[k][t] = hidden_stream.read();
        }
    }

    // 2. INIT ACCUMULATORS
    dtype_acc best_scores[T_BATCH];
    int       best_ids[T_BATCH];
    #pragma HLS ARRAY_PARTITION variable=best_scores complete
    #pragma HLS ARRAY_PARTITION variable=best_ids complete
    
    init_out: for(int t=0; t<T_BATCH; t++) {
        #pragma HLS UNROLL
        best_scores[t] = -1e9f; 
        best_ids[t] = -1;
    }

    // 3. COMPUTE ENGINE
    int num_row_tiles = vocab_size_slice / R_ROWS;

    vocab_loop: for(int r_tile = 0; r_tile < num_row_tiles; r_tile++) {
        #pragma HLS LOOP_TRIPCOUNT min=100 max=2300

        dtype_acc acc[R_ROWS][T_BATCH];
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=0 

        reset_acc: for(int r=0; r<R_ROWS; r++) {
            #pragma HLS UNROLL
            for(int t=0; t<T_BATCH; t++) {
                #pragma HLS UNROLL
                acc[r][t] = 0;
            }
        }

        k_loop: for(int k = 0; k < H_DIM; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
            
            wide_vec_t w_vec = weights_in[r_tile * H_DIM + k];

            compute_core: for(int r=0; r<R_ROWS; r++) {
                #pragma HLS UNROLL
                dtype_in w_val = w_vec.data[r];
                
                for(int t=0; t<T_BATCH; t++) {
                    #pragma HLS UNROLL
                    dtype_in h_val = hidden_buf[k][t];
                    // IMPORTANT: Ensure 'unsafe_math_optimizations' is ON for II=1
                    acc[r][t] += (dtype_acc)w_val * (dtype_acc)h_val;
                }
            }
        }

        update_top1: for(int r=0; r<R_ROWS; r++) {
            #pragma HLS PIPELINE II=1
            int current_vocab_id = (r_tile * R_ROWS) + r;
            
            for(int t=0; t<T_BATCH; t++) {
                if(acc[r][t] > best_scores[t]) {
                    best_scores[t] = acc[r][t];
                    best_ids[t]    = current_vocab_id;
                }
            }
        }
    }

    // 4. WRITE BACK
    write_out: for(int t=0; t<T_BATCH; t++) {
        #pragma HLS PIPELINE II=1
        output.best_id[t]    = best_ids[t];
        output.best_score[t] = best_scores[t];
    }
}