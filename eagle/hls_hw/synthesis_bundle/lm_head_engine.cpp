#include "lm_head_8way.hpp"

// Single Compute Engine (Stream Based)
void lm_head_engine_stream(
    const wide_vec_t* weights_in, 
    hidden_stream_t& hidden_stream,    
    TokenOutput& output,
    int vocab_size_slice,
    int engine_id
) {
    #pragma HLS INLINE off
    // Stream is already a FIFO; keep it as a channel for dataflow.
    #pragma HLS INTERFACE ap_fifo port=hidden_stream

    // Shared weight cache (preloaded once per engine instance)
    static dtype_in weight_cache[NUM_ENGINES][R_ROWS][V_PRELOAD_TILES * H_DIM];
    static bool weight_cache_valid[NUM_ENGINES];
    #pragma HLS BIND_STORAGE variable=weight_cache type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=weight_cache complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight_cache complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight_cache_valid complete dim=1

    // Local URAM Buffer (per-instance)
    dtype_in hidden_buf[H_DIM][T_BATCH];
    #pragma HLS BIND_STORAGE variable=hidden_buf type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=hidden_buf complete dim=2 

    // 0. PRELOAD WEIGHT CACHE (one-time per engine instance)
    if (!weight_cache_valid[engine_id]) {
        preload_rtile: for (int r_tile = 0; r_tile < V_PRELOAD_TILES; r_tile++) {
            #pragma HLS LOOP_TRIPCOUNT min=32 max=32
            preload_k: for (int k = 0; k < H_DIM; k++) {
                #pragma HLS PIPELINE II=1
                wide_vec_t w_vec = weights_in[r_tile * H_DIM + k];
                for (int r = 0; r < R_ROWS; r++) {
                    #pragma HLS UNROLL
                    weight_cache[engine_id][r][r_tile * H_DIM + k] = w_vec.data[r];
                }
            }
        }
        weight_cache_valid[engine_id] = true;
    }

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

    // 3. COMPUTE ENGINE (Output-stationary, K-tiled)
    int num_row_tiles = vocab_size_slice / R_ROWS;
    int num_k_tiles   = H_DIM / K_TILE;

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

        dtype_in weight_tile[2][R_ROWS][K_TILE];
        #pragma HLS BIND_STORAGE variable=weight_tile type=ram_2p impl=bram
        #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2

        if (num_k_tiles > 0) {
            load_w_init: for (int k = 0; k < K_TILE; k++) {
                #pragma HLS PIPELINE II=1
                int k_global = k;
                if (r_tile < V_PRELOAD_TILES) {
                    for (int r = 0; r < R_ROWS; r++) {
                        #pragma HLS UNROLL
                        weight_tile[0][r][k] = weight_cache[engine_id][r][r_tile * H_DIM + k_global];
                    }
                } else {
                    wide_vec_t w_vec = weights_in[r_tile * H_DIM + k_global];
                    for (int r = 0; r < R_ROWS; r++) {
                        #pragma HLS UNROLL
                        weight_tile[0][r][k] = w_vec.data[r];
                    }
                }
            }
        }

        k_tile_loop: for (int kt = 0; kt < num_k_tiles; kt++) {
            #pragma HLS LOOP_TRIPCOUNT min=8 max=8
            int buf = kt & 1;
            int next = buf ^ 1;

            if (kt + 1 < num_k_tiles) {
                load_w_next: for (int k = 0; k < K_TILE; k++) {
                    #pragma HLS PIPELINE II=1
                    int k_global = (kt + 1) * K_TILE + k;
                    if (r_tile < V_PRELOAD_TILES) {
                        for (int r = 0; r < R_ROWS; r++) {
                            #pragma HLS UNROLL
                            weight_tile[next][r][k] = weight_cache[engine_id][r][r_tile * H_DIM + k_global];
                        }
                    } else {
                        wide_vec_t w_vec = weights_in[r_tile * H_DIM + k_global];
                        for (int r = 0; r < R_ROWS; r++) {
                            #pragma HLS UNROLL
                            weight_tile[next][r][k] = w_vec.data[r];
                        }
                    }
                }
            }

            compute_k: for(int k = 0; k < K_TILE; k++) {
                #pragma HLS PIPELINE II=1
                for(int r=0; r<R_ROWS; r++) {
                    #pragma HLS UNROLL
                    dtype_in w_val = weight_tile[buf][r][k];
                    for(int t=0; t<T_BATCH; t++) {
                        #pragma HLS UNROLL
                        dtype_in h_val = hidden_buf[kt * K_TILE + k][t];
                        // IMPORTANT: Ensure 'unsafe_math_optimizations' is ON for II=1
                        acc[r][t] += (dtype_acc)w_val * (dtype_acc)h_val;
                    }
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
