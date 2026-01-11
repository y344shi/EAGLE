#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "lm_head_8way.hpp"

half float_to_half(float f) { return (half)f; }

int main() {
    std::cout << ">>> Starting 8-Way LM Head Testbench (Stream Version)" << std::endl;

    const int num_row_tiles = VOCAB_SLICE / R_ROWS;
    const int vocab_per_engine = num_row_tiles * R_ROWS;
    const int total_vocab_effective = vocab_per_engine * NUM_ENGINES;

    std::vector<wide_vec_t> weights_hbm[NUM_ENGINES];
    for(int i=0; i<NUM_ENGINES; i++) {
        weights_hbm[i].resize(num_row_tiles * H_DIM);
    }

    std::vector<dtype_in> hidden_host(H_DIM * T_BATCH);
    TokenOutput hls_output, cpu_output;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::cout << "    Initializing Data..." << std::endl;
    for(int i=0; i<H_DIM * T_BATCH; i++) hidden_host[i] = float_to_half(dist(gen));

    std::vector<float> cpu_logits(total_vocab_effective * T_BATCH, 0.0f);

    for(int v = 0; v < total_vocab_effective; v++) {
        int engine_idx = v / vocab_per_engine;
        int local_row  = v % vocab_per_engine;
        int r_tile     = local_row / R_ROWS;
        int r          = local_row % R_ROWS;
        for(int k = 0; k < H_DIM; k++) {
            float w_val = dist(gen);
            weights_hbm[engine_idx][r_tile * H_DIM + k].data[r] = float_to_half(w_val);

            for(int t = 0; t < T_BATCH; t++) {
                cpu_logits[v * T_BATCH + t] += w_val * (float)hidden_host[k * T_BATCH + t];
            }
        }
    }

    std::cout << "    Computing Golden Ref..." << std::endl;
    for(int t = 0; t < T_BATCH; t++) {
        float max_val = -1e9;
        int max_id = -1;
        for(int v = 0; v < total_vocab_effective; v++) {
            if(cpu_logits[v * T_BATCH + t] > max_val) {
                max_val = cpu_logits[v * T_BATCH + t];
                max_id = v;
            }
        }
        cpu_output.best_id[t] = max_id;
        cpu_output.best_score[t] = max_val;
    }

    std::cout << "    Running HLS Top Level..." << std::endl;
    lm_head_8way_top(
        weights_hbm[0].data(), weights_hbm[1].data(), weights_hbm[2].data(), weights_hbm[3].data(),
        weights_hbm[4].data(), weights_hbm[5].data(), weights_hbm[6].data(), weights_hbm[7].data(),
        hidden_host.data(), hls_output
    );

    int errors = 0;
    for(int t = 0; t < T_BATCH; t++) {
        if (hls_output.best_id[t] != cpu_output.best_id[t]) {
            if (std::abs(hls_output.best_score[t] - cpu_output.best_score[t]) > 0.05) errors++;
        }
    }

    if(errors == 0) std::cout << ">>> TEST PASSED!" << std::endl;
    else std::cout << ">>> TEST FAILED with " << errors << " errors." << std::endl;

    return (errors == 0) ? 0 : 1;
}
