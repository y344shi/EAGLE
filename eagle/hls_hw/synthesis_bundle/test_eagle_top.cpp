#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// 1. Include the Top Level Header
#include "eagle_tier1_lm_top.hpp"

// 2. Include implementation headers for the SW Reference model
// (These are needed if the testbench re-implements the logic for verification)
#include "rms_norm_stream.hpp"
#include "deep_pipeline_lutmac.hpp"
#include "attention_solver.hpp"
#include "kv_cache_manager.hpp"
#include "stream_utils.hpp"
#include "rope_kernel.hpp"

// 3. Use the namespaces to fix "Undeclared Identifier" errors
using namespace tmac;
using namespace tmac::hls;

// Helper to print debug info
void print_vec(const std::string& name, const vec_t<VEC_W>& v) {
    std::cout << name << ": [";
    for(int i=0; i<VEC_W; i++) std::cout << (float)v[i] << " ";
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << ">>> Starting EAGLE Tier 1 + LM Head Testbench" << std::endl;

    // ====================================================
    // 1. Setup Input Data
    // ====================================================
    int seq_len = 10;
    int current_length = 5;
    
    // Input Stream (Random Token Embedding)
    hls::stream<vec_t<VEC_W>> in_stream("tb_in_stream");
    
    // Fill with random data
    for (int i = 0; i < HIDDEN / VEC_W; i++) {
        vec_t<VEC_W> vec;
        for (int j = 0; j < VEC_W; j++) {
            vec[j] = (half)(0.01f * (i * VEC_W + j)); // Simple pattern
        }
        in_stream.write(vec);
    }

    // ====================================================
    // 2. Allocate Dummy Weights (On Heap to avoid stack overflow)
    // ====================================================
    // We use vectors to manage memory, then pass .data() pointers
    std::vector<pack512> w_q(4096 * HIDDEN / 512 * 4); // Approx size
    std::vector<float>   s_q(4096 * HIDDEN / 128);

    // Initialize with zeros/ones to avoid NaNs
    std::fill(w_q.begin(), w_q.end(), pack512{0});
    std::fill(s_q.begin(), s_q.end(), 1.0f);

    // Reuse these pointers for all weights to save simulation memory/setup time
    // (In a real test, you'd load distinct weights)
    auto* w_ptr = w_q.data();
    auto* s_ptr = s_q.data();

    // HBM KV Cache Buffers
    std::vector<vec_t<VEC_W>> hbm_k(16384);
    std::vector<vec_t<VEC_W>> hbm_v(16384);

    // LM Head Weights (8 Banks)
    // Size: 73448/8 * 4096/32 vectors approx
    int lm_slice_vecs = (73448 / 8) * (4096 / 32);
    std::vector<wide_vec_t> lm_w(lm_slice_vecs); 

    // ====================================================
    // 3. Output Variables
    // ====================================================
    int best_id = -1;
    float best_score = -1.0f;
    
    // RoPE Config
    RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg;

    // ====================================================
    // 4. Run the Top Function
    // ====================================================
    std::cout << "    Running Hardware Kernel..." << std::endl;
    
    eagle_tier1_lm_top(
        in_stream,
        &best_id,
        &best_score,
        // Tier 1 Weights (Sharing pointers for simplicity in TB)
        w_ptr, s_ptr, // Q
        w_ptr, s_ptr, // K
        w_ptr, s_ptr, // V
        w_ptr, s_ptr, // O
        w_ptr, s_ptr, // Gate
        w_ptr, s_ptr, // Up
        w_ptr, s_ptr, // Down
        s_ptr, // Norm1
        s_ptr, // Norm2
        rope_cfg,
        hbm_k.data(),
        hbm_v.data(),
        // LM Head Weights (8 Banks - Sharing pointers)
        lm_w.data(), lm_w.data(), lm_w.data(), lm_w.data(),
        lm_w.data(), lm_w.data(), lm_w.data(), lm_w.data(),
        seq_len,
        current_length
    );

    // ====================================================
    // 5. Report Results
    // ====================================================
    std::cout << ">>> Execution Complete." << std::endl;
    std::cout << "    Best Token ID: " << best_id << std::endl;
    std::cout << "    Best Score:    " << best_score << std::endl;

    // Since inputs were dummy/zero, we expect ID 0 and Score 0 (or close to it)
    if (best_id >= 0) {
        std::cout << ">>> TEST PASSED (Kernel ran without hanging)" << std::endl;
        return 0;
    } else {
        std::cout << ">>> TEST FAILED (Invalid Output)" << std::endl;
        return 1;
    }
}