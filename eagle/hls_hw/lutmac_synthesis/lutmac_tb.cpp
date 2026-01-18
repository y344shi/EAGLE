#define TMAC_ENABLE_LUTMAC_TB 1
#if defined(TMAC_ENABLE_LUTMAC_TB)
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "lutmac_kernel.hpp"

using namespace tmac::hls;

// CPU lutmac simulation
void cpu_golden_lutmac_emulation(
    const std::vector<ap_uint<DEFAULT_ABITS>>& A,
    const std::vector<pack512>& W_packed_tiles,
    std::vector<ap_int<32>>& Y_golden,
    unsigned seq_len,
    unsigned input_dim,
    unsigned output_dim,
    unsigned out_w)
{
    unsigned tiles = output_dim / out_w;
    Y_golden.assign(seq_len * output_dim, 0);

    for (unsigned t = 0; t < tiles; ++t) {
        const pack512* w_tile_ptr = W_packed_tiles.data() + t * input_dim;
        for (unsigned r = 0; r < seq_len; ++r) {
            vec_t<DEFAULT_OUT_W> acc_vec{};
            for(int i = 0; i < DEFAULT_OUT_W; ++i) acc_vec[i] = 0.0f;

            for (unsigned k = 0; k < input_dim; ++k) {
                float a_scalar = static_cast<float>(A[r * input_dim + k]);
                const pack512& w_pkt = w_tile_ptr[k];
                lut_mac_broadcast<DEFAULT_SCALE_EXP, DEFAULT_OUT_W>(a_scalar, w_pkt, acc_vec);
            }

            for (unsigned j = 0; j < out_w; ++j) {
                Y_golden[r * output_dim + (t * out_w + j)] = static_cast<ap_int<32>>(acc_vec[j]);
            }
        }
    }
}


// Weight Packing Helper Function
void pack_weights_for_hw(
    const std::vector<ap_int<DEFAULT_WBITS>>& W_naive,
    std::vector<pack512>& W_packed_tiles,
    unsigned input_dim,
    unsigned output_dim,
    unsigned out_w)
{
    unsigned tiles = output_dim / out_w;
    W_packed_tiles.resize(tiles * input_dim);

    for (unsigned t = 0; t < tiles; ++t) {
        for (unsigned k = 0; k < input_dim; ++k) {
            pack512 current_packet{};
            for (unsigned j = 0; j < out_w; ++j) {
                unsigned src_idx = k * output_dim + (t * out_w + j);
                ap_int<DEFAULT_WBITS> w_signed = W_naive[src_idx];
                uint8_t w_raw = static_cast<uint8_t>(w_signed + 8);

                const int byte_idx = j >> 1;
                const bool high_nibble = j & 1;
                if (high_nibble) {
                    current_packet.bytes[byte_idx] |= (w_raw << 4);
                } else {
                    current_packet.bytes[byte_idx] |= w_raw;
                }
            }
            W_packed_tiles[t * input_dim + k] = current_packet;
        }
    }
}

int main() {
    // 1. Define Test Dimensions
    const unsigned SEQ_LEN = 32;
    const unsigned INPUT_DIM = HIDDEN_DIM;
    const unsigned OUTPUT_DIM = HIDDEN_DIM;
    const unsigned OUT_W = DEFAULT_OUT_W;

    std::printf("--- LUT-MAC Kernel Test Bench (Bit-Accurate Emulation) ---\n");
    std::printf("Dimensions: A(%u x %u) * W(%u x %u)\n", SEQ_LEN, INPUT_DIM, INPUT_DIM, OUTPUT_DIM);

    // 2. Allocate Memory
    std::vector<ap_uint<DEFAULT_ABITS>> A_matrix(SEQ_LEN * INPUT_DIM);
    std::vector<ap_int<DEFAULT_WBITS>> W_matrix_naive(INPUT_DIM * OUTPUT_DIM);
    std::vector<pack512> W_packed_tiles;
    std::vector<ap_int<32>> Y_golden(SEQ_LEN * OUTPUT_DIM);
    std::vector<ap_int<32>> Y_hw(SEQ_LEN * OUTPUT_DIM);

    // 3. Initialize Input Data
    srand(123);
    for (size_t i = 0; i < A_matrix.size(); ++i) A_matrix[i] = rand() % 256;
    for (size_t i = 0; i < W_matrix_naive.size(); ++i) W_matrix_naive[i] = (rand() % 16) - 8;

    // 4. Prepare Weights
    std::printf("Packing and encoding weights...\n");
    pack_weights_for_hw(W_matrix_naive, W_packed_tiles, INPUT_DIM, OUTPUT_DIM, OUT_W);

    // 5. Run Golden CPU Emulation
    std::printf("Running bit-accurate CPU emulation...\n");
    cpu_golden_lutmac_emulation(A_matrix, W_packed_tiles, Y_golden, SEQ_LEN, INPUT_DIM, OUTPUT_DIM, OUT_W);

    // 6. Run the HLS Kernel
    std::printf("Running HLS kernel 'lutmac_fc1'...\n");
    lutmac_fc1(
        A_matrix.data(),
        reinterpret_cast<const ap_int<DEFAULT_WBITS>*>(W_packed_tiles.data()),
        Y_hw.data(),
        SEQ_LEN
    );

    // 7. Verify the Results
    std::printf("Verifying results...\n");
    int errors = 0;
    for (size_t i = 0; i < Y_golden.size(); ++i) {
        if (Y_hw[i] != Y_golden[i]) {
            if (errors < 10) {
                std::printf("Mismatch at index %llu:  Hardware output = %d, Golden output = %d\n",
                            (unsigned long long)i,
                            (int)Y_hw[i],
                            (int)Y_golden[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::printf("\n[PASS] All %llu output values matched the bit-accurate golden reference.\n",
                    (unsigned long long)Y_golden.size());
        return 0;
    } else {
        std::printf("\n[FAIL] Found %d mismatches.\n", errors);
        return 1;
    }
}
#endif