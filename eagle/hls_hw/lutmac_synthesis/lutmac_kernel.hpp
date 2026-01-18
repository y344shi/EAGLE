// tmac_hls.h
#ifndef LUTMAC_KERNEL_HLS_H
#define LUTMAC_KERNEL_HLS_H

#include <ap_int.h>
#include "../../../projects/EAGLE/eagle/hls_hw/synthesis_bundle/deep_pipeline_lutmac.hpp"

namespace tmac {

namespace hls {
#define DEFAULT_ABITS 8
#define DEFAULT_WBITS 4
#define DEFAULT_SCALE_EXP 0
#define HIDDEN_DIM 4096
#define DEFAULT_OUT_W 128

// loads one complete weight tile from HBM into on-chip BRAM
template<int INPUT_DIM>
void load_weight_tile(const pack512* W_dram, pack512 w_bram_buffer[INPUT_DIM], int tile_idx) {
load_w_loop:
    for (int i = 0; i < INPUT_DIM; ++i) {
#pragma HLS PIPELINE II=1
        w_bram_buffer[i] = W_dram[tile_idx * INPUT_DIM + i];
    }
}


// stream the complete activation vector from HBM for the current tile
template <int INPUT_DIM>
void stream_activations_for_tile(
    const ap_uint<DEFAULT_ABITS>* A_dram,
    hls_stream<vec_t<VEC_W>>& a_stream,
    int row_idx) {

    // stream ith chunk of the current tile
stream_activations:
    for (int i = 0; i < INPUT_DIM / VEC_W; ++i) {
        vec_t<VEC_W> a_chunk{};
        for (int j = 0; j < VEC_W; ++j) {
// #pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
            a_chunk[j] = static_cast<float>(A_dram[row_idx * INPUT_DIM + i * VEC_W + j]);
        }
        a_stream.write(a_chunk);
    }
}

// reads results from c_stream and writes them to the correct HBM offset
template <int OUT_DIM, int OUT_W>
void store_fc1_results(
    hls_stream<vec_t<VEC_W>>& c_stream,
    ap_int<32>* Out_dram,
    int tile_idx,
    int row_idx) {

    // Calculate the memory offset for the current output tile
    ap_int<32>* out_tile_ptr = Out_dram + row_idx * OUT_DIM + tile_idx * OUT_W;
    const int out_chunks_per_tile = OUT_W / VEC_W;

store_chunks:
    for (int oc = 0; oc < out_chunks_per_tile; ++oc) {
        vec_t<VEC_W> out_chunk = c_stream.read();
        for (int j = 0; j < VEC_W; ++j) {
// #pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
            const int lane_addr = oc * VEC_W + j;
            out_tile_ptr[lane_addr] = static_cast<ap_int<32>>(out_chunk[j]);
        }
    }
}

extern "C" {
void lutmac_fc1(
    const ap_uint<DEFAULT_ABITS> *A_dram, // pointer to A in DRAM (SEQ_LEN x INPUT_DIM)
    const ap_int<DEFAULT_WBITS>  *W_dram, // pointer to W in DRAM (INPUT_DIM x OUT_DIM)
    ap_int<32>                   *Out_dram, // pointer to output in DRAM (SEQ_LEN x OUT_DIM)
    unsigned                     seq_len
) {
#pragma HLS INTERFACE m_axi port=A_dram offset=slave bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi port=W_dram offset=slave bundle=gmem1 depth=1024
#pragma HLS INTERFACE m_axi port=Out_dram offset=slave bundle=gmem2 depth=1024
#pragma HLS INTERFACE s_axilite port=A_dram bundle=control
#pragma HLS INTERFACE s_axilite port=W_dram bundle=control
#pragma HLS INTERFACE s_axilite port=Out_dram bundle=control
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static_assert(HIDDEN_DIM % DEFAULT_OUT_W == 0, "OUT_DIM must be a multiple of tile width OUT_W");
    constexpr int TILES = HIDDEN_DIM / DEFAULT_OUT_W;
    const pack512* w_dram_packed = reinterpret_cast<const pack512*>(W_dram);
    pack512 w_bram_buffer[HIDDEN_DIM];

tile_loop:
    for (int t = 0; t < TILES; ++t) {
        load_weight_tile<HIDDEN_DIM>(w_dram_packed, w_bram_buffer, t);

    row_loop:
        for (int i = 0; i < seq_len; ++i) {
            // These streams are FIFOs that connect the three hardware stages for a SINGLE tile.
            // They are re-used in each iteration of the loop.
            hls_stream<vec_t<VEC_W>> a_stream;
            hls_stream<vec_t<VEC_W>> c_stream;
        #pragma HLS STREAM variable=a_stream depth=64
        #pragma HLS STREAM variable=c_stream depth=64

            // a three-stage pipeline for processing one tile
        #pragma HLS DATAFLOW
            stream_activations_for_tile<HIDDEN_DIM>(A_dram, a_stream, i);
            dense_projection_production<DEFAULT_SCALE_EXP, HIDDEN_DIM, DEFAULT_OUT_W>(a_stream, c_stream, w_bram_buffer);
            store_fc1_results<HIDDEN_DIM, DEFAULT_OUT_W>(c_stream, Out_dram, t, i);
        }
    }
}
}
}
}
#endif // LUTMAC_KERNEL_HLS_H
