#ifndef LM_HEAD_TOP_HPP
#define LM_HEAD_TOP_HPP

#include "hls_stream.h"
#include "hls_half.h"
#include <cmath>

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================
// Batch Size (Fixed per design doc)
#define T_BATCH 40 

// Engine Dimensions
#define R_ROWS  32  // Vocab rows processed per engine cycle
#define H_DIM   4096 // Hidden dimension size

// Blocking Factors (Tiling)
#define K_TILE  512 // Inner dimension tile size

// Precision
typedef half   dtype_in;  // FP16 for Weights & Hidden
typedef float  dtype_acc; // FP32 for Accumulation

// Data Structures
struct ComputeConfig {
    int vocab_start_idx;
    int vocab_end_idx;
};

struct TokenOutput {
    int   best_id[T_BATCH];
    float best_score[T_BATCH];
};

// Vector types for wide bus access (512-bit / 256-bit)
// Assuming 512-bit AXI width for Weights -> 32 halves per beat
struct wide_vec_t {
    dtype_in data[32];
};

// ============================================================================
// FUNCTION PROTOTYPE
// ============================================================================
void lm_head_engine(
    const wide_vec_t* weights_in, // Stream from HBM
    const dtype_in* hidden_in,    // On-chip BRAM/URAM (4096 x 40)
    TokenOutput& output,          // Result
    int vocab_size_slice          // How many rows this engine handles
);

#endif