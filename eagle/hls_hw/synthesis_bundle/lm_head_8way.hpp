#ifndef LM_HEAD_8WAY_HPP
#define LM_HEAD_8WAY_HPP

#include "hls_stream.h"
#include "hls_half.h"
#include <cmath>

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================
#define T_BATCH 40 
#define R_ROWS  32  // Vocab rows processed per engine cycle
#define H_DIM   4096 
#define K_TILE  512 

// Precision
typedef half   dtype_in;  
typedef float  dtype_acc; 

// Data Structures
struct TokenOutput {
    int   best_id[T_BATCH];
    float best_score[T_BATCH];
};

struct wide_vec_t {
    dtype_in data[32];
};

// Stream Definition for Broadcaster
typedef hls::stream<dtype_in> hidden_stream_t;

// Engine Constants
#define NUM_ENGINES 8
#define TOTAL_VOCAB 73448
#define VOCAB_SLICE ((TOTAL_VOCAB + NUM_ENGINES - 1) / NUM_ENGINES) 

// ============================================================================
// FUNCTION PROTOTYPES
// ============================================================================

// The Sub-Engine (Now accepts stream)
void lm_head_engine_stream(
    const wide_vec_t* weights_in, 
    hidden_stream_t& hidden_stream,    
    TokenOutput& output,
    int vocab_size_slice
);

// The Top Level
void lm_head_8way_top(
    const wide_vec_t* weights_0,
    const wide_vec_t* weights_1,
    const wide_vec_t* weights_2,
    const wide_vec_t* weights_3,
    const wide_vec_t* weights_4,
    const wide_vec_t* weights_5,
    const wide_vec_t* weights_6,
    const wide_vec_t* weights_7,
    const dtype_in* hidden_in,
    TokenOutput& final_output
);

#endif