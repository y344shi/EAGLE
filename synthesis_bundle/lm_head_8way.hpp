#ifndef LM_HEAD_8WAY_HPP
#define LM_HEAD_8WAY_HPP

//#include "hls_stream.h"
#include "tmac_utils.hpp"
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
#define NUM_ENGINES 4
#define TOTAL_VOCAB 73448 / 2
#define VOCAB_SLICE ((TOTAL_VOCAB + NUM_ENGINES - 1) / NUM_ENGINES) 
#define V_PRELOAD_TOTAL 8192
#define V_PRELOAD_PER_ENGINE (V_PRELOAD_TOTAL / NUM_ENGINES)
#define V_PRELOAD_TILES ((V_PRELOAD_PER_ENGINE + R_ROWS - 1) / R_ROWS)

// ============================================================================
// FUNCTION PROTOTYPES
// ============================================================================

// The Sub-Engine (Now accepts stream)
void lm_head_engine_stream(
    const wide_vec_t* weights_in, 
    hidden_stream_t& hidden_stream,    
    TokenOutput& output,
    int vocab_size_slice,
    int engine_id
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
    //const dtype_in* hidden_in,
    hls::stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& hidden_in,
    TokenOutput& final_output
);

#endif
