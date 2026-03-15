#ifndef TMAC_EAGLE4_PARALLEL_CONFIG_HPP
#define TMAC_EAGLE4_PARALLEL_CONFIG_HPP

// Centralized parallel-degree knobs for EAGLE4 HLS kernels.
// Override these from syn.cflags (for example: -DE4D_PAR_W2B_UNROLL=2).

// INT2 batched projection (dense_projection_production_scaled_batched_w2).
#ifndef E4D_PAR_W2B_UNROLL
#define E4D_PAR_W2B_UNROLL 1
#endif

// Attention fanout parallelism controls (major LUT lever).
// Defaults preserve moderate throughput while reducing full replication.
#ifndef E4D_PAR_GQA_TOKEN_UNROLL
#define E4D_PAR_GQA_TOKEN_UNROLL 1
#endif

#ifndef E4D_PAR_GQA_HEAD_UNROLL
#define E4D_PAR_GQA_HEAD_UNROLL 1
#endif

#ifndef E4D_PAR_KV_BCAST_TOKEN_UNROLL
#define E4D_PAR_KV_BCAST_TOKEN_UNROLL 1
#endif

#ifndef E4D_PAR_KV_DUP_HEAD_UNROLL
#define E4D_PAR_KV_DUP_HEAD_UNROLL 1
#endif

// LM-head candidate dequantization.
#ifndef E4D_PAR_LM_DEQUANT_PACK_UNROLL
#define E4D_PAR_LM_DEQUANT_PACK_UNROLL 1
#endif

#ifndef E4D_PAR_LM_DEQUANT_ELEM_UNROLL
#define E4D_PAR_LM_DEQUANT_ELEM_UNROLL 1
#endif

// LM-head candidate dot product.
#ifndef E4D_PAR_LM_DOT_UNROLL
#define E4D_PAR_LM_DOT_UNROLL 1
#endif

// LM-head softmax / best-candidate scans.
#ifndef E4D_PAR_LM_SOFTMAX_UNROLL
#define E4D_PAR_LM_SOFTMAX_UNROLL 1
#endif

// INT2 batched projection inner-kernel controls (major LUT/DSP lever).
#ifndef E4D_PAR_W2B_NUM_BANKS
#define E4D_PAR_W2B_NUM_BANKS 2
#endif

#ifndef E4D_PAR_W2B_COMPUTE_II
#define E4D_PAR_W2B_COMPUTE_II 4
#endif

static_assert(E4D_PAR_W2B_UNROLL > 0, "E4D_PAR_W2B_UNROLL must be > 0");
static_assert(E4D_PAR_GQA_TOKEN_UNROLL > 0, "E4D_PAR_GQA_TOKEN_UNROLL must be > 0");
static_assert(E4D_PAR_GQA_HEAD_UNROLL > 0, "E4D_PAR_GQA_HEAD_UNROLL must be > 0");
static_assert(E4D_PAR_KV_BCAST_TOKEN_UNROLL > 0,
              "E4D_PAR_KV_BCAST_TOKEN_UNROLL must be > 0");
static_assert(E4D_PAR_KV_DUP_HEAD_UNROLL > 0, "E4D_PAR_KV_DUP_HEAD_UNROLL must be > 0");
static_assert(E4D_PAR_LM_DEQUANT_PACK_UNROLL > 0,
              "E4D_PAR_LM_DEQUANT_PACK_UNROLL must be > 0");
static_assert(E4D_PAR_LM_DEQUANT_ELEM_UNROLL > 0,
              "E4D_PAR_LM_DEQUANT_ELEM_UNROLL must be > 0");
static_assert(E4D_PAR_LM_DOT_UNROLL > 0, "E4D_PAR_LM_DOT_UNROLL must be > 0");
static_assert(E4D_PAR_LM_SOFTMAX_UNROLL > 0, "E4D_PAR_LM_SOFTMAX_UNROLL must be > 0");
static_assert(E4D_PAR_W2B_NUM_BANKS > 0, "E4D_PAR_W2B_NUM_BANKS must be > 0");
static_assert(E4D_PAR_W2B_COMPUTE_II > 0, "E4D_PAR_W2B_COMPUTE_II must be > 0");

#endif // TMAC_EAGLE4_PARALLEL_CONFIG_HPP
