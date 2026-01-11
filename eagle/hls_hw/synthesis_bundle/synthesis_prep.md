# Synthesis Prep Notes (EAGLE Tier‑1 / LUTMAC)

Summary of key HLS kernels, pragmas, and design intent before synthesis.

## eagle_tier1_top.hpp
- Top: `#pragma HLS DATAFLOW`; AXIS on in/out streams; AXI m_axi on weights/norms/KV; s_axilite control.
- Q buffer: `q_buf[NUM_HEADS][HEAD_DIM]` with `ARRAY_PARTITION complete dim=2` to feed per-head Q lanes without stalls.
- KV history: `k_hist_buf/v_hist_buf[MAX_CTX][VECS_PER_KV_TOKEN]` with `BIND_STORAGE ram_2p impl=uram` and `ARRAY_PARTITION cyclic factor=2 dim=2` to replay history for all heads at II=1.
- Loops: Q buffer load `PIPELINE II=1` with lane `UNROLL`; KV buffer load `PIPELINE II=1`; per-head replay loops pipelined; `LOOP_TRIPCOUNT` on head loop for estimates.
- Attention uses padded_len to mirror Marlin decode. GQA filtering via block index.
- Projections use `dense_projection_production_scaled<..., ENABLE_TMAC=false>` for o_proj/gate/up/down to match CPU math.
- Residual path: `stream_scale` + `stream_add`; norm via `rms_norm_stream` (eps=1e-5).

## attention_solver.hpp
- AXIS on q/k/v/context; s_axilite seq_len/padded_len.
- Q and output buffers `ARRAY_PARTITION cyclic factor=VEC_W`.
- Token loop `PIPELINE` over K/V chunks; per-lane `UNROLL factor=VEC_W`.
- Optional padded_len masks extra tokens (score = -1e9, v_local = 0).

## deep_pipeline_lutmac.hpp (projection kernels)
- Broadcast GEMM accumulators `ARRAY_PARTITION complete` on tile/bank, cyclic on lane to sustain II=1 scalar broadcast.
- Inner loops `UNROLL factor=16` over lanes; outer input loop `PIPELINE II=1`.
- Scaled variant handles per-group scales; TMAC toggle available (disabled in top).

## rms_norm_stream.hpp
- Buffer `ARRAY_PARTITION cyclic factor=VEC_W`; load+reduce then normalize with `PIPELINE II=1`.

## stream_utils.hpp
- Streaming ops with `PIPELINE II=1`; lane `UNROLL` for vec16 math.

## kv_cache_manager.hpp
- Streams K/V to HBM and history streams once; top buffers to URAM for replay. (Pragmas unchanged in this round.)

## Design rationale
- Dataflow to keep attention+FFN streaming.
- Full partition of Q buffer to drive per-head streams.
- URAM buffering of KV history to avoid stream starvation and allow head replay at II=1.
- Pipelined inner loops to maintain throughput; lane unroll matches VEC_W=16.
- TMAC disabled in projections for numerical parity with CPU reference.
- Padded attention length to align with Marlin decode masking behavior.
