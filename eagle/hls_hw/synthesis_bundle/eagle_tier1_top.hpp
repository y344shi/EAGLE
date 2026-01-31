#ifndef TMAC_EAGLE_TIER1_TOP_HPP
#define TMAC_EAGLE_TIER1_TOP_HPP

// Full Tier-1 pipeline for EAGLE (attention + FFN) with buffered KV history.
// Key fix: buffer history once into URAM so all 32 heads can replay without
// draining the KV cache stream (previously only head0 consumed valid data).

#include "tmac_utils.hpp"
#include "attention_solver.hpp"
#include "deep_pipeline_lutmac.hpp"
#include "kv_cache_manager.hpp"
#include "rms_norm_stream.hpp"
#include "rope_kernel.hpp"
#include "stream_utils.hpp"

namespace tmac {
namespace hls {

constexpr int HIDDEN = 4096;
constexpr int INTERMEDIATE = 16384;
constexpr int HEAD_DIM = 128;
constexpr int NUM_HEADS = 32;
constexpr int NUM_KV_HEADS = 2;
constexpr float RESIDUAL_SCALE = 1.4f / 5.7445626465380286f; // sqrt(33)
constexpr int MAX_CTX = 2048; // maximum sequence length buffered locally

using tmac::hls::vec_t;
using tmac::hls::hls_stream;
using tmac::hls::VEC_W;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void eagle_tier1_top(hls_stream<vec_t<VEC_W>>& in_stream,
                     hls_stream<vec_t<VEC_W>>& out_stream,
                     const pack512* w_q,     const float* s_q,
                     const pack512* w_k,     const float* s_k,
                     const pack512* w_v,     const float* s_v,
                     const pack512* w_o,     const float* s_o,
                     const pack512* w_gate,  const float* gate_scales,
                     const pack512* w_up,    const float* up_scales,
                     const pack512* w_down,  const float* down_scales,
                     const float* norm1_gamma,
                     const float* norm2_gamma,
                     const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& rope_cfg,
                     vec_t<VEC_W>* hbm_k,
                     vec_t<VEC_W>* hbm_v,
                     int seq_len,
                     int current_length);

} // namespace hls
} // namespace tmac

#endif // TMAC_EAGLE_TIER1_TOP_HPP
