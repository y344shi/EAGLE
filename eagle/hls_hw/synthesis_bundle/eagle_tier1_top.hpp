#ifndef TMAC_EAGLE_TIER1_TOP_HPP
#define TMAC_EAGLE_TIER1_TOP_HPP

// Full Tier-1 pipeline for EAGLE (attention + FFN) with buffered KV history.
// Key fix: buffer history once into URAM so all 32 heads can replay without
// draining the KV cache stream (previously only head0 consumed valid data).

#include "tmac_utils.hpp"
#include "attention_solver.hpp"
#include "fused_online_attention_pwl.hpp"
#include "deep_pipeline_lutmac.hpp"
#include "kv_cache_manager.hpp"
#include "rms_norm_stream.hpp"
#include "rope_kernel.hpp"
#include "stream_utils.hpp"

// Attention solver selection:
// 0 = legacy attention_solver (exact exp)
// 1 = fused_online_attention_pwl (PWL exp2)
// 2 = hybrid (legacy for short history, fused for long history)
#ifndef TMAC_ATTN_SOLVER_MODE
#define TMAC_ATTN_SOLVER_MODE 1
#endif

// Used only when TMAC_ATTN_SOLVER_MODE == 2.
#ifndef TMAC_ATTN_FUSED_SWITCH_LEN
#define TMAC_ATTN_FUSED_SWITCH_LEN 256
#endif

// Projection backend selection (dense_projection_production_scaled ENABLE_TMAC flag):
// 0 = DSP multiply path
// 1 = LUT-MAC/TMAC path
#ifndef TMAC_USE_TMAC_QKV
#define TMAC_USE_TMAC_QKV 1
#endif

#ifndef TMAC_USE_TMAC_O
#define TMAC_USE_TMAC_O 1
#endif

#ifndef TMAC_USE_TMAC_FFN
#define TMAC_USE_TMAC_FFN 1
#endif

namespace tmac {
namespace hls {

constexpr int HIDDEN = 4096;
constexpr int INTERMEDIATE = 14336;
constexpr int HEAD_DIM = 128;
constexpr int NUM_HEADS = 32;
constexpr int NUM_KV_HEADS = 32;
constexpr int QKV_INPUT = HIDDEN * 2;
constexpr int DOWN_OUTPUT = HIDDEN * 2;
constexpr float RMS_EPS = 1e-5f;
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

// EAGLE4 layer-0 parity path:
// hidden_norm(hidden) + input_layernorm(embed) -> cat(2H) -> QKV(attn) -> post_attn_norm ->
// MLP down_proj(2H) -> split(to_logits, for_reasoning) -> final_norm(to_logits) + residual add(reasoning).
void eagle_tier1_top_eagle4_l0(hls_stream<vec_t<VEC_W>>& hidden_in_stream,
                               hls_stream<vec_t<VEC_W>>& embed_in_stream,
                               hls_stream<vec_t<VEC_W>>& reasoning_out_stream,
                               hls_stream<vec_t<VEC_W>>& logits_norm_out_stream,
                               const pack512* w_q,     const float* s_q,
                               const pack512* w_k,     const float* s_k,
                               const pack512* w_v,     const float* s_v,
                               const pack512* w_o,     const float* s_o,
                               const pack512* w_gate,  const float* gate_scales,
                               const pack512* w_up,    const float* up_scales,
                               const pack512* w_down,  const float* down_scales,
                               const float* hidden_norm_gamma,
                               const float* embed_norm_gamma,
                               const float* post_attn_norm_gamma,
                               const float* final_norm_gamma,
                               const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& rope_cfg,
                               vec_t<VEC_W>* hbm_k,
                               vec_t<VEC_W>* hbm_v,
                               int seq_len,
                               int current_length);

} // namespace hls
} // namespace tmac

#endif // TMAC_EAGLE_TIER1_TOP_HPP
