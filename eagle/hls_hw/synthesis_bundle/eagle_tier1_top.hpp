#ifndef TMAC_EAGLE_TIER1_TOP_HPP
#define TMAC_EAGLE_TIER1_TOP_HPP

#include "tmac_utils.hpp"
#include "stream_utils.hpp"
#include "rope_kernel.hpp"

namespace tmac {
namespace hls {

// Constants (Visible to other modules like LM Head)
constexpr int HIDDEN = 4096;
constexpr int INTERMEDIATE = 16384;
constexpr int HEAD_DIM = 128;
constexpr int NUM_HEADS = 32;
constexpr int NUM_KV_HEADS = 2;

using tmac::hls::vec_t;
using tmac::hls::hls_stream;
using tmac::hls::VEC_W;

// FUNCTION DECLARATION ONLY (No Body)
void eagle_tier1_top(
    hls_stream<vec_t<VEC_W>>& in_stream,
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
    int current_length
);

} // namespace hls
} // namespace tmac

#endif // TMAC_EAGLE_TIER1_TOP_HPP