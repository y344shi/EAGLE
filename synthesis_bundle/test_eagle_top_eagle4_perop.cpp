#include "eagle_tier1_top.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tmac::hls;

namespace tmac {
namespace hls {
void distribute_q_heads(hls_stream<vec_t<VEC_W>>& s_q_rot, hls_stream<vec_t<VEC_W>> q_head_streams[NUM_HEADS]);
void broadcast_kv_heads(
    hls_stream<vec_t<VEC_W>>& s_k_hist_raw,
    hls_stream<vec_t<VEC_W>>& s_v_hist_raw,
    hls_stream<vec_t<VEC_W>> k_head_streams[NUM_HEADS],
    hls_stream<vec_t<VEC_W>> v_head_streams[NUM_HEADS],
    int hist_len);
void grouped_query_attention(
    hls_stream<vec_t<VEC_W>> q_head_streams[NUM_HEADS],
    hls_stream<vec_t<VEC_W>> k_head_streams[NUM_HEADS],
    hls_stream<vec_t<VEC_W>> v_head_streams[NUM_HEADS],
    hls_stream<vec_t<VEC_W>> ctx_head_streams[NUM_HEADS],
    int hist_len,
    int padded_len);
void collect_ctx(hls_stream<vec_t<VEC_W>>& s_context, hls_stream<vec_t<VEC_W>> ctx_head_streams[NUM_HEADS]);
void concat_embed_hidden(hls_stream<vec_t<VEC_W>>& s_embed_norm,
                         hls_stream<vec_t<VEC_W>>& s_hidden_norm,
                         hls_stream<vec_t<VEC_W>>& s_attn_cat);
void split_down_2hs(hls_stream<vec_t<VEC_W>>& s_down_2hs,
                    hls_stream<vec_t<VEC_W>>& s_to_logits,
                    hls_stream<vec_t<VEC_W>>& s_for_reasoning);
} // namespace hls
} // namespace tmac

namespace {

constexpr int MAX_SEQ = 2048;
constexpr int GROUP_SIZE = 128;
constexpr int VECS_H = HIDDEN / VEC_W;
constexpr int VECS_QKV_IN = QKV_INPUT / VEC_W;
constexpr int VECS_KV = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
constexpr int VECS_INTER = INTERMEDIATE / VEC_W;
constexpr int VECS_DOWN_2H = DOWN_OUTPUT / VEC_W;
constexpr int QKV_CAT_DIM = HIDDEN + 2 * NUM_KV_HEADS * HEAD_DIM;

template <typename T>
std::vector<T> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<T> out(static_cast<size_t>(sz / sizeof(T)));
    if (!f.read(reinterpret_cast<char*>(out.data()), sz)) return {};
    return out;
}

float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

std::vector<float> load_fp16(const std::string& path) {
    auto raw = load_bin<uint16_t>(path);
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) out[i] = fp16_to_float(raw[i]);
    return out;
}

size_t expected_pack_count(int in_dim, int out_dim) {
    return (static_cast<size_t>(in_dim) * static_cast<size_t>(out_dim)) / 128;
}

size_t expected_scale_count(int in_dim, int out_dim) {
    return static_cast<size_t>(in_dim / GROUP_SIZE) * static_cast<size_t>(out_dim);
}

template <typename T>
std::vector<T> drain_stream(hls_stream<T>& s, int n) {
    std::vector<T> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = s.read();
    return out;
}

template <typename T>
void write_stream(const std::vector<T>& in, hls_stream<T>& s) {
    for (const auto& v : in) s.write(v);
}

std::vector<float> flatten_vecs(const std::vector<vec_t<VEC_W>>& vs) {
    std::vector<float> out;
    out.resize(vs.size() * VEC_W);
    for (size_t i = 0; i < vs.size(); ++i) {
        for (int j = 0; j < VEC_W; ++j) {
            out[i * VEC_W + static_cast<size_t>(j)] = vs[i][j];
        }
    }
    return out;
}

std::vector<float> concat_float(const std::vector<float>& a,
                                const std::vector<float>& b,
                                const std::vector<float>& c = {}) {
    std::vector<float> out;
    out.reserve(a.size() + b.size() + c.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    out.insert(out.end(), c.begin(), c.end());
    return out;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

void compute_diff(const std::vector<float>& got,
                  const std::vector<float>& ref,
                  float* max_abs,
                  float* mean_abs) {
    *max_abs = 0.0f;
    *mean_abs = 0.0f;
    if (got.empty() || ref.empty()) return;
    const size_t n = std::min(got.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        if (d > *max_abs) *max_abs = d;
        *mean_abs += d;
    }
    *mean_abs /= static_cast<float>(n);
}

template <int HEAD_DIM_>
void fill_rope_cfg(RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM_>& cfg,
                   const std::vector<float>& inv_freq,
                   int pos) {
    for (int i = 0; i < HEAD_DIM_ / 2; ++i) {
        const float freq = static_cast<float>(pos) * inv_freq[i];
        cfg.cos_vals[i] = std::cos(freq);
        cfg.sin_vals[i] = std::sin(freq);
    }
}

std::vector<float> build_llama3_inv_freq(int head_dim) {
    constexpr float kRopeTheta = 500000.0f;
    constexpr float kScalingFactor = 8.0f;
    constexpr float kLowFreqFactor = 1.0f;
    constexpr float kHighFreqFactor = 4.0f;
    constexpr float kOrigMaxPos = 8192.0f;
    constexpr float kTwoPi = 6.2831853071795864769f;

    std::vector<float> inv_freq(static_cast<size_t>(head_dim / 2));
    const float low_freq_wavelen = kOrigMaxPos / kLowFreqFactor;
    const float high_freq_wavelen = kOrigMaxPos / kHighFreqFactor;
    for (int i = 0; i < head_dim / 2; ++i) {
        const float inv = 1.0f / std::pow(kRopeTheta, (2.0f * i) / head_dim);
        const float wave_len = kTwoPi / inv;
        float out = inv;
        if (wave_len > low_freq_wavelen) {
            out = inv / kScalingFactor;
        } else if (wave_len >= high_freq_wavelen) {
            const float smooth =
                (kOrigMaxPos / wave_len - kLowFreqFactor) / (kHighFreqFactor - kLowFreqFactor);
            out = (1.0f - smooth) * (inv / kScalingFactor) + smooth * inv;
        }
        inv_freq[static_cast<size_t>(i)] = out;
    }
    return inv_freq;
}

std::vector<float> select_token_slice(const std::vector<float>& all, int dim, int token_idx) {
    if (static_cast<int>(all.size()) == dim) return all;
    if (dim <= 0 || all.empty() || (all.size() % static_cast<size_t>(dim)) != 0) return {};
    const int tokens = static_cast<int>(all.size() / static_cast<size_t>(dim));
    int t = token_idx;
    if (t < 0) t = tokens - 1;
    if (t < 0 || t >= tokens) return {};
    std::vector<float> out(static_cast<size_t>(dim));
    std::copy_n(all.begin() + static_cast<size_t>(t) * static_cast<size_t>(dim), dim, out.begin());
    return out;
}

std::vector<vec_t<VEC_W>> float_to_vecs(const std::vector<float>& x) {
    const int vecs = static_cast<int>(x.size()) / VEC_W;
    std::vector<vec_t<VEC_W>> out(static_cast<size_t>(vecs));
    for (int i = 0; i < vecs; ++i) {
        for (int j = 0; j < VEC_W; ++j) {
            out[static_cast<size_t>(i)][j] = x[static_cast<size_t>(i) * VEC_W + static_cast<size_t>(j)];
        }
    }
    return out;
}

struct StageDump {
    std::unordered_map<std::string, std::vector<float>> f;
};

void run_one_token_with_taps(
    const std::vector<float>& hidden_token,
    const std::vector<float>& embed_token,
    StageDump* dump,
    const pack512* w_q, const float* s_q,
    const pack512* w_k, const float* s_k,
    const pack512* w_v, const float* s_v,
    const pack512* w_o, const float* s_o,
    const pack512* w_gate, const float* gate_scales,
    const pack512* w_up, const float* up_scales,
    const pack512* w_down, const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& rope_cfg,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    int current_length) {
    hls_stream<vec_t<VEC_W>> s_hidden_in("s_hidden_in");
    hls_stream<vec_t<VEC_W>> s_embed_in("s_embed_in");
    write_stream(float_to_vecs(hidden_token), s_hidden_in);
    write_stream(float_to_vecs(embed_token), s_embed_in);

    hls_stream<vec_t<VEC_W>> s_hidden_norm("s_hidden_norm");
    hls_stream<vec_t<VEC_W>> s_embed_norm("s_embed_norm");
    rms_norm_stream<HIDDEN>(s_hidden_in, s_hidden_norm, hidden_norm_gamma, RMS_EPS);
    rms_norm_stream<HIDDEN>(s_embed_in, s_embed_norm, embed_norm_gamma, RMS_EPS);
    auto hidden_norm_vecs = drain_stream(s_hidden_norm, VECS_H);
    auto embed_norm_vecs = drain_stream(s_embed_norm, VECS_H);
    dump->f["tensor_101"] = flatten_vecs(hidden_norm_vecs);
    dump->f["tensor_102"] = flatten_vecs(embed_norm_vecs);

    hls_stream<vec_t<VEC_W>> s_hidden_norm_re("s_hidden_norm_re");
    hls_stream<vec_t<VEC_W>> s_embed_norm_re("s_embed_norm_re");
    write_stream(hidden_norm_vecs, s_hidden_norm_re);
    write_stream(embed_norm_vecs, s_embed_norm_re);

    hls_stream<vec_t<VEC_W>> s_attn_cat("s_attn_cat");
    concat_embed_hidden(s_embed_norm_re, s_hidden_norm_re, s_attn_cat);
    auto attn_cat_vecs = drain_stream(s_attn_cat, VECS_QKV_IN);
    dump->f["tensor_103"] = flatten_vecs(attn_cat_vecs);

    hls_stream<vec_t<VEC_W>> s_q_in("s_q_in"), s_k_in("s_k_in"), s_v_in("s_v_in");
    write_stream(attn_cat_vecs, s_q_in);
    write_stream(attn_cat_vecs, s_k_in);
    write_stream(attn_cat_vecs, s_v_in);

    hls_stream<vec_t<VEC_W>> s_q_proj("s_q_proj"), s_k_proj("s_k_proj"), s_v_proj("s_v_proj");
    dense_projection_production_scaled<0, QKV_INPUT, HIDDEN, 128, TMAC_USE_TMAC_QKV>(s_q_in, s_q_proj, w_q, s_q);
    dense_projection_production_scaled<0, QKV_INPUT, NUM_KV_HEADS * HEAD_DIM, 128, TMAC_USE_TMAC_QKV>(s_k_in, s_k_proj, w_k, s_k);
    dense_projection_production_scaled<0, QKV_INPUT, NUM_KV_HEADS * HEAD_DIM, 128, TMAC_USE_TMAC_QKV>(s_v_in, s_v_proj, w_v, s_v);
    auto q_proj_vecs = drain_stream(s_q_proj, VECS_H);
    auto k_proj_vecs = drain_stream(s_k_proj, VECS_KV);
    auto v_proj_vecs = drain_stream(s_v_proj, VECS_KV);
    auto q_proj_f = flatten_vecs(q_proj_vecs);
    auto k_proj_f = flatten_vecs(k_proj_vecs);
    auto v_proj_f = flatten_vecs(v_proj_vecs);
    dump->f["tensor_113"] = q_proj_f;
    dump->f["tensor_114"] = k_proj_f;
    dump->f["tensor_115"] = v_proj_f;
    dump->f["tensor_112"] = concat_float(q_proj_f, k_proj_f, v_proj_f);

    hls_stream<vec_t<VEC_W>> s_q_proj_re("s_q_proj_re");
    hls_stream<vec_t<VEC_W>> s_k_proj_re("s_k_proj_re");
    write_stream(q_proj_vecs, s_q_proj_re);
    write_stream(k_proj_vecs, s_k_proj_re);

    hls_stream<vec_t<VEC_W>> s_q_rot("s_q_rot"), s_k_rot("s_k_rot");
    rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>(s_q_proj_re, s_q_rot, s_k_proj_re, s_k_rot, rope_cfg);
    auto q_rot_vecs = drain_stream(s_q_rot, VECS_H);
    auto k_rot_vecs = drain_stream(s_k_rot, VECS_KV);
    dump->f["tensor_116"] = flatten_vecs(q_rot_vecs);
    dump->f["tensor_117"] = flatten_vecs(k_rot_vecs);

    hls_stream<vec_t<VEC_W>> s_k_rot_re("s_k_rot_re"), s_v_proj_re("s_v_proj_re");
    write_stream(k_rot_vecs, s_k_rot_re);
    write_stream(v_proj_vecs, s_v_proj_re);

    hls_stream<vec_t<VEC_W>> s_k_hist_raw("s_k_hist_raw"), s_v_hist_raw("s_v_hist_raw");
    kv_cache_manager<HEAD_DIM, NUM_KV_HEADS>(
        s_k_rot_re, s_v_proj_re, s_k_hist_raw, s_v_hist_raw, hbm_k, hbm_v, current_length, true, true);

    const int hist_len = current_length + 1;
    const int hist_vecs = hist_len * VECS_KV;
    auto k_hist_vecs = drain_stream(s_k_hist_raw, hist_vecs);
    auto v_hist_vecs = drain_stream(s_v_hist_raw, hist_vecs);

    hls_stream<vec_t<VEC_W>> s_q_rot_re("s_q_rot_re");
    write_stream(q_rot_vecs, s_q_rot_re);
    hls_stream<vec_t<VEC_W>> s_k_hist_re("s_k_hist_re"), s_v_hist_re("s_v_hist_re");
    write_stream(k_hist_vecs, s_k_hist_re);
    write_stream(v_hist_vecs, s_v_hist_re);

    hls_stream<vec_t<VEC_W>> q_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> k_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> v_head_streams[NUM_HEADS];
    hls_stream<vec_t<VEC_W>> ctx_head_streams[NUM_HEADS];
    distribute_q_heads(s_q_rot_re, q_head_streams);
    broadcast_kv_heads(s_k_hist_re, s_v_hist_re, k_head_streams, v_head_streams, hist_len);
    const int padded_len = ((hist_len + 127) / 128) * 128;
    grouped_query_attention(q_head_streams, k_head_streams, v_head_streams, ctx_head_streams, hist_len, padded_len);

    hls_stream<vec_t<VEC_W>> s_context("s_context");
    collect_ctx(s_context, ctx_head_streams);
    auto context_vecs = drain_stream(s_context, VECS_H);
    dump->f["tensor_118"] = flatten_vecs(context_vecs);

    hls_stream<vec_t<VEC_W>> s_context_re("s_context_re"), s_o_proj("s_o_proj");
    write_stream(context_vecs, s_context_re);
    dense_projection_production_scaled<0, HIDDEN, HIDDEN, 128, TMAC_USE_TMAC_O>(s_context_re, s_o_proj, w_o, s_o);
    auto o_proj_vecs = drain_stream(s_o_proj, VECS_H);
    dump->f["tensor_119"] = flatten_vecs(o_proj_vecs);
    dump->f["tensor_104"] = dump->f["tensor_119"];

    hls_stream<vec_t<VEC_W>> s_o_proj_re("s_o_proj_re"), s_hidden_resid("s_hidden_resid"), s_post_resid("s_post_resid");
    write_stream(o_proj_vecs, s_o_proj_re);
    write_stream(float_to_vecs(hidden_token), s_hidden_resid);
    stream_add<VEC_W>(s_o_proj_re, s_hidden_resid, s_post_resid, VECS_H);
    auto post_resid_vecs = drain_stream(s_post_resid, VECS_H);
    dump->f["tensor_106"] = flatten_vecs(post_resid_vecs);

    hls_stream<vec_t<VEC_W>> s_post_resid_re("s_post_resid_re"), s_post_norm("s_post_norm");
    write_stream(post_resid_vecs, s_post_resid_re);
    rms_norm_stream<HIDDEN>(s_post_resid_re, s_post_norm, post_attn_norm_gamma, RMS_EPS);
    auto post_norm_vecs = drain_stream(s_post_norm, VECS_H);
    dump->f["tensor_105"] = flatten_vecs(post_norm_vecs);

    hls_stream<vec_t<VEC_W>> s_gate_in("s_gate_in"), s_up_in("s_up_in");
    write_stream(post_norm_vecs, s_gate_in);
    write_stream(post_norm_vecs, s_up_in);
    hls_stream<vec_t<VEC_W>> s_gate_vec("s_gate_vec"), s_up_vec("s_up_vec");
    dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE, 128, TMAC_USE_TMAC_FFN>(s_gate_in, s_gate_vec, w_gate, gate_scales);
    dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE, 128, TMAC_USE_TMAC_FFN>(s_up_in, s_up_vec, w_up, up_scales);
    auto gate_vecs = drain_stream(s_gate_vec, VECS_INTER);
    auto up_vecs = drain_stream(s_up_vec, VECS_INTER);
    auto gate_f = flatten_vecs(gate_vecs);
    auto up_f = flatten_vecs(up_vecs);
    dump->f["tensor_121"] = gate_f;
    dump->f["tensor_122"] = up_f;
    dump->f["tensor_120"] = concat_float(gate_f, up_f);

    hls_stream<vec_t<VEC_W>> s_gate_re("s_gate_re"), s_up_re("s_up_re"), s_swiglu("s_swiglu");
    write_stream(gate_vecs, s_gate_re);
    write_stream(up_vecs, s_up_re);
    silu_mul_stream<VEC_W>(s_gate_re, s_up_re, s_swiglu, VECS_INTER);
    auto swiglu_vecs = drain_stream(s_swiglu, VECS_INTER);
    dump->f["tensor_123"] = flatten_vecs(swiglu_vecs);

    hls_stream<vec_t<VEC_W>> s_swiglu_re("s_swiglu_re"), s_down_2h("s_down_2h");
    write_stream(swiglu_vecs, s_swiglu_re);
    dense_projection_production_scaled<0, INTERMEDIATE, DOWN_OUTPUT, 128, TMAC_USE_TMAC_FFN>(s_swiglu_re, s_down_2h, w_down, down_scales);
    auto down_vecs = drain_stream(s_down_2h, VECS_DOWN_2H);
    dump->f["tensor_124"] = flatten_vecs(down_vecs);
    dump->f["tensor_107"] = dump->f["tensor_124"];

    hls_stream<vec_t<VEC_W>> s_down_re("s_down_re"), s_to_logits("s_to_logits"), s_for_reason("s_for_reason");
    write_stream(down_vecs, s_down_re);
    split_down_2hs(s_down_re, s_to_logits, s_for_reason);
    auto to_logits_vecs = drain_stream(s_to_logits, VECS_H);
    auto for_reason_vecs = drain_stream(s_for_reason, VECS_H);
    dump->f["tensor_108"] = flatten_vecs(to_logits_vecs);
    dump->f["tensor_109"] = flatten_vecs(for_reason_vecs);

    hls_stream<vec_t<VEC_W>> s_to_logits_re("s_to_logits_re"), s_logits_norm("s_logits_norm");
    write_stream(to_logits_vecs, s_to_logits_re);
    rms_norm_stream<HIDDEN>(s_to_logits_re, s_logits_norm, final_norm_gamma, RMS_EPS);
    auto logits_norm_vecs = drain_stream(s_logits_norm, VECS_H);
    dump->f["tensor_110"] = flatten_vecs(logits_norm_vecs);

    hls_stream<vec_t<VEC_W>> s_for_reason_re("s_for_reason_re"), s_post_resid_re2("s_post_resid_re2"), s_after_add("s_after_add");
    write_stream(for_reason_vecs, s_for_reason_re);
    write_stream(post_resid_vecs, s_post_resid_re2);
    stream_add<VEC_W>(s_for_reason_re, s_post_resid_re2, s_after_add, VECS_H);
    auto after_add_vecs = drain_stream(s_after_add, VECS_H);
    dump->f["tensor_111"] = flatten_vecs(after_add_vecs);
}

void print_required(const std::string& t, const std::string& p, const std::string& n) {
    std::cout << "Required files for per-op Tier1 checker:\n";
    std::cout << "  " << t << "tensor_006_EAGLE_INPUT_prev_embed_ALL.bin\n";
    std::cout << "  " << t << "tensor_007_EAGLE_INPUT_prev_hidden_ALL.bin\n";
    std::cout << "  " << t << "tensor_101..124 + 104..111 stage tensors\n";
    std::cout << "  " << p << "q/k/v/o/gate/up/down weights+scales swizzled\n";
    std::cout << "  " << n << "hidden_norm/input_layernorm/post_attention_layernorm/final_norm\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string base_tensors = "../eagle_verified_pipeline_4bit/cpmcu_tensors/";
    std::string base_weights = "../packed_all/";
    std::string base_norms = "../eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit/";
    int token_idx = -1;
    float tol = 1e-1f;
    bool list_required = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tensor-dir" && i + 1 < argc) base_tensors = argv[++i];
        else if (arg == "--packed-dir" && i + 1 < argc) base_weights = argv[++i];
        else if (arg == "--norm-dir" && i + 1 < argc) base_norms = argv[++i];
        else if (arg == "--token-idx" && i + 1 < argc) token_idx = std::atoi(argv[++i]);
        else if (arg == "--tol" && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (arg == "--list-required-goldens") list_required = true;
        else if (arg.rfind("--tensor-dir=", 0) == 0) base_tensors = arg.substr(13);
        else if (arg.rfind("--packed-dir=", 0) == 0) base_weights = arg.substr(13);
        else if (arg.rfind("--norm-dir=", 0) == 0) base_norms = arg.substr(11);
    }

    if (!base_tensors.empty() && base_tensors.back() != '/') base_tensors.push_back('/');
    if (!base_weights.empty() && base_weights.back() != '/') base_weights.push_back('/');
    if (!base_norms.empty() && base_norms.back() != '/') base_norms.push_back('/');

    if (list_required) {
        print_required(base_tensors, base_weights, base_norms);
        return 0;
    }

    auto embed_all = load_fp16(base_tensors + "tensor_006_EAGLE_INPUT_prev_embed_ALL.bin");
    auto hidden_all = load_fp16(base_tensors + "tensor_007_EAGLE_INPUT_prev_hidden_ALL.bin");
    if (embed_all.size() < HIDDEN || hidden_all.size() < HIDDEN) {
        std::cout << "[FAIL] missing input tensors.\n";
        print_required(base_tensors, base_weights, base_norms);
        return 1;
    }
    const int embed_tokens = static_cast<int>(embed_all.size() / HIDDEN);
    const int hidden_tokens = static_cast<int>(hidden_all.size() / HIDDEN);
    const int total_tokens = std::min(embed_tokens, hidden_tokens);
    if (total_tokens <= 0) {
        std::cout << "[FAIL] invalid token count.\n";
        return 1;
    }
    if (token_idx < 0) token_idx = total_tokens - 1;
    if (token_idx >= total_tokens) {
        std::cout << "[FAIL] token_idx out of range: " << token_idx << " >= " << total_tokens << "\n";
        return 1;
    }

    auto w_q = load_bin<pack512>(base_weights + "q_proj_weights_swizzled.bin");
    auto s_q = load_bin<float>(base_weights + "q_proj_scales_swizzled.bin");
    auto w_k = load_bin<pack512>(base_weights + "k_proj_weights_swizzled.bin");
    auto s_k = load_bin<float>(base_weights + "k_proj_scales_swizzled.bin");
    auto w_v = load_bin<pack512>(base_weights + "v_proj_weights_swizzled.bin");
    auto s_v = load_bin<float>(base_weights + "v_proj_scales_swizzled.bin");
    auto w_o = load_bin<pack512>(base_weights + "o_proj_weights_swizzled.bin");
    auto s_o = load_bin<float>(base_weights + "o_proj_scales_swizzled.bin");
    auto w_gate = load_bin<pack512>(base_weights + "gate_proj_weights_swizzled.bin");
    auto s_gate = load_bin<float>(base_weights + "gate_proj_scales_swizzled.bin");
    auto w_up = load_bin<pack512>(base_weights + "up_proj_weights_swizzled.bin");
    auto s_up = load_bin<float>(base_weights + "up_proj_scales_swizzled.bin");
    auto w_down = load_bin<pack512>(base_weights + "down_proj_weights_swizzled.bin");
    auto s_down = load_bin<float>(base_weights + "down_proj_scales_swizzled.bin");

    auto hidden_norm = load_fp16(base_norms + "hidden_norm.fp16.bin");
    auto embed_norm = load_fp16(base_norms + "input_layernorm.fp16.bin");
    auto post_norm = load_fp16(base_norms + "post_attention_layernorm.fp16.bin");
    auto final_norm = load_fp16(base_norms + "final_norm.fp16.bin");

    bool shape_ok = true;
    shape_ok &= w_q.size() == expected_pack_count(QKV_INPUT, HIDDEN);
    shape_ok &= w_k.size() == expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM);
    shape_ok &= w_v.size() == expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM);
    shape_ok &= w_o.size() == expected_pack_count(HIDDEN, HIDDEN);
    shape_ok &= w_gate.size() == expected_pack_count(HIDDEN, INTERMEDIATE);
    shape_ok &= w_up.size() == expected_pack_count(HIDDEN, INTERMEDIATE);
    shape_ok &= w_down.size() == expected_pack_count(INTERMEDIATE, DOWN_OUTPUT);
    shape_ok &= s_q.size() == expected_scale_count(QKV_INPUT, HIDDEN);
    shape_ok &= s_k.size() == expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM);
    shape_ok &= s_v.size() == expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM);
    shape_ok &= s_o.size() == expected_scale_count(HIDDEN, HIDDEN);
    shape_ok &= s_gate.size() == expected_scale_count(HIDDEN, INTERMEDIATE);
    shape_ok &= s_up.size() == expected_scale_count(HIDDEN, INTERMEDIATE);
    shape_ok &= s_down.size() == expected_scale_count(INTERMEDIATE, DOWN_OUTPUT);
    shape_ok &= hidden_norm.size() >= HIDDEN && embed_norm.size() >= HIDDEN &&
                post_norm.size() >= HIDDEN && final_norm.size() >= HIDDEN;
    if (!shape_ok) {
        std::cout << "[FAIL] missing/mismatched weights or norms.\n";
        return 1;
    }

    std::unordered_map<std::string, int> stage_dims = {
        {"tensor_101", HIDDEN}, {"tensor_102", HIDDEN}, {"tensor_103", QKV_INPUT},
        {"tensor_104", HIDDEN}, {"tensor_105", HIDDEN}, {"tensor_106", HIDDEN},
        {"tensor_107", DOWN_OUTPUT}, {"tensor_108", HIDDEN}, {"tensor_109", HIDDEN},
        {"tensor_110", HIDDEN}, {"tensor_111", HIDDEN}, {"tensor_112", QKV_CAT_DIM},
        {"tensor_113", HIDDEN}, {"tensor_114", NUM_KV_HEADS * HEAD_DIM}, {"tensor_115", NUM_KV_HEADS * HEAD_DIM},
        {"tensor_116", HIDDEN}, {"tensor_117", NUM_KV_HEADS * HEAD_DIM}, {"tensor_118", HIDDEN},
        {"tensor_119", HIDDEN}, {"tensor_120", 2 * INTERMEDIATE}, {"tensor_121", INTERMEDIATE},
        {"tensor_122", INTERMEDIATE}, {"tensor_123", INTERMEDIATE}, {"tensor_124", DOWN_OUTPUT},
    };

    std::unordered_map<std::string, std::vector<float>> golden;
    for (const auto& kv : stage_dims) {
        const std::string path = base_tensors + kv.first + "_EAGLE_L0_" +
            (kv.first == "tensor_101" ? "hidden_norm" :
             kv.first == "tensor_102" ? "embed_norm" :
             kv.first == "tensor_103" ? "attn_input_cat" :
             kv.first == "tensor_104" ? "self_attn_out" :
             kv.first == "tensor_105" ? "post_attn_norm" :
             kv.first == "tensor_106" ? "post_attn_residual" :
             kv.first == "tensor_107" ? "mlp_out" :
             kv.first == "tensor_108" ? "to_logits" :
             kv.first == "tensor_109" ? "for_reasoning" :
             kv.first == "tensor_110" ? "to_logits_after_norm" :
             kv.first == "tensor_111" ? "after_residual_add" :
             kv.first == "tensor_112" ? "qkv_concat" :
             kv.first == "tensor_113" ? "q_linear" :
             kv.first == "tensor_114" ? "k_linear" :
             kv.first == "tensor_115" ? "v_linear" :
             kv.first == "tensor_116" ? "q_rope" :
             kv.first == "tensor_117" ? "k_rope" :
             kv.first == "tensor_118" ? "attn_context" :
             kv.first == "tensor_119" ? "o_proj" :
             kv.first == "tensor_120" ? "mlp_gate_up" :
             kv.first == "tensor_121" ? "mlp_gate" :
             kv.first == "tensor_122" ? "mlp_up" :
             kv.first == "tensor_123" ? "mlp_silu_mul" :
                                      "mlp_down_proj") + ".bin";
        auto all = load_fp16(path);
        auto tok = select_token_slice(all, kv.second, token_idx);
        if (tok.empty()) {
            std::cout << "[FAIL] missing stage tensor: " << path << "\n";
            return 1;
        }
        golden[kv.first] = std::move(tok);
    }

    std::vector<float> inv_freq = build_llama3_inv_freq(HEAD_DIM);

    std::vector<vec_t<VEC_W>> hbm_k(MAX_SEQ * VECS_KV);
    std::vector<vec_t<VEC_W>> hbm_v(MAX_SEQ * VECS_KV);
    StageDump got;

    for (int t = 0; t <= token_idx; ++t) {
        std::vector<float> hidden_tok(HIDDEN), embed_tok(HIDDEN);
        std::copy_n(hidden_all.begin() + static_cast<size_t>(t) * HIDDEN, HIDDEN, hidden_tok.begin());
        std::copy_n(embed_all.begin() + static_cast<size_t>(t) * HIDDEN, HIDDEN, embed_tok.begin());
        RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg{};
        fill_rope_cfg<HEAD_DIM>(rope_cfg, inv_freq, t);

        StageDump cur;
        run_one_token_with_taps(
            hidden_tok,
            embed_tok,
            &cur,
            w_q.data(), s_q.data(),
            w_k.data(), s_k.data(),
            w_v.data(), s_v.data(),
            w_o.data(), s_o.data(),
            w_gate.data(), s_gate.data(),
            w_up.data(), s_up.data(),
            w_down.data(), s_down.data(),
            hidden_norm.data(),
            embed_norm.data(),
            post_norm.data(),
            final_norm.data(),
            rope_cfg,
            hbm_k.data(),
            hbm_v.data(),
            t);
        if (t == token_idx) got = std::move(cur);
    }

    bool fail = false;
    std::vector<std::string> order = {
        "tensor_101", "tensor_102", "tensor_103", "tensor_112", "tensor_113", "tensor_114", "tensor_115",
        "tensor_116", "tensor_117", "tensor_118", "tensor_119", "tensor_104", "tensor_105", "tensor_106",
        "tensor_120", "tensor_121", "tensor_122", "tensor_123", "tensor_124", "tensor_107",
        "tensor_108", "tensor_109", "tensor_110", "tensor_111"};

    std::cout << "[info] token_idx=" << token_idx << " tol=" << tol << "\n";
    for (const auto& key : order) {
        auto itg = got.f.find(key);
        auto itr = golden.find(key);
        if (itg == got.f.end() || itr == golden.end()) {
            std::cout << "[FAIL] missing stage in compare: " << key << "\n";
            fail = true;
            continue;
        }
        if (!all_finite(itg->second)) {
            std::cout << "[FAIL] non-finite stage output: " << key << "\n";
            fail = true;
            continue;
        }
        float mx = 0.0f, mean = 0.0f;
        compute_diff(itg->second, itr->second, &mx, &mean);
        std::cout << "[stage] " << key << " max_abs=" << mx << " mean_abs=" << mean << "\n";
        if (mx > tol) fail = true;
    }

    if (fail) {
        std::cout << "[FAIL] Tier1 per-op parity exceeds tolerance.\n";
        return 1;
    }
    std::cout << "[PASS] Tier1 per-op parity is within tolerance for all stages.\n";
    return 0;
}
