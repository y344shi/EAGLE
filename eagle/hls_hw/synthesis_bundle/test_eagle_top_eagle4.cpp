#include "eagle_tier1_top.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace tmac::hls;

namespace {

constexpr int MAX_SEQ = 2048;
constexpr int GROUP_SIZE = 128;

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

template <typename T>
std::vector<T> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<T> buf(static_cast<size_t>(sz / sizeof(T)));
    if (!f.read(reinterpret_cast<char*>(buf.data()), sz)) return {};
    return buf;
}

float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
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

bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

size_t expected_pack_count(int in_dim, int out_dim) {
    return (static_cast<size_t>(in_dim) * static_cast<size_t>(out_dim)) / 128;
}

size_t expected_scale_count(int in_dim, int out_dim) {
    return static_cast<size_t>(in_dim / GROUP_SIZE) * static_cast<size_t>(out_dim);
}

bool check_size(const char* name, size_t got, size_t expect) {
    if (got != expect) {
        std::cout << "[shape] " << name << " got=" << got << " expected=" << expect << "\n";
        return false;
    }
    return true;
}

void print_required_golden_files(const std::string& base_tensors,
                                 const std::string& base_weights,
                                 const std::string& base_norms) {
    std::cout << "Required files for test_eagle_top_eagle4:\n";
    std::cout << "  [required] " << base_tensors << "tensor_006_EAGLE_INPUT_prev_embed_ALL.bin\n";
    std::cout << "  [required] " << base_tensors << "tensor_007_EAGLE_INPUT_prev_hidden_ALL.bin\n";
    std::cout << "  [required] " << base_tensors << "tensor_110_EAGLE_L0_to_logits_after_norm.bin\n";
    std::cout << "  [required] " << base_tensors << "tensor_014_EAGLE_after_residual.bin\n";
    std::cout << "  [required] " << base_weights << "q_proj_weights_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "q_proj_scales_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "k_proj_weights_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "k_proj_scales_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "v_proj_weights_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "v_proj_scales_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "o_proj_weights_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "o_proj_scales_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "gate_proj_weights_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "gate_proj_scales_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "up_proj_weights_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "up_proj_scales_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "down_proj_weights_swizzled.bin\n";
    std::cout << "  [required] " << base_weights << "down_proj_scales_swizzled.bin\n";
    std::cout << "  [required] " << base_norms << "hidden_norm.fp16.bin\n";
    std::cout << "  [required] " << base_norms << "input_layernorm.fp16.bin\n";
    std::cout << "  [required] " << base_norms << "post_attention_layernorm.fp16.bin\n";
    std::cout << "  [required] " << base_norms << "final_norm.fp16.bin\n";
}

template <int HEAD_DIM_>
void fill_rope_cfg(RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM_>& cfg,
                   const std::vector<float>& inv_freq,
                   int pos) {
    const int half = HEAD_DIM_ / 2;
    for (int i = 0; i < half; ++i) {
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

void random_vec(std::vector<float>& v, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& x : v) x = dist(rng);
}

void random_pack(std::vector<pack512>& v, std::mt19937& rng) {
    std::uniform_int_distribution<int> nib(0, 15);
    for (pack512& p : v) {
        uint8_t* b = reinterpret_cast<uint8_t*>(&p);
        for (int i = 0; i < 64; ++i) {
            const uint8_t lo = static_cast<uint8_t>(nib(rng));
            const uint8_t hi = static_cast<uint8_t>(nib(rng));
            b[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }
}

int run_smoke(int seed) {
    std::mt19937 rng(seed);

    std::vector<pack512> w_q(expected_pack_count(QKV_INPUT, HIDDEN));
    std::vector<pack512> w_k(expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM));
    std::vector<pack512> w_v(expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM));
    std::vector<pack512> w_o(expected_pack_count(HIDDEN, HIDDEN));
    std::vector<pack512> w_gate(expected_pack_count(HIDDEN, INTERMEDIATE));
    std::vector<pack512> w_up(expected_pack_count(HIDDEN, INTERMEDIATE));
    std::vector<pack512> w_down(expected_pack_count(INTERMEDIATE, DOWN_OUTPUT));

    std::vector<float> s_q(expected_scale_count(QKV_INPUT, HIDDEN));
    std::vector<float> s_k(expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM));
    std::vector<float> s_v(expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM));
    std::vector<float> s_o(expected_scale_count(HIDDEN, HIDDEN));
    std::vector<float> s_gate(expected_scale_count(HIDDEN, INTERMEDIATE));
    std::vector<float> s_up(expected_scale_count(HIDDEN, INTERMEDIATE));
    std::vector<float> s_down(expected_scale_count(INTERMEDIATE, DOWN_OUTPUT));

    std::vector<float> hidden_norm(HIDDEN), embed_norm(HIDDEN), post_norm(HIDDEN), final_norm(HIDDEN);
    std::vector<float> hidden(HIDDEN), embed(HIDDEN);

    random_pack(w_q, rng);
    random_pack(w_k, rng);
    random_pack(w_v, rng);
    random_pack(w_o, rng);
    random_pack(w_gate, rng);
    random_pack(w_up, rng);
    random_pack(w_down, rng);

    random_vec(s_q, rng, 0.001f, 0.02f);
    random_vec(s_k, rng, 0.001f, 0.02f);
    random_vec(s_v, rng, 0.001f, 0.02f);
    random_vec(s_o, rng, 0.001f, 0.02f);
    random_vec(s_gate, rng, 0.001f, 0.02f);
    random_vec(s_up, rng, 0.001f, 0.02f);
    random_vec(s_down, rng, 0.001f, 0.02f);

    random_vec(hidden_norm, rng, 0.8f, 1.2f);
    random_vec(embed_norm, rng, 0.8f, 1.2f);
    random_vec(post_norm, rng, 0.8f, 1.2f);
    random_vec(final_norm, rng, 0.8f, 1.2f);
    random_vec(hidden, rng, -0.5f, 0.5f);
    random_vec(embed, rng, -0.5f, 0.5f);

    std::vector<vec_t<VEC_W>> hbm_k(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);
    std::vector<vec_t<VEC_W>> hbm_v(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);

    std::vector<float> inv_freq = build_llama3_inv_freq(HEAD_DIM);

    RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg{};
    fill_rope_cfg<HEAD_DIM>(rope_cfg, inv_freq, 0);

    hls_stream<vec_t<VEC_W>> hidden_stream, embed_stream, reasoning_out, logits_out;
    for (int i = 0; i < HIDDEN / VEC_W; ++i) {
        vec_t<VEC_W> hc;
        vec_t<VEC_W> ec;
        for (int j = 0; j < VEC_W; ++j) {
            hc[j] = hidden[i * VEC_W + j];
            ec[j] = embed[i * VEC_W + j];
        }
        hidden_stream.write(hc);
        embed_stream.write(ec);
    }

    eagle_tier1_top_eagle4_l0(hidden_stream, embed_stream, reasoning_out, logits_out,
                              w_q.data(), s_q.data(), w_k.data(), s_k.data(), w_v.data(), s_v.data(),
                              w_o.data(), s_o.data(), w_gate.data(), s_gate.data(), w_up.data(), s_up.data(),
                              w_down.data(), s_down.data(), hidden_norm.data(), embed_norm.data(),
                              post_norm.data(), final_norm.data(), rope_cfg, hbm_k.data(), hbm_v.data(), 0, 0);

    std::vector<float> reasoning(HIDDEN), logits(HIDDEN);
    for (int i = 0; i < HIDDEN / VEC_W; ++i) {
        auto r = reasoning_out.read();
        auto l = logits_out.read();
        for (int j = 0; j < VEC_W; ++j) {
            reasoning[i * VEC_W + j] = r[j];
            logits[i * VEC_W + j] = l[j];
        }
    }

    if (!all_finite(reasoning) || !all_finite(logits)) {
        std::cout << "[smoke] FAIL: non-finite output.\n";
        return 1;
    }
    std::cout << "[smoke] PASS (EAGLE4 parity path emitted finite reasoning/logits streams).\n";
    return 0;
}

void compute_diff(const std::vector<float>& got,
                  const std::vector<float>& ref,
                  float& max_abs,
                  float& mean_abs) {
    max_abs = 0.0f;
    mean_abs = 0.0f;
    if (got.empty() || ref.empty()) return;
    const size_t n = std::min(got.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    mean_abs /= static_cast<float>(n);
}

} // namespace

int main(int argc, char** argv) {
    bool smoke_mode = false;
    bool list_required = false;
    int smoke_seed = 17;
    std::string base_tensors = "../eagle_verified_pipeline_4bit/cpmcu_tensors/";
    std::string base_weights = "../packed_all/";
    std::string base_norms = "../eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit/";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--smoke") {
            smoke_mode = true;
        } else if (arg == "--list-required-goldens") {
            list_required = true;
        } else if (arg == "--tensor-dir" && i + 1 < argc) {
            base_tensors = argv[++i];
        } else if (arg == "--packed-dir" && i + 1 < argc) {
            base_weights = argv[++i];
        } else if (arg == "--norm-dir" && i + 1 < argc) {
            base_norms = argv[++i];
        } else if (arg.rfind("--tensor-dir=", 0) == 0) {
            base_tensors = arg.substr(13);
        } else if (arg.rfind("--packed-dir=", 0) == 0) {
            base_weights = arg.substr(13);
        } else if (arg.rfind("--norm-dir=", 0) == 0) {
            base_norms = arg.substr(11);
        } else if (arg.rfind("--smoke-seed=", 0) == 0) {
            smoke_seed = std::atoi(arg.substr(13).c_str());
        }
    }

    if (!base_tensors.empty() && base_tensors.back() != '/') base_tensors.push_back('/');
    if (!base_weights.empty() && base_weights.back() != '/') base_weights.push_back('/');
    if (!base_norms.empty() && base_norms.back() != '/') base_norms.push_back('/');

    if (list_required) {
        print_required_golden_files(base_tensors, base_weights, base_norms);
        if (!smoke_mode) return 0;
    }

    if (smoke_mode) {
        return run_smoke(smoke_seed);
    }

    auto embed_all = load_fp16(base_tensors + "tensor_006_EAGLE_INPUT_prev_embed_ALL.bin");
    auto hidden_all = load_fp16(base_tensors + "tensor_007_EAGLE_INPUT_prev_hidden_ALL.bin");
    auto golden_logits_all = load_fp16(base_tensors + "tensor_110_EAGLE_L0_to_logits_after_norm.bin");
    auto golden_reasoning_all = load_fp16(base_tensors + "tensor_014_EAGLE_after_residual.bin");

    if (embed_all.size() < HIDDEN || hidden_all.size() < HIDDEN ||
        golden_logits_all.size() < HIDDEN || golden_reasoning_all.size() < HIDDEN) {
        std::cout << "Missing required tensors for full check.\n";
        print_required_golden_files(base_tensors, base_weights, base_norms);
        return 1;
    }

    const int embed_tokens = static_cast<int>(embed_all.size() / HIDDEN);
    const int hidden_tokens = static_cast<int>(hidden_all.size() / HIDDEN);
    const int total_tokens = std::min(embed_tokens, hidden_tokens);
    if (total_tokens <= 0) {
        std::cout << "Invalid token count in inputs.\n";
        return 1;
    }

    const int target_token = total_tokens - 1;
    std::vector<float> golden_logits(HIDDEN), golden_reasoning(HIDDEN);

    const int logits_tokens = static_cast<int>(golden_logits_all.size() / HIDDEN);
    const int reasoning_tokens = static_cast<int>(golden_reasoning_all.size() / HIDDEN);
    if (logits_tokens == 1) {
        std::copy_n(golden_logits_all.begin(), HIDDEN, golden_logits.begin());
    } else if (logits_tokens > target_token) {
        std::copy_n(golden_logits_all.begin() + target_token * HIDDEN, HIDDEN, golden_logits.begin());
    } else {
        std::cout << "Golden logits tensor does not include target token.\n";
        return 1;
    }
    if (reasoning_tokens == 1) {
        std::copy_n(golden_reasoning_all.begin(), HIDDEN, golden_reasoning.begin());
    } else if (reasoning_tokens > target_token) {
        std::copy_n(golden_reasoning_all.begin() + target_token * HIDDEN, HIDDEN, golden_reasoning.begin());
    } else {
        std::cout << "Golden reasoning tensor does not include target token.\n";
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

    const bool shape_ok =
        check_size("q_proj_weights_swizzled.bin", w_q.size(), expected_pack_count(QKV_INPUT, HIDDEN)) &&
        check_size("k_proj_weights_swizzled.bin", w_k.size(), expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM)) &&
        check_size("v_proj_weights_swizzled.bin", w_v.size(), expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM)) &&
        check_size("o_proj_weights_swizzled.bin", w_o.size(), expected_pack_count(HIDDEN, HIDDEN)) &&
        check_size("gate_proj_weights_swizzled.bin", w_gate.size(), expected_pack_count(HIDDEN, INTERMEDIATE)) &&
        check_size("up_proj_weights_swizzled.bin", w_up.size(), expected_pack_count(HIDDEN, INTERMEDIATE)) &&
        check_size("down_proj_weights_swizzled.bin", w_down.size(), expected_pack_count(INTERMEDIATE, DOWN_OUTPUT)) &&
        check_size("q_proj_scales_swizzled.bin", s_q.size(), expected_scale_count(QKV_INPUT, HIDDEN)) &&
        check_size("k_proj_scales_swizzled.bin", s_k.size(), expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM)) &&
        check_size("v_proj_scales_swizzled.bin", s_v.size(), expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM)) &&
        check_size("o_proj_scales_swizzled.bin", s_o.size(), expected_scale_count(HIDDEN, HIDDEN)) &&
        check_size("gate_proj_scales_swizzled.bin", s_gate.size(), expected_scale_count(HIDDEN, INTERMEDIATE)) &&
        check_size("up_proj_scales_swizzled.bin", s_up.size(), expected_scale_count(HIDDEN, INTERMEDIATE)) &&
        check_size("down_proj_scales_swizzled.bin", s_down.size(), expected_scale_count(INTERMEDIATE, DOWN_OUTPUT));

    if (!shape_ok || hidden_norm.size() < HIDDEN || embed_norm.size() < HIDDEN ||
        post_norm.size() < HIDDEN || final_norm.size() < HIDDEN) {
        std::cout << "Missing or shape-mismatched weights/norms.\n";
        print_required_golden_files(base_tensors, base_weights, base_norms);
        return 1;
    }

    std::vector<float> inv_freq = build_llama3_inv_freq(HEAD_DIM);

    std::vector<vec_t<VEC_W>> hbm_k(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);
    std::vector<vec_t<VEC_W>> hbm_v(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);

    std::vector<float> out_reasoning;
    std::vector<float> out_logits;

    for (int t = 0; t <= target_token; ++t) {
        hls_stream<vec_t<VEC_W>> hidden_stream, embed_stream, reasoning_stream, logits_stream;

        for (int i = 0; i < HIDDEN / VEC_W; ++i) {
            vec_t<VEC_W> h;
            vec_t<VEC_W> e;
            for (int j = 0; j < VEC_W; ++j) {
                h[j] = hidden_all[t * HIDDEN + i * VEC_W + j];
                e[j] = embed_all[t * HIDDEN + i * VEC_W + j];
            }
            hidden_stream.write(h);
            embed_stream.write(e);
        }

        RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg{};
        fill_rope_cfg<HEAD_DIM>(rope_cfg, inv_freq, t);

        eagle_tier1_top_eagle4_l0(hidden_stream, embed_stream, reasoning_stream, logits_stream,
                                  w_q.data(), s_q.data(), w_k.data(), s_k.data(), w_v.data(), s_v.data(),
                                  w_o.data(), s_o.data(), w_gate.data(), s_gate.data(), w_up.data(), s_up.data(),
                                  w_down.data(), s_down.data(), hidden_norm.data(), embed_norm.data(),
                                  post_norm.data(), final_norm.data(), rope_cfg, hbm_k.data(), hbm_v.data(), t, t);

        if (t == target_token) {
            out_reasoning.resize(HIDDEN);
            out_logits.resize(HIDDEN);
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto r = reasoning_stream.read();
                auto l = logits_stream.read();
                for (int j = 0; j < VEC_W; ++j) {
                    out_reasoning[i * VEC_W + j] = r[j];
                    out_logits[i * VEC_W + j] = l[j];
                }
            }
        } else {
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                (void)reasoning_stream.read();
                (void)logits_stream.read();
            }
        }
    }

    if (!all_finite(out_reasoning) || !all_finite(out_logits)) {
        std::cout << "[FAIL] non-finite output from EAGLE4 parity top.\n";
        return 1;
    }

    float reasoning_max = 0.0f;
    float reasoning_mean = 0.0f;
    float logits_max = 0.0f;
    float logits_mean = 0.0f;

    compute_diff(out_reasoning, golden_reasoning, reasoning_max, reasoning_mean);
    compute_diff(out_logits, golden_logits, logits_max, logits_mean);

    std::cout << "[result] reasoning max_abs=" << reasoning_max << " mean_abs=" << reasoning_mean << "\n";
    std::cout << "[result] logits    max_abs=" << logits_max << " mean_abs=" << logits_mean << "\n";

    const float kTol = 1e-1f;
    if (reasoning_max > kTol || logits_max > kTol) {
        std::cout << "[FAIL] exceeds tolerance " << kTol << "\n";
        return 1;
    }

    std::cout << "[PASS] EAGLE4 parity top matches tensor_110/tensor_014 within tolerance.\n";
    return 0;
}
