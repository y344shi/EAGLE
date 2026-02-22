#include "eagle_tier1_lm_top.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace tmac::hls;

namespace {

constexpr int MAX_SEQ = 2048;
constexpr int GROUP_SIZE = 128;

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

template <int HEAD_DIM_>
void fill_rope_cfg(RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM_>& cfg, const std::vector<float>& inv_freq, int pos) {
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

int run_smoke() {
    std::cout << "[smoke] Compile/path smoke only. Use full run with captured files.\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool smoke = false;
    int run_tokens = 1;
    std::string base_tensors = "../eagle_verified_pipeline_4bit/cpmcu_tensors/";
    std::string base_weights = "../packed_all/";
    std::string base_norms = "../eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit/";
    std::string base_lm = "../eagle_verified_pipeline_4bit/hls_4bit/lm_head/";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--smoke") {
            smoke = true;
        } else if (arg == "--tensor-dir" && i + 1 < argc) {
            base_tensors = argv[++i];
        } else if (arg == "--packed-dir" && i + 1 < argc) {
            base_weights = argv[++i];
        } else if (arg == "--norm-dir" && i + 1 < argc) {
            base_norms = argv[++i];
        } else if (arg == "--lm-dir" && i + 1 < argc) {
            base_lm = argv[++i];
        } else if (arg == "--run-tokens" && i + 1 < argc) {
            run_tokens = std::atoi(argv[++i]);
        }
    }

    if (!base_tensors.empty() && base_tensors.back() != '/') base_tensors.push_back('/');
    if (!base_weights.empty() && base_weights.back() != '/') base_weights.push_back('/');
    if (!base_norms.empty() && base_norms.back() != '/') base_norms.push_back('/');
    if (!base_lm.empty() && base_lm.back() != '/') base_lm.push_back('/');

    if (smoke) return run_smoke();

    auto embed_all = load_fp16(base_tensors + "tensor_006_EAGLE_INPUT_prev_embed_ALL.bin");
    auto hidden_all = load_fp16(base_tensors + "tensor_007_EAGLE_INPUT_prev_hidden_ALL.bin");

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

    auto lm_down = load_bin<uint16_t>(base_lm + "efficient_lm_head_down_proj_weight.fp16.bin");
    auto lm_q = load_bin<int32_t>(base_lm + "efficient_lm_head_qweight_row_major.bin");
    auto lm_s = load_bin<uint16_t>(base_lm + "efficient_lm_head_scales_row_major.bin");
    auto lm_z = load_bin<int32_t>(base_lm + "efficient_lm_head_qzeros.bin");
    auto lm_g = load_bin<int32_t>(base_lm + "efficient_lm_head_g_idx.bin");
    auto lm_w = load_bin<uint16_t>(base_lm + "lm_head_weight.fp16.bin");

    if (embed_all.size() < HIDDEN || hidden_all.size() < HIDDEN || w_q.empty() || s_q.empty() || w_k.empty() ||
        s_k.empty() || w_v.empty() || s_v.empty() || w_o.empty() || s_o.empty() || w_gate.empty() ||
        s_gate.empty() || w_up.empty() || s_up.empty() || w_down.empty() || s_down.empty() ||
        hidden_norm.size() < HIDDEN || embed_norm.size() < HIDDEN || post_norm.size() < HIDDEN ||
        final_norm.size() < HIDDEN || lm_down.empty() || lm_q.empty() || lm_s.empty() || lm_w.empty()) {
        std::cout << "[FAIL] Missing required tensors/weights for top-level LM wiring test.\n";
        return 1;
    }

    const bool shape_ok =
        w_q.size() == expected_pack_count(QKV_INPUT, HIDDEN) &&
        w_k.size() == expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM) &&
        w_v.size() == expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM) &&
        w_o.size() == expected_pack_count(HIDDEN, HIDDEN) &&
        w_gate.size() == expected_pack_count(HIDDEN, INTERMEDIATE) &&
        w_up.size() == expected_pack_count(HIDDEN, INTERMEDIATE) &&
        w_down.size() == expected_pack_count(INTERMEDIATE, DOWN_OUTPUT) &&
        s_q.size() == expected_scale_count(QKV_INPUT, HIDDEN) &&
        s_k.size() == expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM) &&
        s_v.size() == expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM) &&
        s_o.size() == expected_scale_count(HIDDEN, HIDDEN) &&
        s_gate.size() == expected_scale_count(HIDDEN, INTERMEDIATE) &&
        s_up.size() == expected_scale_count(HIDDEN, INTERMEDIATE) &&
        s_down.size() == expected_scale_count(INTERMEDIATE, DOWN_OUTPUT);
    if (!shape_ok) {
        std::cout << "[FAIL] projection weight/scale shape mismatch.\n";
        return 1;
    }

    const int tokens = static_cast<int>(std::min(embed_all.size(), hidden_all.size()) / HIDDEN);
    const int rank = static_cast<int>(lm_down.size() / HIDDEN);
    const int rank_packs = rank / 8;
    const int vocab = (rank_packs > 0) ? static_cast<int>(lm_q.size() / rank_packs) : 0;
    const int groups = (GROUP_SIZE > 0) ? rank / GROUP_SIZE : 0;
    const int topk = 512;
    if (tokens <= 0 || rank <= 0 || vocab <= 0 || groups <= 0 || lm_s.size() != static_cast<size_t>(groups) * vocab ||
        lm_g.size() != static_cast<size_t>(rank)) {
        std::cout << "[FAIL] invalid inferred LM dimensions.\n";
        return 1;
    }

    std::vector<float> inv_freq = build_llama3_inv_freq(HEAD_DIM);

    std::vector<vec_t<VEC_W>> hbm_k_wrap(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);
    std::vector<vec_t<VEC_W>> hbm_v_wrap(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);
    std::vector<vec_t<VEC_W>> hbm_k_dir(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);
    std::vector<vec_t<VEC_W>> hbm_v_dir(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);

    std::vector<int> wrap_ids(topk, -1), dir_ids(topk, -1);
    std::vector<float> wrap_logits(topk, 0.0f), dir_logits(topk, 0.0f);
    std::vector<float> wrap_reason(HIDDEN, 0.0f), dir_reason(HIDDEN, 0.0f);
    int wrap_best_id = -1;
    int dir_best_id = -1;
    float wrap_best_score = -std::numeric_limits<float>::infinity();
    float dir_best_score = -std::numeric_limits<float>::infinity();

    if (run_tokens <= 0) run_tokens = 1;
    if (run_tokens > tokens) run_tokens = tokens;

    for (int t = 0; t < run_tokens; ++t) {
        hls_stream<vec_t<VEC_W>> hidden_w, embed_w, hidden_d, embed_d;
        for (int i = 0; i < HIDDEN / VEC_W; ++i) {
            vec_t<VEC_W> hv, ev;
            for (int j = 0; j < VEC_W; ++j) {
                hv[j] = hidden_all[t * HIDDEN + i * VEC_W + j];
                ev[j] = embed_all[t * HIDDEN + i * VEC_W + j];
            }
            hidden_w.write(hv);
            embed_w.write(ev);
            hidden_d.write(hv);
            embed_d.write(ev);
        }

        RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg{};
        fill_rope_cfg<HEAD_DIM>(rope_cfg, inv_freq, t);

        eagle_tier1_lm_top(
            hidden_w, embed_w, &wrap_best_id, &wrap_best_score,
            w_q.data(), s_q.data(), w_k.data(), s_k.data(), w_v.data(), s_v.data(),
            w_o.data(), s_o.data(), w_gate.data(), s_gate.data(), w_up.data(), s_up.data(), w_down.data(), s_down.data(),
            hidden_norm.data(), embed_norm.data(), post_norm.data(), final_norm.data(), rope_cfg, hbm_k_wrap.data(),
            hbm_v_wrap.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, wrap_reason.data(),
            lm_down.data(), lm_q.data(), lm_s.data(), lm_z.empty() ? nullptr : lm_z.data(), lm_g.data(), lm_w.data(),
            rank, vocab, topk, wrap_ids.data(), wrap_logits.data(), t, t);

        eagle_tier1_lm_top_eagle4(
            hidden_d, embed_d, &dir_best_id, &dir_best_score,
            w_q.data(), s_q.data(), w_k.data(), s_k.data(), w_v.data(), s_v.data(),
            w_o.data(), s_o.data(), w_gate.data(), s_gate.data(), w_up.data(), s_up.data(), w_down.data(), s_down.data(),
            hidden_norm.data(), embed_norm.data(), post_norm.data(), final_norm.data(), rope_cfg, hbm_k_dir.data(),
            hbm_v_dir.data(), lm_down.data(), lm_q.data(), lm_s.data(), lm_z.empty() ? nullptr : lm_z.data(), lm_g.data(),
            lm_w.data(), rank, vocab, topk, dir_reason.data(), dir_ids.data(), dir_logits.data(), t, t);
    }

    int id_mismatch = 0;
    float score_diff = std::fabs(wrap_best_score - dir_best_score);
    if (wrap_best_id != dir_best_id) id_mismatch = 1;

    float max_logit_diff = 0.0f;
    int idx_mismatch = 0;
    for (int i = 0; i < topk; ++i) {
        max_logit_diff = std::max(max_logit_diff, std::fabs(wrap_logits[i] - dir_logits[i]));
        if (wrap_ids[i] != dir_ids[i]) idx_mismatch++;
    }

    std::cout << "[result] wrapper_vs_direct best_id=" << wrap_best_id << " / " << dir_best_id << "\n";
    std::cout << "[result] wrapper_vs_direct best_score_diff=" << score_diff << "\n";
    std::cout << "[result] wrapper_vs_direct candidate_idx_mismatch=" << idx_mismatch << "\n";
    std::cout << "[result] wrapper_vs_direct candidate_logits_max_abs=" << max_logit_diff << "\n";

    if (id_mismatch || score_diff > 1e-5f || idx_mismatch != 0 || max_logit_diff > 1e-5f) {
        std::cout << "[FAIL] wrapper did not match direct EAGLE4 LM path.\n";
        return 1;
    }

    std::cout << "[PASS] wrapper wiring selects EAGLE4 LM-head path and matches direct call.\n";
    return 0;
}
