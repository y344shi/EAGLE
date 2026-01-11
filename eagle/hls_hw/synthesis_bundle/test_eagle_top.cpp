#include "eagle_tier1_top.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace tmac::hls;

namespace {

// Max supported sequence length for local KV buffers in the harness.
constexpr int MAX_SEQ = 2048;
constexpr int VOCAB = 73448;
constexpr int GROUP_SIZE = 128;

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

// Load binary into vector<T>
template <typename T>
std::vector<T> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<T> buf(static_cast<size_t>(sz / sizeof(T)));
    if (!f.read(reinterpret_cast<char*>(buf.data()), sz)) return {};
    return buf;
}

// FP16 to float (same as reference)
float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

std::vector<float> load_fp16(const std::string& path) {
    auto raw = load_bin<uint16_t>(path);
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) out[i] = fp16_to_float(raw[i]);
    return out;
}

// Reference row-major 4-bit matvec (qweights/scales layout from weights_all_4bit).
std::vector<float> matvec_row4bit(const std::vector<float>& x,
                                  const std::vector<uint32_t>& qweight,
                                  const std::vector<float>& scales,
                                  int in_dim,
                                  int out_dim) {
    std::vector<float> y(out_dim, 0.f);
    const int in_packed = in_dim / 8;
    for (int o = 0; o < out_dim; ++o) {
        float acc = 0.f;
        for (int ip = 0; ip < in_packed; ++ip) {
            uint32_t packed = qweight[o * in_packed + ip];
            int base = ip * 8;
            int group = base / GROUP_SIZE;
            float scale = scales[group * out_dim + o];
            for (int j = 0; j < 8; ++j) {
                int val = (packed >> (j * 4)) & 0xF;
                float wt = (float)(val - 8) * scale;
                acc += wt * x[base + j];
            }
        }
        y[o] = acc;
    }
    return y;
}

// Fill RoPE config for position pos using inv_freq array (length HEAD_DIM/2)
template <int HEAD_DIM>
void fill_rope_cfg(RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& cfg, const std::vector<float>& inv_freq,
                   int pos) {
    const int half = HEAD_DIM / 2;
    for (int i = 0; i < half; ++i) {
        float freq = static_cast<float>(pos) * inv_freq[i];
        cfg.cos_vals[i] = std::cos(freq);
        cfg.sin_vals[i] = std::sin(freq);
    }
}

} // namespace

int main(int argc, char** argv) {
    // Adjust paths relative to the working directory when running the harness.
    // We keep weights in ../packed_all/ (produced by pack_all_4bit.py).
    // Paths relative to this repo root (deep_pipeline_lutmac).
    const std::string base_tensors = "../eagle_verified_pipeline_4bit/cpmcu_tensors/";
    const std::string base_weights = "../packed_all/";
    const std::string base_norms = "../eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit/";

    std::cout << "EAGLE tier1 end-to-end harness\n";

    // Inputs and golden (FP16 bins)
    auto input_all = load_fp16(base_tensors + "tensor_011_EAGLE_combined_ALL.bin");
    auto golden_all = load_fp16(base_tensors + "tensor_014_EAGLE_after_residual.bin");
    auto embed_all = load_fp16(base_tensors + "tensor_006_EAGLE_INPUT_prev_embed_ALL.bin");
    auto hidden_all = load_fp16(base_tensors + "tensor_007_EAGLE_INPUT_prev_hidden_ALL.bin");
    if (input_all.size() < HIDDEN) {
        std::cout << "Input tensor missing or too small; got " << input_all.size() << "\n";
        return 1;
    }
    const int total_tokens = static_cast<int>(input_all.size() / HIDDEN);
    const int golden_tokens = static_cast<int>(golden_all.size() / HIDDEN);
    // We assume the single-token golden corresponds to the *final* token of the prefill.
    const int target_token = (total_tokens > 0) ? (total_tokens - 1) : 0;
    std::cout << "Total tokens in input: " << total_tokens << ", golden tokens: " << golden_tokens
              << ", target token index " << target_token << "\n";

    std::vector<float> golden;
    bool have_golden = false;
    if (golden_tokens == 1 && golden_all.size() == HIDDEN) {
        // Treat the lone golden vector as the final-token reference.
        golden = golden_all;
        have_golden = true;
    } else if (golden_all.size() >= static_cast<size_t>((target_token + 1) * HIDDEN)) {
        golden.resize(HIDDEN);
        std::copy_n(golden_all.begin() + target_token * HIDDEN, HIDDEN, golden.begin());
        have_golden = true;
    }

    // Load packed weights/scales (must be generated by pack_all_4bit.py)
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

    // Load norms (FP16 -> float)
    auto norm1 = load_fp16(base_norms + "input_layernorm.fp16.bin");
    auto norm2 = load_fp16(base_norms + "post_attention_layernorm.fp16.bin");
    auto norm_fc1 = load_fp16(base_norms + "input_norm1.fp16.bin");
    auto norm_fc2 = load_fp16(base_norms + "input_norm2.fp16.bin");
    // Row-major qweights/scales for reference matvecs (LM head and block projections)
    auto o_proj_qw_row = load_bin<uint32_t>(base_norms + "o_proj_qweight.bin");
    auto o_proj_sc_row = load_fp16(base_norms + "o_proj_scales.bin");
    auto gate_qw_row = load_bin<uint32_t>(base_norms + "gate_proj_qweight.bin");
    auto gate_sc_row = load_fp16(base_norms + "gate_proj_scales.bin");
    auto up_qw_row = load_bin<uint32_t>(base_norms + "up_proj_qweight.bin");
    auto up_sc_row = load_fp16(base_norms + "up_proj_scales.bin");
    auto down_qw_row = load_bin<uint32_t>(base_norms + "down_proj_qweight.bin");
    auto down_sc_row = load_fp16(base_norms + "down_proj_scales.bin");
    // inv_freq with LongRoPE factor from CPU reference
    static const float LONGROPE_FACTOR[64] = {
        0.9977997200264581f, 1.014658295992452f, 1.0349680404997148f, 1.059429246056193f,
        1.0888815016813513f, 1.1243301355211495f, 1.166977103606075f, 1.2182568066927284f,
        1.2798772354275727f, 1.3538666751582975f, 1.4426259039919596f, 1.5489853358570191f,
        1.6762658237220625f, 1.8283407612492941f, 2.0096956085876183f, 2.225478927469756f,
        2.481536379650452f, 2.784415934557119f, 3.1413289096347365f, 3.560047844772632f,
        4.048719380066383f, 4.615569542115128f, 5.2684819496549835f, 6.014438591970396f,
        6.858830049237097f, 7.804668263503327f, 8.851768731513417f, 9.99600492938444f,
        11.228766118181639f, 12.536757560834843f, 13.902257701387796f, 15.303885189125953f,
        16.717837610115794f, 18.119465097853947f, 19.484965238406907f, 20.792956681060105f,
        22.02571786985731f, 23.16995406772833f, 24.217054535738416f, 25.16289275000465f,
        26.007284207271347f, 26.753240849586767f, 27.40615325712662f, 27.973003419175363f,
        28.461674954469114f, 28.880393889607006f, 29.237306864684626f, 29.540186419591297f,
        29.79624387177199f, 30.01202719065413f, 30.193382037992453f, 30.34545697551969f,
        30.47273746338473f, 30.579096895249787f, 30.66785612408345f, 30.741845563814174f,
        30.80346599254902f, 30.85474569563567f, 30.897392663720595f, 30.932841297560394f,
        30.962293553185553f, 30.986754758742034f, 31.007064503249293f, 31.02392307921529f};
    auto inv_freq = std::vector<float>(HEAD_DIM / 2);
    for (int i = 0; i < HEAD_DIM / 2; ++i) {
        float base_freq = 1.0f / std::pow(10000.0f, (2.0f * i) / HEAD_DIM);
        inv_freq[i] = base_freq / LONGROPE_FACTOR[i];
    }

    // Sanity check
    if (w_q.empty() || s_q.empty() || norm1.size() < HIDDEN || norm2.size() < HIDDEN) {
        std::cout << "Missing weights/scales/norms; run pack_all_4bit.py and ensure files exist in "
                  << base_weights << "\n";
        std::cout << "sizes: w_q=" << w_q.size() << " s_q=" << s_q.size()
                  << " norm1=" << norm1.size() << " norm2=" << norm2.size() << "\n";
        return 1;
    }

    // Quick CPU-side RMSNorm sanity for token 0
    {
        std::cout << "\n[debug] CPU RMSNorm token0 first 8 vals\n";
        const float eps = 1e-6f;
        float sumsq = 0.f;
        for (int i = 0; i < HIDDEN; ++i) {
            float v = input_all[i];
            sumsq += v * v;
        }
        float rms = 1.0f / std::sqrt(sumsq / HIDDEN + eps);
        std::cout << "  input[0..7]: ";
        for (int i = 0; i < 8; ++i) std::cout << input_all[i] << " ";
        std::cout << "\n  gamma[0..7]: ";
        for (int i = 0; i < 8; ++i) std::cout << norm1[i] << " ";
        std::cout << "\n  rms=" << rms << "\n  norm_out[0..7]: ";
        for (int i = 0; i < 8; ++i) std::cout << input_all[i] * rms * norm1[i] << " ";
        std::cout << "\n  q_scale[0]=" << (s_q.empty() ? 0.f : s_q[0]) << "\n";
        if (!w_q.empty()) {
            auto b = reinterpret_cast<const uint8_t*>(&w_q[0]);
            std::cout << "  w_q[0] nibbles: ";
            for (int i = 0; i < 8; ++i) {
                uint8_t v = b[i];
                std::cout << int(v & 0xF) << "/" << int((v >> 4) & 0xF) << " ";
            }
            std::cout << "\n";
        }
        // Recompute combined token0 from embed/hidden using packed FC1/FC2 to confirm tensor_011 contents.
        if (embed_all.size() >= HIDDEN && hidden_all.size() >= HIDDEN) {
            hls_stream<vec_t<VEC_W>> e_in, h_in, fc1_out, fc2_out;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> ce, ch;
                for (int j = 0; j < VEC_W; ++j) {
                    ce[j] = embed_all[i * VEC_W + j];
                    ch[j] = hidden_all[i * VEC_W + j];
                }
                e_in.write(ce);
                h_in.write(ch);
            }
            hls_stream<vec_t<VEC_W>> norm_fc1_out, norm_fc2_out;
            rms_norm_stream<HIDDEN>(e_in, norm_fc1_out, norm_fc1.data());
            rms_norm_stream<HIDDEN>(h_in, norm_fc2_out, norm_fc2.data());
            dense_projection_production_scaled<0, HIDDEN, HIDDEN>(norm_fc1_out, fc1_out,
                                                                   load_bin<pack512>(base_weights + "fc1_weights_swizzled.bin").data(),
                                                                   load_bin<float>(base_weights + "fc1_scales_swizzled.bin").data());
            dense_projection_production_scaled<0, HIDDEN, HIDDEN>(norm_fc2_out, fc2_out,
                                                                   load_bin<pack512>(base_weights + "fc2_weights_swizzled.bin").data(),
                                                                   load_bin<float>(base_weights + "fc2_scales_swizzled.bin").data());
            std::vector<float> combined0(HIDDEN);
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> a = fc1_out.read();
                vec_t<VEC_W> b = fc2_out.read();
                for (int j = 0; j < VEC_W; ++j) combined0[i * VEC_W + j] = a[j] + b[j];
            }
            std::cout << "  combined0[0..7] recomputed: ";
            for (int i = 0; i < 8; ++i) std::cout << combined0[i] << " ";
            std::cout << "\n  tensor011[0..7]: ";
            for (int i = 0; i < 8; ++i) std::cout << input_all[i] << " ";
            std::cout << "\n";
        }
    }

    // KV cache backing
    std::vector<vec_t<VEC_W>> hbm_k(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);
    std::vector<vec_t<VEC_W>> hbm_v(MAX_SEQ * (NUM_KV_HEADS * HEAD_DIM) / VEC_W);

    std::vector<float> out_vec;
    RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg_target{};

    // Simulate tokens sequentially to build KV history; capture target_token output.
    for (int t = 0; t <= target_token; ++t) {
        std::vector<float> input(HIDDEN);
        std::copy_n(input_all.begin() + t * HIDDEN, HIDDEN, input.begin());

        hls_stream<vec_t<VEC_W>> in_stream, out_stream;
        for (int i = 0; i < HIDDEN / VEC_W; ++i) {
            vec_t<VEC_W> chunk;
            for (int j = 0; j < VEC_W; ++j) chunk[j] = input[i * VEC_W + j];
            in_stream.write(chunk);
        }

        RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg{};
        fill_rope_cfg<HEAD_DIM>(rope_cfg, inv_freq, t);

        eagle_tier1_top(in_stream, out_stream, w_q.data(), s_q.data(), w_k.data(), s_k.data(),
                        w_v.data(), s_v.data(), w_o.data(), s_o.data(), w_gate.data(), s_gate.data(),
                        w_up.data(), s_up.data(), w_down.data(), s_down.data(), norm1.data(),
                        norm2.data(), rope_cfg, hbm_k.data(), hbm_v.data(), t, t);

        if (t == target_token) {
            out_vec.reserve(HIDDEN);
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> chunk = out_stream.read();
                for (int j = 0; j < VEC_W; ++j) out_vec.push_back(chunk[j]);
            }
            rope_cfg_target = rope_cfg;
        } else {
            // Drain output to keep streams clean
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                (void)out_stream.read();
            }
        }

        // Debug dump for target token: recompute key stages and attention head0 using the cached history.
        if (t == target_token) {
            std::cout << "[debug] Token" << t << " stage dumps (first 8 floats each)\n";
            auto print_vec = [](const std::vector<float>& v, const char* name) {
                std::cout << "  " << name << ":";
                for (int i = 0; i < 8 && i < (int)v.size(); ++i) std::cout << " " << v[i];
                std::cout << "\n";
            };

            // Norm1
            hls_stream<vec_t<VEC_W>> s_in_dbg, s_norm_dbg;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> chunk;
                for (int j = 0; j < VEC_W; ++j) chunk[j] = input[i * VEC_W + j];
                s_in_dbg.write(chunk);
            }
            rms_norm_stream<HIDDEN>(s_in_dbg, s_norm_dbg, norm1.data());
            std::vector<float> norm_dump;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto c = s_norm_dbg.read();
                for (int j = 0; j < VEC_W; ++j) norm_dump.push_back(c[j]);
            }
            print_vec(norm_dump, "norm1_tgt");

            // Q/K/V projections
            hls_stream<vec_t<VEC_W>> s_q_in_dbg, s_k_in_dbg, s_v_in_dbg;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = norm_dump[i * VEC_W + j];
                s_q_in_dbg.write(c);
                s_k_in_dbg.write(c);
                s_v_in_dbg.write(c);
            }
            hls_stream<vec_t<VEC_W>> s_q_proj_dbg, s_k_proj_dbg, s_v_proj_dbg;
            dense_projection_production_scaled<0, HIDDEN, HIDDEN>(s_q_in_dbg, s_q_proj_dbg, w_q.data(), s_q.data());
            dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_k_in_dbg, s_k_proj_dbg, w_k.data(), s_k.data());
            dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_v_in_dbg, s_v_proj_dbg, w_v.data(), s_v.data());

            std::vector<float> q_proj_dump, k_proj_dump, v_proj_dump;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto c = s_q_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) q_proj_dump.push_back(c[j]);
            }
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                auto c = s_k_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) k_proj_dump.push_back(c[j]);
            }
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                auto c = s_v_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) v_proj_dump.push_back(c[j]);
            }
            print_vec(q_proj_dump, "q_proj_tgt");
            print_vec(k_proj_dump, "k_proj_tgt");
            print_vec(v_proj_dump, "v_proj_tgt");

            // RoPE on target
            hls_stream<vec_t<VEC_W>> s_q_proj_re, s_k_proj_re, s_q_rot_dbg, s_k_rot_dbg;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = q_proj_dump[i * VEC_W + j];
                s_q_proj_re.write(c);
            }
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = k_proj_dump[i * VEC_W + j];
                s_k_proj_re.write(c);
            }
            rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>(s_q_proj_re, s_q_rot_dbg, s_k_proj_re, s_k_rot_dbg, rope_cfg_target);
            std::vector<float> q_rot_dump, k_rot_dump;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto c = s_q_rot_dbg.read();
                for (int j = 0; j < VEC_W; ++j) q_rot_dump.push_back(c[j]);
            }
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                auto c = s_k_rot_dbg.read();
                for (int j = 0; j < VEC_W; ++j) k_rot_dump.push_back(c[j]);
            }
            print_vec(q_rot_dump, "q_rot_tgt");
            print_vec(k_rot_dump, "k_rot_tgt");

            // Build history streams for KV head 0 from cached HBM/URAM data.
            hls_stream<vec_t<VEC_W>> k_hist_dbg, v_hist_dbg, q_head_dbg, ctx_dbg;
            // Q head 0 from q_rot_dump
            for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = q_rot_dump[i * VEC_W + j];
                q_head_dbg.write(c);
            }
            // History tokens 0..t, KV head 0 block (first 8 vecs per token)
            const int vecs_per_token = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
            const int vecs_per_head = HEAD_DIM / VEC_W;
            for (int tok = 0; tok <= t; ++tok) {
                int base = tok * vecs_per_token;
                for (int v = 0; v < vecs_per_head; ++v) {
                    k_hist_dbg.write(hbm_k[base + v]);
                    v_hist_dbg.write(hbm_v[base + v]);
                }
            }
            attention_solver<HEAD_DIM>(q_head_dbg, k_hist_dbg, v_hist_dbg, ctx_dbg, t + 1);
            std::vector<float> ctx_dump;
            for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
                auto c = ctx_dbg.read();
                for (int j = 0; j < VEC_W; ++j) ctx_dump.push_back(c[j]);
            }
            print_vec(ctx_dump, "context_head0_tgt");
        }

        // Debug dump for token 0: run a mirror pipeline (no KV replay) and print first few values.
        if (t == 0) {
            std::cout << "[debug] Token0 stage dumps (first 8 floats each)\n";
            auto print_vec = [](const std::vector<float>& v, const char* name) {
                std::cout << "  " << name << ":";
                for (int i = 0; i < 8 && i < (int)v.size(); ++i) std::cout << " " << v[i];
                std::cout << "\n";
            };

            // Stage: Norm1
            hls_stream<vec_t<VEC_W>> s_in_dbg, s_norm_dbg;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> chunk;
                for (int j = 0; j < VEC_W; ++j) chunk[j] = input[i * VEC_W + j];
                s_in_dbg.write(chunk);
            }
            rms_norm_stream<HIDDEN>(s_in_dbg, s_norm_dbg, norm1.data());
            std::vector<float> norm_dump;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto c = s_norm_dbg.read();
                for (int j = 0; j < VEC_W; ++j) norm_dump.push_back(c[j]);
            }
            print_vec(norm_dump, "norm1");

            // Q/K/V proj
            hls_stream<vec_t<VEC_W>> s_q_in_dbg, s_k_in_dbg, s_v_in_dbg;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = norm_dump[i * VEC_W + j];
                s_q_in_dbg.write(c);
                s_k_in_dbg.write(c);
                s_v_in_dbg.write(c);
            }
            hls_stream<vec_t<VEC_W>> s_q_proj_dbg, s_k_proj_dbg, s_v_proj_dbg;
            dense_projection_production_scaled<0, HIDDEN, HIDDEN>(s_q_in_dbg, s_q_proj_dbg, w_q.data(), s_q.data());
            dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_k_in_dbg, s_k_proj_dbg, w_k.data(), s_k.data());
            dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_v_in_dbg, s_v_proj_dbg, w_v.data(), s_v.data());

            std::vector<float> q_proj_dump, k_proj_dump, v_proj_dump;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto c = s_q_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) q_proj_dump.push_back(c[j]);
            }
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                auto c = s_k_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) k_proj_dump.push_back(c[j]);
            }
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                auto c = s_v_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) v_proj_dump.push_back(c[j]);
            }
            print_vec(q_proj_dump, "q_proj");
            print_vec(k_proj_dump, "k_proj");
            print_vec(v_proj_dump, "v_proj");

            // Refill streams for RoPE (since we consumed projections above)
            hls_stream<vec_t<VEC_W>> s_q_proj_re, s_k_proj_re, s_q_rot_dbg, s_k_rot_dbg;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = q_proj_dump[i * VEC_W + j];
                s_q_proj_re.write(c);
            }
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = k_proj_dump[i * VEC_W + j];
                s_k_proj_re.write(c);
            }
            rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>(s_q_proj_re, s_q_rot_dbg, s_k_proj_re, s_k_rot_dbg, rope_cfg_target);
            std::vector<float> q_rot_dump, k_rot_dump;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto c = s_q_rot_dbg.read();
                for (int j = 0; j < VEC_W; ++j) q_rot_dump.push_back(c[j]);
            }
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                auto c = s_k_rot_dbg.read();
                for (int j = 0; j < VEC_W; ++j) k_rot_dump.push_back(c[j]);
            }
            print_vec(q_rot_dump, "q_rot");
            print_vec(k_rot_dump, "k_rot");

            // Attention for head 0 only, seq_len=1
            hls_stream<vec_t<VEC_W>> q_head_dbg, k_hist_dbg, v_hist_dbg, ctx_dbg;
            for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = q_rot_dump[i * VEC_W + j];
                q_head_dbg.write(c);
            }
            // history = this token's K/V
            for (int i = 0; i < (NUM_KV_HEADS * HEAD_DIM) / VEC_W; ++i) {
                vec_t<VEC_W> ck, cv;
                for (int j = 0; j < VEC_W; ++j) {
                    ck[j] = k_rot_dump[i * VEC_W + j];
                    cv[j] = v_proj_dump[i * VEC_W + j];
                }
                // pick KV head 0 block only
                if (i < (HEAD_DIM / VEC_W)) {
                    k_hist_dbg.write(ck);
                    v_hist_dbg.write(cv);
                }
            }
            attention_solver<HEAD_DIM>(q_head_dbg, k_hist_dbg, v_hist_dbg, ctx_dbg, 1);
            std::vector<float> ctx_dump;
            for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
                auto c = ctx_dbg.read();
                for (int j = 0; j < VEC_W; ++j) ctx_dump.push_back(c[j]);
            }
            print_vec(ctx_dump, "context_head0");
        }
    }

    std::cout << "Produced " << out_vec.size() << " floats\n";
    if (have_golden && out_vec.size() == HIDDEN) {
        std::vector<float> final_out(HIDDEN);
        for (int i = 0; i < HIDDEN; ++i) {
            final_out[i] = input_all[target_token * HIDDEN + i] + out_vec[i] * RESIDUAL_SCALE;
        }

        float max_diff = 0.f;
        int max_idx = 0;
        for (int i = 0; i < HIDDEN; ++i) {
            float d = std::fabs(final_out[i] - golden[i]);
            if (d > max_diff) {
                max_diff = d;
                max_idx = i;
            }
        }
        std::cout << "Max diff vs golden: " << max_diff << "\n";
        std::cout << "first 8 final_out: ";
        for (int i = 0; i < 8; ++i) std::cout << final_out[i] << " ";
        std::cout << "\nfirst 8 golden:     ";
        for (int i = 0; i < 8; ++i) std::cout << golden[i] << " ";
        std::cout << "\n";
        float dmax = std::fabs(final_out[max_idx] - golden[max_idx]);
        std::cout << "  worst idx " << max_idx << " final=" << final_out[max_idx]
                  << " golden=" << golden[max_idx] << " diff=" << dmax << "\n";
        // Print a small window around the worst index for localization
        const int window = 4;
        int start = std::max(0, max_idx - window);
        int end = std::min(HIDDEN - 1, max_idx + window);
        std::cout << "  window [" << start << "," << end << "] final / golden:\n";
        for (int i = start; i <= end; ++i) {
            std::cout << "    " << i << ": " << final_out[i] << " / " << golden[i] << "\n";
        }
        // Also compare transformer_out vs inferred golden transformer_out in the window
        std::cout << "  window transformer_out vs inferred_golden (no residual):\n";
        for (int i = start; i <= end; ++i) {
            float inferred_golden = (golden[i] - input_all[target_token * HIDDEN + i]) / RESIDUAL_SCALE;
            std::cout << "    " << i << ": " << out_vec[i] << " / " << inferred_golden << "\n";
        }
        // Compute attention-only debug for head 25 to see if divergence arises before FFN
        if (max_idx >= HEAD_DIM * 25 && max_idx < HEAD_DIM * 26) {
            const int head = 25;
            const int kv_idx = head / (NUM_HEADS / NUM_KV_HEADS); // 1
            const int vecs_per_token = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
            const int vecs_per_head = HEAD_DIM / VEC_W;
            hls_stream<vec_t<VEC_W>> q_head_dbg, k_hist_dbg, v_hist_dbg, ctx_dbg;
            // build q for head
            hls_stream<vec_t<VEC_W>> s_q_proj_re, s_k_proj_re, s_q_rot_dbg, s_k_rot_dbg;
            // re-run projections for target token
            hls_stream<vec_t<VEC_W>> s_q_in_dbg, s_k_in_dbg, s_v_in_dbg;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = norm1[i * VEC_W + j];
                s_q_in_dbg.write(c);
                s_k_in_dbg.write(c);
                s_v_in_dbg.write(c);
            }
            hls_stream<vec_t<VEC_W>> s_q_proj_dbg, s_k_proj_dbg, s_v_proj_dbg;
            dense_projection_production_scaled<0, HIDDEN, HIDDEN>(s_q_in_dbg, s_q_proj_dbg, w_q.data(), s_q.data());
            dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_k_in_dbg, s_k_proj_dbg, w_k.data(), s_k.data());
            dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_v_in_dbg, s_v_proj_dbg, w_v.data(), s_v.data());
            std::vector<float> q_proj_dump, k_proj_dump, v_proj_dump;
            q_proj_dump.reserve(HIDDEN);
            k_proj_dump.reserve(NUM_KV_HEADS * HEAD_DIM);
            v_proj_dump.reserve(NUM_KV_HEADS * HEAD_DIM);
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto c = s_q_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) q_proj_dump.push_back(c[j]);
            }
            for (int i = 0; i < vecs_per_token; ++i) {
                auto c = s_k_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) k_proj_dump.push_back(c[j]);
            }
            for (int i = 0; i < vecs_per_token; ++i) {
                auto c = s_v_proj_dbg.read();
                for (int j = 0; j < VEC_W; ++j) v_proj_dump.push_back(c[j]);
            }
            // RoPE
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = q_proj_dump[i * VEC_W + j];
                s_q_proj_re.write(c);
            }
            for (int i = 0; i < vecs_per_token; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = k_proj_dump[i * VEC_W + j];
                s_k_proj_re.write(c);
            }
            rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>(s_q_proj_re, s_q_rot_dbg, s_k_proj_re, s_k_rot_dbg, rope_cfg_target);
            std::vector<float> q_rot_dump, k_rot_dump;
            for (int i = 0; i < HIDDEN / VEC_W; ++i) {
                auto c = s_q_rot_dbg.read();
                for (int j = 0; j < VEC_W; ++j) q_rot_dump.push_back(c[j]);
            }
            for (int i = 0; i < vecs_per_token; ++i) {
                auto c = s_k_rot_dbg.read();
                for (int j = 0; j < VEC_W; ++j) k_rot_dump.push_back(c[j]);
            }
            for (int i = 0; i < vecs_per_head; ++i) {
                vec_t<VEC_W> c;
                for (int j = 0; j < VEC_W; ++j) c[j] = q_rot_dump[head * HEAD_DIM + i * VEC_W + j];
                q_head_dbg.write(c);
            }
            for (int tok = 0; tok <= target_token; ++tok) {
                int base = tok * vecs_per_token + kv_idx * vecs_per_head;
                for (int v = 0; v < vecs_per_head; ++v) {
                    k_hist_dbg.write(hbm_k[base + v]);
                    v_hist_dbg.write(hbm_v[base + v]);
                }
            }
            attention_solver<HEAD_DIM>(q_head_dbg, k_hist_dbg, v_hist_dbg, ctx_dbg, target_token + 1);
            std::vector<float> ctx_dump;
            for (int i = 0; i < vecs_per_head; ++i) {
                auto c = ctx_dbg.read();
                for (int j = 0; j < VEC_W; ++j) ctx_dump.push_back(c[j]);
            }
            std::cout << "  head25 context first 8: ";
            for (int i = 0; i < 8; ++i) std::cout << ctx_dump[i] << " ";
            std::cout << "\n";

            // CPU attention for head25 using cached HBM (kv_idx=1) to quantify drift vs. hardware attention
            std::vector<float> q_head(HEAD_DIM);
            for (int i = 0; i < HEAD_DIM; ++i) q_head[i] = q_rot_dump[head * HEAD_DIM + i];

            const int hist_len = target_token + 1;
            std::vector<float> scores(hist_len);
            std::vector<float> ctx_cpu(HEAD_DIM, 0.f);
            const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
            float max_score = -1e9f;
            for (int tok = 0; tok < hist_len; ++tok) {
                float score = 0.f;
                int base = tok * vecs_per_token + kv_idx * vecs_per_head;
                for (int v = 0; v < vecs_per_head; ++v) {
                    vec_t<VEC_W> kv = hbm_k[base + v];
                    for (int j = 0; j < VEC_W; ++j) {
                        int idx = v * VEC_W + j;
                        score += q_head[idx] * kv[j];
                    }
                }
                score *= scale;
                scores[tok] = score;
                if (score > max_score) max_score = score;
            }
            float denom = 0.f;
            for (int tok = 0; tok < hist_len; ++tok) {
                scores[tok] = std::exp(scores[tok] - max_score);
                denom += scores[tok];
            }
            for (int tok = 0; tok < hist_len; ++tok) scores[tok] /= denom;
            for (int tok = 0; tok < hist_len; ++tok) {
                int base = tok * vecs_per_token + kv_idx * vecs_per_head;
                for (int v = 0; v < vecs_per_head; ++v) {
                    vec_t<VEC_W> vv = hbm_v[base + v];
                    for (int j = 0; j < VEC_W; ++j) {
                        int idx = v * VEC_W + j;
                        ctx_cpu[idx] += scores[tok] * vv[j];
                    }
                }
            }
            float max_diff_ctx = 0.f;
            int max_idx_ctx = 0;
            for (int i = 0; i < HEAD_DIM; ++i) {
                float d = std::fabs(ctx_dump[i] - ctx_cpu[i]);
                if (d > max_diff_ctx) {
                    max_diff_ctx = d;
                    max_idx_ctx = i;
                }
            }
            std::cout << "  head25 CPU ctx max_diff " << max_diff_ctx << " at elem " << max_idx_ctx
                      << " hw=" << ctx_dump[max_idx_ctx] << " cpu=" << ctx_cpu[max_idx_ctx] << "\n";
        }

        // Full-block CPU recompute using packed weights/scales to localize FFN/O projection drift.
        {
            const float EPS = 1e-5f;
            auto rms_norm_cpu = [&](const std::vector<float>& x, const std::vector<float>& gamma) {
                std::vector<float> out(HIDDEN);
                float sumsq = 0.f;
                for (float v : x) sumsq += v * v;
                float inv = 1.0f / std::sqrt(sumsq / HIDDEN + EPS);
                for (int i = 0; i < HIDDEN; ++i) out[i] = x[i] * inv * gamma[i];
                return out;
            };
            auto vec_to_stream = [&](const std::vector<float>& v, hls_stream<vec_t<VEC_W>>& s) {
                for (size_t i = 0; i < v.size(); i += VEC_W) {
                    vec_t<VEC_W> c;
                    for (int j = 0; j < VEC_W; ++j) c[j] = v[i + j];
                    s.write(c);
                }
            };
            auto stream_to_vec = [&](hls_stream<vec_t<VEC_W>>& s, int chunks) {
                std::vector<float> v;
                v.reserve(chunks * VEC_W);
                for (int i = 0; i < chunks; ++i) {
                    auto c = s.read();
                    for (int j = 0; j < VEC_W; ++j) v.push_back(c[j]);
                }
                return v;
            };

            // Norm1
            std::vector<float> input_tok(HIDDEN);
            std::copy_n(input_all.begin() + target_token * HIDDEN, HIDDEN, input_tok.begin());
            auto norm1_cpu = rms_norm_cpu(input_tok, norm1);

            // Q/K/V projections
            hls_stream<vec_t<VEC_W>> s_q_in, s_k_in, s_v_in, s_q_proj_cpu, s_k_proj_cpu, s_v_proj_cpu;
            vec_to_stream(norm1_cpu, s_q_in);
            vec_to_stream(norm1_cpu, s_k_in);
            vec_to_stream(norm1_cpu, s_v_in);
            dense_projection_production_scaled<0, HIDDEN, HIDDEN>(s_q_in, s_q_proj_cpu, w_q.data(), s_q.data());
            dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_k_in, s_k_proj_cpu, w_k.data(), s_k.data());
            dense_projection_production_scaled<0, HIDDEN, NUM_KV_HEADS * HEAD_DIM>(s_v_in, s_v_proj_cpu, w_v.data(), s_v.data());
            auto q_proj_cpu = stream_to_vec(s_q_proj_cpu, HIDDEN / VEC_W);
            auto k_proj_cpu = stream_to_vec(s_k_proj_cpu, (NUM_KV_HEADS * HEAD_DIM) / VEC_W);
            auto v_proj_cpu = stream_to_vec(s_v_proj_cpu, (NUM_KV_HEADS * HEAD_DIM) / VEC_W);

            // RoPE
            hls_stream<vec_t<VEC_W>> s_q_proj_re, s_k_proj_re, s_q_rot_cpu, s_k_rot_cpu;
            vec_to_stream(q_proj_cpu, s_q_proj_re);
            vec_to_stream(k_proj_cpu, s_k_proj_re);
            rope_apply_stream<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>(s_q_proj_re, s_q_rot_cpu, s_k_proj_re, s_k_rot_cpu, rope_cfg_target);
            auto q_rot_cpu = stream_to_vec(s_q_rot_cpu, HIDDEN / VEC_W);

            // Attention per head using cached HBM
            const int vecs_per_token = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
            const int vecs_per_head = HEAD_DIM / VEC_W;
            const int hist_len = target_token + 1;
            std::vector<float> ctx_all(NUM_HEADS * HEAD_DIM);
            for (int h = 0; h < NUM_HEADS; ++h) {
                const int kv_idx = h / (NUM_HEADS / NUM_KV_HEADS);
                hls_stream<vec_t<VEC_W>> q_hs, k_hs, v_hs, c_hs;
                for (int i = 0; i < vecs_per_head; ++i) {
                    vec_t<VEC_W> c;
                    for (int j = 0; j < VEC_W; ++j) c[j] = q_rot_cpu[h * HEAD_DIM + i * VEC_W + j];
                    q_hs.write(c);
                }
                for (int t = 0; t < hist_len; ++t) {
                    int base = t * vecs_per_token + kv_idx * vecs_per_head;
                    for (int v = 0; v < vecs_per_head; ++v) {
                        k_hs.write(hbm_k[base + v]);
                        v_hs.write(hbm_v[base + v]);
                    }
                }
                attention_solver<HEAD_DIM>(q_hs, k_hs, v_hs, c_hs, hist_len);
                for (int i = 0; i < vecs_per_head; ++i) {
                    auto c = c_hs.read();
                    for (int j = 0; j < VEC_W; ++j) ctx_all[h * HEAD_DIM + i * VEC_W + j] = c[j];
                }
            }

            // Output projection
            hls_stream<vec_t<VEC_W>> ctx_stream, o_proj_cpu_stream;
            vec_to_stream(ctx_all, ctx_stream);
            dense_projection_production_scaled<0, HIDDEN, HIDDEN>(ctx_stream, o_proj_cpu_stream, w_o.data(), s_o.data());
            auto o_proj_cpu = stream_to_vec(o_proj_cpu_stream, HIDDEN / VEC_W);

            // Residual add/scale
            std::vector<float> res1_cpu(HIDDEN);
            for (int i = 0; i < HIDDEN; ++i) res1_cpu[i] = input_tok[i] + o_proj_cpu[i] * RESIDUAL_SCALE;

            // Norm2
            auto norm2_cpu = rms_norm_cpu(res1_cpu, norm2);

            // Gate/Up projections
            hls_stream<vec_t<VEC_W>> s_gate_in, s_up_in, s_gate_out, s_up_out;
            vec_to_stream(norm2_cpu, s_gate_in);
            vec_to_stream(norm2_cpu, s_up_in);
            dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE>(s_gate_in, s_gate_out, w_gate.data(), s_gate.data());
            dense_projection_production_scaled<0, HIDDEN, INTERMEDIATE>(s_up_in, s_up_out, w_up.data(), s_up.data());
            auto gate_vec = stream_to_vec(s_gate_out, INTERMEDIATE / VEC_W);
            auto up_vec = stream_to_vec(s_up_out, INTERMEDIATE / VEC_W);

            // SwiGLU
            auto silu = [](float x) { return x / (1.0f + std::exp(-x)); };
            std::vector<float> swiglu(INTERMEDIATE);
            for (int i = 0; i < INTERMEDIATE; ++i) swiglu[i] = silu(gate_vec[i]) * up_vec[i];

            // Down projection
            hls_stream<vec_t<VEC_W>> s_swiglu, s_down_out;
            vec_to_stream(swiglu, s_swiglu);
            dense_projection_production_scaled<0, INTERMEDIATE, HIDDEN>(s_swiglu, s_down_out, w_down.data(), s_down.data());
            auto cpu_block = stream_to_vec(s_down_out, HIDDEN / VEC_W);

            // Compare CPU recompute vs HLS out_vec
            float max_diff_cpu = 0.f;
            int max_idx_cpu = 0;
            for (int i = 0; i < HIDDEN; ++i) {
                float d = std::fabs(cpu_block[i] - out_vec[i]);
                if (d > max_diff_cpu) {
                    max_diff_cpu = d;
                    max_idx_cpu = i;
                }
            }
            std::cout << "  [cpu_full] max diff vs HLS: " << max_diff_cpu << " at idx " << max_idx_cpu
                      << " cpu=" << cpu_block[max_idx_cpu] << " hls=" << out_vec[max_idx_cpu] << "\n";
            const int w = 4;
            int st = std::max(0, max_idx_cpu - w);
            int ed = std::min(HIDDEN - 1, max_idx_cpu + w);
            std::cout << "  [cpu_full] window [" << st << "," << ed << "] cpu / hls:\n";
            for (int i = st; i <= ed; ++i) {
                std::cout << "    " << i << ": " << cpu_block[i] << " / " << out_vec[i] << "\n";
            }

            // Row-major vs packed projection check (o_proj/gate/up/down)
            {
                auto row_o = matvec_row4bit(ctx_all, o_proj_qw_row, o_proj_sc_row, HIDDEN, HIDDEN);
                std::vector<float> row_res1(HIDDEN);
                for (int i = 0; i < HIDDEN; ++i) row_res1[i] = input_tok[i] + row_o[i] * RESIDUAL_SCALE;
                auto row_norm2 = rms_norm_cpu(row_res1, norm2);
                auto row_gate = matvec_row4bit(row_norm2, gate_qw_row, gate_sc_row, HIDDEN, INTERMEDIATE);
                auto row_up = matvec_row4bit(row_norm2, up_qw_row, up_sc_row, HIDDEN, INTERMEDIATE);
                std::vector<float> row_swiglu(INTERMEDIATE);
                for (int i = 0; i < INTERMEDIATE; ++i) row_swiglu[i] = silu(row_gate[i]) * row_up[i];
                auto row_down = matvec_row4bit(row_swiglu, down_qw_row, down_sc_row, INTERMEDIATE, HIDDEN);

                auto cmp_vecs = [](const std::vector<float>& a, const std::vector<float>& b, const char* name) {
                    float max_diff = 0.f;
                    int max_idx = 0;
                    for (int i = 0; i < (int)a.size(); ++i) {
                        float d = std::fabs(a[i] - b[i]);
                        if (d > max_diff) {
                            max_diff = d;
                            max_idx = i;
                        }
                    }
                    std::cout << "  [row_vs_packed] " << name << " max diff " << max_diff << " at idx " << max_idx
                              << " row=" << a[max_idx] << " packed=" << b[max_idx] << "\n";
                };

                cmp_vecs(row_o, o_proj_cpu, "o_proj");
                cmp_vecs(row_down, cpu_block, "down_proj");
            }
        }

        // =========================
        // LM Head check vs golden logits (tensor_005_BASE_Logits.bin)
        // =========================
        auto embed_tokens = load_fp16(base_norms + "embed_tokens.fp16.bin");
        auto golden_logits = load_fp16(base_tensors + "tensor_005_BASE_Logits.bin");
        if (embed_tokens.size() == static_cast<size_t>(VOCAB * HIDDEN) &&
            golden_logits.size() == static_cast<size_t>(VOCAB)) {
            std::vector<float> logits(VOCAB, 0.f);
            for (int v = 0; v < VOCAB; ++v) {
                float acc = 0.f;
                const float* wrow = embed_tokens.data() + v * HIDDEN;
                for (int i = 0; i < HIDDEN; ++i) acc += wrow[i] * final_out[i];
                logits[v] = acc;
            }
            float max_diff_logits = 0.f;
            int max_idx_logits = 0;
            for (int v = 0; v < VOCAB; ++v) {
                float d = std::fabs(logits[v] - golden_logits[v]);
                if (d > max_diff_logits) {
                    max_diff_logits = d;
                    max_idx_logits = v;
                }
            }
            auto topk = [&](const std::vector<float>& v, int k) {
                std::vector<std::pair<float, int>> tmp;
                tmp.reserve(v.size());
                for (int i = 0; i < (int)v.size(); ++i) tmp.push_back({v[i], i});
                std::partial_sort(tmp.begin(), tmp.begin() + k, tmp.end(),
                                  [](auto& a, auto& b) { return a.first > b.first; });
                std::vector<int> ids;
                for (int i = 0; i < k; ++i) ids.push_back(tmp[i].second);
                return ids;
            };
            auto ours_top5 = topk(logits, 5);
            auto golden_top5 = topk(golden_logits, 5);
            std::cout << "LM head: max diff logits " << max_diff_logits << " at vocab " << max_idx_logits
                      << " ours=" << logits[max_idx_logits] << " golden=" << golden_logits[max_idx_logits] << "\n";
            std::cout << "LM head top1 ours=" << ours_top5[0] << " golden=" << golden_top5[0] << "\n";
            std::cout << "LM head top5 ours: ";
            for (int id : ours_top5) std::cout << id << " ";
            std::cout << "\nLM head top5 golden: ";
            for (int id : golden_top5) std::cout << id << " ";
            std::cout << "\n";
        } else {
            std::cout << "LM head check skipped (missing embed_tokens or golden logits)\n";
        }

    } else {
        std::cout << "Golden tensor missing; skipped comparison.\n";
    }

    return 0;
}
