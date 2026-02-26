#include "cost_draft_tree_fused_wiring_hls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace tmac::hls;

namespace {

constexpr int kGroupSize = 128;
constexpr int kDefaultTreeDepth = 3;  // single draft call with depth > 1
constexpr int kDefaultSeed = 1337;
constexpr int kDefaultMaxSeqTokens = 512;
constexpr int kDefaultRank = 128;     // must be divisible by 8 and group_size
constexpr int kDefaultVocab = 256;

inline size_t expected_pack_count(int in_dim, int out_dim) {
    return (static_cast<size_t>(in_dim) * static_cast<size_t>(out_dim)) / 128;
}

inline size_t expected_scale_count(int in_dim, int out_dim) {
    return static_cast<size_t>(in_dim / kGroupSize) * static_cast<size_t>(out_dim);
}

inline uint16_t fp32_to_fp16(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                 ((mant + 0x1000u) >> 13));
}

inline bool finite_vec(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

inline bool changed_vec(const std::vector<float>& before,
                        const std::vector<float>& after,
                        float eps = 1e-7f) {
    const size_t n = std::min(before.size(), after.size());
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(before[i] - after[i]) > eps) return true;
    }
    return false;
}

inline bool changed_vec_i64(const std::vector<int64_t>& before,
                            const std::vector<int64_t>& after) {
    const size_t n = std::min(before.size(), after.size());
    for (size_t i = 0; i < n; ++i) {
        if (before[i] != after[i]) return true;
    }
    return false;
}

inline bool in_range_i64(const std::vector<int64_t>& v,
                         int64_t low_inclusive,
                         int64_t high_exclusive) {
    for (int64_t x : v) {
        if (x < low_inclusive || x >= high_exclusive) return false;
    }
    return true;
}

inline bool in_range_i64_prefix(const std::vector<int64_t>& v,
                                size_t n,
                                int64_t low_inclusive,
                                int64_t high_exclusive) {
    const size_t use_n = std::min(n, v.size());
    for (size_t i = 0; i < use_n; ++i) {
        const int64_t x = v[i];
        if (x < low_inclusive || x >= high_exclusive) return false;
    }
    return true;
}

inline std::vector<float> build_llama3_inv_freq(int head_dim) {
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

template <int HEAD_DIM_>
void fill_rope_cfg(RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM_>& cfg,
                   const std::vector<float>& inv_freq,
                   int pos) {
    for (int i = 0; i < HEAD_DIM_ / 2; ++i) {
        const float freq = static_cast<float>(pos) * inv_freq[static_cast<size_t>(i)];
        cfg.cos_vals[i] = std::cos(freq);
        cfg.sin_vals[i] = std::sin(freq);
    }
}

void fill_random_pack(std::vector<pack512>& v, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (pack512& p : v) {
        uint8_t* b = reinterpret_cast<uint8_t*>(&p);
        for (int i = 0; i < 64; ++i) {
            b[i] = static_cast<uint8_t>(byte_dist(rng));
        }
    }
}

void fill_random_float(std::vector<float>& v, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& x : v) x = dist(rng);
}

void fill_random_fp16(std::vector<uint16_t>& v, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (uint16_t& x : v) x = fp32_to_fp16(dist(rng));
}

void fill_random_qweight_row_major(std::vector<int32_t>& v, std::mt19937& rng) {
    std::uniform_int_distribution<int> nib(0, 15);
    for (int32_t& x : v) {
        int32_t packed = 0;
        for (int j = 0; j < 8; ++j) {
            packed |= (nib(rng) & 0xF) << (j * 4);
        }
        x = packed;
    }
}

struct CliOptions {
    int tree_depth = kDefaultTreeDepth;
    int stop_depth = 1;
    int seed = kDefaultSeed;
    bool run_stop_scenario = false;
    bool run_width_change_scenario = false;
    bool verbose = true;
};

struct ScenarioConfig {
    int batch_size = 1;
    int tree_depth = kDefaultTreeDepth;
    int curr_depth_start = 0;
    int node_top_k = 4;          // >1 as requested
    int hidden_size = HIDDEN;
    int init_tree_width = 4;
    int init_verify_num = 8;
    int init_cumu_count = 1;
    int max_node_count = 256;
    int max_verify_num = 64;
    int max_tree_width = TREE_WIDTH;
    int prefix_len = 8;
    int efficient_lm_rank = kDefaultRank;
    int efficient_lm_vocab_size = kDefaultVocab;
    int max_seq_tokens = kDefaultMaxSeqTokens;
};

inline int clamp_int_local(int x, int low, int high) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

struct ExpectedOutcome {
    int executed_depths = 0;
    bool stopped_early = false;
    int final_tree_width = 0;
    int final_verify_num = 0;
    int final_cumu_count = 0;
};

struct FixedNoStopPolicy {
    int fixed_tree_width = 4;
    int fixed_verify_num = 8;
    inline void operator()(
        int depth,
        int batch_size,
        int curr_tree_width,
        int node_top_k,
        int max_tree_width,
        int curr_verify_num,
        const float* work_scores,
        int max_verify_num,
        int* next_tree_width,
        int* next_verify_num,
        bool* stop_signal) const {
#pragma HLS INLINE
        (void)depth;
        (void)batch_size;
        (void)curr_tree_width;
        (void)node_top_k;
        (void)max_tree_width;
        (void)curr_verify_num;
        (void)work_scores;
        (void)max_verify_num;
        *next_tree_width = fixed_tree_width;
        *next_verify_num = fixed_verify_num;
        *stop_signal = false;
    }
};

struct StopAtDepthPolicy {
    int fixed_tree_width = 4;
    int fixed_verify_num = 8;
    int stop_depth = 1;
    inline void operator()(
        int depth,
        int batch_size,
        int curr_tree_width,
        int node_top_k,
        int max_tree_width,
        int curr_verify_num,
        const float* work_scores,
        int max_verify_num,
        int* next_tree_width,
        int* next_verify_num,
        bool* stop_signal) const {
#pragma HLS INLINE
        (void)batch_size;
        (void)curr_tree_width;
        (void)node_top_k;
        (void)max_tree_width;
        (void)curr_verify_num;
        (void)work_scores;
        (void)max_verify_num;
        *next_tree_width = fixed_tree_width;
        *next_verify_num = fixed_verify_num;
        *stop_signal = (depth >= stop_depth);
    }
};

struct WidthSchedulePolicy {
    int widths[4] = {4, 2, 4, 2};
    int sched_len = 4;
    int fixed_verify_num = 8;
    inline void operator()(
        int depth,
        int batch_size,
        int curr_tree_width,
        int node_top_k,
        int max_tree_width,
        int curr_verify_num,
        const float* work_scores,
        int max_verify_num,
        int* next_tree_width,
        int* next_verify_num,
        bool* stop_signal) const {
#pragma HLS INLINE
        (void)batch_size;
        (void)curr_tree_width;
        (void)node_top_k;
        (void)max_tree_width;
        (void)curr_verify_num;
        (void)work_scores;
        (void)max_verify_num;
        const int idx = (sched_len > 0) ? (depth % sched_len) : 0;
        *next_tree_width = widths[idx];
        *next_verify_num = fixed_verify_num;
        *stop_signal = false;
    }
};

template <typename WidthPolicy>
ExpectedOutcome simulate_expected_outcome(const ScenarioConfig& cfg, const WidthPolicy& width_policy) {
    ExpectedOutcome out;

    int curr_tree_width = clamp_int_local(cfg.init_tree_width, 0, cfg.max_tree_width);
    curr_tree_width = clamp_int_local(curr_tree_width, 0, cfg.node_top_k);
    int curr_verify_num = clamp_int_local(cfg.init_verify_num, 1, cfg.max_verify_num);
    int curr_cumu_count = clamp_int_local(cfg.init_cumu_count, 0, cfg.max_node_count);

    float dummy_work_scores[1] = {0.0f};

    for (int d = 0; d < cfg.tree_depth; ++d) {
        if (curr_tree_width <= 0) {
            out.stopped_early = true;
            break;
        }

        int next_tree_width = curr_tree_width;
        int next_verify_num = curr_verify_num;
        bool stop_signal = false;
        width_policy(
            d,
            cfg.batch_size,
            curr_tree_width,
            cfg.node_top_k,
            cfg.max_tree_width,
            curr_verify_num,
            dummy_work_scores,
            cfg.max_verify_num,
            &next_tree_width,
            &next_verify_num,
            &stop_signal);

        next_tree_width = clamp_int_local(next_tree_width, 0, cfg.max_tree_width);
        next_tree_width = clamp_int_local(next_tree_width, 0, cfg.node_top_k);
        next_verify_num = clamp_int_local(next_verify_num, 1, cfg.max_verify_num);

        curr_cumu_count += curr_tree_width * cfg.node_top_k;
        if (curr_cumu_count > cfg.max_node_count) {
            curr_cumu_count = cfg.max_node_count;
        }
        ++out.executed_depths;

        if (d + 1 >= cfg.tree_depth || stop_signal || next_tree_width <= 0) {
            out.stopped_early = stop_signal || (next_tree_width <= 0);
            break;
        }

        curr_tree_width = next_tree_width;
        curr_verify_num = next_verify_num;
    }

    out.final_tree_width = curr_tree_width;
    out.final_verify_num = curr_verify_num;
    out.final_cumu_count = curr_cumu_count;
    return out;
}

bool parse_cli(int argc, char** argv, CliOptions* opts, std::string* err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--run-stop-scenario") {
            opts->run_stop_scenario = true;
        } else if (arg == "--run-width-change-scenario") {
            opts->run_width_change_scenario = true;
        } else if (arg == "--quiet") {
            opts->verbose = false;
        } else if (arg == "--tree-depth" && i + 1 < argc) {
            opts->tree_depth = std::atoi(argv[++i]);
        } else if (arg == "--stop-depth" && i + 1 < argc) {
            opts->stop_depth = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            opts->seed = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: cost_draft_tree_multilayer_orchestrator_tb [options]\n"
                << "  --tree-depth <n>               default: 3 (must be >1)\n"
                << "  --run-stop-scenario            also run deterministic early-stop scenario\n"
                << "  --stop-depth <d>               stop depth for stop scenario (default: 1)\n"
                << "  --run-width-change-scenario    run optional width schedule scenario\n"
                << "  --seed <n>                     RNG seed (default: 1337)\n"
                << "  --quiet                        less logging\n";
            return false;
        } else {
            *err = "unknown argument: " + arg;
            return false;
        }
    }

    if (opts->tree_depth <= 1) {
        *err = "--tree-depth must be > 1 for multi-depth single-call verification";
        return false;
    }
    if (opts->stop_depth < 0) {
        *err = "--stop-depth must be >= 0";
        return false;
    }
    return true;
}

template <typename WidthPolicy>
bool run_single_call_scenario(const char* scenario_name,
                              const ScenarioConfig& cfg,
                              const WidthPolicy& width_policy,
                              bool expect_stopped_early,
                              int expected_executed_depths,
                              int seed,
                              bool verbose) {
    std::mt19937 rng(seed);

    if (cfg.batch_size != 1) {
        std::cerr << "[" << scenario_name << "] only batch_size=1 is supported by this bench.\n";
        return false;
    }

    const ExpectedOutcome expected = simulate_expected_outcome(cfg, width_policy);
    if (expected.executed_depths != expected_executed_depths) {
        std::cerr << "[" << scenario_name
                  << "] scenario expectation mismatch (executed_depths). cfg/policy predicts "
                  << expected.executed_depths << " but test expects "
                  << expected_executed_depths << "\n";
        return false;
    }
    if (expected.stopped_early != expect_stopped_early) {
        std::cerr << "[" << scenario_name
                  << "] scenario expectation mismatch (stopped_early). cfg/policy predicts "
                  << expected.stopped_early << " but test expects "
                  << expect_stopped_early << "\n";
        return false;
    }

    const int b = cfg.batch_size;
    const int topk = cfg.node_top_k;
    const int hidden = cfg.hidden_size;
    const int max_tw = cfg.max_tree_width;
    const int rank = cfg.efficient_lm_rank;
    const int vocab = cfg.efficient_lm_vocab_size;
    const int in_packs = rank / 8;
    const int groups = rank / kGroupSize;

    if (rank <= 0 || rank % 8 != 0 || rank % kGroupSize != 0) {
        std::cerr << "[" << scenario_name << "] invalid rank: " << rank << "\n";
        return false;
    }
    if (vocab <= 0 || topk <= 1 || topk > kCdtFusedMaxNodeTopK) {
        std::cerr << "[" << scenario_name << "] invalid topk/vocab values.\n";
        return false;
    }

    // Step working buffers.
    std::vector<int64_t> step_input_tokens(static_cast<size_t>(b) * max_tw, 0);
    std::vector<float> step_input_hidden_states(static_cast<size_t>(b) * max_tw * hidden, 0.0f);
    std::vector<float> step_last_layer_scores(static_cast<size_t>(b) * max_tw, 0.0f);
    std::vector<int64_t> step_topk_indexs_prev(static_cast<size_t>(b) * max_tw, 0);
    std::vector<float> step_topk_probas_sampling(static_cast<size_t>(b) * max_tw * topk, 0.0f);
    std::vector<int64_t> step_topk_tokens_sampling(static_cast<size_t>(b) * max_tw * topk, 0);

    for (int i = 0; i < max_tw; ++i) {
        step_input_tokens[static_cast<size_t>(i)] = i;
        step_last_layer_scores[static_cast<size_t>(i)] = 1.0f - 0.05f * static_cast<float>(i);
        step_topk_indexs_prev[static_cast<size_t>(i)] = i + 1;  // valid existing global indices
    }
    fill_random_float(step_input_hidden_states, rng, -0.05f, 0.05f);

    // Keep snapshots to verify recurrence-related buffers are updated.
    const std::vector<float> before_last_layer_scores = step_last_layer_scores;
    const std::vector<int64_t> before_topk_indexs_prev = step_topk_indexs_prev;
    const std::vector<float> before_input_hidden_states = step_input_hidden_states;

    // EAGLE4 tier1/LM-head weight buffers.
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
    std::vector<float> gate_scales(expected_scale_count(HIDDEN, INTERMEDIATE));
    std::vector<float> up_scales(expected_scale_count(HIDDEN, INTERMEDIATE));
    std::vector<float> down_scales(expected_scale_count(INTERMEDIATE, DOWN_OUTPUT));

    std::vector<float> hidden_norm_gamma(HIDDEN, 1.0f);
    std::vector<float> embed_norm_gamma(HIDDEN, 1.0f);
    std::vector<float> post_attn_norm_gamma(HIDDEN, 1.0f);
    std::vector<float> final_norm_gamma(HIDDEN, 1.0f);

    fill_random_pack(w_q, rng);
    fill_random_pack(w_k, rng);
    fill_random_pack(w_v, rng);
    fill_random_pack(w_o, rng);
    fill_random_pack(w_gate, rng);
    fill_random_pack(w_up, rng);
    fill_random_pack(w_down, rng);
    fill_random_float(s_q, rng, 0.001f, 0.02f);
    fill_random_float(s_k, rng, 0.001f, 0.02f);
    fill_random_float(s_v, rng, 0.001f, 0.02f);
    fill_random_float(s_o, rng, 0.001f, 0.02f);
    fill_random_float(gate_scales, rng, 0.001f, 0.02f);
    fill_random_float(up_scales, rng, 0.001f, 0.02f);
    fill_random_float(down_scales, rng, 0.001f, 0.02f);
    fill_random_float(hidden_norm_gamma, rng, 0.9f, 1.1f);
    fill_random_float(embed_norm_gamma, rng, 0.9f, 1.1f);
    fill_random_float(post_attn_norm_gamma, rng, 0.9f, 1.1f);
    fill_random_float(final_norm_gamma, rng, 0.9f, 1.1f);

    std::vector<uint16_t> efficient_lm_head_down_proj_weight(static_cast<size_t>(rank) * HIDDEN);
    std::vector<int32_t> efficient_lm_head_qweight_row_major(static_cast<size_t>(vocab) * in_packs);
    std::vector<uint16_t> efficient_lm_head_scales_row_major(static_cast<size_t>(groups) * vocab);
    std::vector<uint16_t> lm_head_weight(static_cast<size_t>(vocab) * HIDDEN);

    fill_random_fp16(efficient_lm_head_down_proj_weight, rng, -0.02f, 0.02f);
    fill_random_qweight_row_major(efficient_lm_head_qweight_row_major, rng);
    fill_random_fp16(efficient_lm_head_scales_row_major, rng, 0.001f, 0.02f);
    fill_random_fp16(lm_head_weight, rng, -0.02f, 0.02f);

    // Contiguous KV buffers.
    const size_t kv_tokens = static_cast<size_t>(cfg.max_seq_tokens);
    const size_t kv_vecs = kv_tokens * static_cast<size_t>((NUM_KV_HEADS * HEAD_DIM) / VEC_W);
    std::vector<vec_t<VEC_W>> hbm_k(kv_vecs);
    std::vector<vec_t<VEC_W>> hbm_v(kv_vecs);

    // Optional hot-token remap (disabled in this bench).
    std::vector<int64_t> hot_token_id(static_cast<size_t>(vocab), 0);
    std::iota(hot_token_id.begin(), hot_token_id.end(), 0);
    const bool use_hot_token_id = false;

    // Runtime I/O dims.
    int io_tree_width = cfg.init_tree_width;
    int io_verify_num = cfg.init_verify_num;
    int io_cumu_count = cfg.init_cumu_count;

    // Persistent legacy state.
    std::vector<int64_t> cumu_tokens(static_cast<size_t>(b) * cfg.max_node_count, -1);
    std::vector<float> cumu_scores(static_cast<size_t>(b) * cfg.max_node_count, 0.0f);
    std::vector<int64_t> cumu_deltas(static_cast<size_t>(b) * cfg.max_node_count, -1);
    std::vector<int64_t> prev_indexs(static_cast<size_t>(b) * cfg.max_node_count, -1);
    std::vector<int64_t> next_indexs(static_cast<size_t>(b) * cfg.max_node_count, -1);
    std::vector<int64_t> side_indexs(static_cast<size_t>(b) * cfg.max_node_count, -1);
    std::vector<float> output_scores(static_cast<size_t>(b) * topk, 0.0f);
    std::vector<int64_t> output_tokens(static_cast<size_t>(b) * topk, -1);
    std::vector<float> work_scores(static_cast<size_t>(b) * (cfg.max_verify_num + topk), 0.0f);
    std::vector<float> sort_scores(static_cast<size_t>(b) * cfg.max_verify_num, 0.0f);

    // Seed initial score pools similarly to the Python-side draft initialization.
    work_scores[0] = 1.0f;
    sort_scores[0] = 1.0f;

    // Per-step fused outputs / scratch.
    std::vector<float> output_hidden_states(static_cast<size_t>(b) * topk * hidden, 0.0f);
    std::vector<int64_t> cache_topk_indices(static_cast<size_t>(b) * topk, -1);
    std::vector<float> dbg_curr_layer_scores(static_cast<size_t>(b) * max_tw * topk, 0.0f);
    std::vector<float> dbg_sort_layer_scores(static_cast<size_t>(b) * max_tw * topk, 0.0f);
    std::vector<int64_t> dbg_sort_layer_indices(static_cast<size_t>(b) * max_tw * topk, -1);
    std::vector<int64_t> dbg_parent_indices_in_layer(static_cast<size_t>(b) * topk, -1);
    std::vector<int64_t> dbg_remapped_topk_tokens(static_cast<size_t>(b) * max_tw * topk, -1);

    int executed_depths = 0;
    bool stopped_early = false;

    const std::vector<float> inv_freq = build_llama3_inv_freq(HEAD_DIM);
    RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg{};
    fill_rope_cfg<HEAD_DIM>(rope_cfg, inv_freq, cfg.curr_depth_start);

    // One full draft-call equivalent: single orchestrator invocation with tree_depth > 1.
    cost_draft_tree_multilayer_orchestrator_hls(
        width_policy,
        cfg.tree_depth,
        cfg.curr_depth_start,
        step_input_tokens.data(),
        step_input_hidden_states.data(),
        step_last_layer_scores.data(),
        step_topk_indexs_prev.data(),
        step_topk_probas_sampling.data(),
        step_topk_tokens_sampling.data(),
        w_q.data(), s_q.data(),
        w_k.data(), s_k.data(),
        w_v.data(), s_v.data(),
        w_o.data(), s_o.data(),
        w_gate.data(), gate_scales.data(),
        w_up.data(), up_scales.data(),
        w_down.data(), down_scales.data(),
        hidden_norm_gamma.data(),
        embed_norm_gamma.data(),
        post_attn_norm_gamma.data(),
        final_norm_gamma.data(),
        rope_cfg,
        hbm_k.data(),
        hbm_v.data(),
        efficient_lm_head_down_proj_weight.data(),
        efficient_lm_head_qweight_row_major.data(),
        efficient_lm_head_scales_row_major.data(),
        nullptr,   // qzeros optional
        nullptr,   // g_idx optional
        lm_head_weight.data(),
        rank,
        vocab,
        cfg.prefix_len,
        hot_token_id.data(),
        static_cast<int64_t>(hot_token_id.size()),
        use_hot_token_id,
        b,
        topk,
        hidden,
        &io_tree_width,
        &io_verify_num,
        &io_cumu_count,
        cfg.max_node_count,
        cfg.max_verify_num,
        cfg.max_tree_width,
        cumu_tokens.data(),
        cumu_scores.data(),
        cumu_deltas.data(),
        prev_indexs.data(),
        next_indexs.data(),
        side_indexs.data(),
        output_scores.data(),
        output_tokens.data(),
        work_scores.data(),
        sort_scores.data(),
        output_hidden_states.data(),
        cache_topk_indices.data(),
        dbg_curr_layer_scores.data(),
        dbg_sort_layer_scores.data(),
        dbg_sort_layer_indices.data(),
        dbg_parent_indices_in_layer.data(),
        dbg_remapped_topk_tokens.data(),
        &executed_depths,
        &stopped_early);

    // Assertions requested by plan.
    if (executed_depths != expected_executed_depths) {
        std::cerr << "[" << scenario_name << "] executed_depths mismatch. got=" << executed_depths
                  << " expected=" << expected_executed_depths << "\n";
        return false;
    }
    if (stopped_early != expect_stopped_early) {
        std::cerr << "[" << scenario_name << "] stopped_early mismatch. got=" << stopped_early
                  << " expected=" << expect_stopped_early << "\n";
        return false;
    }

    if (io_cumu_count != expected.final_cumu_count) {
        std::cerr << "[" << scenario_name << "] io_cumu_count mismatch. got=" << io_cumu_count
                  << " expected=" << expected.final_cumu_count << "\n";
        return false;
    }
    if (io_tree_width != expected.final_tree_width) {
        std::cerr << "[" << scenario_name << "] io_tree_width mismatch. got=" << io_tree_width
                  << " expected=" << expected.final_tree_width << "\n";
        return false;
    }
    if (io_verify_num != expected.final_verify_num) {
        std::cerr << "[" << scenario_name << "] io_verify_num mismatch. got=" << io_verify_num
                  << " expected=" << expected.final_verify_num << "\n";
        return false;
    }

    if (!changed_vec(before_last_layer_scores, step_last_layer_scores)) {
        std::cerr << "[" << scenario_name << "] step_last_layer_scores did not update.\n";
        return false;
    }
    if (!changed_vec_i64(before_topk_indexs_prev, step_topk_indexs_prev)) {
        std::cerr << "[" << scenario_name << "] step_topk_indexs_prev did not update.\n";
        return false;
    }
    if (!changed_vec(before_input_hidden_states, step_input_hidden_states)) {
        std::cerr << "[" << scenario_name << "] step_input_hidden_states did not update.\n";
        return false;
    }

    if (!finite_vec(step_last_layer_scores) || !finite_vec(step_input_hidden_states) ||
        !finite_vec(output_scores) || !finite_vec(work_scores) || !finite_vec(sort_scores) ||
        !finite_vec(output_hidden_states)) {
        std::cerr << "[" << scenario_name << "] non-finite values detected.\n";
        return false;
    }

    if (!in_range_i64(output_tokens, 0, vocab)) {
        std::cerr << "[" << scenario_name << "] output_tokens out of vocabulary range.\n";
        return false;
    }
    if (!in_range_i64(cache_topk_indices, 0, cfg.max_node_count)) {
        std::cerr << "[" << scenario_name << "] cache_topk_indices out of node range.\n";
        return false;
    }
    const int prefix_range_n = std::max(cfg.init_tree_width, expected.final_tree_width);
    if (!in_range_i64_prefix(step_topk_indexs_prev, static_cast<size_t>(prefix_range_n), 0,
                             cfg.max_node_count)) {
        std::cerr << "[" << scenario_name << "] step_topk_indexs_prev out of node range.\n";
        return false;
    }
    if (!in_range_i64_prefix(dbg_parent_indices_in_layer, static_cast<size_t>(cfg.node_top_k), 0,
                             cfg.init_tree_width)) {
        std::cerr << "[" << scenario_name
                  << "] dbg_parent_indices_in_layer out of parent slot range.\n";
        return false;
    }

    if (verbose) {
        std::cout << "[" << scenario_name << "] PASS"
                  << " executed_depths=" << executed_depths
                  << " stopped_early=" << stopped_early
                  << " io_tree_width=" << io_tree_width
                  << " io_verify_num=" << io_verify_num
                  << " io_cumu_count=" << io_cumu_count
                  << "\n";
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions opts;
    std::string err;
    if (!parse_cli(argc, argv, &opts, &err)) {
        if (!err.empty()) {
            std::cerr << "[FAIL] " << err << "\n";
            return 1;
        }
        return 0;
    }

    ScenarioConfig cfg;
    cfg.tree_depth = opts.tree_depth;
    cfg.curr_depth_start = 0;
    cfg.batch_size = 1;
    cfg.node_top_k = 4;
    cfg.hidden_size = HIDDEN;
    cfg.init_tree_width = 4;
    cfg.init_verify_num = 8;
    cfg.init_cumu_count = 1;
    cfg.max_node_count = 256;
    cfg.max_verify_num = 64;
    cfg.max_tree_width = TREE_WIDTH;
    cfg.prefix_len = 8;
    cfg.efficient_lm_rank = kDefaultRank;
    cfg.efficient_lm_vocab_size = kDefaultVocab;
    cfg.max_seq_tokens = kDefaultMaxSeqTokens;

    if (cfg.init_tree_width > cfg.max_tree_width || cfg.init_tree_width > cfg.node_top_k) {
        std::cerr << "[FAIL] invalid width configuration.\n";
        return 1;
    }
    if (cfg.prefix_len + cfg.curr_depth_start + cfg.tree_depth * cfg.max_tree_width >=
        cfg.max_seq_tokens) {
        std::cerr << "[FAIL] max_seq_tokens too small for configured depth/prefix.\n";
        return 1;
    }

    // Scenario 1: single orchestrator call, multi-depth, no early stop.
    const FixedNoStopPolicy fixed_policy{cfg.init_tree_width, cfg.init_verify_num};
    if (!run_single_call_scenario(
            "multi_depth_no_stop",
            cfg,
            fixed_policy,
            /*expect_stopped_early=*/false,
            /*expected_executed_depths=*/cfg.tree_depth,
            opts.seed,
            opts.verbose)) {
        return 1;
    }

    // Scenario 2 (optional): deterministic early-stop behavior.
    if (opts.run_stop_scenario) {
        if (opts.stop_depth >= cfg.tree_depth - 1) {
            std::cerr << "[FAIL] --stop-depth should be < tree_depth-1 for early-stop validation.\n";
            return 1;
        }
        const StopAtDepthPolicy stop_policy{cfg.init_tree_width, cfg.init_verify_num, opts.stop_depth};
        const int expected_executed = std::min(cfg.tree_depth, opts.stop_depth + 1);
        if (!run_single_call_scenario(
                "multi_depth_stop_policy",
                cfg,
                stop_policy,
                /*expect_stopped_early=*/true,
                expected_executed,
                opts.seed + 17,
                opts.verbose)) {
            return 1;
        }
    }

    // Scenario 3 (optional): width schedule stress path.
    if (opts.run_width_change_scenario) {
        WidthSchedulePolicy schedule_policy{};
        schedule_policy.widths[0] = 4;
        schedule_policy.widths[1] = 2;
        schedule_policy.widths[2] = 4;
        schedule_policy.widths[3] = 2;
        schedule_policy.sched_len = 4;
        schedule_policy.fixed_verify_num = cfg.init_verify_num;
        if (!run_single_call_scenario(
                "multi_depth_width_schedule",
                cfg,
                schedule_policy,
                /*expect_stopped_early=*/false,
                /*expected_executed_depths=*/cfg.tree_depth,
                opts.seed + 31,
                opts.verbose)) {
            return 1;
        }
    }

    std::cout << "[PASS] cost_draft_tree_multilayer_orchestrator_tb\n";
    return 0;
}
