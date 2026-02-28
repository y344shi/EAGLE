#include "cost_draft_tree_fused_wiring_hls.hpp"
#include "cost_draft_tree_tb_case_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace tmac::hls;

namespace {

constexpr float kDefaultEps = 1e-5f;
constexpr int kMaskFieldCount = 10;
constexpr int kGroupSize = 128;

enum MaskFieldIndex {
    kMaskIoTreeWidth = 0,
    kMaskIoVerifyNum = 1,
    kMaskIoCumuCount = 2,
    kMaskExecutedDepths = 3,
    kMaskStoppedEarly = 4,
    kMaskCumuTokens = 5,
    kMaskCumuScores = 6,
    kMaskCumuDeltas = 7,
    kMaskOutputScores = 8,
    kMaskOutputTokens = 9,
};

struct CliOptions {
    std::string case_file;
    bool dry_run = false;
    int seed = 20260226;
};

struct CaseData {
    int batch_size = 1;
    int node_top_k = 4;
    int hidden_size = 64;
    int tree_depth = 3;
    int curr_depth_start = 0;
    int prefix_len = 8;
    int max_node_count = 128;
    int max_verify_num = 64;
    int max_tree_width = 4;
    int init_tree_width = 4;
    int init_verify_num = 8;
    int init_cumu_count = 1;
    bool enable_initial_loop = true;
    int hot_vocab_size = 8192;
    bool use_hot_token_id = false;
    int efficient_lm_rank = 128;
    int efficient_lm_vocab_size = 256;
    int max_seq_tokens = 512;
    int seed = 20260226;

    float eps_abs = kDefaultEps;
    float eps_rel = kDefaultEps;
    std::string gt_mode = "synthetic";
    std::string policy_mode = "dynamic";
    std::string capture_backend;
    std::string golden_tensor_root;
    std::string packed_dir;
    std::string prefix_hbm_dtype;
    std::string prefix_hbm_k_file;
    std::string prefix_hbm_v_file;
    int prefix_hbm_token_count = 0;
    int prefix_hbm_elems_per_token = 0;

    std::vector<int64_t> step_input_tokens_init;
    std::vector<float> step_input_hidden_states_init;
    std::vector<float> step_last_layer_scores_init;
    std::vector<int64_t> step_topk_indexs_prev_init;

    std::vector<int64_t> hot_token_id;
    std::vector<float> initial_hidden_states;
    std::vector<float> initial_topk_probas;
    std::vector<int64_t> initial_topk_tokens;

    std::vector<int> policy_next_tree_width;
    std::vector<int> policy_next_verify_num;
    std::vector<int> policy_stop_signal;

    std::vector<float> recurrent_topk_probas;
    std::vector<int64_t> recurrent_topk_tokens;
    std::vector<int> expected_mask_recurrent_depth;

    std::vector<int64_t> init_legacy_cumu_tokens;
    std::vector<float> init_legacy_cumu_scores;
    std::vector<int64_t> init_legacy_cumu_deltas;
    std::vector<int64_t> init_legacy_prev_indexs;
    std::vector<int64_t> init_legacy_next_indexs;
    std::vector<int64_t> init_legacy_side_indexs;
    std::vector<float> init_legacy_output_scores;
    std::vector<int64_t> init_legacy_output_tokens;
    std::vector<float> init_legacy_work_scores;
    std::vector<float> init_legacy_sort_scores;

    int expected_io_tree_width = 0;
    int expected_io_verify_num = 0;
    int expected_io_cumu_count = 0;
    int expected_executed_depths = 0;
    bool expected_stopped_early = false;

    std::vector<int64_t> expected_cumu_tokens;
    std::vector<float> expected_cumu_scores;
    std::vector<int64_t> expected_cumu_deltas;
    std::vector<float> expected_output_scores;
    std::vector<int64_t> expected_output_tokens;
    std::vector<int> expected_mask_fields;
};

struct RuntimeState {
    std::vector<int64_t> step_input_tokens;
    std::vector<float> step_input_hidden_states;
    std::vector<float> step_last_layer_scores;
    std::vector<int64_t> step_topk_indexs_prev;
    std::vector<float> step_topk_probas_sampling;
    std::vector<int64_t> step_topk_tokens_sampling;

    std::vector<int64_t> cumu_tokens;
    std::vector<float> cumu_scores;
    std::vector<int64_t> cumu_deltas;
    std::vector<int64_t> prev_indexs;
    std::vector<int64_t> next_indexs;
    std::vector<int64_t> side_indexs;
    std::vector<float> output_scores;
    std::vector<int64_t> output_tokens;
    std::vector<float> work_scores;
    std::vector<float> sort_scores;

    std::vector<float> output_hidden_states;
    std::vector<int64_t> cache_topk_indices;

    std::vector<float> dbg_curr_layer_scores;
    std::vector<float> dbg_sort_layer_scores;
    std::vector<int64_t> dbg_sort_layer_indices;
    std::vector<int64_t> dbg_parent_indices_in_layer;
    std::vector<int64_t> dbg_remapped_topk_tokens;

    int io_tree_width = 0;
    int io_verify_num = 0;
    int io_cumu_count = 0;
    int executed_depths = 0;
    bool stopped_early = false;
};

struct SlmArtifacts {
    std::vector<pack512> w_q;
    std::vector<float> s_q;
    std::vector<pack512> w_k;
    std::vector<float> s_k;
    std::vector<pack512> w_v;
    std::vector<float> s_v;
    std::vector<pack512> w_o;
    std::vector<float> s_o;
    std::vector<pack512> w_gate;
    std::vector<float> gate_scales;
    std::vector<pack512> w_up;
    std::vector<float> up_scales;
    std::vector<pack512> w_down;
    std::vector<float> down_scales;
    std::vector<float> hidden_norm_gamma;
    std::vector<float> embed_norm_gamma;
    std::vector<float> post_attn_norm_gamma;
    std::vector<float> final_norm_gamma;
    std::vector<uint16_t> efficient_lm_head_down_proj_weight;
    std::vector<int32_t> efficient_lm_head_qweight_row_major;
    std::vector<uint16_t> efficient_lm_head_scales_row_major;
    std::vector<int32_t> efficient_lm_head_qzeros;
    std::vector<int32_t> efficient_lm_head_g_idx;
    std::vector<uint16_t> lm_head_weight;
    std::vector<vec_t<VEC_W>> hbm_k;
    std::vector<vec_t<VEC_W>> hbm_v;
    std::vector<RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>> rope_cfg_table;
};

inline int clamp_int(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

bool parse_cli(int argc, char** argv, CliOptions* opts, std::string* err_msg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--case-file") {
            if (i + 1 >= argc) {
                *err_msg = "--case-file requires a path";
                return false;
            }
            opts->case_file = argv[++i];
        } else if (arg == "--dry-run") {
            opts->dry_run = true;
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                *err_msg = "--seed requires an integer";
                return false;
            }
            try {
                opts->seed = std::stoi(argv[++i]);
            } catch (...) {
                *err_msg = "invalid integer for --seed";
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: cost_draft_tree_multilayer_orchestrator_tb [--case-file <path>]"
                << " [--dry-run] [--seed <n>]\n";
            return false;
        } else {
            *err_msg = "unknown argument: " + arg;
            return false;
        }
    }
    return true;
}

template <typename T>
std::vector<T> load_bin(const std::filesystem::path& path) {
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

std::vector<float> load_fp16(const std::filesystem::path& path) {
    auto raw = load_bin<uint16_t>(path);
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) out[i] = fp16_to_float(raw[i]);
    return out;
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
                (kOrigMaxPos / wave_len - kLowFreqFactor) /
                (kHighFreqFactor - kLowFreqFactor);
            out = (1.0f - smooth) * (inv / kScalingFactor) + smooth * inv;
        }
        inv_freq[static_cast<size_t>(i)] = out;
    }
    return inv_freq;
}

template <int HEAD_DIM_>
void fill_rope_cfg(
    RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM_>& cfg,
    const std::vector<float>& inv_freq,
    int pos) {
    for (int i = 0; i < HEAD_DIM_ / 2; ++i) {
        const float freq = static_cast<float>(pos) * inv_freq[static_cast<size_t>(i)];
        cfg.cos_vals[i] = std::cos(freq);
        cfg.sin_vals[i] = std::sin(freq);
    }
}

bool read_optional_string_scalar(const tb_case_io::RawCaseMap& kv,
                                 const std::string& key,
                                 std::string* out) {
    const auto it = kv.find(key);
    if (it == kv.end() || it->second.empty()) {
        return false;
    }
    *out = it->second[0];
    return true;
}

bool nearly_equal(float a, float b, float eps_abs, float eps_rel);
bool mask_enabled(const std::vector<int>& mask, int idx);

bool load_case_file(const std::string& path, CaseData* out, std::string* err_msg) {
    using namespace tmac::hls::tb_case_io;

    RawCaseMap kv;
    if (!parse_key_count_file(path, &kv, err_msg)) {
        return false;
    }

    std::vector<int> meta;
    if (!read_int_array(kv, "meta", 19, &meta, err_msg, true)) {
        return false;
    }

    out->batch_size = meta[0];
    out->node_top_k = meta[1];
    out->hidden_size = meta[2];
    out->tree_depth = meta[3];
    out->curr_depth_start = meta[4];
    out->prefix_len = meta[5];
    out->max_node_count = meta[6];
    out->max_verify_num = meta[7];
    out->max_tree_width = meta[8];
    out->init_tree_width = meta[9];
    out->init_verify_num = meta[10];
    out->init_cumu_count = meta[11];
    out->enable_initial_loop = (meta[12] != 0);
    out->hot_vocab_size = meta[13];
    out->use_hot_token_id = (meta[14] != 0);
    out->efficient_lm_rank = meta[15];
    out->efficient_lm_vocab_size = meta[16];
    out->max_seq_tokens = meta[17];
    out->seed = meta[18];

    std::vector<float> eps_abs;
    std::vector<float> eps_rel;
    if (!read_float_array(kv, "eps_abs", 1, &eps_abs, err_msg, false) ||
        !read_float_array(kv, "eps_rel", 1, &eps_rel, err_msg, false)) {
        return false;
    }
    if (!eps_abs.empty()) out->eps_abs = eps_abs[0];
    if (!eps_rel.empty()) out->eps_rel = eps_rel[0];

    const auto it_gt = kv.find("gt_mode");
    if (it_gt != kv.end() && !it_gt->second.empty()) {
        out->gt_mode = it_gt->second[0];
    }
    const auto it_policy = kv.find("policy_mode");
    if (it_policy != kv.end() && !it_policy->second.empty()) {
        out->policy_mode = it_policy->second[0];
    }
    read_optional_string_scalar(kv, "capture_backend", &out->capture_backend);
    read_optional_string_scalar(kv, "golden_tensor_root", &out->golden_tensor_root);
    read_optional_string_scalar(kv, "packed_dir", &out->packed_dir);
    read_optional_string_scalar(kv, "prefix_hbm_dtype", &out->prefix_hbm_dtype);
    read_optional_string_scalar(kv, "prefix_hbm_k_file", &out->prefix_hbm_k_file);
    read_optional_string_scalar(kv, "prefix_hbm_v_file", &out->prefix_hbm_v_file);

    std::vector<int> prefix_token_count;
    std::vector<int> prefix_elems_per_token;
    if (!read_int_array(kv, "prefix_hbm_token_count", 1, &prefix_token_count, err_msg, false) ||
        !read_int_array(kv, "prefix_hbm_elems_per_token", 1, &prefix_elems_per_token, err_msg, false)) {
        return false;
    }
    if (!prefix_token_count.empty()) out->prefix_hbm_token_count = prefix_token_count[0];
    if (!prefix_elems_per_token.empty()) out->prefix_hbm_elems_per_token = prefix_elems_per_token[0];

    const size_t tree_n = static_cast<size_t>(out->batch_size) * out->max_tree_width;
    const size_t hidden_n = tree_n * out->hidden_size;
    const size_t topk_stage_n = tree_n * out->node_top_k;
    const size_t recurrent_n = static_cast<size_t>(out->tree_depth) * topk_stage_n;
    const size_t node_n = static_cast<size_t>(out->batch_size) * out->max_node_count;
    const size_t out_n = static_cast<size_t>(out->batch_size) * out->node_top_k;
    const size_t work_n = static_cast<size_t>(out->batch_size) *
                          static_cast<size_t>(out->max_verify_num + out->node_top_k);
    const size_t sort_n = static_cast<size_t>(out->batch_size) * out->max_verify_num;

    int expected_stopped_early_i = 0;

    if (!read_i64_array(kv, "step_input_tokens_init", tree_n, &out->step_input_tokens_init, err_msg,
                        true) ||
        !read_float_array(kv, "step_input_hidden_states_init", hidden_n,
                          &out->step_input_hidden_states_init, err_msg, true) ||
        !read_float_array(kv, "step_last_layer_scores_init", tree_n,
                          &out->step_last_layer_scores_init, err_msg, true) ||
        !read_i64_array(kv, "step_topk_indexs_prev_init", tree_n,
                        &out->step_topk_indexs_prev_init, err_msg, true) ||
        !read_i64_array(kv, "hot_token_id", static_cast<size_t>(out->hot_vocab_size),
                        &out->hot_token_id, err_msg, true) ||
        !read_float_array(kv, "initial_hidden_states",
                          static_cast<size_t>(out->batch_size) * out->hidden_size,
                          &out->initial_hidden_states, err_msg, true) ||
        !read_float_array(kv, "initial_topk_probas", out_n, &out->initial_topk_probas, err_msg,
                          true) ||
        !read_i64_array(kv, "initial_topk_tokens", out_n, &out->initial_topk_tokens, err_msg,
                        true) ||
        !read_int_array(kv, "policy_next_tree_width", static_cast<size_t>(out->tree_depth),
                        &out->policy_next_tree_width, err_msg, true) ||
        !read_int_array(kv, "policy_next_verify_num", static_cast<size_t>(out->tree_depth),
                        &out->policy_next_verify_num, err_msg, true) ||
        !read_int_array(kv, "policy_stop_signal", static_cast<size_t>(out->tree_depth),
                        &out->policy_stop_signal, err_msg, true) ||
        !read_float_array(kv, "recurrent_topk_probas", recurrent_n, &out->recurrent_topk_probas,
                          err_msg, true) ||
        !read_i64_array(kv, "recurrent_topk_tokens", recurrent_n, &out->recurrent_topk_tokens,
                        err_msg, true) ||
        !read_int_array(kv, "expected_mask_recurrent_depth", static_cast<size_t>(out->tree_depth),
                        &out->expected_mask_recurrent_depth, err_msg, false) ||
        !read_i64_array(kv, "init_legacy_cumu_tokens", node_n, &out->init_legacy_cumu_tokens,
                        err_msg, true) ||
        !read_float_array(kv, "init_legacy_cumu_scores", node_n, &out->init_legacy_cumu_scores,
                          err_msg, true) ||
        !read_i64_array(kv, "init_legacy_cumu_deltas", node_n, &out->init_legacy_cumu_deltas,
                        err_msg, true) ||
        !read_i64_array(kv, "init_legacy_prev_indexs", node_n, &out->init_legacy_prev_indexs,
                        err_msg, true) ||
        !read_i64_array(kv, "init_legacy_next_indexs", node_n, &out->init_legacy_next_indexs,
                        err_msg, true) ||
        !read_i64_array(kv, "init_legacy_side_indexs", node_n, &out->init_legacy_side_indexs,
                        err_msg, true) ||
        !read_float_array(kv, "init_legacy_output_scores", out_n,
                          &out->init_legacy_output_scores, err_msg, true) ||
        !read_i64_array(kv, "init_legacy_output_tokens", out_n,
                        &out->init_legacy_output_tokens, err_msg, true) ||
        !read_float_array(kv, "init_legacy_work_scores", work_n,
                          &out->init_legacy_work_scores, err_msg, true) ||
        !read_float_array(kv, "init_legacy_sort_scores", sort_n,
                          &out->init_legacy_sort_scores, err_msg, true) ||
        !read_scalar_int(kv, "expected_io_tree_width", &out->expected_io_tree_width, err_msg,
                         true) ||
        !read_scalar_int(kv, "expected_io_verify_num", &out->expected_io_verify_num, err_msg,
                         true) ||
        !read_scalar_int(kv, "expected_io_cumu_count", &out->expected_io_cumu_count, err_msg,
                         true) ||
        !read_scalar_int(kv, "expected_executed_depths", &out->expected_executed_depths, err_msg,
                         true) ||
        !read_scalar_int(kv, "expected_stopped_early", &expected_stopped_early_i, err_msg,
                         true) ||
        !read_i64_array(kv, "expected_cumu_tokens", node_n, &out->expected_cumu_tokens, err_msg,
                        true) ||
        !read_float_array(kv, "expected_cumu_scores", node_n, &out->expected_cumu_scores,
                          err_msg, true) ||
        !read_i64_array(kv, "expected_cumu_deltas", node_n, &out->expected_cumu_deltas, err_msg,
                        true) ||
        !read_float_array(kv, "expected_output_scores", out_n, &out->expected_output_scores,
                          err_msg, true) ||
        !read_i64_array(kv, "expected_output_tokens", out_n, &out->expected_output_tokens,
                        err_msg, true) ||
        !read_int_array(kv, "expected_mask_fields", static_cast<size_t>(kMaskFieldCount),
                        &out->expected_mask_fields, err_msg, false)) {
        return false;
    }

    out->expected_stopped_early = (expected_stopped_early_i != 0);

    if (out->expected_mask_recurrent_depth.empty()) {
        out->expected_mask_recurrent_depth.assign(static_cast<size_t>(out->tree_depth), 0);
    }
    if (out->expected_mask_fields.empty()) {
        out->expected_mask_fields.assign(static_cast<size_t>(kMaskFieldCount), 0);
    }

    return true;
}

void make_synthetic_case(CaseData* out, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> p_dist(0.05f, 0.95f);
    std::uniform_real_distribution<float> h_dist(-1.0f, 1.0f);

    const size_t tree_n = static_cast<size_t>(out->batch_size) * out->max_tree_width;
    const size_t hidden_n = tree_n * out->hidden_size;
    const size_t topk_stage_n = tree_n * out->node_top_k;
    const size_t recurrent_n = static_cast<size_t>(out->tree_depth) * topk_stage_n;
    const size_t node_n = static_cast<size_t>(out->batch_size) * out->max_node_count;
    const size_t out_n = static_cast<size_t>(out->batch_size) * out->node_top_k;
    const size_t work_n = static_cast<size_t>(out->batch_size) *
                          static_cast<size_t>(out->max_verify_num + out->node_top_k);
    const size_t sort_n = static_cast<size_t>(out->batch_size) * out->max_verify_num;

    out->step_input_tokens_init.assign(tree_n, 0);
    out->step_last_layer_scores_init.assign(tree_n, 1.0f);
    out->step_topk_indexs_prev_init.assign(tree_n, 0);
    out->step_input_hidden_states_init.assign(hidden_n, 0.0f);
    for (size_t i = 0; i < tree_n; ++i) {
        out->step_input_tokens_init[i] = static_cast<int64_t>(i);
        out->step_last_layer_scores_init[i] = 1.0f - 0.05f * static_cast<float>(i);
        out->step_topk_indexs_prev_init[i] = static_cast<int64_t>(i);
    }
    for (float& x : out->step_input_hidden_states_init) x = h_dist(rng);

    out->hot_token_id.assign(static_cast<size_t>(out->hot_vocab_size), 0);
    for (int i = 0; i < out->hot_vocab_size; ++i) {
        out->hot_token_id[static_cast<size_t>(i)] = i;
    }

    out->initial_hidden_states.assign(static_cast<size_t>(out->batch_size) * out->hidden_size, 0.0f);
    for (float& x : out->initial_hidden_states) x = h_dist(rng);

    out->initial_topk_probas.assign(out_n, 0.0f);
    out->initial_topk_tokens.assign(out_n, 0);
    float sum = 0.0f;
    for (size_t i = 0; i < out_n; ++i) {
        out->initial_topk_probas[i] = p_dist(rng);
        sum += out->initial_topk_probas[i];
        out->initial_topk_tokens[i] = static_cast<int64_t>(rng() % std::max(1, out->hot_vocab_size));
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i < out_n; ++i) {
            out->initial_topk_probas[i] /= sum;
        }
    }

    out->policy_next_tree_width.assign(static_cast<size_t>(out->tree_depth), out->init_tree_width);
    out->policy_next_verify_num.assign(static_cast<size_t>(out->tree_depth), out->init_verify_num);
    out->policy_stop_signal.assign(static_cast<size_t>(out->tree_depth), 0);

    out->recurrent_topk_probas.assign(recurrent_n, 0.0f);
    out->recurrent_topk_tokens.assign(recurrent_n, 0);
    for (size_t i = 0; i < recurrent_n; ++i) {
        out->recurrent_topk_probas[i] = p_dist(rng);
        out->recurrent_topk_tokens[i] = static_cast<int64_t>(rng() % std::max(1, out->hot_vocab_size));
    }
    out->expected_mask_recurrent_depth.assign(static_cast<size_t>(out->tree_depth), 0);

    out->init_legacy_cumu_tokens.assign(node_n, -777);
    out->init_legacy_cumu_scores.assign(node_n, -3.0f);
    out->init_legacy_cumu_deltas.assign(node_n, -1);
    out->init_legacy_prev_indexs.assign(node_n, -1);
    out->init_legacy_next_indexs.assign(node_n, -1);
    out->init_legacy_side_indexs.assign(node_n, -1);
    out->init_legacy_output_scores.assign(out_n, -4.0f);
    out->init_legacy_output_tokens.assign(out_n, -1);
    out->init_legacy_work_scores.assign(work_n, -6.0f);
    out->init_legacy_sort_scores.assign(sort_n, -2.0f);

    out->expected_io_tree_width = out->init_tree_width;
    out->expected_io_verify_num = out->init_verify_num;
    out->expected_io_cumu_count = out->init_cumu_count;
    out->expected_executed_depths = out->tree_depth;
    out->expected_stopped_early = false;
    out->expected_cumu_tokens.assign(node_n, -1);
    out->expected_cumu_scores.assign(node_n, 0.0f);
    out->expected_cumu_deltas.assign(node_n, -1);
    out->expected_output_scores.assign(out_n, 0.0f);
    out->expected_output_tokens.assign(out_n, -1);
    out->expected_mask_fields.assign(static_cast<size_t>(kMaskFieldCount), 0);
}

bool validate_case(const CaseData& c, std::string* err_msg) {
    if (c.batch_size != 1) {
        *err_msg = "precondition failed: batch_size must be 1 in this milestone";
        return false;
    }
    if (!c.enable_initial_loop) {
        *err_msg = "precondition failed: enable_initial_loop must be true";
        return false;
    }
    if (c.tree_depth <= 1) {
        *err_msg = "precondition failed: tree_depth must be > 1";
        return false;
    }
    if (c.batch_size <= 0 || c.batch_size > kCdtFusedMaxBatch || c.node_top_k <= 0 ||
        c.hidden_size <= 0 || c.max_node_count <= 0 || c.max_verify_num <= 0 ||
        c.max_tree_width <= 0 || c.hot_vocab_size <= 0) {
        *err_msg = "invalid scalar dimensions in case meta";
        return false;
    }
    if (c.node_top_k > kCdtFusedMaxNodeTopK || c.node_top_k > kEagle4LmTopKMax ||
        c.max_tree_width > TREE_WIDTH || c.hidden_size > HIDDEN) {
        *err_msg = "case exceeds compile-time HLS limits";
        return false;
    }
    if (c.init_tree_width < 0 || c.init_tree_width > c.max_tree_width ||
        c.init_tree_width > c.node_top_k) {
        *err_msg = "invalid init_tree_width";
        return false;
    }
    if (c.init_verify_num <= 0 || c.init_verify_num > c.max_verify_num) {
        *err_msg = "invalid init_verify_num";
        return false;
    }
    if (c.init_cumu_count < 0 || c.init_cumu_count > c.max_node_count) {
        *err_msg = "invalid init_cumu_count";
        return false;
    }
    if (c.curr_depth_start < 0 || c.curr_depth_start + c.tree_depth > kCdtControllerMaxDepth) {
        *err_msg = "depth range exceeds controller depth capacity";
        return false;
    }
    if (c.prefix_len + c.curr_depth_start + c.tree_depth * c.max_tree_width >= c.max_seq_tokens) {
        *err_msg = "max_seq_tokens too small for prefix+depth+width envelope";
        return false;
    }
    if (static_cast<int>(c.hot_token_id.size()) != c.hot_vocab_size) {
        *err_msg = "hot_token_id size mismatch";
        return false;
    }
    if (static_cast<int>(c.expected_mask_fields.size()) != kMaskFieldCount) {
        *err_msg = "expected_mask_fields size mismatch";
        return false;
    }
    if (static_cast<int>(c.expected_mask_recurrent_depth.size()) != c.tree_depth) {
        *err_msg = "expected_mask_recurrent_depth size mismatch";
        return false;
    }
    if (!c.capture_backend.empty() &&
        c.capture_backend != "classic_eagle" &&
        c.capture_backend != "eagle4_classic") {
        *err_msg = "unsupported capture_backend in case: " + c.capture_backend;
        return false;
    }
    if (!c.capture_backend.empty()) {
        if (c.golden_tensor_root.empty() || c.packed_dir.empty() ||
            c.prefix_hbm_dtype.empty() || c.prefix_hbm_k_file.empty() ||
            c.prefix_hbm_v_file.empty()) {
            *err_msg = "classic-eagle case is missing artifact path metadata";
            return false;
        }
        if (c.prefix_hbm_dtype != "fp16") {
            *err_msg = "only fp16 prefix_hbm_dtype is supported";
            return false;
        }
        if (c.prefix_hbm_token_count != c.prefix_len) {
            *err_msg = "prefix_hbm_token_count must equal prefix_len";
            return false;
        }
        if (c.prefix_hbm_elems_per_token != NUM_KV_HEADS * HEAD_DIM) {
            *err_msg = "prefix_hbm_elems_per_token does not match NUM_KV_HEADS*HEAD_DIM";
            return false;
        }
    }
    return true;
}

size_t expected_pack_count(int in_dim, int out_dim) {
    return (static_cast<size_t>(in_dim) * static_cast<size_t>(out_dim)) / 128;
}

size_t expected_scale_count(int in_dim, int out_dim) {
    return static_cast<size_t>(in_dim / kGroupSize) * static_cast<size_t>(out_dim);
}

bool load_prefix_hbm_sidecar(const std::filesystem::path& case_dir,
                             const CaseData& c,
                             std::vector<vec_t<VEC_W>>* hbm_k,
                             std::vector<vec_t<VEC_W>>* hbm_v,
                             std::string* err_msg) {
    const std::filesystem::path k_path = case_dir / c.prefix_hbm_k_file;
    const std::filesystem::path v_path = case_dir / c.prefix_hbm_v_file;
    auto raw_k = load_bin<uint16_t>(k_path);
    auto raw_v = load_bin<uint16_t>(v_path);
    const size_t elems_per_token = static_cast<size_t>(c.prefix_hbm_elems_per_token);
    const size_t expected_raw = static_cast<size_t>(c.prefix_hbm_token_count) * elems_per_token;
    if (raw_k.size() != expected_raw || raw_v.size() != expected_raw) {
        *err_msg = "prefix KV sidecar size mismatch";
        return false;
    }
    const size_t vecs_per_token = elems_per_token / VEC_W;
    const size_t total_tokens =
        static_cast<size_t>(c.prefix_len + (c.curr_depth_start + c.tree_depth) * c.max_tree_width);
    hbm_k->assign(total_tokens * vecs_per_token, vec_t<VEC_W>{});
    hbm_v->assign(total_tokens * vecs_per_token, vec_t<VEC_W>{});

    for (int tok = 0; tok < c.prefix_hbm_token_count; ++tok) {
        const size_t raw_tok_base = static_cast<size_t>(tok) * elems_per_token;
        const size_t vec_tok_base = static_cast<size_t>(tok) * vecs_per_token;
        for (size_t vv = 0; vv < vecs_per_token; ++vv) {
            vec_t<VEC_W> kv;
            vec_t<VEC_W> vv_out;
            for (int lane = 0; lane < VEC_W; ++lane) {
                kv[lane] = fp16_to_float(raw_k[raw_tok_base + vv * VEC_W + static_cast<size_t>(lane)]);
                vv_out[lane] = fp16_to_float(raw_v[raw_tok_base + vv * VEC_W + static_cast<size_t>(lane)]);
            }
            (*hbm_k)[vec_tok_base + vv] = kv;
            (*hbm_v)[vec_tok_base + vv] = vv_out;
        }
    }
    return true;
}

bool load_slm_artifacts(const std::filesystem::path& case_dir,
                        const CaseData& c,
                        SlmArtifacts* a,
                        std::string* err_msg) {
    const std::filesystem::path packed_dir = c.packed_dir;
    const std::filesystem::path golden_root = c.golden_tensor_root;
    const std::filesystem::path norm_dir = golden_root / "hls_4bit" / "weights_all_4bit";
    const std::filesystem::path lm_dir = golden_root / "hls_4bit" / "lm_head";

    a->w_q = load_bin<pack512>(packed_dir / "q_proj_weights_swizzled.bin");
    a->s_q = load_bin<float>(packed_dir / "q_proj_scales_swizzled.bin");
    a->w_k = load_bin<pack512>(packed_dir / "k_proj_weights_swizzled.bin");
    a->s_k = load_bin<float>(packed_dir / "k_proj_scales_swizzled.bin");
    a->w_v = load_bin<pack512>(packed_dir / "v_proj_weights_swizzled.bin");
    a->s_v = load_bin<float>(packed_dir / "v_proj_scales_swizzled.bin");
    a->w_o = load_bin<pack512>(packed_dir / "o_proj_weights_swizzled.bin");
    a->s_o = load_bin<float>(packed_dir / "o_proj_scales_swizzled.bin");
    a->w_gate = load_bin<pack512>(packed_dir / "gate_proj_weights_swizzled.bin");
    a->gate_scales = load_bin<float>(packed_dir / "gate_proj_scales_swizzled.bin");
    a->w_up = load_bin<pack512>(packed_dir / "up_proj_weights_swizzled.bin");
    a->up_scales = load_bin<float>(packed_dir / "up_proj_scales_swizzled.bin");
    a->w_down = load_bin<pack512>(packed_dir / "down_proj_weights_swizzled.bin");
    a->down_scales = load_bin<float>(packed_dir / "down_proj_scales_swizzled.bin");

    a->hidden_norm_gamma = load_fp16(norm_dir / "hidden_norm.fp16.bin");
    a->embed_norm_gamma = load_fp16(norm_dir / "input_layernorm.fp16.bin");
    a->post_attn_norm_gamma = load_fp16(norm_dir / "post_attention_layernorm.fp16.bin");
    a->final_norm_gamma = load_fp16(norm_dir / "final_norm.fp16.bin");

    a->efficient_lm_head_down_proj_weight =
        load_bin<uint16_t>(lm_dir / "efficient_lm_head_down_proj_weight.fp16.bin");
    a->efficient_lm_head_qweight_row_major =
        load_bin<int32_t>(lm_dir / "efficient_lm_head_qweight_row_major.bin");
    a->efficient_lm_head_scales_row_major =
        load_bin<uint16_t>(lm_dir / "efficient_lm_head_scales_row_major.bin");
    a->efficient_lm_head_qzeros = load_bin<int32_t>(lm_dir / "efficient_lm_head_qzeros.bin");
    a->efficient_lm_head_g_idx = load_bin<int32_t>(lm_dir / "efficient_lm_head_g_idx.bin");
    a->lm_head_weight = load_bin<uint16_t>(lm_dir / "lm_head_weight.fp16.bin");

    const bool weights_ok =
        a->w_q.size() == expected_pack_count(QKV_INPUT, HIDDEN) &&
        a->w_k.size() == expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM) &&
        a->w_v.size() == expected_pack_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM) &&
        a->w_o.size() == expected_pack_count(HIDDEN, HIDDEN) &&
        a->w_gate.size() == expected_pack_count(HIDDEN, INTERMEDIATE) &&
        a->w_up.size() == expected_pack_count(HIDDEN, INTERMEDIATE) &&
        a->w_down.size() == expected_pack_count(INTERMEDIATE, DOWN_OUTPUT) &&
        a->s_q.size() == expected_scale_count(QKV_INPUT, HIDDEN) &&
        a->s_k.size() == expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM) &&
        a->s_v.size() == expected_scale_count(QKV_INPUT, NUM_KV_HEADS * HEAD_DIM) &&
        a->s_o.size() == expected_scale_count(HIDDEN, HIDDEN) &&
        a->gate_scales.size() == expected_scale_count(HIDDEN, INTERMEDIATE) &&
        a->up_scales.size() == expected_scale_count(HIDDEN, INTERMEDIATE) &&
        a->down_scales.size() == expected_scale_count(INTERMEDIATE, DOWN_OUTPUT);
    if (!weights_ok || a->hidden_norm_gamma.size() < HIDDEN || a->embed_norm_gamma.size() < HIDDEN ||
        a->post_attn_norm_gamma.size() < HIDDEN || a->final_norm_gamma.size() < HIDDEN ||
        a->efficient_lm_head_down_proj_weight.empty() ||
        a->efficient_lm_head_qweight_row_major.empty() ||
        a->efficient_lm_head_scales_row_major.empty() ||
        a->lm_head_weight.empty()) {
        *err_msg = "missing or invalid packed weight artifacts";
        return false;
    }

    if (!load_prefix_hbm_sidecar(case_dir, c, &a->hbm_k, &a->hbm_v, err_msg)) {
        return false;
    }

    a->rope_cfg_table.assign(
        static_cast<size_t>(std::max(kCdtControllerMaxDepth, c.curr_depth_start + c.tree_depth)),
        RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>{});
    const std::vector<float> inv_freq = build_llama3_inv_freq(HEAD_DIM);
    for (size_t d = 0; d < a->rope_cfg_table.size(); ++d) {
        fill_rope_cfg<HEAD_DIM>(
            a->rope_cfg_table[d], inv_freq, c.prefix_len + static_cast<int>(d));
    }
    return true;
}

void init_runtime(const CaseData& c, RuntimeState* s) {
    const size_t tree_n = static_cast<size_t>(c.batch_size) * c.max_tree_width;
    const size_t hidden_n = tree_n * c.hidden_size;
    const size_t topk_stage_n = tree_n * c.node_top_k;
    const size_t node_n = static_cast<size_t>(c.batch_size) * c.max_node_count;
    const size_t out_n = static_cast<size_t>(c.batch_size) * c.node_top_k;
    const size_t work_n = static_cast<size_t>(c.batch_size) *
                          static_cast<size_t>(c.max_verify_num + c.node_top_k);
    const size_t sort_n = static_cast<size_t>(c.batch_size) * c.max_verify_num;

    s->step_input_tokens = c.step_input_tokens_init;
    s->step_input_hidden_states = c.step_input_hidden_states_init;
    s->step_last_layer_scores = c.step_last_layer_scores_init;
    s->step_topk_indexs_prev = c.step_topk_indexs_prev_init;
    s->step_topk_probas_sampling.assign(topk_stage_n, 0.0f);
    s->step_topk_tokens_sampling.assign(topk_stage_n, 0);

    s->cumu_tokens = c.init_legacy_cumu_tokens;
    s->cumu_scores = c.init_legacy_cumu_scores;
    s->cumu_deltas = c.init_legacy_cumu_deltas;
    s->prev_indexs = c.init_legacy_prev_indexs;
    s->next_indexs = c.init_legacy_next_indexs;
    s->side_indexs = c.init_legacy_side_indexs;
    s->output_scores = c.init_legacy_output_scores;
    s->output_tokens = c.init_legacy_output_tokens;
    s->work_scores = c.init_legacy_work_scores;
    s->sort_scores = c.init_legacy_sort_scores;

    s->output_hidden_states.assign(out_n * c.hidden_size, 0.0f);
    s->cache_topk_indices.assign(out_n, -1);
    s->dbg_curr_layer_scores.assign(topk_stage_n, 0.0f);
    s->dbg_sort_layer_scores.assign(topk_stage_n, 0.0f);
    s->dbg_sort_layer_indices.assign(topk_stage_n, -1);
    s->dbg_parent_indices_in_layer.assign(out_n, -1);
    s->dbg_remapped_topk_tokens.assign(topk_stage_n, -1);

    s->io_tree_width = c.init_tree_width;
    s->io_verify_num = c.init_verify_num;
    s->io_cumu_count = c.init_cumu_count;
    s->executed_depths = 0;
    s->stopped_early = false;

    (void)hidden_n;
    (void)node_n;
    (void)work_n;
    (void)sort_n;
}

void schedule_for_depth(const CaseData& c,
                        int depth,
                        int curr_tree_width,
                        int curr_verify_num,
                        int* next_tree_width,
                        int* next_verify_num,
                        bool* stop_signal) {
    int idx = depth;
    if (idx < 0) idx = 0;
    if (!c.policy_next_tree_width.empty()) {
        idx = std::min(idx, static_cast<int>(c.policy_next_tree_width.size()) - 1);
        *next_tree_width = c.policy_next_tree_width[static_cast<size_t>(idx)];
    } else {
        *next_tree_width = curr_tree_width;
    }
    if (!c.policy_next_verify_num.empty()) {
        idx = std::min(depth, static_cast<int>(c.policy_next_verify_num.size()) - 1);
        idx = std::max(idx, 0);
        *next_verify_num = c.policy_next_verify_num[static_cast<size_t>(idx)];
    } else {
        *next_verify_num = curr_verify_num;
    }
    if (!c.policy_stop_signal.empty()) {
        idx = std::min(depth, static_cast<int>(c.policy_stop_signal.size()) - 1);
        idx = std::max(idx, 0);
        *stop_signal = (c.policy_stop_signal[static_cast<size_t>(idx)] != 0);
    } else {
        *stop_signal = false;
    }

    *next_tree_width = clamp_int(*next_tree_width, 0, c.max_tree_width);
    *next_tree_width = clamp_int(*next_tree_width, 0, c.node_top_k);
    *next_verify_num = clamp_int(*next_verify_num, 1, c.max_verify_num);
}

void load_recurrent_topk_for_depth(const CaseData& c,
                                   int depth,
                                   int curr_tree_width,
                                   RuntimeState* s) {
    const int per_batch_src = c.max_tree_width * c.node_top_k;
    const int per_depth_src = c.batch_size * per_batch_src;
    const int per_batch_dst = curr_tree_width * c.node_top_k;

    std::fill(s->step_topk_probas_sampling.begin(), s->step_topk_probas_sampling.end(), 0.0f);
    std::fill(s->step_topk_tokens_sampling.begin(), s->step_topk_tokens_sampling.end(), 0);

    const size_t depth_base = static_cast<size_t>(depth) * per_depth_src;
    for (int b = 0; b < c.batch_size; ++b) {
        const size_t src_base = depth_base + static_cast<size_t>(b) * per_batch_src;
        const size_t dst_base = static_cast<size_t>(b) * per_batch_dst;
        for (int i = 0; i < per_batch_dst; ++i) {
            s->step_topk_probas_sampling[dst_base + static_cast<size_t>(i)] =
                c.recurrent_topk_probas[src_base + static_cast<size_t>(i)];
            s->step_topk_tokens_sampling[dst_base + static_cast<size_t>(i)] =
                c.recurrent_topk_tokens[src_base + static_cast<size_t>(i)];
        }
    }
}

void run_reference_replay(const CaseData& c, RuntimeState* s) {
    init_runtime(c, s);

    int curr_tree_width = clamp_int(s->io_tree_width, 0, c.max_tree_width);
    curr_tree_width = clamp_int(curr_tree_width, 0, c.node_top_k);
    int curr_verify_num = clamp_int(s->io_verify_num, 1, c.max_verify_num);
    int curr_cumu_count = clamp_int(s->io_cumu_count, 0, c.max_node_count);

    int depth_done = 0;
    bool stopped = false;
    int loop_start_depth = 0;

    if (c.enable_initial_loop) {
        std::vector<float> initial_last_layer_scores(static_cast<size_t>(c.batch_size), 1.0f);
        std::vector<int64_t> initial_topk_indexs_prev(static_cast<size_t>(c.batch_size), 0);

        cost_draft_tree_fused_step_hls(
            c.initial_topk_probas.data(),
            c.initial_topk_tokens.data(),
            initial_last_layer_scores.data(),
            c.initial_hidden_states.data(),
            c.hot_token_id.data(),
            static_cast<int64_t>(c.hot_token_id.size()),
            c.use_hot_token_id,
            initial_topk_indexs_prev.data(),
            c.batch_size,
            c.node_top_k,
            1,
            c.hidden_size,
            curr_cumu_count,
            curr_verify_num,
            c.curr_depth_start + 1,
            c.max_node_count,
            c.max_verify_num,
            s->cumu_tokens.data(),
            s->cumu_scores.data(),
            s->cumu_deltas.data(),
            s->prev_indexs.data(),
            s->next_indexs.data(),
            s->side_indexs.data(),
            s->output_scores.data(),
            s->output_tokens.data(),
            s->work_scores.data(),
            s->sort_scores.data(),
            s->output_hidden_states.data(),
            s->cache_topk_indices.data(),
            s->dbg_curr_layer_scores.data(),
            s->dbg_sort_layer_scores.data(),
            s->dbg_sort_layer_indices.data(),
            s->dbg_parent_indices_in_layer.data(),
            s->dbg_remapped_topk_tokens.data());

        curr_cumu_count += c.node_top_k;
        if (curr_cumu_count > c.max_node_count) {
            curr_cumu_count = c.max_node_count;
        }
        ++depth_done;

        int next_tree_width = curr_tree_width;
        int next_verify_num = curr_verify_num;
        bool stop_signal = false;
        schedule_for_depth(
            c,
            0,
            1,
            curr_verify_num,
            &next_tree_width,
            &next_verify_num,
            &stop_signal);

        curr_tree_width = next_tree_width;
        curr_verify_num = next_verify_num;

        if (c.tree_depth <= 1 || stop_signal || next_tree_width <= 0) {
            stopped = stop_signal || (next_tree_width <= 0);
            goto reference_finalize;
        }

        cdt_prepare_next_layer_inputs_hls(
            s->output_scores.data(),
            s->output_tokens.data(),
            s->output_hidden_states.data(),
            s->cache_topk_indices.data(),
            c.batch_size,
            c.node_top_k,
            c.hidden_size,
            next_tree_width,
            c.max_tree_width,
            s->step_input_tokens.data(),
            s->step_last_layer_scores.data(),
            s->step_input_hidden_states.data(),
            s->step_topk_indexs_prev.data());

        loop_start_depth = 1;
    }

    for (int d = loop_start_depth; d < c.tree_depth; ++d) {
        if (curr_tree_width <= 0) {
            stopped = true;
            break;
        }

        int next_tree_width = curr_tree_width;
        int next_verify_num = curr_verify_num;
        bool stop_signal = false;
        if (!c.enable_initial_loop) {
            schedule_for_depth(
                c,
                d,
                curr_tree_width,
                curr_verify_num,
                &next_tree_width,
                &next_verify_num,
                &stop_signal);
        }

        load_recurrent_topk_for_depth(c, d, curr_tree_width, s);

        cost_draft_tree_fused_step_hls(
            s->step_topk_probas_sampling.data(),
            s->step_topk_tokens_sampling.data(),
            s->step_last_layer_scores.data(),
            s->step_input_hidden_states.data(),
            c.hot_token_id.data(),
            static_cast<int64_t>(c.hot_token_id.size()),
            c.use_hot_token_id,
            s->step_topk_indexs_prev.data(),
            c.batch_size,
            c.node_top_k,
            curr_tree_width,
            c.hidden_size,
            curr_cumu_count,
            curr_verify_num,
            c.enable_initial_loop ? (c.curr_depth_start + d + 1) : (c.curr_depth_start + d),
            c.max_node_count,
            c.max_verify_num,
            s->cumu_tokens.data(),
            s->cumu_scores.data(),
            s->cumu_deltas.data(),
            s->prev_indexs.data(),
            s->next_indexs.data(),
            s->side_indexs.data(),
            s->output_scores.data(),
            s->output_tokens.data(),
            s->work_scores.data(),
            s->sort_scores.data(),
            s->output_hidden_states.data(),
            s->cache_topk_indices.data(),
            s->dbg_curr_layer_scores.data(),
            s->dbg_sort_layer_scores.data(),
            s->dbg_sort_layer_indices.data(),
            s->dbg_parent_indices_in_layer.data(),
            s->dbg_remapped_topk_tokens.data());

        curr_cumu_count += curr_tree_width * c.node_top_k;
        if (curr_cumu_count > c.max_node_count) {
            curr_cumu_count = c.max_node_count;
        }

        ++depth_done;
        if (c.enable_initial_loop) {
            schedule_for_depth(
                c,
                d,
                curr_tree_width,
                curr_verify_num,
                &next_tree_width,
                &next_verify_num,
                &stop_signal);
        }

        if (d + 1 >= c.tree_depth || stop_signal || next_tree_width <= 0) {
            curr_tree_width = next_tree_width;
            curr_verify_num = next_verify_num;
            stopped = stop_signal || (next_tree_width <= 0);
            break;
        }

        cdt_prepare_next_layer_inputs_hls(
            s->output_scores.data(),
            s->output_tokens.data(),
            s->output_hidden_states.data(),
            s->cache_topk_indices.data(),
            c.batch_size,
            c.node_top_k,
            c.hidden_size,
            next_tree_width,
            c.max_tree_width,
            s->step_input_tokens.data(),
            s->step_last_layer_scores.data(),
            s->step_input_hidden_states.data(),
            s->step_topk_indexs_prev.data());

        curr_tree_width = next_tree_width;
        curr_verify_num = next_verify_num;
    }

reference_finalize:
    s->io_tree_width = curr_tree_width;
    s->io_verify_num = curr_verify_num;
    s->io_cumu_count = curr_cumu_count;
    s->executed_depths = depth_done;
    s->stopped_early = stopped;
}

void run_orchestrator_under_test(const CaseData& c,
                                 const SlmArtifacts& a,
                                 RuntimeState* s) {
    init_runtime(c, s);
    std::vector<vec_t<VEC_W>> hbm_k = a.hbm_k;
    std::vector<vec_t<VEC_W>> hbm_v = a.hbm_v;

    cost_draft_tree_multilayer_orchestrator_hls(
        c.tree_depth,
        c.curr_depth_start,
        c.policy_next_tree_width.data(),
        c.policy_next_verify_num.data(),
        c.policy_stop_signal.data(),
        c.tree_depth,
        true,
        s->step_input_tokens.data(),
        s->step_input_hidden_states.data(),
        s->step_last_layer_scores.data(),
        s->step_topk_indexs_prev.data(),
        s->step_topk_probas_sampling.data(),
        s->step_topk_tokens_sampling.data(),
        a.w_q.data(), a.s_q.data(),
        a.w_k.data(), a.s_k.data(),
        a.w_v.data(), a.s_v.data(),
        a.w_o.data(), a.s_o.data(),
        a.w_gate.data(), a.gate_scales.data(),
        a.w_up.data(), a.up_scales.data(),
        a.w_down.data(), a.down_scales.data(),
        a.hidden_norm_gamma.data(),
        a.embed_norm_gamma.data(),
        a.post_attn_norm_gamma.data(),
        a.final_norm_gamma.data(),
        a.rope_cfg_table.data(),
        hbm_k.data(),
        hbm_v.data(),
        a.efficient_lm_head_down_proj_weight.data(),
        a.efficient_lm_head_qweight_row_major.data(),
        a.efficient_lm_head_scales_row_major.data(),
        a.efficient_lm_head_qzeros.data(),
        a.efficient_lm_head_g_idx.data(),
        a.lm_head_weight.data(),
        c.efficient_lm_rank,
        c.efficient_lm_vocab_size,
        c.prefix_len,
        c.hot_token_id.data(),
        static_cast<int64_t>(c.hot_token_id.size()),
        c.use_hot_token_id,
        c.batch_size,
        c.node_top_k,
        c.hidden_size,
        &s->io_tree_width,
        &s->io_verify_num,
        &s->io_cumu_count,
        c.max_node_count,
        c.max_verify_num,
        c.max_tree_width,
        s->cumu_tokens.data(),
        s->cumu_scores.data(),
        s->cumu_deltas.data(),
        s->prev_indexs.data(),
        s->next_indexs.data(),
        s->side_indexs.data(),
        s->output_scores.data(),
        s->output_tokens.data(),
        s->work_scores.data(),
        s->sort_scores.data(),
        s->output_hidden_states.data(),
        s->cache_topk_indices.data(),
        s->dbg_curr_layer_scores.data(),
        s->dbg_sort_layer_scores.data(),
        s->dbg_sort_layer_indices.data(),
        s->dbg_parent_indices_in_layer.data(),
        s->dbg_remapped_topk_tokens.data(),
        &s->executed_depths,
        &s->stopped_early,
        c.enable_initial_loop,
        nullptr,
        nullptr,
        0,
        c.initial_topk_probas.data(),
        c.initial_topk_tokens.data(),
        c.initial_hidden_states.data());
}

bool run_slm_depth_parity(const CaseData& c,
                          const SlmArtifacts& a,
                          std::string* err_msg) {
    RuntimeState s;
    init_runtime(c, &s);
    std::vector<vec_t<VEC_W>> hbm_k = a.hbm_k;
    std::vector<vec_t<VEC_W>> hbm_v = a.hbm_v;

    int curr_tree_width = clamp_int(s.io_tree_width, 0, c.max_tree_width);
    curr_tree_width = clamp_int(curr_tree_width, 0, c.node_top_k);
    int curr_verify_num = clamp_int(s.io_verify_num, 1, c.max_verify_num);
    int curr_cumu_count = clamp_int(s.io_cumu_count, 0, c.max_node_count);
    int loop_start_depth = 0;

    int parent_indices_accum[kCdtControllerMaxDepth * TREE_WIDTH];
    std::fill(std::begin(parent_indices_accum), std::end(parent_indices_accum), 0);
    std::vector<int64_t> parent_scratch(static_cast<size_t>(c.batch_size * c.node_top_k), -1);

    if (c.enable_initial_loop) {
        std::vector<float> initial_last_layer_scores(static_cast<size_t>(c.batch_size), 1.0f);
        std::vector<int64_t> initial_topk_indexs_prev(static_cast<size_t>(c.batch_size), 0);

        cost_draft_tree_fused_step_hls(
            c.initial_topk_probas.data(),
            c.initial_topk_tokens.data(),
            initial_last_layer_scores.data(),
            c.initial_hidden_states.data(),
            c.hot_token_id.data(),
            static_cast<int64_t>(c.hot_token_id.size()),
            c.use_hot_token_id,
            initial_topk_indexs_prev.data(),
            c.batch_size,
            c.node_top_k,
            1,
            c.hidden_size,
            curr_cumu_count,
            curr_verify_num,
            c.curr_depth_start + 1,
            c.max_node_count,
            c.max_verify_num,
            s.cumu_tokens.data(),
            s.cumu_scores.data(),
            s.cumu_deltas.data(),
            s.prev_indexs.data(),
            s.next_indexs.data(),
            s.side_indexs.data(),
            s.output_scores.data(),
            s.output_tokens.data(),
            s.work_scores.data(),
            s.sort_scores.data(),
            s.output_hidden_states.data(),
            s.cache_topk_indices.data(),
            s.dbg_curr_layer_scores.data(),
            s.dbg_sort_layer_scores.data(),
            s.dbg_sort_layer_indices.data(),
            parent_scratch.data(),
            s.dbg_remapped_topk_tokens.data());

        for (int t = 0; t < TREE_WIDTH; ++t) {
            int parent_slot = 0;
            if (t < c.node_top_k) {
                const int64_t v = parent_scratch[static_cast<size_t>(t)];
                parent_slot = (v >= 0 && v < c.max_tree_width) ? static_cast<int>(v) : 0;
            }
            parent_indices_accum[c.curr_depth_start * TREE_WIDTH + t] = parent_slot;
        }

        curr_cumu_count = std::min(c.max_node_count, curr_cumu_count + c.node_top_k);

        int next_tree_width = curr_tree_width;
        int next_verify_num = curr_verify_num;
        bool stop_signal = false;
        schedule_for_depth(
            c, 0, 1, curr_verify_num, &next_tree_width, &next_verify_num, &stop_signal);
        curr_tree_width = next_tree_width;
        curr_verify_num = next_verify_num;

        if (c.tree_depth <= 1 || stop_signal || next_tree_width <= 0) {
            return true;
        }

        cdt_prepare_next_layer_inputs_hls(
            s.output_scores.data(),
            s.output_tokens.data(),
            s.output_hidden_states.data(),
            s.cache_topk_indices.data(),
            c.batch_size,
            c.node_top_k,
            c.hidden_size,
            next_tree_width,
            c.max_tree_width,
            s.step_input_tokens.data(),
            s.step_last_layer_scores.data(),
            s.step_input_hidden_states.data(),
            s.step_topk_indexs_prev.data());
        loop_start_depth = 1;
    }

    std::vector<float> slm_topk_probas(static_cast<size_t>(c.batch_size * c.max_tree_width * c.node_top_k), 0.0f);
    std::vector<int64_t> slm_topk_tokens(static_cast<size_t>(c.batch_size * c.max_tree_width * c.node_top_k), 0);

    for (int d = loop_start_depth; d < c.tree_depth; ++d) {
        if (curr_tree_width <= 0) break;
        const int current_depth = c.curr_depth_start + d;
        std::fill(slm_topk_probas.begin(), slm_topk_probas.end(), 0.0f);
        std::fill(slm_topk_tokens.begin(), slm_topk_tokens.end(), 0);

        cdt_run_eagle4_slm_topk_hls(
            s.step_input_hidden_states.data(),
            c.batch_size,
            curr_tree_width,
            c.hidden_size,
            c.node_top_k,
            a.w_q.data(), a.s_q.data(), a.w_k.data(), a.s_k.data(), a.w_v.data(), a.s_v.data(),
            a.w_o.data(), a.s_o.data(), a.w_gate.data(), a.gate_scales.data(),
            a.w_up.data(), a.up_scales.data(), a.w_down.data(), a.down_scales.data(),
            a.hidden_norm_gamma.data(), a.embed_norm_gamma.data(),
            a.post_attn_norm_gamma.data(), a.final_norm_gamma.data(),
            a.rope_cfg_table[static_cast<size_t>(current_depth)],
            hbm_k.data(),
            hbm_v.data(),
            a.efficient_lm_head_down_proj_weight.data(),
            a.efficient_lm_head_qweight_row_major.data(),
            a.efficient_lm_head_scales_row_major.data(),
            a.efficient_lm_head_qzeros.data(),
            a.efficient_lm_head_g_idx.data(),
            a.lm_head_weight.data(),
            c.efficient_lm_rank,
            c.efficient_lm_vocab_size,
            c.prefix_len,
            current_depth,
            parent_indices_accum,
            s.step_input_hidden_states.data(),
            slm_topk_probas.data(),
            slm_topk_tokens.data());

        if (mask_enabled(c.expected_mask_recurrent_depth, d)) {
            const size_t depth_base =
                static_cast<size_t>(d) * c.batch_size * c.max_tree_width * c.node_top_k;
            const int used = c.batch_size * curr_tree_width * c.node_top_k;
            for (int i = 0; i < used; ++i) {
                const float exp_p = c.recurrent_topk_probas[depth_base + static_cast<size_t>(i)];
                const int64_t exp_t = c.recurrent_topk_tokens[depth_base + static_cast<size_t>(i)];
                if (!nearly_equal(slm_topk_probas[static_cast<size_t>(i)], exp_p, c.eps_abs, c.eps_rel)) {
                    *err_msg = "slm topk prob mismatch at depth " + std::to_string(d) +
                               " index " + std::to_string(i) + " got=" +
                               std::to_string(slm_topk_probas[static_cast<size_t>(i)]) +
                               " expected=" + std::to_string(exp_p);
                    return false;
                }
                if (slm_topk_tokens[static_cast<size_t>(i)] != exp_t) {
                    *err_msg = "slm topk token mismatch at depth " + std::to_string(d) +
                               " index " + std::to_string(i) + " got=" +
                               std::to_string(slm_topk_tokens[static_cast<size_t>(i)]) +
                               " expected=" + std::to_string(exp_t);
                    return false;
                }
            }
        }

        load_recurrent_topk_for_depth(c, d, curr_tree_width, &s);
        cost_draft_tree_fused_step_hls(
            s.step_topk_probas_sampling.data(),
            s.step_topk_tokens_sampling.data(),
            s.step_last_layer_scores.data(),
            s.step_input_hidden_states.data(),
            c.hot_token_id.data(),
            static_cast<int64_t>(c.hot_token_id.size()),
            c.use_hot_token_id,
            s.step_topk_indexs_prev.data(),
            c.batch_size,
            c.node_top_k,
            curr_tree_width,
            c.hidden_size,
            curr_cumu_count,
            curr_verify_num,
            c.enable_initial_loop ? (c.curr_depth_start + d + 1) : (c.curr_depth_start + d),
            c.max_node_count,
            c.max_verify_num,
            s.cumu_tokens.data(),
            s.cumu_scores.data(),
            s.cumu_deltas.data(),
            s.prev_indexs.data(),
            s.next_indexs.data(),
            s.side_indexs.data(),
            s.output_scores.data(),
            s.output_tokens.data(),
            s.work_scores.data(),
            s.sort_scores.data(),
            s.output_hidden_states.data(),
            s.cache_topk_indices.data(),
            s.dbg_curr_layer_scores.data(),
            s.dbg_sort_layer_scores.data(),
            s.dbg_sort_layer_indices.data(),
            parent_scratch.data(),
            s.dbg_remapped_topk_tokens.data());

        if (current_depth >= 0 && current_depth < kCdtControllerMaxDepth) {
            for (int t = 0; t < TREE_WIDTH; ++t) {
                int parent_slot = 0;
                if (t < c.node_top_k) {
                    const int64_t v = parent_scratch[static_cast<size_t>(t)];
                    parent_slot = (v >= 0 && v < c.max_tree_width) ? static_cast<int>(v) : 0;
                }
                parent_indices_accum[current_depth * TREE_WIDTH + t] = parent_slot;
            }
        }

        curr_cumu_count = std::min(c.max_node_count, curr_cumu_count + curr_tree_width * c.node_top_k);
        int next_tree_width = curr_tree_width;
        int next_verify_num = curr_verify_num;
        bool stop_signal = false;
        schedule_for_depth(
            c, d, curr_tree_width, curr_verify_num, &next_tree_width, &next_verify_num, &stop_signal);
        if (d + 1 >= c.tree_depth || stop_signal || next_tree_width <= 0) {
            break;
        }
        cdt_prepare_next_layer_inputs_hls(
            s.output_scores.data(),
            s.output_tokens.data(),
            s.output_hidden_states.data(),
            s.cache_topk_indices.data(),
            c.batch_size,
            c.node_top_k,
            c.hidden_size,
            next_tree_width,
            c.max_tree_width,
            s.step_input_tokens.data(),
            s.step_last_layer_scores.data(),
            s.step_input_hidden_states.data(),
            s.step_topk_indexs_prev.data());
        curr_tree_width = next_tree_width;
        curr_verify_num = next_verify_num;
    }
    return true;
}

bool nearly_equal(float a, float b, float eps_abs, float eps_rel) {
    const float diff = std::fabs(a - b);
    const float tol = eps_abs + eps_rel * std::max(std::fabs(a), std::fabs(b));
    return diff <= tol;
}

bool compare_i64_vector(const char* name,
                        const std::vector<int64_t>& actual,
                        const std::vector<int64_t>& expected) {
    if (actual.size() != expected.size()) {
        std::cerr << "[FAIL] " << name << " size mismatch: got=" << actual.size()
                  << " expected=" << expected.size() << "\n";
        return false;
    }
    bool ok = true;
    int printed = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            ok = false;
            if (printed < 12) {
                std::cerr << "[mismatch] " << name << "[" << i << "]: got=" << actual[i]
                          << " expected=" << expected[i] << "\n";
                ++printed;
            }
        }
    }
    return ok;
}

bool compare_float_vector(const char* name,
                          const std::vector<float>& actual,
                          const std::vector<float>& expected,
                          float eps_abs,
                          float eps_rel) {
    if (actual.size() != expected.size()) {
        std::cerr << "[FAIL] " << name << " size mismatch: got=" << actual.size()
                  << " expected=" << expected.size() << "\n";
        return false;
    }
    bool ok = true;
    int printed = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!nearly_equal(actual[i], expected[i], eps_abs, eps_rel)) {
            ok = false;
            if (printed < 12) {
                std::cerr << "[mismatch] " << name << "[" << i << "]: got=" << actual[i]
                          << " expected=" << expected[i] << "\n";
                ++printed;
            }
        }
    }
    return ok;
}

template <typename T>
bool compare_scalar(const char* name, T actual, T expected) {
    if (actual != expected) {
        std::cerr << "[FAIL] " << name << " mismatch: got=" << actual
                  << " expected=" << expected << "\n";
        return false;
    }
    return true;
}

bool mask_enabled(const std::vector<int>& mask, int idx) {
    if (idx < 0 || idx >= static_cast<int>(mask.size())) return false;
    return mask[static_cast<size_t>(idx)] != 0;
}

bool run_and_compare(const CaseData& c,
                     const SlmArtifacts& artifacts,
                     std::string* err_msg) {
    RuntimeState ref_state;
    RuntimeState uut_state;

    if (!run_slm_depth_parity(c, artifacts, err_msg)) {
        return false;
    }
    run_reference_replay(c, &ref_state);
    run_orchestrator_under_test(c, artifacts, &uut_state);

    const bool is_constant_policy = (c.policy_mode == "constant");
    const int default_executed_depths = is_constant_policy ? c.tree_depth : ref_state.executed_depths;
    const bool default_stopped_early = is_constant_policy ? false : ref_state.stopped_early;

    const int exp_io_tree_width =
        mask_enabled(c.expected_mask_fields, kMaskIoTreeWidth) ? c.expected_io_tree_width
                                                                : ref_state.io_tree_width;
    const int exp_io_verify_num =
        mask_enabled(c.expected_mask_fields, kMaskIoVerifyNum) ? c.expected_io_verify_num
                                                                : ref_state.io_verify_num;
    const int exp_io_cumu_count =
        mask_enabled(c.expected_mask_fields, kMaskIoCumuCount) ? c.expected_io_cumu_count
                                                                : ref_state.io_cumu_count;
    const int exp_executed_depths =
        mask_enabled(c.expected_mask_fields, kMaskExecutedDepths) ? c.expected_executed_depths
                                                                   : default_executed_depths;
    const bool exp_stopped_early =
        mask_enabled(c.expected_mask_fields, kMaskStoppedEarly) ? c.expected_stopped_early
                                                                 : default_stopped_early;

    const std::vector<int64_t>& exp_cumu_tokens =
        mask_enabled(c.expected_mask_fields, kMaskCumuTokens) ? c.expected_cumu_tokens
                                                               : ref_state.cumu_tokens;
    const std::vector<float>& exp_cumu_scores =
        mask_enabled(c.expected_mask_fields, kMaskCumuScores) ? c.expected_cumu_scores
                                                               : ref_state.cumu_scores;
    const std::vector<int64_t>& exp_cumu_deltas =
        mask_enabled(c.expected_mask_fields, kMaskCumuDeltas) ? c.expected_cumu_deltas
                                                               : ref_state.cumu_deltas;
    const std::vector<float>& exp_output_scores =
        mask_enabled(c.expected_mask_fields, kMaskOutputScores) ? c.expected_output_scores
                                                                 : ref_state.output_scores;
    const std::vector<int64_t>& exp_output_tokens =
        mask_enabled(c.expected_mask_fields, kMaskOutputTokens) ? c.expected_output_tokens
                                                                 : ref_state.output_tokens;

    bool ok = true;
    ok &= compare_scalar("io_tree_width", uut_state.io_tree_width, exp_io_tree_width);
    ok &= compare_scalar("io_verify_num", uut_state.io_verify_num, exp_io_verify_num);
    ok &= compare_scalar("io_cumu_count", uut_state.io_cumu_count, exp_io_cumu_count);
    ok &= compare_scalar("executed_depths", uut_state.executed_depths, exp_executed_depths);
    ok &= compare_scalar("stopped_early", uut_state.stopped_early, exp_stopped_early);

    ok &= compare_i64_vector("cumu_tokens", uut_state.cumu_tokens, exp_cumu_tokens);
    ok &= compare_float_vector(
        "cumu_scores", uut_state.cumu_scores, exp_cumu_scores, c.eps_abs, c.eps_rel);
    ok &= compare_i64_vector("cumu_deltas", uut_state.cumu_deltas, exp_cumu_deltas);
    ok &= compare_float_vector(
        "output_scores", uut_state.output_scores, exp_output_scores, c.eps_abs, c.eps_rel);
    ok &= compare_i64_vector("output_tokens", uut_state.output_tokens, exp_output_tokens);

    const int strict_fields =
        static_cast<int>(std::count_if(c.expected_mask_fields.begin(),
                                       c.expected_mask_fields.end(),
                                       [](int v) { return v != 0; }));
    const int strict_depths =
        static_cast<int>(std::count_if(c.expected_mask_recurrent_depth.begin(),
                                       c.expected_mask_recurrent_depth.end(),
                                       [](int v) { return v != 0; }));
    const int cov_total = kMaskFieldCount + c.tree_depth;
    const int cov_strict = strict_fields + strict_depths;
    const float cov_pct = (cov_total > 0)
                              ? (100.0f * static_cast<float>(cov_strict) /
                                 static_cast<float>(cov_total))
                              : 0.0f;

    std::cout << "[coverage] strict_fields=" << strict_fields << "/" << kMaskFieldCount
              << " strict_recurrent_depths=" << strict_depths << "/" << c.tree_depth
              << " total_strict=" << cov_strict << "/" << cov_total
              << " (" << cov_pct << "%)\n";

    if (ok) {
        std::cout << "[PASS] cost_draft_tree_multilayer_orchestrator_tb"
                  << " gt_mode=" << c.gt_mode
                  << " policy_mode=" << c.policy_mode
                  << " executed_depths=" << uut_state.executed_depths
                  << " stopped_early=" << uut_state.stopped_early
                  << " io_tree_width=" << uut_state.io_tree_width
                  << " io_verify_num=" << uut_state.io_verify_num
                  << " io_cumu_count=" << uut_state.io_cumu_count
                  << "\n";
    }

    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions opts;
    std::string err_msg;
    if (!parse_cli(argc, argv, &opts, &err_msg)) {
        if (!err_msg.empty()) {
            std::cerr << "[FAIL] " << err_msg << "\n";
            return 1;
        }
        return 0;
    }

    CaseData c;
    if (!opts.case_file.empty()) {
        if (!load_case_file(opts.case_file, &c, &err_msg)) {
            std::cerr << "[FAIL] " << err_msg << "\n";
            return 1;
        }
    } else {
        c.seed = opts.seed;
        make_synthetic_case(&c, opts.seed);
    }

    if (!validate_case(c, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }

    if (opts.dry_run) {
        const int strict_fields = static_cast<int>(std::count_if(
            c.expected_mask_fields.begin(), c.expected_mask_fields.end(), [](int v) { return v != 0; }));
        const int strict_depths = static_cast<int>(std::count_if(
            c.expected_mask_recurrent_depth.begin(), c.expected_mask_recurrent_depth.end(),
            [](int v) { return v != 0; }));

        std::cout << "[DRY-RUN] parsed orchestrator case"
                  << " gt_mode=" << c.gt_mode
                  << " policy_mode=" << c.policy_mode
                  << " capture_backend=" << (c.capture_backend.empty() ? "none" : c.capture_backend)
                  << " dims(B,topk,hidden,depth)="
                  << c.batch_size << "," << c.node_top_k << "," << c.hidden_size << ","
                  << c.tree_depth
                  << " init(tree_width,verify,cumu)="
                  << c.init_tree_width << "," << c.init_verify_num << "," << c.init_cumu_count
                  << " strict_fields=" << strict_fields << "/" << kMaskFieldCount
                  << " strict_recurrent_depths=" << strict_depths << "/" << c.tree_depth
                  << "\n";
        return 0;
    }

    if (opts.case_file.empty()) {
        std::cerr << "[FAIL] --case-file is required for non-dry-run orchestrator parity checks\n";
        return 1;
    }
    if (c.capture_backend != "classic_eagle" && c.capture_backend != "eagle4_classic") {
        std::cerr << "[FAIL] non-dry-run orchestrator parity now requires an eagle4_classic artifact-backed case\n";
        return 1;
    }

    SlmArtifacts artifacts;
    const std::filesystem::path case_dir =
        std::filesystem::absolute(std::filesystem::path(opts.case_file)).parent_path();
    if (!load_slm_artifacts(case_dir, c, &artifacts, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }

    if (!run_and_compare(c, artifacts, &err_msg)) {
        if (!err_msg.empty()) {
            std::cerr << "[FAIL] " << err_msg << "\n";
        }
        std::cerr << "[FAIL] cost_draft_tree_multilayer_orchestrator_tb\n";
        return 1;
    }
    return 0;
}
