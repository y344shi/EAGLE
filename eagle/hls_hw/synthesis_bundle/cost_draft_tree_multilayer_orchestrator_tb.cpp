//#define TMAC_CDT_ORCH_TB_INJECT_TOPK
#include "cost_draft_tree_fused_wiring_hls.hpp"
#include "cost_draft_tree_tb_case_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace tmac::hls;

namespace {

constexpr float kDefaultEps = 1e-5f;
constexpr int kMaskFieldCount = 10;

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
            //try {
                opts->seed = std::stoi(argv[++i]);
            //} catch (...) {
            //    *err_msg = "invalid integer for --seed";
            //    return false;
            //}
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

void run_orchestrator_under_test(const CaseData& c, RuntimeState* s) {
    init_runtime(c, s);

    std::vector<pack512> dummy_pack(1);
    std::vector<float> dummy_scale(1, 1.0f);
    std::vector<uint16_t> dummy_u16(1, 0);
    std::vector<int32_t> dummy_i32(1, 0);
    std::vector<vec_t<VEC_W>> dummy_vec(1);
    RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM> rope_cfg{};

    CdtOrchTbTopkProvider provider;
    provider.recurrent_topk_probas = c.recurrent_topk_probas.data();
    provider.recurrent_topk_tokens = c.recurrent_topk_tokens.data();
    provider.depth_count = c.tree_depth;
    provider.batch_size = c.batch_size;
    provider.max_tree_width = c.max_tree_width;
    provider.node_top_k = c.node_top_k;
    provider.curr_depth_start = c.curr_depth_start;

    cdt_set_orch_tb_topk_provider(&provider);

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
        dummy_pack.data(), dummy_scale.data(),
        dummy_pack.data(), dummy_scale.data(),
        dummy_pack.data(), dummy_scale.data(),
        dummy_pack.data(), dummy_scale.data(),
        dummy_pack.data(), dummy_scale.data(),
        dummy_pack.data(), dummy_scale.data(),
        dummy_pack.data(), dummy_scale.data(),
        dummy_scale.data(),
        dummy_scale.data(),
        dummy_scale.data(),
        dummy_scale.data(),
        &rope_cfg,
        dummy_vec.data(),
        dummy_vec.data(),
        dummy_u16.data(),
        dummy_i32.data(),
        dummy_u16.data(),
        dummy_i32.data(),
        dummy_i32.data(),
        dummy_u16.data(),
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

    cdt_set_orch_tb_topk_provider(nullptr);
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

bool run_and_compare(const CaseData& c) {
    RuntimeState ref_state;
    RuntimeState uut_state;

    run_reference_replay(c, &ref_state);
    run_orchestrator_under_test(c, &uut_state);

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

    if (!run_and_compare(c)) {
        std::cerr << "[FAIL] cost_draft_tree_multilayer_orchestrator_tb\n";
        return 1;
    }
    return 0;
}
