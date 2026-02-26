#include "cost_draft_tree_fused_wiring_hls.hpp"
#include "cost_draft_tree_score_hls.hpp"
#include "cost_draft_tree_tb_case_io.hpp"
#include "cost_draft_tree_update_hls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct LegacyState {
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
};

struct TestCfg {
    int batch_size = 2;
    int node_top_k = 4;
    int tree_width = 4;
    int hidden_size = 16;
    int cumu_count = 8;
    int verify_num = 8;
    int curr_depth = 2;

    int max_node_count = 128;
    int max_verify_num = 16;
    int max_tree_width = 4;

    int hot_vocab_size = 512;
    bool use_hot_token_id = true;
};

struct FusedTestInputs {
    std::vector<float> topk_probas;
    std::vector<int64_t> topk_tokens;
    std::vector<float> last_layer_scores;
    std::vector<float> input_hidden_states;
    std::vector<int64_t> hot_token_id;
    std::vector<int64_t> topk_indexs_prev;
};

struct FusedExpectedOutputs {
    bool has_expected = false;

    LegacyState legacy;

    std::vector<float> output_hidden;
    std::vector<int64_t> cache_topk;
    std::vector<float> dbg_curr;
    std::vector<float> dbg_sort;
    std::vector<int64_t> dbg_sort_idx;
    std::vector<int64_t> dbg_parent;
    std::vector<int64_t> dbg_remap;
};

struct CliOptions {
    std::string case_file;
    bool dry_run = false;
    int multi_depth_steps = 1;
};

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float e = std::fabs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

static size_t mismatch_i64(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    size_t c = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++c;
    }
    return c;
}

static LegacyState make_legacy_state(const TestCfg& cfg, int seed) {
    LegacyState s;

    const size_t node_n = static_cast<size_t>(cfg.batch_size) * cfg.max_node_count;
    const size_t out_n = static_cast<size_t>(cfg.batch_size) * cfg.node_top_k;
    const size_t work_n =
        static_cast<size_t>(cfg.batch_size) * (cfg.max_verify_num + cfg.node_top_k);
    const size_t sort_n = static_cast<size_t>(cfg.batch_size) * cfg.max_verify_num;

    s.cumu_tokens.assign(node_n, -999);
    s.cumu_scores.assign(node_n, -5.0f);
    s.cumu_deltas.assign(node_n, -1);
    s.prev_indexs.assign(node_n, -1);
    s.next_indexs.assign(node_n, -1);
    s.side_indexs.assign(node_n, -1);
    s.output_scores.assign(out_n, -7.0f);
    s.output_tokens.assign(out_n, -3);
    s.work_scores.assign(work_n, -9.0f);
    s.sort_scores.assign(sort_n, -11.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> score_dist(0.5f, 2.0f);

    const int ws0 = std::min(cfg.verify_num, cfg.cumu_count);
    for (int b = 0; b < cfg.batch_size; ++b) {
        for (int i = 0; i < ws0; ++i) {
            s.sort_scores[b * cfg.max_verify_num + i] = score_dist(rng);
        }
        std::sort(s.sort_scores.begin() + b * cfg.max_verify_num,
                  s.sort_scores.begin() + b * cfg.max_verify_num + ws0,
                  std::greater<float>());
    }

    return s;
}

static void run_reference_pipeline(
    const TestCfg& cfg,
    const std::vector<float>& topk_probas,
    const std::vector<int64_t>& topk_tokens,
    const std::vector<float>& last_layer_scores,
    const std::vector<float>& input_hidden_states,
    const std::vector<int64_t>& hot_token_id,
    const std::vector<int64_t>& topk_indexs_prev,
    LegacyState* legacy,
    std::vector<float>* output_hidden_states,
    std::vector<int64_t>* cache_topk_indices,
    std::vector<float>* dbg_curr_scores,
    std::vector<float>* dbg_sort_scores,
    std::vector<int64_t>* dbg_sort_indices,
    std::vector<int64_t>* dbg_parent_idx,
    std::vector<int64_t>* dbg_remapped_tokens) {

    const int total_topk = cfg.tree_width * cfg.node_top_k;

    std::vector<float> stage_curr(cfg.batch_size * total_topk, 0.0f);
    std::vector<float> stage_sort(cfg.batch_size * total_topk, 0.0f);
    std::vector<int64_t> stage_sort_idx(cfg.batch_size * total_topk, -1);
    std::vector<int64_t> stage_cache_topk(cfg.batch_size * cfg.node_top_k, -1);
    std::vector<int64_t> stage_parent(cfg.batch_size * cfg.node_top_k, -1);
    std::vector<int64_t> stage_remapped_tokens(cfg.batch_size * total_topk, -1);
    std::vector<int64_t> stage_output_tokens(cfg.batch_size * cfg.node_top_k, -1);

    tmac::hls::cost_draft_tree_layer_score_hls_with_tokens(
        topk_probas.data(),
        topk_tokens.data(),
        last_layer_scores.data(),
        input_hidden_states.data(),
        hot_token_id.data(),
        static_cast<int64_t>(hot_token_id.size()),
        cfg.use_hot_token_id,
        cfg.batch_size,
        cfg.node_top_k,
        cfg.tree_width,
        cfg.hidden_size,
        cfg.cumu_count,
        stage_curr.data(),
        stage_sort.data(),
        stage_sort_idx.data(),
        stage_cache_topk.data(),
        stage_parent.data(),
        output_hidden_states->data(),
        stage_remapped_tokens.data(),
        stage_output_tokens.data());

    tmac::hls::cost_draft_tree_update_state_hls(
        topk_probas.data(),
        stage_remapped_tokens.data(),
        stage_sort.data(),
        stage_sort_idx.data(),
        stage_parent.data(),
        topk_indexs_prev.data(),
        cfg.batch_size,
        cfg.node_top_k,
        cfg.tree_width,
        cfg.cumu_count,
        cfg.verify_num,
        cfg.curr_depth,
        cfg.max_node_count,
        cfg.max_verify_num,
        legacy->cumu_tokens.data(),
        legacy->cumu_scores.data(),
        legacy->cumu_deltas.data(),
        legacy->prev_indexs.data(),
        legacy->next_indexs.data(),
        legacy->side_indexs.data(),
        legacy->output_scores.data(),
        legacy->output_tokens.data(),
        legacy->work_scores.data(),
        legacy->sort_scores.data());

    *cache_topk_indices = stage_cache_topk;
    *dbg_curr_scores = stage_curr;
    *dbg_sort_scores = stage_sort;
    *dbg_sort_indices = stage_sort_idx;
    *dbg_parent_idx = stage_parent;
    *dbg_remapped_tokens = stage_remapped_tokens;
}

static bool parse_cli(int argc, char** argv, CliOptions* opts, std::string* err_msg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-file") {
            if (i + 1 >= argc) {
                *err_msg = "--case-file requires a path";
                return false;
            }
            opts->case_file = argv[++i];
        } else if (arg == "--dry-run") {
            opts->dry_run = true;
        } else if (arg == "--multi-depth-steps") {
            if (i + 1 >= argc) {
                *err_msg = "--multi-depth-steps requires an integer";
                return false;
            }
            try {
                opts->multi_depth_steps = std::stoi(argv[++i]);
            } catch (...) {
                *err_msg = "invalid integer for --multi-depth-steps";
                return false;
            }
            if (opts->multi_depth_steps <= 0) {
                *err_msg = "--multi-depth-steps must be > 0";
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: cost_draft_tree_fused_wiring_tb [--case-file <path>] [--dry-run]"
                << " [--multi-depth-steps <N>]\n";
            return false;
        } else {
            *err_msg = "unknown argument: " + arg;
            return false;
        }
    }
    return true;
}

static bool load_file_fixture(const std::string& path,
                              TestCfg* cfg,
                              FusedTestInputs* in,
                              LegacyState* legacy_init,
                              FusedExpectedOutputs* expected,
                              std::string* err_msg) {
    using namespace tmac::hls::tb_case_io;
    RawCaseMap kv;
    if (!parse_key_count_file(path, &kv, err_msg)) {
        return false;
    }

    std::vector<int> meta;
    if (!read_int_array(kv, "meta", 16, &meta, err_msg, true)) {
        return false;
    }

    cfg->batch_size = meta[0];
    cfg->node_top_k = meta[1];
    cfg->tree_width = meta[2];
    cfg->hidden_size = meta[3];
    cfg->cumu_count = meta[4];
    // meta[5] (input_count) and meta[8] (max_input_size) removed
    cfg->verify_num = meta[6];
    cfg->curr_depth = meta[7];
    cfg->max_node_count = meta[9];
    cfg->max_verify_num = meta[10];
    cfg->max_tree_width = meta[11];
    // meta[12] (parent_width) and meta[13] (next_tree_width) removed
    cfg->hot_vocab_size = meta[14];
    cfg->use_hot_token_id = (meta[15] != 0);

    if (cfg->batch_size <= 0 || cfg->node_top_k <= 0 || cfg->tree_width <= 0 ||
        cfg->hidden_size <= 0 ||
        cfg->max_node_count <= 0 || cfg->max_verify_num <= 0 || cfg->max_tree_width <= 0) {
        *err_msg = "invalid scalar dimensions in meta";
        return false;
    }

    const int total_topk = cfg->tree_width * cfg->node_top_k;
    const size_t topk_n = static_cast<size_t>(cfg->batch_size) * total_topk;
    const size_t score_n = static_cast<size_t>(cfg->batch_size) * cfg->tree_width;
    const size_t hidden_n =
        static_cast<size_t>(cfg->batch_size) * cfg->tree_width * cfg->hidden_size;
    const size_t tree_n = static_cast<size_t>(cfg->batch_size) * cfg->tree_width;
    const size_t node_n = static_cast<size_t>(cfg->batch_size) * cfg->max_node_count;
    const size_t out_n = static_cast<size_t>(cfg->batch_size) * cfg->node_top_k;
    const size_t work_n = static_cast<size_t>(cfg->batch_size) *
                          static_cast<size_t>(cfg->max_verify_num + cfg->node_top_k);
    const size_t sort_n = static_cast<size_t>(cfg->batch_size) * cfg->max_verify_num;

    if (!read_float_array(kv, "topk_probas_sampling", topk_n, &in->topk_probas, err_msg, true) ||
        !read_i64_array(kv, "topk_tokens_sampling", topk_n, &in->topk_tokens, err_msg, true) ||
        !read_float_array(kv, "last_layer_scores", score_n, &in->last_layer_scores, err_msg, true) ||
        !read_float_array(kv, "input_hidden_states", hidden_n, &in->input_hidden_states, err_msg,
                          true) ||
        !read_i64_array(kv, "topk_indexs_prev", tree_n, &in->topk_indexs_prev, err_msg, true)) {
        return false;
    }

    if (has_key(kv, "hot_token_id")) {
        if (!read_i64_array(kv, "hot_token_id", static_cast<size_t>(cfg->hot_vocab_size),
                            &in->hot_token_id, err_msg, true)) {
            return false;
        }
    } else {
        in->hot_token_id.assign(static_cast<size_t>(cfg->hot_vocab_size), 0);
        for (int i = 0; i < cfg->hot_vocab_size; ++i) {
            in->hot_token_id[static_cast<size_t>(i)] = i;
        }
    }

    legacy_init->cumu_tokens.assign(node_n, -1);
    legacy_init->cumu_scores.assign(node_n, 0.0f);
    legacy_init->cumu_deltas.assign(node_n, -1);
    legacy_init->prev_indexs.assign(node_n, -1);
    legacy_init->next_indexs.assign(node_n, -1);
    legacy_init->side_indexs.assign(node_n, -1);
    legacy_init->output_scores.assign(out_n, 0.0f);
    legacy_init->output_tokens.assign(out_n, -1);
    legacy_init->work_scores.assign(work_n, 0.0f);
    legacy_init->sort_scores.assign(sort_n, 0.0f);

    if (!read_i64_array(kv, "legacy_cumu_tokens", node_n, &legacy_init->cumu_tokens, err_msg,
                        true) ||
        !read_float_array(kv, "legacy_cumu_scores", node_n, &legacy_init->cumu_scores, err_msg,
                          true) ||
        !read_i64_array(kv, "legacy_cumu_deltas", node_n, &legacy_init->cumu_deltas, err_msg,
                        true) ||
        !read_i64_array(kv, "legacy_prev_indexs", node_n, &legacy_init->prev_indexs, err_msg,
                        true) ||
        !read_i64_array(kv, "legacy_next_indexs", node_n, &legacy_init->next_indexs, err_msg,
                        true) ||
        !read_i64_array(kv, "legacy_side_indexs", node_n, &legacy_init->side_indexs, err_msg,
                        true) ||
        !read_float_array(kv, "legacy_output_scores", out_n, &legacy_init->output_scores, err_msg,
                          true) ||
        !read_i64_array(kv, "legacy_output_tokens", out_n, &legacy_init->output_tokens, err_msg,
                        true) ||
        !read_float_array(kv, "legacy_work_scores", work_n, &legacy_init->work_scores, err_msg,
                          true) ||
        !read_float_array(kv, "legacy_sort_scores", sort_n, &legacy_init->sort_scores, err_msg,
                          true)) {
        return false;
    }

    expected->has_expected =
        has_key(kv, "expected_cache_topk_indices") || has_key(kv, "expected_legacy_cumu_tokens");
    if (!expected->has_expected) {
        return true;
    }

    expected->legacy = LegacyState{};
    expected->legacy.cumu_tokens.assign(node_n, -1);
    expected->legacy.cumu_scores.assign(node_n, 0.0f);
    expected->legacy.cumu_deltas.assign(node_n, -1);
    expected->legacy.prev_indexs.assign(node_n, -1);
    expected->legacy.next_indexs.assign(node_n, -1);
    expected->legacy.side_indexs.assign(node_n, -1);
    expected->legacy.output_scores.assign(out_n, 0.0f);
    expected->legacy.output_tokens.assign(out_n, -1);
    expected->legacy.work_scores.assign(work_n, 0.0f);
    expected->legacy.sort_scores.assign(sort_n, 0.0f);

    expected->output_hidden.assign(static_cast<size_t>(cfg->batch_size) * cfg->node_top_k *
                                       cfg->hidden_size,
                                   0.0f);
    expected->cache_topk.assign(out_n, -1);
    expected->dbg_curr.assign(topk_n, 0.0f);
    expected->dbg_sort.assign(topk_n, 0.0f);
    expected->dbg_sort_idx.assign(topk_n, -1);
    expected->dbg_parent.assign(out_n, -1);
    expected->dbg_remap.assign(topk_n, -1);

    if (!read_float_array(kv, "expected_output_hidden_states", expected->output_hidden.size(),
                          &expected->output_hidden, err_msg, true) ||
        !read_i64_array(kv, "expected_cache_topk_indices", expected->cache_topk.size(),
                        &expected->cache_topk, err_msg, true) ||
        !read_float_array(kv, "expected_dbg_curr_layer_scores", expected->dbg_curr.size(),
                          &expected->dbg_curr, err_msg, true) ||
        !read_float_array(kv, "expected_dbg_sort_layer_scores", expected->dbg_sort.size(),
                          &expected->dbg_sort, err_msg, true) ||
        !read_i64_array(kv, "expected_dbg_sort_layer_indices", expected->dbg_sort_idx.size(),
                        &expected->dbg_sort_idx, err_msg, true) ||
        !read_i64_array(kv, "expected_dbg_parent_indices_in_layer", expected->dbg_parent.size(),
                        &expected->dbg_parent, err_msg, true) ||
        !read_i64_array(kv, "expected_dbg_remapped_topk_tokens", expected->dbg_remap.size(),
                        &expected->dbg_remap, err_msg, true) ||
        !read_i64_array(kv, "expected_legacy_cumu_tokens", expected->legacy.cumu_tokens.size(),
                        &expected->legacy.cumu_tokens, err_msg, true) ||
        !read_float_array(kv, "expected_legacy_cumu_scores", expected->legacy.cumu_scores.size(),
                          &expected->legacy.cumu_scores, err_msg, true) ||
        !read_i64_array(kv, "expected_legacy_cumu_deltas", expected->legacy.cumu_deltas.size(),
                        &expected->legacy.cumu_deltas, err_msg, true) ||
        !read_i64_array(kv, "expected_legacy_prev_indexs", expected->legacy.prev_indexs.size(),
                        &expected->legacy.prev_indexs, err_msg, true) ||
        !read_i64_array(kv, "expected_legacy_next_indexs", expected->legacy.next_indexs.size(),
                        &expected->legacy.next_indexs, err_msg, true) ||
        !read_i64_array(kv, "expected_legacy_side_indexs", expected->legacy.side_indexs.size(),
                        &expected->legacy.side_indexs, err_msg, true) ||
        !read_float_array(kv, "expected_legacy_output_scores", expected->legacy.output_scores.size(),
                          &expected->legacy.output_scores, err_msg, true) ||
        !read_i64_array(kv, "expected_legacy_output_tokens", expected->legacy.output_tokens.size(),
                        &expected->legacy.output_tokens, err_msg, true) ||
        !read_float_array(kv, "expected_legacy_work_scores", expected->legacy.work_scores.size(),
                          &expected->legacy.work_scores, err_msg, true) ||
        !read_float_array(kv, "expected_legacy_sort_scores", expected->legacy.sort_scores.size(),
                          &expected->legacy.sort_scores, err_msg, true)) {
        return false;
    }

    return true;
}

static void make_synthetic_fixture(TestCfg* cfg,
                                   FusedTestInputs* in,
                                   LegacyState* legacy_init) {
    const int total_topk = cfg->tree_width * cfg->node_top_k;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> prob_dist(0.01f, 0.99f);
    std::uniform_real_distribution<float> score_dist(0.1f, 1.0f);
    std::uniform_real_distribution<float> hid_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int64_t> tok_dist(0, cfg->hot_vocab_size - 1);

    in->topk_probas.assign(static_cast<size_t>(cfg->batch_size) * total_topk, 0.0f);
    in->topk_tokens.assign(static_cast<size_t>(cfg->batch_size) * total_topk, 0);
    in->last_layer_scores.assign(static_cast<size_t>(cfg->batch_size) * cfg->tree_width, 0.0f);
    in->input_hidden_states.assign(
        static_cast<size_t>(cfg->batch_size) * cfg->tree_width * cfg->hidden_size, 0.0f);
    for (size_t i = 0; i < in->topk_probas.size(); ++i) {
        in->topk_probas[i] = prob_dist(rng);
        in->topk_tokens[i] = tok_dist(rng);
    }
    for (size_t i = 0; i < in->last_layer_scores.size(); ++i) {
        in->last_layer_scores[i] = score_dist(rng);
    }
    for (size_t i = 0; i < in->input_hidden_states.size(); ++i) {
        in->input_hidden_states[i] = hid_dist(rng);
    }

    in->hot_token_id.assign(static_cast<size_t>(cfg->hot_vocab_size), 0);
    for (int i = 0; i < cfg->hot_vocab_size; ++i) {
        in->hot_token_id[static_cast<size_t>(i)] = (i * 7 + 3) % cfg->hot_vocab_size;
    }

    in->topk_indexs_prev.assign(static_cast<size_t>(cfg->batch_size) * cfg->tree_width, 0);
    for (int b = 0; b < cfg->batch_size; ++b) {
        for (int i = 0; i < cfg->tree_width; ++i) {
            in->topk_indexs_prev[b * cfg->tree_width + i] = i;
        }
    }

    *legacy_init = make_legacy_state(*cfg, 1);
}

static bool run_fused_test(const TestCfg& cfg,
                           const FusedTestInputs& in,
                           const LegacyState& legacy_init,
                           const FusedExpectedOutputs* file_expected) {
    const int total_topk = cfg.tree_width * cfg.node_top_k;

    LegacyState ref_legacy = legacy_init;
    LegacyState fused_legacy = legacy_init;

    std::vector<float> ref_output_hidden(cfg.batch_size * cfg.node_top_k * cfg.hidden_size, 0.0f);
    std::vector<float> fused_output_hidden(cfg.batch_size * cfg.node_top_k * cfg.hidden_size, 0.0f);
    std::vector<int64_t> ref_cache_topk(cfg.batch_size * cfg.node_top_k, -1);
    std::vector<int64_t> fused_cache_topk(cfg.batch_size * cfg.node_top_k, -1);
    std::vector<float> ref_dbg_curr(cfg.batch_size * total_topk, 0.0f);
    std::vector<float> fused_dbg_curr(cfg.batch_size * total_topk, 0.0f);
    std::vector<float> ref_dbg_sort(cfg.batch_size * total_topk, 0.0f);
    std::vector<float> fused_dbg_sort(cfg.batch_size * total_topk, 0.0f);
    std::vector<int64_t> ref_dbg_sort_idx(cfg.batch_size * total_topk, -1);
    std::vector<int64_t> fused_dbg_sort_idx(cfg.batch_size * total_topk, -1);
    std::vector<int64_t> ref_dbg_parent(cfg.batch_size * cfg.node_top_k, -1);
    std::vector<int64_t> fused_dbg_parent(cfg.batch_size * cfg.node_top_k, -1);
    std::vector<int64_t> ref_dbg_remap(cfg.batch_size * total_topk, -1);
    std::vector<int64_t> fused_dbg_remap(cfg.batch_size * total_topk, -1);

    run_reference_pipeline(
        cfg,
        in.topk_probas,
        in.topk_tokens,
        in.last_layer_scores,
        in.input_hidden_states,
        in.hot_token_id,
        in.topk_indexs_prev,
        &ref_legacy,
        &ref_output_hidden,
        &ref_cache_topk,
        &ref_dbg_curr,
        &ref_dbg_sort,
        &ref_dbg_sort_idx,
        &ref_dbg_parent,
        &ref_dbg_remap);

    tmac::hls::cost_draft_tree_fused_step_hls(
        in.topk_probas.data(),
        in.topk_tokens.data(),
        in.last_layer_scores.data(),
        in.input_hidden_states.data(),
        in.hot_token_id.data(),
        static_cast<int64_t>(in.hot_token_id.size()),
        cfg.use_hot_token_id,
        in.topk_indexs_prev.data(),
        cfg.batch_size,
        cfg.node_top_k,
        cfg.tree_width,
        cfg.hidden_size,
        cfg.cumu_count,
        cfg.verify_num,
        cfg.curr_depth,
        cfg.max_node_count,
        cfg.max_verify_num,
        fused_legacy.cumu_tokens.data(),
        fused_legacy.cumu_scores.data(),
        fused_legacy.cumu_deltas.data(),
        fused_legacy.prev_indexs.data(),
        fused_legacy.next_indexs.data(),
        fused_legacy.side_indexs.data(),
        fused_legacy.output_scores.data(),
        fused_legacy.output_tokens.data(),
        fused_legacy.work_scores.data(),
        fused_legacy.sort_scores.data(),
        fused_output_hidden.data(),
        fused_cache_topk.data(),
        fused_dbg_curr.data(),
        fused_dbg_sort.data(),
        fused_dbg_sort_idx.data(),
        fused_dbg_parent.data(),
        fused_dbg_remap.data());

    const bool use_file_expected = (file_expected != nullptr && file_expected->has_expected);

    const LegacyState& cmp_legacy = use_file_expected ? file_expected->legacy : ref_legacy;
    const std::vector<float>& cmp_output_hidden =
        use_file_expected ? file_expected->output_hidden : ref_output_hidden;
    const std::vector<int64_t>& cmp_cache_topk =
        use_file_expected ? file_expected->cache_topk : ref_cache_topk;
    const std::vector<float>& cmp_dbg_curr = use_file_expected ? file_expected->dbg_curr : ref_dbg_curr;
    const std::vector<float>& cmp_dbg_sort = use_file_expected ? file_expected->dbg_sort : ref_dbg_sort;
    const std::vector<int64_t>& cmp_dbg_sort_idx =
        use_file_expected ? file_expected->dbg_sort_idx : ref_dbg_sort_idx;
    const std::vector<int64_t>& cmp_dbg_parent =
        use_file_expected ? file_expected->dbg_parent : ref_dbg_parent;
    const std::vector<int64_t>& cmp_dbg_remap =
        use_file_expected ? file_expected->dbg_remap : ref_dbg_remap;

    const float err_hidden = max_abs_diff(fused_output_hidden, cmp_output_hidden);
    const float err_out_scores = max_abs_diff(fused_legacy.output_scores, cmp_legacy.output_scores);
    const float err_work_scores = max_abs_diff(fused_legacy.work_scores, cmp_legacy.work_scores);
    const float err_sort_scores = max_abs_diff(fused_legacy.sort_scores, cmp_legacy.sort_scores);
    const float err_dbg_curr = max_abs_diff(fused_dbg_curr, cmp_dbg_curr);
    const float err_dbg_sort = max_abs_diff(fused_dbg_sort, cmp_dbg_sort);
    const size_t mm_cumu_tokens = mismatch_i64(fused_legacy.cumu_tokens, cmp_legacy.cumu_tokens);
    const size_t mm_prev = mismatch_i64(fused_legacy.prev_indexs, cmp_legacy.prev_indexs);
    const size_t mm_next = mismatch_i64(fused_legacy.next_indexs, cmp_legacy.next_indexs);
    const size_t mm_side = mismatch_i64(fused_legacy.side_indexs, cmp_legacy.side_indexs);
    const size_t mm_out_tokens = mismatch_i64(fused_legacy.output_tokens, cmp_legacy.output_tokens);
    const size_t mm_cache_topk = mismatch_i64(fused_cache_topk, cmp_cache_topk);
    const size_t mm_dbg_sort_idx = mismatch_i64(fused_dbg_sort_idx, cmp_dbg_sort_idx);
    const size_t mm_dbg_parent = mismatch_i64(fused_dbg_parent, cmp_dbg_parent);
    const size_t mm_dbg_remap = mismatch_i64(fused_dbg_remap, cmp_dbg_remap);

    std::cout << "max|hidden diff|       = " << err_hidden << "\n";
    std::cout << "max|output_scores diff|= " << err_out_scores << "\n";
    std::cout << "max|work_scores diff|  = " << err_work_scores << "\n";
    std::cout << "max|sort_scores diff|  = " << err_sort_scores << "\n";
    std::cout << "max|dbg_curr diff|     = " << err_dbg_curr << "\n";
    std::cout << "max|dbg_sort diff|     = " << err_dbg_sort << "\n";
    std::cout << "cumu_tokens mismatches = " << mm_cumu_tokens << "\n";
    std::cout << "prev/next/side mism    = " << mm_prev << "/" << mm_next << "/" << mm_side << "\n";
    std::cout << "output_tokens mism     = " << mm_out_tokens << "\n";
    std::cout << "cache_topk mism        = " << mm_cache_topk << "\n";
    std::cout << "dbg sort/parent/remap  = " << mm_dbg_sort_idx << "/" << mm_dbg_parent
              << "/" << mm_dbg_remap << "\n";

    const bool pass_float =
        (err_hidden <= 1e-6f) && (err_out_scores <= 1e-6f) && (err_work_scores <= 1e-6f) &&
        (err_sort_scores <= 1e-6f) && (err_dbg_curr <= 1e-6f) && (err_dbg_sort <= 1e-6f);
    const bool pass_int =
        (mm_cumu_tokens == 0) && (mm_prev == 0) && (mm_next == 0) && (mm_side == 0) &&
        (mm_out_tokens == 0) && (mm_cache_topk == 0) && (mm_dbg_sort_idx == 0) &&
        (mm_dbg_parent == 0) && (mm_dbg_remap == 0);

    const char* cmp_mode = use_file_expected ? "file-expected tensors" : "reference pipeline";

    if (!pass_float || !pass_int) {
        std::cerr << "[FAIL] cost_draft_tree fused wiring HLS mismatch vs " << cmp_mode << ".\n";
        return false;
    }
    std::cout << "[PASS] cost_draft_tree fused wiring HLS matches " << cmp_mode << ".\n";
    return true;
}

// Multi-depth bench:
// Repeatedly applies fused step with chained state and checks candidate outputs
// against a sequential reference pipeline at each depth.
static bool run_fused_multidepth_candidate_test(const TestCfg& base_cfg,
                                                const FusedTestInputs& seed_in,
                                                const LegacyState& legacy_init,
                                                int steps) {
    if (steps <= 0) {
        std::cerr << "[FAIL] multi-depth steps must be > 0.\n";
        return false;
    }

    const int total_topk = base_cfg.tree_width * base_cfg.node_top_k;

    LegacyState ref_legacy = legacy_init;
    LegacyState fused_legacy = legacy_init;

    std::vector<float> step_topk_probas = seed_in.topk_probas;
    std::vector<int64_t> step_topk_tokens = seed_in.topk_tokens;
    std::vector<float> step_last_layer_scores = seed_in.last_layer_scores;
    std::vector<float> step_input_hidden_states = seed_in.input_hidden_states;
    std::vector<int64_t> step_topk_indexs_prev = seed_in.topk_indexs_prev;

    std::vector<float> prev_fused_output_hidden(
        static_cast<size_t>(base_cfg.batch_size) * base_cfg.node_top_k * base_cfg.hidden_size,
        0.0f);

    for (int depth = 0; depth < steps; ++depth) {
        TestCfg cfg = base_cfg;
        cfg.curr_depth = base_cfg.curr_depth + depth;

        // For depth>0, chain inputs from prior fused outputs to emulate repeated drafting.
        if (depth > 0) {
            // 1) next input hidden features from previous step output hidden
            if (base_cfg.tree_width == base_cfg.node_top_k) {
                step_input_hidden_states = prev_fused_output_hidden;
            } else {
                for (int b = 0; b < base_cfg.batch_size; ++b) {
                    for (int t = 0; t < base_cfg.tree_width; ++t) {
                        const int src_t = (base_cfg.node_top_k > 0) ? (t % base_cfg.node_top_k) : 0;
                        for (int h = 0; h < base_cfg.hidden_size; ++h) {
                            const size_t src = (static_cast<size_t>(b) * base_cfg.node_top_k + src_t) *
                                                   base_cfg.hidden_size +
                                               h;
                            const size_t dst = (static_cast<size_t>(b) * base_cfg.tree_width + t) *
                                                   base_cfg.hidden_size +
                                               h;
                            step_input_hidden_states[dst] = prev_fused_output_hidden[src];
                        }
                    }
                }
            }

            // 2) next parent scores from previous selected output scores
            for (int b = 0; b < base_cfg.batch_size; ++b) {
                for (int t = 0; t < base_cfg.tree_width; ++t) {
                    const int src_t = (base_cfg.node_top_k > 0) ? (t % base_cfg.node_top_k) : 0;
                    const float s = fused_legacy.output_scores[b * base_cfg.node_top_k + src_t];
                    step_last_layer_scores[static_cast<size_t>(b) * base_cfg.tree_width + t] =
                        std::fabs(s) + 1e-3f;
                }
            }

            // 3) deterministic candidate proposals for this depth
            for (int b = 0; b < base_cfg.batch_size; ++b) {
                for (int p = 0; p < base_cfg.tree_width; ++p) {
                    for (int k = 0; k < base_cfg.node_top_k; ++k) {
                        const int flat = b * total_topk + p * base_cfg.node_top_k + k;
                        step_topk_probas[static_cast<size_t>(flat)] =
                            0.2f + 0.03f * static_cast<float>((depth + b + p + k) % 7);
                        const int vocab = std::max(1, base_cfg.hot_vocab_size);
                        step_topk_tokens[static_cast<size_t>(flat)] =
                            static_cast<int64_t>((depth * 131 + b * 29 + p * base_cfg.node_top_k + k) %
                                                 vocab);
                    }
                }
            }

            // 4) previous top-k node ids (keep topk_indexs_prev as-is or update externally)
            // topk_indexs_prev is preserved between steps without frontier chaining
        }

        // Per-step outputs for reference and fused paths.
        std::vector<float> ref_output_hidden(
            static_cast<size_t>(cfg.batch_size) * cfg.node_top_k * cfg.hidden_size, 0.0f);
        std::vector<float> fused_output_hidden(
            static_cast<size_t>(cfg.batch_size) * cfg.node_top_k * cfg.hidden_size, 0.0f);
        std::vector<int64_t> ref_cache_topk(cfg.batch_size * cfg.node_top_k, -1);
        std::vector<int64_t> fused_cache_topk(cfg.batch_size * cfg.node_top_k, -1);
        std::vector<float> ref_dbg_curr(cfg.batch_size * total_topk, 0.0f);
        std::vector<float> fused_dbg_curr(cfg.batch_size * total_topk, 0.0f);
        std::vector<float> ref_dbg_sort(cfg.batch_size * total_topk, 0.0f);
        std::vector<float> fused_dbg_sort(cfg.batch_size * total_topk, 0.0f);
        std::vector<int64_t> ref_dbg_sort_idx(cfg.batch_size * total_topk, -1);
        std::vector<int64_t> fused_dbg_sort_idx(cfg.batch_size * total_topk, -1);
        std::vector<int64_t> ref_dbg_parent(cfg.batch_size * cfg.node_top_k, -1);
        std::vector<int64_t> fused_dbg_parent(cfg.batch_size * cfg.node_top_k, -1);
        std::vector<int64_t> ref_dbg_remap(cfg.batch_size * total_topk, -1);
        std::vector<int64_t> fused_dbg_remap(cfg.batch_size * total_topk, -1);

        run_reference_pipeline(
            cfg,
            step_topk_probas,
            step_topk_tokens,
            step_last_layer_scores,
            step_input_hidden_states,
            seed_in.hot_token_id,
            step_topk_indexs_prev,
            &ref_legacy,
            &ref_output_hidden,
            &ref_cache_topk,
            &ref_dbg_curr,
            &ref_dbg_sort,
            &ref_dbg_sort_idx,
            &ref_dbg_parent,
            &ref_dbg_remap);

        tmac::hls::cost_draft_tree_fused_step_hls(
            step_topk_probas.data(),
            step_topk_tokens.data(),
            step_last_layer_scores.data(),
            step_input_hidden_states.data(),
            seed_in.hot_token_id.data(),
            static_cast<int64_t>(seed_in.hot_token_id.size()),
            cfg.use_hot_token_id,
            step_topk_indexs_prev.data(),
            cfg.batch_size,
            cfg.node_top_k,
            cfg.tree_width,
            cfg.hidden_size,
            cfg.cumu_count,
            cfg.verify_num,
            cfg.curr_depth,
            cfg.max_node_count,
            cfg.max_verify_num,
            fused_legacy.cumu_tokens.data(),
            fused_legacy.cumu_scores.data(),
            fused_legacy.cumu_deltas.data(),
            fused_legacy.prev_indexs.data(),
            fused_legacy.next_indexs.data(),
            fused_legacy.side_indexs.data(),
            fused_legacy.output_scores.data(),
            fused_legacy.output_tokens.data(),
            fused_legacy.work_scores.data(),
            fused_legacy.sort_scores.data(),
            fused_output_hidden.data(),
            fused_cache_topk.data(),
            fused_dbg_curr.data(),
            fused_dbg_sort.data(),
            fused_dbg_sort_idx.data(),
            fused_dbg_parent.data(),
            fused_dbg_remap.data());

        const size_t mm_out_tokens = mismatch_i64(fused_legacy.output_tokens, ref_legacy.output_tokens);
        const size_t mm_cache_topk = mismatch_i64(fused_cache_topk, ref_cache_topk);
        const float err_hidden = max_abs_diff(fused_output_hidden, ref_output_hidden);

        std::cout << "[depth " << depth << "] candidate mism(output/cache)= "
                  << mm_out_tokens << "/" << mm_cache_topk
                  << "  hidden_max_diff=" << err_hidden
                  << "\n";

        if (mm_out_tokens != 0 || mm_cache_topk != 0 || err_hidden > 1e-6f) {
            std::cerr << "[FAIL] multi-depth fused mismatch at depth " << depth << ".\n";
            return false;
        }

        prev_fused_output_hidden = fused_output_hidden;
    }

    std::cout << "[PASS] multi-depth fused-step candidate bench matched reference for " << steps
              << " steps.\n";
    if (!fused_legacy.output_tokens.empty()) {
        std::cout << "[final] output_tokens (batch0): ";
        const int n = std::min(base_cfg.node_top_k, static_cast<int>(fused_legacy.output_tokens.size()));
        for (int i = 0; i < n; ++i) {
            if (i) std::cout << " ";
            std::cout << fused_legacy.output_tokens[i];
        }
        std::cout << "\n";
    }
    return true;
}

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

    TestCfg cfg;
    FusedTestInputs in;
    LegacyState legacy_init;
    FusedExpectedOutputs file_expected;

    if (!opts.case_file.empty()) {
        if (!load_file_fixture(opts.case_file, &cfg, &in, &legacy_init,
                               &file_expected, &err_msg)) {
            std::cerr << "[FAIL] " << err_msg << "\n";
            return 1;
        }
        if (opts.dry_run) {
            std::cout << "[DRY-RUN] Parsed fused wiring case file: " << opts.case_file << "\n";
            std::cout << "[DRY-RUN] dims(B,topk,tree,hidden,cumu,verify)= "
                      << cfg.batch_size << "," << cfg.node_top_k << "," << cfg.tree_width << ","
                      << cfg.hidden_size << "," << cfg.cumu_count << "," << cfg.verify_num
                      << "\n";
            std::cout << "[DRY-RUN] expected payload mode: "
                      << (file_expected.has_expected ? "from-file" : "reference-generated")
                      << "\n";
            return 0;
        }
    } else {
        make_synthetic_fixture(&cfg, &in, &legacy_init);
    }

    if (opts.multi_depth_steps > 1) {
        if (!opts.case_file.empty() && file_expected.has_expected) {
            std::cout << "[info] --multi-depth-steps enabled; file expected tensors are used only"
                      << " as initial seed inputs, per-step compare uses generated reference.\n";
        }
        if (!run_fused_multidepth_candidate_test(
                cfg, in, legacy_init, opts.multi_depth_steps)) {
            return 1;
        }
        return 0;
    }

    const FusedExpectedOutputs* expected_ptr = opts.case_file.empty() ? nullptr : &file_expected;
    if (!run_fused_test(cfg, in, legacy_init, expected_ptr)) {
        return 1;
    }
    return 0;
}
