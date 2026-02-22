#include "cost_draft_tree_controller_hls.hpp"
#include "cost_draft_tree_fused_wiring_hls.hpp"
#include "cost_draft_tree_score_hls.hpp"
#include "cost_draft_tree_tb_case_io.hpp"
#include "cost_draft_tree_update_hls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

struct BoolBuffer {
    BoolBuffer() = default;

    explicit BoolBuffer(size_t n)
        : size_(n), data_(n ? new bool[n] : nullptr) {
        for (size_t i = 0; i < size_; ++i) data_[i] = false;
    }

    BoolBuffer(const BoolBuffer& other)
        : size_(other.size_), data_(other.size_ ? new bool[other.size_] : nullptr) {
        for (size_t i = 0; i < size_; ++i) data_[i] = other.data_[i];
    }

    BoolBuffer& operator=(const BoolBuffer& other) {
        if (this == &other) return *this;
        BoolBuffer tmp(other);
        swap(tmp);
        return *this;
    }

    void swap(BoolBuffer& other) {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
    }

    size_t size() const { return size_; }
    bool* data() { return data_.get(); }
    const bool* data() const { return data_.get(); }

    bool& operator[](size_t i) { return data_[i]; }
    const bool& operator[](size_t i) const { return data_[i]; }

private:
    size_t size_ = 0;
    std::unique_ptr<bool[]> data_;
};

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
    BoolBuffer output_tree_mask;
};

struct ControllerState {
    std::vector<int> node_count;
    std::vector<int64_t> frontier;
    std::vector<int64_t> node_token;
    std::vector<int64_t> node_parent;
    std::vector<int64_t> node_first_child;
    std::vector<int64_t> node_last_child;
    std::vector<int64_t> node_next_sibling;
    std::vector<int64_t> node_depth;
    std::vector<int32_t> node_cache_loc;
};

struct TestCfg {
    int batch_size = 2;
    int node_top_k = 4;
    int tree_width = 4;
    int hidden_size = 16;
    int cumu_count = 8;
    int input_count = 6;
    int verify_num = 8;
    int curr_depth = 2;

    int max_input_size = 16;
    int max_node_count = 128;
    int max_verify_num = 16;
    int max_tree_width = 4;
    int max_prefix_len = 8;
    int parent_width = 4;
    int next_tree_width = 4;

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
    BoolBuffer input_tree_mask;
    std::vector<int32_t> prefix_kv;
    std::vector<int> prefix_lens;
    std::vector<int32_t> selected_cache_locs;
    std::vector<int64_t> seed_tokens;
    std::vector<int32_t> seed_cache;
};

struct FusedExpectedOutputs {
    bool has_expected = false;

    LegacyState legacy;
    ControllerState ctrl;

    std::vector<float> output_hidden;
    std::vector<int64_t> cache_topk;
    std::vector<int64_t> frontier_out;
    std::vector<int32_t> kv_indices;
    BoolBuffer kv_mask;
    std::vector<int> kv_lens;
    std::vector<int64_t> frontier_tokens;
    std::vector<int64_t> frontier_parent_ids;
    std::vector<int64_t> frontier_depths;
    std::vector<int32_t> frontier_cache_locs;
    std::vector<float> dbg_curr;
    std::vector<float> dbg_sort;
    std::vector<int64_t> dbg_sort_idx;
    std::vector<int64_t> dbg_parent;
    std::vector<int64_t> dbg_remap;
};

struct CliOptions {
    std::string case_file;
    bool dry_run = false;
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

static size_t mismatch_i32(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    size_t c = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++c;
    }
    return c;
}

static size_t mismatch_int(const std::vector<int>& a, const std::vector<int>& b) {
    size_t c = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++c;
    }
    return c;
}

static size_t mismatch_bool(const BoolBuffer& a, const BoolBuffer& b) {
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
    const size_t mask_n =
        static_cast<size_t>(cfg.batch_size) * cfg.node_top_k * (cfg.max_input_size + 1);

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
    s.output_tree_mask = BoolBuffer(mask_n);

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

static ControllerState make_controller_state_seeded(const TestCfg& cfg,
                                                    const std::vector<int64_t>& seed_tokens,
                                                    const std::vector<int32_t>& seed_cache) {
    ControllerState st;

    st.node_count.assign(cfg.batch_size, 0);
    st.frontier.assign(static_cast<size_t>(cfg.batch_size) * cfg.max_tree_width, -1);
    st.node_token.assign(static_cast<size_t>(cfg.batch_size) * cfg.max_node_count, -1);
    st.node_parent.assign(static_cast<size_t>(cfg.batch_size) * cfg.max_node_count, -1);
    st.node_first_child.assign(static_cast<size_t>(cfg.batch_size) * cfg.max_node_count, -1);
    st.node_last_child.assign(static_cast<size_t>(cfg.batch_size) * cfg.max_node_count, -1);
    st.node_next_sibling.assign(static_cast<size_t>(cfg.batch_size) * cfg.max_node_count, -1);
    st.node_depth.assign(static_cast<size_t>(cfg.batch_size) * cfg.max_node_count, -1);
    st.node_cache_loc.assign(static_cast<size_t>(cfg.batch_size) * cfg.max_node_count, -1);

    tmac::hls::cdt_controller_reset(
        cfg.batch_size,
        cfg.max_tree_width,
        cfg.max_node_count,
        st.node_count.data(),
        st.frontier.data(),
        st.node_token.data(),
        st.node_parent.data(),
        st.node_first_child.data(),
        st.node_last_child.data(),
        st.node_next_sibling.data(),
        st.node_depth.data(),
        st.node_cache_loc.data());

    tmac::hls::cdt_controller_seed_frontier(
        seed_tokens.data(),
        seed_cache.data(),
        cfg.batch_size,
        cfg.parent_width,
        cfg.max_tree_width,
        cfg.max_node_count,
        st.node_count.data(),
        st.frontier.data(),
        st.node_token.data(),
        st.node_parent.data(),
        st.node_first_child.data(),
        st.node_last_child.data(),
        st.node_next_sibling.data(),
        st.node_depth.data(),
        st.node_cache_loc.data());

    return st;
}

static void run_reference_pipeline(
    const TestCfg& cfg,
    const std::vector<float>& topk_probas,
    const std::vector<int64_t>& topk_tokens,
    const std::vector<float>& last_layer_scores,
    const std::vector<float>& input_hidden_states,
    const std::vector<int64_t>& hot_token_id,
    const std::vector<int64_t>& topk_indexs_prev,
    const BoolBuffer& input_tree_mask,
    const std::vector<int32_t>& prefix_kv,
    const std::vector<int>& prefix_lens,
    const std::vector<int32_t>& selected_cache_locs,
    LegacyState* legacy,
    ControllerState* controller,
    std::vector<float>* output_hidden_states,
    std::vector<int64_t>* cache_topk_indices,
    std::vector<int64_t>* frontier_out,
    std::vector<int32_t>* kv_indices,
    BoolBuffer* kv_mask,
    std::vector<int>* kv_lens,
    std::vector<int64_t>* frontier_tokens,
    std::vector<int64_t>* frontier_parent_ids,
    std::vector<int64_t>* frontier_depths,
    std::vector<int32_t>* frontier_cache_locs,
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
        input_tree_mask.data(),
        cfg.batch_size,
        cfg.node_top_k,
        cfg.tree_width,
        cfg.input_count,
        cfg.cumu_count,
        cfg.verify_num,
        cfg.curr_depth,
        cfg.max_input_size,
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
        legacy->sort_scores.data(),
        legacy->output_tree_mask.data());

    std::vector<int32_t> child_cache_loc(cfg.batch_size * cfg.node_top_k, -1);
    for (int i = 0; i < cfg.batch_size * cfg.node_top_k; ++i) {
        child_cache_loc[static_cast<size_t>(i)] = selected_cache_locs[static_cast<size_t>(i)];
    }

    tmac::hls::cdt_controller_expand_frontier(
        controller->frontier.data(),
        stage_parent.data(),
        stage_output_tokens.data(),
        child_cache_loc.data(),
        cfg.batch_size,
        cfg.parent_width,
        cfg.next_tree_width,
        cfg.max_tree_width,
        cfg.max_node_count,
        controller->node_count.data(),
        frontier_out->data(),
        controller->node_token.data(),
        controller->node_parent.data(),
        controller->node_first_child.data(),
        controller->node_last_child.data(),
        controller->node_next_sibling.data(),
        controller->node_depth.data(),
        controller->node_cache_loc.data());

    tmac::hls::cdt_controller_build_parent_visible_kv(
        frontier_out->data(),
        prefix_kv.data(),
        prefix_lens.data(),
        cfg.batch_size,
        cfg.next_tree_width,
        cfg.max_tree_width,
        cfg.max_prefix_len,
        cfg.max_input_size,
        cfg.max_node_count,
        controller->node_parent.data(),
        controller->node_cache_loc.data(),
        kv_indices->data(),
        kv_mask->data(),
        kv_lens->data(),
        nullptr);

    tmac::hls::cdt_controller_export_frontier(
        frontier_out->data(),
        cfg.batch_size,
        cfg.next_tree_width,
        cfg.max_tree_width,
        cfg.max_node_count,
        controller->node_token.data(),
        controller->node_parent.data(),
        controller->node_depth.data(),
        controller->node_cache_loc.data(),
        frontier_tokens->data(),
        frontier_parent_ids->data(),
        frontier_depths->data(),
        frontier_cache_locs->data());

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
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: cost_draft_tree_fused_wiring_tb [--case-file <path>] [--dry-run]\n";
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
                              ControllerState* ctrl_init,
                              FusedExpectedOutputs* expected,
                              std::string* err_msg) {
    using namespace tmac::hls::tb_case_io;
    RawCaseMap kv;
    if (!parse_key_count_file(path, &kv, err_msg)) {
        return false;
    }

    std::vector<int> meta;
    if (!read_int_array(kv, "meta", 17, &meta, err_msg, true)) {
        return false;
    }

    cfg->batch_size = meta[0];
    cfg->node_top_k = meta[1];
    cfg->tree_width = meta[2];
    cfg->hidden_size = meta[3];
    cfg->cumu_count = meta[4];
    cfg->input_count = meta[5];
    cfg->verify_num = meta[6];
    cfg->curr_depth = meta[7];
    cfg->max_input_size = meta[8];
    cfg->max_node_count = meta[9];
    cfg->max_verify_num = meta[10];
    cfg->max_tree_width = meta[11];
    cfg->max_prefix_len = meta[12];
    cfg->parent_width = meta[13];
    cfg->next_tree_width = meta[14];
    cfg->hot_vocab_size = meta[15];
    cfg->use_hot_token_id = (meta[16] != 0);

    if (cfg->batch_size <= 0 || cfg->node_top_k <= 0 || cfg->tree_width <= 0 ||
        cfg->hidden_size <= 0 || cfg->input_count <= 0 || cfg->max_input_size <= 0 ||
        cfg->max_node_count <= 0 || cfg->max_verify_num <= 0 || cfg->max_tree_width <= 0 ||
        cfg->max_prefix_len <= 0 || cfg->parent_width <= 0 || cfg->next_tree_width < 0) {
        *err_msg = "invalid scalar dimensions in meta";
        return false;
    }

    const int total_topk = cfg->tree_width * cfg->node_top_k;
    const size_t topk_n = static_cast<size_t>(cfg->batch_size) * total_topk;
    const size_t score_n = static_cast<size_t>(cfg->batch_size) * cfg->tree_width;
    const size_t hidden_n =
        static_cast<size_t>(cfg->batch_size) * cfg->tree_width * cfg->hidden_size;
    const size_t tree_n = static_cast<size_t>(cfg->batch_size) * cfg->tree_width;
    const size_t in_mask_n = static_cast<size_t>(cfg->batch_size) * cfg->tree_width *
                             static_cast<size_t>(cfg->input_count - 1);
    const size_t prefix_kv_n = static_cast<size_t>(cfg->batch_size) * cfg->max_prefix_len;
    const size_t selected_cache_n = static_cast<size_t>(cfg->batch_size) * cfg->node_top_k;
    const size_t frontier_n = static_cast<size_t>(cfg->batch_size) * cfg->max_tree_width;
    const size_t node_n = static_cast<size_t>(cfg->batch_size) * cfg->max_node_count;
    const size_t out_n = static_cast<size_t>(cfg->batch_size) * cfg->node_top_k;
    const size_t work_n = static_cast<size_t>(cfg->batch_size) *
                          static_cast<size_t>(cfg->max_verify_num + cfg->node_top_k);
    const size_t sort_n = static_cast<size_t>(cfg->batch_size) * cfg->max_verify_num;
    const size_t out_mask_n = static_cast<size_t>(cfg->batch_size) * cfg->node_top_k *
                              static_cast<size_t>(cfg->max_input_size + 1);

    if (!read_float_array(kv, "topk_probas_sampling", topk_n, &in->topk_probas, err_msg, true) ||
        !read_i64_array(kv, "topk_tokens_sampling", topk_n, &in->topk_tokens, err_msg, true) ||
        !read_float_array(kv, "last_layer_scores", score_n, &in->last_layer_scores, err_msg, true) ||
        !read_float_array(kv, "input_hidden_states", hidden_n, &in->input_hidden_states, err_msg,
                          true) ||
        !read_i64_array(kv, "topk_indexs_prev", tree_n, &in->topk_indexs_prev, err_msg, true) ||
        !read_i32_array(kv, "prefix_kv_locs", prefix_kv_n, &in->prefix_kv, err_msg, true) ||
        !read_int_array(kv, "prefix_lens", cfg->batch_size, &in->prefix_lens, err_msg, true) ||
        !read_i32_array(kv, "selected_cache_locs", selected_cache_n, &in->selected_cache_locs,
                        err_msg, true)) {
        return false;
    }

    std::vector<bool> in_mask_tmp;
    if (!read_bool_array(kv, "input_tree_mask", in_mask_n, &in_mask_tmp, err_msg, true)) {
        return false;
    }
    in->input_tree_mask = BoolBuffer(in_mask_n);
    for (size_t i = 0; i < in_mask_n; ++i) {
        in->input_tree_mask[i] = in_mask_tmp[i];
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

    ctrl_init->node_count.assign(static_cast<size_t>(cfg->batch_size), 0);
    ctrl_init->frontier.assign(frontier_n, -1);
    ctrl_init->node_token.assign(node_n, -1);
    ctrl_init->node_parent.assign(node_n, -1);
    ctrl_init->node_first_child.assign(node_n, -1);
    ctrl_init->node_last_child.assign(node_n, -1);
    ctrl_init->node_next_sibling.assign(node_n, -1);
    ctrl_init->node_depth.assign(node_n, -1);
    ctrl_init->node_cache_loc.assign(node_n, -1);

    if (!read_int_array(kv, "controller_node_count", cfg->batch_size, &ctrl_init->node_count,
                        err_msg, true) ||
        !read_i64_array(kv, "controller_frontier_in", frontier_n, &ctrl_init->frontier, err_msg,
                        true) ||
        !read_i64_array(kv, "controller_node_token_ids", node_n, &ctrl_init->node_token, err_msg,
                        true) ||
        !read_i64_array(kv, "controller_node_parent_ids", node_n, &ctrl_init->node_parent,
                        err_msg, true) ||
        !read_i64_array(kv, "controller_node_first_child_ids", node_n,
                        &ctrl_init->node_first_child, err_msg, true) ||
        !read_i64_array(kv, "controller_node_last_child_ids", node_n, &ctrl_init->node_last_child,
                        err_msg, true) ||
        !read_i64_array(kv, "controller_node_next_sibling_ids", node_n,
                        &ctrl_init->node_next_sibling, err_msg, true) ||
        !read_i64_array(kv, "controller_node_depths", node_n, &ctrl_init->node_depth, err_msg,
                        true) ||
        !read_i32_array(kv, "controller_node_cache_locs", node_n, &ctrl_init->node_cache_loc,
                        err_msg, true)) {
        return false;
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
    legacy_init->output_tree_mask = BoolBuffer(out_mask_n);

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
    std::vector<bool> legacy_mask_tmp;
    if (!read_bool_array(kv, "legacy_output_tree_mask", out_mask_n, &legacy_mask_tmp, err_msg,
                         true)) {
        return false;
    }
    for (size_t i = 0; i < out_mask_n; ++i) {
        legacy_init->output_tree_mask[i] = legacy_mask_tmp[i];
    }

    expected->has_expected =
        has_key(kv, "expected_cache_topk_indices") || has_key(kv, "expected_legacy_cumu_tokens") ||
        has_key(kv, "expected_controller_frontier_out");
    if (!expected->has_expected) {
        return true;
    }

    expected->legacy = LegacyState{};
    expected->ctrl = ControllerState{};
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
    expected->legacy.output_tree_mask = BoolBuffer(out_mask_n);

    expected->ctrl.node_count.assign(static_cast<size_t>(cfg->batch_size), 0);
    expected->ctrl.frontier.assign(frontier_n, -1);
    expected->ctrl.node_token.assign(node_n, -1);
    expected->ctrl.node_parent.assign(node_n, -1);
    expected->ctrl.node_first_child.assign(node_n, -1);
    expected->ctrl.node_last_child.assign(node_n, -1);
    expected->ctrl.node_next_sibling.assign(node_n, -1);
    expected->ctrl.node_depth.assign(node_n, -1);
    expected->ctrl.node_cache_loc.assign(node_n, -1);

    expected->output_hidden.assign(static_cast<size_t>(cfg->batch_size) * cfg->node_top_k *
                                       cfg->hidden_size,
                                   0.0f);
    expected->cache_topk.assign(out_n, -1);
    expected->frontier_out.assign(frontier_n, -1);
    expected->kv_indices.assign(frontier_n * static_cast<size_t>(cfg->max_input_size), -1);
    expected->kv_mask = BoolBuffer(expected->kv_indices.size());
    expected->kv_lens.assign(frontier_n, 0);
    expected->frontier_tokens.assign(frontier_n, -1);
    expected->frontier_parent_ids.assign(frontier_n, -1);
    expected->frontier_depths.assign(frontier_n, -1);
    expected->frontier_cache_locs.assign(frontier_n, -1);
    expected->dbg_curr.assign(topk_n, 0.0f);
    expected->dbg_sort.assign(topk_n, 0.0f);
    expected->dbg_sort_idx.assign(topk_n, -1);
    expected->dbg_parent.assign(out_n, -1);
    expected->dbg_remap.assign(topk_n, -1);

    if (!read_float_array(kv, "expected_output_hidden_states", expected->output_hidden.size(),
                          &expected->output_hidden, err_msg, true) ||
        !read_i64_array(kv, "expected_cache_topk_indices", expected->cache_topk.size(),
                        &expected->cache_topk, err_msg, true) ||
        !read_i64_array(kv, "expected_controller_frontier_out", expected->frontier_out.size(),
                        &expected->frontier_out, err_msg, true) ||
        !read_i32_array(kv, "expected_controller_kv_indices", expected->kv_indices.size(),
                        &expected->kv_indices, err_msg, true) ||
        !read_int_array(kv, "expected_controller_kv_lens", expected->kv_lens.size(),
                        &expected->kv_lens, err_msg, true) ||
        !read_i64_array(kv, "expected_controller_frontier_tokens",
                        expected->frontier_tokens.size(), &expected->frontier_tokens, err_msg,
                        true) ||
        !read_i64_array(kv, "expected_controller_frontier_parent_ids",
                        expected->frontier_parent_ids.size(), &expected->frontier_parent_ids,
                        err_msg, true) ||
        !read_i64_array(kv, "expected_controller_frontier_depths",
                        expected->frontier_depths.size(), &expected->frontier_depths, err_msg,
                        true) ||
        !read_i32_array(kv, "expected_controller_frontier_cache_locs",
                        expected->frontier_cache_locs.size(), &expected->frontier_cache_locs,
                        err_msg, true) ||
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
                          &expected->legacy.sort_scores, err_msg, true) ||
        !read_int_array(kv, "expected_controller_node_count", expected->ctrl.node_count.size(),
                        &expected->ctrl.node_count, err_msg, true) ||
        !read_i64_array(kv, "expected_controller_node_token_ids",
                        expected->ctrl.node_token.size(), &expected->ctrl.node_token, err_msg,
                        true) ||
        !read_i64_array(kv, "expected_controller_node_parent_ids",
                        expected->ctrl.node_parent.size(), &expected->ctrl.node_parent, err_msg,
                        true) ||
        !read_i64_array(kv, "expected_controller_node_first_child_ids",
                        expected->ctrl.node_first_child.size(), &expected->ctrl.node_first_child,
                        err_msg, true) ||
        !read_i64_array(kv, "expected_controller_node_last_child_ids",
                        expected->ctrl.node_last_child.size(), &expected->ctrl.node_last_child,
                        err_msg, true) ||
        !read_i64_array(kv, "expected_controller_node_next_sibling_ids",
                        expected->ctrl.node_next_sibling.size(),
                        &expected->ctrl.node_next_sibling, err_msg, true) ||
        !read_i64_array(kv, "expected_controller_node_depths",
                        expected->ctrl.node_depth.size(), &expected->ctrl.node_depth, err_msg,
                        true) ||
        !read_i32_array(kv, "expected_controller_node_cache_locs",
                        expected->ctrl.node_cache_loc.size(), &expected->ctrl.node_cache_loc,
                        err_msg, true)) {
        return false;
    }

    std::vector<bool> exp_kv_mask_tmp;
    if (!read_bool_array(kv, "expected_controller_kv_mask", expected->kv_mask.size(),
                         &exp_kv_mask_tmp, err_msg, true)) {
        return false;
    }
    for (size_t i = 0; i < expected->kv_mask.size(); ++i) {
        expected->kv_mask[i] = exp_kv_mask_tmp[i];
    }

    std::vector<bool> exp_legacy_mask_tmp;
    if (!read_bool_array(kv, "expected_legacy_output_tree_mask",
                         expected->legacy.output_tree_mask.size(), &exp_legacy_mask_tmp, err_msg,
                         true)) {
        return false;
    }
    for (size_t i = 0; i < expected->legacy.output_tree_mask.size(); ++i) {
        expected->legacy.output_tree_mask[i] = exp_legacy_mask_tmp[i];
    }

    return true;
}

static void make_synthetic_fixture(TestCfg* cfg,
                                   FusedTestInputs* in,
                                   LegacyState* legacy_init,
                                   ControllerState* ctrl_init) {
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

    const size_t in_mask_n =
        static_cast<size_t>(cfg->batch_size) * cfg->tree_width * (cfg->input_count - 1);
    in->input_tree_mask = BoolBuffer(in_mask_n);
    for (size_t i = 0; i < in_mask_n; ++i) {
        in->input_tree_mask[i] = ((i % 2) == 0);
    }

    in->prefix_kv.assign(static_cast<size_t>(cfg->batch_size) * cfg->max_prefix_len, -1);
    in->prefix_lens.assign(static_cast<size_t>(cfg->batch_size), 3);
    for (int b = 0; b < cfg->batch_size; ++b) {
        in->prefix_kv[b * cfg->max_prefix_len + 0] = 11 + b * 10;
        in->prefix_kv[b * cfg->max_prefix_len + 1] = 12 + b * 10;
        in->prefix_kv[b * cfg->max_prefix_len + 2] = 13 + b * 10;
    }

    in->selected_cache_locs.assign(static_cast<size_t>(cfg->batch_size) * cfg->node_top_k, -1);
    for (int b = 0; b < cfg->batch_size; ++b) {
        for (int i = 0; i < cfg->node_top_k; ++i) {
            in->selected_cache_locs[b * cfg->node_top_k + i] = 2100 + b * 100 + i;
        }
    }

    std::vector<int64_t> seed_tokens(static_cast<size_t>(cfg->batch_size) * cfg->parent_width, 0);
    std::vector<int32_t> seed_cache(static_cast<size_t>(cfg->batch_size) * cfg->parent_width, 0);
    for (int b = 0; b < cfg->batch_size; ++b) {
        for (int i = 0; i < cfg->parent_width; ++i) {
            seed_tokens[b * cfg->parent_width + i] = 100 + b * 10 + i;
            seed_cache[b * cfg->parent_width + i] = 1000 + b * 100 + i;
        }
    }

    *legacy_init = make_legacy_state(*cfg, 1);
    *ctrl_init = make_controller_state_seeded(*cfg, seed_tokens, seed_cache);
}

static bool run_fused_test(const TestCfg& cfg,
                           const FusedTestInputs& in,
                           const LegacyState& legacy_init,
                           const ControllerState& ctrl_init,
                           const FusedExpectedOutputs* file_expected) {
    const int total_topk = cfg.tree_width * cfg.node_top_k;

    LegacyState ref_legacy = legacy_init;
    LegacyState fused_legacy = legacy_init;
    ControllerState ref_ctrl = ctrl_init;
    ControllerState fused_ctrl = ctrl_init;

    std::vector<float> ref_output_hidden(cfg.batch_size * cfg.node_top_k * cfg.hidden_size, 0.0f);
    std::vector<float> fused_output_hidden(cfg.batch_size * cfg.node_top_k * cfg.hidden_size, 0.0f);
    std::vector<int64_t> ref_cache_topk(cfg.batch_size * cfg.node_top_k, -1);
    std::vector<int64_t> fused_cache_topk(cfg.batch_size * cfg.node_top_k, -1);
    std::vector<int64_t> ref_frontier_out(cfg.batch_size * cfg.max_tree_width, -1);
    std::vector<int64_t> fused_frontier_out(cfg.batch_size * cfg.max_tree_width, -1);
    std::vector<int32_t> ref_kv_indices(cfg.batch_size * cfg.max_tree_width * cfg.max_input_size, -1);
    std::vector<int32_t> fused_kv_indices = ref_kv_indices;
    BoolBuffer ref_kv_mask(ref_kv_indices.size());
    BoolBuffer fused_kv_mask(ref_kv_indices.size());
    std::vector<int> ref_kv_lens(cfg.batch_size * cfg.max_tree_width, 0);
    std::vector<int> fused_kv_lens = ref_kv_lens;
    std::vector<int64_t> ref_frontier_tokens(cfg.batch_size * cfg.max_tree_width, -1);
    std::vector<int64_t> fused_frontier_tokens = ref_frontier_tokens;
    std::vector<int64_t> ref_frontier_parent_ids(cfg.batch_size * cfg.max_tree_width, -1);
    std::vector<int64_t> fused_frontier_parent_ids = ref_frontier_parent_ids;
    std::vector<int64_t> ref_frontier_depths(cfg.batch_size * cfg.max_tree_width, -1);
    std::vector<int64_t> fused_frontier_depths = ref_frontier_depths;
    std::vector<int32_t> ref_frontier_cache_locs(cfg.batch_size * cfg.max_tree_width, -1);
    std::vector<int32_t> fused_frontier_cache_locs = ref_frontier_cache_locs;
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
        in.input_tree_mask,
        in.prefix_kv,
        in.prefix_lens,
        in.selected_cache_locs,
        &ref_legacy,
        &ref_ctrl,
        &ref_output_hidden,
        &ref_cache_topk,
        &ref_frontier_out,
        &ref_kv_indices,
        &ref_kv_mask,
        &ref_kv_lens,
        &ref_frontier_tokens,
        &ref_frontier_parent_ids,
        &ref_frontier_depths,
        &ref_frontier_cache_locs,
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
        in.input_tree_mask.data(),
        fused_ctrl.frontier.data(),
        in.prefix_kv.data(),
        in.prefix_lens.data(),
        in.selected_cache_locs.data(),
        cfg.batch_size,
        cfg.node_top_k,
        cfg.tree_width,
        cfg.hidden_size,
        cfg.cumu_count,
        cfg.input_count,
        cfg.verify_num,
        cfg.curr_depth,
        cfg.max_input_size,
        cfg.max_node_count,
        cfg.max_verify_num,
        cfg.max_tree_width,
        cfg.max_prefix_len,
        cfg.parent_width,
        cfg.next_tree_width,
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
        fused_legacy.output_tree_mask.data(),
        fused_ctrl.node_count.data(),
        fused_ctrl.node_token.data(),
        fused_ctrl.node_parent.data(),
        fused_ctrl.node_first_child.data(),
        fused_ctrl.node_last_child.data(),
        fused_ctrl.node_next_sibling.data(),
        fused_ctrl.node_depth.data(),
        fused_ctrl.node_cache_loc.data(),
        fused_output_hidden.data(),
        fused_cache_topk.data(),
        fused_frontier_out.data(),
        fused_kv_indices.data(),
        fused_kv_mask.data(),
        fused_kv_lens.data(),
        fused_frontier_tokens.data(),
        fused_frontier_parent_ids.data(),
        fused_frontier_depths.data(),
        fused_frontier_cache_locs.data(),
        fused_dbg_curr.data(),
        fused_dbg_sort.data(),
        fused_dbg_sort_idx.data(),
        fused_dbg_parent.data(),
        fused_dbg_remap.data());

    const bool use_file_expected = (file_expected != nullptr && file_expected->has_expected);

    const LegacyState& cmp_legacy = use_file_expected ? file_expected->legacy : ref_legacy;
    const ControllerState& cmp_ctrl = use_file_expected ? file_expected->ctrl : ref_ctrl;
    const std::vector<float>& cmp_output_hidden =
        use_file_expected ? file_expected->output_hidden : ref_output_hidden;
    const std::vector<int64_t>& cmp_cache_topk =
        use_file_expected ? file_expected->cache_topk : ref_cache_topk;
    const std::vector<int64_t>& cmp_frontier_out =
        use_file_expected ? file_expected->frontier_out : ref_frontier_out;
    const std::vector<int32_t>& cmp_kv_indices =
        use_file_expected ? file_expected->kv_indices : ref_kv_indices;
    const BoolBuffer& cmp_kv_mask = use_file_expected ? file_expected->kv_mask : ref_kv_mask;
    const std::vector<int>& cmp_kv_lens = use_file_expected ? file_expected->kv_lens : ref_kv_lens;
    const std::vector<int64_t>& cmp_frontier_tokens =
        use_file_expected ? file_expected->frontier_tokens : ref_frontier_tokens;
    const std::vector<int64_t>& cmp_frontier_parent_ids =
        use_file_expected ? file_expected->frontier_parent_ids : ref_frontier_parent_ids;
    const std::vector<int64_t>& cmp_frontier_depths =
        use_file_expected ? file_expected->frontier_depths : ref_frontier_depths;
    const std::vector<int32_t>& cmp_frontier_cache_locs =
        use_file_expected ? file_expected->frontier_cache_locs : ref_frontier_cache_locs;
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
    const size_t mm_frontier_out = mismatch_i64(fused_frontier_out, cmp_frontier_out);
    const size_t mm_frontier_tokens = mismatch_i64(fused_frontier_tokens, cmp_frontier_tokens);
    const size_t mm_frontier_parents =
        mismatch_i64(fused_frontier_parent_ids, cmp_frontier_parent_ids);
    const size_t mm_frontier_depths = mismatch_i64(fused_frontier_depths, cmp_frontier_depths);
    const size_t mm_frontier_cache = mismatch_i32(fused_frontier_cache_locs, cmp_frontier_cache_locs);
    const size_t mm_kv_indices = mismatch_i32(fused_kv_indices, cmp_kv_indices);
    const size_t mm_kv_lens = mismatch_int(fused_kv_lens, cmp_kv_lens);
    const size_t mm_kv_mask = mismatch_bool(fused_kv_mask, cmp_kv_mask);
    const size_t mm_ctrl_count = mismatch_int(fused_ctrl.node_count, cmp_ctrl.node_count);
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
    std::cout << "frontier_out mism      = " << mm_frontier_out << "\n";
    std::cout << "frontier fields mism   = " << mm_frontier_tokens << "/" << mm_frontier_parents
              << "/" << mm_frontier_depths << "/" << mm_frontier_cache << "\n";
    std::cout << "kv_indices/lens/mask   = " << mm_kv_indices << "/" << mm_kv_lens
              << "/" << mm_kv_mask << "\n";
    std::cout << "controller_count mism  = " << mm_ctrl_count << "\n";
    std::cout << "dbg sort/parent/remap  = " << mm_dbg_sort_idx << "/" << mm_dbg_parent
              << "/" << mm_dbg_remap << "\n";

    const bool pass_float =
        (err_hidden <= 1e-6f) && (err_out_scores <= 1e-6f) && (err_work_scores <= 1e-6f) &&
        (err_sort_scores <= 1e-6f) && (err_dbg_curr <= 1e-6f) && (err_dbg_sort <= 1e-6f);
    const bool pass_int =
        (mm_cumu_tokens == 0) && (mm_prev == 0) && (mm_next == 0) && (mm_side == 0) &&
        (mm_out_tokens == 0) && (mm_cache_topk == 0) && (mm_frontier_out == 0) &&
        (mm_frontier_tokens == 0) && (mm_frontier_parents == 0) && (mm_frontier_depths == 0) &&
        (mm_frontier_cache == 0) && (mm_kv_indices == 0) && (mm_kv_lens == 0) &&
        (mm_kv_mask == 0) && (mm_ctrl_count == 0) && (mm_dbg_sort_idx == 0) &&
        (mm_dbg_parent == 0) && (mm_dbg_remap == 0);

    const char* cmp_mode = use_file_expected ? "file-expected tensors" : "reference pipeline";

    if (!pass_float || !pass_int) {
        std::cerr << "[FAIL] cost_draft_tree fused wiring HLS mismatch vs " << cmp_mode << ".\n";
        return false;
    }
    std::cout << "[PASS] cost_draft_tree fused wiring HLS matches " << cmp_mode << ".\n";
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
    ControllerState ctrl_init;
    FusedExpectedOutputs file_expected;

    if (!opts.case_file.empty()) {
        if (!load_file_fixture(opts.case_file, &cfg, &in, &legacy_init, &ctrl_init,
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
        make_synthetic_fixture(&cfg, &in, &legacy_init, &ctrl_init);
    }

    const FusedExpectedOutputs* expected_ptr =
        opts.case_file.empty() ? nullptr : &file_expected;
    if (!run_fused_test(cfg, in, legacy_init, ctrl_init, expected_ptr)) {
        return 1;
    }
    return 0;
}
