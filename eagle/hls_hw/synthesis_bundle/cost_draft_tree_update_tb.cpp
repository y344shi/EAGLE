#include "cost_draft_tree_update_hls.hpp"
#include "cost_draft_tree_tb_case_io.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct BoolBuffer {
    BoolBuffer() = default;

    explicit BoolBuffer(size_t n)
        : size_(n), data_(n ? new bool[n] : nullptr) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = false;
        }
    }

    BoolBuffer(const BoolBuffer& other)
        : size_(other.size_), data_(other.size_ ? new bool[other.size_] : nullptr) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = other.data_[i];
        }
    }

    BoolBuffer& operator=(const BoolBuffer& other) {
        if (this == &other) return *this;
        BoolBuffer tmp(other);
        swap(tmp);
        return *this;
    }

    BoolBuffer(BoolBuffer&&) = default;
    BoolBuffer& operator=(BoolBuffer&&) = default;

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

struct StateBuffers {
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

struct CaseConfig {
    int batch_size;
    int node_top_k;
    int tree_width;
    int input_count;
    int cumu_count;
    int verify_num;
    int curr_depth;

    int max_input_size;
    int max_node_count;
    int max_verify_num;

    int vocab_size;
    int seed;
};

struct TestCase {
    CaseConfig cfg;

    std::vector<float> topk_probas;     // [B, tree_width * node_top_k]
    std::vector<int64_t> topk_tokens;   // [B, tree_width * node_top_k]
    std::vector<float> sorted_scores;   // [B, tree_width * node_top_k]
    std::vector<int64_t> sorted_indexs; // [B, tree_width * node_top_k]
    std::vector<int64_t> parent_indexs; // [B, node_top_k]
    std::vector<int64_t> topk_indexs;   // [B, tree_width]
    BoolBuffer input_tree_mask;         // [B, tree_width, input_count - 1]

    StateBuffers initial;
};

struct CliOptions {
    std::string case_file;
    bool dry_run = false;
};

static int safe_parent_idx(int64_t idx, int tree_width) {
    if (idx < 0 || idx >= tree_width) {
        return 0;
    }
    return static_cast<int>(idx);
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float err = std::fabs(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

static size_t count_mismatch_i64(const std::vector<int64_t>& a,
                                 const std::vector<int64_t>& b) {
    size_t mismatch = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++mismatch;
        }
    }
    return mismatch;
}

static size_t count_mismatch_bool(const BoolBuffer& a, const BoolBuffer& b) {
    size_t mismatch = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++mismatch;
        }
    }
    return mismatch;
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
                << "Usage: cost_draft_tree_update_tb [--case-file <path>] [--dry-run]\n";
            return false;
        } else {
            *err_msg = "unknown argument: " + arg;
            return false;
        }
    }
    return true;
}

static bool load_case_file(const std::string& path,
                           TestCase* tc,
                           StateBuffers* expected_from_file,
                           bool* has_expected,
                           std::string* err_msg) {
    using namespace tmac::hls::tb_case_io;
    RawCaseMap kv;
    if (!parse_key_count_file(path, &kv, err_msg)) {
        return false;
    }

    std::vector<int> meta;
    if (!read_int_array(kv, "meta", 10, &meta, err_msg, true)) {
        return false;
    }

    tc->cfg.batch_size = meta[0];
    tc->cfg.node_top_k = meta[1];
    tc->cfg.tree_width = meta[2];
    tc->cfg.input_count = meta[3];
    tc->cfg.cumu_count = meta[4];
    tc->cfg.verify_num = meta[5];
    tc->cfg.curr_depth = meta[6];
    tc->cfg.max_input_size = meta[7];
    tc->cfg.max_node_count = meta[8];
    tc->cfg.max_verify_num = meta[9];
    tc->cfg.vocab_size = 1;
    tc->cfg.seed = 0;

    const CaseConfig& cfg = tc->cfg;
    if (cfg.batch_size <= 0 || cfg.node_top_k <= 0 || cfg.tree_width <= 0 ||
        cfg.input_count <= 0 || cfg.max_input_size <= 0 || cfg.max_node_count <= 0 ||
        cfg.max_verify_num <= 0 || cfg.max_input_size < cfg.input_count - 1) {
        *err_msg = "invalid dimensions in meta";
        return false;
    }

    const size_t topk_n = static_cast<size_t>(cfg.batch_size) * cfg.tree_width * cfg.node_top_k;
    const size_t parent_n = static_cast<size_t>(cfg.batch_size) * cfg.node_top_k;
    const size_t tree_n = static_cast<size_t>(cfg.batch_size) * cfg.tree_width;
    const size_t mask_in_n = static_cast<size_t>(cfg.batch_size) * cfg.tree_width *
                             static_cast<size_t>(cfg.input_count - 1);
    const size_t node_n = static_cast<size_t>(cfg.batch_size) * cfg.max_node_count;
    const size_t out_n = static_cast<size_t>(cfg.batch_size) * cfg.node_top_k;
    const size_t work_n = static_cast<size_t>(cfg.batch_size) *
                          static_cast<size_t>(cfg.max_verify_num + cfg.node_top_k);
    const size_t sort_n = static_cast<size_t>(cfg.batch_size) * cfg.max_verify_num;
    const size_t out_mask_n = static_cast<size_t>(cfg.batch_size) * cfg.node_top_k *
                              static_cast<size_t>(cfg.max_input_size + 1);

    if (!read_float_array(kv, "topk_probas", topk_n, &tc->topk_probas, err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "topk_tokens", topk_n, &tc->topk_tokens, err_msg, true)) {
        return false;
    }
    if (!read_float_array(kv, "sorted_scores", topk_n, &tc->sorted_scores, err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "sorted_indexs", topk_n, &tc->sorted_indexs, err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "parent_indexs", parent_n, &tc->parent_indexs, err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "topk_indexs", tree_n, &tc->topk_indexs, err_msg, true)) {
        return false;
    }

    std::vector<bool> input_mask_tmp;
    if (!read_bool_array(kv, "input_tree_mask", mask_in_n, &input_mask_tmp, err_msg, true)) {
        return false;
    }
    tc->input_tree_mask = BoolBuffer(mask_in_n);
    for (size_t i = 0; i < mask_in_n; ++i) {
        tc->input_tree_mask[i] = input_mask_tmp[i];
    }

    if (!read_i64_array(kv, "initial_cumu_tokens", node_n, &tc->initial.cumu_tokens, err_msg,
                        true)) {
        return false;
    }
    if (!read_float_array(kv, "initial_cumu_scores", node_n, &tc->initial.cumu_scores, err_msg,
                          true)) {
        return false;
    }
    if (!read_i64_array(kv, "initial_cumu_deltas", node_n, &tc->initial.cumu_deltas, err_msg,
                        true)) {
        return false;
    }
    if (!read_i64_array(kv, "initial_prev_indexs", node_n, &tc->initial.prev_indexs, err_msg,
                        true)) {
        return false;
    }
    if (!read_i64_array(kv, "initial_next_indexs", node_n, &tc->initial.next_indexs, err_msg,
                        true)) {
        return false;
    }
    if (!read_i64_array(kv, "initial_side_indexs", node_n, &tc->initial.side_indexs, err_msg,
                        true)) {
        return false;
    }
    if (!read_float_array(kv, "initial_output_scores", out_n, &tc->initial.output_scores, err_msg,
                          true)) {
        return false;
    }
    if (!read_i64_array(kv, "initial_output_tokens", out_n, &tc->initial.output_tokens, err_msg,
                        true)) {
        return false;
    }
    if (!read_float_array(kv, "initial_work_scores", work_n, &tc->initial.work_scores, err_msg,
                          true)) {
        return false;
    }
    if (!read_float_array(kv, "initial_sort_scores", sort_n, &tc->initial.sort_scores, err_msg,
                          true)) {
        return false;
    }

    std::vector<bool> init_out_mask_tmp;
    if (!read_bool_array(kv, "initial_output_tree_mask", out_mask_n, &init_out_mask_tmp, err_msg,
                         true)) {
        return false;
    }
    tc->initial.output_tree_mask = BoolBuffer(out_mask_n);
    for (size_t i = 0; i < out_mask_n; ++i) {
        tc->initial.output_tree_mask[i] = init_out_mask_tmp[i];
    }

    const bool any_expected =
        has_key(kv, "expected_cumu_tokens") || has_key(kv, "expected_cumu_scores") ||
        has_key(kv, "expected_output_scores") || has_key(kv, "expected_output_tree_mask");
    *has_expected = any_expected;
    *expected_from_file = StateBuffers{};

    if (!any_expected) {
        return true;
    }

    if (!read_i64_array(kv, "expected_cumu_tokens", node_n, &expected_from_file->cumu_tokens,
                        err_msg, true)) {
        return false;
    }
    if (!read_float_array(kv, "expected_cumu_scores", node_n, &expected_from_file->cumu_scores,
                          err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "expected_cumu_deltas", node_n, &expected_from_file->cumu_deltas,
                        err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "expected_prev_indexs", node_n, &expected_from_file->prev_indexs,
                        err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "expected_next_indexs", node_n, &expected_from_file->next_indexs,
                        err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "expected_side_indexs", node_n, &expected_from_file->side_indexs,
                        err_msg, true)) {
        return false;
    }
    if (!read_float_array(kv, "expected_output_scores", out_n, &expected_from_file->output_scores,
                          err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "expected_output_tokens", out_n, &expected_from_file->output_tokens,
                        err_msg, true)) {
        return false;
    }
    if (!read_float_array(kv, "expected_work_scores", work_n, &expected_from_file->work_scores,
                          err_msg, true)) {
        return false;
    }
    if (!read_float_array(kv, "expected_sort_scores", sort_n, &expected_from_file->sort_scores,
                          err_msg, true)) {
        return false;
    }
    std::vector<bool> exp_out_mask_tmp;
    if (!read_bool_array(kv, "expected_output_tree_mask", out_mask_n, &exp_out_mask_tmp, err_msg,
                         true)) {
        return false;
    }
    expected_from_file->output_tree_mask = BoolBuffer(out_mask_n);
    for (size_t i = 0; i < out_mask_n; ++i) {
        expected_from_file->output_tree_mask[i] = exp_out_mask_tmp[i];
    }

    return true;
}

static TestCase make_case(const CaseConfig& cfg) {
    TestCase tc;
    tc.cfg = cfg;

    const int total_topk = cfg.tree_width * cfg.node_top_k;
    const int mask_in_width = cfg.input_count - 1;

    tc.topk_probas.resize(static_cast<size_t>(cfg.batch_size) * total_topk);
    tc.topk_tokens.resize(static_cast<size_t>(cfg.batch_size) * total_topk);
    tc.sorted_scores.resize(static_cast<size_t>(cfg.batch_size) * total_topk);
    tc.sorted_indexs.resize(static_cast<size_t>(cfg.batch_size) * total_topk);
    tc.parent_indexs.resize(static_cast<size_t>(cfg.batch_size) * cfg.node_top_k);
    tc.topk_indexs.resize(static_cast<size_t>(cfg.batch_size) * cfg.tree_width);
    tc.input_tree_mask =
        BoolBuffer(static_cast<size_t>(cfg.batch_size) * cfg.tree_width * mask_in_width);

    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<float> prob_dist(0.01f, 0.99f);
    std::uniform_real_distribution<float> score_dist(0.01f, 1.0f);
    std::uniform_int_distribution<int64_t> token_dist(0, cfg.vocab_size - 1);
    std::uniform_int_distribution<int> mask_dist(0, 1);

    for (size_t i = 0; i < tc.topk_probas.size(); ++i) {
        tc.topk_probas[i] = prob_dist(rng);
        tc.topk_tokens[i] = token_dist(rng);
    }

    for (int b = 0; b < cfg.batch_size; ++b) {
        std::vector<std::pair<float, int64_t>> scores;
        scores.reserve(total_topk);
        for (int i = 0; i < total_topk; ++i) {
            // Add tiny deterministic bias to reduce equal-score ties.
            const float s = score_dist(rng) + static_cast<float>(total_topk - i) * 1e-6f;
            scores.emplace_back(s, i);
        }
        std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        for (int i = 0; i < total_topk; ++i) {
            const int idx = b * total_topk + i;
            tc.sorted_scores[idx] = scores[static_cast<size_t>(i)].first;
            tc.sorted_indexs[idx] = scores[static_cast<size_t>(i)].second;
        }

        for (int i = 0; i < cfg.node_top_k; ++i) {
            const int idx = b * cfg.node_top_k + i;
            tc.parent_indexs[idx] = tc.sorted_indexs[b * total_topk + i] / cfg.node_top_k;
        }

        // Parents from previous layer, guaranteed in [0, cumu_count).
        for (int i = 0; i < cfg.tree_width; ++i) {
            const int idx = b * cfg.tree_width + i;
            tc.topk_indexs[idx] = (i * 3 + b) % std::max(1, cfg.cumu_count);
        }
    }

    for (size_t i = 0; i < tc.input_tree_mask.size(); ++i) {
        tc.input_tree_mask[i] = (mask_dist(rng) != 0);
    }

    const size_t node_buf = static_cast<size_t>(cfg.batch_size) * cfg.max_node_count;
    const size_t out_buf = static_cast<size_t>(cfg.batch_size) * cfg.node_top_k;
    const size_t work_buf =
        static_cast<size_t>(cfg.batch_size) * (cfg.max_verify_num + cfg.node_top_k);
    const size_t sort_buf = static_cast<size_t>(cfg.batch_size) * cfg.max_verify_num;
    const size_t tree_mask_buf =
        static_cast<size_t>(cfg.batch_size) * cfg.node_top_k * (cfg.max_input_size + 1);

    tc.initial.cumu_tokens.assign(node_buf, -777);
    tc.initial.cumu_scores.assign(node_buf, -3.0f);
    tc.initial.cumu_deltas.assign(node_buf, -1);
    tc.initial.prev_indexs.assign(node_buf, -1);
    tc.initial.next_indexs.assign(node_buf, -2);
    tc.initial.side_indexs.assign(node_buf, -3);

    tc.initial.output_scores.assign(out_buf, -999.0f);
    tc.initial.output_tokens.assign(out_buf, -999);

    tc.initial.work_scores.assign(work_buf, -123.0f);
    tc.initial.sort_scores.assign(sort_buf, -1234.0f);

    tc.initial.output_tree_mask = BoolBuffer(tree_mask_buf);
    for (size_t i = 0; i < tree_mask_buf; ++i) {
        tc.initial.output_tree_mask[i] = ((i % 3) == 0);
    }

    // Pre-populate sort_scores prefix as descending values.
    const int work_size_0 = std::min(cfg.verify_num, cfg.cumu_count);
    for (int b = 0; b < cfg.batch_size; ++b) {
        float base = 2.5f - static_cast<float>(b) * 0.1f;
        for (int i = 0; i < work_size_0; ++i) {
            tc.initial.sort_scores[b * cfg.max_verify_num + i] =
                base - static_cast<float>(i) * 0.01f;
        }
    }

    return tc;
}

static void run_reference(const TestCase& tc, StateBuffers* s) {
    const CaseConfig& cfg = tc.cfg;

    const int num_new_tokens = cfg.node_top_k * cfg.tree_width;
    const int work_size_0 = std::min(cfg.verify_num, cfg.cumu_count);
    const int work_size_1 = std::min(cfg.verify_num, cfg.cumu_count + num_new_tokens);
    const int mask_in_width = cfg.input_count - 1;

    for (int b = 0; b < cfg.batch_size; ++b) {
        const int topk_offset = b * num_new_tokens;
        const int parent_offset = b * cfg.node_top_k;
        const int topk_indexs_offset = b * cfg.tree_width;
        const int output_offset = b * cfg.node_top_k;
        const int node_offset = b * cfg.max_node_count;
        const int verify_offset = b * cfg.max_verify_num;
        const int work_offset = b * (cfg.max_verify_num + cfg.node_top_k);

        // 1) output_scores and output_tokens.
        for (int i = 0; i < cfg.node_top_k; ++i) {
            s->output_scores[output_offset + i] = tc.sorted_scores[topk_offset + i];

            const int parent_idx =
                safe_parent_idx(tc.parent_indexs[parent_offset + i], cfg.tree_width);
            int64_t original_idx = tc.sorted_indexs[topk_offset + i];
            if (original_idx < 0) {
                original_idx = 0;
            }
            const int64_t child_idx = original_idx % cfg.node_top_k;
            const int64_t token_idx = topk_offset + parent_idx * cfg.node_top_k + child_idx;
            s->output_tokens[output_offset + i] = tc.topk_tokens[token_idx];
        }

        // 2) tree mask.
        for (int i = 0; i < cfg.node_top_k; ++i) {
            const int parent_idx =
                safe_parent_idx(tc.parent_indexs[parent_offset + i], cfg.tree_width);
            for (int j = 0; j < mask_in_width; ++j) {
                const int src_offset =
                    (b * cfg.tree_width + parent_idx) * mask_in_width + j;
                const int dst_offset =
                    (b * cfg.node_top_k + i) * (cfg.max_input_size + 1) + j;
                s->output_tree_mask[dst_offset] = tc.input_tree_mask[src_offset];
            }
        }

        // 3) cumulative buffers and link fields.
        const int start = cfg.cumu_count;
        for (int i = 0; i < num_new_tokens; ++i) {
            const int global_idx = start + i;
            if (global_idx < cfg.max_node_count) {
                s->cumu_tokens[node_offset + global_idx] = tc.topk_tokens[topk_offset + i];
                s->cumu_scores[node_offset + global_idx] = tc.topk_probas[topk_offset + i] * 0.9999f;
                s->cumu_deltas[node_offset + global_idx] = cfg.curr_depth;

                const int parent_node_idx_in_tree = i / cfg.node_top_k;
                s->prev_indexs[node_offset + global_idx] =
                    tc.topk_indexs[topk_indexs_offset + parent_node_idx_in_tree];

                s->next_indexs[node_offset + global_idx] = -1;
                const int child_idx_in_node = i % cfg.node_top_k;
                s->side_indexs[node_offset + global_idx] =
                    (child_idx_in_node == cfg.node_top_k - 1) ? -1 : (global_idx + 1);
            }
        }

        // 4) parent next pointers.
        for (int i = 0; i < cfg.tree_width; ++i) {
            const int64_t parent_global_idx = tc.topk_indexs[topk_indexs_offset + i];
            if (parent_global_idx >= 0 && parent_global_idx < cfg.max_node_count) {
                s->next_indexs[node_offset + parent_global_idx] = start + i * cfg.node_top_k;
            }
        }

        // 5a) work_scores.
        for (int i = 0; i < work_size_0; ++i) {
            s->work_scores[work_offset + i] = s->sort_scores[verify_offset + i];
        }
        for (int i = 0; i < cfg.node_top_k; ++i) {
            s->work_scores[work_offset + work_size_0 + i] = s->output_scores[output_offset + i];
        }

        // 5b) merge two descending score segments.
        std::vector<float> merged(static_cast<size_t>(work_size_1), -1e10f);
        int ia = 0;
        int ib = 0;
        for (int i = 0; i < work_size_1; ++i) {
            const bool has_a = (ia < work_size_0);
            const bool has_b = (ib < num_new_tokens);
            const float a = has_a ? s->sort_scores[verify_offset + ia] : -1e10f;
            const float bscore = has_b ? tc.sorted_scores[topk_offset + ib] : -1e10f;

            if (has_a && (!has_b || a >= bscore)) {
                merged[static_cast<size_t>(i)] = a;
                ++ia;
            } else {
                merged[static_cast<size_t>(i)] = bscore;
                ++ib;
            }
        }
        for (int i = 0; i < work_size_1; ++i) {
            s->sort_scores[verify_offset + i] = merged[static_cast<size_t>(i)];
        }
    }
}

static bool run_case_data(const TestCase& tc,
                          const StateBuffers& expected,
                          const std::string& label) {
    const CaseConfig& cfg = tc.cfg;
    StateBuffers actual = tc.initial;

    tmac::hls::cost_draft_tree_update_state_hls(
        tc.topk_probas.data(),
        tc.topk_tokens.data(),
        tc.sorted_scores.data(),
        tc.sorted_indexs.data(),
        tc.parent_indexs.data(),
        tc.topk_indexs.data(),
        tc.input_tree_mask.data(),
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
        actual.cumu_tokens.data(),
        actual.cumu_scores.data(),
        actual.cumu_deltas.data(),
        actual.prev_indexs.data(),
        actual.next_indexs.data(),
        actual.side_indexs.data(),
        actual.output_scores.data(),
        actual.output_tokens.data(),
        actual.work_scores.data(),
        actual.sort_scores.data(),
        actual.output_tree_mask.data());

    const float err_cumu_scores = max_abs_diff(actual.cumu_scores, expected.cumu_scores);
    const float err_output_scores = max_abs_diff(actual.output_scores, expected.output_scores);
    const float err_work_scores = max_abs_diff(actual.work_scores, expected.work_scores);
    const float err_sort_scores = max_abs_diff(actual.sort_scores, expected.sort_scores);

    const size_t mm_cumu_tokens = count_mismatch_i64(actual.cumu_tokens, expected.cumu_tokens);
    const size_t mm_cumu_deltas = count_mismatch_i64(actual.cumu_deltas, expected.cumu_deltas);
    const size_t mm_prev = count_mismatch_i64(actual.prev_indexs, expected.prev_indexs);
    const size_t mm_next = count_mismatch_i64(actual.next_indexs, expected.next_indexs);
    const size_t mm_side = count_mismatch_i64(actual.side_indexs, expected.side_indexs);
    const size_t mm_output_tokens = count_mismatch_i64(actual.output_tokens, expected.output_tokens);
    const size_t mm_mask = count_mismatch_bool(actual.output_tree_mask, expected.output_tree_mask);

    std::cout << "[" << label << "] max|cumu_scores diff|  = " << err_cumu_scores << "\n";
    std::cout << "[" << label << "] max|output_scores diff|= " << err_output_scores << "\n";
    std::cout << "[" << label << "] max|work_scores diff|  = " << err_work_scores << "\n";
    std::cout << "[" << label << "] max|sort_scores diff|  = " << err_sort_scores << "\n";
    std::cout << "[" << label << "] cumu_tokens mismatches = " << mm_cumu_tokens << "\n";
    std::cout << "[" << label << "] cumu_deltas mismatches = " << mm_cumu_deltas << "\n";
    std::cout << "[" << label << "] prev_indexs mismatches = " << mm_prev << "\n";
    std::cout << "[" << label << "] next_indexs mismatches = " << mm_next << "\n";
    std::cout << "[" << label << "] side_indexs mismatches = " << mm_side << "\n";
    std::cout << "[" << label << "] output_tokens mismatches = " << mm_output_tokens << "\n";
    std::cout << "[" << label << "] output_tree_mask mismatches = " << mm_mask << "\n";

    const bool pass_float =
        (err_cumu_scores <= 1e-6f) &&
        (err_output_scores <= 1e-6f) &&
        (err_work_scores <= 1e-6f) &&
        (err_sort_scores <= 1e-6f);

    const bool pass_int =
        (mm_cumu_tokens == 0) &&
        (mm_cumu_deltas == 0) &&
        (mm_prev == 0) &&
        (mm_next == 0) &&
        (mm_side == 0) &&
        (mm_output_tokens == 0) &&
        (mm_mask == 0);

    return pass_float && pass_int;
}

static bool run_synthetic_case(const CaseConfig& cfg, const std::string& label) {
    TestCase tc = make_case(cfg);
    StateBuffers expected = tc.initial;
    run_reference(tc, &expected);
    return run_case_data(tc, expected, label);
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

    if (!opts.case_file.empty()) {
        TestCase tc;
        StateBuffers expected_from_file;
        bool has_expected = false;
        if (!load_case_file(opts.case_file, &tc, &expected_from_file, &has_expected, &err_msg)) {
            std::cerr << "[FAIL] " << err_msg << "\n";
            return 1;
        }

        if (opts.dry_run) {
            std::cout << "[DRY-RUN] Parsed update case file: " << opts.case_file << "\n";
            std::cout << "[DRY-RUN] dims(B,topk,tree,input,cumu,verify,max_node)= "
                      << tc.cfg.batch_size << "," << tc.cfg.node_top_k << ","
                      << tc.cfg.tree_width << "," << tc.cfg.input_count << ","
                      << tc.cfg.cumu_count << "," << tc.cfg.verify_num << ","
                      << tc.cfg.max_node_count << "\n";
            std::cout << "[DRY-RUN] expected payload mode: "
                      << (has_expected ? "from-file" : "reference-generated") << "\n";
            return 0;
        }

        StateBuffers expected = tc.initial;
        if (has_expected) {
            expected = expected_from_file;
        } else {
            run_reference(tc, &expected);
        }
        const bool pass = run_case_data(tc, expected, "file_case");
        if (!pass) {
            std::cerr << "[FAIL] cost_draft_tree update-state case-file check failed.\n";
            return 1;
        }
        std::cout << "[PASS] cost_draft_tree update-state case-file check passed.\n";
        return 0;
    }

    const CaseConfig case_a{
        /*batch_size=*/2,
        /*node_top_k=*/8,
        /*tree_width=*/8,
        /*input_count=*/17,
        /*cumu_count=*/21,
        /*verify_num=*/64,
        /*curr_depth=*/3,
        /*max_input_size=*/64,
        /*max_node_count=*/1024,
        /*max_verify_num=*/64,
        /*vocab_size=*/8192,
        /*seed=*/1234,
    };

    const CaseConfig case_b{
        /*batch_size=*/1,
        /*node_top_k=*/4,
        /*tree_width=*/2,
        /*input_count=*/6,
        /*cumu_count=*/3,
        /*verify_num=*/8,
        /*curr_depth=*/1,
        /*max_input_size=*/16,
        /*max_node_count=*/128,
        /*max_verify_num=*/16,
        /*vocab_size=*/1024,
        /*seed=*/4321,
    };

    const bool pass_a = run_synthetic_case(case_a, "case_a");
    const bool pass_b = run_synthetic_case(case_b, "case_b");

    if (!pass_a || !pass_b) {
        std::cerr << "[FAIL] cost_draft_tree update-state HLS smoke failed.\n";
        return 1;
    }

    std::cout << "[PASS] cost_draft_tree update-state HLS smoke passed.\n";
    return 0;
}
