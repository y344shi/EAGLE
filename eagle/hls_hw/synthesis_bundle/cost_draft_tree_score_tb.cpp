#include "cost_draft_tree_score_hls.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct KernelCase {
    int batch_size = 0;
    int node_top_k = 0;
    int tree_width = 0;
    int hidden_size = 0;
    int cumu_count = 0;

    bool use_hot_token_id = false;

    std::vector<float> topk_probas_sampling;
    std::vector<int64_t> topk_tokens_sampling;
    std::vector<float> last_layer_scores;
    std::vector<float> input_hidden_states;
    std::vector<int64_t> hot_token_id;

    std::vector<float> expected_curr_layer_scores;
    std::vector<float> expected_sort_layer_scores;
    std::vector<int64_t> expected_sort_layer_indices;
    std::vector<int64_t> expected_cache_topk_indices;
    std::vector<int64_t> expected_parent_indices_in_layer;
    std::vector<float> expected_output_hidden_states;

    // Optional v2 checks for token-path adaptation.
    std::vector<int64_t> expected_topk_tokens_sampling; // After optional hot-token remap.
    std::vector<int64_t> expected_output_tokens; // Top node_top_k tokens by sorted score.
};

struct HlsOutput {
    std::vector<float> curr_layer_scores;
    std::vector<float> sort_layer_scores;
    std::vector<int64_t> sort_layer_indices;
    std::vector<int64_t> cache_topk_indices;
    std::vector<int64_t> parent_indices_in_layer;
    std::vector<float> output_hidden_states;

    std::vector<int64_t> remapped_topk_tokens_sampling;
    std::vector<int64_t> output_tokens;
};

static bool read_float_values(std::istream& in, size_t count, std::vector<float>* out) {
    out->assign(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        if (!(in >> (*out)[i])) return false;
    }
    return true;
}

static bool read_int64_values(std::istream& in, size_t count, std::vector<int64_t>* out) {
    out->assign(count, 0);
    for (size_t i = 0; i < count; ++i) {
        if (!(in >> (*out)[i])) return false;
    }
    return true;
}

static int64_t ref_hot_token_lookup(const KernelCase& tc, int64_t token) {
    if (!tc.use_hot_token_id || tc.hot_token_id.empty()) {
        return token;
    }
    if (token < 0 || token >= static_cast<int64_t>(tc.hot_token_id.size())) {
        return token;
    }
    return tc.hot_token_id[static_cast<size_t>(token)];
}

static bool validate_case_sizes(const KernelCase& tc, std::string* err_msg) {
    const int total_topk = tc.tree_width * tc.node_top_k;
    const size_t topk_count = static_cast<size_t>(tc.batch_size) * total_topk;
    const size_t score_count = static_cast<size_t>(tc.batch_size) * tc.tree_width;
    const size_t topk_choice_count = static_cast<size_t>(tc.batch_size) * tc.node_top_k;
    const size_t hidden_in_count =
        static_cast<size_t>(tc.batch_size) * tc.tree_width * tc.hidden_size;
    const size_t hidden_out_count =
        static_cast<size_t>(tc.batch_size) * tc.node_top_k * tc.hidden_size;

    if (total_topk <= 0 || total_topk > tmac::hls::kCdtSortWidth) {
        *err_msg = "invalid total_topk; must be in [1, 64]";
        return false;
    }
    if (tc.topk_probas_sampling.size() != topk_count) {
        *err_msg = "topk_probas_sampling size mismatch";
        return false;
    }
    if (!tc.topk_tokens_sampling.empty() && tc.topk_tokens_sampling.size() != topk_count) {
        *err_msg = "topk_tokens_sampling size mismatch";
        return false;
    }
    if (tc.last_layer_scores.size() != score_count) {
        *err_msg = "last_layer_scores size mismatch";
        return false;
    }
    if (tc.input_hidden_states.size() != hidden_in_count) {
        *err_msg = "input_hidden_states size mismatch";
        return false;
    }
    if (tc.use_hot_token_id && tc.hot_token_id.empty()) {
        *err_msg = "use_hot_token_id is true but hot_token_id is empty";
        return false;
    }

    if (!tc.expected_curr_layer_scores.empty() &&
        tc.expected_curr_layer_scores.size() != topk_count) {
        *err_msg = "expected_curr_layer_scores size mismatch";
        return false;
    }
    if (!tc.expected_sort_layer_scores.empty() &&
        tc.expected_sort_layer_scores.size() != topk_count) {
        *err_msg = "expected_sort_layer_scores size mismatch";
        return false;
    }
    if (!tc.expected_sort_layer_indices.empty() &&
        tc.expected_sort_layer_indices.size() != topk_count) {
        *err_msg = "expected_sort_layer_indices size mismatch";
        return false;
    }
    if (!tc.expected_cache_topk_indices.empty() &&
        tc.expected_cache_topk_indices.size() != topk_choice_count) {
        *err_msg = "expected_cache_topk_indices size mismatch";
        return false;
    }
    if (!tc.expected_parent_indices_in_layer.empty() &&
        tc.expected_parent_indices_in_layer.size() != topk_choice_count) {
        *err_msg = "expected_parent_indices_in_layer size mismatch";
        return false;
    }
    if (!tc.expected_output_hidden_states.empty() &&
        tc.expected_output_hidden_states.size() != hidden_out_count) {
        *err_msg = "expected_output_hidden_states size mismatch";
        return false;
    }
    if (!tc.expected_topk_tokens_sampling.empty() &&
        tc.expected_topk_tokens_sampling.size() != topk_count) {
        *err_msg = "expected_topk_tokens_sampling size mismatch";
        return false;
    }
    if (!tc.expected_output_tokens.empty() && tc.expected_output_tokens.size() != topk_choice_count) {
        *err_msg = "expected_output_tokens size mismatch";
        return false;
    }

    return true;
}

static bool load_case_file(const std::string& path, KernelCase* tc, std::string* err_msg) {
    std::ifstream in(path);
    if (!in) {
        *err_msg = "cannot open input file: " + path;
        return false;
    }

    std::string key;
    while (in >> key) {
        if (!key.empty() && key[0] == '#') {
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }

        if (key == "meta") {
            if (!(in >> tc->batch_size >> tc->node_top_k >> tc->tree_width >>
                  tc->hidden_size >> tc->cumu_count)) {
                *err_msg = "failed reading meta line";
                return false;
            }
            continue;
        }

        size_t count = 0;
        if (!(in >> count)) {
            *err_msg = "failed reading count for key: " + key;
            return false;
        }

        bool ok = true;
        if (key == "use_hot_token_id") {
            std::vector<int64_t> tmp;
            ok = read_int64_values(in, count, &tmp);
            if (ok && tmp.size() == 1) {
                tc->use_hot_token_id = (tmp[0] != 0);
            } else {
                ok = false;
            }
        } else if (key == "topk_probas_sampling") {
            ok = read_float_values(in, count, &tc->topk_probas_sampling);
        } else if (key == "topk_tokens_sampling") {
            ok = read_int64_values(in, count, &tc->topk_tokens_sampling);
        } else if (key == "last_layer_scores") {
            ok = read_float_values(in, count, &tc->last_layer_scores);
        } else if (key == "input_hidden_states") {
            ok = read_float_values(in, count, &tc->input_hidden_states);
        } else if (key == "hot_token_id") {
            ok = read_int64_values(in, count, &tc->hot_token_id);
        } else if (key == "expected_curr_layer_scores") {
            ok = read_float_values(in, count, &tc->expected_curr_layer_scores);
        } else if (key == "expected_sort_layer_scores") {
            ok = read_float_values(in, count, &tc->expected_sort_layer_scores);
        } else if (key == "expected_sort_layer_indices") {
            ok = read_int64_values(in, count, &tc->expected_sort_layer_indices);
        } else if (key == "expected_cache_topk_indices") {
            ok = read_int64_values(in, count, &tc->expected_cache_topk_indices);
        } else if (key == "expected_parent_indices_in_layer") {
            ok = read_int64_values(in, count, &tc->expected_parent_indices_in_layer);
        } else if (key == "expected_output_hidden_states") {
            ok = read_float_values(in, count, &tc->expected_output_hidden_states);
        } else if (key == "expected_topk_tokens_sampling") {
            ok = read_int64_values(in, count, &tc->expected_topk_tokens_sampling);
        } else if (key == "expected_output_tokens") {
            ok = read_int64_values(in, count, &tc->expected_output_tokens);
        } else {
            std::string skip;
            for (size_t i = 0; i < count; ++i) {
                if (!(in >> skip)) {
                    *err_msg = "failed skipping unknown key payload: " + key;
                    return false;
                }
            }
        }

        if (!ok) {
            *err_msg = "failed reading payload for key: " + key;
            return false;
        }
    }

    if (tc->batch_size <= 0 || tc->node_top_k <= 0 || tc->tree_width <= 0 ||
        tc->hidden_size <= 0) {
        *err_msg = "meta must be provided with positive dimensions";
        return false;
    }

    return validate_case_sizes(*tc, err_msg);
}

static void ref_bitonic_sort_64(float scores[tmac::hls::kCdtSortWidth],
                                int64_t indices[tmac::hls::kCdtSortWidth],
                                int valid_count) {
    for (int size = 2; size <= tmac::hls::kCdtSortWidth; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int tid = 0; tid < tmac::hls::kCdtSortWidth / 2; ++tid) {
                const int i = ((tid / stride) * (stride * 2)) + (tid % stride);
                const int j = i + stride;
                if (i < valid_count && j < valid_count) {
                    const float score_i = scores[i];
                    const float score_j = scores[j];
                    const bool dir = ((i & size) == 0);
                    if ((dir && score_i < score_j) || (!dir && score_i > score_j)) {
                        scores[i] = score_j;
                        scores[j] = score_i;
                        const int64_t tmp = indices[i];
                        indices[i] = indices[j];
                        indices[j] = tmp;
                    }
                }
            }
        }
    }
}

static KernelCase make_synthetic_case() {
    KernelCase tc;
    tc.batch_size = 2;
    tc.node_top_k = 8;
    tc.tree_width = 8;
    tc.hidden_size = 64;
    tc.cumu_count = 17;

    const int total_topk = tc.tree_width * tc.node_top_k;
    tc.topk_probas_sampling.resize(static_cast<size_t>(tc.batch_size) * total_topk);
    tc.topk_tokens_sampling.resize(static_cast<size_t>(tc.batch_size) * total_topk);
    tc.last_layer_scores.resize(static_cast<size_t>(tc.batch_size) * tc.tree_width);
    tc.input_hidden_states.resize(
        static_cast<size_t>(tc.batch_size) * tc.tree_width * tc.hidden_size);

    tc.use_hot_token_id = true;
    const int64_t vocab_size = 512;
    tc.hot_token_id.resize(static_cast<size_t>(vocab_size));
    for (int64_t i = 0; i < vocab_size; ++i) {
        tc.hot_token_id[static_cast<size_t>(i)] = (i * 7 + 3) % vocab_size;
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> prob_dist(0.001f, 0.999f);
    std::uniform_real_distribution<float> score_dist(0.1f, 1.5f);
    std::uniform_real_distribution<float> hidden_dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int64_t> token_dist(0, vocab_size - 1);

    for (size_t i = 0; i < tc.topk_probas_sampling.size(); ++i) {
        tc.topk_probas_sampling[i] = prob_dist(rng);
    }
    for (size_t i = 0; i < tc.topk_tokens_sampling.size(); ++i) {
        tc.topk_tokens_sampling[i] = token_dist(rng);
    }
    for (size_t i = 0; i < tc.last_layer_scores.size(); ++i) {
        tc.last_layer_scores[i] = score_dist(rng);
    }
    for (size_t i = 0; i < tc.input_hidden_states.size(); ++i) {
        tc.input_hidden_states[i] = hidden_dist(rng);
    }

    return tc;
}

static void compute_reference_outputs(KernelCase* tc) {
    const int total_topk = tc->tree_width * tc->node_top_k;
    const size_t topk_count = static_cast<size_t>(tc->batch_size) * total_topk;
    const size_t topk_choice_count =
        static_cast<size_t>(tc->batch_size) * tc->node_top_k;
    const size_t hidden_out_count =
        static_cast<size_t>(tc->batch_size) * tc->node_top_k * tc->hidden_size;

    tc->expected_curr_layer_scores.assign(topk_count, 0.0f);
    tc->expected_sort_layer_scores.assign(topk_count, 0.0f);
    tc->expected_sort_layer_indices.assign(topk_count, 0);
    tc->expected_cache_topk_indices.assign(topk_choice_count, 0);
    tc->expected_parent_indices_in_layer.assign(topk_choice_count, 0);
    tc->expected_output_hidden_states.assign(hidden_out_count, 0.0f);

    if (!tc->topk_tokens_sampling.empty()) {
        tc->expected_topk_tokens_sampling.assign(topk_count, 0);
        tc->expected_output_tokens.assign(topk_choice_count, 0);
    } else {
        tc->expected_topk_tokens_sampling.clear();
        tc->expected_output_tokens.clear();
    }

    for (int b = 0; b < tc->batch_size; ++b) {
        float s_scores[tmac::hls::kCdtSortWidth];
        int64_t s_indices[tmac::hls::kCdtSortWidth];
        int64_t s_tokens[tmac::hls::kCdtSortWidth];

        for (int i = 0; i < tmac::hls::kCdtSortWidth; ++i) {
            s_scores[i] = tmac::hls::kCdtPadScore;
            s_indices[i] = i;
            s_tokens[i] = 0;
        }

        for (int tid = 0; tid < total_topk; ++tid) {
            const int parent = tid / tc->node_top_k;
            const int idx = b * total_topk + tid;
            const float score = tc->topk_probas_sampling[idx] *
                                tc->last_layer_scores[b * tc->tree_width + parent];
            s_scores[tid] = score;
            tc->expected_curr_layer_scores[idx] = score;

            if (!tc->topk_tokens_sampling.empty()) {
                int64_t token = tc->topk_tokens_sampling[idx];
                token = ref_hot_token_lookup(*tc, token);
                s_tokens[tid] = token;
                tc->expected_topk_tokens_sampling[idx] = token;
            }
        }

        ref_bitonic_sort_64(s_scores, s_indices, total_topk);

        for (int tid = 0; tid < total_topk; ++tid) {
            const int out_idx = b * total_topk + tid;
            tc->expected_sort_layer_scores[out_idx] = s_scores[tid];
            tc->expected_sort_layer_indices[out_idx] = s_indices[tid];
            if (tid < tc->node_top_k) {
                const int64_t best_idx = s_indices[tid];
                tc->expected_cache_topk_indices[b * tc->node_top_k + tid] =
                    static_cast<int64_t>(tc->cumu_count) + best_idx;
                tc->expected_parent_indices_in_layer[b * tc->node_top_k + tid] =
                    best_idx / tc->node_top_k;
                if (!tc->expected_output_tokens.empty()) {
                    tc->expected_output_tokens[b * tc->node_top_k + tid] = s_tokens[best_idx];
                }
            }
        }

        for (int k = 0; k < tc->node_top_k; ++k) {
            const int64_t parent = tc->expected_parent_indices_in_layer[b * tc->node_top_k + k];
            const int64_t src_base =
                (static_cast<int64_t>(b) * tc->tree_width + parent) * tc->hidden_size;
            const int64_t dst_base =
                (static_cast<int64_t>(b) * tc->node_top_k + k) * tc->hidden_size;
            for (int h = 0; h < tc->hidden_size; ++h) {
                tc->expected_output_hidden_states[dst_base + h] =
                    tc->input_hidden_states[src_base + h];
            }
        }
    }
}

static HlsOutput run_hls_kernel(const KernelCase& tc) {
    const int total_topk = tc.tree_width * tc.node_top_k;
    const size_t topk_count = static_cast<size_t>(tc.batch_size) * total_topk;
    const size_t topk_choice_count = static_cast<size_t>(tc.batch_size) * tc.node_top_k;
    const size_t hidden_out_count =
        static_cast<size_t>(tc.batch_size) * tc.node_top_k * tc.hidden_size;

    HlsOutput out;
    out.curr_layer_scores.assign(topk_count, 0.0f);
    out.sort_layer_scores.assign(topk_count, 0.0f);
    out.sort_layer_indices.assign(topk_count, -1);
    out.cache_topk_indices.assign(topk_choice_count, -1);
    out.parent_indices_in_layer.assign(topk_choice_count, -1);
    out.output_hidden_states.assign(hidden_out_count, 0.0f);

    const bool has_tokens = !tc.topk_tokens_sampling.empty();
    if (has_tokens) {
        out.remapped_topk_tokens_sampling.assign(topk_count, -1);
        out.output_tokens.assign(topk_choice_count, -1);

        tmac::hls::cost_draft_tree_layer_score_hls_with_tokens(
            tc.topk_probas_sampling.data(),
            tc.topk_tokens_sampling.data(),
            tc.last_layer_scores.data(),
            tc.input_hidden_states.data(),
            tc.hot_token_id.empty() ? nullptr : tc.hot_token_id.data(),
            static_cast<int64_t>(tc.hot_token_id.size()),
            tc.use_hot_token_id,
            tc.batch_size,
            tc.node_top_k,
            tc.tree_width,
            tc.hidden_size,
            tc.cumu_count,
            out.curr_layer_scores.data(),
            out.sort_layer_scores.data(),
            out.sort_layer_indices.data(),
            out.cache_topk_indices.data(),
            out.parent_indices_in_layer.data(),
            out.output_hidden_states.data(),
            out.remapped_topk_tokens_sampling.data(),
            out.output_tokens.data());
    } else {
        tmac::hls::cost_draft_tree_layer_score_hls(
            tc.topk_probas_sampling.data(),
            tc.last_layer_scores.data(),
            tc.input_hidden_states.data(),
            tc.batch_size,
            tc.node_top_k,
            tc.tree_width,
            tc.hidden_size,
            tc.cumu_count,
            out.curr_layer_scores.data(),
            out.sort_layer_scores.data(),
            out.sort_layer_indices.data(),
            out.cache_topk_indices.data(),
            out.parent_indices_in_layer.data(),
            out.output_hidden_states.data());
    }

    return out;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float err = std::fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static size_t count_mismatch(const std::vector<int64_t>& a,
                             const std::vector<int64_t>& b) {
    size_t mismatch = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++mismatch;
    }
    return mismatch;
}

int main(int argc, char** argv) {
    KernelCase tc;
    std::string err_msg;

    std::string input_path;
    bool dry_run = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-file") {
            if (i + 1 >= argc) {
                std::cerr << "[FAIL] --case-file requires a path.\n";
                return 1;
            }
            input_path = argv[++i];
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: cost_draft_tree_score_tb [--case-file <path> | <path>] [--dry-run]\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-' && input_path.empty()) {
            input_path = arg;
        } else {
            std::cerr << "[FAIL] unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (!input_path.empty()) {
        if (!load_case_file(input_path, &tc, &err_msg)) {
            std::cerr << "[FAIL] " << err_msg << "\n";
            return 1;
        }
        if (dry_run) {
            std::cout << "[DRY-RUN] Parsed score case file: " << input_path << "\n";
            std::cout << "[DRY-RUN] dims(B,topk,tree,hidden,cumu,use_hot)= "
                      << tc.batch_size << "," << tc.node_top_k << "," << tc.tree_width << ","
                      << tc.hidden_size << "," << tc.cumu_count << ","
                      << static_cast<int>(tc.use_hot_token_id) << "\n";
            return 0;
        }
        if (tc.expected_curr_layer_scores.empty() || tc.expected_sort_layer_scores.empty() ||
            tc.expected_sort_layer_indices.empty() || tc.expected_cache_topk_indices.empty() ||
            tc.expected_parent_indices_in_layer.empty() ||
            tc.expected_output_hidden_states.empty()) {
            std::cerr << "[FAIL] Input file is missing required expected_* fields for comparison.\n";
            return 1;
        }
        std::cout << "Loaded CUDA golden case from: " << input_path << "\n";
    } else {
        tc = make_synthetic_case();
        compute_reference_outputs(&tc);
        std::cout << "No input file provided, running self-check with synthetic reference.\n";
    }

    HlsOutput hw = run_hls_kernel(tc);

    const float err_curr = max_abs_diff(hw.curr_layer_scores, tc.expected_curr_layer_scores);
    const float err_sort = max_abs_diff(hw.sort_layer_scores, tc.expected_sort_layer_scores);
    const float err_hidden =
        max_abs_diff(hw.output_hidden_states, tc.expected_output_hidden_states);

    const size_t mismatch_sort_idx =
        count_mismatch(hw.sort_layer_indices, tc.expected_sort_layer_indices);
    const size_t mismatch_cache_idx =
        count_mismatch(hw.cache_topk_indices, tc.expected_cache_topk_indices);
    const size_t mismatch_parent =
        count_mismatch(hw.parent_indices_in_layer, tc.expected_parent_indices_in_layer);

    const bool check_remapped_tokens =
        !tc.expected_topk_tokens_sampling.empty() && !hw.remapped_topk_tokens_sampling.empty();
    const bool check_output_tokens =
        !tc.expected_output_tokens.empty() && !hw.output_tokens.empty();

    const size_t mismatch_remapped_tokens =
        check_remapped_tokens
            ? count_mismatch(hw.remapped_topk_tokens_sampling,
                             tc.expected_topk_tokens_sampling)
            : 0;
    const size_t mismatch_output_tokens =
        check_output_tokens
            ? count_mismatch(hw.output_tokens, tc.expected_output_tokens)
            : 0;

    std::cout << "max|curr_layer_scores diff|      = " << err_curr << "\n";
    std::cout << "max|sort_layer_scores diff|      = " << err_sort << "\n";
    std::cout << "max|output_hidden_states diff|   = " << err_hidden << "\n";
    std::cout << "sort_layer_indices mismatches    = " << mismatch_sort_idx << "\n";
    std::cout << "cache_topk_indices mismatches    = " << mismatch_cache_idx << "\n";
    std::cout << "parent_indices mismatches        = " << mismatch_parent << "\n";
    if (check_remapped_tokens) {
        std::cout << "topk_tokens(remapped) mismatches = " << mismatch_remapped_tokens
                  << "\n";
    } else {
        std::cout << "topk_tokens(remapped) mismatches = N/A\n";
    }
    if (check_output_tokens) {
        std::cout << "output_tokens mismatches         = " << mismatch_output_tokens << "\n";
    } else {
        std::cout << "output_tokens mismatches         = N/A\n";
    }

    const bool pass_float = (err_curr <= 1e-6f) && (err_sort <= 1e-6f) && (err_hidden <= 1e-6f);
    const bool pass_int = (mismatch_sort_idx == 0) && (mismatch_cache_idx == 0) &&
                          (mismatch_parent == 0);
    const bool pass_tokens =
        (!check_remapped_tokens || mismatch_remapped_tokens == 0) &&
        (!check_output_tokens || mismatch_output_tokens == 0);

    if (!pass_float || !pass_int || !pass_tokens) {
        std::cerr << "[FAIL] HLS output does not match expected results.\n";
        return 1;
    }

    std::cout << "[PASS] cost_draft_tree score mapping matches expected outputs.\n";
    return 0;
}
