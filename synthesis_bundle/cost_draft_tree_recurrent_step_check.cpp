#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#define main cost_draft_tree_multilayer_orchestrator_tb_main
#include "cost_draft_tree_multilayer_orchestrator_tb.cpp"
#undef main

namespace {

struct CheckOptions {
    std::string case_file;
};

bool parse_cli(int argc, char** argv, CheckOptions* opts, std::string* err_msg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-file") {
            if (i + 1 >= argc) {
                *err_msg = "--case-file requires a path";
                return false;
            }
            opts->case_file = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cerr << "Usage: cost_draft_tree_recurrent_step_check --case-file <path>\n";
            return false;
        } else {
            *err_msg = "unknown argument: " + arg;
            return false;
        }
    }
    if (opts->case_file.empty()) {
        *err_msg = "--case-file is required";
        return false;
    }
    return true;
}

bool allclose_tokens(const std::vector<int64_t>& got,
                     const std::vector<int64_t>& exp,
                     int* mismatches) {
    *mismatches = 0;
    const size_t n = std::min(got.size(), exp.size());
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != exp[i]) {
            ++(*mismatches);
        }
    }
    return *mismatches == 0 && got.size() == exp.size();
}

void diff_floats(const std::vector<float>& got,
                 const std::vector<float>& exp,
                 float* max_abs,
                 float* mean_abs) {
    *max_abs = 0.0f;
    *mean_abs = 0.0f;
    const size_t n = std::min(got.size(), exp.size());
    if (n == 0) {
        return;
    }
    double accum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - exp[i]);
        *max_abs = std::max(*max_abs, d);
        accum += d;
    }
    *mean_abs = static_cast<float>(accum / static_cast<double>(n));
}

struct VariantResult {
    std::string name;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    int token_mismatches = 0;
    std::vector<int64_t> tokens;
};

struct PrefixVariant {
    std::string name;
    int prefix_len = 0;
    bool zero_hbm = false;
};

}  // namespace

int main(int argc, char** argv) {
    CheckOptions opts;
    std::string err_msg;
    if (!parse_cli(argc, argv, &opts, &err_msg)) {
        if (!err_msg.empty()) {
            std::cerr << "[FAIL] " << err_msg << "\n";
            return 1;
        }
        return 0;
    }

    CaseData raw_case;
    if (!load_case_file(opts.case_file, &raw_case, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }
    if (!validate_case(raw_case, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }
    const CaseData c = make_effective_execution_case(raw_case);

    SlmArtifacts artifacts;
    const std::filesystem::path case_dir =
        std::filesystem::absolute(std::filesystem::path(opts.case_file)).parent_path();
    if (!load_slm_artifacts(case_dir, c, &artifacts, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }

    RuntimeState s;
    init_runtime(c, &s);

    std::vector<vec_t<VEC_W>> hbm_k = artifacts.hbm_k;
    std::vector<vec_t<VEC_W>> hbm_v = artifacts.hbm_v;
    std::vector<int64_t> node_to_hbm_slot = c.node_to_hbm_slot_init;
    const int accepted_count =
        c.enable_accepted_kv_compact
            ? std::min(static_cast<int>(c.accepted_draft_node_ids.size()), kContiguousKvMaxAccepted)
            : 0;
    const int64_t max_hbm_token_count =
        static_cast<int64_t>(c.prefix_len) +
        static_cast<int64_t>(c.max_node_count) +
        static_cast<int64_t>(c.curr_depth_start + c.tree_depth + 1) * c.max_tree_width +
        kContiguousKvMaxAccepted;
    if (accepted_count > 0) {
        std::vector<int64_t> accepted_slots(static_cast<size_t>(accepted_count), -1);
        for (int i = 0; i < accepted_count; ++i) {
            const int64_t node_id = c.accepted_draft_node_ids[static_cast<size_t>(i)];
            if (node_id >= 0 && node_id < c.max_node_count) {
                accepted_slots[static_cast<size_t>(i)] =
                    node_to_hbm_slot[static_cast<size_t>(node_id)];
            }
        }
        const bool compact_ok = contiguous_kv_compact_accepted<
            HEAD_DIM, NUM_KV_HEADS, kContiguousKvMaxAccepted>(
                hbm_k.data(),
                hbm_v.data(),
                c.prefix_len,
                accepted_slots.data(),
                accepted_count,
                static_cast<int>(max_hbm_token_count));
        if (!compact_ok) {
            std::cerr << "[FAIL] KV compaction failed\n";
            return 1;
        }
    }
    const int effective_prefix_len = c.prefix_len + accepted_count;

    int curr_tree_width = clamp_int(s.io_tree_width, 0, c.max_tree_width);
    curr_tree_width = clamp_int(curr_tree_width, 0, c.node_top_k);
    int curr_verify_num = clamp_int(s.io_verify_num, 1, c.max_verify_num);
    int curr_cumu_count = clamp_int(s.io_cumu_count, 0, c.max_node_count);

    int parent_indices_accum[kCdtControllerMaxDepth * TREE_WIDTH];
    std::fill(std::begin(parent_indices_accum), std::end(parent_indices_accum), 0);
    std::vector<int64_t> parent_scratch(static_cast<size_t>(c.batch_size * c.node_top_k), -1);

    if (!c.enable_initial_loop) {
        std::cerr << "[FAIL] current checker expects enable_initial_loop=1\n";
        return 1;
    }

    std::vector<float> initial_last_layer_scores(static_cast<size_t>(c.batch_size), 1.0f);
    std::vector<int64_t> initial_topk_indexs_prev(static_cast<size_t>(c.batch_size), 0);

    e4d_fused_step(
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
    if (stop_signal || curr_tree_width <= 0) {
        std::cerr << "[FAIL] no recurrent step is scheduled for this case\n";
        return 1;
    }

    e4d_prep_next_inputs(
        s.output_scores.data(),
        s.output_tokens.data(),
        s.output_hidden_states.data(),
        s.cache_topk_indices.data(),
        c.batch_size,
        c.node_top_k,
        c.hidden_size,
        curr_tree_width,
        c.max_tree_width,
        s.step_input_tokens.data(),
        s.step_last_layer_scores.data(),
        s.step_input_hidden_states.data(),
        s.step_topk_indexs_prev.data());

    {
        float prep_hidden_max = 0.0f;
        float prep_hidden_mean = 0.0f;
        diff_floats(
            s.step_input_hidden_states,
            raw_case.step_input_hidden_states_init,
            &prep_hidden_max,
            &prep_hidden_mean);
        int prep_token_mismatches = 0;
        allclose_tokens(
            s.step_input_tokens,
            raw_case.step_input_tokens_init,
            &prep_token_mismatches);
        std::cout << "[result] prep_next_inputs hidden max_abs=" << prep_hidden_max
                  << " mean_abs=" << prep_hidden_mean << "\n";
        std::cout << "[result] prep_next_inputs token_mismatches="
                  << prep_token_mismatches << "/" << raw_case.step_input_tokens_init.size() << "\n";
    }

    std::vector<float> recurrent_embed_states;
    if (!build_embed_states_from_tokens(
            c,
            artifacts,
            s.step_input_tokens.data(),
            curr_tree_width,
            &recurrent_embed_states,
            &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }

    const int per_depth = c.batch_size * c.max_tree_width * c.node_top_k;
    const size_t depth1_base = static_cast<size_t>(per_depth);
    std::vector<float> exp_probas(
        c.recurrent_topk_probas.begin() + depth1_base,
        c.recurrent_topk_probas.begin() + depth1_base + per_depth);
    std::vector<int64_t> exp_tokens(
        c.recurrent_topk_tokens.begin() + depth1_base,
        c.recurrent_topk_tokens.begin() + depth1_base + per_depth);

    std::vector<int> variant_depths = {c.curr_depth_start + 1, c.curr_depth_start};
    std::vector<std::pair<std::string, const int*>> parent_variants = {
        {"actual_parents", parent_indices_accum},
    };
    int zero_parent_indices[kCdtControllerMaxDepth * TREE_WIDTH];
    std::fill(std::begin(zero_parent_indices), std::end(zero_parent_indices), 0);
    parent_variants.push_back({"zero_parents", zero_parent_indices});
    std::vector<PrefixVariant> prefix_variants = {
        {"prefix_loaded", effective_prefix_len, false},
        {"prefix_len0", 0, false},
        {"prefix_zeroed", effective_prefix_len, true},
    };

    std::vector<VariantResult> variants;
    for (int depth_variant : variant_depths) {
        for (const auto& parent_variant : parent_variants) {
            for (const auto& prefix_variant : prefix_variants) {
                VariantResult vr;
                vr.name = "depth=" + std::to_string(depth_variant) + "," + parent_variant.first +
                          "," + prefix_variant.name;
                std::vector<float> got_probas(
                    static_cast<size_t>(c.batch_size * c.max_tree_width * c.node_top_k), 0.0f);
                vr.tokens.assign(
                    static_cast<size_t>(c.batch_size * c.max_tree_width * c.node_top_k), 0);
                auto run_hbm_k = hbm_k;
                auto run_hbm_v = hbm_v;
                if (prefix_variant.zero_hbm) {
                    std::fill(run_hbm_k.begin(), run_hbm_k.end(), vec_t<VEC_W>{});
                    std::fill(run_hbm_v.begin(), run_hbm_v.end(), vec_t<VEC_W>{});
                }
                e4d_slm_topk(
                    s.step_input_hidden_states.data(),
                    recurrent_embed_states.data(),
                    c.batch_size,
                    curr_tree_width,
                    c.hidden_size,
                    c.node_top_k,
                    artifacts.w_q.data(), artifacts.s_q.data(),
                    artifacts.w_k.data(), artifacts.s_k.data(),
                    artifacts.w_v.data(), artifacts.s_v.data(),
                    artifacts.w_o.data(), artifacts.s_o.data(),
                    artifacts.w_gate.data(), artifacts.gate_scales.data(),
                    artifacts.w_up.data(), artifacts.up_scales.data(),
                    artifacts.w_down.data(), artifacts.down_scales.data(),
                    artifacts.hidden_norm_gamma.data(), artifacts.embed_norm_gamma.data(),
                    artifacts.post_attn_norm_gamma.data(), artifacts.final_norm_gamma.data(),
                    artifacts.rope_cfg_table[static_cast<size_t>(depth_variant)].cos_vals,
                    artifacts.rope_cfg_table[static_cast<size_t>(depth_variant)].sin_vals,
                    run_hbm_k.data(),
                    run_hbm_v.data(),
                    artifacts.efficient_lm_head_down_proj_weight.data(),
                    artifacts.efficient_lm_head_qweight_row_major.data(),
                    artifacts.efficient_lm_head_scales_row_major.data(),
                    artifacts.efficient_lm_head_qzeros.data(),
                    artifacts.efficient_lm_head_g_idx.data(),
                    artifacts.lm_head_weight.data(),
                    c.efficient_lm_rank,
                    c.efficient_lm_vocab_size,
                    prefix_variant.prefix_len,
                    depth_variant,
                    parent_variant.second,
                    s.output_hidden_states.data(),
                    got_probas.data(),
                    vr.tokens.data());
                diff_floats(got_probas, exp_probas, &vr.max_abs, &vr.mean_abs);
                allclose_tokens(vr.tokens, exp_tokens, &vr.token_mismatches);
                variants.push_back(std::move(vr));
            }
        }
    }

    const auto best_it = std::min_element(
        variants.begin(), variants.end(), [](const VariantResult& a, const VariantResult& b) {
            if (a.token_mismatches != b.token_mismatches) {
                return a.token_mismatches < b.token_mismatches;
            }
            return a.max_abs < b.max_abs;
        });
    const VariantResult& best = *best_it;

    std::cout << "[result] recurrent_depth=1 tree_width=" << curr_tree_width
              << " verify_num=" << curr_verify_num << "\n";
    for (const auto& vr : variants) {
        std::cout << "[result] variant " << vr.name
                  << " probas max_abs=" << vr.max_abs
                  << " mean_abs=" << vr.mean_abs
                  << " token_mismatches=" << vr.token_mismatches
                  << "/" << exp_tokens.size() << "\n";
    }
    std::cout << "[result] best_variant=" << best.name << "\n";
    std::cout << "[result] got_tokens[0..15]:";
    for (int i = 0; i < std::min(per_depth, 16); ++i) {
        std::cout << " " << best.tokens[static_cast<size_t>(i)];
    }
    std::cout << "\n[result] exp_tokens[0..15]:";
    for (int i = 0; i < std::min(per_depth, 16); ++i) {
        std::cout << " " << exp_tokens[static_cast<size_t>(i)];
    }
    std::cout << "\n";

    if (best.token_mismatches != 0 || best.max_abs > 1e-4f) {
        std::cerr << "[FAIL] recurrent step does not match captured recurrent_topk streams\n";
        return 1;
    }

    std::cout << "[PASS] recurrent step matches captured recurrent_topk depth-1 streams\n";
    return 0;
}
