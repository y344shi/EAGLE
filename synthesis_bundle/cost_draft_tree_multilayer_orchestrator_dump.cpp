#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#define main cost_draft_tree_multilayer_orchestrator_tb_main
#include "cost_draft_tree_multilayer_orchestrator_tb.cpp"
#undef main

namespace {

struct DumpOptions {
    std::string case_file;
    std::string output_file;
    bool real_slm = true;
};

void usage() {
    std::cerr
        << "Usage: cost_draft_tree_multilayer_orchestrator_dump"
        << " --case-file <path> --output <path> [--real-slm|--recurrent-replay]\n";
}

bool parse_dump_cli(int argc, char** argv, DumpOptions* opts, std::string* err_msg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-file") {
            if (i + 1 >= argc) {
                *err_msg = "--case-file requires a path";
                return false;
            }
            opts->case_file = argv[++i];
        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                *err_msg = "--output requires a path";
                return false;
            }
            opts->output_file = argv[++i];
        } else if (arg == "--real-slm") {
            opts->real_slm = true;
        } else if (arg == "--recurrent-replay") {
            opts->real_slm = false;
        } else if (arg == "-h" || arg == "--help") {
            usage();
            return false;
        } else {
            *err_msg = "unknown arg: " + arg;
            return false;
        }
    }

    if (opts->case_file.empty()) {
        *err_msg = "--case-file is required";
        return false;
    }
    if (opts->output_file.empty()) {
        *err_msg = "--output is required";
        return false;
    }
    return true;
}

void run_dump_uut(const CaseData& c,
                  const SlmArtifacts& a,
                  const std::vector<pack512>& prefill_fc_weight,
                  const std::vector<float>& prefill_fc_scales,
                  bool real_slm,
                  RuntimeState* s,
                  std::vector<int64_t>* node_to_hbm_slot_out) {
    init_runtime(c, s);
    std::vector<vec_t<VEC_W>> hbm_k = a.hbm_k;
    std::vector<vec_t<VEC_W>> hbm_v = a.hbm_v;
    std::vector<int64_t> node_to_hbm_slot = c.node_to_hbm_slot_init;
    const int accepted_count =
        std::min(static_cast<int>(c.accepted_draft_node_ids.size()), kContiguousKvMaxAccepted);

    if (!real_slm) {
        eagle4_draft_set_recurrent_replay(
            c.recurrent_topk_probas.data(),
            c.recurrent_topk_tokens.data(),
            c.tree_depth,
            c.batch_size,
            c.max_tree_width,
            c.node_top_k);
    }

    eagle4_draft(
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
        a.draft_embed_tokens_weight.data(),
        c.prefix_len,
        c.enable_accepted_kv_compact,
        c.accepted_draft_node_ids.data(),
        accepted_count,
        node_to_hbm_slot.data(),
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
        c.initial_hidden_states.data(),
        c.enable_prefill_stage,
        c.prefill_input_hidden_states_3h.data(),
        c.prefill_input_embed_states.data(),
        prefill_fc_weight.data(),
        prefill_fc_scales.data());

    if (!real_slm) {
        eagle4_draft_clear_recurrent_replay();
    }
    *node_to_hbm_slot_out = std::move(node_to_hbm_slot);
}

template <typename T>
void write_json_array(std::ostream& os, const char* name, const std::vector<T>& values, bool last) {
    os << "  \"" << name << "\": [";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) os << ", ";
        if constexpr (std::is_floating_point_v<T>) {
            os << std::setprecision(std::numeric_limits<T>::max_digits10) << values[i];
        } else {
            os << values[i];
        }
    }
    os << "]";
    os << (last ? "\n" : ",\n");
}

void write_json_scalar(std::ostream& os, const char* name, int value) {
    os << "  \"" << name << "\": " << value << ",\n";
}

void write_json_bool(std::ostream& os, const char* name, bool value, bool last) {
    os << "  \"" << name << "\": " << (value ? "true" : "false");
    os << (last ? "\n" : ",\n");
}

}  // namespace

int main(int argc, char** argv) {
    DumpOptions opts;
    std::string err_msg;
    if (!parse_dump_cli(argc, argv, &opts, &err_msg)) {
        if (!err_msg.empty()) {
            std::cerr << "[FAIL] " << err_msg << "\n";
        }
        return err_msg.empty() ? 0 : 1;
    }

    CaseData c;
    if (!load_case_file(opts.case_file, &c, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }
    if (!validate_case(c, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }

    SlmArtifacts artifacts;
    const std::filesystem::path case_dir =
        std::filesystem::absolute(std::filesystem::path(opts.case_file)).parent_path();
    if (!load_slm_artifacts(case_dir, c, &artifacts, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }

    std::vector<pack512> prefill_fc_weight;
    std::vector<float> prefill_fc_scales;
    build_synthetic_prefill_fc(c, &prefill_fc_weight, &prefill_fc_scales);

    RuntimeState state;
    std::vector<int64_t> node_to_hbm_slot;
    run_dump_uut(
        c,
        artifacts,
        prefill_fc_weight,
        prefill_fc_scales,
        opts.real_slm,
        &state,
        &node_to_hbm_slot);

    std::ofstream ofs(opts.output_file);
    if (!ofs) {
        std::cerr << "[FAIL] unable to open output file: " << opts.output_file << "\n";
        return 1;
    }

    ofs << "{\n";
    write_json_scalar(ofs, "io_tree_width", state.io_tree_width);
    write_json_scalar(ofs, "io_verify_num", state.io_verify_num);
    write_json_scalar(ofs, "io_cumu_count", state.io_cumu_count);
    write_json_scalar(ofs, "executed_depths", state.executed_depths);
    write_json_array(ofs, "cumu_tokens", state.cumu_tokens, false);
    write_json_array(ofs, "cumu_scores", state.cumu_scores, false);
    write_json_array(ofs, "cumu_deltas", state.cumu_deltas, false);
    write_json_array(ofs, "prev_indexs", state.prev_indexs, false);
    write_json_array(ofs, "next_indexs", state.next_indexs, false);
    write_json_array(ofs, "side_indexs", state.side_indexs, false);
    write_json_array(ofs, "output_scores", state.output_scores, false);
    write_json_array(ofs, "output_tokens", state.output_tokens, false);
    write_json_array(ofs, "work_scores", state.work_scores, false);
    write_json_array(ofs, "sort_scores", state.sort_scores, false);
    write_json_array(ofs, "output_hidden_states", state.output_hidden_states, false);
    write_json_array(ofs, "cache_topk_indices", state.cache_topk_indices, false);
    write_json_array(ofs, "node_to_hbm_slot", node_to_hbm_slot, false);
    write_json_bool(ofs, "stopped_early", state.stopped_early, true);
    ofs << "}\n";
    ofs.close();

    std::cerr << "[info] dumped orchestrator state to " << opts.output_file
              << " (real_slm=" << (opts.real_slm ? "true" : "false") << ")\n";
    return 0;
}
