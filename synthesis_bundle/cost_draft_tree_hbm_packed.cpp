#include "cost_draft_tree_hbm_packed.hpp"

#include "cost_draft_tree_fused_wiring_hls.hpp"

namespace {

using tmac::hls::DOWN_OUTPUT;
using tmac::hls::HEAD_DIM;
using tmac::hls::HIDDEN;
using tmac::hls::INTERMEDIATE;
using tmac::hls::NUM_KV_HEADS;
using tmac::hls::QKV_INPUT;
using tmac::hls::RopeConfig;
using tmac::hls::TREE_WIDTH;
using tmac::hls::VEC_W;
using tmac::hls::align_words;
using tmac::hls::bytes_to_hbm_words;
using tmac::hls::hbm_word256_t;
using tmac::hls::pack512;
using tmac::hls::pc_word_offset_ptr;

constexpr int64_t kProjGroupSize = 128;

constexpr int64_t kQPackCount = (static_cast<int64_t>(QKV_INPUT) * HIDDEN) / 128;
constexpr int64_t kKPackCount = (static_cast<int64_t>(QKV_INPUT) * (NUM_KV_HEADS * HEAD_DIM)) / 128;
constexpr int64_t kVPackCount = (static_cast<int64_t>(QKV_INPUT) * (NUM_KV_HEADS * HEAD_DIM)) / 128;
constexpr int64_t kOPackCount = (static_cast<int64_t>(HIDDEN) * HIDDEN) / 128;
constexpr int64_t kGatePackCount = (static_cast<int64_t>(HIDDEN) * INTERMEDIATE) / 128;
constexpr int64_t kUpPackCount = (static_cast<int64_t>(HIDDEN) * INTERMEDIATE) / 128;
constexpr int64_t kDownPackCount = (static_cast<int64_t>(INTERMEDIATE) * DOWN_OUTPUT) / 128;

constexpr int64_t kQScaleCount = (static_cast<int64_t>(QKV_INPUT) / kProjGroupSize) * HIDDEN;
constexpr int64_t kKScaleCount =
    (static_cast<int64_t>(QKV_INPUT) / kProjGroupSize) * (NUM_KV_HEADS * HEAD_DIM);
constexpr int64_t kVScaleCount =
    (static_cast<int64_t>(QKV_INPUT) / kProjGroupSize) * (NUM_KV_HEADS * HEAD_DIM);
constexpr int64_t kOScaleCount = (static_cast<int64_t>(HIDDEN) / kProjGroupSize) * HIDDEN;
constexpr int64_t kGateScaleCount = (static_cast<int64_t>(HIDDEN) / kProjGroupSize) * INTERMEDIATE;
constexpr int64_t kUpScaleCount = (static_cast<int64_t>(HIDDEN) / kProjGroupSize) * INTERMEDIATE;
constexpr int64_t kDownScaleCount =
    (static_cast<int64_t>(INTERMEDIATE) / kProjGroupSize) * DOWN_OUTPUT;

constexpr int64_t kKvVecsPerToken = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;

struct PcCursor {
    int64_t words = 0;

    int64_t alloc_bytes(int64_t bytes, int64_t align_to_words = 1) {
        words = align_words(words, align_to_words);
        const int64_t base = words;
        words += bytes_to_hbm_words(bytes);
        return base;
    }
};

struct E4dPackedLayout32Pc {
    int64_t pc_words[32] = {};

    int64_t pc00_w_q = 0;
    int64_t pc00_s_q = 0;
    int64_t pc01_w_k = 0;
    int64_t pc01_s_k = 0;
    int64_t pc02_w_v = 0;
    int64_t pc02_s_v = 0;
    int64_t pc03_w_o = 0;
    int64_t pc03_s_o = 0;
    int64_t pc04_w_gate = 0;
    int64_t pc04_gate_scales = 0;
    int64_t pc05_w_up = 0;
    int64_t pc05_up_scales = 0;
    int64_t pc06_w_down = 0;
    int64_t pc06_down_scales = 0;

    int64_t pc07_hidden_norm_gamma = 0;
    int64_t pc07_embed_norm_gamma = 0;
    int64_t pc07_post_attn_norm_gamma = 0;
    int64_t pc07_final_norm_gamma = 0;
    int64_t pc07_rope_cfg_table = 0;

    int64_t pc10_efficient_lm_head_down_proj_weight = 0;
    int64_t pc10_efficient_lm_head_qweight_row_major = 0;
    int64_t pc11_efficient_lm_head_scales_row_major = 0;
    int64_t pc11_efficient_lm_head_qzeros = 0;
    int64_t pc11_efficient_lm_head_g_idx = 0;

    int64_t pc12_lm_head_weight = 0;
    int64_t pc13_draft_embed_tokens_weight = 0;

    int64_t pc14_step_input_tokens = 0;
    int64_t pc14_step_last_layer_scores = 0;
    int64_t pc14_step_topk_indexs_prev = 0;
    int64_t pc14_step_topk_probas_sampling = 0;
    int64_t pc14_step_topk_tokens_sampling = 0;

    int64_t pc15_step_input_hidden_states = 0;
    int64_t pc15_output_hidden_states = 0;

    int64_t pc16_cumu_tokens = 0;
    int64_t pc17_cumu_scores = 0;
    int64_t pc18_cumu_deltas = 0;
    int64_t pc19_prev_indexs = 0;
    int64_t pc20_next_indexs = 0;
    int64_t pc21_side_indexs = 0;

    int64_t pc22_output_scores = 0;
    int64_t pc22_output_tokens = 0;
    int64_t pc22_work_scores = 0;
    int64_t pc22_sort_scores = 0;
    int64_t pc22_cache_topk_indices = 0;

    int64_t pc23_node_to_hbm_slot = 0;
    int64_t pc23_accepted_draft_node_ids = 0;
    int64_t pc23_io_tree_width = 0;
    int64_t pc23_io_verify_num = 0;
    int64_t pc23_io_cumu_count = 0;
    int64_t pc23_executed_depths = 0;
    int64_t pc23_stopped_early = 0;

    int64_t pc24_initial_logits = 0;
    int64_t pc25_initial_candidate_indices = 0;
    int64_t pc25_initial_topk_probas = 0;
    int64_t pc25_initial_topk_tokens = 0;
    int64_t pc25_initial_hidden_states = 0;

    int64_t pc26_prefill_input_hidden_states_3h = 0;
    int64_t pc26_prefill_input_embed_states = 0;
    int64_t pc27_prefill_fc_weight = 0;
    int64_t pc27_prefill_fc_scales = 0;

    int64_t pc28_dbg_curr_layer_scores = 0;
    int64_t pc28_dbg_sort_layer_scores = 0;
    int64_t pc29_dbg_sort_layer_indices = 0;
    int64_t pc29_dbg_parent_indices_in_layer = 0;
    int64_t pc29_dbg_remapped_topk_tokens = 0;

    int64_t pc30_policy_next_tree_width = 0;
    int64_t pc30_policy_next_verify_num = 0;
    int64_t pc30_policy_stop_signal = 0;
    int64_t pc30_hot_token_id = 0;
};

E4dPackedLayout32Pc build_layout(
    int tree_depth,
    int curr_depth_start,
    int prefix_len,
    int policy_depth,
    int batch_size,
    int node_top_k,
    int hidden_size,
    int max_node_count,
    int max_verify_num,
    int max_tree_width,
    int initial_logits_width,
    int64_t hot_token_vocab_size,
    int efficient_lm_vocab_size) {
    E4dPackedLayout32Pc out{};
    PcCursor pc[32];

    const int64_t step_flat = static_cast<int64_t>(batch_size) * max_tree_width;
    const int64_t topk_flat = step_flat * node_top_k;
    const int64_t out_flat = static_cast<int64_t>(batch_size) * node_top_k;
    const int64_t node_flat = static_cast<int64_t>(batch_size) * max_node_count;
    const int64_t hidden_step_flat = step_flat * hidden_size;
    const int64_t hidden_out_flat = out_flat * hidden_size;
    const int64_t work_flat =
        static_cast<int64_t>(batch_size) * (max_verify_num + node_top_k);
    const int64_t sort_flat = static_cast<int64_t>(batch_size) * max_verify_num;
    const int64_t init_logits_flat = static_cast<int64_t>(batch_size) * initial_logits_width;
    const int64_t init_hidden_flat = static_cast<int64_t>(batch_size) * hidden_size;
    const int64_t prefill_hidden_3h_flat = static_cast<int64_t>(batch_size) * 3 * hidden_size;
    const int64_t prefill_embed_flat = static_cast<int64_t>(batch_size) * hidden_size;
    const int64_t prefill_fc_pack_count = (static_cast<int64_t>(3) * hidden_size * hidden_size) / 128;
    const int64_t prefill_fc_scale_count =
        (static_cast<int64_t>(3) * hidden_size / kProjGroupSize) * hidden_size;

    const int64_t kv_token_count =
        prefix_len + max_node_count +
        static_cast<int64_t>(curr_depth_start + tree_depth + 1) * max_tree_width +
        tmac::hls::kContiguousKvMaxAccepted;
    const int64_t kv_vec_count = kv_token_count * kKvVecsPerToken;
    (void)kv_vec_count;  // layout uses dedicated PCs for KV with base offset 0.

    out.pc00_w_q = pc[0].alloc_bytes(sizeof(pack512) * kQPackCount, 2);
    out.pc00_s_q = pc[0].alloc_bytes(sizeof(float) * kQScaleCount, 1);

    out.pc01_w_k = pc[1].alloc_bytes(sizeof(pack512) * kKPackCount, 2);
    out.pc01_s_k = pc[1].alloc_bytes(sizeof(float) * kKScaleCount, 1);

    out.pc02_w_v = pc[2].alloc_bytes(sizeof(pack512) * kVPackCount, 2);
    out.pc02_s_v = pc[2].alloc_bytes(sizeof(float) * kVScaleCount, 1);

    out.pc03_w_o = pc[3].alloc_bytes(sizeof(pack512) * kOPackCount, 2);
    out.pc03_s_o = pc[3].alloc_bytes(sizeof(float) * kOScaleCount, 1);

    out.pc04_w_gate = pc[4].alloc_bytes(sizeof(pack512) * kGatePackCount, 2);
    out.pc04_gate_scales = pc[4].alloc_bytes(sizeof(float) * kGateScaleCount, 1);

    out.pc05_w_up = pc[5].alloc_bytes(sizeof(pack512) * kUpPackCount, 2);
    out.pc05_up_scales = pc[5].alloc_bytes(sizeof(float) * kUpScaleCount, 1);

    out.pc06_w_down = pc[6].alloc_bytes(sizeof(pack512) * kDownPackCount, 2);
    out.pc06_down_scales = pc[6].alloc_bytes(sizeof(float) * kDownScaleCount, 1);

    out.pc07_hidden_norm_gamma = pc[7].alloc_bytes(sizeof(float) * HIDDEN, 1);
    out.pc07_embed_norm_gamma = pc[7].alloc_bytes(sizeof(float) * HIDDEN, 1);
    out.pc07_post_attn_norm_gamma = pc[7].alloc_bytes(sizeof(float) * HIDDEN, 1);
    out.pc07_final_norm_gamma = pc[7].alloc_bytes(sizeof(float) * HIDDEN, 1);
    out.pc07_rope_cfg_table = pc[7].alloc_bytes(
        sizeof(RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>) *
            tmac::hls::kCdtControllerMaxDepth,
        1);

    out.pc10_efficient_lm_head_down_proj_weight = pc[10].alloc_bytes(
        sizeof(uint16_t) * tmac::hls::kEagle4LmRankMax * tmac::hls::kEagle4LmHiddenMax, 1);
    out.pc10_efficient_lm_head_qweight_row_major = pc[10].alloc_bytes(
        sizeof(int32_t) * tmac::hls::kLmTcVocab * tmac::hls::kLmMaxInPacks, 1);

    out.pc11_efficient_lm_head_scales_row_major = pc[11].alloc_bytes(
        sizeof(uint16_t) * tmac::hls::kLmTcVocab * tmac::hls::kLmMaxGroups, 1);
    out.pc11_efficient_lm_head_qzeros = pc[11].alloc_bytes(
        sizeof(int32_t) * tmac::hls::kLmMaxVocabPacked * tmac::hls::kLmMaxGroups, 1);
    out.pc11_efficient_lm_head_g_idx =
        pc[11].alloc_bytes(sizeof(int32_t) * tmac::hls::kEagle4LmRankMax, 1);

    out.pc12_lm_head_weight = pc[12].alloc_bytes(
        sizeof(uint16_t) * static_cast<int64_t>(efficient_lm_vocab_size) * hidden_size, 1);

    out.pc13_draft_embed_tokens_weight = pc[13].alloc_bytes(
        sizeof(uint16_t) * static_cast<int64_t>(tmac::hls::kEagle4FullVocab) * hidden_size, 1);

    out.pc14_step_input_tokens = pc[14].alloc_bytes(sizeof(int64_t) * step_flat, 1);
    out.pc14_step_last_layer_scores = pc[14].alloc_bytes(sizeof(float) * step_flat, 1);
    out.pc14_step_topk_indexs_prev = pc[14].alloc_bytes(sizeof(int64_t) * step_flat, 1);
    out.pc14_step_topk_probas_sampling = pc[14].alloc_bytes(sizeof(float) * topk_flat, 1);
    out.pc14_step_topk_tokens_sampling = pc[14].alloc_bytes(sizeof(int64_t) * topk_flat, 1);

    out.pc15_step_input_hidden_states = pc[15].alloc_bytes(sizeof(float) * hidden_step_flat, 1);
    out.pc15_output_hidden_states = pc[15].alloc_bytes(sizeof(float) * hidden_out_flat, 1);

    out.pc16_cumu_tokens = pc[16].alloc_bytes(sizeof(int64_t) * node_flat, 1);
    out.pc17_cumu_scores = pc[17].alloc_bytes(sizeof(float) * node_flat, 1);
    out.pc18_cumu_deltas = pc[18].alloc_bytes(sizeof(int64_t) * node_flat, 1);
    out.pc19_prev_indexs = pc[19].alloc_bytes(sizeof(int64_t) * node_flat, 1);
    out.pc20_next_indexs = pc[20].alloc_bytes(sizeof(int64_t) * node_flat, 1);
    out.pc21_side_indexs = pc[21].alloc_bytes(sizeof(int64_t) * node_flat, 1);

    out.pc22_output_scores = pc[22].alloc_bytes(sizeof(float) * out_flat, 1);
    out.pc22_output_tokens = pc[22].alloc_bytes(sizeof(int64_t) * out_flat, 1);
    out.pc22_work_scores = pc[22].alloc_bytes(sizeof(float) * work_flat, 1);
    out.pc22_sort_scores = pc[22].alloc_bytes(sizeof(float) * sort_flat, 1);
    out.pc22_cache_topk_indices = pc[22].alloc_bytes(sizeof(int64_t) * out_flat, 1);

    out.pc23_node_to_hbm_slot = pc[23].alloc_bytes(sizeof(int64_t) * max_node_count, 1);
    out.pc23_accepted_draft_node_ids = pc[23].alloc_bytes(sizeof(int64_t) * max_verify_num, 1);
    out.pc23_io_tree_width = pc[23].alloc_bytes(sizeof(int), 1);
    out.pc23_io_verify_num = pc[23].alloc_bytes(sizeof(int), 1);
    out.pc23_io_cumu_count = pc[23].alloc_bytes(sizeof(int), 1);
    out.pc23_executed_depths = pc[23].alloc_bytes(sizeof(int), 1);
    out.pc23_stopped_early = pc[23].alloc_bytes(sizeof(bool), 1);

    out.pc24_initial_logits = pc[24].alloc_bytes(sizeof(float) * init_logits_flat, 1);
    out.pc25_initial_candidate_indices = pc[25].alloc_bytes(sizeof(int64_t) * init_logits_flat, 1);
    out.pc25_initial_topk_probas = pc[25].alloc_bytes(sizeof(float) * out_flat, 1);
    out.pc25_initial_topk_tokens = pc[25].alloc_bytes(sizeof(int64_t) * out_flat, 1);
    out.pc25_initial_hidden_states = pc[25].alloc_bytes(sizeof(float) * init_hidden_flat, 1);

    out.pc26_prefill_input_hidden_states_3h =
        pc[26].alloc_bytes(sizeof(float) * prefill_hidden_3h_flat, 1);
    out.pc26_prefill_input_embed_states = pc[26].alloc_bytes(sizeof(float) * prefill_embed_flat, 1);
    out.pc27_prefill_fc_weight = pc[27].alloc_bytes(sizeof(pack512) * prefill_fc_pack_count, 2);
    out.pc27_prefill_fc_scales = pc[27].alloc_bytes(sizeof(float) * prefill_fc_scale_count, 1);

    out.pc28_dbg_curr_layer_scores = pc[28].alloc_bytes(sizeof(float) * topk_flat, 1);
    out.pc28_dbg_sort_layer_scores = pc[28].alloc_bytes(sizeof(float) * topk_flat, 1);
    out.pc29_dbg_sort_layer_indices = pc[29].alloc_bytes(sizeof(int64_t) * topk_flat, 1);
    out.pc29_dbg_parent_indices_in_layer = pc[29].alloc_bytes(sizeof(int64_t) * out_flat, 1);
    out.pc29_dbg_remapped_topk_tokens = pc[29].alloc_bytes(sizeof(int64_t) * topk_flat, 1);

    out.pc30_policy_next_tree_width = pc[30].alloc_bytes(sizeof(int) * policy_depth, 1);
    out.pc30_policy_next_verify_num = pc[30].alloc_bytes(sizeof(int) * policy_depth, 1);
    out.pc30_policy_stop_signal = pc[30].alloc_bytes(sizeof(int) * policy_depth, 1);
    out.pc30_hot_token_id =
        pc[30].alloc_bytes(sizeof(int64_t) * static_cast<int64_t>(hot_token_vocab_size), 1);

    for (int i = 0; i < 32; ++i) {
        out.pc_words[i] = pc[i].words;
    }
    (void)efficient_lm_vocab_size;
    return out;
}

}  // namespace

extern "C" void eagle4_draft_packed_32pc(
    hbm_word256_t* pc00,
    hbm_word256_t* pc01,
    hbm_word256_t* pc02,
    hbm_word256_t* pc03,
    hbm_word256_t* pc04,
    hbm_word256_t* pc05,
    hbm_word256_t* pc06,
    hbm_word256_t* pc07,
    hbm_word256_t* pc08,
    hbm_word256_t* pc09,
    hbm_word256_t* pc10,
    hbm_word256_t* pc11,
    hbm_word256_t* pc12,
    hbm_word256_t* pc13,
    hbm_word256_t* pc14,
    hbm_word256_t* pc15,
    hbm_word256_t* pc16,
    hbm_word256_t* pc17,
    hbm_word256_t* pc18,
    hbm_word256_t* pc19,
    hbm_word256_t* pc20,
    hbm_word256_t* pc21,
    hbm_word256_t* pc22,
    hbm_word256_t* pc23,
    hbm_word256_t* pc24,
    hbm_word256_t* pc25,
    hbm_word256_t* pc26,
    hbm_word256_t* pc27,
    hbm_word256_t* pc28,
    hbm_word256_t* pc29,
    hbm_word256_t* pc30,
    hbm_word256_t* pc31,
    int tree_depth,
    int curr_depth_start,
    int policy_depth,
    bool use_policy_schedule,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    int prefix_len,
    bool enable_accepted_kv_compact,
    int accepted_draft_node_count,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
    int batch_size,
    int node_top_k,
    int hidden_size,
    int max_node_count,
    int max_verify_num,
    int max_tree_width,
    bool enable_initial_loop,
    bool enable_prefill_stage,
    int initial_logits_width) {
#pragma HLS INLINE off

#pragma HLS INTERFACE m_axi port = pc00 offset = slave bundle = gm00
#pragma HLS INTERFACE m_axi port = pc01 offset = slave bundle = gm01
#pragma HLS INTERFACE m_axi port = pc02 offset = slave bundle = gm02
#pragma HLS INTERFACE m_axi port = pc03 offset = slave bundle = gm03
#pragma HLS INTERFACE m_axi port = pc04 offset = slave bundle = gm04
#pragma HLS INTERFACE m_axi port = pc05 offset = slave bundle = gm05
#pragma HLS INTERFACE m_axi port = pc06 offset = slave bundle = gm06
#pragma HLS INTERFACE m_axi port = pc07 offset = slave bundle = gm07
#pragma HLS INTERFACE m_axi port = pc08 offset = slave bundle = gm08
#pragma HLS INTERFACE m_axi port = pc09 offset = slave bundle = gm09
#pragma HLS INTERFACE m_axi port = pc10 offset = slave bundle = gm10
#pragma HLS INTERFACE m_axi port = pc11 offset = slave bundle = gm11
#pragma HLS INTERFACE m_axi port = pc12 offset = slave bundle = gm12
#pragma HLS INTERFACE m_axi port = pc13 offset = slave bundle = gm13
#pragma HLS INTERFACE m_axi port = pc14 offset = slave bundle = gm14
#pragma HLS INTERFACE m_axi port = pc15 offset = slave bundle = gm15
#pragma HLS INTERFACE m_axi port = pc16 offset = slave bundle = gm16
#pragma HLS INTERFACE m_axi port = pc17 offset = slave bundle = gm17
#pragma HLS INTERFACE m_axi port = pc18 offset = slave bundle = gm18
#pragma HLS INTERFACE m_axi port = pc19 offset = slave bundle = gm19
#pragma HLS INTERFACE m_axi port = pc20 offset = slave bundle = gm20
#pragma HLS INTERFACE m_axi port = pc21 offset = slave bundle = gm21
#pragma HLS INTERFACE m_axi port = pc22 offset = slave bundle = gm22
#pragma HLS INTERFACE m_axi port = pc23 offset = slave bundle = gm23
#pragma HLS INTERFACE m_axi port = pc24 offset = slave bundle = gm24
#pragma HLS INTERFACE m_axi port = pc25 offset = slave bundle = gm25
#pragma HLS INTERFACE m_axi port = pc26 offset = slave bundle = gm26
#pragma HLS INTERFACE m_axi port = pc27 offset = slave bundle = gm27
#pragma HLS INTERFACE m_axi port = pc28 offset = slave bundle = gm28
#pragma HLS INTERFACE m_axi port = pc29 offset = slave bundle = gm29
#pragma HLS INTERFACE m_axi port = pc30 offset = slave bundle = gm30
#pragma HLS INTERFACE m_axi port = pc31 offset = slave bundle = gm31

#pragma HLS INTERFACE s_axilite port = tree_depth bundle = control
#pragma HLS INTERFACE s_axilite port = curr_depth_start bundle = control
#pragma HLS INTERFACE s_axilite port = policy_depth bundle = control
#pragma HLS INTERFACE s_axilite port = use_policy_schedule bundle = control
#pragma HLS INTERFACE s_axilite port = efficient_lm_rank bundle = control
#pragma HLS INTERFACE s_axilite port = efficient_lm_vocab_size bundle = control
#pragma HLS INTERFACE s_axilite port = prefix_len bundle = control
#pragma HLS INTERFACE s_axilite port = enable_accepted_kv_compact bundle = control
#pragma HLS INTERFACE s_axilite port = accepted_draft_node_count bundle = control
#pragma HLS INTERFACE s_axilite port = hot_token_vocab_size bundle = control
#pragma HLS INTERFACE s_axilite port = use_hot_token_id bundle = control
#pragma HLS INTERFACE s_axilite port = batch_size bundle = control
#pragma HLS INTERFACE s_axilite port = node_top_k bundle = control
#pragma HLS INTERFACE s_axilite port = hidden_size bundle = control
#pragma HLS INTERFACE s_axilite port = max_node_count bundle = control
#pragma HLS INTERFACE s_axilite port = max_verify_num bundle = control
#pragma HLS INTERFACE s_axilite port = max_tree_width bundle = control
#pragma HLS INTERFACE s_axilite port = enable_initial_loop bundle = control
#pragma HLS INTERFACE s_axilite port = enable_prefill_stage bundle = control
#pragma HLS INTERFACE s_axilite port = initial_logits_width bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    if (batch_size <= 0 || node_top_k <= 0 || hidden_size <= 0 || max_node_count <= 0 ||
        max_verify_num <= 0 || max_tree_width <= 0) {
        return;
    }

    const E4dPackedLayout32Pc layout = build_layout(
        tree_depth,
        curr_depth_start,
        prefix_len,
        policy_depth,
        batch_size,
        node_top_k,
        hidden_size,
        max_node_count,
        max_verify_num,
        max_tree_width,
        initial_logits_width,
        hot_token_vocab_size,
        efficient_lm_vocab_size);
    (void)layout.pc_words;
    (void)pc31;  // reserved in this revision.

    const int* policy_next_tree_width = use_policy_schedule
        ? pc_word_offset_ptr<const int>(pc30, layout.pc30_policy_next_tree_width)
        : nullptr;
    const int* policy_next_verify_num = use_policy_schedule
        ? pc_word_offset_ptr<const int>(pc30, layout.pc30_policy_next_verify_num)
        : nullptr;
    const int* policy_stop_signal = use_policy_schedule
        ? pc_word_offset_ptr<const int>(pc30, layout.pc30_policy_stop_signal)
        : nullptr;

    int64_t* step_input_tokens = pc_word_offset_ptr<int64_t>(
        pc14, layout.pc14_step_input_tokens);
    float* step_input_hidden_states = pc_word_offset_ptr<float>(
        pc15, layout.pc15_step_input_hidden_states);
    float* step_last_layer_scores = pc_word_offset_ptr<float>(
        pc14, layout.pc14_step_last_layer_scores);
    int64_t* step_topk_indexs_prev = pc_word_offset_ptr<int64_t>(
        pc14, layout.pc14_step_topk_indexs_prev);
    float* step_topk_probas_sampling = pc_word_offset_ptr<float>(
        pc14, layout.pc14_step_topk_probas_sampling);
    int64_t* step_topk_tokens_sampling = pc_word_offset_ptr<int64_t>(
        pc14, layout.pc14_step_topk_tokens_sampling);

    const pack512* w_q = pc_word_offset_ptr<const pack512>(pc00, layout.pc00_w_q);
    const float* s_q = pc_word_offset_ptr<const float>(pc00, layout.pc00_s_q);
    const pack512* w_k = pc_word_offset_ptr<const pack512>(pc01, layout.pc01_w_k);
    const float* s_k = pc_word_offset_ptr<const float>(pc01, layout.pc01_s_k);
    const pack512* w_v = pc_word_offset_ptr<const pack512>(pc02, layout.pc02_w_v);
    const float* s_v = pc_word_offset_ptr<const float>(pc02, layout.pc02_s_v);
    const pack512* w_o = pc_word_offset_ptr<const pack512>(pc03, layout.pc03_w_o);
    const float* s_o = pc_word_offset_ptr<const float>(pc03, layout.pc03_s_o);
    const pack512* w_gate = pc_word_offset_ptr<const pack512>(pc04, layout.pc04_w_gate);
    const float* gate_scales = pc_word_offset_ptr<const float>(pc04, layout.pc04_gate_scales);
    const pack512* w_up = pc_word_offset_ptr<const pack512>(pc05, layout.pc05_w_up);
    const float* up_scales = pc_word_offset_ptr<const float>(pc05, layout.pc05_up_scales);
    const pack512* w_down = pc_word_offset_ptr<const pack512>(pc06, layout.pc06_w_down);
    const float* down_scales = pc_word_offset_ptr<const float>(pc06, layout.pc06_down_scales);

    const float* hidden_norm_gamma = pc_word_offset_ptr<const float>(
        pc07, layout.pc07_hidden_norm_gamma);
    const float* embed_norm_gamma = pc_word_offset_ptr<const float>(
        pc07, layout.pc07_embed_norm_gamma);
    const float* post_attn_norm_gamma = pc_word_offset_ptr<const float>(
        pc07, layout.pc07_post_attn_norm_gamma);
    const float* final_norm_gamma = pc_word_offset_ptr<const float>(
        pc07, layout.pc07_final_norm_gamma);
    const RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>*
        rope_cfg_table =
            pc_word_offset_ptr<const RopeConfig<tmac::hls::NUM_HEADS,
                                                tmac::hls::NUM_KV_HEADS,
                                                tmac::hls::HEAD_DIM>>(pc07, layout.pc07_rope_cfg_table);

    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_k =
        pc_word_offset_ptr<tmac::hls::vec_t<tmac::hls::VEC_W>>(pc08, 0);
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_v =
        pc_word_offset_ptr<tmac::hls::vec_t<tmac::hls::VEC_W>>(pc09, 0);

    const uint16_t* efficient_lm_head_down_proj_weight = pc_word_offset_ptr<const uint16_t>(
        pc10, layout.pc10_efficient_lm_head_down_proj_weight);
    const int32_t* efficient_lm_head_qweight_row_major = pc_word_offset_ptr<const int32_t>(
        pc10, layout.pc10_efficient_lm_head_qweight_row_major);
    const uint16_t* efficient_lm_head_scales_row_major = pc_word_offset_ptr<const uint16_t>(
        pc11, layout.pc11_efficient_lm_head_scales_row_major);
    const int32_t* efficient_lm_head_qzeros = pc_word_offset_ptr<const int32_t>(
        pc11, layout.pc11_efficient_lm_head_qzeros);
    const int32_t* efficient_lm_head_g_idx = pc_word_offset_ptr<const int32_t>(
        pc11, layout.pc11_efficient_lm_head_g_idx);

    const uint16_t* lm_head_weight = pc_word_offset_ptr<const uint16_t>(
        pc12, layout.pc12_lm_head_weight);
    const uint16_t* draft_embed_tokens_weight = pc_word_offset_ptr<const uint16_t>(
        pc13, layout.pc13_draft_embed_tokens_weight);

    const int64_t* accepted_draft_node_ids =
        (enable_accepted_kv_compact && accepted_draft_node_count > 0)
            ? pc_word_offset_ptr<const int64_t>(pc23, layout.pc23_accepted_draft_node_ids)
            : nullptr;
    int64_t* node_to_hbm_slot = pc_word_offset_ptr<int64_t>(
        pc23, layout.pc23_node_to_hbm_slot);
    const int64_t* hot_token_id = use_hot_token_id
        ? pc_word_offset_ptr<const int64_t>(pc30, layout.pc30_hot_token_id)
        : nullptr;

    int* io_tree_width = pc_word_offset_ptr<int>(pc23, layout.pc23_io_tree_width);
    int* io_verify_num = pc_word_offset_ptr<int>(pc23, layout.pc23_io_verify_num);
    int* io_cumu_count = pc_word_offset_ptr<int>(pc23, layout.pc23_io_cumu_count);

    int64_t* cumu_tokens = pc_word_offset_ptr<int64_t>(pc16, layout.pc16_cumu_tokens);
    float* cumu_scores = pc_word_offset_ptr<float>(pc17, layout.pc17_cumu_scores);
    int64_t* cumu_deltas = pc_word_offset_ptr<int64_t>(pc18, layout.pc18_cumu_deltas);
    int64_t* prev_indexs = pc_word_offset_ptr<int64_t>(pc19, layout.pc19_prev_indexs);
    int64_t* next_indexs = pc_word_offset_ptr<int64_t>(pc20, layout.pc20_next_indexs);
    int64_t* side_indexs = pc_word_offset_ptr<int64_t>(pc21, layout.pc21_side_indexs);
    float* output_scores = pc_word_offset_ptr<float>(pc22, layout.pc22_output_scores);
    int64_t* output_tokens = pc_word_offset_ptr<int64_t>(pc22, layout.pc22_output_tokens);
    float* work_scores = pc_word_offset_ptr<float>(pc22, layout.pc22_work_scores);
    float* sort_scores = pc_word_offset_ptr<float>(pc22, layout.pc22_sort_scores);
    float* output_hidden_states = pc_word_offset_ptr<float>(pc15, layout.pc15_output_hidden_states);
    int64_t* cache_topk_indices = pc_word_offset_ptr<int64_t>(
        pc22, layout.pc22_cache_topk_indices);

    float* dbg_curr_layer_scores = nullptr;
    float* dbg_sort_layer_scores = nullptr;
    int64_t* dbg_sort_layer_indices = nullptr;
    int64_t* dbg_parent_indices_in_layer = nullptr;
    int64_t* dbg_remapped_topk_tokens = nullptr;
    int* executed_depths = pc_word_offset_ptr<int>(pc23, layout.pc23_executed_depths);
    bool* stopped_early = pc_word_offset_ptr<bool>(pc23, layout.pc23_stopped_early);

    const float* initial_logits = pc_word_offset_ptr<const float>(pc24, layout.pc24_initial_logits);
    const int64_t* initial_candidate_indices = pc_word_offset_ptr<const int64_t>(
        pc25, layout.pc25_initial_candidate_indices);
    const float* initial_topk_probas = nullptr;
    const int64_t* initial_topk_tokens = nullptr;
    const float* initial_hidden_states = pc_word_offset_ptr<const float>(
        pc25, layout.pc25_initial_hidden_states);

    const float* prefill_input_hidden_states_3h = pc_word_offset_ptr<const float>(
        pc26, layout.pc26_prefill_input_hidden_states_3h);
    const float* prefill_input_embed_states = pc_word_offset_ptr<const float>(
        pc26, layout.pc26_prefill_input_embed_states);
    const pack512* prefill_fc_weight = pc_word_offset_ptr<const pack512>(
        pc27, layout.pc27_prefill_fc_weight);
    const float* prefill_fc_scales = pc_word_offset_ptr<const float>(
        pc27, layout.pc27_prefill_fc_scales);

    tmac::hls::eagle4_draft_impl(
        tree_depth,
        curr_depth_start,
        policy_next_tree_width,
        policy_next_verify_num,
        policy_stop_signal,
        policy_depth,
        use_policy_schedule,
        step_input_tokens,
        step_input_hidden_states,
        step_last_layer_scores,
        step_topk_indexs_prev,
        step_topk_probas_sampling,
        step_topk_tokens_sampling,
        w_q,
        s_q,
        w_k,
        s_k,
        w_v,
        s_v,
        w_o,
        s_o,
        w_gate,
        gate_scales,
        w_up,
        up_scales,
        w_down,
        down_scales,
        hidden_norm_gamma,
        embed_norm_gamma,
        post_attn_norm_gamma,
        final_norm_gamma,
        rope_cfg_table,
        hbm_k,
        hbm_v,
        efficient_lm_head_down_proj_weight,
        efficient_lm_head_qweight_row_major,
        efficient_lm_head_scales_row_major,
        efficient_lm_head_qzeros,
        efficient_lm_head_g_idx,
        lm_head_weight,
        efficient_lm_rank,
        efficient_lm_vocab_size,
        draft_embed_tokens_weight,
        prefix_len,
        enable_accepted_kv_compact,
        accepted_draft_node_ids,
        accepted_draft_node_count,
        node_to_hbm_slot,
        hot_token_id,
        hot_token_vocab_size,
        use_hot_token_id,
        batch_size,
        node_top_k,
        hidden_size,
        io_tree_width,
        io_verify_num,
        io_cumu_count,
        max_node_count,
        max_verify_num,
        max_tree_width,
        cumu_tokens,
        cumu_scores,
        cumu_deltas,
        prev_indexs,
        next_indexs,
        side_indexs,
        output_scores,
        output_tokens,
        work_scores,
        sort_scores,
        output_hidden_states,
        cache_topk_indices,
        dbg_curr_layer_scores,
        dbg_sort_layer_scores,
        dbg_sort_layer_indices,
        dbg_parent_indices_in_layer,
        dbg_remapped_topk_tokens,
        executed_depths,
        stopped_early,
        enable_initial_loop,
        initial_logits,
        initial_candidate_indices,
        initial_logits_width,
        initial_topk_probas,
        initial_topk_tokens,
        initial_hidden_states,
        enable_prefill_stage,
        prefill_input_hidden_states_3h,
        prefill_input_embed_states,
        prefill_fc_weight,
        prefill_fc_scales);
}
