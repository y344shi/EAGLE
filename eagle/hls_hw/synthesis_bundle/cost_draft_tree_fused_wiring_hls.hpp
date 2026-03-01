#ifndef TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP
#define TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP

#include <cmath>
#include <cstdint>
#include <limits>

#include "cost_draft_tree_controller_hls.hpp"
#include "cost_draft_tree_score_hls.hpp"
#include "cost_draft_tree_update_hls.hpp"
#include "eagle_tier1_lm_top.hpp"

namespace tmac {
namespace hls {

constexpr int kCdtFusedMaxBatch = 128;
constexpr int kCdtFusedMaxNodeTopK = 16;

// Tripcount policy for HLS synthesis latency estimation.
// Configuration: batch=1, node_top_k=8, depth=16, tree_width=TREE_WIDTH(4).
constexpr int kTcBatch = 1;
constexpr int kTcTopK = 8;
constexpr int kTcDepth = 16;
constexpr int kTcTotalTopK = TREE_WIDTH * kTcTopK;                       // 32
constexpr int kTcMaxNodeCount = 128;
constexpr int kTcMaxVerifyNum = 64;
constexpr int kTcLogitsWidth = 1024;  // typical initial logits candidate count
constexpr int kTcParentAccumSize = kCdtControllerMaxDepth * TREE_WIDTH;  // 256
constexpr int kTcInitScratchSize = kCdtFusedMaxBatch * kCdtFusedMaxNodeTopK; // 2048

// Fused step wiring for one draft-tree layer in HLS:
// 1) score/sort + parent pick + hidden gather,
// 2) cumulative state update (cumu_tokens, prev/next/side_indexs, work/sort_scores).
// KV management uses contiguous HBM ancestor-chain; no tree-mask or controller needed.
void cost_draft_tree_fused_step_hls(
    // Score inputs
    const float* topk_probas_sampling,      // [batch, tree_width * node_top_k]
    const int64_t* topk_tokens_sampling,    // [batch, tree_width * node_top_k]
    const float* last_layer_scores,         // [batch, tree_width]
    const float* input_hidden_states,       // [batch, tree_width, hidden]
    const int64_t* hot_token_id,            // [vocab]
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,

    // Update-state inputs
    const int64_t* topk_indexs_prev,        // [batch, tree_width] global cumu indices

    // Shared dims
    int batch_size,
    int node_top_k,
    int tree_width,
    int hidden_size,
    int cumu_count,
    int verify_num,
    int curr_depth,

    // Capacity dims
    int max_node_count,
    int max_verify_num,

    // Persistent legacy draft state (in/out)
    int64_t* cumu_tokens,                   // [batch, max_node_count]
    float* cumu_scores,                     // [batch, max_node_count]
    int64_t* cumu_deltas,                   // [batch, max_node_count]
    int64_t* prev_indexs,                   // [batch, max_node_count]
    int64_t* next_indexs,                   // [batch, max_node_count]
    int64_t* side_indexs,                   // [batch, max_node_count]
    float* output_scores,                   // [batch, node_top_k]
    int64_t* output_tokens,                 // [batch, node_top_k]
    float* work_scores,                     // [batch, max_verify_num + node_top_k]
    float* sort_scores,                     // [batch, max_verify_num]

    // Fused outputs
    float* output_hidden_states,            // [batch, node_top_k, hidden]
    int64_t* cache_topk_indices,            // [batch, node_top_k]

    // Optional debug outputs (can be nullptr)
    float* dbg_curr_layer_scores,           // [batch, tree_width * node_top_k]
    float* dbg_sort_layer_scores,           // [batch, tree_width * node_top_k]
    int64_t* dbg_sort_layer_indices,        // [batch, tree_width * node_top_k]
    int64_t* dbg_parent_indices_in_layer,   // [batch, node_top_k]
    int64_t* dbg_remapped_topk_tokens       // [batch, tree_width * node_top_k]
);

// Fixed tree-width policy helper for orchestrator usage.
// The policy keeps tree_width/verify_num unchanged and never stops early.
struct CdtFixedWidthPolicyHls {
    void operator()(
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
        (void)node_top_k;
        (void)max_tree_width;
        (void)work_scores;
        (void)max_verify_num;
        if (next_tree_width != nullptr) {
            *next_tree_width = curr_tree_width;
        }
        if (next_verify_num != nullptr) {
            *next_verify_num = curr_verify_num;
        }
        if (stop_signal != nullptr) {
            *stop_signal = false;
        }
    }
};

// Select top-k from logits and output softmax probabilities for those winners.
// If candidate_indices is provided, winner indices are remapped to real vocab token IDs.
void cdt_softmax_topk_from_logits_hls(
    const float* logits,                 // [batch, logits_width]
    const int64_t* candidate_indices,    // [batch, logits_width] optional
    int batch_size,
    int logits_width,
    int node_top_k,
    float* topk_probas_out,              // [batch, node_top_k]
    int64_t* topk_tokens_out             // [batch, node_top_k]
);

void cdt_copy_frontier_for_next_depth_hls(
    const int64_t* frontier_src,  // [batch, max_tree_width]
    int batch_size,
    int max_tree_width,
    int64_t* frontier_dst         // [batch, max_tree_width]
);

// Wire per-layer outputs into the next layer's inputs.
// Selects first next_tree_width entries from node_top_k outputs and
// feeds selected hidden states and global indices back into the next SLM call.
void cdt_prepare_next_layer_inputs_hls(
    const float* output_scores,           // [batch, node_top_k]
    const int64_t* output_tokens,         // [batch, node_top_k]
    const float* output_hidden_states,    // [batch, node_top_k, hidden]
    const int64_t* cache_topk_indices,    // [batch, node_top_k]
    int batch_size,
    int node_top_k,
    int hidden_size,
    int next_tree_width,
    int max_tree_width,
    int64_t* next_input_tokens,           // packed [batch, next_tree_width]
    float* next_last_layer_scores,        // packed [batch, next_tree_width]
    float* next_input_hidden_states,      // packed [batch, next_tree_width, hidden]
    int64_t* next_topk_indexs_prev        // packed [batch, next_tree_width]
);

// Run EAGLE4 SLM forward + LM-head top-k for one draft depth.
// The SLM path owns top-k candidate generation; outputs are packed to fused-step layout.
void cdt_run_eagle4_slm_topk_hls(
    const float* input_hidden_states,          // packed [batch, tree_width, hidden]
    int batch_size,
    int tree_width,
    int hidden_size,
    int node_top_k,
    const pack512* w_q,     const float* s_q,
    const pack512* w_k,     const float* s_k,
    const pack512* w_v,     const float* s_v,
    const pack512* w_o,     const float* s_o,
    const pack512* w_gate,  const float* gate_scales,
    const pack512* w_up,    const float* up_scales,
    const pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>& rope_cfg,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    const uint16_t* efficient_lm_head_down_proj_weight,
    const int32_t* efficient_lm_head_qweight_row_major,
    const uint16_t* efficient_lm_head_scales_row_major,
    const int32_t* efficient_lm_head_qzeros,
    const int32_t* efficient_lm_head_g_idx,
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    int prefix_len,
    int current_depth,
    const int* parent_indices_per_layer,
    float* reasoning_hidden_states_out,        // packed [batch, tree_width, hidden]
    float* topk_probas_sampling_out,           // packed [batch, tree_width * node_top_k]
    int64_t* topk_tokens_sampling_out          // packed [batch, tree_width * node_top_k]
);

// Multi-layer orchestrator:
//   optional InitialLoop (PyTorch draft_InitialLoop parity):
//     0) from previous-verify logits -> softmax+top-k (or caller-provided initial top-k),
//     1) run one fused tree step with tree_width=1,
//     2) call width policy for depth-0 and seed the first recurrent frontier.
//   recurrent loop per depth:
//     1) run EAGLE4 SLM forward + LM-head top-k (`eagle_tier1_lm_top_eagle4`),
//     2) run one fused tree step (score + update),
//     3) wire fused outputs into next-layer inputs (hidden recurrence + index carry),
//     4) repeat until tree_depth or stop.
//
// KV management uses contiguous HBM ancestor-chain (parent_indices_accum);
// no controller node graph or tree mask is needed.
//
// Policy schedule contract:
//   If use_policy_schedule=true and schedule arrays are non-null, depth-wise policy comes from:
//     policy_next_tree_width[depth], policy_next_verify_num[depth], policy_stop_signal[depth].
//   Otherwise, fixed policy is used: keep curr_tree_width/curr_verify_num and never stop.
void cdt_apply_policy_schedule_hls(
    int depth,
    int batch_size,
    int curr_tree_width,
    int node_top_k,
    int max_tree_width,
    int curr_verify_num,
    const float* work_scores,
    int max_verify_num,
    const int* policy_next_tree_width,
    const int* policy_next_verify_num,
    const int* policy_stop_signal,
    int policy_depth,
    bool use_policy_schedule,
    int* next_tree_width,
    int* next_verify_num,
    bool* stop_signal);

void cost_draft_tree_multilayer_orchestrator_impl_hls(
    int tree_depth,
    int curr_depth_start,

    // Optional policy schedule (depth-indexed).
    const int* policy_next_tree_width,   // [policy_depth] optional
    const int* policy_next_verify_num,   // [policy_depth] optional
    const int* policy_stop_signal,       // [policy_depth] optional (0/1)
    int policy_depth,
    bool use_policy_schedule,

    // Step-working buffers (in/out across depths)
    int64_t* step_input_tokens,           // packed [batch, tree_width]
    float* step_input_hidden_states,      // packed [batch, tree_width, hidden]
    float* step_last_layer_scores,        // packed [batch, tree_width]
    int64_t* step_topk_indexs_prev,       // packed [batch, tree_width]
    float* step_topk_probas_sampling,     // packed [batch, tree_width * node_top_k]
    int64_t* step_topk_tokens_sampling,   // packed [batch, tree_width * node_top_k]

    // EAGLE4 SLM + LM-head weights/buffers
    const pack512* w_q,     const float* s_q,
    const pack512* w_k,     const float* s_k,
    const pack512* w_v,     const float* s_v,
    const pack512* w_o,     const float* s_o,
    const pack512* w_gate,  const float* gate_scales,
    const pack512* w_up,    const float* up_scales,
    const pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const RopeConfig<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM>* rope_cfg_table,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    const uint16_t* efficient_lm_head_down_proj_weight,
    const int32_t* efficient_lm_head_qweight_row_major,
    const uint16_t* efficient_lm_head_scales_row_major,
    const int32_t* efficient_lm_head_qzeros,
    const int32_t* efficient_lm_head_g_idx,
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,

    // Contiguous KV context
    int prefix_len,

    // Hot-token remap config
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,

    // Runtime dims (in/out)
    int batch_size,
    int node_top_k,
    int hidden_size,
    int* io_tree_width,
    int* io_verify_num,
    int* io_cumu_count,

    // Capacity dims
    int max_node_count,
    int max_verify_num,
    int max_tree_width,

    // Persistent legacy state
    int64_t* cumu_tokens,
    float* cumu_scores,
    int64_t* cumu_deltas,
    int64_t* prev_indexs,
    int64_t* next_indexs,
    int64_t* side_indexs,
    float* output_scores,
    int64_t* output_tokens,
    float* work_scores,
    float* sort_scores,

    // Per-step fused outputs / scratch
    float* output_hidden_states,          // [batch, node_top_k, hidden]
    int64_t* cache_topk_indices,          // [batch, node_top_k]

    // Optional debug outputs (final executed step contents)
    float* dbg_curr_layer_scores,
    float* dbg_sort_layer_scores,
    int64_t* dbg_sort_layer_indices,
    int64_t* dbg_parent_indices_in_layer,
    int64_t* dbg_remapped_topk_tokens,

    // Optional loop outputs
    int* executed_depths,
    bool* stopped_early,

    // Optional InitialLoop (disabled by default).
    // If enabled:
    //   - use caller-provided initial_topk_* if present,
    //   - otherwise compute initial_topk_* from initial_logits (+ optional candidate remap).
    // initial_hidden_states must be [batch, hidden] from previous verify output.
    bool enable_initial_loop,// = false,
    const float* initial_logits,// = nullptr,             // [batch, initial_logits_width]
    const int64_t* initial_candidate_indices,// = nullptr,// [batch, initial_logits_width] optional
    int initial_logits_width,// = 0,
    const float* initial_topk_probas,// = nullptr,        // [batch, node_top_k] optional
    const int64_t* initial_topk_tokens,// = nullptr,      // [batch, node_top_k] optional
    const float* initial_hidden_states// = nullptr       // [batch, hidden]
);

} // namespace hls
} // namespace tmac

void cost_draft_tree_multilayer_orchestrator_hls(
    int tree_depth,
    int curr_depth_start,
    const int* policy_next_tree_width,
    const int* policy_next_verify_num,
    const int* policy_stop_signal,
    int policy_depth,
    bool use_policy_schedule,
    int64_t* step_input_tokens,
    float* step_input_hidden_states,
    float* step_last_layer_scores,
    int64_t* step_topk_indexs_prev,
    float* step_topk_probas_sampling,
    int64_t* step_topk_tokens_sampling,
    const tmac::hls::pack512* w_q,     const float* s_q,
    const tmac::hls::pack512* w_k,     const float* s_k,
    const tmac::hls::pack512* w_v,     const float* s_v,
    const tmac::hls::pack512* w_o,     const float* s_o,
    const tmac::hls::pack512* w_gate,  const float* gate_scales,
    const tmac::hls::pack512* w_up,    const float* up_scales,
    const tmac::hls::pack512* w_down,  const float* down_scales,
    const float* hidden_norm_gamma,
    const float* embed_norm_gamma,
    const float* post_attn_norm_gamma,
    const float* final_norm_gamma,
    const tmac::hls::RopeConfig<tmac::hls::NUM_HEADS, tmac::hls::NUM_KV_HEADS, tmac::hls::HEAD_DIM>* rope_cfg_table,
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_k,
    tmac::hls::vec_t<tmac::hls::VEC_W>* hbm_v,
    const uint16_t* efficient_lm_head_down_proj_weight,
    const int32_t* efficient_lm_head_qweight_row_major,
    const uint16_t* efficient_lm_head_scales_row_major,
    const int32_t* efficient_lm_head_qzeros,
    const int32_t* efficient_lm_head_g_idx,
    const uint16_t* lm_head_weight,
    int efficient_lm_rank,
    int efficient_lm_vocab_size,
    int prefix_len,
    const int64_t* hot_token_id,
    int64_t hot_token_vocab_size,
    bool use_hot_token_id,
    int batch_size,
    int node_top_k,
    int hidden_size,
    int* io_tree_width,
    int* io_verify_num,
    int* io_cumu_count,
    int max_node_count,
    int max_verify_num,
    int max_tree_width,
    int64_t* cumu_tokens,
    float* cumu_scores,
    int64_t* cumu_deltas,
    int64_t* prev_indexs,
    int64_t* next_indexs,
    int64_t* side_indexs,
    float* output_scores,
    int64_t* output_tokens,
    float* work_scores,
    float* sort_scores,
    float* output_hidden_states,
    int64_t* cache_topk_indices,
    float* dbg_curr_layer_scores,
    float* dbg_sort_layer_scores,
    int64_t* dbg_sort_layer_indices,
    int64_t* dbg_parent_indices_in_layer,
    int64_t* dbg_remapped_topk_tokens,
    int* executed_depths,
    bool* stopped_early,
    bool enable_initial_loop,
    const float* initial_logits,
    const int64_t* initial_candidate_indices,
    int initial_logits_width,
    const float* initial_topk_probas,
    const int64_t* initial_topk_tokens,
    const float* initial_hidden_states);

#endif // TMAC_COST_DRAFT_TREE_FUSED_WIRING_HLS_HPP